"""Deck synergy, round 4 (game level): round 2's synergy variants again, now properly trained.

    python scripts/deck_synergy_v4.py MSH

Round 3 found round 2's models under-trained (they stopped at the first plateau); every fit here uses learning-rate
drops. All models carry the deck features and the affinity column (the deck's mean pick-model affinity), and are
scored on the test games paired against `linear + deck features + aff`:

  fm{4,8,16} nonland        pair terms between nonland cards, rank 4 / 8 / 16
  fm4 all cards             pair terms for every card, lands included
  affinity pairs (k=8)      pair vectors are a learned linear map of the pick model's card vectors
  affinity pairs + fm4      both: pick-model pairs plus free nonland pairs

For the draft level (scripts/deck_synergy_v4_draft.py) it also writes, per draft, the most-played deck's game-level
log-odds from the best FM, *cross-fitted*: a training draft is scored by a model fit on the other half of the
training drafts, so a draft model that boosts from this score never sees an in-sample score. Val/test drafts are
scored by the model fit on all training drafts. Alongside: the deck features and affinity column per draft.
Writes results/{SET}.PremierDraft/deck_synergy_v4/{results.json, draft_inputs.npz}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_synergy_v2 import AffinityFM, nll, pick_affinity  # noqa: E402
from deck_synergy_v3 import Profile  # noqa: E402
from mtgdata.cardinfo import card_info, deck_features  # noqa: E402
from mtgdata.games import aligned, flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (calibrate, covariates, deck_metrics, experience, game_split, load_drafts,  # noqa: E402
                                most_played_deck, record_before, record_onehot)
from decks.synergy import DeckFM, _train  # noqa: E402

LR_DROPS = 2
REF = "linear + deck features + aff"
CROSSFIT = "fm4 nonland"


class BothFM(AffinityFM):
    """Pick-model pairs (Z P) plus free nonland pair vectors V: v_i = [Z_i P, V_i]."""

    def __init__(self, n_cards, n_cov, Z, k_aff, k_free, vmask):
        super().__init__(n_cards, n_cov, Z, k_aff, vmask)
        self.V = torch.nn.Parameter(0.01 * torch.randn(n_cards, k_free))

    @property
    def v(self):
        return torch.cat([self.Z @ self.P, self.V], 1) * self.vmask

    def penalty(self, l2_w, l2_v, l2_dw):
        return super().penalty(l2_w, l2_v, l2_dw) + l2_v * 10 * (self.V * self.vmask).pow(2).sum()


def train(make, X, c, y, sp, l2w, l2v):
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    return _train(m, opt, X, c, y, sp, None, lambda: m.penalty(l2w, l2v, 0), 40, 4096, 3, "cpu", lr_drops=LR_DROPS)


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(set_code, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    draft_of_game = g["draft_id"].to_numpy()
    deck, has_deck = most_played_deck(d, g)
    deck = deck.astype(np.float32)
    del g
    info = card_info(names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    F, _ = deck_features(X, info)
    mu, sd = F[sp == 0].mean(0), F[sp == 0].std(0) + 1e-6
    aff = pick_affinity(set_code, names)
    if aff is None:
        print(f"{set_code}: no pick model, skipping")
        return
    S, Z = aff
    prof = Profile(X, S, nonland, sp == 0)
    extra = lambda Xd: np.hstack([(deck_features(Xd, info)[0] - mu) / sd, prof(Xd, slice(0, 1))]).astype(np.float32)  # noqa: E731,E501
    c = np.hstack([ctx, extra(X)]).astype(np.float32)
    C, nc = X.shape[1], c.shape[1]
    te = np.flatnonzero(sp == 2)
    print(f"{set_code}: {len(y)} games (train {np.sum(sp == 0)}, test {len(te)}) [{time.time() - t0:.0f}s]", flush=True)

    specs = {  # name: (constructor, grid of (l2_w, l2_v))
        REF: (lambda: DeckFM(C, nc, 0), [(1e-5, 0), (1e-4, 0)]),
        "fm4 nonland": (lambda: DeckFM(C, nc, 4, vmask=nonland), [(1e-5, 1e-3), (1e-5, 1e-2)]),
        "fm8 nonland": (lambda: DeckFM(C, nc, 8, vmask=nonland), [(1e-5, 1e-3), (1e-5, 1e-2)]),
        "fm16 nonland": (lambda: DeckFM(C, nc, 16, vmask=nonland), [(1e-5, 1e-3), (1e-5, 1e-2)]),
        "fm4 all cards": (lambda: DeckFM(C, nc, 4), [(1e-5, 1e-3), (1e-5, 1e-2)]),
        "affinity pairs (k=8)": (lambda: AffinityFM(C, nc, Z, 8, nonland), [(1e-5, 1e-4), (1e-5, 1e-3)]),
        "affinity pairs (k=8) + fm4 nonland": (lambda: BothFM(C, nc, Z, 8, 4, nonland), [(1e-5, 1e-4), (1e-5, 1e-3)]),
    }
    preds, rows, models = {}, [], {}
    for name, (make, grid) in specs.items():
        t1 = time.time()
        fits = [train(make, X, c, y, sp, *kw) + (kw,) for kw in grid]
        m, v, predict, kw = min(fits, key=lambda f: f[1])
        preds[name], models[name] = predict(te), m
        rows.append({"model": name, "val logloss": v, "setting": kw, "fit seconds": time.time() - t1})
        print(f"  {name:40s} val {v:.5f} {kw} [{time.time() - t1:.0f}s]", flush=True)

    base = nll(y[te], preds[REF])
    trd, ted = d.mask("train"), d.mask("test") & has_deck
    Ed = extra(deck)

    def draft_logit(m):
        with torch.no_grad():
            s = m.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
        return s + Ed @ m.cov.weight.detach().numpy()[0, ctx.shape[1]:]

    for r in rows:
        p = preds[r["model"]]
        dl = nll(y[te], p) - base
        s = draft_logit(models[r["model"]])
        pd_ = calibrate(s[trd], d.rank[trd], d.wins[trd], d.losses[trd], s, d.rank)
        r.update({"test logloss": log_loss(y[te], p), "test auc": roc_auc_score(y[te], p),
                  "vs ref": float(dl.mean()), "vs ref ±95%": float(1.96 * dl.std() / np.sqrt(len(dl))),
                  **{f"draft {k}": float(v) for k, v in deck_metrics(pd_[ted], d.wins[ted], d.losses[ted]).items()}})
        print(f"  {r['model']:40s} vs ref {r['vs ref'] * 1000:+.3f} ± {r['vs ref ±95%'] * 1000:.3f} (x1000)  "
              f"draft spearman {r['draft spearman']:.4f}", flush=True)

    # cross-fitted draft-level score from the CROSSFIT model's setting
    kw = next(r["setting"] for r in rows if r["model"] == CROSSFIT)
    full = draft_logit(models[CROSSFIT])
    ref_full = draft_logit(models[REF])
    tr_ids = np.unique(draft_of_game[sp == 0])
    fold_a = set(tr_ids[np.random.default_rng(2).permutation(len(tr_ids)) < len(tr_ids) // 2].tolist())
    in_a_game = np.array([x in fold_a for x in draft_of_game])
    in_a_draft = np.array([x in fold_a for x in d.draft_id])
    cross, cross_ref = full.copy(), ref_full.copy()
    for side in (True, False):  # fit on games of the other fold, score this fold's training drafts
        s_ = np.where((sp == 0) & (in_a_game == side), -1, sp)
        for name, dst in ((CROSSFIT, cross), (REF, cross_ref)):
            mk, _ = specs[name]
            kw_ = next(r["setting"] for r in rows if r["model"] == name)
            m, _, _ = train(mk, X, c, y, s_, *kw_)
            sel = trd & (in_a_draft == side)
            dst[sel] = draft_logit(m)[sel]
    print(f"  cross-fit train-draft score r with in-sample: {np.corrcoef(cross[trd], full[trd])[0, 1]:.3f}", flush=True)
    np.savez_compressed(out / "draft_inputs.npz", has_deck=has_deck, game_logit=cross, game_logit_ref=cross_ref,
                        deck_features=Ed[:, :-1], aff=Ed[:, -1])
    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "models": rows,
           "crossfit model": CROSSFIT, "crossfit setting": kw, "seconds": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
