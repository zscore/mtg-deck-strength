"""Deck synergy, round 5: learn pair synergy from cards seen together in a game, not just from the decklist.

    python scripts/deck_synergy_v5.py MSH

A deck-level FM only knows two cards shared a deck; most games, at most one of them shows up. Here the pair terms
are fit on the cards each game actually *saw* (opening hand or drawn), nonland only:

  seen FM    logit = context + deck features + aff + game-length buckets + x.w (deck) + h.u (seen)
                     + sum_{i<j} h_i h_j <v_i, v_j>,       h = 1 if card seen this game
             This uses in-game information, so it is not a predictor of the deck. It is only an estimate of pair
             values Q = V V^T. Controls: game length (both cards of a pair are seen more in long games), and the
             number of lands and of spells seen, one-hot (flood / screw), so the low-rank pair term can't
             just count spells drawn.

Q then becomes one deck-level column, the deck's mean pairwise Q over nonland copies, the same way the pick-model
affinity column is built. Every deck-level model below uses deck counts only and trains with learning-rate drops.
They are scored on test games, paired against `linear + deck features + aff`:

  fm4 nonland + deck features + aff          round 4's best game-level model
  linear + deck features + aff + seen Q      does in-game synergy carry to the deck level?
  fm4 nonland + ... + aff + seen Q

Replication: the seen FM and the deck FM are each refit on two halves of the training drafts, and their pair values
are correlated between halves (nonland pairs sharing >= 2,000 training decks).
Writes results/{SET}.PremierDraft/deck_synergy_v5/{results.json, seen_pairs.csv}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_synergy_v2 import mean_pair, nll, pick_affinity  # noqa: E402
from mtgdata.cardinfo import card_info, deck_features  # noqa: E402
from mtgdata.games import aligned, flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (calibrate, covariates, deck_metrics, experience, game_split, load_drafts,  # noqa: E402
                                most_played_deck, record_before, record_onehot)
from decks.synergy import DeckFM, _train  # noqa: E402

LR_DROPS = 2
REF = "linear + deck features + aff"
MIN_CO = 2000


class SeenFM(nn.Module):
    """Input [N, 2C] = deck counts | seen indicators (float16 to halve memory). Pairs only between seen nonland cards."""

    def __init__(self, n_cards, n_cov, k, vmask):
        super().__init__()
        self.C = n_cards
        self.w = nn.Parameter(torch.zeros(n_cards))
        self.u = nn.Parameter(torch.zeros(n_cards))
        self.v_ = nn.Parameter(0.01 * torch.randn(n_cards, k))
        self.register_buffer("vmask", torch.as_tensor(vmask, dtype=torch.float32)[:, None])
        self.cov = nn.Linear(n_cov, 1)

    @property
    def v(self):
        return self.v_ * self.vmask

    def forward(self, xh, c, task=None):
        xh = xh.float()
        x, h = xh[:, :self.C], xh[:, self.C:]
        hv = h @ self.v
        pairs = 0.5 * (hv.pow(2) - h @ self.v.pow(2)).sum(-1)  # h is 0/1, so h^2 = h
        return x @ self.w + h @ self.u + pairs + self.cov(c)[:, 0]

    def penalty(self, l2_w, l2_v):
        return l2_w * (self.w.pow(2).sum() + self.u.pow(2).sum()) + l2_v * self.v.pow(2).sum()


def train(make, X, c, y, sp, l2w, l2v):
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    pen = (lambda: m.penalty(l2w, l2v)) if isinstance(m, SeenFM) else (lambda: m.penalty(l2w, l2v, 0))
    return _train(m, opt, X, c, y, sp, None, pen, 40, 4096, 3, "cpu", lr_drops=LR_DROPS)


def pair_matrix(m):
    v = m.v.detach().numpy()
    Q = v @ v.T
    np.fill_diagonal(Q, 0)
    return Q.astype(np.float32)


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v5"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(set_code, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    H = ((aligned(g, "opening_hand_", names) + aligned(g, "drawn_", names)) > 0).astype(np.float16)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    turns = np.clip(g["num_turns"].cast(pl.Int32, strict=False).fill_null(10).to_numpy(), 3, 20)
    length = np.eye(18, dtype=np.float32)[turns - 3]
    draft_of_game = g["draft_id"].to_numpy()
    deck, has_deck = most_played_deck(d, g)
    deck = deck.astype(np.float32)
    del g
    info = card_info(names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    aff = pick_affinity(set_code, names)
    if aff is None:
        print(f"{set_code}: no pick model, skipping")
        return
    S = aff[0]
    tr = sp == 0
    F = deck_features(X, info)[0]
    fmu, fsd = F[tr].mean(0), F[tr].std(0) + 1e-6
    a_tr = mean_pair(X[tr], S, nonland)
    amu, asd = a_tr.mean(), a_tr.std() + 1e-6
    del F, a_tr

    def extra(Xd):
        return np.hstack([(deck_features(Xd, info)[0] - fmu) / fsd,
                          ((mean_pair(Xd, S, nonland) - amu) / asd)[:, None]]).astype(np.float32)

    c = np.hstack([ctx, extra(X)]).astype(np.float32)
    C = X.shape[1]
    te = np.flatnonzero(sp == 2)
    print(f"{set_code}: {len(y)} games (train {tr.sum()}, test {len(te)}), mean cards seen "
          f"{H[tr].astype(np.float32).sum(1).mean():.1f} [{time.time() - t0:.0f}s]", flush=True)

    # 1. seen FM: pair values from cards seen together
    XH = np.hstack([X.astype(np.float16), H])
    del H
    # flood / screw: how many lands and spells were seen (one-hot), so the pairs can't just count spells drawn
    Hf = XH[:, C:].astype(np.float32)
    n_land = np.clip((Hf @ (1 - nonland)).astype(int), 0, 12)
    n_spell = np.clip((Hf @ nonland).astype(int), 0, 15)
    del Hf
    seen_counts = np.hstack([np.eye(13, dtype=np.float32)[n_land], np.eye(16, dtype=np.float32)[n_spell]])
    cs = np.hstack([c, length, seen_counts])
    make_seen = lambda: SeenFM(C, cs.shape[1], 4, nonland)  # noqa: E731
    fits = [train(make_seen, XH, cs, y, sp, 1e-5, l2v) + (l2v,) for l2v in (1e-3, 1e-2)]
    seen, v_seen, _, l2v_seen = min(fits, key=lambda f: f[1])
    del fits
    Q = pair_matrix(seen)
    print(f"  seen FM val {v_seen:.5f} l2_v={l2v_seen} [{time.time() - t0:.0f}s]", flush=True)

    # 2. deck-level models with the seen-Q column
    q_tr = mean_pair(X[tr], Q, nonland)
    qmu, qsd = q_tr.mean(), q_tr.std() + 1e-6
    seen_col = lambda Xd: ((mean_pair(Xd, Q, nonland) - qmu) / qsd)[:, None].astype(np.float32)  # noqa: E731
    cq = np.hstack([c, seen_col(X)])
    specs = {
        REF: (0, c), "fm4 nonland + deck features + aff": (4, c),
        "linear + deck features + aff + seen Q": (0, cq), "fm4 nonland + deck features + aff + seen Q": (4, cq),
    }
    preds, rows, models = {}, [], {}
    for name, (k, cc) in specs.items():
        t1 = time.time()
        make = lambda k=k, cc=cc: DeckFM(C, cc.shape[1], k, vmask=nonland if k else None)  # noqa: E731
        grid = [(1e-5, 1e-3), (1e-5, 1e-2)] if k else [(1e-5, 0), (1e-4, 0)]
        fits = [train(make, X, cc, y, sp, *kw) + (kw,) for kw in grid]
        m, v, predict, kw = min(fits, key=lambda f: f[1])
        preds[name], models[name] = predict(te), (m, cc is cq)
        row = {"model": name, "val logloss": v, "setting": kw}
        if cc is cq:
            row["seen Q weight (log-odds per sd)"] = float(m.cov.weight.detach().numpy()[0, -1])
            row["aff weight (log-odds per sd)"] = float(m.cov.weight.detach().numpy()[0, -2])
        rows.append(row)
        print(f"  {name:45s} val {v:.5f} {kw} [{time.time() - t1:.0f}s]", flush=True)

    base = nll(y[te], preds[REF])
    trd, ted = d.mask("train"), d.mask("test") & has_deck
    Ed, Qd = extra(deck), seen_col(deck)
    for r in rows:
        p = preds[r["model"]]
        dl = nll(y[te], p) - base
        m, uses_q = models[r["model"]]
        with torch.no_grad():
            s = m.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
        s = s + np.hstack([Ed, Qd] if uses_q else [Ed]) @ m.cov.weight.detach().numpy()[0, ctx.shape[1]:]
        pd_ = calibrate(s[trd], d.rank[trd], d.wins[trd], d.losses[trd], s, d.rank)
        r.update({"test logloss": log_loss(y[te], p), "test auc": roc_auc_score(y[te], p),
                  "vs ref": float(dl.mean()), "vs ref ±95%": float(1.96 * dl.std() / np.sqrt(len(dl))),
                  **{f"draft {k}": float(v) for k, v in deck_metrics(pd_[ted], d.wins[ted], d.losses[ted]).items()}})
        print(f"  {r['model']:45s} vs ref {r['vs ref'] * 1000:+.3f} ± {r['vs ref ±95%'] * 1000:.3f} (x1000)  "
              f"draft spearman {r['draft spearman']:.4f}", flush=True)

    # 3. replication: seen pairs vs deck pairs, each refit on two halves of the training drafts
    Pd = (X[tr] * nonland > 0).astype(np.float32)
    co = Pd.T @ Pd
    del Pd
    iu = np.triu_indices(C, 1)
    ok = co[iu] >= MIN_CO
    ids = np.unique(draft_of_game[tr])
    half = set(ids[np.random.default_rng(1).permutation(len(ids)) < len(ids) // 2].tolist())
    in_a = np.array([x in half for x in draft_of_game])
    deck_kw = next(r["setting"] for r in rows if r["model"] == "fm4 nonland + deck features + aff")
    rep = {"pairs": int(ok.sum())}
    for tag, make, inp, cc, kw in (
            ("seen", make_seen, XH, cs, (1e-5, l2v_seen)),
            ("deck", lambda: DeckFM(C, c.shape[1], 4, vmask=nonland), X, c, deck_kw)):
        Qs = []
        for side in (True, False):
            s_ = np.where(tr & (in_a == side), -1, sp)
            Qs.append(pair_matrix(train(make, inp, cc, y, s_, *kw)[0])[iu][ok])
        rep[f"{tag} pair r"] = float(np.corrcoef(*Qs)[0, 1])
        rep[f"{tag} vs affinity r"] = float(np.mean([np.corrcoef(q, S[iu][ok])[0, 1] for q in Qs]))
    rep["seen vs deck pairs r (full fits)"] = float(np.corrcoef(Q[iu][ok], pair_matrix(
        models["fm4 nonland + deck features + aff"][0])[iu][ok])[0, 1])
    print(f"  replication: {rep}", flush=True)

    i, j = iu[0][ok], iu[1][ok]
    pl.DataFrame({"card a": [names[t] for t in i], "card b": [names[t] for t in j], "seen Q": Q[i, j],
                  "pick affinity": S[i, j], "train decks": co[i, j]}) \
        .sort("seen Q", descending=True).write_csv(out / "seen_pairs.csv")
    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "seen FM val logloss": v_seen,
           "seen FM l2_v": l2v_seen, "models": rows, "replication": rep, "seconds": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
