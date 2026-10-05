"""Deck synergy, round 3: more out of the pick model's affinity than one mean-affinity column.

    python scripts/deck_synergy_v3.py MSH

Same games, split, context and paired test metric as deck_synergy_v2.py, but every model trains with learning-rate
drops on plateau (v2 stopped at the first plateau, under-training everything). For each nonland card i in a deck,
its *fit* is its mean pick-model affinity S_ij to the deck's other nonland cards (j over copies).

  linear + deck features (+ aff)      v2 references (aff = the deck's mean pairwise affinity, one column)
  fm4 nonland + deck features (+ aff) v2's nonland FM, with and without the affinity column
  + affinity profile                  the fit distribution, not just its mean: worst card's fit, sd of fit,
                                      number of cards below the training 10th percentile of fit (off-plan cards)
  + card fit                          per-card synergy sensitivity: sum_i x_i g_i (fit_i - m_i) / sd, with m_i
                                      card i's mean fit in training decks. g_i > 0: a card that is only good
                                      in the right deck (build-around / archetype glue); g_i ~ 0: good anywhere.
                                      Warm-started from the nested model (g = 0), so it can only add
  continued (control)                 the same warm-started continuation with g held at 0: separates the card-fit
                                      gain from simply training longer
"vs v2 best" in the output is against fm4 nonland + deck features + aff, retrained here.

Also writes every model's calibrated draft-level p for all drafts (draft_scores.npz) so
scripts/deck_synergy_v3_blend.py can blend it with the draft-level LightGBM deck model.
Writes results/{SET}.PremierDraft/deck_synergy_v3/{results.json, card_fit.csv, draft_scores.npz}.
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
from deck_synergy_v2 import nll, pick_affinity  # noqa: E402
from openlane.cardinfo import card_info, deck_features  # noqa: E402
from openlane.data import ROOT  # noqa: E402
from openlane.deckstrength import (aligned, calibrate, covariates, deck_metrics, experience, flag,  # noqa: E402
                                   game_split, load_drafts, most_played_deck, record_before, record_onehot)
from openlane.games import game_path  # noqa: E402
from openlane.synergy import DeckFM, _train, fit  # noqa: E402

LR_DROPS = 2  # v2 stopped at the first plateau at lr 3e-3, which left every model under-trained


def card_fit(X, S, nonland, chunk=100_000):
    """[N, C] fit_i = sum_j x_j S_ij / (n - 1) over nonland copies (S has a zero diagonal), and the nonland counts."""
    Xn = X * nonland
    out = np.empty_like(Xn)
    for a in range(0, len(X), chunk):
        x = Xn[a:a + chunk]
        out[a:a + chunk] = (x @ S) / np.maximum(x.sum(1, keepdims=True) - 1, 1)
    return out, Xn


class Profile:
    """Deck-level summaries of card fit, standardised on training games."""
    names = ["aff mean", "aff worst card", "aff sd", "aff off-plan cards"]

    def __init__(self, X, S, nonland, tr):
        self.S, self.nonland = S, nonland
        fit_, Xn = card_fit(X[tr], S, nonland)
        self.lo = float(np.quantile(fit_[Xn > 0], 0.10))
        raw = self.raw(X)
        self.mu, self.sd = raw[tr].mean(0), raw[tr].std(0) + 1e-6

    def raw(self, X):
        fit_, Xn = card_fit(X, self.S, self.nonland)
        has = Xn > 0
        n = np.maximum(Xn.sum(1), 1)
        mean = (Xn * fit_).sum(1) / n
        worst = np.where(has, fit_, np.inf).min(1)
        worst = np.where(np.isfinite(worst), worst, 0)
        sd = np.sqrt(np.maximum((Xn * (fit_ - mean[:, None]) ** 2).sum(1) / n, 0))
        off = (Xn * (fit_ < self.lo)).sum(1)
        return np.stack([mean, worst, sd, off], 1).astype(np.float32)

    def __call__(self, X, cols=slice(None)):
        return ((self.raw(X) - self.mu) / self.sd)[:, cols].astype(np.float32)


class FitFM(DeckFM):
    """DeckFM + per-card synergy sensitivity g_i on the card's (centred, scaled) fit to the rest of the deck."""

    def __init__(self, n_cards, n_cov, k, vmask, S, m, scale):
        super().__init__(n_cards, n_cov, k, vmask=vmask if k else None)
        self.register_buffer("S", torch.as_tensor(S))
        self.register_buffer("m", torch.as_tensor(m, dtype=torch.float32))
        self.register_buffer("nl", torch.as_tensor(vmask, dtype=torch.float32))
        self.scale = float(scale)
        self.g = nn.Parameter(torch.zeros(n_cards))

    def fit_term(self, x):
        xn = x * self.nl
        f = (xn @ self.S) / (xn.sum(1, keepdim=True) - 1).clamp(min=1)
        return xn * (f - self.m) / self.scale

    def deck_score(self, x, task=None):
        return super().deck_score(x, task) + self.fit_term(x) @ self.g

    def penalty(self, l2_w, l2_v, l2_dw, l2_g=0.0):
        return super().penalty(l2_w, l2_v, l2_dw) + l2_g * self.g.pow(2).sum()


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v3"
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
    deck, has_deck = most_played_deck(d, g)
    deck = deck.astype(np.float32)
    del g
    info = card_info(names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    F, _ = deck_features(X, info)
    mu, sd = F[sp == 0].mean(0), F[sp == 0].std(0) + 1e-6
    feat = lambda Xd: ((deck_features(Xd, info)[0] - mu) / sd).astype(np.float32)  # noqa: E731
    aff = pick_affinity(set_code, names)
    if aff is None:
        print(f"{set_code}: no pick model, skipping")
        return
    S = aff[0]
    tr = sp == 0
    prof = Profile(X, S, nonland, tr)
    fit_tr, Xn_tr = card_fit(X[tr], S, nonland)
    m = (Xn_tr * fit_tr).sum(0) / np.maximum(Xn_tr.sum(0), 1)  # card i's mean fit where it is played
    scale = float(np.sqrt(((Xn_tr > 0) * (fit_tr - m) ** 2).sum() / (Xn_tr > 0).sum()))
    del fit_tr, Xn_tr
    te = np.flatnonzero(sp == 2)
    print(f"{set_code}: {len(y)} games (train {tr.sum()}, test {len(te)}) [{time.time() - t0:.0f}s]", flush=True)

    # each model: (k, extra deck-level columns as a function of deck counts, use card fit, grid)
    extras = {
        "none": lambda Xd: np.zeros((len(Xd), 0), np.float32),
        "F": feat,
        "F+aff": lambda Xd: np.hstack([feat(Xd), prof(Xd, slice(0, 1))]),
        "F+profile": lambda Xd: np.hstack([feat(Xd), prof(Xd)]),
    }
    # name: (k, extra columns, card fit, warm start from, play/draw card offsets)
    specs = {
        "linear": (0, "none", False, None),
        "linear + deck features": (0, "F", False, None),
        "linear + deck features + aff": (0, "F+aff", False, None),
        "fm4 nonland + deck features": (4, "F", False, None),
        "fm4 nonland + deck features + aff": (4, "F+aff", False, None),
        "linear + deck features + affinity profile": (0, "F+profile", False, None),
        "linear + deck features + aff + card fit": (0, "F+aff", True, "linear + deck features + aff"),
        "fm4 nonland + deck features + aff + card fit": (4, "F+aff", True, "fm4 nonland + deck features + aff"),
        "fm4 nonland + deck features + aff, continued (control)": (4, "F+aff", "control",
                                                                   "fm4 nonland + deck features + aff"),
    }
    cache = {}

    def ctx_for(key):
        if key not in cache:
            cache[key] = np.hstack([ctx, extras[key](X)]).astype(np.float32)
        return cache[key]

    def fit_one(k, key, use_fit, warm):
        c = ctx_for(key)
        if not use_fit:
            grid = [(1e-5, 1e-3), (1e-5, 1e-2)] if k else [(1e-5, 0), (1e-4, 0)]
            fits = [fit(X, c, y, sp, k, l2w, l2v, log=lambda s: None, vmask=nonland if k else None,
                        lr_drops=LR_DROPS) + ((l2w, l2v),)
                    for l2w, l2v in grid]
            return min(fits, key=lambda f: f[1])
        start = models[warm][0]
        l2w, l2v = rows_by[warm]["setting"][:2]
        fits = []
        for l2g in ((0,) if use_fit == "control" else (1e-3, 1e-2, 1e-1)):
            torch.manual_seed(0)
            mdl = FitFM(X.shape[1], c.shape[1], k, nonland, S, m, scale)
            mdl.load_state_dict(start.state_dict(), strict=False)  # warm start: g = 0 is the nested model
            mdl.g.requires_grad_(use_fit != "control")  # control: same continued training, g held at 0
            opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
            mdl, v, pr = _train(mdl, opt, X, c, y, sp, None,
                                lambda mdl=mdl, l2g=l2g, l2w=l2w, l2v=l2v: mdl.penalty(l2w, l2v, 0, l2g),
                                40, 4096, 3, "cpu", lr_drops=LR_DROPS)
            fits.append((mdl, v, pr, (l2w, l2v, l2g)))
        return min(fits, key=lambda f: f[1])

    preds, rows, models, rows_by = {}, [], {}, {}
    for name, (k, key, use_fit, warm) in specs.items():
        t1 = time.time()
        mdl, v, predict, setting = fit_one(k, key, use_fit, warm)
        preds[name], models[name] = predict(te), (mdl, key)
        row = {"model": name, "val logloss": v, "setting": setting, "fit seconds": time.time() - t1}
        if key in ("F+aff", "F+profile"):
            wc = mdl.cov.weight.detach().numpy()[0, ctx.shape[1] + F.shape[1]:]
            row["affinity weights (log-odds per sd)"] = dict(zip(Profile.names, np.round(wc, 4).tolist()))
        rows.append(row)
        rows_by[name] = row
        print(f"  {name:58s} val {v:.5f} {setting} [{time.time() - t1:.0f}s]", flush=True)

    base = nll(y[te], preds["linear"])
    ref = nll(y[te], preds["fm4 nonland + deck features + aff"])
    for r in rows:
        p = preds[r["model"]]
        dl, dr = nll(y[te], p) - base, nll(y[te], p) - ref
        r.update({"test logloss": log_loss(y[te], p), "test auc": roc_auc_score(y[te], p),
                  "vs linear": float(dl.mean()), "vs linear ±95%": float(1.96 * dl.std() / np.sqrt(len(dl))),
                  "vs v2 best": float(dr.mean()), "vs v2 best ±95%": float(1.96 * dr.std() / np.sqrt(len(dr)))})
        print(f"  {r['model']:58s} vs linear {r['vs linear'] * 1000:+.2f} ± {r['vs linear ±95%'] * 1000:.2f}  "
              f"vs v2 best {r['vs v2 best'] * 1000:+.3f} ± {r['vs v2 best ±95%'] * 1000:.3f} (x1000)", flush=True)

    # draft level: score each draft's most-played deck, calibrate to records on training drafts
    trd, ted = d.mask("train"), d.mask("test") & has_deck
    scores = {}
    for r in rows:
        mdl, key = models[r["model"]]
        with torch.no_grad():
            s = mdl.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
        wc = mdl.cov.weight.detach().numpy()[0, ctx.shape[1]:]
        if len(wc):
            s = s + extras[key](deck) @ wc
        p = calibrate(s[trd], d.rank[trd], d.wins[trd], d.losses[trd], s, d.rank)
        scores[r["model"]] = p
        r.update({f"draft {k}": float(v) for k, v in deck_metrics(p[ted], d.wins[ted], d.losses[ted]).items()})
        print(f"  {r['model']:58s} draft spearman {r['draft spearman']:.4f} nll {r['draft nll']:.4f}", flush=True)
    np.savez_compressed(out / "draft_scores.npz", has_deck=has_deck, **{k: v for k, v in scores.items()})

    # which cards' value depends most on fitting the deck (largest g), from the full model
    mdl = models["linear + deck features + aff + card fit"][0]
    played = (X[tr] > 0).sum(0)
    pl.DataFrame({"card": names, "power w": mdl.w.detach().numpy(), "fit sensitivity g": mdl.g.detach().numpy(),
                  "mean fit": m, "train games": played, "nonland": nonland > 0}) \
        .filter(pl.col("nonland") & (pl.col("train games") >= 2000)).sort("fit sensitivity g", descending=True) \
        .write_csv(out / "card_fit.csv")

    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "fit scale": scale,
           "off-plan threshold": prof.lo, "models": rows, "seconds": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
