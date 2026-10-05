"""Is draft affinity more than co-occurrence or colour, and is its draft-level gain just drafter skill?

    python scripts/affinity_checks.py MSH

deck_synergy_v2.py found that one column, the deck's mean pairwise affinity from the pick model, improves
held-out games on 11/12 sets. Two cheaper explanations are tested here, each as the same kind of single
column (mean over pairs of distinct nonland cards in the deck, standardised on training games):

  pick affinity   S_ij = <a_i, e_j> from the picknet `pool` model (how much j in the pool raises the odds of taking i)
  deck PMI        log P(i, j in a deck) / (P(i) P(j)) over training games' decks: plain co-occurrence
  colour overlap  cosine of the two cards' colour pips: "on-colour", nothing else

All on top of `linear + deck features`. Game level: test log loss vs that model (x1000, ±95%). Draft level:
Spearman of each draft's most-played deck score vs W/(W+L), calibrated on training drafts with rank only (as
before) and with rank + the drafter's skill bucket, to see whether a column's gain survives knowing the skill.
Writes results/{SET}.PremierDraft/deck_synergy_v2/affinity_checks.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_synergy_v2 import mean_pair, nll, pick_affinity  # noqa: E402
from openlane.cardinfo import card_info, deck_features  # noqa: E402
from openlane.data import COLORS, ROOT  # noqa: E402
from openlane.deckstrength import (N_RANKS, aligned, binomial_logreg, covariates, experience, flag,  # noqa: E402
                                   game_split, load_drafts, most_played_deck, record_before, record_onehot)
from openlane.games import game_path  # noqa: E402
from openlane.synergy import fit  # noqa: E402


def pmi(X, nonland, sp):
    P = ((X[sp == 0] * nonland) > 0).astype(np.float32)
    n = len(P)
    co = P.T @ P + 1.0
    p = (P.sum(0) + 1.0) / n
    S = np.log((co / n) / np.outer(p, p))
    np.fill_diagonal(S, 0)
    return S.astype(np.float32)


def colour_overlap(info):
    C = info.select([f"pip_{k}" for k in COLORS]).to_numpy().astype(np.float32)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-6)
    S = C @ C.T
    np.fill_diagonal(S, 0)
    return S


def calibrated_spearman(s, d, has_deck, skill):
    """Binomial logistic on train drafts with [score, rank one-hot (+ skill)], Spearman on test drafts."""
    tr, te = d.mask("train"), d.mask("test") & has_deck
    s = (s - s[tr].mean()) / (s[tr].std() + 1e-9)
    cols = [s[:, None], np.eye(N_RANKS)[d.rank]]
    if skill:
        sk = d.skill.astype(np.float64)
        known = np.isfinite(sk)
        cols += [np.where(known, sk - np.nanmean(sk[tr]), 0)[:, None] * 10, known[:, None]]
    Xc = np.hstack(cols)
    m = binomial_logreg(Xc[tr], d.wins[tr], d.losses[tr], C=1e4)
    p = m.predict_proba(Xc[te])[:, 1]
    w, l = d.wins[te], d.losses[te]
    ok = (w + l) > 0
    return float(spearmanr(p[ok], (w / np.maximum(w + l, 1))[ok]).statistic)


def main(set_code):
    torch.set_num_threads(4)
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
    Fd = (deck_features(deck, info)[0] - mu) / sd
    kernels = {"pick affinity": pick_affinity(set_code, names)[0], "deck PMI": pmi(X, nonland, sp),
               "colour overlap": colour_overlap(info)}
    cols, dcols = {}, {}
    for k, S in kernels.items():
        a = mean_pair(X, S, nonland)
        m_, s_ = a[sp == 0].mean(), a[sp == 0].std() + 1e-6
        cols[k] = ((a - m_) / s_)[:, None].astype(np.float32)
        dcols[k] = ((mean_pair(deck, S, nonland) - m_) / s_)[:, None]
    # how alike are the three columns?
    corr = {f"{a} ~ {b}": float(np.corrcoef(cols[a][sp == 0, 0], cols[b][sp == 0, 0])[0, 1])
            for i, a in enumerate(cols) for b in list(cols)[i + 1:]}
    te = np.flatnonzero(sp == 2)
    variants = {"deck features": [], **{f"+ {k}": [k] for k in kernels}, "+ all three": list(kernels)}
    rows, base = [], None
    for name, use in variants.items():
        c = np.hstack([ctx, (F - mu) / sd] + [cols[k] for k in use]).astype(np.float32)
        fits = [fit(X, c, y, sp, 0, l2, 0, log=lambda _: None) for l2 in (1e-5, 1e-4)]
        m, v, predict = min(fits, key=lambda f: f[1])
        p = predict(te)
        if base is None:
            base = nll(y[te], p)
        dl = nll(y[te], p) - base
        wc = m.cov.weight.detach().numpy()[0, ctx.shape[1]:]
        with torch.no_grad():
            s = m.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
        s = s + np.hstack([Fd] + [dcols[k] for k in use]) @ wc
        r = {"model": name, "val logloss": v, "test logloss": log_loss(y[te], p),
             "vs deck features x1000": float(dl.mean() * 1000), "±95% x1000": float(1.96 * dl.std() / np.sqrt(len(dl)) * 1000),
             "column weights": dict(zip(use, wc[F.shape[1]:].round(4).tolist())),
             "draft spearman (rank)": calibrated_spearman(s, d, has_deck, False),
             "draft spearman (rank + skill)": calibrated_spearman(s, d, has_deck, True)}
        rows.append(r)
        print(f"  {set_code} {name:22s} vs DF {r['vs deck features x1000']:+.2f} ± {r['±95% x1000']:.2f}  "
              f"draft rho {r['draft spearman (rank)']:.4f} / +skill {r['draft spearman (rank + skill)']:.4f}  "
              f"w={r['column weights']}", flush=True)
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v2" / "affinity_checks.json"
    out.write_text(json.dumps({"set": set_code, "column correlations": corr, "models": rows}, indent=1))
    print(f"  {set_code} column correlations: { {k: round(v, 3) for k, v in corr.items()} }", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
