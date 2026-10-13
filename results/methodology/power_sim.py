"""Synthetic power study for the evaluation design (results/methodology/evaluation_design.md, section 4).

    python results/methodology/power_sim.py          (~15 s, synthetic data only, no project files read)

World: each draft has a true deck strength theta ~ N(0, 0.30) (logit; variance.json puts the true logit sd at ~0.30
on MSH) and a player skill k ~ N(0, 0.25). Each game's logit is theta + k - m * (wins - losses so far), where the last
term is record-based matchmaking (stronger opponents after wins). Runs stop at 7 wins or 3 losses. A coarse,
noisy skill bucket b = round(k + noise) is observed, as in 17Lands.

Models: A predicts theta with noise (corr ~0.60 with theta). B is A moved a fraction delta of the way toward theta
(a real improvement in deck strength). C is A plus lambda * k (it learns *who drafted*: skill leakage, no
deck-strength gain). For each pair we report, per set of N test drafts:
  raw Spearman(p, W/(W+L))       the headline metric
  stratified Spearman           mean within-skill-bucket Spearman (weights = stratum size)
  game log loss (ctx-controlled) per-game binomial log loss with skill bucket + record state in a fixed context
and the paired-bootstrap z of the B-A (or C-A) difference. MDE (80% power, two-sided 0.05) is 2.8 x the paired SE.
"""
import time

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

RNG = np.random.default_rng(0)
SD_THETA, SD_SKILL, MM = 0.30, 0.25, 0.08


def simulate(n, rng):
    theta = rng.normal(0, SD_THETA, n)
    k = rng.normal(0, SD_SKILL, n)
    bucket = np.clip(np.round((k + rng.normal(0, 0.10, n)) / 0.1), -4, 4).astype(int)  # 9 coarse buckets
    w = np.zeros(n, int)
    l = np.zeros(n, int)
    live = np.ones(n, bool)
    games = []  # (draft, wins_before, losses_before, won)
    while live.any():
        idx = np.flatnonzero(live)
        z = theta[idx] + k[idx] - MM * (w[idx] - l[idx])
        won = rng.random(len(idx)) < expit(z)
        games.append(np.column_stack([idx, w[idx], l[idx], won]))
        w[idx] += won
        l[idx] += ~won
        live &= (w < 7) & (l < 3)
    return theta, k, bucket, w, l, np.vstack(games)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return np.corrcoef(rx, ry)[0, 1]


def strat_spearman(x, y, s):
    out, wt = 0.0, 0
    for v in np.unique(s):
        m = s == v
        if m.sum() > 20:
            out += m.sum() * spearman(x[m], y[m])
            wt += m.sum()
    return out / wt


def ctx_logloss_per_draft(score, bucket, games, n):
    """Per-draft summed log loss of the fixed-context model + b * score, b fit by a few Newton steps.
    Context = skill-bucket and record-state fixed effects, fit once without any deck score (shared by models)."""
    d, wb, lb, y = games[:, 0], games[:, 1], games[:, 2], games[:, 3].astype(float)
    # fixed context offset: empirical logit per (bucket, record state)
    key = (bucket[d] + 4) * 100 + wb * 10 + lb
    uniq, inv = np.unique(key, return_inverse=True)
    rate = (np.bincount(inv, y) + 1) / (np.bincount(inv) + 2)
    off = np.log(rate / (1 - rate))[inv]
    s = (score - score.mean()) / score.std()
    x = s[d]
    b = 0.0
    for _ in range(8):
        p = expit(off + b * x)
        b += np.sum((y - p) * x) / np.sum(p * (1 - p) * x * x)
    p = np.clip(expit(off + b * x), 1e-9, 1 - 1e-9)
    ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return np.bincount(d, ll, minlength=n), np.bincount(d, minlength=n)


def paired(n, delta, lam, n_boot=200, rng=RNG, r=0.97):
    theta, k, bucket, w, l, games = simulate(n, rng)
    eA = rng.normal(0, SD_THETA * 1.33, n)  # corr(sA, theta) ~ 0.60
    sA = theta + eA
    eta = rng.normal(0, SD_THETA * 1.33, n)  # model-specific noise: corr(eA, eB) = r, as for real model pairs
    sB = theta + (1 - delta) * (r * eA + np.sqrt(1 - r * r) * eta)
    sC = sA + lam * k
    sX = sB if lam == 0 else sC
    wr = w / (w + l)
    llA, ng = ctx_logloss_per_draft(sA, bucket, games, n)
    llX, _ = ctx_logloss_per_draft(sX, bucket, games, n)
    point = {"raw": spearman(sX, wr) - spearman(sA, wr),
             "strat": strat_spearman(sX, wr, bucket) - strat_spearman(sA, wr, bucket),
             "gll": (llA.sum() - llX.sum()) / ng.sum()}  # >0 = X better
    boots = {kk: [] for kk in point}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots["raw"].append(spearman(sX[i], wr[i]) - spearman(sA[i], wr[i]))
        boots["strat"].append(strat_spearman(sX[i], wr[i], bucket[i]) - strat_spearman(sA[i], wr[i], bucket[i]))
        boots["gll"].append((llA[i].sum() - llX[i].sum()) / ng[i].sum())
    true_gain = np.corrcoef(sX, theta)[0, 1] - np.corrcoef(sA, theta)[0, 1]
    return {kk: (point[kk], float(np.std(boots[kk]))) for kk in point}, true_gain, float(np.corrcoef(sA, sX)[0, 1])


if __name__ == "__main__":
    t0 = time.time()
    T = 2.201 + 0.876  # t_{0.975,11} + t_{0.80,11}: HKSJ random-effects test over 12 sets, 80% power
    for n in (10000, 25000):
        for delta in (0.0, 0.03):
            r, g, c = paired(n, delta, 0.0, n_boot=60)
            print(f"N={n} real improvement delta={delta:.2f}: true corr(score,theta) gain {g:+.4f}, corr(sA,sB) {c:.3f}")
            for kk, (pt, se) in r.items():
                u = 1000 if kk == "gll" else 1
                print(f"   {kk:6s} diff {pt * u:+.5f}  paired SE {se * u:.5f}  z {pt / se:+.2f}  per-set MDE {2.8 * se * u:.5f}"
                      f"  12-set MDE tau=0 {T * se / np.sqrt(12) * u:.5f}  tau=SE {T * se * np.sqrt(2 / 12) * u:.5f}"
                      + (f"  MDE in true-corr units (12 sets, tau=0) {T / np.sqrt(12) * g / (pt / se):.4f}" if delta else ""))
    print("Skill leakage (C = A + 0.5 * skill), no deck gain, N=15000")
    r, g, c = paired(15000, 0.0, 0.5, n_boot=60)
    print(f"  true corr(score,theta) gain {g:+.4f}")
    for kk, (pt, se) in r.items():
        print(f"   {kk:6s} diff {pt:+.5f}  paired SE {se:.5f}  z {pt / se:+.2f}")
    print(f"[{time.time() - t0:.0f}s]  (gll in nats/game x 1000)")
