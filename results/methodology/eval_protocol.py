"""Evaluation protocol for deck-strength model comparisons (draft; see evaluation_design.md in this folder).

The question this answers: "is model B better than model A at rating DECKS, across the 12 sets, by more than
test-set noise and training noise?" It does three things:

1. Per set: a paired bootstrap of a metric difference B - A on the same resampled test units (drafts), optionally in
   time blocks (e.g. one block per day, so the format's day-to-day drift and autocorrelation are not treated as
   iid), and optionally with several training seeds per model, whose spread is added as a separate variance
   component:  se^2 = var_bootstrap(mean over seeds of delta) + var_seeds(delta) / S.
2. Across sets: fixed-effect and random-effects (DerSimonian-Laird) meta-analysis of the per-set deltas, with the
   Hartung-Knapp-Sidik-Jonkman (HKSJ) t_{k-1} interval (the right one for k = 12 studies), tau^2, I^2, a
   prediction interval for a new set, a sign test and a Wilcoxon signed-rank test.
3. A pre-registered decision rule (DecisionRule) and Holm's correction when several variants are tested at once.

Metrics are oriented so that HIGHER IS BETTER. Provided:
  spearman_wr          Spearman(p, W/(W+L)): the current headline (kept for continuity, never the decision metric)
  stratified_spearman  size-weighted mean of within-stratum Spearman (strata = skill bucket x rank x experience)
  neg_binomial_nll     -(draft-level binomial NLL per game) x 1000
  gll_metric           -(skill-controlled game log loss per game) x 1000, from per-draft loss sums, which come from
                       ctx_calibrated_game_loss(): a FIXED context logit per game (skill bucket, experience, rank,
                       record state, play/draw, mulligans; fit once, shared by all candidates) plus a + b * deck score,
                       with (a, b) fit on calibration games only (train or val), scored on test games.

Usage sketch (nothing here touches the parquet files; you pass arrays):

    from eval_protocol import (paired_bootstrap, spearman_wr, meta_analysis, decide, DecisionRule)
    per_set = []
    for s in SETS:
        A, B = load_preds(s)                       # [N] or [n_seeds, N] predicted p on test drafts
        w, l, day = load_records(s)
        fn = lambda p, idx: spearman_wr(p[idx], w[idx], l[idx])
        per_set.append(paired_bootstrap(fn, A, B, n_boot=500, seed=0, blocks=day, name=s))
    m = meta_analysis(per_set)
    print(m.summary()); print(decide(m, DecisionRule()))

Self-test (synthetic, < 10 s):  python results/methodology/eval_protocol.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from scipy.special import expit

# ---------------------------------------------------------------------------------------------------- metrics


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def spearman_wr(p: np.ndarray, w: np.ndarray, l: np.ndarray) -> float:
    """Spearman(p, W/(W+L)) over units with at least one game (the current headline metric)."""
    ok = (w + l) > 0
    return _spearman(p[ok], w[ok] / (w[ok] + l[ok]))


def stratified_spearman(p: np.ndarray, w: np.ndarray, l: np.ndarray, strata: np.ndarray, min_n: int = 30) -> float:
    """Size-weighted mean of within-stratum Spearman(p, W/(W+L)) (van Elteren-style). Strata with < min_n units
    are dropped. With strata = skill bucket x rank x experience, between-player skill differences that the strata
    capture cannot raise the score, so a model that learned *who drafted* gains much less than on spearman_wr."""
    ok = (w + l) > 0
    p, w, l, strata = p[ok], w[ok], l[ok], strata[ok]
    wr = w / (w + l)
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n = int(m.sum())
        if n >= min_n:
            r = _spearman(p[m], wr[m])
            if np.isfinite(r):
                num += n * r
                den += n
    return num / den if den else np.nan


def neg_binomial_nll(p: np.ndarray, w: np.ndarray, l: np.ndarray) -> float:
    """-(binomial NLL per game) x 1000 at the draft level (p^W (1-p)^L)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(1000 * (w * np.log(p) + l * np.log1p(-p)).sum() / (w + l).sum())


def ctx_calibrated_game_loss(score: np.ndarray, game_draft: np.ndarray, y: np.ndarray, ctx_logit: np.ndarray,
                             fit_games: np.ndarray, n_drafts: int | None = None, iters: int = 25):
    """Per-draft summed log loss of  logit P(win game g) = ctx_logit[g] + a + b * z(score[draft of g]).

    score:      [D] deck score per draft (a deck-only quantity: e.g. the model's logit at a reference rank, never
                using skill inputs). Standardised on the fit games' drafts.
    game_draft: [G] draft index of each game.  y: [G] 0/1.  ctx_logit: [G] frozen context logit (same for every
                candidate model; fit once on train games without any deck score).
    fit_games:  [G] bool, games used to fit (a, b): train or val games, never test.
    Returns (loss_sum [D], n_games [D]) over the games NOT in fit_games, plus (a, b). Feed loss_sum to gll_metric.
    """
    n_drafts = int(game_draft.max()) + 1 if n_drafts is None else n_drafts
    fd = np.unique(game_draft[fit_games])
    mu, sd = score[fd].mean(), score[fd].std() + 1e-12
    x = ((score - mu) / sd)[game_draft]
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    f = fit_games
    for _ in range(iters):  # Newton on (a, b) with ctx as offset
        pr = expit(ctx_logit[f] + X[f] @ beta)
        g = X[f].T @ (y[f] - pr)
        H = (X[f] * (pr * (1 - pr))[:, None]).T @ X[f] + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    ev = ~fit_games
    pr = np.clip(expit(ctx_logit[ev] + X[ev] @ beta), 1e-9, 1 - 1e-9)
    ll = -(y[ev] * np.log(pr) + (1 - y[ev]) * np.log1p(-pr))
    return (np.bincount(game_draft[ev], ll, minlength=n_drafts),
            np.bincount(game_draft[ev], minlength=n_drafts).astype(float), beta)


def gll_metric(n_games: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    """Metric fn for paired_bootstrap over drafts: -(test game log loss per game) x 1000 from per-draft sums."""
    def fn(loss_sum: np.ndarray, idx: np.ndarray) -> float:
        return float(-1000 * loss_sum[idx].sum() / max(n_games[idx].sum(), 1))
    return fn


# ---------------------------------------------------------------------------------------------- per-set test


@dataclass
class SetResult:
    name: str
    delta: float                 # point estimate of metric(B) - metric(A), mean over seeds
    se: float                    # total SE: sqrt(se_boot^2 + se_seed^2)
    se_boot: float               # test-set sampling noise (paired bootstrap)
    se_seed: float               # training noise: sd over seeds of delta / sqrt(S); 0 with one seed
    n_units: int
    n_seeds: int
    boot: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def z(self) -> float:
        return self.delta / self.se if self.se > 0 else np.nan


def _resample(rng, n: int, blocks: np.ndarray | None, block_index: list | None) -> np.ndarray:
    if blocks is None:
        return rng.integers(0, n, n)
    pick = rng.integers(0, len(block_index), len(block_index))
    return np.concatenate([block_index[i] for i in pick])


def paired_bootstrap(metric: Callable[[np.ndarray, np.ndarray], float], pred_a, pred_b, n_boot: int = 500,
                     seed: int = 0, blocks: np.ndarray | None = None, name: str = "",
                     seed_noise_floor: float = 0.0) -> SetResult:
    """Paired bootstrap of metric(B) - metric(A) on one set's test units.

    metric(pred, idx) -> float, higher is better; idx indexes test units (drafts). pred_a / pred_b are [N] or
    [S, N] (S training seeds, paired by seed index; if the counts differ, all seeds of each are averaged in the
    metric, not in the prediction). blocks: optional [N] block label (e.g. day of draft_time); whole blocks are
    resampled, which widens the interval when errors are correlated in time. seed_noise_floor: an SE to assume for
    training noise when only one seed was run (use the spread measured on a pilot set; 0 = ignore, optimistic).
    """
    A = np.atleast_2d(np.asarray(pred_a))
    B = np.atleast_2d(np.asarray(pred_b))
    n = A.shape[-1]
    rng = np.random.default_rng(seed)
    block_index = None
    if blocks is not None:
        _, inv = np.unique(blocks, return_inverse=True)
        block_index = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    full = np.arange(n)

    def delta(idx):
        return np.mean([metric(b, idx) for b in B]) - np.mean([metric(a, idx) for a in A])

    point = delta(full)
    boot = np.array([delta(_resample(rng, n, blocks, block_index)) for _ in range(n_boot)])
    se_boot = float(np.nanstd(boot, ddof=1))
    if len(A) == len(B) and len(A) > 1:
        per_seed = np.array([metric(b, full) - metric(a, full) for a, b in zip(A, B)])
        se_seed = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
    elif len(A) > 1 or len(B) > 1:
        va = np.var([metric(a, full) for a in A], ddof=1) / len(A) if len(A) > 1 else 0.0
        vb = np.var([metric(b, full) for b in B], ddof=1) / len(B) if len(B) > 1 else 0.0
        se_seed = float(np.sqrt(va + vb))
    else:
        se_seed = float(seed_noise_floor)
    return SetResult(name, float(point), float(np.hypot(se_boot, se_seed)), se_boot, se_seed, n,
                     max(len(A), len(B)), boot)


# ---------------------------------------------------------------------------------------------- meta-analysis


@dataclass
class MetaResult:
    names: list
    deltas: np.ndarray
    ses: np.ndarray
    fe_mean: float
    fe_se: float
    tau2: float
    i2: float
    q_p: float
    re_mean: float
    re_se_hksj: float
    re_ci: tuple
    re_p_two_sided: float
    pred_interval: tuple
    n_positive: int
    sign_p_one_sided: float
    wilcoxon_p_two_sided: float
    n_sig_worse: int             # sets with delta / se < -1.96

    def summary(self) -> str:
        k = len(self.deltas)
        rows = "\n".join(f"  {n:6s} {d:+.5f} ± {s:.5f}  z {d / s:+.2f}" for n, d, s in zip(self.names, self.deltas,
                                                                                      self.ses))
        return (f"{rows}\n  fixed effect {self.fe_mean:+.5f} ± {self.fe_se:.5f}\n"
                f"  random effects (DL + HKSJ) {self.re_mean:+.5f}, 95% CI [{self.re_ci[0]:+.5f}, {self.re_ci[1]:+.5f}],"
                f" p = {self.re_p_two_sided:.4f}; tau = {np.sqrt(self.tau2):.5f}, I2 = {self.i2:.0%}, Q p = {self.q_p:.3f}\n"
                f"  95% prediction interval for a new set [{self.pred_interval[0]:+.5f}, {self.pred_interval[1]:+.5f}]\n"
                f"  sets better {self.n_positive}/{k} (sign test one-sided p = {self.sign_p_one_sided:.3f}), "
                f"Wilcoxon p = {self.wilcoxon_p_two_sided:.3f}, significantly worse on {self.n_sig_worse}")


def meta_analysis(results: Sequence[SetResult]) -> MetaResult:
    """Inverse-variance meta-analysis of per-set paired deltas. Random effects: DerSimonian-Laird tau^2; the mean's
    interval uses HKSJ (weighted residual variance, t with k-1 df), which keeps nominal coverage for k ~ 12 where the
    plain DL z-interval is too narrow. Sets are treated as exchangeable 'studies'; the estimand is the mean effect
    over sets like these, and the prediction interval says what to expect on the next set."""
    d = np.array([r.delta for r in results], float)
    s = np.array([r.se for r in results], float)
    k = len(d)
    w = 1 / s ** 2
    fe = float(np.sum(w * d) / w.sum())
    fe_se = float(np.sqrt(1 / w.sum()))
    Q = float(np.sum(w * (d - fe) ** 2))
    c = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    ws = 1 / (s ** 2 + tau2)
    re = float(np.sum(ws * d) / ws.sum())
    hk = np.sum(ws * (d - re) ** 2) / ((k - 1) * ws.sum())
    # "modified" HKSJ: never narrower than the DL z-variance (guards against HKSJ collapsing when deltas agree
    # more closely than their SEs imply, which is common here because the SEs are bootstrap estimates)
    re_se = float(np.sqrt(max(hk, 1 / ws.sum())))
    tcrit = stats.t.ppf(0.975, k - 1)
    p_re = float(2 * stats.t.sf(abs(re / re_se), k - 1))
    pi_half = stats.t.ppf(0.975, max(k - 2, 1)) * np.sqrt(tau2 + re_se ** 2)
    npos = int(np.sum(d > 0))
    sign_p = float(stats.binomtest(npos, k, 0.5, alternative="greater").pvalue)
    try:
        wil = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        wil = np.nan
    return MetaResult([r.name for r in results], d, s, fe, fe_se, float(tau2), float(i2),
                      float(stats.chi2.sf(Q, k - 1)), re, re_se, (re - tcrit * re_se, re + tcrit * re_se), p_re,
                      (re - pi_half, re + pi_half), npos, sign_p, wil, int(np.sum(d / s < -1.96)))


# ------------------------------------------------------------------------------------------------ decisions


@dataclass
class DecisionRule:
    """Pre-registered adoption rule. Fix it (and the primary metric, the sets, the seeds) BEFORE looking at test.

    alpha:          two-sided level for the random-effects (HKSJ) mean; with several variants tested in one round,
                    pass the Holm-adjusted p (holm()) via `p_override`.
    min_positive:   minimum number of sets with delta > 0 (8 of 12: a consistency guard, not a test by itself;
                    P(>= 8/12 | null) = 0.19).
    min_effect:     smallest effect worth adopting on the primary metric (e.g. 0.02 x1000 nats/game); the CI's
                    lower bound need not exceed it, but the point estimate must.
    max_sig_worse:  sets allowed with delta / se < -1.96 (default 0: no set may be significantly hurt).
    """
    alpha: float = 0.05
    min_positive: int = 8
    min_effect: float = 0.0
    max_sig_worse: int = 0


def decide(m: MetaResult, rule: DecisionRule = DecisionRule(), p_override: float | None = None) -> tuple[bool, list]:
    p = m.re_p_two_sided if p_override is None else p_override
    checks = [
        (m.re_mean > 0 and p < rule.alpha, f"RE mean {m.re_mean:+.5f} > 0 with p = {p:.4f} < {rule.alpha}"),
        (m.n_positive >= rule.min_positive, f"{m.n_positive}/{len(m.deltas)} sets better >= {rule.min_positive}"),
        (m.re_mean >= rule.min_effect, f"RE mean >= min_effect {rule.min_effect}"),
        (m.n_sig_worse <= rule.max_sig_worse, f"{m.n_sig_worse} sets significantly worse <= {rule.max_sig_worse}"),
    ]
    return all(ok for ok, _ in checks), [("PASS " if ok else "FAIL ") + msg for ok, msg in checks]


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control across variants tested in one round)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------------------------------------ self-test


def _synthetic_set(rng, n: int, gain: float, leak: float = 0.0):
    """One synthetic set: true strength theta, skill k, iid Bo1 games to 7 wins / 3 losses, two predictors."""
    theta = rng.normal(0, 0.3, n)
    k = rng.normal(0, 0.25, n)
    w = np.zeros(n)
    l = np.zeros(n)
    live = np.ones(n, bool)
    p_true = expit(theta + k)
    while live.any():
        win = rng.random(n) < p_true
        w += live & win
        l += live & ~win
        live &= (w < 7) & (l < 3)
    eA = rng.normal(0, 0.4, n)
    eB = 0.97 * eA + np.sqrt(1 - 0.97 ** 2) * rng.normal(0, 0.4, n)
    a = theta + eA
    b = theta + (1 - gain) * eB + leak * k
    return expit(a), expit(b), w, l, np.digitize(k, [-0.2, 0, 0.2])


def _self_test():
    import time
    t0 = time.time()
    rng = np.random.default_rng(1)

    # 1. a real gain on 12 sets is detected, a null is not adopted
    for gain, label in ((0.06, "real gain"), (0.0, "null")):
        res = []
        for s in range(12):
            a, b, w, l, _ = _synthetic_set(rng, 6000, gain)
            fn = lambda p, idx, w=w, l=l: spearman_wr(p[idx], w[idx], l[idx])  # noqa: E731
            res.append(paired_bootstrap(fn, a, b, n_boot=60, seed=s, name=f"S{s:02d}"))
        m = meta_analysis(res)
        ok, why = decide(m)
        print(f"[{label}] adopt={ok}  RE {m.re_mean:+.4f} CI [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] "
              f"{m.n_positive}/12 better")
        if label == "real gain":
            assert ok, why
        else:
            assert not ok or m.re_ci[0] < 0.004, why

    # 2. skill leakage: raw Spearman rewards it, within-skill-stratum Spearman much less
    a, b, w, l, strata = _synthetic_set(rng, 20000, 0.0, leak=0.6)
    raw = spearman_wr(b, w, l) - spearman_wr(a, w, l)
    strat = stratified_spearman(b, w, l, strata) - stratified_spearman(a, w, l, strata)
    print(f"[leak] raw Spearman gain {raw:+.4f}, stratified {strat:+.4f}")
    assert raw > 2 * strat > 0 or raw > 0.02 > strat

    # 3. HKSJ keeps ~nominal type-I error with heterogeneity, k = 12 (fast: no bootstrap, known SEs)
    rej = 0
    trials = 2000
    for _ in range(trials):
        se = rng.uniform(0.0012, 0.0025, 12)
        d = rng.normal(0, 0.001, 12) + rng.normal(0, se)  # tau = 0.001, true mean 0
        m = meta_analysis([SetResult(str(i), di, si, si, 0.0, 1, 1) for i, (di, si) in enumerate(zip(d, se))])
        rej += m.re_p_two_sided < 0.05
    print(f"[HKSJ] type-I error at tau=0.001: {rej / trials:.3f} (nominal 0.05)")
    assert rej / trials < 0.09

    # 4. seeds add a variance component; game-loss metric runs end to end
    a, b, w, l, _ = _synthetic_set(rng, 3000, 0.05)
    A = np.stack([np.clip(a + rng.normal(0, 0.01, len(a)), 0.01, 0.99) for _ in range(3)])
    B = np.stack([np.clip(b + rng.normal(0, 0.01, len(b)), 0.01, 0.99) for _ in range(3)])
    r = paired_bootstrap(lambda p, idx: neg_binomial_nll(p[idx], w[idx], l[idx]), A, B, n_boot=40,
                         blocks=np.arange(len(a)) // 100)
    print(f"[seeds] delta {r.delta:+.4f} x1000 nats/game, se_boot {r.se_boot:.4f}, se_seed {r.se_seed:.4f}")
    assert r.se_seed > 0 and r.se >= r.se_boot
    gd = np.repeat(np.arange(len(a)), (w + l).astype(int))
    y = np.concatenate([np.r_[np.ones(int(wi)), np.zeros(int(li))] for wi, li in zip(w, l)])
    fit = gd < len(a) // 2
    ctx = np.zeros(len(y))
    ls_a, ng, _ = ctx_calibrated_game_loss(np.log(a / (1 - a)), gd, y, ctx, fit, len(a))
    ls_b, _, _ = ctx_calibrated_game_loss(np.log(b / (1 - b)), gd, y, ctx, fit, len(a))
    test = np.flatnonzero(ng > 0)
    g = paired_bootstrap(gll_metric(ng), ls_a[None, :], ls_b[None, :], n_boot=40)
    print(f"[gll] skill-controlled game log loss delta {g.delta:+.4f} ± {g.se:.4f} x1000 on {len(test)} drafts")
    print(f"[holm] {holm([0.01, 0.04, 0.03, 0.2]).round(3)}")
    print(f"self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    _self_test()
