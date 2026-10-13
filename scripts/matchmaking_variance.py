"""True win-rate variance under record-based matchmaking: python scripts/matchmaking_variance.py MSH [SET ...]

(after variance_explained.py, which caches the deck models' predictions.) `check` as the first argument runs the
simulation check alone.

Why: a draft's games are not iid. Arena pairs players by record, so a winning run meets stronger opponents, which
pulls records toward 50%. A random-effects model on records alone finds no true variance at all, and free
per-record terms trade off against it (variance_explained.py). Here the matchmaking term is not free; it follows
from the strength spread itself:
  population   strength e ~ N(0, s^2) on the logit scale. Every run starts 0-0, and each game is against a random
               player at the same record: P(win) = E_opp[expit(e - e_opp)]. The strength distribution of players
               at each record (w, l) follows by recursion over 7-win / 3-loss runs (quadrature nodes, no sampling).
  17Lands users  e_i = d + X_i b + u_i, u_i ~ N(0, t^2), facing that population. Each game is scored at its record
               before the game (complete histories only).
Stage 1 (X empty) gives the users' true spread t0 (with s = t0, users as spread out as everyone). Stage 2 adds a
deck model's logit (and optionally the skill bucket), with s fixed: share of true variance explained = 1 - t^2/t0^2.
Writes results/{SET}.PremierDraft/matchmaking_variance.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.games import flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import load_drafts, record_before  # noqa: E402

X_GH, W_GH = np.polynomial.hermite_e.hermegauss(32)
W_GH = W_GH / W_GH.sum()
STATES = [(w, l) for w in range(7) for l in range(3)]
SID = {s: i for i, s in enumerate(STATES)}


def opponents(s: float) -> np.ndarray:
    """[21, nodes] strength distribution of the players at each record (weights on nodes s * X_GH)."""
    f = {st: np.zeros(len(X_GH)) for st in STATES}
    f[(0, 0)] = W_GH.copy()
    for t in range(9):
        for w, l in [st for st in STATES if sum(st) == t]:
            m = f[(w, l)]
            if m.sum() == 0:
                continue
            g = m / m.sum()
            p = expit(s * (X_GH[:, None] - X_GH[None])) @ g  # node k vs a random opponent at this record
            if w + 1 < 7:
                f[(w + 1, l)] += m * p
            if l + 1 < 3:
                f[(w, l + 1)] += m * (1 - p)
    return np.stack([f[st] / max(f[st].sum(), 1e-300) for st in STATES])


def user_nll(mean, t, s, opp, y, gd, gs, D):
    """Mean NLL per game of the users' games: user strength mean[i] + t * node, opponents opp[state]."""
    eta = mean[gd][:, None] + t * X_GH[None]  # [games, nodes]
    p = np.empty_like(eta)
    for k in np.unique(gs):
        sel = gs == k
        p[sel] = expit(eta[sel][..., None] - s * X_GH[None, None]) @ opp[k]
    p = np.clip(p, 1e-9, 1 - 1e-9)
    lg = np.where(y[:, None] > 0, np.log(p), np.log1p(-p))
    ll = np.zeros((D, len(X_GH)))
    np.add.at(ll, gd, lg)
    m = ll.max(1, keepdims=True)
    return -(m.squeeze(1) + np.log(np.exp(ll - m) @ W_GH)).sum() / len(y)


def fit_total(y, gd, gs, D):
    """Stage 1: users' mean d and spread t0, population spread s = t0."""
    def f(th):
        t = np.exp(th[1])
        return user_nll(np.full(D, th[0]), t, t, opponents(t), y, gd, gs, D)
    grid = [(f([0.0, np.log(t)]), t) for t in (0.05, 0.2, 0.4, 0.6, 0.9)]
    t_init = min(grid)[1]
    r = minimize(f, [0.0, np.log(t_init)], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
    return float(r.x[0]), float(np.exp(r.x[1])), {f"{t}": v for v, t in grid}


def fit_covariates(Z, s, y, gd, gs, D):
    """Stage 2: users' strength d + Z b + u, u ~ N(0, t^2); population spread s fixed. Returns (b, t)."""
    opp = opponents(s)
    q = Z.shape[1]

    def f(th):
        return user_nll(th[0] + Z @ th[1:1 + q], np.exp(th[-1]), s, opp, y, gd, gs, D)
    r = minimize(f, np.r_[0.0, np.zeros(q), np.log(s)], method="L-BFGS-B")
    return r.x[1:1 + q], float(np.exp(r.x[-1]))


def simulate(n, s, d, b, t, seed=0):
    """Users with strength d + b z + u (z ~ N(0, 1)) playing full runs against the matchmaking population."""
    rng = np.random.default_rng(seed)
    opp = opponents(s)
    z = rng.standard_normal(n)
    e = d + b * z + t * rng.standard_normal(n)
    y, gd, gs = [], [], []
    w, l = np.zeros(n, int), np.zeros(n, int)
    live = np.ones(n, bool)
    while live.any():
        i = np.flatnonzero(live)
        st = np.array([SID[(a, c)] for a, c in zip(w[i], l[i])])
        node = np.array([rng.choice(len(X_GH), p=opp[k]) for k in st])
        win = rng.random(len(i)) < expit(e[i] - s * X_GH[node])
        y.append(win.astype(float)), gd.append(i), gs.append(st)
        w[i] += win
        l[i] += ~win
        live &= (w < 7) & (l < 3)
    return z, np.concatenate(y), np.concatenate(gd), np.concatenate(gs)


def check():
    """Recover known spread and share from simulated users (share = b^2 / (b^2 + t^2))."""
    for s_true, share in ((0.3, 0.3), (0.5, 0.5)):
        b, t = s_true * np.sqrt(share), s_true * np.sqrt(1 - share)
        z, y, gd, gs = simulate(10000, s_true, 0.05, b, t)
        d0, t0, _ = fit_total(y, gd, gs, 10000)
        bb, tt = fit_covariates(z[:, None], t0, y, gd, gs, 10000)
        naive_sd = np.sqrt(max(np.var(np.bincount(gd, y) / np.bincount(gd)) - 0, 0))
        print(f"  true s {s_true:.2f}, share {share:.2f} -> fitted s {t0:.3f}, share {1 - tt ** 2 / t0 ** 2:.3f} "
              f"(b {bb[0]:.3f} vs {b:.3f}; observed win-rate sd {naive_sd:.3f})", flush=True)


def main(*sets):
    if sets and sets[0] == "check":
        print("simulation check", flush=True)
        check()
        return
    for set_code in sets:
        t_0 = time.time()
        out = ROOT / "results" / f"{set_code}.PremierDraft"
        preds = dict(np.load(out / "variance_preds.npz"))
        d = load_drafts(set_code, "PremierDraft")
        has = np.isfinite(preds["round 4 blend"])
        k = d.mask("test") & has & ((d.wins + d.losses) > 0)
        pos = {x: i for i, x in enumerate(d.draft_id[k].tolist())}
        games = pl.read_parquet(game_path(set_code, "PremierDraft"), columns=["draft_id", "match_number", "won"])
        games = games.filter(pl.col("draft_id").is_in(list(pos)))
        wb, lb, known = record_before(games)
        games, wb, lb = games.filter(pl.Series(known)), wb[known], lb[known]
        y = flag(games["won"]).astype(np.float64)
        gd = np.array([pos[x] for x in games["draft_id"].to_list()])
        gs = np.array([SID[(min(a, 6), min(c, 2))] for a, c in zip(wb, lb)])
        D = int(k.sum())
        d0, t0, grid = fit_total(y, gd, gs, D)
        res = {"test drafts": D, "games": int(len(y)), "users mean": d0, "true logit sd": t0,
               "true win-rate sd (approx)": t0 * 0.25, "stage 1 grid nll": grid,
               "observed win-rate sd": float(np.std(d.wins[k] / (d.wins[k] + d.losses[k]))), "share explained": {}}
        print(f"=== {set_code}: true logit sd {t0:.3f} (win rate ~ +-{t0 * 25:.1f} pts), users mean {d0:+.3f} "
              f"[{time.time() - t_0:.0f}s]", flush=True)
        sk = d.skill[k]
        skill = np.column_stack([np.nan_to_num(sk - np.nanmean(sk)), np.isnan(sk)])
        for name, p in preds.items():
            zz = logit(np.clip(p[k], 1e-6, 1 - 1e-6))
            zz = (zz - zz.mean())[:, None]
            b, t = fit_covariates(zz, t0, y, gd, gs, D)
            _, t_sk = fit_covariates(np.hstack([zz, skill]), t0, y, gd, gs, D)
            res["share explained"][name] = 1 - t ** 2 / t0 ** 2
            res["share explained"][f"{name} + skill bucket"] = 1 - t_sk ** 2 / t0 ** 2
            res[f"{name}: slope"] = float(b[0])
            print(f"  {name:36s} share {1 - t ** 2 / t0 ** 2:.3f}  (+ skill bucket {1 - t_sk ** 2 / t0 ** 2:.3f}) "
                  f"slope {b[0]:.2f} [{time.time() - t_0:.0f}s]", flush=True)
        _, t_sk = fit_covariates(skill, t0, y, gd, gs, D)
        res["share explained"]["skill bucket alone (leaky)"] = 1 - t_sk ** 2 / t0 ** 2
        print(f"  skill bucket alone (leaky)           share {1 - t_sk ** 2 / t0 ** 2:.3f}", flush=True)
        (out / "matchmaking_variance.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
