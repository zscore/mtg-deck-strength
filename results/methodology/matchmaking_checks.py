"""Cheap identification checks behind matchmaking_identification.md (seconds, < 300 MB; reads a few game columns).

python results/methodology/matchmaking_checks.py MSH [SET ...]

1. play/draw instrument: being on the play in game t-1 (a coin flip in Bo1) raises P(win t-1); under record matching
   that extra win buys a stronger opponent in game t. Reduced form / first stage = Wald estimate of
   P(win | state after a win) - P(win | state after a loss), restricted to pre-states (w<=5, l<=1) where game t exists
   whatever happens. Compared with the value implied by population logit spread s (record-matching recursion).
2. opponent colour-pair strength by record before the game (direct evidence that opponents get stronger).
3. time linkage: a draft whose start follows, within 0-2 min, the last game of a run with the same
   (games bucket, win-rate bucket) and the same rank. Placebo: runs ending 0-5 min *after* the draft starts.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mtgdata.games import game_path  # noqa: E402

X, W = np.polynomial.hermite_e.hermegauss(32)
W = W / W.sum()
STATES = [(w, l) for w in range(7) for l in range(3)]


def opponents(s):
    f = {st: np.zeros(len(X)) for st in STATES}
    f[(0, 0)] = W.copy()
    for t in range(9):
        for w, l in [st for st in STATES if sum(st) == t]:
            m = f[(w, l)]
            p = expit(s * (X[:, None] - X[None])) @ (m / m.sum())
            if w + 1 < 7:
                f[(w + 1, l)] += m * p
            if l + 1 < 3:
                f[(w, l + 1)] += m * (1 - p)
    return {st: f[st] / f[st].sum() for st in STATES}, {st: f[st].sum() for st in STATES}


def implied_wald(s, d=0.3):
    opp, mass = opponents(s)
    P = lambda st: float(W @ expit(d + s * X[:, None] - s * X[None]) @ opp[st])  # noqa: E731
    pre = [(w, l) for w, l in STATES if w <= 5 and l <= 1]
    return np.average([P((w + 1, l)) - P((w, l + 1)) for w, l in pre], weights=[mass[k] for k in pre])


def ols(X_, y):
    b, *_ = np.linalg.lstsq(X_, y, rcond=None)
    r = y - X_ @ b
    return b, np.sqrt(np.diag(np.linalg.pinv(X_.T @ X_) * r.var()))


def main(*sets):
    print("implied Wald by population spread s:", {s: round(implied_wald(s), 4) for s in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5)})
    for set_code in sets:
        g = pl.read_parquet(game_path(set_code, "PremierDraft"), columns=[
            "draft_id", "draft_time", "game_time", "match_number", "rank", "user_n_games_bucket",
            "user_game_win_rate_bucket", "on_play", "won", "main_colors", "opp_colors"])
        g = (g.with_columns(pl.col("match_number").cast(pl.Int32), (pl.col("won") == "True").cast(pl.Float64).alias("y"),
                            (pl.col("on_play") == "True").cast(pl.Float64).alias("play"))
             .sort("draft_id", "match_number")
             .with_columns(pl.int_range(pl.len()).over("draft_id").alias("k"),
                           (pl.col("y").cum_sum().over("draft_id") - pl.col("y")).alias("wb")))
        g = g.with_columns((pl.col("k") - pl.col("wb")).alias("lb"), (pl.col("match_number") == pl.col("k") + 1).alias("known"))
        g = g.with_columns(*[pl.col(c).shift(1).over("draft_id").alias(c + "_p") for c in ("play", "wb", "lb", "y")])
        h = g.filter(pl.col("known") & pl.col("play_p").is_not_null() & (pl.col("wb_p") <= 5) & (pl.col("lb_p") <= 1))
        _, st = np.unique((h["wb_p"] * 2 + h["lb_p"]).cast(pl.Int32).to_numpy(), return_inverse=True)
        F = np.eye(st.max() + 1)[st]
        b, se = ols(np.column_stack([h["play_p"].to_numpy(), h["play"].to_numpy(), F]), h["y"].to_numpy())
        b1, _ = ols(np.column_stack([h["play_p"].to_numpy(), F]), h["y_p"].to_numpy())
        print(f"=== {set_code}: N={h.height} reduced form {b[0]:+.4f} (se {se[0]:.4f}), first stage {b1[0]:+.4f}, "
              f"Wald {b[0] / b1[0]:+.3f} (se ~{se[0] / b1[0]:.3f})")
        cs = g.group_by("main_colors").agg(pl.col("y").mean().alias("ocwr"), pl.len().alias("n")).filter(pl.col("n") > 500)
        oc = (g.filter("known").join(cs.rename({"main_colors": "opp_colors"}), on="opp_colors")
              .group_by("wb").agg(pl.col("ocwr").mean()).sort("wb"))
        print("  opp colour-pair WR by wins before:", [round(v, 4) for v in oc["ocwr"].to_list()])
        g = g.with_columns(pl.col("draft_time").str.to_datetime(), pl.col("game_time").str.to_datetime())
        d = g.group_by("draft_id").agg(pl.col("draft_time").first(), pl.col("game_time").max().alias("end"),
                                       pl.col("rank").sort_by("game_time").first().alias("r0"),
                                       pl.col("rank").sort_by("game_time").last().alias("r1"),
                                       pl.col("user_n_games_bucket").first().alias("gb"),
                                       pl.col("user_game_win_rate_bucket").first().alias("wr"))
        s = d.sample(min(8000, d.height), seed=1).select("draft_id", "draft_time", "r0", "gb", "wr")
        e = d.select(pl.col("draft_id").alias("pid"), "end", pl.col("r1").alias("pr1"), "gb", "wr")
        # join only within neighbouring 10-minute time buckets (a plain join on the coarse bucket cells is a
        # near-cartesian product: ~7 GB on a large set)
        tb = lambda c: (pl.col(c).dt.epoch("s") // 600).alias("tb")  # noqa: E731
        s = pl.concat([s.with_columns((tb("draft_time") + k).alias("tb")) for k in (-1, 0, 1)])
        e = e.with_columns(tb("end"))
        j = (s.join(e, on=["gb", "wr", "tb"]).filter(pl.col("pid") != pl.col("draft_id"))
             .with_columns(((pl.col("draft_time") - pl.col("end")).dt.total_seconds() / 60).alias("gap")))
        for same in (True, False):
            jj = j.filter((pl.col("r0") == pl.col("pr1")) == same)
            hist = np.histogram(jj["gap"].to_numpy(), bins=[-5, -2, -1, 0, 1, 2, 3, 5])[0]
            print(f"  linkage candidates per gap bin [-5,-2,-1,0,1,2,3,5] min, {'same' if same else 'other'} rank: {hist.tolist()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
