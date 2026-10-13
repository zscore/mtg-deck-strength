"""Shared helpers for rebuild_targets.py, sc_gll.py and structural_a.py: memory-lean game-file loading and LightGBM.

The game parquet is never read whole (pipeline_audit / next_model_design: column-projected scans only):
  load_games    every game of the drafts in `d`, ~10 narrow columns, file order, with the file row number
  per_game      numpy arrays per game: draft index, y, split, record state before the game, context columns
  main_builds   each draft's most-played build, chosen exactly as decks.deckstrength.most_played_deck does
                (most games, ties to the lowest build_index in that column's own sort order, first game's row)
  deck_rows     deck_ counts for a list of file rows only (one row per build or per draft)
  build_table   one deck per (draft_id, build_index) and each game's build, for game-level models
  padded        per-draft [D, 9] arrays of a draft's games in match order (the layout of mm_likelihood.MMGameModel)
  lgbm_model    improve_deck_v2.lgbm, but returning the booster too (to predict at a reference rank)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results" / "methodology"))
sys.path.insert(0, str(REPO / "results" / "methodology" / "drafts"))
from mtgdata.data import RANK_ORDER  # noqa: E402
from mtgdata.games import aligned, flag  # noqa: E402
from decks.deckstrength import experience, rank_index, record_before  # noqa: E402

NARROW = ("draft_id", "build_index", "match_number", "won", "on_play", "num_mulligans", "opp_num_mulligans", "rank",
          "user_n_games_bucket", "user_game_win_rate_bucket")
G_MAX, N_STATES = 9, 21           # record states (w, l) w < 7, l < 3 in the order w * 3 + l; 21 = history incomplete
REF_RANK = RANK_ORDER["platinum"] + 1   # the reference opponent pool for deck-only LightGBM scores (rank_index units)


def load_games(path, draft_ids, cols=NARROW) -> pl.DataFrame:
    """Games of the given drafts, narrow columns, in file order, plus `row` (the row number in the file)."""
    lf = pl.scan_parquet(path)
    have = set(lf.collect_schema().names())
    return (lf.with_row_index("row").select("row", *[c for c in cols if c in have])
            .filter(pl.col("draft_id").is_in(pl.Series(list(draft_ids)).implode()))
            .collect(engine="streaming").sort("row"))


def per_game(d, g: pl.DataFrame) -> dict:
    """Numpy arrays per game (row order of `g`). gd indexes d's drafts; state as mm_likelihood.state_index."""
    pos = {x: i for i, x in enumerate(d.draft_id.tolist())}
    gd = np.array([pos[x] for x in g["draft_id"].to_list()], dtype=np.int64)
    wb, lb, known = record_before(g)
    num = lambda c: (g[c].cast(pl.Float64, strict=False).to_numpy() if c in g.columns  # noqa: E731
                     else np.full(g.height, np.nan))
    out = {"gd": gd, "y": flag(g["won"]).astype(np.float64), "split": d.split[gd].astype(np.int8),
           "match": g["match_number"].cast(pl.Int32).to_numpy(), "wins_before": wb, "losses_before": lb,
           "known": known, "state": np.where(known, np.minimum(wb, 6) * 3 + np.minimum(lb, 2), N_STATES),
           "build_index": g["build_index"].cast(pl.Utf8).to_numpy()}
    if "on_play" in g.columns:
        out["on_play"] = flag(g["on_play"]).astype(np.float64) - 0.5
        out["mull"] = np.nan_to_num(num("num_mulligans") - num("opp_num_mulligans"))
        out["exp"] = experience(g).astype(np.float64)
        out["rank"] = rank_index(g["rank"])
        out["n_bucket"] = num("user_n_games_bucket")
        out["skill"] = num("user_game_win_rate_bucket")
    return out


def main_builds(d, g: pl.DataFrame):
    """(has [D], main file row [D] (-1 without games), main build_index [D] as str) -- most_played_deck's rule."""
    b = (g.group_by("draft_id", "build_index").agg(pl.len().alias("n"), pl.col("row").min())
         .sort("n", "build_index", descending=[True, False])
         .unique("draft_id", keep="first"))
    row = dict(zip(b["draft_id"].to_list(), b["row"].to_list()))
    bix = dict(zip(b["draft_id"].to_list(), b["build_index"].cast(pl.Utf8).to_list()))
    has = np.array([x in row for x in d.draft_id])
    rows = np.array([row.get(x, -1) for x in d.draft_id], dtype=np.int64)
    return has, rows, np.array([bix.get(x, "") for x in d.draft_id], dtype=object)


def deck_rows(path, rows: np.ndarray, names: list) -> np.ndarray:
    """[len(rows), C] int8 deck_ counts of the given file rows (streaming scan: only those rows are materialised)."""
    lf = pl.scan_parquet(path)
    have = set(lf.collect_schema().names())
    cols = [f"deck_{n}" for n in names if f"deck_{n}" in have]
    u = np.unique(rows)
    df = (lf.select(cols).with_row_index("row")
          .filter(pl.col("row").is_in(pl.Series(u.astype(np.uint32), dtype=pl.UInt32).implode()))
          .collect(engine="streaming").sort("row"))
    got = df["row"].to_numpy().astype(np.int64)
    assert len(got) == len(u) and (np.diff(got) > 0).all()
    X = aligned(df, "deck_", names)
    return X[np.searchsorted(got, rows)]


def main_decks(path, d, g):
    """Like most_played_deck: [D, C] int8 counts of each draft's most-played build and the has-deck mask."""
    has, rows, _ = main_builds(d, g)
    deck = np.zeros((len(d.draft_id), len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    return deck, has


def build_table(path, d, g):
    """Xb [B, C] int8 (one deck per (draft, build_index), the build's first game's row), game_build [G] index into
    Xb, build_draft [B] draft index, main [D] index into Xb of the most-played build (-1 without games)."""
    b = g.group_by("draft_id", "build_index", maintain_order=True).agg(pl.col("row").min())
    key = {k: i for i, k in enumerate(zip(b["draft_id"].to_list(), b["build_index"].cast(pl.Utf8).to_list()))}
    game_build = np.array([key[k] for k in zip(g["draft_id"].to_list(), g["build_index"].cast(pl.Utf8).to_list())])
    pos = {x: i for i, x in enumerate(d.draft_id.tolist())}
    build_draft = np.array([pos[x] for x in b["draft_id"].to_list()])
    Xb = deck_rows(path, b["row"].to_numpy().astype(np.int64), d.names)
    has, _, mb = main_builds(d, g)
    main = np.array([key.get((x, m), -1) if h else -1 for x, m, h in zip(d.draft_id.tolist(), mb, has)])
    return Xb, game_build, build_draft, main


def padded(gd: np.ndarray, match: np.ndarray, D: int, G: int = G_MAX):
    """[D, G] index of each draft's games (in match_number order) into the game arrays, -1 = empty slot; and the
    number of games dropped beyond G (17Lands duplicates)."""
    o = np.lexsort((match, gd))
    gs = gd[o]
    start = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    slot = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    keep = slot < G
    idx = np.full((D, G), -1, np.int64)
    idx[gs[keep], slot[keep]] = o[keep]
    return idx, int((~keep).sum())


def day_blocks(d) -> np.ndarray:
    """Calendar day of each draft (for day-block bootstraps)."""
    return np.array([str(t)[:10] for t in d.draft_time])


def lgbm_model(X, d, fit, va=None, params=None, w=None, n_iter=None, seed=0):
    """improve_deck_v2.lgbm (W/L records as weighted rows; early stop on `va` or exactly n_iter trees), returning
    (booster, trees) so the caller can predict at a reference rank."""
    import lightgbm as lgb
    from improve_deck import LGB_PARAMS
    p = {**LGB_PARAMS, **(params or {}), "seed": seed}
    w = np.ones(len(X)) if w is None else w

    def ds(m, ref=None):
        y = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
        sw = np.r_[d.wins[m] * w[m], d.losses[m] * w[m]]
        k = sw > 0
        return lgb.Dataset(np.vstack([X[m], X[m]])[k], y[k], weight=sw[k], reference=ref)
    dtr = ds(fit)
    if va is not None:
        model = lgb.train(p, dtr, 5000, valid_sets=[ds(va, dtr)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return model, model.best_iteration
    return lgb.train(p, dtr, n_iter), n_iter


def selftest():
    """A tiny synthetic game parquet (temp dir): the lean loaders must reproduce decks.deckstrength.most_played_deck,
    record_before states and a full-file build table."""
    import tempfile
    from types import SimpleNamespace
    from decks.deckstrength import most_played_deck
    rng = np.random.default_rng(0)
    names = [f"c{j}" for j in range(12)]
    rows = []
    for i in range(400):
        nb = rng.choice([1, 1, 2, 3])
        decks = [rng.integers(0, 3, len(names)) for _ in range(nb)]
        n = rng.integers(0, 10) if i % 7 else 0  # some drafts without games
        for k in range(n):
            b = int(rng.integers(0, nb)) if k else 0
            rows.append({"draft_id": f"d{i:04d}", "build_index": str([0, 2, 10][b]), "match_number": k + 1,
                         "won": str(bool(rng.random() < 0.5)), "on_play": str(bool(rng.random() < 0.5)),
                         "num_mulligans": int(rng.integers(0, 2)), "opp_num_mulligans": int(rng.integers(0, 2)),
                         "rank": rng.choice(["gold", "platinum", None]),
                         "user_n_games_bucket": int(rng.choice([1, 50, 500])),
                         "user_game_win_rate_bucket": float(np.round(rng.uniform(0.4, 0.7), 2)),
                         **{f"deck_{c}": int(v) for c, v in zip(names[:-1], decks[b][:-1])}})  # last card absent
    g_all = pl.DataFrame(rows)
    d = SimpleNamespace(names=names, draft_id=np.array([f"d{i:04d}" for i in range(400)] + ["zz"]),
                        pool=np.zeros((401, len(names)), np.int8), split=np.repeat([0, 1, 2], [280, 60, 61]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "g.parquet"
        g_all.write_parquet(path, row_group_size=97)
        g = load_games(path, d.draft_id)
        deck, has = main_decks(path, d, g)
        ref_deck, ref_has = most_played_deck(d, g_all)
        assert (has == ref_has).all() and (deck == ref_deck).all()
        Xb, gb, bd, main = build_table(path, d, g)
        assert (Xb[gb] == aligned(g_all, "deck_", names)).all() and (Xb[main[has]] == deck[has]).all()
        pg = per_game(d, g)
        wb, lb, known = record_before(g_all)
        assert (pg["wins_before"] == wb).all() and known.all() and pg["exp"].shape[1] == 8
        idx, dropped = padded(pg["gd"], pg["match"], len(d.draft_id))
        assert dropped == int(np.maximum(np.bincount(pg["gd"]) - G_MAX, 0).sum())
        assert (pg["match"][idx[idx[:, 0] >= 0, 0]] == 1).all()
    print(f"eval_common self-test OK: {g.height} games, {len(Xb)} builds, {has.sum()} drafts with a deck, "
          f"{dropped} games beyond {G_MAX} slots")


if __name__ == "__main__":
    selftest()
