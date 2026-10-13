"""T0 gate counts (meta_review_2): python scripts/t0_counts.py SET [SET ...]   |   meta SET ...   |   --selftest

Narrow, column-projected scan of the game file (draft_id, build_index, match_number, won, num_turns, opp_colors,
user_n_games_bucket); no deck columns, no draft-file pool. Draft order and split come from a saved per-draft npz
(sc_gll/scores.npz or game_fm_v2/scores.npz: load_drafts order) or, for sets without one, decks.deckstrength
.load_drafts (torch-free shim). Numpy/polars only. Per set:
  multi-build   drafts with >= 2 builds that each played >= k games, k = 1, 2, 3, per split, and the games in them
                (T3 gate: >= 1,000 qualifying TEST drafts per set, or >= 5,000 pooled; reported at k = 1 and 2)
  shares        share of test games (drafts with games) per SC-GLL subset: experienced (n_bucket >= 50), match 1,
                main build (most-played build, sc_gll's rule), multi-build k = 1/2/3, experienced x multi-build
  opp colours   by the number of distinct colours in opp_colors (0 = nothing revealed): games, share, mean / median
                num_turns, user win rate; all games and test games (meta_review_2 (d): turns should rise with colours)
Writes results/{SET}.PremierDraft/t0_counts.json; `meta` prints pooled counts and the pooled gate.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, assert_no_torch, torchfree  # noqa: E402

torchfree()
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

COLS = ("draft_id", "build_index", "match_number", "won", "num_turns", "opp_colors", "user_n_games_bucket")
KS = (1, 2, 3)
GATE_SET, GATE_POOLED = 1000, 5000
SPLITS = ("train", "val", "test")


def draft_order(set_code):
    res = REPO / "results" / f"{set_code}.PremierDraft"
    for f in (res / "sc_gll" / "scores.npz", res / "game_fm_v2" / "scores.npz"):
        if f.exists():
            z = np.load(f)
            return z["draft_id"].astype(str), z["split"].astype(np.int8), str(f.relative_to(REPO))
    from decks.deckstrength import load_drafts
    d = load_drafts(set_code, "PremierDraft")
    return d.draft_id.astype(str), d.split, "load_drafts"


def n_colours(s):
    s = np.array([x if isinstance(x, str) else "" for x in s], dtype=object)
    return np.array([len(set(c for c in x if c in "WUBRG")) for x in s])


def counts(draft_id, split, g: pl.DataFrame) -> dict:
    from mtgdata.games import flag
    pos = {x: i for i, x in enumerate(draft_id.tolist())}
    keep = np.array([x in pos for x in g["draft_id"].to_list()])
    g = g.filter(pl.Series(keep))
    gd = np.array([pos[x] for x in g["draft_id"].to_list()])
    gsp = split[gd]
    D = len(draft_id)
    y = flag(g["won"]).astype(float)
    nb = g["user_n_games_bucket"].cast(pl.Float64, strict=False).to_numpy()
    match = g["match_number"].cast(pl.Int64, strict=False).to_numpy()
    b = (g.select(pl.Series("gd", gd), pl.col("build_index").cast(pl.Utf8)).with_row_index("i")
         .group_by("gd", "build_index").agg(pl.len().alias("n"), pl.col("i").min().alias("first")))
    bgd, bn = b["gd"].to_numpy(), b["n"].to_numpy()
    # main build: most games, ties to the lowest build_index string (eval_common.main_builds)
    mb = b.sort("n", "build_index", descending=[True, False]).unique("gd", keep="first")
    main_bix = np.full(D, None, dtype=object)
    main_bix[mb["gd"].to_numpy()] = mb["build_index"].to_list()
    gbix = g["build_index"].cast(pl.Utf8).to_numpy()
    main_game = gbix == main_bix[gd]
    has = np.bincount(gd, minlength=D) > 0
    out = {"drafts": {s: int((split == k).sum()) for k, s in enumerate(SPLITS)},
           "drafts with games": {s: int(((split == k) & has).sum()) for k, s in enumerate(SPLITS)},
           "games": {s: int((gsp == k).sum()) for k, s in enumerate(SPLITS)},
           "builds per draft (drafts with games)": {str(n): int(c) for n, c in
                                                    zip(*np.unique(np.bincount(bgd, minlength=D)[has], return_counts=True))},
           "multi-build": {}, "test game shares": {}}
    te = gsp == 2
    nte = max(int(te.sum()), 1)
    sh = out["test game shares"]
    sh["experienced (n_bucket >= 50)"] = float((te & (nb >= 50)).sum() / nte)
    sh["match 1"] = float((te & (match == 1)).sum() / nte)
    sh["main build"] = float((te & main_game).sum() / nte)
    for k in KS:
        nq = np.bincount(bgd, (bn >= k).astype(float), D)
        qual = nq >= 2
        qg = qual[gd]
        out["multi-build"][f"k={k}"] = {
            "qualifying drafts": {s: int((qual & (split == j)).sum()) for j, s in enumerate(SPLITS)},
            "share of drafts with games": {s: float((qual & (split == j)).sum() / max(((split == j) & has).sum(), 1))
                                           for j, s in enumerate(SPLITS)},
            "games in qualifying drafts": {s: int((qg & (gsp == j)).sum()) for j, s in enumerate(SPLITS)},
            "experienced test games in qualifying drafts": int((qg & te & (nb >= 50)).sum())}
        sh[f"multi-build k={k}"] = float((qg & te).sum() / nte)
        sh[f"experienced x multi-build k={k}"] = float((qg & te & (nb >= 50)).sum() / nte)
    for k in (1, 2):
        q = out["multi-build"][f"k={k}"]["qualifying drafts"]["test"]
        out[f"T3 gate per set (k={k}, >= {GATE_SET} test drafts)"] = bool(q >= GATE_SET)
    nc = n_colours(g["opp_colors"].to_list()) if "opp_colors" in g.columns else None
    turns = g["num_turns"].cast(pl.Float64, strict=False).to_numpy() if "num_turns" in g.columns else None
    if nc is not None:
        tab = {}
        for tag, m in (("all games", np.ones(len(y), bool)), ("test games", te)):
            rows = {}
            for c in range(6):
                mm = m & (nc == c)
                if not mm.any():
                    continue
                t = turns[mm] if turns is not None else np.array([np.nan])
                t = t[np.isfinite(t)]
                rows["unknown (0)" if c == 0 else str(c)] = {
                    "games": int(mm.sum()), "share": float(mm.sum() / m.sum()), "user win rate": float(y[mm].mean()),
                    "mean num_turns": float(t.mean()) if len(t) else None,
                    "median num_turns": float(np.median(t)) if len(t) else None}
            tab[tag] = rows
        out["num_turns by revealed opponent colours"] = tab
        known = (nc > 0) & np.isfinite(turns) if turns is not None else None
        if known is not None and known.sum() > 10:
            out["spearman(num_turns, opp colours), games with >= 1 colour"] = float(
                __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(turns[known], nc[known]).statistic)
    return out


def main(*sets):
    from mtgdata.games import game_path
    for s in sets:
        t0 = time.time()
        draft_id, split, src = draft_order(s)
        path = game_path(s, "PremierDraft")
        lf = pl.scan_parquet(path)
        have = set(lf.collect_schema().names())
        g = lf.select([c for c in COLS if c in have]).collect(engine="streaming")
        res = {"set": s, "draft order": src, **counts(draft_id, split, g), "seconds": time.time() - t0}
        out = REPO / "results" / f"{s}.PremierDraft" / "t0_counts.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
        mb = res["multi-build"]
        print(f"{s}: test drafts with games {res['drafts with games']['test']}; qualifying test drafts k=1/2/3 "
              + "/".join(str(mb[f'k={k}']['qualifying drafts']['test']) for k in KS)
              + f"; experienced share {res['test game shares']['experienced (n_bucket >= 50)']:.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        for c, r in res.get("num_turns by revealed opponent colours", {}).get("all games", {}).items():
            print(f"    opp colours {c:11s} share {r['share']:.3f}  turns {r['mean num_turns']}  user WR "
                  f"{r['user win rate']:.3f}")
        assert_no_torch()


def meta(*sets):
    tot = {k: 0 for k in KS}
    rows = {}
    for s in sets:
        f = REPO / "results" / f"{s}.PremierDraft" / "t0_counts.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows[s] = {k: r["multi-build"][f"k={k}"]["qualifying drafts"]["test"] for k in KS}
        for k in KS:
            tot[k] += rows[s][k]
    for s, r in rows.items():
        print(f"  {s}: " + "  ".join(f"k={k} {r[k]}" for k in KS))
    print("pooled qualifying test drafts: " + "  ".join(f"k={k} {tot[k]}" for k in KS))
    for k in (1, 2):
        ok_sets = sum(r[k] >= GATE_SET for r in rows.values())
        print(f"T3 gate (k={k}): {ok_sets}/{len(rows)} sets >= {GATE_SET}; pooled {tot[k]} >= {GATE_POOLED}: "
              f"{tot[k] >= GATE_POOLED} -> {'RUN' if tot[k] >= GATE_POOLED or ok_sets == len(rows) else 'KILL'}")
    return tot


def selftest():
    t0 = time.time()
    rng = np.random.default_rng(0)
    rows = []
    D = 3000
    for i in range(D):
        nb = rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        n = rng.integers(0, 10) if i % 9 else 0
        for m in range(n):
            b = (m * nb) // max(n, 1)
            turns = rng.integers(5, 15)
            colours = "".join(rng.choice(list("WUBRG"), min(5, max(0, turns // 3 - 1)), replace=False))
            rows.append({"draft_id": f"d{i}", "build_index": str(b), "match_number": m + 1,
                         "won": str(bool(rng.random() < 0.5)), "num_turns": int(turns), "opp_colors": colours or None,
                         "user_n_games_bucket": int(rng.choice([1, 50, 500]))})
    g = pl.DataFrame(rows)
    draft_id = np.array([f"d{i}" for i in range(D)] + ["nogames"])
    split = np.repeat([0, 1, 2], [2100, 450, 451]).astype(np.int8)
    r = counts(draft_id, split, g)
    # brute force: drafts with >= 2 builds of >= 2 games
    bc = g.group_by("draft_id", "build_index").len()
    q2 = bc.filter(pl.col("len") >= 2).group_by("draft_id").len().filter(pl.col("len") >= 2)["draft_id"].to_list()
    te = set(draft_id[split == 2].tolist())
    assert r["multi-build"]["k=2"]["qualifying drafts"]["test"] == len(set(q2) & te)
    tab = r["num_turns by revealed opponent colours"]["all games"]
    turns = [tab[k]["mean num_turns"] for k in tab if k != "unknown (0)"]
    assert all(np.diff(turns) > 0), turns
    assert abs(sum(v["share"] for v in tab.values()) - 1) < 1e-9
    json.dumps(r)
    print(json.dumps({k: r[k] for k in ("multi-build", "test game shares")}, indent=1)[:900])
    assert_no_torch()
    print(f"t0 self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
