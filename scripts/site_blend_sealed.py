"""site_blend on Sealed games: python scripts/site_blend.py sealed SET [SET ...]

Sealed has no draft table, so "drafts" are events from the game file, split 70 / 15 / 15 by first draft_time exactly
as the site's export does (openlane event_split), and each event's W / L is its game count. Same judge as
site_blend (t_common contexts, sc_gll subsets) on these events. Arms:
  site FM sealed        deck.js 'sealed' model, clamped
  LGBM counts sealed    counts-only LightGBM fit on Sealed events (3 seeds)
  LGBM counts draft     the Premier Draft counts-only LightGBM (site_blend seed 0), applied to Sealed decks
  blend* ...            t1_rescore.blend_star with each LightGBM, and the fixed 0.7
Writes results/{SET}.PremierDraft/site_blend/sealed.json.
"""
import json
import time
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import polars as pl

from site_blend import LGB_SEEDS, N_BOOT, counts_lgbm, out_dir, site_scores

SFM, LCS, LCD = "site FM sealed", "LGBM counts sealed", "LGBM counts draft"
BS, BD, BS7, BD7 = ("blend* counts sealed + site FM", "blend* counts draft + site FM",
                    "blend counts sealed + site FM @0.7", "blend counts draft + site FM @0.7")
SDF, BDD, BDD7 = "site draft FM", "blend* counts draft + site draft FM", "blend counts draft + site draft FM @0.7"
PAIRS = [(SFM, LCS), (SFM, LCD), (SFM, BS), (SFM, BD), (SFM, BS7), (SFM, BD7), (BS, BD),
         (SFM, SDF), (SFM, BDD), (SDF, BDD), (BD, BDD), (SDF, BDD7), (SFM, BDD7)]
SPLITS = ("train", "val", "test")


def sealed_events(path):
    """d-like namespace for Sealed events: draft_id, split, wins, losses, names, mask()."""
    g = pl.scan_parquet(path)
    names = [c[len("deck_"):] for c in g.collect_schema().names() if c.startswith("deck_")]
    ev = (g.group_by("draft_id").agg(pl.col("draft_time").min(), (pl.col("won").cast(pl.Utf8) == "True").sum().alias("w"),
                                     pl.len().alias("n"))
          .sort("draft_time", "draft_id").collect())
    D = ev.height
    split = np.zeros(D, np.int8)
    split[int(0.70 * D):] = 1
    split[int(0.85 * D):] = 2
    w = ev["w"].to_numpy().astype(np.float32)
    d = SimpleNamespace(draft_id=ev["draft_id"].to_numpy().astype(str), split=split, wins=w,
                        losses=ev["n"].to_numpy().astype(np.float32) - w, names=names,
                        draft_time=ev["draft_time"].to_numpy())
    d.mask = lambda s: d.split == SPLITS.index(s)
    return d


def run_set(set_code, n_boot=N_BOOT, log=print):
    from eval_common import NARROW, deck_rows, load_games, main_builds, per_game
    from mtgdata.games import game_path
    from t_common import evaluate_ctx, fit_contexts, game_blocks, split_mull
    import sc_gll
    import t1_rescore as t1
    t0 = time.time()
    out = out_dir(set_code)
    path = game_path(set_code, "Sealed")
    d = sealed_events(path)
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    has, rows, main_bix = main_builds(d, g)
    D = len(d.draft_id)
    blocks = game_blocks(g, pg, D)
    del g
    deck = np.zeros((D, len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    log(f"=== {set_code} Sealed: {D} events, {len(pg['y'])} games, test events {int((has & (d.split == 2)).sum())} "
        f"[{time.time() - t0:.0f}s]")
    clamped, _ = site_scores(set_code, "sealed", d.names, deck[has])
    scores = {SFM: np.full((1, D), np.nan)}
    scores[SFM][0, has] = clamped
    # the site's Premier Draft model on the same Sealed decks
    clamped_d, _ = site_scores(set_code, "draft", d.names, deck[has])
    scores[SDF] = np.full((1, D), np.nan)
    scores[SDF][0, has] = clamped_d
    scores[LCS], trees = counts_lgbm(deck, d, has, log=log)
    # the Premier Draft model on Sealed decks (its features: the draft run's d.names)
    dn = json.loads((out / "builder.json").read_text())["names"]
    pos = {n: i for i, n in enumerate(d.names)}
    Xd = np.zeros((D, len(dn)), np.float32)
    cols = [(i, pos[n]) for i, n in enumerate(dn) if n in pos]
    Xd[:, [a for a, _ in cols]] = deck[:, [b for _, b in cols]]
    bst = lgb.Booster(model_file=str(out / "lgbm_counts_seed0.txt"))
    scores[LCD] = bst.predict(Xd, raw_score=True)[None]
    ctxs, cinfo, _ = fit_contexts(pg, D, ("raw", "dl", "pre"))
    subs = sc_gll.subsets(pg, has, main_bix)
    fit_val = subs["experienced"][0]
    binfo = {}
    for nm, L, F in ((BS, LCS, SFM), (BD, LCD, SFM), (BDD, LCD, SDF)):
        scores[nm], binfo[nm] = t1.blend_star(scores[L], scores[F], pg, ctxs["dl"], fit_val, D, has, d.split,
                                              log=lambda m: None)
        log(f"  {nm}: alphas {binfo[nm]['alphas']}")
    ref = has & (d.split == 0)
    zf = t1.zstd(scores[SFM][0], ref)
    scores[BS7] = np.stack([0.3 * t1.zstd(s, ref) + 0.7 * zf for s in scores[LCS]])
    scores[BD7] = 0.3 * t1.zstd(scores[LCD][0], ref)[None] + 0.7 * zf
    scores[BDD7] = 0.3 * t1.zstd(scores[LCD][0], ref)[None] + 0.7 * t1.zstd(scores[SDF][0], ref)
    res, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, PAIRS, None, n_boot, blocks, set_code, log=lambda m: None)
    for key in ("experienced dl", "all dl"):
        log(f"  {key}: " + "  ".join(f"{m} {res['models'][m][key]['sc_gll']:.3f}" for m in scores))
        for a, b in PAIRS:
            r = res["pairs"][f"{b} - {a}"][key]
            log(f"      {b} - {a}: {r['delta']:+.3f} ± {r['se']:.3f}")
    res.update({"set": set_code, "events": D, "trees": trees, "blend*": binfo, "contexts": cinfo,
                "seconds": time.time() - t0})
    (out / "sealed.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"{set_code} sealed done [{time.time() - t0:.0f}s]")
