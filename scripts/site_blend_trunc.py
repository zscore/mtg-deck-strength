"""site_blend: how many LightGBM trees does the site need?   python scripts/site_blend_trunc.py SET [SET ...]

The counts-only LightGBM of site_blend (seed 0, lgbm_counts_seed0.txt) truncated to its first N trees, blended with
the site FM at the fixed alpha 0.7 (z-standardised on train drafts as in site_blend), scored on SC-GLL (contexts dl
and pre, subsets experienced and match 1) against the untruncated blend and the site FM.
Writes results/{SET}.PremierDraft/site_blend/trunc.json; `meta` pools them.
"""
import json
import sys
import time

from site_blend import REPO, out_dir  # noqa: F401  (torchfree + lightgbm first)
import lightgbm as lgb
import numpy as np

CUTS = (250, 500, 1000)
SITE_FM, FULL = "site FM", "blend @0.7, all trees"
DEV = ["MSH", "BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "OTJ", "SOS", "TDM"]
VAULT = ["ECL", "TLA", "WOE", "MOM", "HOB", "TMT"]


def run_set(s, n_boot=200, log=print):
    from eval_common import NARROW, deck_rows, load_games, main_builds, per_game
    from mtgdata.games import game_path
    from decks.deckstrength import load_drafts
    from t_common import evaluate_ctx, fit_contexts, game_blocks, split_mull
    import sc_gll
    import t1_rescore as t1
    t0 = time.time()
    d = load_drafts(s, "PremierDraft")
    path = game_path(s, "PremierDraft")
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    has, rows, main_bix = main_builds(d, g)
    D = len(d.draft_id)
    blocks = game_blocks(g, pg, D)
    del g
    deck = np.zeros((D, len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    z = np.load(out_dir(s) / "scores.npz")
    assert (z["draft_id"] == np.asarray(d.draft_id).astype(str)).all()
    m = lgb.Booster(model_file=str(out_dir(s) / "lgbm_counts_seed0.txt"))
    n = m.num_trees()
    ref = has & (d.split == 0)
    zf = t1.zstd(z["site_FM"][0].astype(np.float64), ref)
    scores = {SITE_FM: z["site_FM"][0][None].astype(np.float64)}
    full = m.predict(deck.astype(np.float32), raw_score=True)
    assert np.allclose(full[has], z["LGBM_counts"][0][has], atol=1e-4)
    scores[FULL] = (0.3 * t1.zstd(full, ref) + 0.7 * zf)[None]
    te = has & (d.split == 2)
    corr = {}
    for c in CUTS:
        if c >= n:
            continue
        p = m.predict(deck.astype(np.float32), raw_score=True, num_iteration=c)
        scores[f"blend @0.7, {c} trees"] = (0.3 * t1.zstd(p, ref) + 0.7 * zf)[None]
        corr[str(c)] = float(np.corrcoef(p[te], full[te])[0, 1])
    pairs = [(SITE_FM, k) for k in scores if k != SITE_FM] + [(FULL, k) for k in scores if "trees" in k and k != FULL]
    ctxs, _, _ = fit_contexts(pg, D, ("dl", "pre"))
    subs = sc_gll.subsets(pg, has, main_bix)
    subs = {k: subs[k] for k in ("experienced", "match 1")}
    res, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, None, n_boot, blocks, s, log=lambda x: None)
    res.update({"set": s, "trees": n, "test r with all trees": corr, "seconds": time.time() - t0})
    (out_dir(s) / "trunc.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"{s}: {n} trees, r {corr}, " + "  ".join(
        f"{k.split(', ')[-1]} {res['pairs'][f'{k} - {SITE_FM}']['experienced dl']['delta']:+.3f}"
        for k in scores if k != SITE_FM) + f" [{time.time() - t0:.0f}s]")


def meta():
    from eval_protocol import meta_analysis
    from t_common import meta_rows, one_sided_p
    for group, sets in (("dev", DEV), ("vault", VAULT), ("all", DEV + VAULT)):
        files = {s: out_dir(s) / "trunc.json" for s in sets if (out_dir(s) / "trunc.json").exists()}
        print(f"##### {group}: {len(files)} sets")
        for p in (f"{FULL} - {SITE_FM}",) + tuple(f"blend @0.7, {c} trees - {SITE_FM}" for c in CUTS) + tuple(
                f"blend @0.7, {c} trees - {FULL}" for c in CUTS):
            for key in ("experienced dl", "match 1 pre"):
                rs = meta_rows(files, p, key)
                if len(rs) < 2:
                    continue
                mm = meta_analysis(rs)
                print(f"{p:45s} {key:15s} {mm.re_mean:+.3f} [{mm.re_ci[0]:+.3f}, {mm.re_ci[1]:+.3f}] "
                      f"p1={one_sided_p(mm):.4f} {mm.n_positive}/{len(rs)}  (sets with this cut: {len(rs)})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["meta"]:
        meta()
    else:
        for s in sys.argv[1:]:
            run_set(s)
