"""T3b: the within-build referee (t3_within_build's conditional logit) for blend*, the SC-GLL-tuned LightGBM + FM blend.

    python scripts/t3b_blend.py SET [SET ...]
    python scripts/t3b_blend.py meta SET ...

Question: blend* beats the game FM by +0.13 (dev) / +0.23 (vault) on SC-GLL-dl. Is that deck signal (then blend*'s
between-build differences predict within-player outcomes at least as well as the FM's) or player signal (a
draft-constant component, which the conditional logit cancels, so within-build b would not improve)?
Method = t3_within_build exactly (analyse / pair_results / the same informative drafts, record-state control primary,
match-number control as sensitivity, draft-cluster sandwich SEs + seed spread). Per-build scores, no refitting:
  champion   t2_mull_split/build_scores.npz (2 seeds)
  LightGBM   t3_within_build/scores.npz (round 4 recipe, platinum reference rank, the same 3 seeds as T3)
  blend*     per LightGBM seed s: (1 - a_s) z_LGBM(build) + a_s z_champion[s mod 2](build), a_s = T1's saved alpha*
             (t1_rescore/results.json, "blend* (LightGBM + champion)", tuned on VAL SC-GLL-dl); z standardised with
             the mean / sd of each score over TRAIN drafts' main builds (T1's z, which used the draft-level copies)
Pairs (b per within-draft sd, B - A): blend* - champion, blend* - LightGBM, LightGBM - champion (reference).
Leakage veto in meta: one-sided p that blend* lowers b relative to the champion (small p = veto).
Writes results/{SET}.PremierDraft/t3b_blend/results.json; meta -> results/meta/t3b_blend.json.
numpy only (torch-free deckstrength shim; no lightgbm, no torch).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, assert_no_torch, peak_rss_gb, torchfree  # noqa: E402

torchfree()
import numpy as np  # noqa: E402
import t3_within_build as t3  # noqa: E402

CH, LG, BL = "champion", "LightGBM", "blend*"
T1_BLEND = "blend* (LightGBM + champion)"
PAIRS = [(CH, BL), (LG, BL), (CH, LG)]


def blend_builds(L, F, alphas, ref_builds):
    """L [S, B], F [SF, B] per-build logit scores; alphas [S]. z on ref_builds (train main builds)."""
    z = lambda s: (s - s[ref_builds].mean()) / (s[ref_builds].std() + 1e-12)  # noqa: E731
    return np.stack([(1 - a) * z(L[s]) + a * z(F[s % len(F)]) for s, a in enumerate(alphas)])


def run_set(set_code, log=print):
    import polars as pl
    from eval_common import load_games, per_game
    from mtgdata.games import game_path
    from t_common import split_mull
    from decks.deckstrength import load_drafts
    t0 = time.time()
    res_dir = REPO / "results" / f"{set_code}.PremierDraft"
    bz = np.load(res_dir / "t2_mull_split" / "build_scores.npz")
    lz = np.load(res_dir / "t3_within_build" / "scores.npz")
    assert (lz["build_draft_id"] == bz["build_draft_id"]).all() and (lz["build_index"] == bz["build_index"]).all()
    t1 = json.loads((res_dir / "t1_rescore" / "results.json").read_text())
    alphas = [float(a) for a in t1["blend*"][T1_BLEND]["alphas"]]
    assert t1["blend*"][T1_BLEND]["FM member"] == CH
    d = load_drafts(set_code, "PremierDraft")
    g = load_games(game_path(set_code, "PremierDraft"), d.draft_id)
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    key = {k: j for j, k in enumerate(zip(bz["build_draft_id"].tolist(), bz["build_index"].tolist()))}
    game_build = np.array([key[k] for k in zip(g["draft_id"].to_list(), g["build_index"].cast(pl.Utf8).to_list())])
    del g
    D = len(d.draft_id)
    main_build, build_draft = bz["main_build"], bz["build_draft"]
    has = main_build >= 0
    assert (np.asarray(d.draft_id).astype(str)[build_draft] == bz["build_draft_id"]).all()
    F = bz["champion"].astype(np.float64)
    L = lz["LightGBM"].astype(np.float64)
    assert len(alphas) == len(L), (alphas, L.shape)
    ref_b = main_build[has & (d.split == 0)]
    models = {CH: F, LG: L, BL: blend_builds(L, F, alphas, ref_b)}
    # check against T1's draft-level blend* (main builds; T3's LightGBM is a refit of sc_gll's: r ~ 0.9997)
    bl1 = np.load(res_dir / "t1_rescore" / "blend.npz")["blendstar_LightGBM_+_champion"]
    te = np.flatnonzero(has & (d.split == 2))
    r_t1 = [float(np.corrcoef(models[BL][s][main_build[te]], bl1[s][te])[0, 1]) for s in range(len(L))]
    te_main = main_build[te]
    sd_ref = {m: float(np.mean([np.std(s[te_main]) for s in S])) for m, S in models.items()}
    log(f"=== {set_code}: {D} drafts, {len(pg['y'])} games, {len(build_draft)} builds; alpha* {alphas}; "
        f"r(blend* per-build at main, T1 blend*) {np.round(r_t1, 4).tolist()} [{time.time() - t0:.0f}s]")
    res, IFs = t3.analyse(pg, game_build, build_draft, D, d.split, models, sd_ref, 1, (2,), "state", log=log)
    t3.pair_results(res, IFs, PAIRS, log=log)
    log("  sensitivity: match-number control")
    r2, IF2 = t3.analyse(pg, game_build, build_draft, D, d.split, models, sd_ref, 1, (2,), "match", log=log)
    t3.pair_results(r2, IF2, PAIRS, key="within-build b (match control)", log=log)
    res["models (match control)"] = r2["models"]
    for p_, v in r2["pairs"].items():
        res["pairs"].setdefault(p_, {}).update(v)
    # how different are the builds under each score, and how much of blend*'s within-draft variation is LightGBM's
    j = np.flatnonzero(np.isin(build_draft, np.flatnonzero(d.split == 2)))
    dev = {m: [] for m in models}
    for m, S in models.items():
        for s in S:
            mu = np.bincount(build_draft[j], s[j], D) / np.maximum(np.bincount(build_draft[j], minlength=D), 1)
            dev[m].append(s[j] - mu[build_draft[j]])
    zdev = {m: np.mean(v, 0) for m, v in dev.items()}
    res["within-draft build-score correlation (test, all builds)"] = {
        f"{a} vs {b}": float(np.corrcoef(zdev[a], zdev[b])[0, 1]) for a, b in ((CH, LG), (CH, BL), (LG, BL))}
    res.update({"set": set_code, "alphas": alphas, "r(per-build blend* at main build, T1 blend*)": r_t1,
                "total seconds": time.time() - t0, "peak rss GB": peak_rss_gb()})
    out = res_dir / "t3b_blend"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    assert_no_torch()
    log(f"{set_code} done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")
    return res


def meta(*sets):
    from eval_protocol import DecisionRule, decide, meta_analysis
    from t_common import meta_rows, one_sided_p
    files = {s: REPO / "results" / f"{s}.PremierDraft" / "t3b_blend" / "results.json" for s in sets}
    files = {s: f for s, f in files.items() if f.exists()}
    R = {s: json.loads(f.read_text()) for s, f in files.items()}
    out = {"sets": list(R), "pooled qualifying drafts": sum(r["qualifying drafts"] for r in R.values()),
           "models": {}, "pairs": {}, "alphas (seed 0)": {s: r["alphas"][0] for s, r in R.items()}}
    for ctl, mk in (("state", "models"), ("match", "models (match control)")):
        for m in (CH, LG, BL):
            b = np.array([r[mk][m]["b per within-sd"] for r in R.values()])
            se = np.array([np.hypot(r[mk][m]["se per within-sd"], r[mk][m]["seed se"]) for r in R.values()])
            w = 1 / se ** 2
            out["models"].setdefault(m, {})[ctl] = {"b per within-sd FE": float((w * b).sum() / w.sum()),
                                                    "se": float(np.sqrt(1 / w.sum())), "per set": b.tolist()}
            print(f"  [{ctl}] {m:10s} pooled b per within-sd {out['models'][m][ctl]['b per within-sd FE']:+.4f} ± "
                  f"{out['models'][m][ctl]['se']:.4f}")
    for p in (f"{BL} - {CH}", f"{BL} - {LG}", f"{LG} - {CH}"):
        for key in ("within-build b", "within-build b (match control)"):
            rs = meta_rows(files, p, key)
            if len(rs) < 2:
                continue
            mm = meta_analysis(rs)
            ok, why = decide(mm, DecisionRule(min_positive=min(8, len(rs))))
            out["pairs"].setdefault(p, {})[key] = {
                "re mean": mm.re_mean, "ci": list(mm.re_ci), "p two-sided": mm.re_p_two_sided,
                "p one-sided (B lowers b)": 1 - one_sided_p(mm), "sets better": f"{mm.n_positive}/{len(rs)}",
                "sig worse": mm.n_sig_worse, "decide": ok, "why": why, "per set": [r.delta for r in rs]}
            print(f"\n{p} [{key}]\n{mm.summary()}\n  one-sided p that B lowers b: {1 - one_sided_p(mm):.3f}")
    v = out["pairs"].get(f"{BL} - {CH}", {})
    out["leakage veto (blend* lowers b vs champion, one-sided p < 0.05)"] = {
        k: x["p one-sided (B lowers b)"] < 0.05 for k, x in v.items()}
    print("\nleakage veto:", out["leakage veto (blend* lowers b vs champion, one-sided p < 0.05)"])
    out["within-draft build-score correlation (mean over sets)"] = {
        k: float(np.mean([r["within-draft build-score correlation (test, all builds)"][k] for r in R.values()]))
        for k in next(iter(R.values()))["within-draft build-score correlation (test, all builds)"]}
    print(out["within-draft build-score correlation (mean over sets)"])
    dst = REPO / "results" / "meta" / "t3b_blend.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        for s in sys.argv[1:]:
            run_set(s, log=lambda m: print(m, flush=True))
