"""Probes for the blend* puzzle: which simple card-sum partners reproduce LightGBM's gain on SC-GLL-dl?

    python scripts/blend_probe.py SET ...   |   python scripts/blend_probe.py meta SET ...

lgbm_attrib found d = z_LGBM - z_FM ~ linear in card counts, with card coefficients c ~ -FM w + deck_skill (HV ~ 0).
Each probe X is a per-deck sum over the main build's nonland cards of a per-card value built from TRAIN drafts only,
blended with the champion (seed 0) exactly as blend* (t1_rescore.blend_star: alpha on VAL SC-GLL-dl, two-half
cross-fit; partner sign fixed by its expected direction):
  - FM power      X = -sum w_j (champion card power): LightGBM compresses the FM's card powers
  skill markers   X = sum m_j, m_j = mean RAW skill bucket of train drafts whose main deck plays card j (minus the
                  overall mean): a pure player-association score with no outcome information
  dl markers      the same with the de-leaked bucket
  HV              X = sum HV_j (hand_value_v2 train hand values, shrunk): skill-clean card value
  LightGBM        the saved round 4 LightGBM seed 0 (= blend* seed 0, reference)
If "skill markers" alone reproduces a large share of blend*'s gain, SC-GLL-dl rewards player association, i.e. the
judge's de-leaked skill control is incomplete, and LightGBM's gain may be player signal. Scored: experienced dl, all dl,
and by user_n_games stratum (all dl), x1000 nats/game, paired day-block bootstrap.
torch-free, no lightgbm (only saved scores). Writes results/{SET}.PremierDraft/blend_probe/results.json.
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
import lgbm_attrib as la  # noqa: E402

CH = "champion"
PROBES = ("- FM power", "skill markers", "dl markers", "HV", "LightGBM")
STRATA = {"1-10": (0, 10), "50": (50, 50), "100": (100, 100), ">=500": (500, 1e9)}


def markers(deck, nl, skill, fit):
    ok = fit & np.isfinite(skill)
    x = deck[ok].astype(np.float64) * nl
    cnt = x.sum(0)
    m = (x.T @ skill[ok]) / np.maximum(cnt, 1)
    mu = skill[ok].mean()
    k = 200.0  # shrink rare cards to the mean
    return np.where(cnt > 0, (cnt * m + k * mu) / (cnt + k) - mu, 0.0)


def run_set(set_code, log=print):
    import polars as pl
    from mtgdata.data import card_metadata
    from sc_gll import subsets
    from t_common import dl_bucket, evaluate_ctx, fit_contexts, per_draft
    from t1_rescore import blend_star
    t0 = time.time()
    S = la.load(set_code, log)
    d, pg, D, has = S["d"], S["pg"], S["D"], S["has"]
    deck = S["deck"]
    meta = card_metadata(d.names, set_code)
    nl = (~meta["types"].str.contains("Land").to_numpy()).astype(np.float64)
    tr = has & (d.split == 0)
    w = S["cp"]["champion|w"].astype(np.float64)
    raw = per_draft(pg, D, "skill")
    _, bt = dl_bucket(pg, D)
    cv = pl.read_csv(S["res_dir"] / "hand_value_v2" / "card_values.csv")
    pos = {n: i for i, n in enumerate(cv["card"].to_list())}
    hv = np.nan_to_num(cv["train_hv_logit_shrunk"].to_numpy()[[pos[n] for n in d.names]])
    X = deck.astype(np.float64) * nl
    probes = {"- FM power": -(X @ w), "skill markers": X @ markers(deck, nl, raw, tr),
              "dl markers": X @ markers(deck, nl, bt, tr), "HV": X @ hv}
    split = np.full(D, -1, np.int8)
    split[pg["gd"]] = pg["split"]
    ctxs, _, _ = fit_contexts(pg, D, ("dl",))
    subs = subsets(pg, has, S["main_bix"])
    fit_val = subs["experienced"][0]
    F0 = S["F"][:1]
    scores, info = {CH: F0}, {}
    te = has & (d.split == 2)
    for nm in PROBES:
        P = S["L"][:1] if nm == "LightGBM" else probes[nm][None]
        bl, bi = blend_star(P, F0, pg, ctxs["dl"], fit_val, D, has, split, log=lambda m_: None)
        scores[f"blend [{nm}]"] = bl
        zl, zf = la.zstd(S["L"][0], tr), la.zstd(F0[0], tr)
        info[nm] = {"alpha": bi["alphas"][0],
                    "r(probe, d = z_LGBM - z_FM) test": float(np.corrcoef(probes[nm][te] if nm in probes else zl[te],
                                                                         (zl - zf)[te])[0, 1]),
                    "r(probe, z_FM) test": float(np.corrcoef(P[0][te], F0[0][te])[0, 1])}
        log(f"  {nm:14s} alpha* {bi['alphas'][0]:.2f}  r(probe, d) {info[nm]['r(probe, d = z_LGBM - z_FM) test']:+.3f}"
            f"  r(probe, FM) {info[nm]['r(probe, z_FM) test']:+.3f}")
    pairs = [(CH, f"blend [{nm}]") for nm in PROBES]
    nb = per_draft(pg, D, "n_bucket")
    use = {"experienced": subs["experienced"], "all": subs["all"]}
    for st, (lo, hi) in STRATA.items():
        m = ((nb >= lo) & (nb <= hi))[pg["gd"]]
        use[f"all n_games {st}"] = (subs["all"][0], subs["all"][1] & m)   # (a, b) on all val games, scored per stratum
    res, _, _ = evaluate_ctx(scores, pg, {"dl": ctxs["dl"]}, use, D, pairs, None, 300, S["blocks"], "", log)
    res.update({"probes": info, "set": set_code, "total seconds": time.time() - t0, "peak rss GB": peak_rss_gb()})
    out = S["res_dir"] / "blend_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    assert_no_torch()
    log(f"{set_code} done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")


def meta(*sets):
    from eval_protocol import meta_analysis
    R = {s: json.loads((REPO / "results" / f"{s}.PremierDraft" / "blend_probe" / "results.json").read_text())
         for s in sets if (REPO / "results" / f"{s}.PremierDraft" / "blend_probe" / "results.json").exists()}
    out = {"sets": list(R), "pairs": {}, "alphas": {nm: [r["probes"][nm]["alpha"] for r in R.values()] for nm in PROBES},
           "r(probe, d)": {nm: float(np.mean([r["probes"][nm]["r(probe, d = z_LGBM - z_FM) test"] for r in R.values()]))
                           for nm in PROBES}}
    keys = ["experienced dl", "all dl"] + [f"all n_games {st} dl" for st in STRATA]
    for nm in PROBES:
        p = f"blend [{nm}] - {CH}"
        for key in keys:
            rows = [la._rows({s: {"locate": r}}, p, key) for s, r in R.items()]
            rs = [x for rr in rows for x in rr]
            if any(x.se < 1e-6 for x in rs):
                dl_ = np.array([x.delta for x in rs])
                out["pairs"].setdefault(p, {})[key] = {"unweighted mean": float(dl_.mean()),
                                                       "se": float(dl_.std(ddof=1) / np.sqrt(len(dl_))),
                                                       "sets better": f"{int((dl_ > 0).sum())}/{len(rs)}"}
                print(f"{p:32s} {key:22s} unweighted {dl_.mean():+.4f} ± {dl_.std(ddof=1) / np.sqrt(len(dl_)):.4f}")
                continue
            m = meta_analysis(rs)
            out["pairs"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci), "p": m.re_p_two_sided,
                                                   "sets better": f"{m.n_positive}/{len(rs)}"}
            print(f"{p:32s} {key:22s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] {m.n_positive}/{len(rs)}")
    print("alphas:", out["alphas"])
    print("r(probe, d):", out["r(probe, d)"])
    (REPO / "results" / "meta" / "blend_probe.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        for s in sys.argv[1:]:
            run_set(s, log=lambda m: print(m, flush=True))
