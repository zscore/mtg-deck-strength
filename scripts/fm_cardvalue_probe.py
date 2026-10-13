"""Why can't one FM absorb LightGBM's card values?  python scripts/fm_cardvalue_probe.py SET ... | meta SET ...

Numpy only (torch-free shim, no lightgbm, no training beyond ridge). Uses fm_cardvalue's saved scores and lgbm_prep.
Partners blended with the champion (seed 0, or seed s mod 2) exactly as T1's blend* (t1_rescore.blend_star: alpha on
VAL SC-GLL-dl, two-half cross-fit):
  LGBM counts         counts-only LightGBM (fm_cardvalue prep, full-train fit) -- reference (lgbm_attrib: +0.10)
  LGBM counts linear  its best LINEAR-in-counts projection: ridge of the logit on deck counts (train drafts' cross-
                      fitted logits as the target), so per-card values only, no tree interactions
  FM seed 1           the champion's own seed-1 refit (pure ensembling / variance reduction of two FMs)
  FM lgbm prior 0.01  the FM whose card power sits on LightGBM's marginal card values (fm_cardvalue arm)
  FM distill 0.003    the distilled FM (fm_cardvalue arm)
Scored SC-GLL experienced / match 1 under dl and pre (t_common), pairs blend - champion.
Writes results/{SET}.PremierDraft/fm_cardvalue/probe.json; meta -> results/meta/fm_cardvalue_probe.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, assert_no_torch, evaluate_ctx, fit_contexts, torchfree  # noqa: E402

torchfree()
import numpy as np  # noqa: E402

CH = "champion"
KEYS = ("experienced dl", "experienced pre", "match 1 dl", "match 1 pre")


def run_set(s, log=print):
    from sklearn.linear_model import RidgeCV
    from sc_gll import subsets
    from t1_rescore import blend_star, load_set
    t0 = time.time()
    res_dir, draft_id, pg, D, has, main_bix, blocks = load_set(s)
    od = res_dir / "fm_cardvalue"
    z = np.load(od / "scores.npz")
    pz = np.load(od / "lgbm_prep.npz")
    assert (z["draft_id"] == draft_id).all() and (pz["draft_id"] == draft_id).all()
    split = np.full(D, -1, np.int8)
    split[pg["gd"]] = pg["split"]
    split = np.where(split < 0, z["split"], split)
    ctxs, _, _ = fit_contexts(pg, D, ("dl", "pre"))
    subs_all = subsets(pg, has, main_bix)
    subs = {k: subs_all[k] for k in ("experienced", "match 1")}
    fit_val = subs_all["experienced"][0]
    Fm = z["champion"].astype(np.float64)
    full, oof = pz["full"].astype(np.float64), pz["oof"].astype(np.float64)
    deck = pz["deck"].astype(np.float64)
    tr = has & (z["split"] == 0)
    rg = RidgeCV(alphas=(0.1, 1, 10, 100, 1000)).fit(deck[tr], oof[tr])
    lin = rg.predict(deck)
    te = has & (z["split"] == 2)
    info = {"ridge alpha": float(rg.alpha_), "r(linear, LGBM counts) test": float(np.corrcoef(lin[te], full[te])[0, 1]),
            "r(LGBM counts, champion) test": float(np.corrcoef(full[te], Fm[0][te])[0, 1]),
            "r(linear, champion) test": float(np.corrcoef(lin[te], Fm[0][te])[0, 1]),
            "r(champion seed 0, seed 1) test": float(np.corrcoef(Fm[0][te], Fm[-1][te])[0, 1])}
    partners = {"LGBM counts": full[None], "LGBM counts linear": lin[None], "FM seed 1": Fm[1:2],
                "FM lgbm prior 0.01": z["lgbm_prior_0.01"][:1].astype(np.float64),
                "FM distill 0.003": z["distill_0.003"][:1].astype(np.float64)}
    scores = {CH: Fm[:1]}
    for nm, P in partners.items():
        bl, bi = blend_star(P, Fm[:1], pg, ctxs["dl"], fit_val, D, has, split, log=lambda m: None)
        scores[f"blend [{nm}]"] = bl
        info[f"alpha [{nm}]"] = bi["alphas"][0]
    log(f"=== {s}: " + "  ".join(f"{k} {v:.3f}" for k, v in info.items()))
    pairs = [(CH, n) for n in scores if n != CH]
    res, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, None, 300, blocks, s, lambda m: None)
    for k in ("experienced dl", "match 1 dl"):
        log(f"  {k}: " + "  ".join(f"{b} {res['pairs'][f'{b} - {a}'][k]['delta']:+.3f}" for a, b in pairs))
    (od / "probe.json").write_text(json.dumps({"set": s, "info": info, **res, "seconds": time.time() - t0},
                                              indent=1, default=float))
    assert_no_torch()


def meta(*sets):
    from t_common import meta_rows
    from eval_protocol import meta_analysis
    have = {s: REPO / "results" / f"{s}.PremierDraft" / "fm_cardvalue" / "probe.json" for s in sets}
    have = {s: f for s, f in have.items() if f.exists()}
    R = {s: json.loads(f.read_text()) for s, f in have.items()}
    out = {"sets": list(R), "pairs": {}, "info": {}}
    for k in next(iter(R.values()))["info"]:
        out["info"][k] = float(np.mean([r["info"][k] for r in R.values()]))
    print({k: round(v, 3) for k, v in out["info"].items()})
    for p in sorted({p for r in R.values() for p in r["pairs"]}):
        for key in KEYS:
            rs = meta_rows(have, p, key)
            if any(r.se < 1e-6 for r in rs):
                dl = np.array([r.delta for r in rs])
                out["pairs"].setdefault(p, {})[key] = {"unweighted mean": float(dl.mean()),
                                                      "se": float(dl.std(ddof=1) / np.sqrt(len(dl))),
                                                      "sets better": f"{int((dl > 0).sum())}/{len(dl)}"}
                print(f"{p:45s} {key:16s} unweighted {dl.mean():+.4f} ± {dl.std(ddof=1) / np.sqrt(len(dl)):.4f}")
                continue
            m = meta_analysis(rs)
            out["pairs"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci), "p": m.re_p_two_sided,
                                                  "sets better": f"{m.n_positive}/{len(rs)}"}
            print(f"{p:45s} {key:16s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] {m.n_positive}/{len(rs)}")
    dst = REPO / "results" / "meta" / "fm_cardvalue_probe.json"
    dst.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        for s_ in sys.argv[1:]:
            run_set(s_, log=lambda m: print(m, flush=True))
