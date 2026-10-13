"""blend* - champion SC-GLL by user_n_games stratum (from T1's saved per-draft losses; numpy only, seconds).

    python scripts/blend_strata.py SET ...

If blend*'s gain is player signal that proxies skill the de-leaked bucket measures poorly, it should be largest where
the bucket is noisiest (few games) and shrink for players with >= 500 games (bucket precise). Deck signal predicts no
trend. Uses t1_rescore/losses.npz ("all" subset; (a, b) fit on val games of all drafts, as T1) for dl / pre / raw
contexts. Delta = -(loss_blend - loss_champion) x 1000 per test game (higher = blend* better), seeds averaged per model;
SE: draft-level ratio-estimator. Pooled across sets by inverse variance. Writes results/meta/blend_strata.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
STRATA = {"1-10": (0, 10), "50": (50, 50), "100": (100, 100), ">=500": (500, 1e9)}
BL, CH = "blendstar_LightGBM_+_champion", "champion"


def one(set_code, ctx):
    r = REPO / "results" / f"{set_code}.PremierDraft"
    z = np.load(r / "t1_rescore" / "losses.npz")
    c = np.load(r / "sc_gll" / "context.npz")
    key = f"all_{ctx}"
    lb, lc, n = z[f"{BL}|{key}"].mean(0), z[f"{CH}|{key}"].mean(0), z[f"n_games|{key}"]
    D = len(n)
    nb = np.full(D, np.nan)
    nb[c["gd"]] = c["n_bucket"]
    out = {}
    for s, (lo, hi) in STRATA.items():
        m = (n > 0) & (nb >= lo) & (nb <= hi)
        dl = -(lb[m] - lc[m])
        delta = dl.sum() / n[m].sum()
        se = np.sqrt(((dl - delta * n[m]) ** 2).sum()) / n[m].sum()
        out[s] = {"delta": 1000 * delta, "se": 1000 * se, "drafts": int(m.sum()), "games": int(n[m].sum())}
    return out


def main(sets):
    res = {"sets": sets, "per set": {}, "pooled": {}}
    for ctx in ("dl", "pre", "raw"):
        per = {s: one(s, ctx) for s in sets}
        res["per set"][ctx] = per
        for st in STRATA:
            d = np.array([per[s][st]["delta"] for s in sets])
            se = np.array([per[s][st]["se"] for s in sets])
            w = 1 / se ** 2
            res["pooled"].setdefault(ctx, {})[st] = {"delta": float((w * d).sum() / w.sum()), "se": float(np.sqrt(1 / w.sum())),
                                                    "sets better": int((d > 0).sum()),
                                                    "drafts": int(sum(per[s][st]["drafts"] for s in sets))}
        print(f"[{ctx}] " + "  ".join(f"{st}: {v['delta']:+.3f}±{v['se']:.3f} ({v['sets better']}/{len(sets)}, "
                                      f"{v['drafts']} drafts)" for st, v in res["pooled"][ctx].items()))
    (REPO / "results" / "meta" / "blend_strata.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
