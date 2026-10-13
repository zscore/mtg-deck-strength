"""Every deck rater vs one baseline (linear game model), SC-GLL, 12 dev sets, from saved t1_rescore losses.

    python scripts/t1_common_baseline.py results/meta/t1_common_baseline_per_set.json
(writes the pooled summary beside it as t1_common_baseline.json)
"""
import json
import sys
from pathlib import Path

REPO = Path("/Users/zblanton/Documents/repos/mtg-deck-strength")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "results" / "methodology"))
import numpy as np  # noqa: E402
from eval_protocol import gll_metric, meta_analysis, paired_bootstrap  # noqa: E402
from t1_rescore import load_set  # noqa: E402

SETS = "MSH BLB DFT DSK EOE FDN FIN LCI MKM OTJ SOS TDM".split()
BASE = "linear_game_model"
MODELS = ["LightGBM", "round_4_blend", "game_FM", "refit", "blendstar_LightGBM_+_game_FM"]
KEYS = ["experienced_raw", "experienced_dl", "experienced_pre"]
out_f = Path(sys.argv[1])
per_set = json.loads(out_f.read_text()) if out_f.exists() else {}
for s in SETS:
    if s in per_set:
        continue
    _, _, _, D, _, _, blocks = load_set(s)
    z = np.load(REPO / "results" / f"{s}.PremierDraft" / "t1_rescore" / "losses.npz")
    per_set[s] = {}
    for k in KEYS:
        n = z[f"n_games|{k}"]
        units = np.flatnonzero(n > 0)
        metric = gll_metric(n[units])
        A = z[f"{BASE}|{k}"][:, units]
        for m in MODELS:
            r = paired_bootstrap(metric, A, z[f"{m}|{k}"][:, units], n_boot=500, seed=0, blocks=blocks[units], name=s)
            per_set[s].setdefault(m, {})[k] = {"delta": r.delta, "se": r.se, "se_boot": r.se_boot,
                                               "se_seed": r.se_seed, "n_units": r.n_units, "n_seeds": r.n_seeds}
    out_f.write_text(json.dumps(per_set, indent=1))
    print(s, {m: round(per_set[s][m]["experienced_dl"]["delta"], 3) for m in MODELS}, flush=True)

from eval_protocol import SetResult  # noqa: E402
summary = {}
for m in MODELS:
    for k in KEYS:
        rs = [SetResult(name=s, **per_set[s][m][k]) for s in SETS]
        mr = meta_analysis(rs)
        summary.setdefault(m, {})[k] = {"mean": mr.re_mean, "ci": list(mr.re_ci), "better": int(mr.n_positive)}
        print(f"{m:32s} {k:16s} {mr.re_mean:+.3f} [{mr.re_ci[0]:+.3f}, {mr.re_ci[1]:+.3f}] {mr.n_positive}/12")
out_f.with_name("t1_common_baseline.json").write_text(json.dumps(summary, indent=1))
