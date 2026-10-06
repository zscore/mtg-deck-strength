"""Paired bootstrap of the sameset results: are the differences between models bigger than test noise?

    python scripts/de_multiset_bootstrap.py [n_boot]

Reloads the saved sameset models (results/draftencoder_multiset/models/sameset_*.pt), predicts the test
drafts of BLB, FDN and TDM, and resamples test drafts within each set (the same resample for every model).
Statistic: mean over the three sets of Spearman at the last pick, as in the tables.
Writes results/draftencoder_multiset/sameset_bootstrap.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openlane.de_multiset import SetData, Task, card_meta, predict  # noqa: E402
from train_draftencoder_multiset import OUT, SAMESET, build_model, feats, finished, text_norm  # noqa: E402

MODELS = ["lr", "lr+meta", "bag", "bag+meta", "bag_id", "de", "de+meta"]


def fast_spearman(p, y):
    a, b = rankdata(p), rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main(n_boot="1000"):
    n_boot = int(n_boot)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sets = {s: SetData(s) for s in SAMESET}
    mu, sd = text_norm(list(sets.values()))
    test = {s: finished(d, d.window(14, 0)) for s, d in sets.items()}
    meta = {s: card_meta(d, finished(d, d.window(42, 28)))[0] for s, d in sets.items()}
    offs = np.cumsum([0] + [d.C for d in sets.values()])
    id_off = dict(zip(sets, offs[:-1]))
    cfg = {"d": 64, "set_layers": 10, "draft_layers": 5, "dropout": 0.3}
    preds = {}
    for m in MODELS:
        kind, use_meta = m.split("+")[0], m.endswith("+meta")
        tasks = [Task(sets[s], feats(sets[s], mu, sd, meta[s] if use_meta else None), test[s], int(id_off[s]))
                 for s in SAMESET]
        model = build_model(kind, tasks[0].feats.shape[1], int(offs[-1]), cfg)
        model.load_state_dict(torch.load(OUT / "models" / f"sameset_{m}.pt", map_location="cpu"))
        model.to(device)
        preds[m] = [p[:, -1] for p in predict(model, tasks, device)]
        print(m, "predicted", flush=True)
    y = {s: sets[s].wins[test[s]] / (sets[s].wins[test[s]] + sets[s].losses[test[s]]) for s in SAMESET}
    rng = np.random.default_rng(0)
    boot = {m: [] for m in MODELS}
    for _ in range(n_boot):
        idx = {s: rng.integers(0, len(test[s]), len(test[s])) for s in SAMESET}
        for m in MODELS:
            boot[m].append(np.mean([fast_spearman(preds[m][k][idx[s]], y[s][idx[s]]) for k, s in enumerate(SAMESET)]))
    boot = {m: np.array(v) for m, v in boot.items()}
    point = {m: float(np.mean([fast_spearman(preds[m][k], y[s]) for k, s in enumerate(SAMESET)])) for m in MODELS}
    pairs = [("de", "bag"), ("de+meta", "bag+meta"), ("de+meta", "bag_id"), ("bag+meta", "bag"), ("de+meta", "de"),
             ("bag", "lr"), ("bag_id", "bag+meta")]
    out = {"point": point, "ci95": {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                    for m, v in boot.items()}, "diffs": {}}
    for a, b in pairs:
        d = boot[a] - boot[b]
        out["diffs"][f"{a} - {b}"] = {"diff": point[a] - point[b], "ci95": [float(np.percentile(d, 2.5)),
                                      float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}
        print(f"{a:>9s} - {b:<9s} {point[a] - point[b]:+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, "
              f"{np.percentile(d, 97.5):+.4f}]", flush=True)
    (OUT / "sameset_bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
