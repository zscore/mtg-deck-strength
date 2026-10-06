"""Markdown tables from results/draftencoder_multiset/{sameset,newset}.json, next to the paper's numbers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT  # noqa: E402

OUT = ROOT / "results" / "draftencoder_multiset"
LABEL = {"heuristic": "HEURISTIC (top-23 mean in-deck WR)", "lr": "LR (paper's, order-dependent)",
         "lr+meta": "LR+Meta", "bag": "bag of text features (order-free linear)", "bag+meta": "bag + Meta",
         "bag_id": "bag of card ids (per set, can't transfer)", "de": "DraftEncoder", "de+meta": "DraftEncoder+Meta"}
# Table II of the paper (mean over 3 seeds); Fig. 3 gives ~0.10-0.11 (LR) and ~0.20-0.21 (DE) at the last pick
PAPER_NEW = {"heuristic": (0.1391, None), "lr": (0.0690, 0.0694), "lr+meta": (0.0886, 0.0898),
             "de": (0.0879, 0.1641), "de+meta": (0.1529, 0.1809)}


def main():
    s = json.loads((OUT / "sameset.json").read_text()) if (OUT / "sameset.json").exists() else {}
    print("Same set (one model on BLB+FDN+TDM, test = last 14 days), Spearman at the last pick\n")
    print("| model | BLB | FDN | TDM | mean |\n|---|---|---|---|---|")
    for k, lab in LABEL.items():
        if k in s:
            ps = s[k]["per_set"]
            print(f"| {lab} | " + " | ".join(f"{ps[x]['spearman']:.3f}" for x in ("BLB", "FDN", "TDM"))
                  + f" | **{s[k]['spearman_mean']:.3f}** |")
    n = json.loads((OUT / "newset.json").read_text()) if (OUT / "newset.json").exists() else {}
    print("\nNew set (train on 11 other sets, test FDN's last 14 days), Spearman at the last pick\n")
    print("| model | no fine-tuning | + FDN first 4 days | FDN 4 days only | paper: none / 4 days |\n|---|---|---|---|---|")
    for k, lab in LABEL.items():
        if k not in n:
            continue
        f = lambda key: f"{n[key]['spearman_mean']:.3f}" if key in n else "—"
        p = PAPER_NEW.get(k)
        ptxt = f"{p[0]:.3f} / {p[1]:.3f}" if p and p[1] else (f"{p[0]:.3f} / —" if p else "—")
        print(f"| {lab} | {f(k)} | {f(k + ':finetune4d')} | {f(k + ':scratch4d')} | {ptxt} |")


if __name__ == "__main__":
    main()
