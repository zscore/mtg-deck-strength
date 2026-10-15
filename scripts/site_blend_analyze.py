"""Summaries of the site_blend builder study: python scripts/site_blend_analyze.py [SET ...]

Per set, from results/{SET}.PremierDraft/site_blend/builds.json (site_blend_builds.mjs) and the set's card tables:
  agreement        same deck / same main colours / spells that differ, FM build vs blend build
  model views      what each model thinks of the other's deck, in win-rate points at the site's `base`
  vs the human     built deck vs the registered deck under each model
  range            LightGBM / FM scores of built decks as percentiles of training decks (extrapolation check)
  swapped cards    cards the blend adds vs drops: hand value (skill-free), GIH WR, the drafters' skill (skill marker)
  timing, sizes
Writes results/meta/site_blend_builds.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
DEV = ["MSH", "BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "OTJ", "SOS", "TDM"]
VAULT = ["ECL", "TLA", "WOE", "MOM", "HOB", "TMT"]
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


def card_table(s):
    f = REPO / "results" / f"{s}.PremierDraft" / "hand_value_v2" / "card_values.csv"
    if not f.exists():
        return {}
    num = lambda v: float(v) if v not in ("", "NaN", "nan") else np.nan  # noqa: E731
    with f.open() as fh:
        return {r["card"]: {"hv": num(r["train_hv_logit_shrunk"]), "gih": num(r["gih_wr"]), "skill": num(r["deck_skill"]),
                            "land": r["is_land"] == "true"} for r in csv.DictReader(fh)}


def analyze(s):
    d = REPO / "results" / f"{s}.PremierDraft" / "site_blend"
    if not (d / "builds.json").exists():
        return None
    b = json.loads((d / "builds.json").read_text())
    B = json.loads((d / "builder.json").read_text())
    z = np.load(d / "scores.npz")
    tr = z["has"] & (z["split"] == 0)
    train_lg, train_fm = np.sort(z["LGBM_counts"][0][tr]), np.sort(z["site_FM"][0][tr])
    pct = lambda v, ref: float(np.searchsorted(ref, v) / len(ref))  # noqa: E731
    wr = lambda x: 100 * expit(b["base"] + x)  # noqa: E731
    cards = card_table(s)
    skill_of = dict(zip(B["names"], B["card"]["skill"]))
    lands = {n for n, c in cards.items() if c["land"]} | BASICS
    P = b["pools"]
    spells = lambda dk: {k: v for k, v in dk.items() if k not in lands}  # noqa: E731
    out = {"set": s, "pools": len(P), "timing": b["timing"], "sizes": b["sizes"], "alpha": b["cfg"]["alpha"]}
    same = [p["fmDeck"] == p["blDeck"] for p in P]
    same_sp = [spells(p["fmDeck"]) == spells(p["blDeck"]) for p in P]

    def diff(a, c):  # copies in a not in c
        return sum(max(0, n - c.get(k, 0)) for k, n in a.items())

    nd = [diff(spells(p["blDeck"]), spells(p["fmDeck"])) for p in P]
    nd2 = [diff(spells(p["fm2Deck"]), spells(p["fmDeck"])) for p in P]
    best_fm = [max(p["fmBuilt"]["fm"], p["fm2Built"]["fm"]) for p in P]
    out["agreement"] = {"same deck": float(np.mean(same)), "same spells": float(np.mean(same_sp)),
                        "spells swapped (mean, all pools)": float(np.mean(nd)),
                        "spells swapped (mean, pools that differ)": float(np.mean([x for x in nd if x])) if any(nd) else 0,
                        "swaps hist": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(nd, 8), return_counts=True))},
                        "land count differs": float(np.mean([p["nLands"][0] != p["nLands"][1] for p in P])),
                        "FM vs FM other seed: same deck": float(np.mean([p["fmDeck"] == p["fm2Deck"] for p in P])),
                        "FM vs FM other seed: spells swapped": float(np.mean(nd2)),
                        "FM other seed scores worse under FM (pp)": float(np.mean(
                            [wr(p["fmBuilt"]["fm"]) - wr(p["fm2Built"]["fm"]) for p in P])),
                        "blend build beats both FM builds under FM": float(np.mean(
                            [p["blBuilt"]["fm"] > bf + 1e-9 for p, bf in zip(P, best_fm)]))}
    # what each model thinks of the two builds (win-rate points at the site's base; the blend is on the FM's scale)
    fm_loss = [wr(p["blBuilt"]["fm"]) - wr(p["fmBuilt"]["fm"]) for p in P]
    bl_gain = [wr(p["blBuilt"]["blend"]) - wr(p["fmBuilt"]["blend"]) for p in P]
    out["model views (pp)"] = {"FM: blend build - FM build": float(np.mean(fm_loss)),
                               "blend: blend build - FM build": float(np.mean(bl_gain)),
                               "blend gain, pools that differ": float(np.mean([g for g, x in zip(bl_gain, same) if not x]))
                               if not all(same) else 0,
                               "FM loss, pools that differ": float(np.mean([g for g, x in zip(fm_loss, same) if not x]))
                               if not all(same) else 0}
    out["vs registered deck (pp)"] = {
        "FM: FM build - registered": float(np.mean([wr(p["fmBuilt"]["fm"]) - wr(p["reg"]["fm"]) for p in P])),
        "blend: blend build - registered": float(np.mean([wr(p["blBuilt"]["blend"]) - wr(p["reg"]["blend"]) for p in P])),
        "blend: FM build - registered": float(np.mean([wr(p["fmBuilt"]["blend"]) - wr(p["reg"]["blend"]) for p in P])),
        "FM: blend build - registered": float(np.mean([wr(p["blBuilt"]["fm"]) - wr(p["reg"]["fm"]) for p in P]))}
    out["percentile among training decks"] = {
        f"{m} of {k}": float(np.mean([pct(p[k][m], ref) for p in P]))
        for k in ("fmBuilt", "blBuilt", "reg") for m, ref in (("lg", train_lg), ("fm", train_fm))}
    out["share above training 99.5%"] = {
        f"{m} of {k}": float(np.mean([pct(p[k][m], ref) > 0.995 for p in P]))
        for k in ("fmBuilt", "blBuilt") for m, ref in (("lg", train_lg), ("fm", train_fm))}
    # overlap with what skilled drafters registered (Jaccard on spell copies)
    def jac(a, c):
        u = sum(max(a.get(k, 0), c.get(k, 0)) for k in set(a) | set(c))
        return sum(min(a.get(k, 0), c.get(k, 0)) for k in set(a) & set(c)) / max(u, 1)
    for lo, nm in ((0.0, "all"), (0.60, "skill >= 0.60")):
        sel = [p for p in P if p["skill"] is not None and p["skill"] >= lo] if lo else P
        out[f"overlap with registered spells ({nm}, n={len(sel)})"] = {
            "FM build": float(np.mean([jac(spells(p["fmDeck"]), spells(p["regDeck"])) for p in sel])),
            "blend build": float(np.mean([jac(spells(p["blDeck"]), spells(p["regDeck"])) for p in sel]))}
    # swapped cards
    add, drop = [], []
    for p in P:
        a, c = spells(p["blDeck"]), spells(p["fmDeck"])
        for k in set(a) | set(c):
            n = a.get(k, 0) - c.get(k, 0)
            (add if n > 0 else drop).extend([k] * abs(n))
    sk_all = B["card"]["skill all"]

    def prof(lst):
        f = lambda key: [cards[k][key] for k in lst if k in cards and np.isfinite(cards[k][key])]  # noqa: E731
        sk = [skill_of[k] - sk_all for k in lst if skill_of.get(k) is not None]
        return {"n": len(lst), "hand value (logit)": float(np.mean(f("hv"))) if f("hv") else None,
                "GIH WR": float(np.mean(f("gih"))) if f("gih") else None,
                "drafter skill - mean": float(np.mean(sk)) if sk else None}
    out["swapped cards"] = {"blend adds": prof(add), "blend drops": prof(drop)}
    from collections import Counter
    out["most added"] = Counter(add).most_common(8)
    out["most dropped"] = Counter(drop).most_common(8)
    # card-value labels: how different are the per-card numbers the UI shows?
    vs = [p for p in P if p.get("values")]
    if vs:
        r = []
        for p in vs:
            ks = [k for k in p["values"]["fm"] if p["values"]["fm"][k] is not None and p["values"]["bl"].get(k) is not None]
            if len(ks) > 5:
                r.append(np.corrcoef([p["values"]["fm"][k] for k in ks], [p["values"]["bl"][k] for k in ks])[0, 1])
        out["card value label r (FM vs blend, own builds)"] = float(np.nanmean(r))
    return out


def pooled(rows):
    ks = ["same deck", "same spells", "spells swapped (mean, all pools)", "spells swapped (mean, pools that differ)",
          "land count differs", "FM vs FM other seed: same deck", "FM vs FM other seed: spells swapped",
          "FM other seed scores worse under FM (pp)", "blend build beats both FM builds under FM"]
    out = {k: float(np.mean([r["agreement"][k] for r in rows])) for k in ks}
    for sec in ("model views (pp)", "vs registered deck (pp)", "percentile among training decks",
                "share above training 99.5%"):
        out[sec] = {k: float(np.mean([r[sec][k] for r in rows])) for k in rows[0][sec]}
    for side in ("blend adds", "blend drops"):
        for k in ("hand value (logit)", "GIH WR", "drafter skill - mean"):
            v = [r["swapped cards"][side][k] for r in rows if r["swapped cards"][side][k] is not None]
            out.setdefault("swapped cards", {}).setdefault(side, {})[k] = float(np.mean(v)) if v else None
    for k in ("fm build ms", "blend build ms", "fm values ms", "blend values ms"):
        out.setdefault("timing", {})[k] = float(np.mean([r["timing"][k] for r in rows]))
    for k in ("trees", "f16 int16-kids kB", "f32 gz kB", "site deck json kB", "site deck json gz kB"):
        out.setdefault("sizes", {})[k] = [float(np.min([r["sizes"][k] for r in rows])),
                                          float(np.mean([r["sizes"][k] for r in rows])),
                                          float(np.max([r["sizes"][k] for r in rows]))]
    return out


if __name__ == "__main__":
    sets = sys.argv[1:] or DEV + VAULT
    rows = [r for r in (analyze(s) for s in sets) if r]
    res = {"per set": {r["set"]: r for r in rows}, "pooled": pooled(rows) if rows else None,
           "pooled dev": pooled([r for r in rows if r["set"] in DEV]) if any(r["set"] in DEV for r in rows) else None}
    (REPO / "results" / "meta" / "site_blend_builds.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["pooled"], indent=1, default=float))
    for r in rows:
        a = r["agreement"]
        print(f"{r['set']}: same {a['same deck']:.2f}, swaps {a['spells swapped (mean, all pools)']:.2f}, "
              f"FM view {r['model views (pp)']['FM: blend build - FM build']:+.2f} pp, blend view "
              f"{r['model views (pp)']['blend: blend build - FM build']:+.2f} pp; adds {r['swapped cards']['blend adds']}; "
              f"drops {r['swapped cards']['blend drops']}")
