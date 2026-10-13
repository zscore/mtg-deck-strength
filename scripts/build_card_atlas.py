"""Card power and synergy page: python scripts/build_card_atlas.py

Card power is the opening-hand value (hand_value_v2/card_values.csv): the change in game win rate from having the
card in the opening 7, against an average card, skill-free by construction. Synergy is round 8's opening-hand pairs
(deck_synergy_v8/hand_pairs.csv, slope + pairs fit), so card draw can't drive it; individual pairs are noisy
(split-half r ~0.13). Card colour and rarity come from mtg-data's cards.csv. Writes reports/card_atlas.html from
reports/card_atlas_template.html.
"""
import json
import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decks import ROOT  # noqa: E402

SETS = ("BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "MSH", "OTJ", "SOS", "TDM")
MIN_GAMES = 2000          # games with the card in the opening hand, for a card to be shown
N_PAIRS, N_ANTI, N_HUBS = 15, 8, 8
CARDS_CSV = ROOT.parent / "mtg-data" / "data" / "raw" / "cards.csv"


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def card_meta(set_code):
    """name -> (rarity letter, colour identity), preferring the set's own printing."""
    c = pl.read_csv(CARDS_CSV, columns=["expansion", "name", "rarity", "color_identity"], infer_schema_length=0)
    c = c.with_columns((pl.col("expansion") != set_code).alias("other")).sort("other")
    c = c.unique("name", keep="first")
    return {r["name"]: ((r["rarity"] or "?")[0].upper(), r["color_identity"] or "")
            for r in c.iter_rows(named=True)}


def one_set(set_code, meta):
    base = ROOT / "results" / f"{set_code}.PremierDraft"
    hv = pl.read_csv(base / "hand_value_v2" / "card_values.csv")
    hv = hv.filter(~pl.col("is_land") & (pl.col("train_n_games") >= MIN_GAMES)
                   & pl.col("train_hv_pp").is_not_nan())
    cards = []
    for r in hv.sort("train_hv_pp", descending=True).iter_rows(named=True):
        rar, col = meta.get(r["card"], ("?", ""))
        cards.append({"name": r["card"], "hv": round(100 * r["train_hv_pp"], 2),
                      "se": round(100 * r["train_se_pp_cluster"], 2), "games": int(r["train_n_games"]),
                      "gih": round(100 * r["gih_wr"], 1) if finite(r["gih_wr"]) else None,
                      "rarity": rar, "colors": col})

    syn = base / "deck_synergy_v8"
    pairs = pl.read_csv(syn / "hand_pairs.csv").sort("Q", descending=True)

    def rows(df):
        return [{"a": r["card a"], "b": r["card b"], "q": round(r["Q"], 3), "decks": int(r["train decks"])}
                for r in df.iter_rows(named=True)]
    hubs = pl.read_csv(syn / "hand_hubs.csv").sort("partner spread", descending=True).head(N_HUBS)
    hub_rows = [{"card": r["card"], "spread": round(r["partner spread"], 3), "best": r["best partner"],
                 "bestQ": round(r["best Q"], 3), "worst": r["worst partner"], "worstQ": round(r["worst Q"], 3)}
                for r in hubs.iter_rows(named=True)]
    fit = next(m for m in json.loads((syn / "results.json").read_text())["models"]
               if m["input"] == "hand" and m["model"] == "slope + pairs k=4")
    return {"cards": cards, "pairs": rows(pairs.head(N_PAIRS)), "anti": rows(pairs.tail(N_ANTI).reverse()),
            "hubs": hub_rows, "nPairs": pairs.height,
            "pairGain": round(-1000 * fit["vs slope"], 2), "pairGainCI": round(1000 * fit["vs slope ±95%"], 2),
            "splitHalfR": round(fit["split-half pair r"], 2)}


def main():
    data = {}
    for s in SETS:
        data[s] = one_set(s, card_meta(s))
        print(s, len(data[s]["cards"]), "cards", data[s]["nPairs"], "pairs")
    tpl = (ROOT / "reports" / "card_atlas_template.html").read_text()
    out = ROOT / "reports" / "card_atlas.html"
    out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print("wrote", out, f"{out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
