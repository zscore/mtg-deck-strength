"""Draft level: does the skill-adjusted game-level deck model add to the draft-trained LightGBM deck model?

    python scripts/deck_synergy_v3_blend.py MSH [SET ...]     (after deck_synergy_v3.py)

LightGBM on registered-deck counts + structure + rank, exactly as improve_deck.py. Each game-level model's
calibrated draft p (draft_scores.npz) is blended with it in logit space; the weight is picked on validation
drafts (binomial NLL). Scored on the test drafts that have a deck.
Writes results/{SET}.PremierDraft/deck_synergy_v3/blend.json.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from improve_deck import fit_lgbm, structure_features  # noqa: E402
from openlane.data import ROOT  # noqa: E402
from openlane.deckstrength import binomial_nll, deck_metrics, load_drafts, most_played_deck  # noqa: E402
from openlane.games import game_path  # noqa: E402

GAME_MODELS = ("linear + deck features + aff", "fm4 nonland + deck features + aff",
               "fm4 nonland + deck features + aff + card fit")


def main(*sets):
    import polars as pl
    for set_code in sets:
        out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v3"
        sc = np.load(out / "draft_scores.npz")
        d = load_drafts(set_code, "PremierDraft")
        deck, has = most_played_deck(d, pl.read_parquet(game_path(set_code, "PremierDraft")))
        assert (has == sc["has_deck"]).all()
        tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
        struct = structure_features(deck, d.names, set_code)
        p_lgb, _ = fit_lgbm(np.hstack([deck.astype(np.float32), struct, d.rank[:, None].astype(np.float32)]),
                            d, tr, va)
        z = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))  # noqa: E731
        rows = {"lgbm deck+structure": deck_metrics(p_lgb[te], d.wins[te], d.losses[te])}
        for name in GAME_MODELS:
            if name not in sc:
                continue
            pg = sc[name]
            rows[name] = deck_metrics(pg[te], d.wins[te], d.losses[te])
            blend = lambda a: expit((1 - a) * z(p_lgb) + a * z(pg))  # noqa: E731
            grid = np.linspace(0, 1, 21)
            a = grid[np.argmin([binomial_nll(blend(a)[va], d.wins[va], d.losses[va]) for a in grid])]
            rows[f"lgbm + {name} (weight {a:.2f})"] = {**deck_metrics(blend(a)[te], d.wins[te], d.losses[te]),
                                                       "weight": float(a)}
            rows[f"lgbm + {name} (50/50)"] = deck_metrics(blend(0.5)[te], d.wins[te], d.losses[te])
        (out / "blend.json").write_text(json.dumps(rows, indent=1))
        print(f"=== {set_code}", flush=True)
        for k, r in rows.items():
            print(f"  {k:70s} spearman {r['spearman']:.4f} nll {r['nll']:.5f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
