"""Where does the LightGBM deck model's gain come from? python scripts/lgbm_ablation.py SET ..."""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (load before torch; see improve_deck.py)

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from improve_deck import fit_lgbm, structure_features  # noqa: E402
from mtgdata.games import game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import deck_metrics, load_drafts, most_played_deck  # noqa: E402

for s in sys.argv[1:]:
    d = load_drafts(s)
    deck, has = most_played_deck(d, pl.read_parquet(game_path(s)))
    tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
    dk, st = deck.astype(np.float32), structure_features(deck, d.names, s)
    rk = d.rank[:, None].astype(np.float32)
    presence = (deck > 0).astype(np.float32)
    out = {}
    for name, X in (("deck+rank", np.hstack([dk, rk])),
                    ("deck+structure+rank", np.hstack([dk, st, rk])),
                    ("presence+structure+rank", np.hstack([presence, st, rk])),
                    ("structure+rank only", np.hstack([st, rk]))):
        p, it = fit_lgbm(X, d, tr, va)
        out[name] = {**deck_metrics(p[te], d.wins[te], d.losses[te]), "iterations": it}
        print(f"{s} {name:26s} spearman {out[name]['spearman']:.4f} nll {out[name]['nll']:.5f} iters {it}", flush=True)
    (ROOT / "results" / f"{s}.PremierDraft" / "draftencoder" / "lgbm_ablation.json").write_text(json.dumps(out, indent=1))
