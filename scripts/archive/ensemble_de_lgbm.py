"""Do DraftEncoder and the LightGBM deck model make different errors? python scripts/ensemble_de_lgbm.py SET ..."""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (load before torch; see improve_deck.py)

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from improve_deck import fit_draft_model, fit_lgbm, structure_features  # noqa: E402
from mtgdata.games import game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS, deck_metrics, load_drafts, most_played_deck  # noqa: E402

for s in sys.argv[1:]:
    out_dir = ROOT / "results" / f"{s}.PremierDraft" / "draftencoder"
    d = load_drafts(s)
    deck, has = most_played_deck(d, pl.read_parquet(game_path(s)))
    tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
    dk, st = deck.astype(np.float32), structure_features(deck, d.names, s)
    ranks = np.eye(N_RANKS, dtype=np.float32)[d.rank]
    p = {"linear+structure": fit_draft_model([dk, ranks, st], d, tr, va)[0][te],
         "lgbm": fit_lgbm(np.hstack([dk, st, d.rank[:, None].astype(np.float32)]), d, tr, va)[0][te]}
    for run in ("paper+deck", "big+deck"):
        f = out_dir / f"test_pred_{run}.npy"
        if f.exists():
            p[f"de_{run}"] = np.load(f)[-1]
    lg = {k: logit(np.clip(v, 1e-6, 1 - 1e-6)) for k, v in p.items()}
    combos = [[k] for k in p] + [["lgbm", k] for k in p if k.startswith("de_")] + \
             [["lgbm", "linear+structure", k] for k in p if k.startswith("de_")]
    res = {}
    for ks in combos:
        m = deck_metrics(expit(np.mean([lg[k] for k in ks], 0)), d.wins[te], d.losses[te])
        res[" + ".join(ks)] = m
        print(f"{s} {' + '.join(ks):42s} spearman {m['spearman']:.4f} nll {m['nll']:.5f}", flush=True)
    for k in p:
        if k.startswith("de_"):
            print(f"{s} corr(lgbm, {k}) = {np.corrcoef(lg['lgbm'], lg[k])[0, 1]:.3f}")
    (out_dir / "ensemble_de_lgbm.json").write_text(json.dumps(res, indent=1))
