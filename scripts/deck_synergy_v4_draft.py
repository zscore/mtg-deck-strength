"""Deck synergy, round 4 (draft level): beyond averaging the game model with LightGBM.

    python scripts/deck_synergy_v4_draft.py MSH [SET ...]     (after deck_synergy_v4.py)

Test drafts with a deck, records as W/L (binomial), as improve_deck.py:
  lgbm                         LightGBM on registered-deck counts + structure + rank (improve_deck.py)
  lgbm + deck feat + aff       the same, plus the Scryfall deck features and the affinity column as inputs
  game model                   the cross-fitted game-level FM log-odds, calibrated with rank on training drafts
  lgbm + game model, averaged  round 3's logit blend (weight picked on validation drafts)
  lgbm boosted from game ...   LightGBM with the calibrated game-model logit as its starting score (init_score),
                               so the trees only learn what the skill-adjusted game model misses. The training
                               drafts' scores are cross-fitted, so the trees don't learn to trust an in-sample score.
Writes results/{SET}.PremierDraft/deck_synergy_v4/draft.json.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve_deck import LGB_PARAMS, structure_features  # noqa: E402
from openlane.data import ROOT  # noqa: E402
from openlane.deckstrength import binomial_nll, calibrate, deck_metrics, load_drafts, most_played_deck  # noqa: E402
from openlane.games import game_path  # noqa: E402


def lgbm(X, d, tr, va, init=None):
    """LightGBM on W/L records (a positive row weighted W, a negative weighted L); `init` = starting logit per draft."""
    import lightgbm as lgb

    def ds(m, ref=None):
        y = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
        sw = np.r_[d.wins[m], d.losses[m]]
        k = sw > 0
        i = None if init is None else np.r_[init[m], init[m]][k]
        return lgb.Dataset(np.vstack([X[m], X[m]])[k], y[k], weight=sw[k], init_score=i, reference=ref)
    dtr = ds(tr)
    model = lgb.train(LGB_PARAMS, dtr, 5000, valid_sets=[ds(va, dtr)], callbacks=[lgb.early_stopping(100, verbose=False)])
    z = model.predict(X, raw_score=True) + (0 if init is None else init)
    return expit(z), model.best_iteration


def main(*sets):
    import polars as pl
    for set_code in sets:
        out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4"
        inp = np.load(out / "draft_inputs.npz")
        d = load_drafts(set_code, "PremierDraft")
        deck, has = most_played_deck(d, pl.read_parquet(game_path(set_code, "PremierDraft")))
        assert (has == inp["has_deck"]).all()
        tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
        base = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                          d.rank[:, None].astype(np.float32)])
        plus = np.hstack([base, inp["deck_features"], inp["aff"][:, None]])
        z = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))  # noqa: E731
        game = {k: calibrate(inp[k][tr], d.rank[tr], d.wins[tr], d.losses[tr], inp[k], d.rank)
                for k in ("game_logit", "game_logit_ref")}
        preds, info = {}, {}
        preds["lgbm"], info["lgbm"] = lgbm(base, d, tr, va)
        preds["lgbm + deck feat + aff"], info["lgbm + deck feat + aff"] = lgbm(plus, d, tr, va)
        preds["game model (fm4 nonland, cross-fitted)"] = game["game_logit"]
        preds["game model (linear + deck feat + aff, cross-fitted)"] = game["game_logit_ref"]
        for k, src in (("lgbm", "game_logit"), ("lgbm + deck feat + aff", "game_logit")):
            grid = np.linspace(0, 1, 21)
            bl = lambda a: expit((1 - a) * z(preds[k]) + a * z(game[src]))  # noqa: E731
            a = grid[np.argmin([binomial_nll(bl(a)[va], d.wins[va], d.losses[va]) for a in grid])]
            name = f"{k} + game model, averaged"
            preds[name], info[name] = bl(a), float(a)
        for k, X in (("lgbm", base), ("lgbm + deck feat + aff", plus)):
            for src, tag in (("game_logit", "fm4 nonland"), ("game_logit_ref", "linear + deck feat + aff")):
                name = f"{k}, boosted from game model ({tag})"
                preds[name], info[name] = lgbm(X, d, tr, va, init=z(game[src]))
        rows = {k: {**deck_metrics(p[te], d.wins[te], d.losses[te]), "info": info.get(k)} for k, p in preds.items()}
        (out / "draft.json").write_text(json.dumps(rows, indent=1))
        print(f"=== {set_code}", flush=True)
        for k, r in rows.items():
            print(f"  {k:62s} spearman {r['spearman']:.4f} nll {r['nll']:.5f}  {r['info']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
