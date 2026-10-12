"""Improving the deck-strength model: python scripts/improve_deck.py MSH [SET ...]

Starts from `linear_deck` (binomial logistic regression on registered-deck counts + rank) and tests:
  +structure  land count (one-hot 14..19), nonland curve by mana value, creature count,
              number of main colours and splash colours: build choices a card sum can't express
  +archetype  which main colours and which colour pair (archetype-level effects)
  +gih_prior  the deck's mean GIH WR as a feature, so cards with few games lean on public data
  lgbm        LightGBM on deck counts + structure + rank (can learn interactions)
  game_fm     the game-level FM from deck_synergy.py (skill-adjusted power + pair synergy)
  ensembles   logit averages of the draft-level and game-level models
All models are scored on the test drafts that have a deck (same rows as the other deck tables).
"""
if __name__ == "__main__":
    # torch and LightGBM each bundle OpenMP. In the parent, LightGBM must load before torch
    # (the other order segfaults); the torch-only child process (__mp_main__) skips it.
    import lightgbm  # noqa: F401

import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import COLORS, card_metadata, download_cards  # noqa: E402
from mtgdata.games import aligned, flag, game_win_rates, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (N_RANKS, binomial_logreg, calibrate, covariates, deck_metrics, game_split,  # noqa: E402
                                load_drafts, most_played_deck)

C_GRID = (0.003, 0.01, 0.03, 0.1)
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=300,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                  feature_pre_filter=False, verbose=-1, num_threads=4)


def fit_lgbm(X, d, fit_mask, va):
    """LightGBM on the W/L record: each draft is a positive row weighted W and a negative weighted L."""
    import lightgbm as lgb  # imported lazily so the torch child process never loads it
    def ds(m, ref=None):
        y = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
        sw = np.r_[d.wins[m], d.losses[m]]
        k = sw > 0
        return lgb.Dataset(np.vstack([X[m], X[m]])[k], y[k], weight=sw[k], reference=ref)
    dtr = ds(fit_mask)
    model = lgb.train(LGB_PARAMS, dtr, 5000, valid_sets=[ds(va, dtr)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model.predict(X), model.best_iteration


def structure_features(deck: np.ndarray, names: list[str], set_code: str,
                       archetype: bool = False) -> np.ndarray:
    meta = card_metadata(names, set_code)
    cards = pl.read_csv(download_cards(), columns=["name", "types"]).unique("name")
    types = pl.DataFrame({"name": names}).join(cards, on="name", how="left")["types"].fill_null("")
    land = types.str.contains("Land").to_numpy()
    creature = meta["is_creature"].to_numpy() & ~land
    mv = meta["mana_value"].to_numpy()
    colors = meta.select(list(COLORS)).to_numpy().astype(np.float32)
    x = deck.astype(np.float32)
    n_land = x[:, land].sum(1)
    feats = [np.eye(6)[np.clip(n_land - 14, 0, 5).astype(int)]]  # 14..19+ lands
    spells = x * ~land
    for lo, hi in ((0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 99)):  # curve
        feats.append(spells[:, (mv >= lo) & (mv <= hi)].sum(1, keepdims=True) / 5)
    feats.append(x[:, creature].sum(1, keepdims=True) / 10)
    per_color = spells @ colors  # nonland cards of each colour
    feats.append(np.eye(4)[np.clip((per_color >= 4).sum(1), 1, 4).astype(int) - 1])  # main colours
    feats.append(((per_color >= 1) & (per_color < 4)).sum(1, keepdims=True).astype(np.float32))  # splash
    if archetype:  # which main colours, and which exact pair for two-colour decks
        main = per_color >= 4
        feats.append(main.astype(np.float32))
        pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
        two = main.sum(1) == 2
        feats.append(np.column_stack([two & main[:, a] & main[:, b] for a, b in pairs]).astype(np.float32))
    return np.hstack(feats).astype(np.float32)


def fit_draft_model(mats, d, fit_mask, va, C_grid=C_GRID):
    X = sparse.hstack([sparse.csr_matrix(m) for m in mats]).tocsr()
    best = None
    for c in C_grid:
        m = binomial_logreg(X[fit_mask], d.wins[fit_mask], d.losses[fit_mask], c)
        v = deck_metrics(m.predict_proba(X[va])[:, 1], d.wins[va], d.losses[va])["nll"]
        if best is None or v < best[0]:
            best = (v, c, m)
    return best[2].predict_proba(X)[:, 1], best[1], best[0]


def game_fm_scores(set_code: str, event: str) -> np.ndarray:
    """Deck score of the game-level FM for every draft's registered deck (load_drafts order).

    Runs in a child process: torch and LightGBM each bundle their own OpenMP runtime, and using
    both in one process deadlocks on macOS.
    """
    import torch

    from decks.synergy import fit
    torch.set_num_threads(4)
    d = load_drafts(set_code, event)
    games = pl.read_parquet(game_path(set_code, event))
    gs = game_split(d, games["draft_id"])
    deck, _ = most_played_deck(d, games)
    g = games.filter(pl.Series(gs >= 0))
    Xg = aligned(g, "deck_", d.names).astype(np.float32)
    cov, y = covariates(g).astype(np.float32), flag(g["won"]).astype(np.int8)
    m, _, _ = fit(Xg, cov, y, gs[gs >= 0], 4, 1e-5, 1e-3, log=lambda s: None)
    with torch.no_grad():
        return m.deck_score(torch.from_numpy(deck.astype(np.float32))).numpy().astype(np.float64)


def main(*sets: str, event: str = "PremierDraft"):
    for set_code in sets or ("MSH",):
        t0 = time.time()
        out = ROOT / "results" / f"{set_code}.{event}" / "draftencoder"
        d = load_drafts(set_code, event)
        games = pl.read_parquet(game_path(set_code, event))
        gs = game_split(d, games["draft_id"])
        deck, has = most_played_deck(d, games)
        gih = game_win_rates(games.filter(pl.Series(gs == 0)), d.names)["gih_wr"]
        del games
        tr, va = d.mask("train") & has, d.mask("val") & has
        te = d.mask("test") & has
        ranks = np.eye(N_RANKS, dtype=np.float32)[d.rank]
        dk = deck.astype(np.float32)
        struct = structure_features(deck, d.names, set_code)
        arche = structure_features(deck, d.names, set_code, archetype=True)
        n_cards = np.maximum(dk.sum(1, keepdims=True), 1)
        gih_mean = ((dk @ (gih - gih.mean())) / n_cards.squeeze())[:, None].astype(np.float32) * 100

        preds, res = {}, {}
        for name, mats in (("linear_deck", [dk, ranks]),
                           ("+structure", [dk, ranks, struct]),
                           ("+gih_prior", [dk, ranks, gih_mean]),
                           ("+structure+gih_prior", [dk, ranks, struct, gih_mean]),
                           ("+structure+archetype", [dk, ranks, arche])):
            preds[name], c, v = fit_draft_model(mats, d, tr, va)
            res[name] = {"C": c, "val_nll": v}

        preds["lgbm_deck+structure"], it = fit_lgbm(
            np.hstack([dk, struct, d.rank[:, None].astype(np.float32)]), d, tr, va)
        res["lgbm_deck+structure"] = {"iterations": it}

        with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
            s = pool.submit(game_fm_scores, set_code, event).result()
        preds["game_fm"] = calibrate(s[tr], d.rank[tr], d.wins[tr], d.losses[tr], s, d.rank)
        avg = lambda *ks: expit(np.mean([logit(np.clip(preds[k], 1e-6, 1 - 1e-6)) for k in ks], 0))
        best_draft = min((k for k in res if "val_nll" in res[k]), key=lambda k: res[k]["val_nll"])
        preds["ens linear_deck+game_fm"] = avg("linear_deck", "game_fm")
        preds[f"ens {best_draft}+game_fm"] = avg(best_draft, "game_fm")
        preds["ens +structure+lgbm"] = avg("+structure", "lgbm_deck+structure")
        preds["ens +structure+lgbm+game_fm"] = avg("+structure", "lgbm_deck+structure", "game_fm")

        rows = {}
        for name, p in preds.items():
            rows[name] = {**deck_metrics(p[te], d.wins[te], d.losses[te]), **res.get(name, {})}
        (out / "improve_deck.json").write_text(json.dumps(rows, indent=1))
        print(f"=== {set_code} ({te.sum()} test drafts, {time.time() - t0:.0f}s)", flush=True)
        for name, r in rows.items():
            print(f"  {name:36s} spearman {r['spearman']:.4f}  nll {r['nll']:.5f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
