"""Card power + pairwise synergy from game results: python scripts/deck_synergy.py MSH

Each game is one example: P(user wins) = sigma(b + f(deck) + covariates), where deck is the
build registered for that game (lands included) and the covariates are play/draw, mulligans,
skill bucket and both ranks (so the card terms are adjusted for them).
  linear  f(x) = w . x                                      (card power)
  fm      f(x) = w . x + sum_{i<j} x_i x_j <v_i, v_j>        (factorization machine: synergy)
f is then used as a deck score for the held-out draft records, like the other deck models.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT  # noqa: E402
from openlane.deckstrength import (aligned, calibrate, covariates, deck_metrics, flag,  # noqa: E402
                                   game_split, load_drafts, most_played_deck)
from openlane.games import game_path  # noqa: E402
from openlane.synergy import fit  # noqa: E402


def main(set_code: str, event: str = "PremierDraft"):
    out = ROOT / "results" / f"{set_code}.{event}" / "deck_synergy"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(6)
    d = load_drafts(set_code, event)
    names = d.names
    games = pl.read_parquet(game_path(set_code, event))
    gs = game_split(d, games["draft_id"])
    keep = gs >= 0
    games, gs = games.filter(pl.Series(keep)), gs[keep]
    X = aligned(games, "deck_", names).astype(np.float32)
    cov = covariates(games).astype(np.float32)
    y = flag(games["won"]).astype(np.int8)
    deck, has_deck = most_played_deck(d, games)
    del games
    tr, te = d.mask("train"), d.mask("test") & has_deck
    w, l, r = d.wins, d.losses, d.rank
    print(f"games {len(y)} (train {np.sum(gs == 0)}), drafts scored {te.sum()}  [{time.time() - t0:.0f}s]",
          flush=True)

    results, models = {}, {}
    configs = [("cov_only", None, 0, 0), ("linear", 0, 1e-5, 0), ("linear", 0, 1e-4, 0),
               ("fm4", 4, 1e-5, 1e-4), ("fm4", 4, 1e-5, 1e-3),
               ("fm16", 16, 1e-5, 1e-4), ("fm16", 16, 1e-5, 1e-3)]
    for name, k, l2w, l2v in configs:
        Xin = np.zeros_like(X[:, :1]) if k is None else X
        m, v, predict = fit(Xin, cov, y, gs, k or 0, l2w, l2v, log=lambda s: print(s, flush=True))
        if name not in results or v < results[name]["val_logloss"]:
            p = predict(np.flatnonzero(gs == 2))
            results[name] = {"val_logloss": v, "l2_w": l2w, "l2_v": l2v,
                             "test_logloss": log_loss(y[gs == 2], p), "test_auc": roc_auc_score(y[gs == 2], p)}
            models[name] = m
    print("game level (test games):", flush=True)
    for name, res in results.items():
        print(f"  {name:10s} logloss {res['test_logloss']:.5f}  auc {res['test_auc']:.4f}", flush=True)

    print("draft level (test drafts with a deck):", flush=True)
    D = torch.from_numpy(deck.astype(np.float32))
    for name in ("linear", "fm4", "fm16"):
        with torch.no_grad():
            s = models[name].deck_score(D).numpy().astype(np.float64)
        p = calibrate(s[tr], r[tr], w[tr], l[tr], s, r)
        res = deck_metrics(p[te], w[te], l[te])
        results[name].update({f"draft_{k}": v for k, v in res.items()})
        print(f"  {name:10s} spearman {res['spearman']:.4f}  nll {res['nll']:.4f}", flush=True)

    # synergy table from the best FM: <v_i, v_j> for pairs that share enough training decks
    best_fm = min(("fm4", "fm16"), key=lambda n: results[n]["val_logloss"])
    V = models[best_fm].v.detach().numpy()
    wlin = models[best_fm].w.detach().numpy()
    P = (X[gs == 0] > 0).astype(np.float32)
    co = P.T @ P
    iu = np.triu_indices(len(names), 1)
    syn = (V @ V.T)[iu]
    ok = co[iu] >= 2000
    df = pl.DataFrame({"card_a": np.array(names)[iu[0]][ok], "card_b": np.array(names)[iu[1]][ok],
                       "synergy": syn[ok], "games_together": co[iu][ok].astype(int)}).sort("synergy")
    df.write_csv(out / "pair_synergy.csv")
    pl.DataFrame({"name": names, "power_linear": models["linear"].w.detach().numpy(),
                  "power_fm": wlin}).write_csv(out / "card_power.csv")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    with pl.Config(tbl_rows=15, fmt_str_lengths=32, tbl_width_chars=140):
        print(f"best FM: {best_fm}. Strongest positive pairs:")
        print(df.tail(12).reverse())
        print("Strongest negative pairs:")
        print(df.head(12))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
