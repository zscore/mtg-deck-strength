"""Deck strength models: python scripts/train_deckstrength.py MSH

1. Card values from game data (GP / GIH WR) and a Bradley-Terry fit on the cards both players
   cast (replay data), for several turn cutoffs.
2. Deck-level evaluation on held-out drafts, as in DraftEncoder (arXiv 2607.04782):
   Spearman(predicted p, W/(W+L)) and per-game NLL, for rank-only, top-23 heuristics with each
   card value, a linear bag-of-pool model and a pool transformer.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import COLORS, RARITIES, card_metadata  # noqa: E402
from mtgdata.games import aligned, flag, game_win_rates, cast_matrix, game_path, replay_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (N_RANKS, binomial_logreg, calibrate, covariates, deck_metrics, design,  # noqa: E402
                                fit_logreg, game_split, load_drafts, most_played_deck, top_k_mean,
                                train_pool_encoder)

CUTOFFS = (3, 5, 8, 30)
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)


def main(set_code: str, event: str = "PremierDraft"):
    out = ROOT / "results" / f"{set_code}.{event}" / "deckstrength"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load_drafts(set_code, event)
    names, C = d.names, len(d.names)
    tr, va, te = d.mask("train"), d.mask("val"), d.mask("test")
    print(f"drafts {len(d.draft_id)} (train {tr.sum()} / val {va.sum()} / test {te.sum()}), "
          f"cards {C}  [{time.time() - t0:.0f}s]", flush=True)

    # ------------------------------------------------ card values from game data (train only)
    games = pl.read_parquet(game_path(set_code, event))
    gs = game_split(d, games["draft_id"])
    values = game_win_rates(games.filter(pl.Series(gs == 0)), names)
    deck, has_deck = most_played_deck(d, games)
    print(f"games {games.height}, drafts with a deck {has_deck.mean():.1%}  [{time.time() - t0:.0f}s]",
          flush=True)

    # ------------------------------------------------ Bradley-Terry on cards cast by both sides
    replay = pl.read_parquet(replay_path(set_code, event))
    rs = game_split(d, replay["draft_id"])
    y = flag(replay["won"]).astype(np.int8)
    cov = covariates(replay)
    casts, unmapped = cast_matrix(replay, names, CUTOFFS)
    print(f"replay games {replay.height}, unmapped cast ids {unmapped:.2%}  [{time.time() - t0:.0f}s]",
          flush=True)
    game_results = {}

    def game_eval(name, X, extra=None):
        best = None
        for c in C_GRID:
            m = fit_logreg(X[rs == 0], y[rs == 0], c)
            v = log_loss(y[rs == 1], m.predict_proba(X[rs == 1])[:, 1])
            if best is None or v < best[0]:
                best = (v, c, m)
        _, c, m = best
        p = m.predict_proba(X[rs == 2])[:, 1]
        game_results[name] = {"logloss": log_loss(y[rs == 2], p), "auc": roc_auc_score(y[rs == 2], p),
                              "C": c, **(extra or {})}
        print(f"  {name:24s} logloss {game_results[name]['logloss']:.4f}  "
              f"auc {game_results[name]['auc']:.4f}  C={c}", flush=True)
        return m

    game_eval("covariates_only", sparse.csr_matrix(cov))
    n_cast = {}
    for k in CUTOFFS:
        U, O = casts[k]
        m = game_eval(f"cast_bt_t{k}", design(U, O, cov))
        values[f"cast_bt_t{k}"] = m.coef_[0, :C]
        n_cast[f"t{k}"] = (U[rs == 0].sum(0) + O[rs == 0].sum(0)).astype(int)
    U, O = casts[max(CUTOFFS)]
    m = game_eval("cast_bt_t30_asym", design(U, O, cov, symmetric=False))
    values["cast_bt_t30_user"], values["cast_bt_t30_opp"] = m.coef_[0, :C], m.coef_[0, C:2 * C]
    del casts, U, O

    # ------------------------------------------------ deck-level evaluation on test drafts
    w, l, r = d.wins, d.losses, d.rank
    deck_results, preds = {}, {}

    def record(name, p, raw=None):
        preds[name] = p
        res = deck_metrics(p[te], w[te], l[te])
        if raw is not None:
            res["spearman_raw_score"] = deck_metrics(raw[te], w[te], l[te])["spearman"]
        deck_results[name] = res
        print(f"  {name:28s} spearman {res['spearman']:.4f}  nll {res['nll']:.4f}  "
              f"ceiling {res['ceiling']:.3f}" +
              (f"  raw-score spearman {res['spearman_raw_score']:.4f}" if raw is not None else ""),
              flush=True)

    print("deck-level (test drafts):", flush=True)
    ranks = np.eye(N_RANKS)[r]
    record("rank_only", binomial_logreg(ranks[tr], w[tr], l[tr], 1e4).predict_proba(ranks)[:, 1])
    skill = np.nan_to_num(d.skill, nan=np.nanmean(d.skill))[:, None]
    Xs = np.hstack([ranks, skill])
    record("rank+skill_bucket (leaky)",
           binomial_logreg(Xs[tr], w[tr], l[tr], 1e4).predict_proba(Xs)[:, 1])

    for vname, v in values.items():
        for src, mat, k in (("pool_top23", d.pool, 23), ("deck_mean", deck, 40)):
            if src == "deck_mean" and not vname.startswith(("gp", "gih", "cast_bt_t30")):
                continue
            s = top_k_mean(mat, v, k)
            if src == "deck_mean":
                s[~has_deck] = np.nan
            record(f"{src}:{vname}", calibrate(s[tr], r[tr], w[tr], l[tr], s, r), raw=s)

    # linear bag-of-pool (the paper's LR baseline, final pool only)
    Xp = sparse.hstack([sparse.csr_matrix(d.pool.astype(np.float32)), sparse.csr_matrix(ranks)]).tocsr()
    best = None
    for c in C_GRID:
        m = binomial_logreg(Xp[tr], w[tr], l[tr], c)
        v = deck_metrics(m.predict_proba(Xp[va])[:, 1], w[va], l[va])["nll"]
        if best is None or v < best[0]:
            best = (v, m)
    record("linear_pool", best[1].predict_proba(Xp)[:, 1])
    values["linear_pool"] = best[1].coef_[0, :C]

    # pool transformer, without and with per-card meta features
    meta_df = card_metadata(names, set_code)
    meta = np.hstack([
        np.column_stack([(values[k] - values[k].mean()) / values[k].std()
                         for k in ("gp_wr", "gih_wr", "cast_bt_t30")]),
        meta_df.select(list(COLORS)).to_numpy().astype(np.float32),
        meta_df["mana_value"].to_numpy()[:, None] / 5,
        meta_df["is_creature"].to_numpy()[:, None].astype(np.float32),
        np.eye(len(RARITIES))[[RARITIES.index(x) for x in meta_df["rarity"].to_list()]],
    ]).astype(np.float32)
    for name, mt in (("pool_encoder", None), ("pool_encoder+meta", meta)):
        t1 = time.time()
        print(f"training {name}", flush=True)
        _, predict = train_pool_encoder(d, mt, log=lambda s: None)
        record(name, predict(np.arange(len(d.draft_id))))
        deck_results[name]["train_s"] = round(time.time() - t1)

    # ------------------------------------------------ save
    card_df = pl.DataFrame({"name": names, **{k: np.asarray(v, dtype=np.float64) for k, v in values.items()},
                            **{f"n_cast_{k}": v for k, v in n_cast.items()}})
    card_df.write_csv(out / "card_values.csv")
    np.savez_compressed(out / "deck_preds.npz", draft_id=d.draft_id, split=d.split,
                        **{k.replace(":", "__"): v for k, v in preds.items()})
    (out / "results.json").write_text(json.dumps({"game": game_results, "deck": deck_results}, indent=1))
    with pl.Config(tbl_rows=15, fmt_str_lengths=40):
        cols = ["name", "gih_wr", "gp_wr", "cast_bt_t30", "cast_bt_t5", "n_cast_t30"]
        print(card_df.filter(pl.col("n_cast_t30") > 500).sort("cast_bt_t30", descending=True)
              .select(cols).head(15))
        print(card_df.filter(pl.col("n_cast_t30") > 500).sort("cast_bt_t30").select(cols).head(10))
        print(card_df.select(*[pl.corr(a, b).alias(f"corr({a}, {b})") for a, b in
                               (("gih_wr", "cast_bt_t30"), ("gp_wr", "cast_bt_t30"),
                                ("gih_wr", "linear_pool"))]))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
