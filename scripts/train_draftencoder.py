"""DraftEncoder runs: python scripts/train_draftencoder.py MSH [linear] [paper] [big] [big+deck] ...

`linear` fits the linear baselines (final pool, registered deck, both) and the GIH WR heuristics.
Each other argument trains one DraftEncoder size, with `+deck` adding the deck stage.
Everything is scored on the test drafts that have a deck in game data, so rows are comparable.
Results accumulate in results/<SET>.<EVENT>/draftencoder/results.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import COLORS, FEATURES, RARITIES, card_metadata  # noqa: E402
from mtgdata.games import game_win_rates, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (N_RANKS, binomial_logreg, calibrate, deck_metrics, game_split, load_drafts,  # noqa: E402
                                most_played_deck, top_k_mean)
from decks.draftencoder import pack_tokens, train  # noqa: E402

C_GRID = (0.003, 0.01, 0.03, 0.1)


def main(set_code: str, *runs: str, event: str = "PremierDraft"):
    runs = runs or ("linear", "paper", "big", "big+deck")
    out = ROOT / "results" / f"{set_code}.{event}" / "draftencoder"
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    t0 = time.time()

    d = load_drafts(set_code, event)
    games = pl.read_parquet(game_path(set_code, event))
    deck, has_deck = most_played_deck(d, games)
    values = game_win_rates(games.filter(pl.Series(game_split(d, games["draft_id"]) == 0)), d.names)
    del games
    tr, va = d.mask("train"), d.mask("val")
    te = d.mask("test") & has_deck
    w, l, r = d.wins, d.losses, d.rank
    print(f"drafts {len(d.draft_id)}, with deck {has_deck.mean():.1%}, test scored {te.sum()}  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    def record(name, p, extra=None):
        res = {**deck_metrics(p[te], w[te], l[te]), **(extra or {})}
        results[name] = res
        print(f"  {name:34s} spearman {res['spearman']:.4f}  nll {res['nll']:.4f}  "
              f"ceiling {res['ceiling']:.3f}", flush=True)
        res_path.write_text(json.dumps(results, indent=1))

    if "linear" in runs:
        ranks = np.eye(N_RANKS)[r]
        record("rank_only", binomial_logreg(ranks[tr], w[tr], l[tr], 1e4).predict_proba(ranks)[:, 1])
        for name, mat, k in (("pool_top23:gih_wr", d.pool, 23), ("deck_mean:gih_wr", deck, 40)):
            s = top_k_mean(mat, values["gih_wr"], k)
            record(name, calibrate(s[tr], r[tr], w[tr], l[tr], s, r))
        for name, mats in (("linear_pool", [d.pool]), ("linear_deck", [deck]),
                           ("linear_pool+deck", [d.pool, deck])):
            X = sparse.hstack([sparse.csr_matrix(m.astype(np.float32)) for m in mats]
                              + [sparse.csr_matrix(ranks)]).tocsr()
            fit = tr & has_deck if "deck" in name else tr
            best = None
            for c in C_GRID:
                m = binomial_logreg(X[fit], w[fit], l[fit], c)
                v = deck_metrics(m.predict_proba(X[va & has_deck])[:, 1], w[va & has_deck],
                                 l[va & has_deck])["nll"]
                if best is None or v < best[0]:
                    best = (v, c, m)
            record(name, best[2].predict_proba(X)[:, 1], {"C": best[1]})

    nn_runs = [x for x in runs if x != "linear"]
    if not nn_runs:
        return
    cache = FEATURES / f"{set_code}.{event}" / "draftencoder_tokens.npz"
    if cache.exists():
        z = np.load(cache)
        packs, pick_pos = z["packs"], z["pick_pos"]
    else:
        packs, pick_pos = pack_tokens(set_code, event, d.draft_id)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, packs=packs, pick_pos=pick_pos)
    print(f"pack tokens {packs.shape}  [{time.time() - t0:.0f}s]", flush=True)

    meta_df = card_metadata(d.names, set_code)
    meta = np.hstack([
        np.column_stack([(values[k] - values[k].mean()) / values[k].std() for k in ("gp_wr", "gih_wr")]),
        meta_df.select(list(COLORS)).to_numpy().astype(np.float32),
        meta_df["mana_value"].to_numpy()[:, None] / 5,
        meta_df["is_creature"].to_numpy()[:, None].astype(np.float32),
        np.eye(len(RARITIES))[[RARITIES.index(x) for x in meta_df["rarity"].to_list()]],
    ]).astype(np.float32)

    for run in nn_runs:
        size, use_deck = run.split("+")[0], run.endswith("+deck")
        t1 = time.time()
        print(f"training draftencoder_{run}", flush=True)
        _, predict = train(d, packs, pick_pos, meta, deck if use_deck else None, has_deck, size=size,
                           log=lambda s: print(s, flush=True))
        idx = np.flatnonzero(te)
        pk, dk = predict(idx)
        full = np.full(len(d.draft_id), np.nan)
        by_pick = [deck_metrics(pk[:, t], w[idx], l[idx])["spearman"] for t in range(pk.shape[1])]
        extra = {"train_s": round(time.time() - t1), "spearman_by_pick": [round(x, 4) for x in by_pick]}
        full[idx] = pk[:, -1]
        if use_deck:
            record(f"draftencoder_{run}:final_pick_head", full)
            full[idx] = dk
        record(f"draftencoder_{run}", full, extra)
        np.save(out / f"test_pred_{run}.npy", np.stack([pk[:, -1]] + ([dk] if dk is not None else [])))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
