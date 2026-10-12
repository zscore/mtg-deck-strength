"""The paper's linear baseline, rebuilt on card identity: python scripts/paper_baselines.py MSH [SET ...]

DraftEncoder's LinearRegression (models/baselines.py in akulen/MtGDraftEncoder) concatenates the
feature vector of every pick *in pick order* and fits one linear head per pick number, so a card's
weight depends on the slot it was taken in. It uses 768-d oracle-text embeddings; here each slot
gets a one-hot of card identity instead. Compared with the bag-of-cards `linear_pool`, this
isolates what the order-dependent design costs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import download  # noqa: E402
from mtgdata.games import game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS, binomial_logreg, deck_metrics, load_drafts, most_played_deck  # noqa: E402

C_GRID = (0.0003, 0.001, 0.003, 0.01)


def pick_sequence(set_code: str, event: str, draft_ids: np.ndarray, names: list[str]) -> np.ndarray:
    """[D, T] card index taken at each pick, in pick order."""
    idx = {n: i for i, n in enumerate(names)}
    rows = (pl.scan_parquet(download(set_code, event))
            .select("draft_id", "pack_number", "pick_number", "pick")
            .filter(pl.col("draft_id").is_in(pl.Series(draft_ids).implode()))
            .sort("draft_id", "pack_number", "pick_number").collect())
    T = rows.height // len(draft_ids)
    order = {x: i for i, x in enumerate(rows["draft_id"].unique(maintain_order=True).to_list())}
    seq = np.array([idx[p] for p in rows["pick"].to_list()], dtype=np.int64).reshape(-1, T)
    return seq[[order[x] for x in draft_ids.tolist()]]


def main(*sets: str, event: str = "PremierDraft"):
    for set_code in sets or ("MSH",):
        d = load_drafts(set_code, event)
        C = len(d.names)
        seq = pick_sequence(set_code, event, d.draft_id, d.names)
        D, T = seq.shape
        _, has_deck = most_played_deck(d, pl.read_parquet(game_path(set_code, event),
                                                          columns=["draft_id", "build_index"]))
        tr, va = d.mask("train"), d.mask("val")
        te = d.mask("test") & has_deck
        w, l = d.wins, d.losses
        ranks = sparse.csr_matrix(np.eye(N_RANKS)[d.rank])
        rows = np.repeat(np.arange(D), T)
        cols = (np.arange(T)[None] * C + seq).ravel()
        X = sparse.hstack([sparse.csr_matrix((np.ones(D * T, np.float32), (rows, cols)), shape=(D, T * C)),
                           ranks]).tocsr()
        best = None
        for c in C_GRID:
            m = binomial_logreg(X[tr], w[tr], l[tr], c)
            v = deck_metrics(m.predict_proba(X[va])[:, 1], w[va], l[va])["nll"]
            if best is None or v < best[0]:
                best = (v, c, m)
        p = best[2].predict_proba(X[te])[:, 1]
        res = {**deck_metrics(p, w[te], l[te]), "C": best[1], "features": int(X.shape[1])}
        out = ROOT / "results" / f"{set_code}.{event}" / "draftencoder"
        (out / "paper_baselines.json").write_text(json.dumps({"paper_lr_order_dependent": res}, indent=1))
        ref = json.loads((out / "results.json").read_text()).get("linear_pool", {}).get("spearman", float("nan"))
        print(f"{set_code}: paper-style LR spearman {res['spearman']:.4f} nll {res['nll']:.4f} "
              f"(C={best[1]}, {X.shape[1]} features)   vs bag-of-cards linear_pool {ref:.4f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
