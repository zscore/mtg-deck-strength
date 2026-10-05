"""Does more information about the cards, or a deeper network, find more signal in a deck?

    python scripts/deck_features.py MSH

Premier Draft games only, same split and context columns as deck_synergy.ipynb. Adds two things:
  deck features   nonlinear deck facts from Scryfall data (land count, curve shape, colours, how well the
                  lands cast the spells; `cardinfo.deck_features`), as extra standardised context columns
  rich card meta  per-card pips, mana produced, power/toughness, keywords and rules-text roles for the
                  transformer's tokens (`cardinfo.card_matrix`), on top of colours/rarity/mana value/types
and a deeper, wider transformer. Scores every model on the test games, paired against linear.
Writes results/{SET}.PremierDraft/deck_features/{results.json, deck_feature_weights.json, test_preds.parquet}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from openlane.cardinfo import card_info, card_matrix, deck_features  # noqa: E402
from openlane.data import ROOT, card_metadata  # noqa: E402
from openlane.deckstrength import (aligned, covariates, experience, flag, game_split, load_drafts,  # noqa: E402
                                   record_before, record_onehot)
from openlane.games import game_path  # noqa: E402
from openlane.nn import card_features  # noqa: E402
from openlane.synergy import fit, fit_transformer  # noqa: E402


def nll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_features"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(set_code, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    info = card_info(names, set_code)
    F, fnames = deck_features(X, info)
    mu, sd = F[sp == 0].mean(0), F[sp == 0].std(0) + 1e-6
    ctx_f = np.hstack([ctx, (F - mu) / sd]).astype(np.float32)
    meta = card_features(card_metadata(names, set_code))
    meta_rich = np.hstack([meta, card_matrix(info)])
    te = np.flatnonzero(sp == 2)
    print(f"{len(y)} games, {len(fnames)} deck features, card meta {meta.shape[1]} -> {meta_rich.shape[1]}  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    log = lambda s: print(s, flush=True)  # noqa: E731
    runs = [  # (name, description, fitter)
        ("linear", "card counts + context", lambda: fit(X, ctx, y, sp, 0, 1e-4, 0, log=log)),
        ("linear + deck features", "", lambda: fit(X, ctx_f, y, sp, 0, 1e-4, 0, log=log)),
        *[(f"fm4 l2_v={l2v:g} + deck features", "", lambda l2v=l2v: fit(X, ctx_f, y, sp, 4, 1e-5, l2v, log=log))
          for l2v in (1e-3, 1e-2)],
        ("transformer, rich card meta", "d=64, 2 layers",
         lambda: fit_transformer(X, ctx, y, sp, meta_rich, log=log)),
        ("transformer, rich card meta + deck features", "d=64, 2 layers",
         lambda: fit_transformer(X, ctx_f, y, sp, meta_rich, log=log)),
        ("deep transformer, rich card meta + deck features", "d=128, 4 layers, 8 heads",
         lambda: fit_transformer(X, ctx_f, y, sp, meta_rich, d=128, layers=4, heads=8, lr=5e-4, log=log)),
    ]
    preds, rows = {}, []
    # the d=64 transformer with the basic card meta, from deck_synergy_multitask.py (same games, same order)
    prev = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_multitask" / "test_preds.parquet"
    if prev.exists():
        tp = pl.read_parquet(prev).filter(pl.col("event") == "PremierDraft")
        assert (tp["won"].to_numpy() == y[te]).all()
        preds["transformer, basic card meta"] = tp["transformer/draft"].to_numpy()
        rows.append({"model": "transformer, basic card meta", "config": "d=64, 2 layers (deck_synergy_multitask)",
                     "val logloss": None, "fit seconds": None})
    for name, desc, fitter in runs:
        print(f"== {name}", flush=True)
        t1 = time.time()
        m, v, predict = fitter()
        preds[name] = predict(te)
        rows.append({"model": name, "config": desc, "val logloss": v, "fit seconds": time.time() - t1})
        if name == "linear + deck features":
            wf = m.cov.weight.detach().numpy()[0, ctx.shape[1]:]
            (out / "deck_feature_weights.json").write_text(json.dumps(
                {"note": "log-odds per standard deviation of the feature (training games)",
                 "weights": dict(zip(fnames, wf.tolist())), "mean": dict(zip(fnames, mu.tolist())),
                 "sd": dict(zip(fnames, sd.tolist()))}, indent=1))
        _score(rows, preds, y[te])
        (out / "results.json").write_text(json.dumps(rows, indent=1))
    pl.DataFrame({"won": y[te], **preds}).write_parquet(out / "test_preds.parquet")
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


def _score(rows, preds, yt):
    base = nll(yt, preds["linear"]) if "linear" in preds else None
    for r in rows:
        p = preds[r["model"]]
        r["test logloss"], r["test auc"] = log_loss(yt, p), roc_auc_score(yt, p)
        if base is not None:
            dl = nll(yt, p) - base
            r["vs linear"], r["vs linear ±95%"] = float(dl.mean()), float(1.96 * dl.std() / np.sqrt(len(dl)))
    for r in rows:
        print(f"  {r['model']:50s} ll {r['test logloss']:.5f} auc {r['test auc']:.4f}"
              + (f"  vs linear {r['vs linear'] * 1000:+.2f} ± {r['vs linear ±95%'] * 1000:.2f} (×1000)"
                 if "vs linear" in r else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MSH")
