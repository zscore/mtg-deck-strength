"""Linear vs FM vs transformer game models, trained on PremierDraft, Sealed, or both (multitask).

    python scripts/deck_synergy_multitask.py MSH

Both events are best-of-one. Every model is P(win) = sigma(b + f(deck) + gamma . context) with the
same context columns (play/draw, mulligans, skill, rank, experience, record before the game); only
f differs. Each is fitted three ways:
  draft   PremierDraft games only
  sealed  Sealed games only
  multi   both; card terms shared, per-event bias, context weights and card power offset (dw)
and scored on the held-out (latest 15% of events) games of both events.
Writes results/{SET}.PremierDraft/deck_synergy_multitask/: results.json, test_preds.parquet, cards.parquet
(card power from the linear fits) and models/transformer_{regime}.pt.
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
from mtgdata.data import card_metadata  # noqa: E402
from mtgdata.encode import card_features  # noqa: E402
from mtgdata.games import aligned, flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (covariates, event_split, experience, game_split, load_drafts, record_before,  # noqa: E402
                                record_onehot)
from decks.synergy import fit, fit_transformer  # noqa: E402

EVENTS = ("PremierDraft", "Sealed")


def load(set_code: str):
    d = load_drafts(set_code, "PremierDraft")
    parts = []
    for t, event in enumerate(EVENTS):
        g = pl.read_parquet(game_path(set_code, event))
        sp = game_split(d, g["draft_id"]) if event == "PremierDraft" else event_split(g)
        g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
        wb, lb, _ = record_before(g)
        parts.append((aligned(g, "deck_", d.names).astype(np.float32), flag(g["won"]).astype(np.int8),
                      np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32),
                      sp, np.full(len(sp), t, np.int8)))
    X, y, ctx, sp, task = (np.concatenate(a) for a in zip(*parts))
    return d.names, X, y, ctx, sp, task


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_multitask"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    t0 = time.time()
    names, X, y, ctx, sp, task = load(set_code)
    meta = card_features(card_metadata(names, set_code))
    for t, e in enumerate(EVENTS):
        n = [int(((task == t) & (sp == s)).sum()) for s in range(3)]
        print(f"{e}: train/val/test games {n}, win rate {y[task == t].mean():.3f}", flush=True)
    print(f"{X.shape[1]} cards, {ctx.shape[1]} context columns  [{time.time() - t0:.0f}s]", flush=True)

    regimes = {"draft": task == 0, "sealed": task == 1, "multi": np.ones(len(y), bool)}
    log = lambda s: print(s, flush=True)  # noqa: E731
    results, preds, cards = [], {}, {}
    test = {e: np.flatnonzero((task == t) & (sp == 2)) for t, e in enumerate(EVENTS)}

    for regime, rows in regimes.items():
        split = np.where(rows, sp, -1)
        tk = task if regime == "multi" else None
        dws = (1e-4, 1e-3, 1e-2) if tk is not None else (1e-3,)
        # (family, label, fitter); the best of each family on validation is kept
        grid = [("linear", f"l2_w={l2w:g} l2_dw={dw:g}",
                 lambda l2w=l2w, dw=dw: fit(X, ctx, y, split, 0, l2w, 0, task=tk, l2_dw=dw, log=log))
                for l2w in (1e-5, 1e-4) for dw in dws]
        grid += [("fm", f"k={k} l2_v={l2v:g} l2_dw={dw:g}",
                  lambda k=k, l2v=l2v, dw=dw: fit(X, ctx, y, split, k, 1e-5, l2v, task=tk, l2_dw=dw, log=log))
                 for k in (4, 16) for l2v in (1e-3, 1e-2) for dw in dws]
        best = {}
        for fam, label, fitter in grid:
            print(f"[{regime}] {fam} {label}", flush=True)
            t1 = time.time()
            m, v, predict = fitter()
            if fam not in best or v < best[fam][1]:
                best[fam] = (label, v, m, predict, time.time() - t1)
        # transformer: one config, with the multitask card offset penalty the linear model chose
        dw = float(best["linear"][0].split("l2_dw=")[1])
        print(f"[{regime}] transformer (l2_dw={dw:g})", flush=True)
        t1 = time.time()
        m, v, predict = fit_transformer(X, ctx, y, split, meta, task=tk, l2_dw=dw, log=log)
        best["transformer"] = (f"d=64 layers=2 l2_dw={dw:g}", v, m, predict, time.time() - t1)

        for fam, (label, v, m, predict, secs) in best.items():
            row = {"model": fam, "trained on": regime, "config": label, "val logloss": v, "fit seconds": secs}
            for e, idx in test.items():
                p = predict(idx)
                preds[(fam, regime, e)] = p
                row[f"{e} test logloss"] = log_loss(y[idx], p)
                row[f"{e} test auc"] = roc_auc_score(y[idx], p)
            results.append(row)
            print(f"  {fam:12s} {regime:6s} " + "  ".join(
                f"{e}: ll {row[f'{e} test logloss']:.5f} auc {row[f'{e} test auc']:.4f}" for e in EVENTS), flush=True)
            if fam == "linear":
                w = m.w.detach().cpu().numpy()
                cards[f"w_{regime}"] = w
                if m.dw is not None:
                    for t, e in enumerate(EVENTS):
                        cards[f"w_multi_{e}"] = w + m.dw[t].detach().cpu().numpy()
            if fam == "transformer":
                (out / "models").mkdir(exist_ok=True)
                torch.save(m.state_dict(), out / "models" / f"transformer_{regime}.pt")
        (out / "results.json").write_text(json.dumps(results, indent=1))

    tp = []
    for e, idx in test.items():
        df = {"event": [e] * len(idx), "row": idx, "won": y[idx]}
        df.update({f"{fam}/{regime}": p for (fam, regime, ev), p in preds.items() if ev == e})
        tp.append(pl.DataFrame(df))
    pl.concat(tp).write_parquet(out / "test_preds.parquet")
    pl.DataFrame({"card": names, **cards}).write_parquet(out / "cards.parquet")
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MSH")
