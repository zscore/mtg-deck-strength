"""Deck model, round 10: cheap LightGBM-side gains on top of round 4's best model.

    python scripts/improve_deck_v2.py MSH [SET ...]     (after deck_synergy_v4.py, which caches draft_inputs.npz)

Starts from round 4's `lgbm + deck feat + aff` (registered-deck counts + structure + rank + Scryfall deck features +
affinity column), averaged with the cross-fitted game FM. Tests, choosing every setting on validation drafts:
  recency      training drafts weighted by exp(-age / tau), age = position before the end of training, as a share of it
  tuned        a small grid over LightGBM's tree size, leaf size, feature fraction and L2
  refit        the chosen model refit on train + val with the validation-picked number of trees (scaled by data size)
  seeds        the refit averaged over 5 bagging seeds
  + game FM    each averaged in logit space with the game FM; the weight is picked on validation (train-only fits)
Test drafts with a deck. Writes results/{SET}.PremierDraft/deck_synergy_v4/improve_v2.json.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve_deck import LGB_PARAMS, structure_features  # noqa: E402
from mtgdata.games import game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import binomial_nll, calibrate, deck_metrics, load_drafts, most_played_deck  # noqa: E402

TAUS = (None, 2.0, 1.0, 0.5, 0.25)
GRID = [dict(num_leaves=nl, min_data_in_leaf=md, feature_fraction=ff, lambda_l2=l2)
        for nl, md, ff, l2 in itertools.product((7, 15, 31), (100, 300, 1000), (0.3, 0.6), (10,))]
z = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))  # noqa: E731


def lgbm(X, d, fit, va=None, params=None, w=None, n_iter=None, seed=0):
    """LightGBM on W/L records (a positive row weighted W, a negative weighted L), optional per-draft weight `w`.

    With `va`, early-stops on it; otherwise trains exactly `n_iter` trees. Returns (p for every draft, trees)."""
    import lightgbm as lgb
    p = {**LGB_PARAMS, **(params or {}), "seed": seed}
    w = np.ones(len(X)) if w is None else w

    def ds(m, ref=None):
        y = np.r_[np.ones(m.sum()), np.zeros(m.sum())]
        sw = np.r_[d.wins[m] * w[m], d.losses[m] * w[m]]
        k = sw > 0
        return lgb.Dataset(np.vstack([X[m], X[m]])[k], y[k], weight=sw[k], reference=ref)
    dtr = ds(fit)
    if va is not None:
        model = lgb.train(p, dtr, 5000, valid_sets=[ds(va, dtr)], callbacks=[lgb.early_stopping(100, verbose=False)])
        n = model.best_iteration
    else:
        model = lgb.train(p, dtr, n_iter)
        n = n_iter
    return model.predict(X, num_iteration=n), n


def recency(d, fit, tau):
    """exp(-age / tau) with age in [0, 1]: 0 at the newest `fit` draft, 1 at the oldest (drafts are time-sorted)."""
    if tau is None:
        return np.ones(len(d.wins))
    idx = np.flatnonzero(fit)
    age = np.zeros(len(d.wins))
    age[idx] = (idx.max() - idx) / max(idx.max() - idx.min(), 1)
    return np.exp(-age / tau)


def main(*sets):
    import polars as pl
    for set_code in sets:
        t0 = time.time()
        out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4"
        inp = np.load(out / "draft_inputs.npz")
        d = load_drafts(set_code, "PremierDraft")
        deck, has = most_played_deck(d, pl.read_parquet(game_path(set_code, "PremierDraft")))
        assert (has == inp["has_deck"]).all()
        tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
        trva = tr | va
        X = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                       d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
        game_tr = calibrate(inp["game_logit"][tr], d.rank[tr], d.wins[tr], d.losses[tr], inp["game_logit"], d.rank)
        game_trva = calibrate(inp["game_logit"][trva], d.rank[trva], d.wins[trva], d.losses[trva],
                              inp["game_logit"], d.rank)
        vnll = lambda p: binomial_nll(p[va], d.wins[va], d.losses[va])  # noqa: E731
        preds, info = {}, {}

        preds["round 4: lgbm"], n0 = lgbm(X, d, tr, va)
        info["round 4: lgbm"] = {"trees": n0, "val_nll": vnll(preds["round 4: lgbm"])}

        # recency weighting, default params
        rec = {}
        for tau in TAUS:
            p, n = lgbm(X, d, tr, va, w=recency(d, tr, tau))
            rec[tau] = (vnll(p), p, n)
        tau = min(rec, key=lambda t: rec[t][0])
        preds["recency"] = rec[tau][1]
        info["recency"] = {"tau": tau, "trees": rec[tau][2], "val_nll": {str(t): v[0] for t, v in rec.items()}}

        # parameter grid at the chosen tau
        grid = []
        for g in GRID:
            p, n = lgbm(X, d, tr, va, params=g, w=recency(d, tr, tau))
            grid.append((vnll(p), g, p, n))
        v, g, p, n = min(grid, key=lambda r: r[0])
        preds["recency + tuned"] = p
        info["recency + tuned"] = {"params": g, "tau": tau, "trees": n, "val_nll": v,
                                   "grid": [(r[1], r[0], r[3]) for r in grid]}

        # refit on train + val with the chosen settings; trees scaled by data size
        n_fit = int(round(n * trva.sum() / tr.sum()))
        refits = [lgbm(X, d, trva, params=g, w=recency(d, trva, tau), n_iter=n_fit, seed=s)[0] for s in range(5)]
        preds["recency + tuned, refit on train+val"] = refits[0]
        preds["recency + tuned, refit, 5 seeds"] = expit(np.mean([z(r) for r in refits], 0))
        n0_fit = int(round(n0 * trva.sum() / tr.sum()))
        preds["round 4: lgbm, refit on train+val"] = lgbm(X, d, trva, n_iter=n0_fit)[0]
        preds["game FM"] = game_tr
        preds["game FM, calibrated on train+val"] = game_trva

        # blends: weight on the game FM picked on val with the train-only twin of each model
        alphas = np.linspace(0, 1, 21)
        def pick(p_va_model):
            return float(alphas[np.argmin([vnll(expit((1 - a) * z(p_va_model) + a * z(game_tr))) for a in alphas])])
        blend = lambda p, g_, a: expit((1 - a) * z(p) + a * z(g_))  # noqa: E731
        for name, twin, game in (("round 4: lgbm", "round 4: lgbm", game_tr),
                                 ("recency + tuned", "recency + tuned", game_tr),
                                 ("round 4: lgbm, refit on train+val", "round 4: lgbm", game_trva),
                                 ("recency + tuned, refit on train+val", "recency + tuned", game_trva),
                                 ("recency + tuned, refit, 5 seeds", "recency + tuned", game_trva)):
            a = pick(preds[twin])
            preds[f"{name} + game FM"] = blend(preds[name], game, a)
            info[f"{name} + game FM"] = {"alpha": a}

        ref = preds["round 4: lgbm + game FM"]
        rng = np.random.default_rng(0)
        ti = np.flatnonzero(te & ((d.wins + d.losses) > 0))
        wr = d.wins[ti] / (d.wins[ti] + d.losses[ti])
        boots = [rng.integers(0, len(ti), len(ti)) for _ in range(300)]
        rows = {}
        for k, p in preds.items():
            r = deck_metrics(p[te], d.wins[te], d.losses[te])
            dl = [spearmanr(p[ti][b], wr[b]).statistic - spearmanr(ref[ti][b], wr[b]).statistic for b in boots]
            r.update({"vs round 4 blend": r["spearman"] - deck_metrics(ref[te], d.wins[te], d.losses[te])["spearman"],
                      "vs round 4 blend sd": float(np.std(dl)), "info": info.get(k)})
            rows[k] = r
        (out / "improve_v2.json").write_text(json.dumps(rows, indent=1, default=str))
        print(f"=== {set_code} ({te.sum()} test drafts, {time.time() - t0:.0f}s)  tau={tau} params={g}", flush=True)
        for k, r in rows.items():
            print(f"  {k:52s} spearman {r['spearman']:.4f} nll {r['nll']:.5f}  "
                  f"vs r4 blend {r['vs round 4 blend']:+.4f} ± {r['vs round 4 blend sd']:.4f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
