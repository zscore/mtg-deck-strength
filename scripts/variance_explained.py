"""Share of true win-rate variance a deck model explains: python scripts/variance_explained.py MSH [SET ...]

Spearman against W/(W+L) mixes model quality with Bo1 noise. Here each test draft has a latent true win probability,
logit p_i = x_i'b + s * e_i with e_i ~ N(0, 1), and its record is scored with the exact likelihood p^W (1-p)^L (the
7-win / 3-loss stopping rule depends only on results so far, so it drops out). Fitting by marginal likelihood
(Gauss-Hermite quadrature) gives s^2, the true logit variance left unexplained by x:
  x = 1                    s0^2: total true variance across drafts (deck + player + matchmaking), noise removed
  x = 1 + model logit      share explained = 1 - s^2 / s0^2
  + skill bucket / rank    how much of what is left is the player
Models are round 4's (LightGBM + deck features + affinity, averaged with the game FM) and the same with PickNet deck
vectors (improve_deck_picknet.py). Fits use test drafts with a deck. Writes results/{SET}.PremierDraft/variance.json.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve_deck import structure_features  # noqa: E402
from improve_deck_picknet import picknet_inputs  # noqa: E402
from improve_deck_v2 import lgbm, z  # noqa: E402
from mtgdata.games import game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (N_RANKS, binomial_nll, calibrate, load_drafts, most_played_deck,  # noqa: E402
                                record_before, record_onehot)
from mtgdata.games import flag  # noqa: E402

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)  # nodes / weights for E[f(e)], e ~ N(0, 1)
GH_W = GH_W / GH_W.sum()


def random_effects(X, G, y, gd):
    """Max marginal likelihood, per game: logit P(win) = X[draft] b + G c + s e_draft, e ~ N(0, 1).

    X [D, p] draft covariates (first column 1), G [games, q] game covariates (record before the game: under
    record-based matchmaking a winning run meets stronger opponents, so a draft's games are not iid), y [games]
    won, gd [games] draft index. Returns (b, s^2)."""
    D, p = X.shape
    sign = np.where(y > 0, 1.0, -1.0)[:, None]

    def nll(theta):
        b, c, s = theta[:p], theta[p:-1], np.exp(theta[-1])
        eta = ((X @ b)[gd] + G @ c)[:, None] + s * GH_X[None]
        lg = log_expit(sign * eta)
        ll = np.zeros((D, len(GH_X)))
        np.add.at(ll, gd, lg)
        m = ll.max(1, keepdims=True)
        return -(m.squeeze(1) + np.log(np.exp(ll - m) @ GH_W)).sum() / len(y)
    r = minimize(nll, np.r_[np.zeros(p + G.shape[1]), np.log(0.3)], method="L-BFGS-B")
    return r.x[:p], float(np.exp(2 * r.x[-1]))


def main(*sets):
    for set_code in sets:
        out = ROOT / "results" / f"{set_code}.PremierDraft"
        inp = np.load(out / "deck_synergy_v4" / "draft_inputs.npz")
        d = load_drafts(set_code, "PremierDraft")
        deck, has = most_played_deck(d, pl.read_parquet(game_path(set_code, "PremierDraft")))
        tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
        base = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                          d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
        cache = out / "variance_preds.npz"
        if cache.exists():
            preds = dict(np.load(cache))
        else:
            groups, _ = picknet_inputs(set_code, d, deck, tr)
            game = calibrate(inp["game_logit"][tr], d.rank[tr], d.wins[tr], d.losses[tr], inp["game_logit"], d.rank)
            vnll = lambda p: binomial_nll(p[va], d.wins[va], d.losses[va])  # noqa: E731
            alphas = np.linspace(0, 1, 21)
            preds = {"game FM alone": game}
            for name, X in (("round 4 blend", base),
                            ("round 4 blend + PickNet deck vec", np.hstack([base, groups["deck vec"]]))):
                p, _ = lgbm(X, d, tr, va)
                a = alphas[np.argmin([vnll(expit((1 - a) * z(p) + a * z(game))) for a in alphas])]
                preds[name] = expit((1 - a) * z(p) + a * z(game))
            np.savez(cache, **preds)

        k = te & ((d.wins + d.losses) > 0)
        games = pl.read_parquet(game_path(set_code, "PremierDraft"), columns=["draft_id", "match_number", "won"])
        pos = {x: i for i, x in enumerate(d.draft_id[k].tolist())}
        games = games.filter(pl.col("draft_id").is_in(list(pos)))
        wb, lb, known = record_before(games)
        games, wb, lb = games.filter(pl.Series(known)), wb[known], lb[known]  # complete history only
        G = np.column_stack([wb, lb]).astype(np.float64)  # matchmaking: linear in wins and losses so far
        y = flag(games["won"]).astype(np.float64)
        gd = np.array([pos[x] for x in games["draft_id"].to_list()])
        one = np.ones((k.sum(), 1))
        ranks = np.eye(N_RANKS)[d.rank[k]][:, 1:]
        sk = d.skill[k]
        skill = np.column_stack([np.nan_to_num(sk - np.nanmean(sk)), np.isnan(sk)])
        fit = lambda X: random_effects(X, G, y, gd)  # noqa: E731
        b0, s0 = fit(one)
        res = {"test drafts": int(k.sum()), "test games": int(len(y)), "true logit sd": s0 ** 0.5,
               "true win-rate sd (approx)": float(s0 ** 0.5 * 0.25), "share explained": {}}
        res["share explained"]["rank only"] = 1 - fit(np.hstack([one, ranks]))[1] / s0
        res["share explained"]["rank + skill bucket (leaky, reference)"] = 1 - fit(np.hstack([one, ranks, skill]))[1] / s0
        for name, p in preds.items():
            zz = z(p[k])[:, None]
            b, s = fit(np.hstack([one, zz]))
            res["share explained"][name] = 1 - s / s0
            res["share explained"][f"{name} + skill bucket"] = 1 - fit(np.hstack([one, zz, skill]))[1] / s0
            res[f"{name}: slope on its logit"] = float(b[1])
            print(f"  {name}: share {1 - s / s0:.3f}", flush=True)
        res["spearman of round 4 blend, actual"] = float(spearmanr(preds["round 4 blend"][k], d.wins[k] / (d.wins[k] + d.losses[k])).statistic)
        (out / "variance.json").write_text(json.dumps(res, indent=1))
        print(f"=== {set_code}", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
