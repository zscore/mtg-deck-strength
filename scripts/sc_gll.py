"""Skill-controlled game log loss (SC-GLL) of existing deck scores: python scripts/sc_gll.py MSH   (--selftest)
    python scripts/sc_gll.py meta [--file sc_gll/sc_gll.json] [--subset experienced] MSH BLB ...   (meta-analysis)

evaluation_design.md §2.1, with eval_protocol.ctx_calibrated_game_loss. For every test game g of a test draft d:
    logit P(win_g) = c(x_g) + a + b * z(s(deck_d))
  c(x)    frozen context model, fit ONCE per set on train games with no deck term, shared by every candidate: play/draw,
          mulligan difference, experience one-hot, per-game rank, record state before the game (21 + unknown), skill
          bucket (0.04-wide bins + linear, linear term per experience bucket)
  (a, b)  fit on VAL games for every candidate (the train-draft scores are in-sample for LightGBM and cross-fitted
          on two scales for the game FM: pipeline_audit findings 5-6), scored on test games
  s(deck) a deck-only score: no skill, no experience; LightGBM is evaluated at a reference rank (platinum)
Subsets: "experienced" (user_n_games_bucket >= 50, the primary metric), "all", "match 1" (0-0 games: no
matchmaking or record collider; a sign check), "main build" (only games played with the rated, most-played build).
Models (deck-only scores of existing fits; LightGBM refit here, N_SEEDS seeds, round 4 settings):
  game FM                       draft_inputs.npz game_logit (round 4 best FM, cross-fitted on train drafts)
  linear game model             draft_inputs.npz game_logit_ref (round 4's `linear + deck features + aff`)
  LightGBM                      round 4 lgbm (counts + structure + rank + deck features + aff), logit at platinum
  round 4 blend                 (1 - alpha) LightGBM + alpha calibrated game FM, both at platinum; alpha on val NLL
Metric: -1000 x test log loss per game (higher is better). Paired day-block bootstraps over test drafts, with the
LightGBM seed spread as a variance component (eval_protocol.paired_bootstrap), saved as SetResult fields so
`meta` can run eval_protocol.meta_analysis / decide across sets without refitting.
Writes results/{SET}.PremierDraft/sc_gll/{sc_gll.json, context.npz, scores.npz}.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import N_STATES, REF_RANK, day_blocks, deck_rows, lgbm_model, load_games, main_builds, per_game  # noqa: E402,E501
from eval_protocol import (DecisionRule, SetResult, ctx_calibrated_game_loss, decide, gll_metric,  # noqa: E402
                           meta_analysis, paired_bootstrap, spearman_wr)
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS, binomial_nll, calibrate  # noqa: E402

N_SEEDS, N_BOOT, THREADS = 3, 500, 4
SKILL_EDGES = np.arange(0.38, 0.70, 0.04)
PRIMARY = "experienced"
PAIRS = [("game FM", "LightGBM"), ("game FM", "round 4 blend"), ("LightGBM", "round 4 blend"),
         ("linear game model", "game FM")]
z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731


# ---------------------------------------------------------------- the frozen context model

def context_design(pg: dict) -> np.ndarray:
    sk = pg["skill"]
    known = np.isfinite(sk)
    b = np.where(known, np.digitize(np.nan_to_num(sk), SKILL_EDGES) + 1, 0)
    lin = np.where(known, sk - 0.55, 0.0)
    return np.hstack([pg["on_play"][:, None], pg["mull"][:, None], pg["exp"], np.eye(N_RANKS)[pg["rank"]],
                      np.eye(N_STATES + 1)[pg["state"]], np.eye(len(SKILL_EDGES) + 2)[b], lin[:, None] * pg["exp"]])


def fit_context(pg: dict, fit: np.ndarray) -> tuple[np.ndarray, dict]:
    """Context logit for every game, logistic regression fit on `fit` games (train). No deck term."""
    Xc = context_design(pg)
    m = LogisticRegression(C=100.0, max_iter=2000).fit(Xc[fit], pg["y"][fit])
    ctx = m.decision_function(Xc)
    p = np.clip(expit(ctx), 1e-9, 1 - 1e-9)
    ll = -(pg["y"] * np.log(p) + (1 - pg["y"]) * np.log1p(-p))
    return ctx, {"train games": int(fit.sum()), "train logloss": float(ll[fit].mean()),
                 "test logloss (context only)": float(ll[pg["split"] == 2].mean())}


def subsets(pg: dict, has_game_draft: np.ndarray, main_bix=None) -> dict:
    """{name: (fit games, eval games)}: (a, b) fit on val games, scored on test games, both within the subset."""
    base = has_game_draft[pg["gd"]]
    out = {"all": base}
    if "n_bucket" in pg:
        out["experienced"] = base & (pg["n_bucket"] >= 50)
    out["match 1"] = base & (pg["match"] == 1)
    if main_bix is not None:
        out["main build"] = base & (pg["build_index"] == main_bix[pg["gd"]])
    return {k: (m & (pg["split"] == 1), m & (pg["split"] == 2)) for k, m in out.items()}


def sc_losses(score: np.ndarray, pg: dict, ctx: np.ndarray, fit: np.ndarray, ev: np.ndarray, D: int):
    """Per-draft summed test log loss [D], test games [D] and (a, b) for one deck score [D]."""
    sel = fit | ev
    ls, n, beta = ctx_calibrated_game_loss(np.nan_to_num(score), pg["gd"][sel], pg["y"][sel], ctx[sel], fit[sel], D)
    return ls, n, beta


def evaluate(scores: dict, pg: dict, ctx: np.ndarray, subs: dict, D: int, pairs=PAIRS, n_boot=N_BOOT, blocks=None,
             name="", log=print) -> dict:
    """scores: {model: [S, D] deck-only scores (S seeds)}. Returns per-model SC-GLL / b per subset and paired deltas
    (B - A, higher is better) as SetResult fields."""
    res = {"models": {m: {} for m in scores}, "pairs": {}}
    for sub, (fit, ev) in subs.items():
        L, n_ev = {}, None
        for m, S in scores.items():
            out = [sc_losses(s, pg, ctx, fit, ev, D) for s in np.atleast_2d(S)]
            L[m] = np.stack([o[0] for o in out])
            n_ev = out[0][1]
            fd = np.unique(pg["gd"][fit])
            sd = np.mean([np.nanstd(s[fd]) for s in np.atleast_2d(S)])
            res["models"][m][sub] = {"b per sd": float(np.mean([o[2][1] for o in out])),
                                     "b per score unit": float(np.mean([o[2][1] for o in out]) / sd)}
        units = np.flatnonzero(n_ev > 0)
        metric = gll_metric(n_ev[units])
        for m in scores:
            vals = [metric(l[units], np.arange(len(units))) for l in L[m]]
            res["models"][m][sub].update({"sc_gll": float(np.mean(vals)), "seed sd": float(np.std(vals)),
                                          "test games": int(n_ev.sum()), "test drafts": int(len(units))})
        for a, b in pairs:
            if a in L and b in L:
                r = paired_bootstrap(metric, L[a][:, units], L[b][:, units], n_boot=n_boot, seed=0,
                                     blocks=None if blocks is None else blocks[units], name=name)
                res["pairs"].setdefault(f"{b} - {a}", {})[sub] = {k: v for k, v in asdict(r).items() if k != "boot"}
        log(f"  {sub:11s} " + "  ".join(f"{m} {res['models'][m][sub]['sc_gll']:.3f}" for m in scores))
        for a, b in pairs:
            if f"{b} - {a}" in res["pairs"]:
                r = res["pairs"][f"{b} - {a}"][sub]
                log(f"      {b} - {a}: {r['delta']:+.4f} ± {r['se']:.4f} (x1000 nats/game)")
    return res


# ---------------------------------------------------------------- existing models' deck-only scores

def lgbm_scores(X, d, has, game_logit, rank_col, n_seeds=N_SEEDS, log=print):
    """Round 4 LightGBM per seed: own-rank p, platinum logit, and the round 4 blend (own rank / platinum)."""
    tr, va = d.mask("train") & has, d.mask("val") & has
    Xr = X.copy()
    Xr[:, rank_col] = REF_RANK
    ref = np.full(len(d.rank), REF_RANK)
    game_tr = calibrate(game_logit[tr], d.rank[tr], d.wins[tr], d.losses[tr], game_logit, d.rank)
    game_ref = calibrate(game_logit[tr], d.rank[tr], d.wins[tr], d.losses[tr], game_logit, ref)
    vnll = lambda p: binomial_nll(p[va], d.wins[va], d.losses[va])  # noqa: E731
    out = {"lgbm own": [], "lgbm ref": [], "blend own": [], "blend ref": [], "alpha": [], "trees": []}
    alphas = np.linspace(0, 1, 21)
    for s in range(n_seeds):
        t1 = time.time()
        m, n = lgbm_model(X, d, tr, va, params={"num_threads": THREADS}, seed=s)
        p, pr = m.predict(X, num_iteration=n), m.predict(Xr, num_iteration=n)
        a = float(alphas[np.argmin([vnll(expit((1 - a) * z(p) + a * z(game_tr))) for a in alphas])])
        out["lgbm own"].append(z(p))
        out["lgbm ref"].append(z(pr))
        out["blend own"].append((1 - a) * z(p) + a * z(game_tr))
        out["blend ref"].append((1 - a) * z(pr) + a * z(game_ref))
        out["alpha"].append(a)
        out["trees"].append(n)
        log(f"  lgbm seed {s}: {n} trees, alpha {a:.2f} [{time.time() - t1:.0f}s]")
    return {k: np.array(v) for k, v in out.items()}


def main(set_code):
    from improve_deck import structure_features
    from mtgdata.games import game_path
    from decks.deckstrength import load_drafts
    t0 = time.time()
    res_dir = ROOT / "results" / f"{set_code}.PremierDraft"
    out = res_dir / "sc_gll"
    out.mkdir(parents=True, exist_ok=True)
    inp = np.load(res_dir / "deck_synergy_v4" / "draft_inputs.npz")
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id)
    pg = per_game(d, g)
    has, rows, main_bix = main_builds(d, g)
    del g
    assert (has == inp["has_deck"]).all(), "has-deck mask differs from draft_inputs.npz"
    deck = np.zeros((len(d.draft_id), len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    D = len(d.draft_id)
    ctx, cinfo = fit_context(pg, pg["split"] == 0)
    print(f"=== {set_code}: {D} drafts, {len(pg['y'])} games; context train logloss {cinfo['train logloss']:.5f}, "
          f"test {cinfo['test logloss (context only)']:.5f} [{time.time() - t0:.0f}s]", flush=True)
    np.savez_compressed(out / "context.npz", ctx_logit=ctx.astype(np.float32), gd=pg["gd"].astype(np.int32),
                        y=pg["y"].astype(np.int8), match=pg["match"].astype(np.int8), split=pg["split"],
                        state=pg["state"].astype(np.int8), n_bucket=pg["n_bucket"].astype(np.float32),
                        main_build=(pg["build_index"] == main_bix[pg["gd"]]), draft_id=d.draft_id.astype(str))
    X = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                   d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
    rank_col = len(d.names) + structure_features(deck[:1], d.names, set_code).shape[1]
    assert (X[:, rank_col] == d.rank).all()
    lg = lgbm_scores(X, d, has, inp["game_logit"], rank_col, log=lambda s: print(s, flush=True))
    scores = {"game FM": inp["game_logit"][None], "linear game model": inp["game_logit_ref"][None],
              "LightGBM": lg["lgbm ref"], "round 4 blend": lg["blend ref"]}
    np.savez_compressed(out / "scores.npz", draft_id=d.draft_id.astype(str), has=has, split=d.split,
                        **{k.replace(" ", "_"): v.astype(np.float32) for k, v in scores.items()},
                        lgbm_own_rank=lg["lgbm own"].astype(np.float32), blend_own_rank=lg["blend own"].astype(np.float32))
    blocks = day_blocks(d)
    res = {"set": set_code, "context": cinfo, "reference rank": int(REF_RANK), "seeds": N_SEEDS,
           "alphas": lg["alpha"].tolist(), "trees": lg["trees"].tolist(), "primary subset": PRIMARY,
           **evaluate(scores, pg, ctx, subsets(pg, has, main_bix), D, blocks=blocks, name=set_code,
                      log=lambda s: print(s, flush=True))}
    te = d.mask("test") & has
    res["raw spearman (deck-only score vs draft-file W/(W+L))"] = {
        m: float(np.mean([spearman_wr(s[te], d.wins[te], d.losses[te]) for s in S])) for m, S in scores.items()}
    res["raw spearman, own-rank blend"] = float(np.mean([spearman_wr(expit(s[te]), d.wins[te], d.losses[te])
                                                          for s in lg["blend own"]]))
    res["seconds"] = time.time() - t0
    (out / "sc_gll.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


def meta(*args):
    """meta [--file sc_gll/sc_gll.json] [--subset experienced] SET ...: random-effects meta-analysis and decide() of
    every saved pair over the sets that have the file (works for structural_a/results*.json too)."""
    rel, subset, sets = "sc_gll/sc_gll.json", PRIMARY, []
    it = iter(args)
    for a in it:
        if a == "--file":
            rel = next(it)
        elif a == "--subset":
            subset = next(it)
        else:
            sets.append(a)
    res = {}
    for s in sets:
        f = ROOT / "results" / f"{s}.PremierDraft" / rel
        if f.exists():
            r = json.loads(f.read_text())
            res[s] = r["pairs"] if "pairs" in r else r["sc_gll"]["pairs"]
    print(f"{rel}: sets with results {list(res)}  (subset {subset!r})")
    for p in sorted({p for r in res.values() for p in r}):
        rs = [SetResult(**{**r[p][subset], "name": s}) for s, r in res.items() if p in r and subset in r[p]]
        if len(rs) < 2:
            continue
        m = meta_analysis(rs)
        ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
        print(f"\n{p}\n{m.summary()}\n  adopt: {ok}\n    " + "\n    ".join(why))


# ---------------------------------------------------------------- synthetic self-test

def synthetic(D=6000, seed=0, sd_deck=0.3, sd_skill=0.3):
    """Drafts with true deck strength and skill, record-matched 7-3 runs, a leaky skill bucket (includes this run)."""
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    deck = rng.normal(0, sd_deck, D)
    skill = rng.normal(0, sd_skill, D)
    n_hist = rng.choice([5, 10, 50, 100, 500], D, p=[0.1, 0.1, 0.3, 0.3, 0.2])
    rows = []
    W, L = np.zeros(D), np.zeros(D)
    for i in range(D):
        w = l = 0
        while w < 7 and l < 3:
            on_play = rng.random() < 0.5
            win = rng.random() < expit(0.15 + deck[i] + skill[i] + 0.1 * (on_play - 0.5) - 0.12 * (w - l))
            rows.append((i, w + l + 1, w, l, on_play, win))
            w, l = w + win, l + (not win)
        W[i], L[i] = w, l
    gd, match, wb, lb, op, y = (np.array(c) for c in zip(*rows))
    long_run = expit(0.15 + skill + rng.normal(0, 0.1, D))
    bucket = np.round(((n_hist - (W + L)).clip(1) * long_run + W) / np.maximum(n_hist, W + L) / 0.02) * 0.02
    split = np.zeros(D, np.int8)
    split[int(0.7 * D):], split[int(0.85 * D):] = 1, 2
    G = len(y)
    exp_b = np.searchsorted([1, 5, 10, 50, 100, 500, 1000], n_hist) + 1
    pg = {"gd": gd, "y": y.astype(float), "split": split[gd], "match": match, "known": np.ones(G, bool),
          "state": np.minimum(wb, 6) * 3 + np.minimum(lb, 2), "on_play": op - 0.5, "mull": np.zeros(G),
          "exp": np.eye(8)[exp_b[gd]], "rank": np.full(G, 3), "n_bucket": n_hist[gd].astype(float),
          "skill": bucket[gd], "build_index": np.full(G, "0", dtype=object)}
    d = SimpleNamespace(wins=W, losses=L, split=split, draft_time=np.array([f"2025-01-{1 + i * 28 // D:02d}" for i in range(D)]))
    return d, pg, deck, skill


def selftest():
    t0 = time.time()
    names, per_set = ("S1", "S2", "S3"), []
    for k, s in enumerate(names):
        d, pg, deck, skill = synthetic(seed=k)
        rng = np.random.default_rng(10 + k)
        D = len(d.wins)
        ctx, cinfo = fit_context(pg, pg["split"] == 0)
        good = deck + rng.normal(0, 0.15, D)
        scores = {"worse": (deck + rng.normal(0, 0.30, D))[None], "good": np.stack([good + rng.normal(0, 0.02, D)
                                                                                    for _ in range(3)]),
                  "good + skill leak": (good + 0.8 * skill)[None]}
        res = evaluate(scores, pg, ctx, subsets(pg, np.ones(D, bool), np.full(D, "0", dtype=object)), D,
                       pairs=[("worse", "good"), ("good", "good + skill leak")], n_boot=100,
                       blocks=day_blocks(d), name=s, log=(print if k == 0 else (lambda m: None)))
        te = d.split == 2
        raw = {m: spearman_wr(S[0][te], d.wins[te], d.losses[te]) for m, S in scores.items()}
        g = res["pairs"]["good - worse"]["experienced"]
        lk = res["pairs"]["good + skill leak - good"]["experienced"]
        print(f"[{s}] context test logloss {cinfo['test logloss (context only)']:.4f}; SC-GLL good - worse "
              f"{g['delta']:+.3f} ± {g['se']:.3f} (z {g['delta'] / g['se']:+.1f}); skill leak: raw Spearman "
              f"{raw['good + skill leak'] - raw['good']:+.4f}, SC-GLL z {lk['delta'] / lk['se']:+.1f}")
        assert g["delta"] > 0 and g["se_seed"] > 0 and raw["good + skill leak"] > raw["good"]
        json.dumps(res, default=float)
        per_set.append(SetResult(**g))
    m = meta_analysis(per_set)  # in memory: nothing is written under results/
    print(f"[meta] good - worse over {len(names)} synthetic sets:\n{m.summary()}")
    assert m.re_mean > 0
    print(f"self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
