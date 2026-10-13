"""Draft-file vs game-file targets: python scripts/rebuild_targets.py MSH     (after deck_synergy_v4.py; --selftest)

pipeline_audit finding 2: the target W-L comes from the draft file's event_match_wins/losses, which looks like a
snapshot taken partway through the run for about a fifth of drafts (more logged games than W+L). Here:
  1. rebuild  each draft's record from the game file: every game of the run, all builds (count games per draft_id);
              `complete` when the match numbers are exactly 1..n. A reconciled record (game file when complete and at
              least as long as the draft-file record, else the draft file) is saved too, not scored.
  2. compare  with the draft file per split: agree, stale (game file has more games), missing (fewer), other.
  3. rescore  round 4's blend (LightGBM + cross-fitted game FM, alpha on val) and round 10's best (recency + tuned,
              refit on train+val, 5 seeds, + game FM) with each target used for BOTH training (LightGBM records, the
              game FM's calibration, the blend weight, early stopping) and evaluation: a 2 x 2 of train target x eval
              target on test drafts with a deck. The LightGBM settings (tau, params) are improve_v2.json's
              validation choices when that file exists (chosen under the draft-file target), else tau = 2 and the
              default parameters. Paired day-block bootstraps (eval_protocol) of: round 10 vs round 4 under each
              cell, and game-trained vs draft-trained for each model and eval target.
Writes results/{SET}.PremierDraft/targets.npz (aligned to load_drafts order: draft_id, wins_games, losses_games,
n_games, complete, wins_reconciled, losses_reconciled) and targets.json.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import day_blocks, load_games, main_decks  # noqa: E402
from eval_protocol import paired_bootstrap, spearman_wr  # noqa: E402
from improve_deck_v2 import lgbm, recency, z  # noqa: E402
from mtgdata.games import flag  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import binomial_nll, calibrate, load_drafts  # noqa: E402

N_SEEDS, N_BOOT, THREADS = 5, 300, 4
R4, R10 = "round 4 blend", "round 10 best (refit, 5 seeds, + game FM)"


def game_records(d, g: pl.DataFrame) -> dict:
    """Per draft (d order): wins, losses and games in the game file over all builds; complete = matches 1..n."""
    a = (g.select("draft_id", pl.col("match_number").cast(pl.Int64), pl.Series("y", flag(g["won"]).astype(np.int64)))
         .group_by("draft_id").agg(pl.len().alias("n"), pl.col("y").sum().alias("w"),
                                   pl.col("match_number").n_unique().alias("nu"), pl.col("match_number").max().alias("mx"),
                                   pl.col("match_number").min().alias("mn")))
    m = {k: v for k, *v in a.iter_rows()}
    get = lambda j: np.array([m[x][j] if x in m else 0 for x in d.draft_id.tolist()], dtype=np.int64)  # noqa: E731
    n, w, nu, mx, mn = (get(j) for j in range(5))
    return {"wins_games": w, "losses_games": n - w, "n_games": n, "complete": (n > 0) & (nu == n) & (mx == n) & (mn == 1)}


def reconcile(d, r) -> tuple[np.ndarray, np.ndarray]:
    use = r["complete"] & (r["n_games"] >= d.wins + d.losses)
    return np.where(use, r["wins_games"], d.wins), np.where(use, r["losses_games"], d.losses)


def compare(d, r, has) -> dict:
    """Share of drafts per split where the two records agree / the draft file is stale (fewer games than the game
    file) / the game file is missing games / other; among drafts with at least one logged game."""
    W, L, Wg, Lg, n = d.wins, d.losses, r["wins_games"], r["losses_games"], r["n_games"]
    out = {}
    for k, name in enumerate(("train", "val", "test")):
        m = (d.split == k) & has
        tot = W[m] + L[m]
        agree = (W[m] == Wg[m]) & (L[m] == Lg[m])
        stale, missing = n[m] > tot, n[m] < tot
        out[name] = {"drafts": int(m.sum()), "no games": int(((d.split == k) & ~has).sum()),
                     "agree": float(agree.mean()), "stale (game file has more games)": float(stale.mean()),
                     "draft record is a prefix of the game record": float((stale & (W[m] <= Wg[m]) & (L[m] <= Lg[m])).mean()),
                     "missing (game file has fewer games)": float(missing.mean()),
                     "same games, different W/L": float((~agree & ~stale & ~missing).mean()),
                     "game history complete": float(r["complete"][m].mean()),
                     "mean games: draft file": float(tot.mean()), "mean games: game file": float(n[m].mean())}
    return out


def _alpha(vnll, p_va_model, game, alphas=np.linspace(0, 1, 21)):
    return float(alphas[np.argmin([vnll(expit((1 - a) * z(p_va_model) + a * z(game))) for a in alphas])])


def rescore(d, X, has, game_logit, targets: dict, cfg: dict, n_seeds=N_SEEDS, n_boot=N_BOOT, blocks=None, log=print):
    """Train round 4's blend and round 10's best under each target in `targets` ({name: (W, L)}), evaluate every
    prediction under every target on test drafts with a deck. Returns (json-ready dict, preds)."""
    tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
    trva = tr | va
    th = {"num_threads": THREADS}
    params, tau = {**th, **cfg["params"]}, cfg["tau"]
    preds, info = {}, {}
    for tn, (W, L) in targets.items():
        t1 = time.time()
        dt = replace(d, wins=W.astype(np.float32), losses=L.astype(np.float32))
        vnll = lambda p: binomial_nll(p[va], dt.wins[va], dt.losses[va])  # noqa: E731
        p4, n4 = lgbm(X, dt, tr, va, params=th)
        twin, n = lgbm(X, dt, tr, va, params=params, w=recency(dt, tr, tau))
        n_fit = int(round(n * trva.sum() / tr.sum()))
        refits = [lgbm(X, dt, trva, params=params, w=recency(dt, trva, tau), n_iter=n_fit, seed=s)[0]
                  for s in range(n_seeds)]
        refit = expit(np.mean([z(r) for r in refits], 0))
        game_tr = calibrate(game_logit[tr], d.rank[tr], dt.wins[tr], dt.losses[tr], game_logit, d.rank)
        game_trva = calibrate(game_logit[trva], d.rank[trva], dt.wins[trva], dt.losses[trva], game_logit, d.rank)
        a4, a10 = _alpha(vnll, p4, game_tr), _alpha(vnll, twin, game_tr)
        preds[tn] = {"round 4: lgbm": p4, "game FM": game_tr, R4: expit((1 - a4) * z(p4) + a4 * z(game_tr)),
                     "refit, 5 seeds": refit, R10: expit((1 - a10) * z(refit) + a10 * z(game_trva))}
        info[tn] = {"round 4 trees": n4, "twin trees": n, "refit trees": n_fit, "alpha round 4": a4,
                    "alpha round 10": a10, "val nll round 4": vnll(p4), "seconds": time.time() - t1}
        log(f"  trained on {tn:6s} target: trees {n4}/{n}/{n_fit}, alpha {a4:.2f}/{a10:.2f} [{time.time() - t1:.0f}s]")
    ev = np.flatnonzero(te)
    blocks = None if blocks is None else blocks[ev]
    agree = np.ones(len(d.wins), bool)
    if "draft" in targets and "game" in targets:
        agree = (targets["draft"][0] == targets["game"][0]) & (targets["draft"][1] == targets["game"][1])
    res = {"settings": {**cfg, "seeds": n_seeds, "test drafts": int(te.sum()),
                        "test drafts where the targets agree": int((te & agree).sum())},
           "training": info, "spearman": {}, "nll": {}, "spearman, targets agree": {}, "paired": {}}
    for tn, P in preds.items():
        for m, p in P.items():
            for en, (W, L) in targets.items():
                k = f"{m} | train {tn} | eval {en}"
                res["spearman"][k] = spearman_wr(p[ev], W[ev], L[ev])
                ok = ev[(W[ev] + L[ev]) > 0]
                res["nll"][k] = binomial_nll(p[ok], W[ok], L[ok])
                a = ev[agree[ev]]
                res["spearman, targets agree"][k] = spearman_wr(p[a], W[a], L[a])

    def boot(pa, pb, W, L, name):
        fn = lambda p, idx: spearman_wr(p[idx], W[ev][idx], L[ev][idx])  # noqa: E731
        r = paired_bootstrap(fn, pa[ev], pb[ev], n_boot=n_boot, seed=0, blocks=blocks, name=name)
        out = {k: v for k, v in asdict(r).items() if k != "boot"}
        res["paired"][name] = out
        log(f"  {name:72s} {r.delta:+.4f} ± {r.se:.4f}")
    for en, (W, L) in targets.items():
        for tn in preds:
            boot(preds[tn][R4], preds[tn][R10], W, L, f"round 10 - round 4 | train {tn} | eval {en}")
        if "draft" in preds and "game" in preds:
            for m in (R4, R10, "round 4: lgbm"):
                boot(preds["draft"][m], preds["game"][m], W, L, f"train game - train draft | {m} | eval {en}")
    return res, preds


def default_cfg(set_code):
    f = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4" / "improve_v2.json"
    if f.exists():
        c = json.loads(f.read_text())["recency + tuned"]["info"]
        return {"tau": c["tau"], "params": c["params"], "source": str(f.relative_to(ROOT))}
    return {"tau": 2.0, "params": {}, "source": "default (no improve_v2.json): tau 2, LGB_PARAMS"}


def main(set_code):
    from improve_deck import structure_features
    from mtgdata.games import game_path
    t0 = time.time()
    res_dir = ROOT / "results" / f"{set_code}.PremierDraft"
    inp = np.load(res_dir / "deck_synergy_v4" / "draft_inputs.npz")
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id, cols=("draft_id", "build_index", "match_number", "won"))
    rec = game_records(d, g)
    deck, has = main_decks(path, d, g)
    del g
    assert (has == inp["has_deck"]).all(), "has-deck mask differs from draft_inputs.npz"
    wr, lr = reconcile(d, rec)
    np.savez_compressed(res_dir / "targets.npz", draft_id=d.draft_id.astype(str), wins_reconciled=wr,
                        losses_reconciled=lr, **rec)
    cmp = compare(d, rec, has)
    print(f"=== {set_code}: {len(d.wins)} drafts [{time.time() - t0:.0f}s]", flush=True)
    for k, v in cmp.items():
        print(f"  {k:5s} agree {v['agree']:.3f}  stale {v['stale (game file has more games)']:.3f}  missing "
              f"{v['missing (game file has fewer games)']:.3f}  complete {v['game history complete']:.3f}", flush=True)
    X = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                   d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
    targets = {"draft": (d.wins, d.losses), "game": (rec["wins_games"].astype(np.float32),
                                                     rec["losses_games"].astype(np.float32))}
    res, _ = rescore(d, X, has, inp["game_logit"], targets, default_cfg(set_code), blocks=day_blocks(d),
                     log=lambda s: print(s, flush=True))
    res = {"set": set_code, "comparison": cmp, **res, "seconds": time.time() - t0}
    (res_dir / "targets.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["spearman"].items():
        if k.startswith((R4, R10)):
            print(f"  {k:80s} {v:.4f}", flush=True)
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


def selftest():
    """Synthetic drafts with full 7-3 runs; the draft file records a truncated prefix for 20% of drafts and the game
    file misses one game for 10%. Checks the rebuild, the comparison shares, and runs rescore end to end."""
    global THREADS
    THREADS = 2
    from decks.deckstrength import Drafts
    t0 = time.time()
    rng = np.random.default_rng(0)
    D, C = 6000, 30
    X = rng.poisson(0.8, (D, C)).astype(np.int8)
    theta = X @ rng.normal(0, 0.12, C)
    theta = (theta - theta.mean()) / theta.std() * 0.35
    rows, W, L = [], np.zeros(D), np.zeros(D)
    for i in range(D):
        w = l = 0
        while w < 7 and l < 3:
            win = rng.random() < expit(theta[i] - 0.08 * (w - l))
            rows.append((f"d{i:05d}", "0", w + l + 1, "True" if win else "False"))
            w, l = w + win, l + (not win)
        W[i], L[i] = w, l
    games = pl.DataFrame(rows, schema=["draft_id", "build_index", "match_number", "won"], orient="row")
    stale = rng.random(D) < 0.2
    Wd, Ld = W.copy(), L.copy()
    for i in np.flatnonzero(stale):  # draft file = record after the first k games
        k = rng.integers(1, int(W[i] + L[i]))
        y = (games.filter(pl.col("draft_id") == f"d{i:05d}")["won"] == "True").to_numpy()[:k]
        Wd[i], Ld[i] = y.sum(), k - y.sum()
    drop = rng.random(D) < 0.1
    games = games.with_row_index("row").filter(~(pl.col("draft_id").is_in([f"d{i:05d}" for i in np.flatnonzero(drop)])
                                                 & (pl.col("match_number") == 1)))
    split = np.zeros(D, np.int8)
    split[int(0.7 * D):], split[int(0.85 * D):] = 1, 2
    days = np.array([f"2025-01-{1 + i * 28 // D:02d} 12:00:00" for i in range(D)])
    d = Drafts(names=[f"c{j}" for j in range(C)], draft_id=np.array([f"d{i:05d}" for i in range(D)]), draft_time=days,
               pool=X, wins=Wd.astype(np.float32), losses=Ld.astype(np.float32), rank=rng.integers(1, 7, D),
               skill=np.full(D, np.nan), split=split)
    from eval_common import main_builds
    has, _, _ = main_builds(d, games)
    rec = game_records(d, games)
    assert has.all()
    full = ~drop
    assert (rec["wins_games"][full] == W[full]).all() and (rec["n_games"][drop] == (W + L)[drop] - 1).all()
    assert not rec["complete"][drop].any() and rec["complete"][full].all()
    cmp = compare(d, rec, has)
    print(f"[rebuild] test: agree {cmp['test']['agree']:.3f} (true {np.mean(~stale[split == 2] & ~drop[split == 2]):.3f}), "
          f"stale {cmp['test']['stale (game file has more games)']:.3f}, missing "
          f"{cmp['test']['missing (game file has fewer games)']:.3f} [{time.time() - t0:.1f}s]")
    wr, lr = reconcile(d, rec)
    assert (wr[full] == W[full]).all() and (wr[drop & ~stale] == Wd[drop & ~stale]).all()
    Xf = np.hstack([X.astype(np.float32), d.rank[:, None].astype(np.float32)])
    game_logit = theta + rng.normal(0, 0.25, D)
    targets = {"draft": (d.wins, d.losses), "game": (rec["wins_games"].astype(np.float32),
                                                     rec["losses_games"].astype(np.float32))}
    res, _ = rescore(d, Xf, has, game_logit, targets, {"tau": 2.0, "params": {}}, n_seeds=2, n_boot=60,
                     blocks=day_blocks(d), log=lambda s: print(s))
    s_true = {k: spearman_wr(v[d.mask("test")], W[d.mask("test")], L[d.mask("test")])
              for k, v in [("theta", expit(theta))]}
    print(f"[rescore] oracle Spearman vs true full record {s_true['theta']:.3f}; round 4 blend: eval draft "
          f"{res['spearman'][f'{R4} | train draft | eval draft']:.3f}, eval game "
          f"{res['spearman'][f'{R4} | train draft | eval game']:.3f}")
    assert res["spearman"][f"{R4} | train draft | eval game"] > res["spearman"][f"{R4} | train draft | eval draft"]
    json.dumps(res, default=float)
    print(f"self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(*sys.argv[1:])
