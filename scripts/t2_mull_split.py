"""T2 (meta_review_2): champion game FM arms that split the mulligan control into own and opponent terms.

    python scripts/t2_mull_split.py SET [SET ...] [--arms split,own,dl-own] [--refit-arm "- own mull"] [--seeds 2]
           [--threads 4] [--boot 500] [--no-aff] [--no-marginal]
    python scripts/t2_mull_split.py --selftest            (synthetic, CPU 2 threads, < 60 s)
    python scripts/sc_gll.py meta --file t2_mull_split/results.json --subset "experienced pre" SET ...

Built from game_fm_v2 (fit_fm, deck_logit, GRID / FIT, the replayed refit, secondary); torch only, never lightgbm.
Arms (deck part identical to the champion: fm4 nonland + deck features + aff; only the game context changes):
  champion             game_fm_v2's champion (mull = own - opp)
  split mull   (split) own and opponent mulligans as two columns: a neutral re-parameterisation of the champion
                       (the context can now weigh them differently; the deck score's estimand is unchanged)
  - own mull   (own)   opponent mulligans only: the deck gets credit for the mulligans its mana base causes
                       (total effect; meta_review_2 (c))
  rated-dl - own mull  (dl-own) the de-leaked bucket in place of the raw one, and opponent mulligans only
  refit                the champion refit on train+val replaying its schedule, scored with the champion's val (a, b)
  refit [<arm>]        (--refit-arm) the same for another arm, twin-calibrated with that arm
Scoring: t_common contexts raw / dl / pre (dl bucket, no own mulligans: the primary judge for this question) x
sc_gll subsets (experienced primary, all, match 1, main build); keys "<subset> <context>"; pairs arm - champion,
"- own mull" - "split mull", refit - champion; paired day-block bootstraps by seed.
Saved for T3 / T4 (results/{SET}.PremierDraft/t2_mull_split/):
  results.json, scores.npz (per draft, every arm [S, D]), losses.npz
  build_scores.npz  per (draft_id, build_index): build_draft_id, build_index, build_draft, main_build [D], Eb (the
                    build's deck-feature + affinity columns, exactly the FM's), and every arm's deck-only score [S, B]
  card_power.npz    names; per arm (seed 0) w, V (masked pair vectors), the deck-column coefficients; for the champion
                    and "- own mull" the full per-card marginal on training decks, mean over decks containing the card
                    of score(deck) - score(deck minus one copy), and its parts (w, pairs, deck features/affinity)
"""
import os
import sys

_T = "2" if "--selftest" in sys.argv else next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1])
                                                if a == "--threads"), "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game_fm_v2 as g2  # noqa: E402
from t_common import DEFAULT_CONTEXTS, evaluate_ctx, fit_contexts, npz_key, peak_rss_gb, save_losses, split_mull  # noqa: E402
from decks import ROOT  # noqa: E402

CH = g2.CHAMP
ARMS = {"split": "split mull", "own": "- own mull", "dl-own": "rated-dl - own mull"}
DEFAULT_ARMS = ("split", "own")
MARGINAL_ARMS = (CH, "- own mull")
N_MARG = 600


def _no_lightgbm():
    if "lightgbm" in sys.modules:
        raise RuntimeError("lightgbm is loaded in this process; run t2_mull_split.py in its own process")


# ---------------------------------------------------------------- data (game_fm_v2.load_set + mulligans + builds)

def load_set_v2(set_code, use_aff=True):
    """game_fm_v2.load_set, plus own / opponent mulligans per game, the build table (Xb, Eb, game_build, ...) and the
    deck-column function `extra` (for per-card marginals)."""
    import polars as pl
    from mtgdata.cardinfo import card_info, deck_features
    from mtgdata.games import aligned, flag, game_path
    from decks.deckstrength import (N_RANKS, covariates, experience, load_drafts, most_played_deck, record_before,
                                    record_onehot)
    from eval_common import NARROW, main_builds, per_game
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    lf = pl.scan_parquet(path)
    have = lf.collect_schema().names()
    deck_cols = [c for c in have if c.startswith("deck_") and c[5:] in set(d.names)]
    narrow = [c for c in (*NARROW, "opp_rank") if c in have]
    g = (lf.with_row_index("row").select("row", *narrow, *[pl.col(c).cast(pl.Int8) for c in deck_cols])
         .filter(pl.col("draft_id").is_in(pl.Series(list(d.draft_id)).implode()))
         .collect(engine="streaming").sort("row"))
    if "opp_rank" not in g.columns:
        g = g.with_columns(pl.lit(None, pl.Utf8).alias("opp_rank"))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    X = aligned(g, "deck_", d.names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    cov = covariates(g)[:, : 4 + N_RANKS].astype(np.float32)
    assert np.allclose(cov[:, 1], pg["own_mull"] - pg["opp_mull"]), "own - opp != covariates' mulligan difference"
    exp_, rec = experience(g), record_onehot(wb, lb)
    deck, has = most_played_deck(d, g)
    has2, _, main_bix = main_builds(d, g)
    assert (has == has2).all()
    bt = (g.select("draft_id", pl.col("build_index").cast(pl.Utf8)).with_row_index("i")
          .group_by("draft_id", "build_index", maintain_order=True).agg(pl.col("i").first()))
    key = {k: j for j, k in enumerate(zip(bt["draft_id"].to_list(), bt["build_index"].to_list()))}
    game_build = np.array([key[k] for k in zip(g["draft_id"].to_list(), g["build_index"].cast(pl.Utf8).to_list())])
    first = bt["i"].to_numpy()
    del g
    deck = deck.astype(np.float32)
    gsp = pg["split"]
    info = card_info(d.names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    Fg, _ = deck_features(X, info)
    mu, sd = Fg[gsp == 0].mean(0), Fg[gsp == 0].std(0) + 1e-6
    del Fg
    aff = g2.pick_aff(set_code, d.names) if use_aff else None
    if use_aff and aff is None:
        raise SystemExit(f"{set_code}: no picknet pool model; pass --no-aff")
    prof = None
    if aff is not None:
        from deck_synergy_v3 import Profile
        prof = Profile(X, aff[0], nonland, gsp == 0)
    extra = lambda Xd: np.hstack([(deck_features(Xd, info)[0] - mu) / sd]  # noqa: E731
                                 + ([] if prof is None else [prof(Xd, slice(0, 1))])).astype(np.float32)
    E, Ed = extra(X), extra(deck)
    build_draft = pg["gd"][first]
    build_bix = np.asarray(bt["build_index"].to_list(), dtype=object)
    main_build = np.full(len(d.draft_id), -1)
    for j in np.flatnonzero(build_bix == main_bix[build_draft]):
        main_build[build_draft[j]] = j
    cached = None
    inp = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4" / "draft_inputs.npz"
    if inp.exists():
        z = np.load(inp)
        if len(z["has_deck"]) == len(has) and (z["has_deck"] == has).all():
            cached = z["game_logit"].astype(np.float64)
    print(f"=== {set_code}: {len(d.draft_id)} drafts, {len(y)} games, {len(first)} builds, {X.shape[1]} cards, "
          f"{E.shape[1]} deck columns (affinity {'in' if prof is not None else 'omitted'}) [{time.time() - t0:.0f}s]",
          flush=True)
    return {"X": X, "y": y, "gsp": gsp, "gd": pg["gd"], "cov": cov, "exp": exp_, "rec": rec, "E": E, "Ed": Ed,
            "deck": deck, "nonland": nonland, "played": X[gsp == 0].sum(0) > g2.PLAYED_MIN, "d": d, "pg": pg,
            "has": has, "main_bix": main_bix, "cached": cached, "info": {"affinity column": prof is not None},
            "Xb": X[first], "Eb": E[first], "game_build": game_build, "build_draft": build_draft,
            "build_bix": build_bix, "main_build": main_build, "extra": extra,
            "own_mull": pg["own_mull"].astype(np.float32), "opp_mull": pg["opp_mull"].astype(np.float32)}


# ---------------------------------------------------------------- arms

def context_cols_v2(arm, data):
    """game_fm_v2.context_cols for its own variants; split / opponent-only mulligans for the new arms (the mulligan
    difference is column 1 of the context in both the champion and rated-dl bases)."""
    if arm in g2.VARIANTS.values():
        return g2.context_cols(arm, data)
    base = "rated-dl" if arm.startswith("rated-dl") else CH
    c = g2.context_cols(base, data)
    assert np.allclose(c[:, 1], data["cov"][:, 1])
    if arm == "split mull":
        c = np.insert(c, 2, data["opp_mull"].astype(np.float32), axis=1)
        c[:, 1] = data["own_mull"]
    elif arm.endswith("- own mull"):
        c[:, 1] = data["opp_mull"]
    else:
        raise KeyError(arm)
    return c


def fit_arms(data, arms, seeds=2, refit_of=(CH,), fit=g2.FIT, log=print):
    """{arm: {"scores": [S, D], "bscores": [S, B], "w", "model0", ...}}; refit arms replay their base's schedule."""
    X, y, gsp = data["X"], data["y"], data["gsp"]
    tr, va, trva = np.flatnonzero(gsp == 0), np.flatnonzero(gsp == 1), np.flatnonzero(gsp <= 1)
    out = {}

    def finish(name, fits, kw, vbest, base, t1, c):
        sc = np.stack([g2.deck_logit(m, data["deck"], data["Ed"]) for m, _, _ in fits])
        bs = np.stack([g2.deck_logit(m, data["Xb"], data["Eb"]) for m, _, _ in fits]) if "Xb" in data else None
        out[name] = {"scores": sc, "bscores": bs, "w": fits[0][0].w.detach().numpy().copy(), "model0": fits[0][0],
                     "setting": list(kw), "val logloss": float(vbest), "events": [f[2] for f in fits],
                     "epochs": [len(f[2]) for f in fits], "lr drops": [f[2].count("drop") for f in fits],
                     "context columns": int(c.shape[1] - data["E"].shape[1]), "base": base,
                     "fit seconds": time.time() - t1}
        log(f"  {name:22s} val {vbest:.5f} {tuple(kw)} epochs {out[name]['epochs']} [{time.time() - t1:.0f}s]")

    for a in [CH] + [x for x in arms if x != CH]:
        t1 = time.time()
        c = context_cols_v2(a, data)
        grid = [g2.fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=0, fit=fit) + (kw,) for kw in g2.GRID]
        m0, vbest, ev0, kw = min(grid, key=lambda f: f[1])
        fits = [(m0, vbest, ev0)] + [g2.fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=s, fit=fit)
                                     for s in range(1, seeds)]
        finish(a, fits, kw, vbest, a, t1, c)
        del c, grid
    for a in refit_of:
        t1 = time.time()
        c = context_cols_v2(a, data)
        base = out[a]
        fits = [g2.fit_fm(X, c, y, trva, va, *base["setting"], data["nonland"], seed=s, replay=base["events"][s],
                          fit=fit) for s in range(seeds)]
        finish("refit" if a == CH else f"refit [{a}]", fits, base["setting"], np.nan, a, t1, c)
        del c
    return out


def card_marginal(m, data, n_max=N_MARG, seed=0):
    """Per card: mean over training main decks containing it of score(deck) - score(deck - one copy), split into the
    card-power, pair and deck-column (features + affinity) parts. NaN for cards in no training deck."""
    d, has, deck, Ed = data["d"], data["has"], data["deck"], data["Ed"]
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero((d.split == 0) & has)
    coef = m.cov.weight.detach().numpy()[0, -Ed.shape[1]:].astype(np.float64)
    w = m.w.detach().numpy().astype(np.float64)
    with torch.no_grad():
        ds = m.deck_score(torch.from_numpy(deck[tr])).numpy().astype(np.float64)
    C = deck.shape[1]
    out = {k: np.full(C, np.nan) for k in ("total", "w", "pairs", "deck columns", "decks")}
    for c in range(C):
        rows = np.flatnonzero(deck[tr, c] > 0)
        if not len(rows):
            continue
        if len(rows) > n_max:
            rows = np.sort(rng.choice(rows, n_max, replace=False))
        Xm = deck[tr[rows]].copy()
        Xm[:, c] -= 1
        with torch.no_grad():
            ds1 = m.deck_score(torch.from_numpy(Xm)).numpy().astype(np.float64)
        feat = (Ed[tr[rows]].astype(np.float64) - data["extra"](Xm).astype(np.float64)) @ coef
        sd = ds[rows] - ds1
        out["total"][c], out["w"][c] = float(np.mean(sd + feat)), w[c]
        out["pairs"][c], out["deck columns"][c] = float(np.mean(sd) - w[c]), float(np.mean(feat))
        out["decks"][c] = len(rows)
    return out


# ---------------------------------------------------------------- the run

def run(data, arms, seeds=2, refit_of=(CH,), contexts=DEFAULT_CONTEXTS, fit=g2.FIT, n_boot=g2.N_BOOT, marginal=True,
        name="", log=print):
    from sc_gll import subsets
    from eval_common import day_blocks
    t0 = time.time()
    d, pg, has = data["d"], data["pg"], data["has"]
    D = len(d.split)
    data["bt_game"], data["bt_draft"] = g2.deleaked_per_game(pg, D)
    fits = fit_arms(data, arms, seeds, refit_of, fit, log)
    ctxs, cinfo, _ = fit_contexts(pg, D, contexts)
    subs = subsets(pg, has, data["main_bix"])
    scores = {a: f["scores"] for a, f in fits.items()}
    twin = {a: f["base"] for a, f in fits.items() if a.startswith("refit")}
    pairs = [(CH, a) for a in scores if a != CH]
    if "split mull" in scores and "- own mull" in scores:
        pairs.append(("split mull", "- own mull"))
    for a in [a for a in scores if a.startswith("refit [")]:
        pairs.append((fits[a]["base"], a))
    if data.get("cached") is not None:
        scores[g2.CACHED] = data["cached"][None]
        pairs.append((g2.CACHED, CH))
    log("  contexts: " + "  ".join(f"{k} test {v['test logloss (context only)']:.5f}" for k, v in cinfo.items()))
    ev, losses, ngames = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, twin, n_boot, day_blocks(d), name, log)
    sec = g2.secondary({a: {"scores": f["scores"], "w": f["w"]} for a, f in fits.items()}, data, data.get("cached"))
    for a, o in sec.items():
        log(f"  {a:22s} draft spearman {o['draft spearman (calibrated)']:.4f} raw {o['raw spearman - champion']:+.4f}"
            f" strat {o['stratified spearman - champion']:+.4f} sd {o['deck score sd (test drafts)']:.3f}")
    marg = {}
    if marginal and "extra" in data:
        for a in MARGINAL_ARMS:
            if a in fits:
                t1 = time.time()
                marg[a] = card_marginal(fits[a]["model0"], data)
                ok = np.isfinite(marg[a]["total"]) & data["played"]
                log(f"  card marginal {a}: r(total, w) {np.corrcoef(marg[a]['total'][ok], marg[a]['w'][ok])[0, 1]:.3f}"
                    f", sd total/w {np.std(marg[a]['total'][ok]) / np.std(marg[a]['w'][ok]):.2f} [{time.time() - t1:.0f}s]")
    res = {"arms": list(fits), "seeds": seeds, "fit": fit, "grid": g2.GRID, "contexts": cinfo,
           "primary key": "experienced pre", "affinity column": data["info"].get("affinity column"),
           "fits": {a: {k: x for k, x in f.items() if k not in ("scores", "bscores", "w", "model0")} for a, f in fits.items()},
           **ev, "secondary": sec, "seconds": time.time() - t0}
    return res, scores, losses, ngames, fits, marg


def save(out, data, res, scores, losses, ngames, fits, marg):
    out.mkdir(parents=True, exist_ok=True)
    d = data["d"]
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(out / "scores.npz", draft_id=np.asarray(d.draft_id).astype(str), split=d.split,
                        has=data["has"], **{npz_key(a): s.astype(np.float32) for a, s in scores.items()
                                            if a != g2.CACHED})
    save_losses(out / "losses.npz", losses, ngames)
    if "Xb" in data:
        np.savez_compressed(out / "build_scores.npz",
                            build_draft_id=np.asarray(d.draft_id).astype(str)[data["build_draft"]],
                            build_index=data["build_bix"].astype(str), build_draft=data["build_draft"],
                            main_build=data["main_build"], Eb=data["Eb"].astype(np.float32),
                            **{npz_key(a): f["bscores"].astype(np.float32) for a, f in fits.items()})
    cp = {"names": np.asarray(getattr(d, "names", [])).astype(str)}
    for a, f in fits.items():
        m = f["model0"]
        cp[f"{npz_key(a)}|w"] = f["w"].astype(np.float32)
        if m.v is not None:
            cp[f"{npz_key(a)}|V"] = m.v.detach().numpy().astype(np.float32)
        cp[f"{npz_key(a)}|deck_coef"] = m.cov.weight.detach().numpy()[0, -data["Ed"].shape[1]:].astype(np.float32)
    for a, mg in marg.items():
        for k, v in mg.items():
            cp[f"{npz_key(a)}|marginal_{npz_key(k)}"] = v.astype(np.float32)
    np.savez_compressed(out / "card_power.npz", **cp)


def parse(argv):
    sets, arms, seeds, n_boot, aff, threads, refit_arm, marginal = [], [], 2, g2.N_BOOT, True, int(_T), [], True
    it = iter(argv)
    for a in it:
        if a == "--arms":
            arms += next(it).split(",")
        elif a == "--seeds":
            seeds = int(next(it))
        elif a == "--boot":
            n_boot = int(next(it))
        elif a == "--threads":
            threads = int(next(it))
        elif a == "--no-aff":
            aff = False
        elif a == "--no-marginal":
            marginal = False
        elif a == "--refit-arm":
            refit_arm.append(next(it))
        else:
            sets.append(a)
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {list(ARMS)}")
    return sets, [ARMS[x] for x in (arms or DEFAULT_ARMS)], seeds, n_boot, aff, threads, refit_arm, marginal


def main(argv):
    _no_lightgbm()
    sets, arms, seeds, n_boot, aff, threads, refit_arm, marginal = parse(argv)
    torch.set_num_threads(threads)
    for set_code in sets:
        t0 = time.time()
        data = load_set_v2(set_code, aff)
        bad = [a for a in refit_arm if a not in [CH] + arms]
        if bad:
            raise SystemExit(f"--refit-arm {bad} is not among the arms {arms}")
        res, scores, losses, ngames, fits, marg = run(data, arms, seeds, tuple(dict.fromkeys([CH] + refit_arm)),
                                                      n_boot=n_boot, marginal=marginal, name=set_code,
                                                      log=lambda m: print(m, flush=True))
        res = {"set": set_code, "games": int(len(data["y"])), **res, "total seconds": time.time() - t0,
               "peak RSS GB": peak_rss_gb()}
        save(ROOT / "results" / f"{set_code}.PremierDraft" / "t2_mull_split", data, res, scores, losses, ngames, fits,
             marg)
        print(f"{set_code} done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.1f} GB]", flush=True)
        del data


# ---------------------------------------------------------------- synthetic self-test

def synthetic(D=6000, seed=0):
    """game_fm_v2.synthetic's world (skill-leaky card choice, leaky bucket, nonland pairs, record-matched runs), with
    outcomes re-drawn per game from a known model in which OWN mulligans are caused by the land count (a mediator of
    deck strength) and opponent mulligans are exogenous. truth_total = direct effect - 0.6 E[own mull | deck]."""
    data, eta, skill = g2.synthetic(D=D, seed=seed)
    rng = np.random.default_rng(seed + 100)
    deck, gd = data["deck"], data["gd"]
    nl = int(data["nonland"].sum())
    lands = deck[:, nl:].sum(1)
    p_own = np.clip(0.05 + 0.15 * np.abs(lands - 16.5), 0, 0.6)
    G = len(gd)
    own = (rng.random(G) < p_own[gd]).astype(float)
    opp = (rng.random(G) < 0.12).astype(float)
    pg = data["pg"]
    z = (0.15 + eta[gd] + skill[gd] + 0.1 * pg["on_play"] - 0.12 * (pg["wins_before"] - pg["losses_before"])
         - 0.6 * own + 0.6 * opp)
    y = (rng.random(G) < 1 / (1 + np.exp(-z))).astype(np.int8)
    data["y"], pg["y"] = y, y.astype(float)
    pg["mull"], pg["own_mull"], pg["opp_mull"] = own - opp, own, opp
    data["own_mull"], data["opp_mull"] = own.astype(np.float32), opp.astype(np.float32)
    data["cov"][:, 1] = (own - opp).astype(np.float32)
    sp = data["d"].split
    feats = lambda Xd: np.stack([Xd[:, nl:].sum(1), (Xd[:, nl:].sum(1) - 16.5) ** 2, Xd[:, :nl].sum(1)], 1)  # noqa: E731
    Fd = feats(deck)
    mu, sd = Fd[sp == 0].mean(0), Fd[sp == 0].std(0) + 1e-6
    data["extra"] = lambda Xd: ((feats(Xd) - mu) / sd).astype(np.float32)
    data["Ed"] = data["extra"](deck)
    data["E"] = data["Ed"][gd]
    data["Xb"], data["Eb"] = deck, data["Ed"]
    data["game_build"], data["build_draft"] = gd, np.arange(D)
    data["build_bix"], data["main_build"] = np.full(D, "0", dtype=object), np.arange(D)
    data["d"].names = [f"c{j}" for j in range(deck.shape[1])]
    total = eta - 0.6 * p_own
    return data, eta, total


def selftest():
    import tempfile
    _no_lightgbm()
    torch.set_num_threads(2)
    t0 = time.time()
    data, eta, total = synthetic()
    fit = {**g2.FIT, "batch": 1024, "epochs": 25}
    res, scores, losses, ngames, fits, marg = run(data, ["split mull", "- own mull"], seeds=2,
                                                  refit_of=(CH, "- own mull"), fit=fit, n_boot=50, name="SYN")
    te = data["d"].split == 2
    r = lambda s: float(np.corrcoef(s[te], total[te])[0, 1])  # noqa: E731
    rd = lambda s: float(np.corrcoef(s[te], eta[te])[0, 1])  # noqa: E731
    print("[1] r with the TOTAL deck effect (incl. mulligans caused by the mana base) / with the direct effect: "
          + "  ".join(f"{a} {r(s[0]):.3f}/{rd(s[0]):.3f}" for a, s in scores.items()))
    assert r(scores["- own mull"][0]) > r(scores[CH][0]) - 0.005
    wc, ws = fits[CH]["w"][data["played"]], fits["split mull"]["w"][data["played"]]
    assert np.corrcoef(wc, ws)[0, 1] > 0.9, "split mull should re-parameterise the champion"
    for p in ("- own mull - champion", "split mull - champion", "refit - champion", "- own mull - split mull",
              "refit [- own mull] - - own mull"):
        for k in ("experienced pre", "experienced dl", "experienced raw", "match 1 pre"):
            x = res["pairs"][p][k]
            assert np.isfinite(x["delta"]) and x["se"] > 0 and x["n_seeds"] == 2, (p, k)
    print(f"[2] - own mull - champion: pre {res['pairs']['- own mull - champion']['experienced pre']['delta']:+.3f}, "
          f"raw {res['pairs']['- own mull - champion']['experienced raw']['delta']:+.3f} (x1000)")
    assert res["models"]["refit [- own mull]"]["experienced pre"]["calibration"] == "twin: - own mull"
    assert fits["refit"]["epochs"] == fits[CH]["epochs"]
    m = marg[CH]
    ok = np.isfinite(m["total"])
    assert np.allclose(m["total"][ok], (m["w"] + m["pairs"] + m["deck columns"])[ok])
    with tempfile.TemporaryDirectory() as tmp:
        save(Path(tmp), data, {"set": "SYN", **res}, scores, losses, ngames, fits, marg)
        b = np.load(Path(tmp) / "build_scores.npz")
        assert b["champion"].shape == (2, len(eta)) and np.allclose(b["champion"], scores[CH], atol=1e-5)
        cp = np.load(Path(tmp) / "card_power.npz")
        assert "champion|marginal_total" in cp.files and "-_own_mull|V" in cp.files
        json.loads((Path(tmp) / "results.json").read_text())
    assert parse(["MSH", "--arms", "own,dl-own"])[1] == ["- own mull", "rated-dl - own mull"]
    print(f"t2 self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(sys.argv[1:])
