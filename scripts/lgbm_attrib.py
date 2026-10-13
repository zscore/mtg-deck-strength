"""What does LightGBM add beyond the game FM in blend*?   python scripts/lgbm_attrib.py SET [SET ...] | meta SET ...

blend* = (1 - a) z_LGBM + a z_champion (T1; a tuned on VAL SC-GLL-dl). Its departure from the FM is (1 - a) * d with
    d = z_LGBM - z_FM   (per LightGBM seed s, FM seed s mod 2, z on train drafts' main builds; mean over the 3 seeds)
Part 1 -- attribution of d on TEST drafts' main builds (d centred). Feature groups:
  structure   improve_deck.structure_features (land count one-hot 14..19+, curve bins, creatures, # main colours, splash)
  deck feats  the 17 Scryfall deck features (draft_inputs.npz; the FM's own deck columns)
  affinity    PickNet pool affinity (draft_inputs.npz)
  rank        the drafter's rank one-hot (LightGBM is scored at platinum: should be ~0)
  player      de-leaked skill bucket (binned + linear + missing) and user_n_games bucket one-hot (player info that
              LightGBM, trained on the draft record, might proxy through the deck)
  pool        draft-level pool features: on-colour nonland cards in the pool, of them not in the deck (sideboard
              depth), rares+mythics in pool / in deck / left out, the champion's card power (w) of the best 23
              on-colour pool nonlands vs the deck's nonland w sum (build gap)
  card counts the deck's card counts (high-dimensional: ridge)
  Out-of-fold (5-fold) ridge R^2 for: every group alone, all groups, all minus each group (partial R^2 = drop in
  R^2). Also the correlation of d (and of z_LGBM, z_FM) with the de-leaked / raw skill bucket within user_n_games
  strata (pooled within-stratum r = r of stratum-demeaned values).
Part 2 -- locate the gain: refit LightGBM (round 4 settings, seed 0) on feature subsets, blend each with the champion
  (seed 0) exactly as T1 (t1_rescore.blend_star: a on VAL SC-GLL-dl, two-half cross-fit), score SC-GLL experienced
  (dl context) and match 1 (dl) on test:
    full        counts + structure + rank + deck feats + aff (round 4 recipe, scored at platinum)
    - rank      counts + structure + deck feats + aff
    summaries   structure + deck feats + aff (no counts, no rank)
    structure   structure only
    counts      deck counts only
  pairs: blend[v] - champion, blend[v] - blend[full], LightGBM[v] - champion.
Process: torch-free shim + lightgbm (never torch), 4 threads.
Writes results/{SET}.PremierDraft/lgbm_attrib/results.json; meta -> results/meta/lgbm_attrib.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, assert_no_torch, peak_rss_gb, torchfree  # noqa: E402

torchfree()
if __name__ == "__main__" and sys.argv[1:2] != ["meta"]:
    import lightgbm  # noqa: F401
import numpy as np  # noqa: E402

THREADS = 4
CH, LG = "champion", "LightGBM"
T1_BLEND = "blend* (LightGBM + champion)"
VARIANTS = ("full", "- rank", "summaries", "structure", "counts")
GROUPS = ("structure", "deck feats", "affinity", "rank", "player", "pool", "card counts")
SKILL_BINS = np.arange(0.40, 0.70, 0.04)


# ---------------------------------------------------------------- data

def load(set_code, log=print):
    from eval_common import NARROW, deck_rows, load_games, main_builds, per_game
    from mtgdata.games import game_path
    from t_common import game_blocks, split_mull
    from decks.deckstrength import load_drafts
    from improve_deck import structure_features
    t0 = time.time()
    res_dir = REPO / "results" / f"{set_code}.PremierDraft"
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    has, rows, main_bix = main_builds(d, g)
    D = len(d.draft_id)
    blocks = game_blocks(g, pg, D)
    del g
    ref = np.load(res_dir / "sc_gll" / "scores.npz")
    assert (ref["draft_id"].astype(str) == np.asarray(d.draft_id).astype(str)).all() and (ref["has"] == has).all()
    c = np.load(res_dir / "sc_gll" / "context.npz")
    assert (c["gd"] == pg["gd"]).all()
    deck = np.zeros((D, len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    inp = np.load(res_dir / "deck_synergy_v4" / "draft_inputs.npz")
    assert (inp["has_deck"] == has).all()
    sf = structure_features(deck, d.names, set_code)
    g2 = np.load(res_dir / "game_fm_v2" / "scores.npz")
    t1 = json.loads((res_dir / "t1_rescore" / "results.json").read_text())
    log(f"=== {set_code}: {D} drafts, {len(pg['y'])} games [{time.time() - t0:.0f}s]")
    return dict(res_dir=res_dir, d=d, pg=pg, D=D, has=has, main_bix=main_bix, blocks=blocks, deck=deck, inp=inp,
                sf=sf, L=ref["LightGBM"].astype(np.float64), F=g2["champion"].astype(np.float64),
                alphas=[float(a) for a in t1["blend*"][T1_BLEND]["alphas"]],
                cp=np.load(res_dir / "t2_mull_split" / "card_power.npz"))


def pool_features(S, set_code, idx):
    """Pool features for drafts idx (test main builds)."""
    from mtgdata.data import COLORS, card_metadata
    d, deck = S["d"], S["deck"][idx].astype(np.float32)
    meta = card_metadata(d.names, set_code)
    land = meta["types"].str.contains("Land").to_numpy()
    col = meta.select(list(COLORS)).to_numpy().astype(bool)
    rare = meta["rarity"].is_in(["rare", "mythic"]).to_numpy()
    nl = (~land).astype(np.float32)
    per_color = (deck * nl) @ col.astype(np.float32)
    main = per_color >= 4
    top2 = np.argsort(-per_color, 1)[:, :2]
    fb = np.zeros_like(main)
    np.put_along_axis(fb, top2, True, 1)
    main = np.where(main.sum(1, keepdims=True) == 0, fb, main)
    # a card is on-colour iff its colour identity is inside the deck's main colours (colourless always)
    bits = 1 << np.arange(len(COLORS))
    cmask, mmask = col.astype(int) @ bits, main.astype(int) @ bits
    onc = ((cmask[None, :] & ~mmask[:, None]) == 0) & ~land[None, :]
    pool = d.pool[idx].astype(np.float32)
    pool_on = (pool * onc).sum(1)
    deck_on = (deck * onc).sum(1)
    names_cp = [str(x) for x in S["cp"]["names"]]
    assert names_cp == list(d.names), "card_power names differ"
    w = S["cp"]["champion|w"].astype(np.float32)
    pw = np.where(onc, w[None, :], -np.inf)
    cnt = np.minimum(pool, 4).astype(int)
    best = np.zeros(len(pool), np.float32)
    order = np.argsort(-pw, 1)
    for i in range(len(pool)):
        o = order[i]
        reps = np.repeat(pw[i, o], cnt[i, o])
        reps = reps[np.isfinite(reps)][:23]
        best[i] = reps.sum()
    deck_w = (deck * nl) @ w
    X = np.column_stack([pool_on, pool_on - deck_on, (pool * rare).sum(1), (deck * rare).sum(1),
                         ((pool - deck).clip(0) * rare).sum(1), best, deck_w, best - deck_w])
    names = ["on-colour pool nonlands", "on-colour nonlands left out", "rares+mythics in pool",
             "rares+mythics in deck", "rares+mythics left out", "best-23 on-colour pool power (w)",
             "deck nonland power (w)", "build gap (best-23 - deck)"]
    return X.astype(np.float64), names


def player_features(S):
    from t_common import dl_bucket, per_draft
    pg, D = S["pg"], S["D"]
    _, bt = dl_bucket(pg, D)
    nb = per_draft(pg, D, "n_bucket")
    raw = per_draft(pg, D, "skill")
    known = np.isfinite(bt)
    b = np.where(known, np.digitize(np.nan_to_num(bt), SKILL_BINS) + 1, 0)
    lv = sorted({float(x) for x in nb[np.isfinite(nb)]})
    nbi = np.array([lv.index(float(x)) + 1 if np.isfinite(x) else 0 for x in nb])
    X = np.hstack([np.eye(len(SKILL_BINS) + 2)[b], np.where(known, bt - 0.55, 0)[:, None],
                   np.eye(len(lv) + 1)[nbi]])
    return X, bt, raw, nb


# ---------------------------------------------------------------- part 1

def zstd(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-12)


def oof_r2(X, y, alphas=(0.1, 1, 10, 100, 1000, 1e4), k=5, seed=0):
    from sklearn.linear_model import RidgeCV
    n = len(y)
    fold = np.random.default_rng(seed).permutation(n) % k
    mu, sd = X.mean(0), X.std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1)
    pred = np.zeros(n)
    for f in range(k):
        tr = fold != f
        m = RidgeCV(alphas=alphas).fit(X[tr], y[tr])
        pred[~tr] = m.predict(X[~tr])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def within_strata_r(x, y, strata):
    ok = np.isfinite(x) & np.isfinite(y)
    out, xs, ys = {}, np.zeros(ok.sum()), np.zeros(ok.sum())
    xo, yo, so = x[ok], y[ok], strata[ok]
    for s in np.unique(so):
        m = so == s
        if m.sum() >= 50:
            out[str(int(s)) if np.isfinite(s) else "nan"] = {"n": int(m.sum()), "r": float(np.corrcoef(xo[m], yo[m])[0, 1])}
        xs[m], ys[m] = xo[m] - xo[m].mean(), yo[m] - yo[m].mean()
    return {"pooled within-stratum r": float(np.corrcoef(xs, ys)[0, 1]), "n": int(ok.sum()), "by stratum": out}


def attribution(S, set_code, log=print):
    from decks.deckstrength import N_RANKS
    d, has, D = S["d"], S["has"], S["D"]
    ref = has & (d.split == 0)
    te = np.flatnonzero(has & (d.split == 2))
    zl = np.stack([zstd(s, ref) for s in S["L"]])
    zf = np.stack([zstd(S["F"][s % len(S["F"])], ref) for s in range(len(zl))])
    dd = (zl - zf).mean(0)
    y = dd[te] - dd[te].mean()
    Xp, pnames = pool_features(S, set_code, te)
    Xpl, bt, raw, nb = player_features(S)
    groups = {"structure": S["sf"][te], "deck feats": S["inp"]["deck_features"][te],
              "affinity": S["inp"]["aff"][te][:, None], "rank": np.eye(N_RANKS)[d.rank[te]],
              "player": Xpl[te], "pool": Xp, "card counts": S["deck"][te].astype(np.float64)}
    t0 = time.time()
    full = np.hstack([groups[g] for g in GROUPS])
    r2 = {"all": oof_r2(full, y)}
    alone, partial = {}, {}
    for g in GROUPS:
        alone[g] = oof_r2(groups[g], y)
        partial[g] = r2["all"] - oof_r2(np.hstack([groups[h] for h in GROUPS if h != g]), y)
    no_counts = [g for g in GROUPS if g != "card counts"]
    r2["all but card counts"] = oof_r2(np.hstack([groups[g] for g in no_counts]), y)
    deck_side = ["structure", "deck feats", "affinity", "card counts"]
    player_side = ["rank", "player", "pool"]
    r2["deck side (structure, deck feats, aff, counts)"] = oof_r2(np.hstack([groups[g] for g in deck_side]), y)
    r2["player side (rank, player, pool)"] = oof_r2(np.hstack([groups[g] for g in player_side]), y)
    r2["player side beyond deck side"] = r2["all"] - r2["deck side (structure, deck feats, aff, counts)"]
    r2["deck side beyond player side"] = r2["all"] - r2["player side (rank, player, pool)"]
    log(f"  d = z_LGBM - z_FM on {len(te)} test main builds: sd {y.std():.3f}; OOF ridge R^2 all {r2['all']:.3f} "
        f"[{time.time() - t0:.0f}s]")
    log("    alone:   " + "  ".join(f"{g} {v:.3f}" for g, v in alone.items()))
    log("    partial: " + "  ".join(f"{g} {v:+.3f}" for g, v in partial.items()))
    log(f"    deck side {r2['deck side (structure, deck feats, aff, counts)']:.3f}  player side "
        f"{r2['player side (rank, player, pool)']:.3f}  player beyond deck {r2['player side beyond deck side']:+.3f}")
    # interpretable linear coefficients of the small groups (standardised, OLS on structure + pool + player)
    sm = np.hstack([groups["structure"], groups["pool"], np.where(np.isfinite(bt[te]), bt[te], np.nanmean(bt[te]))[:, None]])
    sm_names = ([f"lands {k}" for k in range(14, 20)] + ["mv<=1 /5", "mv2 /5", "mv3 /5", "mv4 /5", "mv5 /5", "mv6+ /5",
                "creatures /10", "1 colour", "2 colours", "3 colours", "4+ colours", "splash colours"] + pnames
                + ["de-leaked skill"])
    r_each = {n: float(np.corrcoef(sm[:, j], y)[0, 1]) if sm[:, j].std() > 0 else 0.0 for j, n in enumerate(sm_names)}
    nbt = nb[te]
    corr = {}
    for nm, v in (("d", dd[te]), ("z_LGBM", zl.mean(0)[te]), ("z_FM", zf.mean(0)[te])):
        corr[nm] = {"de-leaked skill": within_strata_r(v, bt[te], nbt), "raw skill": within_strata_r(v, raw[te], nbt)}
    log("    within-experience r with de-leaked skill: " + "  ".join(
        f"{k} {v['de-leaked skill']['pooled within-stratum r']:+.3f}" for k, v in corr.items())
        + " | raw: " + "  ".join(f"{k} {v['raw skill']['pooled within-stratum r']:+.3f}" for k, v in corr.items()))
    top = sorted(r_each.items(), key=lambda kv: -abs(kv[1]))[:8]
    log("    top single-feature r with d: " + "  ".join(f"{k} {v:+.3f}" for k, v in top))
    return {"test main builds": int(len(te)), "sd of d": float(y.std()), "r2": r2, "alone": alone, "partial": partial,
            "single-feature r with d": r_each, "skill correlations": corr,
            "mean (1 - alpha*)": float(1 - np.mean(S["alphas"]))}


# ---------------------------------------------------------------- part 2

def variant_X(S, v):
    from eval_common import REF_RANK
    d = S["d"]
    deck, sf, df, aff = S["deck"].astype(np.float32), S["sf"], S["inp"]["deck_features"], S["inp"]["aff"][:, None]
    rank = d.rank[:, None].astype(np.float32)
    parts = {"full": [deck, sf, rank, df, aff], "- rank": [deck, sf, df, aff], "summaries": [sf, df, aff],
             "structure": [sf], "counts": [deck]}[v]
    X = np.hstack(parts).astype(np.float32)
    Xr = None
    if v == "full":
        Xr = X.copy()
        Xr[:, deck.shape[1] + sf.shape[1]] = REF_RANK
        assert (X[:, deck.shape[1] + sf.shape[1]] == d.rank).all()
    return X, Xr


def locate(S, log=print):
    from eval_common import lgbm_model
    from sc_gll import subsets
    from t_common import evaluate_ctx, fit_contexts
    from t1_rescore import blend_star
    d, pg, D, has = S["d"], S["pg"], S["D"], S["has"]
    split = np.full(D, -1, np.int8)
    split[pg["gd"]] = pg["split"]
    ctxs, _, _ = fit_contexts(pg, D, ("dl",))
    subs = subsets(pg, has, S["main_bix"])
    fit_val = subs["experienced"][0]
    tr, va = d.mask("train") & has, d.mask("val") & has
    F0 = S["F"][:1]
    scores, info = {CH: F0}, {}
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    for v in VARIANTS:
        t1 = time.time()
        X, Xr = variant_X(S, v)
        m, n = lgbm_model(X, d, tr, va, params={"num_threads": THREADS}, seed=0)
        p = m.predict(X if Xr is None else Xr, num_iteration=n)
        L = z(p)[None]
        del X, Xr
        bl, bi = blend_star(L, F0, pg, ctxs["dl"], fit_val, D, has, split, log=lambda m_: None)
        scores[f"LightGBM [{v}]"], scores[f"blend [{v}]"] = L, bl
        info[v] = {"trees": int(n), "alpha": bi["alphas"][0], "features": int(m.num_feature()),
                   "r with saved LightGBM seed 0 (test)": float(np.corrcoef(L[0][d.split == 2], S["L"][0][d.split == 2])[0, 1])}
        log(f"  lgbm [{v}]: {n} trees, {m.num_feature()} features, alpha* {bi['alphas'][0]:.2f} [{time.time() - t1:.0f}s]")
    pairs = ([(CH, f"blend [{v}]") for v in VARIANTS] + [("blend [full]", f"blend [{v}]") for v in VARIANTS[1:]]
             + [(CH, f"LightGBM [{v}]") for v in VARIANTS])
    use = {"experienced": subs["experienced"], "match 1": subs["match 1"]}
    res, _, _ = evaluate_ctx(scores, pg, {"dl": ctxs["dl"]}, use, D, pairs, None, 300, S["blocks"], "", log)
    res["variants"] = info
    return res


def card_level(S, log=print):
    """Part 3: d's linear card coefficients (ridge of d on the counts of test main builds) against clean card values.
    Deck signal predicts c ~ hand value (HV, randomised opening hands: skill-clean) beyond the FM's own w; a player
    proxy predicts c ~ deck_skill (mean skill bucket of the card's decks) holding HV and w fixed."""
    import polars as pl
    from sklearn.linear_model import RidgeCV
    d, has = S["d"], S["has"]
    f = S["res_dir"] / "hand_value_v2" / "card_values.csv"
    if not f.exists():
        return None
    ref = has & (d.split == 0)
    te = np.flatnonzero(has & (d.split == 2))
    zl = np.stack([zstd(s, ref) for s in S["L"]])
    zf = np.stack([zstd(S["F"][s % len(S["F"])], ref) for s in range(len(zl))])
    y = (zl - zf).mean(0)[te]
    X = S["deck"][te].astype(np.float64)
    m = RidgeCV(alphas=(1, 3, 10, 30, 100, 300, 1000)).fit(X, y)
    c = m.coef_
    cv = pl.read_csv(f)
    pos = {n: i for i, n in enumerate(cv["card"].to_list())}
    assert set(pos) == set(d.names)
    cv = cv[[pos[n] for n in d.names]]
    n_te = (X > 0).sum(0)
    hv = cv["train_hv_logit_shrunk"].to_numpy()
    ok = (~cv["is_land"].to_numpy()) & np.isfinite(hv) & (n_te >= 200) & np.isfinite(cv["deck_skill"].to_numpy())
    cols = {"HV (shrunk)": hv, "FM w": cv["fm_w"].to_numpy(), "deck_skill": cv["deck_skill"].to_numpy(),
            "GIH WR": cv["gih_wr"].to_numpy()}
    r = {k: float(np.corrcoef(c[ok], v[ok])[0, 1]) for k, v in cols.items()}
    Z = np.column_stack([(v[ok] - v[ok].mean()) / v[ok].std() for v in (cols["HV (shrunk)"], cols["FM w"],
                                                                         cols["deck_skill"])])
    cz = (c[ok] - c[ok].mean()) / c[ok].std()
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), Z]), cz, rcond=None)
    res = cz - np.column_stack([np.ones(ok.sum()), Z]) @ beta
    sig2 = res @ res / (ok.sum() - 4)
    cov = sig2 * np.linalg.inv(np.column_stack([np.ones(ok.sum()), Z]).T @ np.column_stack([np.ones(ok.sum()), Z]))
    out = {"cards": int(ok.sum()), "ridge alpha": float(m.alpha_), "r(c, x)": r,
           "std beta (c ~ HV + FM w + deck_skill)": dict(zip(("HV (shrunk)", "FM w", "deck_skill"), beta[1:].tolist())),
           "se": dict(zip(("HV (shrunk)", "FM w", "deck_skill"), np.sqrt(np.diag(cov))[1:].tolist())),
           "r(HV, deck_skill)": float(np.corrcoef(cols["HV (shrunk)"][ok], cols["deck_skill"][ok])[0, 1]),
           "r(FM w, HV)": float(np.corrcoef(cols["FM w"][ok], cols["HV (shrunk)"][ok])[0, 1])}
    log(f"  card level ({ok.sum()} cards): r(c, HV) {r['HV (shrunk)']:+.3f}  r(c, FM w) {r['FM w']:+.3f}  r(c, deck_skill) "
        f"{r['deck_skill']:+.3f}  r(c, GIH) {r['GIH WR']:+.3f} | std beta " + "  ".join(
            f"{k} {b:+.3f}±{out['se'][k]:.3f}" for k, b in out["std beta (c ~ HV + FM w + deck_skill)"].items()))
    return out


# ---------------------------------------------------------------- run / meta

def run_set(set_code, log=print):
    t0 = time.time()
    S = load(set_code, log)
    out = {"set": set_code, "alphas (T1)": S["alphas"]}
    out["attribution"] = attribution(S, set_code, log)
    out["card level"] = card_level(S, log)
    log(f"  part 1 done [{time.time() - t0:.0f}s]")
    out["locate"] = locate(S, log)
    out.update({"total seconds": time.time() - t0, "peak rss GB": peak_rss_gb()})
    dst = S["res_dir"] / "lgbm_attrib"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "results.json").write_text(json.dumps(out, indent=1, default=float))
    assert_no_torch()
    log(f"{set_code} done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")


def meta(*sets):
    from eval_protocol import DecisionRule, decide, meta_analysis
    files = {s: REPO / "results" / f"{s}.PremierDraft" / "lgbm_attrib" / "results.json" for s in sets}
    files = {s: f for s, f in files.items() if f.exists()}
    R = {s: json.loads(f.read_text()) for s, f in files.items()}
    A = {s: r["attribution"] for s, r in R.items()}
    mean = lambda xs: (float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(len(xs))))  # noqa: E731
    out = {"sets": list(R), "attribution": {}, "locate": {}}
    at = out["attribution"]
    at["r2"] = {k: mean([a["r2"][k] for a in A.values()]) for k in next(iter(A.values()))["r2"]}
    at["alone"] = {g: mean([a["alone"][g] for a in A.values()]) for g in GROUPS}
    at["partial"] = {g: mean([a["partial"][g] for a in A.values()]) for g in GROUPS}
    at["single-feature r with d"] = {k: mean([a["single-feature r with d"][k] for a in A.values()])
                                     for k in next(iter(A.values()))["single-feature r with d"]}
    at["skill r (pooled within experience)"] = {
        f"{nm} | {sk}": mean([a["skill correlations"][nm][sk]["pooled within-stratum r"] for a in A.values()])
        for nm in ("d", "z_LGBM", "z_FM") for sk in ("de-leaked skill", "raw skill")}
    at["sd of d"] = mean([a["sd of d"] for a in A.values()])
    print("R^2 (mean ± se over sets):", {k: f"{v[0]:.3f}±{v[1]:.3f}" for k, v in at["r2"].items()})
    print("alone:", {k: f"{v[0]:.3f}" for k, v in at["alone"].items()})
    print("partial:", {k: f"{v[0]:+.3f}±{v[1]:.3f}" for k, v in at["partial"].items()})
    print("skill r:", {k: f"{v[0]:+.3f}±{v[1]:.3f}" for k, v in at["skill r (pooled within experience)"].items()})
    print("single r:", {k: f"{v[0]:+.3f}" for k, v in sorted(at["single-feature r with d"].items(), key=lambda kv: -abs(kv[1][0]))[:10]})
    pairs = sorted({p for r in R.values() for p in r["locate"]["pairs"]})
    for p in pairs:
        for key in ("experienced dl", "match 1 dl"):
            rs = _rows(R, p, key)
            if len(rs) < 2:
                continue
            if any(r_.se < 1e-6 for r_ in rs):   # alpha* = 1 in some sets: blend == champion exactly (se 0)
                dl_ = np.array([r_.delta for r_ in rs])
                out["locate"].setdefault(p, {})[key] = {"unweighted mean": float(dl_.mean()),
                                                        "se": float(dl_.std(ddof=1) / np.sqrt(len(dl_))),
                                                        "sets better": f"{int((dl_ > 0).sum())}/{len(rs)}",
                                                        "sets identical (alpha* = 1)": int(sum(r_.se < 1e-6 for r_ in rs))}
                print(f"{p:45s} {key:15s} unweighted {dl_.mean():+.4f} ± {dl_.std(ddof=1) / np.sqrt(len(dl_)):.4f} "
                      f"{int((dl_ > 0).sum())}/{len(rs)} ({int(sum(r_.se < 1e-6 for r_ in rs))} sets alpha*=1)")
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out["locate"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci), "p": m.re_p_two_sided,
                                                    "sets better": f"{m.n_positive}/{len(rs)}", "sig worse": m.n_sig_worse,
                                                    "adopt": ok}
            print(f"{p:45s} {key:15s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] p={m.re_p_two_sided:.4f} "
                  f"{m.n_positive}/{len(rs)}")
    C = [r["card level"] for r in R.values() if r.get("card level")]
    out["card level"] = {"r(c, x)": {k: mean([c["r(c, x)"][k] for c in C]) for k in C[0]["r(c, x)"]},
                         "std beta": {k: mean([c["std beta (c ~ HV + FM w + deck_skill)"][k] for c in C])
                                      for k in C[0]["std beta (c ~ HV + FM w + deck_skill)"]},
                         "sets": len(C)}
    print("card level:", out["card level"])
    out["alphas"] = {v: {s: r["locate"]["variants"][v]["alpha"] for s, r in R.items()} for v in VARIANTS}
    out["total seconds"] = sum(r["total seconds"] for r in R.values())
    dst = REPO / "results" / "meta" / "lgbm_attrib.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float))


def _rows(R, pair, key):
    from eval_protocol import SetResult
    out = []
    for s, r in R.items():
        x = r["locate"]["pairs"].get(pair, {}).get(key)
        if x:
            out.append(SetResult(**{k: x[k] for k in ("delta", "se", "se_boot", "se_seed", "n_units", "n_seeds")}, name=s))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        for s in sys.argv[1:]:
            run_set(s, log=lambda m: print(m, flush=True))
