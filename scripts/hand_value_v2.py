"""Hand value v2 (meta_review_2 T4): hand_value.py with the deck-model bug fixed and an external rating test.

    python scripts/hand_value_v2.py SET [SET ...]      (~ hand_value's 2-18 min per set, < 2 GB)
    python scripts/hand_value_v2.py meta SET ...       (12-set meta of the sum-of-ratings pairs and the prior arms)
    python scripts/hand_value_v2.py --selftest         (synthetic, < 60 s)

Changes from hand_value.py (everything else -- the Robinson / known-propensity estimator, cross-fitted p_hat,
balance index, core subset, reliability, ratings-vs-test-HV table, skill tilt -- is hand_value's code, imported):
  1. BUG FIX  the round 4 game FM deck score (and its has-deck flag) stays a precision covariate of the outcome model
              p_hat only. The DECK MODEL whose card power w is regressed on HV, and every prior arm, are fit WITHOUT it
              (Data.n_precision trailing dense columns are dropped for them). Before, w was the residual card power
              beyond the FM's deck score, which forced the slope to ~0 and lambda < 0 (meta_review_2 (e)).
  2. ridge grid LAMS widened to 3e-7 ... 3e-1 (the old grid's upper edge 3e-2 was chosen in 11/12 sets); the chosen
              lambda's position on the grid is reported (edge = flag).
  3. FM slope  if T2 ran (t2_mull_split/card_power.npz), HV is also compared with the champion FM's card power w and
              with its FULL per-card marginal (w + pair terms + deck-feature/affinity terms, averaged over training
              decks containing the card): slope, EIV-corrected slope and implied lambda for each.
  4. EXTERNAL sum-of-ratings test (non-circular): every train-game rating (HV shrunk, HV raw, GIH WR, OH WR, GP WR,
              GD WR, IWD, clean deck-model w, and FM w / FM marginal when available), centred (games-weighted) over
              spells with a value, missing -> 0, lands -> 0; deck score of a draft = sum over its most-played build's
              spells of copies x rating. Scored with SC-GLL under the dl and raw contexts (t_common), (a, b) on val
              games, test games; paired day-block bootstraps "<rating> sum - HV shrunk sum" etc. in sc_gll's meta
              format (hand_value_v2/sum_ratings.json). Decision (meta_review_2 T4): HV sum - GIH WR sum >= +0.02 on
              "experienced dl" with decide() -> the "HV is the best card rating" claim becomes external.
  Not done: the selection-free turn-1/2-draw HV variant -- the game file's drawn_ columns are not split by turn.
Writes results/{SET}.PremierDraft/hand_value_v2/{card_values.csv, results.json, test_losses.npz, sum_ratings.json,
sum_losses.npz}. numpy/scipy/polars only: torch-free decks.deckstrength shim, no lightgbm.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, assert_no_torch, torchfree  # noqa: E402

torchfree()
import numpy as np  # noqa: E402

import hand_value as hv1  # noqa: E402
from hand_value import (MIN_TEST, MIN_TRAIN, Data, center_shrink, corr, fit_logit, hand_values, logloss,  # noqa: E402
                        outcome_residuals, stream_sparse, win_rates)

LAMS = (3e-7, 3e-6, 3e-5, 3e-4, 3e-3, 3e-2, 3e-1)
N_BOOT = 500
SUM_KEYS = ("experienced dl", "experienced raw", "match 1 dl", "all dl")


def deck_view(data: Data) -> Data:
    """The same Data without the precision-only dense columns (shallow copy, no sparse recomputation)."""
    k = data.dense.shape[1] - getattr(data, "n_precision", 0)
    v = copy.copy(data)
    v.dense = data.dense[:, :k]
    return v


# ---------------------------------------------------------------------------------------------------- data

def load_set_v2(set_code: str, event: str = "PremierDraft"):
    """hand_value.load_set, keeping which dense columns are precision-only, plus what the external test needs:
    per-game arrays (pg, with own / opponent mulligans), the main build of each draft, day blocks."""
    import polars as pl
    from mtgdata.games import game_path
    from decks.deckstrength import N_RANKS, load_drafts
    from eval_common import NARROW, build_table, load_games, per_game
    from t_common import game_blocks, split_mull
    d = load_drafts(set_code, event)
    path = game_path(set_code, event)
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    Xb, gb, bd, main = build_table(path, d, g)
    rows = g["row"].to_numpy().astype(np.int64)
    H = stream_sparse(path, rows, d.names, "opening_hand_")
    Dr = stream_sparse(path, rows, d.names, "drawn_")
    try:
        from mtgdata.cardinfo import card_info
        is_land = card_info(d.names, set_code)["is_land"].to_numpy()
    except Exception as ex:  # noqa: BLE001
        print(f"  card_info failed ({ex}); lands = basics only", flush=True)
        is_land = np.array([n in hv1.BASICS for n in d.names])
    own_mull = np.clip(np.nan_to_num(g["num_mulligans"].cast(pl.Float64, strict=False).to_numpy()), 0, 2).astype(int)
    skill = pg["skill"]
    sk = np.where(np.isnan(skill), np.nanmean(skill), skill) - 0.5
    dense = [pg["on_play"], pg["mull"], sk, np.isnan(skill).astype(float)]
    n_prec = 0
    f4 = REPO / "results" / f"{set_code}.{event}" / "deck_synergy_v4" / "draft_inputs.npz"
    if f4.exists():
        z = np.load(f4)
        if len(z["game_logit"]) == len(d.draft_id):
            gl = np.where(z["has_deck"], z["game_logit"], 0.0)
            dense += [gl[pg["gd"]], (~z["has_deck"][pg["gd"]]).astype(float)]
            n_prec = 2
            print("  round 4 game_logit: precision covariate of p_hat only (not in the deck model)", flush=True)
    cats = [(pg["state"], 22), (pg["rank"], N_RANKS), (pg["exp"].argmax(1), pg["exp"].shape[1]), (own_mull, 3)]
    data = Data(names=d.names, is_land=is_land, Xb=Xb.astype(np.int8), gb=gb, gd=pg["gd"], n_drafts=len(d.draft_id),
                H=H, Dr=Dr, y=pg["y"], split=pg["split"].astype(np.int8), cats=cats,
                dense=np.column_stack(dense).astype(np.float32), mull=own_mull, skill=skill)
    data.n_precision = n_prec
    D = len(d.draft_id)
    has = main >= 0
    from eval_common import main_builds
    _, _, main_bix = main_builds(d, g)
    ext = {"pg": pg, "main": main, "has": has, "main_bix": main_bix, "blocks": game_blocks(g, pg, D), "D": D}
    del g
    return data, ext


# ---------------------------------------------------------------------------------------------------- analysis

def lam_edge(lam):
    return "lower edge" if lam == LAMS[0] else "upper edge" if lam == LAMS[-1] else "interior"


def slope_lambda(w, hvv, m, rel, drawn, N, q):
    """Slope of a deck-power vector on HV over cards m, EIV-corrected by HV reliability, and the implied lambda."""
    m = m & np.isfinite(w) & np.isfinite(hvv)
    if m.sum() <= 5:
        return {"cards": int(m.sum())}
    slope = float(np.polyfit(hvv[m], w[m], 1)[0])
    k = slope / rel if rel > 0 else np.nan
    lam = (k * N - 7) / (drawn + k * N * q) if np.isfinite(k) else np.nan
    return {"cards": int(m.sum()), "slope (raw)": slope, "slope EIV-corrected": float(k),
            "implied lambda": float(lam), "pearson r": float(np.corrcoef(hvv[m], w[m])[0, 1])}


def analyse_v2(data: Data, log=print, prior_experiment=True, seed=1, fm=None):
    """hand_value.analyse with the fix (deck model and prior arms without the precision-only columns), the wider
    ridge grid, and the FM comparisons. fm: optional {"w": [C], "marginal": [C]} from T2."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    dhalf = rng.random(data.n_drafts) < 0.5
    hand_n = np.asarray(data.H.sum(1)).ravel()
    hc = data.H.tocoo()
    over = hc.data > data.Xb[data.gb[hc.row], hc.col]
    valid = (hand_n == 7) & (np.bincount(hc.row[over], None, len(data.y)) == 0)
    L = np.asarray(data.H[:, data.is_land].sum(1)).ravel()
    res = {"games": int(len(data.y)), "valid games": int(valid.sum()),
           "dropped (hand != 7 or hand card not in deck)": float(1 - valid.mean()),
           "mulligan share": float((data.mull > 0).mean()),
           "precision-only dense columns (p_hat only)": int(getattr(data, "n_precision", 0)), "lambda grid": list(LAMS)}
    log(f"  {len(data.y)} games, {valid.mean():.4f} valid [{time.time() - t0:.0f}s]")
    tr, va, te = (np.flatnonzero(valid & (data.split == s)) for s in (0, 1, 2))
    spell, land = ~data.is_land, data.is_land.copy()
    dv = deck_view(data)

    def estimate(idx, tag):
        e, pw = outcome_residuals(data, idx, dhalf[data.gd[idx]])        # p_hat keeps the precision columns
        out = {c: hand_values(data, idx, e, pw, cls) for c, cls in (("spell", spell), ("land", land))}
        hv = {k: np.where(spell, out["spell"][k], out["land"][k]) for k in out["spell"]}
        for s, msk in {"A": dhalf[data.gd[idx]], "B": ~dhalf[data.gd[idx]]}.items():
            hv[f"hv_logit_{s}"] = hand_values(data, idx[msk], e[msk], pw[msk], spell)["hv_logit"]
        core = np.isin(L[idx], (3, 4)) & (data.mull[idx] == 0)
        hv["hv_logit_core"] = hand_values(data, idx[core], e[core], pw[core], spell)["hv_logit"]
        for side, val in (("play", 0.5), ("draw", -0.5)):
            msk = data.dense[idx, 0] == val
            hv[f"hv_logit_{side}"] = hand_values(data, idx[msk], e[msk], pw[msk], spell)["hv_logit"]
        log(f"  hand values on {tag} games [{time.time() - t0:.0f}s]")
        return hv

    H_tr, H_te = estimate(tr, "train"), estimate(te, "test")
    wr = win_rates(data, tr)
    fits = []
    for lam in LAMS:                                                     # FIX: deck model without the FM score
        pred, par = fit_logit(dv, tr, dv.cats, lam)
        fits.append((logloss(data.y[va], pred(va)).mean(), lam, pred, par))
    v0, lam0, pred0, par0 = min(fits, key=lambda f: f[0])
    w_deck = par0[2]
    log(f"  deck model (no FM score) lambda {lam0:g} ({lam_edge(lam0)}) val {v0:.5f} [{time.time() - t0:.0f}s]")

    sel = spell & (H_tr["n_games"] >= MIN_TRAIN) & (H_te["n_games"] >= MIN_TEST)
    wt = H_tr["n_games"]
    hv_tr_c, tau2 = center_shrink(H_tr["hv_logit"], H_tr["se_logit"], wt, spell & (H_tr["n_games"] >= 200))
    rA, _, _ = corr(H_tr["hv_logit_A"], H_tr["hv_logit_B"], sel)
    rT, _, _ = corr(H_te["hv_logit_A"], H_te["hv_logit_B"], sel)
    rel_tr, rel_te = 2 * rA / (1 + rA), 2 * rT / (1 + rT)
    res.update({"cards in tables": int(sel.sum()), "lambda deck": lam0, "lambda deck grid position": lam_edge(lam0),
                "deck model val logloss by lambda": {f"{f[1]:g}": float(f[0]) for f in fits},
                "split-half r (train, halves)": rA, "reliability train (Spearman-Brown)": rel_tr,
                "split-half r (test, halves)": rT, "reliability test (Spearman-Brown)": rel_te,
                "hv tau (sd of true logit hand values)": float(np.sqrt(tau2)),
                "median se_logit (train, table cards)": float(np.nanmedian(H_tr["se_logit"][sel])),
                "cluster/HC se ratio (pp, median)": float(np.nanmedian(H_tr["se_pp_cluster"][sel] / H_tr["se_pp"][sel])),
                "balance index sd (train)": float(np.nanstd(H_tr["balance"][sel])),
                "corr balance index vs HV (train)": corr(H_tr["balance"], H_tr["hv_logit"], sel)[0],
                "corr hv full vs core (L in 3-4, no mulligan)": corr(H_tr["hv_logit"], H_tr["hv_logit_core"], sel)[0],
                "corr hv play vs draw": corr(H_tr["hv_logit_play"], H_tr["hv_logit_draw"], sel)[0]})
    target = H_te["hv_logit"]
    ratings = {"GP WR": wr["gp_wr"], "OH WR": wr["oh_wr"], "GD WR": wr["gd_wr"], "GIH WR": wr["gih_wr"],
               "IWD": wr["iwd"], "deck model w (clean)": w_deck, "HV train": H_tr["hv_logit"],
               "HV train shrunk": hv_tr_c}
    if fm is not None:
        ratings["FM w (T2 champion)"] = fm["w"]
        if fm.get("marginal") is not None:
            ratings["FM full marginal (T2 champion)"] = fm["marginal"]
    table = {}
    for k, v in ratings.items():
        p, s, n = corr(v, target, sel)
        m = sel & np.isfinite(v) & np.isfinite(target)
        tilt = np.nan
        if m.sum() > 5:
            beta = np.polyfit(target[m], v[m], 1)
            tilt = corr(v - np.polyval(beta, target), wr["deck_skill"], m)[0]
        table[k] = {"pearson vs test HV": p, "spearman vs test HV": s,
                    "pearson disattenuated (test HV reliability)": p / np.sqrt(rel_te) if rel_te > 0 else np.nan,
                    "skill tilt": tilt, "cards": n}
        log(f"    {k:32s} r {p:+.3f}  rho {s:+.3f}  skill tilt {tilt:+.3f}  (n={n})")
    res["ratings vs test-game hand value (circular target: test HV)"] = table
    drawn = float(np.asarray(data.Dr[tr].sum(1)).mean())
    N = float(np.median(data.Xb.sum(1)))
    q = drawn / (N - 7)
    dvh = {"mean cards drawn after the opener": drawn, "deck size": N, "q": q,
           "kappa if drawn = hand value (lambda 1)": (7 + drawn) / (N * (1 - q)),
           "kappa if only the opener matters (lambda 0)": 7 / N,
           "clean deck model w": slope_lambda(w_deck, H_tr["hv_logit"], sel, rel_tr, drawn, N, q)}
    if fm is not None:
        dvh["FM w (T2 champion)"] = slope_lambda(fm["w"], H_tr["hv_logit"], sel, rel_tr, drawn, N, q)
        if fm.get("marginal") is not None:
            dvh["FM full marginal (T2 champion)"] = slope_lambda(fm["marginal"], H_tr["hv_logit"], sel, rel_tr, drawn,
                                                                 N, q)
    res["deck vs hand"] = dvh
    for k, v in dvh.items():
        if isinstance(v, dict) and "slope (raw)" in v:
            log(f"  deck vs hand [{k}]: slope {v['slope (raw)']:.3f} (EIV {v['slope EIV-corrected']:.3f}), implied "
                f"lambda {v['implied lambda']:.2f}")

    losses = {}
    if prior_experiment:                                                 # FIX: prior arms without the FM score too
        prior = np.where(np.isfinite(hv_tr_c) & spell, hv_tr_c, 0.0)
        extra = data.Xs @ prior
        arms = {"ridge": dict(), "ridge + kappa*HV": dict(extra=extra), "kappa*HV only": dict(extra=extra, use_w=False)}
        rows = {}
        for name, kw in arms.items():
            best = None
            for lam in (LAMS if kw.get("use_w", True) else (0.0,)):
                pred, par = fit_logit(dv, tr, dv.cats, lam, **kw)
                v = logloss(data.y[va], pred(va)).mean()
                if best is None or v < best[0]:
                    best = (v, lam, pred, par)
            v, lam, pred, par = best
            lt = logloss(data.y[te], pred(te))
            losses[name] = np.bincount(data.gd[te], lt, data.n_drafts)
            rows[name] = {"val logloss": float(v), "lambda": lam, "lambda grid position": lam_edge(lam) if lam else "-",
                          "kappa": float(par[3]), "test logloss": float(lt.mean()), "_lt": lt}
        base = rows["ridge"]["_lt"]
        nd = np.bincount(data.gd[te], None, data.n_drafts)
        for name, r_ in rows.items():
            dlt = r_.pop("_lt") - base
            sd = np.bincount(data.gd[te], dlt, data.n_drafts) - nd * dlt.mean()
            r_["vs ridge x1000"] = float(dlt.mean() * 1000)
            r_["vs ridge se x1000 (draft cluster)"] = float(np.sqrt((sd ** 2).sum()) / len(dlt) * 1000)
            log(f"    prior arm {name:18s} lam {r_['lambda']:g} kappa {r_['kappa']:+.3f} test {r_['test logloss']:.5f}"
                f"  vs ridge {r_['vs ridge x1000']:+.3f} ± {1.96 * r_['vs ridge se x1000 (draft cluster)']:.3f}")
        res["prior experiment (test games, deck model without the FM score)"] = rows
        losses["n_games"] = nd
    cards = {"card": np.array(data.names), "is_land": data.is_land,
             **{f"train_{k}": v for k, v in H_tr.items()}, "train_hv_logit_shrunk": hv_tr_c,
             "test_hv_logit": H_te["hv_logit"], "test_se_logit": H_te["se_logit"], "deck_w_clean": w_deck,
             **{k: wr[k] for k in ("gp_wr", "oh_wr", "gd_wr", "gih_wr", "gns_wr", "iwd", "deck_skill", "n_gp")}}
    if fm is not None:
        cards["fm_w"] = fm["w"]
        if fm.get("marginal") is not None:
            cards["fm_marginal"] = fm["marginal"]
    res["seconds"] = time.time() - t0
    return res, {"cards": cards, "losses": losses, "ratings": ratings, "n_gp": wr["n_gp"]}


# ---------------------------------------------------------------------------------------------------- external test

def rating_scores(data: Data, ratings: dict, n_gp, main, spells_only=True):
    """{rating: [D] deck score = sum over the main build of copies x centred rating} (missing -> 0)."""
    cls = ~data.is_land if spells_only else np.ones(len(data.names), bool)
    out = {}
    for k, v in ratings.items():
        v = np.asarray(v, float)
        ok = cls & np.isfinite(v) & (n_gp > 0)
        if ok.sum() < 3:
            continue
        c = np.where(ok, v - np.average(v[ok], weights=n_gp[ok]), 0.0)
        s = data.Xs @ c
        sc = np.zeros(len(main))
        sc[main >= 0] = s[main[main >= 0]]
        out[f"{k} sum"] = sc[None]
    return out


def sum_rating_test(data: Data, ext: dict, ratings: dict, n_gp, n_boot=N_BOOT, name="", log=print):
    from sc_gll import subsets
    from t_common import evaluate_ctx, fit_contexts
    pg, D = ext["pg"], ext["D"]
    scores = rating_scores(data, ratings, n_gp, ext["main"])
    ctxs, cinfo, _ = fit_contexts(pg, D, ("dl", "raw"))
    subs = subsets(pg, ext["has"], ext["main_bix"])
    ref = "HV train shrunk sum"
    pairs = [(m, ref) for m in scores if m != ref] + [("GIH WR sum", m) for m in scores
                                                       if m not in ("GIH WR sum", ref)]
    res, losses, ng = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, None, n_boot, ext["blocks"], name, log)
    res.update({"contexts": cinfo, "reference": ref, "primary key": "experienced dl",
                "decision": "HV sum - GIH WR sum >= +0.02 (decide) -> HV best-rating claim external"})
    return res, losses, ng


# ---------------------------------------------------------------------------------------------------- run

def load_fm(set_code, names):
    f = REPO / "results" / f"{set_code}.PremierDraft" / "t2_mull_split" / "card_power.npz"
    if not f.exists():
        return None
    z = np.load(f)
    if list(z["names"].astype(str)) != list(names):
        print("  T2 card_power.npz names differ; FM comparison skipped", flush=True)
        return None
    return {"w": z["champion|w"].astype(float),
            "marginal": z["champion|marginal_total"].astype(float) if "champion|marginal_total" in z.files else None}


def main(*sets):
    import polars as pl
    from t_common import peak_rss_gb, save_losses
    for set_code in sets:
        out = REPO / "results" / f"{set_code}.PremierDraft" / "hand_value_v2"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        data, ext = load_set_v2(set_code)
        fm = load_fm(set_code, data.names)
        print(f"{set_code}: loaded [{time.time() - t0:.0f}s]; T2 FM card power {'found' if fm else 'absent'}", flush=True)
        res, art = analyse_v2(data, log=lambda s: print(s, flush=True), fm=fm)
        sres, slosses, sng = sum_rating_test(data, ext, art["ratings"], art["n_gp"], name=set_code,
                                             log=lambda s: print(s, flush=True))
        res = {"set": set_code, **res, "seconds total": time.time() - t0, "max rss GB": peak_rss_gb()}
        pl.DataFrame(art["cards"]).sort("train_hv_logit_shrunk", descending=True, nulls_last=True) \
            .write_csv(out / "card_values.csv")
        np.savez(out / "test_losses.npz", **{k.replace(" ", "_").replace("*", ""): v for k, v in art["losses"].items()})
        (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
        (out / "sum_ratings.json").write_text(json.dumps({"set": set_code, **sres}, indent=1, default=float))
        save_losses(out / "sum_losses.npz", slosses, sng)
        assert_no_torch()
        print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


def meta(*sets):
    from eval_protocol import DecisionRule, SetResult, decide, meta_analysis
    from t_common import meta_rows
    files = {s: REPO / "results" / f"{s}.PremierDraft" / "hand_value_v2" / "sum_ratings.json" for s in sets}
    files = {s: f for s, f in files.items() if f.exists()}
    out = {"sets": list(files), "sum-of-ratings": {}, "prior arms": {}}
    pairs = sorted({p for f in files.values() for p in json.loads(f.read_text())["pairs"]})
    for p in pairs:
        for key in SUM_KEYS:
            rs = meta_rows(files, p, key)
            if len(rs) < 2:
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out["sum-of-ratings"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci),
                                                           "p": m.re_p_two_sided, "decide": ok}
            print(f"{p:45s} {key:15s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] "
                  f"{m.n_positive}/{len(rs)} adopt={ok}")
    rs = []
    for s in files:
        r = json.loads((files[s].parent / "results.json").read_text())
        arm = r["prior experiment (test games, deck model without the FM score)"]["ridge + kappa*HV"]
        se = arm["vs ridge se x1000 (draft cluster)"]
        rs.append(SetResult(s, -arm["vs ridge x1000"], se, se, 0.0, 0, 1))   # higher = better (loss reduction)
    if len(rs) >= 2:
        m = meta_analysis(rs)
        out["prior arms"]["ridge + kappa*HV - ridge (x1000, loss reduction)"] = {
            "re mean": m.re_mean, "ci": list(m.re_ci), "test as FM prior next round": bool(m.re_mean >= 0.02)}
        print(f"\nHV prior arm (loss reduction vs ridge, x1000):\n{m.summary()}\n  >= +0.02 -> test as an FM prior: "
              f"{m.re_mean >= 0.02}")
    dst = REPO / "results" / "meta" / "hand_value_v2.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------------------------------------------- self-test

def selftest():
    t0 = time.time()
    data, truth, trap = hv1.synthetic(n_drafts=4000)
    # add the "FM deck score" as a precision covariate: the true deck strength of the build (what made v1 wrong)
    sp = ~data.is_land
    true_w = np.where(sp, np.nan_to_num(truth), 0.0)
    deck_true = (data.Xb.astype(float) @ true_w)[data.gb]
    data.dense = np.column_stack([data.dense, deck_true]).astype(np.float32)
    data.n_precision = 1
    res, art = analyse_v2(data, log=lambda s: None, prior_experiment=True)
    s2 = res["deck vs hand"]["clean deck model w"]["slope (raw)"]
    # v1 behaviour: the same deck model WITH the precision column (the bug)
    pred, par = fit_logit(data, np.flatnonzero(data.split == 0), data.cats, res["lambda deck"])
    hvv = art["cards"]["train_hv_logit"]
    m = sp & np.isfinite(hvv)
    s1 = float(np.polyfit(hvv[m], par[2][m], 1)[0])
    print(f"[1] slope of deck-model w on HV: with the FM score in the deck model (v1 bug) {s1:+.3f}; fixed {s2:+.3f}; "
          f"lambda {res['lambda deck']:g} ({res['lambda deck grid position']})")
    assert s2 > 0.05 and s2 > 3 * abs(s1), (s1, s2)
    # 2. external sum-of-ratings test on SC-GLL (synthetic pg: one build per draft)
    G = len(data.y)
    D = data.n_drafts
    pg = {"gd": data.gd, "y": data.y, "split": data.split, "match": np.ones(G, int), "state": np.zeros(G, int),
          "on_play": data.dense[:, 0].astype(float), "mull": data.mull.astype(float), "own_mull": data.mull.astype(float),
          "opp_mull": np.zeros(G), "exp": np.eye(8)[np.full(G, 5)], "rank": np.full(G, 3),
          "n_bucket": np.full(G, 100.0), "skill": np.clip(data.skill, 0, 1), "build_index": np.full(G, "0", dtype=object)}
    ext = {"pg": pg, "main": np.arange(D), "has": np.ones(D, bool), "main_bix": np.full(D, "0", dtype=object),
           "blocks": np.array([f"d{i * 20 // D}" for i in range(D)]), "D": D}
    sres, _, _ = sum_rating_test(data, ext, art["ratings"], art["n_gp"], n_boot=40, name="SYN", log=lambda s: None)
    x = sres["pairs"]["HV train shrunk sum - GIH WR sum"]["experienced dl"]
    print(f"[2] SC-GLL-dl: HV-sum - GIH-WR-sum {x['delta']:+.3f} ± {x['se']:.3f} (x1000); models "
          + "  ".join(f"{k} {v['experienced dl']['sc_gll']:.2f}" for k, v in sres["models"].items()))
    assert np.isfinite(x["delta"]) and x["se"] > 0
    json.dumps(res, default=float)
    json.dumps(sres, default=float)
    assert_no_torch()
    print(f"hand_value_v2 self-test OK [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
