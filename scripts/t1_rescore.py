"""T1 (meta_review_2): every saved deck score under the SC-GLL context matrix, plus the SC-GLL-tuned blend*.

    python scripts/t1_rescore.py SET [SET ...] [--contexts raw,dl,pre] [--boot 500]
    python scripts/t1_rescore.py meta SET ...        (12-set RE meta / decide per pair x key, and the T1 rules)
    python scripts/t1_rescore.py --selftest

Numpy / sklearn only: no training, no torch (torch-free decks.deckstrength shim), no lightgbm. Per set:
  games     the same narrow game-file scan as sc_gll (eval_common.load_games / per_game / main_builds) plus own and
            opponent mulligans; draft order = sc_gll/scores.npz (load_drafts order); alignment with
            sc_gll/context.npz (gd, y) is asserted
  contexts  t_common.CONTEXTS fit once on train games: raw (sc_gll's), dl (de-leaked bucket), pre (dl bucket, no
            own-mulligan term, opponent mulligans kept); --contexts adds pre-raw
  scores    sc_gll/scores.npz: game FM (round 4 cached), linear game model, LightGBM (platinum, 3 seeds), round 4
            blend; game_fm_v2/scores.npz: every arm (refit scored with the champion's val (a, b), seed-paired, as in
            game_fm_v2); structural_a/scores*.npz; game_transformer/scores_<tag>.npz (transformer, its FM path and
            its own champion, suffixed [tag]); t2_mull_split/scores.npz when present (T2 arms, suffixed [T2])
  blend*    (1 - alpha) z(LightGBM) + alpha z(FM), z standardised on train drafts; alpha in 0, 0.05, ..., 1 chosen per
            LightGBM seed on VAL SC-GLL-dl (experienced, dl context), two-half cross-fit: (a, b) fit on one half of
            the val drafts, log loss on the other, summed over both directions. alpha = 1 is the FM. Two versions:
            with the cached round 4 FM and with the game_fm_v2 champion (FM seed = LightGBM seed mod FM seeds)
  subsets   experienced (primary), all, match 1, main build -- each under every context: keys "<subset> <context>"
Pairs (B - A, x1000 nats/game, paired day-block bootstrap, seeds as a variance component): see PAIRS below.
Writes results/{SET}.PremierDraft/t1_rescore/{results.json, losses.npz, blend.npz}. The json is in sc_gll's meta
format: python scripts/sc_gll.py meta --file t1_rescore/results.json --subset "experienced pre" SET ...
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import (CONTEXTS, DEFAULT_CONTEXTS, REPO, assert_no_torch, evaluate_ctx, fit_contexts, game_blocks,  # noqa: E402
                      npz_key, save_losses, split_mull, torchfree)

torchfree()
import numpy as np  # noqa: E402
from types import SimpleNamespace  # noqa: E402

N_BOOT = 500
ALPHAS = np.round(np.linspace(0, 1, 21), 2)
FM_R4, CH, LG = "game FM", "champion", "LightGBM"
BL_R4, BL_CH = "blend* (LightGBM + game FM)", "blend* (LightGBM + champion)"
RULE_KEYS = ("experienced raw", "experienced dl", "experienced pre", "match 1 dl", "match 1 pre")


# ---------------------------------------------------------------- saved scores

def load_scores(res_dir: Path, draft_id: np.ndarray):
    """{model: [S, D]} of every saved deck score aligned to draft_id, {model: twin}, provenance."""
    scores, twin, src = {}, {}, {}

    def take(f, mapping, suffix=""):
        z = np.load(f)
        assert (z["draft_id"].astype(str) == draft_id).all(), f"{f}: draft order differs"
        for k in z.files:
            if k in ("draft_id", "split", "has"):
                continue
            name = mapping.get(k, k.replace("_", " ")) if mapping is not None else k.replace("_", " ")
            if name is None:
                continue
            scores[name + suffix] = np.atleast_2d(z[k]).astype(np.float64)
            src[name + suffix] = str(f.relative_to(REPO))

    take(res_dir / "sc_gll" / "scores.npz", {"game_FM": FM_R4, "linear_game_model": "linear game model",
                                              "LightGBM": LG, "round_4_blend": "round 4 blend",
                                              "lgbm_own_rank": None, "blend_own_rank": None})
    f = res_dir / "game_fm_v2" / "scores.npz"
    if f.exists():
        take(f, {"champion": CH, "-_mull": "- mull", "rated-dl": "rated-dl", "-_skill": "- skill", "refit": "refit",
                 "recency": "recency", "refit_own_val_fit": "refit (own val fit)", "cached_game_FM": None})
        if "refit" in scores:
            twin["refit"] = CH
    for f in sorted((res_dir / "structural_a").glob("scores*.npz")):
        z = np.load(f)
        for k in z.files:
            if k.startswith("A_") and k.replace("_", " ") not in scores:
                assert (z["draft_id"].astype(str) == draft_id).all()
                scores[k.replace("_", " ")] = np.atleast_2d(z[k]).astype(np.float64)
                src[k.replace("_", " ")] = str(f.relative_to(REPO))
    for f in sorted((res_dir / "game_transformer").glob("scores_*.npz")):
        tag = f.stem[len("scores_"):]
        take(f, {"champion": "champion", "transformer": "transformer", "transformer_FM_path": "transformer FM path",
                 "cached_game_FM": None}, f" [{tag}]")
    f = res_dir / "t2_mull_split" / "scores.npz"
    if f.exists():
        take(f, None, " [T2]")
        if "refit [T2]" in scores:
            twin["refit [T2]"] = "champion [T2]"
    return scores, twin, src


def pairs_for(scores):
    P = [(FM_R4, LG), (FM_R4, "round 4 blend"), (FM_R4, BL_R4), ("linear game model", FM_R4)]
    if CH in scores:
        P += [(CH, LG), (CH, BL_CH), (CH, "round 4 blend"), (FM_R4, CH), ("linear game model", CH)]
        P += [(CH, v) for v in ("- mull", "rated-dl", "- skill", "refit", "recency", "refit (own val fit)")]
    P += [(FM_R4, k) for k in scores if k.startswith("A ")]
    for k in scores:
        if k.startswith("transformer [") or k.startswith("transformer FM path ["):
            tag = k[k.index("["):]
            P += [(f"champion {tag}", k)] + ([(CH, k)] if CH in scores else [])
        if k.endswith(" [T2]") and k != "champion [T2]":
            P += [("champion [T2]", k)]
    return [(a, b) for a, b in P if a in scores and b in scores]


# ---------------------------------------------------------------- blend*

def zstd(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-12)


def blend_star(L, Fm, pg, ctx, fit_val, D, has, split, seed=7, log=print):
    """L [S, D] LightGBM logits, Fm [SF, D] FM scores. Per LightGBM seed s: alpha* on val SC-GLL (the given context
    and val subset `fit_val`), two-half cross-fit over val drafts. Returns ([S, D] blend scores, info)."""
    from sc_gll import sc_losses
    ref = has & (split == 0)
    vd = np.flatnonzero(has & (split == 1))
    half = np.zeros(D, bool)
    half[vd[np.random.default_rng(seed).permutation(len(vd)) < len(vd) // 2]] = True
    hg = half[pg["gd"]]
    out, info = [], {"alphas": [], "alpha per half": [], "val cross-fit loss x1000 by alpha": []}
    for s in range(len(L)):
        zl, zf = zstd(L[s], ref), zstd(Fm[s % len(Fm)], ref)
        half_loss, n = np.zeros((len(ALPHAS), 2)), np.zeros(2)
        for i, a in enumerate(ALPHAS):
            sc = (1 - a) * zl + a * zf
            for j, side in enumerate((True, False)):   # (a, b) fit on the other half, log loss on this half
                ls, ng, _ = sc_losses(sc, pg, ctx, fit_val & (hg != side), fit_val & (hg == side), D)
                half_loss[i, j], n[j] = ls.sum(), ng.sum()
        curve = 1000 * half_loss.sum(1) / n.sum()
        k = int(np.argmin(curve))
        a = float(ALPHAS[k])
        out.append((1 - a) * zl + a * zf)
        info["alphas"].append(a)
        info["alpha per half"].append([float(ALPHAS[np.argmin(half_loss[:, j])]) for j in range(2)])
        info["val cross-fit loss x1000 by alpha"].append([float(c) for c in curve])
        log(f"    blend* seed {s}: alpha {a:.2f} (per half {info['alpha per half'][-1]}; val cross-fit loss "
            f"{curve[k]:.3f}, alpha 0 {curve[0]:.3f}, alpha 1 {curve[-1]:.3f})")
    return np.stack(out), info


# ---------------------------------------------------------------- the run

def rescore(pg, D, has, main_bix, blocks, scores, twin, contexts=DEFAULT_CONTEXTS, n_boot=N_BOOT, name="", log=print):
    from sc_gll import subsets
    t0 = time.time()
    split = np.full(D, -1, np.int8)
    split[pg["gd"]] = pg["split"]
    ctxs, cinfo, _ = fit_contexts(pg, D, tuple(dict.fromkeys(("dl",) + tuple(contexts))))
    subs = subsets(pg, has, main_bix)
    log("  contexts: " + "  ".join(f"{k} test {v['test logloss (context only)']:.5f}" for k, v in cinfo.items())
        + f" [{time.time() - t0:.0f}s]")
    fit_val = subs["experienced"][0]
    binfo = {}
    for nm, fm in ((BL_R4, FM_R4), (BL_CH, CH)):
        if LG in scores and fm in scores:
            scores[nm], binfo[nm] = blend_star(scores[LG], scores[fm], pg, ctxs["dl"], fit_val, D, has, split, log=log)
            binfo[nm]["FM member"] = fm
    pairs = pairs_for(scores)
    use = {k: ctxs[k] for k in contexts}
    res, losses, ngames = evaluate_ctx(scores, pg, use, subs, D, pairs, twin, n_boot, blocks, name, log)
    res.update({"contexts": cinfo, "blend*": binfo, "primary keys": list(RULE_KEYS), "seconds": time.time() - t0})
    return res, losses, ngames


def load_set(set_code):
    from eval_common import NARROW, load_games, main_builds, per_game
    from mtgdata.games import game_path
    res_dir = REPO / "results" / f"{set_code}.PremierDraft"
    ref = np.load(res_dir / "sc_gll" / "scores.npz")
    draft_id, split, has = ref["draft_id"].astype(str), ref["split"].astype(np.int8), ref["has"]
    d = SimpleNamespace(draft_id=draft_id, split=split)
    g = load_games(game_path(set_code, "PremierDraft"), draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    assert np.allclose(pg["own_mull"] - pg["opp_mull"], pg["mull"])
    has2, _, main_bix = main_builds(d, g)
    assert (has2 == has).all(), "has-deck mask differs from sc_gll/scores.npz"
    c = np.load(res_dir / "sc_gll" / "context.npz")
    assert (c["gd"] == pg["gd"]).all() and (c["y"] == pg["y"]).all(), "games differ from sc_gll/context.npz"
    D = len(draft_id)
    blocks = game_blocks(g, pg, D)
    del g
    return res_dir, draft_id, pg, D, has, main_bix, blocks


def main(argv):
    contexts, n_boot, sets = list(DEFAULT_CONTEXTS), N_BOOT, []
    it = iter(argv)
    for a in it:
        if a == "--contexts":
            contexts = next(it).split(",")
        elif a == "--boot":
            n_boot = int(next(it))
        else:
            sets.append(a)
    assert all(c in CONTEXTS for c in contexts), contexts
    for s in sets:
        t0 = time.time()
        res_dir, draft_id, pg, D, has, main_bix, blocks = load_set(s)
        scores, twin, src = load_scores(res_dir, draft_id)
        print(f"=== {s}: {D} drafts, {len(pg['y'])} games, {len(scores)} saved scores [{time.time() - t0:.0f}s]",
              flush=True)
        res, losses, ngames = rescore(pg, D, has, main_bix, blocks, scores, twin, contexts, n_boot, s,
                                      log=lambda m: print(m, flush=True))
        out = res_dir / "t1_rescore"
        out.mkdir(parents=True, exist_ok=True)
        res = {"set": s, "sources": src, "twins": twin, **res, "total seconds": time.time() - t0}
        (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
        save_losses(out / "losses.npz", losses, ngames)
        np.savez_compressed(out / "blend.npz", draft_id=draft_id,
                            **{npz_key(k): scores[k].astype(np.float32) for k in (BL_R4, BL_CH) if k in scores})
        assert_no_torch()
        print(f"{s} done [{time.time() - t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- meta + the T1 decision rules

def meta(*sets):
    from eval_protocol import DecisionRule, decide, meta_analysis
    from t_common import meta_rows
    files = {s: REPO / "results" / f"{s}.PremierDraft" / "t1_rescore" / "results.json" for s in sets}
    have = {s: f for s, f in files.items() if f.exists()}
    pairs = sorted({p for f in have.values() for p in json.loads(f.read_text())["pairs"]})
    out = {"sets": list(have), "pairs": {}}
    for p in pairs:
        for key in RULE_KEYS:
            rs = meta_rows(have, p, key)
            if len(rs) < 2:
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out["pairs"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci), "p": m.re_p_two_sided,
                                                  "sets better": f"{m.n_positive}/{len(rs)}",
                                                  "sig worse": m.n_sig_worse, "adopt": ok, "why": why}
            print(f"{p:55s} {key:16s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] p={m.re_p_two_sided:.4f} "
                  f"{m.n_positive}/{len(rs)}  adopt={ok}")
    # rules (meta_review_2 (a))
    rules = {}
    pr = out["pairs"]
    for fm in (FM_R4, CH):
        lg = pr.get(f"{LG} - {fm}", {})
        rules[f"FM over LightGBM confirmed ({fm})"] = all(
            k in lg and lg[k]["ci"][1] < 0 for k in ("experienced dl", "experienced pre"))
    for bl, fm in ((BL_R4, FM_R4), (BL_CH, CH)):
        r = pr.get(f"{bl} - {fm}", {}).get("experienced dl")
        if r:
            alphas = [a for f in have.values() for a in json.loads(f.read_text())["blend*"].get(bl, {}).get("alphas", [])[:1]]
            rules[f"{bl}: FM alone stays (blend* - FM < +0.02, dl)"] = r["re mean"] < 0.02
            rules[f"{bl}: blend* becomes the deck rater (>= +0.02 and decide passes)"] = r["re mean"] >= 0.02 and r["adopt"]
            rules[f"{bl}: alpha* >= 0.8 in sets (seed 0)"] = f"{sum(a >= 0.8 for a in alphas)}/{len(alphas)}"
    out["rules"] = rules
    print("\nT1 rules:\n  " + "\n  ".join(f"{k}: {v}" for k, v in rules.items()))
    dst = REPO / "results" / "meta" / "t1_rescore.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------- synthetic self-test

def selftest():
    import tempfile
    import sc_gll
    t0 = time.time()
    rng = np.random.default_rng(0)
    per_set = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, s in enumerate(("S1", "S2", "S3")):
            d, pg, deck, skill = sc_gll.synthetic(D=6000, seed=k)
            D = len(d.wins)
            G = len(pg["y"])
            # own mulligans caused by the deck (a mediator), opponent mulligans exogenous; outcome re-drawn with both
            own = (rng.random(G) < 0.12 + 0.25 * (deck[pg["gd"]] < -0.3)).astype(float)
            opp = (rng.random(G) < 0.12).astype(float)
            pg["own_mull"], pg["opp_mull"], pg["mull"] = own, opp, own - opp
            eta = 0.15 + deck[pg["gd"]] + skill[pg["gd"]] + 0.1 * pg["on_play"] - 0.12 * (pg["state"] // 3 - pg["state"] % 3)
            pg["y"] = (rng.random(G) < 1 / (1 + np.exp(-(eta - 0.35 * (own - opp))))).astype(float)
            has = np.ones(D, bool)
            good = deck + rng.normal(0, 0.15, D)
            lg = np.stack([deck + rng.normal(0, 0.5, D) + 0.6 * skill for _ in range(3)])
            scores = {FM_R4: good[None], CH: np.stack([good, good + rng.normal(0, 0.02, D)]), LG: lg,
                      "linear game model": (deck + rng.normal(0, 0.3, D))[None],
                      "refit": np.stack([good, good + rng.normal(0, 0.02, D)]) * 1.1,
                      "round 4 blend": 0.5 * lg + 0.5 * good}
            res, losses, ng = rescore(pg, D, has, np.full(D, "0", dtype=object), np.asarray(d.draft_time).astype(str),
                                      scores, {"refit": CH}, ("raw", "dl", "pre"), n_boot=40, name=s,
                                      log=(print if k == 0 else (lambda m: None)))
            json.dumps(res, default=float)
            assert "experienced pre" in res["pairs"][f"{LG} - {CH}"]
            assert res["models"]["refit"]["experienced dl"]["calibration"] == "twin: champion"
            al = res["blend*"][BL_CH]["alphas"]
            assert all(0 <= a <= 1 for a in al)
            per_set[s] = Path(tmp) / s / "t1_rescore" / "results.json"
            per_set[s].parent.mkdir(parents=True)
            per_set[s].write_text(json.dumps(res, default=float))
            print(f"[{s}] blend* alphas {al}; LightGBM - champion (exp dl) "
                  f"{res['pairs'][f'{LG} - {CH}']['experienced dl']['delta']:+.3f}; blend* - champion "
                  f"{res['pairs'][f'{BL_CH} - {CH}']['experienced dl']['delta']:+.3f}")
        from eval_protocol import meta_analysis
        from t_common import meta_rows
        m = meta_analysis(meta_rows(per_set, f"{LG} - {CH}", "experienced dl"))
        assert m.re_mean < 0, m.re_mean   # the leaky, noisier LightGBM loses to the FM once skill is in the context
        mb = meta_analysis(meta_rows(per_set, f"{BL_CH} - {CH}", "experienced dl"))
        print(f"[meta] LightGBM - champion {m.re_mean:+.3f} [{m.re_ci[0]:+.3f}, {m.re_ci[1]:+.3f}]; blend* - champion "
              f"{mb.re_mean:+.3f} [{mb.re_ci[0]:+.3f}, {mb.re_ci[1]:+.3f}]")
        assert mb.re_mean > m.re_mean
    assert_no_torch()
    print(f"t1 self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        main(sys.argv[1:])
