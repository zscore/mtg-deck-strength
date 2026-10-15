"""Joint draft + sealed models on Sealed games:  python scripts/site_joint.py SET [SET ...]  |  meta

Same events, split and judge as site_blend_sealed (Sealed events from the game file, 70 / 15 / 15 by first
draft_time, the site's own split; t_common contexts, sc_gll subsets). Arms (deck-only scores of each event's
most-played build):
  site FM sealed       deck.js 'sealed' (shipped)                   site draft FM   deck.js 'draft' (shipped)
  FM sealed refit      site_joint_fm's rerun of the site's Sealed fit (control: should match site FM sealed)
  joint FM             site_joint_fm's joint model, sealed head     joint FM draft head   its draft head
  LGBM counts draft    site_blend's Premier Draft counts LightGBM (seed 0)
  LGBM joint           counts LightGBM on Premier Draft + Sealed events with an is-sealed column, scored as sealed
                       (draft events from the game file with the same time split; early stop on Sealed val), 3 seeds
  blend* X + Y         t1_rescore.blend_star
All FM arms are scored by deck.js (clamped). Needs site_joint_fm.py's deck.json and site_blend's lgbm_counts_seed0.
Writes results/{SET}.PremierDraft/site_joint/sealed.json; meta writes results/meta/site_joint.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_blend import DEV, LGB_SEEDS, N_BOOT, VAULT, REPO, site_scores  # noqa: E402  (site_blend: lightgbm first)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

SFM, SDF, SR = "site FM sealed", "site draft FM", "FM sealed refit"
JF, JD = "joint FM", "joint FM draft head"
LCD, LJ = "LGBM counts draft", "LGBM joint"
BDD, BJJ = "blend* counts draft + site draft FM", "blend* LGBM joint + joint FM"
BJD, BDJ = "blend* LGBM joint + site draft FM", "blend* counts draft + joint FM"
PAIRS = [(SFM, SR), (SFM, SDF), (SFM, JF), (SDF, JF), (SFM, JD), (JD, JF),
         (SFM, LJ), (LCD, LJ), (SFM, BDD), (SFM, BJJ), (BDD, BJJ), (JF, BJJ), (BDD, BJD), (BDD, BDJ)]
KEYS = ("experienced dl", "all dl", "experienced pre", "match 1 pre")


def joint_lgbm(set_code, names, deck_s, d_s, has_s, seeds=LGB_SEEDS, log=print):
    """[S, D_s] LightGBM logits of the Sealed events, from a model fit on Premier Draft + Sealed events."""
    from eval_common import NARROW, deck_rows, lgbm_model, load_games, main_builds
    from mtgdata.games import game_path
    from site_blend_sealed import sealed_events
    from types import SimpleNamespace
    path = game_path(set_code, "PremierDraft")
    d_d = sealed_events(path)  # event-level W / L and the site's time split; not a sealed-only helper
    g = load_games(path, d_d.draft_id, cols=NARROW)
    has_d, rows, _ = main_builds(d_d, g)
    del g
    deck_d = np.zeros((len(d_d.draft_id), len(names)), np.int8)
    deck_d[has_d] = deck_rows(path, rows[has_d], names)
    nd = len(d_d.draft_id)
    X = np.vstack([np.hstack([deck_d, np.zeros((nd, 1), np.int8)]),
                   np.hstack([deck_s, np.ones((len(deck_s), 1), np.int8)])]).astype(np.float32)
    del deck_d
    d = SimpleNamespace(wins=np.r_[d_d.wins, d_s.wins], losses=np.r_[d_d.losses, d_s.losses])
    has = np.r_[has_d, has_s]
    sealed = np.r_[np.zeros(nd, bool), np.ones(len(deck_s), bool)]
    split = np.r_[d_d.split, d_s.split]
    tr, va = has & (split == 0), has & (split == 1) & sealed
    log(f"  joint LGBM: {tr.sum()} train events ({(tr & sealed).sum()} sealed), {va.sum()} sealed val events")
    out, trees = [], []
    for s in range(seeds):
        t0 = time.time()
        m, n = lgbm_model(X, d, tr, va, params={"num_threads": 4}, seed=s)
        out.append(m.predict(X[sealed], num_iteration=n, raw_score=True))
        trees.append(int(n))
        log(f"  joint LGBM seed {s}: {n} trees [{time.time() - t0:.0f}s]")
    return np.stack(out), trees


def run_set(set_code, n_boot=N_BOOT, log=print):
    from eval_common import NARROW, deck_rows, load_games, main_builds, per_game
    from mtgdata.games import game_path
    from site_blend_sealed import sealed_events
    from t_common import evaluate_ctx, fit_contexts, game_blocks, split_mull
    import sc_gll
    import t1_rescore as t1
    t0 = time.time()
    sb = REPO / "results" / f"{set_code}.PremierDraft" / "site_blend"
    out = REPO / "results" / f"{set_code}.PremierDraft" / "site_joint"
    model = out / "deck.json"
    path = game_path(set_code, "Sealed")
    d = sealed_events(path)
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    has, rows, main_bix = main_builds(d, g)
    D = len(d.draft_id)
    blocks = game_blocks(g, pg, D)
    del g
    deck = np.zeros((D, len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    log(f"=== {set_code} Sealed: {D} events, {len(pg['y'])} games, test events {int((has & (d.split == 2)).sum())} "
        f"[{time.time() - t0:.0f}s]")

    scores, raw = {}, {}
    for nm, ev, mdl in ((SFM, "sealed", None), (SDF, "draft", None), (SR, "sealed_refit", model),
                        (JF, "sealed", model), (JD, "draft", model)):
        c, r = site_scores(set_code, ev, d.names, deck[has], model=mdl)
        scores[nm] = np.full((1, D), np.nan)
        scores[nm][0, has] = c
        raw[nm] = r
    # deck.js vs python on the fixture decks (training decks: unclamped deck logits must agree)
    fm = json.loads((out / "fm.json").read_text())
    names_m = json.loads(model.read_text())["names"]
    check = {}
    for nm, ev in (("sealed", "sealed"), ("draft", "draft"), ("sealed_refit", "sealed_refit")):
        fx = fm["fixtures"][nm]
        Xf = np.zeros((len(fx), len(names_m)), np.int8)
        for i, f in enumerate(fx):
            for j, n in f["deck"].items():
                Xf[i, int(j)] = n
        _, r = site_scores(set_code, ev, names_m, Xf, model=model)
        check[nm] = float(np.max(np.abs(r - np.array([f["logit"] for f in fx]))))
    log(f"  deck.js vs python, max |logit diff| on fixtures: {check}")

    # LightGBMs
    dn = json.loads((sb / "builder.json").read_text())["names"]
    pos = {n: i for i, n in enumerate(d.names)}
    Xd = np.zeros((D, len(dn)), np.float32)
    cols = [(i, pos[n]) for i, n in enumerate(dn) if n in pos]
    Xd[:, [a for a, _ in cols]] = deck[:, [b for _, b in cols]]
    scores[LCD] = lgb.Booster(model_file=str(sb / "lgbm_counts_seed0.txt")).predict(Xd, raw_score=True)[None]
    del Xd
    union = sorted(set(d.names) | set(dn))
    pu = {n: i for i, n in enumerate(union)}
    deck_u = np.zeros((D, len(union)), np.int8)
    deck_u[:, [pu[n] for n in d.names]] = deck
    scores[LJ], trees = joint_lgbm(set_code, union, deck_u, d, has, log=log)
    del deck_u

    ctxs, cinfo, _ = fit_contexts(pg, D, ("raw", "dl", "pre"))
    subs = sc_gll.subsets(pg, has, main_bix)
    fit_val = subs["experienced"][0]
    binfo = {}
    for nm, L, F in ((BDD, LCD, SDF), (BJJ, LJ, JF), (BJD, LJ, SDF), (BDJ, LCD, JF)):
        scores[nm], binfo[nm] = t1.blend_star(scores[L], scores[F], pg, ctxs["dl"], fit_val, D, has, d.split,
                                              log=lambda m: None)
        log(f"  {nm}: alphas {binfo[nm]['alphas']}")
    res, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, PAIRS, None, n_boot, blocks, set_code, log=lambda m: None)
    for key in ("experienced dl", "all dl"):
        log(f"  {key}: " + "  ".join(f"{m} {res['models'][m][key]['sc_gll']:.3f}" for m in scores))
        for a, b in PAIRS:
            r = res["pairs"][f"{b} - {a}"][key]
            log(f"      {b} - {a}: {r['delta']:+.3f} ± {r['se']:.3f}")
    te = has & (d.split == 2)
    first = {m: np.atleast_2d(s)[0] for m, s in scores.items()}
    corr = {f"{a} ~ {b}": float(np.corrcoef(first[a][te], first[b][te])[0, 1])
            for a, b in ((SFM, SR), (SFM, JF), (SDF, JF), (JF, JD), (LCD, LJ))}
    res.update({"set": set_code, "events": D, "trees": trees, "blend*": binfo, "contexts": cinfo,
                "deck.js check": check, "test score correlations": corr, "fm": {k: fm[k] for k in ("grid", "picked")},
                "seconds": time.time() - t0})
    (out / "sealed.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"{set_code} joint sealed done [{time.time() - t0:.0f}s]")


def meta():
    from eval_protocol import DecisionRule, decide, meta_analysis
    from t_common import meta_rows, one_sided_p
    files = {s: REPO / "results" / f"{s}.PremierDraft" / "site_joint" / "sealed.json" for s in DEV + VAULT}
    files = {s: f for s, f in files.items() if f.exists()}
    print(f"##### sealed: {len(files)} sets")
    out = {}
    for a, b in PAIRS:
        p = f"{b} - {a}"
        for key in KEYS:
            rs = meta_rows(files, p, key)
            if len(rs) < 2:
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out.setdefault(p, {})[key] = {
                "re mean": m.re_mean, "ci": list(m.re_ci), "p one-sided": one_sided_p(m),
                "sets better": f"{m.n_positive}/{len(rs)}", "sig worse": m.n_sig_worse,
                "per set": {r.name: r.delta for r in rs}, "adopt": ok, "why": why}
            if key == "experienced dl":
                print(f"{p:55s} {m.re_mean:+.3f} [{m.re_ci[0]:+.3f}, {m.re_ci[1]:+.3f}] p1={one_sided_p(m):.4f} "
                      f"{m.n_positive}/{len(rs)} sigworse={m.n_sig_worse}")
    out["fm picked"] = {s: json.loads(f.read_text())["fm"]["picked"] for s, f in files.items()}
    out["deck.js check"] = {s: json.loads(f.read_text())["deck.js check"] for s, f in files.items()}
    dst = REPO / "results" / "meta" / "site_joint.json"
    dst.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {dst}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["meta"]:
        meta()
    else:
        for s in args:
            run_set(s)
