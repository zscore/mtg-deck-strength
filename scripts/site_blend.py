"""Should the draft sim site (../mtg-draft-sim-site) move from its game FM to blend*?

    python scripts/site_blend.py SET [SET ...] [--boot 500]     per set: scores + SC-GLL (results/{SET}.PremierDraft/site_blend)
    python scripts/site_blend.py meta                            12 dev sets and 6 vault sets, pooled separately
    python scripts/site_blend.py sealed SET [SET ...]            the same question on Sealed games (simplified judge)

The site's deck model is round 4's game FM, retrained by the site's tools/export_deck_model.py with its own changes
(l2 grid with l2_w up to 1e-4, power shrunk by n / (n + 2000), one shared basic weight, clamps to the training
range). Here it is scored by the site's own JavaScript (deck.js DeckModel via node), so the number is what ships.

Arms (deck-only scores of each draft's most-played build, [S, D]):
  site FM            deck.js logit, clamped (what the builder maximises)
  site FM raw        deck.js logit, unclamped
  champion           this repo's game FM (game_fm_v2 champion; vault: fm_scores.npz), 2 seeds
  LGBM r4            round 4 LightGBM (counts + structure + rank + deck features + aff) at platinum, 3 seeds
                     (dev: sc_gll/scores.npz; vault: refit here with sc_gll.lgbm_scores)
  LGBM counts        LightGBM on the build's card counts only (LGB_PARAMS, early stop on val), 3 seeds, fit here
  blend* X + Y       t1_rescore.blend_star: alpha on val SC-GLL-dl (experienced), two-half cross-fit, per LGBM seed
  blend counts + site FM @0.7   fixed alpha 0.7 (FM weight), no tuning
Judge: t_common contexts raw / dl / pre, subsets experienced / all / match 1 / main build (sc_gll.subsets).

Also written for the site-side builder study: the counts-only LightGBM of seed 0 (LightGBM text), the train-draft
mean / sd of each blend member, the tuned alphas, and test-draft pools (results/.../site_blend/builder.json).
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, torchfree  # noqa: E402

torchfree()
import lightgbm  # noqa: E402,F401  (before anything else that could pull in libomp)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.special import expit  # noqa: E402

SITE = Path("/Users/zblanton/Documents/repos/mtg-draft-sim-site")
NODE_SCORER = Path(__file__).resolve().parent / "site_blend_score.mjs"
DEV = ["MSH", "BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "OTJ", "SOS", "TDM"]
VAULT = ["ECL", "TLA", "WOE", "MOM", "HOB", "TMT"]
N_BOOT, LGB_SEEDS, N_POOLS = 500, 3, 300

SITE_FM, SITE_RAW, CH = "site FM", "site FM raw", "champion"
LG4, LGC = "LGBM r4", "LGBM counts"
B4S, BCS, BCS7 = "blend* r4 + site FM", "blend* counts + site FM", "blend counts + site FM @0.7"
B4C, BCC = "blend* r4 + champion", "blend* counts + champion"
PAIRS = [(CH, SITE_FM), (SITE_FM, SITE_RAW), (SITE_FM, LGC), (SITE_FM, LG4),
         (SITE_FM, B4S), (SITE_FM, BCS), (SITE_FM, BCS7), (BCS, B4S), (BCS7, BCS),
         (CH, B4C), (CH, BCC), (B4S, B4C), (BCS, BCC)]
KEYS = ("experienced dl", "experienced raw", "experienced pre", "match 1 dl", "match 1 pre", "all dl", "main build dl")


def out_dir(s):
    p = REPO / "results" / f"{s}.PremierDraft" / "site_blend"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- the site's model, scored by deck.js

def site_scores(set_code, event, names, X, model=None):
    """[N] clamped and raw deck.js logits for decks X [N, C] in `names` order (model: another deck JSON file)."""
    model = model or SITE / "data" / "deck" / f"{set_code}.json"
    j = json.loads(Path(model).read_text())
    pos = {n: i for i, n in enumerate(j["names"])}
    missing = [n for k, n in enumerate(names) if n not in pos and X[:, k].any()]
    assert not missing, f"cards in decks the site model lacks: {missing[:5]}"
    Y = np.zeros((len(X), len(j["names"])), np.int8)
    cols = [k for k, n in enumerate(names) if n in pos]
    Y[:, [pos[names[k]] for k in cols]] = X[:, cols]
    with tempfile.TemporaryDirectory() as tmp:
        Y.tofile(f"{tmp}/x.bin")
        subprocess.run(["node", str(NODE_SCORER), set_code, event, f"{tmp}/x.bin", f"{tmp}/o.bin", str(model)], check=True)
        o = np.fromfile(f"{tmp}/o.bin").reshape(-1, 2)
    return o[:, 0], o[:, 1]


def site_split(path, draft_ids):
    """The split the site's export used (export_deck_model -> openlane event_split): events in the game file ordered
    by first draft_time, 70 / 15 / 15. -1 for drafts without games."""
    ev = (pl.scan_parquet(path).group_by("draft_id").agg(pl.col("draft_time").min())
          .sort("draft_time", "draft_id").collect())
    n = ev.height
    s = np.zeros(n, np.int8)
    s[int(0.70 * n):] = 1
    s[int(0.85 * n):] = 2
    m = dict(zip(ev["draft_id"].to_list(), s.tolist()))
    return np.array([m.get(x, -1) for x in draft_ids], np.int8)


# ---------------------------------------------------------------- LightGBM arms

def counts_lgbm(deck, d, has, seeds=LGB_SEEDS, log=print, save=None):
    from eval_common import lgbm_model
    X = deck.astype(np.float32)
    tr, va = d.mask("train") & has, d.mask("val") & has
    out, trees = [], []
    for s in range(seeds):
        t0 = time.time()
        m, n = lgbm_model(X, d, tr, va, params={"num_threads": 8}, seed=s)
        out.append(m.predict(X, num_iteration=n, raw_score=True))
        trees.append(int(n))
        if s == 0 and save is not None:
            m.save_model(str(save), num_iteration=n)
        log(f"  counts LGBM seed {s}: {n} trees [{time.time() - t0:.0f}s]")
    return np.stack(out), trees


def r4_lgbm(set_code, d, deck, has, vault, log=print):
    if not vault:
        z = np.load(REPO / "results" / f"{set_code}.PremierDraft" / "sc_gll" / "scores.npz")
        assert (z["draft_id"].astype(str) == np.asarray(d.draft_id).astype(str)).all()
        return z["LightGBM"].astype(np.float64), None
    import sc_gll
    from improve_deck import structure_features
    inputs = dict(np.load(REPO / "results" / "vault" / f"{set_code}.PremierDraft" / "draft_inputs.npz"))
    aff = bool(inputs["aff_present"])
    X = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                   d.rank[:, None].astype(np.float32), inputs["deck_features"]]
                  + ([inputs["aff"][:, None]] if aff else []))
    rank_col = len(d.names) + structure_features(deck[:1], d.names, set_code).shape[1]
    lg = sc_gll.lgbm_scores(X, d, has, inputs["game_logit"], rank_col, n_seeds=LGB_SEEDS, log=log)
    return lg["lgbm ref"], lg["trees"].tolist()


def champion(set_code, d, vault):
    f = (REPO / "results" / "vault" / f"{set_code}.PremierDraft" / "fm_scores.npz" if vault
         else REPO / "results" / f"{set_code}.PremierDraft" / "game_fm_v2" / "scores.npz")
    z = np.load(f)
    assert (z["draft_id"].astype(str) == np.asarray(d.draft_id).astype(str)).all()
    return z["champion"].astype(np.float64)


# ---------------------------------------------------------------- per set

def run_set(set_code, n_boot=N_BOOT, log=print):
    from eval_common import NARROW, deck_rows, load_games, main_builds, per_game
    from mtgdata.games import game_path
    from decks.deckstrength import load_drafts
    from t_common import evaluate_ctx, fit_contexts, game_blocks, save_losses, split_mull
    import sc_gll
    import t1_rescore as t1
    t0 = time.time()
    vault = set_code in VAULT
    out = out_dir(set_code)
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    has, rows, main_bix = main_builds(d, g)
    D = len(d.draft_id)
    blocks = game_blocks(g, pg, D)
    del g
    deck = np.zeros((D, len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    log(f"=== {set_code}: {D} drafts ({has.sum()} with games), {len(pg['y'])} games, vault {vault} "
        f"[{time.time() - t0:.0f}s]")

    # how the site's training split lines up with this repo's (leakage check: site-train games in our val / test)
    ss = site_split(path, d.draft_id)
    leak = {f"our {nm} drafts the site trained on": float(np.mean(ss[has & (d.split == k)] == 0))
            for k, nm in ((1, "val"), (2, "test"))}
    log(f"  split overlap: {leak}")

    clamped, raw = site_scores(set_code, "draft", d.names, deck[has])
    scores = {SITE_FM: np.full((1, D), np.nan), SITE_RAW: np.full((1, D), np.nan)}
    scores[SITE_FM][0, has], scores[SITE_RAW][0, has] = clamped, raw
    scores[CH] = champion(set_code, d, vault)
    scores[LG4], trees4 = r4_lgbm(set_code, d, deck, has, vault, log=log)
    scores[LGC], treesc = counts_lgbm(deck, d, has, log=log, save=out / "lgbm_counts_seed0.txt")
    log(f"  scores ready [{time.time() - t0:.0f}s]")

    ctxs, cinfo, _ = fit_contexts(pg, D, ("raw", "dl", "pre"))
    subs = sc_gll.subsets(pg, has, main_bix)
    fit_val = subs["experienced"][0]
    binfo = {}
    q = lambda m: None  # noqa: E731
    for nm, L, F in ((B4S, LG4, SITE_FM), (BCS, LGC, SITE_FM), (B4C, LG4, CH), (BCC, LGC, CH)):
        scores[nm], binfo[nm] = t1.blend_star(scores[L], scores[F], pg, ctxs["dl"], fit_val, D, has, d.split, log=q)
        log(f"  {nm}: alphas {binfo[nm]['alphas']}")
    ref = has & (d.split == 0)
    zc = np.stack([t1.zstd(s, ref) for s in scores[LGC]])
    scores[BCS7] = 0.3 * zc + 0.7 * t1.zstd(scores[SITE_FM][0], ref)
    log(f"  contexts + blends [{time.time() - t0:.0f}s]")

    res, losses, ngames = evaluate_ctx(scores, pg, ctxs, subs, D, PAIRS, None, n_boot, blocks, set_code, log=q)
    for key in ("experienced dl", "match 1 pre"):
        log(f"  {key}: " + "  ".join(f"{m} {res['models'][m][key]['sc_gll']:.3f}" for m in scores))
        for a, b in PAIRS:
            r = res["pairs"][f"{b} - {a}"][key]
            log(f"      {b} - {a}: {r['delta']:+.3f} ± {r['se']:.3f}")

    # correlations between deck scores on test drafts (how different are the models?)
    te = has & (d.split == 2)
    first = {m: np.atleast_2d(s)[0] for m, s in scores.items()}
    corr = {f"{a} ~ {b}": float(np.corrcoef(first[a][te], first[b][te])[0, 1])
            for a, b in ((SITE_FM, CH), (SITE_FM, LGC), (SITE_FM, LG4), (LGC, LG4), (SITE_FM, BCS), (SITE_FM, B4S))}
    res.update({"set": set_code, "vault": vault, "contexts": cinfo, "blend*": binfo, "split overlap": leak,
                "trees": {"LGBM r4": trees4, "LGBM counts": treesc}, "test score correlations": corr,
                "seconds": time.time() - t0})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    save_losses(out / "losses.npz", losses, ngames)
    np.savez_compressed(out / "scores.npz", draft_id=np.asarray(d.draft_id).astype(str), split=d.split, has=has,
                        **{t1.npz_key(k): np.atleast_2d(v).astype(np.float32) for k, v in scores.items()})

    # for the builder study: blend constants (train-draft mean / sd of each member) and test-draft pools
    def ms(s):
        return [float(np.mean(s[ref])), float(np.std(s[ref]) + 1e-12)]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.flatnonzero(te), min(N_POOLS, te.sum()), replace=False))
    builder = {"set": set_code, "names": list(d.names),
               "site FM": ms(scores[SITE_FM][0]), "LGBM counts": ms(scores[LGC][0]),
               "alpha counts (FM weight)": binfo[BCS]["alphas"][0], "alpha r4 (FM weight)": binfo[B4S]["alphas"][0],
               "pools": [{"draft_id": str(d.draft_id[i]),
                          "pool": {d.names[k]: int(n) for k, n in enumerate(d.pool[i]) if n},
                          "deck": {d.names[k]: int(n) for k, n in enumerate(deck[i]) if n},
                          "skill": None if not np.isfinite(d.skill[i]) else float(d.skill[i]),
                          "wins": float(d.wins[i]), "losses": float(d.losses[i])} for i in pick]}
    # per card: training decks holding it, and the mean skill bucket of their drafters (a skill-marker index)
    sk = np.where(np.isfinite(d.skill), d.skill, np.nan)
    tr_has = ref & np.isfinite(sk)
    inc = deck[tr_has] > 0
    builder["card"] = {"decks": inc.sum(0).astype(int).tolist(),
                       "skill": [float(sk[tr_has][inc[:, k]].mean()) if inc[:, k].sum() >= 50 else None
                                 for k in range(len(d.names))],
                       "skill all": float(sk[tr_has].mean())}
    (out / "builder.json").write_text(json.dumps(builder))
    log(f"{set_code} done [{time.time() - t0:.0f}s]")


# ---------------------------------------------------------------- meta

def meta():
    from eval_protocol import DecisionRule, decide, meta_analysis
    from t_common import meta_rows, one_sided_p
    out = {}
    for group, sets in (("dev", DEV), ("vault", VAULT)):
        files = {s: REPO / "results" / f"{s}.PremierDraft" / "site_blend" / "results.json" for s in sets}
        files = {s: f for s, f in files.items() if f.exists()}
        print(f"\n##### {group}: {len(files)} sets")
        for a, b in PAIRS:
            p = f"{b} - {a}"
            for key in KEYS:
                rs = meta_rows(files, p, key)
                if len(rs) < 2:
                    continue
                m = meta_analysis(rs)
                ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
                out.setdefault(group, {}).setdefault(p, {})[key] = {
                    "re mean": m.re_mean, "ci": list(m.re_ci), "p two-sided": m.re_p_two_sided,
                    "p one-sided": one_sided_p(m), "sets better": f"{m.n_positive}/{len(rs)}",
                    "sig worse": m.n_sig_worse, "per set": {r.name: r.delta for r in rs}, "adopt": ok, "why": why}
                if key in ("experienced dl", "match 1 pre", "experienced raw", "experienced pre"):
                    print(f"{p:55s} {key:16s} {m.re_mean:+.3f} [{m.re_ci[0]:+.3f}, {m.re_ci[1]:+.3f}] "
                          f"p1={one_sided_p(m):.4f} {m.n_positive}/{len(rs)} sigworse={m.n_sig_worse} adopt={ok}")
        al = {s: {k: json.loads(f.read_text())["blend*"][k]["alphas"] for k in (B4S, BCS, B4C, BCC)}
              for s, f in files.items()}
        out.setdefault(group, {})["alphas"] = al
    import site_blend_sealed as sb
    files = {s: REPO / "results" / f"{s}.PremierDraft" / "site_blend" / "sealed.json" for s in DEV + VAULT}
    files = {s: f for s, f in files.items() if f.exists()}
    print(f"\n##### sealed: {len(files)} sets")
    for a, b in sb.PAIRS:
        p = f"{b} - {a}"
        for key in ("experienced dl", "all dl", "experienced pre", "match 1 pre"):
            rs = meta_rows(files, p, key)
            if len(rs) < 2:
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out.setdefault("sealed", {}).setdefault(p, {})[key] = {
                "re mean": m.re_mean, "ci": list(m.re_ci), "p one-sided": one_sided_p(m),
                "sets better": f"{m.n_positive}/{len(rs)}", "sig worse": m.n_sig_worse,
                "per set": {r.name: r.delta for r in rs}, "adopt": ok, "why": why}
            print(f"{p:55s} {key:16s} {m.re_mean:+.3f} [{m.re_ci[0]:+.3f}, {m.re_ci[1]:+.3f}] "
                  f"p1={one_sided_p(m):.4f} {m.n_positive}/{len(rs)} sigworse={m.n_sig_worse} adopt={ok}")
    if files:
        out.setdefault("sealed", {})["alphas"] = {s: {k: json.loads(f.read_text())["blend*"][k]["alphas"]
                                                      for k in (sb.BS, sb.BD, sb.BDD)} for s, f in files.items()}
    dst = REPO / "results" / "meta" / "site_blend.json"
    dst.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["meta"]:
        meta()
    elif args[:1] == ["sealed"]:
        import site_blend_sealed
        for s in args[1:]:
            site_blend_sealed.run_set(s)
    else:
        nb = N_BOOT
        if "--boot" in args:
            i = args.index("--boot")
            nb = int(args[i + 1])
            args = args[:i] + args[i + 2:]
        for s in args:
            run_set(s, nb)
