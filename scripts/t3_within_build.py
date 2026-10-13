"""T3 (meta_review_2): the within-build referee -- a conditional logit with draft fixed effects.

    python scripts/t3_within_build.py SET [SET ...] [--min-games 1] [--splits test] [--seeds 3] [--threads 4]
    python scripts/t3_within_build.py meta SET ...      (pooled over sets: FE + RE (DL/HKSJ) of every b and pair)
    python scripts/t3_within_build.py --selftest

For drafts with >= 2 builds that each played >= k games (k = --min-games, default 1):
    logit P(win_g) = alpha_draft + c(record state before the game, play/draw, opponent mulligans) + b * s(build_g)
alpha_draft is conditioned out exactly (Chamberlain's conditional likelihood, sum of wins per draft as the
sufficient statistic; elementary symmetric polynomials by dynamic programming), so player, pool, rank, day and every
player variable cancel; no skill bucket and no own mulligans enter, so mana-base mulligans count as deck strength.
b is reported (i) in raw score units: b ~ 1 means the score's between-build differences predict exactly as much
within a player as the score's scale claims (game-logit scores), b < 1 = the score over-states deck differences or
carries non-deck signal; (ii) per sd of the score over test drafts' main builds: the comparable number across models.
Simulated calibration (rebuilds after losses, record matching): a correctly scaled score gets b ~ 1.2-1.25 with the
record-state control (fixed-effects logit with a lagged-outcome regressor is biased up) and ~0.85-1.0 with the
match-number control (run as a sensitivity: "models (match control)", pair key "within-build b (match control)").
Both biases are close to common scale factors, so model-vs-model PAIRS are the decision quantity, not b vs 1.
b is compared across models (iii) per WITHIN-draft sd of the score (scale-free and blind to any draft-constant
component such as a skill leak). SEs are draft-cluster sandwich SEs; pairs (B - A in b per within-sd) use the stacked influence functions on the same drafts,
plus the seed spread as a variance component (eval_protocol.SetResult fields, sc_gll meta format).
Scores per build:
  FM arms        t2_mull_split/build_scores.npz (champion, split mull, - own mull, refit, ...; T2 must run first)
  LightGBM       round 4 settings (counts + structure + rank + deck features + aff, sc_gll.lgbm_scores' recipe),
                 3 seeds, fit on train drafts' most-played builds (early stop on val) and predicted for EVERY build at
                 the platinum reference rank; the build's deck features + affinity are T2's Eb (the FM's own columns)
  transformer    any results/{SET}.PremierDraft/game_transformer/build_scores*.npz with the same layout (if present)
Gate (T0): >= 1,000 qualifying test drafts per set, or >= 5,000 pooled; reported, never silently skipped.
LightGBM process: torch-free decks.deckstrength shim (never torch and lightgbm in one process).
Writes results/{SET}.PremierDraft/t3_within_build/results.json (and scores.npz with LightGBM's per-build scores).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, assert_no_torch, npz_key, torchfree  # noqa: E402

torchfree()
if __name__ == "__main__" and "--selftest" not in sys.argv and sys.argv[1:2] != ["meta"]:
    import lightgbm  # noqa: F401
import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

M_MAX = 12          # games per draft kept (match order); 17Lands duplicates beyond that are dropped
N_SEEDS, THREADS = 3, 4
GATE_SET, GATE_POOLED = 1000, 5000
CH, LG = "champion", "LightGBM"


# ---------------------------------------------------------------- conditional logit

def groups(gd, match, sel):
    """Padded [N, M] game indices of the selected games per draft (match order), -1 = empty; and draft of each row."""
    idx = np.flatnonzero(sel)
    o = idx[np.lexsort((match[idx], gd[idx]))]
    gs = gd[o]
    start = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    size = np.diff(np.r_[start, len(o)])
    slot = np.arange(len(o)) - np.repeat(start, size)
    keep = slot < M_MAX
    M = int(min(size.max(), M_MAX)) if len(size) else 1
    row = np.repeat(np.arange(len(start)), size)
    G = np.full((len(start), M), -1)
    G[row[keep], slot[keep]] = o[keep]
    return G, gs[start], int((~keep).sum())


def _esp(a, mask):
    """log elementary symmetric polynomials e_0..e_M of exp(a) over masked items: [N, M+1]; prefix/suffix tables."""
    N, M = a.shape
    neg = -np.inf
    pre = np.full((N, M + 1, M + 1), neg)
    pre[:, 0, 0] = 0.0
    for i in range(M):
        cur = pre[:, i].copy()
        inc = np.full_like(cur, neg)
        inc[:, 1:] = cur[:, :-1] + a[:, i:i + 1]
        pre[:, i + 1] = np.where(mask[:, i:i + 1], np.logaddexp(cur, inc), cur)
    suf = np.full((N, M + 1, M + 1), neg)
    suf[:, M, 0] = 0.0
    for i in range(M - 1, -1, -1):
        cur = suf[:, i + 1].copy()
        inc = np.full_like(cur, neg)
        inc[:, 1:] = cur[:, :-1] + a[:, i:i + 1]
        suf[:, i] = np.where(mask[:, i:i + 1], np.logaddexp(cur, inc), cur)
    return pre, suf


def clogit_terms(beta, Xg, Yg, Mk):
    """Per-group conditional log-likelihood [N] and score contributions [N, P]."""
    a = np.where(Mk, Xg @ beta, 0.0)
    k = (Yg * Mk).sum(1).astype(int)
    pre, suf = _esp(a, Mk)
    N, M = a.shape
    logZ = pre[np.arange(N), M, k]
    ll = (Yg * a * Mk).sum(1) - logZ
    # P(item i in the winning set | k) = exp(a_i) e_{k-1}(others) / e_k, with e_{k-1}(others) the convolution of
    # the prefix (items < i) and suffix (items > i) polynomials at degree k - 1 (vectorised over groups and items)
    km1 = k - 1
    t = np.arange(M + 1)
    j = km1[:, None] - t[None, :]                                  # [N, M+1] suffix degree for each prefix degree t
    jj = np.clip(j, 0, M)
    Pt = pre[:, :M, :]                                             # [N, M, M+1]
    St = np.take_along_axis(suf[:, 1:, :], np.broadcast_to(jj[:, None, :], (N, M, M + 1)), axis=2)
    St = np.where((j >= 0)[:, None, :], St, -np.inf)
    val = logsumexp(Pt + St, axis=2)                               # [N, M]
    val = np.where((k > 0)[:, None], val, -np.inf)
    with np.errstate(invalid="ignore"):
        pi = np.where(Mk, np.exp(a + val - logZ[:, None]), 0.0)
    sc = ((Yg - pi) * Mk)[..., None] * Xg
    return ll, sc.sum(1)


def clogit(Xg, Yg, Mk, beta0=None, iters=200, tol=1e-7, ridge=1e-8):
    """Maximum conditional likelihood: BHHH quasi-Newton steps (outer product of the analytic per-group scores) with a
    halving line search, then the finite-difference Hessian of the analytic score at the optimum (for the sandwich).
    Returns (beta, H, per-group scores at beta, log-lik)."""
    P = Xg.shape[-1]
    beta = np.zeros(P) if beta0 is None else beta0.copy()
    ll, sc = clogit_terms(beta, Xg, Yg, Mk)
    for _ in range(iters):
        g = sc.sum(0)
        step = np.linalg.solve(sc.T @ sc + ridge * np.eye(P), g)
        t, L0 = 1.0, ll.sum()
        while True:
            ll_n, sc_n = clogit_terms(beta + t * step, Xg, Yg, Mk)
            if ll_n.sum() >= L0 - 1e-12 or t < 1e-6:
                break
            t /= 2
        beta, ll, sc = beta + t * step, ll_n, sc_n
        if abs(ll.sum() - L0) < tol and np.abs(t * step).max() < 1e-5:
            break
    return beta, hessian(beta, Xg, Yg, Mk) + ridge * np.eye(P), sc, float(ll.sum())


def hessian(beta, Xg, Yg, Mk, eps=1e-5):
    P = len(beta)
    H = np.zeros((P, P))
    for j in range(P):
        e = np.zeros(P)
        e[j] = eps
        gp = clogit_terms(beta + e, Xg, Yg, Mk)[1].sum(0)
        gm = clogit_terms(beta - e, Xg, Yg, Mk)[1].sum(0)
        H[:, j] = -(gp - gm) / (2 * eps)
    return (H + H.T) / 2


def design(pg, G, control="state"):
    """[N, M, P] context: record state one-hot without 0-0 ("state", meta_review_2's pre-registered referee) or match
    number one-hot without match 1 ("match", sensitivity), plus play/draw and opponent mulligans; y and mask."""
    from eval_common import N_STATES
    Mk = G >= 0
    j = np.where(Mk, G, 0)
    if control == "state":
        st = np.eye(N_STATES + 1)[pg["state"][j]][..., 1:]
    elif control == "match":
        st = np.eye(M_MAX)[np.clip(pg["match"][j] - 1, 0, M_MAX - 1)][..., 1:]
    else:
        st = np.zeros(j.shape + (0,))
    Xc = np.concatenate([st, pg["on_play"][j][..., None], pg["opp_mull"][j][..., None]], -1) * Mk[..., None]
    return Xc, np.where(Mk, pg["y"][j], 0.0), Mk


def within_build(pg, G, score_game, sd, Xc, Yg, Mk, null):
    """b for one score (per game: its build's score). The score is standardised by its game-weighted WITHIN-draft sd
    over the informative drafts (constant-within-draft components, e.g. a skill leak, neither count nor inflate it):
    "b per within-sd" is the scale-free number compared across models. Returns (dict, influence of b per within-sd
    per draft)."""
    j = np.where(Mk, G, 0)
    s = np.where(Mk, score_game[j], 0.0)
    n = Mk.sum(1)
    dev = np.where(Mk, s - (s * Mk).sum(1, keepdims=True) / n[:, None], 0.0)
    sd_w = float(np.sqrt((dev ** 2).sum() / n.sum())) + 1e-12
    X = np.concatenate([(dev / sd_w)[..., None], Xc], -1)
    beta, H, sc, ll = clogit(X, Yg, Mk, beta0=np.r_[0.0, null[0]])
    IF = sc @ np.linalg.inv(H).T                      # [N, P] influence of each draft on beta
    b, se = float(beta[0]), float(np.sqrt(IF[:, 0] @ IF[:, 0]))
    return {"b per within-sd": b, "se per within-sd": se, "b raw": b / sd_w, "se raw": se / sd_w,
            "b per sd": b / sd_w * sd, "within-draft sd": sd_w, "score sd": float(sd),
            "within / overall sd": sd_w / sd, "LR chi2 (score vs context only)": float(2 * (ll - null[1])),
            "loglik": ll}, IF[:, 0]


def informative(G, Yg, Mk, game_build):
    """Groups whose outcome varies AND that played >= 2 distinct builds (the only ones that identify b)."""
    k = (Yg * Mk).sum(1)
    n = Mk.sum(1)
    j = np.where(Mk, G, 0)
    gb = np.where(Mk, game_build[j], -1)
    nb = np.array([len(set(r[r >= 0].tolist())) for r in gb])
    return (k > 0) & (k < n) & (nb >= 2)


def analyse(pg, game_build, build_draft, D, split, models, sd_ref, min_games=1, splits=(2,), control="state",
            log=print):
    """models {name: [S, B] per-build scores}; sd_ref {name: score sd (test drafts' main builds)}."""
    t0 = time.time()
    nb = np.bincount(game_build, minlength=len(build_draft))
    ok_b = nb >= min_games
    nq = np.bincount(build_draft[ok_b], minlength=D)
    qual = (nq >= 2) & np.isin(split, splits)
    sel = qual[pg["gd"]]
    G, gdraft, dropped = groups(pg["gd"], pg["match"], sel)
    Xc, Yg, Mk = design(pg, G, control)
    inf = informative(G, Yg, Mk, game_build)
    G, gdraft, Xc, Yg, Mk = G[inf], gdraft[inf], Xc[inf], Yg[inf], Mk[inf]
    res = {"control": control, "min games per build": min_games, "splits": list(map(int, splits)),
           "qualifying drafts": int(qual.sum()), "informative drafts (outcome varies, >= 2 builds played)": int(len(G)),
           "games in informative drafts": int(Mk.sum()), "games beyond M_MAX dropped": dropped, "models": {}, "pairs": {}}
    IFs = {}
    b0, _, _, l0 = clogit(Xc, Yg, Mk)
    null = (b0, l0)
    for m, S in models.items():
        rows, ifs = [], []
        for s in np.atleast_2d(S):
            r, IF = within_build(pg, G, np.nan_to_num(s)[game_build], sd_ref[m], Xc, Yg, Mk, null)
            rows.append(r)
            ifs.append(IF)
        o = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        o["seeds"] = len(rows)
        o["b by seed"] = [r["b per within-sd"] for r in rows]
        o["seed se"] = float(np.std(o["b by seed"], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0
        res["models"][m] = o
        IFs[m] = np.stack(ifs)
        log(f"  {m:28s} b {o['b raw']:+.3f} ± {o['se raw']:.3f} (raw units)   b per within-sd "
            f"{o['b per within-sd']:+.4f} ± {o['se per within-sd']:.4f}   within/overall sd {o['within / overall sd']:.2f}"
            f"   LR {o['LR chi2 (score vs context only)']:.1f}")
    res["seconds"] = time.time() - t0
    return res, IFs


def pair_results(res, IFs, pairs, key="within-build b", log=print):
    for a, b in pairs:
        if a not in IFs or b not in IFs:
            continue
        A, B = res["models"][a], res["models"][b]
        ia, ib = IFs[a].mean(0), IFs[b].mean(0)
        se_boot = float(np.sqrt(np.sum((ib - ia) ** 2)))
        if len(IFs[a]) == len(IFs[b]) and len(IFs[a]) > 1:
            dd = np.array(B["b by seed"]) - np.array(A["b by seed"])
            se_seed = float(dd.std(ddof=1) / np.sqrt(len(dd)))
        else:
            se_seed = float(np.hypot(A["seed se"], B["seed se"]))
        delta = B["b per within-sd"] - A["b per within-sd"]
        res["pairs"].setdefault(f"{b} - {a}", {})[key] = {
            "name": "", "delta": delta, "se": float(np.hypot(se_boot, se_seed)), "se_boot": se_boot, "se_seed": se_seed,
            "n_units": res["informative drafts (outcome varies, >= 2 builds played)"],
            "n_seeds": max(len(IFs[a]), len(IFs[b]))}
        r = res["pairs"][f"{b} - {a}"][key]
        log(f"      {b} - {a}: {delta:+.4f} ± {r['se']:.4f} (b per within-sd)")
    return res


# ---------------------------------------------------------------- real data

def lgbm_build_scores(set_code, d, has, main_build, Xb, Eb, inp, n_seeds=N_SEEDS, log=print):
    """Round 4 LightGBM (sc_gll's features and settings), fit on most-played builds, scored for every build at the
    platinum reference rank. Returns [S, B]."""
    from eval_common import REF_RANK, lgbm_model
    from improve_deck import structure_features
    deck = np.zeros((len(d.draft_id), Xb.shape[1]), np.int8)
    deck[has] = Xb[main_build[has]]
    Xd = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                    d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
    Xbb = np.hstack([Xb.astype(np.float32), structure_features(Xb, d.names, set_code),
                     np.full((len(Xb), 1), REF_RANK, np.float32), Eb[:, :inp["deck_features"].shape[1]],
                     Eb[:, -1:]]).astype(np.float32)
    assert Eb.shape[1] == inp["deck_features"].shape[1] + 1, "T2 build columns must be deck features + affinity"
    tr, va = d.mask("train") & has, d.mask("val") & has
    out = []
    for s in range(n_seeds):
        t1 = time.time()
        m, n = lgbm_model(Xd, d, tr, va, params={"num_threads": THREADS}, seed=s)
        p = np.clip(m.predict(Xbb, num_iteration=n), 1e-6, 1 - 1e-6)
        out.append(np.log(p / (1 - p)))
        log(f"  lgbm seed {s}: {n} trees [{time.time() - t1:.0f}s]")
    return np.stack(out)


def main(argv):
    from eval_common import load_games, per_game
    from mtgdata.games import game_path
    from t_common import split_mull
    from decks.deckstrength import load_drafts
    import polars as pl
    global THREADS
    min_games, splits, n_seeds, sets = 1, (2,), N_SEEDS, []
    it = iter(argv)
    for a in it:
        if a == "--min-games":
            min_games = int(next(it))
        elif a == "--splits":
            splits = tuple({"train": 0, "val": 1, "test": 2}[x] for x in next(it).split(","))
        elif a == "--seeds":
            n_seeds = int(next(it))
        elif a == "--threads":
            THREADS = int(next(it))
        else:
            sets.append(a)
    for set_code in sets:
        t0 = time.time()
        res_dir = REPO / "results" / f"{set_code}.PremierDraft"
        bf = res_dir / "t2_mull_split" / "build_scores.npz"
        if not bf.exists():
            raise SystemExit(f"{set_code}: {bf} missing -- run t2_mull_split.py first")
        bz = np.load(bf)
        d = load_drafts(set_code, "PremierDraft")
        path = game_path(set_code, "PremierDraft")
        g = load_games(path, d.draft_id)
        pg = per_game(d, g)
        pg["own_mull"], pg["opp_mull"] = split_mull(g)
        key = {k: j for j, k in enumerate(zip(bz["build_draft_id"].tolist(), bz["build_index"].tolist()))}
        game_build = np.array([key[k] for k in zip(g["draft_id"].to_list(), g["build_index"].cast(pl.Utf8).to_list())])
        from eval_common import deck_rows
        first = (g.select(pl.Series("b", game_build)).with_row_index("i").group_by("b").agg(pl.col("i").min())
                 .sort("b"))
        assert (first["b"].to_numpy() == np.arange(len(bz["build_index"]))).all(), "T2 builds not all present"
        Xb = deck_rows(path, g["row"].to_numpy()[first["i"].to_numpy()].astype(np.int64), d.names)
        del g
        D = len(d.draft_id)
        main_build = bz["main_build"]
        has = main_build >= 0
        build_draft = bz["build_draft"]
        assert (np.asarray(d.draft_id).astype(str)[build_draft] == bz["build_draft_id"]).all()
        models = {k.replace("_", " "): bz[k].astype(np.float64) for k in bz.files
                  if k not in ("build_draft_id", "build_index", "build_draft", "main_build", "Eb")}
        inp = np.load(res_dir / "deck_synergy_v4" / "draft_inputs.npz")
        Eb = bz["Eb"]
        nf = inp["deck_features"].shape[1]
        diff = float(np.abs(Eb[main_build[has], :nf] - inp["deck_features"][has]).max())
        print(f"=== {set_code}: {D} drafts, {len(pg['y'])} games, {len(Xb)} builds; T2 deck columns vs draft_inputs "
              f"max |diff| {diff:.2e} [{time.time() - t0:.0f}s]", flush=True)
        models[LG] = lgbm_build_scores(set_code, d, has, main_build, Xb, Eb, inp, n_seeds,
                                       log=lambda m: print(m, flush=True))
        for f in sorted((res_dir / "game_transformer").glob("build_scores*.npz")):
            tz = np.load(f)
            tk = {k: j for j, k in enumerate(zip(tz["build_draft_id"].tolist(), tz["build_index"].tolist()))}
            order = np.array([tk.get(k, -1) for k in zip(bz["build_draft_id"].tolist(), bz["build_index"].tolist())])
            for k in tz.files:
                if k.startswith("transformer") and (order >= 0).all():
                    models[f"{k.replace('_', ' ')} [{f.stem}]"] = np.atleast_2d(tz[k])[:, order].astype(np.float64)
        te_main = np.flatnonzero(has & (d.split == 2))
        sd_ref = {m: float(np.mean([np.std(s[main_build[te_main]]) for s in np.atleast_2d(S)])) for m, S in models.items()}
        pairs = [(LG, CH)] + [(CH, m) for m in models if m not in (CH, LG)] + [("split mull", "- own mull")]
        res, IFs = analyse(pg, game_build, build_draft, D, d.split, models, sd_ref, min_games, splits, "state",
                           log=lambda m: print(m, flush=True))
        pair_results(res, IFs, pairs, log=lambda m: print(m, flush=True))
        print("  sensitivity: match-number control instead of the record state", flush=True)
        r2, IF2 = analyse(pg, game_build, build_draft, D, d.split, models, sd_ref, min_games, splits, "match",
                          log=lambda m: print(m, flush=True))
        pair_results(r2, IF2, pairs, key="within-build b (match control)", log=lambda m: print(m, flush=True))
        res["models (match control)"] = r2["models"]
        for p_, v in r2["pairs"].items():
            res["pairs"].setdefault(p_, {}).update(v)
        t0c = res_dir / "t0_counts.json"
        res["T0 gate"] = json.loads(t0c.read_text()).get(f"T3 gate per set (k={min_games}, >= {GATE_SET} test drafts)") \
            if t0c.exists() else None
        res["gate (this run)"] = res["qualifying drafts"] >= GATE_SET
        out = res_dir / "t3_within_build"
        out.mkdir(parents=True, exist_ok=True)
        tag = "" if (min_games, splits) == (1, (2,)) else f"_k{min_games}_" + "".join(map(str, splits))
        (out / f"results{tag}.json").write_text(json.dumps({"set": set_code, **res, "total seconds": time.time() - t0},
                                                           indent=1, default=float))
        np.savez_compressed(out / "scores.npz", build_draft_id=bz["build_draft_id"], build_index=bz["build_index"],
                            **{npz_key(LG): models[LG].astype(np.float32)})
        assert_no_torch()
        print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


def meta(*args):
    from eval_protocol import DecisionRule, decide, meta_analysis
    from t_common import meta_rows, one_sided_p
    rel, sets = "t3_within_build/results.json", []
    it = iter(args)
    for a in it:
        if a == "--file":
            rel = next(it)
        else:
            sets.append(a)
    files = {s: REPO / "results" / f"{s}.PremierDraft" / rel for s in sets}
    files = {s: f for s, f in files.items() if f.exists()}
    R = {s: json.loads(f.read_text()) for s, f in files.items()}
    nq = sum(r["qualifying drafts"] for r in R.values())
    print(f"{rel}: {list(R)}; pooled qualifying drafts {nq} (gate {GATE_POOLED}: {'PASS' if nq >= GATE_POOLED else 'FAIL'})")
    out = {"sets": list(R), "pooled qualifying drafts": nq, "models": {}, "pairs": {}}
    for m in sorted({m for r in R.values() for m in r["models"]}):
        b = np.array([r["models"][m]["b raw"] for r in R.values() if m in r["models"]])
        se = np.array([np.hypot(r["models"][m]["se raw"], r["models"][m]["seed se"] / r["models"][m]["within-draft sd"])
                       for r in R.values() if m in r["models"]])
        w = 1 / se ** 2
        fe, fse = float((w * b).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
        out["models"][m] = {"b raw FE": fe, "se": fse, "per set": b.tolist()}
        print(f"  {m:30s} pooled b (raw units) {fe:+.3f} ± {fse:.3f}  over {len(b)} sets")
    for p, key in [(p, k) for p in sorted({p for r in R.values() for p in r["pairs"]})
                   for k in ("within-build b", "within-build b (match control)")]:
        rs = meta_rows(files, p, key)
        if len(rs) < 2:
            continue
        mm = meta_analysis(rs)
        ok, why = decide(mm, DecisionRule(min_positive=min(8, len(rs))))
        out["pairs"].setdefault(p, {})[key] = {"re mean": mm.re_mean, "ci": list(mm.re_ci), "p two-sided": mm.re_p_two_sided,
                           "p one-sided (B lower)": 1 - one_sided_p(mm), "sets better": f"{mm.n_positive}/{len(rs)}",
                           "decide": ok, "why": why}
        print(f"\n{p} [{key}]\n{mm.summary()}\n  leakage veto (one-sided p that B lowers b): {1 - one_sided_p(mm):.3f}")
    dst = REPO / "results" / "meta" / "t3_within_build.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float))


# ---------------------------------------------------------------- synthetic self-test

def selftest():
    from itertools import combinations
    t0 = time.time()
    rng = np.random.default_rng(0)
    # 1. the DP likelihood and its score equal brute-force enumeration on small groups
    N, M, P = 40, 5, 3
    Xg = rng.normal(size=(N, M, P))
    Mk = rng.random((N, M)) < 0.8
    Mk[:, 0] = True
    Yg = (rng.random((N, M)) < 0.5) & Mk
    beta = rng.normal(0, 0.5, P)
    ll, sc = clogit_terms(beta, Xg, Yg.astype(float), Mk)
    for n in range(N):
        items = np.flatnonzero(Mk[n])
        a = Xg[n, items] @ beta
        k = int(Yg[n, items].sum())
        subs = list(combinations(range(len(items)), k))
        lz = logsumexp([a[list(s)].sum() for s in subs])
        assert abs(ll[n] - (a[Yg[n, items]].sum() - lz)) < 1e-9
        w = np.exp([a[list(s)].sum() - lz for s in subs])
        pi = np.array([sum(wi for wi, s in zip(w, subs) if i in s) for i in range(len(items))])
        g = ((Yg[n, items] - pi)[:, None] * Xg[n, items]).sum(0)
        assert np.allclose(sc[n], g, atol=1e-9)
    print(f"[1] conditional likelihood + score == brute force on {N} groups [{time.time() - t0:.1f}s]")
    # 2. synthetic drafts: player skill + per-build deck strength, record matching, rebuilds after losses
    D = 2000
    skill = rng.normal(0, 0.5, D)
    rows = []
    nb = rng.choice([1, 2, 3], D, p=[0.4, 0.4, 0.2])
    bstr, bd = [], []
    for i in range(D):
        base = rng.normal(0, 0.3)
        ids = []
        for k in range(nb[i]):
            ids.append(len(bstr))
            bstr.append(base + rng.normal(0, 0.35))
            bd.append(i)
        w = l = 0
        cur = 0
        while w < 7 and l < 3 and len(rows) < 10 ** 7:
            if l > 0 and cur < nb[i] - 1 and rng.random() < 0.5:
                cur += 1
            op = rng.random() < 0.5
            om = float(rng.random() < 0.1)
            z = bstr[ids[cur]] + skill[i] + 0.2 * (op - 0.5) + 0.3 * om - 0.15 * (w - l)
            win = rng.random() < 1 / (1 + np.exp(-z))
            rows.append((i, ids[cur], w + l + 1, min(w, 6) * 3 + min(l, 2), op - 0.5, om, float(win)))
            w, l = w + win, l + (not win)
    gd, gb, match, state, onp, om, y = (np.array(c) for c in zip(*rows))
    bstr, bd = np.array(bstr), np.array(bd)
    pg = {"gd": gd, "match": match, "state": state, "on_play": onp, "opp_mull": om, "y": y}
    split = np.full(D, 2, np.int8)
    models = {"truth": bstr[None], "half scale": (0.5 * bstr)[None], "truth + skill leak": (bstr + 0.8 * skill[bd])[None],
              "noisy": np.stack([bstr + rng.normal(0, 0.3, len(bstr)) for _ in range(2)])}
    main_build = np.array([np.flatnonzero(bd == i)[0] for i in range(D)])
    sd_ref = {m: float(np.mean([np.std(s[main_build]) for s in S])) for m, S in models.items()}
    res, IFs = analyse(pg, gb, bd, D, split, models, sd_ref)
    pair_results(res, IFs, [("truth", "noisy"), ("truth", "truth + skill leak")])
    r = res["models"]
    print(f"[2] informative drafts {res['informative drafts (outcome varies, >= 2 builds played)']}; b raw: "
          + "  ".join(f"{m} {o['b raw']:+.2f}±{o['se raw']:.2f}" for m, o in r.items()))
    # known bias of the fixed-effects logit with a lagged-outcome control (record state): b ~ 1.2-1.25 for a
    # correctly scaled score in this world (simulated over seeds; ~0.85-1.0 with the match-number control)
    assert 0.9 < r["truth"]["b raw"] < 1.7
    assert abs(r["half scale"]["b raw"] - 2 * r["truth"]["b raw"]) < 1e-3
    assert abs(r["truth + skill leak"]["b raw"] - r["truth"]["b raw"]) < 1e-6   # a draft-constant leak cancels
    assert abs(r["half scale"]["b per within-sd"] - r["truth"]["b per within-sd"]) < 1e-3   # scale-free
    assert abs(r["truth + skill leak"]["b per within-sd"] - r["truth"]["b per within-sd"]) < 1e-3  # leak-free
    assert r["noisy"]["b raw"] < r["truth"]["b raw"]
    dp = res["pairs"]["noisy - truth"]["within-build b"]
    assert dp["delta"] < 0 and dp["se"] > 0 and dp["n_seeds"] == 2
    r2, _ = analyse(pg, gb, bd, D, split, {"truth": models["truth"]}, sd_ref, control="match", log=lambda m: None)
    print(f"[3] match-number control: truth b raw {r2['models']['truth']['b raw']:+.2f} ± "
          f"{r2['models']['truth']['se raw']:.2f}")
    assert 0.6 < r2["models"]["truth"]["b raw"] < 1.3
    json.dumps(res, default=float)
    import tempfile
    global REPO
    old = REPO
    with tempfile.TemporaryDirectory() as tmp:
        REPO = Path(tmp)
        try:
            for sname in ("S1", "S2"):
                f = REPO / "results" / f"{sname}.PremierDraft" / "t3_within_build" / "results.json"
                f.parent.mkdir(parents=True)
                f.write_text(json.dumps(res, default=float))
            meta("S1", "S2")
            assert (REPO / "results" / "meta" / "t3_within_build.json").exists()
        finally:
            REPO = old
    assert_no_torch()
    print(f"t3 self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        main(sys.argv[1:])
