"""Can a single game-level FM recover blend*'s card-value gain?   (blend_puzzle.md "next step")

    python scripts/fm_cardvalue.py prep SET [SET ...]     LightGBM counts-only (torch-free process, lightgbm)
    python scripts/fm_cardvalue.py SET [SET ...] [--seeds 2] [--boot 300] [--threads 4]   FM arms (torch, no lightgbm)
    python scripts/fm_cardvalue.py meta SET ...           12-set RE meta / decide -> results/meta/fm_cardvalue.json

blend* = (1 - a) z_LGBM + a z_champion beats the champion game FM by +0.13 (dev) / +0.23 (vault), SC-GLL experienced-dl;
counts-only LightGBM recovers 77% of it (LightGBM's per-card valuations: it shrinks the FM's strong/rare cards and
promotes cards skilled drafters play). Arms (every one = the champion recipe of game_fm_v2: fm4 nonland + deck features
+ aff, champion controls, Adam 3e-3, batch 4096, patience 3, 2 lr drops, early stopping on VAL GAME log loss; l2_v =
the champion's val-chosen l2_v; only the listed term changes):
  l2 grid     card-power L2 l2_w in {1e-6, 3e-6, 1e-5 (champion), 3e-5, 1e-4} (+ 3e-4, 1e-3); picked twice: on val game log loss
              ("l2 [val-gll pick]") and on val SC-GLL-dl ("l2 [val-scgll pick]")
  draft aux   + lambda * binomial draft-record NLL (per game) of the same deck-only score on TRAIN drafts (main build),
              calibrated like improve_deck: logit = r[rank] + beta * s(deck) with r, beta learned jointly; lambda in
              {0.1, 0.3} (+ 1.0) -- the joint game + draft likelihood
  lgbm prior  card power penalised toward kappa * c (and initialised there) instead of toward 0: c_j = LightGBM
              counts-only marginal log-odds per copy (mean over train decks holding card j of logit f(x) - logit f(x -
              e_j)), kappa = sd(champion w) / sd(c) over played cards (scale matched); l2 in {1e-5, 1e-4, 1e-3} (+ 1e-2)
  distill     + gamma * mean_d (s_d - t_d)^2 / sd^2 over TRAIN drafts, t_d = mean + sd * z(cross-fitted (2-fold)
              LightGBM counts-only logit) in the champion's deck-score units (its train mean / sd); gamma in {0.003, 0.01, 0.03, 0.1}
              (pilot: 0.1 / 0.3 swamped the game likelihood)
Each family's "[pick]" = setting with the best VAL SC-GLL-dl (experienced, dl context, two-half cross-fit over val
drafts as t1_rescore.blend_star), seed 0. Seed 0 for every setting; seeds 1..S-1 (--seeds) for the champion and the picked settings only.
Scoring: deck-only score (game_fm_v2.deck_logit) under the frozen T1 contexts dl and pre (t_common.fit_contexts),
subsets experienced (primary) and match 1, (a, b) on val, test games; keys "experienced dl" (PRIMARY), "experienced
pre", "match 1 dl", "match 1 pre". Pairs: X - champion (refit here) for every X; X - blend* for the picks, where blend*
= the saved T1 blend* (t1_rescore/blend.npz, 3 seeds) and "blend* (saved alpha, this champion)" = the saved
LightGBM with T1's alpha* per seed and this run's champion (reproduction check: r with game_fm_v2's champion).
Diagnostics: card-power r / sd ratio vs the champion and vs c; r of test deck scores with LightGBM counts-only.
Writes results/{SET}.PremierDraft/fm_cardvalue/{lgbm_prep.npz, results.json, scores.npz}.
"""
import os
import sys

_T = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--threads=")), "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]

L2_GRID = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)   # 3e-4, 1e-3: extension (MSH pick at the 1e-4 edge)
AUX_W = (0.1, 0.3, 1.0)                                # 1.0: extension (MSH: gain rose 0.1 -> 0.3)
PRIOR_L2 = (1e-5, 1e-4, 1e-3, 1e-2)                     # 1e-2: extension (MSH pick at the 1e-3 edge)
DISTILL_W = (0.003, 0.01, 0.03, 0.1)   # MSH pilot: 0.1 / 0.3 wrecked the game fit (-0.35 / -0.42)
CH = "champion"
BL_SAVED, BL_HERE = "blend* (saved T1)", "blend* (saved alpha, this champion)"
KEYS = ("experienced dl", "experienced pre", "match 1 dl", "match 1 pre")
PRIMARY = "experienced dl"
N_BOOT = 300


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def zstd_ref(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-12)


def out_dir(s):
    return REPO / "results" / f"{s}.PremierDraft" / "fm_cardvalue"


# ================================================================ prep: LightGBM counts-only (torch-free)

def prep(set_code, log=print):
    from t_common import assert_no_torch, peak_rss_gb, torchfree
    torchfree()
    import lightgbm  # noqa: F401
    from eval_common import deck_rows, lgbm_model, load_games, main_builds
    from mtgdata.games import game_path
    from decks.deckstrength import load_drafts
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id)
    has, rows, _ = main_builds(d, g)
    del g
    D, C = len(d.draft_id), len(d.names)
    deck = np.zeros((D, C), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    X = deck.astype(np.float32)
    tr, va = d.mask("train") & has, d.mask("val") & has
    P = {"num_threads": 4}
    m, n = lgbm_model(X, d, tr, va, params=P, seed=0)
    full = logit(m.predict(X, num_iteration=n))
    log(f"  {set_code}: counts-only LightGBM {n} trees [{time.time() - t0:.0f}s]")
    # 2-fold cross-fit on train drafts (out-of-fold train logits for distillation)
    oof = full.copy()
    trd = np.flatnonzero(tr)
    fold = np.zeros(D, np.int8) - 1
    fold[trd] = np.random.default_rng(0).permutation(len(trd)) % 2
    trees = [int(n)]
    for k in (0, 1):
        mk, nk = lgbm_model(X, d, tr & (fold != k), va, params=P, seed=0)
        sel = tr & (fold == k)
        oof[sel] = logit(mk.predict(X[sel], num_iteration=nk))
        trees.append(int(nk))
    log(f"  cross-fit trees {trees[1:]}; r(oof, in-sample) on train {np.corrcoef(oof[tr], full[tr])[0, 1]:.3f} "
        f"[{time.time() - t0:.0f}s]")
    # per-card marginal log-odds per copy
    rng = np.random.default_rng(1)
    c, nd = np.zeros(C), np.zeros(C, np.int64)
    for j in range(C):
        idx = np.flatnonzero(tr & (deck[:, j] > 0))
        nd[j] = len(idx)
        if len(idx) < 30:
            continue
        if len(idx) > 1500:
            idx = rng.choice(idx, 1500, replace=False)
        Xm = X[idx].copy()
        Xm[:, j] -= 1
        c[j] = float(np.mean(full[idx] - logit(m.predict(Xm, num_iteration=n))))
    log(f"  marginal card log-odds: {int((nd >= 30).sum())} cards, sd {c[nd >= 30].std():.4f} [{time.time() - t0:.0f}s]")
    out_dir(set_code).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir(set_code) / "lgbm_prep.npz", draft_id=np.asarray(d.draft_id).astype(str), has=has,
                        split=d.split, full=full.astype(np.float32), oof=oof.astype(np.float32), c=c, n_decks=nd,
                        names=np.asarray(d.names).astype(str), trees=np.array(trees), deck=deck)
    assert_no_torch()
    log(f"{set_code} prep done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")


# ================================================================ FM arms (torch)

def fit_arm(G, X, c, y, tr, va, l2w, l2v, vmask, seed=0, w0=None, init_w=None, aux=None, fit=None):
    """game_fm_v2.fit_fm (identical when w0, init_w and aux are None) plus: card-power L2 toward w0, w initialised at
    init_w, and an auxiliary per-step draft term. aux = dict(kind "draft" | "distill", weight, drafts [Dt] indices,
    Xd [D, C] tensor, Ed [D, E] tensor, nE, and wins/losses/rank or target/scale)."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import log_loss
    from decks.deckstrength import N_RANKS
    from decks.synergy import DeckFM
    fit = fit or G.FIT
    torch.manual_seed(seed)
    m = DeckFM(X.shape[1], c.shape[1], fit["k"], vmask=vmask)
    if init_w is not None:
        with torch.no_grad():
            m.w.copy_(torch.as_tensor(init_w, dtype=torch.float32))
    params = list(m.parameters())
    if aux is not None and aux["kind"] == "draft":
        r_rank = torch.nn.Parameter(torch.zeros(N_RANKS))
        beta = torch.nn.Parameter(torch.tensor(1.0))
        params += [r_rank, beta]
    opt = torch.optim.Adam(params, lr=fit["lr"])
    Xt, Ct, Yt = torch.from_numpy(X), torch.from_numpy(c), torch.from_numpy(y.astype(np.float32))
    W0 = None if w0 is None else torch.as_tensor(w0, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    rng_d = np.random.default_rng(1000 + seed)

    def val_loss():
        m.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(m(Xt[b], Ct[b])) for b in np.array_split(va, max(1, len(va) // 8192))])
        return log_loss(y[va], p.numpy())

    def aux_loss(db):
        s = m.deck_score(aux["Xd"][db]) + aux["Ed"][db] @ m.cov.weight[0, -aux["nE"]:]
        if aux["kind"] == "draft":
            z = r_rank[aux["rank"][db]] + beta * s
            Wb, Lb = aux["wins"][db], aux["losses"][db]
            return -(Wb * F.logsigmoid(z) + Lb * F.logsigmoid(-z)).sum() / (Wb + Lb).sum()
        return ((s - aux["target"][db]) ** 2).mean() / aux["scale"] ** 2

    best, state, bad, drops, events = np.inf, None, 0, fit["lr_drops"], []
    for ep in range(fit["epochs"]):
        m.train()
        batches = np.array_split(rng.permutation(tr), max(1, len(tr) // fit["batch"]))
        dbs = None if aux is None else np.array_split(rng_d.permutation(aux["drafts"]), len(batches))
        for i, b in enumerate(batches):
            loss = F.binary_cross_entropy_with_logits(m(Xt[b], Ct[b]), Yt[b])
            if W0 is None:
                loss = loss + m.penalty(l2w, l2v, 0)
            else:
                loss = loss + l2w * (m.w - W0).pow(2).sum() + l2v * m.v.pow(2).sum()
            if aux is not None:
                loss = loss + aux["weight"] * aux_loss(torch.from_numpy(dbs[i]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_loss()
        if v < best - 1e-6:
            best, bad, ev = v, 0, "best"
        else:
            bad, ev = bad + 1, "-"
            if bad >= fit["patience"]:
                ev = "stop" if drops <= 0 else "drop"
                if ev == "drop":
                    drops, bad = drops - 1, 0
        if ev == "best":
            state = {n: t.detach().clone() for n, t in m.state_dict().items()}
            if aux is not None and aux["kind"] == "draft":
                state_aux = (float(beta.detach()), r_rank.detach().numpy().copy())
        elif ev == "drop":
            m.load_state_dict(state)
            for gr in opt.param_groups:
                gr["lr"] /= 3
        events.append(ev)
        if ev == "stop":
            break
    m.load_state_dict(state)
    m.eval()
    extra = {}
    if aux is not None and aux["kind"] == "draft":
        extra = {"draft beta": state_aux[0]}
    return m, best, events, extra


class ValSC:
    """Val SC-GLL-dl (experienced, dl context): (a, b) on one half of the val drafts, log loss on the other, both
    directions (t1_rescore.blend_star's cross-fit). Returns x1000 nats/game, higher is better."""

    def __init__(self, pg, ctx, fit_val, D, has, split, seed=7):
        self.pg, self.ctx, self.fit_val, self.D = pg, ctx, fit_val, D
        vd = np.flatnonzero(has & (split == 1))
        half = np.zeros(D, bool)
        half[vd[np.random.default_rng(seed).permutation(len(vd)) < len(vd) // 2]] = True
        self.hg = half[pg["gd"]]

    def __call__(self, s):
        from sc_gll import sc_losses
        tot, n = 0.0, 0.0
        for side in (True, False):
            ls, ng, _ = sc_losses(s, self.pg, self.ctx, self.fit_val & (self.hg != side), self.fit_val & (self.hg == side),
                                  self.D)
            tot, n = tot + ls.sum(), n + ng.sum()
        return float(-1000 * tot / n)


def run_set(set_code, seeds=2, n_boot=N_BOOT, threads=4, log=print):
    import polars as pl
    import torch
    import game_fm_v2 as G
    from eval_common import NARROW, load_games
    from mtgdata.games import game_path
    from sc_gll import subsets
    from t_common import evaluate_ctx, fit_contexts, peak_rss_gb, split_mull
    torch.set_num_threads(threads)
    G._no_lightgbm()
    t0 = time.time()
    data = G.load_set(set_code, True)
    d, pg, has, D = data["d"], data["pg"], data["has"], len(data["d"].split)
    pz = np.load(out_dir(set_code) / "lgbm_prep.npz")
    assert (pz["draft_id"] == np.asarray(d.draft_id).astype(str)).all() and (pz["has"] == has).all()
    deck_agree = float((pz["deck"][has] == data["deck"][has]).all(1).mean())
    g = load_games(game_path(set_code, "PremierDraft"), d.draft_id, cols=NARROW)
    assert len(g) == len(pg["gd"]) and (g["draft_id"].to_numpy() == np.asarray(d.draft_id)[pg["gd"]]).all()
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    assert np.allclose(pg["own_mull"] - pg["opp_mull"], pg["mull"])
    del g
    split = d.split
    ctxs, cinfo, _ = fit_contexts(pg, D, ("dl", "pre"))
    subs_all = subsets(pg, has, data["main_bix"])
    subs = {k: subs_all[k] for k in ("experienced", "match 1")}
    vsc = ValSC(pg, ctxs["dl"], subs_all["experienced"][0], D, has, split)
    X, y, gsp = data["X"], data["y"], data["gsp"]
    tr, va = np.flatnonzero(gsp == 0), np.flatnonzero(gsp == 1)
    cc = G.context_cols(CH, data)
    nE = data["E"].shape[1]
    vm = data["nonland"]
    played = data["played"]
    log(f"  deck agreement with prep {deck_agree:.4f}; contexts " + "  ".join(
        f"{k} {v['test logloss (context only)']:.5f}" for k, v in cinfo.items()) + f" [{time.time() - t0:.0f}s]")

    fits = {}   # name -> {"setting", "scores" [S, D], "w" (seed 0), "val gll", "val scgll", ...}

    def add(name, setting, seed, m, vbest, ev, extra=None):
        s = G.deck_logit(m, data["deck"], data["Ed"])
        f = fits.setdefault(name, {"setting": setting, "scores": [], "val gll": [], "val scgll": [], "epochs": [],
                                   "extra": []})
        f["scores"].append(s)
        f["val gll"].append(float(vbest))
        f["val scgll"].append(vsc(s))
        f["epochs"].append(len(ev))
        f["extra"].append(extra or {})
        if seed == 0:
            f["w"] = m.w.detach().numpy().astype(np.float64).copy()
        log(f"  {name:28s} seed {seed}: val gll {vbest:.5f}  val SC-GLL-dl {f['val scgll'][-1]:.3f}  epochs {len(ev)}"
            f" [{time.time() - t0:.0f}s]")

    # champion: game_fm_v2 grid on val game log loss, seed 0
    grid = []
    for kw in G.GRID:
        t1 = time.time()
        m, vb, ev, _ = fit_arm(G, X, cc, y, tr, va, *kw, vm, seed=0)
        grid.append((vb, kw, m, ev))
        log(f"  grid {kw}: val gll {vb:.5f} [{time.time() - t1:.0f}s]")
    vb, kw, m, ev = min(grid, key=lambda r: r[0])
    l2v = kw[1]
    add(CH, list(kw), 0, m, vb, ev)
    del grid, m

    # LightGBM-derived quantities
    tdr = np.flatnonzero(has & (split == 0))
    s_ch = fits[CH]["scores"][0]
    mu_ch, sd_ch = float(s_ch[tdr].mean()), float(s_ch[tdr].std())
    cj = pz["c"].astype(np.float64)
    w_ch = fits[CH]["w"]
    ok = played & (pz["n_decks"] >= 30)
    kappa = float(w_ch[ok].std() / (cj[ok].std() + 1e-12))
    kappa_ols = float(np.polyfit(cj[ok], w_ch[ok], 1)[0])
    w0 = kappa * cj
    oof = pz["oof"].astype(np.float64)
    target = mu_ch + sd_ch * (oof - oof[tdr].mean()) / (oof[tdr].std() + 1e-12)
    Xd_t = torch.from_numpy(data["deck"].astype(np.float32))
    Ed_t = torch.from_numpy(data["Ed"].astype(np.float32))
    base_aux = {"drafts": tdr, "Xd": Xd_t, "Ed": Ed_t, "nE": nE}
    draft_aux = {**base_aux, "kind": "draft", "wins": torch.from_numpy(d.wins.astype(np.float32)),
                 "losses": torch.from_numpy(d.losses.astype(np.float32)), "rank": torch.from_numpy(d.rank.astype(np.int64))}
    dist_aux = {**base_aux, "kind": "distill", "target": torch.from_numpy(target.astype(np.float32)), "scale": sd_ch}
    lg_info = {"kappa (sd matched)": kappa, "kappa (OLS of w on c)": kappa_ols,
               "r(c, champion w) played": float(np.corrcoef(cj[ok], w_ch[ok])[0, 1]),
               "cards with c": int(ok.sum()), "champion deck score train mean/sd": [mu_ch, sd_ch],
               "lgbm trees (full, fold0, fold1)": pz["trees"].tolist(), "deck agreement": deck_agree}
    log(f"  c: r with champion w {lg_info['r(c, champion w) played']:+.3f}, kappa {kappa:.3f} (OLS {kappa_ols:.3f})")

    settings = [(f"l2 {v:g}", dict(l2w=v)) for v in L2_GRID if v != kw[0]]
    settings += [(f"draft aux {v:g}", dict(aux=("draft", v))) for v in AUX_W]
    settings += [(f"lgbm prior {v:g}", dict(prior=v)) for v in PRIOR_L2]
    settings += [(f"distill {v:g}", dict(aux=("distill", v))) for v in DISTILL_W]

    def fit_setting(st, seed):
        l2w = st.get("l2w", kw[0])
        a = None
        if "aux" in st:
            a = {**(draft_aux if st["aux"][0] == "draft" else dist_aux), "weight": st["aux"][1]}
        if "prior" in st:
            return fit_arm(G, X, cc, y, tr, va, st["prior"], l2v, vm, seed=seed, w0=w0, init_w=w0)
        return fit_arm(G, X, cc, y, tr, va, l2w, l2v, vm, seed=seed, aux=a)

    for name, st in settings:          # seed 0: every setting
        m, vb, ev, ex = fit_setting(st, 0)
        add(name, st, 0, m, vb, ev, ex)
        del m

    # picks (seed 0 val)
    l2_names = [CH] + [n for n, _ in settings if n.startswith("l2 ")]
    picks = {"l2 [val-gll pick]": min(l2_names, key=lambda n: fits[n]["val gll"][0]),
             "l2 [val-scgll pick]": max(l2_names, key=lambda n: fits[n]["val scgll"][0])}
    for fam in ("draft aux", "lgbm prior", "distill"):
        picks[f"{fam} [pick]"] = max([n for n, _ in settings if n.startswith(fam)], key=lambda n: fits[n]["val scgll"][0])
    log("  picks: " + "  ".join(f"{k} -> {v}" for k, v in picks.items()))

    st_of = dict(settings)
    for seed in range(1, seeds):       # further seeds: the champion and the picked settings only
        m, vb, ev, _ = fit_arm(G, X, cc, y, tr, va, *kw, vm, seed=seed)
        add(CH, list(kw), seed, m, vb, ev)
        for name in dict.fromkeys(n for n in picks.values() if n != CH):
            m, vb, ev, ex = fit_setting(st_of[name], seed)
            add(name, st_of[name], seed, m, vb, ev, ex)
            del m

    scores = {n: np.stack(f["scores"]) for n, f in fits.items()}
    for p, n in picks.items():
        scores[p] = scores[n]
    # blend*
    res_dir = REPO / "results" / f"{set_code}.PremierDraft"
    bz = np.load(res_dir / "t1_rescore" / "blend.npz")
    assert (bz["draft_id"].astype(str) == np.asarray(d.draft_id).astype(str)).all()
    scores[BL_SAVED] = bz["blendstar_LightGBM_+_champion"].astype(np.float64)
    t1r = json.loads((res_dir / "t1_rescore" / "results.json").read_text())
    alphas = t1r["blend*"]["blend* (LightGBM + champion)"]["alphas"]
    Lsv = np.load(res_dir / "sc_gll" / "scores.npz")["LightGBM"].astype(np.float64)
    ref = has & (split == 0)
    Fm = scores[CH]
    scores[BL_HERE] = np.stack([(1 - a) * zstd_ref(Lsv[s], ref) + a * zstd_ref(Fm[s % len(Fm)], ref)
                                for s, a in enumerate(alphas)])
    g2 = np.load(res_dir / "game_fm_v2" / "scores.npz")["champion"].astype(np.float64)
    te = has & (split == 2)
    repro = {"r(champion here, game_fm_v2 champion) test, seed 0": float(np.corrcoef(Fm[0][te], g2[0][te])[0, 1]),
             "max |diff| seed 0": float(np.abs(Fm[0][has] - g2[0][has]).max())}
    log(f"  champion reproduction: {repro}")
    val_blend = {BL_SAVED: [vsc(s) for s in scores[BL_SAVED]], BL_HERE: [vsc(s) for s in scores[BL_HERE]]}

    pairs = [(CH, n) for n in scores if n != CH]
    pairs += [(BL_SAVED, n) for n in [CH, BL_HERE] + list(picks)]
    res, losses, ngames = evaluate_ctx(scores, pg, {"dl": ctxs["dl"], "pre": ctxs["pre"]}, subs, D, pairs, None,
                                       n_boot, G.day_blocks(d), set_code, lambda m: None)
    for k in KEYS:
        log(f"  {k}: " + "  ".join(f"{b.replace(' - champion', '')} {res['pairs'][f'{b} - {a}'][k]['delta']:+.3f}"
                                   for a, b in pairs if a == CH and f"{b} - {a}" in res["pairs"]))
        log(f"    vs blend*: " + "  ".join(f"{b} {res['pairs'][f'{b} - {a}'][k]['delta']:+.3f}"
                                         for a, b in pairs if a == BL_SAVED))
    # diagnostics
    Lc_te = pz["full"].astype(np.float64)[te]
    diag = {}
    for n, f in fits.items():
        w = f["w"]
        diag[n] = {"card w r with champion": float(np.corrcoef(w[played], w_ch[played])[0, 1]),
                   "card w sd ratio vs champion": float(w[played].std() / (w_ch[played].std() + 1e-12)),
                   "card w r with LightGBM c": float(np.corrcoef(w[ok], cj[ok])[0, 1]),
                   "test score r with LightGBM counts-only": float(np.corrcoef(scores[n][0][te], Lc_te)[0, 1]),
                   "test score r with blend* (saved, seed 0)": float(np.corrcoef(scores[n][0][te],
                                                                                 scores[BL_SAVED][0][te])[0, 1]),
                   "val gll": f["val gll"], "val scgll": f["val scgll"], "epochs": f["epochs"],
                   "setting": f["setting"], "extra": f["extra"]}
    out = {"set": set_code, "seeds": seeds, "champion setting": list(kw), "picks": picks, "lgbm": lg_info,
           "reproduction": repro, "val scgll blend*": val_blend, "blend* alphas (T1)": alphas,
           "contexts": cinfo, "diagnostics": diag, **res, "primary key": PRIMARY,
           "total seconds": time.time() - t0, "peak rss GB": peak_rss_gb()}
    od = out_dir(set_code)
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, indent=1, default=float))
    k_ = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "").replace("*", "star")  # noqa: E731
    np.savez_compressed(od / "scores.npz", draft_id=np.asarray(d.draft_id).astype(str), has=has, split=split,
                        **{k_(n): v.astype(np.float32) for n, v in scores.items()})
    log(f"{set_code} done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")


# ================================================================ meta

def meta(*sets):
    from t_common import meta_rows, one_sided_p
    from eval_protocol import DecisionRule, decide, meta_analysis
    files = {s: out_dir(s) / "results.json" for s in sets}
    have = {s: f for s, f in files.items() if f.exists()}
    R = {s: json.loads(f.read_text()) for s, f in have.items()}
    pairs = sorted({p for r in R.values() for p in r["pairs"]})
    out = {"sets": list(have), "pairs": {}}
    for p in pairs:
        for key in KEYS:
            rs = meta_rows(have, p, key)
            if len(rs) < 2:
                continue
            if any(r_.se < 1e-6 for r_ in rs):   # the pick is the champion itself in some set (delta 0, se 0)
                dl_ = np.array([r_.delta for r_ in rs])
                out["pairs"].setdefault(p, {})[key] = {"unweighted mean": float(dl_.mean()),
                                                      "se": float(dl_.std(ddof=1) / np.sqrt(len(dl_))),
                                                      "sets better": f"{int((dl_ > 0).sum())}/{len(rs)}",
                                                      "sets identical": int(sum(r_.se < 1e-6 for r_ in rs))}
                print(f"{p:55s} {key:16s} unweighted {dl_.mean():+.4f} ± {dl_.std(ddof=1) / np.sqrt(len(dl_)):.4f} "
                      f"{int((dl_ > 0).sum())}/{len(rs)}")
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out["pairs"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci), "p": m.re_p_two_sided,
                                                  "p1 (one-sided, >0)": one_sided_p(m),
                                                  "sets better": f"{m.n_positive}/{len(rs)}",
                                                  "sig worse": m.n_sig_worse, "adopt": ok}
            if key in ("experienced dl", "experienced pre", "match 1 dl"):
                print(f"{p:55s} {key:16s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] p={m.re_p_two_sided:.4f} "
                      f"{m.n_positive}/{len(rs)} adopt={ok}")
    out["picks"] = {s: r["picks"] for s, r in R.items()}
    out["champion setting"] = {s: r["champion setting"] for s, r in R.items()}
    out["lgbm"] = {s: r["lgbm"] for s, r in R.items()}
    out["reproduction"] = {s: r["reproduction"] for s, r in R.items()}
    names = list(next(iter(R.values()))["diagnostics"])
    out["diagnostics (mean over sets)"] = {
        n: {k: float(np.mean([r["diagnostics"][n][k] for r in R.values() if n in r["diagnostics"]]))
            for k in ("card w r with champion", "card w sd ratio vs champion", "card w r with LightGBM c",
                      "test score r with LightGBM counts-only", "test score r with blend* (saved, seed 0)")}
        for n in names}
    for n, v in out["diagnostics (mean over sets)"].items():
        print(f"{n:28s} " + "  ".join(f"{k} {x:+.3f}" for k, x in v.items()))
    out["total seconds"] = sum(r["total seconds"] for r in R.values())
    out["peak rss GB (max)"] = max(r["peak rss GB"] for r in R.values())
    dst = REPO / "results" / "meta" / "fm_cardvalue.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=float))
    print(f"picks: {out['picks']}")


def main(argv):
    seeds, n_boot, threads, sets = 2, N_BOOT, int(_T), []
    it = iter(argv)
    for a in it:
        if a == "--seeds":
            seeds = int(next(it))
        elif a == "--boot":
            n_boot = int(next(it))
        elif a == "--threads":
            threads = int(next(it))
        elif a.startswith("--threads="):
            threads = int(a.split("=", 1)[1])
        else:
            sets.append(a)
    for s in sets:
        run_set(s, seeds, n_boot, threads, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    if sys.argv[1:2] == ["prep"]:
        for s in sys.argv[2:]:
            prep(s, log=lambda m: print(m, flush=True))
    elif sys.argv[1:2] == ["meta"]:
        meta(*sys.argv[2:])
    else:
        main(sys.argv[1:])
