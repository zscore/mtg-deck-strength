"""Does LightGBM's non-additive count signal exist at the GAME level?   (fm_cardvalue.md "untested next steps")

    python scripts/game_boost.py fm SET [SET ...]      torch process (no lightgbm): champion refit + count probe
    python scripts/game_boost.py boost SET [SET ...]   torch-free process (lightgbm): arms A, B, C and SC-GLL scoring
    python scripts/game_boost.py meta SET ...          RE meta / decide -> results/meta/game_boost.json

fm_cardvalue showed blend*'s gain (+0.13 dev, +0.23 vault, SC-GLL experienced-dl) is LightGBM's NON-ADDITIVE use of
deck counts: its best linear-in-counts projection gains +0.004. Is that structure visible to a game-level
likelihood, or only to a model trained on draft records?
  fm stage (torch)   the champion of game_fm_v2 (fm4 nonland + deck features + aff, champion controls; grid on val game
                     log loss, seed 0; seed 1 at the chosen setting) refit; saved: per-game full logit (deck score +
                     context) [S, G], per-draft deck-only score [S, D].
                     PROBE: the champion plus non-additive count terms, trained jointly from scratch (same recipe,
                     same l2_w / l2_v): for every nonland card a 2nd-copy indicator 1[x >= 2] and a 3rd+-copy term
                     (x - 2)_+ (so copy 1, 2, 3+ get separate values w, w + d2, w + d3), plus a 4 x 4 one-hot of
                     land count (<=15, 16, 17, 18+) x nonland count (<=21, 22, 23, 24+); all extra terms L2 l2_u in
                     {1e-5, 1e-4, 1e-3} (+ 1e-2), picked on val game log loss ("probe [pick]"; seed 1 for the pick).
  boost stage (lightgbm, no torch)
    A  game-level LightGBM residual: binary objective on TRAIN games, features = the game's build deck counts only,
       init_score = the champion's full game logit (context stays in the offset), early stopping on VAL games;
       deck-only score = champion deck score + raw tree output on the main build (context-free by construction).
       One LightGBM fit per champion seed.
    B  draft-level residual: each train draft's W/L as two weighted rows (improve_deck), features = main-build
       counts, init_score = r[rank] + beta * s_champion (binomial logistic on train drafts); early stopping on val
       drafts; deck-only score = beta * s_champion + tree output. One fit per champion seed.
    C  blend* reproduced from saved scores: sc_gll LightGBM (3 seeds) with T1's alpha* per seed and this run's champion
       (seed s mod 2) vs the saved T1 blend* (sanity, expect ~0).
  LightGBM settings: improve_deck.LGB_PARAMS (lr 0.03, 15 leaves, min_data_in_leaf 300, l2 10), 4 threads.
Scoring: t_common contexts dl / pre, subsets experienced (primary) and match 1, (a, b) on val, test games;
pairs X - champion and X - blend* (saved T1). Paired day-block bootstraps, seeds as a variance component.
Writes results/{SET}.PremierDraft/game_boost/{fm.npz, fm.json, results.json, scores.npz}.
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

L2U = (1e-5, 1e-4, 1e-3, 1e-2)   # 1e-2: extension (MSH: val gll still improving at the 1e-3 edge)
SEEDS = 2
N_BOOT = 300
CH, A, B = "champion", "A game boost", "B draft boost"
BL_SAVED, BL_HERE = "blend* (saved T1)", "C blend* (saved alpha, this champion)"
KEYS = ("experienced dl", "experienced pre", "match 1 dl", "match 1 pre")
PRIMARY = "experienced dl"
LAND_EDGES, NL_EDGES = (15, 18), (21, 24)


def out_dir(s):
    return REPO / "results" / f"{s}.PremierDraft" / "game_boost"


# ================================================================ fm stage (torch)

def probe_feats(x, nl, land):
    """Non-additive count features of a [N, C] torch count tensor: 2nd-copy indicators and 3rd+-copy counts of nonland
    cards, and the land-count x nonland-count bucket one-hot."""
    import torch
    c2 = (x >= 2).float() * nl
    c3 = (x - 2).clamp(min=0) * nl
    L = (x @ land).round().clamp(LAND_EDGES[0], LAND_EDGES[1]) - LAND_EDGES[0]
    N = (x @ nl).round().clamp(NL_EDGES[0], NL_EDGES[1]) - NL_EDGES[0]
    k = (L * 4 + N).long()
    return c2, c3, torch.nn.functional.one_hot(k, 16).float()


def fit_probe(G, X, c, y, tr, va, l2w, l2v, l2u, vmask, land, seed=0, fit=None):
    """game_fm_v2.fit_fm (same schedule / early stopping) with the probe's extra terms d2, d3 [C] and t [16]."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import log_loss
    from decks.synergy import DeckFM
    fit = fit or G.FIT
    torch.manual_seed(seed)
    C = X.shape[1]
    m = DeckFM(C, c.shape[1], fit["k"], vmask=vmask)
    ex = torch.nn.ParameterDict({"d2": torch.nn.Parameter(torch.zeros(C)), "d3": torch.nn.Parameter(torch.zeros(C)),
                                 "t": torch.nn.Parameter(torch.zeros(16))})
    nl, ld = torch.as_tensor(vmask, dtype=torch.float32), torch.as_tensor(land, dtype=torch.float32)
    opt = torch.optim.Adam(list(m.parameters()) + list(ex.parameters()), lr=fit["lr"])
    Xt, Ct, Yt = torch.from_numpy(X), torch.from_numpy(c), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(seed)

    def extra(xb):
        c2, c3, oh = probe_feats(xb, nl, ld)
        return c2 @ ex["d2"] + c3 @ ex["d3"] + oh @ ex["t"]

    def fwd(b):
        return m(Xt[b], Ct[b]) + extra(Xt[b])

    def val_loss():
        m.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(fwd(b)) for b in np.array_split(va, max(1, len(va) // 8192))])
        return log_loss(y[va], p.numpy())

    best, state, bad, drops, events = np.inf, None, 0, fit["lr_drops"], []
    for ep in range(fit["epochs"]):
        m.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // fit["batch"])):
            loss = F.binary_cross_entropy_with_logits(fwd(b), Yt[b]) + m.penalty(l2w, l2v, 0) \
                + l2u * (ex["d2"].pow(2).sum() + ex["d3"].pow(2).sum() + ex["t"].pow(2).sum())
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
            state = ({n: t.detach().clone() for n, t in m.state_dict().items()},
                     {n: t.detach().clone() for n, t in ex.items()})
        elif ev == "drop":
            m.load_state_dict(state[0])
            with torch.no_grad():
                for n, t in state[1].items():
                    ex[n].copy_(t)
            for gr in opt.param_groups:
                gr["lr"] /= 3
        events.append(ev)
        if ev == "stop":
            break
    m.load_state_dict(state[0])
    with torch.no_grad():
        for n, t in state[1].items():
            ex[n].copy_(t)
    m.eval()
    return m, ex, best, events, extra


def fm_stage(set_code, seeds=SEEDS, threads=4, log=print):
    import torch
    import game_fm_v2 as G
    from mtgdata.cardinfo import card_info
    from t_common import peak_rss_gb
    torch.set_num_threads(threads)
    G._no_lightgbm()
    t0 = time.time()
    data = G.load_set(set_code, True)
    d, has = data["d"], data["has"]
    X, y, gsp = data["X"], data["y"], data["gsp"]
    tr, va = np.flatnonzero(gsp == 0), np.flatnonzero(gsp == 1)
    cc = G.context_cols(G.CHAMP, data)
    vm = data["nonland"]
    land = 1.0 - vm
    info_c = card_info(d.names, set_code)
    assert (land == info_c["is_land"].to_numpy().astype(np.float32)).all()
    Xt, Ct = torch.from_numpy(X), torch.from_numpy(cc)

    def full_logit(f):
        with torch.no_grad():
            return np.concatenate([f(b).numpy() for b in np.array_split(np.arange(len(y)), max(1, len(y) // 65536))])

    grid = []
    for kw in G.GRID:
        m, vb, ev = G.fit_fm(X, cc, y, tr, va, *kw, vm, seed=0)
        grid.append((vb, kw, m, ev))
        log(f"  champion grid {kw}: val gll {vb:.5f} epochs {len(ev)} [{time.time() - t0:.0f}s]")
    vb, kw, m, ev = min(grid, key=lambda r: r[0])
    del grid
    ch_models = [(m, vb, ev)] + [G.fit_fm(X, cc, y, tr, va, *kw, vm, seed=s) for s in range(1, seeds)]
    ch_scores = np.stack([G.deck_logit(mm, data["deck"], data["Ed"]) for mm, _, _ in ch_models])
    ch_full = np.stack([full_logit(lambda b, mm=mm: mm(Xt[b], Ct[b])) for mm, _, _ in ch_models]).astype(np.float32)
    ch_val = [float(v) for _, v, _ in ch_models]
    log(f"  champion {kw} seeds {seeds}: val gll {ch_val} [{time.time() - t0:.0f}s]")
    del ch_models
    g2 = np.load(REPO / "results" / f"{set_code}.PremierDraft" / "game_fm_v2" / "scores.npz")["champion"]
    repro = float(np.abs(ch_scores[0][has] - g2[0][has]).max())
    log(f"  reproduction vs game_fm_v2 champion seed 0: max |diff| {repro:.2e}")

    # probe
    Dk = torch.from_numpy(data["deck"])
    nl_t, ld_t = torch.as_tensor(vm), torch.as_tensor(land)
    probe, pinfo = {}, {}
    for l2u in L2U:
        mp, ex, vbp, evp, extra = fit_probe(G, X, cc, y, tr, va, *kw, l2u, vm, land, seed=0)
        with torch.no_grad():
            s = G.deck_logit(mp, data["deck"], data["Ed"]) + extra(Dk).numpy().astype(np.float64)
            pinfo[f"probe {l2u:g}"] = {"val gll": [float(vbp)], "epochs": [len(evp)],
                                       "d2 sd (nonland)": float(ex["d2"][nl_t > 0].std()),
                                       "d3 sd (nonland)": float(ex["d3"][nl_t > 0].std()),
                                       "t (land x nonland bucket, 4x4)": ex["t"].numpy().reshape(4, 4).round(4).tolist()}
        probe[f"probe {l2u:g}"] = [s]
        log(f"  probe l2_u {l2u:g}: val gll {vbp:.5f} (champion {ch_val[0]:.5f}) epochs {len(evp)} "
            f"d2 sd {pinfo[f'probe {l2u:g}']['d2 sd (nonland)']:.4f} [{time.time() - t0:.0f}s]")
        del mp, ex
    pick = min(L2U, key=lambda u: pinfo[f"probe {u:g}"]["val gll"][0])
    pn = f"probe {pick:g}"
    for s_ in range(1, seeds):
        mp, ex, vbp, evp, extra = fit_probe(G, X, cc, y, tr, va, *kw, pick, vm, land, seed=s_)
        with torch.no_grad():
            probe[pn].append(G.deck_logit(mp, data["deck"], data["Ed"]) + extra(Dk).numpy().astype(np.float64))
        pinfo[pn]["val gll"].append(float(vbp))
        pinfo[pn]["epochs"].append(len(evp))
        del mp, ex
    log(f"  probe pick (val gll) l2_u {pick:g} [{time.time() - t0:.0f}s]")
    od = out_dir(set_code)
    od.mkdir(parents=True, exist_ok=True)
    k_ = lambda s: s.replace(" ", "_")  # noqa: E731
    np.savez_compressed(od / "fm.npz", draft_id=np.asarray(d.draft_id).astype(str), has=has, split=d.split,
                        gd=data["gd"].astype(np.int32), y=y.astype(np.int8), xsum=X.sum(1).astype(np.int16),
                        champion=ch_scores.astype(np.float32), champion_full_logit=ch_full,
                        **{k_(n): np.stack(v).astype(np.float32) for n, v in probe.items()})
    (od / "fm.json").write_text(json.dumps({"set": set_code, "champion setting": list(kw), "champion val gll": ch_val,
                                            "reproduction max |diff|": repro, "probe": pinfo, "probe pick": pn,
                                            "seconds": time.time() - t0, "peak rss GB": peak_rss_gb()}, indent=1))
    log(f"{set_code} fm done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")


# ================================================================ boost stage (lightgbm, torch-free)

def boost_stage(set_code, n_boot=N_BOOT, log=print):
    from t_common import assert_no_torch, evaluate_ctx, fit_contexts, peak_rss_gb, split_mull, torchfree
    torchfree()
    import lightgbm as lgb
    from eval_common import NARROW, build_table, day_blocks, load_games, main_builds, per_game
    from improve_deck import LGB_PARAMS
    from mtgdata.games import game_path
    from sc_gll import sc_losses, subsets
    from decks.deckstrength import N_RANKS, binomial_logreg, load_drafts
    t0 = time.time()
    od = out_dir(set_code)
    fz = np.load(od / "fm.npz")
    fj = json.loads((od / "fm.json").read_text())
    d = load_drafts(set_code, "PremierDraft")
    D = len(d.draft_id)
    assert (fz["draft_id"] == np.asarray(d.draft_id).astype(str)).all()
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id, cols=NARROW)
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    assert (pg["gd"] == fz["gd"]).all() and (pg["y"] == fz["y"]).all()
    has, _, main_bix = main_builds(d, g)
    assert (has == fz["has"]).all()
    Xb, game_build, build_draft, main = build_table(path, d, g)
    del g
    assert (Xb[game_build].sum(1).astype(np.int64) == fz["xsum"]).all(), "game build counts differ from the fm stage"
    deck = np.zeros((D, Xb.shape[1]), np.float32)
    deck[has] = Xb[main[has]]
    split = d.split
    blocks = day_blocks(d)
    ch = fz["champion"].astype(np.float64)
    full = fz["champion_full_logit"].astype(np.float64)
    S = len(ch)
    P = {**LGB_PARAMS, "num_threads": 4}
    log(f"=== {set_code}: {D} drafts, {len(pg['y'])} games, {len(Xb)} builds [{time.time() - t0:.0f}s]")

    def ll(y, z):
        p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())

    info = {"A": [], "B": []}
    # ---- A: game-level residual
    gtr, gva = np.flatnonzero(pg["split"] == 0), np.flatnonzero(pg["split"] == 1)
    Xg_tr = Xb[game_build[gtr]].astype(np.float32)
    Xg_va = Xb[game_build[gva]].astype(np.float32)
    ytr, yva = pg["y"][gtr], pg["y"][gva]
    sA = []
    for s in range(S):
        t1 = time.time()
        dtr = lgb.Dataset(Xg_tr, ytr, init_score=full[s][gtr], free_raw_data=False)
        dva = lgb.Dataset(Xg_va, yva, init_score=full[s][gva], reference=dtr)
        bst = lgb.train({**P, "seed": s}, dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        n = bst.best_iteration
        f_va = bst.predict(Xg_va, num_iteration=n, raw_score=True) if n > 0 else np.zeros(len(gva))
        f = bst.predict(deck, num_iteration=n, raw_score=True) if n > 0 else np.zeros(D)
        f = np.where(has, f, 0.0)
        sA.append(ch[s] + f)
        te = has & (split == 2)
        info["A"].append({"trees": int(n), "val gll init": ll(yva, full[s][gva]), "val gll boosted": ll(yva, full[s][gva] + f_va),
                          "residual sd (test drafts)": float(f[te].std()), "champion sd (test drafts)": float(ch[s][te].std()),
                          "r(residual, champion) test": float(np.corrcoef(f[te], ch[s][te])[0, 1]) if f[te].std() > 0 else 0.0,
                          "seconds": time.time() - t1})
        log(f"  A seed {s}: {n} trees, val gll {info['A'][-1]['val gll init']:.5f} -> "
            f"{info['A'][-1]['val gll boosted']:.5f}, residual sd {info['A'][-1]['residual sd (test drafts)']:.4f} "
            f"[{time.time() - t1:.0f}s]")
        del dtr, dva, bst
    del Xg_tr, Xg_va
    # ---- B: draft-level residual from the champion
    dtrm, dvam = has & (split == 0), has & (split == 1)
    sB = []
    for s in range(S):
        t1 = time.time()
        Zc = np.hstack([ch[s][:, None], np.eye(N_RANKS)[d.rank]])
        lr_ = binomial_logreg(Zc[dtrm], d.wins[dtrm], d.losses[dtrm], C=1e4)
        init = lr_.decision_function(Zc)
        beta = float(lr_.coef_[0, 0])

        def ds(msk, ref=None):
            yy = np.r_[np.ones(msk.sum()), np.zeros(msk.sum())]
            sw = np.r_[d.wins[msk], d.losses[msk]].astype(float)
            k = sw > 0
            return lgb.Dataset(np.vstack([deck[msk], deck[msk]])[k], yy[k], weight=sw[k],
                               init_score=np.r_[init[msk], init[msk]][k], reference=ref)
        dtr = ds(dtrm)
        bst = lgb.train({**P, "seed": s}, dtr, 5000, valid_sets=[ds(dvam, dtr)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        n = bst.best_iteration
        f = bst.predict(deck, num_iteration=n, raw_score=True) if n > 0 else np.zeros(D)
        f = np.where(has, f, 0.0)
        sB.append(beta * ch[s] + f)
        te = has & (split == 2)
        info["B"].append({"trees": int(n), "beta": beta, "residual sd (test drafts)": float(f[te].std()),
                          "beta * champion sd (test drafts)": float(beta * ch[s][te].std()),
                          "r(residual, champion) test": float(np.corrcoef(f[te], ch[s][te])[0, 1]) if f[te].std() > 0 else 0.0,
                          "seconds": time.time() - t1})
        log(f"  B seed {s}: {n} trees, beta {beta:.3f}, residual sd {info['B'][-1]['residual sd (test drafts)']:.4f} "
            f"(beta*champion sd {info['B'][-1]['beta * champion sd (test drafts)']:.4f}) [{time.time() - t1:.0f}s]")
        del dtr, bst
    assert_no_torch()

    # ---- scores
    res_dir = REPO / "results" / f"{set_code}.PremierDraft"
    scores = {CH: ch, A: np.stack(sA), B: np.stack(sB)}
    for k in fz.files:
        if k.startswith("probe_"):
            scores[k.replace("_", " ")] = fz[k].astype(np.float64)
    pick = fj["probe pick"]
    scores["probe [pick]"] = scores[pick]
    bz = np.load(res_dir / "t1_rescore" / "blend.npz")
    assert (bz["draft_id"].astype(str) == np.asarray(d.draft_id).astype(str)).all()
    scores[BL_SAVED] = bz["blendstar_LightGBM_+_champion"].astype(np.float64)
    alphas = json.loads((res_dir / "t1_rescore" / "results.json").read_text())["blend*"]["blend* (LightGBM + champion)"]["alphas"]
    Lsv = np.load(res_dir / "sc_gll" / "scores.npz")["LightGBM"].astype(np.float64)
    ref = has & (split == 0)
    zs = lambda s: (s - s[ref].mean()) / (s[ref].std() + 1e-12)  # noqa: E731
    scores[BL_HERE] = np.stack([(1 - a) * zs(Lsv[s]) + a * zs(ch[s % S]) for s, a in enumerate(alphas)])
    te = has & (split == 2)
    diag = {"r(C, saved blend*) test seed 0": float(np.corrcoef(scores[BL_HERE][0][te], scores[BL_SAVED][0][te])[0, 1])}
    for n_ in (A, B, "probe [pick]"):
        diag[f"r({n_}, champion) test seed 0"] = float(np.corrcoef(scores[n_][0][te], ch[0][te])[0, 1])
        diag[f"r({n_}, saved blend*) test seed 0"] = float(np.corrcoef(scores[n_][0][te], scores[BL_SAVED][0][te])[0, 1])
        diag[f"r({n_}, LightGBM) test seed 0"] = float(np.corrcoef(scores[n_][0][te], Lsv[0][te])[0, 1])
    log("  " + "  ".join(f"{k} {v:.3f}" for k, v in diag.items()))

    ctxs, cinfo, _ = fit_contexts(pg, D, ("dl", "pre"))
    subs_all = subsets(pg, has, main_bix)
    subs = {k: subs_all[k] for k in ("experienced", "match 1")}
    # val SC-GLL-dl of every score (two-half cross-fit, as t1_rescore.blend_star)
    fit_val = subs_all["experienced"][0]
    vd = np.flatnonzero(has & (split == 1))
    half = np.zeros(D, bool)
    half[vd[np.random.default_rng(7).permutation(len(vd)) < len(vd) // 2]] = True
    hg = half[pg["gd"]]

    def vsc(s):
        tot, nn = 0.0, 0.0
        for side in (True, False):
            ls, ng, _ = sc_losses(s, pg, ctxs["dl"], fit_val & (hg != side), fit_val & (hg == side), D)
            tot, nn = tot + ls.sum(), nn + ng.sum()
        return float(-1000 * tot / nn)
    val_sc = {n_: [vsc(s) for s in v] for n_, v in scores.items()}
    pairs = [(CH, n_) for n_ in scores if n_ != CH]
    pairs += [(BL_SAVED, n_) for n_ in (CH, A, B, "probe [pick]", BL_HERE)]
    res, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, None, n_boot, blocks, set_code, lambda m: None)
    for k in KEYS:
        log(f"  {k}: " + "  ".join(f"{b_.replace(' - champion', '')} {res['pairs'][f'{b_} - {a_}'][k]['delta']:+.3f}"
                                   for a_, b_ in pairs if a_ == CH))
        log("    vs blend*: " + "  ".join(f"{b_} {res['pairs'][f'{b_} - {a_}'][k]['delta']:+.3f}"
                                         for a_, b_ in pairs if a_ == BL_SAVED))
    out = {"set": set_code, "fm": fj, "boost": info, "diagnostics": diag, "val scgll dl": val_sc,
           "blend* alphas (T1)": alphas, "contexts": cinfo, **res, "primary key": PRIMARY,
           "boost seconds": time.time() - t0, "peak rss GB (boost)": peak_rss_gb()}
    (od / "results.json").write_text(json.dumps(out, indent=1, default=float))
    k_ = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "").replace("*", "star")  # noqa: E731
    np.savez_compressed(od / "scores.npz", draft_id=np.asarray(d.draft_id).astype(str), has=has, split=split,
                        **{k_(n_): v.astype(np.float32) for n_, v in scores.items()})
    log(f"{set_code} boost done [{time.time() - t0:.0f}s, peak RSS {peak_rss_gb():.2f} GB]")


# ================================================================ meta

def meta(*sets):
    from t_common import meta_rows, one_sided_p, torchfree
    torchfree()
    from eval_protocol import DecisionRule, decide, meta_analysis
    have = {s: out_dir(s) / "results.json" for s in sets}
    have = {s: f for s, f in have.items() if f.exists()}
    R = {s: json.loads(f.read_text()) for s, f in have.items()}
    out = {"sets": list(R), "pairs": {}}
    for p in sorted({p for r in R.values() for p in r["pairs"]}):
        for key in KEYS:
            rs = meta_rows(have, p, key)
            if len(rs) < 2:
                continue
            if any(r_.se < 1e-6 for r_ in rs):
                dl_ = np.array([r_.delta for r_ in rs])
                out["pairs"].setdefault(p, {})[key] = {"unweighted mean": float(dl_.mean()),
                                                      "se": float(dl_.std(ddof=1) / np.sqrt(len(dl_))),
                                                      "sets better": f"{int((dl_ > 0).sum())}/{len(rs)}"}
                print(f"{p:58s} {key:16s} unweighted {dl_.mean():+.4f} ± {dl_.std(ddof=1) / np.sqrt(len(dl_)):.4f}")
                continue
            m = meta_analysis(rs)
            ok, why = decide(m, DecisionRule(min_positive=min(8, len(rs)), min_effect=0.02))
            out["pairs"].setdefault(p, {})[key] = {"re mean": m.re_mean, "ci": list(m.re_ci), "p": m.re_p_two_sided,
                                                  "p1 (one-sided, >0)": one_sided_p(m),
                                                  "sets better": f"{m.n_positive}/{len(rs)}",
                                                  "sig worse": m.n_sig_worse, "adopt": ok,
                                                  "per set": {r_.name: r_.delta for r_ in rs}}
            if key in ("experienced dl", "experienced pre", "match 1 dl"):
                print(f"{p:58s} {key:16s} {m.re_mean:+.4f} [{m.re_ci[0]:+.4f}, {m.re_ci[1]:+.4f}] "
                      f"p={m.re_p_two_sided:.4f} {m.n_positive}/{len(rs)} adopt={ok}")
    out["boost info"] = {s: r["boost"] for s, r in R.items()}
    out["probe info"] = {s: {"pick": r["fm"]["probe pick"],
                             **{k: {"val gll": v["val gll"], "d2 sd": v["d2 sd (nonland)"]} for k, v in r["fm"]["probe"].items()},
                             "champion val gll": r["fm"]["champion val gll"]} for s, r in R.items()}
    out["diagnostics (mean)"] = {k: float(np.mean([r["diagnostics"][k] for r in R.values()]))
                                 for k in next(iter(R.values()))["diagnostics"]}
    for k, v in out["diagnostics (mean)"].items():
        print(f"  {k}: {v:.3f}")
    for s, r in R.items():
        print(s, "A trees", [x["trees"] for x in r["boost"]["A"]], "B trees", [x["trees"] for x in r["boost"]["B"]],
              "probe pick", r["fm"]["probe pick"],
              "fm s", round(r["fm"]["seconds"]), "boost s", round(r["boost seconds"]))
    out["seconds"] = {s: {"fm": r["fm"]["seconds"], "boost": r["boost seconds"]} for s, r in R.items()}
    out["peak rss GB"] = {s: [r["fm"]["peak rss GB"], r["peak rss GB (boost)"]] for s, r in R.items()}
    dst = REPO / "results" / "meta" / "game_boost.json"
    dst.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    cmd, sets = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    lg = lambda m: print(m, flush=True)  # noqa: E731
    if cmd == "fm":
        for s_ in sets:
            fm_stage(s_, threads=int(_T), log=lg)
    elif cmd == "boost":
        for s_ in sets:
            boost_stage(s_, log=lg)
    elif cmd == "meta":
        meta(*sets)
    else:
        raise SystemExit(__doc__)
