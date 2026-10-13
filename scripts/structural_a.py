"""Candidate A on real games: python scripts/structural_a.py MSH [0.55 0.7 ...] [--variants record,rated,iid]
                                                                   [--no-aff] [--selftest]

next_model_design.md §A: a matchmaking-aware game-level likelihood (results/methodology/drafts/mm_likelihood.py)
around round 4's reference deck scorer (`linear + deck features + aff`: per-card power + Scryfall deck features +
the affinity column). For draft i, game g at record state before the game:
    e_ig  = d + eta(build_ig) + beta . cov_ig + rho[rank_ig] + t u_i,      u_i ~ N(0, 1) (Gauss-Hermite, 12 nodes)
    P(win) = sum_j pool_s[state_ig, j] expit(e_ig - s x_j)                  structural pool, s FIXED, exact record
                                                                            matching (lambda = 1); unknown history ->
                                                                            the all-games mixture
  eta    w . counts of the build actually played in that game + f . deck features (one score per build)
  cov    record: play/draw, mulligan difference, experience one-hot (no skill bucket, no free record dummies)
         rated:  record + the de-leaked skill bucket (this run's games removed; matchmaking_identification §2.4)
                 and its unknown flag, on the random-effect mean. Deck scores exclude it (predicted at its mean)
  iid    no random effect, no record terms (A1, the pure-compression baseline; independent of s)
s is profiled over the values given on the command line (default 0.55, the pooled play/draw estimate).
Fit on train drafts, early-stopped on the val drafts' marginal NLL per game with 2 lr drops; CPU, 4 torch threads.
The draft's deck score is eta of its most-played build. Evaluation on test drafts with a deck:
  spearman   raw deck-only score vs W/(W+L) (draft-file and game-file records); and calibrated with rank on VAL drafts
  SC-GLL     sc_gll.evaluate with the frozen context model (fit on train games), (a, b) on val games; paired
             day-block bootstraps vs the round 4 game FM (draft_inputs game_logit) and between A variants
  scale      t, d, beta, rho; sd of the score and its regression slope on the game FM logit; the stage-2 slope b of
             matchmaking_variance.fit_covariates at the same s on a sample of test drafts (b ~ 1 = correctly scaled;
             iid game models give 1.2-1.7); card-value sd and r with the iid fit
Affinity: draft_inputs.npz `aff` is per draft (most-played deck), so every build of a draft gets its draft's value
(rebuilds share most cards). Omitted when the npz is missing, misaligned or constant, or with --no-aff.
Writes results/{SET}.PremierDraft/structural_a/{scores.npz, results.json} (suffixed _s{...} when the s list is not the
default). Meta-analysis across sets: python scripts/sc_gll.py meta --file structural_a/results.json SET ...
Do not import lightgbm in this process (its OpenMP and torch's segfault together in the torch matmul here).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import N_STATES, build_table, day_blocks, load_games, padded, per_game  # noqa: E402
from eval_protocol import spearman_wr  # noqa: E402
from mm_likelihood import LinearDeck, MMGameModel, OpponentPool, opponents  # noqa: E402
from sc_gll import evaluate, fit_context, subsets  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS, calibrate  # noqa: E402

FIT = dict(lr=5e-3, batch=512, epochs=40, patience=3, lr_drops=2)
N_BUCKET_NEXT = {1: 5, 5: 10, 10: 50, 50: 100, 100: 500, 500: 1000, 1000: 2000}  # 1000+: 2000 assumed as the cap
SLOPE_DRAFTS = 6000


class FixedPool(OpponentPool):
    """Structural pool with s (and lambda) fixed: the [22, J] log pool is computed once, not every forward."""

    def __init__(self, s: float, n_nodes: int = 16, lam: float = 1.0, match: str = "record"):
        super().__init__("struct", n_nodes=n_nodes, s_init=s, learn_s=False, match=match)
        with torch.no_grad():
            pool, _ = opponents(torch.tensor(float(s)), self.x, self.w, lam, match)
        self.register_buffer("logpool", pool.clamp(min=1e-30).log())

    def log_lik(self, e, state, y):
        sign = (2 * y - 1).unsqueeze(-1)
        zz = e.unsqueeze(-1) - self.s() * self.x
        return torch.logsumexp(self.logpool[state].unsqueeze(-2) + F.logsigmoid(sign.unsqueeze(-1) * zz), -1)


def deleaked_bucket(b, floor, w_run, n_run, min_rest=5):
    """matchmaking_identification §2.4: b~ = (n_u b - W) / (n_u - n), n_u = geometric middle of the games bucket's
    bin (at least this run's games). Unknown (NaN) when the bucket is missing or fewer than min_rest other games."""
    nxt = np.array([N_BUCKET_NEXT.get(int(f), np.nan) if np.isfinite(f) else np.nan for f in floor])
    n_u = np.maximum(np.sqrt(floor * nxt), n_run)
    rest = n_u - n_run
    ok = np.isfinite(b) & np.isfinite(n_u) & (rest >= min_rest)
    return np.where(ok, np.clip((n_u * np.nan_to_num(b) - w_run) / np.where(ok, rest, 1), 0, 1), np.nan)


def per_draft(pg, D, key):
    out = np.full(D, np.nan)
    out[pg["gd"]] = pg[key]  # constant within a draft
    return out


def game_cov(pg, d_split, D, rated: bool):
    """[G, P] covariates; rated adds (de-leaked bucket - train mean) * 4 (about logit units) and an unknown flag."""
    cols = [pg["on_play"][:, None], pg["mull"][:, None], pg["exp"]]
    info = {}
    if rated:
        W, n = np.bincount(pg["gd"], pg["y"], D), np.bincount(pg["gd"], minlength=D).astype(float)
        bt = deleaked_bucket(per_draft(pg, D, "skill"), per_draft(pg, D, "n_bucket"), W, n)
        unk = ~np.isfinite(bt)
        mu = np.nanmean(bt[(d_split == 0) & ~unk])
        btc = np.where(unk, 0.0, 4 * (bt - mu))
        cols += [btc[pg["gd"]][:, None], unk[pg["gd"]].astype(float)[:, None]]
        info = {"de-leaked bucket known (train drafts with games)": float(1 - unk[(d_split == 0) & (n > 0)].mean()),
                "de-leaked bucket train mean": float(mu)}
    return np.hstack(cols).astype(np.float32), info


def tensors(pg, idx, game_build, cov):
    """Per-draft [D, G] tensors in MMGameModel's layout (idx from eval_common.padded); the caller adds Xb / Fb."""
    m = idx >= 0
    j = np.where(m, idx, 0)
    t = torch.from_numpy
    return {"gb": t(np.where(m, game_build[j], 0).astype(np.int64)), "mask": t(m),
            "state": t(np.where(m, pg["state"][j], N_STATES).astype(np.int64)),
            "y": t(np.where(m, pg["y"][j], 0).astype(np.float32)), "rank": t(np.where(m, pg["rank"][j], 0).astype(np.int64)),
            "cov": t(np.where(m[..., None], cov[j], 0).astype(np.float32))}


def fit_a(T, tr, va, variant, s, l2_w, lr, batch, epochs, patience, lr_drops, seed=0, log=lambda m: None):
    """Fit one A variant. Returns (deck scorer, likelihood model, best val NLL per game, epochs run)."""
    torch.manual_seed(seed)
    deck = LinearDeck(T["Xb"].shape[1], 0 if T["Fb"] is None else T["Fb"].shape[1])
    pool = OpponentPool("none") if variant == "iid" else FixedPool(s)
    model = MMGameModel(T["cov"].shape[-1], N_RANKS, pool, random_effect=variant != "iid", n_u=12)
    opt = torch.optim.Adam([p for p in list(deck.parameters()) + list(model.parameters()) if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)

    def loss(i):
        b = T["gb"][i]
        flat = b.reshape(-1)
        eta = deck(T["Xb"][flat], None if T["Fb"] is None else T["Fb"][flat]).view(b.shape)
        return model.nll(eta, T["state"][i], T["y"][i], T["mask"][i], T["cov"][i], T["rank"][i])

    def val():
        with torch.no_grad():
            tot = n = 0.0
            for b in np.array_split(va, max(1, len(va) // 4000)):
                a, c = loss(torch.from_numpy(b))
                tot, n = tot + float(a), n + float(c)
        return tot / n

    best, state, bad, ep = np.inf, None, 0, 0
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // batch)):
            tot, n = loss(torch.from_numpy(b))
            obj = tot / n + deck.penalty(l2_w, 0.0)
            opt.zero_grad()
            obj.backward()
            opt.step()
        v = val()
        log(f"      epoch {ep:2d} val {v:.5f}")
        if v < best - 1e-6:
            best, bad = v, 0
            state = ({k: x.detach().clone() for k, x in deck.state_dict().items()},
                     {k: x.detach().clone() for k, x in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if lr_drops <= 0:
                    break
                lr_drops, bad = lr_drops - 1, 0
                deck.load_state_dict(state[0])
                model.load_state_dict(state[1])
                for gr in opt.param_groups:
                    gr["lr"] /= 3
    deck.load_state_dict(state[0])
    model.load_state_dict(state[1])
    return deck, model, best, ep + 1


def build_scores(deck, T, main):
    with torch.no_grad():
        ok = main >= 0
        b = torch.from_numpy(main[ok])
        s = deck(T["Xb"][b], None if T["Fb"] is None else T["Fb"][b]).numpy().astype(np.float64)
    out = np.full(len(main), np.nan)
    out[ok] = s
    return out


def stage2_slope(score, pg, test_drafts, s, n_max=SLOPE_DRAFTS, seed=0):
    """matchmaking_variance.fit_covariates on test games with complete history: users' strength d + b z + u,
    population spread s. b is the slope of the true logit on the (centred) score: 1 = correctly scaled."""
    from matchmaking_variance import fit_covariates
    rng = np.random.default_rng(seed)
    pick = np.flatnonzero(test_drafts)
    pick = np.sort(rng.choice(pick, min(n_max, len(pick)), replace=False))
    pos = np.full(len(score), -1)
    pos[pick] = np.arange(len(pick))
    gm = (pos[pg["gd"]] >= 0) & pg["known"]
    zz = score[pick] - score[pick].mean()
    b, t = fit_covariates(zz[:, None], s, pg["y"][gm], pos[pg["gd"][gm]], pg["state"][gm], len(pick))
    return float(b[0]), float(t)


def diagnostics(deck, model, score, ref_logit, te, w_iid=None, played=None):
    out = {"t": float(model.t().detach()), "d": float(model.d.detach()),
           "rho": model.rho.weight.detach().numpy()[:, 0].tolist(),
           "beta": None if model.beta is None else model.beta.weight.detach().numpy()[0].tolist(),
           "score sd (test drafts)": float(np.std(score[te])), "card value sd": float(deck.w.detach().std())}
    if deck.f is not None:
        out["deck feature weights"] = deck.f.weight.detach().numpy()[0].tolist()
    if ref_logit is not None:
        c = np.cov(score[te], ref_logit[te])
        out["slope on game FM logit (test)"] = float(c[0, 1] / c[1, 1])
        out["r with game FM logit (test)"] = float(c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]))
        out["sd ratio vs game FM logit"] = float(np.sqrt(c[0, 0] / c[1, 1]))
    if w_iid is not None:
        w = deck.w.detach().numpy()
        out["card value r with iid"] = float(np.corrcoef(w[played], w_iid[played])[0, 1])
        out["card value sd ratio vs iid"] = float(w[played].std() / w_iid[played].std())
    return out


def run(d, pg, Xb, Fb, game_build, main, s_list, variants, l2_w, ref_logit=None, wg=None, lg=None, fit_kw=FIT,
        n_boot=500, slope_drafts=SLOPE_DRAFTS, log=print, name=""):
    """Every variant x s; returns (json-ready results, {key: per-draft score})."""
    t0 = time.time()
    D = len(d.split)
    has = main >= 0
    tr, va, te = (np.flatnonzero((d.split == k) & has) for k in range(3))
    tem = (d.split == 2) & has
    idx, dropped = padded(pg["gd"], pg["match"], D)
    covs = {r: game_cov(pg, d.split, D, r) for r in (False, True)}
    XbT = torch.from_numpy(Xb.astype(np.float32))
    FbT = None if Fb is None else torch.from_numpy(Fb.astype(np.float32))
    T = {r: {**tensors(pg, idx, game_build, covs[r][0]), "Xb": XbT, "Fb": FbT} for r in (False, True)}
    played = np.zeros(Xb.shape[1], bool)
    played[:] = Xb[main[(d.split == 0) & has]].sum(0) > 200
    res = {"drafts": int(D), "games": int(len(pg["y"])), "games beyond 9 slots dropped": dropped,
           "train/val/test drafts with a deck": [len(tr), len(va), len(te)], "s profile": list(s_list),
           "l2_w": l2_w, "fit": fit_kw, "rated covariates": covs[True][1], "fits": {}}
    scores, w_iid = {}, None
    jobs = ([("iid", None)] if "iid" in variants else []) + [(v, s) for s in s_list for v in variants if v != "iid"]
    for v, s in jobs:
        t1 = time.time()
        key = "A iid" if v == "iid" else f"A {v} s={s:g}"
        deck, model, best, ep = fit_a(T[v == "rated"], tr, va, v, s, l2_w, seed=0, **fit_kw)
        sc = build_scores(deck, T[False], main)
        scores[key] = sc
        r = {"variant": v, "s": s, "val nll per game": best, "epochs": ep,
             **diagnostics(deck, model, sc, ref_logit, tem, w_iid, played)}
        if v == "iid":
            w_iid = deck.w.detach().numpy().copy()
        for s_ in (s_list if v == "iid" else [s]):
            b, t = stage2_slope(sc, pg, tem, s_, slope_drafts)
            r[f"stage-2 slope at s={s_:g}"], r[f"stage-2 t at s={s_:g}"] = b, t
        r["seconds"] = time.time() - t1
        res["fits"][key] = r
        log(f"  {key:22s} val {best:.5f} ({ep} ep)  t {r['t']:.3f}  sd {r['score sd (test drafts)']:.3f}  "
            + (f"slope on FM {r.get('slope on game FM logit (test)', np.nan):.2f}  " if ref_logit is not None else "")
            + "  ".join(f"{k[8:]} {x:.2f}" for k, x in r.items() if k.startswith("stage-2 slope"))
            + f" [{time.time() - t1:.0f}s]")
    if ref_logit is not None:
        res["game FM stage-2 slope"] = {f"s={s:g}": stage2_slope(ref_logit, pg, tem, s, slope_drafts)[0]
                                        for s in s_list}
    # Spearman: raw deck-only, and calibrated with rank on val drafts
    targets = {"draft": (d.wins, d.losses)} | ({"game": (wg, lg)} if wg is not None else {})
    allsc = ({"game FM": ref_logit} if ref_logit is not None else {}) | scores
    vam = (d.split == 1) & has
    res["spearman"] = {}
    for k, sc in allsc.items():
        p = calibrate(sc[vam], d.rank[vam], d.wins[vam], d.losses[vam], np.nan_to_num(sc), d.rank)
        for tn, (W, L) in targets.items():
            res["spearman"][f"{k} | raw | {tn}"] = spearman_wr(sc[tem], W[tem], L[tem])
            res["spearman"][f"{k} | val-calibrated with rank | {tn}"] = spearman_wr(p[tem], W[tem], L[tem])
    # SC-GLL with the frozen context model
    ctx, cinfo = fit_context(pg, pg["split"] == 0)
    pairs = [("game FM", k) for k in scores if ref_logit is not None]
    pairs += [("A iid", k) for k in scores if k != "A iid" and "A iid" in scores]
    pairs += [(f"A record s={s:g}", f"A rated s={s:g}") for s in s_list
              if f"A record s={s:g}" in scores and f"A rated s={s:g}" in scores]
    res["context"] = cinfo
    res["sc_gll"] = evaluate({k: v[None] for k, v in allsc.items()}, pg, ctx, subsets(pg, has), D, pairs=pairs,
                             n_boot=n_boot, blocks=day_blocks(d), name=name, log=log)
    res["seconds"] = time.time() - t0
    return res, scores


def _no_lightgbm():
    """torch + LightGBM OpenMP in one process segfaults on macOS (libomp, torch index kernels): refuse to run."""
    if "lightgbm" in sys.modules:
        raise RuntimeError("lightgbm is loaded in this process; run structural_a.py in its own process")


def parse(argv):
    s_list, variants, aff = [], ["record", "rated", "iid"], True
    rest = []
    it = iter(argv)
    for a in it:
        if a == "--variants":
            variants = next(it).split(",")
        elif a.startswith("--variants="):
            variants = a.split("=", 1)[1].split(",")
        elif a == "--no-aff":
            aff = False
        else:
            try:
                s_list.append(float(a))
            except ValueError:
                rest.append(a)
    return rest, s_list or [0.55], variants, aff


def main(argv):
    from mtgdata.cardinfo import card_info, deck_features
    from mtgdata.games import game_path
    from decks.deckstrength import load_drafts
    _no_lightgbm()
    torch.set_num_threads(4)
    (set_code,), s_list, variants, use_aff = parse(argv)
    t0 = time.time()
    res_dir = ROOT / "results" / f"{set_code}.PremierDraft"
    out = res_dir / "structural_a"
    out.mkdir(parents=True, exist_ok=True)
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id)
    pg = per_game(d, g)
    Xb, game_build, build_draft, main_b = build_table(path, d, g)
    del g
    D = len(d.draft_id)
    has = main_b >= 0
    info = card_info(d.names, set_code)
    Fd, fnames = deck_features(Xb.astype(np.float32), info)
    trb = main_b[(d.split == 0) & has]
    Fb = (Fd - Fd[trb].mean(0)) / (Fd[trb].std(0) + 1e-6)
    inp_f = res_dir / "deck_synergy_v4" / "draft_inputs.npz"
    inp = dict(np.load(inp_f)) if inp_f.exists() else None
    aligned_ = inp is not None and len(inp["has_deck"]) == D and (inp["has_deck"] == has).all()
    aff_ok = use_aff and aligned_ and np.isfinite(inp["aff"][has]).all() and inp["aff"][has].std() > 0
    if aff_ok:
        a = inp["aff"][build_draft]
        Fb = np.hstack([Fb, ((a - a[trb].mean()) / (a[trb].std() + 1e-6))[:, None]])
    l2_w = 1e-4
    r4 = res_dir / "deck_synergy_v4" / "results.json"
    if r4.exists():
        l2_w = next((m["setting"][0] for m in json.loads(r4.read_text())["models"]
                     if m["model"] == "linear + deck features + aff"), l2_w)
    tgt = res_dir / "targets.npz"
    if tgt.exists():
        tz = np.load(tgt)
        assert (tz["draft_id"] == d.draft_id.astype(str)).all()
        wg, lg = tz["wins_games"].astype(float), tz["losses_games"].astype(float)
    else:
        wg, lg = np.bincount(pg["gd"], pg["y"], D), np.bincount(pg["gd"], 1 - pg["y"], D)
    print(f"=== {set_code}: {D} drafts, {len(pg['y'])} games, {len(Xb)} builds, {Fb.shape[1]} deck features "
          f"(affinity {'in' if aff_ok else 'omitted'}), l2_w {l2_w:g}, s {s_list}, variants {variants} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    res, scores = run(d, pg, Xb, Fb.astype(np.float32), game_build, main_b, s_list, variants, l2_w,
                      ref_logit=inp["game_logit"] if aligned_ else None, wg=wg, lg=lg, name=set_code,
                      log=lambda m: print(m, flush=True))
    res = {"set": set_code, "affinity column": bool(aff_ok), "deck features": fnames + (["aff"] if aff_ok else []),
           **res, "total seconds": time.time() - t0}
    tag = "" if s_list == [0.55] else "_s" + "_".join(f"{s:g}" for s in s_list)
    np.savez_compressed(out / f"scores{tag}.npz", draft_id=d.draft_id.astype(str), split=d.split, has=has,
                        **{k.replace(" ", "_"): v.astype(np.float32) for k, v in scores.items()})
    (out / f"results{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


def selftest():
    """mm_likelihood.simulate (record-matched 7-3 runs, s = population spread), one build per draft, pushed through the
    same tensors / fit / scores / Spearman / SC-GLL / stage-2 slope path as real data."""
    from types import SimpleNamespace
    from mm_likelihood import simulate
    _no_lightgbm()
    torch.set_num_threads(2)
    t0 = time.time()
    D, C = 14000, 40
    sd_deck, sd_u, d_users = 0.30, 0.25, 0.10
    s_pop = float(np.sqrt(sd_deck ** 2 + sd_u ** 2))
    data, X, eta, u, W, L = simulate(D, C, sd_deck, sd_u, d_users, s_pop, seed=3)
    m = data["mask"].numpy()
    gd, slot = np.nonzero(m)
    G = len(gd)
    rng = np.random.default_rng(0)
    bucket = data["cov"].numpy()[gd, slot, 1]
    st = data["state"].numpy()[gd, slot]
    pg = {"gd": gd, "y": data["y"].numpy()[gd, slot].astype(float), "match": slot + 1, "known": np.ones(G, bool),
          "state": st, "on_play": data["cov"].numpy()[gd, slot, 0].astype(float), "mull": np.zeros(G),
          "exp": np.eye(8)[np.full(G, 5)], "rank": np.full(G, 3), "n_bucket": np.full(G, 100.0),
          "skill": np.clip(0.55 + 0.05 * bucket, 0, 1), "build_index": np.full(G, "0", dtype=object)}
    split = np.zeros(D, np.int8)
    split[int(0.62 * D):], split[int(0.75 * D):] = 1, 2
    pg["split"] = split[gd]
    d = SimpleNamespace(split=split, wins=W.astype(float), losses=L.astype(float), rank=np.full(D, 3),
                        draft_time=np.array([f"2025-01-{1 + i * 28 // D:02d}" for i in range(D)]))
    Fb = rng.normal(0, 1, (D, 2)).astype(np.float32)
    fm_like = eta + rng.normal(0, 0.1, D)
    fit_kw = dict(lr=0.03, batch=2000, epochs=60, patience=3, lr_drops=2)
    res, scores = run(d, pg, X, Fb, gd, np.arange(D), [s_pop, 0.7], ["record", "rated", "iid"], 1e-5,
                      ref_logit=fm_like, fit_kw=fit_kw, n_boot=50, slope_drafts=3000, name="SYN")
    te = split == 2
    scale = {k: float(np.cov(v[te], eta[te])[0, 1] / np.var(eta[te], ddof=1)) for k, v in scores.items()}
    print(f"true s {s_pop:.3f}; scale cov(score, true eta)/var(eta) (1 = right): "
          + "  ".join(f"{k} {v:.2f}" for k, v in scale.items()))
    rec = res["fits"][f"A record s={s_pop:g}"]
    print(f"stage-2 slope, A record at true s: {rec[f'stage-2 slope at s={s_pop:g}']:.2f}; A iid: "
          f"{res['fits']['A iid'][f'stage-2 slope at s={s_pop:g}']:.2f}")
    assert 0.9 < scale[f"A record s={s_pop:g}"] < 1.15 and scale["A iid"] < scale[f"A record s={s_pop:g}"]
    assert scale["A record s=0.7"] > scale[f"A record s={s_pop:g}"]  # larger fixed s -> more decompression
    json.dumps(res, default=float)
    print(f"self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(sys.argv[1:])
