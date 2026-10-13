"""Champion game FM, refit with control / training variants, scored on SC-GLL:
    python scripts/game_fm_v2.py MSH [VARIANT ...] [--variants mull,rated-dl] [--seeds 1] [--threads 4] [--boot 500]
    python scripts/game_fm_v2.py --selftest
    python scripts/sc_gll.py meta --file game_fm_v2/results.json MSH BLB ...      (12-set meta / decide per pair)

The champion deck rater (SC-GLL scoreboard) is round 4's `fm4 nonland`: DeckFM rank-4 pair terms between nonland
cards + Scryfall deck features + the affinity column, Adam 3e-3, batch 4096, patience 3, lr_drops 2, grid
(l2_w, l2_v) in {(1e-5, 1e-3), (1e-5, 1e-2)} picked on val game log loss -- exactly deck_synergy_v4.train, here in
a re-implementation that also takes game weights and logs its early-stopping schedule (selftest: identical to
decks.synergy._train). Every arm drops the opponent-rank one-hot (None in every game file: a constant).
  champion   fm4 nonland + deck features + aff; context: play/draw, mulligan difference, raw skill bucket (+ known),
             rank, experience (500/1000 fix), record before the game
  - mull     champion without the mulligan difference (game_controls: +0.0022 Spearman, 12/12)
  rated-dl   the de-leaked skill bucket (this run's games removed; structural_a.deleaked_bucket) and its unknown flag in
             place of the raw bucket
  - skill    no skill control (leakage reference: expect a Spearman gain and an SC-GLL loss)
  refit      champion trained on train+val games with the champion's val-chosen setting, replaying its schedule epoch
             for epoch (same epochs, same lr drops, the same epoch's state restored); test games only
  recency    champion with training-game weights exp(-age / 2) (age 0 = newest training draft, 1 = oldest), normalised
             to mean 1 (pipeline_audit #8); same grid, chosen on val
Every arm's deck-only score (deck_score of the most-played build + its deck-feature/affinity terms, no context) is
scored with sc_gll's frozen context model (fit here on train games, identical to sc_gll.py's) and sc_gll's helpers:
(a, b) fit on VAL games, test games scored. Subsets: experienced (primary), all, match 1, main build; plus
"experienced dl-ctx" / "match 1 dl-ctx": the same with the de-leaked bucket in the context model (SC-GLL-dl).
refit's val games are in-sample, so it uses its train-only twin's calibration (meta_review_1): the champion's val
(a, b) in raw score units, per seed; "refit (own val fit)" is the in-sample-calibrated version, for reference.
Pairs: every arm minus the champion refit here (not the cached round 4 scores), plus `champion - cached game FM`
(sc_gll's cached draft_inputs game_logit) as a reproduction check. Paired day-block bootstraps; with --seeds S > 1
every arm gets S training seeds (grid picked with seed 0), paired by seed index.
Secondary: draft-level Spearman of the calibrated deck score (train-draft calibration with rank, deck_metrics, as
game_controls.py), raw and stratified (skill bin x rank x experience) Spearman of the deck-only score,
Spearman(score, skill bucket), card-value r with the champion, deck-score sd on test drafts.
Writes results/{SET}.PremierDraft/game_fm_v2/{results.json, scores.npz, losses.npz} (suffix _<variants> when not
all variants run). losses.npz holds per-draft summed test log loss [S, D] and test games [D] per model x subset.
Do not import lightgbm in this process (torch + LightGBM libomp segfault on macOS).
"""
import os
import sys

_T = "2" if "--selftest" in sys.argv else next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--threads=")),
                                                 "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import json  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import NARROW, day_blocks, main_builds, per_game  # noqa: E402
from eval_protocol import gll_metric, paired_bootstrap, spearman_wr, stratified_spearman  # noqa: E402
from sc_gll import SKILL_EDGES, fit_context, sc_losses, subsets  # noqa: E402
from structural_a import deleaked_bucket, per_draft  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS, calibrate, deck_metrics  # noqa: E402
from decks.synergy import DeckFM  # noqa: E402

FIT = dict(k=4, lr=3e-3, epochs=40, batch=4096, patience=3, lr_drops=2)   # deck_synergy_v4.train
GRID = [(1e-5, 1e-3), (1e-5, 1e-2)]                                         # deck_synergy_v4 "fm4 nonland"
TAU = 2.0
N_BOOT = 500
MULL, SKILL = [1], [2, 3]     # columns of covariates()[:, :4 + N_RANKS]: play/draw, mull, skill, skill known, ranks
VARIANTS = {"champion": "champion", "mull": "- mull", "rated-dl": "rated-dl", "skill": "- skill", "refit": "refit",
            "recency": "recency"}
CHAMP, REFIT_OWN, CACHED = "champion", "refit (own val fit)", "cached game FM"
DL_SUBSETS = ("experienced", "match 1")
PLAYED_MIN = 2000


# ---------------------------------------------------------------- training (deck_synergy_v4.train + weights + replay)

def fit_fm(X, c, y, tr, va, l2w, l2v, vmask, seed=0, w=None, replay=None, fit=FIT):
    """DeckFM(k, vmask) trained as decks.synergy._train (seed 0 reproduces it exactly). w: per-game weights (the
    mean BCE becomes mean(w * BCE)). Returns (model, best val log loss, events): events[ep] in {"best", "-", "drop",
    "stop"}, the early-stopping schedule. With replay=events (from another fit) nothing is validated: the same
    epochs are run and the same events applied (save the state / restore it and divide lr by 3 / stop)."""
    torch.manual_seed(seed)
    m = DeckFM(X.shape[1], c.shape[1], fit["k"], vmask=vmask)
    opt = torch.optim.Adam(m.parameters(), lr=fit["lr"])
    Xt, Ct, Yt = torch.from_numpy(X), torch.from_numpy(c), torch.from_numpy(y.astype(np.float32))
    Wt = None if w is None else torch.from_numpy(w.astype(np.float32))
    rng = np.random.default_rng(seed)

    def val_loss():
        m.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(m(Xt[b], Ct[b])) for b in np.array_split(va, max(1, len(va) // 8192))])
        return log_loss(y[va], p.numpy())

    best, state, bad, drops, events = np.inf, None, 0, fit["lr_drops"], []
    for ep in range(fit["epochs"] if replay is None else len(replay)):
        m.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // fit["batch"])):
            bce = F.binary_cross_entropy_with_logits(m(Xt[b], Ct[b]), Yt[b], weight=None if Wt is None else Wt[b])
            loss = bce + m.penalty(l2w, l2v, 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if replay is None:
            v = val_loss()
            if v < best - 1e-6:
                best, bad, ev = v, 0, "best"
            else:
                bad, ev = bad + 1, "-"
                if bad >= fit["patience"]:
                    ev = "stop" if drops <= 0 else "drop"
                    if ev == "drop":
                        drops, bad = drops - 1, 0
        else:
            ev = replay[ep]
        if ev == "best":
            state = {n: t.detach().clone() for n, t in m.state_dict().items()}
        elif ev == "drop":
            m.load_state_dict(state)
            for gr in opt.param_groups:
                gr["lr"] /= 3
        events.append(ev)
        if ev == "stop":
            break
    m.load_state_dict(state)
    m.eval()
    return m, (best if replay is None else np.nan), events


def deck_logit(m, deck, Ed):
    """Deck-only score per draft: card power + pairs + the deck-feature / affinity columns (the last Ed.shape[1]
    context columns), no game context."""
    with torch.no_grad():
        s = m.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
    return s + Ed.astype(np.float64) @ m.cov.weight.detach().numpy()[0, -Ed.shape[1]:].astype(np.float64)


def recency_weights(gd, gsp, tau=TAU):
    """exp(-age / tau) per game, age in [0, 1] over the training drafts (drafts are time-sorted, gd = position),
    normalised to mean 1 over training games; 0 elsewhere."""
    trg = gsp == 0
    lo, hi = gd[trg].min(), gd[trg].max()
    w = np.where(trg, np.exp(-((hi - gd) / max(hi - lo, 1)) / tau), 0.0)
    return w / w[trg].mean()


def deleaked_per_game(pg, D):
    """De-leaked skill bucket per game (NaN = unknown) and its train mean (structural_a.game_cov's recipe)."""
    W, n = np.bincount(pg["gd"], pg["y"], D), np.bincount(pg["gd"], minlength=D).astype(float)
    bt = deleaked_bucket(per_draft(pg, D, "skill"), per_draft(pg, D, "n_bucket"), W, n)
    return bt[pg["gd"]], bt


def context_cols(variant, data):
    cov, pg = data["cov"], data["pg"]
    keep = np.ones(cov.shape[1], bool)
    extra = []
    if variant == "- mull":
        keep[MULL] = False
    elif variant in ("- skill", "rated-dl"):
        keep[SKILL] = False
    if variant == "rated-dl":
        btg, bt = data["bt_game"], data["bt_draft"]
        tr_d = (data["d"].split == 0) & np.isfinite(bt)
        mu = float(np.mean(bt[tr_d]))
        unk = ~np.isfinite(btg)
        extra = [np.where(unk, 0.0, 4 * (np.nan_to_num(btg) - mu))[:, None], unk.astype(float)[:, None]]
        data.setdefault("info", {})["de-leaked bucket"] = {
            "known share (train games)": float(1 - unk[pg["split"] == 0].mean()), "train draft mean": mu}
    return np.hstack([cov[:, keep], data["exp"], data["rec"], *extra, data["E"]]).astype(np.float32)


def fit_variants(data, variants, seeds=1, fit=FIT, log=print):
    """{variant: {"scores": [S, D], "w": card power (seed 0), info...}}; the champion is always fit first."""
    X, y, gsp = data["X"], data["y"], data["gsp"]
    tr, va = np.flatnonzero(gsp == 0), np.flatnonzero(gsp == 1)
    trva = np.flatnonzero(gsp <= 1)
    out = {}
    for v in [CHAMP] + [x for x in variants if x != CHAMP]:
        t1 = time.time()
        c = context_cols(v, data)
        base = CHAMP if v == "refit" else v
        w = recency_weights(data["gd"], gsp) if v == "recency" else None
        if v == "refit":
            ch = out[CHAMP]
            kw, fits = ch["setting"], []
            for s in range(seeds):
                m, _, ev = fit_fm(X, c, y, trva, va, *kw, data["nonland"], seed=s, replay=ch["events"][s], fit=fit)
                fits.append((m, np.nan, ev))
            vbest = np.nan
        else:
            grid = [fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=0, w=w, fit=fit) + (kw,) for kw in GRID]
            m0, vbest, ev0, kw = min(grid, key=lambda f: f[1])
            fits = [(m0, vbest, ev0)] + [fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=s, w=w, fit=fit)
                                         for s in range(1, seeds)]
        sc = np.stack([deck_logit(m, data["deck"], data["Ed"]) for m, _, _ in fits])
        out[v] = {"scores": sc, "w": fits[0][0].w.detach().numpy().copy(), "setting": list(kw),
                  "val logloss": float(vbest), "val logloss per seed": [float(f[1]) for f in fits],
                  "events": [f[2] for f in fits], "epochs": [len(f[2]) for f in fits],
                  "lr drops": [f[2].count("drop") for f in fits], "context columns": int(c.shape[1] - data["E"].shape[1]),
                  "base": base, "fit seconds": time.time() - t1}
        log(f"  {v:10s} val {vbest:.5f} {tuple(kw)} epochs {out[v]['epochs']} [{time.time() - t1:.0f}s]")
        del c
    return out


# ---------------------------------------------------------------- SC-GLL

def fixed_losses(score, pg, ctx, ev, a_raw, b_raw, D):
    """Per-draft test log loss with logit = ctx + a_raw + b_raw * score (a calibration fixed elsewhere)."""
    gd = pg["gd"][ev]
    pr = np.clip(expit(ctx[ev] + a_raw + b_raw * np.nan_to_num(score)[gd]), 1e-9, 1 - 1e-9)
    y = pg["y"][ev]
    ll = -(y * np.log(pr) + (1 - y) * np.log1p(-pr))
    return np.bincount(gd, ll, D), np.bincount(gd, minlength=D).astype(float)


def raw_beta(score, pg, fit, beta):
    """(a, b) of ctx_calibrated_game_loss (standardised score) in raw score units."""
    s = np.nan_to_num(score)
    fd = np.unique(pg["gd"][fit])
    mu, sd = s[fd].mean(), s[fd].std() + 1e-12
    return beta[0] - beta[1] * mu / sd, beta[1] / sd


def evaluate(scores, pg, ctxs, subs, D, pairs, twin=None, n_boot=N_BOOT, blocks=None, name="", log=print):
    """scores {model: [S, D]}; ctxs {tag: ctx logit [G]}; subs {tag: {subset: (fit, ev)}}; twin {model: other}:
    that model is scored with the other's val calibration (raw units, seed-paired). Returns (sc_gll.evaluate-format
    results, per-draft losses {model: {subset: [S, D]}}, test games {subset: [D]})."""
    twin = twin or {}
    res, losses, ngames = {"models": {m: {} for m in scores}, "pairs": {}}, {m: {} for m in scores}, {}
    for tag, ctx in ctxs.items():
        for sub, (fit, ev) in subs[tag].items():
            key = sub + tag
            L, B, n_ev = {}, {}, None
            for m in [m for m in scores if m not in twin] + [m for m in scores if m in twin]:
                rows = []
                for k, s in enumerate(np.atleast_2d(scores[m])):
                    if m in twin:
                        a_raw, b_raw = B[twin[m]][min(k, len(B[twin[m]]) - 1)]
                        ls, n = fixed_losses(s, pg, ctx, ev, a_raw, b_raw, D)
                        beta = (np.nan, b_raw * np.nanstd(np.nan_to_num(s)[np.unique(pg["gd"][fit])]))
                    else:
                        ls, n, beta = sc_losses(s, pg, ctx, fit, ev, D)
                        B.setdefault(m, []).append(raw_beta(s, pg, fit, beta))
                    rows.append((ls, n, beta))
                L[m] = np.stack([r[0] for r in rows])
                n_ev = rows[0][1]
                res["models"][m][key] = {"b per sd": float(np.mean([r[2][1] for r in rows])),
                                         "calibration": f"twin: {twin[m]}" if m in twin else "own val fit"}
            units = np.flatnonzero(n_ev > 0)
            metric = gll_metric(n_ev[units])
            ngames[key] = n_ev
            for m in scores:
                vals = [metric(x[units], np.arange(len(units))) for x in L[m]]
                res["models"][m][key].update({"sc_gll": float(np.mean(vals)), "seed sd": float(np.std(vals)),
                                              "test games": int(n_ev.sum()), "test drafts": int(len(units))})
                losses[m][key] = L[m].astype(np.float32)
            for a, b in pairs:
                if a in L and b in L:
                    r = paired_bootstrap(metric, L[a][:, units], L[b][:, units], n_boot=n_boot, seed=0,
                                         blocks=None if blocks is None else blocks[units], name=name)
                    res["pairs"].setdefault(f"{b} - {a}", {})[key] = {k: v for k, v in asdict(r).items() if k != "boot"}
            log(f"  {key:20s} " + "  ".join(f"{m} {res['models'][m][key]['sc_gll']:.3f}" for m in scores))
            for a, b in pairs:
                if f"{b} - {a}" in res["pairs"]:
                    r = res["pairs"][f"{b} - {a}"][key]
                    log(f"      {b} - {a}: {r['delta']:+.4f} ± {r['se']:.4f} (x1000 nats/game)")
    return res, losses, ngames


# ---------------------------------------------------------------- secondary: Spearman, card values, sd

def secondary(fits, data, cached=None):
    d, has = data["d"], data["has"]
    trd, ted = (d.split == 0) & has, (d.split == 2) & has
    D = len(d.split)
    sk = d.skill
    sbin = np.where(np.isfinite(sk), np.digitize(np.nan_to_num(sk), SKILL_EDGES), -1)
    nb = per_draft(data["pg"], D, "n_bucket")
    strata = (sbin + 1) * 10_000 + d.rank * 100 + np.searchsorted([1, 5, 10, 50, 100, 500, 1000], np.nan_to_num(nb))
    played = data["played"]
    wc = fits[CHAMP]["w"]
    out = {}
    items = {v: f["scores"] for v, f in fits.items()} | ({CACHED: cached[None]} if cached is not None else {})
    for v, S in items.items():
        rows = []
        for s in S:
            p = calibrate(s[trd], d.rank[trd], d.wins[trd], d.losses[trd], s, d.rank)
            r = deck_metrics(p[ted], d.wins[ted], d.losses[ted])
            ok = ted & np.isfinite(sk)
            rows.append({"draft spearman (calibrated)": r["spearman"], "draft nll (calibrated)": r["nll"],
                         "raw spearman": spearman_wr(s[ted], d.wins[ted], d.losses[ted]),
                         "stratified spearman": stratified_spearman(s[ted], d.wins[ted], d.losses[ted], strata[ted]),
                         "spearman(score, skill bucket)": float(spearmanr(s[ok], sk[ok]).statistic),
                         "deck score sd (test drafts)": float(s[ted].std())})
        o = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        if len(rows) > 1:
            o["draft spearman seed sd"] = float(np.std([r["draft spearman (calibrated)"] for r in rows]))
        if v in fits:
            w = fits[v]["w"]
            o["card value spearman r with champion"] = float(spearmanr(w[played], wc[played]).statistic)
            o["card value pearson r with champion"] = float(np.corrcoef(w[played], wc[played])[0, 1])
            o["card value sd ratio vs champion"] = float(w[played].std() / (wc[played].std() + 1e-12))
        out[v] = o
    ch = out[CHAMP]
    for v, o in out.items():
        for k in ("draft spearman (calibrated)", "raw spearman", "stratified spearman"):
            o[f"{k} - champion"] = o[k] - ch[k]
        g_raw, g_str = o["raw spearman - champion"], o["stratified spearman - champion"]
        o["leakage share (1 - stratified gain / raw gain)"] = float(1 - g_str / g_raw) if g_raw > 1e-4 else None
    return out


# ---------------------------------------------------------------- the run

def run(data, variants, seeds=1, fit=FIT, n_boot=N_BOOT, name="", log=print):
    t0 = time.time()
    d, pg, has = data["d"], data["pg"], data["has"]
    D = len(d.split)
    data["bt_game"], data["bt_draft"] = deleaked_per_game(pg, D)
    fits = fit_variants(data, variants, seeds, fit, log)
    pg_dl = {**pg, "skill": data["bt_game"]}
    ctx, cinfo = fit_context(pg, pg["split"] == 0)
    ctx_dl, cinfo_dl = fit_context(pg_dl, pg["split"] == 0)
    subs_all = subsets(pg, has, data["main_bix"])
    subs = {"": subs_all, " dl-ctx": {k: v for k, v in subs_all.items() if k in DL_SUBSETS}}
    scores = {v: f["scores"] for v, f in fits.items()}
    twin = {}
    if "refit" in scores:
        scores[REFIT_OWN] = scores["refit"]
        twin["refit"] = CHAMP
    cached = data.get("cached")
    if cached is not None:
        scores[CACHED] = cached[None]
    pairs = [(CHAMP, v) for v in scores if v not in (CHAMP, CACHED)] + ([(CACHED, CHAMP)] if cached is not None else [])
    log(f"  SC-GLL (context test logloss {cinfo['test logloss (context only)']:.5f}, dl-ctx "
        f"{cinfo_dl['test logloss (context only)']:.5f}) [{time.time() - t0:.0f}s]")
    ev, losses, ngames = evaluate(scores, pg, {"": ctx, " dl-ctx": ctx_dl}, subs, D, pairs, twin, n_boot,
                                  day_blocks(d), name, log)
    sec = secondary(fits, data, cached)
    for v, o in sec.items():
        log(f"  {v:16s} draft spearman {o['draft spearman (calibrated)']:.4f} "
            f"({o['draft spearman (calibrated) - champion']:+.4f})  raw {o['raw spearman - champion']:+.4f}  "
            f"strat {o['stratified spearman - champion']:+.4f}  sd {o['deck score sd (test drafts)']:.3f}"
            + (f"  card r {o['card value spearman r with champion']:.3f}" if "card value spearman r with champion" in o
               else ""))
    res = {"variants": list(fits), "seeds": seeds, "fit": fit, "grid": GRID, "recency tau": TAU,
           "context": cinfo, "context dl": cinfo_dl, "primary subset": "experienced", **data.get("info", {}),
           "fits": {v: {k: x for k, x in f.items() if k not in ("scores", "w")} for v, f in fits.items()},
           **ev, "secondary": sec, "seconds": time.time() - t0}
    return res, scores, losses, ngames


def save(out, tag, d, has, res, scores, losses, ngames):
    out.mkdir(parents=True, exist_ok=True)
    k = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "")  # noqa: E731
    np.savez_compressed(out / f"scores{tag}.npz", draft_id=np.asarray(d.draft_id).astype(str), split=d.split, has=has,
                        **{k(m): v.astype(np.float32) for m, v in scores.items()})
    np.savez_compressed(out / f"losses{tag}.npz", **{f"{k(m)}|{k(s)}": x for m, ls in losses.items() for s, x in ls.items()},
                        **{f"n_games|{k(s)}": n.astype(np.float32) for s, n in ngames.items()})
    (out / f"results{tag}.json").write_text(json.dumps(res, indent=1, default=float))


# ---------------------------------------------------------------- real data

def load_set(set_code, use_aff=True):
    import polars as pl
    from mtgdata.cardinfo import card_info, deck_features
    from mtgdata.games import aligned, flag, game_path
    from decks.deckstrength import covariates, experience, load_drafts, most_played_deck, record_before, record_onehot
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    lf = pl.scan_parquet(path)
    have = lf.collect_schema().names()
    deck_cols = [c for c in have if c.startswith("deck_") and c[5:] in set(d.names)]
    narrow = [c for c in (*NARROW, "opp_rank") if c in have]
    g = (lf.with_row_index("row").select("row", *narrow, *[pl.col(c).cast(pl.Int8) for c in deck_cols])
         .filter(pl.col("draft_id").is_in(pl.Series(list(d.draft_id)).implode()))
         .collect(engine="streaming").sort("row"))   # = deck_synergy_v4's games (file order, drafts in d)
    if "opp_rank" not in g.columns:
        g = g.with_columns(pl.lit(None, pl.Utf8).alias("opp_rank"))
    pg = per_game(d, g)
    X = aligned(g, "deck_", d.names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    cov = covariates(g)[:, : 4 + N_RANKS].astype(np.float32)   # opponent rank (always None) dropped
    exp_, rec = experience(g), record_onehot(wb, lb)
    deck, has = most_played_deck(d, g)
    has2, _, main_bix = main_builds(d, g)
    assert (has == has2).all()
    del g
    deck = deck.astype(np.float32)
    gsp = pg["split"]
    info = card_info(d.names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    Fg, _ = deck_features(X, info)
    mu, sd = Fg[gsp == 0].mean(0), Fg[gsp == 0].std(0) + 1e-6
    del Fg
    aff = pick_aff(set_code, d.names) if use_aff else None
    if use_aff and aff is None:
        raise SystemExit(f"{set_code}: no picknet pool model (results/{set_code}.PremierDraft/picknet/models); "
                         "the champion needs the affinity column (or pass --no-aff)")
    prof = None
    if aff is not None:
        from deck_synergy_v3 import Profile
        prof = Profile(X, aff[0], nonland, gsp == 0)
    extra = lambda Xd: np.hstack([(deck_features(Xd, info)[0] - mu) / sd]  # noqa: E731
                                 + ([] if prof is None else [prof(Xd, slice(0, 1))])).astype(np.float32)
    E, Ed = extra(X), extra(deck)
    cached = None
    inp = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4" / "draft_inputs.npz"
    if inp.exists():
        z = np.load(inp)
        if len(z["has_deck"]) == len(has) and (z["has_deck"] == has).all():
            cached = z["game_logit"].astype(np.float64)
    print(f"=== {set_code}: {len(d.draft_id)} drafts, {len(y)} games (train {np.sum(gsp == 0)}, val {np.sum(gsp == 1)}, "
          f"test {np.sum(gsp == 2)}), {X.shape[1]} cards, {E.shape[1]} deck columns (affinity "
          f"{'in' if prof is not None else 'omitted'}) [{time.time() - t0:.0f}s]", flush=True)
    return {"X": X, "y": y, "gsp": gsp, "gd": pg["gd"], "cov": cov, "exp": exp_, "rec": rec, "E": E, "Ed": Ed,
            "deck": deck, "nonland": nonland, "played": X[gsp == 0].sum(0) > PLAYED_MIN, "d": d, "pg": pg,
            "has": has, "main_bix": main_bix, "cached": cached, "info": {"affinity column": prof is not None}}


def pick_aff(set_code, names):
    from deck_synergy_v2 import pick_affinity
    return pick_affinity(set_code, names)


def _no_lightgbm():
    if "lightgbm" in sys.modules:
        raise RuntimeError("lightgbm is loaded in this process; run game_fm_v2.py in its own process")


def parse(argv):
    sets, variants, seeds, n_boot, aff, threads = [], [], 1, N_BOOT, True, int(_T)
    it = iter(argv)
    for a in it:
        if a == "--variants" or a.startswith("--variants="):
            variants += (a.split("=", 1)[1] if "=" in a else next(it)).split(",")
        elif a == "--seeds":
            seeds = int(next(it))
        elif a == "--boot":
            n_boot = int(next(it))
        elif a == "--threads":
            threads = int(next(it))
        elif a.startswith("--threads="):
            threads = int(a.split("=", 1)[1])
        elif a == "--no-aff":
            aff = False
        elif a in VARIANTS:
            variants.append(a)
        else:
            sets.append(a)
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {list(VARIANTS)}")
    chosen = [VARIANTS[v] for v in (variants or list(VARIANTS))]
    tag = "" if not variants or set(variants) >= set(VARIANTS) - {"champion"} else \
        "_" + "_".join(v for v in VARIANTS if v in variants and v != "champion")
    return sets, chosen, tag, seeds, n_boot, aff, threads


def main(argv):
    _no_lightgbm()
    sets, variants, tag, seeds, n_boot, aff, threads = parse(argv)
    torch.set_num_threads(threads)
    for set_code in sets:
        t0 = time.time()
        data = load_set(set_code, aff)
        res, scores, losses, ngames = run(data, variants, seeds, n_boot=n_boot, name=set_code,
                                          log=lambda m: print(m, flush=True))
        res = {"set": set_code, "games": int(len(data["y"])), **res, "total seconds": time.time() - t0}
        save(ROOT / "results" / f"{set_code}.PremierDraft" / "game_fm_v2", tag, data["d"], data["has"], res, scores,
             losses, ngames)
        print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)
        del data


# ---------------------------------------------------------------- synthetic self-test

def synthetic(D=9000, C=30, n_land=6, seed=0):
    """Drafts whose skilled drafters pick stronger cards (so a model without the skill control leaks skill into card
    values), a leaky skill bucket (includes this run), nonland pair synergy, deck-caused mulligans (land count),
    record-matched 7-3 runs, one build per draft."""
    from decks.deckstrength import record_onehot
    rng = np.random.default_rng(seed)
    w_true = np.r_[rng.normal(0, 0.08, C - n_land), np.zeros(n_land)]
    skill = rng.normal(0, 0.3, D)
    nl = C - n_land
    p_inc = expit(-0.3 + 3.0 * skill[:, None] * w_true[None, :nl] / 0.08 * 0.3)
    deck = np.zeros((D, C), np.float32)
    deck[:, :nl] = rng.binomial(2, p_inc)
    lands = 15 + rng.integers(0, 4, D)
    for j in range(D):
        deck[j, nl:] = rng.multinomial(lands[j], np.ones(n_land) / n_land)
    syn = 0.08 * (deck[:, 0] * deck[:, 1] + deck[:, 2] * deck[:, 3])
    eta = deck @ w_true + syn - 0.04 * (lands - 16.5) ** 2
    eta = eta - eta.mean()
    n_hist = rng.choice([5, 10, 50, 100, 500], D, p=[0.1, 0.1, 0.3, 0.3, 0.2])
    rank = rng.integers(2, 4, D)
    rows, W, L = [], np.zeros(D), np.zeros(D)
    for i in range(D):
        wv = lv = 0
        while wv < 7 and lv < 3:
            op = rng.random() < 0.5
            mull = int(rng.random() < 0.08 + 0.04 * abs(lands[i] - 16.5)) - int(rng.random() < 0.1)
            win = rng.random() < expit(0.15 + eta[i] + skill[i] + 0.1 * (op - 0.5) - 0.12 * (wv - lv) - 0.3 * mull)
            rows.append((i, wv + lv + 1, wv, lv, op, mull, win))
            wv, lv = wv + win, lv + (not win)
        W[i], L[i] = wv, lv
    gd, match, wb, lb, op, mull, y = (np.array(c) for c in zip(*rows))
    long_run = expit(0.15 + skill + rng.normal(0, 0.1, D))
    bucket = np.round(((n_hist - (W + L)).clip(1) * long_run + W) / np.maximum(n_hist, W + L) / 0.02) * 0.02
    split = np.zeros(D, np.int8)
    split[int(0.7 * D):], split[int(0.85 * D):] = 1, 2
    G = len(y)
    exp_b = np.searchsorted([1, 5, 10, 50, 100, 500, 1000], n_hist) + 1
    pg = {"gd": gd, "y": y.astype(float), "split": split[gd], "match": match, "known": np.ones(G, bool),
          "state": np.minimum(wb, 6) * 3 + np.minimum(lb, 2), "on_play": op - 0.5, "mull": mull.astype(float),
          "exp": np.eye(8)[exp_b[gd]], "rank": rank[gd], "n_bucket": n_hist[gd].astype(float),
          "skill": bucket[gd], "wins_before": wb, "losses_before": lb,
          "build_index": np.full(G, "0", dtype=object)}
    d = SimpleNamespace(wins=W.astype(np.float32), losses=L.astype(np.float32), split=split, rank=rank,
                        skill=bucket.astype(np.float32), draft_id=np.array([f"d{i}" for i in range(D)]),
                        draft_time=np.array([f"2025-01-{1 + i * 28 // D:02d}" for i in range(D)]))
    cov = np.hstack([pg["on_play"][:, None], pg["mull"][:, None], (bucket[gd] - 0.5)[:, None], np.ones((G, 1)),
                     np.eye(N_RANKS)[rank[gd]]]).astype(np.float32)
    feats = lambda Xd: np.stack([Xd[:, nl:].sum(1), (Xd[:, nl:].sum(1) - 16.5) ** 2, Xd[:, :nl].sum(1)], 1)  # noqa: E731
    Fd = feats(deck)
    mu, sd = Fd[split == 0].mean(0), Fd[split == 0].std(0) + 1e-6
    Ed = ((Fd - mu) / sd).astype(np.float32)
    X = deck[gd]
    data = {"X": X, "y": y.astype(np.int8), "gsp": split[gd], "gd": gd, "cov": cov, "exp": pg["exp"].astype(np.float32),
            "rec": record_onehot(wb, lb), "E": Ed[gd], "Ed": Ed, "deck": deck,
            "nonland": np.r_[np.ones(nl), np.zeros(n_land)].astype(np.float32),
            "played": X[split[gd] == 0].sum(0) > 50, "d": d, "pg": pg, "has": np.ones(D, bool),
            "main_bix": np.full(D, "0", dtype=object), "info": {}}
    return data, eta, skill


def selftest():
    import tempfile
    import sc_gll
    from decks.synergy import _train
    _no_lightgbm()
    torch.set_num_threads(2)
    t0 = time.time()
    data, eta, skill = synthetic()
    D = len(eta)
    fit = {**FIT, "batch": 1024}
    # 1. the re-implemented loop is decks.synergy._train (deck_synergy_v4.train) at seed 0
    sub = np.flatnonzero(data["gd"] < 3000)
    Xs, ys, cs = data["X"][sub], data["y"][sub], context_cols(CHAMP, data)[sub]
    sps = np.where(data["gd"][sub] < 2100, 0, 1).astype(np.int8)
    tr, va = np.flatnonzero(sps == 0), np.flatnonzero(sps == 1)
    small = {**fit, "batch": 512, "epochs": 12}
    m1, v1, ev1 = fit_fm(Xs, cs, ys, tr, va, 1e-5, 1e-3, data["nonland"], fit=small)
    torch.manual_seed(0)
    m2 = DeckFM(Xs.shape[1], cs.shape[1], 4, vmask=data["nonland"])
    opt = torch.optim.Adam(m2.parameters(), lr=3e-3)
    m2, v2, _ = _train(m2, opt, Xs, cs, ys, sps, None, lambda: m2.penalty(1e-5, 1e-3, 0), 12, 512, 3, "cpu",
                       lr_drops=2)
    assert abs(v1 - v2) < 1e-9 and torch.allclose(m1.w, m2.w), (v1, v2)
    # replay with the same games reproduces the fit exactly
    m3, _, ev3 = fit_fm(Xs, cs, ys, tr, va, 1e-5, 1e-3, data["nonland"], replay=ev1, fit=small)
    assert ev3 == ev1 and torch.allclose(m1.w, m3.w)
    print(f"[1] fit_fm == synergy._train (val {v1:.6f}), replay exact; schedule {''.join(e[0] for e in ev1)} "
          f"[{time.time() - t0:.1f}s]")
    # 2. the full run, all variants, 2 seeds
    res, scores, losses, ngames = run(data, list(VARIANTS.values()), seeds=2, fit=fit, n_boot=60, name="SYN")
    te = data["d"].split == 2
    truth_r = {v: float(np.corrcoef(S[0][te], eta[te])[0, 1]) for v, S in scores.items()}
    print("[2] r(score, true deck strength) on test drafts: " + "  ".join(f"{v} {r:.3f}" for v, r in truth_r.items()))
    sec = res["secondary"]
    print(f"    - skill: raw Spearman {sec['- skill']['raw spearman - champion']:+.4f}, "
          f"spearman(score, skill) {sec['- skill']['spearman(score, skill bucket)']:.3f} vs champion "
          f"{sec[CHAMP]['spearman(score, skill bucket)']:.3f}; SC-GLL exp "
          f"{res['pairs']['- skill - champion']['experienced']['delta']:+.3f}")
    assert truth_r[CHAMP] > 0.6, truth_r
    assert sec["- skill"]["spearman(score, skill bucket)"] > sec[CHAMP]["spearman(score, skill bucket)"]
    f = res["fits"]
    assert f["refit"]["epochs"] == f[CHAMP]["epochs"] and f["refit"]["setting"] == f[CHAMP]["setting"]
    assert all(np.isfinite(scores[v]).all() and scores[v].shape == (2, D) for v in scores)
    for p in ("- mull - champion", "rated-dl - champion", "- skill - champion", "refit - champion",
              "recency - champion", f"{REFIT_OWN} - champion"):
        for s in ("experienced", "all", "match 1", "main build", "experienced dl-ctx"):
            r = res["pairs"][p][s]
            assert np.isfinite(r["delta"]) and r["se"] > 0 and r["n_seeds"] == 2, (p, s, r)
    assert res["models"]["refit"]["experienced"]["calibration"] == "twin: champion"
    # SC-GLL of the champion beats a noise score with the same machinery
    fit_g, ev_g = subsets(data["pg"], data["has"], data["main_bix"])["experienced"]
    ctx, _ = fit_context(data["pg"], data["pg"]["split"] == 0)
    ls, n, _ = sc_losses(np.random.default_rng(1).normal(size=D), data["pg"], ctx, fit_g, ev_g, D)
    noise = -1000 * ls.sum() / n.sum()
    assert res["models"][CHAMP]["experienced"]["sc_gll"] > noise + 1, (res["models"][CHAMP]["experienced"], noise)
    # 3. saved files: sc_gll.py meta reads them (temp ROOT, two copies as two "sets")
    with tempfile.TemporaryDirectory() as tmp:
        for s in ("S1", "S2"):
            save(Path(tmp) / "results" / f"{s}.PremierDraft" / "game_fm_v2", "", data["d"], data["has"],
                 {"set": s, **res}, scores, losses, ngames)
        z = np.load(Path(tmp) / "results" / "S1.PremierDraft" / "game_fm_v2" / "losses.npz")
        assert z["champion|experienced"].shape == (2, D) and "n_games|experienced_dl-ctx" in z.files
        old = sc_gll.ROOT
        sc_gll.ROOT = Path(tmp)
        try:
            sc_gll.meta("--file", "game_fm_v2/results.json", "S1", "S2")
        finally:
            sc_gll.ROOT = old
    assert parse(["MSH"])[2] == "" and parse(["MSH", "mull", "rated-dl"])[2] == "_mull_rated-dl"
    assert parse(["MSH", "--variants", "skill"])[1] == ["- skill"]
    print(f"self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(sys.argv[1:])
