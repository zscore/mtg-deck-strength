"""Hand value: card ratings from the shuffle, a randomised experiment inside every game.

    python scripts/hand_value.py MSH            # one set (~1-3 min, < 1.5 GB; see hand_value_design.md)
    python scripts/hand_value.py --selftest     # synthetic world with known effects (< 60 s)

Design (results/methodology/hand_value_design.md):
  Given the registered deck, which spells sit in the recorded 7-card opening hand is decided by the shuffle.
  17Lands' opening_hand_* always sums to 7, also after a mulligan: it is the final 7 drawn, *before* bottoming.
  Conditional on the deck and on L = lands among those 7 (and the mulligan count, decided before the final 7 was
  dealt), the 7 - L spells are a uniform draw from the deck's S spells, so for spell c with n copies

      E[h_c | deck, L] = (7 - L) n / S          (the known propensity; lands analogously, with L of the deck's lands)

  Robinson's partialling-out with that *known* propensity, r = h - E[h | deck, L], e = y - p_hat(deck, context, L):

      HV_c = sum_g r_gc e_g / sum_g w_g r_gc^2       w = 1: percentage points (design-unbiased for any p_hat)
                                                     w = p_hat (1 - p_hat): one-step logit effect
  Under an additive model this is exactly the swap estimand "c in the opening hand instead of a random other spell
  of the same deck (which then sits in the library)": (b^H_c - mean b^H) - q (b^D_c - mean b^D), q = P(library card
  is drawn). Player skill and deck quality are constant within the deck, so they cannot enter. p_hat only buys
  precision; it is cross-fitted on draft halves. SEs: HC (shuffles are independent across games) and draft-cluster.
  Diagnostic: the balance index sum r / sum E[h] (relative over-representation in recorded hands): non-zero means
  the keep decision selected on spell identity (the one way the design can fail); robustness subset L in {3, 4}.

Evaluation (all splits from load_drafts, games inherit their draft's split):
  - split-half reliability of HV (train drafts in two random halves), and train-vs-test HV correlation
  - correlation of train-game ratings (GP WR, GIH WR, OH WR, GD WR, IWD, deck-model card power, HV) with test-game HV,
    plus a skill tilt: corr(rating residual on HV, mean skill bucket of the decks playing the card)
  - deck vs hand: slope of deck-model card power on HV (errors-in-variables corrected) and the implied lambda
  - prior: a game-level linear deck model (ridge toward 0) vs the same with card power w = kappa * HV + delta
    (ridge on delta), lambda on val games, scored on test games (log loss x1000, draft-cluster SE). Per-draft test
    loss sums are saved for eval_protocol meta-analysis across sets.
Writes results/{SET}.PremierDraft/hand_value/{card_values.csv, results.json, test_losses.npz}.
numpy/scipy/polars/pyarrow only (no torch, no lightgbm).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import REPO, build_table, load_games, per_game  # noqa: E402

MIN_TRAIN, MIN_TEST = 2000, 300      # games with the card in the deck, for correlation tables
LAMS = (3e-6, 3e-5, 3e-4, 3e-3, 3e-2)      # ridge on card power (per-game mean loss scale)
LAM_OUTCOME = 3e-5
BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes")


# ---------------------------------------------------------------------------------------------------- data

@dataclass
class Data:
    names: list
    is_land: np.ndarray      # [C] bool
    Xb: np.ndarray           # [B, C] int8 deck counts per build
    gb: np.ndarray           # [G] build of each game
    gd: np.ndarray           # [G] draft of each game
    n_drafts: int
    H: sparse.csr_matrix     # [G, C] opening-hand counts
    Dr: sparse.csr_matrix    # [G, C] drawn counts
    y: np.ndarray            # [G] float64
    split: np.ndarray        # [G] 0/1/2
    cats: list               # [(codes [G] int, size)] categorical context (pre-treatment)
    dense: np.ndarray        # [G, k] float32 dense context
    mull: np.ndarray         # [G] own mulligans (0, 1, 2+)
    skill: np.ndarray        # [G] raw skill bucket (NaN unknown), for the skill-tilt diagnostic only

    def __post_init__(self):
        self.Xs = sparse.csr_matrix(self.Xb.astype(np.float64))   # ~25 nonzeros per build: no dense float copies
        self.Xp = (self.Xs > 0).astype(np.float64)


def stream_sparse(path, rows: np.ndarray, names: list, prefix: str, batch: int = 65536) -> sparse.csr_matrix:
    """[len(rows), C] int8 csr of `prefix + name` counts for the given (sorted, unique) file rows, streamed in row
    batches so the dense card matrix never exists."""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    have = set(f.schema_arrow.names)
    cols = [(j, prefix + n) for j, n in enumerate(names) if prefix + n in have]
    want = np.zeros(f.metadata.num_rows, bool)
    want[rows] = True
    pos = np.cumsum(want) - 1                     # file row -> index into rows
    R, Cc, V = [], [], []
    start = 0
    last = int(rows.max()) if len(rows) else -1
    for b in f.iter_batches(batch_size=batch, columns=[c for _, c in cols]):
        if start > last:
            break
        n = b.num_rows
        k = want[start:start + n]
        if k.any():
            M = np.zeros((int(k.sum()), len(names)), np.int8)
            for i, (j, c) in enumerate(cols):
                a = b.column(i).to_numpy(zero_copy_only=False)
                M[:, j] = np.nan_to_num(a[k].astype(np.float32)).astype(np.int8)
            r, c = np.nonzero(M)
            R.append(pos[start:start + n][k][r])
            Cc.append(c)
            V.append(M[r, c])
        start += n
    R, Cc, V = (np.concatenate(x) if x else np.zeros(0, int) for x in (R, Cc, V))
    return sparse.csr_matrix((V.astype(np.float32), (R, Cc)), shape=(len(rows), len(names)))


def load_set(set_code: str, event: str = "PremierDraft") -> Data:
    import polars as pl
    from mtgdata.games import game_path
    from decks.deckstrength import N_RANKS, load_drafts
    d = load_drafts(set_code, event)
    path = game_path(set_code, event)
    g = load_games(path, d.draft_id)
    pg = per_game(d, g)
    Xb, gb, _, _ = build_table(path, d, g)
    rows = g["row"].to_numpy().astype(np.int64)
    H = stream_sparse(path, rows, d.names, "opening_hand_")
    Dr = stream_sparse(path, rows, d.names, "drawn_")
    try:
        from mtgdata.cardinfo import card_info
        is_land = card_info(d.names, set_code)["is_land"].to_numpy()
    except Exception as ex:  # noqa: BLE001
        print(f"  card_info failed ({ex}); lands = basics only", flush=True)
        is_land = np.array([n in BASICS for n in d.names])
    own_mull = np.clip(np.nan_to_num(g["num_mulligans"].cast(pl.Float64, strict=False).to_numpy()), 0, 2).astype(int)
    skill = pg["skill"]
    sk = np.where(np.isnan(skill), np.nanmean(skill), skill) - 0.5
    dense = [pg["on_play"], pg["mull"], sk, np.isnan(skill).astype(float)]
    # the round 4 game FM's deck score (cross-fitted on train drafts) as an extra precision covariate, if cached
    f4 = REPO / "results" / f"{set_code}.{event}" / "deck_synergy_v4" / "draft_inputs.npz"
    if f4.exists():
        z = np.load(f4)
        if len(z["game_logit"]) == len(d.draft_id):
            gl = np.where(z["has_deck"], z["game_logit"], 0.0)
            dense += [gl[pg["gd"]], (~z["has_deck"][pg["gd"]]).astype(float)]
            print("  using round 4 game_logit as a covariate", flush=True)
    cats = [(pg["state"], 22), (pg["rank"], N_RANKS), (pg["exp"].argmax(1), pg["exp"].shape[1]), (own_mull, 3)]
    return Data(names=d.names, is_land=is_land, Xb=Xb.astype(np.int8), gb=gb, gd=pg["gd"],
                n_drafts=len(d.draft_id), H=H, Dr=Dr, y=pg["y"], split=pg["split"].astype(np.int8), cats=cats,
                dense=np.column_stack(dense).astype(np.float32), mull=own_mull, skill=skill)


# ---------------------------------------------------------------------------------------------------- logistic

def hand_cats(data: Data):
    """Lands among the recorded 7 x (mulliganed or not): the stratum the design conditions on."""
    L = np.asarray(data.H[:, data.is_land].sum(1)).ravel().astype(int).clip(0, 7)
    return [(L * 2 + (data.mull > 0), 16)]


def fit_logit(data: Data, idx, cats, lam, use_w=True, extra=None, prior=None, maxiter=400):
    """logit = sum_k theta_k[code] + dense @ beta + (Xb @ w)[gb] (+ kappa * extra[gb]); ridge lam on w - prior
    (mean-loss scale) and 1e-6 on theta. Returns a predict(idx) -> logit function and the parameters."""
    y = data.y[idx]
    n = len(idx)
    gbi = data.gb[idx]
    B, C = data.Xb.shape
    codes = [(c[idx], s) for c, s in cats]
    De = data.dense[idx].astype(np.float64)
    sizes = [s for _, s in codes]
    nt, nd = sum(sizes), De.shape[1]
    nw = C if use_w else 0
    nk = 0 if extra is None else 1
    mu = np.zeros(C) if prior is None else prior

    def unpack(x):
        th, x = x[:nt], x[nt:]
        be, x = x[:nd], x[nd:]
        w, x = x[:nw], x[nw:]
        return th, be, w, (x[0] if nk else 0.0)

    def logit_of(x, codes_, De_, gb_):
        th, be, w, k = unpack(x)
        z = De_ @ be
        o = 0
        for (c, s) in codes_:
            z = z + th[o + c]
            o += s
        if nw:
            z = z + (data.Xs @ w)[gb_]
        if nk:
            z = z + k * extra[gb_]
        return z

    def f(x):
        z = logit_of(x, codes, De, gbi)
        p = expit(z)
        loss = np.mean(np.logaddexp(0, z) - y * z)
        th, be, w, k = unpack(x)
        r = (p - y) / n
        g = [np.concatenate([np.bincount(c, r, s) for c, s in codes]) + 2e-6 * th, De.T @ r]
        loss += 1e-6 * th @ th
        if nw:
            rb = np.bincount(gbi, r, B)
            g.append(data.Xs.T @ rb + lam * (w - mu[:nw]))
            loss += 0.5 * lam * np.sum((w - mu[:nw]) ** 2)
        if nk:
            g.append([extra[gbi] @ r])
        return loss, np.concatenate([np.ravel(a) for a in g])

    x0 = np.zeros(nt + nd + nw + nk)
    if nw:
        x0[nt + nd:nt + nd + nw] = mu
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    x = res.x

    def predict(jdx):
        return logit_of(x, [(c[jdx], s) for c, s in cats], data.dense[jdx].astype(np.float64), data.gb[jdx])
    return predict, unpack(x)


def logloss(y, z):
    return np.logaddexp(0, z) - y * z


# ---------------------------------------------------------------------------------------------------- estimator

def hand_values(data: Data, idx, e, pw, cls) -> dict:
    """Robinson / known-propensity estimator for the cards of class `cls` ([C] bool) over games `idx`.
    e = y - p_hat, pw = p_hat (1 - p_hat). Returns per-card arrays (NaN where the card never varies)."""
    B, C = data.Xb.shape
    Hs = data.H[idx]
    Hs = Hs @ sparse.diags(cls.astype(np.float32))
    Hs.eliminate_zeros()
    m = np.asarray(Hs.sum(1)).ravel()
    S = data.Xs @ cls.astype(np.float64)
    gbi = data.gb[idx]
    a = np.where(S[gbi] > 0, m / np.maximum(S[gbi], 1), 0.0)
    coo = Hs.tocoo()
    r, c, h = coo.row, coo.col, coo.data.astype(np.float64)
    nb = data.Xb[gbi[r], c].astype(np.float64)
    XbT = data.Xs.T.tocsr()
    Xb2T = data.Xs.power(2).T.tocsr()

    def bsum(v):
        return np.bincount(gbi, v, B)

    def den(wt):  # sum_g wt_g r_gc^2
        return (np.bincount(c, wt[r] * h * h, C) - 2 * np.bincount(c, wt[r] * h * a[r] * nb, C)
                + Xb2T @ bsum(wt * a * a))
    num = np.bincount(c, h * e[r], C) - XbT @ bsum(a * e)
    d1, dw, de2 = den(np.ones(len(idx))), den(pw), den(e * e)
    # draft-cluster sums s_dc = sum_{g in d} r_gc e_g
    gdi = data.gd[idx]
    T1 = sparse.csr_matrix((h * e[r], (gdi[r], c)), shape=(data.n_drafts, C))
    Ab = bsum(a * e)
    bd = np.zeros(B, np.int64)
    bd[gbi] = gdi
    Xs = data.Xs @ sparse.diags(cls.astype(np.float64))
    T2 = sparse.csr_matrix((Ab, (bd, np.arange(B))), shape=(data.n_drafts, B)) @ Xs
    clus = np.asarray((T1 - T2).power(2).sum(0)).ravel()
    expected = XbT @ bsum(a)
    observed = np.bincount(c, h, C)
    n_games = data.Xp.T @ bsum(np.ones(len(idx)))
    ok = cls & (d1 > 1e-9)
    nan = lambda v: np.where(ok, v, np.nan)  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"hv_pp": nan(num / d1), "se_pp": nan(np.sqrt(de2) / d1), "se_pp_cluster": nan(np.sqrt(clus) / d1),
                "hv_logit": nan(num / dw), "se_logit": nan(np.sqrt(de2) / dw),
                "balance": nan(observed / expected - 1), "n_games": np.where(cls, n_games, 0)}


def center_shrink(v, se, wt, ok):
    """Centre on the occurrence-weighted mean over `ok` cards and empirical-Bayes shrink (normal-normal)."""
    out = np.full_like(v, np.nan)
    if ok.sum() < 3:
        return out, np.nan
    mu = np.average(v[ok], weights=wt[ok])
    vc = v - mu
    tau2 = max(np.var(vc[ok]) - np.mean(se[ok] ** 2), 1e-12)
    out[ok] = vc[ok] * tau2 / (tau2 + se[ok] ** 2)
    return out, float(tau2)


def outcome_residuals(data: Data, idx, halves):
    """Cross-fitted p_hat(deck, context, L stratum): each half of `idx` predicted by the model fit on the other."""
    cats = data.cats + hand_cats(data)
    z = np.zeros(len(idx))
    for side in (True, False):
        fit_i, pred_i = idx[halves == side], halves != side
        pred, _ = fit_logit(data, fit_i, cats, LAM_OUTCOME)
        z[pred_i] = pred(idx[pred_i])
    p = expit(z)
    return data.y[idx] - p, p * (1 - p)


def win_rates(data: Data, idx):
    """17Lands-style GP / OH / GD / GIH / GNS win rates and IWD over games idx."""
    B, C = data.Xb.shape
    y = data.y[idx]
    gbi = data.gb[idx]
    inb = data.Xp.T.tocsr()
    n_gp, w_gp = inb @ np.bincount(gbi, None, B).astype(float), inb @ np.bincount(gbi, y, B)
    Hb = (data.H[idx] > 0).astype(np.float64)
    Sb = ((data.H[idx] + data.Dr[idx]) > 0).astype(np.float64)
    Db = Sb - Hb
    out = {}
    for k, M in (("oh", Hb), ("gd", Db), ("gih", Sb)):
        n = np.asarray(M.sum(0)).ravel()
        out[f"{k}_wr"] = np.where(n > 0, (M.T @ y) / np.maximum(n, 1), np.nan)
        out[f"n_{k}"] = n
    n_gns = n_gp - out["n_gih"]
    w_gns = w_gp - Sb.T @ y
    out["gp_wr"] = np.where(n_gp > 0, w_gp / np.maximum(n_gp, 1), np.nan)
    out["gns_wr"] = np.where(n_gns > 0, w_gns / np.maximum(n_gns, 1), np.nan)
    out["iwd"] = out["gih_wr"] - out["gns_wr"]
    sk = np.where(np.isnan(data.skill[idx]), 0, data.skill[idx])
    kn = (~np.isnan(data.skill[idx])).astype(float)
    out["deck_skill"] = (inb @ np.bincount(gbi, sk, B)) / np.maximum(inb @ np.bincount(gbi, kn, B), 1)
    out["n_gp"] = n_gp
    return out


def corr(a, b, m):
    m = m & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), float(spearmanr(a[m], b[m])[0]), int(m.sum())


# ---------------------------------------------------------------------------------------------------- analysis

def analyse(data: Data, log=print, prior_experiment=True, seed=1) -> tuple[dict, dict]:
    t0 = time.time()
    C = len(data.names)
    rng = np.random.default_rng(seed)
    dhalf = rng.random(data.n_drafts) < 0.5
    hand_n = np.asarray(data.H.sum(1)).ravel()
    hc = data.H.tocoo()
    over = hc.data > data.Xb[data.gb[hc.row], hc.col]
    valid = (hand_n == 7) & (np.bincount(hc.row[over], None, len(data.y)) == 0)
    L = np.asarray(data.H[:, data.is_land].sum(1)).ravel()
    res = {"games": int(len(data.y)), "valid games": int(valid.sum()),
           "dropped (hand != 7 or hand card not in deck)": float(1 - valid.mean()),
           "mulligan share": float((data.mull > 0).mean())}
    log(f"  {len(data.y)} games, {valid.mean():.4f} valid [{time.time() - t0:.0f}s]")
    tr, va, te = (np.flatnonzero(valid & (data.split == s)) for s in (0, 1, 2))
    spell, land = ~data.is_land, data.is_land.copy()

    def estimate(idx, tag):
        e, pw = outcome_residuals(data, idx, dhalf[data.gd[idx]])
        out = {}
        for cname, cls in (("spell", spell), ("land", land)):
            out[cname] = hand_values(data, idx, e, pw, cls)
        hv = {k: np.where(spell, out["spell"][k], out["land"][k]) for k in out["spell"]}
        subs = {"A": dhalf[data.gd[idx]], "B": ~dhalf[data.gd[idx]]}
        for s, msk in subs.items():
            o = hand_values(data, idx[msk], e[msk], pw[msk], spell)
            hv[f"hv_logit_{s}"] = o["hv_logit"]
        core = np.isin(L[idx], (3, 4)) & (data.mull[idx] == 0)
        hv["hv_logit_core"] = hand_values(data, idx[core], e[core], pw[core], spell)["hv_logit"]
        for side, val in (("play", 0.5), ("draw", -0.5)):
            msk = data.dense[idx, 0] == val
            hv[f"hv_logit_{side}"] = hand_values(data, idx[msk], e[msk], pw[msk], spell)["hv_logit"]
        log(f"  hand values on {tag} games [{time.time() - t0:.0f}s]")
        return hv

    H_tr, H_te = estimate(tr, "train"), estimate(te, "test")
    wr = win_rates(data, tr)
    # deck model (deck counts + context, no hand information) on train games, lambda on val
    fits = []
    for lam in LAMS:
        pred, par = fit_logit(data, tr, data.cats, lam)
        fits.append((logloss(data.y[va], pred(va)).mean(), lam, pred, par))
    v0, lam0, pred0, par0 = min(fits, key=lambda f: f[0])
    w_deck = par0[2]
    log(f"  deck model lambda {lam0:g} val {v0:.5f} [{time.time() - t0:.0f}s]")

    sel = spell & (H_tr["n_games"] >= MIN_TRAIN) & (H_te["n_games"] >= MIN_TEST)
    wt = H_tr["n_games"]
    hv_tr_c, tau2 = center_shrink(H_tr["hv_logit"], H_tr["se_logit"], wt, spell & (H_tr["n_games"] >= 200))
    rA, sA, nA = corr(H_tr["hv_logit_A"], H_tr["hv_logit_B"], sel)
    rT, _, _ = corr(H_te["hv_logit_A"], H_te["hv_logit_B"], sel)
    rel_tr, rel_te = 2 * rA / (1 + rA), 2 * rT / (1 + rT)
    res.update({"cards in tables": int(sel.sum()), "lambda deck": lam0,
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
               "IWD": wr["iwd"], "deck model w": w_deck, "HV train": H_tr["hv_logit"], "HV train shrunk": hv_tr_c}
    table = {}
    for k, v in ratings.items():
        p, s, n = corr(v, target, sel)
        m = sel & np.isfinite(v) & np.isfinite(target)
        tilt = np.nan
        if m.sum() > 5:  # skill tilt: does what the rating says beyond hand value follow the decks' skill?
            beta = np.polyfit(target[m], v[m], 1)
            tilt = corr(v - np.polyval(beta, target), wr["deck_skill"], m)[0]
        table[k] = {"pearson vs test HV": p, "spearman vs test HV": s,
                    "pearson disattenuated (test HV reliability)": p / np.sqrt(rel_te) if rel_te > 0 else np.nan,
                    "skill tilt": tilt, "cards": n}
        log(f"    {k:16s} r {p:+.3f}  rho {s:+.3f}  skill tilt {tilt:+.3f}  (n={n})")
    res["ratings vs test-game hand value"] = table
    # deck vs hand value: deck power per copy on HV (both logit); EIV-corrected by HV reliability
    m = sel & np.isfinite(H_tr["hv_logit"])
    slope = float(np.polyfit(H_tr["hv_logit"][m], w_deck[m], 1)[0]) if m.sum() > 5 else np.nan
    drawn = float(np.asarray(data.Dr[tr].sum(1)).mean())
    N = float(np.median(data.Xb.sum(1)))
    q = drawn / (N - 7)
    lam_impl = np.nan
    if np.isfinite(slope) and rel_tr > 0:
        k = slope / rel_tr  # (7 + l D) / (N (1 - l q)) = k  ->  l = (k N - 7) / (D + k N q)
        lam_impl = (k * N - 7) / (drawn + k * N * q)
    res["deck vs hand"] = {"slope w on HV (raw)": slope, "slope EIV-corrected": slope / rel_tr if rel_tr > 0 else np.nan,
                           "mean cards drawn after the opener": drawn, "deck size": N, "q": q,
                           "kappa if drawn = hand value (lambda 1)": (7 + drawn) / (N * (1 - q)),
                           "kappa if only the opener matters (lambda 0)": 7 / N,
                           "implied lambda (value when drawn later / in hand)": float(lam_impl)}
    log(f"  deck vs hand: slope {slope:.3f} (EIV {res['deck vs hand']['slope EIV-corrected']:.3f}), "
        f"implied lambda {lam_impl:.2f}")

    losses = {}
    if prior_experiment:
        prior = np.where(np.isfinite(hv_tr_c) & spell, hv_tr_c, 0.0)
        extra = data.Xs @ prior
        arms = {"ridge": dict(), "ridge + kappa*HV": dict(extra=extra), "kappa*HV only": dict(extra=extra, use_w=False)}
        rows = {}
        for name, kw in arms.items():
            best = None
            for lam in (LAMS if kw.get("use_w", True) else (0.0,)):
                pred, par = fit_logit(data, tr, data.cats, lam, **kw)
                v = logloss(data.y[va], pred(va)).mean()
                if best is None or v < best[0]:
                    best = (v, lam, pred, par)
            v, lam, pred, par = best
            lt = logloss(data.y[te], pred(te))
            losses[name] = np.bincount(data.gd[te], lt, data.n_drafts)
            rows[name] = {"val logloss": float(v), "lambda": lam, "kappa": float(par[3]), "test logloss": float(lt.mean()),
                          "_lt": lt}
        base = rows["ridge"]["_lt"]
        nd = np.bincount(data.gd[te], None, data.n_drafts)
        for name, r_ in rows.items():
            dlt = r_.pop("_lt") - base
            sd = np.bincount(data.gd[te], dlt, data.n_drafts) - nd * dlt.mean()
            r_["vs ridge x1000"] = float(dlt.mean() * 1000)
            r_["vs ridge se x1000 (draft cluster)"] = float(np.sqrt((sd ** 2).sum()) / len(dlt) * 1000)
            log(f"    prior arm {name:18s} lam {r_['lambda']:g} kappa {r_['kappa']:+.3f} test {r_['test logloss']:.5f}"
                f"  vs ridge {r_['vs ridge x1000']:+.3f} ± {1.96 * r_['vs ridge se x1000 (draft cluster)']:.3f}")
        res["prior experiment (test games)"] = rows
        losses["n_games"] = nd
    cards = {"card": np.array(data.names), "is_land": data.is_land,
             **{f"train_{k}": v for k, v in H_tr.items()}, "train_hv_logit_shrunk": hv_tr_c,
             "test_hv_logit": H_te["hv_logit"], "test_se_logit": H_te["se_logit"], "deck_w": w_deck,
             **{k: wr[k] for k in ("gp_wr", "oh_wr", "gd_wr", "gih_wr", "gns_wr", "iwd", "deck_skill", "n_gp")}}
    res["seconds"] = time.time() - t0
    return res, {"cards": cards, "losses": losses}


def main(set_code: str):
    import polars as pl
    out = REPO / "results" / f"{set_code}.PremierDraft" / "hand_value"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = load_set(set_code)
    print(f"{set_code}: loaded [{time.time() - t0:.0f}s]", flush=True)
    res, art = analyse(data, log=lambda s: print(s, flush=True))
    res = {"set": set_code, **res, "seconds total": time.time() - t0}
    try:
        import resource
        res["max rss GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS
    except Exception:  # noqa: BLE001
        pass
    pl.DataFrame(art["cards"]).sort("train_hv_logit_shrunk", descending=True, nulls_last=True) \
        .write_csv(out / "card_values.csv")
    np.savez(out / "test_losses.npz", **{k.replace(" ", "_").replace("*", ""): v for k, v in art["losses"].items()})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


# ---------------------------------------------------------------------------------------------------- self-test

def synthetic(seed=0, n_drafts=7000, lam_draw=0.8):
    """A world with known hand values: skilled drafters prefer 'trap' cards (no true effect), keep decisions depend on
    land count and, for 2-landers, on the spells (a real selection the balance index should flag), and drawn-later
    cards are worth lam_draw x their hand value, with more cards drawn in longer games."""
    rng = np.random.default_rng(seed)
    n_sp, basics = 60, 5
    C = n_sp + basics
    beta = rng.normal(0, 0.15, n_sp)
    trap = np.zeros(n_sp, bool)
    trap[:10] = True
    beta[trap] = 0.0
    skill = rng.normal(0, 0.4, n_drafts)
    Xb = np.zeros((n_drafts, C), np.int8)
    for i in range(n_drafts):
        wgt = np.exp(3.0 * skill[i] * trap + 1.5 * beta * (1 + skill[i]))
        Xb[i, :n_sp] = rng.multinomial(23, wgt / wgt.sum())
        Xb[i, n_sp:] = rng.multinomial(17, np.ones(basics) / basics)
    bval = np.r_[beta, np.zeros(basics)]
    land = np.r_[np.zeros(n_sp, bool), np.ones(basics, bool)]
    fL = {0: -0.8, 1: -0.5, 2: -0.15, 3: 0.0, 4: 0.0, 5: -0.2, 6: -0.4, 7: -0.8}
    gd, Hr, Hc, Hv, Dr_, Dc, Dv, y, mull, onp = [], [], [], [], [], [], [], [], [], []
    g = 0
    for i in range(n_drafts):
        deck = np.repeat(np.arange(C), Xb[i].astype(int))
        for _ in range(rng.integers(3, 10)):
            k = 0
            while True:
                lib = rng.permutation(deck)
                hand = lib[:7]
                nl = land[hand].sum()
                bad = nl < 2 or nl > 5 or (nl == 2 and bval[hand].sum() < -0.1 and rng.random() < 0.7)
                if not bad or k == 2:
                    break
                k += 1
            play = rng.random() < 0.5
            nd = int(rng.integers(6, 15)) + (0 if play else 1)
            drawn = lib[7:7 + nd]   # bottomed cards went under the library
            # bottomed cards (k of the hand, the worst ones) do not count in play
            kept = hand[np.argsort(bval[hand])[k:]] if k else hand
            z = (skill[i] + bval[kept].sum() + lam_draw * bval[drawn].sum() + fL[int(nl)]
                 + 0.1 * (play - 0.5) - 0.3 * k + 0.02 * (Xb[i].astype(float) @ bval))
            y.append(rng.random() < expit(z))
            u, cnt = np.unique(hand, return_counts=True)
            Hr += [g] * len(u); Hc += list(u); Hv += list(cnt)
            u, cnt = np.unique(drawn, return_counts=True)
            Dr_ += [g] * len(u); Dc += list(u); Dv += list(cnt)
            gd.append(i); mull.append(k); onp.append(play)
            g += 1
    gd = np.array(gd)
    split = np.where(gd < 0.7 * n_drafts, 0, np.where(gd < 0.85 * n_drafts, 1, 2)).astype(np.int8)
    mull = np.array(mull)
    H = sparse.csr_matrix((np.array(Hv, np.float32), (Hr, Hc)), shape=(g, C))
    Dr = sparse.csr_matrix((np.array(Dv, np.float32), (Dr_, Dc)), shape=(g, C))
    sk_bucket = np.clip(0.55 + 0.1 * skill[gd] + rng.normal(0, 0.03, g), 0, 1)
    data = Data(names=[f"s{j}" for j in range(n_sp)] + [f"L{j}" for j in range(basics)], is_land=land, Xb=Xb, gb=gd,
                gd=gd, n_drafts=n_drafts, H=H, Dr=Dr, y=np.array(y, float), split=split,
                cats=[(mull.clip(0, 2), 3)],
                dense=np.column_stack([np.array(onp) - 0.5, mull, sk_bucket - 0.5]).astype(np.float32),
                mull=mull.clip(0, 2), skill=sk_bucket)
    q = 10.5 / 33
    truth = np.r_[beta * (1 - lam_draw * q), np.full(basics, np.nan)]
    return data, truth, trap


def selftest():
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    # 1. the streaming sparse loader reproduces a dense read on a subset of rows
    rng = np.random.default_rng(0)
    names = [f"c{j}" for j in range(9)]
    M = rng.integers(0, 3, (500, 9)).astype(np.int8)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "g.parquet"
        pq.write_table(pa.table({f"opening_hand_{n}": M[:, j] for j, n in enumerate(names) if j != 4}), p,
                       row_group_size=77)
        rows = np.sort(rng.choice(500, 200, replace=False))
        S = stream_sparse(p, rows, names, "opening_hand_", batch=60).toarray()
        ref = M[rows].copy()
        ref[:, 4] = 0
        assert (S == ref).all(), "stream_sparse mismatch"
    # 2. known-effect synthetic world
    data, truth, trap = synthetic()
    print(f"synthetic: {len(data.y)} games [{time.time() - t0:.0f}s]")
    res, art = analyse(data, log=print, prior_experiment=True)
    cards = art["cards"]
    hv, se = cards["train_hv_logit"], cards["train_se_logit"]
    sp = ~data.is_land
    w = cards["n_gp"]
    tc = truth - np.average(truth[sp], weights=w[sp])
    hc = hv - np.average(hv[sp], weights=w[sp])
    r_true = np.corrcoef(hc[sp], tc[sp])[0, 1]
    zc = (hc[sp] - tc[sp]) / se[sp]
    cover = np.mean(np.abs(zc) < 1.96)
    gih = cards["gih_wr"]
    trap_hv = np.mean(hc[:60][trap] - tc[:60][trap])
    gih_res = gih[:60] - np.polyval(np.polyfit(tc[:60][~trap], gih[:60][~trap], 1), tc[:60])
    print(f"  corr(HV, true) {r_true:.3f}; 95% coverage {cover:.2f}; trap bias HV {trap_hv:+.3f} logit "
          f"(se {np.mean(se[:60][trap]) / np.sqrt(trap.sum()):.3f}); trap GIH WR residual {gih_res[trap].mean():+.4f} "
          f"vs others {gih_res[~trap].mean():+.4f}")
    bal = cards["train_balance"][:60]
    print(f"  balance index corr with true value (keep selection on 2-landers): "
          f"{np.corrcoef(bal, truth[:60])[0, 1]:+.3f}")
    print(f"  results: {json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, default=float)}")
    assert r_true > 0.85, r_true
    assert 0.80 <= cover <= 0.995, cover
    assert abs(trap_hv) < 4 * np.mean(se[:60][trap]) / np.sqrt(trap.sum()) + 0.02
    assert gih_res[trap].mean() - gih_res[~trap].mean() > 0.005, "synthetic GIH WR should be skill-inflated"
    assert res["ratings vs test-game hand value"]["HV train"]["pearson vs test HV"] > \
        res["ratings vs test-game hand value"]["GIH WR"]["pearson vs test HV"] - 0.05
    print(f"hand_value self-test OK [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(*sys.argv[1:])
