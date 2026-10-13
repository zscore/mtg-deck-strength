"""Champion game FM + opponent context (opponent colours, own x opponent matchups), scored on SC-GLL:
    python scripts/game_fm_opp.py MSH [opp] [oppmx] [--seeds 2] [--threads 4] [--boot 500] [--min-share 0.005]
    python scripts/game_fm_opp.py --selftest
    python scripts/sc_gll.py meta --file game_fm_opp/results.json MSH BLB ...                       (primary)
    python scripts/sc_gll.py meta --file game_fm_opp/results.json --subset "experienced opp-ctx" MSH BLB ...

Opponent strength is noise for deck strength: who you were paired against moves the result, not the deck. The
arms add opponent terms to the champion's game-level CONTEXT (never to the deck score):
  champion      game_fm_v2's champion (fm4 nonland + deck features + aff; play/draw, mull, skill bucket, rank,
                experience, record before the game), refit here with game_fm_v2.fit_fm (same code path, same grid)
  + opp         champion + an opponent colour-combination effect opp[b] (opp_colors as WUBRG strings; combos with
                >= min-share of train games kept, rare ones collapsed to other mono / other pair / other 3+,
                null -> unknown); ridge 1e-6 (identifiability with the bias only). Champion's (l2_w, l2_v).
  + opp x own   + opp plus a matchup matrix M[a, b] between the user's own main_colors level a and the opponent's
                b, double-centred with the train marginals (M - row mean - col mean + grand mean), so it carries no
                own-colour main effect (own colours are a DECK property and must stay in the deck score) and no
                opponent main effect; ridge l2_mx in MX_GRID picked on val game log loss.
The record-before-the-game control stays in every arm: opponents are matched on record, so opponent colours are
partly caused by the user's wins so far (matchmaking_identification §2.6); without the record control the opp
term would also absorb record compression.
The deck-only score is the same in every arm (game_fm_v2.deck_logit: deck_score + deck-feature/affinity terms).

SC-GLL (sc_gll's frozen context model, (a, b) on val, test games; game_fm_v2.evaluate): the frozen context is
shared by all compared models within a subset. Subsets: experienced (primary), all, match 1, main build, plus
  dl-ctx       the de-leaked bucket context (as game_fm_v2)
  opp-ctx      the sc_gll context + an opponent-colour offset fit on train games (no deck term, frozen)
  oppmx-ctx    the same + the double-centred matchup matrix (ridge picked on val, frozen)
for experienced and match 1. The opp-aware contexts stop the comparison from just rewarding the arms for knowing
the opponent (the deck-only score is what is evaluated; when the context already explains the opponent, any gain
left is a cleaner deck score).
Pairs: + opp - champion, + opp x own - champion, + opp x own - + opp, champion - cached game FM.
Reports (results.json):
  opponent strength   per opponent level: logit effect on the user's win with the champion's full game logit as
                      offset (Newton per level, draft-clustered sandwich SE; all games; + = harder opponent,
                      centred on the game-weighted mean), raw user win rate, share; the + opp arm's fitted effects;
                      a heterogeneity Wald Q; mean opponent strength by wins before the game (record gradient).
  matchups            own level x opponent level: cell effect with offset champion logit + opp effect, centred as
                      M, SE, n; the + opp x own arm's fitted M; Wald Q over well-populated cells (rock-paper-
                      scissors test: Q >> df means real matchup structure).
  secondary           game_fm_v2.secondary (Spearman, card-value r with the champion, deck-score sd), plus the
                      correlation of the own-colour mean deck score with the champion's (own-colour effect kept).
Writes results/{SET}.PremierDraft/game_fm_opp/{results.json, scores.npz, losses.npz} (suffix _<arms> when not all
arms run). Do not import lightgbm in this process (torch + LightGBM libomp segfault on macOS).
"""
import os
import sys

_T = "2" if "--selftest" in sys.argv else next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--threads=")),
                                                 "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import functools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game_fm_v2 as G  # noqa: E402  (sets the thread env and the paths, imports torch)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import chi2, pearsonr  # noqa: E402

from eval_common import load_games  # noqa: E402
from sc_gll import fit_context, subsets  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS  # noqa: E402
from decks.synergy import DeckFM  # noqa: E402

CHAMP, CACHED = G.CHAMP, G.CACHED
ARMS = {"opp": "+ opp", "oppmx": "+ opp x own"}
OPP, MX = ARMS["opp"], ARMS["oppmx"]
L2_OPP = 1e-6
MX_GRID = (1e-4, 1e-3)      # ridge on the raw matchup matrix (per mean-BCE; ~ prior sd 0.09 / 0.03 at 600k games)
MIN_SHARE = 0.005
MIN_CELL = 300              # matchup cells with fewer games are reported but left out of Q
OPP_SUBSETS = ("experienced", "match 1")
WUBRG = "WUBRG"


# ---------------------------------------------------------------- colour levels

def colour_levels(raw, fit_mask, min_share=MIN_SHARE):
    """raw [G] colour strings (WUBRG letters, None/'' = unknown) -> (level index [G], labels). Combos with at least
    min_share of the fit games are their own level; the rest collapse by number of colours."""
    s = np.array(["" if x is None else str(x) for x in raw], dtype=object)
    u, inv = np.unique(s, return_inverse=True)
    share = np.bincount(inv[fit_mask], minlength=len(u)) / max(int(fit_mask.sum()), 1)
    key = lambda c: (len(c), [WUBRG.index(ch) if ch in WUBRG else 9 for ch in c])  # noqa: E731
    kept = sorted([c for c, p in zip(u, share) if c and p >= min_share], key=key)
    labels = kept + ["other mono", "other pair", "other 3+", "unknown"]
    pos = {c: i for i, c in enumerate(kept)}
    k = len(kept)
    lev_u = np.array([pos[c] if c in pos else (k + 3 if not c else k + min(len(c), 3) - 1) for c in u])
    return lev_u[inv].astype(np.int64), labels


def centred(R, p_row, p_col):
    """Double centring with marginals: no row (own) or column (opponent) main effect under independence."""
    rm = R @ p_col
    cm = p_row @ R
    return R - rm[:, None] - cm[None, :] + p_row @ R @ p_col


# ---------------------------------------------------------------- the model

class OppFM(DeckFM):
    """DeckFM + opp[b] (+ centred M[a, b]); a, b are passed as two index columns of the context c (zeroed before
    the linear covariate layer, so the deck-feature columns stay last and game_fm_v2.deck_logit is unchanged)."""

    def __init__(self, n_cards, n_cov, k, vmask=None, *, j_opp, j_own, p_own, p_opp, matchup, l2_opp, l2_mx):
        super().__init__(n_cards, n_cov, k, vmask=vmask)
        self.j_opp, self.j_own, self.l2_opp, self.l2_mx = j_opp, j_own, l2_opp, l2_mx
        self.opp = nn.Parameter(torch.zeros(len(p_opp)))
        self.mx_raw = nn.Parameter(torch.zeros(len(p_own), len(p_opp))) if matchup else None
        self.register_buffer("p_own", torch.as_tensor(p_own, dtype=torch.float32))
        self.register_buffer("p_opp", torch.as_tensor(p_opp, dtype=torch.float32))

    def mx(self):
        return None if self.mx_raw is None else centred(self.mx_raw, self.p_own, self.p_opp)

    def opp_term(self, c):
        b = c[:, self.j_opp].long()
        t = self.opp[b]
        if self.mx_raw is not None:
            t = t + self.mx()[c[:, self.j_own].long(), b]
        return t

    def forward(self, x, c, task=None):
        c2 = c.clone()
        c2[:, [self.j_opp, self.j_own]] = 0
        return super().forward(x, c2, task) + self.opp_term(c)

    def penalty(self, l2_w, l2_v, l2_dw):
        p = super().penalty(l2_w, l2_v, l2_dw) + self.l2_opp * self.opp.pow(2).sum()
        if self.mx_raw is not None:
            p = p + self.l2_mx * self.mx_raw.pow(2).sum()
        return p


@contextmanager
def model_class(factory):
    """game_fm_v2.fit_fm builds `DeckFM(...)` from its module globals: swap the class for one fit."""
    old = G.DeckFM
    G.DeckFM = factory
    try:
        yield
    finally:
        G.DeckFM = old


def full_logit(m, X, c, bs=65536):
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(X[i:i + bs]), torch.from_numpy(c[i:i + bs])).numpy()
                               for i in range(0, len(X), bs)]).astype(np.float64)


def fit_arms(data, arms, lev, seeds=1, fit=G.FIT, log=print):
    """{arm: {"scores": [S, D], "w": card power (seed 0), ...}} plus per-arm models; the champion first, exactly
    as game_fm_v2.fit_variants fits it."""
    X, y, gsp = data["X"], data["y"], data["gsp"]
    tr, va = np.flatnonzero(gsp == 0), np.flatnonzero(gsp == 1)
    nE = data["E"].shape[1]
    c = G.context_cols(CHAMP, data)
    out, models = {}, {}

    def record(v, fits, kw, vbest, t1, extra=None):
        out[v] = {"scores": np.stack([G.deck_logit(m, data["deck"], data["Ed"]) for m, _, _ in fits]),
                  "w": fits[0][0].w.detach().numpy().copy(), "setting": list(kw), "val logloss": float(vbest),
                  "val logloss per seed": [float(f[1]) for f in fits], "events": [f[2] for f in fits],
                  "epochs": [len(f[2]) for f in fits], "fit seconds": time.time() - t1, **(extra or {})}
        models[v] = [f[0] for f in fits]
        log(f"  {v:12s} val {vbest:.5f} {tuple(kw)} epochs {out[v]['epochs']} [{time.time() - t1:.0f}s]")

    t1 = time.time()
    grid = [G.fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=0, fit=fit) + (kw,) for kw in G.GRID]
    m0, vbest, ev0, kw = min(grid, key=lambda f: f[1])
    fits = [(m0, vbest, ev0)] + [G.fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=s, fit=fit)
                                 for s in range(1, seeds)]
    record(CHAMP, fits, kw, vbest, t1)
    champ_logit = full_logit(m0, X, c)
    setting = kw
    if not arms:
        return out, models, champ_logit
    co = np.hstack([c[:, :-nE], lev["opp"][:, None].astype(np.float32), lev["own"][:, None].astype(np.float32),
                    c[:, -nE:]])
    del c
    j_opp, j_own = co.shape[1] - nE - 2, co.shape[1] - nE - 1
    base = dict(j_opp=j_opp, j_own=j_own, p_own=lev["p_own"], p_opp=lev["p_opp"], l2_opp=L2_OPP)
    for v in arms:
        t1 = time.time()
        grid_mx = MX_GRID if v == MX else (0.0,)
        cand = []
        for l2_mx in grid_mx:
            with model_class(functools.partial(OppFM, matchup=v == MX, l2_mx=l2_mx, **base)):
                cand.append(G.fit_fm(X, co, y, tr, va, *setting, data["nonland"], seed=0, fit=fit) + (l2_mx,))
        m0, vbest, ev0, l2_mx = min(cand, key=lambda f: f[1])
        with model_class(functools.partial(OppFM, matchup=v == MX, l2_mx=l2_mx, **base)):
            fits = [(m0, vbest, ev0)] + [G.fit_fm(X, co, y, tr, va, *setting, data["nonland"], seed=s, fit=fit)
                                         for s in range(1, seeds)]
        record(v, fits, setting, vbest, t1, {"l2_mx": l2_mx, "val logloss per l2_mx":
                                             {str(f[3]): float(f[1]) for f in cand}})
    assert G.DeckFM is DeckFM
    return out, models, champ_logit


# ---------------------------------------------------------------- frozen opp-aware context models

def fit_offset(off, y, b, a, fit, p_own, p_opp, matchup, l2_opp=L2_OPP, l2_mx=0.0, iters=100):
    """logit = off + bias + opp[b] (+ centred M[a, b]) by L-BFGS on `fit` games (mean BCE + ridge)."""
    o_t, y_t = torch.from_numpy(off[fit]), torch.from_numpy(y[fit].astype(np.float64))
    b_t, a_t = torch.from_numpy(b[fit]), torch.from_numpy(a[fit])
    pr, pc = torch.as_tensor(p_own, dtype=torch.float64), torch.as_tensor(p_opp, dtype=torch.float64)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opp = torch.zeros(len(p_opp), dtype=torch.float64, requires_grad=True)
    R = torch.zeros(len(p_own), len(p_opp), dtype=torch.float64, requires_grad=matchup)
    params = [bias, opp] + ([R] if matchup else [])
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-10)

    def closure():
        opt.zero_grad()
        z = o_t + bias + opp[b_t] + (centred(R, pr, pc)[a_t, b_t] if matchup else 0)
        loss = F.binary_cross_entropy_with_logits(z, y_t) + l2_opp * opp.pow(2).sum() + l2_mx * R.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        M = centred(R, pr, pc).numpy()
        o = opp.numpy().copy()
        logit = off + bias.item() + o[b] + (M[a, b] if matchup else 0)
    return logit, o, M


def nll(z, y):
    p = np.clip(expit(z), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def opp_contexts(ctx, y, lev, gsp):
    """{" opp-ctx": ..., " oppmx-ctx": ...} frozen context logits (fit on train games, matchup ridge on val)."""
    tr, va, te = gsp == 0, gsp == 1, gsp == 2
    b, a = lev["opp"], lev["own"]
    z1, o1, _ = fit_offset(ctx, y, b, a, tr, lev["p_own"], lev["p_opp"], False)
    cand = [(l2,) + fit_offset(ctx, y, b, a, tr, lev["p_own"], lev["p_opp"], True, l2_mx=l2) for l2 in MX_GRID]
    vl = [nll(c[1][va], y[va]) for c in cand]
    l2, z2, o2, M2 = cand[int(np.argmin(vl))]
    info = {"opp-ctx": {"test logloss (context only)": nll(z1[te], y[te]), "val logloss": nll(z1[va], y[va]),
                        "base test logloss": nll(ctx[te], y[te])},
            "oppmx-ctx": {"test logloss (context only)": nll(z2[te], y[te]), "l2_mx": l2,
                          "val logloss per l2_mx": dict(zip(map(str, MX_GRID), vl))}}
    return {" opp-ctx": z1, " oppmx-ctx": z2}, info


# ---------------------------------------------------------------- opponent strength / matchup tables

def level_effects(off, y, lev, gd, n_lev, iters=30):
    """Per-level logit effect theta[k] with a fixed offset (Newton, one-hot => diagonal), draft-clustered sandwich
    SE, games per level and raw win rate."""
    th = np.zeros(n_lev)
    for _ in range(iters):
        p = expit(off + th[lev])
        g, h = np.bincount(lev, y - p, n_lev), np.bincount(lev, p * (1 - p), n_lev)
        th = np.clip(th + np.where(h > 0, g / np.maximum(h, 1e-12), 0), -5, 5)
    p = expit(off + th[lev])
    h = np.bincount(lev, p * (1 - p), n_lev)
    key, inv = np.unique(gd.astype(np.int64) * n_lev + lev, return_inverse=True)
    u = np.bincount(inv, y - p)
    V = np.bincount(key % n_lev, u ** 2, n_lev)
    n = np.bincount(lev, minlength=n_lev).astype(float)
    se = np.where(h > 0, np.sqrt(V) / np.maximum(h, 1e-12), np.nan)
    return th, se, n, np.bincount(lev, y, n_lev) / np.maximum(n, 1)


def wald_q(est, se, ok, df):
    q = float(np.sum((est[ok] / se[ok]) ** 2))
    return {"Q": q, "df": int(df), "p": float(chi2.sf(q, max(df, 1)))}


def opponent_report(champ_logit, data, lev, models):
    y, gd = data["y"].astype(np.float64), data["gd"]
    b, a = lev["opp"], lev["own"]
    nb, na = len(lev["opp labels"]), len(lev["own labels"])
    th, se, n, wr = level_effects(champ_logit, y, b, gd, nb)
    share = n / n.sum()
    strength = -(th - np.nansum(share * th))     # + = harder opponent, game-weighted mean 0
    ok = (n >= MIN_CELL) & np.isfinite(se)
    fitted = {}
    if OPP in models:
        o = models[OPP][0].opp.detach().numpy().astype(np.float64)
        fitted = {"+ opp arm strength": -(o - share @ o)}
    rep = {"levels": [{"opponent": lab, "strength": float(strength[k]), "se": float(se[k]), "games": int(n[k]),
                       "share": float(share[k]), "user win rate": float(wr[k]),
                       **{f: float(v[k]) for f, v in fitted.items()}} for k, lab in enumerate(lev["opp labels"])],
           "heterogeneity": wald_q(strength, se, ok, ok.sum() - 1),
           "note": "logit effect on the user's win with the champion's full game logit (deck + context incl. record) "
                   "as offset; all games (train in-sample for the champion); + = harder; draft-clustered SE"}
    if fitted:
        r = pearsonr(strength[ok], fitted["+ opp arm strength"][ok]).statistic if ok.sum() > 2 else np.nan
        rep["r(table, + opp arm)"] = float(r)
    kn = data["pg"]["known"] if "known" in data["pg"] else np.ones(len(y), bool)
    wb = data["pg"]["wins_before"]
    rep["mean opponent strength by wins before"] = {
        str(w): float(strength[b[kn & (wb == w)]].mean()) for w in range(7) if (kn & (wb == w)).sum() > 0}
    rep["sorted"] = [lab for _, lab in sorted(zip(-strength, lev["opp labels"]))]
    # matchups
    off = champ_logit + th[b]
    cell = a * nb + b
    tc, sc, nc, _ = level_effects(off, y, cell, gd, na * nb)
    T, S, N = tc.reshape(na, nb), sc.reshape(na, nb), nc.reshape(na, nb)
    Tc = centred(np.where(N > 0, T, 0.0), lev["p_own"], lev["p_opp"])
    okc = (N >= MIN_CELL) & np.isfinite(S)
    rows, cols = okc.any(1).sum(), okc.any(0).sum()
    mx = {"own labels": lev["own labels"], "opponent labels": lev["opp labels"],
          "effect (centred, + = own wins more)": Tc.tolist(), "se": np.where(np.isfinite(S), S, -1).tolist(),
          "games": N.astype(int).tolist(),
          "rps test (cells with >= %d games)" % MIN_CELL: wald_q(Tc, S, okc, (rows - 1) * (cols - 1)),
          "note": "cell effect with offset champion logit + opponent effect, double-centred with the train marginals "
                  "(no own / opponent main effect); SE of the uncentred cell (approximate)"}
    if MX in models:
        M = models[MX][0].mx().detach().numpy().astype(np.float64)
        mx["+ opp x own arm M"] = M.tolist()
        mx["r(table, arm M) over cells"] = float(pearsonr(Tc[okc], M[okc]).statistic) if okc.sum() > 2 else None
    return rep, mx, (strength, se, n, Tc, S, N)


def own_colour_check(fits, data, lev):
    """Mean deck score by own colour level (game-weighted, train games), each arm vs the champion: the opp x own
    term must not take the own-colour effect out of the deck score."""
    trg = data["gsp"] == 0
    a, gd = lev["own"][trg], data["gd"][trg]
    na = len(lev["own labels"])
    n = np.bincount(a, minlength=na)
    ok = n >= MIN_CELL
    means = {v: np.bincount(a, f["scores"][0][gd], na) / np.maximum(n, 1) for v, f in fits.items()}
    out = {}
    for v, m in means.items():
        out[v] = {"own-colour mean deck score": {lab: float(m[k]) for k, lab in enumerate(lev["own labels"]) if ok[k]},
                  "r with champion": float(np.corrcoef(m[ok], means[CHAMP][ok])[0, 1]) if ok.sum() > 2 else None,
                  "sd ratio vs champion": float(m[ok].std() / (means[CHAMP][ok].std() + 1e-12))}
    return out


def print_tables(rep, mx, log=print):
    log("  opponent strength (logit, + = harder; SE):")
    for r in sorted(rep["levels"], key=lambda r: -r["strength"]):
        if r["games"] >= MIN_CELL:
            log(f"    {r['opponent']:10s} {r['strength']:+.3f} ± {r['se']:.3f}  share {r['share']:.3f}  "
                f"user WR {r['user win rate']:.3f}" + (f"  arm {r['+ opp arm strength']:+.3f}"
                                                       if "+ opp arm strength" in r else ""))
    h = rep["heterogeneity"]
    log(f"    heterogeneity Q {h['Q']:.1f} df {h['df']} p {h['p']:.2g};  by wins before: "
        + " ".join(f"{k}:{v:+.3f}" for k, v in rep["mean opponent strength by wins before"].items()))
    T, N = np.array(mx["effect (centred, + = own wins more)"]), np.array(mx["games"])
    rows = [i for i in range(len(mx["own labels"])) if (N[i] >= MIN_CELL).sum() > 1]
    cols = [j for j in range(len(mx["opponent labels"])) if (N[:, j] >= MIN_CELL).sum() > 1]
    log("  matchups (own row x opponent col, centred logit x100):")
    log("    " + " " * 10 + "".join(f"{mx['opponent labels'][j][:6]:>7s}" for j in cols))
    for i in rows:
        log(f"    {mx['own labels'][i][:10]:10s}" + "".join(
            f"{100 * T[i, j]:+7.1f}" if N[i, j] >= MIN_CELL else f"{'.':>7s}" for j in cols))
    q = next(v for k, v in mx.items() if k.startswith("rps test"))
    log(f"    rps test Q {q['Q']:.1f} df {q['df']} p {q['p']:.2g}")


# ---------------------------------------------------------------- the run

def levels(data, min_share=MIN_SHARE):
    trg = data["gsp"] == 0
    b, ol = colour_levels(data["opp_raw"], trg, min_share)
    a, al = colour_levels(data["own_raw"], trg, min_share)
    p_opp = np.bincount(b[trg], minlength=len(ol)) / trg.sum()
    p_own = np.bincount(a[trg], minlength=len(al)) / trg.sum()
    return {"opp": b, "own": a, "opp labels": ol, "own labels": al, "p_opp": p_opp, "p_own": p_own}


def run(data, arms, seeds=1, fit=G.FIT, n_boot=G.N_BOOT, min_share=MIN_SHARE, name="", log=print):
    t0 = time.time()
    d, pg, has = data["d"], data["pg"], data["has"]
    D = len(d.split)
    lev = levels(data, min_share)
    log(f"  opponent levels {lev['opp labels']}\n  own levels {lev['own labels']}")
    data["bt_game"], data["bt_draft"] = G.deleaked_per_game(pg, D)
    fits, models, champ_logit = fit_arms(data, arms, lev, seeds, fit, log)
    ctx, cinfo = fit_context(pg, pg["split"] == 0)
    ctx_dl, cinfo_dl = fit_context({**pg, "skill": data["bt_game"]}, pg["split"] == 0)
    octx, oinfo = opp_contexts(ctx, data["y"].astype(np.float64), lev, data["gsp"])
    subs_all = subsets(pg, has, data["main_bix"])
    sub2 = {k: v for k, v in subs_all.items() if k in OPP_SUBSETS}
    subs = {"": subs_all, " dl-ctx": sub2, " opp-ctx": sub2, " oppmx-ctx": sub2}
    scores = {v: f["scores"] for v, f in fits.items()}
    cached = data.get("cached")
    if cached is not None:
        scores[CACHED] = cached[None]
    pairs = [(CHAMP, v) for v in arms] + ([(OPP, MX)] if OPP in arms and MX in arms else []) \
        + ([(CACHED, CHAMP)] if cached is not None else [])
    log(f"  SC-GLL context test logloss: base {cinfo['test logloss (context only)']:.5f}, dl "
        f"{cinfo_dl['test logloss (context only)']:.5f}, opp {oinfo['opp-ctx']['test logloss (context only)']:.5f}, "
        f"oppmx {oinfo['oppmx-ctx']['test logloss (context only)']:.5f} [{time.time() - t0:.0f}s]")
    ev, losses, ngames = G.evaluate(scores, pg, {"": ctx, " dl-ctx": ctx_dl, **octx}, subs, D, pairs, None, n_boot,
                                    G.day_blocks(d), name, log)
    sec = G.secondary(fits, data, cached)
    own = own_colour_check(fits, data, lev)
    for v, o in sec.items():
        log(f"  {v:14s} draft spearman {o['draft spearman (calibrated)']:.4f} "
            f"({o['draft spearman (calibrated) - champion']:+.4f})  sd {o['deck score sd (test drafts)']:.3f}"
            + (f"  card r {o['card value spearman r with champion']:.3f}"
               f" (pearson {o['card value pearson r with champion']:.3f})" if "card value spearman r with champion" in o
               else "") + (f"  own-colour r {own[v]['r with champion']:.3f}" if v in own and own[v]["r with champion"]
                           is not None else ""))
    rep, mx, _ = opponent_report(champ_logit, data, lev, models)
    print_tables(rep, mx, log)
    res = {"arms": list(fits), "seeds": seeds, "fit": fit, "grid": G.GRID, "mx grid": MX_GRID, "l2 opp": L2_OPP,
           "min share": min_share, "context": cinfo, "context dl": cinfo_dl, "context opp": oinfo,
           "primary subset": "experienced", "colour levels": {k: lev[k] for k in ("opp labels", "own labels")},
           "train shares": {"opp": lev["p_opp"].tolist(), "own": lev["p_own"].tolist()}, **data.get("info", {}),
           "fits": {v: {k: x for k, x in f.items() if k not in ("scores", "w")} for v, f in fits.items()},
           **ev, "secondary": sec, "own colour check": own, "opponent strength": rep, "matchups": mx,
           "seconds": time.time() - t0}
    return res, scores, losses, ngames


# ---------------------------------------------------------------- real data

def load_set(set_code, use_aff=True):
    from mtgdata.games import game_path
    data = G.load_set(set_code, use_aff)
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, data["d"].draft_id, cols=("draft_id", "opp_colors", "main_colors"))
    if "opp_colors" not in g.columns or "main_colors" not in g.columns:
        raise SystemExit(f"{set_code}: game file lacks opp_colors / main_colors")
    pos = {x: i for i, x in enumerate(data["d"].draft_id.tolist())}
    gd = np.array([pos[x] for x in g["draft_id"].to_list()])
    assert len(gd) == len(data["y"]) and (gd == data["gd"]).all(), "colour columns misaligned with game_fm_v2 games"
    data["opp_raw"] = np.array(g["opp_colors"].to_list(), dtype=object)
    data["own_raw"] = np.array(g["main_colors"].to_list(), dtype=object)
    return data


def parse(argv):
    sets, arms, seeds, n_boot, aff, threads, ms = [], [], 1, G.N_BOOT, True, int(_T), MIN_SHARE
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
        elif a == "--min-share":
            ms = float(next(it))
        elif a == "--no-aff":
            aff = False
        elif a in ARMS:
            arms.append(a)
        else:
            sets.append(a)
    chosen = [ARMS[a] for a in (arms or list(ARMS))]
    tag = "" if not arms or set(arms) == set(ARMS) else "_" + "_".join(a for a in ARMS if a in arms)
    return sets, chosen, tag, seeds, n_boot, aff, threads, ms


def main(argv):
    G._no_lightgbm()
    sets, arms, tag, seeds, n_boot, aff, threads, ms = parse(argv)
    torch.set_num_threads(threads)
    for set_code in sets:
        t0 = time.time()
        data = load_set(set_code, aff)
        res, scores, losses, ngames = run(data, arms, seeds, n_boot=n_boot, min_share=ms, name=set_code,
                                          log=lambda m: print(m, flush=True))
        res = {"set": set_code, "games": int(len(data["y"])), **res, "total seconds": time.time() - t0}
        G.save(ROOT / "results" / f"{set_code}.PremierDraft" / "game_fm_opp", tag, data["d"], data["has"], res,
               scores, losses, ngames)
        print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)
        del data


# ---------------------------------------------------------------- synthetic self-test

def synthetic(D=6000, seed=0):
    """game_fm_v2.synthetic's structure plus colours: 4 card colours of 6 nonland cards with different mean power
    (an own-pair main effect carried by the cards), each draft plays one of 6 pairs; per game an opponent level
    (6 pairs, mono R, WUB, a rare 4-colour, unknown) with a strength, a double-centred own x opponent matchup
    matrix, and record matching on opponent colours (P(level) tilted towards stronger levels as wins grow)."""
    from decks.deckstrength import record_onehot
    rng = np.random.default_rng(seed)
    C, n_land, nl = 30, 6, 24
    grp = np.repeat(np.arange(4), 6)
    w_true = np.r_[rng.normal(0, 0.06, nl) + np.array([0.03, 0.0, -0.02, 0.01])[grp], np.zeros(n_land)]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    own_lab = ["WU", "WB", "WR", "UB", "UR", "BR"]
    skill = rng.normal(0, 0.3, D)
    own = rng.choice(6, D, p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])
    in_col = np.array([[g in pairs[k] for g in grp] for k in range(6)])[own]
    p_inc = np.where(in_col, expit(0.2 + 3.0 * skill[:, None] * w_true[None, :nl] / 0.06 * 0.3), 0.03)
    deck = np.zeros((D, C), np.float32)
    deck[:, :nl] = rng.binomial(2, p_inc)
    lands = 15 + rng.integers(0, 4, D)
    for j in range(D):
        deck[j, nl:] = rng.multinomial(lands[j], np.ones(n_land) / n_land)
    syn = 0.08 * (deck[:, 0] * deck[:, 1] + deck[:, 12] * deck[:, 13])
    eta = deck @ w_true + syn - 0.04 * (lands - 16.5) ** 2
    eta = eta - eta.mean()
    opp_lab = own_lab + ["R", "WUB", "WUBR", None]
    base_p = np.array([0.17, 0.15, 0.12, 0.14, 0.14, 0.12, 0.06, 0.06, 0.0015, 0.0085])
    base_p /= base_p.sum()
    opp_s = np.array([0.25, -0.1, 0.0, 0.1, -0.2, 0.05, -0.25, 0.15, 0.0, 0.0])   # + = harder
    M = np.zeros((6, len(opp_lab)))
    R = rng.normal(0, 0.25, (6, 6))
    M[:, :6] = R - R.mean(1, keepdims=True) - R.mean(0, keepdims=True) + R.mean()
    n_hist = rng.choice([5, 10, 50, 100, 500], D, p=[0.1, 0.1, 0.3, 0.3, 0.2])
    rank = rng.integers(2, 4, D)
    rows, W, L = [], np.zeros(D), np.zeros(D)
    for i in range(D):
        wv = lv = 0
        while wv < 7 and lv < 3:
            op = rng.random() < 0.5
            mull = int(rng.random() < 0.08 + 0.04 * abs(lands[i] - 16.5)) - int(rng.random() < 0.1)
            pk = base_p * np.exp(0.5 * wv * opp_s)
            k = rng.choice(len(opp_lab), p=pk / pk.sum())
            win = rng.random() < expit(0.15 + eta[i] + skill[i] + 0.1 * (op - 0.5) - 0.12 * (wv - lv) - 0.3 * mull
                                       - opp_s[k] + M[own[i], k])
            rows.append((i, wv + lv + 1, wv, lv, op, mull, k, win))
            wv, lv = wv + win, lv + (not win)
        W[i], L[i] = wv, lv
    gd, match, wb, lb, op, mull, ko, y = (np.array(c) for c in zip(*rows))
    long_run = expit(0.15 + skill + rng.normal(0, 0.1, D))
    bucket = np.round(((n_hist - (W + L)).clip(1) * long_run + W) / np.maximum(n_hist, W + L) / 0.02) * 0.02
    split = np.zeros(D, np.int8)
    split[int(0.7 * D):], split[int(0.85 * D):] = 1, 2
    Gn = len(y)
    exp_b = np.searchsorted([1, 5, 10, 50, 100, 500, 1000], n_hist) + 1
    pg = {"gd": gd, "y": y.astype(float), "split": split[gd], "match": match, "known": np.ones(Gn, bool),
          "state": np.minimum(wb, 6) * 3 + np.minimum(lb, 2), "on_play": op - 0.5, "mull": mull.astype(float),
          "exp": np.eye(8)[exp_b[gd]], "rank": rank[gd], "n_bucket": n_hist[gd].astype(float),
          "skill": bucket[gd], "wins_before": wb, "losses_before": lb, "build_index": np.full(Gn, "0", dtype=object)}
    d = SimpleNamespace(wins=W.astype(np.float32), losses=L.astype(np.float32), split=split, rank=rank,
                        skill=bucket.astype(np.float32), draft_id=np.array([f"d{i}" for i in range(D)]),
                        draft_time=np.array([f"2025-01-{1 + i * 28 // D:02d}" for i in range(D)]))
    cov = np.hstack([pg["on_play"][:, None], pg["mull"][:, None], (bucket[gd] - 0.5)[:, None], np.ones((Gn, 1)),
                     np.eye(N_RANKS)[rank[gd]]]).astype(np.float32)
    feats = lambda Xd: np.stack([Xd[:, nl:].sum(1), (Xd[:, nl:].sum(1) - 16.5) ** 2, Xd[:, :nl].sum(1)], 1)  # noqa: E731
    Fd = feats(deck)
    mu, sd = Fd[split == 0].mean(0), Fd[split == 0].std(0) + 1e-6
    Ed = ((Fd - mu) / sd).astype(np.float32)
    X = deck[gd]
    data = {"X": X, "y": y.astype(np.int8), "gsp": split[gd], "gd": gd, "cov": cov,
            "exp": pg["exp"].astype(np.float32), "rec": record_onehot(wb, lb), "E": Ed[gd], "Ed": Ed, "deck": deck,
            "nonland": np.r_[np.ones(nl), np.zeros(n_land)].astype(np.float32),
            "played": X[split[gd] == 0].sum(0) > 50, "d": d, "pg": pg, "has": np.ones(D, bool),
            "main_bix": np.full(D, "0", dtype=object), "info": {},
            "opp_raw": np.array([opp_lab[k] for k in ko], dtype=object),
            "own_raw": np.array([own_lab[k] for k in own[gd]], dtype=object)}
    return data, {"eta": eta, "opp_s": opp_s, "opp_lab": opp_lab, "M": M, "own_lab": own_lab}


def selftest():
    import tempfile
    import sc_gll
    G._no_lightgbm()
    torch.set_num_threads(2)
    t0 = time.time()
    # 0. colour levels
    raw = np.array(["WU"] * 50 + ["UB"] * 40 + ["R"] * 5 + ["G"] * 1 + ["WUB"] * 1 + ["WUBR"] * 1 + [None, ""] * 1,
                   dtype=object)
    b, lab = colour_levels(raw, np.ones(len(raw), bool), min_share=0.04)
    assert lab == ["R", "WU", "UB", "other mono", "other pair", "other 3+", "unknown"], lab
    assert [lab[k] for k in b[[0, 50, 90, 95, 96, 97, 98, 99]]] == ["WU", "UB", "R", "other mono", "other 3+",
                                                                   "other 3+", "unknown", "unknown"]
    Rm = np.random.default_rng(0).normal(size=(4, 5))
    pr, pc = np.array([.1, .2, .3, .4]), np.full(5, .2)
    Mc = centred(Rm, pr, pc)
    assert np.allclose(Mc @ pc, 0) and np.allclose(pr @ Mc, 0)
    # 1. a fit with the patched class equals the champion's fit when the opponent terms are absent (index cols = 0)
    data, truth = synthetic()
    D = len(truth["eta"])
    fit = {**G.FIT, "batch": 1024}
    small = {**fit, "epochs": 6}
    sub = np.flatnonzero(data["gd"] < 2000)
    Xs, ys, cs = data["X"][sub], data["y"][sub], G.context_cols(CHAMP, data)[sub]
    tr, va = sub[data["gd"][sub] < 1400] - sub[0], sub[data["gd"][sub] >= 1400] - sub[0]
    m1, v1, _ = G.fit_fm(Xs, cs, ys, tr, va, 1e-5, 1e-3, data["nonland"], fit=small)
    nE = data["E"].shape[1]
    cz = np.hstack([cs[:, :-nE], np.zeros((len(sub), 2), np.float32), cs[:, -nE:]])
    kw = dict(j_opp=cz.shape[1] - nE - 2, j_own=cz.shape[1] - nE - 1, p_own=np.ones(1), p_opp=np.ones(1),
              l2_opp=0.0, l2_mx=0.0, matchup=True)
    with model_class(functools.partial(OppFM, **kw)):
        m2, v2, _ = G.fit_fm(Xs, cz, ys, tr, va, 1e-5, 1e-3, data["nonland"], fit=small)
    assert G.DeckFM is DeckFM
    print(f"[1] colour levels / centring OK; OppFM with null opponent terms vs champion fit: val {v1:.6f} vs "
          f"{v2:.6f}, max |dw| {float((m1.w - m2.w).abs().max().detach()):.2e} [{time.time() - t0:.1f}s]")
    assert abs(v1 - v2) < 2e-3
    # 2. the full run
    res, scores, losses, ngames = run(data, list(ARMS.values()), seeds=2, fit=fit, n_boot=60, name="SYN")
    te = data["d"].split == 2
    tr_r = {v: float(np.corrcoef(S[0][te], truth["eta"][te])[0, 1]) for v, S in scores.items()}
    print("[2] r(score, true deck strength) on test drafts: " + "  ".join(f"{v} {r:.3f}" for v, r in tr_r.items()))
    rep, mx = res["opponent strength"], res["matchups"]
    lv = {r["opponent"]: r for r in rep["levels"]}
    true = {lab: s for lab, s in zip(truth["opp_lab"], truth["opp_s"]) if lab in lv and lv[lab]["games"] >= MIN_CELL}
    est = np.array([lv[k]["strength"] for k in true])
    arm = np.array([lv[k]["+ opp arm strength"] for k in true])
    tv = np.array(list(true.values()))
    r_tab, r_arm = np.corrcoef(est, tv)[0, 1], np.corrcoef(arm, tv)[0, 1]
    byw = rep["mean opponent strength by wins before"]
    print(f"    opponent strength r with truth: table {r_tab:.3f}, + opp arm {r_arm:.3f}; by wins before "
          f"0 {byw['0']:+.3f} -> 6 {byw['6']:+.3f}")
    assert r_tab > 0.9 and r_arm > 0.9 and byw["6"] > byw["0"]
    ol, al = mx["opponent labels"], mx["own labels"]
    Mt = np.zeros((len(al), len(ol)))
    for i, a in enumerate(truth["own_lab"]):
        for j, b in enumerate(truth["opp_lab"][:6]):
            Mt[al.index(a), ol.index(b)] = truth["M"][i, j]
    N = np.array(mx["games"])
    ok = N >= MIN_CELL
    r_mx_tab = np.corrcoef(np.array(mx["effect (centred, + = own wins more)"])[ok], Mt[ok])[0, 1]
    r_mx_arm = np.corrcoef(np.array(mx["+ opp x own arm M"])[ok], Mt[ok])[0, 1]
    q = mx[f"rps test (cells with >= {MIN_CELL} games)"]
    print(f"    matchup r with truth: table {r_mx_tab:.3f}, arm M {r_mx_arm:.3f}; rps Q {q['Q']:.0f} df {q['df']}")
    assert r_mx_tab > 0.5 and r_mx_arm > 0.5 and q["p"] < 0.01
    own = res["own colour check"]
    print(f"    own-colour mean deck score r with champion: " + "  ".join(
        f"{v} {o['r with champion']:.3f} (sd x{o['sd ratio vs champion']:.2f})" for v, o in own.items()))
    assert own[MX]["r with champion"] > 0.9 and own[MX]["sd ratio vs champion"] > 0.7
    assert tr_r[MX] > tr_r[CHAMP] - 0.02 and tr_r[OPP] > tr_r[CHAMP] - 0.02
    co = res["context opp"]
    assert co["opp-ctx"]["test logloss (context only)"] < co["opp-ctx"]["base test logloss"]
    for p in (f"{OPP} - {CHAMP}", f"{MX} - {CHAMP}", f"{MX} - {OPP}"):
        for s in ("experienced", "all", "match 1", "main build", "experienced dl-ctx", "experienced opp-ctx",
                  "experienced oppmx-ctx", "match 1 oppmx-ctx"):
            r = res["pairs"][p][s]
            assert np.isfinite(r["delta"]) and r["se"] > 0 and r["n_seeds"] == 2, (p, s, r)
    for s in ("experienced", "experienced opp-ctx"):
        r = res["pairs"][f"{OPP} - {CHAMP}"][s]
        print(f"    + opp - champion, {s}: {r['delta']:+.3f} ± {r['se']:.3f}")
    sec = res["secondary"]
    assert all(np.isfinite(sec[v]["card value spearman r with champion"]) for v in ARMS.values())
    # 3. saved files: sc_gll.py meta reads them
    with tempfile.TemporaryDirectory() as tmp:
        for s in ("S1", "S2"):
            G.save(Path(tmp) / "results" / f"{s}.PremierDraft" / "game_fm_opp", "", data["d"], data["has"],
                   {"set": s, **res}, scores, losses, ngames)
        json.loads((Path(tmp) / "results" / "S1.PremierDraft" / "game_fm_opp" / "results.json").read_text())
        old = sc_gll.ROOT
        sc_gll.ROOT = Path(tmp)
        try:
            sc_gll.meta("--file", "game_fm_opp/results.json", "--subset", "experienced opp-ctx", "S1", "S2")
        finally:
            sc_gll.ROOT = old
    assert parse(["MSH"])[2] == "" and parse(["MSH", "opp"])[1:3] == ([OPP], "_opp")
    print(f"self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(sys.argv[1:])
