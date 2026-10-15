"""Joint Premier Draft + Sealed deck FM for the site: python scripts/site_joint_fm.py SET [SET ...]

Run with wheelnet's .venv (torch). The site's deck FM (../mtg-draft-sim-site/tools/export_deck_model.py) is
fit once per event; Sealed has 10-20x fewer games than Premier Draft. Here one model is fit on both events' games:

    logit P(win) = b_t + ctx . w_ctx,t + w . x + [t = sealed] dw . x + pair_v(x) + [t = sealed] pair_vs(x)
                   + F(x) . w_F,t + aff(x) * w_aff,t

  shared        card power w, pair vectors v (rank K, nonland), as the site's model
  per event t   bias, context weights, deck-feature weights, affinity weight (the site's model has these per event)
  sealed only   power offset dw (L2 l2_dw toward the shared w) and KS extra pair vectors vs (L2 l2_v)
Grid: l2_dw x KS, with (l2_w, l2_v) the site's draft choice; the fit is picked on Sealed validation log loss
(early stopping uses the mean of the two events' validation losses). Splits are each game file's own event_split,
so Sealed test events are the site's Sealed test events.

Post-processing as the site's export: power shrunk by n / (n + SHRINK) training games (the shared terms by both
events' games, the sealed terms by Sealed games), pair vectors by its square root, one shared basic weight; clamps
from each event's own training decks. Each event's head is written in the site's per-event format (the sealed head's
v is [v, vs], so deck.js scores it unchanged), plus "sealed_refit": the site's own fit_event on Sealed games (control).

Writes results/{SET}.PremierDraft/site_joint/deck.json (data/deck/{SET}.json with "draft", "sealed", "sealed_refit"
replaced / added) and fm.json (grid, validation losses, python deck logits of fixture decks for the deck.js check).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

SITE = Path("/Users/zblanton/Documents/repos/mtg-draft-sim-site")
sys.path.insert(0, str(SITE / "tools"))
import export_deck_model as ex  # noqa: E402  (puts mtg-open-lane-synergy/src on the path)
from openlane.cardinfo import card_info, deck_features  # noqa: E402
from openlane.deckstrength import aligned, covariates, event_split, experience, flag, record_before, record_onehot  # noqa: E402,E501
from openlane.games import game_path  # noqa: E402
from openlane.synergy import DeckFM, _train, log_loss  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EVENTS = ("PremierDraft", "Sealed")
GRID = [(l2_dw, ks) for ks in (0, 2) for l2_dw in (1e-4, 1e-3, 1e-2)]


class JointFM(DeckFM):
    """DeckFM with tasks 0 = draft, 1 = sealed: shared w, v; sealed offset dw[1] (dw[0] unused, stays 0); per-task
    covariate layer (bias, context, deck features, affinity); `ks` sealed-only pair vectors."""

    def __init__(self, n_cards, n_cov, k, ks, vmask):
        super().__init__(n_cards, n_cov, k, 2, vmask)
        self.vs_ = nn.Parameter(0.01 * torch.randn(n_cards, ks)) if ks else None

    @staticmethod
    def pair(x, v):
        xv = x @ v
        return 0.5 * (xv.pow(2) - x.pow(2) @ v.pow(2)).sum(-1)

    def deck_score(self, x, task=None):
        s = x @ self.w + self.pair(x, self.v)
        if task is not None:
            extra = x @ self.dw[1]
            if self.vs_ is not None:
                extra = extra + self.pair(x, self.vs_ * self.vmask)
            s = s + extra * (task == 1).float()
        return s

    def penalty(self, l2_w, l2_v, l2_dw):
        p = super().penalty(l2_w, l2_v, l2_dw)
        return p if self.vs_ is None else p + l2_v * (self.vs_ * self.vmask).pow(2).sum()


def load_event(code, event, names):
    g = pl.read_parquet(game_path(code, event))
    sp = event_split(g)
    X = aligned(g, "deck_", names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    return X, y, ctx, sp


def head(m, task, Xt, Ft, mu, sd, at, aff, n_games, keep_all, keep_s, games_with, basic, nctx, ctx_tr, info, Xall):
    """The site's per-event JSON entry for one task of the joint model (Xt, Ft, at, ctx_tr: its training games)."""
    with torch.no_grad():
        w = m.w.numpy() * keep_all + (m.dw[1].numpy() * keep_s if task == 1 else 0)
        v = m.v.numpy() * np.sqrt(keep_all)[:, None]
        if task == 1 and m.vs_ is not None:
            v = np.hstack([v, (m.vs_ * m.vmask).numpy() * np.sqrt(keep_s)[:, None]])
        wc, bias = m.cov.weight.numpy()[task], float(m.cov.bias.numpy()[task])
    w = w.copy()
    w[basic] = (w[basic] * games_with[basic]).sum() / games_with[basic].sum()
    pair = 0.5 * ((Xt @ v) ** 2 - (Xt ** 2) @ (v ** 2)).sum(-1)
    Fs = (Ft - mu) / sd
    deck_logit = Xt @ w + pair + Fs @ wc[nctx:nctx + len(mu)]
    if aff is not None:
        deck_logit = deck_logit + (at - aff[0]) / aff[1] * wc[-1]
    q = lambda z: np.quantile(z, [0.005, 0.995]).round(5).tolist()  # noqa: E731
    r = lambda z, d=5: np.round(np.asarray(z, dtype=np.float64), d).tolist()  # noqa: E731
    fx = np.random.default_rng(0).choice(len(Xt), 25, replace=False)
    fixture = [{"deck": {int(j): int(Xt[i, j]) for j in np.flatnonzero(Xt[i])}, "logit": float(deck_logit[i])}
               for i in fx]
    entry = {"games": int(n_games), "train_games": int(len(Xt)), "base": round(bias + float((ctx_tr @ wc[:nctx]).mean()), 5),
             "deck_logit_mean": round(float(deck_logit.mean()), 5), "deck_logit_sd": round(float(deck_logit.std()), 5),
             "w": r(w), "v": r(v), "games_with": games_with.astype(int).tolist(),
             "ranges": {"pair": q(pair), "features": [q(Ft[:, j]) for j in range(Ft.shape[1])],
                        "aff": None if aff is None else q(at)},
             "features": {"names": list(ex.DECK_FEATURES), "mu": r(mu), "sd": r(sd), "w": r(wc[nctx:nctx + len(mu)]),
                          "centres": ex.centres(Xall, info)},
             "aff": None if aff is None else {"mu": aff[0], "sd": aff[1], "w": round(float(wc[-1]), 5)}}
    return entry, fixture


def run_set(code, log=print):
    t0 = time.time()
    out = REPO / "results" / f"{code}.PremierDraft" / "site_joint"
    out.mkdir(parents=True, exist_ok=True)
    site = json.loads((SITE / "data" / "deck" / f"{code}.json").read_text())
    names = site["names"]
    info = card_info(ex.scryfall_spelling(names), code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    basic = info["is_basic"].to_numpy()
    vec = ex.pool_vectors(code, names)
    l2_w, l2_v = site["draft"]["l2"]

    ev = [load_event(code, e, names) for e in EVENTS]
    X = np.vstack([e[0] for e in ev])
    y = np.concatenate([e[1] for e in ev])
    ctx = np.vstack([e[2] for e in ev])
    sp = np.concatenate([e[3] for e in ev])
    task = np.concatenate([np.full(len(e[1]), t, np.int64) for t, e in enumerate(ev)])
    del ev
    tr = sp == 0
    log(f"=== {code}: {len(y)} games ({(task == 1).sum()} sealed), {len(names)} cards, l2 (w, v) ({l2_w}, {l2_v}) "
        f"[{time.time() - t0:.0f}s]")
    games_with = (X[tr] > 0).sum(0).astype(np.float64)
    games_with_s = (X[tr & (task == 1)] > 0).sum(0).astype(np.float64)
    F, _ = deck_features(X, info)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    cols = [ctx, (F - mu) / sd]
    aff, a = None, None
    if vec is not None:
        a = ex.mean_affinity(X, *vec, nonland)
        aff = (float(a[tr].mean()), float(a[tr].std() + 1e-6))
        cols.append(((a - aff[0]) / aff[1])[:, None])
    c = np.hstack(cols).astype(np.float32)
    del cols
    nctx = ctx.shape[1]

    va_s = np.flatnonzero((sp == 1) & (task == 1))
    va_d = np.flatnonzero((sp == 1) & (task == 0))
    fits = []
    for l2_dw, ks in GRID:
        t1 = time.time()
        torch.manual_seed(0)
        m = JointFM(len(names), c.shape[1], ex.K, ks, nonland)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        m, best, predict = _train(m, opt, X, c, y, sp, task, lambda m=m, b=l2_dw: m.penalty(l2_w, l2_v, b), 40, 4096,
                                  3, "cpu", lr_drops=ex.LR_DROPS)
        ls, ld = log_loss(y[va_s], predict(va_s)), log_loss(y[va_d], predict(va_d))
        fits.append({"l2_dw": l2_dw, "ks": ks, "val mean": float(best), "val sealed": float(ls), "val draft": float(ld),
                     "model": m})
        log(f"  l2_dw {l2_dw:g} ks {ks}: val sealed {ls:.5f} draft {ld:.5f} [{time.time() - t1:.0f}s]")
    pick = min(fits, key=lambda f: f["val sealed"])
    m = pick["model"]
    m.eval()
    keep_all = games_with / (games_with + ex.SHRINK)
    keep_s = games_with_s / (games_with_s + ex.SHRINK)
    heads, fixtures = {}, {}
    for t, nm in ((0, "draft"), (1, "sealed")):
        rows = (task == t) & tr
        heads[nm], fixtures[nm] = head(m, t, X[rows], F[rows], mu, sd, None if a is None else a[rows], aff,
                                       (task == t).sum(), keep_all, keep_s, games_with, basic, nctx, ctx[rows], info, X)
        heads[nm].update({"l2": [l2_w, l2_v, pick["l2_dw"]], "ks": pick["ks"], "joint": True})
    del X, F, c, ctx
    log(f"  picked l2_dw {pick['l2_dw']:g} ks {pick['ks']} [{time.time() - t0:.0f}s]")
    # control: the site's own per-event fit on Sealed games, through the same code
    refit, refit_fx = ex.fit_event(code, "Sealed", names, info, nonland, vec, log)
    deck = {**site, "draft": heads["draft"], "sealed": heads["sealed"], "sealed_refit": refit}
    (out / "deck.json").write_text(json.dumps(deck, separators=(",", ":")))
    (out / "fm.json").write_text(json.dumps({
        "set": code, "grid": [{k: v for k, v in f.items() if k != "model"} for f in fits],
        "picked": {"l2_dw": pick["l2_dw"], "ks": pick["ks"]}, "l2": [l2_w, l2_v],
        "fixtures": {**fixtures, "sealed_refit": refit_fx}, "seconds": time.time() - t0}, indent=1))
    log(f"{code} joint FM done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    torch.set_num_threads(4)
    for s in sys.argv[1:]:
        run_set(s)
