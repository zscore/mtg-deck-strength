"""Deck synergy, round 2: deck features, nonland-only synergy, and synergy priced from how humans draft.

    python scripts/deck_synergy_v2.py MSH [split_half=1]

Premier Draft games, same split and context columns as deck_features.py (play/draw, mulligans, ranks,
skill, experience, record before the game). Every model is scored on the test games, paired against
the linear model (mean log-loss difference, ±95%).

  linear                     card counts + context
  + deck features            nonlinear deck facts from Scryfall data (cardinfo.deck_features: land count,
                             curve shape, colours, castability), as extra context columns
  fm4 nonland + deck feat    pair terms only between nonland cards, with the deck features alongside, so
                             synergy has to be something other than colour consistency / the mana base
  fm4 all + deck feat        pair terms for every card (the old FM) with deck features, for comparison
  + draft affinity           one extra column: the deck's mean pairwise affinity from the pick model
                             (picknet `pool` model: how much having card j raises the odds a human takes
                             card i). Human drafting as a synergy prior with a single fitted weight.
  fm4 nonland + ... + aff.   both: the nonland FM with the affinity column (do they stack?)
  + affinity pairs           the same affinity, but the FM's pair vectors are a learned linear map of the
                             pick model's card vectors (low-rank, shared strength from draft behaviour)

split_half=1 also refits the nonland FM on two random halves of the training drafts and reports how
well its pair values replicate (pairs that share >= 2,000 training games), against card power.
Writes results/{SET}.PremierDraft/deck_synergy_v2/{results.json, test_preds.parquet}.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.cardinfo import card_info, deck_features  # noqa: E402
from openlane.data import PARQUET, ROOT  # noqa: E402
from openlane.deckstrength import (aligned, calibrate, covariates, deck_metrics, experience, flag,  # noqa: E402
                                   game_split, load_drafts, most_played_deck, record_before, record_onehot)
from openlane.games import game_path  # noqa: E402
from openlane.synergy import DeckFM, _train, fit  # noqa: E402


def nll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pick_affinity(set_code: str, names: list[str]):
    """Card vectors from the set's picknet `pool` model: e_j (what a pool card contributes) and a_i (what a
    candidate responds to). S[i, j] = <a_i, e_j>; returns (S symmetrised, [C, 2d] vectors) or None."""
    path = ROOT / "results" / f"{set_code}.PremierDraft" / "picknet" / "models" / "pool_d32.pt"
    if not path.exists():
        return None
    from openlane.picknet import load_model
    cols = pl.scan_parquet(PARQUET / f"{set_code}.PremierDraft.parquet").collect_schema().names()
    pick_names = [c[len("pack_card_"):] for c in cols if c.startswith("pack_card_")]
    m, _ = load_model(path)
    with torch.no_grad():
        e = m.cards().numpy()[:-1]
        a = m.a.weight.numpy()[:-1]
    pos = {n: i for i, n in enumerate(pick_names)}
    order = np.array([pos[n] for n in names])
    e, a = e[order], a[order]
    S = a @ e.T
    S = (S + S.T) / 2
    np.fill_diagonal(S, 0)
    return S.astype(np.float32), np.hstack([e, a]).astype(np.float32)


def mean_pair(X, S, mask):
    """Mean pairwise S over distinct nonland card copies in each deck."""
    Xn = X * mask
    n = Xn.sum(1)
    tot = ((Xn @ S) * Xn).sum(1) - (Xn ** 2 * np.diag(S)).sum(1)
    return tot / np.maximum(n * (n - 1), 1)


class AffinityFM(DeckFM):
    """FM whose pair vectors are v_i = P z_i, z_i fixed card vectors from the pick model (P learned, [2d, k])."""

    def __init__(self, n_cards, n_cov, Z, k, vmask):
        super().__init__(n_cards, n_cov, 0)
        self.register_buffer("Z", torch.as_tensor(Z / (np.linalg.norm(Z, axis=1, keepdims=True).mean() + 1e-6)))
        self.P = nn.Parameter(0.01 * torch.randn(Z.shape[1], k))
        self.vmask = torch.as_tensor(vmask, dtype=torch.float32)[:, None]

    @property
    def v(self):
        return (self.Z @ self.P) * self.vmask

    def penalty(self, l2_w, l2_v, l2_dw):
        return l2_w * self.w.pow(2).sum() + l2_v * self.P.pow(2).sum()


def main(set_code: str, split_half: str = "1"):
    prev = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v2" / "results.json"
    keep_split = json.loads(prev.read_text()).get("split_half") if prev.exists() and split_half == "0" else None
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v2"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(set_code, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    draft_of_game = g["draft_id"].to_numpy()
    deck, has_deck = most_played_deck(d, g)
    del g
    info = card_info(names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    F, fnames = deck_features(X, info)
    mu, sd = F[sp == 0].mean(0), F[sp == 0].std(0) + 1e-6
    ctx_f = np.hstack([ctx, (F - mu) / sd]).astype(np.float32)
    aff = pick_affinity(set_code, names)
    te = np.flatnonzero(sp == 2)
    print(f"{set_code}: {len(y)} games (train {np.sum(sp == 0)}, test {len(te)}), {len(fnames)} deck features, "
          f"pick affinity {'yes' if aff is not None else 'no'}  [{time.time() - t0:.0f}s]", flush=True)
    log = lambda s: None  # noqa: E731

    def best_of(fitter, grid):
        """Fit each setting, keep the best on validation."""
        fits = [(fitter(**kw), kw) for kw in grid]
        (m, v, pr), kw = min(fits, key=lambda f: f[0][1])
        return m, v, pr, kw

    runs = {
        "linear": lambda: best_of(lambda l2w: fit(X, ctx, y, sp, 0, l2w, 0, log=log), [{"l2w": 1e-5}, {"l2w": 1e-4}]),
        "linear + deck features": lambda: best_of(lambda l2w: fit(X, ctx_f, y, sp, 0, l2w, 0, log=log),
                                                  [{"l2w": 1e-5}, {"l2w": 1e-4}]),
        "fm4 nonland + deck features": lambda: best_of(
            lambda l2v: fit(X, ctx_f, y, sp, 4, 1e-5, l2v, log=log, vmask=nonland), [{"l2v": 1e-3}, {"l2v": 1e-2}]),
        "fm4 all cards + deck features": lambda: best_of(
            lambda l2v: fit(X, ctx_f, y, sp, 4, 1e-5, l2v, log=log), [{"l2v": 1e-3}, {"l2v": 1e-2}]),
    }
    if aff is not None:
        S, Z = aff
        a = mean_pair(X, S, nonland)
        a = ((a - a[sp == 0].mean()) / (a[sp == 0].std() + 1e-6))[:, None].astype(np.float32)
        ctx_fa = np.hstack([ctx_f, a])
        runs["linear + deck features + draft affinity"] = lambda: best_of(
            lambda l2w: fit(X, ctx_fa, y, sp, 0, l2w, 0, log=log), [{"l2w": 1e-5}, {"l2w": 1e-4}])
        runs["fm4 nonland + deck features + draft affinity"] = lambda: best_of(
            lambda l2v: fit(X, ctx_fa, y, sp, 4, 1e-5, l2v, log=log, vmask=nonland), [{"l2v": 1e-3}, {"l2v": 1e-2}])

        def fit_aff(l2v, k=8):
            torch.manual_seed(0)
            m = AffinityFM(X.shape[1], ctx_f.shape[1], Z, k, nonland)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            return _train(m, opt, X, ctx_f, y, sp, None, lambda: m.penalty(1e-5, l2v, 0), 40, 4096, 3, "cpu")
        runs["affinity pairs (k=8) + deck features"] = lambda: best_of(fit_aff, [{"l2v": 1e-4}, {"l2v": 1e-3}])

    deck_cols = {"F": lambda Xd: (deck_features(Xd, info)[0] - mu) / sd}
    if aff is not None:
        deck_cols["aff"] = lambda Xd, a_tr=mean_pair(X, S, nonland)[sp == 0]: (
            (mean_pair(Xd, S, nonland) - a_tr.mean()) / (a_tr.std() + 1e-6))[:, None]
    preds, rows, models = {}, [], {}
    for name, run in runs.items():
        t1 = time.time()
        m, v, predict, kw = run()
        preds[name], models[name] = predict(te), m
        row = {"model": name, "val logloss": v, "setting": kw, "fit seconds": time.time() - t1}
        if name == "linear + deck features + draft affinity":
            row["affinity weight (log-odds per sd)"] = float(m.cov.weight.detach().numpy()[0, -1])
        if name == "linear + deck features":
            wf = m.cov.weight.detach().numpy()[0, ctx.shape[1]:]
            row["deck feature weights (per sd)"] = dict(zip(fnames, np.round(wf, 4).tolist()))
        rows.append(row)
        print(f"  {name:45s} val {v:.5f} {kw} [{time.time() - t1:.0f}s]", flush=True)
    base = nll(y[te], preds["linear"])
    for r in rows:
        p = preds[r["model"]]
        dl = nll(y[te], p) - base
        r.update({"test logloss": log_loss(y[te], p), "test auc": roc_auc_score(y[te], p),
                  "vs linear": float(dl.mean()), "vs linear ±95%": float(1.96 * dl.std() / np.sqrt(len(dl)))})
        print(f"  {r['model']:45s} test {r['test logloss']:.5f} auc {r['test auc']:.4f} "
              f"vs linear {r['vs linear'] * 1000:+.2f} ± {r['vs linear ±95%'] * 1000:.2f} (x1000)", flush=True)

    draft_level(d, deck.astype(np.float32), has_deck, models, rows, ctx.shape[1], deck_cols)
    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "models": rows}
    if keep_split:  # rerun without the split-half refits: keep the earlier result
        res["split_half"] = keep_split
    if int(split_half):
        res["split_half"] = replicate(X, ctx_f, y, sp, draft_of_game, nonland, rows, aff)
        print(f"  split-half: {res['split_half']}", flush=True)
    res["seconds"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, indent=1))
    pl.DataFrame({"won": y[te], **preds}).write_parquet(out / "test_preds.parquet")
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


def draft_level(d, deck, has_deck, models, rows, n_ctx, deck_cols):
    """Score each draft's most-played deck with a model's deck terms (cards, pairs, deck-level columns), calibrate
    to the event records on training drafts and report Spearman / NLL on test drafts, as deck_synergy.py does."""
    tr, te = d.mask("train"), d.mask("test") & has_deck
    Fd = deck_cols["F"](deck)
    for r in rows:
        m = models[r["model"]]
        with torch.no_grad():
            s = m.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
        wc = m.cov.weight.detach().numpy()[0, n_ctx:]
        extra = [] if len(wc) == 0 else [Fd] + ([deck_cols["aff"](deck)] if len(wc) > Fd.shape[1] else [])
        if extra:
            s = s + np.hstack(extra) @ wc
        p = calibrate(s[tr], d.rank[tr], d.wins[tr], d.losses[tr], s, d.rank)
        r.update({f"draft {k}": float(v) for k, v in deck_metrics(p[te], d.wins[te], d.losses[te]).items()})
        print(f"  {r['model']:45s} draft spearman {r['draft spearman']:.4f} nll {r['draft nll']:.4f}", flush=True)


def replicate(X, ctx_f, y, sp, draft_of_game, nonland, rows, aff, min_co=2000):
    """Refit on two halves of the training drafts; correlate card power and pair values between halves."""
    l2v = next(r["setting"]["l2v"] for r in rows if r["model"] == "fm4 nonland + deck features")
    drafts = np.unique(draft_of_game[sp == 0])
    half = np.random.default_rng(1).permutation(len(drafts)) < len(drafts) // 2
    in_a = np.isin(draft_of_game, drafts[half])
    fits = []
    for side in (in_a, ~in_a):
        s = np.where((sp == 0) & ~side, -1, sp)  # the other half's training games drop out; val unchanged
        m, _, _ = fit(X, ctx_f, y, s, 4, 1e-5, l2v, log=lambda s: None, vmask=nonland)
        fits.append(m)
    P = (X[sp == 0] * nonland > 0).astype(np.float32)
    co = P.T @ P
    iu = np.triu_indices(len(co), 1)
    ok = co[iu] >= min_co
    S = [(m.v.detach().numpy() @ m.v.detach().numpy().T)[iu][ok] for m in fits]
    W = [m.w.detach().numpy() for m in fits]
    seen = (X[sp == 0] > 0).sum(0) >= min_co
    out = {"l2_v": l2v, "pairs": int(ok.sum()), "pair r": float(np.corrcoef(*S)[0, 1]),
           "power r": float(np.corrcoef(W[0][seen], W[1][seen])[0, 1])}
    if aff is not None:  # do the learned pairs agree with the pick model's affinity?
        out["pair vs draft affinity r"] = float(np.mean([np.corrcoef(s, aff[0][iu][ok])[0, 1] for s in S]))
    top = [set(np.argsort(-s)[:50]) for s in S]
    out["top50 overlap"] = len(top[0] & top[1])
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
