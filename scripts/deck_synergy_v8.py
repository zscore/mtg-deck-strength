"""Deck synergy, round 8: how much of seen-together synergy is partner-specific, and is it causal?

    python scripts/deck_synergy_v8.py MSH

Two problems with rounds 5-7's seen FM (pairs between nonland cards both seen in a game):
1. Most of what its low-rank pairs learned is not partner-specific. A centring penalty (each card's mean pair value
   over its usual partners pushed to 0) removed ~70% of the gain on SOS. The explicit version is a per-card *slope*
   on how many other nonland cards were seen this game.
2. "Seen" includes cards drawn by other cards' effects, so it is partly an outcome of play. A card-draw spell
   that resolves both shows you more spells and wins games: the top slopes on SOS are draw / recursion spells.
   The *opening hand* is dealt before anything happens (mulligans are in the context), so it is the clean input.

For each input (h = 1 if the card is in the opening hand / was seen this game; nonland pairs only):
  no pairs            context + deck features + aff + deck power w + card value u, plus how many lands and spells
                      h holds (one-hot); the seen version also controls game length (post-treatment for the hand)
  slope               + sum_i h_i s_i z_i, z_i = (other nonland cards in h - mean) / sd
  pairs k=4           free low-rank pairs (round 7's model)
  slope + pairs k=4   pairs on top of the slopes: what they add is partner-specific synergy
Pair models: held-out games (vs no pairs, and vs slope for slope + pairs) and split-half pair r.
Writes results/{SET}.PremierDraft/deck_synergy_v8/{results.json, {hand,seen}_pairs.csv, {hand,seen}_hubs.csv,
{hand,seen}_slopes.csv}; pairs and hubs come from the slope + pairs fit.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_synergy_v2 import mean_pair, nll, pick_affinity  # noqa: E402
from deck_synergy_v5 import SeenFM, pair_matrix, train  # noqa: E402
from mtgdata.cardinfo import card_info, deck_features  # noqa: E402
from mtgdata.games import aligned, flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (covariates, experience, game_split, load_drafts, record_before, record_onehot)  # noqa: E402

MIN_CO = 2000
L2V = (3e-5, 1e-4, 3e-4)


class SlopeSeenFM(SeenFM):
    """SeenFM + sum_i h_i s_i z_i, z_i = (other nonland cards in h - mu) / sd. k = 0: no pairs."""

    def __init__(self, n_cards, n_cov, k, nonland, mu, sd, slope=True):
        super().__init__(n_cards, n_cov, max(k, 1), nonland if k else np.zeros(n_cards, np.float32))
        self.register_buffer("nl", torch.as_tensor(nonland, dtype=torch.float32))
        self.mu, self.sd = float(mu), float(sd)
        self.s = torch.nn.Parameter(torch.zeros(n_cards), requires_grad=slope)

    def forward(self, xh, c, task=None):
        out = super().forward(xh, c, task)
        if not self.s.requires_grad:
            return out
        h = xh[:, self.C:].float()
        hn = h * self.nl
        z = (hn.sum(1, keepdim=True) - hn - self.mu) / self.sd
        return out + (h * z) @ self.s

    def penalty(self, l2_w, l2_v):
        return super().penalty(l2_w, l2_v) + l2_w * self.s.pow(2).sum()


def hub_table(m, names, co, X, tr):
    """Per card: u, slope, and the co-occurrence-weighted mean and sd of Q over partners in >= MIN_CO training decks."""
    Q, u, s = pair_matrix(m), m.u.detach().numpy(), m.s.detach().numpy()
    W = np.where(co >= MIN_CO, co, 0).astype(np.float64)
    np.fill_diagonal(W, 0)
    idx = np.flatnonzero((W > 0).sum(1) >= 20)
    Wi = W[idx] / W[idx].sum(1, keepdims=True)
    mean = (Wi * Q[idx]).sum(1)
    sd = np.sqrt((Wi * (Q[idx] - mean[:, None]) ** 2).sum(1))
    Qm = np.where(W[idx] > 0, Q[idx], np.nan)
    return pl.DataFrame({"card": [names[c] for c in idx], "value u": u[idx], "slope": s[idx],
                         "mean Q over partners": mean, "partner spread": sd,
                         "best partner": [names[int(c)] for c in np.nanargmax(Qm, 1)], "best Q": np.nanmax(Qm, 1),
                         "worst partner": [names[int(c)] for c in np.nanargmin(Qm, 1)], "worst Q": np.nanmin(Qm, 1),
                         "partners": (W[idx] > 0).sum(1), "train decks": (X[tr] > 0).sum(0)[idx]})


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v8"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(set_code, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    hand = aligned(g, "opening_hand_", names) > 0
    seen = hand | (aligned(g, "drawn_", names) > 0)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    turns = np.clip(g["num_turns"].cast(pl.Int32, strict=False).fill_null(10).to_numpy(), 3, 20)
    draft_of_game = g["draft_id"].to_numpy()
    del g
    info = card_info(names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    S = pick_affinity(set_code, names)[0]
    tr = sp == 0
    F = deck_features(X, info)[0]
    F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
    a = mean_pair(X, S, nonland)
    a = (a - a[tr].mean()) / (a[tr].std() + 1e-6)
    base_ctx = np.hstack([ctx, F, a[:, None]]).astype(np.float32)
    del F, a
    C = X.shape[1]
    te = np.flatnonzero(sp == 2)
    Pd = (X[tr] * nonland > 0).astype(np.float32)
    co = Pd.T @ Pd
    del Pd
    iu = np.triu_indices(C, 1)
    ok = co[iu] >= MIN_CO
    ids = np.unique(draft_of_game[tr])
    half = set(ids[np.random.default_rng(1).permutation(len(ids)) < len(ids) // 2].tolist())
    in_a = np.array([x in half for x in draft_of_game])
    print(f"{set_code}: {len(y)} games, {ok.sum()} pairs [{time.time() - t0:.0f}s]", flush=True)

    rows = []
    for tag, h, max_l, max_s, length in (("hand", hand, 7, 7, False), ("seen", seen, 12, 15, True)):
        hf = h.astype(np.float32)
        n_l = np.clip((hf @ (1 - nonland)).astype(int), 0, max_l)
        n_s = np.clip((hf @ nonland).astype(int), 0, max_s)
        del hf
        parts = [base_ctx, np.eye(max_l + 1, dtype=np.float32)[n_l], np.eye(max_s + 1, dtype=np.float32)[n_s]]
        if length:
            parts.append(np.eye(18, dtype=np.float32)[turns - 3])
        cs = np.hstack(parts).astype(np.float32)
        mu, sd = float(n_s[tr].mean() - 1), float(n_s[tr].std() + 1e-6)
        XH = np.hstack([X.astype(np.float16), h.astype(np.float16)])
        make = {"no pairs": lambda: SlopeSeenFM(C, cs.shape[1], 0, nonland, mu, sd, slope=False),
                "slope": lambda: SlopeSeenFM(C, cs.shape[1], 0, nonland, mu, sd),
                "pairs k=4": lambda: SlopeSeenFM(C, cs.shape[1], 4, nonland, mu, sd, slope=False),
                "slope + pairs k=4": lambda: SlopeSeenFM(C, cs.shape[1], 4, nonland, mu, sd)}
        nlls, fitted = {}, {}
        for name, mk in make.items():
            grid = L2V if name.endswith("k=4") else (0,)
            fits = [train(mk, XH, cs, y, sp, 1e-5, l2v) + (l2v,) for l2v in grid]
            m, v, pr, l2v = min(fits, key=lambda f: f[1])
            del fits
            p = pr(te)
            nlls[name], fitted[name] = nll(y[te], p), m
            row = {"input": tag, "model": name, "val logloss": v, "l2_v": l2v, "test logloss": log_loss(y[te], p)}
            for ref in ("no pairs", "slope"):
                if ref in nlls and ref != name and (ref == "no pairs" or name == "slope + pairs k=4"):
                    dl = nlls[name] - nlls[ref]
                    row[f"vs {ref}"] = float(dl.mean())
                    row[f"vs {ref} ±95%"] = float(1.96 * dl.std() / np.sqrt(len(dl)))
            if name.endswith("k=4"):
                Qs = []
                for side in (True, False):
                    s_ = np.where(tr & (in_a == side), -1, sp)
                    Qs.append(pair_matrix(train(mk, XH, cs, y, s_, 1e-5, l2v)[0])[iu][ok])
                row["split-half pair r"] = float(np.corrcoef(*Qs)[0, 1])
                row["vs affinity r"] = float(np.corrcoef(pair_matrix(m)[iu][ok], S[iu][ok])[0, 1])
            rows.append(row)
            msg = "  ".join(f"{k} {row[k] * 1000:+.3f} ± {row[k + ' ±95%'] * 1000:.3f}"
                            for k in ("vs no pairs", "vs slope") if k in row)
            print(f"  {tag:5s} {name:18s} l2={l2v:g}  {msg}  r {row.get('split-half pair r', float('nan')):.3f}"
                  f" [{time.time() - t0:.0f}s]", flush=True)
        del XH
        m = fitted["slope + pairs k=4"]
        Q = pair_matrix(m)
        i, j = iu[0][ok], iu[1][ok]
        pl.DataFrame({"card a": [names[t] for t in i], "card b": [names[t] for t in j], "Q": Q[i, j],
                      "pick affinity": S[i, j], "train decks": co[i, j]}) \
            .sort("Q", descending=True).write_csv(out / f"{tag}_pairs.csv")
        hubs = hub_table(m, names, co, X, tr).sort("partner spread", descending=True)
        hubs.write_csv(out / f"{tag}_hubs.csv")
        pl.DataFrame({"card": names, "slope": fitted["slope"].s.detach().numpy(), "nonland": nonland > 0}) \
            .filter("nonland").drop("nonland").sort("slope", descending=True).write_csv(out / f"{tag}_slopes.csv")
        print(f"  {tag} top hubs: " + "; ".join(f"{r['card']} (+ {r['best partner']})"
                                               for r in hubs.head(5).iter_rows(named=True)), flush=True)
    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "pairs": int(ok.sum()), "models": rows,
           "seconds": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
