"""Deck synergy, round 7: final seen-together synergy tables (round 6's model, weaker pair penalty).

    python scripts/deck_synergy_v7.py MSH

Round 6 validated the seen-together pair model (held-out gain on 12/12 sets, unseen-card placebo ~0), but its l2 grid
bottomed out on 11/12 sets, and the published pair table came from round 5's more shrunk fit. Same model, k = 4, l2
grid extended down to 3e-5, the placebo refit on the same grid. From the final seen fit it writes:

  seen_pairs.csv   every nonland pair in >= 2,000 training decks: Q_ij (log-odds when both are seen, beyond each
                   card's own seen value), pick-model affinity, training decks
  card_hubs.csv    per card: its seen value u, and its synergy dependence, the co-occurrence-weighted mean Q_ij over
                   its partners, with the best and worst partner
Writes results/{SET}.PremierDraft/deck_synergy_v7/{results.json, seen_pairs.csv, card_hubs.csv}.
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


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v7"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(set_code, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    seen = (aligned(g, "opening_hand_", names) + aligned(g, "drawn_", names)) > 0
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
    sf = seen.astype(np.float32)
    n_land = np.clip((sf @ (1 - nonland)).astype(int), 0, 12)
    n_spell = np.clip((sf @ nonland).astype(int), 0, 15)
    del sf
    cs = np.hstack([ctx, F, a[:, None], np.eye(18, dtype=np.float32)[turns - 3],
                    np.eye(13, dtype=np.float32)[n_land], np.eye(16, dtype=np.float32)[n_spell]]).astype(np.float32)
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
    for tag, h in (("seen", seen), ("placebo (unseen)", (X > 0) & ~seen)):
        XH = np.hstack([X.astype(np.float16), h.astype(np.float16)])
        m0, v0, pr0 = train(lambda: SeenFM(C, cs.shape[1], 1, np.zeros(C, np.float32)), XH, cs, y, sp, 1e-5, 0)
        base = nll(y[te], pr0(te))
        rows.append({"model": f"{tag}, no pairs", "val logloss": v0, "test logloss": log_loss(y[te], pr0(te))})
        print(f"  {tag:18s} no pairs  val {v0:.5f} [{time.time() - t0:.0f}s]", flush=True)
        for k in (4,):
            make = lambda k=k: SeenFM(C, cs.shape[1], k, nonland)  # noqa: E731
            fits = [train(make, XH, cs, y, sp, 1e-5, l2v) + (l2v,) for l2v in L2V]
            m, v, pr, l2v = min(fits, key=lambda f: f[1])
            del fits
            dl = nll(y[te], pr(te)) - base
            row = {"model": f"{tag}, pairs k={k}", "val logloss": v, "l2_v": l2v, "test logloss": log_loss(y[te], pr(te)),
                   "vs no pairs": float(dl.mean()), "vs no pairs ±95%": float(1.96 * dl.std() / np.sqrt(len(dl))),
                   "pair sd": float(pair_matrix(m)[iu][ok].std())}
            if k == 4:  # split-half replication at the chosen l2
                Qs = []
                for side in (True, False):
                    s_ = np.where(tr & (in_a == side), -1, sp)
                    Qs.append(pair_matrix(train(make, XH, cs, y, s_, 1e-5, l2v)[0])[iu][ok])
                row["split-half pair r"] = float(np.corrcoef(*Qs)[0, 1])
                row["vs affinity r"] = float(np.corrcoef(pair_matrix(m)[iu][ok], S[iu][ok])[0, 1])
            rows.append(row)
            if tag == "seen":
                final_seen = m
            print(f"  {tag:18s} k={k} l2={l2v:g} vs no pairs {row['vs no pairs'] * 1000:+.3f} ± "
                  f"{row['vs no pairs ±95%'] * 1000:.3f} (x1000)  r {row.get('split-half pair r', float('nan')):.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        del XH
    Q = pair_matrix(final_seen)
    u = final_seen.u.detach().numpy()
    i, j = iu[0][ok], iu[1][ok]
    pl.DataFrame({"card a": [names[t] for t in i], "card b": [names[t] for t in j], "seen Q": Q[i, j],
                  "pick affinity": S[i, j], "train decks": co[i, j]}) \
        .sort("seen Q", descending=True).write_csv(out / "seen_pairs.csv")
    W = np.where(co >= MIN_CO, co, 0)
    np.fill_diagonal(W, 0)
    Qm = np.where(W > 0, Q, np.nan)
    has = (W.sum(1) > 0) & (nonland > 0)
    idx = np.flatnonzero(has)
    best = [names[int(np.nanargmax(Qm[c]))] for c in idx]
    worst = [names[int(np.nanargmin(Qm[c]))] for c in idx]
    pl.DataFrame({"card": [names[c] for c in idx], "seen value u": u[idx],
                  "synergy dependence": (W[idx] * Q[idx]).sum(1) / W[idx].sum(1),
                  "best partner": best, "best Q": np.nanmax(Qm[idx], 1),
                  "worst partner": worst, "worst Q": np.nanmin(Qm[idx], 1),
                  "train decks": (X[tr] > 0).sum(0)[idx]}) \
        .sort("synergy dependence", descending=True).write_csv(out / "card_hubs.csv")
    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "pairs": int(ok.sum()), "models": rows,
           "seconds": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
