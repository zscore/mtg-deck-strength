"""Deck synergy, round 6: is the seen-together synergy of round 5 real? Held-out test and a placebo.

    python scripts/deck_synergy_v6.py MSH

Round 5's seen FM (pairs between nonland cards both seen in a game) gave very repeatable pair values, but was never
scored on held-out games, and repeatability alone could come from deck composition. Here, with the same inputs as
round 5 (context, deck features, affinity, game length, lands / spells seen one-hot, deck power w, seen power u):

  seen, no pairs        reference
  seen pairs, k=4 / 8   pair terms between seen nonland cards; finer l2 grid than round 5 (which collapsed on
                        OTJ and EOE at l2 = 1e-2)
  placebo, no pairs     the same model on *unseen* cards: h = in the deck but not seen this game
  placebo pairs, k=4    pair terms between unseen nonland cards. Synergy that happens on the battlefield predicts no
                        gain here; a deck-composition artefact would show up here too

Each pair model is scored on test games, paired against its own no-pairs model (x1000), and refit on two halves of
the training drafts for split-half pair r (nonland pairs in >= 2,000 training decks).
Writes results/{SET}.PremierDraft/deck_synergy_v6/results.json.
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
L2V = (1e-4, 3e-4, 1e-3)


def main(set_code: str):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v6"
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
        for k in ((4, 8) if tag == "seen" else (4,)):
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
            print(f"  {tag:18s} k={k} l2={l2v:g} vs no pairs {row['vs no pairs'] * 1000:+.3f} ± "
                  f"{row['vs no pairs ±95%'] * 1000:.3f} (x1000)  r {row.get('split-half pair r', float('nan')):.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        del XH
    res = {"set": set_code, "games": int(len(y)), "test games": int(len(te)), "pairs": int(ok.sum()), "models": rows,
           "seconds": time.time() - t0}
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{set_code} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
