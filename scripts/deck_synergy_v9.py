"""Deck synergy, round 9: opening-hand synergy from card features, shared across 12 sets.

    python scripts/deck_synergy_v9.py [holdout=SET,SET]

Round 8 found the causally clean synergy (pairs of nonland cards both in the opening hand) real but small and noisy
per set: −0.27 x1000 on top of per-card slopes, split-half r ~0.13. Pairs that both start in hand are rare in one
set's games. Here one pair model reads cards only through their features (rules-text embedding, 17Lands metadata,
Scryfall facts), so all 12 sets' opening hands inform it:

  stage 1 (per set)   round 8's opening-hand model without pairs: context, deck features, affinity, deck power w,
                      opening-hand value u, lands / spells in hand (one-hot); its logit is a fixed offset
  stage 2 (shared)    + sum_{i<j in hand, nonland} <f(z_i), f(z_j)> * scale_set, f = MLP(card features) -> k dims

Scored per set on test games, paired against stage 1 (x1000), next to round 8's per-set free hand pairs.
sets=SOS,BLB restricts the run to those sets (smoke test; writes under smoke/).
holdout=...: those sets are left out of stage 2's training entirely, which is the day-zero question: does
synergy learned on other sets carry to a new set's cards?
Writes results/deck_synergy_v9/{results.json | holdout_<SETS>.json} and, for the full run, <SET>_pairs.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_synergy_v2 import mean_pair, nll, pick_affinity  # noqa: E402
from deck_synergy_v8 import SlopeSeenFM  # noqa: E402
from deck_synergy_v5 import train  # noqa: E402
from mtgdata.cardinfo import card_info, card_matrix, deck_features  # noqa: E402
from mtgdata.data import FEATURES, card_metadata  # noqa: E402
from mtgdata.encode import card_features  # noqa: E402
from mtgdata.games import aligned, flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (covariates, experience, game_split, load_drafts, record_before, record_onehot)  # noqa: E402

SETS = ["BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "MSH", "OTJ", "SOS", "TDM"]
HAND = 7
MIN_CO = 2000


def stage1_cached(s):
    """stage1(s), cached under results/<SET>/deck_synergy_v9/stage1.npz (gitignored)."""
    path = ROOT / "results" / f"{s}.PremierDraft" / "deck_synergy_v9" / "stage1.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: (z[k].tolist() if k == "names" else z[k]) for k in z.files}
    r = stage1(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: (np.array(v, dtype=object) if k == "names" else v) for k, v in r.items()})
    return r


def stage1(s):
    """Per-set opening-hand model without pairs. Returns tokens of nonland hand cards (local ids, -1 pad), the offset
    logit, labels, split, the set's card features, names, nonland mask and training-deck co-occurrence."""
    d = load_drafts(s, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(s, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    hand = aligned(g, "opening_hand_", names) > 0
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    del g
    info = card_info(names, s)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    tr = sp == 0
    Fd = deck_features(X, info)[0]
    Fd = (Fd - Fd[tr].mean(0)) / (Fd[tr].std(0) + 1e-6)
    S = pick_affinity(s, names)[0]
    a = mean_pair(X, S, nonland)
    a = (a - a[tr].mean()) / (a[tr].std() + 1e-6)
    hf = hand.astype(np.float32)
    n_l = np.clip((hf @ (1 - nonland)).astype(int), 0, HAND)
    n_s = np.clip((hf @ nonland).astype(int), 0, HAND)
    cs = np.hstack([ctx, Fd, a[:, None], np.eye(HAND + 1, dtype=np.float32)[n_l],
                    np.eye(HAND + 1, dtype=np.float32)[n_s]]).astype(np.float32)
    C = X.shape[1]
    XH = np.hstack([X.astype(np.float16), hand.astype(np.float16)])
    m, _, _ = train(lambda: SlopeSeenFM(C, cs.shape[1], 0, nonland, 0, 1, slope=False), XH, cs, y, sp, 1e-5, 0)
    with torch.no_grad():
        off = np.concatenate([m(torch.from_numpy(XH[i:i + 65536]), torch.from_numpy(cs[i:i + 65536])).numpy()
                              for i in range(0, len(y), 65536)]).astype(np.float32)
    del XH, cs
    hn = hand & (nonland > 0)
    tok = np.full((len(y), HAND), -1, np.int32)
    r, c = np.nonzero(hn)
    pos = np.arange(len(r)) - np.searchsorted(r, r)  # rank within the row (np.nonzero is row-major)
    keep = pos < HAND
    tok[r[keep], pos[keep]] = c[keep]
    feats = np.hstack([card_features(card_metadata(names, s)), card_matrix(info),
                       np.load(FEATURES / f"{s}.PremierDraft" / "text_emb.npy")]).astype(np.float32)
    Pd = (X[tr] * nonland > 0).astype(np.float32)
    co = Pd.T @ Pd
    return dict(tok=tok, off=off, y=y, sp=sp, feats=feats, names=names, nonland=nonland, co=co)


class SharedPairs(nn.Module):
    def __init__(self, feats, n_sets, k=8, hidden=64, dropout=0.1):
        super().__init__()
        self.register_buffer("feats", torch.as_tensor(feats))
        self.f = nn.Sequential(nn.Linear(feats.shape[1], hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, k))
        nn.init.normal_(self.f[-1].weight, std=0.01)
        nn.init.zeros_(self.f[-1].bias)
        self.scale = nn.Parameter(torch.ones(n_sets))

    def emb(self):
        return torch.cat([self.f(self.feats), torch.zeros(1, self.f[-1].out_features)])  # last row = padding

    def forward(self, tok, off, set_id):
        e = self.emb()[tok]  # tok: global ids, padding -> last row
        pairs = 0.5 * (e.sum(1).pow(2).sum(-1) - e.pow(2).sum((1, 2)))
        return off + self.scale[set_id] * pairs


def main(*kv):
    opts = dict(x.split("=", 1) for x in kv)
    holdout = [s for s in opts.get("holdout", "").split(",") if s]
    global SETS
    if "sets" in opts:  # smoke tests
        SETS = opts["sets"].split(",")
    out = ROOT / "results" / "deck_synergy_v9" / ("smoke" if "sets" in opts else "")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    t0 = time.time()
    data, base = {}, 0
    for s in SETS:
        r = stage1_cached(s)
        r["base"] = base
        base += len(r["names"])
        data[s] = r
        print(f"  {s}: stage 1 done, {len(r['y'])} games [{time.time() - t0:.0f}s]", flush=True)
    feats = np.vstack([data[s]["feats"] for s in SETS])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    pad = len(feats)
    cat = lambda key, fn=lambda r, x: x: np.concatenate([fn(data[s], data[s][key]) for s in SETS])  # noqa: E731
    tok = cat("tok", lambda r, x: np.where(x >= 0, x + r["base"], pad)).astype(np.int64)
    off, y, sp = cat("off"), cat("y").astype(np.float32), cat("sp")
    sid = np.concatenate([np.full(len(data[s]["y"]), i) for i, s in enumerate(SETS)])
    fit_mask = ~np.isin(sid, [SETS.index(s) for s in holdout])
    tr, va = np.flatnonzero((sp == 0) & fit_mask), np.flatnonzero((sp == 1) & fit_mask)
    T = {k: torch.from_numpy(v) for k, v in dict(tok=tok, off=off, y=y, sid=sid).items()}

    model = SharedPairs(feats.astype(np.float32), len(SETS))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    def logits(idx):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(T["tok"][b], T["off"][b], T["sid"][b])
                              for b in np.array_split(idx, max(1, len(idx) // 65536))]).numpy()

    def val_loss():
        z = logits(va)
        return float(np.mean([nll(y[va][sid[va] == i], 1 / (1 + np.exp(-z[sid[va] == i]))).mean()
                              for i in np.unique(sid[va])]))

    rng = np.random.default_rng(0)
    best, state, bad, lr_drops = val_loss(), None, 0, 2
    print(f"  stage 2: {len(tr)} training games, {pad} cards, val start {best:.5f}", flush=True)
    for ep in range(30):
        model.train()
        for b in np.array_split(rng.permutation(tr), len(tr) // 8192):
            loss = F.binary_cross_entropy_with_logits(model(T["tok"][b], T["off"][b], T["sid"][b]), T["y"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_loss()
        print(f"    epoch {ep} val {v:.5f} [{time.time() - t0:.0f}s]", flush=True)
        if v < best - 1e-6:
            best, bad, state = v, 0, {n: t.detach().clone() for n, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                if lr_drops == 0 or state is None:
                    break
                lr_drops, bad = lr_drops - 1, 0
                model.load_state_dict(state)
                for gr in opt.param_groups:
                    gr["lr"] /= 3
    if state is not None:
        model.load_state_dict(state)
    if holdout:  # a held-out set's scale never trained: use the trained sets' mean
        ho = [SETS.index(s) for s in holdout]
        with torch.no_grad():
            trained = [i for i in range(len(SETS)) if i not in ho]
            model.scale[ho] = model.scale[trained].mean()

    with torch.no_grad():
        E = model.emb()[:-1].numpy()
    rows = []
    for i, s in enumerate(SETS):
        te = np.flatnonzero((sp == 2) & (sid == i))
        p = 1 / (1 + np.exp(-logits(te)))
        p0 = 1 / (1 + np.exp(-off[te]))
        dl = nll(y[te], p) - nll(y[te], p0)
        r = data[s]
        Es = E[r["base"]:r["base"] + len(r["names"])]
        Q = (Es @ Es.T) * float(model.scale[i].detach())
        iu = np.triu_indices(len(Q), 1)
        ok = (r["co"][iu] >= MIN_CO) & (r["nonland"][iu[0]] > 0) & (r["nonland"][iu[1]] > 0)
        row = {"set": s, "held out": s in holdout, "vs stage 1": float(dl.mean()),
               "vs stage 1 ±95%": float(1.96 * dl.std() / np.sqrt(len(dl))), "set scale": float(model.scale[i])}
        v8 = ROOT / "results" / f"{s}.PremierDraft" / "deck_synergy_v8"
        if v8.exists():
            per_set = {m["model"]: m for m in json.loads((v8 / "results.json").read_text())["models"] if m["input"] == "hand"}
            row["round 8 per-set hand pairs vs no pairs"] = per_set["pairs k=4"]["vs no pairs"]
            hp = pl.read_csv(v8 / "hand_pairs.csv")
            idx = {n: j for j, n in enumerate(r["names"])}
            q8 = hp["Q"].to_numpy()
            qs = Q[[idx[a] for a in hp["card a"]], [idx[b] for b in hp["card b"]]]
            row["r with round 8 per-set hand pairs"] = float(np.corrcoef(qs, q8)[0, 1])
        rows.append(row)
        print(f"  {s}{' (held out)' if s in holdout else ''}: vs stage 1 {row['vs stage 1'] * 1000:+.3f} ± "
              f"{row['vs stage 1 ±95%'] * 1000:.3f} x1000  (round 8 per-set "
              f"{row.get('round 8 per-set hand pairs vs no pairs', float('nan')) * 1000:+.3f})  "
              f"r with round 8 pairs {row.get('r with round 8 per-set hand pairs', float('nan')):.3f}", flush=True)
        if not holdout:
            i_, j_ = iu[0][ok], iu[1][ok]
            pl.DataFrame({"card a": [r["names"][t] for t in i_], "card b": [r["names"][t] for t in j_],
                          "shared Q": Q[i_, j_], "train decks": r["co"][i_, j_]}) \
                .sort("shared Q", descending=True).write_csv(out / f"{s}_pairs.csv")
    res = {"holdout": holdout, "val": best, "sets": rows, "seconds": time.time() - t0}
    name = "results.json" if not holdout else f"holdout_{'_'.join(holdout)}.json"
    (out / name).write_text(json.dumps(res, indent=1))
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
