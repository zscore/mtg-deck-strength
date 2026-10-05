"""A deck network shared across sets, boosted on each set's own linear card-power model.

    python scripts/deck_multiset.py [per_set_train_games] [key=value ...]

One set's games (0.4-1.2M) are too few for a deck transformer to beat card power (deck_synergy
notebook, section 10). Here the transformer is trained on 12 sets at once. It can't memorise cards:
a card token is only an MLP of the card's features (colour/rarity/mana value/types, Scryfall rules
features, frozen rules-text embedding; the same 465 columns the day-zero pick model uses) plus its
copies, so what it learns has to be general deck structure: curve, colours and mana, creature/removal
balance, cards that work together by what they do.

Stage 1 (per set): the linear game model, card counts + context (+ deck features with deck_feats=1),
fit on that set's training games, exactly as deck_synergy_v2.py. Its logit is a fixed offset.
Stage 2 (shared): logit = offset + g(deck), g a transformer over the deck's distinct cards with a CLS
readout and a per-set scale, trained on all sets' training games (up to per_set_train_games each),
early-stopped on all sets' validation games. Test: every test game of each set, paired against stage 1.
Writes results/deck_multiset/{name}.json.
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
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from openlane.cardinfo import card_info, card_matrix, deck_features  # noqa: E402
from openlane.data import ROOT, card_metadata  # noqa: E402
from openlane.deckstrength import (aligned, covariates, experience, flag, game_split, load_drafts,  # noqa: E402
                                   record_before, record_onehot)
from openlane.games import game_path  # noqa: E402
from openlane.nn import card_features  # noqa: E402
from openlane.synergy import fit  # noqa: E402

SETS = ["MSH", "BLB", "DFT", "DSK", "FDN", "FIN", "LCI", "MKM", "OTJ", "EOE", "SOS", "TDM"]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
M = 40  # max distinct cards per deck token list


def tokens(X):
    """[N, C] counts -> [N, M] card indices (padded with -1) and copies."""
    order = np.argsort(~(X > 0), axis=1, kind="stable")[:, :M]
    cp = np.take_along_axis(X, order, 1)
    return np.where(cp > 0, order, -1).astype(np.int32), cp.astype(np.int8)


def nll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_set(s, deck_feats, per_set, rng):
    d = load_drafts(s, "PremierDraft")
    names = d.names
    g = pl.read_parquet(game_path(s, "PremierDraft"))
    sp = game_split(d, g["draft_id"])
    g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
    X = aligned(g, "deck_", names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    del g
    info = card_info(names, s)
    if deck_feats:
        Fd, _ = deck_features(X, info)
        mu, sd = Fd[sp == 0].mean(0), Fd[sp == 0].std(0) + 1e-6
        ctx = np.hstack([ctx, (Fd - mu) / sd]).astype(np.float32)
    m, v, predict = fit(X, ctx, y, sp, 0, 1e-4, 0, log=lambda _: None)
    with torch.no_grad():
        off = (m(torch.from_numpy(X), torch.from_numpy(ctx))).numpy()
    feats = np.hstack([card_features(card_metadata(names, s)), card_matrix(info),
                       np.load(ROOT / "data" / "features" / f"{s}.PremierDraft" / "text_emb.npy")]).astype(np.float32)
    keep = {0: np.flatnonzero(sp == 0), 1: np.flatnonzero(sp == 1), 2: np.flatnonzero(sp == 2)}
    if per_set and len(keep[0]) > per_set:
        keep[0] = np.sort(rng.choice(keep[0], per_set, replace=False))
    if len(keep[1]) > per_set // 5:
        keep[1] = np.sort(rng.choice(keep[1], per_set // 5, replace=False))
    parts = {}
    for k, ix in keep.items():
        tok, cp = tokens(X[ix])
        parts[k] = dict(tok=tok, cp=cp, y=y[ix], off=off[ix].astype(np.float32))
    return names, feats, parts, v


class SharedDeckNet(nn.Module):
    def __init__(self, feats, n_sets, d=64, heads=4, layers=2, dropout=0.1, max_copies=20):
        super().__init__()
        f = torch.as_tensor(feats)
        self.register_buffer("feats", f)
        self.card = nn.Sequential(nn.Linear(f.shape[1], 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.copies = nn.Embedding(max_copies + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        self.set_scale = nn.Parameter(torch.ones(n_sets))
        self.set_bias = nn.Parameter(torch.zeros(n_sets))
        self.max_copies = max_copies

    def forward(self, tok, cp, set_id):
        pad = tok < 0
        emb = self.card(self.feats)
        h = emb[tok.clamp(min=0)] + self.copies(cp.long().clamp(max=self.max_copies))
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        g = self.head(self.enc(h, src_key_padding_mask=pad)[:, 0]).squeeze(-1)
        return self.set_scale[set_id] * g + self.set_bias[set_id]


def main(per_set="200000", *kv):
    per_set = int(per_set)
    cfg = dict(a.split("=") for a in kv)
    deck_feats = int(cfg.pop("deck_feats", 1))
    epochs, lr, batch = int(cfg.pop("epochs", 8)), float(cfg.pop("lr", 5e-4)), int(cfg.pop("batch", 1024))
    sets = cfg.pop("sets", ",".join(SETS)).split(",")
    name = cfg.pop("name", f"shared{'_df' if deck_feats else ''}")
    net_kw = {k: (float(v) if "." in v else int(v)) for k, v in cfg.items()}
    rng = np.random.default_rng(0)
    torch.set_num_threads(6)
    t0 = time.time()
    data, feats, offs = [], [], [0]
    for i, s in enumerate(sets):
        names, f, parts, v = load_set(s, deck_feats, per_set, rng)
        for p in parts.values():
            p["tok"] = np.where(p["tok"] >= 0, p["tok"] + offs[-1], -1)
            p["set"] = np.full(len(p["y"]), i, np.int64)
        data.append(parts)
        feats.append(f)
        offs.append(offs[-1] + len(names))
        print(f"  {s}: {len(names)} cards, train {len(parts[0]['y'])} val {len(parts[1]['y'])} "
              f"test {len(parts[2]['y'])}, stage-1 val {v:.5f} [{time.time() - t0:.0f}s]", flush=True)
    Fm = np.vstack(feats)
    Fm = ((Fm - Fm.mean(0)) / (Fm.std(0) + 1e-3)).astype(np.float32)
    cat = lambda k: {f: torch.from_numpy(np.concatenate([p[k][f] for p in data])) for f in data[0][k]}  # noqa: E731
    tr, va = cat(0), cat(1)
    model = SharedDeckNet(Fm, len(sets), **net_kw).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(tr["y"])
    steps = epochs * (n // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    print(f"{name}: {n} train games over {len(sets)} sets, params {sum(p.numel() for p in model.parameters()):,} "
          f"cfg={net_kw} deck_feats={deck_feats} [{time.time() - t0:.0f}s]", flush=True)

    def logits(part, idx=None):
        model.eval()
        out = []
        with torch.no_grad():
            N = len(part["y"])
            for a in range(0, N, 8192):
                s = slice(a, a + 8192)
                out.append(model(part["tok"][s].to(DEV), part["cp"][s].to(DEV), part["set"][s].to(DEV)).cpu())
        return torch.cat(out).numpy() + part["off"].numpy()

    def val_loss():
        z = logits(va)
        y = va["y"].numpy()
        return float(np.mean([log_loss(y[va["set"].numpy() == i], 1 / (1 + np.exp(-z[va["set"].numpy() == i])))
                              for i in range(len(sets))]))

    base_val = float(np.mean([log_loss(p[1]["y"], 1 / (1 + np.exp(-p[1]["off"]))) for p in data]))
    best, state, hist = val_loss(), {k: x.detach().clone() for k, x in model.state_dict().items()}, []
    print(f"  stage-1 val (mean over sets) {base_val:.5f}; start {best:.5f}", flush=True)
    for ep in range(epochs):
        model.train()
        t1 = time.time()
        perm = torch.randperm(n)
        for a in range(0, n - batch + 1, batch):
            b = perm[a:a + batch]
            z = model(tr["tok"][b].to(DEV), tr["cp"][b].to(DEV), tr["set"][b].to(DEV)) + tr["off"][b].to(DEV)
            loss = F.binary_cross_entropy_with_logits(z, tr["y"][b].float().to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        v = val_loss()
        hist.append(v)
        print(f"  epoch {ep}: val {v:.5f} ({v - base_val:+.5f} vs stage 1) [{time.time() - t1:.0f}s]", flush=True)
        if v < best:
            best, state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(state)  # the zero-initialised head (= stage 1) if no epoch beat it
    res = {"name": name, "sets": sets, "per_set": per_set, "deck_feats": deck_feats, "cfg": net_kw,
           "val_history": hist, "stage1_val": base_val, "best_val": best, "per_set_test": {}}
    tot = []
    for i, (s, p) in enumerate(zip(sets, data)):
        te = {f: torch.from_numpy(x) for f, x in p[2].items()}
        z = logits(te)
        y = p[2]["y"]
        dl = nll(y, 1 / (1 + np.exp(-z))) - nll(y, 1 / (1 + np.exp(-p[2]["off"])))
        res["per_set_test"][s] = {"stage1 logloss": log_loss(y, 1 / (1 + np.exp(-p[2]["off"]))),
                                  "shared logloss": log_loss(y, 1 / (1 + np.exp(-z))),
                                  "diff": float(dl.mean()), "±95%": float(1.96 * dl.std() / np.sqrt(len(dl))),
                                  "set scale": float(model.set_scale[i].detach())}
        tot.append(dl.mean())
        r = res["per_set_test"][s]
        print(f"  {s}: stage1 {r['stage1 logloss']:.5f} shared {r['shared logloss']:.5f} "
              f"diff {r['diff'] * 1000:+.2f} ± {r['±95%'] * 1000:.2f} (x1000)", flush=True)
    res["mean diff"] = float(np.mean(tot))
    res["sets better"] = int(sum(t < 0 for t in tot))
    res["seconds"] = time.time() - t0
    out = ROOT / "results" / "deck_multiset"
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state": model.state_dict(), "sets": sets, "cfg": net_kw}, out / "models" / f"{name}.pt")
    print(f"{name}: mean diff {res['mean diff'] * 1000:+.3f} x1000, better on {res['sets better']}/{len(sets)} "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
