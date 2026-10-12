"""DraftEncoder pooled over sets, from card text (the paper's full setup, arXiv 2607.04782).

Cards are never ids: each card is its EmbeddingGemma vector of the paper's tagged oracle string
(`scripts/card_text_embeddings_paper.py`), optionally concatenated with META win-rate features, so
one model reads every set and can score a set it never saw. A batch holds drafts from one set; the
set encoder runs over that set's cards.

Models (all output a win-rate logit after every pick, [B, T], trained with the same weighted
binomial loss):
  draftencoder  the paper's model: card features -> linear + LeakyReLU + dropout -> FiLM(rank)
                -> N_E set layers; N_D draft layers (DraftBlock from draftencoder.py); 3-layer head
  lr            the paper's LR baseline: one head per pick number i over the *ordered* features of
                picks 1..i (a weight per pick slot), plus rank
  bag           order-free linear: one head per pick number over the *sum* of picked-card features
  bag_id        order-free linear on card identity (one weight per card per head); same-set only

META (after Bertram et al.; our game files have no sideboard/tutored columns): opening-hand,
drawn, in-deck, in-hand (GIH) and deck-not-seen win rates, plus log play rate, from that set's
games in a given time window, shrunk to the mean and z-scored within the set.
"""
from __future__ import annotations

import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from mtgdata.data import FEATURES as FEAT
from mtgdata.games import aligned, flag, game_path, shrunk_rate

from .deckstrength import N_RANKS, load_drafts
from .draftencoder import DraftBlock, pack_tokens

META_COLS = ("oh_wr", "gd_wr", "gp_wr", "gih_wr", "gns_wr", "log_play")


# ---------------------------------------------------------------- data

class SetData:
    """One set's complete drafts (time-sorted) with pack tokens, plus its card text features."""

    def __init__(self, set_code: str):
        self.set = set_code
        cache = FEAT / f"{set_code}.PremierDraft" / "de_multiset.npz"
        if not cache.exists():
            build_cache(set_code, cache)
        z = np.load(cache, allow_pickle=True)
        self.names = list(z["names"])
        self.draft_id, self.t = z["draft_id"], z["t"]
        self.rank, self.wins, self.losses = z["rank"], z["wins"], z["losses"]
        self.packs, self.pick_pos, self.pool = z["packs"], z["pick_pos"], z["pool"]
        self.C, self.T = len(self.names), self.packs.shape[1]
        self.text = np.load(FEAT / f"{set_code}.PremierDraft" / "text_emb_gemma.npy")
        assert len(self.text) == self.C

    def window(self, lo_days: float, hi_days: float) -> np.ndarray:
        """Indices of drafts in (end - lo_days, end - hi_days], counting back from the last draft."""
        end = self.t.max()
        day = np.timedelta64(86400, "s")
        return np.flatnonzero((self.t > end - lo_days * day) & (self.t <= end - hi_days * day))

    def first_days(self, days: float) -> np.ndarray:
        # MOM-era files start with a few stray pre-release drafts; count from the 1st percentile
        t0 = np.percentile(self.t.astype(np.int64), 1).astype("datetime64[s]")
        return np.flatnonzero(self.t < t0 + np.timedelta64(int(days * 86400), "s"))

    def picked(self) -> np.ndarray:
        """[D, T] card index taken at each pick."""
        return np.take_along_axis(self.packs, self.pick_pos[..., None].astype(np.int64), -1)[..., 0]


def build_cache(set_code: str, path):
    t0 = time.time()
    d = load_drafts(set_code)
    tok = FEAT / f"{set_code}.PremierDraft" / "draftencoder_tokens.npz"
    if tok.exists():
        z = np.load(tok)
        packs, pick_pos = z["packs"], z["pick_pos"]
        good = np.ones(len(packs), dtype=bool)
    else:
        packs, pick_pos, good = pack_tokens(set_code, "PremierDraft", d.draft_id, strict=False)
    # the model drops padding with one fixed layout per batch; keep drafts with the set's usual layout
    layout = packs != len(d.names)
    sizes = layout.sum(-1)
    modal = np.array([np.bincount(sizes[:, t]).argmax() for t in range(sizes.shape[1])])
    ok = good & (sizes == modal).all(1) & (layout[:, :, : modal.max()].sum(-1) == modal).all(1)
    P = int(modal.max())
    t = pl.Series(d.draft_time).str.to_datetime(strict=False).cast(pl.Datetime("ms")).to_numpy()
    t = t.astype("datetime64[s]")
    np.savez(path, names=np.array(d.names, dtype=object), draft_id=d.draft_id[ok], t=t[ok],
             rank=d.rank[ok], wins=d.wins[ok], losses=d.losses[ok],
             packs=packs[ok][:, :, :P].astype(np.int16), pick_pos=pick_pos[ok].astype(np.int8),
             pool=d.pool[ok])
    print(f"  cached {set_code}: {ok.sum()} / {len(ok)} drafts, pack layout {modal.tolist()[:3]}... "
          f"[{time.time() - t0:.0f}s]", flush=True)


def card_meta(s: SetData, draft_idx: np.ndarray) -> np.ndarray:
    """[C, 6] META features from the games of the given drafts, z-scored within the set."""
    ids = pl.Series(s.draft_id[draft_idx]).implode()
    cols = pl.scan_parquet(game_path(s.set)).collect_schema().names()
    want = ["draft_id", "won"] + [c for c in cols if c.startswith(("deck_", "opening_hand_", "drawn_"))]
    g = pl.scan_parquet(game_path(s.set)).select(want).filter(pl.col("draft_id").is_in(ids)).collect()
    won = flag(g["won"]).astype(np.float64)
    deck = aligned(g, "deck_", s.names) > 0
    oh = aligned(g, "opening_hand_", s.names) > 0
    gd = aligned(g, "drawn_", s.names) > 0
    ih = oh | gd
    base = won.mean()
    rate = lambda m: shrunk_rate(won @ m, m.sum(0), base)
    out = np.column_stack([rate(oh), rate(gd), rate(deck), rate(ih), rate(deck & ~ih),
                           np.log1p(deck.sum(0) / max(len(g), 1) * 1000)])
    return ((out - out.mean(0)) / (out.std(0) + 1e-9)).astype(np.float32), out


# ---------------------------------------------------------------- models

def _tokens(packs, pad):
    """Flatten the (fixed) pack layout of a single-set batch: card ids [B, N], pick of each token [N],
    first token of each pick [T]."""
    B, T, P = packs.shape
    valid = packs[0] != pad
    flat = valid.reshape(-1).nonzero().squeeze(1)
    tok_t = flat // P
    offset = torch.cumsum(valid.sum(1), 0) - valid.sum(1)
    return packs.reshape(B, -1)[:, flat], tok_t, offset


class DraftEncoderMS(nn.Module):
    def __init__(self, feat_dim: int, d: int = 64, heads: int = 2, set_layers: int = 10,
                 draft_layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.card_in = nn.Linear(feat_dim, d)
        self.card_drop = nn.Dropout(dropout)
        self.rank = nn.Embedding(N_RANKS, d)
        self.film = nn.Sequential(nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, 2 * d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.set_enc = nn.TransformerEncoder(layer, set_layers, enable_nested_tensor=False)
        self.set_ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(DraftBlock(d, heads, dropout) for _ in range(draft_layers))
        self.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.LeakyReLU(), nn.Linear(2 * d, 2 * d),
                                  nn.LeakyReLU(), nn.Linear(2 * d, 1))

    def forward(self, feats, packs, pick_pos, rank):
        """feats [C, F] for this batch's set, packs [B, T, P] (pad = C), pick_pos [B, T], rank [B]."""
        B, T, P = packs.shape
        C = feats.shape[0]
        ranks, inv = rank.unique(return_inverse=True)
        r_emb = self.rank(ranks)
        x = self.card_drop(F.leaky_relu(self.card_in(feats)))[None]
        g, b = self.film(r_emb).chunk(2, -1)
        x = (1 + 0.1 * torch.tanh(g))[:, None] * x + b[:, None]
        x = self.set_ln(self.set_enc(x))
        cards = torch.cat([x, torch.zeros_like(x[:, :1])], 1)[inv]  # [B, C+1, d]
        ids, tok_t, offset = _tokens(packs, C)
        bi = torch.arange(B, device=packs.device)[:, None]
        h = cards[bi, ids]
        s = cards[bi, packs.gather(-1, pick_pos[..., None])[..., 0]]
        pick_tok = offset[None] + pick_pos
        for blk in self.blocks:
            h, s = blk(h, s, tok_t, pick_tok)
        re = r_emb[inv][:, None].expand(B, T, -1)
        return self.head(torch.cat([s, re], -1)).squeeze(-1)


class LinearMS(nn.Module):
    """kind='lr': paper's order-dependent LR; 'bag': order-free sum of features; 'bag_id': card ids."""

    def __init__(self, kind: str, feat_dim: int, T: int = 45, n_ids: int = 0):
        super().__init__()
        self.kind = kind
        self.bias = nn.Parameter(torch.zeros(T, N_RANKS))
        if kind == "lr":
            self.W = nn.Parameter(torch.zeros(T, T, feat_dim))
            self.register_buffer("tri", torch.tril(torch.ones(T, T)))  # head i sees slots t <= i
        elif kind == "bag":
            self.W = nn.Parameter(torch.zeros(T, feat_dim))
        else:
            self.W = nn.Parameter(torch.zeros(n_ids, T))

    def forward(self, feats, packs, pick_pos, rank, id_offset: int = 0):
        picked = packs.gather(-1, pick_pos[..., None])[..., 0]  # [B, T]
        T = picked.shape[1]
        if self.kind == "bag_id":
            w = self.W[picked + id_offset, :T]  # [B, t, i]
            z = (w * torch.tril(torch.ones(T, T, device=w.device)).T[None]).sum(1)
        else:
            X = feats[picked]  # [B, T, F]
            if self.kind == "lr":
                z = torch.einsum("btf,itf->bi", X, self.W[:T, :T] * self.tri[:T, :T, None])
            else:
                # mean, not sum, of the picks so far: the same model per head (fixed count), better conditioned
                n = torch.arange(1, T + 1, device=X.device, dtype=X.dtype)[None, :, None]
                z = torch.einsum("bif,if->bi", X.cumsum(1) / n, self.W[:T])
        return z + self.bias[:T, rank].T


# ---------------------------------------------------------------- training

def binomial_loss(z, w, l):
    return -(w * F.logsigmoid(z) + l * F.logsigmoid(-z))


class Task:
    """Drafts from one set used in training or evaluation: the set, its feature table and indices."""

    def __init__(self, s: SetData, feats: np.ndarray, idx: np.ndarray, id_offset: int = 0):
        self.s, self.idx, self.id_offset = s, idx, id_offset
        self.feats = torch.from_numpy(feats.astype(np.float32))


def _batches(tasks, batch, rng):
    out = []
    for k, t in enumerate(tasks):
        idx = rng.permutation(t.idx) if rng is not None else t.idx
        out += [(k, b) for b in np.array_split(idx, max(1, math.ceil(len(idx) / batch)))]
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def _run(model, task, b, device):
    s = task.s
    args = (task.feats.to(device), torch.from_numpy(s.packs[b].astype(np.int64)).to(device),
            torch.from_numpy(s.pick_pos[b].astype(np.int64)).to(device), torch.from_numpy(s.rank[b]).to(device))
    if isinstance(model, LinearMS):
        return model(*args, id_offset=task.id_offset)
    return model(*args)


def predict(model, tasks, device, batch=512):
    """List of [n, T] win-rate predictions, one per task."""
    model.eval()
    out = []
    with torch.no_grad():
        for t in tasks:
            ps = [torch.sigmoid(_run(model, t, b, device)).cpu() for _, b in _batches([t], batch, None)]
            out.append(torch.cat(ps).numpy())
    return out


def weighted_nll(p, w, l):
    """The training objective on predictions p [n, T]: per-pick binomial NLL weighted by (t+1)/T."""
    T = p.shape[1]
    tw = np.arange(1, T + 1) / T
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(w[:, None] * np.log(p) + l[:, None] * np.log1p(-p))
    return float((ll * tw).sum() / ((w + l).sum() * tw.sum()))


def fit(model, train_tasks, val_tasks, epochs=15, batch=128, lr=3e-4, wd=0.05, seed=0, device=None,
        log=print, max_minutes=None):
    """One-cycle AdamW; keep the epoch with the best validation objective (pooled over val tasks)."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    n_steps = len(_batches(train_tasks, batch, None))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * n_steps, pct_start=0.05)
    best, state, hist, t_start = np.inf, None, [], time.time()
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for k, b in _batches(train_tasks, batch, rng):
            task = train_tasks[k]
            z = _run(model, task, b, device)
            T = z.shape[1]
            tw = torch.arange(1, T + 1, device=device, dtype=torch.float32) / T
            w = torch.from_numpy(task.s.wins[b]).to(device)
            l = torch.from_numpy(task.s.losses[b]).to(device)
            loss = (binomial_loss(z, w[:, None], l[:, None]) * tw).sum() / ((w + l).sum() * tw.sum())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        ps = predict(model, val_tasks, device)
        v = float(np.mean([weighted_nll(p, t.s.wins[t.idx], t.s.losses[t.idx]) for p, t in zip(ps, val_tasks)]))
        sp = float(np.mean([spearman(p[:, -1], t.s, t.idx) for p, t in zip(ps, val_tasks)]))
        hist.append({"epoch": ep, "train": tot / n_steps, "val": v, "val_spearman": sp})
        log(f"    epoch {ep:2d} train {tot / n_steps:.5f} val {v:.5f} val_spearman {sp:.4f} [{time.time() - t0:.0f}s]")
        if v < best:
            best, state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
        if max_minutes and (time.time() - t_start) / 60 > max_minutes:
            log(f"    stopping: {max_minutes} min budget")
            break
    model.load_state_dict(state)
    return model, hist


def spearman(p, s: SetData, idx) -> float:
    w, l = s.wins[idx], s.losses[idx]
    ok = (w + l) > 0
    return float(spearmanr(p[ok], w[ok] / (w[ok] + l[ok])).statistic)


def heuristic(s: SetData, idx, value: np.ndarray, k: int = 23) -> np.ndarray:
    """[n, T] mean `value` of the best min(k, t+1) cards picked so far (the paper's HEURISTIC)."""
    V = value[s.picked()[idx]]
    out = np.empty(V.shape)
    for t in range(V.shape[1]):
        top = -np.sort(-V[:, : t + 1], axis=1)[:, :k]
        out[:, t] = top.mean(1)
    return out
