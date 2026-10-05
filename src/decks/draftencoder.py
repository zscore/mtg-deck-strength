"""DraftEncoder (arXiv 2607.04782, github.com/akulen/MtGDraftEncoder) in PyTorch, plus a deck stage.

Paper architecture:
  set context   every card in the set is a token (card features -> d), FiLM by player rank,
                then self-attention over the whole set -> set-contextualised card embeddings
  draft layers  at each pick t there are pack-card tokens h[t, j] and one state token s[t]
                (initialised with the picked card). Per layer: each pack card attends causally
                (with RoPE) over the states s[0..t]; then s[t] attends over the pack at t, with
                the picked card as the query.
  head          p(win) after every pick from [s[t], rank embedding]; loss is the binomial
                likelihood of the event record, weighted by (t+1)/T so later picks count more.

The paper encodes cards from oracle text (EmbeddingGemma). There is no oracle text here, so a card
is a learned embedding plus optional meta features (win rates, colours, mana value, rarity).

Deck stage (not in the paper): the deck that was actually registered, lands included, as
(card, copies) tokens plus the final draft state, a few self-attention layers, and a head that
predicts the same event record.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_tensors
from .deckstrength import N_RANKS, Drafts, binomial_nll

SIZES = {
    # the paper's code defaults (its text reports 10 set layers)
    "paper": dict(d=64, heads=2, set_layers=3, draft_layers=5, deck_layers=2),
    "big": dict(d=128, heads=4, set_layers=4, draft_layers=8, deck_layers=3),
}


# ---------------------------------------------------------------- data

def pack_tokens(set_code: str, event: str, draft_ids: np.ndarray, chunk: int = 4096, strict: bool = True):
    """Card indices of every pack seen ([D, T, P], padded with C) and the pick's slot ([D, T]).

    strict=False also returns a [D] mask of drafts whose every pick is in its pack (a few files
    have drafts where it isn't) instead of asserting."""
    t = load_tensors(set_code, event)
    assert not t.p1p1_missing, "sets without P1P1 need a masked first pick; not handled yet"
    # load_tensors and load_drafts sort drafts differently; line them up by draft_id
    row = {x: i for i, x in enumerate(t.draft_id.tolist())}
    missing = np.array([x not in row for x in draft_ids.tolist()])
    assert not strict or not missing.any(), "drafts missing from load_tensors"
    order = np.array([row.get(x, 0) for x in draft_ids.tolist()])
    _, K, N, C = t.pack.shape
    D, T = len(order), K * N
    pick = t.pick[order].reshape(D, T).astype(np.int64)
    P = max(int(t.pack[order[a:a + chunk], 0, 0].sum(1).max()) for a in range(0, D, chunk))
    tokens = np.full((D, T, P), C, dtype=np.int16)
    max_copies = int(t.pack.max())
    for a in range(0, D, chunk):  # index a chunk at a time: a full copy of the pack tensor can be 3GB+
        blk = t.pack[order[a:a + chunk]].reshape(-1, C)
        # one column per (copy k, card c); stable sort brings present cards first, by card index
        present = np.concatenate([blk >= k for k in range(1, max_copies + 1)], axis=1)
        srt = np.argsort(~present, axis=1, kind="stable")[:, :P]
        ok = np.take_along_axis(present, srt, 1)
        tokens[a:a + chunk] = np.where(ok, srt % C, C).reshape(-1, T, P)
    del t
    pick_pos = (tokens == pick[..., None]).argmax(-1)
    ok = (np.take_along_axis(tokens, pick_pos[..., None], -1)[..., 0] == pick).all(1) & ~missing
    if not strict:
        return tokens, pick_pos, ok
    assert ok.all()
    return tokens, pick_pos


def deck_tokens(deck: np.ndarray, max_distinct: int | None = None):
    """[D, C] counts -> distinct card indices [D, M] (padded with C) and copies [D, M]."""
    D, C = deck.shape
    M = max_distinct or int((deck > 0).sum(1).max())
    order = np.argsort(~(deck > 0), axis=1, kind="stable")[:, :M]
    copies = np.take_along_axis(deck, order, 1).astype(np.int64)
    return np.where(copies > 0, order, C).astype(np.int64), copies


# ---------------------------------------------------------------- model

def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on the last dim of x [..., L, dh] at integer positions pos [L]."""
    dh = x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos.float()[:, None] * freqs[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.dropout = heads, dropout
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, xq, xkv, mask=None, pos_q=None, pos_k=None):
        """xq [N, Lq, d], xkv [N, Lk, d]; mask [N or 1, Lq, Lk] True = attend."""
        N, Lq, d = xq.shape
        split = lambda x: x.view(N, -1, self.h, d // self.h).transpose(1, 2)
        q, k, v = split(self.q(xq)), split(self.k(xkv)), split(self.v(xkv))
        if pos_q is not None:
            q, k = rope(q, pos_q), rope(k, pos_k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else mask[:, None],
            dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(N, Lq, d))


def mlp(d: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))


class DraftBlock(nn.Module):
    """One AlternatingResidualAttention layer from the paper."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))
        self.att_h, self.att_s = Attention(d, heads, dropout), Attention(d, heads, dropout)
        self.mlp_h, self.mlp_s = mlp(d, dropout), mlp(d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, s, tok_t, pick_tok):
        """h [B, N, d] pack-card tokens (no padding), s [B, T, d] states,
        tok_t [N] the pick each card token belongs to, pick_tok [B, T] the picked card's token."""
        B, N, d = h.shape
        T = s.shape[1]
        pos = torch.arange(T, device=h.device)
        # pack cards at pick t attend over the draft states up to t (RoPE position t)
        causal = (tok_t[:, None] >= pos[None])[None]
        h = h + self.drop(self.att_h(self.ln[0](h), s, causal, tok_t, pos))
        h = h + self.drop(self.mlp_h(self.ln[1](h)))
        # the state at t attends over the pack at t, queried by the card that was picked
        hn = self.ln[2](h)
        q = hn.gather(1, pick_tok[..., None].expand(B, T, d))
        same_pick = (pos[:, None] == tok_t[None])[None]
        s = s + self.drop(self.att_s(q, hn, same_pick))
        s = s + self.drop(self.mlp_s(self.ln[3](s)))
        return h, s


class DraftEncoder(nn.Module):
    def __init__(self, n_cards: int, meta: np.ndarray | None, d: int = 64, heads: int = 2,
                 set_layers: int = 3, draft_layers: int = 5, deck_layers: int = 2,
                 use_deck: bool = False, dropout: float = 0.1, max_copies: int = 20):
        super().__init__()
        self.C, self.use_deck = n_cards, use_deck
        self.card = nn.Embedding(n_cards, d)
        self.meta_in = None
        if meta is not None:
            self.register_buffer("meta", torch.from_numpy(meta.astype(np.float32)))
            self.meta_in = nn.Linear(meta.shape[1], d)
        self.card_drop = nn.Dropout(dropout)
        self.rank = nn.Embedding(N_RANKS, d)
        self.film = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * d))
        layer = lambda: nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation="gelu",
                                                   batch_first=True, norm_first=True)
        self.set_enc = nn.TransformerEncoder(layer(), set_layers, enable_nested_tensor=False)
        self.set_ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(DraftBlock(d, heads, dropout) for _ in range(draft_layers))
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        if use_deck:
            self.copies = nn.Embedding(max_copies + 1, d)
            self.deck_enc = nn.TransformerEncoder(layer(), deck_layers, enable_nested_tensor=False)
            self.deck_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d), nn.GELU(),
                                           nn.Linear(2 * d, 1))

    def set_context(self, rank_emb):
        """[R, d] rank embeddings -> [R, C + 1, d] contextual card embeddings (last row = pad)."""
        x = self.card.weight
        if self.meta_in is not None:
            x = x + self.meta_in(self.meta)
        x = self.card_drop(F.gelu(x))[None]
        g, b = self.film(rank_emb).chunk(2, -1)
        x = (1 + 0.1 * torch.tanh(g))[:, None] * x + b[:, None]
        x = self.set_ln(self.set_enc(x))
        return torch.cat([x, torch.zeros_like(x[:, :1])], 1)

    def forward(self, packs, pick_pos, rank, deck=None, copies=None):
        """packs [B, T, P], pick_pos [B, T], rank [B] -> (p logits [B, T], deck logit [B] or None)."""
        B, T, P = packs.shape
        # the set context depends only on rank, so encode each rank present once
        ranks, inv = rank.unique(return_inverse=True)
        r_emb = self.rank(ranks)
        cards = self.set_context(r_emb)[inv]  # [B, C+1, d]
        bi = torch.arange(B, device=packs.device)
        # complete drafts all have the same pack sizes, so drop the padding with one fixed layout
        valid = packs[0] != self.C  # [T, P]
        assert bool((valid[None] == (packs != self.C)).all()), "drafts differ in pack layout"
        flat = valid.reshape(-1).nonzero().squeeze(1)  # [N] index into T*P
        tok_t = flat // P
        offset = torch.cumsum(valid.sum(1), 0) - valid.sum(1)  # first token of each pick
        h = cards[bi[:, None], packs.reshape(B, -1)[:, flat]]
        picked = packs.gather(-1, pick_pos[..., None])[..., 0]
        s = cards[bi[:, None], picked]
        pick_tok = offset[None] + pick_pos
        for blk in self.blocks:
            h, s = blk(h, s, tok_t, pick_tok)
        re = r_emb[inv]
        logits = self.head(torch.cat([s, re[:, None].expand(B, T, -1)], -1)).squeeze(-1)
        if not self.use_deck:
            return logits, None
        x = cards[bi[:, None], deck] + self.copies(copies.clamp(max=self.copies.num_embeddings - 1))
        x = torch.cat([s[:, -1:], x], 1)
        pad = torch.cat([torch.zeros_like(deck[:, :1], dtype=torch.bool), deck == self.C], 1)
        z = self.deck_enc(x, src_key_padding_mask=pad)[:, 0]
        return logits, self.deck_head(torch.cat([z, re], -1)).squeeze(-1)


# ---------------------------------------------------------------- training

def binomial_loss(z, w, l):
    return -(w * F.logsigmoid(z) + l * F.logsigmoid(-z))


def train(d: Drafts, packs: np.ndarray, pick_pos: np.ndarray, meta: np.ndarray | None,
          deck: np.ndarray | None = None, has_deck: np.ndarray | None = None, size: str = "big",
          epochs: int = 12, batch: int = 64, lr: float = 3e-4, patience: int | None = None, seed: int = 0,
          device: str | None = None, log=print):
    """Fit on train drafts for a full one-cycle schedule; keep the epoch with the best val NLL of
    the final prediction (deck head if used). Val NLL is noisy, so early stopping is off by default.

    Returns (model, predict) where predict(idx) -> (p per pick [n, T], p deck [n] or None).
    """
    torch.manual_seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    use_deck = deck is not None
    model = DraftEncoder(len(d.names), meta, use_deck=use_deck, **SIZES[size]).to(device)
    log(f"  {size}{'+deck' if use_deck else ''}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    tr, va = np.flatnonzero(d.mask("train")), np.flatnonzero(d.mask("val"))
    steps = epochs * math.ceil(len(tr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    T = packs.shape[1]
    tw = torch.arange(1, T + 1, device=device, dtype=torch.float32) / T
    P = torch.from_numpy(packs.astype(np.int64))
    PP = torch.from_numpy(pick_pos.astype(np.int64))
    R = torch.from_numpy(d.rank)
    W, L = torch.from_numpy(d.wins), torch.from_numpy(d.losses)
    if use_deck:
        DK, CP = (torch.from_numpy(a) for a in deck_tokens(deck))
        HD = torch.from_numpy(has_deck)

    def run(idx):
        args = [P[idx].to(device), PP[idx].to(device), R[idx].to(device)]
        if use_deck:
            args += [DK[idx].to(device), CP[idx].to(device)]
        return model(*args)

    def predict(idx):
        model.eval()
        pk, dk = [], []
        with torch.no_grad():
            for b in np.array_split(idx, max(1, len(idx) // 512)):
                z, zd = run(b)
                pk.append(torch.sigmoid(z).cpu())
                if zd is not None:
                    dk.append(torch.sigmoid(zd).cpu())
        return torch.cat(pk).numpy(), (torch.cat(dk).numpy() if dk else None)

    def val_nll():
        pk, dk = predict(va)
        w, l = d.wins[va], d.losses[va]
        out = {"final_pick": binomial_nll(pk[:, -1], w, l)}
        if dk is not None:
            out["deck"] = binomial_nll(dk[has_deck[va]], w[has_deck[va]], l[has_deck[va]])
        return out

    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in np.array_split(rng.permutation(tr), math.ceil(len(tr) / batch)):
            z, zd = run(b)
            w, l = W[b].to(device), L[b].to(device)
            n = (w + l).sum()
            loss = (binomial_loss(z, w[:, None], l[:, None]) * tw).sum() / (n * tw.sum())
            if use_deck:
                m = HD[b].to(device)
                loss = loss + (binomial_loss(zd, w, l) * m).sum() / ((w + l) * m).sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v = val_nll()
        key = v.get("deck", v["final_pick"])
        log(f"  epoch {ep:2d} train {tot / math.ceil(len(tr) / batch):.5f} "
            + " ".join(f"val_{k} {x:.5f}" for k, x in v.items()) + f"  [{time.time() - t0:.0f}s]")
        if key < best - 1e-5:
            best, bad = key, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, predict
