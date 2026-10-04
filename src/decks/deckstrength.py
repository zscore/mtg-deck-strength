"""Deck strength: how good is a drafted pool, and what is each card worth?

Target (following DraftEncoder, arXiv 2607.04782): the event record W-L of the drafted deck.
Premier Draft stops at 7 wins or 3 losses, but the stopping rule depends only on the results so
far, so the likelihood of a win probability p is just p^W (1-p)^L; this also lets abandoned
runs count. The paper's headline metric is Spearman(predicted p, W / (W + L)) on held-out drafts.

Card values come from three sources:
  gp_wr / gih_wr  17Lands-style win rate when the card is in the deck / drawn (game data)
  cast_bt         a Bradley-Terry fit on replay data: P(user wins) = sigma(sum over cards of
                  v_c * (user casts of c - opponent casts of c) + covariates). Cards cast is the
                  one view of both decks that is recorded the same way, so only those are compared.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from .data import RANK_ORDER, download

SPLITS = ("train", "val", "test")
N_RANKS = len(RANK_ORDER) + 1  # + unknown


def flag(s: pl.Series) -> np.ndarray:
    """"True"/"False" strings (as stored in game/replay files) or booleans -> bool array."""
    return (s.cast(pl.Utf8) == "True").to_numpy() if s.dtype != pl.Boolean else s.to_numpy()


def rank_index(s: pl.Series) -> np.ndarray:
    r = s.replace_strict(RANK_ORDER, default=None).fill_null(-1).to_numpy()
    return (r + 1).astype(np.int64)  # 0 = unknown


@dataclass
class Drafts:
    """One row per complete draft: the final pool and the event record."""

    names: list[str]
    draft_id: np.ndarray
    draft_time: np.ndarray
    pool: np.ndarray  # [D, C] int8 copies
    wins: np.ndarray
    losses: np.ndarray
    rank: np.ndarray
    skill: np.ndarray  # user_game_win_rate_bucket, NaN if unknown
    split: np.ndarray  # 0/1/2 = train/val/test by draft_time

    def mask(self, split: str) -> np.ndarray:
        return self.split == SPLITS.index(split)


def load_drafts(set_code: str, event: str = "PremierDraft", frac=(0.70, 0.85)) -> Drafts:
    lf = pl.scan_parquet(download(set_code, event))
    cols = lf.collect_schema().names()
    names = [c[len("pack_card_"):] for c in cols if c.startswith("pack_card_")]
    n = lf.group_by("draft_id").len()
    full = n.filter(pl.col("len") == pl.col("len").max())
    last = (lf.join(full, on="draft_id")
            .filter((pl.col("pack_number") == pl.col("pack_number").max())
                    & (pl.col("pick_number") == pl.col("pick_number").max()))
            .select("draft_id", "draft_time", "rank", "event_match_wins", "event_match_losses",
                    "user_game_win_rate_bucket", "pick", *[f"pool_{n}" for n in names])
            .sort("draft_time", "draft_id").collect())
    pool = last.select([f"pool_{n}" for n in names]).to_numpy().astype(np.int8)
    idx = {n: i for i, n in enumerate(names)}
    pool[np.arange(last.height), [idx[p] for p in last["pick"].to_list()]] += 1
    D = last.height
    split = np.zeros(D, dtype=np.int8)
    split[int(frac[0] * D):] = 1
    split[int(frac[1] * D):] = 2
    return Drafts(names=names, draft_id=last["draft_id"].to_numpy(),
                  draft_time=last["draft_time"].to_numpy(), pool=pool,
                  wins=last["event_match_wins"].to_numpy().astype(np.float32),
                  losses=last["event_match_losses"].to_numpy().astype(np.float32),
                  rank=rank_index(last["rank"]),
                  skill=last["user_game_win_rate_bucket"].cast(pl.Float32, strict=False).to_numpy(),
                  split=split)


def game_split(d: Drafts, draft_ids: pl.Series) -> np.ndarray:
    """Games inherit their draft's split; games from drafts not in the draft table get -1."""
    m = dict(zip(d.draft_id.tolist(), d.split.tolist()))
    return np.array([m.get(x, -1) for x in draft_ids.to_list()], dtype=np.int8)


def aligned(df: pl.DataFrame, prefix: str, names: list[str]) -> np.ndarray:
    """[rows, C] int8 matrix of `prefix + name` columns, zeros for names the file lacks."""
    out = np.zeros((df.height, len(names)), dtype=np.int8)
    for j, n in enumerate(names):
        c = prefix + n
        if c in df.columns:
            out[:, j] = df[c].fill_null(0).to_numpy()
    return out


def most_played_deck(d: Drafts, games: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """[D, C] deck counts (lands included) of each draft's most-played build, and a has-deck mask."""
    builds = (games.with_row_index("i")
              .group_by("draft_id", "build_index").agg(pl.len().alias("n"), pl.col("i").first())
              .sort("n", "build_index", descending=[True, False])
              .unique("draft_id", keep="first"))
    row = dict(zip(builds["draft_id"].to_list(), builds["i"].to_list()))
    has = np.array([x in row for x in d.draft_id])
    deck = np.zeros_like(d.pool)
    idx = [row[x] for x in d.draft_id[has]]
    deck[has] = aligned(games[idx], "deck_", d.names)
    return deck, has


# ---------------------------------------------------------------- card values

def shrunk_rate(wins: np.ndarray, n: np.ndarray, prior: float, m: float = 200.0) -> np.ndarray:
    return (wins + m * prior) / (n + m)


def game_win_rates(games: pl.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    """GP WR (card in deck) and GIH WR (card in opening hand or drawn), shrunk to the mean."""
    won = flag(games["won"]).astype(np.float64)
    deck = aligned(games, "deck_", names) > 0
    hand = (aligned(games, "opening_hand_", names) + aligned(games, "drawn_", names)) > 0
    base = won.mean()
    return {"gp_wr": shrunk_rate(won @ deck, deck.sum(0), base),
            "gih_wr": shrunk_rate(won @ hand, hand.sum(0), base)}


def covariates(games: pl.DataFrame) -> np.ndarray:
    """Game context, not deck: play/draw, mulligans, user skill bucket, user and opponent rank."""
    on_play = flag(games["on_play"]).astype(np.float32)
    mull = (games["num_mulligans"].cast(pl.Float32, strict=False) - games["opp_num_mulligans"].cast(pl.Float32, strict=False))
    skill = games["user_game_win_rate_bucket"].cast(pl.Float32, strict=False)
    skill_known = skill.is_not_null().to_numpy().astype(np.float32)
    skill = skill.fill_null(skill.mean()).to_numpy() - 0.5
    ranks = np.eye(N_RANKS, dtype=np.float32)
    return np.hstack([on_play[:, None] - 0.5, mull.fill_null(0).to_numpy()[:, None],
                      skill[:, None], skill_known[:, None],
                      ranks[rank_index(games["rank"])], ranks[rank_index(games["opp_rank"])]])


def design(U: np.ndarray, O: np.ndarray, cov: np.ndarray, symmetric: bool = True):
    """Sparse design: card count difference (or both sides separately), spell count diff, cov."""
    diff = U.astype(np.float32) - O.astype(np.float32)
    n = diff.sum(1, keepdims=True)
    cards = [diff] if symmetric else [U.astype(np.float32), -O.astype(np.float32)]
    return sparse.hstack([sparse.csr_matrix(m) for m in cards] + [sparse.csr_matrix(n),
                                                                  sparse.csr_matrix(cov)]).tocsr()


def fit_logreg(X, y, C: float, w=None) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=2000).fit(X, y, sample_weight=w)


# ---------------------------------------------------------------- scoring decks

def top_k_mean(pool: np.ndarray, value: np.ndarray, k: int = 23) -> np.ndarray:
    """Mean value of the k best cards in each pool (the paper's heuristic baseline)."""
    out = np.empty(len(pool), dtype=np.float64)
    order = np.argsort(-value)
    for i, row in enumerate(pool):
        copies = np.repeat(order, row[order])[:k]
        out[i] = value[copies].mean() if len(copies) else np.nan
    return out


def binomial_nll(p, w, l) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(w * np.log(p) + l * np.log1p(-p)).sum() / (w + l).sum())


def deck_metrics(p: np.ndarray, w: np.ndarray, l: np.ndarray, rng_seed: int = 0) -> dict:
    """Spearman vs observed W/(W+L), per-game NLL, and the Spearman you would get if p were true.

    The ceiling simulates each run under the 7-win / 3-loss stopping rule from p; it shows how much
    of the gap to 1.0 is just the noise of a handful of Bo1 games.
    """
    ok = ((w + l) > 0) & np.isfinite(p)
    p, w, l = p[ok], w[ok], l[ok]
    wr = w / (w + l)
    return {"spearman": float(spearmanr(p, wr).statistic), "nll": binomial_nll(p, w, l),
            "ceiling": float(spearmanr(p, simulate_runs(p, rng_seed)).statistic), "n": int(ok.sum())}


def simulate_runs(p: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.zeros(len(p))
    l = np.zeros(len(p))
    live = np.ones(len(p), dtype=bool)
    while live.any():
        win = rng.random(len(p)) < p
        w += live & win
        l += live & ~win
        live &= (w < 7) & (l < 3)
    return w / (w + l)


def calibrate(score_tr, rank_tr, w_tr, l_tr, score_te, rank_te) -> np.ndarray:
    """Map a 1-d deck score (+ rank) to p with a binomial logistic fit on train drafts."""
    def X(s, r):
        return np.hstack([s[:, None], np.eye(N_RANKS)[r]])
    mu, sd = np.nanmean(score_tr), np.nanstd(score_tr) + 1e-9
    s_tr = np.nan_to_num((score_tr - mu) / sd)
    s_te = np.nan_to_num((score_te - mu) / sd)
    m = binomial_logreg(X(s_tr, rank_tr), w_tr, l_tr, C=1e4)
    return m.predict_proba(X(s_te, rank_te))[:, 1]


def binomial_logreg(X, w, l, C: float) -> LogisticRegression:
    """Logistic regression on W wins and L losses per row (two weighted rows per draft)."""
    X2 = sparse.vstack([sparse.csr_matrix(X)] * 2) if sparse.issparse(X) else np.vstack([X, X])
    y = np.r_[np.ones(len(w)), np.zeros(len(l))]
    sw = np.r_[w, l]
    keep = sw > 0
    return fit_logreg(X2[keep], y[keep], C, sw[keep])


# ---------------------------------------------------------------- set transformer

class PoolEncoder(nn.Module):
    """DraftEncoder-lite: self-attention over the final pool, rank FiLM, [CLS] -> p(win).

    The paper encodes the whole pick sequence and uses oracle-text embeddings; this first
    version sees only the final pool, with learned card embeddings plus optional per-card
    meta features (win rates, colours, mana value, rarity).
    """

    def __init__(self, n_cards: int, meta: np.ndarray | None, d: int = 64, heads: int = 4,
                 layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        self.meta = None
        if meta is not None:
            m = np.vstack([meta, np.zeros((1, meta.shape[1]))]).astype(np.float32)
            self.register_buffer("meta_table", torch.from_numpy(m))
            self.meta = nn.Sequential(nn.Linear(m.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.film = nn.Embedding(N_RANKS, 2 * d)
        nn.init.zeros_(self.film.weight)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, cards, rank):
        x = self.emb(cards)
        if self.meta is not None:
            x = x + self.meta(self.meta_table[cards])
        g, b = self.film(rank).chunk(2, -1)
        x = x * (1 + g[:, None]) + b[:, None]
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        pad = torch.cat([torch.zeros_like(cards[:, :1], dtype=torch.bool),
                         cards == self.emb.padding_idx], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, 0]).squeeze(-1)


def pool_tokens(pool: np.ndarray) -> np.ndarray:
    """[D, C] copies -> [D, max_cards] card indices padded with C."""
    D, C = pool.shape
    L = int(pool.sum(1).max())
    out = np.full((D, L), C, dtype=np.int64)
    for i, row in enumerate(pool):
        cards = np.repeat(np.arange(C), row)
        out[i, :len(cards)] = cards
    return out


def train_pool_encoder(d: Drafts, meta: np.ndarray | None, epochs: int = 40, seed: int = 0,
                       device: str | None = None, log=print):
    torch.manual_seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    toks = torch.from_numpy(pool_tokens(d.pool))
    rank = torch.from_numpy(d.rank)
    W, L = torch.from_numpy(d.wins), torch.from_numpy(d.losses)
    tr, va = np.flatnonzero(d.mask("train")), np.flatnonzero(d.mask("val"))
    model = PoolEncoder(len(d.names), meta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    def nll(idx):
        z = model(toks[idx].to(device), rank[idx].to(device))
        w, l = W[idx].to(device), L[idx].to(device)
        return -(w * nn.functional.logsigmoid(z) + l * nn.functional.logsigmoid(-z)).sum(), (w + l).sum()

    def predict(idx):
        model.eval()
        out = []
        with torch.no_grad():
            for b in np.array_split(idx, max(1, len(idx) // 2048)):
                out.append(torch.sigmoid(model(toks[b].to(device), rank[b].to(device))).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), len(tr) // 256):
            loss, n = nll(b)
            opt.zero_grad()
            (loss / n).backward()
            opt.step()
        v = binomial_nll(predict(va), d.wins[va], d.losses[va])
        log(f"  epoch {ep:2d} val nll {v:.5f}")
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)
    return model, predict
