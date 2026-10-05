"""Game-level deck models: card power + low-rank pair synergy (a factorization machine), and a
transformer over the deck's cards.

P(user wins game) = sigma(b + f(deck) + covariates), f(x) = w . x + sum_{i<j} x_i x_j <v_i, v_j>.

Multitask: with a task id per game (e.g. 0 = PremierDraft, 1 = Sealed), the card terms are shared
and each task gets its own bias and covariate weights, plus a per-task card power offset dw[t]
(L2-penalised, so a task only moves away from the shared power where its data insist).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss


class DeckFM(nn.Module):
    """`vmask` ([C] 0/1) limits the pair terms to some cards (e.g. nonland cards only, so synergy can't
    be the mana base); masked cards keep their power w."""

    def __init__(self, n_cards: int, n_cov: int, k: int, n_tasks: int = 1, vmask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_cards))
        self.dw = nn.Parameter(torch.zeros(n_tasks, n_cards)) if n_tasks > 1 else None
        self.v_ = nn.Parameter(0.01 * torch.randn(n_cards, k)) if k else None
        vm = torch.ones(n_cards) if vmask is None else torch.as_tensor(vmask, dtype=torch.float32)
        self.register_buffer("vmask", vm[:, None])
        self.cov = nn.Linear(n_cov, n_tasks)

    @property
    def v(self):
        return None if self.v_ is None else self.v_ * self.vmask

    def deck_score(self, x, task=None):
        s = x @ self.w
        if self.dw is not None and task is not None:
            s = s + (x * self.dw[task]).sum(-1)
        if self.v is not None:
            xv = x @ self.v
            s = s + 0.5 * (xv.pow(2) - (x.pow(2)) @ self.v.pow(2)).sum(-1)
        return s

    def forward(self, x, c, task=None):
        return self.deck_score(x, task) + _task_cov(self.cov(c), task)

    def penalty(self, l2_w, l2_v, l2_dw):
        p = l2_w * self.w.pow(2).sum()
        if self.v is not None:
            p = p + l2_v * self.v.pow(2).sum()
        if self.dw is not None:
            p = p + l2_dw * self.dw.pow(2).sum()
        return p


class DeckTransformer(nn.Module):
    """Self-attention over the distinct cards in a deck: f(x) = w . x + head(CLS).

    Each token is one distinct card: a learned card embedding + a copies embedding (basic lands
    can run to 17) + an MLP of the card's metadata (colours, rarity, mana value, types). The w . x
    skip term is the linear model's card power, so the attention only has to learn what card power
    misses (pairs, curve, colour consistency, ...). With several tasks each gets its own CLS token.
    """

    def __init__(self, n_cards: int, n_cov: int, meta: np.ndarray, n_tasks: int = 1, d: int = 64,
                 heads: int = 4, layers: int = 2, dropout: float = 0.1, max_copies: int = 20):
        super().__init__()
        self.n_cards, self.max_copies = n_cards, max_copies
        self.w = nn.Parameter(torch.zeros(n_cards))
        self.dw = nn.Parameter(torch.zeros(n_tasks, n_cards)) if n_tasks > 1 else None
        self.emb = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        self.copies = nn.Embedding(max_copies + 1, d)
        m = np.vstack([meta, np.zeros((1, meta.shape[1]))]).astype(np.float32)
        self.register_buffer("meta_table", torch.from_numpy(m))
        self.meta = nn.Sequential(nn.Linear(m.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(n_tasks, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)
        self.cov = nn.Linear(n_cov, n_tasks)

    def tokens(self, x):
        """Dense [B, C] counts -> distinct card ids [B, M] (padded with C) and copies [B, M]."""
        M = int((x > 0).sum(1).max())
        order = torch.argsort((x > 0).to(torch.int8), dim=1, descending=True, stable=True)[:, :M]
        copies = x.gather(1, order).long()
        return torch.where(copies > 0, order, self.n_cards), copies.clamp(max=self.max_copies)

    def deck_score(self, x, task=None):
        cards, copies = self.tokens(x)
        h = self.emb(cards) + self.copies(copies) + self.meta(self.meta_table[cards])
        cls = self.cls[0].expand(len(x), -1) if task is None else self.cls[task]
        h = torch.cat([cls[:, None], h], 1)
        pad = torch.cat([torch.zeros_like(cards[:, :1], dtype=torch.bool), cards == self.n_cards], 1)
        s = x @ self.w + self.head(self.enc(h, src_key_padding_mask=pad)[:, 0]).squeeze(-1)
        if self.dw is not None and task is not None:
            s = s + (x * self.dw[task]).sum(-1)
        return s

    def forward(self, x, c, task=None):
        return self.deck_score(x, task) + _task_cov(self.cov(c), task)

    def penalty(self, l2_w, l2_v, l2_dw):
        p = l2_w * self.w.pow(2).sum()
        if self.dw is not None:
            p = p + l2_dw * self.dw.pow(2).sum()
        return p


def _task_cov(z, task):
    return z[:, 0] if task is None else z.gather(1, task[:, None]).squeeze(1)


def fit(X, cov, y, split, k, l2_w, l2_v, task=None, l2_dw=1e-3, epochs=40, batch=4096, lr=3e-3, patience=3,
        log=print, vmask=None, lr_drops=0):
    """Linear (k = 0) or FM. Rows with split 0 train, 1 validate (early stopping); others are ignored.
    `task` ([N] ints) turns on the multitask model; validation is then the mean of per-task log loss.
    `lr_drops` > 0: on a plateau, go back to the best state and divide the learning rate by 3, that many times."""
    torch.manual_seed(0)
    m = DeckFM(X.shape[1], cov.shape[1], k, _n_tasks(task), vmask=vmask)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m, best, predict = _train(m, opt, X, cov, y, split, task, lambda: m.penalty(l2_w, l2_v, l2_dw),
                              epochs, batch, patience, "cpu", lr_drops=lr_drops)
    log(f"    k={k} l2_w={l2_w:g} l2_v={l2_v:g}: val logloss {best:.5f}")
    return m, best, predict


def fit_transformer(X, cov, y, split, meta, task=None, d=64, layers=2, heads=4, dropout=0.1, l2_w=1e-5, l2_dw=1e-3,
                    weight_decay=0.01, epochs=30, batch=512, lr=1e-3, patience=3, device=None, log=print):
    """DeckTransformer with the same interface as `fit`. AdamW decays the attention weights; the card
    power terms get the same explicit L2 as the linear model."""
    torch.manual_seed(0)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = DeckTransformer(X.shape[1], cov.shape[1], meta, _n_tasks(task), d=d, heads=heads, layers=layers,
                        dropout=dropout)
    m.to(device)
    linear = [m.w] + ([m.dw] if m.dw is not None else [])
    rest = [p for p in m.parameters() if all(p is not q for q in linear)]
    opt = torch.optim.AdamW([{"params": linear, "weight_decay": 0}, {"params": rest}], lr=lr,
                            weight_decay=weight_decay)
    m, best, predict = _train(m, opt, X, cov, y, split, task, lambda: m.penalty(l2_w, 0, l2_dw),
                              epochs, batch, patience, device, log=log)
    log(f"    transformer d={d} layers={layers} dropout={dropout}: val logloss {best:.5f}")
    return m, best, predict


def _n_tasks(task):
    return 1 if task is None else int(task.max()) + 1


def _train(m, opt, X, cov, y, split, task, penalty, epochs, batch, patience, device, log=lambda s: None,
           lr_drops=0):
    Xt, Ct, Yt = torch.from_numpy(X), torch.from_numpy(cov), torch.from_numpy(y.astype(np.float32))
    Tt = None if task is None else torch.from_numpy(task.astype(np.int64))
    tr, va = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    rng = np.random.default_rng(0)
    best, state, bad = np.inf, None, 0

    def run(idx):
        t = None if Tt is None else Tt[idx].to(device)
        return m(Xt[idx].to(device), Ct[idx].to(device), t)

    def predict(idx):
        m.eval()
        with torch.no_grad():
            out = [torch.sigmoid(run(b)).cpu() for b in np.array_split(idx, max(1, len(idx) // 8192))]
        return torch.cat(out).numpy()

    def val_loss():
        p = predict(va)
        if task is None:
            return log_loss(y[va], p)
        return np.mean([log_loss(y[va][task[va] == t], p[task[va] == t]) for t in np.unique(task[va])])

    for ep in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // batch)):
            loss = F.binary_cross_entropy_with_logits(run(b), Yt[b].to(device)) + penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_loss()
        log(f"      epoch {ep:2d} val logloss {v:.5f}")
        if v < best - 1e-6:
            best, bad, state = v, 0, {n: t.detach().clone() for n, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if lr_drops <= 0:
                    break
                lr_drops, bad = lr_drops - 1, 0
                m.load_state_dict(state)
                for group in opt.param_groups:
                    group["lr"] /= 3
    m.load_state_dict(state)
    m.eval()
    return m, best, predict
