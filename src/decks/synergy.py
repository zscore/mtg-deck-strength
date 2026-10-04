"""Game-level deck models: card power + low-rank pair synergy (a factorization machine).

P(user wins game) = sigma(b + f(deck) + covariates), f(x) = w . x + sum_{i<j} x_i x_j <v_i, v_j>.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss


class DeckFM(nn.Module):
    def __init__(self, n_cards: int, n_cov: int, k: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_cards))
        self.v = nn.Parameter(0.01 * torch.randn(n_cards, k)) if k else None
        self.cov = nn.Linear(n_cov, 1)

    def deck_score(self, x):
        s = x @ self.w
        if self.v is not None:
            xv = x @ self.v
            s = s + 0.5 * (xv.pow(2) - (x.pow(2)) @ self.v.pow(2)).sum(-1)
        return s

    def forward(self, x, c):
        return self.deck_score(x) + self.cov(c).squeeze(-1)


def fit(X, cov, y, split, k, l2_w, l2_v, epochs=40, batch=4096, lr=3e-3, patience=3, log=print):
    torch.manual_seed(0)
    Xt, Ct, Yt = torch.from_numpy(X), torch.from_numpy(cov), torch.from_numpy(y.astype(np.float32))
    tr, va = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    m = DeckFM(X.shape[1], cov.shape[1], k)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    best, state, bad = np.inf, None, 0

    def predict(idx):
        with torch.no_grad():
            return torch.sigmoid(m(Xt[idx], Ct[idx])).numpy()

    for ep in range(epochs):
        for b in np.array_split(rng.permutation(tr), len(tr) // batch):
            loss = F.binary_cross_entropy_with_logits(m(Xt[b], Ct[b]), Yt[b])
            loss = loss + l2_w * m.w.pow(2).sum() + (l2_v * m.v.pow(2).sum() if m.v is not None else 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = log_loss(y[va], predict(va))
        if v < best - 1e-6:
            best, bad, state = v, 0, {n: t.clone() for n, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    log(f"    k={k} l2_w={l2_w:g} l2_v={l2_v:g}: val logloss {best:.5f} after {ep + 1} epochs")
    return m, best, predict
