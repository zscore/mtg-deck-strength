"""Candidate A (DRAFT, not run on real data): a matchmaking-aware game-level likelihood for deck strength.

    python results/methodology/drafts/mm_likelihood.py          # synthetic self-test, ~10-25 s on 2 CPU threads

Model. Draft i plays games g = 1..G_i (Bo1, so one game per match) at records (w_g, l_g) before each game:

    e_ig   = d + deck_strength(deck_ig) + beta . cov_ig + rho[rank_ig] + t * u_i,      u_i ~ N(0, 1)
    P(win) = sum_j pool[state_ig, j] * expit(e_ig - s * x_j)                           (structural pool)
       or  = expit(e_ig - m(w_g, l_g))                                                 (parametric / free pool)

  deck_strength  card power w . x (+ FM pairs, deck features, affinity column: same scorer as round 4's game FM)
  cov            play/draw, mulligan difference, experience bucket. NOT the skill bucket (leaky) and NOT the
                 record before the game as free dummies (that is the thing being replaced)
  rho[rank]      the *net* rank effect: rank is both the player's skill and the opponent pool they are matched into
                 (opp_rank is None everywhere). Only the net effect is identified; deck ratings exclude it.
  u_i            per-draft random effect (player skill + deck residual + build luck), integrated out with
                 Gauss-Hermite quadrature: the likelihood of a draft is  int prod_g P(y_g | u) phi(u) du.
  pool[state]    opponent strength distribution at each record, from the recursion of
                 scripts/matchmaking_variance.opponents (population ~ N(0, s^2), everyone starts 0-0, players at a
                 record meet players at the same record), with a fuzziness lam: with prob (1 - lam) the opponent
                 comes from the all-games mixture instead. Differentiable in s, so s is learned jointly (or tied to
                 the users' own spread, s^2 = Var(eta) + t^2).
  m(w, l)        constrained parametric alternative: m = A[w] - B[l], A and B non-decreasing (cumulative softplus),
                 m(0, 0) = 0.

Why free record dummies are wrong here: the record before game g is caused by the deck (and by u). Conditioning on it
opens a collider: among drafts at 5-0, weak decks are there because of high u, so within a record cell deck strength
and u are negatively correlated and the deck coefficients shrink. The dummy also mixes two opposite effects
("this record means a stronger player", "this record means stronger opponents"). Integrating u out and tying the
opponent term to the strength distribution separates them, which is what makes the scale identified.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_W, MAX_L, G_MAX = 7, 3, 9
STATES = [(w, l) for w in range(MAX_W) for l in range(MAX_L)]  # same order as matchmaking_variance.SID
SID = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)  # 21; index 21 = record unknown (incomplete history) -> all-games mixture


def gh(n: int):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(w / w.sum(), dtype=torch.float32)


def state_index(wins_before: np.ndarray, losses_before: np.ndarray, known: np.ndarray | None = None) -> np.ndarray:
    s = np.minimum(wins_before, MAX_W - 1) * MAX_L + np.minimum(losses_before, MAX_L - 1)
    return np.where(known if known is not None else True, s, N_STATES).astype(np.int64)


# ---------------------------------------------------------------- opponent pool

def opponents(s: torch.Tensor, x: torch.Tensor, w: torch.Tensor, lam: float | torch.Tensor = 1.0,
              match: str = "record", n_iter: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """[22, J] opponent strength distribution (weights on nodes s * x) at each record state, row 21 the all-games
    mixture; and [21] the share of games played at each state. Differentiable in s (and lam).

    match="record": paired with players at the same (w, l); "wins": the same win count, any losses.
    lam < 1 or match="wins" needs the population of other states, so the recursion is iterated to a fixed point."""
    J = len(x)
    beat = torch.sigmoid(s * (x[:, None] - x[None, :]))  # [J, J] node k beats node j
    marg = w
    pools = None
    for _ in range(n_iter if (match != "record" or (not isinstance(lam, float) or lam < 1.0)) else 1):
        mass = [torch.zeros(J) for _ in STATES]
        mass[0] = w
        g_prev = pools
        for t in range(MAX_W + MAX_L - 1):
            for st in [st for st in STATES if sum(st) == t]:
                i = SID[st]
                m = mass[i]
                tot = m.sum()
                if float(tot) == 0.0:
                    continue
                if match == "record" or g_prev is None:
                    g = m / tot
                else:
                    g = g_prev[i]
                opp = lam * g + (1 - lam) * marg
                p = beat @ opp
                wl = (st[0] + 1, st[1])
                if wl[0] < MAX_W:
                    mass[SID[wl]] = mass[SID[wl]] + m * p
                ll = (st[0], st[1] + 1)
                if ll[1] < MAX_L:
                    mass[SID[ll]] = mass[SID[ll]] + m * (1 - p)
        M = torch.stack(mass)  # [21, J]
        share = M.sum(1)
        marg = M.sum(0) / share.sum()
        g_rec = M / share.clamp(min=1e-30)[:, None]
        if match == "wins":
            byw = torch.stack([M[[SID[(a, l)] for l in range(MAX_L)]].sum(0) for a in range(MAX_W)])
            byw = byw / byw.sum(1, keepdim=True).clamp(min=1e-30)
            g_rec = byw[[st[0] for st in STATES]]
        pools = g_rec
    out = lam * pools + (1 - lam) * marg[None]
    return torch.cat([out, marg[None]]), share / share.sum()


class OpponentPool(nn.Module):
    """log P(y | user strength e, state). mode: none | free | param | struct."""

    def __init__(self, mode: str = "struct", n_nodes: int = 16, s_init: float = 0.4, lam: float = 1.0,
                 learn_lam: bool = False, match: str = "record", tie_s: bool = False, learn_s: bool = True):
        super().__init__()
        self.mode, self.match, self.tie_s = mode, match, tie_s
        x, w = gh(n_nodes)
        self.register_buffer("x", x)
        self.register_buffer("w", w)
        if mode == "free":
            self.m = nn.Parameter(torch.zeros(N_STATES + 1))  # state 0 and "unknown" fixed at 0 in forward
        elif mode == "param":
            self.a = nn.Parameter(torch.full((MAX_W - 1,), -2.0))  # increments of A[w] (softplus > 0)
            self.b = nn.Parameter(torch.full((MAX_L - 1,), -2.0))
        elif mode == "struct":
            self.log_s = nn.Parameter(torch.tensor(float(np.log(s_init))), requires_grad=learn_s and not tie_s)
            self.lam_logit = nn.Parameter(torch.tensor(float(np.log(lam / (1 - lam + 1e-6)) if lam < 1 else 6.0)),
                                          requires_grad=learn_lam)
        self.s_tied = None  # set by the model each step when tie_s

    def s(self):
        if self.tie_s and self.s_tied is not None:
            return self.s_tied
        return self.log_s.exp()

    def offsets(self):
        """[22] deterministic opponent offset m(state) for free / param modes."""
        if self.mode == "none":
            return torch.zeros(N_STATES + 1, device=self.x.device)
        if self.mode == "free":
            keep = torch.ones(N_STATES + 1, device=self.x.device)
            keep[0] = 0
            keep[N_STATES] = 0
            return self.m * keep
        A = torch.cat([torch.zeros(1), torch.cumsum(F.softplus(self.a), 0)])
        B = torch.cat([torch.zeros(1), torch.cumsum(F.softplus(self.b), 0)])
        m = torch.stack([A[w] - B[l] for w, l in STATES])
        return torch.cat([m, m.mean()[None]])  # unknown state: average offset (refine with game shares if needed)

    def log_lik(self, e: torch.Tensor, state: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """e [..., U] user strength at the u-nodes; state [...] long; y [...] 0/1. Returns [..., U]."""
        sign = (2 * y - 1).unsqueeze(-1)
        if self.mode != "struct":
            return F.logsigmoid(sign * (e - self.offsets()[state].unsqueeze(-1)))
        lam = torch.sigmoid(self.lam_logit)
        pool, _ = opponents(self.s(), self.x.cpu(), self.w.cpu(), lam.cpu(), self.match)
        logpool = pool.clamp(min=1e-30).log().to(e.device)[state]          # [..., J]
        z = e.unsqueeze(-1) - self.s() * self.x                             # [..., U, J]
        return torch.logsumexp(logpool.unsqueeze(-2) + F.logsigmoid(sign.unsqueeze(-1) * z), -1)


# ---------------------------------------------------------------- deck scorer + full model

class LinearDeck(nn.Module):
    """w . counts (+ optional FM pairs on a card mask, + linear deck features). Production: swap in
    decks.synergy.DeckFM.deck_score plus the Scryfall deck features / affinity columns of round 4."""

    def __init__(self, n_cards: int, n_feat: int = 0, k: int = 0, vmask=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_cards))
        self.v = nn.Parameter(0.01 * torch.randn(n_cards, k)) if k else None
        self.register_buffer("vmask", torch.ones(n_cards, 1) if vmask is None else torch.as_tensor(vmask)[:, None])
        self.f = nn.Linear(n_feat, 1, bias=False) if n_feat else None

    def forward(self, x, feat=None):
        s = x @ self.w
        if self.v is not None:
            v = self.v * self.vmask
            xv = x @ v
            s = s + 0.5 * (xv.pow(2) - x.pow(2) @ v.pow(2)).sum(-1)
        if self.f is not None and feat is not None:
            s = s + self.f(feat).squeeze(-1)
        return s

    def penalty(self, l2_w, l2_v):
        p = l2_w * self.w.pow(2).sum()
        return p + (l2_v * (self.v * self.vmask).pow(2).sum() if self.v is not None else 0)


class MMGameModel(nn.Module):
    """Per-draft batch: builds [B, G] (index into a build table), state [B, G], y [B, G], cov [B, G, P],
    rank [B, G], mask [B, G]. `deck_eta` [n_builds] is computed by the caller (any deck scorer: linear, FM,
    transformer), so this module is only the likelihood around it."""

    def __init__(self, n_cov: int, n_ranks: int, pool: OpponentPool, random_effect: bool = True,
                 n_u: int = 12, t_init: float = 0.25):
        super().__init__()
        self.pool, self.random_effect = pool, random_effect
        self.d = nn.Parameter(torch.zeros(()))
        self.beta = nn.Linear(n_cov, 1, bias=False) if n_cov else None
        self.rho = nn.Embedding(n_ranks, 1)
        nn.init.zeros_(self.rho.weight)
        self.log_t = nn.Parameter(torch.tensor(float(np.log(t_init))), requires_grad=random_effect)
        ux, uw = gh(n_u) if random_effect else (torch.zeros(1), torch.ones(1))
        self.register_buffer("ux", ux)
        self.register_buffer("uw", uw)

    def t(self):
        return self.log_t.exp() if self.random_effect else torch.zeros(())

    def nll(self, eta_deck, state, y, mask, cov=None, rank=None, reduce=True):
        """eta_deck [B, G]: deck strength of the build played in each game. Returns summed -loglik and n games
        (reduce) or per-draft -loglik [B]."""
        e = self.d + eta_deck
        if self.beta is not None and cov is not None:
            e = e + self.beta(cov).squeeze(-1)
        if rank is not None:
            e = e + self.rho(rank).squeeze(-1)
        if self.pool.tie_s:  # users are as spread out as the population they are matched into
            self.pool.s_tied = (eta_deck[mask].var().detach() + self.t() ** 2).sqrt()
        eu = e.unsqueeze(-1) + self.t() * self.ux                     # [B, G, U]
        ll = self.pool.log_lik(eu, state, y.float()) * mask.unsqueeze(-1)
        per_draft = -torch.logsumexp(ll.sum(1) + self.uw.log(), -1)   # [B]
        return (per_draft.sum(), mask.sum()) if reduce else per_draft


# ---------------------------------------------------------------- training loop (generic)

def fit(deck, model, data, tr, va, l2_w=1e-4, l2_v=1e-3, epochs=60, batch=2000, lr=0.05, patience=4, lr_drops=2,
        seed=0, log=lambda s: None):
    """data: dict of tensors per draft (X [D, G, C] or builds, state, y, mask, cov, rank, feat). Early-stops on the
    validation drafts' marginal NLL per game, with learning-rate drops (as synergy._train)."""
    torch.manual_seed(seed)
    params = list(deck.parameters()) + list(model.parameters())
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)

    def loss(idx):
        feat = data["feat"][idx] if "feat" in data else None
        eta = deck(data["X"][idx], feat)
        if eta.dim() == 1:  # one build per draft (synthetic): broadcast over its games
            eta = eta[:, None].expand(-1, data["y"].shape[1])
        tot, n = model.nll(eta, data["state"][idx], data["y"][idx], data["mask"][idx], data["cov"][idx],
                           data["rank"][idx])
        return tot, n

    def val():
        with torch.no_grad():
            tot, n = 0.0, 0.0
            for b in np.array_split(va, max(1, len(va) // 4000)):
                a, c = loss(b)
                tot, n = tot + float(a), n + float(c)
        return tot / n

    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // batch)):
            tot, n = loss(b)
            obj = tot / n + deck.penalty(l2_w, l2_v)
            opt.zero_grad()
            obj.backward()
            opt.step()
        v = val()
        log(f"    epoch {ep} val {v:.5f}")
        if v < best - 1e-6:
            best, bad = v, 0
            state = ({k: t.detach().clone() for k, t in deck.state_dict().items()},
                     {k: t.detach().clone() for k, t in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if lr_drops <= 0:
                    break
                lr_drops, bad = lr_drops - 1, 0
                deck.load_state_dict(state[0])
                model.load_state_dict(state[1])
                for g in opt.param_groups:
                    g["lr"] /= 3
    deck.load_state_dict(state[0])
    model.load_state_dict(state[1])
    return best


# ---------------------------------------------------------------- synthetic self-test

def simulate(D, C, sd_deck, sd_u, d_users, s_pop, lam_true=1.0, seed=0, skill_deck_corr=0.0):
    """skill_deck_corr: skilled players favour a *different* card direction a (orthogonal to the true values w):
    u = sd_u (c * std(X a) + sqrt(1 - c^2) z). A deck score that absorbs it predicts records better but rates
    cards worse; this is the skill confound the real game FM controls with the leaky bucket."""
    """Drafts with true deck strength eta = X w, skill u, playing full 7-win / 3-loss runs against the matchmaking
    population (opponent strength drawn from the recursion pool at each record). Returns per-draft padded arrays."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.full(C, 0.5), size=8)               # 8 "archetypes" of card popularity
    arch = rng.integers(0, 8, D)
    X = np.stack([rng.multinomial(23, pi[a]) for a in arch]).astype(np.float32)
    w_true = rng.standard_normal(C)
    eta = X @ w_true
    eta = (eta - eta.mean()) / eta.std() * sd_deck
    z = rng.standard_normal(D)
    a = rng.standard_normal(C)
    a -= (a @ w_true) / (w_true @ w_true) * w_true
    xa = X @ a
    xa = (xa - xa.mean()) / xa.std()
    u = sd_u * (skill_deck_corr * xa + np.sqrt(1 - skill_deck_corr ** 2) * z)
    e = d_users + eta + u
    x, w = gh(24)
    with torch.no_grad():
        pool, _ = opponents(torch.tensor(s_pop), x, w, lam_true)
    pool, xs = pool.numpy().astype(np.float64), x.numpy()
    pool /= pool.sum(1, keepdims=True)
    st = np.full((D, G_MAX), N_STATES, np.int64)
    y = np.zeros((D, G_MAX), np.float32)
    on_play = (rng.random((D, G_MAX)) < 0.5).astype(np.float32)
    mask = np.zeros((D, G_MAX), bool)
    W, L = np.zeros(D, int), np.zeros(D, int)
    live = np.ones(D, bool)
    for g in range(G_MAX):
        i = np.flatnonzero(live)
        if not len(i):
            break
        s_i = W[i] * MAX_L + L[i]
        cdf = np.cumsum(pool[s_i], 1)
        node = (rng.random(len(i))[:, None] > cdf).sum(1).clip(max=len(xs) - 1)
        p = 1 / (1 + np.exp(-(e[i] + 0.12 * (on_play[i, g] - 0.5) - s_pop * xs[node])))
        win = rng.random(len(i)) < p
        st[i, g], y[i, g], mask[i, g] = s_i, win, True
        W[i] += win
        L[i] += ~win
        live &= (W < MAX_W) & (L < MAX_L)
    # a 17Lands-style skill bucket: the user's win rate over ~10 drafts, this one included (so ~10% of it is this
    # run's own outcome, which is why it is leaky), on the logit-ish scale
    own = (W - L * 7 / 3) / np.maximum(W + L, 1)
    others = u + d_users + sd_deck * rng.standard_normal(D) / 3 + 0.12 * rng.standard_normal(D)
    bucket = (0.9 * others + 0.1 * own).astype(np.float32)
    bucket = (bucket - bucket.mean()) / bucket.std()
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    cov = np.stack([on_play - 0.5, np.repeat(bucket[:, None], G_MAX, 1)], -1)
    data = {"X": t(X), "state": t(st), "y": t(y), "mask": t(mask),
            "cov": t(cov), "rank": torch.zeros(D, G_MAX, dtype=torch.long)}
    return data, X, eta, u, W, L


def run_specs(data, X, eta, u, W, L, specs, tr, va, te, s_pop):
    from scipy.stats import pearsonr, spearmanr
    wr = W[te] / np.maximum(W[te] + L[te], 1)
    C = X.shape[1]
    rows = []
    for name, mode, re_, kw, bucket in specs:
        t1 = time.time()
        dat = dict(data, cov=data["cov"] if bucket else data["cov"][..., :1])
        deck = LinearDeck(C)
        pool = OpponentPool(mode, n_nodes=12, **kw)
        model = MMGameModel(dat["cov"].shape[-1], 1, pool, random_effect=re_, n_u=10)
        best = fit(deck, model, dat, tr, va, l2_w=1e-5, epochs=60, batch=3000, lr=0.03, patience=3, lr_drops=2)
        with torch.no_grad():
            score = deck(torch.from_numpy(X[te])).numpy()
            score = score - score.mean()
            extra = f"t {float(model.t()):.3f}"
            if mode == "struct":
                extra += f", s {float(pool.s()):.3f}"
            if mode in ("free", "param"):
                extra += f", m(6-0)-m(0-2) {float(pool.offsets()[SID[(6, 0)]] - pool.offsets()[SID[(0, 2)]]):+.2f}"
        slope = np.cov(score, eta[te])[0, 1] / np.var(eta[te], ddof=1)  # scale: 1 = right, < 1 = compressed
        r = pearsonr(score, eta[te]).statistic
        r_tot = pearsonr(score, eta[te] + u[te]).statistic
        rho = spearmanr(score, wr).statistic
        rows.append(dict(model=name, val_nll=best, slope=slope, r_deck=r, r_deck_plus_skill=r_tot, spearman_wr=rho,
                         info=extra))
        print(f"  {name:54s} val {best:.5f}  scale {slope:.2f}  r(deck) {r:.4f}  r(deck+skill) {r_tot:.4f}  "
              f"Spearman(WR) {rho:.4f}  {extra} [{time.time() - t1:.1f}s]", flush=True)
    return rows


def selftest():
    from scipy.stats import spearmanr
    torch.set_num_threads(2)
    t0 = time.time()
    C, D = 40, 24000
    sd_deck, sd_u, d_users = 0.30, 0.25, 0.10
    s_pop = float(np.sqrt(sd_deck ** 2 + sd_u ** 2))
    tr, va, te = np.arange(0, 15000), np.arange(15000, 18000), np.arange(18000, D)
    out = {}
    for scen, corr in (("1: skill independent of deck", 0.0), ("2: skill correlated with an off-value card direction (r=0.6)", 0.6)):
        data, X, eta, u, W, L = simulate(D, C, sd_deck, sd_u, d_users, s_pop, seed=1, skill_deck_corr=corr)
        wr = W[te] / np.maximum(W[te] + L[te], 1)
        print(f"scenario {scen}: {D} drafts, {int(data['mask'].sum())} games; sd(deck) {sd_deck}, sd(u) {sd_u}, "
              f"population s {s_pop:.3f}; oracle Spearman(true deck, test WR) {spearmanr(eta[te], wr).statistic:.3f}, "
              f"(true deck + skill) {spearmanr(eta[te] + u[te], wr).statistic:.3f} [{time.time() - t0:.1f}s]")
        specs = [
            ("iid games + free record dummies + skill bucket (round 4)", "free", False, {}, True),
            ("iid games, no record terms", "none", False, {}, False),
            ("iid games + free record dummies", "free", False, {}, False),
            ("RE + free record dummies", "free", True, {}, False),
            ("RE + monotone parametric pool", "param", True, {}, False),
            ("RE + structural pool, s free", "struct", True, {}, False),
            ("RE + structural pool, s tied to users' spread", "struct", True, {"tie_s": True}, False),
            ("RE + structural pool, s fixed at truth", "struct", True, {"s_init": s_pop, "learn_s": False}, False),
        ]
        if corr > 0:
            specs = [sp for sp in specs if sp[0].startswith(("iid games + free record dummies", "iid games, no",
                                                              "RE + monotone", "RE + structural pool, s fixed"))]
            specs.append(("RE + monotone parametric pool + skill bucket", "param", True, {}, True))
        out[scen] = run_specs(data, X, eta, u, W, L, specs, tr, va, te, s_pop)
    print(f"total {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    selftest()
