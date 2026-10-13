"""Candidate B (DRAFT, not run on real data): a game-level set transformer over the registered deck, PickNet-initialised
card embeddings, cross-set trunk with per-set heads, trained with candidate A's matchmaking-aware likelihood plus a
distillation auxiliary toward the LightGBM blend's out-of-fold logits.

    python results/methodology/drafts/game_set_transformer.py               # synthetic self-test (~15 s, ~0.4 GB)
    python results/methodology/drafts/game_set_transformer.py time 128 3 256   # one production-size step timing

Per draft i of set s, each build b the drafter registered:

  tokens     one per distinct card in the build (lands included; <= 48), plus a set token and a deck-feature token
  card emb   h_c = W_in LN(z_c) + delta_c + copies[n_c]        z_c: PickNet all_d256 card vector (256-d, frozen;
             results/<SET>/picknet/feats.npz:card_vec), delta_c: learned residual (L2, init 0). Ablation: z_c random.
  trunk      L pre-norm self-attention blocks (shared over sets), PMA pooling with k seeds -> [k * d]
  deck score eta_b = sum_c n_c w_c  +  f_s . F_b  +  head_s(pool)       w_c: per-set card power (the linear model as a
             skip path), F_b: Scryfall deck features + affinity column, head_s: per-set linear readout (init 0, so the
             model starts as the linear game model and the trunk only learns what it misses)
  games      candidate A: P(win g) = sum_j pool_s[state_g, j] expit(d_s + eta_{b(g)} + beta_s . cov_g + rho_s[rank_g]
             + t_s u_i - s_s x_j), u_i integrated by Gauss-Hermite (mm_likelihood.MMGameModel, one per set)
  distill    zhat_i = a_s eta_{main build of i} + c_s[rank_i] + b_s,  loss alpha * mean((zhat - teacher_i)^2) / var_s
             teacher_i: the round-10 LightGBM + game-FM blend logit, cross-fitted (drafts scored by folds that did
             not train on them; see oof_teacher.py)

Batches are single-set (a set is drawn with probability proportional to n_drafts^0.75, then B drafts from it), so
the opponent pool recursion runs once per step and per-set parameters need no gather.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mm_likelihood import G_MAX, MMGameModel, OpponentPool, simulate  # noqa: E402

MAX_TOK = 48
MAX_COPIES = 20


# ---------------------------------------------------------------- model

class CardEncoder(nn.Module):
    """Global card slots (one per (set, card column)); `base` [V, Dz] PickNet vectors (frozen) or None (random init)."""

    def __init__(self, n_slots: int, d: int, base: np.ndarray | None = None):
        super().__init__()
        self.pad = n_slots
        if base is not None:
            z = torch.as_tensor(base, dtype=torch.float32)
            self.register_buffer("base", torch.cat([z, z.new_zeros(1, z.shape[1])]))
            self.norm = nn.LayerNorm(z.shape[1])
            self.proj = nn.Linear(z.shape[1], d)
            self.delta = nn.Embedding(n_slots + 1, d, padding_idx=n_slots)
            nn.init.zeros_(self.delta.weight)
        else:
            self.base = None
            self.delta = nn.Embedding(n_slots + 1, d, padding_idx=n_slots)
            nn.init.normal_(self.delta.weight, std=0.02)
        self.copies = nn.Embedding(MAX_COPIES + 1, d)
        nn.init.normal_(self.copies.weight, std=0.02)

    def forward(self, ids, cnt):
        h = self.delta(ids) + self.copies(cnt.clamp(max=MAX_COPIES))
        if self.base is not None:
            h = h + self.proj(self.norm(self.base[ids]))
        return h


class PMA(nn.Module):
    """Pooling by multihead attention (Set Transformer): k learned seeds attend over the tokens."""

    def __init__(self, d: int, heads: int, k: int):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(k, d) * 0.02)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, pad):
        q = self.seeds[None].expand(len(h), -1, -1)
        o, _ = self.att(q, h, h, key_padding_mask=pad, need_weights=False)
        return self.ln(o).flatten(1)


class DeckSetTransformer(nn.Module):
    def __init__(self, n_slots: int, n_sets: int, n_feat: int, d: int = 128, layers: int = 3, heads: int = 4,
                 ff_mult: int = 2, k_seeds: int = 4, dropout: float = 0.1, base: np.ndarray | None = None):
        super().__init__()
        self.cards = CardEncoder(n_slots, d, base)
        self.set_tok = nn.Embedding(n_sets, d)
        nn.init.normal_(self.set_tok.weight, std=0.02)
        self.feat_tok = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d)) if n_feat else None
        layer = nn.TransformerEncoderLayer(d, heads, ff_mult * d, dropout, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pma = PMA(d, heads, k_seeds)
        self.head_w = nn.Parameter(torch.zeros(n_sets, k_seeds * d))  # per-set readouts, init 0
        self.head_b = nn.Parameter(torch.zeros(n_sets))
        self.w = nn.Embedding(n_slots + 1, 1, padding_idx=n_slots)       # per-card power (slots are set-specific)
        nn.init.zeros_(self.w.weight)
        self.fw = nn.Parameter(torch.zeros(n_sets, max(n_feat, 1)))      # per-set deck-feature weights
        self.n_feat = n_feat

    def forward(self, ids, cnt, set_id: int, feat=None):
        """ids, cnt [B, M] (padded with the pad slot / 0); feat [B, F]. Returns eta [B] and the linear part [B]."""
        B = len(ids)
        pad = ids == self.cards.pad
        h = self.cards(ids, cnt)
        extra = [self.set_tok.weight[set_id].expand(B, 1, -1)]
        if self.feat_tok is not None:
            extra.append(self.feat_tok(feat)[:, None])
        h = torch.cat(extra + [h], 1)
        pad = torch.cat([torch.zeros(B, len(extra), dtype=torch.bool, device=ids.device), pad], 1)
        h = self.trunk(h, src_key_padding_mask=pad)
        z = self.pma(h, pad)
        lin = (cnt.float() * self.w(ids).squeeze(-1)).sum(1)
        if self.n_feat:
            lin = lin + feat @ self.fw[set_id]
        return lin + z @ self.head_w[set_id] + self.head_b[set_id], lin

    def penalty(self, l2_w=1e-5, l2_delta=1e-4):
        return l2_w * self.w.weight.pow(2).sum() + l2_delta * self.cards.delta.weight.pow(2).sum() * (
            self.cards.base is not None)


class Distill(nn.Module):
    """Per-set affine map from deck score (+ draft rank) to the teacher's logit scale."""

    def __init__(self, n_sets: int, n_ranks: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_sets))
        self.b = nn.Parameter(torch.zeros(n_sets))
        self.c = nn.Parameter(torch.zeros(n_sets, n_ranks))

    def loss(self, eta, rank, teacher, set_id, var):
        zhat = self.a[set_id] * eta + self.b[set_id] + self.c[set_id][rank]
        return ((zhat - teacher) ** 2).mean() / var


# ---------------------------------------------------------------- data layout

def tokens(X: np.ndarray, slot0: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense [N, C] counts of one set -> [N, MAX_TOK] global slot ids (slot0 + column, pad = -1 for now) and copies."""
    N = len(X)
    ids = np.full((N, MAX_TOK), -1, np.int64)
    cnt = np.zeros((N, MAX_TOK), np.int64)
    for i, row in enumerate(X):
        nz = np.flatnonzero(row)[:MAX_TOK]
        ids[i, :len(nz)] = slot0 + nz
        cnt[i, :len(nz)] = row[nz]
    return ids, cnt


class SetData:
    """One set's tensors. Production: builds come from games' (draft_id, build_index) deck_ columns; here one build
    per draft (synthetic). games: build [D, G] index into the build table, state/y/mask/cov/rank [D, G]."""

    def __init__(self, set_id, ids, cnt, feat, game, teacher, split, rank_draft, main_build=None):
        self.set_id, self.ids, self.cnt, self.feat, self.main_build = set_id, ids, cnt, feat, main_build
        self.game, self.teacher, self.split, self.rank_draft = game, teacher, split, rank_draft
        self.tvar = float(np.nanvar(teacher[split == 0]))

    @classmethod
    def from_npz(cls, set_id, data_path, teacher_path, slot0):
        """Real data from prepare_next_model.py. Card columns become global slots slot0 + column; drafts without a
        registered deck are dropped from training (they have no games)."""
        z, t = np.load(data_path), np.load(teacher_path)
        ids = np.where(z["build_ids"] >= 0, z["build_ids"].astype(np.int64) + slot0, -1)
        keep = z["build_of_draft_main"] >= 0
        split = np.where(keep, z["split"], -1)
        game = {"build": z["game_build"], "state": z["game_state"], "y": z["game_y"], "mask": z["game_mask"],
                "cov": z["game_cov"], "rank": z["game_rank"]}
        return cls(set_id, ids, z["build_cnt"], z["build_feat"], game, t["teacher"], split, z["rank"],
                   main_build=z["build_of_draft_main"])


class Trainer:
    def __init__(self, sets: list[SetData], n_slots: int, n_ranks: int, n_cov: int, cfg: dict, base=None,
                 pool_mode="param", device="cpu"):
        self.sets, self.cfg, self.dev = sets, cfg, device
        n_feat = sets[0].feat.shape[1]
        self.net = DeckSetTransformer(n_slots, len(sets), n_feat, d=cfg["d"], layers=cfg["layers"],
                                      heads=cfg["heads"], k_seeds=cfg["k"], dropout=cfg["dropout"], base=base).to(device)
        self.lik = nn.ModuleList(MMGameModel(n_cov, n_ranks, OpponentPool(pool_mode, n_nodes=12), n_u=12)
                                 for _ in sets).to(device)
        self.dis = Distill(len(sets), n_ranks).to(device)
        frozen = []  # PickNet base is a buffer, so nothing to freeze explicitly; lower lr for the projection
        proj = list(self.net.cards.proj.parameters()) if base is not None else []
        rest = [p for p in list(self.net.parameters()) + list(self.lik.parameters()) + list(self.dis.parameters())
                if all(p is not q for q in proj + frozen)]
        self.opt = torch.optim.AdamW([{"params": rest}, {"params": proj, "lr": cfg["lr"] * 0.3}],
                                     lr=cfg["lr"], weight_decay=cfg["wd"])
        n = np.array([(s.split == 0).sum() for s in sets], float)
        self.p_set = n ** 0.75 / (n ** 0.75).sum()
        self.rng = np.random.default_rng(cfg.get("seed", 0))

    def _pad(self, ids):
        return torch.where(ids < 0, torch.full_like(ids, self.net.cards.pad), ids)

    def batch_loss(self, s: SetData, idx, alpha):
        g = {k: torch.as_tensor(v[idx]).to(self.dev) for k, v in s.game.items() if k != "build"}
        if "build" in s.game:  # real data: rows of the build table; each game gathers its own build's score
            gb = s.game["build"][idx]                                   # [B, G], -1 = no game
            main = s.main_build[idx]
            ub, inv = np.unique(np.r_[gb[gb >= 0], main], return_inverse=True)
            ids = self._pad(torch.as_tensor(s.ids[ub].astype(np.int64))).to(self.dev)
            eta_u, _ = self.net(ids, torch.as_tensor(s.cnt[ub].astype(np.int64)).to(self.dev), s.set_id,
                                torch.as_tensor(s.feat[ub].astype(np.float32)).to(self.dev))
            pos = torch.as_tensor(inv[:int((gb >= 0).sum())]).to(self.dev)
            eta_g = torch.zeros(gb.shape, device=self.dev)
            eta_g[torch.as_tensor(gb >= 0).to(self.dev)] = eta_u[pos]
            eta = eta_u[torch.as_tensor(inv[int((gb >= 0).sum()):]).to(self.dev)]  # main build: the rated deck
        else:  # synthetic: one build per draft
            ids = self._pad(torch.as_tensor(s.ids[idx])).to(self.dev)
            eta, _ = self.net(ids, torch.as_tensor(s.cnt[idx]).to(self.dev), s.set_id,
                              torch.as_tensor(s.feat[idx]).to(self.dev))
            eta_g = eta[:, None].expand(-1, G_MAX)
        tot, n = self.lik[s.set_id].nll(eta_g, g["state"].long(), g["y"], g["mask"], g["cov"].float(), g["rank"].long())
        loss = tot / n
        if alpha > 0:
            ok = np.isfinite(s.teacher[idx])
            if ok.any():
                k = torch.as_tensor(ok).to(self.dev)
                loss = loss + alpha * self.dis.loss(eta[k], torch.as_tensor(s.rank_draft[idx][ok]).long().to(self.dev),
                                                    torch.as_tensor(s.teacher[idx][ok]).to(self.dev), s.set_id, s.tvar)
        return loss, tot, n, eta

    def epoch(self, steps, alpha):
        self.net.train()
        for _ in range(steps):
            s = self.sets[self.rng.choice(len(self.sets), p=self.p_set)]
            idx = self.rng.choice(np.flatnonzero(s.split == 0), self.cfg["batch"], replace=False)
            loss, *_ = self.batch_loss(s, idx, alpha)
            loss = loss + self.net.penalty()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()

    @torch.no_grad()
    def evaluate(self, split=1):
        self.net.eval()
        out = []
        for s in self.sets:
            idx = np.flatnonzero(s.split == split)
            tot, n, etas = 0.0, 0.0, []
            for b in np.array_split(idx, max(1, len(idx) // 1024)):
                _, t_, n_, eta = self.batch_loss(s, b, 0.0)
                tot, n = tot + float(t_), n + float(n_)
                etas.append(eta.cpu().numpy())
            out.append((tot / n, np.concatenate(etas)))
        return out


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- synthetic self-test

def make_sets(n_sets=2, C=(40, 50), D=4000, seed=0):
    """Synthetic sets from mm_likelihood.simulate; 'PickNet' vectors = noisy random projection of true card values."""
    sets, bases, slot0 = [], [], 0
    rng = np.random.default_rng(seed)
    truth = []
    for s in range(n_sets):
        data, X, eta, u, W, L = simulate(D, C[s], 0.3, 0.25, 0.1, 0.39, seed=seed + s)
        # recover the true card values by least squares on eta (simulate does not return w)
        w_true = np.linalg.lstsq(X - X.mean(0), eta - eta.mean(), rcond=None)[0]
        z = np.outer(w_true / w_true.std(), rng.standard_normal(256)) + rng.standard_normal((C[s], 256))
        bases.append(z.astype(np.float32))
        ids, cnt = tokens(X.astype(np.int64), slot0)
        slot0 += C[s]
        split = np.zeros(D, np.int8)
        split[int(0.7 * D):] = 1
        split[int(0.85 * D):] = 2
        teacher = (eta + 0.5 * u + 0.1 * rng.standard_normal(D)).astype(np.float32)  # absorbs some skill
        game = {k: data[k].numpy() if hasattr(data[k], "numpy") else data[k] for k in ("state", "y", "mask", "cov",
                                                                                        "rank")}
        game["cov"] = game["cov"][..., :1]  # play/draw only, no skill bucket
        feat = rng.standard_normal((D, 3)).astype(np.float32)
        sets.append(SetData(s, ids, cnt, feat, game, teacher, split, np.zeros(D, np.int64)))
        truth.append((eta, W, L))
    return sets, np.concatenate(bases), slot0, truth


def invariance_checks(net, ids, cnt, feat):
    net.eval()
    pad = net.cards.pad
    ids_t = torch.where(torch.as_tensor(ids) < 0, pad, torch.as_tensor(ids))
    cnt_t = torch.as_tensor(cnt)
    with torch.no_grad():
        e0, _ = net(ids_t, cnt_t, 0, torch.as_tensor(feat))
        perm = torch.randperm(ids_t.shape[1])
        e1, _ = net(ids_t[:, perm], cnt_t[:, perm], 0, torch.as_tensor(feat))
        ids2 = torch.cat([ids_t, torch.full((len(ids_t), 5), pad)], 1)
        cnt2 = torch.cat([cnt_t, torch.zeros(len(ids_t), 5, dtype=cnt_t.dtype)], 1)
        e2, _ = net(ids2, cnt2, 0, torch.as_tensor(feat))
    return float((e0 - e1).abs().max()), float((e0 - e2).abs().max())


def timing(d, layers, batch, n_slots=4000, n_sets=12, M=30, steps=3):
    """CPU (2 threads) forward+backward of the deck network at production size; MPS is typically 3-8x faster."""
    net = DeckSetTransformer(n_slots, n_sets, 30, d=d, layers=layers, heads=4, k_seeds=4,
                             base=np.random.randn(n_slots, 256).astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    ids = torch.randint(0, n_slots, (batch, M))
    cnt = torch.randint(1, 3, (batch, M))
    feat = torch.randn(batch, 30)
    ts = []
    for _ in range(steps + 1):
        t = time.time()
        eta, _ = net(ids, cnt, 0, feat)
        eta.pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
        ts.append(time.time() - t)
    return n_params(net), float(np.median(ts[1:]))


def selftest():
    from scipy.stats import pearsonr, spearmanr
    torch.set_num_threads(2)
    torch.manual_seed(0)
    t0 = time.time()
    sets, base, n_slots, truth = make_sets()
    cfg = dict(d=32, layers=2, heads=4, k=2, dropout=0.0, lr=3e-3, wd=0.01, batch=128, seed=0)
    for name, b, alpha in (("PickNet-init, no distill", base, 0.0), ("PickNet-init, distill a=0.5", base, 0.5),
                           ("random-init, no distill", None, 0.0)):
        t1 = time.time()
        tr = Trainer(sets, n_slots, 1, 1, cfg, base=b)
        if name.startswith("PickNet-init, no"):
            e_perm, e_pad = invariance_checks(tr.net, sets[0].ids[:64], sets[0].cnt[:64], sets[0].feat[:64])
            print(f"  permutation invariance max |d eta| {e_perm:.2e}, padding invariance {e_pad:.2e}")
            assert e_perm < 1e-4 and e_pad < 1e-4
            # the real-data (build table) path must give the same loss as the one-build-per-draft path
            s0 = sets[0]
            gb = np.where(s0.game["mask"], np.arange(len(s0.ids))[:, None], -1)
            alt = SetData(0, s0.ids, s0.cnt, s0.feat, dict(s0.game, build=gb), s0.teacher, s0.split, s0.rank_draft,
                          main_build=np.arange(len(s0.ids)))
            tr.net.eval()
            with torch.no_grad():
                idx = np.arange(200)
                a, b = tr.batch_loss(s0, idx, 0.5)[0], tr.batch_loss(alt, idx, 0.5)[0]
            print(f"  build-table path vs per-draft path: loss {float(a):.6f} vs {float(b):.6f}")
            assert abs(float(a) - float(b)) < 1e-5
        v0 = np.mean([v for v, _ in tr.evaluate(1)])
        for ep in range(3):
            tr.epoch(30, alpha)
        res = tr.evaluate(2)
        v1 = np.mean([v for v, _ in tr.evaluate(1)])
        r = [pearsonr(e, truth[i][0][sets[i].split == 2]).statistic for i, (_, e) in enumerate(res)]
        sp = []
        for i, (_, e) in enumerate(res):
            eta, W, L = truth[i]
            k = sets[i].split == 2
            sp.append(spearmanr(e, W[k] / np.maximum(W[k] + L[k], 1)).statistic)
        print(f"  {name:30s} val nll/game {v0:.4f} -> {v1:.4f}; test r(true deck) {np.round(r, 3)}, "
              f"Spearman(WR) {np.round(sp, 3)} [{time.time() - t1:.1f}s]", flush=True)
        assert v1 < v0
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        torch.set_num_threads(2)
        d, L, B = map(int, sys.argv[2:5])
        n, sec = timing(d, L, B)
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        print(f"d={d} layers={L} batch={B}: {n / 1e6:.2f}M trainable params, {sec * 1000:.0f} ms/step (CPU, 2 threads), "
              f"peak RSS {rss:.0f} MB")
    else:
        selftest()
