"""Candidate B in production form: a set transformer over the deck's cards on top of the champion game FM, scored on
SC-GLL against a champion refit in the same process.

    python scripts/game_transformer.py MSH [SET ...] [--device mps] [--init picknet|random] [--distill W] [--seeds S]
           [--epochs E] [--context raw|dl] [--warm champion|none] [--freeze-fm] [--pooled] [--d 128] [--layers 3]
           [--batch 512] [--lr 1e-3] [--threads 4] [--boot 500] [--tag NAME]
    python scripts/game_transformer.py --selftest            (synthetic, CPU 2 threads, ~60-90 s)
    python scripts/game_transformer.py --mps-smoke           (one ~10 s timing of a production-size step on MPS)
    python scripts/sc_gll.py meta --file game_transformer/results_picknet_a0.json MSH FDN BLB ...

Design B (results/methodology/next_model_design.md) adapted to the findings in results/research_log.md:
  * likelihood   per-game binary log loss (iid), with the champion's game context: play/draw, mulligan difference,
                 raw skill bucket + known, rank, experience (500/1000 fix), record before the game (game_fm_v2's
                 `champion`); --context dl swaps the raw bucket for the de-leaked bucket (game_fm_v2 `rated-dl`).
                 No structural matchmaking term: model A changed scale, not ranking.
  * deck score   eta_b = [ w . x_b + FM pairs (nonland, rank 4) + f . F_b ]   +   head(SetTransformer(tokens_b))
                 the bracket is the champion DeckFM (game_fm_v2.fit_fm's model; F_b = Scryfall deck features +
                 affinity column, as context columns of the FM, exactly as the champion); --warm champion (default)
                 copies the champion's fitted state of the same seed into it, so at step 0 the model IS the champion
                 and the transformer, whose readout is zero-initialised, only learns what the FM misses.
                 tokens: one per distinct card of the build (lands incl., <= 48): h_c = W LN(z_c) + delta_c + copies[n_c]
                 with z_c the frozen PickNet all_d256 card vector (results/<SET>/picknet/feats.npz card_vec) and
                 delta_c a learned residual (init 0, L2 1e-4); --init random: delta_c ~ N(0, 0.02), no z. Plus a
                 [CLS] token and a deck-feature token (MLP of F_b). Pre-norm self-attention x L (d, 4 heads, FF 2d,
                 dropout 0.1), PMA pooling (4 seeds) + a copy-weighted sum of the card states (DeepSets; attention
                 pooling alone is an average and could not learn count thresholds in the selftest), MLP readout (last
                 layer zero-init).
  * distill W    W * mean((a eta_main + c - t_i)^2) / var_train(t) over training drafts' main builds, t_i = the
                 cross-fitted round-4 game logit (deck_synergy_v4/draft_inputs.npz game_logit); (a, c) learned.
  * training     batches of builds (all their training games come along; one transformer pass per build), AdamW
                 (FM path Adam-like, no decay, the champion's explicit L2 at its val-chosen setting), early stopping on
                 val game log loss with the champion's schedule (patience 3, 2 lr drops back to the best state, /3),
                 evaluated before the first step too (so the result can never be worse on val than the warm start).
                 Seeds: --seeds S trains S transformers (seed k warm-started from the champion's seed k).
  * pooled       --pooled: one trunk ([CLS], copies, feature token, attention, PMA) shared over the given sets, per-set
                 card encoders / readouts / FM paths; batches are single-set, each epoch visits every set's training
                 builds once in a random interleaving; early stopping on the pooled val log loss. RAM grows with the
                 sets loaded at once (<= 3 large sets under 3 GB).
Evaluation = game_fm_v2's: the champion refit here (game_fm_v2 grid + seeds; fit_fm_b is game_fm_v2.fit_fm with the
deck rows gathered from the build table, selftest-checked identical), sc_gll's frozen context model (and the dl-ctx
twin), (a, b) on val games, test games scored; subsets experienced (primary), all, match 1, main build, experienced
dl-ctx, match 1 dl-ctx; paired day-block bootstraps by seed. Models: champion, transformer, `transformer FM path`
(the jointly trained FM part alone: tells whether a gain is the transformer or more FM training), cached game FM.
Pairs: transformer - champion, transformer FM path - champion, transformer - transformer FM path, champion - cached.
Secondary: game_fm_v2.secondary (calibrated / raw / stratified draft Spearman, card-value r, score sd) plus residual
diagnostics. Writes results/{SET}.PremierDraft/game_transformer/{results,scores,losses}_<tag>.npz/json
(tag = <init>_a<W>[_dl][_pooled][_cold][_frozen]), readable by `sc_gll.py meta --file game_transformer/results_<tag>.json`.
Memory: the game file is scanned column-projected (eval_common); deck counts live in a build table (int8) and every
game indexes its build; per-game arrays are the context matrix only. Do not import lightgbm in this process.
"""
import os
import sys

_T = "2" if any(a in sys.argv for a in ("--selftest", "--mps-smoke")) else next(
    (sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--threads"), "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import game_fm_v2 as gfm  # noqa: E402
from eval_common import NARROW, day_blocks  # noqa: E402
from sc_gll import fit_context, subsets  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import N_RANKS  # noqa: E402
from decks.synergy import DeckFM  # noqa: E402

MAX_TOK, MAX_COPIES = 48, 20
SUM_SCALE = 40.0                  # cards in a deck
CHAMP, TF, TF_FM, CACHED = "champion", "transformer", "transformer FM path", "cached game FM"
CFG = dict(d=128, layers=3, heads=4, ff_mult=2, k_seeds=4, dropout=0.1, lr=1e-3, lr_fm=1e-3, lr_fm_cold=3e-3,
           wd=0.01, l2_delta=1e-4, batch=512, eval_batch=4096, epochs=30, patience=3, lr_drops=2, clip=1.0)
OUT = "game_transformer"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # bytes on macOS


# ---------------------------------------------------------------- champion fit on the build table

def fit_fm_b(Xb, gb, c, y, tr, va, l2w, l2v, vmask, seed=0, fit=gfm.FIT):
    """game_fm_v2.fit_fm (no weights / replay) with game g's deck row gathered as Xb[gb[g]] per batch instead of a
    dense [G, C] matrix: numerically identical (selftest), ~G x C x 4 bytes less RAM."""
    torch.manual_seed(seed)
    m = DeckFM(Xb.shape[1], c.shape[1], fit["k"], vmask=vmask)
    opt = torch.optim.Adam(m.parameters(), lr=fit["lr"])
    Ct, Yt = torch.from_numpy(c), torch.from_numpy(y.astype(np.float32))
    X = lambda b: torch.from_numpy(Xb[gb[b]].astype(np.float32))  # noqa: E731
    rng = np.random.default_rng(seed)

    def val_loss():
        m.eval()
        with torch.no_grad():
            p = torch.cat([torch.sigmoid(m(X(b), Ct[b])) for b in np.array_split(va, max(1, len(va) // 8192))])
        return log_loss(y[va], p.numpy())

    best, state, bad, drops, events = np.inf, None, 0, fit["lr_drops"], []
    for ep in range(fit["epochs"]):
        m.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // fit["batch"])):
            loss = F.binary_cross_entropy_with_logits(m(X(b), Ct[b]), Yt[b]) + m.penalty(l2w, l2v, 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_loss()
        if v < best - 1e-6:
            best, bad, ev = v, 0, "best"
        else:
            bad, ev = bad + 1, "-"
            if bad >= fit["patience"]:
                ev = "stop" if drops <= 0 else "drop"
                if ev == "drop":
                    drops, bad = drops - 1, 0
        if ev == "best":
            state = {n: t.detach().clone() for n, t in m.state_dict().items()}
        elif ev == "drop":
            m.load_state_dict(state)
            for gr in opt.param_groups:
                gr["lr"] /= 3
        events.append(ev)
        if ev == "stop":
            break
    m.load_state_dict(state)
    m.eval()
    return m, best, events


def fit_champion(s, seeds, fit=gfm.FIT, log=print):
    """game_fm_v2.fit_variants' champion arm: grid on seed 0, chosen setting for seeds 1..S-1."""
    t1 = time.time()
    tr, va = s.games_of_split(0), s.games_of_split(1)
    grid = [fit_fm_b(s.Xb, s.gb, s.c, s.y, tr, va, *kw, s.nonland, seed=0, fit=fit) + (kw,) for kw in gfm.GRID]
    m0, vbest, ev0, kw = min(grid, key=lambda f: f[1])
    fits = [(m0, vbest, ev0)] + [fit_fm_b(s.Xb, s.gb, s.c, s.y, tr, va, *kw, s.nonland, seed=k, fit=fit)
                                 for k in range(1, seeds)]
    log(f"  [{s.name}] champion val {vbest:.5f} {tuple(kw)} epochs {[len(f[2]) for f in fits]} "
        f"[{time.time() - t1:.0f}s]")
    return {"models": [f[0] for f in fits], "setting": list(kw), "val logloss per seed": [float(f[1]) for f in fits],
            "epochs": [len(f[2]) for f in fits], "events": [f[2] for f in fits], "fit seconds": time.time() - t1}


# ---------------------------------------------------------------- the deck transformer

class Trunk(nn.Module):
    """Shared part: [CLS], copies embedding, deck-feature token, L pre-norm self-attention blocks, PMA pooling."""

    def __init__(self, n_feat, d=128, layers=3, heads=4, ff_mult=2, k_seeds=4, dropout=0.1):
        super().__init__()
        self.d, self.k = d, k_seeds
        self.copies = nn.Embedding(MAX_COPIES + 1, d)
        nn.init.normal_(self.copies.weight, std=0.02)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.feat = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d)) if n_feat else None
        layer = nn.TransformerEncoderLayer(d, heads, ff_mult * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.seeds = nn.Parameter(torch.randn(k_seeds, d) * 0.02)
        self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.out_dim = (k_seeds + 1) * d

    def forward(self, h, cnt, pad, feat):
        """h [B, M, d] card embeddings, cnt [B, M] copies, pad [B, M] bool, feat [B, F] -> pooled [B, (k + 1) d]:
        PMA (attention pooling: composition) and the copy-weighted sum of the card states / 40 (DeepSets: counts, so
        the readout MLP can threshold on e.g. the number of two-drops; attention pooling is an average)."""
        B = len(h)
        h = h + self.copies(cnt.clamp(max=MAX_COPIES))
        extra = [self.cls.expand(B, 1, -1)] + ([self.feat(feat)[:, None]] if self.feat is not None else [])
        h = torch.cat(extra + [h], 1)
        pad = torch.cat([torch.zeros(B, len(extra), dtype=torch.bool, device=h.device), pad], 1)
        h = self.enc(h, src_key_padding_mask=pad)
        o, _ = self.pma(self.seeds[None].expand(B, -1, -1), h, h, key_padding_mask=pad, need_weights=False)
        n = len(extra)
        w = (cnt.float() * (~pad[:, n:]).float())[..., None] / SUM_SCALE   # copy-weighted sum over the cards
        tot = (h[:, n:] * w).sum(1)
        return torch.cat([self.ln(o).flatten(1), tot], 1)


class SetCards(nn.Module):
    """Per-set card embeddings: W LN(z_c) + delta_c (PickNet init, z frozen) or delta_c alone (random init)."""

    def __init__(self, n_cards, d, base=None):
        super().__init__()
        self.pad = n_cards
        self.delta = nn.Embedding(n_cards + 1, d, padding_idx=n_cards)
        if base is not None:
            z = torch.as_tensor(np.asarray(base, np.float32))
            self.register_buffer("base", torch.cat([z, z.new_zeros(1, z.shape[1])]))
            self.norm = nn.LayerNorm(z.shape[1])
            self.proj = nn.Linear(z.shape[1], d)
            nn.init.zeros_(self.delta.weight)
        else:
            self.base = None
            nn.init.normal_(self.delta.weight, std=0.02)
            with torch.no_grad():
                self.delta.weight[n_cards].zero_()

    def forward(self, ids):
        h = self.delta(ids)
        if self.base is not None:
            h = h + self.proj(self.norm(self.base[ids]))
        return h


class SetModel(nn.Module):
    """One set: the FM path (a DeckFM, the champion's model class), the card encoder, the transformer readout and the
    distillation affine map. Game logit = fm.deck_score(x_b) + resid_b + fm.cov(c_g)."""

    def __init__(self, n_cards, n_ctx, n_feat, vmask, d, k_seeds, base=None, k_fm=4):
        super().__init__()
        self.n_feat = n_feat
        self.fm = DeckFM(n_cards, n_ctx, k_fm, vmask=vmask)
        self.cards = SetCards(n_cards, d, base)
        self.head = nn.Sequential(nn.Linear((k_seeds + 1) * d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)
        self.dis = nn.Parameter(torch.tensor([1.0, 0.0]))

    def resid(self, trunk, ids, cnt, feat):
        pad = ids == self.cards.pad
        return self.head(trunk(self.cards(ids), cnt, pad, feat)).squeeze(-1)

    def build_scores(self, trunk, xb, ids, cnt, feat):
        """(FM card + pair score [B] (no feature terms), transformer residual [B])."""
        return self.fm.deck_score(xb), self.resid(trunk, ids, cnt, feat)

    def feat_score(self, feat):
        return feat @ self.fm.cov.weight[0, -self.n_feat:]

    def penalty(self, l2w, l2v, l2_delta):
        p = self.fm.penalty(l2w, l2v, 0)
        if self.cards.base is not None and l2_delta > 0:
            p = p + l2_delta * self.cards.delta.weight.pow(2).sum()
        return p


# ---------------------------------------------------------------- data

class SetData:
    """One set's arrays, build-indexed. Required fields (load_set / synthetic): name, d, pg, has, main_bix, main [D]
    (build row of the most-played build, -1 = none), Xb [B, C] int8, gb [G] build of each game, bdraft [B], Eb [B, F]
    standardised deck features + affinity, cov / exp / rec [G, .], nonland [C], played [C], card_vec [C, 256] or None,
    cached [D] or None (round-4 game logit = the distillation teacher), info."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def prepare(self, context="raw"):
        """Context matrix (game_fm_v2.context_cols of `champion` or `rated-dl`; deck-feature columns last), tokens,
        build -> games CSR, build split / main flags, teacher per build."""
        d, pg = self.d, self.pg
        D = len(d.split)
        self.gsp, self.gd = pg["split"], pg["gd"]
        self.bt_game, self.bt_draft = gfm.deleaked_per_game(pg, D)
        dd = {"cov": self.cov, "exp": self.exp, "rec": self.rec, "E": self.Eb[self.gb], "pg": pg, "d": d,
              "bt_game": self.bt_game, "bt_draft": self.bt_draft}
        self.variant = {"raw": CHAMP, "dl": "rated-dl"}[context]
        self.c = gfm.context_cols(self.variant, dd)
        self.info = {**self.info, **dd.get("info", {})}
        del self.cov, self.exp, self.rec, dd
        self.n_feat = self.Eb.shape[1]
        B = len(self.Xb)
        nz = self.Xb > 0
        self.ntok = np.minimum(nz.sum(1), MAX_TOK).astype(np.int16)
        self.tok = np.full((B, MAX_TOK), -1, np.int16)
        self.tcnt = np.zeros((B, MAX_TOK), np.int8)
        r, col = np.nonzero(nz)                               # row-major: per build, cards in column order
        k = np.arange(len(r)) - np.r_[0, np.cumsum(nz.sum(1))][r]
        ok = k < MAX_TOK
        self.tok[r[ok], k[ok]] = col[ok]
        self.tcnt[r[ok], k[ok]] = np.minimum(self.Xb[r[ok], col[ok]], 127)
        self.y = self.y.astype(np.int8)
        self.yf = self.y.astype(np.float32)
        self.order = np.argsort(self.gb, kind="stable")       # games grouped by build
        self.gptr = np.r_[0, np.cumsum(np.bincount(self.gb, minlength=B))]
        n_g = np.diff(self.gptr)
        self.bsplit = np.where(n_g > 0, d.split[self.bdraft], -1)
        self.is_main = np.zeros(B, bool)
        self.is_main[self.main[self.main >= 0]] = True
        t = np.full(B, np.nan, np.float32)
        if self.cached is not None:
            ok = self.main >= 0
            t[self.main[ok]] = self.cached[ok]
        self.teacher = np.where(self.is_main & (self.bsplit == 0), t, np.nan).astype(np.float32)
        tt = self.teacher[np.isfinite(self.teacher)]
        self.tvar = float(tt.var()) if len(tt) > 1 else 1.0
        self.builds = {k: np.flatnonzero(self.bsplit == k) for k in (0, 1, 2)}
        return self

    def games_of_split(self, k):
        return np.flatnonzero(self.gsp == k)

    def games_of(self, bidx):
        """Games of the given builds (grouped by build) and each game's position in bidx."""
        lo, n = self.gptr[bidx], np.diff(self.gptr)[bidx]
        pos = np.repeat(np.arange(len(bidx)), n)
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        return self.order[lo[pos] + off], pos

    def batch(self, bidx, dev, games=True):
        m = int(max(1, self.ntok[bidx].max()))
        ids = torch.from_numpy(self.tok[bidx, :m].astype(np.int64))
        ids = torch.where(ids < 0, self.Xb.shape[1], ids)
        out = {"ids": ids.to(dev), "cnt": torch.from_numpy(self.tcnt[bidx, :m].astype(np.int64)).to(dev),
               "xb": torch.from_numpy(self.Xb[bidx].astype(np.float32)).to(dev),
               "feat": torch.from_numpy(self.Eb[bidx].astype(np.float32)).to(dev)}
        if games:
            g, pos = self.games_of(bidx)
            out.update(c=torch.from_numpy(self.c[g]).to(dev), y=torch.from_numpy(self.yf[g]).to(dev),
                       pos=torch.from_numpy(pos).to(dev))
        return out


# ---------------------------------------------------------------- training

def game_loss(sm, trunk, b):
    """Summed BCE over the batch's games, the batch's deck-only scores and game count."""
    s_fm, r = sm.build_scores(trunk, b["xb"], b["ids"], b["cnt"], b["feat"])
    logit = (s_fm + r)[b["pos"]] + sm.fm.cov(b["c"])[:, 0]
    bce = F.binary_cross_entropy_with_logits(logit, b["y"], reduction="sum")
    return bce, s_fm + sm.feat_score(b["feat"]) + r, len(b["y"])


@torch.no_grad()
def val_loss(sets, models, trunk, dev, cfg, split=1):
    trunk.eval()
    tot, n, per = 0.0, 0, []
    for s, sm in zip(sets, models):
        sm.eval()
        t_s, n_s = 0.0, 0
        bl = s.builds[split]
        for a in range(0, len(bl), cfg["eval_batch"]):
            bce, _, k = game_loss(sm, trunk, s.batch(bl[a:a + cfg["eval_batch"]], dev))
            t_s, n_s = t_s + float(bce), n_s + k
        per.append(t_s / max(n_s, 1))
        tot, n = tot + t_s, n + n_s
    return tot / max(n, 1), per


@torch.no_grad()
def deck_scores(s, sm, trunk, dev, cfg):
    """Per draft: (full deck-only score, FM-path deck-only score, transformer residual) of the main build; 0 for
    drafts without games."""
    trunk.eval()
    sm.eval()
    D = len(s.d.split)
    out = np.zeros((3, D))
    has = np.flatnonzero(s.main >= 0)
    for a in range(0, len(has), cfg["eval_batch"]):
        di = has[a:a + cfg["eval_batch"]]
        b = s.batch(s.main[di], dev, games=False)
        s_fm, r = sm.build_scores(trunk, b["xb"], b["ids"], b["cnt"], b["feat"])
        f = s_fm + sm.feat_score(b["feat"])
        out[:, di] = torch.stack([f + r, f, r]).float().cpu().double().numpy()
    return out


def train(sets, champs, seed, cfg, init="picknet", distill=0.0, warm=True, freeze_fm=False, dev="cpu", log=print):
    """One seed: a shared trunk + one SetModel per set; returns (trunk, models, info)."""
    torch.manual_seed(1000 + seed)
    rng = np.random.default_rng(seed)
    trunk = Trunk(sets[0].n_feat, cfg["d"], cfg["layers"], cfg["heads"], cfg["ff_mult"], cfg["k_seeds"],
                  cfg["dropout"]).to(dev)
    models = []
    for s, ch in zip(sets, champs):
        sm = SetModel(s.Xb.shape[1], s.c.shape[1], s.n_feat, s.nonland, cfg["d"], cfg["k_seeds"],
                      base=s.card_vec if init == "picknet" else None)
        if warm:
            sm.fm.load_state_dict(ch["models"][min(seed, len(ch["models"]) - 1)].state_dict())
        if freeze_fm:
            for p in sm.fm.parameters():
                p.requires_grad_(False)
        models.append(sm.to(dev))
    fm_p = [p for sm in models for p in sm.fm.parameters() if p.requires_grad]
    proj = [p for sm in models if sm.cards.base is not None for p in sm.cards.proj.parameters()]
    skip = {id(p) for p in fm_p + proj}
    rest = [p for p in list(trunk.parameters()) + [p for sm in models for p in sm.parameters()]
            if id(p) not in skip and p.requires_grad]
    groups = [{"params": rest, "lr": cfg["lr"], "weight_decay": cfg["wd"]}]
    if proj:
        groups.append({"params": proj, "lr": 0.3 * cfg["lr"], "weight_decay": cfg["wd"]})
    if fm_p:
        groups.append({"params": fm_p, "lr": cfg["lr_fm"] if warm else cfg["lr_fm_cold"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    net_p = [p for g in groups[:len(groups) - bool(fm_p)] for p in g["params"]]
    settings = [ch["setting"] for ch in champs]

    def snapshot():
        return [{n: t.detach().clone() for n, t in m.state_dict().items()} for m in [trunk] + models]

    def restore(st):
        for m, x in zip([trunk] + models, st):
            m.load_state_dict(x)

    t0 = time.time()
    v0, per0 = val_loss(sets, models, trunk, dev, cfg)
    best, state, bad, drops, events, hist = v0, snapshot(), 0, cfg["lr_drops"], ["init"], [v0]
    log(f"    seed {seed}: val {v0:.5f} at start " + " ".join(f"{s.name} {v:.5f}" for s, v in zip(sets, per0)))
    for ep in range(cfg["epochs"]):
        trunk.train()
        for sm in models:
            sm.train()
        chunks = [(k, c) for k, s in enumerate(sets)
                  for c in np.array_split(rng.permutation(s.builds[0]), max(1, len(s.builds[0]) // cfg["batch"]))]
        for j in rng.permutation(len(chunks)):
            k, bidx = chunks[j]
            s, sm = sets[k], models[k]
            b = s.batch(bidx, dev)
            bce, sdeck, n = game_loss(sm, trunk, b)
            loss = bce / n + sm.penalty(*settings[k], cfg["l2_delta"])
            if distill > 0:
                t = s.teacher[bidx]
                ok = np.isfinite(t)
                if ok.any():
                    okt = torch.from_numpy(ok).to(dev)
                    z = sm.dis[0] * sdeck[okt] + sm.dis[1]
                    loss = loss + distill * ((z - torch.from_numpy(t[ok].astype(np.float32)).to(dev)) ** 2).mean() / s.tvar
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net_p, cfg["clip"])
            opt.step()
        v, per = val_loss(sets, models, trunk, dev, cfg)
        hist.append(v)
        if v < best - 1e-6:
            best, bad, ev = v, 0, "best"
            state = snapshot()
        else:
            bad, ev = bad + 1, "-"
            if bad >= cfg["patience"]:
                ev = "stop" if drops <= 0 else "drop"
                if ev == "drop":
                    drops, bad = drops - 1, 0
                    restore(state)
                    for gr in opt.param_groups:
                        gr["lr"] /= 3
        events.append(ev)
        log(f"    seed {seed} epoch {ep:2d} val {v:.5f} ({v - v0:+.5f}) {ev} "
            + " ".join(f"{s.name} {x:.5f}" for s, x in zip(sets, per)) + f" [{time.time() - t0:.0f}s]")
        if ev == "stop":
            break
    restore(state)
    _, per_best = val_loss(sets, models, trunk, dev, cfg)
    info = {"val logloss at start": float(v0), "val logloss best": float(best), "val per set": per_best,
            "val per set at start": per0, "events": events, "epochs": len(events) - 1,
            "best epoch": int(np.argmin(hist)), "val history": [float(x) for x in hist],
            "seconds": time.time() - t0}
    return trunk, models, info


# ---------------------------------------------------------------- evaluation (game_fm_v2's)

def eval_set(s, champ, tf_scores, tf_w, tf_info, n_boot, log=print):
    """SC-GLL + secondary for one set. tf_scores [3, S, D] (full, FM path, residual)."""
    d, pg = s.d, s.pg
    D = len(d.split)
    fm_ch = np.stack([gfm.deck_logit(m, deck_rows_f32(s), ed(s)) for m in champ["models"]])
    scores = {CHAMP: fm_ch, TF: tf_scores[0], TF_FM: tf_scores[1]}
    if s.cached is not None:
        scores[CACHED] = s.cached[None]
    pairs = [(CHAMP, TF), (CHAMP, TF_FM), (TF_FM, TF)] + ([(CACHED, CHAMP)] if s.cached is not None else [])
    ctx, cinfo = fit_context(pg, pg["split"] == 0)
    ctx_dl, cinfo_dl = fit_context({**pg, "skill": s.bt_game}, pg["split"] == 0)
    subs_all = subsets(pg, s.has, s.main_bix)
    subs = {"": subs_all, " dl-ctx": {k: v for k, v in subs_all.items() if k in gfm.DL_SUBSETS}}
    ev, losses, ngames = gfm.evaluate(scores, pg, {"": ctx, " dl-ctx": ctx_dl}, subs, D, pairs, {}, n_boot,
                                      day_blocks(d), s.name, log)
    fits = {CHAMP: {"scores": fm_ch, "w": champ["models"][0].w.detach().numpy().copy()},
            TF: {"scores": tf_scores[0], "w": tf_w}, TF_FM: {"scores": tf_scores[1], "w": tf_w}}
    sec = gfm.secondary(fits, {"d": d, "has": s.has, "pg": pg, "played": s.played}, s.cached)
    te = (d.split == 2) & s.has
    diag = {"residual sd (test drafts)": float(np.mean([r[te].std() for r in tf_scores[2]])),
            "FM-path score sd (test drafts)": float(np.mean([r[te].std() for r in tf_scores[1]])),
            "r(residual, champion score) test": float(np.mean([np.corrcoef(r[te], fm_ch[0][te])[0, 1]
                                                               for r in tf_scores[2]])),
            "r(transformer, champion) test": float(np.mean([np.corrcoef(r[te], fm_ch[0][te])[0, 1]
                                                            for r in tf_scores[0]]))}
    if s.cached is not None:
        diag["r(transformer, teacher) test"] = float(np.mean([np.corrcoef(r[te], s.cached[te])[0, 1]
                                                              for r in tf_scores[0]]))
    for v, o in sec.items():
        log(f"  {v:20s} draft spearman {o['draft spearman (calibrated)']:.4f} "
            f"({o['draft spearman (calibrated) - champion']:+.4f})  raw {o['raw spearman - champion']:+.4f}  "
            f"strat {o['stratified spearman - champion']:+.4f}  sd {o['deck score sd (test drafts)']:.3f}")
    log("  " + "  ".join(f"{k} {v:.3f}" for k, v in diag.items()))
    res = {"context": cinfo, "context dl": cinfo_dl, "primary subset": "experienced",
           "champion fit": {k: v for k, v in champ.items() if k != "models"}, "transformer fits": tf_info,
           **ev, "secondary": sec, "diagnostics": diag}
    return res, scores, losses, ngames


def deck_rows_f32(s):
    out = np.zeros((len(s.main), s.Xb.shape[1]), np.float32)
    ok = s.main >= 0
    out[ok] = s.Xb[s.main[ok]]
    return out


def ed(s):
    out = np.zeros((len(s.main), s.n_feat), np.float32)
    ok = s.main >= 0
    out[ok] = s.Eb[s.main[ok]]
    return out


def run(sets, cfg, seeds=1, init="picknet", distill=0.0, warm=True, freeze_fm=False, dev="cpu", n_boot=gfm.N_BOOT,
        fit=gfm.FIT, log=print):
    """Champion refit + transformer per seed (one trunk over `sets`), then per-set evaluation.
    Returns {set name: (res, scores, losses, ngames)}."""
    champs = [fit_champion(s, seeds, fit, log) for s in sets]
    S = {s.name: np.zeros((3, seeds, len(s.d.split))) for s in sets}
    W, infos = {}, {s.name: [] for s in sets}
    for k in range(seeds):
        trunk, models, info = train(sets, champs, k, cfg, init, distill, warm, freeze_fm, dev, log)
        for j, (s, sm) in enumerate(zip(sets, models)):
            S[s.name][:, k] = deck_scores(s, sm, trunk, dev, cfg)
            if k == 0:
                W[s.name] = sm.fm.w.detach().cpu().numpy().copy()
            infos[s.name].append({**{a: b for a, b in info.items() if a not in ("val per set", "val per set at start")},
                                  "val logloss (this set)": info["val per set"][j],
                                  "val logloss at start (this set)": info["val per set at start"][j],
                                  "champion val logloss (this seed)": champs[j]["val logloss per seed"][
                                      min(k, len(champs[j]["models"]) - 1)]})
        del trunk, models
        if dev == "mps":
            torch.mps.empty_cache()
    out = {}
    for s, ch in zip(sets, champs):
        log(f"  [{s.name}] SC-GLL")
        out[s.name] = eval_set(s, ch, S[s.name], W[s.name], infos[s.name], n_boot, log)
    return out


# ---------------------------------------------------------------- real data

def load_set(set_code):
    """Column-projected game scan (eval_common), build table, deck features + affinity per build (standardised over
    training games, i.e. weighted by each build's training games, as game_fm_v2 does per game), PickNet vectors,
    the cached round-4 game logit."""
    import polars as pl
    from eval_common import build_table, load_games, main_builds, per_game
    from mtgdata.cardinfo import card_info, deck_features
    from mtgdata.games import game_path
    from decks.deckstrength import covariates, experience, load_drafts, record_before, record_onehot
    t0 = time.time()
    d = load_drafts(set_code, "PremierDraft")
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id, (*NARROW, "opp_rank"))
    if "opp_rank" not in g.columns:
        g = g.with_columns(pl.lit(None, pl.Utf8).alias("opp_rank"))
    pg = per_game(d, g)
    Xb, gb, bdraft, main = build_table(path, d, g)
    Xb = Xb.astype(np.int8)
    has, _, main_bix = main_builds(d, g)
    assert ((main >= 0) == has).all()
    wb, lb, _ = record_before(g)
    cov = covariates(g)[:, : 4 + N_RANKS].astype(np.float32)
    exp_, rec = experience(g), record_onehot(wb, lb)
    y = (pg["y"] > 0.5).astype(np.int8)
    del g
    gsp = pg["split"]
    ntr = np.bincount(gb[gsp == 0], minlength=len(Xb)).astype(np.float64)    # training games per build
    info = card_info(d.names, set_code)
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    Fb = np.vstack([deck_features(Xb[a:a + 50_000].astype(np.float32), info)[0] for a in range(0, len(Xb), 50_000)])
    mu = (Fb * ntr[:, None]).sum(0) / ntr.sum()
    sd = np.sqrt((((Fb - mu) ** 2) * ntr[:, None]).sum(0) / ntr.sum()) + 1e-6
    cols = [(Fb - mu) / sd]
    del Fb
    aff = gfm.pick_aff(set_code, d.names)
    if aff is None:
        raise SystemExit(f"{set_code}: no picknet pool model (the champion needs the affinity column)")
    cols.append(weighted_profile(Xb, aff[0], nonland, ntr))
    Eb = np.hstack(cols).astype(np.float32)
    played = np.zeros(Xb.shape[1])
    for a in range(0, len(Xb), 50_000):
        played += ntr[a:a + 50_000] @ Xb[a:a + 50_000].astype(np.float32)
    cached = None
    inp = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4" / "draft_inputs.npz"
    if inp.exists():
        z = np.load(inp)
        if len(z["has_deck"]) == len(has) and (z["has_deck"] == has).all():
            cached = z["game_logit"].astype(np.float64)
    card_vec, have = picknet_vectors(set_code, d.names)
    print(f"=== {set_code}: {len(d.draft_id)} drafts, {len(y)} games (train {np.sum(gsp == 0)}, val {np.sum(gsp == 1)},"
          f" test {np.sum(gsp == 2)}), {len(Xb)} builds, {Xb.shape[1]} cards, PickNet vectors for {have.sum()}, "
          f"teacher {'yes' if cached is not None else 'NO'} [{time.time() - t0:.0f}s, peak RSS {rss_gb():.2f} GB]",
          flush=True)
    return SetData(name=set_code, d=d, pg=pg, has=has, main_bix=main_bix, main=main, Xb=Xb, gb=gb, bdraft=bdraft,
                   Eb=Eb, cov=cov, exp=exp_, rec=rec, y=y, nonland=nonland, played=played > gfm.PLAYED_MIN,
                   card_vec=card_vec, cached=cached,
                   info={"affinity column": True, "picknet vectors": int(have.sum()), "cards": int(Xb.shape[1]),
                         "builds": int(len(Xb))})


def weighted_profile(Xb, S, nonland, w):
    """deck_synergy_v3.Profile's `aff mean` column (standardised on training games) for build rows weighted by their
    training games (the off-plan quantile, unused by the aff-mean column, is skipped)."""
    from deck_synergy_v3 import card_fit
    out = np.empty(len(Xb), np.float32)
    for a in range(0, len(Xb), 50_000):
        x = Xb[a:a + 50_000].astype(np.float32)
        fit_, Xn = card_fit(x, S, nonland)
        out[a:a + 50_000] = (Xn * fit_).sum(1) / np.maximum(Xn.sum(1), 1)
    mu = (out * w).sum() / w.sum()
    sd = np.sqrt((((out - mu) ** 2) * w).sum() / w.sum()) + 1e-6
    return ((out - mu) / sd)[:, None].astype(np.float32)


def picknet_vectors(set_code, names):
    f = np.load(ROOT / "results" / f"{set_code}.PremierDraft" / "picknet" / "feats.npz", allow_pickle=True)
    vec = dict(zip(f["card_names"].tolist(), f["card_vec"]))
    out = np.zeros((len(names), f["card_vec"].shape[1]), np.float32)
    have = np.array([n in vec for n in names])
    for i, n in enumerate(names):
        if n in vec:
            out[i] = vec[n]
    return out, have


# ---------------------------------------------------------------- CLI

def parse(argv):
    o = dict(sets=[], device="cpu", init="picknet", distill=0.0, seeds=1, epochs=None, context="raw", warm=True,
             freeze_fm=False, pooled=False, threads=int(_T), boot=gfm.N_BOOT, tag=None, cfg={})
    it = iter(argv)
    num = {"--d": ("d", int), "--layers": ("layers", int), "--batch": ("batch", int), "--lr": ("lr", float),
           "--lr-fm": ("lr_fm", float), "--dropout": ("dropout", float)}
    for a in it:
        if a == "--device":
            o["device"] = next(it)
        elif a == "--init":
            o["init"] = next(it)
        elif a == "--distill":
            o["distill"] = float(next(it))
        elif a == "--seeds":
            o["seeds"] = int(next(it))
        elif a == "--epochs":
            o["epochs"] = int(next(it))
        elif a == "--context":
            o["context"] = next(it)
        elif a == "--warm":
            o["warm"] = next(it) == "champion"
        elif a == "--freeze-fm":
            o["freeze_fm"] = True
        elif a == "--pooled":
            o["pooled"] = True
        elif a == "--threads":
            o["threads"] = int(next(it))
        elif a == "--boot":
            o["boot"] = int(next(it))
        elif a == "--tag":
            o["tag"] = next(it)
        elif a in num:
            k, f = num[a]
            o["cfg"][k] = f(next(it))
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}")
        else:
            o["sets"].append(a)
    assert o["init"] in ("picknet", "random") and o["context"] in ("raw", "dl"), o
    if o["epochs"] is not None:
        o["cfg"]["epochs"] = o["epochs"]
    if o["tag"] is None:
        o["tag"] = (f"{o['init']}_a{o['distill']:g}" + ("_dl" if o["context"] == "dl" else "")
                    + ("_pooled" if o["pooled"] else "") + ("" if o["warm"] else "_cold")
                    + ("_frozen" if o["freeze_fm"] else ""))
    return o


def main(argv):
    if "lightgbm" in sys.modules:
        raise RuntimeError("lightgbm is loaded in this process; run game_transformer.py in its own process")
    o = parse(argv)
    torch.set_num_threads(o["threads"])
    dev = o["device"]
    if dev == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    cfg = {**CFG, **o["cfg"]}
    groups = [o["sets"]] if o["pooled"] else [[s] for s in o["sets"]]
    for grp in groups:
        t0 = time.time()
        sets = [load_set(s).prepare(o["context"]) for s in grp]
        print(f"  prepared {grp} [peak RSS {rss_gb():.2f} GB]", flush=True)
        out = run(sets, cfg, o["seeds"], o["init"], o["distill"], o["warm"], o["freeze_fm"], dev, o["boot"],
                  log=lambda m: print(m, flush=True))
        for s in sets:
            res, scores, losses, ngames = out[s.name]
            res = {"set": s.name, "games": int(len(s.y)), "config": {**{k: v for k, v in o.items() if k != "sets"},
                                                                     "cfg": cfg, "pooled sets": grp},
                   "context variant": s.variant, **s.info, **res, "total seconds": time.time() - t0,
                   "peak RSS GB": rss_gb()}
            gfm.save(ROOT / "results" / f"{s.name}.PremierDraft" / OUT, "_" + o["tag"], s.d, s.has, res, scores,
                     losses, ngames)
            p = res["pairs"]["transformer - champion"]["experienced"]
            print(f"{s.name} done: transformer - champion (experienced) {p['delta']:+.4f} ± {p['se']:.4f} "
                  f"[{time.time() - t0:.0f}s, peak RSS {rss_gb():.2f} GB]", flush=True)
        del sets, out


# ---------------------------------------------------------------- synthetic self-test

GROUP = np.arange(6, 14)          # "two-drops": the curve threshold counts these
TRIPLE = (14, 15, 16)             # a 3-card combo


def nonlinear(Xd, n_land=6):
    """Deck effects outside the FM's span: a curve / land-count mismatch (a low curve (>= 7 two-drops) wants <= 16
    lands, a high curve wants >= 17: thresholds on two counts, and lands are outside the FM's nonland pairs) and a
    3-card combo (all three present). Least squares on [counts, all nonland pairs, lands^2] explains ~5% / ~65%."""
    n2 = Xd[:, GROUP].sum(1)
    lands = Xd[:, -n_land:].sum(1)
    combo = (Xd[:, list(TRIPLE)] > 0).all(1)
    return 0.9 * ((n2 >= 7) == (lands <= 16)) + 0.5 * combo


def synthetic_set(name="SYN", D=12000, C=32, n_land=6, seed=0, multi=0.15):
    """Drafts with skill-correlated picks, a leaky skill bucket, record-matched 7-3 runs, an FM-capturable pair term,
    a land-count term (deck features), the nonlinear effect above, and second builds for 15% of drafts."""
    from decks.deckstrength import record_onehot
    rng = np.random.default_rng(seed)
    nl = C - n_land
    w_true = np.r_[rng.normal(0, 0.08, nl), np.zeros(n_land)]
    skill = rng.normal(0, 0.3, D)
    p_inc = expit(-0.4 + 0.9 * skill[:, None] * w_true[None, :nl] / 0.08)
    builds, bdraft = [], []
    for i in range(D):
        x = np.zeros(C, np.int8)
        x[:nl] = rng.binomial(2, p_inc[i])
        x[nl:] = rng.multinomial(15 + rng.integers(0, 4), np.ones(n_land) / n_land)
        builds.append(x)
        bdraft.append(i)
        if rng.random() < multi:
            x2 = x.copy()
            j = rng.choice(nl, 3, replace=False)
            x2[j] = rng.binomial(2, 0.5, 3)
            builds.append(x2)
            bdraft.append(i)
    Xb, bdraft = np.array(builds), np.array(bdraft)
    lands = Xb[:, nl:].sum(1)
    eta_lin = Xb @ w_true + 0.10 * (Xb[:, 0] * Xb[:, 1] + Xb[:, 2] * Xb[:, 3]) - 0.04 * (lands - 16.5) ** 2
    nlin = nonlinear(Xb)
    eta_b = eta_lin + nlin
    eta_b = eta_b - eta_b.mean()
    first = np.r_[0, np.flatnonzero(np.diff(bdraft)) + 1]
    second = {int(bdraft[b]): b for b in range(1, len(bdraft)) if bdraft[b] == bdraft[b - 1]}
    n_hist = rng.choice([5, 10, 50, 100, 500], D, p=[0.1, 0.1, 0.3, 0.3, 0.2])
    rank = rng.integers(2, 4, D)
    rows, W, L = [], np.zeros(D), np.zeros(D)
    for i in range(D):
        wv = lv = 0
        b, sw = first[i], (rng.integers(1, 5) if i in second else 99)
        while wv < 7 and lv < 3:
            if wv + lv == sw:
                b = second[i]
            op = rng.random() < 0.5
            mull = int(rng.random() < 0.1) - int(rng.random() < 0.1)
            win = rng.random() < expit(0.15 + eta_b[b] + skill[i] + 0.1 * (op - 0.5) - 0.12 * (wv - lv) - 0.3 * mull)
            rows.append((i, b, wv + lv + 1, wv, lv, op, mull, win))
            wv, lv = wv + win, lv + (not win)
        W[i], L[i] = wv, lv
    gd, gb, match, wb, lb, op, mull, y = (np.array(c) for c in zip(*rows))
    G = len(y)
    nb = np.bincount(gb, minlength=len(Xb))
    main = np.full(D, -1)
    for i in range(D):
        cand = [first[i]] + ([second[i]] if i in second else [])
        main[i] = max(cand, key=lambda b: (nb[b], -b))
    bix = np.array([str(b - first[bdraft[b]]) for b in range(len(Xb))], dtype=object)
    long_run = expit(0.15 + skill + rng.normal(0, 0.1, D))
    bucket = np.round(((n_hist - (W + L)).clip(1) * long_run + W) / np.maximum(n_hist, W + L) / 0.02) * 0.02
    split = np.zeros(D, np.int8)
    split[int(0.7 * D):], split[int(0.85 * D):] = 1, 2
    exp_b = np.searchsorted([1, 5, 10, 50, 100, 500, 1000], n_hist) + 1
    pg = {"gd": gd, "y": y.astype(float), "split": split[gd], "match": match, "known": np.ones(G, bool),
          "state": np.minimum(wb, 6) * 3 + np.minimum(lb, 2), "on_play": op - 0.5, "mull": mull.astype(float),
          "exp": np.eye(8)[exp_b[gd]], "rank": rank[gd], "n_bucket": n_hist[gd].astype(float),
          "skill": bucket[gd], "wins_before": wb, "losses_before": lb, "build_index": bix[gb]}
    d = SimpleNamespace(wins=W.astype(np.float32), losses=L.astype(np.float32), split=split, rank=rank,
                        skill=bucket.astype(np.float32), draft_id=np.array([f"{name}{i}" for i in range(D)]),
                        draft_time=np.array([f"2025-01-{1 + i * 28 // D:02d}" for i in range(D)]))
    cov = np.hstack([pg["on_play"][:, None], pg["mull"][:, None], (bucket[gd] - 0.5)[:, None], np.ones((G, 1)),
                     np.eye(N_RANKS)[rank[gd]]]).astype(np.float32)
    Fb = np.stack([lands, (lands - 16.5) ** 2, Xb[:, :nl].sum(1)], 1).astype(np.float64)
    ntr = np.bincount(gb[split[gd] == 0], minlength=len(Xb)).astype(float)
    mu = (Fb * ntr[:, None]).sum(0) / ntr.sum()
    sd = np.sqrt((((Fb - mu) ** 2) * ntr[:, None]).sum(0) / ntr.sum()) + 1e-6
    Eb = ((Fb - mu) / sd).astype(np.float32)
    # "PickNet" vectors: noisy views of card value and of the card's curve slot / combo membership (what a pick
    # model learns about a card), random elsewhere
    roles = np.stack([w_true / 0.08, np.isin(np.arange(C), GROUP), np.isin(np.arange(C), TRIPLE),
                      np.arange(C) >= nl], 1).astype(float)
    card_vec = (roles @ rng.standard_normal((4, 256)) + 0.5 * rng.standard_normal((C, 256))).astype(np.float32)
    has = main >= 0
    eta_d = eta_b[main]
    cached = (eta_d - nlin[main] + 0.3 * skill + rng.normal(0, 0.1, D)).astype(np.float64)  # an FM-like teacher
    s = SetData(name=name, d=d, pg=pg, has=has, main_bix=bix[main], main=main, Xb=Xb, gb=gb, bdraft=bdraft, Eb=Eb,
                cov=cov, exp=pg["exp"].astype(np.float32), rec=record_onehot(wb, lb), y=y.astype(np.int8),
                nonland=np.r_[np.ones(nl), np.zeros(n_land)].astype(np.float32),
                played=(ntr @ Xb) > 50, card_vec=card_vec, cached=cached, info={})
    return s, {"eta": eta_d, "nonlinear": nlin[main] - nlin[main].mean()}


def selftest():
    import tempfile
    import sc_gll
    from scipy.stats import pearsonr
    torch.set_num_threads(2)
    t0 = time.time()
    s, truth = synthetic_set()
    s.prepare("raw")
    D = len(s.d.split)
    print(f"synthetic: {D} drafts, {len(s.Xb)} builds, {len(s.y)} games; nonlinear effect sd "
          f"{truth['nonlinear'].std():.3f}, total deck sd {truth['eta'].std():.3f} [{time.time() - t0:.1f}s]")
    fit = {**gfm.FIT, "batch": 1024}
    # 1. fit_fm_b on the build table == game_fm_v2.fit_fm on the dense per-game matrix
    sub = np.flatnonzero(s.gd < 3000)
    sps = np.where(s.gd[sub] < 2100, 0, 1)
    tr, va = np.flatnonzero(sps == 0), np.flatnonzero(sps == 1)
    small = {**fit, "batch": 512, "epochs": 6}
    m1, v1, e1 = fit_fm_b(s.Xb, s.gb[sub], s.c[sub], s.y[sub], tr, va, 1e-5, 1e-3, s.nonland, fit=small)
    m2, v2, e2 = gfm.fit_fm(s.Xb[s.gb[sub]].astype(np.float32), s.c[sub], s.y[sub], tr, va, 1e-5, 1e-3, s.nonland,
                            fit=small)
    assert v1 == v2 and e1 == e2 and torch.equal(m1.w, m2.w), (v1, v2)
    print(f"[1] fit_fm_b == game_fm_v2.fit_fm exactly (val {v1:.6f}) [{time.time() - t0:.1f}s]")
    # 2. tokens / CSR / invariances
    g, pos = s.games_of(np.arange(len(s.Xb)))
    assert (s.gb[g] == pos).all() and len(g) == len(s.y) and len(np.unique(g)) == len(g)
    dec = np.zeros_like(s.Xb)
    r = np.repeat(np.arange(len(s.Xb)), s.ntok)
    k = s.tok[s.tok >= 0]
    dec[r, k] = s.tcnt[s.tok >= 0]
    assert (dec == s.Xb).all()
    cfg = {**CFG, "d": 32, "layers": 2, "heads": 4, "k_seeds": 2, "batch": 128, "eval_batch": 2048, "epochs": 16,
           "lr": 3e-3, "dropout": 0.0}
    trunk = Trunk(s.n_feat, 32, 2, 4, 2, 2, 0.0).eval()
    sm = SetModel(s.Xb.shape[1], s.c.shape[1], s.n_feat, s.nonland, 32, 2, base=s.card_vec)
    nn.init.normal_(sm.head[2].weight, std=0.1)
    b = s.batch(np.arange(64), "cpu", games=False)
    with torch.no_grad():
        r0 = sm.resid(trunk, b["ids"], b["cnt"], b["feat"])
        perm = torch.randperm(b["ids"].shape[1])
        r1 = sm.resid(trunk, b["ids"][:, perm], b["cnt"][:, perm], b["feat"])
        ids2 = torch.cat([b["ids"], torch.full((64, 5), s.Xb.shape[1])], 1)
        r2 = sm.resid(trunk, ids2, torch.cat([b["cnt"], torch.zeros(64, 5, dtype=torch.long)], 1), b["feat"])
    e_perm, e_pad = float((r0 - r1).abs().max()), float((r0 - r2).abs().max())
    assert e_perm < 1e-5 and e_pad < 1e-5 and float(r0.std()) > 1e-3, (e_perm, e_pad)
    print(f"[2] tokens decode to the build table; CSR covers every game once; permutation / padding invariance "
          f"{e_perm:.1e} / {e_pad:.1e} [{time.time() - t0:.1f}s]")
    # 3. the full run: champion refit + PickNet-init transformer, 2 seeds; warm start = champion at step 0
    out = run([s], cfg, seeds=2, init="picknet", distill=0.0, n_boot=60, fit=fit,
              log=lambda m: print(m) if not m.startswith("    seed") or "start" in m or "epoch" in m and
              m.split("epoch")[1].split()[0] in ("0", "5", "10") else None)
    res, scores, losses, ngames = out[s.name]
    te = (s.d.split == 2) & s.has
    tfi = res["transformer fits"]
    for f in tfi:
        assert abs(f["val logloss at start"] - f["champion val logloss (this seed)"]) < 2e-5, f
        assert f["val logloss best"] <= f["val logloss at start"] + 1e-9
    rr = {m: float(np.mean([pearsonr(x[te], truth["eta"][te]).statistic for x in scores[m]]))
          for m in (CHAMP, TF, TF_FM)}
    resid = np.zeros((2, D))
    resid[:] = scores[TF] - scores[TF_FM]
    r_nl = float(np.mean([pearsonr(x[te], truth["nonlinear"][te]).statistic for x in resid]))
    p = res["pairs"]["transformer - champion"]["experienced"]
    pf = res["pairs"]["transformer - transformer FM path"]["experienced"]
    print(f"[3] val logloss champion {np.mean([f['champion val logloss (this seed)'] for f in tfi]):.5f} -> "
          f"transformer {np.mean([f['val logloss best'] for f in tfi]):.5f}; r(score, true deck) on test drafts: "
          + "  ".join(f"{m} {v:.3f}" for m, v in rr.items())
          + f"\n    r(transformer residual, true nonlinear effect) {r_nl:.3f}; SC-GLL experienced transformer - "
          f"champion {p['delta']:+.3f} ± {p['se']:.3f}, transformer - its FM path {pf['delta']:+.3f} ± {pf['se']:.3f}"
          f" [{time.time() - t0:.1f}s]")
    assert rr[TF] > rr[CHAMP] + 0.05, rr
    assert r_nl > 0.3 and r_nl > abs(float(np.mean([pearsonr(x[te], truth["nonlinear"][te]).statistic
                                                     for x in scores[CHAMP]]))) + 0.2, r_nl
    assert p["delta"] > 2 * p["se"] and p["n_seeds"] == 2, p
    assert np.isfinite(res["secondary"][TF]["draft spearman (calibrated)"])
    # 4. random init + distillation plumbing (short), pooled trunk over two sets (short)
    s2, _ = synthetic_set("SYB", D=4000, C=28, seed=1)
    s2.prepare("dl")
    short = {**cfg, "epochs": 2}
    o2 = run([s2], short, seeds=1, init="random", distill=0.5, n_boot=30, fit={**fit, "epochs": 4},
             log=lambda m: None)
    f2 = o2["SYB"][0]["transformer fits"][0]
    assert np.isfinite(o2["SYB"][0]["pairs"]["transformer - champion"]["experienced"]["delta"])
    assert o2["SYB"][0]["diagnostics"]["r(transformer, teacher) test"] > 0.3
    s3, _ = synthetic_set("SYC", D=3000, C=30, seed=2)
    s3.prepare("raw")
    o3 = run([s2, s3], short, seeds=1, init="picknet", distill=0.25, n_boot=30, fit={**fit, "epochs": 4},
             log=lambda m: None)
    assert set(o3) == {"SYB", "SYC"} and all(np.isfinite(o3[k][0]["pairs"]["transformer - champion"]
                                                         ["experienced"]["delta"]) for k in o3)
    print(f"[4] random-init + distill 0.5 (dl context): val {f2['val logloss at start']:.5f} -> "
          f"{f2['val logloss best']:.5f}; pooled trunk over 2 sets runs [{time.time() - t0:.1f}s]")
    # 5. saved files: sc_gll.py meta reads them (temp ROOT, two copies as two "sets")
    with tempfile.TemporaryDirectory() as tmp:
        for nm in ("S1", "S2"):
            gfm.save(Path(tmp) / "results" / f"{nm}.PremierDraft" / OUT, "_picknet_a0", s.d, s.has,
                     {"set": nm, **res}, scores, losses, ngames)
        json.loads((Path(tmp) / "results" / "S1.PremierDraft" / OUT / "results_picknet_a0.json").read_text())
        old = sc_gll.ROOT
        sc_gll.ROOT = Path(tmp)
        try:
            sc_gll.meta("--file", f"{OUT}/results_picknet_a0.json", "S1", "S2")
        finally:
            sc_gll.ROOT = old
    assert parse(["MSH", "--init", "random", "--distill", "0.5"])["tag"] == "random_a0.5"
    assert parse(["MSH", "FDN", "--pooled", "--context", "dl"])["tag"] == "picknet_a0_dl_pooled"
    print(f"self-test OK in {time.time() - t0:.1f}s, peak RSS {rss_gb():.2f} GB")


def mps_smoke(steps=12):
    """Production-size training step on MPS (d=128, L=3, 512 builds x ~5 games, ~30 tokens, 300 cards)."""
    dev = "mps"
    torch.manual_seed(0)
    C, B, M, Fn, P = 300, 512, 30, 18, 58
    trunk = Trunk(Fn).to(dev)
    sm = SetModel(C, P, Fn, np.ones(C, np.float32), 128, 4, base=np.random.randn(C, 256).astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(list(trunk.parameters()) + list(sm.parameters()), 1e-3)
    n_params = sum(p.numel() for p in list(trunk.parameters()) + list(sm.parameters()) if p.requires_grad)
    ids = torch.randint(0, C, (B, M), device=dev)
    b = {"ids": ids, "cnt": torch.randint(1, 3, (B, M), device=dev), "xb": torch.rand(B, C, device=dev),
         "feat": torch.randn(B, Fn, device=dev), "c": torch.randn(B * 5, P, device=dev),
         "y": torch.randint(0, 2, (B * 5,), device=dev).float(), "pos": torch.arange(B, device=dev).repeat_interleave(5)}
    ts = []
    for _ in range(steps):
        t = time.time()
        bce, _, n = game_loss(sm, trunk, b)
        loss = bce / n + sm.penalty(1e-5, 1e-3, 1e-4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.mps.synchronize()
        ts.append(time.time() - t)
    with torch.no_grad():
        trunk.eval()
        sm.eval()
        big = {k: v for k, v in b.items()}
        t = time.time()
        for _ in range(3):
            game_loss(sm, trunk, big)
        torch.mps.synchronize()
        te = (time.time() - t) / 3
    print(f"MPS: {n_params / 1e6:.2f}M trainable params; train step (512 builds, 2560 games) "
          f"{1000 * np.median(ts[2:]):.0f} ms (median of {steps - 2}); eval batch of 512 {1000 * te:.0f} ms; "
          f"peak RSS {rss_gb():.2f} GB, MPS allocated {torch.mps.driver_allocated_memory() / 2 ** 30:.2f} GB")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--mps-smoke" in sys.argv:
        mps_smoke()
    else:
        main(sys.argv[1:])
