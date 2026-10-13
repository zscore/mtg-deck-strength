"""Shared helpers for meta_review_2's plan (T0-T5): python scripts/t_common.py   (self-test, < 20 s)

  torchfree()      installs a torch-free `decks.deckstrength` (the module's own source, minus every top-level node
                   that touches torch / nn / F), so numpy-only and LightGBM processes can import eval_common, sc_gll,
                   improve_deck, improve_deck_v2 and rebuild_targets WITHOUT loading torch (hard rule: never torch and
                   lightgbm in one process). Call it before any of those imports; it refuses if torch is loaded.
  split_mull       own and opponent mulligans per game, zeroed where either is missing, so own - opp is exactly the
                   champion's `mull` column (decks.deckstrength.covariates: null difference -> 0)
  deleaked_bucket  structural_a.deleaked_bucket (copied: structural_a imports torch; selftest checks equality)
  CONTEXTS         the frozen SC-GLL context models of meta_review_2 T1, all fit on train games, no deck term:
                     raw      sc_gll.context_design exactly (raw skill bucket, own - opp mulligans)
                     dl       the de-leaked bucket in place of the raw one (= game_fm_v2's "dl-ctx")
                     pre      dl bucket and NO own-mulligan term (the opponent's mulligans kept): the judge without the
                              own-mulligan mediator
                     pre-raw  raw bucket, no own mulligans (sensitivity only)
  evaluate_ctx     game_fm_v2.evaluate (copied: game_fm_v2 imports torch), keys "<subset> <context>", optional twin
                   calibration (a model scored with another model's val (a, b) in raw units, seed-paired)
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.special import expit

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", Path(__file__).resolve().parent, REPO / "results" / "methodology"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TORCH_NAMES = {"torch", "nn", "F"}
CONTEXTS = {"raw": ("raw", True), "dl": ("dl", True), "pre": ("dl", False), "pre-raw": ("raw", False)}
DEFAULT_CONTEXTS = ("raw", "dl", "pre")
N_BUCKET_NEXT = {1: 5, 5: 10, 10: 50, 50: 100, 100: 500, 500: 1000, 1000: 2000}


# ---------------------------------------------------------------- torch-free decks.deckstrength

def _uses_torch(node) -> bool:
    return any(isinstance(n, ast.Name) and n.id in TORCH_NAMES for n in ast.walk(node))


def torchfree():
    """Install (once) a torch-free decks.deckstrength and return it. Raises if torch is already imported."""
    if "torch" in sys.modules:
        raise RuntimeError("torch is already imported; torchfree() must run before any torch-importing module")
    mod = sys.modules.get("decks.deckstrength")
    if mod is not None:
        if not getattr(mod, "__torchfree__", False):
            raise RuntimeError("the real decks.deckstrength is loaded")
        return mod
    import decks
    path = REPO / "src" / "decks" / "deckstrength.py"
    src = path.read_text()
    keep, dropped = [], []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            if any(n == "torch" or n.startswith("torch.") for n in names):
                continue
        elif _uses_torch(node):
            dropped.append(getattr(node, "name", type(node).__name__))
            continue
        keep.append(node)
    mod = types.ModuleType("decks.deckstrength", ast.get_docstring(ast.parse(src)))
    mod.__file__ = str(path)
    mod.__torchfree__ = True
    mod.__dropped__ = dropped
    sys.modules["decks.deckstrength"] = mod
    decks.deckstrength = mod
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), mod.__dict__)
    return mod


def assert_no_torch():
    if "torch" in sys.modules:
        raise RuntimeError("torch got imported in a process that must stay torch-free")


# ---------------------------------------------------------------- per-game extras

def split_mull(g):
    """(own, opp) mulligans per game (rows of g); both zero where either is missing (own - opp = covariates' mull)."""
    import polars as pl
    own = g["num_mulligans"].cast(pl.Float64, strict=False).to_numpy()
    opp = g["opp_num_mulligans"].cast(pl.Float64, strict=False).to_numpy()
    ok = np.isfinite(own) & np.isfinite(opp)
    return np.where(ok, own, 0.0), np.where(ok, opp, 0.0)


def deleaked_bucket(b, floor, w_run, n_run, min_rest=5):
    """structural_a.deleaked_bucket, verbatim."""
    nxt = np.array([N_BUCKET_NEXT.get(int(f), np.nan) if np.isfinite(f) else np.nan for f in floor])
    n_u = np.maximum(np.sqrt(floor * nxt), n_run)
    rest = n_u - n_run
    ok = np.isfinite(b) & np.isfinite(n_u) & (rest >= min_rest)
    return np.where(ok, np.clip((n_u * np.nan_to_num(b) - w_run) / np.where(ok, rest, 1), 0, 1), np.nan)


def per_draft(pg, D, key):
    out = np.full(D, np.nan)
    out[pg["gd"]] = pg[key]
    return out


def dl_bucket(pg, D):
    """De-leaked skill bucket per game and per draft (game_fm_v2.deleaked_per_game)."""
    W, n = np.bincount(pg["gd"], pg["y"], D), np.bincount(pg["gd"], minlength=D).astype(float)
    bt = deleaked_bucket(per_draft(pg, D, "skill"), per_draft(pg, D, "n_bucket"), W, n)
    return bt[pg["gd"]], bt


def game_blocks(g, pg, D):
    """Calendar day per draft from the game file's draft_time ('' for drafts without games)."""
    out = np.full(D, "", dtype=object)
    if "draft_time" in g.columns:
        out[pg["gd"]] = np.array([str(t)[:10] for t in g["draft_time"].to_list()], dtype=object)
    return out


# ---------------------------------------------------------------- the frozen context models

def context_design(pg, skill, own=True):
    """sc_gll.context_design with the skill column given and, if own is False, -opp mulligans in place of own - opp."""
    from sc_gll import SKILL_EDGES
    from decks.deckstrength import N_RANKS
    from eval_common import N_STATES
    known = np.isfinite(skill)
    b = np.where(known, np.digitize(np.nan_to_num(skill), SKILL_EDGES) + 1, 0)
    lin = np.where(known, skill - 0.55, 0.0)
    mull = pg["mull"] if own else -pg["opp_mull"]
    return np.hstack([pg["on_play"][:, None], mull[:, None], pg["exp"], np.eye(N_RANKS)[pg["rank"]],
                      np.eye(N_STATES + 1)[pg["state"]], np.eye(len(SKILL_EDGES) + 2)[b], lin[:, None] * pg["exp"]])


def fit_contexts(pg, D, names=DEFAULT_CONTEXTS, fit=None):
    """{name: context logit [G]} fit on train games (sc_gll.fit_context's estimator), info, dl bucket per game."""
    from sklearn.linear_model import LogisticRegression
    fit = pg["split"] == 0 if fit is None else fit
    bt_game, _ = dl_bucket(pg, D) if "skill" in pg else (None, None)
    ctxs, info = {}, {}
    for nm in names:
        sk_kind, own = CONTEXTS[nm]
        if not own and "opp_mull" not in pg:
            raise KeyError("pg lacks opp_mull (use split_mull)")
        Xc = context_design(pg, pg["skill"] if sk_kind == "raw" else bt_game, own)
        m = LogisticRegression(C=100.0, max_iter=2000).fit(Xc[fit], pg["y"][fit])
        ctx = m.decision_function(Xc)
        p = np.clip(expit(ctx), 1e-9, 1 - 1e-9)
        ll = -(pg["y"] * np.log(p) + (1 - pg["y"]) * np.log1p(-p))
        ctxs[nm] = ctx
        info[nm] = {"skill": sk_kind, "own mulligans": own, "train games": int(fit.sum()),
                    "train logloss": float(ll[fit].mean()), "test logloss (context only)": float(ll[pg["split"] == 2].mean())}
        del Xc
    return ctxs, info, bt_game


# ---------------------------------------------------------------- SC-GLL with twins (game_fm_v2.evaluate, torch-free)

def fixed_losses(score, pg, ctx, ev, a_raw, b_raw, D):
    gd = pg["gd"][ev]
    pr = np.clip(expit(ctx[ev] + a_raw + b_raw * np.nan_to_num(score)[gd]), 1e-9, 1 - 1e-9)
    y = pg["y"][ev]
    ll = -(y * np.log(pr) + (1 - y) * np.log1p(-pr))
    return np.bincount(gd, ll, D), np.bincount(gd, minlength=D).astype(float)


def raw_beta(score, pg, fit, beta):
    s = np.nan_to_num(score)
    fd = np.unique(pg["gd"][fit])
    mu, sd = s[fd].mean(), s[fd].std() + 1e-12
    return beta[0] - beta[1] * mu / sd, beta[1] / sd


def evaluate_ctx(scores, pg, ctxs, subs, D, pairs, twin=None, n_boot=500, blocks=None, name="", log=print):
    """scores {model: [S, D]}; ctxs {tag: ctx logit}; subs {subset: (fit, ev)} (shared by every context) or
    {tag: {subset: (fit, ev)}}; keys "<subset> <tag>". Returns (sc_gll-meta-format results, losses, test games)."""
    from eval_protocol import gll_metric, paired_bootstrap
    from sc_gll import sc_losses
    twin = twin or {}
    res, losses, ngames = {"models": {m: {} for m in scores}, "pairs": {}}, {m: {} for m in scores}, {}
    order = [m for m in scores if m not in twin] + [m for m in scores if m in twin]
    for tag, ctx in ctxs.items():
        sub_t = subs[tag] if tag in subs else subs
        for sub, (fit, ev) in sub_t.items():
            key = f"{sub} {tag}"
            L, B, n_ev = {}, {}, None
            for m in order:
                rows = []
                for k, s in enumerate(np.atleast_2d(scores[m])):
                    if m in twin:
                        a_raw, b_raw = B[twin[m]][min(k, len(B[twin[m]]) - 1)]
                        ls, n = fixed_losses(s, pg, ctx, ev, a_raw, b_raw, D)
                        beta = (np.nan, b_raw * np.nanstd(np.nan_to_num(s)[np.unique(pg["gd"][fit])]))
                    else:
                        ls, n, beta = sc_losses(s, pg, ctx, fit, ev, D)
                        B.setdefault(m, []).append(raw_beta(s, pg, fit, beta))
                    rows.append((ls, n, beta))
                L[m] = np.stack([r[0] for r in rows])
                n_ev = rows[0][1]
                res["models"][m][key] = {"b per sd": float(np.mean([r[2][1] for r in rows])),
                                         "calibration": f"twin: {twin[m]}" if m in twin else "own val fit"}
            units = np.flatnonzero(n_ev > 0)
            metric = gll_metric(n_ev[units])
            ngames[key] = n_ev
            for m in scores:
                vals = [metric(x[units], np.arange(len(units))) for x in L[m]]
                res["models"][m][key].update({"sc_gll": float(np.mean(vals)), "seed sd": float(np.std(vals)),
                                              "test games": int(n_ev.sum()), "test drafts": int(len(units))})
                losses[m][key] = L[m].astype(np.float32)
            for a, b in pairs:
                if a in L and b in L:
                    r = paired_bootstrap(metric, L[a][:, units], L[b][:, units], n_boot=n_boot, seed=0,
                                         blocks=None if blocks is None else blocks[units], name=name)
                    res["pairs"].setdefault(f"{b} - {a}", {})[key] = {k: v for k, v in asdict(r).items() if k != "boot"}
            log(f"  {key:22s} " + "  ".join(f"{m} {res['models'][m][key]['sc_gll']:.3f}" for m in scores))
            for a, b in pairs:
                r = res["pairs"].get(f"{b} - {a}", {}).get(key)
                if r is not None:
                    log(f"      {b} - {a}: {r['delta']:+.4f} ± {r['se']:.4f}")
    return res, losses, ngames


def save_losses(path, losses, ngames):
    k = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "").replace("*", "star")  # noqa: E731
    np.savez_compressed(path, **{f"{k(m)}|{k(s)}": x for m, ls in losses.items() for s, x in ls.items()},
                        **{f"n_games|{k(s)}": n.astype(np.float32) for s, n in ngames.items()})


def npz_key(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("*", "star")


# ---------------------------------------------------------------- meta helpers

def flip(r: dict) -> dict:
    """A SetResult dict for A - B from B - A."""
    return {**r, "delta": -r["delta"]}


def meta_rows(files: dict, pair: str, key: str, sign: float = 1.0):
    """[SetResult] for one pair / subset key over {set: results json path}; missing sets skipped."""
    from eval_protocol import SetResult
    out = []
    for s, f in files.items():
        f = Path(f)
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        cands = [r.get("pairs"), r.get("sc_gll", {}).get("pairs"), r.get("paired")]
        r = next((c for c in cands if c and pair in c), {})
        x = r.get(pair, {})
        x = x.get(key) if key is not None else x
        if x:
            out.append(SetResult(**{**{k: x[k] for k in ("delta", "se", "se_boot", "se_seed", "n_units", "n_seeds")},
                                    "delta": sign * x["delta"], "name": s}))
    return out


def one_sided_p(m):
    from scipy import stats
    k = len(m.deltas)
    return float(stats.t.sf(m.re_mean / m.re_se_hksj, max(k - 1, 1))) if m.re_se_hksj > 0 else np.nan


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30


# ---------------------------------------------------------------- self-test

def _selftest():
    import subprocess
    import time
    t0 = time.time()
    ds = torchfree()
    import eval_common  # noqa: F401
    import sc_gll
    assert_no_torch()
    assert "PoolEncoder" in ds.__dropped__ and hasattr(ds, "load_drafts") and hasattr(ds, "calibrate")
    print(f"[torchfree] dropped {ds.__dropped__}; eval_common + sc_gll imported without torch")
    # deleaked_bucket == structural_a's, and real deckstrength functions == shim (both in a torch subprocess)
    rng = np.random.default_rng(0)
    b, fl = rng.uniform(0.3, 0.7, 500), rng.choice([1.0, 5, 10, 50, 100, 500, 1000, np.nan], 500)
    W, n = rng.integers(0, 7, 500).astype(float), rng.integers(1, 10, 500).astype(float)
    b[::17] = np.nan
    mine = deleaked_bucket(b, fl, W, n)
    code = ("import sys, numpy as np; sys.path[:0] = [%r, %r, %r]; from structural_a import deleaked_bucket; "
            "z = np.load(sys.argv[1]); np.save(sys.argv[2], deleaked_bucket(z['b'], z['fl'], z['W'], z['n']))"
            % (str(REPO / "src"), str(REPO / "scripts"), str(REPO / "results" / "methodology" / "drafts")))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(Path(tmp) / "in.npz", b=b, fl=fl, W=W, n=n)
        subprocess.run([sys.executable, "-c", code, str(Path(tmp) / "in.npz"), str(Path(tmp) / "out.npy")],
                       check=True, env={**__import__("os").environ, "OMP_NUM_THREADS": "2"})
        ref = np.load(Path(tmp) / "out.npy")
    assert np.allclose(np.nan_to_num(mine, nan=-9), np.nan_to_num(ref, nan=-9)), "deleaked_bucket copy differs"
    print("[deleaked] copy == structural_a.deleaked_bucket")
    # context: raw == sc_gll's; pre drops own mulligans
    d, pg, deck, skill = sc_gll.synthetic(D=3000)
    G = len(pg["y"])
    pg["opp_mull"] = rng.integers(0, 2, G).astype(float)
    own = rng.integers(0, 2, G).astype(float)
    pg["mull"] = own - pg["opp_mull"]
    D = len(d.wins)
    ctxs, info, bt = fit_contexts(pg, D, ("raw", "dl", "pre", "pre-raw"))
    ref_ctx, _ = sc_gll.fit_context(pg, pg["split"] == 0)
    assert np.allclose(ctxs["raw"], ref_ctx), "raw context differs from sc_gll.fit_context"
    assert np.allclose(context_design(pg, pg["skill"])[:, 1], pg["mull"])
    assert np.allclose(context_design(pg, pg["skill"], own=False)[:, 1], -pg["opp_mull"])
    print("[context] raw == sc_gll.fit_context; " + "  ".join(f"{k} test {v['test logloss (context only)']:.4f}"
                                                              for k, v in info.items()))
    # evaluate_ctx: own-fit model reproduces sc_gll.evaluate's numbers; a twin of itself scores identically
    subs = sc_gll.subsets(pg, np.ones(D, bool), np.full(D, "0", dtype=object))
    good = deck + rng.normal(0, 0.15, D)
    scores = {"good": np.stack([good, good]), "twin of good": np.stack([good, good])}
    res, losses, ng = evaluate_ctx(scores, pg, {"raw": ctxs["raw"]}, subs, D, [("good", "twin of good")],
                                   twin={"twin of good": "good"}, n_boot=30, log=lambda m: None)
    ref = sc_gll.evaluate({"good": scores["good"]}, pg, ref_ctx, subs, D, pairs=[], n_boot=10, log=lambda m: None)
    assert abs(res["models"]["good"]["experienced raw"]["sc_gll"] - ref["models"]["good"]["experienced"]["sc_gll"]) < 1e-9
    assert abs(res["models"]["twin of good"]["experienced raw"]["sc_gll"]
               - sc_gll.evaluate({"g": good[None]}, pg, ref_ctx, subs, D, pairs=[], n_boot=5,
                                 log=lambda m: None)["models"]["g"]["experienced"]["sc_gll"]) < 1e-6
    print(f"[evaluate_ctx] == sc_gll.evaluate; twin calibration exact [{time.time() - t0:.1f}s]")
    assert_no_torch()
    print(f"t_common self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    _selftest()
