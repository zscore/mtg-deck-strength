"""T5 (meta_review_2): vault confirmation of the frozen pipelines on sets never used for model selection.

    python scripts/t5_vault.py freeze [--adopt "- own mull"|"rated-dl - own mull"|"rated-dl"|"split mull"|"blend*"]
                                      [--note TEXT] [--refreeze]
    python scripts/t5_vault.py run SET [--no-aff] [--stage all|fm|lgbm] [--threads 4] [--boot 500] [--rerun]
    python scripts/t5_vault.py report
    python scripts/t5_vault.py --selftest           (synthetic: freeze / hash guard / report in a temp dir, then the
                                                    fm stage (torch) and the lgbm stage (lightgbm) in two subprocesses)

freeze  writes results/vault/frozen_pipeline.json ONCE (plus .sha256, read-only 0444, and a ledger line): the exact
        champion v2 deck-strength recipe, the round 10 record recipe, the metrics, the vault sets and the one-sided
        hypotheses H1-H5, and the sha256 of every script the vault runs execute. --refreeze is allowed only while no
        vault stage has completed; the old file is kept as frozen_pipeline.<time>.json.
run     executes the frozen pipelines on one vault set with the existing code paths, in two processes (torch and
        lightgbm never share one):
          fm    (torch; game_fm_v2 + t2_mull_split code) champion fm4 nonland, train-only, 2 seeds (grid on seed 0);
                its refit on train+val replaying the schedule (champion v2, twin-calibrated); the linear game model
                (round 4 `linear + deck features + aff`, k = 0); the adopted H5 arm if any; and, for the record
                pipeline, deck_synergy_v4's draft inputs: the champion's deck score with TRAIN drafts cross-fitted on
                two halves (fold rule of deck_synergy_v4) and val/test drafts scored by the model fit on all train
                games, plus the deck features / affinity columns. SC-GLL raw / dl / pre (t_common.evaluate_ctx).
          lgbm  (lightgbm, torch-free shim) sc_gll.lgbm_scores (round 4 LightGBM at platinum, 3 seeds, round 4 blend)
                scored on SC-GLL with the fm stage's FM scores; rebuild_targets.game_records / compare / rescore with
                the frozen record config (round 4 blend vs round 10: tau 2, default LightGBM params, refit train+val,
                5 seeds, + game FM, alpha on val record NLL; draft and game targets).
        --no-aff: the pre-declared fallback for sets without a PickNet pool model (HOB, MOM, TMT, WOE): champion,
        linear model and LightGBM all without the affinity column. Every run re-checks the frozen json's sha256 and
        the code hashes before and after, never opens the frozen json for writing, refuses to overwrite a completed
        stage (vault touched once; --rerun only after a crash, logged in the ledger).
report  evaluates H1-H5 over the vault sets (affinity sets, no-aff sets, all) with eval_protocol.meta_analysis
        (DL + HKSJ) and decide(), the one-sided p, the 12-dev-set + vault pooled estimate, and the frozen verdict rule;
        writes results/vault/report.json.
Outputs: results/vault/<SET>.PremierDraft/{fm_results.json, fm_scores.npz, draft_inputs.npz, lgbm_results.json,
targets.json, targets.npz}; results/vault/{ledger.jsonl, report.json}.
"""
import os
import sys

_T = "2" if any(a.startswith("--selftest") for a in sys.argv) else next(
    (sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--threads"), "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _T)

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t_common import REPO, sha256  # noqa: E402

import numpy as np  # noqa: E402

VAULT = REPO / "results" / "vault"
FROZEN_NAME = "frozen_pipeline.json"
CH, LG, LIN, REFIT = "champion", "LightGBM", "linear game model", "refit"
BLEND = "blend* (LightGBM + champion)"
AFF_SETS, NOAFF_SETS = ("ECL", "TLA"), ("WOE", "MOM", "HOB", "TMT")
ARMS = ("- own mull", "rated-dl - own mull", "rated-dl", "split mull", "blend*")
CODE = ["scripts/t5_vault.py", "scripts/t_common.py", "scripts/game_fm_v2.py", "scripts/t2_mull_split.py",
        "scripts/t1_rescore.py", "scripts/sc_gll.py", "scripts/eval_common.py", "scripts/rebuild_targets.py",
        "scripts/improve_deck_v2.py", "scripts/improve_deck.py", "scripts/deck_synergy_v2.py",
        "scripts/deck_synergy_v3.py", "results/methodology/eval_protocol.py", "src/decks/deckstrength.py",
        "src/decks/synergy.py"]


# ---------------------------------------------------------------- the frozen object

def _fm_constants():
    """game_fm_v2's FIT / GRID without importing torch (parsed from the source)."""
    import ast
    src = (REPO / "scripts" / "game_fm_v2.py").read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ("FIT", "GRID", "PLAYED_MIN"):
            out[node.targets[0].id] = ast.literal_eval(node.value) if node.targets[0].id != "FIT" else \
                {k.arg: ast.literal_eval(k.value) for k in node.value.keywords}
    return out


def build_frozen(adopt=None, note=""):
    import mtgdata
    fm = _fm_constants()
    mt = Path(mtgdata.__file__).resolve().parent
    code = {c: sha256(REPO / c) for c in CODE}
    code.update({f"mtgdata/{f}": sha256(mt / f) for f in ("games.py", "cardinfo.py", "data.py")})
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        commit, dirty = None, []
    h5_key = "experienced pre" if adopt in ("- own mull", "rated-dl - own mull") else "experienced dl"
    hyp = {
        "H1": {"claim": "game FM (champion, train-only) - LightGBM (round 4, platinum) > 0", "file": "lgbm_results.json",
               "pair": f"{CH} - {LG}", "key": "experienced dl",
               "dev": {"file": "t1_rescore/results.json", "pair": f"{LG} - {CH}", "key": "experienced dl", "sign": -1}},
        "H2": {"claim": "game FM (champion) - linear game model > 0", "file": "fm_results.json",
               "pair": f"{CH} - {LIN}", "key": "experienced dl",
               "dev": {"file": "t1_rescore/results.json", "pair": f"{CH} - {LIN}", "key": "experienced dl", "sign": 1}},
        "H3": {"claim": "refit on train+val (champion v2) - train-only champion > 0 (twin calibration)",
               "file": "fm_results.json", "pair": f"{REFIT} - {CH}", "key": "experienced dl",
               "special rule": "underpowered at k = 6 (MDE ~0.06): confirmed if the vault mean > 0 AND the dev + vault "
                               "pooled one-sided p < 0.05",
               "dev": {"file": "t1_rescore/results.json", "pair": f"{REFIT} - {CH}", "key": "experienced dl", "sign": 1,
                       "fallback": {"file": "game_fm_v2/results.json", "key": "experienced dl-ctx"}}},
        "H4": {"claim": "round 10 record recipe - round 4 blend > 0 (test Spearman; train draft | eval draft primary)",
               "file": "targets.json", "pair": "round 10 - round 4 | train draft | eval draft", "key": None,
               "secondary": "round 10 - round 4 | train game | eval game",
               "dev": {"file": "targets.json", "pair": "round 10 - round 4 | train draft | eval draft", "key": None,
                       "sign": 1}},
        "H5": {"claim": f"adopted arm {adopt!r} - champion v2 > 0" if adopt else "not active (nothing adopted by T1/T2)",
               "active": adopt is not None, "arm": adopt,
               "file": "lgbm_results.json" if adopt == "blend*" else "fm_results.json",
               "pair": (f"{BLEND} - {CH}" if adopt == "blend*" else f"{adopt} - {CH}") if adopt else None,
               "key": h5_key,
               "dev": ({"file": "t1_rescore/results.json", "pair": f"{BLEND} - {CH}", "key": "experienced dl", "sign": 1}
                       if adopt == "blend*" else
                       {"file": "t2_mull_split/results.json", "pair": f"{adopt} - {CH}", "key": h5_key, "sign": 1})
               if adopt else None},
    }
    return {
        "created": datetime.now().isoformat(timespec="seconds"), "note": note,
        "status": "frozen before any vault file was read; vault runs must not change this file or the hashed code",
        "git": {"commit": commit, "tracked files modified at freeze": dirty},
        "code sha256": code,
        "vault sets": {"with affinity": list(AFF_SETS), "no-aff fallback (no PickNet pool model)": list(NOAFF_SETS),
                       "excluded": {"NEO": "no draft parquet (raw CSV only), no game_time",
                                    "ONE": "no draft parquet (raw CSV only), no game_time"},
                       "order": ["ECL", "TLA", "WOE", "MOM", "HOB", "TMT"],
                       "notes": ["ECL was used for play/draw outcome statistics in the matchmaking memo (not model "
                                 "selection)", "MOM has no game_time column (nothing here uses it; day blocks use "
                                 "draft_time)", "no-aff sets are pooled separately from ECL/TLA and in 'all'"]},
        "split": "decks.deckstrength.load_drafts: drafts sorted by draft_time, 70% train / 15% val / 15% test",
        "deck strength pipeline (champion v2)": {
            "model": "DeckFM fm4 nonland: card power w (all cards) + rank-4 factorised pairs between nonland cards + "
                     "17 standardised Scryfall deck features + the PickNet pool_d32 affinity profile column "
                     "(deck_synergy_v3.Profile 'aff mean'); --no-aff sets: no affinity column",
            "context (game only, not in the deck score)": "play/draw, mulligan difference (own - opp), raw "
                     "user_game_win_rate_bucket + known flag, per-game rank one-hot, experience one-hot (500/1000 fix), "
                     "record before the game (21 one-hot); opponent rank dropped",
            "fit": fm.get("FIT"), "grid (l2_w, l2_v)": fm.get("GRID"), "grid chosen on": "val game log loss, seed 0",
            "seeds": 2, "refit": "train+val games, the champion's val-chosen setting, its early-stopping schedule "
                                 "replayed epoch for epoch (game_fm_v2.fit_fm replay), per seed",
            "calibration of the refit": "the train-only twin's val (a, b) in raw score units, seed-paired",
            "deck score": "deck-only logit of the most-played build: card power + pairs + deck-feature/affinity terms",
            "adopted change (T1/T2)": adopt},
        "linear game model": {"model": "DeckFM k = 0 + deck features (+ aff)", "grid (l2_w, l2_v)": [[1e-5, 0], [1e-4, 0]],
                              "seeds": 2},
        "record pipeline (round 10)": {
            "LightGBM features": "most-played build's card counts + structure features + own rank + deck features "
                                 "(+ affinity)",
            "round 4 blend": "LightGBM (LGB_PARAMS, early stop on val) logit-blended with the train-calibrated game FM; "
                             "alpha in 0..1 step 0.05 on val record NLL",
            "round 10": "tau 2 recency weights, default LGB_PARAMS (no improve_v2.json on vault sets), refit on "
                        "train+val with trees scaled by data size, 5 seeds averaged in logit space, blended with the "
                        "train+val-calibrated game FM, alpha from the train-only twin on val record NLL",
            "cfg": {"tau": 2.0, "params": {}, "source": "frozen (t5_vault)"}, "refit seeds": 5,
            "game FM input": "the champion (train-only, seed 0); training drafts cross-fitted on two halves "
                             "(deck_synergy_v4's fold rule: rng(2) permutation of the training draft ids), val/test "
                             "drafts scored by the fit on all training games (deck_synergy_v4 draft_inputs)",
            "targets": "draft-file record (primary) and game-file record; evaluated at the player's own rank"},
        "metrics": {
            "primary": "SC-GLL experienced (user_n_games_bucket >= 50), de-leaked-bucket context ('experienced dl'), "
                       "-1000 x test game log loss, (a, b) on val games, paired day-block bootstrap (500) over test "
                       "drafts with the seed spread as a variance component",
            "contexts": ["raw", "dl", "pre"], "LightGBM seeds": 3, "reference rank": "platinum",
            "record metric": "test Spearman(p, W/(W+L)), paired day-block bootstrap (300)"},
        "hypotheses": hyp,
        "decision rule": {
            "per hypothesis": "random-effects (DL + HKSJ) mean over the vault sets; one-sided p = P(t_{k-1} > mean / "
                              "se); CONFIRMED if mean > 0 and one-sided p < 0.05 (H3: mean > 0 and pooled dev + vault "
                              "one-sided p < 0.05); 'DEV-TEST ONLY' if the vault 95% CI includes 0 and the mean <= 0; "
                              "otherwise NOT CONFIRMED. eval_protocol.decide (min_positive = ceil(2k/3), min_effect 0) "
                              "is reported alongside.",
            "groups": ["affinity sets (ECL, TLA)", "no-aff sets", "all vault sets"],
            "no re-tuning on the vault, ever": True}}


def vault_dir(s):
    return VAULT / f"{s}.PremierDraft"


def ledger(event: dict):
    VAULT.mkdir(parents=True, exist_ok=True)
    with open(VAULT / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **event}) + "\n")


def completed_stages():
    return sorted(str(p.relative_to(VAULT)) for p in VAULT.glob("*.PremierDraft/stage_*.done"))


def freeze(argv):
    adopt, note, refreeze = None, "", False
    it = iter(argv)
    for a in it:
        if a == "--adopt":
            adopt = next(it)
        elif a == "--note":
            note = next(it)
        elif a == "--refreeze":
            refreeze = True
    if adopt is not None and adopt not in ARMS:
        raise SystemExit(f"--adopt must be one of {ARMS}")
    f = VAULT / FROZEN_NAME
    if f.exists():
        if not refreeze:
            raise SystemExit(f"{f} exists; the frozen pipeline is never rewritten (use --refreeze only before any vault "
                             "stage has completed)")
        done = completed_stages()
        if done:
            raise SystemExit(f"refusing to refreeze: vault stages already completed {done}")
        arch = VAULT / f"frozen_pipeline.{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        os.chmod(f, 0o644)
        f.rename(arch)
        f.with_suffix(".sha256").unlink(missing_ok=True)
        ledger({"event": "refreeze", "archived": arch.name})
    VAULT.mkdir(parents=True, exist_ok=True)
    obj = build_frozen(adopt, note)
    f.write_text(json.dumps(obj, indent=1))
    h = sha256(f)
    f.with_suffix(".sha256").write_text(h + "\n")
    os.chmod(f, 0o444)
    os.chmod(f.with_suffix(".sha256"), 0o444)
    ledger({"event": "freeze", "sha256": h, "adopt": adopt})
    print(f"frozen: {f} (sha256 {h[:12]}...), H5 {'active: ' + adopt if adopt else 'inactive'}")
    return obj


def load_frozen(check_code=True):
    f = VAULT / FROZEN_NAME
    if not f.exists():
        raise SystemExit(f"{f} missing: run `t5_vault.py freeze` first")
    want = f.with_suffix(".sha256").read_text().strip()
    got = sha256(f)
    if got != want:
        raise SystemExit(f"{f} was modified after freezing (sha256 {got[:12]} != {want[:12]})")
    obj = json.loads(f.read_text())
    drift = []
    if check_code:
        import mtgdata
        mt = Path(mtgdata.__file__).resolve().parent
        for c, h in obj["code sha256"].items():
            p = mt / c.split("/", 1)[1] if c.startswith("mtgdata/") else REPO / c
            if not p.exists() or sha256(p) != h:
                drift.append(c)
    return obj, got, drift


# ---------------------------------------------------------------- fm stage (torch)

def fit_linear(data, seeds, grid, fit=None, log=print):
    import game_fm_v2 as g2
    X, y, gsp = data["X"], data["y"], data["gsp"]
    tr, va = np.flatnonzero(gsp == 0), np.flatnonzero(gsp == 1)
    c = g2.context_cols(CH, data)
    fit = {**(fit or g2.FIT), "k": 0}
    t1 = time.time()
    fits = [g2.fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=0, fit=fit) + (kw,) for kw in grid]
    m0, v0, ev0, kw = min(fits, key=lambda f: f[1])
    ms = [m0] + [g2.fit_fm(X, c, y, tr, va, *kw, data["nonland"], seed=s, fit=fit)[0] for s in range(1, seeds)]
    log(f"  {LIN:22s} val {v0:.5f} {tuple(kw)} [{time.time() - t1:.0f}s]")
    return np.stack([g2.deck_logit(m, data["deck"], data["Ed"]) for m in ms]), {"setting": list(kw), "val logloss": v0}


def crossfit(data, champ, fit=None, log=print):
    """deck_synergy_v4's draft input: training drafts scored by a fit on the other half of the training drafts (same
    setting, seed 0, early-stopped on val), val/test drafts by the champion fit on all training games."""
    import game_fm_v2 as g2
    d, X, y, gsp = data["d"], data["X"], data["y"], data["gsp"]
    full = champ["scores"][0].copy()
    draft_ids = np.asarray(d.draft_id).astype(str)
    gdraft = draft_ids[data["gd"]]
    tr_ids = np.unique(gdraft[gsp == 0])
    fold_a = set(tr_ids[np.random.default_rng(2).permutation(len(tr_ids)) < len(tr_ids) // 2].tolist())
    in_a_game = np.array([x in fold_a for x in gdraft])
    in_a_draft = np.array([x in fold_a for x in draft_ids])
    c = g2.context_cols(CH, data)
    va = np.flatnonzero(gsp == 1)
    trd = d.split == 0
    for side in (True, False):
        t1 = time.time()
        tr_s = np.flatnonzero((gsp == 0) & (in_a_game != side))
        m, v, _ = g2.fit_fm(X, c, y, tr_s, va, *champ["setting"], data["nonland"], seed=0, fit=fit or g2.FIT)
        sel = trd & (in_a_draft == side)
        full[sel] = g2.deck_logit(m, data["deck"], data["Ed"])[sel]
        log(f"  cross-fit half {int(side)}: val {v:.5f} [{time.time() - t1:.0f}s]")
    return full


def fm_core(data, fz, seeds, n_boot, name="", log=print, fit=None):
    import game_fm_v2 as g2
    import t2_mull_split as t2
    from eval_common import day_blocks
    from sc_gll import subsets
    from t_common import evaluate_ctx, fit_contexts
    t0 = time.time()
    d, pg, has = data["d"], data["pg"], data["has"]
    D = len(d.split)
    data["bt_game"], data["bt_draft"] = g2.deleaked_per_game(pg, D)
    adopt = fz["hypotheses"]["H5"]["arm"]
    arms = [adopt] if adopt not in (None, "blend*") else []
    fit = fit or g2.FIT
    fits = t2.fit_arms(data, arms, seeds=seeds, refit_of=(CH,), fit=fit, log=log)
    lin, lin_info = fit_linear(data, seeds, [tuple(x) for x in fz["linear game model"]["grid (l2_w, l2_v)"]], fit, log)
    scores = {a: f["scores"] for a, f in fits.items()}
    scores[LIN] = lin
    gl = crossfit(data, fits[CH], fit, log)
    ctxs, cinfo, _ = fit_contexts(pg, D, tuple(fz["metrics"]["contexts"]))
    subs = subsets(pg, has, data["main_bix"])
    pairs = [(CH, REFIT), (LIN, CH)] + [(CH, a) for a in arms]
    ev, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, {REFIT: CH}, n_boot, day_blocks(d), name, log)
    nf = data["Ed"].shape[1] - (1 if data["info"].get("affinity column") else 0)
    inputs = {"has_deck": has, "game_logit": gl, "game_logit_ref": lin[0], "deck_features": data["Ed"][:, :nf],
              "aff": data["Ed"][:, -1] if data["info"].get("affinity column") else np.zeros(D, np.float32),
              "aff_present": np.array(bool(data["info"].get("affinity column")))}
    res = {"arms": list(scores), "seeds": seeds, "contexts": cinfo, "affinity column": data["info"].get("affinity column"),
           "fits": {a: {k: x for k, x in f.items() if k not in ("scores", "bscores", "w", "model0")} for a, f in fits.items()},
           "linear game model fit": lin_info,
           "cross-fit r with in-sample (train drafts)": float(np.corrcoef(gl[(d.split == 0) & has],
                                                                          fits[CH]["scores"][0][(d.split == 0) & has])[0, 1]),
           **ev, "seconds": time.time() - t0}
    return res, scores, inputs


def fm_stage(set_code, fz, use_aff, n_boot, threads):
    import torch
    import t2_mull_split as t2
    from t_common import peak_rss_gb
    if "lightgbm" in sys.modules:
        raise RuntimeError("lightgbm loaded in the torch stage")
    torch.set_num_threads(threads)
    _check_constants(fz)
    t0 = time.time()
    data = t2.load_set_v2(set_code, use_aff)
    res, scores, inputs = fm_core(data, fz, fz["deck strength pipeline (champion v2)"]["seeds"], n_boot, set_code,
                                  log=lambda m: print(m, flush=True))
    out = vault_dir(set_code)
    out.mkdir(parents=True, exist_ok=True)
    d = data["d"]
    np.savez_compressed(out / "fm_scores.npz", draft_id=np.asarray(d.draft_id).astype(str), split=d.split,
                        has=data["has"], **{k.replace(" ", "_"): v.astype(np.float32) for k, v in scores.items()})
    np.savez_compressed(out / "draft_inputs.npz", **inputs)
    res = {"set": set_code, "no-aff fallback": not use_aff, **res, "total seconds": time.time() - t0,
           "peak RSS GB": peak_rss_gb()}
    (out / "fm_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{set_code} fm stage done [{time.time() - t0:.0f}s, {peak_rss_gb():.1f} GB]", flush=True)


def _check_constants(fz):
    import game_fm_v2 as g2
    dp = fz["deck strength pipeline (champion v2)"]
    if dp["fit"] != g2.FIT or [list(x) for x in g2.GRID] != [list(x) for x in dp["grid (l2_w, l2_v)"]]:
        raise SystemExit("game_fm_v2 FIT / GRID differ from the frozen pipeline")


# ---------------------------------------------------------------- lgbm stage (lightgbm, no torch)

def lgbm_core(d, pg, has, main_bix, blocks, X, rank_col, inputs, fm_scores, fz, n_boot, name="", log=print,
              n_lgbm_seeds=None, records=None):
    """SC-GLL of LightGBM / round 4 blend vs the fm stage's scores; the record pipeline when `records` is given
    (rebuild_targets.game_records output)."""
    import sc_gll
    import rebuild_targets as rt
    from t_common import evaluate_ctx, fit_contexts
    t0 = time.time()
    D = len(d.split)
    lg = sc_gll.lgbm_scores(X, d, has, inputs["game_logit"], rank_col,
                            n_seeds=n_lgbm_seeds or fz["metrics"]["LightGBM seeds"], log=log)
    scores = {CH: fm_scores[CH], REFIT: fm_scores[REFIT], LIN: fm_scores[LIN], LG: lg["lgbm ref"],
              "round 4 blend": lg["blend ref"]}
    ctxs, cinfo, _ = fit_contexts(pg, D, tuple(fz["metrics"]["contexts"]))
    subs = sc_gll.subsets(pg, has, main_bix)
    pairs = [(LG, CH), ("round 4 blend", CH), (LG, REFIT)]
    binfo = None
    if fz["hypotheses"]["H5"]["arm"] == "blend*":
        import t1_rescore as t1
        scores[BLEND], binfo = t1.blend_star(scores[LG], scores[CH], pg, ctxs["dl"], subs["experienced"][0], D, has,
                                             d.split, log=log)
        pairs.append((CH, BLEND))
    ev, _, _ = evaluate_ctx(scores, pg, ctxs, subs, D, pairs, {REFIT: CH}, n_boot, blocks, name, log)
    res = {"contexts": cinfo, "alphas (round 4 blend, val NLL)": lg["alpha"].tolist(), "trees": lg["trees"].tolist(),
           "blend*": binfo, **ev}
    tres = None
    if records is not None:
        cfg = fz["record pipeline (round 10)"]["cfg"]
        targets = {"draft": (d.wins, d.losses), "game": (records["wins_games"].astype(np.float32),
                                                         records["losses_games"].astype(np.float32))}
        tres, _ = rt.rescore(d, X, has, inputs["game_logit"], targets, {"tau": cfg["tau"], "params": cfg["params"]},
                             n_seeds=fz["record pipeline (round 10)"]["refit seeds"], blocks=blocks, log=log)
        tres["comparison"] = rt.compare(d, records, has)
    res["seconds"] = time.time() - t0
    return res, tres, scores


def lgbm_stage(set_code, fz, use_aff, n_boot):
    from t_common import torchfree, assert_no_torch, split_mull, peak_rss_gb
    torchfree()
    import lightgbm  # noqa: F401
    import rebuild_targets as rt
    from eval_common import NARROW, day_blocks, deck_rows, load_games, main_builds, per_game
    from improve_deck import structure_features
    from mtgdata.games import game_path
    from decks.deckstrength import load_drafts
    t0 = time.time()
    out = vault_dir(set_code)
    inputs = dict(np.load(out / "draft_inputs.npz"))
    fz_scores = np.load(out / "fm_scores.npz")
    d = load_drafts(set_code, "PremierDraft")
    assert (fz_scores["draft_id"] == np.asarray(d.draft_id).astype(str)).all()
    path = game_path(set_code, "PremierDraft")
    g = load_games(path, d.draft_id, cols=NARROW + ("draft_time",))
    pg = per_game(d, g)
    pg["own_mull"], pg["opp_mull"] = split_mull(g)
    has, rows, main_bix = main_builds(d, g)
    assert (has == inputs["has_deck"]).all(), "has-deck mask differs from the fm stage"
    records = rt.game_records(d, g)
    del g
    deck = np.zeros((len(d.draft_id), len(d.names)), np.int8)
    deck[has] = deck_rows(path, rows[has], d.names)
    aff = bool(inputs["aff_present"])
    if aff != use_aff:
        raise SystemExit(f"--no-aff mismatch between stages (fm stage affinity {aff})")
    X = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                   d.rank[:, None].astype(np.float32), inputs["deck_features"]]
                  + ([inputs["aff"][:, None]] if aff else []))
    rank_col = len(d.names) + structure_features(deck[:1], d.names, set_code).shape[1]
    assert (X[:, rank_col] == d.rank).all()
    wr, lr = rt.reconcile(d, records)
    np.savez_compressed(out / "targets.npz", draft_id=d.draft_id.astype(str), wins_reconciled=wr, losses_reconciled=lr,
                        **records)
    fm_scores = {k.replace("_", " "): fz_scores[k].astype(np.float64) for k in fz_scores.files
                 if k not in ("draft_id", "split", "has")}
    print(f"=== {set_code} lgbm stage: {len(d.draft_id)} drafts, {len(pg['y'])} games, affinity {aff} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    res, tres, _ = lgbm_core(d, pg, has, main_bix, day_blocks(d), X, rank_col, inputs, fm_scores, fz, n_boot,
                             set_code, log=lambda m: print(m, flush=True), records=records)
    (out / "lgbm_results.json").write_text(json.dumps({"set": set_code, "no-aff fallback": not use_aff, **res,
                                                       "total seconds": time.time() - t0, "peak RSS GB": peak_rss_gb()},
                                                      indent=1, default=float))
    (out / "targets.json").write_text(json.dumps({"set": set_code, **tres}, indent=1, default=float))
    assert_no_torch()
    print(f"{set_code} lgbm stage done [{time.time() - t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- run

def run(argv):
    set_code, stage, use_aff, n_boot, threads, rerun, allow_drift = None, "all", True, 500, int(_T), False, False
    it = iter(argv)
    for a in it:
        if a == "--stage":
            stage = next(it)
        elif a == "--no-aff":
            use_aff = False
        elif a == "--boot":
            n_boot = int(next(it))
        elif a == "--threads":
            threads = int(next(it))
        elif a == "--rerun":
            rerun = True
        elif a == "--allow-code-drift":
            allow_drift = True
        else:
            set_code = a
    fz, h, drift = load_frozen()
    if drift and not allow_drift:
        raise SystemExit(f"code changed since the freeze: {drift} (protocol: re-freeze before any vault stage, or "
                         "--allow-code-drift, which is logged as a deviation)")
    allowed = set(fz["vault sets"]["with affinity"]) | set(fz["vault sets"]["no-aff fallback (no PickNet pool model)"])
    if set_code not in allowed:
        raise SystemExit(f"{set_code} is not a frozen vault set {sorted(allowed)}")
    if set_code in fz["vault sets"]["no-aff fallback (no PickNet pool model)"] and use_aff:
        raise SystemExit(f"{set_code} is a pre-declared no-aff set: pass --no-aff")
    if stage == "all":
        for st in ("fm", "lgbm"):
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", set_code, "--stage", st, "--boot", str(n_boot),
                   "--threads", str(threads)] + (["--no-aff"] if not use_aff else []) + (["--rerun"] if rerun else []) \
                + (["--allow-code-drift"] if allow_drift else [])
            subprocess.run(cmd, check=True)
        return
    done = vault_dir(set_code) / f"stage_{stage}.done"
    if done.exists() and not rerun:
        raise SystemExit(f"{done} exists: the vault is touched once (use --rerun only after a crash)")
    if stage == "lgbm" and not (vault_dir(set_code) / "stage_fm.done").exists():
        raise SystemExit("run the fm stage first")
    ledger({"event": "start", "set": set_code, "stage": stage, "frozen sha256": h, "no-aff": not use_aff,
            "rerun": rerun, "code drift": drift})
    (fm_stage(set_code, fz, use_aff, n_boot, threads) if stage == "fm" else lgbm_stage(set_code, fz, use_aff, n_boot))
    _, h2, drift2 = load_frozen()
    if h2 != h:
        raise SystemExit("the frozen json changed during the run")
    done.write_text(json.dumps({"frozen sha256": h, "time": datetime.now().isoformat(timespec="seconds")}))
    ledger({"event": "done", "set": set_code, "stage": stage, "frozen sha256": h2, "code drift": drift2})


# ---------------------------------------------------------------- report

def _rows(hyp, sets, root, dev=False):
    from t_common import meta_rows
    spec = hyp["dev"] if dev else hyp
    if spec is None:
        return []
    files = {s: (root / f"{s}.PremierDraft" / spec["file"]) for s in sets}
    rs = meta_rows(files, spec["pair"], spec["key"], spec.get("sign", 1))
    if dev and spec.get("fallback"):
        have = {r.name for r in rs}
        fb = spec["fallback"]
        rs += meta_rows({s: root / f"{s}.PremierDraft" / fb["file"] for s in sets if s not in have}, spec["pair"],
                        fb["key"], spec.get("sign", 1))
    return rs


def evaluate_h(name, hyp, vault_rows, dev_rows):
    from eval_protocol import DecisionRule, decide, meta_analysis
    from t_common import one_sided_p
    out = {"claim": hyp["claim"], "vault sets": [r.name for r in vault_rows]}
    k = len(vault_rows)
    if k == 0:
        out["verdict"] = "no vault results"
        return out
    if k == 1:
        r = vault_rows[0]
        from scipy import stats
        p1 = float(stats.norm.sf(r.delta / r.se))
        out.update({"vault mean": r.delta, "se": r.se, "one-sided p (z, k = 1)": p1})
        mean, lo, p = r.delta, r.delta - 1.96 * r.se, p1
    else:
        m = meta_analysis(vault_rows)
        p = one_sided_p(m)
        ok, why = decide(m, DecisionRule(min_positive=math.ceil(2 * k / 3), min_effect=0.0))
        out.update({"vault mean": m.re_mean, "ci": list(m.re_ci), "one-sided p": p, "sets better": f"{m.n_positive}/{k}",
                    "sig worse": m.n_sig_worse, "decide": ok, "decide checks": why, "summary": m.summary()})
        mean, lo = m.re_mean, m.re_ci[0]
    pooled_p = None
    if len(dev_rows) + k >= 2:
        mp = meta_analysis(dev_rows + vault_rows)
        pooled_p = one_sided_p(mp)
        out["pooled dev + vault"] = {"sets": len(dev_rows) + k, "re mean": mp.re_mean, "ci": list(mp.re_ci),
                                     "one-sided p": pooled_p}
    if name == "H3":
        conf = mean > 0 and pooled_p is not None and pooled_p < 0.05
    else:
        conf = mean > 0 and p < 0.05
    out["verdict"] = "CONFIRMED" if conf else ("DEV-TEST ONLY" if (mean <= 0 and lo <= 0) else "NOT CONFIRMED")
    return out


def report(root=None, dev_root=None, write=True):
    fz, h, drift = load_frozen(check_code=False)
    root = root or VAULT
    dev_root = dev_root or (REPO / "results")
    vs = fz["vault sets"]
    aff, noaff = vs["with affinity"], vs["no-aff fallback (no PickNet pool model)"]
    dev_sets = ["BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "MSH", "OTJ", "SOS", "TDM"]
    out = {"frozen sha256": h, "code drift since freeze": drift, "hypotheses": {}}
    for name, hyp in fz["hypotheses"].items():
        if name == "H5" and not hyp.get("active"):
            out["hypotheses"][name] = {"claim": hyp["claim"], "verdict": "inactive"}
            continue
        dev_rows = _rows(hyp, dev_sets, dev_root, dev=True)
        o = {}
        for grp, sets in (("affinity sets", aff), ("no-aff sets", noaff), ("all vault sets", aff + noaff)):
            o[grp] = evaluate_h(name, hyp, _rows(hyp, sets, root), dev_rows)
        o["dev sets with results"] = [r.name for r in dev_rows]
        if hyp.get("secondary"):
            o["secondary (all vault sets)"] = evaluate_h(name, {**hyp, "pair": hyp["secondary"]},
                                                        _rows({**hyp, "pair": hyp["secondary"]}, aff + noaff, root), [])
        out["hypotheses"][name] = o
        a = o["all vault sets"]
        print(f"{name}: {hyp['claim']}\n    all vault sets: {a.get('verdict')}  mean {a.get('vault mean', float('nan')):+.4f}"
              f"  one-sided p {a.get('one-sided p', a.get('one-sided p (z, k = 1)', float('nan'))):.4f}  "
              f"(affinity: {o['affinity sets'].get('verdict')}, no-aff: {o['no-aff sets'].get('verdict')})")
    if write:
        (root / "report.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------- self-tests

def _synthetic_records(d, pg):
    D = len(d.split)
    W = np.bincount(pg["gd"], pg["y"], D).astype(np.int64)
    n = np.bincount(pg["gd"], minlength=D).astype(np.int64)
    return {"wins_games": W, "losses_games": n - W, "n_games": n, "complete": n > 0}


def selftest_fm():
    """Synthetic data (t2_mull_split.synthetic) through fm_core with the frozen settings (fast fit)."""
    import torch
    import game_fm_v2 as g2
    import t2_mull_split as t2
    torch.set_num_threads(2)
    t0 = time.time()
    fz = build_frozen()
    data, eta, _ = t2.synthetic(D=5000)
    res, scores, inputs = fm_core(data, fz, 2, 40, "SYN", log=lambda m: None,
                                  fit={**g2.FIT, "batch": 1024, "epochs": 15})
    te = data["d"].split == 2
    r = {k: float(np.corrcoef(v[0][te], eta[te])[0, 1]) for k, v in scores.items()}
    print(f"[fm] r(score, truth) test: {r}; cross-fit r {res['cross-fit r with in-sample (train drafts)']:.3f}; "
          f"H2 pair {res['pairs'][f'{CH} - {LIN}']['experienced dl']['delta']:+.3f}, H3 pair "
          f"{res['pairs'][f'{REFIT} - {CH}']['experienced dl']['delta']:+.3f} [{time.time() - t0:.0f}s]")
    assert r[CH] > 0.6 and np.isfinite(inputs["game_logit"]).all()
    assert 0.8 < res["cross-fit r with in-sample (train drafts)"] < 1.0
    json.dumps(res, default=float)
    print("fm stage self-test OK")


def selftest_lgbm():
    """Synthetic drafts through lgbm_core (sc_gll.lgbm_scores, SC-GLL, rebuild_targets.rescore)."""
    from t_common import torchfree, assert_no_torch
    torchfree()
    import lightgbm  # noqa: F401
    import sc_gll
    from decks.deckstrength import Drafts
    t0 = time.time()
    sc_gll.THREADS = 2
    import rebuild_targets
    rebuild_targets.THREADS = 2
    fz = build_frozen()
    dd, pg, deck, skill = sc_gll.synthetic(D=5000)
    D = len(dd.wins)
    rng = np.random.default_rng(0)
    C = 20
    Xd = rng.poisson(0.8, (D, C)).astype(np.float32)
    pg["own_mull"], pg["opp_mull"] = np.zeros(len(pg["y"])), np.zeros(len(pg["y"]))
    rank = np.full(D, 3)
    d = Drafts(names=[f"c{j}" for j in range(C)], draft_id=np.array([f"d{i}" for i in range(D)]),
               draft_time=np.asarray(dd.draft_time), pool=Xd.astype(np.int8), wins=dd.wins.astype(np.float32),
               losses=dd.losses.astype(np.float32), rank=rank, skill=np.full(D, np.nan), split=dd.split)
    X = np.hstack([Xd + deck[:, None] * 0.5, rank[:, None].astype(np.float32), rng.normal(size=(D, 2)).astype(np.float32)])
    fm = deck + rng.normal(0, 0.2, D)
    fm_scores = {CH: np.stack([fm, fm + rng.normal(0, 0.02, D)]), REFIT: np.stack([fm, fm]) * 1.05,
                 LIN: (deck + rng.normal(0, 0.4, D))[None]}
    inputs = {"game_logit": fm, "has_deck": np.ones(D, bool)}
    res, tres, _ = lgbm_core(d, pg, np.ones(D, bool), np.full(D, "0", dtype=object), np.asarray(d.draft_time).astype(str),
                             X.astype(np.float32), C, inputs, fm_scores, fz, 40, "SYN", log=lambda m: None,
                             n_lgbm_seeds=2, records=_synthetic_records(d, pg))
    h1 = res["pairs"][f"{CH} - {LG}"]["experienced dl"]
    h4 = tres["paired"]["round 10 - round 4 | train draft | eval draft"]
    print(f"[lgbm] H1 pair {h1['delta']:+.3f} ± {h1['se']:.3f}; H4 pair {h4['delta']:+.4f} ± {h4['se']:.4f} "
          f"[{time.time() - t0:.0f}s]")
    assert np.isfinite(h1["delta"]) and np.isfinite(h4["delta"])
    json.dumps(res, default=float)
    json.dumps(tres, default=float)
    assert_no_torch()
    print("lgbm stage self-test OK")


def selftest():
    import tempfile
    t0 = time.time()
    global VAULT
    old = VAULT
    from eval_protocol import SetResult
    with tempfile.TemporaryDirectory() as tmp:
        VAULT = Path(tmp) / "vault"
        try:
            obj = freeze(["--adopt", "- own mull", "--note", "selftest"])
            assert (VAULT / FROZEN_NAME).stat().st_mode & 0o222 == 0, "frozen json must be read-only"
            try:
                freeze([])
                raise AssertionError("second freeze must be refused")
            except SystemExit:
                pass
            fz, h, drift = load_frozen()
            assert not drift and fz["hypotheses"]["H5"]["active"]
            # fake vault results: H1 positive everywhere, H3 null; dev results under a fake dev root
            rng = np.random.default_rng(0)
            dev = Path(tmp) / "dev"

            def put(root, s, rel, pair, key, delta):
                f = root / f"{s}.PremierDraft" / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                r = json.loads(f.read_text()) if f.exists() else {"pairs": {}, "paired": {}}
                x = {"name": s, "delta": delta, "se": 0.05, "se_boot": 0.05, "se_seed": 0.0, "n_units": 1000,
                     "n_seeds": 2}
                if key is None:
                    r["paired"][pair] = x
                else:
                    r["pairs"].setdefault(pair, {})[key] = x
                f.write_text(json.dumps(r))
            for s in AFF_SETS + NOAFF_SETS:
                put(VAULT, s, "lgbm_results.json", f"{CH} - {LG}", "experienced dl", 0.5 + rng.normal(0, 0.05))
                put(VAULT, s, "fm_results.json", f"{CH} - {LIN}", "experienced dl", 0.15 + rng.normal(0, 0.05))
                put(VAULT, s, "fm_results.json", f"{REFIT} - {CH}", "experienced dl", rng.normal(-0.02, 0.03))
                put(VAULT, s, "fm_results.json", f"- own mull - {CH}", "experienced pre", rng.normal(0.0, 0.03))
                put(VAULT, s, "targets.json", "round 10 - round 4 | train draft | eval draft", None, 0.002)
            for s in ("BLB", "MSH", "DSK"):
                put(dev, s, "t1_rescore/results.json", f"{LG} - {CH}", "experienced dl", -0.6)
                put(dev, s, "t1_rescore/results.json", f"{REFIT} - {CH}", "experienced dl", 0.04)
            rep = report(VAULT, dev)
            H = rep["hypotheses"]
            assert H["H1"]["all vault sets"]["verdict"] == "CONFIRMED", H["H1"]["all vault sets"]
            assert H["H1"]["all vault sets"]["pooled dev + vault"]["sets"] == 9
            assert H["H3"]["all vault sets"]["verdict"] in ("DEV-TEST ONLY", "NOT CONFIRMED")
            assert H["H5"]["all vault sets"]["verdict"] != "CONFIRMED"
            assert sha256(VAULT / FROZEN_NAME) == h, "report must not touch the frozen json"
            # tampering is detected
            os.chmod(VAULT / FROZEN_NAME, 0o644)
            (VAULT / FROZEN_NAME).write_text(json.dumps({**obj, "note": "tampered"}))
            try:
                load_frozen()
                raise AssertionError("tampering not detected")
            except SystemExit:
                pass
            assert SetResult  # noqa
        finally:
            VAULT = old
    print(f"[freeze/report] OK [{time.time() - t0:.1f}s]")
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    for flag in ("--selftest-fm", "--selftest-lgbm"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), flag], check=True, env=env)
    print(f"t5 self-test OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--selftest-fm" in sys.argv:
        selftest_fm()
    elif "--selftest-lgbm" in sys.argv:
        selftest_lgbm()
    elif sys.argv[1:2] == ["freeze"]:
        freeze(sys.argv[2:])
    elif sys.argv[1:2] == ["run"]:
        run(sys.argv[2:])
    elif sys.argv[1:2] == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)
