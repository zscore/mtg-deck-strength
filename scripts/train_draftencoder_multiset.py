"""Replicate DraftEncoder's experiments (arXiv 2607.04782) with text features and pooled sets.

    python scripts/train_draftencoder_multiset.py sameset [models ...] [key=value ...]
    python scripts/train_draftencoder_multiset.py newset  [models ...] [key=value ...]

Models: heuristic lr lr+meta bag bag+meta bag_id (sameset only) de de+meta.
Windows follow the paper's code (github.com/akulen/MtGDraftEncoder, data.py), 14-day blocks counted
back from each set's last draft, finished runs only (7 wins or 3 losses):
  sameset  one model on BLB+FDN+TDM: train days 42-28 before the end, val 28-14, test the last 14
  newset   train days 28-14 of MOM WOE LCI MKM OTJ BLB DSK DFT TDM FIN EOE, val their last 14 days;
           test FDN's last 14 days, then fine-tune on FDN's first 4 days (3 train + 1 val, as the
           code splits it) and test again
META is computed from games of each set's *training* drafts only (FDN in newset: everything before
its test window); the paper's code uses the whole game file, test period included.
Results: results/draftencoder_multiset/{mode}.json (merged across runs).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openlane.data import ROOT  # noqa: E402
from openlane.de_multiset import (DraftEncoderMS, LinearMS, SetData, Task, card_meta, fit,  # noqa: E402
                                  heuristic, predict, spearman, weighted_nll)
from openlane.deckstrength import binomial_nll  # noqa: E402

ALL12 = ["MOM", "WOE", "LCI", "MKM", "OTJ", "BLB", "DSK", "FDN", "DFT", "TDM", "FIN", "EOE"]
SAMESET = ["BLB", "FDN", "TDM"]
NEWSET_TRAIN = [s for s in ALL12 if s != "FDN"]
OUT = ROOT / "results" / "draftencoder_multiset"


def finished(s: SetData, idx):
    return idx[(s.wins[idx] == 7) | (s.losses[idx] == 3)]


def text_norm(sets):
    """Standardise text-embedding dims over the cards of all 12 sets (no outcome information)."""
    allt = np.vstack([s.text for s in sets])
    return allt.mean(0), allt.std(0) + 1e-6


def feats(s, mu, sd, meta):
    x = (s.text - mu) / sd
    return np.hstack([x, meta]) if meta is not None else x


def evaluate(name, model_or_preds, tasks, device, results, mode, extra=None):
    preds = model_or_preds if isinstance(model_or_preds, list) else predict(model_or_preds, tasks, device)
    per_set = {}
    for p, t in zip(preds, tasks):
        w, l = t.s.wins[t.idx], t.s.losses[t.idx]
        per_set[t.s.set] = {
            "spearman": spearman(p[:, -1], t.s, t.idx),
            "nll_final": binomial_nll(p[:, -1], w, l), "weighted_nll": weighted_nll(p, w, l),
            "spearman_by_pick": [round(spearman(p[:, i], t.s, t.idx), 4) for i in range(p.shape[1])],
            "n": len(t.idx)}
    res = {"spearman_mean": float(np.mean([v["spearman"] for v in per_set.values()])),
           "per_set": per_set, **(extra or {})}
    if len(tasks) > 1 and not name.startswith("heuristic"):  # pooled over sets (comparable scale)
        P = np.concatenate([p[:, -1] for p in preds])
        W = np.concatenate([t.s.wins[t.idx] for t in tasks])
        L = np.concatenate([t.s.losses[t.idx] for t in tasks])
        from scipy.stats import spearmanr
        res["spearman_pooled"] = float(spearmanr(P, W / (W + L)).statistic)
    results[name] = res
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{mode}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old[name] = res
    path.write_text(json.dumps(old, indent=1))
    print(f"  == {name}: spearman mean {res['spearman_mean']:.4f} "
          + " ".join(f"{k} {v['spearman']:.4f}" for k, v in per_set.items())
          + (f" pooled {res['spearman_pooled']:.4f}" if "spearman_pooled" in res else ""), flush=True)
    return preds


def build_model(kind, F, n_ids, cfg):
    if kind == "de":
        return DraftEncoderMS(F, d=cfg["d"], heads=2, set_layers=cfg["set_layers"],
                              draft_layers=cfg["draft_layers"], dropout=cfg["dropout"])
    return LinearMS(kind, F, T=45, n_ids=n_ids)


def main(mode, *args):
    models = [a for a in args if "=" not in a] or ["heuristic", "lr", "lr+meta", "bag", "bag+meta", "de", "de+meta"]
    cfg = {"d": 64, "set_layers": 10, "draft_layers": 5, "dropout": 0.3, "epochs": 15, "lin_epochs": 15,
           "ft_epochs": 10, "batch": 256, "lr": 1e-4, "lin_lr": 3e-3, "lr_lr": 1e-4, "wd": 0.05, "seed": 0, "max_minutes": 0,
           "per_set": 0, "tag": ""}
    for a in args:
        if "=" in a:
            k, v = a.split("=")
            cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], int) or "." not in v else float(v)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    log = lambda m: print(m, flush=True)
    sets = {s: SetData(s) for s in (SAMESET if mode == "sameset" else ALL12)}
    mu, sd = text_norm([SetData(s) if s not in sets else sets[s] for s in ALL12]) if mode == "newset" \
        else text_norm(list(sets.values()))
    rng = np.random.default_rng(cfg["seed"])
    D14 = 14
    win = {}
    if mode == "sameset":
        for name, s in sets.items():
            win[name] = {"train": finished(s, s.window(3 * D14, 2 * D14)),
                         "val": finished(s, s.window(2 * D14, D14)), "test": finished(s, s.window(D14, 0))}
            win[name]["meta_from"] = win[name]["train"]
        train_sets, test_sets = SAMESET, SAMESET
    else:
        for name, s in sets.items():
            if name == "FDN":
                ft = finished(s, s.first_days(4))
                day3 = s.first_days(3)
                win[name] = {"test": finished(s, s.window(D14, 0)),
                             "ft_train": np.intersect1d(ft, day3), "ft_val": np.setdiff1d(ft, day3),
                             "meta_from": s.window(10_000, D14)}
            else:
                tr = finished(s, s.window(2 * D14, D14))
                if cfg["per_set"] and len(tr) > cfg["per_set"]:
                    tr = np.sort(rng.choice(tr, cfg["per_set"], replace=False))
                win[name] = {"train": tr, "val": finished(s, s.window(D14, 0)), "meta_from": tr}
        train_sets, test_sets = NEWSET_TRAIN, ["FDN"]
    for name in sets:
        log(f"{name}: " + ", ".join(f"{k} {len(v)}" for k, v in win[name].items() if k != "meta_from"))
    meta, meta_raw = {}, {}
    for name, s in sets.items():
        meta[name], meta_raw[name] = card_meta(s, win[name]["meta_from"])
    log(f"data ready [{time.time() - t0:.0f}s]  cfg {cfg}")

    offsets = np.cumsum([0] + [sets[s].C for s in sets])
    id_off = dict(zip(sets, offsets[:-1]))
    results = {}
    tag = cfg["tag"]

    def tasks(names, split, use_meta):
        return [Task(sets[n], feats(sets[n], mu, sd, meta[n] if use_meta else None), win[n][split],
                     int(id_off[n])) for n in names]

    for m in models:
        kind, use_meta = m.split("+")[0], m.endswith("+meta")
        t1 = time.time()
        if kind == "heuristic":  # mean in-deck (GP) win rate of the best 23 picks, as in the paper's code
            te = tasks(test_sets, "test", False)
            evaluate("heuristic", [heuristic(t.s, t.idx, meta_raw[t.s.set][:, 2]) for t in te], te, device,
                     results, mode)
            continue
        if kind == "bag_id" and mode != "sameset":
            continue
        tr, va, te = (tasks(train_sets, "train", use_meta), tasks(train_sets, "val", use_meta),
                      tasks(test_sets, "test", use_meta))
        F = tr[0].feats.shape[1]
        model = build_model(kind, F, int(offsets[-1]), cfg)
        n_par = sum(p.numel() for p in model.parameters())
        is_de = kind == "de"
        # the per-slot LR has 45x45x768 weights and diverges at the bag models' learning rate
        lin_lr = cfg["lr_lr"] if kind == "lr" else cfg["lin_lr"]
        log(f"training {m}{tag}: {n_par / 1e6:.2f}M params, {sum(len(t.idx) for t in tr)} train drafts")
        model, hist = fit(model, tr, va, epochs=cfg["epochs"] if is_de else cfg["lin_epochs"], batch=cfg["batch"],
                          lr=cfg["lr"] if is_de else lin_lr, wd=cfg["wd"] if is_de else 0.0,
                          seed=cfg["seed"], device=device, log=log,
                          max_minutes=cfg["max_minutes"] or None)
        extra = {"params": n_par, "history": hist, "train_s": round(time.time() - t1), "cfg": cfg}
        evaluate(f"{m}{tag}", model, te, device, results, mode, extra)
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), OUT / "models" / f"{mode}_{m}{tag}.pt")
        if mode == "newset":
            ft_tr, ft_va = tasks(["FDN"], "ft_train", use_meta), tasks(["FDN"], "ft_val", use_meta)
            log(f"fine-tuning {m}{tag} on FDN first 4 days: {len(ft_tr[0].idx)} train, {len(ft_va[0].idx)} val")
            model, hist = fit(model, ft_tr, ft_va, epochs=cfg["ft_epochs"], batch=cfg["batch"],
                              lr=cfg["lr"] if is_de else lin_lr, wd=cfg["wd"] if is_de else 0.0,
                              seed=cfg["seed"], device=device, log=log)
            evaluate(f"{m}{tag}:finetune4d", model, te, device, results, mode, {"history": hist})
            # the same model trained on the 4 days alone, to see what the other sets contribute
            scratch = build_model(kind, F, int(offsets[-1]), cfg)
            log(f"from scratch {m}{tag} on FDN first 4 days")
            scratch, hist = fit(scratch, ft_tr, ft_va, epochs=cfg["ft_epochs"] * 2, batch=cfg["batch"],
                                lr=cfg["lr"] if is_de else lin_lr, wd=cfg["wd"] if is_de else 0.0,
                                seed=cfg["seed"], device=device, log=log)
            evaluate(f"{m}{tag}:scratch4d", scratch, te, device, results, mode, {"history": hist})
        log(f"  [{time.time() - t1:.0f}s, total {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(*sys.argv[1:])
