"""Deck model with PickNet features: python scripts/improve_deck_picknet.py MSH [SET ...]

(after deck_synergy_v4.py and picknet_features.py.) Adds three groups of PickNet features to round 4's LightGBM
(registered-deck counts + structure + rank + Scryfall deck features + affinity column):
  deck vec   count-weighted mean of PickNet's card vectors over the deck's nonland cards, PCA to 16 (fit on training
             decks): what the deck is made of in PickNet's learned card space. A deck feature.
  state      PickNet's history state after the last pick, PCA to 16: its summary of the whole draft (pool, signals,
             what was passed). Mostly a pool / draft feature.
  surprise   mean pick NLL (all and non-forced picks, and per pack) and top-1 agreement with PickNet, skill buckets
             hidden: how much the drafter's picks differ from a typical drafter's at their rank. A drafter feature,
             i.e. a non-leaky skill proxy rather than deck strength.
Each is fit on training drafts (early-stopped on val), then averaged with the cross-fitted game FM (weight picked on
val), as round 4. Test drafts with a deck; paired bootstrap of the Spearman difference against round 4's blend.
Writes results/{SET}.PremierDraft/picknet/improve.json.
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve_deck import structure_features  # noqa: E402
from improve_deck_v2 import lgbm, z  # noqa: E402
from mtgdata.data import download_cards  # noqa: E402
from mtgdata.games import game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import binomial_nll, calibrate, deck_metrics, load_drafts, most_played_deck  # noqa: E402


def picknet_inputs(set_code, d, deck, tr):
    f = np.load(ROOT / "results" / f"{set_code}.PremierDraft" / "picknet" / "feats.npz", allow_pickle=True)
    row = {x: i for i, x in enumerate(f["draft_id"].tolist())}
    k = np.array([row.get(x, -1) for x in d.draft_id.tolist()])
    ok = k >= 0
    k = np.where(ok, k, 0)
    pick = lambda a: np.where(ok.reshape(-1, *[1] * (a.ndim - 1)), a[k], np.nan).astype(np.float32)  # noqa: E731

    surprise = np.column_stack([pick(f["nll"]), pick(f["nll_nf"]), pick(f["top1_nf"]), pick(f["nll_pack"])])
    st = pick(f["state"].astype(np.float32))
    good = tr & ok
    state = np.full((len(d.wins), 16), np.nan, np.float32)
    state[ok] = PCA(16, random_state=0).fit(st[good]).transform(st[ok])

    vec = dict(zip(f["card_names"].tolist(), f["card_vec"]))
    V = np.stack([vec.get(n, np.zeros(f["card_vec"].shape[1], np.float32)) for n in d.names])
    cards = pl.read_csv(download_cards(), columns=["name", "types"]).unique("name")
    types = pl.DataFrame({"name": d.names}).join(cards, on="name", how="left")["types"].fill_null("")
    nonland = ~types.str.contains("Land").to_numpy()
    x = deck.astype(np.float32) * nonland
    mean = (x @ V) / np.maximum(x.sum(1, keepdims=True), 1)
    deckvec = PCA(16, random_state=0).fit(mean[tr]).transform(mean).astype(np.float32)
    return {"deck vec": deckvec, "state": state, "surprise": surprise}, float(ok.mean())


def main(*sets):
    for set_code in sets:
        t0 = time.time()
        out = ROOT / "results" / f"{set_code}.PremierDraft"
        inp = np.load(out / "deck_synergy_v4" / "draft_inputs.npz")
        d = load_drafts(set_code, "PremierDraft")
        deck, has = most_played_deck(d, pl.read_parquet(game_path(set_code, "PremierDraft")))
        assert (has == inp["has_deck"]).all()
        tr, va, te = d.mask("train") & has, d.mask("val") & has, d.mask("test") & has
        base = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                          d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
        groups, cover = picknet_inputs(set_code, d, deck, tr)
        game = calibrate(inp["game_logit"][tr], d.rank[tr], d.wins[tr], d.losses[tr], inp["game_logit"], d.rank)
        vnll = lambda p: binomial_nll(p[va], d.wins[va], d.losses[va])  # noqa: E731
        variants = {"round 4: lgbm": [], "+ deck vec": ["deck vec"], "+ state": ["state"],
                    "+ surprise": ["surprise"], "+ deck vec + state": ["deck vec", "state"],
                    "+ all three": ["deck vec", "state", "surprise"]}
        alphas = np.linspace(0, 1, 21)
        preds, info = {}, {}
        for name, gs in variants.items():
            X = np.hstack([base, *[groups[g] for g in gs]])
            p, n = lgbm(X, d, tr, va)
            a = float(alphas[np.argmin([vnll(expit((1 - a) * z(p) + a * z(game))) for a in alphas])])
            preds[name], preds[f"{name} + game FM"] = p, expit((1 - a) * z(p) + a * z(game))
            info[name], info[f"{name} + game FM"] = {"trees": n, "val_nll": vnll(p)}, {"alpha": a}

        ti = np.flatnonzero(te & ((d.wins + d.losses) > 0))
        wr = d.wins[ti] / (d.wins[ti] + d.losses[ti])
        rng = np.random.default_rng(0)
        boots = [rng.integers(0, len(ti), len(ti)) for _ in range(300)]
        rows = {}
        for k, p in preds.items():
            ref = preds["round 4: lgbm + game FM" if k.endswith("game FM") else "round 4: lgbm"]
            dl = [spearmanr(p[ti][b], wr[b]).statistic - spearmanr(ref[ti][b], wr[b]).statistic for b in boots]
            r = deck_metrics(p[te], d.wins[te], d.losses[te])
            r.update({"vs round 4": r["spearman"] - deck_metrics(ref[te], d.wins[te], d.losses[te])["spearman"],
                      "vs round 4 sd": float(np.std(dl)), "info": info[k]})
            rows[k] = r
        (out / "picknet" / "improve.json").write_text(json.dumps({"coverage": cover, "rows": rows}, indent=1))
        print(f"=== {set_code} ({te.sum()} test drafts, picknet coverage {cover:.3f}, {time.time() - t0:.0f}s)", flush=True)
        for k, r in rows.items():
            print(f"  {k:36s} spearman {r['spearman']:.4f} nll {r['nll']:.5f}  "
                  f"vs round 4 {r['vs round 4']:+.4f} ± {r['vs round 4 sd']:.4f}  {r['info']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
