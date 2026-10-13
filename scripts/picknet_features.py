"""PickNet features for the deck model: python scripts/picknet_features.py MSH [SET ...]

Runs picknet's all-sets model (`all_d256_ep3`: 8.4M params, 17 sets, ../picknet/results/picknet_all) over every
complete PremierDraft draft of a set and saves, per draft (in the shard's draft order, with its draft_id):
  nll / nll_nf / top1_nf     mean pick NLL over valid picks, NLL and top-1 agreement over non-forced picks
  nll_pack                   mean pick NLL in each of the three packs
  state                      the history state after the last pick (256-d): the model's summary of the draft
and, per card of the set, the model's card vector (`card_vec`, 256-d, aligned to `card_names`).

The skill buckets in PickNet's context (user_game_win_rate_bucket, user_n_games_bucket) are set to "unknown":
the win-rate bucket is partly computed from the same games the deck model predicts. Rank and time are kept.
PickNet was trained on its own train split of every set (chronological 70/15/15, as here), so these features
are in-sample for training drafts; `split` is saved so the deck model can check that.
Writes results/{SET}.PremierDraft/picknet/feats.npz.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.data import FEATURES, download  # noqa: E402
from decks import ROOT  # noqa: E402

PICKNET = Path(__file__).resolve().parents[2] / "picknet"
sys.path.insert(0, str(PICKNET / "scripts"))
from train_picknet_all import N_PICKS, Shard, grow, to_torch  # noqa: E402
from picknet.model import PickNet  # noqa: E402

MODEL = PICKNET / "results" / "picknet_all" / "models" / "all_d256_ep3.pt"
SHARDS = FEATURES / "picks_all"
SKILL = slice(7, 11)  # ctx columns: wr (centred), wr unknown, log10 n games, n games unknown
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def shard_draft_ids(set_code: str, n_picks: int, p1p1_missing: bool) -> np.ndarray:
    """draft_id of each shard row: load_tensors' order (complete drafts by first-seen time, then draft_id)."""
    keys = pl.scan_parquet(download(set_code, "PremierDraft")).select("draft_id", "draft_time").collect()
    T = 3 * n_picks - p1p1_missing
    k = (keys.group_by("draft_id").agg(pl.len(), pl.col("draft_time").min().alias("t0"))
         .filter(pl.col("len") == T).sort("t0", "draft_id"))
    return k["draft_id"].to_numpy()


def load():
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    cards = np.load(SHARDS / "cards.npz")
    cfg = ck["cfg"]
    model = PickNet(len(cards["name_id"]), N_PICKS, d=cfg["d"], heads=cfg["heads"], hist_layers=cfg["hist_layers"],
                    pack_layers=cfg["pack_layers"], dropout=cfg["dropout"],
                    feats=(cards["feats"] - ck["feat_mu"]) / ck["feat_sd"], id_map=cards["name_id"],
                    n_ctx=ck["n_ctx"], n_sets=len(cards["sets"]), deck_head=cfg["deck_w"] > 0)
    # the card table is append-only and has grown since this model was saved; grow() maps its weights
    model.load_state_dict(grow(ck["state"], model, list(ck["sets"]), list(cards["sets"])))
    return model.to(DEV).eval(), cards, list(ck["sets"])


@torch.no_grad()
def main(*sets):
    model, cards, model_sets = load()
    tab = model.cards().float().cpu().numpy()
    for set_code in sets:
        t0 = time.time()
        sh = Shard(SHARDS / f"{set_code}.PremierDraft", len(cards["name_id"]))
        assert sh.info["set"] in model_sets, f"{set_code} is not one of the model's sets"
        D = sh.info["drafts"]
        ids = shard_draft_ids(set_code, sh.info["n_picks"], sh.info["p1p1_missing"])
        assert len(ids) == D, (len(ids), D)
        out = {k: np.zeros(D, np.float32) for k in ("nll", "nll_nf", "top1_nf")}
        out["nll_pack"] = np.zeros((D, 3), np.float32)
        out["state"] = np.zeros((D, model.pos.weight.shape[1]), np.float16)
        for a in range(0, D, 256):
            rows = np.arange(a, min(a + 256, D))
            b = sh.get(rows)
            b["ctx"] = b["ctx"].copy()
            b["ctx"][:, SKILL] = np.array([0, 1, 0, 1], np.float32)
            x = [v.to(DEV) for v in to_torch(b)[:7]]
            target, valid = torch.from_numpy(b["target"]).long().to(DEV), torch.from_numpy(b["valid"]).to(DEV)
            logit, h, _ = model(*x, return_state=True)
            lp = logit.float().masked_fill(x[1] == 0, float("-inf")).log_softmax(-1)
            nll = -lp.gather(-1, target[..., None]).squeeze(-1)
            nf = valid & ((x[1] > 0).sum(-1) > 1)
            hit = (lp.argmax(-1) == target) & nf
            v, nff = valid.float(), nf.float()
            nll = nll.masked_fill(~valid, 0)
            out["nll"][rows] = ((nll * v).sum(1) / v.sum(1).clamp(min=1)).cpu().numpy()
            out["nll_nf"][rows] = ((nll * nff).sum(1) / nff.sum(1).clamp(min=1)).cpu().numpy()
            out["top1_nf"][rows] = (hit.float().sum(1) / nff.sum(1).clamp(min=1)).cpu().numpy()
            vp, np_ = v.view(len(rows), 3, N_PICKS), nll.view(len(rows), 3, N_PICKS)
            out["nll_pack"][rows] = ((np_ * vp).sum(2) / vp.sum(2).clamp(min=1)).cpu().numpy()
            last = (torch.arange(h.shape[1], device=DEV) * valid).argmax(1)  # last valid pick slot
            out["state"][rows] = h[torch.arange(len(rows), device=DEV), last].float().cpu().numpy()
        sel = cards["set_of"] == list(cards["sets"]).index(set_code)
        dst = ROOT / "results" / f"{set_code}.PremierDraft" / "picknet"
        dst.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst / "feats.npz", draft_id=ids, split=sh.split, card_vec=tab[:-1][sel],
                            card_names=cards["printing"][sel], **out)
        by = {s: float(out["nll_nf"][sh.split == i].mean()) for i, s in enumerate(("train", "val", "test"))}
        print(f"{set_code}: {D} drafts, non-forced NLL by split {by} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
