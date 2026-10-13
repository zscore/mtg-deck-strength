"""DRAFT (not run): per-set data prep for candidates A and B, and the cross-fitted LightGBM-blend teacher.

    python results/methodology/drafts/prepare_next_model.py MSH [SET ...]       # ~3-6 min/set, < 1.5 GB RAM

Writes results/{SET}.PremierDraft/next_model/{data.npz, teacher_train.npz, teacher_trainval.npz}. Only syntax-checked so far; every step reuses
functions that the round 4 / round 10 scripts already run, so the open risk is glue, not method.

RAM: the game file is never read whole. Two lazy scans: (1) one row per (draft_id, build_index) with the deck_
columns (builds, ~1.2 per draft), (2) every game row with ~10 narrow columns. Peak is the build matrix
(~150k builds x ~340 cards, int8 = 50 MB) plus polars overhead.

data.npz, per draft (load_drafts order) with G = 9 game slots:
  build_ids [B, 48] int16 card columns of each build's distinct cards (-1 pad), build_cnt [B, 48] int8,
  build_feat [B, F] float16 (Scryfall deck features + affinity column, standardised on train),
  build_of_draft_main [D] (the most-played build: the one the draft-level metric rates),
  game_build [D, G] int32, game_state [D, G] int8 (21 = record unknown), game_y [D, G] int8, game_mask [D, G] bool,
  game_cov [D, G, P] float16 (on_play - 0.5, mulligan difference, experience one-hot), game_rank [D, G] int8,
  split [D], wins [D], losses [D], rank [D]; card_vec [C, 256] (PickNet all_d256, aligned to d.names; zeros + flag
  for cards PickNet lacks).
teacher_{train,trainval}.npz: teacher [D] float32 logit, has [D] bool. K-fold out-of-fold over train (screening) or
  train + val (final refit) drafts (the LightGBM part);
  the game-FM part is round 4's 2-fold cross-fit (draft_inputs.npz:game_logit). Test drafts: NaN (never used).
"""
if __name__ == "__main__":
    import lightgbm  # noqa: F401  (must load before torch, see improve_deck.py)

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

G_MAX, MAX_TOK, K_FOLDS = 9, 48, 5


def build_tables(set_code, d, info, train_draft_mask):
    import polars as pl
    from mtgdata.games import aligned, flag, game_path
    from decks.deckstrength import record_before, rank_index, experience
    path = game_path(set_code, "PremierDraft")
    lf = pl.scan_parquet(path)
    deck_cols = [f"deck_{n}" for n in d.names]
    builds = (lf.select("draft_id", "build_index", *deck_cols)
              .unique(["draft_id", "build_index"], keep="first", maintain_order=True)
              .collect(engine="streaming"))
    pos = {x: i for i, x in enumerate(d.draft_id.tolist())}
    builds = builds.filter(pl.col("draft_id").is_in(list(pos)))
    Xb = aligned(builds, "deck_", d.names).astype(np.int8)
    bkey = {(a, b): i for i, (a, b) in enumerate(zip(builds["draft_id"].to_list(), builds["build_index"].to_list()))}
    games = (lf.select("draft_id", "build_index", "match_number", "won", "on_play", "num_mulligans",
                       "opp_num_mulligans", "rank", "user_n_games_bucket")
             .filter(pl.col("draft_id").is_in(list(pos))).collect(engine="streaming"))
    wb, lb, known = record_before(games)
    from mm_likelihood import state_index  # results/methodology/drafts
    st = state_index(wb, lb, known)
    D = len(d.draft_id)
    gb = np.full((D, G_MAX), -1, np.int32)
    gs = np.full((D, G_MAX), 21, np.int8)
    gy = np.zeros((D, G_MAX), np.int8)
    gm = np.zeros((D, G_MAX), bool)
    cov_g = np.hstack([flag(games["on_play"])[:, None] - 0.5,
                       (games["num_mulligans"].cast(pl.Float32, strict=False)
                        - games["opp_num_mulligans"].cast(pl.Float32, strict=False)).fill_null(0).to_numpy()[:, None],
                       experience(games)]).astype(np.float16)
    gc = np.zeros((D, G_MAX, cov_g.shape[1]), np.float16)
    gr = np.zeros((D, G_MAX), np.int8)
    rk = rank_index(games["rank"])
    order = np.lexsort((games["match_number"].to_numpy(), games["draft_id"].to_numpy()))
    slot = {}
    y = flag(games["won"]).astype(np.int8)
    did, bidx = games["draft_id"].to_list(), games["build_index"].to_list()
    for r in order:  # python loop over ~1M games: ~10 s; vectorise later if needed
        i = pos[did[r]]
        k = slot.get(i, 0)
        if k >= G_MAX:
            continue
        slot[i] = k + 1
        gb[i, k], gs[i, k], gy[i, k], gm[i, k] = bkey[(did[r], bidx[r])], st[r], y[r], True
        gc[i, k], gr[i, k] = cov_g[r], rk[r]
    # main build = most games (ties: lowest build_index), as decks.deckstrength.most_played_deck
    main = np.full(D, -1, np.int64)
    for i in range(D):
        b = gb[i][gm[i]]
        if len(b):
            v, c = np.unique(b, return_counts=True)
            main[i] = v[np.argmax(c)]
    ids = np.full((len(Xb), MAX_TOK), -1, np.int16)
    cnt = np.zeros((len(Xb), MAX_TOK), np.int8)
    for j, row in enumerate(Xb):
        nz = np.flatnonzero(row)[:MAX_TOK]
        ids[j, :len(nz)], cnt[j, :len(nz)] = nz, row[nz]
    return dict(Xb=Xb, build_ids=ids, build_cnt=cnt, build_of_draft_main=main, game_build=gb, game_state=gs,
                game_y=gy, game_mask=gm, game_cov=gc, game_rank=gr)


def build_features(set_code, d, Xb, main, info):
    """Round 4's deck features + affinity column on every build, standardised on the main builds of train drafts."""
    from mtgdata.cardinfo import deck_features
    from deck_synergy_v2 import pick_affinity
    from deck_synergy_v3 import Profile
    nonland = (~info["is_land"].to_numpy()).astype(np.float32)
    S, _ = pick_affinity(set_code, d.names)
    tr = np.zeros(len(Xb), bool)
    tr[main[d.mask("train") & (main >= 0)]] = True
    F, _ = deck_features(Xb.astype(np.float32), info)
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    prof = Profile(Xb.astype(np.float32), S, nonland, tr)
    return np.hstack([(F - mu) / sd, prof(Xb.astype(np.float32), slice(0, 1))]).astype(np.float16)


def picknet_vectors(set_code, names):
    from decks import ROOT
    f = np.load(ROOT / "results" / f"{set_code}.PremierDraft" / "picknet" / "feats.npz", allow_pickle=True)
    vec = dict(zip(f["card_names"].tolist(), f["card_vec"]))
    out = np.zeros((len(names), f["card_vec"].shape[1]), np.float32)
    have = np.array([n in vec for n in names])
    for i, n in enumerate(names):
        if n in vec:
            out[i] = vec[n]
    return out, have


def oof_teacher(set_code, d, deck, has_, scope="train"):
    """scope="train": folds over train drafts only (for screening runs, which early-stop on val: a teacher fit on val
    outcomes would leak them into the training targets). scope="trainval": folds over train + val (final refits).
    K-fold OOF LightGBM (round 10 settings: recency tau, tuned params, trees scaled) over train+val drafts,
    blended in logit space with round 4's cross-fitted game FM at round 10's validation-picked weight."""
    import json
    from scipy.special import logit
    from improve_deck import structure_features
    from improve_deck_v2 import lgbm, recency
    from decks import ROOT
    from decks.deckstrength import calibrate
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v4"
    inp = np.load(out / "draft_inputs.npz")
    info = json.loads((out / "improve_v2.json").read_text())
    cfg = info["recency + tuned"]["info"]
    params, tau, n_tr = cfg["params"], cfg["tau"], cfg["trees"]
    alpha = info["recency + tuned, refit, 5 seeds + game FM"]["info"]["alpha"]
    assert (has_ == inp["has_deck"]).all()
    X = np.hstack([deck.astype(np.float32), structure_features(deck, d.names, set_code),
                   d.rank[:, None].astype(np.float32), inp["deck_features"], inp["aff"][:, None]])
    trva = (d.mask("train") | (d.mask("val") if scope == "trainval" else False)) & has_
    fold = np.random.default_rng(7).integers(0, K_FOLDS, len(d.wins))
    teach = np.full(len(d.wins), np.nan)
    for k in range(K_FOLDS):
        fit = trva & (fold != k)
        n_k = int(round(n_tr * fit.sum() / (d.mask("train") & has_).sum()))
        p, _ = lgbm(X, d, fit, w=recency(d, fit, tau), params=params, n_iter=n_k, seed=k)
        sel = trva & (fold == k)
        teach[sel] = logit(np.clip(p[sel], 1e-6, 1 - 1e-6))
    tr = d.mask("train") & has_
    game = calibrate(inp["game_logit"][tr], d.rank[tr], d.wins[tr], d.losses[tr], inp["game_logit"], d.rank)
    teach = (1 - alpha) * teach + alpha * logit(np.clip(game, 1e-6, 1 - 1e-6))
    return teach.astype(np.float32), trva


def main(*sets):
    from mtgdata.cardinfo import card_info
    from decks import ROOT
    from decks.deckstrength import load_drafts
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for set_code in sets:
        out = ROOT / "results" / f"{set_code}.PremierDraft" / "next_model"
        out.mkdir(parents=True, exist_ok=True)
        d = load_drafts(set_code, "PremierDraft")
        info = card_info(d.names, set_code)
        t = build_tables(set_code, d, info, d.mask("train"))
        Xb, main_b = t.pop("Xb"), t["build_of_draft_main"]
        feat = build_features(set_code, d, Xb, main_b, info)
        deck = np.where((main_b >= 0)[:, None], Xb[np.maximum(main_b, 0)], 0)  # = most_played_deck
        vec, have = picknet_vectors(set_code, d.names)
        np.savez_compressed(out / "data.npz", build_feat=feat, card_vec=vec.astype(np.float16), card_vec_have=have,
                            split=d.split, wins=d.wins, losses=d.losses, rank=d.rank, draft_id=d.draft_id, **t)
        for scope in ("train", "trainval"):
            teach, has = oof_teacher(set_code, d, deck, main_b >= 0, scope)
            np.savez_compressed(out / f"teacher_{scope}.npz", teacher=teach, has=has)
        print(f"{set_code}: {len(d.wins)} drafts, {len(feat)} builds, teacher on {has.sum()} drafts", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
