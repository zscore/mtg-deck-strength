"""Which game-context controls should a skill-adjusted deck rating use? python scripts/game_controls.py MSH [SET ...]

Round 4's reference game model (`linear + deck features + aff`: card power + Scryfall deck features + affinity column;
the affinity column is left out when the set's picknet pool model is missing)
refit with different context columns. Some controls may soak up deck strength: the record before a game is partly
caused by the deck (though it also stands for record-based matchmaking), and mulligans are partly caused by the mana
base. Dropping a control can't be judged on game log loss (fewer covariates always fit worse), so each variant's
deck score (card power + deck features + affinity, no context) is calibrated to the draft records like the other
heuristics and scored on test drafts, and its card values are compared with the full model's.
  full              play/draw, mulligan difference, skill bucket, rank, opponent rank (always None), experience, record
  - record          without the record before the game
  - mulligans       without the mulligan difference
  - record, mull    without both
  - skill           without the skill bucket (how much the skill adjustment changes the ratings)
Writes results/{SET}.PremierDraft/game_controls.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_synergy_v2 import pick_affinity  # noqa: E402
from deck_synergy_v3 import Profile  # noqa: E402
from deck_synergy_v4 import train  # noqa: E402
from mtgdata.cardinfo import card_info, deck_features  # noqa: E402
from mtgdata.games import aligned, flag, game_path  # noqa: E402
from decks import ROOT  # noqa: E402
from decks.deckstrength import (N_RANKS, calibrate, covariates, deck_metrics, experience, game_split,  # noqa: E402
                                load_drafts, most_played_deck, record_before, record_onehot)
from decks.synergy import DeckFM  # noqa: E402

MULL, SKILL = [1], [2, 3]  # columns of covariates(): play/draw, mulligans, skill, skill known, ranks, opp ranks


def main(*sets):
    torch.set_num_threads(4)
    for set_code in sets:
        t0 = time.time()
        d = load_drafts(set_code, "PremierDraft")
        g = pl.read_parquet(game_path(set_code, "PremierDraft"))
        sp = game_split(d, g["draft_id"])
        g, sp = g.filter(pl.Series(sp >= 0)), sp[sp >= 0]
        X = aligned(g, "deck_", d.names).astype(np.float32)
        y = flag(g["won"]).astype(np.int8)
        wb, lb, _ = record_before(g)
        cov = covariates(g)
        cov = cov[:, : 4 + N_RANKS]  # opponent rank is None in every 17Lands game file: a constant
        exp_, rec = experience(g), record_onehot(wb, lb)
        deck, has = most_played_deck(d, g)
        deck = deck.astype(np.float32)
        del g
        info = card_info(d.names, set_code)
        nonland = (~info["is_land"].to_numpy()).astype(np.float32)
        F, _ = deck_features(X, info)
        mu, sd = F[sp == 0].mean(0), F[sp == 0].std(0) + 1e-6
        aff = pick_affinity(set_code, d.names)  # None when the set's picknet pool model is missing
        prof = None if aff is None else Profile(X, aff[0], nonland, sp == 0)
        extra = lambda Xd: np.hstack([(deck_features(Xd, info)[0] - mu) / sd]  # noqa: E731
                                     + ([] if prof is None else [prof(Xd, slice(0, 1))])).astype(np.float32)
        E, Ed = extra(X), extra(deck)
        keep_all = np.ones(cov.shape[1], bool)
        variants = {"full": (keep_all, True), "- record": (keep_all, False)}
        for name, drop, rec_on in (("- mulligans", MULL, True), ("- record, mull", MULL, False), ("- skill", SKILL, True)):
            k = keep_all.copy()
            k[drop] = False
            variants[name] = (k, rec_on)
        trd, ted = d.mask("train"), d.mask("test") & has
        res, weights = {}, {}
        for name, (k, rec_on) in variants.items():
            ctx = np.hstack([cov[:, k], exp_] + ([rec] if rec_on else [])).astype(np.float32)
            c = np.hstack([ctx, E]).astype(np.float32)
            fits = [train(lambda: DeckFM(X.shape[1], c.shape[1], 0), X, c, y, sp, *kw) + (kw,)
                    for kw in [(1e-5, 0), (1e-4, 0)]]
            m, v, _, kw = min(fits, key=lambda f: f[1])
            with torch.no_grad():
                s = m.deck_score(torch.from_numpy(deck)).numpy().astype(np.float64)
            s = s + Ed @ m.cov.weight.detach().numpy()[0, ctx.shape[1]:]
            p = calibrate(s[trd], d.rank[trd], d.wins[trd], d.losses[trd], s, d.rank)
            weights[name] = m.w.detach().numpy().copy()
            r = deck_metrics(p[ted], d.wins[ted], d.losses[ted])
            res[name] = {"draft spearman": r["spearman"], "draft nll": r["nll"], "val game logloss": v,
                         "setting": kw, "deck score sd (test drafts)": float(s[ted].std())}
            print(f"  {set_code} {name:16s} draft spearman {r['spearman']:.4f}  deck score sd {s[ted].std():.3f}  "
                  f"val {v:.5f} [{time.time() - t0:.0f}s]", flush=True)
        if True:
            played = X[sp == 0].sum(0) > 2000
            for name, w in weights.items():
                res[name]["card value r with full"] = float(spearmanr(w[played], weights["full"][played]).statistic)
        out = ROOT / "results" / f"{set_code}.PremierDraft" / "game_controls.json"
        res["affinity column"] = prof is not None
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
