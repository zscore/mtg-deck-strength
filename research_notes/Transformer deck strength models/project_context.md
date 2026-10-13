# Project context and in-house assets (from the coordinator, not web research)

## Current deck model (this repo, README)
- Target: event record W/L per draft (Bo1, 7-win/3-loss stopping), binomial likelihood; metric Spearman(p, W/(W+L)) on last 15% of drafts (time split 70/15/15); 12 sets.
- Mean Spearman over 12 sets: linear bag-of-pool 0.212; linear registered deck 0.244; LightGBM on deck counts + structure + rank 0.271; + Scryfall deck features + pick-model affinity column 0.275; averaged with game-level FM (skill-adjusted card power + rank-4 nonland pair synergy, trained on per-game rows) 0.279 (round 4 best).
- Transformers tried: per-set pool transformer 0.235 (MSH); DraftEncoder + deck stage 0.258-0.266 (MSH), beats linear deck on FDN/BLB (larger sets) but not LightGBM (0.283/0.276); logits correlate 0.86-0.93 with LightGBM/linear; ensembling adds ~0.004 or less. Shared cross-set deck transformer on card features (text embedding + Scryfall) on residuals of linear+deck features: -0.09 x1000 logloss, tiny. Multi-set DraftEncoder + META from text wins zero-shot (0.187 vs 0.175 bag+META) but 4 days of target data makes a per-set linear bag as good.
- Synergy: real but small (deck-level pair terms ~ -0.18 x1000 game logloss; opening-hand pairs ~ -0.3 x1000). Card-flow (draw) dominates "seen together" effects.
- Skill confound: user_game_win_rate_bucket is leaky (0.352 Spearman); rank is the only player feature used.

## New result today (MSH only, round 10 script scripts/improve_deck_v2.py)
- Round 4 blend (LightGBM + deck feat + aff, averaged with game FM) on MSH: 0.2882.
- Refit LightGBM on train+val (85% of drafts, using val-chosen tree count scaled by data size), + recency weighting (exp(-age/tau), tau=2 chosen), 5 bagging seeds, then blend with game FM: 0.2934 (+0.0052 ± 0.0016 bootstrap sd). Most of the gain is from refitting on the most recent data. Parameter grid didn't help.
- Implication: format drift / recency matters; any transformer must also be refit on train+val to be compared fairly.

## In-house asset: PickNet (../picknet)
- A transformer predicting human picks from 17Lands drafts. Multi-set model `all_d256` (8.4M params, d=256, 8 heads, 4 history + 3 pack layers), trained 3 epochs over 17 sets (Premier + Traditional), 70.9% top-1 on chronological test (MSH 70.1%; puder v1 69.9%). Cards enter through a CardTable built from features (rules-text embeddings + metadata), so it produces a 256-d card vector for any set, including unseen ones (zero-shot feature-only variant exists; MSH zero-shot top-1 0.508, with target data 0.686).
- Per-set `pool` model (rank-32 bilinear candidate x pool) is already used for the affinity column.
- The user offered these as free starting embeddings, uncertain how helpful. They encode how humans value cards and which cards go together; they know nothing about game results directly.
- Hardware: 10-core Mac, 16GB RAM, MPS. DraftEncoder big run ~1h per set.
