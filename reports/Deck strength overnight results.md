# Deck strength: overnight results (2026-10-12 → 13)

Full trail: `results/research_log.md`. Methodology memos: `results/methodology/`. Frozen vault pipeline:
`results/vault/frozen_pipeline.json` (sha256 87155c50…).

## What changed in how we judge models

- **Spearman against W/(W+L) is no longer the deciding metric.** It rewards skill leakage: dropping the skill
  control "gains" +0.010 Spearman but loses −0.37 ×1000 on held-out games.
- **The primary metric is SC-GLL-dl**: skill-controlled held-out game log loss on experienced users. It takes a
  deck-only score at a reference rank, uses a frozen context model with a de-leaked skill bucket, and decides over
  12 sets by random-effects meta-analysis.
- **Changes are adopted only through a pre-registered rule**, then confirmed on 6 untouched vault sets.
- **The records are compressed by matchmaking.** Arena pairs players by record, and rank indexes the opponent
  pool. A coin-flip play/draw instrument puts the population strength spread at about **0.36 logit** (17 sets).
- **Data quirks found:**
  - `opp_rank` is always None.
  - `opp_colors` is post-treatment (the colours revealed during a game).
  - The draft-file W/L is stale for 16–24% of drafts.
  - Five sets' game files miss games on about 28% of runs.

## Best models now

| goal | model | evidence |
|---|---|---|
| **Deck / card strength** | **blend\***: game FM (rank-4 nonland pairs + deck features + affinity, skill-adjusted) averaged in logit space with LightGBM on deck counts at a platinum reference rank, α tuned on val SC-GLL-dl | +0.13 over the FM on dev (11/12 sets); **vault +0.233, 6/6, p = 0.0001** |
| Record prediction | round 10: LightGBM refit on train+val, recency τ = 2, 5 seeds, + game FM | **vault +0.0040 Spearman, 6/6**; mean 0.295 on game-file labels |

Why blend\* works: LightGBM's added value comes from the **card counts** (counts-only LightGBM recovers 77% of it),
not from structure, curve, rank or player features. It holds for 500+-game users. The within-build test doesn't veto
it, but it isn't powerful enough to confirm a gain. It is **non-additive** use of the counts: an FM given LightGBM's
per-card values gains only +0.034, and the best linear projection of LightGBM gains +0.004. No single FM variant
recovers it (best: a draft-record auxiliary loss, +0.036, about 27%). **Ship blend\*.**

## Confirmed on the vault (6 untouched sets)

- **Game FM vs LightGBM alone:** +0.264, 5/6.
- **FM synergy pairs vs linear:** +0.218, 5/6. This passes the frozen rule, but its HKSJ CI crosses 0.
- **Refit on train+val:** vault +0.005 (n.s.); pooled with dev, +0.027 (p = 0.02). Small.

## Tried and closed

- PickNet features in LightGBM: card vectors, pick surprise and draft state. The draft state hurts.
- Transformers:
  - The pool and deck transformers tie linear models.
  - A game-level set transformer gives +0.000 in every config, including with the FM frozen. There is no learnable
    non-FM deck signal at this data size.
- Model A, the structural matchmaking likelihood, fixes the scale by about +15% but doesn't change the ranking.
- Opponent-colour context (it is post-treatment), recency weighting, the LightGBM parameter grid, a de-leaked bucket
  instead of the raw one, and using hand values as a prior.

## New products

- **Hand values:** randomized opening-hand card effects, `scripts/hand_value_v2.py`.
  - They replicate train→test at r 0.78 with low skill tilt, against GIH WR's 0.62 / 0.68.
  - As a summed deck score they only tie GIH WR.
  - A card drawn later is worth about 30–50% of having it in the opening 7.
- **The evaluation stack:** `eval_protocol.py`, `sc_gll.py`, `t1_rescore.py`, and the within-build referee
  `t3_within_build.py`.

## Open

- Why blend\* works is partly open. A non-additive count encoding in the FM (+0.013), a game-level LightGBM residual
  (+0.021) and a draft-record residual boosted from the FM (+0.011) all fail (blend\* is +0.175 on the same 3 sets).
  Only an independently trained draft-record LightGBM, mixed at a weight tuned on SC-GLL, gets the gain. Untested: a
  cross-fitted offset for the boosted version.
- Dropping only the **own** mulligan control: +0.050 on dev ("pre" context), not tested on the vault.
- The promoted cards lean toward what skilled drafters play. That needs a stronger within-player test (linking runs
  by time gives about 15% of drafts).
