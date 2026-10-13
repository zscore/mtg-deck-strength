# The blend* puzzle: deck signal or player signal?

**Setup.** blend* = (1 − α) z_LightGBM + α z_FM, with α tuned on val SC-GLL-dl. It beats the champion game FM by
+0.13 on the 12 dev sets and +0.23 on the 6 vault sets (×1000 nats/game, SC-GLL experienced-dl). The game-level
transformer found no non-FM signal. So what is LightGBM adding?

**Answer: it is mostly deck signal, but at the level of individual card values, not structure or curve.** LightGBM's
departure from the FM is a linear re-valuation of individual cards, learned from the draft record. It holds up for
players whose skill is measured precisely, and pure player-association scores can't reproduce it. The within-build
referee doesn't veto it, but that test is too weak to confirm a gain either. One caveat stays open: the re-valued
cards lean toward cards that skilled drafters play.

Scripts: `scripts/t3b_blend.py`, `scripts/lgbm_attrib.py`, plus two follow-ups added to settle the question:
`scripts/blend_strata.py` (blend* gain by user_n_games stratum) and `scripts/blend_probe.py` (simple card-sum partners
in place of LightGBM). Meta files are in `results/meta/{t3b_blend,lgbm_attrib,blend_strata,blend_probe}.json`, and
per-set results in `results/<SET>.PremierDraft/{t3b_blend,lgbm_attrib,blend_probe}/results.json`. All 12 dev sets
(MSH EOE SOS TDM FDN DFT LCI FIN MKM BLB DSK OTJ) were run serially with nice 10, 4 threads, and no torch.

## 1. Within-build referee (T3b)

This is the same conditional logit as T3: draft fixed effects are conditioned out, and blend* is scored per build
using each set's saved α* per seed. LightGBM uses T3's per-build platinum scores (3 seeds). The per-build blend* at
the main build reproduces T1's blend* with r ≥ 0.9998. Units are b per within-draft sd, pooled with an RE meta
(DL + HKSJ) over 12 sets.

| pair | state control (primary) | match control |
|---|---|---|
| blend* − champion | +0.0015 [−0.0019, +0.0048], 7/12 | −0.0003 [−0.0028, +0.0022], 5/12 |
| blend* − LightGBM | −0.0001 [−0.0065, +0.0063], 6/12 | +0.0012 [−0.0033, +0.0057], 9/12 |
| LightGBM − champion | +0.0019 [−0.0078, +0.0116], 7/12 (T3: −0.002) | −0.0017 [−0.0087, +0.0053] |

- **Leakage veto:** one-sided p that blend* lowers b = 0.825 (state) and 0.400 (match). **Not vetoed.**
- **Power:** the pooled b of the champion itself is only +0.0096 ± 0.0058 (state) and +0.0019 ± 0.0042 (match).
  Builds within a draft differ little: the within/overall sd ratio is about 0.12, and within-draft build scores
  correlate 0.97 between blend* and the champion.
- **What the test can and can't say:** it rules out blend* cutting within-build b by more than about 0.002, which is
  about 20% of the champion's b. It cannot detect an improvement.
- **Verdict:** blend* is "not down", not "up". This test does not decide the question.

## 2. What d = z_LGBM − z_FM is made of (test main builds, out-of-fold ridge R²; mean over 12 sets)

The sd of d is about 0.46 in train-z units. The blend puts (1 − α*) ≈ 0.3 on it.

| group | R² alone | partial R² (drop from full) |
|---|---|---|
| card counts | 0.537 | **+0.342** |
| structure (lands, curve, creatures, colours, splash) | 0.083 | +0.001 |
| 17 Scryfall deck features | 0.063 | +0.017 |
| affinity | 0.012 | +0.003 |
| rank | 0.009 | −0.000 |
| player (de-leaked skill, n_games bucket) | 0.024 | +0.000 |
| pool (on-colour depth, rares in pool/deck, best-23 power, build gap) | 0.103 | +0.001 |

Combined groups:

- all groups: R² = 0.569
- deck side (counts, structure, deck features, affinity): 0.568
- player side (rank, player, pool): 0.132
- **player side beyond deck side: +0.001**
- deck side beyond player side: +0.437

Correlations of d:

- **Single features:** d is lower for decks with more rares/mythics (r −0.21), higher best-23 pool power (−0.21) or
  higher deck nonland FM power (−0.18). It is higher for higher de-leaked skill (+0.13).
- **Skill, pooled within user_n_games strata:**

  | score | de-leaked skill | raw skill |
  |---|---|---|
  | d | +0.124 | +0.119 |
  | z_LGBM | +0.174 | +0.221 |
  | z_FM | +0.119 | +0.168 |

  So d is about as skill-correlated as the FM score itself. Better players build better decks under either score.
- **Card level** (ridge coefficients c_j of d on card counts; standardised regression of c on clean hand value (HV),
  FM w and deck_skill, mean over 12 sets):

  | predictor | β |
  |---|---|
  | FM w | −0.86 ± 0.02 |
  | deck_skill | +0.53 ± 0.02 |
  | HV | +0.02 ± 0.02 |

  In words, LightGBM compresses the FM's card powers and up-weights cards that skilled drafters play, beyond those
  cards' hand values. This is the one sign of a player proxy.

## 3. Locating the gain: LightGBM refit on feature subsets (seed 0), blended as blend*

SC-GLL experienced-dl, ×1000 nats/game:

| partner | blend − champion | blend − blend[full] | α* |
|---|---|---|---|
| full (round 4 recipe, platinum) | +0.130 [+0.097, +0.164], 11/12 | — | 0.55–0.80 |
| − rank | +0.132 [+0.086, +0.177], 11/12 | +0.005 (n.s.) | 0.55–0.75 |
| counts only | **+0.100 [+0.063, +0.138], 12/12** | −0.025 [−0.053, +0.002] | 0.65–0.85 |
| summaries (structure + deck feats + aff) | +0.008 ± 0.008 (unweighted; α* = 1 in 2 sets) | −0.114 | 0.80–1.0 |
| structure only | −0.004 ± 0.008 (unweighted; α* = 1 in 5 sets) | −0.129 | 0.90–1.0 |

**About 77% of the gain comes from card counts alone. Structure, curve and deck summaries add nothing, and rank is
irrelevant.** Match-1 dl gives the same pattern: full +0.165, counts +0.114, structure −0.036.

## 4. Two follow-up tests on player signal

**(a) Gain by experience stratum** (T1's saved losses, "all" subset, dl context, pooled over 12 sets). If blend* were
proxying skill that the de-leaked bucket measures badly, its gain should vanish for players with 500 or more games,
whose bucket is precise.

| user_n_games | blend* − champion (dl) | sets better | (raw context) |
|---|---|---|---|
| 1–10 | +0.221 ± 0.040 | 11/12 | −0.123 |
| 50 | +0.136 ± 0.030 | 9/12 | −0.042 |
| 100 | +0.117 ± 0.015 | 11/12 | +0.034 |
| ≥ 500 | **+0.178 ± 0.041** | 9/12 | **+0.190** |

- The gain does not shrink where skill is measured precisely.
- At ≥ 500 games the raw and de-leaked judges agree (+0.19 vs +0.18), because the raw bucket hardly leaks there.
- This explains T1's raw-context drop to +0.037: it comes from low-experience players, whose raw bucket is mostly made
  of this draft's own games. That is a judge artefact, not evidence of a player proxy.

**(b) Simple card-sum partners** in place of LightGBM, blended the same way:

| partner | blend − champion (exp-dl) | α* |
|---|---|---|
| skill-marker score (sum over the deck's cards of the mean raw skill of train decks playing each card) | +0.003 ± 0.002 | 0.95–1.0 |
| the same with the de-leaked bucket | −0.001 ± 0.005 | 0.90–1.0 |
| − FM power (compressing the FM's card w) | +0.004 ± 0.005 | 0.90–1.0 |
| HV sum | +0.003 ± 0.004 | 0.90–1.0 |
| LightGBM (reference) | +0.130 | 0.55–0.80 |

- SC-GLL-dl does not reward pure player association.
- The two linear directions found in d (skill markers, FM-power compression) reproduce none of the gain.
- LightGBM's gain therefore comes from card-specific values it learned from the W/L record, not from a skill proxy.

## Verdict

| test | deck signal predicts | player signal predicts | observed |
|---|---|---|---|
| within-build b (T3b) | up / not down | down or flat | not down (veto p 0.83); underpowered |
| attribution of d | structure/curve or cards | skill / pool features | cards (partial R² 0.34); player and pool beyond deck +0.001 |
| feature-subset refits | counts/structure carry the gain | — | counts +0.10 of +0.13; structure 0 |
| ≥ 500-game players | gain persists | gain vanishes | +0.18 (dl), +0.19 (raw) |
| skill-marker partner | no gain | gain | 0 (α* ≈ 1) |

**Deck signal, at the card-value level.** LightGBM, trained on draft records, re-values individual cards relative to
the game FM's shrunk card power w. It compresses the FM's strong and rare cards and promotes others, and the game FM
misses this. The gain is not curve or structure (the improve_deck features carry zero), not rank, and not
pool or player information.

**Open caveat.** The cards LightGBM promotes correlate with skilled drafters' choices (β +0.53 on deck_skill, where HV
scores ≈ 0). The within-build referee can't positively confirm the gain. Still, a pure skill-marker score earns
nothing, and the gain survives for ≥ 500-game players.

**Next step.** Move the FM's card-level values toward LightGBM's. For example, fit FM w with a draft-record auxiliary
loss, or use less ridge on w, or distill LightGBM's linear card coefficients into a w prior. Then check that the FM
alone recovers blend*'s +0.13.

## Runtime

Machine idle, serial, nice 10, 4 threads:

| script | sets | total time | peak RSS |
|---|---|---|---|
| t3b_blend | 12 | 6.3 min | 4.4 GB |
| lgbm_attrib | 12 | 20.6 min (5 LightGBM fits per set) | 4.5 GB |
| blend_probe | 12 | 2.4 min | 5.2 GB |
| blend_strata | 12 | under 10 s | — |
| metas | — | under 1 min | — |

Total wall time is about 31 minutes.
