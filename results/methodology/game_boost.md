# Does LightGBM's non-additive count signal exist at the game level?

**Question.** fm_cardvalue.md found that blend*'s gain over the champion game FM comes from LightGBM's
**non-additive** use of deck counts: its best linear-in-counts projection gains only +0.004. Is that structure visible
to a game-level likelihood, so that a game model could learn it? Or does it only show up when a model is trained on
draft records?

**Answer: neither game-level design recovers it, and neither does the draft-level boost from the champion.** On the
3-set gate (MSH, FDN, BLB; SC-GLL experienced-dl, ×1000 nats/game):

- blend* gains **+0.175** over the champion (3/3 sets).
- A (game-level LightGBM residual on the champion's game logit) gains **+0.021** (3/3, n.s.).
- The per-copy / land-interaction FM probe gains **+0.013** (2/3).
- B (draft-record LightGBM boosted from the champion) gains **+0.011** (1/3), with match 1 at −0.25.

All three arms stay 0.14–0.17 below blend* in every set. The gate (any arm ≥ +0.05) **failed**, so the remaining 9
sets were not run.

So the extra signal is **not game-level deck structure**:

- The game likelihood, given full freedom over counts, finds almost nothing beyond the champion. A's early stopping
  chose 1–142 trees, and on FDN it chose 1 tree.
- The probe's less-shrunk settings are strongly harmful (−0.26 at l2_u 1e-5). Per-copy and land-count freedom only
  overfits at the game level.

The signal is also **not simply "trained on draft records"**. B uses the draft records, the same counts and the same
LightGBM settings, and still fails (mixed signs, negative on match 1). Only blend* gets it. blend* is a LightGBM
trained on draft records **independently of the FM** (counts + structure + rank + deck features + aff), then mixed
with the FM at a weight α tuned on val SC-GLL-dl.

The defensible reading is that the gain is tied to draft-record training of a model that is *not* anchored on the
game FM, combined with SC-GLL-tuned shrinkage of that model toward the FM. It is not a deck pattern that game outcomes
reveal. One open alternative remains for B: its offset is in-sample on train drafts (see the caveats).

## Setup

- **Script.** `scripts/game_boost.py`, in two stages per set, run in separate processes: `fm` (torch, no LightGBM) and
  `boost` (torch-free shim plus LightGBM and the scoring).
- **Champion.** The game_fm_v2 recipe: fm4 nonland + deck features + aff, champion controls, the grid picked on val
  game log loss, seeds 0 and 1. It is refit here: seed 0 reproduces game_fm_v2's saved champion to 5e-8 on MSH. The
  per-game full logit (deck score + context) and the per-draft deck-only score are saved to `game_boost/fm.npz`.
- **A (game-level residual).** LightGBM with a binary objective on **train games**.
  - Features: the game's own build deck counts only.
  - `init_score`: the champion's full game logit, so the context stays in the offset.
  - Early stopping: on val games (patience 100).
  - Deck-only score: champion deck score + raw tree output on the main build. This is context-free by construction.
  - One fit per champion seed.
- **B (draft-level residual, "boosted from the FM").**
  - Rows: each train draft's W/L as two weighted rows (as in improve_deck), with features = main-build counts.
  - `init_score`: r[rank] + β·s_champion, from a binomial logistic fit on train drafts (β ≈ 1.04–1.13).
  - Early stopping: on val drafts.
  - Deck-only score: β·s_champion + tree output.
- **C.** blend* rebuilt from the saved sc_gll LightGBM, T1's α* per seed and this run's champion. Against the saved T1
  blend* it gives Δ = 0.0000 and r = 1.000, so the reference is like for like.
- **Probe (torch).** The champion recipe, trained jointly from scratch, plus non-additive count terms:
  - For every nonland card, a 2nd-copy indicator 1[x ≥ 2] and a 3rd+-copy term (x − 2)₊. So copies 1, 2 and 3+ get
    separate values w, w + d2 and w + d3.
  - A 4 × 4 one-hot of land count (≤15, 16, 17, 18+) × nonland count (≤21, 22, 23, 24+).
  - All extra terms have L2 penalty l2_u ∈ {1e-5, 1e-4, 1e-3}, plus 1e-2. The 1e-2 point is an extension added after
    MSH, where val log loss was still improving at the 1e-3 edge.
  - The pick is made on val game log loss, and it picked 1e-2 on all three sets. The pick has 2 seeds.
- **LightGBM settings.** improve_deck.LGB_PARAMS, 4 threads.
- **Scoring.** The t_common contexts dl and pre, with subsets experienced (primary) and match 1. (a, b) are fit on val
  and scored on test games. Paired day-block bootstraps (300), with seeds as a variance component, and an RE meta
  (DL + HKSJ, eval_protocol).

## Results (MSH, FDN, BLB; ×1000 nats/game; RE mean [95% CI], sets better)

| arm − champion | experienced dl (primary) | experienced pre | match 1 dl |
|---|---|---|---|
| **blend\* (saved T1, reference)** | **+0.175 [−0.007, +0.358], 3/3** | +0.210 [+0.057, +0.362], 3/3 | +0.170, 3/3 |
| C blend\* rebuilt | identical (Δ vs saved 0.0000) | identical | identical |
| A game-level residual | +0.021 [−0.058, +0.100], 3/3 | +0.017, 3/3 | +0.018, 3/3 |
| B draft-level residual | +0.011 [−0.263, +0.285], 1/3 | +0.098 [−0.116, +0.312], 3/3 | **−0.249**, 1/3 |
| probe [pick] (l2_u 1e-2) | +0.013 [−0.059, +0.085], 2/3 | +0.014, 2/3 | −0.005, 2/3 |
| probe 1e-3 / 1e-4 / 1e-5 | −0.025 / −0.167 / −0.263 (0/3) | −0.012 / −0.155 / −0.254 | −0.109 / −0.281 / −0.359 |

Against blend\* on experienced-dl, every arm loses in every set: A −0.136, B −0.173, probe −0.147 (0/3 each).

Per set, experienced-dl:

| set | blend\* | A | B | probe [pick] |
|---|---|---|---|---|
| MSH | +0.263 | +0.052 | +0.116 | +0.062 |
| FDN | +0.130 | +0.000 | −0.041 | −0.029 |
| BLB | +0.138 | +0.036 | −0.082 | +0.007 |

### Diagnostics

| | MSH | FDN | BLB |
|---|---|---|---|
| A trees (seed 0 / 1) | 103 / 142 | 1 / 1 | 11 / 99 |
| A val game log loss: champion → boosted | 0.66011 → 0.66002 | 0.66070 → 0.66071 | 0.65730 → 0.65729 |
| A residual sd (champion deck-score sd ≈ 0.24–0.27) | 0.021–0.024 | 0.001 | 0.004–0.016 |
| B trees | 486 / 475 | 603 / 433 | 596 / 423 |
| B residual sd (β·champion sd ≈ 0.27) | 0.094–0.097 | 0.065–0.074 | 0.066–0.075 |
| probe val game log loss, l2_u 1e-5 → 1e-2 (champion) | 0.66122 → 0.66040 (0.66011) | 0.66088 → 0.66050 (0.66070) | 0.65759 → 0.65729 (0.65730) |

Test-draft correlations, seed 0, mean over the 3 sets:

| score | r with champion | r with blend\* | r with LightGBM |
|---|---|---|---|
| A | 0.999 | — | 0.90 |
| B | 0.96 | — | 0.95 |
| probe | 0.997 | — | 0.91 |
| blend\* (vs B) | — | B: 0.977 | — |

- **A.** The game likelihood gives the boosted count residual almost no room. Val game log loss moves by ≤ 1e-4. On
  FDN early stopping keeps a single tree, and on MSH, where it does grow trees, the gain is +0.05. The non-additive
  count structure that helps blend\* is not something that lowers game log loss.
- **Probe.** Per-copy and land-interaction terms raise val game log loss unless they are shrunk to near zero
  (d2 sd 0.003 at the pick). Their SC-GLL tracks that: the less shrinkage, the worse. Copy-count thresholds and
  land × spell-count shapes are not the missing signal, at least not at the game level.
- **B.** Here the draft-record residual is large: sd ≈ 0.07–0.10, i.e. 25–35% of the champion's sd, with r 0.95 to
  LightGBM. It is the same kind of correction LightGBM makes. But it helps only on MSH, and it is badly negative on
  match 1 (BLB −0.48). Its experienced-pre gain (+0.10) is larger than its dl gain (+0.01). That pattern is
  record-side or own-mulligan-side signal, not deck signal that transfers. Round 4's "boosted from the FM ≈ averaging"
  (old FM) is now "boosted from the champion < champion alone on 2/3 sets".

## Interpretation

| hypothesis | predicts | observed |
|---|---|---|
| game-level deck structure the FM misses | A or the probe recovers much of blend\*'s gain | A +0.021, probe +0.013 vs +0.175. **Rejected** |
| any model trained on draft records gets it | B ≈ blend\* | B +0.011 (−0.25 on match 1). **Rejected** in this form |
| an independent draft-record model, shrunk toward the FM by an SC-GLL-tuned α | only blend\* gets it | consistent |

What separates blend\* from B:

1. **LightGBM is fit independently.** It is not anchored on the FM, and it sees structure, rank, deck features and aff
   as well as counts. The counts-only version still recovers 77%.
2. **The mixing weight is tuned on val SC-GLL-dl,** the judge's own target. That puts (1 − α) ≈ 0.3 on LightGBM.
   B's weight is set implicitly by boosting on the draft record.
3. **The champion's train-draft scores are in-sample.** B's offset on train drafts comes from a model trained on those
   drafts' games, so B's residual partly learns to undo in-sample overfit. In blend\*, LightGBM never sees the FM.

## Caveats

- **Gate only.** 3 sets, 2 seeds per arm. The gate failed, so per the plan nothing was run on the other 9 dev sets and
  there is no vault test. The 3-set CIs are wide (HKSJ, 2 df).
- **B's offset is in-sample** on train drafts (caveat 3 above). A cross-fitted champion offset (2-fold over train
  drafts, two more FM fits per fold) is the clean version of B. It is the one cheap follow-up that could still move the
  answer toward "draft-record training".
- **B is not α-tuned.** It uses its own implicit weight. An "α-tuned B" (blend the B residual with the champion at α
  chosen on val SC-GLL-dl, numpy only) would separate "B's signal" from "B's weighting". Not run, because the gate
  failed.
- **The probe is one specific encoding** (per-copy thresholds, 4 × 4 land / nonland buckets). Higher-order card
  interactions beyond the rank-4 FM are covered only by A, whose trees can split on any counts.

## Files and runtime

- Per set: `results/<SET>.PremierDraft/game_boost/{fm.npz, fm.json, results.json, scores.npz}` (MSH, FDN, BLB).
- Meta: `results/meta/game_boost.json`.
- Serial, nice 10, 4 threads, torch and LightGBM in separate processes.

| step | wall time | peak RSS |
|---|---|---|
| fm stage (champion 3 fits + probe 5 fits) | 114 s / 156 s / 213 s (MSH / FDN / BLB) | 4.8 / 6.3 / 6.1 GB |
| boost stage (A 2 fits, B 2 fits, scoring) | 28 s / 44 s / 53 s | 3.0 / 4.9 / 5.0 GB |
| meta | < 5 s | — |

About 12 minutes of compute in total; with an earlier MSH fm run before the 1e-2 extension, about 35 minutes of wall
time. FDN's fm stage peaked at 6.26 GB, slightly over the 6 GB target. That is loading the game matrix and is the same
as in fm_cardvalue. The vault was not touched.
