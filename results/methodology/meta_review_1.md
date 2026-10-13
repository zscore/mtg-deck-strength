# Meta-review 1: critique, reframe, and the next 12 hours (2026-10-12)

Reviewer pass over `results/research_log.md`, README (skim), `evaluation_design.md`, `pipeline_audit.md`,
`matchmaking_identification.md`, `next_model_design.md` and `reports/Transformer deck strength models.md`.
No compute was run, apart from reading `MSH/deck_synergy_v4/improve_v2.json` and `ps`. At review time the queue was
running `scripts/improve_deck_v2.py BLB` (started 18:51, about 4.2 GB RSS, 3.5 cores).

**Verdict.**
- The four memos are good. They agree on the main diagnosis: the current headline rewards skill leakage, MSH-only
  wins are noise, and matchmaking compresses scale but not order.
- They disagree on one thing that matters: **which player control goes into the evaluation context**. Settle it
  before any SC-GLL number is quoted.
- The project should switch now. The primary objective becomes a **skill-adjusted, game-level deck rating**, judged
  by SC-GLL with a de-leaked context. Record Spearman is demoted to a secondary metric, computed on reconciled labels.
- The next 12 hours should be spent mostly on measurement (labels, eval tables, pool models, saved predictions),
  not on new architectures. The transformer runs only if the structural likelihood (A) passes its screen.

---

## 1. Critique

### 1.1 Where the memos disagree with each other or with the log

1. **The player control in SC-GLL's context is inconsistent across memos. This is the most important fix.**
   - `evaluation_design.md` §2.1 puts the **raw** `user_game_win_rate_bucket` into the frozen context c(x). It
     mitigates the leak by restricting to users with ≥ 50 games.
   - `matchmaking_identification.md` §5.3 says "never use the raw (leaky) bucket". Its §2.4 shows the bucket is an
     export-time per-set total, correlated 0.91 with the current run's WR at bucket "1".
   - `next_model_design.md` uses a de-leaked b̃ in A5.
   - If the primary metric keeps the raw bucket, part of this run's outcome sits in c(x). Every deck score is then
     judged on the residual after the run's own luck has been partly explained away. That costs power, and it
     penalises scores most where the bucket is closest to the run record.
   - **Resolution:** the context uses b̃, the de-leaked bucket (§2.4 formula, n_u integrated log-uniform over the bin),
     on users with bucket ≥ 50. The raw-bucket SC-GLL is reported once as a sensitivity row and never decides
     anything.
2. **The record-state dummies in the SC-GLL context are the collider that A is designed to remove.**
   - `evaluation_design` conditions c(x) on the 21 record states and accepts the collider bias as "conservative".
   - `next_model_design` §A.2 argues that free record dummies are mechanism 2 of the compression.
   - Under a state-conditioned metric, A's main structural advantage is mostly invisible. Its fixed (a, b)
     recalibration also erases A's scale gain by construction.
   - So SC-GLL can only tell whether A *ranks* decks better, which A's own simulation says it will not.
   - **Resolution:**
     - Judge A on SC-GLL for ranking.
     - Judge it separately on the first-game (0-0) calibration slope and the trophy-rate calibration for scale. That
       is a validity criterion, not a decide() criterion.
     - Report match-1-only SC-GLL with every result, as the collider-free sign check.
3. **PickNet surprise.** The transformer report recommends it as a "non-leaky skill/style proxy". The evaluation
   memo calls it "a drafter feature by design" that raw Spearman would reward. The 12-set meta settled the empirical
   question (+0.00009, p = 0.83). The conceptual point stands: surprise is a player feature. If it is used at all,
   it belongs in the context, never in the deck score. **Closed.**
4. **The champion definition is not settled.**
   - `next_model_design` §0 takes the champion to be "LightGBM refit on train+val + recency + 5 seeds, blended".
   - That recipe has only been measured on MSH. Its 12-set champion number (0.279) is round 4: train-only, one
     seed.
   - Every comparison in the design memo therefore inherits an unreplicated recipe.
   - **Resolution:** the round 10 recipe is replicated across 12 sets in Q2 below, before it becomes everyone's
     baseline.
5. **The multi-build design: the counts quoted are not the count needed.**
   - The log says "19% of drafts have > 1 build (same-player/different-deck design viable)".
   - The audit says 19-20% have > 1 build, and 17.7% have a most-played build other than build 0.
   - Neither count is the one that matters: drafts with **≥ 2 builds that each played games**.
   - Rebuild-after-loss drafts often have one or two games on build 0, so the usable share could be well under 19%.
     The "viable" claim is unverified. Count it in Q0.
6. **The audit's fallback for stale labels has its own selection problem.**
   - The audit suggests "report Spearman on the subset where the two sources agree". That subset over-selects runs
     that finished before the snapshot, which skews short (early 0-3s) or long-complete.
   - The "+0.01 to +0.03" Spearman uplift on it is a guess, and the selection makes it uninterpretable.
   - **Resolution:** use the reconciled label on all drafts (Q2). Never use the agreeing subset as a headline.
7. **The scale anchor s comes from a heterogeneous, mostly out-of-sample trio.**
   - MSH gives −0.102 ± 0.043 and EOE −0.014 ± 0.039, a gap of 1.5σ. ECL is not one of the 12 modelling sets.
   - MSH is again the extreme set, as it has been on every synergy measure.
   - s ≈ 0.55 (CI 0.1-1.0) is barely an estimate. The design memo is right to profile A over {0.4…0.85} and not fix
     a point.
   - Since ranking is s-invariant in simulation, s matters only for the interpretability of the scale. That is
     second-order for the user's goal, so pool s over 12 sets cheaply and do not block on it.

### 1.2 Under-evidenced claims in the log

| claim | why it is weak | status |
|---|---|---|
| Round 10: refit + recency + 5 seeds, MSH 0.2882 → 0.2934 (+0.0052 ± 0.0016) | One set, and the smallest and most outlying one. The ± covers test-draft bootstrap only, not seed noise (about 0.0015 per the audit). The 5-seed average is in the treatment arm only, worth +0.0005 to +0.001 by itself. Refit vs train-only confounds "more and more recent data" with "modelling". α comes from the train-only twin and is probably too high for the more confident refit model. | **Plausibly real (it is a data-quantity effect), but unreplicated.** Replicate on 12 sets with paired seeds (Q2). It is also not comparable to any train-only arm. |
| "Param grid: no gain" | MSH only. Val NLL differences of 1e-5 sit inside the unnormalised-recency confound (audit #8). | Fine as a null. Do not revisit. |
| "Model slope 1.2-1.7; blend explains 25-45%, ceiling 0.45-0.60" | The `variance_explained.py` shares are retired (audit #13: "game FM alone" share = −0.14). The `matchmaking_variance.py` shares depend on s, which is unidentified from records. | **Strike the 25-45% / 0.45-0.60 numbers from the log.** Keep only "β² Var(z) is a lower bound". |
| "82% of games from users with ≥ 50 games" | MSH only. FIN has 9% in 500+, so the per-set share differs. It decides how much power the experienced-only filter costs. | Measure per set in Q0/Q1. |
| ~19% stale labels ("draft-file record is a mid-run snapshot") | Measured on MSH (and FIN). The mechanism is inferred, not shown. Duplicates, joins on the wrong event, or rebuild games were not ruled out. 25-29% go the *other* way (the game file is missing games), so the game file is not uniformly better. | Forensics in Q0 before any relabel. |
| Linkage: 15% of drafts at 80-85% precision | Precision comes from one placebo window on an 8k MSH sample, with the same pattern by eye on EOE/ECL. Links select fast re-queuers (heavier, better players). | Promising. A big bet (§4), not a queue item. |
| Synthetic results: leaky model +0.050 raw Spearman (z = 25); bucket drop 0.248 → 0.327; A changes scale not rank | Each uses hand-set parameters (true sd 0.30, skill sd 0.25, skill-favoured off-value direction r = 0.6). They show **direction and mechanism**, not real magnitudes. | Use them as reasons to test, never as results. |
| PickNet features closed, draft state hurts (Holm p = 0.023, 5 sets significantly worse) | Judged by raw Spearman, the leak-rewarding metric. A null under a metric that *favours* drafter features is a strong null. | Credible. Closed. |

### 1.3 Things that look too good, or too clean

- **"rank + skill bucket explains 99.99% of latent WR variance"** (`variance.json`). This is leakage plus an
  unidentified total, as all three memos agree. Remove it from any writeup.
- **The MSH deck_metrics "ceiling" (0.2445) is below the model's score (0.2882).** The ceiling is retired, but
  `improve_v2.json` and others still emit a `ceiling` field. Delete the field from the scripts so it cannot creep
  back into tables.
- **Rank is an input to the "deck strength" LightGBM, and its predictions are scored with rank in them.**
  - Rank is a player and pool variable, so part of every historical draft-level Spearman (0.271-0.293) is "who is
    this player", not "what is this deck".
  - The evaluation memo's reference-rank deck-only score fixes this for SC-GLL. The historical Spearman series is
    still a record-prediction number, not a deck-rating number, and should be labelled that way.
- **The round 10 improvement is the largest single-step gain since round 3.** It arrived in a round that reported
  about 13 rows on reused test drafts. The prior should be that some of it is selection. Paired 12-set replication
  is the only cure.
- **The test split has been reused for about 10 rounds and is now dev-test.** There are **7 local sets the 12-set
  pipeline has never touched**: the matchmaking memo cites 19 sets, ECL among them. They are a free vault. Do not
  spend them on screening (§4, bet 1).

---

## 2. Reframe: the objective and the metric

**The objective.**
- Produce a **skill-adjusted deck rating**: the effect of the registered 40 on per-game win probability against a
  reference opponent pool, with the player held fixed.
- Card ratings come from the same model, with the deck rating as their sum or readout.
- Record prediction is a downstream use, not the objective.

**Why switch now.** Three findings each break the old headline independently:
1. Compression: the scale of record-trained scores is wrong.
2. Leakage: raw Spearman pays for player features. Rank is in the score, and synthetic leakage gains are 10× real
   gains.
3. Stale labels: the evaluation target itself is wrong for about 1 in 5 drafts.

Game-level metrics use the game files, so they are immune to finding 3 by construction. That alone makes the game
level the right place to judge.

**The primary metric (decides adoption): SC-GLL-dl.**
- Held-out **test-game** log loss ×1000, on games by users with games bucket ≥ 50.
- The frozen per-set context c(x) is fit on train games. Its inputs are:
  - de-leaked bucket b̃ (plus its noise variance as a feature)
  - experience bucket, with the 500/1000 fix
  - **per-game** rank
  - record state before the game
  - play/draw
  - mulligan difference, included only if `game_controls.py` keeps it; freeze that choice once
- The candidate enters only as a + b·z(s(deck)), where s is a deck-only score at a reference rank.
- (a, b) are fit on **val games** for every candidate. Refit models use their train-only twin's calibration.
- The decision is made with `eval_protocol.decide()` on a 12-set random-effects meta, with
  `DecisionRule(min_effect=0.02)` (×1000 nats/game), Holm across the finalists of a round.

**Vetoes (any one blocks adoption).**
- (i) Leakage index: the stratified-Spearman gain is < 40% of the raw-Spearman gain.
- (ii) Match-1-only SC-GLL-dl has the opposite sign, and its 12-set p < 0.1.
- (iii) Within-build b falls (one-sided p < 0.1), once Q0 shows enough multi-build drafts.

**Secondary metrics (always reported, never decisive).**
- Spearman vs the **reconciled** W/(W+L). This is the record-prediction headline, kept for paper continuity.
- Spearman vs the draft-file W/(W+L), reported once for each historical model so that old numbers stay comparable.
- Stratified Spearman.
- Card-rating split-time reliability: correlation of card values fit on the first vs second half of train.
- First-game calibration slope, for the scale.

**Retired.** The deck_metrics ceiling, the variance_explained shares, the raw-bucket SC-GLL as a decision metric,
and single-set "wins".

---

## 3. The ranked experiment queue (about 12 h serial, one heavy job at a time)

General rules:
- Niced, ≤ 4 threads (`OMP_NUM_THREADS=4`, LightGBM `num_threads=4`), MPS for at most one job.
- Every script saves **per-draft and per-game predictions for every seed** to `results/{SET}/eval/` so that
  meta-analysis never needs retraining.
- Every run is logged in `research_log.md`, losers included.

**Before the queue starts.** If the running queue continues `improve_deck_v2.py` on more sets, let BLB finish and
then cancel the rest. Q2 subsumes it, and Q2 saves predictions.

### Q0. Label and context forensics (0.3 h, CPU light, < 1 GB) [blocking]
- **Goal:** decide whether to rebuild labels, and validate the context ingredients.
- **Inputs:** the draft file (last-pick rows: `draft_id`, `event_match_wins/losses`, `draft_time`) and a game-file
  column scan (`draft_id`, `match_number`, `game_number`, `won`, `build_index`, `game_time`, `rank`, both buckets).
  Use MSH, FIN, DSK, BLB.
- **Script:** to write, `scripts/label_forensics.py`. Reuse `prepare_next_model.build_tables` projections.
- **Checks:**
  1. Among drafts with game-file W+L > draft-file W+L:
     - (a) duplicate (draft_id, match_number) rows;
     - (b) whether the extra games come strictly *after* the draft-file record's implied last game in time;
     - (c) whether the game-file record obeys the stopping rule (ends at 7 wins or 3 losses, or is still running at
       the window end);
     - (d) whether the extra games all sit on later builds.
  2. Bucket definition: within a set, at fixed games bucket, the WR bucket should not trend with `draft_time` if it
     is a per-set export total.
  3. Count of drafts with ≥ 2 builds and ≥ 2 games on each, per split.
  4. Share of test games with games bucket ≥ 50, per set.
- **Go/kill.**
  - **Go on relabel** if (a) < 1% and (c) ≥ 95%.
  - **Kill the relabel** (keep the draft-file labels and accept the noise) if the extra games are duplicates or come
    from another event.
  - The **within-build veto is enabled** only if ≥ 1,000 qualifying test drafts per set, or ≥ 5,000 across sets.
  - If the ≥ 50 share is < 60% on any set, use ≥ 10 for that set and note it.
- **Expected value:** high, because it gates Q1-Q3. Cost is trivial.

### Q1. Eval tables and the frozen context per set (1.0 h, CPU, < 1.5 GB per set)
- **Goal:** build, once for all 12 sets:
  - the reconciled W/L per draft: game-file W/L when its history is complete (contiguous match numbers from 1) and
    has ≥ as many games; otherwise the draft-file record;
  - the per-game table for val and test games: draft idx, build idx, y, play, mull diff, per-game rank, record
    state, experience, raw bucket, b̃ and its noise variance, match number, day;
  - the context logits c(x) fit on train games: a logistic model, a few hundred parameters, seconds.
- **Script:** to write, `scripts/build_eval_tables.py`, writing `results/{SET}/eval/tables.npz` with `draft_id`
  saved (audit #14). Optionally fold in `results/methodology/matchmaking_checks.py` on the same scan to pool s over
  12 sets (+20 min, about 650 MB).
- **Go/kill.** These are sanity gates, not decide().
  - The context alone must beat the intercept on test GLL in 12/12 sets.
  - The b̃ context must lose to the raw-bucket context on test GLL. It should, because the raw bucket leaks. If b̃
    is *not* worse, the de-leak is broken.
- **Expected value:** high. This is the infrastructure for every later decision.

### Q2. Relabel round 4 and replicate round 10 on 12 sets, with saved predictions (2.0-2.5 h, CPU 4 threads)
- **Goal:**
  - (a) Measure how much the stale labels hurt training and evaluation.
  - (b) Replicate round 10's refit + recency on 12 sets.
  - (c) Measure the LightGBM seed floor.
- **Inputs:** cached `deck_synergy_v4/draft_inputs.npz`, with game_logit and affinity, so no pool models are needed.
  Also `eval/tables.npz`.
- **Arms** (the round 4 LightGBM feature set, logit-averaged with the cached game FM, α on val):
  - L0 = draft-file labels, train-only
  - L1 = reconciled labels, train-only
  - R1 = reconciled labels, refit train+val, τ = 2 with **normalised** weights (audit #8)
  - each arm × 3 paired seeds
  - also the round 4 averaged blend with the audit #5 fold-standardisation fix (numpy only)
  - about 9 LightGBM fits per set, roughly 1 min each at 4 threads, so about 2 h with data load
  - save natural-rank and reference-rank predictions
- **Script:** to write, `scripts/round4_relabel.py`, forked from `improve_deck_v2.py`. Drop the τ and param grids,
  since both were null on MSH.
- **Go/kill:**
  - The reconciled label becomes the **evaluation** target on validity grounds if Q0 passed. No performance test.
  - **Training** on reconciled labels (L1 vs L0) is adopted if `decide(meta_analysis(...), DecisionRule(min_effect=0.001))`
    passes on Spearman vs the reconciled labels **and** Q3's SC-GLL-dl is not worse: 90% CI lower bound > −0.02.
  - R1 vs L1 is adopted as the standard recipe for all arms under the same rule.
  - If R1 fails, round 10 was MSH noise. Remove "refit + 5 seeds" from the champion definition in
    `next_model_design`.
- **Expected value:** high.
  - The relabel likely moves absolute Spearman by +0.005 to +0.02 (a guess) and model deltas by ≤ 0.002. Mostly it
    fixes the target.
  - The refit replication decides the champion for every later comparison.

### Q3. SC-GLL-dl calibration of existing models (0.5 h, CPU, numpy only)
- **Goal:** the first deck-rating scoreboard. Find how much of the historical progress survives skill control.
- **Inputs:** Q1 tables, Q2 predictions, and the cached game_logit (the deck-only FM score).
- **Models:**
  - game FM alone
  - round 4 LGBM
  - round 4 blend
  - round 4 blend + fold fix
  - L1, R1
  - the blend with α re-chosen on **val SC-GLL**, not record NLL (cross-fit on val halves)
- **Metrics:** SC-GLL-dl (primary), raw-bucket SC-GLL, match-1 SC-GLL, raw and stratified Spearman, leakage index,
  day-block bootstrap.
- **Script:** to write, `scripts/score_scgll.py`, wrapping `eval_protocol.ctx_calibrated_game_loss`,
  `paired_bootstrap`, `meta_analysis` and `decide`.
- **Go/kill:** this is a calibration run; decide() is reported, not acted on. One exception: if the game FM alone
  beats the blend on SC-GLL-dl (decide passes), **the deck-rating product is the game FM**, and LightGBM is
  demoted to a record-prediction head. My guess is that this happens, because the LightGBM gains are record gains
  and rank is in the score.
- **Expected value:** very high per hour. It reframes the whole program on existing artifacts.

### Q4. Retrain the PickNet pool_d32 models and restore affinity (1.0-1.5 h, MPS)
- **Goal:** unblock A, B, `game_controls.py`, FM refits and the within-build scoring. All of them need per-build
  affinity.
- **Script:** existing, `picknet/scripts/train_picknet.py <SET> pool d=32`, with the hyperparameters from
  `pool_d32.json`. Save to `results/{SET}.PremierDraft/picknet/models/pool_d32.pt`. Also make `pick_affinity` raise
  when the file is missing.
- **Go/kill:** recompute affinity for MSH and FDN train drafts and compare with the backed-up npz column.
  - r ≥ 0.95: go.
  - r < 0.90: kill, and drop affinity from **all** arms of Q5-Q6. Its value is about +0.005 raw Spearman, partly
    leaky (README).
- **Expected value:** medium on its own, but on the critical path. Check one set's wall time first; if it exceeds
  10 min per set, run only MSH, FDN and BLB now.

### Q5. Structural likelihood A, screened on MSH, FDN and BLB (2.5 h, CPU 4 threads)
- **Goal:** find whether A (record variant and rated variant) is non-inferior or better as a deck rating, and
  whether it is fit to serve as B's likelihood.
- **Inputs:** `drafts/prepare_next_model.py` tables, Q4 affinity, Q1 tables.
- **Trimmed grid,** 6 fits per set (out of 11):
  - A0: v4 reproduction, raw bucket
  - A1: iid, no record, no bucket
  - A2: RE + free record dummies
  - A4 at s = 0.55 and at s = 0.85
  - A5: best A4 + b̃
  Drop A3 (it overshoots on the ridge) and the λ / match = wins rows until A passes.
- **Script:** to write, `scripts/mm_screen.py`, a thin driver over `drafts/mm_likelihood.py`. The first real-data
  run is a 5-min smoke test on MSH with RSS logging. **Abort if RSS > 3 GB.**
- **Go/kill:**
  - **Kill (bug):** A0 is more than 1e-4/game from `deck_synergy_v4` val loss.
  - **Kill (misspecification):** SC-GLL-dl differs by > 0.02 ×1000 between s = 0.55 and 0.85.
  - **Go to 12 sets and to B:**
    `decide(meta, DecisionRule(alpha=0.2, min_positive=3, min_effect=-0.02), p_override=FE p)` passes for A4 or A5
    vs A0 on **val** SC-GLL-dl, i.e. non-inferiority on 3/3 sets. In addition, A4's first-game calibration slope must
    be within [0.9, 1.15].
  - A5 > A4 by ≥ +0.02 marks it as the **deck-rating champion candidate** for the 12-set run.
- **Expected value:**
  - Medium for ranking: the simulation predicts ±0.002.
  - High for the user's goal: calibrated, decompressed card ratings, plus a cleaner player control than the raw
    bucket.

### Q6a (if Q5 passes). Transformer B1, overnight (4-5 h, MPS, ≤ 3 GB RSS)
- **Goal:** two answers. Does a PickNet-initialised game-level set transformer add to the champion? And does
  distillation help?
- **Configs:** `game_set_transformer.py` on the A likelihood at the Q5-best s, d = 128, L = 3, on MSH/FDN/BLB:
  - B1.1: random init, α = 0
  - B1.2: PickNet init, α = 0
  - B1.3: PickNet init, α = best of {0.1, 0.3}
  - B1.5: PickNet init, iid likelihood (likelihood ablation)
  - 1 seed each, plus a second seed of the best
- **Script:** existing draft. Needs a driver that writes per-draft and per-game predictions.
- **Go/kill:**
  - **Kill:** B alone is > 0.01 Spearman, or > 0.2 ×1000 SC-GLL-dl, below the round 4 FM on val (a bug); or
    student-teacher r > 0.97; or the seed sd exceeds half the gain.
  - **Go to B2/B3 (a later night):** the best B1 in the blend gains ≥ +0.02 ×1000 SC-GLL-dl on the 3-set val mean,
    or B1.2 − B1.1 ≥ +0.02.
- **Expected value:** low to medium (+0.000 to +0.004 by the memos' own estimates). It runs overnight only because
  the GPU is otherwise idle.

### Q6b (if Q5 fails, or after Q6a). Game FM refit with the audit fixes (3-4 h, CPU)
- **Goal:** the cheapest real deck-rating upgrade.
- **Changes:** refit the round 4 game FM on 12 sets with:
  - the experience 500/1000 fix
  - opp_rank columns dropped
  - **b̃ in place of the raw bucket**
  - per-game rank as an offset
  - calibration on val, not train (audit #6)
  Arms: raw bucket vs b̃ vs no bucket, which is `game_controls.py`'s "- skill" variant extended.
- **Script:** existing `game_controls.py` / `deck_synergy_v4.py`, with a small edit for the b̃ column from Q1.
- **Go/kill:** decide() on test SC-GLL-dl across 12 sets, Holm across the 2 challengers, plus the vetoes.
- **Expected value:** medium to high for the deck-rating product. It decides the player-control question on real
  data rather than in simulation.

**Schedule.**
- Q0 0.3 + Q1 1.0 + Q2 2.2 + Q3 0.5 + Q4 1.2 + Q5 2.5 ≈ 7.7 h, then Q6a or Q6b for the remaining 4-5 h.
- If anything slips, cut Q6 first. Never cut Q0-Q3.
- **Not in the queue, and why:**
  - More LightGBM tuning: the grid was null.
  - Any PickNet feature variant: closed.
  - The full 11-row A grid: wait until A passes.
  - B2/B3: 20-35 GPU-h, gated on B1.

---

## 4. Big bets beyond the queue

1. **The shuffle is a randomised experiment. Use it for causal card ratings.**
   - Within a fixed deck and player, which cards get drawn is random.
   - A drawn-vs-not-drawn model inside the game likelihood (deck and player fixed effects absorbed by the draft
     random effect) identifies each card's marginal contribution **free of skill leakage**. This is a properly
     controlled IWD.
   - The project found GIH WR beats IWD as a predictor of deck strength (19/19 sets). That is exactly what you
     expect if GIH WR carries player signal. It is not evidence that IWD is the wrong estimand.
   - Use the shuffle-identified effects as a **prior or regulariser for card power w** in A or the FM, and as a
     validation metric for card ratings: does a model's card ranking agree with the randomised estimates?
   - Data: the drawn and opening-hand columns are already used in rounds 8-9.
   - Highest expected value for "skill-adjusted card ratings".
2. **A pseudo-user panel from time-linking, for within-user validation and a player random effect.**
   - About 15% of drafts link at about 80% precision.
   - A user random effect shared across linked runs splits player from deck directly, and β_within / β_cross says
     how much of any deck model is the player.
   - This is the only way to validate deck ratings against a held-fixed player at scale; the within-build design
     is probably small.
   - Correct for link precision with the placebo window, and reweight for fast re-queuers.
3. **Freeze the pipeline and confirm on the 7 untouched sets, then pool to 19.**
   - The current test split is spent.
   - Build `draft_inputs` and eval tables for the 7 extra sets (ECL and the others), freeze the champion recipe
     after Q2-Q6, and evaluate once.
   - Afterwards, 19-set meta-analysis lowers the MDE by about √(12/19), so +0.001 Spearman or 0.03 ×1000 SC-GLL
     become detectable.
   - It also gives the 19-set s anchor (±0.1).
4. **Sealed as a skill-free test of deck ratings.**
   - Sealed pools are random, so deck quality there is pack luck plus build skill, with no drafting skill.
   - Score Sealed decks with the Draft-trained card ratings (same set) and measure SC-GLL on Sealed games.
   - A model that wins in Draft but not in Sealed is learning the drafter.
   - The Sealed game files are local and no new collection is needed.
5. **Bo3 game data as the second scale anchor** (lower priority). If 17Lands publishes TradDraft game files,
   within-match covariance identifies the pool spread directly and validates s. This only matters for calibrated
   "win probability vs average opponent" ratings, not for order. Check availability first, and do not build
   anything until the data exists.

---

## 5. Log housekeeping (no compute)

- Strike from the log: the 25-45% / 0.45-0.60 variance shares, every `ceiling` number, and "rank + bucket 99.99%".
- Mark round 10 "+0.0052" as **MSH-only, single seed baseline, unreplicated**.
- Note that the champion number (0.279) is a record-prediction metric that includes rank.
- Add a running denominator: the number of variants scored on test per round.
