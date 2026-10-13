# Evaluation design for deck-strength models

Written 2026-10-12 without compute. I read the code and the existing result JSONs. No model or project data was
loaded. Companion files in this folder:

- `eval_protocol.py`: a drop-in module with the paired bootstrap, seeds as a variance component, the meta-analysis,
  the decision rule and Holm's correction. Its synthetic self-test runs in about 4 s.
- `power_sim.py`: a synthetic power and leakage study. It runs in about 5 s.

**Bottom line.**
- The current comparison is "Spearman(p, W/(W+L)) on the last 15% of drafts, one set, one seed, bootstrap over test
  drafts". It cannot tell a +0.002 deck-strength gain from noise, skill leakage or luck in choosing variants.
- With 12 sets and a random-effects meta-analysis, the smallest gain you can reliably detect (80% power) is
  **about +0.0011 to +0.0017 Spearman**. In skill-controlled game log loss it is **about 0.03 to 0.05 ×1000
  nats/game**. Recent gains (+0.001 to +0.005) are at or just above that line, so they need the full 12-set
  protocol.
- Make the decision metric a **skill-controlled, held-out game log loss on a deck-only score**. Keep raw Spearman as
  a reported secondary metric, not the decision metric.

---

## 1. Critique of the current metric and protocol

### 1.1 The metric measures "predict this player's record", not "rate this deck"

1. **Skill leakage is rewarded.** The draft-level LightGBM learns from records with `rank` as its only player
   feature. Any deck trait that marks a strong drafter therefore raises its score, whether or not the trait makes
   the deck stronger. Examples are coherent two-colour builds, 16 lands, high pick-model affinity, or "lean builds"
   (README: "skilled players may favour lean builds").
   - The project already measured this. With the skill bucket in the draft-level calibration, affinity's Spearman
     gain shrinks from +0.012 to +0.005 (README, deck synergy v2, point 4).
   - In `power_sim.py`, a model that adds 0.5 × skill to its score, with *lower* true deck-strength correlation
     (−0.018), gains **+0.050 raw Spearman (z = 25)**. A real 2-3% noise reduction gains only +0.004 to +0.006.
   - The PickNet "surprise" feature is a drafter feature by design. Under the current metric it scores as a deck
     improvement.
2. **The blend weight is tuned for leakage.** The game FM's deck score is skill-adjusted: the skill bucket enters
   its context, not its deck score. But the weight α between the FM and LightGBM is picked on *val record NLL*. That
   pushes α toward whichever member better predicts records, skill included. So the blend gives up skill-adjusted
   signal for leaked signal.
3. **Matchmaking distorts records.** Arena pairs players by record, so W/(W+L) is compressed toward 50%.
   - `variance.json` reports a slope of 1.3 on the model logit. `matchmaking_variance.py` shows that total true
     variance cannot be identified from records alone.
   - Spearman is invariant to monotone transforms, so uniform compression would not hurt it. The compression is
     not uniform, though. It depends on the record path, so it reshuffles ranks among mid-record decks.
4. **W/(W+L) throws information away.** A 0-1 abandoned run and a 0-3 run both score 0.0. A 1-3 run (0.25) ranks
   below a 2-3 run (0.40) even though both lost three times. The target has only about 20 distinct values, so ties
   are heavy. Each draft gets the same weight however many games it has, while the binomial likelihood weights by
   games.
5. **Spearman is not a proper scoring rule.** It only checks ranking. That is fine for a deck rating, but it cannot
   reward a better-calibrated model, and blend weights tuned on NLL are then judged by a different criterion.
6. **The simulated "ceiling" in `deck_metrics` is invalid.** `simulate_runs` assumes independent games at fixed p
   and no matchmaking, and it treats the model's own p as the truth.
   - That p is *under-dispersed*: the slope is 1.3, so the true spread is larger. That makes the ceiling too low.
   - Ignoring matchmaking makes it too high.
   - The two errors don't cancel in any known way. On MSH the "ceiling" is **0.2445 while the model scores 0.2882**.
     DFT, DSK, EOE, FDN, LCI and MKM have ceiling ≈ score. Neither number is interpretable.
   - **Retire it.** Replace it with the share of true logit variance explained, from `matchmaking_variance.py`,
     which models matchmaking explicitly.

### 1.2 Testing protocol

1. **Multiple comparisons.**
   - One round (`improve_deck_picknet.py`) reports 12 rows × 12 sets = 144 test-set comparisons. Round 10 reports
     about 13 rows. About ten rounds have now reused the same test drafts.
   - With 12 null variants at |z| > 2, P(at least one "win") = 1 − 0.954¹² ≈ 0.43 per set.
   - Reporting the best row per round as the new champion picks winners on test. That makes the test split
     effectively a second validation split (adaptive overfitting, the "garden of forking paths").
   - The only defence that scales is a **pre-declared primary comparison per round, Holm across the declared
     finalists, and a vault holdout touched once** (§3.4).
2. **Validation is reused for many choices with a mismatched objective.** Val drafts early-stop every LightGBM fit
   and choose τ (5 values), the parameter grid (18 settings) and α (21 values), all by record NLL. The test then
   judges by Spearman.
   - In `improve_deck_v2.py`, α for the refit models is chosen with the *train-only twin* and then applied to a
     model fit on train+val. That is defensible but untested: the refit model is more confident, so its best α is
     probably lower.
   - None of this biases the test, but it makes the selection noisy. Selection noise then shows up as "variant X
     didn't help on set Y".
3. **The bootstrap covers test drafts, not training.**
   - The paired bootstrap in both scripts treats the fitted models as fixed. It answers "is this fitted model better
     on these drafts", not "is this method better".
   - LightGBM with `feature_fraction` < 1 and bagging has seed variance that was never measured. The README reports
     a transformer seed-to-seed spread of 0.003, larger than most of the gaps being tested.
   - The bootstrap uses 300 resamples with no time blocking (see the next point).
4. **Time drift and temporal correlation.**
   - The test split is the last 15% chronologically. That is right for deployment, but it means:
     - Test drafts are correlated within a day (metagame, who is playing, end-of-season rank rushes). An iid
       bootstrap understates the SE. A day-block bootstrap is the cheap fix.
     - Any method trained on data closer in time to the test window wins for recency, not modelling. Comparisons
       must keep the training window fixed across arms. Round 10 compares refit-on-train+val models against
       train-only models. The PickNet round uses train-only fits, so its gains are not additive with round 10's.
     - The late-format population (fewer casual players, more experienced 17Lands users) differs from training.
       Effects measured on the last 15% may not hold earlier in a format.
5. **Sets are treated as iid replicates, but they are not equal.** Test sizes range from 9.8k (MSH) to 23.6k
   (DSK) drafts, and the "mean over sets" weights them equally. Screening on MSH, the smallest and noisiest set
   and the outlier on every synergy measure, is the worst choice of screening set.

---

## 2. A primary metric for deck strength

The estimand: **the effect of the registered deck on the probability of winning a game, holding the player and the
opponent pool fixed.** A candidate must output a **deck-only score** s(deck). That means no skill bucket, no
experience and no PickNet surprise. Rank is allowed only through a standardisation step: evaluate LightGBM at a
fixed reference rank such as Platinum, so that rank × deck interactions cannot carry rank information.

### 2.1 Primary metric: skill-controlled held-out game log loss (SC-GLL)

For every test game g of a test draft d:

  logit P(win_g) = c(x_g) + a + b · z(s(deck_d))

- c(x) is a **frozen context model**, fit once per set on *train* games with no deck term and shared by every
  candidate. Its inputs are: skill-bucket one-hot, experience bucket, rank, record state before the game (21
  states, standing for the matchmaking opponent pool), play/draw and the mulligan difference.
- (a, b) are fit on train games, or on val games for models that were refit on train+val. That is two parameters
  per candidate, so calibration differences between candidates do not count. This matters because SC-GLL is meant
  to compare *rankings of decks*, as Spearman does.
- Score = −(mean test-game log loss) × 1000. Compare candidates with paired per-draft loss sums and a draft-level
  (or day-block) bootstrap.

`ctx_calibrated_game_loss` and `gll_metric` in `eval_protocol.py` implement this.

Why this metric:

- **It is a strictly proper scoring rule at the level where the data are iid given the state.** Bo1 games within a
  run are conditionally independent given deck, player and opponent, and the stopping rule factorises (README).
  Every game counts once, so abandoned runs and long runs get their correct weight.
- **Skill leakage is mostly removed.** Whatever the bucket captures about the player sits in c(x), so a score gets
  credit only for what it predicts *beyond* the player. In `power_sim.py` the leaky model's z falls from 25 (raw
  Spearman) to 7.5 (stratified Spearman) to **2.7 (SC-GLL)**. That is with a deliberately noisy bucket. A real
  bucket is a closer measure of skill, so the residual reward would be smaller still.
- **Matchmaking is handled exactly.** Within a record state every player faces the same opponent pool.

Caveats, with mitigations:

- **The skill bucket contains this draft's own results.** I believe `user_game_win_rate_bucket` is computed over
  the user's games in the format, including the test games.
  - `variance.json` supports this: "rank + skill bucket" explains ≈ 100% of the true between-draft variance. The
    bucket therefore absorbs deck strength as well as skill, especially for users with few games.
  - Mitigation: run the **primary metric only on games where the user is experienced (`user_n_games_bucket` ≥ 50)**.
    There, one 5-9 game draft is a small share (about 10-18%) of the bucket.
  - Report the all-users version as secondary. **Check the bucket's definition** before relying on this. Test it:
    if the bucket is computed over the whole format, then within a set it should not trend with draft_time for
    the same `user_n_games_bucket`.
- **Conditioning on record state is conditioning on a collider.** Deck strength and skill both cause the record.
  - Within a state (e.g. 2-2), strong decks sit with weaker players. This lowers b for every candidate, and lowers
    it more for skill-correlated scores. That direction is acceptable for our purpose.
  - Robustness check: **SC-GLL on match 1 only (record 0-0).** There is no selection there: every entrant faces
    the 0-0 pool at their rank. The cost is one game per draft, about 1/6 of the games, so the SE is about 2.4×
    larger. Use it as a sign check, not a decision metric.
- **Mulligans are partly caused by the mana base.** `game_controls.py` already tests dropping them. Pick the context
  set once, from that script's results, and freeze it.

### 2.2 Secondary metrics (always reported, never decisive)

| metric | what it adds | implemented |
|---|---|---|
| Raw Spearman(p, W/(W+L)) | continuity with the paper and the history | `spearman_wr` |
| Stratified Spearman | mean of within-stratum Spearman over strata of skill bucket × experience × rank (van Elteren weights). The same units as the headline, much less leak-prone. | `stratified_spearman` |
| SC-GLL, all users | power check; shows how much the experience filter costs | `ctx_calibrated_game_loss` |
| SC-GLL, match 1 only | matchmaking- and collider-free sign check | same, with a mask |
| **Leakage index** | (raw Spearman gain) − (stratified Spearman gain), plus Spearman(score, skill bucket) on test drafts. A change that raises the first and not the second is learning the drafter. | compute from the above |
| Share of true variance explained | replaces the ceiling (`matchmaking_variance.py`) | existing script |

### 2.3 "Same player, different deck": the within-draft build design

17Lands has no user id, so true same-player designs are impossible. The closest available proxy:
**`build_index` within a draft.** When a player re-registers a different 40 from the same pool, the player, pool,
rank and day are all held fixed exactly. Only the deck changes.

- The test is a conditional logit with draft fixed effects: logit P(win_g) = α_draft + c(record state, play/draw) +
  b · s(build_g).
- It is identified only from drafts with ≥ 2 builds that each played games. A higher b means the model better
  ranks two decks built by the same person from the same cards.
- This is the only design here that is fully immune to skill leakage.
- Problems:
  - Rebuilds are endogenous. Players switch after losses, so the state control is essential.
  - The sample is probably small. I could not count multi-build drafts without loading game data, which was off
    limits here.
- Cheap check for later, one set, two columns: the share of drafts with `build_index.n_unique() > 1` and ≥ 2 games
  on each build. If it is ≥ 5% (≥ 1k test drafts per set), add this design as a **leakage veto**: a change that
  improves SC-GLL but lowers within-draft b, meta-analysed across sets, is not adopted.

### 2.4 Matched pairs (an equivalent, interpretable form)

- Pair test drafts exactly on (skill bucket, experience bucket, rank, ISO week).
- Within each pair with different records, score 1 if the model ranks the better-record deck higher.
- The concordance rate is a stratified Kendall τ, equivalent to the stratified Spearman with slightly less power.
- Use it for communication ("in 54.1% of matched pairs the model picks the deck that did better"), not for
  decisions.

### 2.5 Power of the candidate metrics (synthetic, `power_sim.py`)

The world in the simulation:
- Each draft has a true logit sd of 0.30 and a skill sd of 0.25.
- Matchmaking is 0.08 logit per net win, and runs stop at 7 wins or 3 losses.
- The two models correlate 0.98 with each other, as real candidate pairs do.

The z-score for the *same* true improvement (+0.012 to +0.013 in corr(score, θ)):

| N test drafts | raw Spearman | stratified Spearman | SC-GLL |
|---|---|---|---|
| 10k | 3.45 | 4.26 | 4.01 |
| 25k | 4.48 | 4.59 | 4.88 |

The skill-controlled metrics are **at least as powerful** as raw Spearman, and 5-20% better in these runs, because
the context removes outcome variance that is not about the deck. Moving to SC-GLL costs no power. The
experience-only filter costs power in proportion to the share of games it drops. Measure that share on one set
first.

---

## 3. Cross-set aggregation and the decision rule

### 3.1 Per set: a paired delta with two variance components

For set j, candidate B against champion A, both trained with the **same training window and the same recipe**
(refit or not, the same recency τ):

- Δ_j = mean over seeds s of [M(B_s) − M(A_s)] on the same test drafts.
- SE_j² = Var_bootstrap(Δ_j) + Var_seeds(Δ_{j,s}) / S.
  - The bootstrap resamples **day blocks** of test drafts, so time-correlated error widens the interval.
  - Use S ≥ 3 seeds for anything stochastic (LightGBM bagging and feature fraction, neural nets). With one seed,
    pass a `seed_noise_floor` measured once per model family on 2-3 sets.

`paired_bootstrap` does this. A and B may be [S, N] arrays.

### 3.2 Across sets: a random-effects meta-analysis

- Treat each set as a study. Use DerSimonian-Laird τ² with the **Hartung-Knapp-Sidik-Jonkman** interval (t with 11
  df), floored at the DL variance ("modified HKSJ").
- The plain DL z-interval is anti-conservative at k = 12. HKSJ is the standard fix.
- In the self-test, the type-I error at τ = 0.001 is 0.029 against a nominal 0.05: slightly conservative, which is
  what we want.
- Also report the fixed-effect mean, I², Cochran's Q, a 95% **prediction interval for the next set**, the sign
  count and the Wilcoxon test.

Example: the existing per-set results for "+ deck vec + game FM" against round 4 (9 sets, from
`picknet/improve.json`; SEs are the existing test-only bootstrap):

```
random effects (DL + HKSJ) +0.00012, 95% CI [-0.00096, +0.00119], p = 0.81; tau = 0.00026, I2 = 3%
sets better 4/9; prediction interval for a new set [-0.0011, +0.0014]  -> not adopted
```

The log said "≈ +0.002 on MSH, BLB". The meta-analysis says the effect is zero: MSH and BLB were the upper tail of
noise. **I² ≈ 3% means sets hardly differ in their true effect**, so between-set heterogeneity is not what limits
detection here. The within-set SE is.

### 3.3 Pre-registered decision rule (`DecisionRule`, `decide`)

Write down before running: the primary metric (SC-GLL, experienced users), the champion, the candidate(s), the
seeds, the sets (all 12) and the rule. **Adopt B only if all of the following hold:**

1. The random-effects mean on the primary metric is > 0 with two-sided HKSJ p < 0.05. If K candidates are tested in
   one round, use Holm-adjusted p (`holm`).
2. **≥ 8/12 sets** have Δ_j > 0. This is a consistency guard, not a test: P(≥ 8/12 | null) = 0.19.
3. **No set is significantly worse** (Δ_j / SE_j < −1.96).
4. The point estimate is ≥ the smallest effect of interest. I suggest **+0.02 ×1000 nats/game** on SC-GLL (about
   +0.001 Spearman). Below that, the change is not worth its complexity even if it is real.
5. **Leakage veto.** The stratified Spearman gain must be ≥ 40% of the raw Spearman gain. If the within-build design
   is available, its meta-analytic b must not fall (one-sided p > 0.1).

Stop exploring a direction, with TOST equivalence, when the 90% HKSJ CI lies entirely inside ±(smallest effect of
interest). That says "no meaningful gain", which is more useful than "not significant".

### 3.4 Protocol hygiene

- **Screen on validation, confirm on test.**
  - Variants compete on val SC-GLL across all 12 sets (cheap, since val already exists).
  - At most 1-3 finalists per round reach test, and Holm runs across them.
  - The test row of a variant that lost on val is never reported as a "result".
- **Use a vault.**
  - The current test split has been reused for about 10 rounds, so treat it as dev-test from now on.
  - Prospective confirmation: hold out entirely each **new set** released after today (the 13th and 14th sets)
    and evaluate a frozen pipeline on them once.
  - Optionally carve off the last 5% of drafts of each existing set as an untouched vault for a final report.
- **Choose blend weights on the primary metric**, out-of-fold on val, never on record NLL. Two-component logit
  blends have one parameter, so cross-fitting on val (two halves) costs nothing.
- **Fix the training window.** Every arm gets the same refit/no-refit treatment and the same τ. "Refit on train+val"
  is a recipe change and is tested separately once. After that it is part of every arm.
- **Screen on the larger sets** (DSK, MKM, BLB, FIN: test N ≈ 20-24k), not MSH (9.8k), or skip screening and run
  all 12 sets.
- **Log the denominator.** In `research_log.md`, record every variant run on test, including the losers.

---

## 4. Power analysis

**Formula.** Spearman is roughly Pearson on ranks. For two predictors of the same outcome that correlate r_AB with
each other, at a common correlation ρ̄ ≈ 0.28 with the outcome, Steiger's (1980) dependent-correlation variance
gives:

  SE(Δρ) ≈ (1 − ρ̄²) · sqrt(2(1 − c) / N),  where  c = [r_AB(1 − 2ρ̄²) − ½ρ̄²(1 − 2ρ̄² − r_AB²)] / (1 − ρ̄²)²

| r_AB (A vs B) | N test drafts | per-set SE | per-set MDE (80%, α = .05) | 12-set MDE, τ = 0 | 12-set MDE, τ = 0.001 |
|---|---|---|---|---|---|
| 0.95 | 10k | 0.0030 | 0.0085 | 0.0027 | 0.0028 |
| 0.95 | 25k | 0.0019 | 0.0054 | 0.0017 | 0.0019 |
| 0.98 | 10k | 0.0019 | 0.0054 | 0.0017 | 0.0019 |
| 0.98 | 15k | 0.0016 | 0.0044 | 0.0014 | 0.0017 |
| 0.98 | 25k | 0.0012 | 0.0034 | 0.0011 | 0.0014 |
| 0.99 | 15k | 0.0011 | 0.0031 | 0.0010 | 0.0013 |
| 0.99 | 25k | 0.0009 | 0.0024 | 0.0008 | 0.0012 |

How the columns are computed:
- Per-set MDE = 2.8 × SE.
- 12-set MDE = (t₀.₉₇₅,₁₁ + t₀.₈,₁₁) · sqrt((SE² + τ²)/12) = 3.08 · sqrt((SE² + τ²)/12).

Validation:
- The simulation agrees with the formula. At r_AB = 0.981 the paired bootstrap SE is 0.0021 at 10k and 0.0011 at
  25k, against 0.0019 and 0.0012 from the formula.
- Both agree with the observed per-set SDs (0.0010-0.0024) in `picknet/improve.json`.

Takeaways:
- **Single set:** nothing under +0.003 to +0.005 is detectable. The MSH refit gain of +0.0052 ± 0.0016 is
  right at the limit for one set, and it has no seed component.
- **12 sets:** the MDE is +0.0011 to +0.0017 Spearman for typical candidate pairs. Gains of +0.001 need either
  r_AB ≥ 0.99 (small, targeted changes) or more test data.
- **SC-GLL:** the per-set paired SE is about 0.037 (25k) to 0.064 (10k) ×1000 nats/game, so the 12-set MDE is about
  0.03-0.05 ×1000. For scale, previous structural gains were 0.1-1.2 ×1000 (README tables).
- **Seeds:** with seed SD σ_s per set and S seeds, add σ_s²/S to SE². If σ_s ≈ SE (the transformer case), use S ≥ 4
  to keep the power loss under 20%.
- **More power**, best first:
  1. Pool all 12 sets (already planned).
  2. Use SC-GLL over every game rather than one record per draft.
  3. Raise r_AB by changing one thing at a time.
  4. Add new sets as they appear.
  - Extending the test window instead would trade against drift.

---

## 5. How to adopt this (order of work, all cheap)

1. Copy `eval_protocol.py` into `src/decks/` as `decks/evalproto.py`, or import it from here. It needs only
   numpy and scipy.
2. Add a **reference-rank deck-only score** to each candidate's saved predictions: per test draft, the model logit
   with rank set to a constant and no player features.
3. Write the frozen context c(x) per set once (train games) into `deck_synergy_v4/draft_inputs.npz` or a sibling
   file, along with per-test-game (draft index, y, ctx_logit, experience bucket, match number).
4. Change the improve scripts so they (a) save per-draft predictions for every seed, not only the summary, and
   (b) call `paired_bootstrap` with day blocks for SC-GLL, stratified Spearman and raw Spearman. A separate
   aggregation script then runs `meta_analysis` and `decide` over the 12 saved sets, so meta-analysis needs no
   retraining.
5. Run once, as a calibration of the new protocol rather than a decision: round 4 blend against round 10 blend.
   This shows how much of the +0.005 survives skill control.
6. Before relying on the experience filter and the build design, verify on one set, two columns at a time:
   - the definition of `user_game_win_rate_bucket`
   - the share of games with `user_n_games_bucket` ≥ 50
   - the share of multi-build drafts

## Open assumptions to verify

- `user_game_win_rate_bucket` and `user_n_games_bucket` are format-wide aggregates that include the test games.
  This is inferred from the near-100% variance explained in `variance.json`, not confirmed.
- Multi-build drafts are common enough for §2.3. Unverified: data loading was not allowed in this pass.
- Synthetic power figures assume the noise structure in `power_sim.py` (true sd 0.30, skill sd 0.25, matchmaking
  0.08 per net win). The analytic Spearman table does not depend on it and matches the observed SDs.
