# Meta-review 2: audit of the overnight verdicts, current champions, and the next 8 hours (2026-10-13)

What I read: `research_log.md` (all of it), `meta_review_1.md`, `evaluation_design.md`, the logs (`sc_gll`,
`game_fm_v2`, `game_fm_opp`, `hand_value`, `structural_a`, `rebuild_targets`, `game_controls`), and the per-set
JSONs under `results/<SET>.PremierDraft/{sc_gll,game_fm_v2,game_fm_opp,hand_value,targets}`. I also read the
scripts that produced them: `sc_gll.py`, `game_fm_v2.py`, `hand_value.py`, and the parts of `rebuild_targets.py`
and `game_fm_opp.py` that matter here.

Compute was limited to numpy meta-analysis over the saved per-set JSONs, a few seconds each. No parquet or game
file was opened. The transformer queue (PID 30099, MSH/FDN/BLB) was still running on MPS when I finished. Every
re-aggregation below uses `eval_protocol.meta_analysis` (DL + HKSJ), with units of ×1000 nats/game, higher is
better.

**Verdict in one paragraph.**
- Most of the night's adoptions hold up, but three of them rest on a metric whose context contains the
  champion's own controls.
- **Robust:**
  - The game FM beats LightGBM as a deck rater.
  - Pair terms are real (FM beats the linear game model).
  - Skill control is essential.
  - opp_colors is post-treatment.
  - The round 10 record recipe replicates.
- **Not robust, or not yet decided:**
  - "The round 4 blend is worse than the FM alone". The gap (−0.21) is the same size as the context-matching bias
    I measure below (about 0.18-0.24).
  - "Keep mulligans". The judge conditions on a mediator of deck strength.
  - "The raw bucket stays". On the pre-registered primary metric, SC-GLL-dl, it is a tie.
- **Hand-value deck-power slope:** a bug in the analysis, not a finding.
- **Process deviation:** meta-review 1 made SC-GLL-dl the primary metric and the raw-bucket context a sensitivity
  row. Tonight's verdicts used the raw-bucket context as primary, and `sc_gll.py` never computed dl-ctx at all.

---

## 1. Audit of tonight's verdicts

### (a) "The game FM beats LightGBM as a deck rater": robust for LightGBM, not for the blend

**What the log under-states.** The log says the result held "on match-1-only and main-build subsets (MSH)". In
fact `sc_gll.log` has every subset for all 12 sets, so the evidence is stronger than written:

| LightGBM − game FM | all | experienced | match 1 | main build |
|---|---|---|---|---|
| sets better | 0/12 | 0/12 | 0/12 | 0/12 |
| typical size | −0.44 … −1.16 | −0.66 [−0.80, −0.52] RE | −0.20 (DFT, n.s.) … −1.68 | −0.51 … −1.25 |

**The problem with the judge.** Every subset in `sc_gll.py` uses the same context, which contains the **raw**
skill bucket. The raw bucket includes this run's own games, so part of each deck's realised strength already sits
in c(x).
- The champion FM was *trained* with exactly that covariate set: play/draw, mull diff, raw bucket, rank,
  experience, record. Its deck coefficients are therefore the part of deck strength that is orthogonal to the
  leaky bucket, which is exactly what the judge credits. That is a home-context advantage.
- LightGBM was trained on records with rank only. The part of its genuine deck signal that runs parallel to the
  leaky bucket is redundant under this judge.

**How large the home advantage is.** `game_fm_v2` lets me measure it:

| arm vs champion | raw ctx | dl ctx | how much the gap moves |
|---|---|---|---|
| rated-dl (FM trained on the de-leaked bucket) | −0.219 (1/12) | −0.037 [−0.121, +0.047], 4/12, n.s. | +0.18 |
| − skill | −0.374 (0/12) | −0.134 [−0.252, −0.016] | +0.24 |

So swapping the context moves paired deltas by about 0.2 ×1000.

**What survives.**
- LightGBM − FM (−0.66) is about 3× that bias and is negative on 12/12 sets in every subset. Applying the − skill
  shift (+0.24) still leaves about −0.4. **Robust.**
- Round 4 blend − FM (−0.21, 0/12 better, 6 significantly worse) is about the size of the bias. Separately, the
  blend's α was picked on **val record NLL**: FM weight 0.17-0.50, so LightGBM dominates. Meta-review 1 asked for
  α on val SC-GLL and that was not done.
- **"Blend < FM" is therefore unproven.** A blend with α chosen on SC-GLL-dl cannot lose to the FM except by
  cross-fitting noise, because α = 1 *is* the FM.

**Supporting signal.** At the reference rank, LightGBM has the *higher* raw Spearman (0.2517 vs 0.2477, 12-set
mean) but the lower SC-GLL slope (b/sd 0.212 vs 0.226). It ranks records better and predicts beyond-player game
outcomes worse. That is the skill/record signature, consistent with the conclusion.

**Decisive test (T1, numpy only, about 30 min for 12 sets).** Rescore the saved scores under the full context
matrix.
- Inputs: `sc_gll/scores.npz` (LightGBM 3 seeds, blend, FM, linear) and `game_fm_v2/scores.npz` (all arms).
- Contexts: {raw, dl, pre-raw, pre-dl}. "pre" is the context without *own* mulligans, explained in (c).
- Add `blend*`: α chosen per set on val SC-GLL-dl, cross-fitted over two val halves.
- **Rule:**
  - FM-over-LightGBM is confirmed if LightGBM − FM < 0 with the 95% HKSJ CI excluding 0 under dl and pre-dl.
  - The FM alone stays the deck rater if blend* − FM < +0.02 under dl. (If α* ≥ 0.8 in ≥ 9/12 sets, LightGBM
    adds nothing.)
  - If blend* − FM ≥ +0.02 and decide() passes, **the deck rater becomes the SC-GLL-tuned blend.**
- A second leg that removes the home advantage completely: LightGBM trained on draft records with the per-draft
  summed dl-context logit as an offset (or with the dl bucket as a feature), then scored at the reference skill.
  This is only needed if T1 is ambiguous.

### (b) "Refit on train+val" with the champion's val calibration: no optimism found, but the evidence is weaker than "ADOPT" implies

**Mechanics: clean.**
- The refit replays the champion's val-chosen grid point and epoch/lr-drop schedule. It does no validation on val
  games, which are now in-sample.
- It is calibrated with the champion's val (a, b) in raw units, paired per seed.
- The context model is fit on train games only. Test games are untouched.

**The direction of the borrowed calibration is conservative.**
- refit with borrowed (a, b) = +0.064. refit with its own, in-sample val fit = +0.054.
- So the borrowed calibration does not flatter the refit.

**Weaknesses.**
1. On the **pre-registered primary metric** (experienced dl-ctx), the gain is **+0.036 [+0.003, +0.069],
   p = 0.035, 10/12, 0 worse**.
   - That passes decide() standing alone.
   - It **fails Holm** across the 5 challengers scored that round. With p's 0.0095 (− mull), 0.029 (− skill), 0.035
     (refit), …, Holm stops at the second.
   - Match-1 dl-ctx: +0.055 [−0.035, +0.145], 9/12. Right sign, not significant.
2. The secondary metric disagrees in sign. Draft Spearman is −0.0009 (4/12), raw −0.0007, stratified −0.0003. The
   gain sits only in game log loss, so part of it may be scale or intercept that the twin calibration happens to
   suit better.
3. The gain mixes "more data" with "data closer to the test window" (val is the most recent 15% before test). The
   recency arm is null (−0.002), which favours data quantity. Either way it is a deployment-valid gain, not a
   modelling insight.
4. Operational: a refit champion has no clean val. Every future screen must be done on train-only twins, and every
   deployment needs a twin for (a, b).

**Verdict.** Keep the refit in the champion recipe, provisionally. **The vault decides it** (T5): 6-8 fresh sets
give a direct confirmatory test of the refit, pre-registered as one-sided.

### (c) The mulligan and skill-control reversals: yes, the SC-GLL context biases them toward the champion's controls

**Mulligans.**
- The context term is `num_mulligans − opp_num_mulligans`.
  - The *own* mulligan is partly **caused by the deck**: land count, curve, colour requirements.
  - The *opponent's* mulligan is exogenous noise.
- Every SC-GLL context used tonight (raw, dl, opp-ctx, all subsets) contains the mull difference. So the judge
  credits only the deck's effect **not mediated by own mulligans**, which is the champion's estimand by
  construction.
  - The − mull arm learns the total effect, including mana-base mulligans, and its extra component is "already
    explained" by the context.
  - Its −0.060 (raw) / −0.041 (dl) / −0.054 (match 1 dl) is therefore an expected artefact, not evidence.
- The opposing evidence is also weak:
  - The − mull Spearman gain (+0.0021 calibrated, 12/12) is +0.0018 raw but only **+0.0003 stratified (5/12)**, so
    about 85% "leakage".
  - But the strata use the raw bucket, which over-controls in the same way.
- **Status: undecided.** It is also an estimand choice. For "how good is this 40", mana-base mulligans *are* deck
  strength, so the total effect is the natural target. The judge must not condition on own mulligans.

**Skill.**
- − skill loses under every context, including dl: −0.134, 1/12. Match-1 dl: −0.270, 1/12. Stratified Spearman:
  −0.0019, 2/12.
- An evaluation context without any skill term would be the − skill arm's home and would reward leakage, so dl-ctx
  is the fairest available judge. **"Skill control essential" stands.**
- **Raw vs de-leaked bucket** is the real open question.
  - The raw bucket wins only at home (−0.219 raw ctx). Under dl-ctx it is a tie (−0.037, 4/12). Match-1 dl:
    −0.115, p = 0.045, 3/12.
  - The rated-dl arm has 8% larger card-value spread and 12% larger deck-score sd, consistent with the raw bucket
    attenuating deck effects (conditioning on a descendant of the outcome).
  - **Status: tie on the primary metric.** Keep the raw bucket by default, labelled unresolved.

**Decisive test.** The bias comes from what the judge's context contains, so the decisive judges are ones with no
player or mediator variables in them:
- **T1 (context matrix, above):** add a "pre" context with own mulligans removed (opp mulligans kept, dl bucket).
  Report every arm × every context. A choice is adopted only if it wins **off its home context**.
  - The champion's home is raw + mull.
  - The − mull home is pre.
  - rated-dl's home is dl.
- **T2 (new arms, game_fm_v2):**
  - `− own mull`: split the mull difference into own and opp columns and drop own.
  - `rated-dl − own mull`.
  - A neutral re-parameterisation of the champion with split columns.
  - 2 seeds, 12 sets, about 50 min.
  - Also save the per-build deck scores and the card power w for T3/T4.
- **T3 (the referee): a within-build conditional logit.**
  - Draft fixed effect. Context = record state, play/draw, opp mulligans only: no player variables and no own
    mulligans.
  - Fit on drafts with ≥ 2 builds that each played games.
  - Player, pool, rank and day cancel exactly, and mana-base mulligans count as deck strength.
  - The arm with the higher within-build b rates decks better, whichever context it was trained with.
  - **Gate:** ≥ 1,000 qualifying test drafts per set, or ≥ 5,000 pooled. The count (T0) was still not done
    tonight, although meta-review 1 asked for it.
- **Go/kill:**
  - Adopt `− own mull` (or rated-dl) if both:
    - T1 pre-dl Δ ≥ +0.02 and decide() passes;
    - T3 within-build b is not lower (one-sided p > 0.1).
  - If T1 pre-dl Δ ≤ 0 **and** T3 is not higher, close mulligans for good: keep the control.
  - If T1 and T3 disagree, T3 wins when it has power. Otherwise keep the status quo and record the estimand caveat.

### (d) opp_colors is post-treatment: verified

From `game_fm_opp/results.json`. Opponent strength is a logit effect with the champion's full game logit as
offset, game-weighted over 12 sets and grouped by the number of colours revealed:

| opp colours revealed | strength | user WR | share of games |
|---|---|---|---|
| 1 | −0.88 | 74.4% | 4.8% |
| 2 | +0.03 | 54.5% | 65.6% |
| 3 | +0.10 | 52.6% | 23.8% |
| 4 | +0.21 | 49.6% | 2.8% |
| 5 | +0.31 | 47.0% | 1.4% |
| unknown (nothing revealed) | **−3.00** | **96.2%** | 0.3% |

- The gradient is monotone in the colour count.
- "Unknown" wins 96% of the time: the opponent conceded before revealing a coloured card.
- 4.8% "mono-colour" opponents is far above the real mono share in Limited.

This is the signature of a variable measured by the game itself. Long games reveal more colours.

Two more consequences the log does not draw:
- The opp-ctx improves the *context-only* test log loss by about 5.5 ×1000 (0.6599 → 0.6544 on MSH). That is 30×
  any deck effect, and it is only possible because opp_colors encodes the outcome.
- The **RPS matchup Q test** (e.g. BLB p = 1e-39) and **"opponent strength rises with wins before"** (a 0.011
  logit range after the record offset) are equally contaminated:
  - Higher-record games face stronger opponents.
  - Those games are longer, so more colours are revealed.
  - Neither result should be cited as matchup or record-matching evidence. The play/draw anchor remains the only
    valid record-matching evidence.

**Remaining check, cheap:** mean `num_turns` by revealed colour count. It is in the slim game parquet's META. Run
it as a one-column scan inside T0. Expect it to rise with colour count.

**Verdict.** Not adopted, for the right reason. opp_colors is dead as a control. The only legitimate use is
opponent-archetype effects from *pre-game* information, which the data does not contain.

### (e) The hand-value puzzle (deck-power vs HV slope 0.006-0.05, implied λ < 0): an analysis bug

**Diagnosis from `hand_value.py`.**
1. **The deck model contains the FM deck score.** `load_set` appends the round 4 `game_logit` (the game FM deck
   score) and a has-deck flag to `dense` ("using round 4 game_logit as a covariate", logged for all 12 sets).
   - `dense` is meant to be a precision covariate for the outcome model p̂. That use is fine: the Robinson
     estimator is unbiased for any p̂.
   - But the **same `dense` goes into the "deck model"** (`fit_logit(data, tr, data.cats, lam)`) whose card power w
     is then regressed on HV.
   - So w is the *residual* card power beyond the FM's deck score. Its slope on HV must be close to 0, giving an
     implied λ of about −0.6 to −0.7, exactly what all 12 sets show.
2. **The ridge sits on the grid's upper edge.** λ_deck = 0.03, the maximum of `LAMS`, in 11/12 sets (DFT 0.003).
   With the FM score present there is nothing left for w to explain, so validation picks maximal shrinkage. That
   compresses w further.
3. **The prior experiment has the same flaw.** "HV as a ridge prior for card power" was tested against a model that
   already contains the FM deck score.
   - "+0.006 ± 0.013" therefore means "HV adds nothing beyond the game FM's deck score". It does not mean "HV is a
     useless prior for card power".
   - Per set the sign is mixed: MSH −0.085, FIN +0.066.
4. **Mislabel.** The log row "game FM card power w (r 0.44, tilt 0.08) ← least skill-tilted" is this residual
   ridge w, not the FM's w. Its low skill tilt is an artefact of residualising on the FM score. **Strike that row**
   from any write-up.
5. **The mapping is mis-specified even after the fix.**
   - The FM attributes card value to 17 Scryfall deck features (curve, creature count, removal and so on) and to
     pair terms as well as w.
   - A slope against w alone will sit below κ(λ = 0) = 7/N ≈ 0.175.
   - Compare HV against the card's full marginal: w + Σ_k β_k ∂f_k/∂card + mean pair contribution in the decks
     playing it.

**Two further caveats on the HV headline** ("beats every 17Lands stat").
- **The target is circular.** Ratings are judged against *test-game HV*. HV-train shares that estimator's specific
  biases, so it wins partly by construction.
- **Keep-selection is present.**
  - The balance index correlates with HV at **+0.18 to +0.59 in 12/12 sets**: good spells are over-represented in
    recorded hands. That is the design's own failure flag.
  - The core-subset robustness check (L ∈ {3, 4}, no mulligan) is reassuring, with r = 0.96-0.99, but it is not a
    fix.

**Fix (T4).**
- Remove `game_logit` from the deck model and the prior arms. Keep it in p̂.
- Widen `LAMS` to 3e-7 … 3e-1.
- Compute the slope against the champion FM's full per-card marginal (needs T2's saved w and β).
- Add a **non-circular rating test**: for each rating (HV, GIH, OH, IWD, FM w), score decks by the sum of card
  ratings and compare them on SC-GLL-dl test games.
- Add a selection-free HV variant: turn-1 and turn-2 *draws*, which are random given the library and involve no
  keep decision. Compare it with the opener HV.
- **Expected:** a slope of about 0.1-0.3 and λ ∈ [0, 1]. If λ is still < 0 after the fix, the additive
  hand-to-deck mapping really fails, and that becomes a finding.

---

## 2. Current best models (precisely)

### (i) Deck strength: "champion v2"

- **Model:** game-level `DeckFM` "fm4 nonland":
  - linear card power w over all cards
  - rank-4 factorised pair terms between **nonland** cards only
  - 17 standardised Scryfall deck features
  - the PickNet pool_d32 affinity profile column
- **Context, game only, not in the deck score:** play/draw, mulligan difference (own − opp), raw
  `user_game_win_rate_bucket` + known flag, per-game rank one-hot, experience one-hot (500/1000 fix), and the
  record state before the game. Opponent rank is dropped (always None).
- **Training:**
  - Adam 3e-3, batch 4096, patience 3, 2 lr drops.
  - Grid (l2_w, l2_v) ∈ {(1e-5, 1e-3), (1e-5, 1e-2)} chosen on val game log loss.
  - **Refit on train+val**, replaying the champion's schedule.
  - Calibrated with the train-only twin's val (a, b).
- **Deck score:** the deck-only logit of the most-played build (card power + pairs + deck-feature and affinity
  terms, no context). Test-draft sd is about 0.25 logit; it is about 15% compressed per model A.
- **Numbers (12 sets, experienced, ×1000 nats/game):**
  - vs LightGBM: +0.657 [+0.519, +0.795], 12/12.
  - vs the linear game model: +0.155 [+0.098, +0.213], 11/12. MSH −0.016 and FDN +0.025 are flat.
  - vs the round 4 blend: +0.206 [+0.093, +0.318], 0/12 for the blend. **Unresolved, see (a).**
  - refit vs train-only: +0.064 raw / **+0.036 dl** (p = 0.035, not Holm-significant).
  - Secondary: calibrated draft Spearman 0.2670 train-only and 0.2660 refit. Raw deck-only Spearman about 0.247.
- **Unresolved:**
  - blend* (α on SC-GLL) vs the FM.
  - The own-mulligan control (estimand).
  - Raw vs de-leaked bucket (a tie on dl-ctx).
  - Vault confirmation of the refit and of the FM-over-LightGBM claim.
  - The transformer (running).
  - Within-build validity (never measured).

### (ii) Record prediction: "round 10 recipe"

- **Model:**
  - LightGBM with the round 4 feature set: card counts + structure features + rank + deck features + affinity.
  - Refit on train+val with recency weights τ = 2.
  - Tuned params (`num_leaves` 15, `min_data_in_leaf` 300, `feature_fraction` 0.6, `lambda_l2` 10). These come
    from MSH's `improve_v2.json`; the other sets use defaults with τ = 2 (see `targets.json` settings).
  - 5 seeds averaged.
  - Logit-blended with the **cached round 4 game FM** (not champion v2), α from the train-only twin on val record
    NLL.
  - Evaluated at the player's own rank.
- **Numbers (12-set mean test Spearman):**
  - Trained and evaluated on game-file targets: **0.2946**. Trained on game targets, evaluated on draft targets:
    0.2810.
  - Round 4 blend: 0.2783 on draft targets / 0.2915 on game targets.
  - Round 10 − round 4: +0.0023 [+0.0014, +0.0033] on draft targets, +0.0028 [+0.0020, +0.0036] on game targets,
    12/12.
  - Training on game targets instead of draft targets: +0.0006 (p = 0.02, 10/12).
- **Unresolved:**
  - Seed averaging exists only in the treatment arm, worth about +0.0005 to +0.001 of the +0.0023.
  - The FM member is the old one. Swapping in champion v2 costs nothing to try.
  - The game-file target is incomplete for 26-31% of drafts in MSH/EOE/FIN/SOS/TDM, so "eval game" numbers are
    not comparable across sets.
  - This is a record-prediction product that includes rank and skill-correlated signal. It is **not** a deck
    rating and must not be presented as one.

---

## 3. Ranked plan for the next ~8 h of serial compute (after the transformer queue)

**Ground rules.**
- One heavy job at a time, niced, ≤ 4 threads, MPS only after PID 30099 exits.
- Every script saves per-draft and per-game artefacts so that meta-analysis never needs a refit.
- **Pre-register in `research_log.md` before each step:** the metric (SC-GLL-dl experienced primary, raw-ctx
  sensitivity) and the go/kill rule. Holm across the arms of a step.
- Measured runtimes make all of this cheap:
  - game_fm_v2: 2-5 min per set for 6 arms × 2 seeds
  - sc_gll: 1-3 min per set
  - deck_synergy_v4: about 4 min per set
  - hand_value: 2-18 min per set

### T0. Gate counts (0.25 h, CPU, < 1 GB, narrow columns)
- **Per set:** the number of test drafts with ≥ 2 builds that each have ≥ 2 games.
- **Per set:** the share of experienced test games.
- **Per set:** mean `num_turns` by revealed opponent colour count, to close (d).
- **Gate:** T3 runs only if ≥ 1,000 qualifying test drafts per set, or ≥ 5,000 pooled.

### T1. SC-GLL context matrix plus SC-GLL-tuned blend (0.5 h, numpy, no training)
- Rebuild the per-game tables (the same loaders as sc_gll, narrow columns).
- Fit 4 contexts on train games: raw, dl, pre-raw, pre-dl.
- Rescore every saved score. Add blend* (α on val SC-GLL-dl, two-half cross-fit).
- **Go/kill:** see (a) and (c). Of all the steps here, this one carries the most decision value for its cost.

### T2. game_fm_v2 arms with split mulligans, saving per-build scores and w (1.0 h, CPU)
- Arms: champion with split mull (neutral), − own mull, rated-dl − own mull, refit of the winner. 2 seeds.
- Score under T1's contexts.
- **Go/kill:** see (c). If `− own mull` passes, it joins the frozen pipeline **before** T5.

### T3. Within-build referee (0.5 h, CPU; gated by T0)
- A conditional logit with draft fixed effects (context = record state, play/draw, opp mulligans) on the per-build
  scores of every arm, plus LightGBM per build (3 seeds, ~1 min per set).
- **Go/kill:** within-build b is the leakage veto from meta-review 1. Any arm adopted in T1/T2 must not lower it
  (one-sided p > 0.1). It also arbitrates raw vs dl bucket.
- If T0 fails the gate, report N and kill.

### T4. Hand-value repair (1.0 h, CPU)
- Run the fix described in (e).
- **Go/kill:**
  - If the HV-sum deck score beats the GIH-sum deck score by ≥ 0.02 on SC-GLL-dl (decide()), the claim "HV is the
    best card rating" is upgraded from circular to external.
  - If the fixed prior arm gains ≥ +0.02 vs ridge (12-set RE), test it as an FM prior next round. Otherwise close
    it.

### T5. Vault confirmation of the frozen pipeline (3.0 h; MPS for pool models, CPU otherwise) [never cut]

**Freeze first.**
- Write `results/methodology/frozen_pipeline.json` and add a log entry before touching any vault file. It
  records:
  - the champion v2 recipe, plus any T1/T2 adoption
  - the record recipe (round 10 with the champion v2 FM swapped in if T1 has run; otherwise round 10 as is)
  - seeds = 2
  - the confirmatory hypotheses below, each one-sided

**Readiness, checked tonight.**

| sets | draft parquet | game parquet | picknet pool_d32 | other |
|---|---|---|---|---|
| ECL, TLA | yes | yes | yes | Scryfall feature dirs exist |
| HOB, MOM, TMT, WOE | yes | yes | **no** (no picknet results dir) | need pick shards + pool training |
| NEO, ONE | **no** (raw draft CSV only) | yes | no | no `game_time` column; **exclude** |

- ECL was used for play/draw outcome statistics in the matchmaking memo. That is not model selection and is
  acceptable; note it.
- MOM also lacks `game_time` (day blocks fall back to `draft_time`; check this).

**Per set, about 25-30 min:**
1. Pick shards + pool_d32, MPS, train split only. HOB/MOM/TMT/WOE only.
2. `deck_synergy_v4`, for draft_inputs.
3. `sc_gll` (+ dl-ctx).
4. `game_fm_v2` champion + refit.
5. `rebuild_targets`: round 4 vs round 10 Spearman.

Order: ECL, TLA, WOE, MOM, HOB, TMT. TMT is the smallest.

**Kill rule for affinity.**
- If pool training takes more than 15 min per set, run HOB/MOM/TMT/WOE with a pre-declared **no-aff** champion and
  no-aff LightGBM (both arms equal).
- Pool them separately from ECL/TLA.

**Confirmatory hypotheses (one-sided, RE meta over the vault sets, then a 12+k pooled estimate):**
- H1: FM − LightGBM > 0, dl-ctx.
- H2: FM − linear game model > 0.
- H3: refit − train-only > 0, dl-ctx. Underpowered at k = 6 (MDE ≈ 0.06): it "confirms" if the vault mean > 0 and
  the 18-set pooled p < 0.05.
- H4: round 10 − round 4 Spearman > 0.
- H5 (only if adopted by T1/T2): the new arm vs champion v2.

**Kill / downgrade.** If an H's vault CI includes 0 with its point estimate ≤ 0, that claim is labelled
"dev-test only" in every write-up. No re-tuning on the vault, ever.

### T6. Morning write-up (0.5 h, no compute)
- Rewrite the research-log headline. Then produce `reports/` "Deck strength: state of the evidence", with one
  table per product: deck rater (SC-GLL-dl, within-build b, vault) and record predictor (Spearman, vault).
- Include a struck-claims box:
  - the HV "FM w" row
  - opp RPS / opp-strength-by-wins
  - "blend < FM", unless T1 confirms it
  - "keep mulligans", unless T1/T3 confirm it
- Keep a running denominator of variants scored on test.

**Schedule:** T0 0.25 + T1 0.5 + T2 1.0 + T3 0.5 + T4 1.0 + T5 3.0 + T6 0.5 ≈ 6.75 h, leaving about 1.25 h of
slack.
- If time slips, cut T4 first and T3 second. **Never cut T5.**
- If the transformer finishes after about 06:00, start T0-T1 at once. They are CPU-light and do not touch MPS.
  Defer only T5's pool training until MPS is free.

**Transformer (already running).** Score it under T1's dl and pre-dl contexts, not raw only.
- **Kill B2/B3** unless the best B1 config gains ≥ +0.02 on val SC-GLL-dl over champion v2 (3-set mean) **and**
  does not lower within-build b.
- A gain only under raw-ctx does not count.
