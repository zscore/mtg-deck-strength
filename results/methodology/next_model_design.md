# Next-generation deck-strength model: design

Memo, 2026-10-12. This is design work only. Nothing here was trained on real data. The code drafts are in
`results/methodology/drafts/`. Their synthetic self-tests ran in under 30 s each on 2 CPU threads, and their outputs
are saved next to them (`*_selftest.txt`, `mm_likelihood_large_s.txt`).

Related memos: `matchmaking_identification.md` covers what records can identify (the t ≈ s ridge, and s from the
play/draw shock). `evaluation_design.md` and `eval_protocol.py` cover SC-GLL, the 12-set meta-analysis and
`decide()`. `pipeline_audit.md` covers leaks and stale records. This design depends on all three and does not
repeat them.

## Bottom line

1. **Candidate A (matchmaking-aware likelihood)** fixes the *scale* of deck ratings. In simulation it does
   **not** change their *order*. At the population spread the identification memo estimates (s ≈ 0.55-0.85), iid
   game models give deck scores at 0.78-0.82 of the true scale. That is a slope of 1.22-1.28, the same
   compression seen on real data (1.2-1.7). The structural likelihood with s fixed recovers 1.03. Spearman
   against records is identical to the third decimal in every scenario where matchmaking is the only difference
   (0.3469 vs 0.3478; 0.3067 vs 0.3065). So adopt A for what it is: calibrated, interpretable ratings ("win
   probability against an average opponent") and the correct likelihood for B. Do not adopt it for Spearman.
   Expected blend effect: −0.002 to +0.002.
2. **A's real ranking lever is what replaces the leaky skill bucket.** In the skill-confounded simulation, dropping
   the bucket raised record Spearman from 0.248 to 0.327. It also made card ratings worse: r with the true deck
   value fell from 0.948 to 0.903. That is the same trade-off the real data shows between the game FM (bucket in)
   and LightGBM (no bucket). A therefore ships in two variants, *record* (random effect only) and *rated*
   (random effect + de-leaked bucket). They are tested separately, because the blend may want the decorrelated
   one.
3. **The scale is identified only through s.** The self-test reproduces the memo's ridge. With s free or tied, the
   fit drifts up the (s, t) ridge: s = 0.47-0.57 against a true 0.39, and the scale overshoots to 1.04-1.13. With
   s fixed at the truth it lands at 1.02. **Fix s from the play/draw estimate and profile it over {0.4, 0.55,
   0.7, 0.85}.** Ranking does not depend on s; only scale does.
4. **Candidate B (game-level set transformer)** is a d=128, 3-layer set transformer with 1.05M trainable
   parameters. Each card is a frozen PickNet all_d256 vector plus a learned residual. It has a per-set
   linear-power skip path, a trunk shared across sets with per-set heads, A's likelihood, and an α-weighted
   distillation term toward the cross-fitted LightGBM-blend logit. Measured on CPU (2 threads, machine at load
   10): 176 ms per step at batch 256 and 661 MB peak RSS. The estimated full 12-set run is 1-2.5 h on MPS, and
   the whole grid is about 35-55 GPU-hours. Expected gain on the blend: +0.000 to +0.004 Spearman.
   Distillation's synthetic win is meaningless (the synthetic teacher was near-oracle). The deciding ablations
   are PickNet-init vs random-init and α = 0 vs α > 0, judged by the 12-set paired rule.
5. **Order of work.** Data prep and teachers come first (CPU, about 3 h). Then the A grid on MSH/FDN/BLB (CPU,
   about 3 h). B is gated on A being non-inferior as a likelihood, because B is built on it.

---

## 0. What the design has to respect

| Finding | Consequence for the design |
|---|---|
| Champion: LightGBM (counts + structure + Scryfall deck features + affinity, rank) logit-averaged with the game FM. Mean 0.279 over 12 sets; refit on train+val + recency (τ = 2) + 5 seeds gives MSH 0.2934 (+0.005) | Every candidate is judged *inside* that blend, with the same refit/recency treatment and the blend weight picked on val with train-only twins (`improve_deck_v2.py` pattern) |
| Records are compressed by record matchmaking (stage-2 slope 1.2-1.7). Total variance is not identified from records | A models the opponent pool explicitly. Its scale depends on s, which comes from outside the records (play/draw shock) |
| Rank indexes the opponent pool; `opp_rank` is None everywhere | Rank enters only as a *net* per-set offset ρ[rank]. Deck ratings exclude it |
| The skill bucket is leaky (it includes this run's games) | No raw bucket in A/B's *record* variant. The *rated* variant uses the de-leaked bucket (`matchmaking_identification.md` §2.4) as a covariate on the random-effect mean |
| Transformers tie linear per set; logit r with LightGBM 0.86-0.93; ensembles add ≤ 0.004 | B is a blend member and a carrier for the likelihood, not a replacement. Its gate is a 12-set paired gain, not a single-set win |
| PickNet features add about 0 within a set (+0.002 on MSH/BLB, ≤ 0 elsewhere) | PickNet vectors are *initialisation and prior* (frozen base + residual), not extra signal. Expect stability and small-set/rare-card gains, not a step change |
| Gains are at the 0.001-0.005 scale; the detectable effect at 80% power is about 0.0011-0.0017 (12-set meta) | All decisions go through `eval_protocol.decide()`: SC-GLL primary, Spearman secondary, ≥ 3 seeds for neural members |

---

## A. Matchmaking-aware game-level likelihood

Code: `drafts/mm_likelihood.py` (`opponents`, `OpponentPool`, `MMGameModel`, `fit`, `simulate`, `selftest`).

### A.1 Model

For draft i of a set, game g at record state σ_g = (w, l) before the game:

```
e_ig    = d + η(deck_b(g)) + β·cov_g + ρ[rank_g] + t·u_i                     u_i ~ N(0,1), integrated (Gauss-Hermite, 12 nodes)
P(win)  = Σ_j pool[σ_g, j] · expit(e_ig − s·x_j)                              structural
P(win)  = expit(e_ig − m(σ_g)),  m(w,l) = A[w] − B[l],  A, B ↑ (cumsum softplus), m(0,0) = 0     constrained parametric
L_i     = log ∫ Π_g P(y_g | u) φ(u) du                                       exact per-draft marginal likelihood
```

- **η(deck)** is any deck scorer. For A it is round 4's best: per-card power w, rank-4 nonland FM pairs, Scryfall
  deck features, and the affinity column. For B it is the transformer. η is evaluated per *build*, so a mid-run
  rebuild gets its own score. The draft-level rating is η of the most-played build, as
  `decks.deckstrength.most_played_deck` defines it.
- **cov** is on_play − ½, the mulligan difference and the experience one-hot. It drops the raw skill bucket, the
  constant opp_rank one-hot and the 21 free record dummies. Those dummies are the things being replaced.
  (`game_controls.py` and evaluation_design §2.1 question whether mulligans belong at all, since the mana base
  partly causes them. Keep them for comparability with v4, then ablate.)
- **pool[σ]** comes from the `matchmaking_variance.opponents` recursion, rewritten in torch so it is
  differentiable in s. Everyone starts 0-0 from N(0, s²). Players at a record meet players at the same record
  (`match="record"`) or the same win count (`match="wins"`). With fuzziness λ, the opponent instead comes from
  the all-games mixture. That mixture is a fixed-point iteration, 3 passes. Row 21 serves games whose record is
  unknown because the 17Lands history is incomplete. Without it those games would have to be dropped.
- **s.** Fix it per set (or pooled) from the play/draw Wald estimate and profile over a grid. The self-test shows
  why it should not be learned. **d** is the users' mean above the pool (about 0.3 on MSH). **t** is learned:
  it is the residual spread of player plus run luck.

### A.2 Why it should give decompressed ratings, and when they will be better ranked

**Three mechanisms compress an iid game model.**

1. **Omitted opponent strength.** A strong deck climbs and meets harder opponents. That omitted variable is
   negatively correlated with η, so iid logistic coefficients shrink.
2. **Collider bias from free record dummies.** The record before game g is caused by both the deck and u. Inside
   a record cell, the weak decks at 5-0 got there through high u, so η and u are negatively correlated *within*
   the cell. The dummy also mixes "this record means a strong player" with "this record means strong
   opponents".
3. **The leaky bucket.** It soaks up part of the deck's own effect on this run's results.

The random effect integrates u out instead of conditioning on its proxies. The structural pool ties the opponent
term to the same strength distribution that produced the records. Together they remove mechanisms 1-2 and replace
3 with an outcome-free u.

**Synthetic evidence** (`drafts/mm_likelihood_selftest.txt`, `mm_likelihood_large_s.txt`). The setup is 24k drafts,
40 cards, 8 archetypes and full 7-3 runs against the recursion pool; the fit uses 15k train, 3k val and 6k test
drafts. "Scale" is cov(score, η)/var(η), with 1 correct and < 1 compressed; the reported slope is about
1/scale.

| Fit | s = 0.39: scale | Spearman(WR) | s = 0.67, d = 0.3: scale | Spearman(WR) |
|---|---|---|---|---|
| iid games, no record terms | 0.89 | 0.3065 | **0.78** | 0.3478 |
| iid + free record dummies | 0.93 | 0.3057 | 0.82 | 0.3476 |
| iid + free dummies + skill bucket (round 4 practice) | 0.95 | 0.3050 | 0.92 | 0.3464 |
| RE + free record dummies | 0.99 | 0.3070 | | |
| RE + monotone parametric pool | 1.04 (t = 0.42 vs 0.25 true) | 0.3056 | | |
| RE + structural, s learned | 1.04 (s = 0.47 vs 0.39) | 0.3059 | | |
| RE + structural, s tied to users' spread | 1.13 (s = 0.57) | 0.3070 | | |
| **RE + structural, s fixed** | **1.02** | 0.3067 | **1.03** | 0.3469 |

Skill-confounded scenario, s = 0.39. Here skilled players favour an *off-value* card direction (r = 0.6), so
absorbing skill changes ranking:

| Fit | scale | r(score, true deck) | Spearman(WR) |
|---|---|---|---|
| iid + free dummies + bucket (round 4) | 0.99 | **0.948** | 0.248 |
| RE + parametric + bucket (A "rated") | 0.98 | 0.948 | 0.248 |
| iid, no record | 0.93 | 0.901 | **0.328** |
| RE + structural, s fixed (A "record") | 1.05 | 0.903 | 0.327 |

What this says:
- **Decompression is real and sized right.** At s ≈ 0.67, matchmaking alone gives an iid slope of about 1.28,
  inside the observed 1.2-1.7. The fixed-s structural model removes it. Some of the real slope is probably also
  ordinary L2/early-stopping shrinkage, which A does not touch. Predict A's real stage-2 slope at 1.0-1.3, not
  1.0.
- **Decompression does not reorder decks.** Under matchmaking alone, compression is close to proportional,
  because the record depends on X only through η. Every model's r with the true deck is 0.993 ± 0.001, and
  Spearman agrees to within ±0.002. Any real Spearman gain from A has to come from mechanisms the simulation
  leaves out:
  - (a) the RE no longer counts a trophy run as 9 independent games, which reweights the cards concentrated in
    deep runs;
  - (b) the opponent pool depends on rank × record, so compression differs by rank. That is untested;
  - (c) bucket handling, which is a ranking lever but trades record prediction against rating fidelity.
- **The parametric pool overshoots t.** It fits 0.42 against 0.25 true, and its 6-0 vs 0-2 offset is +0.89
  against a structural +0.54 at s = 0.39. It sits on the same ridge as s/t. Use it only as a robustness check;
  the structural pool with s fixed is the primary.

**Better-ranked "deck ratings"** therefore means two things, and they must be kept apart (evaluation_design §1.1):
- A *record predictor*: the A-record variant may help or hurt the blend, roughly neutral in expectation.
- A *deck rating*: the A-rated variant should beat the round 4 FM on SC-GLL and the within-build design, and
  costs nothing on scale.

### A.3 Evaluation

Run per set: MSH, FDN and BLB to screen, then all 12. Fit on train and early-stop on the val draft-marginal NLL;
test on test. Then refit on train+val with recency τ = 2 (per-draft weights on L_i) for the blend.

| Metric | What it checks | Target / gate |
|---|---|---|
| Stage-2 slope b (`matchmaking_variance.fit_covariates` on the deck logit, same s) | decompression | b ∈ [0.9, 1.15] for A, vs 1.2-1.7 now |
| Trophy-rate calibration: P(7 wins) and E[W] by decile of predicted e, simulated with the structural pool vs observed | decompression in the units people use | top-decile trophy rate within ±1.5 pts (compressed models under-predict it) |
| Draft marginal NLL of test records under the A likelihood | proper score of records given rank | report; compare A variants only, since models without the likelihood can't be scored by it |
| SC-GLL (`eval_protocol.ctx_calibrated_game_loss`) on the deck-only score | deck rating quality (primary for A-rated) | `decide()` vs round 4 FM |
| Raw and stratified Spearman, neg-binomial NLL; within-build design (evaluation_design §2.3) | record prediction; leakage veto | secondary |
| Blend: champion LightGBM (refit, 5 seeds) ⊕ A, weight on val | does it earn a place | `decide()` on 12 sets; non-inferior (−0.001) is enough to use A as B's likelihood |
| Card ratings: r(w_A, w_v4), r(w_A, GIH WR), sd(w_A)/sd(w_v4), largest movers | sanity | sd ratio ≈ 1.2-1.4; movers should be cards concentrated in deep runs or skill-favoured archetypes |
| s-profile: every metric at s ∈ {0.4, 0.55, 0.7, 0.85} | identification | Spearman/SC-GLL flat across s; only the scale moves. If not, something is mis-specified |

**A grid** (CPU, 4 threads, scorer = FM rank 4 nonland + deck features + affinity):

| id | Likelihood | Purpose |
|---|---|---|
| A0 | iid + free record dummies + raw bucket (round 4, in the new code) | reproduction: val loss within 1e-4/game of `deck_synergy_v4` |
| A1 | iid, no record, no bucket | the pure-compression baseline |
| A2 | RE + free record dummies | collider without the pool |
| A3 | RE + parametric pool | robustness |
| A4a-d | RE + structural, s ∈ {0.4, 0.55, 0.7, 0.85}, λ = 1, match = record | primary; s profile |
| A4e, f | A4 at best s with λ = 0.8; with match = wins | matching rule sensitivity |
| A5 | best A4 + de-leaked bucket on the random-effect mean | A-rated |

That is 11 fits per set. Estimated 3-8 min each (the RE costs about 144 node evaluations per game, still small
next to the FM's sparse matmul), so about 1 h per set: 3 h for the screen, 12 h for all 12 sets. Data loading
must use `drafts/prepare_next_model.py`'s column-projected scans. Never the full game parquet.

---

## B. Game-level set transformer (report options #2/#3, adapted)

Code: `drafts/game_set_transformer.py` (`CardEncoder`, `PMA`, `DeckSetTransformer`, `Distill`, `SetData`,
`Trainer`, `selftest`, `timing`). Data and teachers: `drafts/prepare_next_model.py`.

### B.1 Architecture

```
tokens     distinct cards of the build (lands incl., ≤ 48) + [SET] token + [DECKFEAT] token (MLP of ~30 Scryfall deck features + affinity)
card h_c   W_in · LN(z_c) + δ_c + copies[n_c]
             z_c   PickNet all_d256 card vector (256-d), frozen buffer; source results/<SET>/picknet/feats.npz:card_vec
             δ_c   learned residual, init 0, L2 1e-4  (ablation "random-init": δ_c ~ N(0, 0.02), no z)
             slots one per (set, card column), ~4,000 over 12 sets; reprints and basics are separate slots
trunk      L × pre-norm self-attention (d, 4 heads, FF 2d, dropout 0.1), shared across sets
pool       PMA, k = 4 seeds → 4d
deck η_b   Σ_c n_c·w_c  +  f_s·F_b  +  head_s(pool)        w: per-slot power (= the linear model), head_s: per-set readout, init 0
likelihood candidate A per set: MMGameModel(set) with d_s, β_s, ρ_s[rank], t_s and that set's pool (s_s fixed from A's profile)
distill    ẑ_i = a_s·η_main(i) + c_s[rank_i] + b_s;   α · mean((ẑ − teacher_i)²) / var_s(teacher)
loss       Σ_i r_i·(−L_i) / Σ games  +  α·distill  +  L2(w, δ)            r_i = recency weight exp(−age/τ) in refits
```

Design choices, each tied to a finding:
- **Linear skip + zero-init heads.** At step 0 the model *is* A's linear game model, and the trunk learns only the
  residual. The per-set results show the transformer's whole problem is relearning card power from 5-9 games
  per deck, and the skip removes that.
- **Frozen PickNet base + residual.** Pick-trained vectors give a population-valuation prior for rare cards and
  small sets. The residual lets games overrule it. Freezing keeps the 8.4M-parameter PickNet out of the
  optimiser and out of memory; only the 256 → d projection is trained, at 0.3 × lr.
- **Cross-set trunk, per-set heads.** The trunk shares "what makes a 40-card deck work" (curve, colours, synergy
  shape) across about 1M training drafts. Card power, deck-feature weights, the readout, the rank offsets and
  the opponent pool stay per set. Batches are single-set: the set is sampled ∝ n^0.75, so the pool recursion
  runs once per step and per-set parameters need no gather.
- **Distillation teacher.** The teacher is the round 10 LightGBM + game-FM blend logit, cross-fitted: 5-fold
  LightGBM over train drafts for screening, or over train+val for refits, plus round 4's 2-fold game-FM
  cross-fit (`prepare_next_model.oof_teacher`). The affine map ẑ absorbs the teacher's compressed, rank-inclusive
  scale. The teacher carries the record/skill-absorbing view that A's likelihood deliberately does not, which
  makes α the knob between "deck rating" and "record predictor".
- **Leak checks.**
  - Screening runs early-stop on val, so they use `teacher_train`. A teacher trained on val outcomes would leak
    them into the targets.
  - Test drafts never get a teacher.
  - PickNet was trained on the same chronological 70/15/15 split, so its card vectors have seen no val/test picks.
    Assert the alignment (pipeline_audit #11).

### B.2 Sizes, memory, time (16 GB Mac, MPS, RAM < 3 GB)

Measured with `game_set_transformer.py time D L B`: one forward+backward step of the deck network on CPU, 2
threads, about 4,000 card slots, 30 tokens per deck, with the machine at load 10. MPS was not benchmarked, since
the GPU belongs to the training queue. The MPS figures are an estimate of 3-5× faster than 2-thread CPU at these
sizes.

| config | trainable params | CPU ms/step (meas.) | peak RSS of that step (meas.) | MPS ms/step (est.) | use |
|---|---|---|---|---|---|
| d=64, L=2, B=256 | 0.37M | 81 | 474 MB | 20-30 | size ablation |
| **d=128, L=3, B=256** | **1.05M** | **176** | **661 MB** | **40-60** | **default** |
| d=128, L=3, B=512 | 1.05M | 347 | 987 MB | 70-110 | if the GPU is free |
| d=192, L=4, B=256 | 2.22M | 322 | 927 MB | 70-100 | size ablation |

RAM budget at the default:

| item | size |
|---|---|
| Data, 12 sets: builds (~1.4M × 48 tokens × 3 B) | ~200 MB |
| Data: build features (fp16) | ~85 MB |
| Data: games ([~1.4M drafts × 9] × ~24 B) | ~300 MB |
| Data: teachers, card vectors | < 20 MB |
| torch, model, AdamW, activations (measured above) | 0.6-1.0 GB |
| **Total** | **≈ 1.3-1.7 GB** |

Save the arrays as uncompressed `.npy` and open them with `mmap_mode="r"` to stay near the low end.
**Stay at B ≤ 512; B = 256 is the default.** The likelihood adds only B × 9 × 12 × 12 floats per step.

Time, with about 0.95M training drafts over 12 sets:

| Run | Steps/epoch | Time/epoch | Epochs | Wall-clock |
|---|---|---|---|---|
| Cross-set, B = 256 | ≈ 3,700 | 2.5-4 min at 40-60 ms | 12-20 (early stop on pooled val marginal NLL, 2 lr drops) | **≈ 1-1.5 h** on a free GPU, 2-2.5 h shared |
| Per-set, MSH (~46k train drafts) | ≈ 180 | ≈ 10 s | ~30 | ≈ 5 min |
| Per-set, FDN/BLB | | | | ≈ 10-15 min |

Data prep (all 12 sets) is about 1 h CPU. Teachers (5 folds × 2 scopes of LightGBM) are about 2 h CPU.

### B.3 Experiment and ablation grid

All B runs use A's structural likelihood (s fixed at A's chosen value) unless the row says otherwise. Defaults:
d = 128, L = 3, B = 256, AdamW lr 1e-3, wd 0.01, 2 lr drops, and seed 0 in phases B1-B2.

**Phase B1: per-set screen on MSH, FDN, BLB** (single-set trunk; 8 configs × 3 sets ≈ 3-5 GPU-h)

| id | init | α (distill) | likelihood | other | question |
|---|---|---|---|---|---|
| B1.1 | random | 0 | A | | baseline transformer on the new likelihood |
| B1.2 | PickNet | 0 | A | | **does PickNet init help?** (vs B1.1) |
| B1.3 | PickNet | 0.25 | A | | distillation, light |
| B1.4 | PickNet | 1.0 | A | | distillation, heavy (Das & Sanghavi: the optimum can exceed the label weight) |
| B1.5 | PickNet | best of B1.2-4 | round 4 iid + free dummies + bucket | | **does the likelihood matter for B?** |
| B1.6 | PickNet | best | A | no linear skip | is the skip path doing the work? |
| B1.7 | PickNet | best | A | d = 64, L = 2 | size |
| B1.8 | PickNet | best | A | d = 192, L = 4 | size |

**Phase B2: cross-set trunk** (12 sets; 3 runs ≈ 4-7 GPU-h)

| id | change from best B1 | question |
|---|---|---|
| B2.1 | cross-set trunk, per-set heads | does pooling the trunk help within set? |
| B2.2 | B2.1 with random init | does pooling need PickNet? |
| B2.3 | B2.1 + 1 epoch per-set fine-tune of heads, w, δ only (trunk frozen) | adaptor variant |

**Phase B3: confirm** (best of B1/B2; 3 seeds, train-only fits for weights plus refit on train+val with recency τ = 2; 12 sets)

- If the best is cross-set: 3 seeds × 2 = 6 runs ≈ 8-15 GPU-h.
- If the best is per-set: 6 × 12 sets × 5-15 min ≈ 10 GPU-h.
- Optional: BatchEnsemble k = 4 in place of seeds, at ~1.5× the cost of one run (report option #2).
- Then stack. Take the 3-way logit blend of champion LightGBM (refit, 5 seeds), A-record or the round 4 FM, and
  B. Pick the weights on val drafts from the train-only twins (simplex grid, step 0.05). Apply them to the refit
  members on test.

**Decision rules**, all through `eval_protocol.decide()` (SC-GLL primary, Spearman secondary, Holm across the
rows of a phase):

| Stage | Rule |
|---|---|
| B1 → B2 | Go if the best B1 config's val blend gain is ≥ +0.001 Spearman on the 3-set mean, **or** B1.2 − B1.1 ≥ +0.001 (PickNet matters, so pooling might too). Otherwise skip B2 and run B3 on the best per-set config only |
| Kill | B alone is > 0.01 below the round 4 FM on val (bug); student-teacher logit r > 0.97 at the chosen α (collapsed onto the teacher, so lower α); seed sd > half the gain being tested |
| Adopt B into the blend | The 12-set random-effects mean gain has HKSJ p < 0.05, ≥ 8/12 sets positive, no set significantly worse, point estimate ≥ +0.001 Spearman (or ≥ +0.02 × 1000 SC-GLL), and the leakage veto passes |
| Stop the direction | 90% CI inside ±0.001 (TOST), stated as "no meaningful gain" |

Total GPU budget: B1 3-5 h + B2 4-7 h + B3 8-15 h, plus repeats ≈ **20-35 h of free GPU**, or about 35-55 h
shared with the queue. B1 alone (≈ 5 h) answers the two questions that matter most: PickNet init, and
likelihood × distillation.

### B.4 What the synthetic B self-test does and does not show

`drafts/game_set_transformer_selftest.txt`, 2 synthetic sets, 4k drafts each, d = 32, 90 steps:
- Output is invariant to permuting tokens and to extra padding (max |Δη| = 0).
- The build-table data path (one score per build, gathered per game) gives exactly the same loss as the
  one-build-per-draft path (1.246815 both).
- Val NLL falls under all three configs. Random-init vs PickNet-init are within noise on synthetic data (val
  0.7036 vs 0.7055), which is expected: the synthetic "PickNet" vectors are a noisy projection of the true
  values.
- Distillation lifts r(true deck) to 0.98-0.99 only because the synthetic teacher is nearly the oracle
  (η + 0.5u + noise). **This shows the plumbing works, not that distillation helps.**

---

## C. Shared protocol and risks

- **Data hygiene.**
  - Column-projected lazy scans only (`prepare_next_model.build_tables`): builds via `unique(draft_id,
    build_index)` with deck_ columns, and games with about 10 narrow columns.
  - Record state = `record_before`, with state 21 when the history is incomplete.
  - The audit's stale-record finding (#2) affects the draft-level target, not the game likelihood. A and B train
    on games, which is one more reason to judge them on SC-GLL.
- **Same treatment as the champion.** Refit on train+val with recency τ = 2 and trees/epochs scaled by data size.
  Blend weights are picked on val from train-only twins.
- **Seeds.** At least 3 for B, with seed variance in the per-set SE (`eval_protocol.SetResult`).
- **The main risks.**
  - (1) The A likelihood is mis-specified (Arena matches on wins plus hidden MMR, not exact record). The λ and
    match = wins rows and the s profile bound this. Ranking should be robust to it; scale is not.
  - (2) B collapses onto its teacher, or onto the linear skip. Caught by student-teacher r and the B1.6
    ablation.
  - (3) The CPU→MPS timing estimate is off by 2×. That changes the schedule, not the design.
- **What would change the plan.**
  - If A's s-profile shows Spearman or SC-GLL moving with s, the pool is mis-specified. Fix it before B.
  - If B1.5 (B on the old likelihood) ties B1.2-4, the likelihood does not matter for ranking, consistent with
    §A.2. B can then be run on whichever likelihood is cheaper (iid + free dummies is about 10× cheaper per step
    than RE + structural).

## Files

- `results/methodology/drafts/mm_likelihood.py`: candidate A (torch). The opponent-pool recursion is
  differentiable in s and λ, with record- and wins-matching. The module covers the RE marginal likelihood, the
  structural, parametric, free and none pools, the training loop, the simulator and the self-test. Output is in
  `mm_likelihood_selftest.txt` (15 s, 0.44 GB) and `mm_likelihood_large_s.txt`.
- `results/methodology/drafts/game_set_transformer.py`: candidate B model, trainer (synthetic and build-table
  data paths), self-test and timing. Output is in `game_set_transformer_selftest.txt` (9 s, 0.48 GB). The timing
  rows ran as separate processes at 0.47-0.99 GB peak, which is above the 500 MB snippet guideline for a few
  seconds each.
- `results/methodology/drafts/prepare_next_model.py`: per-set data prep (builds, padded games, record states,
  deck features, PickNet vectors) and the cross-fitted teachers (train / train+val scopes). Syntax-checked only.
  Not run, because it reads game files.
