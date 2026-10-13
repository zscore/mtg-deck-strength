# Skill-Adjusted Deck Strength from 17Lands Data

Oct 13, 2026 · Zane Blanton

Editable version with the chart: https://claude.ai/code/artifact/f4a9e805-1709-4a75-9b82-ef2d3bdf1176

## Abstract

We estimate how strong a drafted Magic: The Gathering limited deck is, separately from who plays it, using 17Lands game data from 12 PremierDraft sets (about 1.6M drafts, 9.2M games). The best deck rating combines a skill-adjusted game-level factorization machine with a LightGBM model on deck card counts. On 6 sets never used for development it beats the factorization machine alone by 0.233 × 10⁻³ nats per game on all 6 sets (one-sided p = 0.0001), under a pipeline frozen and hashed before those sets were opened.

Three findings shape the method. First, the usual metric (Spearman correlation of predicted win rate with the event record) rewards models for learning *who* drafted: removing the skill control raises Spearman by 0.010 while making held-out game predictions worse by 0.37 × 10⁻³ nats. We replace it with a skill-controlled held-out game log loss. Second, Arena pairs players by record and rank, which compresses records toward 50%; a play/draw coin-flip instrument puts the true strength spread at about 0.36 logits. Third, a randomized "opening-hand value" for each card gives skill-free card ratings that replicate across games (r = 0.78) far better than 17Lands' GIH WR does.

Many ideas did not help. Transformers (including one trained only on what the factorization machine misses), PickNet card embeddings, a matchmaking-aware likelihood and opponent-colour controls all gave no gain. We report these in full.

## 1 Introduction

A deck's strength is the win probability it would give an average player against an average opponent. That is not the same as predicting a player's record: a record also reflects the player's skill, their luck in a handful of Bo1 games, and who the matchmaker paired them with. This paper is about the first quantity, and shows that confusing the two leads to wrong model choices.

The starting point was a deck model that predicted records well. A LightGBM on registered-deck card counts reached 0.271 mean Spearman over 12 sets, and blending it with a game-level factorization machine (FM) reached 0.279. That beats the published DraftEncoder (about 0.21–0.23 within a set) and the 17Lands-style heuristic (top-23 mean GIH WR, 0.18). But Spearman against records cannot tell a model that rates decks well from one that recognises good players.

Contributions:

1. **An evaluation protocol for deck strength** (Section 3): a skill-controlled held-out game log loss (SC-GLL), random-effects meta-analysis over sets, pre-registered decision rules, and a frozen confirmation on held-out sets ("the vault").
2. **A deck rating confirmed out of sample**: blend\*, the FM averaged with counts-only LightGBM at a weight tuned on SC-GLL (Sections 4, 5, 9).
3. **Evidence that matchmaking compresses records** and an instrument that measures the true strength spread anyway (Section 6).
4. **Randomized opening-hand card values**, a skill-free card rating (Section 7).
5. **A set of careful negative results**: transformers, PickNet embeddings, structural likelihoods, opponent context (Section 10).

## 2 Data and setting

We use 17Lands public PremierDraft data: best-of-one events that end at 7 wins or 3 losses. Each draft has a pick sequence, a final pool and an event record (W, L). Each game row has the registered deck (card counts), the opening hand, the cards drawn, play/draw, both players' mulligans, the user's rank, and two per-user buckets: `user_game_win_rate_bucket` (long-run win rate) and `user_n_games_bucket` (games logged).

**Sets.** Twelve development sets (MSH, BLB, DFT, DSK, EOE, FDN, FIN, LCI, MKM, OTJ, SOS, TDM) and six vault sets touched only once at the end (ECL, TLA, WOE, MOM, HOB, TMT). NEO and ONE were excluded for missing columns.

**Splits.** Within each set, drafts are sorted by time: the first 70% train, the next 15% validation, the last 15% test. Games inherit their draft's split. A draft's deck is its most-played build; about 19% of drafts register more than one build.

**Likelihood of a record.** The run stops on results so far, so the likelihood of a record is just p^W (1 − p)^L and abandoned runs still count. That is only true if games are independent given p, which Section 6 shows is false.

**Data quirks found.** Each changed a conclusion, so we list them.

| quirk | consequence |
| --- | --- |
| `opp_rank` is "None" in every game file | The "opponent rank" control was a constant |
| `opp_colors` are the colours the opponent revealed during the game | Post-treatment: 1 colour seen → 74% user win rate, 5 colours → 47%; game length rises 3.9 → 10.9 turns with colours seen. Unusable as a control |
| The draft-file record is stale for 16–24% of drafts | Records rebuilt from game files; raises every Spearman by about 0.013 |
| Five sets' game files miss games for about 28% of runs (MSH, EOE, FIN, SOS, TDM) | MSH, the smallest set, is a poor screening set |
| `user_game_win_rate_bucket` includes the current run | Leaky as a feature; usable only as a control, ideally de-leaked |
| Experience buckets 500 and 1000 were coded as unknown | Fixed in code |

## 3 Evaluation

The deciding metric is a skill-controlled held-out game log loss, not Spearman against records, because Spearman rewards learning who the drafter is.

### 3.1 The problem with Spearman

A model that sees a deck can partly infer the player: strong drafters take and play recognisable cards. On records, that inference looks like deck quality. Two measurements show the size of the effect:

- In a synthetic test where a model learns only "who drafted" (its true deck accuracy is lower), raw Spearman rises by 0.050 (z = 25).
- On real data, removing the skill-bucket control from the game FM raises draft Spearman by 0.0098 on 12/12 sets, but makes held-out game predictions worse by 0.37 × 10⁻³ nats per game, with 0/12 sets better.

### 3.2 Skill-controlled game log loss (SC-GLL)

A context model is fit once on training games and frozen; every candidate is judged with the same one. Its inputs are play/draw, mulligans, experience, rank, the record before the game, and the skill bucket. Each candidate supplies only a deck score s(d), computed at a fixed reference rank (platinum). Two calibration numbers are fit on validation games:

```latex
\operatorname{logit} P(y_g = 1) = c_g + a + b \, s(d_g)
```

where c_g is the frozen context logit of game g and d_g its deck. The score is the mean test-game log likelihood, × 1000, on experienced users (at least 50 games logged, about 82–85% of games). A candidate is better if its paired difference from a reference is positive.

### 3.3 Three judge contexts

The context model can give a home advantage to candidates that use the same controls. We therefore score everything under three contexts:

| context | skill term | mulligan term |
| --- | --- | --- |
| raw | raw bucket (includes this run's games) | own − opponent |
| dl | de-leaked bucket (this run's games removed) | own − opponent |
| pre | de-leaked bucket | opponent only (own mulligans are partly caused by the deck) |

Moving from raw to dl shifts paired differences by about 0.18–0.24. Conclusions that flip between contexts are reported as unresolved. The primary context is dl.

### 3.4 Aggregation and decision rule

Each set gives a paired difference with a bootstrap standard error over test drafts in day blocks, plus a seed component when several seeds exist. Sets are pooled by DerSimonian–Laird random effects with the Hartung–Knapp–Sidik–Jonkman interval. A change is adopted only if:

1. the pooled mean is above 0 with p < 0.05 (Holm-corrected when several arms are tested),
2. at least 8 of 12 sets improve,
3. no set is significantly worse, and
4. the mean is at least 0.02 × 10⁻³ nats per game.

With 12 sets this detects about 0.03–0.05 × 10⁻³ nats (or 0.0011–0.0017 Spearman) at 80% power. A single set cannot detect gains under about 0.003–0.005 Spearman.

### 3.5 Within-build referee

About 19% of drafts register more than one build, so the same player in the same draft plays two decks. A conditional logit with a draft fixed effect regresses game outcomes on the within-draft difference in deck score. Player skill is constant within a draft, so a model that only recognises players gains nothing here. We compare models on the slope per within-draft standard deviation. The test is unbiased but weak (about 1,800 qualifying test drafts in MSH).

### 3.6 The vault

The test split was reused over about ten rounds of development, so it counts as development data. Before opening the six vault sets we wrote the whole pipeline and five one-sided hypotheses to a JSON file, hashed it together with every script it runs, and made the vault runner refuse to start if any hash changed.

## 4 Models

The recommended deck rating, blend\*, averages two models that see the deck differently: a game-level factorization machine with skill controls, and a draft-level LightGBM on card counts.

### 4.1 Game-level factorization machine (game FM)

Each game is one logistic observation. The log-odds of a win is a deck score plus a context term:

```latex
\operatorname{logit} P(\text{win}) = \underbrace{\sum_c w_c x_c + \sum_{i<j} \langle v_i, v_j \rangle x_i x_j + \beta^\top F(x) + \gamma \, \mathrm{aff}(x)}_{\text{deck score } s(x)} + \underbrace{\delta^\top z_g}_{\text{context}}
```

- x: the deck's card counts; w_c: each card's power.
- v_i: rank-4 synergy vectors, nonland cards only, so the pairs cannot learn the mana base.
- F(x): 17 deck features from Scryfall data (land count, curve shape, colours and splash, coloured-source shortfall).
- aff(x): the deck's mean pairwise affinity under a pick model, which measures how often drafters take cards together. It is learned from picks alone, never from results.
- z_g: play/draw, mulligan difference, the skill bucket, experience, rank, and the record before the game.

The deck score excludes z_g, so it rates the deck with the player and situation held fixed. Fits use Adam with L2 penalties, early stopping on validation games and two learning-rate drops. The "champion v2" is the same model refit on train and validation games with the chosen number of epochs.

### 4.2 Draft-level LightGBM

LightGBM with a binomial objective on each draft's (W, L): a positive row weighted W and a negative row weighted L. Features are the registered deck's card counts and the player's rank; earlier variants added structure features, deck features and affinity. For deck rating it is evaluated at a fixed reference rank (platinum). The "counts-only" variant drops everything but card counts.

### 4.3 blend\*

A logit average of the two deck scores, with a per-set weight α chosen on validation SC-GLL-dl using a two-half cross-fit:

```latex
z_{\text{blend}} = (1-\alpha) \, z_{\text{FM}} + \alpha \, z_{\text{LGBM}}
```

The earlier "round 4 blend" was the same average with α chosen on validation record loss. That choice turns out to matter (Section 5). The chosen α is at least 0.8 in only 2 of 12 sets, so LightGBM carries real weight.

### 4.4 Record-prediction pipeline (round 10)

For predicting a player's record rather than rating a deck: LightGBM on counts, structure, rank, deck features and affinity, refit on train and validation drafts with recency weights exp(−age/2), averaged over 5 seeds, then logit-averaged with the game FM.

### 4.5 Models that did not help

- **Model A, matchmaking-aware likelihood.** Each game is won with probability Σ_j π_s(state, j) · expit(e − s x_j). Here π_s is the strength distribution of opponents at the player's current record, derived by recursion from a fixed population spread s, and e is the player's strength (a per-draft random effect).
- **Model B, game-level set transformer.** The game FM plus a transformer residual over the deck's cards. Card tokens are frozen PickNet vectors plus a learned residual, pooled by attention and a copy-weighted sum. Variants: PickNet or random initialisation, with or without distillation toward the blend, and FM frozen or jointly trained.

## 5 Results on the development sets

On the 12 development sets, blend\* is the only change that beats the game FM by a large margin under every judge context. All numbers are random-effects means over 12 sets, in 10⁻³ nats per test game; positive is better.

### 5.1 Deck-rating scoreboard

Every deck rater is scored against one shared baseline: the linear game model, which is the game FM without its synergy pairs. Each number is how much better than that baseline the rater predicts held-out games, so higher is a better rater. Rows are sorted best first by the primary (dl) judge.

| deck rater | dl (primary) [95% CI] | beats baseline (dl) | raw | pre |
| --- | --- | --- | --- | --- |
| **blend\* (FM + LightGBM, α on SC-GLL)** | **+0.27 [+0.22, +0.33]** | **12/12** | +0.18 | +0.30 |
| game FM, refit on train+val | +0.18 [+0.09, +0.28] | 9/12 | +0.20 | +0.17 |
| game FM | +0.16 [+0.10, +0.23] | 11/12 | +0.16 | +0.16 |
| round 4 blend (α on record loss) | +0.14 [+0.06, +0.23] | 10/12 | −0.06 | +0.22 |
| linear game model (baseline) | 0 | — | 0 | 0 |
| LightGBM alone | −0.25 [−0.37, −0.13] | 1/12 | −0.51 | −0.15 |

blend\* also leads under pre. Under raw, whose skill term is the same leaky bucket the FM was trained on, the refit FM edges it.

Three things follow. LightGBM alone is the only rater worse than the baseline, so its Spearman edge was record prediction, not deck rating. The synergy pairs help: the game FM is +0.16 above the linear model. And blend\* is clearly best: paired directly against the game FM it gains +0.13 [+0.10, +0.17] on 11/12 sets, while the round 4 blend, the same two models weighted on record loss, only ties the FM (−0.01). On first games only (no record-based matchmaking yet), blend\* beats the FM by +0.18 under pre.

### 5.2 Which controls to keep

Spearman and SC-GLL disagree about the controls, and SC-GLL is the one that can tell deck strength from skill.

| removed from the game FM | draft Spearman | SC-GLL raw | SC-GLL dl | SC-GLL pre | verdict |
| --- | --- | --- | --- | --- | --- |
| skill bucket | +0.0098 (12/12) | −0.37 | −0.13 | −0.07 | keep: the Spearman gain is leakage |
| both mulligan terms | +0.0022 (12/12) | −0.06 | −0.04 | +0.03 | judge-dependent |
| own mulligans only | — | — | −0.02 | +0.05 (11/12) | dev-adopted; not vault-tested |
| record before the game | ±0.000 | — | — | — | keep: deck scores compress without it |
| raw bucket → de-leaked bucket | — | −0.22 | −0.04 | +0.01 | tie |
| recency weighting (added) | — | −0.002 | −0.001 | — | no effect |

The record control is not a bad control: it absorbs opponent strength under record-based matchmaking. Own mulligans are partly caused by a bad mana base, which is why dropping them helps under the pre context and hurts under contexts that still contain them.

### 5.3 Record prediction

For predicting a player's record, the round 10 pipeline beats the round 4 blend by +0.0023 Spearman [+0.0014, +0.0033] with draft-file records and +0.0028 [+0.0020, +0.0036] with game-file records, on 12/12 sets both times. Its mean test Spearman on game-file records is 0.295. Earlier reference points on draft-file records: linear deck model 0.244, LightGBM 0.271, round 4 blend 0.279. Training on rebuilt game-file records instead of draft-file records helps only slightly (+0.0006, 10/12).

## 6 Matchmaking compresses records

Arena pairs players by record within a run and roughly by rank across runs, so records understate how different decks and players really are. A coin-flip instrument still recovers the true spread: about 0.36 logits.

### 6.1 Records are less spread out than coin flips

Fit a random-effects model in which each draft has its own true win rate and its record is binomial given that rate. The likelihood is maximised at zero spread. Records are *less* dispersed than if every draft had the same true win rate. That is what record-based pairing does: a 5–0 deck meets other 5–0 decks, so its per-game win chance falls as it wins.

A simulation with exact record matching makes the point. Whatever the true spread s of player-and-deck strength, from 0.35 to 1.0 logits, the standard deviation of observed win rates stays at 0.23. Records alone therefore cannot identify s. What records can identify is a model's *slope*: how strongly its predictions track outcomes.

**Figure 1 (see the online version for the chart).** Simulated Spearman of the true win probability against records, under exact record matching (20,000 runs per spread):

| true strength spread s (logit sd) | perfect-model Spearman |
| --- | --- |
| 0.35 | 0.34 |
| 0.50 | 0.46 |
| 0.65 | 0.54 |
| 0.80 | 0.60 |
| 1.00 | 0.67 |

The play/draw estimate of s is 0.23–0.44 (centre 0.36); the round 4 blend's observed Spearman on MSH is 0.288.

The records barely change with s, but a perfect model's Spearman does. At the measured spread a perfect model would reach about 0.34, so the blend's 0.29 on MSH leaves less headroom than the old ceiling suggested and more than zero.

### 6.2 A play/draw instrument

Being on the play in Bo1 is a coin flip (50.0% at every match number), and it raises the chance of winning that game. Under record matching, that extra win buys a stronger opponent in the next game. The size of this knock-on effect depends on s.

We estimate it as a Wald ratio: the effect of being on the play in game t−1 on winning game t, divided by its effect on winning game t−1. Pooled over 17 sets, the ratio is −0.0278 ± 0.0063, and the sets agree (Q = 17.2 on 16 degrees of freedom). Mapped through the matchmaking recursion this gives **s ≈ 0.36 logits, 95% interval 0.23–0.44**, about ±9 win-rate points. The estimate assumes exact record matching; looser matching would bias it downward.

Two other observations support the picture. Opponents' strength rises with the user's wins before the game, although that check uses revealed opponent colours and is weaker evidence. And a structural model with s fixed (model A) gives deck scores 12–17% more spread out than the game FM's. The FM's scores are compressed by about 15%, and the ranking is unchanged.

### 6.3 Rank indexes the opponent pool

Game win rate falls with rank while the players' long-run win rate stays flat:

| rank | game win rate (MSH) | long-run win-rate bucket (MSH) |
| --- | --- | --- |
| bronze | 0.622 | 0.563 |
| silver | 0.597 | 0.556 |
| gold | 0.567 | 0.543 |
| platinum | 0.551 | 0.538 |
| diamond | 0.544 | 0.550 |
| mythic | 0.557 | 0.557 |

17Lands users beat the bronze queue and then meet stronger opponents as they climb. The drop is about 0.3–0.35 logits of opponent strength from bronze to diamond. So rank is mostly an opponent-pool offset, which is why rank alone predicts records poorly (Spearman 0.057), and why deck scores are evaluated at a fixed reference rank.

### 6.4 Consequences

- The old Spearman "ceiling", simulated as if games were independent, was below the scores models actually reach. It is retired.
- At s ≈ 0.36 a perfect model would reach a Spearman of about 0.34 against records; with looser matching the true s is higher and so is the ceiling.
- Training on records under-states deck differences. Reading FM scores as win-rate differences against an average opponent needs a factor of about 1.15.

## 7 Randomized opening-hand card values

The shuffle decides which cards start in the opening hand, so comparing games with and without a card in hand gives card ratings that player skill cannot contaminate. They replicate well but do not make a better deck score.

### 7.1 Design

17Lands records the kept seven cards before any are bottomed, mulligans included. Given the deck, the mulligan count and the number of lands L in the seven, which spells are in hand is a uniform draw. So the probability that card c is in hand is known exactly:

```latex
P(c \in \text{hand} \mid \text{deck}, k, L) = \frac{(7-L) \, n_c}{S}
```

where n_c is the copies of c and S the deck's spell count. We regress each game's outcome residual on (in hand − that probability), Robinson-style. The estimator is unbiased whatever the outcome model is, so skill and deck quality cannot leak in. Ratings are shrunk by empirical Bayes, and lands are rated the same way within the land slots.

The design fails only if players keep or mulligan based on *which* spells they hold at a given land count. A per-card balance index flags this on 12/12 sets (correlation with hand value +0.18 to +0.59). But re-estimating on hands that are almost always kept (no mulligan, 3 or 4 lands) agrees at r = 0.96–0.99.

### 7.2 Results

Hand values (HV) estimated on training games predict hand effects on test games better than any 17Lands statistic, with far less skill tilt. Skill tilt is how much of a rating, beyond its hand value, tracks the skill of the drafters who play the card.

| card rating | r with test-game hand values | skill tilt |
| --- | --- | --- |
| **HV (train games, shrunk)** | **0.78** | **0.22** |
| OH WR | 0.73 | 0.61 |
| GIH WR | 0.62 | 0.68 |
| IWD | 0.59 | 0.39 |
| GP WR | 0.51 | 0.72 |

Means over 12 sets. The target is itself a hand value, so this comparison favours HV; read it as replication, not as proof of superiority.

As a deck score, the sum of a deck's hand values only ties the sum of GIH WR (−0.15 × 10⁻³ nats, 5/12 sets, not significant). Both lose badly to the deck model's own card powers (+1.45 over GIH, 12/12). Using HV as a prior for the FM's card powers gives nothing (−0.001).

### 7.3 Early versus late

Regressing the FM's card powers on hand values gives slopes of 0.17–0.34 across sets. Under an additive model this implies a card drawn later in the game is worth roughly 30–50% of having it in the opening seven (set range 0.07–0.54).

## 8 What blend\* captures

LightGBM's contribution comes from non-additive use of the card counts. It does not come from structure, rank or player features, and no single FM variant reproduces it.

### 8.1 It is not skill

- **Within the same draft.** Comparing two builds by the same player, blend\* predicts outcomes about as well as the champion: +0.0015 [−0.0019, +0.0048] in slope per within-draft standard deviation. The leakage veto is not triggered (one-sided p = 0.83). The test is weak, so this says "not worse", not "better".
- **By experience.** The gain holds where the skill bucket is most precise, users with 500+ games: +0.18 ± 0.04.
- **Skill-marker partners.** Replacing LightGBM with a pure "skill marker" card score gives +0.003.

### 8.2 It is card counts, used non-additively

Regressing the gap between LightGBM's and the FM's deck logits on everything we can measure:

| feature group | partial R² |
| --- | --- |
| card counts | +0.34 |
| structure and curve | +0.001 |
| rank | 0 |
| player buckets | 0 |
| pool features | +0.001 |

Refitting LightGBM on feature subsets locates the gain the same way:

| LightGBM inside blend\* | blend − champion (dl) |
| --- | --- |
| full | +0.130 |
| without rank | +0.132 |
| counts only | +0.100 (12/12) |
| structure, deck features, affinity | +0.008 |
| structure only | −0.004 |

The gain is not in per-card values. An FM forced to LightGBM's per-card log-odds (r = 0.99) gains only +0.034. The best linear projection of LightGBM's scores, blended the same way, gains +0.004. A second seed of the champion as the partner gains 0.000, so it is not ordinary ensembling either.

### 8.3 Attempts to fold it into one model

| attempt | gain over champion (dl) |
| --- | --- |
| blend\* (same 3 sets) | +0.175 |
| card-power L2 tuned on SC-GLL | +0.004 |
| FM + auxiliary draft-record loss | +0.036 (10/12; best single model) |
| FM + per-copy indicators and land × spell-count terms | +0.013 |
| game-level LightGBM residual on the FM | +0.021 |
| draft-record LightGBM boosted from the FM | +0.011 |

The first, fourth, fifth and sixth rows use only MSH, FDN and BLB (the screen stopped there); the other two use all 12 sets. So the signal is neither game-level deck structure nor anything that any draft-record model picks up. It appears when a separately trained draft-record LightGBM is mixed in at a weight tuned on SC-GLL. One version remains untested: the boosted model with a cross-fitted starting score.

## 9 Vault confirmation

On six sets never used for development, four of the five frozen hypotheses hold on the vault alone, and blend\*'s gain is larger than on the development sets.

| hypothesis (frozen before the vault) | vault mean [95% CI] | one-sided p | sets better |
| --- | --- | --- | --- |
| H5: blend\* − champion v2 | **+0.233 [+0.172, +0.294]** | 0.0001 | 6/6 |
| H1: game FM − LightGBM alone | +0.264 [+0.113, +0.415] | 0.003 | 5/6 |
| H2: game FM − linear game model | +0.218 [−0.036, +0.472] | 0.039 | 5/6 |
| H3: refit on train+val − train-only champion | +0.005 [−0.058, +0.068] | 0.43 | 4/6 |
| H4: round 10 − round 4, record Spearman | +0.0040 [+0.0020, +0.0059] | 0.0016 | 6/6 |

H1–H3 and H5 are SC-GLL-dl differences in 10⁻³ nats per game; H4 is Spearman. The vault sets are ECL and TLA (with affinity) and WOE, MOM, HOB and TMT (no affinity model available; pre-declared fallback). Per-set H5 gains run from +0.218 (MOM) to +0.284 (WOE).

Two results need care. H2 passes the frozen rule (mean above 0, one-sided p < 0.05), but its two-sided interval crosses zero. H3 is labelled confirmed by the frozen rule only because it pools development and vault sets (+0.027, p = 0.023); the vault alone shows nothing. Read refitting on train+val as small and probably positive.

## 10 Negative results

Every architecture-level idea tried here tied or lost to the game FM, which matches prior work on drafting and on team-composition prediction.

| idea | result | why it fails, as far as we can tell |
| --- | --- | --- |
| PickNet features in LightGBM (deck card vectors, pick surprise, final draft state) | +0.0005 / +0.0001 / **−0.0017** Spearman over 12 sets; draft state significantly worse on 5 sets | Within a set, card identity already carries what the embeddings know |
| Pool or deck transformer, per set | 0.235–0.266 vs linear 0.244–0.268 on MSH | Ties linear models; logits correlate 0.86–0.93 |
| Game-level set transformer (4 configs × 3 sets × 2 seeds) | −0.019 to +0.000 | Residual never moved off zero |
| The same, with the FM frozen and a higher learning rate | −0.0009 to 0.0000 | No epoch beat the champion on validation: nothing learnable beyond the FM |
| Structural matchmaking likelihood (model A) | ranking unchanged (−0.011 to +0.013) | Fixes the scale (+15%), not the order |
| Opponent colour context and matchup matrix | −0.038 (1/12) | Colours are revealed during the game, so they encode game length |
| De-leaked skill bucket instead of the raw one | tie under the dl context | |
| Recency weighting of training games | −0.002 | |
| Hand values as a prior for card power | −0.001 | |
| LightGBM hyperparameter grid | no gain on MSH | |

Unless marked as Spearman, results are SC-GLL differences in 10⁻³ nats per game. One pattern runs through them. At 50k–200k drafts per set, the bag of card counts with skill controls is close to what the data can support. The gains came from *how models are judged and combined*, not from richer architectures.

## 11 Discussion and limitations

The practical lesson is that, for deck strength, the evaluation decides the model. The same two models, combined with a weight chosen on record loss, look worse than the FM. Chosen on a skill-controlled game loss, they beat it on every vault set.

**What to use.**

- To rate decks: blend\*.
- To read card values and synergies: the game FM, with its scores multiplied by about 1.15 to read them as win-rate differences against an average opponent.
- To predict a player's record: round 10.
- For skill-free card ratings: hand values.

**Limitations.**

- One seat only: we never see the opponent's deck, and opponent rank is missing.
- 17Lands users are above average. Results describe them, not all of Arena.
- The skill bucket is a coarse, partly leaky proxy. The within-build test that is immune to it is weak (about 2k qualifying drafts per set).
- The cards blend\* rates up lean toward cards that skilled drafters play. We found no direct evidence that this is skill leakage, but we cannot rule it out.
- The play/draw estimate of the strength spread assumes exact record matching; real matchmaking also uses rank and possibly hidden ratings.
- Two vault sets had the affinity model, so the affinity subgroup on its own is descriptive only.

**Future work.**

- Link a player's consecutive drafts by timestamp (about 15% of drafts link at about 80% precision) for a stronger within-player test.
- Find what LightGBM's non-additive count use is. A cross-fitted boosted model is the untested candidate.
- Test dropping only the own-mulligan control on new sets.
- Pool sets with shared card features once more replay data exists; same-set signal looks saturated.

## Appendix: reproducibility

Everything lives in the `mtg-deck-strength` repo; data comes from `mtg-data`, pick models from `picknet`. Nothing from this work is committed to git yet.

| what | where |
| --- | --- |
| Full experiment log, in order | `results/research_log.md` |
| Methodology memos and reviews | `results/methodology/` (evaluation_design, pipeline_audit, matchmaking_identification, next_model_design, meta_review_1, meta_review_2, blend_puzzle, fm_cardvalue, game_boost, hand_value_design) |
| Evaluation protocol (bootstrap, meta-analysis, decision rule) | `results/methodology/eval_protocol.py` |
| SC-GLL and judge contexts | `scripts/sc_gll.py`, `scripts/t_common.py`, `scripts/t1_rescore.py` |
| Game FM champion and variants | `scripts/game_fm_v2.py`, `scripts/t2_mull_split.py`, `scripts/fm_cardvalue.py` |
| LightGBM and record pipeline | `scripts/improve_deck_v2.py`, `scripts/rebuild_targets.py` |
| Within-build referee | `scripts/t3_within_build.py`, `scripts/t3b_blend.py` |
| Matchmaking | `scripts/matchmaking_variance.py`, `results/methodology/matchmaking_checks.py`, `scripts/structural_a.py` |
| Hand values | `scripts/hand_value_v2.py` |
| Transformers | `scripts/game_transformer.py` |
| Vault | `results/vault/frozen_pipeline.json` (sha256 87155c50…), `scripts/t5_vault.py`, `results/vault/report.json` |

All runs were serial, `nice`-d, at most 4 threads, on one 10-core, 16 GB Mac. Torch and LightGBM never shared a process (they clash over OpenMP on macOS).
