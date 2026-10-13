# Hand value: card ratings from the shuffle (design memo, 2026-10-12)

Meta-review 1, big bet 1 ("the shuffle is a randomised experiment"), turned into an estimator, a script
(`scripts/hand_value.py`) and an evaluation plan. No real game data was loaded. I read the schema and did two
row-limited checks on 20-40k MSH rows. The synthetic self-test passes.

**Bottom line.**
- Given the registered 40, which *spells* sit in the recorded opening 7 is decided by the shuffle. Conditional on
  the deck and on L (the number of lands among those 7), the spells are a uniform draw from the deck's spells.
- The propensity is therefore **known exactly**: E[h_c | deck, L] = (7 − L)·n_c / S.
- Robinson partialling-out with that known propensity gives a design-unbiased card rating, the **hand value (HV)**:
  the win-rate effect of c being in the opener instead of a random other spell of the same deck.
- Player skill, deck quality and archetype are constant within a deck, so they cannot enter HV, whatever outcome
  model is used.
- HV is the unconfounded target that GIH WR, IWD and the deck models' card powers can be judged against. It can
  also serve as a prior for card power.

---

## 1. What 17Lands records (checked on MSH, 20-40k rows)

| fact | evidence | consequence |
|---|---|---|
| `opening_hand_*` always sums to **7**, also after mulligans (0: 18,237 games, 1: 1,704, 2: 59, all = 7) | row sums | It is the **final 7 drawn, before the London-mulligan bottoming**. The k bottomed cards are inside the recorded 7, and which ones were bottomed is not recorded. |
| The land distribution of the recorded 7 is the same for 0 and 1 mulligans (mean basics 2.57 vs 2.60, none ≥ 6) | row sums | It is the *kept* hand, not the first mulliganed one. |
| hand + drawn > deck for some card: 3.8% / 6.8% / 10.9% of games at 0 / 1 / 2 mulligans | cell check | Consistent with bottomed cards that get drawn again (shuffle effects) and with recursion. `drawn_` is not clean either. |
| Decks of 40 cards: 94.5%; 41-44: about 5% | `deck_` sums | Use each deck's own N and S, never 40. |
| `tutored_*` is not in the slim parquet; `drawn_` minus turns ≈ +1.0 on the play, +1.4 on the draw | | "Seen" mixes in card draw (the round 8 confound). Only the opener is pre-treatment. |
| 17Lands' docs say only "cards in the opening hand (the one that was kept)"; the London-mulligan blog post says hand-level mulligan data was not tracked | web | The bottoming detail above is inferred from the data, not documented. |

---

## 2. The estimand

**Setup.** Take a game g with deck b: N cards, of which S_b are spells and N − S_b are lands.
- The shuffle draws the 7. The player keeps or mulligans, and the k = `num_mulligans` count is fixed *before* the
  final 7 is dealt, so it is pre-treatment.
- h_gc is the number of copies of c in the recorded 7, and L_g is the number of lands among them.
- **Randomisation assumption (A):** given the deck, k, play/draw and L, the multiset of the 7 − L spells is a
  uniform draw from the deck's spells.
- A holds exactly before the keep decision. The keep decision is mostly a function of L, which we condition on.
  A fails only insofar as players keep or mulligan **on spell identity at a fixed land count** (for example,
  keeping 2-landers with cheap spells). That failure is testable (§2.4).

**The swap estimand.** HV_c is the change in P(win) when c is in the opening hand rather than a random other spell
of the same deck. That other spell then sits in the library in c's place. Averaged over decks, it is weighted by
each deck's design variance.

**What HV measures under an additive model.** Write
logit or P(win) = α_deck,player + f(L) + Σ_c h_c β^H_c + Σ_c d_c β^D_c, with d = drawn later. Then the Robinson
coefficient (below) is exactly:

  HV_c = (β^H_c − β̄^H_{−c}) − q (β^D_c − β̄^D_{−c}),  q = P(a given library card is drawn) ≈ D̄ / (N − 7).

β̄_{−c} is the copy-weighted mean over the deck's other spells. The derivation uses the multivariate
hypergeometric covariances: Cov(Y, h_c) / Var(h_c) collapses to the contrast against the deck's other spells. That
makes the univariate per-card regression the right one: it *is* the swap contrast. A joint regression would only
identify β up to a per-deck constant anyway, because Σh = 7.

**Lands** are rated the same way, inside the land slots: a dual land in place of a random other land of the deck,
at a fixed L. The land-count curve f(L) is itself randomised given the deck, but the keep decision selects on it
heavily, so it is a nuisance here, not a rating.

**Copies.** h is a count. HV is per copy in hand. A second copy's marginal value can be added later as a separate
1{h ≥ 2} term with propensity from the hypergeometric pmf.

**Play/draw** is a coin flip, so HV_c^play and HV_c^draw are both identified. The script reports both and their
correlation.

**Mulligans.** Games with k ≥ 1 stay in. Their 7 is still a shuffle draw given the deck, but the k bottomed cards
dilute the effect. The outcome model stratifies by L × (k > 0), and a "core" estimate (k = 0, L ∈ {3, 4}, where
almost every 7 is kept) is reported as the robustness row.

### 2.1 Estimator (known-propensity Robinson / AIPW)

  r_gc = h_gc − (7 − L_g) n_bc / S_b,   e_g = y_g − p̂_g(deck, context, L × mulligan),

  HV_c^pp    = Σ_g r_gc e_g / Σ_g r_gc²           (percentage points; unbiased for **any** p̂ that does not use h)
  HV_c^logit = Σ_g r_gc e_g / Σ_g p̂(1 − p̂) r_gc²   (one-step logit, the scale of the deck models)

**What p̂ is.**
- p̂ comes from a logistic model with the deck's card counts, the context (play/draw, mulligans, record state,
  rank, experience, raw skill bucket) and the round 4 game FM's deck score where cached.
- It is cross-fitted on draft halves.
- It only reduces variance. Because the propensity is known, the estimator stays unbiased even if p̂ is wrong. This
  is the model-X / conditional-randomisation-test situation (Candès et al. 2018) and Robinson (1988) / DML
  (Chernozhukov et al. 2018) with an exact first stage.

**Everything is computed from sparse sums**, never a games × cards matrix: hand triplets, about 6 per game;
per-build sums of a·e; and a sparse build × card deck matrix.

### 2.2 Standard errors
- **HC (design-based):** shuffles are independent across games, even within a draft. Conditional on the potential
  outcomes, Var = Σ r² e² / (Σ r²)², with no clustering needed.
- **Draft-cluster** sandwich as a check. The self-test ratio is 1.00. On real data a ratio > 1.1 would mean
  e is correlated within a draft in a way that interacts with r, which is worth knowing.
- Empirical-Bayes shrinkage (normal-normal, τ² by moments) for the prior and for display.

### 2.3 Power
- A common spell in about 15% of decks appears in about 100k train games of a 1M-game set.
- Var(h_c | ·) ≈ m·(n/S)(1 − n/S)(S − m)/(S − 1) ≈ 0.15 for one copy (m ≈ 4.4 spells in hand, S ≈ 23).
- That gives SE ≈ 0.5 / √(15k) ≈ **0.4 pp**, or about 0.017 logit.
- Rares in about 2% of decks: SE ≈ 1.2 pp.
- Expect true hand-value sd of 2-4 pp. The rating is then precise for most commons and uncommons, and noisy but
  usable (shrunk) for rares.

### 2.4 Diagnostics that can falsify the design
- **Balance index:** Σ_g h_gc / Σ_g E[h_gc] − 1, the relative over-representation of c in recorded hands. Under A it
  is 0 ± noise for every spell.
  - Non-zero values that track HV mean that keep decisions select on spell identity (in the synthetic world, where
    2-landers with bad spells get mulliganed, corr = +0.64). That biases HV slightly *toward* the cards players
    keep for.
  - Compare against the core subset (k = 0, L ∈ {3, 4}), where the keep is nearly automatic. Corr(full, core) is
    reported. If it falls well below the split-half reliability, the selection matters, and core becomes the
    primary estimate.
- **Placebo:** HV for basic lands within lands should be ≈ 0, apart from colour: a Forest in a mono-red-splash
  deck is not neutral. A cleaner placebo is identical-function cards (two basic printings would be ideal, but they
  share a name).

---

## 3. Relation to deck value

**The deck value of one copy in the 40,** relative to a random other card and over shuffles:

  w_c ≈ (7/N)(β^H_c − β̄^H) + (D̄/N)(β^D_c − β̄^D).

**With β^D = λ β^H,** where λ is the value of a top-deck relative to the value in the opener:

  w_c / HV_c = κ(λ) = (7 + λD̄) / (N (1 − λD̄/(N − 7))).

With N = 40 and D̄ ≈ 10.4 (MSH):
- κ(0) = 0.175
- κ(0.5) = 0.36
- κ(1) = 0.64

So:
1. **The cross-card slope of deck-model card power on HV estimates λ**, after correcting for errors in variables by
   HV's reliability. The script reports it, with the implied λ. The deck model's ridge shrinks w, so use the val-
   chosen λ and read the slope as a lower bound.
2. **Card-level residuals from that line are interpretable.**
   - Above the line (deck value > κ·HV): cards that are better drawn late than in the opener, such as bombs,
     card advantage and expensive finishers (λ_c > 1).
   - Below the line: early-game cards (λ_c < 1).
   That is a clean "curve role" measure. Note that HV is *net* of the library-displacement term
   −q(β^D_c − β̄^D), so a card that is just as good at any time has HV = (1 − q)β.
3. **HV is not a deck-strength predictor by itself.** It is a causal **component**. A deck model whose card powers
   track κ·HV plus a role adjustment is consistent with the experiment. One whose card powers carry variation that
   HV lacks and that correlates with the drafters' skill is carrying the drafter.

---

## 4. What this buys

1. **An unconfounded evaluation target for card ratings.** Fit HV on **test** games and correlate it with every
   rating fit on **train** games: GP WR, OH WR, GD WR, GIH WR, IWD, the deck model's w and HV_train.
   - Disattenuate by test-HV reliability (split-half, Spearman-Brown).
   - **Skill tilt:** corr(rating residual on HV, mean skill bucket of the decks playing the card). It measures how
     much of what a rating says beyond the randomised effect follows *who plays the card*.
   - Synthetic self-test, a world where skilled drafters prefer zero-effect "trap" cards:

     | rating | skill tilt |
     |---|---|
     | GIH WR | +0.83 |
     | GP WR | +0.95 |
     | IWD | +0.36 |
     | deck model | +0.48 |
     | **HV** | **+0.03** |

     The trap cards' GIH WR residual is +7.9 pp, against an HV bias of −0.023 logit (1.4 SE).
   - This directly tests the meta-review's reading of "GIH WR beats IWD on 19/19 sets": if GIH WR wins on deck
     prediction but loses on HV agreement or carries skill tilt, the win is player signal.
2. **A prior or regulariser for card power** (§5, implemented as a linear proxy): w = κ·HV + δ with ridge on δ.
   It helps where deck data are thin (rares) and nudges card power toward the causal ordering.
3. **Luck-adjusted outcomes for variance reduction** (not implemented). AIVAT-style (Burch et al. 2018, poker
   agent evaluation), since the chance distribution is known:
   - ỹ_g = y_g − Σ_c r_gc HV_c − [f̂(L_g) − E_hypergeom f̂(L | deck)].
   - It has the same conditional mean given the deck, so it is unbiased for deck strength, with less variance. The
     land-count term (screw/flood) is the bigger part.
   - It could tighten SC-GLL-type comparisons or Spearman targets. HV_c and f̂ must be fit on other games.
4. **Analogues.**
   - A within-family genetic association study: deck = parents, shuffle = meiosis, the alleles are randomised
     given the parents.
   - A stratified randomised experiment with known assignment probabilities (Horvitz-Thompson / Neyman
     inference).
   - Lottery and other designs where the propensity is known.

**Literature on 17Lands metrics.**
- GIH WR = WR of games in which the card was in the opening hand or drawn.
- OH WR = in the kept opening hand.
- GD WR = drawn after the opener.
- GNS = in the deck but not seen.
- IWD = GIH WR − GNS WR.

**Known biases of these metrics.**
- GP, GIH, OH and GD are *between-deck* comparisons, so they are confounded by drafter skill and archetype.
- GIH and GD are also confounded by game length, because long games see more cards and card draw adds more. GD
  and GIH favour cards that are seen when games go long (round 8: the seen-card effect is mostly card flow).
- GNS is enriched for short games.
- IWD removes the between-deck confound but keeps the length and card-draw confound, plus the mulligan/keep
  selection on OH.
- OH WR is the only one dealt before play, but 17Lands compares it *between* decks. The within-deck contrast with
  the known propensity is what turns OH into an experiment.

---

## 5. Hand value as a prior: comparison design

**Arms.** All arms are game-level, with the same frozen context as the deck model and deck counts only, so there
is no hand information at prediction time. HV is fit on **train games only**, EB-shrunk, for spells; lands get a
prior of 0.

| arm | model | tuning |
|---|---|---|
| ridge | w ~ N(0, 1/λ) | λ on val |
| ridge + κ·HV | w = κ·HV + δ, κ free, δ ~ N(0, 1/λ) | λ on val |
| κ·HV only | diagnostic: how far the causal ratings go alone | — |

**Scoring.**
- Test-game log loss ×1000, paired, with a draft-cluster SE.
- Per-draft loss sums are saved (`test_losses.npz`), so the 12-set decision is
  `eval_protocol.meta_analysis` + `decide(DecisionRule(min_effect=0.02))` with no refits.

**For the real decision**, move it to the meta-review's frame:
- Replace the linear proxy with the round 4 FM's w, using a prior-centred penalty λ_w‖w − κ·HV‖² in
  `synergy.DeckFM.penalty`. That is a new subclass, because existing files must not be edited.
- Score with SC-GLL-dl (de-leaked bucket, users ≥ 50 games) on the deck-only score.
- Apply the meta-review's vetoes: match-1 sign, leakage index.
- Expected gain is small (synthetic: +0.008 ± 0.08, null), because with 100k+ games per set the deck model already
  pins common cards. The value is in correctness: skill tilt and HV agreement. Report those as a second, validity
  axis. A prior that leaves SC-GLL flat but cuts skill tilt is worth adopting.

---

## 6. Script, runtime and checks

`python scripts/hand_value.py SET` writes `results/SET.PremierDraft/hand_value/{card_values.csv, results.json,
test_losses.npz}`. `--selftest` runs in about 6 s and 0.4 GB.

**What the self-test checks.**
- The streaming sparse loader matches a dense read. It was also checked on 1,000 real MSH rows, with early stop.
- In the synthetic world (42k games, 60 spells, skill-biased deck choice, spell-dependent keeps for 2-landers,
  London-mulligan bottoming, drawn cards worth 0.8× the hand value):
  - corr(HV, truth) = 0.85
  - 95% CI coverage 0.95
  - the trap-card bias is within noise, while GIH WR is skill-inflated
  - HV agrees with test-game HV at least as well as GIH WR does

**Data loading.**
- Only these columns are read: `eval_common.load_games` (narrow columns), `build_table` (one deck row per build),
  and `opening_hand_*` / `drawn_*` streamed in 65k-row batches into sparse triplets.
- No dense games × cards matrix is ever built. The same goes for torch and lightgbm.

**Runtime and RAM.** Extrapolated from synthetic scaling (180k games: 15 s of analysis, 0.63 GB including the
generator) and from the loaders' footprints:

| set size | load (drafts + streaming) | analysis | peak RSS |
|---|---|---|---|
| MSH (378k games) | ~1 min | ~0.5-1 min | ~0.7-1.0 GB |
| typical 0.8-1.0M games (BLB, DSK, MKM, FDN…) | 1-2 min | 1.5-3 min | ~1.2-1.5 GB |
| OTJ (1.17M) | ~2 min | ~3-4 min | ~1.5-1.8 GB |

The analysis is about 20 L-BFGS logistic fits on about 70% of the games: 2 + 2 outcome cross-fits, 5 deck λ, and
5 + 5 + 1 for the prior arms. `load_drafts` (the draft file) dominates loading, as in every script. 12 sets take
about 40-60 min serial, at 2-4 threads (`OMP_NUM_THREADS`).

**Queue suggestion.**
1. Run MSH first and check RSS.
2. Then run the 12 sets. Read three things, in order:
   - (i) split-half reliability. HV is useful as a target only if it is ≥ 0.5 on table cards.
   - (ii) corr(full, core) and the balance-index sd, to validate the design.
   - (iii) the ratings table: GIH WR vs deck model vs HV against test HV, and skill tilt.
3. Then decide on the prior (§5) with meta-analysis.

**Open items and risks.**
- **Keep selection on spells at a fixed L** is the one threat to the design. It is diagnosed (balance, core
  subset) and not modelled.
- Colour-mismatched spells (an off-colour splash in hand) are part of the effect. That is correct for "value in
  this deck", but it means HV mixes power and castability. Reporting HV by deck-colour fit is a possible extension.
- HV uses the raw skill bucket only inside p̂, for precision. The bucket's leak cannot bias HV, because of the
  design.
- The prior arm uses the raw bucket in the deck model's context, the same in both arms. For the decision, use
  SC-GLL-dl as above.
- Replay data (MSH, HOB) would add the turn each card was drawn. The first draw step is also randomised, so HV
  could be extended to "drawn on turn t ≤ 3", which is still pre-play.
