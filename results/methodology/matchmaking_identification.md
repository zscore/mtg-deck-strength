# Identification under record + rank matchmaking

Memo, 2026-10-12. Scope: what the 17Lands Premier Draft data can and cannot tell us about the spread of true
per-draft strength, how that spread splits into deck and player, and what a "decompressed" deck-strength scale
would be. I ran only cheap checks (a few game columns, seconds). Their code is in
`results/methodology/matchmaking_checks.py`. Numbers marked **[new]** come from those checks. The rest come from
`variance.json`, `matchmaking_variance.py` and the research log.

**Bottom line.**
1. Records pin down how spread out 17Lands users are *relative to* the opponents they are matched against. They do not pin down the absolute scale.
2. That scale is one number: s, the logit spread of the Arena population inside a rank pool. Records leave a flat ridge along t ≈ s, where t is the users' spread.
3. A randomized shock of known size breaks the ridge. Bo1 play/draw is a coin flip. Being on the play in game t-1 buys an extra win, and under record matching that win buys a harder opponent in game t. The size of that knock-on effect measures s.
4. MSH alone gives a Wald effect of −0.10 ± 0.04, which implies s ≈ 0.85. Pooling MSH, EOE and ECL gives −0.05 ± 0.024, which implies s ≈ 0.55. Pooling all 19 sets should give ±0.1 on s.
5. The deck/player split comes from two places. One is the long-run win-rate bucket, which carries a known amount of binomial noise. The other is a time-based linkage of consecutive runs by the same user. That linkage exists **[new]**: about 15% of drafts can be linked, at roughly 80% precision.

---

## 1. Formal model

### 1.1 Setup

- Run i is played by user u(i) with deck D_i. Its strength on the logit scale is θ_i = δ_i + a_u. Here δ_i is the
  deck's contribution (including the user's drafting skill as it shows up in the deck) and a_u is play skill.
  Decompose δ_i = δ̄_u + ε_i: the user's typical deck quality plus run-specific pack luck.
- The Arena population in rank pool r has strength distribution F_r with mean μ_r and sd s_r. The 17Lands users
  in pool r follow G_r with mean μ_r + d_r and sd t_r. They are a selected, stronger subsample (d ≈ 0.3 logit:
  their first-game win rate is 0.567).
- Game k of run i is played at state σ_ik = (w, l) (record before the game), rank r_ik, on the play π_ik ∈ {±½},
  with mulligan difference m_ik. Then

      P(y_ik = 1 | θ_i, θ_j, ·) = Λ(θ_i − θ_j + h π_ik + γ m_ik),    θ_j ~ F_{r, w, l}

  where Λ is the logistic function and F_{r,w,l} is the strength distribution of the population runs that are
  sitting at (w, l) in pool r.
- Under exact record matching, F_{r,w,l} follows from F_r by the forward recursion already in
  `matchmaking_variance.py::opponents`:

      f_{w+1,l}(θ) += f_{w,l}(θ) · E_{θ'~F_{w,l}} Λ(θ − θ'),     f_{w,l+1}(θ) += f_{w,l}(θ) · (1 − ·)

  The recursion stops at 7 wins or 3 losses. Each pool is a closed system (rank mostly indexes the pool).
- The stopping rule, rank promotions and matching all depend only on past outcomes. So for likelihood inference
  given θ_i they are ignorable in Rubin's (1976) sense, the same argument that makes p^W(1−p)^L valid.
  **What is not ignorable is that the opponent distribution depends on the unknown F.**

### 1.2 Proposition 1: the population's record distribution carries no information about F

With exact record matching, every game at state (w, l) has one winner and one loser, and both come from (w, l).
So exactly half the mass at (w, l) moves to (w+1, l) and half moves to (w, l+1), whatever F is. The population's
record distribution is therefore the fair-coin one: P(7-x) = 0.0898 and P(0-3) = 0.125, for s = 0.3 and s = 1.2
alike (checked numerically). If 17Lands users were a random sample (G = F), their records would carry **zero**
information about s. This is the same reason a Swiss system reveals the ranking but not the scale.

### 1.3 Proposition 2: records identify spread relative to the pool, not the absolute spread

For G ≠ F, the users' record distribution depends on (d, t, s). I computed the exact final-record distribution
(Gauss–Hermite, 24 nodes) for the truth s = t = 0.8, then refit with s fixed elsewhere and (t, d) free:

| assumed s | best t | best d | KL per draft | drafts needed to reject at 2σ |
|---|---|---|---|---|
| 0.3 | 0.26 | 0.25 | 6.0e-5 | ~33,000 |
| 0.5 | 0.49 | 0.28 | 1.9e-5 | ~106,000 |
| 1.2 | 1.21 | 0.44 | 1.5e-5 | ~133,000 |

With s *fixed* at 0.8, t is sharp. For t = 0.6 or 1.0 the KL is about 8e-3 per draft, so about 250 drafts reject
it. Records therefore pin down (t/s, d/s) but leave a flat ridge along s. The small curvature along that ridge
comes from the logistic shape and normal tails, i.e. from parametric assumptions. Imperfect matching or a
non-normal F would swamp it. Stage 1 of `matchmaking_variance.py`, which sets s = t, sits on this ridge. Its
estimate of s is not credible on its own, however tight its optimizer looks. The random-effects fit in
`variance_explained.py` (true sd 0.30, slope 1.30) uses free linear record terms. Those terms absorb the
matchmaking gradient, which is itself a function of s, so that fit is biased toward a small s (see 1.5).

### 1.4 Proposition 3: slopes of observed scores are identified given s, and only weakly depend on it

Write θ_i = β z_i + u_i, where z_i is a deck model's logit, with Var(u) = τ². Inside a state, users with different
z face the *same* opponent distribution. So within-state contrasts identify β once F_{w,l} is known up to its
mean, and state-specific intercepts would absorb the mean. β = 1.2–1.7 (MSH) is therefore robust: the model
logits are compressed. τ² sits on the same ridge as t².

**β² Var(z) is an identified lower bound on Var(θ_i).** Every further predictor (de-leaked skill bucket, linked
prior run) raises the bound. That is the honest headline until s is anchored.

### 1.5 Heterogeneity vs state dependence

A draft's games are a short panel. Two forces correlate them:
- spurious state dependence: persistent θ_i, which makes outcomes positively correlated;
- true state dependence: a win moves you to a harder state, which makes the next outcome negatively correlated.

This is Heckman's (1981) problem. With free state effects α_{w,l}, the two are confounded: a decreasing α_{w,l}
is exactly what heterogeneity plus survivorship would also produce. The naive OLS of y_t on y_{t−1}, with
pre-state fixed effects, is −0.0004 ± 0.0022 in MSH **[new]**: the two forces cancel. The textbook way to separate
them is an exogenous shock to the lagged outcome. Bo1 play/draw is exactly that (section 2.1).

### 1.6 What is and isn't identified

| quantity | from records alone | with the play/draw anchor | other sources |
|---|---|---|---|
| user mean vs pool, d_r / s | yes (first game) | yes | |
| user spread vs pool, t / s | yes, sharp | yes | |
| pool spread s (absolute) | no (parametric curvature only) | **yes** | opponent colour gradient (weak), Bo3 within-match correlation |
| slope β of a score | yes, mildly dependent on s | yes | first-game-only regression |
| τ², the residual variance | only as t/s | yes | |
| rank pool offsets μ_r | confounded with who sits at each rank | no help | within-run promotions, season reset |
| Var(a) vs Var(δ̄) vs Var(ε) | no | no | bucket (repeated measure), linked runs, Sealed |
| spread by rank, s_r | no | per rank, weak power | |

---

## 2. Identification sources in this data

### 2.1 Bo1 play/draw (best source; randomized) **[new]**

Play is 50.0% at every match number (0.494–0.507), consistent with a coin flip. Being on the play raises the win
rate from 0.536 to 0.586.

**Design.** Take pairs of consecutive games (t−1, t) with complete history and pre-state w ≤ 5, l ≤ 1. That
restriction means game t exists whatever happens in t−1. Regress y_t on play_{t−1} plus play_t and pre-state
fixed effects. The first stage is the effect of play_{t−1} on y_{t−1}. The Wald ratio estimates the gap between
P(win | state after a win) and P(win | state after a loss), which is the matchmaking gradient.

| set | N pairs | reduced form | first stage | Wald (≈ se) |
|---|---|---|---|---|
| MSH | 212k | −0.0051 (0.0022) | +0.0505 | −0.102 (0.043) |
| EOE | 312k | −0.0006 (0.0018) | +0.0459 | −0.014 (0.039) |
| ECL | 298k | −0.0021 (0.0018) | +0.0436 | −0.049 (0.042) |
| pooled (inverse variance) | | | | −0.052 (0.024) |

**Mapping to s.** These are the Wald values the exact record-matching recursion implies (users' d = 0.3):

| s | 0.3 | 0.5 | 0.8 | 1.0 | 1.2 | 1.5 |
|---|---|---|---|---|---|---|
| implied Wald | −0.020 | −0.048 | −0.094 | −0.122 | −0.146 | −0.174 |

The gradient is roughly −0.12 s in this range. Pooled, the estimate is s ≈ 0.55, with a 95% interval of roughly
0.1–1.0. All 19 sets should give a Wald standard error of about 0.01, which is about ±0.1 on s.

**Assumptions and threats.**
1. Play/draw has no direct effect on the next game. This is plausible: Bo1 games are independent.
2. Abandonment after a loss could depend on play_{t−1} through y_{t−1}. Check this by regressing "game t is
   observed" on play_{t−1}. The pre-state restriction removes the mechanical part but not voluntary quitting.
3. A win can also trigger a rank promotion. That is part of the treatment, it is small (about 2% of games), and it
   can be modeled.
4. The mapping to s assumes exact record matching and normal F. Soft matching (opponents drawn from neighbouring
   states when the queue is thin) makes the gradient smaller, so the mapping would understate s. The full
   likelihood in section 4 handles this better than the Wald mapping.

Opponent mulligans are a weaker second instrument (Wald −0.02 ± 0.02). They are also less clean, because an
opponent's mulligan correlates with their deck's mana base, which is part of θ_j.

### 2.2 The 0-0 first game

At 0-0 the opponent is a random entrant from pool r: there is no record selection.
- The first-game win rate by rank (bronze 0.65, silver 0.61, gold 0.57, platinum 0.56, diamond 0.54, mythic 0.55)
  identifies E_G[θ] − E_{F_r}[θ] for each pool.
- A regression of y_1 alone on a deck score has no matchmaking compression. It does have logistic
  non-collapsibility (Gail et al. 1984): the slope is about β / sqrt(1 + π(s² + τ²)/8).
- The first game is the natural place to *define* the decompressed scale (section 4.3), and it gives a clean
  evaluation slice (section 5).

### 2.3 Rank as a pool offset, and three shocks that move it

The game WR gradient by rank (0.62 → 0.54) alongside a flat long-run bucket (0.54–0.56) mixes two things: pool
strength, and *which users* are at that rank at that moment.

- **Within-run rank changes [new].** Rank is recorded per game. 12% of drafts (8.2k in MSH) change rank mid-run:
  7.5k promotions and 0.9k demotions. Deck and player are fixed within the run, so comparing games at the
  *same* state before and after a promotion identifies the pool offset μ_{r+1} − μ_r. Whether a 3-0 player gets
  promoted depends on their tier progress within the rank, which comes from earlier runs and is close to
  exogenous given rank. Promotion is still caused by wins, so condition on state.
- **Season reset [new].** Seasons run about a month. Limited ranks reset (Mythic drops to Platinum 4), and Bronze to
  Gold give +2 tiers per win. In MSH, days 1–7 have far more gold and platinum games and far fewer mythic (1.4k vs
  30k for day 15+). Gold's game WR is 0.560 in days 1–7 against 0.573 later: the early gold pool contains
  displaced strong players. Day-of-season × rank therefore shifts pool composition while the users' own skill
  distribution stays put. Use it as a second shock to μ_r.
- `opp_rank` is always null, so all of this has to be inferred.

### 2.4 Long-run WR bucket: a noisy, leaky repeated measure of the player

What the data say **[new]**:
- Both buckets are constant within a draft and look like **export-time totals for the user in this set and
  format**. Drafts with games bucket "1" have up to 4 games in the current run, and their bucket correlates 0.91
  with the current run's own WR.
- The games bucket is a floor (1, 5, 10, 50, 100, 500, 1000).
- The WR bucket is 0.02 wide, and it is coarsened at high counts: the 1000 bucket has a single value.

**Measurement model.** b_u = round(Ȳ_u, 0.02), where Ȳ_u is the mean over n_u games, and n_u ∈ [floor, next
floor). The current run's weight in the bucket is n_run / n_u, so the leak is about 10% at n_u = 50 and about 2%
at 250.

**De-leaked measure.** b̃_i = (n_u b_u − W_i) / (n_u − n_i), with n_u integrated over the bin (log-uniform). Its
noise variance is ≈ b̃(1 − b̃)/(n_u − n_i) + 0.02²/12.

**Implied spread of true long-run WR (MSH).** Taking sd(bucket)² minus the binomial noise at the bin's geometric
mean:

| games bucket | 10 | 50 | 100 | 500 |
|---|---|---|---|---|
| implied true sd | 0.048 | 0.057 | 0.061 | 0.069 |

That is about 0.25 logit. The users' persistent strength is clearly real.

**Caveat.** Long-run WR is *itself* compressed by the rank ladder and by record matching. A user who climbs meets
harder pools, which is the "win rate goes to 50%" property of rating ladders. So b̃ measures P_u = a_u + δ̄_u
relative to the pools the user played in. It has to be passed through the same structural map, not used raw.

### 2.5 Linking runs by the same user (no user id) **[new]**

There is no user id. The buckets are a per-user, per-set constant, but they are coarse (144 cells in MSH, median
120 drafts per cell). Adding time makes them useful: a user's next draft often *starts within 0–2 minutes of
their previous run's last game*.

Take candidate predecessors with the same (games bucket, WR bucket) and the same rank (end rank of the previous
run = start rank of the next) and count them by gap. For an 8,000-draft sample in MSH:

| gap (min) | −2..−1 | −1..0 | 0..1 | 1..2 | 2..3 |
|---|---|---|---|---|---|
| same rank | 165 | 154 | **831** | **840** | 401 |
| other rank (placebo) | 298 | 311 | 323 | 305 | 372 |

There is an excess of about 1,350 true links per 8,000 drafts at about 80–85% precision. The excess appears only
when rank is continuous, which validates it. EOE and ECL show the same pattern, peaking at 1–2 minutes.

- **Uses.** Within-user contrasts between two decks (player fixed effect) and across-run covariance, which equals
  Var(P_u). The false-link rate can be estimated from the placebo window and corrected, because false links
  shrink across-run covariances by the precision.
- **Cross-set.** Nothing works: draft ids and buckets are per set, and nothing persists across sets. Chains across
  sets would rest on timestamps alone and are not credible.
- **Same set, other formats.** Linking to Sealed or TradDraft is also not viable, because those buckets are
  computed per format.

### 2.6 Opponent colours: a partly observed θ_j **[new]**

`opp_colors` is recorded. Score each opponent colour pair by the users' own win rate with that pair:

| | MSH | EOE | ECL |
|---|---|---|---|
| opponent strength at 0-0 → 6-x | 0.5515 → 0.5553 | 0.5494 → 0.5518 | 0.5376 → 0.5428 |

The rise is monotonic in all three sets. This is direct, model-free evidence of record matching. Under normality,
E[c_j | w, l] = (Var c / Var θ) E[θ_j | w, l], so the gradient identifies Var(c)·κ(s), where κ only varies from 2.7
to 2.0 as s goes from 0.3 to 0.8. That makes it a good specification check and a weak handle on s. The colour
gradient by rank is flat (0.552–0.553).

### 2.7 Within-run sequence structure

- The path, not just the final (W, L), is informative: the same 5-2 means different things if the losses came at
  0-0 or at 5-0, because the opponents differ.
- The full path likelihood already uses this.
- It adds little information about s beyond the play/draw shocks, which enter through the path.

### 2.8 Traditional Draft (Bo3)

What the data say **[new]**:
- Rank is null for every TradDraft row.
- There is no local TradDraft game file; only the draft files are present.
- Complete 3-match runs are **under-dispersed** compared with a binomial at the users' own mean: var(wins) is
  0.641 vs 0.669 (MSH, n = 5.4k) and 0.647 vs 0.699 (FDN, 15.4k). With random pairing, heterogeneity would make
  them *over*-dispersed. So TradDraft also matches by record (or something equally compressing). Users win 63–66%
  of matches there.

**Why Bo3 games would matter.** If 17Lands publishes TradDraft game files (check
https://www.17lands.com/public_datasets), the 2–3 games of a match share an opponent:

    Cov(y_ig, y_ig' | same match) ≈ Λ'² · Var(θ_i − θ_j | state)
    Cov(y_ig, y_ig' | different matches) ≈ Λ'² · Var(θ_i) − matching term

The difference identifies Var(θ_j | state), the pool spread at that state, directly. That is a second
independent anchor for s.

**Caveats.**
- Games 2–3 are post-sideboard.
- In games 2–3 the loser of the previous game chooses play/draw, so play is not random there. Use only the
  game-1 play/draw.
- The Bo3 population differs from the Bo1 one, so s_Trad need not equal s_Premier. This anchor validates rather
  than substitutes.

### 2.9 Sealed: exogenous deck luck

Sealed game files are local. Sealed pools are random, so pool quality is pack luck alone, independent of drafting
skill. The variance of a pool-quality score's effect in Sealed (fit with the same structural model) gives a clean
Var(ε)-type deck-luck component. That component is not confounded with drafting skill the way δ̄_u is in draft.

---

## 3. Literature

- **Rating systems.**
  - Elo (1978), *The Rating of Chessplayers, Past and Present*.
  - Glickman (1999), "Parameter estimation in large dynamic paired comparison experiments", *JRSS-C* 48 (Glicko).
  - Herbrich, Minka & Graepel (2006), "TrueSkill", NIPS.
  - Dangauthier et al. (2007), "TrueSkill Through Time", NIPS: smoothing in both directions and initial-skill
    calibration.
  - Minka, Cleven & Zaykov (2018), "TrueSkill 2: An improved Bayesian skill rating system", MSR-TR
    ([pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/03/trueskill2.pdf)). TrueSkill 2 adds
    experience and quit-tendency covariates, which parallel our games bucket and abandonment. All of these
    systems identify skill **because opponents' ratings are jointly estimated from a connected graph of
    identified players**. We see one seat and anonymous opponents, so the graph is a star. That is the root of the
    problem.
- **Win rate under matchmaking.** Under rating-based matchmaking a player's win rate is pushed toward 50% by
  construction: winning raises your rating, which raises your opponents' strength. Win rate then measures the
  matchmaker's error, not skill. This is folklore in ladder design (for example the Hearthstone wiki entry on
  matchmaking), and the I3D 2024 paper on skill-based matchmaking for two-player games discusses it
  ([Yuksel 2024](https://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking.pdf)). Arena's record
  matching is a per-event version of the same mechanism, and its rank ladder is a slow, cross-event version
  ([Draftsim on Arena matchmaking](https://draftsim.com/mtg-arena-matchmaking/): record first, rank second;
  [ranking system](https://draftsim.com/mtg-arena-ranking-system/)).
- **Swiss systems and tournament design.** Record matching is a Swiss system with an elimination rule.
  - [Sziklai, Biró & Csató (2022), "The efficacy of tournament designs", *Computers & OR*](https://arxiv.org/abs/2103.06023)
    find the Swiss system is the most efficient at recovering the *ranking*. Our Proposition 1 is the flip side:
    it destroys information about the *scale*.
  - [Glickman & Jensen (2005), "Adaptive paired comparison design", *JSPI* 127](https://www.sciencedirect.com/science/article/abs/pii/S0378375803003215)
    cover optimal adaptive pairing.
- **Skill-based matchmaking in industry.**
  - [Chen, Xue & Kolen (2017), "EOMM: An Engagement Optimized Matchmaking Framework", WWW](https://arxiv.org/abs/1702.06820).
  - [Activision (2024), "The Role of Skill in Matchmaking"](https://www.activision.com/cdn/research/CallofDuty_Matchmaking_Series_2.pdf).
    Their *randomized* loosening of the skill constraint is exactly the kind of exogenous variation that
    identifies the spread. Our play/draw coin flip is a small natural version of it.
- **Adaptive testing analogue.** Computerized adaptive testing (Lord 1980; van der Linden & Glas 2010, *Elements
  of Adaptive Testing*) matches item difficulty to ability, so every examinee ends near 50% correct. Ability is
  still recovered because items are *pre-calibrated*. Here the "items" (opponents) are uncalibrated, which is why
  the scale is lost. The fix is the same: a calibrated reference, here the play/draw effect measured at 0-0.
- **Panel econometrics.**
  - [Heckman (1981), "Heterogeneity and State Dependence"](https://www.nber.org/books-and-chapters/studies-labor-markets/heterogeneity-and-state-dependence).
  - [Wooldridge (2005), "Simple solutions to the initial conditions problem…", *JAE* 20](https://onlinelibrary.wiley.com/doi/10.1002/jae.770):
    condition the random effect on the initial state. Here everyone starts at 0-0, so the initial condition is
    trivial within a run. It is *not* trivial across runs, because rank is the initial condition of a run.
- **Measurement error.** Spearman (1904) on attenuation; Carroll, Ruppert, Stefanski & Crainiceanu (2006),
  *Measurement Error in Nonlinear Models*, for the bucket as an error-prone covariate with known error variance
  (regression calibration and SIMEX). Gail, Wieand & Piantadosi (1984, *Biometrika*) on non-collapsibility of
  logistic coefficients when an independent variable (the opponent) is omitted.
- **17Lands data.** [Public datasets](https://www.17lands.com/public_datasets): the buckets are user-level win
  rate (2% increments) and game count.

---

## 4. Estimators

### 4.1 Estimator A: structural matchmaking likelihood anchored by play/draw (main proposal)

Extend `matchmaking_variance.py` with:
1. play/draw and mulligans inside both the user's game likelihood *and* the population recursion;
2. t ≠ s free (now identified);
3. rank pools with offsets μ_r, a shared s, and optionally s_r;
4. per-game rank;
5. optionally, soft matching.

**Parameters.**
- s: population spread in a pool;
- μ_r: rank offsets;
- h, γ: play and mulligan effects;
- users' θ_i = μ_{r_i1} + d + β z_i + λ b̃_i + u_i, with u_i ~ N(0, τ²);
- optional matching tolerance ρ: an opponent is drawn from state (w, l) with probability 1 − ρ, and otherwise from
  an adjacent state at the same number of games played.

```
def opponent_tables(s, h, gamma, rho):          # [R, 21, K] weights on nodes s*X
    for each rank pool r:  forward recursion over (w,l); a node's win prob vs the pool at (w,l) =
        E_{pi, m} E_{theta'~F_{r,w,l}} Lambda(s*x - theta' + h*pi + gamma*m)   # play is +-1/2 symmetric, so h cancels in the mean but not in the dispersion
        apply soft matching: F~_{w,l} = (1-rho) F_{w,l} + rho * mix(neighbours)
def nll(params):
    opp = opponent_tables(...)
    for chunks of drafts:                         # memory: chunk so [games, K, K] stays < 200 MB
        eta[g, k] = mu[rank_g] + d + beta*z[i_g] + lam*bt[i_g] + tau*X[k] + h*play_g + gamma*mull_g
        p[g, k]   = sum_j opp[rank_g, state_g, j] * Lambda(eta[g, k] - mu[rank_g] - s*X[j])
        ll[i, k] += log p or log(1-p);  marginalize k with GH weights (logsumexp)
    return -sum
fit with L-BFGS in torch/JAX (autodiff); start from the Wald-implied s; profile s on a grid 0.2..1.6 for a CI.
```

The play shock's effect on later opponents is what curves the profile likelihood in s. This is the same
information the Wald estimator uses, used efficiently.

- **Outputs:**
  - (i) Var(θ) = s_users² = β² Var(z) + λ² Var(b̃) + τ², with a profile CI;
  - (ii) see 4.2;
  - (iii) see 4.3.
- **Compute.** The recursion is trivial: R × 21 × K² with K = 32. The likelihood costs games × K × K: about
  4×10⁸ flops per evaluation for MSH (0.4M games) and about 1.5×10⁹ for big sets. With autodiff that is about 1–3
  s per evaluation and 200–500 evaluations, so about 5–20 minutes per set on CPU, with peak RAM about 1 GB when
  chunked. Profiling s over 8 grid points multiplies that by about 4 (warm starts). **Run it only once the
  training queue is idle.**

### 4.2 Estimator B: moment-based, transparent (cross-check, and the deck/player split)

**B1. Scale.**
- Wald_m per set (section 2.1), pooled by inverse variance.
- ŝ = g⁻¹(Wald), with g computed from the recursion at the fitted d and with soft-matching sensitivity bands.
- Cost: seconds.

**B2. Users' spread given ŝ.**
- Fit stage 2 of `matchmaking_variance.py` with s = ŝ fixed and t free. The ridge result says t is then sharp.
- Cost: a few minutes per set.

**B3. Deck/player split, with variance components on the latent scale.** Within run, games share θ_i = P_u + ε_i.
Across linked runs, games share P_u. The bucket measures P_u with known noise.

```
Var(theta)  = t^2                                    (B2)
Var(P_u)    = Cov(theta_run1, theta_run2)            linked pairs, structural likelihood with a bivariate normal
              (P_u, eps1, eps2); divide the across-run covariance by link precision (placebo-estimated)
            ~= Var(true long-run WR) mapped to the logit scale    (bucket EIV: Var(b~) - E[noise])
Var(eps)    = t^2 - Var(P_u)                         run-specific deck luck + form
Var(dbar_u) vs Var(a_u):  regress P_u-hat on the user's mean deck score over linked runs (zbar_u):
              Var(dbar_u) >= beta_within^2 Var(zbar_u),  Var(a_u) <= Var(P_u) - that
beta_within: slope of outcome on z within linked pairs (user fixed effect) = causal deck slope;
             beta_cross - beta_within = how much the cross-sectional deck score is proxying the player
```

The split rests on these assumptions:
- P_u is constant within a set (no learning within the set; check this by gap length and by games bucket);
- links are a random subset of runs (the users who re-queue fast play more and are better, so report the split
  for linked users and reweight by games bucket);
- the deck model captures the deck part (the residual deck part ends up in "player").

Sealed (section 2.9) gives an external value for the pack-luck share of Var(ε).

### 4.3 The decompressed deck-strength scale (iii)

Define two estimands for a deck with model score z (player at the 17Lands mean, or at the population mean):
- **p₀ (vs an average opponent):** Λ(β z + d) against an opponent of exactly mean pool strength, averaged over
  play/draw. This is interpretable as edge in logits.
- **p₁ (vs a random entrant):** E_{θ_j ~ F_{r*,0,0}} E_π Λ(βz + d − θ_j + hπ). This is the 0-0 win probability in a
  reference pool r* (platinum, the modal pool). It is directly checkable against first-game outcomes.

Implementation: report β̂ and ŝ, then map the model's logit z to p₀ and p₁. **Validation:** the calibration of p₁
on held-out *first games only* should have slope 1 and intercept 0 (by rank pool, with μ_r).

### 4.4 Simulation-based validation plan

Build `sim_arena.py` (small, numpy).
- A population of N_pop ≈ 2×10⁵ runs per pool, across 6 pools with offsets, and a 17Lands subsample of about 10%,
  selected on P_u.
- θ = a_u + δ̄_u + ε_i, with the deck score z = δ_i + noise (known R²).
- Users have several runs at Poisson rates, with re-queue gaps drawn from the observed mixture (some immediate),
  and the buckets computed at export with the leak and the 0.02 rounding.
- Bo1 coin-flip play with h = 0.21 logit, and mulligans.
- Matching is by record within rank with tolerance ρ (soft matching when the queue is thin, i.e. at high records
  and off-hours), plus rank promotions by tier steps and a monthly reset.
- Abandonment depends on the record, with optional dependence on y_{t−1}.

**Grid.** s ∈ {0.3, 0.6, 0.9, 1.2}; share of P_u ∈ {0.3, 0.6}; ρ ∈ {0, 0.1, 0.3}; F normal vs skew-normal / t₅;
selection strength.

**Checks** (200 replicates for the moment estimators, 30 for A):
1. bias and RMSE of ŝ, t̂, β̂, and of the three variance components;
2. CI coverage (profile likelihood for A, delta method or bootstrap over drafts for B);
3. sensitivity: fit assuming ρ = 0 when the truth is ρ = 0.3, and fit normal F when the truth is skewed (expect s
   biased low under soft matching);
4. the linkage pipeline run on simulated timestamps, recovering the true link precision.

**Success criteria.** |bias(ŝ)| < 0.1 at ρ ≤ 0.1, and coverage ≥ 90%. If soft matching breaks it, add ρ to
Estimator A and check that ρ is identified, from the variance of opponent colour strength within a state and from
off-hour versus peak-hour games.

Cost: minutes, no data loading.

---

## 5. Consequences for training and evaluation

1. **Training target.**
   - The run-level p^W(1−p)^L assumes a fixed opponent. Under record matching it is a *compressed* target: β > 1 is
     the symptom.
   - Ranking metrics are invariant to a monotone squash, so this does not by itself hurt Spearman. It does
     (a) waste signal, because a win at 5-1 against a strong opponent counts the same as a win at 0-0, and
     (b) distort any blend of features whose compression differs. Player-like features are compressed more by the
     ladder than deck features are by record matching.
   - Proposal: a game-level loss Λ(f(deck) − ō_{r,w,l} + hπ + γm). The offsets ō are **fixed** from Estimator A
     (E[θ_j | r, w, l]), **not** learned as free state effects, since free effects trade off against f (section
     1.5). A cheaper version is to fit the logistic-normal approximation of E_j Λ.
   - A/B this against the current target on Spearman and on first-game NLL. The expected gain is modest. The real
     value is a correctly scaled f.
2. **Rank** belongs in the offset (opponent pool), not in deck features. Use per-game rank, since 12% of runs
   change rank mid-run. Predict deck strength at a reference pool. `rank` in the LightGBM features currently mixes
   "stronger pool" with "who is at this rank in this week".
3. **Player control.**
   - Use the de-leaked bucket b̃ (with its noise variance) as a *nuisance covariate in training*, then predict at
     b̃ = mean. This removes part of the player confounding of card and deck values.
   - Tradeoff: b̃ also absorbs drafting skill (δ̄_u), so this over-controls toward "deck effect within user".
     Report both versions.
   - Never use the raw (leaky) bucket. Its 0.99+ "share explained" in `variance.json` is mostly leakage plus the
     unidentified total.
4. **Evaluation.**
   - Keep Spearman vs W/(W+L) for paper comparability.
   - Stop quoting the iid-simulated "ceiling" in `deck_metrics`. Replace it with a ceiling simulated under record
     matching at the ŝ from section 2.1.
   - Add four metrics:
     - (a) share of true variance explained, β² Var(z)/t̂², with a CI propagated from ŝ;
     - (b) calibration slope of the game-level structural model (target 1);
     - (c) first-game-only NLL and Spearman (no matchmaking);
     - (d) within-user slope on linked pairs, β_within / β_cross, which says how much of the deck model's skill is
       really player.
5. **Pipeline note [new].** `deckstrength.N_GAMES_BUCKETS` = ("1", "5", "10", "50", "100") is missing "500" and
   "1000". `experience()` maps those 15k MSH games to "unknown". Small, but it affects the games-bucket control
   and any measurement-error model of the bucket.

## Assumptions, stated plainly

- Strength is additive on the logit scale with a symmetric logistic link, and F is roughly normal within a pool.
  Estimator A's s depends on this. The Wald check depends on it less, because its mapping only needs the mean
  gradient.
- Matching is exact by record within rank. It certainly is not exact at the margins. ρ is the main
  misspecification risk, and it biases s *down*.
- The Arena population in a pool is stationary over a set's window. This is violated around season resets, which
  is also why reset timing is useful as a shock.
- 17Lands users are a selected subsample, as the model allows. Their *opponents* are mostly non-17Lands players,
  so s describes the Arena population, not the users.
- The linkage results describe fast re-queuers, who are heavier players. Treat them as a subpopulation.
- Of the numbers here, only s is the new headline (MSH, EOE, ECL). Everything else is design. The pooled s ≈ 0.55
  (0.1–1.0) is consistent with the earlier 0.5–0.8 guess, but it is not yet precise enough to settle the share
  explained. All 19 sets should get it to ±0.1.

Resource note: `matchmaking_checks.py` peaks at about 650 MB RSS on MSH, mostly from the parquet read. An earlier
version with an unbounded linkage join hit about 7 GB on EOE/ECL. That has been fixed with time-bucketed joins.
Run it on the remaining sets one at a time when the machine is free.
