# Pipeline audit: leakage, split alignment, silent bugs (2026-10-12)

Scope: `scripts/deck_synergy_v4.py`, `deck_synergy_v4_draft.py`, `improve_deck.py`, `improve_deck_v2.py`,
`improve_deck_picknet.py`, `picknet_features.py`, `variance_explained.py`, `matchmaking_variance.py`,
`game_controls.py`, `src/decks/{deckstrength,synergy}.py`, the mtgdata pieces they call (`games.py`, `data.py
load_tensors`, `cardinfo.deck_features`, `encode.splits_for`), and the picknet training scripts used for the
two pick models. Only code reading and small checks were run: column scans of the parquet files, the cached npz
and json results. No training was run.

**Bottom line.** I found no test-set leakage into the headline draft-level Spearman numbers. The train/val/test
splits match exactly across every source. The problems that matter are these:
1. The queued `game_controls.py` will crash, and the affinity column can't be rebuilt.
2. The target (the W-L record in the draft file) is stale for about 1 in 5 drafts.
3. The reported uncertainties leave out training-seed noise, which is about the same size as most of the
   reported gains.
4. The cross-fitted game-FM scores are on a different scale from the val/test scores.

---

## Answers to the specific questions

**Q1. Split alignment.** The splits are aligned. I checked 9 sets (MSH BLB DFT DSK EOE FDN FIN LCI MKM):
- `draft_time` never varies within a draft. The number of distinct values per draft was greater than 1 for 0
  drafts.
- So `load_drafts`, which sorts by the last-pick row's time (deckstrength.py:62-67), and `load_tensors`, which
  sorts by first-seen time (data.py:155-156), give the same order. Both break ties on `draft_id` and use the
  same completeness rule: in every set the mode of the row count per draft equals 3 x picks.
- The 70/15/15 split differs on 0 drafts.
- The PickNet shard `split.npy` comes from `splits_for`, which reads the meta.pkl splits. It matches
  `load_drafts` exactly, with a diagonal confusion matrix (MSH: 48026/10291/10292). Every `load_drafts` draft
  appears in `feats.npz`, so coverage is 1.0.
- `game_split` inherits each game's split from its draft.

Neither pick model trains on test-period drafts:
- The per-set `pool_d32` model (affinity column) trains on the `splits_for` train split
  (picknet/scripts/train_picknet.py:99).
- The all-sets PickNet (`all_d256_ep3`) trains on the train split of every set and event. That includes
  TradDraft, whose own 70% train split ends before the Premier test period starts (MSH: Trad train ends
  07-11, Premier test starts 07-16; FIN: 07-01 vs 07-07). It does overlap the Premier val period.
- Training a pick model on test-period *picks* would be acceptable anyway for outcome prediction. Picks come
  before outcomes, and they are known at registration time. The one outcome-adjacent target is the PickNet deck
  head's `pick_maindeck_rate`, and that is train-split only.

**Q2. Whole-period quantities.**
- GIH WR (improve_deck.py:126) uses train games only. Good.
- `card_info` and `structure_features` are static Scryfall/17Lands card facts. No leakage.
- Deck-feature standardisation: mu/sd come from train games (deck_synergy_v4.py:89). `deck_features` also
  centres several features on the batch mean, which is transductive; that is finding 10, and it is negligible.
- `Profile` is fit on train games.
- The skill bucket is the one real label leak (finding 3). It contains the outcomes being predicted, so it is a
  leaky control at game level. It is excluded from every draft-level score, so the draft test metrics are not
  inflated. It does over-control the card values, and it inflates absolute game-level logloss/AUC.

**Q3. improve_deck_v2.**
- No leakage into test. Every choice is made on val with train-only fits (tau, grid, number of trees, blend
  alpha), and the refit models are never scored on val. The alpha learned from each train-only twin carries over
  to its refit model, which is sound.
- Two small issues: val is slightly optimistic for the game FM, and the train+val calibrator mixes two score
  scales (finding 5).
- The recency weights are computed correctly but not normalised (finding 8).

**Q4. Constant and null covariates.** Checked on MSH, BLB, FIN and LCI:
- `opp_rank` is the *string* "None" in 100% of games, so the 7 opp-rank one-hot columns are constant.
- `opp_num_mulligans` is fully populated, so the mulligan difference is valid.
- `user_n_games_bucket` values '500' and '1000' fall into the "unknown" slot (4% of games in MSH, 9% in FIN).
- `game_number` is constant (Bo1).
- `build_index` is a String, so its sort is lexicographic.

Details in finding 7.

**Q5. most_played_deck.**
- 2.9-3.1% of drafts have a tie for the most-played build. The tie goes to the lowest `build_index` (by string
  order).
- 19-20% of drafts have more than one build, and in 17.7% the most-played build is not build 0.
- Drafts without games are 1.0% of train and 2.2% of test. They are dropped, and most of them have no record
  anyway (MSH: 106 of 825 have W+L > 0).
- Test decks are chosen using the game counts of test games. That is consistent across splits and is not label
  leakage. Still, it is information you would not have when the deck is registered (finding 9).

**Q6. Numerical and statistical checks.**
- PCA in `improve_deck_picknet` is fit on train only (lines 52, 61). No leak.
- No NaN or inf in any LightGBM input (`draft_inputs.npz`, `feats.npz`).
- Spearman with tied W/(W+L) uses scipy's average ranks, which is correct.
- The bootstraps are paired (the same resample indices for every model), which is correct. They miss the seed
  variance (finding 4).
- The game-level ±95% treats games as iid (finding 12).

---

## Findings ranked by severity

### 1. HIGH (blocks the queue and reproducibility): the pick-model weights are gone, so game_controls.py will crash
- `pick_affinity` looks for `results/{SET}.PremierDraft/picknet/models/pool_d32.pt` (deck_synergy_v2.py:55-57).
- No `pool_d32.pt` exists anywhere under `~/Documents/repos`. Only the `pool_d32.json` metrics survive in
  `picknet/results` and `mtg-open-lane-synergy/results`. `results/**/models/` is gitignored, so the repo split
  (72112ee) didn't carry the weights over.
- Consequences:
  - `game_controls.py:62` does `S, _ = pick_affinity(...)`, which raises
    `TypeError: cannot unpack non-iterable NoneType` once the data has loaded. **The queued job will fail.**
  - `deck_synergy_v4.py:90-93` would print "no pick model, skipping" and *return without writing anything*.
  - Every downstream script reads the cached `deck_synergy_v4/draft_inputs.npz` (Oct 5, gitignored). It is
    currently the only copy of the affinity column and the cross-fitted `game_logit`.
- Fix:
  - Retrain the pool models: `picknet/scripts/train_picknet.py <SET> pool d=32 ...` with the settings in
    `pool_d32.json`, saving to this repo's `results/{SET}.PremierDraft/picknet/models/`. Or point
    `pick_affinity` at wherever the weights live.
  - Make `pick_affinity` raise rather than return None when the weights are missing.
  - Back up the `draft_inputs.npz` files.
  - Until the weights exist, `game_controls.py` could fall back to dropping the affinity column. Its comparisons
    are between control sets, so that is acceptable if done consistently.
- Impact on reported numbers: none today. Any rerun of rounds 4 and later is impossible as things stand.
  Retrained pool models will not be bit-identical, so expect the affinity-dependent numbers to move by about
  ±0.001-0.002 Spearman.

### 2. HIGH (target quality): the draft-file event record is stale or incomplete for many drafts
- The target `wins`/`losses` comes from the draft file's `event_match_wins/losses` (deckstrength.py:76-78).
- Compared with the game file (MSH, last-pick rows):
  - 36k of 68k drafts agree.
  - **18.8% of train drafts (17-21% across splits) have more games in the game file than W+L.** Example: a 4-1
    record, but 7 logged games with 5 wins and match numbers up to 9.
  - 25-29% have fewer games (17Lands is missing those games).
- So the draft-file record looks like a snapshot taken partway through the run for about a fifth of drafts.
- This is consistent across splits, so it is not leakage. It does add label noise, and it biases the `ceiling`
  in `deck_metrics` (deckstrength.py:196-219) upward, because the ceiling simulates complete 7-3 runs.
- Fix:
  - Build a reconciled record per draft: take the game-file W/L when its history is complete
    (`record_before`'s `known`) and it has at least as many games, and otherwise keep the draft-file record.
  - At minimum, report the headline Spearman on the subset where the two sources agree.
  - Make the ceiling simulation reproduce the observed truncation.
- Impact: absolute test Spearman is probably understated by a few hundredths. My guess is +0.01 to +0.03 on the
  agreeing subset; this is not measured. Comparisons between models stay valid.

### 3. HIGH for interpretation, none for draft-level test leakage: the skill bucket contains the outcome
- `covariates()` puts `user_game_win_rate_bucket` into every game model (deckstrength.py:122-124). The bucket is
  computed over the user's games, including this run's.
- Evidence: in `results/MSH.PremierDraft/variance.json`, "rank + skill bucket" explains 99.99% of the latent
  win-rate variance of the *test* drafts.
- Effects:
  - (a) Card values `w` are learned with the outcome partly controlled for (over-control, or collider-like).
    For low-volume users the bucket is close to this run's own record, so deck strength is absorbed and the card
    values shrink.
  - (b) Game-level `test logloss` and `test auc` in `deck_synergy_v4/results.json` are optimistic in absolute
    terms. The "vs ref" differences are fair, because every model shares the control.
  - (c) Draft-level scores do **not** use it. `draft_logit` uses only `deck_score` plus the extra columns
    (deck_synergy_v4.py:124-127), and the LightGBM inputs have no skill column. PickNet features hide it
    (picknet_features.py:35, 79; I verified that slice 7:11 is wr, wr-unknown, log n-games and n-games-unknown in
    `draft_context`).
  - (d) `experience()` (user_n_games_bucket) is also whole-period, but it is a count, not an outcome. Minor.
- Fix:
  - Let `game_controls.py` (its "- skill" variant) decide, once finding 1 is fixed.
  - Better: drop the bucket, or replace it with a skill proxy that doesn't include the label. Options are the
    PickNet surprise features, or the bucket only for users in the 100+ n-games bucket, where one run's weight
    is small.
- Impact: unknown sign for draft Spearman. Dropping the bucket could *raise* game-FM draft Spearman, since card
  values would no longer be shrunk, or lower it if skill confounding dominates. Absolute game AUC/logloss
  would fall.

### 4. MEDIUM-HIGH: seed noise is not in the reported uncertainty, and is about the size of the gains
- The same features and parameters on MSH give two different answers:
  - `lgbm + deck feat + aff` in draft.json (LGB_PARAMS, default seed) scores **0.2831**.
  - `round 4: lgbm` in improve_v2.json (`seed=0`, improve_deck_v2.py:45) scores **0.2816**.
  - Their blends: 0.2892 vs 0.2882.
- The `vs ... sd` columns (improve_deck_v2.py:139-148, improve_deck_picknet.py:91-102) bootstrap test drafts
  only, with a fixed fit. Many reported PickNet and v2 deltas (+0.001 to +0.003) are within seed noise.
- The "5 seeds" refit row shows seed averaging itself moving the number by +0.0005 to +0.001.
- Fix: run every variant over 3-5 bagging seeds, report the mean and the seed sd, and pair seeds across variants.
  Or switch off bagging and feature_fraction for the comparison runs.
- Impact: treat |Δ| < about 0.002 as unresolved. The refit gain (+0.004 to +0.005) probably survives. Most
  PickNet single-group gains do not clearly survive.

### 5. MEDIUM: the cross-fitted game-FM scores are on a different scale from the val/test scores
- Train drafts are scored by two half-data models; val and test are scored by the full model
  (deck_synergy_v4.py:141-158).
- From `draft_inputs.npz`:
  - The train-score sd is 15-20% larger (MSH 0.299 vs 0.251 on test; FIN 0.277 vs 0.255).
  - **The two folds differ in mean by 0.05-0.06 logit** (MSH +0.143 vs +0.090; EOE +0.032 vs +0.092), about
    0.2 sd. The cause is the arbitrary intercept split between `deck_score` and the bias.
- Consequences:
  - The `calibrate` fit on train (deck_synergy_v4_draft.py:62, improve_deck_v2.py:86-88,
    improve_deck_picknet.py:77) learns an attenuated slope and is applied to a different distribution.
  - The `init_score` boosting variants (deck_synergy_v4_draft.py:75-78) train on a noisier, fold-shifted init
    than they see at test. That may explain why boosting (0.2838-0.2873) loses to averaging (0.2892).
  - The averaged blends are mostly protected, because alpha is picked on val, which uses the same full-model
    scores as test.
  - `game_trva` (improve_deck_v2.py:87) fits one calibrator across both scales.
- Fix: standardise each fold's scores to the full model's train-draft mean and sd (or the val mean and sd).
  Alternatively, score val and test with the *average of the two half-models*, so every split uses the same
  estimator, or fit the calibrator on val.
- Impact: blends, ≤0.001-0.002 Spearman; game FM alone, small NLL gain; boosted variants, possibly +0.002 to
  +0.005 (a guess).

### 6. MEDIUM: in-sample calibration in the game-level scripts' draft metrics
- `deck_synergy_v4.py:121, 133` and `game_controls.py:72, 83` calibrate the full model's scores on **train
  drafts**, whose games trained the model. In-sample scores are more spread out and more predictive, so the
  balance between score and rank is off on test.
- `trd` also lacks `& has`. Harmless, since only about 0.2% of no-deck drafts have W+L > 0 and they are
  zero-weighted otherwise.
- The headline draft numbers (v4_draft and v2) use cross-fitted scores, so they are unaffected.
- Fix: calibrate on val drafts, or use the cross-fitted scores.
- Impact: the "draft spearman" column of `deck_synergy_v4/results.json`, and the game_controls comparisons, can
  shift by about 0.001-0.003. The variant ranking is probably unchanged.

### 7. LOW-MEDIUM: constant or mis-bucketed covariates (verified on MSH, BLB, FIN, LCI)
- `opp_rank` is the string "None" in 100% of games, so `rank_index` returns 0 for all of them
  (deckstrength.py:128). The result is 1 constant-one column, collinear with the `cov` bias, and 6 all-zero
  columns. This is harmless for predictions and the cov weights aren't penalised. `game_controls.py:53` already
  drops these columns. Fix: drop them everywhere.
- `opp_num_mulligans` is populated (0-3+), so `mull` is fine.
- `user_n_games_bucket` takes values 1/5/10/50/100/**500/1000**, but `N_GAMES_BUCKETS` stops at 100
  (deckstrength.py:131-139). 500+ users (4% of MSH games, 9% of FIN) go into "unknown". No games are truly
  unknown, so slot 0 is effectively a "500+" column, and the result is a mislabel rather than lost information.
  Fix: add "500" and "1000".
- `build_index` is a String, so `most_played_deck`'s tie-break sort (deckstrength.py:106) is lexicographic
  ("10" < "2"). This only matters for builds 10 and up. Fix: cast to Int.
- `game_number` is constant at 1, apart from 3-4 stray rows. `user_game_win_rate_bucket` has 127-254 nulls,
  filled with the whole-period mean. Negligible.

### 8. LOW-MEDIUM: recency weighting in improve_deck_v2 is correct but not normalised
- `recency` (improve_deck_v2.py:63-70) computes age as an index position from 0 (newest fit draft) to 1
  (oldest). Drafts are time-sorted, so this is correct.
- Two caveats:
  - (a) The weights aren't normalised. The mean weight is 0.79 at the chosen tau=2 and 0.25 at tau=0.25.
    `lambda_l2` and `min_sum_hessian` act in weighted-hessian units, so a smaller tau also means stronger
    regularisation. That confounds the tau choice (val NLL differences were only about 1e-5 to 6e-5).
  - (b) Age is measured in draft count, not time. Weekend surges compress time.
- The refit recomputes age over train+val, so the newest draft is the end of val. That is consistent.
- Tree scaling `n * trva/tr` (line 115) is a heuristic and ignores the weights.
- Fix: `w /= w[fit].mean()`; use `draft_time` for age.
- Impact: ≤0.001. Recency alone was within noise.

### 9. LOW-MEDIUM: deck choice and missing games in most_played_deck
- The deck is the build with the most games, ties broken by lowest build_index (deckstrength.py:102-113).
  Rebuilds tend to follow bad starts, so which build gets chosen depends on the run's outcomes (19-20% of
  drafts have more than one build; 17.7% choose a build other than build 0). The rule is the same in every
  split and the outcome is not a feature, so this is not leakage. It is information you would not have at
  registration.
- Drafts without games are excluded, which drops more test drafts (2.2%) than train drafts (1.0%). The
  end-of-window effect explains the difference.
- Fix: run a sensitivity check with the build-0 deck, or weight each draft's builds by their games.
- Impact: probably within ±0.003 Spearman.

### 10. LOW: `deck_features` centres on the batch mean (cardinfo.py:133-134, 159-160)
- `curve_c`, `mv_mean`, `creatures` and `removal` are centred on the mean of whatever rows are passed in. That is
  all games including test games at game level, and all drafts including empty decks at draft level. It is
  transductive (unsupervised) and inconsistent between the two call sites.
- Measured: the standardised features have a mean of ≤0.03 sd on train drafts (`draft_inputs.npz`).
- Fix: pass training means as a reference.
- Impact: negligible.

### 11. LOW: PickNet features drift over time; alignment isn't asserted
- `feats.npz` saves `split`, but `picknet_inputs` (improve_deck_picknet.py:40-46) never checks it against
  `d.split`. I verified that they agree today. Fix: add an assert.
- The surprise features drift: non-forced NLL falls from train to test (MSH 0.857, 0.842, 0.836; FIN 0.844,
  0.822, 0.814). This is meta convergence, not overfitting, but the trees learn absolute thresholds on the
  train distribution. Fix: rank-transform or standardise within a time window.
- `state` is taken at the last valid slot (picknet_features.py:94). That is the state *for* the last pick, so
  the doc's "after the last pick" is slightly off. This is semantics only.
- Impact: may lower "+ surprise" a little. Not optimistic.

### 12. LOW: CIs that treat clustered data as iid
- The game-level `vs ref ±95%` (deck_synergy_v4.py:135) uses the per-game sd/√n, but games cluster within
  drafts and users. The CI is probably 1.3-2x too narrow.
- The draft bootstraps ignore repeat users, which is impossible to fix without user IDs.
- Fix: bootstrap by draft for the game-level differences.

### 13. LOW: variance scripts
- `variance_explained.py` puts free linear terms in record-before (`G`, line 93). That variable is
  endogenous to the random effect (an initial-conditions problem), so s0 is weakly identified. The reported
  "game FM alone" share is **-0.14**, a sign the estimator is ill-posed. Retire its shares in favour of
  `matchmaking_variance.py`.
- The `variance_preds.npz` cache (variance_explained.py:71-73) never invalidates, and matchmaking_variance
  depends on it. Fix: key it by input mtime or hash.
- `matchmaking_variance.py:138`: `has = isfinite(preds)` is always True, because no-deck drafts get finite
  predictions. No-deck test drafts with W+L > 0 enter `k`. They have no game-file games, so the likelihood is
  unchanged, but the "test drafts" count is slightly inflated. Fix: use `most_played_deck`'s `has`.

### 14. LOW: draft_inputs.npz has no draft_id
- Downstream scripts check alignment only with `has == inp["has_deck"]` (deck_synergy_v4_draft.py:56 and
  others). I verified today that D and `has` match `load_drafts` for MSH, BLB, FIN, LCI and EOE.
- Fix: save `draft_id` in the npz and assert it equals `d.draft_id`.

### 15. Minor notes (no action needed)
- The game FM's val-draft scores come from a model early-stopped and grid-picked on val games, so the game FM
  looks slightly better on val. The blend alpha is therefore biased slightly toward the FM. Conservative for
  test.
- Temporal overlap between splits:
  - 1.2% of train-draft games are played after val starts.
  - 0.1-0.14% of train-draft games are played after test starts.
  - 4% of val-draft games are played after test starts. This only matters for the train+val refits.
  - Negligible.
- `record_onehot` as a game-level control is post-treatment: the record is partly caused by the deck.
  game_controls tests this, once finding 1 is fixed.

---

## Checks run (all < 1 s each, column scans only)
- Split alignment across load_drafts, load_tensors and PickNet shards on 9 sets.
- Covariate nulls and constants on 4 sets.
- Game file vs draft-file record, rebuilds and ties on MSH and FIN.
- Cross-fit fold score statistics on MSH, FIN and EOE.
- feats.npz statistics by split.
- TradDraft train-window end dates.
- Overlap between split boundaries and game times.

The check script is in the session scratchpad (`split_check.py`), not in the repo.
