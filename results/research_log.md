# Research log: deck-strength model (autonomous loop, started 2026-10-12)

Resource rules: one heavy job at a time, niced, ≤4 threads.

## Done before the loop (2026-10-12)
- Round 10 cheap LightGBM tweaks (scripts/improve_deck_v2.py), MSH: refit on train+val + recency (tau=2) + 5 seeds,
  blended with game FM: 0.2882 → 0.2934 (+0.0052 ± 0.0016). Param grid: no gain.
- PickNet features (scripts/picknet_features.py, improve_deck_picknet.py): deck card vectors / pick surprise ≈ +0.002 on
  MSH, BLB, ~0 or negative on DFT, DSK, EOE, FDN; draft state hurts. Pending last sets.
- Variance (scripts/variance_explained.py, matchmaking_variance.py): record-based matchmaking makes total true variance
  unidentifiable from records; model slope 1.2–1.7 (compressed); blend explains ~25–45%, ceiling ~0.45–0.60.
- opp_rank is None in all game files (constant control). Game WR falls with rank; rank = opponent pool.

## Loop iterations
- 18:40 Launched 4 compute-free methodology agents → results/methodology/: matchmaking_identification.md,
  evaluation_design.md (+ eval_protocol.py), pipeline_audit.md, next_model_design.md (+ drafts/).
  Meta-research: each loop iteration a reviewer agent critiques this log and re-ranks the queue.
- Evaluation design done (results/methodology/evaluation_design.md, eval_protocol.py, power_sim.py):
  * raw Spearman rewards skill leakage (synthetic: who-drafted model +0.050, z=25); deck_metrics ceiling invalid → retire.
  * protocol issues: many variants on same test drafts; val reused for tau/grid/alpha; no seed/day-block variance;
    refit arms confound recency; MSH (smallest) used as screening set.
  * proposed primary metric SC-GLL: skill-controlled held-out game log loss with a frozen shared context model and a
    deck-only score; secondary: within-strata Spearman, match-1-only SC-GLL, same-draft multi-build design.
  * decision rule: random-effects meta mean > 0 (HKSJ, Holm), >= 8/12 sets, no set sig. worse, leakage veto.
  * power: 12 sets detect ~+0.0011–0.0017 Spearman at 80%.
  * PickNet deck vec, 9 sets meta: +0.00012 [−0.0010, +0.0012], p=0.81 → noise. Dropped.
  * MSH check: 82% of games from users with ≥50 games; 19% of drafts have >1 build (same-player/different-deck design viable).
- Pipeline audit done (results/methodology/pipeline_audit.md): no test leakage in headline draft Spearman; splits align
  exactly across load_drafts / load_tensors / picknet shards / game_split. Actions taken:
  * picknet pool_d32.pt models are missing everywhere (only json) → affinity can't be rebuilt; backed up all 12
    draft_inputs.npz to results/backup_draft_inputs/. game_controls.py now runs without affinity (same features in all
    variants). TODO: retrain pool models (cheap) to restore affinity.
  * fixed experience(): buckets 500/1000 were falling into "unknown" (4–9% of games). Affects game models refit from now on.
  * open: ~19% of drafts have more logged games than W+L (stale record label); build_index is a string (tie sort);
    cross-fitted train-draft game scores have 15–20% more spread than val/test (may explain boosting < averaging);
    training-seed noise ~0.0015 → treat < 0.002 as unresolved; variance_explained.py numbers retired.
- PickNet features, all 12 sets (eval_protocol meta-analysis, blended with game FM, vs round 4 blend):
  deck vec +0.00048 [−0.00041, +0.00137] p=0.26, 7/12 · surprise +0.00009 p=0.83, 6/12 ·
  state −0.00172 [−0.00279, −0.00066] p=0.005 (Holm 0.023), 2/12, 5 sets sig. worse ·
  deck vec + state −0.00096 · all three −0.00007. VERDICT: no PickNet feature helps; draft state hurts. Closed.
- Matchmaking identification memo done (results/methodology/matchmaking_identification.md, matchmaking_checks.py):
  records under exact record matching are uninformative about absolute spread s (only users' spread relative to s).
  New anchor: Bo1 play/draw is a coin flip; being on the play in game t−1 → extra win → harder opponent in game t.
  MSH −0.10 ± 0.043 (s ≈ 0.85); pooled MSH/EOE/ECL −0.052 ± 0.024 (s ≈ 0.55, CI ~0.1–1.0). Opponent colour-pair
  strength rises with wins (direct evidence of record matching). 12% of runs change rank mid-run (rank offsets
  identified). Time-linking consecutive drafts links ~15% at ~80% precision (same-user design). Win-rate bucket is a
  per-set user total incl. current run; true long-run WR sd ≈ 0.06. Queued matchmaking_checks.py over 12 sets to pool s.
  NOTE: agent's first check peaked at 7.3 GB (bad join), fixed to ~650 MB.
- Next-model design done (results/methodology/next_model_design.md, drafts/): A = structural matchmaking likelihood
  (torch opponents() recursion, s fixed from play/draw, profile s ∈ {0.4,0.55,0.7,0.85}); synthetic: fixes scale
  (0.78–0.82 → 1.03) not ranking (Spearman flat). Real ranking lever = replacing leaky skill bucket (variants "record"
  vs "rated"). ~3 h CPU to screen MSH/FDN/BLB. B = game-level set transformer d=128×3 (1.05M params), frozen PickNet
  vectors + residual, per-set linear path + heads, shared trunk, A's likelihood + distillation to cross-fitted LGBM
  blend logit; ~1–1.5 h per 12-set run on free GPU; grid B1–B3 ≈ 20–35 GPU-h; expected +0.000–0.004.
- Meta-review 1 (results/methodology/meta_review_1.md) adopted:
  * PRIMARY OBJECTIVE now: skill-adjusted game-level deck/card rating. PRIMARY METRIC: SC-GLL-dl (de-leaked skill
    context, users ≥ 50 games) on a deck-only score at a reference rank, decide() over 12 sets, min_effect 0.02 ×1000.
    Vetoes: leakage index, match-1 sign, within-build slope. Spearman vs reconciled W/L = secondary.
  * STRUCK: variance_explained shares (25–45%); deck_metrics ceiling. Round 10's +0.0052 is MSH-only, unreplicated,
    confounded (refit + seeds only in treatment). The test split has been reused ~10 rounds → treat as dev-test; confirm
    final pipeline on untouched sets (ECL, HOB, MOM, NEO, ONE, TLA, TMT, WOE).
  * Stale-label mechanism unverified (also 25–29% of drafts have fewer games in game file). Multi-build count must be
    drafts with ≥2 builds that each played games.
  * Queue rerouted: improve_deck_v2 stops after BLB (replaced by Q2: relabel + 3 paired seeds + saved preds);
    now matchmaking_checks (20 sets) → game_controls (12 sets) → Q0–Q5 when the implementer's scripts land.
- 19:13 two short-lived Python self-test processes (implementer agent) segfaulted: torch+LightGBM libomp clash. Not the
  queue. Told the agent to import lightgbm first / separate processes / cap threads.
- Play/draw anchor, 17 sets (matchmaking_checks.py; NEO/ONE/MOM lack game_time column → skipped): pooled Wald
  −0.0278 ± 0.0063, homogeneous (Q=17.2, df=16) → population strength spread s ≈ 0.36 logit [0.23, 0.44] under exact
  record matching (soft matching biases s down, per memo). Much tighter than MSH alone (s≈0.85, noisy).
  Implication: at s≈0.36, the round 4 blend's explained true sd (~0.30 at s=0.3 fit) is close to s → the blend may
  already capture most of the matchable strength spread; re-check with model A at s∈{0.3,0.36,0.45}.
- Implementer delivered scripts/eval_common.py, rebuild_targets.py, sc_gll.py, structural_a.py (self-tests pass;
  structural_a refuses to run with lightgbm loaded). Queued after game_controls: sc_gll ×12 + meta → rebuild_targets ×12
  → structural_a MSH/FDN/BLB at s ∈ {0.3, 0.36, 0.45}.
- Hand-value design done (results/methodology/hand_value_design.md, scripts/hand_value.py): opening 7 (recorded
  pre-bottoming, incl. mulligans) is a randomized treatment given deck, mulligans and land count; Robinson-style
  residual-on-propensity estimator, unbiased regardless of outcome model; balance index checks keep-selection.
  Synthetic: corr(HV, truth) 0.85, coverage 0.95, GIH WR inflates trap cards +7.9 pp while HV doesn't.
  deck-power / HV slope estimates λ (late-draw value). Queued 12 sets after structural_a (~40–60 min total).
- Game-model controls ablation, 12 sets (scripts/game_controls.py; reference linear + deck features, NO affinity —
  pool models missing; draft Spearman of calibrated deck-only score; mean change vs full):
  * − record-before: −0.0000 Spearman; deck-score sd −0.019 (more compressed); card r 0.986. Keep record control —
    without it, matchmaking compression bleeds into card values (as the matchmaking memo predicts). Not a bad control.
  * − mulligans: +0.0022, better on 12/12 sets; card r 0.990. The mulligan control absorbs deck-caused (mana-base)
    mulligans. CANDIDATE: drop mulligans from the deck-rating model (caveat: mulligan decisions are partly skill;
    confirm on SC-GLL-dl / within-build design before adopting).
  * − record & mull: +0.0022 (same as − mulligans).
  * − skill bucket: +0.0098, deck-score sd +0.04 → the leakage reward predicted by evaluation_design. Not adopted.
- SC-GLL scoreboard, 12 sets (scripts/sc_gll.py; experienced users; ×1000 nats/game, RE meta, decide()):
  * LightGBM − game FM: −0.657 [−0.795, −0.519], 0/12 better, 12 sig. worse.
  * round 4 blend − game FM: −0.206 [−0.318, −0.093], 0/12 better, 6 sig. worse.
  * game FM − linear game model: +0.155 [+0.098, +0.213], 11/12 → pair synergy real on the deck-rating metric. ADOPT.
  * round 4 blend − LightGBM: +0.439, 12/12.
  ⇒ As a DECK RATER the skill-adjusted game FM alone is best (meta-review's prediction confirmed). LightGBM's draft-level
  Spearman gains are record prediction (skill/build-choice signal), not deck strength. Same on match-1-only and
  main-build subsets (MSH). Caveat: SC-GLL is native to game-level likelihoods; LightGBM loses on match 1 too.
  NEW CHAMPION for deck strength: game FM (fm4 nonland + deck features + aff). Next improvements target the game FM.
- CORRECTION: picknet pool_d32.pt models exist for all 12 sets in ../picknet/results/<SET>.PremierDraft/picknet/models
  (the repo split just didn't point there). Symlinked into results/<SET>.PremierDraft/picknet/models; pick_affinity
  loads again (MSH S 339×339). Affinity is restorable; game_controls ran without it (same for all variants).
- scripts/game_fm_v2.py delivered (champion fm4 nonland refit + variants: − mull, rated-dl, − skill, refit, recency;
  SC-GLL experienced / dl-ctx primary; 2 seeds). Queued after hand_value (~2–3 h for 12 sets at 2 seeds).
- Label rebuild (in progress): MSH draft-file record stale 17–21%, game file missing games 25–29% (only 72% of runs
  complete in game file); BLB stale 19–24%, missing 3%, 99% complete. MSH game file is unusually incomplete.
- scripts/game_fm_opp.py delivered: champion + opponent colour-combo effects (+ opp) and centred own×opp matchup
  matrix (+ opp x own) as game context only; reports opponent-strength table, matchup matrix (RPS Q test), SC-GLL incl.
  opp-aware context subsets. Synthetic: opp strength r 0.98, matchup r 0.92. Queued after game_fm_v2 (~2 h).
- Label rebuild, 12 sets (scripts/rebuild_targets.py): draft-file record stale (game file has more games) on 16–24% of
  test drafts in every set; game file MISSING games on 26–31% for MSH, EOE, FIN, SOS, TDM (3–4% elsewhere).
  Meta over 12 sets (Spearman, 1 seed per arm):
  * round 10 (refit train+val, recency tau=2, 5 seeds, + game FM) − round 4 blend: +0.0023 [+0.0014, +0.0033] with
    draft targets, +0.0028 [+0.0020, +0.0036] with game targets; 12/12 both. REPLICATES (smaller than MSH's +0.005).
    ADOPT for record prediction (caveat: seed averaging only in treatment arm).
  * training on game-file targets vs draft-file: blend +0.0004 (p=0.46), LightGBM +0.0012 (p=0.16), round 10 +0.0006
    (p=0.02, 10/12). Small. Evaluating on game-file targets raises every Spearman (~+0.013) — label quality, not model.
  * New record-prediction best: round 10 trained+evaluated on game targets, mean Spearman 0.2946 (vs 0.2783 round 4
    on draft targets; not comparable across targets).
- Model A (structural matchmaking likelihood, linear reference scorer w/o FM pairs; MSH/FDN/BLB; SC-GLL experienced):
  * A record − A iid ≈ 0 (−0.011 … +0.013) at every s → structural likelihood does NOT change ranking (as predicted).
  * Scale: A record deck scores 1.12–1.17× the game FM's at s=0.36 (1.17–1.24 at 0.45) → decompression ~15%.
  * A rated (de-leaked skill bucket) − A iid/record: +0.09 … +0.29 ×1000 on all 3 sets → skill adjustment with the
    de-leaked bucket helps deck rating. Key open question (game_fm_v2 rated-dl): de-leaked vs RAW bucket on the FM.
  * A (no pairs) remains below the game FM (−0.2 … −0.6), consistent with pairs mattering.
  VERDICT: use A only for calibrated/decompressed scale; not a ranking improvement. Transformer B with A's likelihood
  is therefore not expected to gain from the likelihood itself.
- 22:20 Overnight: caffeinate -dimsu for 14 h (AC power). Queue: hand_value ×12 → game_fm_v2 ×12 (2 seeds) →
  game_fm_opp ×12. Implementer productionizing transformer B (scripts/game_transformer.py: PickNet-init set
  transformer + parallel FM path, iid likelihood, optional distillation) to queue on MPS at the end.
- scripts/game_transformer.py delivered (FM path warm-started from champion + set-transformer residual over PickNet
  card vectors; PMA + copy-weighted pooling; optional distillation). Synthetic: recovers a non-FM effect (r 0.77), FM
  path alone no gain. Queued last: MSH/FDN/BLB × {picknet, random} × {distill 0, 0.5}, 2 seeds, MPS (~4.5–5 h).
- Hand values, 12 sets (scripts/hand_value.py; randomized opening-hand card effects). Mean over sets, correlation with
  held-out TEST-game hand values (r) and "skill tilt" (how much of a rating beyond HV tracks drafter skill):
    HV (train games, shrunk)  r 0.78  tilt 0.22   ← replicates train→test; cleanest card ratings
    OH WR (17Lands)           r 0.73  tilt 0.61
    GIH WR                    r 0.62  tilt 0.68
    IWD                       r 0.59  tilt 0.39
    GP WR                     r 0.51  tilt 0.72
    game FM card power w      r 0.44  tilt 0.08   ← least skill-tilted; but measures deck value, not hand value
  * deck-power vs HV slope ~0.01–0.05 (implied lambda < 0: deck model's per-card w are heavily shrunk/absorbed by deck
    features; additive mapping doesn't hold — needs a look).
  * HV as a ridge prior for card power: +0.006 ± 0.013 (12-set inverse-variance mean) → no gain. Not adopted as prior.
  ⇒ HV is a strong, skill-clean card-rating product in its own right (beats every 17Lands stat at predicting held-out
    hand effects, with much less skill tilt), but doesn't improve deck prediction.
- game_fm_v2, 12 sets × 2 seeds, SC-GLL experienced (raw-bucket ctx) [dl-ctx in brackets], ×1000 nats/game:
  * refit on train+val − champion: +0.064 [+0.033, +0.094], 10/12, 0 worse → ADOPT [dl-ctx +0.036, 10/12, adopt].
  * − mulligans: −0.060, 0/12 [−0.041] → KEEP mulligan control. (Its +0.0022 draft-Spearman gain was record-side.)
  * − skill: −0.374, 0/12, 11 worse [−0.134] → skill control essential; Spearman's +0.0098 was pure leakage reward.
  * rated-dl (de-leaked bucket) − champion: −0.219, 1/12 [−0.037, n.s.] → raw bucket stays. Model A's "rated" gain
    was vs. no skill control, not vs. the raw bucket.
  * recency: −0.002 (n.s.) [−0.001]. champion reproduces cached FM (+0.001).
  ⇒ DECK-STRENGTH CHAMPION v2: game FM (fm4 nonland + deck features + aff, current controls) REFIT ON TRAIN+VAL.
- game_fm_opp, 12 sets × 2 seeds: + opp −0.038 (1/12) on standard ctx; ≈0 under opp-aware ctx; matchup matrix +0.0005
  (n.s.). NOT adopted. Reason found: opp_colors are colours REVEALED during the game (post-treatment) — mono-colour
  "opponents" are short games (W: user WR 72%, −0.69 logit), 5-colour ones are long games. Not a valid control.
  Opponent strength still rises with wins before the game (record matching, again).
- Transformer queue crashed after training on a float64→MPS cast (game_transformer.py:367, and the distill target at
  :430). Fixed both; MSH smoke (1 seed, 2 epochs, distill 0.5) runs end-to-end in 57 s, 3.1 GB. Requeued 4 configs ×
  MSH/FDN/BLB × 2 seeds at 00:57.
- Meta-review 2 (results/methodology/meta_review_2.md) — corrections adopted:
  * SC-GLL "home advantage": the judge's context (raw bucket, own−opp mulligans) matches the champion's controls;
    swapping raw→de-leaked context moves paired deltas by ~0.18–0.24. ROBUST: game FM ≫ LightGBM (−0.66, every
    subset) and FM pairs (+0.155). NOT ROBUST: "round 4 blend worse than FM" (−0.21; blend weight was tuned on record
    NLL), "keep mulligans" (own mulligans are a mediator → judge artefact), raw vs de-leaked bucket (tie on dl-ctx).
  * refit train+val: +0.064 raw-ctx / +0.036 dl-ctx (p=0.035, fails Holm over 5 arms) → PROVISIONAL; vault decides.
  * opp_colors post-treatment VERIFIED (1 colour revealed −0.88, 74% user WR; 5 colours +0.31, 47%; unknown 96%).
    The matchup RPS test and the "opponent strength rises with wins" check from game_fm_opp are contaminated — STRUCK.
  * hand_value.py BUG: the deck model whose w is compared with HV includes the round-4 FM deck score as an input, so w
    is residual power beyond the FM; ridge at grid edge 11/12. STRIKE the "game FM card power w: r 0.44, tilt 0.08"
    row, the deck-vs-hand slope / lambda, and the HV-prior verdict. HV-vs-17Lands comparison is partly circular
    (target = test HV); keep-selection flag fires on 12/12 sets (core-subset check r 0.96–0.99 holds).
  * Record model best = round 10 recipe; its FM member is the old FM, not champion v2.
  * Vault: ECL, TLA ready; HOB, MOM, TMT, WOE need picknet pool models/shards (or no-aff fallback); NEO, ONE excluded.
  * Plan: T0 counts → T1 rescore all saved scores under raw / dl / "pre" (no own-mulligan) contexts + blend tuned on
    val SC-GLL-dl → T2 own/opp mulligan split arms → T3 within-build conditional logit → T4 hand_value fix → T5 vault
    (pre-registered frozen pipeline) → T6 write-up. Transformer B2/B3 only if ≥ +0.02 on val SC-GLL-dl.
- T0–T5 scripts delivered (scripts/t_common.py, t0_counts.py, t1_rescore.py, t2_mull_split.py, t3_within_build.py,
  hand_value_v2.py, t5_vault.py; results/vault/RUNBOOK.md). T0 on MSH: 1,799 / 715 / 208 qualifying multi-build
  test drafts at k=1/2/3; num_turns rises with opponent colours revealed 3.9 → 10.9 (opp_colors post-treatment, closed).
  PRE-REGISTERED (from meta_review_2/RUNBOOK): T1 primary = FM vs LightGBM under dl & pre; blend* − FM < +0.02.
  T2 primary key "experienced pre": adopt "− own mull" iff ≥ +0.02, decide() passes, and T3 doesn't lower b.
  T3: decide on pairs (b per within-draft sd), not b vs 1. T4: HV external claim iff HV-sum − GIH-sum ≥ +0.02 on
  experienced dl. Queued T0→T4 after the transformer; decision checkpoint + freeze + T5 vault by hand afterwards.
- Transformer B (game_transformer.py), MSH/FDN/BLB × {picknet, random} × {distill 0, 0.5} × 2 seeds, SC-GLL
  experienced: transformer − champion −0.0004 / −0.0049 / −0.0186 / +0.0000 (all n.s.); transformer − its own FM path
  ≈ +0.00000–0.00003 → the transformer residual learned ~nothing on real data (synthetic test showed it can). Either
  the FM already spans what's learnable here, or optimisation (zero-init readout + early stopping at warm start)
  never left the champion. Peak RSS 5.8 GB on FDN/BLB. VERDICT: no gain; consistent with literature & per-set
  transformer history. Possible follow-up (low priority): unfrozen/no-warm-start residual with longer schedule.
- Queued after T4: transformer residual-only test (FM frozen, lr 3e-3, picknet init, 1 seed, MSH/FDN/BLB). Joint runs' residual sd stayed 0.000–0.002 (MSH val gain came via the FM path; FDN never beat warm start).
- T0–T4 results (12 dev sets; ×1000 nats/game unless noted):
  * T1 judge contexts: FM ≫ LightGBM robust under every context (raw −0.66, dl −0.40, pre −0.29, match-1 pre −0.32;
    0/12). FM pairs robust (+0.14–0.16, 11/12). Champion v2 refit: raw +0.064, dl +0.036 (10/12), pre +0.033 (p=0.06).
    Round 4 blend vs FM: raw −0.21 → dl −0.01 → pre +0.07 (judge-dependent, as meta-review 2 said).
    *** blend* (LightGBM + FM, α tuned on VAL SC-GLL-dl, two-half cross-fit): − FM = +0.132 [+0.099, +0.165] dl 11/12,
    +0.161 pre 12/12, +0.176 match-1 pre 10/12 (with champion v2: +0.132 / +0.172 / +0.180). Large and robust. α* ≥ 0.8
    in only 2/12 sets → LightGBM carries real weight.
  * T3 within-build (same draft, different builds; b per within-draft sd): champion − LightGBM −0.002 (n.s.) → LightGBM's
    deck differences predict within-player outcomes as well as the FM's → its blend contribution is not just skill.
  * T2 mulligans: − own mull (opp mull kept) − champion: pre +0.050 [+0.034, +0.066] 11/12; dl −0.020; match-1 pre +0.037
    (n.s.). T3 veto: state control p=0.275 (pass), match control p=0.042 (fails sensitivity). Split mull ≈ 0.
    → meets the pre-registered rule on the primary control; logged as DEV-ADOPTED (not vault-tested, see below).
  * T4 hand values (fixed): FM card power vs HV slope 0.17–0.34, implied λ 0.07–0.54 (late draw worth ~30–50% of
    opening-hand presence) — puzzle resolved. HV-sum − GIH-sum as deck score: −0.15 (n.s., 5/12) → external claim
    FAILS; HV stays a card-rating product only. Clean deck-model w sum ≫ GIH sum (+1.45, 12/12). HV prior: −0.001 (closed).
- DECISION CHECKPOINT (05:05): freeze with H5 = blend* (largest dev effect, the open question from meta-review 2;
  freeze supports one H5 arm). "− own mull" recorded as dev-adopted, vault-untested. Vault next.
- Frozen results/vault/frozen_pipeline.json (sha256 87155c5019ac…, H5 = blend*). Vault queued after the residual-only transformer: ECL, TLA (aff), WOE, MOM, HOB, TMT (no-aff), then report.
- Transformer residual-only (FM frozen, lr 3e-3, picknet init; MSH/FDN/BLB): no epoch improves val over the champion
  (MSH ±0.00001, FDN +0.00003…+0.00023) → restores the champion; test Δ −0.0000/−0.0009/−0.0000. With the FM frozen
  the residual is free to learn, so this is real evidence: no learnable non-FM deck signal at this data size. CLOSED.
- VAULT (6 untouched sets ECL, TLA, WOE, MOM, HOB, TMT; frozen pipeline 87155c50…; SC-GLL experienced-dl ×1000):
  * H5 blend* − champion v2: +0.233 [+0.172, +0.294], p1 = 0.0001, 6/6 → CONFIRMED (dev was +0.13).
  * H1 FM − LightGBM: +0.264 [+0.113, +0.415], p1 = 0.003, 5/6 → CONFIRMED.
  * H2 FM − linear: +0.218 [−0.036, +0.472], p1 = 0.039, 5/6 → CONFIRMED by frozen rule; decide() fails (HKSJ CI).
  * H4 round 10 − round 4 (Spearman): +0.0040 [+0.0020, +0.0059], p1 = 0.0016, 6/6 → CONFIRMED.
  * H3 refit − train-only: vault +0.005, p1 = 0.43, 4/6 → vault alone does NOT confirm; frozen H3 rule (dev+vault pooled
    +0.027, p1 = 0.023) labels it CONFIRMED. Treat as "small, probably positive".
  * Affinity-only group (k=2) is descriptive (HKSJ 1 df), as pre-declared.
- Open puzzle → next: transformer finds NO non-FM game-level signal, yet draft-trained LightGBM adds +0.23 in blend*.
  Either deck structure the game likelihood misses, or player-level info the de-leaked bucket leaves. Writing T3 for
  blend* (within-build) + attribution of LightGBM − FM.
- Blend puzzle (results/methodology/blend_puzzle.md; t3b_blend.py, lgbm_attrib.py, blend_strata.py, blend_probe.py):
  * Within-build (T3 for blend*): blend* − champion +0.0015 [−0.0019, +0.0048] (state), −0.0003 (match); veto not
    triggered (p 0.83 / 0.40). Low power (champion's own pooled b +0.0096 ± 0.006) → "not down", not "up".
  * d = z_LGBM − z_FM: card counts explain it (partial R² +0.34); structure/curve +0.001; rank 0; player 0; pool +0.001.
  * Feature-subset LightGBM in blend*: full +0.130, rank removed +0.132, COUNTS ONLY +0.100 (12/12, 77% of gain),
    structure+deck features+aff +0.008, structure only −0.004. → the gain is LightGBM's per-card valuations.
  * LightGBM card coefs ≈ −0.86·FM power + 0.53·deck_skill + 0.02·HV (std.): shrinks the FM's strong/rare cards,
    promotes cards skilled drafters play. Gain holds for users with ≥500 games (+0.18 ± 0.04) where the bucket is
    precise; skill-marker card-sum partners give +0.003 → not a proxy for unmeasured skill (caveat: promoted cards
    lean skill-associated; within-build can't confirm positively).
  ⇒ Verdict: mostly deck signal through card valuation. Next: move the FM's card values toward LightGBM's (weaker L2
    on card power, draft-record auxiliary loss, or LightGBM card coefficients as a prior) and see if the FM alone
    recovers the gain.
- FM card-value recovery (scripts/fm_cardvalue.py, fm_cardvalue_probe.py; results/methodology/fm_cardvalue.md), 12 dev
  sets, SC-GLL exp-dl vs champion: blend* +0.134 (ref). l2 grid +0.004; LightGBM-prior +0.003 (r 0.99 with LightGBM's
  per-card log-odds yet no gain); distill +0.005; draft-record aux 0.1 +0.036 [+0.015, +0.056] 10/12 (passes decide,
  ~27% of blend*, 0 on match 1, not vault-tested). Every arm 0.09–0.13 below blend* (0/12).
  Probe: counts-only LightGBM partner +0.100; its best LINEAR-in-counts projection (r 0.88) +0.004; FM with LightGBM's
  card values +0.034; champion seed-1 refit as partner −0.000 (not an ensembling effect).
  ⇒ CORRECTION to blend_puzzle: the gain is LightGBM's NON-ADDITIVE use of counts (copy counts / thresholds /
  interactions), not per-card values. SHIP blend* (vault-confirmed). Next idea: non-additive count encoding in the FM
  (per-copy indicators, land-count interactions) or a boosted residual on the FM score.
- Game-level boost (scripts/game_boost.py; results/methodology/game_boost.md), gate MSH/FDN/BLB, SC-GLL exp-dl vs
  champion: blend* +0.175 (3/3); A game-level LightGBM residual on the champion's game logit +0.021 (residual sd
  0.001–0.024, early stops at 1–142 trees); B draft-record residual on the champion +0.011 (per set +0.116 / −0.041 /
  −0.082; match-1 −0.249; pre +0.098 → record-side); FM + per-copy indicators + land×spell-count buckets +0.013 (less
  shrinkage only overfits). Gate FAILED; 9 sets not run.
  ⇒ The extra signal is NOT game-level deck structure, and NOT "any draft-record model": only an independently trained
  draft-record LightGBM mixed at a val-SC-GLL-tuned weight (blend*) gets it. Untested caveats: cross-fitted offset for
  B; B re-weighted on SC-GLL (numpy only). Status: SHIP blend*; mechanism partly open.
- 09:15 loop wrap-up: main objectives complete (champion identified and vault-confirmed; record model confirmed;
  transformers/PickNet/structural/opp-context closed). Summary in reports/Deck strength overnight results.md.
