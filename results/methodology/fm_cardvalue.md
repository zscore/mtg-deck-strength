# Can one game FM recover blend*'s gain?

**Question.** blend* = (1 − α) z_LightGBM + α z_FM beats the champion game FM by +0.13 on dev and +0.23 on vault
(×1000 nats/game, SC-GLL experienced-dl). blend_puzzle.md traced most of that to LightGBM's per-card valuations:
counts-only LightGBM recovers 77% of it. Can a single game-level FM pick up those valuations and match blend*?

**Answer: no.** The best single-FM arm, a joint game + draft-record likelihood (draft aux λ = 0.1), gains
+0.036 [+0.015, +0.056] on experienced-dl, 10/12 sets. That is about 27% of blend*'s +0.134, and the gain is gone on
match 1 (+0.000). Every arm stays 0.09–0.13 below blend* on every key, and no arm beats blend* in more than 3/12 sets.

A follow-up probe shows why. The part of LightGBM that helps is **not linear in card counts**. Its best linear-in-counts
projection (r = 0.88 with LightGBM on test) adds +0.004 in blend*, while LightGBM itself adds +0.100. An FM whose card
powers sit on LightGBM's marginal card values (r 0.99 with them) adds only +0.034 even as a blend partner. So
blend_puzzle's "per-card valuations" should read "LightGBM's non-additive count structure". The FM's additive card
power plus rank-4 pairs can't express it, whatever its card values are.

**Recommendation: ship blend\*** (frozen pipeline H5, vault-confirmed +0.233, 6/6). Keep the champion FM as its game
member and as the interpretable card-power model. A single FM is not a substitute. If one model is ever required,
`draft aux 0.1` is the best single FM found (+0.036 dev), but it has no vault test and fails the match-1 sign check.

Scripts: `scripts/fm_cardvalue.py` (prep / arms / meta) and `scripts/fm_cardvalue_probe.py`. Per-set output is in
`results/<SET>.PremierDraft/fm_cardvalue/{lgbm_prep.npz, results.json, scores.npz, probe.json}`, and the metas are in
`results/meta/fm_cardvalue.json` and `results/meta/fm_cardvalue_probe.json`. The vault was not touched.

## Setup

- **Champion.** game_fm_v2's recipe: fm4 nonland + deck features + aff with the champion controls. The (l2_w, l2_v)
  grid is picked on val game log loss, with early stopping on val game log loss. The champion is refit in-process, and
  seed 0 reproduces game_fm_v2's saved champion to 5e-8 on all 12 sets. Every arm changes one term and keeps the
  champion's l2_v.
- **LightGBM inputs** come from a torch-free prep process: a counts-only LightGBM in the round 4 settings, fit on
  train drafts' main builds and early-stopped on val.
  - **c_j:** the marginal log-odds per copy, i.e. the mean over train decks holding card j of logit f(x) − logit f(x − e_j).
    It correlates 0.89–0.92 with the champion's card power w.
  - **Distillation target:** a 2-fold cross-fitted logit on train drafts.
- **Arms:**
  - l2 grid: l2_w ∈ {1e-6, 3e-6, 1e-5 = champion, 3e-5, 1e-4}, plus an extension to 3e-4 and 1e-3.
  - draft aux: + λ × the per-game binomial draft-record NLL of the same deck-only score on train drafts. Calibration is
    r[rank] + β·s, learned jointly. λ ∈ {0.1, 0.3}, plus an extension to 1.0.
  - lgbm prior: w is penalised toward κc and initialised there. κ is the sd-matched scale, about 0.9–1.06. l2 ∈
    {1e-5, 1e-4, 1e-3}, plus an extension to 1e-2.
  - distill: + γ × the MSE between the deck score and the LightGBM target, in the champion's train-sd units. A MSH pilot
    at γ = 0.1 / 0.3 swamped the game likelihood (−0.35 / −0.42), so the grid is γ ∈ {0.003, 0.01, 0.03, 0.1}.
- **Picks.** Each family's `[pick]` is chosen on val SC-GLL-dl at seed 0 (two-half cross-fit, as for blend*). The l2
  grid is also picked on val game log loss.
- **Seeds.** Every setting has seed 0. The champion and the picked settings also have seed 1.
- **Scoring.** The frozen T1 contexts dl and pre, with subsets experienced and match 1. Paired day-block bootstraps, and
  an RE meta (DL + HKSJ) over 12 sets.
- **blend\*** is the saved T1 blend\* (3 seeds). Rebuilding it from the saved LightGBM, T1's α\* and this run's champion
  gives exactly the same result (Δ 0.0000), so the comparison is like for like.
- **Gate.** The first three sets passed the 3-set gate (draft aux [pick] +0.052 and distill [pick] +0.068 ≥ +0.05 on
  MSH/FDN/BLB), so all 12 dev sets were run.

## Results (12 dev sets, ×1000 nats/game, RE mean [95% CI], sets better)

| arm − champion | experienced dl (primary) | experienced pre | match 1 dl |
|---|---|---|---|
| **blend\* (T1, reference)** | **+0.134 [+0.093, +0.174], 11/12** | **+0.173 [+0.139, +0.207], 12/12** | **+0.146 [+0.052, +0.241], 10/12** |
| l2 [val-gll pick] | −0.000 [−0.002, +0.002], 4/12 | −0.000 | +0.001 |
| l2 [val-scgll pick] (unweighted; = champion in OTJ) | +0.004 ± 0.005, 6/12 | +0.001 | +0.016 ± 0.010 |
| l2 1e-4 / 3e-4 / 1e-3 | +0.004 / +0.004 / −0.044 (2/12) | +0.003 / +0.002 / −0.051 | +0.002 / +0.026 / +0.016 |
| draft aux 0.1 | **+0.036 [+0.015, +0.056], 10/12** (p1 0.0015) | +0.045 [+0.018, +0.072], 10/12 | +0.000, 7/12 |
| draft aux 0.3 / 1.0 | +0.019 (8/12) / −0.056 (4/12) | +0.061 / +0.014 | −0.047 / −0.126 |
| draft aux [pick] | +0.026 [+0.004, +0.048], 9/12 | +0.045 [+0.011, +0.078], 10/12 | −0.019, 2/12 |
| lgbm prior 1e-4 / 1e-3 / 1e-2 | +0.005 / +0.022 (9/12) / −0.035 | +0.003 / +0.019 / −0.030 | +0.003 / +0.005 / −0.062 |
| lgbm prior [pick] | +0.003 [−0.014, +0.021], 7/12 | +0.002 | +0.002 |
| distill 0.003 / 0.01 / 0.03 / 0.1 | −0.035 / −0.126 / −0.361 / −0.572 | −0.021 / −0.077 / −0.289 / −0.487 | −0.039 / −0.139 / −0.404 / −0.593 |
| distill [pick] (0.003 in 11/12) | +0.005 [−0.039, +0.050], 7/12 | +0.026 | −0.025 |

Against blend\*, every pick loses on experienced-dl: l2 −0.128 to −0.130, draft aux −0.094, lgbm prior −0.116, distill
−0.128 (0–1/12 sets better). Experienced-pre and match-1-dl agree, at −0.11 to −0.18.

Per set, experienced-dl:

| set | blend\* − champion | draft aux 0.1 − champion |
|---|---|---|
| MSH | +0.26 | +0.09 |
| EOE | +0.13 | +0.04 |
| SOS | +0.09 | +0.01 |
| TDM | +0.05 | −0.03 |
| FDN | +0.13 | −0.02 |
| DFT | +0.18 | +0.05 |
| LCI | +0.12 | +0.02 |
| FIN | +0.13 | +0.08 |
| MKM | +0.21 | +0.18 |
| BLB | +0.14 | +0.04 |
| DSK | −0.00 | +0.03 |
| OTJ | +0.15 | +0.06 |

### Did the arms move the FM toward LightGBM? Yes, and it doesn't help

Means over 12 sets:

| arm | r(w, champion w) | r(w, LightGBM c) | r(test score, LightGBM counts) | r(test score, blend\*) |
|---|---|---|---|---|
| champion | 1.000 | 0.906 | 0.877 | 0.989 |
| l2 1e-3 (w sd ×0.72) | 0.968 | 0.868 | 0.871 | 0.984 |
| draft aux 0.1 / 1.0 | 0.998 / 0.976 | 0.915 / 0.936 | 0.883 / 0.898 | 0.989 / 0.975 |
| lgbm prior 1e-3 / 1e-2 | 0.986 / 0.943 | 0.952 / **0.991** | 0.885 / 0.897 | 0.989 / 0.982 |
| distill 0.003 / 0.1 | 0.991 / 0.913 | 0.935 / 0.957 | 0.903 / 0.943 | 0.986 / 0.942 |

- The arms can make the FM's card values match LightGBM's almost exactly: lgbm prior 1e-2 reaches r 0.99.
- Doing so moves the FM's deck score away from blend\*, not toward it, and the gain doesn't follow.
- Stronger card-power L2 does what blend_puzzle described (it shrinks strong cards), and it gains nothing: at most
  +0.004, then −0.044 at 1e-3.

## Probe: what in LightGBM carries the gain? (`fm_cardvalue_probe.py`, partner blended with the champion as blend\*)

| partner | blend − champion, exp-dl | match 1 dl | mean α\* |
|---|---|---|---|
| LightGBM counts-only | **+0.100 [+0.063, +0.138], 12/12** | +0.114, 11/12 | 0.76 |
| its linear-in-counts projection (ridge; r 0.88 with LightGBM on test) | +0.004 ± 0.009 | +0.011 ± 0.005 | 0.96 |
| FM with card power on LightGBM's c (lgbm prior 1e-2) | +0.034 ± 0.017 | +0.046 ± 0.024 | 0.70 |
| distilled FM (0.003) | +0.019 ± 0.010 | −0.001 ± 0.016 | 0.56 |
| the champion's own seed-1 refit (pure ensembling) | −0.000 ± 0.004 | +0.009 ± 0.008 | 0.62 |

(Partners whose α\* = 1 in some set are reported as an unweighted mean ± se.)

Three things follow:

1. **Not ensembling.** Averaging two FMs gains nothing.
2. **Not additive card values.** A linear projection of LightGBM, or an FM that adopts LightGBM's marginal card values,
   recovers at most a third of the gain.
3. **The useful signal is in LightGBM's non-additive use of the counts.** This is the 12–23% of its variance that no
   additive model reaches. It could be thresholds or saturation on copies, high-order card interactions, or interactions
   with basic-land counts. The rank-4 nonland FM and its card power can't represent it.

This corrects the wording in blend_puzzle.md. The gain is in the counts, but not in per-card values: blend_puzzle's
partial R² of 0.34 was from a linear ridge of d, which explains d's variance but not its predictive value.

## Caveats

- **Seeds.** Single-seed rows (the non-picked settings) carry no seed variance. The picks and the champion have 2
  seeds; blend\* has 3.
- **Extension grid points** (l2 3e-4 / 1e-3, aux 1.0, prior 1e-2, and distill's lower γ) were added after the MSH
  pilot. They enter the family picks, and the pre-registered points are reported separately above.
- **draft aux 0.1** is the only setting that passes `decide()` on experienced-dl, but val SC-GLL-dl picked it in only 5/12
  sets. Its match-1 delta is 0, and its experienced-pre gain is partly the draft record, which has no
  skill control. Treat it as "small, possibly real", not as a candidate replacement.
- **Untested next steps:** a non-additive count encoding in the FM (per-copy card indicators, i.e. a 1st/2nd/3rd-copy
  w, and land-count interactions), or a gradient-boosted residual on the FM's deck score. The probe points there, not at
  card values.

## Runtime

Serial, nice 10, 4 threads, torch and LightGBM in separate processes:

| step | wall time | peak RSS |
|---|---|---|
| prep (counts-only LightGBM, 3 fits + marginals per set) | about 12 min total, 22–80 s per set | 3–4 GB |
| arms (about 25 FM fits per set + scoring) | 70 min total, 233–518 s per set | 4.7–5.6 GB; OTJ 6.4 GB |
| probe | about 6 min | — |
| metas | under 1 min | — |

Total is about 1 h 45 min of wall time.
