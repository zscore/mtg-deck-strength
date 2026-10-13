# Runbook: meta_review_2 T0–T5 (serial, niced, ≤ 4 threads)

Plan: `results/methodology/meta_review_2.md`.

Every command below runs from the repo root. Run them **one at a time**, in this order. Every heavy step waits until the
transformer queue (PID 30099, MPS) has exited.

Each script has a `--selftest`. All of them passed on 2026-10-13: synthetic data, ≤ 2 threads, < 15 s each.

## Process rules (all built into the scripts)

- **torch and lightgbm never share a process.**
  - torch only: `t2_mull_split.py`, `t5_vault.py run --stage fm`.
  - lightgbm only: `t3_within_build.py`, `t5_vault.py run --stage lgbm`.
  - These use `t_common.torchfree()`, a torch-free `decks.deckstrength` built from the module's own source, and
    assert at exit that torch was never loaded.
  - numpy only (torch-free shim, no lightgbm): `t0_counts.py`, `t1_rescore.py`, `hand_value_v2.py`.
- **No existing script is edited.** New files:
  - `scripts/t_common.py`, `t0_counts.py`, `t1_rescore.py`, `t2_mull_split.py`, `t3_within_build.py`,
    `hand_value_v2.py`, `t5_vault.py`.
  - Outputs go to new subfolders: `results/<SET>.PremierDraft/{t0_counts.json, t1_rescore/, t2_mull_split/,
    t3_within_build/, hand_value_v2/}`, `results/meta/`, `results/vault/`.
- **Pre-register before each step.** Add a `research_log.md` entry before running it, giving the primary key and the
  go/kill rule from meta_review_2.

```bash
cd /Users/zblanton/Documents/repos/mtg-deck-strength
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
SETS="MSH EOE SOS TDM FDN DFT LCI FIN MKM BLB DSK OTJ"     # small -> large
mkdir -p results/t_queue
while kill -0 30099 2>/dev/null; do sleep 60; done          # transformer queue must be finished (MPS + RAM)
```

## Runtime and RAM estimates

Measured references:
- `game_fm_v2` took 122-317 s per set (6 arms × 2 seeds).
- `sc_gll` took 46-196 s.
- `hand_value` took 134-1084 s at 1.2-2.9 GB.
- `rebuild_targets` took 123-1316 s.
- T0 on MSH took 1.5 s at 0.4 GB RSS. The T1 loader on MSH took 1.8 s at 0.65 GB.

| step | per set | 12 dev sets | peak RAM | process |
|---|---|---|---|---|
| T0 counts | 2-6 s | ~1 min | < 1 GB | numpy |
| T1 rescore (~25 saved scores × 3 contexts × 4 subsets, 500 boots) | 2-5 min | 30-50 min | 1-2 GB | numpy |
| T2 mull split (champion, split, − own mull, refit; 2 seeds; marginals) | 3-6 min | 40-70 min | 3-5 GB (1 M games) | torch CPU |
| T3 within-build (LightGBM 3 seeds + conditional logits, 2 controls) | 3-6 min | 40-70 min | 1.5-3 GB | lightgbm |
| T4 hand_value_v2 (7-point ridge grid, + sum-of-ratings SC-GLL) | 4-25 min | 1.5-2.5 h | 1.5-3.5 GB | numpy |
| T5 vault, per set (fm + lgbm stages) | 12-40 min | 6 sets: 2-3.5 h | 3-6 GB (MOM/WOE ≈ 1-1.25 M games) | torch, then lightgbm |

Total is about 6-9 h. If time slips, cut T4 first and T3 second (meta_review_2). **Never cut T5.**

## T0: gate counts

```bash
for s in $SETS; do nice -n 10 python scripts/t0_counts.py $s; done 2>&1 | tee results/t_queue/t0.log
python scripts/t0_counts.py meta $SETS | tee -a results/t_queue/t0.log
```

- The T3 gate: ≥ 1,000 qualifying test drafts per set, or ≥ 5,000 pooled. The run reports it at k = 1 and k = 2.
- MSH (already run): 1,799 / 715 / 208 qualifying test drafts at k = 1/2/3.
  - MSH is the smallest set, so the pooled gate at k = 1 will pass easily.
  - num_turns rises monotonically with the number of opponent colours revealed: 3.9 (none), 6.9, 9.1, 9.9, 10.5, 10.9.
    That closes meta_review_2 (d).

## T1: context matrix + blend*

Rescores every saved score under raw / dl / pre, plus blend* (α chosen on val SC-GLL-dl, two-half cross-fit).

```bash
for s in $SETS; do nice -n 10 python scripts/t1_rescore.py $s; done 2>&1 | tee results/t_queue/t1.log
python scripts/t1_rescore.py meta $SETS | tee results/t_queue/t1_meta.log       # -> results/meta/t1_rescore.json
```

How to read the result:
- The rules are printed at the end: FM over LightGBM under dl and pre; blend* − FM < +0.02; α* ≥ 0.8 count.
- T1 also picks up transformer scores (`game_transformer/scores_*.npz`). Re-run T1 at the very end to score the full
  transformer set under dl / pre. That re-run takes about 40 min and can run after T5.

## T2: own / opponent mulligan split

```bash
for s in $SETS; do nice -n 10 python scripts/t2_mull_split.py $s --seeds 2 --threads 4; done 2>&1 | tee results/t_queue/t2.log
for k in "experienced pre" "experienced dl" "match 1 pre" "experienced raw"; do
  python scripts/sc_gll.py meta --file t2_mull_split/results.json --subset "$k" $SETS; done | tee results/t_queue/t2_meta.log
```

- Optional arms: `--arms split,own,dl-own` adds `rated-dl − own mull`. `--refit-arm "- own mull"` adds the
  refitted winner.
- **Go:** `− own mull − champion` on `experienced pre` is ≥ +0.02 and decide() passes, **and** T3 does not lower b.
- **Kill:** pre Δ ≤ 0 and T3 is not higher. In that case, close mulligans for good.

## T3: within-build referee

Requires T2's `build_scores.npz`.

```bash
for s in $SETS; do nice -n 10 python scripts/t3_within_build.py $s --threads 4; done 2>&1 | tee results/t_queue/t3.log
python scripts/t3_within_build.py meta $SETS | tee results/t_queue/t3_meta.log   # -> results/meta/t3_within_build.json
```

- The primary control is the record state, plus play/draw and opponent mulligans. The match-number control is a
  sensitivity check.
- **Simulated calibration:**
  - A correctly scaled score gets b ≈ 1.2-1.25 under the state control. That is the known upward bias of a
    fixed-effects logit with a lagged-outcome regressor.
  - It gets b ≈ 0.85-1.0 under the match control.
  - So decide on **pairs**, which are in b per within-draft sd and blind to anything constant within a draft. Do not
    decide on b vs 1.
- **Veto:** an adopted arm must not lower b (one-sided p > 0.1, printed as the "leakage veto").

## T4: hand-value repair

Uses T2's `card_power.npz` for the FM slope.

```bash
for s in $SETS; do nice -n 10 python scripts/hand_value_v2.py $s; done 2>&1 | tee results/t_queue/t4.log
python scripts/hand_value_v2.py meta $SETS | tee results/t_queue/t4_meta.log     # -> results/meta/hand_value_v2.json
```

- **Upgrade the HV claim to external** if `HV train shrunk sum − GIH WR sum` on `experienced dl` is ≥ +0.02 and
  decide() passes.
- **Test HV as an FM prior next round** if the prior arm gains ≥ +0.02 against ridge (12-set RE). Otherwise close
  that direction.

## Decision checkpoint, then freeze

No vault file has been read before this point.

1. From T1/T2/T3, decide whether anything is adopted: `− own mull`, `rated-dl − own mull`, `rated-dl`, `split mull`
   or `blend*`.
2. Write the decision in `research_log.md`.
3. Freeze. This writes `results/vault/frozen_pipeline.json` (read-only) and its `.sha256`, and records the code
   hashes:

```bash
python scripts/t5_vault.py freeze --note "after T1-T4, <date>"                     # nothing adopted
# or: python scripts/t5_vault.py freeze --adopt "- own mull" --note "..."        # H5 active
```

- Every later vault run verifies the json's sha256 and the code hashes, and refuses on drift.
- To fix a bug after freezing, use `freeze --refreeze`. It only works while no vault stage has completed. The old
  file is archived, not overwritten.

## T5: vault

- ECL and TLA have PickNet pool models in `../picknet/results/`. Link them where `pick_affinity` looks (one-time
  setup, outside the hashed code):

```bash
for s in ECL TLA; do mkdir -p results/$s.PremierDraft/picknet
  ln -sfn /Users/zblanton/Documents/repos/picknet/results/$s.PremierDraft/picknet/models results/$s.PremierDraft/picknet/models; done
```

- Each `run SET` executes the fm stage (torch) and then the lgbm stage (lightgbm) as two subprocesses.
- Each stage refuses to run twice. Use `--rerun` only after a crash; it is logged in `results/vault/ledger.jsonl`.

```bash
for s in ECL TLA; do nice -n 10 python scripts/t5_vault.py run $s --threads 4; done 2>&1 | tee results/t_queue/t5_aff.log
for s in WOE MOM HOB TMT; do nice -n 10 python scripts/t5_vault.py run $s --no-aff --threads 4; done 2>&1 | tee results/t_queue/t5_noaff.log
python scripts/t5_vault.py report | tee results/t_queue/t5_report.log             # -> results/vault/report.json
```

Notes:
- **Run order:** ECL, TLA, WOE, MOM, HOB, TMT.
- **MOM and WOE** have about 1.0-1.25 M games, so the fm stage holds about 5-6 GB. Run them alone.
- **HOB, MOM, TMT and WOE** are the pre-declared no-aff fallback (no PickNet pool model). They are pooled separately
  from ECL/TLA and also in "all vault sets".
  - With k = 2, the affinity group alone is descriptive only: HKSJ uses t with 1 df.
- **Verdicts** (frozen rule):
  - CONFIRMED: vault mean > 0 and one-sided p < 0.05. For H3, the vault mean must be > 0 and the dev + vault pooled
    one-sided p < 0.05.
  - DEV-TEST ONLY: the CI includes 0 and the mean is ≤ 0.
  - NOT CONFIRMED: anything else.
  - decide() is reported alongside. There is no re-tuning on the vault.
- **Record pipeline FM input.** On vault sets the FM input to the round 10 / round 4 blend is the train-only champion.
  Its training drafts are cross-fitted on two halves using deck_synergy_v4's fold rule; val/test drafts come from the
  full-train fit. This reproduces `draft_inputs.npz`.
  - Swapping the champion v2 refit into the record blend was not frozen. It would need its own twin and in-sample
    handling, so it is a dev-set experiment first.

## After T5 (T6, no compute)

- Re-run T1 for the transformer sets (`t1_rescore.py MSH FDN BLB`).
- Write the research-log headline and `reports/` "Deck strength: state of the evidence".
