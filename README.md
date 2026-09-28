# mtg-open-lane

**Will this card wheel?** A model that predicts whether a card you pass in a Magic draft will come
back to you 8 picks later, based on the pack contents and the signals you have seen so far. It is
trained on 17Lands human draft data. See [project-plan.md](project-plan.md) for background.

Data: [17Lands public datasets](https://www.17lands.com/public_datasets). Please attribute 17Lands
and check their [usage guidelines](https://www.17lands.com/usage_guidelines).

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_dataset.py MSH      # download + labels + features (~75s)
.venv/bin/python scripts/train.py MSH              # baselines + LightGBM ablations
.venv/bin/python scripts/train_nn.py MSH 8         # WheelNet transformer (MPS/CPU)
```

Any set code with a 17Lands draft file works (`FDN`, `DFT`, `TDM`, `FIN`, `EOE`, `TLA`, `ECL`,
`SOS`, `MSH`, ...). Pass `TradDraft` as an extra arg for Bo3 pods.

## How the labels work

Packs have 14 cards and there are 8 drafters, so the pack you see at pick `p` (p = 0..5) returns at
pick `p+8`. 17Lands records the pack contents at both picks. Take the pack at `p`, subtract your
pick, and the pack at `p+8` must be a subset of what's left. This check holds on every row except
one malformed MSH draft. One example = one distinct card you passed; label = at least one copy came back.

MSH Premier: 68,609 drafts, split chronologically 70/15/15. That gives 9.1M / 1.9M / 1.9M examples
with a base wheel rate of 33.3%. Card statistics (ATA, ALSA, per-pick wheel rates) are fit on
train only.

## Models

| model | what it sees |
|---|---|
| `prior_by_pick` | pick number only |
| `alsa_ata_logreg` | card ALSA + ATA, with a separate curve per pick (**the baseline from the plan**) |
| `card_x_pick_rate` | empirical wheel rate for this card at this pick (strongest card-only lookup) |
| `lgbm_card` | card identity + static stats |
| `lgbm_card_pack` | + what else is in the pack being passed (how many better cards protect it) |
| `lgbm_card_signals` | + table signals: colours arriving late this round, earlier rounds' wheel surprises, own pool, rank |
| `lgbm_all` | card + pack + signals |
| `wheelnet` | transformer over the **raw** draft history (every pack seen + every pick), with no hand-built features |

## Results (MSH PremierDraft, held-out last 15% of drafts)

<!-- results -->
| model | logloss | Brier | AUC | ECE | logloss vs ALSA baseline | logloss P1 / P2 / P3 |
|---|---|---|---|---|---|---|
| `prior_by_pick` | 0.6067 | 0.2099 | 0.6389 | 0.0000 | -31.1% | 0.607 / 0.607 / 0.607 |
| `alsa_ata_logreg` | 0.4629 | 0.1517 | 0.8309 | 0.0073 | +0.0% | 0.453 / 0.466 / 0.470 |
| `card_x_pick_rate` | 0.4602 | 0.1511 | 0.8328 | 0.0088 | +0.6% | 0.450 / 0.463 / 0.468 |
| `lgbm_card` | 0.4588 | 0.1506 | 0.8339 | 0.0084 | +0.9% | 0.447 / 0.462 / 0.467 |
| `lgbm_card_pack` | 0.4397 | 0.1444 | 0.8502 | 0.0076 | +5.0% | 0.428 / 0.443 / 0.448 |
| `lgbm_card_signals` | 0.4496 | 0.1474 | 0.8418 | 0.0057 | +2.9% | 0.446 / 0.454 / 0.449 |
| `lgbm_all` | 0.4315 | 0.1414 | 0.8566 | 0.0052 | +6.8% | 0.428 / 0.435 / 0.432 |
| `wheelnet` | 0.4316 | 0.1414 | 0.8566 | 0.0018 | +6.8% | 0.426 / 0.435 / 0.434 |
| `ensemble` | 0.4293 | 0.1406 | 0.8585 | 0.0062 | +7.3% | 0.426 / 0.433 / 0.429 |
<!-- /results -->

### What we learned (first pass)

1. **Wheel prediction beats ALSA.** The best model cuts log loss 7.3% relative to the ALSA/ATA
   baseline, and AUC rises from 0.831 to 0.859. The per-card-by-pick lookup is only +0.6%, so
   card identity alone is nearly exhausted by ALSA.
2. **Pack context is the biggest single lever (+5.0%).** The top features are how many cards in
   the passed pack are usually taken before this one (`n_better`) and the pack's total take
   propensity. This is "the other 7 drafters have better things to take", read from what is in the pack.
3. **Signals are real and grow over the draft.** Signals alone add +2.9%, and they stack with
   pack context (+5.0% → +6.8%). The gain over the card-only lookup rises from about 3–4% per pick
   in pack 1 to about 6–7% in pack 3. That fits the idea that earlier rounds' wheels reveal the
   table's colours: `resid_prior_card` (did this colour wheel more than expected earlier?) is the
   #4 feature by gain. This runs against the "models don't read signals" result from puder, though
   predicting other drafters' picks is a different task from picking for yourself.
4. **Two independent routes land on the same number.** WheelNet learns only from the raw
   pack/pick history and ties the hand-built GBM (0.4316 vs 0.4315), with better calibration. So
   the feature set captures most of what's there. Averaging the two gives the best result, which
   means they aren't making identical errors.

### Next ideas

- Train across several sets (card-agnostic features plus per-set embeddings) and test on a held-out set.
- Out-of-fold card target encoding. Also add the per-seat structure: packs pass left in P1/P3 and
  right in P2, so P2's upstream drafters are the P1/P3 downstream ones.
- Break results down by card type: rares/bombs vs playables vs the chaff that always wheels.
  The useful product question is "will my *second-best* card come back?"
- A draft-overlay-style demo: given a live pack, show P(wheel) for each card.

## Layout

```
src/openlane/data.py      download, card metadata, dense [draft, pack, pick, card] tensors
src/openlane/features.py  wheel labels, train-only card stats, feature groups
src/openlane/nn.py        WheelNet
src/openlane/evaluate.py  logloss / Brier / AUC / ECE
scripts/                  build_dataset.py, train.py, train_nn.py
results/<SET>.<EVENT>/    results.json, predictions, models (gitignored)
```

## Caveats

- We only see one seat, and 17Lands users skew above average. The model describes what happens
  around a 17Lands user, not the general Arena population.
- `wheel_cp` and the other card stats are fit on the same train rows the GBM learns from. The
  counts are large (about 27k examples per card), so leakage is small, but out-of-fold encoding
  would be cleaner.
