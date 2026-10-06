# mtg-open-lane

**Will this card wheel?** A model that predicts whether a card you pass in a Magic draft will come
back to you 8 picks later, based on the pack contents and the signals you have seen so far. It is
trained on 17Lands human draft data. See [project-plan.md](project-plan.md) for background.

**[WRITEUP.md](WRITEUP.md) is the one-page summary.** Notebooks: `wheel`, `drafting`, `decks`, `synergy`.
Superseded notebooks and dead-end scripts live in `notebooks/archive/` and `scripts/archive/`.

Data: [17Lands public datasets](https://www.17lands.com/public_datasets). Please attribute 17Lands
and check their [usage guidelines](https://www.17lands.com/usage_guidelines).

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_dataset.py MSH      # download + labels + features (~75s)
.venv/bin/python scripts/train.py MSH              # baselines + LightGBM ablations
.venv/bin/python scripts/train_nn.py MSH           # WheelNet transformer (MPS/CPU), 8 epochs
.venv/bin/python scripts/train_deckstrength.py MSH # deck strength + card values (~6 min)
.venv/bin/python scripts/train_draftencoder.py MSH linear paper big+deck  # DraftEncoder (~1h per big run)
.venv/bin/python scripts/deck_synergy.py MSH       # game-level card power + pair synergy (~1 min)
.venv/bin/python scripts/improve_deck.py MSH       # deck-model improvements incl. LightGBM (~1 min/set)
.venv/bin/python scripts/paper_baselines.py MSH    # the paper's order-dependent linear baseline
.venv/bin/python scripts/train_picknet.py MSH picknet 6 batch=128   # pick model (~35 min; rating/pool ~1 min)
.venv/bin/python scripts/simulate_draft.py MSH picknet 1000         # 8-bot tables vs real drafts (~4 min)
.venv/bin/python scripts/card_text_embeddings.py MSH ...  # rules-text embeddings (needs sentence-transformers)
.venv/bin/python scripts/train_picknet_multiset.py MSH 3 20000      # day-zero pick model (~45 min)
.venv/bin/python scripts/deck_synergy_v2.py MSH    # deck features, nonland FM, draft affinity (~3-10 min/set)
.venv/bin/python scripts/affinity_checks.py MSH     # draft affinity vs co-occurrence vs colour; skill check (~1 min)
.venv/bin/python scripts/archive/deck_multiset.py 200000   # (archived) deck transformer shared across 12 sets (~55 min)
.venv/bin/python scripts/deck_synergy_v3.py MSH    # retrained game-level models, affinity profile, card fit (~3 min/set)
.venv/bin/python scripts/archive/deck_synergy_v3_blend.py MSH  # (archived) blend with the draft-level LightGBM deck model (~1 min/set)
.venv/bin/python scripts/deck_synergy_v4.py MSH    # pair-term variants, properly trained; cross-fitted draft scores (~4 min/set)
.venv/bin/python scripts/deck_synergy_v4_draft.py MSH  # LightGBM with deck features / affinity, boosted from the game model
.venv/bin/python scripts/deck_synergy_v5.py MSH    # pair synergy from cards seen together in a game (~3-5 min/set)
.venv/bin/python scripts/deck_synergy_v6.py MSH    # seen-together synergy: held-out test and unseen-card placebo (~5 min/set)
.venv/bin/python scripts/deck_synergy_v7.py MSH    # final seen-together pair tables and per-card synergy hubs (~4 min/set)
.venv/bin/python scripts/picknet_report.py         # markdown tables for the two sections above
.venv/bin/python scripts/train_wheelnet_multiset.py MSH 4 20000 with_target=1  # wheel model on 8 sets (~50 min)
.venv/bin/python scripts/wheel_breakdown.py MSH     # wheel accuracy by the card's rank in the pack (~1 min)
.venv/bin/python scripts/open_lane.py MSH           # lane openness: persistence, following, payoff (~1 min)
.venv/bin/python scripts/force_vs_follow.py MSH 2000  # follow vs force in the table simulator (~10 min)
.venv/bin/python scripts/open_lane_report.py        # markdown tables for the open-lane / breakdown sections
scripts/card_text_embeddings_paper.py MOM WOE ...   # (sentence-transformers env) DraftEncoder's card strings, EmbeddingGemma
.venv/bin/python scripts/train_draftencoder_multiset.py sameset   # DraftEncoder paper replication, same set (~3h)
.venv/bin/python scripts/train_draftencoder_multiset.py newset per_set=12000  # ... unseen set + 4-day fine-tune (~2.5h)
.venv/bin/python scripts/de_multiset_bootstrap.py   # paired bootstrap of the sameset models
.venv/bin/python scripts/de_multiset_report.py      # markdown tables for the replication section
```

`scripts/download.py` fetches, converts and sanity-checks every set below (or pass set codes).
It writes `data/parquet/inventory.csv`. The feature pipeline currently assumes 14-card packs with
all 42 picks present, so it only works as-is on the "14, complete" sets.

## Data inventory (downloaded 2026-09-28)

| pack size | picks/draft | sets | wheel-eligible picks |
|---|---|---|---|
| 15 | 45 | LCI | 0–6 |
| 14 | 42 | OTJ, DSK, FDN, DFT, TDM, FIN, SOS, MSH | 0–5 |
| 14 | 41: **P1P1 missing from every draft** | TLA, TMT | 0–5 (skip P1P1) |
| 13 | 39 | MKM, BLB, EOE, ECL | 0–4 |

Premier draft totals: about 1.9M drafts and 76M pick rows, plus about 250k Traditional drafts.
Sampled wheel-subset checks: about 1M pack pairs, 3 violations, all in TradDraft.
Streaming conversion with Int8 columns keeps peak memory around 2.4 GB for each file.

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
| `wheelnet` | transformer over the **raw** draft history (every pack seen, every pick, what the other 7 took from packs that came back), plus card colour/rarity/mana value; no hand-built features |

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
| `wheelnet` | 0.4249 | 0.1390 | 0.8618 | 0.0024 | +8.2% | 0.424 / 0.427 / 0.424 |
| `ensemble` | 0.4258 | 0.1393 | 0.8612 | 0.0054 | +8.0% | 0.425 / 0.428 / 0.424 |
<!-- /results -->

### What we learned (first pass)

1. **Wheel prediction beats ALSA.** The best model cuts log loss 8.2% relative to the ALSA/ATA
   baseline, and AUC rises from 0.831 to 0.862. The per-card-by-pick lookup is only +0.6%, so
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
4. **The raw-history transformer now beats the hand-built features.** The first WheelNet tied the
   GBM (0.4316 vs 0.4315). Three changes took it to 0.4249 (val 0.4293 → 0.4226):
   - Each passed card cross-attends to the history up to its pick, instead of reading one pooled
     state vector (val 0.4293 → 0.4243, with the next change).
   - Explicit "taken by the other 7" tokens: at picks 9–14, the passed pack minus the returning pack.
   - Dropout 0.2 over 8 epochs, since it overfit after epoch 5 at 0.1 (→ 0.4235), and card
     colour/rarity/mana value added to the learned embeddings (→ 0.4227).

   More depth did not help: 6+4 layers scored 0.4228 and trained 35% slower. Averaging with
   `lgbm_all` no longer helps either (0.4258), so the GBM's features appear to be subsumed.

### One wheel model for every set (`notebooks/wheel.ipynb` §7)

The pick model's best result came from dropping per-card ids for card features shared across sets (colour,
rarity, mana value, types, Scryfall rules features, a frozen rules-text embedding). The same WheelNet with that
features-only card table (`ids=False`), trained on 20k drafts from each of the 7 other 14-card sets, MSH test:

| model | logloss | AUC | ECE | vs ALSA baseline |
|---|---|---|---|---|
| `prior_by_pick` (no card knowledge) | 0.6067 | 0.639 | 0.000 | −31.1% |
| **day zero**: 7 other sets, never saw MSH | 0.5245 | 0.776 | 0.048 | −13.3% |
| `alsa_ata_logreg` (needs the set's own data) | 0.4629 | 0.831 | 0.007 | 0 |
| per-set WheelNet (learned card ids), all inputs / best inputs (`wheelnet_time`) | 0.4249 / 0.4218 | 0.862 / 0.864 | 0.002 / 0.007 | +8.2% / +8.9% |
| **7 other sets + MSH's training drafts** (4.1M params, 4 epochs) | **0.4188** | **0.866** | **0.003** | **+9.5%** |

1. **Pretraining on other sets gives the best wheel model**, better than per-set WheelNet in every slice of the
   breakdown below (best passed card: 0.1306 vs 0.1341). It is the same result as the pick model: card features
   shared across sets beat memorising one set's cards.
2. **Day zero is far behind ALSA.** With no MSH drafts, the model knows how tables behave but not how good each new
   card is. It closes 57% of the gap from the pick prior to the baseline, but ALSA after a few days of
   data is much better, and the zero-shot model is miscalibrated (ECE 0.048). A few days of the set's own data is
   the practical starting point.

### Next ideas

- Out-of-fold card target encoding. Also add the per-seat structure: packs pass left in P1/P3 and
  right in P2, so P2's upstream drafters are the P1/P3 downstream ones.
- A draft-overlay-style demo: given a live pack, show P(wheel) for each card.

## Where wheel prediction matters: the cards you'd want back (`notebooks/wheel.ipynb` §8)

The useful product question is "will my *second-best* card come back?", not "will the chaff wheel?". Each passed
card is ranked within its pack by train-set ATA (rank 1 = the best card you passed). Log-loss gain of the per-set
WheelNet over the ALSA/ATA baseline, by rank, on the test drafts of all 14 sets with a WheelNet:

| set | ALSA → WheelNet, all passed cards | best passed card (rank 1) | rank 2 | rank 3 | ECE rank 1–3, ALSA / WheelNet |
|---|---|---|---|---|---|
| BLB | +7.4% | +11.6% | +9.2% | +8.3% | 0.017 / 0.005 |
| DFT | +8.0% | +13.7% | +10.6% | +10.0% | 0.019 / 0.005 |
| DSK | +8.4% | +14.2% | +10.5% | +9.7% | 0.018 / 0.004 |
| ECL | +8.2% | +12.8% | +9.7% | +8.9% | 0.018 / 0.003 |
| FDN | +8.9% | +16.6% | +11.4% | +10.1% | 0.021 / 0.004 |
| FIN | +8.3% | +14.4% | +10.5% | +9.6% | 0.025 / 0.007 |
| LCI | +8.8% | +14.5% | +10.8% | +9.8% | 0.029 / 0.009 |
| MKM | +6.8% | +12.8% | +9.5% | +8.2% | 0.021 / 0.004 |
| MSH | +8.2% | +14.1% | +10.7% | +9.7% | 0.024 / 0.004 |
| OTJ | +9.3% | +16.7% | +12.7% | +10.5% | 0.026 / 0.008 |
| SOS | +11.5% | +14.8% | +11.8% | +11.4% | 0.021 / 0.003 |
| TDM | +8.7% | +14.5% | +10.2% | +9.2% | 0.021 / 0.003 |
| TLA | +8.3% | +13.4% | +10.4% | +10.3% | 0.021 / 0.002 |
| TMT | +9.8% | +16.1% | +11.1% | +10.5% | 0.025 / 0.005 |

What we learned:
1. **The model earns most where it matters.** On every set the gain over ALSA is largest for the best passed card
   (+12–17%, vs +7–12% over all cards) and falls with rank. The best passed cards are the ones whose fate depends on
   the table, not the card.
2. **ALSA is overconfident about your good cards.** It is calibrated on average but 3–6× worse calibrated on the
   top three passed cards (ECE 0.017–0.029 vs WheelNet's 0.002–0.009). On MSH, when ALSA gives the 2nd-best passed
   card a 30–50% chance to wheel it comes back 15% of the time; WheelNet's 30–50% bucket comes back 32% of the
   time. ALSA is a per-card average over every pack the card appears in. Being the best card *you* pass usually means
   the pack is thin, so there is little else to absorb the other seven picks, and the card's chance of lasting is
   lower than its average.
3. **"Will it wheel?" for a top-3 card is almost always "no".** WheelNet puts ≥ 50% on under 1% of them. Its value
   is separating 1% from 25%: on MSH a third of rank-2 cards get < 2% (and wheel 1.1%), while the 9% it puts above
   20% wheel 26% of the time.

## Does the open lane pay? (`notebooks/drafting.ipynb` part 2)

The repo's namesake question, on all 15 sets. **Openness** of colour *j* in pack *k* is how many good cards of
that colour reached the drafter later than they usually last: Σ over picks 1–7 of the pack's cards' colour share ×
max(0, pick − ALSA). Picks 8+ are excluded (those packs went round the whole table). It is z-scored per (pack,
colour) within each draft week, so set-wide shifts in colour popularity cancel. The drafter's own picks never enter
it. Main colours are the top two by playability-weighted picks. Payoff is a binomial logit on the event record
(p^W (1−p)^L) with colour-pair fixed effects, rank and the 17Lands skill bucket; per sd of openness, averaged over
the two main colours.

| set | drafts | corr P1–P3 (same neighbour) | corr P1–P2 | follow slope (bronze → mythic) | payoff / sd, + skill | win rate, P1-openness quintile 1 → 5 | most-open P1 pair: in / not |
|---|---|---|---|---|---|---|---|
| BLB | 154,585 | 0.218 | 0.084 | 0.33 (0.30 → 0.36) | +0.056 ± 0.003 | 0.531 → 0.562 | 0.559 / 0.544 |
| DFT | 141,166 | 0.246 | 0.102 | 0.37 (0.32 → 0.39) | +0.060 ± 0.003 | 0.532 → 0.566 | 0.563 / 0.548 |
| DSK | 167,672 | 0.236 | 0.087 | 0.37 (0.36 → 0.39) | +0.057 ± 0.003 | 0.530 → 0.562 | 0.555 / 0.544 |
| ECL | 99,289 | 0.234 | 0.114 | 0.35 (0.32 → 0.37) | +0.065 ± 0.004 | 0.527 → 0.565 | 0.562 / 0.543 |
| EOE | 104,377 | 0.259 | 0.096 | 0.41 (0.37 → 0.44) | +0.056 ± 0.004 | 0.535 → 0.566 | 0.560 / 0.547 |
| FDN | 130,935 | 0.238 | 0.088 | 0.37 (0.32 → 0.43) | +0.063 ± 0.003 | 0.529 → 0.565 | 0.560 / 0.544 |
| FIN | 140,237 | 0.225 | 0.090 | 0.36 (0.31 → 0.40) | +0.059 ± 0.003 | 0.534 → 0.570 | 0.564 / 0.551 |
| LCI | 134,970 | 0.177 | 0.062 | 0.33 (0.28 → 0.37) | +0.052 ± 0.003 | 0.541 → 0.570 | 0.566 / 0.552 |
| MKM | 157,553 | 0.203 | 0.068 | 0.34 (0.29 → 0.38) | +0.049 ± 0.003 | 0.538 → 0.566 | 0.562 / 0.548 |
| MSH | 68,609 | 0.234 | 0.106 | 0.36 (0.33 → 0.39) | +0.057 ± 0.005 | 0.540 → 0.573 | 0.572 / 0.554 |
| OTJ | 195,312 | 0.179 | 0.056 | 0.33 (0.28 → 0.36) | +0.050 ± 0.003 | 0.527 → 0.556 | 0.551 / 0.540 |
| SOS | 107,459 | 0.234 | 0.120 | 0.37 (0.32 → 0.40) | +0.057 ± 0.004 | 0.534 → 0.569 | 0.565 / 0.547 |
| TDM | 101,323 | 0.183 | 0.065 | 0.35 (0.31 → 0.36) | +0.052 ± 0.004 | 0.533 → 0.564 | 0.559 / 0.544 |
| TLA | 102,793 | 0.229 | 0.097 | 0.37 (0.33 → 0.39) | +0.049 ± 0.004 | 0.544 → 0.569 | 0.564 / 0.555 |
| TMT | 32,151 | 0.236 | 0.100 | 0.35 (0.29 → 0.37) | +0.047 ± 0.007 | 0.557 → 0.585 | 0.579 / 0.561 |
| **mean** | | 0.222 | 0.089 | 0.36 | +0.055 | +3.2 pts | |

What we learned:
1. **Signals are real and travel through your neighbour.** Packs pass left in P1 and P3 and right in P2, so P1 and
   P3 share an upstream neighbour. A colour's P1 openness correlates 0.22 with its P3 openness and 0.09 with P2, on
   every set. P1 tells you about P3, as the folk wisdom says.
2. **Drafters follow signals, and better drafters follow more.** +1 sd of P1 openness raises the log-odds that a
   colour ends up in the deck by 0.36. The slope rises with rank (bronze 0.30 → mythic 0.39, mean over sets) and with the
   17Lands win-rate bucket (0.32 → 0.39, on every set). Still, only 17% end in the most open P1 pair (10% by chance).
3. **Being in the open lane wins games.** +1 sd of openness in your colours is worth +0.055 log-odds per game with
   skill controlled (±0.003–0.007 per set, 15/15 sets); bottom to top quintile is +3.2 points of game win rate
   (e.g. MSH 54.0% → 57.3%). Drafters who ended in the most open P1 pair win 1–2 points more.
4. **But the payoff is supply, not the read itself.** Openness in P2, which you cannot read before committing,
   pays as much as P1 openness (+0.055 vs +0.040; P3 +0.061, all three fitted together). And holding your colours'
   openness fixed, openness in the colours you *didn't* take has no effect (mean −0.002, negative on only 9/15
   sets). So more good cards in your colours win games. That is what reading signals is for, but this data
   doesn't separate the value of the decision from the luck of the seat. That needs a counterfactual, e.g. the
   PickNet table simulator with a signal-blind seat.

**Counterfactual: follow or force?** (`scripts/force_vs_follow.py`, MSH and FDN). The observational payoff can't separate
the read from the luck of the seat, so this tests it in the PickNet table simulator: 2,000 tables of real opened
packs, seats 1–7 always the human-like bot, seat 0 plays a strategy, with the same tables and random stream for every
strategy. Pools are scored as decks: the best colour pair's top 23 cards by training-game GIH WR (missing slots get
a 10th-percentile card).

| seat 0 strategy | MSH deck score | vs following | FDN deck score | vs following |
|---|---|---|---|---|
| follow: draft like a human (the bot reads signals as people do) | 57.36 | — | 55.78 | — |
| force a random pair (sample from the same model, only cards that fit) | 57.39 | +0.02 ± 0.03 | 55.78 | -0.00 ± 0.02 |
| force the best pair (MSH WU, FDN WB) | 58.67 | +1.30 | 56.26 | +0.48 |
| read P1 picks 1–5 for the most open pair, then lock it | 56.94 | -0.42 ± 0.03 | 55.52 | -0.26 ± 0.02 |

Deck score = mean GIH WR (%) of the best pair's top 23. Forcing, by pair, MSH: WU 58.7 · WG 58.3 · UG 58.1 · WB 57.8 · UB 57.6 · BG 57.3 · WR 57.0 · UR 56.9 · RG 56.6 · BR 55.8. FDN: WB 56.3 · WU 56.2 · UB 56.1 · BR 55.9 · WR 55.9 · WG 55.8 · BG 55.6 · UG 55.4 · UR 55.4 · RG 55.4.

5. **In simulation, forcing a random pair costs nothing, and forcing the strongest pair wins.** On both sets,
   committing to a random pair from P1P1 ties drafting like a human (±0.03). Which pair matters much more than
   when you commit: 2.9 points separate the best and worst pair to force on MSH, and 0.9 on FDN, a flatter
   format. Locking in whatever looks open after five picks, ignoring the cards already taken, is worse than both
   (−0.4 MSH, −0.3 FDN). Human-style drafting balances card quality, pool and
   signals, and pure signal-chasing loses that.
   Caveats: the bots under-draft WU (17% of bot decks vs 23% of real drafters), so a WU forcer faces less
   competition than at a real table; the deck score ignores curve and synergy; and every seat is an imitation of
   17Lands users, not a strong drafter.

## Deck strength (MSH PremierDraft)

How good is a drafted pool, and what is each card worth? This follows DraftEncoder
([arXiv 2607.04782](https://arxiv.org/abs/2607.04782)). The target is the event record: W wins and
L losses, with runs ending at 7 wins or 3 losses. That stopping rule depends only on results so
far, so the likelihood is just p^W (1-p)^L, and abandoned runs count too. The headline metric is
Spearman(predicted p, W/(W+L)) on the last 15% of drafts. `rank` is the only player feature used.

Card values come from three sources, all fit on training games only:
- **GP WR / GIH WR:** the 17Lands win rates, computed from game data.
- **`cast_bt`:** a Bradley-Terry fit on replay data that compares only the cards *both* players cast
  (that is the only view of the opponent's deck). P(win) = σ(Σ v_c · (user casts of c − opp casts
  of c) + play/draw, mulligans, skill, ranks), counting casts up to each player's turn t.

<!-- deckstrength -->
| model | Spearman | NLL / game |
|---|---|---|
| rank only | 0.057 | 0.6871 |
| top-23 pool cards, mean GP WR (the paper's heuristic: in-deck WR) | 0.166 | 0.6850 |
| top-23 pool cards, mean GIH WR | 0.213 | 0.6834 |
| top-23 pool cards, mean `cast_bt` (t=8, best cutoff) | 0.142 | 0.6856 |
| actual deck, mean `cast_bt` (t=30) | 0.151 | 0.6853 |
| **linear bag-of-pool** (binomial logistic regression) | **0.244** | **0.6816** |
| pool transformer (learned card embeddings) | 0.235 | 0.6820 |
| pool transformer + card meta (win rates, colours, mana value, rarity) | 0.238 | 0.6820 |
| *rank + `user_game_win_rate_bucket` (leaky, reference only)* | *0.352* | *0.6741* |
<!-- /deckstrength -->

The heuristics use a 1-d score plus a rank logistic fit on training drafts. `results.json` also
lists the "ceiling" for each model: the Spearman you would see if its own p were exactly right,
after simulating the 7-win / 3-loss runs. The best models reach about 0.25, so Spearman is mostly
capped by Bo1 noise, not by the model.

What we learned:
1. **A linear model on the final pool beats everything.** It scores 0.244 against 0.213 for the
   GIH WR heuristic. The paper's LR baseline reached about 0.11, but that was on other sets and
   included predictions made mid-draft. With about 48k training drafts, the transformer doesn't
   beat the linear model. The paper pooled 12 sets.
2. **Comparing only the cards both players cast is confounded by game state.** Those values explain
   the games well after the fact (test AUC 0.84 with all turns vs 0.62 from covariates alone),
   because winners cast more. As deck ratings, though, they are weaker than GIH WR. They rate
   bombs highly (cast while ahead, and they end games) and mark down tricks, protection and card
   draw (cast while behind), plus cheap creatures that come down every game anyway. Correlation
   with GIH WR is only 0.58. Using one set of values for both sides costs nothing: separate
   user/opponent values give AUC 0.8413 vs 0.8408.
3. **The skill bucket dominates** (0.35), but it is computed partly from these same games.
   Handling skill properly is the biggest open problem for card values.

Next: see the DraftEncoder section below for the pick-by-pick model, registered decks and synergy.
Still open: pooling sets with shared card features, and using opponent casts only to adjust for
opponent strength.

## DraftEncoder, registered decks and synergy

**DraftEncoder** (`src/openlane/draftencoder.py`) ports the paper's model
([github.com/akulen/MtGDraftEncoder](https://github.com/akulen/MtGDraftEncoder)) to PyTorch:
- **Set context:** every card in the set is one token, FiLM-conditioned on rank, then self-attention.
- **Draft layers:** at each pick, the pack's cards attend causally (RoPE) over earlier picks, and the
  draft state attends over the current pack, with the picked card as the query.
- **Output:** a prediction after every pick, with the loss weighted by (t+1)/T.

Differences from the paper:
- Cards are learned embeddings plus meta features (GP/GIH WR, colours, mana value, rarity), because
  there is no oracle text here.
- The pack tokens drop padding: every complete draft sees the same 315 cards, laid out the same way.
  This makes training 3× faster on MPS.
- An optional **deck stage** encodes the deck that was actually registered, lands included. For each
  draft that is the build with the most games; about 20% of drafts rebuild mid-run. Tokens are
  (card, copies) plus the final draft state. Training uses a fixed 12-epoch one-cycle schedule and
  keeps the best val epoch.

MSH, 10,068 test drafts that have a deck:
<!-- draftencoder -->
| model | params | Spearman | NLL / game |
|---|---|---|---|
| linear, final pool | — | 0.244 | 0.6816 |
| DraftEncoder, paper size (d=64, 3 set + 5 draft layers) | 0.50M | 0.236 | 0.6821 |
| DraftEncoder, big (d=128, 4 + 8 layers) | 2.9M | 0.237 | 0.6822 |
| **linear, registered deck** | — | **0.268** | **0.6800** |
| DraftEncoder paper size + deck stage | 0.59M | 0.258 | 0.6807 |
| DraftEncoder big + deck stage | 3.3M | 0.266 | 0.6808 |
| average of linear deck + big + deck (logit mean) | — | **0.272** | **0.6798** |
<!-- /draftencoder -->

The per-pick curve has the paper's shape. Spearman is 0.10 after P1P1, 0.18 at the end of pack 1,
0.22 at the end of pack 2 and 0.24 by pack 3.

**Registered deck vs pool, and synergy, on 12 sets** (test drafts with a deck). The last two
columns come from `scripts/deck_synergy.py`: a per-game logistic model on the registered deck plus
play/draw, mulligans, skill bucket and both ranks. `FM` adds low-rank pair terms
Σ x_i x_j ⟨v_i, v_j⟩ (a factorization machine). Its deck score is then calibrated to the draft
records like the other heuristics.

| set | GIH WR top-23 | linear pool | linear deck | game-level linear | game-level FM | FM vs linear, test-game logloss |
|---|---|---|---|---|---|---|
| MSH | 0.213 | 0.243 | 0.268 | 0.251 | 0.251 | +0.00059 |
| BLB | 0.165 | 0.199 | 0.240 | 0.229 | 0.245 | −0.00078 |
| DFT | 0.164 | 0.195 | 0.227 | 0.211 | 0.213 | −0.00002 |
| DSK | 0.205 | 0.213 | 0.245 | 0.235 | 0.248 | −0.00040 |
| FDN | 0.182 | 0.235 | 0.269 | 0.256 | 0.247 | +0.00018 |
| FIN | 0.205 | 0.222 | 0.255 | 0.239 | 0.258 | −0.00056 |
| LCI | 0.204 | 0.236 | 0.274 | 0.258 | 0.276 | −0.00069 |
| MKM | 0.186 | 0.203 | 0.231 | 0.221 | 0.234 | −0.00050 |
| OTJ | 0.164 | 0.201 | 0.228 | 0.206 | 0.221 | −0.00041 |
| EOE | 0.186 | 0.199 | 0.219 | 0.207 | 0.223 | −0.00040 |
| SOS | 0.174 | 0.201 | 0.243 | 0.234 | 0.258 | −0.00153 |
| TDM | 0.147 | 0.194 | 0.225 | 0.209 | 0.228 | +0.00003 |
| **mean** | 0.183 | 0.212 | **0.244** | 0.230 | **0.242** | FM better on 9/12 |

What we learned:
1. **Use the deck that was registered, not the pool.** It is better on all 12 sets, by +0.032
   Spearman on average. The deck captures build choices the pool can't: which colours, which
   playables, and how many lands (basics are in the deck columns).
2. **A bigger transformer doesn't help on one set.** Paper size and big tie with and without the
   deck (0.236 vs 0.237 on the pool, 0.258 vs 0.266 with the deck). Neither beats a linear model on
   the same input, and the big model starts to overfit after about 5 epochs. The paper's gains came
   from pooling sets. Our card embeddings are learned per set, so multi-set training would need
   shared card features, such as oracle-text embeddings.
3. **Transformer and linear learn nearly the same thing.** Their test logits correlate at 0.93, and
   averaging them gains only +0.004 Spearman.
4. **Synergy is real but small.** Pair terms beat card power alone on held-out games in 9 of 12
   sets, and lift draft-level Spearman from 0.230 to 0.242 on average. The strongest pairs are
   archetype glue, not combos. In MSH, the top pairs are the artifact/HYDRA shell (Hydraulic
   Helper, Machinesmith Automaton, HYDRA Assault Robot, Yellowjacket) and the Wakanda cards
   (Birnin Zana Plaza with Wakandan Royal Guard and Patriot). The most negative pairs combine
   cards from different archetypes. See `results/<SET>/deck_synergy/pair_synergy.csv`.
5. **The game-level FM nearly matches the draft-level deck model while controlling for skill**
   (0.242 vs 0.244). The draft-level model can soak up skill: cards that strong players draft look
   good. So the game-level card and pair values are the ones to trust as ratings.

## Replication on the paper's sets, and improving the deck model

**Replication.** The DraftEncoder runs use the paper size with the 12-epoch schedule. Test drafts
with a deck, Spearman:

| set | paper-style LR (order-dependent) | linear, pool (bag of cards) | DraftEncoder, picks | linear, deck | DraftEncoder + deck | **LightGBM, deck** |
|---|---|---|---|---|---|---|
| MSH | 0.188 | 0.243 | 0.236 | 0.268 | 0.258 (big: 0.266) | 0.279 |
| FDN | 0.179 | 0.235 | 0.226 | 0.269 | 0.275 | 0.283 |
| BLB | 0.151 | 0.199 | 0.203 | 0.240 | 0.252 | 0.276 |
| TDM | 0.154 | 0.194 | 0.186 | 0.225 | 0.228 | 0.250 |

- **The DraftEncoder numbers reproduce.** It reaches 0.19–0.24 on the pool, close to the paper's
  roughly 0.21 in-distribution.
- **The paper's gap over its linear baseline comes from the baseline.** Its `LinearRegression`
  (`models/baselines.py`) concatenates picks in order and fits one head per pick number, so a card's
  weight depends on the slot it was taken in. Rebuilt with card identity instead of text
  embeddings (`scripts/paper_baselines.py`), that design scores 0.04–0.06 below a bag-of-cards
  model on the same picks, on all four sets.
- **The transformer needs data.** It trails linear on MSH (69k drafts) and edges it on the larger
  FDN and BLB (130k+). With the deck stage it beats the linear deck model on FDN and BLB.

**Improving the deck model** (`scripts/improve_deck.py`, all 12 sets, test drafts with a deck):

| model | mean Spearman | mean NLL / game | better than linear deck |
|---|---|---|---|
| linear, registered deck (baseline) | 0.244 | 0.6822 | — |
| + deck structure (land count, curve, creature count, # colours, splash) | 0.259 | 0.6813 | 12/12 |
| + structure + archetype (main colours, colour pair) | 0.259 | 0.6813 | 12/12 (+0.0005 over structure) |
| + GIH WR prior (deck's mean GIH WR as a feature) | 0.244 | 0.6822 | 10/12 (≈ no change) |
| game-level FM (skill-adjusted power + synergy) | 0.243 | 0.6822 | 6/12 |
| **LightGBM on deck counts + structure + rank** | **0.271** | **0.6806** | **12/12** |
| logit average, LightGBM + linear + structure | 0.271 | 0.6806 | 12/12 |

What we learned:
1. **LightGBM on the registered deck is the best deck model:** +0.028 Spearman (+11%) over the
   linear deck model, and better on every set. An ablation (`scripts/archive/lgbm_ablation.py`) shows it
   doesn't need the structure features: deck counts + rank alone score the same. Basic lands and
   colours are visible in the deck columns, so trees find land count, colour and curve effects on
   their own. Structure + rank alone reaches only about 0.10.
2. **For linear models, deck structure is the cheap win** (+0.015, 12/12). On MSH:
   - two-colour decks beat three-colour ones by about 0.23 logit, and each splash costs about 0.07;
   - 16 lands beats 17 by about 0.065 logit;
   - more creatures helps.

   These are observational: skilled players may favour lean builds.
3. **Ensembling the transformer barely helps.** Its logits correlate 0.86–0.90 with LightGBM's.
   A three-way average gains about +0.004 on MSH and FDN and loses on BLB and TDM, so LightGBM
   alone is the practical choice.
4. **Things that did not help:** a GIH WR prior; archetype identity; blending in the game-level FM;
   and a non-leaky skill proxy from the picks (how often the drafter takes the crowd's ATA
   favourite). That proxy predicts wins on its own (rank-only 0.057 → 0.111) but adds nothing once
   the deck is known (0.2796 → 0.2798).

Engineering notes:
- torch and LightGBM each bundle their own `libomp`. In one process on macOS, LightGBM loaded
  after torch segfaults, and using both deadlocks. `improve_deck.py` imports LightGBM first and runs
  the torch FM in a spawned child process.
- LCI's game file writes card counts as `0.0`, so `games.py` now parses them as floats and casts
  to Int8.

## Full DraftEncoder replication: card text, pooled sets, unseen set

The sections above port DraftEncoder with learned per-set card embeddings. This one replicates the
paper's actual setup: cards are read only through their text, one model is trained on several sets,
and it is tested on a set it never saw.

**Setup**
- **Data:** the paper's 12 sets (MOM and WOE added). Card counts match the paper's Table I exactly.
- **Card text:** each card becomes the paper's tagged string (`[NAME] … [MC] … [EFFECTS] …`). That
  string is embedded with EmbeddingGemma-300M using the task prompt from the paper's code
  (`scripts/card_text_embeddings_paper.py`).
- **META:** opening-hand, drawn, in-deck, GIH and deck-not-seen win rates, plus play rate. Our game
  files have no sideboard or tutored columns.
- **Model:** d=64, 10 set layers and 5 draft layers, dropout 0.3, batch 256, AdamW.
- **Windows:** taken from the paper's code (`akulen/MtGDraftEncoder`, `data.py`): 14-day blocks
  counted back from each set's last draft, finished runs only (7 wins or 3 losses).
- Code: `src/openlane/de_multiset.py` and `scripts/train_draftencoder_multiset.py`.

**Baselines**
- The paper's LR: one head per pick number, with a separate weight for every pick slot.
- **bag:** the same text features summed over the picks so far, so order doesn't matter.
- **bag_id:** one weight per card per set. It is a same-set reference only, since it can't transfer.

Linear learning rates were chosen on validation loss.

**Where we differ from the paper**
- **META without the leak:** our META comes from each set's training-window games only. The paper's
  code computes it from the whole game file, test weeks included.
- **Training length:** DraftEncoder trains for 12 epochs at lr 3e-4 (one-cycle), not 100 epochs at
  1e-4. At 1e-4, 12 epochs is clearly too short.
- **Seeds:** one seed, not three. The paired bootstrap below covers test-set noise, not seed-to-seed
  variance.
- **Cross-set sample:** 12k training drafts per source set, about 132k in total (the paper used about
  400k).

**Same set** (one model on BLB, FDN and TDM; train days 42–28 before each set's end, val 28–14, test
the last 14; 119k training drafts, as in the paper). Spearman at the last pick, plus the mean over the
three sets at the end of each pack:

| model | BLB | FDN | TDM | **mean** | P1P1 | end P1 | end P2 |
|---|---|---|---|---|---|---|---|
| HEURISTIC (top-23 mean in-deck WR) | 0.109 | 0.099 | 0.123 | 0.111 | 0.066 | 0.098 | 0.110 |
| LR (the paper's, order-dependent) | 0.098 | 0.115 | 0.097 | 0.103 | 0.048 | 0.074 | 0.084 |
| LR + META | 0.137 | 0.145 | 0.128 | 0.136 | 0.064 | 0.104 | 0.121 |
| **bag** of text features (order-free linear) | 0.200 | 0.225 | 0.196 | **0.207** | 0.078 | 0.135 | 0.170 |
| **bag + META** | 0.215 | 0.230 | 0.204 | **0.216** | 0.079 | 0.149 | 0.186 |
| bag of card ids (per set) | 0.216 | 0.232 | 0.205 | 0.218 | 0.081 | 0.146 | 0.186 |
| DraftEncoder (text) | 0.188 | 0.215 | 0.177 | 0.193 | 0.082 | 0.140 | 0.171 |
| DraftEncoder + META | 0.211 | 0.233 | 0.194 | 0.213 | 0.086 | 0.151 | 0.188 |

Paired bootstrap over test drafts (`scripts/de_multiset_bootstrap.py`, 1000 resamples; differences in
mean Spearman):

| comparison | difference | 95% CI |
|---|---|---|
| DraftEncoder − bag | −0.014 | [−0.018, −0.010] |
| DraftEncoder + META − bag + META | −0.004 | [−0.007, −0.001] |
| bag − the paper's LR | +0.104 | [+0.097, +0.113] |

**Unseen set.** The model is trained on days 28–14 before the end of 11 sets and validated on their
last 14 days. It is then tested on FDN's last 14 days and fine-tuned on FDN's first 4 days (3 to train,
1 to validate, as the paper's code splits it). "4 days only" is the same model trained on those FDN
days alone. The DraftEncoder rows and the "12k" rows use 12k training drafts per source set. The other
linear rows use 20k. The paper's numbers are from its Table II.

| model | no fine-tuning | + FDN first 4 days | FDN 4 days only | paper: none / 4 days |
|---|---|---|---|---|
| HEURISTIC (top-23 mean in-deck WR) | 0.117 | — | — | 0.139 / — |
| HEURISTIC, META incl. test weeks (paper's code) | 0.147 | — | — | |
| LR (the paper's) | 0.049 | 0.109 | 0.093 | 0.069 / 0.069 |
| LR + META | 0.118 | 0.141 | 0.111 | 0.089 / 0.090 |
| bag, 12k per set | 0.094 | 0.223 | 0.221 | — |
| bag + META, 12k per set | 0.175 | 0.224 | 0.223 | — |
| bag + META, 12k per set, META incl. test weeks | 0.190 | 0.225 | 0.224 | — |
| bag + META, 20k per set | 0.184 | 0.219 | 0.218 | — |
| DraftEncoder (text) | 0.075 | 0.213 | 0.204 | 0.088 / 0.164 |
| **DraftEncoder + META** | **0.187** | **0.228** | 0.209 | 0.153 / 0.181 |

A single set's 22k test drafts give each Spearman a standard error of about 0.007. Differences
between two models on the same drafts are less noisy (about ±0.003–0.004 in the same-set
bootstrap), so gaps under about 0.01 here are not reliable.

What we learned:
1. **We reproduce the paper's numbers.** DraftEncoder reaches 0.19 from text and 0.21 with META,
   against the paper's roughly 0.20–0.21. The paper's LR reaches 0.10, against its roughly 0.10–0.11.
2. **The "2× over the linear baseline" comes from the baseline.** The same text features, summed
   instead of laid out by pick slot, score 0.207 at the last pick. That is above DraftEncoder (0.193)
   and twice the paper's LR. With META the two tie: 0.216 vs 0.213, with the bag ahead by a small
   margin that does hold up in the bootstrap. A per-set card-id bag (0.218) is as good as anything.
   Text features lose almost nothing against card identity within a set.
3. **The transformer's only edge is early in the draft.** It is slightly ahead at P1P1 and at the
   end of pack 1 (0.151 vs 0.149 with META), then falls behind by the end. Whatever it reads from
   packs and signals is worth less than a clean sum over the final pool.
4. **On an unseen set, DraftEncoder + META is the one clear transformer win.** It scores 0.187 with no
   target-set data, against 0.175 for the linear bag + META on the same 12k-per-set sample. From text
   alone, transfer is weak for every model: 0.075–0.094, as in the paper (0.088). Rules text by
   itself doesn't tell a model which cards win.
5. **Four days of the new set beat everything else.** Once FDN's first 3 days are available, every
   model lands at 0.21–0.23. A linear bag trained on those days *alone* (0.221–0.223) is as good as
   any model pretrained on 11 sets and fine-tuned. Pretraining helps the transformer (+0.009 to +0.019
   over its own scratch run) but not the linear model. Our fine-tuned numbers sit well above the
   paper's (0.164–0.181). We gave fine-tuning 8 epochs at lr 3e-4; the paper's code uses a fixed
   schedule.
6. **The paper's META leaks, and the leak is worth about +0.015–0.03.** Computing META from the
   whole game file, test weeks included as the paper's code does, lifts the heuristic from 0.117 to
   0.147 (the paper reports 0.139). It lifts bag + META's zero-shot score from 0.175 to 0.190. Our
   DraftEncoder + META beats the paper's (0.187 vs 0.153) *without* the leak, probably because of the
   training schedule and our extra META columns (play rate, deck-not-seen WR).
7. **Practical takeaway, consistent with the deck-model sections above:** for deck strength, learn
   per-card values linearly from whatever target-set data exists. Text and META help only before
   that data arrives. The transformer adds at most about 0.01 (zero-shot with META) and otherwise
   ties or loses.

The wheel model predicts what the *other* seven drafters take. PickNet predicts the drafter's own pick: a
softmax over exactly the cards in the pack (a conditional logit, as in DraftFM), conditioned on everything the
drafter has seen. Same chronological split as everything else; MSH test = 10,292 drafts, 432k picks.

| model | params | NLL / pick | top-1 | top-1, non-forced picks |
|---|---|---|---|---|
| `rating`: one number per card (a pick-order bot) | 340 | 1.324 | 0.497 | 0.459 |
| `pool`: rating + rank-32 bilinear candidate × pool term | 26k | 0.954 | 0.644 | 0.617 |
| `picknet_nohist`: candidates attend to each other; state = pool bag | 0.57M | 0.855 | 0.673 | 0.647 |
| **`picknet`**: + causal transformer over the whole draft history | 1.3M | **0.826** | **0.682** | **0.658** |
| `picknet_big`: d=192, 4 history + 3 pack layers | 3.9M | 0.815 | 0.686 | 0.662 |
| `multiset_zeroshot`: features only (no card ids), trained on 7 other sets, never saw MSH | 1.4M | 1.411 | 0.508 | 0.470 |
| **`multiset_plus_target`**: the same features-only model, 7 other sets + MSH's training drafts | 1.4M | **0.810** | **0.686** | **0.661** |
| *DraftFM, MSH zero-shot (paper; 29 training sets, 149M picks)* | 1.6M | 1.135 | 0.567 | 0.534 |
| *statistical-drafting / puder (other sets, in-set)* | | | ~0.70–0.71 | |

PickNet's history token at pick t is the bag of the pack shown, the card taken at t−1, the cards the other
seven took from a pack that came back, the position, and the drafter's rank / skill / date. Causal
self-attention means state t sees exactly what the drafter knew. Candidates (card embedding + copies + state)
alternate self-attention within the pack with cross-attention to the history, then a logit per card.

What we learned:
1. **Context is worth 18 points of agreement over a pick order** (0.497 → 0.682). Most of it is the pool
   (+15: stay in your colours), then **the pack** (+3: which cards compete with each other), then **the history**
   (+1.0, −0.029 NLL; most in pack 3, 0.663 → 0.676). So signals do matter for predicting human picks, unlike
   puder's +0.57% from pack contents. The gain over the rating bot is biggest in the middle of each pack.
2. **Strong drafters are more predictable.** Agreement rises from 0.656 (Bronze) to 0.696 (Mythic), and from
   0.66 to 0.70 across the 17Lands win-rate buckets. The rating bot barely moves (0.475 → 0.504): strong
   players agree with each other about *context*, not about raw card quality.
3. **Rating vs pool on 14 sets.** The bilinear pool model adds 15–22 points over a pick order on every set,
   including the 13- and 15-card-pack sets and TLA (P1P1 missing). See `results/<SET>/picknet/`.
4. **Day zero is much harder.** Without card ids, trained on 7 other sets (133k drafts), PickNet reaches 0.508 on
   MSH against 0.686 on held-out drafts of its own training sets. The loss is mostly at P1P1-style picks, where
   only card quality matters: card strength from rules text and features alone is the hard part. DraftFM gets
   0.567 with 4× the sets and ~25× the picks. **But other sets are a strong pretraining
   signal:** the same features-only model trained on the 7 sets *plus* MSH's own training drafts is the best pick
   model here (NLL 0.810, top-1 0.686), better than PickNet with learned per-card ids (0.826) and matching the
   3× bigger id model (0.815), in 3 epochs. Card features shared across sets beat memorising cards one set at a time.

**Simulating whole tables** (`scripts/simulate_draft.py`). Eight copies of a model draft against each other from
real opened packs (8 random test drafters' P1/P2/P3 packs, with their context), passing left-right-left, each
bot *sampling* its pick from the softmax. 1,000 tables, scored with the same statistics as the real test drafts:

| bots | ATA r | ATA MAE (picks) | per-card wheel r | wheel MAE | pack-3 picks in top-2 colours | colour-pair TV distance | P1P1 r |
|---|---|---|---|---|---|---|---|
| `rating` | 0.989 | 0.44 | 0.984 | 0.029 | 0.559 | 0.150 | 0.875 |
| `pool` | 0.987 | 0.47 | 0.988 | 0.021 | 0.773 | 0.122 | 0.954 |
| `picknet` | 0.979 | 0.53 | 0.954 | 0.050 | **0.802** | **0.066** | **0.973** |
| real drafters | | | | | 0.815 | | |

The rating bots match per-card ATA and wheel rates best: those are marginal statistics, and a per-card rating is
fitted to exactly the marginal choices. But they don't draft like people: only 56% of their pack-3 picks are in
their two main colours, against 81.5% for humans. PickNet bots commit like humans do (≈0.80, tracking the real
curve through each pack) and pick P1P1 like humans (r = 0.97), at the cost of slightly worse per-card ATA:
small errors compound when every seat is a model reacting to other models. Every kind of bot under-drafts the format's
most popular pair: WU is 23% of real MSH drafters but 17% of PickNet bots and 11% of rating bots. Humans
converge on the best archetype harder than a model that imitates one pick at a time. Sampling more sharply
(temperature 0.7) doesn't fix it (WU 19%) and over-commits (0.828), so temperature 1 is the calibrated setting.

## Deck synergy, round 2 (`notebooks/synergy.ipynb`)

`notebooks/archive/deck_synergy.ipynb` found that on MSH pair synergy didn't replicate, and that what synergy there was mostly
learned the mana base. `scripts/deck_synergy_v2.py` runs its suggested next steps, plus two new ideas, on all 12
sets with game data. Game-level models with the full context (play/draw, mulligans, both ranks, skill,
experience, record before the game); test-game log loss vs the linear card-power model, ×1000, paired:

| model | mean vs linear (×1000) | sets better | draft-level Spearman (mean) |
|---|---|---|---|
| linear (card power) | 0 | — | 0.230 |
| + deck features (Scryfall: land count, curve shape, colours/splash, castability) | −0.70 | 12/12 | 0.249 |
| FM, all cards + deck features | −0.87 | 12/12 | 0.252 |
| FM, **nonland cards only** + deck features | −1.02 | 11/12 | 0.256 |
| linear + deck features + **draft affinity (one column)** | −1.10 | 11/12 | **0.262** |
| FM pairs as a learned map of the pick model's card vectors + deck features | −1.06 | 12/12 | 0.260 |
| **FM nonland + deck features + draft affinity** | **−1.20** | 11/12 | 0.260 |
| shared cross-set deck transformer on top of linear + deck features | −0.09 more | 9/12 | |

What we learned:
1. **Synergy is real once the mana base is taken out.** With deck features in the model and pair terms only
   between nonland cards, the pair values now replicate across independent halves of the drafts on 11 of 12 sets
   (r = 0.10–0.31, against 0.83–0.92 for card power), and none of MSH's (r = −0.01). MSH is the smallest set
   (377k games) and the outlier on every synergy measure.
2. **Human drafting is a free synergy prior.** The pick model's affinity S<sub>ij</sub> (how much card j in
   your pool raises the odds you take card i) is learned from picks alone, never from a game result. Added as a
   single column (the deck's mean pairwise affinity), it beats the whole nonland FM (−1.10 vs −1.02), and the FM's
   learned pairs agree with it (r = 0.09–0.22 on the 11 sets where synergy replicates). Together they stack a little (−1.20). It is
   not just co-occurrence or colour (`scripts/affinity_checks.py`): plain deck co-occurrence (PMI) as the same
   single column gains less (−0.26 vs −0.40 on top of deck features), colour overlap almost nothing (−0.04), and
   fitted together, pick affinity takes all the weight.
3. **Deck features: castability and colours.** Across sets the largest effects are coloured-source shortfall
   (−0.065 log-odds per sd), splash (−0.04), 3+ main colours (−0.035), 16 lands beats 17 (+0.035) and a higher
   average mana value among nonland cards (+0.04, with card power held fixed).
4. **At the draft level** these game-level, skill-adjusted ratings close most of the gap to the draft-trained
   LightGBM deck model (0.262 vs 0.271 mean Spearman over the same 12 sets, from 0.230 for linear). The draft-level
   gain from affinity is larger than its game-level gain on MSH, so part of it may be that coherent builds mark
   stronger drafters: with the skill bucket in the draft-level calibration, affinity's gain shrinks from +0.012 to
   +0.005 Spearman. The game-level comparison is the skill-adjusted one.
5. **A deck transformer pooled over 12 sets learns little that per-set card power + deck features miss.** It
   reads cards only through their features (rules-text embedding, Scryfall features), trained on 2.4M games to
   predict what stage 1 misses: −0.09 ×1000, better on 9/12 sets, an order of magnitude smaller than the deck
   features or affinity. Without the deck features in stage 1 (`shared_nodf`), the same network gains −0.64
   ×1000 over plain linear on 12/12 sets: from card features alone it rediscovers about 90% of what the
   hand-built deck features (−0.70) capture. What remains is set-specific, which is where affinity and the
   nonland FM come in.

## Deck synergy, round 3 (`notebooks/synergy.ipynb` part 2 covers rounds 3–7)

`scripts/deck_synergy_v3.py` tries to get more out of the pick model's affinity than one mean-affinity column. For
each nonland card in a deck, its *fit* is its mean affinity to the deck's other nonland cards. Same 12 sets, games,
context and paired test metric as round 2. Test-game log loss ×1000, mean over sets:

| model | vs linear | vs FM + aff (sets better, significant) | draft Spearman |
|---|---|---|---|
| linear (card power) | 0 | +1.40 | 0.233 |
| + deck features | −0.74 | +0.66 | 0.252 |
| + deck features + affinity column | −1.22 | +0.18 | 0.265 |
| FM nonland + deck features | −1.24 | +0.16 | 0.263 |
| **FM nonland + deck features + affinity column** | **−1.40** | 0 | **0.268** |
| linear + deck features + **affinity profile** (worst card's fit, sd of fit, off-plan cards) | −1.22 | +0.17 | 0.265 |
| linear + deck features + aff + **card fit** (per-card synergy sensitivity) | −1.40 | −0.01 (5/12, 2) | 0.266 |
| FM nonland + deck features + aff + card fit | −1.46 | −0.07 (10/12, 4) | 0.268 |
| FM nonland + deck features + aff, continued training (control for card fit) | −1.43 | −0.03 (8/12, 5) | 0.267 |

Draft level, blended in logit space with the draft-trained LightGBM deck model (`scripts/archive/deck_synergy_v3_blend.py`,
weight chosen on validation drafts; it lands at 0.35–0.55):

| model | mean Spearman | mean NLL / game | better than LightGBM |
|---|---|---|---|
| LightGBM, deck counts + structure + rank | 0.271 | 0.68057 | — |
| FM nonland + deck features + aff (game level, skill-adjusted) | 0.268 | 0.68079 | 2/12 |
| **LightGBM + FM nonland + deck features + aff** | **0.278** | **0.68016** | **12/12** |
| LightGBM + linear + deck features + aff | 0.277 | 0.68023 | 12/12 |

What we learned:
1. **Round 2's models were under-trained.** They stopped at the first validation plateau at learning rate 3e-3. A
   warm-started continuation at a lower rate was the first thing to "help" here, and a control showed the gain was
   all extra training. `synergy._train` now takes `lr_drops` (on a plateau, return to the best state and divide
   the rate by 3). With two drops the nonland FM + affinity goes from −1.20 to −1.40 against linear, which is
   retrained the same way. The ranking of the round 2 models doesn't change. `lr_drops` defaults to 0, so earlier
   scripts still reproduce their numbers.
2. **The mean is all of the affinity signal.** The worst card's fit, the spread of fit and the count of off-plan cards
   add nothing once the mean is in (−1.22 either way). Fitted together, the mean takes the weight: +0.13 log-odds
   per sd, positive on 12/12. The others average +0.01 to +0.03.
3. **Per-card synergy sensitivity is real but tiny.** Card fit gives each card its own weight on how well it fits the
   rest of the deck. Against the continued-training control it gains −0.04 ×1000, and it is worth more only on MKM
   (−0.21) and OTJ (−0.08). The cards it picks out make sense. In MKM the most fit-sensitive cards are gold
   build-arounds (Trostani, Three Whispers; Izoni, Center of the Web; Doppelgang). The least are cheap
   investigate/disguise cards that are fine anywhere (`results/<SET>/deck_synergy_v3/card_fit.csv`). It isn't worth
   the extra parameters as a model.
4. **The skill-adjusted game model and LightGBM are complementary at the draft level.** Neither beats the other on its
   own (0.268 vs 0.271), but their logit average is the best deck model so far: 0.278 mean Spearman, better than
   LightGBM on all 12 sets. LightGBM finds build effects from draft records; the game model contributes card values
   and synergy freed of the skill confound. Plain linear + deck features + affinity blends almost as well (0.277),
   so the FM pairs add little here.
5. **Dropped:** per-card play/draw offsets, a multitask model with "on the play" as the task. They overfit, losing
   +0.3 to +0.6 ×1000 against the FM on SOS and BLB.

## Deck synergy, round 4

Round 2 compared its pair-term variants on under-trained models. `scripts/deck_synergy_v4.py` reruns them with
learning-rate drops. Every model has the deck features and the affinity column. Test-game log loss ×1000 is paired
against `linear + deck features + aff`, averaged over the 12 sets:

| pair terms | vs linear + deck features + aff | sets better (significant) | draft Spearman |
|---|---|---|---|
| none | 0 | — | 0.265 |
| **FM rank 4, nonland cards** | **−0.18** | **12/12 (9)** | 0.268 |
| FM rank 8, nonland | −0.18 | 12/12 (8) | 0.267 |
| FM rank 16, nonland | −0.17 | 11/12 (7) | 0.267 |
| FM rank 4, all cards | −0.14 | 10/12 (7) | 0.267 |
| learned map of the pick model's card vectors (k=8) | −0.17 | 11/12 (5) | 0.268 |
| pick-model map + free rank-4 nonland pairs | −0.16 | 10/12 (5) | 0.269 |

At the draft level, `scripts/deck_synergy_v4_draft.py` scores the game model *cross-fitted*: each training draft is
scored by a model fit on the other half of the training drafts. That lets LightGBM start from the game model's
logit (init_score) without learning to trust an in-sample score. Mean Spearman on the test drafts with a deck:

| model | mean Spearman | mean NLL / game | better than round 3's blend |
|---|---|---|---|
| LightGBM, deck counts + structure + rank | 0.271 | 0.68057 | 0/12 |
| LightGBM + deck features + affinity column as inputs | 0.275 | 0.68031 | 4/12 |
| LightGBM + game FM, logit average (round 3) | 0.278 | 0.68022 | — |
| LightGBM + deck features + aff, averaged with the game FM | **0.279** | **0.68015** | 7/12 |
| LightGBM boosted from the game FM's logit | 0.278 | 0.68017 | 7/12 |

What we learned:
1. **Pair terms are saturated.** Properly trained, every pair-term variant beats the affinity column alone by about the
   same amount: −0.14 to −0.18 ×1000, and the nonland FM is significant on 9 of 12 sets. Higher rank doesn't help,
   lands in the pairs slightly hurt, and building pairs from the pick model's card vectors ties free pairs. The
   synergy this data can detect is low-rank and small: about a seventh of what the deck features and the affinity
   column give (−1.22 against linear).
2. **LightGBM can use the deck features and affinity** (0.271 → 0.275, 11/12). Trees on raw counts can't compute
   castability or pairwise fit for themselves.
3. **Boosting from the game model is no better than averaging with it** (0.278 vs 0.278). The best draft-level model
   is LightGBM with deck features + affinity, averaged with the game FM: 0.279. That is 0.001 above round 3's blend,
   which is within noise.
4. **Where this leaves the deck model.** With registered decks, records and the pick model's affinity as inputs, the
   game-level and draft-level models have converged. Better pair terms, ranks, profiles and stacking each move test
   Spearman by 0.001 or less. Further gains need new information: what the opponent cast (replay data), which cards
   actually got drawn, or pooling synergy across sets through card features.

## Deck synergy, round 5: synergy from cards seen together

A deck-level FM only knows that two cards shared a deck, and in most games at most one of them shows up.
`scripts/deck_synergy_v5.py` fits the pair terms on the nonland cards each game actually *saw* (opening hand or
drawn): a "seen FM" with deck power, seen-card power, game-length buckets, and the number of lands and of spells seen
(so the low-rank pair term can't just count spells drawn, i.e. flood / screw). It uses in-game information, so it
only *estimates* pair values Q. Those become one deck-level column (the deck's mean pairwise Q), like the affinity
column, and the deck-only models are scored as before.

| | mean over sets |
|---|---|
| split-half pair r, **seen FM** (nonland pairs in ≥ 2,000 training decks; the 10 sets where it fits pairs) | **0.66** (0.44–0.76) |
| split-half pair r, deck FM, same pairs | 0.13 |
| seen pairs vs deck pairs, r | 0.24 |
| deck level: linear + deck features + aff **+ seen Q**, vs without (×1000) | −0.03 (10/12 better, 2 significant) |
| deck level: FM nonland + deck features + aff + seen Q, vs without | +0.01 |

On OTJ and EOE validation picked the strong pair penalty and the seen FM learned almost no pairs.

What we learned:
1. **In-game synergy is measurable and very repeatable.** Pair values from cards seen together replicate at r ≈ 0.66
   between independent halves of the drafts, against 0.13 for the deck FM. The top pairs are payoff × enabler hubs:
   SOS's Potioner's Trove with instants and sorceries (Divergent Equation, Wisdom of Ages, Zealous Lorecaster), BLB's
   Heaped Harvest with Food makers (Cache Grab, Wick's Patrol), and MKM's Topiary Panther with green commons. They are
   mostly *not* what the pick model's affinity finds (r ≈ 0.07): drafters group cards by archetype, while these pairs
   are about what happens in the same game. See `results/<SET>/deck_synergy_v5/seen_pairs.csv`.
2. **It barely moves deck-level prediction.** The seen-Q column has a positive weight on all 12 sets, but gains only
   −0.03 ×1000, and nothing once the nonland FM is in. A pair only matters in the ~10% of games where both cards
   show up, and its deck-level average is already mostly captured by the FM and the affinity column. Good synergy
   estimates don't make better deck predictions here.
3. **So the deck model is limited by Bo1 noise, not by its synergy estimates.** For card evaluation (which pairs
   are real), the seen FM is the one to trust. For deck strength, round 4's models stand.

## Deck synergy, round 6: is seen-together synergy real?

Round 5's seen-together pairs replicated well but were never scored on held-out games, and repeatability alone could
come from deck composition. `scripts/deck_synergy_v6.py` adds both checks. The inputs are as in round 5, with a finer
pair penalty grid; round 5's grid had collapsed to no pairs on OTJ and EOE. Each pair model is scored on test games
against the same model without pairs (log loss ×1000, mean over 12 sets):

| pair terms between | vs no pairs | sets better (significant) | split-half pair r |
|---|---|---|---|
| **nonland cards both seen this game, rank 4** | **−2.42** | **12/12 (12)** | 0.27–0.61 (mean 0.47) |
| the same, rank 8 | −2.37 | 12/12 (12) | |
| *placebo:* nonland cards in the deck but both *unseen* this game, rank 4 | −0.08 | 9/12 (6) | −0.02–0.22 (mean 0.06) |

What we learned:
1. **In-game synergy is real.** Pairs of cards seen together improve held-out prediction on every set, by −2.4 ×1000 on
   average (−0.7 on MSH, up to −4.0 on DFT). That is 13× the deck-level FM's pair gain (−0.18) and more than the deck
   features and affinity give together. The same pair terms on cards that sat unseen in the library gain 30× less
   and don't replicate. So the effect comes from cards meeting on the battlefield, not from what decks they're in.
2. **Rank 4 is enough** (rank 8: −2.37). Validation picked the lowest pair penalty on 11 of 12 sets, so a weaker
   penalty may gain a little more. The split-half r is lower than round 5's 0.66 because the pairs are less shrunk.
3. **Why this doesn't help deck strength (round 5):** a deck only gets a pair's effect in the games where both cards
   show up. Averaged over the deck, that is already mostly in the deck-level FM and affinity. The seen-together pair
   values are the synergy numbers to use for rating card pairs; see round 7 for the final tables.

**Round 7: final tables** (`scripts/deck_synergy_v7.py`). The same model with the pair penalty grid extended down to
3e-5. The held-out gain is unchanged (−2.43 vs −2.42 ×1000, 12/12 significant; placebo −0.05). The weakest penalty
is now chosen on only 3 sets, so the grid brackets the optimum. From these fits:
- `results/<SET>/deck_synergy_v7/seen_pairs.csv`: every nonland pair in ≥ 2,000 training decks, with Q (extra
  log-odds when both are seen) and the pick-model affinity.
- `results/<SET>/deck_synergy_v7/card_hubs.csv`: each card's own seen value and its *synergy dependence*, the
  co-occurrence-weighted mean Q over its partners, with its best and worst partner.

Caveat on synergy dependence: a card's own value and its average pair effect trade off in this model (r = −0.55 to
−0.69 across cards), so weak cards look synergy-dependent partly by construction. `notebooks/synergy.ipynb` part 2
ranks hubs by the *spread* of their pair values across partners instead, which is much less tied to own value; it
picks out mostly the same cards. The cards with the highest synergy dependence are payoffs that are weak on their own. They are BLB's Talents (Caretaker's
Talent with Carrot Cake, Q = +0.36; Scavenger's Talent with Heaped Harvest), MKM's Chalk Outline with Insidious
Roots, DSK's Emerge from the Cocoon with Vile Mutilator and Shroudstomper, and SOS's Wisdom of Ages with Zealous
Lorecaster. Cards that are good anywhere (SOS's Colorstorm Stallion, Practiced Offense) sit near zero.

## Layout

```
src/openlane/data.py      download, card metadata, dense [draft, pack, pick, card] tensors
src/openlane/features.py  wheel labels, train-only card stats, feature groups
src/openlane/nn.py        WheelNet
src/openlane/games.py     17Lands game + replay data, cards cast per game
src/openlane/deckstrength.py  deck-strength data, card values, metrics, pool transformer
src/openlane/draftencoder.py  DraftEncoder (paper model) + deck stage
src/openlane/synergy.py   game-level card power + pair synergy (factorization machine)
src/openlane/cardinfo.py  Scryfall card facts (pips, mana produced, keywords, rules roles), deck features
src/openlane/picknet.py   pick models: rating, pool-bilinear, PickNet
src/openlane/evaluate.py  logloss / Brier / AUC / ECE
scripts/                  build_dataset.py, train.py, train_nn.py, train_deckstrength.py,
                          train_draftencoder.py, deck_synergy.py, improve_deck.py,
                          paper_baselines.py,
                          train_picknet.py, train_picknet_multiset.py, simulate_draft.py,
                          card_text_embeddings.py, deck_synergy_v2.py, deck_synergy_v3.py,
                          deck_synergy_v4.py, deck_synergy_v4_draft.py,
                          deck_synergy_v5.py, deck_synergy_v6.py, deck_synergy_v7.py,
                          train_wheelnet_multiset.py, wheel_breakdown.py, open_lane.py, force_vs_follow.py
results/<SET>.<EVENT>/    results.json, predictions, models (gitignored)
scripts/archive/, notebooks/archive/, results/archive/   dead ends and superseded work (see WRITEUP.md)
```

## Caveats

- We only see one seat, and 17Lands users skew above average. The model describes what happens
  around a 17Lands user, not the general Arena population.
- `wheel_cp` and the other card stats are fit on the same train rows the GBM learns from. The
  counts are large (about 27k examples per card), so leakage is small, but out-of-fold encoding
  would be cleaner.
