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
.venv/bin/python scripts/deck_multiset.py 200000   # deck transformer shared across 12 sets (~55 min)
.venv/bin/python scripts/picknet_report.py         # markdown tables for the two sections above
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

### Next ideas

- Train across several sets (card-agnostic features plus per-set embeddings) and test on a held-out set.
- Out-of-fold card target encoding. Also add the per-seat structure: packs pass left in P1/P3 and
  right in P2, so P2's upstream drafters are the P1/P3 downstream ones.
- Break results down by card type: rares/bombs vs playables vs the chaff that always wheels.
  The useful product question is "will my *second-best* card come back?"
- A draft-overlay-style demo: given a live pack, show P(wheel) for each card.

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
| top-23 pool cards, mean GP WR | 0.166 | 0.6850 |
| top-23 pool cards, mean GIH WR (paper's heuristic) | 0.213 | 0.6834 |
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
   linear deck model, and better on every set. An ablation (`scripts/lgbm_ablation.py`) shows it
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

## Pick model: predicting and simulating human picks (`notebooks/picknet.ipynb`)

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

## Deck synergy, round 2 (`notebooks/deck_synergy_v2.ipynb`)

`deck_synergy.ipynb` found that on MSH pair synergy didn't replicate, and that what synergy there was mostly
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
                          paper_baselines.py, lgbm_ablation.py, ensemble_de_lgbm.py,
                          train_picknet.py, train_picknet_multiset.py, simulate_draft.py,
                          card_text_embeddings.py, deck_synergy_v2.py, deck_multiset.py
results/<SET>.<EVENT>/    results.json, predictions, models (gitignored)
```

## Caveats

- We only see one seat, and 17Lands users skew above average. The model describes what happens
  around a 17Lands user, not the general Arena population.
- `wheel_cp` and the other card stats are fit on the same train rows the GBM learns from. The
  counts are large (about 27k examples per card), so leakage is small, but out-of-fold encoding
  would be cleaner.
