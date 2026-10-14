# mtg-deck-strength

How good is a drafted Magic deck, what is each card worth, and is card synergy real? Deck-strength models
trained on 17Lands game data, a full replication of the DraftEncoder paper (arXiv 2607.04782), and nine
rounds of card-synergy experiments.

Split out of [wheelnet](https://github.com/zscore/wheelnet) (formerly mtg-open-lane), with history. 17Lands data comes
from [mtg-data](../mtg-data); please attribute 17Lands and check their
[usage guidelines](https://www.17lands.com/usage_guidelines).

Notebooks: `decks` (deck strength, DraftEncoder, next-game odds) and `synergy` (rounds 2–9). Superseded work
is in `notebooks/archive/`, `scripts/archive/` and `results/archive/`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../mtg-data -r requirements.txt
python ../mtg-data/scripts/download.py MSH        # draft data; game data downloads on first use
python scripts/train_deckstrength.py MSH # deck strength + card values (~6 min)
python scripts/train_draftencoder.py MSH linear paper big+deck  # DraftEncoder (~1h per big run)
python scripts/deck_synergy.py MSH       # game-level card power + pair synergy (~1 min)
python scripts/improve_deck.py MSH       # deck-model improvements incl. LightGBM (~1 min/set)
python scripts/paper_baselines.py MSH    # the paper's order-dependent linear baseline
python scripts/deck_synergy_v2.py MSH    # deck features, nonland FM, draft affinity (~3-10 min/set)
python scripts/affinity_checks.py MSH     # draft affinity vs co-occurrence vs colour; skill check (~1 min)
python scripts/archive/deck_multiset.py 200000   # (archived) deck transformer shared across 12 sets (~55 min)
python scripts/deck_synergy_v3.py MSH    # retrained game-level models, affinity profile, card fit (~3 min/set)
python scripts/archive/deck_synergy_v3_blend.py MSH  # (archived) blend with the draft-level LightGBM deck model (~1 min/set)
python scripts/deck_synergy_v4.py MSH    # pair-term variants, properly trained; cross-fitted draft scores (~4 min/set)
python scripts/deck_synergy_v4_draft.py MSH  # LightGBM with deck features / affinity, boosted from the game model
python scripts/deck_synergy_v5.py MSH    # pair synergy from cards seen together in a game (~3-5 min/set)
python scripts/deck_synergy_v6.py MSH    # seen-together synergy: held-out test and unseen-card placebo (~5 min/set)
python scripts/deck_synergy_v7.py MSH    # final seen-together pair tables and per-card synergy hubs (~4 min/set)
python scripts/deck_synergy_v8.py MSH    # card-flow slope vs partner synergy; opening hand vs seen (~6-10 min/set)
python scripts/deck_synergy_v9.py        # opening-hand synergy from card features, shared across 12 sets (~15 min)
scripts/card_text_embeddings_paper.py MOM WOE ...   # (sentence-transformers env) DraftEncoder's card strings, EmbeddingGemma
python scripts/train_draftencoder_multiset.py sameset   # DraftEncoder paper replication, same set (~3h)
python scripts/train_draftencoder_multiset.py newset per_set=12000  # ... unseen set + 4-day fine-tune (~2.5h)
python scripts/de_multiset_bootstrap.py   # paired bootstrap of the sameset models
python scripts/de_multiset_report.py      # markdown tables for the replication section
```

`deck_synergy_v2.py` can score drafts with a saved PickNet. That needs `pip install -e ../picknet`.

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
| *second seed:* bag + META | 0.215 | 0.230 | 0.204 | 0.216 | | | |
| *second seed:* DraftEncoder + META | 0.213 | 0.235 | 0.200 | 0.216 | | | |

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
   and twice the paper's LR. With META the two tie. The bag led 0.216 vs 0.213 with seed 0, a gap
   the bootstrap says holds up against test noise. With a second seed it is 0.216 vs 0.216: the
   transformer's seed-to-seed spread (0.003) is as large as the gap, and the linear model's is 0.0001.
   A per-set card-id bag (0.218) is as good as anything.
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

> **Correction (round 8):** most of this effect is card draw, not synergy. "Seen" includes cards drawn by other cards'
> effects, so a draw spell that resolves both shows you more cards and wins games; the placebo can't catch that. On
> the opening hand, which is dealt before play, partner-specific synergy is −0.27 ×1000, about the size of
> deck-level pairs. The "13×" comparison below and round 7's hub tables overstate synergy.

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

**Round 7: final tables** (`scripts/deck_synergy_v7.py`; superseded by round 8, which shows most of this effect is card draw). The same model with the pair penalty grid extended down to
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

## Deck synergy, round 8: is seen-together synergy just card draw?

Rounds 5–7's seen-together pairs have two problems. First, most of their gain isn't specific to any
partner: on SOS, centring each card's pair values removed about 70% of it. Second, a card counts as
"seen" when it is drawn, so card draw counts too. A draw spell that resolves both shows you more cards
and wins games. `scripts/deck_synergy_v8.py` separates the two:
- **slope:** a per-card slope on how many other nonland cards were seen this game. This is "this card
  is better when you see more cards", with no particular partner.
- **pairs on top of the slope:** what is left is synergy with specific partners.

It also refits everything on the **opening hand** only. The hand is dealt before anything happens
(mulligans are in the context), so card draw can't drive it. Each model is scored on held-out games:
log loss ×1000, mean over 12 sets, and "significant" means the 95% interval excludes 0.

| input | model | vs | change | sets better (significant) | split-half pair r |
|---|---|---|---|---|---|
| seen this game | slope | no pairs | **−4.53** | 12/12 (12) | |
| seen this game | pairs k=4 (rounds 5–7) | no pairs | −2.44 | 12/12 (12) | 0.27–0.62 (mean 0.46) |
| seen this game | slope + pairs | slope | **−0.59** | 12/12 (10) | 0.09–0.39 (mean 0.21) |
| opening hand | slope | no pairs | −0.13 | 10/12 (2) | |
| opening hand | pairs k=4 | no pairs | −0.30 | 11/12 (8) | 0.06–0.25 (mean 0.13) |
| opening hand | slope + pairs | slope | **−0.27** | 11/12 (8) | 0.06–0.24 (mean 0.13) |

What we learned:
1. **The big seen-together effect is mostly card flow, not synergy.** A per-card slope on how many
   cards you saw beats rounds 5–7's pair model by itself (−4.5 vs −2.4). The top slopes are draw,
   selection and recursion cards; on SOS: Glimpse of Nature, Wisdom of Ages, Cauldron of Essence, Vampiric Tutor.
   Specific partners add only −0.6 on top, a quarter of what the pair model alone appeared to find.
2. **Synergy with specific partners is real but small.** On the opening hand, where card draw can't be
   the explanation, pairs still improve held-out games on 11 of 12 sets (significantly on 8). The
   gain is −0.27 ×1000, about a tenth of the seen-together number. MSH and EOE are flat. The hand
   pairs replicate only weakly across split halves (r ≈ 0.13), so individual pair values are noisy.
   The top ones are recognisable archetype glue: FDN's Ajani's Pridemate + Healer's Hawk, BLB's
   Caretaker's Talent + Carrot Cake and Heartfire Hero + Manifold Mouse, DSK's Optimistic Scavenger
   with enchantments.
3. **So for deck strength, deck-level synergy (small) is the honest answer,** as round 5 found from
   the draft side. Rounds 5–7's seen-together pair tables mix real synergy with card flow. Use them
   as "these cards like to be drawn together", not as synergy values.

Per set: `results/<SET>/deck_synergy_v8/{results.json, hand_pairs.csv, seen_pairs.csv, *_hubs.csv,
*_slopes.csv}` and `results/deck_synergy_v8_<SET>.log`. Pairs and hubs come from the slope + pairs fit.

## Deck synergy, round 9: opening-hand synergy from card features, across sets

`scripts/deck_synergy_v9.py`. Opening-hand pairs are rare in one set's games, so round 9 pools all 12 sets: stage 1
is each set's opening-hand model without pairs (a fixed offset), and stage 2 is one pair model shared by every set,
reading cards only through their features (rules-text embedding, 17Lands metadata, Scryfall facts):
Q<sub>ij</sub> = ⟨f(z<sub>i</sub>), f(z<sub>j</sub>)⟩ × a per-set scale. Test-game log loss ×1000 vs no pairs:

| pair model | mean | sets better (significant) |
|---|---|---|
| per-set free pairs (round 8) | −0.30 | 11/12 (8) |
| **shared, from card features** | −0.20 | 11/12 (6) |
| shared, set held out (day zero): DFT / BLB / LCI | +0.75 / +0.26 / −0.24 | 1/3 |

1. **Card features carry part of the synergy, not most.** The shared pairs correlate 0.1–0.45 with each set's own pairs
   and gain less than per-set pairs.
2. **Day-zero synergy fails.** On a set it never trained on, the shared model's pairs barely match that set's real
   pairs (r 0.03–0.09). It hurts DFT and BLB, and helps only LCI. (The held-out set's scale is set to the trained
   sets' mean; the first run left it at 1.0 for all 12 sets, kept in `results/deck_synergy_v9/holdout_scale1/`.)
3. **Where synergy stands.** Clean card synergy is real but small (about −0.3 ×1000 per set), comparable to deck-level
   pairs, and has to be learned within each set. Going further needs replay data for more sets (which cards were
   cast together, and when); only MSH has it here.

## Layout

```
src/decks/deckstrength.py  deck-strength data (drafts, splits, decks), metrics, calibration, pool transformer
src/decks/draftencoder.py  DraftEncoder (paper model) + deck stage, one set
src/decks/de_multiset.py   DraftEncoder pooled over sets from card text, plus linear baselines
src/decks/synergy.py       game-level card power + pair synergy (factorization machine)
scripts/                   training and synergy rounds (deck_synergy.py, deck_synergy_v2..v9.py)
results/<SET>.<EVENT>/     deckstrength/, draftencoder/, deck_synergy*/ (models are gitignored)
results/                   multi-set runs and logs (draftencoder_multiset/, deck_synergy_v9/, *.log)
```

Data loading, game/replay files, card info and encodings come from `mtgdata` (../mtg-data).

## Caveats

- We only see one seat, and 17Lands users skew above average. Results describe 17Lands users, not the general
  Arena population.
