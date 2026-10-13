# ML for CCG drafting, deckbuilding and deck evaluation (MTG limited focus): prior-work inventory

Research date: 2026-10-12. About 25 search/fetch calls. Several PDFs (Jakubik 2018, Bertram 2024 PDF) came back as binary and could not be parsed. For those, numbers come from arXiv HTML versions or abstracts. Anything marked UNVERIFIED comes from search snippets or secondary summaries, not from a primary page I read.

## Q1. Which papers and repos model MTG drafts or decks with ML?

### Takeaway
Almost all MTG work is **pick prediction** (imitating human picks, top-1 accuracy). Within a single set it plateaus at about 66-71% top-1, whether the model is an MLP, a Siamese net, a contrastive embedding model, a transformer or a fine-tuned LLM. Architecture matters little within a set. Card representation (text embeddings, meta statistics) matters a lot for unseen sets. Only one paper, DraftEncoder (Rigaux & Kashima, CoG 2026), targets deck outcome.

### Cited Findings
**Academic pick-prediction line**
- Ward, Brooks, Troha, Mills, Khakhalin, "AI solutions for drafting in MTG" (arXiv 2009.00655, 2020; IEEE CoG 2021 per later citations). Data: Draftsim.com M19 drafts, 107,949 drafts (86,359 train / 21,590 test), 265 cards. Top-1 pick accuracy: RandomBot 22.15%, RaredraftBot 30.53%, DraftsimBot (expert heuristics) 44.54%, BayesBot 43.35%, NNetBot (deep MLP) **48.67%** (p<2e-16). — [arXiv 2009.00655](https://arxiv.org/html/2009.00655v1)
- Bertram, Fürnkranz, Müller, "Predicting Human Card Selection in MTG with Contextual Preference Ranking" (arXiv 2105.11864, 2021). Siamese networks compare two candidate extensions of a pool ("twin" network vs "triplet" network that models each candidate's contribution). They report accuracy up more than 56% and pick distance down more than 83% relative to prior results. — [Semantic Scholar / arXiv 2105.11864](https://www.semanticscholar.org/paper/Predicting-Human-Card-Selection-in-Magic:-The-with-Bertram-F%C3%BCrnkranz/fc776c38b9d83d66ef87f95ce3736ca3a3896f46); follow-up comparison of contextual vs non-contextual ranking: [arXiv 2107.04438](https://arxiv.org/html/2107.04438)
- Bertram et al., "Contrastive Learning of Preferences with a Contextual InfoNCE Loss" (arXiv 2407.05898, 2024). They adapt InfoNCE to card-to-pool contrastive embedding and reach **68.80% top-1**, the best of the methods they tested. This is the main contrastive-learning precedent for MTG cards. — [arXiv 2407.05898](https://arxiv.org/html/2407.05898v1)
- Bertram, Fürnkranz, Müller, "Learning With Generalised Card Representations for MTG" (CoG 2024, best-paper nominee). Data: 14 sets (2018-2023), about 100M picks. Architecture: Siamese, with a card encoder (4 FC layers × 1024) and a deck encoder (6 conv layers). Card representations tested: hand-coded features plus sentence-transformer text embeddings, image autoencoder, 17Lands meta statistics (16 scalars), and random/one-hot. On known cards (NEO), every representation scores about 67-68%. On unseen cards: Features+Meta+Image 42.87%, Meta only 42.14%, Features only 33.57%, Image-1024 31.10%, Random 23.79%. A model trained on 75M picks from prior sets gets **55.44% on a completely unseen set**. Quote: "input representation is largely irrelevant when modelling singular, fixed sets". — [arXiv 2407.05879 HTML](https://arxiv.org/html/2407.05879v1); [abstract](https://arxiv.org/abs/2407.05879)
- Bertram, "UrzaGPT: LoRA-tuned LLMs for card selection" (arXiv 2508.08382, Aug 2025). Zero-shot GPT-4o gets 43%. LoRA-tuned Llama-3-8B gets **66.2%** after 10k steps. Untuned small LLMs were "completely ineffective". The author concedes it does not match specialist models. — [arXiv 2508.08382](https://arxiv.org/abs/2508.08382)
- Ward, "DraftFM: A Foundation Model for Day-Zero Drafting" (arXiv 2608.19568, Aug 2026). 1.64M-param discrete-choice scorer with no card-ID parameters. Each card is a 775-dim vector: 391 structured features plus a 384-dim frozen **BAAI/bge-small-en-v1.5** embedding of type line + rules text, with card names masked. A cross-attention pool summarizer (4 learned queries) feeds the scorer. Trained on 149.5M 17Lands picks from 29 sets. Zero-shot top-1 on held-out sets: BRO 50.8%, FDN 60.4%, MSH 56.7%. In-distribution: 68.6%, which the paper places within the "66–71% band" of within-set specialists. Picks by expert drafters are harder to predict. The model is overconfident by 0.03-0.08. A sealed pre-release forecast for HOB (The Hobbit) correlated Spearman 0.40-0.64 with six expert graders, versus 0.49-0.79 between pairs of humans. The paper does not address deck win rate. — [arXiv 2608.19568](https://arxiv.org/html/2608.19568v1)

**Deck-outcome work (directly on-target)**
- Rigaux & Kashima, "Predicting Drafted Deck Strength for MTG" / DraftEncoder (arXiv 2607.04782, July 2026, IEEE CoG 2026; code [akulen/MtGDraftEncoder](https://github.com/akulen/MtGDraftEncoder)). Details under Q2. — [arXiv abs](https://arxiv.org/abs/2607.04782), [HTML](https://arxiv.org/html/2607.04782)

**Community / hobbyist models**
- Czerner, "Puder" (nicze.de). A transformer with causal attention over the pick sequence plus a smaller transformer that encodes the pack. About 7M params, trained on about 2M 17Lands drafts (~100 GPU-hours). Two heads: next pick, and per-card probability of making the final deck. **71% top-1** on 1,024 test drafts, versus 23% random and about 48% for "always take the best card". 79-83% on expert drafts. Seeing pack contents (signals) added only +0.57%. The predicted deck differed from expert builds by about 3 of 24 cards. — [nicze.de/philipp/articles/puder](https://nicze.de/philipp/articles/puder/)
- Brooks, statistical-drafting (danieljbrooks, now forked as claydeck). Per-set ML pick models on 17Lands data from 20+ sets. Trains in under 20 min per set and infers in under 1 ms per pick. Adds "synergy adjustments" for color, archetype and splash. DraftFM reports these per-set MLPs at about 70% within-set agreement. — [GitHub](https://github.com/claydeck/statistical-drafting), [site](https://statisticaldrafting.com/about), [DraftFM related work](https://arxiv.org/html/2608.19568v1)
- Ryan Saxe, RyanSaxe/mtg ("State of the Art MTG Draft and DeckBuilder AI"): a transformer draft model plus a deckbuilder. The README claims the AI reached Mythic on Arena with a 66% win rate over 23 drafts (#27 Mythic). There is no published benchmark versus baselines. The older repo is RyanSaxe/MagicDraftBot. — [GitHub RyanSaxe/mtg](https://github.com/RyanSaxe/mtg), [MagicDraftBot](https://github.com/RyanSaxe/MagicDraftBot)
- MTG-card2vec (afreefaw): word2vec (gensim) over 17Lands decks, with card identity as the only input. — [GitHub](https://github.com/afreefaw/MTG-card2vec). A separate Card2Vec web app trained word2vec on about 3.6M scraped constructed decklists (Apr 2023). — [card2vec app](https://card2vec.herokuapp.com/) (deck count from search snippet, UNVERIFIED)
- MTGA_Draft_17Lands (bstaple1): an overlay tool built on 17Lands ratings. It is not a learned model. — [GitHub](https://github.com/bstaple1/MTGA_Draft_17Lands)
- mtg-bench (zachmay): eight AI agents cube-draft, build and play each other in the Forge engine, producing ratings. It is an LLM/agent benchmark, not win-rate regression. — [GitHub](https://github.com/zachmay/mtg-bench)

### Inferences
- Within a set, every architecture from MLP to transformer to LLM lands at 66-71% top-1. This mirrors the user's finding that transformers tie linear models per set. Transformers gain on unseen sets, through text embeddings, and on breadth across many sets.
- Treating "drafted-deck inclusion" as an auxiliary head (Puder) is a cheap multi-task signal for a deck-strength model.

### Gaps
- Saxe's architecture and training objective are not documented in the README I read. I could not verify a "17lands draft AI" by 17Lands staff.
- "Causal RL for Complex Card Games: An MTG Benchmark" (arXiv 2605.06066) appeared in search results, but I did not read it.

## Q2. Which works predict deck win rate (not picks), and how did transformers compare with linear/GBDT baselines?

### Takeaway
DraftEncoder is the only MTG deck-outcome paper. It reports Spearman of about 0.22 for the transformer versus about 0.11 for its linear baseline on BLB/FDN/TDM, against a simulated noise ceiling of about 0.23. Its linear baseline is far weaker than the user's 0.244 bag-of-cards model. No paper compares against GBDT. In Hearthstone, the AAIA'18 challenge was won by an ensemble of logistic regressions plus a deep net, not by a pure deep model.

### Cited Findings
- **DraftEncoder data.** 17Lands, 12 expansions, about 1.2M drafts (77k-149k per set; 276-381 cards; packs of 13-15). Target: win probability from a censored 7-win/3-loss run. Metric: Spearman ρ. — [arXiv 2607.04782 HTML](https://arxiv.org/html/2607.04782)
- **DraftEncoder architecture.** Oracle text goes through frozen EmbeddingGemma (768-d), is reduced in dimension, FiLM-conditioned on rank, then passed through 10 attention layers over the set's cards (set-contextualized card embeddings). Next come 5 bidirectional transformer blocks over pack and draft-state tokens with RoPE. The output is concatenated with a rank embedding and fed to a 3-layer MLP with sigmoid. — [arXiv 2607.04782 HTML](https://arxiv.org/html/2607.04782)
- **DraftEncoder baselines.** (a) A heuristic: the mean 17Lands GIH ("in deck") win rate of the best 23 picks. (b) Linear regression on features from the first i picks plus a rank embedding. — [arXiv 2607.04782 HTML](https://arxiv.org/html/2607.04782)
- **DraftEncoder same-set results** (read from Fig. 3, so approximate). DraftEncoder about 0.22 ρ at the final pick, DraftEncoder+Meta near the ceiling, linear about 0.11, i.e. "nearly 2×". — [arXiv 2607.04782 HTML](https://arxiv.org/html/2607.04782)
- **DraftEncoder zero-shot on FDN** (trained on the other 11 sets, Table II). DraftEncoder 0.0879±0.0131. DraftEncoder+Meta **0.1529±0.0262**. Heuristic 0.1391±0.0203 (the strongest baseline). LR+Meta 0.0898±0.0081. After fine-tuning on 4 days of FDN data: DraftEncoder 0.1641±0.0318, +Meta 0.1809±0.0381. — [arXiv 2607.04782 HTML](https://arxiv.org/html/2607.04782)
- **Hearthstone, AAIA'18 Data Mining Challenge** (KnowledgePit). Data: bot games for 400 training decks. Task: predict win rates of 200 new decks for each of 4 bots (800 predictions). Metric: RMSE. Final scores: winner hieuvq 5.573, 2nd 5.648, 3rd 5.668 (win rate in percentage points). — [KnowledgePit](https://knowledgepit.ai/predicting-winrates-of-hearthstone-decks/)
- The winning AAIA'18 solution was an ensemble of **two logistic regressions plus one deep network**, using card attributes (cost, attack, health) as extra data. — [ResearchGate abstract (via search snippet)](https://www.researchgate.net/publication/327893488_Predicting_Win-rates_of_Hearthstone_Decks_Models_and_Features_that_Won_AAIA'2018_Data_Mining_Challenge)
- Jakubik, "A Neural Network Approach to Hearthstone Win Rate Prediction" (FedCSIS 2018, AAIA'18). The win-rate network is trained by maximizing the log-likelihood of observed match results. The deck representation comes from a second network with dual objectives: reconstruct the full deck and decode individual card properties. This is an early "pretrained deck autoencoder" precedent. The page did not give an RMSE or rank. — [annals-csis](https://annals-csis.org/Volume_15/drp/365.html)
- Another AAIA'18 entry clustered decks into archetypes (DBSCAN on shared-card distance) and predicted win rate from the archetype. — [ResearchGate](https://www.researchgate.net/publication/327892902_Predicting_winrate_of_Hearthstone_decks_using_their_archetypes) (via search snippet)
- Clash Royale win-rate challenge (FedCSIS 2019 / AAIA'19), "How to Select Training Decks for Win-rate Prediction". Decks are 8 cards, with win rates estimated from more than 160M games. The search snippet says the baselines were RBF SVR scored by R². — [annals-csis Vol 18](https://annals-csis.org/Volume_18/drp/pdf/365.pdf) (details UNVERIFIED, PDF not read)
- Clash Royale "transition quality" (arXiv 2605.21868, 2026). A TQP-Transformer predicts the win-rate change after a deck switch, with MAE 0.0566 and 75.0% directional accuracy (search snippet). — [arXiv 2605.21868](https://arxiv.org/pdf/2605.21868) (UNVERIFIED, not read)

### Inferences
- DraftEncoder's "2× over linear" result does not carry over to the user's setup. Its linear baseline (about 0.11) is less than half the user's bag-of-cards LR (0.244), and its input is the draft sequence, not the registered deck. The user's tie or loss against strong linear models is consistent with the paper once baseline strength is accounted for.
- The user's 0.279 exceeds DraftEncoder's 0.23 "noise ceiling". That ceiling comes from a simulation assuming p~N(0.53, 0.05). Either the user's target or eval differs (for example, wins plus games played, or rank/skill features adding variance), or the true spread of p is wider than 0.05 SD. The ceiling depends on its assumptions and should be re-estimated on the user's own data.
- No published work benchmarks GBDT on MTG deck outcome, so the LightGBM+FM blend appears to be unpublished territory.

### Gaps
- Exact DraftEncoder per-set numbers for the same-set setting come only from a figure. I did not see a table.
- I could not get the AAIA'18 winning ensemble's ablation (LR alone vs NN alone) because the ResearchGate page returned 403.

## Q3. Hearthstone / LoR / Gwent deck win-rate prediction and deckbuilding

### Takeaway
Hearthstone has the only substantial literature: the AAIA'18 win-rate challenge, evolutionary and quality-diversity deckbuilding with learned surrogates, Q-DeckRec, and Cardsformer for language-grounded unseen cards. I found no academic ML work on Legends of Runeterra or Gwent deck win-rate prediction, only stats sites and spreadsheet matchup models.

### Cited Findings
- Q-DeckRec (Chen, Amato et al., CIG 2018, arXiv 1806.09771). Deckbuilding framed as sequential combinatorial optimization against a specific opponent. Q-learning with an MLP approximator learns a search policy, so new instances need much less compute than search baselines. — [arXiv 1806.09771](https://arxiv.org/pdf/1806.09771), [UCLA PDF](https://web.cs.ucla.edu/~yzsun/papers/2018_cig_deckrec.pdf) (no numbers extracted)
- DSA-ME, "Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding" (Zhang, Fontaine, Hoover, Nikolaidis, GECCO 2022). A deep surrogate is trained online to predict game outcomes and behavior measures from candidate decks, and it guides MAP-Elites. It beats offline-trained surrogates and linear baselines in sample efficiency. Code: icaros-usc/EvoStone2. — [arXiv 2112.03534](https://arxiv.org/abs/2112.03534)
- Evolutionary Deckbuilding in Hearthstone (García-Sánchez et al., CIG 2016). An evolutionary algorithm searches decks, and simulated AI play evaluates them. — [ResearchGate](https://www.researchgate.net/publication/304246423_Evolutionary_Deckbuilding_in_HearthStone)
- Cardsformer (Xia, Yang, Ruan, Xing, Xu, ECAI 2023). A prediction model is trained on offline trajectories to predict state transitions from card descriptions (language), and a policy model generalizes to unseen cards. It beats the Hearthstone state of the art when the deck contains untrained cards. DraftFM cites it as the language-based unseen-card precedent. — [IOS Press](https://ebooks.iospress.nl/doi/10.3233/FAIA230581)
- Drafting in CCGs via RL (Vieira et al., SBGames 2020, Legends of Code and Magic). — [SBGames PDF](https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf) (not read)
- Gwent: the community deck win-rate prediction method is the sum of matchup win rates weighted by opponent popularity (a spreadsheet, not ML). — [leriohub](https://leriohub.com/how-to-predict-deck-winrate-on-gwent-pro-ladder-interactive-sheet/)
- hearthstone-card2vec (peter1591): card embeddings for board-value estimation. — [GitHub](https://github.com/peter1591/hearthstone-card2vec)

### Inferences
- The Hearthstone evidence points to the same conclusion as the MTG work. With few decks and noisy outcomes, linear or ensemble models are competitive, and deep models help mostly through card-attribute or text grounding and through use as surrogates inside search.
- For a matchup-aware model, the Gwent/HSReplay-style decomposition (deck × opponent-archetype matchups weighted by popularity) suggests a factorized deck-vs-field term.

### Gaps
- I found no HSReplay-based academic win-rate model, no LoR or Gwent ML papers, and no transformer deck-recommendation paper for these games. Slay the Spire-style deck ML: not searched in depth, so this is not covered.

## Q4. Masked-card pretraining, contrastive learning, card-text embeddings, cross-set transfer

### Takeaway
Text embeddings of cards are standard in 2024-2026 work: sentence-transformer (Bertram 2024), bge-small (DraftFM) and EmbeddingGemma (DraftEncoder). They help almost exclusively on unseen cards and sets. Contrastive (InfoNCE) card-pool learning is the best-documented self-supervised-style objective. I found no MTG paper using BERT-style masked-card pretraining on decks. The closest relatives are word2vec card2vec and Jakubik's deck-reconstruction autoencoder.

### Cited Findings
- Representation is irrelevant within a set (about 67-68% for all variants) but decisive for unseen cards (23.8% random → 42.9% best). 17Lands meta statistics alone (42.1%) are nearly as good as features+meta+image. — [arXiv 2407.05879](https://arxiv.org/html/2407.05879v1)
- Contextual InfoNCE: 68.80% top-1, best among their methods. — [arXiv 2407.05898](https://arxiv.org/html/2407.05898v1)
- DraftFM's transfer comes from frozen text and structured features alone, with no usage statistics: 50.8-60.4% zero-shot. Bertram 2024's 55.4% needed post-release usage statistics. — [arXiv 2608.19568](https://arxiv.org/html/2608.19568v1)
- In DraftEncoder, win-rate META features are what make zero-shot work: 0.088 without them, 0.153 with them. The simple GIH heuristic (0.139) beats the meta-free transformer. — [arXiv 2607.04782](https://arxiv.org/html/2607.04782)
- In Jakubik 2018, the deck encoder's auxiliary tasks are full-deck reconstruction plus card-property decoding, a reconstruction-style pretraining precedent. — [annals-csis](https://annals-csis.org/Volume_15/drp/365.html)
- card2vec builds co-occurrence embeddings from 17Lands decks (word2vec). — [GitHub afreefaw/MTG-card2vec](https://github.com/afreefaw/MTG-card2vec)

### Inferences
- If the per-set win-rate signal is mostly additive card power plus small synergy, pretraining tasks such as masked-card prediction on decks will mostly learn co-occurrence (archetype structure). That should help the FM/synergy term or cross-set transfer more than per-set Spearman. This is a hypothesis; I found no published result.

### Gaps
- I found no published masked-card or MLM-on-decks experiment for any CCG, and no contrastive deck-vs-outcome objective.

## Q5. Metric ceilings and noise discussions for Bo1 win prediction

### Takeaway
The only quantitative ceiling is DraftEncoder's simulated Spearman of about 0.23 for 7-win/3-loss run outcomes, which assumes p~N(0.53, 0.05). Community data put the skill spread at roughly 50% to 61% game win rate across player tiers, which supports the conclusion that deck-level prediction is heavily noise-bounded.

### Cited Findings
- DraftEncoder: "label noise inherent to the sampled data limits the achievable correlation". The ceiling is about 0.23, simulated with p~N(0.53, 0.05), Bernoulli games, and the 7W/3L stopping rule. — [arXiv 2607.04782 HTML](https://arxiv.org/html/2607.04782)
- DraftFM: within-set pick agreement saturates at 66-71% across very different architectures. This is an empirical ceiling for pick imitation. Expert picks are 2.8-5.3 pp harder to predict. — [arXiv 2608.19568](https://arxiv.org/html/2608.19568v1)
- 17Lands-derived tiers: top players win about 60.7% of Bo1 games, middle about 57.1%, bottom about 50%. The overall 17Lands user base is about 55%, which is a selection effect. — [MTG Arena Zone, "Learning Draft From Top 17Lands Players' Data"](https://mtgazone.com/learning-draft-from-top-17lands-players-data/) (figures via search snippet)
- AAIA'18 Hearthstone: top-3 RMSE values were within 0.1 pp of each other (5.57-5.67), a typical plateau when labels are noisy. — [KnowledgePit](https://knowledgepit.ai/predicting-winrates-of-hearthstone-decks/)

### Inferences
- The user's 0.279 sits above the published 0.23 ceiling. The ceiling must therefore depend on assumptions (an SD of 0.05 may be too small once rank and skill vary, or the user's target or eval differs). A ceiling estimated from the user's own data, via split-half reliability or simulation from fitted p, would be more meaningful than citing 0.23.

### Gaps
- I found no 17Lands blog post that quantifies the share of deck-run variance explained by the deck versus the player versus luck. The project's own "bomb luck" analysis may already be ahead of the public literature.
