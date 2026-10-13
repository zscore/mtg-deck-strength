# Team / lineup composition outcome prediction (esports + sports) as an analog to MTG deck strength

Scope: how "set of components -> win probability" is modeled in MOBAs, Pokémon, Hearthstone/other PvP games, and sports lineups; transformer/attention vs LR/FM/GBDT; handling of player skill and opponent strength; set-completion recommenders. Research cutoff: October 2026. About 20 tool calls. Several primary PDFs (Semenov et al. 2016; Hodge et al. 2017 mixed-rank; Kinkade UCSD project) could not be fully read, so their numbers come from search snippets and are flagged.

## Dota 2 / League of Legends draft win prediction: LR, FM, NN, transformers; attention gains; synergy vs individual strength

### Takeaway
Draft-only prediction in MOBAs sits close to coin-flip (roughly 52-58% accuracy for LoL/Dota on modern ranked data). Every model family lands within about 1-3 points of every other. The best published transformer (DraftRec, WWW 2022) beats LR by +2.8 pts on LoL and only +0.05 pts on Dota2, and most of that LoL gain comes from **player-history** tokens, not from better modeling of champion interactions. This is the same pattern seen with MTG: transformers tie linear models on component tokens alone.

### Cited Findings
- **DraftRec (Lee, Hwang et al., WWW 2022)**: a hierarchical model with two transformers. A *player network* encodes each player's last L matches (champion, role, 40+ performance stats such as KDA, damage, gold). A *match network* then attends over the 10 player representations together with the picked champions. Data: LoL has 279,893 matches, 156 champions, 62,466 players, average 66.38 matches of history per player. Dota2 has 50,000 matches, 111 heroes, 140,931 players, average 2.12 matches of history. — [DraftRec arXiv HTML](https://arxiv.org/html/2204.12750); code at [GitHub dojeon-ai/DraftRec](https://github.com/dojeon-ai/DraftRec)
- DraftRec match-outcome accuracy, LoL / Dota2: majority 0.5040 / 0.5180; **Logistic Regression 0.5255 / 0.5750**; MLP 0.5263 / 0.5748; HOI (factorization-style higher-order interactions) 0.5264 / 0.5716; OptMatch 0.5411 / 0.5751; NeuralAC 0.5266 / 0.5739; **DraftRec 0.5535 / 0.5755**. MAE: LR 0.4973 / 0.4819 vs DraftRec 0.4842 / 0.4782. — [DraftRec](https://arxiv.org/html/2204.12750)
  - On LoL, the models without player history (LR, NN, HOI, NeuralAC) all land at 0.526. The two models that do use player or interaction representations (OptMatch 0.541, DraftRec 0.554) are the only ones that move. On Dota2, where players have about 2 matches of history, DraftRec equals LR. — same source
  - Ablation: removing player history ("DraftRec-no-history") collapses *recommendation* HR@1 from 50.42% to 14.56% on LoL. Note this is the pick-recommendation task, not outcome accuracy. — same source
- **OptMatch (Gong et al., KDD 2020, NetEase Fuxi)**: mines a player "synergy" graph (won/lost together) and a "suppression" graph (beat each other), learns player embeddings from them, and uses an attention model over intra-team and inter-team interactions to predict match results for matchmaking. — [OptMatch KDD'20 PDF](https://linxiagong.github.io/OptMatch/KDD2020_OptMatch.pdf); [Semantic Scholar](https://www.semanticscholar.org/paper/OptMatch:-Optimized-Matchmaking-via-Modeling-the-on-Gong-Feng/aad855fe6a700f8fd531351f49bf968eacdff58b)
- **Semenov et al. 2016 (AIST, "Performance of ML Algorithms in Predicting Game Outcome from Drafts in Dota 2")**: compared Naive Bayes, LR, GBDT/XGBoost and Factorization Machines on hero picks, split by skill bracket. FM was best. A snippet reports FM AUC of about 0.706 at "normal" skill, and prediction is harder at higher skill. (UNVERIFIED: the full text was paywalled, and the snippet's list of other AUCs, "FM 0.66, XGBoost 0.65", is garbled.) — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-319-52920-2_3); [ResearchGate](https://www.researchgate.net/publication/313010005_Performance_of_Machine_Learning_Algorithms_in_Predicting_Game_Outcome_from_Drafts_in_Dota_2)
- An OpenDota hobby project: about 150k Divine-1+ Ranked All Pick matches. Always-Radiant baseline 53.7%, **LR 55.7%, tuned XGBoost 56.0% ("statistically tied")**. The author concludes that the bottleneck is features rather than model capacity, and that the draft adds about 2.6 pts over the side advantage because "skill and in-game execution are not captured." — [GitHub WinThant16/d2shenanigans](https://github.com/WinThant16/d2shenanigans)
- Older class projects report much higher draft-only accuracies: "pure LR asymptotically approaches 69.42%", and "augmented LR with hero interactions 74.1%". (UNVERIFIED / likely not comparable: these come from 2013-2015 student projects on early-era data. The primary PDF was unreachable, and these figures disagree with the ~56-58% reported on modern data by DraftRec and the OpenDota project.) — [Kinkade, UCSD CSE258 project](http://jmcauley.ucsd.edu/cse258/projects/fa15/018.pdf)
- A blog post claims that in a Dota draft model "the top 200 hero pairs contributed more to predictions than individual hero strength" (UNVERIFIED, blog, no held-out ablation shown). — [DEV Community post](https://dev.to/t9mple_0d32462141e31abf09/building-a-real-time-dota-2-draft-prediction-system-with-machine-learning-4n39)
- **Lin et al. 2024, Neural Rating Table / Neural Counter Table** (arXiv 2408.17180). NRT is a Siamese neural network: it maps a whole composition to a scalar Bradley-Terry rating, which captures nonlinear synergy. NCT adds a vector-quantized "counter category" per composition (M categories) and a residual win table, which captures intransitivity. Test accuracy on LoL (182,527 matches, 348,498 distinct compositions): BT-linear 53.6%, NRT 51.1%, NCT 51.0%, pairwise win table 50.9%. **Compositional models did not beat linear BT on LoL.** — [arXiv 2408.17180](https://arxiv.org/html/2408.17180)
- **CHAMP (Wang et al., CIKM 2026 Applied)**: a LoL matchmaking win-rate network built on cross-mode player behavior sequences (temporal, spatial and permutation encoders). It reports 67.73% offline win-rate accuracy, "outperforming attention and sequence baselines", and up to a 20.73% reduction in "5-minute kill crushing" for low tiers in an A/B test. The much higher accuracy than draft-only models comes from player features. — [arXiv 2609.04870](https://arxiv.org/abs/2609.04870)
- Hodge et al. 2017, "Win Prediction in Esports: Mixed-Rank Match Prediction in MOBA Games": models trained on mixed-rank public matches can predict professional matches "with slightly reduced accuracy". (Numbers not extracted: the PDF was unreadable.) — [arXiv 1711.06498](https://arxiv.org/abs/1711.06498)

### Inferences
- Across DraftRec, the OpenDota project and NRT/NCT, the honest ceiling for composition-only signal in MOBAs is about 2-5 pts above the base rate. LR, FM, MLP and attention-without-player-context are statistically indistinguishable. This matches the MTG finding that transformers over card tokens tie linear models.
- The main win for attention in MOBA work comes from **adding player tokens and their history**, not from better modeling of hero-hero interactions. For MTG, the analogous move is to add tokens for the drafter's history or skill proxies (17Lands user win rate, rank, number of games) alongside card tokens, rather than going deeper on card tokens alone.
- Synergy/counter terms (FM, pairwise) helped in early Dota work (Semenov). They do not clearly help in modern LoL data (NCT ≈ BT). Their value is game-dependent.

### Gaps
- No peer-reviewed 2023-2026 paper found that reports a *pure-draft* transformer beating LR/FM on a large public Dota or LoL dataset with a clean ablation. Hero2Vec-style embedding papers were not located with numbers.
- Exact Semenov per-bracket AUCs for LR, NB, XGBoost and FM could not be verified.
- No source found that quantifies the share of outcome variance due to synergy vs individual hero strength on modern data, beyond the unverified blog claim.

## Pokémon Showdown, Hearthstone, Overwatch and other PvP composition models

### Takeaway
When the components carry strong matchup structure (Pokémon types, Hearthstone archetypes, AoE2 civilizations), explicit pairwise or counter modeling adds 10-14 pts of accuracy over scalar-strength models. When it doesn't (LoL comps), it adds nothing. A team-only Pokémon model (transformer + pairwise matchup matrix, Elo equalized) reaches AUC 0.73-0.84 and beats XGBoost on most modern formats.

### Cited Findings
- **NRT vs NCT across games** (test accuracy; NRT = scalar neural BT rating, NCT = adds learned counter categories; PairWin = full pairwise table) — [arXiv 2408.17180](https://arxiv.org/html/2408.17180):
  - Hearthstone (10.15M matches, 58 archetypes): BT 81.1%, NRT 81.2%, **NCT 94.8%**, PairWin 95.4%
  - Age of Empires II (1.26M matches, 45 civs): BT 64.5%, NRT 64.5%, **NCT 75.4%**, PairWin 75.7%
  - Brawl Stars (180k matches, 94k comps): BT 53.0%, NRT 82.8%, NCT 83.4%. Here the *nonlinear composition rating* (NRT) is what gives the gain over linear BT: +29.8 pts.
  - LoL: all about 51-54%; no gain.
  - Note: these "accuracies" are for classifying the strength relation between compositions (aggregated), not per-game outcome prediction, which explains the high Hearthstone numbers.
- **FutureSightML** (open-source Pokémon team builder): species/move/item/ability embeddings, then a 3-layer transformer encoder with **cross-attention between the two teams** plus a "blade-chest" matchup-matrix decomposition (a 6×6 pairwise branch). This is ensembled with XGBoost on 642 hand-made features. Trained on about 30.5M Showdown replays. Team-only AUC on held-out data, "with equalized ratings (no Elo leakage)" and with identical teams kept on the same side of the split: Gen 9 OU neural 0.7259 vs XGB 0.6809; Gen 9 Ubers 0.8207 vs 0.7695; Champions OU 0.7679 vs 0.7107; Gen 1 OU 0.8351 vs 0.8368 (a tie). It also reports Gen 9 VGC 2026 neural AUC 0.7470. No ablation of the matchup branch. (Hobby project, not peer-reviewed.) — [GitHub HotHams/FutureSightML](https://github.com/HotHams/FutureSightML)
- **Metamon (2025)** uses transformers trained by offline RL on Showdown replays, with critic Q-values serving as in-battle win estimates. This is a battle-state model, not a team-composition model. — [arXiv 2504.04395](https://arxiv.org/html/2504.04395v1)
- VGC-Bench (UT Austin) provides a pipeline over 700k+ open-team-sheet Gen 9 VGC replays. It is a team-strategy benchmark for agents. — [VGC-Bench PDF](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/angliss2026vgc.pdf)
- Applying LSA to about 5,000 Showdown logs to predict lead pairs from team composition (a small-scale analysis). — [Journal of Geek Studies 2025](https://jgeekstudies.org/2025/07/11/predicting-competitive-pokemon-vgc-leads-using-latent-semantic-analysis-a-data-driven-approach-to-team-matchups/)

### Inferences
- Pokémon is the closest positive example for transformers on composition. The architecture that wins is **cross-attention between own and opponent sets plus an explicit pairwise matchup term**, not self-attention on one side alone. The MTG analog is per-game modeling of deck vs opponent deck (17Lands game data has opponent colors, not the full opponent deck). This is consistent with the user's FM on game outcomes being their best single model.
- MTG limited is probably closer to "LoL comps" than to "Hearthstone archetypes" in counter structure, because decks are idiosyncratic and nearly every deck is unique. That suggests limited headroom from counter modeling beyond color-pair-level matchups. This is my inference, not tested.

### Gaps
- I found no Overwatch hero-composition win-prediction papers with numbers.
- No Hearthstone per-game *decklist-level* (card-token) outcome model with LR/GBDT baselines found. NCT works at the archetype level.

## Sports lineups: RAPM, Bradley-Terry with embeddings, neural lineup models

### Takeaway
In sports the standard is ridge-regularized additive player models (RAPM), which play the same role as the user's linear card model. Embedding and neural lineup models report gains, but on small data. Transformer models (HIGFormer, soccer foundation models) mostly embed player IDs and attend across both teams.

### Cited Findings
- **RAPM (Sill, MIT Sloan 2010)**: ridge regression on a player-indicator design matrix with signed +1/-1 entries for the two lineups. It deals with collinearity from teammates who often play together, and "nearly doubles" out-of-sample accuracy over unregularized APM. — [Sill 2010 PDF](https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf); [Wikipedia: Adjusted Plus Minus](https://en.wikipedia.org/wiki/Adjusted_Plus_Minus)
- Extensions: Lineup-RAPM with informed priors ([ResearchGate L-RAPM](https://www.researchgate.net/publication/399962748_Lineup_Regularized_Adjusted_Plus-Minus_L-RAPM_Basketball_Lineup_Ratings_with_Informed_Priors)); "Dummy RAPM" for low-minute players, i.e. pooling rare entities ([arXiv 2608.19454](https://arxiv.org/pdf/2608.19454)); RAPM for soccer on possession sequences ([arXiv 2407.17832](https://arxiv.org/pdf/2407.17832)); NHL ridge APM ([arXiv 1201.0317](https://arxiv.org/pdf/1201.0317)).
- **LinNet (Pelechrinis 2017)**: network embedding of lineups (d=128) followed by Bradley-Terry logistic regression on the embedding features. NBA 2007-08 to 2011-12: **69% out-of-sample accuracy vs 56% for adjusted plus-minus**, with well-calibrated probabilities. The task is pairwise lineup-vs-lineup outcome, which is not directly comparable to per-game accuracy. — [arXiv 1707.01855](https://arxiv.org/abs/1707.01855)
- **NBA2Vec**: Word2Vec-style player embeddings learned by predicting possession outcomes from the 10 players on court. The 5 offensive and 5 defensive embeddings are averaged (a DeepSets-style sum pooling). — [arXiv 2302.13386](https://arxiv.org/abs/2302.13386)
- **Martonosi et al. 2023**: classifies "elite" NBA lineups from *order statistics* of individual player metrics (min, 40th, 60th, 80th percentile, max), using an all-or-nothing ensemble of 7 classifiers. Test precision 86.7%; 76.9% precision on the next season vs 62.1% prevalence. **The weakest player's plus-minus was the top predictor.** — [arXiv 2303.04963](https://arxiv.org/html/2303.04963v1)
- **HIGFormer (2025)**: a player-team heterogeneous interaction graph transformer for soccer. Player-ID embeddings are attended across both teams ("Match Comparison Transformer"), and win probability comes from the difference of team embeddings. — [arXiv 2507.10626](https://arxiv.org/pdf/2507.10626)
- A soccer foundation model ([arXiv 2407.14558](https://arxiv.org/pdf/2407.14558)) and "Modeling Matches as Language", a generative transformer for counterfactual player valuation ([arXiv 2603.15212](https://arxiv.org/pdf/2603.15212)), model action sequences rather than static lineups.
- Spectral analysis to identify group (synergy) contributions in NBA lineups. — [Devlin & Uminsky 2020, J. Sports Analytics](https://content.iospress.com/articles/journal-of-sports-analytics/jsa200407)

### Inferences
- Martonosi's order-statistics result (the weakest member predicts lineup quality) suggests a cheap MTG feature: quantiles of card quality within the 23 spells (for example the minimum, 10th percentile and median of card GIH WR), in addition to the mean. GBDT can use these directly. In a transformer, a pooling head that can express min-like aggregations (attention pooling or PMA from the Set Transformer) could pick them up.
- The RAPM tricks (ridge shrinkage, pooling rare players into a "dummy") map onto handling rare cards and mythics with few observations.

### Gaps
- No recent head-to-head of a transformer lineup model vs RAPM on matched NBA data with reported numbers found.
- I did not find HIGFormer's numeric results vs baselines; the PDF was not read.

## Treatment of player skill, matchmaking confounding, MMR

### Takeaway
The strongest esports results come from explicitly modeling the player, either through history sequences (DraftRec, CHAMP) or player embeddings (OptMatch). The cleanest composition-only evaluations *neutralize* rating: FutureSightML equalizes Elo and keeps identical teams on the same side of the split. Matchmaking compresses the skill gap within each game, which caps accuracy, but between-player skill still leaks into composition choice.

### Cited Findings
- DraftRec's gains over LR appear on LoL (66 matches of history per player) and vanish on Dota2 (2 matches per player). This indicates the gain comes from player context. — [DraftRec](https://arxiv.org/html/2204.12750)
- CHAMP reaches 67.73% accuracy using cross-mode player behavior sequences, vs about 52-58% for draft-only models in other papers. — [CHAMP](https://arxiv.org/abs/2609.04870); [DraftRec](https://arxiv.org/html/2204.12750)
- FutureSightML reports "team-only AUC ... with equalized ratings (no Elo leakage)". Its split keeps battles with an identical team on the same side "so rematches and repeated teams cannot inflate the score". — [FutureSightML](https://github.com/HotHams/FutureSightML)
- Semenov et al.: accuracy depends on skill bracket, and high-skill games are less predictable from the draft (snippet-level). — [Semenov 2016](https://www.researchgate.net/publication/313010005_Performance_of_Machine_Learning_Algorithms_in_Predicting_Game_Outcome_from_Drafts_in_Dota_2)
- The OpenDota project attributes the ~56% ceiling to unobserved "skill and in-game execution". — [d2shenanigans](https://github.com/WinThant16/d2shenanigans)
- Mixed-rank training transfers to pro matches with slightly lower accuracy, so composition effects are partly rank-dependent. — [Hodge et al. 2017](https://arxiv.org/abs/1711.06498)
- OptMatch learns player representations from synergy and suppression graphs and uses them together with team-up effects. This is a factorization of *players* rather than heroes. — [OptMatch](https://linxiagong.github.io/OptMatch/KDD2020_OptMatch.pdf)

### Inferences
- MTG analog to the DraftRec player network: encode the drafter's past 17Lands decks and records as a sequence and use it as a context token. The analog to the FutureSightML protocol: evaluate deck strength conditional on, or residualized against, rank and user win-rate bins, with the same user kept on one side of the split. That prevents player identity from leaking through idiosyncratic card choices.
- Since 17Lands includes player rank and opponent rank per game, the RAPM/BT formulation suggests a per-game logistic model of the form logit P(win) = f(deck) + g(player rank, play/draw) − h(opponent rank). Card effects are then estimated net of skill covariates. The user's FM on game outcomes is already close to this.

### Gaps
- No paper found that formally separates composition strength from player strength using causal or hierarchical methods (for example random effects per player plus composition effects) in esports, with numbers.
- No esports study quantifying selection bias (skilled players picking certain heroes) found.

## Recommender-style set completion: transformers learning item compatibility

### Takeaway
In fashion outfit compatibility, self-attention over an unordered item set (OutfitTransformer, WACV 2023) is the reference architecture, and it beats earlier pairwise and sequence (Bi-LSTM) models on compatibility AUC and fill-in-the-blank. These domains have rich item content features (images, text), which is where attention helped. Pure-ID settings like MOBA heroes show little gain.

### Cited Findings
- **OutfitTransformer (Sarkar et al., Amazon, WACV 2023)** treats an outfit as an unordered set and uses self-attention to learn item relations, with a learned "outfit token" for compatibility and set-to-item embeddings for complementary retrieval. The authors report it outperforms prior SOTA on compatibility prediction, fill-in-the-blank and retrieval. (Exact numbers not extracted.) — [CVF open access](https://openaccess.thecvf.com/content/WACV2023/papers/Sarkar_OutfitTransformer_Learning_Outfit_Representations_for_Fashion_Recommendation_WACV_2023_paper.pdf); [Amazon Science](https://www.amazon.science/publications/outfit-transformer-outfit-representations-for-fashion-recommendation)
- Outfit completion via conditional set transformation, formulated as set retrieval with a compatibility regularizer. — [arXiv 2311.16630 summary](https://awesomepapers.io/generative-models/papers/2311.16630)
- History-aware transformers for personalized outfit recommendation, which combine user history with the item set. This parallels DraftRec's player network. — [arXiv 2407.00289](https://arxiv.org/html/2407.00289v1)

### Inferences
- The lesson that carries across: attention helped most where items have **content features** (image/text embeddings). For MTG, card tokens could carry card-text/oracle embeddings, mana value, type and color features, plus 17Lands per-card stats, rather than bare IDs. That gives a 40-card model shared structure across rare cards, and it is where it might break the tie with linear models.
- A recipe consistent with the analogs: a Set Transformer / OutfitTransformer-style encoder (no positional encoding, a [DECK] token or PMA pooling) over count-weighted card tokens with content features, plus player and context tokens (rank, user history, play/draw), plus an optional cross-attention or FM head against the opponent's colors. Then blend with GBDT, as FutureSightML does with XGBoost.

### Gaps
- Bundle recommendation and playlist continuation (RecSys Challenge 2018) were not researched in depth within the call budget, so there are no numbers for transformer vs matrix factorization on those tasks.
- OutfitTransformer's exact AUC and FITB numbers vs baselines were not extracted.
