# Transformers should join the blend, not replace it

The literature says that on this problem a transformer will not beat the LightGBM + game-level FM blend by being a better architecture. Its realistic job is to be a decorrelated ensemble member, or a way to bring in information the trees do not see. Every comparable domain finds the same thing: MTG pick models, MOBA draft prediction, CTR models on sparse multi-hot data, and tabular benchmarks. Once the input is a bag of components and the label is noisy, model families converge to within noise of each other. The gains that do appear come from ensembling, the correct likelihood, and side information (player context, card content features). Deeper attention over the same tokens does not produce them. Your own results already show this. Per-set transformers score 0.235-0.266 against 0.244 for the linear deck model and 0.271-0.283 for LightGBM. Their logits correlate 0.86-0.93 with LightGBM, and ensembling them adds about 0.004 or less. The best-supported next steps are cheap. First, a seed or BatchEnsemble-style deck transformer, refit on train+val with the same recency weighting that just gave +0.0052 on MSH, stacked out-of-fold into the blend. Second, inference-only features from PickNet, chiefly per-draft pick-surprise as a non-leaky skill/style proxy, fed into LightGBM. Third, a game-level transformer fine-tuned from PickNet with a teacher-distillation auxiliary loss. Expect gains of **+0.001 to +0.005 Spearman** at best. Those are only distinguishable from noise with paired bootstrap over test drafts, multiple seeds, and pooling across all 12 sets.

## Every comparable domain finds architecture-independent plateaus

The closest prior work is DraftEncoder (Rigaux & Kashima, CoG 2026), the only paper that predicts MTG deck outcome rather than picks. It reports a transformer reaching about **0.22-0.23 Spearman versus about 0.11 for linear regression**, against a simulated noise ceiling of about 0.23 ([arXiv 2607.04782](https://arxiv.org/html/2607.04782)). That "2× over linear" result does not transfer to this project. The paper's linear baseline is less than half of your bag-of-cards model (0.244), and its input is the draft sequence, not the registered deck. Once you correct for baseline strength, your tie with linear models is consistent with the paper. The paper also has no GBDT baseline, so the LightGBM + FM blend sits in unpublished territory. Your 0.279-0.293 exceeds the paper's 0.23 ceiling, which shows that ceiling depends on its assumptions (p ~ N(0.53, 0.05), Bernoulli games, a 7-win/3-loss stop). Rank and deck variation probably make the true spread of p wider. A ceiling simulated from your own fitted p would be more informative than the published number.

Pick prediction shows the same plateau. Within a set, models from MLPs to Siamese nets to transformers to LoRA-tuned Llama-3-8B land at **66-71% top-1** ([DraftFM, arXiv 2608.19568](https://arxiv.org/html/2608.19568v1); [UrzaGPT, arXiv 2508.08382](https://arxiv.org/abs/2508.08382); [Puder](https://nicze.de/philipp/articles/puder/)). Bertram et al. state that "input representation is largely irrelevant when modelling singular, fixed sets": every card representation scores about 67-68% on known cards, and representations only matter for unseen cards (23.8% random vs 42.9% best) ([arXiv 2407.05879](https://arxiv.org/html/2407.05879v1)). MOBA draft prediction matches. In DraftRec (WWW 2022), logistic regression, MLP and factorization-style interaction models all score about **0.526 accuracy on LoL**. Only models with player-history context move, and DraftRec's transformer gains nothing on Dota2 (0.5755 vs LR 0.5750), where players average about 2 matches of history ([arXiv 2204.12750](https://arxiv.org/html/2204.12750)). Neural composition ratings fail to beat linear Bradley-Terry on LoL (51.1% vs 53.6%) ([arXiv 2408.17180](https://arxiv.org/html/2408.17180)). An OpenDota study found LR and tuned XGBoost "statistically tied" at 55.7% vs 56.0% ([d2shenanigans](https://github.com/WinThant16/d2shenanigans)). In CTR prediction, the field built specifically for sparse multi-hot inputs and noisy binary outcomes, a uniform re-benchmark of 24 models found DNN, DeepFM, xDeepFM, AutoInt+ and DCN within **3e-4 AUC** of each other on Criteo, and many published gains did not reproduce ([FuxiCTR, arXiv 2009.05794](https://arxiv.org/pdf/2009.05794v1)).

The theory explains why. A 40-card deck over a vocabulary of about 300 cards is fully captured by its count vector, so "linear on counts" is already a Deep Sets model with a one-hot encoder and a linear readout ([Zaheer et al., arXiv 1703.06114](https://arxiv.org/abs/1703.06114)). What a Set Transformer adds over that is shared pairwise and higher-order interactions through embeddings ([Lee et al., arXiv 1810.00825](https://arxiv.org/abs/1810.00825)). That is the inductive bias the factorization machine already supplies with <v_i, v_j>. Your transformer is structurally AutoInt on card tokens ([arXiv 1810.11921](https://arxiv.org/abs/1810.11921)). The CTR evidence predicts it ties the FM, and it does. Your synergy measurements (pair terms worth about 0.2-0.3 thousandths of game log-loss) say the interaction signal is real but tiny relative to binomial noise. Attention cannot extract signal the data does not contain.

Two domains show transformers clearly winning on compositions, and both have features this project lacks. Pokémon team models reach AUC 0.73-0.82 and beat XGBoost on most formats. They use cross-attention between the two teams plus an explicit pairwise matchup branch, and they have strong type-matchup structure ([FutureSightML](https://github.com/HotHams/FutureSightML)). Hearthstone archetypes gain 14 points from learned counter categories ([arXiv 2408.17180](https://arxiv.org/html/2408.17180)). MTG limited decks are nearly all unique, and 17Lands only exposes opponent colors, so the project probably resembles LoL more than Hearthstone. That is an inference, not a tested result.

## Ensembling and the right likelihood are the documented levers

Tabular benchmarks have reached a consensus that applies directly. On TabArena (51 datasets, 500-250k rows), CatBoost ranks first under standard tuning. **With post-hoc ensembling the top three become TabM, LightGBM and RealMLP**, and cross-family ensembles beat every single model ([arXiv 2506.16791](https://arxiv.org/html/2506.16791)). TabM gets its strength from averaging many weak, overfit submodels trained with shared weights. It beats an ensemble of independent MLPs as well as FT-Transformer and SAINT ([arXiv 2410.24210](https://arxiv.org/pdf/2410.24210)). Attention-based tabular transformers are no longer the best neural option. Earlier, XGBoost beat new deep tabular models on datasets those models had not chosen, but **XGBoost plus the deep models together** beat everything ([Shwartz-Ziv & Armon, arXiv 2106.03253](https://arxiv.org/pdf/2106.03253)). Grinsztajn et al. attribute the trees' edge to NNs' bias toward smooth functions and their sensitivity to uninformative features ([arXiv 2207.08815](https://arxiv.org/abs/2207.08815)). Both apply here, since most card columns carry almost no signal. The Jane Street Kaggle winner, working on low-SNR financial targets, blended XGBoost with a supervised-autoencoder MLP that used a Gaussian-noise input layer and several related targets ([writeup](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-01-30-Kaggle-exp-s2/)).

The second lever is the likelihood. Training per game on Bernoulli outcomes is unaffected by the 7-win/3-loss stopping rule, because the likelihood of a game sequence factorizes. It also lets play/draw, opponent rank and mulligans absorb noise that an aggregate win rate cannot. Your game-level FM, the best single component, is already built this way. Any transformer you try should be too. For noisy labels, self-distillation theory shows soft teacher targets act as extra regularization ([Mobahi et al., arXiv 2002.05715](https://arxiv.org/pdf/2002.05715)). In the high-noise regime, the optimal weight on the teacher can exceed the weight on the raw label ([Das & Sanghavi, ICML 2023](https://proceedings.mlr.press/v202/das23d/das23d.pdf)). A student trained only on the teacher is capped near the teacher's quality, so distillation earns its place only as an auxiliary loss that lowers variance while the student's own inductive bias adds something.

Pretraining helps most when labels are scarce and the pretraining objective is aware of the target. Rubachev et al. find that "using the object target labels during the pretraining stage is beneficial", and that properly done pretraining often lets tabular DL beat GBDTs ([arXiv 2207.03208](https://arxiv.org/abs/2207.03208)). Contrastive pretraining lets deep rankers beat GBDTs under label scarcity, and the advantage shrinks as labels become abundant ([arXiv 2308.00177](https://arxiv.org/pdf/2308.00177)). With 77k-149k drafts per set, you are closer to the abundant end per set, though each label carries only 5-9 games. Generic masked-item pretraining has a poor replication record. BERT4Rec's numbers reproduce only with up to 30× longer training, and part of its edge comes from the loss rather than the masking ([Petrov & Macdonald, arXiv 2207.07483](https://arxiv.org/abs/2207.07483)). I found no published masked-card or deck-completion pretraining result for any card game.

## PickNet helps most as a source of features

PickNet all_d256 is a strong, free asset. It is 8.4M parameters trained on 17 sets and reaches 70.9% top-1, which is in the 66-71% within-set band that published specialists reach ([DraftFM](https://arxiv.org/html/2608.19568v1)). It can also embed cards from unseen sets through its feature-built CardTable. The literature still argues for modest expectations on per-set Spearman. Pick-trained representations encode *population* card valuations and co-occurrence. A deck-strength model needs *deviations* from consensus, and LightGBM with one-hot counts already learns per-card value from 100k+ decks per set. Bertram's result that representation does not matter within a fixed set ([arXiv 2407.05879](https://arxiv.org/html/2407.05879v1)) predicts that PickNet card vectors, used as inputs, will mostly reproduce what the count columns already give LightGBM. You have already used the per-set pool model as the affinity column. Its share of the 0.271 → 0.275 step is a fair preview of what the card-valuation side of PickNet is worth.

PickNet helps in three places. **The first is cold start.** DraftEncoder shows text-only card features transfer badly for win rate (0.088 zero-shot), while META stats lift that to 0.153 and 4 days of fine-tuning gets to 0.181 ([arXiv 2607.04782](https://arxiv.org/html/2607.04782)). Your own multi-set run found the same thing. PickNet vectors are a better day-0 card prior than raw text, but only for the first days of a format. **The second is rare cards.** Initializing FM or transformer card embeddings from PickNet works like RAPM's informed priors for low-minute players ([Sill 2010](https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf); [L-RAPM](https://www.researchgate.net/publication/399962748_Lineup_Regularized_Adjusted_Plus-Minus_L-RAPM_Basketball_Lineup_Ratings_with_Informed_Priors)). It shrinks mythics with few observations toward sensible values, which matters most for small sets. **The third is drafter signal, the most interesting and least proven.** DraftRec's entire gain over LR comes from player context ([arXiv 2204.12750](https://arxiv.org/html/2204.12750)), and CHAMP reaches 67.7% versus about 52-58% for draft-only models by using player behavior sequences ([arXiv 2609.04870](https://arxiv.org/abs/2609.04870)). DraftFM finds high-win-rate players' picks are 2.8-5.3 points *harder* to predict ([arXiv 2608.19568](https://arxiv.org/html/2608.19568v1)), so PickNet's per-pick surprise carries information about who is drafting. That signal comes from the draft itself, not the leaky user win-rate bucket. Whether it predicts wins beyond rank is an open question, and it is a cheap one to answer. Weak drafters also deviate from consensus, so the useful signal may be non-monotonic. Leave it to LightGBM to sort out.

## Transformer options for this project

The ranking is by expected Spearman gain per hour of effort, given that realistic gains are a few thousandths. Every option must be compared with the current champion under the same protocol: refit on train+val (85%) with recency weights (tau=2), multiple seeds, and blend weights chosen out-of-fold.

| Rank | Option | Expected gain on top of the 0.293 MSH blend | Effort on 10-core/16GB/MPS | Risk |
|---|---|---|---|---|
| 1 | PickNet-derived features into LightGBM | +0.000 to +0.004 | Low: inference only, under 1 h/set | Low |
| 2 | Seed/BatchEnsemble deck transformer, refit + OOF stacking | +0.001 to +0.004 | Medium: about 4-8 h/set | Low |
| 3 | PickNet-initialized game-level transformer with distillation auxiliary | +0.001 to +0.005 | Medium-high: 2-3 days of code, about 2 h/set | Medium |
| 4 | Residual transformer on the LightGBM + FM OOF logit | +0.000 to +0.003 | Low-medium | Medium |
| 5 | Opponent-aware game transformer (cross-attention to opponent colors + FM head) | +0.000 to +0.003 | Medium | Medium |
| 6 | Tabular foundation model (TabICLv2 / TabPFN-3) on compressed features | +0.000 to +0.003 | Medium; hardware-bound | High |
| 7 | Cross-set trunk with per-set adaptors | Around 0 at day 14+; positive only for days 0-3 | High | Medium |
| 8 | Masked-card or deck-completion pretraining, or a deeper per-set Set Transformer | Around 0 | High | High |

The gain ranges are my judgment, scaled from your observed ensemble increments (≤0.004) and the refit gain (+0.0052 ± 0.0016). They are not literature numbers.

### 1. PickNet features into the existing LightGBM

**What it is.** Run all_d256 in inference over each draft. Feed LightGBM a few summaries: mean and minimum per-pick log-probability of the human's actual picks (pick surprise), surprise split by pack and by pick number, the history encoder's final pool-state vector reduced to 8-16 PCA components, and deck-level mean/quantile statistics of PickNet card vectors projected onto a few principal directions. Martonosi et al. found that the *weakest* lineup member's metric was the top predictor of NBA lineup quality ([arXiv 2303.04963](https://arxiv.org/html/2303.04963v1)), which argues for including minimum and quantile pooling, not only means.

**Why it might beat the current model.** Pick surprise is the only available drafter-skill/style signal that is not the leaky win-rate bucket. Player context is the one thing that consistently moves composition models in MOBAs ([DraftRec](https://arxiv.org/html/2204.12750)). LightGBM handles a non-monotonic relationship natively.

**Evidence.** Expert picks are harder to predict ([DraftFM](https://arxiv.org/html/2608.19568v1)), and player-history gains appear in DraftRec and CHAMP ([CHAMP](https://arxiv.org/abs/2609.04870)). No source tests pick surprise as a win predictor. That part is new.

**Cost.** PickNet inference over about 100k drafts × 42 picks takes minutes on MPS. LightGBM refits are the existing pipeline. Half a day of code.

**Risk.** Low. Watch for a subtle leak: if PickNet's training data overlaps the deck model's test drafts, the surprise feature becomes memorization. Use a PickNet checkpoint trained only on drafts before the deck model's test window, or verify the chronological split matches.

### 2. Seed or BatchEnsemble deck transformer, refit and stacked

**What it is.** Take the existing DraftEncoder + deck-stage model and train 5-8 seeds, or one BatchEnsemble model with k=8-16 shared-weight heads as in TabM. Add SWA over the last epochs. Refit on train+val with recency weights and the val-chosen epoch count scaled by data size. Combine with LightGBM and the FM using stacking weights fit on out-of-fold predictions.

**Why it might beat the current model.** The existing transformer already wins on FDN/BLB over the linear model, and its 0.07-0.14 of logit decorrelation from LightGBM is the raw material blends exploit. The earlier ensemble test probably handicapped it: a single seed, no refit, and no recency weighting. The refit gain shows recent data matters, and that must apply to the neural member too.

**Evidence.** In TabArena, ensembled neural models reach the top three ([arXiv 2506.16791](https://arxiv.org/html/2506.16791)). TabM's collective of weak members beats independent ensembles ([arXiv 2410.24210](https://arxiv.org/pdf/2410.24210)). GBDT + NN blends beat either alone ([arXiv 2106.03253](https://arxiv.org/pdf/2106.03253)).

**Cost.** At about 1 h per set per DraftEncoder run, 5 seeds is about 5 h per set. That is overnight for MSH, or about 2.5 days for all 12 sets. BatchEnsemble costs roughly 1.3-2× one run.

**Risk.** Low, with a low ceiling. If the refit transformer adds under 0.002 on MSH, stop, because more seeds will not fix it.

### 3. PickNet-initialized game-level transformer with a distillation auxiliary loss

**What it is.** Initialize the card table and history encoder from all_d256. Replace the pick head with a deck head over the registered 23 spells plus lands, using count-weighted tokens and PMA pooling. Train on **per-game Bernoulli outcomes** with play/draw, opponent rank and mulligans as logit offsets, plus an auxiliary MSE toward OOF LightGBM + FM predictions (α around 0.3-0.5 on the teacher). Optionally add a Puder-style auxiliary head that predicts which pool cards made the deck ([Puder](https://nicze.de/philipp/articles/puder/)). Use a low learning rate on pretrained layers.

**Why it might beat the current model.** It is the only option that combines all three levers the literature supports: pretraining under label scarcity, the correct game-level likelihood, and a denoised teacher target. Unlike your earlier per-set transformers, it does not have to learn card identities from 5-9 games per deck.

**Evidence.** Target-aware pretraining helps tabular DL ([arXiv 2207.03208](https://arxiv.org/abs/2207.03208)). Pretraining wins under label scarcity ([arXiv 2308.00177](https://arxiv.org/pdf/2308.00177)). Soft targets help under high noise ([Das & Sanghavi](https://proceedings.mlr.press/v202/das23d/das23d.pdf)). DraftEncoder's FiLM-on-rank conditioning is a template for skill covariates ([arXiv 2607.04782](https://arxiv.org/html/2607.04782)). No source shows pick-pretraining improves win-rate regression.

**Honest view of PickNet here.** Expect faster convergence and stability across seeds, with possible gains on small sets. Do not expect a step change: within a set, representation barely matters ([arXiv 2407.05879](https://arxiv.org/html/2407.05879v1)). The ablation that decides it is PickNet-init versus random-init with the same head and losses.

**Cost.** 8.4M parameters fine-tunes comfortably in 16 GB on MPS. Budget 1-3 h per set and 2-3 days to write the deck head and game-level data loader.

**Risk.** Medium. The game-level loader and the teacher's OOF folds must be leak-free. The student may simply converge to the teacher.

### 4. Residual transformer on the blend's OOF logit

**What it is.** Feed the LightGBM + FM OOF logit as a fixed offset and train a small set transformer, or the option 3 model, to predict only the residual. Use strong weight decay and early stopping.

**Why it might beat the current model.** Attention capacity goes only to what the trees miss: context-dependent interactions such as curve × archetype or splash structure.

**Evidence.** Hybrid GBDT/NN designs such as DeepGBM exist mainly for deployability ([DeepGBM](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf)), and there is no strong evidence that the student beats the teacher. Your cross-set residual run moved game log-loss by only -0.09 ×1000.

**Cost.** Low once option 3's loader exists. Risk: medium. The prior is that residuals are mostly noise.

### 5. Opponent-aware game transformer

**What it is.** A game-level model whose deck tokens cross-attend to opponent-color tokens, plus an FM-style deck × opponent-color pairwise head, in the style of FutureSightML's cross-attention + matchup-matrix design ([FutureSightML](https://github.com/HotHams/FutureSightML)).

**Why it might beat the current model.** It captures matchup structure, such as aggro versus slow colors, that a deck-only score averages away.

**Evidence.** Matchup modeling wins big in games with strong counter structure and adds nothing on LoL ([arXiv 2408.17180](https://arxiv.org/html/2408.17180)). MTG limited probably sits nearer LoL.

**Cost.** Medium. **Risk.** Medium-high. Deck-level Spearman averages over opponents, so even real matchup effects mostly cancel in the metric.

### 6. Tabular foundation model as an ensemble member

**What it is.** TabICLv2 (open weights) or TabPFN-3 run in-context on compressed features: about 50-100 columns such as color pair, curve, removal count, FM score, LightGBM OOF prediction, and top SVD components of the count matrix.

**Why it might beat the current model.** It is a third model family for the blend. TFMs dominate TabArena at small scale.

**Evidence.** TabICLv2 claims to beat tuned RealTabPFN-2.5 and handle 600k rows, but it was pretrained on ≤100 features, and its timings are from an H100 ([arXiv 2602.11139](https://arxiv.org/html/2602.11139v1)). TabPFN-3's "beats 8-hour-tuned GBDTs up to 1M rows and 200 features" is an unreplicated vendor claim ([arXiv 2605.13986](https://arxiv.org/abs/2605.13986)). Nothing shows how synthetic-prior TFMs behave on a target with achievable R² of about 0.05.

**Cost.** In-context inference over 100k training rows will be slow or memory-bound on 16 GB MPS. Plan to subsample the context to 20-50k recent drafts or use CPU offload.

**Risk.** High: the low SNR mismatches the prior, it is hardware-limited, and TabPFN license terms need checking.

### 7. Cross-set trunk with per-set adaptors

**What it is.** One trunk over PickNet card vectors trained on all 12 sets, with a lightweight per-set adaptor (FiLM/LoRA or a UniSRec-style MoE adaptor) fine-tuned on each set ([UniSRec, arXiv 2206.05941](https://arxiv.org/abs/2206.05941)).

**Why it might beat the current model.** Only in the first days of a format, before a per-set linear model has data. DraftEncoder's 4-day fine-tune closes most of the gap to its META model ([arXiv 2607.04782](https://arxiv.org/html/2607.04782)).

**Cost.** High: a multi-set run is several hours to a day. **Risk.** Medium. Your own test already showed 4 days of target data makes a per-set linear model as good. This targets a different product (day-0 deck rating), not the 0.293 number.

### 8. Masked-card pretraining or a deeper per-set Set Transformer (not recommended)

Masked-card objectives on decks would mostly relearn archetype co-occurrence, which PickNet and the FM already capture. BERT4Rec-style results replicate poorly ([arXiv 2207.07483](https://arxiv.org/abs/2207.07483)). More SAB/ISAB layers per set adds capacity for interactions the data barely contains. The Set Transformer evidence comes from clean synthetic or perceptual tasks ([arXiv 1810.00825](https://arxiv.org/abs/1810.00825)), and your per-set pool transformer (0.235) already lost to the linear deck model.

## Gains of a few thousandths demand paired, multi-seed tests

Your own numbers set the scale. The refit gain is +0.0052 with a bootstrap SD of 0.0016. Ensemble increments have been ≤0.004. The AAIA'18 Hearthstone win-rate challenge's top three finished within 0.1 pp RMSE of each other, the usual plateau under noisy labels ([KnowledgePit](https://knowledgepit.ai/predicting-winrates-of-hearthstone-decks/)). At that resolution, a single-seed, single-set win is not evidence. Bootstrap the test drafts as paired resamples: recompute both models' Spearman on the same resample and take the difference. Run at least 3-5 seeds for neural members and include seed variance in the interval. Screen on MSH, then confirm on all 12 sets and report the mean difference with its interval. Averaging over sets shrinks the noise substantially, provided set-level differences are not correlated with each other. A sensible acceptance bar is a paired 12-set mean gain whose 95% interval excludes zero. On MSH alone, anything under about +0.003 cannot be distinguished from noise.

Two framing cautions apply. First, the published 0.23 ceiling has already been beaten, so re-estimate yours by simulating records from the blend's fitted p under the 7/3 stopping rule. If the blend sits near that ceiling, remaining architecture work has almost no payoff. Second, if the goal is deck-building advice rather than raw Spearman, residualizing on skill proxies DML-style ([Chernozhukov et al., arXiv 1608.00060](https://arxiv.org/abs/1608.00060)) aims the model at deck value. That will probably *lower* the headline metric, and mixing the two goals hides which one improved.

### Recommended experiment plan (pick any)

| Experiment | Steps | Time | Go/no-go |
|---|---|---|---|
| A. Ceiling check | Simulate W/L from the blend's p under the 7/3 stop; compute the Spearman distribution | 1-2 h | If the blend is within about 0.01 of the simulated ceiling, deprioritize B-D |
| B. PickNet features (option 1) | Pick surprise + pool-state PCs + card-vector quantiles into LightGBM; refit protocol; MSH then 12 sets | 1 day | Keep if the paired 12-set CI excludes 0 |
| C. Refit transformer ensemble (option 2) | 5 seeds of DraftEncoder+deck, train+val refit, recency weights, OOF stacking with LightGBM+FM | 1-3 days of compute | Stop if MSH gain is under +0.002 |
| D. PickNet-init game-level model (option 3) | Deck head, per-game Bernoulli, teacher aux; ablate PickNet-init vs random-init | About 1 week | Only worth it if C shows the neural member still adds to the blend |
| E. TFM member (option 6) | TabICLv2 on about 64 compressed features with a recent-20k-row context | 1-2 days | Exploratory only; same paired test |

## Conclusion

Architecture is not where this project will find more Spearman. The literature and your own correlation numbers agree that the deck-strength signal is mostly additive card quality plus faint synergy, measured through heavy binomial noise, and every model family recovers about the same part of it. The open questions are about information and protocol. Can the draft itself reveal something about the drafter that rank does not, which PickNet is uniquely placed to measure? Does a neural member still add to the blend once it gets the same refit-on-recent-data treatment that just gave the trees +0.005? Seen this way, PickNet's main value is its history encoder's view of *how* someone drafted, not its card vectors.

Today's refit result also says something on its own: format drift is now a larger lever than any model change tested. That points to recency-aware training, through weighting or time-decayed fine-tuning, as a recipe to apply to every model, transformer or not. The ceiling simulation should come before any week-long build, because if the blend is already near it, the right next project is a different target (skill-adjusted deck value, or day-0 ratings), not a better 0.293.
