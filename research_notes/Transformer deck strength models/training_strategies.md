# Training strategies for neural deck-strength models under weak, noisy outcome labels

Scope: pretraining, auxiliary/multi-task losses, distillation, noise-robust training/ensembling, cross-set transfer through shared item features, and skill confounding, as they apply to predicting a 17Lands deck's win rate (noise-capped Spearman around 0.3; current best is 0.279 from a LightGBM + FM blend).

Verification legend: "(fetched)" means I read the claim on the primary page this session. "(search snippet)" means it came only from a search-result summary. "(not verified this session)" means it is my prior knowledge and the writer should check it before stating it as fact.

---

## 1. Self-supervised pretraining on sets/baskets (masked item, contrastive, next-pick, deck completion)

### Takeaway
The pretraining signal most relevant here is pick prediction. It has already been run at scale on 17Lands data (DraftFM: 149.5M picks, 29 sets, text-only card features) and transfers zero-shot to unseen sets at 50-60% top-1. Nobody has yet shown that pick-pretraining improves *win-rate* regression. In tabular and learning-to-rank work, pretraining beats GBDTs mainly when labels are scarce, and target-aware pretraining does best. Generic masked-item pretraining (BERT4Rec-style) has a poor replication record.

### Cited Findings
- **DraftFM** (Brian Ward, independent researcher, arXiv 2608.19568, 2026) is a discrete-choice net trained on **149.5M training picks + 7.9M validation picks from 29 expansions (STX through MSH)**. Each card is represented *only* by 391 structured Scryfall features plus a 384-dim frozen bge-small-en-v1.5 embedding of the rules text, with names stripped. There are no card IDs, no set IDs and no post-release stats. The pool is encoded by cross-attention over up to 46 cards, and the output is a softmax over the cards in the current pack. The selected model has d=256 and about 1.64M parameters (fetched) — [DraftFM](https://arxiv.org/html/2608.19568v1)
- DraftFM zero-shot top-1 pick agreement on held-out sets: **BRO 50.8%, FDN 60.4%, MSH 56.7%**, against roughly 7% for random at P1P1. Within-set specialised models reach about 68-71%. The author warns that BRO, FDN and MSH were also used for architecture selection, so the numbers are "optimistic". Agreement is lower on the picks of high-win-rate players (−2.8 to −5.3 normalised points), so the model imitates the population rather than optimal play. The paper makes **no win-rate claims and has no text-vs-ID ablation** (fetched) — [DraftFM](https://arxiv.org/html/2608.19568v1)
- A sealed prospective forecast for an unreleased set (The Hobbit, published 36h before Arena release, Aug 9 2026) matched six expert reviewers at an exact-match rate of 11.8-26.1%. That is within the range at which the humans agree with each other (17.5-30.8%) (fetched) — [DraftFM](https://arxiv.org/html/2608.19568v1)
- **BERT4Rec replicability** (Petrov & Macdonald, RecSys '22): comparisons with SASRec across the literature are inconsistent. Public implementations often fail to reproduce the original numbers with default settings. The original results reproduce only with **up to 30x longer training**, and a HuggingFace reimplementation needs up to 95% less training time. Part of the SASRec/BERT4Rec gap comes from the loss (BCE vs full cross-entropy), not from the masked objective itself (search snippet) — [Petrov & Macdonald](https://arxiv.org/abs/2207.07483); [repo](https://github.com/asash/bert4rec_repro)
- **Rubachev et al. 2022, "Revisiting Pretraining Objectives for Tabular Deep Learning"**: "using the object target labels during the pretraining stage is beneficial for the downstream performance". They advocate target-aware objectives. "Properly performed pretraining significantly increases the performance of tabular DL models, which often leads to their superiority over GBDTs" (fetched abstract; per-dataset numbers not extracted) — [arXiv 2207.03208](https://arxiv.org/abs/2207.03208)
- **Hou et al. (arXiv 2308.00177), "Pretrained deep models outperform GBDTs in Learning-To-Rank under label scarcity"**: SimCLR-style contrastive pretraining on unlabeled rows lets deep rankers beat GBDTs when labels are scarce. The advantage narrows as labels become abundant (summary of fetched PDF; exact NDCG numbers not extracted) — [Hou et al.](https://arxiv.org/pdf/2308.00177)
- **Recformer** ("Text Is All You Need", KDD '23) drops item IDs and represents items as flattened attribute "sentences". It pretrains with MLM plus item-item contrastive objectives and then fine-tunes in two stages. Its gains are strongest in low-resource and cold-start/zero-shot settings (search snippet; magnitudes not extracted) — [arXiv 2305.13731](https://arxiv.org/abs/2305.13731); [repo](https://github.com/AaronHeee/RecFormer)

### Inferences
- The project's situation is "labels scarce and noisy, unlabeled structure abundant": about 5-9 games per deck, but millions of picks. That is the regime where Hou et al. and Rubachev et al. find pretraining beats GBDTs. The strongest candidate is a **DraftFM-style pick model (text + structured features) used as a frozen or fine-tuned card/pool encoder** under a win-rate head.
- Picks encode *population* card valuations. Win-rate needs *deviations* from consensus. The pick-pretrained embedding may therefore mostly rediscover what META / GIH WR features already supply. The ablation that matters is pick-pretrained encoder + META vs META alone.
- Pretraining should be target-aware, following Rubachev: for example, pretrain jointly on next-pick and game-level win, or on masked-card prediction within winning decks only.
- Deck completion (predict the held-out maindeck card from the pool) is a cheap extra objective. It teaches build skill, which is plausibly where the residual signal beyond card quality sits.

### Gaps
- No source found showing that pick/basket pretraining improves a downstream *noisy regression* target. All the evidence is for classification or ranking with clean-ish labels.
- No source compared text-embedding vs ID card representations for 17Lands transfer. DraftFM has no such ablation.
- I could not find published results for contrastive set pretraining (SimCLR over decks) on game outcomes.

---

## 2. Multi-task / auxiliary losses and game-level likelihoods

### Takeaway
Auxiliary targets help in proportion to how related they are to the main target and how much signal they carry. The general MTL evidence is solid. The direct evidence for game-level vs aggregate likelihoods in this domain is thin. The cleanest win is to train on per-game Bernoulli outcomes (with play/draw, opponent rank and mulligans as covariates) rather than on the aggregate win rate.

### Cited Findings
- Statistical-physics analysis of multitask nets: "multitask benefit is correlated with task relatedness and signal-to-noise ratio for related tasks, yet negatively correlated with signal-to-noise ratio for unrelated tasks" (search snippet) — [arXiv 1910.13593](https://arxiv.org/pdf/1910.13593)
- Hierarchical MTL for stock forecasting (a low-SNR domain like this one): auxiliary volume/volatility targets improve all metrics by **1.39% to 7.80%**. The authors argue auxiliary targets give supervision "beyond simple regularization" (search snippet) — [arXiv 2609.25617](https://arxiv.org/pdf/2609.25617)
- **Jane Street Kaggle 1st place** (noisy financial targets): XGBoost + a supervised-autoencoder MLP. The NN *alone* would also have taken first. Techniques used: a Gaussian-noise input layer, joint reconstruction + target-prediction heads, **multiple related targets predicted at once** (5 return horizons), and a "de-noised target" built from all five (search snippet) — [solution writeup](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-01-30-Kaggle-exp-s2/); [Numerai forum](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338)
- Ruder's MTL survey is the standard reference for hard/soft parameter sharing and auxiliary-task selection (not verified this session) — [arXiv 1706.05098](https://arxiv.org/pdf/1706.05098)
- Rigaux & Kashima note that the 7-win/3-loss stopping rule censors the sampling. They attribute their reported noise ceiling (~0.23) to this structure (fetched) — [arXiv 2607.04782](https://arxiv.org/html/2607.04782)

### Inferences
- **Per-game likelihood.** With per-game rows, model logit P(win game) = f(deck) + g(play/draw, opp rank, mulligan, user skill) and maximise a Bernoulli likelihood over games. This handles the stopping rule correctly. The likelihood of a sequence of Bernoulli games factorises, so it is unaffected by the 7-win/3-loss stop, and the per-game covariates absorb noise that the aggregate win rate cannot. Training on the aggregate win rate with MSE weights decks wrongly: a 7-0 and a 0-3 carry very different amounts of information.
- **Per-card in-game auxiliaries.** Predict, per card, "drawn in a won game" vs "lost game" (a GIH-WR-like contrast at the token level). This gives the transformer token-level supervision that is denser than one label per deck.
- Game-length targets are only useful if the data carries turn counts. Play/draw is a covariate, not a target (it is randomised).

### Gaps
- No published study found comparing game-level vs aggregate (binomial/beta-binomial) likelihoods for deck or team win-rate models.
- No evidence found on per-card in-game value as an auxiliary target in card-game models.

---

## 3. Knowledge distillation from GBDT/ensembles; teacher soft labels as denoised targets

### Takeaway
GBDT→NN distillation is an established technique (DeepGBM, KDD '19). Self-distillation theory shows that soft teacher targets act as extra regularisation, and that this helps most under high label noise. For this project, out-of-fold (OOF) LightGBM/FM predictions used as an auxiliary or blended target are the cheapest way to give a transformer a lower-variance signal.

### Cited Findings
- **DeepGBM** (Ke, Xu, Zhang, Bian, Liu; KDD '19 oral) combines CatNN, for sparse categorical features, with **GBDT2NN**, which distils a LightGBM model's knowledge into an NN for dense features. It was motivated by GBDT's weakness on sparse categoricals and on online updates (search snippet; per-dataset numbers not extracted because the PDF was unreadable) — [paper](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf); [repo](https://github.com/motefly/DeepGBM)
- In GBDT2NN, the NN learns the trees' leaf-index embeddings and outputs, with trees grouped and each group distilled into an NN sub-module (not verified this session).
- **Mobahi, Farajtabar, Bartlett (NeurIPS 2020)**: each round of self-distillation progressively restricts the basis functions available, so it *amplifies regularisation*. A few rounds reduce overfitting; too many underfit (search snippet) — [arXiv 2002.05715](https://arxiv.org/pdf/2002.05715)
- **Das & Sanghavi (ICML 2023)** analyse self-distillation under noisy labels. In the high-noise regime the optimal weight on the teacher's predictions can exceed 1, meaning the student should trust the teacher *more* than the raw label (search snippet) — [PMLR v202](https://proceedings.mlr.press/v202/das23d/das23d.pdf)
- "Retraining with Predicted Hard Labels Provably Increases Model Accuracy" (arXiv 2406.11206) is relevant to training on teacher predictions under label noise (title only; claims not verified) — [arXiv 2406.11206](https://arxiv.org/pdf/2406.11206)
- Ensemble knowledge distillation for CTR prediction (arXiv 2011.04106) distils an ensemble of teachers into a single student for click prediction, a noisy binary outcome (title only; numbers not verified) — [arXiv 2011.04106](https://arxiv.org/pdf/2011.04106)

### Inferences
- Recipe: fit the LightGBM+FM blend with K-fold OOF predictions, then train the transformer on loss = α·BCE(game outcomes) + (1−α)·MSE(OOF teacher prediction). Das & Sanghavi suggest the best α is small when the noise is this high. Distillation alone caps the student at roughly teacher quality. Gains above the teacher come only if the student's inductive bias (card interactions) adds information, which the final NN+GBDT blend then captures.
- Teacher predictions must be OOF, or the student will learn teacher overfit.
- Distillation also makes cross-set pooling easier: teachers trained per set can label every set's decks for a single pooled transformer.

### Gaps
- No source found quantifying GBDT→transformer distillation on *set-structured* inputs.
- I could not extract DeepGBM's numeric tables (PDF parsing failed).

---

## 4. Label-noise-robust training, regularisation, SWA, ensembling; when NN+GBDT blends help

### Takeaway
On tabular-like data, single NNs rarely beat tuned GBDTs. Parameter-efficient ensembles of MLPs (TabM) close the gap, and **GBDT + NN ensembles beat either alone**. Under heavy noise, the strongest neural levers are averaging many weak learners and blending with the GBDT, not architectural complexity.

### Cited Findings
- **Grinsztajn et al. (NeurIPS '22 Datasets & Benchmarks)**, 45 datasets: tree models remain SOTA on medium data (~10K samples) even after extensive tuning. They give three reasons NNs lose. NNs are biased toward smooth functions. They are hurt by uninformative features, because their rotation invariance works against them. And they struggle with irregular target functions (search snippet) — [arXiv 2207.08815](https://www.alphaxiv.org/abs/2207.08815)
- **Shwartz-Ziv & Armon (Information Fusion 2022)**: XGBoost beats the deep tabular models, even on the datasets those models' own papers used. However, **an ensemble of the deep models (TabNet, NODE, DNF-Net, 1D-CNN) plus XGBoost achieves the best results**, better than any single model or a non-deep ensemble (search snippet) — [arXiv 2106.03253](https://arxiv.org/pdf/2106.03253)
- **TabM (Gorishniy et al., ICLR 2025)**, 46 datasets: a BatchEnsemble-style MLP with k=32 implicit submodels by default. It is the best tabular-DL model, "easily competes with GBDT" without claiming superiority, and beats FT-Transformer, SAINT, TabR and ModernNCA. Its individual submodels "are weak and overfitted individually, while their collective prediction exhibits substantially better generalization", and TabM beats an ensemble of independent MLPs. Tuned weight decay ranges over {0, LogUniform[1e-4, 1e-1]}, dropout over {0, U[0, 0.5]} (fetched) — [ar5iv 2410.24210](https://ar5iv.labs.arxiv.org/html/2410.24210); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)
- Jane Street 1st place blended XGBoost + an NN, with a Gaussian-noise input layer as the regulariser (search snippet) — [writeup](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-01-30-Kaggle-exp-s2/)
- **SWA** (Izmailov et al., 2018) averages weights along an SGD trajectory with a cyclic or high constant LR, which finds flatter optima and improves generalisation at almost no cost (not verified this session) — [arXiv 1803.05407](https://arxiv.org/abs/1803.05407)
- Hybrid tree-MLP work (Team up GBDTs and DNNs, arXiv 2407.09790) and differentiable GBDT-in-PyTorch packages (GBNet, JOSS) also exist (titles only) — [arXiv 2407.09790](https://arxiv.org/pdf/2407.09790); [GBNet](https://joss.theoj.org/papers/10.21105/joss.08047.pdf)

### Inferences
- The fact that per-set transformers tie linear models is consistent with Grinsztajn's smooth-bias finding and with a target that is mostly additive in card quality.
- Highest expected-value neural moves: (a) a TabM-style or seed ensemble of 16-32 small DraftEncoders with SWA, (b) blending with LightGBM+FM using OOF stacking weights. Even a transformer that is only *as good as* the GBDT should add to a blend if its errors are decorrelated. Measure the correlation between the residuals of the two models before investing further.
- Noise-robust losses designed for flipped classification labels (symmetric CE, co-teaching) are a poor fit. This is aleatoric outcome noise, not mislabeling. The right tool is the correct likelihood (Section 2) plus shrinkage.

### Gaps
- No source quantifying SWA or seed-ensemble gains on regression targets this noisy (Spearman ceiling ~0.3).

---

## 5. Cross-domain/cross-set transfer with shared item features (text/LLM embeddings, adaptors, meta-learning)

### Takeaway
Text-only item representations (UniSRec, Recformer, DraftFM) are the established way to transfer across domains with disjoint item vocabularies. On 17Lands specifically, text alone transfers pick behaviour well, but for win rate it needs META stats or a few days of target-set data. That matches the project's own finding.

### Cited Findings
- **Rigaux & Kashima (Kyoto; arXiv 2607.04782, 2026), DraftEncoder.** Frozen EmbeddingGemma 768-d card-text embeddings, optionally concatenated with 17Lands META win-rate stats, are FiLM-conditioned on player rank and passed through a set-level Transformer for expansion-contextualised cards, then a draft-state encoder over the pick sequence (fetched) — [arXiv 2607.04782](https://arxiv.org/html/2607.04782)
  - Spearman ρ, in-distribution: DraftEncoder about **0.23**, "almost twice" the baselines.
  - New expansion, zero-shot: DraftEncoder 0.0879; **DraftEncoder+Meta 0.1529**; LR+Meta 0.0886; heuristic (mean in-deck WR of the best 23 picks) 0.1391.
  - After 4 days of fine-tuning: DraftEncoder 0.1641; DraftEncoder+Meta 0.1809; LR+Meta 0.0898.
  - Reported noise ceiling about 0.23. The authors say fine-tuning on 4 days closes about 90% of the gap to the meta-augmented model. META features consistently help, and text alone does not generalise to new sets (fetched summary)
  - **Conflict to flag.** The project reports a noise ceiling near 0.3 and a zero-shot DraftEncoder+META of 0.187 vs 0.175. The paper's numbers are lower: different target definition, splits and sets. Its LR+Meta (0.09) is far weaker than the project's per-set linear model with 3-4 days of data. The writer should not mix the two sets of numbers.
- **UniSRec (Hou et al., KDD '22)** uses item description text with parametric whitening and an MoE-enhanced adaptor for universal item representations. Two contrastive pretraining tasks use multi-domain negatives. It allows parameter-efficient transfer to new domains or platforms, including cross-platform transfer with *no overlapping users or items* (search snippet; numeric gains not extracted) — [arXiv 2206.05941](https://arxiv.org/abs/2206.05941)
- **Recformer**: ID-free, text-only, with the strongest gains in zero-shot/cold-start transfer (search snippet) — [arXiv 2305.13731](https://arxiv.org/abs/2305.13731)
- **DraftFM** shows that frozen small-sentence-model embeddings (bge-small, 384-d) of rules text plus structured features, pooled over 29 sets, are enough for 50-60% zero-shot pick agreement (fetched) — [DraftFM](https://arxiv.org/html/2608.19568v1)
- "LLM Embeddings for Deep Learning on Tabular Data" (arXiv 2502.11596) is relevant to encoding categorical items with LLM embeddings (title only) — [arXiv 2502.11596](https://arxiv.org/pdf/2502.11596)

### Inferences
- UniSRec's design maps directly onto this problem: a shared trunk over text-embedded cards, a *per-set lightweight adaptor* (whitening + MoE, or a per-set FiLM/LoRA head), and fine-tuning only the adaptor on 1-4 days of new-set data. That is "mixture of per-domain heads with shared trunk" with evidence behind it.
- A per-set linear model with 3-4 days of data already matches everything. The pooled-transformer niche is therefore days 0-3 of a format, plus possibly a blend component afterwards. The day-0-3 window is real and the DraftEncoder paper confirms it.
- META features act like a per-set hypernetwork input: they summarise the set's environment. A hypernetwork that generates per-set head weights from set-level aggregates (colour-pair win rates, average curve) is a natural next step, but untested.

### Gaps
- No source found applying hypernetworks or MAML-style meta-learning to cross-set card-game outcome prediction.
- No numeric UniSRec/Recformer transfer gains extracted this session.

---

## 6. Causal/skill-confound handling in learned outcome models (brief)

### Takeaway
Skill confounds both deck composition and outcome. 17Lands-based causal work exists but gave no extractable numbers. In practice, condition on skill proxies (rank, user historical win rate) as nuisance covariates so the deck function learns "deck given skill".

### Cited Findings
- "Embarrassingly Causal: Causal Use of Associational Data in Magic The Gathering Drafts" (Mark Louie F. Ramos et al., arXiv 2604.18314). It argues that 17Lands observational data can legitimately guide draft choices despite "substantial confounding, selection effects, and post treatment conditioning" (fetched abstract; methods and numbers not extractable) — [arXiv 2604.18314](https://arxiv.org/abs/2604.18314)
- DraftEncoder FiLM-conditions card embeddings on player rank, so skill enters as a modulating covariate rather than a predictor to be adjusted out (fetched) — [arXiv 2607.04782](https://arxiv.org/html/2607.04782)
- DraftFM finds that population pick behaviour diverges from high-win-rate players' picks. Pick-pretrained representations therefore carry a skill-dependent bias (fetched) — [DraftFM](https://arxiv.org/html/2608.19568v1)
- Double/debiased ML (Chernozhukov et al., 2018) gives Neyman-orthogonal, cross-fitted estimation of a target effect with ML nuisance models (not verified this session) — [arXiv 1608.00060](https://arxiv.org/abs/1608.00060)

### Inferences
- Partialling-out (DML-style): residualise the outcome on skill proxies (user WR from other drafts, rank) with cross-fitting, then fit the deck model to the residual. This removes the "good players draft good-looking decks" shortcut and aims the model at deck value. It will likely *lower* raw Spearman against outcomes while improving usefulness for deck-building advice. The project should decide which target it cares about.
- In a per-game likelihood, opponent rank and user skill fit naturally as additive logit offsets.

### Gaps
- No numeric evidence found on how much skill adjustment changes deck-strength model rankings on 17Lands data.
