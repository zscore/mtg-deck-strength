# Set/Multiset Architectures, Tabular Deep Learning, and Tabular Foundation Models vs GBDT/Linear (for deck win-rate prediction)

Scope note: items marked **[fetched]** were confirmed against a page retrieved during this research (Oct 2026). Items marked **[from memory]** are well-known results cited to the canonical paper URL but not re-fetched in this session. Treat [from memory] numbers as approximately right and check them before quoting exact figures.

## Set architectures (Deep Sets, Set Transformer, Janossy, Perceiver, set2set/slot attention): what interactions they capture, and evidence on set-level regression

### Takeaway
Every permutation-invariant architecture is a pooled sum/attention over element embeddings. A linear bag-of-cards model is already a Deep Sets model with a one-hot phi and a linear rho. A Deep Sets model with a nonlinear rho adds arbitrary interactions, but only through the bottleneck of the pooled vector. Set Transformer, Janossy pooling, and Perceiver add explicit pairwise or k-ary element interactions *before* pooling. On a fixed vocabulary of about 300 cards with sets of 40, the theoretical limits on Deep Sets do not bind. The real limit is whether the data has enough signal to learn interactions at all. I found no published evidence that set transformers beat GBDTs on noisy set-level regression with about 100k examples.

### Cited Findings
- **Deep Sets (Zaheer et al., NeurIPS 2017):** any permutation-invariant f(X) can be written as rho(sum_x phi(x)) for countable domains. Permutation-equivariant layers take the form sigma(lambda*I + gamma*11^T)x. [from memory] — [arXiv 1703.06114](https://arxiv.org/abs/1703.06114)
- **Wagstaff et al. (ICML 2019):** for continuous representation, sum-decomposition needs latent dimension >= M (the maximum set size) to represent all continuous permutation-invariant functions on sets of size <= M. Fixed-dimension constructions need highly discontinuous mappings. [fetched] — [PMLR v97](http://proceedings.mlr.press/v97/wagstaff19a/wagstaff19a.pdf), [arXiv 1901.09006](https://arxiv.org/abs/1901.09006)
- Later work (Wang et al. 2023, "Polynomial width is sufficient for set representation…") argues that polynomial latent width in set size and feature dimension suffices. This partially relaxes the Wagstaff bound. [from memory, not verified this session] — [arXiv 2307.04001](https://arxiv.org/abs/2307.04001)
- **Set Transformer (Lee et al., ICML 2019):**
  - SAB = multi-head self-attention block (O(n^2)).
  - ISAB = attention through m learned inducing points (O(nm)).
  - PMA = pooling by multi-head attention with k learned seed vectors. This replaces mean/sum pooling.
  - Stacked SABs model pairwise and, by composition, higher-order interactions among elements.
  - Reported gains over Deep Sets on max-regression, unique-character counting, amortized clustering, and point-cloud classification. These are mostly synthetic or perceptual tasks with clean targets.
  - [from memory] — [arXiv 1810.00825](https://arxiv.org/abs/1810.00825)
- **Janossy pooling (Murphy et al., ICLR 2019):** averages a permutation-sensitive function over all orderings. k-ary Janossy (applying f to k-subsets) interpolates between Deep Sets (k=1) and full permutation-sensitive models. k=2 captures pairwise interactions explicitly. Tractable approximations: canonical orderings, pi-SGD sampling, k-ary restriction. [from memory] — [arXiv 1811.01900](https://arxiv.org/abs/1811.01900)
- **Perceiver / Perceiver IO (Jaegle et al. 2021):** cross-attention from a small latent array to the inputs, then latent self-attention. This is effectively ISAB-like and decouples depth from input size. It was built for very large inputs (pixels, audio), so it brings no benefit for 40-element sets. [from memory] — [arXiv 2103.03206](https://arxiv.org/abs/2103.03206), [arXiv 2107.14795](https://arxiv.org/abs/2107.14795)
- **Order Matters / set2set (Vinyals et al. 2016)** and **Slot Attention (Locatello et al. 2020):** attention-based iterative pooling, where slots compete for elements. Slot attention targets object discovery, and neither has a tabular or regression track record. [from memory] — [arXiv 1511.06391](https://arxiv.org/abs/1511.06391), [arXiv 2006.15055](https://arxiv.org/abs/2006.15055)
- **A survey of set-function learning (2025)** covers these families and their applications. [fetched title only] — [arXiv 2501.14991](https://arxiv.org/pdf/2501.14991)
- **Closest analog (team-composition win prediction, Dota 2):**
  - Logistic regression on hero picks: about 69.8% test accuracy on 18k matches; about 74.1% with augmented features. [fetched snippet]
  - Some reports claim 88-93% for NN/LSTM. That almost certainly includes in-game features or leakage, because draft-only results cluster around 60-70%.
  - Treat these numbers as unreliable and not comparable. — [arXiv 2106.01782](https://arxiv.org/pdf/2106.01782); [ResearchGate draft study](https://www.researchgate.net/publication/313010005_Performance_of_Machine_Learning_Algorithms_in_Predicting_Game_Outcome_from_Drafts_in_Dota_2)

### Inferences
- **Why the current transformer ties linear.** A deck is a multiset over a fixed vocabulary of about 300 cards, with set size 40. A count vector in R^300 is an injective, continuous encoding. So "linear on counts" = Deep Sets with one-hot phi and linear rho, "MLP on counts" = Deep Sets with a learned nonlinear rho, and the representational ceiling is the same as any set model. The tie (0.235-0.266 vs 0.244, with logit correlations of 0.86-0.93) means the model learned mostly additive card quality. That is expected when the interaction signal is small relative to binomial noise.
- **What a set transformer adds** is an inductive bias for pairwise interactions that is shared across cards through embeddings: cards with similar embeddings get similar interactions. That is the same bias a factorization machine has (pairwise <v_i, v_j>), and the FM already gives the best blend (0.279). A Set Transformer is roughly a deeper, attention-gated FM. Expect gains only if useful interactions are higher-order, or depend on context such as curve, colors, or archetype.
- **Most promising set-architecture variants**, in my judgement (untested):
  - (a) Initialize or tie card embeddings to FM factors, or to card-attribute features (color, CMC, type, rarity, GIH WR), so attention starts from a good place. This reduces data needs.
  - (b) Use 2-ary Janossy or an explicit FM term plus a light SAB stack, trained with a binomial likelihood on wins/games rather than per-draft mean win rate.
  - (c) Pool across sets so card attributes generalize across sets. Per-set training with about 100k noisy rows starves attention.

### Gaps
- I found no paper that benchmarks Deep Sets or Set Transformer against GBDT on noisy set-level regression with about 100k examples. The published Set Transformer evidence is on synthetic or perceptual tasks.
- Draft-only results for Dota/LoL composition models with transformers are scattered and of low quality. I found no rigorous comparison.

## Tabular deep learning vs GBDT: what the benchmarks say, and what helps neural nets

### Takeaway
By 2025-26 the consensus has shifted. Well-tuned GBDTs (CatBoost, LightGBM) remain top single models under standard tuning. MLP-ensemble models (TabM, RealMLP) match or beat them once post-hoc ensembling is used. Attention-based tabular transformers (FT-Transformer, SAINT) are no longer the best neural option: they are costlier and not better than TabM. GBDT and NN ensembles consistently beat either alone. NNs do best on "regular" data (smooth, not heavy-tailed, few uninformative features). Bag-of-counts data is fairly regular, but a near-noise-ceiling target compresses all differences.

### Cited Findings
- **Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022 D&B):**
  - Trees remain state of the art on medium-sized data (about 10K samples), even ignoring speed.
  - Three reasons NNs struggle: (1) poor robustness to uninformative features; (2) MLPs are rotation-invariant and lose the data's natural orientation; (3) NNs are biased toward smooth functions and struggle with irregular targets.
  - [fetched] — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815), [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)
- **McElfresh et al. (NeurIPS 2023 D&B, "TabZilla"):**
  - 19 algorithms, 176 datasets, more than 500k model fits.
  - GBDTs win on "irregular" datasets: skewed, heavy-tailed, or high-variance features. Dataset regularity predicts NN wins.
  - Released 36 "hard" datasets.
  - The paper also reports that for many datasets the NN-vs-GBDT gap is smaller than the gain from tuning, and that CatBoost and TabPFN (v1) were strong. [the last part from memory]
  - [fetched] — [arXiv 2305.02997](https://arxiv.org/pdf/2305.02997), [repo](https://github.com/naszilla/tabzilla)
- **Gorishniy et al. 2021, "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer):**
  - Feature Tokenizer plus Transformer is the strongest universal DL model at the time.
  - Well-tuned ResNet and MLP baselines are competitive.
  - Still "no universally superior solution" vs GBDT.
  - [from memory] — [arXiv 2106.11959](https://arxiv.org/abs/2106.11959)
- **Gorishniy et al. 2022, "On Embeddings for Numerical Features":** piecewise-linear and periodic embeddings of numeric inputs substantially improve MLPs and Transformers, closing much of the gap to GBDT. [from memory] — [arXiv 2203.05556](https://arxiv.org/abs/2203.05556)
- **TabR (Gorishniy et al., ICLR 2024):** a retrieval-augmented MLP with kNN-like attention over training rows. Reported best average among DL models and competitive with GBDT, but heavier. [from memory; listed in search] — [arXiv 2307.14338](https://arxiv.org/pdf/2307.14338)
- **TabM (Gorishniy et al., ICLR 2025):**
  - An MLP with BatchEnsemble-style parameter-efficient ensembling, producing k predictions per row and averaging them.
  - "Easily competes with GBDT and outperforms prior tabular DL models", and is more efficient than attention- and retrieval-based models.
  - The individual predictions are weak and overfit; their average is strong. Gains come from collective training plus weight sharing.
  - Reported 25th of 3300+ in a Kaggle competition using TabM alone.
  - [fetched] — [arXiv 2410.24210](https://arxiv.org/pdf/2410.24210), [repo](https://github.com/yandex-research/tabm)
- **TabArena (Erickson et al., NeurIPS 2025 D&B; living benchmark):**
  - 51 curated IID datasets with 500 to 250k training rows. Elo rating; metrics are ROC AUC, log-loss, and RMSE.
  - Under standard tuning, CatBoost ranks first.
  - With post-hoc ensembling, the top three are TabM, LightGBM, and RealMLP. All three would fall below CatBoost without ensembling.
  - TabPFNv2 and TabICL dominate at 10K rows or fewer.
  - Cross-model ensembles improve further, and the top leaderboard models are not necessarily the highest-weighted ensemble members.
  - [fetched] — [arXiv 2506.16791](https://arxiv.org/html/2506.16791), [tabarena.ai](https://tabarena.ai/)
- **TALENT (Ye et al., 2024-25; JMLR 2025):**
  - 300 datasets (120 binary, 80 multiclass, 100 regression) plus a separate 22-dataset large benchmark.
  - CatBoost has the best average rank in most classification and regression tasks.
  - Among DL methods, ModernNCA and RealMLP are best overall and often statistically tied with CatBoost and LightGBM. ModernNCA is strongest in regression.
  - [fetched] — [arXiv 2407.00956](https://arxiv.org/pdf/2407.00956), [JMLR](http://jmlr.org/papers/volume26/25-0512/25-0512.pdf), [repo](https://github.com/LAMDA-Tabular/TALENT)
- **RealMLP (Holzmüller et al., NeurIPS 2024, "Better by Default"):** an MLP with tuned defaults: robust scaling, numeric embeddings, a specific schedule, and label smoothing. Its pre-tuned defaults rival tuned GBDTs. [from memory; listed in search] — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf)
- **Kadra et al. 2021, "Well-tuned Simple Nets Excel on Tabular Datasets":** a cocktail of regularizers (dropout, weight decay, mixup/cutmix, SWA, shake-shake, and more), tuned per dataset, lets plain MLPs beat XGBoost on 40 datasets. [from memory] — [arXiv 2106.11189](https://arxiv.org/abs/2106.11189)
- **Shwartz-Ziv & Armon 2021, "Tabular Data: Deep Learning Is Not All You Need":**
  - XGBoost beat the then-new DL models (TabNet, NODE, DNF-Net, 1D-CNN) on datasets that those papers did not choose.
  - The best result was an ensemble of XGBoost and the DL models.
  - [from memory] — [arXiv 2106.03253](https://arxiv.org/abs/2106.03253)
- **SAINT (Somepalli et al. 2021):** row and column attention, with contrastive pretraining. **NODE (Popov et al. 2019):** differentiable oblivious-tree ensembles. Neither is a top performer in later independent benchmarks such as TabZilla, TALENT, and TabArena. [from memory] — [arXiv 2106.01342](https://arxiv.org/abs/2106.01342), [arXiv 1909.06312](https://arxiv.org/abs/1909.06312)
- **2026 position paper, "State-of-the-Art Claims Require State-of-the-Art Evidence":** critiques tabular SOTA claims. [title only, fetched in search] — [arXiv 2605.17273](https://arxiv.org/pdf/2605.17273)

### Inferences
- **For this problem**, the relevant result is TabArena: ensembling, not attention, closes the gap. TabM on card counts plus structure features is the highest-expected-value "neural" option. It fits a regime where the target is noisy, rows number about 100k, and features are moderately regular. It is cheap to try and is designed to be averaged with LightGBM.
- **Card-count features** are bounded non-negative integers, mostly 0-2, with no heavy tails. By McElfresh's criterion that is "regular" data, which favors NNs relative to typical tabular data. The many near-zero-signal cards are uninformative features, which Grinsztajn says favor trees. Net effect: expect parity, with gains coming from ensembling diversity.
- **Expected gain size.** With logit correlations of 0.86-0.93 between models and a noise ceiling near 0.3, realistic headroom is a few thousandths of Spearman. The 0.271 to 0.279 FM blend shows that is the scale. Any "beats GBDT" claim needs repeated-seed or cross-validated confidence intervals at about ±0.003.

### Gaps
- No benchmark specifically covers sparse bag-of-items tabular data with near-ceiling noise. TabArena and TALENT are mostly dense, heterogeneous features.
- I could not extract per-dataset-size breakdowns (for example, TabM vs LightGBM at 50-250k rows) from TabArena in this session.

## Tabular foundation models (TabPFN v2/2.5/3/3.5, TabICL/TabICLv2, LimiX, Mitra): limits and usability at about 100k rows x 300 sparse features

### Takeaway
As of October 2026 the frontier has moved fast. TabPFN v2 (Nature, January 2025) was limited to about 10k rows and 500 features. TabPFN-2.5 (November 2025) recommends up to 50k rows and 2,000 features. TabPFN-3 (May 2026) claims to scale to 1M rows on one H100 and to beat 8-hour-tuned GBDTs up to 1M rows and 200 features. TabICLv2 (February 2026, open weights) handles about 600k rows to millions but was pretrained on 100 features or fewer. Our task (about 100k rows, about 300 mostly-zero count features, very low signal-to-noise) is now inside the operating range of TabPFN-3/3.5 and TabICLv2 (via feature reduction). It is untested territory for these models' synthetic priors, which rarely produce SNR this low.

### Cited Findings
- **TabPFN v2 (Hollmann et al., Nature, January 2025):** in-context transformer pretrained on synthetic SCM data. Designed for 10,000 samples or fewer and 500 features or fewer. It outperformed tuned GBDT ensembles in that regime. [from memory] — [Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)
- **TabPFN-2.5 (November 2025):**
  - Scales to 50,000 samples and 2,000 features, a 5x increase in rows and 4x in features over v2. That is the recommended operating range.
  - The limit is "not strict": strong results on benchmarks up to 100k training samples.
  - In exploratory runs, about 160k x 500 (classification) and about 85k x 500 (regression) fit on an H100 80GB with FP16 and FlashAttention-3.
  - "Real-TabPFN-2.5" is a variant fine-tuned on real data.
  - [fetched] — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667)
- **TabPFN-3 (technical report, arXiv May 2026):**
  - "Scale[s] state-of-the-art performance to datasets with 1M training rows."
  - A forward pass outperforms all other models on TabArena, including tuned and ensembled baselines.
  - "Beats 8-hour-tuned gradient-boosted-tree baselines on datasets up to 1M training rows and 200 features."
  - Up to 20x faster than 2.5. A reduced KV cache plus row-chunking reach 1M rows on one H100.
  - The API-only "TabPFN-3-Plus (Thinking)" uses test-time compute and beats all non-TabPFN models by more than 200 Elo on TabArena, rising to 420 Elo on the largest-data subset.
  - These are vendor claims (Prior Labs), not independently replicated in the sources I saw.
  - [fetched abstract] — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- **TabPFN-3.5 (arXiv September 2026):** claims a new state of the art on TabArena and extends to non-IID (temporal or grouped splits), high-cardinality categoricals, and "wide tables with many features". The Fast variant is 3x faster than TabPFN-3; Thinking mode is 12x faster than TabPFN-3-Thinking. The abstract gives no concrete row or feature limits. [fetched abstract] — [arXiv 2609.17895](https://arxiv.org/abs/2609.17895)
- **TabICLv2 (Qu et al., February 2026; soda-inria, open weights):**
  - Claims to surpass RealTabPFN-2.5 (tuned, ensembled, and real-data fine-tuned) on TabArena and TALENT without tuning, including on datasets with more than 20K samples.
  - 10.6x faster than TabPFN-2.5 at 50K samples on an H100: under 10 s for 50K x 100.
  - Handles 600K rows where TabPFN-2.5 runs out of memory, and million-scale datasets in under 50 GB GPU memory with CPU/disk offloading.
  - Pretrained on 300 to 60K rows and "up to 100 features throughout all stages".
  - Uses "QASSMax" scalable softmax to prevent attention fading at large context.
  - [fetched] — [arXiv 2602.11139](https://arxiv.org/html/2602.11139v1), [repo](https://github.com/soda-inria/tabicl)
- **TabICL v1 (ICML 2025):** column-then-row attention for large data. Pretrained on up to 60K rows; handles up to 500K rows. [from memory] — [arXiv 2502.05564](https://arxiv.org/pdf/2502.05564)
- **Mitra (Amazon/AutoGluon, July 2025):** maximum 10,000 rows, 500 features, and 10 classes. Strongest under 5,000 samples and 100 features. [fetched, AutoGluon docs] — [AutoGluon TFM docs](https://auto.gluon.ai/dev/tutorials/tabular/tabular-foundational-models.html). A "Mitra-v2" technical report exists (arXiv 2609.04540), but I did not read its limits. — [arXiv 2609.04540](https://arxiv.org/pdf/2609.04540)
- **LimiX (Stable AI & Tsinghua, September 2025):** one model for classification, regression, imputation, and more, with SCM priors. LimiX-2M (November 2025) is a lighter, faster variant; LimiX-2 is from September 2026. Row and feature limits were not confirmed. [fetched snippets] — [arXiv 2509.03505](https://arxiv.org/html/2509.03505v1), [LimiX-2](https://arxiv.org/html/2609.17488v1)
- **Fine-tuning options:**
  - Real-TabPFN-2.5 is fine-tuned on real data. [fetched]
  - AutoGluon 1.4+ integrates TabPFNv2/2.5, TabICL, and Mitra, with fine-tuning support for some of them. [fetched docs page title; details from memory]
  - "TFM-Retouche" is a lightweight input-space adapter for TFMs. [title only] — [arXiv 2605.06047](https://arxiv.org/pdf/2605.06047)
- **Licensing:** TabPFN weights since v2 carry a non-commercial-style license, and "Plus/Thinking" variants are API-only. TabICLv2 is open. [from memory for the TabPFN license; flag]

### Inferences
- **How to run TFMs at about 100k x 300:**
  - TabPFN-3 local weights or TabPFN-3.5 can likely ingest the data directly. Prefer ≤200 features per the TabPFN-3 claim.
  - For TabICLv2, reduce to 100 features or fewer: aggregated structure features, top cards by variance, or FM/SVD components of the count matrix.
  - Alternatively, use TFMs as a second-stage model on compact features: color pair, curve, removal count, sum of GIH WR, FM score, LightGBM out-of-fold prediction.
- **Why caution is warranted.** TFM priors are synthetic SCMs. A target where binomial noise dominates (about 7 games per row) fits the "noisy regression" part of the prior poorly, and the target is a rate whose noise depends on the number of games. Before trusting a TFM, compare it against LightGBM on identical folds. Consider fitting on drafts with many games, or using the wins/(wins+losses) target with sample weights if supported.
- **Practical value.** The cheapest high-upside experiment is an ensemble member. TabArena shows that cross-family ensembles (TFM + GBDT + MLP) beat single models.

### Gaps
- No public evidence covers how TFMs behave on very low-SNR targets (achievable R^2 of about 0.05-0.1) or on sparse count matrices.
- Exact TabPFN-3/3.5 feature limits and current license terms were not confirmed in this session; the PDF text could not be extracted.
- No independent replication of the TabPFN-3 "beats tuned GBDT up to 1M rows" claim was found.

## CTR-style interaction models (FM, NFM, DeepFM, xDeepFM, DCN-v2, AutoInt) on sparse multi-hot data

### Takeaway
These are the models built for exactly "sparse multi-hot to noisy binary outcome". Under uniform tuning, the many named architectures differ by only about 1e-4 to 4e-4 AUC on Criteo and Avazu, and a well-tuned DNN or FM-plus-DNN is nearly as good as anything. Explicit pairwise-interaction terms (FM, DCN-v2 cross layers) are the useful core. Self-attention (AutoInt) is a cross layer with different parameterization and was not clearly better in reproducible benchmarks.

### Cited Findings
- **FuxiCTR / BARS open benchmark (Zhu et al. 2021-22):**
  - More than 4,600 experiments and more than 12,000 GPU hours re-evaluating 24 models on Criteo and Avazu under a uniform protocol.
  - DNN, PNN, DeepFM, xDeepFM, AutoInt+, and DCN differ by less than 3e-4 AUC on Criteo.
  - Wide&Deep, DeepFM, DCN, and DCN-V2 differ by less than 4e-4 on both Criteo and Avazu.
  - Many reported gains in the literature were not reproducible.
  - [fetched] — [arXiv 2009.05794](https://arxiv.org/pdf/2009.05794v1), [BARS](https://arxiv.org/pdf/2205.09626), [leaderboard](https://openbenchmark.github.io/BARS/CTR/)
- **Factorization Machines (Rendle 2010):** pairwise interactions <v_i, v_j> x_i x_j, computable in O(kn). This is the canonical model for sparse interactions. [from memory] — [FM paper](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
- **NFM (He & Chua 2017):** a bi-interaction pooling layer (sum of pairwise elementwise products of embeddings) followed by an MLP. [from memory] — [arXiv 1708.05027](https://arxiv.org/abs/1708.05027)
- **DeepFM (Guo et al. 2017):** FM and DNN in parallel over shared embeddings. [from memory] — [arXiv 1703.04247](https://arxiv.org/abs/1703.04247)
- **xDeepFM (Lian et al. 2018):** a Compressed Interaction Network that builds explicit vector-wise interactions of bounded degree. [from memory] — [arXiv 1803.05170](https://arxiv.org/abs/1803.05170)
- **DCN-v2 (Wang et al., WWW 2021):** cross layers x_{l+1} = x_0 ⊙ (W x_l + b) + x_l, with a low-rank mixture-of-experts variant. Deployed at Google scale. Each cross layer raises the polynomial degree by one. [from memory] — [arXiv 2008.13535](https://arxiv.org/abs/2008.13535)
- **AutoInt (Song et al., CIKM 2019):** multi-head self-attention over field embeddings with residual connections. This is a "transformer over features", structurally close to a set transformer over card tokens. [from memory] — [arXiv 1810.11921](https://arxiv.org/abs/1810.11921)

### Inferences
- **Our transformer is already AutoInt-like.** The DraftEncoder-style model (attention over card tokens) is essentially AutoInt on multi-hot input. CTR evidence predicts it will roughly tie FM and DNN, which matches what we see.
- **DCN-v2 is a cheap next step.** Input: count vector or card embeddings plus structure features. Add 2-3 low-rank cross layers in parallel with an MLP. It is easy to implement and is a strong, well-reproduced explicit-interaction model that generalizes the FM to degree 3+. Expect it to land near the FM, so its value is as an ensemble member.
- **Fields.** The count multi-hot can also be treated as "fields" (card slot, or color/CMC buckets) so the interaction modules see structure.

### Gaps
- Few CTR papers compare against GBDT on the same data. CTR datasets have billions of rows, a regime where neural nets dominate, so the evidence transfers poorly to 100k rows.

## Distillation from GBDT into NNs, and GBDT+NN ensembling

### Takeaway
Ensembling GBDT and NN is the most reliable, well-replicated route to beating a single GBDT on tabular data. Distilling GBDT into NNs (DeepGBM and successors) is mainly about deployability and online updates, not accuracy over the teacher. On a noisy target, using GBDT out-of-fold predictions as soft targets or as an input feature can reduce the NN's label noise, but no strong published evidence shows the student beats the teacher plus ensemble.

### Cited Findings
- **TabArena:** post-hoc ensembling is essential to measure peak performance, and cross-model ensembles beat the best single model. [fetched] — [arXiv 2506.16791](https://arxiv.org/html/2506.16791)
- **Shwartz-Ziv & Armon 2021:** an ensemble of XGBoost and deep models outperformed XGBoost alone. [from memory] — [arXiv 2106.03253](https://arxiv.org/abs/2106.03253)
- **DeepGBM (Ke et al., KDD 2019, Microsoft):**
  - CatNN handles sparse categorical features (FM-like embeddings). GBDT2NN distills tree structure (leaf indices and outputs) into an NN for dense numerical features.
  - Reported to beat GBDT and NN baselines on several online-prediction tasks, and supports online updates.
  - [fetched snippet; numbers not extracted] — [DeepGBM PDF](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/deepgbm_kdd2019__CR_.pdf)
- **DeepTLF (2022):** encodes data via trained tree nodes (TreeDrivenEncoder) and then trains a DNN. [fetched snippet] — [DeepTLF](https://link.springer.com/article/10.1007/s41060-022-00350-z)
- **Reverse direction:** distilling TFMs into GBDTs ("Pocket Foundation Models", 2026) is a related approach. [title only] — [arXiv 2605.18654](https://arxiv.org/pdf/2605.18654)

### Inferences
- **Recommended options, in order of expected value per effort:**
  1. TabM or RealMLP on the LightGBM feature set, blended with LightGBM and the FM using out-of-fold stacking.
  2. A DCN-v2 or FM-plus-MLP member.
  3. A TFM member (TabPFN-3 or TabICLv2) on compressed features.
  4. A set transformer only with shared card-attribute embeddings, cross-set pretraining, a binomial loss, and the LightGBM out-of-fold logit as an input/residual target (train the NN on residuals of GBDT), so attention only has to model what trees miss.
- **Distillation for denoising.** Training the transformer on GBDT out-of-fold predictions (soft targets) instead of raw 7-game records is a reasonable experiment. It cuts target variance a lot, but the student is capped near the teacher unless it adds new inductive bias.

### Gaps
- I found no published head-to-head of GBDT-to-NN distillation vs plain ensembling on noisy regression targets.
- Concrete DeepGBM gain numbers were not extracted.
