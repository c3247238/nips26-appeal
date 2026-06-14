## 3. Methodology

### 3.1 Overview

Our analysis pipeline operates entirely on pretrained SAEs without any additional training of the autoencoder weights. We evaluate two model families: Gemma-2-2B with JumpReLU SAEs from GemmaScope (Lieberum et al., 2024) and GPT-2 Small with ReLU SAEs from SAELens. For each model, we sample three layers spanning shallow, intermediate, and deeper depths to capture layer-dependent effects. All experiments use a fixed random seed (42) for reproducibility of probe training and data splitting; SAE weights are pretrained and fixed.

Figure 1 illustrates the end-to-end pipeline. WordNet categories provide the semantic foundation; hyponym expansion yields balanced probe datasets; SAE activations are extracted in a single forward pass; logistic probes are trained with L1 regularization; and three absorption metrics are computed from each probe. This design ensures that every step after SAE loading is training-free with respect to the autoencoder, isolating architectural differences in the pretrained representations.

![Figure 1: Semantic probe pipeline for measuring feature absorption. WordNet categories are expanded to 15 hyponyms each, fed through the model to extract SAE activations, then used to train logistic probes for absorption detection.](figures/fig1_pipeline.pdf)

### 3.2 Semantic Probe Construction

**Category selection.** We select 10 semantic categories from WordNet (Miller, 1995): animal, vehicle, food, plant, tool, instrument, container, building, body_part, and substance. These categories span concrete nouns with clear hyponym hierarchies, following the protocol established by Chanin et al. (2024) but expanding the hyponym set for our semantic probe pipeline.

**Hyponym expansion.** For each category, we extract 15 hyponyms (e.g., "dog", "cat", "horse" for animal), expanding from 5 in prior work. This expansion addresses a critical scalability gap: our pilot with 5 hyponyms (Iteration 2) observed 3 failed probes out of 30 (AUROC $<$ 0.7). The current experiment with 15 hyponyms achieves 30/30 valid probes. While we hypothesize that increased hyponym count improves class balance and probe reliability, we cannot rule out other differences between experiments.

**Prompt generation.** Each hyponym is embedded in a template prompt: "A photo of a {hyponym}." This simple template avoids confounding syntactic variation while ensuring the model processes the target concept. Prompts are tokenized and fed through the language model to extract residual stream activations at the target layer.

**Probe training.** We train an L1-regularized logistic regression probe on SAE latent activations. For a given category $c$, the probe learns a weight vector $w \in \mathbb{R}^{d_{\text{SAE}}}$ that predicts whether the input contains a hyponym of $c$. The L1 penalty encourages sparsity in the probe weights, making the top-weighted latent $j^* = \arg\max_j |w_j|$ interpretable as the primary feature for that concept. A probe is considered valid if its test AUROC exceeds 0.7.

The L1 penalty strength $C=1.0$ was chosen as scikit-learn's default; we did not tune this hyperparameter. The choice of $C$ affects which latent is selected as $j^*$ and therefore influences projection absorption values.

Each split has 15 positive examples (one per hyponym in the target category) and 135 total negative examples drawn from other categories. The training split contains 105 examples (15 positive, 90 negative: 6 other categories $\times$ 15 hyponyms), and the test split contains 45 examples (15 positive, 30 negative: 2 other categories $\times$ 15 hyponyms).

### 3.3 Absorption Metrics

We compute three complementary metrics for each probe. Each metric captures a different facet of absorption, and their disagreement is itself informative about metric sensitivity. The ablation-based and projection-based metrics are computed from probe weights, while the training-free detector uses only SAE parameters.

#### 3.3.1 Ablation-Based Metric

The ablation score $A_{\text{abl}}$ measures the accuracy difference between the full probe and the probe with the top latent $j^*$ ablated:

$$
A_{\text{abl}} = \text{acc}_{\text{full}} - \text{acc}_{\text{ablated}}
$$

To ablate latent $j^*$, we zero out $z_{j^*}$ in the SAE activation vector before decoding, then pass the modified reconstruction through the probe without retraining. Following Chanin et al. (2024), we flag absorption when $A_{\text{abl}} < \tau_{\text{abs}}$ (default $\tau_{\text{abs}} = 0.05$). A near-zero score indicates that ablating the top latent does not degrade probe performance, suggesting the concept is redundantly encoded or the latent is functionally inactive. Our experiments confirm the known limitation of this metric: ablation scores are near-universally zero across both architectures (GemmaScope: mean $0.0016 \pm 0.0082$; GPT-2: mean $0.0192 \pm 0.0358$), yielding ablation rates of only 0.0% (GemmaScope E3v2) and 33.3% (GPT-2 E7) respectively despite projection absorption exceeding 90%.

#### 3.3.2 Projection-Based Metric

Projection absorption $A_{\text{proj}}$ measures the fraction of the probe's weight magnitude concentrated in its top latent:

$$
A_{\text{proj}} = \frac{|w_{j^*}|}{\sum_j |w_j|}
$$

The projection ratio $R_{\text{proj}} = 1 - A_{\text{proj}}$ indicates how much weight is distributed across other latents. We classify a probe as absorbed when $A_{\text{proj}} > \tau_{\text{proj}}$ (default $\tau_{\text{proj}} = 0.5$). This metric is substantially more sensitive than ablation: across 60 probes (30 per architecture), 100% exceed the 0.5 threshold, with mean $A_{\text{proj}} = 0.982$ on GemmaScope and $0.912$ on GPT-2. The 7.7% difference is statistically significant ($p < 0.001$) with a large effect size (Cohen's $d = 1.82$), though the absolute difference is modest, supporting the metric's cross-architecture stability.

#### 3.3.3 Training-Free Detector ($A_j$)

The $A_j$ detector, proposed by Chanin et al. (2024), computes a proxy for absorption directly from SAE weights without probe training:

$$
A_j = \frac{\|d_j\|^2}{d_j^\top e_j}
$$

where $d_j \in \mathbb{R}^{d_{\text{model}}}$ is the $j$-th decoder column and $e_j \in \mathbb{R}^{d_{\text{model}}}$ is the $j$-th encoder row (treated as a column vector in the dot product). The numerator is the squared decoder norm; the denominator is the alignment between decoder and encoder. High $A_j$ indicates a decoder vector that is large relative to its encoder alignment, which Chanin et al. hypothesize correlates with absorption.

We compute $A_j$ for the top latent $j^*$ of each probe and correlate it with projection absorption $A_{\text{proj}}$ using Spearman's $\rho$, stratifying by layer to detect layer-dependent patterns.

### 3.4 Cross-Architecture Comparison Protocol

**Matched layer selection.** We select layers at comparable relative depths where possible. Gemma-2-2B has 26 layers; we sample layers 5, 10, and 15 (relative depths 0.19, 0.38, 0.58). GPT-2 Small has 12 layers; we sample layers 5, 8, and 10 (relative depths 0.42, 0.67, 0.83). Layer 8 of GPT-2 (relative depth 0.67) is the closest match to Gemma layer 15 (relative depth 0.58), though exact alignment is impossible given the different layer counts. We match layers by relative depth (layer index / total layers) as a pragmatic heuristic, acknowledging that representational development may not scale linearly. Alternative matching criteria (representation similarity, functional role) would require additional analysis beyond our scope.

**Statistical comparison.** We compare absorption metrics between architectures using the Mann-Whitney $U$ test (non-parametric, no normality assumption) and report Cohen's $d$ for effect size. For correlation comparisons across layers, we use Fisher's r-to-z transformation to compute z-statistics and two-tailed p-values.

**Decoder norm analysis.** We verify decoder norm constraints by computing $\|d_j\|$ across all latents in each SAE. Both architectures show norms tightly constrained around 1.0: GPT-2 ReLU has mean decoder norm $1.000045 \pm 5.4 \times 10^{-6}$ across all layers; GemmaScope JumpReLU also maintains norms near 1.0, though whether this is enforced architecturally or emerges from training is not verified in our analysis. This finding contradicts our original hypothesis that unconstrained decoder norms on GPT-2 would explain detector differences.

### 3.5 Implementation Details

All experiments use PyTorch with the SAELens library (Bloom, 2024) for SAE loading. GemmaScope SAEs are loaded via the GemmaScope hook point API; GPT-2 SAEs use the SAELens pretrained registry. Probes are trained with scikit-learn's `LogisticRegression` (L1 penalty, $C=1.0$, solver='liblinear', max_iter 1000, stratified train/test split). AUROC is computed with `roc_auc_score` on the held-out test split. The $A_j$ computation uses batch matrix operations over all $d_{\text{SAE}}$ latents, with the top latent per probe selected post-hoc (ties broken by lowest index). All code, data, and figure generation scripts are available in the supplementary materials (see Supplementary Materials, Section A).

<!-- FIGURES
- Figure 1: fig1_pipeline_desc.md, fig1_pipeline.pdf --- Semantic probe pipeline architecture diagram (TikZ)
- Table 1: inline --- SAE configurations per model family
-->
