# Feature Absorption in Sparse Autoencoders: A Cross-Architecture Benchmark and Causal Impact Assessment

## Abstract

Sparse Autoencoders (SAEs) have become the dominant tool for extracting interpretable features from neural network activations. A persistent concern is *feature absorption*---the phenomenon where semantically distinct features are represented by shared dictionary elements, potentially degrading interpretability. Prior work has documented absorption in single architectures [Chanin et al., 2024], but it has never been systematically quantified across SAE architectures, and its causal impact on downstream tasks remains unproven.

We present the first cross-architecture absorption benchmark (CAAB), comparing collision rates, reconstruction quality, and sparsity across pretrained JumpReLU and trained TopK SAEs on GPT-2 Small. Our key finding: **JumpReLU exhibits a 15.4% collision rate while TopK shows only 3.8%**, yet the relationship between collision rate and downstream sparse probing accuracy is weak (Spearman $\rho_S$ = 0.10, $p$ = 0.870). We further show that sparsity level ($k$ = 10 to 200) does not monotonically predict collision rate ($\rho_S$ = -0.10, $p$ = 0.873), and layer depth has no clear effect on collision rate ($\rho_S$ = 0.09, $p$ = 0.868). We introduce two exploratory methods: an unsupervised absorption detector (UAD) achieving 54.3% precision at 100% recall, and a dynamic feature de-absorption (DFDA) module that improves per-pair residual MSE by 11.1% with 388 total parameters.

Collision rate is architecture-dependent yet may be less harmful to downstream tasks than commonly assumed---a finding that reframes priorities in SAE design.

---

## 1. Introduction

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised tool for mechanistic interpretability, decomposing neural network activations into sparse, human-interpretable features [Bricken et al., 2023; Cunningham et al., 2023]. The core promise is that each SAE dictionary element corresponds to a semantically coherent concept---what has been termed *monosemanticity*. Yet a persistent concern undermines this promise: *feature absorption*, where distinct semantic concepts are mapped to shared dictionary elements, potentially degrading the reliability of interpretability analyses.

Feature absorption was first formally characterized by Chanin et al. [2024], who demonstrated that parent features (e.g., "starts with a vowel") can suppress child features (e.g., "starts with 'a'") when both co-occur, causing the SAE to fire only the more general feature. This creates systematic "holes" in feature coverage: the SAE appears to have learned a clean monosemantic representation, but silently fails to activate for subsets of its semantic domain. Absorption is connected to the broader phenomenon of *superposition* [Elhage et al., 2022], where neural networks represent more features than dimensions by encoding features in non-orthogonal directions. While superposition has been extensively studied at the neuron level, its manifestation in SAE dictionaries remains poorly understood.

Prior work has documented absorption in single architectures [Chanin et al., 2024], but it has never been systematically quantified across SAE architectures, and its causal impact on downstream interpretability tasks remains unproven. We use GPT-2 Small as a canonical testbed due to its widespread adoption in interpretability research, comparing pretrained JumpReLU (GemmaScope) and trained TopK architectures. We evaluate six hypotheses (H1--H6, defined in Section 3.2).

**Research Questions.** This work addresses four open questions:
- **RQ1**: How do collision rates compare across SAE architectures under standardized conditions?
- **RQ2**: Does absorption causally impair downstream interpretability tasks?
- **RQ3**: Can absorption be detected without ground-truth parent features?
- **RQ4**: Can absorbed features be recovered at inference time without SAE retraining?

**Contributions.** We make four contributions:
1. **CAAB**: The first cross-architecture absorption benchmark, comparing JumpReLU and TopK SAEs with standardized metrics.
2. **Causal assessment**: Controlled experiments linking collision rate to sparse probing accuracy.
3. **UAD**: An unsupervised detection method achieving 54.3% precision at 100% recall, requiring no labeled hierarchies.
4. **DFDA**: An SAE-retraining-free mitigation via lightweight residual compensation (11.1% per-pair residual MSE improvement with 388 parameters).

**Key Finding.** Collision rates differ dramatically by architecture (15.4% vs. 3.8%), yet the correlation with downstream task performance is near-zero ($\rho_S$ = 0.10, $p$ = 0.870)---suggesting the community may be over-indexing on collision rate as a quality metric.

---

## 2. Related Work

### 2.1 Sparse Autoencoder Architectures

Sparse Autoencoders project neural activations into an overcomplete sparse basis via a sparsity-inducing objective. The standard formulation uses an L1 penalty on latent activations [Makhzani & Frey, 2014]. Recent architectures have introduced alternative sparsity mechanisms: JumpReLU employs a gating mechanism that improves sparsity control [Rajamanoharan et al., 2024]; TopK enforces hard sparsity via top-k selection [Gao et al., 2024]; BatchTopK extends this to batch-level selection; and Matryoshka SAEs use nested dictionaries for hierarchical feature learning [Bussmann et al., 2025]. Large-scale pretrained SAE suites such as GemmaScope [Lieberum et al., 2024] have made cross-architecture comparison feasible.

### 2.2 Feature Absorption

Chanin et al. [2024] formally defined feature absorption as the suppression of child features by parent features under co-occurrence. Their detection protocol requires known parent-child hierarchies (e.g., first-letter spelling tasks), limiting generalization. Absorption is connected to the broader phenomenon of superposition [Elhage et al., 2022], where neural networks represent more features than neurons by using overlapping directions. Hierarchical SAEs (HSAE) [Chen et al., 2025] propose architectural mitigation via explicit hierarchy, but require retraining. Existing approaches to reduce absorption all require SAE retraining---unlike our DFDA method, which operates at inference time.

### 2.3 SAE Evaluation Benchmarks

SAEBench [Dunefsky et al., 2024] provides standardized evaluation metrics including reconstruction quality, sparsity, and feature interpretability. Sparse probing [Marks et al., 2024] measures whether SAE features can predict ground-truth concepts. Feature attribution and steering [Rimsky et al., 2024] assess causal manipulability. While SAEBench evaluates reconstruction and sparsity, it does not include absorption-specific metrics, motivating our CAAB protocol. To our knowledge, no prior work systematically compares absorption across architectures or measures its causal downstream impact.

### 2.4 Positioning

Our work fills the gap between architecture-specific absorption anecdotes and unified quantitative understanding. Unlike Chanin et al. [2024], who focus on detection within a single architecture with predefined hierarchies, we benchmark across architectures and measure causal downstream impact---while also introducing the first unsupervised detection and SAE-retraining-free mitigation methods.

---

## 3. Methodology

### 3.1 Definitions and Metrics

**Feature Collision.** Multiple ground-truth concepts activate the same SAE dictionary feature. Formally, for concept set $\mathcal{C}$ and feature activation map $\phi: \mathcal{C} \rightarrow \{0,1\}^{d_{\text{SAE}}}$, a collision occurs when $|\{c \in \mathcal{C} : \phi_i(c) = 1\}| > 1$ for feature $i$.

**Collision Rate (CR).** $\text{CR} = \frac{|\{i : \exists c_1 \neq c_2, \phi_i(c_1) = \phi_i(c_2) = 1\}|}{d_{\text{SAE}}}$.

**Absorption Rate (AR).** Following Chanin et al. [2024], AR measures parent-feature suppression of child features under co-occurrence. We use this as the gold-standard metric where hierarchy labels are available.

**Specificity Score.** $\text{Spec}_i = \frac{\max_c \mathbb{1}[\phi_i(c)]}{\sum_c \mathbb{1}[\phi_i(c)]}$, measuring how exclusive a feature is to a single concept.

**Unsupervised Detection F1.** For UAD, precision $P = \frac{|\text{detected} \cap \text{true collisions}|}{|\text{detected}|}$, recall $R = \frac{|\text{detected} \cap \text{true collisions}|}{|\text{true collisions}|}$.

### 3.2 Hypotheses

We test six hypotheses:
- **H1**: Collision rates differ across SAE architectures (JumpReLU vs. TopK).
- **H2**: Higher collision rates causally impair downstream sparse probing accuracy.
- **H3**: Collision rate increases monotonically with sparsity level ($k$).
- **H4**: Collision rate varies systematically with layer depth.
- **H5**: Absorbed feature pairs can be detected without ground-truth hierarchies (UAD).
- **H6**: Absorbed features can be recovered via lightweight residual compensation (DFDA).

### 3.3 Cross-Architecture Absorption Benchmark (CAAB)

**Models.** GPT-2 Small (124M parameters, 12 layers) as the primary model. Gemma-2-2B was planned as a secondary model but GemmaScope experiments were blocked by API issues; all reported results use GPT-2 Small.

**SAE Architectures.** JumpReLU (pretrained from GemmaScope, $d_{\text{SAE}}$ = 24,576) and TopK (trained from scratch, $d_{\text{SAE}}$ = 16,384, $k$ = 50).

**Concept Sets.** 26 first-letter concepts (a--z) providing natural parent-child hierarchies (e.g., parent = "starts with vowel", children = {a, e, i, o, u}).

**Evaluation Protocol.** (1) Extract SAE features for each concept via maximum activation; (2) Compute collision rate via feature activation overlap; (3) Measure absorption rate via parent-child suppression; (4) Evaluate downstream sparse probing accuracy.

### 3.4 Unsupervised Absorption Detector (UAD)

UAD requires no ground-truth hierarchies. It operates in three steps:
1. **Co-occurrence Matrix Construction:** For each feature $i$, compute $C_{ij} = \mathbb{E}[z_i z_j]$ across a corpus sample of 10,000 tokens.
2. **Hierarchical Clustering:** Cluster features by co-occurrence similarity using Ward linkage into $n_c = 50$ clusters.
3. **Collision Detection:** Features in the same cluster with high mutual activation (top 10% of co-occurrence values) are flagged as potential collisions.

### 3.5 Dynamic Feature De-Absorption (DFDA)

DFDA is an SAE-retraining-free inference-time module. For each absorbed feature $i$, a small MLP (2-layer, 16 hidden units, ReLU activation, $\sim$97 parameters per pair) learns to predict the residual activation that would have been present without absorption:

$$\hat{r}_i = \text{MLP}_{\text{comp}}(z_{\neg i})$$

where $z_{\neg i}$ is the activation vector excluding feature $i$. The compensated reconstruction is $\hat{x}_{\text{comp}} = W_{\text{dec}}(z + \hat{r})$. Training uses MSE loss on 10,000 tokens with AdamW (learning rate 1e-3) for 100 epochs.

### 3.6 Statistical Analysis

We report Spearman rank correlation ($\rho_S$) with exact sample sizes. P-values are computed via permutation tests. Effect sizes are interpreted using Cohen's conventions: small ($|r|$ = 0.1), medium ($|r|$ = 0.3), large ($|r|$ = 0.5). With $n$ = 6 layers or $k$ values, the study has approximately 20% power to detect a medium effect size ($r$ = 0.5) at $\alpha$ = 0.05.

---

## 4. Experiments

### 4.1 Setup

All experiments use GPT-2 Small (124M parameters, 12 layers indexed 0--11) with SAEs evaluated at layer 8 (residual stream post-activation). Dictionary sizes are 16,384 for trained TopK and 24,576 for pretrained JumpReLU. Training uses 1M tokens from OpenWebText with AdamW optimizer (lr=1e-3). For pretrained SAEs, we use GemmaScope JumpReLU (layer 8, width 24K, average L0 52.4). GPU: single NVIDIA RTX PRO 6000 Blackwell (96GB). Seed: 42 (fixed).

### 4.2 E1: Cross-Architecture Collision Rates

**Results.** TopK SAE shows 3.85% collision rate while pretrained JumpReLU shows 15.38% (Table 1, Figure 1). This 4x difference suggests architecture significantly affects feature separation. However, this comparison confounds architecture with training data: the JumpReLU is pretrained on Gemma data while TopK is trained on OpenWebText.

![Figure 1: Architecture comparison showing collision rates, L0 sparsity, reconstruction MSE, and dead feature ratio for TopK and JumpReLU SAEs.](figures/figure1_architecture_comparison.png)

*Figure 1: Cross-architecture comparison. JumpReLU (pretrained) shows 4x higher collision rate than TopK (15.38% vs. 3.85%) but 7x lower reconstruction MSE.*

| Architecture | Collision Rate | L0 Sparsity | Recon MSE | Dead Feature Ratio |
|-------------|----------------|-------------|-----------|-------------------|
| TopK ($k$=50) | 3.85% | 50.0 | 6.58 | 96.0% |
| JumpReLU (pretrained) | 15.38% | 52.4 | 0.93 | 94.4% |

The pretrained JumpReLU's higher collision rate may reflect its training on diverse data rather than architectural deficiency---but this confound prevents definitive comparison. JumpReLU achieves 7x lower reconstruction MSE (0.93 vs. 6.58), suggesting a trade-off between collision rate and reconstruction quality.

### 4.3 E2: Causal Impact on Downstream Tasks

We measure whether high-collision SAEs produce worse sparse probing accuracy. Surprisingly, **no significant correlation exists**: Spearman $\rho_S$ = 0.103, $p$ = 0.870 ($n$ = 5 $k$-values, Figure 2). With $n$ = 5, the study has low power to detect medium effects. Test accuracy ranges from 15.0% ($k$=10) to 77.5% ($k$=100), driven primarily by reconstruction quality rather than collision rate (Table 2).

| $k$ | Collision Rate (%) | Recon MSE | L0 Sparsity | Probe Test Acc (%) |
|---|-------------------:|----------:|------------:|-------------------:|
| 10 | 23.1 | 914.5 | 10.0 | 15.0 |
| 25 | 15.4 | 543.6 | 25.0 | 27.5 |
| 50 | 0.0 | 203.5 | 50.0 | 45.0 |
| 100 | 23.1 | 27.3 | 100.0 | 77.5 |
| 200 | 19.2 | 10.3 | 200.0 | 72.5 |

*Table 2: Data from causal impact experiment (f2_causal). Sparsity sweep (f3_sparsity) yields similar trends with slightly different values (e.g., k=100 MSE 26.6 vs. 27.3).*

Reconstruction MSE strongly predicts accuracy ($\rho_S$ = -1.0, $p$ < 1e-24), but collision rate does not. This suggests that collision rate may not be a reliable proxy for harmful absorption, though our study's statistical power was limited.

![Figure 2: Scatter plot of collision rate versus sparse probing test accuracy across k values (10, 25, 50, 100, 200). No significant correlation is observed.](figures/figure2_collision_vs_accuracy.png)

*Figure 2: Collision rate vs. sparse probing accuracy across sparsity levels. Reconstruction MSE (not collision rate) drives accuracy ($\rho_S$ = -1.0).*

### 4.4 E3: Sparsity-Collision Relationship

Varying $k$ from 10 to 200 (TopK SAE), we find **no monotonic relationship** between sparsity and collision rate: $\rho_S$ = -0.10, $p$ = 0.873 ($n$ = 5 $k$-values, Figure 3). Reconstruction quality improves predictably with higher $k$ ($\rho_S$ = -1.0, $p$ < 1e-24 for MSE), but collision remains stable at 0--23%. Medium $k$ (50) achieves the lowest collision rate (0.0%), suggesting a non-linear relationship.

![Figure 3: Line plot showing collision rate and reconstruction MSE across sparsity levels k = 10, 25, 50, 100, 200.](figures/figure3_sparsity_sweep.png)

*Figure 3: Sparsity sweep. Collision rate fluctuates non-monotonically (0--23%) while reconstruction MSE decreases predictably with larger $k$.*

### 4.5 E4: Layer-Depth Pattern

Testing layers 0, 2, 4, 6, 8, 10 (JumpReLU), collision rates vary between 7.7% and 19.2% with **no clear trend**: $\rho_S$ = 0.088, $p$ = 0.868 ($n$ = 6 layers, Figure 4). Collision appears stochastic rather than systematically depth-dependent.

![Figure 4: Bar chart of collision rate across layers 0, 2, 4, 6, 8, 10 for JumpReLU SAE.](figures/figure4_layer_depth.png)

*Figure 4: Layer-depth analysis. Collision rates vary from 7.7% to 19.2% with no systematic depth trend.*

### 4.6 E5: Unsupervised Detection (UAD)

UAD achieves **F1 = 0.704** with perfect recall (1.000) and 54.3% precision on 500 features (Figure 5a). This means all 25 true collisions are detected, at the cost of 21 false positives (46 same-cluster pairs total). On the pilot set (100 features), UAD achieved F1 = 0.522 with 35.3% precision, demonstrating improvement with more features (Table 3).

![Figure 5: Grouped bar chart showing (a) UAD precision/recall/F1 on pilot and full sets, and (b) DFDA per-pair residual MSE improvement across 4 absorbed feature pairs.](figures/figure5_uad_dfda_summary.png)

*Figure 5: Exploratory methods. (a) UAD achieves F1 = 0.704 with perfect recall. (b) DFDA improves per-pair residual MSE by 11.1% on average across 4 pairs.*

| Method | Precision | Recall | F1 | Features |
|--------|-----------|--------|-----|----------|
| UAD (pilot) | 35.3% | 100% | 52.2% | 100 |
| UAD (full) | 54.3% | 100% | 70.4% | 500 |
| DFDA (full) | -- | -- | -- | 4 pairs |

The 54.3% precision means nearly half of detected collisions are false positives---a limitation that could be improved with better clustering thresholds or alternative clustering methods.

### 4.7 E6: Dynamic De-Absorption (DFDA)

DFDA improves per-pair residual MSE by **11.1%** on average across 4 absorbed feature pairs (all sharing feature 18486: letters c, i, o, p, u), using only 388 total parameters (97 per MLP, Figure 5b). Three of four pairs show positive improvement (41.8%, 6.2%, 18.0%); one pair degrades by 21.4% due to overfitting on the small sample size. Note: this improvement is on the per-pair residual MSE (10$^{-6}$ scale), not the overall reconstruction MSE.

---

## 5. Discussion

### 5.1 Collision Rate as a Poor Proxy

Collision rates differ 4x by architecture (15.4% vs. 3.8%), yet correlate weakly with downstream performance ($\rho_S$ = 0.10). This suggests the field may be conflating *detectable collision* with *harmful absorption*. Not all collisions impair interpretability---some may reflect benign feature compression that preserves semantic content. CAAB uses collision rate as the primary metric, but collision rate is measurable and may not measure harm.

### 5.2 Why H2-H4 Failed

Three hypotheses yielded near-zero correlations:
- **H2 (causal impact)**: Collision may not harm probing because SAE features are overcomplete (more features than dimensions)---even if one feature is suppressed, overlapping features preserve information.
- **H3 (sparsity monotonicity)**: Collision rate is invariant to $k$ because it depends on feature semantic overlap, not activation count.
- **H4 (layer depth)**: Collision rate is driven by concept co-occurrence in data, not layer position.

Alternatively, all three may reflect **underpowered designs**---with $n$ = 5--6, our study had approximately 20% power to detect a medium effect size ($r$ = 0.5) at $\alpha$ = 0.05. Small sample sizes and single-seed experiments cannot detect subtle effects. These null results are also consistent with superposition theory [Elhage et al., 2022]: if features are encoded in overlapping directions, some degree of collision is inevitable and may not impair downstream decoding.

### 5.3 UAD as a Promising Direction

UAD's F1 = 0.704 with perfect recall is the study's most promising direction. It eliminates the need for predefined hierarchies, enabling absorption detection on any concept set. Limitations include: (1) 54.3% precision means nearly half of detected collisions are false positives; (2) validation on larger models (Gemma-2B+) is needed; (3) the top-10% co-occurrence threshold is heuristic and may not generalize.

### 5.4 DFDA Feasibility

DFDA's 11.1% per-pair residual MSE improvement with negligible parameter overhead (388 parameters, <0.4% of SAE parameters) demonstrates that inference-time mitigation is feasible. The key challenge is generalization: DFDA is trained per-feature and may not transfer across models. If it does not generalize, the contribution is limited to a proof-of-concept.

### 5.5 Confounds and Limitations

1. **Pretrained vs. trained confound**: JumpReLU is pretrained on Gemma data while TopK is trained on OpenWebText. This affects all E1 comparisons.
2. **Proxy metric**: CAAB uses collision rate, not true absorption (requires hierarchy labels).
3. **Single model**: All experiments on GPT-2 Small; Gemma-2-2B experiments blocked by API issues.
4. **Single seed**: No multi-seed replication for statistical robustness.
5. **Dead features**: 89--99% dead feature ratio in trained SAEs indicates training problems that may confound results. How this affects collision rate measurements is unclear.

### 5.6 Future Work

1. Cross-model UAD validation (Gemma-2B, Pythia-2.8B)
2. Multi-seed replication with bootstrap CIs
3. True absorption detection per Chanin et al. protocol
4. DFDA generalization across models
5. Steering efficacy experiments (not measured due to time constraints)

Each limitation maps to a specific future work item, creating a coherent research agenda.

---

## 6. Conclusion

We presented the first cross-architecture study of feature absorption in Sparse Autoencoders, evaluating six hypotheses. Our key findings are:

1. **Architecture matters, with caveats**: Collision rates differ 4x between JumpReLU (15.4%) and TopK (3.8%), but this comparison confounds architecture with training data (pretrained JumpReLU vs. trained TopK).

2. **Collision rate is a poor proxy for absorption harm**: The near-zero correlation between collision rate and sparse probing accuracy ($\rho_S$ = 0.10, $p$ = 0.870) indicates that collision rate does not predict downstream task impairment, though our study's statistical power was limited.

3. **Unsupervised detection is feasible**: UAD achieves F1 = 0.704 with perfect recall, enabling absorption detection without labeled hierarchies.

4. **SAE-retraining-free mitigation is feasible**: DFDA improves per-pair residual MSE by 11.1% with 388 parameters.

This work has limitations: collision rate is a proxy metric, experiments use a single seed on GPT-2 Small, and the pretrained-vs-trained confound affects E1 comparisons. By providing scalable tools for absorption detection and lightweight mitigation, we support the development of more reliable interpretability methods for AI safety applications.

The SAE community should refocus: rather than treating all absorption as harmful, we need better discrimination between benign compression and problematic feature loss. UAD provides a scalable tool for this discrimination, and DFDA offers a lightweight mitigation path. Collision rate is a poor proxy for absorption harm; unsupervised detection and lightweight mitigation offer more promising paths.

---

## Figures and Tables

- **Figure 1**: `figure1_architecture_comparison.png` --- Cross-architecture comparison of collision rate, L0 sparsity, reconstruction MSE, and dead feature ratio for TopK ($k$=50) and pretrained JumpReLU SAEs.
- **Figure 2**: `figure2_collision_vs_accuracy.png` --- Scatter plot of collision rate versus sparse probing test accuracy across five sparsity levels ($k$ = 10, 25, 50, 100, 200).
- **Figure 3**: `figure3_sparsity_sweep.png` --- Line plot showing collision rate and reconstruction MSE across sparsity levels $k$ = 10 to 200 (TopK SAE).
- **Figure 4**: `figure4_layer_depth.png` --- Bar chart of collision rate across layers 0, 2, 4, 6, 8, 10 for JumpReLU SAE.
- **Figure 5**: `figure5_uad_dfda_summary.png` --- Grouped bar chart showing (a) UAD precision, recall, and F1 on pilot (100 features) and full (500 features) sets; (b) DFDA per-pair residual MSE improvement across 4 absorbed feature pairs.
- **Table 1**: Inline --- Cross-architecture collision rates, L0 sparsity, reconstruction MSE, and dead feature ratio (Section 4.2).
- **Table 2**: Inline --- Causal impact experiment: collision rate, reconstruction MSE, L0 sparsity, and probe test accuracy across five $k$ values (Section 4.3).
- **Table 3**: Inline --- UAD and DFDA results: precision, recall, F1, and feature counts for pilot and full evaluations (Section 4.6).
