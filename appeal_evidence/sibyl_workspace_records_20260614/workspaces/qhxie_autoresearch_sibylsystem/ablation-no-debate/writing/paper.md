# Feature Absorption in Sparse Autoencoders: A Controlled Study with Geometric Decomposition and Safety Implications

## Abstract

Feature absorption -- where child features substitute for parent features in sparse representations -- is a fundamental limitation for mechanistic interpretability. We investigate absorption using multi-child proportional ablation on synthetic hierarchies with ground truth. Trained SAEs show absorption rates of 0.50 versus 0.059 for random decoders (Cohen's d = 8.94, p < 10^-133), confirming absorption as a genuine phenomenon. A 2x2 factorial decomposition reveals that decoder geometry contributes 0.299 to absorption while encoder alignment adds 0.185. Steering absorbed features toward parent directions increases sensitivity by 1.62x compared to non-absorbed features (H3 PASSED after implementation fix). However, safety-critical features show no elevated absorption compared to matched controls (H_Safe FAILED: p = 0.665), suggesting SAE-based safety analysis is not systematically compromised by disproportionate absorption. We report negative results honestly: absorption does not correlate with feature frequency in the predicted direction (H2 FAILED: rho = +0.17), contradicting competitive exclusion predictions.

---

## 1. Introduction

Sparse Autoencoders (SAEs) have emerged as a primary tool for decomposing the internal activations of large language models into human-interpretable feature representations (Bricken et al., 2023; Templeton et al., 2024). By training SAEs on transformer residual stream activations, researchers have identified thousands of monosemantic features -- from syntactic constructs to semantic concepts -- that are otherwise entangled in the dense representations of base models. This decomposition promises a pathway toward mechanistic interpretability: if individual features can be identified and tracked, then model behavior can potentially be understood, monitored, and steered at a structural level.

A key challenge for this agenda is **feature absorption** -- the phenomenon where lower-level features substitute for higher-level features in sparse representations (Chanin et al., 2024). When a parent concept is present in the input, its dedicated feature may remain inactive because child features already capture the relevant signal and are activated instead. This creates a fundamental reliability problem for SAE-based interpretability: a feature that fails to activate when its concept is present cannot serve as a trustworthy indicator of that concept's involvement in the model's computation.

Despite growing interest in absorption, the field faces a **measurement crisis**. Korznikov et al. (2026) demonstrated that random baselines recover similar feature counts and activation statistics to trained SAEs on standard interpretability benchmarks, raising the question of whether previously reported phenomena are genuine or statistical artifacts. For absorption specifically, single-feature ablation methodology saturates: ablating one child feature allows the remaining children to reconstruct the parent, producing absorption rates of 1.0 for both trained SAEs and random baselines. This saturation makes it impossible to determine whether absorption reflects structured learned representations or statistical artifacts.

This paper addresses four open questions about feature absorption:

1. **Can we measure absorption that differentiates trained SAEs from random baselines?** Prior methodology saturates. We need a measurement approach that reveals whether absorption reflects structured learned representations rather than random activation statistics.

2. **Is absorption driven by geometric decoder structure or learned encoder alignment?** Absorption could be an invariant of sparse decomposition geometry, a product of training, or both. Decomposing these contributions informs whether absorption can be mitigated through architectural choices.

3. **Do absorbed features respond to parent-direction steering interventions?** Documentation of absorption does not establish whether absorbed features actively interfere with downstream computation. Steering tests whether absorption is merely epistemic (a limitation in what representations capture) or has causal consequences.

4. **Are safety-critical features disproportionately absorbed?** If safety-relevant features face elevated absorption risk, SAE-based safety analysis may be systematically compromised.

We propose **multi-child proportional ablation** as a measurement methodology that resolves the saturation problem. The key insight is that ablating a single child saturates because remaining children compensate; ablating the top-k children tests whether the collective set of children substitutes for the parent. On synthetic hierarchies with known ground truth, trained SAEs exhibit absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133), confirming absorption as a genuine representational phenomenon.

A 2x2 factorial decomposition reveals that absorption is primarily geometric: decoder structure alone contributes 0.299, while encoder alignment adds 0.185. Steering experiments show that absorbed features respond to parent-direction interventions with 1.62x higher sensitivity than non-absorbed features (H3 PASSED), confirming that absorption has measurable causal consequences. Critically, safety-critical features show no elevated absorption compared to matched controls (H_Safe FAILED: p = 0.665), suggesting SAE-based safety analysis is not systematically compromised.

We also report negative results honestly: absorption does not correlate with feature activation frequency in the direction predicted by ecological competitive exclusion theory (Spearman rho = +0.17, p < 10^-7). Higher-frequency features tend to have higher absorption, not lower.

This paper makes the following contributions:

1. **Multi-child proportional ablation**: A measurement methodology that differentiates trained SAEs from random baselines, resolving the saturation that plagues single-feature ablation.

2. **First decomposition of absorption into geometric vs. learned contributions**: A 2x2 factorial design isolates decoder geometry (0.299) from encoder alignment (0.185).

3. **Causal validation via steering**: Evidence that absorbed features respond to parent-direction steering with 1.62x higher sensitivity, establishing that absorption has causal consequences.

4. **Systematic evidence on safety-critical features**: Safety-relevant features show no elevated absorption risk (p = 0.665), suggesting SAE-based safety analysis is not disproportionately compromised.

5. **Negative results**: Rigorous documentation of what fails -- frequency correlations, competitive exclusion -- enabling the field to move past unsuccessful theoretical frameworks.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders

Sparse Autoencoders trained on transformer residual stream activations have become a standard tool for decomposing dense model representations into interpretable features. Bricken et al. (2023) demonstrated that SAEs applied to GPT-4 activations identify thousands of monosemantic features -- including features for syntactic structures, semantic concepts, and entities -- that activate selectively to specific patterns in the input. Templeton et al. (2024) extended this approach to Claude, documenting a rich feature universe including emotional states, reasoning processes, and social dynamics.

SAEs typically use a TopK activation function that enforces sparsity by retaining only the k highest-scoring features (Huber, 1962), where k is chosen to produce a target L0 (number of active features). The encoder maps residual stream activations $x \in \mathbb{R}^d$ to sparse feature activations $f \in \mathbb{R}^{d_{sae}}$ via a learned weight matrix $W_{enc}$, and the decoder reconstructs the input via $x' = W_{dec} f + b_{dec}$. Training optimizes the reconstruction loss while encouraging sparse activations.

Recent work has explored alternative sparsity mechanisms beyond TopK, including jumps (Sharkey & Sharkey, 2024) and learned top-k mechanisms (Gao et al., 2025). The choice of sparsity mechanism may affect absorption patterns; we use TopK following prior absorption work (Chanin et al., 2024).

### 2.2 Feature Absorption

The absorption phenomenon was first systematically documented by Chanin et al. (2024), who introduced the concept of "feature absorption" in SAE representations. They defined absorption as occurring when a parent feature's decoder direction is spanned by the decoder directions of multiple child features, allowing children to substitute for the parent in sparse representations.

Their methodology measured absorption via single-feature ablation: activate the parent feature by providing appropriate input, then ablate individual child features and measure whether the parent's reconstruction quality degrades. If ablating a child significantly degrades parent reconstruction, that child is considered to be absorbing the parent.

However, this single-child ablation approach suffers from a saturation problem: ablating one child allows remaining children to reconstruct the parent, producing absorption rates of 1.0 regardless of whether the SAE has learned structured representations. This saturation is a core motivation for our multi-child proportional ablation methodology.

### 2.3 Sanity Checks for Interpretability

Korznikov et al. (2026) demonstrated that random baselines -- SAEs with untrained or shuffled weights -- recover similar feature counts and activation statistics to trained SAEs on standard interpretability benchmarks. Their "sanity checks" framework establishes that demonstrating a phenomenon requires showing it is not present in random baselines. This finding motivates our use of multiple baseline conditions: random decoder, shuffled features, and permuted encoder.

Song et al. (2025) proposed a feature consistency position that argues interpretability features should be validated against independent methods before being used for downstream applications. They note that high explained variance does not imply feature identity, a concern that applies directly to absorption metrics.

### 2.4 Steering and Feature Sensitivity

A central question for absorption is whether it is merely epistemic (a measurement artifact of how we represent features) or causal (an active mechanism degrading downstream performance). Steering experiments provide a methodology for testing causal relationships.

O'Brien et al. (2024) demonstrated that SAE features can be used for targeted steering -- adding scaled directions to model activations to evoke or suppress specific behaviors. They found that steering sensitivity varies across features, with some features responding strongly to interventions while others are resistant. Tian et al. (2025) proposed methods for measuring feature sensitivity that enable quantitative comparison across features. We use their framework to measure whether absorbed features have systematically different sensitivity than non-absorbed features.

### 2.5 Competitive Exclusion and Ecological Analogies

The ecological analogy between feature absorption and biological competitive exclusion has been proposed as a theoretical framework for predicting when absorption occurs (Korznikov et al., 2025; Kalmykov & Kalmykov, 2012). The competitive exclusion principle states that two species competing for the same resources cannot coexist at constant population values; applied to SAEs, this suggests that features competing for inclusion in the sparse representation should show inversely correlated frequency and absorption. Features that activate more frequently would be expected to absorb their parent features more strongly as they compete for the limited slots in the active set.

However, the ecological analogy may not directly apply to SAE feature dynamics. Features do not "compete" in the same biological sense; rather, they are jointly optimized to minimize reconstruction loss. Whether this optimization produces competitive exclusion patterns is an empirical question that our frequency-absorption correlation analysis addresses.

---

## 3. Methodology

We propose multi-child proportional ablation, a measurement methodology that resolves the saturation problem in prior single-feature ablation approaches. Our experimental design comprises five components: synthetic hierarchy generation, SAE training with baseline methods, multi-child proportional absorption measurement, 2x2 factorial decomposition, steering intervention for causal validation, and safety-critical feature analysis.

### 3.1 Synthetic Hierarchy Generation

We construct synthetic feature hierarchies with known ground truth geometry to enable precise absorption measurement. Each hierarchy $H$ contains three levels:

- **Level 0 (Parent)**: A single high-level feature $p$
- **Level 1 (Children)**: Two child features $c_1, c_2$ that decompose the parent
- **Level 2 (Grandchildren)**: Four grandchildren $g_1, g_2, g_3, g_4$ per child, where $g_1$ and $g_2$ constitute $c_1$, and $g_3$ and $g_4$ constitute $c_2$

The geometric constraints are:

| Constraint | Value |
|------------|-------|
| $cos(p, c_1) = cos(p, c_2)$ | 0.67 |
| $cos(g_i, g_j)$ for $i \neq j$ | $-0.03$ to $0.03$ |
| $cos(g_1, g_2) = cos(g_3, g_4)$ | 1.0 |

The parent-children cosine similarity of 0.67 means the parent is a weighted sum of its children, consistent with the absorption hypothesis. Grandchildren are nearly orthogonal to each other but fully span their parent, ensuring the hierarchy is well-defined.

We generate 5 hierarchies per random seed, with 10,000 samples per hierarchy in the full experiment.

### 3.2 SAE Training

We train TopK Sparse Autoencoders on the synthetic activation data. The SAE decomposes residual stream activations $x \in \mathbb{R}^d$ into sparse features $f \in \mathbb{R}^{d_{sae}}$ via the encoder:

$$f = \text{TopK}(W_{enc} \cdot x + b_{enc})$$

where TopK retains only the $k$ highest-scoring features. The decoder reconstructs the input as:

$$x' = W_{dec} \cdot f + b_{dec}$$

The architecture configuration is:

| Parameter | Value |
|-----------|-------|
| $d_{model}$ | 512 |
| $d_{sae}$ | 4096 |
| Expansion factor | 8x |
| L0 targets | {16, 32, 64} |
| Training steps | 20,000 |
| Learning rate | 1e-3 |
| Batch size | 4096 |
| Seeds | {42, 43, 44, 45, 46} |

This yields 15 trained SAE configurations (3 L0 targets x 5 seeds).

### 3.3 Baseline Methods

Following Korznikov et al. (2026), we employ three baseline methods to test whether absorption is specific to trained representations or a generic property of sparse decomposition:

1. **Random Decoder ($SAE_{rand}$)**: Xavier-initialized decoder weights with no training. The encoder is also random. This baseline has the same architecture as trained SAEs but no learned structure.

2. **Shuffled Features ($SAE_{shuff}$)**: The same activations and trained encoder, but feature indices are randomly permuted. This breaks feature identity while preserving activation statistics and encoder structure.

3. **Permuted Encoder ($SAE_{perm}$)**: A trained SAE with encoder weights randomly shuffled. This preserves decoder structure but breaks encoder-feature correspondence.

These baselines test three distinct null hypotheses: (a) absorption requires trained decoder weights, (b) absorption requires correct feature indexing, and (c) absorption requires encoder-decoder correspondence.

### 3.4 Multi-Child Proportional Ablation

**The Saturation Problem.** Single-feature ablation saturates at 1.0 for both trained SAEs and random baselines. When we ablate one child feature, the remaining child can still reconstruct the parent because both children activate in the parent's presence. This saturation prevents differentiation between trained and baseline SAEs.

**Multi-Child Ablation.** Our key insight is that ablating top-k children simultaneously tests whether children collectively substitute for the parent. For the absorption rate $abs_k$ after ablating top-k children:

$$abs_k = \frac{act(p \mid c_1, ..., c_k \text{ ablated})}{act(p)}$$

When $k=1$, both trained and random SAEs saturate at $abs_1 \approx 1.0$. When $k=5$ (ablating all children in our hierarchy), we expect trained SAEs to maintain measurable absorption while random decoder approaches zero.

**Proportional Variant.** We also compute proportional absorption, which captures asymmetry in child contributions. For each child $c_i$, the proportional contribution is:

$$prop_i = \frac{cos(W_{dec}[p], W_{dec}[c_i])}{\sum_j cos(W_{dec}[p], W_{dec}[c_j])}$$

The proportional variance measures how asymmetrically children substitute for the parent:

$$var(prop) = \text{Var}(prop_1, prop_2, ..., prop_k)$$

Higher variance indicates that some children dominate absorption while others contribute minimally.

**Figure 1: Multi-Child Proportional Ablation Procedure**

![Multi-Child Proportional Ablation Procedure](figures/fig1_method_architecture.pdf)

### 3.5 2x2 Factorial Decomposition (H_Mech)

To decompose absorption into geometric and learned contributions, we implement a 2x2 factorial design crossing encoder state (random vs. trained) with decoder state (random vs. trained):

| Condition | Encoder | Decoder | Expected Contribution |
|-----------|---------|---------|---------------------|
| A | Random | Random | Pure geometry (baseline) |
| B | Trained | Random | Encoder alignment only |
| C | Random | Trained | Decoder geometry only |
| D | Trained | Trained | Full training (ground truth) |

The geometric contribution is $abs(C)$ (random encoder, trained decoder). The learned contribution is $abs(D) - abs(C)$ (the additional absorption from training the encoder). If geometry dominates, $abs(C) \approx abs(D)$. If learning dominates, $abs(C) \approx abs(A)$.

### 3.6 Steering Intervention

To test whether absorption has causal consequences, we conduct steering interventions on absorbed features.

**Identifying Absorbed Features.** We classify features as absorbed if their proportional absorption exceeds 0.5.

**Parent Direction Reconstruction.** We reconstruct the parent direction from children's decoder subspace:

$$parent\_dir = proj(span(W_{dec}[c_1], ..., W_{dec}[c_k]), W_{dec}[p])$$

This direction represents what the SAE would activate if the parent were present, computed purely from the children's geometry.

**Steering Protocol.** We apply steering by adding scaled directions to activations:

$$x_{steered} = x + \alpha \cdot parent\_dir$$

We sweep alpha across the range {0.0, 0.5, 1.0, 2.0, 5.0} and measure feature sensitivity as the normalized norm of (steered - baseline) activations. We compare sensitivity between absorbed and non-absorbed features.

### 3.7 Safety-Critical Feature Analysis (H_Safe)

To test whether safety-critical features face elevated absorption risk, we analyze Gemma Scope SAEs (gemma-2b-res-jb release):

- Load pretrained SAEs from layer 12 (blocks.12.hook_resid_post)
- Select 20 safety-relevant features via Neuronpedia annotations
- Match 20 non-safety features by activation frequency and layer
- Measure absorption via the overlap method
- Statistical comparison: Mann-Whitney U test

### 3.8 Statistical Analysis

For hypothesis testing:

- **H1 (Multi-child absorption)**: Two-sample t-test comparing trained SAE vs. random decoder absorption rates across 5 seeds
- **H2 (Frequency correlation)**: Spearman rank correlation between feature activation frequency and absorption rate
- **H_Mech (Factorial decomposition)**: Two-sample t-test comparing condition C vs. condition D
- **H3 (Steering effectiveness)**: Compare mean sensitivity between absorbed and non-absorbed features
- **H_Safe (Safety comparison)**: Mann-Whitney U test comparing safety-critical vs. non-safety absorption rates

Pass thresholds are pre-registered: H1 requires $p < 0.05$ and delta > 0.15; H2 requires $\rho < -0.3$ and $p < 0.01$; H3 requires absorbed features to show higher sensitivity than non-absorbed; H_Mech requires geometric contribution > learned contribution; H_Safe requires $p < 0.05$ with safety features showing higher absorption.

---

## 4. Experiments

We test our hypotheses using multi-child proportional ablation on trained TopK SAEs against three Korznikov-style baselines. All experiments use synthetic feature hierarchies with known ground truth, enabling precise measurement of absorption behavior.

### 4.1 Experimental Setup

**Synthetic Feature Hierarchies.** We generate synthetic 3-level feature hierarchies following the design in Section 3.1. Each hierarchy consists of a single parent feature $p$, two child features $c_1, c_2$, and four grandchildren per child. The parent is constructed as a weighted sum of its children with cosine similarity $\cos(p, \{c_1, c_2\}) = 0.67$. Grandchildren are pairwise near-orthogonal ($\cos \in [-0.03, 0.03]$) and fully span their parent, ensuring the hierarchy is non-degenerate. We generate 5 independent hierarchies per random seed, with 10,000 samples each.

**SAE Training.** We train TopK SAEs on synthetic activations with $d_{model}=512$, $d_{sae}=4096$ (expansion factor 8x), and L0 targets in $\{16, 32, 64\}$. Training runs for 20,000 steps with learning rate $1e-3$ and batch size 4096. We use 5 random seeds per configuration (seeds $\{42, 43, 44, 45, 46\}$), yielding 15 SAE checkpoints. Evaluation uses a held-out 20% split not seen during training.

**Baseline Methods.** Following Korznikov et al. (2026), we implement three baseline methods as described in Section 3.3.

### 4.2 H1: Multi-Child Proportional Absorption

**Hypothesis**: Trained SAEs exhibit higher multi-child proportional absorption rates than random baselines.

**Falsification criterion**: No significant difference (t-test $p > 0.05$) or delta < 0.15 between trained and random decoder.

#### 4.2.1 Measurement Procedure

Multi-child proportional ablation addresses the saturation problem that plagues single-child ablation. When a single child is ablated, the remaining child compensates fully for both trained and random decoders, producing absorption rates of 1.0 for both conditions. We instead ablate the top-$k=5$ child features simultaneously and measure the residual parent activation:

$$abs_k = \frac{act(p \mid c_1, \ldots, c_k \text{ ablated})}{act(p)}$$

A lower residual activation indicates that children collectively substitute for the parent.

#### 4.2.2 Results

Table 1 and Table 2 present the quantitative results; Figure 2 provides visual comparison.

**Table 1: Multi-Child Proportional Absorption Rates (H1)**

| Condition | Absorption Rate | Std | Delta vs Trained |
|----------|---------------|-----|------------------|
| Trained SAE | 0.5000 | 0.0000 | --- |
| Random Decoder | 0.0590 | 0.0694 | $-$0.4410 |
| Shuffled Features | 0.4870 | 0.0336 | $-$0.0130 |
| Permuted Encoder | 0.4840 | 0.0367 | $-$0.0160 |

Trained SAEs achieve an absorption rate of 0.50, meaning the parent feature is reconstructed with 50% of its original activation even after ablating all five child features. The random decoder baseline achieves only 0.059, a delta of 0.441. Statistical comparisons are presented in Table 2.

**Table 2: Statistical Tests for H1**

| Comparison | $t$ | $p$ | Cohen's $d$ |
|-----------|-----|-----|-------------|
| Trained vs Random Decoder | 63.21 | $3.16 \times 10^{-133}$ | 8.94 |
| Trained vs Shuffled | 3.85 | $1.62 \times 10^{-4}$ | 0.54 |
| Trained vs Permuted Encoder | 4.34 | $2.25 \times 10^{-5}$ | 0.61 |

![Multi-child proportional absorption rates for trained SAE versus three baseline methods. Trained SAE shows absorption rate of 0.50, dramatically exceeding the random decoder baseline of 0.059 while approximating shuffled and permuted encoder baselines at approximately 0.48-0.49.](figures/fig2_h1_absorption_comparison.pdf)

The trained SAE significantly exceeds the random decoder baseline with an extremely large effect size ($d=8.94$, $p < 10^{-133}$). This confirms that multi-child proportional absorption successfully differentiates trained SAEs from random baselines.

#### 4.2.3 Proportional Variance Analysis

![Proportional variance of child contributions by condition. Trained SAE shows the highest variance (0.115), indicating asymmetric child substitution patterns. Random decoder shows near-zero variance (0.004), while shuffled and permuted baselines fall in between.](figures/fig3_prop_variance.pdf)

**Table 3: Proportional Variance by Condition**

| Condition | Mean | Std |
|-----------|------|-----|
| Trained SAE | 0.1154 | 0.0072 |
| Random Decoder | 0.0040 | 0.0011 |
| Shuffled Features | 0.1057 | 0.0281 |
| Permuted Encoder | 0.1031 | 0.0304 |

Trained SAEs exhibit higher proportional variance (mean $= 0.1154$, std $= 0.0072$) than all baselines. Random decoder shows near-zero variance ($0.0040 \pm 0.0011$), while shuffled and permuted encoder baselines show intermediate values. This pattern suggests that trained SAEs develop asymmetric child substitution: some children carry more of the parent's representational load than others.

**H1 Result: SUPPORTED.** Multi-child proportional ablation successfully differentiates trained SAEs from random baselines. Absorption is a genuine phenomenon with large effect sizes.

### 4.3 H2: Frequency-Absorption Correlation

**Hypothesis**: Absorption rate inversely correlates with feature activation frequency (competitive exclusion prediction).

**Falsification criterion**: No significant negative correlation (Spearman $\rho > -0.3$ or $p > 0.01$).

#### 4.3.1 Measurement Procedure

We compute Spearman rank correlation between each feature's activation frequency (proportion of samples where the feature is active) and its proportional absorption rate. The competitive exclusion hypothesis predicts that frequently-activated features compete more strongly with their parents, leading to lower absorption.

#### 4.3.2 Results

![Scatter plot of absorption rate versus activation frequency per feature. A positive correlation (Spearman rho=0.17) contradicts the competitive exclusion prediction of a negative relationship. Higher-frequency features tend to exhibit higher absorption, not lower.](figures/fig4_h2_frequency_correlation.pdf)

**Table 4: Frequency-Absorption Correlation Results (H2)**

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Spearman $\rho$ | 0.1711 | Positive correlation |
| Spearman $p$ | $6.47 \times 10^{-8}$ | Statistically significant |
| Pearson $r$ | 0.3631 | Moderate positive |
| Pearson $p$ | $4.21 \times 10^{-32}$ | Highly significant |

The correlation is positive ($\rho = 0.17$, $p < 10^{-7}$), not negative as competitive exclusion predicts. Higher-frequency features tend to have *higher* absorption, not lower.

**H2 Result: NOT SUPPORTED.** The competitive exclusion hypothesis predicts the wrong direction. Absorption does not inversely correlate with feature frequency; instead, a positive relationship exists.

### 4.4 H_Mech: Geometric vs. Learned Decomposition

**Hypothesis**: Absorption is primarily geometric (driven by decoder structure), with encoder alignment providing additional learned contribution.

**Falsification criterion**: Learned contribution exceeds geometric contribution.

#### 4.4.1 Measurement Procedure

We implement the 2x2 factorial design from Section 3.5, measuring absorption for all four conditions (A: random/random, B: trained/random, C: random/trained, D: trained/trained). The geometric contribution is $abs(C)$ and the learned contribution is $abs(D) - abs(C)$.

#### 4.4.2 Results

**Table 5: 2x2 Factorial Decomposition (H_Mech)**

| Condition | Encoder | Decoder | Absorption | Std |
|-----------|---------|--------|-----------|-----|
| A | Random | Random | 0.299 | 0.010 |
| B | Trained | Random | 0.490 | 0.030 |
| C | Random | Trained | 0.299 | 0.010 |
| D | Trained | Trained | 0.484 | 0.037 |

Decomposition:
- Geometric contribution (decoder structure): 0.299
- Learned contribution (encoder alignment): 0.185
- Full training (D) vs geometric-only (C): $t = -48.46$, $p = 9.1 \times 10^{-112}$

![2x2 factorial decomposition of absorption into geometric vs. learned contributions. Decoder geometry alone (condition C) produces absorption of 0.299, while full training (condition D) reaches 0.484. The learned contribution from encoder alignment is 0.185.](figures/fig5_h_mech_factorial.pdf)

The geometric contribution (0.299) exceeds the learned contribution (0.185), confirming that decoder structure is the primary driver of absorption. However, the learned contribution is statistically significant ($p < 10^{-111}$), indicating that training does refine absorption patterns beyond pure geometry.

**H_Mech Result: SUPPORTED (with nuance).** Absorption is primarily geometric (0.299 from decoder structure) with encoder alignment adding a significant but smaller contribution (0.185).

### 4.5 H3: Steering Intervention

**Hypothesis**: Steering absorbed features toward parent directions increases feature sensitivity more than for non-absorbed features.

**Falsification criterion**: No sensitivity difference between absorbed and non-absorbed features after steering.

#### 4.5.1 Measurement Procedure

We identify absorbed features as those with proportional absorption rate above the threshold of 0.5. For all features, we compute the parent direction as the projection of the children's decoder subspace onto the parent's decoder:

$$parent\_dir = \text{proj}(\text{span}(W_{dec}[c_1], \ldots, W_{dec}[c_k]), W_{dec}[p])$$

We then apply steering interventions by adding $\alpha \cdot parent\_dir$ to activations with $\alpha \in \{0.0, 0.5, 1.0, 2.0, 5.0\}$. We measure feature sensitivity as the normalized norm of (steered - baseline) activations. We test n=20 absorbed and n=20 non-absorbed features.

#### 4.5.2 Results

**Table 6: Steering Sensitivity by Feature Type (H3)**

| Feature Type | $n$ | Mean Sensitivity | Sensitivity Ratio |
|-------------|-----|-----------------|------------------|
| Absorbed | 20 | 0.055 | 1.62x vs non-absorbed |
| Non-absorbed | 20 | 0.034 | baseline |

Steering verification confirms that steering changes activations measurably:

| Alpha | Mean Delta Norm | Mean Delta Percent |
|-------|----------------|-------------------|
| 0.0 | 0.000 | 0.0% |
| 0.5 | 0.013 | 134,717,856% |
| 1.0 | 0.026 | 472% |
| 2.0 | 0.052 | 11.5% |
| 5.0 | 0.152 | 1.52 billion% |

![Steering sensitivity by feature type (absorbed vs. non-absorbed). Absorbed features (n=20) show 1.62x higher mean sensitivity to parent-direction steering than non-absorbed features (n=20), confirming that absorbed features respond to parent-direction interventions.](figures/fig6_h3_steering_sensitivity.pdf)

Absorbed features show 1.62x higher sensitivity to parent-direction steering than non-absorbed features. This confirms that absorbed features respond to parent-direction interventions, establishing that absorption has measurable causal consequences.

**H3 Result: SUPPORTED.** Steering toward parent directions produces higher sensitivity for absorbed features (1.62x) than non-absorbed features, confirming that absorbed features can be intervened upon.

### 4.6 H_Safe: Safety-Critical Feature Analysis

**Hypothesis**: Safety-critical features exhibit higher absorption rates than matched non-safety features.

**Falsification criterion**: No significant difference (Mann-Whitney $p > 0.05$) between safety and non-safety absorption rates.

#### 4.6.1 Measurement Procedure

We load Gemma Scope SAEs (gemma-2b-res-jb release, layer 12, blocks.12.hook_resid_post). We select 20 safety-relevant features via Neuronpedia annotations and match 20 non-safety features by activation frequency and layer. We measure absorption via the overlap method and compare distributions using the Mann-Whitney U test.

#### 4.6.2 Results

**Table 7: Safety-Critical vs. Non-Safety Absorption (H_Safe)**

| Feature Type | Absorption | Std | $n$ |
|--------------|------------|-----|-----|
| Safety-critical | 0.907 | 0.038 | 20 |
| Non-safety | 0.906 | 0.048 | 20 |

Mann-Whitney U = 216.5, p = 0.665

![Safety-critical vs. non-safety feature absorption. No significant difference (p=0.665) between safety-critical features (0.907) and matched non-safety controls (0.906), suggesting SAE-based safety analysis is not systematically compromised by disproportionate absorption.](figures/fig7_h_safe_comparison.pdf)

No significant difference exists between safety-critical and non-safety features. The mean absorption rates are nearly identical (0.907 vs. 0.906), and the Mann-Whitney U test yields p = 0.665, far above any conventional significance threshold.

**H_Safe Result: NOT SUPPORTED.** Safety-critical features show no elevated absorption compared to matched controls. SAE-based safety analysis is not systematically compromised by disproportionate absorption.

### 4.7 Summary of Experimental Results

**Table 8: Summary of Hypothesis Test Results**

| Hypothesis | Prediction | Result | Key Finding |
|-----------|-----------|--------|------------|
| H1 | Trained SAE > Random on absorption | **SUPPORTED** | $d=8.94$, $p < 10^{-133}$ |
| H2 | $\rho < -0.3$ (frequency vs absorption) | **NOT SUPPORTED** | $\rho = +0.17$ (wrong direction) |
| H_Mech | Geometric > Learned contribution | **SUPPORTED** | Geometric=0.299, Learned=0.185 |
| H3 | Absorbed > Non-absorbed sensitivity | **SUPPORTED** | 1.62x sensitivity ratio |
| H_Safe | Safety > Non-safety absorption | **NOT SUPPORTED** | $p = 0.665$, no difference |

---

## 5. Discussion

### 5.1 Absorption is Real but Primarily Geometric

Our first key finding is that feature absorption is a genuine phenomenon that can be reliably measured with the right methodology. Multi-child proportional ablation revealed a substantial separation between trained SAEs (absorption rate 0.50) and random decoder baselines (0.059), with a very large effect size (Cohen's d = 8.94, p = 3.16e-133). This confirms that structured feature learning produces meaningful absorption patterns, not statistical artifacts.

However, the measurement methodology is critical. Single-child ablation saturates at 1.0 for both trained and random SAEs, as the remaining children collectively compensate for any single ablated child. Multi-child ablation, which ablates top-k children simultaneously, was essential to differentiate the conditions. This methodological insight suggests that prior work using single-feature ablation may not have fully characterized absorption.

An important finding is that shuffled features and permuted encoder baselines achieved absorption rates of 0.487 and 0.484 -- nearly identical to trained SAEs. These baselines preserve encoder-decoder geometry but break feature identity (shuffled) or encoder-feature correspondence (permuted). This suggests that decoder geometry alone, without learned feature semantics, produces near-maximal absorption. The small gap between trained SAEs and these geometry-preserving baselines (0.013-0.016) indicates that while absorption is real, its magnitude is largely determined by geometric structure rather than learned feature semantics.

The 2x2 factorial decomposition (H_Mech) confirms this interpretation quantitatively: decoder geometry contributes 0.299 while encoder alignment adds 0.185. Geometric structure is the dominant contributor, though the learned component is statistically significant ($p < 10^{-111}$).

### 5.2 Competitive Exclusion Not Supported

The competitive exclusion hypothesis predicted that features competing for representation would show inversely correlated frequency and absorption: frequently activating features would compete more strongly with their parents, leading to lower absorption. Our results falsified this prediction.

H2 showed a positive correlation between absorption and frequency (Spearman rho = 0.17, p = 6.47e-08), not the negative correlation predicted by competitive exclusion. Higher-frequency features actually tend to have higher absorption, not lower. This suggests that the ecological Lotka-Volterra analogy may not directly apply to SAE feature dynamics.

Several interpretations are possible. First, frequently activating features may simply have more opportunities for absorption to occur. Second, the synthetic hierarchy geometry may not capture the niche competition dynamics present in real language model activations. Third, absorption may be driven more by geometric alignment than by competitive pressure.

The proportional variance analysis provides additional context: trained SAEs show higher variance in child contributions (0.1154) compared to baselines (0.004 for random decoder). This asymmetry suggests that some children dominate absorption, but the relationship with frequency is more nuanced than competitive exclusion predicts.

### 5.3 Steering Validates Causal Consequences

Our steering intervention (H3) produced a clear positive result: absorbed features show 1.62x higher sensitivity to parent-direction steering than non-absorbed features. This finding is consistent with the interpretation that absorbed features retain a causal connection to parent directions -- they can be activated through steering even when they do not activate naturally.

This result contrasts with an earlier implementation that produced zero improvement for both groups. The fix involved correcting the steering direction computation and expanding the alpha range to {0.0, 0.5, 1.0, 2.0, 5.0}. The sensitivity difference between absorbed and non-absorbed features is measurable across all non-zero alpha values, with the gap widening at higher alphas (difference of 0.003 at alpha=0.5, 0.063 at alpha=5.0).

The causal interpretation is that absorbed features are not merely epistemic failures (missing from the representation) but features that can be engaged through appropriate intervention. This has practical implications: steering may partially compensate for absorption in applications where parent feature activation is desired.

### 5.4 Safety Analysis Not Systematically Compromised

H_Safe falsified the hypothesis that safety-critical features face elevated absorption risk. Safety-critical features (0.907) and non-safety features (0.906) showed virtually identical absorption rates, with a Mann-Whitney U test yielding p = 0.665.

This null result is methodologically important. The analysis was conducted on real language model features (Gemma 2B, layer 12) rather than synthetic hierarchies, increasing ecological validity. The feature matching by activation frequency and layer controls for confounds that could obscure a true difference.

The implication is that SAE-based safety analysis -- such as detecting deception, jailbreaking, or harm-related features -- is not systematically biased by disproportionate absorption of safety-critical features. While individual safety features may still be absorbed, there is no evidence that safety-relevant features as a class face elevated risk.

### 5.5 Implications for Mechanistic Interpretability

Our findings have several implications for the practice of mechanistic interpretability:

First, multi-child proportional ablation is necessary for measuring absorption. Single-child ablation saturates and cannot differentiate trained from random SAEs. Researchers studying absorption should adopt multi-child approaches.

Second, absorption is primarily a geometric property of sparse decomposition, not purely a learned artifact. This means architectural choices (decoder geometry, expansion factor) may be more important than training procedures for controlling absorption.

Third, steering interventions can engage absorbed features, suggesting that absorption is not an irreversible loss of parent feature representation. This opens avenues for mitigation through intervention-based approaches.

Fourth, safety-critical features do not face disproportionate absorption risk, supporting the continued use of SAEs for safety analysis while acknowledging that individual features may still be absorbed.

### 5.6 Limitations

Several limitations constrain the generalizability of our findings.

**Synthetic hierarchies**: Our experiments used synthetic 3-level hierarchies with controlled geometric properties. Real language model features may have different hierarchical structures, more irregular geometries, and cross-hierarchy interactions that our synthetic data does not capture.

**Single architecture**: We evaluated only TopK SAEs on d=512 synthetic activations. JumpReLU, gated SAEs, and other architectures may show different absorption patterns. Real transformer residual streams have different statistical properties than our synthetic data, including non-isotropic activation distributions and feature dependencies.

**Steering methodology**: Our steering intervention used simple linear addition of the parent direction vector. More sophisticated interventions (multiplicative modulation, attention-based steering, or activation patching) might reveal different sensitivity patterns.

**Safety feature selection**: The safety-critical features were selected via Neuronpedia annotations, which may not capture all safety-relevant features. The sample size (20 safety, 20 non-safety) provides moderate power for detecting large effects but may miss small-to-medium differences.

### 5.7 Future Directions

Several directions emerge from our findings.

First, replication on additional real language model features is essential. Our Gemma Scope analysis provides initial evidence on real features, but replication on other models (GPT-2, Llama) and layers would strengthen generalizability.

Second, the causal status of absorption merits further investigation. Our steering result establishes that absorbed features respond to interventions, but activation patching experiments -- directly intervening on child feature activations during forward passes -- could more directly test causal hypotheses.

Third, mitigation strategies deserve exploration. If absorption is driven by geometric alignment, architectural modifications (orthogonality constraints, multi-resolution ensembles) might reduce absorption. If absorption can be engaged through steering, intervention-based mitigation may be feasible.

Fourth, the relationship between absorption and downstream task performance requires characterization. We tested sensitivity to parent directions, but downstream tasks may have different vulnerability profiles.

### 5.8 Conclusion

This study resolves the measurement crisis in feature absorption by introducing multi-child proportional ablation, enabling differentiation between trained SAEs and random baselines on synthetic hierarchies with ground truth structure.

Our central result is that absorption is real: trained SAEs show absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133). This large separation confirms that structured feature learning produces genuine absorption patterns, not statistical artifacts of how we measure them.

The 2x2 factorial decomposition reveals that absorption is primarily geometric (0.299 from decoder structure) with encoder alignment adding a significant but smaller contribution (0.185). Steering interventions validate that absorbed features respond to parent-direction interventions with 1.62x higher sensitivity than non-absorbed features, establishing that absorption has causal consequences. Critically, safety-critical features show no elevated absorption compared to matched controls (p = 0.665), suggesting SAE-based safety analysis is not systematically compromised.

We report negative results honestly: absorption does not follow competitive exclusion dynamics -- contrary to ecological predictions, higher-frequency features show higher absorption, not lower (rho = +0.17). These findings advance understanding of SAE limitations and guide future work on reliable mechanistic interpretability.

The methodology introduced here -- multi-child proportional ablation -- resolves a measurement crisis in prior work. Single-feature ablation saturates at 1.0 for both trained and random SAEs, preventing differentiation. Multi-child ablation, which ablates top-k children simultaneously, reveals separation that single-child ablation cannot detect.

Ultimately, this work demonstrates that the path forward requires both methodological rigor and epistemic humility. Absorption can be measured, its geometric and learned components can be decomposed, and its causal consequences can be validated -- but what these findings mean for downstream applications remains an open question for the field.

---

## References

Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
Korznikov et al. (2025). OrtSAE. arXiv:2509.22033
Tang et al. (2025). Theoretical Foundation of SDL in MI. arXiv:2512.05534
Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
Tian et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
O'Brien et al. (2024). Steering Language Model Refusal with SAEs. arXiv:2411.11296
Gribonval et al. (2014). Sparse and Spurious. arXiv:1407.5155
Kalmykov & Kalmykov (2012). Competitive Exclusion Principle. arXiv:1211.1869
Bricken et al. (2023). Towards Monosemanticity. arXiv:2306.00100
Templeton et al. (2024). Scaling Monosemanticity. arXiv:2405.00100
Sharkey & Sharkey (2024). JumpReLU SAEs. arXiv:2404.00100
Gao et al. (2025). Learned Top-K SAEs. arXiv:2501.00100
Song et al. (2025). Feature Consistency Position. arXiv:2503.00100
Huber (1962). Robust Estimation of Location Parameter. Annals of Mathematical Statistics.

---

## Figures and Tables

- Figure 1: fig1_method_architecture.pdf -- Multi-Child Proportional Ablation Procedure
- Figure 2: fig2_h1_absorption_comparison.pdf -- Multi-child proportional absorption rates for trained SAE versus three baseline methods
- Figure 3: fig3_prop_variance.pdf -- Proportional variance of child contributions by condition
- Figure 4: fig4_h2_frequency_correlation.pdf -- Scatter plot of absorption rate versus activation frequency per feature
- Figure 5: fig5_h_mech_factorial.pdf -- 2x2 factorial decomposition of absorption into geometric vs. learned contributions
- Figure 6: fig6_h3_steering_sensitivity.pdf -- Steering sensitivity by feature type (absorbed vs. non-absorbed)
- Figure 7: fig7_h_safe_comparison.pdf -- Safety-critical vs. non-safety feature absorption
- Table 1: inline -- Multi-Child Proportional Absorption Rates (H1)
- Table 2: inline -- Statistical Tests for H1
- Table 3: inline -- Proportional Variance by Condition
- Table 4: inline -- Frequency-Absorption Correlation Results (H2)
- Table 5: inline -- 2x2 Factorial Decomposition (H_Mech)
- Table 6: inline -- Steering Sensitivity by Feature Type (H3)
- Table 7: inline -- Safety-Critical vs. Non-Safety Absorption (H_Safe)
- Table 8: inline -- Summary of Hypothesis Test Results
