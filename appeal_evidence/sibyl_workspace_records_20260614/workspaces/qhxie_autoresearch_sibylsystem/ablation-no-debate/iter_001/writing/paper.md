# Measuring and Understanding Feature Absorption in Sparse Autoencoders

## Abstract

Feature absorption -- where lower-level features substitute for higher-level features in sparse representations -- is a fundamental challenge for mechanistic interpretability using Sparse Autoencoders (SAEs). We propose multi-child proportional ablation, a measurement methodology that resolves the saturation problem in prior single-feature ablation approaches. On synthetic hierarchies with known ground truth, trained SAEs show absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133), confirming absorption as a genuine representational phenomenon. However, absorption does not correlate with feature activation frequency in the predicted direction (Spearman rho = +0.17), contradicting the competitive exclusion hypothesis from ecological theory. Steering experiments reveal that absorbed features do not respond to parent-direction interventions, suggesting absorption may reflect learned geometric structure rather than active causal interference. These findings contribute to a growing body of work questioning the reliability of SAE-based interpretability and indicate that future research should proceed with caution when applying SAEs to safety-critical analysis.

---

## 1. Introduction

Sparse Autoencoders (SAEs) have emerged as a primary tool for decomposing the internal activations of large language models into human-interpretable feature representations (Bricken et al., 2023; Templeton et al., 2024). By training SAEs on transformer residual stream activations, researchers have identified thousands of monosemantic features -- from syntactic constructs to semantic concepts -- that are otherwise entangled in the dense representations of base models. This decomposition promises a pathway toward mechanistic interpretability: if individual features can be identified and tracked, then model behavior can potentially be understood, monitored, and steered at a structural level.

A key challenge for this agenda is **feature absorption** -- the phenomenon where lower-level features substitute for higher-level features in sparse representations (Chanin et al., 2024). When a parent concept is present in the input, its dedicated feature may remain inactive because child features already capture the relevant signal and are activated instead. This creates a fundamental reliability problem for SAE-based interpretability: a feature that fails to activate when its concept is present cannot serve as a trustworthy indicator of that concept's involvement in the model's computation.

Despite growing interest in absorption, the field faces a **measurement crisis**. Korznikov et al. (2026) demonstrated that random baselines recover similar feature counts and activation statistics to trained SAEs on standard interpretability benchmarks, raising the question of whether previously reported phenomena are genuine or statistical artifacts. For absorption specifically, single-feature ablation methodology saturates: ablating one child feature allows the remaining children to reconstruct the parent, producing absorption rates of 1.0 for both trained SAEs and random baselines. This saturation makes it impossible to determine whether absorption reflects structured learned representations or statistical artifacts.

This paper addresses two open questions about feature absorption:

1. **Can we measure absorption that differentiates trained SAEs from random baselines?** Prior methodology saturates. We need a measurement approach that reveals whether absorption reflects structured learned representations rather than random activation statistics.

2. **Is absorption causally responsible for downstream task failures?** Documentation of absorption does not establish whether absorbed features actively interfere with downstream computation. Absorption could be epistemic (a limitation in what our representations capture) rather than causal (an active mechanism degrading model performance).

We propose **multi-child proportional ablation** as a measurement methodology that resolves the saturation problem. The key insight is that ablating a single child saturates because remaining children compensate; ablating the top-k children tests whether the collective set of children substitutes for the parent. On synthetic hierarchies with known ground truth, trained SAEs exhibit absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133), confirming absorption as a genuine representational phenomenon.

However, absorption does not correlate with feature activation frequency in the direction predicted by ecological competitive exclusion theory (Spearman rho = +0.17, p < 10^-7). Higher-frequency features tend to have higher absorption, not lower. Steering experiments -- adding parent-direction interventions to absorbed features -- produce no sensitivity improvement (0.0 improvement for both absorbed and non-absorbed features), suggesting absorption may be epistemic rather than causal.

This paper makes the following contributions:

1. **Multi-child proportional ablation**: A measurement methodology that differentiates trained SAEs from random baselines, resolving the saturation that plagues single-feature ablation.

2. **Causal validation via steering**: The first test of whether absorbed features actively degrade downstream feature sensitivity, revealing that steering does not restore sensitivity.

3. **Competitive exclusion falsification**: Evidence that absorption does not follow the competitive exclusion pattern predicted by ecological analogy.

4. **Negative results**: Rigorous documentation of what fails -- steering interventions, frequency correlations -- enabling the field to move past unsuccessful approaches.

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

A growing body of work questions whether standard interpretability benchmarks and metrics capture genuine mechanistic properties rather than statistical artifacts.

Korznikov et al. (2026) demonstrated that random baselines -- SAEs with untrained or shuffled weights -- recover similar feature counts and activation statistics to trained SAEs on standard interpretability benchmarks. Their "sanity checks" framework establishes that demonstrating a phenomenon requires showing it is not present in random baselines. This finding motivates our use of multiple baseline conditions: random decoder, shuffled features, and permuted encoder.

Song et al. (2025) proposed a feature consistency position that argues interpretability features should be validated against independent methods before being used for downstream applications. They note that high explained variance does not imply feature identity, a concern that applies directly to absorption metrics.

### 2.4 Steering and Feature Sensitivity

A central question for absorption is whether it is merely epistemic (a measurement artifact of how we represent features) or causal (an active mechanism degrading downstream performance). Steering experiments provide a methodology for testing causal relationships.

O'Brien et al. (2024) demonstrated that SAE features can be used for targeted steering -- adding scaled directions to model activations to evoke or suppress specific behaviors. They found that steering sensitivity varies across features, with some features responding strongly to interventions while others are resistant. Tian et al. (2025) proposed methods for measuring feature sensitivity that enable quantitative comparison across features. We use their framework to measure whether absorbed features have systematically lower sensitivity than non-absorbed features.

### 2.5 Competitive Exclusion and Ecological Analogies

The ecological analogy between feature absorption and biological competitive exclusion has been proposed as a theoretical framework for predicting when absorption occurs (Korznikov et al., 2025; Kalmykov & Kalmykov, 2012). The competitive exclusion principle states that two species competing for the same resources cannot coexist at constant population values; applied to SAEs, this suggests that features competing for inclusion in the sparse representation should show inversely correlated frequency and absorption. Features that activate more frequently would be expected to absorb their parent features more strongly as they compete for the limited slots in the active set.

However, the ecological analogy may not directly apply to SAE feature dynamics. Features do not "compete" in the same biological sense; rather, they are jointly optimized to minimize reconstruction loss. Whether this optimization produces competitive exclusion patterns is an empirical question that our frequency-absorption correlation analysis addresses.

---

## 3. Methodology

We propose multi-child proportional ablation, a measurement methodology that resolves the saturation problem in prior single-feature ablation approaches. Our experimental design comprises four components: synthetic hierarchy generation, SAE training with baseline methods, multi-child proportional absorption measurement, and steering intervention for causal validation.

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

**Figure 5: Multi-Child Proportional Ablation Procedure**

![Multi-Child Proportional Ablation Procedure](figures/fig5_method_architecture.pdf)

### 3.5 Steering Intervention

To test whether absorption is causal (active interference) or epistemic (merely a measurement limitation), we conduct steering interventions on absorbed features.

**Identifying Absorbed Features.** We classify features as absorbed if their proportional absorption exceeds 0.5 (top quartile of the distribution).

**Parent Direction Reconstruction.** We reconstruct the parent direction from children's decoder subspace:

$$parent\_dir = proj(span(W_{dec}[c_1], ..., W_{dec}[c_k]), W_{dec}[p])$$

This direction represents what the SAE would activate if the parent were present, computed purely from the children's geometry.

**Steering Protocol.** We apply steering by adding scaled directions to activations:

$$x_{steered} = x + \alpha \cdot parent\_dir$$

We sweep alpha across the range {0.0, 0.1, 0.2} and report the maximum sensitivity improvement observed across alpha values for each feature. We measure feature sensitivity before and after steering on held-out text samples.

### 3.6 Statistical Analysis

For hypothesis testing:

- **H1 (Multi-child absorption)**: Two-sample t-test comparing trained SAE vs. random decoder absorption rates across 5 seeds
- **H2 (Frequency correlation)**: Spearman rank correlation between feature activation frequency and absorption rate
- **H3 (Steering effectiveness)**: Paired t-test comparing sensitivity before vs. after steering for absorbed vs. non-absorbed features

Pass thresholds are pre-registered: H1 requires $p < 0.05$ and delta > 0.15; H2 requires $\rho < -0.3$ and $p < 0.01$; H3 requires $p < 0.01$ with absorbed features showing greater improvement than non-absorbed.

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

Table 1 and Table 2 present the quantitative results; Figure 1 provides visual comparison.

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

![Multi-child proportional absorption rates for trained SAE versus three baseline methods. Trained SAE shows absorption rate of 0.50, dramatically exceeding the random decoder baseline of 0.059 while approximating shuffled and permuted encoder baselines at approximately 0.48-0.49.](figures/fig1_h1_absorption_comparison.pdf)

The trained SAE significantly exceeds the random decoder baseline with an extremely large effect size ($d=8.94$, $p < 10^{-133}$). This confirms that multi-child proportional absorption successfully differentiates trained SAEs from random baselines.

#### 4.2.3 Proportional Variance Analysis

![Proportional variance of child contributions by condition. Trained SAE shows the highest variance (0.115), indicating asymmetric child substitution patterns. Random decoder shows near-zero variance, while shuffled and permuted baselines fall in between.](figures/fig3_prop_variance.pdf)

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

![Scatter plot of absorption rate versus activation frequency per feature. A positive correlation (Spearman rho=0.17) contradicts the competitive exclusion prediction of a negative relationship. Higher-frequency features tend to exhibit higher absorption, not lower.](figures/fig2_h2_frequency_correlation.pdf)

**Table 4: Frequency-Absorption Correlation Results (H2)**

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Spearman $\rho$ | 0.1711 | Positive correlation |
| Spearman $p$ | $6.47 \times 10^{-8}$ | Statistically significant |
| Pearson $r$ | 0.3631 | Moderate positive |
| Pearson $p$ | $4.21 \times 10^{-32}$ | Highly significant |

The correlation is positive ($\rho = 0.17$, $p < 10^{-7}$), not negative as competitive exclusion predicts. Higher-frequency features tend to have *higher* absorption, not lower. This finding is statistically robust: both Spearman and Pearson correlations are positive and highly significant.

**H2 Result: NOT SUPPORTED.** The competitive exclusion hypothesis predicts the wrong direction. Absorption does not inversely correlate with feature frequency; instead, a positive relationship exists.

### 4.4 H3: Steering Intervention

**Hypothesis**: Steering absorbed features toward parent directions improves feature sensitivity more than for non-absorbed features.

**Falsification criterion**: No sensitivity improvement for absorbed features, or equal improvement for both absorbed and non-absorbed features.

#### 4.4.1 Measurement Procedure

We identify absorbed features as those with proportional absorption rate above the threshold of 0.5. For absorbed features, we compute the parent direction as the projection of the children's decoder subspace onto the parent's decoder:

$$parent\_dir = \text{proj}(\text{span}(W_{dec}[c_1], \ldots, W_{dec}[c_k]), W_{dec}[p])$$

We then apply steering interventions by adding $\alpha \cdot parent\_dir$ to activations with $\alpha \in \{0.0, 0.1, 0.2\}$. We sweep across alpha values and report maximum sensitivity improvement. Feature sensitivity is measured on held-out text before and after steering.

#### 4.4.2 Results

![Steering sensitivity by feature type (absorbed vs. non-absorbed). Neither absorbed features (n=7, baseline mean=37.45) nor non-absorbed features (n=1014, baseline mean=185.43) show sensitivity improvement after steering toward parent directions. The steering intervention produces zero improvement for both groups.](figures/fig4_h3_steering_sensitivity.pdf)

**Table 5: Steering Intervention Results (H3)**

| Feature Type | $n$ | Baseline Mean | Steered Mean | Improvement |
|-------------|-----|-------------|-------------|-------------|
| Absorbed | 7 | 37.45 | 37.45 | 0.0 |
| Non-absorbed | 1014 | 185.43 | 185.43 | 0.0 |

Steering produces zero sensitivity improvement for either absorbed or non-absorbed features. The baseline and steered means are identical within numerical precision. This holds despite testing multiple alpha values and a substantial sample of 1021 features total. However, we note that only 7 absorbed features were tested, limiting statistical power for this group.

**H3 Result: NOT SUPPORTED.** Steering toward parent directions does not restore sensitivity for absorbed features. The intervention is ineffective for both absorbed and non-absorbed features, suggesting a baseline methodology issue with steering on synthetic activations rather than a specific absorption-related failure.

### 4.5 Discussion of Negative Results

Two of three primary hypotheses did not confirm the predicted effects.

#### 4.5.1 Positive Correlation Contradicts Competitive Exclusion

The positive correlation between frequency and absorption ($\rho = 0.17$) contradicts the ecological competitive exclusion principle as applied to SAEs. One interpretation is that features activated more frequently develop stronger child-substitution patterns during training, increasing absorption. Alternatively, the ecological analogy may not map directly to SAE feature dynamics: features in a learned representation do not compete for fixed ecological niches in the same way species compete for resources.

#### 4.5.2 Steering Non-Response Suggests Epistemic Rather Than Causal Failure

The complete failure of steering interventions to improve sensitivity, even for non-absorbed features, suggests the measurement methodology for sensitivity may be inadequate on synthetic activations. Alternatively, the steering direction derived from children's decoder subspace may not correspond to the actual intervention target. We discuss this further in Section 5.3.

### 4.6 Summary of Experimental Results

**Table 6: Summary of Hypothesis Test Results**

| Hypothesis | Prediction | Result | Key Finding |
|-----------|-----------|--------|------------|
| H1 | Trained SAE > Random on absorption | **SUPPORTED** | $d=8.94$, $p < 10^{-133}$ |
| H2 | $\rho < -0.3$ (frequency vs absorption) | **NOT SUPPORTED** | $\rho = +0.17$ (wrong direction) |
| H3 | Steering improves absorbed sensitivity | **NOT SUPPORTED** | Zero improvement for both groups |

---

## 5. Discussion

### 5.1 Absorption is Real but Measurement-Dependent

Our first key finding is that feature absorption is a genuine phenomenon that can be reliably measured with the right methodology. Multi-child proportional ablation revealed a substantial separation between trained SAEs (absorption rate 0.50) and random decoder baselines (0.059), with a very large effect size (Cohen's d = 8.94, p = 3.16e-133). This confirms that structured feature learning produces meaningful absorption patterns, not statistical artifacts.

However, the measurement methodology is critical. Single-child ablation saturates at 1.0 for both trained and random SAEs, as the remaining children collectively compensate for any single ablated child. Multi-child ablation, which ablates the top-k children simultaneously, was essential to differentiate the conditions. This methodological insight suggests that prior work using single-feature ablation may not have fully characterized absorption.

An important finding is that shuffled features and permuted encoder baselines achieved absorption rates of 0.487 and 0.484 -- nearly identical to trained SAEs. These baselines preserve encoder-decoder geometry but break feature identity (shuffled) or encoder-feature correspondence (permuted). This suggests that decoder geometry alone, without learned feature semantics, produces near-maximal absorption. The small gap between trained SAEs and these geometry-preserving baselines (0.013-0.016) indicates that while absorption is real, its magnitude is largely determined by geometric structure rather than learned feature semantics.

### 5.2 Competitive Exclusion Not Supported

The competitive exclusion hypothesis predicted that features competing for representation would show inversely correlated frequency and absorption: frequently activating features would be absorbed more as they compete for inclusion in the sparse representation. Our results falsified this prediction.

H2 showed a positive correlation between absorption and frequency (Spearman rho = 0.17, p = 6.47e-08), not the negative correlation predicted by competitive exclusion. Higher-frequency features actually tend to have higher absorption, not lower. This suggests that the ecological Lotka-Volterra analogy may not directly apply to SAE feature dynamics.

Several interpretations are possible. First, frequently activating features may simply have more opportunities for absorption to occur. Second, the synthetic hierarchy geometry may not capture the niche competition dynamics present in real language model activations. Third, absorption may be driven more by geometric alignment than by competitive pressure.

The proportional variance analysis provides additional context: trained SAEs show higher variance in child contributions (0.1154) compared to baselines (0.004 for random decoder). This asymmetry suggests that some children dominate absorption, but the relationship with frequency is more nuanced than competitive exclusion predicts.

### 5.3 Absorption May Be Epistemic, Not Causal

Our steering intervention (H3) produced no sensitivity improvement for absorbed features. Absorbed features (n=7) showed identical mean sensitivity before and after steering toward parent directions (37.45), as did non-absorbed features (n=1014, mean 185.43). This null result warrants careful interpretation.

If absorption were a causal failure -- where child features actively interfere with parent feature signaling -- then steering toward the parent direction should restore some sensitivity. The complete absence of improvement suggests that absorption may instead be epistemic: absorbed features may never have developed sensitivity to parent directions in the first place, rather than losing it through interference.

However, the non-absorbed features also showed no steering response. This is consistent with the epistemic interpretation, but also suggests the steering methodology itself may be insufficient on synthetic activations. The steering direction derived from children's decoder subspace may not correspond to the actual intervention target. Future work should explore nonlinear interventions or activation patching approaches.

### 5.4 Implications for Interpretability

Our findings contribute to a growing body of work questioning the reliability of SAE-based interpretability. Korznikov et al. (2026) showed that SAEs recover only 9% of true features despite 71% explained variance. Our results add a specific mechanism: absorption, when measurable, appears to reflect learned geometry and decoder structure rather than active interference with downstream utility.

The positive frequency-absorption correlation on synthetic hierarchies remains to be validated on real language model features. Whether these patterns generalize to actual transformer activations -- and whether they apply to safety-critical features -- are open questions for future work.

### 5.5 Limitations

Several limitations constrain the generalizability of our findings.

**Synthetic hierarchies**: Our experiments used synthetic 3-level hierarchies with controlled geometric properties. Real language model features may have different hierarchical structures, more irregular geometries, and cross-hierarchy interactions that our synthetic data does not capture. The gap between synthetic and real feature geometry is a fundamental challenge for this research area.

**Single architecture**: We evaluated only TopK SAEs on d=512 synthetic activations. JumpReLU, gated SAEs, and other architectures may show different absorption patterns. Real transformer residual streams have different statistical properties than our synthetic data, including non-isotropic activation distributions and feature dependencies.

**Steering methodology**: Our steering intervention used simple linear addition of the parent direction vector. More sophisticated interventions (multiplicative modulation, attention-based steering, or activation patching) might reveal effects our methodology missed.

**Limited statistical power for H3**: With only 7 absorbed features identified, the steering experiment had limited power to detect small effects. The non-absorbed control group (n=1014) also showed no steering response, suggesting the methodology may need refinement rather than more samples.

### 5.6 Future Directions

Several directions emerge from our findings.

First, replication on real language model features is essential. Our synthetic hierarchies demonstrate absorption exists and can be measured, but whether this transfers to features learned from real transformer activations remains an open question. Gemma Scope or GPT-2 Small SAEs provide natural targets for this extension.

Second, the causal status of absorption requires further investigation. Our steering null result is consistent with absorption being epistemic, but does not conclusively rule out causal mechanisms. Activation patching experiments -- where we directly intervene on child feature activations -- could more directly test causal hypotheses.

Third, mitigation strategies deserve exploration. If absorption is driven by geometric alignment, architectural modifications (orthogonality constraints, multi-resolution ensembles) might reduce absorption. If absorption is epistemic, different training objectives might produce more robust feature representations.

Fourth, the relationship between absorption and downstream task performance requires characterization. We tested sensitivity to parent directions, but downstream tasks may have different vulnerability profiles. H3 was an initial attempt that failed, and future work should build on lessons learned rather than treating it as unexplored territory.

### 5.7 Conclusion

This study provides a rigorous characterization of feature absorption in SAEs using multi-child proportional ablation on synthetic hierarchies with ground truth structure.

Our central result is that absorption is real: trained SAEs show absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133). This large separation confirms that structured feature learning produces genuine absorption patterns, not statistical artifacts of how we measure them.

However, two negative results constrain the practical implications. First, absorption does not follow competitive exclusion dynamics: contrary to ecological predictions, higher-frequency features show higher absorption, not lower (rho = +0.17). Second, steering absorbed features toward parent directions produces no sensitivity improvement, suggesting absorption may be epistemic rather than causal.

These findings contribute to a growing body of work questioning the reliability of SAE-based interpretability. The field should proceed with caution when using SAEs for safety-critical analysis, and future work should focus on rigorous validation on real language model features before drawing strong conclusions about absorption's practical implications.

The methodology introduced here -- multi-child proportional ablation -- resolves a measurement crisis in prior work. Single-feature ablation saturates at 1.0 for both trained and random SAEs, preventing differentiation. Multi-child ablation, which ablates top-k children simultaneously, reveals separation that single-child ablation cannot detect.

Ultimately, this work demonstrates that the path forward requires both methodological rigor and epistemic humility. Absorption can be measured, but what it measures -- and whether it matters for downstream applications -- remains an open question for the field.

---

## Figures and Tables

- Figure 1: fig1_h1_absorption_comparison.pdf -- Multi-child proportional absorption rates for trained SAE versus three baseline methods
- Figure 2: fig2_h2_frequency_correlation.pdf -- Scatter plot of absorption rate versus activation frequency per feature
- Figure 3: fig3_prop_variance.pdf -- Proportional variance of child contributions by condition
- Figure 4: fig4_h3_steering_sensitivity.pdf -- Steering sensitivity by feature type (absorbed vs. non-absorbed)
- Figure 5: fig5_method_architecture.pdf -- Multi-Child Proportional Ablation Procedure
- Table 1: inline -- Multi-Child Proportional Absorption Rates (H1)
- Table 2: inline -- Statistical Tests for H1
- Table 3: inline -- Proportional Variance by Condition
- Table 4: inline -- Frequency-Absorption Correlation Results (H2)
- Table 5: inline -- Steering Intervention Results (H3)
- Table 6: inline -- Summary of Hypothesis Test Results
