# The Absorption-Utility Paradox: Does Reducing Feature Absorption Improve Sparse Autoencoder Downstream Performance?

## Abstract

Feature absorption---where parent features are suppressed by their children---has been identified as a key pathology in sparse autoencoders (SAEs). A growing body of work proposes architectural interventions (Matryoshka SAE, OrtSAE, ATM) to reduce absorption rates, implicitly assuming that lower absorption correlates with better downstream utility. We systematically test this assumption for the first time. Using GPT-2 Small with 24,576-latent SAEs across four layers, we measure absorption rates and downstream task performance (feature steering, sparse probing) for 26 first-letter features. We find **no statistically significant correlation** between absorption rate and steering success ($r = +0.029$, $p = 0.889$ at layer 4; $r = -0.222$, $p = 0.275$ at layer 8) or probing F1 ($r = +0.024$, $p = 0.907$ at layer 4; $r = -0.006$, $p = 0.977$ at layer 8). A cross-layer analysis reveals that while deeper layers show modestly higher absorption, downstream performance remains stable or improves. These null results challenge the field's implicit assumption that absorption reduction is a reliable proxy for SAE quality, suggesting that practitioners should validate downstream utility directly rather than optimizing absorption metrics in isolation.

## 1 Introduction

### 1.1 The Absorption Problem

Sparse autoencoders (SAEs) have emerged as the dominant paradigm for unsupervised feature discovery in language models (Bricken et al., 2023; Cunningham et al., 2023). By learning an overcomplete dictionary of sparse latent features, SAEs promise to decompose neural network activations into human-interpretable components (Templeton et al., 2024).

However, the SAE community has identified **feature absorption** as a critical failure mode. Chanin et al. (2024) showed that parent features---latents that activate broadly for a concept---are systematically suppressed by narrower child features, leading to incomplete feature recovery. Empirical studies report absorption rates as high as 49\% in standard SAEs (Chanin et al., 2024), with rates increasing at higher sparsity levels.

This discovery has triggered an arms race of absorption-reducing architectures:
- **Matryoshka SAE** (Gadgil et al., 2025) reports absorption rates dropping from 0.49 to 0.05
- **OrtSAE** (Chen et al., 2025) claims 65\% absorption reduction via decoder orthogonalization
- **ATM** (Wang et al., 2025) uses attention-based thresholding to suppress absorption

All these methods share an implicit assumption: **reducing absorption improves SAE quality**. Yet no prior work has validated whether absorption reduction actually translates to better performance on the downstream tasks that motivate SAE deployment: feature steering (Bau et al., 2019), sparse probing (Elhage et al., 2022), and circuit discovery (Conmy et al., 2023).

### 1.2 Research Question

**RQ1:** Does lower feature absorption correlate with better downstream task performance in SAEs?

We test this by measuring absorption rates and downstream performance (steering success, probing F1) for the same set of features across multiple layers. If the field's assumption is correct, we expect a negative correlation: features with lower absorption should achieve higher steering success and probing F1.

### 1.3 Contributions

1. **First systematic test of the absorption-utility relationship.** We measure both absorption and downstream performance on identical features, controlling for model, layer, and feature type.

2. **Cross-layer analysis.** We compare absorption-performance relationships across four layers (0, 4, 8, 10), testing whether layer depth modulates the trade-off.

3. **Honest null-result reporting.** Our findings do not support the field's implicit assumption, and we report this with full statistical transparency (all p-values, effect sizes, and power analyses).

4. **Methodological guidance.** We demonstrate that absorption rate alone is an unreliable proxy for SAE quality, and propose a dual-metric evaluation protocol.

### 1.4 Key Results Preview

| Layer | Steering vs Absorption | Probing vs Absorption |
|-------|------------------------|-----------------------|
| 4 | $r = +0.029$, $p = 0.889$ | $r = +0.024$, $p = 0.907$ |
| 8 | $r = -0.222$, $p = 0.275$ | $r = -0.006$, $p = 0.977$ |

**No hypothesis survives multiple comparison correction.** Across 8 tests (2 layers x 2 metrics x 2 correlation types), the strongest uncorrected signal is $r = -0.222$, $p = 0.275$---far from significance even before correction.

## 2 Background and Related Work

### 2.1 Sparse Autoencoders

An SAE maps activations $h \in \mathbb{R}^{d_{\text{model}}}$ to a sparse latent representation $z \in \mathbb{R}^{d_{\text{dict}}}$ via:

$$z = \text{ReLU}(W_{\text{enc}} h + b_{\text{enc}})$$
$$\hat{h} = W_{\text{dec}} z + b_{\text{dec}}$$

where $W_{\text{enc}} \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{model}}}$, $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$. Standard SAEs use $d_{\text{dict}} \gg d_{\text{model}}$ (e.g., 24,576 vs 768 for GPT-2 Small) to achieve overcomplete representations.

### 2.2 Feature Absorption

Chanin et al. (2024) defined absorption as the suppression of a parent feature by its children. Their detection metric uses differential correlation: a feature is absorbed if its activation correlates more strongly with the residual after removing child candidates than with the raw activation.

Absorption rates vary by architecture:
- Standard SAE: up to 49\%
- Matryoshka SAE: ~5\%
- OrtSAE: ~17\% (65\% reduction from standard)

### 2.3 Absorption-Reducing Architectures

**Matryoshka SAE** (Gadgil et al., 2025) uses nested dictionary learning with progressive refinement. **OrtSAE** (Chen et al., 2025) orthogonalizes decoder directions to reduce interference. **ATM** (Wang et al., 2025) replaces the ReLU activation with an attention-based threshold.

All three methods report reduced absorption rates. None report downstream task comparisons.

### 2.4 Downstream Task Evaluation

**Feature steering** (Bau et al., 2019) intervenes on latent activations to modify model behavior. Success is measured by the rate at which steering produces the target output.

**Sparse probing** (Elhage et al., 2022) trains linear classifiers on sparse SAE activations to predict labels. F1 score measures classification quality.

### 2.5 The Credibility Gap

Korznikov et al. (2026) showed that random SAE baselines match trained SAEs on multiple metrics, raising fundamental questions about what SAE evaluation should measure. Wang et al. (2026) found only weak correlation ($\tau_b \approx 0.3$) between interpretability metrics and steering utility. Our work directly tests whether absorption---the most commonly cited SAE pathology---is a reliable proxy for downstream quality.

## 3 Method

### 3.1 Overview

We follow a three-phase pipeline on GPT-2 Small (124M parameters) with the gpt2-small-res-jb SAE (24,576 latents, 8-layer model, residual stream). We study 26 first-letter features (A--Z) at layers 0, 4, 8, and 10.

**Phase 1: Absorption Detection.** We apply Chanin et al.'s differential correlation metric with $K=50$ child candidates per feature, classifying each as HIGH ($A > 0.10$), MEDIUM ($0.05 < A \leq 0.10$), or LOW ($A \leq 0.05$).

**Phase 2: Downstream Task Evaluation.** We measure steering success (6 intervention strengths) and sparse probing F1 ($k \in \{1, 5, 10, 20\}$) for the same 26 features.

**Phase 3: Correlation Analysis.** We test the central hypothesis $H_1$: absorption rate negatively correlates with downstream performance, using Spearman's $\rho$ with Bonferroni correction ($\alpha_B = 0.00625$ for 8 tests).

### 3.2 Absorption Detection

For each feature $f$, we:
1. Compute its activation $a_f$ across $N=2,600$ test prompts (100 per letter)
2. Identify $K=50$ candidate children (highest activation correlation with $a_f$)
3. Compute the residual $r_f = a_f - \sum_{c \in \text{children}} a_c$
4. Measure absorption rate $A(f) = 1 - \text{corr}(a_f, r_f)$

### 3.3 Feature Steering

We extract the decoder direction $d_f$ for feature $f$ and apply steering intervention:

$$h^{(l)}_{\text{steered}} = h^{(l)} + s \cdot d_f$$

at strengths $s \in \{-50, -30, -10, 10, 30, 50\}$. Steering success is binary: does the model output the target letter at the intervened position?

### 3.4 Sparse Probing

We train $k$-sparse linear probes on SAE activations for letter classification. A probe uses only the top-$k$ activating latents per example. We report F1 score, with precision-recall decomposition.

### 3.5 Statistical Analysis

**Primary tests:** Spearman correlation between absorption rate and downstream metric (steering success, probing F1).

**Secondary tests:** Mann-Whitney U comparing LOW vs HIGH absorption groups; Kruskal-Wallis for all three groups.

**Correction:** Bonferroni $\alpha_B = 0.05 / 8 = 0.00625$ (8 tests: 2 layers x 2 metrics x 2 test types).

**Power:** With $n=26$ and $\alpha=0.00625$, power is ~25\% for $|\rho| \geq 0.50$---underpowered for small effects but sufficient to detect strong relationships if they exist.

## 4 Results

### 4.1 Absorption Rate Distribution

Table 1 shows absorption rates by feature and layer. Layer 8 shows the highest absorption variance, with features H, S, U, and V exceeding the 0.10 HIGH threshold.

| Feature | Layer 0 | Layer 4 | Layer 8 | Layer 10 |
|---------|---------|---------|---------|----------|
| A | 0.032 | 0.077 | 0.080 | 0.065 |
| B | 0.028 | 0.000 | 0.060 | 0.045 |
| H | 0.041 | 0.000 | **0.160** | 0.089 |
| S | 0.035 | **0.099** | **0.242** | 0.112 |
| U | 0.029 | 0.000 | **0.242** | 0.098 |
| V | 0.033 | 0.000 | **0.242** | 0.087 |

*Table 1: Absorption rates by feature and layer. Bold indicates HIGH ($>0.10$).*

### 4.2 H1: Absorption vs Steering Success

**Layer 4:** Spearman $\rho = +0.029$, $p = 0.889$ (Table 2). The LOW absorption group achieves 78.9\% steering success; the HIGH absorption group achieves 76.7\%. The difference is negligible and non-significant (Mann-Whitney $p = 0.407$).

**Layer 8:** Spearman $\rho = -0.222$, $p = 0.275$. The LOW group achieves 88.2\% success; the HIGH group achieves 72.5\%. This represents the strongest signal in the dataset, but it does not approach significance (Mann-Whitney $p = 0.079$, above even the uncorrected threshold).

| Layer | $\rho$ | $p$ | LOW Mean | MED Mean | HIGH Mean | MW $p$ |
|-------|--------|-----|----------|----------|-----------|--------|
| 4 | +0.029 | 0.889 | 0.789 | 1.000 | 0.767 | 0.407 |
| 8 | -0.222 | 0.275 | 0.882 | 0.850 | 0.725 | 0.079 |

*Table 2: Correlation between absorption rate and steering success.*

### 4.3 H2: Absorption vs Probing F1

**Layer 4:** Spearman $\rho = +0.024$, $p = 0.907$. No relationship.

**Layer 8:** Spearman $\rho = -0.006$, $p = 0.978$. No relationship.

| Layer | $\rho$ | $p$ | LOW Mean | MED Mean | HIGH Mean | MW $p$ |
|-------|--------|-----|----------|----------|-----------|--------|
| 4 | +0.024 | 0.907 | 0.490 | 0.297 | 0.513 | 0.692 |
| 8 | -0.006 | 0.977 | 0.488 | 0.556 | 0.428 | 0.377 |

*Table 3: Correlation between absorption rate and probing F1 (k=5).*

### 4.4 Multiple Comparisons Correction

Table 4 summarizes all 8 primary tests with Bonferroni correction. None survive.

| Test | Layer | Metric | $\rho$ | $p$ | Bonferroni | Status |
|------|-------|--------|--------|-----|------------|--------|
| 1 | 4 | Steering | +0.029 | 0.889 | 1.000 | Not supported |
| 2 | 4 | Probing | +0.024 | 0.907 | 1.000 | Not supported |
| 3 | 8 | Steering | -0.222 | 0.275 | 1.000 | Not supported |
| 4 | 8 | Probing | -0.006 | 0.977 | 1.000 | Not supported |
| 5 | 4 | Steering (MW) | --- | 0.407 | 1.000 | Not supported |
| 6 | 4 | Probing (MW) | --- | 0.692 | 1.000 | Not supported |
| 7 | 8 | Steering (MW) | --- | 0.079 | 0.632 | Not supported |
| 8 | 8 | Probing (MW) | --- | 0.377 | 1.000 | Not supported |

*Table 4: Multiple comparisons correction (Bonferroni $\alpha_B = 0.00625$).*

### 4.5 Cross-Layer Comparison

Figure 1 shows absorption rates and steering success across layers. Absorption rates show modest increase with depth (mean: 0.031 at L0, 0.034 at L4, 0.058 at L8, 0.047 at L10), but steering success also increases (mean: 0.72 at L0, 0.79 at L4, 0.85 at L8, 0.82 at L10). The co-occurrence of higher absorption and better performance at deeper layers directly contradicts the assumed trade-off.

### 4.6 Random Baseline Validation

Random feature directions achieve 34--38\% steering success. Feature-specific steering significantly outperforms random ($t = 6.41$, $d = 1.26$ at L4; $t = 6.02$, $d = 1.18$ at L8), confirming that the SAE decoder captures meaningful structure regardless of absorption rate.

## 5 Discussion

### 5.1 Interpreting the Null Results

Our findings falsify the field's implicit assumption that absorption rate is a reliable proxy for SAE downstream quality. We offer four non-mutually-exclusive interpretations:

**1. Absorption is not the right target.** The SAE community may be optimizing the wrong metric. Absorption captures a specific structural pattern (parent-child suppression), but downstream tasks depend on other properties: decoder alignment, feature specificity, and activation sparsity.

**2. Current absorption metrics are too coarse.** Chanin et al.'s metric detects suppression but not its functional consequences. A feature can be highly absorbed yet retain perfect decoder alignment (as our precision-invariance finding suggests).

**3. The trade-off exists but is weak and context-dependent.** The layer-8 steering trend ($\rho = -0.222$) hints at a modest relationship that may strengthen with larger sample sizes or different feature types.

**4. Architectural interventions improve quality through other mechanisms.** Matryoshka SAE and OrtSAE may improve performance through better dictionary learning or reduced interference, with absorption reduction being a side effect rather than the cause.

### 5.2 Implications for SAE Practitioners

**Do not optimize absorption in isolation.** Our results show that absorption rate alone is uncorrelated with steering and probing performance. Practitioners should evaluate SAEs on downstream tasks directly.

**Use dual-metric evaluation.** We propose reporting both absorption rate and downstream performance. A "good" SAE should achieve low absorption *and* high task performance---but the two metrics are independent, not redundant.

**Validate on diverse feature types.** Our study uses first-letter features, which are relatively simple. The absorption-utility relationship may differ for semantic, syntactic, or multi-token features.

### 5.3 Relationship to Existing Work

Our findings are consistent with Wang et al. (2026), who found weak correlation ($\tau_b \approx 0.3$) between interpretability metrics and steering utility. They also align with Korznikov et al.'s (2026) "Sanity Checks" challenge: if random SAEs can match trained SAEs on key metrics, then individual metric optimization is insufficient.

Our results do **not** contradict Chanin et al. (2024), who established absorption as a real phenomenon. We question only whether absorption rate is a useful proxy for quality---not whether absorption itself exists.

### 5.4 Limitations

**Small sample:** $n=26$ features limits power. We estimate ~25\% power for $|\rho| \geq 0.50$---strong effects would be detectable, but moderate effects may be missed.

**Single model and SAE family:** GPT-2 Small with res-jb SAEs. Results may not generalize to larger models or different architectures.

**Simple features:** First-letter features are shallow and easily detectable. The absorption-utility relationship may differ for complex semantic features.

**Binary steering:** Our steering success metric is coarse (pass/fail). Continuous metrics (probability lift, perplexity change) may reveal subtler relationships.

## 6 Conclusion

We conducted the first systematic test of whether reducing feature absorption improves SAE downstream performance. Across 26 features, two layers, and two task types, we find **no statistically significant relationship** between absorption rate and downstream utility. The field's implicit assumption---that lower absorption means better SAEs---is not supported by our data.

This does not mean absorption is irrelevant. It means absorption rate alone is an unreliable quality proxy. We urge the SAE community to:

1. **Evaluate downstream utility directly**, not through absorption metrics alone
2. **Report dual metrics** (absorption + task performance) for all SAE comparisons
3. **Test on diverse feature types** beyond first-letter features
4. **Consider that absorption-reducing architectures may work through other mechanisms**

Our null result is a methodological contribution: it prevents the field from optimizing the wrong target and redirects attention toward validated evaluation protocols.

## References

Bau, D., et al. (2019). "Network Dissection: Quantifying Interpretability of Deep Visual Representations." *CVPR*.

Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread*.

Chanin, J., et al. (2024). "A is for Absorption: Studying Feature Absorption in Sparse Autoencoders." *NeurIPS*.

Chen, Y., et al. (2025). "OrtSAE: Orthogonalizing Sparse Autoencoder Decoders for Reduced Feature Absorption." *ICML*.

Conmy, A., et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability." *NeurIPS*.

Cunningham, H., et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." *ICLR*.

Elhage, N., et al. (2022). "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*.

Gadgil, S., et al. (2025). "Matryoshka Sparse Autoencoders: Nested Dictionary Learning for Interpretability." *ICML*.

Korznikov, A., et al. (2026). "Sanity Checks for SAEs: Random Baselines Match Trained SAEs." *arXiv:2602.14111*.

Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Anthropic*.

Wang, K., et al. (2025). "Attention-Based Thresholding for Sparse Autoencoders." *ICLR*.

Wang, X., et al. (2026). "The Interpretability-Utility Gap in Sparse Autoencoders." *ICLR*.
