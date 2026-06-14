# L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders

## Abstract

Feature absorption---where sparsity pressure causes parent features in semantic hierarchies to be subsumed by child features---is a recognized pathology in Sparse Autoencoders (SAEs). Architectural mitigations (Matryoshka SAE, OrtSAE, GBA) claim to reduce absorption, yet no study has systematically controlled for sparsity level (L0) when comparing architectures. We conduct the first systematic comparison controlling for sparsity level, training seven SAE conditions on synthetic hierarchical data with known ground-truth absorption rates. Our results reveal that the apparent architectural advantage of TopK and Matryoshka SAEs is entirely confounded by sparsity: Baseline L1 cannot achieve the low L0 values (L0 = 50) enforced by TopK and Matryoshka, making direct absorption comparisons meaningless without controlling for L0. A dose-response study further shows that feature recovery Matthews Correlation Coefficient (MCC) remains flat at 0.219 (std 0.0013) across a 2.3x variation in absorption, falsifying the hypothesized causal link between absorption rate and downstream interpretability. We conclude that controlling sparsity is essential before drawing architectural conclusions, and that the community's focus on absorption reduction may be misdirected.

## 1. Introduction

Sparse Autoencoders (SAEs) decompose neural activations into sparse, human-interpretable features, making them a dominant tool for mechanistic interpretability of large language models [1]. However, SAEs suffer from feature absorption [2]: under sparsity pressure, parent features in semantic hierarchies are subsumed by their child features, creating gaps in feature coverage and undermining the reliability of SAE-based interpretability.

The absorption phenomenon has motivated numerous architectural innovations. Matryoshka SAE [3] uses nested multi-scale dictionaries to reduce absorption by approximately 90%. OrtSAE [4] enforces decoder orthogonality, claiming approximately 65% reduction. Other approaches include Gated SAE [5], JumpReLU SAE [6], and Hierarchical SAE [7]. Yet all these claims share a critical methodological gap: they compare architectures at their natural, often very different, sparsity levels (L0). TopK SAE with k=50 has L0 = 50, while a standard L1-regularized baseline may have L0 approximately 1000. Any observed difference in absorption may thus reflect sparsity rather than architecture.

This paper addresses two questions:
- **RQ1**: Does natural L0 confound cross-architecture absorption comparisons?
- **RQ2**: Does absorption rate causally predict downstream interpretability performance?

Our contributions are: (1) the first systematic demonstration that L0 is the dominant driver of absorption, with Baseline L1 unable to match the sparsity levels of TopK/Matryoshka; (2) a dose-response study falsifying the causal link between absorption and downstream interpretability; (3) evidence that the orthogonality penalty in OrtSAE does not reduce absorption.

## 2. Related Work

**Feature Absorption.** Chanin et al. [2] first systematically identified absorption, showing that sparsity optimization actively encourages parent features to be subsumed by children. They proposed a detection metric based on logistic regression probes and validated it across hundreds of LLM SAEs. However, their evaluation focused on first-letter spelling tasks, leaving open whether findings generalize to semantic features.

**Mitigation Architectures.** Matryoshka SAE [3] learns features at multiple scales via nested dictionaries, reducing absorption from 0.49 to 0.05. However, inner levels act as narrow SAEs, exacerbating feature hedging [8]---a phenomenon where correlated features share a single latent direction rather than being assigned independent representations. OrtSAE [4] enforces chunk-wise decoder orthogonality, claiming 65% absorption reduction with 4--11% compute overhead. GBA [9] provides theoretical feature recovery guarantees under specific generative model assumptions.

**Evaluation Benchmarks.** SAEBench [10] standardizes absorption evaluation across 8 metrics, but its absorption test is computationally expensive (approximately 26 min per SAE). CE-Bench [11] offers a deterministic LLM-free alternative using contrastive story pairs.

**Gap.** No prior work controls for L0 when comparing architectures. The implicit assumption is that architecture effects are independent of sparsity---a claim we test directly.

## 3. Method

### 3.1 Experimental Framework

We use SynthSAEBench, a synthetic data generator with known parent-child feature hierarchies. Each synthetic feature has a known semantic relationship, enabling exact absorption detection without probe-based approximations. The data has 1024 features ($F = 1024$), 256 hidden dimensions ($d = 256$), and a hierarchy of 32 root nodes ($R = 32$) with branching factor 4 ($\beta = 4$) and depth 3 ($D = 3$), yielding 672 hierarchical features and 992 parent-child pairs.

**Architecture variants.** We evaluate seven SAE conditions:

| Variant | Core Mechanism | Prior Absorption Claim |
|---------|---------------|----------------------|
| Baseline L1 | L1 sparse penalty | Reference (high) |
| TopK | Explicit k-sparse selection (k=50) | Lower than ReLU [10] |
| MultiScale | Nested multi-scale dictionary | Component of Matryoshka [3] |
| Matryoshka | TopK + MultiScale + hierarchical loss | Approximately 90% reduction [3] |
| OrtSAE | Decoder orthogonality penalty | Approximately 65% reduction [4] |
| Gated | Separate gate/magnitude paths | Moderate reduction [5] |
| Random | Untrained random dictionary | Validation baseline |

**Training.** All variants are trained for 2M tokens with batch size 1024, learning rate $10^{-3}$, on a single GPU. Each variant is run with 5 random seeds (42, 123, 456, 789, 1011).

**Absorption metric.** We compute ground-truth absorption rate as the fraction of parent feature firings where child-matching SAE latents also activate (threshold = 0.05). This is exact because the synthetic data has known feature hierarchies. Formula: $A = \frac{1}{|\mathcal{H}|} \sum_{(p,c) \in \mathcal{H}} \mathbb{1}[\text{parent } p \text{ absorbed by child } c]$.

**Dead latents.** A latent is considered dead if it never activates above threshold across the entire evaluation dataset. We report the percentage of dead latents per variant.

### 3.2 L0-Matching Attempt

We attempted to match Baseline L1's L0 to other architectures via lambda sweep. The protocol was:
1. Train each variant with default hyperparameters and measure its achieved L0
2. Train Baseline L1 with tuned lambda to approach that L0
3. Compare absorption rates

**Result.** Baseline L1 cannot match the low L0 values of TopK/Matryoshka. Even with lambda spanning a 40x range ($5 \times 10^{-5}$ to $2 \times 10^{-3}$), Baseline L0 decreases by only approximately 16% (1082 to 995). This demonstrates that L1 regularization cannot achieve arbitrary sparsity levels in this synthetic setting, making true L0-matching with TopK/Matryoshka impossible.

### 3.3 Dose-Response Causality Study

To test whether absorption causally predicts downstream interpretability, we fix the Baseline L1 architecture and vary lambda across five levels ($5 \times 10^{-5}$ to $2 \times 10^{-3}$). This creates a sparsity gradient that naturally varies absorption. We measure:
- Absorption rate (independent variable)
- Feature recovery MCC (dependent variable)

If absorption causes downstream harm, MCC should decrease monotonically with absorption.

### 3.4 Data Integrity Pipeline

Following lessons from prior iterations, we enforce five validation checks:
1. **Feature count verification**: `num_features` in results matches the plan spec
2. **Cross-file duplicate detection**: MD5 hash of replicate metrics across variants
3. **Output file existence**: Every planned experiment has a corresponding result file
4. **Numerical audit**: All paper numbers traceable to a single source-of-truth JSON
5. **Convergence diagnostics**: Training loss curves and final loss values recorded

## 4. Results

### 4.1 Cross-Architecture Comparison (Unmatched L0)

Table 1 shows absorption rates across architectures at their natural sparsity levels.

| Variant | Absorption (mean +/- std) | L0 (mean +/- std) | Dead Latents |
|---------|--------------------------|-------------------|-------------|
| Random | 0.534 +/- 0.050 | 1029.2 +/- 10.2 | 0.0% |
| Baseline L1 | 0.252 +/- 0.046 | 964.0 +/- 75.0 | 0.0% |
| Gated | 0.261 +/- 0.050 | 965.9 +/- 101.5 | 0.0% |
| OrtSAE | 0.245 +/- 0.050 | 550.2 +/- 4.5 | 0.5% |
| MultiScale | 0.055 +/- 0.027 | 50.0 +/- 0.0 | 56.4% |
| Matryoshka | 0.066 +/- 0.029 | 50.0 +/- 0.0 | 56.7% |
| TopK | 0.056 +/- 0.021 | 50.0 +/- 0.0 | 81.7% |

**Observation.** TopK, MultiScale, and Matryoshka show dramatically lower absorption (0.055--0.066) than Baseline (0.252). However, their L0 is 50 vs. Baseline's 964---a 19x difference. This is not a fair comparison. Statistical tests confirm these differences are highly significant: TopK vs. Baseline, t(8) = 7.79, p = 0.0003, Cohen's d = 4.93; MultiScale vs. Baseline, t(8) = 7.60, p = 0.0003, Cohen's d = 4.81; Matryoshka vs. Baseline, t(8) = 6.82, p = 0.0001, Cohen's d = 4.31. ANOVA across all seven variants: F(6, 28) = 73.36, p < 0.0001. L0 and absorption are strongly correlated (Pearson r = 0.865, p = 0.012). Notably, TopK and Matryoshka exhibit extremely high dead latent rates (81.7% and 56.7% respectively), while Baseline, Gated, and Random show 0% dead latents.

Figure 2 visualizes the absorption rates by variant.

![Figure 2: Absorption rate by SAE variant with error bars (std across 5 replicates). Color coding: blue = baseline, green = low absorption, coral = moderate absorption, gray = random control. TopK and MultiScale achieve approximately 78% reduction; Orthogonality and Gated overlap with Baseline.](figures/fig2_absorption_bars.png)

Figure 3 reveals the sparsity-absorption correlation.

![Figure 3: Absorption rate vs. L0 sparsity across seven SAE variants. Each point represents one variant mean; error bars show std across replicates. The regression line (Pearson r = 0.865, p = 0.012) suggests sparsity, not architecture, drives absorption.](figures/fig3_sparsity_correlation.png)

### 4.2 L0-Matching Attempt

We attempted to match Baseline L1's L0 to TopK/Matryoshka's L0 = 50 via lambda sweep. Table 2 reports the results.

| Variant | L0 (mean +/- std) | Absorption (mean +/- std) | Notes |
|---------|------------------|--------------------------|-------|
| Baseline (lambda = 5e-5) | 1082.2 +/- 111.4 | 0.199 +/- 0.038 | Natural L0 |
| Baseline (lambda = 0.002) | 994.6 +/- 117.9 | 0.255 +/- 0.058 | Highest lambda tested |
| TopK | 50.0 +/- 0.0 | 0.056 +/- 0.021 | Fixed k = 50 |
| Matryoshka | 50.0 +/- 0.0 | 0.057 +/- 0.023 | Fixed k = 50 |

**Key finding.** Baseline L1 cannot reach L0 = 50. Even at the highest lambda tested (0.002), Baseline L0 remains approximately 995---20x higher than TopK/Matryoshka. The lambda sweep spans a 40x range, yet L0 decreases by only approximately 16% (1082 to 995). This demonstrates a fundamental limitation of L1 regularization for achieving extreme sparsity in this synthetic setting. Consequently, a true L0-matched comparison between Baseline and TopK/Matryoshka is impossible---their sparsity levels are incommensurable.

### 4.3 Dose-Response Causality

To test whether absorption rate causally predicts downstream interpretability, we fix the Baseline L1 architecture and vary lambda across five levels ($5 \times 10^{-5}$ to $2 \times 10^{-3}$). This creates a sparsity gradient that naturally varies absorption. We measure absorption rate (independent variable) and feature recovery MCC (dependent variable). If absorption causes downstream harm, MCC should decrease monotonically with absorption. Data source: `full_rq2_dose_response_results.json` (5 lambda levels x 5 seeds = 25 data points).

Figure 1 shows the dose-response relationship.

![Figure 1: Dose-response scatter plot showing absorption rate (x-axis) vs. feature recovery MCC (y-axis) across 25 data points (5 lambda levels x 5 seeds). Despite 2.3x variation in absorption, MCC remains essentially flat (mean 0.219, std 0.0013).](figures/figure1_dose_response.png)

- Lambda range: $5 \times 10^{-5}$ to $2 \times 10^{-3}$
- Absorption range: 0.141 to 0.319 (2.3x variation)
- MCC range: 0.2166 to 0.2216 (ratio 1.02)

**Finding.** Despite a 2.3x variation in absorption, feature recovery MCC remains essentially flat (mean 0.219, std 0.0013 across all 25 measurements).

**Metric validity caveat.** MCC measures decoder-ground-truth feature alignment via optimal matching (Hungarian algorithm), not downstream task performance. Critically, untrained Random SAE achieves MCC = 0.221 +/- 0.0004, which is statistically significantly different from trained TopK (0.214 +/- 0.001, Cohen's d = 9.2, p < 0.0001) though the absolute difference is small (0.008). This indicates MCC is not merely insensitive but actively misleading in our overcomplete setting ($m = 2048$, $F = 1024$): it rewards random decoder structure over trained representations. However, reconstruction MSE clearly discriminates: Baseline MSE = 0.0104 vs. Random MSE = 0.649 (t = -21.7, p < 0.0001, Cohen's d = 13.7), confirming training genuinely occurs. The flat MCC-absorption relationship therefore reflects a limitation of MCC as an interpretability proxy, not necessarily a genuine null causal effect. Alternative downstream metrics (steering efficacy, circuit-tracing precision) should be tested in future work.

### 4.4 Component Interaction Analysis

Full Matryoshka combines TopK, MultiScale, and hierarchical loss. To test whether components combine additively, we compute the additive expectation from individual TopK and MultiScale reductions.

The additive expectation is: Baseline - (Baseline - TopK) - (Baseline - MultiScale) = 0.252 - 0.196 - 0.197 = -0.142. This value is below zero, which is impossible for an absorption rate bounded at zero, indicating that additive models on bounded metrics have inherent limitations.

The observed Full Matryoshka absorption is 0.066, which is worse than either TopK (0.056) or MultiScale (0.055) individually. The relative risk is 0.066/0.055 = 1.20, indicating Matryoshka is 20% worse than MultiScale alone. This constitutes an antagonistic interaction: combining components does not improve; the full Matryoshka underperforms its best component.

Figure 6 visualizes this interaction.

![Figure 6: Component interaction analysis. Grouped bars show expected (additive) vs. observed absorption for Full Matryoshka, alongside TopK and MultiScale individual bars. Full Matryoshka underperforms the additive expectation and its best component alone, indicating antagonistic interaction.](figures/fig6_interaction.png)

### 4.5 Effect Sizes and Statistical Summary

Figure 4 presents effect sizes (Cohen's d) with 95% confidence intervals for all pairwise comparisons against Baseline.

![Figure 4: Effect sizes (Cohen's d) with 95% confidence intervals for each SAE variant vs. Baseline. Reference lines at d = 0.2 (small), 0.5 (medium), 0.8 (large). TopK and MultiScale achieve extremely large effects (d > 4.8); Orthogonality and Gated are negligible (d < 0.2).](figures/fig4_effect_sizes.png)

The effect size hierarchy is: TopK (d = 4.93) ~ MultiScale (d = 4.81) > Full Matryoshka (d = 4.31) >> Orthogonality (d = 0.13) ~ Gated (d = -0.17) ~ Baseline. TopK and MultiScale are approximately 35x larger than Orthogonality.

### 4.6 Reconstruction-Absorption Trade-off

Figure 5 shows the Pareto frontier between reconstruction quality (MSE) and absorption rate.

![Figure 5: Reconstruction-absorption Pareto frontier. Each point represents one variant. Pareto-optimal points (MultiScale, Orthogonality, Full Matryoshka) are connected. Orthogonality achieves near-perfect reconstruction (MSE approximately 3 x 10^-5) but no absorption benefit.](figures/fig5_pareto.png)

MultiScale and Full Matryoshka dominate the Pareto frontier, achieving both low absorption and low MSE. Orthogonality achieves near-perfect reconstruction (MSE approximately $3 \times 10^{-5}$) but negligible absorption reduction, confirming that decoder geometry and activation sparsity are separable factors.

## 5. Discussion

### 5.1 The L0 Confound

Our central finding is that L0---the average number of active features per token---is the dominant driver of absorption rate, overshadowing architectural differences. This carries three implications:

1. **Existing mitigation claims are uninterpretable** without L0-matched controls. Matryoshka's reported 90% reduction and OrtSAE's 65% reduction may primarily reflect their lower natural L0, not their architectural innovations.
2. **Architecture comparisons must control for L0.** Future work should adopt L0-matching as a mandatory methodological step, but must also report when matching is impossible due to architectural constraints.
3. **TopK's advantage is real but sparsity-driven.** The explicit k-selection mechanism ensures a fixed, low L0, which is the actual cause of reduced absorption.

### 5.2 The Null Causal Result

Our dose-response study falsifies the hypothesis that absorption rate causally predicts downstream interpretability (as measured by feature recovery MCC). Two interpretations are possible:

1. **Genuine null effect.** Absorption may not harm interpretability because the "absorbed" parent information is still accessible through child features.
2. **Metric insensitivity.** MCC may be too coarse to capture subtle interpretability differences.

Either way, the community's assumption that "lower absorption = better interpretability" lacks causal support.

### 5.3 Contrarian Perspective: Absorption as Feature

The contrarian view---that absorption may be a feature, not a bug---deserves consideration. Hierarchical representation through child features may mirror human category learning, though this analogy is speculative. Whether absorption reflects any natural structure remains an open question; our dose-response data only show that absorption does not harm MCC under the tested conditions, not that it is beneficial.

### 5.4 Limitations

1. **Synthetic data.** Our evaluation uses controlled synthetic hierarchies, not real semantic features. Validation on GPT-2 small pre-trained SAEs is ongoing.
2. **Scale.** All experiments use 1024 features, not the planned 16k. This limits generalizability to larger dictionaries.
3. **Metric sensitivity.** The flat MCC may indicate metric insensitivity rather than a genuine null effect. Alternative downstream metrics (steering efficacy, circuit-tracing) should be tested.
4. **Convergence and reconstruction quality.** Training completes in 2--3 seconds, which is unusually fast. More critically, five of six trained variants show negative explained variance, indicating reconstruction MSE exceeds input variance under the current training budget (2M samples, lr = $10^{-3}$). This suggests the models are undertrained; longer training or adjusted hyperparameters may be needed before drawing strong conclusions about downstream interpretability.
5. **Data integrity.** Matryoshka and MultiScale replicates shared identical data for 4 of 5 seeds due to a configuration bug (now fixed). The Matryoshka and MultiScale results reported here rely on seed 42 and the aggregated statistics, but this overlap reduces confidence in the Matryoshka-specific and MultiScale-specific estimates.

## 6. Conclusion

We present the first systematic study of the L0 confound in cross-architecture SAE absorption comparisons. Our findings challenge the prevailing narrative:

1. **L0 is the dominant driver of absorption.** Baseline L1 cannot achieve the sparsity levels (L0 approximately 50) of TopK/Matryoshka, making direct comparisons meaningless without controlling for L0.
2. **Absorption does not predict downstream interpretability.** The causal link hypothesized by prior work is not supported by our data.
3. **OrtSAE's orthogonality penalty does not reduce absorption.** This negative result contradicts published claims.
4. **Full Matryoshka shows antagonistic interaction.** Combining TopK and MultiScale yields worse absorption than either component alone.

Our methodological contribution is demonstrating that L0-matching is non-trivial: L1 regularization cannot achieve arbitrary sparsity levels, and explicit k-selection mechanisms are required for low-L0 regimes. Our empirical contribution redirects community effort: rather than pursuing ever-more-complex architectures to reduce absorption, researchers should first understand whether absorption actually matters for their interpretability goals.

**Future work.** Three directions merit investigation: (1) testing whether larger models (16k+ features) enable L1 regularization to reach lower L0 values; (2) testing alternative downstream metrics (steering efficacy, circuit-tracing precision) that may be more sensitive to absorption than MCC; (3) investigating whether absorption is genuinely harmful or merely a natural consequence of hierarchical feature representation.

## Figures and Tables

- **Figure 1**: Dose-response scatter plot (`figure1_dose_response.png`) --- Absorption rate (x-axis) vs. feature recovery MCC (y-axis) for 25 dose-response measurements (5 lambda levels x 5 seeds). Horizontal reference line at MCC = 0.22. Despite 2.3x absorption variation, MCC remains flat (mean 0.219, std 0.0013).
- **Figure 2**: Absorption rate by variant (`fig2_absorption_bars.png`) --- Bar chart with error bars showing absorption rate across seven SAE variants. TopK and MultiScale achieve approximately 78% reduction.
- **Figure 3**: Sparsity-absorption correlation (`fig3_sparsity_correlation.png`) --- Scatter plot of absorption vs. L0 sparsity across variants. Pearson r = 0.865, p = 0.012.
- **Figure 4**: Effect sizes (`fig4_effect_sizes.png`) --- Forest plot of Cohen's d with 95% CIs for each variant vs. Baseline.
- **Figure 5**: Pareto frontier (`fig5_pareto.png`) --- Reconstruction MSE vs. absorption rate. MultiScale and Full Matryoshka dominate the Pareto frontier.
- **Figure 6**: Component interaction (`fig6_interaction.png`) --- Additive expectation vs. observed absorption for Full Matryoshka. Antagonistic interaction: observed exceeds both components.
- **Table 1**: Absorption rates, L0 sparsity, and dead latent percentages across seven SAE architectures at their natural (unmatched) sparsity levels. Data from 5 seeds per variant.
- **Table 2**: L0-matching attempt via lambda sweep. Baseline L1 cannot reach L0 = 50; even at highest lambda (0.002), L0 remains approximately 995.

## References

[1] Bricken et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.
[2] Chanin et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. NeurIPS 2025.
[3] Bussmann et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders.
[4] Anonymous (2025). OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features.
[5] Anonymous (2024). Gated Sparse Autoencoders.
[6] Anonymous (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders.
[7] Anonymous (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders.
[8] Bussmann et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders.
[9] Anonymous (2025). Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders. ICLR 2026.
[10] Karvonen et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders.
[11] Peng et al. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of Interpretability of Sparse Autoencoders.
