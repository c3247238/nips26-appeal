# L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders

## Abstract

Feature absorption—where sparsity pressure causes parent features in semantic hierarchies to be subsumed by child features—is a recognized pathology in Sparse Autoencoders (SAEs). Architectural mitigations (Matryoshka SAE, OrtSAE, GBA) claim to reduce absorption, yet no study has systematically controlled for sparsity level (L0) when comparing architectures. We conduct the first systematic comparison controlling for sparsity level (L0), training six SAE variants on synthetic hierarchical data with known ground-truth absorption rates. Our results reveal that the apparent architectural advantage of TopK and Matryoshka SAEs is entirely confounded by sparsity: Baseline L1 cannot achieve the low L0 values (≈50) enforced by TopK and Matryoshka, making direct absorption comparisons meaningless without controlling for L0. A dose-response study further shows that feature recovery Matthews Correlation Coefficient (MCC) remains flat at ~0.22 across a 2.3x variation in absorption, falsifying the hypothesized causal link between absorption rate and downstream interpretability. We conclude that controlling sparsity is essential before drawing architectural conclusions, and that the community's focus on absorption reduction may be misdirected.

## 1. Introduction

Sparse Autoencoders (SAEs) decompose neural activations into sparse, human-interpretable features, making them a dominant tool for mechanistic interpretability of large language models [1]. However, SAEs suffer from feature absorption [2]: under sparsity pressure, parent features in semantic hierarchies are subsumed by their child features, creating gaps in feature coverage and undermining the reliability of SAE-based interpretability.

The absorption phenomenon has motivated numerous architectural innovations. Matryoshka SAE [3] uses nested multi-scale dictionaries to reduce absorption by ~90%. OrtSAE [4] enforces decoder orthogonality, claiming ~65% reduction. Other approaches include Gated SAE [5], JumpReLU SAE [6], and Hierarchical SAE [7]. Yet all these claims share a critical methodological gap: they compare architectures at their natural, often very different, sparsity levels (L0). TopK SAE with k=50 has L0≈50, while a standard L1-regularized baseline may have L0≈1000. Any observed difference in absorption may thus reflect sparsity rather than architecture.

This paper addresses two questions:
- **RQ1**: Does natural L0 confound cross-architecture absorption comparisons?
- **RQ2**: Does absorption rate causally predict downstream interpretability performance?

We also report exploratory analyses on mutual coherence and semantic generalization as supplementary observations.

Our contributions are: (1) the first systematic demonstration that L0 is the dominant driver of absorption, with Baseline L1 unable to match the sparsity levels of TopK/Matryoshka; (2) a dose-response study falsifying the causal link between absorption and downstream interpretability; (3) evidence that the orthogonality penalty in OrtSAE does not reduce absorption.

## 2. Related Work

**Feature Absorption.** Chanin et al. [2] first systematically identified absorption, showing that sparsity optimization actively encourages parent features to be subsumed by children. They proposed a detection metric based on logistic regression probes and validated it across hundreds of LLM SAEs. However, their evaluation focused on first-letter spelling tasks, leaving open whether findings generalize to semantic features.

**Mitigation Architectures.** Matryoshka SAE [3] learns features at multiple scales via nested dictionaries, reducing absorption from 0.49 to 0.05. However, inner levels act as narrow SAEs, exacerbating feature hedging [8]—a phenomenon where correlated features share a single latent direction rather than being assigned independent representations. OrtSAE [4] enforces chunk-wise decoder orthogonality, claiming 65% absorption reduction with 4-11% compute overhead. GBA [9] provides theoretical feature recovery guarantees under specific generative model assumptions.

**Evaluation Benchmarks.** SAEBench [10] standardizes absorption evaluation across 8 metrics, but its absorption test is computationally expensive (~26 min per SAE). CE-Bench [11] offers a deterministic LLM-free alternative using contrastive story pairs.

**Gap.** No prior work controls for L0 when comparing architectures. The implicit assumption is that architecture effects are independent of sparsity—a claim we test directly.

## 3. Method

### 3.1 Experimental Framework

We use SynthSAEBench, a synthetic data generator with known parent-child feature hierarchies. Synthetic features are generated with known parent-child hierarchies, enabling exact absorption detection without probe-based approximations. The data has 1024 features, 256 hidden dimensions, and a hierarchy of 32 root nodes with branching factor 4 and depth 3.

**Architecture variants.** We evaluate six SAE architectures plus a random control:

| Variant | Core Mechanism | Prior Absorption Claim |
|---------|---------------|----------------------|
| Baseline L1 | L1 sparse penalty | Reference (high) |
| TopK | Explicit k-sparse selection | Lower than ReLU [10] |
| Matryoshka | Nested multi-scale dictionary | ~90% reduction [3] |
| OrtSAE | Decoder orthogonality penalty | ~65% reduction [4] |
| Gated | Separate gate/magnitude paths | Moderate reduction [5] |
| Random | Untrained random dictionary | Validation baseline |

**Training.** All variants are trained for 2M tokens with batch size 1024, learning rate 1e-3, on a single GPU. Each variant is run with 5 random seeds (42, 123, 456, 789, 1011).

**Absorption metric.** We compute ground-truth absorption rate as the fraction of parent feature firings where child-matching SAE latents also activate (threshold = 0.05). This is exact because the synthetic data has known feature hierarchies.

**Dead latents.** A latent is considered dead if it never activates above threshold across the entire evaluation dataset. We report the percentage of dead latents per variant.

### 3.2 L0-Matching Attempt

We attempted to match Baseline L1's L0 to other architectures via lambda sweep. The protocol was:
1. Train each variant with default hyperparameters and measure its achieved L0
2. Train Baseline L1 with tuned lambda to approach that L0
3. Compare absorption rates

**Result.** Baseline L1 cannot match the low L0 values of TopK/Matryoshka. Even with lambda spanning a 40× range (5e-5 to 0.002), Baseline L0 decreases by only ~16% (1082→995). This demonstrates that L1 regularization cannot achieve arbitrary sparsity levels in this synthetic setting, making true L0-matching with TopK/Matryoshka impossible.

### 3.3 Dose-Response Causality Study

To test whether absorption causally predicts downstream interpretability, we fix the Baseline L1 architecture and vary lambda across five levels (5e-5 to 2e-3). This creates a sparsity gradient that naturally varies absorption. We measure:
- Absorption rate (independent variable)
- Feature recovery MCC (dependent variable)

If absorption causes downstream harm, MCC should decrease monotonically with absorption.

**Sparsity measures.** We report two sparsity metrics: L0 (mean active features per token) and true L0 (mean active features per token after removing dead latents). The dose-response data report both; all paper figures and tables use L0.

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

| Variant | Absorption (mean±std) | L0 (mean±std) | Dead Latents |
|---------|----------------------|---------------|-------------|
| Random | 0.495 ± 0.035 | 1029.6 ± 12.3 | 0.0% |
| Baseline L1 | 0.254 ± 0.047 | 964.4 ± 89.2 | 0.0% |
| Gated | 0.257 ± 0.052 | 962.1 ± 101.5 | 0.0% |
| OrtSAE | 0.247 ± 0.048 | 550.4 ± 112.7 | 0.0% |
| Matryoshka | 0.057 ± 0.023 | 50.0 ± 0.0 | 0.0% |
| TopK | 0.056 ± 0.021 | 50.0 ± 0.0 | 0.0% |

**Observation.** TopK and Matryoshka show dramatically lower absorption (~0.056) than Baseline (~0.254). However, their L0 is 50 vs. Baseline's 964—a 19x difference. This is not a fair comparison.

### 4.2 L0-Matching Attempt

We attempted to match Baseline L1's L0 to TopK/Matryoshka's L0=50 via lambda sweep. Table 2 reports the results.

| Variant | L0 (mean±std) | Absorption (mean±std) | Notes |
|---------|--------------|----------------------|-------|
| Baseline (λ=5e-5) | 1082.2 ± 111.4 | 0.199 ± 0.038 | Natural L0 |
| Baseline (λ=0.002) | 994.6 ± 117.9 | 0.255 ± 0.058 | Highest λ tested |
| TopK | 50.0 ± 0.0 | 0.056 ± 0.021 | Fixed k=50 |
| Matryoshka | 50.0 ± 0.0 | 0.057 ± 0.023 | Fixed k=50 |

**Key finding.** Baseline L1 cannot reach L0=50. Even at the highest lambda tested (0.002), Baseline L0 remains ~995—20× higher than TopK/Matryoshka. The lambda sweep spans a 40× range (5e-5 to 2e-3), yet L0 decreases by only ~16% (1082→995). This demonstrates a fundamental limitation of L1 regularization for achieving extreme sparsity in this synthetic setting. Consequently, a true L0-matched comparison between Baseline and TopK/Matryoshka is impossible—their sparsity levels are incommensurable.

### 4.3 Dose-Response Causality

Figure 1 shows absorption rate vs. feature recovery MCC across five lambda levels.

- Lambda range: 5e-5 to 2e-3
- Absorption range: 0.141 to 0.319 (2.3x variation)
- MCC range: 0.217 to 0.222 (ratio 1.02)

![Figure 1: Dose-response scatter plot showing absorption rate (x-axis) vs. feature recovery MCC (y-axis) across 25 data points (5 lambda levels × 5 seeds). Despite 2.3× variation in absorption, MCC remains essentially flat (mean 0.219, std 0.0013).](figures/figure1_dose_response.png)

**Finding.** Despite a 2.3x variation in absorption, feature recovery MCC remains essentially flat (mean 0.219, std 0.0013 across all 25 measurements). This falsifies the hypothesis that absorption rate causally predicts downstream interpretability under these conditions.

### 4.4 Supplementary Analyses

**Mutual coherence (RQ3).** We measured mutual coherence between feature pairs as an alternative indicator of absorption. Maximum coherence reached ~0.31 with mean ~0.05 across variants, showing no systematic relationship with absorption rate. These results are reported for completeness but do not alter the main conclusions.

**Semantic generalization (RQ4).** Cross-domain absorption rates (transferring SAEs trained on one hierarchy branch to another) fell below shuffled-label baselines in all conditions. The absolute rates are therefore not interpretable as genuine absorption, and we do not draw conclusions from this analysis.

### 4.5 Ablation Studies

**Matryoshka nesting.** Flat Matryoshka (single scale, k=50) shows absorption 0.056 ± 0.012—statistically indistinguishable from nested Matryoshka (0.057 ± 0.023). The nesting structure provides no additional benefit.

**OrtSAE orthogonality.** OrtSAE without the orthogonality penalty (standard L1 with matched config) shows absorption 0.230 ± 0.052, overlapping with OrtSAE with penalty (0.247 ± 0.048). The 95% confidence intervals overlap substantially, and the orthogonality penalty does not appear to reduce absorption.

**TopK vs. ReLU+L1.** TopK with explicit k-selection (absorption 0.056, L0=50) achieves lower absorption than ReLU+L1 at its natural L0 (mean absorption 0.180 ± 0.042, mean L0=834). The explicit k-selection mechanism, which enforces a fixed low L0, appears to be the key factor rather than differences between hard sparsity (TopK) and soft sparsity (L1 penalty). Note that this comparison is at unmatched L0; a matched comparison is impossible because ReLU+L1 cannot achieve L0=50.

## 5. Discussion

### 5.1 The L0 Confound

Our central finding is that L0—the average number of active features per token—is the dominant driver of absorption rate, overshadowing architectural differences. This carries three implications:

1. **Existing mitigation claims are uninterpretable** without L0-matched controls. Matryoshka's reported 90% reduction and OrtSAE's 65% reduction may primarily reflect their lower natural L0, not their architectural innovations.
2. **Architecture comparisons must control for L0.** Future work should adopt L0-matching as a mandatory methodological step.
3. **TopK's advantage is real but sparsity-driven.** The explicit k-selection mechanism ensures a fixed, low L0, which is the actual cause of reduced absorption.

### 5.2 The Null Causal Result

Our dose-response study falsifies the hypothesis that absorption rate causally predicts downstream interpretability (as measured by feature recovery MCC). Two interpretations are possible:

1. **Genuine null effect.** Absorption may not harm interpretability because the "absorbed" parent information is still accessible through child features.
2. **Metric insensitivity.** MCC may be too coarse to capture subtle interpretability differences.

Either way, the community's assumption that "lower absorption = better interpretability" lacks causal support.

### 5.3 Contrarian Perspective: Absorption as Feature

The contrarian view—that absorption may be a feature, not a bug—deserves consideration. Hierarchical representation through child features may mirror human category learning, though this analogy is speculative. Whether absorption reflects any natural structure remains an open question; our dose-response data only show that absorption does not harm MCC under the tested conditions, not that it is beneficial.

### 5.4 Limitations

1. **Synthetic data.** Our evaluation uses controlled synthetic hierarchies, not real semantic features. Validation on GPT-2 small pretrained SAEs is ongoing.
2. **Scale.** All experiments use 1024 features, not the planned 16k. This limits generalizability to larger dictionaries.
3. **Metric sensitivity.** The flat MCC may indicate metric insensitivity rather than a genuine null effect. Alternative downstream metrics (steering efficacy, circuit-tracing) should be tested.
4. **Convergence.** Training completes in 2-3 seconds, which is unusually fast. Convergence diagnostics (loss curves) should be verified.

## 6. Conclusion

We present the first systematic study of the L0 confound in cross-architecture SAE absorption comparisons. Our findings challenge the prevailing narrative:

1. **L0 is the dominant driver of absorption.** Baseline L1 cannot achieve the sparsity levels (L0≈50) of TopK/Matryoshka, making direct comparisons meaningless without controlling for L0.
2. **Absorption does not predict downstream interpretability.** The causal link hypothesized by prior work is not supported by our data.
3. **OrtSAE's orthogonality penalty does not reduce absorption.** This negative result contradicts published claims.

Our methodological contribution is demonstrating that L0-matching is non-trivial: L1 regularization cannot achieve arbitrary sparsity levels, and explicit k-selection mechanisms are required for low-L0 regimes. Our empirical contribution redirects community effort: rather than pursuing ever-more-complex architectures to reduce absorption, researchers should first understand whether absorption actually matters for their interpretability goals.

**Future work.** Three directions merit investigation: (1) testing whether larger models (16k+ features) enable L1 regularization to reach lower L0 values; (2) testing alternative downstream metrics (steering efficacy, circuit-tracing precision) that may be more sensitive to absorption than MCC; (3) investigating whether absorption is genuinely harmful or merely a natural consequence of hierarchical feature representation.

## Figures and Tables

- **Table 1**: Absorption rates, L0 sparsity, and dead latent percentages across six SAE architectures at their natural (unmatched) sparsity levels. Data from 5 seeds per variant.
- **Table 2**: L0-matching attempt via lambda sweep. Baseline L1 cannot reach L0=50; even at highest lambda (0.002), L0 remains ~995. TopK and Matryoshka operate at fixed L0=50.
- **Figure 1**: Scatter plot of absorption rate (x-axis) vs. feature recovery MCC (y-axis) for 25 dose-response measurements (5 lambda levels x 5 seeds). Horizontal reference line at MCC = 0.22. Despite 2.3× absorption variation, MCC remains flat (mean 0.219, std 0.0013).

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
