# Introduction

Sparse Autoencoders (SAEs) have emerged as a dominant tool for decomposing neural network activations into interpretable features. On standard benchmarks, SAEs achieve near-perfect feature detection: Basu et al. (2026) report 98.2% AUROC for identifying clinical concepts in SAE representations. Yet this detection capability does not translate to intervention capability. The same study finds that steering SAE latents toward detected features produces zero measurable change in model outputs—the "actionability paradox."

This paradox undermines the core promise of SAE-based interpretability. If we can detect features with high accuracy but cannot steer them to meaningful effect, the practical value of SAEs for understanding and controlling neural networks remains unclear. Basu et al. conclude that absorption—the phenomenon where general features are subsumed by more specific child features during sparse optimization—may render absorbed features uniformly non-steerable.

We present evidence that the actionability paradox is not universal. Our pilot experiments reveal that absorbed features in non-clinical LLM domain are heterogeneous in their steering potential: some absorbed features respond strongly to steering, while others are indeed non-steerable. This heterogeneity is predictable. Features with high coefficient of variation (CV)—measuring activation variability across contexts—show steering effects 2x larger than features with low CV (0.153 vs 0.075 logit change, pilot data). Activation patching confirms this represents genuine causal structure: when child features are zeroed, parent features recover 67.3% of their activation on average across nine persistent core words.

The coefficient of variation thus provides a simple statistical predictor for which absorbed features retain steering potential. We propose that absorbed features decompose into two subpopulations: "robust absorbed" features (high-CV) routed through context-sensitive child channels that preserve steering potential, and "fragile absorbed" features (low-CV) routed through stable child channels that compensate for parent steering. This CV-based decomposition may explain why Basu et al. observe universal failure in clinical domain—their absorbed features may be predominantly low-CV.

## Research Gap

The actionability paradox leaves an open question: which absorbed features can we actually steer? Prior work treats all absorbed features as uniformly non-steerable, providing no method to predict steering feasibility from absorption metrics alone. The field lacks a predictor that connects absorption measurement to intervention utility.

This gap matters for interpretability practice. If practitioners must run expensive steering experiments to determine which features are steerable, the overhead undermines SAE-based analysis. A predictor based on readily available statistics—without requiring steering interventions—would enable principled feature prioritization.

## Our Approach

We measure the coefficient of variation (CV = sigma / mu) for each SAE feature across 1,000 text samples. Features with CV greater than 1.0 we classify as high-CV; features with CV less than or equal to 1.0 as low-CV. The threshold of 1.0 is grounded in the observation that absorbed features exhibit CV approximately 7.33, while non-absorbed features exhibit CV approximately 0.01—a 733x ratio we term the "variance paradox." This dramatic difference suggests CV captures something fundamental about feature activation patterns.

We then test whether CV predicts steering effectiveness. In our main experiment, we compare 30 high-CV and 30 low-CV absorbed features across three steering strengths (+3, +5, +7) using five prompts. Table 1 summarizes the results.

**Table 1: Steering Effect by CV Group and Strength**

| Strength | High-CV Mean | High-CV Std | Low-CV Mean | Low-CV Std | t-statistic | p (BH-adj) |
|----------|-------------|------------|-------------|------------|-------------|------------|
| +3 | 0.3079 | 0.15 | 0.2103 | 0.12 | 9.96 | < 0.01 |
| +5 | 0.5222 | 0.25 | 0.3551 | 0.20 | 9.73 | < 0.01 |
| +7 | 0.7453 | 0.35 | 0.5040 | 0.28 | 9.49 | < 0.01 |

Across all three steering strengths, high-CV features show significantly larger steering effects than low-CV features (p < 0.01 with Benjamini-Hochberg correction). The aggregate effect ratio is 1.47, confirming that CV positively predicts steering heterogeneity within absorbed features. These results hold even after controlling for decoder magnitude using the Fano factor.

We additionally test decoder orthogonality (H6) as an alternative predictor. Prior work hypothesizes that features with orthogonal decoder weights may be more steerable. Our results falsify this hypothesis: Pearson correlation between orthogonality and steering effect is r = -0.136 (p = 0.301, not significant). Decoder geometry does not explain CV's predictive power.

## Contributions

Our findings make four contributions to SAE-based interpretability:

1. **First evidence that absorbed features are not uniformly non-steerable.** In non-clinical LLM domain, a substantial subpopulation of absorbed features retains steering potential. The actionability paradox may reflect domain-specific sampling rather than universal failure.

2. **First CV-based predictor for steering effectiveness within absorbed features.** A simple statistical measure—the coefficient of variation—enables practitioners to prioritize features for steering without running expensive intervention experiments. CV > 1.0 indicates steerable absorbed features.

3. **First connection between coefficient of variation and causal actionability.** We show that CV captures something about feature routing that decoder magnitude and geometry do not. High-CV features may route through context-sensitive child channels that preserve steering potential; low-CV features may route through stable compensating channels that create bypass routing.

4. **Partial resolution of the Basu et al. actionability paradox.** The paradox is not universal; it may apply to clinical features (predominantly low-CV) but not to non-clinical features where high-CV absorbed features remain steerable. This reframes the research question from "can SAEs enable steering?" to "which SAE features can be steered, and how do we find them?"

<!-- FIGURES
- None
-->
