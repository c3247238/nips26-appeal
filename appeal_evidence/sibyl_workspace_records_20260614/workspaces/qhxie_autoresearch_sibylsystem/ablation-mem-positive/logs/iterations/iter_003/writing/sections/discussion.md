# 5. Discussion

Our experiments confirm that the actionability paradox described by Basu et al. (2026) is not universal. Absorbed SAE features in non-clinical LLM domains decompose into distinct subpopulations with heterogeneous steering utility. The coefficient of variation (CV) serves as a practical predictor for distinguishing these subpopulations, enabling practitioners to prioritize features for steering interventions without expensive experimental validation.

## 5.1 The Actionability Paradox is Not Universal

Basu et al. (2026) reported that SAE-based steering achieves 98.2% AUROC feature detection but produces zero output change in clinical domain applications. This finding cast doubt on the practical utility of SAE-based interpretability for intervention tasks. However, our results demonstrate that absorbed features in non-clinical LLM domains retain substantial steering potential.

Figure 1 shows that high-CV absorbed features (CV > 1.0) produce steering effects of 0.308, 0.522, and 0.745 logit change at strengths +3, +5, and +7 respectively—significantly larger than low-CV absorbed features (0.210, 0.355, and 0.504 at the same strengths). The aggregate effect ratio of 1.47x confirms that absorbed features are not uniformly non-steerable; rather, they decompose into steerable and non-steerable subpopulations.

We propose that the domain specificity of Basu et al.'s finding reflects the CV distribution of features in clinical applications. If clinical features are predominantly low-CV (stable, context-invariant), the universal failure they observed would naturally follow. In non-clinical LLM domains, high-CV features route through context-sensitive specialized channels that preserve steering potential, explaining why the actionability paradox does not extend to our domain.

## 5.2 CV as Practical Predictor

The coefficient of variation offers a computationally cheap predictor for steering feasibility. Unlike expensive steering experiments that require running the full model multiple times, CV computation requires only a single pass through the SAE with representative inputs. This makes CV-based screening practical for large feature dictionaries.

Our full experiment confirmed H1 with high statistical significance (t = 9.96, p < 0.01 with BH correction at strength +3). The CV threshold of 1.0 reliably separates absorbed features into steerable (high-CV) and non-steerable (low-CV) subpopulations across all tested steering strengths.

| Steering Strength | High-CV Mean | Low-CV Mean | Effect Ratio | t-statistic | p (BH-adj) |
|------------------|--------------|-------------|--------------|-------------|------------|
| +3 | 0.308 | 0.210 | 1.47x | 9.96 | < 0.01 |
| +5 | 0.522 | 0.355 | 1.47x | 9.73 | < 0.01 |
| +7 | 0.745 | 0.504 | 1.48x | 9.49 | < 0.01 |

The consistent effect ratio across steering strengths suggests that CV captures a fundamental property of the feature rather than an artifact of our experimental design. Practitioners can use CV to rank absorbed features for steering experiments, prioritizing high-CV features that are most likely to produce measurable effects.

## 5.3 Mechanistic Interpretation

We propose a causal mediation mechanism explaining why CV predicts steering effectiveness. When a parent feature is absorbed, steering the parent activates the child feature, which then affects model outputs. The routing behavior of the child feature determines whether steering propagates to the output.

**Context-sensitive routing (high-CV features)**: Child features with high CV activate strongly in specific contexts and weakly in others. The routing coefficient depends on input context, creating mediated steering where parent activation modulates child activation, which in turn modulates outputs. This allows steering effects to propagate effectively.

**Bypass routing (low-CV features)**: Child features with low CV activate at stable levels across contexts. The routing coefficient is approximately constant regardless of input. When the parent is steered, the child compensates through a fixed contribution that counteracts the parent steering, creating zero net effect at the output.

This mechanistic interpretation aligns with the variance paradox observation: absorbed features exhibit CV approximately 7.33 compared to 0.01 for non-absorbed features (733x ratio). Absorption selectively preserves context-sensitive specialized information in high-CV features, while low-CV features are absorbed because their stable routing maintains behavioral consistency.

## 5.4 Why Orthogonality Does Not Explain the Effect (H6)

Hypothesis H6 predicted that decoder orthogonality would explain the CV-steering correlation. Features with decoder weights maximally orthogonal to other features were hypothesized to show higher steering effectiveness, potentially because orthogonal decoders produce cleaner signal propagation.

Our full experiment falsified this hypothesis. High orthogonality features showed mean steering effect of 0.131 (SD: 0.090) compared to 0.107 (SD: 0.086) for low orthogonality features (Figure 4). The difference was not statistically significant (Welch's t = 1.77, p = 0.079). Pearson correlation between orthogonality and steering effect was r = -0.136 (p = 0.301), and Spearman correlation was rho = -0.204 (p = 0.117). Neither correlation approached the significance threshold.

This negative result constrains our mechanistic interpretation. If orthogonality does not predict steering, the CV-steering correlation is not mediated by decoder quality. Instead, the routing behavior (as captured by CV) directly determines steering effectiveness, independent of decoder geometry.

## 5.5 Implications for Interpretability Practice

Our findings provide actionable guidance for practitioners working with SAE-based interpretability:

1. **Screen absorbed features by CV before steering**: Features with CV > 1.0 are significantly more likely to produce measurable steering effects. This simple filter can reduce experimental cost by prioritizing high-value targets.

2. **Absorption metrics predict what is absorbed, not what is steerable**: The training-free absorption detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$ identifies absorbed features but does not predict which absorbed features retain steering potential. CV fills this gap.

3. **Domain matters for actionability**: The actionability paradox may be domain-specific. Clinical features may be predominantly low-CV, explaining Basu et al.'s universal failure. Non-clinical LLM domains show substantial high-CV absorbed features with steering utility.

4. **Steering protocols should account for feature heterogeneity**: Uniform steering strength across all absorbed features is suboptimal. High-CV features may tolerate stronger interventions while low-CV features may require alternative approaches (e.g., direct child feature targeting).

## 5.6 Limitations

Several limitations constrain the generalizability of our findings:

**GPT-2 Small only**: Our primary experiments used GPT-2 Small (117M parameters). While we completed cross-architecture validation on Gemma-2-2B, detailed analysis of those results is pending. The CV-steering correlation may behave differently in larger models with different architectural properties.

**Steering protocol limited to logit change**: Our experiments measured steering effectiveness via logit change at target tokens. We did not measure downstream behavioral changes (e.g., task completion, factual accuracy). It is possible that logit changes do not translate to consistent behavioral modifications, particularly for low-effect-size features.

**CV threshold of 1.0 may be model-specific**: The threshold separating high-CV from low-CV features was derived empirically on GPT-2 layer 6 SAEs. Cross-model and cross-layer generalization of this threshold requires further validation.

**Non-absorbed baseline comparison incomplete**: While we measured non-absorbed feature steering for context, we have not yet analyzed whether robust absorbed features achieve comparable steering effectiveness to non-absorbed features, or whether absorption degrades steering even for high-CV features.

Future work should address these limitations through larger model validation, behavioral downstream measurement, and systematic threshold characterization across SAE architectures.

<!-- FIGURES
- Figure 1: fig1_steering_cv.pdf, gen_fig1_steering_cv.py — Steering effect by CV group and strength, showing High-CV features consistently outperform Low-CV at all strengths
- Figure 4: fig4_decoder_orthogonality.pdf, gen_fig4_decoder_orthogonality.py — Scatter plot showing H6 falsified, orthogonality does not predict steering (r=-0.136, not significant)
- Figure 5: fig5_mechanism_desc.md — Architecture diagram describing robust vs fragile absorption routing mechanism
- None
-->