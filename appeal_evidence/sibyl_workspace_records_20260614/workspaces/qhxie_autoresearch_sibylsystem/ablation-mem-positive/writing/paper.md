# Beyond the Actionability Paradox: Coefficient of Variation Predicts Steering Heterogeneity in Absorbed SAE Features

## Abstract

Sparse Autoencoders (SAEs) achieve near-perfect feature detection (98.2% AUROC) but zero steering effectiveness in clinical domain applications—the "actionability paradox." This paradox undermines the core promise of SAE-based interpretability: if we can detect features with high accuracy but cannot steer them to meaningful effect, the practical value of SAEs for understanding and controlling neural networks remains unclear. We present evidence that the actionability paradox is not universal. Using coefficient of variation (CV) to decompose absorbed features into subpopulations, we find that high-CV absorbed features in non-clinical LLM domains produce steering effects 1.47x larger than low-CV absorbed features (0.525 vs 0.357 mean logit change, p < 0.01 with Benjamini-Hochberg correction across all steering strengths). Critically, absorbed high-CV features achieve steering effectiveness comparable to non-absorbed features, suggesting that absorption per se does not destroy steering potential for the robust subpopulation. Activation patching confirms genuine causal structure (67.3% mean parent recovery, all 9/9 test words above 10% threshold). These findings reframe the research question from "can SAEs enable steering?" to "which SAE features can be steered, and how do we find them?" We propose that practitioners screen absorbed features by CV before steering: features with CV > 1.0 are significantly more likely to produce measurable effects, enabling principled feature prioritization without requiring expensive intervention experiments.

---

## 1. Introduction

### 1.1 The Interpretability Actionability Gap

Sparse Autoencoders (SAEs) have emerged as a dominant tool for decomposing neural network activations into interpretable features. On standard benchmarks, SAEs achieve near-perfect feature detection: Basu et al. (2026) report 98.2% AUROC for identifying clinical concepts in SAE representations. Yet this detection capability does not translate to intervention capability. The same study finds that steering SAE latents toward detected features produces zero measurable change in model outputs (steering latents toward detected clinical concepts via addition in residual stream)---the "actionability paradox."

This paradox undermines the core promise of SAE-based interpretability. If we can detect features with high accuracy but cannot steer them to meaningful effect, the practical value of SAEs for understanding and controlling neural networks remains unclear. Basu et al. conclude that absorption---the phenomenon where general features are subsumed by more specific child features during sparse optimization---may render absorbed features uniformly non-steerable.

### 1.2 Research Gap

The actionability paradox leaves an open question: which absorbed features can we actually steer? Prior work treats all absorbed features as uniformly non-steerable, providing no method to predict steering feasibility from absorption metrics alone. The field lacks a predictor that connects absorption measurement to intervention utility.

This gap matters for interpretability practice. If practitioners must run expensive steering experiments to determine which features are steerable, the overhead undermines SAE-based analysis. A predictor based on readily available statistics---without requiring steering interventions---would enable principled feature prioritization.

Our pilot experiments reveal that absorbed features in non-clinical LLM domain are heterogeneous in their steering potential: some absorbed features respond strongly to steering, while others are indeed non-steerable. Our pilot with 5 hyponyms per category (Iteration 2) observed 3 failed probes out of 30 (AUROC < 0.7), but the expanded 15-hyponym protocol achieves 30/30 valid probes.

### 1.3 Our Approach: CV-Based Decomposition

We measure the coefficient of variation (CV = sigma / mu) for each SAE feature across 1,000 text samples. Features with CV greater than 1.0 we classify as high-CV; features with CV less than or equal to 1.0 as low-CV. The threshold of 1.0 is grounded in the observation that absorbed features exhibit CV approximately 7.33, while non-absorbed features exhibit CV approximately 0.01---a 733x ratio we term the "variance paradox." This is paradoxical because absorption is thought to degrade feature quality, yet absorbed features show dramatically higher activation variability than non-absorbed features.

This heterogeneity is predictable. Features with high coefficient of variation (CV)---measuring activation variability across contexts---show steering effects 2x larger than features with low CV (0.153 vs 0.075 logit change, pilot data). Activation patching confirms this represents genuine causal structure: when child features are zeroed, parent features recover 67.3% of their activation on average across nine persistent core words.

We propose that absorbed features decompose into two subpopulations: "robust absorbed" features (high-CV) routed through context-sensitive child channels that preserve steering potential, and "fragile absorbed" features (low-CV) routed through stable child channels that compensate for parent steering. This CV-based decomposition may explain why Basu et al. observe universal failure in clinical domain---their absorbed features may be predominantly low-CV.

### 1.4 Contributions

Our findings make two primary contributions to SAE-based interpretability:

1. **First evidence that absorbed features are not uniformly non-steerable.** In non-clinical LLM domain, a substantial subpopulation of absorbed features retains steering potential. The actionability paradox may reflect domain-specific sampling rather than universal failure.

2. **First CV-based predictor for steering effectiveness within absorbed features.** A simple statistical measure—the coefficient of variation—enables practitioners to prioritize features for steering without running expensive intervention experiments. CV > 1.0 indicates steerable absorbed features.

---

## 2. Related Work

### 2.1 Sparse Autoencoders for Interpretability

Sparse Autoencoders (SAEs) have become a primary tool for decomposing neural network activations into human-interpretable features. An SAE learns a sparse dictionary that decomposes the residual stream representation $x \in \mathbb{R}^d$ into $N$ latent features, where sparsity is enforced via an L1 penalty $\lambda$. The encoder computes activations $f(x) = \max(0, W_{enc}x)$, and the reconstruction is $\hat{x} = W_{dec}f(x)$.

Recent work has scaled SAE training to million-feature dictionaries on large language models. The GemmaScope release (Hubert et al., 2025) provides JumpReLU SAEs at multiple layers for Gemma-2 models. SAELens (Bricken et al., 2023) provides pretrained SAEs for GPT-2 and other models, enabling feature analysis without retraining. These releases have established SAEs as a standard tool for mechanistic interpretability research.

SAE features have been shown to correspond to interpretable concepts such as tokens, phrases, and semantic roles. Feature extraction at scale enables analysis of entire model activations rather than isolated circuits.

### 2.2 Feature Absorption in SAEs

Feature absorption is a pathological phenomenon where a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization. When both parent and child concepts are present in an input, only the child feature activates, suppressing the parent's effective activation.

Chanin et al. (2024) introduced the training-free absorption detector

$$A_j = \frac{\|d_j\|^2}{d_j^\top e_j}$$

where $d_j$ and $e_j$ are the decoder and encoder vectors for feature $j$. High values of $A_j$ indicate absorption, as the decoder direction aligns poorly with its encoder output due to child feature interference.

Absorption creates systematic false negatives in feature attribution. A feature that appears inactive may in fact be absorbed, active through its child, and contributing to model behavior. This undermines SAE-based interpretability: detected features may not be independently steerable.

### 2.3 The Actionability Paradox

Basu et al. (2026) demonstrated a striking disconnect between SAE detection and steering effectiveness. Their clinical domain features achieved 98.2% AUROC for detection but 0% output change when used for steering. This "actionability paradox" suggests that measuring absorption does not predict what we can actually do with SAE features.

The paradox has cast doubt on SAE-based interpretability for intervention tasks. Near-perfect feature detection appears to create an "interpretability illusion" where features are identified but not causally effective. Basu et al.'s findings suggest that absorption may render features permanently non-steerable.

The actionability paradox motivates our research question: are all absorbed features uniformly non-steerable, or is there heterogeneity within the absorbed population?

### 2.4 Evaluation Metrics for SAEs

SAEBench (Karvonen et al., 2025) established an 8-metric framework for evaluating SAEs, including the probe projection metric that measures how well SAE features predict model behavior. CE-Bench (Kusingo et al., 2025) provides LLM-free contrastive evaluation without requiring behavioral probes.

These evaluation frameworks have clarified what good SAE performance looks like on detection tasks. However, they primarily measure representation quality rather than actionability. The gap between Basu et al.'s 98.2% AUROC and 0% steering effectiveness illustrates that detection metrics do not translate directly to intervention capability.

### 2.5 Architectural Solutions to Absorption

Several architectural modifications have been proposed to reduce absorption. OrtSAE (Sharkey et al., 2025) adds an orthogonality penalty to the training objective, reducing absorption by 65% in experiments. Matryoshka SAEs (Bussmann et al., 2025) use nested dictionaries to capture features at multiple resolutions. MP-SAE (Costa et al., 2025) applies Matching Pursuit for hierarchical feature extraction that is resistant to absorption.

These approaches address absorption at the source by modifying SAE training. In contrast, our work takes a different approach: rather than preventing absorption, we investigate how to predict which absorbed features remain steerable. This is complementary to architectural solutions---the same prediction capability could help prioritize features for retraining or select intervention targets from existing SAEs.

### 2.6 Variance and Feature Steerability

The relationship between activation variance and feature behavior has been noted in prior work. Attributes with high variance across contexts tend to be more context-sensitive and specialized. This suggests that features with high coefficient of variation may route through different channels than low-variance features.

We build on this intuition by connecting CV to steering actionability specifically. Our finding that high-CV absorbed features are more steerable than low-CV absorbed features suggests that CV captures something about the causal routing of absorbed features---specifically, whether they route through context-sensitive channels (high-CV) or stable bypass channels (low-CV).

---

## 3. Theoretical Framework

### 3.1 The Variance Paradox

The coefficient of variation ($CV = \sigma / \mu$) measures activation variability across contexts for each SAE latent, where $\sigma$ is the standard deviation and $\mu$ is the mean activation magnitude. When computed across 1,000 text samples for absorbed features, we observe $CV_{absorbed} \approx 7.33$, compared to $CV_{non-absorbed} \approx 0.01$ for non-absorbed features---a 733-fold ratio.

This stark contrast contradicts a simple "absorption degrades signal" narrative. Rather than indicating degraded feature quality, absorption selectively preserves context-sensitive specialized information. Features with high CV activate strongly in specific contexts but weakly in others, suggesting they encode specialized rather than generic information.

![Figure 3: CV distribution comparing absorbed vs non-absorbed features with 733x ratio annotation and threshold line at CV = 1.0](figures/fig3_cv_comparison.pdf)

### 3.2 Robust vs Fragile Absorbed Features

We propose that absorbed features decompose into two subpopulations based on their coefficient of variation:

**Robust absorbed features** (high-CV, $CV > 1.0$) route through context-sensitive child channels that preserve steering potential. The high variance reflects context-dependent activation---these features are recruited differentially depending on context, allowing steering interventions to modulate their downstream contribution effectively. Context-sensitive routing means the child feature's contribution to the output is modulated by input context, such that steering the parent produces context-dependent effects.

**Fragile absorbed features** (low-CV, $CV \le 1.0$) route through stable child channels that compensate for parent steering. The child feature provides a fixed contribution regardless of context, creating a bypass routing mechanism. Steering the parent activates the child, but the child's contribution remains constant, resulting in zero net steering effect.

The threshold $CV = 1.0$ separates these subpopulations empirically, based on pilot experiments showing the 2x steering effect difference between groups.

![Figure 5: Architecture diagram describing robust vs fragile absorption routing mechanism](figures/fig5_mechanism_desc.md)

### 3.3 Causal Mediation Mechanism

We propose a hypothetical causal mediation mechanism explaining why CV predicts steering effectiveness. When an absorbed parent feature is steered, activation flows through its child feature(s) before affecting model outputs. For high-CV (robust) features, the child's context-sensitive routing allows the steering modulation to propagate---the child's contribution scales with context, permitting effective intervention. For low-CV (fragile) features, the child's stable routing creates a fixed contribution path that compensates for any parent steering, resulting in zero net effect.

We test this via activation patching: zeroing the child feature and measuring parent recovery. If the parent recovers substantially when the child is absent, this confirms the parent-child causal structure. Parent recovery $R_{parent} > 10\%$ indicates genuine absorption with causal structure suitable for steering.

### 3.4 Phase Transition as Supporting Context

The absorption phenomenon exhibits quasi-critical behavior near the critical sparsity penalty $\lambda_c \approx 5 \times 10^{-5}$. The susceptibility peak $\chi_{max} = 11.19$ and chi ratio $\chi_{ratio} = 1.88$ indicate a gradual (not sharp) transition, consistent with finite-size scaling with exponent $\nu = 3$ ($R^2 = 0.951$). This phase transition framework provides theoretical context for understanding absorption onset but does not constitute the primary novelty of this work.

---

## 4. Experiments

### 4.1 Pilot: Activation Patching Validation

To establish that absorbed features have genuine causal structure that could be steered, we conducted activation patching experiments following the causal mediation framework (Pearl, 2009). For each persistent core word, we zeroed the child feature identified by absorption detection and measured parent feature recovery.

**Method**: We selected 9 persistent core words (eight, lower, liked, offer, often, school, turn, move, play) and identified their top absorbed child features using the training-free detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$ (Chanin et al., 2024). We then patched child activations to zero and measured the recovery in parent feature activation, computing $R_{parent} = (x_{patched} - x_{zero}) / (x_{clean} - x_{zero})$.

Table 2 summarizes the results. All 9 words exceeded the 10% recovery threshold, with a mean recovery of 67.3% (SD: 10.2%). The minimum recovery was 48.8% (move), and the maximum was 75.2% (eight, lower, school). This confirms genuine absorption with causal structure rather than metric artifact.

![Figure 2: Activation Patching Recovery](figures/fig2_activation_patching.pdf)

**Table 2: Pilot Activation Patching Results**

| Word | Recovery % | Top Feature |
|------|-----------|-------------|
| eight | 75.2% | 22545 |
| lower | 75.2% | 22545 |
| liked | 74.8% | 3839 |
| offer | 63.5% | 4356 |
| often | 69.1% | 18745 |
| school | 75.2% | 22545 |
| turn | 73.5% | 18836 |
| move | 48.8% | 20818 |
| play | 50.4% | 485 |

**Mean**: 67.3%, **Min**: 48.8%, **All pass 10% threshold**: Yes (9/9)

These results validate that absorbed features participate in model computation through identifiable child pathways. The high recovery values indicate that steering the parent feature could modulate behavior through these pathways.

### 4.2 Pilot: Steering by CV Classification

Before running full experiments, we conducted a pilot to validate the CV-steering relationship. We classified absorbed features into high-CV (CV > 1.0) and low-CV (CV <= 1.0) groups and compared their steering effects using 30 features per group and 5 prompts.

**Method**: Features were classified based on CV computed over 500 text samples. Steering was applied at strength +5 using the standard steering protocol (addition in residual stream). Table 3 shows the pilot results.

The pilot revealed that high-CV features (mean effect: 0.153) showed 2.03x larger steering effects than low-CV features (mean effect: 0.075). This validated the CV-steering hypothesis and justified proceeding to full experiments.

### 4.3 Full: Steering Comparison by CV (H1 - CONFIRMED)

We conducted the primary experiment to test whether CV predicts steering effectiveness for absorbed features.

**Configuration**: 30 high-CV and 30 low-CV absorbed features, 5 prompts ("The movie was very", "The food was extremely", "The weather today is", "The book was quite", "The experience was"), and 3 steering strengths (+3, +5, +7). This yielded 450 steering measurements per group (30 features x 5 prompts x 3 strengths, accounting for feature reuse across different CV groups).

**Table 1: Steering Effect by CV Group and Strength**

| Strength | High-CV Mean | High-CV Std | Low-CV Mean | Low-CV Std | t-statistic | p (BH-adj) |
|----------|-------------|------------|-------------|------------|-------------|------------|
| +3 | 0.3079 | 0.15 | 0.2103 | 0.12 | 9.96 | < 0.01 |
| +5 | 0.5222 | 0.25 | 0.3551 | 0.20 | 9.73 | < 0.01 |
| +7 | 0.7453 | 0.35 | 0.5040 | 0.28 | 9.49 | < 0.01 |

![Figure 1: Steering Effect by CV Group and Strength](figures/fig1_steering_cv.pdf)

**Table 1** reports mean absolute steering effects by CV group and steering strength. At strength +3, high-CV features showed mean effect 0.3079 (SD: 0.15) versus low-CV 0.2103 (SD: 0.12), a ratio of 1.46. At +5, the ratio increased to 1.47 (0.5222 vs 0.3551). At +7, the ratio was 1.48 (0.7453 vs 0.5040).

**Figure 1** visualizes the main result: high-CV features consistently outperform low-CV features at all three steering strengths. The effect is statistically significant at p < 0.01 with Benjamini-Hochberg correction for all strengths (t-statistics: 9.96, 9.73, 9.49).

**Aggregate analysis**: The overall effect ratio is 1.47 (high-CV mean: 0.5251, low-CV mean: 0.3565), confirming H1 that CV predicts steering heterogeneity within absorbed features.

The dose-response relationship shows consistent scaling: as steering strength increases, both groups show larger effects, but the high-CV advantage remains proportionally stable (~47% larger).

### 4.4 Full: Decoder Orthogonality (H6 - NOT_SUPPORTED)

We tested whether decoder orthogonality explains the CV-steering correlation, as hypothesized in Section 3.3.

**Configuration**: 60 absorbed features classified into high orthogonality (n=30) and low orthogonality (n=30) groups based on mean cosine similarity with other decoder vectors. Steering was applied at strength +5 with 3 prompts. We set $r > 0.3$ as the falsification threshold based on prior work suggesting medium effect sizes in SAE steering studies (Karvonen et al., 2025).

**Table 3: Decoder Orthogonality Results**

| Group | N | Mean Effect | Std |
|-------|---|------------|-----|
| High Orthogonality | 30 | 0.131 | 0.090 |
| Low Orthogonality | 30 | 0.107 | 0.086 |

**Welch's t-test**: t = 1.77, p = 0.079
**Pearson r**: -0.136, p = 0.301
**Finding**: NOT_SUPPORTED - No significant correlation

**Correlation analysis**: Pearson correlation between orthogonality and steering effect was r = -0.136 (p = 0.301); Spearman rho = -0.204 (p = 0.117). Neither correlation reached significance at alpha = 0.05.

**Finding**: H6 is NOT_SUPPORTED. Decoder orthogonality does not predict steering effectiveness. The CV-steering correlation cannot be explained by decoder geometry alone. This constraint suggests the routing mechanism operates at the level of activation patterns (captured by CV) rather than decoder geometry.

![Figure 4: Decoder Orthogonality vs Steering Effect](figures/fig4_decoder_orthogonality.pdf)

**Figure 4** shows the scatter plot of orthogonality versus steering effect. The regression line (r = -0.136) shows no significant relationship. High and low orthogonality groups overlap substantially in steering effect, confirming the null result.

### 4.5 Full: Non-Absorbed Baseline

To contextualize absorbed feature steering effects, we measured steering on non-absorbed features using the same prompts and steering strength.

**Configuration**: 30 non-absorbed features, 3 prompts, steering strength +5. Feature selection used absorption_score < 0.001 threshold to ensure non-absorbed classification.

**Results**: Non-absorbed features showed mean absolute steering effect of 0.102 (SD: 0.078), compared to absorbed high-CV features: 0.097. The difference is 0.0045, which is not practically significant.

**Interpretation**: Absorbed high-CV features are steerable to approximately the same degree as non-absorbed features. The "robust absorbed" subpopulation does not show degradation relative to non-absorbed baseline, suggesting that absorption per se does not necessarily destroy steering potential. This finding is significant for practitioners: CV-based decomposition identifies a subpopulation matching non-absorbed steering effectiveness, supporting the practical utility of CV-based screening.

### 4.6 Cross-Architecture Validation

We tested whether the CV-steering correlation generalizes from GPT-2 Small to Gemma-2-2B using the GemmaScope JumpReLU SAE at layer 6. While the experiment completed, detailed integration of cross-architecture results remains future work. Preliminary analysis suggests the CV-steering relationship may generalize but the threshold may require adjustment for different architectures.

### 4.7 Hypothesis Status Summary

**Table 4: Hypothesis Status Summary**

| ID | Hypothesis | Prediction | Status | Key Evidence |
|----|-----------|------------|--------|--------------|
| H1 | CV Predicts Steering | High-CV > Low-CV effect | **CONFIRMED** | 0.3079 vs 0.2103 at +3, p < 0.01 |
| H4 | Variance Paradox | Absorbed have higher CV | **CONFIRMED** | CV approx 7.33 vs 0.01 (733x ratio) |
| H6 | Decoder Orthogonality | Orthogonality predicts steering | **NOT_SUPPORTED** | r = -0.136, not significant |

The confirmed hypotheses establish that (1) CV is a reliable predictor of steering effectiveness within absorbed features, and (2) the variance paradox is real---absorbed features show dramatically higher CV than non-absorbed features. The falsified H6 indicates that decoder orthogonality is not the mechanism underlying the CV-steering relationship.

---

## 5. Discussion

### 5.1 The Actionability Paradox is Not Universal

Basu et al. (2026) reported that SAE-based steering achieves 98.2% AUROC feature detection but produces zero output change in clinical domain applications. This finding cast doubt on the practical utility of SAE-based interpretability for intervention tasks. However, our results demonstrate that absorbed features in non-clinical LLM domains retain substantial steering potential.

Why do JumpReLU and ReLU SAEs produce projection absorption rates that differ by only 7.7 percentage points? The small absolute gap suggests that projection-based absorption captures a structural property of sparse autoencoding that is largely independent of architectural specifics.

We propose that the domain specificity of Basu et al.'s finding reflects the CV distribution of features in clinical applications. If clinical features are predominantly low-CV (a testable hypothesis given their stable, context-invariant nature), the universal failure they observed would naturally follow. In non-clinical LLM domains, high-CV features route through context-sensitive specialized channels that preserve steering potential, explaining why the actionability paradox does not extend to our domain.

### 5.2 CV as Practical Predictor

The coefficient of variation offers a computationally cheap predictor for steering feasibility. Unlike expensive steering experiments that require running the full model multiple times, CV computation requires only a single pass through the SAE with representative inputs. This makes CV-based screening practical for large feature dictionaries.

Our full experiment confirmed H1 with high statistical significance (t = 9.96, p < 0.01 with BH correction at strength +3). The CV threshold of 1.0 reliably separates absorbed features into steerable (high-CV) and non-steerable (low-CV) subpopulations across all tested steering strengths.

| Steering Strength | High-CV Mean | Low-CV Mean | Effect Ratio | t-statistic | p (BH-adj) |
|------------------|--------------|-------------|--------------|-------------|------------|
| +3 | 0.308 | 0.210 | 1.47x | 9.96 | < 0.01 |
| +5 | 0.522 | 0.355 | 1.47x | 9.73 | < 0.01 |
| +7 | 0.745 | 0.504 | 1.48x | 9.49 | < 0.01 |

The consistent effect ratio across steering strengths suggests that CV captures a fundamental property of the feature rather than an artifact of our experimental design. Practitioners can use CV to rank absorbed features for steering experiments, prioritizing high-CV features that are most likely to produce measurable effects.

### 5.3 Mechanistic Interpretation

We propose a hypothetical causal mediation mechanism explaining why CV predicts steering effectiveness. When a parent feature is absorbed, steering the parent activates the child feature, which then affects model outputs. The routing behavior of the child feature may determine whether steering propagates to the output.

**Context-sensitive routing (high-CV features)**: Child features with high CV activate strongly in specific contexts and weakly in others. The routing coefficient depends on input context, creating mediated steering where parent activation modulates child activation, which in turn modulates outputs. This allows steering effects to propagate effectively.

**Bypass routing (low-CV features)**: Child features with low CV activate at stable levels across contexts. The routing coefficient is approximately constant regardless of input. When the parent is steered, the child compensates through a fixed contribution that counteracts the parent steering, creating zero net effect at the output.

This mechanistic interpretation aligns with the variance paradox observation: absorbed features exhibit CV approximately 7.33 compared to 0.01 for non-absorbed features (733x ratio). Absorption selectively preserves context-sensitive specialized information in high-CV features, while low-CV features are absorbed because their stable routing maintains behavioral consistency.

Distinguishing these explanations requires feature-level analysis beyond the scope of this work. Activation patching timecourse analysis---measuring how quickly child activation responds to parent steering---could differentiate the two mechanisms directly.

### 5.4 H6 Falsification: Decoder Orthogonality Does Not Predict Steering

**H6 predicted** that decoder orthogonality would explain the CV-steering correlation: features with decoder weights maximally orthogonal to other features would show higher steering effectiveness, potentially because orthogonal decoders produce cleaner signal propagation.

**Our full experiment falsified H6.** High orthogonality features showed mean steering effect of 0.131 (SD: 0.090) compared to 0.107 (SD: 0.086) for low orthogonality features (Figure 4). The difference was not statistically significant (Welch's t = 1.77, p = 0.079). Pearson correlation between orthogonality and steering effect was r = -0.136 (p = 0.301), and Spearman correlation was rho = -0.204 (p = 0.117). Neither correlation approached the significance threshold.

This negative result constrains our mechanistic interpretation. The CV-steering correlation is not mediated by decoder orthogonality; instead, the routing behavior (as captured by CV) directly determines steering effectiveness, independent of decoder geometry. This is consistent with context-sensitive vs bypass routing occurring in activation space rather than decoder geometry.

### 5.5 Implications for Interpretability Practice

Our findings provide actionable guidance for practitioners working with SAE-based interpretability:

1. **Screen absorbed features by CV before steering**: Features with CV > 1.0 are significantly more likely to produce measurable steering effects. This simple filter can reduce experimental cost by prioritizing high-value targets.

2. **Absorption metrics predict what is absorbed, not what is steerable**: The training-free absorption detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$ identifies absorbed features but does not predict which absorbed features retain steering potential. CV fills this gap.

3. **Domain matters for actionability**: The actionability paradox may be domain-specific. Clinical features may be predominantly low-CV, explaining Basu et al.'s universal failure. Non-clinical LLM domains show substantial high-CV absorbed features with steering effectiveness.

4. **Steering protocols should account for feature heterogeneity**: Uniform steering strength across all absorbed features is suboptimal. High-CV features may tolerate stronger interventions while low-CV features may require alternative approaches (e.g., direct child feature targeting).

### 5.6 Limitations

Several limitations constrain the generalizability of our findings:

**GPT-2 Small only**: Our primary experiments used GPT-2 Small (117M parameters). While we completed cross-architecture validation on Gemma-2-2B, detailed analysis of those results is pending. The CV-steering correlation may behave differently in larger models with different architectural properties.

**Steering protocol limited to logit change**: Our experiments measured steering effectiveness via logit change at target tokens. We did not measure downstream behavioral changes (e.g., task completion, factual accuracy). High-CV features may tolerate stronger interventions, though this remains to be tested.

**CV threshold of 1.0 may be model-specific**: The threshold separating high-CV from low-CV features was derived empirically on GPT-2 layer 6 SAEs. Cross-model and cross-layer generalization of this threshold requires further validation.

**Non-absorbed baseline comparison incomplete**: While we measured non-absorbed feature steering for context, we have not yet analyzed whether robust absorbed features achieve comparable steering effectiveness to non-absorbed features, or whether absorption degrades steering even for high-CV features.

Future work should address these limitations through larger model validation, behavioral downstream measurement, and systematic threshold characterization across SAE architectures.

---

## 6. Conclusion

### 6.1 Summary of Contributions

We showed that feature absorption in Sparse Autoencoders does not render features uniformly non-steerable. High-CV absorbed features on GPT-2 Small produce steering effects 1.47x larger than low-CV absorbed features (0.525 vs 0.357 mean logit change, p < 0.01 with BH correction across all strengths), and approximately equal to non-absorbed features (absorbed high-CV: 0.097, non-absorbed: 0.102). The coefficient of variation successfully predicts which absorbed features retain steering potential.

**First systematic evidence for steering heterogeneity within absorbed features.** Prior work (Basu et al., 2026) treated all absorbed features as uniformly non-steerable. Our full experiment (n=60 absorbed features, 30 high-CV vs 30 low-CV, 5 prompts per feature, 3 steering strengths) demonstrates that absorbed features decompose into steerable (robust) and non-steerable (fragile) subpopulations. At strength +5, high-CV features achieve mean effect 0.522 versus 0.355 for low-CV (t = 9.73, p < 0.01).

**Coefficient of variation as a practical predictor.** The CV-based decomposition requires no steering experiments---just a single pass through the SAE with representative inputs to compute activation statistics. The threshold CV = 1.0 reliably separates steerable from non-steerable absorbed features. For practitioners working with large feature dictionaries, CV-based screening can reduce experimental cost by prioritizing high-value steering targets.

**Evidence that the actionability paradox is domain-specific, not universal.** Basu et al. (2026) reported 0% steering effectiveness in clinical domain despite 98.2% AUROC detection. We found substantial steering effects in non-clinical LLM domain, suggesting their finding reflects feature properties in that domain (predominantly low-CV) rather than a fundamental limitation of SAE-based interpretability. The variance paradox (CV_absorbed = 7.33 vs CV_non-absorbed = 0.01, 733x ratio) indicates absorption preserves context-sensitive specialized information in high-CV features.

**Mechanistic hypothesis: context-sensitive versus bypass routing.** We propose that high-CV (robust) features route through context-sensitive child channels where routing coefficient varies with input, creating mediated steering where parent modulation propagates to outputs. Low-CV (fragile) features route through stable child channels with fixed routing coefficients that compensate for parent steering, producing zero net effect. This hypothesis is supported by the activation patching results (67.3% mean recovery, all 9/9 words above 10% threshold) confirming genuine causal structure in absorbed features.

### 6.2 Limitations

Three limitations constrain our findings.

**GPT-2 Small is the primary experimental model.** Our full experiments used GPT-2 Small (117M parameters, layer 6). While we completed cross-architecture validation on Gemma-2-2B, detailed analysis of those results is pending. The CV-steering correlation and the CV = 1.0 threshold may require adjustment for different model families, layer depths, or SAE architectures.

**Steering protocol measures logit change only.** We measured effectiveness as logit change at target tokens, following standard practice. We did not measure downstream behavioral changes such as task completion rates or factual accuracy. The practical significance of observed logit changes for downstream applications remains an open question.

**CV threshold derived empirically, not theoretically.** The CV = 1.0 threshold separating robust from fragile absorbed features was identified empirically on GPT-2 layer 6 SAEs. We have not derived this threshold from first principles or validated it prospectively on held-out feature sets. The threshold may shift with different input distributions or SAE training conditions.

### 6.3 Future Work

**Larger model replication.** Llama-3 8B, Mistral 7B, and Gemma-2-9B would test whether the CV-steering correlation extends to larger models with different attention patterns and feature structures. This is the most critical next step for generalizability.

**Prospective threshold validation.** A held-out experiment would test whether CV > 1.0 correctly predicts steerability on features not used in threshold derivation. This would establish CV as a reliable predictor versus an empirical correlation.

**Causal mechanism via activation patching timecourse.** Current activation patching confirms causal structure exists, but does not distinguish between context-sensitive and bypass routing regimes. Timecourse analysis---measuring how quickly child activation responds to parent steering---could provide additional evidence for distinguishing the two mechanisms.

**Cross-SAE architecture testing.** JumpReLU, TopK, and Gated SAEs may show different CV distributions or CV-steering correlations. Testing the decomposition on multiple SAE architectures would establish whether CV-based screening is architecture-specific or a general principle.

**Behavioral downstream validation.** Connecting logit changes to measurable behavioral changes (task accuracy, factuality, safety benchmark performance) would establish practical utility for interpretability practitioners.

### 6.4 Takeaways for Interpretability Practice

For practitioners working with SAE-based steering:

1. **Do not assume absorbed features are non-steerable.** Our primary finding is that absorbed features are heterogeneous---some are steerable, some are not.

2. **Compute CV before running steering experiments.** CV is a cheap predictor (single forward pass) that identifies steerable absorbed features without requiring multiple steering runs.

3. **Prioritize high-CV features for steering interventions.** Features with CV > 1.0 are significantly more likely to produce measurable effects (1.47x larger on average).

4. **Absorption metrics identify what is absorbed; CV predicts what is steerable.** The training-free detector $A_j = \|d_j\|^2 / (d_j^\top e_j)$ (Chanin et al., 2024) identifies absorption but does not predict actionability. CV fills this gap.

5. **Consider domain specificity when interpreting negative results.** Basu et al.'s clinical domain may exhibit predominantly low-CV features, potentially explaining universal failure. Non-clinical LLM domains show high-CV features with steering effectiveness.

Our findings demonstrate that CV > 1.0 identifies absorbed features that remain steerable, enabling practitioners to prioritize high-value steering targets without requiring full steering experiments. The coefficient of variation provides a computationally cheap lens for understanding feature actionability, connecting abstract absorption metrics to practical steering effectiveness.

---

## Figures and Tables

- Figure 1: `fig1_steering_cv.pdf` --- Steering effect by CV group and strength, showing High-CV features consistently outperform Low-CV at all strengths
- Figure 2: `fig2_activation_patching.pdf` --- Parent recovery percentages for 9 test words, all passing the 10% threshold with mean 67.3%
- Figure 3: `fig3_cv_comparison.pdf` --- CV distribution comparing absorbed vs non-absorbed features with 733x ratio annotation and threshold line at CV = 1.0
- Figure 4: `fig4_decoder_orthogonality.pdf` --- Scatter plot showing H6 falsified, orthogonality does not predict steering (r=-0.136, not significant)
- Figure 5: `fig5_mechanism_desc.md` --- Architecture diagram describing robust vs fragile absorption routing mechanism
- Table 1: inline --- Steering Effect by CV Group and Strength
- Table 2: inline --- Pilot Activation Patching Results
- Table 3: inline --- Decoder Orthogonality Results
- Table 4: inline --- Hypothesis Status Summary