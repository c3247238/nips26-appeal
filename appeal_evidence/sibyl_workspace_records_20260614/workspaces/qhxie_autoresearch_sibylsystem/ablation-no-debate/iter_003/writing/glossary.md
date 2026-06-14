# Glossary: Encoder-Driven Feature Absorption in SAEs

This document unifies key terminology definitions used throughout the paper. All section writers must reference this file for consistency.

---

## 1. Core SAE Terminology

### 1.1 Sparse Autoencoder (SAE)
- **Definition**: A neural network trained to learn sparse, interpretable representations by reconstructing its input through a bottleneck layer with sparsity constraints
- **Preferred phrasing**: "Sparse Autoencoder" or "SAE" (not "sparse auto-encoder" or "sparse encoder")
- **Abbreviation**: SAE (plural: SAEs)
- **Notes**: In this paper, "SAE" refers specifically to toy/synthetic SAEs trained on synthetic hierarchies OR pretrained SAEs from SAELens (e.g., Gemma Scope)

### 1.2 Feature
- **Definition**: A distinct, interpretable pattern of neuron activations that corresponds to a semantically meaningful concept (e.g., "a feature detecting political statements")
- **Preferred phrasing**: "feature" (not "neuron", "unit", or "latent")
- **Notes**: Features are learned representations, not raw neurons; features are identified by index (e.g., "feature 1024")

### 1.3 Latent Representation / Latent Code
- **Definition**: The intermediate sparse representation z produced by the SAE encoder
- **Preferred phrasing**: "latent representation" or "latent code" (not "hidden representation" or "embedding")

### 1.4 Feature Hierarchy
- **Definition**: A structured relationship where child features (Level 1) and grandchild features (Level 2) are derived from or related to a parent feature (Level 0)
- **Preferred phrasing**: "feature hierarchy" or "hierarchical structure" (not "feature tree" or "neuron hierarchy")
- **In this paper**: 3-level synthetic hierarchies with tunable parent-child cosine similarity and stochastic noise (epsilon ~ N(0, 0.05))

---

## 2. Absorption Terminology

### 2.1 Feature Absorption
- **Definition**: A phenomenon where child features in an SAE substitute for parent features in sparse representations, causing the parent feature's activation to decrease when its child features are activated
- **Preferred phrasing**: "feature absorption" or "absorption" (not "feature substitution" or "parent-child absorption")
- **Key nuance**: Absorption can exceed 1.0 (indicating the parent was being suppressed by children; ablating children releases the parent's true activation)

### 2.2 Multi-Child Proportional Absorption
- **Definition**: A measurement method where absorption rate is computed as the ratio of parent activation after ablating k child features to the original parent activation
- **Preferred phrasing**: "multi-child proportional absorption" or "multi-child absorption rate" (not "proportional absorption" alone)
- **Formula**: $A_{multi}(p) = a_p^{after} / a_p^{before}$
- **Default k**: 5 child features

### 2.3 Absorption Rate
- **Definition**: A scalar value representing the proportion of parent activation explained by child feature substitution; can exceed 1.0 when child ablation releases suppressed parent activation
- **Preferred phrasing**: "absorption rate" (not "absorption level" or "absorption coefficient")
- **Range**: $[0, \infty)$ -- values > 1.0 indicate net suppression

### 2.4 Encoder-Driven Absorption
- **Definition**: Absorption that arises specifically from the encoder's learned alignment with hierarchical structure in the training data, as opposed to decoder geometry
- **Preferred phrasing**: "encoder-driven absorption" (not "encoder-induced absorption")
- **Key finding**: The encoder's learned alignment is a primary driver, but decoder contribution is configuration-dependent (not uniformly zero as pilot suggested)

### 2.5 Child Feature / Parent Feature
- **Definition**: In a feature hierarchy, the parent feature is the higher-level concept (Level 0); child features are subordinate features (Level 1) that may absorb the parent
- **Preferred phrasing**: "parent feature" and "child feature" (not "parent neuron" or "child neuron")

---

## 3. Experimental Design Terms

### 3.1 2x2 Factorial Design
- **Definition**: An experimental design crossing all levels of two factors (encoder: random/trained; decoder: random/trained) to detect interactions and isolate main effects
- **Conditions**: A (random/random), B (trained/random), C (random/trained), D (trained/trained)
- **Preferred phrasing**: "2x2 factorial design" or "factorial experiment" (not "full factorial" alone)
- **Interpretation**:
  - B ≈ D: Encoder sufficient (encoder-driven mechanism confirmed)
  - C ≈ A: Decoder irrelevant (decoder-geometry hypothesis rejected)
  - Full experiment result: B ≈ D holds (delta = 0.037); C ≈ A FAILS (delta = 12.10, extreme Condition C variance)

### 3.2 Stochastic Hierarchy
- **Definition**: A feature hierarchy generated with controlled random noise ($\epsilon \sim \mathcal{N}(0, 0.05)$) to test robustness across random seeds
- **Preferred phrasing**: "stochastic hierarchy" (not "random hierarchy" or "noisy hierarchy")
- **In this paper**: 3-level synthetic hierarchies; used to expose decoder contributions masked by deterministic hierarchies in pilots

### 3.3 Hierarchy Strength
- **Definition**: The cosine similarity between parent and child feature directions; higher values indicate stronger hierarchical structure
- **Preferred phrasing**: "hierarchy strength" (not "hierarchy overlap" or "parent-child similarity")
- **Sweep values**: {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}

### 3.4 Pareto Frontier / Pareto Front
- **Definition**: The set of optimal points where improving one objective (e.g., sensitivity) necessarily degrades another (e.g., absorption), representing non-dominated solutions
- **Preferred phrasing**: "Pareto frontier" (American English, not "Pareto front" which is British)
- **In this paper**: The sensitivity-absorption trade-off; H_Pareto attempted to quantify this but found degenerate results (absorption = 0 across all L0 levels)

---

## 4. Safety-Critical Feature Terminology

### 4.1 Safety-Critical Feature
- **Definition**: A feature that, if unreliable due to absorption, could lead to incorrect interpretability-based safety assessments (e.g., features related to deception, jailbreaking, harm)
- **Preferred phrasing**: "safety-critical feature" or "safety-relevant feature" (not "dangerous feature" or "critical feature" alone)
- **In this paper**: Features annotated in Neuronpedia as related to deception, jailbreak, harm, manipulation

### 4.2 Safety Assessment Reliability
- **Definition**: The extent to which SAE-based interpretability can be trusted for safety-critical decisions
- **Preferred phrasing**: "safety assessment reliability" (not "interpretability safety")

---

## 5. Evaluation Metrics

### 5.1 Feature Sensitivity
- **Definition**: A measure (from Hu et al., 2025) of how much a feature's output changes in response to steering interventions, computed as the variance of steering coefficients
- **Preferred phrasing**: "feature sensitivity" or "sensitivity" (not "steering sensitivity" or "activation sensitivity")
- **Reference**: Hu et al., 2025 (arXiv:2509.23717)
- **In this paper**: Sensitivity was measured for H_Pareto but was stable (~0.1054) across all L0 levels; absorption collapsed to zero

### 5.2 Mann-Whitney U Test
- **Definition**: A non-parametric statistical test comparing whether one distribution is stochastically greater than another
- **Preferred phrasing**: "Mann-Whitney U test" or "Mann-Whitney U" (not "Mann-Whitney-Wilcoxon" unless formally referencing both authors)
- **Threshold used**: p < 0.05 for H_Safe

### 5.3 Spearman Correlation
- **Definition**: A rank-based correlation coefficient measuring monotonic relationships
- **Preferred phrasing**: "Spearman correlation" or "Spearman's $\rho$" (not "Spearman rank correlation")

### 5.4 R-squared / Coefficient of Determination
- **Definition**: The proportion of variance in the dependent variable explained by the independent variable(s)
- **Preferred phrasing**: "$R^2$" or "$R$-squared" (not "R2" or "R squared")
- **Threshold for H_Comp**: R² > 0.8 for monotonic fit

---

## 6. Model and Dataset Names

### 6.1 Gemma Scope
- **Definition**: A suite of pretrained SAEs trained on Gemma 2B models, released by SAELens
- **Preferred phrasing**: "Gemma Scope" (not "GemmaScope" or "gemma-scope")
- **Specific variant used**: gemma-2b-res (Gemma 2B residual stream SAE, layer 12, d=16384)

### 6.2 SAELens
- **Definition**: An open-source library for working with sparse autoencoders and analyzing neural network features
- **Preferred phrasing**: "SAELens" (not "SAE-Lens", "saelens", or "SAE lens")

### 6.3 GPT-2 Small
- **Definition**: The 124M parameter version of the GPT-2 language model
- **Preferred phrasing**: "GPT-2 Small" (not "gpt2" or "GPT2")
- **SAE used**: blocks.8.hook_resid_pre (residual stream SAE)

### 6.4 Neuronpedia
- **Definition**: An online platform for exploring and annotating SAE features
- **Preferred phrasing**: "Neuronpedia" (not "neuronpedia")

---

## 7. Domain Terms

### 7.1 Mechanistic Interpretability
- **Definition**: A field of research focusing on understanding the internal mechanisms of neural networks by identifying causal relationships between features and model behavior
- **Preferred phrasing**: "mechanistic interpretability" (not "mech interp" or "MI")

### 7.2 Steering / Steering Interventions
- **Definition**: A technique where feature activations are artificially modified (via addition or ablation) to test feature causality
- **Preferred phrasing**: "steering" or "steering intervention" (not "activation steering" or "feature steering")

### 7.3 Ablation
- **Definition**: The deliberate removal or suppression of a feature's activation to measure its contribution
- **Preferred phrasing**: "ablation" (not "knockout" or "suppression")
- **Multi-child ablation**: Ablating the top k child features to measure parent absorption

### 7.4 Information Geometry
- **Definition**: A mathematical framework studying the geometric properties of probability distributions and their manifolds; used here to theoretically characterize absorption dynamics
- **Preferred phrasing**: "information geometry" (not "geometric analysis" or "probability geometry")

---

## 8. Abbreviations

| Abbreviation | Expansion |
|--------------|------------|
| SAE | Sparse Autoencoder (plural: SAEs) |
| L0 | L0 norm (count of non-zero elements in latent code) |
| MI | Mechanistic Interpretability |
| Gemma 2B | Gemma 2 Billion Parameter Model |
| SAELens | SAE Lens (library name, not acronym) |
| Neuronpedia | Online feature exploration platform |

---

## 9. Formatting Conventions

- **SAE features**: When referring to specific feature indices, use: "feature 1024" (not "feature[1024]")
- **Hypothesis names**: Use H_Mech, H_Comp, H_Pareto, H_Safe format
- **Experiment conditions**: Use Condition A/B/C/D format
- **Statistical notation**: Use proper mathematical symbols ($\rho$, $R^2$, $p$) not their spelled-out names
- **Absorption rate values**: Report raw values with appropriate precision; values > 1.0 are valid and indicate net suppression

---

## 10. Experimental Results Summary (Authoritative)

### 10.1 H_Mech (Encoder-Driven Mechanism)
- **Status**: CONDITIONALLY CONFIRMED
- **Full result** (5 seeds × 4 conditions × 100 samples):
  - B ≈ D: delta = 0.037 (**passes** -- encoder sufficient confirmed)
  - C ≈ A: delta = 12.10 (**FAILS** -- decoder contribution is non-zero in some configurations)
  - Condition C has extreme variance (std = 17.13, range 0-43.84)
- **Cross-model validation** (GPT-2 Small held-out): encoder-driven check TRUE, delta B-D = 0.0, delta C-A = 0.0
- **Interpretation**: The encoder-driven mechanism is conditionally confirmed; the decoder's role is configuration-dependent. Prior work attributing absorption to decoder geometry is not wrong but incomplete.

### 10.2 H_Comp (Hierarchy Strength Dependence)
- **Status**: FAILED
- **Full result** (3 seeds × 6 levels × 100 samples): R² = 0.04 (target > 0.8)
- **Regression**: slope = -0.296, p = 0.703 (not significant)
- **Absorption range**: 0.51 to 1.20 across cosine levels
- **Interpretation**: No monotonic relationship between hierarchy strength and absorption; the phase-transition framing is not supported.

### 10.3 H_Pareto (Sensitivity-Absorption Frontier)
- **Status**: INCONCLUSIVE (degenerate result)
- **Full result** (3 seeds × 4 L0 levels × 100 samples):
  - **absorption_mean = 0.0** (std = 0.08) across all L0 levels
  - sensitivity_mean = 0.1054 (std = 0.0008) stable across L0 levels
  - Frontier fit degenerates to a = 1.0, b = -0.5
- **Interpretation**: No Pareto frontier detected; the theoretical prediction of a sensitivity-absorption trade-off is not supported in synthetic SAEs.

### 10.4 H_Safe (Safety-Critical Feature Absorption)
- **Status**: NULL RESULT (positive for safety analysis)
- **Gemma Scope pilot** (5 per group, 100 samples): p = 1.0; both groups at 0.0 absorption
- **GPT-2 Small held-out** (20 per group, 100 samples): p = 0.345; safety_mean = 233.13, non_safety_mean = 221.70
- **Interpretation**: Safety-critical features do NOT show elevated absorption in either Gemma Scope SAEs or GPT-2 Small. This is a negative result for the hypothesis but a positive finding for SAE-based safety analysis -- safety features appear robust to absorption.
