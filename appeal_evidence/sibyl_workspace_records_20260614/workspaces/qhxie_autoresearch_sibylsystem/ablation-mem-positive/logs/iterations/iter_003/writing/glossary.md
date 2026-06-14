# Glossary: CV-Based Actionability Decomposition in Absorbed SAE Features

## Core Concepts

### Feature Absorption
**Definition**: A phenomenon in Sparse Autoencoders where a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization. The parent latent exhibits systematic false negatives—it fails to activate on inputs containing the parent concept when a child concept is also present.

**Key terms**: Parent feature, child feature, subsumption, false negative, interpretability illusion

### Actionability Paradox
**Definition**: The observation (Basu et al., 2026) that SAE-based steering achieves 98.2% AUROC feature detection but 0% output change. Near-perfect feature detection does not guarantee steering utility.

**Key terms**: steering, actionability, AUROC, steering effectiveness

### Robust Absorbed Features
**Definition**: Absorbed features with high coefficient of variation (CV > 1.0) that route through context-sensitive child channels and retain steering potential. These features are steerable despite being absorbed.

**Key terms**: high-CV, context-sensitive, steerable, intervention target

### Fragile Absorbed Features
**Definition**: Absorbed features with low coefficient of variation (CV <= 1.0) that route through stable child channels and do not retain steering potential. Steering the parent activates the child, but the child's contribution is fixed (bypass routing).

**Key terms**: low-CV, stable routing, non-steerable, bypass

### Variance Paradox
**Definition**: The unexpected observation that absorbed features exhibit coefficient of variation (CV) approximately 733x higher than non-absorbed features (7.33 vs 0.01). This indicates that absorption selectively preserves context-sensitive specialized information.

**Key terms**: CV, coefficient of variation, variance, high-variance features, context-sensitive

### CV-Based Decomposition
**Definition**: Our methodological contribution—classifying absorbed features into robust (high-CV) and fragile (low-CV) subpopulations based on coefficient of variation, enabling prediction of which absorbed features remain steerable.

**Key terms**: decomposition, CV classification, actionability prediction

---

## Technical Terms

### Sparse Autoencoder (SAE)
**Definition**: An autoencoder with a sparsity penalty (typically L1) that learns a sparse representation of input data. In mechanistic interpretability, SAEs decompose neural network activations into human-interpretable features.

**Key terms**: L1 penalty, latent, dictionary, encoder, decoder, reconstruction

### Coefficient of Variation (CV)
**Definition**: The ratio of standard deviation to mean: $CV = \sigma / \mu$. Used to measure activation variability across contexts for each SAE feature. Absorbed features show CV ≈ 7.33; non-absorbed show CV ≈ 0.01.

**Key terms**: CV, standard deviation, mean, variability, activation magnitude

### Decoder Cosine Similarity
**Definition**: A training-free absorption detection metric computed as the cosine similarity between decoder vectors: $cos(d_j, d_k) = \frac{d_j \cdot d_k}{\|d_j\|\|d_k\|}$. High cosine similarity between two features suggests one may absorb the other.

**Key terms**: cosine similarity, decoder vector, d_j, feature direction

### Training-Free Detection
**Definition**: Methods for detecting absorption that do not require training probes or ablating model components. Examples include decoder cosine similarity and the $A_j$ metric.

**Key terms**: training-free, no-probe, detector, A_j

### Steering
**Definition**: An intervention technique where SAE latents are directly manipulated (typically via addition or zeroing) to test their causal effect on model outputs. Steering effectiveness is measured by logit change at target tokens.

**Key terms**: intervention, steering strength, logit change, target token

### Activation Patching
**Definition**: An ablation technique where a feature's activation is zeroed (patched) to measure its causal contribution. Used to validate that absorbed features have genuine causal effects by measuring parent recovery when child is zeroed.

**Key terms**: ablation, zeroing, recovery, causal effect

### Context-Sensitive Routing
**Definition**: A routing mechanism where high-CV (robust absorbed) features activate strongly in specific contexts but weakly in others. This variability allows steering to modulate behavior effectively.

**Key terms**: variability, context-dependent, specialized activation

### Bypass Routing
**Definition**: A routing mechanism where low-CV (fragile absorbed) features route through stable child channels that compensate for parent steering. The child's contribution is fixed regardless of parent steering, creating zero net effect.

**Key terms**: stable routing, compensation, fixed contribution

---

## Architectural Terms

### GPT-2 Small
**Definition**: A 117M parameter GPT model used in experiments. Layer 6 (blocks.6.hook_resid_pre) was identified as an absorption hotspot in prior experiments.

**Key terms**: gpt2-small, residual stream, hook_name, blocks.6.hook_resid_pre

### SAELens
**Definition**: A library providing pretrained Sparse Autoencoders for various models including GPT-2 and Gemma.

**Key terms**: saelens, gpt2-small-res-jb, feature-splitting, pretrained SAE

### JumpReLU
**Definition**: An SAE architectural variant that applies a non-zero threshold before the ReLU activation, improving reconstruction fidelity compared to standard ReLU SAEs.

**Key terms**: JumpReLU, threshold, activation function, architectural variant

### GemmaScope
**Definition**: Google's SAE release for Gemma models, providing JumpReLU SAEs at multiple layers.

**Key terms**: GemmaScope, Gemma-2, JumpReLU, layer 6

---

## Experimental Terms

### Absorption Rate
**Definition**: The fraction of SAE features classified as absorbed ($\alpha = n_{absorbed}/N$). Values range from 0 to 1, where 1 indicates all features are absorbed at a given sparsity.

**Key terms**: alpha, n_absorbed, fraction, absorption_rate

### Critical Lambda ($\lambda_c$)
**Definition**: The sparsity penalty value at which absorption exhibits a quasi-critical onset (supporting theoretical context). Identified as $\lambda_c = 5 \times 10^{-5}$, with susceptibility peak $\chi_{max} = 11.19$.

**Key terms**: lambda_c, critical value, threshold, sparsity

### Chi Ratio
**Definition**: The ratio of peak susceptibility to average susceptibility: $\chi_{ratio} = \chi_{max} / \bar{\chi}$. Values > 3 indicate sharp transition; $\chi_{ratio} = 1.88$ indicates quasi-critical (gradual) transition.

**Key terms**: chi_ratio, sharpness, transition type, susceptibility

### Steering Strength ($\tau$)
**Definition**: The magnitude of intervention applied to SAE latents. In experiments, steering strengths of +3, +5, +7 are applied to measure dose-response relationships.

**Key terms**: tau, steering magnitude, intervention strength

### Logit Change ($\Delta_{logit}$)
**Definition**: The change in output logits at a target token resulting from steering intervention. Primary metric for steering effectiveness.

**Key terms**: delta_logit, output change, steering effect

---

## Statistical Terms

### Pearson Correlation ($r$)
**Definition**: A measure of linear correlation between two variables, ranging from -1 to 1. Used to validate CV-steering correlation.

**Key terms**: r, correlation, linear relationship, Pearson

### p-value
**Definition**: The probability of observing a result as extreme as the data under a null hypothesis. $p < 0.01$ indicates statistical significance with BH correction.

**Key terms**: p-value, significance, hypothesis test, null hypothesis

### Welch's t-test
**Definition**: A t-test that does not assume equal variances between groups. Used to compare steering effects between high-CV and low-CV feature groups.

**Key terms**: Welch's t-test, t-statistic, mean comparison

### Benjamini-Hochberg Correction
**Definition**: A method for correcting p-values for multiple comparisons to control false discovery rate. Applied across the three steering strengths tested.

**Key terms**: BH correction, FDR, multiple comparisons

### Coefficient of Variation (CV)
**Definition**: The ratio of standard deviation to mean: $CV = \sigma / \mu$. A normalized measure of dispersion that allows comparison of variability across features with different mean activations.

**Key terms**: CV, standard deviation, mean, normalized dispersion

---

## Preferred Phrasings

| Use This | Instead of |
|----------|-----------|
| "actionability paradox" | "steering failure" (for Basu et al.) |
| "robust absorbed features" | "absorbing parent" |
| "fragile absorbed features" | "non-steerable feature" |
| "variance paradox" | "CV reversal" |
| "context-sensitive routing" | "variable activation path" |
| "bypass routing" | "stable path" |
| "CV-based decomposition" | "CV classification" |
| "steering effect" | "steering utility" |
| "feature absorption" | "absorption phenomenon" |
| "giant component" | "largest component" |
| "sparsity penalty" | "L1 regularization" (in SAE context) |
| "dictionary size" | "latent count", "number of features" |
| "decoder vector" | "decoder direction", "feature direction" |
| "absorption graph" | "feature graph" |
| "cross-layer" | "inter-layer" |
| "critical point" | "threshold point" |

---

## Domain-Specific Abbreviations

| Abbreviation | Full Term | First Use |
|--------------|-----------|-----------|
| SAE | Sparse Autoencoder | First paragraph of Introduction |
| CV | Coefficient of Variation | Define on first use |
| ReLU | Rectified Linear Unit | Define on first use |
| GPT | Generative Pre-Trained Transformer | Define on first use |
| LLM | Large Language Model | Define on first use |
| GemmaScope | Gemma Scope (SAE release) | Define on first use |
| SAELens | Sparse Autoencoder Lens | Define on first use |
| AUROC | Area Under ROC Curve | Define on first use |
| BH | Benjamini-Hochberg | Define on first use |

---

## Notation Style Notes

- Use italic for mathematical symbols: $m$, $\lambda$, $\chi$
- Use upright for text: SAE, GPT-2
- Subscripts and superscripts should be clear: $\lambda_c$ (critical lambda), $CV_{absorbed}$
- Vector notation: $\mathbb{R}^d$ for d-dimensional real vectors
- Set notation: $\in$ for element membership
- Greek letters preferred for physical quantities: $\lambda$ (lambda), $\chi$ (chi), $\nu$ (nu), $\tau$ (tau)
- Scientific notation for small numbers: $5 \times 10^{-5}$
- CV formula: $CV = \sigma / \mu$

---

## Figure/Table Reference Style

References should appear BEFORE the figure/table in the text:
- "Figure 1 shows the steering effect comparison by CV group..."
- "Table 1 summarizes the hypothesis test results..."
- "As shown in Equation 1, the CV is computed as..."
- "The high-CV group shows 2x larger effect (Table 1)"

---

## Key References

- Chanin et al. (2024): A is for Absorption (detection metric, A_j)
- Basu et al. (2026): Interpretability without Actionability (actionability paradox)
- Cui et al. (2026): On the Limits of SAEs (theoretical limits)
- Karvonen et al. (2025): SAEBench (probe projection metric)
- Costa et al. (2025): MP-SAE (hierarchical feature recovery)
- Pearl (2009): Causality (causal mediation framework)
