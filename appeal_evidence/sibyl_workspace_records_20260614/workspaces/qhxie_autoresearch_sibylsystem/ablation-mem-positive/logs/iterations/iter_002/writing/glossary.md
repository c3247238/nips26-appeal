# Glossary: Phase Transitions in SAE Feature Absorption

## Core Concepts

### Feature Absorption
**Definition**: A phenomenon in Sparse Autoencoders where a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization. The parent latent exhibits systematic false negatives—it fails to activate on inputs containing the parent concept when a child concept is also present.

**Key terms**: Parent feature, child feature, subsumption, false negative, interpretability illusion

### Quasi-Critical Phase Transition
**Definition**: A gradual transition in system behavior at a critical value of a control parameter, exhibiting a susceptibility peak that is broad rather than delta-function sharp. In SAE absorption, chi_ratio = 1.88 (< 3.0 threshold) indicates quasi-critical rather than sharp phase transition behavior.

**Key terms**: Critical point, susceptibility peak, chi_ratio, gradual transition, control parameter

### Susceptibility
**Definition**: The rate of change of absorption with respect to sparsity: $\chi = dm/d\lambda$. Peaks at the critical point, analogous to magnetic susceptibility peaks at phase transitions. Maximum observed: $\chi_{max} = 11.19$ at $\lambda_c = 5 \times 10^{-5}$.

**Key terms**: chi, peak, critical point, phase transition, dm/dlambda

### Finite-Size Scaling
**Definition**: The phenomenon where the sharpness of a phase transition scales with system size. For SAEs, transition width $\delta\lambda \propto N^{-1/\nu}$ where $N$ is dictionary size and $\nu$ is the critical exponent. Best fit: $\nu = 3$, $R^2 = 0.951$.

**Key terms**: Scaling collapse, critical exponent, nu, system size, dictionary size

### Variance Paradox (CV Reversal)
**Definition**: The unexpected observation that absorbed features exhibit coefficient of variation (CV) approximately 733x higher than non-absorbed features (7.33 vs 0.01). This is the opposite of the original H4 prediction and represents a genuine discovery requiring new theoretical explanation.

**Key terms**: CV, coefficient of variation, variance, high-variance features, context-sensitive

---

## Technical Terms

### Sparse Autoencoder (SAE)
**Definition**: An autoencoder with a sparsity penalty (typically L1) that learns a sparse representation of input data. In mechanistic interpretability, SAEs decompose neural network activations into human-interpretable features.

**Key terms**: L1 penalty, latent, dictionary, encoder, decoder, reconstruction

### Decoder Cosine Similarity
**Definition**: A training-free absorption detection metric computed as the cosine similarity between decoder vectors: $cos(d_j, d_k) = \frac{d_j \cdot d_k}{\|d_j\|\|d_k\|}$. High cosine similarity between two features suggests one may absorb the other.

**Key terms**: cosine similarity, decoder vector, d_j, feature direction

### Coefficient of Variation (CV)
**Definition**: The ratio of standard deviation to mean: $CV = \sigma / \mu$. Used to compare variability of absorbed vs non-absorbed feature activations. Absorbed features show CV ≈ 7.33; non-absorbed show CV ≈ 0.01.

**Key terms**: CV, standard deviation, mean, variability, activation magnitude

### Graph Topology
**Definition**: The structure of the absorption graph where nodes are SAE features and edges represent absorption relationships between parent-child feature pairs. Metrics include connected components, giant component size, and mean degree.

**Key terms**: connected components, giant component, degree, edge weight, graph structure

### Information Bottleneck
**Definition**: A theoretical framework stating that absorbed features are "explained away" by child features due to co-occurrence patterns. High co-occurrence causes the network to rely on the child feature, reducing the parent's activation.

**Key terms**: co-occurrence, explained away, information theory, compression

### Actionability Paradox
**Definition**: The observation (Basu et al., 2026) that SAE-based steering achieves 98.2% AUROC feature detection but 0% output change. Near-perfect feature detection does not guarantee steering utility.

**Key terms**: steering, actionability, AUROC, steering effectiveness

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

### Feature-Splitting SAE
**Definition**: An SAE variant where different dictionary sizes are trained on different layers, used in experiments for the dictionary size sweep (H2). Layer 8 feature-splitting SAEs available at d_sae = {6144, 12288, 24576}.

**Key terms**: feature-splitting, layer 8, d_sae, dictionary size

---

## Experimental Terms

### Absorption Rate
**Definition**: The fraction of SAE features classified as absorbed ($\alpha = n_{absorbed}/N$). Values range from 0 to 1, where 1 indicates all features are absorbed at a given sparsity.

**Key terms**: alpha, n_absorbed, fraction, absorption_rate

### Critical Lambda ($\lambda_c$)
**Definition**: The sparsity penalty value at which absorption exhibits a quasi-critical onset. Identified as $\lambda_c = 5 \times 10^{-5}$ in the sparsity sweep experiments, with susceptibility peak $\chi_{max} = 11.19$.

**Key terms**: lambda_c, critical value, threshold, sparsity

### Scaling Collapse
**Definition**: The observation that absorption curves for different dictionary sizes overlay when the sparsity axis is rescaled by $N^{-1/\nu}$. Best collapse achieved with $\nu = 3$, $R^2 = 0.951$.

**Key terms**: collapse, rescaling, universal curve, nu

### Chi Ratio
**Definition**: The ratio of peak susceptibility to average susceptibility: $\chi_{ratio} = \chi_{max} / \bar{\chi}$. Values > 3 indicate sharp transition; $\chi_{ratio} = 1.88$ indicates quasi-critical (gradual) transition.

**Key terms**: chi_ratio, sharpness, transition type

### Training-Free Detection
**Definition**: Methods for detecting absorption that do not require training probes or ablating model components. Examples include decoder cosine similarity and the $A_j$ metric.

**Key terms**: training-free, no-probe, detector, A_j

---

## Statistical Terms

### Pearson Correlation ($r$)
**Definition**: A measure of linear correlation between two variables, ranging from -1 to 1. Used to validate the co-occurrence formula: revised $r = 0.647$ vs baseline $r = -0.52$.

**Key terms**: r, correlation, linear relationship, Pearson

### p-value
**Definition**: The probability of observing a result as extreme as the data under a null hypothesis. $p < 0.01$ indicates statistical significance. Co-occurrence revised formula: $p \approx 10^{-261}$.

**Key terms**: p-value, significance, hypothesis test, null hypothesis

### t-statistic
**Definition**: The ratio of the difference between two means to their combined standard error. Used in CV comparison t-tests. All layers show $|t| > 1000$ for CV difference.

**Key terms**: t-statistic, t-test, mean comparison, standard error

### R-squared ($R^2$)
**Definition**: The proportion of variance in the dependent variable explained by the model. Used to assess scaling collapse quality: $R^2 = 0.951$ for $\nu = 3$.

**Key terms**: R^2, coefficient of determination, fit quality

---

## Preferred Phrasings

| Use This | Instead of |
|----------|-----------|
| "feature absorption" | "absorption phenomenon" |
| "quasi-critical behavior" | "sharp transition" (when chi_ratio < 3) |
| "variance paradox" or "CV reversal" | "failed H4" |
| "layer saturation" | "layer heterogeneity" (for saturated case) |
| "critical exponent" | "scaling exponent" |
| "finite-size scaling" | "size-dependent scaling" |
| "giant component" | "largest component" |
| "sparsity penalty" | "L1 regularization" (when referring to SAE context) |
| "dictionary size" | "latent count", "number of neurons" |
| "decoder vector" | "decoder direction", "feature direction" |
| "absorption graph" | "feature graph" |
| "actionability paradox" | "steering failure" |
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
| LSTM | Long Short-Term Memory | Not used in this paper |

---

## Notation Style Notes

- Use italic for mathematical symbols: $m$, $\lambda$, $\chi$
- Use upright for text: SAE, GPT-2
- Subscripts and superscripts should be clear: $\lambda_c$ (critical lambda), $N^{-1/\nu}$ (inverse power)
- Vector notation: $\mathbb{R}^d$ for d-dimensional real vectors
- Set notation: $\in$ for element membership
- Greek letters preferred for physical quantities: $\lambda$ (lambda), $\chi$ (chi), $\nu$ (nu)
- Scientific notation for small numbers: $5 \times 10^{-5}$

---

## Figure/Table Reference Style

References should appear BEFORE the figure/table in the text:
- "Figure 1 shows the quasi-critical phase transition diagram..."
- "Table 1 summarizes the hypothesis test results..."
- "As shown in Equation 1, the susceptibility is..."
- "The critical point at $\lambda_c = 5 \times 10^{-5}$ (Figure 1)"
- "The variance paradox (CV_reversal) is illustrated in Figure 3"

---

## Key References

- Chanin et al. (2024): A is for Absorption (detection metric, A_j)
- Basu et al. (2026): Interpretability without Actionability (actionability paradox)
- Cui et al. (2026): On the Limits of SAEs (theoretical limits)
- Karvonen et al. (2025): SAEBench (probe projection metric)
- Costa et al. (2025): MP-SAE (hierarchical feature recovery)
- Pearl (2009): Causality (causal mediation framework)
