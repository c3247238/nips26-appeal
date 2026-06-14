# Notation Table: Beyond Competitive Exclusion

All symbols used in the paper. Every section writer and critic must reference this file for consistency.

---

## 1. Data and Model Inputs

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $x$ | Input activation (residual stream of the base LLM at a specific layer) | $x \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{model}}$ | Base model hidden dimension | scalar (2304 for Gemma 2 2B; 768 for GPT-2 Small) |
| $\hat{x}$ | SAE reconstruction of $x$ | $\hat{x} \in \mathbb{R}^{d_{\text{model}}}$ |
| $e$ | Reconstruction error: $e = x - \hat{x}$ | $e \in \mathbb{R}^{d_{\text{model}}}$ |

---

## 2. SAE Architecture

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $d_{\text{SAE}}$ | SAE dictionary width (number of latent features) | scalar (16384 for "16k", 65536 for "65k") |
| $W_e \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ | SAE encoder weight matrix | matrix |
| $w_{e,j} \in \mathbb{R}^{d_{\text{model}}}$ | Row $j$ of $W_e$: encoder direction for latent $j$ | vector |
| $b_e \in \mathbb{R}^{d_{\text{SAE}}}$ | SAE encoder bias | vector |
| $W_d \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ | SAE decoder weight matrix | matrix |
| $d_j \in \mathbb{R}^{d_{\text{model}}}$ | Column $j$ of $W_d$: decoder direction for latent $j$ (unit-normalized: $\|d_j\| = 1$) | vector |
| $z \in \mathbb{R}^{d_{\text{SAE}}}$ | Latent activation vector (sparse) | vector |
| $z_j$ | Activation of latent $j$ | scalar; typically zero (sparse) |
| $L_0$ | Sparsity level: number of non-zero latents per forward pass; $L_0 = \|z\|_0$ | scalar |
| $\theta$ | JumpReLU threshold parameter (Gemma Scope SAEs); $z_j = (w_{e,j}^\top x + b_{e,j}) \cdot \mathbb{1}[w_{e,j}^\top x + b_{e,j} > \theta_j]$ | per-latent scalar |

---

## 3. Absorption Measurement

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $p$ | Index of a parent latent (more general feature, e.g., "starts-with-A words") | integer $\in \{1, \ldots, d_{\text{SAE}}\}$ |
| $c$ | Index of a child latent (more specific feature, e.g., "starts-with-A proper nouns") | integer $\in \{1, \ldots, d_{\text{SAE}}\}$ |
| $v_p \in \mathbb{R}^{d_{\text{model}}}$ | Probe direction for parent feature (trained via logistic regression on SAE latents) | unit-normed vector |
| FN rate | False negative rate for parent latent: fraction of parent-positive inputs on which $z_p = 0$ | scalar $\in [0, 1]$ |
| $\alpha$ | Absorption rate: fraction of parent-positive inputs where all k probe-associated latents fail to activate despite correct probe classification | scalar $\in [0, 1]$ |
| $\tau_{\cos}$ | Cosine similarity threshold for main feature identification: $\cos(d_j, v_p) \geq \tau_{\cos}$ | scalar (default 0.025) |
| $\tau_{\text{mag}}$ | Magnitude gap threshold for absorption criterion | scalar (default 1.0) |

---

## 4. Confound Decomposition

| Symbol / Term | Definition |
|--------------|-----------|
| **Hedging** | False-negative mechanism where the parent feature's information is spread across many latents, none clearing the activation threshold; resolves at higher L0 values as sparsity relaxes |
| **Hierarchy-driven absorption** | False-negative mechanism where the child latent actively suppresses the parent; persists across all tested L0 values regardless of sparsity pressure |
| **Reconstruction error** | False-negative mechanism where the SAE fails to reconstruct the parent direction entirely |
| $f_{\text{hedge}}(L_0)$ | Fraction of false negatives classified as hedging at a given $L_0$ | scalar $\in [0, 1]$ |
| $f_{\text{hier}}(L_0)$ | Fraction of false negatives classified as hierarchy-driven at a given $L_0$ | scalar $\in [0, 1]$ |

---

## 5. Rate-Distortion and Information Theory

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $I(X; f_p \mid f_c)$ | Conditional mutual information: unique information the parent feature carries beyond the child | scalar $\geq 0$ (nats) |
| CMI | Shorthand for $I(X; f_p \mid f_c)$ | scalar |
| $d'$ | Dimensionality of the decoder subspace used for CMI estimation | scalar (primary: $d' = 10$) |
| $c(w_P, w_C)$ | Geometric constant: $c = \|w_P\|^2 (1 - \cos^2(w_P, w_C))$ | scalar $\in [0, \|w_P\|^2]$ |
| $\lambda$ | Effective sparsity pressure parameter inferred from the L0-absorption relationship | scalar |
| $L_{0,\text{crit}}$ | Predicted critical L0 from rate-distortion theory: $L_{0,\text{crit}} = \lambda / (\text{CMI} \cdot c)$ | scalar |

---

## 6. SAE Architecture Variants

| Term | Definition |
|------|-----------|
| **JumpReLU SAE** | SAE with hard threshold activation: $z_j = \text{pre}(x)_j \cdot \mathbb{1}[\text{pre}(x)_j > \theta_j]$; used by Gemma Scope (Rajamanoharan et al., 2024) |
| **L1-ReLU SAE** | SAE with soft L1 penalty: $z_j = \text{ReLU}(\text{pre}(x)_j)$; standard architecture in GPT-2 SAELens |
| **L0 operating point** | The specific L0 value at which a JumpReLU SAE is configured to operate; Gemma Scope provides multiple L0 settings per layer/width |

---

## 7. Evaluation Metrics

| Symbol | Definition |
|--------|-----------|
| AUROC | Area Under the Receiver Operating Characteristic curve |
| F1 | Harmonic mean of precision and recall for probe classification |
| $\rho_s$ | Spearman rank correlation |
| CI$_{95}$ | 95% bootstrap confidence interval (10,000 resamples, seed = 42) |
| Cohen's $d$ | Effect size: $d = (\mu_1 - \mu_2) / \sigma_{\text{pooled}}$ |
| BC | Bimodality coefficient: $\text{BC} = (\gamma^2 + 1) / \kappa$ where $\gamma$ = skewness, $\kappa$ = kurtosis; BC > 0.555 suggests bimodality |
| CV | Coefficient of variation: $\text{CV} = \sigma / \mu$ |
| MCC | Matthews Correlation Coefficient |
| GAM | Generalized Additive Model (for scaling surface analysis) |
| OLS | Ordinary Least Squares regression |

---

## 8. Dataset and Experimental Identifiers

| Identifier | Meaning |
|-----------|---------|
| L12-16k-L0=22 | Gemma Scope SAE at layer 12, $d_{\text{SAE}} = 16384$, L0 operating point = 22 |
| L12-16k-L0=82 | Gemma Scope SAE at layer 12, $d_{\text{SAE}} = 16384$, L0 = 82 (primary config for cross-domain) |
| L12-65k | Gemma Scope SAE at layer 12, $d_{\text{SAE}} = 65536$ |
| L10-16k | Gemma Scope SAE at layer 10, $d_{\text{SAE}} = 16384$ |
| L20-16k | Gemma Scope SAE at layer 20, $d_{\text{SAE}} = 16384$ |
| GPT2-L8/L10/L11 | SAELens GPT-2 Small SAE at layers 8, 10, 11 ($d_{\text{SAE}} = 24576$, L1-ReLU) |
| RAVEL | Relational Attribute Verbalization for Entity probing dataset (`hij/ravel` on HuggingFace) |

---

## 9. Control Suite

| Control | Definition | Expected Outcome |
|---------|-----------|-----------------|
| C1: Random probe | Probe direction drawn from random unit sphere | Absorption rate < 2% |
| C2: Shuffled labels | Probe trained with randomly shuffled parent assignments | Absorption rate < measured rate |
| C3: Dense probe | Logistic regression on raw model activations (not SAE latents) | Ceiling for probe quality |
| C4: Untrained SAE | SAE with randomly initialized encoder/decoder of the same dimensions | Absorption rate = 0% |

---

## 10. Conventions

- All decoder columns $d_j$ are unit-normalized: $\|d_j\| = 1$ (Gemma Scope convention).
- Cosine similarity: $\cos(u, v) = (u \cdot v) / (\|u\| \|v\|)$.
- Bootstrap CIs are reported as $[\text{lower}, \text{upper}]$ with 10,000 resamples and seed = 42 throughout.
- L0 is reported as the configured operating point (not measured per-sample).
- Absorption rates are reported as percentages (e.g., 15.96%) unless otherwise noted.
- All p-values are two-sided unless explicitly stated as one-sided.
- Multiple comparison correction: Bonferroni applied where noted; uncorrected p-values also reported.
