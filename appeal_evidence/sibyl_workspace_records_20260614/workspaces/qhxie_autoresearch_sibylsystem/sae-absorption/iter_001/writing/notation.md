# Notation Table: Anatomy of Feature Absorption in Sparse Autoencoders

All symbols used in the paper. Every section writer and critic must reference this file for consistency.

---

## 1. Data and Model Inputs

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $x$ | Input activation (residual stream of the base LLM) | $x \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{model}}$ | Base model hidden dimension | scalar (e.g., 2304 for Gemma 2 2B) |
| $\hat{x}$ | SAE reconstruction of $x$ | $\hat{x} \in \mathbb{R}^{d_{\text{model}}}$ |
| $e$ | Reconstruction error: $e = x - \hat{x}$ | $e \in \mathbb{R}^{d_{\text{model}}}$ |

---

## 2. SAE Architecture

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $d_{\text{SAE}}$ | SAE dictionary width (number of latent features) | scalar (e.g., 16384 for "16k" SAE, 65536 for "65k") |
| $W_e \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ | SAE encoder weight matrix | matrix |
| $w_{e,j} \in \mathbb{R}^{d_{\text{model}}}$ | Row $j$ of $W_e$: encoder direction for latent $j$ | $j$-th row of $W_e$ |
| $b_e \in \mathbb{R}^{d_{\text{SAE}}}$ | SAE encoder bias | vector |
| $W_d \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ | SAE decoder weight matrix | matrix |
| $d_j \in \mathbb{R}^{d_{\text{model}}}$ | Column $j$ of $W_d$: decoder direction for latent $j$ | $j$-th column of $W_d$ |
| $z \in \mathbb{R}^{d_{\text{SAE}}}$ | Latent activation vector (sparse) | $z = \text{ReLU}(W_e x + b_e)$ |
| $z_j$ | Activation of latent $j$ | scalar; typically zero (sparse) |
| $L_0$ | Sparsity: number of non-zero latents per forward pass | $L_0 = \|z\|_0$ |

---

## 3. Absorption and EDA

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $\text{EDA}(j)$ | Encoder-Decoder Alignment metric for latent $j$ | scalar $\in [0, 2]$; $\text{EDA}(j) = 1 - \cos(w_{e,j}, d_j)$ |
| $\delta$ | Absorption degree: magnitude of suppression of parent latent | scalar $\geq 0$ |
| $\theta_{jc}$ | Angle between decoder directions $d_j$ and $d_c$ | scalar $\in [0, \pi]$ |
| $c$ | Index of a child latent (more specific feature) | integer $\in \{1, \ldots, d_{\text{SAE}}\}$ |
| $p$ | Index of a parent latent (more general feature) | integer $\in \{1, \ldots, d_{\text{SAE}}\}$ |
| $v_p$ | Probe direction for the parent feature | $v_p \in \mathbb{R}^{d_{\text{model}}}$, unit-normed |

---

## 4. D-EDA (Directional EDA)

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $r_j$ | Residual of $w_{e,j}$ after projecting out $d_j$: $r_j = w_{e,j} - \frac{w_{e,j} \cdot d_j}{\|d_j\|^2} d_j$ | $r_j \in \mathbb{R}^{d_{\text{model}}}$; $r_j \perp d_j$ |
| $\beta \in \mathbb{R}^{d_{\text{SAE}}}$ | Sparse coefficients for decomposing $r_j$ in the decoder dictionary: $r_j \approx \sum_k \beta_k d_k$ | sparse vector; solved via sparse regression |
| $\|\beta\|_0$ | Number of non-zero coefficients in $\beta$ | scalar; low for absorption, high for polysemanticity |
| $S_j$ | Absorbing source set: $S_j = \{k : |\beta_k| \text{ significant} \wedge \cos(d_k, d_j) > 0.1\}$ | set of latent indices |

---

## 5. Three-Subtype Taxonomy

| Symbol / Term | Definition |
|--------------|-----------|
| **Early absorption** | Subtype where $\max_k \cos(d_k, v_p) < \tau$ for all decoder columns; the parent feature was never learned; no encoder correction possible |
| **Late absorption** | Subtype where $\max_k \cos(d_k, v_p) \geq \tau$ but $z_p = 0$ on parent-positive inputs; encoder is suppressed despite decoder presence |
| **Partial absorption** | Subtype where $\max_k \cos(d_k, v_p) \geq \tau$ and $z_p > 0$ on some parent-positive inputs but not all; context-dependent failure |
| $\tau$ | Decoder cosine similarity threshold for subtype classification; primary value $\tau = 0.3$; sensitivity tested at $\{0.20, 0.25, 0.30, 0.35, 0.40\}$ | scalar |

---

## 6. ITAC (Inference-Time Absorption Correction)

| Symbol | Definition | Dimension / Type |
|--------|-----------|-----------------|
| $z_j^{\text{corr}}$ | ITAC-corrected activation for parent latent $j$: $z_j^{\text{corr}} = \max(0, d_j^\top (e + z_{\text{abs}} d_{\text{abs}}))$ | scalar |
| $z_{\text{abs}}$ | Activation of the absorbing child latent (the one responsible for suppression) | scalar |
| $d_{\text{abs}}$ | Decoder direction of the absorbing child latent | $\in \mathbb{R}^{d_{\text{model}}}$ |

---

## 6b. Polysemanticity Stratification

| Symbol / Term | Definition | Dimension / Type |
|--------------|-----------|-----------------|
| $\rho_j$ | Feature density of latent $j$: fraction of tokens on which $z_j > 0$ (SAEBench proxy for polysemanticity) | scalar $\in [0, 1]$ |
| $\bar{\rho}$ | Median feature density across all latents in an SAE; used as split threshold for median-split stratification | scalar |
| AUROC$_{\text{poly}}$ | EDA AUROC restricted to latents with $\rho_j \geq \bar{\rho}$ (polysemantic half) | scalar; e.g., 0.922 at L12-16k |
| AUROC$_{\text{mono}}$ | EDA AUROC restricted to latents with $\rho_j < \bar{\rho}$ (monosemantic half) | scalar; e.g., 0.643 at L12-16k |

---

## 7. Evaluation Metrics

| Symbol | Definition |
|--------|-----------|
| AUROC | Area Under the Receiver Operating Characteristic curve; primary detection performance metric |
| AUPRC | Area Under the Precision-Recall Curve; supplementary metric for class-imbalanced problems |
| FN rate | False negative rate for parent latent: fraction of parent-positive inputs on which $z_p = 0$ |
| FVU | Fraction of Variance Unexplained: $\text{FVU} = \|x - \hat{x}\|^2 / \|x - \bar{x}\|^2$ |
| $\rho_s$ | Spearman rank correlation; used for cross-domain and scaling analyses |
| CI$_{95}$ | 95% bootstrap confidence interval (10,000 resamples) |
| Cohen's $d$ | Effect size for group separation; $d = (\mu_1 - \mu_2) / \sigma_{\text{pooled}}$ |

---

## 8. Dataset and Experimental Identifiers

| Identifier | Meaning |
|-----------|---------|
| L5-16k | Gemma Scope Gemma 2 2B SAE at layer 5, $d_{\text{SAE}} = 16384$ |
| L12-16k | Gemma Scope SAE at layer 12, $d_{\text{SAE}} = 16384$ |
| L12-65k | Gemma Scope SAE at layer 12, $d_{\text{SAE}} = 65536$ |
| L19-16k | Gemma Scope SAE at layer 19, $d_{\text{SAE}} = 16384$ |
| L19-65k | Gemma Scope SAE at layer 19, $d_{\text{SAE}} = 65536$ |
| GPT2-L6 | SAELens GPT-2 Small SAE at layer 6, $d_{\text{SAE}} = 24576$ |
| GPT2-L10 | SAELens GPT-2 Small SAE at layer 10, $d_{\text{SAE}} = 24576$ |
| RAVEL | Relational Attribute Verbalization for Entity probing; dataset `hij/ravel` |
| SynthSAEBench | Synthetic SAE benchmark with known ground-truth absorption labels (arXiv:2602.14687) |

---

## 9. Conventions

- All decoder columns $d_j$ are unit-normalized: $\|d_j\| = 1$.
- All encoder row directions $w_{e,j}$ are not necessarily unit-normalized; when computing $\text{EDA}(j)$, both are explicitly normalized in the formula.
- Cosine similarity: $\cos(u, v) = (u \cdot v) / (\|u\| \|v\|)$.
- The biconvex SDL loss: $\mathcal{L}(\theta) = \mathbb{E}_x [\|x - \hat{x}\|^2 + \lambda \|z\|_1]$ subject to $\|d_j\| = 1 \ \forall j$.
- Bootstrap CIs are reported as $[\text{lower}, \text{upper}]$ with 10,000 resamples and seed = 42.
