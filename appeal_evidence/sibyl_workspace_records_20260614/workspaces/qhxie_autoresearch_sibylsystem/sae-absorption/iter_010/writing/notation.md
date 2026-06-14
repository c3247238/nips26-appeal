# Notation Table

All mathematical symbols and notation used in the paper. Section writers, critics, and the editor must reference this file for consistency.

## Model and Architecture

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{x}$ | Input activation vector (residual stream at layer $\ell$) | $\mathbf{x} \in \mathbb{R}^d$ |
| $d$ | Model hidden dimension (Gemma 2 2B: $d = 2304$) | scalar |
| $\ell$ | Transformer layer index | $\ell \in \{6, 12, 18, 24\}$ |
| $L$ | Total number of transformer layers | scalar ($L = 26$ for Gemma 2 2B) |

## Sparse Autoencoder

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{W}_{\text{enc}}$ | SAE encoder weight matrix | $\mathbb{R}^{m \times d}$ |
| $\mathbf{W}_{\text{dec}}$ | SAE decoder weight matrix | $\mathbb{R}^{d \times m}$ |
| $\mathbf{b}_{\text{enc}}$ | SAE encoder bias | $\mathbb{R}^m$ |
| $\mathbf{b}_{\text{dec}}$ | SAE decoder bias (pre-encoder bias) | $\mathbb{R}^d$ |
| $m$ | SAE dictionary size (number of latent features) | $m \in \{16384, 65536\}$ |
| $\mathbf{z}$ | SAE latent activations (post-activation function) | $\mathbf{z} \in \mathbb{R}^m$ |
| $z_i$ | Activation of SAE feature $i$ | scalar, $z_i \geq 0$ |
| $\hat{\mathbf{x}}$ | SAE reconstruction: $\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$ | $\hat{\mathbf{x}} \in \mathbb{R}^d$ |
| $\mathbf{d}_i$ | Decoder vector for feature $i$ (column of $\mathbf{W}_{\text{dec}}$) | $\mathbf{d}_i \in \mathbb{R}^d$, $\|\mathbf{d}_i\| = 1$ |
| $L_0$ | Average number of active features per input | scalar |

## Feature Hierarchy

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{H}$ | Feature hierarchy (e.g., first-letter, city-continent) | set |
| $p$ | Parent concept (general feature, e.g., "starts with S", "in Europe") | categorical |
| $c$ | Child concept (specific feature, e.g., "Saturday", "Paris") | categorical |
| $G = (V, E)$ | Hierarchy graph with parent-child edges | directed graph |
| $\mathcal{C}(p)$ | Set of child concepts for parent $p$ | set, $\mathcal{C}(p) \subseteq V$ |
| $K$ | Number of parent classes in a hierarchy | scalar (e.g., 26 for first-letter, 6 for city-continent) |

## Probes and Absorption Measurement

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{w}_p$ | Linear probe weight vector for parent class $p$ | $\mathbf{w}_p \in \mathbb{R}^d$ |
| $\hat{y}_{\text{raw}}$ | Probe prediction on raw (un-encoded) activations | categorical |
| $\hat{y}_{\text{SAE}}$ | Probe prediction on SAE-reconstructed activations | categorical |
| $y$ | Ground-truth label | categorical |
| $\text{FN}$ | False negative: $\hat{y}_{\text{raw}} = y$ but $\hat{y}_{\text{SAE}} \neq y$ | binary indicator |
| $\alpha(\mathcal{H}, \text{SAE})$ | Absorption rate for hierarchy $\mathcal{H}$ on a given SAE | $\alpha \in [0, 1]$ |
| $F_1$ | Probe F1 score (harmonic mean of precision and recall) | $F_1 \in [0, 1]$ |

## Probe Degradation Ablation (NEW for iter_010)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{W}_p^{(\epsilon)}$ | Probe weights with injected noise: $\mathbf{W}_p + \epsilon \cdot \mathcal{N}(0, I)$ | $\mathbb{R}^{K \times d}$ |
| $\epsilon$ | Noise scale parameter controlling probe degradation | scalar, $\epsilon \geq 0$ |
| $F_1^{(\epsilon)}$ | Probe F1 at noise level $\epsilon$ | $F_1^{(\epsilon)} \in [0, 1]$ |
| $\alpha^{(\epsilon)}$ | Absorption rate measured with degraded probe at noise level $\epsilon$ | $\alpha^{(\epsilon)} \in [0, 1]$ |
| $\hat{\alpha}_{\text{lin}}(F_1)$ | Linear prediction of absorption from probe F1: $\hat{\alpha} = \beta_0 + \beta_1 F_1$ | scalar |
| $\beta_1$ | Slope of probe degradation curve | scalar ($\beta_1 = -0.398$) |
| $R^2$ | Coefficient of determination for regression fit | $R^2 \in [0, 1]$ |

## Activation Patching

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{z}^{(c \to 0)}$ | SAE latents with child feature $c$ zeroed | $\mathbb{R}^m$ |
| $\mathbf{z}^{(\text{ctrl} \to 0)}$ | SAE latents with random (control) feature zeroed | $\mathbb{R}^m$ |
| $R_c$ | Parent probe recovery rate after child zeroing | $R_c \in [0, 1]$ |
| $R_{\text{ctrl}}$ | Parent probe recovery rate after control zeroing | $R_{\text{ctrl}} \in [0, 1]$ |
| $\Delta R$ | Recovery difference: $R_c - R_{\text{ctrl}}$ | $\Delta R \in [-1, 1]$ |

## Decoder Information Entanglement (reframed from "Benign vs. Pathological")

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\Delta_{\text{logit}}$ | Logit change when parent direction is ablated from child decoder | scalar (nats) |
| $\tau$ | Classification threshold | $\tau \in \{0.05, 0.1, 0.2\}$ |
| $\mathbf{d}_c^{(\neg p)}$ | Child decoder with parent direction removed: $\mathbf{d}_c - (\mathbf{d}_c \cdot \hat{\mathbf{w}}_p)\hat{\mathbf{w}}_p$ | $\mathbb{R}^d$ |

## Hedging Decomposition

| Symbol | Definition |
|--------|-----------|
| FN$_{\text{strict}}$ | False negative where main parent feature does NOT fire (genuine feature gap) |
| FN$_{\text{comp}}$ | False negative where main parent feature fires but probe is wrong (compensatory) |
| FN$_{\text{persist}}$ | Persistent false negative (residual probe boundary error) |

## Statistical Measures

| Symbol | Definition |
|--------|-----------|
| $\rho$ | Spearman rank correlation coefficient |
| $r$ | Pearson correlation coefficient |
| $d$ | Cohen's $d$ effect size (mean difference / pooled SD) |
| $h$ | Cohen's $h$ effect size for proportions |
| CI | Bootstrap 95% confidence interval (10,000 resamples unless stated otherwise) |
| $p_{\text{Bonf}}$ | Bonferroni-corrected $p$-value |

## Rate-Distortion Predictors (Appendix)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\cos(\mathbf{d}_p, \mathbf{d}_c)$ | Decoder cosine similarity between parent and child feature | $[-1, 1]$ |
| $P(c \mid p)$ | Co-occurrence: probability child feature fires given parent fires | $[0, 1]$ |
| $R(p)$ | Reconstruction importance of parent feature (MSE increase on ablation) | scalar |
| $T(G)$ | Absorption Tax: minimum additional $L_0$ cost for absorption-free representation of hierarchy $G$ | scalar, $T(G) \geq 0$ |

## Subscript/Superscript Conventions

| Convention | Meaning |
|-----------|---------|
| Subscript $p$ | Refers to parent feature/concept |
| Subscript $c$ | Refers to child feature/concept |
| Subscript $i, j$ | Generic SAE feature indices |
| Superscript $(\ell)$ | At transformer layer $\ell$ |
| Superscript $(c \to 0)$ | With child feature zeroed |
| Superscript $(\epsilon)$ | At probe noise level $\epsilon$ |
| Subscript $\text{lin}$, $\text{quad}$ | Linear or quadratic regression model |
