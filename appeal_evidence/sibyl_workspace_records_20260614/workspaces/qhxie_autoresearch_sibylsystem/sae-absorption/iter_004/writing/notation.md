# Notation Table

All mathematical symbols and notation used in the paper, grouped by category. Section writers must reference this file for consistency.

---

## Model and SAE Components

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{x}$ | Input token activation (residual stream) | $\mathbf{x} \in \mathbb{R}^d$ |
| $d$ | Model residual stream dimension | Scalar (e.g., 768 for GPT-2 Small, 2304 for Gemma 2 2B) |
| $D$ | Number of SAE latents (dictionary size / width) | Scalar (e.g., 16384, 24576, 65536, 131072) |
| $\mathbf{W}_{\text{enc}}$ | SAE encoder weight matrix | $\mathbf{W}_{\text{enc}} \in \mathbb{R}^{D \times d}$ |
| $\mathbf{W}_{\text{dec}}$ | SAE decoder weight matrix | $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{d \times D}$ |
| $\mathbf{d}_i$ | Decoder column vector for latent $i$ | $\mathbf{d}_i \in \mathbb{R}^d$ |
| $a_i$ | Activation of SAE latent $i$ on a given input | Scalar, $a_i \geq 0$ |
| $L_0$ | Average number of active latents per token (empirical sparsity) | Scalar |

## Activation Statistics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $f_i$ | Activation frequency of latent $i$: fraction of tokens where $a_i > 0$ | $[0, 1]$ |
| $P(a_i > 0, a_j > 0)$ | Pairwise co-activation rate of latents $i$ and $j$ | $[0, 1]$ |
| $\sigma_{ij}$ | Normalized co-activation (niche overlap): $\sigma_{ij} = P(a_i > 0, a_j > 0) / \min(f_i, f_j)$ | $[0, 1]$ |

## Lotka-Volterra Competition Framework

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\alpha_{ij}$ | LV competition coefficient: $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ | $> 1$ predicts absorption of $i$ by $j$ |
| $\tau$ | Threshold for absorption prediction: predict absorbed if $\max_j \alpha_{ij} > \tau$ | Calibrated on training letters |

## Absorption Metrics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\text{DAS}(P, k)$ | Distributed Absorption Score: measures how much parent $P$'s information is captured by top-$k$ children | $[0, 1]$; higher = more absorbed |
| $\text{DAS}(P, k=1)$ | Single-child absorption; equivalent to Chanin metric | Standard metric |
| $\text{DAS}(P, k=3)$ | Three-child distributed absorption; estimated via logistic regression | Proposed extension |

## Corpus Statistics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\text{PMI}(l, t)$ | Pointwise Mutual Information between letter category $l$ and token $t$ | $\log \frac{P(l, t)}{P(l) \cdot P(t)}$ |

## Regression Model

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\beta_0$ | Intercept | |
| $\beta_1$ | Coefficient on $\log(L_0)$ | Sparsity effect |
| $\beta_2$ | Coefficient on $\log(D)$ | Width effect |
| $\beta_3$ | Coefficient on layer index | Layer effect |
| $\beta_4$ | Coefficient on $\log(\text{PMI})$ | Corpus co-occurrence effect (H2) |

## Evaluation Metrics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $r$ | Pearson correlation coefficient | |
| $\rho$ | Spearman rank correlation coefficient | |
| $R^2$ | Coefficient of determination | |
| Partial $R^2$ | $R^2$ increase attributable to a single predictor | $(R^2_{\text{full}} - R^2_{\text{reduced}}) / (1 - R^2_{\text{reduced}})$ |
| F1 | Harmonic mean of precision and recall | |
| AUC | Area under the ROC curve | |
| AIC | Akaike Information Criterion | Lower = better fit |

## Subscript / Superscript Conventions

| Convention | Meaning |
|-----------|---------|
| $i, j$ | SAE latent indices |
| $P$ | Parent feature (general, e.g., "first letter = A") |
| $C_1, C_2, C_3$ | Child features ranked by $\alpha_{ij}$ (specific, absorbing) |
| $l$ | Letter index ($l \in \{A, B, \ldots, Z\}$) |
