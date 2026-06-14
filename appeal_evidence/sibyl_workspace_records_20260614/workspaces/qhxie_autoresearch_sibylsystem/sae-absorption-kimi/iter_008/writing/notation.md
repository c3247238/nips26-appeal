# Notation Table

## Inputs and Data

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{D}$ | Training dataset (synthetic hierarchical features) | — |
| $x \in \mathbb{R}^d$ | Input activation vector | $d = 256$ |
| $n$ | Number of ground-truth features | $n = 1024$ |
| $d_{\text{sae}}$ | SAE dictionary size (number of latents) | $d_{\text{sae}} = 2048$ |
| $B$ | Batch size | $B = 1024$ |
| $N$ | Total training tokens | $N = 2 \times 10^6$ |
| $S$ | Number of random seeds | $S = 5$ |

## Model Components

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $W_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}} \times d}$ | Encoder weight matrix | $2048 \times 256$ |
| $W_{\text{dec}} \in \mathbb{R}^{d \times d_{\text{sae}}}$ | Decoder weight matrix | $256 \times 2048$ |
| $b_{\text{pre}} \in \mathbb{R}^d$ | Pre-encoder bias | $256$ |
| $b_{\text{post}} \in \mathbb{R}^{d_{\text{sae}}}$ | Post-encoder bias | $2048$ |
| $z \in \mathbb{R}^{d_{\text{sae}}}$ | Latent activation vector (post-sparsity) | $2048$ |
| $\hat{x} \in \mathbb{R}^d$ | Reconstructed activation vector | $256$ |
| $\theta$ | Model parameters (all trainable weights and biases) | — |

## Sparsity Measures

| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{L0}$ | Mean number of active features per token | $[0, d_{\text{sae}}]$ |
| $\text{true\_L0}$ | Mean active features per token after removing dead latents | $[0, d_{\text{sae}}]$ |
| $\lambda$ | L1 sparsity penalty coefficient | $(0, \infty)$ |
| $k$ | TopK explicit sparsity parameter (number of active latents) | $\{1, \dots, d_{\text{sae}}\}$ |
| $\tau$ | Activation threshold for dead latent detection | $\tau = 0.05$ |

## Absorption Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $A$ | Absorption rate: fraction of parent firings where child-matching latents also activate | $[0, 1]$ |
| $A_{\text{random}}$ | Absorption rate for untrained random dictionary | $[0, 1]$ |
| $A_{\text{active}}$ | Absorption rate computed on active latents only | $[0, 1]$ |

## Feature Recovery Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{MCC}$ | Matthews Correlation Coefficient between ground-truth features and SAE latents (via Hungarian matching) | $[-1, 1]$ |
| $\text{MSE}$ | Mean squared reconstruction error: $\|x - \hat{x}\|_2^2 / d$ | $[0, \infty)$ |
| $\text{EV}$ | Explained variance: $1 - \text{Var}(x - \hat{x}) / \text{Var}(x)$ | $(-\infty, 1]$ |

## Statistical Notation

| Symbol | Definition |
|--------|-----------|
| $\mu$ | Sample mean |
| $\sigma$ | Sample standard deviation |
| $t_{\text{Welch}}$ | Welch's t-test statistic |
| $d_{\text{Cohen}}$ | Cohen's d effect size |
| $r$ | Pearson correlation coefficient |
| $\text{CI}_{95}$ | 95% confidence interval |
| $p$ | Two-tailed p-value |
| $F$ | F-statistic (ANOVA) |
| $H_0$ | Null hypothesis |
| $H_1$ | Alternative hypothesis |

## Architecture Variants

| Abbreviation | Full Name |
|-------------|-----------|
| Baseline L1 | Standard ReLU SAE with L1 sparsity penalty |
| TopK | SAE with explicit k-sparse selection |
| Matryoshka | Nested multi-scale dictionary SAE |
| OrtSAE | Orthogonal Sparse Autoencoder (decoder orthogonality penalty) |
| Gated | Gated SAE with separate gate and magnitude paths |
| Random | Untrained random dictionary baseline |

## Hierarchy Parameters

| Symbol | Definition | Value |
|--------|-----------|-------|
| $R$ | Number of root nodes | $R = 32$ |
| $b$ | Branching factor | $b = 4$ |
| $D$ | Tree depth | $D = 3$ |
| $P$ | Total number of parent-child pairs | $P = 992$ |
