# Notation Table

## Inputs and Data

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{D}$ | Training / evaluation dataset | — |
| $x$ | Input text sequence | — |
| $t$ | Individual token in vocabulary | $t \in \mathcal{V}$ |
| $\mathcal{V}$ | Model vocabulary | $|\mathcal{V}| \approx 50{,}000$ |
| $N$ | Number of sentences per concept in synthetic dataset | $N = 100$ |
| $s_i$ | $i$-th sentence in synthetic dataset | — |

## Model Components

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $f$ | Base language model (e.g., Pythia-160M) | — |
| $h^{(l)}(x)$ | Residual-stream activation at layer $l$ for input $x$ | $h^{(l)} \in \mathbb{R}^d$ |
| $d$ | Residual-stream dimension | $d = 768$ (Pythia-160M) |
| $\phi$ | Sparse autoencoder (SAE) | — |
| $W_{\text{enc}}$ | SAE encoder matrix | $W_{\text{enc}} \in \mathbb{R}^{m \times d}$ |
| $W_{\text{dec}}$ | SAE decoder matrix | $W_{\text{dec}} \in \mathbb{R}^{d \times m}$ |
| $b_{\text{enc}}$ | SAE encoder bias | $b_{\text{enc}} \in \mathbb{R}^m$ |
| $b_{\text{dec}}$ | SAE decoder bias | $b_{\text{dec}} \in \mathbb{R}^d$ |
| $m$ | SAE latent dimension (number of features) | $m = 2048$ |
| $z$ | SAE latent activation vector | $z \in \mathbb{R}^m$ |
| $\hat{h}$ | SAE reconstruction of residual activation | $\hat{h} \in \mathbb{R}^d$ |
| $k$ | Number of top-k latents for k-sparse probing | $k = 10$ |
| $z^{(k)}$ | Top-k sparse latent vector (k largest entries retained) | $z^{(k)} \in \mathbb{R}^m$ |

## Probes and Classification

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $p$ | Parent concept in a hierarchy (e.g., "building") | — |
| $c$ | Child concept in a hierarchy (e.g., "house") | — |
| $\mathcal{H}$ | Set of all parent-child hierarchies | $|\mathcal{H}| = 10$ |
| $\mathcal{C}_p$ | Set of children for parent $p$ | $|\mathcal{C}_p| \in \{2, 3\}$ |
| $w$ | Logistic regression probe weights | $w \in \mathbb{R}^d$ or $w \in \mathbb{R}^m$ |
| $b$ | Logistic regression probe bias | $b \in \mathbb{R}$ |
| $\sigma(\cdot)$ | Sigmoid function | $\sigma: \mathbb{R} \to [0, 1]$ |
| $\hat{y}$ | Predicted probability from probe | $\hat{y} \in [0, 1]$ |
| $\text{AUROC}$ | Area under the receiver operating characteristic curve | $\in [0, 1]$ |
| $\text{acc}$ | Classification accuracy | $\in [0, 1]$ |

## Absorption Metric

| Symbol | Definition | Range |
|--------|-----------|-------|
| $\tau_{\text{fs}}$ | Feature-splitting threshold for k-sparse probing | $\{0.01, 0.03, 0.05\}$ |
| $\tau_{\text{pa}}$ | Parent accuracy threshold | $\tau_{\text{pa}} = 0$ |
| $\tau_{\text{ps}}$ | Parent specificity threshold | $\tau_{\text{ps}} = -1$ |
| $A_{\text{full}}$ | Full absorption score for a single hierarchy | $[0, 1]$ |
| $A_{\text{frac}}$ | Absorption fraction score | $[0, 1]$ |
| $\bar{A}_{\text{FL}}$ | Mean first-letter absorption across hierarchies | $[0, 1]$ |
| $\bar{A}_{\text{SH}}$ | Mean semantic-hierarchy absorption across hierarchies | $[0, 1]$ |
| $\bar{A}_{\text{NH}}$ | Mean non-hierarchy control absorption across pairs | $[0, 1]$ |
| $A_{\text{random}}$ | Absorption score for Random-SAE control | $[0, 1]$ |

**Absorption Formula (SAEBench):**

$$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$$

where:
- $\text{acc}_{\text{resid}}$ = probe accuracy on base model residual activations
- $\text{acc}_{\text{sae}}$ = probe accuracy on full SAE latents
- $\text{acc}_{\text{k-sparse}}$ = probe accuracy on top-k SAE latents

## Statistical Metrics

| Symbol | Definition |
|--------|-----------|
| $r$ | Pearson correlation coefficient | $r \in [-1, 1]$ |
| $\text{CI}_{95}$ | 95% confidence interval | |
| $B$ | Bootstrap resampling count | $B = 10{,}000$ |
| $t$ | Paired t-test statistic | |
| $p$ | Statistical significance (p-value) | |
| $n$ | Sample size (number of SAEs) | $n = 7$ (trained) or $n = 8$ (with random) |
| $s$ | Standard deviation | |
| $\mu$ | Mean | |

## Architecture Abbreviations

| Abbreviation | Full Name |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| ReLU | Rectified Linear Unit |
| TopK | Top-K activation gating |
| BatchTopK | Batch-wise Top-K gating |
| GatedSAE | Gated Sparse Autoencoder |
| JumpReLU | Jumping ReLU activation |
| PAnneal | Penalty annealing SAE |
| Matryoshka | Nested (multi-scale) SAE |
| SAEBench | Standardized SAE evaluation benchmark |
| SAELens | Library for training and analyzing SAEs |
| WordNet | Lexical database of English |
