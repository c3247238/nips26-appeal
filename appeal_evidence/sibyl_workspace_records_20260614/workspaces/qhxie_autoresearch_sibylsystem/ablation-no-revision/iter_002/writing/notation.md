# Notation Table

Mathematical symbols and notation used throughout this paper.

## SAE Model Components

| Symbol | Definition | Dimensionality |
|--------|------------|-----------------|
| $x \in \mathbb{R}^d$ | Original residual stream activation (pre-SAE) | $d = 768$ for GPT-2 small |
| $\hat{x} \in \mathbb{R}^d$ | SAE reconstruction of $x$ | $d = 768$ |
| $f \in \mathbb{R}^{d_{\text{sae}}}$ | SAE latent activation vector (sparse) | $d_{\text{sae}} \in \{2048, 8192, 24576\}$ |
| $W_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}} \times d}$ | SAE encoder weight matrix | -- |
| $b_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}}}$ | SAE encoder bias vector | -- |
| $W_{\text{dec}} \in \mathbb{R}^{d \times d_{\text{sae}}}$ | SAE decoder weight matrix | -- |
| $b_{\text{dec}} \in \mathbb{R}^d$ | SAE decoder bias vector | -- |
| $\lambda$ | L1 sparsity penalty coefficient in SAE training loss | scalar |
| $\text{act}_f(t) = W_{\text{dec}}[f] \cdot x_t + b_{\text{dec}}[f]$ | Raw decoder output for latent $f$ on token $t$ (reconstruction contribution before nonlinearity) | scalar |
| $a_f(t)$ | Activation value of latent $f$ on token $t$ | scalar |
| $\text{top5}(f, t)$ | Set of 5 latents with highest activation simultaneously with $f$ on token $t$ | set of 5 indices |

## Absorption Score Computation

| Symbol | Definition |
|--------|------------|
| $T_f$ | Set of activating tokens for latent $f$ (activation > 1% of corpus-maximum for that latent) |
| $|T_f|$ | Number of activating tokens for latent $f$ |
| $x_t$ | Residual stream activation at token $t$ |
| $x_t^{\text{partial}} = W_{\text{dec}}[f] \cdot \text{act}_f(t) + \sum_{c \in \text{top5}(f, t)} W_{\text{dec}}[c] \cdot \text{act}_c(t) + b_{\text{dec}}$ | Partial reconstruction of $x_t$ using feature $f$ and its top-5 co-firers |
| $\text{RVE}_f(t) = 1 - \frac{\text{Var}(x_t - x_t^{\text{partial}})}{\text{Var}(x_t)}$ | Per-token reconstruction variance explained by the partial reconstruction |
| $A_f = \frac{1}{\|T_f\|} \sum_{t \in T_f} \mathbb{1}[\text{RVE}_f(t) > 0.80]$ | Absorption score for latent $f$: fraction of activating tokens where co-firers explain >80% of variance |
| $\mathbb{1}[\cdot]$ | Indicator function (1 if true, 0 otherwise) |

## Layer and Model Parameters

| Symbol | Definition |
|--------|------------|
| $L$ | Number of transformer layers ($L=12$ for GPT-2 small) |
| $\ell \in \{0, 2, 4, 6, 8, 10\}$ | Audited layers in experiments |
| $d_{\text{model}} = 768$ | Transformer model dimension |
| $d_{\text{sae}}$ | SAE dictionary size (number of latents) |
| $\text{L0}(\ell)$ | L0 norm (number of non-zero activations) at layer $\ell$, mean over corpus |

## Token Frequency (H2, Pending)

| Symbol | Definition |
|--------|------------|
| $\text{freq}(t)$ | Log token frequency of token $t$ in Pile validation set |
| $\text{median\_freq}(f) = \text{median}_{t \in T_f} \text{freq}(t)$ | Median token frequency for latent $f$'s activating tokens |
| $Q_1, Q_2, Q_3, Q_4$ | Quartiles of latents binned by median token frequency |

## Faithfulness Metrics

| Symbol | Definition |
|--------|------------|
| $\Delta_{\text{logit}} = \log p_{\text{clean}}(y) - \log p_{\text{corrupt}}(y)$ | Logit difference between clean and corrupted runs |
| $\Delta_{\text{patch}}$ | Logit difference after activation patching |
| $\text{faithfulness} = \frac{\Delta_{\text{patch}}}{\Delta_{\text{logit}}}$ | Fraction of original logit difference restored by patching |

## Statistical Measures

| Symbol | Definition |
|--------|------------|
| $\rho$ or $r$ | Pearson correlation coefficient |
| $r_s$ | Spearman rank correlation coefficient |
| $\mu$ | Mean |
| $\sigma$ | Standard deviation |
| $p$ | p-value for statistical significance |

## Experiment Notation

| Symbol | Definition |
|--------|------------|
| $N_{\text{seq}} = 100$ (pilot) or $1024$ (full) | Number of sequences in analysis corpus |
| $T = 128$ | Sequence length in tokens |
| $N_{\text{total}} = N_{\text{seq}} \times T$ | Total tokens in analysis corpus |