# Notation Table

This document establishes all mathematical symbols and notation used throughout the paper. All section writers should reference this file for consistency.

---

## Inputs and Data

| Symbol | Definition | Dimensionality |
|--------|------------|----------------|
| $x$ | Input activation vector from a transformer layer | $x \in \mathbb{R}^d$ |
| $\mathcal{D}$ | Held-out corpus (activation dataset) | — |
| $N$ | Number of checkpoints or samples in an analysis | — |
| $d$ | Dimensionality of the residual stream (model hidden size) | $d \in \mathbb{N}$ |
| $d_{\text{in}}$ | Input dimensionality to the SAE encoder | $d_{\text{in}} \in \mathbb{N}$ |

## SAE Components

| Symbol | Definition | Dimensionality |
|--------|------------|----------------|
| $\text{SAE}(\cdot)$ | Sparse autoencoder function mapping activations to reconstructed activations | $\text{SAE}: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{in}}}$ |
| $W_{\text{enc}}$ | SAE encoder weight matrix | $W_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}} \times d_{\text{in}}}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $W_{\text{dec}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{sae}}}$ |
| $b_{\text{enc}}$ | SAE encoder bias vector | $b_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}}}$ |
| $b_{\text{dec}}$ | SAE decoder bias vector | $b_{\text{dec}} \in \mathbb{R}^{d_{\text{in}}}$ |
| $z$ | SAE latent representation (post-activation) | $z \in \mathbb{R}^{d_{\text{sae}}}$ |
| $\hat{x}$ | Reconstructed activation vector output by the SAE decoder | $\hat{x} \in \mathbb{R}^{d_{\text{in}}}$ |
| $d_{\text{sae}}$ | SAE dictionary size (number of latents) | $d_{\text{sae}} \in \mathbb{N}$ |

## Sparsity and Reconstruction Metrics

| Symbol | Definition | Range |
|--------|------------|-------|
| $L_0$ | Average number of active (non-zero) latents per token | $L_0 \in [0, d_{\text{sae}}]$ |
| $\text{EV}$ | Explained variance: fraction of input variance recovered by the SAE reconstruction | $\text{EV} \in (-\infty, 1]$ |
| $\text{MSE}_{\text{recon}}$ | Mean squared error between original and reconstructed activations | $\text{MSE}_{\text{recon}} \in [0, \infty)$ |
| $\text{CE}_{\text{orig}}$ | Cross-entropy loss of the original model on next-token prediction | $\text{CE}_{\text{orig}} \in [0, \infty)$ |
| $\text{CE}_{\text{rec}}$ | Cross-entropy loss when SAE-reconstructed activations are substituted into the model | $\text{CE}_{\text{rec}} \in [0, \infty)$ |
| $\text{CE}_{\text{recovered}}$ | CE loss recovered, defined as $\text{CE}_{\text{orig}} / \text{CE}_{\text{rec}}$ | $\text{CE}_{\text{recovered}} \in [0, \infty)$ |
| $\delta_{\text{dead}}$ | Dead-neuron rate: fraction of latents with near-zero activation frequency ($< 10^{-5}$) | $\delta_{\text{dead}} \in [0, 1]$ |

## Absorption and Hedging Metrics

| Symbol | Definition | Range |
|--------|------------|-------|
| $\alpha$ | Absorption rate: fraction of parent-feature tokens where the parent is not directly represented but the child is | $\alpha \in [0, 1]$ |
| $\alpha_{\text{FL}}$ | First-letter absorption rate (Chanin et al., 2024 benchmark) | $\alpha_{\text{FL}} \in [0, 1]$ |
| $\alpha_{\text{TA}}$ | Task-agnostic absorption rate (automated hierarchy probe metric) | $\alpha_{\text{TA}} \in [0, 1]$ |
| $h$ | Hedging rate: fraction of correlated token pairs where the SAE activates the same top feature for both | $h \in [0, 1]$ |
| $\tau$ | Cosine similarity threshold for latent-probe alignment in task-agnostic absorption detection | $\tau = 0.7$ |

## Downstream Interpretability Metrics

| Symbol | Definition | Range |
|--------|------------|-------|
| $\text{F1}_{\text{probe}}$ | Sparse probing F1 score on downstream concept classification tasks | $\text{F1}_{\text{probe}} \in [0, 1]$ |
| $\text{RAVEL}_{\text{cause}}$ | RAVEL Cause score measuring causal disentanglement of features | $\text{RAVEL}_{\text{cause}} \in [0, 1]$ |
| $\text{RAVEL}_{\text{iso}}$ | RAVEL Isolation score measuring causal specificity of features | $\text{RAVEL}_{\text{iso}} \in [0, 1]$ |
| $\text{TPP}$ | Token-level predictive performance (SAEBench metric, where available) | $\text{TPP} \in [0, 1]$ |
| $\text{SCR}$ | Sparse coding recall (SAEBench metric, where available) | $\text{SCR} \in [0, 1]$ |

## Statistical and Regression Notation

| Symbol | Definition |
|--------|------------|
| $r$ | Pearson correlation coefficient |
| $\rho$ | Spearman rank correlation coefficient |
| $r_{\text{partial}}$ | Partial correlation coefficient controlling for covariates |
| $\beta$ | Standardized OLS regression coefficient |
| $\text{SE}$ | Standard error |
| $U$ | Mann-Whitney U test statistic |
| $p$ | Two-tailed p-value |
| $R^2$ | Coefficient of determination |
| $\mathcal{H}_0$ | Null hypothesis |
| $\mathcal{H}_1, \mathcal{H}_2, \mathcal{H}_3$ | Primary, secondary, and tertiary hypotheses of this paper |

## Architecture and Model Families

| Symbol / Abbreviation | Definition |
|-----------------------|------------|
| SAE | Sparse Autoencoder |
| Standard | Standard SAE with $L_1$ sparsity penalty |
| TopK | Top-$K$ activation SAE |
| JumpReLU | JumpReLU SAE with adaptive thresholding |
| GatedSAE | Gated SAE with separate gating and magnitude paths |
| Matryoshka | Matryoshka SAE with multi-scale dictionary learning |
| PAnneal | Penalty-annealed SAE |
| FS | Feature-splitting SAE |
