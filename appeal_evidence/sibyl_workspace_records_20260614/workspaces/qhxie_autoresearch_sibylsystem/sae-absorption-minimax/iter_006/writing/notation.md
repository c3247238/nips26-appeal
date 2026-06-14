# Notation Table

This document defines mathematical symbols and notation used consistently throughout the paper.

## Core Symbols

| Symbol | Definition |
|--------|------------|
| $\text{UAS}(f)$ | Unsupervised Absorption Score for feature $f$ |
| $\cosvar(f)$ | Variance of cosine similarities between feature $f$'s decoder direction and all other feature directions |
| $\text{act\_freq}(f)$ | Fraction of tokens where feature $f$ is active |
| $W_{\text{dec}}[f]$ | Decoder weight vector for feature $f$ |
| $\beta$ | Steering magnitude (scalar multiplier) |
| $d_{\text{sae}}$ | SAE latent dimension |
| $\rho$ | Spearman correlation coefficient |

## Absorption Detection (Chanin Protocol)

| Symbol | Definition |
|--------|------------|
| $\tau_{\text{fs}} = 0.03$ | Feature selection threshold |
| $\tau_{\text{ps}} = 0.025$ | Positive set precision threshold |
| $\tau_{\text{pa}} = 0.4$ | Positive set accuracy threshold |
| $\text{acc}_{\text{resid}}$ | Probe accuracy on residual stream (upper bound) |
| $\text{acc}_{\text{sae}}$ | Probe accuracy on SAE features (lower bound) |
| $\text{UAS}_{\text{chanin}} = (\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}) / (1 - \text{acc}_{\text{sae}})$ | Chanin absorption score |

## Sensitivity Measurement (Tian Protocol)

| Symbol | Definition |
|--------|------------|
| AUC$_{\text{para}}$ | Paraphrase AUC - reliability across semantically equivalent inputs |
| $N_{\text{para}}$ | Number of paraphrase pairs (50 in our experiments) |

## Quadrant Classification

| Quadrant | Definition |
|----------|------------|
| Q1 | High absorption (UAS < 0.4) + Low sensitivity (AUC < 0.6) - doubly-compromised |
| Q2 | High absorption + High sensitivity |
| Q3 | Low absorption + Low sensitivity |
| Q4 | Low absorption + High sensitivity - best-case |

## Statistical Symbols

| Symbol | Definition |
|--------|------------|
| $p$ | p-value |
| $r$ | Spearman correlation |
| CI | Confidence interval |
| SE | Standard error |