# Notation Table

## Model and Activation Notation

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $M$ | Language model (GPT-2 Small or Pythia-70M) | GPT-2: 124M parameters (85M non-embedding); Pythia: 70M parameters (19M non-embedding) |
| $L$ | Number of model layers | 12 for GPT-2 Small; 6 for Pythia-70M |
| $\ell$ | Layer index | $\ell \in \{0, 1, \ldots, L-1\}$ |
| $a_\ell$ | Residual stream activation at layer $\ell$ | $a_\ell \in \mathbb{R}^d$ |
| $d$ | Model hidden dimension | $d = 768$ for GPT-2 Small; $d = 512$ for Pythia-70M |

## SAE Notation

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\text{SAE}$ | Sparse autoencoder | -- |
| $W_{\text{enc}}$ | SAE encoder weight matrix | $W_{\text{enc}} \in \mathbb{R}^{n_{\text{latent}} \times d}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $W_{\text{dec}} \in \mathbb{R}^{d \times n_{\text{latent}}}$ |
| $b_{\text{pre}}$ | SAE pre-encoder bias | $b_{\text{pre}} \in \mathbb{R}^{n_{\text{latent}}}$ |
| $n_{\text{latent}}$ | Number of SAE latents (dictionary size) | $n_{\text{latent}} = 24{,}576$ for GPT-2 res-jb; $n_{\text{latent}} = 32{,}768$ for Pythia-70M res-sm |
| $f(\cdot)$ | ReLU activation function | $f(x) = \max(0, x)$ |
| $z$ | SAE latent activations (post-ReLU) | $z \in \mathbb{R}^{n_{\text{latent}}}$ |
| $\hat{a}$ | SAE reconstruction of activation | $\hat{a} = W_{\text{dec}} \cdot z \in \mathbb{R}^d$ |
| $z_i$ | Activation of latent $i$ (scalar) | $z_i \in \mathbb{R}_{\geq 0}$ |
| $W_{\text{dec}}[i]$ | Decoder direction for latent $i$ | $W_{\text{dec}}[i] \in \mathbb{R}^d$ |

## Feature and Absorption Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{F}$ | Set of first-letter features, $\mathcal{F} = \{A, B, \ldots, Z\}$ |
| $f$ | Individual feature (e.g., "starts with A") | $f \in \mathcal{F}$ |
| $\phi(f)$ | Parent latent ID for feature $f$ | $\phi(f) \in \{0, \ldots, n_{\text{latent}}-1\}$ |
| $A(f)$ | Absorption rate for feature $f$ | $A(f) \in [0, 1]$ |
| $\mathcal{C}(f)$ | Set of child features for feature $f$ | $\mathcal{C}(f) \subset \mathcal{F}$ |
| $c$ | Individual child feature | $c \in \mathcal{C}(f)$ |
| $\rho(f, c)$ | Differential correlation between parent $f$ and child $c$ | $\rho \in [-1, 1]$ |

## Steering Notation

| Symbol | Definition |
|--------|-----------|
| $s$ | Steering strength (scalar multiplier) | $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$ |
| $d_f$ | Steering direction for feature $f$ | $d_f = W_{\text{dec}}[\phi(f)] \in \mathbb{R}^d$ |
| $a'_\ell$ | Steered activation at layer $\ell$ | $a'_\ell = a_\ell + s \cdot d_f$ |
| $P_f(t)$ | Probability of target token $t$ under feature $f$ steering | $P_f(t) \in [0, 1]$ |
| $P_0(t)$ | Baseline probability of token $t$ (no steering) | $P_0(t) \in [0, 1]$ |
| $S(f, s)$ | Raw steering success rate for feature $f$ at strength $s$ | $S(f, s) \in [0, 1]$ |
| $S_{\text{rand}}(s)$ | Random baseline steering success rate at strength $s$ | $S_{\text{rand}}(s) \in [0, 1]$ |
| $\Delta S(f, s)$ | Delta steering success: feature-specific minus random baseline | $\Delta S(f, s) = S(f, s) - S_{\text{rand}}(s)$ |
| $\Delta P(f)$ | Probability lift: mean increase in target token probability | $\Delta P(f) = \mathbb{E}[P_f(t) - P_0(t)]$ |
| $\text{EC50}(f)$ | Median effective steering strength for feature $f$ | $\text{EC50} \in \mathbb{R}_{>0}$ |

## Probing Notation

| Symbol | Definition |
|--------|-----------|
| $k$ | Sparsity level for k-sparse probe | $k \in \{1, 5, 10, 20\}$ |
| $w_k$ | k-sparse linear probe weights | $w_k \in \mathbb{R}^{n_{\text{latent}}}$ with $\|w_k\|_0 \leq k$ |
| $\hat{y}$ | Predicted label from probe | $\hat{y} \in \{A, B, \ldots, Z\}$ |
| $\text{F1}(f, k)$ | F1 score for feature $f$ with k-sparse probe | $\text{F1} \in [0, 1]$ |
| $\text{Prec}(f, k)$ | Precision for feature $f$ with k-sparse probe | $\text{Prec} \in [0, 1]$ |
| $\text{Rec}(f, k)$ | Recall for feature $f$ with k-sparse probe | $\text{Rec} \in [0, 1]$ |
| $\text{F1}_{\text{full}}$ | F1 using all latents (non-sparse probe) | $\text{F1}_{\text{full}} \in [0, 1]$ |
| $\delta_{\text{F1}}(f)$ | F1 degradation for feature $f$ | $\delta_{\text{F1}}(f) = \text{F1}_{\text{full}} - \text{F1}(f, k)$ |

## Statistical Notation

| Symbol | Definition |
|--------|-----------|
| $r$ | Pearson correlation coefficient | $r \in [-1, 1]$ |
| $\rho$ | Spearman rank correlation coefficient | $\rho \in [-1, 1]$ |
| $p$ | p-value (two-tailed) | $p \in [0, 1]$ |
| $R^2$ | Coefficient of determination | $R^2 \in [0, 1]$ |
| $\beta$ | Linear regression slope | $\beta \in \mathbb{R}$ |
| $\epsilon$ | Error term | $\epsilon \sim \mathcal{N}(0, \sigma^2)$ |
| $\text{CV}$ | Coefficient of variation | $\text{CV} = \sigma / |\mu|$ (absolute value on denominator because regression slopes can have opposite signs, making the raw mean near zero and the raw CV negative and uninterpretable) |
| $n$ | Sample size | $n = 26$ features |
| $N$ | Number of prompts per feature | $N = 100$ |
| $\mathcal{H}_0$ | Null hypothesis | -- |
| $\mathcal{H}_1$ | Alternative hypothesis | -- |
| $t$ | t-statistic (Student's t-test) | -- |
| $d$ | Cohen's d (effect size) | $d = (\mu_1 - \mu_2) / \sigma_{\text{pooled}}$ |

## Classification Notation

| Symbol | Definition |
|--------|-----------|
| $\text{HIGH}$ | High absorption category | $A(f) \geq 0.10$ |
| $\text{MEDIUM}$ | Medium absorption category | $0.05 \leq A(f) < 0.10$ |
| $\text{LOW}$ | Low absorption category | $A(f) < 0.05$ |
| $\text{NO\_ABSORPTION}$ | No-absorption baseline | $A(f) < 0.05$ |
