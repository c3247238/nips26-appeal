# Notation Table

## Inputs and Data

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{D}$ | Training dataset of synthetic activations | $N \times d$ |
| $x$ | Input activation vector | $x \in \mathbb{R}^d$ |
| $d$ | Input dimension (hidden dimension of the source model) | $d = 256$ |
| $N$ | Number of training samples | $N = 2{,}000{,}000$ |
| $B$ | Batch size during training | $B = 1024$ |

## SAE Architecture

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $m$ | SAE latent dimension (dictionary size) | $m = 2048$ |
| $W_{\text{enc}}$ | Encoder weight matrix | $W_{\text{enc}} \in \mathbb{R}^{m \times d}$ |
| $W_{\text{dec}}$ | Decoder weight matrix | $W_{\text{dec}} \in \mathbb{R}^{d \times m}$ |
| $b_{\text{enc}}$ | Encoder bias vector | $b_{\text{enc}} \in \mathbb{R}^m$ |
| $b_{\text{dec}}$ | Decoder bias vector (pre-encoder bias) | $b_{\text{dec}} \in \mathbb{R}^d$ |
| $z$ | SAE latent activation vector (post-activation) | $z \in \mathbb{R}^m$ |
| $\hat{x}$ | Reconstructed activation vector | $\hat{x} \in \mathbb{R}^d$ |
| $k$ | TopK sparsity parameter (number of active latents) | $k = 50$ |
| $\lambda_1$ | L1 sparsity coefficient | $\lambda_1 = 5 \times 10^{-3}$ |
| $\lambda_{\text{ortho}}$ | Orthogonality penalty coefficient | $\lambda_{\text{ortho}} = 10^{-3}$ |

## Synthetic Data Structure

| Symbol | Definition | Value |
|--------|-----------|-------|
| $F$ | Total number of ground-truth features | $F = 1{,}024$ |
| $F_h$ | Number of hierarchical features | $F_h = 672$ |
| $R$ | Number of root trees | $R = 32$ |
| $D$ | Tree depth | $D = 3$ |
| $\beta$ | Branching factor | $\beta = 4$ |
| $f_i$ | Ground-truth feature $i$ | $f_i \in \{0, 1\}$ (binary presence) |
| $\mathcal{H}$ | Set of all parent-child hierarchical pairs | $|\mathcal{H}| = 992$ |
| $(p, c)$ | A parent-child feature pair | $p, c \in \{1, \dots, F\}$ |

## Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $A$ | Absorption rate | $A \in [0, 1]$ |
| $A_{\text{full}}$ | Full absorption score (SAEBench protocol) | $[0, 1]$ |
| $\text{MCC}$ | Matthews correlation coefficient (feature recovery) | $[-1, 1]$ |
| $\text{MSE}$ | Mean squared error (reconstruction quality) | $[0, \infty)$ |
| $L_0$ | Average number of active (non-zero) latents per sample | $[0, m]$ |
| $H$ | Feature hedging score | $[0, 1]$ |
| $\text{dead}$ | Fraction of never-active (dead) latents | $[0, 1]$ |
| $\text{AUROC}$ | Area under receiver operating characteristic curve | $[0, 1]$ |

## Statistical Notation

| Symbol | Definition |
|--------|-----------|
| $\mu$ | Population mean |
| $\bar{x}$ | Sample mean |
| $\sigma$ | Population standard deviation |
| $s$ | Sample standard deviation |
| $d$ | Cohen's d (effect size) |
| $r$ | Pearson correlation coefficient |
| $\tau$ | Kendall's tau rank correlation |
| $t$ | t-statistic |
| $F$ | F-statistic (ANOVA) |
| $p$ | p-value |
| $\alpha$ | Significance level | $\alpha = 0.05$ |
| $\text{CI}$ | Confidence interval |
| $H_0$ | Null hypothesis |
| $H_1, H_2, H_3$ | Research hypotheses |

## Model Variants

| Abbreviation | Full Name | Key Component |
|-------------|-----------|---------------|
| Baseline | Standard ReLU SAE | ReLU activation + L1 sparsity |
| +TopK | TopK SAE | Hard top-k activation ($k=50$) |
| +MultiScale | Matryoshka/MultiScale SAE | Nested dictionaries (2 levels) |
| +Orthogonality | OrtSAE | Decoder orthogonality penalty |
| +Gating | Gated SAE | Decoupled detection/magnitude paths |
| +Full Matryoshka | Full Matryoshka SAE | TopK + MultiScale + hierarchical loss |
| Random | Random decoder control | Untrained, permuted decoder |
| L0-matched | L1-tuned Baseline | L1 coefficient tuned to match target L0 |

## SAE-Specific Terms

| Symbol | Definition |
|--------|-----------|
| $\text{dead latents}$ | Latents that never activate during training |
| $\text{shrinkage}$ | Ratio of SAE output norm to input norm |
| $\text{uniqueness}$ | Fraction of latents with non-overlapping feature preferences |
| $\tau_{\text{fs}}$ | Feature-splitting threshold for k-sparse probing |

## Sets and Indices

| Symbol | Definition |
|--------|-----------|
| $i, j$ | Feature / latent indices |
| $s$ | Seed index (for replicates) |
| $v$ | Variant index |
| $r$ | Replicate index |
| $\mathcal{V}$ | Set of all SAE variants | $\mathcal{V} = \{\text{Baseline}, +\text{TopK}, +\text{MultiScale}, +\text{Orthogonality}, +\text{Gating}, +\text{Full Matryoshka}\}$ |
| $\mathcal{S}$ | Set of random seeds | $\mathcal{S} = \{42, 123, 456, 789, 1011\}$ |
