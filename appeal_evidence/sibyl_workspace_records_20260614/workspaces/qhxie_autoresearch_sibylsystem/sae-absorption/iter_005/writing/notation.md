# Notation Table

All mathematical symbols used in the paper, grouped by category. Section writers must reference this file to ensure consistency.

---

## SAE Architecture

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $d$ | Residual stream dimension of the language model | scalar |
| $m$ | SAE dictionary width (number of latents) | scalar |
| $W_e \in \mathbb{R}^{m \times d}$ | SAE encoder weight matrix | $m \times d$ |
| $W_d \in \mathbb{R}^{d \times m}$ | SAE decoder weight matrix | $d \times m$ |
| $\mathbf{b}_e \in \mathbb{R}^m$ | SAE encoder bias | $m$ |
| $\mathbf{b}_d \in \mathbb{R}^d$ | SAE decoder bias | $d$ |
| $\mathbf{x} \in \mathbb{R}^d$ | Input activation (residual stream at a given position) | $d$ |
| $\hat{\mathbf{x}} \in \mathbb{R}^d$ | SAE reconstruction of $\mathbf{x}$ | $d$ |
| $\mathbf{z} \in \mathbb{R}^m$ | SAE latent activations (post-activation function) | $m$ |
| $z_j$ | Activation of the $j$-th SAE latent | scalar |
| $\mathbf{d}_j \in \mathbb{R}^d$ | Decoder direction for the $j$-th latent (column of $W_d$) | $d$ |

## Sparsity and Training

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $L_0$ | Expected number of active latents per input (L0 sparsity) | scalar |
| $\lambda$ | Sparsity penalty coefficient (L1 regularization weight) | scalar |
| $k$ | Number of active features in TopK/k-sparse SAEs | scalar |

## Absorption Measurement

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{p} \in \mathbb{R}^d$ | Linear probe weight vector for a target concept | $d$ |
| $\theta_{p,c}$ | Angle between parent probe direction $\mathbf{p}$ and child decoder direction $\mathbf{d}_c$ | scalar (radians) |
| $\text{cos}(\mathbf{p}, \mathbf{d}_j)$ | Cosine similarity between probe direction and latent $j$'s decoder | scalar $\in [-1, 1]$ |
| $\tau_{\text{cos}}$ | Cosine similarity threshold for absorption detection | scalar (default: 0.025) |
| $\tau_{\text{dom}}$ | Dominance ratio threshold (ratio of top to second-highest ablation effect) | scalar (default: 1.0 or 2.0) |
| $\alpha_j(\mathbf{x})$ | Ablation effect of latent $j$ on input $\mathbf{x}$ (integrated gradients) | scalar |
| $\text{FN}(f)$ | False-negative set: inputs where all split latents for feature $f$ fail to fire but the probe classifies correctly | set of inputs |
| $R_{\text{abs}}$ | Absorption rate: fraction of features exhibiting absorption | scalar $\in [0, 1]$ |

## Confound Resolution (Phase 1)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $r$ | Pearson correlation coefficient | scalar $\in [-1, 1]$ |
| $r_{\text{partial}}$ | Partial correlation controlling for specified covariates | scalar $\in [-1, 1]$ |
| $\rho$ | Spearman rank correlation coefficient | scalar $\in [-1, 1]$ |
| $a$ | Standardized path coefficient: log(L0) $\to$ Absorption | scalar |
| $b$ | Standardized path coefficient: Absorption $\to$ Quality | scalar |
| $c$ | Total effect: log(L0) $\to$ Quality (not controlling for Absorption) | scalar |
| $c'$ | Direct effect: log(L0) $\to$ Quality (controlling for Absorption) | scalar |
| $ab$ | Indirect (mediated) effect $= c - c'$ | scalar |
| $\Gamma$ | Rosenbaum sensitivity parameter: odds ratio of hidden confounder at which significance is lost | scalar $\geq 1$ |

## Scaling Surface (Phase 3)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $s(\cdot)$ | Smooth univariate spline term in the GAM | function |
| $\text{ti}(\cdot, \cdot)$ | Tensor interaction term in the GAM (tests joint dependence) | function |
| $R^2$ | Coefficient of determination (proportion of variance explained) | scalar $\in [0, 1]$ |

## Quality Metrics

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{SP-F1}$ | Sparse probing F1 score (SAEBench metric) | scalar $\in [0, 1]$ |
| $\text{SCR}$ | Spurious correlation removal score (SAEBench metric) | scalar |
| $\text{TPP}$ | RAVEL true positive proportion (SAEBench metric) | scalar $\in [0, 1]$ |
| $\text{UL}$ | Unlearning score (SAEBench metric) | scalar |
