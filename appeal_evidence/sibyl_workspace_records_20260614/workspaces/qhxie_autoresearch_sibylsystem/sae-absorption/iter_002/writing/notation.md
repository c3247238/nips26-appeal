# Notation Table

All symbols used in the paper, grouped by category. Section writers must use these definitions
without deviation.

---

## Model and SAE Architecture

| Symbol | Definition | Dimensionality / Notes |
|--------|-----------|------------------------|
| $d$ | Residual stream dimension | $\mathbb{R}$; for GPT-2 Small: $d = 768$ |
| $d_\text{sae}$ | SAE dictionary size (number of latent features) | $\mathbb{R}$; tested range: 12,288–98,304 |
| $E$ | SAE encoder weight matrix | $E \in \mathbb{R}^{d_\text{sae} \times d}$ |
| $D$ | SAE decoder weight matrix | $D \in \mathbb{R}^{d \times d_\text{sae}}$; columns $d_j$ are unit-norm |
| $b$ | SAE encoder bias vector | $b \in \mathbb{R}^{d_\text{sae}}$ |
| $e_j$ | Encoder direction for feature $j$ (row $j$ of $E$) | $e_j \in \mathbb{R}^d$ |
| $\hat{e}_j$ | Unit-normalized encoder direction | $\hat{e}_j = e_j / \|e_j\|_2$ |
| $d_j$ | Decoder direction (column $j$ of $D$) | $d_j \in \mathbb{R}^d$, $\|d_j\|_2 = 1$ |
| $f(\cdot)$ | SAE activation function (ReLU or TopK) | applied element-wise |
| $z$ | SAE latent activation vector | $z = f(Ex + b) \in \mathbb{R}^{d_\text{sae}}$ |
| $z_j$ | Activation of feature $j$ | $z_j \in \mathbb{R}_{\geq 0}$ |
| $\lambda$ | Sparsity penalty coefficient | $\lambda > 0$; approximately $1/L_0$ in expectation |
| $L_0$ | Mean number of active features per input | scalar; $L_0 = \mathbb{E}[\|z\|_0]$ |

---

## Feature Hierarchy and Absorption

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $p$ | Parent feature index | e.g., letter-class feature |
| $c$ | Child feature index | e.g., specific-token feature absorbed by $p$ |
| $\theta_{p,c}$ | Decoder angle between parent and child | $\theta_{p,c} = \arccos(d_p \cdot d_c)$ |
| $p_\text{co}$ | Co-occurrence probability of parent and child concepts | $p_\text{co} = P(\text{parent present} \cap \text{child present})$ |
| $\alpha$ | Absorption rate for a (parent, child) pair | $\alpha = P(z_c = 0 \mid \text{child present, parent active})$ |
| $\delta$ | Absorption degree (continuous; used in iter_001 proposal) | $\delta \in [0,1]$ |

---

## Detection Metrics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\text{EDA}(j)$ | Encoder-Decoder Alignment (dissociation) for feature $j$ | $\text{EDA}(j) = 1 - \cos(\hat{e}_j, d_j) \in [0, 2]$; higher = more dissociated |
| $\cos(\hat{e}_j, d_j)$ | Cosine similarity between encoder and decoder directions | $\in [-1, 1]$ |
| $\cos(\hat{e}_p, d_c)$ | Cross-directional: parent encoder aligned with child decoder | Used as alternative detector; AUROC = 0.730 |
| $\cos(\hat{e}_c, d_p)$ | Cross-directional: child encoder aligned with parent decoder | Mean-aggregated across candidate parents |
| $\rho$ | Spearman rank correlation | Used for scaling analyses |

---

## Experimental Notation

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\mathcal{L}_\text{SAE}$ | SAE training loss | $\mathcal{L}_\text{SAE} = \mathbb{E}[\|x - Df(Ex+b)\|_2^2] + \lambda \mathbb{E}[\|f(Ex+b)\|_0]$ |
| $\text{AUROC}$ | Area under receiver operating characteristic curve | Primary detection metric |
| $\text{AUPRC}$ | Area under precision-recall curve | Reported alongside AUROC for imbalanced labels |
| $d_\text{Cohen}$ | Cohen's $d$ effect size | $({\mu_+} - {\mu_-}) / \sigma_\text{pooled}$ |
| $\text{BIC}$ | Bayesian Information Criterion | Used for sigmoid vs. linear model comparison |
| $\text{LRT}$ | Likelihood-ratio test $p$-value | For nested model comparison |
| $z_\text{null}$ | $z$-score above permutation null | $({\text{AUROC}_\text{observed}} - \mu_\text{null}) / \sigma_\text{null}$ |

---

## Architecture-Specific Notation

| Term | Meaning |
|------|---------|
| jb (or res-jb) | Standard SAE with L1 penalty (Bricken et al. training); primary suite |
| AJT (or ajt) | Alternative training regime with different sparsity formulation; shows reversed EDA polarity |
| TopK | SAE variant with exact $k$ active features; no L1 penalty |
| L$\ell$ | Layer $\ell$ of GPT-2 Small (0-indexed); "L6" = layer 6 residual stream pre-MLP |

---

## Propositions Reference

| Label | Statement |
|-------|-----------|
| Proposition 1 | Absorbed solution achieves lower expected SAE loss iff $\lambda > \sin^2(\theta_{p,c})$ |
| Corollary 1 | The absorption threshold is independent of co-occurrence frequency $p_\text{co}$ |
| Corollary 2 | Absorption propensity is monotonically increasing in sparsity penalty $\lambda$ |
| Proposition 2 | (Mechanistic Conjecture) Under conditions C1–C3, the encoder of an absorbed child feature is pulled toward the parent decoder direction during training |
