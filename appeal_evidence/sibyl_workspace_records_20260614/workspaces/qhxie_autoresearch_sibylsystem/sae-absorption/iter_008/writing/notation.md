# Notation Table

All mathematical symbols used throughout the paper, grouped by category.

---

## Model and Representations

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathbf{x}$ | Input token embedding / residual stream activation at a given layer | $\mathbf{x} \in \mathbb{R}^d$ |
| $d$ | Residual stream dimension of the base model (Gemma 2 2B: $d = 2304$) | scalar |
| $L$ | Layer index in the transformer model | $L \in \{0, 1, \ldots, 25\}$ |
| $\mathbf{x}^{(L)}$ | Residual stream activation at layer $L$ | $\mathbf{x}^{(L)} \in \mathbb{R}^d$ |

## Sparse Autoencoder

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{SAE}$ | Sparse autoencoder mapping | $\mathbb{R}^d \to \mathbb{R}^d$ |
| $\hat{\mathbf{x}}$ | SAE-reconstructed activation | $\hat{\mathbf{x}} \in \mathbb{R}^d$ |
| $M$ | SAE dictionary size (number of latent features) | scalar ($M \in \{16384, 32768, 65536\}$) |
| $\mathbf{a}$ | SAE latent activation vector (sparse) | $\mathbf{a} \in \mathbb{R}^M$ |
| $a_i$ | Activation of the $i$-th SAE latent | scalar, $a_i \geq 0$ |
| $\mathbf{W}_{\text{enc}}$ | SAE encoder weight matrix | $\mathbf{W}_{\text{enc}} \in \mathbb{R}^{M \times d}$ |
| $\mathbf{W}_{\text{dec}}$ | SAE decoder weight matrix | $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{d \times M}$ |
| $\mathbf{d}_i$ | Decoder direction for the $i$-th latent (column of $\mathbf{W}_{\text{dec}}$) | $\mathbf{d}_i \in \mathbb{R}^d$ |
| $\mathbf{b}_{\text{enc}}$, $\mathbf{b}_{\text{dec}}$ | Encoder and decoder bias vectors | $\mathbb{R}^M$, $\mathbb{R}^d$ |
| $L_0$ | Expected number of active latents per input (sparsity) | scalar |

## Feature Hierarchy

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $G = (V, E)$ | Feature hierarchy graph (directed, parent $\to$ child) | graph |
| $p$ | Parent feature (general concept, e.g., "starts with s") | node in $G$ |
| $c$ | Child feature (specific concept, e.g., "snake") | node in $G$ |
| $(p, c) \in E$ | Parent-child edge: $c$ implies $p$ | directed edge |
| $\pi_c$ | Prior probability of child feature $c$ being active | scalar, $\pi_c \in [0, 1]$ |
| $K$ | Number of classes in a hierarchy (e.g., 26 for first-letter, 6 for city-continent) | scalar |

## Probes and Classification

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $f_{\text{probe}}$ | Linear probe (L2-regularized logistic regression) | $\mathbb{R}^d \to \{1, \ldots, K\}$ |
| $\mathbf{w}_k$ | Probe weight vector for class $k$ | $\mathbf{w}_k \in \mathbb{R}^d$ |
| $y$ | True class label | $y \in \{1, \ldots, K\}$ |
| $\hat{y}_{\text{raw}}$ | Probe prediction on raw residual stream: $f_{\text{probe}}(\mathbf{x})$ | $\hat{y}_{\text{raw}} \in \{1, \ldots, K\}$ |
| $\hat{y}_{\text{sae}}$ | Probe prediction on SAE output: $f_{\text{probe}}(\hat{\mathbf{x}})$ | $\hat{y}_{\text{sae}} \in \{1, \ldots, K\}$ |
| $\text{F1}$ | Weighted macro-F1 score of the probe | scalar, $[0, 1]$ |

## Absorption Metrics

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{FN}$ | False negative set: tokens where $\hat{y}_{\text{raw}} = y$ but $\hat{y}_{\text{sae}} \neq y$ | set of tokens |
| $n_{\text{FN}}$ | Number of false negatives | scalar |
| $\text{AR}$ | Absorption rate: fraction of classes with at least one absorbed false negative | scalar, $[0, 1]$ |
| $\text{IG}_i(t)$ | Integrated-gradients attribution of latent $i$ for token $t$ | scalar |
| $\cos(\mathbf{d}_i, \mathbf{w}_k)$ | Cosine similarity between decoder direction $i$ and probe direction $k$ | scalar, $[-1, 1]$ |
| $\tau_{\cos}$ | Cosine similarity threshold for absorption detection (default: 0.025) | scalar |
| $\tau_{\text{gap}}$ | Magnitude gap threshold (default: 1.0) | scalar |

## Activation Patching

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{do}(a_c := 0)$ | Intervention: set child feature activation to zero | -- |
| $\text{RR}_{\text{child}}$ | Recovery rate when child feature is zeroed | scalar, $[0, 1]$ |
| $\text{RR}_{\text{ctrl}}$ | Recovery rate when control (random) feature is zeroed | scalar, $[0, 1]$ |
| $\Delta\text{RR}$ | Recovery difference: $\text{RR}_{\text{child}} - \text{RR}_{\text{ctrl}}$ | scalar |

## Hedging Decomposition

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $L_0^{\text{base}}$ | Base SAE sparsity level (e.g., 22) | scalar |
| $L_0^{\text{target}}$ | Target SAE sparsity level for hedging test (e.g., 176) | scalar |
| $H_{\text{strict}}$ | Strict hedging rate: parent feature recovers at $L_0^{\text{target}}$ | scalar, $[0, 1]$ |
| $H_{\text{comp}}$ | Compensatory resolution rate: non-parent features resolve FN | scalar, $[0, 1]$ |
| $H_{\text{persist}}$ | Persistent rate: FN persists at $L_0^{\text{target}}$ | scalar, $[0, 1]$ |

## Absorption Tax (Appendix)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $T(G)$ | Absorption Tax: minimum additional $L_0$ cost for absorption-free representation | scalar |
| $R_{pc}$ | Redundancy ratio between parent $p$ and child $c$ | scalar, $[0, 1]$ |
| $T(G) = \sum_{(p,c) \in E} \pi_c \cdot R_{pc}$ | Absorption Tax formula | scalar |

## Geometric Absorption Score (Appendix)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{GAS}(i \to j)$ | Geometric Absorption Score: decoder-activation co-occurrence mismatch from feature $i$ absorbing into $j$ | scalar |
| $\text{AUROC}$ | Area under the ROC curve for GAS as an absorption classifier | scalar, $[0, 1]$ |

## Statistical Tests

| Symbol | Definition |
|--------|-----------|
| $d$ (Cohen's) | Standardized effect size: $d = (\bar{X}_1 - \bar{X}_2) / s_{\text{pooled}}$ |
| $\rho$ (Spearman) | Rank correlation coefficient |
| CI | Bootstrap 95% confidence interval (10k resamples) |
| $p$ | $p$-value from the specified test (Wilcoxon, Mann-Whitney, permutation, or Kruskal-Wallis) |
