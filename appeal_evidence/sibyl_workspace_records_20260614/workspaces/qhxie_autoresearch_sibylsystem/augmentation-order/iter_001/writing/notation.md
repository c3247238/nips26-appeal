# Notation Table

All mathematical symbols and notation used in this paper.

## Data and Distributions

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{D}$ | Training dataset | $\{(x_i, y_i)\}_{i=1}^n$ |
| $\mu$ | Data distribution over images | $\mu \in \mathcal{P}(\mathbb{R}^{C \times H \times W})$ |
| $x$ | Input image | $x \in \mathbb{R}^{C \times H \times W}$ (C=3, H=W=32 for CIFAR) |
| $y$ | Class label | $y \in \{1, \ldots, K\}$ (K=10 or K=100) |
| $n$ | Training set size | Scalar |

## Transforms and Composition

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $t_i$ | $i$-th augmentation operation (stochastic channel) | $t_i : \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H \times W}$ |
| $\sigma$ | Permutation of $\{1, \ldots, K_{ops}\}$ specifying operation ordering | $\sigma \in S_{K_{ops}}$ |
| $A_\sigma$ | Composed augmentation pipeline under permutation $\sigma$ | $A_\sigma = t_{\sigma(K_{ops})} \circ \cdots \circ t_{\sigma(1)}$ |
| $t_i \circ t_j$ | Composition: apply $t_j$ first, then $t_i$ | $\mathbb{R}^{C \times H \times W} \to \mathbb{R}^{C \times H \times W}$ |
| $K_{ops}$ | Number of augmentation operations in the pipeline | Scalar (3 for Tier 1, 6 for Tier 2) |
| $\#$ | Pushforward operator | $t \# \mu$ = distribution of $t(x)$ when $x \sim \mu$ |

## Non-Commutativity Measure

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $W_2(\cdot, \cdot)$ | 2-Wasserstein distance between distributions | Scalar $\geq 0$ |
| $\text{NC}_2(t_i, t_j; \mu)$ | Wasserstein Non-Commutativity of transforms $t_i, t_j$ under $\mu$ | Scalar $\geq 0$; defined as $W_2(t_i \circ t_j \# \mu,\; t_j \circ t_i \# \mu)$ |
| $\text{SWD}(\cdot, \cdot)$ | Sliced Wasserstein Distance (tractable proxy for $W_2$) | Scalar $\geq 0$ |
| $L$ | Lipschitz constant of a transform | Scalar $\geq 1$ |

## Information-Theoretic Quantities

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $I(y; x)$ | Mutual information between labels and images | Scalar $\geq 0$ (nats) |
| $I(y; A_\sigma(x))$ | Mutual information between labels and augmented images under ordering $\sigma$ | Scalar $\geq 0$ |
| $\eta_i$ | Contraction coefficient of channel $t_i$ | $\eta_i \in [0, 1]$ |
| $\hat{I}_{\text{NCE}}$ | InfoNCE lower bound on mutual information | Scalar $\geq 0$ |

## Generalization

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{gen}(\sigma)$ | Generalization gap under ordering $\sigma$ | Scalar; $\text{gen}(\sigma) = R(\sigma) - \hat{R}(\sigma)$ |
| $R(\sigma)$ | Population risk under ordering $\sigma$ | Scalar $\geq 0$ |
| $\hat{R}(\sigma)$ | Empirical risk under ordering $\sigma$ | Scalar $\geq 0$ |

## Metrics and Statistics

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{acc}_{\sigma, a, d}$ | Top-1 accuracy for ordering $\sigma$, architecture $a$, dataset $d$ | Scalar $\in [0, 1]$ |
| $\Delta_{\max}$ | Max-min accuracy spread across orderings | $\Delta_{\max} = \max_\sigma \text{acc}_\sigma - \min_\sigma \text{acc}_\sigma$ |
| $\rho_s$ | Spearman rank correlation coefficient | Scalar $\in [-1, 1]$ |
| $d_{\text{Cohen}}$ | Cohen's $d$ effect size | Scalar |
| $\eta^2_p$ | Partial eta-squared (ANOVA effect size) | Scalar $\in [0, 1]$ |
| $M$ | Augmentation magnitude level | Scalar $\in \{5, 9, 14\}$ (RandAugment scale) |

## Architecture Notation

| Symbol | Definition |
|--------|-----------|
| ResNet-18 | Residual network, 18 layers, 11M parameters |
| ViT-S/4 | Vision Transformer, Small variant, patch size 4, 22M parameters |

## Ordering Labels (Shorthand)

| Label | Meaning |
|-------|---------|
| Crop | `RandomCrop(32, padding=4)` |
| Flip | `RandomHorizontalFlip(p=0.5)` |
| CJ | `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)` |
| G | Geometric transform category (Crop, Flip, Rotation) |
| P | Photometric transform category (ColorJitter, Grayscale, GaussianBlur) |
