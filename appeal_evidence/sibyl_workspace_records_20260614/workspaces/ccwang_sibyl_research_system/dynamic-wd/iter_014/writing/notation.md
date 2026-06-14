# Notation Table

All mathematical symbols used in the paper. Section writers must reference this file for consistency.

## Training Variables

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $t$ | Training step index | $t \in \{1, 2, \ldots, T\}$ |
| $T$ | Total number of training steps | $T \in \mathbb{N}$ |
| $l$ | Layer index | $l \in \{1, 2, \ldots, L\}$ |
| $L$ | Total number of parameter groups (layers) | $L \in \mathbb{N}$ |
| $b$ | Mini-batch size | $b \in \mathbb{N}$ |

## Parameters and Gradients

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $w_t^l$ | Parameters of layer $l$ at step $t$ | $w_t^l \in \mathbb{R}^{d_l}$ |
| $g_t^l$ | Stochastic gradient for layer $l$ at step $t$ | $g_t^l \in \mathbb{R}^{d_l}$ |
| $d_l$ | Dimensionality of layer $l$ parameters | $d_l \in \mathbb{N}$ |
| $\|w_t^l\|$ | $\ell_2$ norm of layer $l$ parameters at step $t$ | $\|w_t^l\| = \|w_t^l\|_2$ |
| $\|g_t^l\|$ | $\ell_2$ norm of layer $l$ gradient at step $t$ | $\|g_t^l\| = \|g_t^l\|_2$ |

## Learning Rate and Weight Decay

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $\eta_t$ | Learning rate at step $t$ | $\eta_t \in \mathbb{R}_{>0}$ |
| $\eta_0$ | Initial learning rate | $\eta_0 \in \mathbb{R}_{>0}$ |
| $\lambda_t^l$ | Effective weight decay coefficient for layer $l$ at step $t$ | $\lambda_t^l \in \mathbb{R}_{\geq 0}$ |
| $\lambda_{\text{base}}$ | Base (user-specified) weight decay coefficient | $\lambda_{\text{base}} \in \mathbb{R}_{>0}$ |
| $\lambda_0$ | Initial weight decay coefficient (= $\lambda_{\text{base}}$) | $\lambda_0 = \lambda_{\text{base}}$ |
| $\tau$ | EMA timescale of weight decay | $\tau = 1 / (\lambda_0 \cdot \eta_0)$ |

## Control Variables

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $\rho_t^l$ | Gradient-to-weight ratio for layer $l$ at step $t$ | $\rho_t^l = \|g_t^l\| / (\|w_t^l\| + \epsilon)$ |
| $\rho^*(t)$ | Target (prescribed) gradient-to-weight ratio trajectory | $\rho^*(t) = \eta_t / \tau$ |
| $e_t^l$ | Per-layer control error at step $t$ | $e_t^l = \rho_t^l - \rho^*(t)$ |
| $\alpha_t^l$ | Gradient-weight alignment cosine for layer $l$ at step $t$ | $\alpha_t^l = \langle g_t^l, w_t^l \rangle / (\|g_t^l\| \cdot \|w_t^l\|) \in [-1, 1]$ |
| $\delta_t^l$ | Alignment signal (alternative notation for $\alpha_t^l$ in theoretical sections) | $\delta_t^l \in [-1, 1]$ |
| $\hat{\delta}_t$ | Estimated (finite-batch) alignment | $\hat{\delta}_t \in [-1, 1]$ |

## PID Gains

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $K_p$ | Proportional gain | $K_p \in \mathbb{R}_{\geq 0}$ |
| $K_i$ | Integral gain | $K_i \in \mathbb{R}_{\geq 0}$ |
| $K_d$ | Derivative/alignment gain | $K_d \in \mathbb{R}_{\geq 0}$ |

## Unified Control Law

$$\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + K_i \cdot \text{EMA}(e_t^l) - K_d \cdot \alpha_t^l \cdot e_t^l$$

## UDWDC-Specific

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $\beta$ | EMA smoothing coefficient for $\rho_t^l$ (UDWDC-v2) | $\beta = 0.99$ |
| $\lambda_{\min}$ | Floor clipping threshold for effective WD (UDWDC-v2) | $\lambda_{\min} = 0.1 \cdot \lambda_{\text{base}}$ |

## Evaluation Metrics

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $\text{BEM}$ | Budget Equivalence Metric | $\text{BEM} = (\text{acc} - \text{acc}_{\text{NoWD}}) / \text{TotalWDBudget}$ |
| $\text{TotalWDBudget}$ | Cumulative WD budget | $\sum_{t=1}^{T} \sum_{l=1}^{L} \lambda_t^l \|w_t^l\|^2$ |
| $\text{CSI}$ | Coupling Stability Index | $\text{CSI} = 1 / (1 + \text{Var}_t[\lambda_{\text{eff}}^l / \text{mean}_t[\lambda_{\text{eff}}^l]])$, averaged across last 25% of training. FixedWD normalized to 1.0 as reference. |
| $\text{CSI}_{\text{temporal}}$ | Temporal component of CSI | Within-layer stability over training |
| $\text{CSI}_{\text{spatial}}$ | Spatial component of CSI | Cross-layer consistency at each step |
| $\text{CSI}_{\text{combined}}$ | Combined CSI | $(\text{CSI}_{\text{temporal}} + \text{CSI}_{\text{spatial}}) / 2$ |
| $\text{AIS}$ | Alignment Informativeness Score | Spearman $\rho(\bar{\alpha}_t^l, \Delta\text{GenGap}_t)$ where $\Delta\text{GenGap}_t = \text{GenGap}_t - \text{GenGap}_{t-1}$ is the per-epoch change in generalization gap |
| $\text{SNR}$ | Signal-to-noise ratio of alignment signal | $\text{SNR} = |\mathbb{E}[\alpha_t]| / \text{Std}[\alpha_t]$ |

## Theoretical Analysis

| Symbol | Definition | Domain / Dimensionality |
|--------|-----------|------------------------|
| $\mathcal{L}$ | Loss function | $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ |
| $L_{\text{smooth}}$ | Lipschitz smoothness constant | $L_{\text{smooth}} \in \mathbb{R}_{>0}$ |
| $C_T$ | Cumulative contraction over $T$ steps | $C_T = \sum_{t=1}^{T} \eta_t \lambda_t \|w_t\|^2$ |
| $A_T$ | Alignment-weighted contraction | $A_T = \sum_{t=1}^{T} \eta_t \lambda_t \|w_t\|^2 \cdot \phi(\delta_t)$ |
| $\phi(\cdot)$ | Alignment-weighting function | $\phi: [-1, 1] \to \mathbb{R}_{\geq 0}$ |
| $r_l^*$ | Per-layer steady-state gradient-to-weight ratio | $r_l^* = \lambda_{\text{base}} \cdot \gamma / \phi(\delta_l^*)$ |
| $\delta_l^*$ | Per-layer steady-state alignment | $\delta_l^* \in [-1, 1]$ |
| $\gamma$ | Normalization-related constant (depends on layer type) | $\gamma \in \mathbb{R}_{>0}$ |
| $P_t$ | Adam preconditioner (diagonal second-moment estimate) | $P_t \in \mathbb{R}^{d \times d}$ (diagonal) |
| $\delta_t^P$ | Geometry-corrected alignment for Adam | $\delta_t^P = \langle P_t^{-1} g_t, w_t \rangle / (\|P_t^{-1} g_t\| \cdot \|w_t\|)$ |

## Constants

| Symbol | Definition | Value |
|--------|-----------|-------|
| $\epsilon$ | Numerical stability constant in $\rho_t^l$ | $\epsilon = 10^{-8}$ |

## Conventions

- Norms are $\ell_2$ unless otherwise specified
- $\langle \cdot, \cdot \rangle$ denotes the standard inner product
- $\text{EMA}(x_t, \beta) = \beta \cdot \text{EMA}(x_{t-1}, \beta) + (1-\beta) \cdot x_t$
- Subscript $t$ denotes training step; superscript $l$ denotes layer index
- "WD" always refers to weight decay
- "BN" refers to Batch Normalization; "LN" refers to Layer Normalization
