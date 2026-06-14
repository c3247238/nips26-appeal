# Notation Table

All mathematical symbols used in the paper, organized by category.
Section writers, critics, and editors must reference this file for consistency.

---

## Model and Training

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\theta_t$ | Model parameters at step $t$ | $\theta_t \in \mathbb{R}^d$ |
| $d$ | Total number of trainable parameters | scalar |
| $L$ | Number of layers | scalar |
| $\theta_{l,t}$ | Parameters of layer $l$ at step $t$ | $\theta_{l,t} \in \mathbb{R}^{d_l}$ |
| $g_t$ | Gradient $\nabla_\theta \mathcal{L}(\theta_t)$ at step $t$ | $g_t \in \mathbb{R}^d$ |
| $g_{l,t}$ | Gradient of layer $l$ at step $t$ | $g_{l,t} \in \mathbb{R}^{d_l}$ |
| $\eta_t$ | Learning rate at step $t$ | scalar |
| $T$ | Total training steps (or epochs, context-dependent) | scalar |
| $n$ | Training set size | scalar |
| $b$ | Mini-batch size | scalar |

## AdamW Optimizer

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $m_t$ | First moment estimate (momentum) | $m_t \in \mathbb{R}^d$ |
| $v_t$ | Second moment estimate (adaptive LR) | $v_t \in \mathbb{R}^d$ |
| $\hat{m}_t$ | Bias-corrected first moment | $\hat{m}_t \in \mathbb{R}^d$ |
| $\hat{v}_t$ | Bias-corrected second moment | $\hat{v}_t \in \mathbb{R}^d$ |
| $\beta_1, \beta_2$ | Exponential decay rates for moments | scalars, default $(0.9, 0.999)$ |
| $\varepsilon$ | Numerical stability constant | scalar, default $10^{-8}$ |
| $u_t$ | AdamW update direction $\hat{m}_t / (\hat{v}_t^{1/2} + \varepsilon)$ | $u_t \in \mathbb{R}^d$ |

## Weight Decay

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\lambda$ | Base weight decay coefficient | scalar, default $5 \times 10^{-4}$ |
| $\phi(t, \theta, g)$ | Phi modulator function | $\phi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ |
| $\lambda_{\text{eff}}(t)$ | Effective WD at step $t$: $\lambda \cdot \mathbb{E}_\theta[\phi(t, \theta, g)]$ | scalar |
| $\bar{\lambda}$ | Time-averaged effective WD: $\frac{1}{T}\int_0^T \lambda_{\text{eff}}(t)\,dt$ | scalar |
| $\lambda^*(t)$ | Optimal WD schedule (Theorem 3 / PMP-WD) | scalar |
| $\lambda_{\max}$ | Upper clipping bound for PMP-WD | scalar |

## Key Ratios and Alignment

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\rho_t$ | Gradient-to-weight ratio: $\|g_t\| / \|\theta_t\|$ | scalar |
| $\rho_{l,t}$ | Per-layer ratio: $\|g_{l,t}\| / \|\theta_{l,t}\|$ | scalar |
| $\hat{\rho}_t$ | EMA-smoothed ratio (momentum 0.9) | scalar |
| $\rho^*$ | Target steady-state ratio (from Defazio 2025) | scalar, $\rho^* \approx \sqrt{2\lambda / \gamma}$ |
| $\hat{\delta}_t$ | Cosine similarity: $\cos(g_t, \theta_t) = \frac{g_t \cdot \theta_t}{\|g_t\|\|\theta_t\|}$ | scalar, $\hat{\delta}_t \in [-1, 1]$ |
| $\hat{\delta}_{l,t}$ | Per-layer cosine similarity | scalar |

## Diagnostic Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| BEM | Budget Equivalence Metric: $\|\lambda_{\text{eff}}^{\text{method}} - \lambda_{\text{eff}}^{\text{constant}}\| / \lambda_{\text{eff}}^{\text{constant}}$ | $[0, 1]$; 0 = matched budget |
| CSI | Coupling Stability Index: $w_1 \cdot \text{CV}(\|\theta\|_{\text{traj}}) + w_2 \cdot \log\kappa(H) + w_3 \cdot \text{CV}(\eta_{\text{eff, layers}})$ | $[0, \infty)$; higher = more unstable |
| $\text{CSI}_{\text{param}}$ | Per-parameter CSI (used in Theorem 2 bound) | $[0, \infty)$ |
| AIS | Alignment Informativeness Score: $\text{Spearman}_\rho(\cos(\theta_i, g_i), \Delta\text{loss}_i)$ | $[0, 1]$; AIS > 0.2 = informative |
| $w_1, w_2, w_3$ | CSI component weights | $(0.4, 0.3, 0.3)$ |

## Theorem 1 Quantities

| Symbol | Definition | Context |
|--------|-----------|---------|
| $C$ | Constant depending on architecture and loss Lipschitz | Theorem 1 |
| $\sigma^2$ | Gradient noise variance | Theorem 1 |
| $\Delta\text{CSI}$ | CSI perturbation from binary masking: $\text{CSI}(\phi_{\text{CWD}}) - \text{CSI}(\phi_{\text{const}})$ | Theorem 1 |
| $\text{AIS}^*$ | Threshold AIS for adaptive WD to be beneficial: $(C\sigma^2/n) \cdot \Delta\text{CSI} / \bar{\lambda}$ | Theorem 1 corollary |

## Theorem 3 / PMP-WD Quantities

| Symbol | Definition | Context |
|--------|-----------|---------|
| $\kappa$ | Feedback gain from Riccati equation | Theorem 3, default $\kappa = 1$ |
| $\gamma$ | AdamW step size (effective per-layer LR) | Defazio 2025 |
| $\beta_0$ | QA-WD proportionality constant ($\lambda^* = \beta_0 \cdot \hat{\delta}^2_t$) | Remark 3.1 (RG derivation) |

## Proposition 1 Quantities

| Symbol | Definition | Context |
|--------|-----------|---------|
| CV | Coefficient of variation: $\text{std}(X) / \text{mean}(X)$ | Proposition 1 |
| $k$ | Minimum EMA aggregation horizon (steps) | Proposition 1, $k \geq 10$ |

## Statistical Testing

| Symbol | Definition | Context |
|--------|-----------|---------|
| $p$ | p-value from paired t-test or TOST | Tables 2-3 |
| Cohen's $d$ | Standardized mean difference (pooled std) | Effect size |
| $\Delta_{\text{TOST}}$ | Equivalence margin for TOST | $\pm 0.3\%$ or $\pm 1\%$ |

## Experimental Quantities

| Symbol | Definition | Values |
|--------|-----------|--------|
| $\Phi_{\text{spread}}$ | Max accuracy - min accuracy across methods | percentage points |
| acc | Best test accuracy over training | percentage |
| gen gap | Train accuracy - test accuracy at final epoch | percentage points |

---

*Version: 2.0 | Iteration: 7 | Generated: 2026-03-18*
