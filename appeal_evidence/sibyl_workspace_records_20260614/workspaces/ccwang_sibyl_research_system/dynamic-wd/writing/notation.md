# Notation Table

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{\theta}_t$ | Model parameters at step $t$ |
| $\mathbf{g}_t$ | Gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$ at step $t$ |
| $\eta_t$ | Learning rate at step $t$ |
| $\lambda$ | Weight decay coefficient (base, constant) |
| $\lambda_{\mathrm{eff}}(t)$ | Effective weight decay: $\lambda \cdot \mathbb{E}[\varphi(t, \boldsymbol{\theta}, \mathbf{g})]$ |
| $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ | Phi modulator function $\mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ |
| $\hat{\mathbf{m}}_t$ | Bias-corrected first moment estimate (Adam) |
| $\hat{\mathbf{v}}_t$ | Bias-corrected second moment estimate (Adam) |
| $\epsilon$ | Adam stability constant |
| $\beta_1, \beta_2$ | Adam exponential decay rates for moment estimates |
| $\odot$ | Element-wise (Hadamard) product |
| $\mathbf{u}_t$ | Optimizer update direction $\hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ |
| $T$ | Total number of training steps |
| $\tau$ | Target weight norm (AdamWN) |
| $\sigma(\cdot)$ | Sigmoid function |
| $\beta$ | Temperature parameter for soft CWD |
| $h(\cdot)$ | SWD gradient-norm sensitivity function |
| $\boldsymbol{\alpha}_l$ | Per-layer decay coefficient (AlphaDecay) |
| $\kappa(\mathbf{H})$ | Spectral condition number of Hessian $\mathbf{H}$ |
| $\mathrm{CV}(\cdot)$ | Coefficient of variation |
| $\rho_S$ | Spearman rank correlation |
| BEM | Budget Equivalence Metric |
| CSI | Coupling Stability Index |
| AIS | Alignment Informativeness Score |
| WD | Weight Decay |
| CWD | Cautious Weight Decay (Chen et al., 2026a) |
| SWD | Scheduled Weight Decay (Xie et al., 2023) |
| AdamWN | Weight Norm Control (Loshchilov, 2023) |
| BN | Batch Normalization |
