# Notation Table

| Symbol | Meaning |
|--------|---------|
| $w_t$ | Model parameters at step $t$ |
| $g_t$ | Gradient $\nabla f(w_t)$ at step $t$ |
| $\gamma_t$ | Learning rate at step $t$ |
| $\lambda$ | Weight decay coefficient (constant) |
| $\lambda_t$ | Time-varying weight decay at step $t$ |
| $\lambda_{\text{eff}}(t)$ | Effective weight decay: $\phi(t,w,g) \cdot \lambda_{\text{base}}$ |
| $\lambda_{\text{base}}$ | Base weight decay coefficient |
| $\phi(t,w,g)$ | Phi modulator function |
| $\Lambda_{\max}$ | Maximum allowed weight decay (PMP-WD bound) |
| $V_t$ | Lyapunov function value at step $t$ |
| $\mu_t$ | Lyapunov weighting coefficient at step $t$ |
| $f(w)$ | Training loss function |
| $L$ | Smoothness constant (Lipschitz constant of gradient) |
| $\delta_t$ | Gradient-weight alignment: $\cos(g_t, w_t)$ |
| $\bar{\delta}_T$ | Cumulative alignment: $(1/T)\sum_t \delta_t$ |
| $\delta_T^{\sup}$ | Worst-case alignment: $\sup_t \delta_t$ |
| $n_l(t)$ | Weight norm of layer $l$: $\|w_l\|$ |
| $r_l(t)$ | Gradient-to-weight ratio of layer $l$: $\|g_l\|/\|w_l\|$ |
| $p(t)$ | Costate (adjoint) variable in PMP formulation |
| $\sigma(t)$ | Switching function: $\langle p(t), w(t) \rangle$ |
| $H(w,p,\lambda)$ | Hamiltonian in optimal control formulation |
| BEM | Budget Equivalence Metric |
| CSI | Coupling Stability Index |
| AIS | Alignment Informativeness Score |
| BN | Batch Normalization |
| WD | Weight Decay |
| PMP | Pontryagin's Maximum Principle |
| CWD | Cautious Weight Decay |
| SWD | Scheduled Weight Decay |
