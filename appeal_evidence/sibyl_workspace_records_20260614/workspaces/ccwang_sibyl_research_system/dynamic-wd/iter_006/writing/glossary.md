# Glossary

| Term | Definition |
|------|-----------|
| Phi modulator framework | Unified formulation expressing all WD methods as $\lambda_{\text{eff}}(t) = \phi(t,w,g) \cdot \lambda_{\text{base}}$ |
| Certified band | The interval $[\lambda_{\min}(t), \lambda_{\max}(t)]$ within which any WD schedule guarantees Lyapunov convergence |
| Bang-bang control | Optimal control strategy where the control input switches between extreme values (0 and $\Lambda_{\max}$) |
| PMP-WD | Weight decay schedule derived from Pontryagin's Maximum Principle; exhibits bang-bang behavior |
| Cautious Weight Decay (CWD) | Binary sign-alignment mask: decay only when $\text{sign}(w) = \text{sign}(\text{update})$ |
| Scheduled Weight Decay (SWD) | Gradient-norm-aware dynamic WD (Xie et al., NeurIPS 2023) |
| Budget Equivalence Metric (BEM) | $|\overline{\lambda}_t - \lambda_{\text{const}}| / \lambda_{\text{const}}$; measures deviation of mean effective WD from constant baseline |
| Coupling Stability Index (CSI) | Coefficient of variation of weight norm changes; measures WD-optimization coupling stability |
| Alignment Informativeness Score (AIS) | Entropy of per-layer alignment distribution; quantifies how much alignment varies across layers |
| Decoupled weight decay | WD applied as $w \leftarrow (1-\lambda)w$ independent of gradient scaling (AdamW) |
| Gradient-weight alignment | Cosine similarity $\cos(g, w) = \langle g, w \rangle / (\|g\|\|w\|)$ |
| Lyapunov function | $V_t = f(w_t) + \mu_t \|w_t\|^2$; convergence certificate |
| Costate variable | Adjoint $p(t)$ in PMP; approximated by momentum buffer in SGD |
| Switching function | $\sigma(t) = \langle p(t), w(t) \rangle$; determines PMP-WD decay application |
| Weight decay illusion | The finding that on BN architectures, the certified band is narrow enough that WD method choice has negligible accuracy impact |
| Subsumption | Verification that a WD method's effective $\lambda(t)$ lies within the certified band for $\geq 95\%$ of training steps |
