# Glossary

| Term | Definition |
|------|-----------|
| Phi modulator framework | Unified formulation expressing all WD methods via a per-parameter modulation function $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ applied to the decay term |
| Phi Invariance Conjecture | The hypothesis that AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, rendering $\varphi$'s functional form irrelevant to final generalization |
| Cautious Weight Decay (CWD) | Binary sign-alignment mask: decay only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$ (Chen et al., ICLR 2026) |
| Scheduled Weight Decay (SWD) | Gradient-norm-aware dynamic WD (Xie et al., NeurIPS 2023) |
| Weight Norm Control (AdamWN) | Generalization of decoupled WD to target arbitrary weight norm $\tau$ rather than zero (Loshchilov, 2023) |
| AlphaDecay | Module-wise WD guided by heavy-tailed spectral density (He et al., 2025) |
| Budget Equivalence Metric (BEM) | Normalized deviation of a method's mean effective WD from the constant baseline; BEM $\in [0, 1]$ |
| Coupling Stability Index (CSI) | Composite metric measuring stability of WD-optimizer coupling via weight norm CV, spectral condition number, and effective LR CV |
| Alignment Informativeness Score (AIS) | Spearman correlation between gradient-weight alignment and per-step loss improvement; measures whether alignment carries useful signal |
| Decoupled weight decay | WD applied as $\boldsymbol{\theta} \leftarrow (1 - \lambda)\boldsymbol{\theta}$ independent of gradient scaling (AdamW formulation) |
| Gradient-weight alignment | Cosine similarity $\cos(\mathbf{g}, \boldsymbol{\theta}) = \langle \mathbf{g}, \boldsymbol{\theta} \rangle / (\|\mathbf{g}\| \|\boldsymbol{\theta}\|)$ |
| Modulation axis | One of four orthogonal dimensions along which $\varphi$ can vary: temporal, directional, spatial, target-norm |
| Budget equivalence | Two WD strategies are budget-equivalent when their mean effective WD over training is equal |
