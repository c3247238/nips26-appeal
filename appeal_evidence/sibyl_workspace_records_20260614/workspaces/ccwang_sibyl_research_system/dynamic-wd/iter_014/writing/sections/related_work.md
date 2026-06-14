# 2. Background and Related Work

This section surveys the four WD sub-traditions in detail, identifies the gradient-to-weight ratio $\rho_t$ as their shared underlying quantity, and positions our work relative to other unified optimizer perspectives.

## 2.1 WD Scheduling

WD scheduling modulates $\lambda_t$ as a function of training progress or gradient statistics, without per-layer differentiation.

**SWD** (Xie et al., NeurIPS 2023) identifies that fixed WD causes gradient-norm spikes during learning rate decay, harming generalization.  SWD senses $\|\nabla \mathcal{L}\|$ to modulate $\lambda_t$, reducing decay when gradients are large.  In our framework, this corresponds to proportional control: $\lambda_t$ reacts to the current gradient magnitude --- a proxy for the control error $e_t = \rho_t - \rho^*(t)$.  SWD achieves 90.39 $\pm$ 0.19% on CIFAR-10/ResNet-20 in our experiments, close to FixedWD (90.68 $\pm$ 0.11%).

**ADANA** (Ferbach et al., 2026) extends scheduling to logarithmic-time decay of $\lambda$, $\beta_1$, and $\beta_2$ simultaneously, achieving a 40% compute efficiency gain on language modeling tasks.  The log-time WD schedule is a specific monotonic trajectory $\lambda(t)$ that does not sense any per-layer feedback signal.

**Defazio corrective WD** (Defazio, 2025) compensates for the interaction between WD and learning rate schedules by applying a corrective term proportional to the current learning rate ratio $\eta_t / \eta_0$.  Under cosine annealing, this produces monotonically decreasing effective WD.  The corrective term can be viewed as a feedforward compensation signal in the control framework, operating on $\rho^*(t)$ rather than on the error signal.

## 2.2 Alignment-Aware WD

Alignment-aware methods condition WD on the geometric relationship between gradient and weight vectors.

**CWD** (Chen et al., ICLR 2026) applies a binary mask: WD is applied only when $\text{sign}(g_t) = \text{sign}(w_t)$ element-wise, meaning the gradient would move weights *away* from zero.  CWD achieves +0.61% on ImageNet ViT-S/16 over standard AdamW.  In our experiments, CWD's effective WD is approximately 50% of FixedWD ($\lambda_{\text{eff}} \approx 5 \times 10^{-5}$ vs. $10^{-4}$), raising a confound: does CWD improve accuracy through alignment information or simply through magnitude reduction?

**GWA** (Holzl et al., NeurIPS 2025) quantifies per-sample gradient-weight coherence as a generalization proxy.  While GWA does not directly modulate WD, it provides the theoretical basis for using $\alpha_t$ as a feedback signal: higher alignment coherence predicts better generalization.

**AdamO** (Chen, Yuan, and Zhang, 2026) decomposes the optimizer update into radial (norm-changing) and tangential (direction-changing) components, identifying the "Radial Tug-of-War" between WD and gradient.  AdamO resolves this by decoupling the two dynamics, applying SGD-style norm control alongside Adam-style tangential updates.  This is an alignment-aware structural intervention rather than WD coefficient modulation.

## 2.3 Decoupled WD

The decoupled WD tradition addresses the mathematical distinction between $L_2$ regularization and WD in adaptive optimizers.

**AdamW** (Loshchilov & Hutter, ICLR 2019) demonstrated that in Adam, adding $\lambda w$ to the gradient (L2 regularization) is not equivalent to subtracting $\eta \lambda w$ from the parameter update (WD).  The adaptive second-moment scaling in Adam distorts the effective regularization strength per parameter.  AdamW decouples the two, becoming the standard optimizer for transformer training.  Zhang et al. (2018) independently identified three distinct mechanisms through which WD improves generalization, confirming that WD consistently outperforms L2 across SGD, Adam, and K-FAC.

**Scaling rules.**  Wang & Aitchison (ICML 2025) recast optimal WD as an EMA timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$ that remains constant across model and dataset scales.  Chou (2025) derives WD proportional to $\gamma^2$ for stable weight norms.  Kosson et al. (2025) show that WD stabilizes update dynamics across widths more effectively than maximal update parameterization ($\mu$P).  These scaling results provide the basis for the target trajectory $\rho^*(t) = \eta_t / \tau$ in our framework.

## 2.4 Norm-Matched WD

Norm-matched methods drive weight norms toward explicit targets rather than zero.

**CPR** (Franke et al., NeurIPS 2024) formulates WD as a per-parameter-matrix constraint optimization problem via augmented Lagrangian.  When weight norms exceed the target, the Lagrange multiplier accumulates, producing progressively larger effective WD.  In our experiments, CPR's effective WD reaches $10 \times$ the baseline ($\lambda_{\text{eff}} \approx 10^{-3}$ vs. $10^{-4}$), a direct manifestation of integral control.  CPR achieves 91.74 $\pm$ 0.07% on CIFAR-10/ResNet-20, the highest among all methods, and 74.74 $\pm$ 0.05% on ImageNet/ResNet-50.

**AdamWN** (Loshchilov, 2023) generalizes decoupled WD to target arbitrary weight norms (target = 0 recovers standard WD).  **AlphaDecay** (He et al., 2025) assigns module-wise WD coefficients guided by spectral heavy-tailedness (HT-SR theory), scaling from 60M to 1B parameters.

## 2.5 Theoretical Foundations

**Defazio's gradient-to-weight ratio analysis** (2025) proves that WD drives $\rho_t = \|g_t\| / \|w_t\|$ to a steady state for all normalized layers, providing a clean explanation for the Adam vs. AdamW performance gap.  All normalized layers converge to the same $\rho^*$, a "layer balancing" effect.  This is the key insight that identifies $\rho_t$ as the natural control variable for WD.

**Sun et al.'s nonconvex convergence theory** (CVPR 2025) proves that WD does not accelerate convergence (the convergence rate remains $O(1/\sqrt{T})$ regardless of $\lambda$) but strictly improves generalization through an alignment-dependent mechanism.  The generalization bound depends on the cumulative alignment-weighted contraction $A_T$ rather than the worst-case alignment.  We extend this framework to time-varying $\lambda_t$ with alignment modulation in Section 4.

**D'Angelo et al.** (NeurIPS 2024) provide a unifying empirical perspective: WD is never useful as explicit regularization but instead acts as a training dynamics modifier through the "loss stabilization mechanism" for SGD and the "bias-variance tradeoff" for near-one-epoch LLM training.

## 2.6 Distinction from PIDAO

PIDAO (Nature Communications 2024) applies PID control to the optimizer step itself --- the gradient update direction and magnitude.  Our work applies PID control to the WD coefficient $\lambda_t$ --- a different control target entirely.  PIDAO optimizes *how the gradient is used*; we optimize *how much regularization is applied*.  The two approaches are complementary: PIDAO could be combined with our WD coefficient control for joint optimization.

## 2.7 The Shared Control Variable

The four sub-traditions, despite their different formulations, all implicitly manipulate $\rho_t$:

- **Scheduling methods** (SWD, ADANA, Defazio corrective) adjust $\lambda_t$, which directly modulates the WD force on $\|w_t\|$ and thus shifts the $\rho_t$ steady state.
- **Alignment-aware methods** (CWD) selectively disable WD for parameters where $\alpha_t > 0$, changing the *effective* $\lambda_t$ per layer and thus the realized $\rho_t$ trajectory.
- **Decoupled methods** (AdamW, AdamO) ensure that $\lambda$ acts on parameter norms without distortion from adaptive scaling, preserving the theoretical $\rho_t$ dynamics.
- **Norm-matched methods** (CPR, AdamWN) enforce explicit norm targets, which is equivalent to specifying a target $\rho^*$ via the relationship $\rho^* \propto \lambda / \|w^*\|$.

Section 3 formalizes this observation into a unified control law.

<!-- FIGURES
- None
-->
