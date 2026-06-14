# 2. Related Work

## 2.1 Classical and Decoupled Weight Decay

Weight decay originates as L2 regularization: adding $(\lambda/2)\|w\|^2$ to the loss produces the gradient update $w_{t+1} = w_t - \gamma_t(g_t + \lambda w_t)$. Loshchilov and Hutter (2019) showed that this coupled formulation is suboptimal in adaptive optimizers like Adam, because the preconditioner rescales the regularization term unevenly across parameters. Their decoupled formulation $w_{t+1} = (1-\gamma_t\lambda)w_t - \gamma_t\hat{g}_t$ (AdamW) separates weight decay from gradient scaling, becoming the standard in transformer training. Ding et al. (2023) extend the convergence analysis to general Adam-family optimizers with decoupled WD. Outmezguine and Levi (2024) generalize decoupled WD to $\ell_p$ norms, enabling sparsification via $p < 1$. Guo and Fan (2025) replace the L2 penalty with a Huber regularizer (AdamHD), bounding decay gradients for large weights.

These works establish the structural separation between gradient updates and weight decay but treat $\lambda$ as a fixed hyperparameter.

## 2.2 Dynamic Weight Decay Schedules

The first practical WD scheduler, SWD (Xie et al., NeurIPS 2023), reduces WD when gradient norms are large to avoid over-regularization, effectively setting $\lambda_{\text{eff}} \propto 1/\|g\|$. Ferbach et al. (2026) propose ADANA, using logarithmic-time schedules for $\beta_1$, $\beta_2$, and WD jointly, reporting 40% compute efficiency gains. Cosine-scheduled and linearly-decayed WD appear frequently in practice, often coupled with matching LR schedules. The WSD (Warmup-Stable-Decay) paradigm, adopted by DeepSeek-V3 among others, applies constant LR and WD during a stable phase before joint decay.

Chou (2025) derives that WD should scale as $\gamma^2$ for stable weight norm maintenance, providing a principled scaling rule that relates WD to the learning rate schedule. Wang and Aitchison (2024) frame WD as an exponential moving average timescale, showing the optimal timescale is constant in epochs across model and dataset scales.

## 2.3 Alignment-Aware and Geometry-Sensitive Methods

CWD (Chen et al., ICLR 2026) conditions WD on the sign alignment between weights and optimizer updates: decay is applied only when $\text{sign}(w_i) = \text{sign}(\Delta w_i)$, implementing a binary mask with a bilevel Pareto-optimal interpretation. The authors connect CWD to sliding-mode control theory. AdamO (Chen, Yuan, and Zhang, 2026) decomposes the update into radial (norm-changing) and tangential (direction-changing) components, eliminating the "radial tug-of-war" conflict between WD and gradient descent. SPD (Tian et al., 2024) modulates per-layer WD by loss reduction consistency for robust fine-tuning.

Sun et al. (CVPR 2025) provide the first theoretical proof that WD improves generalization (not convergence speed) in nonconvex SGD, using a worst-case alignment bound: the generalization gap scales with $\delta_T = \sup_t \cos(g_t, w_t)$. Holzl et al. (NeurIPS 2025) empirically show that gradient-weight alignment (GWA) predicts generalization but lack a formal bound.

## 2.4 Norm-Matched and Spectral Methods

Loshchilov (2023) generalizes decoupled WD to target arbitrary weight norms (AdamWN): the update drives $\|w\|$ toward a target $\tau$ rather than toward zero. He et al. (2025) propose AlphaDecay, assigning per-module WD coefficients guided by heavy-tailed self-regularization (HT-SR) theory and spectral density analysis. Galanti et al. (2022) and Kobayashi et al. (2024) show that WD induces low-rank bias in weight matrices and attention layers respectively, connecting WD to implicit spectral regularization.

## 2.5 Batch Normalization Interaction

The interaction between WD and batch normalization is well-established. For scale-invariant layers (those followed by BN), WD does not directly regularize the effective weights but instead controls the effective learning rate: larger norms reduce the effective LR, and WD prevents norm growth, maintaining a higher effective LR (Kosson et al., 2023). D'Angelo et al. (NeurIPS 2024) demonstrate that WD on BN architectures acts primarily as a "loss stabilization mechanism" rather than as explicit regularization. Defazio (2025) shows WD drives the gradient-to-weight ratio $\|g\|/\|w\|$ to a steady-state equilibrium across all normalized layers.

This body of work suggests that on BN architectures, the specific form of WD scheduling matters less than whether WD is applied at all -- a prediction our certified band analysis formalizes and our experiments confirm.

## 2.6 Optimal Control Perspectives

Optimal control theory has been applied to learning rate scheduling (Naganuma et al., 2026; Ferbach et al., 2026) but not, to our knowledge, to weight decay scheduling. Kondo and Iiduka (2025) apply Lyapunov analysis to dynamic learning rate and batch size selection. CWD's sliding-mode control interpretation (Chen et al., ICLR 2026) is the closest precursor to our control-theoretic framework, but their analysis is restricted to binary WD ($\lambda \in \{0, \lambda_0\}$). We extend the Lyapunov analysis to continuous time-varying schedules and derive the optimal control law via Pontryagin's Maximum Principle.

<!-- FIGURES
- None
-->
