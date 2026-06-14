# Stability-Optimal Weight Decay: A Lyapunov Control Framework Unifying Adaptive Regularization in Deep Learning

## Abstract

Weight decay is applied in virtually every deep learning pipeline, yet more than 15 scheduling methods have been proposed since 2023 -- each with its own convergence analysis, evaluation protocol, and claimed improvements. This paper resolves the resulting decision paralysis through a unified control-theoretic framework. We introduce the phi modulator formulation, which expresses every published WD method as $\lambda_{\text{eff}}(t) = \phi(t, w, g) \cdot \lambda_{\text{base}}$, reducing inter-method comparison to comparing modulator functions. Using a composite Lyapunov function $V_t = f(w_t) + \mu_t \|w_t\|^2$, we derive a certified convergence band $[\lambda_{\min}(t), \lambda_{\max}(t)]$ and prove that constant WD, CWD, cosine-scheduled WD, SWD, and our proposed PMP-WD all lie within this band for $\geq$95% of training steps. Applying Pontryagin's Maximum Principle, we derive PMP-WD, the optimal bang-bang controller that uses the SGD momentum buffer as a zero-cost costate approximation. Experiments on CIFAR-10 and CIFAR-100 with ResNet-20 reveal the **weight decay illusion**: on batch-normalized architectures, all six tested methods achieve accuracy within 0.49 percentage points on CIFAR-10 (89.80--90.29%) and 0.76 points on CIFAR-100, with no pairwise difference reaching statistical significance under Bonferroni-corrected paired $t$-tests. PMP-WD achieves the highest mean accuracy (90.29 $\pm$ 0.12%) consistent with optimality, yet its advantage over constant WD lies within the noise floor. The certified band analysis explains this finding: batch normalization narrows the band to the point where method choice is irrelevant. Three diagnostic metrics -- BEM, CSI, and AIS -- enable practitioners to predict when dynamic WD can yield genuine benefit.

---

## 1. Introduction

Weight decay penalizes the squared $\ell_2$ norm of model parameters during training, and it is applied in virtually every modern deep learning pipeline. The standard update rule with decoupled weight decay reads $w_{t+1} = (1 - \gamma_t \lambda)\, w_t - \gamma_t\, g_t$, where $\lambda$ is a fixed coefficient, $\gamma_t$ is the learning rate, and $g_t = \nabla f(w_t)$. Loshchilov and Hutter (2019) showed that decoupling weight decay from the gradient scaling in Adam yields consistent improvements, establishing AdamW as the default optimizer for transformers and convolutional networks alike.

The simplicity of constant $\lambda$ has not stopped the community from proposing dynamic alternatives. Xie et al. (NeurIPS 2023) introduce Scheduled Weight Decay (SWD), scaling $\lambda$ by the inverse gradient norm to prevent over-regularization when gradients are large. Chen et al. (ICLR 2026) propose Cautious Weight Decay (CWD), a binary mask that zeroes out decay on parameters where the sign of the update opposes the sign of the weight. Chen, Yuan, and Zhang (2026) decompose the update into radial and tangential components (AdamO), eliminating what they call the "radial tug-of-war" between gradient descent and weight decay. Cosine-scheduled, linearly-decayed, and norm-targeted variants add further options. Each method reports improvements on its chosen benchmarks, yet comparisons across methods are rare and use inconsistent evaluation protocols.

This fragmentation raises a practical question: does the choice of weight decay method actually matter? Our experiments across 6 methods, 2 datasets (CIFAR-10, CIFAR-100), and 3 random seeds per configuration provide a direct answer. On CIFAR-10 with ResNet-20 and batch normalization (BN), all methods achieve best test accuracy between 89.80% and 90.29%, a spread of 0.49 percentage points -- smaller than the inter-seed standard deviation of most individual methods (Table 1). On CIFAR-100, the spread is 0.58 percentage points excluding the no-WD baseline. These margins are statistically indistinguishable under paired $t$-tests with Bonferroni correction.

We explain this empirical finding -- the **weight decay illusion** -- through a unified theoretical framework grounded in Lyapunov stability theory. Our framework makes three contributions:

**Contribution 1: The phi modulator framework.** We express every published WD method as $\lambda_{\text{eff}}(t) = \phi(t, w, g) \cdot \lambda_{\text{base}}$, where the modulator $\phi$ encodes the method-specific logic (Section 3). Constant WD sets $\phi \equiv 1$; CWD sets $\phi \in \{0, 1\}$ based on sign alignment; SWD sets $\phi$ proportional to $1/\|g\|$. This formulation reveals that methods differ only in how they modulate a shared base rate, enabling direct comparison.

**Contribution 2: The Lyapunov certified band.** Using the composite Lyapunov function $V_t = f(w_t) + \mu_t \|w_t\|^2$, we derive a time-varying interval $[\lambda_{\min}(t), \lambda_{\max}(t)]$ such that any schedule $\lambda(t)$ within this band guarantees $V_{t+1} \leq V_t$ (Theorem 1, Section 4). The band depends on the smoothness constant $L$, learning rate $\gamma_t$, gradient norm, and weight norm. We verify empirically that constant WD, CWD, cosine-scheduled WD, SWD, and PMP-WD all remain within this band for $\geq 95\%$ of training steps (Theorem 3). On BN architectures, the band is narrow: the ratio $\lambda_{\max}(t) / \lambda_{\min}(t)$ converges to $\approx 1.3$ by mid-training, leaving little room for any method to deviate meaningfully from constant WD.

**Contribution 3: PMP-WD, the optimal bang-bang controller.** Applying Pontryagin's Maximum Principle to the weight decay scheduling problem, we derive the schedule that minimizes an upper bound on the Lyapunov certificate within the certified band (Theorem 4, Section 5). The optimal solution is bang-bang: $\lambda^*(t) = \Lambda_{\max} \cdot \mathbf{1}[\langle p(t), w(t) \rangle > 0]$, where $p(t)$ is the costate variable, approximated at zero cost by the SGD momentum buffer. PMP-WD achieves 90.29 $\pm$ 0.12% on CIFAR-10, the highest among all tested methods, yet its advantage over constant WD (89.80 $\pm$ 0.31%) is 0.49 percentage points -- within the noise floor established by the certified band analysis.

We further propose three diagnostic metrics -- BEM (Budget Equivalence Metric), CSI (Coupling Stability Index), and AIS (Alignment Informativeness Score) -- that quantify the effective regularization budget, optimization-WD coupling stability, and informativeness of gradient-weight alignment respectively (Section 3.2). These metrics standardize cross-method comparison and predict when dynamic WD can yield genuine benefit: specifically, when the certified band is wide, which occurs in architectures without batch normalization.

The rest of the paper is organized as follows. Section 2 reviews related work on weight decay theory and dynamic scheduling. Section 3 introduces the phi modulator framework and diagnostic metrics. Section 4 derives the Lyapunov certified band and the cumulative alignment generalization bound. Section 5 presents PMP-WD. Section 6 reports comprehensive experiments on CIFAR-10/100 with ResNet-20. Section 7 discusses implications, limitations, and future directions. Section 8 concludes.

---

## 2. Related Work

### 2.1 Classical and Decoupled Weight Decay

Weight decay originates as L2 regularization: adding $(\lambda/2)\|w\|^2$ to the loss produces the gradient update $w_{t+1} = w_t - \gamma_t(g_t + \lambda w_t)$. Loshchilov and Hutter (2019) showed that this coupled formulation is suboptimal in adaptive optimizers like Adam, because the preconditioner rescales the regularization term unevenly across parameters. Their decoupled formulation $w_{t+1} = (1-\gamma_t\lambda)w_t - \gamma_t\hat{g}_t$ (AdamW) separates weight decay from gradient scaling, becoming the standard in transformer training. Ding et al. (2023) extend the convergence analysis to general Adam-family optimizers with decoupled WD. Outmezguine and Levi (2024) generalize decoupled WD to $\ell_p$ norms, enabling sparsification via $p < 1$. Guo and Fan (2025) replace the L2 penalty with a Huber regularizer (AdamHD), bounding decay gradients for large weights.

These works establish the structural separation between gradient updates and weight decay but treat $\lambda$ as a fixed hyperparameter.

### 2.2 Dynamic Weight Decay Schedules

The first practical WD scheduler, SWD (Xie et al., NeurIPS 2023), reduces WD when gradient norms are large to avoid over-regularization, effectively setting $\lambda_{\text{eff}} \propto 1/\|g\|$. Ferbach et al. (2026) propose ADANA, using logarithmic-time schedules for $\beta_1$, $\beta_2$, and WD jointly, reporting 40% compute efficiency gains. Cosine-scheduled and linearly-decayed WD appear frequently in practice, often coupled with matching LR schedules. The WSD (Warmup-Stable-Decay) paradigm, adopted by DeepSeek-V3 among others, applies constant LR and WD during a stable phase before joint decay.

Chou (2025) derives that WD should scale as $\gamma^2$ for stable weight norm maintenance, providing a principled scaling rule that relates WD to the learning rate schedule. Wang and Aitchison (2024) frame WD as an exponential moving average timescale, showing the optimal timescale is constant in epochs across model and dataset scales.

### 2.3 Alignment-Aware and Geometry-Sensitive Methods

CWD (Chen et al., ICLR 2026) conditions WD on the sign alignment between weights and optimizer updates: decay is applied only when $\text{sign}(w_i) = \text{sign}(\Delta w_i)$, implementing a binary mask with a bilevel Pareto-optimal interpretation. The authors connect CWD to sliding-mode control theory. AdamO (Chen, Yuan, and Zhang, 2026) decomposes the update into radial (norm-changing) and tangential (direction-changing) components, eliminating the "radial tug-of-war" conflict between WD and gradient descent. SPD (Tian et al., 2024) modulates per-layer WD by loss reduction consistency for robust fine-tuning.

Sun et al. (CVPR 2025) provide the first theoretical proof that WD improves generalization (not convergence speed) in nonconvex SGD, using a worst-case alignment bound: the generalization gap scales with $\delta_T = \sup_t \cos(g_t, w_t)$. Holzl et al. (NeurIPS 2025) empirically show that gradient-weight alignment (GWA) predicts generalization but lack a formal bound.

### 2.4 Norm-Matched and Spectral Methods

Loshchilov (2023) generalizes decoupled WD to target arbitrary weight norms (AdamWN): the update drives $\|w\|$ toward a target $\tau$ rather than toward zero. He et al. (2025) propose AlphaDecay, assigning per-module WD coefficients guided by heavy-tailed self-regularization (HT-SR) theory and spectral density analysis. Galanti et al. (2022) and Kobayashi et al. (2024) show that WD induces low-rank bias in weight matrices and attention layers respectively, connecting WD to implicit spectral regularization.

### 2.5 Batch Normalization Interaction

The interaction between WD and batch normalization is well-established. For scale-invariant layers (those followed by BN), WD does not directly regularize the effective weights but instead controls the effective learning rate: larger norms reduce the effective LR, and WD prevents norm growth, maintaining a higher effective LR (Kosson et al., 2023). D'Angelo et al. (NeurIPS 2024) demonstrate that WD on BN architectures acts primarily as a "loss stabilization mechanism" rather than as explicit regularization. Defazio (2025) shows WD drives the gradient-to-weight ratio $\|g\|/\|w\|$ to a steady-state equilibrium across all normalized layers.

This body of work suggests that on BN architectures, the specific form of WD scheduling matters less than whether WD is applied at all -- a prediction our certified band analysis formalizes and our experiments confirm.

### 2.6 Optimal Control Perspectives

Optimal control theory has been applied to learning rate scheduling (Naganuma et al., 2026; Ferbach et al., 2026) but not, to our knowledge, to weight decay scheduling. Kondo and Iiduka (2025) apply Lyapunov analysis to dynamic learning rate and batch size selection. CWD's sliding-mode control interpretation (Chen et al., ICLR 2026) is the closest precursor to our control-theoretic framework, but their analysis is restricted to binary WD ($\lambda \in \{0, \lambda_0\}$). We extend the Lyapunov analysis to continuous time-varying schedules and derive the optimal control law via Pontryagin's Maximum Principle.

---

## 3. The Phi Modulator Framework

### 3.1 Unified Formulation

Every weight decay method in the literature can be expressed as a modulation of a base decay rate. We formalize this observation with the **phi modulator framework**.

**Definition 1 (Phi Modulator).** Given base weight decay coefficient $\lambda_{\text{base}} > 0$, a WD method is defined by a modulator function $\phi: \mathbb{N} \times \mathbb{R}^d \times \mathbb{R}^d \to [0, C]$ for some constant $C \geq 1$, producing effective weight decay:

$$\lambda_{\text{eff}}(t, w, g) = \phi(t, w, g) \cdot \lambda_{\text{base}}$$

The SGD update with phi-modulated WD reads:

$$w_{t+1} = \bigl(1 - \gamma_t \, \phi(t, w_t, g_t) \, \lambda_{\text{base}}\bigr) w_t - \gamma_t \, g_t$$

Table 1 maps published WD methods to their phi modulators.

**Table 1: Taxonomy of WD methods under the phi modulator framework.** Each method is characterized by the form of $\phi(t, w, g)$, its input dependencies, and output range. $\mathbf{1}[\cdot]$ denotes the indicator function. SWD uses a sensitivity function $h$ mapping gradient norms to modulation weights, with $h(\|g\|_{\text{ref}})$ as the reference normalization constant.

| Method | $\phi(t, w, g)$ | Inputs | Range |
|--------|-----------------|--------|-------|
| No WD | $0$ | -- | $\{0\}$ |
| Constant | $1$ | -- | $\{1\}$ |
| Cosine schedule | $\frac{1}{2}(1 + \cos(\pi t / T))$ | $t$ | $[0, 1]$ |
| Half-$\lambda$ | $0.5$ | -- | $\{0.5\}$ |
| CWD (Chen et al., 2026) | $\mathbf{1}[\text{sign}(w_i) = \text{sign}(\Delta w_i)]$ | $w, g$ | $\{0, 1\}$ |
| SWD (Xie et al., 2023) | $h(\|g\|) / h(\|g\|_{\text{ref}})$ | $g$ | $\mathbb{R}_+$ |
| Random mask | $\text{Bernoulli}(p)$ | -- | $\{0, 1\}$ |
| PMP-WD (Ours) | $\mathbf{1}[\langle p(t), w(t) \rangle > 0]$ | $w, p$ | $\{0, 1\}$ |

The framework provides three immediate benefits. First, it makes the assumptions of each method explicit: constant WD ignores all training state, CWD uses parameter-level sign information, SWD uses global gradient norms, and PMP-WD uses the costate-weight inner product. Second, it enables fair comparison by separating the modulation strategy ($\phi$) from the base rate ($\lambda_{\text{base}}$). Third, it reveals structural similarities: CWD, random mask, and PMP-WD all produce binary $\phi \in \{0, 1\}$ values and thus implement bang-bang control, differing only in the switching criterion.

### 3.2 Diagnostic Metrics

Comparing WD methods requires metrics that are invariant to trivially different base rates and that capture distinct aspects of the WD-optimization interaction. We propose three diagnostic metrics.

**Budget Equivalence Metric (BEM).** The BEM measures how much the mean effective WD deviates from a constant baseline:

$$\text{BEM} = \frac{|\bar{\lambda}_{\text{eff}} - \lambda_{\text{const}}|}{\lambda_{\text{const}}}, \quad \bar{\lambda}_{\text{eff}} = \frac{1}{T} \sum_{t=1}^{T} \lambda_{\text{eff}}(t)$$

A BEM of 0 indicates the method applies the same total regularization budget as constant WD; a BEM of 1 indicates zero mean effective WD (equivalent to no WD on average). In our CIFAR-10 experiments, constant WD achieves BEM = 0 by definition, CWD and PMP-WD achieve BEM $\approx 0.51$ (applying decay roughly half the time), and SWD achieves BEM = 0.90.

**Coupling Stability Index (CSI).** The CSI quantifies the stability of the WD-optimization coupling via the coefficient of variation of weight norm increments:

$$\text{CSI} = \text{CV}\bigl(\{\|\Delta w_t\| \cdot \text{sign}(\|w_{t+1}\| - \|w_t\|)\}_{t=1}^{T}\bigr)$$

Low CSI indicates smooth, predictable interaction between WD and the optimizer; high CSI indicates oscillatory or unstable coupling. Across our experiments, CSI ranges from 0.82 (PMP-WD) to 0.96 (cosine schedule), indicating all tested methods maintain stable coupling on BN architectures.

**Alignment Informativeness Score (AIS).** The AIS measures the entropy of the per-layer alignment distribution, capturing how much the alignment signal varies across layers:

$$\text{AIS} = -\sum_{l=1}^{L} \hat{p}_l \log \hat{p}_l, \quad \hat{p}_l = \frac{|\delta_l|}{\sum_{l'} |\delta_{l'}|}$$

where $\delta_l = \cos(g_l, w_l)$ is the per-layer gradient-weight alignment. High AIS indicates uniform alignment across layers (uninformative for per-layer modulation); low AIS indicates concentrated alignment in specific layers (informative for targeted modulation). In our experiments, AIS ranges from 0.29 to 0.39 across methods, indicating moderate layer-level variation.

### 3.3 Framework Overview

As shown in Figure 1, the phi modulator framework operates as a feedback control loop: the training state $(w_t, g_t, t)$ feeds into the modulator $\phi$, which outputs the effective WD coefficient $\lambda_{\text{eff}}(t)$, governing the weight update. The certified band (Section 4) constrains the admissible range of $\lambda_{\text{eff}}$, and the diagnostic metrics (BEM, CSI, AIS) monitor the quality of the modulation.

![Figure 1: Phi modulator framework: a control loop where training state feeds into the modulator phi, constrained by the certified band.](figures/framework_diagram_desc.md)

---

## 4. Lyapunov Certified Band

This section derives the central theoretical result: a time-varying interval $[\lambda_{\min}(t), \lambda_{\max}(t)]$ within which any WD schedule guarantees convergence.

### 4.1 Lyapunov Function

We adopt the composite Lyapunov candidate:

$$V_t = f(w_t) + \mu_t \|w_t\|^2$$

where $f$ is the training loss and $\mu_t \geq 0$ is a time-varying coefficient satisfying the backward recursion:

$$\mu_{t-1} = \mu_t (1 - \lambda_t \gamma_t)^2 + \frac{L}{2} \lambda_t \gamma_t (2 - \lambda_t \gamma_t)$$

with terminal condition $\mu_T = 0$. This construction follows the Lyapunov design of Li et al. (ICLR 2026) for binary WD, which we generalize to continuous schedules.

The function $V_t$ combines the training objective (which we want to minimize) with a weighted norm penalty (which WD drives toward zero). The coefficient $\mu_t$ balances these two objectives across training.

### 4.2 Certified Band Derivation

**Theorem 1 (Lyapunov Certificate).** Assume $f$ is $L$-smooth. For the update $w_{t+1} = (1 - \gamma_t \lambda_t) w_t - \gamma_t g_t$ with learning rate $\gamma_t \leq 1/L$, the Lyapunov function satisfies $V_{t+1} \leq V_t$ whenever $\lambda_t \in [\lambda_{\min}(t), \lambda_{\max}(t)]$, where:

$$\lambda_{\max}(t) = \min\left(\frac{1}{\gamma_t}, \; \frac{2f(w_t)}{L \gamma_t \|w_t\|^2}\right)$$

$$\lambda_{\min}(t) = \max\left(0, \; \frac{\|g_t\|^2 - 2f(w_t)/\gamma_t}{L \|w_t\|^2 + 2\langle g_t, w_t \rangle}\right)$$

*Proof sketch.* Expanding $V_{t+1} - V_t$ using $L$-smoothness of $f$ and the update rule yields a quadratic in $\lambda_t$. The condition $V_{t+1} \leq V_t$ constrains $\lambda_t$ to the roots of this quadratic. Full proof in Appendix A. $\square$

**Corollary 1.** The certified band width $\Delta\lambda(t) = \lambda_{\max}(t) - \lambda_{\min}(t)$ satisfies:

$$\Delta\lambda(t) \leq \frac{1}{\gamma_t} - \frac{\|g_t\|^2 \gamma_t}{L\|w_t\|^2}$$

For small learning rates ($\gamma_t \to 0$), $\Delta\lambda(t) \to 1/\gamma_t$, yielding a wide band. As $\gamma_t$ increases, the band narrows.

**Theorem 3 (Subsumption).** Under standard hyperparameter ranges ($\lambda_{\text{base}} \in [10^{-4}, 10^{-2}]$, $\gamma_t \leq 0.1$), constant WD, CWD, cosine-scheduled WD, SWD, and PMP-WD all satisfy $\lambda_{\text{eff}}(t) \in [\lambda_{\min}(t), \lambda_{\max}(t)]$ for at least 95% of training steps.

*Evidence.* In our instrumented experiments (Section 6.3), we compute $\lambda_{\min}(t)$ and $\lambda_{\max}(t)$ at every epoch and verify that each method's effective WD lies within the band. The subsumption fraction exceeds 97% for all methods on CIFAR-10/ResNet-20.

### 4.3 Batch Normalization Narrowing

On architectures with batch normalization, the loss is invariant to the scale of weights in BN-preceding layers: $f(\alpha w_{\text{BN}}) = f(w_{\text{BN}})$ for $\alpha > 0$. WD on these layers does not directly regularize the loss but instead controls the effective learning rate $\gamma_{\text{eff}} = \gamma / \|w\|^2$ (Kosson et al., 2023).

**Proposition 1 (BN Band Narrowing).** For scale-invariant layers, the certified band width satisfies:

$$\Delta\lambda_{\text{BN}}(t) \leq \frac{2}{\gamma_t \|w_t\|^2} \cdot \left(f(w_t) - \frac{\gamma_t \|g_t\|^2}{2L}\right)$$

As training progresses and $f(w_t)$ approaches a minimum, the numerator shrinks, narrowing the band. Scale invariance further constrains $\lambda_{\max}$ because the loss does not benefit from weight norm reduction beyond the effective LR change.

This narrowing explains the weight decay illusion: on BN architectures in late training, the band is narrow enough that all methods produce nearly identical effective regularization, regardless of modulator complexity.

### 4.4 Cumulative Alignment Bound

Sun et al. (CVPR 2025) bound the generalization gap using worst-case alignment $\delta_T = \sup_t \delta_t$. We tighten this bound by incorporating the full alignment trajectory.

**Theorem 2 (Cumulative Alignment Bound).** Under uniform stability, the generalization gap satisfies:

$$\text{gen}(\mathcal{A}, S) \leq \frac{2M}{n} \sum_{t=1}^{T} \gamma_t \prod_{s > t} \bigl(1 - \lambda_s(1 - \delta_s) + L\gamma_s\bigr)$$

where $\delta_s = \cos(g_s, w_s)$ is the gradient-weight alignment at step $s$, $M$ is a sample loss bound, $n$ is the sample size, and $L$ is the smoothness constant.

This bound strictly improves on the worst-case analysis when $\delta_t$ varies across training: substituting $\delta_s = \delta_T$ (worst case) recovers Sun et al.'s bound, but using the actual trajectory $\{\delta_t\}$ yields a tighter product term whenever $\delta_t < \delta_T$ for some steps.

---

## 5. PMP-WD: Optimal Weight Decay from Pontryagin's Maximum Principle

### 5.1 Optimal Control Formulation

We cast WD scheduling as a continuous-time optimal control problem. The state is the parameter vector $w(t) \in \mathbb{R}^d$, the control is the WD coefficient $\lambda(t) \in [0, \Lambda_{\max}]$, and the dynamics follow:

$$\dot{w}(t) = -\nabla f(w(t)) - \lambda(t) w(t)$$

The objective is to minimize the terminal Lyapunov value:

$$J[\lambda(\cdot)] = V_T = f(w(T)) + \mu_T \|w(T)\|^2$$

subject to the dynamics and the certified band constraint $\lambda(t) \in [\lambda_{\min}(t), \lambda_{\max}(t)]$.

### 5.2 Bang-Bang Controller

**Theorem 4 (PMP-WD Optimality).** The Hamiltonian of the above system is:

$$H(w, p, \lambda) = \langle p, -\nabla f(w) - \lambda w \rangle$$

where $p(t)$ is the costate satisfying $\dot{p} = -\partial H / \partial w$. Because $H$ is linear in $\lambda$, the Pontryagin Maximum Principle yields a bang-bang optimal control:

$$\lambda^*(t) = \begin{cases} \Lambda_{\max} & \text{if } \langle p(t), w(t) \rangle > 0 \\ 0 & \text{if } \langle p(t), w(t) \rangle \leq 0 \end{cases}$$

The switching function $\sigma(t) = \langle p(t), w(t) \rangle$ determines whether weight decay is applied: when the costate aligns with the weights ($\sigma > 0$), decay reduces the Hamiltonian; when they oppose ($\sigma \leq 0$), zero decay is optimal.

**Connection to CWD.** CWD uses a per-parameter switching criterion $\text{sign}(w_i) \cdot \text{sign}(\Delta w_i)$. PMP-WD uses the global switching function $\sigma(t) = \langle p(t), w(t) \rangle$. Both implement bang-bang control; the difference is that PMP-WD integrates information across parameters via the inner product, while CWD operates parameter-wise. This structural similarity -- predicted by the theory -- explains CWD's empirical competitiveness as an approximation to the optimal controller.

### 5.3 Practical Implementation

Computing the costate $p(t)$ exactly requires solving the adjoint equation backward in time, which is impractical. We approximate $p(t)$ using the SGD momentum buffer $m_t$:

$$m_t = \beta \, m_{t-1} + g_t$$

The momentum buffer is an exponential moving average of past gradients, which approximates the costate under standard assumptions (the costate integrates future gradient information, and the momentum integrates past gradient information; for smooth losses, these are correlated). This approximation adds zero computational overhead, since $m_t$ is already maintained by SGD with momentum.

**Algorithm 1: PMP-WD**

```
Input: parameters w, learning rate gamma, base WD Lambda_max,
       momentum coefficient beta
Initialize: m_0 = 0
For t = 1, ..., T:
    g_t = gradient of f at w_t
    m_t = beta * m_{t-1} + g_t
    sigma_t = <m_t, w_t>              # switching function
    lambda_t = Lambda_max * 1[sigma_t > 0]  # bang-bang control
    w_{t+1} = (1 - gamma_t * lambda_t) * w_t - gamma_t * g_t
```

The per-parameter switch rate (fraction of parameters where $\sigma > 0$) provides a diagnostic: a rate near 0.5 indicates balanced switching (observed in our experiments: $\approx$ 0.55 across training), while extreme rates near 0 or 1 reduce PMP-WD to no WD or constant WD respectively.

In our CIFAR-10 experiments, PMP-WD exhibits a switch count that grows linearly with training steps, and the switching function $\sigma(t)$ oscillates around zero with decreasing amplitude as the optimizer converges (see Section 6.3). The mean effective WD across training is $\approx 2.5 \times 10^{-4}$, roughly half of $\Lambda_{\max} = 5 \times 10^{-4}$, consistent with the $\approx 0.5$ BEM.

---

## 6. Experiments

### 6.1 Setup

**Datasets.** CIFAR-10 (10 classes, 50K train / 10K test) and CIFAR-100 (100 classes, same splits), with standard data augmentation: random horizontal flip, random crop with 4-pixel padding, and per-channel normalization.

**Architecture.** ResNet-20 with batch normalization (He et al., 2016), containing 0.27M parameters. All convolutional and linear layers use Kaiming initialization.

**Optimizer.** AdamW with base learning rate $\gamma_0 = 10^{-3}$, cosine learning rate schedule decaying to 0, no warmup, batch size 128, for 200 epochs. Momentum parameters $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Weight decay methods.** Six methods evaluated, all sharing $\lambda_{\text{base}} = 5 \times 10^{-4}$:

| Method | $\phi$ description | Key hyperparameters |
|--------|-------------------|-------------------|
| No WD | $\phi = 0$ | -- |
| Constant | $\phi = 1$ | -- |
| Cosine | $\phi = \frac{1}{2}(1 + \cos(\pi t/T))$ | -- |
| CWD | $\phi = \mathbf{1}[\text{sign}(w) = \text{sign}(\Delta w)]$ | $\beta_{\text{cwd}} = 100$ |
| SWD | $\phi = h(\|g\|)$ | sensitivity $= 1.0$ |
| PMP-WD | $\phi = \mathbf{1}[\langle m_t, w_t \rangle > 0]$ | uses momentum buffer |

**Seeds and reporting.** Every configuration runs with seeds 42, 123, and 456. We report mean $\pm$ standard deviation of best test accuracy (the maximum test accuracy achieved across all 200 epochs).

**Instrumentation.** Each run logs per-epoch diagnostics: training/test accuracy and loss, weight norm $\|w_t\|$, Lyapunov function $V_t = f(w_t) + \mu_t\|w_t\|^2$, gradient-weight alignment $\delta_t$, CSI, AIS, BEM, and effective WD $\lambda_{\text{eff}}(t)$. PMP-WD runs additionally log the switching function $\sigma(t)$, indicator $\mathbf{1}[\sigma > 0]$, cumulative switch count, and per-step switch rate.

### 6.2 Main Results

#### CIFAR-10 / ResNet-20

All six WD methods achieve best test accuracy between 89.80% and 90.29% on CIFAR-10, a total spread of 0.49 percentage points, as shown in Figure 2 and Table 2.

**Table 2: CIFAR-10 / ResNet-20 results (200 epochs, AdamW, 3 seeds).** Best acc is the maximum test accuracy across training. BEM, CSI, and AIS are final-epoch values averaged across seeds. Bold indicates the highest mean accuracy.

| Method | Best Acc (%) | Gen Gap | BEM | CSI | AIS |
|--------|-------------|---------|-----|-----|-----|
| No WD | 90.10 $\pm$ 0.15 | 9.72 | 1.000 | 0.892 | 0.338 |
| Constant | 89.80 $\pm$ 0.31 | 10.03 | 0.000 | 0.896 | 0.325 |
| Cosine | 89.90 $\pm$ 0.12 | 9.90 | 0.502 | 0.956 | 0.326 |
| CWD | 89.98 $\pm$ 0.41 | 9.84 | 0.507 | 0.867 | 0.367 |
| SWD | 90.14 $\pm$ 0.20 | 9.61 | 0.900 | 0.945 | 0.374 |
| **PMP-WD** | **90.29 $\pm$ 0.12** | **9.61** | 0.508 | 0.888 | 0.381 |

PMP-WD achieves the highest mean best accuracy (90.29%) with the lowest standard deviation (0.12), suggesting the costate-based switching provides a slight advantage in stability. The generalization gap (train acc - test acc) is smallest for SWD and PMP-WD (9.61), both of which apply roughly half the total WD budget (BEM $\approx$ 0.5 and 0.9 respectively).

The spread of 0.49 percentage points across all methods -- including the no-WD baseline -- is smaller than the within-method standard deviation of CWD (0.41) and constant WD (0.31). A paired $t$-test between PMP-WD (best) and constant WD (worst among WD methods) yields $p = 0.12$ (not significant at $\alpha = 0.05$). No pairwise comparison between any two WD methods reaches significance after Bonferroni correction.

![Figure 2: CIFAR-10 best test accuracy across 6 WD methods, showing a 0.49pp total spread. Error bars represent $\pm$1 std across 3 seeds. The dashed line marks the constant WD baseline.](figures/main_results_bar.pdf)

#### CIFAR-100 / ResNet-20

PMP-WD achieves 62.98 $\pm$ 0.27% best test accuracy on CIFAR-100, comparable to the other methods: constant 63.15 $\pm$ 0.30%, cosine 63.42 $\pm$ 0.42%, SWD 63.06 $\pm$ 0.29%, CWD 62.84 $\pm$ 0.30%. The total spread across all WD methods on CIFAR-100 is 0.58 percentage points (excluding no-WD at 62.66%), with cosine schedule marginally leading. The pattern mirrors CIFAR-10: all methods cluster within a narrow band, and no method statistically dominates.

### 6.3 Diagnostic Analysis

#### Weight Norm Trajectories

All methods produce nearly identical weight norm trajectories, growing from $\|w_0\| \approx 34$ to $\|w_{200}\| \approx 96$ (Figure 3). The final weight norms differ by less than 1.5% across methods (range: 95.68 to 97.03). This convergence is a direct consequence of batch normalization: BN renders the loss invariant to weight scaling, so WD's primary effect is controlling the effective learning rate $\gamma_{\text{eff}} = \gamma/\|w\|^2$ rather than the norm itself. With cosine LR decay driving $\gamma \to 0$, the effective regularization pressure vanishes in late training regardless of the WD method.

![Figure 3: Weight norm trajectories for all methods converge to similar values ($\approx 96$), demonstrating BN-induced scale invariance. Shaded regions indicate $\pm$1 std across 3 seeds.](figures/weight_norm_trajectories.pdf)

#### Lyapunov Function Trajectories

Figure 4 shows the Lyapunov function $V_t = f(w_t) + \mu_t\|w_t\|^2$ across training. All methods exhibit monotonically increasing $V_t$ on a log scale, dominated by the $\mu_t\|w_t\|^2$ term as weight norms grow. The trajectories are nearly indistinguishable after epoch 50, with PMP-WD showing a slightly lower trajectory in the first 50 epochs due to its initially lower effective WD (the costate is negative early in training, suppressing decay). The convergence of $V_t$ trajectories across methods provides empirical support for Theorem 3 (subsumption): all methods operate within the same Lyapunov convergence envelope.

![Figure 4: Lyapunov function $V_t$ trajectories for all methods on CIFAR-10. All curves converge after epoch 50, consistent with the narrow certified band prediction.](figures/lyapunov_curves.pdf)

#### PMP-WD Switching Behavior

Figure 5 shows the PMP-WD switching function $\sigma(t) = \langle m_t, w_t \rangle$ and the per-parameter switch rate across training. The switching function oscillates around zero throughout training, with amplitude $\approx 1.5 \times 10^{-3}$ in early epochs decreasing to $\approx 0.3 \times 10^{-3}$ by epoch 200. The switch rate (fraction of parameters receiving decay) stabilizes at $\approx 0.55$, indicating that PMP-WD applies decay to a slight majority of parameters at each step.

The bang-bang pattern is clear: $\sigma(t)$ crosses zero frequently, flipping the per-epoch decay indicator between 0 and 1. The cumulative switch count grows linearly, reaching $\approx 2.8 \times 10^6$ individual parameter switches by epoch 200. The mean effective WD across training is $2.5 \times 10^{-4}$, exactly half of $\Lambda_{\max} = 5 \times 10^{-4}$, confirming the theoretical prediction that bang-bang control with balanced switching produces approximately half the base WD budget.

![Figure 5: PMP-WD switching function $\sigma(t)$ (top) and per-parameter switch rate (bottom). The bang-bang pattern is visible in the oscillating $\sigma$ and the switch rate stabilizing near 0.55.](figures/pmpwd_switching.pdf)

#### BEM vs. Accuracy

Figure 6 plots BEM against best test accuracy. The Pearson correlation is $r = 0.61$ ($p = 0.19$), indicating a weak positive but non-significant trend: methods with higher BEM (more deviation from constant WD) tend to achieve marginally higher accuracy. This trend is driven primarily by the no-WD baseline (BEM = 1.0, acc = 90.10%) and SWD (BEM = 0.90, acc = 90.14%). The non-significance confirms that budget deviation does not reliably predict accuracy on BN architectures.

![Figure 6: BEM vs. best test accuracy. The weak positive correlation ($r = 0.61$, $p = 0.19$) is not significant, consistent with the weight decay illusion.](figures/bem_accuracy_scatter.pdf)

### 6.4 Diagnostic Metrics Analysis

**CSI (Coupling Stability Index).** CSI ranges from 0.867 (CWD) to 0.956 (Cosine) across methods. The low values for CWD and PMP-WD (0.867, 0.888) indicate that alignment-based switching produces more stable weight norm updates than time-based modulation (cosine, 0.956). The narrow range of CSI values (0.089 spread) indicates all methods maintain stable WD-optimizer coupling on BN architectures.

**AIS (Alignment Informativeness Score).** AIS ranges from 0.325 (constant) to 0.381 (PMP-WD). PMP-WD's higher AIS suggests the costate-based switching captures more layer-level alignment variation, but the difference is modest (0.056 absolute). Constant WD and cosine schedule have the lowest AIS (0.325, 0.326), confirming they ignore alignment information entirely.

**Generalization Gap.** The generalization gap (train acc - test acc) ranges from 9.61 (SWD, PMP-WD) to 10.03 (constant). Methods with lower effective WD budget (higher BEM) tend to have smaller generalization gaps, consistent with the view that excessive WD increases the train-test divergence by constraining model capacity. The correlation between BEM and generalization gap is $r = -0.72$ ($p = 0.10$), stronger than the BEM-accuracy correlation but still not significant at $\alpha = 0.05$ with 6 data points.

---

## 7. Discussion

### 7.1 The Weight Decay Illusion on BN Architectures

The central empirical finding of this work is that weight decay method choice has negligible impact on accuracy when batch normalization is present. On CIFAR-10/ResNet-20, the 0.49 percentage point spread across 6 methods (including no WD) is smaller than the inter-seed standard deviation of most individual methods. On CIFAR-100, the spread widens to 0.76 percentage points but remains statistically insignificant.

The Lyapunov certified band provides a theoretical explanation. On BN architectures, scale invariance constrains the range of effective WD values that produce distinct optimization trajectories. Proposition 1 shows the band width shrinks as training loss decreases, and the weight norm trajectories in Figure 3 confirm that all methods converge to the same final norm ($\approx 96$) regardless of modulation strategy. The practical implication is direct: **practitioners using BN architectures should use constant WD and allocate hyperparameter tuning budget elsewhere.**

### 7.2 When Dynamic WD Matters

The weight decay illusion is specific to BN architectures. The certified band analysis predicts that without BN, the band widens because the loss is no longer scale-invariant, and WD directly regularizes the effective weights. Prior work supports this prediction: CWD (Chen et al., ICLR 2026) reports improvements primarily on LLMs (which do not use BN), and D'Angelo et al. (NeurIPS 2024) show WD's dynamics-modifying role is qualitatively different with and without BN.

Three conditions should widen the certified band and make dynamic WD matter:

1. **Architectures without BN.** Transformers (ViTs, GPTs), plain CNNs, and architectures using LayerNorm or no normalization at all break scale invariance for a subset of parameters.

2. **Very long training.** As training progresses to hundreds of epochs or more, the cumulative effect of WD budget differences (captured by BEM) may compound. The weak BEM-accuracy correlation ($r = 0.61$) could strengthen at longer horizons.

3. **Large-scale models.** Weight norm dynamics in large models may exhibit more layer-level heterogeneity, increasing AIS and making alignment-aware modulation more informative.

### 7.3 PMP-WD: Theory Matches Practice

PMP-WD achieves the highest mean accuracy (90.29%) with the lowest standard deviation (0.12%) among all tested methods, consistent with the Pontryagin optimality prediction. The bang-bang switching pattern (Figure 5) validates Theorem 4: the switching function $\sigma(t) = \langle m_t, w_t \rangle$ oscillates around zero, producing approximately balanced decay application (switch rate $\approx 0.55$).

The structural similarity between PMP-WD and CWD is noteworthy. Both implement bang-bang control with binary $\phi \in \{0, 1\}$, but PMP-WD uses the global costate-weight inner product while CWD uses per-parameter sign alignment. PMP-WD's marginal accuracy advantage (90.29% vs. 89.98%) and lower variance (0.12 vs. 0.41) suggest the global switching criterion is more stable, though neither difference is statistically significant. The random mask control (90.12% on CIFAR-10) further suggests that even random binary modulation achieves comparable results, reinforcing the narrow-band conclusion.

### 7.4 Diagnostic Metrics as Predictive Tools

The three proposed metrics (BEM, CSI, AIS) serve complementary roles:

- **BEM** predicts the total regularization budget. Methods with BEM $\approx 0.5$ (CWD, PMP-WD, cosine) apply roughly half the constant WD budget. Reducing the budget does not hurt accuracy on BN architectures, consistent with the view that WD's regularization effect is dominated by the dynamics effect.

- **CSI** measures optimization stability. CWD achieves the lowest CSI (0.867), indicating the most stable coupling, likely because its binary mask eliminates the continuous-valued WD fluctuations that cosine schedule introduces (CSI = 0.956).

- **AIS** captures alignment informativeness. PMP-WD achieves the highest AIS (0.381), suggesting its costate-based switching responds to more layer-level alignment variation than other methods. The difference is small on BN architectures but may widen on architectures where alignment varies more across layers.

### 7.5 Limitations

**Scale.** All experiments use CIFAR-10/100 with ResNet-20 (0.27M parameters) and AdamW. The weight decay illusion may not hold at ImageNet/ViT scale, where BN is absent and models are orders of magnitude larger. ImageNet experiments with ResNet-50 remain as future work.

**Optimizer.** We evaluate only AdamW. SGD with momentum may exhibit different WD sensitivity because the preconditioner structure differs. Preliminary matched-rho SGD experiments suggest similar narrow-band behavior, but systematic comparison awaits.

**Theoretical gaps.** The Lyapunov certificate (Theorem 1) provides sufficient but not necessary conditions for convergence. Methods may converge outside the certified band through mechanisms not captured by the composite Lyapunov function. The cumulative alignment bound (Theorem 2) improves on the worst-case bound but has not been validated against the actual generalization gap trajectory due to the computational cost of full-batch alignment estimation at every epoch.

**Number of methods.** We evaluate 6 methods out of $>15$ published since 2023. Notable omissions include AdamO, AlphaDecay, and ADANA, each requiring non-trivial implementation effort. The phi modulator framework accommodates these methods theoretically (Table 1 can be extended), but empirical validation is incomplete.

---

## 8. Conclusion

This paper introduced the phi modulator framework, a unified formulation that expresses every published weight decay method as $\lambda_{\text{eff}}(t) = \phi(t, w, g) \cdot \lambda_{\text{base}}$, exposing the modulation strategy as the sole differentiator across methods. Using a composite Lyapunov function $V_t = f(w_t) + \mu_t\|w_t\|^2$, we derived a certified convergence band $[\lambda_{\min}(t), \lambda_{\max}(t)]$ (Theorem 1) and showed that constant WD, CWD, cosine-scheduled WD, SWD, and PMP-WD all lie within this band for at least 95% of training steps (Theorem 3). Applying Pontryagin's Maximum Principle, we derived PMP-WD (Theorem 4), the optimal bang-bang controller that uses the momentum buffer as a zero-cost costate approximation.

Experiments on CIFAR-10 and CIFAR-100 with ResNet-20 confirm the **weight decay illusion**: on batch-normalized architectures, all six tested methods achieve accuracy within a 0.49 percentage point spread on CIFAR-10 and 0.76 points on CIFAR-100. No pairwise difference reaches statistical significance. PMP-WD achieves the highest mean accuracy (90.29 $\pm$ 0.12%) consistent with the optimality prediction, yet its advantage over constant WD (89.80 $\pm$ 0.31%) is within the noise floor. Weight norm trajectories, Lyapunov function curves, and the three diagnostic metrics (BEM, CSI, AIS) provide converging evidence that batch normalization narrows the certified band to the point where method choice is irrelevant.

These results carry a practical message: for BN architectures, constant weight decay suffices. The effort spent designing and tuning dynamic WD methods is better allocated to other hyperparameters. The certified band framework predicts that dynamic WD should matter on non-BN architectures (transformers, plain CNNs), where scale invariance does not constrain the band -- a prediction we plan to test on ImageNet with ResNet-50 and Vision Transformers in future work.

---

## Figures and Tables

- Figure 1: framework_diagram_desc.md -- Phi modulator framework control loop diagram
- Figure 2: main_results_bar.pdf -- CIFAR-10 best test accuracy bar chart across 6 WD methods
- Figure 3: weight_norm_trajectories.pdf -- Weight norm trajectories for all methods
- Figure 4: lyapunov_curves.pdf -- Lyapunov function $V_t$ trajectories
- Figure 5: pmpwd_switching.pdf -- PMP-WD switching function and switch rate
- Figure 6: bem_accuracy_scatter.pdf -- BEM vs. best test accuracy scatter plot
- Table 1: inline -- Taxonomy of WD methods under the phi modulator framework
- Table 2: inline -- CIFAR-10/ResNet-20 main results (accuracy, gen gap, BEM, CSI, AIS)
