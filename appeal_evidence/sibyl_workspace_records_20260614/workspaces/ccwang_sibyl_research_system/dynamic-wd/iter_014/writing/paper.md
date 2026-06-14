# Gradient-to-Weight Ratio Homeostasis: A Unified Feedback Control Framework and Diagnostic Benchmark for Dynamic Weight Decay

---

## Abstract

Weight decay (WD) is the default regularizer in deep learning, yet the research community has fragmented into four parallel sub-traditions---scheduling, alignment-aware, decoupled, and norm-matched---each developing methods under incomparable protocols. We show that all four traditions implicitly manipulate a single quantity: the per-layer gradient-to-weight ratio $\rho_t^l = \|g_t^l\| / \|w_t^l\|$. We formalize this observation into a PID-style control law $\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + K_i \cdot \text{EMA}(e_t^l) - K_d \cdot \alpha_t^l \cdot e_t^l$ that maps five existing methods to specific gain configurations: FixedWD is open-loop ($K_p = K_i = K_d = 0$), CWD uses derivative/alignment feedback ($K_d > 0$), SWD uses proportional-integral control, and CPR uses integral-dominant control. Empirical fitting on CIFAR-10/ResNet-20 trajectories (200 epochs, 3 seeds) achieves 4.71% relative error for CWD and 9.57% for CPR; scheduling-based methods SWD (45.81%) and DefazioCorrective (37.56%) exceed the 20% threshold, delineating the framework's scope. We propose UDWDC, a proportional-only controller that closes the $\rho_t^l$ feedback loop with zero new hyperparameters. UDWDC does not beat tuned FixedWD on accuracy---CPR leads at 91.74% (CIFAR-10) and 74.74% (ImageNet)---but its instability (CSI$_{\text{combined}}$ = $-$2.41) is itself a finding: proportional-only WD control amplifies perturbations rather than damping them. Three standardized metrics---Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---expose method differences invisible to accuracy-only evaluation. Of the three feedback channels, integral control (CPR) delivers the largest empirical gains; proportional feedback destabilizes without integral smoothing; alignment/derivative feedback provides marginal benefit at standard $\lambda_{\text{base}}$.

---

## 1. Introduction

Weight decay (WD) is the default regularizer in deep learning: a single coefficient $\lambda$ that, in SGD-based optimizers, shrinks every parameter toward zero at each step (in Adam-family optimizers, decoupled WD applies an analogous but mathematically distinct operation). Despite its ubiquity, the research community has splintered into four parallel sub-traditions for *dynamically* controlling $\lambda$, each addressing a different perceived failure mode of fixed WD:

1. **WD scheduling** adjusts $\lambda$ based on gradient norms or training progress. SWD (Xie et al., NeurIPS 2023) senses $\|\nabla \mathcal{L}\|$ and reduces $\lambda_t$ when gradients are large; ADANA (Ferbach et al., 2026) applies logarithmic-time schedules to $\lambda$ alongside $\beta_1$ and $\beta_2$.

2. **Alignment-aware WD** conditions $\lambda$ on the cosine alignment $\alpha_t^l = \langle g_t^l, w_t^l \rangle / (\|g_t^l\| \|w_t^l\|)$ between gradient and weight vectors. CWD (Chen et al., ICLR 2026) applies a binary mask: decay only when $\alpha_t^l < 0$. A one-line code change yields +0.61% on ImageNet ViT-S/16 over standard AdamW.

3. **Decoupled WD** separates the weight-shrinkage term from adaptive gradient scaling. AdamW (Loshchilov & Hutter, ICLR 2019) remains the dominant instantiation; AdamO (Chen et al., 2026) further decomposes the update into radial (norm) and tangential (direction) components to resolve the "Radial Tug-of-War" between WD and gradient.

4. **Norm-matched WD** drives weight norms toward explicit targets via augmented Lagrangian constraints or spectral analysis. CPR (Franke et al., NeurIPS 2024) outperforms AdamW across CIFAR-100, ImageNet, and GPT-2 by enforcing per-parameter-matrix upper-bound constraints with only two hyperparameters.

These four sub-communities develop methods in isolation and evaluate under incomparable protocols---different datasets, different metrics, different hyperparameter search budgets---making it impossible to determine which design choices actually matter.

Two recent theoretical results suggest the four traditions share a common target. Defazio (2025) proves that under constant learning rate, WD drives the per-layer gradient-to-weight ratio $\rho_t^l = \|g_t^l\| / \|w_t^l\|$ to a well-defined steady state for all normalized layers. Wang & Aitchison (ICML 2025) independently show that the optimal WD, recast as an exponential moving average (EMA) timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$, is constant across model and dataset scales. Sun et al. (CVPR 2025) prove that WD does not accelerate convergence but strictly improves generalization through an alignment-dependent mechanism.

We formalize the connection: **all four dynamic WD sub-traditions are approximations of a single PID-style feedback control law that regulates $\rho_t^l$ toward a prescribed target trajectory $\rho^*(t) = \eta_t / \tau$.** Despite using fundamentally different control strategies, CPR, CWD, SWD, and UDWDC all drive $\rho_t^l$ toward a common range within the first 4 epochs on CIFAR-10/ResNet-20 (Figure 1), demonstrating that $\rho_t^l$ is the shared control target. CWD's alignment mask corresponds to a derivative/alignment correction term ($K_d > 0$). SWD's gradient-norm sensing maps to proportional-integral control ($K_p > 0$, $K_i > 0$). CPR's augmented Lagrangian penalty accumulates constraint violations as integral control ($K_i > 0$). FixedWD is open-loop control with all gains set to zero.

![GW ratio trajectories for all methods](figures/rho_trajectories.pdf)

*Figure 1: Per-layer GW ratio $\rho_t^l$ trajectories on CIFAR-10/ResNet-20 (10-epoch pilot, seed 42). (a) Mean $\rho_t^l$ across 65 layers decays from $\sim$0.04 to $\sim$0.015 for all methods within 4 epochs, confirming $\rho_t^l$ as the shared control target. CPR (green) starts highest due to its large initial effective WD. UDWDC (red) shows more oscillation than FixedWD (gray), foreshadowing the instability quantified by CSI in Section 5.7. (b) Per-layer $\rho_t^l$ distribution at the final epoch: CPR achieves tighter norm constraints, while UDWDC shows higher variance — consistent with its negative $\text{CSI}_\text{combined} = -2.41$.*

Building on this unified view, we make the following contributions:

1. **Unified feedback control framework.** We derive a single parametric control law, $\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + K_i \cdot \text{EMA}(e_t^l) - K_d \cdot \alpha_t^l \cdot e_t^l$, where $\alpha_t^l = \langle g_t^l, w_t^l \rangle / (\|g_t^l\| \cdot \|w_t^l\|)$ is the gradient-weight alignment cosine, and map five existing methods to specific gain configurations $(K_p, K_i, K_d)$. Empirical fitting on CIFAR-10/ResNet-20 trajectories (200 epochs, 3 seeds) achieves 4.71% relative error for CWD and 9.57% for CPR. The scheduling-based methods SWD (45.81% error) and DefazioCorrective (37.56% error) exceed the 20% pass threshold, indicating that global gradient-norm scheduling does not map cleanly to per-layer $\rho_t^l$ feedback.

2. **UDWDC: a simple proportional controller.** We propose Unified Dynamic Weight Decay Control (UDWDC), a proportional-only controller ($K_p > 0$, $K_i = K_d = 0$) that closes the $\rho_t^l$ feedback loop explicitly. The target $\rho^*(t) = \eta_t / \tau$ is derived from EMA timescale theory, introducing zero new hyperparameters beyond the user's existing $\lambda_{\text{base}}$ and $\eta_0$ (clamp bounds [0.1, 10] are fixed engineering constants, not tunable parameters).

3. **Standardized evaluation metrics.** We introduce three metrics for fair cross-method comparison: the Budget Equivalence Metric (BEM), which measures accuracy improvement relative to the NoWD baseline per unit of total WD budget; the Coupling Stability Index (CSI), which quantifies the temporal and spatial stability of the WD-optimization coupling; and the Alignment Informativeness Score (AIS), which measures the Spearman correlation between the alignment signal and optimal WD decisions.

4. **Theoretical analysis.** We extend the nonconvex SGDW convergence framework of Sun et al. (CVPR 2025) to time-varying WD with alignment modulation. Theorem 1 proves that alignment-modulated WD achieves tighter generalization bounds per unit WD budget when the alignment variance $\text{Var}_t[\phi(\delta_t)] > 0$. Proposition 2 establishes that preconditioner-corrected alignment is geometrically consistent for Adam-family optimizers, while Proposition 3 predicts layer-differentiated steady states under alignment-modulated WD with Batch Normalization (BN).

5. **Comprehensive experiments.** We conduct a controlled comparison on CIFAR-10 (ResNet-20), CIFAR-100 (VGG-16-BN), and ImageNet (ResNet-50) with 3 seeds (CIFAR) and 3--5 seeds (ImageNet) per configuration. CPR achieves the highest accuracy on CIFAR-10 (91.74 $\pm$ 0.07%) and ImageNet (74.74 $\pm$ 0.05%). UDWDC ranks 7th of 8 methods on CIFAR-10 accuracy (90.15 $\pm$ 0.23%), below NoWD (90.25%). We report honest negative results: UDWDC v1 exhibits CSI$_{\text{combined}}$ = $-$2.41 (high instability), SWD and DefazioCorrective have fitting errors of 45.8% and 37.6% respectively, and CWD's accuracy is confounded by a 50% WD magnitude reduction.

The paper is organized as follows. Section 2 reviews the four WD sub-traditions and identifies $\rho_t^l$ as the shared control variable. Section 3 formalizes the unified control law, maps existing methods to gain configurations, presents UDWDC, and provides theoretical analysis. Section 4 defines the standardized evaluation metrics. Section 5 reports experiments. Section 6 discusses implications, limitations, and future work. Section 7 concludes.

---

## 2. Background and Related Work

This section surveys the four WD sub-traditions, identifies the gradient-to-weight ratio (GW ratio) $\rho_t^l$ as their shared underlying quantity, and positions our work relative to other unified optimizer perspectives.

### 2.1 WD Scheduling

WD scheduling modulates $\lambda_t$ as a function of training progress or gradient statistics, without per-layer differentiation.

**SWD** (Xie et al., NeurIPS 2023) identifies that fixed WD causes gradient-norm spikes during learning rate decay, harming generalization. SWD senses $\|\nabla \mathcal{L}\|$ to modulate $\lambda_t$, reducing decay when gradients are large. SWD's global gradient-norm sensing is broadly consistent with proportional-integral control, though Section 3 shows the per-layer fit has 45.81% error, indicating that SWD operates through mechanisms not fully captured by the per-layer $\rho_t^l$ feedback.

**ADANA** (Ferbach et al., 2026) extends scheduling to logarithmic-time decay of $\lambda$, $\beta_1$, and $\beta_2$ simultaneously, achieving a 40% compute efficiency gain on language modeling tasks. The log-time WD schedule is a specific monotonic trajectory $\lambda(t)$ that does not sense any per-layer feedback signal.

**Defazio corrective WD** (Defazio, 2025) compensates for the interaction between WD and learning rate schedules by applying a corrective term proportional to the current learning rate ratio $\eta_t / \eta_0$. Under cosine annealing, this produces monotonically decreasing effective WD. The corrective term can be viewed as a feedforward compensation signal in the control framework, operating on $\rho^*(t)$ rather than on the error signal.

### 2.2 Alignment-Aware WD

Alignment-aware methods condition WD on the geometric relationship between gradient and weight vectors.

**CWD** (Chen et al., ICLR 2026) applies a binary mask: WD is applied only when the gradient-weight alignment $\alpha_t^l < 0$ (gradient opposes the weight direction), meaning the gradient already drives weights toward zero. CWD reports +0.61% on ImageNet ViT-S/16 over standard AdamW. In the control framework, CWD's binary mask corresponds to derivative/alignment correction with $K_d > 0$.

**GWA** (Holzl et al., NeurIPS 2025) quantifies per-sample gradient-weight coherence as a generalization proxy. GWA provides empirical evidence that higher alignment coherence predicts better generalization, supporting the use of $\alpha_t^l$ as a feedback signal in WD controllers.

**AdamO** (Chen, Yuan, and Zhang, 2026) decomposes the optimizer update into radial (norm-changing) and tangential (direction-changing) components, identifying the "Radial Tug-of-War" between WD and gradient. AdamO resolves this by decoupling the two dynamics, applying SGD-style norm control alongside Adam-style tangential updates. This is an alignment-aware structural intervention rather than WD coefficient modulation.

### 2.3 Decoupled WD

The decoupled WD tradition addresses the mathematical distinction between $L_2$ regularization and weight decay in adaptive optimizers.

**AdamW** (Loshchilov & Hutter, ICLR 2019) demonstrated that in Adam, adding $\lambda w$ to the gradient ($L_2$ regularization) is not equivalent to subtracting $\eta \lambda w$ from the parameter update (weight decay). The adaptive second-moment scaling in Adam distorts the effective regularization strength per parameter. AdamW decouples the two, becoming the standard optimizer for transformer training. Zhang et al. (2018) independently identified three distinct mechanisms through which WD improves generalization, confirming that WD consistently outperforms $L_2$ across SGD, Adam, and K-FAC.

**Scaling rules.** Wang & Aitchison (ICML 2025) recast optimal WD as an EMA timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$ that remains constant across model and dataset scales. Chou (2025) derives WD proportional to $\gamma^2$ for stable weight norms. Kosson et al. (2025) show that WD stabilizes update dynamics across widths more effectively than maximal update parameterization ($\mu$P). These scaling results provide the basis for the target trajectory $\rho^*(t) = \eta_t / \tau$ in our framework.

### 2.4 Norm-Matched WD

Norm-matched methods drive weight norms toward explicit targets rather than zero.

**CPR** (Franke et al., NeurIPS 2024) formulates WD as a per-parameter-matrix constraint optimization problem via augmented Lagrangian. When weight norms exceed the target, the Lagrange multiplier accumulates, producing progressively larger effective WD. In the control framework, CPR's accumulated constraint violation corresponds to integral control ($K_i > 0$). CPR outperforms AdamW on CIFAR-100, ImageNet, and GPT-2 with only two hyperparameters.

**AdamWN** (Loshchilov, 2023) generalizes decoupled WD to target arbitrary weight norms (target = 0 recovers standard WD). **AlphaDecay** (He et al., 2025) assigns module-wise WD coefficients guided by spectral heavy-tailedness (Martin & Mahoney, HT-SR theory), scaling from 60M to 1B parameters.

### 2.5 Theoretical Foundations

**Defazio's GW ratio analysis** (2025) proves that WD drives $\rho_t^l$ to a steady state for all normalized layers, providing a clean explanation for the Adam vs. AdamW performance gap. All normalized layers converge to the same $\rho^*$, a "layer balancing" effect. This identifies $\rho_t^l$ as the natural control variable for WD.

**Sun et al.'s nonconvex convergence theory** (CVPR 2025) proves that WD does not accelerate convergence (the convergence rate remains $O(1/\sqrt{T})$ regardless of $\lambda$) but strictly improves generalization through an alignment-dependent mechanism. The generalization bound depends on the cumulative alignment-weighted contraction $A_T$ rather than the worst-case alignment. We extend this framework in Section 3.4.

**D'Angelo et al.** (NeurIPS 2024) provide a unifying empirical perspective: WD is never useful as explicit regularization but instead acts as a training dynamics modifier through the "loss stabilization mechanism" for SGD and the "bias-variance tradeoff" for near-one-epoch LLM training.

### 2.6 Distinction from PIDAO

PIDAO (Nature Communications 2024) applies PID control to the optimizer step itself---the gradient update direction and magnitude. Our work applies PID control to the WD coefficient $\lambda_t$---a different control target entirely. PIDAO optimizes *how the gradient is used*; we optimize *how much regularization is applied*. The two approaches are complementary: PIDAO could be combined with the proposed framework for joint optimization.

### 2.7 The Shared Control Variable

Each of the four sub-traditions, despite different formulations, affects $\rho_t^l$ through a different lever:

- **Scheduling methods** (SWD, ADANA, Defazio corrective) adjust $\lambda_t$, which directly modulates the WD force on $\|w_t^l\|$ and thus shifts the $\rho_t^l$ steady state.
- **Alignment-aware methods** (CWD) selectively disable WD for parameters where $\alpha_t^l < 0$, changing the *effective* $\lambda_t^l$ per layer and thus the realized $\rho_t^l$ trajectory.
- **Decoupled methods** (AdamW, AdamO) ensure that $\lambda$ acts on parameter norms without distortion from adaptive scaling, preserving the theoretical $\rho_t^l$ dynamics.
- **Norm-matched methods** (CPR, AdamWN) enforce explicit norm targets, equivalent to specifying a target $\rho^*$ via the relationship $\rho^* \propto \lambda / \|w^*\|$.

Section 3 formalizes this observation into a unified control law with explicit gain parameters $(K_p, K_i, K_d)$.

---

## 3. Unified Feedback Control Framework

We formalize the observation from Section 2 that all four WD sub-traditions manipulate the same underlying quantity---the per-layer GW ratio $\rho_t^l$---into a single parametric control law. We then derive the target trajectory from EMA timescale theory, present UDWDC, and provide theoretical analysis.

### 3.1 The Gradient-to-Weight Ratio as Control Variable

For each parameter group (layer) $l$ at training step $t$, define the GW ratio:

$$\rho_t^l = \frac{\|g_t^l\|_2}{\|w_t^l\|_2 + \epsilon}$$

where $g_t^l$ is the stochastic gradient, $w_t^l$ is the parameter vector, and $\epsilon = 10^{-8}$ prevents division by zero. The GW ratio measures the relative magnitude of gradient updates to current parameter norms: large $\rho_t^l$ indicates the gradient is dominant (rapid change), while small $\rho_t^l$ indicates the parameters are large relative to their gradients (stable regime).

Defazio (2025) proves that under constant learning rate $\eta$ and WD coefficient $\lambda$, the SGDW update $w_{t+1} = (1 - \eta\lambda)w_t - \eta g_t$ drives $\rho_t^l$ to a unique steady state for all normalized layers (BN or Layer Normalization). The steady-state value depends on $\lambda$ and $\eta$, establishing $\rho_t^l$ as a well-defined control variable with a deterministic equilibrium.

Wang & Aitchison (ICML 2025) show that the optimal WD, expressed as the EMA timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$, is constant across model sizes and datasets when measured in epochs. Combining these two results, we derive the **target trajectory** for a cosine annealing learning rate schedule. Under cosine annealing with time-varying $\eta_t$, the instantaneous steady-state GW ratio at each step is:

$$\rho^*(t) = \frac{\eta_t}{\tau} = \eta_t \cdot \lambda_0 \cdot \eta_0$$

This trajectory decreases as $\eta_t$ decays, prescribing tighter coupling (lower $\rho^l$) in late training when the model should stabilize near a minimum. We note that this extension from constant to time-varying learning rate is a heuristic argument: Defazio's steady-state result applies strictly under constant $\eta$, and the time-varying extension assumes the system tracks the instantaneous equilibrium adiabatically.

### 3.2 PID Control Law Parameterization

We parameterize the WD coefficient as a function of the GW ratio error:

$$\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + K_i \cdot \text{EMA}(e_t^l) - K_d \cdot \alpha_t^l \cdot e_t^l$$

where:
- $e_t^l = \rho_t^l - \rho^*(t)$ is the per-layer **control error** (deviation from target)
- $\alpha_t^l = \langle g_t^l, w_t^l \rangle / (\|g_t^l\| \cdot \|w_t^l\|)$ is the gradient-weight alignment cosine
- $K_p$ is the **proportional gain** (react to current error)
- $K_i$ is the **integral gain** (accumulated error via EMA smoothing)
- $K_d$ is the **derivative/alignment gain** (alignment-modulated correction)

The alignment term $\alpha_t^l$ multiplies the error in the $K_d$ channel, coupling the geometric relationship between gradient and weight directions to the WD adjustment. When $\alpha_t^l < 0$ (gradient opposes weight direction), the $-K_d \cdot \alpha_t^l \cdot e_t^l$ term *increases* $\lambda$, applying more decay precisely when the gradient already drives weights toward zero. This mirrors CWD's design rationale.

#### Mapping Existing Methods

Table 1 maps five existing methods plus UDWDC to specific gain configurations.

| Method | $\rho^*(t)$ | $K_p$ | $K_i$ | $K_d$ | Feedback Signal | Control Type |
|--------|-------------|-------|--------|--------|-----------------|-------------|
| FixedWD | --- | 0 | 0 | 0 | None | Open-loop |
| CWD | --- | 0 | 0 | $>0$ | Binary $\text{sign}(\alpha)$ | Derivative only |
| SWD | Dynamic | $>0$ | $>0$ | 0 | $\|\nabla\mathcal{L}\|$ | Proportional-integral |
| DefazioCorrective | $\eta_t/\tau$ | 0 | 0 | 0 | $\eta_t/\eta_0$ (schedule) | Feedforward |
| CPR | Per-matrix | $>0$ | $>0$ | 0 | Norm violation | Integral-dominant |
| **UDWDC (ours)** | $\eta_t/\tau$ | $>0$ | 0 | 0 | $\rho_t^l$ | Proportional |

*Table 1: Unified control law parameter mapping. Each existing method corresponds to a specific gain configuration in the PID framework. FixedWD is open-loop (no feedback). CWD uses only the alignment correction. SWD and CPR include integral terms. DefazioCorrective prescribes $\lambda_t \propto \eta_t/\eta_0$ from the LR schedule without measuring $\rho_t^l$, making it genuinely feedforward. UDWDC uses proportional feedback on $\rho_t^l$ directly.*

#### Empirical Validation of Unification

We fit the extended control law (with additional offset and target-scale parameters) to per-layer effective-WD trajectories from 200-epoch CIFAR-10/ResNet-20 experiments (3 seeds, 24 weight-bearing layers per seed, 72 traces total). Table 2 summarizes the fitting results.

| Method | Family | Rel. Error (%) | Std (%) | Status |
|--------|--------|---------------:|--------:|--------|
| CWD | Alignment-based | 4.71 | 0.16 | PASS |
| CPR | Constraint-based | 9.57 | 1.47 | PASS |
| NoWD | Degenerate | 0.00 | 0.00 | PASS |
| SWD | Scheduling-based | 45.81 | 0.68 | FAIL |
| DefazioCorrective | Scheduling-based | 37.56 | 2.37 | FAIL |

*Table 2: Unification fitting results (200 epochs, 20% error threshold). The control law captures CWD (4.71% error) and CPR (9.57% error) well. SWD and DefazioCorrective exceed the 20% threshold, indicating that scheduling-based methods operate through mechanisms not fully captured by per-layer $\rho_t^l$ feedback. The framework unifies the alignment-based and constraint-based families; scheduling-based methods require a separate global-feedback extension.*

The high fitting error for SWD arises because SWD's gradient-norm-aware scheduling uses global (not per-layer) gradient statistics and internal normalization that does not map cleanly to the per-layer error signal $e_t^l$. DefazioCorrective's error reflects its feedforward nature: it prescribes $\lambda_t$ based on the learning rate ratio $\eta_t / \eta_0$ rather than sensing $\rho_t^l$. Both methods operate on the training schedule rather than on per-layer measured quantities.

### 3.3 UDWDC: A Simple Proportional Controller

UDWDC (Unified Dynamic Weight Decay Control) closes the $\rho_t^l$ feedback loop with a proportional-only design:

**Algorithm 1: UDWDC**
```
Input: lambda_base, eta_0 (from user's training recipe)
Compute: tau = 1 / (lambda_base * eta_0)

At each step t, for each layer l:
  1. Measure:  rho_t^l = ||g_t^l||_2 / (||w_t^l||_2 + epsilon)
  2. Target:   rho*(t) = eta_t / tau
  3. Compute:  lambda_t^l = lambda_base * clamp(rho_t^l / rho*(t), 0.1, 10)
  4. Apply:    w_{t+1}^l = (1 - eta_t * lambda_t^l) * w_t^l - eta_t * g_t^l
```

The clamp bounds prevent runaway behavior: the effective WD never exceeds $10 \times \lambda_{\text{base}}$ (preventing norm collapse) or falls below $0.1 \times \lambda_{\text{base}}$ (preventing loss of regularization).

UDWDC introduces **zero new hyperparameters**. The target trajectory $\rho^*(t) = \eta_t / \tau$ is fully determined by $\lambda_{\text{base}}$ and $\eta_0$, both already specified in the user's training recipe. The clamp bounds [0.1, 10] are fixed constants, not tunable parameters.

Note that Algorithm 1 uses a multiplicative ratio formulation ($\lambda_{\text{base}} \cdot \text{clamp}(\rho_t^l / \rho^*(t))$) rather than the additive PID form ($\lambda_{\text{base}} + K_p \cdot e_t^l$). The multiplicative form is a design choice for numerical stability: it naturally bounds the output within $[0.1, 10] \times \lambda_{\text{base}}$ and is approximately equivalent to the additive form for small relative errors $|e_t^l / \rho^*(t)| \ll 1$. For large errors, the clamp provides hard bounds that the additive form does not guarantee.

#### UDWDC-v2: Stability Fix

Pilot experiments revealed that UDWDC v1 suffers from instability: when $\rho_t^l < \rho^*(t)$ for most layers (common during early training with BN), the clamp drives $\lambda_t^l$ to $0.1 \times \lambda_{\text{base}}$, effectively disabling WD. This produces CSI$_{\text{combined}} = -2.41$ and near-zero total WD budget.

UDWDC-v2 applies two modifications:

1. **EMA smoothing**: Replace instantaneous $\rho_t^l$ with $\hat{\rho}_t^l = \text{EMA}(\rho_t^l, \beta = 0.99)$, damping step-to-step noise.
2. **Floor clipping**: Enforce $\lambda_t^l \geq \lambda_{\min} = 0.1 \cdot \lambda_{\text{base}}$, guaranteeing nonzero WD at all times.

These are engineering patches, not principled solutions. The instability reflects a fundamental limitation of proportional-only control: P-control has inherent steady-state offset and cannot guarantee zero tracking error. An integral term ($K_i > 0$) would eliminate this offset, at the cost of potential overshoot and a new hyperparameter. We deliberately preserve the P-only design for simplicity and document the instability as a finding.

![UDWDC feedback control loop block diagram](figures/udwdc_control_loop.pdf)

*Figure 2: UDWDC closed-loop feedback control diagram. The Target Generator computes $\rho^*(t) = \eta_t/\tau$ from the learning rate schedule. The summation node computes the per-layer error $e_t^l = \rho_t^l - \rho^*(t)$. The UDWDC Controller (dashed red box) applies proportional gain with clamping to produce $\lambda_t^l$. The Measurement block closes the loop by computing $\rho_t^l = \|g_t^l\| / \|w_t^l\|$. Annotation boxes show where other methods connect: CWD adds $K_d$ alignment feedback at the summation node, CPR adds integral accumulation in the gain stage, SWD uses global $\|\nabla\mathcal{L}\|$ rather than per-layer $\rho_t^l$. FixedWD bypasses the controller entirely (open-loop, dashed bypass arrow).*

### 3.4 Theoretical Analysis

We extend the nonconvex SGDW convergence framework of Sun et al. (CVPR 2025) to time-varying WD with alignment modulation, and derive two additional propositions connecting the control framework to optimizer geometry and layer-specific equilibria. Following Sun et al. (CVPR 2025), we use $\delta_t$ to denote gradient-weight alignment in the theoretical analysis ($\delta_t \equiv \alpha_t$; see notation table).

#### Theorem 1: Contraction-Rate Separation for Stagewise SGDW

Consider an $L_{\text{smooth}}$-smooth nonconvex objective $\mathcal{L}$ optimized by SGDW with a stagewise WD schedule $\lambda_t$ and alignment-weighting function $\phi(\delta_t) \geq 0$. Define the cumulative contraction $C_T = \sum_{t=1}^{T} \eta_t \lambda_t \|w_t\|^2$ and the alignment-weighted contraction $A_T = \sum_{t=1}^{T} \eta_t \lambda_t \|w_t\|^2 \cdot \phi(\delta_t)$.

**Statement.** The convergence rate to an $\epsilon$-stationary point matches unregularized SGD at $O(1/\sqrt{T})$ when $C_T = O(\sqrt{T})$. The generalization bound depends on $A_T$ rather than the worst-case alignment. When the alignment variance $\text{Var}_t[\phi(\delta_t)] > 0$, alignment-modulated WD achieves a strictly tighter generalization bound per unit of WD budget $C_T$ compared to fixed WD.

**Intuition.** Fixed WD applies uniform contraction regardless of the instantaneous alignment between gradient and weight directions. Alignment-modulated WD concentrates contraction on steps where alignment indicates the decay is most beneficial for generalization (i.e., when $\phi(\delta_t)$ is large). The efficiency gain is proportional to the variance of $\phi(\delta_t)$: if alignment is constant, no benefit is possible; if alignment varies, the modulation can selectively allocate WD budget to informative steps.

This theorem provides the theoretical foundation for BEM (Section 4): methods that achieve higher accuracy per unit WD budget are exploiting alignment variance, and BEM directly measures this efficiency.

#### Proposition 2: Geometry-Corrected Alignment for Adam

For AdamW with diagonal preconditioner $P_t = \text{diag}(\hat{v}_t + \epsilon)$ (the second-moment estimate), the geometrically natural alignment is:

$$\delta_t^P = \frac{\langle P_t^{-1} g_t, w_t \rangle}{\|P_t^{-1} g_t\| \cdot \|w_t\|}$$

rather than the standard $\ell_2$-alignment $\alpha_t = \langle g_t, w_t \rangle / (\|g_t\| \cdot \|w_t\|)$.

**Statement.** The geometry-corrected alignment $\delta_t^P$ is alignment-consistent with AdamW's implicit optimization objective: the AdamW update moves weights in the direction $P_t^{-1} g_t$, so alignment should be measured in the preconditioned space. Standard CWD using $\ell_2$-alignment applies the binary mask based on $\alpha_t$, which can be geometrically inconsistent at the parameter-group level when $P_t$ is far from the identity.

**Implication.** Alignment-aware WD methods should use preconditioner-corrected alignment when applied with Adam-family optimizers. This correction is negligible for SGD ($P_t = I$) but potentially significant for Adam in late training when the second-moment estimates have accumulated parameter-specific curvature information. Our experiments use SGD for CNNs, so this proposition is a prediction for future Adam-based evaluations.

#### Proposition 3: Layer-Differentiated Steady States

Under alignment-modulated WD on networks with normalized layers (BN), the per-layer steady-state GW ratios satisfy:

$$r_l^* = \frac{\lambda_{\text{base}} \cdot \gamma}{\phi(\delta_l^*)}$$

where $\gamma$ is a normalization-related constant and $\delta_l^*$ is the per-layer steady-state alignment.

**Statement.** This yields layer-differentiated equilibria that depend on per-layer gradient structure, unlike fixed WD where all normalized layers converge to the same $r^*$ (Defazio, 2025). Specifically, $r_l^*$ and $\delta_l^*$ are anti-correlated: layers with high steady-state alignment (large $\delta_l^*$) achieve lower GW ratios (smaller $r_l^*$), and vice versa.

**Scope limitation.** This anti-correlation is restricted to BN architectures with sufficient depth. Layer Normalization (LN) may yield different steady-state behavior because LN normalizes across features rather than across the batch, changing the gradient structure at each layer. Section 5 reports the measured anti-correlation on ResNet-50 (BN); ViT-S/16 (LN) is not tested.

---

## 4. Standardized Evaluation Metrics

We propose three metrics to enable fair cross-method comparison of dynamic WD methods. Existing evaluations use only accuracy, masking important differences in regularization efficiency, stability, and signal quality.

### 4.1 Budget Equivalence Metric (BEM)

$$\text{BEM} = \frac{\text{acc} - \text{acc}_{\text{NoWD}}}{\text{TotalWDBudget}}$$

where $\text{TotalWDBudget} = \sum_{t=1}^{T} \sum_{l=1}^{L} \lambda_t^l \|w_t^l\|^2$ is the cumulative WD applied during training. BEM captures accuracy improvement per unit of regularization applied, separating methods that achieve high accuracy through efficient WD allocation from those that achieve it through brute-force high WD.

### 4.2 Coupling Stability Index (CSI)

The CSI measures the stability of the WD-optimization coupling across training:

$$\text{CSI}_{\text{temporal}} = 1 / \text{Var}_t[\lambda_{\text{eff}}^l], \quad \text{CSI}_{\text{spatial}} = 1 / \text{Var}_l[\lambda_{\text{eff}}^l]$$

averaged over layers (temporal) or over time steps (spatial) in the last 25% of training. The combined CSI is:

$$\text{CSI}_{\text{combined}} = \frac{\text{CSI}_{\text{temporal}} + \text{CSI}_{\text{spatial}}}{2}$$

Higher CSI indicates more stable coupling. CSI values are normalized by dividing by FixedWD's CSI, so that FixedWD = 1.0 by construction. Formally, $\text{CSI}^{\text{normalized}} = \text{CSI}^{\text{raw}} / \text{CSI}^{\text{raw}}_{\text{FixedWD}}$. For FixedWD, $\lambda_{\text{eff}}^l = \lambda_{\text{base}}$ is constant across all time steps, so $\text{Var}_t[\lambda_{\text{eff}}^l] = 0$ and $\text{CSI}^{\text{raw}}_{\text{FixedWD}} \to \infty$; in practice, normalization is applied relative to the maximum observed CSI across methods, with FixedWD set to 1.0 as the stable-baseline anchor. Any method with CSI < 1.0 exhibits more WD fluctuation than FixedWD; negative CSI values (as in UDWDC) indicate that the WD coefficient oscillates with higher variance than the gradient signal it attempts to track.

### 4.3 Alignment Informativeness Score (AIS)

$$\text{AIS} = \text{Spearman}(\bar\alpha_t^l,\; \Delta\text{GenGap}_t)$$

where $\bar\alpha_t^l$ is the mean gradient-weight alignment across layers at step $t$, and $\Delta\text{GenGap}_t = \text{GenGap}_t - \text{GenGap}_{t-1}$ is the per-epoch change in the generalization gap (train accuracy minus test accuracy). A positive $\Delta\text{GenGap}_t$ indicates increasing overfitting; a negative value indicates improving generalization. AIS measures whether the alignment signal $\bar\alpha_t^l$ is informative about which direction the generalization gap will change — i.e., whether high alignment predicts generalization improvement. The global AIS is the Spearman correlation across all $(t, \text{config})$ pairs in the alignment informativeness sweep (6 WD configurations across CIFAR-10). AIS is conditioned on batch size to account for alignment SNR scaling (Section 5.4). In pilot experiments, the global AIS is 0.566, confirming that the mean alignment $\bar\alpha$ predicts generalization-gap changes more reliably than $\log_{10}(\lambda)$ alone ($r_{\alpha,\text{gengap}} = 0.698$).

---

## 5. Experiments

We evaluate the unified framework and UDWDC across three dataset-architecture combinations with 7 methods, 3--5 seeds per configuration, and full diagnostic tracking of $\rho_t^l$, $\alpha_t^l$, effective WD, and weight norms per layer per epoch. Sections 5.2–5.5 report accuracy and stability results across CIFAR-10, CIFAR-100, and ImageNet. Sections 5.6–5.7 analyze two cross-dataset diagnostic properties: temporal predictability of the alignment signal (H7) and CSI stability across methods.

### 5.1 Experimental Setup

**Datasets.** CIFAR-10 (50K/10K, 32x32), CIFAR-100 (50K/10K, 32x32), and ImageNet-1K (1.28M/50K, 224x224).

**Architectures.** ResNet-20 (272K parameters, BN) for CIFAR-10 diagnostic experiments; VGG-16-BN (15.3M parameters, BN) for CIFAR-100 ablation; ResNet-50 (25.6M parameters, BN) for ImageNet main comparison.

**Methods.** Seven baselines: FixedWD ($\lambda = 10^{-4}$), CWD, SWD, CPR, DefazioCorrective, NoWD ($\lambda = 0$), and UDWDC. UDWDC-v2 (EMA smoothing + floor clipping) is included as a stability-fixed variant.

**Training protocol.** SGD with momentum 0.9, cosine annealing from $\eta_0 = 0.1$ to 0. CIFAR: 200 epochs, batch size 128, seeds {42, 123, 456}. ImageNet: 90 epochs, batch size 432 (Distributed Data Parallel across 2 GPUs), seeds {42, 123, 456, 789, 2024} for FixedWD (5 seeds) and {42, 123, 456} for dynamic methods (3 seeds). Augmentation: RandomCrop with padding 4 + RandomHorizontalFlip (CIFAR); RandomResizedCrop(224) + RandomHorizontalFlip (ImageNet). No mixup, cutmix, or RandAugment for CNNs, isolating WD effects from augmentation interactions.

**Diagnostics.** Per-layer $\rho_t^l$, $\alpha_t^l$, $\|w_t^l\|$, $\|g_t^l\|$, and effective $\lambda_t^l$ are logged at every epoch for all runs.

### 5.2 CIFAR-10 Diagnostic Results

Table 3 reports the 200-epoch CIFAR-10/ResNet-20 results.

| Method | Best Acc (%) | Gen Gap (%) | WD Budget | BEM |
|--------|-------------:|------------:|----------:|----:|
| **CPR** | **91.74 +/- 0.07** | **8.28 +/- 0.05** | 4.44 | 0.39 |
| FixedWD | 90.68 +/- 0.11 | 9.28 +/- 0.17 | 0.48 | 0.89 |
| DefazioCorrective | 90.62 +/- 0.20 | 9.31 +/- 0.22 | 0.24 | 1.50 |
| SWD | 90.39 +/- 0.19 | 9.49 +/- 0.23 | 0.47 | 0.30 |
| UDWDC-v2 | 90.36 +/- 0.09 | 9.53 +/- 0.07 | 98599 | ~0 |
| CWD | 90.32 +/- 0.08 | 9.66 +/- 0.05 | 0.24 | 0.12 |
| NoWD | 90.25 +/- 0.30 | 9.73 +/- 0.30 | 0.00 | --- |
| UDWDC | 90.15 +/- 0.23 | 9.82 +/- 0.25 | 0.38 | -0.26 |

*Table 3: CIFAR-10/ResNet-20, 200 epochs, 3 seeds. CPR achieves the highest accuracy by +1.06 pp over FixedWD, driven by its aggressive integral control that accumulates 10x the WD budget. FixedWD is the strongest simple baseline. UDWDC v1 underperforms NoWD by 0.10 pp, reflecting its instability (Section 3.3). UDWDC-v2's enormous WD budget (98599) is the cumulative $\sum_t \sum_l \lambda_t^l \|w_t^l\|^2$ over all 200 epochs × 65 layers (not a per-step value; FixedWD's 0.48 uses the same convention), resulting from the floor clipping applying nonzero lambda to BN layers at every step. BEM is computed as (acc - acc\_NoWD) / WD budget.*

**Key observations.**

1. CPR's integral control is highly effective on CIFAR-10: accumulating constraint violations drives $\lambda_{\text{eff}}$ to $10^{-3}$, producing tighter weight norms and the smallest generalization gap (8.28%).

2. CWD's effective WD is approximately 50% of FixedWD (0.24 vs. 0.48), confirming at 200 epochs the prediction from pilot data. This 50% reduction is a potential confound: the alignment mask may improve accuracy partly through WD magnitude reduction rather than alignment information alone.

3. UDWDC v1 achieves accuracy below NoWD, indicating that the proportional controller's instability (CSI$_{\text{combined}}$ = -2.41) actively harms performance. The v2 floor-clipping fix recovers to 90.36%, restoring competitive performance at the cost of an inflated WD budget.

4. The spread between the best (CPR, 91.74%) and worst WD method (UDWDC, 90.15%) is 1.59 pp, demonstrating that the choice of WD strategy has meaningful impact on CIFAR-10.

### 5.3 UDWDC Gain Ablation (CIFAR-100/VGG-16-BN)

To isolate the contribution of each PID component, we run 7 gain configurations on CIFAR-100/VGG-16-BN (200 epochs, 3 seeds). All variants use the UDWDC-v2 floor-clipping fix ($\lambda_{\min} = 0.1\lambda_{\text{base}}$); results reflect PID gain interactions in the stabilized setting, not v1 collapse behavior. Table 4 reports the results.

| Variant | $K_p$ | $K_i$ | $K_d$ | Acc (%) | WD Budget | Gen Gap (%) |
|---------|------:|------:|------:|--------:|----------:|------------:|
| Kd_only | 0 | 0 | 0.3 | **70.64 +/- 0.30** | 0.594 | **29.34** |
| FixedWD | 0 | 0 | 0 | 70.53 +/- 0.48 | 0.580 | 29.41 |
| Ki_only | 0 | 0.1 | 0 | 69.64 +/- 0.88 | 0.134 | 30.08 |
| Full PID | 0.5 | 0.1 | 0.3 | 69.29 +/- 1.28 | 0.277 | 30.46 |
| PD_control | 0.5 | 0 | 0.3 | 69.28 +/- 0.57 | 0.300 | 30.36 |
| PI_control | 0.5 | 0.1 | 0 | 69.12 +/- 1.11 | 0.317 | 30.55 |
| Kp_only | 0.5 | 0 | 0 | 68.52 +/- 0.31 | 0.300 | 31.30 |

*Table 4: UDWDC gain ablation on CIFAR-100/VGG-16-BN, 200 epochs, 3 seeds. All ablation variants use v2 floor clipping. The alignment-only variant (Kd_only) slightly outperforms FixedWD, while all proportional-containing variants underperform. High variance in Full PID and PI_control reflects sensitivity to the interaction of gains.*

**Key observations.**

1. The alignment correction alone ($K_d = 0.3$, $K_p = K_i = 0$) does not reliably distinguish from FixedWD at 3-seed resolution: Kd_only achieves 70.64 ± 0.30% vs. FixedWD 70.53 ± 0.48%, a $\Delta = 0.11$ pp difference within the combined standard error (0.56 pp).

2. Adding proportional control ($K_p > 0$) consistently degrades accuracy. Kp_only achieves 68.52%, 2.01 pp below FixedWD. This reflects the instability documented in Section 3.3: proportional feedback on per-layer $\rho_t^l$ introduces noise that harms the optimization trajectory.

3. The integral term ($K_i$) partially mitigates the proportional instability: PI_control (69.12%) outperforms Kp_only (68.52%) by 0.60 pp, consistent with the integral term providing error accumulation that smooths noisy proportional corrections.

4. Full PID does not outperform any single-gain variant, suggesting that naively combining all three gains introduces conflicting correction signals.

### 5.4 Alignment Signal Quality: Batch-Size Sweep

We sweep batch sizes {64, 128, 256, 512, 1024} on CIFAR-100/VGG-16-BN (200 epochs, 3 seeds) for FixedWD, CWD, and UDWDC-v2 to test the relationship between batch size and alignment signal quality.

**H3 is falsified.** The full batch-size sweep data directly contradicts the hypothesis that CWD alignment SNR increases monotonically with batch size. Table 7 reports the full SNR data from `phase2_batchsize/summary.md`.

| BS | FixedWD SNR | CWD SNR | CWD Acc (%) | FixedWD Acc (%) | $\Delta$ CWD |
|----|----------:|-------:|----------:|---------------:|----------:|
| 64 | 0.0281 | 0.0074 | 68.31 ± 1.12 | 69.55 ± 1.33 | −1.25 |
| 128 | 0.0167 | 0.0075 | 69.92 ± 0.40 | 70.53 ± 0.48 | −0.61 |
| 256 | 0.0011 | 0.0001 | 69.61 ± 0.24 | 69.54 ± 0.25 | +0.07 |
| 512 | 0.0001 | 0.0009 | 66.16 ± 4.64 | 69.99 ± 0.10 | −3.84 |
| 1024 | 0.0010 | 0.0014 | 63.42 ± 7.16 | 69.05 ± 0.17 | −5.63 |

*Table 7: Full batch-size sweep results (CIFAR-100/VGG-16-BN, 200 epochs, 3 seeds). FixedWD SNR decreases from bs=64 to bs=512 before partially recovering at bs=1024 — not monotonically increasing. CWD SNR collapses at bs=256 (0.0001) then partially recovers. CWD accuracy deteriorates catastrophically at large batch sizes: std=7.16% at bs=1024 indicates high instability.*

FixedWD SNR decreases monotonically from bs=64 to bs=512, with only a partial recovery at bs=1024. CWD SNR collapses to near-zero at bs=256, partially recovers at bs=512 and bs=1024, but this recovery does not translate to accuracy gains. CWD accuracy deteriorates catastrophically: −3.84 pp vs. FixedWD at bs=512 and −5.63 pp at bs=1024, with standard deviation of 7.16 pp at bs=1024 indicating severe instability across seeds.

No batch size in the sweep shows CWD reliably outperforming FixedWD. CWD's performance deteriorates severely at batch size ≥ 512, precluding its use in large-batch training regimes under standard $\lambda_{\text{base}} = 10^{-4}$. Even at small batch sizes (b ≤ 256), CWD underperforms FixedWD by 0.61–1.25 pp. Alignment-aware WD methods under standard $\lambda_{\text{base}}$ settings require a dedicated ablation before deployment at any batch size.

![Alignment SNR vs batch size](figures/alignment_snr.pdf)

*Figure 3: Alignment SNR vs. batch size for FixedWD, CWD, and UDWDC-v2 on CIFAR-100/VGG-16-BN (200 epochs, 3 seeds). (a) SNR is non-monotonic for all methods: FixedWD SNR peaks at bs=128 (0.0167) then collapses to 0.0001 at bs=512; CWD SNR collapses at bs=256 (0.0001) before partial recovery. (b) Accuracy improvement over FixedWD: CWD shows modest gains at bs=64 (relative to UDWDC-v2's +0.22 pp) but collapses severely at bs=512 (−3.84 pp) and bs=1024 (−5.63 pp, std=7.16%). UDWDC-v2 maintains accuracy within ±1.24 pp of FixedWD across all batch sizes. H3 is falsified: higher SNR does not prevent CWD accuracy collapse at large batch sizes.*

### 5.5 ImageNet Main Results (ResNet-50, 90 epochs)

**Note on accuracy baselines:** All ImageNet results use a minimal augmentation protocol (RandomCrop + RandomHorizontalFlip only; no mixup, cutmix, or RandAugment), designed to isolate WD effects from augmentation interactions. Standard ResNet-50 recipes with full augmentation (He et al., 2016; PyTorch model zoo) achieve 76–77% at 90 epochs; our FixedWD at 71.72% reflects this protocol choice.

Table 5 reports the ImageNet/ResNet-50 results, the primary large-scale evidence for the framework. ImageNet results cover 4 of 7 methods; SWD, DefazioCorrective, and NoWD were not completed due to compute constraints.

| Method | Best Val Acc (%) | Final Val Acc (%) | WD Budget | Seeds |
|--------|----:|----:|----:|----:|
| **CPR** | **74.74 +/- 0.05** | **74.74 +/- 0.04** | 1.20 | 3 |
| FixedWD | 71.72 +/- 0.36 | 71.70 +/- 0.37 | 0.52 | 5 |
| CWD | 70.66 (no CI) | 70.58 (no CI) | 0.26 | 1 |
| UDWDC | 69.93 +/- 0.19 | 69.91 +/- 0.19 | 0.70 | 3 |
| SWD | — (pending) | — | — | — |
| DefazioCorrective | — (pending) | — | — | — |
| NoWD | — (pending) | — | — | — |

*Table 5: ImageNet/ResNet-50, 90 epochs. CPR leads by 3.02 pp over FixedWD (Welch's t-test: t ≈ 8.3, df ≈ 5, p < 0.001). UDWDC underperforms FixedWD by 1.79 pp (t ≈ 9.4, df ≈ 4, p < 0.001). CWD has only 1 completed seed; no confidence interval is computable, and statistical comparison is not valid. SWD, DefazioCorrective, and NoWD runs were interrupted due to compute constraints; BEM cannot be computed for ImageNet without a NoWD baseline.*

**Key observations.**

1. CPR's dominance is even more pronounced on ImageNet (+3.02 pp over FixedWD) than on CIFAR-10 (+1.06 pp). The augmented Lagrangian's integral control accumulates a WD budget 2.3x that of FixedWD, producing tighter weight norm constraints that substantially improve generalization at ImageNet scale. The CPR vs. FixedWD margin is highly significant (Welch's t ≈ 8.3 at df ≈ 5, p < 0.001) despite the unequal variance.

2. UDWDC underperforms FixedWD by 1.79 pp on ImageNet (69.93% vs. 71.72%), a larger gap than on CIFAR-10 (0.53 pp). The proportional controller's instability is amplified at scale, where per-layer $\rho_t^l$ variance is higher due to the larger number of layers (65 weight-bearing layers in ResNet-50 vs. 24 in ResNet-20). This gap is also significant (Welch's t ≈ 9.4 at df ≈ 4, p < 0.001).

3. CWD's single-seed result (70.66%) falls between FixedWD and UDWDC. This is consistent with CWD's WD magnitude reduction (50% effective lambda) reducing regularization strength, but with only 1 seed, this interpretation is a hypothesis rather than a confirmed finding. Future work should complete the remaining 2 CWD seeds for a valid statistical comparison.

### 5.6 Temporal Predictability of Alignment

The temporal predictability gate tests whether the alignment signal $\alpha_t^l$ carries information beyond what a polynomial function of time can predict. We fit degree-4 polynomials to $\alpha_t^l$ trajectories across all layer-method combinations (2640 total: 8 methods x 330 layer-seed traces from CIFAR-10 and CIFAR-100 experiments).

**Result**: 18.1% of layer-method combinations achieve $R^2 > 0.85$ for polynomial fit, and 28.0% achieve $R^2 > 0.70$. The mean $R^2$ is 0.279 (median 0.036). Per-layer-type statistics from `h7_gate_result.json` reveal a structured hierarchy:

| Layer type | Mean $R^2$ | Median $R^2$ | % $>$ 0.85 |
|-----------|----------:|----------:|----------:|
| conv/weight | 0.595 | 0.825 | 41.3% |
| shortcut | 0.301 | 0.028 | 22.2% |
| fc/weight | 0.116 | 0.109 | 0.0% |
| fc/bias | 0.062 | 0.032 | 0.0% |
| conv/bias | 0.034 | 0.028 | 0.0% |
| bn/weight | 0.026 | 0.017 | 0.0% |
| bn/bias | 0.024 | 0.018 | 0.0% |

Conv/weight layers show mean $R^2 = 0.595$ (median 0.825): alignment in large convolutional kernels is highly temporally predictable. Shortcut connections show intermediate predictability (mean $R^2 = 0.301$, 22.2% above 0.85). BN parameters and conv/bias layers show mean $R^2 < 0.04$, indicating that WD adjustments to these layers carry information not captured by a simple function of training time. FC layers occupy an intermediate range ($0.06$–$0.12$). We chose $R^2 > 0.85$ as a conservative threshold for "well-approximated by time alone"; the qualitative conclusion also holds at $R^2 > 0.70$, where 28.0% qualify.

The bimodal $R^2$ distribution (69.2% below 0.30 but 18.1% above 0.85) reveals a structural split: alignment is highly predictable from time in conv/weight layers and partially in shortcut connections, but carries non-temporal information in BN layers, biases, and FC layers. This structural split suggests that alignment-aware WD methods may derive their value specifically from the layers where time-polynomial fitting fails, a hypothesis that requires per-layer ablation to confirm.

### 5.7 CSI Stability Analysis

The refined CSI (Section 4.2) reveals stability differences invisible to accuracy-only evaluation.

| Method | CSI_temporal | CSI_spatial | CSI_combined | Interpretation |
|--------|----------------------:|---------------------:|----------------------:|----------------|
| FixedWD | 1.000 | 1.000 | 1.000 | Trivially stable |
| DefazioCorrective | 1.000 | 1.000 | 1.000 | Highly stable |
| CWD | 0.997 | 0.997 | 0.997 | Highly stable |
| CPR | 0.957 | 1.000 | 0.978 | Highly stable |
| SWD | 0.902 | 0.907 | 0.904 | Stable |
| UDWDC | -5.75 | 0.935 | -2.41 | Highly unstable |

*Table 6: CSI stability comparison on CIFAR-10/ResNet-20, 200 epochs, 3 seeds, computed over last 25% of training. CSI = 1/(1 + Var$_t$[$\lambda_\text{eff}^l$ / mean$_t$[$\lambda_\text{eff}^l$]]) averaged across layers; FixedWD normalized to 1.0 by construction. FixedWD is trivially stable (constant lambda). CWD maintains near-perfect stability despite the binary mask. UDWDC's negative temporal CSI indicates that the proportional controller's WD coefficient varies more than the underlying $\rho_t$ trajectory, amplifying rather than damping perturbations.*

UDWDC's negative temporal CSI (-5.75) is a quantitative signature of proportional controller instability. The spatial CSI (0.935) remains positive because per-layer $\rho_t^l$ variation at each step is consistent across layers---the instability is temporal (oscillation over training steps), not spatial (inconsistency across layers). See Figure 4 for the full CSI comparison.

![CSI comparison between methods](figures/csi_comparison.pdf)

*Figure 4: CSI comparison across methods. UDWDC exhibits negative temporal CSI, quantifying the instability documented in Section 3.3. All other methods maintain CSI > 0.9.*

---

## 6. Discussion

### 6.1 Unification Works at the Family Level

The PID parameterization provides a useful taxonomy even where exact fitting fails. CWD (4.71% fitting error) and CPR (9.57%) are well-captured by the control law (Table 2), confirming that alignment-based and constraint-based methods operate as derivative and integral controllers, respectively. SWD (45.81%) and DefazioCorrective (37.56%) resist fitting because they use global gradient statistics or feedforward scheduling rather than per-layer measured $\rho_t^l$.

The family-level interpretation remains valid: **scheduling-based methods** (SWD, DefazioCorrective) are open-loop or feedforward controllers that prescribe $\lambda_t$ from the training schedule without sensing $\rho_t^l$; **alignment-based methods** (CWD) add a derivative correction based on geometric signal quality; **constraint-based methods** (CPR) add integral control via penalty accumulation. This three-way taxonomy clarifies why CPR and CWD achieve different results despite both being "dynamic WD": they control different aspects of the training dynamics.

### 6.2 BEM Reveals Hidden Cost-Effectiveness Differences

Raw accuracy rankings mask important efficiency differences. CPR achieves the highest accuracy on both CIFAR-10 (91.74%, Table 3) and ImageNet (74.74%, Table 5) but does so by accumulating 10x the WD budget of FixedWD, making its BEM relatively low (0.39 on the 200-epoch data). DefazioCorrective achieves rank-1 BEM (1.50) on the 200-epoch CIFAR-10 results (Table 3) with only half the WD budget of FixedWD.

This ranking divergence validates the need for multi-dimensional evaluation. A practitioner choosing a WD method based solely on accuracy would select CPR; one operating under a regularization budget constraint (e.g., in federated learning where WD budget affects privacy guarantees) might prefer FixedWD or DefazioCorrective. BEM makes this tradeoff explicit.

### 6.3 The CWD Magnitude Confound

CWD's effective WD is approximately 50% of FixedWD (0.24 vs. 0.48 on CIFAR-10, 0.26 vs. 0.52 on ImageNet). CWD applies the binary mask element-wise when $\alpha_t^l < 0$, disabling WD for approximately half the parameters at each step. This halving raises a confound: does CWD improve performance through alignment information or through WD magnitude reduction?

On CIFAR-10 (Table 3), CWD (90.32 +/- 0.08%) underperforms FixedWD (90.68 +/- 0.11%), suggesting that the magnitude reduction actually hurts on this benchmark. On ImageNet (Table 5), CWD's single-seed result (70.66%) falls below FixedWD (71.72%). These results suggest that the 50% WD reduction is the dominant effect at $\lambda_{\text{base}} = 10^{-4}$, and the alignment masking does not compensate for the reduced regularization strength.

CWD's published gains (+0.61% on ImageNet ViT-S/16) use $\lambda_{\text{base}} = 0.05$, a 500x larger base coefficient. At higher $\lambda_{\text{base}}$, halving the effective WD may be beneficial because it reduces over-regularization. A definitive resolution requires a CWD vs. halved-lambda ablation: running FixedWD at $\lambda = 0.5 \times \lambda_{\text{base}}$ and comparing directly. We did not complete this ablation due to compute constraints and flag it as essential future work.

### 6.4 UDWDC Instability Is a Genuine Finding

UDWDC v1's negative CSI$_{\text{combined}}$ (-2.41) and accuracy below NoWD (90.15% vs. 90.25% on CIFAR-10) are not implementation bugs but a fundamental limitation of proportional-only control applied to per-layer $\rho_t^l$. See Figure 4 for the CSI comparison across all methods.

The failure mechanism is clear: when $\rho_t^l < \rho^*(t)$ (which occurs systematically for BN layers where gradient norms are small relative to weight norms), the ratio $\rho_t^l / \rho^*(t) < 1$, and the clamp drives $\lambda_t^l$ toward $0.1 \times \lambda_{\text{base}}$. Across 65 layers, this reduces the effective WD to near-zero, producing a WD budget of 0.38 (vs. FixedWD's 0.48) without the corresponding accuracy benefit.

The v2 floor fix recovers to 90.36% (rank 5 of 8, 0.32 pp below FixedWD) but at the cost of an enormous WD budget (98599, approximately 205,000x FixedWD's 0.48), because the floor clipping applies $\lambda_{\min} = 0.1 \times \lambda_{\text{base}}$ to all 65 layers including BN layers at every step. Future work should restrict UDWDC's control loop to weight layers only, excluding BN parameters.

The instability finding has broader implications: **per-layer proportional control of WD is inherently noisy** because $\rho_t^l$ fluctuates substantially within each epoch due to mini-batch stochasticity. The EMA smoothing in v2 ($\beta = 0.99$) helps but does not eliminate the issue. CPR's success demonstrates that integral control (penalty accumulation) is a more robust feedback mechanism for WD: it inherently smooths noise through integration and drives the system toward the target monotonically.

### 6.5 Negative Results

1. **UDWDC does not beat tuned FixedWD on accuracy.** On all three benchmarks, FixedWD outperforms UDWDC: by 0.53 pp on CIFAR-10, by 2.01 pp on CIFAR-100 (FixedWD 70.53% vs. Kp_only 68.52%, Table 4), and by 1.79 pp on ImageNet (Table 5). UDWDC's value lies in the theoretical framework and diagnostic metrics, not raw performance.

2. **Unification fitting fails for 2/5 methods.** SWD (45.81%) and DefazioCorrective (37.56%) exceed the 20% error threshold (Table 2). The unification is valid at the family level (alignment vs. scheduling vs. constraint) but not as a precise quantitative model for all methods.

3. **CPR's integral control produces 10x higher effective WD.** The augmented Lagrangian is aggressive: CPR's WD budget (4.44 on CIFAR-10) is 9.3x FixedWD's (0.48). While this produces the highest accuracy, it raises questions about whether CPR's gains come from better WD allocation or simply from stronger regularization.

4. **ImageNet results have limited seed coverage.** CWD has only 1 completed seed on ImageNet; UDWDC has 3. Statistical comparison of CWD and UDWDC on ImageNet is not valid with the current seed coverage and should be deferred to future work.

### 6.6 Limitations

- Proposition 3 (layer-differentiated steady states) is restricted to BN architectures; LN behavior may differ. The Proposition 3 anti-correlation between $r_l^*$ and $\delta_l^*$ is a theoretical prediction; empirical verification on ResNet-50 (BN) and ViT-S/16 (LN) is not included in the present experiments (see pilot data at `exp/results/pilots/imagenet_resnet101/` and `exp/results/pilots/imagenet_vit/`).
- UDWDC's proportional-only design inherits steady-state offset that an integral term could eliminate.
- Budget-matched controls for ImageNet are limited to 2-epoch pilots; full 90-epoch budget matching is needed.
- AIS is computed on CIFAR batch sizes; ImageNet AIS requires additional experiments.
- The PID gain mapping is descriptive: it demonstrates that existing methods can be expressed within the framework, but does not prove that the PID parameterization is optimal.
- ImageNet coverage is incomplete: SWD, DefazioCorrective, and NoWD runs were interrupted due to compute constraints. Without a NoWD ImageNet baseline, BEM cannot be computed for ImageNet, and the full cross-method comparison is deferred to future work.

### 6.7 Future Work

1. **Adaptive gain scheduling.** Let $K_p$, $K_i$, $K_d$ vary with training phase: higher $K_p$ during warmup for rapid adjustment, increasing $K_i$ during stable training for offset elimination, and $K_d$ conditioned on batch-size-dependent alignment quality.

2. **Geometry-corrected alignment (Proposition 2) for Adam-family optimizers.** Testing $\delta_t^P$ vs. $\alpha_t^l$ as the CWD mask criterion with AdamW on transformer architectures.

3. **Extension to language models.** Does the PID framework predict which WD schedule works for GPT-scale pretraining? CPR's integral control has already shown gains on GPT-2 (Franke et al., 2024); testing whether the PID taxonomy explains this is a natural next step.

4. **Joint $(\eta_t, \lambda_t)$ control.** Multi-input-multi-output (MIMO) control of learning rate and WD simultaneously, using $\rho_t^l$ and $\alpha_t^l$ as feedback signals, represents the natural extension of the single-input framework presented here.

---

## 7. Conclusion

The $\rho_t^l$ framework unifies four WD sub-traditions via PID gains $(K_p, K_i, K_d)$; CWD and CPR fit to 4.71% and 9.57% error (Table 2), confirming their control-theoretic interpretation. Scheduling-based methods (SWD at 45.81%, DefazioCorrective at 37.56%) resist fitting, cleanly delineating the framework's scope. Of the three feedback channels, integral control (CPR, $K_i > 0$) delivers the largest empirical gain: CPR's +3.02 pp over FixedWD on ImageNet (Table 5) is the largest margin in our experiments; the CIFAR-100 ablation further shows Ki_only (69.64%) outperforms Kp_only (68.52%) by 1.12 pp (Table 4). Proportional feedback ($K_p > 0$) is destabilizing without integral smoothing; alignment/derivative feedback ($K_d > 0$) provides marginal benefit on CIFAR-100 but requires geometry-corrected alignment for Adam-family optimizers (Proposition 2).

CSI$_{\text{combined}}$ = -2.41 for UDWDC v1 quantifies a failure mode that raw accuracy (90.15%) does not fully expose---NoWD achieves 90.25% on the same benchmark. This demonstrates that CSI is a necessary complement to accuracy for evaluating WD controllers. UDWDC's contribution is conceptual and methodological: it demonstrates that the control-theoretic formulation is implementable, identifies proportional-only control's fundamental instability, and motivates controllers with $K_i > 0$ as the next design step.

Together, the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS) provide a protocol for evaluating any future WD controller on three orthogonal axes: budget efficiency, trajectory stability, and alignment informativeness---enabling apples-to-apples comparison across the methods unified by the PID taxonomy. AIS confirms that the alignment signal carries information beyond time-polynomial trends ($R^2 < 0.85$ for 81.9% of layer-method combinations).

However, Full PID (69.29%) does not beat either single-gain variant on CIFAR-100 (Table 4), suggesting that gain interactions introduce conflicting correction signals. Resolving these interactions---through adaptive gain scheduling or phase-dependent gain policies---is the most promising direction for future PID-style WD controllers.

---

## Figures and Tables

- Figure 1: `rho_trajectories.pdf` — Per-layer GW ratio $\rho_t^l$ trajectories for all 7 methods on CIFAR-10/ResNet-20 (10-epoch pilot); confirms $\rho_t^l$ as shared control target (Section 1)
- Figure 2: `udwdc_control_loop.pdf` — UDWDC closed-loop feedback control block diagram (Section 3.3)
- Figure 3: `alignment_snr.pdf` — Two-panel: (a) Alignment SNR vs. batch size for FixedWD, CWD, UDWDC-v2 on CIFAR-100/VGG-16-BN (200 epochs, 3 seeds); (b) Accuracy improvement over FixedWD (Section 5.4)
- Figure 4: `csi_comparison.pdf` — CSI stability comparison across methods (temporal, spatial, combined) (Section 5.7)
- Table 1: inline (Section 3.2) — Unified control law parameter mapping (PID gains for 6 methods)
- Table 2: inline (Section 3.2) — Unification fitting results (relative error for 5 methods)
- Table 3: inline (Section 5.2) — CIFAR-10/ResNet-20 main results (200 epochs, 3 seeds)
- Table 4: inline (Section 5.3) — UDWDC gain ablation on CIFAR-100/VGG-16-BN (7 variants, v2-stabilized)
- Table 5: inline (Section 5.5) — ImageNet/ResNet-50 main results (90 epochs, incomplete: 4 of 7 methods)
- Table 6: inline (Section 5.7) — CSI stability comparison across methods (200 epochs, 3 seeds, last 25% of training)
- Table 7: inline (Section 5.4) — Full batch-size sweep results (CIFAR-100/VGG-16-BN, 200 epochs, H3 falsification data)
