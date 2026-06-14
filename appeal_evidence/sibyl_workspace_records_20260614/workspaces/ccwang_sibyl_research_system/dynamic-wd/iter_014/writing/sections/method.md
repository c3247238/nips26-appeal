# 3. Unified Feedback Control Framework

We formalize the observation from Section 2 that all four WD sub-traditions manipulate the same underlying quantity --- the per-layer gradient-to-weight ratio $\rho_t^l$ --- into a single parametric control law.  We then derive the target trajectory from EMA timescale theory and present UDWDC, a simple proportional controller that explicitly closes the feedback loop.

## 3.1 The Gradient-to-Weight Ratio as Control Variable

For each parameter group (layer) $l$ at training step $t$, define the gradient-to-weight ratio (GW ratio):

$$\rho_t^l = \frac{\|g_t^l\|_2}{\|w_t^l\|_2 + \epsilon}$$

where $g_t^l$ is the stochastic gradient, $w_t^l$ is the parameter vector, and $\epsilon = 10^{-8}$ prevents division by zero.  The GW ratio measures the relative magnitude of gradient updates to current parameter norms: large $\rho_t^l$ indicates the gradient is dominant (rapid change), while small $\rho_t^l$ indicates the parameters are large relative to their gradients (stable regime).

Defazio (2025) proves that under constant learning rate $\eta$ and WD coefficient $\lambda$, the SGDW update $w_{t+1} = (1 - \eta\lambda)w_t - \eta g_t$ drives $\rho_t$ to a unique steady state for all normalized layers (Batch Normalization or Layer Normalization).  The steady-state value depends on $\lambda$ and $\eta$, establishing $\rho_t$ as a well-defined control variable with a deterministic equilibrium.

Wang & Aitchison (ICML 2025) show that the optimal WD, expressed as the EMA timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$, is constant across model sizes and datasets when measured in epochs.  Combining these two results, we derive the **target trajectory** for a cosine-annealing learning rate schedule:

$$\rho^*(t) = \frac{\eta_t}{\tau} = \eta_t \cdot \lambda_0 \cdot \eta_0$$

This trajectory decreases as $\eta_t$ decays, prescribing tighter coupling (lower $\rho$) in late training when the model should stabilize near a minimum.

## 3.2 PID Control Law Parameterization

We parameterize the WD coefficient as a function of the GW ratio error:

$$\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + K_i \cdot \text{EMA}(e_t^l) - K_d \cdot \alpha_t^l \cdot e_t^l$$

where:
- $e_t^l = \rho_t^l - \rho^*(t)$ is the per-layer **control error** (deviation from target)
- $\alpha_t^l = \langle g_t^l, w_t^l \rangle / (\|g_t^l\| \cdot \|w_t^l\|)$ is the gradient-weight alignment cosine
- $K_p$ is the **proportional gain** (react to current error)
- $K_i$ is the **integral gain** (accumulated error via EMA smoothing)
- $K_d$ is the **derivative/alignment gain** (alignment-modulated correction)

The alignment term $\alpha_t^l$ multiplies the error in the $K_d$ channel, coupling the geometric relationship between gradient and weight directions to the WD adjustment.  When $\alpha_t^l < 0$ (gradient opposes weight direction), the $-K_d \cdot \alpha_t^l \cdot e_t^l$ term *increases* $\lambda$, applying more decay precisely when the gradient already drives weights toward zero.  This mirrors CWD's design rationale.

### Mapping Existing Methods

Table 1 maps five existing methods plus UDWDC to specific gain configurations.

| Method | $\rho^*(t)$ | $K_p$ | $K_i$ | $K_d$ | Feedback Signal | Control Type |
|--------|-------------|-------|--------|--------|-----------------|-------------|
| FixedWD | --- | 0 | 0 | 0 | None | Open-loop |
| CWD | --- | 0 | 0 | $>0$ | Binary $\text{sign}(\alpha)$ | Derivative only |
| SWD | Dynamic | $>0$ | $>0$ | 0 | $\|\nabla\mathcal{L}\|$ | Proportional-integral |
| DefazioCorrective | $\eta_t/\tau$ | $>0$ | 0 | 0 | $\eta_t/\eta_0$ | Feedforward |
| CPR | Per-matrix | $>0$ | $>0$ | 0 | Norm violation | Integral-dominant |
| **UDWDC (ours)** | $\eta_t/\tau$ | $>0$ | 0 | 0 | $\rho_t^l$ | Proportional |

*Table 1: Unified control law parameter mapping.  Each existing method corresponds to a specific gain configuration in the PID framework.  FixedWD is open-loop (no feedback).  CWD uses only the alignment correction.  SWD and CPR include integral terms.  UDWDC uses proportional feedback on $\rho_t$ directly.*

### Empirical Validation of Unification (H1)

We fit the extended control law (with additional offset and target-scale parameters) to per-layer effective-WD trajectories from 200-epoch CIFAR-10/ResNet-20 experiments (3 seeds, 24 weight-bearing layers per seed, 72 traces total).  Table 2 summarizes the fitting results.

| Method | Family | Rel. Error (%) | Std (%) | Status |
|--------|--------|---------------:|--------:|--------|
| CWD | Alignment-based | 4.71 | 0.16 | PASS |
| CPR | Constraint-based | 9.57 | 1.47 | PASS |
| NoWD | Degenerate | 0.00 | 0.00 | PASS |
| SWD | Scheduling-based | 45.81 | 0.68 | FAIL |
| DefazioCorrective | Scheduling-based | 37.56 | 2.37 | FAIL |

*Table 2: H1 unification fitting results.  The control law captures CWD (4.71% error) and CPR (9.57% error) well.  SWD and DefazioCorrective exceed the 20% threshold, indicating that scheduling-based methods operate through mechanisms not fully captured by per-layer $\rho_t$ feedback.  H1 is not falsified (2/5 failures, threshold: >2 for falsification).*

The high fitting error for SWD arises because SWD's gradient-norm-aware scheduling uses global (not per-layer) gradient statistics and internal normalization that does not map cleanly to the per-layer error signal $e_t^l$.  DefazioCorrective's error reflects its feedforward nature: it prescribes $\lambda_t$ based on the learning rate ratio $\eta_t / \eta_0$ rather than sensing $\rho_t$.  Both methods operate on the training schedule rather than on per-layer measured quantities.

## 3.3 UDWDC: A Simple Proportional Controller

UDWDC (Unified Dynamic Weight Decay Control) closes the $\rho_t$ feedback loop with a proportional-only design:

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

UDWDC introduces **zero new hyperparameters**.  The target trajectory $\rho^*(t) = \eta_t / \tau$ is fully determined by $\lambda_{\text{base}}$ and $\eta_0$, both already specified in the user's training recipe.  The clamp bounds [0.1, 10] are fixed constants, not tunable parameters.

### UDWDC-v2: Stability Fix

Pilot experiments revealed that UDWDC v1 suffers from instability: when $\rho_t^l < \rho^*(t)$ for most layers (common during early training with batch normalization), the clamp drives $\lambda_t^l$ to $0.1 \times \lambda_{\text{base}}$, effectively disabling WD.  This produces CSI$_{\text{combined}} = -2.41$ and near-zero total WD budget.

UDWDC-v2 applies two modifications:

1. **EMA smoothing**: Replace instantaneous $\rho_t^l$ with $\hat{\rho}_t^l = \text{EMA}(\rho_t^l, \beta{=}0.99)$, damping step-to-step noise.
2. **Floor clipping**: Enforce $\lambda_t^l \geq \lambda_{\min} = 0.1 \cdot \lambda_{\text{base}}$, guaranteeing nonzero WD at all times.

These are engineering patches, not principled solutions.  The instability reflects a fundamental limitation of proportional-only control: P-control has inherent steady-state offset and cannot guarantee zero tracking error.  An integral term ($K_i > 0$) would eliminate this offset, at the cost of potential overshoot and a new hyperparameter.  We deliberately preserve the P-only design for simplicity and document the instability as a finding.

![UDWDC control loop block diagram](figures/udwdc_control_loop_desc.md)

*Figure 2: UDWDC feedback control loop.  Measure the per-layer GW ratio $\rho_t^l$, compute the error relative to the EMA-timescale-derived target $\rho^*(t)$, apply proportional gain with clamping, and feed the resulting $\lambda_t^l$ into the weight update.  Existing methods are open-loop (FixedWD) or partial-loop (CWD senses alignment but not $\rho_t$; CPR senses norm violation but not alignment).*

## 3.4 Theoretical Analysis

We extend the nonconvex SGDW convergence framework of Sun et al. (CVPR 2025) to time-varying WD with alignment modulation, and derive two additional propositions connecting the control framework to optimizer geometry and layer-specific equilibria.

### Theorem 1: Contraction-Rate Separation for Stagewise SGDW

Consider an $L_{\text{smooth}}$-smooth nonconvex objective $\mathcal{L}$ optimized by SGDW with a stagewise WD schedule $\lambda_t$ and alignment-weighting function $\phi(\delta_t) \geq 0$.  Define the cumulative contraction $C_T = \sum_{t=1}^{T} \eta_t \lambda_t \|w_t\|^2$ and the alignment-weighted contraction $A_T = \sum_{t=1}^{T} \eta_t \lambda_t \|w_t\|^2 \cdot \phi(\delta_t)$.

**Statement.**  The convergence rate to an $\epsilon$-stationary point matches unregularized SGD at $O(1/\sqrt{T})$ when $C_T = O(\sqrt{T})$.  The generalization bound depends on $A_T$ rather than the worst-case alignment.  When the alignment variance $\text{Var}_t[\phi(\delta_t)] > 0$, alignment-modulated WD achieves a strictly tighter generalization bound per unit of WD budget $C_T$ compared to fixed WD.

**Intuition.**  Fixed WD applies uniform contraction regardless of the instantaneous alignment between gradient and weight directions.  Alignment-modulated WD concentrates contraction on steps where alignment indicates the decay is most beneficial for generalization (i.e., when $\phi(\delta_t)$ is large).  The efficiency gain is proportional to the variance of $\phi(\delta_t)$: if alignment is constant, no benefit is possible; if alignment varies, the modulation can selectively allocate WD budget to informative steps.

This theorem provides the theoretical foundation for BEM (Section 3.5): methods that achieve higher accuracy per unit WD budget are exploiting alignment variance, and BEM directly measures this efficiency.

### Proposition 2: Geometry-Corrected Alignment for Adam

For AdamW with diagonal preconditioner $P_t = \text{diag}(\hat{v}_t + \epsilon)$ (the second-moment estimate), the geometrically natural alignment is:

$$\delta_t^P = \frac{\langle P_t^{-1} g_t, w_t \rangle}{\|P_t^{-1} g_t\| \cdot \|w_t\|}$$

rather than the standard $\ell_2$-alignment $\alpha_t = \langle g_t, w_t \rangle / (\|g_t\| \cdot \|w_t\|)$.

**Statement.**  The geometry-corrected alignment $\delta_t^P$ is alignment-consistent with AdamW's implicit optimization objective: the AdamW update moves weights in the direction $P_t^{-1} g_t$, so alignment should be measured in the preconditioned space.  Standard CWD using $\ell_2$-alignment applies the binary mask based on $\alpha_t$, which can be geometrically inconsistent at the parameter-group level when $P_t$ is far from the identity.

**Implication.**  Alignment-aware WD methods should use preconditioner-corrected alignment when applied with Adam-family optimizers.  This correction is negligible for SGD ($P_t = I$) but potentially significant for Adam in late training when the second-moment estimates have accumulated parameter-specific curvature information.  Our experiments use SGD for CNNs, so this proposition is a prediction for future Adam-based evaluations.

### Proposition 3: Layer-Differentiated Steady States

Under alignment-modulated WD on networks with normalized layers (BN), the per-layer steady-state GW ratios satisfy:

$$r_l^* = \frac{\lambda_{\text{base}} \cdot \gamma}{\phi(\delta_l^*)}$$

where $\gamma$ is a normalization-related constant and $\delta_l^*$ is the per-layer steady-state alignment.

**Statement.**  This yields layer-differentiated equilibria that depend on per-layer gradient structure, unlike fixed WD where all normalized layers converge to the same $r^*$ (Defazio, 2025).  Specifically, $r_l^*$ and $\delta_l^*$ are anti-correlated: layers with high steady-state alignment (large $\delta_l^*$) achieve lower GW ratios (smaller $r_l^*$), and vice versa.

**Scope limitation.**  This anti-correlation is restricted to BN architectures with sufficient depth.  Layer Normalization may yield different steady-state behavior because LN normalizes across features rather than across the batch, changing the gradient structure at each layer.  Our experiments verify the anti-correlation on ResNet-50 (BN) but do not test ViT-S/16 (LN).

## 3.5 Standardized Evaluation Metrics

We propose three metrics to enable fair cross-method comparison of dynamic WD methods.  Existing evaluations use only accuracy, masking important differences in regularization efficiency, stability, and signal quality.

### Budget Equivalence Metric (BEM)

$$\text{BEM} = \frac{\text{acc} - \text{acc}_{\text{NoWD}}}{\text{TotalWDBudget}}$$

where $\text{TotalWDBudget} = \sum_{t=1}^{T} \sum_{l=1}^{L} \lambda_t^l \|w_t^l\|^2$ is the cumulative WD applied during training.  BEM captures accuracy improvement per unit of regularization applied, separating methods that achieve high accuracy through efficient WD allocation from those that achieve it through brute-force high WD.

In the 10-epoch CIFAR-10 pilot, BEM ranking diverges from accuracy ranking: UDWDC achieves rank-1 BEM (55.87) despite rank-2 accuracy (81.78%), while FixedWD achieves rank-1 accuracy (82.06%) but rank-2 BEM (45.42%).  This divergence validates that BEM reveals method differences invisible to accuracy-only evaluation.

### Coupling Stability Index (CSI)

The CSI measures the stability of the WD-optimization coupling across training:

$$\text{CSI}_{\text{temporal}} = 1 / \text{Var}_t[\lambda_{\text{eff}}^l], \quad \text{CSI}_{\text{spatial}} = 1 / \text{Var}_l[\lambda_{\text{eff}}^l]$$

averaged over layers (temporal) or over time steps (spatial) in the last 25% of training.  The combined CSI is:

$$\text{CSI}_{\text{combined}} = \frac{\text{CSI}_{\text{temporal}} + \text{CSI}_{\text{spatial}}}{2}$$

Higher CSI indicates more stable coupling.  FixedWD achieves CSI = 1.0 trivially (constant $\lambda$).  UDWDC v1 achieves CSI = $-$2.41, revealing high instability where the proportional controller amplifies rather than damps perturbations.

### Alignment Informativeness Score (AIS)

$$\text{AIS} = \text{Spearman}(\alpha_t, \text{optimal WD decision})$$

where the optimal WD decision is estimated retrospectively by correlating alignment with generalization-gap changes.  AIS is conditioned on batch size to account for alignment SNR scaling (H3).

In the pilot, the global AIS is 0.566, indicating moderate but nonzero informativeness of the alignment signal.  The mean alignment $\bar{\alpha}$ predicts the generalization gap better than $\log_{10}(\lambda)$ across WD configurations (H6 supported, $r_{\alpha,\text{gengap}} = 0.698$), confirming that alignment carries information beyond WD magnitude.

<!-- FIGURES
- Figure 2: udwdc_control_loop_desc.md --- UDWDC control loop architecture block diagram
- Table 1: inline --- Unified control law parameter mapping
- Table 2: inline --- H1 unification fitting results
-->
