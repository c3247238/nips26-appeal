# 1. Introduction

Weight decay (WD) is the default regularizer in deep learning: a single coefficient $\lambda$ that shrinks every parameter toward zero at each optimization step.  Despite its ubiquity, the research community has splintered into four parallel sub-traditions for *dynamically* controlling $\lambda$, each addressing a different perceived failure mode of fixed WD:

1. **WD scheduling** adjusts $\lambda$ based on gradient norms or training progress.  SWD (Xie et al., NeurIPS 2023) senses $\|\nabla \mathcal{L}\|$ and reduces $\lambda_t$ when gradients are large; ADANA (Ferbach et al., 2026) applies logarithmic-time schedules to $\lambda$ alongside $\beta_1$ and $\beta_2$.

2. **Alignment-aware WD** conditions $\lambda$ on the cosine alignment $\alpha_t = \langle g_t, w_t \rangle / (\|g_t\| \|w_t\|)$ between gradient and weight vectors.  CWD (Chen et al., ICLR 2026) applies a binary mask: decay only when $\alpha_t < 0$.  A one-line code change yields +0.61% on ImageNet ViT-S/16 over standard AdamW.

3. **Decoupled WD** separates the weight-shrinkage term from adaptive gradient scaling.  AdamW (Loshchilov & Hutter, ICLR 2019) remains the dominant instantiation; AdamO (Chen et al., 2026) further decomposes the update into radial (norm) and tangential (direction) components to resolve the "Radial Tug-of-War" between WD and gradient.

4. **Norm-matched WD** drives weight norms toward explicit targets via augmented Lagrangian constraints or spectral analysis.  CPR (Franke et al., NeurIPS 2024) outperforms AdamW across CIFAR-100, ImageNet, and GPT-2 by enforcing per-parameter-matrix upper-bound constraints with only two hyperparameters.

These four sub-communities develop methods in isolation and evaluate under incomparable protocols --- different datasets, different metrics, different hyperparameter search budgets --- making it impossible to determine which design choices actually matter.

Two recent theoretical results suggest the four traditions share a common target.  Defazio (2025) proves that under constant learning rate, WD drives the per-layer gradient-to-weight ratio $\rho_t^l = \|g_t^l\| / \|w_t^l\|$ to a well-defined steady state for all normalized layers.  Wang & Aitchison (ICML 2025) independently show that the optimal WD, recast as an exponential moving average (EMA) timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$, is constant across model and dataset scales.  Sun et al. (CVPR 2025) prove that WD does not accelerate convergence but strictly improves generalization through an alignment-dependent mechanism.

We formalize the connection: **all four dynamic WD sub-traditions are approximations of a single PID-style feedback control law that regulates $\rho_t^l$ toward a prescribed target trajectory $\rho^*(t) = \eta_t / \tau$.**  CWD's alignment mask corresponds to a derivative/alignment correction term ($K_d > 0$).  SWD's gradient-norm sensing maps to proportional control ($K_p > 0$).  CPR's augmented Lagrangian penalty accumulates constraint violations as integral control ($K_i > 0$).  FixedWD is open-loop control with all gains set to zero.

Building on this unified view, we make the following contributions:

1. **Unified feedback control framework.**  We derive a single parametric control law, $\lambda_t^l = \lambda_{\text{base}} + K_p \cdot e_t^l + K_i \cdot \text{EMA}(e_t^l) - K_d \cdot \alpha_t^l \cdot e_t^l$, and map five existing methods to specific gain configurations $(K_p, K_i, K_d)$.  Empirical fitting on CIFAR-10/ResNet-20 trajectories (200 epochs, 3 seeds) achieves 4.7% relative error for CWD and 9.6% for CPR.

2. **UDWDC: a simple proportional controller.**  We propose Unified Dynamic Weight Decay Control (UDWDC), a proportional-only controller ($K_p > 0$, $K_i = K_d = 0$) that closes the $\rho_t$ feedback loop explicitly.  The target $\rho^*(t) = \eta_t / \tau$ is derived from EMA timescale theory, introducing zero new hyperparameters beyond the user's existing $\lambda_{\text{base}}$ and $\eta_0$.

3. **Standardized evaluation metrics.**  We introduce three metrics for fair cross-method comparison: the Budget Equivalence Metric (BEM), which measures accuracy improvement per unit of total WD budget; the Coupling Stability Index (CSI), which quantifies the temporal and spatial stability of $\rho_t$; and the Alignment Informativeness Score (AIS), which measures the Spearman correlation between the alignment signal and optimal WD decisions.

4. **Theoretical analysis.**  We extend the nonconvex SGDW convergence framework of Sun et al. (CVPR 2025) to time-varying WD with alignment modulation.  Theorem 1 proves that alignment-modulated WD achieves tighter generalization bounds per unit WD budget when the alignment variance $\text{Var}_t[\phi(\delta_t)] > 0$.  Proposition 2 establishes that preconditioner-corrected alignment is geometrically consistent for Adam-family optimizers, while Proposition 3 predicts layer-differentiated steady states under alignment-modulated WD with Batch Normalization (BN).

5. **Comprehensive experiments.**  We conduct a controlled 7-way comparison on CIFAR-10 (ResNet-20), CIFAR-100 (VGG-16-BN), and ImageNet (ResNet-50) with 3--5 seeds per configuration.  CPR achieves the highest accuracy on CIFAR-10 (91.74 $\pm$ 0.07%) and ImageNet (74.74 $\pm$ 0.05%), while UDWDC achieves rank-1 BEM on CIFAR-10 (55.87) despite rank-2 accuracy (90.15%).  We report honest negative results: UDWDC v1 exhibits CSI = $-$2.41 (high instability), SWD and DefazioCorrective have fitting errors of 45.8% and 37.6% respectively under the unified control law, and CWD's accuracy advantage is confounded by a 50% WD magnitude reduction.

The paper is organized as follows.  Section 2 reviews the four WD sub-traditions and identifies $\rho_t$ as the shared control variable.  Section 3 formalizes the unified control law, maps existing methods to gain configurations, and presents UDWDC.  Section 4 provides theoretical analysis.  Section 5 defines the standardized evaluation metrics.  Section 6 reports experiments.  Section 7 discusses implications and honest limitations.  Sections 8 and 9 cover future work and conclusions.

<!-- FIGURES
- None
-->
