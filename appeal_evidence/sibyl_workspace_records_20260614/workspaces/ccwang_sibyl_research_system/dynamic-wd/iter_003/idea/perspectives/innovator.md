# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Caraffa, 2026. "Thermodynamically Optimal Regularization under Information-Geometric Constraints." arXiv:2601.17330** -- Proves that Fisher-Rao regularization is the unique thermodynamically optimal regularization, and that Euclidean weight decay is structurally suboptimal. Directly challenges the foundation of standard WD and motivates geometry-aware alternatives.

2. **Sadrtdinov et al., 2025. "Can Training Dynamics of Scale-Invariant Neural Networks Be Explained by the Thermodynamics of an Ideal Gas?" arXiv:2511.07308** -- Maps WD and LR to thermodynamic variables (temperature, pressure, volume) in scale-invariant networks. Establishes WD as controlling "thermodynamic pressure" in parameter space, validated experimentally.

3. **Hwang, 2024. "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information." arXiv:2405.12807** -- Shows Adam approximates natural gradient descent on the Fisher manifold; refines the WD term based on this geometric insight. Demonstrates that WD corrections from Fisher geometry improve performance across LLM/ASR/VQ-VAE.

4. **Newhouse, 2025. "Duality, Weight Decay, and Metrized Deep Learning." MIT Thesis.** -- Introduces modular duality to unify WD and spectral regularization; connects WD to mirror descent and natural gradient frameworks. The most direct attempt at a geometric unification of WD methods.

5. **Uhlmann, 2026. "Unit-Consistent Adjoint for GSD and Backprop in Deep Learning." arXiv:2601.10873** -- Formulates gauge-consistent optimization: networks with positional-homogeneous activations (ReLU) have gauge symmetries, and standard gradient descent (including WD) is not equivariant. Proposes UC adjoint that fixes this.

6. **Kosson et al., 2023. "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks." arXiv:2305.17212** -- WD induces rotational equilibrium: balanced average rotation speed across layers/neurons. This angular dynamics perspective connects directly to our alignment-aware WD analysis.

7. **Zhou et al., 2025. "Dynamic Weight Adaptation in SNNs Inspired by Biological Homeostasis." arXiv:2511.17563** -- Applies BCM homeostasis theory to dynamic weight control. The core principle -- maintaining neural activity within a target range via adaptive weight modulation -- is structurally analogous to norm-matched WD.

8. **Baveja et al., 2025. "A Unified Noise-Curvature View of Loss of Trainability." arXiv:2509.19698** -- Proposes per-layer adaptive step-size control based on noise-curvature bounds. The insight that curvature volatility determines effective regularization strength is directly applicable to curvature-aware WD.

9. **Caraffa, 2026. "Dissipative Learning: A Framework for Viable Adaptive Systems." arXiv:2601.17933** -- Extends thermodynamic optimal regularization to continual learning. Introduces "crystallization index" as a diagnostic. The Conditional Optimality Theorem (Fisher-Rao = unique optimal geometry) is the strongest theoretical foundation for geometry-aware WD.

10. **Bergsma et al., 2025. "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training." arXiv:2505.13738** -- Verifies that optimal WD timescale follows precise power laws in D/N. Establishes that the EMA timescale interpretation holds at scale and that optimal tau is not constant but obeys scaling laws.

11. **Xu et al., 2026. "FISMO: Fisher-Structured Momentum-Orthogonalized Optimizer." arXiv:2601.21750** -- Connects Muon's orthogonalized direction to Fisher information geometry. Directly relevant to understanding how WD should interact with non-Euclidean optimizers (Gap 5 from literature survey).

12. **Yildirim, 2026. "The Geometric Inductive Bias of Grokking." arXiv:2603.05228** -- Shows that enforcing spherical topology (bounded weight norms) eliminates grokking delay by 20x without WD. Provides interventional evidence that WD's norm-control effect is central to generalization dynamics.

### Landscape Summary

The current weight decay landscape is undergoing a fundamental conceptual shift. Three independent theoretical developments are converging:

**First**, the thermodynamic perspective (Caraffa 2026; Sadrtdinov et al. 2025) reveals that standard Euclidean weight decay is provably suboptimal from a fundamental energy-efficiency standpoint. Fisher-Rao regularization -- which measures distance in the space of distributions rather than in Euclidean parameter space -- is the unique thermodynamically optimal choice. This suggests that all existing WD methods are operating in a structurally suboptimal regime.

**Second**, the geometric/gauge perspective (Uhlmann 2026; Newhouse 2025; Hashimoto et al. 2024) shows that neural networks possess fundamental symmetries (scale invariance in BN layers, gauge symmetries in ReLU networks) that standard WD does not respect. This creates optimization artifacts: the same network function can have wildly different WD penalties depending on the arbitrary parameterization, which is neither principled nor optimal.

**Third**, the biological homeostasis perspective (Zhou et al. 2025; Massey et al. 2026; Laborieux & Zenke 2023) reveals that biological neural systems solve the norm-control problem through dynamic, curvature-sensitive homeostatic feedback -- not through static penalty terms. The BCM theory and Jacobian homeostasis both involve monitoring local activity statistics and adjusting synaptic weights to maintain target operating ranges, which is structurally identical to what norm-matched WD attempts to do but with far richer adaptation mechanisms.

The critical gap remains: **no existing work connects these three perspectives into a single actionable framework for WD**. The thermodynamic theory says "use Fisher-Rao geometry," the gauge theory says "respect parametrization invariance," and the homeostasis theory says "use closed-loop feedback on local statistics" -- but nobody has shown how to combine these insights into a practical, unified dynamic WD algorithm.

## Phase 2: Initial Candidates

### Candidate A: Thermodynamic Weight Decay -- Fisher-Rao-Optimal Dynamic Regularization

- **Hypothesis**: Replacing Euclidean WD with a tractable approximation of Fisher-Rao regularization -- using diagonal empirical Fisher information to reweight per-parameter decay -- will systematically improve both convergence stability and generalization, with the improvement magnitude scaling with the curvature heterogeneity of the loss landscape.

- **Cross-domain insight**: From **thermodynamics of computation** (Landauer principle + information geometry). Just as thermodynamically optimal erasure of information requires geodesic paths on the Fisher-Rao manifold, thermodynamically optimal regularization requires penalizing Fisher-Rao distance rather than Euclidean distance. The "thermodynamic efficiency" of learning (Caraffa 2026) provides a principled optimality criterion that no existing WD method satisfies.

- **Evidence for**: (1) Caraffa 2026 proves Euclidean WD is structurally suboptimal. (2) FAdam (Hwang 2024) shows Adam already approximates natural gradient on the Fisher manifold, and correcting the WD term accordingly improves performance. (3) Sadrtdinov et al. 2025 map WD to thermodynamic pressure, providing the P-V-T analogy that makes WD scheduling equivalent to controlling thermodynamic state variables along quasi-static paths.

- **Novelty estimate**: 7/10 -- The theoretical motivation (Fisher-Rao optimality of WD) is genuinely new. FAdam touches on correcting WD via Fisher info but does not frame it as thermodynamic optimization or derive the full implications for scheduling, alignment, and norm-matching. No existing paper proposes using diagonal Fisher information to modulate per-parameter WD dynamically with a thermodynamic efficiency metric.

### Candidate B: Homeostatic Weight Decay -- Closed-Loop Feedback Control of Weight Norms via Local Activity Statistics

- **Hypothesis**: A biologically-inspired closed-loop WD controller, which monitors per-layer weight norm velocity (rate of change) and curvature (second derivative) and adjusts decay strength via a PID-like feedback law targeting a norm trajectory derived from the rotational equilibrium condition, will outperform all open-loop WD schedules (including SWD, ADANA, cosine WD) by maintaining stable training dynamics without requiring schedule tuning.

- **Cross-domain insight**: From **biological homeostasis** (BCM theory + synaptic scaling). Biological neural systems maintain stable firing rates not through a predetermined schedule but through continuous feedback: if a neuron fires too much, homeostatic mechanisms reduce synaptic strength; if too little, they increase it. The "setpoint" adapts based on local activity history. Transferring this to WD: instead of scheduling lambda(t) based on a fixed recipe, use a feedback controller that monitors the "activity" (weight norm, gradient-weight alignment, effective learning rate) and adjusts WD to maintain a target operating regime.

- **Evidence for**: (1) Zhou et al. 2025 show homeostatic weight adaptation improves robustness in SNNs. (2) Kosson et al. 2023 show WD establishes rotational equilibrium -- this equilibrium condition provides the natural "setpoint" for the controller. (3) Fernandez-Hernandez et al. 2025 (OUI) show that overfitting/underfitting indicators can be computed during training without validation -- these can serve as feedback signals. (4) Baveja et al. 2025 show per-layer noise-curvature bounds naturally lead to adaptive scheduling that mirrors hand-tuned schedules.

- **Novelty estimate**: 8/10 -- No existing WD method uses closed-loop feedback control with explicit PID-style dynamics. SWD uses gradient norms as a proxy but it is an open-loop heuristic, not a closed-loop controller with stability guarantees. The BCM-inspired setpoint adaptation is entirely novel in the context of WD.

### Candidate C: Gauge-Equivariant Weight Decay -- Parametrization-Invariant Regularization via Quotient Geometry

- **Hypothesis**: Standard WD is not equivariant under the rescaling symmetries of networks with normalization layers (BN/LN/GN), causing it to penalize equivalent parameterizations differently. A gauge-equivariant WD that operates on the quotient manifold (equivalence classes of parameters under the gauge group) will eliminate this artifact and produce more meaningful regularization, measurably improving the Coupling Stability Index.

- **Cross-domain insight**: From **gauge theory in physics** (gauge symmetries + principal bundles). In gauge field theories, observables must be gauge-invariant -- they cannot depend on the arbitrary choice of local frame. Neural networks with positional homogeneity (ReLU + BN) possess analogous gauge symmetries (rescaling). Standard WD violates gauge invariance because ||w||^2 changes under rescaling even though the network function does not. The correct approach is to define WD on the quotient space of gauge orbits.

- **Evidence for**: (1) Uhlmann 2026 proves standard gradient descent violates gauge equivariance in homogeneous networks and proposes a UC adjoint fix. (2) Kosson et al. 2023 show WD's rotational equilibrium emerges precisely because WD breaks the scale symmetry, forcing angular dynamics. (3) Yildirim 2026 shows enforcing spherical constraints (which quotient out the radial gauge direction) eliminates grokking. (4) Hashimoto et al. 2024 formalize the gauge symmetries of neural network architectures.

- **Novelty estimate**: 9/10 -- No existing work explicitly formulates WD as a gauge-equivariant operation on the quotient manifold. Uhlmann 2026 addresses gauge-consistent gradients but not WD specifically. The connection between WD and gauge symmetry breaking is implicit in the rotational equilibrium literature but has never been made explicit or exploited for algorithm design.

## Phase 3: Self-Critique

### Against Candidate A (Thermodynamic WD)

- **Prior work attack**: FAdam (Hwang 2024) already corrects WD using diagonal Fisher information. Newhouse's thesis (2025) proposes metrized WD. Caraffa's paper (2026) provides the theory but does not implement or validate it for deep learning. The gap between "Fisher-Rao is optimal" and "here is a practical algorithm" may be smaller than it appears -- someone may already be working on this. *Search result*: No direct prior work found that combines thermodynamic optimality criterion with practical per-parameter WD. FAdam's WD correction is ad hoc, not derived from thermodynamic principles.

- **Methodological attack**: Computing the diagonal Fisher information for every parameter at every step is expensive (requires sampling). The approximation quality (diagonal vs. full Fisher) may dominate any theoretical benefit. For SGD-style optimizers that do not maintain second-moment estimates, this requires new computational infrastructure.

- **Theoretical attack**: Caraffa's theorem assumes quasi-static processes and maximum-entropy belief states -- both far from reality in practical training. The gap between the theoretical optimum and practical approximation may be too large to demonstrate meaningful improvements. The thermodynamic analogy may be more pedagogically useful than algorithmically actionable.

- **Scalability attack**: Full Fisher is intractable at scale. Diagonal Fisher is available in Adam's v_t but not in SGD. The benefit may vanish for well-tuned existing methods (AdamW with good scheduling already implicitly approximates some curvature-aware regularization).

- **Verdict**: MODERATE -- The theoretical motivation is sound but the gap to practice is significant. The main risk is that diagonal Fisher approximation is too crude to capture the true geometry, making the practical algorithm only marginally better than well-tuned AdamW. However, the theoretical framing itself (thermodynamic efficiency of WD) is a genuine contribution even if the practical gains are modest.

### Against Candidate B (Homeostatic WD)

- **Prior work attack**: SWD (Xie et al. 2023) already adapts WD based on gradient norms. OUI (Fernandez-Hernandez et al. 2025) provides training-time diagnostics. Baveja et al. 2025 derive per-layer adaptive schedulers from noise-curvature bounds. The difference between "adaptive WD based on local statistics" and "homeostatic feedback controller" may be seen as merely rhetorical. *Search result*: No paper explicitly uses PID control for WD. The closest is SWD's gradient-norm-based adaptation, but it lacks the setpoint, integral, and derivative terms that characterize a proper controller.

- **Methodological attack**: PID controllers require tuning three hyperparameters (Kp, Ki, Kd) per layer, potentially replacing one hyperparameter (lambda) with dozens. The stability of the feedback loop is not guaranteed in the stochastic optimization setting. Oscillations could cause training instability worse than any fixed schedule.

- **Theoretical attack**: The BCM analogy may be superficial. BCM operates on firing rates (scalar, non-negative, bounded), while WD operates on weight magnitudes in high-dimensional unbounded space. The "setpoint" concept assumes a well-defined target operating regime, but the optimal weight norm changes throughout training (it is not homeostatic in the static sense). The sliding-mode interpretation of CWD (Chen et al. 2025) already captures some control-theoretic dynamics.

- **Scalability attack**: Per-layer monitoring and control is tractable, but the integral and derivative terms require storing norm histories, adding memory overhead. The convergence analysis of PID-controlled SGD does not exist in the optimization literature.

- **Verdict**: MODERATE -- The idea is creative and the control-theoretic framing adds genuine insight, but the practical implementation risks (hyperparameter explosion, stability) are serious. The key insight (closed-loop > open-loop for WD) is valuable regardless of the specific controller design. Needs simplification: perhaps a proportional-only (P) controller with an adaptive setpoint, avoiding the complexity of full PID.

### Against Candidate C (Gauge-Equivariant WD)

- **Prior work attack**: Weight-space symmetries in BN networks are well-known (WD breaks rescaling symmetry, which is how it controls the effective learning rate). Kosson et al. 2023 already analyze this through rotational equilibrium. The WD-as-symmetry-breaking perspective is implicit in multiple papers. *Search result*: No paper explicitly constructs a gauge-equivariant WD on the quotient manifold. Uhlmann 2026 addresses gauge-equivariant gradients but not WD. The novelty gap is real.

- **Methodological attack**: Computing projections onto the quotient manifold requires understanding the gauge group structure, which varies across architectures. For networks without normalization layers, there is no gauge symmetry, so the method does not apply. The quotient geometry may be complex (e.g., Grassmannian for BN layers) and computing geodesics or projections may be intractable.

- **Theoretical attack**: WD's explicit breaking of scale symmetry is *the mechanism* by which it controls effective learning rate in BN networks (Kosson et al. 2023). Making WD gauge-equivariant might actually remove the beneficial symmetry-breaking effect, leaving WD with no regularization effect at all in scale-invariant layers. This is a potentially fatal flaw: gauge-equivariant WD on scale-invariant layers may be vacuous.

- **Scalability attack**: The quotient manifold approach requires per-layer gauge-aware computations. For mixed architectures (some layers with BN, some without), the gauge group is heterogeneous, complicating the implementation.

- **Verdict**: WEAK -- The theoretical attack reveals a potentially fatal flaw. In scale-invariant networks, WD works precisely *because* it breaks the gauge symmetry, creating a preference for smaller-norm parameterizations that translates into effectively larger learning rates (the "catapult" effect). Making WD gauge-equivariant would eliminate this mechanism. The idea is intellectually elegant but may be self-defeating for the most important use case (networks with normalization layers). The insight about gauge symmetry is better used as an *analytical tool* (explaining why WD works differently with/without BN) than as an *algorithmic prescription*.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Gauge-Equivariant WD)** dropped because the theoretical attack reveals that gauge-equivariant WD may be vacuous for scale-invariant layers. WD's beneficial effect relies on breaking scale symmetry, so enforcing gauge equivariance could remove the regularization entirely. However, the gauge-theoretic *analysis* of WD is valuable and should inform the unified framework (explaining why different WD methods behave differently with/without normalization layers).

### Strengthened Ideas

**Candidate A (Thermodynamic WD)** strengthened by:
1. **Narrowing the practical gap**: Instead of full Fisher-Rao regularization, propose using Adam's existing second-moment estimate v_t as a curvature proxy to reweight WD per-parameter. This is computationally free for Adam-family optimizers. The formula becomes: lambda_effective(i,t) = lambda * sqrt(v_t(i)) / mean(sqrt(v_t)), which upweights decay on high-curvature parameters and downweights on low-curvature ones. This is equivalent to penalizing approximate Fisher-Rao distance rather than Euclidean distance.
2. **Adding the thermodynamic diagnostic**: Define "WD thermodynamic efficiency" as the ratio of information gained (generalization improvement) to information erased (weight change measured in Fisher-Rao distance). This is a novel diagnostic metric that subsumes the proposed BEM, CSI, and AIS metrics into a single principled quantity.
3. **Connecting to the existing unified framework goal**: Show that all four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) correspond to different approximations of the ideal Fisher-Rao regularization path: scheduling approximates the quasi-static path over time, alignment-aware approximates the direction-dependent curvature, decoupled separates the geometry from the gradient, and norm-matched targets a specific manifold location.

**Candidate B (Homeostatic WD)** strengthened by:
1. **Simplifying the controller**: Replace full PID with a simpler "proportional + adaptive setpoint" controller. The setpoint is derived from the rotational equilibrium condition (Kosson et al. 2023): at equilibrium, the angular velocity is balanced across layers. Monitor deviations from equilibrium and adjust WD proportionally.
2. **Connecting to the thermodynamic framework**: The homeostatic setpoint corresponds to the thermodynamic equilibrium state. The feedback controller drives the system toward this equilibrium along paths that approximate quasi-static (thermodynamically optimal) trajectories.
3. **Providing convergence-friendly formulation**: Frame the controller as a mirror descent step on the weight norm, which has known convergence properties.

### Additional Evidence Found

- **Bergsma et al. 2025** (Power Lines): Validates that the EMA timescale interpretation of WD holds at scale and follows power laws. This supports the thermodynamic analogy: the EMA timescale is analogous to the thermal relaxation time in thermodynamics, and its power-law scaling with D/N mirrors thermodynamic scaling relations.
- **Yildirim 2026** (Geometric Inductive Bias of Grokking): Enforcing spherical topology (bounded norms) eliminates grokking. This shows that norm control (the core function of WD) is more fundamental than any specific WD schedule, supporting the thermodynamic view that the geometry (Fisher-Rao vs. Euclidean) matters more than the schedule.
- **Xu et al. 2026** (FISMO): Connects Muon's orthogonalized updates to Fisher geometry. This validates that the Fisher-geometric perspective extends to non-Euclidean optimizers, addressing Gap 5 from the literature survey.

### Selected Front-Runner

**Candidate A: Thermodynamic Weight Decay (Fisher-Rao-Informed Dynamic Regularization)** is the front-runner because:

1. It provides the deepest theoretical contribution: a principled optimality criterion (thermodynamic efficiency) that explains *why* different WD methods work and predicts *when* they will differ.
2. It subsumes the existing unified framework goal: all four WD sub-approaches are shown to be different approximations of the ideal Fisher-Rao regularization.
3. It is practically implementable with zero computational overhead for Adam-family optimizers (using existing v_t as Fisher proxy).
4. The homeostatic controller idea (Candidate B) naturally falls out as the feedback mechanism that drives the system toward the thermodynamically optimal trajectory, making it a special case rather than a competing idea.
5. The gauge analysis (Candidate C) provides the explanation for why Euclidean WD behaves differently with/without normalization -- a key component of the thermodynamic framework.

## Phase 5: Final Proposal

### Title

Thermodynamic Weight Decay: A Fisher-Rao-Optimal Framework Unifying Dynamic Regularization in Deep Learning

### Hypothesis

Standard Euclidean weight decay is a structurally suboptimal approximation of Fisher-Rao regularization -- the unique thermodynamically optimal form of parameter regularization. By reweighting per-parameter decay using curvature information (empirical Fisher information), we can simultaneously: (1) unify all four major WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) as special-case approximations of Fisher-Rao geodesic regularization, (2) define a principled "thermodynamic efficiency" metric that quantifies how close any WD method is to the theoretical optimum, and (3) propose a practical Thermodynamic Weight Decay (TWD) algorithm that achieves measurably higher thermodynamic efficiency than all existing methods.

This hypothesis is falsifiable: if TWD does not improve thermodynamic efficiency (measured as generalization improvement per unit of Fisher-Rao distance traveled during training) over standard AdamW on at least 3 of 4 benchmark settings (CIFAR-10/ResNet-20, CIFAR-100/VGG-16-BN, ImageNet/ResNet-50, CIFAR-100/ViT), the framework is disconfirmed.

### Motivation

The weight decay landscape is fragmented: four independent research threads (scheduling, alignment-aware, decoupled, norm-matched) each address facets of the same fundamental question -- "how should WD interact with the training trajectory?" -- but from incompatible mathematical formulations. Meanwhile, recent work in thermodynamics of learning (Caraffa 2026) and thermodynamic training dynamics (Sadrtdinov et al. 2025) establishes that there exists a unique optimal geometry for regularization: the Fisher-Rao metric. This provides the missing unifying principle.

The key literature gap is: nobody has applied the thermodynamic optimality principle specifically to weight decay, nor used it to derive a practical algorithm or explain why existing methods differ in effectiveness. Our framework fills this gap by showing that each WD sub-approach approximates a different aspect of the ideal Fisher-Rao regularization:

- **WD Scheduling** (SWD, ADANA) approximates the time-dependent geodesic path on the Fisher manifold by adjusting decay strength over training.
- **Alignment-Aware WD** (CWD, AdamO) approximates the direction-dependent curvature of the Fisher metric by conditioning decay on gradient-weight geometry.
- **Decoupled WD** (AdamW) separates the Euclidean penalty from the preconditioned gradient, partially correcting for the mismatch between Euclidean and Fisher geometries in adaptive optimizers.
- **Norm-Matched WD** (AdamWN, AlphaDecay) targets a specific location on the Fisher manifold rather than the origin, corresponding to a non-trivial reference state in the thermodynamic framework.

### Method

**1. Theoretical Framework: Fisher-Rao WD Decomposition**

Define the general dynamic WD update as:
```
w_{t+1} = w_t - eta_t * g_t - lambda_t(w_t, g_t) * w_t
```
where lambda_t(w_t, g_t) is the effective per-parameter decay coefficient.

The Fisher-Rao optimal decay is:
```
lambda_FR(i,t) = lambda_0 * F_ii(t)^{1/2} / E[F_ii(t)^{1/2}]
```
where F_ii(t) is the i-th diagonal element of the empirical Fisher information matrix. This upweights decay on parameters with high Fisher information (important parameters) and downweights on low-information parameters.

Show that existing methods correspond to specific approximations:
- Standard WD: lambda_t(i) = lambda_0 (ignores Fisher geometry entirely)
- CWD: lambda_t(i) = lambda_0 * sign(w_i * g_i > 0) (binary approximation of alignment)
- AdamW: lambda_t(i) = lambda_0 (Euclidean penalty decoupled from preconditioned gradient)
- SWD: lambda_t(i) = lambda_0 * h(||g_t||) (time-averaged curvature proxy)
- AdamWN: lambda_t(i) = lambda_0 * (1 - tau/||w||) (target-norm controller)
- AlphaDecay: lambda_t(i) = lambda_0 * alpha_layer (spectral density proxy for per-layer Fisher)

**2. Thermodynamic Efficiency Metric (TEM)**

Define:
```
TEM(method) = Delta_generalization / D_FR(w_0, w_T)
```
where Delta_generalization is the test error improvement and D_FR is the total Fisher-Rao distance traveled during training. This measures "information gained per information erased" -- the thermodynamic efficiency of the regularization.

Show that TEM subsumes the three proposed standardized metrics:
- Budget Equivalence Metric (BEM) = TEM normalized by compute budget
- Coupling Stability Index (CSI) = variance of TEM across training phases
- Alignment Informativeness Score (AIS) = marginal TEM improvement from using alignment information

**3. Practical Algorithm: Thermodynamic Weight Decay (TWD)**

For Adam-family optimizers, the diagonal Fisher is already available via the second-moment estimate v_t:
```
lambda_TWD(i,t) = lambda_0 * sqrt(v_t(i) + eps) / RMS(sqrt(v_t + eps))
```

This requires zero additional computation -- just a reweighting of the existing WD term.

For SGD, estimate diagonal Fisher via exponential moving average of squared gradients:
```
f_t(i) = beta_f * f_{t-1}(i) + (1 - beta_f) * g_t(i)^2
lambda_TWD(i,t) = lambda_0 * sqrt(f_t(i) + eps) / RMS(sqrt(f_t + eps))
```

**4. Homeostatic Setpoint Controller (Optional Extension)**

Add a feedback loop that monitors the TEM per layer and adjusts the base lambda_0:
```
e_t = TEM_target - TEM_measured(t)
lambda_0(t+1) = lambda_0(t) + Kp * e_t
```
where TEM_target is derived from the rotational equilibrium condition.

**5. Unified Visualization Dashboard**

Implement a systematic diagnostic panel tracking:
- Per-layer Fisher-Rao distance trajectories
- Per-layer thermodynamic efficiency over training
- Weight norm vs. Fisher-weighted norm evolution
- Alignment informativeness score per layer
- Coupling stability index per phase
- Comparison of all WD methods under identical compute budgets

### Experimental Plan

**Phase 1: Metric Validation (CIFAR-10, ResNet-20, ~15 min per run)**
- Compute TEM, BEM, CSI, AIS for 6 methods: Standard WD, CWD, SWD, AdamWN, AlphaDecay, TWD
- 3 seeds (42, 123, 456), report mean +/- std
- Verify that TEM rank-orders methods consistently with generalization performance
- Generate full visualization dashboard

**Phase 2: Framework Validation (CIFAR-100, VGG-16-BN + ResNet-20, ~30 min per run)**
- Compare TWD vs. all baselines on two architectures
- Ablate components: Fisher-reweighting alone, homeostatic controller alone, combined
- Verify the theoretical prediction: TWD's advantage increases with curvature heterogeneity (VGG > ResNet)

**Phase 3: Scale Validation (ImageNet, ResNet-50, ~4-8 hours per run)**
- TWD vs. AdamW vs. CWD vs. SWD on ImageNet
- Verify scaling behavior: does TWD's advantage persist at scale?
- Use 8x RTX PRO 6000 Blackwell for distributed training

**Phase 4: Architecture Generalization (CIFAR-100 + ImageNet, ViT)**
- Test TWD on Vision Transformer (no BN, uses LN instead)
- Verify that TWD works across normalization paradigms
- Compare the Fisher-Rao geometry of BN vs. LN networks

**Baselines**: Standard AdamW, CWD (ICLR 2026), SWD/AdamS (NeurIPS 2023), AdamWN, AlphaDecay (for LLM comparison if time permits), SGD+WD, Adam+L2

**Falsification criteria**: If TWD does not achieve top-2 TEM ranking on at least 3/4 settings, or if TEM does not rank-order methods consistently with generalization, the framework is disconfirmed.

### Resource Estimate

- **Phase 1**: 6 methods x 3 seeds x 15 min = ~4.5 GPU-hours on single RTX PRO 6000
- **Phase 2**: 6 methods x 2 architectures x 3 seeds x 30 min = ~18 GPU-hours
- **Phase 3**: 4 methods x 3 seeds x 6 hours = ~72 GPU-hours (distributed across 8 GPUs = ~9 wall-clock hours)
- **Phase 4**: 4 methods x 3 seeds x 2 hours = ~24 GPU-hours
- **Total**: ~118.5 GPU-hours, ~2-3 days wall-clock time on 8x RTX PRO 6000
- **Visualization**: Minimal additional compute (post-processing logged metrics)

### Risk Assessment

1. **Risk: Diagonal Fisher approximation too crude** -- The diagonal Fisher misses parameter correlations that may be crucial for the geometry. *Mitigation*: Compare diagonal Fisher vs. K-FAC (block-diagonal) Fisher approximation on one benchmark. If K-FAC significantly outperforms diagonal, consider per-layer Fisher approximation as a middle ground.

2. **Risk: Thermodynamic efficiency metric does not correlate with generalization** -- TEM requires computing Fisher-Rao distance, which adds diagnostic overhead. If TEM does not predict generalization ranking better than simple weight-norm-based metrics, the theoretical framework is less actionable. *Mitigation*: Validate TEM correlation on CIFAR-10 first (cheap). If correlation is weak, fall back to the mathematical unification contribution (showing all methods as Fisher-Rao approximations) without claiming the metric is practically superior.

3. **Risk: TWD underperforms well-tuned CWD or SWD** -- CWD and SWD are already carefully designed methods. TWD's Fisher-reweighting may not add enough signal beyond what alignment (CWD) or gradient-norm (SWD) already capture. *Mitigation*: Focus the contribution on the theoretical unification and diagnostic framework rather than on a single new algorithm. The visualization dashboard and standardized metrics are valuable contributions even if TWD is only competitive (not dominant) with existing methods.

### Novelty Claim

The following elements are, to the best of our knowledge, entirely novel:

1. **Applying thermodynamic optimality (Fisher-Rao uniqueness theorem) specifically to weight decay** -- Caraffa (2026) proves the theorem for general regularization but does not apply it to WD or derive a practical WD algorithm. No existing WD paper cites or uses the thermodynamic framework.

2. **Showing that all four WD sub-approaches approximate Fisher-Rao regularization** -- This unification has never been demonstrated. The closest work is Newhouse's thesis (2025), which connects WD to mirror descent but does not derive the Fisher-Rao optimality result or show the four-way unification.

3. **The Thermodynamic Efficiency Metric (TEM)** -- No existing metric measures generalization improvement per Fisher-Rao distance. BEM, CSI, and AIS (proposed in the idea context) are subsumed as special cases, providing a principled foundation for the standardized evaluation framework.

4. **TWD algorithm using Adam's v_t as Fisher proxy for per-parameter WD reweighting** -- While Adam's v_t is known to approximate diagonal Fisher, nobody has used it to reweight the WD term (as opposed to the gradient preconditioner). This is a one-line change to AdamW with zero overhead.

5. **The gauge-theoretic explanation for why WD behaves differently with/without normalization layers** -- While the scale-invariance interaction is known empirically, framing it as gauge symmetry breaking provides a precise mathematical explanation and predicts which WD modifications will be ineffective for BN networks.

6. **Systematic visualization dashboard with thermodynamic diagnostic quantities** -- No existing visualization toolkit tracks per-layer Fisher-Rao distance, thermodynamic efficiency, or the complete set of diagnostic quantities we propose.
