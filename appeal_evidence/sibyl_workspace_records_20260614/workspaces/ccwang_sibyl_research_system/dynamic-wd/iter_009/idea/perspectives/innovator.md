# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Sun et al., CVPR 2025. "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD."** — The foundational paper for this project: first theoretical proof that WD improves generalization in nonconvex SGD via alignment quantity δ_T < 1. Proves WD does NOT accelerate convergence. This paper's worst-case δ framing is the specific gap our unified framework targets.

2. **Chen et al., ICLR 2026 / arXiv:2510.12402. "Cautious Weight Decay (CWD)."** — Sign-alignment-based binary masking of WD. Key competitor: proves asymptotic stability via Lyapunov/LaSalle invariance. Uses discrete (binary) alignment only. Gap: no continuous modulation, no cumulative theory, no connection to norm-matched WD.

3. **Loshchilov, arXiv:2311.11446. "Weight Norm Control (AdamWN)."** — Generalizes decoupled WD to target-norm control (τ=0 recovers standard WD). Provides the mathematical anchor for norm-matched sub-approach. Gap: fixed target norm, no alignment sensitivity.

4. **D'Angelo et al., NeurIPS 2024 / arXiv:2310.04415. "Why Do We Need Weight Decay in Modern Deep Learning?"** — Unifies WD as a training dynamics modifier (not regularizer). Loss stabilization mechanism for SGD. Critical conceptual grounding.

5. **Defazio, arXiv:2506.02285. "Why Gradients Rapidly Increase Near the End of Training."** — Reveals that WD drives gradient-to-weight ratio ‖g‖/‖w‖ to a steady state ("layer balancing"), explaining Adam vs AdamW gap. Unused insight: this ratio is a candidate unifying signal for a feedback-control formulation of WD.

6. **Chen, Yuan, Zhang, arXiv:2602.05136. "Decoupled Orthogonal Dynamics (AdamO)."** — Identifies "Radial Tug-of-War" between WD and gradient. Separates radial (norm) and tangential (direction) dynamics. Closest existing work to a decomposed understanding; gap is formalization into a unified framework.

7. **Newhouse, MIT MEng Thesis, 2025. "Duality, Weight Decay, and Metrized Deep Learning."** — Connects SGD, Adam, Shampoo via duality maps under different norms; WD equilibrium occurs at 1/λ norm bound. This duality lens offers an algebraic approach to unification.

8. **Yunis et al., arXiv:2408.11804. "Approaching Deep Learning through the Spectral Dynamics of Weights."** — WD promotes rank minimization via spectral dynamics; distinguishes generalizing from memorizing networks. Gap: no adaptive WD using spectral rank as feedback signal.

9. **Xie et al., arXiv:2011.11152 / NeurIPS 2023. "Scheduled Weight Decay (SWD)."** — First practical WD scheduler using gradient-norm as a signal. Gap: one scheduling heuristic, no theoretical optimality, no alignment awareness.

10. **Kuzborskij & Abbasi-Yadkori, arXiv:2502.17340. "Low-rank Bias, Weight Decay, and Model Merging."** — WD at stationary points induces parameter-gradient alignment, norm preservation, low-rank bias. Gap: static analysis at stationary points, not trajectory-level.

11. **Aarts et al., 2025. "From SGD to Spectra: A Theory of Neural Network Weight Dynamics."** (arXiv:2507.12709) — SGD noise causes singular values to behave like interacting particles (Dyson Brownian motion with eigenvalue repulsion). This mean-field particle interpretation of weight spectral dynamics provides a novel physical lens for understanding how WD shapes the energy landscape.

12. **arXiv:2505.22578. "Benignity of Loss Landscape with Weight Decay."** — Studies training dynamics of two-layer ReLU networks under WD; shows WD drives grokking: after near-interpolation, inner weights adjust to stationary point. Directly relevant to understanding how alignment-aware WD interacts with different training phases.

### Landscape Summary

The field of weight decay (WD) has undergone a deep conceptual shift from "explicit regularizer" to "training dynamics modifier." Five distinct sub-traditions have evolved independently: (A) WD scheduling (SWD, ADANA, WSD), (B) alignment-aware WD (CWD, AdamO, SPD), (C) decoupled WD (AdamW, AdamD, Lp-norm WD), (D) norm-matched WD (AdamWN, AlphaDecay, CPR), and (E) implicit structural effects (low-rank bias, neural collapse, spectral dynamics). Each has its own theoretical language, experimental protocol, and evaluation metric — creating a fragmented landscape.

Two powerful but underexplored theoretical lenses exist: (1) Defazio's gradient-to-weight ratio ‖g‖/‖w‖ as a controller signal that WD drives to equilibrium — this is a feedback-control interpretation that unifies all WD sub-approaches as different controllers targeting the same plant, and (2) the spectral/mean-field interacting-particle interpretation (Dyson Brownian motion, arXiv:2507.12709) where WD shapes the effective repulsive potential between singular values. Neither lens has been formalized into a unified theoretical framework or used to derive adaptive WD algorithms.

The core gap is structural: no paper has shown that CWD, AdamWN, SWD, and standard decoupled WD are special cases of a single optimization principle. This prevents practitioners from knowing when to use which method, and prevents theorists from deriving optimal WD schedules from first principles.

---

## Phase 2: Initial Candidates

### Candidate A: Lyapunov-Optimal Weight Decay — Deriving Dynamic WD Schedules from Stability Certificates

**Hypothesis**: For nonconvex SGD/AdamW, there exists an optimal time-varying weight decay schedule λ_t* that is uniquely determined by the requirement that a Lyapunov function V(w_t) = f(w_t) + β‖w_t‖² decreases at the maximum certified rate, and this optimal λ_t* recovers known schedules (SWD, WSD, cosine decay) as special cases under different trajectory assumptions.

**Cross-domain insight**: In adaptive control theory (σ-modification), the decay term λw is introduced to maintain Lyapunov stability of error dynamics. The optimal σ is derived from the requirement that the Lyapunov derivative remains strictly negative. The same principle applied to neural network training yields a provably stability-optimal WD schedule that is derived from the training trajectory rather than chosen heuristically.

**Evidence for**: (1) CWD proves asymptotic stability via Lyapunov/LaSalle invariance (ICLR 2026) — the machinery exists for WD + Lyapunov; (2) JAIR 2025 shows exponential decay scheduling derived from Lyapunov stability guarantees convergence with explicit rates; (3) Sun et al. CVPR 2025 use an augmented Lyapunov-like potential Φ_t = f(w_t) + β‖w_t‖² for the convergence analysis of SGDW; (4) the σ-modification in adaptive control is mathematically equivalent to fixed WD — extending it to optimal σ(t) via Lyapunov derivative maximization is a direct analogue.

**Novelty estimate**: 8/10 — The derivation of λ_t* from Lyapunov derivative maximization has not been done for neural network training. Existing work uses Lyapunov functions to prove convergence of fixed or binary-threshold WD, but not to derive optimal schedules. The unification of SWD/WSD/cosine as special cases would be a strong theoretical contribution.

---

### Candidate B: Gradient-to-Weight Ratio as Universal WD Controller — A Feedback Control Unification

**Hypothesis**: All major WD methods (fixed WD, SWD, CWD, AdamWN, norm-matched WD) can be unified as different implementations of a feedback controller targeting the gradient-to-weight ratio R_t = ‖g_t‖/‖w_t‖, where the target setpoint R* determines the training regime (generalization vs. convergence speed), and the Coupling Stability Index (CSI) measures controller quality as the variance of R_t around R* across layers and training phases.

**Cross-domain insight**: In classical control theory, different controllers (P, PI, PID, sliding-mode, model-predictive) all target the same setpoint but with different tradeoffs (overshoot, settling time, robustness to noise). The discovery by Defazio (arXiv:2506.02285) that WD drives ‖g‖/‖w‖ to a steady state across all normalized layers is the "plant model" for this controller analogy. Alignment-aware WD (CWD, AdamO) is a proportional-derivative controller (reacts to instantaneous misalignment), norm-matched WD (AdamWN) is a setpoint controller, and WD scheduling (SWD) is a gain-scheduled P controller.

**Evidence for**: (1) Defazio 2025 explicitly shows ‖g‖/‖w‖ → steady state under WD, calling WD a "layer balancing" mechanism; (2) CWD proves asymptotic stability, corresponding to setpoint regulation with sign-alignment as the error signal; (3) PIDAO (Nature Communications, 2024) shows PID control of optimizer dynamics is effective and provides convergence guarantees; (4) the σ-modification in adaptive control for nonlinear systems is mathematically identical to WD, and control theory has an extensive literature on optimal σ(t) design.

**Novelty estimate**: 9/10 — No existing paper casts all WD methods as controllers of a common plant (the ‖g‖/‖w‖ ratio). The CSI metric as controller quality measure follows directly from this framing. This is a genuine conceptual unification with immediate prescriptive value: optimal WD = optimal controller design for a dynamical plant that can be characterized from training data.

---

### Candidate C: Spectral Rank as Decay Signal — Closing the Loop Between WD and Low-Rank Bias

**Hypothesis**: Training neural networks with WD drives singular value distributions toward low-rank structure (Yunis et al., Kuzborskij et al.), and this rank evolution can be used as a feedback signal to modulate WD strength dynamically: increase WD to accelerate rank compression when rank is far from target, decrease WD when desired structural rank is approached, creating a rank-converging WD schedule that achieves better generalization with less total regularization than fixed WD.

**Cross-domain insight**: In compressed sensing and sparse signal recovery, iterative algorithms like ISTA/FISTA adapt the regularization weight based on the "sparsity gap" between the current solution and the target sparsity level. The analogy: singular value count (effective rank) in neural network weights plays the role of "sparsity," and rank-aware WD scheduling is the deep-learning equivalent of adaptive compressed sensing regularization.

**Evidence for**: (1) Yunis et al. arXiv:2408.11804 show WD drives rank minimization; stronger WD → faster rank compression; (2) arXiv:2410.02176 proves that WD induces approximately rank-2 weight matrices in two-layer ReLU nets, showing theoretical rank-WD coupling; (3) AlphaDecay uses spectral density (HT-SR theory) as module-wise WD signal, showing spectral feedback is practically feasible; (4) arXiv:2512.22192 "Frequency Regularization" shows L2 regularization suppresses high-frequency energy by >3x, quantifiable via SSR metric — a spectral feedback signal for adaptive scheduling.

**Novelty estimate**: 7/10 — AlphaDecay uses spectral density for module-wise WD, which is related. However, AlphaDecay uses static HT-SR analysis, not per-iteration rank tracking as a dynamic feedback signal. The compressed sensing analogy and the derivation of a rank-target-driven WD schedule has not been done. Lower novelty than A or B because AlphaDecay occupies part of this space.

---

## Phase 3: Self-Critique

### Against Candidate A: Lyapunov-Optimal Weight Decay

**Prior work attack**: Searched specifically for "Lyapunov optimal weight decay schedule neural network" and "stability-optimal regularization schedule nonconvex." CWD (ICLR 2026) already analyzes WD via Lyapunov/LaSalle invariance, but only to prove stability of a fixed binary-threshold rule — not to derive an optimal schedule. LyAm (arXiv:2507.11262, July 2025) integrates Lyapunov stability with Adam to enforce monotonically decreasing loss, but does not address WD scheduling. No paper derives λ_t* from Lyapunov derivative maximization. The gap appears genuine.

**Methodological attack**: The Lyapunov derivative approach requires the loss to be approximately bounded and differentiable, which holds for smooth architectures but may fail with BatchNorm/LayerNorm where Lyapunov function construction is non-trivial. Stochastic gradients introduce noise that perturbs the Lyapunov derivative estimate — the analysis may only be valid in expectation, limiting practical utility. Furthermore, "maximum certified rate" may conflict with convergence: a maximally fast Lyapunov decrease may over-decay the weights, catastrophically reducing learning capacity.

**Theoretical attack**: The augmented Lyapunov function Φ_t = f(w_t) + β‖w_t‖² used by Sun et al. treats β as fixed. Making β (or equivalently λ_t) time-varying in a way that maximizes the certified Lyapunov decrease rate may not yield a well-defined optimization problem — if λ_t is chosen too large, Φ_t decreases trivially by shrinking weights to zero, which trivially satisfies Lyapunov decrease but destroys training. A constraint on ‖w_t‖ or a bi-objective formulation is needed.

**Scalability attack**: Computing the exact Lyapunov derivative requires evaluating the gradient on the full training set (population gradient), which is O(N) per step. A stochastic approximation introduces errors that could violate the Lyapunov certificate. This may limit the approach to batch or near-full-batch settings, not practical SGD with mini-batches.

**Verdict**: MODERATE — Fatal flaw is the requirement for population-gradient computation of Lyapunov derivative; this needs to be addressed by using a stochastic proxy (similar to δ̂_t in the project spec). The theoretical core is sound (Lyapunov derivative maximization → optimal schedule) but requires careful qualification of "optimality" and connection to stochastic setting.

---

### Against Candidate B: Gradient-to-Weight Ratio as Universal WD Controller

**Prior work attack**: Searched "gradient-to-weight ratio controller weight decay neural network" and "feedback control interpretation weight decay 2025 2026." Defazio's arXiv:2506.02285 is the source paper for ‖g‖/‖w‖ as steady-state signal, but it only proposes a corrective WD term for LR-schedule interaction, not a general controller framing. PIDAO (Nature Communications 2024) applies PID control to optimizer dynamics but to the loss descent direction, not to ‖g‖/‖w‖ ratio control. No paper frames all WD methods as controllers of the ‖g‖/‖w‖ plant. The conceptual unification appears genuinely novel.

**Methodological attack**: The "plant model" (‖g‖/‖w‖ dynamics) is only well-characterized for normalized layers where Defazio's analysis holds. For unnormalized layers or embedding tables, the ‖g‖/‖w‖ dynamics are more complex and may not admit a simple steady state. The controller analogy may break down for AdamW with very small WD (near zero), where ‖g‖/‖w‖ diverges rather than converging to a setpoint.

**Theoretical attack**: The control theory analogy requires that R_t = ‖g‖/‖w‖ is a well-defined "output" of the "plant" (the training dynamics). But ‖g‖ is a function of ‖w‖ through the loss landscape, creating a feedback loop within the plant itself. The closed-loop stability analysis would require characterizing how the loss landscape curvature changes with ‖w‖, which is essentially as hard as the original convergence problem. The PID/sliding-mode analogy is conceptually useful but may not yield tractable stability proofs.

**Scalability attack**: Computing ‖g‖/‖w‖ per layer requires tracking full layer gradient and weight norms at each step, which adds O(D) overhead where D is layer size. This is affordable (similar overhead to gradient clipping) but the controller design (choosing the target setpoint R* and the gain) requires training-time meta-optimization or offline calibration, which may not scale to large models.

**Verdict**: STRONG — The prior work attack found no competing paper. Methodological issues are manageable (layer normalization scope, affordable overhead). The theoretical attack identifies a real complication (closed-loop stability), but this is addressable by bounding the curvature locally (standard in non-convex optimization proofs). The conceptual contribution of unifying all WD methods as controllers of a common plant is genuinely new and immediately useful.

---

### Against Candidate C: Spectral Rank as Decay Signal

**Prior work attack**: Searched "rank-adaptive weight decay spectral singular value 2025 2026." AlphaDecay (arXiv:2506.14562) uses HT-SR spectral density for per-module WD assignment. While it is static (computed before training), it establishes that spectral feedback for WD is a known direction. More critically, arXiv:2512.22192 ("Frequency Regularization") tracks spectral properties during training with a dynamic SSR metric — this is closer to our proposal. Also, arXiv:2410.02176 proves rank-2 convergence under WD but doesn't use rank as a feedback signal. The core dynamic rank-feedback WD is not yet done, but AlphaDecay occupies adjacent territory.

**Methodological attack**: Computing the effective rank of weight matrices at each training step requires SVD, which costs O(min(m,n)·max(m,n)²) per layer — prohibitively expensive for large models (ViT, ResNet-50). Even approximate rank (stable rank or nuclear norm proxy) requires O(mn) computation. The feedback signal is too expensive to compute frequently enough to be useful for dynamic WD scheduling.

**Theoretical attack**: The compressed sensing analogy (sparsity gap → rank gap) is superficially appealing but the mapping is imprecise. In compressed sensing, the "target sparsity" is known a priori. In neural networks, the "target rank" is not known and may depend on the task — there is no equivalent to the measurement matrix rank condition that guarantees recovery. The analogy breaks down at the level of optimality guarantees.

**Scalability attack**: For large models (ResNet-50, ViT-B), SVD per layer per iteration is completely infeasible. Even with 10-iteration intervals, SVD of 512x512 matrices across 50 layers at 100ms per SVD = 5s overhead every 10 iterations, which is prohibitive. Approximate rank via power iteration is faster but introduces noise that may destabilize the feedback signal.

**Verdict**: WEAK — The AlphaDecay prior work, combined with the prohibitive computational cost of rank computation, makes this approach less compelling as a standalone idea. The compressed sensing analogy is imprecise. This direction is better incorporated as a sub-component of Candidate B (spectral rank as one input to the controller state).

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (Spectral Rank as Decay Signal)** dropped because: (1) AlphaDecay occupies adjacent space with per-module static spectral WD; (2) per-iteration SVD is computationally prohibitive; (3) the compressed sensing analogy lacks the theoretical precision needed for convergence guarantees; (4) the core insight (spectral properties as WD signal) is better incorporated as a diagnostic visualization component in the unified framework rather than as a primary algorithm.

### Strengthened Ideas

**Candidate A (Lyapunov-Optimal WD)**: Fatal flaw was population-gradient requirement. Fix: replace the Lyapunov derivative computation with a stochastic proxy using mini-batch statistics. Specifically, use the EMA of the mini-batch loss decrease as a proxy for the Lyapunov derivative (this is analogous to the δ̂_t proxy for alignment in the project spec). The "maximum certified rate" constraint is replaced with a "minimum guaranteed decrease" constraint: λ_t* = argmax_λ {λ : E[ΔΦ_t | λ] ≤ -α‖∇f‖²}, where α is a user-specified descent rate guarantee. This is well-defined in expectation and connects to the project's alignment quantity δ_t: when δ_t is small (gradient and weight are aligned), the Lyapunov function decreases faster per unit of WD, justifying larger λ_t.

Additional evidence found during refinement: LyAm (arXiv:2507.11262, July 2025) applies Lyapunov stability to Adam and achieves convergence in nonconvex settings by enforcing monotonic loss decrease. The paper does not address WD scheduling, confirming the gap. This also shows the practical viability of stochastic Lyapunov methods.

**Candidate B (‖g‖/‖w‖ Controller)**: Theoretical attack identified closed-loop stability as a gap. Fix: restrict the controller design to the class of "proportional decay controllers" λ_t = k_P · (R* - R̂_t)_+ where R̂_t is the running estimate of ‖g_t‖/‖w_t‖ and R* is the target ratio. Show that for this P-controller, the closed-loop system is stable when the gain k_P satisfies k_P < 2/(L_R · η) where L_R is the Lipschitz constant of the ratio dynamics (bounded by standard smoothness assumptions) and η is the learning rate. This reduces the open-ended "optimal controller design" to a tractable P-controller stability analysis.

Additional evidence: The PIDAO paper (Nature Communications, 2024) shows PID optimizers are theoretically stable in nonconvex settings. The σ-modification in adaptive control (which is equivalent to WD) has an extensive literature on gain bounds for stability (the "σ-modification gain bound" is a standard result in adaptive control). The Defazio 2025 analysis implicitly characterizes the gain-delay tradeoff in the ‖g‖/‖w‖ steady state.

### Additional Evidence Found

- **Benignity of Loss Landscape with Weight Decay** (arXiv:2505.22578, 2025): WD drives neural networks through different training phases, with a grokking transition. This confirms that WD has qualitatively different effects in different training phases — supporting the phase-sensitive WD scheduling that Candidate A and B both motivate.
- **LyAm** (arXiv:2507.11262, July 2025): Lyapunov-inspired optimizer that integrates Adam with stability constraints achieves CIFAR-10/100 SOTA. Confirms that Lyapunov-based training constraints are practically effective on the same benchmarks we use.
- **PIDAO** (Nature Communications, 2024): PID feedback control of optimizer dynamics achieves convergence guarantees in nonconvex settings. The direct antecedent to our controller framing.

### Selected Front-Runner

**Candidate B: Gradient-to-Weight Ratio as Universal WD Controller**

Rationale:
1. **Strongest novelty**: No existing paper unifies all WD methods (SWD, CWD, AdamWN, standard decoupled WD) under a single controller framework targeting ‖g‖/‖w‖.
2. **Survives prior work attacks**: Defazio's 2025 paper is the only foundation paper, and it does not propose the controller framing.
3. **Provides all three proposed metrics naturally**: CSI = controller stability measure (variance of R_t around R*), BEM = compute-normalized controller performance, AIS = mutual information between alignment signal and optimal controller gain.
4. **Prescriptive value**: The framework immediately tells practitioners which WD method to use — the one whose controller architecture best matches the dynamical behavior of their training setup.
5. **Testable**: The P-controller λ_t = k_P · (R* - R̂_t)_+ is implementable in <10 lines of PyTorch code, and the theoretical gain bound k_P < 2/(L_R · η) is verifiable empirically on CIFAR-10/100.
6. **Incorporates Candidate A as a special case**: When the Lyapunov derivative proxy equals (R* - R̂_t) (i.e., Lyapunov decrease rate is proportional to ratio deviation), the Lyapunov-optimal schedule IS the P-controller. This demonstrates the frameworks are compatible and mutually reinforcing.

---

## Phase 5: Final Proposal

### Title

**A Feedback Control Theory of Weight Decay: Unifying Dynamic WD Methods via Gradient-to-Weight Ratio Control**

### Hypothesis

All major weight decay methods — fixed WD, scheduled WD (SWD, WSD), alignment-aware WD (CWD, AdamO), and norm-matched WD (AdamWN, CPR) — can be characterized as special-case implementations of a proportional feedback controller targeting the gradient-to-weight ratio R_t = ‖g_t‖/‖w_t‖, with different methods corresponding to different controller architectures (P, PI, PD, sliding-mode, setpoint) of the same dynamical plant. An optimal controller derived from this framework — the Ratio-Adaptive Weight Decay (RAWD) optimizer — achieves better generalization-convergence tradeoff than any single fixed WD strategy, and this is falsifiable by showing: (i) RAWD Pareto-dominates fixed WD, SWD, and CWD on generalization-vs-convergence curves on CIFAR-10/100/ImageNet, and (ii) the CSI metric quantitatively predicts final generalization gap across WD methods.

### Motivation

Defazio (arXiv:2506.02285, 2025) discovered that WD acts as a "layer balancing" mechanism that drives the gradient-to-weight ratio R_t = ‖g_t‖/‖w_t‖ to the same steady state across all normalized layers. This single observation, which has not been connected to the alignment-aware or scheduled WD literature, suggests a unified plant model for all WD methods: all WD methods are targeting the same output variable (R_t), just with different controller architectures and different target setpoints.

The field currently lacks a unifying theoretical language for comparing WD methods. Five research threads (scheduling, alignment-aware, decoupled, norm-matched, structural) use incompatible formulations. Practitioners cannot answer: "When does CWD outperform SWD? When is fixed WD optimal?" The controller framework answers this directly: the best WD method for a given training run is the controller architecture that best matches the plant's dynamical characteristics (bandwidth, noise level, nonlinearity).

This research also addresses the three proposed standardized metrics:
- **Budget Equivalence Metric (BEM)**: Compute-normalized comparison of controller performance (convergence speed at equal FLOPs)
- **Coupling Stability Index (CSI)**: Variance of R_t around R* across layers and training phases — directly measures controller quality
- **Alignment Informativeness Score (AIS)**: Mutual information I(δ̂_t; optimal_λ_t | R̂_t) — quantifies how much the alignment signal beyond R̂_t improves WD decisions

### Method

**Step 1: Plant Characterization.** For a given architecture and dataset, run a 5-epoch pilot to characterize the ‖g_t‖/‖w_t‖ dynamics per layer: measure the natural frequency of R_t oscillations, the noise level σ_R, and the decay time constant τ_R. This characterizes the "plant" that WD controls.

**Step 2: Theoretical Unification.** Derive the exact controller architecture corresponding to each WD method:
- Fixed WD (λ=const): Integral controller (drives R_t toward a fixed setpoint R* = λ/η in steady state per Defazio's analysis)
- SWD (gradient-norm-aware WD): Proportional controller with gain proportional to ‖g_t‖ (error signal: observed R_t minus reference)
- CWD (sign-alignment mask): Sliding-mode controller with binary switching law sign(⟨g_t, w_t⟩)
- AdamWN (target-norm τ): Setpoint controller driving ‖w_t‖ → τ, which implies R_t → ‖g*‖/τ at convergence
- CPR (augmented Lagrangian): PI controller (the Lagrange multiplier integrates the norm violation error)

**Step 3: Optimal Controller Design (RAWD).** Design an optimal P-controller:
```
λ_t = k_P · max(R̂_t - R*, 0) + λ_min
```
where R̂_t is an EMA estimate of ‖g_t‖/‖w_t‖ averaged across layers, R* is the target ratio (chosen from pilot as the ratio at which validation loss begins to decrease stably), and k_P is the controller gain bounded by k_P < 2/(L_R · η_t) (stability condition).

Layer-specific extension:
```
λ_t^{(l)} = k_P · max(R̂_t^{(l)} - R*^{(l)}, 0) + λ_min
```

**Step 4: Empirical Unification Evidence.** Track R_t trajectories for all baseline methods and show they all converge to approximately the same R* value (within a factor of 2), confirming they are all targeting the same plant output.

**Step 5: Standardized Metrics.** Implement and validate:
- CSI = Var(R_t^{(l)} / R*)_t,l (lower = better controlled)
- BEM = (final_test_accuracy - baseline_accuracy) / (total_FLOPs_RAWD / total_FLOPs_baseline)
- AIS = I(δ̂_t; λ_t* | R̂_t) estimated via binned mutual information over training trajectory

### Experimental Plan

**Experiment 1: Plant Characterization (Pilot, 15 min).**
- Architecture: ResNet-20 on CIFAR-10
- Run for 5 epochs with no WD; record R_t^{(l)} = ‖g_t^{(l)}‖/‖w_t^{(l)}‖ per layer
- Measure natural frequency, noise level, time constant of R_t dynamics
- Output: plant model parameters (τ_R, σ_R, ω_R per layer)

**Experiment 2: Baseline Controller Characterization (30 min).**
- Implement fixed WD, SWD, CWD, AdamWN with full R_t tracking
- Show R_t trajectories for all methods on CIFAR-10/ResNet-20 (3 seeds)
- Confirm all converge to similar R* setpoint (within factor of 2)
- Compute CSI for each method
- Falsification criterion: if CSI values do not differentiate methods, the controller quality metric is uninformative

**Experiment 3: RAWD vs. Baselines (45 min per run × 5 settings).**
- Settings: CIFAR-10/ResNet-20, CIFAR-100/VGG-16-BN (3 seeds each)
- Baselines: fixed WD (λ=5e-4), SWD, CWD, AdamWN, no WD
- RAWD variants: P-controller (k_P tuned from pilot), per-layer RAWD
- Metrics: test accuracy (mean±std), BEM, CSI, AIS
- Expected result: RAWD ≥ best baseline on generalization; RAWD CSI < all baselines
- Falsification criterion: if RAWD does not achieve lower CSI and ≥ baseline accuracy, the hypothesis fails

**Experiment 4: AIS Validation (15 min analysis).**
- From Experiment 2/3 trajectories, compute I(δ̂_t; λ_t* | R̂_t) per epoch
- Expected: AIS > 0 in early training, declines in later epochs (alignment signal provides diminishing returns once R_t is stabilized)
- Falsification: if AIS ≈ 0 throughout, the alignment signal provides no information beyond R̂_t

**Experiment 5: ImageNet Extension (4-8 hours, within project spec).**
- ResNet-50 on ImageNet-1K
- RAWD vs. AdamW vs. CWD
- Report top-1 accuracy at 90 epochs (3 seeds)

### Resource Estimate

- Experiments 1-4: ~4 hours total on 1x RTX PRO 6000 (local GPU)
- Experiment 5 (ImageNet): ~6 hours on 2x RTX PRO 6000 (within project constraint)
- Implementation: RAWD optimizer in PyTorch ~200 lines, R_t tracking ~50 lines
- Visualization: R_t trajectory panels, CSI heat maps across layers and epochs
- Total compute: well within project constraint (8x RTX PRO 6000 available)

### Risk Assessment

**Risk 1: R_t does not converge to a well-defined setpoint for non-normalized layers.**
- Probability: Medium (embedding tables, first conv layers may behave differently)
- Mitigation: Apply RAWD only to normalized layers (BN/LN layers), use standard WD for others. If necessary, restrict the claim to "all BatchNorm/LayerNorm-equipped layers."

**Risk 2: k_P stability bound is too conservative, making RAWD equivalent to fixed WD.**
- Probability: Low (the bound k_P < 2/(L_R · η) is unlikely to be tight in practice, as L_R is very loose)
- Mitigation: In experiments, use grid search over k_P ∈ [0.01, 10] × λ_default and show the performance landscape. If the optimum k_P ≈ 0 (equivalent to fixed WD), the hypothesis is falsified and reported honestly.

**Risk 3: CWD (ICLR 2026) is too strong a baseline and RAWD offers no empirical improvement.**
- Probability: Medium (CWD is well-tuned and widely validated)
- Mitigation: Even if RAWD does not improve over CWD numerically, the contribution is the theoretical unification and the CSI/AIS metrics, which are valid independent of whether RAWD is state-of-the-art. The framework value persists even without empirical superiority.

### Novelty Claim

What is exactly new:

1. **Identification of ‖g‖/‖w‖ as the unified plant output**: Defazio (2025) showed WD drives R_t to steady state but did not frame this as a controller design problem and did not connect it to alignment-aware or scheduled WD methods.

2. **Unified controller taxonomy**: The specific mapping of (fixed WD → integral controller), (SWD → P controller), (CWD → sliding-mode), (AdamWN → setpoint controller), (CPR → PI controller) is new. No paper has established these correspondences.

3. **RAWD algorithm**: The specific P-controller λ_t = k_P(R̂_t - R*)_+ with stability-derived gain bound is a new algorithm not covered by any existing WD method.

4. **Three standardized metrics derived from the controller framework**: CSI, BEM, and AIS have clear operational definitions from the controller framing (controller stability, compute-normalized gain, conditional mutual information) that do not require arbitrary choices.

5. **Evidence not yet found**: Searched "gradient-to-weight ratio controller feedback weight decay unified 2025 2026" — no papers frame this unification. The closest is Defazio 2025 (identifies the plant) and PIDAO 2024 (applies PID to optimizer dynamics but not to ‖g‖/‖w‖ control specifically).
