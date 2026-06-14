# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Defazio, 2025. "Why Gradients Rapidly Increase Near the End of Training." arXiv:2506.02285** — Shows WD controls the gradient-to-weight ratio ||g||/||w|| and drives all normalized layers to the same steady-state ratio ("layer balancing"). This is the closest existing work to a dynamical-systems view of WD, but treats WD as static and does not formalize feedback control.

2. **Mirzabeigi, 2025. "LyAm: Robust Non-Convex Optimization for Stable Learning in Noisy Environments." arXiv:2507.11262** — First optimizer to explicitly incorporate Lyapunov stability conditions into Adam. Proves monotonic decrease of a Lyapunov energy function. However, it targets gradient noise robustness, NOT weight decay scheduling. Does not address WD at all.

3. **Chen et al., 2025/2026. "Cautious Weight Decay (CWD)." arXiv:2510.12402 (ICLR 2026)** — Binary sign-alignment mask for WD. Bilevel Pareto interpretation + sliding-mode control analogy. The sliding-mode interpretation is a key bridge to control theory, but CWD does not exploit continuous control signals.

4. **Chen, Yuan & Zhang, 2026. "Decoupled Orthogonal Dynamics (AdamO)." arXiv:2602.05136** — Identifies "Radial Tug-of-War" between WD and gradient; decouples radial (norm) and tangential (direction) dynamics. Implicitly uses a phase-space decomposition but does not formalize it as a dynamical system with stability analysis.

5. **Kosson et al., 2023. "Rotational Equilibrium." arXiv:2305.17212** — WD induces rotational equilibrium: balanced average rotation across layers. Defines an equilibrium condition but does not derive adaptive WD from stability requirements.

6. **Chen, Li & Liu, 2025. "Muon Optimizes Under Spectral Norm Constraints." arXiv:2506.15054** — Full Lyapunov convergence analysis for Lion-K/Muon family. Shows Lyapunov functions decrease monotonically despite nonmonotonic loss. Proves the viability of Lyapunov analysis for modern optimizers with WD.

7. **Nikitin et al., 2025. "Biologically Inspired Neural Network Layer with Homeostatic Regulation." Nature Scientific Reports** — Demonstrates how homeostatic regulation (firing rate set points + STDP bounded by protein reserve) maintains stability while enabling adaptation. The multi-timescale homeostatic principle directly parallels what adaptive WD should achieve.

8. **Han et al., 2026. "Weight Decay Improves Language Model Plasticity." arXiv:2602.11137** — Larger WD during pretraining produces more plastic (adaptable) models. Connects WD to maintaining network adaptability — a homeostasis-like property.

9. **Truong & Truong, 2026. "Norm-Hierarchy Transitions." arXiv:2603.07323** — WD traverses norm hierarchy from shortcut to structured representations; transition delay is logarithmic in norm ratio. Reveals WD drives phase transitions in representation structure.

10. **NeurIPS 2024. "A PID Controller Approach for Adaptive Gradient Decay." OpenReview** — Uses PID controller to dynamically adjust gradient decay rate for model calibration. Demonstrates viability of feedback-control-based adaptation of regularization strength during training.

11. **An et al., 2018/2024. "PID Controller Approach for Stochastic Optimization." CVPR 2018; "Incremental PID Learning Rate Scheduler." IEEE TNNLS 2024** — PID-based learning rate scheduling uses feedback control to determine LR from training losses. Achieves state-of-the-art with reduced hyperparameter sensitivity. Directly applicable architecture for WD scheduling.

12. **Yunis et al., 2024. "Spectral Dynamics of Weights." arXiv:2408.11804** — WD promotes rank minimization; spectral dynamics distinguish memorizing from generalizing networks. Provides the spectral-structure observable that could serve as a state variable in a dynamical-systems WD controller.

### Landscape Summary

The field of dynamic weight decay is fragmented into four largely independent sub-approaches (scheduling, alignment-aware, decoupled, norm-matched), each with its own theoretical lens and evaluation criteria. The literature survey reveals a critical missing perspective: **no existing work treats the (weight norm, gradient-to-weight ratio, alignment) tuple as a formal dynamical system and derives adaptive WD from stability requirements of that system.**

Three converging trends make this gap especially timely. First, Defazio (2025) showed WD drives a gradient-to-weight ratio equilibrium — implicitly defining a dynamical system with an attractor, but without formalizing the control structure. Second, the Lion-K/Muon line (Chen et al., 2025) proved that Lyapunov analysis yields tight convergence guarantees for modern optimizers with WD, establishing the mathematical machinery. Third, CWD's sliding-mode interpretation (Chen et al., ICLR 2026) and AdamO's radial/tangential decomposition (2026) both hint at control-theoretic structure but stop short of exploiting it for adaptive scheduling.

Meanwhile, cross-domain work in feedback control for optimization (PID-based LR scheduling, Lyapunov-based optimizers like LyAm) and biological homeostatic regulation (firing-rate set points, multi-timescale adaptation) provide mature conceptual frameworks that have NOT been applied to weight decay adaptation. The homeostatic plasticity literature is especially relevant: biological neural networks maintain stability through distributed, multi-timescale regulatory mechanisms that balance excitation and inhibition — precisely the kind of adaptive regulation that weight decay should provide.

## Phase 2: Initial Candidates

### Candidate A: Lyapunov-Guided Adaptive Weight Decay (LyaWD)

- **Hypothesis**: Weight decay strength can be derived as the control input of a Lyapunov-stable dynamical system defined over the (||w||, ||g||/||w||, cos(w,g)) state space, yielding a principled, continuous, alignment-aware WD schedule that provably stabilizes training dynamics and subsumes CWD, SWD, and norm-matched WD as special cases.
- **Cross-domain insight**: From nonlinear control theory — specifically Lyapunov's direct method and feedback linearization. The key transplanted principle is: rather than heuristically scheduling a regularization parameter, derive it as the unique control signal that guarantees a chosen Lyapunov function (encoding desired training properties) decreases monotonically. This is how control engineers design adaptive controllers for nonlinear systems with uncertain dynamics.
- **Evidence for**: (1) Defazio (2025) showed ||g||/||w|| converges to a steady state under fixed WD — this IS an equilibrium of a dynamical system, ripe for stability analysis. (2) LyAm (arXiv:2507.11262) proved Lyapunov analysis works for Adam-style optimizers. (3) Lion-K analysis (arXiv:2506.15054) provided full Lyapunov convergence proofs for optimizers with decoupled WD. (4) PID-based LR scheduling (IEEE TNNLS 2024) demonstrated feedback control successfully adapts optimization hyperparameters.
- **Novelty estimate**: 8/10 — LyAm uses Lyapunov for Adam's update rule but does NOT address WD. CWD uses sliding-mode control analogy but does NOT derive WD from a formal Lyapunov function. AdamO decomposes radial/tangential dynamics but does NOT use Lyapunov stability to derive the radial (WD) control law. The specific combination of Lyapunov stability theory + weight decay derivation + state-space formulation over (||w||, ||g||/||w||, cos(w,g)) appears to be genuinely new.

### Candidate B: Homeostatic Weight Decay via Multi-Timescale Set-Point Regulation

- **Hypothesis**: Modeling weight decay as a homeostatic regulatory mechanism — with per-layer "set points" for weight norms that adapt on a slow timescale while WD strength adapts on a fast timescale — will outperform both fixed and heuristically scheduled WD by maintaining each layer in its optimal operating regime throughout training.
- **Cross-domain insight**: From computational neuroscience — specifically homeostatic synaptic scaling and structural plasticity. Biological neural networks maintain stable function through a hierarchy of regulatory mechanisms operating at different timescales: fast synaptic scaling (seconds-minutes) adjusts synapse strengths to maintain a firing-rate set point, while slow structural plasticity (hours-days) adds/removes synapses when scaling is insufficient. The transplanted principle: WD should operate as a fast regulator maintaining a norm set point, while the set point itself slowly adapts based on long-term training trajectory health.
- **Evidence for**: (1) Loshchilov's AdamWN (2023) showed target-norm control is effective but uses a FIXED target. (2) Han et al. (2026) showed WD maintains "plasticity" — an adaptability property analogous to homeostatic regulation. (3) Nikitin et al. (2025, Nature SR) demonstrated homeostatic regulation + STDP bounded by protein reserve maintains stability in artificial neural networks. (4) The eLife study (2025) showed synaptic scaling + structural plasticity interact to maintain robust firing rates.
- **Novelty estimate**: 7/10 — AlphaDecay does module-wise WD but uses static spectral-density-guided assignment, not dynamic set-point regulation. AdamWN provides target-norm control but with fixed targets. The multi-timescale homeostatic framing with adaptive set points appears new, though the connection to biological homeostasis is more metaphorical than mathematical.

### Candidate C: Phase-Space Diagnostic Framework for Weight Decay

- **Hypothesis**: Plotting training trajectories in a (||w_l||, ||g_l||/||w_l||, delta_l) phase space per layer reveals characteristic signatures — limit cycles, bifurcations, divergent spirals — that diagnose WD pathologies and predict optimal WD strategies, enabling a systematic "phase portrait" classification of WD methods that subsumes existing ad hoc visualizations.
- **Cross-domain insight**: From dynamical systems theory — specifically phase portrait analysis and bifurcation theory. In physics and engineering, plotting state variables against each other (not against time) reveals qualitative behavior (attractors, limit cycles, saddle points) invisible in time-series plots. The transplanted principle: by plotting weight norm vs. gradient-to-weight ratio (with alignment as a third axis), we can classify WD methods by the geometry of their induced trajectories rather than by scalar performance metrics.
- **Evidence for**: (1) Defazio's ||g||/||w|| equilibrium defines an attractor in phase space. (2) AdamO's "Radial Tug-of-War" is literally a competing-force picture that phase portraits would reveal. (3) Truong & Truong's norm-hierarchy transitions are phase transitions that should manifest as bifurcations in phase space. (4) The saddle-to-saddle dynamics literature (OpenReview 2025) shows training traverses equilibria — phase portraits are the natural tool.
- **Novelty estimate**: 6/10 — Spectral dynamics (Yunis et al., 2024) provides a related visualization framework for singular values, but NOT for the (||w||, ||g||/||w||, alignment) state space. OUI (2025) provides a diagnostic metric but not phase portraits. The phase-space framing is moderately novel but more of a diagnostic/analysis contribution than a new algorithm.

## Phase 3: Self-Critique

### Against Candidate A (LyaWD)

- **Prior work attack**: Searched for "Lyapunov weight decay" and "Lyapunov regularization adaptive optimizer." Found LyAm (2507.11262) which uses Lyapunov for the optimizer update rule but NOT for weight decay. Found Lion-K Lyapunov analysis (2506.15054) which analyzes convergence with fixed WD but does NOT derive adaptive WD from Lyapunov conditions. Found "Abstract Lyapunov Control Optimizer" (arXiv:2407.01019) which is a general framework but does not address WD. **No existing work derives weight decay strength from Lyapunov stability conditions on a (||w||, ||g||/||w||, alignment) state space.** The specific combination is novel.

- **Methodological attack**: (1) The Lyapunov function choice is not unique — different V(w) yield different WD schedules, and there is no canonical choice. Risk: the "principled" derivation may reduce to an arbitrary design choice. Mitigation: show that a family of natural Lyapunov functions (quadratic in ||w||, incorporating ||g||/||w||) all yield WD schedules within a common parametric class. (2) Computing the Lyapunov derivative requires gradient information that introduces the same stochastic noise as alignment-based methods. Mitigation: use EMA-smoothed state variables, same as CWD/SWD. (3) Per-step overhead of computing V_dot and solving for lambda_t. Mitigation: for quadratic V, the solution is closed-form (like CWD's one-line formula).

- **Theoretical attack**: The continuous-time ODE approximation of SGD is standard but becomes less accurate with large learning rates or heavy-tailed gradient noise. The Lyapunov stability guarantees hold for the ODE but may not transfer cleanly to the discrete stochastic recursion. Mitigation: use discrete-time Lyapunov theory (Lyapunov difference inequalities) instead of continuous-time, following the Lion-K approach.

- **Scalability attack**: Per-layer state tracking (||w_l||, ||g_l||/||w_l||, cos_l) is cheap — these are scalar quantities per parameter group, already computed in most modern training loops. The control law evaluation is a simple algebraic expression. No scalability concern.

- **Verdict**: **STRONG** — Withstands all attacks. The novelty is genuine (no prior work derives WD from Lyapunov conditions), the methodology is sound (discrete-time Lyapunov theory is well-established), and the implementation is lightweight. The main risk (non-uniqueness of V) is addressable by showing convergence of the parametric family.

### Against Candidate B (Homeostatic WD)

- **Prior work attack**: Searched for "homeostatic weight decay neural network" and "set point regulation weight norm." Found: (1) Loshchilov's AdamWN targets a fixed norm — a fixed set point without adaptation. (2) AlphaDecay assigns per-module WD based on spectral density — a form of set-point assignment but static, not adaptive. (3) No work explicitly frames WD as homeostatic regulation with adaptive set points. The novelty holds.

- **Methodological attack**: (1) How to set the initial set point per layer? If it depends on architecture heuristics, the method is no better than manual tuning. (2) What drives set-point adaptation on the slow timescale? If it's validation loss, we've just reinvented hyperparameter tuning with extra steps. (3) The analogy to biological homeostasis is loose — biological systems have specific molecular mechanisms (receptor sensitivity, gene expression) that provide the set-point adaptation, while we'd need to invent an artificial analogue.

- **Theoretical attack**: The biological analogy, while inspiring, does not yield precise mathematical guarantees. "Multi-timescale" regulation is vague without specifying the timescale ratio and the adaptation dynamics. In contrast, Candidate A's Lyapunov framework provides rigorous guarantees by construction.

- **Scalability attack**: No concern — per-layer set points are cheap to maintain.

- **Verdict**: **MODERATE** — The idea is creative and the biological inspiration is compelling, but the lack of mathematical precision makes it more of a heuristic than a principled framework. The set-point adaptation mechanism needs a concrete specification to be rigorous. Could be strengthened by grounding the set-point dynamics in Lyapunov theory (merging with Candidate A).

### Against Candidate C (Phase-Space Diagnostics)

- **Prior work attack**: Searched for "phase portrait optimizer training dynamics" and "weight norm gradient norm phase space." Found: (1) Spectral dynamics (Yunis et al., 2024) tracks singular values — related but different state space. (2) Saddle-to-saddle dynamics (OpenReview 2025) analyzes loss plateaus and transitions. (3) Defazio's ||g||/||w|| analysis implicitly defines a 1D phase portrait. (4) No existing work plots full (||w||, ||g||/||w||, alignment) phase portraits for WD comparison. Novelty holds but is incremental.

- **Methodological attack**: Phase portraits are descriptive, not prescriptive. They reveal pathologies but do not directly suggest optimal WD schedules. The contribution is diagnostic, not algorithmic.

- **Theoretical attack**: The utility of phase portraits depends on the dynamics being low-dimensional enough to visualize. With per-layer state, we have 3L dimensions — phase portraits become projections that may lose critical information.

- **Scalability attack**: Computing per-layer phase trajectories requires logging 3 scalars per layer per step — trivial overhead.

- **Verdict**: **MODERATE** — Useful as a diagnostic tool and paper visualization, but insufficient as a standalone contribution for a top venue. Best used as the analysis/visualization component of a larger framework (e.g., Candidate A).

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C** (Phase-Space Diagnostics) dropped as a standalone idea because it is descriptive rather than prescriptive. However, the phase-space visualization will be incorporated into Candidate A as the diagnostic/analysis component.

### Strengthened Ideas

- **Candidate A (LyaWD)** strengthened by:
  1. **Incorporating Candidate B's multi-timescale insight**: The Lyapunov function can encode a norm set-point that adapts on a slow timescale (outer loop), while the WD control law adapts on a fast timescale (inner loop). This gives the homeostatic intuition mathematical rigor through nested Lyapunov functions (a standard technique in adaptive control).
  2. **Incorporating Candidate C's phase-space diagnostics**: Phase portraits in (||w_l||, ||g_l||/||w_l||) space serve as the primary visualization tool for understanding and comparing WD methods. Each WD method induces a different vector field in this phase space; LyaWD's vector field is designed to guarantee convergence to a desired operating point.
  3. **Addressing non-uniqueness of V**: Define a natural 2-parameter family of Lyapunov functions V(w) = alpha * ||w||^2 + beta * (||g||/||w|| - r*)^2, where r* is the target gradient-to-weight ratio (connecting to Defazio's layer-balancing insight). Show that for any (alpha, beta) > 0, the derived WD schedule has the same qualitative form but different quantitative parameters — establishing a parametric family rather than a single formula.
  4. **Connecting to existing methods as special cases**: Show that:
     - CWD corresponds to a bang-bang controller (binary switching on alignment sign) — the simplest controller in the family
     - SWD corresponds to a proportional controller on gradient norm — a P-controller
     - Fixed WD corresponds to open-loop control (no feedback) — the degenerate case
     - AdamWN/target-norm corresponds to a proportional controller on norm error
     - The full LyaWD is a PID-like controller derived from V-dot ≤ 0 conditions

### Additional Evidence Found

- **LyAm (arXiv:2507.11262, Jul 2025)**: Confirmed that this paper uses Lyapunov stability for Adam's update rule but does NOT address weight decay at all. LyAm treats the loss function itself as the Lyapunov function and modifies the learning rate to ensure monotonic decrease. Our approach is fundamentally different: we treat a function of the training state (not the loss) as the Lyapunov function and derive the WD schedule (not the LR) as the control input.

- **Abstract Lyapunov Control Optimizer (arXiv:2407.01019, Jul 2024)**: A general framework for designing optimizers via Lyapunov control. Focuses on local stabilization and global convergence of the optimizer itself, not on deriving regularization schedules. Complementary but non-overlapping.

- **PID-based gradient decay (NeurIPS 2024, OpenReview)**: Uses PID controller to adapt gradient decay for calibration. Confirms the viability of control-theoretic adaptation of training parameters, but addresses a different target (gradient decay for calibration, not weight decay for regularization/dynamics).

### Selected Front-Runner

**Candidate A (LyaWD)** — enriched with Candidate B's multi-timescale structure and Candidate C's phase-space diagnostics. This is the strongest idea because:
1. It provides a **principled mathematical framework** (Lyapunov stability) rather than heuristics
2. It **unifies existing methods** as special cases of a single control-theoretic framework
3. It yields **provable guarantees** (V-dot ≤ 0 ensures training stability by construction)
4. The cross-domain transfer is **structural, not metaphorical** — Lyapunov's direct method is a rigorous mathematical tool, not a vague analogy
5. The implementation is **lightweight** — closed-form WD update from quadratic Lyapunov functions

## Phase 5: Final Proposal

### Title

Lyapunov-Guided Weight Decay: A Control-Theoretic Framework for Adaptive Regularization in Deep Learning

### Hypothesis

For the dynamical system defined by the per-layer state variables s_l(t) = (||w_l(t)||, ||g_l(t)||/||w_l(t)||, cos(g_l(t), w_l(t))), there exists a quadratic Lyapunov function V(s) such that the weight decay schedule lambda_l(t) derived from the condition dV/dt ≤ 0 (i) provably stabilizes training dynamics, (ii) achieves equal or better generalization than the best fixed WD within comparable compute budgets, and (iii) subsumes CWD (binary sign controller), SWD (proportional gradient-norm controller), fixed WD (open-loop), and target-norm WD (proportional norm controller) as special cases of the control-theoretic family.

### Motivation

Weight decay is the most universally applied regularization technique in deep learning, yet its scheduling remains predominantly heuristic. The 2023-2026 period has seen an explosion of dynamic WD methods — CWD (ICLR 2026), AdamO (arXiv 2026), SWD (NeurIPS 2023), AlphaDecay (2025), ADANA (2026) — each proposing a different adaptation strategy based on different intuitions (alignment, orthogonal decomposition, gradient norms, spectral density, logarithmic schedules). Despite this diversity, no existing work provides a principled mathematical framework that:

1. **Derives** WD schedules from first principles (stability requirements) rather than proposing them heuristically
2. **Unifies** the seemingly disparate approaches by revealing them as special cases of a single framework
3. **Guarantees** training stability by construction rather than verifying it empirically post hoc

The key insight is that Defazio (2025) showed WD drives the gradient-to-weight ratio to a steady-state equilibrium — which IS the attractor of a dynamical system. CWD's sliding-mode interpretation (ICLR 2026) and AdamO's radial/tangential decomposition (2026) both hint at control-theoretic structure. We make this structure explicit by formulating the (||w||, ||g||/||w||, cos(g,w)) trajectory as a controlled dynamical system where WD is the control input, and applying Lyapunov's direct method to derive the optimal control law.

### Method

**State-Space Formulation**: Define per-layer state: s_l(t) = (n_l(t), r_l(t), delta_l(t)) where:
- n_l = ||w_l|| (weight norm)
- r_l = ||g_l||/||w_l|| (gradient-to-weight ratio, following Defazio)
- delta_l = cos(g_l, w_l) (gradient-weight alignment, following Sun et al. CVPR 2025)

The discrete-time dynamics under SGD with WD lambda_l(t) are:
- n_l(t+1) = (1 - lambda_l(t)) * n_l(t) - gamma_t * ||g_l(t)|| * delta_l(t) + O(gamma_t^2)
- r_l and delta_l evolve as functions of the loss landscape (treated as disturbances)

**Lyapunov Function Design**: Define a quadratic Lyapunov candidate:
V_l(s) = (1/2) * alpha * (n_l - n_l*)^2 + (1/2) * beta * (r_l - r_l*)^2

where (n_l*, r_l*) is the desired operating point (target norm and target gradient-to-weight ratio). The target r_l* can be set via Defazio's layer-balancing equilibrium condition.

**Control Law Derivation**: Require Delta_V_l = V_l(s(t+1)) - V_l(s(t)) ≤ -epsilon * V_l(s(t)) for some epsilon > 0 (exponential stability). Solving this inequality for lambda_l(t) yields a closed-form expression:

lambda_l(t) = f(n_l(t), r_l(t), delta_l(t), n_l*, r_l*, gamma_t, alpha, beta)

The solution is algebraic (no iterative optimization) because V is quadratic and the dynamics are approximately affine in lambda.

**Unification as Special Cases**:
- **Fixed WD**: lambda_l = const — open-loop control (ignoring state feedback)
- **CWD**: lambda_l = lambda_0 * 1[sign(w) = sign(g)] — bang-bang control on alignment sign
- **SWD**: lambda_l proportional to 1/||g|| — proportional control on gradient norm
- **Target-norm WD (AdamWN)**: lambda_l proportional to (||w|| - tau) — proportional control on norm error
- **LyaWD**: Full state-feedback control derived from Lyapunov stability — the most general case

**Multi-Timescale Extension**: The operating points (n_l*, r_l*) can themselves adapt on a slower timescale (every K steps), driven by an outer Lyapunov function defined over training-loss trajectory health (e.g., V_outer based on validation loss trend). This gives a nested adaptive control structure mirroring homeostatic regulation.

### Experimental Plan

**Metrics** (addressing the evaluation gap):
- Standard: test accuracy, validation loss, convergence speed (wall-clock)
- Dynamical: weight norm trajectory stability (variance), gradient-to-weight ratio convergence, alignment trajectory
- **Budget Equivalence Metric (BEM)**: Normalize all comparisons to equal FLOPs
- **Coupling Stability Index (CSI)**: Spectral radius of the linearized (||w||, ||g||/||w||) dynamics around equilibrium — smaller = more stable
- **Alignment Informativeness Score (AIS)**: Mutual information between alignment signal and optimal WD decision

**Experiments**:

1. **Phase-space analysis** (diagnostic, Candidate C contribution): Plot (||w_l||, ||g_l||/||w_l||) trajectories for CWD, SWD, fixed WD, AdamWN, and LyaWD on CIFAR-10/ResNet-20. Show qualitative differences (limit cycles vs. convergence to attractor vs. divergent spirals).

2. **CIFAR-10/100 comparison**: ResNet-20, VGG-16-BN. Baselines: No WD, Fixed WD (0.0005), CWD, SWD, AdamWN. LyaWD with quadratic V. Seeds: 42, 123, 456. Epochs: 200. Report mean +/- std of test accuracy, final ||w||, CSI.

3. **ImageNet comparison**: ResNet-50. Same baselines. LyaWD. Seeds: 42, 123, 456. Epochs: 90. Report top-1/top-5 accuracy, convergence curves, CSI.

4. **Ablation on Lyapunov function parameters**: Vary alpha/beta ratio in V to show robustness of the parametric family. Vary (n*, r*) targets to show sensitivity.

5. **Unification demonstration**: For each existing method (CWD, SWD, AdamWN), find the LyaWD parameters (alpha, beta, n*, r*) that best approximate it, and show the approximation is close — proving the subsumption claim.

6. **Falsification criterion**: If LyaWD with ANY reasonable (alpha, beta, n*, r*) fails to match or exceed the best fixed WD on CIFAR-10/ResNet-20 within the same compute budget, the hypothesis is falsified for the quadratic Lyapunov family.

### Resource Estimate

- **CIFAR-10/100 experiments**: ResNet-20 and VGG-16-BN, ~5 min/run on RTX PRO 6000. With 6 methods x 3 seeds x 2 datasets x 2 architectures = 72 runs. Total: ~6 hours.
- **ImageNet experiments**: ResNet-50, ~4-6 hours/run. With 6 methods x 3 seeds = 18 runs, parallelizable across 8 GPUs. Total: ~12-15 hours wall-clock.
- **Phase-space analysis**: Reuses training logs from above, post-processing only. ~30 min.
- **Ablation**: 5x parameter variations x 3 seeds on CIFAR-10/ResNet-20. Total: ~2 hours.
- **Total estimated compute**: ~20 hours wall-clock with 8 GPUs.

### Risk Assessment

1. **Risk: Lyapunov function family too restrictive** — Quadratic V may not capture the true training dynamics well enough, leading to conservative WD schedules that underperform heuristic methods.
   - Mitigation: Extend to polynomial or neural Lyapunov functions if quadratic is insufficient. Start with quadratic as the simplest case.

2. **Risk: Stochastic noise in state estimates degrades control quality** — The per-step (||w||, ||g||/||w||, cos(g,w)) estimates are noisy due to minibatch gradients.
   - Mitigation: Use EMA smoothing (same as CWD). The Lyapunov framework naturally accommodates noise through stochastic Lyapunov theory (expected decrease conditions).

3. **Risk: Theoretical guarantees hold for ODE but not discrete SGD** — The continuous-time Lyapunov analysis may not transfer to discrete, stochastic updates.
   - Mitigation: Use discrete-time Lyapunov difference inequalities (following Lion-K analysis, arXiv:2506.15054). Accept slightly weaker guarantees (expected decrease rather than deterministic decrease).

### Novelty Claim

**What is new**: The derivation of weight decay schedules from Lyapunov stability conditions on the (weight norm, gradient-to-weight ratio, alignment) state space. Specifically:

1. **No existing work derives WD from Lyapunov stability**. LyAm (arXiv:2507.11262) uses Lyapunov for LR adaptation; Lion-K (arXiv:2506.15054) uses Lyapunov for convergence analysis of fixed WD; neither derives WD itself from stability conditions.

2. **No existing work formulates the (||w||, ||g||/||w||, cos(g,w)) tuple as a controlled dynamical system**. Defazio (2025) analyzes ||g||/||w|| equilibrium informally; AdamO (2026) decomposes radial/tangential dynamics; neither treats WD as the control input of a formal state-space system.

3. **No existing work unifies CWD, SWD, fixed WD, and target-norm WD as special cases of a single control-theoretic framework**. Each method is currently presented as an independent contribution with independent motivation.

**Supporting evidence**: Systematic search of arXiv, Google Scholar, and web sources for "Lyapunov weight decay," "control theory weight decay optimizer," "Lyapunov regularization scheduling," and "feedback control weight decay" yielded zero results combining Lyapunov stability theory with weight decay schedule derivation.
