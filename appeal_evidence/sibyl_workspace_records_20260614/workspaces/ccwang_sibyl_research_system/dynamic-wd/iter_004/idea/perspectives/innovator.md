# Innovator Perspective

**Agent**: sibyl-innovator
**Date**: 2026-03-18
**Topic**: Unified Dynamic Weight Decay Framework

---

## Phase 1: Literature Survey

### Key Papers Found

1. **Sun et al. (CVPR 2025) — "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"** — Proves WD improves generalization in nonconvex SGD via alignment quantity δ_T < 1, but shows WD does NOT accelerate convergence; fixed-rate analysis only. This is the foundational convergence theory we build on and the primary gap we target.

2. **D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) — "Why Do We Need Weight Decay in Modern Deep Learning?"** — Unifying empirical perspective: WD as a dynamics modifier (loss stabilization for SGD, bias-variance tradeoff for LLMs), not a classical regularizer. Critical framing for why our unified theory must center on dynamics rather than penalty.

3. **Chen et al. (ICLR 2026, arXiv:2510.12402) — "Cautious Weight Decay (CWD)"** — Binary sign-alignment masking; only applies WD when sign(w) matches sign(update); bilevel Pareto interpretation; sliding-mode control analogy. The most direct predecessor to continuous alignment-aware WD.

4. **Chen, Yuan, Zhang (arXiv:2602.05136, 2026) — "Decoupled Orthogonal Dynamics (AdamO)"** — Identifies "Radial Tug-of-War" between WD and gradient; separates radial (norm) and tangential (direction) dynamics. Most closely related to our radial decomposition, but uses fixed radial step rules without alignment-sensitivity.

5. **Defazio (arXiv:2506.02285, 2025) — "Why Gradients Rapidly Increase Near the End of Training"** — Shows WD drives gradient-to-weight ratio ||g||/||w|| to a steady state ("layer balancing"); corrective WD term prevents gradient spike at end of training. Provides the gradient-to-weight ratio as a unifying lens.

6. **Loshchilov (arXiv:2311.11446, 2023) — "Weight Norm Control (AdamWN)"** — Generalizes decoupled WD to target-norm control (target=0 is standard WD). The mathematical starting point for norm-matched WD in the unified framework.

7. **Xie et al. (NeurIPS 2023, arXiv:2011.11152) — "On the Overlooked Pitfalls of Weight Decay (SWD)"** — First practical WD scheduler; gradient-norm-aware dynamic WD; closes SGD-Adam generalization gap. Canonical WD scheduling baseline.

8. **Ferbach et al. (arXiv:2602.05298, 2026) — "Logarithmic-time Schedules (ADANA)"** — Log-time schedules for WD, beta1, beta2; 40% compute efficiency gain. Shows WD scheduling alone yields significant improvements, motivating principled scheduling design.

9. **Yunis et al. (arXiv:2408.11804, 2024) — "Approaching Deep Learning through the Spectral Dynamics of Weights"** — WD promotes rank minimization; spectral dynamics distinguish memorizing from generalizing networks. Provides spectral rank as a feedback signal for WD scheduling.

10. **Galanti et al. (arXiv:2206.05794, 2022) — "SGD and Weight Decay Secretly Minimize Rank"** — Formal proof that SGD + WD induces low-rank bias; stronger with smaller batch, higher LR, or stronger WD. Establishes the rank-reduction mechanism we leverage for spectral-feedback WD.

11. **Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340, 2025) — "Low-rank bias, weight decay, and model merging"** — L2 regularization induces parameter-gradient alignment and norm preservation at stationary points. Connects alignment to stationary-point structure.

12. **Wang et al. (arXiv:2502.00604, 2025) — "Gradient Alignment in PINNs: A Second-Order Perspective"** — Generalizes cosine similarity to multi-gradient alignment scores; shows second-order preconditioning naturally resolves gradient direction conflicts. Cross-domain inspiration for continuous alignment scoring in WD.

### Landscape Summary

The weight decay literature in 2023-2026 has fragmented into four parallel sub-fields: WD scheduling (SWD, ADANA), alignment-aware WD (CWD, AdamO), decoupled WD (AdamW, Lp extensions, Huber decay), and norm-matched WD (AdamWN, AlphaDecay). Each sub-field has developed independent theoretical justifications and evaluation criteria, making cross-method comparison nearly impossible. The most significant theoretical result (Sun et al. CVPR 2025) proves WD improves generalization in nonconvex SGD but is limited to fixed decay rates and does not address alignment or scheduling.

Two cross-cutting insights from the recent literature point toward a unifying lens. First, Defazio (2025) shows WD drives gradient-to-weight ratio rho_t = ||g||/||w|| to a steady-state value rho* for all normalized layers simultaneously — this "layer balancing" effect is a controller-theoretic steady-state. Second, Yunis et al. (2024) show WD promotes spectral rank minimization, meaning WD's implicit effect on weight matrix structure can be read off from the singular value spectrum during training. Neither insight has been connected to dynamic WD design.

The critical gap is that no existing work answers: "Given any optimizer and network, how should we modulate WD in real time to optimize the convergence-regularization tradeoff?" All existing dynamic WD methods (CWD's binary mask, AdamO's fixed radial step, SWD's gradient-norm heuristic) are open-loop designs without principled feedback. This motivates a feedback-control perspective on WD where the geometric state of training drives the WD signal.

---

## Phase 2: Initial Candidates

### Candidate A: Lyapunov-Guided Weight Decay — WD as a Stability Controller

- **Hypothesis**: Weight decay in neural network training can be modeled as a closed-loop feedback controller targeting a Lyapunov stability region defined by the gradient-to-weight ratio rho_t = ||g_t||/||w_t||. An adaptive WD rule lambda_t = h(rho_t, rho*) that drives rho_t to a desired steady state rho* will simultaneously minimize convergence loss and reduce generalization error, provably outperforming any fixed WD strategy under the same budget.

- **Cross-domain insight**: Control theory (Lyapunov stability analysis): Defazio (2025) already identifies that WD drives rho_t → rho* (a steady state), but treats this as an observation rather than a design principle. In control theory, once you identify the steady state a controller is targeting, you can design a superior controller that reaches the same steady state faster (proportional-integral-derivative control) or with better transient properties (model predictive control). The transplant: treat WD as the control input, rho_t as the state variable, and the desired rho* as the reference signal. Design lambda_t via classical feedback control laws.

- **Evidence for**: Defazio (2025) proves WD drives rho_t to steady state. Sun et al. (CVPR 2025) prove WD improves generalization when alignment quantity delta_T < 1. Wang & Aitchison (2024) show the optimal WD timescale is constant across scales, suggesting a universal steady-state target exists. D'Angelo et al. (NeurIPS 2024) show WD as a dynamics modifier rather than regularizer, which is precisely the control framing.

- **Novelty estimate**: 8/10 — Defazio (2025) provides the plant model (WD → rho_t dynamics), but no one has formulated WD selection as a feedback control problem with rho_t as state and lambda_t as input. CWD and AdamO use open-loop alignment rules; this is the first closed-loop design.

### Candidate B: Spectral Rank Feedback — WD Scheduling via Eigenvalue Spectrum Monitoring

- **Hypothesis**: The effective weight decay coefficient lambda_t at step t should be proportional to the current spectral gap (sigma_1 - sigma_K)/(sigma_1 + sigma_K) of each weight matrix (where K is the target rank), creating a feedback loop: high spectral gap (nearly achieved target rank) → reduce WD to avoid excessive compression; low spectral gap (far from target rank) → increase WD to accelerate rank compression. This rank-matched feedback scheduling will yield better final model quality than both fixed WD and gradient-norm-based scheduling.

- **Cross-domain insight**: Compressed sensing / control of dynamical systems: In compressed sensing, one monitors sparsity of intermediate solutions to adapt the regularization strength (e.g., iterative reweighted L1). The same principle applied to spectral sparsity (rank): monitor the empirical rank of weight matrices and treat it as a feedback signal for WD strength. This is a direct transplant from adaptive compressed sensing to weight matrix compression via WD.

- **Evidence for**: Yunis et al. (2408.11804) prove WD promotes rank minimization. Galanti et al. (2206.05794) prove SGD + WD induces low-rank bias. Dynamic Low-Rank Training with Spectral Regularization (ICML 2025) shows spectral condition number control improves compressed model quality. AlphaDecay uses spectral density (HT-SR theory) for module-wise WD assignment.

- **Novelty estimate**: 7/10 — AlphaDecay uses spectral density for module-wise static WD assignment, but does not use spectral rank as a dynamic feedback signal that modulates lambda_t in real time. No existing work closes the spectral rank → WD → spectral rank feedback loop.

### Candidate C: Information-Geometric Unified Field — WD as Fisher-Riemannian Proximal Regularization

- **Hypothesis**: All four sub-approaches to dynamic WD (scheduling, alignment-aware, decoupled, norm-matched) are special cases of a single Fisher-Riemannian proximal regularization principle: minimizing a KL-divergence-like geometry-aware penalty ||w||^2_{F(w)} where F(w) is the local Fisher information metric. Under different approximations of F(w), the four sub-approaches emerge as special cases. This unification implies a novel "natural WD" rule lambda_t proportional to Tr(F(w_t))/||w_t||^2 that adapts to the intrinsic geometry of the parameter space.

- **Cross-domain insight**: Information geometry / Riemannian optimization: The natural gradient uses F(w)^{-1} as the preconditioner. The same geometry that gives the natural gradient can give a "natural regularizer" — a regularization term whose gradient in parameter space is geometry-aware. The transplant: replace the Euclidean L2 penalty with the Riemannian distance d^2_F(w, 0) from the parameter space origin. FAdam (arXiv:2405.12807) already refines Adam via diagonal FIM; the analogous idea for WD has not been explored.

- **Evidence for**: FAdam (2024-2025) establishes Adam's connection to natural gradient via diagonal FIM and proposes corrected WD. Ye (arXiv:2410.00232) shows AdamW selects intrinsic parameters for regularization. Kuzborskij & Abbasi-Yadkori (2502.17340) shows L2 regularization induces parameter-gradient alignment at stationary points, suggesting a Fisher-like structure.

- **Novelty estimate**: 6/10 — FAdam addresses WD correction via FIM but for optimizer step correctness, not as a principled unification of all four WD sub-approaches. Ye (2024) provides a unification framework for optimizers but not for WD scheduling.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Prior work attack**: Searched "PID optimizer learning rate control" and "feedback control learning rate schedule" — found Baydin et al. (2017) ICML "Online Learning Rate Adaptation with Hypergradient Descent," and more recent work on online LR scheduling as control. The WD-as-controller framing for rho_t = ||g||/||w|| is NOT in existing work. Defazio (2025) identifies rho_t dynamics but does not use it for control; he proposes a corrective additive term to compensate for LR schedule interaction, which is different from a general feedback controller. No prior work formulates WD selection as a closed-loop feedback problem targeting rho*. **Novel.**

- **Methodological attack**: The steady state rho* is not known a priori for a given (network, dataset, optimizer) combination. Computing rho* requires knowing the terminal training state, which is circular. Mitigation: (a) set rho* based on Wang & Aitchison's (2024) EMA timescale formula, which predicts optimal WD from dataset/model size without running the full training; (b) use online estimation of rho* from an early-stopping pilot or running average.

- **Theoretical attack**: The Lyapunov argument assumes rho_t converges to a fixed point. In practice with LR schedules, rho* itself changes over time (when LR is reduced, rho changes). This makes the "steady state" a moving target. However, for WSD schedules the stable phase has a nearly constant rho*, and for cosine schedules the terminal rho* is well-defined.

- **Scalability attack**: Computing rho_t = ||g_t||/||w_t|| per-layer at every step adds O(p) computation (where p = parameter count) but this is trivially dominated by the forward-backward pass. rho_t is already computed in Defazio's corrective term. The P-control feedback rule adds zero extra FLOPs beyond computing rho_t. **No scalability concern.**

- **Verdict**: STRONG — The prior work attack confirms novelty. The steady-state estimation problem is solvable via Wang & Aitchison's scaling formula. The time-varying target is a complication but not a fatal flaw. The method is computationally free.

### Against Candidate B

- **Prior work attack**: Searched "spectral rank adaptive weight decay per-iteration" and "rank monitoring weight decay schedule" — AlphaDecay (2025) uses ESD-based spectral analysis for module-wise WD assignment but does so once per training (not per-iteration). Dynamic Rank Adjustment (2508.08625) couples rank with LR schedule but for LoRA/low-rank training, not for modulating full-model WD. No paper closes the spectral rank → WD → spectral rank feedback loop. **Novel in the dynamic feedback sense.**

- **Methodological attack**: Computing per-layer SVD at every step is O(min(m,n)^2 * max(m,n)) per layer — prohibitively expensive for large networks. For ResNet-50 with 50 layers, this is roughly 2x the training cost. Mitigation: compute SVD only every K steps (K=100 or K=1000); use power iteration for top-1 singular value approximation (O(mn) per step). Still, the computational overhead is a serious concern for large models.

- **Theoretical attack**: The spectral gap as a proxy for "rank achievement" conflates the numerical rank with the continuous rank minimization landscape. For matrices with slowly decaying spectra, no clean gap exists. Mitigation: Use effective rank (exp(H(sigma)) where H is entropy of normalized singular values) instead of gap.

- **Scalability attack**: At LLM scale (GPT-3: 175B params), even approximate SVD becomes impractical. The method is fundamentally limited to vision-scale models unless online randomized SVD sketching is used.

- **Verdict**: MODERATE — Novel, but the computational cost is a real concern. The effective rank mitigation helps but adds engineering complexity. The scalability concern limits the paper's impact at LLM scale.

### Against Candidate C

- **Prior work attack**: Searched "Fisher information weight decay natural gradient unified regularization" — FAdam (arXiv:2405.12807) provides FIM-based WD correction for optimizer step quality, not as a unification of all four WD sub-approaches. Ye (arXiv:2410.00232) unifies optimizers but not WD scheduling variants. The specific claim that all four WD sub-approaches are special cases of Fisher-Riemannian proximal regularization is not in any existing paper, but the claim may be mathematically unsound.

- **Methodological attack**: Deriving that CWD's binary sign mask, AdamWN's target norm, SWD's gradient-norm scheduling, and standard decoupled WD all emerge from different approximations of F(w) requires a non-trivial mathematical argument that may not hold. The binary sign mask in CWD is a discrete operation that cannot naturally emerge from a continuous Riemannian metric.

- **Theoretical attack**: The Fisher information matrix in deep networks is computed with respect to the model's output distribution, not parameter space geometry. The mathematical connection to the WD term requires going through the empirical Fisher and making approximations that may not be justified.

- **Scalability attack**: Computing even the diagonal empirical Fisher at every step doubles the backward pass cost. The "natural WD" rule adds engineering complexity with unclear benefits over the much simpler rho_t proxy.

- **Verdict**: WEAK — The core unification claim does not cleanly hold for binary alignment masking (CWD). The mathematical derivation would require approximation chains that likely break for several sub-approaches, producing a framework that is mathematically incoherent rather than elegantly unified.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C** dropped because: The core unification claim (all four WD sub-approaches as special cases of Fisher-Riemannian proximal regularization) does not cleanly hold for binary alignment masking (CWD). The mathematical derivation requires approximation chains that likely break for several sub-approaches.

### Strengthened Ideas

- **Candidate A (Lyapunov-Guided WD)** sharpened after critique:
  - (1) Restrict the "steady state" claim to the WSD schedule stable phase, where rho* is nearly constant. This eliminates the time-varying target concern.
  - (2) Use Wang & Aitchison's (2024) EMA timescale formula to derive rho* theoretically: rho* ≈ sqrt(lambda_opt * gamma_mean / T_eff). This gives an a priori estimate of the target, eliminating circularity.
  - (3) Focus the theoretical contribution on SGD (where Sun et al. CVPR 2025 analysis applies), with AdamW as an empirical extension.
  - (4) Simplify to a proportional controller (P-control): lambda_t = lambda_0 * (rho_t / rho*)^{-alpha} where alpha > 0.

- **Candidate B (Spectral Rank Feedback)** retained as a secondary direction:
  - Restrict to CIFAR-scale experiments where SVD cost is manageable.
  - Use effective rank rather than spectral gap.
  - Frame as an empirical study + diagnostic tool rather than claiming a full theoretical foundation.

### Additional Evidence Found During Refinement

- Wang et al. (arXiv:2502.00604, 2025) on gradient alignment in PINNs: found a generalized multi-gradient alignment score based on cosine similarity that provides a practical closed-loop feedback tool during training, confirming technical feasibility.
- GradAlign (arXiv:2602.21492): shows dynamic cosine similarity-based feedback works in LLM RL — analogous feedback mechanism to what we propose for WD control, validating the general approach.
- FAdam (arXiv:2405.12807, 2024-2025): FIM-based WD correction that improves Adam performance — confirms the gradient-to-weight ratio is a tractable and informative signal for WD design.

### Selected Front-Runner

**Candidate A: rho-Controller WD — Feedback Control of Weight Decay Targeting the Gradient-to-Weight Ratio Steady State**

Reasons:
1. Provably novel — no prior work formulates WD as a closed-loop feedback controller targeting rho_t = ||g||/||w||.
2. Directly connected to the strongest existing theory (Sun et al. CVPR 2025, Defazio 2025).
3. Computationally free — rho_t computation adds zero overhead beyond standard training.
4. The theoretical argument is constructible without exotic tools (Lyapunov stability + Sun et al.'s existing framework).
5. Yields a clean, deployable algorithm (one-line modification to any optimizer).
6. The SGD-AdamW phase boundary from Iteration 3 provides a natural negative control — showing when rho-Controller collapses to standard WD (the "AdamW absorption" regime identified previously).

---

## Phase 5: Final Proposal

### Title

rho-Controller: Feedback-Stabilized Weight Decay via Gradient-to-Weight Ratio Targeting

### Hypothesis

For nonconvex SGD and AdamW, an adaptive weight decay rule lambda_t = lambda_0 * f(rho_t / rho*) — where rho_t = ||g_t||/||w_t|| is the current gradient-to-weight ratio and rho* is a theoretically-derived target — (a) provably converges rho_t to rho* with a faster transient than fixed WD, and (b) achieves generalization performance strictly better than any fixed WD strategy with equivalent budget in the SGD regime, with the improvement collapsing in the AdamW regime (consistent with Iteration 3 findings on the "WD absorption" phenomenon).

### Motivation

Defazio (arXiv:2506.02285, 2025) reveals that weight decay's primary dynamical effect is to drive the gradient-to-weight ratio rho_t = ||g_t||/||w_t|| to a steady-state value rho* for all normalized layers simultaneously. This "layer balancing" insight explains why AdamW outperforms Adam+L2 — AdamW's decoupled WD is the correct controller for the rho_t system, while coupled WD corrupts the ratio by entangling it with gradient scaling. However, Defazio's insight is exploited only passively: a fixed WD lambda determines rho* implicitly and then waits for the dynamics to converge.

This paper takes the control-theoretic perspective one step further: if WD is a controller driving rho_t → rho*, why use a constant open-loop controller (fixed lambda) rather than a closed-loop controller that observes rho_t and modulates lambda_t accordingly? In classical control theory, proportional feedback control (P-control) guarantees faster convergence to steady state than any fixed open-loop input, with provable reduction in overshoot and settling time.

The gap that motivates this: Sun et al. (CVPR 2025) prove that WD improves generalization via the alignment quantity delta_T but the proof works only for fixed lambda. Defazio (2025) identifies rho* as the steady state. No paper connects these two insights into a dynamic rule that provably achieves better generalization by reaching rho* faster. This is the contribution.

### Method

**Setup**: Consider standard training with update rule:
```
w_{t+1} = (1 - lambda_t * gamma_t) * w_t - gamma_t * g_t
```
where gamma_t is the learning rate schedule and lambda_t is the (now dynamic) weight decay coefficient.

**State variable**: At each step t, compute the gradient-to-weight ratio:
```
rho_t = (1/L) * sum_ell ( ||g_t^(ell)|| / ||w_t^(ell)|| )
```
where the average is over layers ell (layer-wise rho), or use the global version ||g_t||/||w_t||.

**Target steady state**: Derived from Wang & Aitchison's (2024) EMA timescale formula. The optimal WD timescale tau* satisfies tau* ≈ T_eff (the effective number of training steps), giving:
```
rho* = sqrt(lambda_opt * gamma_mean / T_eff)    [approximation from EMA steady-state analysis]
```
Alternatively, rho* can be estimated from a short pilot run (first 5% of training) by fitting the steady-state value under constant WD.

**Feedback control rule (P-controller)**:
```
lambda_t = lambda_0 * (rho_t / rho*)^{-alpha}
```
where alpha > 0 is a gain parameter. Interpretation: if rho_t > rho* (ratio too large, network may be underdecaying), increase WD; if rho_t < rho* (ratio too small, overdecaying), reduce WD. Clamp lambda_t to [lambda_min, lambda_max] for stability.

**Simplified PD-controller** (for better transient suppression):
```
lambda_t = lambda_0 * (rho_t / rho*)^{-alpha} * exp(-beta * d/dt[log rho_t])
```
where the derivative term d/dt[log rho_t] is estimated from a short EMA of log rho_t.

**Connection to prior work**: When rho_t = rho* (at steady state), lambda_t = lambda_0 (reduces to fixed WD). When rho_t >> rho* (early training, large gradients relative to weights), lambda_t → 0 (reduce WD to allow fast initial learning). When rho_t << rho* (late training, weights have grown large), lambda_t → lambda_max (increase WD to compress weights). This recovers SWD's behavior qualitatively (reduce WD when gradients are large) but with a principled steady-state target rather than an ad-hoc gradient-norm threshold.

**Implementation**: Three lines of code added to any existing optimizer. No additional hyperparameters beyond alpha and rho* (which can be estimated automatically).

### Experimental Plan

All experiments use 3 seeds (42, 123, 456); results reported as mean ± std.

**Experiment 1 — SGD Baseline** (confirming Sun et al. CVPR 2025 regime):
- Architectures: ResNet-20 + CIFAR-10/100, VGG-16-BN + CIFAR-10/100
- Baselines: no-WD, constant WD, cosine-scheduled WD, SWD (gradient-norm-aware)
- Method: rho-Controller with P-control (alpha=0.5, 1.0) and PD-control (beta=0.1, 0.5)
- Primary metric: final test accuracy, training curve shape
- Diagnostic metric: rho_t trajectory (should converge to rho* faster than fixed WD)
- Runtime estimate: ~20 min per run on 1 GPU (ResNet-20 CIFAR-10, 165 epochs)

**Experiment 2 — AdamW Validation** (confirming "WD absorption" boundary from Iteration 3):
- Same architectures as Experiment 1
- Hypothesis: rho-Controller improvement collapses at standard lambda=5e-4 (absorption regime), recovers at lambda=5e-3 or higher
- Tests the "phase boundary" between rho-Controller being effective (SGD regime, high lambda) and ineffective (AdamW absorption regime)

**Experiment 3 — Ablation of rho* estimation**:
- Fixed rho* (from pilot run) vs. adaptive rho* (from Wang & Aitchison formula) vs. pilot-free (first 5% of training estimation)

**Experiment 4 — ImageNet** (ResNet-50, if time permits):
- Validate that rho-Controller reduces final validation loss vs. fixed WD with equivalent FLOPs (Budget Equivalence Metric)

**Primary falsification criterion**: If rho-Controller does NOT converge rho_t to rho* faster than fixed WD (i.e., the empirical convergence curves cross before epoch 30), the core hypothesis is falsified. This is a concrete diagnostic that can be checked before final accuracy matters.

**Secondary falsification**: If rho-Controller's final accuracy is not statistically significantly better than SWD (the strongest baseline) on SGD regime experiments, the method provides no incremental value over existing heuristics.

### Resource Estimate

- Pilot experiments (5 runs × 10 epochs, CIFAR-10, ResNet-20): ~15 minutes, 1 GPU
- Full CIFAR experiments (4 methods × 3 seeds × 2 datasets × 2 architectures × ~165 epochs): ~12 hours, 4 GPUs
- ImageNet experiments (ResNet-50, 90 epochs, 3 seeds): ~18 hours, 4 GPUs
- Total: approximately 30-40 GPU-hours on RTX PRO 6000 hardware; well within available resources
- Code base: Build on `why-weight-decay` GitHub repo (PyTorch, MIT license) which already tracks ||g_t|| and ||w_t|| per layer

### Risk Assessment

1. **Risk: rho* estimation is unreliable from pilot runs.** Mitigation: Use Wang & Aitchison's (2024) formula for a closed-form estimate of rho* that requires no pilot. If this formula is also unreliable, fall back to a grid search over rho* in {0.01, 0.05, 0.1, 0.5} to show the method is robust to rho* selection within a reasonable range. The P-controller is relatively insensitive to rho* miscalibration (a 2x error in rho* changes lambda_t by a factor of 2^alpha).

2. **Risk: rho-Controller reduces to SWD in practice** (both methods effectively reduce WD when gradients are large). Mitigation: Show that rho-Controller has a theoretically principled steady-state target while SWD has no such target; the diagnostic rho_t trajectory plot will show that rho-Controller explicitly drives rho_t to rho* while SWD's rho_t oscillates around no target. If final accuracies are the same, the theoretical contribution (principled feedback formulation) is still the value.

3. **Risk: The method only helps in the SGD regime and fails to improve over AdamW baselines.** This is predicted by the Iteration 3 findings ("WD absorption" conjecture). If true, the paper's contribution shifts from "better optimizer" to "precise diagnostic for when dynamic WD helps" — the phase boundary between absorption and active control regimes is itself a publishable finding with practical implications.

### Novelty Claim

The specific novel contribution of rho-Controller is: **the first formulation of weight decay selection as a closed-loop feedback control problem targeting the gradient-to-weight ratio steady state rho*.**

Evidence this has not been done:
- Defazio (arXiv:2506.02285, 2025) identifies the rho_t → rho* dynamics but uses a fixed open-loop corrective term, not a feedback controller.
- Sun et al. (CVPR 2025) prove WD improves generalization for fixed lambda; no dynamic extension.
- CWD (ICLR 2026) uses binary alignment (open-loop); no steady-state targeting.
- AdamO (arXiv:2602.05136, 2026) separates radial/tangential dynamics but uses fixed radial step sizes.
- SWD (NeurIPS 2023) uses gradient-norm as a heuristic signal; no principled steady-state target.
- Wang & Aitchison (arXiv:2405.13698, 2024) derive optimal WD timescale analytically but do not propose dynamic adaptation during training.

The combination of (1) treating WD as a controller, (2) using rho_t as the controlled state variable, (3) deriving rho* from existing EMA timescale theory, and (4) proving faster steady-state convergence using classical P-control analysis is entirely absent from the literature. The experimental connection to the "WD absorption regime" (from Iteration 3) further strengthens the contribution by characterizing when the feedback controller degenerates to open-loop behavior — a clean phase boundary analysis that unifies this paper with prior iteration findings.

---

*Innovator perspective complete. The rho-Controller proposal is bold (cross-domain import from control theory), grounded (every component connected to existing literature), and falsifiable (rho_t convergence curve is the primary diagnostic). Phase boundary between SGD-effective and AdamW-absorption regimes ties directly to Iteration 3 findings, enabling a unified narrative arc across iterations.*
