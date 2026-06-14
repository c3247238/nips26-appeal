# Innovator Perspective — Iteration 6

**Agent**: sibyl-innovator
**Date**: 2026-03-18
**Topic**: Unified Dynamic Weight Decay Framework — Novel Cross-Domain Idea Generation
**Focus**: Fresh unconventional angles beyond the Phi invariance / ρ-regime paradigm already explored in Iterations 1–5

---

## Phase 1: Literature Survey

### Key Papers Found

1. **Defazio (2025), arXiv:2506.02285** — WD drives all normalized layers' gradient-to-weight ratio ‖g‖/‖w‖ to the same steady-state value ("layer balancing"). This provides a clean control-theoretic interpretation: WD is a proportional controller driving ‖g‖/‖w‖ to a setpoint. No paper has formalized alignment-aware WD within this feedback-control framework.

2. **Sun et al. (CVPR 2025)** — First proof that WD improves generalization but does NOT accelerate convergence in nonconvex SGD. The alignment quantity δ_T < 1 is the key condition. This is the theoretical bedrock our project builds upon.

3. **Chen et al. (ICLR 2026), arXiv:2510.12402 — CWD** — Binary sign-alignment mask selectively applies WD only when sign(w) agrees with sign(update). The sliding-mode control interpretation is mentioned but not formalized. Novelty limit: binary-only, no continuous modulation theory.

4. **Chen, Yuan, Zhang (arXiv:2602.05136) — AdamO** — Identifies "Radial Tug-of-War" (WD opposes gradient along radial direction); decouples radial/tangential dynamics. Structural key insight for our framework: every existing WD method implicitly fights a tug-of-war; the question is how to resolve it.

5. **Xie & Li (2024), arXiv:2404.04454** — AdamW implicitly performs ℓ∞-constrained optimization via Frank-Wolfe. This explains the Phi invariance phenomenon found in Iterations 1–4: under AdamW, all WD strategies collapse to the same ℓ∞ ball, washing out schedule differences.

6. **Yunis et al. (arXiv:2408.11804)** — Spectral dynamics as unifying lens; WD promotes rank minimization; spectral dynamics distinguish memorizing from generalizing networks. Directly relevant: no paper uses the instantaneous spectral rank as a feedback signal for dynamic WD scheduling.

7. **PIDAO (Nature Communications, 2024)** — PID controller structure applied to neural network optimization; proportional-integral-derivative accelerated optimizer. The key insight: gradient descent is a proportional controller; adding integral and derivative terms creates qualitatively different dynamics. No paper applies PID control structure to the WD coefficient itself.

8. **Galanti et al. (arXiv:2206.05794)** — SGD + WD secretly minimize rank; stronger with smaller batch, higher LR, stronger WD. Static analysis; no connection to dynamic WD.

9. **D'Angelo et al. (NeurIPS 2024), arXiv:2310.04415** — WD as training dynamics modifier via "loss stabilization mechanism" for SGD and "bias-variance tradeoff" for LLMs. For LLMs, WD is beneficial for plasticity (Han et al., arXiv:2602.11137), creating more adaptable representations.

10. **Han et al. (arXiv:2602.11137)** — Larger WD during pretraining produces more plastic (adaptable) LLM representations. The plasticity-stability tradeoff from continual learning maps onto WD magnitude: larger WD = more plasticity, smaller WD = more stability. This maps directly to a continual learning analogy for WD scheduling.

11. **Truong & Truong (arXiv:2603.07323)** — WD traverses norm hierarchy from shortcut to structured representations; transition delay logarithmic in norm ratio. This means WD has a phase-transition character: when norm crosses a threshold, qualitatively new behavior emerges. Dynamic WD could deliberately trigger or suppress these transitions.

12. **Flatness-aware SGLD (arXiv:2510.02174)** — Random weight perturbation explicitly links to Hessian-trace regularization. Key insight: WD and random perturbation are related mechanisms; combining them with curvature feedback creates a unified regularization.

### Landscape Summary

The current landscape has four converging insights that no single paper has synthesized:

**Insight A (Control Theory)**: WD acts as a controller driving ‖g‖/‖w‖ to a setpoint (Defazio 2025). This is proportional control. The literature has not yet considered integral or derivative control analogs for WD — which would address integral drift and anticipate regime transitions.

**Insight B (Phase Transitions)**: WD drives qualitative transitions — from high-rank to low-rank representations (Galanti 2022, Yunis 2024), from shortcut to structured representations (Truong 2026). These are threshold phenomena, not smooth processes. A feedback controller that tracks the proximity to phase transitions and adjusts WD accordingly could accelerate desirable transitions and suppress undesirable ones.

**Insight C (Plasticity-Stability)**: WD magnitude maps to the plasticity-stability tradeoff (Han 2026, D'Angelo 2024). In continual learning, this tradeoff is managed explicitly through task-dependent elastic regularization (EWC uses a Fisher-weighted penalty). The same principle could apply within a single training run: early training benefits from high plasticity (low WD), late training benefits from high stability (high WD), but the transition timing should be data-dependent and geometry-dependent rather than time-dependent.

**Insight D (Phi Invariance Under AdamW)**: The Phi invariance finding from our Iterations 1–4 shows that under standard AdamW, all WD strategies are equivalent because the ℓ∞ constraint washes out schedule differences (Xie & Li 2024). This suggests that breaking out of the Phi-invariant regime requires changing the optimizer's geometric structure, not just the WD schedule. The novel approach should either: (a) operate in a regime where Phi invariance fails (high ρ, SGD, or large-scale settings), or (b) propose a WD formulation that structurally escapes the ℓ∞ ball collapse.

The research gap that emerges: a principled, curvature-aware, phase-transition-tracking WD controller that synthesizes control theory (PID structure), spectral feedback (rank proximity), and the plasticity-stability tradeoff — and which is specifically designed to operate in the Phi-non-invariant regime by coupling WD to the Hessian rather than just the gradient-weight angle.

---

## Phase 2: Initial Candidates

### Candidate A: Curvature-Conditioned Weight Decay (CCWD)

**Hypothesis**: Weight decay coefficient λ_t should be inversely proportional to the current loss landscape sharpness (local Hessian trace), such that WD is reduced in sharp regions (where weight contraction would displace from the desired flat minimum) and increased in flat regions (where aggressive norm control is safe). This coupling between WD and curvature will yield faster convergence to flat minima than time-schedule-based WD, while maintaining convergence guarantees of SGDW.

**Cross-domain insight**: From SAM (Foret et al. 2021) and flatness-aware SGLD (2025): weight perturbation is equivalent to Hessian-trace regularization. If WD is a norm-contracting regularizer, conditioning it on local curvature avoids the failure mode of SAM+WD combinations (they can interfere by simultaneously pushing toward and away from sharp regions). The PID controller analogy: alignment-aware WD methods are proportional controllers; Hessian-conditioned WD adds a derivative term that anticipates curvature changes.

**Evidence for**: (1) CR-SAM (arXiv:2312.13555) shows combining Hessian-trace regularization with training yields 2.4% accuracy gain on ResNets; (2) flatness-aware SGLD (arXiv:2510.02174) links random perturbation to Hessian-trace control; (3) CRR (arXiv:2511.01438) shows input-space curvature regularization is competitive with SAM without extra compute; (4) our own results: alignment-based methods (CWD, SWD) do not outperform constant WD under AdamW — but Hessian curvature information operates at a different geometric level than gradient-weight angle.

**Novelty estimate**: 7/10 — CR-SAM combines SAM with Hessian regularization, but no paper proposes conditioning the WD coefficient itself on local curvature (rather than using WD alongside a separate Hessian penalty).

---

### Candidate B: Spectral-Phase-Transition WD Scheduler (SPWD)

**Hypothesis**: Effective rank of weight matrices undergoes a phase transition during training: from high rank (exploration) to low rank (compression). WD is the primary driver of this transition (Galanti 2022, Yunis 2024). A WD schedule that detects phase-transition proximity via a rank velocity signal — the rate of change of effective spectral rank — and pulses WD at maximal strength during the rapid rank collapse phase, then reduces WD post-collapse, will produce qualitatively better representations than any time-based schedule.

**Cross-domain insight**: From condensed matter physics — phase transitions are detected by susceptibility (derivative of order parameter near the critical point). The spectral rank is the order parameter for the low-rank bias phenomenon. The susceptibility analog is |drank/dt|. A WD controller that maximizes |drank/dt| when rank is still high (to accelerate the transition) and minimizes WD when rank has stabilized (to avoid over-compression) is a phase-transition-tracking controller. This maps the Spectral Dynamics work (Yunis 2024) onto a closed-loop control problem.

**Evidence for**: (1) Yunis et al. (2408.11804) show spectral rank dynamics distinguish generalizing from memorizing networks; WD promotes rank minimization; (2) Galanti et al. (2206.05794) show stronger WD → lower rank; (3) ARSVD (arXiv:2504.20078, 2025) uses spectral entropy for per-layer adaptive rank selection in compression — analogous per-layer spectral signal; (4) AlphaDecay uses spectral heavy-tailedness for per-module WD assignment, but statically, not dynamically tracking transitions.

**Novelty estimate**: 8/10 — no existing method uses rank velocity as a WD feedback signal; closest is AlphaDecay (static per-module spectral analysis). Phase-transition tracking in optimization is a genuinely new angle.

---

### Candidate C: PID-Weight-Decay (PID-WD)

**Hypothesis**: The standard decoupled WD update w ← (1-λ)w - η·u is a pure proportional controller driving ‖w‖ toward zero. Adding integral and derivative terms to the WD coefficient creates qualitatively richer dynamics:
- Integral term: accumulates the "debt" of alignment (Σ_t cos(g_t, w_t)) — if the trajectory has been persistently misaligned over many steps, apply a corrective burst of WD
- Derivative term: anticipates alignment trend changes — if alignment is rapidly deteriorating, pre-emptively reduce WD to avoid over-contraction

The resulting PID-WD rule: λ_t = λ_P · (1-δ_t) + λ_I · Σ_{s=0}^{t} (1-δ_s) + λ_D · (δ_{t-1} - δ_t)

This controller will achieve faster convergence to well-regularized solutions than proportional-only methods (CWD), with formal stability guarantees analogous to PID-controlled dynamical systems.

**Cross-domain insight**: From control theory — PID controllers are optimal for linear systems and robust for nonlinear systems. The WD problem is precisely a control problem: the "plant" is the optimization trajectory, the "setpoint" is a target alignment/norm regime, and the "controller" adjusts λ_t. Pure proportional (P) control creates oscillation; integral (I) eliminates steady-state error; derivative (D) reduces overshoot. PIDAO (Nature Communications 2024) applies PID structure to the gradient update direction, but no paper applies PID structure to the WD coefficient itself.

**Novelty estimate**: 8/10 — PIDAO applies PID to gradient updates; no paper applies PID to WD. The connection to alignment signals (δ_t as error signal for a PID WD controller) is genuinely novel.

---

## Phase 3: Self-Critique

### Against Candidate A (CCWD)

**Prior work attack**: CR-SAM already combines Hessian-trace regularization with training. SAM+L2 combinations have been studied. The specific claim of conditioning WD on Hessian trace is different — but reviewers will ask why not just use SAM+WD directly. Response needed: SAM modifies the update direction; CCWD modifies the WD coefficient — these operate on orthogonal dimensions and compose rather than overlap.

**Methodological attack**: Computing the Hessian trace at every step is O(d) via Hutchinson estimator (random probing), not free. This adds ~10-20% training overhead per step. Under AdamW (Phi-invariant regime), CCWD may still be washed out by the ℓ∞ constraint.

**Theoretical attack**: The causal chain "high sharpness → reduce WD → less displacement → better flat minima" is plausible but lacks formal proof. In nonconvex settings, reducing WD in sharp regions could delay escape from sharp bad minima rather than help.

**Scalability attack**: Hessian computation (even Hutchinson) is expensive for large models. AlphaDecay uses spectral analysis once per module, not per-step — CCWD would require per-step curvature estimation.

**Verdict**: MODERATE — core insight is sound, but compute overhead and potential redundancy with SAM-family methods are real weaknesses. Salvageable but not the front-runner.

---

### Against Candidate B (SPWD)

**Prior work attack**: AlphaDecay uses HT-SR spectral density for per-module WD — close in spirit. AdaDecay uses per-parameter gradient norms. ARSVD uses spectral entropy for adaptive rank selection in compression. None use the rate of change of rank (rank velocity) as a feedback signal. This specific novelty appears to hold.

**Methodological attack**: Computing effective rank (stable rank = ‖W‖_F²/‖W‖₂²) per layer per epoch is cheap — O(layer size), fully parallelizable. Main risk: rank velocity is a noisy signal, especially early in training when rank fluctuates. Need EMA smoothing of rank velocity to avoid chattering.

**Theoretical attack**: The phase-transition analogy from condensed matter physics is suggestive but imprecise. Weight matrix rank in neural networks does not exhibit a true thermodynamic phase transition — it is a gradual process with noisy dynamics. The "critical point" interpretation requires that rank velocity actually peaks near a transition, which is empirically verifiable but not theoretically guaranteed.

**Scalability attack**: Per-layer stable rank computation at every epoch (not every step) is computationally tractable. For ResNet-50 with 50 layers, this is feasible. For a 70B LLM, full SVD per layer is too expensive, but approximate methods are standard.

**Verdict**: STRONG — the core idea is genuinely novel, the compute cost is manageable, and the prior work gap is confirmed. Main risk is the noisiness of rank velocity, mitigated by smoothing.

---

### Against Candidate C (PID-WD)

**Prior work attack**: PIDAO applies PID to the gradient update. CWD uses a binary alignment mask. Defazio (2025) notes WD is a proportional controller on ‖g‖/‖w‖. No paper applies continuous PID structure to the WD coefficient conditioned on alignment. Gap confirmed.

**Methodological attack**: The PID rule has three new hyperparameters (λ_P, λ_I, λ_D). Tuning a PID controller is notoriously difficult in practice — no neural-network analog of Ziegler-Nichols exists. The integral term accumulates over T steps, potentially causing wind-up (integral saturation) if the alignment signal is persistently negative. Standard fix: integral clipping and anti-windup.

**Theoretical attack**: The analogy to PID-controlled linear systems breaks down severely in nonconvex settings. The "plant" (optimization trajectory) is not linear, and stability theorems for PID apply to LTI systems. The formal convergence theory for PID-WD in nonconvex SGD would require significant new mathematical machinery.

**Scalability attack**: δ_t = cos(g_t, w_t) requires a dot product between gradient and weight vector — O(d) per step, manageable. The integral is a running sum, O(1) to update.

**Verdict**: MODERATE — genuinely novel and computationally efficient, but hyperparameter tuning challenge and lack of formal theory are real weaknesses. Salvageable with simplified P+I formulation.

---

## Phase 4: Refinement

### Dropped Ideas

**Candidate A (CCWD)** is demoted to secondary. The Hessian computation overhead under AdamW (in the Phi-invariant regime) makes it an expensive solution to a problem that may be structurally blocked. Better suited as a SAM-family extension paper. Dropped as front-runner.

### Strengthened Ideas

**Candidate B (SPWD)**: Strengthened with three refinements:
1. *Rank velocity computation*: Use stable rank r(W) = ‖W‖_F²/‖W‖₂² (computable without full SVD — ‖W‖₂ approximated via power iteration, 1-2 steps per epoch) rather than numerical rank. This reduces per-epoch computation from O(d²) to O(d·k) for k power iteration steps.
2. *Smoothing*: Use EMA with momentum β=0.9 for rank velocity to suppress noise: v_t = β·v_{t-1} + (1-β)·(r_t - r_{t-1}).
3. *Coupling to Phi framework*: Define SPWD as λ_t = λ_0 · φ_rank(v_t) where φ_rank(v) = 1 + α·tanh(-v) (increase WD when rank is rapidly decreasing, reduce when rank has stabilized). This makes SPWD a new Phi modulator in the existing framework, directly comparable to CWD, SWD, and cosine_schedule.

Additional evidence found for SPWD: ARSVD paper (arXiv:2504.20078) validates spectral rank signals are computationally tractable and information-rich. Truong & Truong (arXiv:2603.07323) shows WD traverses norm hierarchy with logarithmic transition delays, providing direct theoretical support for the phase-transition character of WD dynamics.

**Candidate C (PID-WD)**: Strengthened with three refinements:
1. *Simplify to P+I*: Drop derivative term (most fragile), keeping: λ_t = λ_P·(1-δ_t) + λ_I·μ_t where μ_t = β·μ_{t-1} + (1-β)·(1-δ_t). Only 2 hyperparameters.
2. *Anti-windup via saturation*: Clip μ_t to [0, 1] at every step.
3. *Connect to Phi framework*: Define φ_PID = normalized(λ_P·(1-δ_t) + λ_I·μ_t). Makes PID-WD another Phi modulator, directly testable within existing benchmark infrastructure.

### Additional Evidence Found

From PID control search (PIDAO, Nature Communications 2024): PID structure applied to gradient updates yields convergence improvements on CIFAR/ImageNet. Our application of PID to the WD coefficient itself is the WD-dimension analog. The paper explicitly notes: "applying feedback control to the optimizers may provide another perspective for exploring more robust, accurate, and explainable optimization algorithms."

From the continual learning literature (Nature 2024, loss of plasticity): WD in continual learning prevents catastrophic forgetting but causes plasticity loss. This directly supports SPWD's core motivation. The counterintuitive prediction: SPWD prescribes high WD when rank is rapidly decreasing (to accelerate the beneficial transition) and low WD when rank has stabilized (to avoid over-regularizing a settled representation). This runs against the intuitive "anneal WD near the end" heuristic.

### Selected Front-Runner

**Front-runner: Candidate B — Spectral-Phase-Transition WD Scheduler (SPWD)**

Reasoning:
1. Genuine novelty confirmed: No paper uses rank velocity as a WD feedback signal. The gap is real.
2. Fits the Phi framework: SPWD is a new Phi modulator φ_rank(v_t), directly comparable in our benchmark.
3. No additional hyperparameters beyond α (a single exponent). Comparable simplicity to CWD.
4. Addresses Phi invariance: In the SGD regime (where Phi invariance fails), SPWD should show the strongest effect because SGD's rank minimization dynamics are stronger. This directly leverages our experimental finding that SGD shows 18.3× more WD sensitivity than AdamW.
5. Testable and falsifiable: The core prediction — that WD should be highest during rapid rank collapse phases — is directly verifiable by tracking rank velocity and WD values together.

Secondary front-runner: Candidate C (PID-WD) — retained as a complementary idea with stronger theoretical elegance but more tuning complexity.

---

## Phase 5: Final Proposal

### Title
Spectral-Phase-Transition Weight Decay: Closing the Loop Between Rank Dynamics and Regularization Strength

### Hypothesis
A WD scheduler conditioned on the rate of change of per-layer stable rank (rank velocity v_t = d/dt [‖W‖_F²/‖W‖₂²]) will outperform all time-based and alignment-based WD schedules on SGD-trained networks, and will produce qualitatively different rank collapse trajectories. Specifically:
- H1: SPWD will achieve higher test accuracy than constant WD, cosine_schedule, CWD, and SWD on CIFAR-10/100 with ResNet-20 using SGD (which is in the Phi-non-invariant regime).
- H2: SPWD will produce faster rank collapse to a lower effective rank than constant WD, accompanied by higher generalization.
- H3: The timing of WD peaks in SPWD will correlate with local maxima of |v_t| (rank velocity peaks) — if this correlation is absent, the mechanism claim is falsified.

### Motivation

Weight decay is the primary driver of low-rank bias in deep networks (Galanti et al. 2022, Yunis et al. 2024). The current approach treats this rank minimization as a side effect to be managed, not a signal to be exploited. The Truong & Truong (2026) finding that WD traverses norm hierarchy in logarithmic-delay phase transitions reveals that WD's effect on representation structure is genuinely phase-transition-like — abrupt, threshold-dependent, and history-dependent.

Meanwhile, all existing WD scheduling methods (SWD, cosine, CWD) condition WD on either gradient norms or gradient-weight angles — signals about the current optimization step, not about the representational state of the network. SPWD is the first method that conditions WD on the structural state (spectral rank) rather than the dynamics state (gradient), closing a conceptual gap that has been open since Galanti et al. (2022).

The experimental finding from our Iterations 1–4 that AdamW neutralizes all WD scheduling strategies provides the key target regime: SPWD is designed for SGD (where rank minimization dynamics are stronger and Phi invariance fails), exactly where our experiments show the most differentiation between WD strategies.

### Method

**Algorithm: SPWD-SGD**

```
Inputs: λ_0 (base WD), α (responsiveness ∈ [0, 1]), β (EMA smoothing = 0.9), ε (numerical stability)

Initialize: r_0^(l) = stable_rank(W_0^(l)) for each layer l; v_0 = 0

For each training step t:
  1. Compute gradient g_t
  2. (Per epoch) Compute per-layer stable rank:
       r_t^(l) = ||W_t^(l)||_F^2 / (power_iteration(W_t^(l), 2_iters)^2 + ε)
  3. (Per epoch) Compute global rank velocity:
       dr_t = mean_l(r_t^(l) - r_{t-1}^(l))
  4. (Per epoch) Update EMA rank velocity:
       v_t = β * v_{t-1} + (1-β) * dr_t
  5. Compute Phi modulator:
       φ_t = 1 + α * tanh(-v_t)
       [When v_t < 0 (rank decreasing): φ_t > 1: increase WD to accelerate rank collapse]
       [When v_t > 0 (rank increasing): φ_t < 1: reduce WD to prevent over-compression]
  6. Apply weight decay:
       w_{t+1} = (1 - φ_t * λ_0 * γ_t) * w_t - γ_t * g_t
       [where γ_t is the learning rate schedule]
```

Key design choices:
- `tanh(-v_t)`: smooth, bounded response to rank velocity; saturates at ±1 to prevent instability
- `1 + α * tanh(-v_t)`: keeps φ_t positive and close to 1 for α ≤ 1; base behavior is constant WD when α=0
- Per-layer stable rank: computable with 1-2 power iteration steps per epoch (amortized cost, not per-step)
- Decoupled update: WD coefficient is updated per-epoch (using previous epoch's rank), not per-step

Implementation details:
- Stable rank: ‖W‖_F² (sum of squares of all weights, O(d)) divided by ‖W‖₂² (2-step power iteration, O(d·k))
- Total overhead: ~5% additional computation per epoch
- Fits directly into the existing `WDModulator` abstract class from our benchmark infrastructure

### Experimental Plan

**Phase 1: SGD-regime validation (primary)**

| Experiment | Dataset | Model | Optimizer | Seeds | Epochs | GPU-time |
|---|---|---|---|---|---|---|
| SPWD vs baselines | CIFAR-10 | ResNet-20 | SGD | 42,123,456 | 200 | ~3 × 1hr |
| SPWD vs baselines | CIFAR-100 | ResNet-20 | SGD | 42,123,456 | 200 | ~3 × 1hr |
| SPWD vs baselines | CIFAR-10 | VGG-16-BN | SGD | 42,123,456 | 200 | ~3 × 2hr |

Baselines: constant, cosine_schedule, CWD, SWD, no_wd, half_lambda (all already run in this project)

Primary metrics:
- Best test accuracy (mean ± std over 3 seeds)
- Effective rank per layer at end of training
- Rank velocity trajectory (v_t per epoch)
- WD multiplier trajectory (φ_t per epoch)
- Correlation between |v_t| peaks and φ_t peaks (mechanistic validation)

Falsification criteria:
- If SPWD does not outperform constant WD on SGD (p < 0.05), H1 is falsified
- If rank velocity and WD multiplier are uncorrelated (r < 0.3), H3 is falsified

**Phase 2: AdamW-regime comparison (secondary)**

Run SPWD with AdamW on CIFAR-10/100 to compare with Phi-invariant results.
Expected: SPWD should NOT outperform constant WD under AdamW (consistent with Phi invariance).
This would confirm SPWD's benefit is specific to SGD, providing clean mechanistic evidence.

**Phase 3: ImageNet extension (optional, aligned with project roadmap)**

SPWD-SGD on ResNet-50 / ImageNet (90 epochs) — 3 seeds.
Expected: SPWD should show larger WD differentiation than constant WD (extending H1 to larger scale).

### Resource Estimate

- Phase 1: ~18 GPU-hours (6 experiments × 3 seeds × ~1hr each)
- Phase 2: ~6 GPU-hours (2 experiments × 3 seeds × ~1hr each)
- Phase 3 (optional): ~9 GPU-hours (1 experiment × 3 seeds × ~3hr each)
- Total: ~24-33 GPU-hours
- Wall-clock: ~3-4 hours with 8x RTX PRO 6000 (8 parallel runs)
- Code changes: ~50-100 LOC to implement SPWD modulator, reusing all existing infrastructure

### Risk Assessment

**Risk R1: Stable rank computation is too noisy for reliable feedback signal (probability: 35%)**
- Mitigation: Use EMA with β=0.9 to smooth rank velocity; if noise remains, switch to epoch-level updates
- Fallback: Use 5-epoch rolling average of rank change instead of EMA

**Risk R2: SPWD shows no benefit over constant WD even on SGD (probability: 30%)**
- Mitigation: This is a falsifiable negative result — it reveals that rank velocity is not an actionable signal for WD decisions, which itself is a contribution to the unified framework
- Fallback: Re-test SPWD at higher ρ values (Regime III in the Phi trichotomy), where rank dynamics are more pronounced

**Risk R3: SPWD is numerically unstable with small λ_0 (probability: 15%)**
- Mitigation: Hard-clip φ_t to [0.1, 10] to prevent extreme modulation; use α ≤ 0.5 in initial experiments

### Novelty Claim

SPWD is novel on three grounds, each with literature evidence:

1. **First WD scheduler conditioned on structural state (spectral rank) rather than dynamics state (gradient/alignment)**: AlphaDecay uses spectral density statically; our method uses rank velocity dynamically. SWD uses gradient norms; CWD uses gradient-weight angle. All existing dynamic WD methods use gradient-space signals. SPWD uniquely uses a weight-space structural signal.

2. **First WD method that deliberately exploits the phase-transition character of rank minimization**: Truong & Truong (2026) identify logarithmic-delay transitions in WD's effect on representations; Galanti et al. (2022) identify WD as a rank regularizer. No method connects these findings to design a phase-transition-aware WD controller.

3. **First WD method that predicts a counterintuitive schedule**: SPWD prescribes maximal WD during rapid rank collapse (when representation structure is most volatile) and minimal WD when rank has stabilized — the opposite of conventional wisdom that recommends reducing WD as training converges. This is a strong, falsifiable claim that distinguishes SPWD from any trivially ad hoc schedule.

---

## Addendum: Secondary Proposal — PID-Weight-Decay (PID-WD)

### Title
PID-Controlled Weight Decay: Proportional-Integral Alignment Feedback for Nonconvex SGD

### Hypothesis
A WD controller with integral history of gradient-weight misalignment (μ_t = EMA of (1-δ_t)) will outperform purely proportional alignment-based methods (CWD) by addressing accumulated misalignment debt that instantaneous controllers ignore.

### Method (simplified P+I formulation)
```
λ_t = λ_P · (1-δ_t) + λ_I · μ_t
where μ_t = β · μ_{t-1} + (1-β) · (1-δ_t), β = 0.9 (anti-windup: clip μ_t to [0, 1])

Phi modulator: φ_PID = (λ_P · (1-δ_t) + λ_I · μ_t) / (λ_P + λ_I)
```

### Why this is different from CWD
CWD is a bang-bang controller (δ_t > 0 → decay on, δ_t < 0 → decay off). PID-WD is a continuous proportional-integral controller with memory. The integral term prevents the "WD holiday" problem where CWD reduces WD exactly when the optimizer is making good progress (high positive alignment), even if that progress has been preceded by extended misalignment.

### Experimental plan
- Same setup as SPWD Phase 1 (SGD/CIFAR/ResNet-20)
- Compare PID-WD vs CWD vs constant: does adding integral history improve CWD?
- Key diagnostic: plot μ_t trajectory alongside δ_t — does integral term activate when δ_t has been persistently low?
- Resource: ~6 GPU-hours (2 CIFAR experiments × 3 seeds × ~1hr each), shares all infrastructure with SPWD

---

*This perspective was generated by the Sibyl Innovator Agent (Iteration 6). Core innovative direction: exploit the phase-transition character of WD-driven rank minimization (Galanti 2022, Yunis 2024, Truong & Truong 2026) to create a spectral-feedback WD controller (SPWD) that operates in the Phi-non-invariant SGD regime identified by our own experiments in Iterations 1–4.*

Sources consulted:
- [FAdam (arXiv:2405.12807)](https://arxiv.org/html/2405.12807v5)
- [Sparse Training via Weight Symmetry (Calgary ML, ICML 2025)](https://www.calgaryml.com/blog/2025/sparse-rebasin/)
- [Sign-In to the Lottery (arXiv:2504.12801)](https://arxiv.org/html/2504.12801v1)
- [CR-SAM (arXiv:2312.13555)](https://arxiv.org/html/2312.13555v2)
- [Flatness-aware SGLD (arXiv:2510.02174)](https://arxiv.org/html/2510.02174v1)
- [Sassha (arXiv:2502.18153)](https://arxiv.org/html/2502.18153)
- [Curvature Rate λ (arXiv:2511.01438)](https://arxiv.org/html/2511.01438)
- [PID Optimizer PIDAO (Nature Comm.)](https://www.nature.com/articles/s41467-024-54451-3)
- [Spectral Dynamics of Weights (arXiv:2408.11804)](https://arxiv.org/html/2408.11804v1)
- [SGD Minimizes Rank (arXiv:2206.05794)](https://arxiv.org/pdf/2206.05794)
- [Low-Rank Approximation ARSVD (arXiv:2504.20078)](https://arxiv.org/html/2504.20078v2)
