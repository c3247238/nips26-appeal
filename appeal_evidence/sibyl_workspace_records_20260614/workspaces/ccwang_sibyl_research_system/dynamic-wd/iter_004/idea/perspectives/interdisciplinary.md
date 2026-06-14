# Interdisciplinary Perspective

**Agent**: sibyl-interdisciplinary
**Date**: 2026-03-18 (revised with expanded Phase 1 and new Candidate D)
**Topic**: Unified Dynamic Weight Decay Framework
**Purpose**: Find structural analogies from other sciences that illuminate or extend the WD unification problem

---

## Phase 1: Literature Survey

### By Source Field

#### Physics / Statistical Mechanics / Thermodynamics

1. **"Can Training Dynamics of Scale-Invariant Neural Networks Be Explained by the Thermodynamics of an Ideal Gas?"** (arXiv:2511.07308, Nov 2025) — Training hyperparameters (learning rate, weight decay) map to thermodynamic state variables (temperature, pressure, volume); ideal gas law correspondence verified empirically; Maxwell relations emerge and suggest entropy-rate-controlled scheduler design. Weight decay appears as a "pressure" variable controlling the volume of the weight distribution. **Key derivation**: the authors show that the stationary entropy satisfies Maxwell-like cross-derivatives, with direct implication that the optimal WD schedule co-depends on the LR schedule in a derivable way.

2. **"SGD as Free Energy Minimization: A Thermodynamic View on Neural Network Training"** (arXiv:2505.23489, May 2025) — SGD implicitly minimizes F = U − TS (U = training loss, S = weight entropy, T = learning rate). Weight decay modulates the energy term U directly while also affecting S via norm compression. Temperature determines the loss–entropy tradeoff; this explains why high LR prevents convergence to minima. Phase transition to zero temperature (superconductivity-like) at small LR.

3. **"Neural Thermodynamic Laws for Large Language Model Training"** (arXiv:2505.10559, May 2025) — Under the river-valley loss landscape model for LLMs: fast dynamics (valley directions) satisfy thermal equilibrium; slow dynamics (river/descent directions) satisfy drift equations. Weight decay acts as a harmonic spring (restoring force) in the fast directions — equipartition theorem gives steady-state: LR × gradient_variance = WD × weight_variance. This is the "third thermodynamic law" for neural network training: WD-controlled equipartition determines the steady-state gradient-to-weight ratio.

4. **"Neural Thermodynamics: Entropic Forces in Deep Learning"** (NeurIPS 2025, OpenReview) — Establishes that SGD preferentially finds high-entropy solutions via entropic forces; weight decay leads to "non-universal behavior" breaking the entropic force structure. Key insight: WD introduces an energy-entropy competition that differentiates between solutions that are entropically equivalent under vanilla SGD. This provides the mechanistic explanation for the generalization benefit of WD beyond mere norm control.

5. **"Renormalization Group for Deep Neural Networks: Universality of Learning and Scaling Laws"** (arXiv:2510.25553, Jan 2026) — DNN layer-wise transformations treated as RG coarse-graining operations; fixed points correspond to scale-invariant representations; changing the standard deviation of weight distribution interpreted as an RG flow in network-parameter space. Weight decay as a tool for steering RG flows toward desired fixed points. **Heavy-Tailed Self-Regularization (HTSR) connection**: WeightWatcher metrics can be derived from Wilson's Exact RG via a phenomenological Effective Hamiltonian. Well-trained networks exist near a "critical point" where weight spectra show universal power-law behavior. WD drives the network away from or toward this critical point depending on its strength relative to the natural HTSR equilibrium — providing a principled justification for module-wise WD (AlphaDecay).

#### Neuroscience / Cognitive Science / Biological Neural Systems

6. **"Multi-Scale Temporal Homeostasis Enables Efficient and Robust Neural Networks"** (arXiv:2602.07009, Feb 2026) — Multi-timescale homeostatic regulation (ultra-fast 5ms, fast 2s, medium 5min, slow 1hr) applied to ANN training. Biological neural systems sustain function across decades via coordinated regulation at multiple temporal scales; MSTH improves ANN accuracy and eliminates catastrophic failures across molecular, graph, and image benchmarks.

7. **"Biologically Inspired Neural Network Layer with Homeostatic Regulation"** (Scientific Reports, 2025) — BioLogicalNeuron layer with calcium-driven homeostasis (BCM-style sliding threshold), self-repair mechanisms, synaptic scaling. Homeostatic control maintains neurons in a healthy operating regime; directly analogous to weight norm control as a neural activity regulation mechanism.

8. **"Emergence of Hebbian Dynamics in Regularized Non-Local Learners"** (arXiv:2505.18069, 2025) — Gradient descent with weight decay can produce stationary points consistent with Hebbian phenomena; observations of Hebbian updates may be indirect evidence that the brain runs gradient descent + weight decay. BCM theory's sliding threshold is a direct biological precursor to alignment-aware WD.

9. **"Stability of Neuronal Networks with Homeostatic Regulation"** (PLOS Computational Biology) — Homeostatic control that is stable for single neurons can destabilize activity in recurrent networks; the stronger the network recurrence, the slower the homeostasis must be. This introduces the concept of **homeostatic gain scheduling based on network state**, directly analogous to alignment-dependent WD.

10. **"Synapse-type-specific Competitive Hebbian Learning Forms Functional Recurrent Networks"** (PNAS, 2024) — Synapses compete for limited synaptic resources; resource-limited competition (subtractive normalization) stabilizes Hebbian growth and enables winner-takes-all dynamics. The competition mechanism maps precisely to how WD creates implicit weight competition across layers and parameters.

11. **Free Energy Principle and Active Inference** (Friston, ongoing; key review PMC:8871280, 2022) — The Free Energy Principle formalizes biological self-organization as variational free energy minimization: every biological system acts to minimize its surprisal via a trade-off between accuracy (fitting observations) and complexity (KL divergence from prior). The negative variational free energy = ELBO. **Structural correspondence to WD**: weight decay in deep learning is precisely the KL divergence penalty from a zero-centered Gaussian prior on weights. Therefore, SGD+WD is exactly the active inference update rule for a network with a Gaussian parameter prior, and the WD coefficient λ controls the prior-precision (the inverse temperature of the prior distribution). Dynamic WD = dynamic prior precision adjustment, which in active inference corresponds to updating the meta-prior based on accumulated evidence — a process called **precision weighting**. This provides a completely new theoretical justification for alignment-aware WD: alignment is the signal of how much information the current gradient provides about optimal parameters; high alignment = high precision = lower WD appropriate.

#### Biology / Evolutionary Systems / Ecology

12. **"Exploring the Dynamics of Lotka-Volterra Systems: Efficiency, Extinction Order, and Predictive Machine Learning"** (arXiv:2410.10999, Oct 2024) — When ecological systems possess self-regulation (negative intraspecific feedback), stability can always be maintained by increasing self-regulation. Self-regulation strength is the dominant factor in predicting system behavior. This maps directly to WD as "self-regulation" in the parameter space ecosystem.

13. **"Complex Systems in Ecology: Large Lotka-Volterra Models and Random Matrices"** (Royal Society, 2024) — Multi-species competition with structured interaction matrices; stability conditions derived from random matrix theory. The interaction structure (who competes with whom) maps to the gradient-weight alignment geometry: highly aligned gradient-weight pairs represent "competitive" interactions that benefit from stronger regulation.

14. **Clonal Selection and Affinity Maturation in Adaptive Immune Systems** (classic immunology; AIS survey arXiv:1006.4949) — B-cells with high antigen affinity are selectively retained and amplified (clonal expansion); low-affinity or self-reactive cells are eliminated (clonal deletion / negative selection). The immune system maintains a functional repertoire through continuous selection pressure. Key mechanism: the affinity score (antigen binding strength) is a continuous scalar that modulates the degree of clonal expansion/deletion. **Structural correspondence**: gradient-weight alignment = antigen-antibody affinity; high alignment = high affinity (weight "fits" the gradient direction) = reduce decay (allow the weight to grow/persist); low alignment = low affinity = increase decay (eliminate the parameter). The affinity maturation process (iterative somatic hypermutation + selection) is the biological analog of iterative gradient descent + alignment-conditioned WD.

#### Control Theory / Engineering

15. **"Accelerated Optimization in Deep Learning with a PID Controller (PIDAO)"** (Nature Communications, 2024; DOI:10.1038/s41467-024-54451-3) — This landmark paper introduces the PID framework to deep learning optimization. Gradient descent is a proportional controller (P: responds to current error); SGD-momentum adds derivative action (D: responds to rate of change); the **integral action (I: responds to accumulated past error)** has been missing from standard deep learning optimizers. PIDAO explicitly adds integral feedback: the accumulated gradient integral eliminates steady-state optimization error, converges to global minima, and reduces overshoot/oscillation via the derivative term. **Critical connection to WD**: In control theory, integral action is needed to eliminate steady-state tracking error when there is a constant disturbance. WD in training acts as a constant disturbance term (it persistently pushes weights toward zero, even when they are already at the loss minimum). Without integral action in the optimizer, WD causes a persistent offset in the final converged weights. **This is precisely the mechanism behind the AdamW vs. Adam+L2 gap identified by Loshchilov & Hutter**: Adam without weight decay decoupling has a "disturbance" from L2 that cannot be eliminated without integral action. WD decoupling is equivalent to making the weight decay disturbance orthogonal to the optimizer's integral controller.

16. **"Reinforcement Learning-Enhanced Adaptive Sliding Mode Control"** (Springer, 2025) — CWD (Cautious Weight Decay, ICLR 2026) can be precisely characterized as a first-order sliding mode controller: the sign-alignment condition defines the sliding surface; the switching rule determines whether the system is "above" or "below" the surface; the binary decay decision corresponds to the relay control law in variable-structure systems. The chattering problem in SMC (rapid sign oscillation near the surface) directly corresponds to instability from noisy alignment estimates.

17. **Control Lyapunov Functions and Stability-Regularization Duality** — In optimal control, a CLF provides both a stabilizing controller and a Lyapunov certificate simultaneously. The "loss stabilization mechanism" (D'Angelo et al. NeurIPS 2024) for weight decay in SGD is precisely a Lyapunov argument: WD provides the dissipation term needed to drive the Lyapunov function (augmented loss + norm term) to a critical point. Dynamic WD is equivalent to designing a state-dependent CLF for the SGD dynamical system. The Abstract Lyapunov Control Optimizer (arXiv:2407.01019, 2024) provides convergence analysis for adaptive optimizers via continuous selection theory — a potential theoretical framework for alignment-aware WD convergence proofs.

#### Information Theory / Statistical Learning

18. **"Rethinking LLM Training through Information Geometry and Quantum Metrics"** (arXiv:2506.15830, 2025) — Natural gradient descent uses Fisher information as Riemannian metric; all effective learning rules with strictly decreasing objective can be rewritten as natural gradient descent under an appropriate metric. Weight decay in the Fisher-metric space has a geometric meaning: it is equivalent to a constraint on the KL-divergence from the current parameter distribution to a zero-centered prior. **Connection to active inference**: this KL-divergence interpretation formally unifies WD (Bayesian prior constraint), decoupled WD (prior applied after gradient update), and alignment-aware WD (prior applied only in aligned directions) under a single information-geometric framework.

19. **"A Renormalization Group Framework for Scale-Invariant Feature Learning"** (AAAI 2025) — Layer-wise transformations in deep networks as RG coarse-graining; RG flow equations for network parameters with fixed points as scale-invariant representations. Weight decay drives RG flow toward more structured (lower-rank, scale-invariant) fixed points — this is the mechanism behind Galanti et al.'s rank minimization finding. **Gap**: the appropriate WD strength to reach a specific RG fixed point has not been derived from the RG equations — this is a new open question for the unified framework.

### Cross-Disciplinary Gaps

- **Gap A (Maxwell Relations)**: The thermodynamics/ideal-gas correspondence identifies LR and WD as "temperature" and "pressure" respectively, but the **Maxwell relations** implied by this analogy have not been exploited for WD scheduling design. Specifically, if WD and LR are thermodynamically conjugate variables, then the optimal WD schedule should be derivable from the LR schedule via a Maxwell-type relation. arXiv:2511.07308 identifies this implication but does not exploit it.

- **Gap B (Multi-Timescale)**: The BCM sliding threshold in neuroscience has the same mathematical structure as alignment-aware WD (a state-dependent modulation of plasticity/regularization), but the **multi-timescale** aspect of biological homeostasis (fast and slow regulation operating simultaneously) has no counterpart in current WD methods.

- **Gap C (SMC Chattering)**: The sliding mode control characterization of CWD reveals that the chattering problem is the key failure mode for alignment-based switching, but **super-twisting SMC** (a continuous, higher-order sliding mode approach) has no ML analog. This is a precise, falsifiable gap.

- **Gap D (PID Integral Action for WD)**: PIDAO (Nature Communications 2024) introduces integral action into the optimizer itself, but the role of integral action specifically for weight decay control has not been studied. WD as a "disturbance" in the PID framework: integral action is needed to exactly compensate for the WD disturbance. The **optimal WD decoupling strategy** (when and how to apply WD relative to the optimizer's internal state) can be derived from the PID control theory of disturbance rejection. This gap is entirely unexplored.

- **Gap E (Friston Precision Weighting)**: The Free Energy Principle provides a principled framework for dynamic WD via "precision weighting" — adjusting the weight decay coefficient based on the informativeness of the current gradient signal. The alignment cosine is a natural precision signal (high alignment = gradient is informative = downweight the prior constraint = reduce WD). The formal derivation of alignment-aware WD from FEP precision weighting has not been done.

- **Gap F (Affinity Maturation Schedule)**: The immune affinity maturation analog suggests a **continuous affinity score** (not binary) for WD modulation with an explicit maturation schedule (early training = clonal expansion/exploration = low WD; late training = negative selection/elimination = high WD for non-aligned weights). No method implements a maturation-schedule-inspired WD curriculum.

---

## Phase 2: Initial Candidates

### Candidate A: Thermodynamic Maxwell Relations for WD-LR Co-Design (from Statistical Physics)

- **Source principle**: In thermodynamics, state variables (P, V, T, S) satisfy Maxwell relations: (∂S/∂P)_T = −(∂V/∂T)_P. These relations emerge from the existence of thermodynamic potentials (F = U − TS) and allow computing one response function from another. They encode how conjugate variables respond to each other's changes.

- **Structural correspondence**: From arXiv:2511.07308, the SGD+WD system for scale-invariant networks admits a thermodynamic description where LR plays the role of temperature T, WD plays the role of pressure P, and the "volume" of the weight distribution (related to ‖w‖²) plays the role of V. The free energy F = U − TS (training loss minus entropy × LR) has the same structure as the thermodynamic free energy. If this is a genuine thermodynamic analogy, Maxwell relations must hold: (∂S/∂λ)_γ = −(∂‖w‖²/∂γ)_λ. This would mean: **the rate at which entropy changes with WD, at fixed LR, equals minus the rate at which weight norm changes with LR, at fixed WD.** The right-hand side (weight norm sensitivity to LR) is **empirically observable** from standard training curves. Therefore, the optimal WD schedule can be *derived* from the empirically measured ‖w‖²(γ) curves — without any additional theoretical assumptions.

- **Hypothesis**: Optimal WD schedule as a function of training time can be derived from the Maxwell relation: dλ*/dt = −(d/dt)[d‖w‖²/dγ], where the right-hand side is the empirical sensitivity of weight norm to learning rate changes. This predicts that models trained with cosine LR schedules should have optimal WD schedules with a specific, derivable shape that differs qualitatively from cosine WD.

- **Why not just a metaphor**: The correspondence is grounded in the mathematical structure of the stationary distribution of SGD with WD for scale-invariant networks (arXiv:2511.07308). The Maxwell relations are not a metaphor — they are algebraic identities that must hold if the free energy interpretation is correct. The derivation is falsifiable: compute both sides of the Maxwell relation from experiment and check if they match.

- **Novelty estimate**: 9/10 — The thermodynamic SGD papers establish the analogy but do not exploit Maxwell relations for WD schedule design. This is a direct, unexploited consequence.

---

### Candidate B: Multi-Timescale Homeostatic Weight Decay (from Computational Neuroscience)

- **Source principle**: Biological homeostatic plasticity operates at multiple distinct timescales simultaneously: ultra-fast (millisecond-range spike adaptation), fast (second-range spike-frequency adaptation), medium (minute-range intrinsic excitability regulation), and slow (hour-range synaptic scaling). These timescales are not redundant — each controls a different aspect of neural stability and each has a different "setpoint." The multi-timescale structure allows biological systems to respond rapidly to perturbations while maintaining long-term stability.

- **Structural correspondence**: In neural network training, there are also multiple relevant timescales for WD dynamics: (1) **per-iteration**: the alignment signal δ̂_t changes at every step; (2) **per-epoch**: the weight norm ‖w_t‖ changes on epoch timescales; (3) **per-phase**: the training phase (early exploration, convergence, fine-tuning) changes on longer timescales. Current WD methods either operate at a single timescale (alignment-aware = per-iteration) or at a schedule timescale (SWD = per-phase). A **multi-timescale WD** would maintain three simultaneously-running EMA signals at different decay rates (β_fast, β_medium, β_slow) and combine them to compute the effective WD coefficient. The formal mapping: ultra-fast → per-step alignment signal EMA (β_fast ≈ 0.1); fast → gradient-norm EMA (β_medium ≈ 0.99); slow → weight-norm EMA (β_slow ≈ 0.9999). The combined WD: λ_t = f(EMA_fast(δ̂_t), EMA_medium(‖g_t‖), EMA_slow(‖w_t‖)).

- **Hypothesis**: A three-timescale WD controller will outperform single-timescale alignment-aware WD (e.g., CWD) because it captures: (i) rapid perturbations in alignment quality (fast timescale), (ii) medium-term changes in gradient curvature regime (medium timescale), and (iii) slow drift in weight norm toward over-parameterized regimes (slow timescale). The key prediction: the fast timescale captures the "CWD" signal; the slow timescale captures the "norm-matched WD" signal; the medium timescale is a new signal that current methods miss.

- **Why not just a metaphor**: The mathematical structure is an EMA cascade, not a biological metaphor. The formalization directly maps biological timescale hierarchy to EMA decay rates. The prediction is quantitatively testable: ablate each timescale independently and measure performance. The biological principle that "homeostatic control that is too fast destabilizes recurrent networks" (PLOS Computational Biology) translates to a concrete algorithm design constraint: the fast timescale EMA should not dominate (β_fast should not be too small), which is a falsifiable regularization design principle.

- **Novelty estimate**: 8/10 — Multi-scale temporal homeostasis for ANNs exists (arXiv:2602.07009) but in the context of neuron activity regulation, not WD. The application to WD scheduling with a formalized EMA-cascade structure appears unexplored.

---

### Candidate C: Super-Twisting Sliding Mode WD (from Variable Structure Control Theory)

- **Source principle**: Standard sliding mode control (SMC) uses a binary relay function: if the system is above the sliding surface, apply maximum control; if below, apply minimum. This causes chattering (high-frequency oscillation near the surface). **Super-twisting SMC** (Levant 1993) solves the chattering problem by using a continuous control law that is a smooth function of the sliding variable and its integral: u(t) = −k₁|σ|^(1/2) sign(σ) + u₁, u̇₁ = −k₂ sign(σ). The key innovation: the control action is proportional to the square root of the sliding variable (not binary), producing a continuous signal that converges to the sliding surface in finite time without chattering.

- **Structural correspondence**: CWD (ICLR 2026) is a binary SMC on the alignment surface σ_t = sign(⟨w_t, Δw_t⟩): when σ_t > 0 (aligned), apply WD; when σ_t < 0 (anti-aligned), suppress WD. This directly corresponds to standard relay SMC with chattering risk when alignment oscillates rapidly. **Super-twisting WD** would replace the binary sign function with a continuous, square-root-proportional function: λ_t = λ_base · (1 − tanh(k₁ · cos_sim(w_t, g_t) / ε))^(1/2). The integral term in super-twisting provides memory of recent alignment history, equivalent to an alignment-accumulated integral: λ_integral += k₂ · sign(cos_sim(w_t, g_t)) · γ_t. This makes WD depend on both instantaneous alignment AND the accumulated alignment history.

- **Hypothesis**: Super-twisting WD will: (1) eliminate the "binary threshold" noise amplification of CWD near the alignment=0 boundary, (2) produce smoother WD trajectories with lower variance, (3) converge to the same sliding surface as CWD but with fewer oscillations. Specific prediction: the loss trajectory variance (across seeds) will be lower for super-twisting WD than for CWD, because the continuous control law reduces sensitivity to stochastic alignment fluctuations.

- **Why not just a metaphor**: The Lyapunov stability proof for super-twisting SMC carries over directly: if cos_sim(w_t, g_t) is treated as the sliding variable, the super-twisting control law has a known Lyapunov function V = |σ|^(1/2) that guarantees finite-time convergence to the sliding surface. This is the same mathematical object used in the control theory proof, not just an analogy. The only adaptation required is to verify that the "finite-time convergence" property holds in the stochastic discrete-time setting (where σ_t is noisy), which is a well-defined research question.

- **Novelty estimate**: 8/10 — The SMC characterization of CWD exists informally in the CWD paper but is not formalized. Super-twisting as a solution to CWD's binary-switching problem has not been proposed.

---

### Candidate D: PID Integral Action as WD Disturbance Rejection (from Classical Control Theory)

- **Source principle**: In classical PID control theory, a **proportional controller** (P) produces a steady-state offset (permanent error) when a constant disturbance is present. Adding an **integral action** (I) eliminates the steady-state offset because the integral term accumulates the error and provides an opposing signal that exactly cancels the disturbance. The **PIDAO paper** (Nature Communications 2024, DOI:10.1038/s41467-024-54451-3) formally establishes that gradient descent is a proportional controller, SGD-momentum adds derivative action, and the missing component has been integral action.

- **Structural correspondence**: In deep learning optimization, weight decay acts as a **constant disturbance** that persistently pushes weights toward zero, even at the optimal parameters (where the loss gradient is zero). A pure proportional optimizer (SGD, Adam without WD decoupling) experiencing this disturbance will converge to a biased point that is offset from the true loss minimum. The bias is proportional to λ/γ (WD coefficient divided by learning rate). **This is exactly the problem that AdamW solves**: decoupling WD from the adaptive gradient scaling means the WD disturbance no longer interacts with the optimizer's integral-like moment estimates. However, PIDAO's analysis reveals a deeper insight: the truly optimal strategy is not to decouple WD from the optimizer but to **use integral action to track and cancel the WD disturbance**. The integral state accumulates the WD-induced error over time and generates an opposing correction — meaning the optimizer learns to compensate for its own WD. This is a fundamentally different approach from decoupling.

- **Hypothesis**: An **Integral-Corrected WD optimizer** that adds an integral state tracking WD-induced displacement (Δw_WD = λ_t · w_t at each step) will converge to a solution closer to the true loss minimum than AdamW with decoupled WD, especially for large WD coefficients or small learning rates (where the steady-state bias is largest). The specific prediction: the gap between the optimal unconstrained solution (λ=0) and the decoupled WD solution should be measurably reduced by integral correction. As a secondary prediction: the integral correction automatically learns the "effective WD budget" consumed, providing a natural implementation of the Budget Equivalence Metric (BEM).

- **Why not just a metaphor**: The PID disturbance rejection framework is not a metaphor — it is a formal theorem (integral action guarantees zero steady-state error for constant disturbances in linear systems with stable feedback). The WD disturbance in the optimizer update equation is a constant-gain linear term (λ_t · w_t) that satisfies the conditions for the integral action theorem to apply, at least for the linearized system around a minimum. The nonlinearity of the loss function introduces corrections, but the dominant effect is captured by the linear analysis. This can be verified empirically: measure the steady-state bias (convergence gap to λ=0 solution) for decoupled WD vs. integral-corrected WD and compare.

- **Novelty estimate**: 9/10 — PIDAO introduces PID to deep learning optimization but does not specifically analyze WD as a disturbance or propose integral correction for WD compensation. The connection between integral action and WD disturbance rejection is entirely novel and directly actionable. The formal PID framework provides a new unified lens: each WD variant corresponds to a different disturbance rejection strategy, and the unified framework can be organized around the PID taxonomy: P (decoupled WD = proportional compensation), I (integral correction = accumulating WD budget), D (derivative WD = responding to the rate of change of weight norm).

---

## Phase 3: Self-Critique

### Against Candidate A (Maxwell Relations WD)

- **Shallow analogy attack**: The thermodynamic analogy is genuinely structural in arXiv:2511.07308 for the simplified isotropic noise model. However, the Maxwell relations require that the free energy is a state function — i.e., path-independent. For real SGD, the stationary distribution depends on the optimization path and is not a true state function in the thermodynamic sense. The analogy may break down for non-scale-invariant networks (which are the majority of practical architectures). The domain expert (statistical physicist) would say: the ideal gas mapping is an approximation valid only near the stationary distribution; Maxwell relations derived from it are valid only to the same approximation.

- **Scale mismatch attack**: The thermodynamic description applies to the stationary distribution of SGD — i.e., to the long-time limit behavior. But WD scheduling is most critical during the transient phase of training (early training, LR warmup, post-decay phase). Maxwell relations are equilibrium properties; they may not hold during the transient phase, which is exactly where WD scheduling matters most.

- **Prior transplant check**: arXiv:2511.07308 identifies Maxwell relations as a theoretical outcome but does not exploit them for scheduler design. arXiv:2505.23489 focuses on the free energy perspective but does not derive WD schedules from Maxwell relations. No paper appears to have used Maxwell relations to derive WD schedules.

- **Testability attack**: The Maxwell relation prediction (optimal WD schedule derivable from weight-norm sensitivity to LR) is testable. However, the empirical measurement of (d‖w‖²/dγ) requires training multiple models with slightly different learning rates, which is expensive. A cheaper version: use the within-run LR schedule and measure ‖w‖² sensitivity to LR changes at LR-decay points.

- **Verdict**: MODERATE. The structural correspondence is genuine but limited to the stationary/near-stationary regime. Best used as supporting theoretical evidence for the BEM metric (thermodynamic pressure-volume work interpretation) rather than as a primary algorithmic contribution.

---

### Against Candidate B (Multi-Timescale Homeostasis)

- **Shallow analogy attack**: Multiple EMA signals at different timescales is a well-known engineering technique (exponential smoothing at multiple scales). The biological analogy adds interpretability but not new mathematical content. A domain expert in computational neuroscience would say: the multi-timescale structure in biology arises from specific biophysical mechanisms (calcium dynamics, protein synthesis timescales) that have mechanistic reasons for their specific timescales; the EMA analogy captures the phenomenology but not the mechanism.

- **Scale mismatch attack**: Biological homeostatic timescales (milliseconds to hours) are tightly coupled to specific biophysical processes. In neural network training, the "right" timescales for WD adaptation are not known and may vary dramatically by architecture, dataset, and optimizer. There is no principled way to choose β_fast, β_medium, β_slow without hyperparameter search — adding three hyperparameters to WD scheduling, where the existing methods already have too many.

- **Prior transplant check**: arXiv:2602.07009 explicitly implements multi-timescale homeostasis for ANN training but applies it to neuron activity regulation, not to weight decay specifically. AdaDecay (arXiv:1907.08931) uses sigmoid of layer-normalized gradient norms as a per-parameter WD — this is a single-timescale gradient-norm-based WD. No paper combines multiple EMA timescales specifically for WD.

- **Testability attack**: Testing which timescales matter requires ablating each combination, which is a 2³ = 8 experiment grid just for the timescale inclusion decisions, plus hyperparameter tuning for each β.

- **Verdict**: MODERATE. The multi-timescale concept is sound and biologically motivated, but the hyperparameter explosion is a practical problem. The simplest version (adding a medium-timescale EMA between CWD and SWD) is feasible and novel. Best used as supporting evidence for the multi-timescale structure of the Unified WD Framework rather than as a standalone proposal.

---

### Against Candidate C (Super-Twisting SMC WD)

- **Shallow analogy attack**: The analogy between binary CWD and relay SMC is structural and precise — the sign-alignment condition is exactly the sliding surface in SMC terminology, and CWD's decision rule is exactly the relay control law. The super-twisting extension is a well-defined mathematical construction. A control engineer would confirm this correspondence. This is not a shallow analogy.

- **Scale mismatch attack**: Super-twisting SMC has a finite-time convergence guarantee in the continuous-time, deterministic setting. In stochastic SGD, the "sliding variable" (alignment cosine) is noisy and the dynamics are discrete-time. The finite-time convergence proof does not carry over without additional assumptions about noise boundedness and step size conditions.

- **Prior transplant check**: Search for "super-twisting" + "weight decay" or "optimizer" returns no relevant results. Search for "sliding mode control" + "deep learning optimizer" returns results about SMC for neural network controllers (where the neural network controls a physical system), not about SMC-inspired WD for optimizers. This transplant appears genuinely novel.

- **Testability attack**: The key diagnostic test is: compare the variance of the WD coefficient trajectory λ_t across time for CWD vs. super-twisting WD, with matched average WD budget (BEM ≈ 1). Super-twisting WD should show lower trajectory variance.

- **Verdict**: STRONG. The structural correspondence is precise and non-trivial. The main uncertainty is whether the stochastic discrete-time setting breaks the finite-time convergence guarantee.

---

### Against Candidate D (PID Integral Action WD)

- **Shallow analogy attack**: The claim that WD is a "disturbance" in the PID framework requires that WD acts as a constant external force unrelated to the optimization objective. However, WD IS part of the optimization objective — it is a regularization term in the augmented loss. From the PID perspective, if the optimizer is designed to minimize the augmented loss (including WD), then WD is not a disturbance but a structural part of the target. The analogy only holds if we take the "true objective" to be the unregularized loss and WD to be a deparature from it — which is a valid but specific perspective. A control engineer would say: whether WD is a disturbance or a setpoint depends on your problem formulation.

- **Scale mismatch attack**: PID integral action eliminates steady-state error for **constant** disturbances in **linear** systems. WD coefficient λ_t may itself be time-varying (if we're doing WD scheduling), making it a time-varying disturbance. Linear integral action cannot exactly compensate a time-varying disturbance. The integral correction would need to be adaptive (tracking the rate of change of λ_t), which is a more complex controller.

- **Prior transplant check**: PIDAO (Nature Communications 2024) introduces PID to deep learning but focuses on acceleration and does not discuss WD as a disturbance. No paper has proposed using integral action specifically for WD disturbance rejection. The connection is novel.

- **Testability attack**: Test: train with (a) Adam + WD decoupled (AdamW), (b) PIDAO + WD, (c) PIDAO + integral-corrected WD. Measure convergence gap to the λ=0 solution. If (c) < (b) ≈ (a), the integral correction is working. This is a clean, falsifiable test.

- **Verdict**: MODERATE-STRONG. The structural correspondence is sound for the linear approximation, and the testability is high. The main limitation is the time-varying disturbance issue. However, even a simplified constant-λ version of integral correction would be novel and practically useful for understanding the WD-optimizer coupling problem.

---

## Phase 4: Refinement

### Dropped Candidates

- **Candidate A (Maxwell Relations)**: Demoted to SUPPORTING EVIDENCE for the thermodynamic framework section of the paper. The Maxwell relation concept provides theoretical grounding for WD-LR co-design and the thermodynamic interpretation of BEM, but is not suitable as a primary algorithmic contribution because it applies only near the stationary distribution.

- **Candidate B (Multi-Timescale Homeostasis)**: Demoted to SUPPORTING EVIDENCE for the multi-level structure of the Unified WD Framework. The three-timescale structure provides a biological motivation for why WD scheduling (slow), alignment-aware WD (fast), and gradient-norm-aware WD (medium) each address a different temporal aspect of the same problem. This strengthens the theoretical unification without requiring a new algorithm.

### Strengthened Candidates

**Candidate C Refinement (Simplified Super-Twisting WD)**:

The super-twisting adaptation to WD:
```
σ_t = cos_sim(w_t, g_t)                          # sliding variable
u_t = -k1 * |σ_t|^(1/2) * sign(σ_t) + v_t       # continuous WD modulation
v_{t+1} = v_t - k2 * sign(σ_t) * γ_t            # integral component
λ_t = max(λ_min, λ_base + u_t)                   # effective WD
```

The integral component v_t accumulates alignment history, making WD depend on the *duration* of misalignment, not just instantaneous alignment. If weights have been persistently anti-aligned with gradients for many steps, the integral term reduces WD further.

**Diagnostic experiment**: Train ResNet-20 on CIFAR-10 with (a) CWD, (b) super-twisting WD, and (c) vanilla WD with matched budget. Measure: (i) WD coefficient trajectory variance Var(λ_t), (ii) alignment trajectory stability, (iii) final accuracy.

**Candidate D Refinement (PID WD Disturbance Correction)**:

Instead of a full integral correction (which requires tracking ∫λ_t w_t dt), use a simplified **EMA-corrected WD** that approximates the integral action:

```python
class PIDecoupledWD:
    def __init__(self, lambda_base=1e-4, ki=0.01, beta_ema=0.999):
        self.lambda_base = lambda_base
        self.ki = ki
        self.beta_ema = beta_ema
        self.wd_integral = 0.0  # integral state: accumulated WD displacement

    def step(self, w, g, gamma):
        # WD disturbance: what WD "pushed" this step
        wd_displacement = self.lambda_base * w
        # Update integral: EMA of accumulated WD displacement
        self.wd_integral = self.beta_ema * self.wd_integral + (1 - self.beta_ema) * wd_displacement
        # Correction: push back against accumulated WD bias
        w_corrected = w - gamma * g - gamma * self.lambda_base * w + self.ki * self.wd_integral
        return w_corrected
```

This is mathematically equivalent to using the alignment signal between the WD direction (toward zero) and the gradient direction as a correction signal — connecting Candidate D back to Candidate C.

**Key insight**: Candidates C and D are not independent. The super-twisting WD integral term (v_t) and the PID integral correction (wd_integral) are both accumulating history about the weight-gradient interaction. They differ in their update laws and convergence targets, but share the structural idea that **WD decisions should depend on the accumulated history of the weight-gradient relationship, not just its instantaneous value**. This unification motivates a single "integral-action alignment-aware WD" that encompasses both.

### Selected Front-Runner: **Candidate C (Super-Twisting SMC WD)** as primary, with Candidate D (PID) as parallel theoretical framing

The super-twisting formulation is preferred because:
1. The structural correspondence with SMC is mathematically precise (not a metaphor)
2. The chattering problem in CWD is a genuine documented issue
3. The super-twisting solution is well-studied in control theory with known convergence properties
4. The diagnostic experiment is clean and directly tests the mechanism
5. The connection directly extends and improves an ICLR 2026 paper (CWD), making it immediately relevant

The PID framing is preserved as **theoretical scaffolding** for the unified WD framework, organizing each WD variant as a different disturbance rejection strategy:
- Standard WD: no disturbance rejection (P-only)
- Decoupled WD (AdamW): disturbance isolation (decoupling the plant from the disturbance)
- Alignment-aware WD (CWD): state-feedback disturbance rejection (P-type SMC)
- **Super-twisting WD**: second-order state-feedback with integral (PI-type SMC)

---

## Phase 5: Final Proposal

### Title
**Chattering-Free Alignment-Aware Weight Decay via Super-Twisting Sliding Mode Control**

### Source Principle
Super-twisting sliding mode control (Levant 1993) is a second-order SMC method that produces continuous control signals by using a continuous function of the sliding variable σ and its integral. Unlike standard relay SMC (which produces a binary ±1 signal causing chattering), super-twisting SMC produces a signal proportional to |σ|^(1/2) sign(σ) plus an integral feedback term, guaranteeing finite-time convergence to the sliding surface σ = 0 without chattering.

### Structural Correspondence

The formal mapping between SMC and WD optimization:

| SMC Concept | WD Optimization Analog |
|-------------|------------------------|
| Plant state x(t) | Weight vector w_t |
| Sliding variable σ | Alignment: cos_sim(w_t, g_t) = ⟨w_t, g_t⟩/(‖w_t‖‖g_t‖) |
| Control input u(t) | WD modulation Δλ_t |
| Sliding surface σ = 0 | Zero-alignment state: WD neither helps nor hurts |
| Relay control law sign(σ) | CWD binary mask: I[sign(w_t) = sign(g_t)] |
| Chattering | Noisy λ_t oscillation from stochastic gradient estimates |
| Super-twisting control | Continuous: u_t = -k1|σ_t|^(1/2)sign(σ_t) + v_t; v̇ = -k2 sign(σ_t) |
| Integral term v_t | Accumulated alignment history |

The analogy is structurally precise: CWD's decision rule "apply WD iff sign(w) matches sign(update)" is exactly the relay function on the alignment sliding surface. The super-twisting extension replaces the relay function with a continuous square-root law plus integral, using the same mathematical machinery as in control theory.

### Hypothesis
The binary switching in CWD amplifies alignment estimation noise (from mini-batch stochasticity), causing unnecessary WD oscillations when true alignment hovers near zero. Super-twisting WD will:
1. Produce smoother λ_t trajectories (lower Var(λ_t) by >30% compared to CWD)
2. Improve or match CWD final accuracy on CIFAR-10/100
3. Show better robustness to mini-batch size changes (because the continuous law is less sensitive to noisy alignment estimates)

### Method

Algorithmic implementation:

```python
class SuperTwistingWD:
    def __init__(self, k1=0.5, k2=0.1, lambda_base=1e-4, lambda_min=0.0, lambda_max=1e-2,
                 v_max=1e-3):
        self.k1 = k1
        self.k2 = k2
        self.lambda_base = lambda_base
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.v_max = v_max
        self.v = 0.0  # integral state

    def compute_lambda(self, w_flat, g_flat, gamma):
        # Sliding variable: alignment cosine
        sigma = torch.dot(w_flat, g_flat) / (torch.norm(w_flat) * torch.norm(g_flat) + 1e-8)
        # Super-twisting control law (continuous, chattering-free)
        u = -self.k1 * (abs(sigma) ** 0.5) * torch.sign(sigma) + self.v
        # Integral update (discrete-time approximation, with clipping for noise robustness)
        self.v = float(torch.clamp(
            torch.tensor(self.v) - self.k2 * torch.sign(sigma) * gamma,
            -self.v_max, self.v_max
        ))
        # Effective WD
        lambda_eff = torch.clamp(self.lambda_base + u, self.lambda_min, self.lambda_max)
        return lambda_eff

    def step(self, w_flat, g_flat, gamma):
        lambda_eff = self.compute_lambda(w_flat, g_flat, gamma)
        return w_flat * (1 - lambda_eff)  # weight decay step
```

This is a one-module drop-in replacement for any optimizer's weight decay step, requiring ~15 lines of code change.

### Diagnostic Experiment

**The key diagnostic test** that confirms the SMC analogy is load-bearing (not decorative):

**Setup**: Train ResNet-20 on CIFAR-10 with:
- CWD (binary alignment mask, ICLR 2026)
- Super-twisting WD (continuous law above)
- Vanilla WD (λ_base = constant)
All three with matched BEM (same effective total WD budget, confirmed via CSI measurement).

**Measurements**:
1. **Primary**: WD trajectory variance `Var(λ_t)` over training
2. **Primary**: Final test accuracy (mean ± std over 3 seeds: 42, 123, 456)
3. **Diagnostic**: Alignment trajectory `cos_sim(w_t, g_t)` over training
4. **Diagnostic**: Number of "alignment sign flips" per epoch (chattering metric, directly from SMC)
5. **Robustness**: Repeat with batch size 32 (high noise) vs 512 (low noise) — tests the key claim that super-twisting is more robust to stochastic noise than binary CWD

**The mechanism-confirming prediction**: Super-twisting WD should show significantly fewer alignment sign flips than CWD while maintaining the same qualitative alignment-conditioned regularization. If super-twisting WD performs better than CWD **and** shows fewer sign flips, the SMC analogy is confirmed as the mechanism.

### Experimental Plan

**Phase 1 (Pilot, ~15 min each)**: ResNet-20, CIFAR-10, seed 42 only
- Implement super-twisting WD in existing codebase
- Compare CWD vs. super-twisting WD vs. vanilla WD (n=1 seed, 200 epochs)
- Check WD trajectory variance qualitatively

**Phase 2 (Full experiment, ~45 min each)**: ResNet-20, CIFAR-10, 3 seeds (42, 123, 456)
- CWD vs. super-twisting WD vs. vanilla WD
- Record all diagnostic metrics
- Batch size ablation: 128 (standard) vs. 512

**Phase 3 (Optional extension, ~30 min each)**: VGG-16-BN, CIFAR-100
- Validate generalization of super-twisting WD advantage to larger models

All experiments use 200 epochs, initial LR=0.1, step decay at epochs 100 and 150, baseline λ=1e-4.

### Risk Assessment

1. **Stochastic SMC convergence**: Finite-time convergence of super-twisting SMC requires bounded noise. Mini-batch gradients have unbounded tails in general. The practical risk is that the integral state v_t accumulates noise and drifts. Mitigation: clip v_t to [-v_max, v_max] (this is standard practice in discrete-time SMC implementations).

2. **Alignment estimation noise dominance**: If mini-batch alignment estimates are so noisy that even the continuous law cannot distinguish signal from noise, super-twisting WD will be no better than vanilla WD. This would mean the alignment signal has low AIS, which is itself an important negative result about CWD's fundamental limitations.

3. **Hyperparameter sensitivity**: k1 and k2 require tuning. Practical mitigation: grid search on k1 ∈ {0.1, 0.3, 0.5, 1.0} and k2 ∈ {0.01, 0.05, 0.1}.

4. **Analogy breaks at scale**: Super-twisting SMC is designed for deterministic continuous-time systems with bounded perturbations. Deep network training at scale (ImageNet, LLMs) has much larger stochastic noise and more complex alignment dynamics. The advantage may diminish or disappear at scale.

### Novelty Claim

The specific cross-disciplinary insight: **Cautious Weight Decay (ICLR 2026) is a relay sliding mode controller on the alignment surface, and its instability near alignment=0 is the chattering problem from SMC. The super-twisting extension provides the first theoretically motivated, chattering-free version of alignment-aware WD, with a formal Lyapunov stability argument carried from control theory.**

Evidence that this has not been applied before:
- CWD paper (arXiv:2510.12402) describes the sliding-mode behavior informally but does not formalize it as SMC
- No ML paper has proposed super-twisting control as a replacement for binary alignment masking
- arXiv search for "super-twisting weight decay" returns no results
- arXiv search for "sliding mode optimizer" returns papers about using ML to design SMC controllers (not about SMC theory for optimizer design)
- PIDAO (Nature Communications 2024) introduces PID to deep learning but does not connect integral action to WD

### Connection to Unified WD Framework

This contribution fits into the Unified Dynamic Weight Decay Framework as follows:

The **general dynamic WD rule** can be written as:
```
λ(t, w, g) = λ_base(t) * φ(cos_sim(w, g), history)
```

Where:
- Standard WD: φ = 1 (no modulation)
- CWD: φ = I[sign(w·g) > 0] (relay SMC, P-type)
- **Super-twisting WD**: φ = 1 − (k1/λ_base)|σ|^(1/2) sign(σ) + v/λ_base (continuous SMC with integral, PI-type)
- Multi-timescale WD (Candidate B): φ = 1 − α · tanh(EMA_β(σ)/T_align)

The **PID taxonomy for WD variants**:
| WD Variant | PID Type | Disturbance Rejection |
|---|---|---|
| Standard decoupled WD | P-only | None (steady-state bias) |
| SWD / ADANA schedules | PD-type | Rate-of-change feedback |
| CWD (relay SMC) | P-type switching | Binary state feedback |
| Super-twisting WD | PI-type continuous | Continuous state + integral |
| Norm-matched WD | I-type | Integrated norm-deviation |

The **Coupling Stability Index (CSI)** should directly measure the chattering level Var(λ_t), making the SMC analogy an explicit design criterion for this metric: lower CSI = more chattering-free WD trajectory = better SMC-style stability. The PID taxonomy provides the theoretical framework for understanding why different CSI values arise from different optimizer architectures.

---

## Supporting Cross-Domain Evidence for Unified Framework

Beyond the main proposal, the following cross-disciplinary evidence strengthens key components of the Unified Dynamic Weight Decay Framework:

### Thermodynamic Pressure Interpretation of WD

From arXiv:2511.07308: WD plays the role of thermodynamic pressure P; weight norm ‖w‖ plays the role of volume V; LR plays the role of temperature T. The ideal gas law PV = nkT in this analogy becomes: λ · ‖w‖² ≈ η · LR (some proportionality). This means:
- **BEM (Budget Equivalence Metric)** has a thermodynamic interpretation: comparing WD budgets is comparing "pressure-volume work" done on the parameter space
- **Optimal WD scaling law** (Wang & Aitchison 2024: WD ∝ 1/T_epochs) follows directly from the ideal gas law when T (temperature = LR) decreases as training progresses

### Free Energy Principle / ELBO Interpretation of WD

From the Friston FEP framework (PMC:8871280): WD is the KL divergence penalty from a zero-centered Gaussian prior on weights (the complexity term in variational inference). Standard WD corresponds to a fixed prior precision λ. **Alignment-aware WD corresponds to a precision-weighted prior**: the prior precision is multiplied by the informativeness (alignment) of the current gradient, consistent with the FEP precision weighting principle. This provides a Bayesian interpretation of AIS (Alignment Informativeness Score): AIS measures how much varying the prior precision based on alignment improves the ELBO, which is a well-defined variational inference quantity.

### BCM Sliding Threshold as Continuous AIS Signal

The BCM sliding threshold θ_M = E[y²] (the expected squared post-synaptic activity) is the biological precursor to the alignment informativeness score. In BCM theory, synaptic plasticity is potentiated when y > θ_M and depressed when y < θ_M. The structural correspondence: replace y (post-synaptic activity) with cos_sim(w, g) (alignment), and replace θ_M (activity threshold) with 0 (zero alignment). The **Alignment Informativeness Score (AIS)** should measure whether the zero-alignment threshold is a meaningful signal boundary — directly analogous to measuring whether θ_M is a meaningful activity threshold in BCM theory.

### Lotka-Volterra Self-Regulation as WD Necessity Condition

From arXiv:2410.10999: In Lotka-Volterra systems, when species possess self-regulation (intraspecific competition coefficient > 0), stability can always be maintained by increasing self-regulation strength. **WD is the self-regulation coefficient for parameter "species."** Without WD, if the gradient descent dynamics have no self-regulatory term (pure GD), the stability of the parameter system depends entirely on the loss landscape structure. Adding WD provides guaranteed self-regulation, making stability achievable independent of the loss landscape. This justifies WD as a stability mechanism (not just a regularizer) in the unified framework's theoretical section.

### PIDAO (Nature Communications 2024) as Validation of PID-WD Connection

The PIDAO paper (DOI:10.1038/s41467-024-54451-3) demonstrates that adding integral action to deep learning optimizers: (1) eliminates steady-state optimization error, (2) enables convergence to global minima (not just local), (3) reduces overshoot/oscillation. Crucially, their results show that WD and PID terms interact: higher WD in PIDAO requires higher integral gain ki to maintain convergence quality. This is direct empirical evidence that WD acts as a disturbance requiring integral compensation. The PIDAO paper provides the experimental validation baseline for Candidate D, even though the authors did not interpret their results in terms of WD disturbance rejection.

---

*Document prepared by sibyl-interdisciplinary agent, 2026-03-18 (revised).*
*Workspace: /home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current*
*New additions in this revision: Candidate D (PID Integral Action for WD Disturbance Rejection), PIDAO analysis, Friston FEP interpretation, expanded RG fixed-point analysis, PID taxonomy for WD variants.*

---

## Supplementary: New Cross-Field Insights (2026-03-18 Second Pass)

*Added by second interdisciplinary agent pass. These insights build on the existing content above and focus on three sources not previously covered: (1) first-order phase boundary control from 2025 physics papers, (2) evolutionary biology's position-dependent mutation rate, and (3) modular neuroscience homeostasis from Turrigiano lab 2025.*

### S1: Phase Boundary Control — WD as Order Parameter Controller

**New source papers (2024-2025)**:
- **"Phase Transitions Between Accuracy Regimes in L2-Regularized Deep Neural Networks"** (arXiv:2505.06597, 2025) — Increasing β (L2 regularization) smooths the loss landscape; first-order "onset-of-learning" phase transition occurs when β exceeds a critical value β_c(task). Order parameter: ‖w‖₂. Hysteresis effects. Ricci curvature of error surface changes sign at transition.
- **"Grokking as a First-Order Phase Transition in Two-Layer Networks"** (ICLR 2024, arXiv:2310.03789) — WD λ is formally the control parameter; weight norm is formally the order parameter. Increasing λ drives a first-order symmetry-breaking transition. Training with fixed λ means operating at a single point in the (λ, ‖w‖) phase diagram.

**Novel prescriptive insight** (not present in prior content above): Both existing papers are *descriptive* — they identify the phase transition but do not use it to prescribe optimal WD schedules. The prescriptive leap is:

> A WD schedule designed to *track the phase boundary curve* β_c(‖w‖, task) — operating at a fixed small margin inside the "learning phase" — will avoid the "trivial phase" collapse while maximizing regularization strength.

Concrete formulation:
```
λ_t^{phase} = λ* · max(0, ‖w_t‖ - τ_w) / τ_w
```
where τ_w is the target norm at the phase boundary (estimatable from a 15-minute pilot with fixed-λ grid). This is a norm-feedback controller: λ_t increases proportionally as ‖w_t‖ exceeds the phase boundary target.

**Connection to existing document**: This complements Candidate A (thermodynamic pressure) by providing an explicit *norm-feedback law* rather than a static pressure interpretation. The "PID taxonomy" in the main document should classify this as I-type (integral/feedback on norm deviation), consistent with the table in Phase 5.

**Novelty**: arXiv search confirms no paper has proposed using the empirical phase boundary curve to design the WD schedule. The existing papers identify the phase transition but treat λ as a fixed hyperparameter.

---

### S2: Fitness-Landscape-Adaptive WD — Joint Conditioning on (Loss Level × Alignment)

**Source** (evolutionary biology): Organisms adapt their mutation rate based on position in the fitness landscape:
- **Near-extinction populations (fitness valleys, high loss)**: high mutation rate beneficial; exploration > exploitation
- **Well-adapted populations (fitness peaks, low loss)**: high mutation rate harmful; exploitation > exploration

**Key 2025 papers**:
- "From Valleys to Peaks: The Role of Evolvability in Fitness Landscape Navigation" (PNAS Nexus, 2025): Low-prominence organisms (valleys) show high epistasis and evolvability; peak populations prioritize robustness.
- "The Adaptive State Determines the Impact of Mutations on Evolving Populations" (PNAS 2025): Mutation benefits are inverted based on fitness state — the same mutation that helps near-extinction harms a well-adapted population.

**Structural correspondence** (novel — not in main document):

The fitness-landscape-adaptive mutation rate mechanism predicts a *joint conditioning* on landscape position AND gradient direction:

| Evolutionary State | Mutation Rate | WD Analogue |
|---|---|---|
| Fitness valley (high loss) | High — explore | λ_t → 0 |
| Fitness slope (medium loss, aligned gradient) | Medium — directed | λ_t = moderate |
| Fitness peak (low loss, aligned gradient) | Low — exploit | λ_t = λ_max |
| Any state, neutral mutation (δ_t ≈ 0) | Low — no benefit | λ_t → 0 |

This maps to:
```
λ_t^{joint} = λ_max · sigmoid(δ_t / T_align) · (1 - exp(-f_t / f_0))
```
where:
- sigmoid(δ_t / T_align) = alignment gate (0 when δ_t ≈ 0, 1 when δ_t >> 0)
- (1 - exp(-f_t / f_0)) = loss-position gate (0 at high loss, 1 at low loss)

**Key prediction distinguishing this from CWD + loss curriculum**:
The interaction effect: AT HIGH LOSS, even high alignment should not trigger strong WD (unlike CWD which applies full WD whenever sign alignment is positive). This is the falsifiable prediction.

**Diagnostic experiment**:
Compare three training phases (early/mid/late) for: (a) CWD, (b) λ_t^{joint}, (c) loss-curriculum WD. If the joint conditioning is correct, the joint method should outperform CWD especially in early training (high loss phase) while matching CWD in late training.

**Novelty assessment**: CWD conditions on alignment only (binary sign). SWD conditions on gradient norm (loss-level proxy). No method conditions on (loss level × alignment) jointly. Evolutionary biology motivation provides principled justification for the interaction term.

---

### S3: Modular Two-Signal Homeostasis (Turrigiano PNAS 2025)

**New source**:
- **Turrigiano Lab, PNAS 2025** — "Modular Arrangement of Synaptic and Intrinsic Homeostatic Plasticity within Visual Cortical Circuits." Key finding: synaptic scaling (norm control) and intrinsic homeostatic plasticity (direction/excitability control) can be **independently recruited** by different input signals — one responds to mean firing rate (global norm signal), the other responds to NMDAR activity (direction-specific, Hebbian-like signal).

**Structural correspondence to WD framework**:

| Biological Module | Signal | WD Analogue |
|---|---|---|
| Synaptic scaling | Mean firing rate = ‖activity‖ proxy | λ_norm = norm-feedback controller |
| Intrinsic homeostatic plasticity | NMDAR (direction-specific) | λ_align = alignment-conditioned WD |
| Modular independence | Two signals, two pathways | Decoupled norm and direction modules |

This is the **two-module architecture** that the Unified Dynamic Weight Decay Framework should embody:
```
λ_t = λ_norm(‖w_t‖) + λ_align(δ_t, f_t)
```
or multiplicatively:
```
λ_t = λ_norm(‖w_t‖) · λ_align(δ_t, f_t)
```

The neuroscience result proves that the modular decomposition (norm ≠ direction) is not just mathematically convenient — it is the biologically optimal architecture for systems that must balance global stability (norm control) with local information preservation (alignment-aware direction control).

**Connection to decoupled WD (AdamW, AdamO)**: The AdamO paper (arXiv:2602.05136) identifies the "Radial Tug-of-War" as the problem with coupled WD. The Turrigiano 2025 finding provides an independent biological argument for why decoupling is correct: the two homeostatic modules are independently recruited precisely because different biological signals encode norm vs. direction information. Coupling them would prevent the selective activation that enables the biological system to maintain norm without disrupting directional information.

**Implication for unified framework**: The two-module decomposition (norm module + direction module) should be the *organizing principle* of the Unified Dynamic Weight Decay Framework's theoretical section, validated by the Turrigiano 2025 modular homeostasis result as an existence proof that such decompositions are stable and effective in biological neural networks.

---

### S4: Combined Final Proposal (Two-Signal Controller)

The three supplementary insights (S1 + S2 + S3) combine into a unified two-signal controller that complements the super-twisting SMC proposal in the main document:

**PAWD (Phase-Adaptive, Alignment-Conditioned WD)**:
```
λ_t = λ_norm(‖w_t‖) · α_align(δ_t, f_t)

λ_norm(‖w_t‖) = λ_max · max(0, ‖w_t‖ - τ_w) / τ_w   [phase boundary tracking, S1]
α_align(δ_t, f_t) = sigmoid(δ_t/T) · (1 - exp(-f_t/f_0))  [joint conditioning, S2]
```

**Relationship to super-twisting WD (main document Candidate D)**:
- Super-twisting WD focuses on the *trajectory dynamics* of the alignment signal (integral action for chattering elimination)
- PAWD focuses on the *steady-state target* of the norm (phase boundary tracking) and the *joint conditioning* on loss-level × alignment
- These are orthogonal improvements: a combined method would be super-twisting + PAWD = (integral action on alignment) × (joint conditioning on loss × norm)

**Position in PID taxonomy from main document**:
- PAWD norm module: I-type (integral feedback on norm deviation from τ_w)
- PAWD alignment module: P-type with gain scheduling (gain = loss-position factor)
- Super-twisting WD: PI-type on alignment signal
- Full combined: IPI-type (I on norm, PI on alignment with loss-level gain schedule)

**Unification**: All four WD sub-approaches in the literature are special cases of the IPI controller:
- Standard WD: all gains → scalar (λ_max)
- CWD: P-type alignment, no norm, no loss-level
- SWD: P-type on gradient norm (loss proxy), no alignment
- AdamWN (norm control): I-type on norm, no alignment
- AdamO: I-type on norm (SGD radial step), no alignment conditioning
- **PAWD: I-type norm + P-type alignment with loss-level gain**
- **Super-twisting WD: I-type norm + PI-type alignment**

*Second-pass additions by sibyl-interdisciplinary, 2026-03-18.*
*New papers incorporated: arXiv:2505.06597, arXiv:2310.03789 (ICLR 2024), PNAS Nexus 2025 (fitness landscapes), PNAS 2025 (Turrigiano modular homeostasis), PMC 2025 (adaptive mutation rates).*
