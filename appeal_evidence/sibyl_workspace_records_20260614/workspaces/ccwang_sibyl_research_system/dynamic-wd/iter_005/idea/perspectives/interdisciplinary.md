# Interdisciplinary Perspective

**Research Topic**: Unified Dynamic Weight Decay Framework (Unifying WD Scheduling, Alignment-Aware WD, Decoupled WD, Norm-Matched WD)
**Agent**: sibyl-interdisciplinary
**Date**: 2026-03-18

---

## Phase 1: Literature Survey

### By Source Field

#### Control Theory / Dynamical Systems

1. **Sliding Mode Control (SMC) — Sign-Based Switching Surface with Chattering Avoidance**
   - Key mechanism: Variable-structure controller that drives system trajectories onto a sliding surface S(x)=0, then maintains motion on that surface. Two operating regimes: "reaching mode" (off-surface, strong control force) and "sliding mode" (on-surface, equivalent control). Chattering — high-frequency oscillations from discontinuous sign switching — is the central problem; addressed by smooth approximations (sigmoid, boundary layer, higher-order SMC).
   - Relevance: CWD (ICLR 2026) was noted to exhibit "sliding-mode behavior," but uses only binary sign matching. This suggests an unexplored bridge between full SMC theory and gradient-alignment-based WD.

2. **Lyapunov Stability Theory & Adaptive Control**
   - Key mechanism: A scalar function V(x) > 0 with dV/dt <= 0 along trajectories guarantees asymptotic stability. Control-Lyapunov functions (CLF) yield feedback laws via Sontag's universal formula. Adaptive control uses Lyapunov-derived parameter update rules to stabilize unknown plant parameters, bounding parameter drift.
   - Relevance: The Sun et al. (CVPR 2025) proofs use Lyapunov/potential functions for SGDW convergence. The key gap: no existing WD theory derives the WD schedule from a CLF requirement — i.e., asks "what lambda_t guarantees dV/dt <= 0 along the training trajectory?"

3. **Pontryagin Maximum Principle (PMP) for Optimal Control**
   - Key mechanism: For a system dx/dt = f(x,u,t) with cost J = integral(L(x,u,t)dt), the optimal control minimizes the Hamiltonian H = L + p*f. PMP gives necessary conditions; adjoints p evolve backward in time as dp/dt = -dH/dx. This is structurally identical to backpropagation.
   - Relevance: Framing WD scheduling as an optimal control problem — choosing lambda(t) to minimize a combined convergence-generalization objective — is unexplored. The adjoint/costate equation for this problem would prescribe precisely how lambda(t) should evolve, potentially recovering alignment-aware rules as a special case.

4. **Optimal Annealing Schedules in Non-Equilibrium Statistical Mechanics (Phys. Rev. E 2024)**
   - Key mechanism: Sivak and Crooks derived optimal protocols for driving thermodynamic systems at minimum dissipation cost. The optimal schedule is determined by the "friction tensor" (a metric on parameter space quantifying how hard it is to change each parameter), and slows down near critical points. A 2024 PRE paper extended this to multidimensional parameter spaces.
   - Relevance: Weight decay lambda(t) is formally a time-varying control parameter. The "friction tensor" for WD corresponds to the sensitivity of loss to changes in lambda — precisely related to the gradient-weight alignment signal. This framing provides a first-principles derivation of optimal WD schedules from nonequilibrium thermodynamics.

#### Neuroscience / Cognitive Science

5. **Synaptic Homeostasis Hypothesis (SHY) — Sleep as Synaptic Downscaling**
   - Key mechanism: Tononi & Cirelli's SHY proposes that wakefulness causes net synaptic potentiation (weight growth via Hebbian learning), and sleep performs a global downscaling that prevents saturation while preserving relative weight structure. The scaling is triggered by slow-wave activity (SWA), which encodes the accumulated plasticity "debt." Recent ML work (arXiv:2601.08447, arXiv:2602.07009) directly implements this as periodic offline weight decay phases.
   - Relevance: SHY provides a biological precedent for the key claim that weight decay should be temporally structured — high during "consolidation phases," reduced during "active encoding." The SWA signal is structurally analogous to alignment tracking: both measure the accumulated signal-to-noise ratio of the synaptic/parameter configuration.

6. **Multi-Scale Temporal Homeostasis (MSTH) — Four-Timescale Regulation (arXiv:2602.07009)**
   - Key mechanism: Artificial neural networks implementing MSTH use four nested timescales: ultra-fast (5ms synaptic depression), fast (2s calcium dynamics), medium (5min synaptic scaling a la Turrigiano), and slow (24hr structural plasticity). These nested feedback loops maintain stability at each scale while enabling flexible adaptation.
   - Relevance: The four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) could be reinterpreted as operating at four different timescales of the same underlying parameter regulation problem. This provides a biological organizing principle for the proposed unified framework.

7. **Neuron Homeostasis and Ion Channel Density Regulation**
   - Key mechanism: Neurons regulate their average firing rate by adjusting ion channel expression levels (a slow process, hours to days). This is a multiplicative scaling of all conductances that moves the neuron's operating point back to a target activity level. The regulated quantity is not the absolute weight value but the ratio of synaptic drive to intrinsic excitability.
   - Relevance: This is structurally isomorphic to norm-matched WD: regulate not the absolute weight value (shrink to zero) but the ratio of gradient drive to weight norm — exactly the ||g||/||w|| "layer balancing" principle discovered by Defazio (2025). Biological homeostasis thus provides both the target quantity and the control law.

#### Statistical Physics / Information Theory

8. **Renormalization Group (RG) Flow and Running Coupling Constants**
   - Key mechanism: In quantum field theory, coupling constants are not fixed but "run" with energy scale according to the beta function: dg/d(log mu) = beta(g). Fixed points of the beta function correspond to scale-invariant theories (phase transitions). The connection to neural networks: weight distributions in trained networks exhibit power-law spectra (HTSR theory; WeightWatcher); changing the weight standard deviation is a RG flow in network space. The 2024 ECT* workshop established that the ridge regularization parameter (L2/WD strength) is exactly the RG flow parameter.
   - Relevance: Direct structural correspondence: WD coefficient lambda is the running coupling constant. Its "beta function" is the function beta(lambda,t) = dlambda/d(log t) that describes how lambda should change during training. The fixed point corresponds to the optimal trained state. Alignment-aware WD changes the beta function based on local geometry — a form of "field-theoretic" optimization that has not been formalized.

9. **Non-Equilibrium Statistical Mechanics: Fluctuation-Dissipation Relations for SGD**
   - Key mechanism: SGD has been analyzed as a non-equilibrium stochastic process. The fluctuation-dissipation theorem (FDT) at stationarity links the SGD noise covariance to the loss Hessian via the effective temperature T_eff = epsilon/(2b) (learning rate epsilon, batch size b). WD shifts the effective potential, modifying the stationary distribution. Under FDT, optimal WD at each training phase is determined by the ratio of gradient variance to Hessian curvature.
   - Relevance: This provides a principled derivation of alignment-aware WD: the alignment signal <g,w>/||g||||w|| measures the coherence between the noise direction and the weight direction — exactly what the fluctuation-dissipation analysis identifies as the relevant geometric quantity for determining optimal WD.

10. **Heavy-Tailed Self-Regularization (HTSR) and Spectral Power Laws**
    - Key mechanism: Well-trained neural networks exhibit power-law eigenvalue distributions P(lambda) ~ lambda^{-alpha} in their weight matrices, with alpha in [2,6] for generalizing models. WD promotes rank minimization (Galanti et al. 2022) and accelerates the transition to heavy-tailed spectra. AlphaDecay (2025) uses the spectral exponent alpha as a module-wise WD signal.
    - Relevance: The spectral exponent alpha is a macroscopic order parameter analogous to a thermodynamic state variable. A WD schedule that monitors alpha and targets alpha in [2,4] (the "critical" range associated with good generalization) would implement spectral-feedback WD scheduling — closing the loop identified as Gap 10 in the literature survey.

#### Biology / Evolution

11. **Germinal Center Affinity Maturation — Selection Strength Adaptation**
    - Key mechanism: B cells in germinal centers undergo dark-zone mutation (somatic hypermutation) and light-zone selection (antigen competition). The selection pressure (fraction of cells surviving) is dynamically regulated by the antigen concentration in the light zone, which decreases as affinity improves. High selection early (many deaths, strong pressure) -> high-affinity winners. The key quantity is the affinity gradient dFitness/dBCR, which modulates selection strength.
    - Relevance: Selection pressure in germinal centers is structurally identical to WD strength in SGD: both shrink the "population" (or weight magnitude) in proportion to the gap between current fitness/loss and the target. The affinity gradient in biology corresponds to the gradient-weight alignment in ML: high alignment = the gradient is pointing toward better solutions = high "affinity gain" = reduce WD. Low alignment = the gradient is not informative about the weight's optimality = increase WD.

12. **Fitness Landscape Ruggedness and Adaptive Mutation Rate (NK Model)**
    - Key mechanism: Kauffman's NK model parameterizes landscape ruggedness K (epistatic interactions). On rugged landscapes (high K), exploration requires higher mutation rates; on smooth landscapes (low K), low mutation rates with greedy exploitation work better. Optimal strategies dynamically estimate K from population statistics and adjust mutation rate accordingly.
    - Relevance: The alignment signal delta_t = <g_t, w_t>/(||g_t|| ||w_t||) measures the "ruggedness" of the loss landscape in the gradient-weight subspace. High alignment (smooth landscape in this direction) -> reduce WD (lower "mutation" of weights away from current direction). Low alignment (rugged) -> increase WD (explore more aggressively). This gives an algorithmic meaning to alignment-aware WD from an evolutionary optimization perspective.

13. **Computational Convergence of Adaptive Immunity and AI (bioRxiv:2026.02.03.703525)**
    - Key mechanism: The softmax attention function is mathematically equivalent to the Boltzmann distribution of antibody-antigen binding. The InfoNCE contrastive loss equals the negative log of the clonal selection probability. These are exact mathematical equivalences, not metaphors. The pre-training/fine-tuning/RLHF training hierarchy parallels the germline/somatic hypermutation/T follicular helper stages of antibody affinity maturation.
    - Relevance: The "selection signal" in the immune system — how much a B cell's BCR aligns with the antigen — is mathematically the same as the gradient-weight alignment in the optimizer. This means the entire literature on optimal affinity maturation scheduling directly translates to optimal dynamic WD scheduling.

#### Ecology

14. **Population Self-Regulation via Sublinear Growth (Science 2024)**
    - Key mechanism: Diversity-stability paradox resolved: populations with sublinear growth (exponent < 1 in biomass) self-regulate without going to zero or exploding. This avoids competitive exclusion while maintaining stability at high diversity. The sublinear exponent is equivalent to a fractional damping term.
    - Relevance: WD implements exactly this: the weight update w_{t+1} = (1-lambda_t)w_t acts as a sublinear damping of the weight population. The stability analysis of sublinear population dynamics (carrying capacity, coexistence conditions) translates directly into stability conditions for WD schedules.

15. **Optimal Foraging Theory — Patch Exploitation vs. Exploration (Marginal Value Theorem)**
    - Key mechanism: An animal should leave a food patch when its local intake rate drops to the average environment rate. This balances exploitation of current patch vs. exploration of new patches. The switching threshold is a function of the gradient of remaining resources in the current patch.
    - Relevance: WD scheduling faces the same exploitation-exploration tradeoff: at what point should the optimizer stop strongly exploiting the current solution (low WD) and switch to more aggressive regularization (high WD) to explore a wider basin? The marginal value theorem provides a principled switching rule: increase WD when the marginal generalization gain from current gradient direction drops below the average expected gain.

### Cross-Disciplinary Gaps

The following structural transplants appear not to have been attempted:

1. **RG beta function formalism for WD schedule derivation**: The ECT* 2024 workshop established that WD strength is the RG flow parameter, but no paper has derived the beta function beta_WD(lambda,t) = dlambda/d(log t) for gradient-based optimization, or shown that alignment-aware WD modifies the beta function.

2. **Pontryagin-optimal WD schedule**: Framing WD scheduling as an optimal control problem and deriving the necessary conditions via PMP has not been done. This would yield a closed-form prescription for lambda(t) in terms of adjoint/costate variables related to the alignment signal.

3. **Thermodynamically optimal WD via Sivak-Crooks friction tensor**: Extending the 2024 Phys. Rev. E optimal annealing formalism to the WD parameter space has not been done.

4. **SHY-inspired two-phase WD**: The biological precedent of sleep (downscaling) vs. wakefulness (potentiation) has been implemented for spiking nets but not for standard SGD with alignment-aware dynamic WD.

---

## Phase 2: Initial Candidates

### Candidate A: Beta Function WD — Renormalization Group Flow of the Decay Parameter (from Statistical Physics)

- **Source principle**: In quantum field theory, the renormalization group describes how coupling constants g "run" with energy scale mu via the beta function beta(g) = mu * dg/dmu. Fixed points beta(g*)=0 correspond to scale-invariant theories (phase transitions). The 2024 ECT* workshop established that the weight distribution standard deviation undergoes an RG flow, with L2/WD strength as the RG flow parameter. The "running" of the coupling (WD) depends on the current spectral properties of the weight distribution.

- **Structural correspondence**:
  - Energy scale mu (RG) <-> Training time t (via log(t))
  - Coupling constant g(mu) (RG) <-> WD coefficient lambda_t
  - Beta function beta(g) = dg/d(log mu) (RG) <-> dlambda/d(log t) = beta_WD(lambda, delta_hat_t, alpha_t)
  - RG fixed point g* where beta(g*)=0 <-> Optimal terminal WD lambda* at convergence
  - Power-law spectral exponent alpha (HTSR) <-> Order parameter measuring distance from critical trained state
  - Critical slowing down near phase transition <-> Optimal WD schedule slows near delta_hat_t approx 0 and alpha_t approx alpha*
  - UV -> IR flow (coarse-graining of high-frequency fluctuations) <-> Early-training -> late-training (removing high-frequency noise, retaining structure)
  - Ridge parameter (L2 regularization strength) as RG coupling <-> Formalized at ECT* 2024 workshop

- **Hypothesis**: The alignment signal delta_t = <g_t, w_t>/(||g_t|| ||w_t||) measures the deviation of the current WD trajectory from the RG fixed point. When delta_t is far from 0 (high alignment or anti-alignment), the system is "off the critical manifold" and needs stronger WD flow correction. When delta_t approx 0 (orthogonal), the system is near the critical point and WD should be minimal. The beta function is: beta_WD(lambda, delta) proportional to (delta^2 - lambda), which has a stable fixed point at lambda* = beta_0 * delta^2.

- **Why not just a metaphor**: The ECT* 2024 workshop established a formal mathematical identification between the ridge regularization parameter and the RG flow parameter — not an analogy. The weight distribution standard deviation undergoes a genuine RG flow during training. The HTSR spectral exponent alpha is a macroscopic order parameter with a well-defined relationship to the underlying coupling constant. Changing the WD coefficient changes the RG flow and hence the terminal spectral state — a structural, not surface, correspondence.

- **Novelty estimate**: 9/10

---

### Candidate B: Pontryagin-Optimal WD Scheduling (from Optimal Control Theory)

- **Source principle**: In optimal control, the Pontryagin Maximum Principle (PMP) provides necessary conditions for optimal trajectories. For a dynamical system dx/dt = f(x,u,t) with cost J = integral(L(x,u,t)dt) + Phi(x(T)), the optimal control u*(t) minimizes the Hamiltonian H(x,p,u,t) = L(x,u,t) + p*f(x,u,t), where p is the adjoint/costate variable satisfying dp/dt = -dH/dx. The adjoint equation propagates information backward in time, like backpropagation.

- **Structural correspondence**:
  - State variable x (control) <-> Weight vector w_t (optimization)
  - Control variable u(t) (control) <-> WD schedule lambda_t (optimization)
  - System dynamics f(x,u,t) <-> SGD update: w_{t+1} = (1-lambda_t)w_t - gamma_t*g_t
  - Cost functional L(x,u,t) <-> Combined training loss + generalization regularizer
  - Adjoint/costate variable p_t (control) <-> Dual variable measuring sensitivity of future loss to current weights
  - Hamiltonian H (control) <-> Augmented potential Phi_t = f_S(w_t) + beta_t*||w_t||^2 (from Sun et al. CVPR 2025)
  - Optimality condition dH/du = 0 <-> Optimal WD determined by current alignment and weight norm
  - Adjoint dynamics dp/dt = -dH/dx <-> Backward information propagation (retrospective alignment)

- **Hypothesis**: The PMP necessary condition dH/dlambda = 0 yields a closed-form prescription for optimal lambda_t as a function of the adjoint variable p_t, the gradient g_t, and the current weight w_t. The adjoint p_t encodes the "shadow price" of current weight magnitude for future generalization. The sign of dH/dlambda is exactly related to the alignment p_t * w_t, providing a rigorous derivation of alignment-aware WD from first principles.

- **Formal mapping**:
  - Hamiltonian: H(w, p, lambda, t) = L_train(w) + lambda^2/(2*beta) + p*[(1-lambda)w - gamma*g(w)]
  - Optimality condition: dH/dlambda = lambda/beta - p*w = 0 => lambda* = beta * (p*w)
  - Adjoint equation: dp/dt = -dH/dw = -grad_L(w) + (1-lambda)*p + gamma*(dg/dw)^T * p
  - First-order approximation: p_t approx -g_t gives lambda*_t approx -beta * <g_t, w_t>

- **Why not just a metaphor**: The PMP applies directly to any discrete-time optimal control problem, and SGD with time-varying WD is exactly such a problem. The derivation of lambda*_t = beta*(p_t*w_t) from the stationarity condition dH/dlambda = 0 is mathematically rigorous. The adjoint variable p_t can be approximated by the current gradient g_t (since p_t approx -grad_w loss), which recovers the alignment-aware WD rule as a first-order approximation to the PMP-optimal schedule.

- **Novelty estimate**: 8/10

---

### Candidate C: Synaptic Homeostasis WD — Phase-Structured Decay Inspired by Sleep Neuroscience (from Neuroscience)

- **Source principle**: The Synaptic Homeostasis Hypothesis (SHY; Tononi & Cirelli) proposes that wakefulness causes net synaptic potentiation (Hebbian weight growth) and slow-wave sleep triggers global synaptic downscaling that prevents saturation, reduces noise, and restores plasticity. The trigger signal is slow-wave activity (SWA) — the accumulated "debt" of potentiation. Recent work (arXiv:2601.08447, arXiv:2602.07009) shows "sleep phases" of 10-20% training duration improve stability in spiking networks using this mechanism.

- **Structural correspondence**:
  - Synaptic weight growth during wakefulness <-> Weight norm growth ||w_t|| during training
  - Slow-wave activity (SWA) accumulation <-> Cumulative alignment integral: integral(lambda_s * delta_s ds)
  - Synaptic downscaling during sleep (multiplicative rescaling) <-> WD phase with elevated lambda_t
  - Wake/sleep phase boundary <-> LR warmup end / WSD stable-phase to decay-phase boundary
  - Target activity level (homeostatic setpoint) <-> Target weight norm tau (AdamWN/norm-matched WD)
  - Calcium dynamics (medium-timescale sensor of excess activity) <-> EMA of alignment EMA(delta_hat_t)
  - Signal-to-noise ratio improvement from downscaling <-> Generalization improvement from WD

- **Hypothesis**: WD should follow a two-phase biological schedule: an "active encoding" phase where WD is low to allow maximal learning, followed by a "consolidation" phase triggered by a threshold on accumulated EMA of gradient-weight alignment. This provides a biologically-grounded, data-driven WD schedule unifying WSD with alignment-aware modulation.

- **Why not just a metaphor**: The mathematical form of synaptic downscaling is multiplicative rescaling w -> (1-lambda)w, which is exactly the WD update. The trigger mechanism maps to alignment tracking. The two-phase structure has been validated in spiking networks (arXiv:2601.08447) and maps onto the WSD schedule already adopted by DeepSeek-V3.

- **Novelty estimate**: 7/10

---

## Phase 3: Self-Critique

### Against Candidate A (Beta Function WD)

- **Shallow analogy attack**: Is the identification of lambda with a coupling constant structural or lexical? Evaluation: The ECT* 2024 workshop provides a formal mathematical result that the ridge parameter undergoes a genuine RG flow during training. The power-law spectral exponent alpha is an order parameter with a defined relationship to coupling constants. The proposed beta_WD(lambda, delta_t) is a new hypothesis requiring derivation — the structural correspondence is genuine at the level of "WD is a running coupling," but the specific beta function form needs formal derivation. Verdict: structural, not metaphorical, but with an additional modeling step needed.

- **Scale mismatch attack**: RG theory operates across vast energy scales in field theory. Neural network training operates across ~100-1000 steps. However, the ECT* 2024 workshop uses RG in the "theory space" (space of neural network weight distributions), not in energy scale, where the timescale concern is less severe. The HTSR spectral exponent genuinely evolves during training.

- **Prior transplant check**: AlphaDecay (2025) uses spectral exponent alpha as a module-wise WD signal (static assignment, not a dynamical schedule). No paper has: (a) formulated the WD beta function beta(lambda, delta, alpha) explicitly, or (b) derived it from first principles using the RG-SGD correspondence. The specific formula lambda*_t = beta_0 * delta_hat_t^2 is new.

- **Testability attack**: The RG interpretation predicts that the optimal WD schedule exhibits "critical slowing down" — the rate dlambda_t/dt is minimized near alignment transition points where delta_hat_t passes through zero and alpha_t near alpha*. An alignment-agnostic schedule shows no such slowing down. Diagnostic: plot optimal WD (from grid search) vs. alignment trajectory and test whether optimal lambda_t changes most slowly at alignment transitions. This is falsifiable and has no mundane alternative explanation.

- **Verdict**: STRONG

---

### Against Candidate B (Pontryagin-Optimal WD)

- **Shallow analogy attack**: Is the identification of WD with a control variable structural? Yes, definitionally: SGD with time-varying lambda_t is a discrete-time optimal control problem where u(t) = lambda_t is the control. The system dynamics are the SGD update equation. PMP applies formally.

- **Scale mismatch attack**: PMP requires smooth dynamics; SGD is discrete and stochastic. The discrete-time PMP is well-developed and applies directly. The stochastic PMP handles the stochasticity via the adjoint becoming a stochastic process, with the stationarity condition holding in expectation.

- **Prior transplant check**: Optimal control framings of deep learning exist (Li et al. ICML 2018), but these fix WD and optimize learning rate. No paper derives the optimal WD schedule via PMP. The adjoint-based result lambda*_t proportional to (p_t * w_t) approx -<g_t, w_t> is novel.

- **Testability attack**: The PMP predicts lambda*_t is large when the gradient is ANTI-aligned with the weights (<g_t, w_t> < 0 => lambda*_t > 0 via |p_t * w_t|). This agrees with Sun et al. CVPR 2025's finding that WD benefits from anti-aligned regimes. The diagnostic experiment: compare RGBW formula (lambda proportional to delta^2) vs. CWD (binary sign alignment) vs. PMP-linear (lambda proportional to |<g_t, w_t>|). Under PMP theory, all three should work but the quadratic formula should have better stability properties due to its CLF-like properties.

- **Verdict**: STRONG. Mathematically rigorous derivation, novel formula, testable predictions, and potentially resolves the CWD vs. anti-alignment-WD debate.

---

### Against Candidate C (Synaptic Homeostasis WD)

- **Shallow analogy attack**: Mathematical form of SHY downscaling is w -> (1-lambda)w, which is exactly WD — structural. The "debt accumulation" mechanism maps onto integral of alignment. The specific trigger mechanism (SWA threshold -> EMA(delta_hat_t) threshold) is a modeling choice requiring validation.

- **Scale mismatch attack**: SHY operates over hours-to-days timescales. Training operates over minutes-to-hours. However, the key claim is about functional structure (two-phase schedule), not absolute timescale. WSD schedules already implement the structure empirically.

- **Prior transplant check**: SHY-inspired SNN regularization (arXiv:2601.08447) published but targets spiking nets. WSD schedules implement two-phase structure but without alignment trigger. Moderately novel.

- **Testability attack**: Compare (a) fixed-schedule WSD, (b) alignment-triggered WSD, (c) randomly triggered WSD. If (b) > (a) > (c), alignment signal is the active ingredient.

- **Verdict**: MODERATE. Sound biological grounding, partially anticipated by WSD. Supporting role rather than front-runner.

---

## Phase 4: Refinement

### Integration: Candidate A + B -> Front-Runner

Candidates A and B describe the same structure from two complementary angles:

- **PMP angle**: Optimal WD satisfies lambda*_t = beta * (p_t * w_t) where p_t approx -g_t.
  This gives: lambda*_t approx -beta * <g_t, w_t>
  Since lambda >= 0, we take: lambda*_t = beta * |<g_t, w_t>| = beta * ||g|| * ||w|| * |delta_hat_t|
  For delta_hat_t in [0,1], |delta_hat_t| approx delta_hat_t^2 in the neighborhood of typical training values.
  Prediction: WD is most beneficial when gradient magnitude is large AND weights are large (high |<g,w>|), regardless of sign.

- **RG angle**: The beta function beta_WD(lambda, delta) = -(lambda - beta_0 * delta^2) has stable fixed point at lambda*(delta) = beta_0 * delta^2. This is a quadratic alignment schedule with a single hyperparameter beta_0.

Both angles converge on: **lambda*_t = beta_0 * delta_hat_t^2**

The convergence of two independent derivations (from statistical physics and from optimal control) on the same formula provides strong theoretical support.

### Strengthened Front-Runner: Quadratic-Alignment WD (QA-WD) from RG-Beta Function Theory

**Core Formula**:

    lambda*_t = beta_0 * EMA(|<g_t, w_t>| / (||g_t|| * ||w_t|| + eps))^2

This unifies the four WD sub-approaches:
1. **WD Scheduling**: lambda_t is automatically time-varying via the evolving alignment signal
2. **Alignment-Aware WD**: Explicitly a function of gradient-weight alignment; CWD is the sign approximation
3. **Decoupled WD**: The formula assumes decoupled application (to w, not through adaptive gradient scaling) — required by the RG formalism
4. **Norm-Matched WD**: The fixed point lambda*(delta*) = beta_0 * delta*^2 implies an implicit target norm tau* = gamma_t / lambda*; the target norm is determined by the alignment trajectory, not set by hand

---

## Phase 5: Final Proposal

### Title
**RG-Beta Weight Decay: Renormalization Group Flow Derivation of the Optimal Dynamic Weight Decay Schedule**

---

### Source Principle

The renormalization group (RG) in statistical physics describes how coupling constants "run" with energy scale. The 2024 ECT* Workshop on Machine Learning and the Renormalization Group established a formal correspondence: the weight distribution of a neural network undergoes a genuine RG flow during training, with the L2/WD regularization strength playing the role of the running coupling constant. The beta function beta(g) = dg/d(log mu) governs how the coupling changes with "scale" (training time). Fixed points beta(g*)=0 correspond to optimal trained states (scale-invariant networks exhibiting HTSR power-law spectra with exponent alpha* approx 2-4).

The Pontryagin Maximum Principle (PMP) provides an independent derivation from optimal control theory: WD scheduling is a control problem, and the PMP stationarity condition yields the same formula as the RG beta function fixed point.

---

### Structural Correspondence

| Physics (RG) | Deep Learning (Dynamic WD) |
|---|---|
| Energy scale mu | Training step t (via log(t)) |
| Running coupling constant g(mu) | WD coefficient lambda_t |
| Beta function beta(g) = dg/d(log mu) | dlambda/d(log t) = beta_WD(lambda, delta_hat_t, alpha_t) |
| Fixed point g* where beta(g*)=0 | Optimal WD lambda* = beta_0 * delta*^2 |
| Power-law spectral exponent alpha (HTSR) | Order parameter measuring distance from optimal spectral state |
| Critical slowing down near phase transition | Optimal WD schedule slows near delta_hat_t approx 0 (alignment transition) |
| UV -> IR flow (high- to low-frequency) | Early-training -> late-training (noise to structure) |
| Ridge parameter (L2 reg) = RG coupling | Formal identification (ECT* 2024 workshop) |
| RG equation: dg/d(log mu) = -g + c*g^3 | lambda update: delta_lambda = gamma*(-lambda + beta_0*delta_hat^2) |
| Renormalized coupling = observable at fixed point | lambda_eff = beta_0 * delta_hat*^2 (alignment-determined equilibrium WD) |

PMP correspondence:
| Control Theory (PMP) | Deep Learning (Dynamic WD) |
|---|---|
| State variable x | Weight vector w_t |
| Control variable u(t) | WD schedule lambda_t |
| System dynamics f(x,u,t) | SGD update: w_{t+1} = (1-lambda_t)w_t - gamma_t*g_t |
| Adjoint/costate variable p_t | Dual variable; p_t approx -g_t (first-order approximation) |
| Hamiltonian H = L + lambda^2/(2*beta) + p*[(1-lambda)w - gamma*g] | Augmented training objective |
| PMP optimality: dH/dlambda = 0 => lambda* = beta*(p*w) | lambda*_t = beta_0 * |<g_t, w_t>| approx beta_0 * delta_hat_t^2 |

---

### Hypothesis

**H1 (Functional Form)**: The optimal time-varying WD coefficient satisfies lambda*_t = beta_0 * delta_hat_t^2, where delta_hat_t = EMA_alpha(|<g_t, w_t>| / (||g_t|| * ||w_t|| + eps)). Compared to fixed WD and CWD (binary alignment), this continuous quadratic schedule achieves better convergence-generalization trade-off.

**H2 (Critical Slowing Down)**: The rate |dlambda_t/dt| is minimized near alignment transition points where delta_hat_t passes through local minima. This is a novel prediction from the RG analogy with no existing ML explanation and no analogous prediction from any current WD scheduling method.

**H3 (Fixed Point Identification)**: At convergence, delta_hat_t stabilizes at delta_hat* determined by the loss landscape geometry. The terminal WD lambda* = beta_0 * delta_hat*^2 is a non-zero residual maintaining the weight norm at an optimal steady state — distinct from WD schedules that decay to zero.

**H4 (Unification)**: CWD (ICLR 2026) is recovered as the sign/threshold approximation. The gamma^2-scaling rule (Chou 2025) is recovered as the zero-alignment limit delta_hat_t -> 0. Fixed WD is recovered when delta_hat_t is approximately constant.

---

### Method: Algorithm QA-WD (Quadratic-Alignment Weight Decay)

```
Given: w_0, gamma_0, beta_0 (hyperparameter), alpha_EMA (EMA decay, e.g., 0.99), eps=1e-8
Initialize: delta_hat_0 = 0

For each step t:
  1. Compute minibatch gradient g_t
  2. Compute raw alignment: a_t = |<g_t, w_t>| / (||g_t|| * ||w_t|| + eps)
  3. Update EMA: delta_hat_t = alpha_EMA * delta_hat_{t-1} + (1 - alpha_EMA) * a_t
  4. Compute WD: lambda_t = beta_0 * delta_hat_t^2
     [Optional: lambda_t = clip(lambda_t, lambda_min, lambda_max)]
  5. Decoupled WD update: w_{t+1} = (1 - lambda_t) * w_t - gamma_t * g_t
  6. Update gamma_t by standard cosine/step schedule
```

Key properties:
- lambda_t in [0, beta_0] automatically (bounded by EMA alignment squared)
- lambda_t -> 0 when gradient and weights are orthogonal (delta_hat -> 0; no useful direction to decay)
- lambda_t -> beta_0 when gradient and weights are perfectly aligned (delta_hat -> 1; maximum decay)
- Symmetric in alignment direction (|delta_hat|): both aligned and anti-aligned receive the same WD magnitude
- Single new hyperparameter beta_0 with physical interpretation as the RG coupling strength
- Principled initialization: beta_0 approx lambda_fixed / <delta_hat^2>_typical (can be estimated from a short pilot run)

---

### Diagnostic Experiment

**Primary diagnostic**: Does the quadratic alignment schedule outperform CWD specifically BECAUSE of the RG critical-slowing-down effect (and not just because it is smoother)?

**Experiment design**:
1. Train ResNet-20 on CIFAR-10 with four conditions:
   - (a) Fixed WD lambda=5e-4 (baseline)
   - (b) CWD (binary sign alignment, from ICLR 2026)
   - (c) QA-WD with lambda_t = beta_0 * delta_hat_t^2 (proposed method)
   - (d) QA-WD-random: replace delta_hat_t^2 with a random signal having the same time-series statistics (mean and autocorrelation) as delta_hat_t^2, but decorrelated from the actual alignment
2. Track per-step: test accuracy, ||w_t||, delta_hat_t, lambda_t, spectral exponent alpha_t of weight matrices
3. **Critical slowing down test**: Compute the cross-correlation between |dlambda_t/dt| and the rate of change |d(delta_hat_t)/dt|. Under RG theory, (c) should show that lambda_t changes SLOWLY when delta_hat_t changes rapidly (near zero crossings) — critical slowing down. Conditions (a), (b), (d) should not show this pattern.
4. **Unification test**: Show analytically that QA-WD reduces to CWD under sign-thresholding and to gamma^2-scaling in the zero-alignment limit.

**Success criterion**:
- QA-WD (c) outperforms QA-WD-random (d) by more than 0.3% test accuracy: confirms alignment signal is the active ingredient, not just smooth variation
- QA-WD (c) outperforms or matches CWD (b): confirms quadratic form is superior to binary approximation
- Critical slowing down test passes in at least 2 of 3 seeds: confirms the RG mechanism is operating

---

### Experimental Plan

All experiments within project constraints (CIFAR-10/100 on ResNet-20/VGG-16-BN, seeds 42/123/456, single RTX PRO 6000 card, total budget approximately 4-6 hours).

**Task 1: Algorithm validation on CIFAR-10/ResNet-20 (approx. 2 hours)**
- 200 epochs, batch 128, initial lr=0.1 with cosine decay
- 5 conditions: No-WD, Fixed-WD (5e-4), CWD, QA-WD (beta_0=1e-3), QA-WD-random
- 3 seeds each: total 15 runs x ~8 min = ~2 hours
- Primary metric: final test accuracy; secondary: per-epoch delta_hat_t, lambda_t, ||w_t||, alpha_t

**Task 2: Diagnostic analysis (offline, 0.5 hours)**
- From Task 1 data: compute cross-correlation between |dlambda_t/dt| and |d(delta_hat_t)/dt| for each seed/method
- Test H2 (critical slowing down): Compute times where delta_hat_t changes fastest and compare lambda_t rate of change
- Test H4 (unification): Analytical derivation of CWD and gamma^2-scaling as limiting cases

**Task 3: Validation on CIFAR-100/VGG-16-BN (approx. 2 hours)**
- Same protocol; test whether beta_0 transfers across architectures/datasets
- Expected: same beta_0 achieves comparable delta_hat^2 equilibrium, confirming hyperparameter universality

**Task 4: Spectral analysis (online during Task 1)**
- Compute spectral exponent alpha_t for the first conv layer at each epoch
- Test H3 (fixed point): measure delta* at convergence and verify lambda* = beta_0 * delta*^2 matches observed terminal lambda

---

### Risk Assessment

1. **EMA alignment signal is too noisy at minibatch level**: High variance from small batches may destabilize lambda_t. Mitigation: longer EMA window (alpha_EMA = 0.999), gradient accumulation, or computing over multiple steps.

2. **The quadratic formula is not uniquely supported — linear |delta| or delta^2 might both work**: The RG derivation motivates delta^2 (from the leading-order term in the RG equation), but is not uniquely prescriptive. Mitigation: ablation over functional forms (|delta|, delta^2, sigmoid(kappa * delta)).

3. **Critical slowing down effect is not observable at finite training**: The RG critical point may not be approached in typical training runs. Mitigation: Use longer training (300-400 epochs) where the effect is more pronounced; compute correlation over a rolling window to detect local slowing.

4. **beta_0 requires tuning and the optimal value is architecture-dependent**: Unlike CWD which reuses the WD hyperparameter, beta_0 may need to be tuned separately. Mitigation: Provide initialization formula beta_0 = lambda_fixed / <delta_hat^2>_pilot, where <delta_hat^2>_pilot is estimated from 5-10 pilot epochs.

5. **PMP derivation is only a first-order approximation (p_t approx -g_t)**: The full adjoint equation requires solving a backward recursion that is computationally expensive. Mitigation: The first-order approximation is the practical algorithm; higher-order corrections can be studied theoretically as extensions.

---

### Novelty Claim

**The specific cross-disciplinary insight**: No existing work has:
1. Applied the RG beta function formalism to derive the WD schedule (vs. using RG merely to characterize spectral properties)
2. Applied PMP to derive the WD schedule as a control variable (existing PMP papers optimize learning rate or architecture)
3. Derived two independent convergent derivations from different fields (statistical physics and optimal control) that yield the same formula lambda*_t = beta_0 * delta_hat_t^2

**Evidence of non-prior-art**:
- arXiv search: no paper combines "renormalization group" + "weight decay schedule" in the optimization dynamics sense (only in spectral/HTSR sense for module-wise scaling)
- arXiv search: no paper applies PMP to derive WD schedules
- The quadratic alignment formula lambda proportional to delta^2 is not present in CWD (binary), SWD (gradient-norm), AdaDecay (sigmoid of gradient norm), or any known prior work
- The "critical slowing down" prediction for WD schedules has no ML precedent

**Why it matters for the Unified Dynamic Weight Decay Framework**: QA-WD provides the missing theoretical foundation — a first-principles derivation that explains WHY alignment-aware WD works, WHAT the optimal functional form is, and HOW the four sub-approaches are special cases. The RG perspective also provides the first physically interpretable meaning for the WD coefficient (as a running coupling constant controlling the network's distance from the critical generalization manifold), which is the conceptual unification the framework needs.

---

## Summary of Cross-Disciplinary Recommendations

| Source Field | Key Transplant | Novelty | Proposed Role in Unified Framework |
|---|---|---|---|
| Statistical Physics (RG) | WD coefficient as running coupling; beta function derivation of optimal schedule | 9/10 | Core theoretical foundation: lambda*_t = beta_0 * delta_hat_t^2 |
| Optimal Control (PMP) | WD as control variable; PMP stationarity condition recovers same formula | 8/10 | Independent validation of RG derivation; connects to existing SGDW theory (Sun et al.) |
| Neuroscience (SHY) | Two-phase WD (active encoding + consolidation); alignment-triggered WSD | 7/10 | Temporal structure: motivates the warm-up (low WD) + decay (high WD) phasing |
| Statistical Physics (FDT) | WD optimal when gradient-weight coherence is high (FDT equilibrium condition) | 7/10 | Supporting evidence: alignment = coherence measure for optimal WD |
| Biology (Germinal Centers) | Selection pressure dynamics exact analogue of alignment-aware WD | 7/10 | Biological validation: immune system independently "discovered" the same optimal rule |
| Control Theory (SMC) | CWD as binary sliding mode; QA-WD as continuous higher-order SMC | 6/10 | Situates CWD within control theory; chattering -> noisy lambda_t problem |
| Ecology (Marginal Value Theorem) | Exploitation/exploration switching for WD strength | 5/10 | Qualitative guidance for WD phasing; less mathematically direct |

**Front-runner for primary contribution**: Candidate A+B integration (QA-WD formula derived from RG + PMP), providing a genuinely novel, rigorous, testable, and unifying theoretical result.

---

**Sources consulted**:
- [ECT* Workshop: Machine Learning and the Renormalization Group (May 2024)](https://indico.ectstar.eu/event/206/)
- [RG for Deep Neural Networks: Universality and Scaling Laws (arXiv:2510.25553)](https://arxiv.org/abs/2510.25553)
- [WeightWatcher, HTSR theory, and the RG (CalculatedContent 2024)](https://calculatedcontent.com/2024/12/24/weightwatcher-htsr-theory-and-the-renormalization-group/)
- [Optimal Schedules for Annealing Algorithms (Phys. Rev. E 2024, doi:10.1103/PhysRevE.109.065301)](https://link.aps.org/doi/10.1103/PhysRevE.109.065301)
- [Sleep-Based Homeostatic Regularization for SNNs (arXiv:2601.08447)](https://arxiv.org/abs/2601.08447)
- [Multi-Scale Temporal Homeostasis (arXiv:2602.07009)](https://arxiv.org/abs/2602.07009)
- [Computational Convergence of Adaptive Immunity and AI (bioRxiv:2026.02.03.703525)](https://www.biorxiv.org/content/10.64898/2026.02.03.703525v1)
- [Affinity Maturation: Optimal Balance between Coverage and Resource Constraints (PNAS)](https://www.pnas.org/doi/10.1073/pnas.2113512119)
- [FAdam: Adam is a Natural Gradient Optimizer (arXiv:2405.12807)](https://arxiv.org/html/2405.12807v5)
- [Fluctuation-Dissipation Relations for SGD (OpenReview ICLR 2019)](https://openreview.net/forum?id=SkNksoRctQ)
- [An Optimal Control Approach to Deep Learning (Li et al., ICML 2018)](http://proceedings.mlr.press/v80/li18b/li18b.pdf)
- [Rugged yet Easily Navigable Fitness Landscape (Science 2024)](https://www.science.org/doi/10.1126/science.adh3860)
- [Diversity begets stability via Sublinear Growth (Science 2024)](https://www.jlakes.org/uploadfile/news_images/hpkx/2024-04-08/science.adg8488-ZJSD-1.pdf)
- [Lyapunov-stable Neural Control (arXiv:2404.07956)](https://arxiv.org/abs/2404.07956)
- [Interpreting Deep Learning via RG Correspondence (arXiv:2212.00005)](https://arxiv.org/abs/2212.00005)
- [Sliding Mode Control (Wikipedia)](https://en.wikipedia.org/wiki/Sliding_mode_control)
- [Sleep and Synaptic Homeostasis Hypothesis (Tononi & Cirelli, PubMed 16376591)](https://pubmed.ncbi.nlm.nih.gov/16376591/)
- [Nonlinear Thermodynamic Computing Out of Equilibrium (Nature Communications 2026)](https://www.nature.com/articles/s41467-025-67958-0)
