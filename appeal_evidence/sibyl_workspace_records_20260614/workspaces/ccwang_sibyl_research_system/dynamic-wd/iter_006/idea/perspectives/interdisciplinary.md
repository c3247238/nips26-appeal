# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Homeostatic Synaptic Scaling (Turrigiano 1998, updated 2024-2025)** -- Neurons multiplicatively rescale all synaptic weights by a single factor to maintain a target firing rate. The factor is activity-dependent: when firing is too high, weights are globally scaled down; when too low, scaled up. Koesters et al. (2024, *The Neuroscientist*) show scaling is actually non-uniform/divergent across synapses, suggesting a richer per-synapse modulation mechanism. This is structurally analogous to weight decay as a global norm controller, with the "firing rate sensor" mapping to an alignment/norm signal.

2. **Sleep-Based Synaptic Renormalization (SHy Hypothesis, arXiv:2601.08447)** -- During sleep, high-activity synapses undergo more aggressive down-scaling than low-activity ones. This activity-weighted, selective renormalization preserves important connections while pruning noise. Direct parallel to alignment-aware weight decay where decay strength is modulated by the "activity" (alignment) of each parameter.

3. **Two-Factor Synaptic Consolidation (PNAS 2025)** -- Reconciles robustness with pruning and homeostatic scaling via a self-supervised consolidation model. Memory strength determines which connections are protected from pruning vs. subjected to multiplicative scaling. Structurally maps to selective weight decay where "well-aligned" parameters (useful signal) are protected while "misaligned" parameters (noise) are aggressively decayed.

4. **Loss of Plasticity in Continual Learning (Nature 2024)** -- L2 regularization (weight decay) is shown to be one of few mechanisms that maintains plasticity over thousands of sequential tasks. This establishes weight decay as a homeostatic mechanism for *learning capacity preservation*, not just norm control.

5. **Free Energy Principle / Active Inference (Friston)** -- Biological agents minimize variational free energy F = E_q[log q(z) - log p(x,z)], which decomposes into accuracy (data fit) minus complexity (KL divergence from prior). Weight decay is formally the complexity penalty in this framework. Active inference suggests the penalty should be *adaptive*: increasing when the model is confident (low uncertainty) and decreasing during exploration (high uncertainty).

#### Physics / Information Theory / Thermodynamics

6. **Simulated Annealing Temperature Scheduling (Kirkpatrick 1983; Optimal schedules arXiv:2402.14717, 2024)** -- Temperature in SA controls exploration-exploitation tradeoff via Boltzmann acceptance probability. Optimal annealing schedules in multidimensional parameter spaces have been derived using non-equilibrium statistical mechanics (Sivak & Crooks framework). The temperature is the regularization strength analogue; optimal scheduling depends on the entropy of the current state distribution -- a thermodynamic analogue to alignment informativeness.

7. **Entropy Production & Dissipation Optimization (Karbowski 2024, *Entropy*)** -- Information thermodynamics unifies energy dissipation and information processing in neural systems. Local entropy balance involves both dissipation and information flow. Weight decay can be viewed as a dissipation operator that removes thermodynamic entropy (parameter uncertainty) from the optimization trajectory. Optimal dissipation rate depends on the current information content of the parameters -- high-information parameters should dissipate less.

8. **Landauer's Principle (Chattopadhyay et al. 2025, *Rep. Prog. Phys.*)** -- Information erasure has a minimum thermodynamic cost (kT ln 2 per bit). Correlations between parameters can reduce this cost. This suggests that weight decay should be cheaper (lower regularization needed) when parameters are correlated (aligned directions), and more expensive (stronger decay needed) when parameters are independent (misaligned).

9. **Replicated Simulated Annealing & Replica Method** -- Uses the replica trick from statistical physics to regularize non-convex landscapes. Dense regions of good configurations (flat minima) are amplified while isolated optima are suppressed. This is conceptually identical to weight decay promoting flat minima, but the replica method provides a principled way to adaptively tune regularization strength based on the landscape geometry around the current solution.

#### Biology / Evolution

10. **Adaptive Mutation Rate Regulation (PNAS 2025; Nature Ecol. Evol. 2024)** -- Organisms dynamically regulate mutation rate based on fitness landscape position: high mutation near fitness valleys (exploration), low mutation near peaks (exploitation). The "adaptive state" (distance from optimum) determines whether mutations help or harm. Direct structural analogy: weight decay = mutation rate; alignment = fitness gradient signal; decay should increase when far from optimum (low alignment) and decrease near optimum (high alignment).

11. **Evolution of Evolvability (PNAS 2025)** -- Multiple pathways to evolvability emerge: (a) evolved mutational landscapes for known environments (= scheduled WD), (b) higher mutation rates for novel environments (= adaptive WD). This dual pathway maps exactly to the scheduling vs. alignment-aware WD dichotomy in the ML literature.

12. **Fitness-Dependent Mutation Rate (Variable Mutation Rates as Adaptive Strategy, PLoS ONE)** -- There is no optimal universal mutation rate; organisms adjust it to momentary adaptive needs. Under environmental stress, higher mutation rates enable rapid adaptation. The structural correspondence: environmental stress = high loss / low alignment, mutation rate = weight decay strength.

#### Control Theory

13. **Sliding Mode Control (SMC) with Adaptive Switching Gain (2024-2025)** -- CWD (ICLR 2026) already identifies its connection to sliding mode control. SMC drives the system state to a sliding surface and maintains it there via high-frequency switching. The chattering problem (oscillation around the sliding surface due to discrete switching) is structurally identical to CWD's binary sign-alignment mask. Solutions: (a) boundary layer smoothing (= continuous alignment modulation), (b) adaptive gain (= alignment-dependent decay strength), (c) higher-order SMC (= using higher-order alignment statistics). Variable Gains SMC uses gains proportional to tracking error, avoiding overestimation of disturbances.

14. **Adaptive Gain Scheduling** -- Gain-scheduled controllers adapt their parameters based on the current operating point of a nonlinear system. The operating point in the WD context is the (alignment, norm, gradient magnitude) state. Gain scheduling theory provides formal guarantees (Lyapunov stability, H-infinity robustness) for parameter-varying controllers -- directly applicable to proving convergence of alignment-aware WD.

15. **Feedback Linearization** -- Transforms a nonlinear system into a linear one via state-dependent coordinate change. The Radial Tug-of-War identified by AdamO (2026) is precisely a nonlinear coupling that feedback linearization can decouple. Weight decay's role as a "radial controller" while gradients act as "tangential controller" is a canonical feedback linearization decomposition.

### Cross-Disciplinary Gaps

1. **Homeostatic synaptic scaling -> weight decay**: The divergent (non-uniform) scaling discovered by Koesters et al. (2024) has NOT been applied to ML weight decay. All current WD methods use either uniform decay or binary masks, not a principled per-parameter scaling law derived from homeostatic regulation.

2. **Optimal annealing schedules from non-equilibrium thermodynamics -> WD scheduling**: The Sivak-Crooks framework for optimal temperature schedules has NOT been adapted to derive optimal WD schedules. This could yield thermodynamically-principled WD scheduling.

3. **Fitness-dependent mutation rate -> alignment-aware WD**: While the metaphor exists, the precise mathematical mapping (mutation rate regulation ODEs -> WD adaptation ODEs) has not been formalized.

4. **Information thermodynamics of dissipation -> WD as entropy removal**: The Landauer-bound perspective on weight decay cost has NOT been applied to derive minimum-dissipation WD strategies.

---

## Phase 2: Initial Candidates

### Candidate A: Homeostatic Weight Decay (from Neuroscience -- Synaptic Homeostasis)

- **Source principle**: Homeostatic synaptic scaling maintains neural circuit stability by sensing a global activity variable (firing rate) and multiplicatively rescaling all synaptic weights to restore a target set-point. The scaling factor is computed as s = r_target / r_actual, where r is the firing rate (a function of all synaptic weights and input statistics). Crucially, recent work (Koesters 2024) shows scaling is non-uniform: the factor depends on each synapse's contribution to the activity deviation.

- **Structural correspondence**:
  - Firing rate r(w) <-> Training loss gradient norm ||nabla f(w)|| or effective learning rate eta_eff
  - Target firing rate r_target <-> Target weight norm tau (as in AdamWN) or target gradient-to-weight ratio (as in Defazio 2025)
  - Scaling factor s_i = r_target / r_actual * alpha_i <-> Weight decay strength lambda_i(t) = lambda_0 * (tau / ||w||) * phi(delta_i), where phi(delta_i) is a per-parameter modulation based on alignment delta_i
  - Homeostatic set-point control <-> Lyapunov stability of norm trajectory

  The ODE for homeostatic scaling: ds/dt = -beta * (r(w*s) - r_target) becomes, in the WD context: d(lambda)/dt = -beta * (||w|| - tau) * g(alignment), a feedback law that increases decay when norm exceeds target and the parameter is poorly aligned.

- **Hypothesis**: A homeostatic WD rule lambda_i(t) = lambda_0 * max(0, ||w_i|| - tau_i) / ||w_i|| * (1 - |delta_i|)^p, where tau_i is a per-layer target norm derived from initial norm statistics (analogous to the developmental firing rate set-point), will: (a) subsume constant WD (tau=0, p=0), CWD (tau=0, binary delta threshold), and AdamWN (arbitrary tau, p=0) as special cases; (b) maintain Lyapunov stability via the homeostatic set-point mechanism; (c) achieve better generalization by selectively decaying parameters that both exceed target norm AND are misaligned.

- **Why not just a metaphor**: The mathematical structure is preserved: both systems use negative feedback on a global variable (firing rate / norm) with per-element modulation (synapse activity / alignment). The stability guarantee in both systems comes from the same Lyapunov argument: V = (r - r_target)^2 decreases along trajectories because the feedback is sign-correct. The non-uniform scaling discovered in neuroscience (Koesters 2024) directly predicts that uniform WD should be suboptimal compared to per-parameter alignment-modulated WD.

- **Novelty estimate**: 7/10 -- The CWD paper mentions sliding mode (a control theory analogy) but nobody has formalized the homeostatic synaptic scaling correspondence with per-parameter divergent scaling.

### Candidate B: Thermodynamic Dissipation-Optimal Weight Decay (from Statistical Physics)

- **Source principle**: In non-equilibrium thermodynamics, the optimal protocol for driving a system between two states minimizes total entropy production (dissipation). The Sivak-Crooks (PRL 2012) framework shows that optimal control protocols follow geodesics in a thermodynamic metric space, where the metric depends on the system's response functions (susceptibilities). For simulated annealing, the optimal temperature schedule minimizes the excess work (wasted computation) by adapting the cooling rate to the local heat capacity of the landscape.

- **Structural correspondence**:
  - Temperature T <-> Weight decay strength lambda (both control the regularization/exploration tradeoff)
  - Heat capacity C(T) <-> "WD susceptibility" chi(lambda) = d||w||^2 / d(lambda) (how much the norm changes per unit WD change)
  - Entropy production sigma <-> "Optimization inefficiency" = gradient norm after accounting for WD contribution (wasted gradient steps fighting WD)
  - Optimal cooling rate dT/dt proportional to 1/sqrt(C(T)) <-> Optimal WD schedule d(lambda)/dt proportional to 1/sqrt(chi(lambda))
  - Thermodynamic length integral L = integral sqrt(g_ij * dx_i/dt * dx_j/dt) dt <-> Optimization path length in parameter space under WD

  The key mathematical object is the Fisher information metric (which appears in both thermodynamic and information-geometric optimization). The optimal WD schedule minimizes: integral_0^T [lambda(t) * ||w(t)||^2 - f(w(t))]^2 / sigma^2(lambda(t)) dt, where sigma^2 is the variance of the WD effect.

- **Hypothesis**: A thermodynamically-derived WD schedule lambda*(t) that adapts based on the local "susceptibility" chi(t) = d||w_t||^2/d(lambda) will: (a) produce less wasted computation (lower cumulative ||lambda * w - projection of lambda*w onto gradient direction||); (b) be derivable as a geodesic in a Fisher-information metric on the (lambda, w) space; (c) connect to Pontryagin's maximum principle (PMP-WD in the proposal) via the minimum entropy production principle.

- **Why not just a metaphor**: The mathematical isomorphism is precise. The Fisher information metric appears in both: (1) optimal thermodynamic protocols (Sivak-Crooks), and (2) natural gradient descent in optimization. The minimum entropy production principle in thermodynamics IS the minimum regret principle in online convex optimization (Cesa-Bianchi & Lugosi). The Landauer bound (minimum energy to erase information) maps to the minimum WD needed to "erase" a parameter's contribution (drive it to zero), which is exactly ||w||^2 / (2 * learning_rate) -- the actual WD loss term.

- **Novelty estimate**: 8/10 -- The connection between simulated annealing and learning rate scheduling is well-known, but applying the Sivak-Crooks optimal protocol framework specifically to WD scheduling via Fisher information geometry has NOT been done. The "WD susceptibility" concept is novel.

### Candidate C: Fitness-Adaptive Mutation Rate Control (from Evolutionary Biology)

- **Source principle**: Organisms regulate mutation rates based on their position on the fitness landscape. Near fitness peaks (well-adapted), low mutation rates preserve good solutions. In fitness valleys (maladapted), high mutation rates enable exploration. The mutation rate mu(t) adapts via a second-level selection: lineages with mutation rates that match the landscape topology outcompete those with fixed rates. The mathematical framework (Nature Ecol. Evol. 2024) models this as a coupled ODE system: d(fitness)/dt = f(mu, landscape) and d(mu)/dt = g(fitness_gradient, mu), where g increases mu when the fitness gradient is small (flat/valley) and decreases mu when gradient is large (slope toward peak).

- **Structural correspondence**:
  - Mutation rate mu <-> Weight decay strength lambda
  - Fitness f <-> Negative training loss -L(w)
  - Fitness gradient |nabla f| <-> Gradient-weight alignment |cos(g, w)| (= alignment delta)
  - Fitness peak (adapted) <-> Good alignment (gradient aligned with weight direction -> decay is useful)
  - Fitness valley (maladapted) <-> Poor alignment (gradient orthogonal to weight -> decay is wasteful)
  - Mutation rate ODE: d(mu)/dt = -alpha * (|nabla f| - threshold) <-> WD adaptation ODE: d(lambda)/dt = -alpha * (delta_t - delta_target)

  HOWEVER, the mapping inverts at a subtle point: in evolution, high mutation is for exploration (moving away from current state), while in WD, high decay is for regularization (shrinking toward origin). The correct mapping reverses: high decay when alignment is high (parameter aligned with gradient -> WD is "constructive regularization" per Sun et al. CVPR 2025), low decay when alignment is low (parameter orthogonal to gradient -> WD wastes effort).

  After correcting: lambda(t) increases when delta_t is high (analogous to low mutation rate at fitness peak = preserving + regularizing). lambda(t) decreases when delta_t is low (analogous to high mutation rate in valley = allowing exploration).

  Wait -- this matches CWD's direction (decay when sign-aligned) but with continuous modulation instead of binary switching.

- **Hypothesis**: A fitness-adaptive WD rule derived from the mutation rate ODE: d(lambda)/dt = alpha * (delta_hat_t - delta_target) * (tau - ||w_t||), where delta_hat_t is the stochastic alignment proxy and delta_target is a moving-average baseline, will exhibit: (a) automatic transition between exploration (low WD) and exploitation (high WD) phases; (b) population-level robustness (multi-seed stability) analogous to evolvability; (c) convergence guarantees derivable from the evolutionary game theory framework (replicator dynamics).

- **Why not just a metaphor**: The coupled ODE structure is preserved. The "second-level selection" on mutation rates (fitness of the mutation rate itself) maps to a bilevel optimization on lambda: the inner problem is SGD with WD, the outer problem optimizes lambda for generalization. CWD (ICLR 2026) already formalizes this as a bilevel Pareto-optimality problem. The evolutionary framework provides an additional insight: the rate of adaptation of mu itself (the "evolvability of evolvability") determines how quickly the system can switch between exploration and exploitation regimes.

- **Novelty estimate**: 6/10 -- The evolutionary analogy for learning rate adaptation is well-explored (evolutionary strategies, CMA-ES). The specific mapping to WD alignment modulation is somewhat novel but less mathematically rigorous than Candidates A and B.

---

## Phase 3: Self-Critique

### Against Candidate A (Homeostatic Weight Decay)

- **Shallow analogy attack**: Is "firing rate -> norm" a deep correspondence or just vocabulary mapping? ASSESSMENT: The correspondence is structural. Both are negative feedback systems operating on a scalar integral of the system state. The mathematical form (multiplicative scaling to maintain a set-point) is identical. The non-uniform scaling prediction (Koesters 2024) is a testable new claim that goes beyond the standard analogy. A neuroscientist would agree that the homeostatic control loop structure is preserved, not just the language. **VERDICT: Structural correspondence is genuine.**

- **Scale mismatch attack**: Synaptic scaling operates on ~10^4 synapses with a single global sensor (calcium/firing rate). DNNs have ~10^6-10^9 parameters. Does the mechanism scale? ASSESSMENT: The key insight is that homeostatic scaling is *modular* (Wen & Turrigiano, PNAS 2025) -- different homeostatic mechanisms operate on different subnetworks independently. This maps well to per-layer or per-module WD (like AlphaDecay). The scaling is not a fundamental issue because the feedback loop operates on aggregated statistics (per-layer norm), not individual parameters. **VERDICT: Scale mismatch is addressable.**

- **Prior transplant check**: Has homeostatic plasticity been applied to weight decay? Searched: "homeostatic plasticity weight decay neural network optimization". The sleep-based homeostatic regularization paper (arXiv:2601.08447) applies SHy to spiking neural networks but uses weight decay as a crude implementation tool, not as the analogized mechanism. No paper formalizes the divergent synaptic scaling -> per-parameter WD modulation correspondence. **VERDICT: Novel application.**

- **Testability attack**: Can we design an experiment that distinguishes "homeostatic WD works because of the homeostatic feedback loop" from "it works because it's just another adaptive WD"? YES: the diagnostic test is whether *restoring the target norm set-point after perturbation* (the defining property of homeostatic systems) actually matters. Perturb the norm (e.g., by reinitializing a subset of weights) mid-training and measure recovery speed under homeostatic WD vs. other adaptive WD methods. A truly homeostatic system should recover faster because it explicitly targets the set-point. **VERDICT: Testable.**

- **VERDICT: STRONG** -- Structural correspondence is genuine, non-uniform scaling prediction is novel, testable with diagnostic experiment.

### Against Candidate B (Thermodynamic Dissipation-Optimal WD)

- **Shallow analogy attack**: "Temperature = WD" is a common loose analogy. Is the Sivak-Crooks framework actually applicable? ASSESSMENT: The mathematical structure requires the system to be near equilibrium (linear response regime) for the thermodynamic metric to be valid. SGD training is far from equilibrium. However, recent work on stochastic thermodynamics (Wolpert 2024, PNAS) extends these principles to far-from-equilibrium systems. The Fisher information metric connection is rigorous -- it appears in both natural gradient descent and thermodynamic optimal protocols as the Riemannian metric on parameter space. **VERDICT: Deeper than surface analogy, but the near-equilibrium assumption is a real limitation.**

- **Scale mismatch attack**: Thermodynamic optimal protocols are typically derived for systems with a few degrees of freedom. DNN parameter spaces have millions. ASSESSMENT: The key insight is that the relevant "thermodynamic" variables are aggregated quantities (loss, total norm, alignment), not individual parameters. The optimal protocol operates on these low-dimensional projections. This is analogous to how thermodynamics works despite having 10^23 particles -- because temperature and pressure are sufficient statistics. **VERDICT: Addressable via coarse-graining.**

- **Prior transplant check**: Has the Sivak-Crooks framework been applied to ML optimization? Searched: "Sivak Crooks optimal protocol machine learning neural network". Found: connections to optimal learning rate schedules and annealing (AnnealSGD, entropy SGD), but NO direct application to WD scheduling. The "thermodynamic length" concept has appeared in information geometry for natural gradient methods but not for WD. **VERDICT: Novel application to WD.**

- **Testability attack**: Can we distinguish "thermodynamic WD works because it minimizes dissipation" from "it's just a good heuristic"? The diagnostic test: compute the actual "excess work" (wasted gradient computation) and "thermodynamic length" of different WD schedules and check whether the dissipation-optimal schedule is also the empirically best schedule. If the correlation between dissipation efficiency and test accuracy is high, the thermodynamic principle is the active ingredient. **VERDICT: Testable but requires defining "dissipation" precisely in the ML context.**

- **VERDICT: MODERATE-to-STRONG** -- Genuine mathematical connection via Fisher information, but near-equilibrium assumption is a weakness. Novel application. Testability requires careful metric definition.

### Against Candidate C (Fitness-Adaptive Mutation Rate)

- **Shallow analogy attack**: "Mutation rate = WD" is a very high-level analogy. Is there a precise mathematical isomorphism? ASSESSMENT: The coupled ODE structure (d(fitness)/dt, d(mu)/dt) is shared, but the fitness landscape in evolution is static (or slowly changing), while the loss landscape in training is dynamically shaped by the optimizer. The "second-level selection" on mutation rates requires population-level competition, which has no direct analogue in single-trajectory SGD. Multi-seed experiments could be seen as a "population" but this stretches the analogy. **VERDICT: Moderate -- the ODE structure is shared but the population-level mechanism doesn't transfer cleanly.**

- **Scale mismatch attack**: Evolutionary mutation rate regulation operates over generations (10^3-10^6), while WD adaptation operates over iterations (10^3-10^5). The timescale is comparable. ASSESSMENT: Timescale match is reasonable. However, the dimensionality of the "genotype" (few key traits) vs. DNN parameters (millions) is very different. In evolution, mutation rate is typically a single scalar or per-gene; in WD, we want per-parameter adaptation. **VERDICT: Moderate mismatch.**

- **Prior transplant check**: Evolutionary strategies (ES) and CMA-ES are well-established in ML optimization. Has the specific mutation-rate-adaptation ODE been applied to WD? Found no direct application. However, the "meta-learning the learning rate" literature (e.g., hypergradient descent) is conceptually similar. **VERDICT: Partially explored territory.**

- **Testability attack**: What distinguishes "evolutionary WD" from "adaptive WD with alignment feedback"? The evolutionary framework specifically predicts *hysteresis*: the mutation rate (WD) should lag behind fitness changes due to the second-level selection timescale. This is testable: inject a sudden change in the loss landscape (e.g., switch datasets mid-training) and check whether the WD adaptation shows the predicted lag. **VERDICT: Testable but the prediction is not very distinctive.**

- **VERDICT: MODERATE** -- Useful conceptual framework but less mathematically rigorous than A and B. Population-level mechanism doesn't map cleanly to single-trajectory optimization.

---

## Phase 4: Refinement

### Dropped
- **Candidate C (Fitness-Adaptive Mutation Rate)**: While the conceptual framework is appealing, the population-level selection mechanism that drives adaptive mutation rates does not have a clean single-trajectory analogue. The ODE structure is shared but shallow. The existing bilevel optimization perspective (CWD, ICLR 2026) already captures the "optimize the optimizer" idea more rigorously. Retain as a supplementary framing but not as the primary cross-disciplinary contribution.

### Strengthened

#### Candidate A (Homeostatic Weight Decay) -- PRIMARY
Formalized structural correspondence:

**Mapping Table:**
| Neuroscience (Homeostatic Scaling) | ML (Weight Decay) |
|---|---|
| Synapse weight w_syn | Network parameter w |
| Firing rate r = phi(sum w_syn * x) | Training signal: ||w|| or ||nabla f|| |
| Target firing rate r_target | Target norm tau (AdamWN) or target gradient-to-weight ratio (Defazio) |
| Scaling factor s = r_target/r_actual | WD coefficient lambda(t) |
| Activity-dependent modulation alpha_i | Alignment modulation phi(delta_i) |
| Calcium sensor / TNF-alpha pathway | Alignment proxy delta_hat_t |
| Developmental set-point calibration | Initial norm statistics {||w_0^l||}_{l=1..L} |
| Modular independence (Wen 2025) | Per-layer/per-module WD independence |

**Key novel prediction from the analogy**: The divergent (non-uniform) scaling factor discovered by Koesters et al. (2024) predicts that the optimal WD modulation is NOT uniform across parameters even within a layer. The modulation should depend on each parameter's individual "activity contribution" to the deviation from the homeostatic set-point. In WD terms: lambda_i should depend on w_i's contribution to the alignment deviation, not just on the global alignment signal.

This can be formalized as: lambda_i(t) = lambda_0 * max(0, (||w^l|| - tau^l) / ||w^l||) * (1 + gamma * (delta_i - mean(delta)))

where delta_i = sign(w_i) * sign(g_i) is the per-parameter sign alignment and gamma controls the divergent scaling strength. When gamma=0, this reduces to uniform norm-targeted WD (AdamWN). When tau=0 and the norm factor is dropped, the sign part reduces to CWD. The unified formula subsumes both as special cases.

**Diagnostic experiment**: Train with homeostatic WD and then perturb the weight norm (e.g., multiply all weights by 2x at epoch 50). Measure:
1. How quickly the norm returns to the set-point under homeostatic WD vs. constant WD vs. CWD
2. Whether test accuracy during recovery is better preserved under homeostatic WD
3. Whether the recovery dynamics match the exponential return predicted by the homeostatic ODE

#### Candidate B (Thermodynamic Dissipation-Optimal WD) -- SECONDARY (STRENGTHENED)
The key refinement: instead of requiring full near-equilibrium assumption, use the **thermodynamic uncertainty relation** (TUR), which holds far from equilibrium:

(signal-to-noise ratio)^2 <= 2 * (entropy production rate)

In the WD context: the precision of the alignment signal delta_hat is bounded by the amount of "optimization dissipation" (wasted gradient steps). This yields:

Var(delta_hat_t) >= |mean(delta_hat_t)|^2 / (2 * sigma_t)

where sigma_t is the entropy production rate. This provides a fundamental lower bound on how informative the alignment signal can be, given the current WD-induced dissipation. If the alignment signal variance exceeds this bound, additional alignment information is available and can be exploited by adaptive WD. If it's at the bound, alignment-aware WD cannot outperform constant WD -- explaining the "narrow certified band" prediction.

### Selected Front-Runner: **Candidate A (Homeostatic Weight Decay)**

Reasons:
1. Most concrete structural correspondence with testable novel predictions
2. Directly addresses Gap 3 (continuous alignment modulation) and Gap 4 (connections between sub-approaches) from the literature survey
3. Subsumption of CWD and AdamWN as special cases is formally provable
4. Diagnostic experiment is clean and distinctive
5. The Candidate B insight (thermodynamic uncertainty relation as a bound on alignment informativeness) naturally integrates as a theoretical result WITHIN the homeostatic framework -- it explains WHEN the homeostatic feedback loop has sufficient signal to outperform constant WD

---

## Phase 5: Final Proposal

### Title
**Homeostatic Weight Decay: A Neuroscience-Inspired Unified Framework for Dynamic Parameter Regularization**

### Source Principle
Homeostatic synaptic scaling in biological neural circuits: neurons maintain stable activity by multiplicatively rescaling synaptic weights via a negative feedback loop. The scaling factor depends on a global activity sensor (firing rate / calcium concentration) with per-synapse modulation based on individual activity contribution. Recent work (Koesters et al. 2024; Wen & Turrigiano 2025) reveals that this scaling is (a) non-uniform across synapses (divergent scaling) and (b) modular across subcircuits (different homeostatic mechanisms independently regulate different network features).

### Structural Correspondence
The formal mapping preserves the mathematical structure of the negative feedback loop:

1. **Set-point control**: Both systems maintain a target variable via proportional feedback.
   - Neuroscience: d(s)/dt = -beta * (r(w*s) - r_target)
   - Weight decay: d(lambda)/dt = -beta * (N(w,lambda) - N_target)
   where N(w,lambda) is the "regularization state" (e.g., ||w||, gradient-to-weight ratio, spectral norm).

2. **Per-element modulation**: The feedback is not uniform but depends on each element's "contribution" to the deviation.
   - Neuroscience: s_i = s_global * alpha_i, where alpha_i depends on synapse i's activity relative to the population
   - Weight decay: lambda_i = lambda_global * phi(delta_i), where delta_i is parameter i's alignment contribution

3. **Modular independence**: Homeostatic mechanisms operate independently across subcircuits.
   - Neuroscience: Visual cortex synaptic and intrinsic homeostasis are independently recruited
   - Weight decay: Per-layer/per-module WD coefficients (AlphaDecay, SPD)

4. **Stability guarantee**: Both systems are Lyapunov-stable with V = (state - target)^2 as the Lyapunov function.

### Hypothesis
A homeostatic WD framework with the update rule:

lambda_i(t+1) = clip(lambda_i(t) + alpha * (||w^l_t|| - tau^l) / ||w^l_t|| * phi(delta_i(t)), lambda_min, lambda_max)

where phi(delta) = sigmoid(kappa * (delta - delta_threshold)) is a continuous alignment modulation function, will:

1. **Unify existing methods**: Constant WD (alpha=0), CWD (phi=step function, tau=0), AdamWN (phi=1, arbitrary tau), SWD (tau proportional to 1/||g||) are all recoverable as limit cases.

2. **Predict when alignment-aware WD helps**: The thermodynamic uncertainty relation bounds the informativeness of the alignment signal. When Var(delta_hat) >> |mean(delta_hat)|^2 / (2*sigma), the homeostatic mechanism has sufficient signal -- predicting improvement on non-BN architectures and under high WD regimes (matching the existing empirical pattern).

3. **Exhibit homeostatic recovery**: After norm perturbation, the system will exponentially return to the set-point, preserving test accuracy better than non-homeostatic alternatives.

### Method
1. Implement the homeostatic WD optimizer as a wrapper around SGD/AdamW
2. The per-layer target norm tau^l is initialized from the network's initial norm statistics (analogous to developmental calibration of firing rate set-points)
3. The alignment modulation function phi is parameterized by a single hyperparameter kappa controlling the sensitivity (analogous to the calcium sensor gain)
4. The adaptation rate alpha is set to ensure the Lyapunov decrease condition is satisfied (computable from L-smoothness and bounded gradient assumptions)

### Diagnostic Experiment
**The homeostatic recovery test**: This is the key experiment that distinguishes homeostatic WD from other adaptive WD methods.

Protocol:
1. Train ResNet-20 on CIFAR-10 with homeostatic WD, constant WD, CWD, and SWD for 100 epochs
2. At epoch 50, multiply all weights in conv layers by 2x (norm perturbation)
3. Continue training for 50 more epochs
4. Measure: (a) norm recovery time (epochs to return within 10% of pre-perturbation norm), (b) test accuracy recovery curve, (c) alignment trajectory post-perturbation

Prediction: Homeostatic WD recovers fastest because its feedback loop explicitly targets the pre-perturbation set-point. Constant WD will over-decay (it doesn't know the target), CWD may oscillate (binary switching), and SWD's gradient-norm-based adaptation is orthogonal to the norm perturbation signal.

### Experimental Plan
All experiments use seeds {42, 123, 456}, reporting mean +/- std.

**Standard evaluation** (within project constraints):
- CIFAR-10: ResNet-20, VGG-16-BN -- compare homeostatic WD vs. constant WD, CWD, SWD, AdamWN
- CIFAR-100: ResNet-20, VGG-16-BN -- same comparison
- ImageNet: ResNet-50 -- homeostatic WD vs. constant WD, CWD (may require 4-8 hours per run)
- Metrics: test accuracy, weight norm trajectory, alignment trajectory, CSI, AIS, BEM

**Diagnostic experiments**:
- Homeostatic recovery test (described above)
- BN vs. non-BN comparison: test whether homeostatic WD provides larger improvement on non-BN architectures (predicted by the thermodynamic bound)
- Ablation: vary kappa (alignment sensitivity) and alpha (adaptation rate) to trace the Pareto frontier between CWD (kappa->infinity) and AdamWN (kappa=0)

Target: pilot experiments on CIFAR-10/ResNet-20 in ~15 minutes each; full comparison in ~1 hour.

### Risk Assessment
1. **The alignment signal may be too noisy** for the homeostatic feedback loop to improve over constant WD in practice, especially with batch normalization (where the "certified band" is narrow per the proposal's H5). MITIGATION: The thermodynamic uncertainty bound predicts exactly when this happens, and we validate on both BN and non-BN architectures.

2. **Per-parameter WD may be too expensive** to compute at scale. MITIGATION: Use per-layer aggregation (matching AlphaDecay's granularity) as the default, with per-parameter as a theoretical ideal.

3. **The homeostatic recovery test may not be ecologically valid** -- real training doesn't involve sudden 2x norm perturbations. MITIGATION: Also test with more realistic perturbations: learning rate warmup restart, fine-tuning after pretraining, continual learning task switches.

4. **The divergent scaling prediction from Koesters (2024) may not transfer** because biological synapse populations are fundamentally different from DNN parameter populations. MITIGATION: This is empirically testable and failing would itself be an interesting negative result about the limits of the analogy.

### Novelty Claim
The specific cross-disciplinary insight is threefold:

1. **Divergent homeostatic scaling -> per-parameter WD modulation**: The 2024 neuroscience finding that synaptic scaling is non-uniform directly predicts that optimal WD should be non-uniform within layers, modulated by per-parameter alignment. This has NOT been proposed in the ML literature.

2. **Modular homeostatic independence -> formal justification for per-layer WD**: The 2025 finding that synaptic and intrinsic homeostasis are independently recruited across subcircuits provides biological justification for per-layer WD (AlphaDecay) and predicts that different layers should use DIFFERENT homeostatic strategies (not just different decay coefficients).

3. **Thermodynamic uncertainty relation -> alignment informativeness bound**: Imported from non-equilibrium statistical physics, this provides the first fundamental lower bound on when alignment-aware WD can outperform constant WD, explaining the empirical observation that WD methods rarely outperform each other on BN architectures.

Evidence of novelty: Searched arXiv for "homeostatic weight decay", "synaptic scaling weight decay optimizer", "divergent scaling regularization" -- no matches found. The sleep-based regularization paper (arXiv:2601.08447) uses SHy for spiking networks but does not formalize the divergent scaling or apply it to standard DNN weight decay.
