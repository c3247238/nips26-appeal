# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Statistical Physics / Thermodynamics

1. **Sadrtdinov et al., 2025. "Can Training Dynamics of Scale-Invariant Neural Networks Be Explained by the Thermodynamics of an Ideal Gas?" arXiv:2511.07308** -- Develops a thermodynamic framework for SGD with weight decay on scale-invariant networks. Establishes rigorous analogies: learning rate <-> temperature, weight decay <-> pressure, parameter norm <-> volume. The key result is that SGD dynamics under weight decay follow an ideal gas law PV = NkT, where P ~ weight decay, V ~ ||w||^2, T ~ learning rate * gradient noise. Three training protocols (fixed norm, fixed effective LR, fixed LR) map to isothermal, isobaric, and isochoric processes respectively. **Critical insight for our work**: dynamic weight decay scheduling corresponds to thermodynamic process design -- choosing the optimal path through (P,V,T) space to reach a target state. The concept of "thermodynamic entropy" of stationary distributions provides a principled alternative to ad hoc metrics for comparing WD strategies.

2. **Caraffa, 2026. "Thermodynamically Optimal Regularization under Information-Geometric Constraints." arXiv:2601.17330** -- Proves that under three axioms (parametrization invariance, maximum entropy beliefs, quasi-static optimality), the unique thermodynamically optimal regularization is minimization of squared Fisher-Rao distance to a reference state. Euclidean weight decay (L2 regularization) is shown to be *structurally incapable* of guaranteeing thermodynamic optimality. **Core structural correspondence**: the ratio of Euclidean to Fisher-Rao squared distance can be arbitrarily large or small depending on local uncertainty, explaining why fixed WD coefficients are suboptimal -- they apply uniform "pressure" regardless of the local geometry of parameter space.

3. **Ahr, Biehl & Schloesser, 1999. "Weight decay induced phase transitions in multilayer neural networks." cond-mat/9901179** -- Uses equilibrium statistical mechanics to show that weight decay induces first-order phase transitions between states with long vs. teacher-matched student vectors. Additionally, a phase transition between specialized and unspecialized phases occurs. **Structural analogy**: dynamic WD can be understood as controlling which thermodynamic phase the network occupies; WD scheduling is analogous to annealing schedules that navigate phase boundaries.

4. **Richemond & Guo, 2019. "Combining learning rate decay and weight decay with complexity gradient descent." arXiv:1902.02881** -- Uses insights from statistical physics and random fields theory to introduce "complexity" as a parameter combining loss level and remaining nonconvexity. Derives novel annealing schemes for L2 regularization strength during SGD. **Key mechanism**: the "complexity" parameter is a thermodynamic state variable that should govern WD scheduling.

5. **Boursier, Pesme & Dragomir, 2025. "A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation." arXiv:2505.20172** -- Proves that gradient flow with small weight decay exhibits a two-phase behavior: (1) fast phase following unregularized flow to a manifold of critical points, then (2) slow drift phase at timescale O(1/lambda) following Riemannian gradient flow minimizing L2 norm. **Deep structural correspondence**: the grokking phenomenon is a phase transition between "kinetic" (learning-dominated) and "thermodynamic" (WD-dominated) regimes. The transition timescale 1/lambda directly maps to our Coupling Stability Index.

6. **Allahverdyan & Galstyan, 2011. "Le Chatelier Principle in Replicator Dynamics." Phys. Rev. E 84, 041117** -- Shows that the Le Chatelier principle (equilibria resist perturbations via negative feedback) can be reformulated as a majorization relation in evolutionary game theory. Defines a stability notion that generalizes evolutionary stability. **Structural correspondence to WD**: weight decay acts as the Le Chatelier mechanism in optimizer dynamics -- it provides a restoring force proportional to parameter magnitude that resists the "perturbation" caused by gradient updates. Dynamic WD can be interpreted as an adaptive Le Chatelier response where the feedback gain varies with the alignment between the perturbation (gradient) and the current state (weights).

7. **SGD as Free Energy Minimization (arXiv:2505.23489, 2025)** -- Interprets SGD's stationary behavior under fixed learning rate as minimizing a free energy functional F = U - TS, where U is the training loss (energy), S is entropy, and T is an effective temperature proportional to learning rate and inversely proportional to batch size. Weight decay adds an additional energy penalty that constrains model complexity.

#### Neuroscience / Synaptic Homeostasis

8. **Tononi & Cirelli, 2003/2006. "Synaptic Homeostasis Hypothesis (SHY)." Sleep/Brain Res. Bull.** -- The foundational hypothesis: during wakefulness, learning strengthens synapses globally; during sleep, slow-wave activity globally downscales synaptic weights to restore homeostasis. This prevents saturation, improves signal-to-noise ratio, and enables continued learning. **Core structural correspondence to weight decay**: Weight decay in neural network training is mathematically identical to synaptic downscaling -- both implement w <- (1 - lambda)w at each step. The biological insight is that downscaling should be *periodic* (sleep/wake cycle) rather than continuous, and *activity-dependent* (stronger synapses are downscaled more in absolute terms, but preserved in relative ranking). This directly maps to our insight that alignment-aware WD (which modulates decay based on gradient-weight alignment) is superior to uniform WD.

9. **Massey et al., 2026. "Sleep-Based Homeostatic Regularization for Stabilizing STDP in Recurrent SNNs." arXiv:2601.08447** -- Implements the synaptic homeostasis hypothesis as a neuromorphic regularization scheme: periodic offline phases with stochastic weight decay toward a homeostatic baseline. **Key finding**: low-to-intermediate sleep durations (10-20% of training) provide optimal stability. This directly suggests WD scheduling: periodic phases of stronger WD interspersed with phases of weaker WD, rather than constant WD throughout training.

10. **Zenke, Hennequin & Gerstner, 2013. "Synaptic plasticity in neural networks needs homeostasis with a fast rate detector." PLoS Comput. Biol.** -- Shows that simple weight decay is insufficient for homeostatic regulation; a fast rate-dependent feedback mechanism is needed. **Direct implication**: alignment-aware WD (which conditions decay on gradient-weight alignment, a proxy for "activity rate") is the computational analogue of the biological fast rate detector.

11. **Stock et al., 2022. "Synaptic Balancing: A Biologically Plausible Local Learning Rule." PLoS Comput. Biol.** -- Proposes synaptic balancing as a novel model of homeostatic plasticity that is significantly better than weight decay in the post-learning regime. The key insight: biological homeostasis does not simply shrink weights toward zero (as standard WD does) but toward a *balanced state* that preserves learned structure. **Structural correspondence**: this is the biological analogue of norm-matched WD (AdamWN), which targets a nonzero weight norm rather than zero.

12. **Mayzel & Schneidman, 2024. "Homeostatic synaptic normalization optimizes learning." eLife 96566** -- Demonstrates that synaptic normalization has a dual role: maintaining homeostasis and optimizing network performance. Normalization improves learning efficiency of neural codes. **Key bridge**: the biological dual role (homeostasis + optimization) parallels the dual role of weight decay in deep learning (regularization + training dynamics stabilization).

13. **"A Correspondence Between Normalization Strategies in Artificial and Biological Neural Networks." bioRxiv 2020.07.17.197640** -- Directly maps batch normalization, weight normalization, and other DL normalization techniques to biological homeostatic mechanisms across four spatial scales: single neuron, synaptic weights, layer, and network. **Critical insight**: the correspondence operates at the level of mathematical structure, not just metaphor.

#### Control Theory / Feedback Systems

14. **PIDAO: PID-Accelerated Optimizer for Deep Learning. (He et al., Nature Communications, 2024)** -- Develops a complete PID control framework for neural network optimization. The proportional term maps to current gradient (SGD), integral to momentum accumulation, derivative to curvature estimation. Weight decay is included as a regularization term. **Structural correspondence**: weight decay is the "setpoint bias" in a PID controller -- it defines the reference state toward which the controller drives the system. Dynamic WD corresponds to adaptive setpoint scheduling. The key insight from control theory is that the optimal setpoint should vary based on the system state (alignment, norm) and the error signal (loss), not remain fixed.

15. **Dai et al., 2023. "PID controller-based adaptive gradient optimizer." IET Control Theory & Appl.** -- Proposes Adaptive-PID optimizer that incorporates P, I, D components with adaptive gain scheduling. **Key insight for WD**: the D (derivative) term, which measures rate of change of gradient, is mathematically related to our Coupling Stability Index -- both track how rapidly the optimization dynamics are changing.

16. **ControlVAE (Shao et al., 2020. arXiv:2011.01754)** -- Uses a nonlinear PI controller to dynamically tune the KL-divergence weight in VAE training. **Direct structural correspondence**: the KL weight in VAE training plays the same role as the WD coefficient in standard training -- both control the balance between fitting (energy) and regularization (entropy). ControlVAE demonstrates that PID-based adaptive scheduling of this coefficient outperforms fixed or heuristic schedules. This is the control-theoretic version of "alignment-aware WD."

#### Evolutionary Biology / Evolutionary Game Theory

17. **Rappeport & Nitzan, 2025. "Fitness and Overfitness: Implicit Regularization in Evolutionary Dynamics." arXiv:2508.03187** -- Establishes a mathematical isomorphism between the replicator equation (evolutionary dynamics) and sequential Bayesian learning. Shows that implicit regularization in evolution prevents "overfitness" (the evolutionary analogue of overfitting) -- organisms whose complexity exceeds environmental complexity are penalized. Frequently changing environments decrease selected complexity. **Structural correspondence**: weight decay in neural networks prevents "overfitness" by constraining parameter complexity. The evolutionary insight that frequently changing environments favor lower complexity maps to the finding that WD should be stronger during early training (rapidly changing loss landscape) and weaker during late training (stabilized landscape).

18. **Gonzalez et al., 2010/2020. "Effective Regularization Through Loss-Function Metalearning." arXiv:2010.00788** -- Proves that evolved loss functions in neural networks balance two forces: "pull toward zero error" and "push away from it to avoid overfitting." This is a general principle underlying all regularization, including weight decay. **Key insight**: the balance between these two forces should be dynamic, not fixed -- exactly our thesis for dynamic WD.

#### Information Geometry / Natural Gradient

19. **Hwang, 2024. "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information." arXiv:2405.12807** -- Establishes Adam's connection to natural gradient descent via the Fisher-Rao manifold. Derives a corrected weight decay term based on information geometry that respects the local curvature of parameter space. **Direct structural correspondence**: the information-geometric correction to WD is the computational implementation of the thermodynamic optimality principle from Caraffa (2026) -- both demand that regularization respect the intrinsic geometry rather than the Euclidean embedding.

20. **Wan et al., 2020. "Spherical Motion Dynamics." arXiv:2006.08419** -- Proves that under batch normalization + weight decay + SGD, dynamics live on a sphere. Weight norm converges at linear rate to equilibrium. Introduces "angular update" as the true learning measure. **Bridge to physics**: the spherical constraint + weight decay creates an Ornstein-Uhlenbeck-like process on the sphere, directly connecting to the thermodynamic framework.

### Cross-Disciplinary Gaps (Where Transplants Haven't Been Tried)

1. **Thermodynamic process optimization -> WD scheduling**: While Sadrtdinov et al. (2025) establish the ideal gas analogy, nobody has used thermodynamic process optimization (e.g., Carnot efficiency, finite-time thermodynamics) to derive optimal WD scheduling strategies. The concept of "entropy production minimization" from nonequilibrium thermodynamics could yield provably efficient WD schedules.

2. **Allostatic regulation -> adaptive WD**: Allostasis (maintaining stability through change, as opposed to homeostasis which maintains a fixed setpoint) provides a richer framework than homeostasis for understanding dynamic WD. In allostasis, the organism proactively adjusts its setpoints based on anticipated demands -- this maps to alignment-aware WD that adjusts the decay rate based on gradient-weight alignment rather than reacting to norm deviations after the fact.

3. **Le Chatelier principle -> WD as negative feedback**: The formal reformulation of Le Chatelier as a majorization relation (Allahverdyan & Galstyan, 2011) has never been applied to weight decay dynamics. This could formalize when WD amplifies (rather than dampens) perturbations, explaining pathological WD behaviors.

4. **Biological sleep-wake cycling -> periodic WD modulation**: The finding that periodic offline downscaling (10-20% of training time) is optimal (Massey et al., 2026) has never been tested as a WD scheduling strategy in standard deep learning.

5. **Fisher-Rao regularization -> geometry-aware WD**: Caraffa (2026) proves that Euclidean WD is structurally suboptimal, but nobody has implemented Fisher-Rao WD in practical deep learning and compared it to alignment-aware methods like CWD.


## Phase 2: Initial Candidates

### Candidate A: Thermodynamic Process Optimization for WD Scheduling (from Statistical Physics)

- **Source principle**: In thermodynamics, the efficiency of transforming a system from state A to state B depends critically on the *path* taken through state space. Quasi-static processes (infinitesimally slow) are maximally efficient but impractical; finite-time thermodynamics (Curzon-Ahlborn efficiency) characterizes the optimal tradeoff between speed and efficiency. The key variables are temperature T, pressure P, and volume V, connected by the equation of state PV = NkT.

- **Structural correspondence**: As established by Sadrtdinov et al. (2025), SGD with weight decay on scale-invariant networks obeys an ideal gas law where:
  - Temperature T ~ eta * sigma^2 (learning rate * gradient noise variance)
  - Pressure P ~ lambda (weight decay coefficient)
  - Volume V ~ ||w||^2 (squared parameter norm)
  - Entropy S ~ log(||w||^d * T^{d/2}) (d = number of parameters)

  The four WD sub-approaches correspond to different thermodynamic constraints:
  - **Fixed WD** (standard AdamW): isobaric process (P = const), V and T vary freely
  - **WD scheduling**: designed thermodynamic process P(t) that optimizes a target functional
  - **Norm-matched WD** (AdamWN): isochoric process (V = const), P adjusts to maintain target norm
  - **Alignment-aware WD** (CWD): local P varies by direction, like anisotropic pressure

  The hypothesis is that optimal WD scheduling corresponds to the thermodynamic process that minimizes entropy production (irreversible information loss) while achieving a target state in finite time.

- **Hypothesis**: WD scheduling derived from finite-time thermodynamics (minimizing entropy production rate) will outperform heuristic schedules (cosine, linear, gradient-norm-based) by 1-3% test accuracy on CIFAR-100 and ImageNet. Specifically, the optimal WD schedule lambda(t) follows the "minimum entropy production" path:

  lambda(t) = lambda_0 * T(t) / T(0) * (||w(0)||^2 / ||w(t)||^2)

  which couples WD to both the effective temperature (learning rate schedule) and current norm, rather than being set independently.

- **Why it's not just a metaphor**: The correspondence is mathematical, not verbal. The SDE for SGD with weight decay on scale-invariant networks is:

  d||w||^2 = (-2*lambda*||w||^2 + eta*tr(Sigma(w))) dt + noise

  which is an Ornstein-Uhlenbeck process with drift toward ||w||^2 = eta*tr(Sigma)/(2*lambda). This is *identical* to the thermodynamic relaxation equation for an ideal gas under constant pressure, with the mapping given above. The entropy production can be computed exactly.

- **Novelty estimate**: 8/10 -- The ideal gas analogy is established (Sadrtdinov et al., 2025), but nobody has used thermodynamic process optimization to derive WD schedules. The concept of "entropy production minimization" as an optimization criterion for WD scheduling is novel.

### Candidate B: Allostatic Weight Regulation (from Neuroscience/Physiology)

- **Source principle**: Allostasis (Sterling & Eyer, 1988) is the process of maintaining stability through predictive change, in contrast to homeostasis which maintains a fixed setpoint via reactive feedback. The brain does not maintain a single homeostatic setpoint for synaptic weights; instead, it proactively adjusts setpoints based on anticipated demands. Key mechanisms include:
  1. **Anticipatory adjustment**: Cortisol and other hormones shift regulatory setpoints *before* a stressor arrives, based on learned predictions
  2. **Multi-timescale regulation**: Fast feedback (seconds: firing rate homeostasis) and slow adaptation (hours-days: synaptic scaling, sleep-dependent downscaling)
  3. **Activity-dependent scaling**: Synaptic downscaling during sleep is proportional to synaptic strength (stronger synapses lose more absolute weight, preserving relative ranking)
  4. **Periodic vs. continuous regulation**: The sleep-wake cycle implements periodic downscaling, not continuous

- **Structural correspondence**:
  - **Homeostasis = fixed WD** (standard lambda = const): reactive, maintains fixed norm setpoint, no anticipation
  - **Allostasis = dynamic WD**: proactive, adjusts lambda based on predicted optimization needs
  - **Fast rate detector = gradient-weight alignment**: Zenke et al. (2013) showed biological homeostasis requires a fast rate-dependent signal; in ML, this is cos(w, g) -- the alignment between weights and gradients
  - **Slow adaptation = norm tracking over EMA**: the EMA timescale of Wang & Aitchison (2024)
  - **Sleep-dependent downscaling = periodic WD boost**: Massey et al. (2026) found 10-20% sleep duration is optimal
  - **Activity-dependent scaling = alignment-modulated WD**: CWD's sign-alignment mask is a binary version; the biological mechanism suggests a continuous, magnitude-weighted version

  The mathematical formulation is:

  lambda_allostatic(t, w, g) = lambda_base * [1 + alpha * anticipatory_signal(t)] * [1 + beta * activity_signal(w, g)]

  where:
  - anticipatory_signal(t) = EMA of loss change rate (predicts upcoming difficulty)
  - activity_signal(w, g) = continuous alignment measure (e.g., cos(w, g) scaled by ||g||/||w||)

- **Hypothesis**: Allostatic WD, which proactively adjusts decay strength based on predicted optimization needs (not just current state), will outperform both fixed WD and reactive alignment-aware WD (CWD) by avoiding the latency inherent in reactive feedback. Specifically:
  1. On CIFAR-100/ResNet-20: 0.5-1.5% improvement over CWD
  2. The anticipatory component will show the most benefit during learning rate transitions (warmup end, cosine decay inflection points)
  3. The periodic WD boost ("sleep phase") during every N-th epoch will improve weight norm stability by >20% compared to continuous WD

- **Why it's not just a metaphor**: The correspondence preserves the key structural property of allostasis: *predictive* rather than *reactive* regulation. In biology, allostatic regulation uses internal models (cortisol dynamics) to predict upcoming metabolic demands. In our formulation, the anticipatory signal uses the EMA of loss change rate as an internal model of upcoming optimization difficulty. The periodic WD boost directly implements the sleep-wake mechanism, which is not merely a metaphor but a functionally identical operation (periodic multiplicative downscaling of connection weights). The biological insight that the sleep period should be 10-20% of the total cycle provides a concrete, testable prediction.

- **Novelty estimate**: 9/10 -- While synaptic homeostasis has been loosely compared to weight decay before, (a) the allostasis distinction (proactive vs. reactive) has never been applied to WD, (b) the specific computational formulation with anticipatory + activity signals is novel, and (c) the periodic WD boost derived from sleep research is unexplored.

### Candidate C: Le Chatelier Feedback Control for WD Stability (from Physical Chemistry / Control Theory)

- **Source principle**: Le Chatelier's principle states that a system at equilibrium, when subjected to a perturbation, will adjust to partially counteract the perturbation and establish a new equilibrium. Allahverdyan & Galstyan (2011) formalized this as a majorization relation: the response vector (how the system adjusts) must be "more ordered" (in the majorization sense) than the perturbation vector. The principle identifies when equilibria resist perturbations (via negative feedback) and when they amplify them (positive feedback violating Le Chatelier).

  Combined with PID control theory (PIDAO, Nature Communications 2024), this yields a formal framework for adaptive feedback control of regularization.

- **Structural correspondence**:
  - **Equilibrium**: The stationary state of SGD + WD (rotational equilibrium of Kosson et al., 2023)
  - **Perturbation**: Gradient updates that push weights away from the WD-imposed norm target
  - **Le Chatelier response**: WD acts as negative feedback, pulling weights back toward the target norm
  - **Majorization condition**: The Le Chatelier principle is satisfied when the WD response is "less complex" than the gradient perturbation. This is equivalent to requiring that WD acts uniformly across parameter space -- which is precisely what standard (non-adaptive) WD does.
  - **Violation of Le Chatelier**: When the gradient perturbation is highly structured (e.g., concentrated in a few directions), uniform WD can amplify the perturbation by disproportionately shrinking the informative dimensions. This explains why alignment-aware WD (CWD, AdamO) outperforms standard WD: they restore Le Chatelier compliance by making the response match the structure of the perturbation.

  The PID formulation:
  - **P (Proportional)**: lambda * ||w||^2 (standard WD, proportional to current norm deviation)
  - **I (Integral)**: lambda * integral(||w(s)||^2 ds) (accumulated norm deviation -- related to the "WD budget" in Budget Equivalence Metric)
  - **D (Derivative)**: lambda * d/dt(||w||^2) (rate of norm change -- detects instability, related to Coupling Stability Index)

  The Coupling Stability Index (CSI) is then formally the D-component of the Le Chatelier response: it measures whether the system is approaching or departing from equilibrium.

- **Hypothesis**: WD strategies that satisfy the Le Chatelier majorization condition (i.e., whose response structure matches the gradient perturbation structure) will exhibit superior coupling stability. Standard WD violates Le Chatelier for structured gradients; CWD partially restores it (binary alignment); a fully Le Chatelier-compliant WD (continuous alignment with magnitude matching) will be optimal. Specifically:
  1. Le Chatelier-compliant WD will have CSI values 2-5x lower than standard WD
  2. On tasks with highly structured gradients (ResNet-50/ImageNet), the improvement will be largest
  3. The PID formulation provides a natural decomposition for diagnosing WD failures

- **Why it's not just a metaphor**: The Le Chatelier majorization relation is a mathematical theorem about the response of equilibrium systems to perturbations. The mapping to WD is structural: the Ornstein-Uhlenbeck dynamics of SGD+WD define an equilibrium distribution, gradient updates are perturbations, and the WD term is the restoring force. The majorization condition provides a quantitative criterion (not just a qualitative analogy) for when WD will stabilize vs. destabilize training. Furthermore, the PID decomposition of the WD response (P = proportional to norm, I = integrated norm budget, D = rate of norm change) maps exactly to three components of a physical feedback controller, with the D-term providing the novel Coupling Stability Index.

- **Novelty estimate**: 7/10 -- The PID analogy for optimizers has been explored (PIDAO, 2024) and CWD already has a sliding-mode control interpretation. The novel contribution is (a) the Le Chatelier majorization formalization for WD stability, (b) the connection between majorization violation and WD failure modes, and (c) the formal identification of CSI as the D-component of a Le Chatelier response.


## Phase 3: Self-Critique

### Against Candidate A: Thermodynamic Process Optimization for WD Scheduling

- **Shallow analogy attack**: Is the ideal gas law for SGD+WD truly structural, or just a coincidence of functional form? The correspondence is derived from the SDE of SGD on scale-invariant networks (Sadrtdinov et al., 2025), which is mathematically rigorous for the isotropic noise model. However, real networks violate isotropy: different layers have different gradient variances, and the noise covariance is not proportional to the identity matrix. The thermodynamic analogy may break down for anisotropic dynamics, which are precisely the settings where alignment-aware WD matters most. A domain expert in statistical mechanics would likely object that "ideal gas" implies non-interacting particles (parameters), but neural network parameters are strongly coupled through the loss function. The analogy is best understood as a *mean-field approximation*, which is valid for aggregate quantities (total norm, average entropy) but not for per-parameter dynamics.

- **Scale mismatch attack**: Thermodynamic concepts are defined in the thermodynamic limit (N -> infinity). Modern networks have 10^6 to 10^9 parameters, which is arguably large enough. However, the correspondence between parameter count and "number of particles" is imprecise: some parameters are more important than others. The effective dimension of the optimization problem is often much smaller than the parameter count (due to low-rank structure induced by WD itself). Verdict: moderate concern; the aggregate predictions should hold, but per-layer deviations are expected.

- **Prior transplant check**: Sadrtdinov et al. (2025) establish the basic analogy. The concept of "minimum entropy production" in optimization has been explored tangentially (Caraffa, 2026 on thermodynamic efficiency). But nobody has derived concrete WD schedules from finite-time thermodynamics. The specific prediction of lambda(t) = lambda_0 * T(t)/T(0) * ||w(0)||^2/||w(t)||^2 appears novel.

- **Testability attack**: The predictions are concrete and testable: (a) entropy production can be computed from the SDE, (b) the proposed schedule can be compared against baselines, (c) the coupling lambda(t) ~ T(t)/||w(t)||^2 is measurable during training. A diagnostic experiment could compare the entropy production of different WD schedules on the same task.

- **Verdict**: **STRONG**. The mathematical correspondence is rigorous (for the mean-field case), the predictions are concrete and testable, and the novel contribution (thermodynamic process optimization for WD scheduling) is clearly defined. The main weakness is the anisotropy issue, which can be addressed by treating each layer as a separate "gas" with its own temperature and pressure.

### Against Candidate B: Allostatic Weight Regulation

- **Shallow analogy attack**: The distinction between homeostasis (reactive) and allostasis (predictive) is conceptual, not mathematical. Is the "anticipatory signal" truly analogous to cortisol dynamics, or is it just an EMA of the loss derivative with a biological label? A neuroscience expert would note that allostatic regulation involves complex neuroendocrine feedback loops with nonlinear dynamics, not simple EMAs. The formal mapping between cortisol dynamics and EMA of loss change rate is loose at best.

  However, the *functional* correspondence is tight: in both cases, the system uses historical information to predict future demands and proactively adjusts regulatory parameters. The specific implementation details differ (cortisol vs. EMA), but the computational principle (predictive regulation) is preserved.

- **Scale mismatch attack**: Biological allostasis operates over timescales of hours to days, with the sleep-wake cycle at ~24 hours. Training runs span thousands of gradient steps, with learning rate schedules over hundreds of epochs. The timescale mapping is not one-to-one, but the ratio (regulation period / total training time) can be preserved. The biological finding of 10-20% sleep duration translating to periodic WD boosts every 5-10 epochs is plausible but not rigorously justified.

- **Prior transplant check**: Synaptic homeostasis has been compared to weight decay many times (Carlson et al., 2013; Stock et al., 2022; Hakim & Alam, 2025). The correspondence between normalization strategies in artificial and biological networks is well-documented (bioRxiv 2020.07.17.197640). However, the *allostasis* distinction (proactive vs. reactive) has not been applied to WD. The specific computational formulation with anticipatory + activity signals, and the periodic WD boost, appear novel.

- **Testability attack**: The predictions are testable: (a) anticipatory WD should show most benefit at learning rate transitions, (b) periodic WD boost with 10-20% duty cycle should improve norm stability, (c) the continuous alignment signal should outperform binary CWD. A clean diagnostic experiment: compare three conditions -- constant WD (homeostasis), CWD (reactive homeostasis with fast detector), and allostatic WD (predictive + periodic) -- on identical tasks.

- **Verdict**: **MODERATE-STRONG**. The biological inspiration is rich and the functional correspondence is genuine, but the formal mathematical mapping is looser than for Candidate A. The anticipatory component is the most novel and testable contribution, but its superiority over simpler reactive methods is not guaranteed. The periodic WD boost from sleep research is a concrete, novel prediction that could be easily tested.

### Against Candidate C: Le Chatelier Feedback Control for WD Stability

- **Shallow analogy attack**: The Le Chatelier principle is typically stated for systems near equilibrium, but SGD with WD spends most of training time far from equilibrium (early training has large gradients, rapidly changing loss). The majorization formalization (Allahverdyan & Galstyan, 2011) is specific to replicator dynamics, and the mapping to SGD dynamics is not immediate. A physicist would object that Le Chatelier requires thermodynamic equilibrium (maximum entropy), whereas SGD trajectories are non-equilibrium processes.

  However, the rotational equilibrium of Kosson et al. (2023) and the spherical motion dynamics of Wan et al. (2020) show that SGD+WD+normalization *does* reach a quasi-equilibrium (weight norm converges). The Le Chatelier analysis applies to perturbations of this quasi-equilibrium.

- **Scale mismatch attack**: The PID analogy is well-established (PIDAO, 2024) and operates at the correct scale. The Le Chatelier majorization requires computing the response of all parameters to a perturbation, which is tractable. No scale mismatch issue.

- **Prior transplant check**: CWD already has a sliding-mode control interpretation (from their paper). PIDAO applies PID control to optimization directly. The specific Le Chatelier majorization formalization for WD stability appears novel, but it's an incremental extension of existing control-theoretic perspectives.

- **Testability attack**: The majorization condition is computable in principle but expensive (requires comparing the singular values of the gradient perturbation and WD response matrices). The PID decomposition is straightforward to implement. The prediction that Le Chatelier-compliant WD has lower CSI is testable.

- **Verdict**: **MODERATE**. The control-theoretic perspective is well-explored (PIDAO, CWD), and the Le Chatelier formalization adds rigor but limited novelty. The main contribution is reframing existing ideas within a unified mathematical framework, rather than proposing a genuinely new mechanism.


## Phase 4: Refinement

### Dropped
- **Candidate C** (Le Chatelier Feedback Control) is downgraded. While mathematically sound, it is largely a reformulation of existing ideas (PID control for optimization, sliding-mode interpretation of CWD). The Le Chatelier majorization condition adds rigor but not a new mechanism. Elements of this candidate are absorbed into the other two.

### Strengthened Survivors

**Candidate A (Thermodynamic Process Optimization)** -- strengthened with several refinements:

1. **Formalized structural correspondence**: The mapping SGD+WD <-> ideal gas is:

   | Thermodynamic Quantity | SGD+WD Analogue | Formula |
   |---|---|---|
   | Temperature T | Effective temperature | T_eff = eta * tr(Sigma(w)) / d |
   | Pressure P | Weight decay | P = lambda |
   | Volume V | Squared parameter norm | V = ||w||^2 |
   | Entropy S | Log-volume of stationary dist. | S = (d/2) * log(T_eff / lambda) + const |
   | Internal energy U | Training loss | U = L(w) |
   | Free energy F | Regularized objective | F = L(w) + (lambda/2)*||w||^2 |
   | Equation of state | Ideal gas law | lambda * ||w||^2 = eta * tr(Sigma(w)) |
   | Heat capacity C_V | Gradient noise sensitivity | C_V = d * eta * delta(tr(Sigma))/delta(T_eff) |

2. **Per-layer extension**: Treat each layer l as a separate "gas" with its own (T_l, P_l, V_l), connected through the loss function. This introduces *thermal contact* between layers (gradient flow couples their temperatures) and *mechanical coupling* (shared loss landscape). The optimal scheduling problem becomes a multi-component thermodynamic optimization.

3. **Finite-time thermodynamics bound**: The minimum entropy production for transitioning from state (lambda_0, ||w_0||^2) to (lambda_f, ||w_f||^2) in time T is bounded below by:

   sigma_min >= (S_f - S_0)^2 / (T * C_eff)

   where C_eff is the effective heat capacity. This provides an information-theoretic lower bound on the "cost" of WD scheduling.

4. **Additional support**: The grokking theory of Boursier et al. (2025) confirms the two-phase structure: fast kinetic phase (T-dominated) followed by slow thermodynamic phase (P-dominated). The transition happens at t* ~ 1/lambda, which is the "equilibration timescale" in the thermodynamic framework.

**Candidate B (Allostatic Weight Regulation)** -- strengthened with formalized mapping:

1. **Formal three-level regulation hierarchy**:
   - **Level 1 (Fast/reactive, ~1 step)**: Per-parameter alignment masking (analogous to fast firing-rate homeostasis). Implementation: CWD-style sign alignment or continuous cos(w, g) modulation.
   - **Level 2 (Medium/adaptive, ~100 steps)**: Norm-tracking with adaptive target (analogous to slow synaptic scaling). Implementation: EMA of ||w|| with target adjusted by loss curvature.
   - **Level 3 (Slow/predictive, ~1000 steps)**: Anticipatory WD schedule adjustment (analogous to allostatic setpoint modulation). Implementation: Lambda adjusted based on predicted upcoming difficulty (EMA of d(Loss)/dt).

2. **Periodic "sleep phase" formalization**: Every N epochs, increase lambda by factor k for M epochs. The biological prediction: N ~ 5-10 epochs, M/N ~ 0.1-0.2 (10-20% duty cycle), k ~ 2-5x. During sleep phase, the network consolidates by preferentially reducing parameters that are large but weakly connected to the loss gradient (high ||w_i|| but low |grad_i|) -- this is exactly the complement of what CWD protects (CWD protects parameters where sign(w) = sign(update), i.e., gradient-aligned).

3. **Connection to thermodynamic framework**: The allostatic setpoint modulation (Level 3) corresponds to changing the "pressure" P = lambda in the thermodynamic framework. The fast reactive regulation (Level 1) corresponds to anisotropic pressure (per-direction lambda). The medium-timescale norm tracking (Level 2) corresponds to maintaining target volume V = ||w_target||^2. Thus, the allostatic hierarchy *subsumes* the thermodynamic framework by adding per-parameter and predictive dimensions.

### Selected Front-Runner: **Candidate B (Allostatic Weight Regulation)**

Rationale: Candidate B provides the richest set of novel, testable predictions and the most complete framework. While Candidate A provides a more rigorous mathematical foundation, its main contribution (thermodynamic process optimization for WD scheduling) is a special case of Candidate B's Level 3 (anticipatory scheduling). Candidate B subsumes Candidate A's insights while adding the biologically inspired multi-timescale hierarchy and periodic regulation, which are genuinely novel and experimentally actionable.

However, the thermodynamic vocabulary from Candidate A provides the mathematical language for the framework, and the Le Chatelier stability analysis from Candidate C provides the diagnostic tools. The final proposal synthesizes all three.


## Phase 5: Final Proposal

### Title
**Allostatic Weight Decay: A Multi-Timescale Regulation Framework Inspired by Synaptic Homeostasis and Thermodynamic Process Optimization**

### Source Principle
The proposal draws on three cross-disciplinary sources:

1. **Synaptic allostasis** (neuroscience): Living neural systems regulate synaptic weights through a multi-timescale hierarchy -- fast firing-rate homeostasis (milliseconds), slow synaptic scaling (hours), and periodic sleep-dependent downscaling (daily). The key insight from allostasis is that regulation should be *predictive* (adjusting setpoints based on anticipated demands) rather than *reactive* (correcting deviations after they occur).

2. **Thermodynamic process optimization** (statistical physics): SGD with weight decay on scale-invariant networks obeys an ideal gas law (Sadrtdinov et al., 2025). Optimal WD scheduling corresponds to thermodynamic process design that minimizes entropy production -- the irreversible information loss during training.

3. **Le Chatelier negative feedback** (physical chemistry / control theory): Weight decay provides the restoring force that maintains training stability. When the WD response matches the structure of gradient perturbations (Le Chatelier compliance), stability is maximized.

### Structural Correspondence

The formal mapping between biological synaptic regulation and weight decay:

| Biological Mechanism | WD Analogue | Mathematical Form |
|---|---|---|
| Firing-rate homeostasis (fast) | Per-parameter alignment masking | lambda_i(t) = lambda * phi(cos(w_i, g_i)) |
| Synaptic scaling (slow) | Norm-tracking with adaptive target | target_norm(t) = EMA_slow(||w(t)||) |
| Allostatic setpoint modulation | Anticipatory schedule adjustment | lambda(t) = lambda_0 * [1 + alpha * EMA(dL/dt)] |
| Sleep-dependent downscaling | Periodic WD boost phase | lambda -> k*lambda every N epochs for M steps |
| Activity-dependent scaling | Magnitude-weighted decay | |delta w_i| prop. to |w_i| * phi(alignment_i) |
| Signal-to-noise optimization | Entropy production minimization | min integral(sigma(t) dt) over training |

### Hypothesis
The allostatic multi-timescale WD framework will outperform all single-mechanism approaches (fixed WD, CWD, SWD, AdamWN) by providing:

1. **Superior stability** (CSI < 0.5x baseline): The multi-timescale regulation prevents both the instability of pure alignment-based methods (which can oscillate) and the sluggishness of pure scheduling methods (which cannot react to local conditions).

2. **Improved test accuracy** (0.5-2% on CIFAR-100, 0.3-1% on ImageNet): The anticipatory component reduces the latency of WD adaptation at critical training phases (LR transitions, plateau escapes).

3. **Better norm dynamics** (norm variance < 0.3x baseline): The periodic WD boost phase restores optimal signal-to-noise ratio, preventing gradual norm drift.

### Method

**Three-level adaptive WD controller:**

```
Level 1 (Per-Step, Per-Parameter):
  alpha_i = continuous_alignment(w_i, g_i)  # e.g., tanh(cos(w_i, g_i) * ||g_i||/||w_i||)
  lambda_fast_i = lambda_base * (1 + beta_fast * alpha_i)

Level 2 (Per-Epoch, Global):
  norm_target = EMA(||w||, tau_slow)
  norm_error = ||w|| - norm_target
  lambda_medium = lambda_base * (1 + beta_medium * norm_error / norm_target)

Level 3 (Every N Epochs, Global):
  loss_trend = EMA(dL/dt, tau_anticipate)
  if epoch % N < M:  # "sleep phase"
    lambda_slow = k * lambda_base  # boost factor k ~ 2-5
  else:
    lambda_slow = lambda_base * (1 + beta_slow * loss_trend)

Final: lambda_i(t) = lambda_fast_i * lambda_medium / lambda_base * lambda_slow / lambda_base
```

The three levels correspond to the P, I, D components of a PID controller:
- Level 1 (fast/proportional): responds to current gradient-weight alignment
- Level 2 (medium/integral): accumulates norm deviation over time
- Level 3 (slow/derivative + periodic): anticipates future needs + periodic correction

### Diagnostic Experiment
The key test that confirms the analogy is load-bearing (not just decorative):

**Ablation of the three levels independently**:
1. Level 1 alone = CWD (alignment-aware WD) -- should match CWD performance
2. Level 2 alone = AdamWN-like (norm-matched WD) -- should match AdamWN performance
3. Level 3 alone = SWD-like (scheduled WD) -- should match SWD performance
4. Levels 1+2 = alignment + norm = new (no existing method)
5. Levels 1+2+3 = full allostatic WD = best performance

If the full system outperforms the sum of individual levels, this confirms that the multi-timescale hierarchy (from biological allostasis) provides emergent benefits beyond what any single mechanism achieves.

**The "sleep phase" diagnostic**: Compare continuous WD with periodic WD boost (10%, 15%, 20% duty cycle). If the periodic boost improves performance (as predicted by the synaptic homeostasis hypothesis and Massey et al., 2026), this confirms that the biological mechanism is functional, not just metaphorical.

**The "anticipation" diagnostic**: At learning rate transition points (e.g., cosine decay inflection), measure whether Level 3 adjusts lambda *before* the transition (anticipatory) or *after* (reactive). If the EMA of loss derivative provides genuine anticipation, the norm trajectory should be smoother at transitions compared to purely reactive methods.

### Experimental Plan
- **Small models** (CIFAR-10/100): ResNet-20, VGG-16-BN with SGD and AdamW. 3 seeds (42, 123, 456). Target: ~30 min per experiment.
- **Medium models** (CIFAR-100): ResNet-50 with full allostatic WD. Target: ~45 min per experiment.
- **Large models** (ImageNet): ResNet-50, ViT-Small. Target: 4-8 hours per experiment (override default 1-hour limit per project spec).
- **Baselines**: Standard WD, CWD, SWD, AdamWN, cosine WD schedule.
- **Metrics**: Test accuracy, weight norm trajectory, CSI, entropy production estimate, norm variance during "sleep phase" vs. continuous phase.
- **Visualization**: Thermodynamic state diagrams (lambda vs. ||w||^2 vs. loss), alignment cosine histograms, per-layer "temperature" evolution.

### Risk Assessment

1. **The anticipatory component may not provide meaningful benefit over reactive methods** (probability: 40%). If the loss derivative EMA is too noisy to provide genuine prediction, Level 3 degenerates to a smoothed version of gradient-norm-based scheduling. Mitigation: use longer EMA windows (tau ~ 100-500 steps) and validate prediction quality empirically.

2. **The periodic WD boost may disrupt training rather than improve it** (probability: 30%). If the "sleep phase" is too aggressive (k too large or M too long), it could destroy learned representations. Mitigation: start with conservative settings (k=1.5, M/N=0.1) and gradually increase.

3. **The three-level hierarchy may be overfit to the problem domain** (probability: 25%). If the levels interact negatively (e.g., Level 1 and Level 3 provide conflicting signals), the composite WD may underperform simpler methods. Mitigation: ablation study validates each level independently.

4. **The thermodynamic vocabulary may obscure rather than illuminate** (probability: 20%). If the ideal gas analogy breaks down for non-scale-invariant networks, the entropy production analysis becomes approximate. Mitigation: validate the PV=NkT relationship empirically before relying on it for scheduling.

### Novelty Claim

The specific cross-disciplinary insight: **Weight decay in deep learning is the computational analogue of synaptic homeostasis in biological neural networks, and the four WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) correspond to different levels of the biological regulation hierarchy (allostatic setpoint modulation, fast rate-dependent feedback, structural separation, and target-norm control).** This mapping is structural, not metaphorical: the mathematical forms are identical (multiplicative weight reduction), the functional roles are the same (preventing saturation, improving signal-to-noise, enabling continued learning), and the biological finding of multi-timescale regulation with periodic downscaling has never been applied to deep learning WD.

Evidence of novelty:
- arXiv search for "allostasis weight decay" / "allostatic regularization" / "allostatic weight decay" returns 0 results
- arXiv search for "synaptic homeostasis weight decay deep learning" returns results comparing the *analogy* but not implementing a multi-timescale allostatic controller
- The "sleep phase" periodic WD boost has no precedent in the deep learning optimization literature
- The three-level hierarchy formalization is novel
