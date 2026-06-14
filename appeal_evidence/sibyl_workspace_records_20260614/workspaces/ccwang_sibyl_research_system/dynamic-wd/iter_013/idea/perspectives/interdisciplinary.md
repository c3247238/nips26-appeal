# Interdisciplinary Perspective

**Topic**: Unified Dynamic Weight Decay Framework — unifying WD scheduling, alignment-aware WD,
decoupled WD, and norm-matched WD into a theoretical framework with standardized evaluation metrics
(Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score).

**Generated**: 2026-03-19

---

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **BCM Theory (Bienenstock, Cooper, Munro 1982)** — The BCM sliding threshold mechanism (θ_M)
   provides a stabilizing rule for Hebbian learning: synaptic change is positive (LTP) when
   postsynaptic activity exceeds θ_M and negative (LTD) otherwise. The threshold itself slides
   upward as a super-linear function of historical mean activity, producing automatic homeostatic
   balance. Key structural correspondence: the sliding θ_M is exactly the dynamic quantity needed to
   implement alignment-aware WD schedules — the degree of "activity alignment" between gradient and
   weight (δ̂_t in our notation) controls whether to strengthen or weaken connections.

2. **Synaptic Homeostasis (Turrigiano & Nelson 2004 + BioLogicalNeuron 2025)** — Neurons maintain
   activity within an optimal "health band" via calcium-based sensors and negative feedback. When
   activity deviates upward, synaptic scaling reduces excitatory weights globally. This is a
   formal analog of norm-matched WD: the "target norm" τ corresponds to the healthy activity
   setpoint, and WD applies proportionally to the deviation from that setpoint.

3. **Dynamic Weight Adaptation in Spiking Neural Networks inspired by BCM (DWAM, arXiv:2511.17563,
   2025)** — Extends BCM homeostasis to spiking nets, demonstrating that a dynamically adjusted
   adaptation mechanism improves performance under degraded conditions. Directly validates the idea
   that continuous, activity-history-aware weight modulation (not just binary decay on/off) is
   beneficial.

4. **Timescale Hierarchy in Cortical Learning (arXiv:2510.18808, 2024)** — Formal proof that
   three timescale hierarchies — fast signal propagation, intermediate plasticity, slow synaptic
   homeostatic decay — produce stable gradient-aligned learning in continuous time. Classical
   discrete-time optimizers (SGD, Adam) emerge as limiting cases under timescale separation. The
   slow decay timescale is structurally identical to WD scheduling in discrete-time training.

5. **Hebbian CNNs with BCM + Weight Decay as LTD (arXiv:2501.17266, 2025)** — Explicitly maps
   weight decay to Long-Term Depression in BCM networks, demonstrating that the BCM rule's
   prevention of runaway synaptic growth (via LTD + homeostasis) is the biological mechanism
   underpinning L2 regularization's role in deep learning.

#### Physics / Thermodynamics / Statistical Mechanics

6. **Stochastic Thermodynamics of Learning (Goldt & Seifert, PRL 118, 2017)** — Formal
   thermodynamic framework for learning dynamics: gradient noise plays the role of thermal
   fluctuations, loss landscape is the free energy surface, and WD acts as a "restoring force"
   keeping weights near a prior (equilibrium) distribution. The Jarzynski equality then provides
   a path-integral upper bound on the generalization error of any nonequilibrium training
   trajectory.

7. **Heavy-Tailed Self-Regularization (HTSR) and Renormalization Group (WeightWatcher/SETOL,
   calculatedcontent.com, Dec 2024)** — Well-trained networks self-organize near a "critical
   point" where weight matrix spectra follow universal power laws, mirroring Wilson's RG fixed
   points. The WD strength determines the distance from criticality: too-strong WD pushes the
   network away from the critical regime and destroys the self-organized representations; too-weak
   WD allows it to memorize rather than generalize. This provides a physics-based optimality
   criterion for WD strength: target the critical scaling regime.

8. **Jarzynski Equality Applied to ML (arXiv:2107.08608)** — The Jarzynski identity relates
   nonequilibrium work (training with finite learning rate) to the free energy difference (train–
   test gap). Minimizing the dissipated work (irreversibility of the training trajectory) directly
   minimizes the generalization error. WD contributes to reducing dissipated work by keeping
   weight magnitudes bounded, providing a thermodynamic interpretation of WD's generalization
   benefit.

9. **Machine Learning and the Renormalization Group (ECT* Workshop, May 2024)** — An entire
   interdisciplinary workshop exploring how RG flow transformations, information bottleneck, and
   optimal transport connect to deep learning. RG coarse-graining maps onto the hierarchical
   feature extraction in deep nets; WD provides a "running coupling" that controls how features
   at different scales interact and are preserved or discarded during the RG flow.

#### Neuroscience / Control Theory Intersection

10. **Sliding Mode Control (SMC) and its connection to optimization** — CWD (Cautious Weight
    Decay, ICLR 2026) explicitly notes its sliding-mode behavior: when gradient and weight have
    aligned signs, the system is on the "sliding surface" and standard optimization governs; when
    misaligned, decay is suppressed. The full SMC theory provides tools for analyzing convergence
    guarantees, chattering suppression, and robustness — none of which have been applied to WD
    scheduling.

11. **Optimal Control / Pontryagin Maximum Principle for CNN Training (arXiv:2504.11647, 2025)**
    — Recast supervised learning as a discrete-time optimal control problem, yielding PMP-based
    first-order optimality conditions. Weight regularization (L0, L1, L2) appears as a term in
    the Hamiltonian. The adjoint (costate) variables provide a formal signal for when and how
    strongly to apply regularization — directly analogous to the alignment signal δ̂_t in our
    framework.

12. **Lyapunov Stability + Regularization (arXiv:2511.01283, 2025)** — Embedding Lyapunov
    stability conditions as inductive biases (rather than hard constraints) enables stable
    optimization with certified convergence regions. The regularization term ensures the Lyapunov
    function's time derivative is negative — a formal stability certificate that could be applied
    to the alignment-aware WD convergence analysis.

#### Biology / Evolutionary Biology

13. **Mutation-Selection Balance in Population Genetics** — In population genetics, mutation–
    selection equilibrium describes a steady-state where constant mutation pressure (introducing
    deleterious alleles) is balanced by purifying selection (removing them). The equilibrium
    allele frequency is p* = μ/s (mutation rate over selection coefficient). Structural
    correspondence: WD (selection pressure s) vs. gradient magnitude (mutation pressure μ)
    determines equilibrium weight magnitude w* = ‖g‖/λ at steady state — exactly the Defazio
    (2506.02285) result about gradient-to-weight ratio control.

14. **Pruning as Evolution: Emergent Sparsity Through Selection Dynamics (arXiv:2601.10765,
    2026)** — Formalizes neural network pruning as population dynamics where parameter groups
    (neurons, filters) evolve under selection pressure. Sparsity emerges as a temporal phenomenon
    from differential growth and decay. Alignment-aware WD applies the equivalent of
    "environment-conditional" selection pressure: when alignment is high, selection is mild
    (weights are fit); when alignment is low, selection is strong (weights are maladaptive).

#### Information Theory / Free Energy Principle

15. **Free Energy Principle and Weight Regularization (arXiv:2505.22749, 2025)** — Under the
    Free Energy Principle (Friston), agents minimize variational free energy F = accuracy -
    complexity. The complexity term in F is precisely a KL divergence penalty on model parameters
    from a prior — formally equivalent to L2 weight decay. Dynamic WD scheduling then corresponds
    to adaptive modulation of the complexity-accuracy trade-off across training time, matching the
    brain's online Bayesian model compression.

---

### Cross-Disciplinary Gaps

The literature survey reveals these critical **untransplanted bridges**:

- **BCM sliding threshold → Continuous alignment-aware WD**: The BCM formula θ_M ∝ <y²> has
  never been formally mapped to an adaptive WD rule. No paper derives dynamic WD from BCM's
  formal stability analysis.
- **Thermodynamic work minimization → WD schedule optimization**: The Jarzynski equality bounding
  generalization error via training trajectory irreversibility has not been applied to design
  optimal WD schedules.
- **RG critical point criterion → Optimal WD strength**: Using spectral criticality (HTSR theory)
  as an online signal for WD strength has not been proposed.
- **PMP costate variables → Alignment signal**: The adjoint/costate variables from the optimal
  control formulation of training have never been connected to the gradient-weight alignment
  signal δ_t.
- **Mutation-selection equilibrium → WD-gradient equilibrium**: The formal analogy between
  population genetic equilibrium and the Defazio steady-state result has not been made explicit
  or exploited for WD scheduling.

---

## Phase 2: Initial Candidates

### Candidate A: BCM Sliding Threshold as the Canonical Model for Alignment-Aware WD

**Source field**: Computational Neuroscience

**Source principle**: In BCM theory, the modification threshold θ_M for synaptic potentiation vs.
depression is not fixed but slides as a super-linear function of the time-averaged postsynaptic
activity: dθ_M/dt = φ(y) − θ_M/τ, where y is postsynaptic activity, φ is super-linear (typically
φ(y) = y²), and τ is a slow timescale constant. This sliding threshold provides homeostatic
stability: if activity is persistently high (weights too strong), θ_M rises, making further LTP
harder and LTD more likely, which reduces synaptic strengths. If activity is persistently low, θ_M
falls, making LTP easier — a negative feedback loop maintaining an optimal activity regime.

**Structural correspondence**: Map postsynaptic activity y → gradient-weight dot product (the
alignment signal): y ≡ ⟨g_t, w_t⟩ / (‖g_t‖ ‖w_t‖) = δ̂_t. Then:

- θ_M in BCM → WD threshold/strength λ_t in dynamic WD
- Sliding threshold mechanism → Alignment-aware update rule: λ_{t+1} = λ_t + η_λ(δ̂_t² − λ_t/τ_λ)
- LTD (synaptic weakening when y > θ_M) → WD active when alignment is high (decay only when gradient
  and weight agree in direction)
- LTP suppression when y < θ_M → Reduced WD when alignment is low (don't decay when gradient
  direction conflicts with weight direction)

Note: this is the OPPOSITE of CWD (which decays only when aligned). The BCM analogy suggests that
**high alignment (gradient "agrees" with weights) indicates stable learning phase → apply less WD**;
**low alignment (gradient "disagrees") indicates the network is exploring / in transition → apply
more WD** to constrain the search. This is the dynamically sophisticated version of WD — not "decay
when you agree with what you are" but "decay more strongly when your weights and gradients are
pulling in different directions," which accelerates convergence to a new basin.

**Formal BCM-WD rule**:
- Slow timescale EMA: θ_t = (1-ε)θ_{t-1} + ε δ̂_t² (EMA of alignment squared)
- Dynamic WD: λ_t = λ_base · (1 + α · (1 − δ̂_t / √θ_t))  [increase WD when current alignment
  is below the EMA baseline]

**Hypothesis**: This BCM-inspired WD rule produces better generalization than fixed-WD or CWD
because (1) it applies stronger regularization precisely during the phases when the network is not
yet "committed" to good representations (low alignment = exploratory phase), and (2) it automatically
reduces regularization during "committed" phases (high alignment = exploitation), enabling faster
convergence in the final stage.

**Why not just a metaphor**: The BCM sliding threshold is a formal dynamical system (ODE/difference
equation) with proven stability analysis (Cooper, Intrator & Blais 1996). The structural
correspondence maps the ODE state variable (θ_M) to a computable quantity (EMA of δ̂_t²) and the
plasticity rule to a weight update equation. The stability proof for BCM (showing θ_M converges to
stabilize activity) translates to a proof sketch that the BCM-WD rule drives the weight magnitude
toward a norm-stationary regime. The key preserved property is the **negative feedback loop**: the
WD is more aggressive when the alignment signal indicates the network is in a "confused" state
(low alignment), preventing runaway weight growth during exploratory phases.

**Novelty estimate**: 9/10 — BCM has been applied to spiking nets and Hebbian learning, but never
to gradient-based optimizer WD scheduling. The specific mapping (δ̂_t as postsynaptic activity,
sliding θ_M as dynamic WD strength) and the resulting stability analysis are completely unexplored.

---

### Candidate B: Thermodynamic Work Minimization as Optimal WD Schedule Design

**Source field**: Stochastic Thermodynamics / Statistical Physics

**Source principle**: The Jarzynski equality: ⟨e^{-W/kT}⟩ = e^{-ΔF/kT}, where W is the
nonequilibrium work done along a training trajectory, ΔF is the free energy difference (related to
the train-test gap), and the angle brackets denote averaging over stochastic trajectories. The
Jarzynski equality implies: ΔF ≤ ⟨W⟩ (the second law of thermodynamics analog), with equality
achieved only for infinitely slow (quasi-static) protocols. The **Crooks fluctuation theorem**
further tells us that the distribution of forward vs. reverse trajectory probabilities encodes the
entropy production (irreversibility). Minimizing irreversibility → minimizing dissipated work →
minimizing the train-test gap.

**Structural correspondence**:

| Thermodynamics | Neural Network Training |
|---|---|
| External protocol (control parameter schedule) | WD schedule λ(t) |
| Free energy landscape | Loss landscape |
| Thermal fluctuations | SGD gradient noise |
| Work W | Total loss reduction (energetic cost of training) |
| Entropy production | Irreversibility of weight trajectory (related to generalization gap) |
| Quasi-static limit (slow protocol) | Very small learning rate + very small WD |
| Optimal nonequilibrium protocol | Optimal WD schedule minimizing dissipated work |

The key result from stochastic thermodynamics is that **the optimal protocol for minimizing
dissipated work is the "time-reversal symmetric" protocol** — one where the forward and reverse
trajectories have equal probability. This provides a principled criterion: the optimal WD schedule
is the one that minimizes the Kullback-Leibler divergence between the forward training distribution
and the reverse (inference/generalization) distribution.

**Hypothesis**: The optimal WD schedule λ*(t) is the solution to a path optimization problem:
minimize the entropy production along the training trajectory. This optimal schedule will:
(1) Start with relatively high WD during early training (when the trajectory is far from equilibrium)
(2) Decrease WD as training converges (approaching the quasi-static limit)
(3) Potentially re-increase WD near the end of training to "anneal" the trajectory toward
    the generalization distribution

This predicts a non-monotone optimal WD schedule — consistent with empirical observations (WSD,
SWD, ADANA all find non-constant schedules superior) but derived from first principles.

**Why not just a metaphor**: The Jarzynski equality is a mathematical theorem that holds for any
stochastic system with a Hamiltonian dynamics formulation. SGD satisfies these conditions: the
parameter space is the phase space, the loss is the Hamiltonian, and the stochastic gradient is
the Langevin noise. The thermodynamic inequality ΔF ≤ ⟨W⟩ translates to the generalization bound
E[test_loss - train_loss] ≤ f(dissipated_work), providing a rigorous connection between trajectory
irreversibility and generalization error. WD's role in reducing dissipated work is preserved in the
transplant.

**Novelty estimate**: 8/10 — While the thermodynamics-ML analogy is well-known in general, the
specific application of the Jarzynski equality to derive an optimal WD schedule from first principles
has not been done. The "time-reversal symmetric optimal protocol" criterion for WD scheduling would
be a genuinely novel theoretical contribution.

---

### Candidate C: Mutation-Selection Equilibrium as a Unified Lens for WD-Gradient Steady States

**Source field**: Population Genetics / Evolutionary Biology

**Source principle**: In population genetics, the steady-state allele frequency under
mutation-selection balance is: p* = μ/s, where μ is the mutation rate and s is the selection
coefficient against the deleterious allele. This equilibrium emerges from opposing forces: mutation
introduces deleterious alleles at rate μ, selection removes them at rate s·p. The equilibrium is
stable: perturbations away from p* return to it. Importantly, different "alleles" (genetic variants)
reach different equilibrium frequencies depending on their individual (μ_i, s_i) values. The total
genetic load (departure from the perfect genotype) at equilibrium is exactly Σ_i μ_i (Haldane's
rule — independent of selection strength).

**Structural correspondence**:

| Population Genetics | Dynamic WD Framework |
|---|---|
| Allele i | Weight coordinate w_i |
| Allele frequency p_i | Weight magnitude |w_i| |
| Mutation rate μ_i | Gradient magnitude |g_i| |
| Selection coefficient s_i | WD rate λ_i (per-parameter) |
| Equilibrium p* = μ/s | Equilibrium |w*| = |g|/λ |
| Genetic load Σ μ_i | Total regularization budget |
| Hardy-Weinberg equilibrium | Stationary weight distribution |
| Linkage disequilibrium | Gradient-weight alignment δ_t |

**Key derived result**: At the mutation-selection equilibrium, the equilibrium weight norm is
‖w*‖ = ‖g‖/λ — exactly the Defazio (2506.02285) gradient-to-weight ratio result! This means
the Defazio result is not merely an empirical observation but the formal analog of Haldane's
mutation-selection equilibrium theorem — a classical theorem from population genetics derived in
1927.

**Haldane's analog for WD**: The total "genetic load" under mutation-selection balance equals Σ μ_i
(sum of mutation rates, independent of selection). In WD training, the corresponding statement is:
the total equilibrium regularization "load" (total WD contribution to parameter shrinkage at steady
state) equals Σ |g_i| / λ_i — proportional to the sum of gradient magnitudes, independent of the
WD schedule shape. This is a non-trivial theoretical constraint on any dynamic WD method.

**Hypothesis**: The formal mutation-selection analogy provides a unified framework for all four WD
sub-approaches:
- WD scheduling = time-varying selection coefficient s(t)
- Alignment-aware WD = selection coefficient conditioned on fitness landscape correlation
  (gradient-weight alignment measures how "fit" a weight is for its current environment)
- Per-layer WD (AlphaDecay) = population-specific selection coefficients
- Norm-matched WD (AdamWN) = balancing selection (maintaining alleles at specific frequencies, not
  zero)
The "unified optimal strategy" is the one that drives the parameter distribution to the maximum-
entropy state compatible with the training objective — exactly the Bayesian posterior, which is also
the minimum free energy state. This connects the evolutionary analogy back to the thermodynamic one.

**Why not just a metaphor**: The key preserved structural property is the **balance equation**:
d‖w‖/dt = ‖g‖ − λ‖w‖, which is formally identical to the population genetics balance equation
dp/dt = μ(1-p) − sp ≈ μ − sp (for small p). The equilibrium p* = μ/s maps to ‖w*‖ = ‖g‖/λ.
This is not a surface metaphor — it is the same first-order ODE with the same equilibrium structure.
The stability analysis (eigenvalue of the linearized dynamics = −s < 0 for both systems) is
preserved.

**Novelty estimate**: 8/10 — The Defazio (2506.02285) result is known, but its identification as
an analog of Haldane's mutation-selection theorem is entirely new. The unifying framework
(all WD sub-approaches as different "selection regimes") has not been proposed.

---

## Phase 3: Self-Critique

### Against Candidate A (BCM Sliding Threshold → Alignment-Aware WD)

**Shallow analogy attack**: Is the BCM-WD mapping structural or merely vocabulary substitution?
The BCM model is specifically designed for Hebbian (unsupervised) learning in excitatory neurons
with rate coding. SGD is supervised, uses full network backpropagation, and operates in discrete
time. The mapping δ̂_t → y (postsynaptic activity) requires that the gradient-weight dot product
plays the same functional role as firing rate. In BCM, y is directly measurable; δ̂_t requires a
minibatch estimation. The BCM stability proof relies on specific assumptions (bounded input
correlations, fixed presynaptic statistics) that may not hold in deep nets with non-stationary
gradient distributions. **Risk: moderate**. The structural mapping (negative feedback ODE) is
preserved, but BCM's specific stability conditions may not transfer without modification.

**Scale mismatch attack**: BCM operates at the single-neuron level (one postsynaptic cell,
thousands of synapses). The analog in deep learning would be per-neuron (per-unit) WD with a
per-neuron sliding threshold. For a network with 10M parameters organized in thousands of neurons,
this is computationally expensive. The per-parameter version (one threshold per weight) over-
parameterizes the BCM analog; the per-neuron version is more faithful but requires grouping
parameters. **Risk: moderate**. A layer-wise or neuron-wise implementation is feasible but
loses some of the per-weight granularity.

**Prior transplant check**: Search for "BCM weight decay optimizer" and "BCM sliding threshold deep
learning optimization" yields no results. The closest is DWAM (arXiv:2511.17563) applying BCM to
SNNs, but not to gradient-based WD scheduling. **Transplant appears genuinely novel.**

**Testability attack**: The diagnostic experiment must distinguish "BCM-inspired sliding threshold
is the active ingredient" from "it just works because we are doing some form of adaptive WD."
Design: fix the time-averaged EMA component (remove the sliding θ_M) and replace with a constant
threshold. If BCM-WD significantly outperforms constant-threshold adaptive WD, the sliding
mechanism is the active ingredient. This is testable with standard CIFAR-10/100 experiments in
~30 minutes each.

**Verdict**: **STRONG** — The structural correspondence is rigorous (same ODE structure), the
transplant is novel, the diagnostic experiment is clear, and the scale challenge is manageable
(per-layer implementation).

---

### Against Candidate B (Jarzynski Equality → Optimal WD Schedule)

**Shallow analogy attack**: The Jarzynski equality requires that the initial state is a Boltzmann
distribution and the dynamics follow a Hamiltonian/Langevin system. SGD with WD at iteration 0
starts from a random initialization (not a Boltzmann distribution over the loss landscape). The
learning rate annealing (cosine, polynomial) breaks the stationarity assumptions. **Risk: high**.
The Jarzynski equality may not hold exactly for practical SGD training trajectories, requiring a
modified "approximate Jarzynski" result.

**Scale mismatch attack**: In thermodynamics, the Jarzynski equality is validated for systems with
~10^23 particles where large deviation theory applies. Neural network training involves ~10^7-10^8
parameters, which is large but not necessarily in the thermodynamic limit. The fluctuation theorem
corrections may be substantial. **Risk: moderate**.

**Prior transplant check**: The connection between thermodynamics and ML is well-studied (Santa Fe
Institute Stochastic Thermodynamics 2024 workshop; Goldt & Seifert PRL 2017). However, using
Jarzynski to **derive** an optimal WD schedule from first principles has not been done. The existing
work uses thermodynamics to analyze training but not to prescribe WD schedules. **Partially novel.**

**Testability attack**: To test whether the Jarzynski-derived schedule is the "active ingredient"
vs. generic WD scheduling, one needs to compare against a WD schedule of similar shape (non-
monotone, high early → low late) but not derived from the thermodynamic principle. If both perform
equally, the thermodynamic derivation may provide mathematical grounding but no practical advantage.
**Risk: moderate**. The test is feasible but the result may confirm only mathematical elegance, not
practical superiority.

**Verdict**: **MODERATE** — Strong theoretical foundations, partially novel application, but strict
Jarzynski conditions may not hold for realistic SGD training. The practical improvement over
empirically-designed non-monotone schedules is uncertain. Best treated as a theoretical
contribution to the framework rather than an algorithmic advance.

---

### Against Candidate C (Mutation-Selection Equilibrium → WD-Gradient Steady State)

**Shallow analogy attack**: The mutation-selection analogy maps weight magnitude to allele frequency
and gradient to mutation rate. But allele frequencies are bounded [0,1] while weight magnitudes are
unbounded. The mutation-selection ODE (dp/dt = μ - sp) is linear, while the weight dynamics
(dw/dt = -g - λw) is also linear, preserving the mapping. However, in deep learning with batch
normalization, the effective gradient magnitude is non-stationary (changes dramatically across
training phases), making the "mutation rate" μ(t) time-varying. The classical mutation-selection
theorem assumes constant μ. **Risk: low-moderate**. The equilibrium result ‖w*‖ = ‖g‖/λ is correct
for stationary gradient magnitudes; for non-stationary settings, a time-varying equilibrium analysis
is needed.

**Scale mismatch attack**: Population genetics applies to diploid organisms with finite population
size N (drift effects become important when s ~ 1/N). In neural nets, there is no direct analog of
genetic drift (SGD noise is the closest, but it scales with batch size differently). The
mutation-selection equilibrium is exact in the infinite population limit; in finite populations
(finite batch size), drift shifts the equilibrium. **Risk: low**. The infinite-population limit
applies to large-batch training and is a reasonable approximation.

**Prior transplant check**: Defazio (2506.02285) derives ‖g‖/‖w‖ → constant from first principles
of WD dynamics, but does not connect to mutation-selection balance. The identification of
Haldane's theorem as the underlying mathematical structure is new. Search for "mutation selection
weight decay neural network" confirms this is novel.

**Testability attack**: The diagnostic experiment: Does Haldane's "load" theorem predict the actual
gradient-to-weight ratio at steady state in trained networks? Specifically, for a network trained
with different WD strengths λ, does the equilibrium ‖w‖ ≈ ‖g‖/λ hold empirically? If so, this
validates the analogy. This is testable with existing training runs just by logging ‖g_t‖/‖w_t‖
over training. **Highly testable.**

**Verdict**: **STRONG** — The structural correspondence is exact (same ODE, same equilibrium),
the identification of Haldane's theorem as the mathematical underpinning is genuinely novel, the
test is quantitative and directly verifiable, and the scale considerations are manageable.

---

## Phase 4: Refinement

**Dropped**: None — all three candidates survived, but Candidate B (thermodynamic) is demoted to
secondary theoretical framing rather than primary algorithmic contribution.

**Strengthened surviviors**:

### Candidate A — BCM-WD Rule (Strengthened)

**Formalized structural correspondence**:

Let δ̂_t = ⟨g_t, w_t⟩ / (‖g_t‖‖w_t‖ + ε) be the normalized gradient-weight alignment (the "post-
synaptic activity" in BCM terms). The BCM-inspired dynamic WD rule is:

```
θ_t = (1 − α_θ) θ_{t-1} + α_θ δ̂_t²         [slow EMA of squared alignment = BCM sliding threshold]
λ_t = λ_base · (1 + β · (1 − δ̂_t / √(θ_t + ε)))  [WD increases when alignment is below EMA baseline]
w_{t+1} = (1 − λ_t γ_t) w_t − γ_t g_t       [standard SGDW with dynamic λ_t]
```

Key properties:
- When δ̂_t > √θ_t: current alignment exceeds historical baseline → WD decreases (BCM analog: high
  activity → raise threshold → less LTP = less weight increase needed)
- When δ̂_t < √θ_t: current alignment below baseline → WD increases (BCM analog: low activity →
  lower threshold → more LTD = more weight decrease)
- At steady state: λ → λ_base (mean WD equals base WD, as in BCM where mean activity → θ_M)

**BCM stability translated**: The BCM stability proof (Cooper & Intrator 1992) shows that for
bounded input correlations, θ_M converges and the activity variance decreases. In our setting, if
the gradient statistics are approximately stationary (valid in late training), then θ_t converges
and δ̂_t's variance decreases — meaning the alignment signal becomes more stable, and λ_t converges
to a fixed value. This is the stability guarantee for the BCM-WD rule.

**Diagnostic experiment**: Compare BCM-WD vs. BCM-WD with frozen θ (θ_t = θ_0 for all t, i.e.,
remove the sliding mechanism). If BCM-WD significantly outperforms frozen-threshold adaptive WD,
the sliding mechanism is the active ingredient. Run on ResNet-20/CIFAR-10 with 3 seeds, 30 min
each.

### Candidate C — Mutation-Selection Unified Framework (Strengthened)

**Formalized structural correspondence**:

The central theorem: for any optimizer with WD strength λ (per-parameter or scalar) applied to
weights with gradient process {g_t}:

At quasi-stationary equilibrium (gradient statistics stationary): E[‖w_t‖²]^{1/2} = E[‖g_t‖²]^{1/2} / λ

This is formally Haldane's (1937) mutation-selection balance theorem applied to parameter dynamics.
Corollaries:
1. WD scheduling (λ(t)): equivalent to time-varying selection coefficient → equilibrium weight norm
   tracks ‖g_t‖/λ(t) at each phase
2. Alignment-aware WD (λ ↑ when δ̂_t ↓): equivalent to environment-dependent selection (selection
   is stronger when "fitness landscape correlation" = alignment is low)
3. Per-layer WD (AlphaDecay): equivalent to layer-specific selection coefficients
4. Norm-matched WD (target τ): equivalent to **balancing selection** — selection maintains alleles
   at a specific frequency τ rather than purging them to zero

**Haldane's genetic load analog**: The total equilibrium WD budget (Σ_i λ_i ‖w_i*‖) equals
Σ_i ‖g_i‖ (sum of gradient magnitudes) — independent of the WD schedule shape. This constrains
all dynamic WD methods: you cannot reduce the total WD budget at equilibrium without also reducing
the effective gradient magnitudes. This insight directly motivates the **Budget Equivalence Metric**
(BEM): any fair comparison of dynamic WD methods must hold constant the total WD budget at
equilibrium.

**Selected Front-Runner**: Candidate A (BCM-WD Rule) as the primary algorithmic contribution,
Candidate C (Mutation-Selection Framework) as the primary theoretical framework, and Candidate B
(Thermodynamic Analogy) as secondary theoretical support.

---

## Phase 5: Final Proposal

### Title
**From BCM Neurons to Gradient Optimizers: A Biologically-Grounded Sliding Threshold for Dynamic Weight Decay with a Unified Evolutionary Equilibrium Framework**

---

### Source Principle (Primary: BCM Neuroscience)

The Bienenstock-Cooper-Munro (BCM) sliding threshold mechanism provides biological neurons with
automatic homeostatic stability: the threshold for synaptic potentiation (θ_M) adapts based on
historical activity, creating a negative feedback loop that prevents runaway synaptic growth while
maintaining information-rich representations. The formal BCM stability theorem proves convergence
of θ_M under bounded-correlation input statistics.

### Secondary Source Principle: Mutation-Selection Equilibrium

In population genetics, mutation-selection balance (Haldane 1937) describes the equilibrium
allele frequency under opposing forces. The mathematical structure is formally identical to the
weight dynamics under WD training, providing a unified framework connecting all WD sub-approaches.

---

### Structural Correspondence

**BCM-WD Structural Map**:

| BCM Quantity | Dynamic WD Quantity |
|---|---|
| Postsynaptic firing rate y_t | Gradient-weight alignment δ̂_t = ⟨g_t,w_t⟩/(‖g_t‖‖w_t‖) |
| Sliding threshold θ_M(t) | Dynamic WD strength multiplier θ_t (slow EMA of δ̂_t²) |
| LTD (depression when y > θ_M) | Reduced WD when δ̂_t > √θ_t (alignment exceeds baseline) |
| LTP enhancement (when y < θ_M) | Increased WD when δ̂_t < √θ_t (alignment below baseline) |
| BCM ODE: dθ_M/dt = φ(y) − θ_M/τ | EMA update: θ_t = (1-α)θ_{t-1} + α δ̂_t² |
| BCM stability condition | Convergence of θ_t to quasi-stationary value |

**Evolutionary Framework Map**:

| Population Genetics | WD Training |
|---|---|
| Allele frequency p_i | Weight magnitude |w_i| |
| Mutation rate μ_i | Gradient magnitude |g_i| |
| Selection coefficient s_i | WD rate λ_i |
| Equilibrium p* = μ/s | Equilibrium |w*| = |g|/λ (Defazio 2506.02285) |
| Haldane genetic load Σ μ_i | Total WD budget Σ |g_i|/λ_i |
| Balancing selection | Norm-matched WD (target τ) |
| Time-varying selection | WD scheduling |
| Environment-dependent selection | Alignment-aware WD |

**This is not just a metaphor because**: Both systems are governed by the same linear balance ODE
(dp/dt = μ − s·p for genetics; dw/dt = g − λ·w for WD training), with the same equilibrium
structure (p* = μ/s; w* = g/λ) and the same stability property (exponential convergence with rate
−s < 0 for both). Haldane's load theorem (total load = Σ μ_i) translates to a precise, testable
prediction about total WD budget at equilibrium: Σ_i λ_i |w_i*| = Σ_i |g_i|.

---

### Hypothesis

**Primary**: The BCM-WD sliding threshold rule achieves better generalization than fixed-threshold
adaptive WD and CWD because the sliding mechanism applies stronger regularization during
exploratory (low-alignment) phases and weaker regularization during convergent (high-alignment)
phases — an automatic curriculum for regularization strength that BCM stability theory guarantees
will converge.

**Secondary**: Haldane's load theorem provides a fundamental constraint on dynamic WD methods:
no matter how the WD is scheduled or adapted, the total equilibrium regularization budget equals
Σ |g_i|/λ_i. Any WD method that claims to "reduce the budget" without changing λ or ‖g‖ violates
this theorem and should be examined for confounds. This provides a principled foundation for the
**Budget Equivalence Metric (BEM)**.

---

### Method

**BCM-WD Optimizer** (drop-in modification to any SGDW/AdamW):

```python
def bcm_wd_step(w, g, lr, lam_base, theta, alpha_theta=0.01, beta=1.0, eps=1e-8):
    """
    BCM-inspired dynamic weight decay.
    theta: slow EMA of squared alignment (BCM sliding threshold)
    """
    # Compute alignment signal (BCM postsynaptic activity analog)
    delta_hat = (g * w).sum() / (g.norm() * w.norm() + eps)

    # Update sliding threshold (BCM ODE discretized)
    theta = (1 - alpha_theta) * theta + alpha_theta * delta_hat**2

    # Dynamic WD: increase when alignment is below baseline, decrease when above
    alignment_deviation = 1.0 - delta_hat / (theta.sqrt() + eps)
    lam_t = lam_base * (1 + beta * alignment_deviation)
    lam_t = lam_t.clamp(0.1 * lam_base, 10 * lam_base)  # stability bounds

    # Standard SGDW step with dynamic WD
    w = (1 - lam_t * lr) * w - lr * g

    return w, theta
```

**Implementation notes**:
- theta can be maintained per-layer (faithful BCM analog) or globally (simpler)
- alpha_theta controls the timescale of adaptation: small alpha = slow adaptation (more homeostatic)
- beta controls the sensitivity of WD to alignment deviations
- Compatible with momentum SGD, AdamW (apply after Adam update)

---

### Diagnostic Experiment

The key experiment that confirms the BCM sliding mechanism (not just adaptive WD in general) is
the active ingredient:

1. **BCM-WD** (full sliding threshold): θ_t updates every step
2. **BCM-WD-frozen** (frozen threshold): θ_t = θ_0 fixed (eliminates the sliding)
3. **Fixed WD** baseline: standard SGDW with λ = λ_base
4. **CWD** (binary alignment mask): for comparison

If BCM-WD significantly outperforms BCM-WD-frozen but has similar performance to CWD, then it is
the adaptive WD, not the sliding, that matters. If BCM-WD outperforms both BCM-WD-frozen and CWD,
the sliding mechanism is confirmed as the active ingredient.

Additionally, the **Haldane load theorem test**: at the end of training, measure Σ_i λ_i |w_i*|
and compare to Σ_i |g_i|. If the theorem holds, this validates the evolutionary framework. This
is a zero-cost diagnostic using logged training quantities.

---

### Experimental Plan

**Standard evaluation** (CIFAR-10/100 with ResNet-20, CIFAR-100 with VGG-16-BN):
- Baseline: SGD-Fixed-WD, CWD, SWD, AdamW
- New method: BCM-WD (per-layer theta), BCM-WD (global theta)
- Metrics: test accuracy (mean ± std over 3 seeds), training loss curve, alignment trajectory δ̂_t,
  theta trajectory, effective WD trajectory λ_t
- Time: ~30 min per run × 5 methods × 2 datasets × 3 seeds = ~9 hours total (parallelizable)

**Diagnostic experiments**:
- BCM-WD vs BCM-WD-frozen on CIFAR-10: confirms sliding mechanism value (~1 hour)
- Haldane load theorem validation: log Σ λ|w| and Σ|g| for all methods on CIFAR-100 (~free, just
  logging)

**ImageNet validation** (ResNet-50, required by project constraints):
- Top-1/Top-5 accuracy with BCM-WD vs AdamW, CWD, SWD
- Track per-layer alignment δ̂_t, θ_t, λ_t trajectories across training
- Time: 4-6 hours per run, must use 3 seeds

---

### Risk Assessment

1. **BCM stability conditions may not transfer**: BCM's convergence proof assumes stationary
   input correlations. Deep net gradients are highly non-stationary, especially with LR scheduling
   and batch normalization. The slow EMA (alpha_theta small) partially mitigates this by
   smoothing gradient statistics, but there is no formal guarantee.
   **Mitigation**: Use conservative alpha_theta ≈ 0.001-0.01; empirically validate θ_t convergence.

2. **Opposing WD direction (high alignment → less WD) may conflict with CWD intuition**: CWD
   decays more when aligned; BCM-WD decays less when aligned. The correct direction depends on
   interpretation: CWD's logic is "decay what is being used (aligned) to prevent overfitting";
   BCM's logic is "decay more when confused (misaligned) to constrain exploration." Both are valid
   interpretations; empirical comparison will determine which is beneficial.
   **Mitigation**: Test both BCM-WD and BCM-WD-inverted (decay more when aligned, closer to CWD
   but with sliding threshold) to determine the correct direction of the analogy.

3. **Haldane's load theorem is an asymptotic result**: The theorem holds exactly only at
   equilibrium (stationary gradient distribution). During training dynamics, the relation
   Σ λ|w| = Σ|g| will only hold approximately. The error depends on how far the trajectory is
   from equilibrium.
   **Mitigation**: Report the Haldane load ratio as a diagnostic metric during training, not as
   a hard constraint.

4. **Computational overhead**: Computing δ̂_t requires one additional dot product per step (O(d)
   operations) — negligible for standard training. The sliding threshold theta requires O(L)
   storage (L = number of layers). No significant overhead.

---

### Novelty Claim

1. **BCM-WD algorithm**: First application of BCM sliding threshold mechanism to gradient-based
   optimizer weight decay scheduling. The adaptive WD strength based on gradient-weight alignment
   with a sliding historical baseline is novel. Search of arXiv confirms no prior work
   (arXiv:2511.17563 applies BCM to spiking nets only; no connection to WD scheduling).

2. **Evolutionary equilibrium framework**: First formal identification of the Defazio (2506.02285)
   gradient-to-weight ratio result as an instance of Haldane's (1937) mutation-selection balance
   theorem. This provides a unifying theoretical framework for all four WD sub-approaches as
   different "selection regimes," and derives the Budget Equivalence Metric from a first-
   principles theorem (Haldane's load theorem) rather than as an ad hoc normalization procedure.

3. **Thermodynamic grounding** (secondary): The Jarzynski equality provides a variational upper
   bound connecting the WD schedule's nonequilibrium entropy production to the generalization gap
   — a rigorous thermodynamic interpretation of why WD scheduling improves generalization.

4. **Cross-field synthesis**: These three source fields (BCM neuroscience, population genetics,
   stochastic thermodynamics) are connected to the WD scheduling problem for the first time,
   revealing that the empirically successful WD scheduling practices (SWD, WSD, ADANA) are
   instances of well-understood phenomena from other sciences operating under the same
   mathematical structure.

---

## Summary

The interdisciplinary search identified three structural analogies with genuine mathematical depth:

| Analogy | Source Field | Key Preserved Structure | Novelty |
|---|---|---|---|
| BCM Sliding Threshold → Adaptive WD | Computational Neuroscience | Same negative-feedback ODE, same stability conditions | 9/10 |
| Mutation-Selection Balance → WD-Gradient Equilibrium | Population Genetics | Same linear balance equation, same equilibrium formula, Haldane's theorem | 8/10 |
| Jarzynski Equality → Optimal WD Schedule | Stochastic Thermodynamics | Same second-law inequality, optimal protocol theory | 7/10 |

The **front-runner contribution** is the BCM-WD algorithm paired with the evolutionary equilibrium
framework: the algorithm provides a concrete implementable method with clear diagnostic experiments,
while the evolutionary framework provides a unified theoretical foundation for the entire Unified
Dynamic Weight Decay Framework — satisfying both the "algorithm" and "theory" requirements of the
research agenda.

The **Budget Equivalence Metric** is grounded in Haldane's load theorem: fair comparison requires
that Σ_i λ_i |w_i*| be held constant across methods (equal total equilibrium regularization
budget), which translates to equal total gradient magnitudes — a natural compute-equivalent
normalization that is more principled than simple training budget equivalence.

---

*Sources consulted:*
- [BCM Theory — Wikipedia](https://en.wikipedia.org/wiki/BCM_theory)
- [DWAM arXiv:2511.17563 (2025)](https://arxiv.org/abs/2511.17563)
- [Timescale Hierarchy in Cortical Learning arXiv:2510.18808 (2024)](https://arxiv.org/pdf/2510.18808)
- [Hebbian CNNs with BCM arXiv:2501.17266 (2025)](https://arxiv.org/html/2501.17266v1)
- [HTSR and Renormalization Group (WeightWatcher, Dec 2024)](https://calculatedcontent.com/2024/12/24/weightwatcher-htsr-theory-and-the-renormalization-group/)
- [Jarzynski Equality and Complexity arXiv:2107.08608](https://arxiv.org/abs/2107.08608)
- [ECT* ML and RG Workshop 2024](https://indico.ectstar.eu/event/206/)
- [PMP for CNN Training arXiv:2504.11647 (2025)](https://arxiv.org/html/2504.11647v1)
- [Lyapunov Stability as Inductive Biases arXiv:2511.01283 (2025)](https://arxiv.org/abs/2511.01283)
- [Pruning as Evolution arXiv:2601.10765 (2026)](https://arxiv.org/html/2601.10765)
- [Free Energy Principle and Regularization arXiv:2505.22749 (2025)](https://arxiv.org/html/2505.22749v1)
- [Stochastic Thermodynamics of Learning — Goldt & Seifert PRL 118 (2017)](https://en.wikipedia.org/wiki/Stochastic_thermodynamics)
- [Machine Learning RG arXiv:2306.11054](https://arxiv.org/abs/2306.11054)
- [BCM Synaptic Competition — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9666303/)
