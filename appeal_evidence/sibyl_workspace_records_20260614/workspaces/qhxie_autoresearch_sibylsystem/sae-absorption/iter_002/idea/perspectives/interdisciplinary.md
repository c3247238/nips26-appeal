# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Immunology / Evolutionary Biology
1. **Ferretti & Kardar (2024), "Universal characterization of epitope immunodominance from a multi-scale model of clonal competition in germinal centers," Phys. Rev. E 109, 064409** -- Formal mathematical model (parabolic Anderson model on hypercubes) of how dominant B-cell clones suppress subdominant clones in germinal centers. Immunodominance emerges from statistical properties of affinity landscapes, not from any single-clone property. The structural analogy to feature absorption is striking: a "parent" clone with broad antigen affinity absorbs the niche of more specific clones, exactly as a child feature absorbs a parent feature in SAEs.
2. **PNAS (2024), "A speed limit on serial strain replacement from original antigenic sin"** -- Mathematical model showing that Original Antigenic Sin (OAS) -- where memory B cells from a prior infection suppress de novo responses to novel epitopes -- imposes quantifiable bounds on pathogen evolution speed. OAS is immunological "absorption": a dominant memory response absorbs the immune system's capacity to mount a fresh response.
3. **Gause (1934), competitive exclusion principle / Lotka-Volterra competition** -- Two species with identical niches cannot coexist; the dominant competitor absorbs the niche of the weaker. Coexistence requires intraspecific competition exceeding interspecific competition (alpha_12 * alpha_21 < 1). This is the ecological formalization of the absorption dynamic.
4. **eLife (2025), "Innate immunity and training to subvert original antigenic sin"** -- Mechanistic detail: dominant memory B cells consume available antigen, lowering naive B cell activation probability. This is exactly the "sparsity budget consumption" mechanism in SAE absorption.
5. **Modern coexistence theory (Chesson 2000; Levine et al. 2024)** -- Formalizes conditions under which competing species coexist despite niche overlap: stabilizing niche differences must exceed fitness differences. Directly maps to conditions under which parent and child features can both be represented in an SAE.

#### Statistical Physics / Information Theory
6. **"A Renormalization Group Framework for Scale-Invariant Feature Learning in Deep Neural Networks" (AAAI 2025)** -- Formal correspondence between neural network layers and RG coarse-graining transformations. Each layer implements a coarse-graining operator that contracts Fisher information geometry. Predicts required depth scales logarithmically with data correlation length.
7. **"Interpreting Deep Learning via Renormalization Group on the Ising Model" (Science China Mathematics, 2025)** -- Rigorous mapping between DNN feature extraction and RG transformations on the Ising model. The Ising model's spontaneous symmetry breaking (where one magnetization direction "absorbs" the other below T_c) provides a formal phase-transition analogy for feature absorption.
8. **"Parameter Symmetry Breaking and Restoration Determines Hierarchical Learning in AI Systems" (arXiv 2502.05300, 2025)** -- Identifies expansion, reduction, and transmission phases in deep learning, governed by symmetry breaking. The reduction phase where irrelevant information is discarded is analogous to the sparsity-induced absorption where redundant features are pruned.
9. **Information Bottleneck method (Tishby et al. 1999; Agmon 2024)** -- Rate-distortion framework for lossy compression of relevant information. Sub-optimal IB solutions collide or exchange optimality at bifurcations of the rate-information curve. This directly models the sparsity-fidelity tradeoff in SAEs and predicts absorption as a bifurcation phenomenon where two features merge into one coding unit.
10. **MIPT-SSM (arXiv 2604.07716, 2026)** -- Neural sequence architecture with a measurement-induced phase transition between a "wave phase" (distributed interference) and a "particle phase" (state collapse). The phase transition at critical sequence length provides an analogy for how SAE sparsity constraints induce a transition from distributed to localized (absorbed) feature representations.

#### Computational Neuroscience
11. **Vafaii, Galor & Yates (NeurIPS 2025), "Brain-like Variational Inference" (FOND/iP-VAE)** -- Derives that minimizing variational free energy under Poisson assumptions yields spiking neural networks with emergent lateral competition and sparse coding. Competitive interactions underlie the sparsification dynamics observed in cortical circuits. The lateral competition mechanism provides a biologically principled alternative to L1/TopK sparsity that may avoid absorption.
12. **Coultrip, Granger & Lynch (1992), "A cortical model of winner-take-all competition via lateral inhibition"** -- Seminal model of how excitatory-inhibitory circuits in cortex implement WTA competition. The key insight: WTA naturally creates a hierarchy where the strongest-activated neuron suppresses others, which is the neural mechanism underlying absorption-like dynamics.
13. **Xie, Hahnloser & Seung (2002), "Learning winner-take-all competition between groups of neurons in lateral inhibitory networks"** -- Shows how to organize lateral inhibition so that groups (not individual neurons) compete. This group-WTA mechanism is directly relevant to feature splitting in SAEs, where groups of child features compete with a parent feature.
14. **Sparse-Coding Variational Autoencoders (Geadah et al., Neural Computation 2024)** -- Formal connection between sparse coding and VAE inference, showing that sparsity emerges from probabilistic inference under appropriate priors rather than from explicit L1 penalties.

#### Signal Processing / Compressed Sensing
15. **He & Joseph (2024), "A Hierarchical View of Structured Sparsity in Kronecker Compressive Sensing" (arXiv:2409.08699)** -- Hierarchical sparsity model where nonzero entries concentrate in a few blocks, each with only a subset of nonzero entries. The RIP analysis for hierarchical sparsity provides mathematical tools for understanding when hierarchical features can be separately recovered vs. when they collapse (absorb).
16. **Mallat (1989), Multiresolution Analysis / Wavelet Representation** -- The foundational framework for hierarchical signal decomposition. Wavelets decompose signals into detail coefficients at multiple scales, where each scale captures different structural information. The perfect reconstruction property guarantees no information loss across scales -- exactly what SAEs fail to achieve when absorption occurs.
17. **"Hierarchical Resolution Transformers" (arXiv:2509.20581, 2025)** -- Wavelet-inspired neural architecture processing language at multiple resolutions simultaneously. Three principles from wavelet theory (multi-resolution decomposition, frequency separation, perfect reconstruction) directly address the feature hierarchy problem in SAEs.

### Cross-Disciplinary Gaps

The following structural correspondences have **not** been explored in the SAE absorption literature:

1. **Immunodominance --> Feature absorption**: No paper has formalized the analogy between B-cell immunodominance (dominant clones suppress subdominant clones through antigen consumption) and SAE feature absorption (child features suppress parent features through sparsity budget consumption). The mathematical model of Ferretti & Kardar (2024) provides ready-made analytical tools.

2. **Lotka-Volterra competitive exclusion --> Feature coexistence conditions**: The ecology literature provides precise mathematical conditions for when two competing species coexist (alpha_12 * alpha_21 < 1) vs. when one excludes the other. No analogous formal condition has been derived for when two hierarchically related SAE features can coexist vs. when absorption occurs.

3. **Renormalization group coarse-graining --> Multi-scale feature representation**: While the RG-deep learning analogy is active, it has not been applied to the specific problem of feature absorption in SAEs, where the "coarse-graining" of a parent feature into a child feature is precisely the phenomenon to be understood.

4. **Wavelet perfect reconstruction --> Absorption-free SAE design**: The wavelet literature's perfect reconstruction condition has not been adapted as a design principle for SAEs to guarantee that hierarchical features are represented without mutual interference.

---

## Phase 2: Initial Candidates

### Candidate A: Immunodominance-Absorption Isomorphism (from Immunology)

- **Source principle**: Immunodominance in germinal centers -- dominant B-cell clones with broad antigen affinity suppress subdominant clones by consuming shared antigen (the limited resource). The dominant clone's antibodies bind available antigen before naive/subdominant B cells can access it, preventing their clonal expansion. Ferretti & Kardar (2024) model this as competition on disconnected fitness landscapes, where the dominant class's affinity landscape has a higher peak, and clones on that landscape proliferate faster, consuming the shared antigen resource.

- **Structural correspondence**: 
  | Immunology | SAE Feature Absorption |
  |---|---|
  | Antigen (shared resource) | Reconstruction variance (shared resource) |
  | Dominant B-cell clone (broad affinity) | Child feature (specific, captures both parent and child variance) |
  | Subdominant B-cell clone (narrow affinity) | Parent feature (general, captures only parent variance) |
  | Antigen consumption by dominant clone | Sparsity budget consumption by child feature |
  | Immunodominance hierarchy | Feature absorption hierarchy |
  | Clonal expansion rate | Feature activation frequency |
  | Affinity maturation (somatic hypermutation) | SAE training (gradient updates) |
  | Germinal center capacity | SAE L0 budget |
  | Original Antigenic Sin | Absorption persistence across training |

  The mathematical structure is isomorphic: in both systems, a limited shared resource (antigen / reconstruction variance) is consumed by the entity with highest fitness/utility (dominant clone / child feature), preventing the lower-fitness entity (subdominant clone / parent feature) from accessing the resource. The sparsity penalty in SAEs plays the role of immune system resource constraints.

- **Hypothesis**: Feature absorption rate can be predicted by an "immunodominance score" computed from the ratio of child-feature affinity (cosine similarity to the activation subspace) to parent-feature affinity, modulated by the "antigen availability" (fraction of variance unexplained by other features). Specifically, the Ferretti-Kardar quasispecies dynamics should predict that absorption probability is a sigmoidal function of the affinity ratio, with a critical threshold determined by the SAE's L0 budget.

- **Why it's not just a metaphor**: The mathematical structure is identical -- both are instances of competitive resource allocation under capacity constraints, governed by fitness-proportional selection. The Ferretti-Kardar model uses quasispecies dynamics (a well-characterized mathematical object) that can be directly adapted to describe feature competition during SAE training. The key preserved property is the "antigen consumption" mechanism: in immunology, dominant clones physically consume antigen molecules; in SAEs, child features consume reconstruction variance, reducing the gradient signal that would train the parent feature. Both lead to suppression of the subdominant entity through resource depletion, not through direct inhibition.

- **Novelty estimate**: 8/10 -- The immunodominance-absorption analogy has never been proposed in the SAE literature. The mathematical tools (quasispecies dynamics, affinity landscape competition) are well-developed in immunology but have not been applied to dictionary learning.

### Candidate B: Lotka-Volterra Feature Coexistence Theory (from Mathematical Ecology)

- **Source principle**: The Lotka-Volterra competitive exclusion principle and modern coexistence theory. Two species competing for the same niche coexist if and only if intraspecific competition exceeds interspecific competition (alpha_12 * alpha_21 < 1). The competition coefficients alpha quantify niche overlap. Character displacement -- evolutionary divergence of competing species' traits -- is the ecological analog of feature splitting.

- **Structural correspondence**:
  | Ecology | SAE Feature Dynamics |
  |---|---|
  | Species i population N_i | Feature i activation magnitude a_i |
  | Carrying capacity K_i | Maximum reconstruction contribution of feature i |
  | Intrinsic growth rate r_i | Learning rate * gradient magnitude for feature i |
  | Competition coefficient alpha_ij | Cosine similarity between decoder vectors d_i and d_j |
  | Niche overlap | Shared variance between features |
  | Competitive exclusion | Feature absorption |
  | Character displacement | Feature splitting |
  | Coexistence equilibrium | Both parent and child features active |
  | Resource (food, space) | Reconstruction variance (L2 loss budget) |

  The Lotka-Volterra equations for two features become:
  ```
  da_1/dt = r_1 * a_1 * (1 - a_1/K_1 - alpha_12 * a_2/K_1)
  da_2/dt = r_2 * a_2 * (1 - a_2/K_2 - alpha_21 * a_1/K_2)
  ```
  where alpha_ij = cosine_similarity(d_i, d_j) * variance_overlap(i,j). Absorption occurs when alpha_12 * alpha_21 > 1 (interspecific competition dominates), causing one feature to go extinct (activation -> 0).

- **Hypothesis**: The product alpha_12 * alpha_21 (computed from decoder cosine similarities and variance overlaps) provides a closed-form predictor of absorption. For any pair of hierarchically related features, absorption occurs when this product exceeds a critical threshold determined by the SAE's sparsity constraint. The critical threshold decreases with increasing sparsity (lower L0 budget), exactly as resource scarcity increases competitive pressure in ecology.

- **Why it's not just a metaphor**: The Lotka-Volterra framework provides concrete, falsifiable predictions: (1) a closed-form coexistence condition (alpha product < 1), (2) the direction of exclusion (which feature absorbs which), and (3) the equilibrium activation magnitudes when coexistence occurs. These predictions can be directly tested by computing decoder similarities and activation statistics from existing SAEs, without any training. The mathematical correspondence is not between surface phenomena but between the optimization dynamics themselves.

- **Novelty estimate**: 7/10 -- The Lotka-Volterra framework has been applied to neural network competition in other contexts (e.g., GANs, multi-task learning), but never specifically to feature competition in sparse autoencoders or to predict feature absorption.

### Candidate C: Renormalization Group Coarse-Graining and Phase Transitions (from Statistical Physics)

- **Source principle**: The renormalization group (RG) in statistical physics implements systematic coarse-graining: microscopic degrees of freedom are integrated out to produce effective theories at larger scales. At critical points (phase transitions), the system exhibits scale invariance, and RG fixed points characterize universal behavior. The Ising model's spontaneous symmetry breaking below T_c, where one magnetization direction "absorbs" the other, provides a formal phase-transition framework.

- **Structural correspondence**:
  | Statistical Physics | SAE Feature Absorption |
  |---|---|
  | Microscopic degrees of freedom | Fine-grained (child) features |
  | Effective (coarse-grained) degrees of freedom | Coarse-grained (parent) features |
  | RG transformation | SAE encoding (projection to sparse basis) |
  | Irrelevant operators (integrated out) | Absorbed features (suppressed by sparsity) |
  | Relevant operators (survive coarse-graining) | Active features (above sparsity threshold) |
  | Critical temperature T_c | Critical sparsity level L0* |
  | Spontaneous symmetry breaking | Absorption onset (parent feature extinction) |
  | RG fixed point | Converged SAE feature configuration |
  | Universality class | Architecture-invariant absorption pattern |

  The key insight: feature absorption is a phase transition. As sparsity increases (analogous to lowering temperature below T_c), the system transitions from a "disordered phase" (both parent and child features active) to an "ordered phase" (only child feature active, parent absorbed). The critical sparsity L0* at which this transition occurs should be predictable from the "coupling strength" between parent and child features (analogous to the interaction strength J in the Ising model).

- **Hypothesis**: There exists a critical sparsity level L0* for each parent-child feature pair, below which absorption is inevitable and above which both features coexist. L0* is determined by the feature hierarchy depth, co-occurrence frequency, and decoder similarity -- analogous to how T_c is determined by coupling constants and lattice geometry. Near L0*, absorption should exhibit critical fluctuations (intermittent on/off behavior of the parent feature), and the absorption-sparsity curve should follow a universal scaling law with critical exponents independent of the specific SAE architecture.

- **Why it's not just a metaphor**: The RG framework makes quantitative predictions about scaling behavior near the critical point. Specifically: (1) the absorption rate A(L0) should follow A ~ |L0 - L0*|^beta near the critical sparsity, with beta a universal exponent; (2) the "susceptibility" (variance of parent feature activation) should diverge at L0*; (3) features within the same universality class should show identical critical exponents regardless of architecture (L1 vs. TopK vs. JumpReLU). These are concrete, falsifiable predictions that go far beyond metaphor.

- **Novelty estimate**: 6/10 -- The RG-deep learning connection is actively studied (AAAI 2025, Science China 2025), and phase transitions in neural networks are a known research area. However, applying the RG phase-transition framework specifically to feature absorption in SAEs, with predictions about critical exponents and universality, is novel.

---

## Phase 3: Self-Critique

### Against Candidate A (Immunodominance)

- **Shallow analogy attack**: The correspondence is deeper than vocabulary mapping. Both systems involve: (i) a shared limited resource consumed competitively, (ii) fitness-proportional selection under resource constraints, (iii) hierarchical structure where broad-affinity entities subsume narrow-affinity ones. The Ferretti-Kardar quasispecies dynamics and the SAE training dynamics are both gradient-like optimization processes on fitness landscapes under capacity constraints. An immunologist would recognize the structural parallel: antigen consumption and sparsity budget consumption are both instances of competitive resource depletion driving selective suppression. **Verdict: correspondence is structural, not superficial.**

- **Scale mismatch attack**: Immunological systems involve ~10^3--10^6 B-cell clones competing for ~10^2--10^4 epitopes. SAEs have ~10^4--10^6 features competing for ~10^1--10^2 L0 slots. The scales are comparable. The germinal center operates over days-weeks of affinity maturation; SAE training operates over thousands of gradient steps. Both are iterated selection processes with comparable numbers of selection rounds. **No major scale mismatch.**

- **Prior transplant check**: Searched arXiv for "immunodominance" + "sparse autoencoder", "immunodominance" + "dictionary learning", "clonal competition" + "feature learning" -- no results. The immunodominance analogy has been used in reinforcement learning (immune-inspired exploration), but never in sparse dictionary learning or feature absorption. **Novel transplant.**

- **Testability attack**: The diagnostic experiment is clear: compute an "immunodominance score" (affinity ratio modulated by antigen availability) for each parent-child feature pair, then test whether it predicts the measured absorption rate. The null hypothesis (absorption is random, not predicted by the immunodominance score) is easily falsifiable. The specific prediction (sigmoidal absorption probability as a function of affinity ratio, with threshold set by L0 budget) is quantitative and distinguishable from simpler hypotheses (e.g., absorption proportional to cosine similarity alone). **Testable.**

- **Verdict: STRONG**

### Against Candidate B (Lotka-Volterra)

- **Shallow analogy attack**: The mapping from species populations to feature activations is mathematically rigorous: both are dynamical systems where entities compete for a shared resource under capacity constraints. The Lotka-Volterra equations are a well-characterized dynamical system with known stability conditions, and the mapping preserves the key structural property (coexistence iff intraspecific > interspecific competition). However, one concern: SAE features are not independently reproducing populations -- they are updated simultaneously by gradient descent, which introduces correlations not present in the Lotka-Volterra model. The correspondence is structural at the level of equilibrium analysis (coexistence conditions) but may break down for transient dynamics. **Correspondence is structural for equilibria, weaker for dynamics.**

- **Scale mismatch attack**: The Lotka-Volterra model is designed for 2-species or few-species competition. SAEs have ~10^4--10^6 features. The multi-species Lotka-Volterra generalization exists (random matrix theory approaches, Akjouj et al. 2024) but the analysis becomes complex. For the specific application to parent-child feature pairs, the 2-species model is sufficient since absorption is a pairwise phenomenon. **Manageable if restricted to pairwise analysis.**

- **Prior transplant check**: Searched arXiv for "Lotka-Volterra" + "sparse autoencoder", "competitive exclusion" + "feature learning", "Lotka-Volterra" + "dictionary learning" -- no direct results. The Lotka-Volterra framework has been applied to GANs (adversarial dynamics), multi-agent RL, and network pruning, but not to SAE feature dynamics. **Novel in the SAE context.**

- **Testability attack**: The key prediction (absorption iff alpha_12 * alpha_21 > 1) can be computed from decoder cosine similarities and variance overlaps of existing pre-trained SAEs, then compared against measured absorption rates. This requires no new training. The null hypothesis (alpha product does not predict absorption) is clearly falsifiable. One concern: the alpha coefficients may be hard to estimate accurately from finite activation data. **Testable, with practical estimation challenges.**

- **Verdict: MODERATE-STRONG**

### Against Candidate C (Renormalization Group)

- **Shallow analogy attack**: The RG-deep learning correspondence is well-studied, but applying it to SAE feature absorption specifically requires showing that (1) the SAE encoding is a bona fide coarse-graining transformation, and (2) feature absorption corresponds to a genuine phase transition with critical exponents. Concern: SAE encoding is a single-step projection, not an iterated coarse-graining. The RG analogy is most natural for deep networks where each layer is a coarse-graining step, but SAEs are typically single-layer. The phase transition language may be evocative but not technically precise unless one can identify a genuine order parameter and critical behavior. **The single-layer nature of SAEs weakens the RG analogy considerably.**

- **Scale mismatch attack**: Phase transitions require thermodynamic limits (N -> infinity). SAEs with ~10^4 features may not be large enough to exhibit genuine critical behavior, and "critical exponents" may not be well-defined. The Ising model analogy requires translational invariance and local interactions, while SAE features have long-range correlations (any two features can interact via the encoder). **Significant scale and structure mismatch.**

- **Prior transplant check**: The RG-deep learning connection is a large and active literature (AAAI 2025, NeurIPS workshops). Phase transitions in neural networks are well-studied (double descent, grokking). However, the specific application to SAE feature absorption as a phase transition has not been proposed. **Semi-novel -- new application of existing framework.**

- **Testability attack**: Measuring critical exponents requires systematically varying sparsity near the critical point and fitting power laws. This is feasible with existing Gemma Scope SAEs (which span multiple sparsity levels). However, distinguishing a genuine phase transition from a smooth crossover requires very fine-grained sparsity sweeps that may not be available in existing SAE suites. The "universality" prediction (same exponents across architectures) is powerful if confirmed but requires training SAEs with multiple architectures at matched conditions. **Partially testable with existing resources; full test requires new SAE training.**

- **Verdict: MODERATE (weakened by single-layer SAE structure and measurement difficulty)**

---

## Phase 4: Refinement

### Dropped
- **Candidate C (Renormalization Group)**: While intellectually attractive, the single-layer structure of SAEs fundamentally weakens the RG coarse-graining analogy. The phase transition predictions are interesting but hard to test with existing resources and may not exhibit genuine critical behavior at finite feature counts. The analogy is inspirational rather than load-bearing.

### Strengthened: Candidate A (Immunodominance) -- Selected as Front-Runner

**Formalized structural correspondence:**

Let F = {f_1, ..., f_n} be the SAE features and H = {(f_p, f_c)} the set of parent-child pairs in the feature hierarchy. Define:

1. **Affinity** of feature f_i to activation subspace S: A(f_i, S) = ||proj_S(d_i)||^2 / ||d_i||^2, where d_i is the decoder vector.

2. **Antigen availability** for feature f_i: R(f_i) = Var(x - sum_{j != i} a_j * d_j), the residual variance available to feature f_i.

3. **Immunodominance score** for a parent-child pair (f_p, f_c):
   ```
   ID(f_p, f_c) = A(f_c, S_{p,c}) / A(f_p, S_{p,c}) * R(f_c) / R(f_p)
   ```
   where S_{p,c} is the subspace spanned by both features' activation patterns.

4. **Absorption prediction**: P(absorption | f_p, f_c) = sigma(ID(f_p, f_c) - theta(L0))
   where sigma is the sigmoid function and theta(L0) is a threshold that increases with L0 budget (more sparsity budget -> higher threshold -> less absorption).

This maps directly onto the Ferretti-Kardar framework where:
- A(f_i, S) corresponds to the peak affinity of a B-cell clone for an epitope
- R(f_i) corresponds to the available antigen concentration for clone i
- ID(f_p, f_c) corresponds to the immunodominance ratio between competing epitope classes
- theta(L0) corresponds to the germinal center capacity

**Diagnostic experiment design:**

The key test that confirms the analogy is load-bearing:

1. Compute ID(f_p, f_c) for all parent-child pairs across Gemma Scope SAEs using the Chanin et al. (2024) first-letter task to identify pairs.
2. Compare ID score against measured absorption rate. If the immunodominance framework is correct:
   - ID score should predict absorption rate with R^2 > 0.5 (better than cosine similarity alone)
   - The sigmoid threshold theta should shift predictably with L0
   - ID score should generalize to new feature hierarchies (e.g., entity type, knowledge taxonomies)
3. **Critical control**: Compare against a "naive" predictor using only decoder cosine similarity. If ID score (which includes antigen availability / residual variance) significantly outperforms cosine similarity alone, this confirms that the resource-depletion mechanism (from immunology) is the active ingredient, not just geometric proximity.

**Additional support from immunology:**

The immunology literature provides two additional predictions not obvious from the SAE literature alone:

1. **Original Antigenic Sin analog**: Once a child feature has absorbed a parent feature early in training, subsequent training will reinforce rather than reverse the absorption, because the child feature's decoder vector has been shaped to capture the parent's variance. This predicts that absorption is training-order dependent and that SAEs trained with curriculum learning (general features first, then specific) should show lower absorption than standard random-order training.

2. **Affinity maturation analog**: During SAE training, features undergo "affinity maturation" (progressive refinement of decoder vectors). The immunodominance framework predicts that absorption worsens during training (as features become more specialized), consistent with the empirical observation that longer-trained JumpReLU SAEs show worse absorption (literature survey finding).

### Strengthened: Candidate B (Lotka-Volterra) -- Retained as Complementary Framework

The Lotka-Volterra framework is retained as a complementary analytical tool because it provides a different and independently testable prediction:

**Coexistence condition**: Parent feature f_p and child feature f_c coexist (no absorption) iff:
```
alpha_pc * alpha_cp < 1
```
where alpha_pc = cos(d_p, d_c)^2 * freq_overlap(p,c) measures the competitive effect of the child on the parent, and alpha_cp is the reverse.

This is equivalent to: **the parent and child features must have sufficiently different "niches" (activation patterns) that each limits its own activation more than the other's.**

The ecological framework additionally predicts **character displacement**: when two features are forced to coexist (e.g., in wider SAEs), their decoder vectors should diverge, analogous to how competing species evolve more distinct niches when sympatric. This predicts that wider SAEs should show not just less absorption but also greater decoder angle between parent and child features -- a prediction that can be tested on Gemma Scope SAEs at different widths.

---

## Phase 5: Final Proposal

### Title: Feature Absorption as Immunodominance: A Cross-Disciplinary Framework for Understanding and Predicting Hierarchical Feature Competition in Sparse Autoencoders

### Source Principle

Immunodominance in adaptive immunity is the phenomenon where dominant B-cell clones with high affinity for an antigen epitope suppress subdominant clones by consuming the shared antigen resource, preventing subdominant clones from expanding and maturing. This is formalized by Ferretti & Kardar (2024) as competitive quasispecies dynamics on disconnected fitness landscapes, where the statistical properties of the affinity landscape determine which epitope class dominates. The key mechanism is **resource depletion**: the dominant clone physically consumes the antigen, reducing its availability to competitors. Original Antigenic Sin (OAS) extends this: memory from a prior dominant response suppresses de novo responses to novel antigens, creating persistent immunological blind spots. The PNAS (2024) model shows OAS imposes a "speed limit" on immune adaptation to new strains.

### Structural Correspondence

The formal mapping between immunodominance and SAE feature absorption:

**Resource competition structure**: In both systems, entities (B-cell clones / SAE features) compete for a shared limited resource (antigen / reconstruction variance) under a capacity constraint (germinal center size / L0 sparsity budget). The entity with higher fitness (affinity / reconstruction utility) consumes more of the resource, reducing the gradient signal for competitors.

**Hierarchy and dominance**: In both systems, hierarchical structure (epitope specificity hierarchy / semantic feature hierarchy) creates asymmetric competition where the more specific entity (higher-affinity clone / child feature) can capture the resource that would otherwise go to the more general entity (lower-affinity clone / parent feature), because the specific entity encodes all the information the general one does plus more.

**Persistence mechanism**: In both systems, dominance is self-reinforcing: in immunology, dominant clones' antibodies feed back to suppress naive B cell activation (OAS); in SAEs, the child feature's gradient reinforces its decoder to capture parent variance, making reversal increasingly unlikely during training.

**Mathematical formalization**:

Define the immunodominance score for a parent-child feature pair (f_p, f_c):

```
ID(f_p, f_c) = [A(f_c, S_shared) / A(f_p, S_shared)] * [R(f_c) / R(f_p)]
```

where:
- A(f_i, S) = affinity of feature i to shared activation subspace = ||proj_S(d_i)||^2 / ||d_i||^2
- R(f_i) = residual variance available to feature i = Var(x - reconstruction_without_i)
- S_shared = subspace of activations where both features are relevant

Absorption probability:
```
P(absorption) = sigmoid(ID(f_p, f_c) - theta(L0))
```

where theta(L0) is a monotonically increasing function of the L0 budget (higher L0 -> more "germinal center capacity" -> higher threshold for absorption).

Complementary Lotka-Volterra coexistence condition:
```
Coexistence iff alpha_pc * alpha_cp < 1
where alpha_ij = cos^2(d_i, d_j) * overlap(activation_patterns_i, activation_patterns_j)
```

### Hypothesis

1. **Primary**: The immunodominance score ID(f_p, f_c) predicts absorption rate across parent-child feature pairs with significantly higher accuracy than existing predictors (decoder cosine similarity, co-occurrence frequency alone).

2. **Secondary (OAS analog)**: Absorption established early in training is irreversible -- SAEs exhibit "antigenic sin" where early feature configurations constrain later feature learning. Curriculum training (general -> specific) should reduce absorption compared to standard training.

3. **Tertiary (Lotka-Volterra complement)**: The product of competition coefficients alpha_pc * alpha_cp provides a necessary-and-sufficient condition for absorption: absorption occurs iff this product exceeds a critical value that depends on L0.

4. **Corollary (character displacement)**: In wider SAEs where absorption is reduced, parent-child decoder vectors should show greater angular separation than in narrow SAEs -- the SAE analog of ecological character displacement.

### Method

**Phase 1: Compute immunodominance scores** (training-free, existing SAEs)
1. Load Gemma Scope SAEs at multiple widths (16k, 65k, 131k) and layers
2. Identify parent-child feature pairs using the Chanin et al. first-letter task (existing methodology)
3. Compute ID scores for all identified pairs
4. Measure absorption rate using the Chanin et al. metric
5. Fit the sigmoid model and estimate theta(L0)

**Phase 2: Test predictive power**
1. Compare ID score vs. baselines (cosine similarity, co-occurrence frequency, random) as predictors of absorption
2. Cross-validate across SAE widths and layers
3. Test generalization to entity-type hierarchies (city -> country, animal -> species)

**Phase 3: Test OAS and Lotka-Volterra predictions**
1. Compute alpha products for all feature pairs; test coexistence condition
2. Measure decoder angle separation across SAE widths (character displacement test)
3. If resources permit: train small SAEs with curriculum vs. standard ordering, measure absorption difference (OAS test)

### Diagnostic Experiment

The key experiment that confirms the immunodominance analogy is load-bearing (not decorative):

**Experiment**: For 200+ parent-child feature pairs across Gemma Scope SAEs, compute three predictors of absorption: (a) decoder cosine similarity alone, (b) ID score without residual variance term (affinity ratio only), (c) full ID score (affinity ratio * residual variance ratio).

**If the immunodominance framework is correct**: Predictor (c) should significantly outperform (a) and (b), because the resource-depletion mechanism (residual variance) is the active ingredient transplanted from immunology. If only (a) or (b) suffices, the immunological analogy adds no explanatory power beyond geometric proximity.

**If the Lotka-Volterra framework is correct**: The alpha product threshold should separate absorbed from non-absorbed pairs with AUC > 0.8, and the threshold should shift predictably with L0.

### Experimental Plan

- **Models**: Gemma 2 2B (via Gemma Scope SAEs), GPT-2-small (via EleutherAI SAEs)
- **SAE configurations**: Multiple widths (16k, 65k, 131k for Gemma; 768, 3072, 12288 for GPT-2), multiple layers
- **Tasks**: First-letter spelling (established), entity-type hierarchy (novel extension)
- **Libraries**: SAELens, sae-spelling (Chanin et al. codebase), TransformerLens
- **Compute**: All analyses are training-free (compute ID scores and alpha products from pre-trained SAE parameters and activation statistics); small-scale SAE training for OAS experiment only
- **Time estimate**: Phase 1 (ID score computation + absorption measurement): ~30 min per SAE configuration. Phase 2 (cross-validation): ~1 hour total. Phase 3 (Lotka-Volterra + character displacement): ~1 hour total. OAS curriculum experiment: ~1 hour if needed.

### Risk Assessment

1. **ID score may not outperform cosine similarity**: If decoder geometry alone (cosine similarity) is sufficient to predict absorption, the immunological analogy adds complexity without explanatory power. Mitigation: the residual variance term (the immunology-specific component) is independently meaningful and may capture cases where geometry alone fails.

2. **First-letter task is too narrow**: The first-letter hierarchy may not be representative of richer semantic hierarchies. Mitigation: extend to entity-type hierarchies as a validation set.

3. **Lotka-Volterra assumptions may not hold**: SAE features are trained jointly by gradient descent, not independently reproducing populations. The Lotka-Volterra equilibrium analysis may not capture the transient dynamics of training. Mitigation: restrict analysis to converged SAEs (equilibrium only), which is the experimentally accessible regime.

4. **Immunodominance model assumes discrete competing classes**: The Ferretti-Kardar model has discrete epitope classes, while SAE features occupy a continuous high-dimensional space. Mitigation: the analysis is restricted to identified parent-child pairs, which form discrete competing classes.

### Novelty Claim

The specific cross-disciplinary insight is the formal identification of SAE feature absorption with immunodominance in adaptive immunity, grounded in the shared mathematical structure of competitive resource allocation under capacity constraints. This insight has three novel consequences:

1. **A new quantitative predictor of absorption** (the immunodominance score) that incorporates resource depletion dynamics beyond simple geometric proximity.
2. **A testable prediction about training-order dependence** (Original Antigenic Sin analog) that suggests curriculum learning as a principled mitigation strategy.
3. **A formal coexistence condition** (Lotka-Volterra complement) that provides the first closed-form necessary-and-sufficient condition for when two hierarchically related features can coexist in an SAE.

Evidence of novelty: arXiv searches for "immunodominance" + "sparse autoencoder", "immunodominance" + "dictionary learning", "Lotka-Volterra" + "feature absorption", and "competitive exclusion" + "sparse coding" returned zero results. The immunological perspective on sparse coding pathologies has not been explored.
