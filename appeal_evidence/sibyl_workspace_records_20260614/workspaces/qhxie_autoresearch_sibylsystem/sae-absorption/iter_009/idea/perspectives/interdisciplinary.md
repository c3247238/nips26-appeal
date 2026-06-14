# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Immunology: Immunodominance and Original Antigenic Sin
1. **Frank (2002), "Immunology and Evolution of Infectious Disease," Ch. 13: Immunodominance Within Hosts** -- The immune system organizes T cell and B cell responses into reproducible hierarchies where dominant epitopes suppress subdominant ones. The mechanism involves active regulatory suppression by induced regulatory T cells (iTregs), competition for APC surface, and BCR affinity thresholds ("Goldilocks zone"). Key insight: dominant responses actively suppress nascent competing responses, not merely outcompete them passively.
2. **Francis (1960), "Original Antigenic Sin" / Gostic et al. (2016)** -- The immune system preferentially recalls antibody repertoires from the first-encountered pathogen variant, even when subsequent variants would benefit from de novo responses. Memory B cells outcompete naive B cells for germinal center access, creating a persistent bias toward early-imprinted features. This is a direct analog of how early-learned SAE features absorb later features.
3. **Epitope-spanning antigenic variation (Nature Communications, 2026)** -- Targeted variation across multiple hemagglutinin head sites between sequential vaccines can reprogram epitope hierarchy, redirecting recall toward conserved subdominant epitopes. This demonstrates that immunodominance hierarchies can be broken by specific training regime modifications.
4. **eLife (2025), "Innate immunity and training to subvert original antigenic sin"** -- Trained innate immunity (e.g., via beta-glucan) enhances germinal center dynamics and enables naive B cells to compete more effectively against dominant memory B cells. Demonstrates that environmental modification (analogous to training regime changes) can counteract dominance hierarchies.

#### Ecology: Competitive Exclusion, Niche Partitioning, and Character Displacement
5. **Gause (1934), Competitive Exclusion Principle** -- Two species competing for identical resources cannot stably coexist; the superior competitor always wins. Directly parallels how a high-frequency child feature "excludes" the parent feature in an SAE when they compete for the same reconstruction capacity.
6. **Ohlert et al. (Ecology, 2025), "Role of dominant species in semiarid grasslands"** -- 23-year removal experiment showing that dominant grasses suppress both richness and abundance of subordinate species. When dominants are removed, subordinate diversity increases but total production decreases. This mirrors the absorption-reconstruction tradeoff: removing absorption improves feature diversity but worsens reconstruction.
7. **Zou et al. (Journal of Animal Ecology, 2025), "Temporal niche partitioning"** -- Challenges assumption that temporal partitioning always promotes coexistence; sometimes leads to priority effects rather than coexistence. Relevant caution: simply partitioning SAE features temporally or spatially may not solve absorption.
8. **Character displacement (Brown & Wilson, 1956)** -- Competing species develop increasingly distinct traits in sympatry to minimize niche overlap. Analogous to how orthogonality constraints (OrtSAE) force features to diverge, reducing absorption.

#### Neuroscience: Winner-Take-All, Lateral Inhibition, and Sparse Coding
9. **Coultrip, Granger & Lynch (1992), "A cortical model of winner-take-all competition via lateral inhibition"** -- In cortical networks, lateral inhibitory feedback causes the strongest-responding neuron to suppress all competitors, converging to a solution that relays the winner's identity while suppressing losers with arbitrary precision. Feature absorption in SAEs mirrors this: the strongest-activating (most specific) feature suppresses the weaker (more general) feature.
10. **Pastor et al. (Journal of Neurophysiology, 2025), "Dynamics of energy-efficient coding in visual cortex"** -- Stimulus onset initially drives broad cortical activation (low sparseness, high mutual information); over time, competitive interactions refine the population response, maintaining information as sparseness increases. Demonstrates that the brain's sparse coding naturally undergoes an absorption-like process where broad features are refined into specific ones.
11. **Vafaii, Galor & Yates (NeurIPS 2025), "Brain-like Variational Inference"** -- Shows variational free energy minimization unifies VAEs, sparse coding, and predictive coding. Under Poisson assumptions, variational inference yields sparse representations with Gabor-like basis vectors. Key insight: sparse coding and SAEs are both performing approximate inference, and absorption is a consequence of the inference objective.
12. **Competition, stability, and functionality in excitatory-inhibitory circuits (arXiv, Dec 2025)** -- Combines game-theoretic and energetic perspectives on lateral inhibition microcircuits as contrast enhancers, formalizing how networks maintain stability while amplifying differences. Provides mathematical tools (Lyapunov diagonal stability) applicable to analyzing SAE feature competition.

#### Statistical Physics: Phase Transitions and Symmetry Breaking
13. **Ambrogioni et al. (Entropy, 2025), "Statistical Thermodynamics of Generative Diffusion Models"** -- Generative diffusion models undergo second-order phase transitions corresponding to symmetry breaking, always in a mean-field universality class due to self-consistency conditions. Feature emergence in generative models maps to spontaneous symmetry breaking in physics. Absorption could be understood as a symmetry-breaking phenomenon where the SAE's loss landscape has degenerate minima.
14. **Landau theory of phase transitions (classical)** -- Order parameters emerge when the free energy landscape develops multiple minima; the system "chooses" one. In SAEs, absorption represents the system choosing to represent a parent-child pair with one feature (absorbed state) rather than two (non-absorbed state), analogous to a symmetry-broken ground state.

#### Information Theory: Rate-Distortion and Compression
15. **Rate-Distortion-Perception trade-off (PMC, 2025)** -- Incorporating perceptual constraints into lossy compression does not decrease required rate; sometimes additional rate is needed. Directly analogous to how eliminating absorption (improving "perceptual quality" of features) requires higher L0 (more rate).
16. **Geometry of efficient codes (PLOS Comp Bio, 2025)** -- Under rate-distortion trade-offs, latent representations undergo prototypization, specialization, and orthogonalization. Absorption is a form of specialization where the code collapses general features into specific ones to minimize rate at a given distortion.

#### Genetics: Epistasis and Dominance
17. **Phillips (2008), "Epistasis -- the essential role of gene interactions" (Nature Reviews Genetics)** -- Epistasis (masking of one gene's effect by another) is the genetic mechanism underlying canalization. In classical genetics, epistasis orders genes in regulatory pathways: a dominant gene masks the effect of a recessive gene. Feature absorption is structurally analogous: a specific (child) feature masks the activation of a general (parent) feature.
18. **Lehner et al. (Nature Communications, 2023), "Dominance vs epistasis: biophysical origins"** -- Non-additive interactions arise naturally even in simple biophysical systems; altering ligand concentration can switch alleles from dominant to recessive. Analogously, changing SAE hyperparameters (sparsity penalty, dictionary size) may switch which features dominate/absorb others.

### Cross-Disciplinary Gaps

The following cross-field transplants have NOT been explored in the SAE/mechanistic interpretability literature:

1. **Immunodominance-inspired mitigation**: No work has applied the immunological insight that "trained innate immunity can subvert original antigenic sin" to SAE training. The structural analog would be a training regime that prevents early-learned features from permanently dominating later ones.

2. **Ecological character displacement as a formal framework**: While OrtSAE uses orthogonality penalties (superficially similar to character displacement), no work has formalized the connection to ecological coexistence theory or used the mathematical framework of Lotka-Volterra competition models to predict absorption severity.

3. **Rate-distortion theory for absorption quantification**: No work has used the rate-distortion framework to derive a closed-form expression for the absorption rate as a function of SAE parameters. The information-theoretic perspective could yield the missing "quantitative causal theory of absorption magnitude" identified as Gap 1 in the literature.

4. **Phase transition framework for absorption onset**: No work has modeled the transition from non-absorbed to absorbed states as a phase transition with a well-defined critical point, order parameter, and universality class.

---

## Phase 2: Initial Candidates

### Candidate A: Immunodominance Hierarchy Theory (from Immunology)

- **Source principle**: In adaptive immunity, when multiple epitopes compete for T cell and B cell responses, immunodominant epitopes actively suppress subdominant ones through: (1) memory B cells outcompeting naive B cells for germinal center access, (2) induced regulatory T cells (iTregs) causing premature contraction of slower responses, and (3) BCR affinity competition creating a "Goldilocks zone" where intermediate-affinity responses dominate. Crucially, this is not passive competition but active suppression, and the hierarchy is established by the temporal order of encounters (original antigenic sin).

- **Structural correspondence**: 
  - Epitopes --> SAE dictionary atoms (features)
  - Antibody titers --> Feature activation magnitudes
  - Germinal center --> The SAE's optimization landscape during training
  - Memory B cells outcompeting naive B cells --> Early-learned specific features suppressing later general features during training
  - iTreg-mediated premature contraction --> Sparsity penalty actively suppressing features that co-occur with stronger ones
  - Immunodominance hierarchy --> Feature absorption hierarchy (child absorbs parent)
  - Original antigenic sin --> The order in which features are learned during training creates persistent biases
  - Trained innate immunity subverting OAS --> Modified training regimes (e.g., Matryoshka nested losses, masked regularization) breaking the absorption hierarchy

- **Hypothesis**: Feature absorption severity is predicted by the temporal order in which features are learned during SAE training (analogous to antigenic seniority). Features learned earlier will absorb features learned later when they share overlapping activation patterns. Furthermore, absorption can be disrupted by "vaccination-like" training interventions that periodically reset the competition -- analogous to how heterotypic vaccination reprograms epitope hierarchies.

- **Why not just a metaphor**: The formal correspondence is between the Lotka-Volterra competition dynamics governing T cell clonal expansion and the gradient dynamics governing SAE feature activation during training. Both systems minimize an objective (immune fitness / reconstruction loss) under a resource constraint (limited APC surface / limited L0 budget). The key testable prediction -- that temporal learning order determines absorption hierarchy -- goes beyond metaphor and makes a falsifiable claim about SAE training dynamics.

- **Novelty estimate**: 8/10 -- The immunodominance framework has never been applied to SAE feature dynamics, and the specific prediction about temporal learning order is novel and testable.

### Candidate B: Rate-Distortion Phase Transition Theory (from Statistical Physics + Information Theory)

- **Source principle**: In rate-distortion theory, lossy compression of a source at rate R bits with distortion D follows a convex trade-off curve R(D). At critical values of the Lagrange multiplier beta (which controls the rate-distortion trade-off), the optimal codebook undergoes phase transitions: new codewords emerge (or existing ones merge) discontinuously. These bifurcations in the information bottleneck framework correspond to spontaneous symmetry breaking in statistical physics, where the free energy landscape develops new minima. The transitions follow mean-field universality (Ambrogioni et al., 2025).

- **Structural correspondence**:
  - Source distribution --> Distribution of LLM activations
  - Codebook/codewords --> SAE dictionary atoms
  - Rate R --> L0 (number of active features, sparsity)
  - Distortion D --> Reconstruction error (MSE)
  - Lagrange multiplier beta --> Sparsity penalty coefficient (lambda for L1, k for TopK)
  - Codebook bifurcation (codeword merging) --> Feature absorption (parent feature merges into child)
  - Phase transition critical point --> Critical sparsity level at which absorption onset occurs
  - Order parameter --> Absorption rate
  - Free energy landscape --> SAE loss landscape

- **Hypothesis**: Feature absorption is a phase transition in the rate-distortion sense. There exists a critical sparsity level beta_c (controlled by L1 coefficient or TopK k) below which absorption does not occur and above which it emerges discontinuously. The absorption rate (order parameter) follows a scaling law near the critical point: A ~ (beta - beta_c)^gamma, where gamma is a universal exponent determined by the feature hierarchy structure. This yields a closed-form prediction of absorption severity as a function of SAE hyperparameters.

- **Why not just a metaphor**: The SAE objective IS a rate-distortion problem: minimize reconstruction error (distortion) subject to a sparsity constraint (rate). This is not an analogy -- it is mathematical identity. The unified SDL theory paper (arXiv:2512.05534) already casts SAE training as piecewise biconvex optimization; the phase transition framework provides the missing piece by characterizing the bifurcation structure of the solutions. The prediction of critical exponents and scaling laws near the absorption onset is a concrete, quantitative claim derivable from rate-distortion theory.

- **Novelty estimate**: 9/10 -- While the connection between SAEs and lossy compression has been noted (Bereska et al., 2025; Ayonrinde et al., 2024), nobody has derived the phase transition structure of absorption onset or predicted scaling laws for absorption rate as a function of sparsity parameters.

### Candidate C: Ecological Niche Partitioning and Competitive Exclusion (from Ecology)

- **Source principle**: In ecology, Gause's competitive exclusion principle states that N species cannot stably coexist on fewer than N resources. When species compete for the same resources, the superior competitor excludes the inferior one. Coexistence is maintained through niche partitioning (spatial, temporal, or trophic differentiation) and character displacement (evolutionary divergence of traits in sympatry). The Lotka-Volterra competition equations formalize these dynamics: dx_i/dt = r_i * x_i * (1 - sum_j(alpha_ij * x_j / K_i)), where alpha_ij is the competition coefficient between species i and j.

- **Structural correspondence**:
  - Species --> SAE features (dictionary atoms)
  - Resource --> Reconstruction capacity (share of the activation vector each feature explains)
  - Species abundance --> Feature activation magnitude
  - Competitive exclusion --> Feature absorption (parent feature excluded by child)
  - Competition coefficient alpha_ij --> Decoder cosine similarity between features i and j
  - Carrying capacity K_i --> Maximum activation magnitude for feature i (set by sparsity constraint)
  - Niche partitioning --> Orthogonality constraints (OrtSAE) forcing features to explain different aspects
  - Character displacement --> Feature splitting (features diverging to reduce overlap)
  - Competitive release --> When an absorbing feature is removed, the parent feature reactivates
  - Species richness vs. productivity tradeoff --> Feature diversity vs. reconstruction quality tradeoff

- **Hypothesis**: The Lotka-Volterra competition framework predicts that absorption occurs when the competition coefficient alpha_ij (decoder cosine similarity between features i and j) exceeds a critical threshold relative to their carrying capacities. Specifically, feature j absorbs feature i when alpha_ij * K_j > K_i (the competitive effect of j on i exceeds i's carrying capacity). This yields a computable, pre-training predictor of which feature pairs will exhibit absorption, based solely on decoder geometry.

- **Why not just a metaphor**: The Lotka-Volterra equations can be derived as gradient flow of a Lyapunov function that is structurally similar to the SAE reconstruction loss. The competition coefficients alpha_ij map directly to decoder weight inner products, which are measurable quantities. The critical threshold prediction is falsifiable: one can measure decoder cosine similarities and carrying capacities (maximum activations) for all feature pairs and predict which will exhibit absorption, then validate against the Chanin et al. absorption metric. Character displacement predicts that features in "sympatry" (co-occurring in the same context) will diverge more than features in "allopatry" (occurring in different contexts), which is testable.

- **Novelty estimate**: 7/10 -- The ecological analogy has been briefly mentioned in passing (the literature survey notes "hierarchical topic models face analogous absorption"), but no one has formalized the Lotka-Volterra framework for SAE features or derived the critical competition coefficient threshold.

---

## Phase 3: Self-Critique

### Against Candidate A: Immunodominance Hierarchy Theory

- **Shallow analogy attack**: The analogy is moderately deep. The formal structure (competing populations under resource constraints with active suppression) maps well, but the biological details (MHC presentation, BCR affinity maturation, iTreg induction) have no direct counterpart in SAE training. A domain expert immunologist would agree that the competition dynamics are structurally similar but would note that the biological mechanisms (germinal centers, T follicular helper cells) are far more complex than gradient descent. The "original antigenic sin" component is the strongest part: the prediction that learning order matters is genuinely novel and testable. Verdict: the structural correspondence holds at the population dynamics level but not at the molecular mechanism level.

- **Scale mismatch attack**: Immunodominance operates at the level of clonal populations (10^3-10^6 cells); SAE features are individual dictionary atoms. The timescales are also different: immune responses unfold over days-weeks; SAE training over hours-days of gradient steps. However, the population-level dynamics (competition for limited resources under a fitness objective) are scale-invariant. The key question is whether the temporal learning order effect (OAS) genuinely manifests in SAE training or whether SGD's stochastic nature washes it out. This is an empirical question.

- **Prior transplant check**: I searched for "immunodominance sparse autoencoder," "original antigenic sin machine learning," and "immune hierarchy feature learning." No prior work applies immunodominance theory to SAE feature absorption or to dictionary learning more broadly. The closest work is Bereska et al. (2025) connecting SAE superposition to lossy compression, but this does not invoke immunological analogies.

- **Testability attack**: The key prediction (temporal learning order determines absorption hierarchy) is directly testable: one can track when individual SAE features first become active during training and correlate this with which features are absorbed vs. absorbing. If early-learned specific features absorb later general features (as OAS predicts), the analogy is confirmed. The "vaccination intervention" prediction (periodic training regime changes can break absorption) is also testable via experiments with learning rate restarts or data distribution shifts during training. HOWEVER, distinguishing "this works because of the immunodominance principle" from "this works because of standard optimization dynamics" requires careful control experiments.

- **Verdict**: MODERATE -- The temporal learning order prediction is novel and testable, but the deeper biological mechanisms do not map cleanly, and the prediction could potentially be explained by simpler optimization dynamics without invoking immunological principles.

### Against Candidate B: Rate-Distortion Phase Transition Theory

- **Shallow analogy attack**: This is NOT an analogy -- it is a mathematical identity. SAE training IS a rate-distortion problem. The SAE minimizes reconstruction error (distortion) subject to a sparsity constraint (rate). The information bottleneck framework is the correct mathematical formalism for this problem. The phase transition prediction follows from the known bifurcation structure of information bottleneck solutions (Agmon, 2024). A statistical physicist would agree that the SAE loss landscape should exhibit the same bifurcation structure as the IB Lagrangian. The only question is whether the idealized IB analysis (which assumes optimal solutions) applies to the non-convex optimization landscape of real SAE training.

- **Scale mismatch attack**: Rate-distortion theory applies to arbitrary sources and codebooks, so there is no scale mismatch. The main concern is that IB phase transitions are derived for optimal solutions (global minima of the Lagrangian), while SAE training converges to local minima. However, the piecewise biconvex analysis in arXiv:2512.05534 already addresses this partially by characterizing the structure of local minima. The Ambrogioni et al. (2025) result showing that generative model phase transitions are mean-field (due to self-consistency conditions) suggests that the same should hold for SAEs.

- **Prior transplant check**: I searched for "rate distortion phase transition sparse autoencoder," "information bottleneck feature absorption," and "codebook bifurcation dictionary learning." The closest work is: (1) Ayonrinde et al. (2024), "Interpretability as Compression: Reconsidering SAE Explanations with MDL-SAEs," which uses minimum description length (related to rate-distortion) to train SAEs but does not analyze absorption as a phase transition; (2) Bereska et al. (2025), "Superposition as Lossy Compression," which measures superposition via rate-distortion metrics but does not predict absorption onset; (3) the unified SDL theory paper (arXiv:2512.05534), which characterizes SAE optimization landscape but does not derive phase transition scaling laws for absorption. Nobody has combined these threads to predict absorption onset as a critical phenomenon.

- **Testability attack**: The phase transition prediction is highly testable. One can: (1) train SAEs at many values of the sparsity parameter (lambda or k) and measure absorption rate at each; (2) plot absorption rate vs. sparsity parameter and look for a critical point (sharp onset); (3) fit a power-law scaling near the critical point to extract the critical exponent gamma; (4) compare across different feature hierarchies to test universality. The "diagnostic experiment" would be: if absorption onset follows a scaling law with mean-field exponent gamma = 1/2 (as predicted by the Ambrogioni framework), this confirms the rate-distortion phase transition interpretation. If the onset is gradual (no critical point), the phase transition framework is falsified.

- **Verdict**: STRONG -- This is not an analogy but a mathematical identity. The predictions are quantitative, falsifiable, and novel. The main risk is that non-convex optimization dynamics may blur the sharp phase transitions predicted by the convex IB theory.

### Against Candidate C: Ecological Niche Partitioning

- **Shallow analogy attack**: The analogy has both deep and shallow components. The deep part: Lotka-Volterra dynamics can indeed be derived as gradient flow on a potential function, and this potential function has structural similarity to the SAE loss. The competition coefficients alpha_ij map to decoder cosine similarities, which are measurable. The shallow part: ecological systems have spatial structure, evolutionary timescales, trophic interactions, and environmental stochasticity that have no counterpart in SAE training. An ecologist would agree that the competitive exclusion principle applies abstractly but would note that real ecosystems have far more complexity (mutualism, predation, environmental fluctuations) than SAE training. The "character displacement" parallel with OrtSAE is genuinely insightful but somewhat obvious once stated.

- **Scale mismatch attack**: Ecological competition operates over generations and spatial landscapes; SAE feature competition operates over training steps in a high-dimensional loss landscape. The timescale mapping is straightforward (generations --> training epochs), but the spatial dimension is absent in SAEs. The Lotka-Volterra framework assumes well-mixed populations, which maps well to SAEs (all features compete globally).

- **Prior transplant check**: The literature survey already notes that "hierarchical topic models face analogous absorption of general topics by specific subtopics" and that competitive exclusion has been invoked in NLP multi-label classification contexts. The specific application of Lotka-Volterra equations to SAE features is novel, but the general idea of competition between dictionary atoms in sparse coding is well-established (e.g., in the Locally Competitive Algorithm by Rozell et al., 2008, which explicitly implements lateral inhibition between dictionary atoms). Rozell's work is directly relevant and somewhat anticipates this connection, though it does not address feature absorption specifically.

- **Testability attack**: The critical threshold prediction (alpha_ij * K_j > K_i implies absorption) is directly testable using existing pre-trained SAEs and the Chanin et al. absorption metric. One can compute decoder cosine similarities, measure maximum activations, and predict absorption. However, the prediction may be somewhat trivially true: features with high decoder similarity are already known to be prone to absorption (this is essentially what the absorption metric measures). The novelty would be in the quantitative threshold prediction and in using it prospectively (before measuring absorption) rather than retrospectively.

- **Verdict**: MODERATE -- The framework provides useful vocabulary and a quantitative prediction, but the competitive exclusion analogy is somewhat obvious and partially anticipated by existing work on lateral competition in sparse coding. The deepest contribution would be the quantitative threshold prediction, which is novel but may reduce to a trivially true statement about decoder similarity.

---

## Phase 4: Refinement

### Dropped: Candidate A (Immunodominance)
While the temporal learning order prediction is interesting, the deeper biological mechanisms do not provide new mathematical tools beyond what standard optimization theory already offers. The "original antigenic sin" analog could be a compelling narrative for a paper introduction, but it does not yield concrete new experimental methods or theoretical results. The prediction about temporal learning order is worth testing as a supplementary experiment but does not constitute a full research direction. Dropped as the primary proposal, but retained as a supporting narrative and source of one testable prediction.

### Strengthened: Candidate B (Rate-Distortion Phase Transition) -- FRONT-RUNNER
This candidate is the strongest because:
1. It is not an analogy but a mathematical identity (SAE training IS rate-distortion optimization)
2. It yields quantitative, falsifiable predictions (critical point, scaling exponents)
3. It addresses the most important open gap (Gap 1: no quantitative causal theory of absorption magnitude)
4. It connects to existing theoretical work (unified SDL theory, IB bifurcations) in a novel way
5. It is experimentally feasible with existing pre-trained SAEs (training-free: sweep sparsity parameters and measure absorption)

**Formalized structural correspondence:**

Let x be an LLM activation vector (source), z be the SAE latent code (compressed representation), and x_hat be the SAE reconstruction. The SAE objective is:

L(theta) = E[||x - x_hat||^2] + lambda * ||z||_1

This is equivalent to minimizing distortion D = E[||x - x_hat||^2] subject to a rate constraint R = E[||z||_0] (approximately, since L1 is a convex relaxation of L0). The Lagrange multiplier lambda controls the rate-distortion trade-off.

In the information bottleneck framework, the optimal codebook undergoes bifurcations at critical values of beta = 1/lambda. At each bifurcation, either a new codeword emerges (feature splitting) or two codewords merge (feature absorption). The bifurcation type (pitchfork, saddle-node) determines whether the transition is continuous or discontinuous.

**Prediction:** Absorption rate A(lambda) follows a piecewise scaling law:
- For lambda < lambda_c (subcritical): A = 0 (no absorption)
- For lambda > lambda_c (supercritical): A ~ (lambda - lambda_c)^gamma

where lambda_c is the critical sparsity penalty and gamma is a critical exponent. If the SAE's loss landscape is effectively mean-field (supported by Ambrogioni et al. 2025), then gamma = 1/2.

**Diagnostic experiment:** Train (or evaluate pre-trained) SAEs at a fine grid of sparsity levels. Plot absorption rate vs. sparsity parameter. Fit the scaling law to extract lambda_c and gamma. Compare gamma across different feature hierarchies (first-letter, entity-type, sentiment) to test universality.

### Retained as supporting framework: Candidate C (Ecological Niche Partitioning)
The Lotka-Volterra competition coefficient provides a useful intuitive frame and a pre-training predictor of absorption (decoder cosine similarity vs. maximum activation). Retained as a complementary analysis tool, not the primary theoretical contribution.

### Selected front-runner: Candidate B

---

## Phase 5: Final Proposal

### Title
**Feature Absorption as a Rate-Distortion Phase Transition: Predicting When Sparse Autoencoders Break**

### Source Principle
In information theory, the optimal codebook for lossy compression undergoes phase transitions (bifurcations) at critical values of the rate-distortion trade-off parameter. When the rate constraint is tightened beyond a critical threshold, codewords merge (analogous to features merging/absorbing). These transitions follow universal scaling laws near the critical point, with exponents determined by the symmetry class of the source distribution. The information bottleneck framework (Tishby et al., 1999) formalizes these bifurcations, and recent work (Ambrogioni et al., 2025) shows that such transitions in generative models are always mean-field.

### Structural Correspondence

| Rate-Distortion / Stat. Physics | SAE Feature Absorption |
|---|---|
| Source distribution p(x) | Distribution of LLM activations |
| Codebook {c_1, ..., c_M} | SAE dictionary {d_1, ..., d_M} |
| Rate R = I(X; Z) | Effective sparsity L0 = E[||z||_0] |
| Distortion D = E[d(X, X_hat)] | Reconstruction error MSE |
| Lagrange multiplier beta = 1/lambda | Inverse sparsity penalty |
| Codeword merging (bifurcation) | Feature absorption (child absorbs parent) |
| Critical point beta_c | Critical sparsity threshold lambda_c |
| Order parameter | Absorption rate A |
| Scaling exponent gamma | Universal (predicted gamma = 1/2 for mean-field) |
| Free energy landscape | SAE loss landscape (piecewise biconvex per arXiv:2512.05534) |
| Symmetry breaking | SAE "choosing" to represent parent-child pair with one feature |
| Phase coexistence | Bistability between absorbed and non-absorbed states |

This is not an analogy -- the SAE objective is literally a rate-distortion Lagrangian. The phase transition framework adds the prediction that absorption onset is sharp (not gradual) and follows universal scaling laws.

### Hypothesis
Feature absorption in SAEs is a rate-distortion phase transition. There exists a critical sparsity level lambda_c such that:
1. For lambda < lambda_c, the optimal SAE solution represents parent and child features separately (no absorption)
2. For lambda > lambda_c, the optimal solution merges them (absorption occurs)
3. Near the critical point, the absorption rate scales as A ~ (lambda - lambda_c)^gamma with gamma = 1/2 (mean-field universality)
4. The critical point lambda_c depends on the feature hierarchy's statistics: co-occurrence frequency, frequency ratio between parent and child, and the number of child features per parent
5. Different feature hierarchies (first-letter, entity-type, knowledge taxonomies) share the same critical exponent gamma (universality) but have different critical points lambda_c (non-universal)

### Method

**Theoretical component:**
1. Formulate the SAE objective as an information bottleneck problem with structured source (hierarchical features)
2. Derive the bifurcation conditions analytically for a two-level hierarchy (one parent feature, N child features)
3. Compute the critical sparsity lambda_c as a function of: parent/child frequency ratio, co-occurrence probability, dictionary size, and parent/child decoder similarity
4. Predict the critical exponent gamma from the IB framework (expected: gamma = 1/2)

**Empirical component (training-free, using pre-trained SAEs):**
1. Use Gemma Scope SAEs (JumpReLU) and SAEBench SAEs (7 architectures) on Gemma-2-2B
2. For the first-letter spelling task (Chanin et al. absorption metric):
   - Evaluate absorption rate at each available sparsity level (SAEBench provides 6 sparsity levels per width)
   - Plot A(lambda) and fit the scaling law to extract lambda_c and gamma
3. Extend to entity-type hierarchies (country > city, using Wikidata probes):
   - Train logistic regression probes for "is a city" (parent) and "is [specific city]" (child)
   - Measure absorption rate using adapted Chanin et al. metric
   - Extract lambda_c and gamma for this hierarchy and compare with first-letter results
4. Test universality: compare gamma across hierarchies and architectures
5. Test the Lotka-Volterra predictor (Candidate C supplement): compute decoder cosine similarity matrix, predict which feature pairs will exhibit absorption, validate against measured absorption

### Diagnostic Experiment
The key test that confirms the rate-distortion phase transition interpretation (and not just a smooth increase):
- **Prediction from the theory**: absorption rate vs. sparsity should show a SHARP onset (not gradual). Specifically, A(lambda) should be approximately zero below lambda_c and grow as a power law above it. A smooth, gradual increase would falsify the phase transition interpretation.
- **Procedure**: Plot absorption rate vs. L0 for SAEs spanning a wide range of sparsity levels (SAEBench provides L0 from ~5 to ~200 on Gemma-2-2B layer 12). If a sharp knee is visible, fit (lambda - lambda_c)^gamma near the knee. If the fit yields gamma close to 0.5, this confirms mean-field universality.
- **Null hypothesis**: absorption rate increases linearly with sparsity (no phase transition). This would suggest absorption is a gradual optimization phenomenon, not a critical phenomenon.
- **Control**: compare the scaling behavior across architectures (TopK, JumpReLU, BatchTopK, Matryoshka). If gamma is consistent across architectures but lambda_c differs, this confirms universality (architecture-independent critical exponent with architecture-dependent critical point).

### Experimental Plan

**Phase 1 (Pilot, ~15 min):**
- Load 3-4 pre-trained SAEs from SAEBench at different sparsity levels on Gemma-2-2B layer 12
- Run Chanin et al. absorption metric on first-letter task
- Plot A vs. L0 to check for sharp onset

**Phase 2 (Main experiment, ~45 min):**
- Extend to all available SAEBench SAEs (~30 SAEs across 6 sparsity levels x 3 widths)
- Fit scaling law A ~ (L0_c - L0)^gamma (note: lower L0 = higher effective lambda)
- Extract lambda_c and gamma with confidence intervals via bootstrap

**Phase 3 (Cross-domain validation, ~45 min):**
- Construct entity-type hierarchy probes (country > city, animal > species)
- Adapt absorption metric for these hierarchies
- Measure A vs. L0 for the same SAEs
- Test universality: compare gamma across hierarchies

**Phase 4 (Lotka-Volterra supplement, ~30 min):**
- Compute decoder cosine similarity matrix for top SAE features
- Compute maximum activations (carrying capacity proxy)
- Predict absorption using alpha_ij * K_j > K_i threshold
- Validate predictions against measured absorption

Target models: Gemma-2-2B (primary), GPT-2 Small (secondary for reproducibility). All experiments use pre-trained SAEs (training-free, per project constraints). Total estimated time: ~2.5 hours across 4 phases.

### Risk Assessment

1. **Non-convexity may blur the phase transition**: Real SAE training converges to local minima, which may smear the sharp onset predicted by convex IB theory. Mitigation: use many sparsity levels to improve statistical power; if the transition is blurred, report the "crossover region" width as a measure of non-convexity.

2. **SAEBench sparsity granularity may be too coarse**: 6 sparsity levels may not be enough to resolve a sharp onset. Mitigation: if available, use Gemma Scope SAEs which span a wider range; alternatively, interpolate by training additional SAEs at intermediate sparsity levels (within time budget).

3. **Absorption metric noise**: The Chanin et al. metric has inherent noise (depends on probe quality, false negative threshold). Mitigation: use SAEBench's modified metric which works across all layers; average over multiple letters/features.

4. **Universality may not hold**: Different feature hierarchies may have different critical exponents, breaking the universality prediction. This would actually be an interesting finding (indicating that feature hierarchy structure matters beyond simple parent-child statistics).

5. **The connection may be obvious in retrospect**: Once stated, the connection between SAE sparsity and rate-distortion trade-off is straightforward. The novelty is in the specific predictions (sharp onset, scaling exponent, universality) rather than the general framework. Mitigation: the concrete scaling law predictions are what elevate this from a framing exercise to a testable theory.

### Novelty Claim

The specific cross-disciplinary insight is: feature absorption in SAEs is not a gradual optimization artifact but a PHASE TRANSITION in the information-theoretic sense, with a sharp critical point, universal scaling exponents, and predictable onset as a function of sparsity parameters. This reframes absorption from "an annoying failure mode to be patched" to "a fundamental consequence of rate-distortion optimization that can be predicted and controlled."

Evidence this has not been applied before:
- The unified SDL theory paper (arXiv:2512.05534) characterizes the optimization landscape but does not derive phase transition scaling laws
- Bereska et al. (2025) measure superposition via rate-distortion metrics but do not predict absorption onset
- Ayonrinde et al. (2024) use MDL (related to rate-distortion) for SAE training but do not analyze absorption
- Ambrogioni et al. (2025) derive phase transitions in diffusion models but not in SAEs/dictionary learning
- Chanin et al. (2024) measure absorption empirically but do not derive theoretical predictions for its onset
- No paper in the SAE or mechanistic interpretability literature models absorption as a critical phenomenon with scaling laws

The combination of rate-distortion theory + statistical physics phase transitions + SAE feature absorption is novel and yields concrete, falsifiable predictions that existing frameworks cannot make.
