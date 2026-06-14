# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Lian & Burkitt (2025). "Relating sparse and predictive coding to divisive normalization." PLoS Computational Biology.** -- Unifies three computational neuroscience principles: sparse coding, predictive coding, and divisive normalization within a single two-layer neural model. Demonstrates that lateral inhibition in sparse coding produces competitive dynamics where features suppress each other. The homeostatic function shapes nonlinearity of neural responses, replicating divisive normalization. Key mechanism: winner-take-all (extreme sparsity) naturally produces competitive exclusion among neural representations, where one representation absorbs the role of another. This is the neuroscience analog of feature absorption -- when the sparsity constraint is too aggressive, lateral inhibition operates across hierarchy levels instead of within them.

2. **Rozanski et al. (2025 revival). "Structured Sparse Coding via Lateral Inhibition."** -- Shows that lateral inhibition in sparse coding produces structured competition where features at the same hierarchical level suppress each other. The locally competitive algorithm (LCA) solves sparse coding by minimizing MSE + coefficient cost, with terms for input projection, membrane potential leak, and lateral inhibition from neurons whose membrane potential surpassed a threshold. Provides the mechanistic bridge: absorption is what happens when lateral inhibition (sparsity penalty) operates across hierarchy levels instead of within them.

3. **Kriegeskorte & Wei. "Neural tuning and representational geometry."** -- Reviews how the Fisher information of neural population codes determines decodable information through the geometry of representational patterns in multivariate response space. Tuning analyses use Fisher information to characterize sensitivity of neural responses to small changes. Provides a formal framework connecting population coding geometry to feature discriminability -- directly relevant to understanding why some SAE features "absorb" others from a representational geometry perspective.

4. **NeurIPS 2024 Poster. "Unconditional stability of a recurrent neural circuit implementing divisive normalization."** -- Addresses stability of recurrent circuits implementing divisive normalization (ORGaNICs). Traditional cortical circuit models are difficult to train due to expansive nonlinearities. Establishes that biologically plausible recurrent circuits can dynamically achieve divisive normalization and simulate neurophysiological phenomena -- analogous to how SAE training dynamics settle into absorption-producing equilibria.

5. **Salvatori et al. (2025). "A Survey on Neuro-mimetic Deep Learning via Predictive Coding."** -- Comprehensive survey connecting predictive coding to variational inference and the evidence lower bound (ELBO). Establishes that iterative updating schemes in predictive coding minimize variational free energy. Relevant because SAE training can be viewed as a form of variational inference where absorption represents a local minimum of the free energy landscape.

#### Immunology / Adaptive Immunity

6. **Francis (1960) and modern reviews (Yewdell & Santos, 2021). "Original Antigenic Sin" (immune imprinting).** -- The immune system preferentially uses immunological memory based on a first infection when encountering a second slightly different pathogen. First antigen exposure creates dominant memory B/T cells that suppress de novo responses to novel variants. Immunodominance hierarchies are kinetically driven and can be reversed by prior exposure history. The structural correspondence to feature absorption is deep: a "general" immune response (parent feature) is silently suppressed by a "specific" memory response (child feature) that absorbed its representational role, leaving the immune system unable to properly respond to novel stimuli -- exactly as absorbed SAE features leave systematic gaps in recall.

7. **Welsh et al. (2004). "T cell immunodominance and maintenance of memory regulated by unexpectedly cross-reactive pathogens." Nature Immunology.** -- T cell cross-reactivity between heterologous viruses influences the immunodominance of CD8+ T cells: cross-reactive epitopes dominate acute responses and are selectively maintained in memory while non-cross-reactive epitopes are lost. This is feature absorption in the immune repertoire: the cross-reactive (child) feature absorbs the activation budget of the specific (parent) feature.

8. **Bipartite network models and the GNK theory.** -- The GNK model considers the full diversity of 10^8 B/T cells and can predict immunodominance, cross-reactivity, and original antigenic sin. Traditional ODE models can be fit to but cannot predict immunodominance because they do not consider repertoire diversity. The key insight: predicting which features are absorbed requires modeling the full feature space competition, not just pairwise interactions.

#### Physics / Statistical Mechanics / Information Theory

9. **AAAI 2025. "A Renormalization Group Framework for Scale-Invariant Feature Learning in Deep Neural Networks."** -- Proposes that layer-wise transformations in deep networks can be viewed as RG transformations, each implementing a coarse-graining operation extracting increasingly abstract features. Scale-aware activation functions and RG flow equations for network parameters lead to fixed points corresponding to scale-invariant representations. Feature absorption maps onto "relevant operator absorption" in the RG sense: a fine-scale (child) feature captures all information of a coarse-scale (parent) feature, making the parent irrelevant at that resolution.

10. **Ganguli (2012). "Statistical Mechanics of Compressed Sensing."** -- Applies statistical physics techniques (replica method, belief propagation) to compressed sensing, revealing phase transitions in sparse recovery as measurement count varies. The sparse recovery phase transition is structurally identical to the absorption phase transition: below a critical ratio of dictionary capacity to feature count, exact recovery fails and features merge (are absorbed). Provides formal tools (free energy landscape, order parameters) for analyzing when absorption becomes thermodynamically inevitable.

11. **Krzakala, Mezard et al. (2012). "Statistical-Physics-Based Reconstruction in Compressed Sensing." Phys. Rev. X.** -- Designs reconstruction algorithms approaching the theoretical limit using probabilistic approach + message passing + measurement matrix design inspired by crystal nucleation theory. The connection to SAE absorption: message-passing algorithms (belief propagation on factor graphs) can detect and potentially prevent absorption by explicitly modeling feature dependencies.

12. **arXiv:2501.11905 (2025). "Phase Transitions in Phase-Only Compressed Sensing."** -- Derives asymptotically precise formulas for phase transition locations in compressed sensing. Disproves earlier conjecture about measurement requirements. The phase transition framework directly predicts when absorption becomes unavoidable as a function of SAE width and feature density.

13. **Mallat (1989). "A Theory for Multiresolution Signal Decomposition: The Wavelet Representation."** -- Foundational: shows the difference of information between signal approximations at resolutions 2^(j+1) and 2^j can be extracted by decomposing on a wavelet orthonormal basis. Computed with pyramidal algorithm based on convolutions with quadrature mirror filters. The multi-resolution decomposition framework is directly applicable to analyzing SAE decoder geometry at multiple scales -- absorption is the dictionary learning analog of a signal component disappearing at one wavelet scale but being encoded within another component at a finer scale.

14. **Hammond, Vandergheynst, Gribonval. "Wavelets on Graphs via Spectral Graph Theory."** -- Defines spectral graph wavelets using the graph Laplacian eigendecomposition or Chebyshev polynomial approximation. Provides multi-resolution decomposition on arbitrary graph structures. If we construct a graph from SAE decoder cosine similarities, spectral graph wavelets at different scales can identify absorption pairs as features that merge across resolution scales.

#### Ecology / Evolutionary Biology

15. **Gause (1934), Hardin (1960). "The Competitive Exclusion Principle."** -- Two species competing for the same limited resource cannot coexist at constant population sizes. When one species has even the slightest advantage, it dominates long-term, leading to extinction of the weaker competitor or niche shift. The principle has a precise mathematical formulation in Lotka-Volterra: coexistence requires intraspecific competition to exceed interspecific competition (niche overlap rho < 1). Feature absorption IS competitive exclusion in representation space: the child feature excludes the parent from activation because they compete for the same sparse activation slots.

16. **Chesson (2000). "Mechanisms of maintenance of species diversity." Annual Review of Ecology.** -- Establishes the modern coexistence framework: species coexistence depends on the balance of niche differences (stabilizing) and fitness differences (equalizing). The ratio rho between inter- vs. intra-specific competition measures species similarity. When rho approaches 1, competitive exclusion is inevitable. Provides the formal machinery for predicting when absorption occurs as a function of "niche overlap" between features.

17. **Mackay & Anholt (2024). "Pleiotropy, epistasis and the genetic architecture of quantitative traits." Nature Reviews Genetics.** -- Pleiotropy (one gene affecting multiple traits) constrains evolution by introducing a "cost of complexity." Gene expression networks show modular structure where large modules are positively correlated with one trait and negatively with another. The structural parallel: SAE features are pleiotropic (one latent responds to multiple inputs), and absorption is the dictionary learning analog of pleiotropic constraint -- when a feature's pleiotropic burden exceeds the capacity budget, it is absorbed by a more specialized feature.

### Cross-Disciplinary Gaps

The following transplants have NOT been explored in the SAE/mechanistic interpretability literature:

1. **Immunological imprinting / Original Antigenic Sin --> Feature absorption**: No prior work maps the immune system's hierarchical recognition failure to SAE feature dynamics. The structural correspondence (immunodominance hierarchy = feature hierarchy, cross-reactivity = feature co-occurrence, repertoire competition = sparsity constraint) is deep and unmapped.

2. **Ecological competitive exclusion --> Quantitative absorption prediction**: The Innovator perspective (existing in the workspace) proposes this analogy but no published work formalizes it. The Lotka-Volterra coexistence threshold directly maps to an absorption onset threshold.

3. **Statistical physics phase transitions in compressed sensing --> SAE absorption phase boundaries**: While Ganguli's work on compressed sensing phase transitions is foundational, and the unified SDL theory (arXiv:2512.05534) provides the optimization landscape view, nobody has derived the SAE-specific phase diagram using replica methods or cavity methods from statistical physics.

4. **Spectral graph wavelets on decoder geometry --> Unsupervised absorption detection**: The Innovator perspective proposes SGWA but it has not been implemented or validated. The GWNN (ICLR 2019) framework provides the computational machinery, but applying it to SAE decoder weight graphs is entirely novel.

5. **Predictive coding / divisive normalization --> Absorption as a normalization failure**: Lian & Burkitt's unification of sparse coding with divisive normalization implies that absorption occurs when divisive normalization operates across hierarchy levels. This reframing has not been applied to SAE analysis.

---

## Phase 2: Initial Candidates

### Candidate A: Immunological Imprinting Theory of Feature Absorption (from Immunology)

- **Source principle**: Original Antigenic Sin (OAS) / immune imprinting -- the immune system's first exposure to a pathogen creates a dominant memory response that suppresses de novo responses to related but distinct antigens encountered later. Immunodominance hierarchies are kinetically driven: the fastest-responding clone wins the competition for antigen, starving slower clones of stimulation.

- **Structural correspondence**: The formal mapping between immune dynamics and SAE feature dynamics is:

| Immune System | SAE Feature Dynamics |
|---|---|
| Antigen (input stimulus) | Input activation vector x |
| T/B cell clone (specific receptor) | SAE latent feature f_i with decoder direction w_i |
| Clonal expansion (proliferation after antigen recognition) | Feature activation a_i = ReLU(W_enc * x + b) |
| Immunodominance hierarchy | Feature hierarchy (parent > child) |
| Cross-reactivity (one clone recognizes multiple antigens) | Feature co-occurrence / cosine similarity of decoders |
| Repertoire size | SAE dictionary width N |
| Activation budget (limited APC surface, IL-2 supply) | Sparsity constraint (L0 / L1 penalty) |
| Original antigenic sin (first clone suppresses new responses) | Feature absorption (child feature suppresses parent activation) |
| Immune escape (virus mutates to evade dominant clone) | Absorption-induced false negatives (tokens where absorbed feature should fire but doesn't) |
| Immunodominance reversal (prior priming of subdominant clone) | Potential mitigation: pre-conditioning or anchoring parent features |

The mathematical structure is: In the immune system, competition for a limited activation budget (cytokine supply, APC surface area) among cross-reactive clones of varying affinity produces immunodominance, where the highest-affinity clone for the presented antigen monopolizes the response. In SAEs, competition for a limited sparsity budget (L0 constraint) among co-occurring features of varying specificity produces absorption, where the most specific feature for a given input monopolizes the activation.

- **Hypothesis**: The kinetic immunodominance model predicts that absorption severity is determined by the ratio of (a) the "affinity advantage" of the child feature over the parent (measured as the relative reconstruction fidelity gain from using the child vs. parent decoder direction) to (b) the "repertoire bottleneck" (measured as L0/N, the fraction of features that can be active per token). Specifically, absorption occurs when the child's affinity advantage exceeds a critical threshold that scales inversely with the repertoire bottleneck. This predicts that increasing L0 (relaxing the budget) should reduce absorption monotonically, while increasing N (expanding the repertoire) should reduce absorption only up to a saturation point -- matching the empirical observation that wider SAEs can show MORE absorption.

- **Why not just a metaphor**: The correspondence is structural, not verbal. Both systems solve the same optimization problem: allocate a limited budget of active responses among a repertoire of detectors with overlapping sensitivity profiles under a penalty for simultaneous activation. The Lotka-Volterra dynamics governing immunodominance (clonal competition for limited cytokine/APC resources) have the same mathematical form as the gradient dynamics governing feature activation competition under L1 sparsity. The critical difference that makes this more than a metaphor: the immune system's solution to OAS (broadly neutralizing antibodies, vaccine design strategies) suggests specific mitigation strategies for SAE absorption that have NOT been tried (see Phase 5).

- **Novelty estimate**: 9/10 -- No prior work connects immunological imprinting to SAE feature absorption. The closest related work is on artificial immune systems (early 2000s), which used immune-inspired algorithms for optimization/clustering but never addressed the specific pathology of OAS as an analog for dictionary learning failure modes.

### Candidate B: Statistical Physics Phase Diagram of Absorption (from Statistical Mechanics)

- **Source principle**: Phase transitions in sparse recovery from compressed sensing / statistical physics. In the statistical mechanics of compressed sensing, the replica method and cavity method predict sharp phase transitions in the recoverability of sparse signals as a function of the measurement-to-signal ratio (analogous to SAE width / number of true features) and sparsity level. Below a critical ratio, exact sparse recovery fails and signals "merge" -- a phase transition structurally identical to the onset of absorption.

- **Structural correspondence**: 

| Statistical Physics (Compressed Sensing) | SAE Feature Absorption |
|---|---|
| Signal x in R^n with sparsity k | True feature activation pattern (k features active per token) |
| Measurement matrix A (m x n) | SAE encoder W_enc |
| Measurement vector y = Ax | LLM activation vector x |
| Reconstruction x_hat = argmin ||x||_1 s.t. Ax = y | SAE encoding: a = argmin ||a||_1 s.t. ||x - W_dec * a||^2 small |
| Phase transition at m/n = f(k/n) | Absorption onset at N/d = g(L0/N) |
| Replica-symmetric free energy | SAE loss landscape |
| Order parameter (overlap with true signal) | Feature recovery fidelity |
| Ferromagnetic phase (exact recovery) | No absorption: all features correctly separated |
| Spin-glass phase (many metastable states) | Absorption: features trapped in local minima where parent is absorbed |

The statistical mechanics framework predicts that SAE absorption exhibits a sharp phase transition as a function of two control parameters: (1) the overcomplete ratio N/d (dictionary width / model dimension) and (2) the sparsity ratio L0/N (active features / total features). Below a critical surface in this 2D parameter space, absorption is thermodynamically inevitable -- it is the equilibrium state of the optimization, not an artifact of training.

- **Hypothesis**: There exists a critical absorption boundary in the (N/d, L0/N) plane, analogous to the Donoho-Tanner phase transition in compressed sensing, below which absorption rate exceeds a threshold (say >10%) and above which it is negligible (<5%). The boundary shape is predicted by the replica method to follow N/d > C * (L0/N)^(-alpha) for constants C, alpha that depend on the feature co-occurrence structure. The empirically observed L0 ~ 7-14 phase transition from iteration 5 is a cross-section of this boundary.

- **Why not just a metaphor**: The correspondence is mathematical, not analogical. SAE training literally IS a form of sparse dictionary learning, and compressed sensing phase transitions are proven results for sparse recovery problems. The only question is whether the specific geometric structure of LLM activations (which differ from i.i.d. Gaussian measurements) shifts the phase boundary or changes the transition order (sharp vs. gradual). This is a quantitative question with a testable answer.

- **Novelty estimate**: 7/10 -- The connection between compressed sensing theory and dictionary learning is well-known. The specific application to SAE absorption phase diagrams is new but incremental relative to the unified SDL theory paper (arXiv:2512.05534) which already provides an optimization landscape analysis. The statistical physics tools (replica method) would add quantitative predictions that the SDL theory paper lacks.

### Candidate C: Divisive Normalization Failure Model of Absorption (from Computational Neuroscience)

- **Source principle**: Divisive normalization is a canonical neural computation where each neuron's response is divided by the summed activity of a pool of neighboring neurons. Lian & Burkitt (2025) proved that sparse coding with lateral inhibition is equivalent to divisive normalization in the single-neuron case. When divisive normalization operates within a hierarchical level (siblings competing), it produces healthy feature competition and clean separation. When it operates ACROSS levels (parent competing with child), it produces pathological suppression -- the parent's response is divided away by the child's strong activation.

- **Structural correspondence**:

| Divisive Normalization (Neuroscience) | SAE Feature Absorption |
|---|---|
| Neuron's raw response r_i | SAE pre-activation z_i = W_enc[i] * x + b_i |
| Normalization pool sum: sigma + sum_j w_ij * r_j | Effective competition: sum of activations of co-occurring features |
| Normalized response: r_i / (sigma + sum_j w_ij * r_j) | Post-sparsity activation: feature survives only if pre-activation exceeds threshold set by competitors |
| Within-level inhibition (cross-orientation inhibition) | Healthy feature competition (siblings split input space) |
| Cross-level inhibition (fine features suppress coarse) | Feature absorption (child suppresses parent) |
| sigma (semi-saturation constant) | Sparsity threshold / bias term |
| Normalization pool membership | Feature co-occurrence graph |

The key insight: absorption occurs when the normalization pool is "too global" -- when the suppressive competition includes features at different hierarchical levels. In healthy sparse coding, competition should be within-level (siblings compete, ensuring only the best-matching feature fires). Absorption occurs when cross-level competition exists (a child feature's activation enters the normalization pool of its parent, suppressing it).

- **Hypothesis**: Absorption severity is predicted by the fraction of a feature's normalization pool that consists of hierarchically related (parent-child) features rather than sibling features. This can be estimated from the SAE decoder cosine similarity structure: features whose decoder directions are aligned along a hierarchical axis (high conditional cosine similarity, low global cosine similarity) are in a cross-level normalization relationship prone to absorption. Features whose decoder directions are nearly orthogonal are in a within-level competitive relationship that produces healthy feature splitting.

- **Why not just a metaphor**: Lian & Burkitt (2025) proved the mathematical equivalence of sparse coding and divisive normalization. SAE encoding IS sparse coding with lateral inhibition (the L1/TopK penalty creates implicit lateral inhibition). Therefore, the divisive normalization framework applies directly and formally, not by analogy. The specific prediction (cross-level normalization pool membership determines absorption) is testable by measuring the conditional cosine similarity structure of decoder weights.

- **Novelty estimate**: 8/10 -- No prior work frames SAE absorption as a divisive normalization failure. Lian & Burkitt (2025) establish the sparse coding / divisive normalization equivalence but do not apply it to SAEs or absorption. The conditional cosine similarity analysis (from the Geometry of Concepts paper, Michaud et al. 2025) provides the empirical tools but does not connect to divisive normalization theory.

---

## Phase 3: Self-Critique

### Against Candidate A (Immunological Imprinting Theory)

- **Shallow analogy attack**: Is this really structural or just vocabulary mapping? The critical test: would an immunologist agree that the SAE sparsity constraint plays the same role as the limited cytokine/APC budget? Yes -- both create a zero-sum competition where one detector's activation directly reduces the resources available to others. The Lotka-Volterra dynamics are the same mathematical object in both cases. However, an immunologist might object that real immunodominance involves additional feedback loops (regulatory T cells, affinity maturation) that have no SAE analog. **Verdict**: The core competition dynamics are structural; the regulatory feedback loops are specific to immunology and do not transfer.

- **Scale mismatch attack**: The immune system has ~10^8 unique clones competing; SAEs have 10^4 - 10^6 features. The immune system operates over weeks (clonal expansion timescale); SAE training operates over thousands of gradient steps. These scale differences do not invalidate the mathematical correspondence (Lotka-Volterra equations are scale-free), but they may affect the sharpness of the competitive exclusion threshold. **Verdict**: Not a fundamental problem.

- **Prior transplant check**: Searched "original antigenic sin machine learning" and "immunodominance neural network feature." Found: (1) Artificial Immune Systems (AIS) literature from early 2000s uses immune-inspired algorithms for optimization but never addresses OAS or absorption specifically. (2) The catastrophic forgetting / continual learning literature draws a loose analogy to immune memory stability-plasticity tradeoffs but does not formalize it for dictionary learning. (3) No paper maps immunological imprinting to SAE feature absorption. **No direct competition found.**

- **Testability attack**: The theory makes specific quantitative predictions: (a) absorption onset scales with L0/N (sparsity budget / repertoire size), (b) increasing L0 reduces absorption monotonically, (c) increasing N reduces absorption only to a saturation point, (d) the "affinity advantage" (relative reconstruction fidelity) of child over parent predicts which features get absorbed. All four predictions are testable with existing pre-trained SAEs and the Chanin et al. metric. The immune-inspired mitigation (repertoire diversification via multi-exposure / multi-scale training) maps to specific SAE training modifications (curriculum learning, progressive dictionary growth). **Verdict**: Testable.

- **Verdict**: **STRONG** -- The structural correspondence is deep and formal, the predictions are quantitative and testable, and no prior work has explored this mapping. The main risk is that the analogy may be correct but adds no predictive power beyond what simpler models (e.g., the ecological model from the Innovator perspective) already provide.

### Against Candidate B (Statistical Physics Phase Diagram)

- **Shallow analogy attack**: This is not an analogy at all -- SAE training IS a sparse recovery problem, and compressed sensing phase transitions are proven mathematical results for sparse recovery. The only question is whether the specific distributional assumptions (i.i.d. Gaussian measurements) required by the replica method hold for LLM activations. They almost certainly do not hold exactly, but the phase transition behavior is known to be robust to distributional assumptions (the Donoho-Tanner transition holds for a wide class of measurement matrices). **Verdict**: Mathematical correspondence, not analogy.

- **Scale mismatch attack**: The compressed sensing literature primarily studies the underdetermined regime (m < n, more features than measurements), while SAEs operate in the overcomplete regime (N > d, more dictionary elements than dimensions). However, the relevant ratio is different: in SAEs, the effective "measurement" count is the number of data points times the sparsity level, not the ambient dimension. **Verdict**: Requires careful reformulation but not fundamentally problematic.

- **Prior transplant check**: Searched "phase transition sparse autoencoder" and "statistical mechanics dictionary learning absorption." Found: (1) The unified SDL theory paper (arXiv:2512.05534) provides an optimization landscape analysis that touches on absorption but does not use statistical physics tools (replica method, cavity method). (2) Ganguli (2012) analyzes compressed sensing via statistical mechanics but does not address overcomplete dictionaries or SAEs. (3) No paper derives the SAE-specific absorption phase diagram using replica methods. **Partially explored territory** -- the optimization landscape is mapped but not via statistical physics tools.

- **Testability attack**: Deriving the phase diagram requires either (a) applying the replica method to the SAE loss function, which is technically demanding and may require simplifying assumptions that limit practical relevance, or (b) numerically mapping the phase boundary by sweeping (N/d, L0/N) on synthetic data, which is feasible but may not yield a clean analytical result. **Verdict**: The analytical approach is high-risk, high-reward; the numerical approach is feasible but less novel.

- **Verdict**: **MODERATE** -- Mathematically rigorous but (a) the analytical derivation may exceed project scope, (b) the numerical phase boundary mapping is less novel than it sounds (the L0-width interaction from iteration 5 is already a cross-section of this), and (c) the unified SDL theory paper occupies adjacent territory.

### Against Candidate C (Divisive Normalization Failure)

- **Shallow analogy attack**: Lian & Burkitt (2025) prove mathematical equivalence between sparse coding and divisive normalization. SAE encoding is sparse coding. Therefore, this is not an analogy -- it is a direct application. The question is whether the equivalence, which was proven for a simple two-layer model, extends to the specific architecture of modern SAEs (overcomplete dictionaries, ReLU/TopK/JumpReLU activations, tied decoders). **Verdict**: The equivalence holds in principle but the specific form of divisive normalization may differ from the classical form.

- **Scale mismatch attack**: Divisive normalization in neuroscience typically involves normalization pools of 10-100 neurons. SAE features compete globally (all N features compete under the sparsity constraint). This scale difference may mean that the "normalization pool" concept becomes vacuous when all features are in the same pool. **Mitigation**: Define the normalization pool locally (only features with cosine similarity above a threshold). **Verdict**: Addressable.

- **Prior transplant check**: Searched "divisive normalization sparse autoencoder" and "lateral inhibition feature absorption." Found: (1) The Innovator perspective mentions Rozanski et al. connecting competitive dynamics to feature absorption. (2) No paper explicitly frames SAE absorption as a divisive normalization pathology. (3) The LCA (locally competitive algorithm) literature is closely related but focuses on computational efficiency of sparse coding, not failure mode analysis. **No direct competition, but the theoretical infrastructure is close.**

- **Testability attack**: The theory predicts that absorption severity correlates with the fraction of a feature's cosine-similarity neighbors that are hierarchically related (parent-child) vs. same-level (siblings). This is testable: for each absorbed feature identified by the Chanin et al. metric, measure the conditional cosine similarity profile of its decoder weight neighborhood. If the divisive normalization theory is correct, absorbed features will have a higher proportion of cross-level neighbors. **Verdict**: Testable.

- **Verdict**: **STRONG** -- The theoretical foundation is rigorous (proven equivalence), the prediction is specific and testable, and the reframing offers practical mitigation strategies (restrict normalization pools to within-level competitors). The main weakness is that the equivalence is proven only for simple models and may not perfectly capture SAE dynamics.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Statistical Physics Phase Diagram)** is demoted from primary to supporting framework. Rationale: (a) The analytical derivation via replica methods exceeds the project's scope given the training-free constraint. (b) The numerical phase boundary mapping is partially captured by the iteration 5 L0-width interaction analysis. (c) The unified SDL theory paper (arXiv:2512.05534) already provides the optimization landscape view. However, the phase transition concept from statistical physics is retained as the interpretive framework for Candidate A's absorption threshold prediction.

### Strengthened Ideas

**Candidate A (Immunological Imprinting Theory) strengthened**:

1. Formalized the key quantitative prediction using the GNK-style repertoire model:
   - Define "clonal affinity" of feature i for input x as: a_i(x) = w_enc_i . x + b_i (pre-activation)
   - Define "activation budget" as the effective L0: B = sum of binary indicators of feature activation
   - The competition for activation budget is a discrete analog of the competition for cytokine supply
   - The immunodominance criterion becomes: feature i absorbs feature j if and only if (a) on inputs where both i and j should fire, i's affinity exceeds j's by more than the "exclusion threshold" delta (b) the exclusion threshold delta = f(B, N) decreases as budget B increases relative to repertoire N
   - This predicts: absorption rate = Pr[affinity_child - affinity_parent > delta(L0, N)] integrated over the input distribution

2. Derived the specific mitigation strategy inspired by immunological solutions:
   - **Multi-valent vaccination** (exposing the immune system to multiple variants simultaneously) maps to **multi-scale training** (training SAE on activations at multiple layers/contexts simultaneously, as in Matryoshka SAEs)
   - **Adjuvant boosting** (enhancing subdominant responses by supplementing the activation signal) maps to **parent feature anchoring** (adding a regularization term that prevents parent features from being fully suppressed)
   - **Broadly neutralizing antibodies** (engineering antibodies that recognize conserved epitopes across variants) maps to **hierarchically constrained decoders** (enforcing that parent decoder directions are preserved as subspace components of child decoder directions)
   - These mitigation strategies are independently motivated by the immune analogy but converge with existing SAE solutions (Matryoshka, feature anchoring, OrtSAE), providing a unified theoretical explanation for why they work.

3. Added diagnostic experiment: The "immune escape test" -- if absorption is truly analogous to immunodominance, then artificially suppressing the child feature (by zeroing its activation) should cause the parent feature to "recover" and fire on inputs where it was previously absorbed. This is testable via activation patching: zero the absorbed child feature and measure whether the parent feature's activation increases on false-negative tokens. If absorption follows immunodominance dynamics, the recovery should be proportional to the "affinity advantage" that caused the absorption.

**Candidate C (Divisive Normalization Failure) strengthened**:

1. Formalized the "normalization pool" partition:
   - For each feature i, define its normalization pool N(i) as the set of features with cosine similarity > tau
   - Partition N(i) into same-level N_s(i) (siblings) and cross-level N_c(i) (parent-child)
   - Same-level membership: features whose decoder directions are nearly orthogonal to the hierarchical axis but similar to feature i
   - Cross-level membership: features whose decoder directions lie along the hierarchical axis (high conditional cosine similarity)
   - Define the cross-level normalization ratio: CLR(i) = |N_c(i)| / |N(i)|
   - Prediction: absorption rate for feature i is monotonically increasing in CLR(i)

2. Clarified connection to existing proposal's ITAC metric:
   - The ITAC (Information-Theoretic Absorption Coefficient) from the existing proposal measures Var(residual projection | child active, parent inactive) / Var(residual projection | neither active)
   - Under the divisive normalization framework, ITAC >> 1 occurs precisely when the parent feature's decoder direction is in the child's normalization pool -- the child's activation "divides away" the parent's contribution, leaving it in the residual
   - This provides a theoretical interpretation of ITAC: it measures the degree of cross-level divisive normalization

### Selected Front-Runner

**Candidate A (Immunological Imprinting Theory)** is selected as the primary cross-disciplinary contribution, with **Candidate C (Divisive Normalization Failure)** integrated as the mechanistic explanation at the neural computation level.

**Rationale for this combination**: The immunological theory provides the most novel and counterintuitive predictions (absorption as immunodominance, mitigation via multi-valent training, immune escape test diagnostic). The divisive normalization framework provides the mechanistic explanation at the algorithmic level (why specific feature pairs are absorbed). Together they form a two-level theory: immunology provides the population-level dynamics (which features win the competition), while divisive normalization provides the single-feature mechanistic explanation (how the winning feature suppresses the loser).

This is stronger than the Innovator perspective's ecological model because:
1. The immune system has a specific pathology (OAS) that is a closer structural match to absorption than generic competitive exclusion
2. The immunological model naturally handles the temporal dynamics of SAE training (clonal expansion = feature refinement over training steps)
3. The mitigation strategies inspired by immunology (multi-valent vaccination, adjuvant boosting) map to specific, novel SAE training modifications
4. The "immune escape test" diagnostic provides a unique experimental signature that distinguishes the immune theory from other explanations

However, the ecological competitive exclusion framework from the Innovator perspective remains valuable and complementary -- it provides the equilibrium analysis, while the immunological framework provides the kinetic (dynamic) analysis.

---

## Phase 5: Final Proposal

### Title

Immunological Imprinting in Dictionary Learning: How Repertoire Competition and Divisive Normalization Explain Feature Absorption in Sparse Autoencoders

### Source Principles

**Primary**: Original Antigenic Sin / Immune Imprinting (immunology) -- the immune system's first exposure to a pathogen creates dominant memory cells that suppress de novo responses to related variants. This is driven by kinetic competition for limited activation resources (cytokine supply, APC surface area) among cross-reactive clones of varying affinity.

**Secondary**: Divisive Normalization (computational neuroscience) -- a canonical neural computation where each neuron's response is divided by the summed activity of its normalization pool. When the normalization pool includes features at different hierarchical levels, it produces pathological cross-level suppression.

### Structural Correspondence

The formal mapping has three levels:

**Level 1: Population dynamics (Immunology --> SAE training)**
- Clonal competition for cytokine/APC resources = Feature competition for L0 budget
- Immunodominance hierarchy = Feature hierarchy (absorption ordering)
- Original antigenic sin = Feature absorption
- Broadly neutralizing antibody design = Hierarchically constrained decoder design
- Multi-valent vaccine = Multi-scale / Matryoshka training

**Level 2: Single-unit mechanism (Divisive normalization --> SAE encoding)**
- Divisive normalization pool = Co-occurring features in the decoder cosine similarity neighborhood
- Within-level normalization (healthy competition) = Sibling features competing (feature splitting)
- Cross-level normalization (pathological suppression) = Parent-child features competing (absorption)
- Semi-saturation constant sigma = Sparsity threshold / bias

**Level 3: Information theory (Phase transition --> Absorption boundary)**
- Sparse recovery phase transition = Absorption onset boundary in (N/d, L0/N) space
- Free energy landscape = SAE loss landscape
- Replica-symmetric order parameter = Feature recovery fidelity

### Hypothesis

Feature absorption in SAEs follows the same kinetic competition dynamics as immunodominance in the adaptive immune system. Specifically:

1. **Absorption onset threshold**: Absorption occurs for a parent-child feature pair when the child's "affinity advantage" (relative reconstruction fidelity gain) exceeds a critical threshold that scales as delta_crit ~ L0 / N (the per-feature activation budget). This is the SAE analog of the immunodominance threshold.

2. **Absorption severity scaling**: The fraction of absorbed features scales as: absorption_rate ~ Phi((mean_affinity_advantage - delta_crit) / sigma_affinity), where Phi is the normal CDF, predicting a smooth phase transition (not sharp) because the affinity advantages are distributed, not fixed.

3. **Recovery under intervention**: Suppressing (zeroing) an absorbing child feature will cause the absorbed parent feature to "recover" with activation magnitude proportional to the affinity advantage that caused absorption. This is the "immune escape" signature.

4. **Normalization pool predicts absorption partners**: The probability that feature j is absorbed by feature i is monotonically increasing in the cross-level normalization ratio CLR(i,j) = conditional_cosine_similarity(i,j) / global_cosine_similarity(i,j).

### Method

**Stage 1: Theoretical Framework** (no compute needed)
- Formalize the immunological imprinting model as a system of Lotka-Volterra competition equations with SAE-specific parameters
- Derive the absorption onset threshold delta_crit as a function of L0, N, and feature co-occurrence statistics
- Derive the divisive normalization pool partition criterion using decoder cosine similarity
- Show mathematical equivalence between the immunodominance criterion and the Chanin et al. "sparsity gain" argument, demonstrating that the immune framework subsumes the existing explanation

**Stage 2: Immune Escape Diagnostic Experiment** (training-free, ~30 min per SAE)
- Load pre-trained Gemma Scope SAEs (layer 12, widths 16k and 65k) on Gemma 2 2B
- Identify absorbed features using the Chanin et al. probe-based metric on the first-letter task
- For each absorbed parent-child pair:
  - Zero the child feature's activation via activation patching
  - Measure whether the parent feature's activation increases on the false-negative tokens
  - Record the recovery magnitude and correlate with the predicted "affinity advantage"
- This is the key diagnostic: if the immunological theory is correct, parent features must recover when the absorbing child is suppressed, with recovery magnitude proportional to affinity advantage
- Target: recovery observed in >70% of absorbed pairs; correlation between recovery magnitude and affinity advantage r > 0.5

**Stage 3: Cross-Level Normalization Analysis** (training-free, ~20 min per SAE)
- Compute decoder cosine similarity matrix for each SAE
- For each feature, compute the normalization pool partition (same-level vs. cross-level neighbors)
- Calculate the cross-level normalization ratio CLR for all features
- Correlate CLR with the probe-based absorption rate
- Target: Spearman rho > 0.4 between CLR and absorption rate

**Stage 4: Absorption Threshold Prediction** (training-free, ~1 hour total)
- Across all available Gemma Scope SAE configurations (different widths, layers)
- For each configuration, compute L0/N (per-feature activation budget) and mean affinity advantage
- Plot absorption rate vs. L0/N and fit the predicted sigmoid form: absorption_rate ~ Phi((mean_advantage - C * L0/N) / sigma)
- Compare the fitted phase boundary with the L0 ~ 7-14 transition observed in iteration 5
- Target: R^2 > 0.6 for the sigmoid fit across SAE configurations

### Diagnostic Experiment: The Immune Escape Test

This is the key experiment that distinguishes the immunological theory from simpler explanations:

**Setup**: Identify 50+ parent-child pairs where absorption is confirmed by the Chanin et al. metric.

**Intervention**: For each pair, zero the child feature's activation using SAELens hooks and re-run the SAE on the false-negative tokens.

**Prediction (from immunological theory)**: The parent feature will "recover" (begin firing) on exactly those tokens where it was previously absorbed. The recovery magnitude will be proportional to the child's "affinity advantage" (how much more specifically the child encodes the token's feature than the parent would).

**Alternative prediction (from simpler models)**: If absorption is just a training artifact (the parent direction was never learned), zeroing the child should not cause recovery -- the information is lost, not suppressed.

**Distinguishing test**: If recovery is observed (parent fires after child is zeroed) with magnitude correlated to affinity advantage (r > 0.5), the immunological theory is supported over the "lost information" alternative.

### Experimental Plan

| Phase | Task | Model | Time Estimate |
|-------|------|-------|---------------|
| 1 | Theory formalization | N/A (pen and paper) | 2 hours |
| 2 | Immune escape test on first-letter task | Gemma 2 2B + Gemma Scope 16k/65k | 30 min |
| 3 | CLR normalization pool analysis | Gemma 2 2B + Gemma Scope 16k/65k | 20 min |
| 4 | Absorption threshold curve fitting | Gemma 2 2B + Gemma Scope (all configs) | 1 hour |
| 5 | Cross-domain validation (city-country) | Gemma 2 2B + Gemma Scope | 45 min |
| Total | | | ~4.5 hours |

All experiments are training-free, using pre-trained Gemma Scope SAEs via SAELens + TransformerLens.

### Risk Assessment

1. **Risk: The immune escape test fails (parent features do not recover when child is zeroed).**
   - Implication: Absorption may be a true information loss (the parent direction was never properly learned) rather than a suppression phenomenon. This would falsify the immunological theory but is itself an informative result.
   - Mitigation: Run the test on multiple SAE widths. Wider SAEs are more likely to show recovery because they have more capacity to represent the parent direction.

2. **Risk: The CLR metric does not correlate with absorption rate.**
   - Implication: The normalization pool concept may be too coarse to capture the specific conditions for absorption.
   - Mitigation: Refine the normalization pool definition by incorporating firing rate asymmetry (from the Innovator perspective's SGWA proposal) in addition to cosine similarity.

3. **Risk: The absorption threshold sigmoid does not fit well.**
   - Implication: The immunodominance threshold may not be the right functional form for SAE absorption. Alternative: the transition may be sharper (discontinuous, more like a physical phase transition) or smoother (power law) than the predicted sigmoid.
   - Mitigation: Fit multiple functional forms (sigmoid, power law, logistic) and report the best fit with model comparison statistics (AIC/BIC).

4. **Risk: The immunological framework is correct but adds no predictive power beyond the simpler "sparsity gain" argument from Chanin et al.**
   - Implication: The analogy is true but vacuous. The Lotka-Volterra dynamics reduce to the same qualitative prediction as the existing analysis.
   - Mitigation: The immune framework MUST produce at least one prediction that the existing analysis cannot: the recovery magnitude in the immune escape test. If Chanin et al.'s framework predicts recovery equally well, the contribution is diminished.

### Novelty Claim

The specific cross-disciplinary insights that are new:

1. **Immunological imprinting as a formal model of absorption**: No prior work connects OAS / immunodominance to SAE feature dynamics. The formal mapping (Table in Structural Correspondence) is entirely novel.

2. **The immune escape test**: The diagnostic experiment (zeroing the child feature to see if the parent recovers) has never been performed in the SAE absorption literature. It provides a unique experimental signature that distinguishes suppression-based absorption from information-loss-based absorption.

3. **Cross-level divisive normalization as the mechanistic explanation**: Framing absorption as a failure of level-appropriate normalization (the child's activation enters the parent's normalization pool) is novel and connects to the proven sparse coding / divisive normalization equivalence.

4. **Mitigation strategies from immunology**: The mapping of multi-valent vaccination to multi-scale training, and adjuvant boosting to parent feature anchoring, provides a principled theoretical rationale for existing SAE improvements (Matryoshka SAEs, feature anchoring) that was previously lacking.

5. **Absorption onset threshold from repertoire competition**: The prediction that absorption rate follows a sigmoid in L0/N with a specific critical threshold derived from immunodominance dynamics is quantitatively novel.
