# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Olshausen & Field (1997)** -- "Sparse coding with an overcomplete basis set: A strategy employed by V1?" *Vision Research*. Foundational work showing V1 uses overcomplete sparse coding where lateral inhibition between neurons implements "explaining away" -- only the most relevant basis functions fire for each input, suppressing redundant neighbors. This is the direct biological precursor of SAE-style dictionary learning.

2. **Friston (2010)** -- "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*. The brain minimizes variational free energy (= surprise) through hierarchical predictive coding. Error units at each level send prediction errors upward; state units send predictions downward. The system decomposes signals into hierarchical representations where higher levels "explain away" lower-level variance.

3. **Paiton et al. (2020)** -- "Selectivity and robustness of sparse coding networks." *Journal of Vision*. Shows that population nonlinearities from lateral inhibition in sparse coding networks improve selectivity to preferred stimuli and protect against adversarial perturbations. The iso-response surfaces curve outward, creating sharper feature boundaries.

4. **King et al. (2013)** -- "Inhibitory Interneurons Decorrelate Excitatory Cells to Drive Sparse Code Formation in a Spiking Model of V1." *J. Neuroscience*. Demonstrates that a separate inhibitory population (not just lateral inhibition) is needed for biological sparse coding -- inhibitory interneurons decorrelate excitatory cell activity, driving sparse code formation.

5. **Mysore et al. (2020)** -- "Mechanisms of competitive selection: A canonical neural circuit framework." *eLife*. Identifies the "reciprocal inhibition of inhibition" motif as the most efficient circuit for achieving flexible selection boundaries in winner-take-all neural competition.

#### Physics / Information Theory

6. **Mehta & Schwab (2014)** -- "An exact mapping between the Variational Renormalization Group and Deep Learning." Shows an exact correspondence between Kadanoff block spin renormalization and restricted Boltzmann machines: each layer of deep learning implements a coarse-graining step that preserves long-distance correlations while discarding short-range detail.

7. **Phase transitions in compressed sensing** (Donoho et al.; Wu & Verdu, 2012) -- Sharp phase transitions exist in sparse recovery: below a critical measurement rate, L1 minimization fails completely; above it, exact recovery occurs with high probability. The replica symmetry framework from statistical mechanics (Ising model) is used to derive optimal thresholds.

8. **Phase transitions in deep neural networks** (Phys. Rev. Research, 2023; arXiv:2512.11866, 2025) -- Loss landscapes of deep networks exhibit first- and second-order phase transitions. Competition between prediction error and model complexity causes symmetry-breaking transitions between a trivial phase and a feature-learning phase. Hierarchical phase transitions between accuracy basins correspond to specific digits in MNIST.

9. **Koch-Janusz & Ringel (2018)** -- "Mutual information, neural networks and the renormalization group." *Nature Physics*. Uses mutual information as the bridge between RG coarse-graining and neural network feature learning, showing deep networks learn to preserve the most informative degrees of freedom at each scale.

10. **Sciety preprint (2026)** -- "Renormalization-Group Principles for Deep Neural Architectures." Establishes formal correspondence between network depth and RG scale transformations via Fisher information geometry. Layer-k representations correspond to effective theories at correlation scale xi_k; required depth scales logarithmically with data correlation length.

#### Biology / Evolution / Immunology

11. **Thomas Francis Jr. (1960)** -- "On the Doctrine of Original Antigenic Sin." Foundational immunology: the immune system becomes permanently biased toward the first pathogen strain encountered. Memory B cells from the first exposure outcompete naive B cells responding to novel variants, creating systematic "holes" in the immune repertoire for new threats.

12. **Selin et al. (2004)** -- "T cell immunodominance and maintenance of memory regulated by unexpectedly cross-reactive pathogens." *Nature Immunology*. Cross-reactive memory T cells compete with naive T cells, altering immunodominance hierarchies. This modulates clonal dominance and narrows the TCR repertoire. Mathematical models show active attrition tempers the immunodominance of low-affinity but high-frequency cross-reactive cells.

13. **Sewell (2012)** -- "Why must T cells be cross-reactive?" *Nature Reviews Immunology*. The TCR repertoire is far smaller than the space of possible pathogen peptides, so each T cell must recognize multiple peptide-MHC complexes. This compromise on specificity creates systematic biases in which epitopes get recognized.

14. **Gause (1934) / Chesson (2000)** -- Competitive exclusion principle and modern coexistence theory. Two species using identical resources cannot coexist; coexistence requires niche differentiation (intraspecific competition > interspecific competition). The Lotka-Volterra competition equations predict extinction of the weaker competitor when niches overlap completely.

15. **Holt (1977)** -- "Predation, apparent competition, and the structure of prey communities." *Theoretical Population Biology*. Apparent competition: two prey species that share a predator negatively affect each other indirectly -- increased abundance of one prey boosts predator density, which suppresses the other. No direct competition needed for exclusion.

### Cross-Disciplinary Gaps

The following transplants have NOT been explored in the context of SAE feature absorption:

1. **Original Antigenic Sin --> Feature Absorption**: The structural parallel between immune imprinting (first-exposure memory B cells suppress novel responses) and SAE absorption (high-frequency features suppress low-frequency co-occurring features) has never been formalized. No ML paper draws this analogy.

2. **Apparent Competition --> Feature Absorption**: The indirect competition mechanism (where a shared "predator" mediates exclusion between prey that do not directly compete) has not been applied to understand how the sparsity penalty mediates absorption between hierarchically related features.

3. **Renormalization Group Coarse-Graining --> SAE Feature Hierarchy**: While RG-deep learning connections exist for general architectures, no work applies RG principles specifically to analyze how SAEs handle multi-scale feature hierarchies and why coarse-graining (absorption of fine-grained into coarse-grained features) emerges from sparsity optimization.

---

## Phase 2: Initial Candidates

### Candidate A: Immunological Imprinting Theory of Feature Absorption (from Immunology)

- **Source principle**: Original Antigenic Sin (OAS) / Immunodominance hierarchy. When the immune system encounters a new pathogen variant, pre-existing memory B cells (from a prior, related infection) outcompete naive B cells that could mount a better-targeted response. The result: the immune repertoire develops systematic blind spots for novel epitopes that co-occur with previously seen ones.

- **Structural correspondence**:
  | Immune System | SAE |
  |---|---|
  | Pathogen epitope | Input activation pattern |
  | B cell clone / T cell receptor | SAE latent feature (dictionary atom) |
  | Memory B cell (first exposure) | High-frequency feature with learned decoder direction |
  | Naive B cell (novel response) | Low-frequency feature that should fire but does not |
  | Antibody affinity | Cosine similarity between decoder direction and input |
  | Clonal competition during germinal center reaction | Feature competition during SAE encoding (sparsity penalty) |
  | Immunodominance hierarchy | Feature frequency hierarchy (Zipfian distribution) |
  | OAS: memory cells suppress naive response | Absorption: high-frequency feature suppresses co-occurring low-frequency feature |
  | TCR repertoire narrowing | Reduced effective dictionary coverage (dead or absorbed features) |
  | "Antigenic seniority" (childhood strains ranked highest) | Feature "seniority" by training frequency -- features learned early and reinforced often dominate |

  The formal mapping: Let f_parent be a general (high-frequency) feature and f_child be a specific (low-frequency) feature that always co-occurs with f_parent. In SAE optimization, activating both f_parent and f_child on co-occurrence tokens costs 2 units of L0 but gains only marginally better reconstruction than activating f_parent alone (which already partially reconstructs the child's direction). The sparsity pressure acts like the clonal competition mechanism: the "memory" feature (f_parent, already well-established with high activation frequency) outcompetes the "naive" feature (f_child, rarely seen independently), causing f_child to be absorbed.

- **Hypothesis**: Feature absorption severity follows an "antigenic seniority" law: features learned earlier in training and activated more frequently have stronger absorptive power over co-occurring features, independent of the hierarchical relationship. Specifically: (1) The absorption rate of feature f_child by f_parent scales with the ratio of their activation frequencies (analogous to the ratio of memory vs. naive B cell precursor frequencies). (2) "Cross-reactive" absorption exists: a feature can absorb semantically unrelated features that happen to co-occur, just as cross-reactive memory T cells alter immunodominance hierarchies for unrelated pathogens. (3) Disrupting co-occurrence patterns during training (analogous to priming with conserved epitopes) should reduce absorption, which the recent masked regularization paper (arXiv:2604.06495) already provides preliminary evidence for.

- **Why not just a metaphor**: The mapping preserves the key mathematical structure -- competitive dynamics under resource constraints (L0 budget = finite immune capacity) where established entities (memory cells / high-frequency features) have a systematic advantage over novel ones (naive cells / low-frequency features) due to co-occurrence (cross-reactivity / feature hierarchy). The prediction about "cross-reactive absorption" (absorption of semantically unrelated co-occurring features) is specific to this analogy and not predicted by the standard hierarchical absorption model.

- **Novelty estimate**: 8/10. The OAS-SAE connection has not been drawn in the literature. The prediction about cross-reactive (non-hierarchical) absorption and the frequency-ratio scaling law are novel and testable.

### Candidate B: Ecological Apparent Competition Theory of Absorption (from Community Ecology)

- **Source principle**: Apparent competition via shared predator (Holt, 1977). Two prey species that share a predator indirectly suppress each other: when prey A increases, the predator population grows, which then suppresses prey B -- even though A and B never directly compete.

- **Structural correspondence**:
  | Ecology | SAE |
  |---|---|
  | Prey species | SAE feature (dictionary atom) |
  | Predator | Sparsity penalty (L1/L0 constraint) |
  | Prey abundance | Feature activation frequency |
  | Predator functional response | How sparsity penalty responds to total active features |
  | Carrying capacity per prey | Reconstruction benefit per feature |
  | Apparent competition: prey A boosts predator, which suppresses prey B | Absorption: feature A's high activation frequency "consumes" sparsity budget, pushing sparsity penalty to suppress co-occurring feature B |
  | Competitive exclusion | Feature death or absorption |
  | Niche differentiation for coexistence | Orthogonal decoder directions enable feature coexistence |

  The Lotka-Volterra competition framework predicts: two features can coexist if and only if intra-feature competition (self-overlap) exceeds inter-feature competition (cross-overlap). In SAE terms, this translates to: two features avoid absorption when the reconstruction benefit of activating both (their decoder directions being sufficiently orthogonal) exceeds the sparsity cost of the extra activation. The "apparent competition" lens adds a crucial insight: even features with orthogonal decoder directions can experience indirect competition through the shared sparsity budget. When a high-frequency feature fires, it raises the "predator density" (effective sparsity pressure on the remaining budget), indirectly suppressing all other features in that activation -- not just hierarchically related ones.

  Formally, let a_i be the activation of feature i, d_i the decoder direction, and lambda the sparsity penalty. The Lotka-Volterra analogy gives:
  da_i/dt ~ r_i * a_i * (1 - a_i/K_i) - alpha_{ij} * a_i * a_j - lambda * P(a_i)
  where r_i is the reconstruction gradient, K_i is the feature capacity, alpha_{ij} = (d_i . d_j)^2 captures direct competition via decoder overlap, and lambda * P(a_i) is the "predation" by the sparsity penalty.

- **Hypothesis**: (1) The Lotka-Volterra competition coefficient alpha_{ij} = (d_i . d_j)^2 between decoder directions predicts absorption probability better than simple cosine similarity. (2) Features experience "diffuse apparent competition" -- even features with zero decoder overlap can suppress each other if they frequently co-activate, because they share the sparsity budget. (3) Increasing dictionary width acts like increasing habitat heterogeneity in ecology: more niches become available, allowing more features to coexist, which predicts the empirical observation that wider SAEs reduce (but do not eliminate) absorption. (4) Matryoshka SAE's success is analogous to explicit niche partitioning -- forcing different width levels to specialize at different hierarchy levels prevents competitive exclusion.

- **Why not just a metaphor**: The Lotka-Volterra framework provides actual differential equations that can be instantiated for the SAE optimization dynamics. The decoder overlap (d_i . d_j)^2 maps precisely to the ecological competition coefficient. The "diffuse apparent competition" prediction (sparsity-budget-mediated indirect competition) is a specific quantitative prediction that goes beyond the simple hierarchical absorption model.

- **Novelty estimate**: 7/10. The Locally Competitive Algorithm (Rozell et al., 2008) already uses competition dynamics for sparse coding, and winner-take-all autoencoders exist. However, the specific Lotka-Volterra formalization of absorption dynamics and the "apparent competition" mechanism have not been applied to understand SAE feature absorption.

### Candidate C: Renormalization Group Phase Transition Theory of Absorption (from Statistical Physics)

- **Source principle**: Renormalization group (RG) coarse-graining and phase transitions. In statistical physics, the RG iteratively integrates out short-distance degrees of freedom, producing an effective theory at each scale. At certain parameter values, the system undergoes phase transitions where the effective theory changes qualitatively. Kadanoff block spin transformations explicitly merge fine-grained variables into coarse-grained ones.

- **Structural correspondence**:
  | Statistical Physics | SAE |
  |---|---|
  | Microscopic degrees of freedom (spins) | Fine-grained features (specific tokens, narrow concepts) |
  | Coarse-grained block spins | General features (broad categories, parent concepts) |
  | RG flow (integrating out short-distance modes) | Sparsity-driven absorption of fine-grained into coarse-grained features |
  | Relevant operators (grow under RG) | High-frequency features that strengthen during training |
  | Irrelevant operators (shrink under RG) | Low-frequency features that get absorbed or die |
  | Phase transition (qualitative change in behavior) | Critical sparsity threshold where absorption emerges |
  | Correlation length | Feature hierarchy depth |
  | Fixed point of RG flow | Converged SAE feature set |
  | Universality class | Architecture-independent absorption patterns |

  The formal analogy: SAE training with increasing sparsity penalty implements an RG-like flow where fine-grained features are progressively "integrated out" (absorbed) into coarse-grained features. There should exist a critical sparsity value lambda_c below which the SAE maintains distinct features at all hierarchy levels (disordered phase) and above which absorption cascades emerge (ordered phase). The order parameter is the absorption rate itself.

- **Hypothesis**: (1) A phase transition in absorption rate exists at a critical sparsity value lambda_c, analogous to the compressed sensing phase transition. Below lambda_c, SAEs maintain both parent and child features; above lambda_c, absorption cascades as fine-grained features are systematically absorbed. (2) Absorption follows RG "universality" -- the critical exponents (how absorption rate scales near lambda_c) are independent of SAE architecture (TopK, JumpReLU, L1) and depend only on the feature hierarchy statistics. (3) The depth of the feature hierarchy determines the required SAE "correlation length" (dictionary width): deeper hierarchies require exponentially wider dictionaries, analogous to how longer correlation lengths require larger simulation boxes.

- **Why not just a metaphor**: The phase transition prediction is quantitative and falsifiable. Compressed sensing already has proven phase transitions in sparse recovery; the claim here is that SAE absorption exhibits an analogous transition. The universality prediction (architecture independence of critical exponents) can be tested by measuring absorption vs. sparsity curves across multiple SAE architectures on the same model.

- **Novelty estimate**: 7/10. RG-deep learning connections exist (Mehta & Schwab, 2014), and phase transitions in compressed sensing are well-studied. However, applying the RG framework specifically to SAE feature absorption, predicting a sparsity-driven phase transition in absorption rate, and the universality hypothesis are novel.

---

## Phase 3: Self-Critique

### Against Candidate A (Immunological Imprinting)

- **Shallow analogy attack**: The mapping preserves the essential mathematical structure: competitive dynamics under resource constraints where established entities suppress novel ones. However, a key difference exists -- in OAS, the memory cells are functionally equivalent but have a kinetic advantage (faster activation); in SAEs, the absorbing feature may genuinely encode the absorbed feature's information within its decoder direction (the cosine similarity > 0.025 criterion from Chanin et al.). This means SAE absorption is partly about information redundancy, not just competitive dynamics. A domain expert in immunology might object that OAS is fundamentally about response speed, not information encoding. **Mitigation**: The analogy is strongest for the "cross-reactive absorption" prediction (semantically unrelated co-occurring features absorbing each other), where the information-redundancy explanation does not apply and the competitive dynamics explanation is necessary.

- **Scale mismatch attack**: OAS operates at the population level (millions of B cell clones), while SAE absorption operates at the level of individual latent dimensions (thousands of features). The population dynamics may not transfer to the optimization landscape. **Mitigation**: The Lotka-Volterra framework (Candidate B) provides the formal bridge -- competition dynamics apply at both scales. SAE training via SGD can be viewed as a population-level process where gradient updates play the role of clonal expansion.

- **Prior transplant check**: Searched arXiv for "original antigenic sin" + "neural network" / "machine learning" / "autoencoder" / "feature learning". Found no papers applying OAS to SAE or dictionary learning. The catastrophic forgetting literature discusses the stability-plasticity dilemma (which is the inverse of OAS) but does not make the specific absorption connection. **Verdict**: No prior transplant found.

- **Testability attack**: The "cross-reactive absorption" prediction is directly testable: measure whether SAE features absorb semantically unrelated but co-occurring features (e.g., features for common function words absorbing features for specific content words that always appear in the same context). This can be done using the Chanin et al. absorption metric framework extended beyond the first-letter task. The frequency-ratio scaling law can be tested by correlating absorption rate with the log ratio of parent/child activation frequencies across many feature pairs.

- **Verdict**: STRONG. The analogy has genuine structural depth (competitive dynamics under resource constraints + frequency hierarchy), novel testable predictions (cross-reactive absorption, frequency scaling law), and no prior transplant. Weakness: the information-redundancy aspect of absorption is not captured by the immune analogy alone.

### Against Candidate B (Ecological Apparent Competition)

- **Shallow analogy attack**: The mapping between ecological competition and sparse coding competition is well-grounded mathematically -- the Locally Competitive Algorithm already instantiates this connection. The "apparent competition" refinement (indirect competition through shared sparsity budget) adds a genuine novel prediction. However, a domain ecologist might note that in ecology, prey species have their own growth dynamics independent of other prey, while SAE features are entirely defined by their relationship to the training data -- features do not have "intrinsic fitness." **Mitigation**: The reconstruction gradient r_i serves as the SAE analog of intrinsic growth rate, and it does depend on the data distribution independently of other features.

- **Scale mismatch attack**: Ecological dynamics unfold over generations and geographic space; SAE optimization unfolds over gradient steps in parameter space. The mapping of time scales is informal. **Mitigation**: The mathematical structure (Lotka-Volterra differential equations) is scale-invariant -- it applies to any competitive dynamical system regardless of the underlying physical mechanism.

- **Prior transplant check**: The Locally Competitive Algorithm (Rozell et al., 2008) uses Lotka-Volterra-like competition dynamics for sparse coding inference. Winner-Take-All autoencoders (Makhzani & Frey, 2015) use explicit competition. However, neither applies the ecological framework specifically to analyze SAE *failure modes* (absorption, hedging) or uses the "apparent competition" concept. The specific application to feature absorption is novel.

- **Testability attack**: The alpha_{ij} = (d_i . d_j)^2 predictor can be directly tested against the simpler cosine similarity predictor for absorption. The "diffuse apparent competition" prediction (absorption between features with zero decoder overlap) is testable but may be hard to distinguish from confounds (features with zero decoder overlap rarely co-occur, so the signal may be weak).

- **Verdict**: MODERATE. The mathematical framework is well-suited and provides testable predictions, but the connection between competition dynamics and sparse coding already exists in the literature (LCA). The novelty lies specifically in the "apparent competition" mechanism and its application to absorption, which is a refinement rather than a wholly new transplant.

### Against Candidate C (RG Phase Transition)

- **Shallow analogy attack**: The mapping between RG coarse-graining and SAE absorption is structurally appealing -- both involve integrating out fine-grained information into coarser representations. However, a physicist might object that RG flow is a systematic procedure applied to a fixed Hamiltonian, while SAE training simultaneously changes the Hamiltonian (learned decoder) and the effective theory (feature activations). The RG analogy may describe the *result* of absorption (coarse-graining) without explaining the *mechanism* (why it happens during optimization). **Mitigation**: The phase transition prediction does not require the full RG formalism -- only the existence of a sharp threshold in the sparsity-absorption relationship, which is a more modest claim.

- **Scale mismatch attack**: RG operates on spatially extended systems with genuine scale separation; SAEs operate on fixed-dimensional activation vectors without spatial structure. The "correlation length" mapping to hierarchy depth is informal. **Mitigation**: Compressed sensing phase transitions occur in non-spatial settings (random matrices), showing that sharp thresholds can emerge from sparsity constraints alone.

- **Prior transplant check**: Mehta & Schwab (2014) established the RG-deep learning connection. Phase transitions in compressed sensing are well-studied (Donoho, Wu & Verdu). The specific application to SAE absorption and the universality prediction are novel, but the general framework is well-explored.

- **Testability attack**: The phase transition prediction requires sweeping sparsity across many values for each SAE configuration, which is computationally expensive but feasible using existing pretrained SAEs at different sparsity levels (SAEBench provides these). The universality prediction requires comparing absorption-vs-sparsity curves across architectures on the same model -- also feasible with SAEBench's 200+ SAEs. However, "phase transition" is a strong claim -- the absorption-sparsity relationship might be smooth (crossover) rather than sharp (true transition). Distinguishing these requires careful finite-size scaling analysis.

- **Verdict**: MODERATE. The phase transition and universality predictions are bold and testable, but the RG-deep learning connection is already established territory. The specific absorption application is novel but may not yield sharp transitions in practice (finite-size effects in practical SAE widths may smear any transition).

---

## Phase 4: Refinement

### Dropped

**Candidate C (RG Phase Transition)** is dropped as the front-runner for the following reasons:
- The general RG-deep learning connection is already well-explored (Mehta & Schwab, 2014)
- Phase transitions in compressed sensing are known territory
- The novelty is mainly in *applying* known frameworks rather than *discovering* a new structural correspondence
- The experimental plan would require extensive compute to sweep sparsity values systematically
- The prediction may be softened by finite-size effects

However, the **phase transition insight from Candidate C is retained as a supporting prediction** within the winning proposal: the existence of a critical sparsity threshold for absorption onset can be tested as a secondary experiment.

### Strengthened Survivors

#### Candidate A (Immunological Imprinting) -- Selected as Front-Runner

**Strengthened formal correspondence**:

The key structural isomorphism is between two systems that solve the same computational problem: **allocating finite capacity to represent a combinatorial space under exposure-frequency bias**.

Define:
- Immune system: Finite B cell repertoire R of size |R| must cover pathogen epitope space E of size |E| >> |R|. Each B cell clone c_i has affinity a(c_i, e_j) for epitope e_j. Sparsity constraint: only k clones are activated per infection (germinal center capacity). Frequency bias: clones activated more often have more memory cells, giving them competitive advantage.
- SAE system: Finite dictionary D of size |D| must cover feature space F of size |F| >> |D|. Each dictionary atom d_i has alignment cos(d_i, f_j) with true feature f_j. Sparsity constraint: only L0 atoms are activated per input. Frequency bias: atoms activated more often have stronger encoder weights, giving them competitive advantage.

The OAS phenomenon emerges in both systems when:
1. A high-frequency entity (memory B cell / high-frequency SAE feature) partially covers the activation pattern of a low-frequency entity (novel epitope / rare feature)
2. The capacity constraint (germinal center / L0 budget) forces a choice
3. The high-frequency entity wins due to competitive advantage (faster activation kinetics / stronger encoder weights)
4. The low-frequency entity fails to activate even when it would provide better coverage

**Diagnostic experiment** (the key test that confirms the analogy is load-bearing):

The OAS model makes a unique prediction not made by the standard hierarchical absorption model: **cross-reactive absorption** -- absorption of features that are semantically unrelated but statistically co-occurring. The standard model explains absorption only through feature hierarchy (parent absorbs child). The OAS model predicts that ANY high-frequency feature can absorb ANY low-frequency feature it co-occurs with, regardless of semantic relationship, purely through competitive dynamics.

Test: Construct a controlled feature hierarchy with two types of feature pairs:
- **Hierarchical pairs**: parent-child (e.g., "starts with letter A" absorbs "the token 'apple'")
- **Co-occurrence pairs**: semantically unrelated but co-occurring (e.g., "common determiner" and "rare noun that always follows determiners")

If the OAS model is correct, both types should exhibit absorption at rates predicted by the frequency ratio. If only the standard model is correct, only hierarchical pairs should show absorption.

#### Candidate B (Ecological Apparent Competition) -- Retained as Supporting Framework

The apparent competition mechanism provides the mathematical formalization (Lotka-Volterra equations) for the competitive dynamics hypothesized in Candidate A. Retained as the quantitative backbone for the experimental plan.

---

## Phase 5: Final Proposal

### Title

**Immunological Imprinting in Sparse Autoencoders: A Cross-Reactive Theory of Feature Absorption**

### Source Principle

**Original Antigenic Sin (OAS)** from immunology: the immune system's tendency to recall responses against the primary antigen encountered rather than generating de novo responses to novel variants. Memory B cells from prior exposure outcompete naive B cells, creating systematic blind spots ("absorbed" epitopes) in the immune repertoire. This phenomenon is driven by competitive dynamics under finite capacity constraints, where exposure frequency determines competitive advantage.

### Structural Correspondence

Both the adaptive immune system and SAEs solve the same computational problem: representing a combinatorial space (pathogen epitopes / neural activation patterns) using a sparse, overcomplete dictionary (B cell repertoire / SAE features) under a finite capacity constraint (germinal center throughput / L0 sparsity). The OAS phenomenon maps precisely onto feature absorption:

| Immune System Concept | SAE Concept | Mathematical Mapping |
|---|---|---|
| B cell clone c_i | Dictionary atom d_i | Direction vector in activation space |
| Epitope e_j | True feature f_j | Direction in activation space |
| Affinity a(c_i, e_j) | Decoder alignment cos(d_i, f_j) | Cosine similarity |
| Memory B cell frequency | Feature activation frequency | Proportion of training tokens where feature fires |
| Germinal center capacity | L0 sparsity budget | Maximum simultaneous activations |
| Clonal competition | Encoder competition via sparsity penalty | Gradient competition during SGD |
| OAS (memory suppresses naive) | Absorption (frequent suppresses rare) | High-freq feature absorbs co-occurring low-freq feature |
| Cross-reactive immunity | Cross-reactive absorption | Absorption of semantically unrelated co-occurring features |
| Antigenic seniority | Training-frequency seniority | Features established earlier in training have stronger absorptive power |
| Repertoire narrowing | Effective dictionary shrinkage | Reduction in features with non-trivial activation patterns |

### Hypothesis

1. **Cross-reactive absorption exists**: SAE features absorb co-occurring features regardless of semantic/hierarchical relationship, driven purely by frequency-based competitive advantage. This goes beyond the current understanding that absorption is exclusively a hierarchical phenomenon (parent absorbs child).

2. **Frequency-ratio scaling law**: The probability of feature f_j being absorbed by feature f_i scales with log(freq(f_i) / freq(f_j)) -- analogous to the "antigenic seniority" model where response magnitude scales with exposure recency/frequency.

3. **Training-order imprinting**: Features established in early training have disproportionate absorptive power over features that emerge later, even at matched final frequencies. This mirrors the "first flu is forever" phenomenon.

4. **Masked regularization as "priming with conserved epitopes"**: Disrupting co-occurrence patterns during training (masked regularization, arXiv:2604.06495) reduces absorption by the same mechanism that priming with broadly conserved epitopes reduces OAS in vaccines -- it prevents the "memory response" from monopolizing the capacity budget.

### Method

**Phase 1: Establish Cross-Reactive Absorption (Novel Finding)**

Using pre-trained Gemma Scope SAEs (training-free analysis):
1. For each SAE feature, compute activation frequency and co-occurrence matrix with all other features
2. Identify feature pairs that are (a) semantically unrelated (low decoder cosine similarity < 0.1) but (b) statistically co-occurring (co-activation rate > 5x chance level)
3. Apply an extended version of the Chanin et al. absorption metric to these "cross-reactive" pairs:
   - Train logistic regression probes for known concept hierarchies (city/country, entity type, syntactic role)
   - Identify false negatives: probe says feature should fire but SAE feature does not
   - Check if a co-occurring but semantically unrelated feature explains the false negative (via integrated gradients)
4. Compare cross-reactive absorption rates against hierarchical absorption rates

**Phase 2: Test Frequency-Ratio Scaling Law**

Using SAEBench's 200+ pre-trained SAEs across multiple architectures:
1. For each SAE, extract all feature pairs where absorption is detected
2. Compute log frequency ratio for each absorbed pair
3. Fit logistic regression: P(absorption) ~ log(freq_ratio) + controls (decoder similarity, L0, width)
4. Test whether frequency ratio is a significant predictor above and beyond decoder similarity (the standard hierarchical predictor)

**Phase 3: Diagnostic Experiment -- Distinguishing OAS Model from Hierarchical Model**

Construct controlled probe tasks with known feature hierarchies on GPT-2 small:
- **Hierarchical pairs**: First-letter features (standard Chanin et al. task)
- **Cross-reactive pairs**: Function word features (determiners, prepositions) paired with content word features (specific nouns) that co-occur due to syntactic patterns
- **Control pairs**: Features with matched frequency ratios but no co-occurrence

If the OAS model is correct: absorption rate for cross-reactive pairs > control pairs (after controlling for frequency ratio and decoder similarity). If only the hierarchical model is correct: cross-reactive pairs = control pairs.

**Phase 4: Test Mitigation Predictions**

Compare absorption rates across SAE architectures through the OAS lens:
- Matryoshka SAE = explicit niche partitioning (different hierarchy levels get dedicated capacity) -- predicts reduced hierarchical absorption but not necessarily cross-reactive absorption
- OrtSAE = forced decoder orthogonality -- predicts reduced both types (reduces "affinity overlap")
- Masked regularization = disrupted co-occurrence ("priming with conserved epitopes") -- predicts reduced cross-reactive absorption specifically
- Test whether these predictions match empirical absorption patterns in SAEBench

### Diagnostic Experiment

The key test that confirms the analogy is load-bearing (not just decorative):

**If cross-reactive absorption exists at rates predicted by the frequency ratio** (independent of decoder similarity and hierarchical relationship), then the OAS competitive dynamics model explains a phenomenon that the standard hierarchical model cannot. Conversely, if absorption is found exclusively in hierarchical pairs, the OAS analogy is merely decorative and the standard model suffices.

Secondary diagnostic: If training-order imprinting is confirmed (features established early have stronger absorptive power at matched frequency), this demonstrates that the temporal dynamics of the OAS analogy, not just the steady-state competition, are relevant to SAE behavior.

### Experimental Plan

| Phase | Task | Model | Time Estimate |
|---|---|---|---|
| 1a | Compute feature co-occurrence matrix | Gemma-2-2B + Gemma Scope 16k SAE | ~20 min |
| 1b | Identify cross-reactive pairs | (analysis of 1a output) | ~10 min |
| 1c | Extended absorption metric on cross-reactive pairs | Gemma-2-2B | ~30 min |
| 2 | Frequency-ratio regression across SAEBench SAEs | Pythia-160M + Gemma-2-2B | ~30 min |
| 3a | Construct controlled probe tasks | GPT-2 Small | ~15 min |
| 3b | Measure absorption by pair type | GPT-2 Small + pre-trained SAEs | ~30 min |
| 4 | Architecture comparison through OAS lens | SAEBench SAEs | ~20 min |
| Total | | | ~2.5 hours |

Each phase is independently meaningful and can be reported separately. Phase 3b is the critical diagnostic experiment.

Tools: SAELens v6, TransformerLens, sae-spelling (absorption metric code), SAEBench, Gemma Scope pretrained SAEs, GPT-2 SAEs from Neuronpedia.

### Risk Assessment

1. **Cross-reactive absorption may not exist or may be very rare**: If features with zero decoder overlap never absorb each other, the OAS analogy reduces to the standard hierarchical model with extra vocabulary. Mitigation: measure at a range of decoder similarity thresholds, not just zero.

2. **Frequency ratio may be confounded with decoder similarity**: High-frequency features may tend to have higher decoder similarity with many features simply because they are broader. Mitigation: use partial correlation / regression with decoder similarity as a control variable.

3. **Training-order effects may be washed out by convergence**: SGD dynamics may erase early-training biases by the time the SAE converges. Mitigation: analyze SAE training checkpoints (available for some SAEBench SAEs) to measure absorption emergence over training.

4. **The immune analogy may be too complex for the actual mechanism**: SAE absorption may have a simpler explanation (pure optimization artifact) that does not require the full OAS framework. Mitigation: the diagnostic experiment (Phase 3) explicitly tests whether the OAS model explains phenomena the simpler model cannot.

### Novelty Claim

The specific cross-disciplinary insight is that SAE feature absorption is an instance of a broader phenomenon -- **competitive capacity allocation under frequency bias** -- that was first characterized in immunology as Original Antigenic Sin. This insight:

1. **Has not been applied to SAEs or dictionary learning before** (verified via arXiv, Google Scholar, and web search -- no prior work connects OAS to feature absorption or sparse coding)

2. **Generates a novel testable prediction** (cross-reactive absorption of semantically unrelated co-occurring features) that the standard hierarchical model does not make

3. **Provides a unified framework** for understanding multiple SAE failure modes: absorption (= OAS), feature hedging (= insufficient repertoire diversity), dead features (= clonal deletion), and feature inconsistency across training runs (= private TCR repertoire variation between individuals)

4. **Suggests novel mitigation strategies** inspired by vaccine design: "heterologous prime-boost" training schedules that expose the SAE to different co-occurrence patterns at different training stages, analogous to sequential vaccination with different pathogen variants to broaden the immune response
