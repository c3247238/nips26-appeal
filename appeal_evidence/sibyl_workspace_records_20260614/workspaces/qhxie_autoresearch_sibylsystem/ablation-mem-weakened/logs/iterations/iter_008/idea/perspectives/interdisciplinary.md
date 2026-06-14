# Interdisciplinary Perspective

## Phase 1: Literature Survey

### By Source Field

#### Neuroscience / Cognitive Science

1. **Hierarchical Feature Organization in Visual Cortex** — The visual cortex exhibits a hierarchical organization: simple cells detect edges, complex cells combine edge orientations, and hypercomplex cells detect corners and terminations. This hierarchy creates an analogous problem where higher-level features (e.g., "dog") might "absorb" lower-level features (e.g., "fur texture") through polysynaptic chains. The structural insight is that biological systems solve the same sparse coding problem but with metabolic efficiency constraints that prevent the over-compression seen in SAEs. *Key principle: Lateral inhibition and competitive activation in cortical columns.*

2. **Cortical Inhibitory Interneurons and Sparse Coding** — GABAergic interneurons enforce sparsity in biological neural networks through inhibition-weighted competition. The principle of "winner-take-all" inhibition in cortical microcircuits suggests a mechanism for preventing absorption: active features actively suppress related but distinct features. This offers a potential regularization strategy for SAEs. *Key principle: Activity-dependent inhibition as a sparsity mechanism.*

3. **Predictive Coding and Precision Weighting** — Free Energy Principle frameworks (Friston) propose that the brain uses hierarchical predictive coding where precision (inverse variance) weights determine how prediction errors propagate. In this framework, absorption might correspond to a failure of precision weighting at intermediate layers, where child features "hijack" the prediction error that should represent the parent feature. *Key principle: Precision-weighted prediction error as the unit of information.*

4. **Neural Tuning Curves and Feature Overlap** — Single neuron tuning curves in sensory cortices show that features are represented as distributed, overlapping response functions. The brain never achieves perfect monosemanticity—each neuron responds to multiple features—but the overlap is metabolically bounded. This contrasts with SAEs where absorption creates systematic false negatives. *Key principle: Graceful degradation under representational constraints.*

#### Physics / Information Theory

5. **Phase Transitions in Sparse Coding** — The statistical physics of sparse coding systems exhibits phase transitions as sparsity is varied. At high sparsity (low temperature), the system transitions from a disordered phase (features recovered independently) to an ordered phase (features absorbed). This transition has been studied in spin glass models and combinatorial optimization. *Key principle: Order parameter as absorption indicator; critical sparsity threshold.*

6. **Rate-Distortion Theory and Lossy Compression** — Shannon's rate-distortion theory quantifies the fundamental tradeoff between compression ratio and distortion. Feature absorption can be viewed as a form of representational distortion: the SAE "compresses" the feature space by merging related features. The absorption rate is analogous to the distortion-rate function. *Key principle: Information-theoretic bounds on absorption inevitability.*

7. **Hierarchical Sparse Coding in Computational Neuroscience** — The neuroscience literature on hierarchical sparse coding (Olshausen & Field, 1996; 1997) established that V1 simple cells form efficient codes through sparse coding. Later work (Hyvarinen et al.) extended this to hierarchical models. These frameworks explicitly model the competition between features and could offer formal tools for analyzing absorption. *Key principle: Hierarchical sparse coding objectives with lateral competition terms.*

8. **Replica Symmetry Breaking in Compositional Models** — Statistical physics tools from spin glass theory (Parisi replica symmetry breaking) have been applied to compositional models and superposition. The structure of the free energy landscape determines whether features form discrete compositional clusters or merge continuously. *Key principle: Full replica symmetry breaking as a model for feature absorption.*

#### Biology / Evolution

9. **Gene Duplication and Subfunctionalization (DDC/DAD Model)** — Ohno's classical model: genes duplicate, then partition functions (subfunctionalization) or acquire new functions (neofunctionalization). The DDC (Duplication-Degeneration-Complementation) model describes how duplicated genes lose different subfunctions while preserving the total. This is structurally isomorphic to feature absorption: a "parent" feature duplicates, and the children partition the parent's representational space until one absorbs the other. *Key principle: Degenerative mutations in regulatory elements as the mechanism of absorption.*

10. **Phylogenetic Feature Trees and Evolutionary Distance** — In evolutionary genomics, the evolutionary distance between gene copies indicates their duplication history. Features in an SAE dictionary could be analyzed similarly: absorbed features should show evidence of "evolutionary" relationship to their absorbing partners (high cosine similarity, shared "ancestry" in activation space). U-shaped vs. ladder-shaped phylogenetic trees distinguish between absorption-like vs. branching evolutionary histories. *Key principle: Hierarchical clustering of feature embeddings as "phylogeny."*

11. **Neutral Drift and Mutational Robustness** — Kimura's neutral theory proposes that most evolutionary change is driven by random drift of neutral mutations. In SAE training, absorption might occur through a "neutral drift" process where the network explores the loss landscape without selective pressure to preserve feature independence. This predicts that absorbed features retain latent "neutral" variants detectable through noise injection. *Key principle: Neutral space exploration during training as driver of absorption.*

12. **Immune System Affinity Maturation and Competitive Binding** — During affinity maturation, B-cells undergo somatic hypermutation and compete for antigen binding. Low-affinity clones are outcompeted and absorbed (deleted). The "affinity" of a feature for the residual stream might determine its survival—high-frequency child features outcompete low-frequency parent features for dictionary representation. *Key principle: Competitive selection pressure as absorption mechanism.*

#### Information Theory / Causal Inference

13. **Causal Mediation Analysis** — Absorption is fundamentally a causal phenomenon: the parent feature's causal effect on the output is mediated through the child feature. The natural indirect effect (NIE) measures the proportion of effect mediated. This formal causal framework could provide a principled definition and measurement protocol for absorption that current ablation-based methods lack. *Key principle: Causal mediation decomposition as the formal structure.*

14. **Frontier Development: Dictionary Learning and Compressed Sensing** — The compressed sensing literature (Donoho, Candes & Tao) provides theoretical guarantees for sparse recovery under certain conditions. The failure mode of absorption corresponds to violating the mutual incoherence condition—when dictionary atoms become too correlated, sparse recovery fails systematically. *Key principle: Mutual incoherence violation as absorption predictor.*

### Cross-Disciplinary Gaps

Where transplants haven't been tried yet:

1. **Evolutionary genomics methods for feature family analysis**: No work has applied phylogenetic tree reconstruction, DDC/DAD models, or evolutionary sequence analysis to SAE feature dictionaries to identify absorbed feature lineages.

2. **Statistical physics phase transition analysis**: While superposition has been studied via spin glass models, no work has formally characterized the absorption transition using order parameters and critical exponents in realistic SAE setups.

3. **Causal mediation framework for absorption**: Current absorption metrics rely on ablation/projection. A formal causal mediation decomposition has not been established as the principled measurement framework.

4. **Immune-inspired competitive dynamics**: The clonal selection analogy has not been formalized for understanding absorption dynamics during SAE training.

5. **Cortical inhibition mechanisms for mitigation**: Lateral inhibition-inspired regularization has not been explored as a training-free mitigation strategy for pretrained SAEs.

---

## Phase 2: Initial Candidates

### Candidate A: Phase Transition Framework (from Statistical Physics)

- **Source principle**: Phase transitions in sparse systems—when compression increases beyond a critical threshold, the system undergoes an order-disorder transition where distinct representational states collapse into a disordered absorbed state. The critical point is characterized by order parameters, susceptibility, and critical exponents.

- **Structural correspondence**: The SAE optimization objective (L1 sparsity + L2 reconstruction) is mathematically analogous to a spin glass Hamiltonian. The sparsity k corresponds to inverse temperature. The absorption transition corresponds to a phase transition from ferromagnetic (ordered, features distinct) to paramagnetic (disordered, features merged) states. The order parameter M = cos(parent_decoder, child_decoder) measures magnetization-like alignment. The susceptibility chi = dM/d(1/k) diverges at the critical point.

- **Hypothesis**: Absorption in SAEs exhibits sharp phase transition behavior as sparsity increases: below a critical L0, absorption is rare; above it, absorption increases continuously with a universal critical exponent. The critical sparsity threshold depends on feature hierarchy depth and co-occurrence frequency.

- **Why not just a metaphor**: The mathematical structure is preserved: partition functions, free energy minimization, and order parameters all have direct analogues in the SAE objective. Phase transition theory makes sharp, falsifiable predictions (e.g., critical exponent universality classes) that can be tested empirically.

- **Novelty estimate**: 8/10 — While superposition has been studied via spin glass models (Elhage et al.), the specific phase transition behavior of absorption and its quantitative prediction has not been characterized.

### Candidate B: Evolutionary Feature Phylogeny (from Evolutionary Biology)

- **Source principle**: Gene duplication followed by subfunctionalization (DDC model): duplicated genes initially share all functions, then partition subfunctions through degenerative mutations. Over time, one copy may be absorbed (loses all unique functions) while the other maintains the full ancestral function. The key diagnostic is the phylogenetic tree structure: absorbed genes show ladder-like trees with short external branches, while independently evolving genes show bushy trees.

- **Structural correspondence**: SAE features undergo a similar process during training on hierarchical data. A "parent" feature (general concept) duplicates into child features (specific concepts) through dictionary learning. Under sparsity pressure, children partition the parent's activation patterns until one absorbs the parent entirely. The feature embedding space contains a "phylogeny" of absorbed features detectable through hierarchical clustering.

- **Hypothesis**: Absorbed features in SAEs show phylogenetic signatures of duplication events: (1) high cosine similarity to absorbing features (short internal branches), (2) clustering in embedding space with other absorbed features (phylogenetic ladder structure), (3) more "relatives" in the feature family tree than independent features. The absorption probability increases with phylogenetic proximity to absorbing features.

- **Why not just a metaphor**: The DDC model makes specific quantitative predictions about the correlation between tree topology and absorption likelihood. Hierarchical clustering algorithms and phylogenetic tree reconstruction methods can be directly imported from computational genomics to analyze SAE dictionaries.

- **Novelty estimate**: 9/10 — This is a genuinely novel application of evolutionary genomics methods to SAE analysis. No existing SAE literature applies phylogenetic tree analysis, DDC models, or evolutionary sequence analysis to feature dictionaries.

### Candidate C: Causal Mediation Decomposition (from Causal Inference)

- **Source principle**: Causal mediation analysis (Pearl, Imai et al.) decomposes total effect into direct and indirect (mediated) effects. The natural indirect effect (NIE) measures the proportion of effect transmitted through a mediator. The key identification assumption is sequential ignorability: the mediator is independent of unobserved confounders given the treatment and covariates.

- **Structural correspondence**: Feature absorption is the natural indirect effect of the parent feature through the child feature. When we ablate the parent but not the child, we eliminate both the direct effect of the parent and its indirect effect through the child. The absorption rate is the proportion of parent's causal effect on the output that flows through the child. The child mediates the parent's effect on the SAE reconstruction or model output.

- **Hypothesis**: The absorption rate equals NIE(parent -> output | child active) / Total Effect(parent on output). This can be estimated via the mediator substitution approach: compare SAE output when (1) parent active + child inactive, (2) parent inactive + child active, (3) both active. The absorption metric is the ratio of indirect to total effect.

- **Why not just a metaphor**: Causal mediation is the formal mathematical framework for decomposing effects through mediators. The structural causal model (SCM) for absorption is directly mappable: parent feature -> child feature -> output, with direct path parent -> output. This gives precise identification assumptions and estimation formulas.

- **Novelty estimate**: 7/10 — Causal framing of absorption has been mentioned qualitatively in the literature (文献调研), but no work has operationalized causal mediation analysis as a measurement framework or used mediation decomposition to define the absorption metric formally.

### Candidate D: Cortical Inhibition Regularization (from Neuroscience)

- **Source principle**: In cortical microcircuits, GABAergic interneurons enforce sparse coding through lateral inhibition. Active neurons inhibit related neurons, preventing redundant representation. The inhibition strength is activity-dependent: more active features suppress less active related features proportionally. This prevents "absorption" of low-activity features by high-activity neighbors.

- **Structural correspondence**: SAE absorption occurs because related features (parent-child) compete for dictionary representation but the loss function doesn't penalize representational redundancy. In cortical models, the sparsity penalty is augmented by an inhibition term: L_total = L_recon + lambda * L_sparsity + rho * sum_i sum_j W_ij * a_i * a_j where W_ij is inhibition weight between features i and j, and W_ij increases with feature similarity.

- **Hypothesis**: Adding cortical-inspired lateral inhibition regularization to pretrained SAEs (without retraining) would reduce absorption by forcing absorbed features to reactivate. The inhibition strength needed to un-absorb a feature is proportional to its absorption depth (how much its decoder has been subsumed by the child).

- **Why not just a metaphor**: The mathematical form of activity-dependent lateral inhibition can be directly implemented as a regularization term on pretrained SAE activations. The key insight from neuroscience is that inhibition must be proportional and reciprocal—features inhibit each other proportionally to their activation and similarity.

- **Novelty estimate**: 6/10 — Some related work exists (e.g., orthogonality penalties in OrtSAE), but the specific application of cortical inhibition dynamics to pretrained SAE absorption mitigation has not been explored.

---

## Phase 3: Self-Critique

### Against Candidate A: Phase Transition Framework

- **Shallow analogy attack**: The analogy to phase transitions is not shallow—it is mathematically deep. Both SAE optimization and spin systems minimize energy functions. However, realistic SAEs with overcomplete dictionaries (d > 1) don't map cleanly onto conventional spin systems. The mapping requires approximations whose validity is unclear.

- **Scale mismatch attack**: Phase transitions are typically studied in thermodynamic limits (infinite systems). SAEs have finite dictionaries. Critical phenomena in finite systems show rounding effects, but the scaling laws might still hold qualitatively.

- **Prior transplant check**: Elhage et al. (2022) in "Toy Models of Superposition" used spin glass models for superposition, but did not specifically analyze absorption phase transitions. No prior work has characterized the critical exponents for absorption.

- **Testability attack**: The key test is whether absorption exhibits critical behavior: (1) does absorption probability change sharply at a critical sparsity? (2) do susceptibility (variance in absorption across runs) peak at the critical point? (3) do critical exponents follow universality classes? These are falsifiable predictions.

- **Verdict**: STRONG candidate. The phase transition framework provides a falsifiable, mathematically rigorous model of absorption. The main challenge is operationalizing the order parameter for realistic SAE setups.

### Against Candidate B: Evolutionary Feature Phylogeny

- **Shallow analogy attack**: The gene duplication analogy is structurally deep—the DDC model is specifically about how duplicated genes partition ancestral functions, exactly analogous to how child features partition parent features. However, gene evolution involves actual sequence changes over time, while SAE features evolve through optimization in a high-dimensional continuous space. The mapping from discrete sequence evolution to continuous embedding evolution requires justification.

- **Scale mismatch attack**: Gene duplication events occur over evolutionary timescales (millions of years). SAE features "evolve" during training (hours to days). The timescales are irrelevant for the structural analogy but the mutation rates and selection pressures differ fundamentally.

- **Prior transplant check**: No prior work has applied phylogenetic methods to SAE feature analysis. The closest work is analyzing feature geometry through clustering and similarity networks, but not phylogenetic tree reconstruction.

- **Testability attack**: The key test is whether absorbed features form ladder-like phylogenetic trees in embedding space. This can be measured by: (1) hierarchical clustering of feature decoders, (2) computing cophenetic distances, (3) comparing tree topology (ladderness score) for absorbed vs. independent features. A causal experiment: inject noise to create "neutral variants" of absorbed features and see if they cluster phylogenetically.

- **Verdict**: STRONG candidate. This is the most novel approach—it imports phylogenetic methods from evolutionary genomics to diagnose absorption in SAEs without any prior approach. The main uncertainty is whether continuous embedding space evolution follows the same ladder-like tree topology patterns as discrete sequence evolution.

### Against Candidate C: Causal Mediation Decomposition

- **Shallow analogy attack**: The causal mediation framing is the most structurally precise of all candidates—the isomorphism between causal mediation diagrams and feature absorption hierarchies is exact. However, causal mediation analysis assumes the causal graph is known, while in SAEs the feature hierarchy structure must be inferred.

- **Scale mismatch attack**: Causal mediation is well-established for macroscopic systems. In SAEs, the "treatment" (activating parent feature) and "mediator" (child feature) are high-dimensional continuous variables, not discrete categorical variables. The sequential ignorability assumption is unlikely to hold exactly in the SAE setting.

- **Prior transplant check**: The literature survey mentions "Causal mediation analysis frameworks could provide more principled metrics" but no work has actually implemented this. The gap is real but the challenge is significant.

- **Testability attack**: The key test is whether the mediation decomposition recovers ground-truth absorption rates in synthetic settings (where the causal graph is known, as in SynthSAEBench). If the NIE-based absorption metric correlates with ground-truth absorption in synthetic data, the framework is validated.

- **Verdict**: MODERATE candidate. The causal framework is conceptually elegant but identification assumptions are unlikely to hold exactly in practice. It may provide a useful approximation rather than ground-truth measurement.

### Against Candidate D: Cortical Inhibition Regularization

- **Shallow analogy attack**: The lateral inhibition analogy is widely used in ML (competition networks, WTA circuits). It's not a novel analogy. The specific application to pretrained SAE absorption mitigation is novel, but the core mechanism is well-known.

- **Scale mismatch attack**: Cortical inhibition operates through ion channel dynamics (ms timescale) with specific neural architecture constraints. The mathematical abstraction (activity-dependent inhibition) is transferable, but the specific parameters (inhibition strength, time constants) don't map.

- **Prior transplant check**: OrtSAE already uses orthogonality penalties inspired by similar intuitions. The specific cortical inspiration adds little beyond what's already in OrtSAE.

- **Testability attack**: The key test is whether adding lateral inhibition regularization to pretrained SAEs (via activation modification rather than retraining) measurably reduces absorption on standard benchmarks.

- **Verdict**: WEAK candidate as a novel mechanism. The inhibition idea is too close to existing approaches (OrtSAE orthogonality). However, it could be valuable as a specific implementation variant if combined with the evolutionary phylogeny approach (inhibition weighted by phylogenetic proximity).

---

## Phase 4: Refinement

### Dropped Candidates

- **Candidate D (Cortical Inhibition)**: WEAK — too close to existing OrtSAE approach. The cortical inspiration doesn't add sufficient novelty beyond orthogonality penalties.

### Strengthened Candidates

**Candidate B (Evolutionary Feature Phylogeny)** — Selected as front-runner:

1. **Formalization of structural correspondence**:
   - Define "feature phylogeny" as the hierarchical clustering tree of SAE decoder vectors
   - Compute cophenetic correlation for each latent: how high in the tree does it cluster with its parent?
   - Absorption depth = cophenetic distance where absorption occurred during training
   - The "DDC model" maps to: absorbed feature = leaf node whose internal branch length is much shorter than expected under neutral evolution

2. **Diagnostic experiment**:
   - Take a pretrained SAE (e.g., GemmaScope layer 10, 65k width)
   - Collect decoder vectors W_dec (d x n_latents)
   - Perform hierarchical clustering (UPGMA) on W_dec
   - Compute ladderness score: proportion of internal nodes that are "comb-like" (one long branch, many short branches) vs. "bush-like" (balanced)
   - Compare ladderness for features identified as absorbed (via ablation) vs. non-absorbed
   - **Prediction**: Absorbed features have significantly higher ladderness scores

3. **Absorption phylogeny index (API)**:
   - For each feature f, compute API = (median cophenetic distance to k nearest absorbed neighbors) / (median cophenetic distance to k nearest independent neighbors)
   - High API indicates feature is phylogenetically close to absorbed features
   - Use API as a training-free predictor of absorption vulnerability

**Candidate A (Phase Transition)** — Secondary candidate:

1. **Formalization**:
   - Define absorption order parameter: M = mean_{parent-child pairs} |cos(theta_parent, theta_child)|^2
   - Below critical sparsity: M ~ 0 (features orthogonal, no absorption)
   - Above critical sparsity: M increases continuously (absorption accumulates)
   - Critical exponent beta characterizes how fast absorption accumulates: M ~ (k - k_c)^beta for k > k_c

2. **Diagnostic experiment**:
   - Train SAEs at different L0 values (k = 32, 64, 128, 256, 512, 1024)
   - Measure absorption rate using Chanin et al. metric at each k
   - Fit critical point k_c and exponent beta
   - **Prediction**: Absorption vs. k follows universal scaling form with beta ~ 0.5 (mean field)

3. **Connection to phylogeny**:
   - Features that absorb early (at low k) are those with close phylogenetic neighbors
   - Phylogenetic proximity predicts critical point: more "relatives" = absorb at higher k

### Selected Front-Runner: Evolutionary Feature Phylogeny (Candidate B)

**Rationale**:
1. Genuinely novel—no prior SAE work applies phylogenetic methods
2. Training-free diagnostic—doesn't require retraining SAEs or ground-truth probes
3. Connects to existing literature on feature geometry and clustering
4. Generates testable predictions about absorption vulnerability
5. Potential for developing absorption-aware SAE training (phylogeny-regularized training)

---

## Phase 5: Final Proposal

### Title

**Feature Phylogeny: An Evolutionary Genomics Framework for Detecting and Quantifying Absorption in Sparse Autoencoders**

### Source Principle

Gene duplication and subfunctionalization (DDC model) from evolutionary genomics: When genes duplicate, they initially share all ancestral functions. Under selective pressure for specialization (analogous to sparsity pressure in SAEs), duplicated genes partition their ancestral functions through degenerative mutations. Eventually, one copy may lose all unique functions and become absorbed into the other. The phylogenetic tree structure of gene families—comb-like (ladder) vs. bushy—directly reveals absorption history: absorbed genes cluster at the tips of ladder-like trees with short external branches.

### Structural Correspondence

SAE features undergo an analogous process during dictionary learning on hierarchical data:

| Evolutionary Biology | SAE Feature Space |
|---------------------|------------------|
| Gene duplication event | Emergence of child feature during overcomplete dictionary learning |
| Ancestral gene function | Parent feature's representational role |
| Subfunctionalization | Child feature partitioning parent's activation patterns |
| DDC process (Duplication-Degeneration-Complementation) | Hierarchical feature specialization under sparsity |
| Gene absorption (total loss of unique function) | Feature absorption (parent's effect fully mediated by child) |
| Phylogenetic tree (ladder vs. bush) | Feature decoder clustering tree |
| dN/dS ratio (nonsynonymous/synonymous substitution) | Cophenetic distance / cosine similarity ratio |

### Hypothesis

**H1**: Absorbed features in pretrained SAEs form ladder-like phylogenetic structures in decoder space—short external branches immediately below internal nodes corresponding to absorbing features.

**H2**: The absorption phylogeny index (API), measuring phylogenetic proximity to known absorbed features, predicts which features are vulnerable to absorption before ablation testing.

**H3**: The distribution of cophenetic distances for absorbed features is significantly narrower than for independent features (ladder structure vs. star/bush structure).

**H4**: Features with more "phylogenetic relatives" (close neighbors in embedding space) absorb at lower sparsity levels, following a predictability function that could enable absorption-aware training.

### Method

**Phase 1: Feature Phylogeny Construction**

1. Load pretrained SAE (e.g., Gemma-2-2B SAE from SAELens, layer 10, 65k width)
2. Extract decoder weight matrix W_dec \in R^{d \times n_latents}
3. Compute pairwise cosine similarity matrix S \in R^{n_latents \times n_latents}
4. Perform hierarchical clustering (average linkage/UPGMA) on 1 - S to produce phylogenetic tree T
5. For each feature f, compute:
   - Cophenetic distance to each other feature in T
   - Ladderness score: proportion of T's internal nodes that are comb-like when f is the root
   - Mean cophenetic distance to k=10 nearest neighbors

**Phase 2: Absorption Validation**

1. Identify absorbed features using Chanin et al. ablation metric (or SAEBench projection method)
2. Validate H1: Compare ladderness scores for absorbed vs. non-absorbed features (Mann-Whitney U test)
3. Validate H2: Compute API for all features, check if high-API features are enriched in absorbed set (precision-recall analysis)
4. Validate H3: Compare cophenetic distance distributions

**Phase 3: Predictive Model**

1. Train classifier: API + ladderness + feature frequency -> absorption probability
2. Cross-validate on held-out features
3. Evaluate: does the classifier identify absorption-vulnerable features better than random?

### Diagnostic Experiment

**The Phylogenetic Fingerprint Test** (key test for H1-H3):

1. Take GPT-2-small SAE (12 layers, ~20k features via SAELens)
2. Construct phylogenetic tree T of all features using decoder cosine similarity
3. Identify absorbed features using ablation on a probe task (first-letter spelling)
4. For each absorbed feature f_absorbed:
   - Find its absorbing feature f_absorber in T (immediate parent in ladder)
   - Compute: cos(W_dec[:, f_absorbed], W_dec[:, f_absorber])
5. **Prediction**: Absorbed features have cos > 0.7 with their absorber, and form ladder structures (not bush/star). Non-absorbed features show no such pattern.

**The Causality Test for H4**:

1. Perform phylogenetic analysis on SAEs trained at different L0 values (k = 32, 64, 128)
2. Track how the phylogenetic tree changes as k increases
3. **Prediction**: Features absorb in order of phylogenetic proximity—the most "related" features absorb first, creating a cascade effect predicted by tree topology

### Experimental Plan

**Pilot experiment (10-15 minutes)**:
- Load GPT-2-small SAE from SAELens (no retraining)
- Implement phylogenetic tree construction on layer 6
- Compute ladderness scores for a random subset of 500 features
- Validate ladderness-absorbed correlation on 50 features manually annotated via Neuronpedia

**Main experiment (30-60 minutes per model)**:
- Full phylogenetic analysis on GPT-2 SAE (all 12 layers)
- Ablation-based absorption ground-truth on layer 6 (subset of 1000 features)
- H1-H3 validation with statistical tests
- API classifier training and evaluation

**Extension (60 minutes)**:
- Gemma-2-2B SAE (layers 5, 10, 15)
- Cross-architecture validation (Llama-3.2-1B if available)
- Ablation-free absorption prediction using API alone

### Risk Assessment

1. **Identification risk**: Phylogenetic proximity could correlate with absorption for reasons other than evolutionary dynamics (e.g., geometry alone). *Mitigation*: Control for cosine similarity—ladderness should predict absorption beyond what raw similarity predicts.

2. **Ground-truth risk**: Ablation-based absorption measurement is noisy (Chanin et al. note ~26 min per SAE). *Mitigation*: Use SAEBench projection method for cleaner ground-truth; validate on synthetic data (SynthSAEBench).

3. **Generalization risk**: Phylogenetic patterns might be specific to certain SAE training regimes. *Mitigation*: Test on multiple SAE architectures (TopK, JumpReLU, Gated).

4. **Metric risk**: Ladderness is a new metric without established baseline. *Mitigation*: Compare to established clustering metrics (Silhouette score, Dunn index).

5. **Confounding risk**: Feature frequency and phylogenetic proximity are correlated in ways that could confound the analysis. *Mitigation*: Partial correlation analysis controlling for feature activation frequency.

### Novelty Claim

**This is the first application of evolutionary genomics methods—specifically phylogenetic tree analysis, DDC subfunctionalization models, and evolutionary sequence analysis—to SAE feature dictionaries.**

The core innovation is treating the SAE feature space as an evolutionary system: features "duplicate" during overcomplete learning, compete under sparsity "selection pressure," and undergo "absorption" analogous to gene loss. This perspective generates novel diagnostic tools (ladderness score, API, phylogenetic absorption cascade) that have no precedent in the SAE literature.

**Why this matters**: Current absorption detection requires expensive ablation studies. The phylogenetic approach offers a training-free, geometry-based alternative that could enable rapid absorption screening across large SAE dictionaries.
