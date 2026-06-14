# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025 Oral)** -- Defines feature absorption, proves it occurs in toy models under hierarchical features, proposes the canonical absorption rate metric using integrated-gradients ablation. Validated on hundreds of SAEs across Gemma Scope, Llama 3.2, Qwen2. Absorption rate 15-35% across tested SAEs. Only studies first-letter spelling task; metric requires known probe directions.

2. **Li et al., 2024/2025. "The Geometry of Concepts: Sparse Autoencoder Feature Structure." arXiv:2410.19750 (Entropy 2025)** -- Reveals that SAE decoder weight vectors exhibit rich multi-scale geometric structure: "atomic" parallelogram crystals (analogy structures), "brain"-scale functional lobes (math/code clusters), and "galaxy"-scale anisotropic point clouds. Critically, semantically related features cluster spatially in decoder weight space. This geometric regularity is *exactly* what absorption exploits: parent-child features that are geometrically close can be merged by the sparsity objective.

3. **Wu et al., 2025. "Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders." arXiv:2502.15576** -- Proposes mutual information-based feature explanation objective for SAEs, addressing frequency bias in activation-based explanations. Directly relevant: MI between feature activations and concepts could serve as the foundation for an unsupervised absorption detection signal.

4. **Montanari & Wang, 2026. "Phase Transitions for Feature Learning in Neural Networks." arXiv:2602.01434** -- Proves sharp phase transitions in feature learning as a function of sample-to-dimension ratio. Establishes that feature recovery undergoes a discrete threshold transition, not a smooth degradation. This theoretical framework could be transplanted to predict absorption onset as a function of SAE width/sparsity.

5. **"Sparse Coding and Lateral Inhibition" (Yu et al., J. Neuroscience 2014) + "Relating Sparse and Predictive Coding to Divisive Normalization" (PLOS Comp Bio 2025)** -- The neuroscience of sparse coding reveals that biological systems solve the feature competition problem via lateral inhibition: active neurons suppress similar/overlapping neurons. Winner-take-all is the extreme case. SAE feature absorption is precisely the failure mode that lateral inhibition *prevents* in biological sparse coding.

6. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547 (ICML 2025)** -- Nested prefix losses create a natural feature hierarchy, reducing absorption rate to ~0.03 vs BatchTopK ~0.29. However, inner levels suffer from feature hedging, revealing a fundamental absorption-hedging tradeoff.

7. **Chanin et al., 2025. "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756** -- Shows absorption and hedging trade off against each other. Proposes balanced Matryoshka SAE with compound multiplier ~0.75. Critical insight: absorption and hedging are complementary failure modes that cannot both be minimized simultaneously under standard SAE objectives.

8. **Shu, 2024. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** -- Casts all SDL variants as piecewise biconvex optimization; explains absorption as convergence to spurious local minima; proposes feature anchoring. Only validated on synthetic benchmarks.

9. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** -- Reduces absorption ~70% via orthogonality penalty on decoder columns. Key mechanism: preventing the cosine similarity between decoder vectors from becoming too high, which is the geometric precondition for absorption.

10. **Karvonen et al., 2025. "SAEBench." arXiv:2503.09532 (ICML 2025)** -- 200+ SAEs, 8 metrics including absorption. Key finding: proxy metrics (CE loss, sparsity) do NOT predict practical performance. This means absorption cannot be predicted from standard training curves.

11. **Tian et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** -- Frames absorption as a special case of poor feature sensitivity. Shows many "interpretable" features have poor recall even when activation examples look monosemantic. Generalizes absorption beyond hierarchical features to any sensitivity failure.

12. **Narayanaswamy et al., 2026. "Improving Robustness In Sparse Autoencoders via Masked Regularization." arXiv:2604.06495** -- Very recent (April 2026). Token masking during training disrupts co-occurrence patterns, reducing absorption. Key insight: absorption is driven by training-time co-occurrence statistics, not just architectural choices.

### Landscape Summary

The SAE feature absorption problem sits at a critical juncture. The phenomenon is now well-characterized empirically (Chanin et al., 2024) and multiple architectural solutions exist (Matryoshka, OrtSAE, ATM, masked regularization), yet three fundamental gaps remain:

**First, absorption measurement is tethered to supervised probes.** The canonical absorption metric requires pre-specifying which features *should* have fired, which requires training LR probes on known tasks. This makes absorption measurement impossible in the general case where we do not know the ground-truth feature taxonomy in advance. No unsupervised alternative exists.

**Second, the mechanistic understanding is incomplete.** We know absorption occurs because sparsity optimization rewards merging parent features into child features when they co-occur (Chanin et al.'s toy model). But there is no quantitative theory predicting *how much* absorption will occur as a function of (a) the feature hierarchy's depth and branching factor, (b) parent-child co-occurrence frequency, (c) SAE width relative to the number of hierarchical features, and (d) the sparsity penalty strength. The phase transition framework from statistical physics (Montanari & Wang, 2026) suggests absorption onset may be a sharp threshold phenomenon, not a gradual degradation.

**Third, the connection between decoder geometry and absorption is unexploited.** Li et al. (2025) show decoder vectors have rich structure -- parallelogram crystals, functional lobes, power-law eigenvalue spectra. OrtSAE (2025) shows that enforcing orthogonality reduces absorption. But nobody has systematically studied whether the geometric properties of decoder vectors *predict* which features will be absorbed, or whether geometric signatures in the trained decoder can serve as unsupervised absorption detectors. The neuroscience literature on lateral inhibition in sparse coding suggests that the solution to feature competition is precisely about the geometry of feature representations -- neurons with overlapping receptive fields inhibit each other proportionally to their similarity.

## Phase 2: Initial Candidates

### Candidate A: Geometric Absorption Forensics -- Unsupervised Detection of Feature Absorption via Decoder Weight Topology

- **Hypothesis**: The geometric structure of SAE decoder weight vectors contains sufficient information to identify absorbed features without any supervised probes. Specifically, when feature B absorbs feature A, the decoder vector of B shifts toward the subspace spanned by A's "natural" direction, creating detectable geometric anomalies: (1) abnormally high cosine similarity between B's decoder vector and the mean direction of B's semantic cluster neighbors, (2) a specific pattern of asymmetric influence in the decoder Gram matrix, and (3) a discrepancy between the decoder vector's direction and the direction that maximizes mutual information between B's activation pattern and the input distribution.
- **Cross-domain insight**: In neuroscience, lateral inhibition circuits detect and resolve feature competition by computing pairwise similarity between neural responses. The Locally Competitive Algorithm (LCA) for sparse coding explicitly uses inner products between feature vectors as inhibition signals -- when two features have high inner product, the more active one suppresses the less active one. We invert this: rather than using geometry to *prevent* absorption, we use geometry to *detect* it post hoc. Additionally, from ecology, the competitive exclusion principle predicts that species (features) occupying overlapping niches cannot stably coexist -- the subordinate species (parent feature) is driven to extinction (absorption). The "niche overlap" between features is precisely their decoder cosine similarity.
- **Evidence for**: OrtSAE shows that reducing decoder cosine similarity reduces absorption by ~70%, establishing a causal link between decoder geometry and absorption. Li et al. (2025) show decoder vectors cluster by semantic similarity, meaning absorption-susceptible feature pairs (parent-child) are geometrically proximate. Tian et al. (2025) show poor feature sensitivity (generalized absorption) is widespread, meaning the problem scope is large enough to warrant an unsupervised detector.
- **Novelty estimate**: 8/10 -- The idea of using decoder geometry to *detect* (not prevent) absorption is new. Existing work uses geometry to *mitigate* absorption (OrtSAE) or to *study* SAE structure (Geometry of Concepts), but nobody has built an unsupervised absorption detector from decoder topology. The ecological competitive exclusion analogy and the LCA lateral inhibition connection are also novel framings.

### Candidate B: Absorption Phase Diagrams -- Mapping the Critical Boundaries of Feature Absorption in (Width, Sparsity, Hierarchy-Depth) Space

- **Hypothesis**: Feature absorption exhibits sharp phase transitions as a function of SAE configuration parameters. Specifically, for a given feature hierarchy of depth d and branching factor b, there exists a critical SAE width W_c(d, b, L0) below which absorption rate jumps discontinuously from near-zero to a regime-level value, analogous to percolation thresholds in statistical physics. The phase boundary can be predicted from the ratio of SAE capacity to the number of features in the hierarchy, and the transition sharpness increases with hierarchy depth.
- **Cross-domain insight**: Phase transitions in feature learning (Montanari & Wang, 2026) show that neural networks undergo sharp transitions in their ability to recover latent structure as the ratio of data to parameters crosses a critical threshold. The L2 regularization phase transition work (2025) shows that sparsity-inducing regularization creates discrete regime boundaries. Translating this to SAEs: absorption is the "disordered phase" where the SAE fails to maintain separate representations for hierarchically related features, and there should be a critical ratio of width-to-feature-count where the transition occurs.
- **Evidence for**: Chanin et al. find absorption rate 15-35% across different SAE widths (16k, 65k) but do not systematically map the width-absorption curve. SAEBench evaluates 3 widths (4k, 16k, 65k) and 6 sparsities, showing architecture-dependent absorption patterns. The absorption-hedging tradeoff (Chanin et al., 2025) suggests the system is operating near a critical point where small parameter changes shift between failure modes.
- **Novelty estimate**: 7/10 -- Phase transition analysis of neural network phenomena is well-established in theory, but applying it specifically to SAE feature absorption with controlled hierarchical features is new. The quantitative prediction of critical boundaries would be the key contribution. Risk: the transition may be smooth rather than sharp, which would weaken the physics analogy.

### Candidate C: Information-Theoretic Absorption Certificates -- Quantifying Feature Completeness via Conditional Mutual Information

- **Hypothesis**: For any SAE feature f_i, the conditional mutual information I(X; Y | f_i active, f_j inactive for all j != i) -- where X is the input token and Y is a downstream property -- provides a probe-free certificate of whether f_i has absorbed information that "belongs to" other features. If f_i is a clean, non-absorbing feature, then conditioning on f_i being active should make X and Y approximately independent for the specific Y that f_i represents. If f_i has absorbed feature f_j, then there exists a Y (the property f_j should track) for which substantial residual MI remains even after conditioning on f_i.
- **Cross-domain insight**: In information theory, the Data Processing Inequality states that processing cannot increase information. Feature absorption violates the spirit of this principle applied to SAE features: if feature f_i is supposed to represent concept C_i, then the activation of f_i should carry all the information about C_i that the original activation had. Absorption creates "information leaks" where concept C_j's information bleeds into f_i's activation pattern. This is analogous to channel crosstalk in communication theory, where signals in one channel contaminate another. The MI framework from Wu et al. (2025) can be adapted to detect this crosstalk.
- **Evidence for**: Wu et al. (2025) show MI-based feature explanations outperform activation-based explanations, indicating MI captures feature semantics better. Chanin et al.'s absorption metric is fundamentally an information-theoretic statement: absorbed features carry information about tokens they "shouldn't" represent. The unified SDL theory (arXiv:2512.05534) shows absorption is a convergence to spurious minima where feature identity is lost -- this identity loss should be detectable via MI analysis.
- **Novelty estimate**: 7/10 -- MI-based analysis of SAE features exists (Wu et al., 2025) but is used for explanation, not absorption detection. The conditional MI certificate idea is new but may be computationally expensive and require large sample sizes to estimate accurately. The channel crosstalk analogy is a clean framing but the technical challenge is in efficient estimation.

## Phase 3: Self-Critique

### Against Candidate A (Geometric Absorption Forensics)

- **Prior work attack**: OrtSAE already uses decoder cosine similarity to *mitigate* absorption, and Geometry of Concepts (Li et al.) already studies decoder structure. How different is "using geometry to detect" from "using geometry to prevent"? Searched for "decoder geometry absorption prediction" and "unsupervised absorption detection decoder" -- found no direct prior art. The key distinction is that OrtSAE uses geometry as a *training loss* to prevent absorption, while Candidate A proposes geometry as a *post-hoc diagnostic* to detect absorption in already-trained SAEs. This is genuinely different: it enables auditing any existing SAE (including the 400+ Gemma Scope SAEs) without retraining.
- **Methodological attack**: The hypothesis assumes absorption creates detectable geometric anomalies, but what if absorption simply shifts the decoder vector to a new position that is geometrically unremarkable? If the absorbed parent feature's direction is already close to the child feature's natural direction (which it often is, since parent and child are semantically related), the geometric shift may be too subtle to detect. Validation requires ground-truth absorption labels from the supervised metric, creating a chicken-and-egg problem for the initial development.
- **Theoretical attack**: The ecological competitive exclusion analogy is compelling but may be superficial. In ecology, exclusion happens because resources are finite. In SAEs, the "resource" is the sparsity budget -- but the sparsity budget is not finite in the same way (a wider SAE has more capacity). The analogy holds best when SAE width is fixed and insufficient, but breaks down when the SAE is wide enough to represent all features. However, Chanin et al. show absorption occurs even in 65k-width SAEs, suggesting the analogy may hold surprisingly well in practice.
- **Scalability attack**: Computing the full decoder Gram matrix is O(d * n^2) where d is the activation dimension and n is the dictionary size. For a 65k SAE, this is manageable. For 1M SAEs (Gemma Scope's largest), it's ~5e11 operations, which is expensive but feasible with GPU acceleration. The real scalability concern is whether the geometric signature is robust across different model architectures and training procedures.
- **Verdict**: **STRONG** -- The core idea of unsupervised absorption detection via decoder geometry fills Gap 7 (no probe-free absorption metric), is technically distinct from prior work, and has clear practical utility. The main risk is that geometric signatures may be too subtle for reliable detection, but this is an empirical question that can be tested. The training-free constraint of the project is perfectly matched.

### Against Candidate B (Absorption Phase Diagrams)

- **Prior work attack**: Searched for "phase transition sparse autoencoder feature recovery" and "critical width SAE scaling." Found Montanari & Wang (2026) on phase transitions in feature learning and the unified SDL theory (2512.05534) which characterizes spurious minima. The unified SDL theory already provides conditions under which absorption occurs (piecewise biconvex optimization structure), but does NOT map the phase boundary as a function of hierarchy parameters. SAEBench evaluates 3 widths x 6 sparsities but does not frame results as phase diagrams. So the specific contribution (quantitative phase diagrams parametrized by hierarchy properties) appears novel.
- **Methodological attack**: Testing the phase transition hypothesis requires training many SAEs with controlled feature hierarchies. The project constraint is training-free analysis, which limits experiments to existing pre-trained SAEs or synthetic settings. With existing SAEs, we cannot independently vary hierarchy depth/branching while holding other variables constant. We would need to either (a) find natural feature hierarchies of varying depth in real models and measure absorption at each depth, or (b) use SynthSAEBench's synthetic data generation, which does allow controlled hierarchy manipulation but is synthetic. The training-free constraint is a serious limitation for this idea.
- **Theoretical attack**: Phase transitions require a thermodynamic limit (large system size). In SAEs, the "system size" is the dictionary size, which ranges from 4k to 1M. It is unclear whether this range is sufficient to observe sharp transitions, or whether absorption is a smooth crossover that only sharpens in the limit of infinite dictionary size. Montanari & Wang's result is for two-layer networks in the proportional limit n,d -> infinity, which may not transfer cleanly to the SAE setting.
- **Scalability attack**: Mapping a phase diagram requires evaluating absorption at many (width, sparsity, hierarchy_depth) points. If each evaluation takes ~30 minutes (reasonable for the Chanin et al. metric), mapping a 10x10x5 grid requires 250 hours of compute. This exceeds the project's time budget unless we use efficient proxy metrics or restrict to a coarse grid.
- **Verdict**: **MODERATE** -- The idea is theoretically interesting and would be highly cited if the phase transition is sharp. However, the training-free constraint severely limits experimental design, the phase transition may not be sharp enough to be compelling, and the computational cost of mapping the full phase diagram is high. Could work as a supporting analysis within a larger paper, but may not stand alone.

### Against Candidate C (Information-Theoretic Absorption Certificates)

- **Prior work attack**: Wu et al. (2025) already use MI for SAE feature explanation. Chanin et al.'s absorption metric is implicitly information-theoretic (it measures whether features carry information about tokens they shouldn't). The feature sensitivity metric (Tian et al., 2025) also captures absorption as a special case of low sensitivity, which is related to MI. The novelty of Candidate C lies in the specific conditional MI formulation, but this may be seen as "just another way to measure the same thing" rather than a fundamentally new insight.
- **Methodological attack**: Conditional MI estimation in high dimensions is notoriously difficult. Estimating I(X; Y | f_i active) requires samples where f_i is active, partitioned by input tokens X and downstream properties Y. For rare features (which are precisely the ones most susceptible to absorption), the sample size may be too small for reliable MI estimation. Additionally, the "downstream property Y" must be chosen, which reintroduces a form of supervision -- unless we use all possible Ys, which is combinatorially intractable.
- **Theoretical attack**: The claim that "conditioning on f_i being active should make X and Y approximately independent if f_i is clean" is only true if f_i captures ALL the information about Y. But SAE features are known to be incomplete (Leask et al., 2025) -- even clean features do not capture all information about their concept. So low conditional MI between X and Y given f_i could indicate either (a) clean feature or (b) generally poor feature, not specifically absorption. The diagnostic specificity is questionable.
- **Scalability attack**: Computing MI for all features x all possible downstream properties is O(n * |Y|), where n is the dictionary size (up to 65k) and |Y| is the number of candidate properties. For a meaningful analysis, |Y| must be large, making total computation expensive.
- **Verdict**: **WEAK** -- The conditional MI formulation has fundamental specificity problems (cannot distinguish absorption from general feature incompleteness), the estimation is computationally intractable without supervision to select Y, and the novelty relative to existing MI-based work (Wu et al.) and sensitivity metrics (Tian et al.) is modest. The idea is intellectually clean but practically unworkable.

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C** (Information-Theoretic Absorption Certificates) dropped because: the conditional MI estimation is intractable without selecting downstream properties (reintroducing supervision), the diagnostic cannot distinguish absorption from general feature incompleteness, and the novelty over existing MI and sensitivity metrics is insufficient.

### Strengthened Ideas

**Candidate A (Geometric Absorption Forensics)** -- strengthened as follows:
1. **Addressing the subtlety concern**: Rather than relying on a single geometric signal, we propose a multi-signal ensemble: (a) decoder cosine similarity with semantic cluster neighbors, (b) activation correlation structure (features that should co-fire but don't), (c) "decoder residual" -- the component of a feature's decoder vector that is unexplained by its own activation pattern but correlated with another feature's pattern. The residual signal is strongest: if feature B has absorbed feature A, then B's decoder vector contains a component in A's direction that activates when A *should* fire but doesn't. This component is detectable by comparing decoder directions with activation covariance.
2. **Validation strategy**: We validate against the supervised Chanin et al. metric on the first-letter spelling task, establishing correlation between geometric signals and known absorption. Then we deploy the geometric detector on domains where supervised probes cannot be easily constructed (entity type hierarchies, sentiment-topic hierarchies), demonstrating the method's unique value.
3. **Computational efficiency**: The decoder Gram matrix computation can be chunked and cached. For 16k SAEs (most common in Gemma Scope), the matrix is 16k x 16k, easily fitting in GPU memory. We add efficient approximate methods using locality-sensitive hashing for 65k+ SAEs.

**Candidate B (Absorption Phase Diagrams)** -- incorporated as a supporting analysis within Candidate A:
1. Rather than mapping a full phase diagram (which requires retraining SAEs), we leverage the existing SAEBench collection (200+ SAEs across 3 widths and 6 sparsities) and Gemma Scope (400+ SAEs across multiple widths) to extract an empirical "absorption surface" as a function of width and sparsity, using our geometric detector as the measurement tool. This is training-free and computationally feasible.
2. We test for phase-transition-like behavior by fitting absorption rate vs. width curves to sigmoid/step functions and comparing against smooth alternatives, rather than proving a formal phase transition.

### Additional Evidence Found

- **Geometric structure is hierarchy-correlated**: Li et al. (2025) find that semantic categories form spatial clusters in decoder space, with hierarchically related concepts (e.g., "math" containing "algebra" and "geometry") forming nested clusters. This means absorption-susceptible pairs (parent-child) will have specific geometric signatures: high cosine similarity and cluster co-membership.
- **Activation covariance reveals hidden structure**: Leask et al. (2025, ICLR) show that meta-SAEs can decompose "atomic" features into sub-features. The activation patterns of absorbed features should reveal this decomposition: if feature B absorbed A, then B's activation pattern on A's tokens will have a distinct distribution from B's activation on B's own tokens. This activation pattern analysis is fully unsupervised.
- **LCA lateral inhibition is a direct algorithmic precedent**: The Locally Competitive Algorithm computes an inhibition matrix G_ij = <d_i, d_j> (inner product of decoder vectors) and uses it to resolve feature competition. This is *exactly* the decoder Gram matrix we propose to use for absorption detection. The LCA literature provides theoretical analysis of when competition resolves (features specialize) vs. when it fails (one feature dominates) -- this directly informs our phase boundary analysis.

### Selected Front-Runner

**Candidate A: Geometric Absorption Forensics** -- selected because:
1. It fills the most critical gap (Gap 7: no unsupervised absorption detection method) that would unlock the broadest downstream impact.
2. It is perfectly aligned with the project's training-free constraint -- the entire analysis works on existing pre-trained SAEs.
3. The cross-domain connections (ecological competitive exclusion, neuroscience lateral inhibition, decoder geometry) provide a rich theoretical narrative.
4. It naturally incorporates Candidate B's phase diagram analysis as a downstream application of the geometric detector.
5. The experimental plan requires only SAELens, TransformerLens, and pre-trained SAE weights -- all readily available.

## Phase 5: Final Proposal

### Title
Geometric Forensics of Feature Absorption: Unsupervised Detection via Decoder Weight Topology in Sparse Autoencoders

### Hypothesis
The geometric structure of SAE decoder weight vectors -- specifically, pairwise cosine similarities, cluster-level topological properties, and the discrepancy between decoder directions and activation-pattern principal components -- contains sufficient information to identify absorbed features with >0.7 AUROC relative to the supervised Chanin et al. absorption metric, without requiring any probe directions or task-specific labels. Furthermore, the geometric absorption signal generalizes to semantic domains (entity types, knowledge taxonomies) where supervised probes have not been constructed.

### Motivation
Feature absorption is the most fundamental failure mode of SAEs: it creates a false sense of interpretability by producing features that *appear* monosemantic but systematically fail to fire on a subset of their concept's instances. Chanin et al. (2024, NeurIPS 2025 Oral) showed absorption rates of 15-35% across all tested SAEs, and DeepMind's 2025 negative results attributed part of their observed 10-40% performance degradation to this phenomenon. However, the canonical absorption metric requires training supervised probes on known feature hierarchies, limiting its applicability to the first-letter spelling task and similar controlled settings. This creates a paradox: absorption is defined as a failure to detect features you expected to find, but you can only measure it when you already know what to expect.

We break this paradox by showing that absorption leaves detectable geometric traces in the SAE's learned decoder weights. Our insight draws from three converging fields:
1. **Neuroscience**: Biological sparse coding resolves feature competition via lateral inhibition, where neurons with overlapping receptive fields (high decoder cosine similarity) suppress each other. Absorption is precisely the failure of this resolution -- one feature suppresses another entirely rather than co-existing.
2. **Ecology**: The competitive exclusion principle states that two species occupying the same ecological niche cannot coexist -- the subordinate species goes extinct. In SAEs, the "niche" is the region of activation space a feature represents, and the "extinction" is absorption.
3. **Geometry**: Li et al. (2025) show SAE decoder vectors have rich multi-scale geometric structure, with hierarchically related features forming spatial clusters. OrtSAE (2025) shows that reducing decoder cosine similarity causally reduces absorption. We connect these observations: the geometric preconditions for absorption (high parent-child cosine similarity, asymmetric activation overlap) are detectable in the trained decoder without any supervised signal.

### Method

**Phase 1: Geometric Signal Construction**

For a trained SAE with decoder matrix W_dec in R^{n x d} (n features, d activation dimensions), we compute:

1. **Decoder Gram matrix** G_ij = cos(w_i, w_j): The pairwise cosine similarity matrix of all decoder vectors. From the LCA literature, G directly encodes the "inhibition structure" -- feature pairs with high G_ij are in competition. We identify features where G_ij > threshold but one feature's activation rate is much lower than expected given G_ij, signaling potential absorption.

2. **Activation-Decoder Misalignment (ADM)**: For each feature i, compute the principal component of the activation vectors x when feature i fires. Compare this empirical activation direction with the decoder vector w_i. A clean feature should have high alignment. An absorbing feature's decoder vector points partly toward the absorbed feature's direction, creating misalignment with its own activation pattern.

3. **Asymmetric Co-activation Deficit (ACD)**: For feature pairs (i, j) with high G_ij, measure the empirical co-activation rate P(i active AND j active) vs. the expected rate under independence P(i active) * P(j active). Absorption manifests as *unexpectedly low* co-activation for geometrically similar features: if j has absorbed i, then i will not fire when j fires, even though their decoder vectors are similar and they *should* co-fire on shared tokens.

4. **Residual Absorption Score (RAS)**: For each feature j, project its decoder vector onto the subspace of its k nearest neighbors in decoder space. The projection coefficient onto neighbor i, weighted by the activation-deficit between i and j, gives the absorption score: RAS(j, i) = G_ij * (expected_coactivation(i,j) - observed_coactivation(i,j)). High RAS indicates j may have absorbed i.

**Phase 2: Validation on Supervised Ground Truth**

Apply the geometric detector to Gemma Scope SAEs (16k, 65k widths) on the first-letter spelling task, where Chanin et al.'s supervised absorption metric provides ground-truth labels. Measure:
- AUROC of geometric absorption scores vs. supervised absorption labels
- Precision-recall curves at different detection thresholds
- Correlation between RAS and supervised absorption rate across letters/SAE configurations

**Phase 3: Cross-Domain Generalization**

Extend the geometric detector to domains where supervised probes do not exist:
- **Entity type hierarchy**: Train simple probes for "is_city" and "is_country" to establish partial ground truth, then use the geometric detector to find additional absorbed features that probes cannot identify.
- **Knowledge taxonomy** (city -> country -> continent): Construct hierarchical probe tasks with known parent-child structure. Measure whether absorption rate varies with hierarchy depth.
- **Polysemantic features**: For features identified as polysemantic by Neuronpedia labels, test whether polysemanticity correlates with high RAS scores (suggesting absorption is a mechanism behind some apparent polysemanticity).

**Phase 4: Absorption Landscape Mapping**

Using the geometric detector as a scalable measurement tool, map the "absorption landscape" across:
- All layers of Gemma 2 2B (using Gemma Scope SAEs at each layer)
- Multiple SAE widths (1k, 4k, 16k, 65k)
- Multiple sparsity levels (from SAEBench's 6 sparsity settings)

Test whether the absorption landscape shows phase-transition-like behavior: fit absorption-vs-width curves to parametric models (smooth vs. step-function) at each layer and sparsity level.

### Experimental Plan

**Experiment 1 (Pilot, ~15 min)**: Compute decoder Gram matrix for a single Gemma Scope 16k SAE (layer 12, residual stream). Identify feature pairs with G_ij > 0.3 and compute co-activation statistics on 10k tokens from OpenWebText. Verify that the distribution of G_ij values is non-trivial (not all near-zero) and that co-activation deficits exist.

**Experiment 2 (Main validation, ~45 min)**: Run full geometric absorption detection pipeline on Gemma-2-2B, layer 12, 16k SAE. Compare against Chanin et al. supervised absorption metric on first-letter task. Report AUROC, precision@k, and Spearman correlation. Baselines: (a) random scores, (b) decoder cosine similarity alone (without activation analysis), (c) activation frequency alone.

**Experiment 3 (Cross-domain, ~45 min)**: Apply geometric detector to Gemma-2-2B across layers 0, 6, 12, 18, 24 using 16k Gemma Scope SAEs. Construct entity-type hierarchy probes (city/country/continent). Measure absorption rate by hierarchy level and compare geometric detector's findings with probe-based detection.

**Experiment 4 (Absorption landscape, ~1 hour)**: Map geometric absorption score distribution across all available Gemma Scope SAE widths (16k, 65k) and multiple layers. Test for phase-transition signatures by fitting absorption-vs-width curves.

**Experiment 5 (GPT-2 replication, ~30 min)**: Replicate Experiment 2 on GPT-2 Small SAEs to test generalization across model families.

**Falsification criteria**:
- If geometric AUROC < 0.6 on the supervised validation task, the geometric signals do not reliably detect absorption.
- If ACD (co-activation deficit) does not correlate with supervised absorption labels (Spearman rho < 0.3), the core mechanism hypothesis is wrong.
- If absorption rate does not vary with hierarchy depth in the cross-domain experiments, the hierarchy-dependence of absorption may not generalize beyond the spelling task.

### Resource Estimate
- **Models**: Gemma-2-2B (via TransformerLens, ~5GB VRAM), GPT-2 Small (~0.5GB VRAM)
- **SAEs**: Gemma Scope pre-trained SAEs (16k, 65k widths; loaded via SAELens), GPT-2 SAEs from Neuronpedia
- **Data**: OpenWebText subset (10-50k tokens for activation statistics), first-letter spelling task from sae-spelling
- **Compute**: Single GPU with 24GB+ VRAM. Total experiment time: ~4 hours across 5 experiments. Each individual experiment fits within the 1-hour budget.
- **Libraries**: SAELens, TransformerLens, sae-spelling (for validation), numpy/scipy (for geometric computations), scikit-learn (for AUROC evaluation)
- **No training required**: All analysis uses pre-trained SAEs and models. Purely training-free.

### Risk Assessment

1. **Risk: Geometric signals may be too weak** -- If decoder cosine similarity and activation covariance do not reliably distinguish absorbed from non-absorbed features, the AUROC will be low. *Mitigation*: We propose four complementary signals (Gram matrix, ADM, ACD, RAS) and will evaluate each independently before combining. Even if individual signals are weak, their combination may be powerful. Additionally, we can fall back to reporting which geometric properties *correlate* with absorption (even if not sufficiently for detection), which is itself a novel finding.

2. **Risk: First-letter spelling task is too narrow for validation** -- The supervised ground truth only covers one type of feature hierarchy. If our detector works on spelling but fails on entity types or knowledge hierarchies, the generalization claim is weakened. *Mitigation*: We design cross-domain experiments with partial supervision (entity type probes) to triangulate the geometric detector's reliability. We also compare against Neuronpedia human-labeled feature descriptions as a soft ground truth.

3. **Risk: Computational cost of Gram matrix for large SAEs** -- For 65k SAEs, the Gram matrix has 2.1 billion entries. *Mitigation*: We use batch cosine similarity computation on GPU (16k: ~1 second; 65k: ~15 seconds on modern GPU). We also propose sparse approximations using locality-sensitive hashing for the 1M+ SAEs if needed, but the core experiments focus on 16k and 65k SAEs where exact computation is trivial.

### Novelty Claim

**What is new:**
1. **First unsupervised absorption detection method**: No existing approach can detect feature absorption without pre-specified probe directions. Our geometric detector enables absorption auditing of any SAE on any task, filling Gap 7 identified in the literature survey.

2. **First systematic cross-domain absorption study**: By enabling probe-free detection, we can characterize absorption in entity type hierarchies, knowledge taxonomies, and other semantic domains where the Chanin et al. metric cannot be applied. This addresses Gaps 2 and 6.

3. **Decoder geometry as absorption predictor**: While OrtSAE uses decoder geometry to *prevent* absorption (training-time intervention), we show decoder geometry *predicts* absorption (post-hoc diagnosis). This is a conceptually distinct contribution with different practical implications -- it enables auditing existing SAEs vs. requiring retraining.

4. **Ecological/neuroscience framing**: The connection between SAE feature absorption and ecological competitive exclusion / neuroscience lateral inhibition provides a new theoretical lens for understanding why absorption occurs and when it will be severe. This framing generates testable predictions (e.g., absorption severity scales with niche overlap / decoder cosine similarity).

**What is NOT claimed as new:** The absorption phenomenon itself (Chanin et al., 2024), the geometric structure of SAE features (Li et al., 2025), or the use of decoder cosine similarity for SAE improvement (OrtSAE, 2025). Our novelty lies in the specific combination: using geometry for unsupervised absorption *detection* and applying it to *cross-domain* absorption characterization.

**Evidence of novelty:** Systematic search for "unsupervised absorption detection," "decoder geometry absorption prediction," and "probe-free feature absorption metric" on arXiv, Google Scholar, and web search returned no prior art as of April 2026. The closest work is OrtSAE (training-time mitigation, not post-hoc detection) and Feature Sensitivity (Tian et al., 2025, which generalizes absorption but still requires concept-specific evaluation).
