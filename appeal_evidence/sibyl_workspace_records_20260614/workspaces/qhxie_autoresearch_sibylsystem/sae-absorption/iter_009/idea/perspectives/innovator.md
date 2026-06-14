# Innovator Perspective

## Phase 1: Literature Survey
### Key Papers Found
1. [Chanin et al., 2024. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507] --- Defines and quantifies feature absorption on the first-letter spelling task; proves absorption occurs in toy hierarchical settings; finds 15--35% absorption rates across all tested SAE architectures and models (Gemma Scope, Llama 3.2, Qwen2). The canonical reference and starting point for all absorption work.
2. [Karvonen et al., 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532] --- Standardized 8-metric evaluation including absorption; reveals proxy metrics (CE loss, sparsity) do not predict practical performance; shows TopK and JumpReLU significantly worsen absorption. Critical for benchmarking any new method.
3. [Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547] --- Nested prefix losses create natural hierarchy; absorption rate ~0.03 vs BatchTopK ~0.29. Best existing architectural mitigation. Key insight: training abstract latents *without* specific latents sometimes present forces them to fire on all instances.
4. [Chanin et al., 2025. Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756] --- Identifies hedging as complementary failure to absorption; shows the two trade off against each other in Matryoshka SAEs. Establishes that solving absorption alone is insufficient.
5. [Tian et al., 2025. Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717] --- Frames absorption as a special case of poor feature sensitivity; many seemingly monosemantic features have poor sensitivity. Generalizes the absorption concept beyond the first-letter task.
6. [Leask et al., 2025. Sparse Autoencoders Do Not Find Canonical Units of Analysis. arXiv:2502.04878] --- Meta-SAEs decompose features into sub-features; SAE features are neither complete nor atomic; smaller SAEs miss information captured by larger ones. Challenges the fundamental unit-of-analysis assumption.
7. [Fel et al., 2025. Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction. ICML 2025] --- Constraining decoder atoms to the data's convex hull dramatically improves stability (cosine stability ~0.5 for TopK vs much higher for A-SAE). Addresses feature inconsistency but not absorption directly. Key cross-domain insight: geometric anchoring provides identifiability.
8. [Tang et al., 2024. A Unified Theory of Sparse Dictionary Learning. arXiv:2512.05534] --- Piecewise biconvex optimization framework; principled explanation for absorption and dead neurons; proposes feature anchoring but only validates on synthetics.
9. [Chanin & Garriga-Alonso, 2025. Sparse but Wrong: Incorrect L0 Leads to Incorrect Features. arXiv:2508.16560] --- Most open-source SAEs have L0 that is too low; incorrect L0 causes feature mixing that is confounded with absorption.
10. [Lotka-Volterra competition models / Competitive Exclusion Principle (Gause, 1934)] --- Two species competing for identical resources cannot coexist; niche partitioning allows coexistence by reducing interspecific competition below intraspecific competition. Cross-domain analogy: SAE features competing for activation capacity under sparsity constraints parallel ecological species competing for niche space.
11. [Westphal et al., 2025. A Generalized Information Bottleneck Theory of Deep Learning. arXiv:2509.26327] --- Reformulates IB through synergy; demonstrates universal compression phases across architectures. Relevant for understanding hierarchical information loss in SAE encoding.
12. [Hierarchical Disentangled Information framework (Nature Scientific Reports, 2025)] --- Introduces Hierarchical Disentanglement Score (HDS) combining completeness, reconstruction, and hierarchy organization metrics. Directly relevant to measuring hierarchical feature quality in SAEs.

### Landscape Summary

Feature absorption is now well-established as a fundamental failure mode of sparse autoencoders, present across every architecture tested (L1, TopK, BatchTopK, JumpReLU) and every model (Gemma 2, GPT-2, Llama, Qwen2, Pythia). The canonical measurement relies on a narrow first-letter spelling task with known probe directions, leaving open whether absorption patterns generalize to richer semantic hierarchies. Several architectural mitigations exist (Matryoshka SAE, OrtSAE, ATM, masked regularization), but they have never been compared under matched conditions, and some trade absorption for hedging or inconsistency.

The field is at an inflection point: DeepMind has deprioritized SAEs, Anthropic continues to use them for circuit tracing, and the community is searching for principled theoretical frameworks to guide SAE design. The most glaring gap is the absence of an unsupervised, probe-free method for detecting and quantifying absorption --- the current metric requires knowing which features to look for in advance, which severely limits the scope of absorption analysis.

A deeper theoretical gap is the lack of a quantitative model that predicts absorption severity as a function of feature hierarchy statistics and SAE hyperparameters. Without such a model, practitioners cannot make principled decisions about SAE width, sparsity, or architecture to minimize absorption for their use case.

## Phase 2: Initial Candidates
### Candidate A: Information-Theoretic Absorption Completeness (ITAC) --- Unsupervised Absorption Detection via Decoder Geometry

- **Hypothesis**: Feature absorption can be detected and quantified without supervised probe directions by analyzing the information-theoretic completeness of SAE decoder weight geometry: specifically, if a feature direction d_parent in the model's representation space is "absorbed" by a child feature d_child, then the decoder column of d_child will contain a component aligned with d_parent that is not independently represented by any dedicated latent, creating a measurable "information completeness gap" detectable from decoder cosine similarity structure alone.
- **Cross-domain insight**: From the disentanglement literature (VAEs), the Mutual Information Gap (MIG) metric detects when one latent variable "captures" information that should belong to another, without requiring ground-truth labels for the captured variable. I transplant this principle to SAE decoders: instead of measuring MI between latents and ground-truth factors, I measure the "representation completeness" of the decoder matrix by detecting directions in activation space that are partially represented by multiple decoder columns but fully owned by none.
- **Evidence for**: (1) Leask et al. show meta-SAEs decompose "atomic" features into sub-features, proving decoder columns contain compositional structure. (2) Chanin et al. show absorbing latents have cosine similarity >0.025 with the absorbed probe direction, confirming geometric signatures exist. (3) The "Projecting Assumptions" paper (2025) shows decoder cosine similarity reveals cross-concept mixing. (4) SynthSAEBench provides ground truth to validate unsupervised metrics against known absorption.
- **Novelty estimate**: 8/10 --- No existing work attempts unsupervised absorption detection. The MIG-to-decoder transplant is conceptually novel. However, analyzing decoder geometry is not new per se (OrtSAE, Geometry of Concepts paper both use it).

### Candidate B: Ecological Niche Theory of Feature Absorption --- Competitive Exclusion and Partitioning in SAE Feature Space

- **Hypothesis**: Feature absorption in SAEs is formally analogous to competitive exclusion in ecology, where two species (features) competing for the same niche (activation capacity under sparsity constraint) cannot coexist, and the specialist (child/token-level feature) excludes the generalist (parent/abstract feature) because it achieves higher "fitness" (lower reconstruction loss per unit of L0 budget). The severity of absorption can be predicted by a modified Lotka-Volterra competition coefficient derived from feature co-occurrence statistics and decoder alignment.
- **Cross-domain insight**: Gause's competitive exclusion principle states that two species with identical niches cannot coexist; species persist by niche partitioning (using different resources or occupying different spatial/temporal niches). In SAEs, parent and child features compete for the same "niche" (activating on overlapping token sets under a shared sparsity budget). Absorption is competitive exclusion. Matryoshka SAEs work because they create temporal niche partitioning (training abstract features sometimes without specific features present). Wider SAEs work because they increase "habitat size" (more niche space).
- **Evidence for**: (1) Chanin et al. show absorption saves +1 L0 per parent-child pair --- directly analogous to the energetic efficiency of competitive exclusion. (2) Matryoshka SAEs reduce absorption by forcing abstract features to sometimes train alone --- exactly like temporal niche partitioning in ecology. (3) The absorption-hedging tradeoff (Chanin et al., 2025) parallels the ecological tradeoff between competitive exclusion and character displacement. (4) Lotka-Volterra models have well-studied conditions for coexistence vs. exclusion, providing a mathematical framework.
- **Novelty estimate**: 9/10 --- No one has connected ecological competition theory to SAE feature dynamics. The cross-domain mapping is deep, not superficial: it yields testable quantitative predictions about when absorption will and won't occur.

### Candidate C: Cross-Domain Absorption Taxonomy --- Systematic Characterization Beyond First-Letter Spelling

- **Hypothesis**: Feature absorption severity varies systematically with the structural properties of the feature hierarchy (depth, branching factor, frequency imbalance between parent and child), and the first-letter spelling task underestimates absorption in knowledge-rich hierarchies (e.g., entity type > specific entity, country > city) where hierarchy depth is greater and frequency imbalance is more extreme.
- **Cross-domain insight**: From hierarchical topic modeling (LDA with hierarchical extensions), it is well-known that general topics tend to be "absorbed" by specific subtopics when the generative model favors sparsity. The same structural dynamics apply to SAE features, but the severity depends on hierarchy geometry. Multi-label classification with label implication also exhibits this pattern (a "mammal" label is subsumed by "dog" when sparsity is enforced).
- **Evidence for**: (1) Chanin et al. show absorption is universal across architectures but only measure it on first-letter (2-level hierarchy). (2) The ICLR 2025 entity knowledge paper shows SAE latents can encode entity-type knowledge with cross-type generalization, implying knowledge hierarchies are represented. (3) SynthSAEBench shows hierarchy and Zipfian distributions create realistic absorption-like failures. (4) No existing work systematically varies hierarchy structure properties and measures their effect on absorption.
- **Novelty estimate**: 7/10 --- Extending absorption measurement to new tasks is somewhat incremental, but the systematic taxonomy linking hierarchy geometry to absorption severity would be a genuine contribution. The risk is that it becomes "more of the same" empirically.

## Phase 3: Self-Critique
### Against Candidate A (ITAC)
- **Prior work attack**: Searched for "unsupervised absorption detection decoder geometry" and "probe-free feature completeness SAE". No paper proposes this specific method. OrtSAE uses decoder cosine similarity for orthogonality regularization but not for absorption detection. The Geometry of Concepts paper analyzes decoder structure at multiple scales but does not connect it to absorption. The Projecting Assumptions paper (2025) identifies cross-concept mixing in decoder space but does not formalize it as an absorption metric. **Verdict: genuinely novel as a detection method.**
- **Methodological attack**: The main risk is that decoder geometry alone may be insufficient to distinguish absorption from legitimate feature composition. Two decoder columns can have high cosine similarity because (a) one is absorbing the other (bad) or (b) they represent genuinely correlated concepts (acceptable). Distinguishing these cases may require activation statistics, not just weight geometry. Mitigation: combine decoder geometry with activation co-occurrence analysis. Also, validation against known absorption (first-letter task + SynthSAEBench) is straightforward.
- **Theoretical attack**: The MIG analogy has limits. In VAEs, the MI is computed between latents and ground-truth factors, both of which are accessible. In SAEs, the "ground-truth factors" are unknown by assumption. The metric must work purely from the decoder and activations, which is a strictly harder problem. However, the completeness analysis does not require knowing specific features --- it identifies "orphaned directions" that should be represented but aren't.
- **Scalability attack**: The naive approach of computing all pairwise decoder cosine similarities is O(d^2) where d is the dictionary size, which could be 65k or larger. For d=65k, this is ~2B pairs, which is computationally expensive but feasible with batched GPU operations. Hierarchical clustering or random projection approximations could reduce this further.
- **Verdict**: STRONG --- the core idea is novel, the validation path is clear, and the methodological weakness (confusing absorption with composition) can be mitigated with activation statistics.

### Against Candidate B (Ecological Niche Theory)
- **Prior work attack**: Searched for "competitive exclusion neural network features" and "Lotka-Volterra sparse coding features". Found no direct precedent in the SAE/interpretability literature. The closest is Anthropic's informal notion of feature "competition" for activation capacity, but this is not formalized using ecological models. The compressed sensing "hierarchical sparsity" literature studies multi-scale dictionaries but does not use ecological metaphors. **Verdict: genuinely novel cross-domain connection.**
- **Methodological attack**: The main risk is that the analogy may be too loose to yield quantitative predictions. Lotka-Volterra models assume continuous population dynamics with specific functional forms (linear competition terms, logistic growth). SAE feature dynamics during training are discrete (gradient steps) and highly nonlinear (ReLU/TopK gating). The competition coefficient between features may not map cleanly to a single scalar. Mitigation: focus on the steady-state predictions (which features survive) rather than the dynamics (how they get there). The coexistence condition (interspecific competition < intraspecific competition) can be operationalized using decoder alignment and activation overlap statistics.
- **Theoretical attack**: The ecological analogy assumes features compete pairwise, but SAE optimization is a global problem where all features interact simultaneously. Lotka-Volterra can be extended to N-species (generalized competition), but the analysis becomes complex. Also, features don't literally "die" in the way species go extinct --- a feature can have zero activation without being removed from the dictionary. However, the functional outcome (feature never fires) is analogous to local extinction.
- **Scalability attack**: Computing pairwise competition coefficients for d features requires O(d^2) work, similar to ITAC. Predictions need validation on controlled benchmarks before scaling to large SAEs.
- **Verdict**: MODERATE --- the analogy is deep and yields interesting predictions, but the gap between ecological dynamics and SAE optimization dynamics may limit the quantitative precision. The ecological framing is best used for intuitive understanding and qualitative predictions, with the mathematical predictions requiring careful calibration.

### Against Candidate C (Cross-Domain Taxonomy)
- **Prior work attack**: Searched for "absorption semantic hierarchy knowledge entities SAE". The ICLR 2025 entity knowledge paper shows SAE latents encode entity types with cross-type generalization, but does not measure absorption. Muchane et al. (2025) H-SAE paper tests absorption on first-letter task and observes cross-lingual features, but does not systematically vary hierarchy properties. No paper systematically measures absorption across multiple hierarchy types. **Verdict: novel systematic study, though each individual task may have been partially explored.**
- **Methodological attack**: Designing probe tasks for richer hierarchies is harder than for first-letter spelling. For entity type hierarchies (city > country), one needs reliable probes at each hierarchy level. Probe quality directly affects absorption measurement quality. If probes are imperfect, observed "absorption" may partly reflect probe error rather than SAE failure.
- **Theoretical attack**: The hypothesis that "knowledge-rich hierarchies show worse absorption" is plausible but could be wrong. First-letter hierarchies have extreme frequency imbalance (26 parent classes vs thousands of child tokens), which may actually be worse than flatter knowledge hierarchies. The direction of the effect is uncertain.
- **Scalability attack**: Each new hierarchy task requires designing prompts, training probes, and running the full absorption pipeline. Scaling to many hierarchy types is time-consuming but straightforward.
- **Verdict**: MODERATE --- solid empirical contribution but lacks the conceptual novelty of the other two candidates. Risk of being seen as an incremental extension of Chanin et al.

## Phase 4: Refinement
### Dropped Ideas
- **Candidate C (Cross-Domain Taxonomy)** --- Not dropped outright but deprioritized. While valuable, it is more incremental than the other two. The key insight (different hierarchies yield different absorption patterns) should be incorporated as a validation component of whichever idea is selected, rather than as the primary contribution.

### Strengthened Ideas
- **Candidate A (ITAC) strengthened**: To address the confusion between absorption and legitimate composition, I add a two-stage detection pipeline: (1) identify candidate "orphaned directions" via decoder geometry, then (2) validate using activation conditional independence tests --- if a parent direction's activation is conditionally independent of the input given the child's activation, absorption has occurred. This connects to the feature sensitivity metric (Tian et al., 2025) and provides a principled statistical test.
- **Candidate B (Ecological Niche Theory) strengthened**: Rather than forcing exact Lotka-Volterra dynamics (which may not map cleanly), I reformulate the ecological analogy as a "resource competition" framework where the shared resource is L0 budget and the competition coefficient is quantified by decoder alignment * activation overlap. This yields cleaner predictions about the coexistence condition without requiring the analogy to hold at the dynamical level. The key prediction becomes: two features will coexist (no absorption) if and only if their "niche overlap" (decoder alignment * activation co-occurrence frequency) is below a threshold that depends on the SAE's effective sparsity budget.

### Additional Evidence Found
- The Hierarchical Disentanglement Score (HDS) framework from the VAE literature provides a ready-made template for combining completeness, disentanglement, and hierarchy organization into a single evaluation suite --- directly applicable to ITAC's metric design.
- The Archetypal SAE (Fel et al., ICML 2025) demonstrates that geometric anchoring of decoder atoms dramatically improves stability. This suggests that decoder geometry contains rich information about feature quality, supporting ITAC's premise.
- SynthSAEBench provides the ideal validation environment: ground-truth hierarchical features with known parent-child relationships, allowing direct comparison of ITAC's unsupervised detection against the known absorption ground truth.

### Selected Front-Runner

**Candidate A (ITAC) is the front-runner**, with elements of Candidate B incorporated as the theoretical motivation. The reasoning:

1. **Highest impact**: An unsupervised absorption detection method addresses Gap 7 (no metric without known probe directions), which is one of the most practically limiting gaps in the field. Currently, absorption can only be studied on tasks where researchers know the feature hierarchy in advance.
2. **Clear validation path**: SynthSAEBench ground truth + first-letter spelling task (canonical supervised metric) provide direct benchmarks. If ITAC detects the same absorption that supervised probes detect on the first-letter task, it validates the method. Then ITAC can be applied to settings where no probes exist.
3. **Ecological framework provides intuition**: The niche competition framing from Candidate B provides the theoretical motivation for *why* decoder geometry should reveal absorption (features that absorb others will have decoder columns aligned with the absorbed direction).
4. **Feasible with training-free analysis**: ITAC operates on pre-trained SAE decoder weights and cached activations, fully compatible with the project's training-free constraint.
5. **Extensible**: Once validated, ITAC can be applied to characterize absorption across domains (incorporating Candidate C's goals) without requiring new probes for each domain.

## Phase 5: Final Proposal
### Title
Information-Theoretic Absorption Completeness: Unsupervised Detection and Quantification of Feature Absorption in Sparse Autoencoders

### Hypothesis
Feature absorption in sparse autoencoders creates a measurable "completeness gap" in the decoder weight geometry: directions in the model's activation space that should be independently represented by dedicated SAE latents are instead partially encoded as components of other latents' decoder vectors, and this gap can be detected and quantified without supervised probe directions by analyzing the spectral and cosine similarity structure of the decoder matrix combined with activation conditional independence statistics.

Precisely: Define the *absorption completeness score* (ACS) for an SAE as the fraction of principal directions in the activation space that are "owned" by at least one dedicated decoder column (cosine similarity > threshold) versus being representable only as a combination of other decoder columns. An SAE with high absorption will have low ACS because absorbed directions have no dedicated owner. We hypothesize that ACS correlates negatively with the supervised absorption rate (Chanin et al. metric) at r > 0.7 across SAEs of varying width and architecture.

### Motivation
The current absorption metric (Chanin et al., 2024) requires training logistic regression probes on known features to identify which features "should have" fired but didn't. This creates a Catch-22: to study absorption, you must already know which features exist. This limits absorption analysis to narrow controlled tasks (first-letter spelling) and prevents characterizing absorption in the vast majority of SAE features where no ground truth is available. An unsupervised metric would:
1. Enable absorption analysis at scale (all 50M+ latents on Neuronpedia, not just 26 first-letter probes)
2. Identify previously unknown cases of absorption in safety-relevant, knowledge, or reasoning features
3. Provide a training signal for absorption-aware SAE optimization without requiring task-specific probes
4. Close Gap 7 identified in the literature survey

### Method
The ITAC framework has three components:

**Component 1: Decoder Completeness Analysis**
For a trained SAE with decoder matrix W_dec (shape: d_model x d_sae), compute the "coverage" of the model's activation space by dedicated decoder columns:
1. Compute the principal directions of the activation distribution (via PCA or using the model's layer-norm eigenspectrum as proxy) to get directions V = {v_1, ..., v_k} that represent the "ground truth" directions the SAE should capture
2. For each principal direction v_i, find the best-matching decoder column: max_j cos(v_i, w_j)
3. Identify "well-covered" directions (max cosine > tau_own) and "orphaned" directions (max cosine < tau_own but reconstructable as a linear combination of multiple decoder columns)
4. Orphaned directions are absorption candidates: information that should be independently represented but is instead distributed across multiple latents

**Component 2: Activation Conditional Independence Test**
For each absorption candidate pair (orphaned direction v, candidate absorber w_j):
1. Collect activation data on a diverse text corpus
2. Test whether the variation explained by v is conditionally independent of the input given the activation of latent j
3. If the absorber's activation fully mediates the relationship between v and the input, absorption is confirmed
4. Quantify absorption severity as the fraction of v's variance explained by the absorber

**Component 3: ITAC Score Aggregation**
1. Compute the Absorption Completeness Score (ACS) = fraction of principal directions that are "owned" (not orphaned)
2. Compute the Absorption Severity Distribution = histogram of absorption severity scores across all detected absorption pairs
3. Compute the Hierarchy-Weighted Absorption Score = weight each absorption instance by the estimated hierarchy depth (parent-child distance in decoder space)

**Validation Protocol:**
- **Stage 1 (Ground truth)**: Validate on SynthSAEBench where true feature hierarchies and absorption instances are known. Measure correlation of ITAC-detected absorption with ground-truth absorption.
- **Stage 2 (Supervised benchmark)**: Validate on first-letter spelling task (Chanin et al.) by comparing ITAC scores with supervised absorption rates across 50+ Gemma Scope SAEs.
- **Stage 3 (Cross-task generalization)**: Apply ITAC to detect absorption in entity-type hierarchies, syntactic hierarchies, and sentiment hierarchies where no prior absorption measurement exists. Validate with post-hoc probe training.

### Experimental Plan
**Baselines:**
- Supervised absorption rate (Chanin et al., 2024) on first-letter spelling task
- SAEBench absorption metric (modified version)
- Random baseline: shuffled decoder columns with same architecture

**SAEs to evaluate:**
- Gemma Scope JumpReLU SAEs (Gemma 2 2B, layers 0-25, widths 16k and 65k)
- SAEBench SAEs (Pythia-160M, Gemma-2-2B; 7 architectures x 3 widths x 6 sparsities)
- Matryoshka SAEs (from Gemma Scope 2) as low-absorption control

**Metrics:**
1. ITAC-supervised correlation: Pearson/Spearman correlation between ITAC ACS and supervised absorption rate across SAEs
2. ITAC detection precision/recall: Using SynthSAEBench ground truth, measure how accurately ITAC identifies known absorption instances
3. Cross-task consistency: Does ITAC detect similar absorption patterns on the first-letter task and entity-type tasks for the same SAE?
4. Architecture ranking preservation: Does ITAC rank SAE architectures in the same order as the supervised metric (Matryoshka < OrtSAE < BatchTopK < JumpReLU)?

**What would falsify the hypothesis:**
- If ITAC ACS does not correlate with supervised absorption rate (r < 0.4)
- If decoder geometry alone cannot distinguish absorption from legitimate feature composition (precision < 0.6)
- If absorption patterns detected by ITAC are inconsistent across text corpora (suggesting the metric is noise-dependent)

### Resource Estimate
- **Models**: Gemma 2 2B (loaded via TransformerLens, ~5GB GPU memory)
- **SAEs**: Pre-trained Gemma Scope SAEs (loaded via SAELens, ~1-2GB each)
- **Computation per SAE**: Decoder PCA (~2 min), cosine similarity matrix (~5 min for 16k width), activation collection on 10k tokens (~10 min), conditional independence tests (~15 min). Total: ~30 min per SAE.
- **Total experiment time**: 50 SAEs x 30 min = ~25 hours. Can be parallelized across GPUs.
- **Pilot experiment**: Validate ITAC on 3 Gemma Scope 16k SAEs (layers 5, 12, 20) against supervised absorption metric. ~1.5 hours.
- **Hardware**: Single A100/H100 GPU sufficient. Training-free --- only inference and matrix operations.

### Risk Assessment
1. **Risk: Decoder geometry insufficient for absorption detection.** The decoder matrix alone may not contain enough information to distinguish absorption from composition. **Mitigation**: Component 2 (conditional independence test) uses activation statistics to disambiguate. If decoder-only analysis fails, the activation-based component may still succeed.
2. **Risk: Computational cost of pairwise analysis at large dictionary sizes.** For d=65k, the full pairwise cosine similarity matrix is 65k x 65k = 4.2B entries. **Mitigation**: Use approximate nearest-neighbor search (FAISS) to identify candidate pairs efficiently, then compute exact tests only on candidates. Alternatively, focus on 16k-width SAEs for the main analysis.
3. **Risk: The metric may detect feature splitting (benign) rather than absorption (harmful).** Feature splitting and absorption have different geometric signatures (splitting: multiple decoder columns aligned with the same direction; absorption: one decoder column partially aligned with an "orphaned" direction). **Mitigation**: The conditional independence test in Component 2 distinguishes these: splitting latents will be conditionally dependent (both carry independent information), while absorption creates conditional independence.

### Novelty Claim
1. **First unsupervised absorption detection method**: No existing work detects feature absorption without pre-specified probe directions. ITAC closes Gap 7 identified in the literature.
2. **Novel cross-domain transplant**: The MIG/completeness framework from the VAE disentanglement literature has never been adapted to SAE decoder analysis. The ecological niche competition framing for feature absorption is also original.
3. **Decoder-activation joint analysis**: Existing decoder geometry analyses (OrtSAE, Geometry of Concepts) study structure in isolation. ITAC combines decoder geometry with activation conditional independence for a more complete picture.
4. **Validated on multiple ground-truth sources**: Unlike the supervised metric (validated only on first-letter spelling), ITAC will be validated on SynthSAEBench (synthetic ground truth), first-letter (supervised ground truth), and entity-type hierarchies (post-hoc validation).

**Evidence this hasn't been done**: Extensive search across arXiv, Google Scholar, and the web for "unsupervised absorption detection", "probe-free absorption metric", "decoder geometry absorption", and "information-theoretic absorption completeness" returned no matches. The closest works are: (a) Tian et al.'s feature sensitivity metric, which generalizes absorption but still requires knowing which features to test; (b) OrtSAE's orthogonality analysis, which uses decoder geometry for training but not for absorption detection; (c) the Projecting Assumptions paper, which identifies decoder mixing but does not connect it to absorption quantification.
