# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025)** -- Defines and quantifies feature absorption; proves it arises from hierarchical features under sparsity optimization; proposes the canonical absorption rate metric on first-letter spelling task. Foundational paper that establishes the phenomenon but only measures it on one task and proposes no mitigation.

2. **Park et al., 2025. "The Geometry of Concepts: Sparse Autoencoder Feature Structure." arXiv:2410.19750** -- Reveals multi-scale geometric structure in SAE decoder weight space (crystals, lobes, galaxies); validates using mutual information between cosine-similarity-based clustering and co-occurrence-based clustering at 954 sigma significance. Critical insight: decoder geometry encodes functional relationships, suggesting absorption could leave geometric signatures.

3. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547 (ICML 2025)** -- Nested dictionary sizes create natural feature hierarchy; achieves absorption rate ~0.03 vs BatchTopK ~0.29. Best existing mitigation, but inner levels suffer from feature hedging, revealing a fundamental absorption-hedging tradeoff.

4. **Chanin et al., 2025. "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756** -- Identifies feature hedging as complementary failure mode; shows Matryoshka SAEs trade absorption for hedging; proposes balanced Matryoshka with compound multiplier ~0.75. Key insight: absorption and hedging are coupled failure modes, not independent.

5. **Shu et al., 2024. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** -- Casts all SDL methods as piecewise biconvex optimization; provides theoretical explanation for absorption as spurious minima; proposes feature anchoring. But anchoring only validated on synthetic benchmarks.

6. **Leask et al., 2025. "Sparse Autoencoders Do Not Find Canonical Units of Analysis." arXiv:2502.04878 (ICLR 2025)** -- Meta-SAEs decompose features into sub-features; larger SAEs learn novel latents missed by smaller ones. Demonstrates non-canonicality -- absorbed features may be detectable as "hidden sub-features" within absorbing latents.

7. **Tian et al., 2025. "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717** -- Frames absorption as a special case of poor feature sensitivity; shows many interpretable features have systematic blind spots even when activation examples appear monosemantic. Generalizes absorption beyond hierarchy.

8. **Karznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** -- Orthogonality penalty on decoder reduces absorption ~70% vs BatchTopK. Mechanism: prevents absorbing latent from aligning its decoder direction with parent feature direction.

9. **Prashant et al., 2024. "Differentiable Causal Discovery for Latent Hierarchical Causal Models." arXiv:2411.19556** -- Learns latent hierarchical causal structure from observations using rank constraints and conditional independence. Key cross-domain insight: the feature hierarchy that causes absorption IS a latent causal structure recoverable from activation statistics.

10. **Westphal et al., 2025. "A Generalized Information Bottleneck Theory of Deep Learning." arXiv:2509.26327** -- Reformulates information bottleneck through synergy; shows synergistic features generalize better. Cross-domain insight: absorption can be reframed as a pathological compression where synergistic information between parent and child features is lost.

11. **Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." arXiv:2503.09532 (ICML 2025)** -- 8-metric evaluation across 200+ SAEs; reveals proxy metrics do not predict practical performance; modified absorption metric extends Chanin et al. to all layers. Essential benchmark infrastructure.

12. **Wu et al., 2025. "Interpreting and Steering LLMs with Mutual Information-Based Explanations on Sparse Autoencoders." arXiv:2502.15576** -- Directly applies MI to interpret SAE features. Suggests MI-based tools can quantify information relationships between features, potentially detecting absorption without supervision.

### Landscape Summary

Feature absorption sits at a critical juncture in the SAE interpretability research program. The phenomenon is well-characterized on a single proxy task (first-letter spelling), has been shown to be universal across architectures (including BatchTopK which lacks L1 loss), and several architectural mitigations exist (Matryoshka SAE, OrtSAE, ATM, masked regularization). However, three fundamental gaps remain:

First, absorption has only been measured where the researcher already knows the hierarchical feature structure -- the current metric requires supervised probe directions. This creates a chicken-and-egg problem: you cannot measure absorption on features you have not already identified, which means the vast majority of absorption in a typical SAE goes undetected.

Second, the relationship between absorption and other SAE failure modes (hedging, inconsistency, dark matter) is poorly understood. The balanced Matryoshka SAE result -- showing that reducing absorption increases hedging -- hints at a deeper tradeoff structure that no theoretical framework captures.

Third, every mitigation paper evaluates on different models, layers, and metrics, making it impossible to determine which approach actually works best. The field lacks a controlled comparison, and more importantly, lacks understanding of WHY certain mitigations work -- is it because they break co-occurrence patterns (masked regularization), enforce geometric separation (OrtSAE), or explicitly model hierarchy (Matryoshka)?

The cross-domain literature reveals powerful untapped analogies. Causal discovery methods for latent hierarchical models (rank constraints, conditional independence testing) offer principled tools for recovering feature hierarchies from data -- exactly the prerequisite for unsupervised absorption detection. The information bottleneck framework provides a lens for understanding absorption as pathological compression. And the ecological concept of competitive exclusion (Gause's law) -- where two species competing for the same niche cannot coexist -- maps precisely onto the absorption dynamic where parent and child features compete for activation slots under sparsity pressure.

The most promising research direction combines these insights: using the geometric and statistical signatures of absorption (decoder cosine similarity, conditional co-occurrence patterns) to detect absorption without supervision, then using this detection capability to characterize absorption across diverse feature hierarchies and SAE configurations.


## Phase 2: Initial Candidates

### Candidate A: Absorption Forensics via Decoder Geometry and Conditional Co-occurrence Graphs

**Hypothesis**: Feature absorption leaves characteristic geometric and statistical signatures in the SAE decoder weight matrix and feature activation co-occurrence patterns that can be detected without any supervised probe directions, enabling the first unsupervised absorption detection method.

**Cross-domain insight**: From causal discovery (d-separation and conditional independence testing for latent hierarchical models). In causal structure learning, hidden hierarchical variables are recovered by testing conditional independence between observed variables -- if X and Y are conditionally independent given Z, then Z mediates their relationship. Analogously, if SAE feature A (child/specific) and feature B (parent/general) are in an absorption relationship, then B's activation pattern conditional on A being inactive will show systematic "holes" corresponding to exactly those inputs where A has absorbed B. These conditional activation holes, combined with high decoder cosine similarity between A and B, form a joint geometric-statistical fingerprint of absorption.

**Evidence for**: (1) Park et al. (2025) show decoder cosine similarity encodes meaningful functional structure at 954-sigma significance. (2) Chanin et al. (2024) show absorbing latents have measurably high cosine similarity with probe directions. (3) The canonical absorption mechanism (specific feature absorbs general feature on co-occurring inputs) directly implies a conditional co-occurrence pattern: the general feature's false negatives are concentrated exactly where the specific feature fires. (4) Meta-SAE decomposition (Leask et al., 2025) shows absorbed information persists as sub-features within absorbing latents.

**Novelty estimate**: 8/10 -- No unsupervised absorption detection method exists (Gap 7 in literature survey). The combination of decoder geometry analysis with conditional co-occurrence testing for absorption detection is entirely novel. Individual components (cosine similarity analysis, co-occurrence statistics) exist but have never been combined or applied to absorption detection.


### Candidate B: Absorption as Competitive Exclusion -- An Ecological Framework for SAE Feature Dynamics

**Hypothesis**: Feature absorption follows the same mathematical structure as ecological competitive exclusion (Gause's law): when two features compete for the same "niche" (activation budget under sparsity pressure), the more specific feature always excludes the more general one on co-occurring inputs. This analogy predicts that absorption severity scales with the "niche overlap" (decoder cosine similarity x co-occurrence frequency) between feature pairs, and that absorption can be mitigated by "niche partitioning" -- forcing features to specialize in non-overlapping activation contexts, analogous to how species avoid competitive exclusion by specializing in different resources.

**Cross-domain insight**: From ecology (competitive exclusion principle and niche partitioning). Gause's law states that two species competing for exactly the same resource cannot coexist -- one will always drive the other to extinction. In SAEs, parent and child features "compete" for activation slots (the limited sparsity budget). On inputs where both should fire, the child feature provides more reconstruction benefit per activation slot (it encodes both parent and child information with +1 L0 vs +2 L0 without absorption). This is mathematically analogous to competitive exclusion where the specialist always outcompetes the generalist for the shared resource.

**Evidence for**: (1) Chanin et al. (2024) explicitly note that absorption is an "effective strategy" saving +1 L0 per parent-child pair -- this IS competitive exclusion of the general feature. (2) Matryoshka SAEs work by creating separate "niches" (dictionary sizes) for features at different hierarchy levels -- this IS niche partitioning. (3) OrtSAE's orthogonality penalty reduces decoder direction overlap -- this IS reducing niche overlap. (4) The balanced Matryoshka result (absorption-hedging tradeoff) parallels the ecological observation that reducing competitive exclusion can increase interspecific competition (hedging).

**Novelty estimate**: 6/10 -- The ecological analogy is conceptually elegant and provides a unifying narrative for existing mitigations, but it may be more of a reframing than a concrete methodological contribution. The predictive power (absorption scales with niche overlap) would be novel if validated quantitatively, but the risk is that this becomes a "nice metaphor" without actionable new methods.


### Candidate C: Information-Theoretic Absorption Diagnosis via Conditional Mutual Information Decomposition

**Hypothesis**: Feature absorption can be precisely quantified as the conditional mutual information I(Y; B | A=0) - I(Y; B | A>0) between target concept Y and parent feature B, conditioned on whether child feature A is active or inactive. When absorption occurs, B carries information about Y only when A is inactive (because when A fires, it absorbs B's role), creating a measurable CMI asymmetry. This metric is computable from activation statistics alone, requires no supervised probes, and decomposes naturally into components corresponding to absorption severity, hierarchy depth, and co-occurrence frequency.

**Cross-domain insight**: From information-theoretic feature selection (mRMR, Partial Information Decomposition). In the feature selection literature, redundancy between features is quantified via conditional mutual information -- if feature X provides no additional information about Y given feature Z, then X is redundant with Z. Feature absorption is precisely this: B provides no additional information about target concept Y given A on co-occurring inputs, because A has absorbed B's informational role. The PID framework (Unique, Redundant, Synergistic information decomposition) further allows us to decompose exactly how much of B's information is unique vs. absorbed by A.

**Evidence for**: (1) Wu et al. (2025) demonstrate MI-based SAE feature analysis is computationally feasible. (2) The PID framework (JMLR 2024) provides rigorous redundancy definitions. (3) Chanin et al.'s absorption mechanism directly implies CMI asymmetry: B fires on "starts with s" tokens unless A (e.g., "snake") fires, in which case A absorbs B's information. (4) The information bottleneck interpretation: absorption is pathological compression where the SAE's sparse bottleneck discards the synergistic information between parent and child features.

**Novelty estimate**: 9/10 -- No information-theoretic framework for absorption diagnosis exists. The CMI asymmetry metric is entirely novel and directly addresses Gap 7 (unsupervised absorption detection). The PID decomposition into unique/redundant/synergistic components of absorption is unprecedented. If validated, this would provide the first theoretically grounded, unsupervised, and architecture-agnostic absorption metric.


## Phase 3: Self-Critique

### Against Candidate A: Absorption Forensics via Decoder Geometry and Conditional Co-occurrence Graphs

**Prior work attack**: Park et al. (2025) already analyze decoder geometry extensively, and SAE stitching (Leask et al., 2025) already uses cosine similarity between decoder vectors to match features across SAEs. However, neither paper applies these tools specifically to absorption detection. The OrtSAE paper measures decoder cosine similarity as a proxy for absorption reduction. So while the individual components exist, their combination for unsupervised absorption detection is indeed novel. Searched "unsupervised absorption detection SAE decoder geometry" -- no direct hits.

**Methodological attack**: The main weakness is false positives. High decoder cosine similarity between two features could indicate: (a) absorption, (b) feature splitting (legitimate decomposition of a concept), or (c) coincidental geometric proximity. Disentangling these requires the conditional co-occurrence analysis, but this analysis depends on having enough activation data to estimate conditional statistics reliably. For rare features, the estimates may be noisy.

**Theoretical attack**: The approach is fundamentally empirical -- it identifies signatures of absorption but does not explain WHY those signatures arise from the optimization landscape. It provides a diagnostic tool, not a causal understanding. However, this is a strength for a training-free analysis paper.

**Scalability attack**: Computing pairwise decoder cosine similarities is O(d_sae^2 * d_model), which for a 65k-latent SAE is ~4.2 billion operations -- feasible but expensive. The conditional co-occurrence analysis requires running the SAE on a large corpus and tracking all pairwise conditional statistics, which is O(n_tokens * d_sae^2) -- potentially infeasible for large SAEs without sampling.

**Verdict**: MODERATE -- The core idea is sound and novel, but the scalability concern is real and the distinction between absorption vs. feature splitting via geometry alone is tricky. Needs a clear algorithmic pipeline that handles scalability (e.g., only analyzing features with cosine similarity above a threshold) and validates against the supervised absorption metric.


### Against Candidate B: Absorption as Competitive Exclusion

**Prior work attack**: Searching "competitive exclusion neural network feature learning" and "ecological analogy sparse coding" yields no direct results. The analogy itself appears novel. However, the predictive claim (absorption scales with niche overlap) is essentially saying absorption correlates with decoder cosine similarity x co-occurrence frequency, which is close to what Chanin et al. already observe informally. The "niche partitioning" interpretation of Matryoshka SAEs, while elegant, is a reframing of their explicit design intent (nested hierarchical dictionaries).

**Methodological attack**: The main risk is that this is a "just-so story" -- a post-hoc narrative that reframes existing results without generating new testable predictions. The ecological analogy needs to make predictions that differ from simpler explanations (e.g., "features with high decoder overlap show more absorption"). What does the competitive exclusion framework predict that the simple overlap story does not?

**Theoretical attack**: The analogy breaks down in important ways. In ecology, competitive exclusion leads to one species going extinct (permanently). In SAEs, absorption is input-dependent: the parent feature is only suppressed on inputs where the child also fires, but still fires on other inputs. This is closer to "niche partitioning" (coexistence through specialization) than "competitive exclusion" (one species eliminated). The analogy conflates two distinct ecological phenomena.

**Scalability attack**: The framework is conceptual, not computational. It does not propose a concrete algorithm or metric, just an interpretive lens. This limits its impact in a field that values quantitative contributions.

**Verdict**: WEAK -- The analogy is intellectually stimulating but breaks down under scrutiny, provides limited concrete predictions beyond existing understanding, and risks being a narrative overlay without substantive methodological contribution. The strongest element (niche overlap predicts absorption) can be tested without the ecological framing.


### Against Candidate C: Information-Theoretic Absorption Diagnosis via CMI Decomposition

**Prior work attack**: Wu et al. (2025, arXiv:2502.15576) apply MI to SAE feature interpretation, but for a different purpose (explaining individual feature activations, not detecting absorption). The PID framework has been used for feature selection but never applied to SAE feature quality analysis. Searched "conditional mutual information sparse autoencoder absorption" and "partial information decomposition SAE features" -- no direct hits. The novelty appears genuine.

**Methodological attack**: (1) CMI estimation in high dimensions is notoriously difficult. SAE features are high-dimensional (d_sae = 16k-65k), and estimating CMI between continuous activation values requires either binning (loses information), kernel density estimation (curse of dimensionality), or neural MI estimators (require training). However, the key insight is that SAE features are SPARSE -- most are zero most of the time -- so the conditioning on A=0 vs A>0 is a binary split, and the resulting CMI estimation reduces to comparing two conditional distributions of B's activations, which is much more tractable. (2) PID decomposition for more than 2-3 variables is computationally expensive and theoretically contested (multiple PID definitions exist). For the two-feature (parent-child) case, PID is well-defined, but extending to multi-level hierarchies (grandparent-parent-child) is harder.

**Theoretical attack**: The CMI asymmetry metric assumes that absorption is the primary cause of conditional dependence between features. But features could be conditionally dependent for many reasons: they might represent genuinely correlated concepts (e.g., "French" and "Paris" are correlated but not in an absorption relationship), or they might share reconstruction burden (feature hedging). The metric may conflate absorption with other forms of feature interaction. Mitigation: combine CMI asymmetry with decoder cosine similarity to filter for absorption-specific patterns (high decoder overlap + high CMI asymmetry = likely absorption; low decoder overlap + high CMI asymmetry = likely legitimate correlation).

**Scalability attack**: Pairwise CMI computation for all feature pairs is O(d_sae^2), but we only need to compute it for feature pairs with high decoder cosine similarity (potential absorption candidates), which dramatically reduces the number of pairs. The binary conditioning (A=0 vs A>0) makes the CMI estimation itself efficient -- it is just comparing two conditional activation distributions.

**Verdict**: STRONG -- The core idea is theoretically well-grounded, genuinely novel, computationally tractable (given the sparsity structure), and directly addresses the most important gap in the field (unsupervised absorption detection). The main weakness (confounding with legitimate correlation) has a clear mitigation path (joint CMI + decoder geometry filtering). The PID extension is a bonus, not the core contribution.


## Phase 4: Refinement

### Dropped Ideas

- **Candidate B (Ecological Framework)** dropped because: The competitive exclusion analogy breaks down under scrutiny (absorption is input-conditional, not permanent elimination), provides limited concrete predictions beyond existing understanding, and risks being a narrative overlay without substantive methodological contribution. However, the "niche overlap" quantification (decoder cosine similarity x co-occurrence frequency as a predictor of absorption severity) is a useful element that can be incorporated into the surviving ideas.

### Strengthened Ideas

**Candidate C (CMI Absorption Diagnosis) strengthened into the front-runner by incorporating elements from Candidates A and B:**

1. **Combined metric**: Instead of pure CMI asymmetry, the final method uses a two-stage pipeline: (Stage 1) Identify candidate absorption pairs via decoder cosine similarity screening (from Candidate A), then (Stage 2) confirm absorption via CMI asymmetry measurement (from Candidate C). This addresses the confounding concern: high decoder overlap + high CMI asymmetry = absorption; high decoder overlap + low CMI asymmetry = feature splitting; low decoder overlap + high CMI asymmetry = legitimate correlation.

2. **"Niche overlap" as absorption predictor**: From Candidate B, the quantitative prediction that absorption severity scales with decoder cosine similarity x co-occurrence frequency is incorporated as a testable scaling law. This is a concrete, falsifiable prediction that goes beyond existing understanding.

3. **Practical scope narrowed**: Instead of attempting full PID decomposition (theoretically contested for >2 variables), the core metric is the simpler CMI asymmetry ratio, which is well-defined and efficiently computable.

4. **Validation strategy sharpened**: The unsupervised metric will be validated against the supervised Chanin et al. absorption metric on the first-letter spelling task (where ground-truth absorption is known), then applied to domains where no supervised metric exists.

### Additional Evidence Found

- The PNAS 2025 protein SAE paper demonstrates that SAE features align with Gene Ontology hierarchies in a completely unsupervised fashion, proving that SAE activation statistics encode hierarchical structure even without supervision. This supports the feasibility of detecting hierarchical absorption relationships from activation statistics alone.

- The "Geometry of Concepts" paper validates that decoder cosine similarity captures meaningful functional relationships (not just geometric coincidence) at extremely high statistical significance (954 sigma). This strengthens the decoder geometry screening step.

### Selected Front-Runner

**Candidate C (strengthened)**: Information-theoretic absorption diagnosis via conditional mutual information asymmetry, combined with decoder geometry screening. This is the strongest because it (1) addresses the most important open gap (unsupervised absorption detection), (2) is theoretically grounded in well-established information theory, (3) is computationally tractable given SAE sparsity structure, (4) generates a novel, falsifiable scaling law (absorption ~ decoder overlap x co-occurrence frequency), and (5) is training-free, leveraging existing pre-trained SAEs.


## Phase 5: Final Proposal

### Title

Detecting Feature Absorption Without Supervision: An Information-Theoretic Diagnostic for Sparse Autoencoders

### Hypothesis

Feature absorption in sparse autoencoders creates a measurable conditional mutual information asymmetry between parent-child feature pairs: the parent feature's mutual information with its target concept drops sharply when conditioned on the child feature being active versus inactive. This asymmetry, combined with high decoder cosine similarity, provides a sufficient statistic for detecting absorption without requiring any supervised probe directions.

Formally: For a candidate parent feature B and child feature A, define the **absorption score** as:

AS(A, B) = cos(d_A, d_B) * [H(B | A=0) - H(B | A>0)] / H(B)

where d_A, d_B are decoder weight vectors, and H(B | A=c) is the conditional entropy of B's activation given A's activation state. When A absorbs B, B's activation entropy drops dramatically when A is active (because B is suppressed), creating a high absorption score.

This hypothesis is falsifiable: if the absorption score does not correlate with the supervised Chanin et al. metric on the first-letter spelling task, the hypothesis is rejected.

### Motivation

The ability to detect absorption is currently limited by a fundamental chicken-and-egg problem: the canonical absorption metric (Chanin et al., 2024) requires supervised probe directions for the "should-have-fired" feature, which means researchers must already know which features exist to measure whether they have been absorbed. This restricts absorption analysis to narrow proxy tasks (first-letter spelling) where feature hierarchies are known in advance.

This limitation has three consequences. First, we have no idea how prevalent absorption is across the full feature space of a typical SAE -- the 15-35% absorption rate measured on first-letter features may be atypical. Second, we cannot measure absorption on the safety-relevant features (deception, harmful intent, bias) that motivate SAE research in the first place. Third, we cannot efficiently evaluate whether mitigation methods (Matryoshka SAE, OrtSAE, ATM, masked regularization) actually reduce absorption globally, or only on the narrow task used for evaluation.

An unsupervised absorption detection method would break this bottleneck, enabling absorption characterization across any feature hierarchy, any model, and any SAE architecture -- without requiring manual identification of the features involved.

### Method

**Stage 1: Candidate Pair Identification (Decoder Geometry Screening)**

For each pair of SAE latents (i, j), compute the cosine similarity between their decoder weight vectors: sim(i, j) = cos(d_i, d_j). Retain pairs with sim > tau_geo (e.g., tau_geo = 0.3), which identifies features whose decoder directions partially overlap -- a necessary condition for absorption (the absorbing feature must encode part of the absorbed feature's direction).

To handle scale (d_sae = 16k-65k latents), use approximate nearest neighbor search (FAISS) on decoder vectors to find high-similarity pairs in O(d_sae * log(d_sae)) instead of O(d_sae^2).

**Stage 2: CMI Asymmetry Computation**

For each candidate pair (A, B), run the SAE on a corpus of N tokens and record activations. Partition tokens into two sets: S_0 = {tokens where A=0} and S_1 = {tokens where A>0}. Compute:

- H(B | A=0): Entropy of B's activation values over S_0
- H(B | A>0): Entropy of B's activation values over S_1
- CMI_asym(A, B) = [H(B | A=0) - H(B | A>0)] / H(B)

When A absorbs B, B fires normally on S_0 but is suppressed on S_1, so H(B | A>0) << H(B | A=0), yielding high CMI_asym. The normalization by H(B) makes the metric comparable across features with different base rates.

**Stage 3: Absorption Score and Directionality**

Combine geometry and information:

AS(A, B) = sim(A, B) * CMI_asym(A, B)

Directionality: If AS(A, B) > AS(B, A), then A absorbs B (A is the child/specific feature, B is the parent/general feature). This asymmetry naturally emerges because the child suppresses the parent, not vice versa.

**Stage 4: Absorption Graph Construction**

Build a directed graph where nodes are SAE latents and edges are absorption relationships (weighted by AS). This graph reveals:
- **Hub nodes**: Features that absorb many others (highly specific "token-level" features)
- **Source nodes**: Features that are absorbed by many children (highly general features vulnerable to absorption)
- **Chains**: Multi-level absorption hierarchies (grandparent -> parent -> child)
- **Clusters**: Groups of features involved in mutual absorption relationships

**Stage 5: Validation**

On the first-letter spelling task (where ground-truth absorption is known):
1. Compute AS for all latent pairs
2. Compare top-scoring absorption pairs against Chanin et al.'s supervised absorption metric
3. Measure precision/recall: What fraction of AS-detected absorptions are confirmed by the supervised metric? What fraction of supervised absorptions are detected by AS?

**Stage 6: Cross-Domain Application**

Apply the validated AS metric to feature hierarchies where no supervised metric exists:
- Entity type hierarchies (country > city, animal > species)
- Semantic hierarchies (sentiment > topic, syntax > semantics)
- Safety-relevant features (harmful intent > specific harm category)

For each domain, identify the most-absorbed features and characterize how absorption severity varies with SAE architecture, width, sparsity, and layer.

### Experimental Plan

**Phase A: Metric Validation (Pilot, ~15 min)**
- Model: Gemma-2-2B with Gemma Scope SAEs (layer 12, 16k width)
- Task: First-letter spelling absorption (ground truth from Chanin et al.)
- Measure: Spearman correlation between AS metric and supervised absorption rate across 26 letters
- Baseline: Random pairs with same decoder cosine similarity (control for geometry-only detection)
- Success criterion: Spearman rho > 0.6 between AS and supervised metric

**Phase B: Architecture Comparison (~30 min)**
- SAEs: SAEBench suite on Gemma-2-2B layer 12 (BatchTopK, TopK, JumpReLU, Matryoshka, at 16k width)
- Measure: Global absorption score distribution for each architecture
- Prediction: Matryoshka < OrtSAE < BatchTopK < TopK < JumpReLU (consistent with SAEBench supervised results)
- Novel finding: Architecture ranking on AS metric vs. supervised metric may differ, revealing absorption that the first-letter task misses

**Phase C: Cross-Domain Absorption Characterization (~45 min)**
- Construct entity-type hierarchies using Gemma-2-2B knowledge:
  - Country > City (e.g., "France" feature absorbing "Paris" feature, or vice versa)
  - Animal > Species (e.g., "reptile" feature absorbing "snake" feature)
  - Profession > Specific person (e.g., "scientist" absorbing "Einstein")
- For each hierarchy, compute AS across all latent pairs and identify absorption patterns
- Compare absorption rates across domains: Is absorption more severe for syntactic (first-letter) vs. semantic (entity-type) hierarchies?

**Phase D: Scaling Law Validation (~30 min)**
- Vary SAE width (4k, 16k, 65k) and sparsity (L0 = 25, 50, 100, 200) using SAEBench SAEs
- For each configuration, compute mean AS across top-1000 candidate pairs
- Test prediction: mean AS ~ alpha * (1/width) * (1/L0) * mean(decoder_cosim * cooccurrence_freq)
- If the scaling law holds, it provides the first principled guidance for choosing SAE hyperparameters to minimize absorption

**Baselines:**
1. Supervised Chanin et al. metric (upper bound -- uses probe directions)
2. Decoder cosine similarity alone (geometry-only baseline)
3. Co-occurrence frequency alone (statistics-only baseline)
4. Random feature pairs (null baseline)

**Falsification criteria:**
- If AS metric has Spearman rho < 0.3 with supervised metric on first-letter task, the hypothesis is rejected
- If architecture ranking under AS disagrees qualitatively with SAEBench supervised ranking, the metric is unreliable
- If cross-domain absorption rates show no systematic patterns (e.g., completely random across hierarchy types), the metric may be measuring noise

### Resource Estimate

- **Compute**: All experiments use pre-trained SAEs (Gemma Scope, SAEBench SAEs) -- no SAE training required. Main cost is running SAEs on corpus for activation collection (~10 min per SAE on single GPU) and pairwise computation (~5 min per SAE with FAISS).
- **Models**: Gemma-2-2B (fits on single A100/H100 in bf16), GPT-2 Small (trivial)
- **Total GPU time**: ~2-3 hours across all phases
- **Libraries**: SAELens, TransformerLens, FAISS, scipy (for entropy/MI estimation)
- **Wall-clock time**: Each phase independently fits within 1 hour; total ~3-4 hours

### Risk Assessment

**Risk 1: CMI asymmetry confounded with legitimate feature correlation.**
Two features representing genuinely correlated but non-hierarchical concepts (e.g., "rain" and "umbrella") may show CMI asymmetry not caused by absorption. 
*Mitigation*: The decoder cosine similarity filter (Stage 1) addresses this -- absorption requires the absorbing feature to encode part of the absorbed feature's decoder direction, while merely correlated features typically have low decoder overlap. The combination of high decoder cosim + high CMI asymmetry is much more specific to absorption than either alone.

**Risk 2: Entropy estimation noise for sparse/rare features.**
Features that fire very rarely may have insufficient activation samples for reliable entropy estimation, leading to noisy AS scores.
*Mitigation*: (a) Use large corpus (>1M tokens) for activation collection. (b) Apply minimum activation count threshold (e.g., feature must fire on >= 100 tokens to be included). (c) Use Bayesian entropy estimators (Nemenman-Shafee-Bialek) that are robust to small sample sizes.

**Risk 3: Scaling to large SAEs (65k+ latents).**
Pairwise computation is O(d_sae^2) which is ~4 billion pairs for 65k latents.
*Mitigation*: FAISS approximate nearest neighbor search reduces candidate pair identification to O(d_sae * log(d_sae)). Only pairs above the cosine similarity threshold proceed to CMI computation, typically reducing the number of pairs by 100-1000x.

### Novelty Claim

This work introduces the first **unsupervised** method for detecting feature absorption in sparse autoencoders, directly addressing Gap 7 in the literature. Specifically:

1. **The absorption score (AS) metric** combines decoder geometry (cosine similarity) with information-theoretic asymmetry (conditional entropy ratio) into a single, principled diagnostic. No prior work combines these two signals for absorption detection.

2. **The absorption graph** -- a directed graph of absorption relationships between SAE latents -- is an entirely new analytical object that reveals the hierarchical structure of absorption in a trained SAE without any external supervision.

3. **Cross-domain absorption characterization** -- measuring absorption on semantic/knowledge hierarchies beyond the first-letter spelling task -- has never been done because no unsupervised detection method existed. This work enables it for the first time.

4. **The scaling law prediction** (absorption severity as a function of decoder overlap, co-occurrence frequency, SAE width, and L0) is a novel quantitative prediction that, if validated, provides the first principled guidance for configuring SAEs to minimize absorption.

The key innovation is recognizing that absorption's defining characteristic -- a parent feature being suppressed specifically when a child feature fires -- creates a detectable conditional information asymmetry. This transforms an unobservable phenomenon (requiring supervised probes) into an observable one (requiring only activation statistics), dramatically expanding the scope of absorption analysis.
