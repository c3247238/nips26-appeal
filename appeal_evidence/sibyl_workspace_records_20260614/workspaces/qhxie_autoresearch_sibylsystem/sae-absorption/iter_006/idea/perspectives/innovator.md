# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507 (NeurIPS 2025)** -- Defines the absorption phenomenon, provides a toy model proof, and establishes the canonical metric on the first-letter spelling task. Finds 15-35% absorption rates across all tested SAEs. This is the paper we must extend.

2. **Michaud, Engels, Baek, Sun, Tegmark, 2025. "The Geometry of Concepts: Sparse Autoencoder Feature Structure." Entropy 27(4):344** -- Demonstrates that spectral clustering of SAE decoder weight vectors via cosine similarity reveals hierarchical, modular "lobes" in feature space that correspond to functional co-occurrence structure. Establishes that decoder geometry contains rich hierarchical information at multiple scales ("crystals," "lobes," "continents"). Critical for the unsupervised detection angle: decoder geometry IS informative about feature relationships.

3. **arXiv:2509.02565, 2025. "Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds."** -- Shows that when many SAE latents tile a low-dimensional feature manifold, their decoder directions have high pairwise cosine similarity. Provides the theoretical grounding for why absorbed parent-child pairs might have specific geometric signatures in decoder space: the child feature's decoder direction partially overlaps with the parent's manifold.

4. **Rozanski et al., 2025. "Relating Sparse and Predictive Coding to Divisive Normalization." PLOS Computational Biology** -- Unifies three computational neuroscience principles: sparse coding, predictive coding, and divisive normalization. The key insight: winner-take-all (extreme sparsity) naturally produces competitive exclusion among neural representations, where one representation absorbs the role of another. This is the neuroscience analog of feature absorption.

5. **Beny, 2013/2025 revival. "Deep Learning and the Renormalization Group" + AAAI 2025 extensions** -- The RG framework shows that hierarchical feature extraction in deep networks is formally analogous to coarse-graining in statistical physics. Each layer implements a scale transformation. Feature absorption maps onto "relevant operator absorption" in the RG sense: a fine-scale (child) feature captures all information of a coarse-scale (parent) feature, making the parent irrelevant at that resolution. This gives a principled multi-scale lens for understanding absorption.

6. **Hammond et al. "The Spectral Graph Wavelet Transform" + WaveGC (ICML 2025)** -- Spectral graph wavelets provide multi-resolution decomposition on arbitrary graph structures. If we construct a graph from SAE decoder cosine similarities, spectral graph wavelets at different scales can identify absorption pairs as features that merge across resolution scales -- exactly the "multi-resolution decoder geometry" approach the current proposal needs, but with a principled spectral foundation rather than ad hoc hierarchical clustering.

7. **Szlam & Gregor. "Structured Sparse Coding via Lateral Inhibition"** -- Shows that lateral inhibition in sparse coding produces structured competition where features at the same hierarchical level suppress each other. This provides the mechanistic bridge: absorption is what happens when lateral inhibition (sparsity penalty) operates across hierarchy levels instead of within them.

8. **Tang et al., 2024. "A Unified Theory of Sparse Dictionary Learning." arXiv:2512.05534** -- Casts all SDL variants as piecewise biconvex optimization. Proposes feature anchoring to restore identifiability. The theoretical foundation for why absorption is a spurious minimum of the optimization landscape.

9. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547 (ICML 2025)** -- Best existing mitigation via nested dictionaries that enforce hierarchical feature organization. Reduces absorption by giving the optimizer a structured pathway for parent-child relationships.

10. **Chanin & Garriga-Alonso, 2025. "Sparse but Wrong." arXiv:2508.16560** -- Shows most open-source SAEs have incorrect L0, causing feature hedging. Critical confound that must be controlled in any absorption study.

11. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders." arXiv:2509.22033** -- Reduces absorption 65% via orthogonality penalty. Provides three complementary absorption metrics (mean absorption fraction, full-absorption score, feature splits).

12. **Li et al., 2025. "Time-Aware Feature Selection: Adaptive Temporal Masking." arXiv:2510.08855** -- Best reported absorption scores (0.0068 vs TopK 0.1402) via per-latent importance scoring. Shows that dynamic sparsity allocation dramatically reduces absorption.

### Landscape Summary

The field is at a critical inflection point. Feature absorption is now recognized as one of the most important failure modes of SAEs, with direct implications for safety (DeepMind's deprioritization of SAE research in 2025) and mechanistic interpretability (undermining Anthropic's circuit tracing program). The canonical Chanin et al. study established the phenomenon and metric but is limited to the first-letter spelling task and requires supervised probes. Multiple architectural mitigations exist (Matryoshka, OrtSAE, ATM, masked regularization) but have never been compared head-to-head, and their theoretical basis for WHY they reduce absorption is poorly understood.

The most critical gap I see is methodological: all absorption detection requires pre-specified probe directions. This creates a fundamental bootstrapping problem -- you can only detect absorption for features you already know about. This limits absorption analysis to toy settings and prevents the field from understanding how widespread absorption is "in the wild." The current proposal's ITAC approach is a step in the right direction, but I see an opportunity to ground it in a much more principled theoretical framework.

A second critical gap is the absence of any cross-domain explanation for WHY absorption occurs with varying severity across different types of feature hierarchies. The existing theoretical account (sparsity incentive) predicts absorption should be universal but provides no quantitative prediction of rates. My cross-domain analog -- ecological competitive exclusion -- suggests that the severity depends on "niche overlap" (the extent to which parent and child features share the same representational space), which is a testable, quantitative prediction.

## Phase 2: Initial Candidates

### Candidate A: Spectral Graph Wavelet Absorption Detection (SGWA)

- **Hypothesis**: Feature absorption can be detected without supervision by applying spectral graph wavelet transforms to the SAE decoder weight graph, where absorption pairs manifest as features that merge at a specific wavelet scale, creating a scale-dependent signature that distinguishes absorption from mere semantic similarity.

- **Cross-domain insight**: From signal processing / spectral graph theory. In multi-resolution signal decomposition, a signal component that "disappears" at a particular scale but re-emerges as part of another component at a finer scale is precisely the signature of a hierarchical dependency. Feature absorption is the dictionary learning analog: the parent feature's signal disappears from the SAE's explicit representation but is implicitly encoded within the child feature. Spectral graph wavelets on the decoder cosine similarity graph can detect exactly this pattern across multiple resolution scales simultaneously.

- **Why it might work**: (a) Michaud et al. (2025) proved that SAE decoder geometry contains rich multi-scale structure ("crystals," "lobes"). (b) The feature manifold scaling paper shows absorbed features tile overlapping manifold regions with characteristic cosine similarity patterns. (c) Spectral graph wavelets (Hammond et al.) provide a mathematically principled way to decompose graph structure at multiple resolutions without the arbitrary threshold choices that plague hierarchical clustering. (d) The current proposal's multi-resolution decoder geometry analysis uses ad hoc resolution thresholds; SGWT replaces this with principled spectral decomposition.

- **Novelty estimate**: 8/10 -- No prior work applies spectral graph wavelets to SAE decoder matrices for ANY purpose, let alone absorption detection. The Geometry of Concepts paper uses basic spectral clustering; we upgrade to full multi-resolution wavelet analysis. The connection between scale-dependent merging events in the wavelet decomposition and absorption is entirely novel.

### Candidate B: Ecological Niche Theory of Feature Absorption

- **Hypothesis**: Feature absorption in SAEs obeys a quantitative analog of the competitive exclusion principle from ecology: two features competing for the same representational "resource" (activation budget under sparsity constraint) cannot stably coexist if their "niche overlap" (measured by conditional cosine similarity weighted by co-occurrence frequency) exceeds a critical threshold that depends on the effective carrying capacity (SAE width / number of active features).

- **Cross-domain insight**: From theoretical ecology (Gause's competitive exclusion principle, Lotka-Volterra competition models). In ecology, two species competing for the same resource cannot coexist at equilibrium; one will exclude the other. But if niches differ sufficiently (resource partitioning), both survive. Feature absorption is competitive exclusion in representation space: the child feature "excludes" the parent from activation because they compete for the same sparse activation slots. The critical niche overlap threshold from ecology maps to a critical cosine similarity / co-occurrence ratio threshold for absorption onset.

- **Why it might work**: (a) Chanin et al.'s "sparsity gain" argument is precisely an energy argument about competition for a limited resource (L0 budget). (b) The Lotka-Volterra framework provides closed-form predictions for competitive outcomes as a function of niche overlap and carrying capacity. (c) SAE width (carrying capacity) and L0 (resource budget) are known to affect absorption rates. (d) The ecology framework predicts a phase transition (absorption onset at critical overlap), which matches the L0 ~ 7-14 phase transition found in iteration 5.

- **Novelty estimate**: 9/10 -- No prior work connects ecological competition theory to SAE feature dynamics. The Lotka-Volterra analogy provides quantitative predictions (critical overlap threshold, carrying capacity dependence) that are directly testable on existing SAEs. This would be the first mathematical model of absorption that produces quantitative rate predictions from measurable feature properties.

### Candidate C: Renormalization Group Absorption Diagnostic

- **Hypothesis**: Feature absorption can be understood as a failure of scale separation in the renormalization group sense: when the RG transformation (coarse-graining from fine features to coarse features) is not properly invertible, information is lost at scale boundaries. The severity of absorption is predicted by the spectral gap of the coarse-graining operator restricted to hierarchically related feature pairs.

- **Cross-domain insight**: From statistical physics (renormalization group). In RG, irrelevant operators at one scale are absorbed into the effective description at the next scale -- their individual effects become invisible. This is structurally identical to feature absorption: the parent feature (coarse-scale operator) is "absorbed" into the child feature (fine-scale operator). The RG framework predicts when this happens: when the spectral gap between relevant and irrelevant operators is small.

- **Why it might work**: (a) AAAI 2025 and Phys. Rev. E 2026 papers establish rigorous RG-DNN correspondence. (b) The Matryoshka SAE's nested dictionary structure is literally a discrete RG transformation. (c) The successive refinement theorem (already in the current proposal) is the information-theoretic cousin of RG coarse-graining. (d) The spectral gap prediction is quantitative and testable.

- **Novelty estimate**: 7/10 -- The RG-deep learning connection is well-established qualitatively, but applying it specifically to SAE feature absorption is novel. The spectral gap diagnostic is new. However, making the connection rigorous enough for a tight prediction requires substantial theoretical work that may exceed the project's scope.

## Phase 3: Self-Critique

### Against Candidate A (Spectral Graph Wavelet Absorption Detection)

- **Prior work attack**: Searched "spectral graph wavelet sparse autoencoder" and "graph wavelet dictionary learning feature detection" -- zero results combining SGWTs with SAE analysis. The Geometry of Concepts paper uses basic spectral clustering, not wavelets. The LessWrong post "Looking for feature absorption automatically" attempted raw cosine similarity without any multi-resolution structure and failed. Our approach is methodologically distinct. **No direct competition found.**

- **Methodological attack**: The main risk is computational scalability. For an N-feature SAE, the decoder cosine similarity matrix is N x N. For 16k features, this is 256M entries -- feasible. The SGWT requires computing the graph Laplacian eigendecomposition or Chebyshev approximation, which scales as O(N^2) to O(N^3). For N=16k, Chebyshev approximation is tractable. For N=65k, full eigendecomposition is infeasible, but Chebyshev approximation still works. The bigger risk is that the graph may be too dense (too many non-zero cosine similarities) for meaningful wavelet decomposition -- need to threshold the graph at a reasonable sparsity level.

- **Theoretical attack**: The analogy between signal decomposition and feature absorption holds structurally: both involve hierarchical components that merge at specific scales. However, the mapping is not exact. In classical SGWT, the "signal" on the graph is a scalar function on vertices. In our case, we need to define what the "signal" is -- perhaps the feature activation pattern across a corpus, or the residual projection. The wavelet coefficients then measure scale-dependent variation of this signal, which we interpret as absorption signatures. This mapping is reasonable but requires careful formalization.

- **Scalability attack**: Works for 16k and plausibly 65k features with Chebyshev approximation. May struggle with 1M features (some Gemma Scope SAEs), but those are not the primary target for this study.

- **Verdict**: **STRONG** -- Computationally feasible, theoretically grounded (more so than ad hoc threshold sweeps), and genuinely novel. The main implementation risk (graph density) is addressable by thresholding.

### Against Candidate B (Ecological Niche Theory)

- **Prior work attack**: Searched "competitive exclusion machine learning features," "ecological competition neural network representation," "Lotka-Volterra feature competition." Found competitive learning in neural networks (winner-take-all) but NO prior work mapping ecological competition theory specifically to SAE features or feature absorption. The competitive learning literature is about unsupervised clustering, not about analyzing trained overcomplete dictionaries. **No direct competition found.**

- **Methodological attack**: The Lotka-Volterra equations describe continuous-time dynamics of two competing populations. SAE training is a discrete optimization process. The mapping requires: (1) defining "population size" for a feature (activation frequency? magnitude?), (2) defining "carrying capacity" (SAE width? L0?), (3) defining "niche overlap" (conditional cosine similarity? co-activation rate?). Each mapping choice introduces degrees of freedom. The risk is that with enough choices, any framework can be fit post hoc. **Mitigation**: Pre-register the specific mappings and derive quantitative predictions BEFORE running experiments.

- **Theoretical attack**: The competitive exclusion principle strictly applies to equilibrium dynamics of well-mixed populations with density-dependent growth. SAE features are not populations, SAE training is not equilibrium dynamics, and the sparsity constraint is not exactly density-dependent growth. However, many ecological models have been successfully applied by analogy to other competition settings (e.g., species in fitness landscapes, memes in cultural evolution, products in markets). The key question is whether the analogy is structural (same mathematical structure) or merely metaphorical. The Lotka-Volterra competition model has the same structure as two features competing for activation under an L0 constraint if we make the right identifications. I believe the structural correspondence holds, but a critic would rightly push on whether the quantitative predictions (not just qualitative ones) match.

- **Scalability attack**: The ecological model is per-pair (parent-child), so it scales linearly with the number of hierarchical pairs. Computing niche overlap for all candidate pairs requires the same cosine similarity computation as Candidate A. No scalability concern.

- **Verdict**: **MODERATE** -- Highly novel and potentially high-impact, but the analogy needs to be very carefully formalized to avoid being dismissed as a metaphor. The quantitative predictions are the key deliverable; if they match empirical absorption rates (even approximately), this is a strong contribution. If they don't, it's a pretty metaphor with no teeth.

### Against Candidate C (Renormalization Group Diagnostic)

- **Prior work attack**: Searched "renormalization group sparse autoencoder" and "RG feature absorption." Found no direct application of RG to SAE absorption. The RG-deep learning literature (Beny 2013, AAAI 2025, Phys. Rev. E 2026) establishes general connections between deep learning and RG but does not address sparse overcomplete dictionaries or absorption specifically.

- **Methodological attack**: Making the RG connection rigorous requires defining: (1) the "block-spin" transformation that corresponds to going from child to parent features, (2) the "Hamiltonian" that the SAE is implicitly optimizing, (3) the notion of "relevance" (in the RG sense) for features. These are non-trivial theoretical constructs. The risk is that the formalization is so approximate that the "spectral gap" prediction is no better than simpler metrics (e.g., just measuring cosine similarity).

- **Theoretical attack**: The RG analogy works best for systems with genuine scale separation (e.g., wavelets, convolutional networks with pooling). SAEs are not hierarchical by construction -- they are flat overcomplete dictionaries. The hierarchy only emerges implicitly from training on hierarchically structured data. This means the "RG transformation" has to be inferred post hoc, which undermines the predictive power of the framework.

- **Scalability attack**: Requires eigendecomposition of a coarse-graining operator for each candidate feature pair -- potentially expensive but not prohibitive.

- **Verdict**: **WEAK** -- The theoretical formalization burden is too high for the expected payoff. The RG framework is beautiful but overkill for this problem: the key predictions (scale-dependent merging, spectral gap) can be captured more directly by Candidate A's spectral graph wavelets without the heavy theoretical machinery.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C (RG Diagnostic)** dropped because: The theoretical formalization burden exceeds the project's scope and capabilities. The key useful predictions (scale-dependent merging, spectral gap) are captured more directly and rigorously by Candidate A's spectral graph wavelet approach without requiring the full RG apparatus. The RG connection can be mentioned as a motivating framework in the Discussion but should not be the core method.

### Strengthened Ideas

- **Candidate A (SGWA) strengthened**: 
  - Addressed graph density concern: Threshold the cosine similarity graph at the 95th percentile to create a sparse graph, then apply Chebyshev-approximated SGWT. The threshold sensitivity will be an explicit ablation.
  - Clarified the "signal" definition: The signal on the graph vertices is the mean activation magnitude across a corpus. Absorption manifests as a feature with high activation magnitude at fine wavelet scales (child fires strongly on specific inputs) but missing activation at coarse scales (parent fails to fire on the same inputs). The wavelet coefficient ratio (fine/coarse) at absorption scale directly quantifies absorption severity.
  - Added validation protocol: Compare SGWA scores with Chanin et al. probe-based metric on first-letter task as ground truth. Target Spearman rho > 0.5.

- **Candidate B (Ecological Niche Theory) strengthened**:
  - Pre-registered the specific variable mappings:
    - "Population size" = mean activation magnitude of a feature across corpus
    - "Carrying capacity" K = SAE width / mean L0 (available activation slots per token)
    - "Niche overlap" alpha_ij = (# tokens where both features i,j would ideally fire) / (# tokens where feature i would ideally fire), estimated from probe predictions
    - "Competition coefficient" = alpha_ij / K
  - Derived the specific quantitative prediction from Lotka-Volterra: absorption occurs when alpha_ij > K_child / K_parent, i.e., when the niche overlap exceeds the ratio of carrying capacities. With K_child approximately K_parent (same SAE), absorption occurs when alpha_ij > 1, which simplifies to: absorption occurs for all parent-child pairs where the child's coverage exceeds the parent's. This is testable.
  - Added the key differentiator from the existing "sparsity gain" argument: Chanin et al.'s argument is qualitative (absorption saves L0). Our ecological model quantifies HOW MUCH absorption occurs as a function of specific measurable properties. The Lotka-Volterra equilibrium predicts the steady-state activation rates, which can be compared to empirical activation rates.

### Additional Evidence Found

- **Park et al. (2025, "The Geometry of Concepts")**: Confirmed that decoder cosine similarity clustering produces meaningful feature families. This directly supports Candidate A's use of the decoder cosine similarity graph as the substrate for SGWT analysis.

- **Feature manifold scaling paper (arXiv:2509.02565)**: Confirmed that absorbed features tile overlapping manifold regions with high cosine similarity. This validates the geometric basis of both Candidate A (wavelet detection of merging manifolds) and Candidate B (niche overlap measured by cosine similarity).

- **Lotka-Volterra competition in neural networks**: While no paper applies Lotka-Volterra to SAE absorption specifically, the competitive learning literature (winner-take-all) establishes that competitive exclusion dynamics arise naturally in neural networks with lateral inhibition / sparsity constraints. Rozanski et al. (2025) unify sparse coding with competitive dynamics.

### Selected Front-Runner

**Candidate A (Spectral Graph Wavelet Absorption Detection)** is selected as the primary methodological innovation, with **Candidate B (Ecological Niche Theory)** integrated as the theoretical framework that explains WHY the SGWA method works and provides quantitative predictions.

**Rationale for this combination**: SGWA provides the practical detection method (what to compute), while the ecological niche framework provides the interpretive theory (why it works and what the numbers mean). Together they form a complete contribution: a principled unsupervised absorption detection method grounded in a quantitative theory of feature competition. This combination is stronger than either alone because:

1. SGWA without theory is "just another unsupervised metric" -- the ecological theory explains why scale-dependent merging in the wavelet decomposition corresponds to absorption
2. The ecological theory without a detection method is "just an analogy" -- SGWA operationalizes the niche overlap concept into a concrete computational pipeline
3. The combination provides both a tool (SGWA) and a theory (ecological competitive exclusion) that the community can use independently

## Phase 5: Final Proposal

### Title

Spectral Graph Wavelet Analysis of Feature Competition: Multi-Resolution Unsupervised Detection of Absorption in Sparse Autoencoders

### Hypothesis

Feature absorption in SAEs creates a characteristic multi-scale signature in the spectral decomposition of the decoder weight graph: absorbed parent-child pairs exhibit high wavelet coefficient energy at the scale where they merge (the "absorption scale"), and the ratio of fine-to-coarse wavelet energy at this scale predicts absorption severity as measured by the gold-standard probe-based metric (Spearman rho > 0.5). Furthermore, the absorption scale at which merging occurs is predicted by the niche overlap between parent and child features, with a phase transition at the critical overlap threshold derived from Lotka-Volterra competition dynamics.

### Motivation

Feature absorption is the most critical failure mode undermining SAE-based mechanistic interpretability. Yet detection currently requires pre-trained probes for known features -- a fundamental bootstrapping problem that limits analysis to toy settings. The current proposal already includes a multi-resolution decoder geometry approach and ITAC, but these rely on ad hoc threshold choices (cosine similarity thresholds swept from 0.1 to 0.9 in steps of 0.05) and hierarchical clustering with arbitrary linkage functions. We propose to replace these ad hoc choices with a mathematically principled spectral graph wavelet analysis that provides:

1. **Principled multi-resolution decomposition**: No arbitrary threshold choices; the wavelet scales are determined by the spectral properties of the graph Laplacian
2. **Quantitative absorption scale identification**: Each absorption pair has a characteristic "absorption scale" where the parent feature's wavelet coefficient energy transfers to the child's
3. **Theoretical grounding via ecological competition**: The absorption scale is predicted by the niche overlap between features, providing the first quantitative theory of absorption severity

This addresses Gap 7 (no unsupervised detection) with a method that is more principled than the current ITAC pipeline, and Gap 1 (no quantitative theory) via the ecological niche framework.

### Method

**Stage 1: Decoder Graph Construction**

1. Load pre-trained SAE decoder weight matrix W_dec (shape: N_features x d_model) from Gemma Scope
2. Compute pairwise cosine similarity matrix S = W_dec @ W_dec^T / (||W_dec|| ||W_dec||^T)
3. Threshold S at the p-th percentile (default p=95, ablated over {90, 95, 97, 99}) to create a sparse adjacency matrix A
4. Compute the normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}
5. Time: O(N^2 * d) for cosine similarity, O(N * K * M) for Chebyshev wavelet computation where K is polynomial order and M is the number of scales

**Stage 2: Spectral Graph Wavelet Transform**

1. Choose a wavelet generating kernel g(x) (default: Mexican hat wavelet, ablated with heat kernel and simple difference)
2. Select J wavelet scales logarithmically spaced from the minimum to maximum eigenvalue of L (default J=10, ablated over {5, 8, 10, 15, 20})
3. For each scale j, compute the wavelet operator Psi_j using Chebyshev polynomial approximation of order K=30 (no eigendecomposition needed): Psi_j = g(s_j * L) where s_j is the j-th scale
4. Define the "signal" on the graph: f_i = mean activation magnitude of feature i across a corpus of 100k tokens
5. Compute wavelet coefficients: c_j(i) = (Psi_j f)(i) for each feature i at each scale j
6. This gives a J-dimensional wavelet signature for each feature

**Stage 3: Absorption Score Computation**

1. For each pair of features (i, k) with cosine similarity S(i,k) > threshold:
   - Compute the "absorption scale" j* = argmax_j |c_j(i) - c_j(k)| -- the scale at which the two features are most different
   - Compute the "absorption energy" E(i,k) = |c_{j*}(i) - c_{j*}(k)| / max(c_{j*}(i), c_{j*}(k))
   - Compute the "scale asymmetry" A(i,k) = (sum of c_j(i) for j > j*) / (sum of c_j(i) for j <= j*) -- ratio of fine-to-coarse wavelet energy for the candidate parent
2. Define the SGWA absorption score for feature i as:
   - SGWA(i) = max over neighbors k of [E(i,k) * A(i,k) * firing_rate_asymmetry(i,k)]
   - where firing_rate_asymmetry = max(fr_i, fr_k) / min(fr_i, fr_k) if fr_i != fr_k else 0
3. The firing rate asymmetry filter ensures we only flag asymmetric (parent-child) pairs, not symmetric (sibling) pairs

**Stage 4: Ecological Niche Interpretation (Quantitative Theory)**

1. For each candidate absorption pair identified by SGWA:
   - Compute niche overlap alpha = co-activation rate (fraction of tokens where both features would ideally activate, estimated from probe predictions on ground-truth task or from raw co-activation statistics)
   - Compute effective carrying capacity K = SAE_width / mean_L0
   - Predict absorption occurrence: alpha > 1 (in the symmetric capacity case) -- parent is absorbed iff it provides no unique information beyond the child
2. Validate the phase transition prediction: plot absorption rate vs. niche overlap across feature pairs, test for a threshold effect at alpha = 1

**Stage 5: Validation and Integration**

1. Validate SGWA on the first-letter spelling task against Chanin et al. gold-standard probe-based metric: report Spearman rho, precision@50%, AUROC
2. Compare SGWA with the current proposal's ITAC and conditional cosine similarity methods on the same validation task
3. If SGWA validates (rho > 0.5), deploy on cross-domain tasks (city-country, city-continent, city-language) where probe-based metrics are also available for comparison
4. Deploy SGWA on domains where no probes exist (open-ended absorption audit) -- this is the killer application that no existing method can do

### Cross-domain insight

The key transplanted principle is from spectral graph signal processing: multi-resolution analysis on a graph reveals hierarchical structure at different scales, and features that merge at a specific scale indicate a dependency relationship at that level of abstraction. This structural correspondence holds because:

1. The SAE decoder cosine similarity matrix IS a weighted graph whose structure encodes feature relationships
2. Hierarchical feature dependencies (parent-child) create specific patterns in this graph: high local similarity within feature families, with cross-family connections at higher hierarchy levels
3. Absorption pairs are specifically features that SHOULD be at different hierarchy levels but are collapsed by the sparsity constraint -- they appear as merging events at the absorption scale in the wavelet decomposition
4. The ecological niche theory completes the picture: the absorption scale corresponds to the hierarchy level where the niche overlap exceeds the critical threshold for competitive exclusion

### Experimental Plan

| Experiment | Description | Config | Time | Purpose |
|---|---|---|---|---|
| Pilot: SGWA computation | Compute SGWT on Gemma 2 2B L12 16k decoder matrix | 16k features, 10 scales, Chebyshev K=30 | 10 min | Verify computational feasibility and examine wavelet coefficient distributions |
| Exp A1: SGWA validation (first-letter) | Compare SGWA absorption scores with Chanin et al. probe-based metric on 26 letters | L12, 16k | 30 min | Primary validation (target: rho > 0.5) |
| Exp A2: SGWA vs ITAC comparison | Head-to-head comparison of SGWA, ITAC, conditional cosine + firing rate | L12, 16k | 30 min | Determine which unsupervised method best correlates with gold standard |
| Exp A3: Cross-domain deployment | Apply SGWA to city-country, city-continent, city-language with probe comparison | L12, 16k | 45 min | Cross-domain validation |
| Exp A4: Open-ended audit | Apply SGWA to all 16k features without any probes; characterize absorption landscape | L12, 16k | 30 min | Demonstrate the unsupervised killer application |
| Exp B1: Niche overlap measurement | Compute pairwise niche overlap for all first-letter feature pairs using probe data | L12, 16k | 20 min | Quantify ecological theory variables |
| Exp B2: Phase transition test | Plot absorption rate vs niche overlap; test for threshold at alpha=1 | L12, 16k | 15 min | Ecological theory prediction |
| Exp B3: Carrying capacity sweep | Vary SAE width (16k, 32k, 65k) and measure absorption vs. K | L12, multiple widths | 45 min | Test carrying capacity prediction |
| Ablation 1 | Graph threshold sensitivity (90th-99th percentile) | L12, 16k | 15 min | Robustness |
| Ablation 2 | Wavelet kernel choice (Mexican hat, heat, difference) | L12, 16k | 15 min | Robustness |
| Ablation 3 | Number of wavelet scales (5-20) | L12, 16k | 15 min | Robustness |

**Total**: approximately 4.5 hours of GPU time. Each task under 1 hour.

### Resource Estimate

- **Model**: Gemma 2 2B (~5GB VRAM)
- **SAEs**: Gemma Scope pre-trained (HuggingFace download). No training required.
- **Software**: SAELens (>= 4.0), TransformerLens, PyGSP (graph signal processing library for SGWT), scipy.sparse (sparse Laplacian), scikit-learn (clustering), sae-spelling (Chanin metric)
- **Compute**: Single A100. All experiments are inference-only.
- **Total GPU-hours**: approximately 4-5 hours
- **Training cost**: Zero. Entirely training-free analysis.

### Risk Assessment

**Risk 1: SGWA wavelet coefficients do not discriminate absorption pairs (rho < 0.3)**
- Probability: 25%
- Mitigation: Fall back to the current proposal's ITAC + conditional cosine approach, which is already specified. SGWA becomes a "bonus" contribution rather than the centerpiece. Report the negative result for spectral methods.

**Risk 2: Graph too dense after thresholding (wavelets don't reveal meaningful scale structure)**
- Probability: 20%
- Mitigation: Try k-nearest-neighbor graph construction (k=10, 20, 50) instead of threshold. Try SGWT on the co-activation graph (much sparser) instead of the cosine similarity graph.

**Risk 3: Ecological niche overlap does not predict absorption rate (rho < 0.2)**
- Probability: 30%
- Mitigation: Report as important negative result -- absorption may depend on variables not captured by the ecological analogy. The SGWA method stands independently of the ecological theory. Fall back to purely empirical characterization.

### Novelty Claim

1. **First application of spectral graph wavelet transforms to SAE decoder analysis for ANY purpose**. Prior work (Geometry of Concepts) uses basic spectral clustering on decoder cosine similarity. We introduce full multi-resolution wavelet decomposition, which provides principled scale selection, scale-dependent feature signatures, and absorption-scale identification.

2. **First quantitative theory of absorption based on feature competition dynamics**. The Lotka-Volterra / ecological niche overlap framework provides closed-form predictions (absorption occurs when alpha > 1, severity scales with alpha / K) that are directly testable. Prior theoretical accounts (Chanin et al. toy model, unified SDL theory) explain that absorption occurs but do not predict how severe it will be for specific feature pairs.

3. **First unsupervised absorption detection with principled scale selection**. The current ITAC pipeline requires arbitrary threshold sweeps. SGWA determines the relevant scales from the spectral properties of the decoder graph Laplacian -- no manual parameter tuning beyond the initial graph construction threshold (which is ablated).

4. **Integration of SGWA into the existing cross-domain proposal as an upgrade to the unsupervised detection pipeline**. This does not replace the current proposal's structure but provides a more principled implementation of the multi-resolution decoder geometry analysis (Stage 2a in the current proposal).

**What is explicitly NOT novel** (and we do not claim):
- The SGWT itself (Hammond et al.)
- The ecological competition framework (Lotka-Volterra, 1920s)
- The use of decoder cosine similarity for SAE analysis (Michaud et al., 2025)
- The existence or characterization of feature absorption (Chanin et al., 2024)
- The cross-domain absorption measurement approach (from the current proposal)
