# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Chanin et al., 2024. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507, NeurIPS 2025]** — The canonical paper defining feature absorption; proves via toy model that hierarchical co-occurrence causes SAEs to learn gerrymandered latents with false negatives; introduces the integrated-gradients-based absorption rate metric; validates on hundreds of Gemma Scope, Llama 3.2, Qwen2 SAEs; finds absorption in every architecture tested. Critical baseline for any absorption study.

2. **[Karvonen et al., 2025. SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532]** — Establishes 8-metric evaluation suite (including absorption) across 200+ SAEs; reveals that proxy metrics (CE loss, sparsity) do not reliably predict practical performance; hierarchical architectures outperform others on absorption by 30–40% margins. Standard evaluation framework for comparing absorption across architectures.

3. **[Bussmann et al., 2025. Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547, ICML 2025]** — Proposes nested SAE dictionaries that train hierarchical feature families simultaneously; reduces absorption; superior on sparse probing and concept erasure; but inner levels suffer from feature hedging. Shows hierarchy-aware training can reduce absorption.

4. **[Chanin, Dulka, & Garriga-Alonso, 2025. Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756]** — Identifies "feature hedging" as the opposite failure regime to absorption (too narrow → merge correlated features; hierarchy → absorption); proposes balanced Matryoshka SAE. Critical framing: hedging and absorption form a duality under SAE capacity constraints.

5. **[Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033]** — Enforces pairwise orthogonality on decoder vectors; reduces absorption 65%, composition 15%; discovers 9% more distinct features. Structural orthogonality as a partial remedy.

6. **[Li et al., 2025. Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training. arXiv:2510.08855]** — Per-latent importance scoring with adaptive temporal masking; achieves absorption score 0.0068 vs. TopK 0.1402 on Gemma-2-2B; best reported absorption numbers. Shows dynamic training interventions can dramatically reduce absorption.

7. **[Shu et al., 2025. A Unified Theory of Sparse Dictionary Learning: Piecewise Biconvexity and Spurious Minima. arXiv:2512.05534]** — First unified theoretical framework casting all SDL variants as piecewise biconvex optimization; provides principled explanation for absorption as spurious local minima; proposes feature anchoring. The theoretical anchor for understanding absorption mechanics.

8. **[Li, Michaud, Baek, Engels, Sun, Tegmark, 2024. The Geometry of Concepts: Sparse Autoencoder Feature Structure. arXiv:2410.19750]** — Analyzes SAE feature co-occurrence structure at three scales (atomic/brain/galaxy); finds co-occurring features cluster spatially; measures adjusted mutual information between co-occurrence-based and geometry-based clusters. Directly relevant: co-occurrence statistics are a structural cause of absorption.

9. **[Hindupur, Lubana, Fel, & Ba, 2025. Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry. arXiv:2503.01822, NeurIPS 2025]** — Shows each SAE architecture imposes structural assumptions that bias what concepts it can detect; introduces bilevel optimization framing; SAEs are not interchangeable—architecture shapes what can be seen. Critical: absorption is not just a training artifact but reflects a structural mismatch between SAE assumptions and concept geometry.

10. **[Ivanov et al., 2026. Spectral Superposition: A Theory of Feature Geometry. arXiv:2602.02224]** — Develops spectral theory for feature geometry via the frame operator F = WW^T; analyzes how features allocate norm across eigenspaces; provides new theoretical tools for understanding SAE representational structure. Opens a new mathematical lens on why certain feature hierarchies cause absorption.

11. **[Narayanaswamy et al., 2026. Improving Robustness in Sparse Autoencoders via Masked Regularization. arXiv:2604.06495]** — Token masking during training disrupts co-occurrence patterns; reduces absorption OOD robustness; very recent (April 2026); the closest prior art to a co-occurrence-informed mitigation. Key competitor for cross-domain ideas.

12. **[DeepMind Safety Research Team, 2025. Negative Results for Sparse Autoencoders On Downstream Tasks. Medium blog]** — SAE probes fail catastrophically at detecting harmful intent OOD while dense linear probes succeed; feature absorption is a key culprit; team publicly deprioritized SAE research. Provides the highest-stakes motivation: absorption is not an academic curiosity but a blocker for AI safety applications.

### Landscape Summary

The field has established that feature absorption is universal across SAE architectures and models—it is an inherent consequence of optimizing for sparsity when the data-generating process exhibits feature hierarchies. The canonical metric (Chanin et al.) is limited to the first-letter spelling task, requiring pre-specified probe directions. No study has demonstrated a practical method for detecting absorption without advance knowledge of which features to look for.

The theoretical picture is partially established: absorption arises at spurious local minima of the piecewise biconvex SDL objective when a child feature's sparsity gain can subsidize the parent feature's recall cost. However, no closed-form prediction of absorption rate as a function of SAE hyperparameters and data statistics exists. Architectural mitigations (ATM-SAE achieving 0.0068 absorption score vs. TopK's 0.1402) show dramatic improvement is possible, but none of the proposed remedies have been evaluated under a unified benchmark with controlled conditions.

A crucial gap is identified from two directions simultaneously: (1) The "Geometry of Concepts" paper shows that co-occurrence statistics have strong mutual information with feature geometry—features that co-occur cluster together in representation space. (2) The "Projecting Assumptions" paper shows that SAE architectures are biased by their structural assumptions about concept geometry. Together, these imply that absorption is fundamentally about a mismatch between the co-occurrence structure of the data and the geometric assumptions baked into SAE design. No study has directly operationalized this connection to predict or detect absorption from co-occurrence statistics alone—without requiring ground-truth probes. This is the critical gap the most promising novel idea will target.

---

## Phase 2: Initial Candidates

### Candidate A: Absorption Prediction via Feature Co-occurrence Topology

- **Hypothesis**: The absorption rate of SAE latent i with respect to a ground-truth feature concept f can be predicted (without any probe training) from the co-occurrence graph topology of SAE latents: specifically, latents that form "absorbing stars" (one high-degree hub latent connected to many low-frequency leaf latents in the co-occurrence graph) will show high absorption fractions. The topological structure of the co-occurrence graph is a sufficient statistic for absorption severity, enabling unsupervised absorption detection.

- **Cross-domain insight**: This transplants ideas from **network epidemic spreading models** (specifically the "superspreader" phenomenon in network epidemiology—nodes with disproportionately high degree in contact graphs cause disproportionate infection spread). In SAE co-occurrence networks, a high-frequency "generalist" latent (analogous to a superspreader node) that co-fires with many specific latents is the signature of a latent that absorbs the parent feature. The graph-theoretic framing (degree centrality, k-core decomposition, PageRank of co-occurrence graphs) provides precise, computable predictions.

- **Evidence for**: (a) The Geometry of Concepts (arXiv:2410.19750) directly shows co-occurrence structure has high mutual information with geometric structure—the structural cause of absorption. (b) Chanin et al. describe absorption as occurring when a child latent with high activation rate absorbs the parent latent's firing; "high activation rate" maps directly to "high degree node" in the co-occurrence graph. (c) Token entropy is already known to be a strong predictor of feature interpretability (lower frequency → more interpretable), consistent with a graph-degree-based absorption predictor. (d) Network epidemic theory has well-developed metrics (degree, betweenness centrality, k-core) that can be efficiently computed on large co-occurrence graphs.

- **Novelty estimate**: 9/10 — No existing paper uses co-occurrence graph topology (rather than probe-based metrics) to predict or detect feature absorption. The Geometry of Concepts studies co-occurrence structure but does not connect it to absorption. The masked regularization paper (arXiv:2604.06495) disrupts co-occurrence during training but does not use co-occurrence structure as a diagnostic tool.

### Candidate B: Absorption Scaling Law from Rate-Distortion Theory

- **Hypothesis**: Feature absorption rate follows a quantitative scaling law of the form AR(W, L0, α) ∝ (W/N)^{-β} · L0^γ · α^δ, where W is SAE width, L0 is mean sparsity, α is the hierarchy depth (ratio of parent-feature frequency to child-feature frequency), and N is the number of tokens in the corpus. The exponents β, γ, δ can be estimated from a sweep of existing Gemma Scope SAEs (which span widths 1k–1M) without any new training. This provides the first closed-form absorption prediction formula.

- **Cross-domain insight**: Rate-distortion theory from information theory establishes that the minimum distortion achievable at a given compression rate follows a power-law tradeoff. In SAEs, the "rate" is L0 (average active features per token) and the "distortion" is reconstruction error. The SAE scaling laws paper (OpenAI, arXiv:2406.04093) already found clean power laws between SAE width and CE loss. Extending this perspective: absorption rate is a form of "feature distortion" in the hierarchy domain—the number of child features that successfully "save L0 budget" by absorbing their parents. Rate-distortion theory predicts this tradeoff should also follow clean power laws.

- **Evidence for**: (a) SAE scaling laws (arXiv:2406.04093) demonstrate empirically that CE loss follows power laws in SAE width and L0. (b) Understanding SAE scaling with feature manifolds (arXiv:2509.02565) extends scaling analysis to manifold-structured features, providing theoretical grounding. (c) Chanin et al. observe empirically that wider/more-sparse SAEs show higher absorption—directional consistency with a scaling law. (d) Gemma Scope provides 400+ SAEs spanning widths 1k–1M on the same model, enabling a clean empirical measurement of the absorption-vs-width curve using the existing absorption metric.

- **Novelty estimate**: 8/10 — No paper has attempted to derive quantitative absorption scaling laws. The directional observation ("wider SAEs show higher absorption") exists but has not been quantified into a predictive formula. The key gap (Gap 1 in the literature survey) is precisely this.

### Candidate C: Absorption as Multi-Label Implication Failure—a Label-Implication-Aware Decoder Regularization

- **Hypothesis**: Feature absorption is structurally analogous to the "label absorption" failure mode in hierarchical multi-label classification (HMC), where a specific child label fires but the implied parent label does not. In HMC, this is addressed by imposing label-implication constraints during training. Translating this to SAEs: if we can infer a partial feature hierarchy from the data (using mutual information between SAE latents' activation patterns), then adding a "parent-child consistency loss" that penalizes child activation without parent activation will reduce absorption in a training-free-compatible way (applied during fine-tuning of the SAE encoder, keeping the decoder frozen). Hypothesis: the label-implication-inspired consistency loss reduces absorption rate by >30% while maintaining ≥95% reconstruction quality on Gemma Scope SAEs.

- **Cross-domain insight**: The hierarchical multi-label classification (HMC) literature has extensively studied "label implication" constraints: if a document is labeled "terrier" it should also be labeled "dog" (the parent). The core algorithmic solution is either (a) adding an implication-consistency loss that penalizes predictions violating the hierarchy, or (b) propagating labels up the hierarchy during inference. Both approaches are directly translatable to SAEs: (a) detecting implied parent latents from child activation patterns and penalizing missing parent activations, or (b) "parent propagation" post-processing that activates implied parent latents during SAE inference without changing the decoder.

- **Evidence for**: (a) The HTC literature shows that hierarchical consistency losses dramatically improve label recall without hurting precision (analogous to reducing false negatives without adding false positives). (b) Chanin et al. explicitly characterize absorption as a false-negative problem for parent features when child features fire. (c) OrtSAE's 65% absorption reduction shows that structural regularization during training can be effective. (d) The Masked Regularization paper (arXiv:2604.06495) shows that training interventions that disrupt co-occurrence patterns reduce absorption—our approach would do the inverse: reinforce the co-occurrence implication structure rather than disrupting it.

- **Novelty estimate**: 7/10 — The HMC analogy has not been applied to SAEs before. However, the masked regularization paper (arXiv:2604.06495) has already partially explored the "disrupt co-occurrence during training" direction. The key novelty here is using the co-occurrence structure prescriptively (to identify and enforce implication constraints) rather than destructively (masking to break spurious co-occurrences). The specific mechanism (parent-child consistency loss derived from mutual information between latent activations) is new.

---

## Phase 3: Self-Critique

### Against Candidate A

- **Prior work attack**: Searched for "SAE co-occurrence graph absorption detection unsupervised" and "SAE feature graph topology interpretability"—no papers found that directly use co-occurrence graph topology for absorption detection. The closest is the Geometry of Concepts paper (arXiv:2410.19750) which studies co-occurrence at the spatial/geometric level but does not connect to absorption. The masked regularization paper uses co-occurrence implicitly but only to justify token masking, not as a detection tool. The prior work attack passes: this is genuinely novel territory.

- **Methodological attack**: The proposed "absorbing star" topology pattern (high-degree hub connected to many low-frequency leaves) could arise for non-absorption reasons—e.g., a genuine hub feature that legitimately co-fires with many other features (such as a syntax/punctuation feature). Without ground-truth probes, false positives in the absorption prediction are unavoidable. The correlation between graph topology and absorption rate needs to be validated against the canonical Chanin et al. absorption metric, which requires probe labels for a subset of the features.

- **Theoretical attack**: The epidemic superspreader analogy, while evocative, may not hold quantitatively. In epidemic spreading, high-degree nodes are superspreaders because of the random contact structure; in SAEs, high-degree co-occurrence nodes are high-frequency features that co-fire with many others due to the data distribution, not a random contact structure. The mechanistic link between hub degree and absorption may be correlational rather than causal. Furthermore, absorption is direction-specific (a child absorbs the parent, not vice versa), which is not captured by undirected co-occurrence graph degree alone.

- **Scalability attack**: Computing full pairwise co-occurrence statistics for a 16k-latent SAE requires O(16k²) = 256M comparisons—manageable, but for 1M-latent SAEs this becomes infeasible. The claim of "unsupervised absorption detection" would need to be validated on large SAEs; at million-latent scale, sparse approximations of the co-occurrence graph would be needed, introducing approximation errors.

- **Verdict**: STRONG — The prior work attack passes cleanly. The methodological and theoretical concerns are fixable: directed co-occurrence (using conditional activation patterns P(f_parent | f_child)) rather than undirected degree would address the directionality concern, and a partial validation against ground-truth Chanin et al. metrics on a subset of features is sufficient. The scalability concern is real but can be bounded by noting the project focuses on 16k–65k width SAEs (Gemma Scope), not million-latent SAEs.

### Against Candidate B

- **Prior work attack**: Searched for "SAE absorption scaling law width L0 formula quantitative"—no papers found that provide closed-form scaling predictions for absorption. The Chanin et al. paper's observation ("wider SAEs show higher absorption") is empirical and directional; no formula exists. The scaling laws paper (arXiv:2406.04093) focuses on CE loss and reconstruction error, not absorption. The gap is real.

- **Methodological attack**: The proposed scaling law uses Gemma Scope SAEs which span widths 1k–1M, but these SAEs were trained with different hyperparameters (L0 targets vary, training duration varies). A confounded measurement of absorption vs. width could produce a spurious apparent scaling law. Carefully controlling for L0 while varying width, or using the many SAEs of the same width but different L0 targets available in Gemma Scope, would be needed.

- **Theoretical attack**: Rate-distortion theory applies to a well-defined source and distortion measure. The "distortion" here (absorption rate) is not a standard distortion measure in the information-theoretic sense; it depends on the specific feature hierarchy structure in the data, which varies across domains. The analogy to rate-distortion curves may be too loose to yield a clean functional form.

- **Scalability attack**: Clean scaling laws typically require many data points spanning 2–3 orders of magnitude. For absorption, the measurement at each width requires running the full Chanin et al. metric (k-sparse probing + integrated gradients), which is computationally expensive. Covering widths 1k, 4k, 16k, 65k, 131k, 524k, 1M requires ~7 different SAEs × 26 letters × multiple layers, potentially requiring many GPU-hours—borderline for the ≤1hr/experiment constraint.

- **Verdict**: MODERATE — The prior work attack passes (genuine gap). The confounding and rate-distortion concerns are real: the scaling law may not be clean due to confounded training across widths. The computational concern can be addressed by using pre-computed Gemma Scope SAEs and restricting to a small number of carefully chosen layers. The risk is that the result may be descriptive ("wider SAEs absorb more, approximately as a power law") rather than predictive (explaining the exponents from first principles). This is still publishable but scientifically weaker than Candidate A.

### Against Candidate C

- **Prior work attack**: Searched specifically for "SAE label implication hierarchy consistency loss absorption reduction"—no such paper exists. However, the masked regularization paper (arXiv:2604.06495) from April 2026 is very recent and addresses absorption via training intervention. It uses token masking to disrupt co-occurrence patterns, while this proposal uses detected hierarchy to enforce implication constraints. These are distinct mechanisms. Also, the ATM-SAE (arXiv:2510.08855) with absorption score 0.0068 is a very strong existing baseline—any new method must beat this.

- **Methodological attack**: The proposed approach requires "training-free compatible" application (fine-tuning the encoder with frozen decoder). But fine-tuning any SAE component constitutes training, which contradicts the project's "training-free analysis" constraint in the spec (see context file). This is a serious implementation conflict. If the project is strictly analysis-only, Candidate C would require a different realization (e.g., post-hoc "parent propagation" inference without any training).

- **Theoretical attack**: The HMC analogy assumes we can correctly identify the feature hierarchy from activation co-occurrence statistics. However, absorption itself corrupts the observable co-occurrence patterns—the parent feature fires less often precisely because it is being absorbed. Identifying the "true" hierarchy from corrupted co-occurrence statistics is a chicken-and-egg problem: you need to know the hierarchy to identify absorption, but you need to identify absorption to recover the true hierarchy.

- **Scalability attack**: The parent-child consistency loss requires iterating over all latent pairs that might have implication relationships, scaling quadratically with dictionary size. For large dictionaries, this is computationally impractical as a training objective.

- **Verdict**: WEAK — The project spec explicitly specifies "training-free analysis" as the primary constraint; any method requiring SAE fine-tuning or retraining is out of scope. The chicken-and-egg problem in hierarchy identification is also a fundamental theoretical obstacle. While the HMC analogy is intellectually interesting, this direction does not fit the project's constraints and has a fatal methodological flaw in the hierarchy identification step. **Drop.**

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C** dropped because: (a) requires fine-tuning SAEs, violating the project's training-free analysis constraint; (b) the fundamental chicken-and-egg problem (absorption corrupts the co-occurrence signal needed to identify the hierarchy) makes hierarchy identification unreliable; (c) the ATM-SAE (absorption score 0.0068) and masked regularization are strong existing baselines that would be hard to beat with this approach; (d) the project context explicitly states "research method focused on training-free analysis of existing pre-trained SAEs."

### Strengthened Ideas

**Candidate A (Co-occurrence Topology → Unsupervised Absorption Detection):**
Strengthened with the following refinements:
1. Use **directed co-occurrence** (conditional activation probability P(f_j fires | f_i fires) rather than undirected co-occurrence count) to capture the directionality of absorption (child absorbs parent, not vice versa). This addresses the theoretical directionality concern.
2. Define the "absorbing hub" pattern more precisely: latent i is a potential absorbing latent if ∃ latent j with high frequency such that P(j fires | i fires) > τ_high AND P(i fires | j fires) < τ_low. This is the topological signature of "i is a specific child of high-frequency parent j."
3. Validate the unsupervised prediction against ground-truth Chanin et al. absorption metrics on a subset of Gemma Scope 16k SAEs (where probe-based metrics are available). This provides validation without requiring full probe training for all features.
4. Additional search found **The Geometry of Concepts** (arXiv:2410.19750) specifically measures "adjusted mutual information between co-occurrence clusters and geometry clusters"—this provides the exact mathematical scaffold for quantifying how well co-occurrence topology predicts absorption patterns.

**Candidate B (Absorption Scaling Laws):**
Strengthened with the following refinements:
1. Restrict to Gemma Scope SAEs at a fixed model and layer, varying only width while using pre-specified L0 targets that are matched (e.g., L0 ≈ 50 across all widths). This controls the confounding variable.
2. Reframe: rather than claiming a "closed-form formula from first principles," claim "empirical scaling law for absorption" (analogous to how the OpenAI paper claimed empirical scaling laws for CE loss before theoretical derivation). This is more defensible.
3. Integrate with Candidate A: the scaling law study and the co-occurrence topology study are complementary—the co-occurrence hub degree (from Candidate A) should also follow a scaling law as dictionary size grows.

### Additional Evidence Found
- **arXiv:2509.02565 (Understanding SAE Scaling in the Presence of Feature Manifolds)**: Provides formal analysis of SAE scaling behavior for manifold-structured features; observes that loss can decrease by finely "tiling" common feature manifolds at the expense of rarer features—directly analogous to absorption (child features tile the parent's activation region). This provides theoretical grounding for the scaling law prediction in Candidate B.
- **arXiv:2602.02224 (Spectral Superposition)**: Introduces the frame operator F = WW^T and its eigenspectrum as a tool for analyzing SAE feature geometry. The eigenspectrum of the co-occurrence-weighted frame operator could provide a principled measure of absorption risk that is more interpretable than raw co-occurrence counts.
- **arXiv:2503.01822 (Projecting Assumptions)**: Shows SAE architecture imposes structural assumptions about concept geometry. The co-occurrence topology approach (Candidate A) provides the empirical data about concept geometry that the SAE fails to adequately represent—this connection can be explicitly made in the paper.

### Selected Front-Runner
**Candidate A: Absorption Prediction via Feature Co-occurrence Topology** is selected as the front-runner for the following reasons:

1. **Addresses the most critical gap**: Gap 7 (no unsupervised absorption detection) is the most impactful missing piece. All other studies depend on pre-specified probe directions; an unsupervised method would democratize absorption analysis across thousands of SAE latents.

2. **Training-free and analysis-only**: Entirely compatible with the project's constraint. Requires only forward passes through existing SAEs (Gemma Scope) and co-occurrence statistics computation—no training.

3. **Unique conceptual framing**: The network epidemiology / superspreader analogy provides a genuinely novel lens that no existing paper has applied to SAE absorption.

4. **Highly practical output**: Produces a method that the community can immediately apply to any SAE to identify likely absorption candidates, prioritizing which latents to scrutinize further.

5. **Integrates with existing benchmarks**: Can be validated against Chanin et al.'s canonical metric on Gemma Scope, and the co-occurrence structure data is already analyzable via the Geometry of Concepts framework.

6. **Candidate B is incorporated as a supporting experiment**: The scaling law analysis can be reported as a secondary result within the same paper, using the co-occurrence hub degree statistics across different SAE widths.

---

## Phase 5: Final Proposal

### Title
Absorption Fingerprints: Detecting SAE Feature Absorption from Co-occurrence Network Topology Without Ground-Truth Probes

### Hypothesis
The directed co-occurrence network of SAE latent activations contains sufficient information to predict feature absorption at the individual latent level without any ground-truth probes or pre-specified feature hierarchies. Specifically: (1) latents that exhibit the "hub-to-leaf directed asymmetry" pattern in their co-occurrence neighborhood—latent i has low activation frequency relative to latent j, yet P(j fires | i fires) is high while P(i fires | j fires) is low—will be identified as absorbed features with precision > 0.70 and recall > 0.60 against the canonical Chanin et al. absorption metric on Gemma Scope SAEs. (2) The degree of this hub-leaf asymmetry will correlate with measured absorption severity across SAE widths (1k–131k) following an approximately log-linear relationship.

### Motivation
Feature absorption is the most operationally consequential failure mode in SAE-based mechanistic interpretability: it causes false negatives in feature activation that undermine causal circuit analysis, feature steering, and safety-relevant feature detection. DeepMind's 2025 deprioritization of SAE research for harmful intent detection is the starkest evidence that absorption blocks practical safety applications.

The canonical detection method (Chanin et al., arXiv:2409.14507) requires pre-training logistic regression probes for each target concept—a supervised procedure that presupposes knowledge of which features to scrutinize. This creates a fundamental scalability barrier: safety researchers need to detect absorption in features they did not know to look for in advance (e.g., a deception-related feature being absorbed into a context feature for roleplay scenarios).

The critical insight motivating this proposal comes from connecting two independent threads in the literature: (a) The Geometry of Concepts (arXiv:2410.19750) demonstrates that co-occurring SAE features cluster geometrically, establishing a strong structural relationship between co-occurrence statistics and the feature representation space. (b) Projecting Assumptions (arXiv:2503.01822, NeurIPS 2025) shows that SAE architectures impose structural assumptions about concept geometry that shape what they can detect—absorption is a symptom of this mismatch. Together, these papers imply that the co-occurrence structure of the data contains the "ground truth" about feature hierarchy that the SAE fails to adequately represent. An unsupervised absorption detector built on this co-occurrence structure would therefore detect exactly the gap between what the data implies and what the SAE represents.

The cross-domain inspiration is network epidemiology's "superspreader" phenomenon: in contact networks, nodes with high degree and high betweenness centrality disproportionately spread infection—analogously, in SAE co-occurrence networks, high-frequency "generalist" features that appear in many contexts are the typical absorption receptors that subsume the activation budget of rarer, more specific parent features.

### Method

**Step 1: Build the directed co-occurrence graph G**
- Load pre-trained Gemma Scope SAEs (residual stream, layers 12/20, widths 1k/4k/16k/65k) via SAELens.
- Pass 100k tokens from OpenWebText through the model; record all SAE latent activations.
- For each ordered pair (i, j), compute: (a) f_i = activation frequency of latent i; (b) c_{i→j} = P(j fires | i fires) = joint activation count / activation count of i.
- Construct the directed weighted co-occurrence graph G = (V, E) where V = latents, edge weight w_{i→j} = c_{i→j} · (f_j / f_i) (the "absorption propensity" of j absorbing i: high when i fires rarely relative to j, and i tends to co-occur with j).

**Step 2: Compute Absorption Fingerprint Scores (AFS)**
- For each latent i, compute AFS(i) = max_j [ w_{j→i} ] where j ranges over latents with f_j > k · f_i for a threshold k (default k=3, i.e., potential absorbers must be at least 3× more frequent).
- High AFS(i) indicates that i tends to fire in contexts dominated by a much more frequent latent j, the signature of absorption.
- Also compute graph-theoretic features: in-degree in G_threshold (edges above w > τ), clustering coefficient, and the "hub dominance" score (ratio of max absorber strength to total neighborhood strength).

**Step 3: Validate against canonical absorption metric (partial ground truth)**
- Run the Chanin et al. first-letter absorption metric on 5 letters ('A', 'E', 'I', 'O', 'T') for each SAE configuration. This produces ground-truth absorbed vs. non-absorbed latents for those letters.
- Compute AUROC, precision, and recall for predicting which latents are absorption-positive using AFS scores.
- Report calibration curve: does AFS score rank absorbed latents higher than non-absorbed latents?

**Step 4: Cross-architecture absorption characterization**
- Apply AFS analysis to SAEs across 5 architectures available in SAEBench: TopK, Gated, JumpReLU, Matryoshka, BatchTopK.
- Report AFS distribution statistics (mean, 95th percentile, fraction of latents with AFS > τ) for each architecture.
- Correlate AFS statistics with SAEBench absorption score—does AFS serve as a cheap proxy for the expensive SAEBench metric?

**Step 5: Absorption scaling law analysis**
- Measure AFS statistics across SAE widths 1k, 4k, 16k, 65k, 131k (Gemma Scope, matched L0 ≈ 50).
- Fit log(AFS_{mean}) vs. log(W) and log(AFS_{mean}) vs. log(L0) regressions.
- Report empirical scaling exponents and confidence intervals.

**Step 6: Novel cross-domain characterization**
- Apply AFS to identify absorption in domains beyond the first-letter task: use existing Neuronpedia feature labels (entity type hierarchies, topic hierarchies) as partial ground truth.
- Report whether AFS-identified absorption candidates match conceptually expected hierarchical relationships (e.g., "France" absorbed into "European country" feature).

### Experimental Plan

| Experiment | Resources | Time | Goal |
|---|---|---|---|
| Pilot: Build G for Gemma-2-2B layer 12, 16k width, 100k tokens | 1 GPU, ~10 min | 10 min | Verify co-occurrence graph computation is feasible and AFS values are interpretable |
| Pilot: Validate AFS on 3 letters vs. Chanin metric on 16k SAE | 1 GPU, ~20 min | 30 min | Confirm AFS predicts absorption better than random (AUROC > 0.60) |
| Main: AFS computation for 4 layers × 4 widths of Gemma Scope | 1 GPU, ~2 hr total | 4× 30 min runs | Coverage for scaling law analysis |
| Main: Chanin metric validation on 5 letters × 4 SAEs | 1 GPU, ~1 hr total | 4× 15 min runs | Ground-truth validation set |
| Cross-architecture: AFS for 5 architectures from SAEBench | 1 GPU, ~2 hr total | 5× 25 min runs | Architecture-comparison absorption fingerprinting |
| Scaling law fit: regression analysis | CPU, ~5 min | 5 min | Scaling law exponents |
| Cross-domain: entity/topic hierarchy absorption via Neuronpedia labels | CPU, ~1 hr | 1 hr | Generalization evidence |

**Baselines**: Random latent ranking; activation frequency ranking alone (without directionality); Chanin et al. full absorption metric (treated as gold standard). Target: AFS achieves AUROC > 0.70 on validation set.

**Falsification criteria**: If AFS AUROC < 0.55 on first-letter validation set, the co-occurrence topology does not predict absorption better than chance; the hypothesis is refuted. If the scaling law R² < 0.80, no clean power law exists and the scaling hypothesis is weakened to a descriptive trend.

### Resource Estimate
- **Models**: Gemma-2-2B (pre-loaded via Gemma Scope, no training). GPT-2 small SAEs as secondary validation (faster runs).
- **GPU**: ~8–10 GPU-hours total (1 × A100 or 2 × A40); all experiments are forward passes, no gradient computation.
- **Storage**: ~20 GB for activation caches (reusable across experiments).
- **Timeline**: 
  - Pilot experiments: 1 hour
  - Main AFS computation + Chanin validation: 3–4 hours
  - Analysis, fitting, cross-domain: 2 hours
  - Total: ≤8 GPU-hours, ≤1 day wall-clock (fits within ≤1 hr per experiment task constraint by splitting into subtasks)

### Risk Assessment

**Risk 1: AFS precision is too low for the method to be useful (false positive rate too high)**
- Mitigation: The AFS score is designed as a ranking/prioritization tool, not a binary classifier. AUROC ≥ 0.65 is sufficient to establish the method's utility even if precision at a given threshold is modest. Additionally, adding a secondary filter (require high cosine similarity between decoder vectors of the candidate absorber-absorbed pair) should improve precision—the absorption relationship requires the absorbing latent to span the absorbed feature's direction.

**Risk 2: Co-occurrence statistics from 100k tokens are too noisy for rare features**
- Mitigation: Use 1M tokens instead of 100k for the main experiment (still feasible as forward-pass-only). Additionally, focus validation on features with at least 50 activations in the dataset to ensure reliable co-occurrence estimates. Gemma Scope's activation caches can be reused to avoid recomputation.

**Risk 3: The "directed asymmetry" pattern reflects legitimate data structure, not SAE failure**
- Mitigation: This is the key confound. Distinguish genuine hierarchical absorption (where a latent "should" have fired but didn't due to an absorber) from a latent that legitimately never fires when a co-occurring latent fires (e.g., two mutually exclusive context features). Use the cosine similarity between decoder vectors as a discriminator: true absorption requires the absorbed latent's decoder to be approximately in the span of the absorber's decoder (since the absorber must encode the same information). Add this geometric filter: only flag pairs where decoder cosine similarity > 0.1.

### Novelty Claim
What is new: This is the first paper to propose an unsupervised, training-free method for detecting feature absorption in SAEs based on co-occurrence network topology—without requiring pre-specified probe directions or known feature hierarchies. Specifically:
1. The directed co-occurrence graph construction and the AFS scoring function are novel operationalizations not previously proposed in any existing SAE or mechanistic interpretability paper.
2. The network epidemiology "superspreader" framing (mapping absorption to hub-node dynamics in co-occurrence networks) is a genuinely new cross-domain analogy with no precedent in the SAE literature.
3. The connection between The Geometry of Concepts findings (co-occurrence geometry mutual information) and the Projecting Assumptions findings (SAE architecture biases) as jointly motivating the co-occurrence topology approach is a synthesis not made in prior work.
4. The empirical absorption scaling law analysis across Gemma Scope widths 1k–131k provides the first quantitative characterization of how absorption rate changes with SAE scale—addressing Gap 1 in the literature.

Supporting evidence that this has not been done: (a) The SAEBench absorption metric requires probe labels; (b) OrtSAE uses orthogonality of decoder vectors (not co-occurrence patterns) to reduce absorption; (c) Masked Regularization uses token masking to disrupt co-occurrence during training (not using co-occurrence topology for post-hoc detection); (d) The Geometry of Concepts studies co-occurrence geometry but does not apply it to absorption; (e) no arXiv search for "unsupervised absorption detection SAE co-occurrence" returned any relevant prior work.
