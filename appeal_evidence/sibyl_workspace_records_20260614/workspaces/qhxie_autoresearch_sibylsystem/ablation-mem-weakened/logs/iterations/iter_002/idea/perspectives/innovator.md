# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Chanin et al., 2024/2025. A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507]** — Foundational work defining feature absorption; establishes that hierarchical features cause absorption as a logical consequence of sparsity loss. Validates across hundreds of SAEs.

2. **[Gadgil et al., 2025. Ensembling Sparse Autoencoders. arXiv:2505.16077 / ICLR 2026 Under Review]** — Shows that independently initialized SAEs share only 30-42% of features. Boosting discovers complementary features and achieves >99% explained variance. Output-space ensemble averaging is equivalent to feature-space concatenation.

3. **[Chanin et al., 2025. Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756]** — Identifies hedging as a distinct failure mode from absorption, with an inherent trade-off: wider SAEs worsen absorption while narrower SAEs worsen hedging.

4. **[Rozell et al., 2008. Sparse coding via thresholding and local competition in neural circuits. IEEE Trans. Signal Processing]** — Foundational neuroscience-inspired sparse coding via Locally Competitive Algorithms (LCA). Neurons compete via one-way lateral inhibition where active nodes suppress weaker nodes based on receptive field similarity.

5. **[Korznikov et al., 2025. OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033]** — Enforces decoder orthogonality; reduces absorption by 65%. Evaluates on SAEBench downstream tasks (SCR, TPP, sparse probing, RAVEL).

6. **[Bussmann et al., 2025. Matryoshka Sparse Autoencoders. arXiv:2503.17547 / ICML 2025]** — Nested multi-level dictionaries reduce absorption from 0.49 to 0.05 but introduce hedging trade-off in inner levels.

7. **[Li et al., 2025. Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training. arXiv:2510.08855 / ICLR-W 2025]** — ~40% reduction in absorption via temporal EMA tracking of activation magnitudes/frequencies during training.

8. **[Wang et al., ICLR 2026. Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders. arXiv:2510.03659]** — Weak correlation (tau_b ~ 0.3) between interpretability and steering utility. Challenges the assumption that metric improvements translate to practical value.

9. **[Cui et al., 2025. On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963]** — Formal identifiability analysis proving conditions for recovering ground-truth features. Theoretical foundation for understanding when absorption is inevitable.

10. **[Jedryszek & Crook, 2026. Stable and Steerable Sparse Autoencoders with Weight Regularization. arXiv:2603.04198]** — L2 weight regularization produces a "core of highly aligned features" with bimodal cosine-similarity structure. Regularization increases cross-seed feature consistency and improves steering success.

11. **[MINERVA: Mutual Information Neural Estimation for Supervised Feature Selection, arXiv:2510.02610, 2025]** — Neural estimation of mutual information between features and targets with sparsity-inducing regularizers to handle feature redundancy. Captures higher-order feature interactions.

12. **[Attribution Projection Calculus: A Novel Framework for Causal Inference in Bayesian Networks, arXiv:2505.12094, 2025]** — AP-Calculus identifies "deconfounders" vs. "confounders" among intermediate nodes. Extends traditional do-calculus for supervised learning contexts with O(mnp) computational complexity.

### Landscape Summary

The SAE field is at a critical inflection point. The current project (iterations 1-8) has produced a null result study connecting absorption to downstream tasks, finding that raw steering and probing show no significant correlation with absorption, with only delta-corrected steering at layer 8 showing significance (r=-0.431, p=0.028). The core hypothesis is not strongly supported, yet the methodological contribution (baseline correction, precision-recall decomposition) has value.

Three cross-domain gaps emerge as opportunities for genuine novelty:

1. **No causal framework for absorption**: Despite absorption being a causal phenomenon (parent feature activation causes child feature suppression), no work has applied causal inference tools (do-calculus, intervention analysis) to formally characterize absorption's effect on downstream tasks.

2. **No information-theoretic reframing**: Absorption redistributes information from parent to child features. No work has quantified this redistribution using mutual information or rate-distortion theory to explain why absorption coexists with functional steering.

3. **No spectral analysis of decoder structure**: The decoder weight matrix W_dec defines a geometric structure in activation space. Spectral clustering on this structure could reveal communities of competing features that correspond to absorption hierarchies.

## Phase 2: Initial Candidates

### Candidate A: Causal Absorption Graph — Do-Calculus for SAE Feature Interventions

- **Hypothesis**: Feature absorption can be formally modeled as a causal mediation effect, where child feature activation mediates the effect of parent feature presence on downstream task performance. Using do-calculus (or AP-Calculus), we can compute the "natural direct effect" of parent features independent of child-mediated pathways, and show that this direct effect is what absorption suppresses.
- **Cross-domain insight**: From causal inference (Pearl's do-calculus, AP-Calculus 2025). The core insight: absorption is not merely correlation --- it is a causal mechanism where child feature firing (mediator) blocks parent feature contribution (treatment) to model output (outcome). The do-operator lets us ask "what would the output be if we forced the parent feature to fire while holding the child feature constant?" --- exactly the counterfactual that reveals absorption's true cost.
- **Evidence for**: Chanin et al. (2024) prove absorption is a causal consequence of sparsity loss under hierarchical features. AP-Calculus (2025) shows do-calculus extends to neural network feature attribution. The current project's null result on raw metrics may be because raw metrics measure total effect (direct + mediated), while absorption specifically suppresses the direct effect.
- **Novelty estimate**: 9/10 --- No work has applied causal inference formalism to absorption. This would be the first causal characterization of a major SAE failure mode.

### Candidate B: Information-Theoretic Absorption Decomposition

- **Hypothesis**: Absorption can be decomposed into an information-theoretic quantity: the mutual information I(parent; output | child) measures the information the parent contributes to the output that is NOT redundant with the child. Absorption reduces this conditional mutual information. We can quantify the "information loss" from absorption and show it predicts downstream task degradation better than the Chanin correlation metric.
- **Cross-domain insight**: From information theory (rate-distortion, information bottleneck). The core analogy: SAE features are a compressed representation of model activations. Absorption is a lossy compression artifact where hierarchical redundancy is exploited. Rate-distortion theory tells us that lossy compression can preserve task-relevant information while discarding redundant structure --- explaining why absorption coexists with functional steering (the "distortion" is in a subspace orthogonal to the task).
- **Evidence for**: Wang et al. (ICLR 2026) show weak correlation (~0.3) between interpretability and utility --- suggesting interpretability metrics capture information orthogonal to task performance. MINERVA (2025) shows neural MI estimation can capture higher-order feature interactions. The current project's finding that precision is invariant to absorption (H5 supported) is exactly what rate-distortion predicts: the "rate" (sparsity) is preserved, but the "distortion" (recall loss) is structured.
- **Novelty estimate**: 8/10 --- Information-theoretic analysis of SAEs exists but not specifically applied to absorption. The connection to rate-distortion is novel.

### Candidate C: Spectral Hierarchy Recovery from Decoder Communities

- **Hypothesis**: The decoder weight matrix W_dec (d_model x n_latents) defines a geometric structure where absorbed parent-child feature pairs form low-dimensional submanifolds. By applying spectral clustering to the decoder correlation graph and measuring the algebraic connectivity (Fiedler value) of feature communities, we can (1) detect absorption hierarchies without running the Chanin metric, and (2) quantify the "stiffness" of the hierarchy (how strongly parent and child are coupled).
- **Cross-domain insight**: From spectral graph theory and manifold learning. The core analogy: decoder vectors live on a high-dimensional sphere. Absorbed feature pairs are not randomly distributed --- they cluster along hierarchical submanifolds (parent at the "center," children along orthogonal directions). Spectral clustering detects these communities. The Fiedler value measures how well-connected the community is --- low Fiedler means the hierarchy is "stiff" (hard to separate), predicting high absorption.
- **Evidence for**: Jedryszek & Crook (2026) show L2 regularization produces a "core of highly aligned features" with bimodal cosine-similarity structure --- evidence that decoder vectors DO cluster. MetaSAE (2025) trains on decoder weight matrices to assess atomicity, showing decoder structure carries meaningful information. The current project's finding that absorption is layer-dependent suggests the decoder geometry varies with layer depth.
- **Novelty estimate**: 7/10 --- Spectral clustering on neural network weights exists, but applying it to SAE decoder geometry for absorption detection is novel. The Fiedler value as a predictor of absorption stiffness is genuinely new.

## Phase 3: Self-Critique

### Against Candidate A (Causal Absorption Graph)

- **Prior work attack**: AP-Calculus (2025) applies do-calculus to Bayesian networks, not SAEs. While the formalism extends, no prior work has connected causal inference to SAE feature interactions. However, the "Causal Layer Attribution Technique" (2025) does layer-wise causal attribution in neural networks --- this is adjacent work that could be seen as overlapping.
- **Methodological attack**: Computing do(parent_feature = 1) requires counterfactual interventions on SAE latents. This is technically feasible (set latent activation to 1, propagate through decoder), but the "holding child constant" part requires do(parent=1, child=0) --- a joint intervention that may violate the SAE's natural dynamics. The causal graph structure (which features cause which) is unknown and must be estimated from data.
- **Theoretical attack**: The causal graph for SAE features is dense and cyclic (features interact in complex ways). Do-calculus requires a known DAG structure. AP-Calculus addresses this for Bayesian networks but SAE features are not structured as a Bayesian network. The structural assumptions may not hold.
- **Scalability attack**: Estimating causal effects for 24K+ latents requires massive computation. Even with approximations (local neighborhoods), the number of potential confounders is enormous. The method may work for toy examples (first-letter features) but not scale to real semantic features.
- **Verdict**: MODERATE --- The causal framing is intellectually powerful and addresses the project's core weakness (null results on correlational analysis). However, scalability and structural assumptions are serious concerns. A restricted version (causal analysis on a small set of known hierarchical features) could work.

### Against Candidate B (Information-Theoretic Decomposition)

- **Prior work attack**: The Information Bottleneck (IB) principle has been applied to deep learning extensively, including SAEs. However, no work has specifically decomposed absorption into conditional mutual information terms. The "rate-distortion" framing for SAEs is novel but the general IB-SAE connection is not.
- **Methodological attack**: Estimating I(parent; output | child) requires knowing which latents correspond to parent and child features. This requires running the Chanin absorption metric first --- the information-theoretic decomposition is a post-hoc analysis, not a standalone detection method. MI estimation in high dimensions is notoriously noisy.
- **Theoretical attack**: Rate-distortion theory assumes a known source distribution and distortion measure. For SAEs, the "source" is model activations (unknown distribution) and the "distortion" is reconstruction error (not task performance). The analogy is suggestive but not formally tight.
- **Scalability attack**: Neural MI estimation (MINE, InfoNCE) requires training auxiliary networks. For each parent-child pair, a separate MI estimator may be needed. This is computationally expensive for large SAEs.
- **Verdict**: STRONG --- The information-theoretic reframing directly explains the project's key findings (precision invariant, recall varies; steering capability intact, efficiency subtle). It provides a theoretical language for why absorption coexists with functional tasks. The computational concerns are manageable with approximations.

### Against Candidate C (Spectral Hierarchy Recovery)

- **Prior work attack**: MetaSAE (2025) already trains autoencoders on decoder weight matrices to assess atomicity. SDSNet (2025) combines autoencoders with spectral clustering. Jedryszek & Crook (2026) analyze decoder cosine-similarity structure. The individual components exist; the novelty is in applying spectral clustering specifically to absorption detection.
- **Methodological attack**: Spectral clustering requires constructing a similarity graph. For SAE decoders, the natural similarity is cosine similarity, but this may be dominated by noise for weakly-active features. The graph may have many small disconnected components, making community detection unreliable.
- **Theoretical attack**: The claim that absorbed pairs form "low-dimensional submanifolds" is a geometric intuition, not a proven property. Decoder vectors are trained to reconstruct activations, not to form hierarchies. The hierarchical structure may not be geometrically manifest in the decoder weights.
- **Scalability attack**: Spectral clustering on 24K nodes is O(n^3) for exact eigendecomposition. Approximate methods (Nystrom, landmark-based) exist but add error. For 1M latents (GemmaScope), exact methods are infeasible.
- **Verdict**: MODERATE --- The geometric intuition is compelling but unproven. Scalability concerns are real but addressable with approximations. The Fiedler value as a predictor of absorption stiffness is genuinely novel but needs validation.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate A (Causal Absorption Graph)** dropped because: The causal graph structure for SAE features is unknown and likely dense/cyclic. Do-calculus requires DAG assumptions that may not hold. Scalability to real SAEs is questionable. However, the core insight --- that absorption is a mediation effect, not a direct effect --- is incorporated into the strengthened Candidate B.

### Strengthened Ideas

- **Candidate B (Information-Theoretic Decomposition)**: Strengthened by incorporating the causal insight from Candidate A. Instead of pure information theory, we frame absorption as a **causal information decomposition**: the total information a parent feature contributes to the output decomposes into (1) information shared with child features (redundant, survives absorption), (2) information unique to the parent (lost to absorption), and (3) synergistic information (emerges from parent-child interaction). This is the Partial Information Decomposition (PID) framework from Williams & Beer (2010), applied to SAE features for the first time. PID provides a rigorous mathematical decomposition that explains WHY absorption coexists with functional steering: absorption only destroys the unique information (component 2), while the redundant information (component 1) preserves task performance.

- **Candidate C (Spectral Hierarchy Recovery)**: Strengthened by restricting to a **local spectral analysis** on known absorption pairs (from Chanin metric). Instead of clustering all 24K latents, we compute the spectral properties (Fiedler value, community structure) of small subgraphs around confirmed absorption pairs. This is O(k^3) for k-neighborhoods (k ~ 10-50), trivially scalable. The question becomes: "Do absorption pairs have lower Fiedler values (stiffer hierarchies) than non-absorbed correlated pairs?" This is a direct, testable hypothesis.

### Additional Evidence Found

- **[Williams & Beer, 2010. Nonnegative Decomposition of Multivariate Information. arXiv:1004.2515]** --- PID framework decomposes mutual information into redundant, unique, and synergistic components. Exactly the mathematical tool needed for the strengthened Candidate B.

- **[Timme et al., 2014. Synergy, redundancy, and multivariate information]** --- Shows PID applies to neural network feature interactions. Provides algorithms for computing PID from activation data.

- **[Makkeh et al., 2021. A Differentiable measure of pointwise shared information]** --- Differentiable PID estimators enable gradient-based optimization. Could be applied to SAE training objectives.

- **[Von Luxburg, 2007. A Tutorial on Spectral Clustering. Statistics and Computing]** --- Spectral clustering on k-nearest-neighbor graphs is well-understood and scalable. Local neighborhood analysis is standard practice.

### Selected Front-Runner

**Candidate B (strengthened): Partial Information Decomposition of Feature Absorption — Why Absorption Coexists with Functional Steering**

This is the strongest candidate because:

1. **Directly explains the project's null results**: PID decomposes parent feature information into redundant (shared with child), unique (lost to absorption), and synergistic (emergent) components. The project's finding that steering capability is intact (even absorbed features achieve 100% success) means the redundant information is sufficient for the task. The delta-corrected correlation at layer 8 (r=-0.431) captures the loss of unique information.

2. **Mathematically rigorous**: PID is a well-established framework with axiomatic foundations. Unlike the causal graph approach, it does not require knowing the full causal structure --- only the joint distribution of parent activation, child activation, and task output.

3. **Training-free**: All quantities can be estimated from pretrained SAE activations and task outputs. No retraining or architectural modification needed.

4. **Scalable**: PID for two variables (parent, child) and one target (task output) involves computing four mutual information terms: I(parent; output), I(child; output), I(parent, child; output), and I(parent; child). Each can be estimated with neural MI estimators or binned histograms for discrete tasks.

5. **Novel**: No prior work has applied PID to SAE absorption. The connection between absorption and information redundancy is entirely new.

6. **Addresses the project's core weakness**: The current paper's main vulnerability is the weak/absent correlation between absorption and task performance. PID reframes this from "absorption doesn't matter" to "absorption matters for unique information, but redundant information preserves task performance" --- a nuanced, theoretically-grounded finding.

## Phase 5: Final Proposal

### Title

**"Partial Information Decomposition Reveals Why Feature Absorption Coexists with Functional Steering: Redundancy Preserves Task Performance, Uniqueness is Lost"**

Alternative: **"The Information Geometry of Feature Absorption: A PID Analysis of SAE Feature Redundancy"**

### Hypothesis

For a hierarchical parent-child feature pair in an SAE, the total information the parent contributes to a downstream task decomposes into three PID components: (H1) redundant information (shared with child, survives absorption), (H2) unique information (lost to absorption), and (H3) synergistic information (emerges from parent-child interaction). The Chanin absorption metric correlates with the loss of unique information (H2), not with total information. This explains why absorption coexists with functional steering: tasks that depend only on redundant information are unaffected by absorption, while tasks requiring unique information degrade.

### Motivation

The current project has established that feature absorption shows limited correlation with downstream task performance when measured by raw metrics, but a significant negative correlation emerges after baseline correction at layer 8. This pattern --- null on raw, significant on delta-corrected --- is exactly what PID predicts: raw metrics measure total effect (redundant + unique), while delta-corrected metrics isolate the unique component by removing the baseline (redundant) contribution.

No existing work provides an information-theoretic explanation for why absorption coexists with functional steering. The field currently has two incompatible narratives: (1) absorption is a serious failure mode that degrades interpretability (Chanin et al.), and (2) absorption is benign because steering still works (current project's null results). PID reconciles these: absorption is serious for unique information but benign for redundant information.

### Method

**Step 1: Identify Absorption Pairs**
Use Chanin et al.'s absorption detection method to generate ground-truth parent-child absorption pairs on the first-letter task (A-Z). For each absorbed feature, identify the absorbing child latents.

**Step 2: Compute PID Components**
For each parent-child pair and each task (steering success, probing accuracy):
- Define three variables: P (parent activation, binary: fires/doesn't fire), C (child activation, binary), T (task outcome, binary: success/failure).
- Compute the PID decomposition: I(P, C; T) = Red(P, C; T) + Unq(P; T) + Unq(C; T) + Syn(P, C; T)
- Use the I_min redundancy measure (Williams & Beer, 2010) for interpretability.
- Estimate MI terms using binned histograms (for discrete tasks) or neural estimators (for continuous tasks).

**Step 3: Correlate Absorption Rate with PID Components**
- Test whether absorption rate A correlates with Unq(P; T) / I(P, C; T) (the fraction of unique information).
- Test whether A correlates with Red(P, C; T) / I(P, C; T) (the fraction of redundant information).
- Predict: A should negatively correlate with unique fraction and be uncorrelated with redundant fraction.

**Step 4: Layer-Dependent Analysis**
- Repeat Steps 2-3 for layers 4 and 8.
- Test whether the unique information fraction varies with layer depth.
- Predict: deeper layers have more unique information (stronger hierarchies), so absorption has larger effect --- matching the layer 8 significance finding.

**Step 5: Cross-Task Comparison**
- Compare steering (where null result was found) vs. probing (where null result was also found).
- Predict: steering depends more on redundant information (direction in activation space), while probing depends more on unique information (specific feature detection).
- If supported, this explains why both tasks show weak raw correlation but different delta-corrected patterns.

### Cross-Domain Insight

The key transplanted principle is **Partial Information Decomposition from information theory (Williams & Beer, 2010)**. PID decomposes the information two (or more) variables provide about a target into non-overlapping components: redundancy, uniqueness, and synergy. In neuroscience, PID has been used to analyze how neural populations encode information redundantly vs. synergistically. We transplant this to SAEs: parent and child features are two "neurons" in a population, and the task output is the "stimulus." Absorption is the suppression of one neuron (parent), which eliminates its unique information contribution while preserving redundant information shared with the other neuron (child).

The structural correspondence is exact: PID requires only the joint distribution P(P, C, T), which is directly observable from SAE activations and task outcomes. No causal graph or structural assumptions are needed.

### Experimental Plan

| Experiment | Model | SAE | Metrics | Time |
|---|---|---|---|---|
| E1: PID computation for steering task | GPT-2 Small | gpt2-small-res-jb (24K latents) | Red, Unq, Syn components per absorption pair | ~30 min |
| E2: PID computation for probing task | GPT-2 Small | Same | Same, for k-sparse probing | ~30 min |
| E3: Correlation analysis | GPT-2 Small | Same | Pearson r(A, Unq/I_total), r(A, Red/I_total) | ~10 min |
| E4: Layer comparison (l4 vs l8) | GPT-2 Small | Same | PID components by layer | ~20 min |
| E5: Cross-task comparison | GPT-2 Small | Same | Steering PID vs. probing PID | ~10 min |
| E6: Validation on non-absorbed pairs | GPT-2 Small | Same | Control: PID for correlated but non-absorbed pairs | ~15 min |

**Total estimated time**: ~2 GPU-hours (well within project constraints).

**Falsification criteria**:
- H1 falsified if unique information fraction does not correlate with absorption rate (r < 0.3, p > 0.05)
- H2 falsified if redundant information fraction correlates with absorption rate (suggesting absorption affects redundancy)
- H3 falsified if synergistic information dominates (indicating parent-child interaction, not suppression)

### Resource Estimate

- **Models**: GPT-2 Small (124M, local GPU)
- **SAEs**: Pretrained from SAELens (no training required)
- **Compute**: ~2 GPU-hours total
- **Time**: 1 day for implementation + execution
- **Data**: First-letter features (A-Z) for consistency with current project; existing steering and probing data can be reused
- **Software**: PID computation via `dit` Python package (dedicated information theory library) or custom implementation

### Risk Assessment

1. **Risk: PID computation is numerically unstable**
   - *Mitigation*: Use the `dit` package (well-tested PID implementations). For small sample sizes, use bias-corrected estimators. Validate with synthetic data where ground-truth PID is known.
   - *Fallback*: Report bounds (minimum/maximum unique information) instead of point estimates.

2. **Risk: PID components don't correlate with absorption as predicted**
   - *Mitigation*: This would mean the information-theoretic reframing is wrong, which itself is a finding. The paper becomes "Information Decomposition Fails to Explain Absorption's Weak Task Impact" --- a negative result with methodological value.
   - *Fallback*: The PID framework still provides a novel language for analyzing SAE feature interactions, independent of the specific correlation predictions.

3. **Risk: Steering/probing tasks are too coarse for PID**
   - *Mitigation*: PID requires discrete or binned variables. Steering success is already binary (success/failure). Probing accuracy can be binned (correct/incorrect). For continuous metrics, use MI neural estimators.
   - *Fallback*: Focus on binary task outcomes where PID is most reliable.

### Novelty Claim

This proposal makes three novel contributions:

1. **First application of Partial Information Decomposition to SAE feature absorption**: PID provides a rigorous mathematical framework for decomposing the information contribution of absorbed parent features into redundant, unique, and synergistic components.

2. **First information-theoretic explanation for why absorption coexists with functional steering**: The null result on raw metrics is explained by redundancy preservation --- the child feature carries redundant information that maintains task performance. The significant delta-corrected correlation is explained by unique information loss --- absorption specifically suppresses the parent's unique contribution.

3. **First layer-dependent information decomposition**: The finding that absorption's impact varies by layer depth (significant at layer 8, absent at layer 4) is explained by varying ratios of unique-to-redundant information across layers.

No prior work combines these elements. PID has been applied to neural population coding in neuroscience but never to SAE interpretability. The connection between absorption and information redundancy is entirely new and directly addresses the project's central puzzle.
