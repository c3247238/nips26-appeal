# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **[Chanin et al., 2024. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507, NeurIPS 2025]** — The canonical paper defining feature absorption; proves hierarchical features cause absorption in toy models; measures 15–35% absorption rate across all tested SAEs; introduces the probe-directed ablation metric. Critical foundation.

2. **[Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547, ICML 2025]** — Best architecture for reducing absorption (SAEBench best-in-class); shows clear diagonal decoder-similarity structure vs. vanilla SAE's off-diagonal high-similarity (i.e., parent-child confusion); directly validates that explicit hierarchy encoding reduces absorption.

3. **[Chanin, Dulka & Garriga-Alonso, 2025. "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756]** — Identifies the dual of absorption: hedging (too narrow → merge correlated features); establishes absorption and hedging as opposite failure regimes of sparsity optimization under feature correlation.

4. **[Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033]** — Achieves 65% absorption reduction via pairwise cosine-similarity orthogonality penalty; provides a complementary angle: absorption correlates with non-orthogonal decoder directions.

5. **[Karvonen et al., 2025. "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." arXiv:2503.09532]** — 200+ SAEs, 8 metrics; key finding: proxy metrics (CE, sparsity) do not predict absorption; Matryoshka SAEs dominate absorption metric; TopK/JumpReLU hurt absorption even as they improve reconstruction.

6. **[Author TBD, 2024. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima." arXiv:2512.05534]** — First formal proof that absorption arises as a spurious local minimum of the piecewise biconvex SDL objective; proposes feature anchoring to restore identifiability; validated only on synthetic benchmarks.

7. **[Author TBD, 2026. "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." arXiv:2602.14687]** — Controlled synthetic benchmark with known ground-truth feature hierarchy, Zipfian firing, and correlation; logistic probes reach 0.974 F1 while best SAEs substantially underperform; enables precise causal ablation of hierarchy depth vs. absorption rate.

8. **[Li et al., 2025. "Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training (ATM)." arXiv:2510.08855]** — Per-latent importance scoring achieves absorption score 0.0068 vs. TopK 0.1402 on Gemma-2-2B; best reported absorption metric; mechanism reveals that temporal co-occurrence disruption is a powerful lever.

9. **[Narayanaswamy et al., 2026. "Improving Robustness In Sparse Autoencoders via Masked Regularization." arXiv:2604.06495]** — Token masking during training disrupts co-occurrence patterns, reduces absorption; very recent; suggests co-occurrence statistics in training data are the proximal cause of absorption.

10. **[Author TBD, 2025. "Logical Reasoning in Latent Activation Spaces (ActivationReasoning)." arXiv:2510.18184]** — Proposes relational-feature representations: concepts defined by relations among SAE features, not individual latents; motivates graph-structured view of SAE latent co-activation.

11. **[Author TBD, 2025. "Meta-SAEs." arXiv:2502.04878, ICLR 2025]** — Decomposes SAE latents into meta-latents with shared activation edges; directly maps the co-activation graph of SAE latents; relevant to understanding structural preconditions for absorption.

12. **[Author TBD, 2024. "The Geometry of Concepts: Sparse Autoencoder Feature Structure." arXiv:2410.19750]** — Shows hierarchically related SAE concepts are geometrically orthogonal, while categorical concepts form polytopes; establishes a geometric vocabulary for understanding absorption as violation of orthogonality between hierarchically related latents.

### Landscape Summary

The SAE feature absorption problem is well-defined empirically (Chanin et al., 2024) but poorly understood quantitatively and theoretically at the level required for principled remedies. The sparsity optimization under feature hierarchy is the root cause: when feature A implies feature B, the SAE can save one activation by absorbing A's direction into B, creating a latent that fires on B∩¬A rather than A. This is a local minimum of the SDL objective (arXiv:2512.05534), and it is stable because it achieves perfect reconstruction at lower L0.

The architectural remedy landscape (Matryoshka, OrtSAE, ATM, masked regularization, KronSAE) has grown rapidly, but each approach attacks a symptom without a quantitative theory of absorption magnitude as a function of SAE hyperparameters and data statistics. SAEBench reveals that proxy metrics never reliably predict absorption outcomes, confirming that absorption is structurally decoupled from the reconstruction objective. The geometry of SAE features (arXiv:2410.19750) suggests that hierarchically related features ought to be orthogonal—but the SAE objective provides no incentive for this.

The cross-domain landscape is rich: multi-label learning with label implication constraints, compressed-sensing multi-scale wavelet hierarchies, latent variable models with conditional independence structure, and information-theoretic mutual information minimization all handle the "general absorbs specific" problem in different ways. None of these cross-domain solutions have been transplanted to SAE absorption.

---

## Phase 2: Initial Candidates

### Candidate A: Co-Activation Graph as an Unsupervised Absorption Oracle

**Hypothesis**: Feature absorption is detectable without any labeled probes by analyzing the decoder geometry and co-activation statistics of SAE latents. Specifically, when latent j absorbs latent i, the following observable signatures appear simultaneously: (1) high cosine similarity between decoder vectors d_i and d_j; (2) conditional dependency I(x; i | j) ≈ 0—i.e., latent i carries near-zero additional information given latent j on the tokens where i fails to fire. A co-activation graph edge (i, j) weighted by conditional mutual information drop, combined with decoder cosine similarity filtering, constitutes an unsupervised absorption detection oracle.

**Cross-domain insight**: Causal feature selection in multi-label learning uses conditional mutual information (CMI) to distinguish spurious label co-occurrence from genuine label implication. When label i is absorbed into label j in a hierarchical multi-label model, CMI(label_i; Y | label_j) ≈ 0 and their parameter vectors align—exactly the SAE absorption signature. Transplanting the CMI-based spurious correlation detection framework (used in CVPR 2025 multi-task learning work) to SAE latents gives an unsupervised absorption detector.

**Evidence for**: Chanin et al.'s absorption metric already relies on cosine similarity between decoder directions and probe vectors; OrtSAE shows absorption correlates with non-orthogonal decoder directions; arXiv:2410.19750 shows hierarchically related concepts are geometrically orthogonal in well-trained SAEs; the meta-SAE paper (arXiv:2502.04878) shows co-activation graph structure captures SAE compositional relationships.

**Novelty estimate**: 8/10 — The probe-free angle is explicitly identified as Gap 7 in the literature survey. No paper has combined CMI over co-activation statistics with decoder cosine geometry to detect absorption without probes. The closest is the sensitivity metric (arXiv:2509.23717), which requires per-latent activation examples and measures reliability, not absorption specifically.

---

### Candidate B: Absorption Scaling Laws via Information-Theoretic Toy Model

**Hypothesis**: The severity of absorption between a parent feature P (low frequency) and child feature C (high frequency, P implies C) is a monotonically increasing function of the frequency ratio f_C / f_P and the sparsity target L0, and a monotonically decreasing function of SAE width D. Specifically: absorption_rate(P, C) ≈ σ(α · (f_C/f_P) · (1/L0) · (D_min/D)), where D_min is the minimum width needed to represent P and C independently, σ is a sigmoid-like saturation function, and α is a dataset-specific constant. This scaling law can be empirically validated on a controlled synthetic hierarchy (using SynthSAEBench-style generation) and then transferred to real Gemma Scope SAEs.

**Cross-domain insight**: Compressed sensing multi-scale analysis (wavelets, hierarchical dictionary learning) has established that the ability to recover fine-scale coefficients from sparse measurements degrades predictably with the ratio of coarse-to-fine energy and the measurement budget—an exact structural analogy to the absorption ratio f_C/f_P and L0 budget. The Donoho-Tanner phase transition from compressed sensing provides a mathematical template: a critical measurement budget separates the "recovery possible" from "recovery impossible" regimes. Translating this to SAE absorption: there exists a critical (L0, D, f_C/f_P) surface below which absorption is inevitable.

**Evidence for**: Chanin et al. empirically observe that absorption increases with higher sparsity (lower L0) and wider SAEs; SynthSAEBench enables controlled manipulation of frequency ratios and hierarchy depth; the unified SDL theory (arXiv:2512.05534) provides an energy-based characterization of spurious minima that could support derivation of the scaling law.

**Novelty estimate**: 9/10 — No paper derives a closed-form or empirically validated scaling law for absorption as a function of these three variables jointly. Gap 1 in the literature survey is precisely this. The compressed-sensing phase transition analogy is novel to the SAE field.

---

### Candidate C: Absorption-Aware Feature Recovery via Hierarchical Contrastive Regularization

**Hypothesis**: Adding a contrastive regularization term to SAE training that explicitly penalizes high conditional co-activation between a general latent i and specific latents {j : d_j ≈ d_i} on tokens where i should fire but does not, forces the SAE to break the absorption equilibrium. This can be implemented training-free (post-hoc) as a decoder reweighting step, or during training as an auxiliary loss. The regularization term is: L_contrast = Σ_{(i,j) ∈ absorption_pairs} -log P(i fires | x) when j fires and d_i · d_j > τ.

**Cross-domain insight**: Contrastive learning in multi-label classification with label implication constraints (hierarchical text classification, ontology-constrained classifiers) uses exactly this negative signal: when a specific-level label fires but the general-level label does not fire on the same example, the model is penalized. This "hard negative" on implied labels is the standard solution in hierarchical multi-label learning. Applying this to SAE training imposes a semantic implication constraint from outside the reconstruction objective.

**Evidence for**: Masked regularization (arXiv:2604.06495) shows training-time interventions on co-occurrence reduce absorption; OrtSAE's orthogonality penalty is a weaker version of this (decoder direction constraint rather than firing constraint); Matryoshka SAE's hierarchical loss implicitly enforces that general latents remain active when specific latents fire.

**Novelty estimate**: 7/10 — The contrastive angle is somewhat anticipated by masked regularization and Matryoshka SAE. The key novelty is the explicit implication-based hard-negative contrastive signal rather than a structural penalty, and the training-free post-hoc reweighting formulation. However, this is more incremental than Candidates A and B.

---

## Phase 3: Self-Critique

### Against Candidate A

**Prior work attack**: Searched specifically for "SAE unsupervised absorption detection CMI co-activation graph." Found: (1) Feature sensitivity (arXiv:2509.23717) measures how reliably latents fire on similar texts—a related but distinct probe-free metric. (2) Meta-SAE (arXiv:2502.04878) builds a co-activation graph but for decomposing latents, not detecting absorption. (3) Chanin et al. use cosine similarity in their detection metric, but require pre-specified probe directions. No paper uses CMI over co-activation statistics + decoder cosine geometry jointly for absorption detection without probes.

**Methodological attack**: CMI estimation from finite activation samples may be unreliable for rare latents (low-frequency parent features fire infrequently, making CMI estimates noisy). The threshold τ for cosine similarity and the CMI threshold are hyperparameters that may require tuning per SAE architecture.

**Theoretical attack**: CMI-based detection may confuse absorption with feature splitting: if latent i splits into j and k, then I(x; i | j, k) ≈ 0 too. Need to add a decoder direction constraint to distinguish absorption (d_i ≈ d_j projection) from splitting (d_i is well-represented by d_j + d_k combination).

**Scalability attack**: Computing pairwise CMI over all latent pairs for a 65k-wide SAE requires O(D²) evaluations, which is computationally prohibitive. Need to pre-filter pairs using cosine similarity threshold before computing CMI.

**Verdict**: STRONG — The weaknesses are all engineering-level (CMI estimation, complexity reduction via cosine pre-filtering). The core insight is sound: absorption leaves a joint signature in decoder geometry AND conditional information. The theoretical attack is resolvable by requiring both high cosine similarity AND CMI drop AND unidirectional (asymmetric) conditional dependency (C fires → I fires, not vice versa). The pre-filtering attack is resolvable by using cosine similarity > 0.3 to reduce candidate pairs to O(D·k) where k is small.

---

### Against Candidate B

**Prior work attack**: Searched for "SAE absorption scaling law sparsity frequency hierarchy prediction." Found: Chanin et al. provide empirical observations (absorption increases with lower L0 and wider SAEs) but no closed-form law. The unified SDL theory (arXiv:2512.05534) provides energy-based characterization but no closed-form scaling law. No paper derives the three-variable joint scaling law.

**Methodological attack**: The proposed functional form σ(α · (f_C/f_P) · (1/L0) · (D_min/D)) is guessed. Derivation from first principles requires solving a constrained optimization problem over the biconvex SDL objective, which the unified theory paper acknowledges is hard. Empirical fitting may find a good curve but lack predictive power outside the training regime.

**Theoretical attack**: The compressed-sensing analogy is imperfect: in compressed sensing, the measurement matrix is random and the Donoho-Tanner phase transition is sharp. In SAE training, the "measurement matrix" is the encoder, which adapts to the data. The phase transition in SAE absorption may not be sharp and may depend on initialization and training dynamics in ways the CS analogy cannot capture.

**Scalability attack**: SynthSAEBench-style controlled experiments require training multiple SAEs at different (L0, D, f_C/f_P) settings. Even training-free analysis of existing Gemma Scope SAEs requires access to precomputed activation caches and sufficient compute to run the absorption metric across a grid of configurations.

**Verdict**: MODERATE — The CS phase transition analogy may be too strong; the functional form is speculative. However, the empirical validation component (fitting the scaling law to existing Gemma Scope SAEs and SynthSAEBench data) is feasible training-free. The main risk is that the scaling law is not clean (non-parametric interactions between variables). Mitigation: focus on empirical characterization first, use the CS analogy as motivation for the functional form choice, and stress-test with cross-validation.

---

### Against Candidate C

**Prior work attack**: Searched for "contrastive regularization SAE hierarchical implication training." Found: Masked regularization (arXiv:2604.06495, April 2026) is very recent and also disrupts co-occurrence patterns, albeit via input masking rather than a contrastive activation loss. Matryoshka SAE's nested reconstruction loss is structurally similar: it forces general-level latents to remain active independently of specific-level latents. The training-free post-hoc reweighting variant is new, but the training-time version is already anticipated by two existing papers.

**Methodological attack**: Identifying "absorption pairs" during training is a chicken-and-egg problem: you need to know which latent pairs exhibit absorption to apply the contrastive loss, but absorption only becomes visible after training converges.

**Theoretical attack**: The contrastive loss term as written is non-differentiable (the "when j fires" condition creates a discrete branch). Would need a soft relaxation (e.g., multiply by the probability of j firing), introducing additional hyperparameters.

**Scalability attack**: Online identification of absorption pairs during training adds significant overhead (requires tracking co-activation patterns and decoder cosine similarity at every step).

**Verdict**: WEAK — Multiple fatal flaws: (1) the approach is substantially anticipated by masked regularization and Matryoshka SAE; (2) the chicken-and-egg identification problem is fundamental; (3) the project spec emphasizes training-free analysis. This idea is dropped.

---

## Phase 4: Refinement

### Dropped Ideas

- **Candidate C** dropped because: substantially anticipated by masked regularization (arXiv:2604.06495) and Matryoshka SAE; chicken-and-egg problem for identifying absorption pairs during training is fundamental; project spec emphasizes training-free analysis on existing SAEs.

### Strengthened Ideas

**Candidate A (Co-Activation Graph as Unsupervised Absorption Oracle)**:
- **Changes made**: Added asymmetry constraint to distinguish absorption from splitting—absorption requires d_i projected onto d_j is large (d_j "contains" d_i's direction) AND the conditional firing is asymmetric: P(j fires | i should fire) is high but P(i fires | j fires) is low. Added cosine pre-filtering (threshold 0.3–0.5) before CMI computation to achieve computational tractability.
- **Additional evidence**: The meta-SAE paper (arXiv:2502.04878) shows SAE co-activation edges can be efficiently enumerated; feature geometry paper (arXiv:2410.19750) validates that decoder cosine similarity carries semantic information about feature relationships. The existing sae-spelling codebase provides direct measurement to validate the unsupervised detector against ground-truth absorption labels on the first-letter task.

**Candidate B (Absorption Scaling Laws)**:
- **Changes made**: Relaxed the functional form claim—instead of assuming a specific parametric form, the proposal is to empirically characterize absorption rate as a function of (L0, D, f_C/f_P) on a grid of existing Gemma Scope SAEs (covering widths 1k–1M, multiple layer depths) using the existing sae-spelling absorption metric. The CS phase transition is used as a motivating analogy for the shape of the phase diagram, but the actual characterization is empirical.
- **Additional evidence**: Gemma Scope provides SAEs varying over 3 orders of magnitude in width (1k to 1M), and the sae-spelling absorption metric can be applied to all of them training-free. The SynthSAEBench generation framework allows controlled frequency-ratio manipulation.

### Additional Evidence Found

- arXiv:2502.04878 (Meta-SAEs, ICLR 2025): Co-activation graph structure between SAE latents can be efficiently computed and decoded into interpretable hierarchies. This directly supports the CMI co-activation graph approach for Candidate A.
- arXiv:2410.19750 (Geometry of Concepts): Hierarchically related SAE features are geometrically orthogonal in well-trained SAEs—this provides the normative expectation that violation of orthogonality + conditional firing asymmetry = absorption.
- DeepMind blog (2025): Dense linear probes dramatically outperform SAE probes on safety-relevant downstream tasks; absorption is a key culprit. This motivates the downstream validation component of both Candidates A and B.
- CVPR 2025 multi-task spurious correlation work: CMI-based detection of spurious task correlations is computationally efficient when pre-filtered by feature similarity—directly applicable to the pre-filtering step in Candidate A.

### Selected Front-Runner

**Candidate A: Co-Activation Graph as Unsupervised Absorption Oracle**

Rationale for selection:
1. **Addresses the highest-impact gap**: Gap 7 (unsupervised absorption detection) is the only gap that fundamentally expands the scope of absorption analysis. Every other gap (scaling laws, cross-domain extension, mitigation comparison) requires the researcher to know what features to look for in advance. An unsupervised detector enables the entire field to study absorption at scale without pre-specified probe directions.
2. **Training-free and directly feasible**: The CMI + cosine geometry approach operates entirely on existing SAE activations, requiring no retraining. This perfectly matches the project constraint of training-free analysis on existing Gemma Scope SAEs.
3. **Cleanly falsifiable**: The proposed detector can be validated against the ground-truth absorption labels from the sae-spelling first-letter task (precision/recall against Chanin et al.'s metric). This gives a concrete falsification criterion.
4. **Broader scientific contribution**: A validated unsupervised absorption detector would become a community tool enabling systematic absorption studies on arbitrary feature hierarchies—including safety-relevant features—without requiring domain-specific probe engineering.
5. **Stronger vs. Candidate B**: Candidate B's scaling law, while scientifically valuable, is primarily descriptive (characterizes absorption but doesn't enable new detection). Candidate A both characterizes absorption AND enables new analyses previously impossible.

---

## Phase 5: Final Proposal

### Title

**Absorption Without Probes: Unsupervised Detection of Feature Absorption in Sparse Autoencoders via Co-Activation Geometry**

### Hypothesis

Feature absorption between latents i and j in a trained SAE leaves a jointly detectable, probe-free signature: (1) high cosine similarity between decoder directions d_i and d_j (d_j "subsumes" d_i's direction); (2) asymmetric conditional firing—P(j fires | token∈D_i) is high, but P(i fires | token∈D_i) is systematically lower than expected; and (3) near-zero conditional mutual information I(X; activation_i | activation_j). The conjunction of these three observable conditions (with appropriate thresholds) achieves precision ≥ 0.80 and recall ≥ 0.70 against ground-truth absorption labels from the sae-spelling first-letter task, across multiple SAE architectures (TopK, L1, JumpReLU, Matryoshka) and model families (Gemma 2 2B, GPT-2).

### Motivation

The current state-of-the-art absorption metric (Chanin et al., 2024) requires pre-specified probe directions—the researcher must know in advance which features to measure absorption for. This fundamental bottleneck limits absorption analysis to narrow, researcher-defined proxy tasks (e.g., first-letter spelling) and prevents systematic absorption characterization for the safety-relevant, knowledge-level, and reasoning-level features that most motivate SAE research. DeepMind's 2025 internal study found that SAE probes fail catastrophically on harmful-intent detection while dense linear probes succeed—absorption is the suspected culprit—but this hypothesis cannot be tested systematically without a way to detect absorbed features when the target concept is not known a priori.

The literature gap is explicit: no paper has developed an unsupervised absorption detector (Gap 7 from literature survey). The current metric's probe dependence creates circular reasoning: one can only measure absorption for features one already has probes for, which are typically the simple syntactic features that are easiest to probe, not the complex semantic features most relevant to safety.

The cross-domain insight that makes this feasible: in multi-label learning, label implication (label A implies label B) creates exactly the same statistical signature—high classifier-direction similarity between labels A and B, asymmetric conditional firing, and near-zero CMI(label_A; target | label_B)—and CMI-based spurious correlation detection (used in CVPR 2025 multi-task work) efficiently detects these relationships without ground-truth implication knowledge. SAE latents are structurally identical to multi-label classifiers under feature hierarchy.

### Method

**Step 1: Candidate Pair Generation (Cosine Pre-Filtering)**
For each SAE, compute the pairwise cosine similarity matrix of decoder weight vectors W_dec ∈ R^{D×d}. Retain all pairs (i, j) with cos(d_i, d_j) > τ_cos (initial threshold τ_cos = 0.25, ablated over 0.1–0.5). This reduces the O(D²) problem to O(D·k) where k is the average number of near-parallel decoder pairs per latent—typically small (< 20 even for 65k SAEs based on OrtSAE's orthogonality analysis).

**Step 2: Asymmetric Conditional Firing Score**
For each candidate pair (i, j) from Step 1, collect all tokens t in the activation dataset where latent i's decoder direction contributes significantly to the reconstruction of t (i.e., ⟨W_dec_i, activation(t)⟩ > percentile_75). This is the "expected activation set" for i. Compute: P(i fires | t∈D_i) and P(j fires | t∈D_i). The Absorption Asymmetry Score AAS(i, j) = P(j fires | t∈D_i) / max(P(i fires | t∈D_i), ε). High AAS (j fires when i should but doesn't) with simultaneously low P(i fires | t∈D_i) is the directional absorption signature.

**Step 3: Conditional Mutual Information Drop**
For pairs with AAS(i, j) > τ_aas, compute CMI(X; latent_i | latent_j) using a k-nearest-neighbor CMI estimator (EDGE estimator, efficient for high-dimensional activation spaces). Low CMI indicates that latent i's information is subsumed by latent j—the defining criterion for absorption.

**Step 4: Absorption Score Assembly**
AbsorptionScore(i, j) = sigmoid(w1·cos(d_i, d_j) + w2·AAS(i, j) + w3·(1 - CMI(i|j)/CMI(i))) where weights w1, w2, w3 are fit by logistic regression against the ground-truth absorption labels on the first-letter task (training set: 13 letters; test set: 13 letters). For a latent i, UnsupervisedAbsorptionRate(i) = max_{j≠i} AbsorptionScore(i, j).

**Step 5: Cross-Architecture and Cross-Domain Validation**
Apply the trained detector to: (a) remaining SAE architectures (JumpReLU, Gated, Matryoshka) on Gemma 2 2B to measure cross-architecture generalization; (b) semantic feature hierarchies (entity type ⊃ specific entity, constructed from a wikidata-derived token taxonomy) to test whether the detector generalizes beyond first-letter; (c) safety-relevant features identified via SAEBench's existing safety steering tests to quantify how much absorption explains the linear-probe vs. SAE-probe performance gap.

### Experimental Plan

**Pilot experiment (10–15 min)**: On Gemma 2 2B, layer 12, Gemma Scope 16k SAE: compute pairwise decoder cosine similarity for all latent pairs, measure the cosine distribution, identify the top-50 candidate absorption pairs by cosine similarity, and manually verify 10 pairs against the sae-spelling ground truth. This validates the cosine pre-filtering step before investing in CMI computation.

**Main experiment 1 (30 min)**: On 5 Gemma Scope SAEs (Gemma 2 2B, layers 6/12/18/24, widths 16k and 65k), compute the full unsupervised absorption score on first-letter features. Measure precision/recall against sae-spelling ground truth for all 26 letters. Primary metric: F1 score of unsupervised detector vs. probe-directed Chanin et al. metric. Baseline: cosine-similarity-only detector (Step 1 only).

**Main experiment 2 (30 min)**: Cross-architecture replication on GPT-2-small SAEs (EleutherAI, layers 6 and 10, widths 768 and 4096). Validate that the logistic regression weights fit on Gemma 2 transfer with < 10% F1 degradation, or require architecture-specific recalibration.

**Main experiment 3 (45 min)**: Entity-type absorption: construct a first-name ⊃ specific-person hierarchy (e.g., "Barack" implies "person named Barack" which implies "U.S. President"), identify SAE latents for each level using the sae-spelling k-sparse probing approach, measure the unsupervised detector's recall on these semantic hierarchies. Compare to Chanin et al.'s metric on the same hierarchy. Goal: establish whether absorption occurs and is detectable at semantic hierarchy levels beyond first-letter syntax.

**What would falsify the hypothesis**: If the unsupervised detector achieves F1 < 0.5 against sae-spelling ground truth, the three-condition conjunction is insufficient and the hypothesis is falsified. If cosine similarity alone achieves equivalent F1 without CMI and AAS, the additional conditions are not needed and the method reduces to a simpler form.

### Resource Estimate

All experiments are training-free, operating on pre-loaded Gemma Scope SAEs via SAELens and sae-spelling.
- Memory: Gemma 2 2B requires ~5 GB VRAM; SAEs up to 65k width require ~500 MB additional.
- Compute: Pilot experiment: < 5 min CPU (cosine matrix computation for 16k×768 = 12M pairs, trivially parallelized). Main experiments 1–3: < 1 hour total on a single GPU (GPU needed for activation caching on the Gemma 2 2B model, ~30 min for 10k text samples × 2 SAE widths).
- CMI estimation: k-NN CMI estimator on 10k activation samples for ~500 candidate pairs (post cosine filtering) takes < 5 min on CPU.
- No SAE retraining required.
- Total: well within the 1-hour per experiment constraint.

### Risk Assessment

**Risk 1**: Cosine pre-filtering misses true absorption pairs with low decoder cosine similarity (< 0.25 threshold).
- Mitigation: Ablate τ_cos from 0.1 to 0.5; compare recall at each threshold against sae-spelling ground truth. If high-recall requires τ_cos < 0.1, the assumption that absorption correlates with decoder alignment is wrong—revisit Step 1 using activation correlation instead of decoder cosine similarity.

**Risk 2**: CMI estimation is unreliable for rare parent features (very low firing frequency → few samples for CMI estimation).
- Mitigation: Use minimum sample count N_min = 50 activating tokens as a threshold for including a latent in the analysis. For rare latents below this threshold, fall back to the AAS score alone (no CMI). Report the coverage fraction separately.

**Risk 3**: The logistic regression weights fit on first-letter task do not transfer to semantic hierarchy tasks.
- Mitigation: Treat the logistic regression weights as calibration parameters specific to the first-letter task, and separately validate raw (uncalibrated) AbsorptionScore on semantic hierarchies using a leave-one-out protocol. If calibration doesn't transfer, report both calibrated and uncalibrated results and analyze why (different feature frequency distributions, different co-occurrence structures).

### Novelty Claim

This is the first work to propose and validate a **probe-free absorption detector** for SAE latents. The specific novelty over existing work:

1. **vs. Chanin et al. (arXiv:2409.14507)**: Their metric requires pre-specified probe directions (linear regression probes for target concepts), limiting analysis to researcher-defined features. Our approach requires no probes.
2. **vs. Feature sensitivity (arXiv:2509.23717)**: Sensitivity measures how reliably a latent fires on texts similar to its activating examples—this detects general unreliability but not the specific absorption structure (one latent hijacking another). Our method explicitly detects the directional absorption relationship.
3. **vs. OrtSAE's cosine similarity analysis**: OrtSAE uses cosine similarity as a penalty term but does not combine it with conditional firing asymmetry or CMI to produce an absorption score.
4. **vs. Meta-SAE co-activation graph (arXiv:2502.04878)**: Meta-SAEs decompose latents into meta-latents for interpretability, not to detect absorption. Our use of the co-activation graph is specifically designed to detect the asymmetric conditional firing signature of absorption.

Supporting evidence for novelty: (a) The literature survey explicitly identifies this as Gap 7 ("No metric for absorption that does not require known probe directions"); (b) no paper in the literature survey or subsequent web searches describes CMI-based absorption detection applied to SAE latents; (c) the cross-domain transplant from multi-label learning CMI-based spurious correlation detection (CVPR 2025) has not been applied in the SAE absorption context.
