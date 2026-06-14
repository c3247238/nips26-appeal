# Novelty Report — Iteration 2

**Generated:** 2026-04-13
**Candidate:** `cand_ars_amortization_crosshierarchy` (front-runner, three-contribution proposal)
**Backup candidates also assessed:** `cand_lca_sae`, `cand_successive_refinement`, `cand_recoverability`

---

## Executive Summary

The front-runner proposal's three primary contributions are assessed as follows:

| Contribution | Novelty Score | Recommendation | Key Risk |
|---|---|---|---|
| C1: ARS Unsupervised Absorption Detector | 8/10 | Proceed | "Geometry of Concepts" uses co-occurrence but not directed asymmetry for absorption detection; Alignment Forum NPMI work is the closest competitor but not absorption-focused |
| C2: Amortization Gap Controlled Dictionary Experiment | 9/10 | Proceed | O'Neill et al. proves the gap but does NOT run the fixed-dictionary controlled experiment; MP-SAE retrains decoder jointly — cannot isolate encoder from dictionary |
| C3: Cross-Hierarchy Absorption Characterization | 9/10 | Proceed | No prior work uses entity-type or syntactic hierarchies with matched negative controls; Resurrecting the Salmon explicitly acknowledges first-letter metric is insufficient |
| Backup B: Successive Refinement | 7/10 | Proceed as supplementary | Equitz-Cover framing is novel; width comparison experiments exist but not in successive-refinement terms |
| Backup C: Recoverability Analysis | 8/10 | Proceed if triggered | Lossless vs. lossy absorption distinction is novel; information-theoretic framing not in literature |

**Overall novelty: HIGH** — all three primary contributions achieve score ≥ 8. No prior work performs the specific controlled experiment (C2), uses entity-type/syntactic hierarchies with negative controls (C3), or builds an unsupervised absorption detector from co-occurrence network topology with absorption detection as the explicit goal (C1).

---

## Contribution 1: Absorption Risk Score (ARS) — Unsupervised Absorption Detector

### Core novelty claim
ARS(i,j) = O(i,j) × A(i,j) × cos²(d_i, d_j) is the first unsupervised, training-free method for predicting SAE feature absorption from co-occurrence network topology, without requiring pre-specified probe directions or known feature hierarchies.

### Prior work found

**Closest competitor: "The Geometry of Concepts: Sparse Autoencoder Feature Structure" (Li et al., arXiv:2410.19750, published Entropy 2025)**
- Studies co-occurrence structure of SAE features using affinity scores (Jaccard similarity, simple matching coefficient, Dice coefficient, overlap coefficient, Phi coefficient on co-occurrence histograms) and performs spectral clustering.
- Observes "brain-like" modularity where co-occurring feature clusters form spatially coherent lobes.
- **Does NOT connect co-occurrence statistics to absorption detection.** The paper's goal is geometric characterization of the feature space, not detecting which features are absorbed. No absorption metric is computed or validated.
- **Overlap severity:** related_work — shares the co-occurrence analysis approach but not the absorption detection goal or the directed asymmetry component.

**Second competitor: "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit" / Alignment Forum "Towards data-centric interpretability with sparse" (arXiv:2512.10092, Dec 2025)**
- Uses NPMI as co-occurrence metric to find "interesting" latent pairs.
- Explicitly computes conditional probabilities P(j|i) and P(i|j) as directed/asymmetric statistics to identify directional relationships (e.g., "most offensive text mentions religion" vs. "most texts mentioning religion are offensive").
- **Does NOT use co-occurrence topology for absorption detection.** The goal is discovering novel correlations in model representations, not detecting absorbed features. No validation against Chanin et al. absorption labels or any absorption ground truth. The method also requires LLM-based filtering of non-semantic latents, adding overhead not present in ARS.
- **Overlap severity:** partial_overlap — the directed conditional probability component (A(i,j) in ARS) is conceptually similar, but the overall goal, the ecological motivation, the Jaccard × asymmetry × decoder-cosine product form, and the absorption detection framing are all distinct.

**OrtSAE (Korznikov et al., arXiv:2509.22033, Sep 2025)**
- Uses decoder cosine similarity as an implicit feature to enforce orthogonality and reduce absorption by 65%.
- The cos²(d_i, d_j) component in ARS overlaps with OrtSAE's orthogonality penalty signal.
- **Overlap severity:** related_work — OrtSAE uses decoder cosine similarity for architectural regularization during training; ARS uses it as a post-hoc detection signal combined with two co-occurrence statistics. OrtSAE requires retraining; ARS is training-free.

**ATM SAE — "Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training" (arXiv:2510.08855, Oct 2025, ICLR 2025 workshop)**
- Uses token masking during training to disrupt co-occurrence statistics and reduce absorption (absorption score 0.0068 vs. 0.1402 for TopK).
- **Overlap severity:** related_work — ATM uses co-occurrence disruption as a training-time technique; ARS uses co-occurrence measurement as a post-hoc detection technique. Completely different stage (training vs. inference).

**Masked Regularization (referenced in proposal as arXiv:2604.06495)**
- This arXiv ID was not found in indexed literature as of April 2026. The paper appears to be very recent or pre-indexed. The proposal's description of it (token masking to disrupt co-occurrence during training) corresponds to the ATM SAE paper (2510.08855), which was found. No separate paper at 2604.06495 could be verified — this reference may be a future paper or a mis-recalled ID. This does not affect the novelty of ARS.

**"Chanin et al. A is for Absorption" (arXiv:2409.14507, NeurIPS 2024)**
- Foundational paper. Uses decoder cosine similarity to identify main latents for each letter, then uses k-sparse probing (not co-occurrence topology) to detect absorption.
- Notes that "absorption leads to an asymmetric pattern in the encoder and decoder of the SAE, so it may be possible to use this insight to detect absorption" — but does NOT develop an unsupervised detector from this observation.
- **Overlap severity:** related_work — provides ground truth labels that ARS will be validated against; does not implement co-occurrence-based detection.

**"Measuring SAE Feature Sensitivity" (arXiv:2509.23717, Sep 2025)**
- Measures how reliably features activate on semantically similar texts. Uses an LM-based generation approach to test sensitivity, not co-occurrence statistics.
- **Overlap severity:** related_work — addresses feature reliability (different dimension than absorption).

### Verdict for C1
**Novelty score: 8/10.** The ARS product form (Jaccard overlap × activation asymmetry × decoder cosine similarity) motivated by ecological competitive exclusion theory, and applied specifically for unsupervised absorption detection against Chanin et al. ground truth labels, is novel. The closest competitor (arXiv:2512.10092) uses directed co-occurrence statistics but not for absorption detection and not in the ARS product form. The ecological competitive exclusion motivation is entirely absent from the literature. The EDA contrast (from iter_001, AUROC = 0.776 but failing for 75% early absorption) is also a differentiator that positions ARS as complementary to prior work.

**Differentiation note:** The proposal must clearly state that the Alignment Forum NPMI work (arXiv:2512.10092) uses directed conditional probabilities for correlation discovery (not absorption detection), to prevent reviewers from conflating the two. The ARS ecological framing and the specific absorption detection validation are the novel elements.

---

## Contribution 2: Amortization Gap Controlled Dictionary Experiment

### Core novelty claim
No prior paper has fixed the pre-trained decoder dictionary while varying encoding method (feedforward vs. OMP vs. 2-pass) to directly test the amortization gap vs. sparsity landscape explanations of absorption.

### Prior work found

**O'Neill et al. "Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders" (arXiv:2411.13117, ICML 2025)**
- Proves theoretically that a linear-nonlinear encoder has a provably non-zero amortization gap using compressed sensing theory.
- Decouples encoding and decoding and compares different encoding strategies (standard encoder, MLP encoder, SAE+ITO — inference-time optimization).
- Evaluates on GPT-2 activations and shows more expressive encoders produce more interpretable features.
- **Critical gap:** Does NOT measure absorption rate as the outcome. Does not use the Chanin et al. absorption metric. Does not test the specific hypothesis "does OMP reduce absorption when the decoder dictionary is fixed?" The paper evaluates mean correlation coefficient (MCC) for sparse code recovery and reconstruction error — not the absorption rate on a labeled feature set. Also notably does NOT mention OMP specifically as a tested encoder.
- **Overlap severity:** partial_overlap — establishes the theoretical framework and the decoder-fixing approach, but does not run the absorption-focused controlled experiment that the proposal describes.

**Costa et al. "From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit" (arXiv:2506.03093, NeurIPS 2025)**
- Trains MP-SAE by jointly retraining both the dictionary and the iterative encoder.
- Demonstrates reduction in absorption on MNIST-like hierarchical tasks.
- **Critical differentiator:** MP-SAE retrains the decoder jointly with the encoder. This means it cannot distinguish whether absorption reduction is due to better encoder inference (encoder hypothesis) or a changed dictionary geometry (landscape hypothesis). The proposal's experiment fixes the pre-trained decoder and only varies the encoder — this isolation is the scientific contribution.
- **Overlap severity:** partial_overlap — MP-SAE uses matching pursuit and reduces absorption, but cannot adjudicate the encoder vs. dictionary debate that the controlled experiment is designed to settle.

**Costa et al. "Evaluating Sparse Autoencoders: From Shallow Design to Matching Pursuit" (arXiv:2506.05239)**
- Companion evaluation paper testing MP-SAE vs. shallow SAEs on MNIST.
- Does not use Gemma Scope, does not fix pre-trained decoder, does not measure Chanin absorption rates.
- **Overlap severity:** related_work.

**Tang et al. "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima" (arXiv:2512.05534)**
- Theoretical framework characterizing absorption as a stable spurious minimum of the piecewise biconvex SDL loss.
- Argues the dictionary geometry encodes absorption; encoder changes alone cannot fix it.
- **Does NOT run the controlled experiment.** This paper's core argument is exactly what the proposal's H2 is designed to test — if Tang et al. are right, OMP should NOT reduce absorption when the decoder is fixed.
- **Overlap severity:** related_work — provides the competing theoretical prediction that C2 will empirically test.

**ATM SAE (arXiv:2510.08855)**
- Training-time masking approach, not inference-time encoding variation.
- Does not isolate encoder from dictionary.
- **Overlap severity:** related_work.

### Verdict for C2
**Novelty score: 9/10.** The specific controlled experiment — fixing the pre-trained decoder dictionary and varying only the encoding method to measure absorption rate using the Chanin et al. metric — has not been done by any prior paper. O'Neill et al. (2411.13117) is the closest prior work but measures code recovery quality (MCC), not absorption rate, and does not test on Chanin-labeled features. MP-SAE (2506.03093) uses matching pursuit but retrains the dictionary, confounding the isolation. The proposal's experiment is the only one that can directly adjudicate the Tang et al. vs. O'Neill et al. debate. Both possible outcomes (amortization gap dominant vs. sparsity landscape dominant) are scientifically significant and publishable.

---

## Contribution 3: Cross-Hierarchy Absorption Characterization with Negative Controls

### Core novelty claim
No prior paper measures SAE feature absorption in entity-type or syntactic hierarchies with proper matched-frequency-ratio negative controls. All published absorption measurements use the first-letter task.

### Prior work found

**"A is for Absorption" / SAEBench (Chanin et al., arXiv:2409.14507, NeurIPS 2024; SAEBench arXiv:2503.09532, 2025)**
- All absorption measurements are on first-letter spelling tasks.
- SAEBench explicitly notes: "The feature absorption metric in SAEBench is a variation on the metric defined in the original feature absorption work (Chanin et al., 2024)." Only first-letter hierarchy is used.
- **Overlap severity:** related_work — establishes the benchmark this proposal extends to other hierarchies.

**"Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders" (arXiv:2508.09363)**
- Explicitly states: "SAEBench evaluates feature absorption by using features for 'word starts with x', which is not useful for evaluating domain-specific feature absorption."
- Calls for absorption measurement beyond first-letter but does NOT implement alternative hierarchies.
- **Overlap severity:** related_work — directly motivates C3 but does not execute it.

**RAVEL (Variational Evaluation of Entity-Level Disentanglement)**
- Evaluates whether SAE latents can disentangle entity attributes (e.g., country vs. continent for cities).
- Used by Matryoshka SAE papers as a cross-hierarchy test. The RAVEL metric tests if targeted latent interventions change one attribute without changing related ones.
- **Overlap severity:** partial_overlap — RAVEL tests a cross-hierarchy disentanglement question, but it is different from measuring absorption rate. RAVEL does not use matched negative controls; it tests whether interventions on continent-latents affect country-latents. The proposal tests whether parent features are absorbed by child features in entity-type hierarchies — a different measurement using the Chanin absorption metric framework, not RAVEL's intervention framework. Additionally, iter_001 found that RAVEL was confounded by wrong-model probes; this proposal fixes that.

**H-SAE (Muchane et al., arXiv:2506.01197, Jun 2025)**
- Introduces an SAE architecture that explicitly encodes semantic hierarchies with parent features gating child features.
- Tests hierarchical concept representation in LLMs.
- **Overlap severity:** related_work — an architectural approach to hierarchical SAEs; does not measure absorption rates across different hierarchy types with negative controls.

**Matryoshka SAE (multiple papers, 2025)**
- Tests on first-letter task (Chanin absorption score) and RAVEL. Does not extend to entity-type or syntactic hierarchies with negative controls.
- **Overlap severity:** related_work.

**"Analyzing (In)Abilities of SAEs via Formal Languages" (arXiv:2410.11767, NAACL 2025)**
- Tests SAEs on formal language hierarchies (Dyck-2, Expr). Different from natural-language entity-type hierarchies. No negative controls.
- **Overlap severity:** related_work.

**No paper found that:**
- Uses entity-type hierarchies (ANIMAL ⊃ specific animals) or syntactic hierarchies (PAST-TENSE ⊃ IRREGULAR-PAST-TENSE) with the Chanin absorption metric.
- Matches non-hierarchical control pairs on frequency ratio.
- Tests whether frequency ratio ρ predicts absorption rate across hierarchy types.

### Verdict for C3
**Novelty score: 9/10.** The combination of (a) entity-type and syntactic hierarchies, (b) matched-frequency negative controls, and (c) the frequency ratio ρ as a predictor of absorption severity is entirely absent from the literature. The "Resurrecting the Salmon" paper provides direct textual support for this gap's existence ("not useful for evaluating domain-specific feature absorption"). RAVEL provides a partial precedent for cross-hierarchy evaluation but uses a different metric framework and has no negative controls.

---

## Backup Candidate Assessment

### Backup A: LCA-SAE (cand_lca_sae)

**Novelty score: 6/10.** The parallel LCA lateral competition approach applied to SAE absorption is novel vs. existing SAEs (which use feedforward encoders). The key differentiator from MP-SAE (sequential greedy vs. simultaneous competition) is real. However, the general LCA + lateral inhibition from decoder Gram matrix has been described in the signal processing literature for some time. The novelty depends critically on whether the MP-SAE papers (2506.03093, 2506.05239) can be distinguished: the proposal claims parallel competition is different from sequential matching pursuit, which is a defensible but not highly differentiated claim. The MP-SAE results at NeurIPS 2025 make the "iterative encoder for absorption reduction" space more crowded. **Recommendation: proceed only as backup if amortization gap experiment confirms encoder effects are significant.**

### Backup B: Successive Refinement Test (cand_successive_refinement)

**Novelty score: 7/10.** The Equitz-Cover (1991) successive refinement framework applied to SAE absorption is novel. Width comparison experiments exist (SAEBench compares 4k/16k/65k SAEs on absorption) but no paper asks: "are absorbed features in narrower SAEs recovered in wider SAEs?" in the successive refinement sense. The weight-only analysis approach is lightweight and the theoretical framing is novel. **Recommendation: proceed as supplementary.**

### Backup C: Recoverability Analysis (cand_recoverability)

**Novelty score: 8/10.** The lossless vs. lossy absorption distinction and the information recovery ratio metric (I(parent; child_activations) / I(parent; parent_activations)) are novel. No paper distinguishes whether late-absorbed latents can be recovered from child activations via linear readout. This is practically important for mitigation strategy selection (inference-time correction vs. retraining). **Recommendation: proceed if triggered by amortization gap experiment.**

---

## Candidate Ranking

| Rank | Candidate | Novelty | Recommendation |
|------|-----------|---------|----------------|
| 1 | cand_ars_amortization_crosshierarchy (C1+C2+C3 combined) | 9/10 | Proceed — front-runner |
| 2 | cand_recoverability | 8/10 | Proceed if triggered |
| 3 | cand_successive_refinement | 7/10 | Proceed as supplementary |
| 4 | cand_lca_sae | 6/10 | Proceed only as fallback |

---

## Key References for Related Work Section (Paper Writing)

All of these should be cited as related work, not as collisions:

1. Chanin et al. (2024). "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." NeurIPS 2024. arXiv:2409.14507.
2. O'Neill et al. (2024/2025). "Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders." ICML 2025. arXiv:2411.13117. — **Directly related to C2; paper must clearly distinguish their MCC metric from C2's absorption rate metric.**
3. Costa et al. (2025). "From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit." NeurIPS 2025. arXiv:2506.03093. — **Must differentiate: MP-SAE retrains decoder; C2 fixes decoder.**
4. Tang et al. (2025). "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima." arXiv:2512.05534. — **The competing theoretical prediction that C2 empirically tests.**
5. Li et al. (2024/2025). "The Geometry of Concepts: Sparse Autoencoder Feature Structure." Entropy 2025. arXiv:2410.19750. — **Uses co-occurrence but not for absorption detection; C1 extends this to directed absorption detection.**
6. Karzaurov/Alignment Forum (2025). "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit." arXiv:2512.10092. — **Uses directed NPMI but not for absorption detection; must differentiate.**
7. Korznikov et al. (2025). "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033. — **Uses decoder cosine similarity for training regularization; C1 uses it post-hoc in ARS product.**
8. Bussmann et al. (2025). "Matryoshka Sparse Autoencoders." Alignment Forum. — **Reduces absorption via hierarchical nested training; related to C3 architecture baseline.**
9. Muchane et al. (2025). "Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures." arXiv:2506.01197. — **H-SAE; hierarchical architecture approach, distinct from cross-hierarchy measurement.**
10. Karvonen et al. (2025). "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders." arXiv:2503.09532. — **Benchmark; C3 extends it beyond first-letter hierarchy.**
11. "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders." arXiv:2508.09363. — **Directly calls for C3-type extension.**

---

## Risks Flagged

1. **Partial collision risk for C1 with arXiv:2512.10092:** The directed conditional probability component of ARS is conceptually similar to the NPMI + conditional probability analysis in arXiv:2512.10092. The paper must explicitly position ARS against this work. The ARS novelty rests on: (a) absorption detection as the goal (not correlation discovery), (b) the three-factor product form motivated by ecology, (c) the Chanin et al. validation protocol, and (d) the specific application to early absorption detection where EDA fails.

2. **Crowded encoder improvement space for C2:** O'Neill et al. (ICML 2025) is a highly visible paper at the same intersection of amortization gap + SAE + encoder improvement. The proposal must be explicit that C2 is a controlled absorption-measurement experiment, not a general encoder improvement study. The Chanin absorption metric as the outcome variable is the critical differentiator.

3. **RAVEL as partial precedent for C3:** The use of RAVEL in Matryoshka SAE papers provides a partial precedent for cross-hierarchy evaluation. The proposal must distinguish: RAVEL tests disentanglement via interventions on entity attributes; C3 tests absorption rates in hierarchical feature detection via the Chanin metric. These are different measurements of different phenomena.

4. **arXiv:2604.06495 (Masked Regularization) reference:** This paper could not be verified in indexed literature. If it exists and is published before C1's submission, it should be cited as related work. Based on the ATM SAE (2510.08855) which describes the same training-time token masking approach, the post-hoc ARS detection approach is still distinct. No verification needed before proceeding with experiments.
