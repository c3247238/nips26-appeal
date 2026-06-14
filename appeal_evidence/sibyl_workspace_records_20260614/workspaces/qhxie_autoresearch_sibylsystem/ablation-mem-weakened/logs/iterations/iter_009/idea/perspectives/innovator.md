# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **Chanin et al., 2024/2025. "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507** — Foundational work that defines absorption and proves it is a logical consequence of sparsity loss. This is the basis for our research question.

2. **Bussmann et al., 2025. "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547** — Nested architecture reduces absorption from 0.49 to 0.05, but at the cost of introducing hedging trade-off. Confirms absorption is modifiable but not necessarily "fixable."

3. **Korznikov et al., 2025. "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033** — Decoder orthogonality reduces absorption by 65%. Shows architectural constraints can reduce absorption, but the fact that random SAEs still show high absorption (our H7) suggests structural factors dominate.

4. **Li et al., 2025. "Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training." arXiv:2510.08855** — Achieves ~40% absorption reduction via EMA tracking. Represents the training-time approach to absorption mitigation.

5. **Wang et al., ICLR 2026. "Does Higher Interpretability Imply Better Utility?" arXiv:2510.03659** — Weak correlation (~0.3) between interpretability and steering utility. Our null results (H1-H4) are consistent with this finding.

6. **Tang et al., 2025. "On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534** — First theoretical explanation of absorption as spurious local minima. Introduces "feature anchoring" concept.

7. **Cui et al., 2025. "On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy." arXiv:2506.15963** — Formal identifiability analysis. Proves conditions for recovering ground-truth features.

8. **Elhage et al., 2022. "Toy Models of Superposition." Anthropic Blog** — Foundational theory. Superposition hypothesis explains why SAEs are needed and why absorption is inherent.

9. **Sanity Checks for SAEs, 2026. arXiv:2602.14111** — Frozen/random baselines match trained SAEs on standard metrics. Our H7 finding (trained < random on absorption specifically) directly extends this challenge.

### Landscape Summary

The SAE field has moved through three phases: (1) discovery of absorption as a phenomenon requiring mitigation, (2) architectural solutions (Matryoshka, OrtSAE, ATM) that reduce absorption metrics, and (3) validation challenges questioning whether absorption metrics actually matter (Sanity Checks, Wang et al.).

Our work sits at the intersection of phase 2 and phase 3: we question whether absorption reduction is meaningful by showing that (a) absorption does not degrade downstream tasks, and (b) trained SAEs naturally have lower absorption than random baselines, suggesting absorption is partially a structural artifact.

The field lacks theoretical consensus on whether absorption is a failure mode or optimal behavior. Chanin et al.'s Proposition 2 proves absorption is a consequence of sparsity optimization, but does not prove it is harmful.

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Absorption Decomposition
- **Hypothesis**: Absorption can be decomposed into structural (inevitable given overcomplete dictionaries) and pathological (learnable, harmful) components using mutual information measures.
- **Cross-domain insight**: Source separation in signal processing uses information-theoretic criteria (ICA, InfoMax) to separate signal from noise. The same principle could decompose absorption.
- **Why it might work**: Our H7 shows trained SAEs have 8x lower absorption than random SAEs. If absorption has a structural component, mutual information between encoder and decoder could measure the "recoverable" vs "lost" information.
- **Novelty estimate**: 8/10 — No work has applied information-theoretic decomposition to SAE absorption specifically.

### Candidate B: Absorption as Compression-Fidelity Phase Transition
- **Hypothesis**: SAE training exhibits a phase transition where absorption emerges suddenly at a critical dictionary size/compression ratio threshold, analogous to glass transitions in amorphous materials.
- **Cross-domain insight**: Statistical physics phase transitions explain emergent behavior in overcomplete systems. SAE dictionary learning may exhibit similar transitions.
- **Why it might work**: Chanin et al.'s Proposition 2 suggests absorption is a sparsity trade-off. A systematic study of absorption vs. compression ratio could reveal phase behavior.
- **Novelty estimate**: 7/10 — No systematic phase-transition analysis of SAE training exists, though Matryoshka's nested structure implicitly uses multiple compression levels.

### Candidate C: Cross-Model Absorption Transferability
- **Hypothesis**: Features with high absorption in one SAE model (GPT-2) exhibit consistent absorption patterns in other models (Gemma, Llama), suggesting absorption is a property of the feature hierarchy itself, not model-specific.
- **Cross-domain insight**: Transfer learning theory suggests features that are "hard to learn" in one domain are consistently hard across domains. Absorption may be such a property.
- **Why it might work**: If absorption transferability holds, it would support the claim that absorption reflects real feature hierarchy structure rather than training artifacts.
- **Novelty estimate**: 6/10 — Sanity Checks (2026) suggests limited transferability of SAE features generally; whether absorption specifically transfers is unknown.

---

## Phase 3: Self-Critique

### Against Candidate A (Information-Theoretic Decomposition)
- **Prior work attack**: Tang et al. (2025) provides theoretical foundation for absorption as local minima. Their "feature anchoring" concept may already capture what mutual information decomposition would measure.
- **Methodological attack**: Mutual information estimation on high-dimensional SAE activations is notoriously difficult and requires binning or KDE approximations that add noise.
- **Theoretical attack**: The structural/pathological decomposition assumes these are separable. But if absorption is purely structural (as H7 suggests), the decomposition has no pathological component.
- **Scalability attack**: Computing mutual information across 24K SAE latents for multiple layers and models may be computationally expensive.
- **Verdict**: MODERATE — The idea is theoretically sound but may be difficult to operationalize. The mutual information decomposition could validate the structural/pathological split, but H7 already suggests most absorption is structural.

### Against Candidate B (Phase Transition Analysis)
- **Prior work attack**: SplInterp (Budd et al., 2025) uses spline theory to characterize SAE geometry but does not analyze phase transitions. No direct competitor.
- **Methodological attack**: Defining the "order parameter" for SAE absorption phase transition is non-trivial. Compression ratio is a proxy, not a direct measure.
- **Theoretical attack**: Phase transitions typically require thermodynamic limit (n→∞), but SAEs have finite dimensionality. The analogy to physical systems may be superficial.
- **Scalability attack**: Sweeping dictionary sizes and measuring absorption across the phase boundary would require training multiple SAEs, contradicting our training-free constraint.
- **Verdict**: WEAK — Phase transition framing is compelling but (a) requires training SAEs (violating project constraint), (b) lacks clear order parameter, (c) is theoretical without clear empirical test.

### Against Candidate C (Cross-Model Transferability)
- **Prior work attack**: Gao et al. (EMNLP 2025) study SAE feature transferability across layers and tasks. They find limited transferability generally, suggesting absorption transferability may also be limited.
- **Methodological attack**: Comparing absorption patterns across model families (GPT-2 vs Gemma) requires aligning feature semantics, which is non-trivial without a shared feature ontology.
- **Theoretical attack**: Different model architectures (Transformer vs. state space models like Mamba) encode features differently, making cross-architecture absorption comparison problematic.
- **Scalability attack**: Would require loading and analyzing GemmaScope SAEs in addition to our GPT-2 SAEs. This is feasible with SAELens but adds complexity.
- **Verdict**: MODERATE — The idea tests whether absorption is model-specific or reflects universal feature hierarchy structure. The main risk is that limited transferability would not advance the paper's narrative.

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate B (phase transition)** dropped because: (1) requires SAE training, violating project constraint; (2) no clear order parameter; (3) theoretical analogy to physical systems is superficial.

### Strengthened Ideas
- **Candidate A (information-theoretic decomposition)**: The key insight is that our H7 finding (trained < random absorption) suggests absorption has a structural component. If we can measure this structural component independently, it would strengthen the paper's contribution. However, mutual information estimation is noisy and may not add value over existing measures.
- **Candidate C (cross-model transferability)**: This is feasible with SAELens and GemmaScope. A quick test on Gemma-2-2B could either (a) support the universal feature hierarchy claim, strengthening the paper, or (b) show limited transferability, which is also informative.

### Additional Evidence Found
- The literature suggests that SAEBench (Karvonen et al., ICML 2025) is the standard evaluation framework. Our evaluation framework (steering, probing, EC50) is complementary but not standardized.
- Tang et al.'s "feature anchoring" concept suggests absorption may be related to local minima in the loss landscape. Our H5 (precision invariant, recall variable) is consistent with this: absorption affects encoder activation (recall) but not decoder alignment (precision).

### Selected Front-Runner
**Candidate A (Information-Theoretic Absorption Decomposition)** is selected as the front-runner for the innovator perspective, with the caveat that it may be more valuable as a theoretical framing than an empirical contribution.

However, given the project is in consolidation phase with all experiments complete, the innovator's role is to suggest NEW angles rather than repeat existing findings. The existing front-runner (cand_g) is already well-supported. The innovator should identify angles that:
1. Could strengthen the paper's theoretical contribution
2. Are feasible given the training-free constraint
3. Add novelty without contradicting existing findings

**Revised front-runner**: The innovator suggests that the theoretical contribution (optimal compression framing) could be strengthened by formalizing the structural/pathological decomposition using information-theoretic measures. This is a NEW contribution that builds on H7 but goes beyond it.

---

## Phase 5: Final Proposal

### Title
**"Decomposing Feature Absorption: Structural Artifacts vs. Pathological Learning in Sparse Autoencoders"**

Alternative: **"What Does Feature Absorption Measure? An Information-Theoretic Decomposition"**

### Hypothesis
Feature absorption in SAEs consists of two components: (1) a structural component that is inevitable given overcomplete dictionary learning, and (2) a pathological component that represents harmful learned representations. The structural component dominates in well-trained SAEs, explaining why absorption is real but benign.

### Motivation
The field lacks consensus on whether absorption is a failure mode or optimal behavior. Chanin et al. proved absorption is a consequence of sparsity optimization, but did not decompose it into structural vs. pathological components. Our H7 (trained SAEs have 8x lower absorption than random SAEs) suggests that training reduces absorption, but does not explain why. An information-theoretic decomposition would provide the theoretical grounding for the "absorption is benign" claim.

### Method
1. **Estimate mutual information I(encoder; decoder)**: Measure how much information the encoder activations share with decoder directions across absorbed vs. non-absorbed features.
2. **Decompose absorption**: Separate structural absorption (high mutual information, decoder direction intact) from pathological absorption (low mutual information, decoder direction corrupted).
3. **Validate with steering experiments**: Test whether features with high structural absorption retain steering capability (they should, as decoder direction is preserved).

### Cross-Domain Insight
The structural/pathological decomposition is analogous to signal/noise separation in ICA (Independent Component Analysis). In ICA, the goal is to recover independent sources from mixed signals. In SAEs, the goal is to recover independent features from superimposed activations. Absorption corresponds to the case where the "source" (parent feature) is not recovered because the "mixture" (child feature direction) dominates.

The key transplanted principle: Information-theoretic criteria (e.g., InfoMax) can distinguish between recoverable signal (structural) and irrecoverable noise (pathological). The same principle could distinguish structural absorption from pathological absorption.

### Experimental Plan
1. **Compute mutual information**: For each absorbed feature, compute I(encoder_activation; decoder_direction) using KDE-based estimation
2. **Classify features**: Features with high mutual information = structural absorption; features with low mutual information = pathological absorption
3. **Validate with steering**: Test whether structural vs. pathological absorbed features differ in steering success
4. **Cross-layer validation**: Repeat at L4 and L8 to confirm pattern consistency

**Resource estimate**: Small-scale analysis on existing data. No new experiments required. Mutual information estimation on 26 features × 2 layers is computationally trivial (< 10 minutes).

### Resource Estimate
- Mutual information estimation: ~10 minutes (small-scale)
- Steering validation: Already completed in prior iterations
- Cross-layer: Already completed in prior iterations
- **Total new work**: ~15-30 minutes (analysis only)

### Risk Assessment
1. **Risk**: Mutual information estimation is noisy and may not provide clear separation between structural and pathological absorption.
   - **Mitigation**: Use multiple MI estimators (KDE, binning, kNN) and check for consistency.

2. **Risk**: The decomposition may not reveal anything new beyond what H5 (precision invariant, recall variable) already shows.
   - **Mitigation**: H5 shows decoder alignment is preserved. MI decomposition could explain WHY decoder alignment is preserved (because the structural component has high mutual information).

3. **Risk**: The theoretical contribution may be seen as incremental over Chanin et al.'s Proposition 2.
   - **Mitigation**: Chanin proved absorption is a consequence of sparsity optimization. We would prove that most absorption is structural (not pathological) by showing mutual information is preserved. This is complementary.

### Novelty Claim
First information-theoretic decomposition of SAE absorption into structural vs. pathological components. The structural component corresponds to "optimal compression" behavior where the decoder direction is preserved (high mutual information). The pathological component would correspond to cases where absorption corrupts the decoder direction (low mutual information). Our H7 suggests most absorption in well-trained SAEs is structural.

---

## Integration with Existing Research

The innovator perspective adds theoretical grounding to the existing empirical findings:

| Existing Finding | Innovator Contribution |
|---|---|
| H7 (trained < random absorption) | Explains WHY: trained SAEs minimize pathological absorption while preserving structural absorption |
| H5 (precision invariant, recall variable) | Explains WHY: decoder alignment (precision) is preserved because structural absorption has high mutual information |
| H1-H4 (null results) | Consistent with structural absorption: if absorption is structural, it should not degrade downstream tasks |
| cand_g (optimal compression) | Information-theoretic framing strengthens the optimal compression claim |

## Recommendations

1. **Adopt information-theoretic framing** as a theoretical contribution alongside the empirical findings
2. **Conduct quick MI analysis** on existing data to validate the structural/pathological split
3. **If MI decomposition validates the hypothesis**, the paper's contribution becomes: (a) empirical null results, (b) metric validation (trained < random), (c) theoretical decomposition of absorption
4. **If MI analysis is inconclusive**, retain the optimal compression framing (cand_g) without the decomposition

The innovator perspective should be seen as a potential enhancement to the existing front-runner, not a replacement. The existing findings (null results, H7, H5) are solid. The information-theoretic framing could strengthen the theoretical contribution if it validates.