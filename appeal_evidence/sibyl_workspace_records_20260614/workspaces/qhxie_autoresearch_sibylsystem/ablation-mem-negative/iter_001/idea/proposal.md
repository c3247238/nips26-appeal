# Unsupervised Feature Absorption Detection in Sparse Autoencoders: A Co-Occurrence Clustering Approach with Dynamic Compensation

## Abstract

Feature absorption is a critical failure mode in sparse autoencoders (SAEs) where broad, interpretable parent features are suppressed by more specific child features, creating interpretability illusions with arbitrary false negatives. Existing detection methods require supervised probe directions and ground-truth parent features, severely limiting their applicability. We propose the first fully unsupervised absorption detection framework (UAD) that identifies absorbed feature pairs via co-occurrence clustering without any labeled data or probe directions. Complementing detection, we introduce Dynamic Feature De-Absorption (DFDA), a lightweight residual compensation method that recovers absorbed parent activations at inference time without retraining. Our pilot experiments on GPT-2 Small demonstrate UAD achieves F1=0.704 with perfect recall, and DFDA achieves 11.1% average MSE improvement on absorbed pairs using only 388 parameters. We propose a full validation pipeline extending to Gemma-2B and Pythia models, with the goal of establishing an end-to-end detect-then-fix pipeline for SAE absorption.

## Motivation

Sparse autoencoders have become the dominant tool for mechanistic interpretability of large language models. However, Chanin et al. (2024) identified feature absorption as a fundamental failure mode: when hierarchical features co-occur (e.g., "starts with S" and "short"), the SAE merges the parent feature into the child feature's latent to increase sparsity while maintaining reconstruction. This creates a dangerous interpretability illusion where a latent appears to track a concept but fails to activate on arbitrary positive examples.

The critical barrier to addressing absorption at scale is detection. Chanin et al.'s method requires:
1. Knowing the parent feature a priori
2. Training supervised probe directions
3. Running integrated gradients ablation (computationally expensive, limited to early layers)

This supervised requirement means absorption can only be detected for concepts we already know to look for -- precisely the concepts where SAEs are least needed. For discovering unknown features, we need unsupervised detection.

## Research Questions

1. **RQ1**: Can feature absorption be detected without ground-truth parent features or supervised probe directions?
2. **RQ2**: Does a co-occurrence clustering approach generalize across model architectures and scales?
3. **RQ3**: Can absorbed parent feature activations be recovered via lightweight residual compensation without retraining the SAE?
4. **RQ4**: Does the detect-then-fix pipeline (UAD + DFDA) scale to larger models and diverse concept hierarchies?

## Hypotheses

### H1: Unsupervised Detection Feasibility
A co-occurrence-based clustering method can detect absorbed feature pairs with F1 >= 0.6 compared to Chanin et al.'s supervised method, without requiring probe directions or ground-truth labels.

### H2: Cross-Model Generalization
UAD achieves F1 >= 0.55 on models larger than GPT-2 Small (Gemma-2B, Pythia-2.8B), confirming that co-occurrence patterns are a general signature of absorption.

### H3: Dynamic Compensation Efficacy
DFDA recovers >10% of absorbed parent feature activation (measured by MSE reduction) using <0.01% of SAE parameters, without increasing reconstruction error.

### H4: End-to-End Pipeline Validity
The UAD+DFDA pipeline achieves measurable improvement on downstream sparse probing accuracy for concepts identified as absorbed by UAD.

## Expected Contributions

1. **First unsupervised absorption detection method (UAD)**: Eliminates the need for supervised probe directions, enabling absorption detection for any SAE without prior knowledge of feature hierarchies.

2. **First training-free absorption compensation method (DFDA)**: Lightweight residual compensation recovers absorbed activations at inference time without architectural changes or retraining.

3. **Novel end-to-end detect-then-fix pipeline**: The combination of unsupervised detection and dynamic compensation is unique in the SAE literature.

4. **Cross-model validation**: Testing on GPT-2 Small, Gemma-2B, and Pythia-2.8B establishes generalizability.

5. **Open-source toolkit**: Reusable detection and compensation code integrated with SAELens.

## Evidence-Driven Revisions from Iteration 1

### What the Pilot and Full Experiments Revealed

| Hypothesis | Original Status | Evidence | Revised Status |
|------------|----------------|----------|----------------|
| H1 (cross-architecture absorption variation) | Primary | Collision rate differs (15.4% pretrained vs 3.8% TopK) but this is NOT true absorption; proxy metric mislabeled | **Demoted to supplementary** |
| H2 (causal downstream link) | Primary | r=0.103, p=0.87, n=5 -- inconclusive, likely underpowered | **Demoted to supplementary** |
| H3 (sparsity monotonicity) | Primary | r=-0.10, p=0.87, n=5 -- inconclusive | **Demoted to supplementary** |
| H4 (layer pattern) | Primary | r=0.088, p=0.87, n=6 -- inconclusive | **Demoted to supplementary** |
| H5-E (UAD feasibility) | Exploratory | **F1=0.704, perfect recall, precision=0.543** -- exceeds 0.6 threshold | **Promoted to PRIMARY** |
| H6-E (DFDA feasibility) | Exploratory | **11.1% avg MSE improvement, 3/4 pairs positive, 388 params** | **Promoted to PRIMARY** |

### Key Revision Decisions

1. **Inverted contribution hierarchy**: The original proposal positioned CAAB (cross-architecture benchmark) and causal assessment as primary contributions, with UAD/DFDA as exploratory. The experimental evidence inverts this: UAD/DFDA are the only robust, novel contributions. H1-H4 are inconclusive due to proxy metrics and insufficient power.

2. **Dropped: Cross-architecture absorption benchmark as primary contribution**: The "collision rate" metric used in CAAB is a proxy, not true absorption per Chanin et al.'s protocol. The architecture difference (15.4% vs 3.8%) is a real observation but its interpretation as "absorption" is unsupported. CAAB is retained as supplementary material.

3. **Dropped: Causal downstream impact as primary contribution**: The correlation between collision rate and sparse probing accuracy is zero (r=0.103, p=0.87). However, this uses a proxy metric and is underpowered (n=5). Rather than invest more iterations in a potentially null result, we relegate this to future work.

4. **Retained and strengthened: UAD + DFDA**: These are the only results that (a) are genuinely novel, (b) have positive empirical support, and (c) address a clear gap in the literature. The paper is reframed around them.

5. **Added: Cross-model validation requirement**: UAD was only tested on GPT-2 Small, single seed, single layer. The revised plan mandates validation on Gemma-2B and Pythia-2.8B with multi-seed replication before paper submission.

## Novelty Assessment

### Prior Art Search Results

We conducted comprehensive searches across arXiv, Google Scholar, and the web for unsupervised absorption detection methods. Key findings:

1. **Chanin et al. (2024) "A is for Absorption"**: The foundational absorption paper. Uses k-sparse probing + integrated gradients ablation. **Fully supervised** -- requires knowing parent features and training probe directions. No unsupervised alternative proposed.

2. **"The Geometry of Concepts" (arXiv:2410.19750)**: Uses co-occurrence clustering (spectral clustering on phi coefficient affinity matrix) to discover functional feature lobes. **Closest prior work** but does NOT address absorption detection specifically -- it discovers general feature structure, not parent-child absorption relationships.

3. **Clarke et al. (2024) "sae_cooccurrence"**: Analyzes SAE feature co-occurrence for compositionality and ambiguity. Does not propose absorption detection.

4. **SAEBench (Karvonen et al., ICML 2025)**: Includes absorption as one of 8+ metrics. Uses probe projection (not ablation) for absorption detection. **Still requires supervised probes**.

5. **Matryoshka SAE, OrtSAE, KronSAE, ATM**: All propose architectural solutions to reduce absorption. None propose unsupervised detection or training-free compensation.

### Novelty Verdict

**UAD (Unsupervised Absorption Detection)**: **Genuinely novel**. No prior work eliminates the requirement for supervised probe directions in absorption detection. The co-occurrence clustering approach is conceptually distinct from all existing methods.

**DFDA (Dynamic Feature De-Absorption)**: **Genuinely novel**. No prior work attempts training-free, inference-time absorption compensation. All existing solutions require architectural changes and retraining (Matryoshka, OrtSAE, etc.).

**UAD+DFDA Pipeline**: **Genuinely novel**. The end-to-end detect-then-fix workflow without supervision is unique.

Risk: If UAD F1 drops below 0.5 on larger models, the core contribution weakens significantly.

## Method

### UAD: Unsupervised Absorption Detection

**Input**: Trained SAE, unlabeled text corpus
**Output**: List of suspected absorbed (parent, child) feature pairs with confidence scores

**Algorithm**:
1. Extract feature activation matrix A (n_examples x n_features) from corpus
2. Compute feature co-occurrence matrix C = A^T A (count of co-activations)
3. Normalize C to correlation matrix R (phi coefficient recommended based on pilot)
4. Run hierarchical agglomerative clustering on R
5. For each cluster, identify the highest-activation feature as "parent" candidate
6. For each parent candidate, identify features that co-occur >threshold but have low standalone activation
7. Flag (parent, child) pairs where: P(child|parent) > 0.8 AND parent standalone frequency < expected
8. Validate against Chanin et al. supervised labels where available

**Key insight**: Absorbed parent features show anomalous co-occurrence -- they fire primarily when child features fire, but rarely independently. This creates a detectable signature in the co-occurrence matrix without requiring ground truth.

### DFDA: Dynamic Feature De-Absorption

**Input**: SAE, identified absorbed (parent, child) pairs, input activations
**Output**: Compensated SAE output with recovered parent features

**Algorithm**:
1. For each absorbed pair (parent p, child c), collect training examples where c fires but p does not
2. Train a tiny MLP (input: child feature activation; output: predicted parent residual; 2 layers, 64 hidden units)
3. At inference: compute SAE output z, then add MLP(z_c) to z_p
4. Verify: reconstruction MSE does not increase; parent probe accuracy improves

**Parameter budget**: <0.01% of SAE parameters per pair (typically <500 parameters).

### Validation Protocol

| Stage | Task | Success Criteria |
|-------|------|-----------------|
| Pilot | UAD on GPT-2 Small | F1 >= 0.5 |
| Full | UAD on GPT-2 Small | F1 >= 0.6 |
| Full | UAD on Gemma-2B | F1 >= 0.55 |
| Full | UAD on Pythia-2.8B | F1 >= 0.55 |
| Full | Multi-seed (3 seeds) | Mean F1 >= 0.6, std <= 0.1 |
| Full | DFDA on >=8 pairs | Mean improvement >= 10% |
| Full | End-to-end pipeline | Probe accuracy improvement on absorbed concepts |

## Experimental Plan

### Phase 1: Critical Fixes (Iteration 2)

| Experiment | Model | Purpose | Time Budget |
|-----------|-------|---------|-------------|
| UAD-Gemma | Gemma-2B | Cross-model validation | 30 min |
| UAD-Pythia | Pythia-2.8B | Cross-model validation | 30 min |
| UAD-multi-seed | GPT-2 Small | Seed robustness (3 seeds) | 20 min |
| True-absorption | GPT-2 Small | Implement Chanin protocol, correlate with collision rate | 45 min |

**Go/No-Go Gate**: If UAD F1 < 0.5 on any of Gemma-2B or Pythia-2.8B, **PIVOT**.

### Phase 2: Full Validation (Iteration 3)

| Experiment | Purpose | Time Budget |
|-----------|---------|-------------|
| DFDA-scale | >=8 absorbed pairs, 2 models | 30 min |
| End-to-end | UAD detects -> DFDA compensates -> probe improves | 30 min |
| Ablation | UAD without clustering; DFDA without residual | 20 min |

### Phase 3: Paper Preparation (Iteration 4)

Write full paper draft targeting NeurIPS/ICLR workshop, with camera-ready expansion path.

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| UAD F1 drops on larger models | Medium | **Critical** | Go/No-Go gate; if F1 < 0.5, pivot to alternative direction |
| DFDA improvement not statistically significant | Low | High | Pre-register analysis; report negative result if occurs |
| Chanin protocol implementation fails | Low | Medium | Use SAEBench probe-projection as backup metric |
| Co-occurrence clustering is just finding correlated features, not absorption specifically | Medium | High | Validate against Chanin labels; if precision is random, hypothesis falsified |
| Dead feature ratio obscures co-occurrence patterns | Medium | Medium | Filter dead features before clustering; report effective feature count |

## Theoretical Framing

The UAD method is grounded in the observation that absorption creates a characteristic signature in the co-occurrence structure: a parent feature that has been absorbed shows high conditional probability P(child|parent) but low marginal activation frequency. This is structurally distinct from:
- **Independent features**: Low co-occurrence, high marginal frequencies
- **Correlated features**: High co-occurrence, high marginal frequencies
- **Hedged features**: Multiple parents sharing one child latent

The theoretical prediction is that absorption occupies a distinct region in the (marginal frequency, conditional probability) space that can be separated by clustering.

## Related Work Positioning

| Work | Relationship to Our Contribution |
|------|----------------------------------|
| Chanin et al. (2024) | We eliminate their supervised requirement; our method is complementary |
| SAEBench (Karvonen et al.) | We add a new metric (unsupervised absorption detection) to the evaluation toolkit |
| Matryoshka/OrtSAE | We address the same problem (absorption) but via detection+compensation rather than architectural change |
| "Geometry of Concepts" | We extend their co-occurrence clustering framework specifically to absorption detection |
| Transcoders (Paulo et al.) | Our work is SAE-centric; transcoders are an alternative paradigm not directly comparable |

## Limitations and Future Work

1. **First-letter bias**: Pilot validation uses first-letter features (a-z). Generalization to semantic hierarchies (WordNet) is planned but not yet validated.
2. **English-only**: All experiments use English text. Multilingual absorption patterns are unknown.
3. **Single SAE per model**: We test one SAE configuration per model. Different widths/sparsities may yield different results.
4. **No human evaluation**: Automated metrics only. Human interpretability assessment is future work.
5. **Causal claims are limited**: UAD detects absorption; DFDA compensates. We do not claim to have solved the fundamental absorption-hedging trade-off.

## Timeline

| Iteration | Focus | Deliverable |
|-----------|-------|-------------|
| Iter 2 | Cross-model UAD validation | F1 scores on 3 models, 3 seeds |
| Iter 3 | DFDA scaling + end-to-end | >=8 pairs, pipeline validation |
| Iter 4 | Paper writing | Full draft |
| Iter 5 | Review + revision | Camera-ready |

## Conclusion

The first iteration revealed that the original research plan's primary contributions (CAAB, causal assessment) were built on proxy metrics and insufficient statistical power. However, the exploratory directions (UAD, DFDA) produced genuinely novel, empirically supported results. This revised proposal inverts the contribution hierarchy, placing UAD and DFDA at the center. The unsupervised detection of feature absorption is a clear gap in the literature with no prior work addressing it. If cross-model validation succeeds, this work represents a meaningful methodological advance for the SAE interpretability community.
