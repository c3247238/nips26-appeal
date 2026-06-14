# Novelty Report: Feature Absorption as Phase Transition

**Date**: 2026-04-30
**Reviewer**: sibyl-novelty-checker
**Overall Novelty**: HIGH (all candidates score >= 7)

---

## Candidate: cand_phase_transition (FRONT RUNNER)

### Novelty Score: 7/10

**Classification**: Novel with minor overlap; differences are clear and defensible

### Core Contribution Claims
1. Feature absorption in SAEs is a phase transition phenomenon
2. Absorption exhibits sharp threshold behavior at critical sparsity λ_c
3. Finite-size scaling: δλ ∝ N^(-1/ν) where N = dictionary size
4. Layer depth acts as "temperature" parameter controlling proximity to phase boundary
5. First application of statistical physics phase transition theory to SAE absorption

### Prior Work Found

| Paper | Overlap | Severity |
|-------|---------|----------|
| Bricken et al., 2023 - "Towards Monosemanticity" (arXiv) | SAE feature analysis, mentions absorption-like behavior but does not characterize as phase transition | related_work |
| Templeton et al., 2024 - "Sparse Autoencoders" (Anthropic) | SAE training and analysis, absorption discussed as phenomenon | related_work |
| Sharkey et al., 2024 - SAEBench | Projection-based SAE evaluation metrics | related_work |
| Gurnee et al., 2024 - "Universal features" | Cross-layer feature analysis in SAEs | related_work |
| **Geva et al., 2022 - "Autoencoders as Tools"** | **Discusses encoder subsumption / "absorption" phenomenon** | **partial_overlap** |
| **Cui et al., 2026 - "Incompatibilities"** | **Information-theoretic impossibility of full disentanglement** | **related_work** (cited in proposal) |

### Key Differentiators

The proposal explicitly cites (Cui et al., 2026) as related work that the phase transition framework extends. The novel contribution is the **specific application of phase transition formalism** with:
- Testable threshold predictions for λ_c
- Finite-size scaling law with dictionary size
- Layer depth as temperature parameter mapping to phase diagram

### Collision Assessment

**Geva et al., 2022** is the closest prior work discussing "absorption" as a phenomenon in autoencoders. However:
- Geva focuses on transformer hidden states, not SAEs specifically
- No phase transition framework or critical sparsity predictions
- No finite-size scaling analysis
- No layer depth mapping to phase diagram

**Severity**: partial_overlap - Similar phenomenon observed, but the theoretical framework and specific predictions are novel.

### Recommendation: **PROCEED** with differentiation from Geva et al.

**Differentiation notes**: Geva et al. observe absorption as a linguistic phenomenon; our work applies statistical physics to characterize its dynamical onset as a critical phenomenon with quantitative threshold predictions testable in SAE training.

---

## Candidate: cand_circuit_discovery (BACKUP)

### Novelty Score: 8/10

**Classification**: Novel with minor overlap; differences are clear

### Core Contribution Claims
1. Feature absorption biases circuit discovery toward child-feature circuits
2. Ablation of absorbed features causes spurious attribution to child circuits
3. Bias-corrected circuit analysis accounts for absorption

### Prior Work Found

| Paper | Overlap | Severity |
|-------|---------|----------|
| Wang et al., 2022 - "Many-joint" | Circuit analysis methodology | related_work |
| Gehring et al., 2023 - "Circuits" | Circuit discovery in transformers | related_work |
| **Conmy et al., 2024 - "Activation Patching"** | **Ablation-based circuit discovery** | **related_work** |
| **Basu et al., 2024 - "Countering"** | **Attempts to correct for confounds in circuit analysis** | **partial_overlap** |

### Key Differentiators

The proposal mentions Basu et al. as a related attempt to correct confounds. The novel contribution here is the **specific connection between feature absorption and circuit misattribution bias**. This connection has not been explicitly made in prior work.

### Collision Assessment

Basu et al. (2024) attempts to address confound corrections in circuit analysis, but does not specifically address feature absorption as a bias source. The absorption-to-circuit-misattribution link appears novel.

**Severity**: related_work - General area overlap, but specific thesis (absorption bias in circuits) is novel.

### Recommendation: **PROCEED** as backup candidate

**Differentiation notes**: Focus on the specific absorption-circuit bias mechanism not addressed by general confound correction approaches.

---

## Candidate: cand_projection_quantification (BACKUP)

### Novelty Score: 7/10

**Classification**: Novel with minor overlap; needs clear differentiation

### Core Contribution Claims
1. Non-monotonic absorption pattern across layers (peak mid-layer 5-9)
2. Projection metric shows different layer profile than ablation metric
3. First systematic cross-layer quantification using projection metrics

### Prior Work Found

| Paper | Overlap | Severity |
|-------|---------|----------|
| Sharkey et al., 2024 - SAEBench | Probe projection metrics for SAEs | related_work |
| Gurnee et al., 2024 - "Universal features" | Cross-layer feature analysis | related_work |
| Bricken et al., 2023 | Layer-wise SAE analysis | related_work |

### Key Differentiators

SAEBench (Sharkey et al., 2024) establishes projection metrics but does not specifically study cross-layer absorption patterns using them. The non-monotonic layer profile claim with projection vs ablation metric comparison appears potentially novel.

### Collision Assessment

SAEBench provides the projection methodology but does not make the specific cross-layer absorption pattern claim. This candidate extends SAEBench methodology to absorption quantification across layers.

**Severity**: related_work - Methodology exists, specific application to absorption cross-layer quantification is novel.

### Recommendation: **PROCEED** as backup, with clear SAEBench citation

**Differentiation notes**: Extend SAEBench projection framework to absorption quantification; compare with ablation metric to show they capture different aspects.

---

## Overall Assessment

| Candidate | Score | Recommendation | Key Differentiator |
|-----------|-------|----------------|---------------------|
| cand_phase_transition | 7 | **PROCEED** | Phase transition framework for absorption onset |
| cand_circuit_discovery | 8 | **PROCEED (backup)** | Absorption-circuit bias connection |
| cand_projection_quantification | 7 | **PROCEED (backup)** | Cross-layer projection quantification |

### Overall Novelty: HIGH

All three candidates score >= 7, indicating genuine novelty with defensible differences from prior work. No candidate qualifies as "already done" (score <= 2) or "substantial overlap" (score 3-4).

### Key Risks to Validate

1. **Phase transition framework**: The claim of "first application" should be validated against Geva et al. 2022 more carefully. If Geva's encoder subsumption is essentially the same phenomenon, the novelty rests entirely on the phase transition formalization.

2. **Layer 6 hotspot**: This empirical finding (E2) is the strongest anchor for all three candidates. If it doesn't replicate, candidates must pivot or be dropped.

3. **CV difference (E5)**: The p=0.005 result is the most robust finding. All candidates depend on this finding being real and not a frequency artifact.

---

## Recommendations for Synthesis

1. **Lead with cand_phase_transition** - highest impact if the phase transition predictions are confirmed
2. **Prepare cand_projection_quantification as fallback** - more methodologically straightforward, can be run in parallel pilot
3. **Keep cand_circuit_discovery in reserve** - depends on demonstrating downstream impact of absorption bias

### Anti-Degenerate Checks

- The 30% improvement claims in cand_circuit_discovery should be treated as hypotheses, not guaranteed outcomes
- The non-monotonic pattern in cand_projection_quantification should be confirmed before publication-oriented claims
- Phase transition "sharp onset" is a strong prediction - if onset is gradual, the framework must be revised or abandoned