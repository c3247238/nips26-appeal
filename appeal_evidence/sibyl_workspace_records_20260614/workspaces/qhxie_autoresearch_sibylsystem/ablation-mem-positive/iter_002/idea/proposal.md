# Research Proposal: Feature Absorption as Phase Transition in Sparse Autoencoders

## Status: DRAFT FOR PILOT

## Abstract

We propose that feature absorption in Sparse Autoencoders (SAEs) is a phase transition phenomenon. Using statistical physics as our theoretical framework, we predict that absorption exhibits sharp threshold behavior at critical sparsity values rather than gradual increase. Our prior experiments identified Layer 6 as an "absorption hotspot" with 0.549% absorption rate and the most fragmented graph structure (9 components) — consistent with a system near its critical point. We propose experiments to: (1) confirm the phase transition boundary, (2) validate finite-size scaling with dictionary size variation, and (3) map layer depth as a control parameter on the phase diagram. The empirical anchor is the robust E5 finding that absorbed features have significantly lower coefficient of variation (CV: 1.07 vs 1.46, p=0.005).

## Motivation

Feature absorption — where hierarchical features cause general (parent) features to be subsumed by specific (child) features during SAE sparse optimization — is a central problem in mechanistic interpretability. However:

1. **Detection is broken**: The ablation-based suppression_ratio metric is uniformly 1.0 across all pairs (no discrimination); no high-confidence pairs (>0.7) found in 5 rounds of experiments
2. **Theoretical gap**: Why co-occurrence correlates negatively with absorption score (r=-0.52) is unexplained by existing theory
3. **Layer 6 hotspot unexplained**: The consistent finding that Layer 6 shows highest absorption signal across multiple metrics lacks a mechanistic account

The phase transition framework provides a unifying explanation for all three issues.

## Research Questions

1. Does SAE absorption exhibit threshold (critical) behavior as sparsity penalty λ varies, or does it increase gradually?
2. Is Layer 6's absorption hotspot explained by the system being near a critical point in the phase diagram?
3. Does layer depth act as a "temperature" parameter controlling proximity to the absorption phase boundary?

## Hypotheses

| ID | Hypothesis | Prediction | Test |
|----|-----------|------------|------|
| H1 | Critical sparsity: absorption onset is sharp, not gradual | Below λ_c: near-zero absorption; above λ_c: sharp increase | Sparsity sweep λ ∈ [1e-5, 1e-2], measure m(λ) |
| H2 | Finite-size scaling: transition sharpness scales with dictionary size | δλ ∝ N^(-1/ν) where N = num latents | Compare 8k/16k/32k dictionaries |
| H3 | Layer depth as tuning parameter | Early layers: λ << λ_c (no absorption); Mid layers: λ ≈ λ_c (peak heterogeneity); Late layers: λ >> λ_c (saturated) | Cross-layer absorption measurement |
| H4 | CV difference as critical phenomenon indicator | Absorbed features (at critical point) have lower CV, consistent with E5 | Validate CV-absorbed correlation |

## Evidence-Driven Revisions

### Prior Evidence (E1-E5)

From our pilot experiments on GPT-2 Layer 6:
- **E1**: suppression_ratio = 1.0 uniformly — detection metric is degenerate
- **E2**: Layer 6 has 0.549% absorption rate (highest), 9 graph components (most fragmented), mean edge weight 0.559 (highest) — consistent with critical point
- **E4**: Co-occurrence negatively correlates with absorption score (r=-0.52) — the "paradox" explained by information bottleneck theory
- **E5**: Absorbed features have significantly lower CV (1.07 vs 1.46, p=0.005) — this is our most robust finding

### How This Proposal Addresses Prior Feedback

The E5 CV finding anchors the empirical validity of absorption. The phase transition framework explains WHY absorbed features have lower CV: at the critical point, feature representations coalesce into dominant channels, reducing variance. This is analogous to susceptibility peaks in physical systems at phase transitions.

## Method

### Phase Transition Framework

| Physics Concept | SAE Concept |
|-----------------|-------------|
| Order parameter m | Mean absorption score across candidate pairs |
| Control parameter λ | Sparsity penalty / feature hierarchy depth |
| Critical point λ_c | Threshold where absorption onset becomes widespread |
| Susceptibility χ = dm/dλ | Rate of absorption change with sparsity |
| Domain formation | Absorption graph components |
| Layer depth as "T" | Early=low T, mid=critical, late=high T |

### Experimental Design

**E6: Sparsity Sweep (Critical Sparsity)**
- Target: GPT-2 layer 6 SAE, 16k latents
- Vary L1 sparsity λ ∈ [1e-5, 1e-2] in 8 steps
- Measure: mean absorption score m(λ), susceptibility χ(λ)
- Expected: sharp onset at λ_c where χ is maximized
- Duration: 30 min

**E7: Finite-Size Scaling**
- Target: GPT-2 layer 6 with 8k/16k/32k dictionaries
- Measure: m(λ) curves for each dictionary size
- Expected: scaling collapse to universal function
- Duration: 40 min

**E8: Layer Phase Diagram**
- Target: Layers 0, 3, 6, 9, 11
- Measure: effective λ_c per layer
- Expected: Layer 6 at critical point (λ ≈ λ_c)
- Duration: 45 min

### Absorption Measurement (Metric Choice)

Given the degeneracy of ablation-based metrics, we use **decoder cosine similarity** as the primary signal:
- `score = decoder_cosine × log(freq_ratio)` (simplified from E4 findings)
- Candidates: top 100 pairs by decoder cosine similarity
- Ground truth validation: CV difference (E5 result) as convergent evidence

## Connection to Prior Theoretical Work

Our framework synthesizes:
1. **Cui et al. (ICLR 2026)**: Information-theoretic impossibility of full disentanglement — reframed as absorption being mathematically optimal for compression efficiency
2. **Theoretical perspective (Candidate A)**: Information bottleneck explains negative co-occurrence correlation — high co-occurrence causes parent to be "explained away" by child
3. **E5 CV finding**: Absorbed features have lower CV — explained as critical point phenomenon where representation coalesces

## Risks and Mitigations

| Risk | Mitigation |
|------|-------------|
| Phase transition is gradual, not sharp | Fall back to rate-distortion framework (Candidate C from interdisciplinary) |
| Layer 6 hotspot doesn't replicate in other models | Cross-model validation in E9 |
| CV difference is a frequency artifact | Match on activation frequency, not raw CV |

## Novelty Assessment

This is the **first application of statistical physics phase transition theory to SAE absorption**. Prior work (Chanin et al., 2024; Cui et al., 2026) describes absorption but does not characterize its dynamical onset as a critical phenomenon with testable threshold predictions.

## Expected Contributions

1. **Predictive framework**: Sharp threshold predictions for absorption onset
2. **Unified explanation**: Layer 6 hotspot as critical point phenomenon
3. **Design guidelines**: Operating SAEs at λ << λ_c avoids absorption
4. **Connection to physics**: Methods from disordered systems become available

## Changes from Prior Round

This is the first proposal for this project. No prior proposal exists.

---

## Backup Ideas (for potential pivot)

### Backup 1: Absorption-Adjusted Circuit Discovery

**Title**: Absorption-Adjusted Circuit Discovery: Quantifying and Correcting Feature Absorption Bias in Mechanistic Interpretability

**Core insight**: Feature absorption systematically biases circuit discovery toward child-feature circuits. When circuit analysis ablates absorbed features, the resulting output change is spuriously attributed to the child circuit rather than the parent circuit.

**Why backup**: Directly addresses Gap 3 (downstream impact on interpretability tasks). Uses existing circuit analysis tools without requiring new SAE training. Addresses the contrarian's actionability concern by connecting absorption to a concrete interpretability failure mode.

**Pilot focus**: Validate absorption bias in circuit discovery on GPT-2 layer 6.

### Backup 2: Projection-Based Cross-Layer Quantification

**Title**: Layer-Dependent Feature Absorption in Sparse Autoencoders: A Projection-Based Quantification

**Core insight**: Absorption rates follow a non-monotonic pattern across network depth (mid-layer peaks), robust across architectures when using probe projection metrics rather than ablation.

**Why backup**: Addresses Gap 1 (cross-layer quantification) with the more robust SAEBench probe projection metric. Even null results (no cross-layer variation) are publishable as correction to field assumptions.

**Pilot focus**: Run projection-based absorption on GPT-2 layer 6 to establish baseline.
