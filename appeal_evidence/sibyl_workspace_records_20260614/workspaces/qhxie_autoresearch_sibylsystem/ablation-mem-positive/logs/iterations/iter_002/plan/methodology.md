# Methodology: SAE Feature Absorption Phase Transitions
## Updated from Iteration 2 Experiments

## Overview

This methodology refines the phase transition analysis based on completed experiments (iterations 1-2) and addresses critical validation gaps identified in the evolution lessons.

## Key Findings from Completed Experiments

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: Critical Sparsity | SUPPORTED | λ_c=5e-5, χ_peak=11.19, chi_ratio=1.88 (gradual transition) |
| H2: Finite-Size Scaling | SUPPORTED | ν=3, R²=0.951 across dictionary sizes |
| H3: Layer Criticality | NOT_SUPPORTED | All layers saturate at absorption=1.0 at λ=0.001 |
| H4: CV Difference | SUPPORTED (reversed) | Absorbed CV=6.22 >> Non-absorbed CV=0.01 |
| H5: Co-occurrence | SUPPORTED | r=0.647 (vs baseline -0.52) |
| H6: Graph Topology | NOT_SUPPORTED | Component count decreases with layer |

## Critical Gaps Identified (from Evolution Lessons)

### 1. Activation Patching Validation (NEVER EXECUTED)
The 9 persistent core words (e.g., 'eight', 'lower', 'liked', 'offer', 'often') are claimed as genuine hierarchy-driven absorption, but validation by activation patching was **never performed**.

**Why this is critical:**
- This is the ONLY experiment that can distinguish 'metric miscalibrated' from 'JumpReLU genuinely has minimal absorption'
- Without this, the claim rests on observational cross-L0 persistence classification with known bias
- Estimated: 0.5-1 GPU-hour

### 2. H10 Homeostatic Rebalancing (Phantom Contribution)
H10 appeared as Contribution #4 ('First training-free post-hoc repair') but was never executed (Section 4.7: 'not executed due to negative H6 result'). This is a phantom contribution that must be validated or removed.

### 3. LCA Framework Logical Incoherence
H6 is decisively falsified (precision@20=0.0, p=1.0 Fisher exact). Yet paper continues to rely on LCA framework, claiming mechanism is 'supported' via H7 precision-recall asymmetry. This is logically incoherent.

## Model and SAE Configuration

**Primary Model**: GPT-2 Small (117M params)
- hook_name: "blocks.6.hook_resid_pre" (layer 6 identified as absorption hotspot)

**SAE Configuration**:
- Release: `gpt2-small-res-jb`
- d_model: 768 (GPT-2 residual dimension)
- Sparsity sweep range: λ ∈ [1e-5, 5e-2] (12 log-spaced values)

## Experiment Design

### Phase 1: Activation Patching Validation (NEW - CRITICAL)

**Objective**: Validate that the 9 persistent core words represent genuine hierarchy-driven absorption vs metric artifact.

**Procedure**:
1. Identify 9 persistent core words from CV analysis (high-CV absorbed features that persist across layers)
2. For each word w, compute:
   - Clean activation: SAE encoding on prompt containing w
   - Child-zeroed activation: Zero out the child feature, measure parent recovery
   - Metric: logit change at semantically appropriate token positions

**Pass Criterion**:
- Parent recovers >50% of original activation when child is zeroed → genuine absorption
- Parent recovers <10% → metric artifact, absorption is an epiphenomenon

**Falsification**: If parent shows zero recovery, the persistence is metric artifact.

### Phase 2: Cross-Layer at True Critical Sparsity (REVISED)

**Objective**: Measure absorption at λ_c=5e-5 (not λ=0.001) to test whether layer heterogeneity appears at true critical point.

**Layers**: [0, 3, 6, 9, 11]

**Pass Criterion**: Different layers show different absorption rates at λ_c (not uniform saturation).

### Phase 3: Steering Effectiveness Test (H4/H7 Connection)

**Objective**: Test whether CV predicts steering utility (connects H4 variance paradox to steering actionability).

**Procedure**:
1. Select 30 high-CV absorbed features and 30 low-CV non-absorbed features
2. Apply steering at ±3, ±5, ±7 steering strengths
3. Measure logit change at semantically appropriate tokens

**Pass Criterion**: High-CV features show larger steering effect → CV predicts actionability.

### Phase 4: Figure Generation

**Objective**: Generate publication-ready figures from completed experiments.

**Figures**:
- Figure 1: Phase diagram m(λ) with susceptibility inset (H1)
- Figure 2: Scaling collapse plot for different N (H2)
- Figure 3: Cross-layer absorption heatmap (H3-revised)
- Figure 4: CV comparison bar chart (H4 - reversed direction)
- Figure 5: Activation patching results (new validation)
- Table 1: Summary of hypothesis test results

## Evaluation Criteria

### Critical Threshold Validation
| Criterion | Target | Description |
|-----------|--------|-------------|
| Susceptibility peak | χ_max > 3×χ_avg | Indicates phase transition |
| chi_ratio | > 1.5 for "quasi-critical" | Gradual vs sharp transition |
| Scaling collapse | R² > 0.9 | Curves overlay after rescaling |

### Activation Patching Validation
| Criterion | Target | Description |
|-----------|--------|-------------|
| Parent recovery | > 50% when child zeroed | Indicates genuine absorption |
| Steering effect | High-CV > Low-CV | CV predicts actionability |

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Activation patching shows zero recovery | Medium | Report as metric artifact; update paper narrative |
| Cross-layer still saturated at λ_c | Medium | Conclude layer heterogeneity not observable |
| Steering shows no CV correlation | Low | Report as negative result; Basu et al. similar finding |

## Expected Visualizations

1. **Figure 1**: Phase diagram m(λ) with susceptibility χ inset (H1)
   - X-axis: λ (log scale), Y-axis: absorption rate m(λ)
   - Inset: χ vs λ showing peak at λ_c

2. **Figure 2**: Scaling collapse plot (H2)
   - Rescaled x-axis: λ × N^(1/ν), overlay curves for different N

3. **Figure 3**: Cross-layer heatmap at λ_c=5e-5 (H3-revised)
   - X-axis: layer, Y-axis: λ, color: absorption rate

4. **Figure 4**: CV comparison (H4 - reversed)
   - Bar chart: absorbed (CV~7) vs non-absorbed (CV~0.01)

5. **Figure 5**: Activation patching validation
   - Bar chart: parent recovery % for 9 core words

6. **Table 1**: Hypothesis test summary
   - Columns: Hypothesis, Status, Key Evidence, Interpretation

## Reproducibility

- Seed: 42 for all experiments
- Model version: GPT-2 small (gpt2-small, no finetune)
- SAE release: gpt2-small-res-jb
- Activation cache saved for reuse