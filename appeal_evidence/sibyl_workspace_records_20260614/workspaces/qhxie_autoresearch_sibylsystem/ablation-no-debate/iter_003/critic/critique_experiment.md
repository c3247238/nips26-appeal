# Critique: Experimental Design and Execution

## Overview

The experimental design is generally sound—a 2×2 factorial for mechanism decomposition is the correct approach. However, several critical issues undermine confidence in the results.

---

## Critical Issues

### 1. H_Mech Self-Contradiction in Results Interpretation

**Location**: writing/paper.md Section 4.1

**Issue**: The paper states:
- Condition B (trained encoder, random decoder) absorption = 0.076
- Condition D (trained encoder, trained decoder) absorption = 0.017

The conclusion claims "B ≈ D confirms encoder-driven; C ≈ A confirms decoder-irrelevant"

**Math Check**:
- |B - D| = 0.059
- B is 4.5× larger than D

**This is Not Approximately Equal**: If B "approximately equals" D, the difference should be small relative to the values. Here, B is 4.5× larger. The paper's own pass criterion was |B - D| < 0.1, which technically passes, but the interpretation is misleading.

**What the Data Actually Shows**:
1. Encoder alignment creates absorption (B >> A: 0.076 vs 0.004)
2. Decoder training SUPPRESSES absorption in joint training (D < B: 0.017 vs 0.076)

The finding is actually more interesting than framed: encoder alignment creates absorption tendency, but decoder training during joint training regulates/suppresses it. This is a regulatory role, not a "contributes nothing" role.

---

### 2. H_Safe Is Not Actually Tested

**Location**: exp/results/pilots/h_safe_gemma_pilot.json, writing/paper.md Section 4.4

**Evidence**:
```json
"safety_feature_indices": [1024, 2048, 3072, 4096, 5120],
"non_safety_feature_indices": [100, 200, 300, 400, 500],
```

These are arbitrary multiples of 1024 and multiples of 100—they are PLACEHOLDERS, not actual Neuronpedia-validated safety features.

**The pilot explicitly says**: "Pilot experiment - Gemma Scope safety feature absorption." A pilot on placeholder data is not a finding.

**Paper Claims**: "Preliminary experiments (H_Safe pilot) on Gemma Scope SAEs show null absorption for safety-critical features, though this result requires validation with curated feature sets."

**Reality**: The experiment was not done. The features were placeholders. There is no result to report.

---

### 3. Sensitivity Metric Values Are Suspicious

**Location**: exp/results/pilots/h_pareto_pilot.json

**Evidence**:
- L0=16: sensitivity_mean = 1.525, std = 0.1
- L0=64: sensitivity_mean = 0.932

**Issue**: The metric is defined as Var(Δlogits | Δa_f) — variance of logit changes. While variance isn't bounded at 1, if this is being used as a normalized sensitivity metric meant to be in [0,1], values >1 suggest:
1. Wrong formula implementation, OR
2. Missing normalization, OR
3. Wrong scale interpretation

**Impact on Pareto Frontier**: The entire H_Pareto result (R² = 0.963 frontier fit) depends on these sensitivity values. If the metric is incorrectly implemented, the frontier shape is meaningless.

---

### 4. Statistical Fragility in H_Comp

**Location**: writing/paper.md Section 4.2

**Issue**: The monotonic fit R² = 0.984 is based on n=4 unique L0 levels. This is very fragile.

More critically, the table shows:
| Cosine Similarity | Mean Absorption | Std | N |
|-------------------|-----------------|-----|---|
| 0.50 | 0.479 | 0.082 | 3 |
| 0.60 | 0.566 | 0.071 | 3 |
| 0.70 | 0.646 | 0.065 | 3 |
| 0.80 | 0.690 | 0.058 | 3 |
| 0.90 | 0.749 | 0.051 | 3 |
| 0.95 | 0.821 | 0.048 | 3 |

**N=3 per level** for a fit with 6 parameters. The degrees of freedom are extremely limited. R² = 0.984 looks impressive but could be overfitting to noise with n=3 per level.

---

### 5. MCC Validation Shows Chance-Level Recovery

**Issue**: The paper mentions MCC ~0.21 across ALL variants including Random control in evolution lessons but does not adequately address this in the paper proper.

**What This Means**: Hungarian matching on overcomplete dictionaries yields approximately chance-level recovery of ground-truth feature correspondence regardless of training. If the ground-truth matching is not working, what does "absorption reduction" actually mean?

This could reflect:
- Sparsity-induced suppression rather than genuine hierarchical feature recovery
- Metric implementation issues
- Fundamental limitation of the matching approach

---

## Major Issues

### 6. Multi-Seed Validation Gap

**Issue**: The proposal claims "5 seeds for robust fit" (methodology.md) and "Multi-seed validation: Across 5 seeds..." (paper.md Section 4.1). But:
- h_mech_pilot_seed42.json shows only seed 42
- experiment_state.json shows h_mech_full as "running" not completed
- h_comp_pilot.json shows only seed 42 (not the promised 5 seeds)
- h_pareto_pilot.json shows only seed 42

The multi-seed validation promised in the methodology is NOT actually completed. The paper reports "across 5 seeds" but the pilot data is from single seeds.

---

### 7. Experiment State Shows Running Tasks

**Location**: exp/experiment_state.json

```json
"h_mech_full": {"status": "running", ...},
"h_pareto_pilot": {"status": "running", ...}
```

The paper's results section presents findings as if experiments are complete, but h_mech_full and h_pareto_pilot are still marked as "running". This creates a gap between claimed results and actual experimental state.

---

## Minor Issues

### 8. k Value Inconsistency

Methodology specifies k=5 for multi-child proportional ablation. The pilot experiments should verify this is actually used. Different k values would give different absorption rates.

### 9. Layer Mismatch in Cross-Model Validation

GPT-2 Small validation uses layer 8; Gemma Scope uses layer 12. This is not a true cross-layer validation.

---

## Recommendations

### Must Fix
1. **Reword H_Mech interpretation**: "Encoder alignment creates absorption (B >> A) but decoder training in joint training suppresses it (D < B), suggesting a regulatory role for the decoder"
2. **Remove H_Safe claims**: State "methodology proposed, awaiting validated Neuronpedia annotations" not "preliminary experiments show null absorption"
3. **Verify sensitivity metric**: Sensitivity > 1 needs explanation or correction
4. **Resolve experiment state**: Complete h_mech_full and h_pareto_pilot before claiming full results

### Should Fix
5. **Acknowledge statistical fragility**: With n=3 per level, R² = 0.984 may be overfitting
6. **Add MCC discussion to paper proper**: Address what chance-level recovery implies for absorption interpretation

### Consider
7. **Execute full 5-seed validation** as promised in methodology before claiming multi-seed robustness