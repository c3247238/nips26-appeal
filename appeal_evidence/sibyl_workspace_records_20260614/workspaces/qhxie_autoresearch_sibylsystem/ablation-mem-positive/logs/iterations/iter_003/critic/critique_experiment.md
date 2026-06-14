# Critique: Experiment Design and Execution

## Overview
The experimental execution is generally sound with appropriate controls and statistical corrections. The main experiment (H1: CV predicts steering) is well-designed with proper replication. However, there are significant issues with the non-absorbed baseline comparison and the incomplete cross-architecture validation.

## Major Issues

### 1. Non-Absorbed Baseline Comparison Is Flawed
**Experiment**: `full_non_absorbed_baseline`

**Problem**: The non-absorbed baseline experiment used different conditions than the main H1 experiment:
- **Main H1 experiment**: 30 high-CV + 30 low-CV absorbed features, 5 prompts, strengths +3/+5/+7
- **Non-absorbed baseline**: 30 non-absorbed features, 3 prompts, only strength +5

The paper claims (Section 4.5): "absorbed high-CV features are steerable to approximately the same degree as non-absorbed features. The difference is 0.0045, which is not practically significant."

**Impact**: This comparison is misleading because:
1. The absolute effect for high-CV absorbed at +5 (main experiment) is **0.5222**, not 0.097
2. The 0.097 figure appears to be from a different sub-analysis or pilot
3. A proper comparison would use identical prompts and strengths

**Evidence**:
- `full_steering_cv.json`: high_cv_mean_abs_effect at +5 = 0.5222
- `full_non_absorbed_baseline.json`: non_absorbed mean_abs_effect = 0.1019 (at +5)
- `full_non_absorbed_baseline.json`: absorbed_high_cv mean_abs_effect = 0.0975

The 0.0975 vs 0.1019 comparison appears valid but uses only 3 prompts vs 5. The 0.5222 vs 0.0975 gap suggests either:
- The 3-prompt vs 5-prompt average dilutes the high-CV effect, OR
- The non-absorbed baseline features were selected differently

**Recommendation**: Rerun non-absorbed baseline with identical conditions (5 prompts, +3/+5/+7 strengths) or qualify the claim as "preliminary under reduced prompt set."

### 2. Cross-Architecture Validation Is Incomplete
**Experiment**: `full_cross_architecture`

**Status**: DONE marker exists but no quantitative results in paper

Section 4.6 states: "While the experiment completed, detailed integration of cross-architecture results remains future work."

**Impact**: The paper makes cross-architecture generalization claims in abstract and conclusions without presenting results. This is a significant gap.

**Recommendation**: Remove all cross-architecture generalization claims until results are properly integrated. The preliminary nature should be explicit.

### 3. H6 Falsification Lacks Mechanism Discussion
**Experiment**: `full_decoder_orthogonality`

**Result**: NOT_SUPPORTED - r = -0.136, p = 0.30

The experiment was well-executed (proper orthogonality computation, appropriate statistical tests). However, the paper does not discuss:
- Why orthogonality was hypothesized as the mechanism
- What the negative result implies for the context-sensitive vs bypass routing hypothesis
- Whether the routing mechanism operates at a different level than decoder geometry

### 4. Pilot-to-Full Effect Size Discrepancy
**Pilot** (Section 4.2): High-CV = 0.153, Low-CV = 0.075 (2.03x ratio)
**Full experiment** (Section 4.3): High-CV = 0.522, Low-CV = 0.355 (1.47x ratio at +5)

The effect sizes increased substantially (5x for high-CV, 4.7x for low-CV) while the ratio decreased (2.03x to 1.47x).

**Possible explanations**:
1. Different feature sets (pilot used different feature selection)
2. Different prompt sets (5 prompts vs pilot's implicit prompt set)
3. Different random seeds or feature sampling

This discrepancy should be explained or acknowledged.

## Minor Issues

### 5. Multiple Hypotheses Tested, Single Primary Outcome
The experimental program tested H1-H6 across multiple experiments. Only H1 is confirmed as primary finding. The multiple hypothesis framework is appropriate but the paper should clarify:
- What percentage of experimental time/resources went to H1 vs other hypotheses
- Whether H6 and H3 were pre-registered or exploratory

### 6. Statistical Reporting
The p-values reported as 0.0 are likely lower-bound artifacts (e.g., scipy returns p < 1e-300). This is fine for significance reporting but should note the actual lower bound.

### 7. Feature Selection Reproducibility
The classification of features into high-CV/low-CV and absorbed/non-absorbed depends on:
- Input distribution (1000 text samples)
- Random seed for feature selection
- Absorption_score threshold (0.5 is arbitrary)

These should be reported more completely for reproducibility.

## What Works Well

1. **Appropriate statistical correction**: BH correction across steering strengths is correct
2. **Proper negative result reporting**: H6 falsification is clearly stated
3. **Adequate sample sizes**: 30 features per group, 150 samples per condition
4. **Multiple steering strengths**: Good dose-response design
5. **Pilot validation before full experiment**: Appropriate sequential validation
6. **Activation patching confirmation**: Establishes genuine causal structure
7. **Clean experiment organization**: full_* experiments are clearly separated with DONE markers