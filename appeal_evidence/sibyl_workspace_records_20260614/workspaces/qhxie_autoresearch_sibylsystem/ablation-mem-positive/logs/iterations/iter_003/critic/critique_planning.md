# Critique: Planning

## Overview
The planning documents (methodology.md) are comprehensive and well-structured. However, there is a significant gap between what was planned and what was executed, particularly regarding the cross-architecture validation.

## Major Issues

### 1. Cross-Architecture Validation Planned but Not Completed
**In methodology.md**:
- Section "Cross-Architecture Generalization" is listed as a primary experiment
- Table in Section 8 lists E5: Gemma-2-2B validation (30 min)
- Risk assessment in Section 10 lists "Gemma-2 shows no CV effect" as a medium-risk mitigation

**Actual execution**:
- `full_cross_architecture_DONE` marker exists
- No quantitative Gemma-2-2B results appear in the paper
- Section 4.6 explicitly says results "remain future work"

**Impact**: The planning document suggests cross-architecture validation was a primary goal, but it was never integrated into the paper. This creates a gap between stated intentions and actual contributions.

### 2. Ablation Design vs Execution Mismatch
**In methodology.md** (Section 46-50):
- Ablation 1: High-CV vs Low-CV steering comparison
- Ablation 2: Decoder orthogonality as alternative predictor
- Control: Fano factor (CV²/mean) to control for magnitude

**Actual execution**:
- Ablation 1: Completed (H1 confirmed)
- Ablation 2: Completed (H6 falsified)
- Control (Fano factor): Not explicitly tested

The Fano factor control is mentioned in planning but never explicitly executed or discussed in results. This should be acknowledged.

### 3. Non-Absorbed Baseline Was Not Pre-Registered
**In methodology.md**: No mention of non-absorbed baseline experiment

**Actual execution**: `full_non_absorbed_baseline.json` exists and is discussed in Section 4.5

This appears to have been added ad-hoc after the main experiments. The paper should clarify whether this was planned or post-hoc.

## Minor Issues

### 4. Risk Assessment Not Updated
The risk assessment in methodology.md lists "Gemma-2 shows no CV effect" as a mitigation. But since no Gemma-2 results are presented, this risk cannot be assessed.

### 5. Timeline Estimates Were Inaccurate
Planning estimated ~110 min total. The actual experimental campaign was spread across multiple sessions with separate experiments:
- `full_cv_analysis` (completed separately)
- `full_steering_cv` (primary H1 experiment)
- `full_decoder_orthogonality` (H6)
- `full_non_absorbed_baseline` (supplementary)
- `full_cross_architecture` (incomplete)

Total runtime likely exceeded the estimate significantly.

### 6. Validation Status Table Is Incomplete
The validation status table in methodology.md shows:
| Experiment | Status | Key Evidence |
|------------|--------|--------------|
| pilot_activation_patching | PASSED | 67.3% mean recovery |
| pilot_steering_cv | PASSED | 2x difference |
| full_cv_analysis | COMPLETED | CV difference confirmed |
| full_activation_patching | COMPLETED | Validated in experiment_state |

But the actual completed experiments include:
- `full_steering_cv` (H1)
- `full_decoder_orthogonality` (H6)
- `full_non_absorbed_baseline` (baseline)
- `full_cross_architecture` (incomplete)

The status table is outdated.

## What Works Well

1. **Clear hypothesis structure**: H1, H6, and cross-architecture are clearly separated
2. **Appropriate statistical thresholds**: alpha = 0.01 with BH correction is appropriate
3. **Multiple steering strengths**: Good dose-response design (+3, +5, +7)
4. **Pilot before full experiment**: Appropriate sequential validation
5. **Risk assessment**: Correctly identifies key risks (CV threshold, effect size, Gemma-2)