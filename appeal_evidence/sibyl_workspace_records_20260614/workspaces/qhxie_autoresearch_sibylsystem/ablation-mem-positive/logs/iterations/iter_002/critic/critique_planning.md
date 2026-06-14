# Planning Critique

## Overall Assessment

The planning documents show a sophisticated research design centered on phase transition phenomenology in SAE feature absorption. The methodology is well-structured with clear pass/fail criteria and risk mitigation strategies. However, critical gaps remain: the synthetic-to-real framework was abandoned without documentation, the full_cross_layer_critical experiment results are not integrated, and several validation steps (lambda_c prospective validation, negative controls) were planned but not executed. The planning shows adaptive learning but suffers from scope creep and incomplete execution tracking.

**Score: 5/10**

---

## Critical Issues

### 1. Synthetic-to-Real Framework Abandoned Without Documentation

**Location**: `plan/methodology.md` vs actual experiments

The original proposal described a five-phase synthetic-to-real validation framework with ground-truth labels and ROC-AUC validation. This was entirely abandoned in favor of real-SAE experiments. The deviation is not documented, making it appear as a planning failure rather than a principled pivot.

**Fix**: Add a "Deviations from Plan" section documenting why the synthetic approach was abandoned and what was learned.

### 2. Full_cross_layer_critical Results Not Integrated

**Location**: `exp/experiment_state.json` vs `writing/integrated_paper.md`

The experiment was completed (status: completed) but its results are not visible in the paper. This represents a significant disconnect between planned and executed research.

**Fix**: Integrate results or document as negative finding. If results show uniform saturation at λ_c, this is important negative evidence.

### 3. Evaluation Criteria Not Applied

**Location**: `plan/methodology.md`, Evaluation Criteria

The plan specifies targets (ROC-AUC > 0.85, calibration error < 0.10, cross-parameter stability CV < 15%, etc.) that were never evaluated because the synthetic phase was skipped. The actual evaluation uses simpler pass/fail thresholds.

**Fix**: Acknowledge that the planned evaluation criteria were not applied and explain why the simplified criteria were used instead.

---

## Major Issues

### 4. Risk Mitigation Table Not Followed

**Location**: `plan/methodology.md`, Risk Mitigation table

Several risk mitigations were not followed because the synthetic approach was abandoned. For example:
- "Test 3 complexity levels" for synthetic data → N/A (no synthetic data)
- "5-fold cross-validation" for metric calibration → No
- "SAE training on synthetic data" → N/A

This is understandable given the pivot, but the disconnect should be documented.

### 5. Time Estimates Underestimated Non-Experiment Time

**Location**: `idea/proposal.md`

The proposal estimates "~90 min" total, but the actual iteration clearly took much longer when including analysis, debate, writing, and revision. Experiment execution times (from experiment_state.json) are short (~3 min each) but overall iteration time is much longer.

### 6. Expected Visualizations Not Fully Produced

**Location**: `plan/methodology.md`, Expected Visualizations

The plan lists 7 figures and 2 tables. The actual paper has 6 figures and 4 tables, but several planned figures (calibration plots, parameter sweep heatmaps, confidence interval coverage plot) were omitted.

---

## What Works Well

1. **Clear phase structure**: The five-phase methodology is well-organized with explicit objectives and pass criteria.

2. **Risk identification**: The risk table identifies genuine risks (synthetic simplicity, real SAE hierarchy clarity, metric instability) that would have been relevant if the synthetic approach had been pursued.

3. **Reproducibility planning**: Fixed seeds, JSON configs, and ground-truth label storage are all good practices.

4. **Concrete pass criteria**: The thresholds (e.g., >50% parent recovery for activation patching) are specific and measurable.