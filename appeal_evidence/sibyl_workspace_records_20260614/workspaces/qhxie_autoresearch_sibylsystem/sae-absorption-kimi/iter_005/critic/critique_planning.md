# Planning Critique: Component-Isolated Study of SAE Absorption Reduction

## Executive Summary

The experimental plan is well-designed in principle: a component-isolated ablation study on ground-truth synthetic data is the correct approach for causal attribution. The plan includes appropriate controls (Random SAE), statistical procedures (ANOVA, Tukey HSD, Cohen's d, bootstrap CI), and pilot validation. However, the execution deviated from the plan in ways that undermine validity: (1) the plan promised 6 variants with 5 replicates each, but only 3 were completed; (2) the plan's pre-registered primary comparisons (MultiScale vs Baseline, Full Matryoshka vs Baseline) could not be tested; (3) the plan's ANOVA and mixed-effects model were not reported; (4) the L0-matched comparison promised in the plan was not delivered.

## Critical Issues

### 1. Plan Promised 6 Variants, Delivered 3 (CRITICAL)

The task_plan.json and methodology.md both specify 6 SAE variants with 5 replicates each:
- Baseline (full: 5 replicates)
- +TopK (full: 5 replicates)
- +MultiScale (pilot only: 1 replicate, 1024 features)
- +Orthogonality (full: 5 replicates)
- +Gating (not trained)
- +Full Matryoshka (not trained)

The gpu_progress.json shows all tasks as "completed," but this is misleading: "full_multiscale," "full_gating," and "full_matryoshka" may be marked complete without actual 5-replicate data on 16384 features.

The paper's scope note (Section 1.5) acknowledges this honestly, but the Abstract and Conclusion present findings as if the full plan was executed.

**Fix**: The plan should have included explicit go/no-go criteria for proceeding with partial data. The current paper should be reframed as a "partial results" or "preliminary findings" paper, not a complete study.

### 2. Pre-Registered Comparisons Could Not Be Tested (CRITICAL)

The plan pre-registered two primary comparisons:
- +MultiScale vs. Baseline
- +Full Matryoshka vs. Baseline

These were chosen because they correspond to architectures with the strongest prior claims (Bussmann et al., 2025). However:
- MultiScale has only pilot data (not comparable to full Baseline)
- Full Matryoshka was not trained at all

The pre-registered primary comparisons could not be executed as planned. The paper instead focuses on TopK vs Baseline, which was not a pre-registered primary comparison. This is a deviation from the pre-registered design.

**Fix**: Acknowledge explicitly that the pre-registered primary comparisons could not be tested due to incomplete data. State that TopK vs Baseline emerged as the strongest finding from the available data.

### 3. Statistical Analysis Plan Partially Abandoned (MAJOR)

The plan promises five statistical procedures:
1. One-way ANOVA across all completed variants — NOT REPORTED
2. Pre-registered primary comparisons — COULD NOT BE TESTED
3. Post-hoc Tukey HSD — NOT REPORTED (only 3 variants, limited value)
4. Effect sizes (Cohen's d) — REPORTED (this is the main analysis)
5. Correlation analysis (L0 vs absorption) — REPORTED but overstated

The paper effectively abandoned the ANOVA and Tukey HSD in favor of pairwise effect sizes. This is reasonable given only 3 completed variants, but the plan should have been updated to reflect this.

**Fix**: Either report the ANOVA or remove it from the Method section. Update the statistical analysis plan to reflect what was actually done.

### 4. L0-Matched Comparison Promised but Not Delivered (MAJOR)

Section 3.6 of the paper promises: "Since variants may achieve different sparsity levels, we report absorption per unit L0 to control for sparsity differences."

This control is never reported in the Results. Given that the paper's central claim is about sparsity-absorption coupling, the L0-matched comparison is essential for validating whether the effect is genuinely about sparsity or about something else.

**Fix**: Compute and report absorption/L0 for each variant:
- Baseline: 0.252 / 964 = 2.61e-4 per unit L0
- TopK: 0.056 / 50 = 1.12e-3 per unit L0
- Orthogonality: 0.245 / 550 = 4.45e-4 per unit L0

If the sparsity-absorption coupling is genuine, absorption/L0 should be similar across variants. If not, the effect is not purely sparsity-driven.

## Moderate Issues

### 5. Pilot-to-Full Scale-Up Risk Not Addressed

The pilot used 1024 features; the full experiment uses 16384 features (16x scale-up). The pilot results (e.g., MultiScale absorption = 0.050) may not generalize to the full scale. The plan does not address this risk explicitly.

**Fix**: Add a discussion of scale-up risks in the Limitations section.

### 6. Single Sparsity Level for TopK

The plan tests only k=50 for TopK. The optimal k for absorption-reconstruction trade-offs is unknown. The plan does not include a sparsity sweep.

**Fix**: Add a sparsity sweep to Future Work (already partially present in Section 5.6).

### 7. Missing Baseline: L1-Tuned Baseline

The Baseline uses L1 sparsity with lambda_1 = 5e-3, yielding L0 = 964. A tuned L1 baseline that achieves L0 = 50 (matching TopK) would isolate whether the effect is from the sparsity level or the TopK mechanism specifically.

**Fix**: This is acknowledged as a limitation. Consider adding an L1-tuned baseline to future work.

## What Works Well

1. **Pilot validation before full experiment.** The 4-condition pilot with go/no-go criteria is excellent experimental design.

2. **Random control.** Including an untrained Random SAE validates metric discrimination.

3. **Multiple replicates.** 5 replicates with different seeds provides variance estimates.

4. **Ground-truth metrics.** Eliminating probes removes a major confound.

## Summary

The plan was sound but the execution fell short. The paper should be reframed as reporting partial results from a planned 6-variant study, with the component ranking treated as provisional. The pre-registered primary comparisons could not be tested, and several promised controls (L0-matched, ANOVA) were not delivered.
