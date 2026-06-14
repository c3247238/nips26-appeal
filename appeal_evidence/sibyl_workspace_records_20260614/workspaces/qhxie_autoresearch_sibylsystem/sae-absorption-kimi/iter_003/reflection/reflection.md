# Reflection Report: Iteration 004

## Iteration Summary

Iteration 004 successfully reverted from the problematic Goodhart's Law framing of iter_003 back to a clean construct-validity study. The paper now presents three research questions (H1-H3) with real empirical data throughout. The supervisor score improved from 6.5 (iter_003) to 7.0 (iter_004), driven by honest reframing, removal of estimated data, and methodological soundness. The core iter_002 results---H2 hierarchy specificity failure (t=-4.748, p=0.003, d=-1.794)---remain the strongest evidence and survive scrutiny.

However, iter_004 introduced a new and serious data integrity issue: the paper reports Random-SAE semantic-hierarchy absorption as 0.352 (identical to Standard SAE), but the actual experimental data shows Random=0.175. This appears to be a copy-paste error from iter_003's e1_decomposition_results.json, where estimated Random scores were copied from Standard. The iter_004 paper inherited this error in Table 1 and throughout the text. This is the most critical issue requiring immediate correction.

Other issues from previous iterations persist: Cohen's d inconsistency (-1.68 reported vs -1.794 actual), missing multiple comparison correction, perfect probe ceiling effect unaddressed, and thin experimental breadth (one model, one layer, 10 shallow hierarchies).

---

## Issue Analysis by Category

### EXPERIMENT (Critical)

**1. Random-SAE data error: 0.352 reported but actual data shows 0.175 (CRITICAL)**

The paper's Table 1 and abstract claim "Random-SAE control yields semantic-hierarchy absorption of 0.352, identical to the Standard SAE." However, the actual experimental output (semantic_hierarchy_pythia_results.json) shows Random mean=0.175, and statistical_analysis_summary.json confirms Random semantic-hierarchy=0.175, non-hierarchy=0.233. The paper's central claim that Random=Standard on semantic hierarchies is factually wrong.

**Root cause**: This error traces back to iter_003's e1_decomposition_results.json, which copied Standard SAE scores (0.352, 0.416) into the Random row as estimated values. Iter_004's paper inherited this error---the writer used the incorrect values from the previous iteration's estimated data rather than verifying against the actual iter_004 experimental output.

**Impact**: The paper's most striking finding (Random=Standard) is contradicted by the actual data. If reviewers check the data, this will severely damage credibility.

**2. Cohen's d inconsistency: -1.68 reported but source shows -1.794 (CRITICAL)**

The abstract and Section 3.3 report Cohen's d=-1.68, but e3_ttest_results.json shows d=-1.7944989617096718. This inconsistency persists from iter_002 and iter_003. The value -1.68 does not appear in any source JSON file.

**Root cause**: The writer used a rounded or incorrect value without verifying against the source data. This may have been manually computed or inherited from an earlier draft.

**3. H1 underpowered with n=7 (MAJOR)**

The construct-validity test (H1) has bootstrap 95% CI [-0.389, 0.981], spanning nearly the full correlation range. The abstract features the point estimate (r=0.463) without sufficient emphasis on the uncertainty.

**Root cause**: Small sample size is a genuine limitation, not an error. The paper correctly labels this "inconclusive" but the abstract could emphasize uncertainty more prominently.

**4. Perfect probe ceiling effect (AUROC=1.0 for all hierarchies) (MAJOR)**

All 10 semantic hierarchies achieve perfect probe AUROC=1.0 on residual activations (80/80 scores). This ceiling effect means the absorption formula's numerator is bounded by (1.0 - acc_sae), minimizing absolute absorption scores. The paper mentions this as a limitation but does not address whether the ceiling effect itself invalidates the metric adaptation.

**Root cause**: The experimental design uses hierarchies that are too easily discriminable by the base model. This is a design limitation, not a data error.

**5. GPT-2 replication near-zero scores (MAJOR)**

GPT-2 small shows near-zero absolute scores (Standard: 0.000 hierarchy, 0.025 non-hierarchy; TopK: 0.003 hierarchy, 0.098 non-hierarchy). The paper interprets this as "model-specific behavior," but an alternative is that the metric adaptation simply doesn't work on GPT-2.

**Root cause**: The GPT-2 SAEs achieve perfect k-sparse accuracy on most hierarchies, suggesting they are not being stressed by the semantic-hierarchy task. This could indicate either genuine model differences or metric adaptation failure.

**6. Non-hierarchy control boundary blurring (MAJOR)**

Pairs like "tree-wood" and "river-water" involve meronymy (part-whole) or strong thematic relationships, arguably types of hierarchical structure. This blurs the hierarchy-specificity test.

**Root cause**: Control pair selection did not rigorously exclude all hierarchical relationships.

### ANALYSIS (Major)

**7. No multiple comparison correction (MAJOR)**

The paper does not apply multiple comparison correction across H1-H3 and various tests within each. At alpha=0.05 with ~8 reported tests, ~0.4 false positives are expected. The hierarchy specificity result (p=0.0032) would survive Bonferroni correction (0.0032 * 8 = 0.026 < 0.05), but this is not reported.

**Root cause**: This has been flagged across all iterations but never addressed. The statistical analysis code lacks a multiple comparison correction step.

### WRITING (Minor)

**8. Figure extension mismatch (MINOR)**

Figure references use .pdf extension (fig2_scatter.pdf, fig3_specificity.pdf, fig4_robustness.pdf) but actual files are .png. This will cause LaTeX compilation errors.

**Root cause**: The iter_003 figure references were not updated when figures were regenerated as .png files.

**9. Missing code repository URL (MINOR)**

The paper mentions "open-source replication materials" but does not provide a code repository URL.

**10. k=10 sensitivity untested (MINOR)**

The paper uses k=10 for k-sparse probing but does not test whether this is appropriate for all architectures.

---

## Resource Efficiency Assessment

### GPU Utilization

The gpu_progress.json shows 15 completed tasks with zero failures. However, the actual GPU utilization assessment is complicated:

- **Iter_004 experiments**: All tasks produced real empirical data (no estimates). The semantic hierarchy evaluation (119s), non-hierarchy control (72s), and statistical analysis were efficient.
- **Tau_fs robustness**: 1340s total for 3 thresholds across 8 architectures---reasonable given the full SAEBench evaluation pipeline.
- **GPT-2 replication**: 31s for 2 architectures---very fast, but the near-zero scores suggest the task may not be stressing the SAEs.
- **First-letter baseline**: 544s for 8 architectures from iter_002---reused without re-running.

**Estimated GPU utilization**: ~90% (all tasks completed without idle time).

### Bottleneck Analysis

1. **Data integrity verification**: The most time-consuming issue was NOT computational but the manual verification required to discover the Random-SAE data error. A pre-writing automated check comparing paper numbers to source JSON would have caught this in seconds.
2. **LaTeX compilation**: Figure extension mismatches caused compilation delays.
3. **Small sample size for H1**: Cannot be fixed computationally---requires collecting more SAE architectures.

### Scheduling Improvements

- Add a pre-writing data integrity script that verifies every number in the paper against source JSON files (~2 minutes, zero GPU)
- The tau_fs robustness analysis could be parallelized across thresholds if multiple GPUs are available
- GPT-2 replication with only 2 architectures is thin; adding more architectures would be better use of GPU time

---

## Quality Trend Assessment

| Iteration | Score | Novelty | Soundness | Experiments | Reproducibility |
|-----------|-------|---------|-----------|-------------|-----------------|
| iter_001  | 5.5   | 7       | 5         | 4           | 6               |
| iter_002  | 5.5   | 7       | 5         | 5           | 5               |
| iter_003  | 6.5   | 8       | 6         | 6           | 6               |
| iter_004  | 7.0   | 8       | 7         | 6           | 7               |

The quality trajectory is **improving but fragile**. The score improved from 6.5 to 7.0, driven by:
- Successful reframing back to construct-validity study (removing Goodhart's Law overreach)
- All data is now real (no estimates)
- Honest reporting of negative results continues
- Writing quality is strong (score 8/10 from writing review)

However, the improvement is undermined by:
- The Random-SAE data error (critical credibility risk)
- Cohen's d inconsistency (data integrity issue)
- Persistent issues across iterations (multiple comparison correction, ceiling effect)

The score improvement is genuine this time (unlike iter_003's false positive driven by framing novelty), but the data error is a ticking time bomb.

---

## Root Cause Analysis

### Why did the Random-SAE data error persist?

The error originated in iter_003's e1_decomposition_results.json, where estimated Random SAE scores were copied from Standard SAE values. When iter_004 reverted to the construct-validity framing, the writer likely referenced the previous iteration's paper or summary files rather than verifying against the actual iter_004 experimental output (semantic_hierarchy_pythia_results.json). The system lacks a **pre-writing data integrity gate** that automatically compares all numbers in the paper to their source JSON files.

### Why does Cohen's d keep being wrong?

The value -1.68 has persisted across iter_002, iter_003, and iter_004 despite the source data showing -1.794. This suggests the writer is using a cached/incorrect value from an early draft rather than recomputing or checking the source. The statistical analysis summary markdown does not include Cohen's d, so the writer may not know where to find the correct value.

### Why is multiple comparison correction still missing?

This has been flagged in every iteration's reflection but never addressed. The statistical analysis code (statistical_analysis.py) does not include a multiple comparison correction step. This is a systemic issue: reflection recommendations are not automatically propagated to the experiment code.

---

## System Self-Check Response

No `logs/self_check_diagnostics.json` was present for this iteration. However, the issues identified would have been caught by the following self-checks:

1. **Data provenance check**: Verify all values in tables have corresponding pipeline execution records. The Random-SAE value 0.352 in Table 1 has no source in semantic_hierarchy_pythia_results.json (which shows 0.175).
2. **Cross-iteration consistency check**: When reverting framing or reusing previous data, verify that all numbers are re-sourced from the current iteration's experimental output, not inherited from previous iterations.
3. **Statistical test completeness check**: Verify that all reported statistical tests include appropriate corrections (Bonferroni, BH) when multiple comparisons are made.
4. **Effect size verification**: Verify that all reported effect sizes (Cohen's d, eta-squared) match the source statistical output files.

These checks should be added to the pre-writing pipeline and run automatically before LaTeX compilation.
