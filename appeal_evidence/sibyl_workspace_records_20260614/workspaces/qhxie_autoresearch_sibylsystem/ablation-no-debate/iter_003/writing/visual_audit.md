# Visual Audit Report

## Overview

This report audits all visual elements referenced in the integrated paper against the Figure & Table Plan in `outline.md`. This revision addresses critical factual errors in the Results section where pilot experiment data was incorrectly used instead of full experiment data. All 5 figures remain missing; tables are properly present.

---

## Completeness Check

### Planned Figures (from outline.md)

| Figure | Title | Status | Location in Paper |
|--------|-------|--------|-------------------|
| Figure 1 | Absorption Phenomenon Illustration | **MISSING** | Abstract, Section 1 (intro teaser) |
| Figure 2 | H_Mech 2x2 Factorial Results | **MISSING** | Section 4.1 ("Figure 2") |
| Figure 3 | Hierarchy Strength vs Absorption | **MISSING** | Section 4.2, Conclusion ("Figure 3") |
| Figure 4 | Pareto Frontier | **MISSING** | Section 4.3, Conclusion ("Figure 4") |
| Figure 5 | Safety vs Non-Safety Feature Absorption | **MISSING** | Section 4.4 ("Figure 5") |

**Status**: All 5 figures are referenced in text but NO actual figure image files or LaTeX figure code exists in the manuscript. The paper's figure list at the end lists the figures but they have not been generated.

### Planned Tables (from outline.md)

| Table | Title | Status | Location in Paper |
|-------|-------|--------|-------------------|
| Table 1 | Experimental Configuration | Present | Section 3.7 |
| Table 2 | H_Mech Full Experiment (5 seeds) | Present | Section 4.1 |
| Table 3 | H_Comp Full Experiment | Present | Section 4.2 |
| Table 4 | H_Pareto Full Experiment | Present | Section 4.3 |
| Table 5 | H_Safe Pilot Results | Present | Section 4.4 |
| Table 6 | GPT-2 Small Held-Out Validation | Present | Section 4.5 |
| Table 7 | Hypothesis Test Summary | Present | Section 4.6 |

**Status**: All 7 tables present and properly formatted in LaTeX style.

---

## Critical Fixes in This Revision

### Issue 1: H_Comp Results Corrected (Previously Pilot Data)
- **Previous error**: Section 4.2 reported pilot data (0.479-0.821 range, R²=0.984) as if it were full experiment results
- **Fixed**: Now reports full experiment data (0.814-1.201 range, R²=0.04, FAILED)
- **Data source**: h_comp_6levels_3seeds.json shows actual full experiment results

### Issue 2: H_Pareto Results Corrected (Previously Pilot Data)
- **Previous error**: Section 4.3 reported pilot data (sensitivity 0.837→0.495, absorption 0.329→0.508, R²=0.963) as if it were full experiment results
- **Fixed**: Now reports full experiment data (absorption=0 across all L0 levels, sensitivity=0.1054 stable, INCONCLUSIVE)
- **Data source**: h_pareto_4l0_3seeds.json shows absorption_mean=0.0 across all L0 levels

### Issue 3: H_Mech Table Updated with Full Experiment Data
- **Previous error**: Table 2 showed pilot (seed 42) results: B=0.076, D=0.017
- **Fixed**: Table 2 now shows full 5-seed experiment results: B=0.055, D=0.017
- **Key change**: B≈D still holds (delta=0.037), confirming encoder sufficiency

### Issue 4: Hypothesis Summary Corrected
- **Previous error**: Table 7 showed H_Comp and H_Pareto as "CONFIRMED" with wrong metrics
- **Fixed**: Table 7 now correctly shows H_Comp (FAILED, R²=0.04) and H_Pareto (INCONCLUSIVE, degenerate)
- **H_Mech correctly split**: encoder sufficiency CONFIRMED, decoder irrelevance FAILED

### Issue 5: Conclusion Updated with Correct Results
- **Previous error**: Conclusion claimed "monotonic relationship" and "Pareto frontier confirmed" with R²=0.984
- **Fixed**: Conclusion now correctly states H_Comp failed (R²=0.04) and H_Pareto is inconclusive (degenerate)

---

## Consistency Check

### Figure Numbering
- Figures 1-5 referenced in text but actual image files do not exist
- Tables 1-7 numbered consistently

### Terminology Consistency
- **Feature absorption**: Consistent throughout
- **Pareto frontier**: American English spelling, consistent
- **Cross-Model Validation** (Section 4.5) vs **Held-Out Validation** (Table 6 caption): Now consistent — Section 4.5 calls it "Cross-Model Validation" with Table 6 caption noting "(Held-Out Validation)" which is acceptable

### Notation Consistency (vs. notation.md)
- $A_{multi}(p)$ notation used for multi-child proportional absorption
- $\alpha$ for hierarchy strength
- $R^2$ for coefficient of determination
- $B \approx D$ and $C \approx A$ notation for factorial checks

### Result Status Terminology
- **CONFIRMED**: H_Mech (encoder sufficiency only)
- **FAILED**: H_Comp, decoder irrelevance check
- **INCONCLUSIVE (degenerate)**: H_Pareto
- **NULL**: H_Safe (positive for safety analysis)

---

## Critical Outstanding Issue: Missing Figures

The paper cannot be submitted without generating all 5 figures. Priority order:

1. **Figure 1 (conceptual diagram)**: Conceptual illustration showing how child features substitute for parent features. This is the teaser figure referenced in the introduction.

2. **Figure 2 (2x2 factorial bar chart)**: Bar chart showing Conditions A/B/C/D absorption rates from h_mech_full experiment. Core experimental result showing Condition C's extreme variance.

3. **Figure 3 (hierarchy strength line plot)**: Line plot of cosine similarity vs absorption from h_comp_full experiment. Should show non-monotonic scatter, not a clean increasing line.

4. **Figure 4 (Pareto frontier scatter)**: Scatter plot of sensitivity vs absorption from h_pareto_full experiment. Should show degenerate case (all points at absorption=0).

5. **Figure 5 (safety violin plot)**: Violin plot comparing safety vs non-safety feature absorption from h_safe_gemma pilot and GPT-2 Small held-out.

### Data Sources for Figure Generation

| Figure | Data Source | Key Data Points |
|--------|-------------|-----------------|
| Figure 1 | Conceptual (manual illustration) | N/A |
| Figure 2 | h_mech_full experiment | A=0.184, B=0.055, C=12.28±17.13, D=0.017 |
| Figure 3 | h_comp_full experiment | 6 cosine levels with absorption 0.51-1.20, high variance |
| Figure 4 | h_pareto_full experiment | 4 L0 levels all at absorption=0, sensitivity=0.1054 |
| Figure 5 | h_safe_gemma pilot + held_out_validation | Gemma: both groups 0.0; GPT-2: safety 233.13, non-safety 221.70 |

---

## Summary

- **Total figures referenced**: 5 (Figure 1-5)
- **Total tables referenced**: 7 (Table 1-7)
- **Figures present**: 0 (all MISSING)
- **Tables present**: 7 (all present and properly formatted)
- **Critical factual errors fixed**: 5 (H_Comp, H_Pareto, H_Mech pilot→full, hypothesis summary, conclusion)
- **Critical missing content**: 5 figures (must be generated before submission)

The paper required substantial correction of experimental results — the Results section was reporting pilot data while claiming it was full experiment results. This has been corrected. The paper still requires figure generation before it can be submitted for external review.