# Visual Audit Report (Revision Round 1)

## Summary

- **Total figures**: 6 (Figures 1--6)
- **Total tables**: 4 (Tables 1--4)
- **Generated PDFs on disk**: 4 (fig1_teaser.pdf, fig2_layer_absorption.pdf, fig3_perclass_heatmap.pdf; table1_probe_quality.pdf, table2_crossdomain.pdf, table4_hypothesis_verdicts.pdf)
- **Missing PDFs**: 3 (fig4_patching_comparison.pdf, fig5_pathological_histogram.pdf, fig6_architecture_comparison.pdf)

## Completeness Check

### Figures present and referenced

| Figure | File | Status | First reference |
|--------|------|--------|-----------------|
| Figure 1 | fig1_teaser.pdf | EXISTS | Section 1 (Introduction) |
| Figure 2 | fig2_layer_absorption.pdf | EXISTS | Section 4.2 |
| Figure 3 | fig3_perclass_heatmap.pdf | EXISTS | Section 4.3 |
| Figure 4 | fig4_patching_comparison.pdf | MISSING | Section 5.1 |
| Figure 5 | fig5_pathological_histogram.pdf | MISSING | Section 5.3 |
| Figure 6 | fig6_architecture_comparison.pdf | MISSING | Section 6 |

### Tables present and referenced

| Table | File | Status | First reference |
|-------|------|--------|-----------------|
| Table 1 | table1_probe_quality.pdf | EXISTS | Section 3.2 |
| Table 2 | table2_crossdomain.pdf | EXISTS | Section 4.1 |
| Table 3 | inline markdown | PRESENT | Section 5.4 |
| Table 4 | table4_hypothesis_verdicts.pdf | EXISTS | Section 7 |

## Consistency Check

- **Figure numbering**: Sequential (1--6) with no gaps. All figures referenced before appearance.
- **Table numbering**: Sequential (1--4) with no gaps. All tables referenced before appearance.
- **Caption style**: Uniform sentence case, descriptive, self-explanatory. Each caption includes key statistical results.
- **Table 3 column headers**: Updated to use formal notation (FN$_{\text{strict}}$, FN$_{\text{comp}}$, FN$_{\text{persist}}$) consistent with notation.md.

## Flow Check

- Every figure and table is referenced in the text BEFORE it appears.
- No orphan figures.
- Visual narrative supports text narrative:
  - Teaser (Fig 1) in Introduction.
  - Probe quality (Table 1) precedes all absorption measurements.
  - Layer profile (Fig 2) and per-class heatmap (Fig 3) in Results.
  - Patching comparison (Fig 4) and pathological histogram (Fig 5) in Mechanism Analysis.
  - Architecture bars (Fig 6) in Architecture Analysis.
  - Hypothesis summary (Table 4) in Discussion.
- "Figures and Tables" listing at the end of paper removed (reviewer recommendation: unnecessary for conference format).

## Issues Found and Fixed (Revision Round 1)

### Critical

1. **Table 3 data mismatch resolved**: FN counts and percentages corrected to match authoritative data source (`hedging_crossdomain.json`, `cross_hierarchy_comparison_L24_16k`). Old values: first-letter 156 FN, city-continent 325 FN with nonzero persistent fractions. Corrected values: first-letter 291 FN, city-continent 418 FN, city-language 124 FN, city-country 515 FN; persistent = 0.0% for all hierarchies. Compensatory percentages also corrected (e.g., city-continent: 88.3% -> 93.8%).

2. **Three figure PDFs remain missing**: fig4_patching_comparison.pdf, fig5_pathological_histogram.pdf, fig6_architecture_comparison.pdf. These need generation from source data before LaTeX compilation.

### Major

3. **Section 7.2 verbatim repetition eliminated**: Three sentences that duplicated Section 5.1--5.2 content were rewritten. Section 7.2 now provides new interpretive content (mitigation design space constraints, hedging decomposition mechanism explanation) instead of restating results.

4. **Architecture Kruskal-Wallis p-values corrected**: Hierarchy p-values updated from 0.005/0.041 (consolidation summary, older analysis) to 0.010/0.063 (authoritative full-mode architecture_comparison.json). L24 hierarchy effect is now honestly reported as marginal (p = 0.063), not significant.

5. **Kruskal-Wallis N corrected**: Changed from 3,566 (source unclear) to 3,545 (1,330 + 1,073 + 1,142 probe-correct instances across three RAVEL hierarchies, from full-mode cross-domain analysis). Clarified that the test covers three RAVEL hierarchies, not four.

6. **Probe quality sourcing clarified**: Added sentence in Section 3.2 documenting that all absorption measurements use probes from a single full-mode training run (seed 42).

### Minor

7. **H3 verdict footnoted**: Table 4 now includes a dagger footnote explaining the discrepancy between SUPPORTED (full-mode data, chi-square p = 1.0e-19) and PARTIALLY_SUPPORTED (pilot data, consolidation summary).

8. **Banned pattern removal**: "These findings do not invalidate SAEs" replaced with evidence-first paragraph. "These findings have direct implications" replaced with specific statement. Section 2.3 generic opener removed.

9. **Conclusion opener revised**: "This study provides the first systematic..." replaced with direct statement about what was done.

10. **End-of-paper figure listing removed**: Per reviewer guidance, the "Figures and Tables" section at the end was removed.

## Missing Visuals -- Action Required

Three figure PDFs need generation before the paper can compile:

1. **fig4_patching_comparison.pdf**: Paired box plots showing child-zeroed vs. control recovery rates for first-letter (left) and city-continent (right). Data source: iter_008 activation patching + phase2_activation_patching_crossdomain.json.
2. **fig5_pathological_histogram.pdf**: Histogram of |logit change| for parent direction ablation with inset control distribution. Data source: phase2/benign_pathological.json.
3. **fig6_architecture_comparison.pdf**: Grouped bar chart of absorption rate by architecture x hierarchy. Data source: phase1/architecture_comparison.json. Note: p-values in caption updated to match corrected values.

## Suggestions

No additional visuals needed for the main text. The 6-figure, 4-table count is appropriate for a ~10-page main body. Appendix figures (GAS scatter, rate-distortion scatter, threshold sensitivity heatmap) should be included when appendices are written.
