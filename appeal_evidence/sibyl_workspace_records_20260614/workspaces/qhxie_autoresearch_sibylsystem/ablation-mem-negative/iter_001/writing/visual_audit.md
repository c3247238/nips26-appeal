# Visual Audit Report

## Summary

- **Total figures planned**: 5
- **Total tables planned**: 3
- **Figures embedded**: 5 (all present in manuscript)
- **Tables inline**: 3 (all present in manuscript)

## Completeness Check

| Planned Visual | Status | Location in Paper |
|---------------|--------|-------------------|
| Figure 1: Architecture comparison bar chart | EMBEDDED | Section 4.2 |
| Figure 2: Absorption vs. accuracy scatter | EMBEDDED | Section 4.3 |
| Figure 3: Sparsity sweep line plot | EMBEDDED | Section 4.4 |
| Figure 4: Layer-depth bar chart | EMBEDDED | Section 4.5 |
| Figure 5: UAD/DFDA grouped bar chart | EMBEDDED | Section 4.6 |
| Table 1: Cross-architecture comparison | PRESENT | Section 4.2 |
| Table 2: Sparsity sweep | PRESENT | Section 4.3 |
| Table 3: Exploratory methods | PRESENT | Section 4.6 |

## Missing Visuals

None. All 5 planned figures are now embedded in the manuscript with proper markdown image references.

## Consistency Issues Found and Fixed

1. **Figure numbering**: Verified consistent (Figure 1--5) across all section references.
2. **Table numbering**: Verified consistent (Table 1--3) across all section references.
3. **Terminology consistency**: "training-free" changed to "SAE-retraining-free" throughout (per Conclusion critique) to clarify that DFDA does train an MLP, just not the SAE itself.
4. **Statistical notation**: Standardized on Spearman $\rho_S$ with exact $p$-values reported (0.870, 0.873, 0.868).
5. **Layer indexing**: Clarified GPT-2 Small uses layers 0--11, with experiments at layers 0, 2, 4, 6, 8, 10.
6. **"15 experiments" claim**: Removed from Conclusion; the exact count is ambiguous (6 full experiments + 4 pilots + analyses).
7. **Table 2 provenance**: Added footnote clarifying data source (f2_causal) and noting f3_sparsity differences.
8. **Section 4.8 Summary**: Removed as redundant with Conclusion.
9. **Discussion 5.1 title**: Changed from "The Absorption Paradox" to "Collision Rate as a Poor Proxy" to avoid overclaiming.

## Flow Check

- All figures are referenced in text before they appear --- OK
- Figure 2 (the central finding) is placed after Table 1 and Figure 1 establish context --- OK
- Figure 5 (exploratory methods) appears at the end of Experiments --- OK
- Figures and Tables consolidated list added at end of paper --- OK

## Quality Check

- Table captions are self-explanatory with clear headers
- Table 1 includes 4 architectures/metrics for comprehensive comparison
- Table 2 shows the full sparsity sweep with all relevant metrics plus data provenance footnote
- Table 3 summarizes both exploratory methods in one view
- All tables use consistent decimal precision (1--3 significant figures)
- Figure captions are self-contained and reference key statistics

## Suggestions for Additional Visuals

1. **Correlation matrix heatmap**: Show relationships between all metrics (collision rate, MSE, L0, probing accuracy, dead feature ratio) across all $k$ values. Would strengthen the "collision rate as poor proxy" narrative.

2. **UAD confusion matrix**: A 2x2 confusion matrix (TP/FP/TN/FN) would make the 54.3% precision / 100% recall more intuitive than the current bar chart plan.

3. **DFDA before/after reconstruction comparison**: A side-by-side visualization of reconstruction quality with and without DFDA compensation for a sample input would make the 11.1% improvement more concrete.

4. **Dead feature ratio overlay**: Adding dead feature ratio as a third axis or annotation to Figure 3 would highlight the confound mentioned in limitations (89--99% dead features).
