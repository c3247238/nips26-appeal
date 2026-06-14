# Visual Audit Report

## Summary

- **Total figures:** 5 (all present as PDF files)
- **Total tables:** 2 formal inline (Table 1: SAE configurations; Table 2: Control results by domain)
- **Additional inline tables:** 3 (cross-domain hierarchy suite in Section 3.6; dimension sensitivity in Section 6.5; dimension sensitivity with Cohen's d in Section 6.5)

## Completeness Check

### Figures from Outline Plan vs. Integrated Paper

| Planned Figure | Status | File | Section |
|---------------|--------|------|---------|
| Figure 1: Universal Control Failure | Present | `figures/control_failure.pdf` | Section 4.1 |
| Figure 2: Hedging Decomposition | Present | `figures/hedging_decomposition.pdf` | Section 4.2 |
| Figure 3: L0-Absorption Phase Transition | Present | `figures/l0_phase_transition.pdf` | Section 5.1 |
| Figure 4: CMI vs Absorption Rate | Present | `figures/cmi_vs_absorption.pdf` | Section 6.2 |
| Figure 5: Dimension Sensitivity | Present | `figures/cmi_dimension_sensitivity.pdf` | Section 6.5 |

All five main-body figures are present. Figure 5 (dimension sensitivity) was promoted from the appendix to the main text at Section 6.5, where the dimension analysis is first conducted, and is cross-referenced from Section 7.3. No placeholders needed.

### Tables from Outline Plan vs. Integrated Paper

| Planned Table | Status | Section |
|--------------|--------|---------|
| Table 1: SAE Configurations | Present (inline) | Section 3.1 |
| Table 2: Control Results by Domain | Present (inline) | Section 4.1 |
| Table 3: Improved First-Letter Results (per-letter) | Not included (detailed per-letter breakdown described in text; suited for appendix) | -- |
| Table 4: L0 x Width Absorption Grid | Not included (results described through regression statistics in Section 5.2) | -- |

Tables 3 and 4 from the outline are referenced in prose rather than as formal tables. The per-letter detail of Table 3 is better suited for an appendix. The width-L0 interaction in Table 4 is sufficiently conveyed through the regression statistics. Neither omission affects comprehension of the main argument.

## Consistency Check

- **Figure numbering:** Sequential (1, 2, 3, 4, 5) across sections. No gaps or duplicates.
- **Table numbering:** Sequential (1, 2). No gaps or duplicates.
- **Color scheme:** Verified against `figures/style_config.py`:
  - Blue (#2196F3) for measured rates / hedging / non-absorbed / layer 12
  - Pink (#E91E63) for shuffled controls
  - Purple (#9C27B0) for random probe controls
  - Red (#F44336) for hierarchy-driven / absorbed letters
  - Gray (#9E9E9E) for reconstruction error / baselines
  - Orange (#FF9800) for layer 10
  - Green (#4CAF50) for layer 20
- **Caption style:** Sentence case, period at end, self-explanatory. Consistent across all five figures.
- **Font sizes:** Governed by style_config.py (FONT_SIZE=11). Consistent.

## Flow Check

- **Figure 1** (control_failure.pdf): Referenced in Section 4.1 text ("As shown in Figure 1") before it appears. First reference precedes figure placement.
- **Figure 2** (hedging_decomposition.pdf): Referenced in Section 4.2 text ("Figure 2 shows"). First reference precedes figure placement.
- **Figure 3** (l0_phase_transition.pdf): Referenced in Section 5.1 text ("As shown in Figure 3"). First reference precedes figure placement.
- **Figure 4** (cmi_vs_absorption.pdf): Referenced in Section 6.2 text ("As shown in Figure 4"). First reference precedes figure placement.
- **Figure 5** (cmi_dimension_sensitivity.pdf): Referenced in Section 6.5 text ("Figure 5 and the table below show") before it appears. Cross-referenced from Section 7.3 and 7.5.
- **Table 1:** Referenced in Section 3.1 ("Table 1 summarizes") before it appears.
- **Table 2:** Referenced in Section 4.1 ("Table 2 quantifies") before it appears.
- No orphan figures (all referenced in text).
- Visual narrative supports text narrative: control failure (Section 4) before decomposition (Section 4.2) before phase transition (Section 5) before theoretical predictor (Section 6) before dimension caveat (Section 6.5).

## Quality Check

- All captions are self-explanatory: a reader can understand each figure's key message from the caption alone.
- Table 2 bolds the Ratio column values for emphasis and includes "Shuffled/Measured" in the header.
- No redundant figures (each shows a distinct analysis).

## Consistency Issues Found and Fixed (Integration Round)

1. **CMI group mean corrected to source data.** The previous integration used 0.687 for absorbed mean CMI throughout the paper, but source data (`exp/results/full/cmi_estimation.json`, field `mann_whitney_test.absorbed_mean`) records 0.6492. The integrated paper now uses 0.649 consistently (Abstract, Introduction, Section 6.2, Conclusion). Non-absorbed mean (0.861) matches source data (0.8612).

2. **Mann-Whitney statistics corrected to source data.** The previous integration used U = 41.0 and p = 0.042. Source data records U = 28.0 and p = 0.045 (two-sided). The integrated paper now uses U = 28.0 and p = 0.045 throughout.

3. **L0=82 absorption rate discrepancy.** Section 5.1 reports 14.39% from the confound decomposition pipeline; Section 4.3 reports 15.96% from the improved first-letter protocol. Both values are retained with an explicit explanatory note in Section 5.1 clarifying the vocabulary size difference (1,195 vs. 1,203 words) and probe set difference.

4. **Confound decomposition terminology.** The paper uses the correct operational definitions: hedging = resolves at higher L0; hierarchy-driven = persists at all L0 values. (Note: notation.md and glossary.md contained inverted definitions and should be updated separately.)

5. **Random probe range corrected.** Section 4.1 previously stated "9.2--34.3% across other domains"; corrected to "12.9--34.3% on the other four domains" per Table 2 values.

6. **Author name standardized.** "Rajamanoharan et al., 2024" used consistently throughout the paper for the JumpReLU SAE citation.

7. **Control C4 clarified.** Added explanation that the high probe F1 (0.943) on untrained SAE latents reflects random projection preserving gross linear structure.

8. **Dimension sensitivity figure promoted.** Figure 5 (cmi_dimension_sensitivity.pdf) moved from appendix plan to Section 6.5 in the main text, where the dimension analysis is conducted. This improves transparency and allows the discussion (Section 7.3) to reference it naturally.

9. **Conclusion caveats restored.** Added activation patching caveat for 9 persistent words, cross-architecture confound note, Bonferroni-corrected p-value prominence for CMI, and section references for recommendations.

10. **Discussion synthesis paragraph added.** Section 7 now opens with a synthetic paragraph connecting all three findings before discussing each in detail.

11. **Confound decomposition sensitivity limitation added.** Section 7.5 now includes a limitation bullet about the decomposition's dependence on the specific L0 values tested.

## Suggestions for Additional Visuals

The paper is well-balanced between text and visuals with 5 main-body figures. Two optional additions for an appendix:
1. A violin plot comparing JumpReLU vs. L1-ReLU per-letter distributions (Figure A2 from the outline) would strengthen Section 5.3.
2. A per-letter summary table (Table 3 from the outline) would strengthen Section 4.3 by showing probe F1 vs. absorption rate directly.

Neither is required for the main body.
