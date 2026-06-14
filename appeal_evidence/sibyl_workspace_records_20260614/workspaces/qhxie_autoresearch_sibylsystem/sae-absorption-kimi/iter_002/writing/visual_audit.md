# Visual Audit Report

## Audit Date
2026-04-23

## Summary
- **Total figures:** 5
- **Total tables:** 6
- **Missing visuals:** 0 (all planned figures and tables are present)
- **Consistency issues found:** 2 (minor, fixed during integration)
- **Suggestions for additional visuals:** 1

---

## Completeness Check

### Figures
| # | Filename | Section | Status |
|---|----------|---------|--------|
| 1 | fig1_architecture_ranking.png | Results (4.1) | Present |
| 2 | fig2_firstletter_vs_semantic_scatter.png | Results (4.2) | Present |
| 3 | fig3_semantic_vs_nonhierarchy.png | Results (4.3) | Present |
| 4 | fig4_tau_fs_robustness.png | Results (4.4) | Present |
| 5 | fig5_gpt2_replication.png | Results (4.6) | Present |

All 5 planned figures from the outline are present in the manuscript.

### Tables
| # | Description | Section | Status |
|---|-------------|---------|--------|
| 1 | SAE architecture selection | Method (3.1) | Present |
| 2 | WordNet semantic hierarchies | Method (3.3) | Present |
| 3 | Non-hierarchy control pairs | Method (3.5) | Present |
| 4 | Per-architecture absorption scores | Results (4.1) | Present |
| 5 | tau_fs robustness | Results (4.4) | Present |
| 6 | GPT-2 replication | Results (4.6) | Present |

All 6 planned tables are present. Note: The outline originally planned Tables 4 and 5 for the Appendix, but they are more effectively placed inline in the Results section where they are discussed.

---

## Consistency Check

### Figure Numbering
- All figures use consistent sequential numbering: Figure 1, 2, 3, 4, 5.
- No duplicate or skipped numbers.

### Table Numbering
- All tables use consistent sequential numbering: Table 1, 2, 3, 4, 5, 6.
- Note: During integration, table numbers were reassigned to reflect their actual appearance order in the manuscript (Method tables precede Results tables).

### Color Scheme
- Figure 1: Blue (first-letter) and orange (semantic-hierarchy) --- consistent with standard contrast palette.
- Figure 2: Scatter with regression line --- uses default matplotlib colors, consistent with style_config.py.
- Figure 3: Coral (semantic-hierarchy) and green (non-hierarchy) --- distinct from Figure 1's palette to avoid confusion.
- Figure 4: Line plot with error bars --- single series, no color conflict.
- Figure 5: Grouped bar chart --- reuses coral/green from Figure 3 for consistency.

### Caption Style
- All captions use sentence case.
- All captions end with a period.
- All captions are self-explanatory (a reader can understand the key takeaway without reading the text).
- Figure 4 caption includes a parenthetical clarifying the H1 threshold: "(the pre-registered threshold for supporting construct validity)" --- added during integration to address critique feedback.

### Issues Found and Fixed
1. **Table 1 bolding inconsistency (critique):** The original section had both best (lowest) and worst (highest) values bolded, contradicting the caption. Fixed: only lowest (best) values are bolded per column.
2. **Figure 2 caption clarity (critique):** Added clarification that $n = 7$ refers to trained SAEs (excluding Random).

---

## Flow Check

### Text References Before Appearance
- [x] Figure 1 is referenced in Section 4.1 before its display.
- [x] Figure 2 is referenced in Section 4.2 before its display.
- [x] Figure 3 is referenced in Section 4.3 before its display.
- [x] Figure 4 is referenced in Section 4.4 before its display.
- [x] Figure 5 is referenced in Section 4.6 before its display.
- [x] Table 4 is referenced in Section 4.1 before its display.
- [x] Table 5 is referenced in Section 4.4 before its display.
- [x] Table 6 is referenced in Section 4.6 before its display.

### Orphan Check
- [x] No orphan figures (all figures are referenced in the text).
- [x] No orphan tables (all tables are referenced in the text).

### Proximity Check
- [x] Figures appear within 1-2 paragraphs of their first reference.
- [x] Tables appear immediately after their first reference.

### Visual Narrative
- Figure 1 (architecture ranking) appears alongside the main results discussion, supporting the ranking-inversion claim.
- Figure 2 (scatter plot) appears alongside the H1 construct-validity discussion, visualizing the wide CI.
- Figure 3 (hierarchy specificity) appears alongside the H2 rejection, making the direction reversal visually apparent.
- Figure 4 (robustness) appears alongside the H3 discussion, showing stability across thresholds.
- Figure 5 (GPT-2 replication) appears alongside the cross-model comparison.

---

## Quality Check

### Caption Self-Explanatoryness
- [x] Figure 1: Yes --- states what is shown, the key comparison, and the main takeaway.
- [x] Figure 2: Yes --- states the correlation value, CI, and interpretation.
- [x] Figure 3: Yes --- states the test statistic and conclusion.
- [x] Figure 4: Yes --- states the stability and the inconclusiveness.
- [x] Figure 5: Yes --- states the model and the near-zero finding.

### Table Quality
- [x] Table 1: Clear headers, proper alignment, bold best results.
- [x] Table 2: Clear headers, documents exact hierarchies for reproducibility.
- [x] Table 3: Clear headers, documents control pairs for reproducibility.
- [x] Table 4: Clear headers, bold best (lowest) per column, comprehensive.
- [x] Table 5: Clear headers, reports full statistical results.
- [x] Table 6: Clear headers, concise replication summary.

### Redundancy Check
- [x] No redundant figures (each figure shows a distinct aspect of the results).
- [x] No redundant tables (each table serves a distinct purpose).

---

## Suggestions for Additional Visuals

The paper is well-supported visually and not text-heavy. One potential addition:

**Suggested: Method pipeline diagram (Figure 6).** The outline's `fig_method_pipeline_desc.md` describes a flowchart showing the three-branch evaluation pipeline (first-letter, semantic-hierarchy, non-hierarchy control). This would be valuable for readers trying to understand the experimental design at a glance, particularly the relationship between the three conditions. The figure description is already drafted in `figures/fig_method_pipeline_desc.md` and could be rendered with TikZ or matplotlib. Placement: Section 3.4 (Absorption Measurement Protocol) or as a full-page figure at the start of the Method section.

Priority: Low. The current manuscript is visually complete and well-balanced.
