# Visual Audit Report

## Summary

- **Total figures**: 6
- **Total tables**: 9
- **Missing visuals**: 0 (all planned figures have corresponding PDFs; no orphaned TODOs)

## Completeness Check

| Planned Figure/Table | Status | Location |
|---------------------|--------|----------|
| Figure 1: Teaser saturation curve | Present | `figures/fig1_teaser.pdf` |
| Figure 2: Theory framework | Present | `figures/fig2_theory_framework.pdf` |
| Figure 3: H1 saturation fit | Present | `figures/fig3_h1_saturation.pdf` |
| Figure 4: H2 Ea by level | Present | `figures/fig4_h2_ea_levels.pdf` |
| Figure 5: H3 routing failure | Present | `figures/fig5_h3_routing_failure.pdf` |
| Figure 6: Routing signal comparison | Present | `figures/hypothesis_summary.pdf` |
| Table 1: Related work comparison | Present | Section 2 |
| Table 2: Hypotheses | Present | Section 3 |
| Table 3: Experiment config | Present | Section 4 |
| Table 4: Accuracy vs. k | Present | Section 5.1 |
| Table 5: Ea by level | Present | Section 5.2 |
| Table 6: Routing performance | Present | Section 5.3 |
| Table 7: Hypothesis summary | Present | Section 5.5 |
| Table 8: Error classification | Present | Section 6.2 |
| Table 9: Routing signal comparison | Present | Section 6.4 |

## Consistency Check

- **Figure numbering**: Sequential 1--6 across sections. No duplicates or gaps.
- **Table numbering**: Sequential 1--9 by appearance order. Renumbered from source files to ensure consistency.
- **Caption style**: Sentence case with period at end. All captions are self-explanatory.
- **Notation**: Consistent use of $E_a$, $k_0$, $P_\infty$ throughout.

## Flow Check

| Figure/Table | First Reference | Section |
|-------------|-----------------|---------|
| Figure 1 | "Figure 1 shows this saturation pattern..." | Introduction |
| Figure 2 | "Figure 2 illustrates the hypothesized pipeline..." | Method |
| Table 1 | "Table 1 summarizes the relationship..." | Related Work |
| Table 2 | "Table 2 reports..." | Method |
| Table 3 | "Table 3 summarizes..." | Experimental Setup |
| Figure 3 | "Figure 3 visualizes the saturation curve..." | Results 5.1 |
| Table 4 | "Table 4 reports accuracy..." | Results 5.1 |
| Figure 4 | "Figure 4 shows the distribution..." | Results 5.2 |
| Table 5 | "Table 5 reports..." | Results 5.2 |
| Figure 5 | "Figure 5 shows the routing failure..." | Results 5.3 |
| Table 6 | "Table 6 reports..." | Results 5.3 |
| Table 7 | "Table 7 summarizes..." | Results 5.5 |
| Table 8 | "Table 8 classifies..." | Discussion 6.2 |
| Table 9 | "Table 9 compares..." | Discussion 6.4 |
| Figure 6 | "Figure 6 visualizes the routing signal landscape..." | Discussion 6.4 |

All figures and tables are referenced before they appear. No orphan figures.

## Quality Check

- **Captions**: Each caption states the key takeaway (e.g., "$R^2 = 0.924$ confirms aggregate Arrhenius kinetics").
- **Tables**: Clear headers, proper alignment, bold best results where applicable.
- **Redundancy**: No redundant figures. Each figure shows distinct information.
- **Self-containment**: Table 7 (hypothesis summary) and Table 9 (routing comparison) can be understood without reading the full text.

## Issues Found and Fixed

1. **Citation inconsistency**: Source files mixed numbered citations [1], [2] with author-year (Yang et al., 2025). Standardized to numbered [1]--[11] throughout.
2. **Table renumbering**: Source files had Table 1 (experiment config), Table 3 (related work), Table 5 (routing signals) with gaps. Renumbered sequentially 1--9 by appearance order.
3. **Section structure mismatch**: Outline had duplicate Related Work sections (2 and 7). Merged into single Section 2; updated intro and conclusion cross-references accordingly.
4. **Data inconsistency**: Two versions of Discussion existed (discussion.md with n=50 data, Discussion.md with older n=30 data). Used consistent n=50 dataset throughout.
5. **Mini-conclusion redundancy**: discussion.md had a Section 6.6 mini-conclusion that repeated conclusion.md content. Removed to avoid near-verbatim repetition.
6. **Missing figure references**: Figure 1 was not referenced in intro.md. Added forward reference in Introduction.

## Suggestions

- The paper has 6 figures and 9 tables for approximately 7 sections. The figure-to-text ratio is healthy.
- Consider adding a supplementary figure showing the per-problem $R^2$ distribution (mentioned in Section 5.1 caveat) if space permits in the appendix.
- The bimodal $E_a$ distribution (Figure 4) is a key visual; ensure it is rendered with sufficient contrast to show the two clusters clearly.
