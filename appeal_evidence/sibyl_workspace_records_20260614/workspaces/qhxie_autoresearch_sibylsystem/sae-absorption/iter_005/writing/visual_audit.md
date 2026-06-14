# Visual Audit Report

## Summary

- **Total figures in manuscript**: 4 (Figures 1, 2, 3, 5)
- **Total tables in manuscript**: 7 (Tables 1--7)
- **Missing visuals from outline plan**: 4 (Figures 4, 6, 7, 8)
- **Consistency issues found and fixed**: 5

---

## Completeness Check

### Figures Present

| Figure | File | Section | Status |
|--------|------|---------|--------|
| Figure 1 | fig1_partial_correlations.pdf | 4.1 | Present, referenced before appearance |
| Figure 2 | fig2_mediation_path.pdf | 4.1 | Present, referenced in intro and Section 3 before appearance |
| Figure 3 | fig3_crossdomain_absorption.pdf | 4.2 | Present, referenced in intro before appearance |
| Figure 5 | fig5_scaling_surface.pdf | 4.3 | Present, referenced in intro before appearance |

### Figures Missing

| Figure | Planned File | Section | Status | Impact |
|--------|-------------|---------|--------|--------|
| Figure 4 | fig7_crossdomain_by_layer.png | 4.2 | Generated (PNG only, no PDF) | Medium -- layer-wise breakdown supports the text but Table 5 covers the same data |
| Figure 6 | P3_gradient_surface.png | 4.3 | Generated (PNG only, no PDF) | High -- the phase boundary detection paragraph describes gradient ridges purely in text; this figure would strengthen the argument |
| Figure 7 | fig6_taxonomy_correction.png | 4.4 | Generated (PNG only, no PDF) | Medium -- the taxonomy correction results are described in text, but a visual comparison of original vs. corrected rates would be useful |
| Figure 8 | fig5_rosenbaum_sensitivity.png | 4.1 | Generated (PNG only, no PDF) | Medium -- Rosenbaum Gamma values are in Table 4, but the visual would highlight the robustness result |

**Note**: All four missing figures have generated PNG files but lack publication-quality PDF versions. Figure numbering in the generated filenames is inconsistent with the paper numbering (e.g., the crossdomain-by-layer figure is saved as `fig7_crossdomain_by_layer.png` but was planned as Figure 4 in the outline).

### Tables Present

All 7 tables are present in the manuscript:
- Table 1: Partial correlations (Section 4.1) -- present
- Table 2: Width-stratified correlations (Section 4.1) -- present
- Table 3: Mediation results (Section 4.1) -- present
- Table 4: Rosenbaum sensitivity (Section 4.1) -- present
- Table 5: Cross-domain absorption rates (Section 4.2) -- present
- Table 6: Scaling surface model comparison (Section 4.3) -- present
- Table 7: Bradford Hill criteria (Section 5.1) -- present

---

## Consistency Issues Found and Fixed

1. **Table numbering collision resolved.** The original Discussion and Experiments sections both used "Table 6" for different tables (Scaling Surface Model Comparison and Bradford Hill Criteria Assessment). The integrated paper renumbers the Bradford Hill table as Table 7.

2. **Figure numbering gap acknowledged.** Figures are numbered 1, 2, 3, 5 with no Figure 4. This gap arises because the outline planned Figure 4 (crossdomain by layer) and Figure 6 (gradient surface) were not included due to missing PDF-quality versions. If these figures are added later, the numbering will be sequential (1--8).

3. **Cross-section data inconsistency resolved (taxonomy correction).** The original sections contained contradictory claims: the Experiments body text reported 19 letters changed classification (corrected rate 19.2%), while the Conclusion and outline summary claimed 0 letters changed (rate 92.3% unchanged). The integrated paper follows the Experiments body text as authoritative, reporting the 19.2% corrected rate consistently in Sections 4.4, 5.4, and 6.

4. **Cross-domain absorption rate range corrected.** The original intro stated "51--85% absorption rates." The actual data in Table 5 shows 11.3--96.2%. The integrated paper reports the full range.

5. **54 vs. 48 SAE count clarified.** Added parenthetical "(the 54 in the prior study minus 6 lacking reported $L_0$)" in the intro and "Six SAEs from the original 54 that lack reported $L_0$ values are excluded" in the Method and Experiments, ensuring the reader understands the relationship.

---

## Flow Check

- All 4 included figures are referenced in running text before they appear.
- Figure 1 is first referenced in the intro (Contribution 1) and appears in Section 4.1.
- Figure 2 is first referenced in the intro (Contribution 1) and Section 3 opening, appears in Section 4.1.
- Figure 3 is first referenced in the intro (Contribution 2), appears in Section 4.2.
- Figure 5 is first referenced in the intro (Contribution 3), appears in Section 4.3.
- No orphan figures (all included figures are referenced).
- Table 7 (Bradford Hill) is referenced in the Discussion before it appears. All other tables are referenced in their respective sections.

---

## Quality Check

- All figure captions are self-explanatory with key takeaway noted.
- Tables have clear headers.
- Table 3 includes a footnote ($^\dagger$) explaining the indirect-only mediation classification.
- Table 5 includes an explanatory note about the $R_{\text{abs}}$ = FN Rate equivalence at $\tau_{\text{dom}} = 1.0$.
- Bold formatting used in Table 1 (SP-F1 result), Table 3 (SCR full mediation), Table 4 (Mahalanobis matching), Table 6 (interaction GAM).

---

## Suggestions for Additional Visuals

1. **High priority**: Generate PDF versions of Figures 4 and 6 (gradient surface). The gradient surface is particularly important -- the phase boundary detection paragraph is the most text-heavy portion of Section 4.3 without visual support.

2. **Medium priority**: Generate PDF version of Figure 8 (Rosenbaum sensitivity bars). Currently Table 4 carries this information, but a visual would make the Gamma = 2.65 result more memorable.

3. **Optional**: A summary contribution table in the Conclusion (mapping Contribution -> Key Finding -> Practical Implication) would improve scannability.

4. **Optional**: A pipeline overview figure at the top of Section 3 showing the four analysis phases, their inputs, methods, and outputs would help readers navigate the method section.
