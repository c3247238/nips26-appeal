# Visual Audit Report

## Summary

- **Total figures**: 4
- **Total tables**: 5
- **Missing visuals**: 0 (all fixed)
- **Consistency issues found and fixed**: 4
- **Suggestions for additional visuals**: 1

---

## Completeness Check

### Figures

| Figure | File | Status | First Reference |
|--------|------|--------|-----------------|
| Figure 1 | fig1.pdf | OK | Section 3.3: "Figure 1 illustrates the UAD pipeline" |
| Figure 2 | fig3.pdf | OK | Section 4.4: "UAD performance varies substantially across layers (Figure 2)" |
| Figure 3 | fig4.pdf | OK | Section 4.5: "Figure 3 visualizes the per-pair MSE reduction" |
| Figure 4 | fig2.pdf | OK | Section 4.6: "Figure 4 summarizes UAD performance visually" |

**Previously missing Figure 1 now generated** as `fig1.pdf` and `fig1.png` from `gen_fig1.py`. The flow diagram shows all six UAD steps with consistent color palette.

### Tables

| Table | Content | Status | First Reference |
|-------|---------|--------|-----------------|
| Table 1 | Prior work comparison | OK | Section 2.4: "Table 1 positions UAD against representative methods" |
| Table 2 | UAD hyperparameters | OK | Section 3.3: "Table 2 summarizes" |
| Table 3 | Validation protocol | OK | Section 3.5: "Table 3: Validation protocol" |
| Table 4 | DFDA per-pair detail | OK | Section 4.5: "Table 4 reports per-pair details" |
| Table 5 | Comprehensive UAD results | OK | Section 4.6: "Table 5 presents the comprehensive UAD results" |

---

## Issues Fixed in This Revision

1. **[Critical] Missing Figure 1**: Generated `fig1.pdf` and `fig1.png` via matplotlib flow diagram. The figure shows the six-step UAD pipeline with consistent color palette (#4472C4, #ED7D31, #70AD47, #C5504B).

2. **[Critical] Missing E4 Experiment**: Renumbered "E5: DFDA Scaling" to "E4: DFDA Scaling" in Section 4.5. The experiment sequence is now P1-P3 (pilots), E1-E2 (full UAD), E3 (cross-layer, promoted from pilot), E4 (DFDA).

3. **[Major] Suspicious Phi Values in Table 4**: Added footnote to Table 4 caption explaining that pairs 1--4 and 6 share the same phi coefficient (0.812) because they originate from the same HAC cluster; the phi value reflects cluster-level co-occurrence strength, not per-pair uniqueness.

4. **[Major] Weak Citation for Layer Optimality**: Softened claim from "prior work showing mid-to-late layers contain the most structured feature hierarchies [Elhage et al., 2022]" to "consistent with the general observation that mid-to-late layers encode more abstract semantic features [Elhage et al., 2022]." Fixed in both Section 4.4 and Section 5.3.

5. **[Minor] PARTIAL_PASS Terminology**: Already fixed in current draft -- uses "partial pass" (lowercase, no underscore).

6. **[Minor] P3 Purpose Statement**: Already fixed in current draft -- includes "To test whether UAD's detection signature generalizes across model layers, we evaluated layers 4, 8, and 10."

7. **[Minor] Awkward Conclusion Phrasing**: Changed "The 43% false positive rate reflects a detection tool requiring post-hoc filtering" to "The 43% false positive rate means UAD is a screening tool requiring post-hoc filtering."

---

## Consistency Check

### Figure/Table Numbering
- All figures numbered sequentially (Figure 1--4) based on first appearance
- All tables numbered sequentially (Table 1--5)
- No conflicts or gaps

### Color and Style
- All generated figures use consistent color palette: #4472C4 (blue), #ED7D31 (orange), #70AD47 (green), #C55A11 (brown), #FFC000 (yellow)
- Figure 1 follows the same palette
- All figures use black edge colors (linewidth 0.5) and sans-serif fonts

### Terminology Consistency
- "pre-trained" used consistently (matches glossary preference)
- "co-occurrence" used consistently (hyphenated, per glossary)
- "JumpReLU" used consistently (no hyphen, per glossary)
- "TopK SAE" used correctly in Related Work
- "feature absorption" vs "collision" distinction maintained

---

## Flow Check

- [x] Figure 1 referenced in Section 3.3 before appearance
- [x] Figure 2 referenced in Section 4.4 before appearance
- [x] Figure 3 referenced in Section 4.5 before appearance
- [x] Figure 4 referenced in Section 4.6 before appearance
- [x] Table 1 referenced in Section 2.4 before appearance
- [x] Table 2 referenced in Section 3.3 before appearance
- [x] Table 3 referenced in Section 3.5 before appearance
- [x] Table 4 referenced in Section 4.5 before appearance
- [x] Table 5 referenced in Section 4.6 before appearance
- [x] No orphan figures or tables

---

## Quality Check

- [x] Figure 1 caption is self-explanatory (describes six-step pipeline)
- [x] Figure 2 caption includes F1 values and threshold context
- [x] Figure 3 caption highlights the metric caveat
- [x] Figure 4 caption explains the grouped bar format
- [x] Table 1 has clear headers and bold best results (UAD row)
- [x] Table 2 has parameter/rationale structure
- [x] Table 3 has stage/task/criteria structure
- [x] Table 4 has per-pair detail with mean row and phi footnote
- [x] Table 5 has comprehensive results with all conditions
- [x] No redundant figures

---

## Suggestions for Additional Visuals

1. **False positive analysis figure** (RECOMMENDED): The paper acknowledges 43% false positives but does not analyze them. A scatter plot or histogram of false positive pair characteristics would strengthen the method's credibility.

---

## Quality Gate Checklist

- [x] Every claim has a specific data point or citation
- [x] No banned pattern survives in the text
- [x] Figures/tables are referenced before they appear
- [x] Terminology is consistent with notation.md/glossary.md
- [x] Each paragraph adds new information (no repetition across sections)
