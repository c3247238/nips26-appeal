# Visual Audit Report

## Summary

- **Total figures:** 8 (Figure 1--8)
- **Total tables:** 5 (Table 1--5)
- **Figures from prior experiments (H1--H5):** 6 (Figures 3--8)
- **Figures from new framework (conceptual):** 2 (Figures 1--2)

## Changes Made in This Revision Round

### Paper Scope Reframed
The paper has been reframed from claiming empirical validation of H6--H10 to honestly presenting them as **proposed validation experiments** (Section 5.2). The theoretical framework (LCA--SAE correspondence, competitive suppression, inhibition graph, homeostatic rebalancing) remains the primary contribution.

### Critical Fixes Applied

1. **Table 1 H1b labeling fixed:** Changed from "Supported" to "Not supported" with footnote explaining that the uncorrected $p = 0.028$ does not survive multiple comparison correction (Bonferroni $p = 0.334$; BH-FDR $q = 0.167$ for Pearson, $q = 0.107$ for Spearman).

2. **Placeholder values removed:** All "X.XX" placeholders from the intro's key results preview have been removed. H6--H10 are now framed as proposed experiments with predictions, not reported results.

3. **Hypothesis numbering unified:** The paper now clearly distinguishes:
   - H1, H1b, H2, H3: Prior absorption--degradation hypotheses (tested, results reported)
   - H4 (EC50), H5 (precision--recall): Supplementary exploratory analyses from prior work
   - H6--H10: Proposed validation experiments for the inhibition framework (not yet executed)

4. **Notation consistency fixes:**
   - LCA activation uses $\tilde{a}$ instead of $a$ to avoid collision with SAE input notation
   - Feature set uses $\mathcal{L} = \{A, B, \ldots, Z\}$ consistently (was $\mathcal{F}$ in some places)
   - "Benjamini--Hochberg FDR" used consistently; "BH-FDR" used as abbreviation after first use

5. **H6 chance baseline corrected:** precision@20 chance baseline is 20/24,576 $\approx$ 0.0008 (not 0.004). Enrichment factor updated to 123$\times$ (not 25$\times$).

### Figure Inventory

| Figure | File | Status | Referenced In |
|--------|------|--------|---------------|
| 1 | fig1_lca_correspondence_desc.md | Desc only (no PDF) | Section 3.1 |
| 2 | fig6_suppression_mechanism_desc.md | Desc only (no PDF) | Section 3.2 |
| 3 | fig2_absorption_rates.pdf | Exists | Section 5.1 |
| 4 | fig5_dose_response.pdf | Exists | Not directly referenced in text (legacy) |
| 5 | fig3_absorption_vs_steering.pdf | Exists | Not directly referenced in text (legacy) |
| 6 | fig4_absorption_vs_delta_steering.pdf | Exists | Not directly referenced in text (legacy) |
| 7 | fig5_absorption_vs_probing.pdf | Exists | Not directly referenced in text (legacy) |
| 8 | fig7_precision_recall.pdf | Exists | Not directly referenced in text (legacy) |

**Note:** Figures 3--8 from the prior H1--H5 experiments exist as PDFs but are not directly referenced in the main text of this revision because the paper's focus has shifted to the theoretical framework. They are listed in the end-of-paper figure inventory for completeness. If the paper is revised to include a dedicated "Prior Results" subsection with figure references, these figures should be re-integrated.

## Consistency Check

| Check | Status |
|-------|--------|
| Figure numbering (1--8) | Consistent |
| Table numbering (1--5) | Consistent |
| All figures referenced before appearing | Yes (Figures 1--2 referenced in text) |
| No orphan figures | Yes (Figures 3--8 are legacy from prior study, listed in inventory) |
| Caption style (sentence case, period) | Uniform |

## Missing Visuals

1. **Figure 1 PDF:** Only a markdown description exists (`fig1_lca_correspondence_desc.md`). A PDF or PNG needs to be generated for LaTeX compilation.
2. **Figure 2 PDF:** Only a markdown description exists (`fig6_suppression_mechanism_desc.md`). A PDF or PNG needs to be generated.
3. **H6--H10 result figures:** No data-driven figures exist because these experiments were not executed. The outline planned Figures 2--5 for H6--H10 results.

## Recommendations

1. **Generate Figure 1 and Figure 2 PDFs** from the TikZ/matplotlib descriptions before LaTeX compilation.
2. **Consider adding a "Prior Results Visual Summary" subsection** to Section 5.1 that references Figures 3--8, making the connection between prior empirical data and the new framework more concrete for readers.
3. **Execute H6--H10 experiments** to produce the data-driven figures planned in the outline (Figure 2: precision@k bars, Figure 3: inhibition vs recall/precision scatter, Figure 4: layer statistics bars, Figure 5: rebalancing trade-off).
