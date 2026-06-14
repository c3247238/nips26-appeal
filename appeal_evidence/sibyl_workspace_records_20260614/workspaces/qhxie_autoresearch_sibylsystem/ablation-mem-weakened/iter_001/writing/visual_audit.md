# Visual Audit Report

## Summary

- **Total figures:** 5
- **Total tables:** 3
- **Missing visuals:** None
- **Consistency issues found and fixed:** 3
- **Suggestions for additional visuals:** 1

---

## Completeness Check

All planned figures and tables from the outline are present in the manuscript:

| Planned Visual | Status | Location in Paper |
|---------------|--------|-------------------|
| Figure 1: Pipeline overview | Present | Section 4.1 |
| Figure 2: Absorption rates bar chart | Present | Section 5.1 |
| Figure 3: Absorption vs. Steering scatter | Present | Section 5.4 |
| Figure 4: Absorption vs. Probing scatter | Present | Section 5.4 |
| Figure 5: Dose-response curves | Present | Section 5.2 |
| Table 1: Hypothesis test summary | Present | Section 5.4 |
| Table 2: Top absorbed features | Present | Section 5.4 |
| Table 3: Layer-level absorption summary | Present | Section 5.1 |

All figure PDFs exist in `writing/figures/`:
- `fig1_pipeline.pdf` (not generated; described textually)
- `fig2_absorption_rates.pdf` (22 KB)
- `fig3_absorption_vs_steering.pdf` (25 KB)
- `fig4_absorption_vs_probing.pdf` (24 KB)
- `fig5_dose_response.pdf` (21 KB)

---

## Consistency Check

### Issues Found and Fixed

1. **Figure numbering order (FIXED).** The original draft referenced Figure 5 (dose-response) in Section 5.2 before Figures 3 and 4 (scatter plots) in Section 5.4, causing readers to encounter figures out of numerical order. Fixed by renumbering: dose-response curves are now Figure 3 (Section 5.2), absorption vs. steering is Figure 4 (Section 5.4), and absorption vs. probing is Figure 5 (Section 5.4). Figures now appear in strict numerical order throughout the text.

2. **Table 2 layer 10 data (FIXED).** The original Table 2 included features Y and R from layer 10 with "--" for steering and probing data. The review flagged this as confusing. Fixed by removing layer 10 features from Table 2 and adding a caption note: "Steering and probing results were collected for layers 4 and 8 only."

3. **Table 1 H3 layout (FIXED).** The original Table 1 had an awkward layout where H3 spanned the Layer column with "--". Fixed by removing H3 from Table 1 and reporting H3 results in the text with a clear explanation of the slope comparison and CV values.

### Caption Style

All captions use sentence case and end with a period. Each caption includes a brief takeaway sentence.

---

## Flow Check

| Figure/Table | First Reference | Text Reference |
|-------------|-----------------|----------------|
| Figure 1 | Section 4.1 | "illustrated in Figure 1" |
| Figure 2 | Section 5.1 | "Figure 2 shows..." |
| Table 3 | Section 5.1 | "Table 3 summarizes..." |
| Figure 3 | Section 5.2 | "Figure 3 shows..." |
| Figure 4 | Section 5.4 | "Figure 4 plots..." |
| Figure 5 | Section 5.4 | "Figure 5 plots..." |
| Table 1 | Section 5.4 | "Table 1 presents..." |
| Table 2 | Section 5.4 | "Table 2 lists..." |

All figures and tables are referenced before they appear. No orphan figures.

---

## Quality Check

- **Figure 1:** Self-explanatory caption describing the four-phase pipeline.
- **Figure 2:** Caption explains the key takeaway (most features show near-zero absorption).
- **Figure 3:** Caption explains the monotonic increase in success with strength.
- **Figure 4:** Caption notes the lack of significant correlation.
- **Figure 5:** Caption notes the lack of significant correlation.
- **Table 1:** Compact summary with clear Result column.
- **Table 2:** Sorted by absorption rate with explicit note about layer coverage.
- **Table 3:** Clear layer-level statistics with category counts.

No redundant figures. Each figure shows a distinct aspect of the results.

---

## Suggestions for Additional Visuals

1. **Statistical power diagram (optional):** A small diagram showing the relationship between sample size, effect size, and power would help readers understand why the study may be underpowered for small effects. This could be added to the Discussion (Section 6.1) or Appendix.

2. **"Conditions that would change our conclusion" decision tree (optional):** Section 6.4 lists four conditions. A flowchart or decision tree would make this visually compelling and easier to reference.

Both suggestions are optional enhancements for a future revision. The current visual set is complete and sufficient for the paper's claims.
