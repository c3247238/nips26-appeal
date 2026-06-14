# Visual Audit Report

## Summary

- **Total figures: 7** (Figures 1-7)
- **Total tables: 4** (Tables 1-4)
- **Compiled PDFs: 7/7** (all figures compiled)
- **Figure source files**: All figures have source files (TikZ .tex or Python .py scripts)

---

## Completeness Check

### Figures

| Figure | Source File | Compiled PDF | Status | First Referenced |
|--------|-------------|--------------|--------|------------------|
| Figure 1 | fig1_pipeline_desc.md, fig1_pipeline.tex | fig1_pipeline.pdf | Compiled | Section 3.1 |
| Figure 2 | fig2_metric_comparison_desc.md, fig2_metric_comparison.tex | fig2_metric_comparison.pdf | Compiled | Not referenced in text (optional) |
| Figure 3 | gen_fig3_auroc.py | fig3_auroc.pdf | Compiled | Section 4.1 |
| Figure 4 | gen_fig4_aj_scatter.py | fig4_aj_scatter.pdf | Compiled | Section 4.2 |
| Figure 5 | gen_fig5_cross_arch.py | fig5_cross_arch.pdf | Compiled | Section 4.3 |
| Figure 6 | gen_fig6_layer_corr.py | fig6_layer_corr.pdf | Compiled | Section 4.3 |
| Figure 7 | gen_fig7_dec_norm.py | fig7_dec_norm.pdf | Compiled | Section 4.4 |

All 7 figures are now compiled to PDF and available in `writing/figures/`.

### Tables

| Table | Location | Status |
|-------|----------|--------|
| Table 1 | Section 4.1 | Present --- Probe AUROC and absorption metrics per GemmaScope layer |
| Table 2 | Section 4.2 | Present --- Layer statistics and A_j correlations per GPT-2 layer |
| Table 3 | Section 4.3 | Present --- Cross-architecture statistical comparison |
| Table 4 | Section 4.4 | Present --- Pairwise correlation comparisons |

All four planned tables are present and match the outline.

---

## Consistency Check

### Figure Numbering
- Figures are numbered sequentially: 1, 3, 4, 5, 6, 7.
- Figure 2 is present as a compiled PDF but not referenced in the paper text (it was an optional figure from the outline).
- All figure references in the text match the actual figure numbers.

### Table Numbering
- Tables are numbered sequentially: 1, 2, 3, 4.
- All table references in the text match the actual table numbers.

### Caption Style
- All captions use sentence case.
- All captions end with a period.
- All captions include the key statistic or takeaway.
- Captions are self-explanatory (a reader can understand the figure without reading the text).

### Terminology Consistency
- "ablation-based" and "projection-based" are hyphenated consistently throughout.
- "training-free" is hyphenated consistently.
- "cross-architecture" is hyphenated consistently.
- "layer-dependent" is hyphenated consistently.
- "mid-layer" is hyphenated consistently.
- Notation matches notation.md: $A_{\text{abl}}$, $A_{\text{proj}}$, $A_j$, $j^*$, $\rho$, etc.

---

## Flow Check

### Forward References
- [x] Figure 1 is referenced in Section 3.1 before any other figure.
- [x] Figure 3 is referenced in Section 4.1.
- [x] Figure 4 is referenced in Section 4.2.
- [x] Figure 5 is referenced in Section 4.3.
- [x] Figure 6 is referenced in Section 4.3 (before Figure 7).
- [x] Figure 7 is referenced in Section 4.4.
- [x] All tables are referenced in the text before they appear.

### Orphan Check
- [x] No orphan figures (all referenced figures exist as PDFs).
- [x] No orphan tables (all tables are referenced in the text).

### Visual Narrative
- [x] Figure 1 (pipeline) appears before detailed method description.
- [x] Figure 3 (AUROC) appears alongside probe quality results.
- [x] Figure 4 (scatter) appears alongside A_j correlation discussion.
- [x] Figure 5 (cross-arch) appears alongside cross-architecture comparison.
- [x] Figure 6 (decoder norms) appears alongside decoder norm discussion.
- [x] Figure 7 (layer correlation) appears alongside layer-dependent pattern analysis.

---

## Quality Check

### Caption Quality
- All captions state what the figure shows, the data source, and the key takeaway.
- Figure 4 caption includes the correlation coefficient and p-value.
- Figure 5 caption includes significance markers.
- Figure 7 caption explains the color coding (orange = significant, gray = non-significant).

### Table Quality
- All tables have clear headers with proper alignment.
- Best results are bolded where applicable (Table 2: layer 8 correlation).
- Table 3 includes a footnote explaining the misleading percentage difference.
- Table 2 includes 95% confidence intervals for correlations.

### Redundancy Check
- [x] No redundant figures (each figure shows a distinct aspect of the results).
- [x] No redundant tables (each table serves a distinct purpose).

---

## Revisions Applied (Editor Round 1)

The following fixes were applied to `paper.md` based on the review (`review.md`):

1. **Ablation rate inconsistency fixed** (Section 5.4, Conclusion 6.1): Changed "30.0% (GemmaScope)" to "0.0% on GemmaScope E3v2, 33.3% on GPT-2 E7" to match experimental data. Clarified that the 60-70 point gap between ablation and projection rates---not the absolute ablation rate---is the evidence for functional insensitivity.

2. **"Training-invariant" overclaim fixed** (Contribution 4, Conclusion 6.1): Weakened to "Evidence that decoder norm constraints persist across the two architectures tested" with explicit caveat that architectural effects cannot be ruled out with only two SAE families.

3. **Discussion redundancy reduced** (Sections 5.1-5.3): Replaced number restatement with mechanistic interpretation. Section 5.1 now leads with "Why do JumpReLU and ReLU SAEs... produce projection absorption rates that differ by only 7.7 percentage points?" Section 5.2 leads with the implication for $A_j$ interpretation. Section 5.3 leads with practical guidance for practitioners.

4. **H2 failure framing improved** (Conclusion 6.1): Added clear statement: "H2 failed as originally stated. The hypothesis... was falsified: mean rho = -0.194 across layers, far below the 0.6 threshold." The layer 8 finding is now explicitly framed as exploratory requiring validation.

5. **Banned patterns**: Verified no "Moreover," "Furthermore," or "It is worth noting that" survive in the text.

## Revisions Applied (Editor Round 2)

The following fixes were applied based on section critiques:

1. **Pilot study citation added** (Section 1.2, Gap 1): Added parenthetical citation to the pilot study, enabling reviewer verification.

2. **A_j formula corrected** (Section 3.3.3): Fixed from $W_{\text{enc}, j}$ (j-th column, dimensionally inconsistent) to $e_j$ (j-th encoder row). Updated notation.md and glossary.md to match.

3. **Contribution 3 reframed** (Section 1.3): Changed from "Discovery of layer-dependent A_j correlation pattern" to "Unexpected layer-dependent A_j correlation pattern" with explicit H2 failure context.

4. **Contribution 4 weakened** (Section 1.3): Removed "training-invariant" claim; now states "Evidence that decoder norm constraints persist across the two architectures tested."

5. **Gap 2 narrowed** (Section 1.2): Changed from "other architectures" (broad) to "Cross-architecture validation between JumpReLU and standard ReLU SAEs."

6. **Cohen's d characterization fixed** (Sections 1.3, 3.3.2, 5.1): Now correctly states d = 1.82 is a large effect size by conventional thresholds, while the absolute difference (7.7%) is modest.

7. **Ablation insensitivity clarified** (Section 5.4): Now explicitly states the 60-70 point gap between ablation (0-33%) and projection (91-98%) rates is the evidence for functional insensitivity.

8. **H2 failure reframed as discovery** (Section 5.2): Added paragraph treating H2 failure as a meaningful scientific outcome that revealed layer depth as the primary moderator.

9. **Section 5.6 expanded** (Discussion): Expanded from 2 paragraphs to 6 with four concrete benchmark recommendations.

10. **Reproducibility details added** (Section 3.5): Added solver='liblinear', stratified split, AUROC on test set, tie-breaking rule, and supplementary materials reference.

11. **Table 3 footnote clarified** (Section 4.3): Rewrote footnote to explicitly warn about the misleading percentage difference.

12. **Architectural solutions connection added** (Section 5.1): Added paragraph connecting findings to OrtSAE, MP-SAE, and WSAE literature.

---

## File Inventory

### Figure Files (in `writing/figures/`)

| File | Type | Description |
|------|------|-------------|
| fig1_pipeline.pdf | TikZ compiled | Pipeline architecture diagram |
| fig1_pipeline_desc.md | Markdown | TikZ description and metadata |
| fig2_metric_comparison.pdf | TikZ compiled | Ablation vs projection bar chart |
| fig2_metric_comparison_desc.md | Markdown | TikZ description and metadata |
| fig3_auroc.pdf | Python/matplotlib | AUROC distribution boxplot |
| gen_fig3_auroc.py | Python script | Generation script |
| fig4_aj_scatter.pdf | Python/matplotlib | A_j vs projection scatter |
| gen_fig4_aj_scatter.py | Python script | Generation script |
| fig5_cross_arch.pdf | Python/matplotlib | Cross-architecture comparison |
| gen_fig5_cross_arch.py | Python script | Generation script |
| fig6_layer_corr.pdf | Python/matplotlib | Layer-dependent correlation |
| gen_fig6_layer_corr.py | Python script | Generation script |
| fig7_dec_norm.pdf | Python/matplotlib | Decoder norm statistics |
| gen_fig7_dec_norm.py | Python script | Generation script |
