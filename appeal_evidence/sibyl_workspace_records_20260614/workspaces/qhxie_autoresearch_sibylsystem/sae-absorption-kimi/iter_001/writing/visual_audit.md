# Visual Audit Report

## Completeness Check

**Total figures: 4, Total tables: 3**

All planned figures from the outline are present in the manuscript:
- Figure 1 (Pareto front scatter) --- Section 4.1
- Figure 2 (Partial correlation scatter) --- Section 5
- Figure 3 (Task-agnostic vs. first-letter) --- Section 6
- Figure 4 (Combined regression coefficients / scatter) --- Section 7.1

All planned tables are present:
- Table 1 (Architecture family comparison) --- Section 4.1
- Table 2 (OLS regression coefficients) --- Section 5
- Table 3 (Per-checkpoint task-agnostic vs. first-letter) --- Section 6

**Missing visuals:**
- The Method section references a pipeline overview figure (`fig_method_pipeline.pdf`) in the source section file, but this PDF does not exist in `writing/figures/`. The figure description file (`fig_method_pipeline_desc.md`) exists, but no generated figure accompanies it. This reference was removed during integration to avoid citing a non-existent artifact.

## Consistency Check

**Figure numbering:** Figures are numbered sequentially 1--4 across the manuscript. No duplicate or skipped numbers.

**Table numbering:** Tables are numbered sequentially 1--3 across the manuscript.

**Color scheme:** All generated figures use the shared `style_config.py` palette, ensuring uniform colors per architecture family.

**Font sizes and formatting:** Consistent via `style_config.py` (10 pt base, 11 pt labels, 12 pt titles, 300 dpi export).

**Caption style:** All captions use sentence case and end with a period. Captions are self-contained and explain what the reader should see.

## Flow Check

- Figure 1 is referenced before it appears (Section 4.1, paragraph 4).
- Figure 2 is referenced before it appears (Section 5, paragraph 3).
- Figure 3 is referenced before it appears (Section 6, paragraph 3).
- Figure 4 is referenced before it appears (Section 7.1, paragraph 2).
- No orphan figures.
- Figures appear close to their first reference.
- The visual narrative supports the text narrative: the Pareto scatter supports the tradeoff claim, the partial-correlation scatter supports the downstream-cost claim, and the task-agnostic scatter supports the benchmark-generalization claim.

## Quality Check

- Each caption is self-explanatory and includes the key statistic or takeaway.
- Tables have clear headers and proper alignment.
- No redundant figures (each figure communicates a distinct finding).

## Suggestions

The manuscript is well-supported visually. The only gap is the missing method pipeline figure. If space permits, generating `fig_method_pipeline.pdf` from `fig_method_pipeline_desc.md` and inserting it in Section 3 would improve readability, but it is not a pre-submission blocker given that the methodology is already described in detail.
