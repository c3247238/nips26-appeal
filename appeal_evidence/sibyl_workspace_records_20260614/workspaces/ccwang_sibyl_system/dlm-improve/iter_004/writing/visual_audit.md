# Visual Audit

- Total figures: 4
- Total tables: 2

## Present visuals

- Figure 1: `fig1_entropy_routed_compute_desc.md`
- Figure 2: `gen_fig2_quality_speed.py`, `fig2_quality_speed.pdf`
- Figure 3: `gen_fig3_stopping_hist.py`, `fig3_stopping_hist.pdf`
- Figure 4: `gen_fig4_runtime_lineage.py`, `fig4_runtime_lineage.pdf`
- Table 1: inline main-results table
- Table 2: inline paired repair/harm table

## Completeness Check

- All required minimum visual categories are present:
  - 1 method/architecture diagram placeholder or description
  - 1 main results table
  - 1 analysis figure
- No planned result figure from the outline is missing.

## Consistency Check

- Figure numbering is consistent from Figure 1 to Figure 4.
- Table numbering is consistent from Table 1 to Table 2.
- The rendered charts share a common style configuration through `style_config.py`.
- Captions are sentence-case and framed as self-contained summaries.

## Flow Check

- Figure 1 is referenced in Method before its caption.
- Table 1 and Figure 2 are referenced in Experiments before their captions.
- Figure 3 and Figure 4 are referenced in the relevant analysis/discussion flow.

## Remaining Issues

- Figure 1 is still a description file rather than a rendered diagram. This is acceptable for the current markdown draft, but a camera-ready pipeline should convert it into TikZ or an equivalent rendered figure.
- The paper still needs explicit citation anchors and a bibliography pass before it can be considered submission-ready.

## Suggestions

- Add one optional teaser visual only if a later revision identifies a truly compact summary view; do not add a teaser figure just to increase visual count.
- If the paper grows longer, consider moving the runtime-lineage figure to an appendix and leaving a compact summary table in the main text.
