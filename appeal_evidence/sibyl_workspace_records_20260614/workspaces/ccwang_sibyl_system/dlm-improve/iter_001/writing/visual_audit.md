# Visual Audit

- Total figures: 5
- Total tables: 4

## Present Visuals

- Figure 1: `fig1_teaser.pdf`
- Figure 2: `fig_protocol_flow_desc.md`
- Figure 3: `fig2_honest_compute.pdf`
- Figure 4: `fig3_signal_gap.pdf`
- Figure 5: `fig4_code_boundary.pdf`
- Table 1: inline in `paper.md`
- Table 2: inline in `paper.md`
- Table 3: inline in `paper.md`
- Table 4: inline in `paper.md`

## Completeness Check

- The outline's teaser figure is present.
- The honest-compute comparison figure is present.
- The observer-controller audit figure is present.
- The code-boundary figure is present.
- The method/protocol flow is present as a detailed description file, but not yet rendered as a PDF.

## Missing Visuals

- `[TODO: Figure 2]` should be rendered from `fig_protocol_flow_desc.md` during the LaTeX stage.

## Consistency Notes

- Numbering was unified in the integrated manuscript as Figures 1-5 and Tables 1-4.
- All rendered figures use the shared `style_config.py`.
- Figure references appear in the text before the corresponding captions.
- The manuscript now follows a result-first visual narrative:
  - teaser in the Introduction;
  - protocol placeholder before detailed results;
  - honest-compute figure with GSM8K discussion;
  - signal-audit figure with observer/controller discussion;
  - code-boundary figure with HumanEval discussion.

## Suggestions

- If the paper feels text-heavy after LaTeX integration, the first additional visual should be a compact benefit-bucket chart once that analysis exists.
- The appendix should eventually include a runtime fairness table derived from the same result JSONs used by the main figures.
