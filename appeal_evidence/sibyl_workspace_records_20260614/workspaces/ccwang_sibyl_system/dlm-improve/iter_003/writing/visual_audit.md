# Visual Audit Report

- Total planned figures in manuscript flow: 3
- Total planned tables in manuscript flow: 3
- Rendered figure files present: 3 (`audit_template.pdf`, `repair_harm.pdf`, `mbpp_harm_profile.pdf`)
- Diagram-spec figure files present: 1 (`audit_template_desc.md`)

## Completeness

- Figure 1, Figure 2, and Figure 3 are present as rendered PDF artifacts and are referenced before their placements.
- Table 1, Table 2, and Table A1 are present inline in `writing/paper.md`.

## Consistency checks

- Figure numbering is consistent across the integrated manuscript.
- Table numbering is consistent across the integrated manuscript.
- The shared style file `writing/figures/style_config.py` exists and is referenced by both rendered plotting scripts.
- Caption style was normalized to sentence case with terminal periods in the integrated manuscript.

## Issues found and fixed

- Added a rendered Figure 1 PDF so the integrated manuscript no longer depends on a TODO placeholder.
- Consolidated all figures and tables into a single end-of-manuscript list.
- Ensured each rendered figure is referenced in the text before the placeholder placement line.

## Suggestions

- Replace the draft references block with proper citation keys and bibliography entries during LaTeX finalization.
- If space allows, consider converting Table A1 into a more compact appendix table in LaTeX to reduce prose density in the integrated manuscript.
