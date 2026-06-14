# Visual Audit Report -- ComposeAccel (Iteration 2)

## Summary

- **Total figures**: 7 planned (5 present as PDF, 2 missing)
- **Total tables**: 7 (all present as inline markdown)

## Figure Inventory

| Figure | File | Status | Section |
|--------|------|--------|---------|
| Figure 1: Speed-Quality Landscape Teaser | fig_teaser.pdf | PRESENT | 1 (Introduction) |
| Figure 2: ComposeAccel Architecture | fig2_architecture.pdf | MISSING (description exists: fig2_architecture_desc.md) | 3 (Methods) |
| Figure 3: Single-Method Pareto Curves | fig3_single_pareto.pdf | PRESENT | 4.1 (Experiments) |
| Figure 4: Pairwise Orthogonality Bars | fig4_ortho_bars.pdf | PRESENT | 4.2 (Experiments) |
| Figure 5: Combined Pareto Frontier | fig5_combined_pareto.pdf | PRESENT | 4.3 (Experiments) |
| Figure 6: IGSD T_draft Ablation | fig6_tdraft_ablation.pdf | PRESENT | 4.4 (Experiments) |
| Figure 7: Per-Step KL Divergence Profile | [not generated] | MISSING (data exists: igsd_kl_profiles_raw.json) | 4.4 (Experiments) |

## Table Inventory

| Table | Status | Section |
|-------|--------|---------|
| Table 1: Published DLM Acceleration Methods | PRESENT (added during integration) | 2.2 (Related Work) |
| Table 2: Metric Definitions | PRESENT | 3.1 (Methods) |
| Table 3: Single-Method Pareto Results | PRESENT | 4.1 (Experiments) |
| Table 4: Pairwise Orthogonality Matrix | PRESENT | 4.2 (Experiments) |
| Table 5: Three-Way Composition Operating Points | PRESENT | 4.3 (Experiments) |
| Table 6: Cross-Model Comparison | PRESENT | 5.1 (Cross-Model) |
| Table 7: AR vs. DLM Comparison | PRESENT | 5.2 (AR Comparison) |

## Missing Visuals

### Figure 2 (BLOCKING)
- **Status**: Only a description file (`fig2_architecture_desc.md`) exists. No PDF generated.
- **Impact**: The architecture diagram is critical for readers to understand the IGSD pipeline and M1/M3 integration points. A `[TODO]` placeholder is included in the paper.
- **Action required**: Generate from the detailed specification in `fig2_architecture_desc.md` using TikZ, draw.io, or an equivalent tool.

### Figure 7 (HIGH PRIORITY)
- **Status**: Raw data exists (`igsd_kl_profiles_raw.json`), but no generation script or PDF.
- **Impact**: This figure is the primary evidence for refuting the inverted-U KL hypothesis and explaining IGSD's tau insensitivity. The finding is described in text only, which reduces its persuasiveness.
- **Action required**: Generate a line plot with shaded std band. X-axis: denoising step (0--63). Y-axis: mean token-level KL divergence. Add horizontal reference lines at tau = {0.7, 0.85, 0.9}.

### Figure 8 (DEFERRED)
- **Status**: Planned in outline (Batch Size Sensitivity) but not referenced in any section draft.
- **Impact**: The batch size analysis in Section 5.3 is brief (3 sentences). A figure would strengthen it but is not essential for the main argument.
- **Action**: Generate if space permits; defer to appendix otherwise.

## Consistency Issues Found and Fixed

1. **Figure numbering**: Renumbered consistently across all sections. Original section drafts used independent numbering; integrated paper uses sequential Figure 1--7 and Table 1--7.

2. **Figure forward-references**: All present figures are referenced in the text before they appear. Figure 2 is forward-referenced in Section 3.5 (after all methods are described) rather than at the top of Section 3 (before any method content), improving pedagogical flow per critique feedback.

3. **Table caption placement**: All table captions now appear above the table content, consistent with ML venue conventions.

4. **Bold formatting in tables**: Reviewed for consistency. Bold marks the best value per column within each method group (Table 3) or the best pair (Table 4).

## Numerical Consistency Fixes Applied

1. **M3 speedup**: Standardized to 1.68x across all sections (was 1.65x in some places).
2. **M3 AccRet at $w_g$=0.3**: Standardized to 103.9% (was 102.5% in some places).
3. **Baseline TPS**: Standardized to 33.8 TPS for GSM8K throughout the main text. The pilot baseline (58.5 TPS from 100-sample subsets) is noted where pairwise/three-way experiments use it (Section 4.2 caveat).
4. **Opening TPS**: Changed from rounded "34 TPS" to precise "33.8 TPS" in the introduction.

## Suggestions for Additional Visuals

1. **Composition taxonomy summary diagram** (Discussion Section 6.3): A simple 2x2 or flow diagram showing the three pairs and their verdicts (synergy/near-orthogonal/task-dependent/interference) would make the three design principles more memorable.

2. **MATH500 companion to Table 3**: Table 3 reports only GSM8K results. A companion table (or additional columns) with MATH500 data would allow readers to verify combined metric claims directly.

3. **Confidence interval visualization for pairwise Ortho**: Since pairwise results are pilot-scale (single seed, 100 samples), adding error bars or bootstrap confidence intervals to Figure 4 would improve transparency.
