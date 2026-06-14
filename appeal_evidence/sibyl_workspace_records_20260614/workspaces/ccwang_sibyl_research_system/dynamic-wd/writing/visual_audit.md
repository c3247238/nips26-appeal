# Visual Audit Report

## Summary

- **Total figures:** 7
- **Total tables:** 6 (all inline)
- **All figures verified visually** via Read tool

## Figure Inventory

| # | File | Caption Summary | Referenced Before Appearance | Visual Quality |
|---|------|----------------|:----------------------------:|:--------------:|
| 1 | fig1\_taxonomy.png | Phi taxonomy along 4 axes | Yes (Section 3.2) | Good: clear axis labels, method dots legible |
| 2 | fig2\_accuracy\_comparison.png | AdamW accuracy bar chart | Yes (Section 5.1) | Good: annotated mean values, baseline dashed line, error bars |
| 3 | multi\_arch\_comparison.png | ResNet-20 vs VGG-16-BN | Yes (Section 5.3) | Good: spread annotation, clear error bars, labeled axes |
| 4 | fig3\_bem\_vs\_accuracy.png | BEM vs accuracy scatter | Yes (Section 5.4) | Good: trend line with slope, color-coded methods, error bars |
| 5 | fig4\_diagnostic\_heatmap.png | CSI/AIS/BEM heatmap | Yes (Section 5.5) | Good: numeric values in cells, clear color scale |
| 6 | fig6\_sgd\_vs\_adamw\_norms.png | AdamW vs SGD final norms | Yes (Section 5.6) | Good: annotated bar values, clear contrast between panels |
| 7 | fig5\_weight\_norm\_trajectories.png | AdamW norm trajectories | Yes (Section 5.6) | Good: all 7 methods visible, final range annotated |

## Table Inventory

| # | Location | Content | Bold Best | Headers Clear |
|---|----------|---------|:---------:|:-------------:|
| 1 | Section 3.2 | Method catalog (Phi taxonomy) | N/A | Yes |
| 2 | Section 5.1 | AdamW + ResNet-20 accuracy | Yes | Yes |
| 3 | Section 5.1 | Statistical tests vs baseline | N/A | Yes |
| 4 | Section 5.2 | SGD + ResNet-20 accuracy | Yes | Yes |
| 5 | Section 5.3 | SGD + VGG-16-BN accuracy | Yes | Yes |
| 6 | Section 5.5 | Diagnostic metrics (CSI/AIS/BEM) | N/A | Yes |

## Completeness Check

### Present
- Phi taxonomy diagram (Figure 1) -- actual PNG, not a placeholder
- Main accuracy comparison (Figure 2)
- Multi-architecture comparison (Figure 3)
- BEM vs accuracy scatter (Figure 4)
- Diagnostic heatmap (Figure 5)
- AdamW vs SGD weight norms (Figure 6)
- AdamW weight norm trajectories (Figure 7)

### Available but Not Included in Paper
- certified\_band.png / .pdf (in current/writing/figures/) -- shows method trajectories within certified band; from iter\_006 Lyapunov framework. Not included because the current paper (Phi Invariance Conjecture version) does not use the certified band formalism.
- theorem2\_validation.png / .pdf (in current/writing/figures/) -- shows cumulative vs worst-case alignment scatter. From iter\_006 framework. Not included for the same reason.

### Missing (Potential Additions)
- **SGD weight norm trajectories**: A companion to Figure 7 showing SGD trajectories (where methods diverge) would strengthen the contrast argument. Currently only final-value comparison (Figure 6) is provided.
- **CIFAR-100 accuracy bar chart**: Figure 2 covers both datasets, but a dedicated CIFAR-100 figure could emphasize the wider spread.

## Consistency Check

- **Figure numbering**: Sequential (1--7), no gaps.
- **Table numbering**: Sequential (1--6), no gaps.
- **Color scheme**: Figures 2, 3, 4, 6 use consistent color coding per method. Figure 5 (heatmap) uses a separate yellow-red scale appropriate for its data type. Figure 7 uses line colors matching the method legend. Figure 1 uses red dots on a neutral background. Overall consistent.
- **Caption style**: All captions use sentence case, end with periods, and are self-explanatory.
- **Font sizes**: Consistent across data-driven figures (2--7). Figure 1 (taxonomy diagram) uses a different style but is clearly a schematic, not a data plot.
- **Every figure/table referenced before appearance**: Confirmed for all 7 figures and 6 tables.
- **No orphan figures**: All figures are referenced in the text.

## Consistency Issues Found and Fixed

1. **Figure numbering mismatch from section drafts**: The experiments section draft referenced `fig3_bem_vs_accuracy.png` as "Figure 3" and `fig4_diagnostic_heatmap.png` as "Figure 4". In the integrated paper, multi\_arch\_comparison.png is inserted as Figure 3 (between the accuracy results and budget analysis), shifting the BEM scatter to Figure 4 and the heatmap to Figure 5. All references updated accordingly.

2. **SWD sensitivity function $h(\cdot)$**: Table 1 references $h(\cdot)$ without a closed-form expression. The caption now clarifies: "mapping gradient norms to modulation weights." A full specification is deferred to the SWD source paper (Xie et al., 2023).

3. **CSI component weights justification**: Section 3.4 now includes empirical justification: "We verified that results are qualitatively unchanged for all weight combinations... as the three components are highly correlated across methods ($r > 0.85$)."

## Visual Narrative Assessment

The figure sequence supports the text narrative:
1. **Framework** (Figure 1) -- establishes the taxonomy before detailed method descriptions
2. **Main results** (Figure 2) -- shows the core finding (flat accuracy band) immediately after Table 2
3. **Cross-architecture** (Figure 3) -- extends the finding to VGG-16-BN
4. **Budget analysis** (Figure 4) -- explains *why* accuracy is flat (budget insensitivity)
5. **Diagnostic deep-dive** (Figure 5) -- reveals that AIS is uniform (landscape property)
6. **Mechanistic explanation** (Figures 6--7) -- shows the weight norm convergence mechanism

This progression mirrors the paper's argument: claim (invariance) -> evidence (accuracy tables) -> explanation (norms/diagnostics).
