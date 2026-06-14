# Figure 3: CARD Method Diagram

## Layout
Single-row flow diagram, full page width (12 inches x 3 inches).

## Flow (left to right)

### Box 1: Input
- Prompt tokens (fixed, shown in dark gray) followed by a fully masked response region: "[M][M][M]...[M]"
- Label: "Input"

### Arrow 1
- Label: "Phase 1: 64 cosine steps"
- Color: COLORS['standard']

### Box 2: Draft Output
- Full sequence with most tokens resolved.
- Token coloring: each token has a background gradient proportional to its entropy.
  - Low-entropy tokens (confident): light blue background
  - High-entropy tokens (uncertain): orange/red background
- Entropy heatmap bar below the sequence showing the entropy distribution.
- Label: "Draft + Entropy Scoring (1 NFE)"
- Annotation: "SC-ECE = 0.22, but ranking reliable (r = 0.44)"

### Arrow 2
- Label: "Re-mask top-10% entropy"
- Color: COLORS['ablation'] (orange)

### Box 3: Revision Input
- Same sequence, but the top-10% highest-entropy positions are replaced with [M].
- Remaining tokens shown with blue (frozen context) background.
- Re-masked positions highlighted in orange.
- Label: "Selective Remasking"

### Arrow 3
- Label: "Phase 2: 3 revision steps"
- Color: COLORS['ours'] (blue)

### Box 4: Revised Output
- Full sequence with all tokens resolved.
- Previously re-masked positions now shown in green (corrected).
- Label: "Revised Output"
- Annotation: "Total NFE: ~71"

## Key Visual Elements
- Entropy heatmap bar should use a diverging colormap (blue-white-red)
- Frozen context tokens should be visually distinct (muted blue) from revision targets (bright orange → green)
- Arrows should be thick with clear labels

## Generation
TikZ recommended for clean vector output. Alternatively, a high-quality conceptual diagram tool.
