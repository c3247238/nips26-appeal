# Figure 2: Competitive Suppression Mechanism

## Type
Flow diagram / mechanism illustration (TikZ)

## Layout
Three-panel horizontal layout showing causal chain.

## Panel 1: Co-occurrence (left, 30% width)
- Title: "(1) Co-occurrence"
- Input text: "The quick brown fox..."
- Two neurons side by side:
  - Parent neuron (blue): "starts with any letter"
  - Child neuron (red): "starts with 'Q'"
- Both neurons have medium activation bars (equal height)
- Label: "Input contains 'quick' -- both concepts are present"

## Panel 2: Child Fires, Inhibits Parent (center, 35% width)
- Title: "(2) Competitive Suppression"
- Same two neurons:
  - Parent neuron (blue, dimmed): activation bar LOW
  - Child neuron (red, bright): activation bar HIGH
- Arrow from child to parent labeled "G_ij > 0" (inhibition)
- Equation: "inh_i = G_ij * z_j"
- Label: "Child fires strongly; inhibits parent via decoder correlation"
- Decoder directions shown as small vectors:
  - d_i (parent decoder): unchanged
  - d_j (child decoder): unchanged

## Panel 3: Outcome (right, 35% width)
- Title: "(3) Precision--Recall Asymmetry"
- Two outcome boxes stacked:
  - Top box (green): "Precision = 1.0" with checkmark
    - Subtext: "Decoder direction d_i unchanged"
    - Subtext: "No false positives -- selectivity preserved"
  - Bottom box (orange): "Recall < 1.0" with warning
    - Subtext: "Parent fails to fire when child present"
    - Subtext: "False negatives -- coverage reduced"
- Central equation: "z_i' = max(0, z_i - inh_i)"

## Color Scheme
- Parent: blue (#2E86AB)
- Child: red (#C73E1D)
- Inhibition arrow: red dashed
- Precision box: green (#4CAF50) with white text
- Recall box: orange (#FF9800) with white text
- Background: white

## Key Annotation
- Bottom: "Competitive suppression explains why absorption affects recall but not precision"
- Arrow labels use notation from the paper: G_ij, z_j, z_i

## TikZ Elements
- Use tikz shapes.rectangle for neurons
- Use tikz shapes.circle for activation indicators
- Use pgfplots for activation bar charts (simple rectangles)
- Use tikz arrows.meta for inhibition arrow
- Font: sans-serif, small size for subtext
