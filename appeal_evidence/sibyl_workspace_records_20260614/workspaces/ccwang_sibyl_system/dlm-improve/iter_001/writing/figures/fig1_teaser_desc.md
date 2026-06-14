# Figure 1: Problem Illustration and Key Results (Teaser)

## Layout
Composite 3-panel figure, full page width (12 inches x 3.5 inches).

## Panel (a): DLM Denoising Overconfidence
- Visual: A sequence of tokens at a mid-denoising step. Above each token, a confidence bubble with the model's stated confidence (e.g., 0.85) and the actual accuracy against the final output (e.g., 0.62).
- Annotation: "SC-ECE = 0.22" with an arrow pointing to the gap between confidence and accuracy.
- Color: Confidence bubbles in pink/red (COLORS['sc']), actual accuracy in gray.
- Message: "DLMs are systematically overconfident during denoising."

## Panel (b): Accuracy vs NFE — Diminishing Returns
- Plot type: Scatter plot with connected line.
- X-axis: NFE (Number of Forward Evaluations), range 30-130.
- Y-axis: GSM8K Accuracy (%), range 28-36.
- Points:
  - Standard-32: (32, 29.3%) — gray circle
  - Standard-64: (64, 30.9%) — gray circle
  - DNB-84: (83, 30.9%) — brown diamond, labeled "DNB-84 = Std-64"
  - Standard-128: (124, 32.1%) — gray circle
  - CARD-84: (71, 34.9%) — blue star, larger, labeled "CARD-84"
- Annotation: Dashed horizontal line at 30.9% connecting Std-64 and DNB-84 to show equivalence.
- Pareto frontier line connecting Std-64 → CARD-84.
- Message: "More standard steps waste compute; CARD redirects compute to targeted revision."

## Panel (c): CARD Method Overview
- Flow diagram, left to right:
  1. [Box] "Input prompt + [M][M]...[M]" (fully masked)
  2. [Arrow] "64 cosine steps"
  3. [Box] "Draft output" with some tokens colored (most blue/correct, a few red/uncertain)
  4. [Arrow] "Entropy scoring"
  5. [Box] Same sequence with top-10% highest-entropy tokens highlighted in orange and re-masked
  6. [Arrow] "3 revision steps"
  7. [Box] "Revised output" with previously-red tokens now green/corrected
- Below the flow: "Total NFE: ~71 | +4.0 pp on GSM8K"

## Data Source
- Panel (b): `pareto_analysis.json`, `full_llada_gsm8k_monitor.json`
- Panels (a), (c): Conceptual/schematic
