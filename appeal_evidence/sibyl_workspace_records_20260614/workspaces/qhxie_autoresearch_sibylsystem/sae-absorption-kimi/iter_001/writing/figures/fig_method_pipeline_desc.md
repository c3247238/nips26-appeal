# Figure: Method Pipeline Overview

## Type
Architecture / flow diagram (TikZ)

## Description
A horizontal flowchart showing the training-free evaluation pipeline from left to right.

**Box 1 (Input):** "Public Checkpoints"
- Two sub-bullets below:
  - SAELens: 27 GPT-2 Small checkpoints (Standard, TopK, feature splitting)
  - SAEBench: 314 checkpoints (7 architectures, Gemma-2-2B / Pythia-160M)

**Arrow 1:** "Load via SAELens / HF Datasets"

**Box 2 (Processing):** "Training-Free Metric Computation"
- Three parallel sub-boxes inside:
  - Absorption & Hedging
  - Reconstruction Fidelity (L0, EV, CE Rec)
  - Downstream Metrics (F1, RAVEL)

**Arrow 2:** "Normalize & Aggregate"

**Box 3 (Analysis):** "Multi-Objective Analysis"
- Three parallel sub-boxes inside:
  - Pareto Front + Mann-Whitney U (E1)
  - Partial Correlation + OLS (E2)
  - Task-Agnostic Validation (E3)

**Arrow 3:** "Synthesize"

**Box 4 (Output):** "Conclusion: No architecture dominates; absorption has unique causal cost; first-letter benchmark may not generalize."

## Style Notes
- Use rectangular nodes with rounded corners.
- Use light blue fill for input/output boxes, light gray fill for processing/analysis boxes.
- Arrows should be thick, dark gray.
- Keep the diagram compact, suitable for single-column width.
