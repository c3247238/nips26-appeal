# Figure 2: Metric Comparison -- Ablation vs Projection Absorption Rates

## Description
Side-by-side bar chart comparing ablation-based and projection-based absorption rates across architectures.

## TikZ Structure
- Grouped bar chart with ybar style
- X-axis: GemmaScope, GPT-2
- Y-axis: Absorption Rate (0 to 1.1)
- Blue bars: Ablation-based rates (0.0 for GemmaScope, 0.333 for GPT-2)
- Red bars: Projection-based rates (1.0 for both)
- Legend at bottom

## Key Takeaway
Projection metric detects absorption at 91-98% while ablation metric detects 0-33%, demonstrating the dramatic sensitivity difference.

## Data Source
e3v2_semantic_scaled.json, e7_cross_architecture.json
