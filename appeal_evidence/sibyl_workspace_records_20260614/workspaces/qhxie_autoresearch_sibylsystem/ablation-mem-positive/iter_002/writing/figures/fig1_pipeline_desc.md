# Figure 1: Semantic Probe Pipeline Architecture

## Description
End-to-end pipeline diagram showing the absorption detection workflow.

## TikZ Structure
- Vertical flow diagram with 5 nodes connected by arrows
- Node 1 (blue): WordNet Categories (10 categories)
- Node 2 (blue): 15 Hyponyms per Category
- Node 3 (green): SAE Activations (GemmaScope / GPT-2)
- Node 4 (orange): Logistic Probe (AUROC evaluation)
- Node 5 (red): Absorption Metrics (Ablation + Projection)
- Side labels show technical details for each step

## Key Takeaway
The complete training-free analysis pipeline from semantic categories to absorption detection.

## Source
Methodology description in Section 3.
