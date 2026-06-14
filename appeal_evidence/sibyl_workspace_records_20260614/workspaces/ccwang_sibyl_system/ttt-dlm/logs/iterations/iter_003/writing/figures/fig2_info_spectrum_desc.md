# Figure 2: Information Augmentation Spectrum

## Purpose
Visualize the four levels of cross-step information transfer and their mechanisms, establishing the systematic ablation framework used throughout the paper.

## Layout: Four stacked horizontal panels, each showing one denoising step transition (step t -> step t-1)

### Panel 0: Vanilla (Level 0)
- Left side: Sequence at step t with mix of gray [M] and blue revealed tokens
- Right side: Sequence at step t-1 with more revealed tokens
- Arrow between: solid black arrow labeled "sample & reveal"
- Between-step channel: thin dashed line labeled "discrete tokens only"
- Right margin: "1x compute" and "No memory"

### Panel 1: DMI (Level 1 --- Diffusion Memory Injection)
- Same left-right sequence transition as above
- Additional element: An orange curved arrow from step t's logits (shown as a small probability bar chart above the sequence) through a "softmax(tau) @ E" operation box, producing a soft embedding vector
- The soft embedding vector is shown merging (alpha blend) into the masked positions of step t-1's input
- Right margin: "~1x compute" and "1-step embedding echo"
- Formula annotation: "e_soft = softmax(z/tau) . E"

### Panel 2: SCP (Level 2 --- Self-Contradiction Probing)
- Same left-right sequence transition
- Additional element: A magnifying glass icon over the revealed tokens at step t
- For each revealed token, a small leave-one-out check shown: mask that position -> forward pass -> compare prediction
- Red highlights on 2-3 positions where the model's re-prediction differs (contradictions detected)
- These red positions are remasked (turned back to gray [M]) before step t-1
- Right margin: "~7x compute" and "Token-level error detection"
- Annotation: "1.9% contradiction rate on Dream-7B"

### Panel 3: DTA (Level 3 --- Denoising-Time Adaptation)
- Same left-right sequence transition
- Additional element: A blue gradient arrow flowing from step t downward through a "backward pass" box (showing gradient symbols), then into LoRA parameter blocks (Delta_theta), then curving forward to step t-1's model
- The LoRA block shows cumulative update: Delta_theta <- gamma * Delta_theta - eta * grad(L)
- A fading trail of previous updates shown behind, suggesting accumulation
- Right margin: "~4x compute" and "Full-trajectory parameter memory"

### Right Margin Summary Column
A vertical axis on the far right with labels:
- Top: "Information Granularity" with an upward arrow
  - Level 0: "None"
  - Level 1: "Embedding"
  - Level 2: "Token"
  - Level 3: "Parameter"
- Compute cost indicators: 1x, ~1x, ~7x, ~4x

### Visual Style
- Consistent color scheme across all panels: gray=masked, blue=revealed, orange=DMI elements, red=SCP contradictions, dark blue/purple=DTA gradients
- Each panel has a thin border and a level label on the left (Level 0, 1, 2, 3)
- Panels are vertically stacked with small spacing
- A vertical "increasing expressivity" arrow on the left side spanning all four panels

### Caption
"**Figure 2: Information augmentation spectrum.** Four levels of cross-step information transfer during DLM denoising. Level 0 (Vanilla): only discrete tokens pass between steps. Level 1 (DMI): soft embedding injection from previous step's logits with near-zero overhead. Level 2 (SCP): leave-one-out probing detects self-contradictory tokens at ~7x compute cost. Level 3 (DTA): online LoRA updates create persistent parameter-level memory at ~4x overhead. The spectrum provides a systematic ablation of cross-step information value at increasing computational cost and expressivity."

## Key Takeaway
The spectrum provides a principled framework for ablating the value of cross-step information transfer in DLM inference, spanning from zero-cost embedding injection to full parameter-level adaptation. Each level introduces a qualitatively different mechanism, enabling controlled comparison of information types and persistence.

## Generation Method
TikZ (LaTeX) or equivalent vector diagram tool. Full-width figure at NeurIPS format (~7.5 inches wide).
