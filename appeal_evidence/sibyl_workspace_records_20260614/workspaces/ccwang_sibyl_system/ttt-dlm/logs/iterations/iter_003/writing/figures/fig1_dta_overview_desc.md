# Figure 1: DTA Algorithm Overview

## Purpose
Illustrate the E-step/M-step alternation within the denoising loop, contrasting standard DLM denoising (top) with DTA-enhanced denoising (bottom).

## Layout: Two-row comparison diagram

### Top Row: Standard DLM Denoising (Vanilla)
- Left: Fully masked sequence x_T represented as a row of gray [M] boxes
- Arrows pointing right through 3--4 intermediate steps (x_{T-1}, x_{T-2}, ..., x_1)
- At each step, some [M] boxes turn into colored token boxes (blue for revealed tokens)
- Between each pair of steps, a dashed vertical line with a red "X" and label: "representations discarded"
- Right: Fully revealed sequence x_0 with all boxes colored blue
- Annotation above: "Information Islands --- no state carries across steps"

### Bottom Row: DTA-Enhanced Denoising
- Same left-to-right progression of sequences x_T -> x_0
- At each step (after warmup), a cycle diagram below the sequence:
  1. E-step box (green): "Predict masked tokens via f_{theta+Delta_theta}" with arrow to revealed tokens
  2. M-step box (orange): "Mask 20% of revealed -> compute MLM loss -> update LoRA (1 AdamW step)"
  3. A blue curved arrow labeled "Delta_theta (gamma-decayed)" connecting M-step output to the next step's E-step
- LoRA adapter illustration: small side module (two small rectangles labeled A and B with r<<d annotation) attached to the last 2 Transformer layer blocks (shown as a simplified transformer stack inset)
- First 20% of steps have a "warmup: skip M-step" annotation with lighter coloring
- Annotation below: "Persistent parameter-level memory across all denoising steps"

### Visual Details
- Color scheme: Gray = masked, Blue = revealed tokens, Green = E-step, Orange = M-step, Purple = LoRA adapters
- The LoRA adapter inset shows: FFN block with gate_proj / up_proj / down_proj, each with a small LoRA branch (A->B) drawn alongside
- Key metrics annotated: "540K params (0.007% of 7.6B)", "~4x overhead"
- Zero-initialization symbol: Delta_theta = 0 at step T with a "=" sign to f_theta (showing initial equivalence)

### Caption
"**Figure 1: DTA algorithm overview.** Standard DLM denoising (top) discards all continuous representations between steps, creating Information Islands. DTA (bottom) adds a lightweight M-step after each E-step: masking 20% of revealed tokens, computing an MLM loss, and updating zero-initialized LoRA adapters (rank 4, last 2 layers, 540K parameters) via a single AdamW step. The cumulative decay $\gamma = 0.95$ prevents parameter drift while preserving cross-step memory. Updates begin after a 20% warmup phase."

## Key Takeaway
DTA converts the implicit TTT structure of DLM denoising into explicit test-time learning by adding a lightweight parameter update (M-step) at each denoising step, while preserving the base model through zero-initialized LoRA and cumulative decay.

## Generation Method
TikZ (LaTeX) or equivalent vector diagram tool. The diagram should be clean and publication-ready at NeurIPS format width (~5.5 inches for single-column or ~7.5 inches for full-width).
