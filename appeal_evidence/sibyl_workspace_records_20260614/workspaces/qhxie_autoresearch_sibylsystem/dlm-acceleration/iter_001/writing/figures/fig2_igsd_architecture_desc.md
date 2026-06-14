# Figure 2: IGSD Algorithm Architecture Diagram

## Description (for TikZ or manual rendering)

A horizontal three-phase flow chart with the following structure:

### Phase 1: DRAFT (left block, light blue)
- Input: prompt tokens (shown as filled rectangles) + fully masked canvas (shown as gray [MASK] rectangles)
- Arrow labeled "$T_{\text{draft}} = 16$ steps"
- Output: draft sequence $x_{\text{draft}}$ with per-token confidence scores $c_i$
- Each token position annotated with confidence: high-confidence tokens in green, low-confidence in orange

### Phase 2: PARTITION (center, thin vertical divider)
- Decision boundary: $c_i \geq \tau$ (threshold line at $\tau = 0.9$)
- Splits tokens into two sets:
  - $S_{\text{accept}}$ (green, ~52% of positions): tokens accepted from draft, FROZEN for remainder
  - $S_{\text{refine}}$ (orange, ~48% of positions): tokens requiring full refinement

### Phase 3: REFINE (right block, light orange)
- Shows the sequence with frozen tokens (green, locked icon) providing stable KV context
- Arrow labeled "$T_{\text{full}} = 64$ steps"
- Only $S_{\text{refine}}$ tokens undergo denoising (orange, animated/active)
- Annotation: "Frozen tokens $\to$ $H_i = 0$ $\to$ KV-cache always hits" (connecting to M1 synergy)

### Output (far right)
- Merged final sequence: green (accepted) + refined orange tokens

### Key visual elements:
- Color coding: green = accepted/frozen, orange = refine, gray = masked
- Lock icons on frozen tokens in REFINE phase
- Dashed arrow from frozen tokens to "KV-cache hit" annotation (showing synergy with M1)
- Confidence histogram below PARTITION phase showing the tau=0.9 split point

### Dimensions:
- Full width ~\textwidth, height ~4cm
- Three phases should be visually equal width
- Use sans-serif font, consistent with paper style
