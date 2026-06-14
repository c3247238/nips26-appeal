# Figure 2: ComposeAccel Architecture Diagram

## Purpose
Illustrate the IGSD draft-partition-refine pipeline alongside the standard DLM denoising baseline, and show where M1 (entropy-based KV caching) and M3 (AR-guided unmasking) integrate into the pipeline.

## Layout: Two-row horizontal flow

### Row 1 (Top): Standard DLM Denoising (Baseline)
- A single horizontal arrow spanning steps $t = 0$ to $t = T = 64$.
- Each step is a box labeled "Forward Pass $f_\theta$".
- Between consecutive boxes, a small icon represents the full $O(N^2)$ attention computation.
- Below the arrow, label: "All $N$ tokens processed at every step. No KV reuse."
- Color: light gray background for all 64 step boxes.

### Row 2 (Bottom): IGSD + M1 + M3 Composed Pipeline
Split into three phases separated by vertical dashed lines:

#### Phase A: Draft Phase (steps $0 \to T_{\text{draft}}$)
- $T_{\text{draft}}$ boxes (e.g., 32 of 64) in orange.
- All $N$ tokens processed at each step (same as baseline).
- M3 integration point: a green arrow labeled "$g_\phi$" from a small Qwen2.5-0.5B icon above, pointing into each draft step box. The arrow is annotated with "$\tilde{p}_t = (1-w_g) p_t + w_g q_t$".
- M1 integration point: a blue dashed arrow labeled "KV reuse ($H_i^{t-1} < \eta$)" connecting from step $t-1$'s KV output to step $t$'s KV input. Annotation: "CHR = 56--93%".

#### Phase B: Confidence Partitioning (at step $T_{\text{draft}}$)
- A single vertical gate box labeled "Confidence Gate ($\tau$)".
- Input: logit confidence for each token position from step $T_{\text{draft}}$.
- Two output arrows:
  - Upward arrow to a blue set labeled "$\mathcal{S}_{\text{accept}}$ (frozen, $\alpha \approx 88.6\%$)".
  - Downward arrow to a red set labeled "$\mathcal{S}_{\text{reject}}$ (continue refining)".
- Small annotation: "KL$_t$ guides threshold".

#### Phase C: Refine Phase (steps $T_{\text{draft}} \to T_{\text{full}} = 64$)
- $(T_{\text{full}} - T_{\text{draft}})$ boxes in dark blue.
- Only $|\mathcal{S}_{\text{reject}}|$ tokens processed (narrow boxes to indicate reduced computation).
- Frozen tokens ($\mathcal{S}_{\text{accept}}$) shown as a blue bar above, connected by a "skip" arrow that bypasses the refine boxes.
- M1 integration: blue dashed arrow more prominent here. Annotation: "Refine CHR = 94.3% (frozen tokens have near-zero entropy)".
- M3 integration: green arrow still present but thinner (optional in refine phase).

### Output
- Final box at the right: "Output $\hat{\mathbf{x}}$" combining frozen and refined tokens.

## Key Visual Annotations
1. A brace below the draft phase labeled "$T_{\text{draft}} = 32$ (Pareto-optimal)".
2. A brace below the refine phase labeled "$T_{\text{full}} - T_{\text{draft}} = 32$ remaining steps".
3. A speed callout: "Effective computation: $T_{\text{draft}} \times N + (T_{\text{full}} - T_{\text{draft}}) \times |\mathcal{S}_{\text{reject}}|$".
4. Color legend in top-right corner:
   - Orange = Draft phase
   - Blue = Refine phase / M1 KV caching
   - Green = M3 AR guidance
   - Gray = Baseline (no acceleration)

## Style
- Clean, minimal TikZ-style diagram with sharp edges.
- Use consistent colors with the rest of the paper figures.
- Font: sans-serif for labels, math mode for symbols.
- Horizontal flow, left to right. Total width ~\textwidth.
