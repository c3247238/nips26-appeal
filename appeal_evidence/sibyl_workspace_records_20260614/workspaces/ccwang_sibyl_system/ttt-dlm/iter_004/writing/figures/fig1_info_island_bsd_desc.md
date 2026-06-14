# Figure 1: Information Island Problem and BSD Teaser

## Layout

Full-width figure with two main panels (Left, Right) and a small inset bar chart (bottom-right corner).

## Left Panel: Vanilla MDLM Denoising ("Information Islands")

A vertical pipeline showing 3 denoising steps (t=T, t=T-1, t=T-2), connected by downward arrows.

Each step consists of:
1. **Input row**: A sequence of boxes representing token positions. Filled boxes = revealed tokens (light blue), hatched boxes = mask embeddings (gray with diagonal lines). Label: `x_t`.
2. **Forward pass**: Arrow into a rounded rectangle labeled "Transformer $f_\theta$".
3. **Output row**: Full logit distributions shown as small bar charts above each position (all positions, including revealed ones). Label: "Logit predictions $\ell^t$".
4. **Argmax + Discard**: A red "X" symbol over the logit bars for mask positions that are NOT selected, with a downward arrow labeled "argmax" pointing to the next step's input. A red dashed box around the discarded logits with annotation: "Distributional info discarded".

Between steps, show that the newly revealed tokens carry over, but mask positions reset to plain gray mask_emb boxes (no memory of previous logits).

Annotation on the left margin: "Each step is an information island — no cross-step memory".

## Right Panel: BSD Denoising ("Continuous Belief Accumulation")

Same vertical pipeline structure (3 steps), but with key differences:

Each step consists of:
1. **Input row**: Revealed tokens (light blue) + **belief state boxes** (gradient-filled, green-to-blue spectrum indicating distributional richness, NOT gray mask_emb). Label: `[x_p; b^t]`.
2. **Forward pass**: Arrow into "Transformer $f_\theta$".
3. **Output row**: Logit distributions as small bar charts.
4. **EMA Update**: A green circular arrow from the logit bars back to the belief boxes of the next step, labeled "$b_i^{t-1} = (1-\alpha^t) b_i^t + \alpha^t \sum_v p(v) e_v$". NO red X — logits are retained in beliefs.

Show the belief boxes becoming progressively more "concentrated" (the gradient becoming more peaked / narrower distribution) across steps, visually conveying information accumulation.

At the bottom, show a **Phase transition**: A horizontal dashed line labeled "Phase 1 → Phase 2 (step k)". Below it, the final 1-2 steps use standard confidence-based unmasking from the now-concentrated belief states.

Annotation on the right margin: "Beliefs accumulate information continuously across steps".

## Inset Bar Chart (Bottom-Right)

Small grouped bar chart showing accuracy on Countdown:
- Vanilla: 4.7% (gray bar)
- DMI: 9.3% (light green bar)
- BSD: 6.2--12.5% (medium green bar, use hatching to indicate range)
- A-CFG: 12.5% (dark blue bar)

Y-axis: "Accuracy (%)", range 0--15%.
X-axis: Method names.
Title: "Countdown Accuracy".

## Color Scheme

- Mask embeddings (vanilla): Gray (#CCCCCC) with diagonal hatching
- Revealed tokens: Light blue (#AED6F1)
- Belief states: Green-to-blue gradient (#82E0AA to #5DADE2)
- Discarded info (vanilla): Red dashed border (#E74C3C)
- EMA update arrows: Green (#27AE60)
- Phase transition line: Orange dashed (#F39C12)

## Typography

- All labels in sans-serif (Helvetica/Arial)
- Math in standard LaTeX math mode
- Step labels: bold "Step t=T", "Step t=T-1", etc.
- Panel titles: bold, 12pt, "Vanilla MDLM" and "Belief-State Diffusion (BSD)"
