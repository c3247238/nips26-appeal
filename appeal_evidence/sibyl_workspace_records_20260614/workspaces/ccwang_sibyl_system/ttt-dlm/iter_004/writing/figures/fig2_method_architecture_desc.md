# Figure 2: Method Architecture — BSD and A-CFG Pipelines

## Layout
Two-panel horizontal figure. Panel A (left, ~60% width): BSD pipeline. Panel B (right, ~40% width): A-CFG pipeline.

---

## Panel A: Belief-State Diffusion (BSD)

### Phase 1: Continuous Belief Refinement (steps T to k+1)

**Flow (top-down loop):**

1. **Input row**: `[prompt embeddings | b^t (belief vectors)]`
   - Belief vectors shown as gradient-colored blocks (uniform gray at t=T, increasingly colored as beliefs concentrate)

2. **Transformer block**: `f_theta` forward pass
   - Output: logit vectors `l^t` per position

3. **Softmax + Embedding mixture**:
   - `p_i^t = softmax(l_i^t / tau_t)`
   - `b_hat_i^t = sum_v p_i^t(v) * e_v`
   - Shown as weighted sum of colored embedding vectors

4. **EMA Update**:
   - `b_i^{t-1} = (1-alpha^t) * b_i^t + alpha^t * b_hat_i^t`
   - Arrow shows feedback loop back to step 1
   - Label: "alpha: 0.1 -> 0.8 (linear)"

5. **L2 Normalization**:
   - `||b_i|| -> ||mask_emb||`
   - Small icon: norm constraint

**Loop annotation**: "NO argmax, NO unmask — beliefs evolve continuously"

**FLOPs annotation**: "Phase 1: ~1.0x per step (same as vanilla)"

### Phase 2: Hard Token Reveal (steps k to 1)

**Flow (straight-through):**

1. **Input**: `[prompt | beliefs b^k]` (now highly concentrated)
2. **Forward pass**: standard `f_theta`
3. **Confidence-based unmasking**: top-confidence positions commit to hard tokens
4. **Output**: progressively revealed sequence

**Transition annotation**: Dashed line between phases with "k = 0.75T (25% belief, 75% hard reveal)"

**Total FLOPs annotation**: "BSD total: ~1.1x vanilla"

---

## Panel B: Adaptive Classifier-Free Guidance (A-CFG)

**Flow (top-down):**

1. **Input sequence**: `x_t` with mix of revealed tokens and masks
   - Some positions highlighted in red (low confidence)

2. **Confidence scoring**:
   - `c_i = max_v p(x_i = v | x_t)`
   - Bar chart showing per-position confidence
   - Bottom m=10% highlighted

3. **Re-masking**:
   - Arrow from x_t to x_tilde_t
   - Bottom-10% confidence positions replaced with [MASK]
   - Label: "Unconditional input"

4. **Dual forward pass** (two parallel arrows):
   - Left: `l^+ = f_theta(x_t)` — "Conditional"
   - Right: `l^- = f_theta(x_tilde_t)` — "Unconditional"

5. **Guided combination**:
   - `l_guided = l^+ + w * (l^+ - l^-)`
   - w = 1.5 (fixed), w_max = 2.0
   - Arrow merging the two logit streams

6. **Output**: Enhanced logits for unmasking

**FLOPs annotation**: "A-CFG: ~2.0x vanilla (dual forward pass)"

---

## Color Scheme
- BSD belief vectors: gradient from gray (uncertain) to blue/green (concentrated)
- A-CFG: red for low-confidence positions, green for high-confidence
- Dashed lines for phase transitions
- Bold boxes for key operations (EMA, normalization, CFG combination)

## Key Callout
Bottom center, spanning both panels:
"BSD and A-CFG operate at orthogonal levels — representation layer (BSD) vs. prediction layer (A-CFG)"
