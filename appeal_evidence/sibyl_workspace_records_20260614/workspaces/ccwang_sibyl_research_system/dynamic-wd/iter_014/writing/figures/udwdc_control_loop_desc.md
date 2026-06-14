# UDWDC Control Loop Block Diagram

## Description for TikZ rendering

The diagram shows a closed-loop feedback control system with the following blocks and signals, arranged left-to-right with a feedback loop returning from right to left:

### Blocks (left to right):
1. **Target Generator** (rounded box): Computes $\rho^*(t) = \eta_t / \tau$ from the learning rate schedule and EMA timescale $\tau = 1/(\lambda_0 \cdot \eta_0)$.
2. **Summation** (circle with $\Sigma$): Computes the error signal $e_t^l = \rho_t^l - \rho^*(t)$.  Positive input from the measured $\rho_t^l$ (feedback), negative input from $\rho^*(t)$ (reference).
3. **Proportional Gain** (rectangular box labeled "$K_p$"): Applies the proportional gain.  For UDWDC, this is equivalent to computing $\lambda_{\text{base}} \cdot (\rho_t^l / \rho^*(t))$.
4. **Clamp** (rectangular box labeled "clamp [0.1, 10]"): Bounds the ratio to prevent extreme values.
5. **Weight Update** (rectangular box labeled "$w_{t+1} = (1 - \eta_t \lambda_t^l) w_t - \eta_t g_t$"): Applies the computed WD coefficient to the SGDW/AdamW update rule.

### Feedback path (bottom, right to left):
6. **Measurement** (rectangular box labeled "$\rho_t^l = \|g_t^l\| / \|w_t^l\|$"): Computes the per-layer GW ratio from the current weights and gradients.  Output feeds back to the summation node.

### Signals:
- $\eta_t$ enters the Target Generator from above (external schedule).
- $\rho^*(t)$ flows from Target Generator to the negative input of the summation.
- $e_t^l$ flows from summation to $K_p$ block.
- $\lambda_t^l$ flows from Clamp to Weight Update.
- $w_t^l, g_t^l$ enter the Measurement block from the Weight Update (plant output).
- $\hat{\rho}_t^l$ (or $\rho_t^l$) feeds from Measurement back to the positive input of the summation.

### Color coding:
- Target generator and $\rho^*$ path: blue.
- Error and gain path: red (our method).
- Measurement feedback path: gray.
- Weight update (plant): black.

### Annotation:
- A dashed box groups the $K_p$ and Clamp blocks, labeled "UDWDC Controller."
- Below the diagram, dotted boxes indicate where other methods connect: "CWD: $K_d$ with $\alpha_t$ here", "CPR: $K_i$ accumulation here", "SWD: $K_p$ on global $\|\nabla\mathcal{L}\|$ here".
- Label the open-loop path (FixedWD): a straight line bypassing all control blocks, from $\lambda_{\text{base}}$ directly to Weight Update, labeled "Open-loop (FixedWD)".
