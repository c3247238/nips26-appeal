# Figure 3: PMP-WD Control Diagram (TikZ)

## Description
Block diagram comparing PMP-WD (closed-loop state-feedback) with CWD (binary open-loop) and cosine schedule (feedforward).

## Layout (three rows)

### Row 1: PMP-WD (State-Feedback, Closed-Loop)
- Block "Measure rho_hat_t" (input: training state theta_t, g_t)
- Arrow to: Comparator circle (rho* - rho_hat_t)
- Arrow to: Block "Gain kappa"
- Arrow to: Block "Clip [0, lambda_max]"
- Arrow to: Output "lambda*(t)"
- Feedback arrow from training state back to measurement block
- Label: "Closed-loop: corrects deviations"

### Row 2: CWD (Binary Mask, No Feedback)
- Block "Compute sign(theta), sign(u_t)"
- Arrow to: Block "Binary mask: 1 or 0"
- Arrow to: Output "lambda * mask"
- No feedback loop
- Label: "Open-loop binary: no continuous modulation"

### Row 3: Cosine Schedule (Feedforward, No Measurement)
- Block "Time index t"
- Arrow to: Block "0.5*(1+cos(pi*t/T))"
- Arrow to: Output "lambda * schedule(t)"
- No measurement of training state
- Label: "Feedforward: ignores training state"

## Key Elements
- PMP-WD path highlighted in blue (preferred)
- CWD path in red (binary limitation)
- Cosine path in gray (no state information)
- Comparator node clearly shows the error signal (rho* - rho_hat_t)

## Caption
"PMP-WD (top) is the first closed-loop WD controller: it measures the gradient-to-weight ratio rho_hat_t, compares it to the steady-state target rho*, and adjusts lambda proportionally. CWD (middle) uses a binary mask without continuous feedback. Cosine schedule (bottom) is purely feedforward, ignoring training state entirely."
