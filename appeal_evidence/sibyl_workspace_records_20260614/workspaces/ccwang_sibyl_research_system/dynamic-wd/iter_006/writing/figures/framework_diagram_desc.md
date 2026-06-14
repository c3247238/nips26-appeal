# Phi Modulator Framework Diagram (TikZ)

## Description
A control loop diagram showing the phi modulator framework.

## Components (left to right)
1. **Training State** box (input): Contains $w_t$, $g_t$, $t$ (epoch)
2. **Arrow** from Training State to Phi Modulator
3. **Phi Modulator** box (central, highlighted): $\phi(t, w, g)$ with sub-labels showing method variants:
   - $\phi = 1$ (Constant)
   - $\phi = \mathbf{1}[\text{sign}]$ (CWD)
   - $\phi = h(\|g\|)$ (SWD)
   - $\phi = \mathbf{1}[\langle p, w \rangle > 0]$ (PMP-WD)
4. **Arrow** from Phi Modulator to Effective WD
5. **Effective WD** box: $\lambda_{\text{eff}} = \phi \cdot \lambda_{\text{base}}$
6. **Certified Band** bracket (dashed, overlaying Effective WD): $[\lambda_{\min}, \lambda_{\max}]$
7. **Arrow** from Effective WD to Weight Update
8. **Weight Update** box: $w_{t+1} = (1 - \gamma_t \lambda_{\text{eff}}) w_t - \gamma_t g_t$
9. **Feedback arrow** from Weight Update back to Training State (loop)
10. **Diagnostic Metrics** box (below, connected by dashed arrows to Effective WD and Weight Update):
    - BEM: budget deviation
    - CSI: coupling stability
    - AIS: alignment informativeness

## Style
- Main loop: solid arrows, thick lines
- Certified band: dashed blue bracket
- Diagnostic metrics: dashed connections, gray box
- Phi modulator box: blue fill
- All text in serif math font
