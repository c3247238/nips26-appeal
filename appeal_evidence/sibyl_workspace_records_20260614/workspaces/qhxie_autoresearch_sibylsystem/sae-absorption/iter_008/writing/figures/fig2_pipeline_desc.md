# Figure 2: Experimental Pipeline Schematic

## Description for TikZ rendering

The diagram flows left-to-right with a top branch and a bottom branch diverging after the SAE encoding step.

### Main Pipeline (top-to-bottom, then left-to-right)

1. **Input** (rounded box): "Token $t$ in context"
2. **Arrow** down to **Model** (box): "Gemma 2 2B, Layer $L$"
3. **Arrow** right to **Residual Stream** (box): "$\mathbf{x}^{(L)}$"
4. **Fork**: Two arrows from Residual Stream

#### Upper Branch (Absorption Measurement)
4a. **Arrow** right to **SAE Encoding** (box): "$\text{SAE}(\mathbf{x}^{(L)}) \to \hat{\mathbf{x}}, \mathbf{a}$"
5a. **Arrow** right to **Probe on SAE Output** (box): "$f_{\text{probe}}(\hat{\mathbf{x}}) \to \hat{y}_{\text{sae}}$"
6a. **Arrow** right to **Compare** (diamond): "$\hat{y}_{\text{raw}} = y$ but $\hat{y}_{\text{sae}} \neq y$?"
7a. Yes arrow down to **False Negative** (red box)
8a. **Arrow** right to **IG Attribution** (box): "Integrated Gradients (10 steps)"
9a. **Arrow** right to **Absorption Detection** (box): "$\cos(\mathbf{d}_i, \mathbf{w}_k) > \tau_{\cos}$ and gap $\geq \tau_{\text{gap}}$"
10a. **Arrow** right to **Output** (rounded box): "Absorption Rate (AR)"

#### Lower Branch (Direct Probe)
4b. **Arrow** down-right to **Probe on Raw** (box): "$f_{\text{probe}}(\mathbf{x}^{(L)}) \to \hat{y}_{\text{raw}}$"
5b. **Dashed arrow** up to Compare diamond (step 6a)

### Side Branch (Activation Patching) -- diverges from step 5a
- From SAE Encoding: **Arrow** down to **Intervention** (orange box): "$\text{do}(a_c := 0)$"
- **Arrow** right to **Modified SAE Output** (box): "$\hat{\mathbf{x}}'$"
- **Arrow** right to **Probe on Modified** (box): "$f_{\text{probe}}(\hat{\mathbf{x}}') \to \hat{y}'$"
- **Arrow** right to **Recovery?** (diamond): "$\hat{y}' = y$?"
- **Arrow** right to **Output** (rounded orange box): "Recovery Rate ($\text{RR}_{\text{child}}$)"
- Parallel: **Control branch** with gray styling: zero 15 random features instead of child

### Visual Style
- Main pipeline boxes: light blue fill, dark blue border
- False negative / absorption boxes: light red fill
- Activation patching boxes: light orange fill
- Control boxes: light gray fill
- Arrows: solid black, 1pt
- Decision diamonds: white fill, black border
- Font: sans-serif, 9pt
- Overall dimensions: full-width (12cm x 6cm)

### Key Labels
- Above upper branch: "Absorption Measurement Pipeline"
- Above lower branch: "Activation Patching (Causal Validation)"
- Between branches: arrow labeled "For each detected FN"
