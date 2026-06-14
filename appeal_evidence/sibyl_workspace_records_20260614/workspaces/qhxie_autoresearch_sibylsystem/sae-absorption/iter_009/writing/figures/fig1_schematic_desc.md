# Figure 1 Left Panel: Absorption Measurement Schematic

## TikZ Description

A two-path diagram illustrating feature absorption measurement:

### Top Path (Raw Activations -- No Absorption)
1. **Input box**: "Input: 'Saturday is...'" (rounded rectangle)
2. **Arrow** pointing right to:
3. **Residual stream box**: "$\mathbf{x}^{(\ell)}$" (circle, labeled "Raw activation")
4. **Arrow** pointing right to:
5. **Probe box**: "Probe $\mathbf{w}_p$" (diamond shape)
6. **Arrow** pointing right to:
7. **Output box**: "$\hat{y}_{\text{raw}}$ = 'S'" with green checkmark (rounded rectangle, green border)

### Bottom Path (SAE Activations -- Absorption Occurs)
1. Same input box (connected from top)
2. **Arrow** pointing right through:
3. **SAE box**: Large dashed rectangle labeled "SAE" containing:
   - Encoder: "$\mathbf{W}_{\text{enc}}$"
   - Latent space: two feature circles:
     - Feature $z_c$ (child, "Saturday") -- **highlighted red, large activation bar**
     - Feature $z_p$ (parent, "starts with S") -- **grayed out, zero activation bar**
   - Decoder: "$\mathbf{W}_{\text{dec}}$"
   - Reconstruction: "$\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}}$"
4. **Arrow** pointing right to:
5. **Same probe**: "Probe $\mathbf{w}_p$"
6. **Arrow** pointing right to:
7. **Output box**: "$\hat{y}_{\text{SAE}}$ = 'C'" with red X (rounded rectangle, red border)

### Annotation
- A curly brace between the two outputs labeled "**False Negative (Absorption)**"
- Below: "Parent feature suppressed by child feature via competitive exclusion"

### Style
- Clean, minimal design
- Blue (#2196F3) for parent features and correct predictions
- Red (#F44336) for child features and absorption indicators
- Gray for suppressed/inactive elements
- Consistent with paper color palette (style_config.py)
