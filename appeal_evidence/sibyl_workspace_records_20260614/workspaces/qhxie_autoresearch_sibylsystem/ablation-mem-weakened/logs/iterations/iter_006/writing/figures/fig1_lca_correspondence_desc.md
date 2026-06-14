# Figure 1: LCA-SAE Structural Correspondence and Inhibition Graph Construction

## Type
Architecture diagram (TikZ)

## Layout
Three-panel horizontal layout, single column width.

## Panel 1: LCA Dynamics (left, 30% width)
- Title: "LCA Dynamics (Rozell et al., 2008)"
- Equation box: tau * du/dt = -u + (b - G * a), a = T(u)
- Label b = W_enc^T x (feedforward input)
- Label G = W_dec^T W_dec (inhibition matrix)
- Label T(u) = max(0, u) (threshold)
- Visual: neuron icons with lateral inhibition arrows between them
- Inhibition arrows labeled with G_ij

## Panel 2: SAE Architecture (center, 35% width)
- Title: "SAE Forward Pass"
- Input box: a (activation)
- Arrow down to: W_enc * a + b_pre
- Arrow down to: ReLU box
- Arrow down to: z (latents)
- Arrow down to: W_dec * z + b_dec
- Arrow down to: a_hat (reconstruction)
- Right side: label G = W_dec^T W_dec with brace connecting to W_dec
- Highlight box around G = W_dec^T W_dec in red
- Label: "With tied weights: W_enc = W_dec^T, so G = W_enc W_enc^T"

## Panel 3: Inhibition Graph (right, 35% width)
- Title: "Local Inhibition Graph"
- Node i (parent) with edges to top-k neighbors
- Node j (child) highlighted in red
- Edge i->j labeled G_ij
- Edge j->i labeled G_ji
- Text: "For each latent i: N(i) = argmax_{|J|=k} sum_{j in J} |G_ij|"
- Text below: "k = 20--50 neighbors"
- Complexity note: "O(k * d_dict * d_model)"

## Color Scheme
- LCA panel: blue tones (#2E86AB)
- SAE panel: gray boxes with blue highlights
- Graph panel: nodes in light blue, edges in gray, highlight edge in red (#F44336)
- Correspondence arrow (between panels 1 and 2): thick double-headed arrow labeled "Structural Correspondence" in dark blue

## Key Annotation
- Bottom center: "G = W_dec^T W_dec is exactly the LCA inhibition matrix"
- This is the central claim of the figure

## TikZ Elements
- Use tikz-cd for the equation
- Use tikz shapes for SAE boxes
- Use tikz graph library for the inhibition graph
- All text in sans-serif font (\sffamily)
- Font size: small (9pt) for labels, normalsize for titles
