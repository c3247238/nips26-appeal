# Figure 7: 2x2 Factorial Design Architecture Diagram

## TikZ Description

Create a 2x2 grid diagram with the following structure:

### Grid Layout
- Two rows (Encoder: Random, Trained)
- Two columns (Decoder: Random, Trained)
- Four labeled quadrants: A (top-left), B (top-right), C (bottom-left), D (bottom-right)

### Quadrant Contents
Each quadrant shows a simplified SAE architecture:
- Input node (circle) labeled "x" on the left
- Encoder box (rectangle) in the middle-left
- Sparse feature nodes (small circles) in the center, arranged vertically
  - Parent feature P (top, colored blue)
  - Child features C1, C2 (middle, colored orange)
  - Grandchild features G1-G4 (bottom, colored gray)
- Decoder box (rectangle) in the middle-right
- Output node (circle) labeled "x̂" on the right

### Quadrant-Specific Visual Encoding

**Quadrant A (Random Encoder, Random Decoder)**:
- Encoder box: light gray fill, label "Random"
- Decoder box: light gray fill, label "Random"
- Absorption pathway: faint dashed arrow from P to C1/C2
- Label: "α(A) ≈ 0.03 (baseline)"

**Quadrant B (Trained Encoder, Random Decoder)**:
- Encoder box: blue fill, label "Trained"
- Decoder box: light gray fill, label "Random"
- Absorption pathway: thick solid arrow from P to C1/C2 (strong absorption)
- Label: "α(B) ≈ 0.86 (encoder drives absorption)"

**Quadrant C (Random Encoder, Trained Decoder)**:
- Encoder box: light gray fill, label "Random"
- Decoder box: green fill, label "Trained"
- Absorption pathway: faint dashed arrow from P to C1/C2 (no absorption)
- Label: "α(C) ≈ 0.03 (decoder alone insufficient)"

**Quadrant D (Trained Encoder, Trained Decoder)**:
- Encoder box: blue fill, label "Trained"
- Decoder box: green fill, label "Trained"
- Absorption pathway: solid arrow from P to C1/C2, but thinner than B (decoder disentanglement)
- Label: "α(D) ≈ 0.44 (full training, decoder reduces absorption)"

### Formula Bar
Below the grid, display the decomposition formula:
```
α(D) = α(A) + E_enc + E_dec + E_int
       baseline   encoder    decoder   interaction
                  effect     effect
```

With values:
- α(A) = 0.018
- E_enc = 0.843
- E_dec = 0.011
- E_int = -0.428 (decoder disentanglement)

### Annotations
- Large curly brace on the right side grouping B and D, labeled "Encoder Effect = 0.843"
- Large curly brace on the bottom grouping C and D, labeled "Decoder Effect = 0.011"
- Arrow pointing from B to D with label "Decoder disentanglement"

### Style
- Use clean, minimal lines
- Box sizes: encoder/decoder boxes 1.2cm x 0.8cm
- Feature circles: 0.3cm diameter
- Arrow thickness: 1pt for faint, 2pt for strong
- Font: sans-serif, small (\small)
- Colors: blue (#4472C4) for trained encoder, green (#70AD47) for trained decoder, orange (#ED7D31) for child features, gray (#A5A5A5) for random/untrained
