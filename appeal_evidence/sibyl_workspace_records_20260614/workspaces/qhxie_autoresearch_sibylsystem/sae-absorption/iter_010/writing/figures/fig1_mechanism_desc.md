# Figure 1 Left Panel: Absorption Mechanism Schematic

## Description for TikZ rendering

The left panel is a schematic diagram showing how feature absorption is measured.
It contains two parallel pathways:

### Top pathway (Raw activations -- correct prediction)
1. Input token "Paris" enters the model
2. Raw activation vector x at layer 24 is extracted
3. Linear probe (continent classifier) is applied: probe(x) = "Europe" (CORRECT, green checkmark)

### Bottom pathway (SAE-reconstructed activations -- incorrect prediction)
1. Same input token "Paris"
2. Raw activation x is encoded through SAE: z = SAE_enc(x)
3. Child feature z_Paris fires strongly (highlighted in orange)
4. Parent feature z_Europe does NOT fire (highlighted in red, crossed out)
5. SAE reconstruction: x_hat = SAE_dec(z)
6. Same linear probe applied: probe(x_hat) = "Asia" (WRONG, red X)

### Key visual elements
- Arrow from child feature z_Paris to parent feature z_Europe with "competitive exclusion" label
- Color coding: green for correct, red for incorrect/absent
- SAE latent space shown as a sparse vector with most entries zero
- The false negative (FN) is the absorption event

### Annotations
- "Absorption = probe correct on raw, wrong on SAE"
- "Child feature suppresses parent via encoder competition"
