# Figure 1: Absorption Mechanism and EDA Geometry -- TikZ Description

## Panel (a): Feature Absorption in Action

Two-column layout showing a Sparse Autoencoder processing the input token "Amsterdam".

**Left column -- Normal operation (no child active):**
- Input: residual stream activation $x$ for token "Amsterdam"
- Encoder fires parent latent $z_p$ ("starts-with-A words") with activation $> 0$
- Reconstruction includes $d_p$ contribution
- Label: "Parent latent active"

**Right column -- Absorption (child active):**
- Same input $x$ for token "Amsterdam"
- Child latent $z_c$ ("starts-with-A proper nouns") fires with high activation
- Parent latent $z_p$ is suppressed to zero (crossed out, red)
- Reconstruction uses $d_c$ instead; parent direction $d_p$ is absent
- Label: "Parent latent absorbed"
- Annotation arrow: "Sparsity pressure suppresses parent when child is active"

**Color scheme:**
- Parent latent/direction: blue
- Child latent/direction: orange
- Suppression indicator: red cross / dashed line
- Activation bars: green (active), gray (inactive)

## Panel (b): EDA and D-EDA Geometry

Vector diagram in $\mathbb{R}^{d_\text{model}}$ (shown as 2D projection).

**Vectors:**
- $d_j$ (decoder direction for latent $j$): solid black arrow pointing right.
- $w_{e,j}$ (encoder direction for latent $j$): solid blue arrow, rotated away from $d_j$.
- The angle $\theta$ between $w_{e,j}$ and $d_j$ is marked with an arc; labeled "EDA$(j) = 1 - \cos\theta$."
- $r_j$ (residual): dashed red arrow from the projection of $w_{e,j}$ onto $d_j$ to the tip of $w_{e,j}$, perpendicular to $d_j$.
- $d_c$ (decoder direction of absorbing child): dotted gray arrow, roughly aligned with $r_j$.

**Layout:** Two sub-panels side by side:
- Left: "Healthy latent" -- $w_{e,j}$ aligned with $d_j$ (small angle, low EDA)
- Right: "Absorbed latent" -- $w_{e,j}$ rotated toward $d_c$, large angle with $d_j$ (high EDA)

**Key takeaway caption:** Absorption manifests as angular divergence between a latent's encoder and decoder directions. EDA measures this angle; D-EDA decomposes the residual to identify which decoder directions absorbed the encoder's alignment.
