# Figure 1: Lotka-Volterra Competition Framework for SAE Feature Absorption

## Layout
Two-panel horizontal diagram with a central mathematical mapping bridge.

## Left Panel: Ecological Niche Overlap
- Two overlapping ellipses representing species niches in resource space.
- Species A (rarer, smaller ellipse, dashed border) labeled "Rare general species."
- Species B (more frequent, larger ellipse, solid border) labeled "Frequent specialist species."
- Overlap region shaded, labeled "$\sigma_{ij}$: niche overlap."
- Arrow from B into A's niche with annotation: "When $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i) > 1$, species A is competitively excluded."

## Center Bridge: Mathematical Mapping
Vertical alignment of three correspondences:
- $\sigma_{ij}$ (normalized co-activation) $\longleftrightarrow$ niche overlap
- $f_j / f_i$ (frequency ratio) $\longleftrightarrow$ carrying capacity ratio
- $\alpha_{ij} > 1$ (competition coefficient) $\longleftrightarrow$ competitive exclusion / absorption

## Right Panel: SAE Decoder Space
- A 2D projection of the SAE decoder space (e.g., first two principal components of decoder columns).
- Parent latent $\mathbf{d}_P$ (e.g., "first letter = A") shown as a thin arrow, labeled "Parent (rare, general)."
- Child latent $\mathbf{d}_C$ (e.g., "the token 'April'") shown as a thick arrow at a small angle to $\mathbf{d}_P$, labeled "Child (frequent, specific)."
- Decoder cosine similarity $\cos(\mathbf{d}_P, \mathbf{d}_C) > 0.15$ indicated by the small angle.
- A faded/ghost version of $\mathbf{d}_P$ to indicate suppression when $\alpha_{PC} > 1$.
- Annotation: "Parent latent suppressed: information captured by child."

## Bottom Strip: Pipeline Flow
Left-to-right flow:
1. "Collect activation statistics ($f_i$, co-activation rates)" 
2. "$\to$ Compute $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$"
3. "$\to$ Predict absorption if $\max_j \alpha_{ij} > \tau$"

## Caption
**Figure 1.** Mapping from ecological competitive exclusion to SAE feature absorption. Left: two species with overlapping niches; the rarer species is excluded when the competition coefficient exceeds unity. Right: in SAE decoder space, a rare parent latent (e.g., "first letter = A") is suppressed by a frequent child latent (e.g., "April") that occupies a similar decoder direction. The competition coefficient $\alpha_{ij}$ formalizes this analogy using co-activation statistics and activation frequencies, requiring no pre-specified probe directions. Our experiments show that the sharp threshold prediction ($\alpha_{ij} \approx 1$) does not hold (Section 5.1), but the conceptual decomposition into niche overlap and frequency imbalance captures the relevant factors.

## Generation Method
TikZ or manual illustration. No experimental data required.
