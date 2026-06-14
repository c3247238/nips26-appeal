# Figure 1: Method Pipeline Architecture Diagram

## Description

A flowchart showing the three-branch evaluation pipeline for absorption measurement.

### Layout

**Left column (Input):**
- Box: "Pythia-160M Layer 8 (resid_post, d=768)"
- Arrow down to: "8 SAE Architectures (m=2048)"

**Center column (Three parallel branches):**

Branch 1 (Top): First-Letter Absorption
- Box: "SAEBench Official Evaluator"
- Box: "First-letter hierarchies (e.g., 'starts with S' vs 'short')"
- Box: "Ground-truth logistic probe on h^(8)(x)"
- Box: "K-sparse probe on top-10 latents (tau_fs=0.03)"
- Box: "A_FL = mean_full_absorption_score"

Branch 2 (Middle): Semantic-Hierarchy Absorption
- Box: "Custom probe pipeline"
- Box: "10 WordNet hierarchies (e.g., 'building' vs 'house')"
- Box: "Ground-truth logistic probe on h^(8)(x)"
- Box: "K-sparse probe on top-10 latents (tau_fs=0.03)"
- Box: "A_SH = max(0, (acc_resid - acc_sae)/acc_resid, (acc_resid - acc_ksparse)/acc_resid)"

Branch 3 (Bottom): Non-Hierarchy Control
- Box: "Custom probe pipeline"
- Box: "10 correlated non-hierarchical pairs (e.g., 'big' vs 'large')"
- Box: "Ground-truth logistic probe on h^(8)(x)"
- Box: "K-sparse probe on top-10 latents"
- Box: "A_NH = same absorption formula"

**Right column (Output):**
- Three boxes stacked vertically:
  - "A_FL: First-letter absorption [0,1]"
  - "A_SH: Semantic-hierarchy absorption [0,1]"
  - "A_NH: Non-hierarchy control absorption [0,1]"
- Arrow down to: "Statistical tests: Pearson r, paired t-test, bootstrap CI"

### Style Notes
- Use TikZ with rectangular nodes, rounded corners
- Branch boxes in light blue, light orange, light green respectively
- Arrows show data flow from left to right
- Dashed box around the three branches labeled "Same absorption formula applied to all three conditions"
- Random-SAE control shown as a small inset box: "Control: permuted W_dec from Standard SAE"
