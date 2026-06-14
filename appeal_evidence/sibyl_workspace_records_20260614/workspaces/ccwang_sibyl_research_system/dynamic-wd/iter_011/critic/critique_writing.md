# Writing Critique — Iteration 11

## Overall Assessment: MINOR REVISION REQUIRED

The writing is clear, well-structured, and professionally presented. The main issues are figure cross-reference errors, misleading abstract framing, and some overclaiming.

## Major Issues

### 1. Figure 6 Cross-Reference Error (Major)
Section 5.6 text references:
- "Figure 6a" → AdamW weight norms bar chart (correct, matches Figure 6 panel a)
- "Figure 6b" → "full training trajectory" (WRONG — Figure 6(b) is the SGD bar chart; the trajectory is Figure 7)
- "Figure 6c" → "Under SGD, the picture changes dramatically" (WRONG — Figure 6 has only panels a and b)

The text treats two separate figures (Figure 6: bar charts, Figure 7: trajectories) as three panels of one figure. This will confuse readers.

**Fix**: Replace "Figure 6b" with "Figure 7" for the trajectory reference. Replace "Figure 6c" with "Figure 6(b)" for the SGD norms reference.

### 2. Abstract Misleads About Tested Methods (Major)
The abstract says: "Weight Norm Control (AdamWN), and AlphaDecay---each reporting improvements under different conditions." This implies these methods are included in the study. They appear in Table 1 but are never empirically tested. A reader finishing the abstract expects a 10-method comparison and finds a 7-method one with 3 trivial controls.

**Fix**: Rewrite abstract to distinguish framework-recoverable methods from empirically-tested methods.

## Minor Issues

### 3. Unreferenced Figure (certified_band.png)
This file exists in the figures directory but is never cited. It also contains "PMP-WD" in the legend — a method removed in this iteration. Should be deleted.

### 4. Figure Filename Mismatch
| Text Figure # | Filename | Expected |
|---|---|---|
| Figure 4 | fig3_bem_vs_accuracy.png | fig4_... |
| Figure 5 | fig4_diagnostic_heatmap.png | fig5_... |
| Figure 7 | fig5_weight_norm_trajectories.png | fig7_... |

This won't affect the markdown but will cause confusion during LaTeX conversion.

### 5. Overclaiming Without Qualification
Several places use absolute language that should be conditioned on the experimental scope:
- "the choice of weight decay schedule does not matter" (Section 6.3) — add "under the tested conditions"
- "all dynamic weight decay variants are statistically equivalent" (multiple places) — add "on CIFAR-scale experiments"
- Title says "When" (good), but body doesn't consistently match this conditional framing

### 6. Missing Reproducibility Statement
NeurIPS requires a reproducibility statement. Include: code URL, hardware (GPU model), total compute (GPU-hours), random seed handling, software versions.

### 7. Figure 5 (Heatmap) Caption
Caption says "AIS is remarkably uniform across methods (0.336--0.410)" but the heatmap shows AIS values of 0.336, 0.352, 0.359, 0.410, 0.343, 0.368, 0.360. The range is correct, but "remarkably uniform" is subjective — the max/min ratio is 1.22x. Let the numbers speak for themselves.

### 8. Minor Notation
- Section 3.4 AIS formula uses $\rho_S$ (Spearman) but the AIS range is stated as [0,1]. Spearman correlation ranges [-1,1]. The paper presumably takes the absolute value but doesn't state this explicitly.
- The modulator $\varphi$ is defined as non-negative ($\mathbb{R}^d_{\geq 0}$), but the normalization convention says $\mathbb{E}[\varphi] = 1$. For the no-WD case, $\varphi = 0$, which violates the normalization. The paper should note this is an exception.

## Strengths
- Clear section structure following standard ML paper format
- Good use of paired statistical tests (rare in ML papers)
- Honest limitations section that anticipates reviewer concerns
- Effective use of the "boundary condition" narrative (AdamW vs SGD)
- Table formatting is clean and consistent
- Good paper roadmap in Section 1.4
