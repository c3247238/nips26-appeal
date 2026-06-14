# Planning Critique

## Overall Assessment

No formal methodology.md or plan document was found in the workspace. The planning appears to have been done implicitly during ideation, with the final experimental design emerging from the empiricist perspective's Candidate B. This lack of a formal plan makes it difficult to evaluate whether the experimental design was carefully considered or ad hoc.

## Issues Identified

### 1. Missing Formal Plan

The plan/ directory is empty. There is no:
- Formal methodology document with pre-registered hypotheses and analysis plan
- Power analysis documenting the rationale for n=26 features
- Pre-registered falsification criteria beyond the simple p<0.05 threshold
- Sample size justification

The paper's hypotheses (H1-H3) are well-defined with directional predictions, but the planning behind the experimental parameters (26 features, 100 samples per feature, 6 steering strengths, 4 sparsity levels) is not documented.

**Fix:** For future work, create a formal plan document with pre-registered hypotheses, power analysis, and analysis plan before conducting experiments.

### 2. Power Analysis Gap

The paper mentions that n=26 provides ~65% power for |r|>=0.50, but this power analysis appears to have been done post-hoc (it appears in the Results section, not the Methodology). A pre-registered power analysis would have revealed that:
- To detect r=-0.30 (the observed layer 8 trend) with 80% power requires n~84
- To detect r=-0.10 (a small but potentially meaningful effect) requires n~782
- The restricted absorption variance (most features near zero) further reduces effective power

If this power analysis had been done before the experiment, the researchers might have:
- Increased the feature set (e.g., include digits 0-9, or semantic features)
- Accepted that only large effects would be detectable and framed the study accordingly
- Added more layers to increase the total sample size for H3

### 3. Missing Controls in the Design

The planning did not include:
- **Random feature baseline** for steering (flagged as a limitation but not planned)
- **Alternative absorption metric** validation (flagged as future work but not planned)
- **Semantic hierarchy features** (only first-letter features were used)
- **Cross-model validation** (only GPT-2 Small)

These omissions appear to be resource constraints rather than deliberate design choices, but they are not documented as such.

### 4. Steering Strength Selection

The steering strengths (s in {1.0, 2.0, 5.0, 10.0, 20.0, 50.0}) span two orders of magnitude. The rationale for this specific set is not explained. Why not include s=100? Why include s=1.0 and 2.0 if they produce near-zero success rates? The dose-response curves (Figure 3) show success increasing monotonically, suggesting that the lower strengths add little information.

**Fix:** Document the rationale for steering strength selection. Consider whether fewer strengths (e.g., {10, 20, 50}) would have been sufficient, freeing up computational budget for other analyses.

### 5. Sparsity Level Selection

The sparsity levels (k in {1, 5, 10, 20}) are tested but only k=5 is used for the primary analysis. The rationale for k=5 as the primary point is stated ("sparse enough to isolate individual feature contributions but rich enough to capture correlated latents"), but no sensitivity analysis is provided showing whether results differ at other k values.

**Fix:** Add a brief analysis showing correlation results at all four k values. If results are consistent across k, this strengthens the null finding. If they differ, this is an interesting result.

### 6. Layer Selection Rationale

Layers 0, 4, 8, and 10 were selected, but only layers 4 and 8 were used for steering and probing. The rationale for this subset is stated ("representative mid-network layers"), but:
- Why not layer 6 (midway between 4 and 8)?
- Why not include layer 10 steering to test whether the unembedding confound affects results?
- Layer 0 absorption data is collected but never used in any analysis

**Fix:** Use layer 0 and 10 absorption data for H3 (consistency testing) even if steering/probing were not conducted there. This would increase the n for H3 from 2 to 4 layers.

## Positive Aspects

1. **Training-free design** is well-justified and makes the study accessible
2. **Four-phase pipeline** is logically structured and replicable
3. **Fixed random seed** ensures reproducibility
4. **Pre-trained SAEs** from public releases reduce experimental variance
5. **Honest negative result reporting** is methodologically sound

## Score: 5/10

The absence of a formal plan document and the apparent lack of pre-registered power analysis are significant weaknesses. The experimental parameters seem chosen for feasibility rather than statistical rigor.
