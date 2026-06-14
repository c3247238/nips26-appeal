# Paper Outline: Compound Failure Modes in Sparse Autoencoders

## Status
Draft paper based on iteration 6-8 experiments. All hypotheses were falsified in pilot, but negative results are actionable.

## Title
**Compound Failure Modes in Sparse Autoencoders: Absorption and Sensitivity Are Positively Correlated, Not Independent**

## Abstract
Feature absorption (Chanin et al., 2024) and feature sensitivity (Tian et al., 2025) are two documented but independently-studied failure modes in sparse autoencoders. We provide the first joint analysis of these failure modes, examining their correlation and combined effect on steering effectiveness. Contrary to our hypothesis that these failure modes would be independent, we find a significant positive correlation (Spearman r = 0.59, p < 0.001). Features that are absorbed tend to also have low sensitivity. Additionally, we find that absorption level does not predict steering effectiveness (p = 0.299), contradicting the compound failure hypothesis. The predicted best-case quadrant (low absorption + high sensitivity) is empty, suggesting most features experience at least one failure mode.

## Figure & Table Plan

### Introduction
- **Figure 1**: Bar chart comparing steering effects across absorption levels (steering_by_absorption_bar.pdf)
  - X-axis: beta values (1, 3, 5, 10, 20)
  - Y-axis: steering effect (max abs logit difference)
  - Groups: high absorption, low absorption, random baseline
  - Caption: "Steering effects by absorption level across beta values. High-absorption and low-absorption features show equivalent steering sensitivity at all magnitudes."

### Related Work
- No figures required

### Method
- **Table 1**: Feature counts by quadrant (tab:quadrant-counts)
  - Columns: Quadrant, Features
  - Rows: Q1 (doubly-compromised), Q2 (absorbed + sensitive), Q3 (not absorbed + insensitive), Q4 (best-case), Total
  - Caption: "Feature counts by quadrant from pilot classification"

### Experiments
- **Figure 2**: Scatter plot of absorption vs sensitivity (scatter_absorption_sensitivity.pdf)
  - X-axis: Absorption score (UAS)
  - Y-axis: Sensitivity (paraphrase AUC)
  - Points colored by quadrant
  - Caption: "Feature distribution in absorption-sensitivity space. Q1 (doubly-compromised) features cluster in the high-absorption/low-sensitivity region. Q4 (best-case) is empty. The positive correlation (r = 0.59) suggests shared underlying causes."
- **Table 2**: Steering by absorption level (tab:steering-by-absorption) - already referenced in method section
- **Figure 3**: Quadrant model diagram (quadrant_model_desc.md) - architecture diagram
  - Four quadrants with feature counts
  - Shows empty Q2 and Q4

### Discussion
- No new figures required

### Conclusion
- No new figures required

## Section Structure

1. **Introduction** (~1 page)
   - Problem: Sanity Check crisis (random baselines match SAEs)
   - Iter 4: Absorption does not predict steering (p = 0.299)
   - Iter 7: Mutual coherence protective effect ($r = -0.786$) - not replicated
   - Iter 8 pilot: All hypotheses falsified, Q4 empty
   - Research questions and contributions

2. **Related Work** (~1 page)
   - SAEs for interpretability
   - Feature absorption (Chanin 2024)
   - Feature sensitivity (Tian 2025)
   - Sanity Check crisis (Korznikov 2026)
   - Steering interventions

3. **Method** (~1.5 pages)
   - UAS metric
   - Absorption detection protocol (Chanin)
   - Sensitivity measurement protocol (Tian)
   - Steering protocol
   - Quadrant classification
   - Table 1: Feature counts by quadrant

4. **Experiments** (~1.5 pages)
   - H5 (independence): Falsified (r = 0.59)
   - H6 (saturation): Falsified (ratio = 1.0)
   - H1-R (protective): Falsified (r = +0.36)
   - Q4 empty
   - Table 2: Steering by absorption level
   - Figure 2: Scatter plot

5. **Discussion** (~1 page)
   - Implications for steering research
   - Positive correlation suggests common cause
   - Empty Q4 requires theoretical revision
   - Limitations

6. **Conclusion** (~0.5 page)
   - Summary of key findings
   - Implications
   - Future directions

## Key Data Points

| Metric | Value |
|--------|-------|
| Spearman r (absorption, sensitivity) | 0.59 (p = 3.15e-05) |
| Decoder norm ratio (high-abs / low-abs) | 1.0 |
| Spearman r (coherence, absorption) | +0.36 (not replicated: -0.786) |
| Q1 features | 15 |
| Q4 features | 0 |
| Steering p-value (aggregated) | 0.299 |
| Random vs feature steering | p < 10^-12 |

## References

1. Cunningham et al. (2023) - Sparse Autoencoders Find Highly Interpretable Features
2. Bricken et al. (2023) - Towards Monosemanticity
3. Chanin et al. (2024) - A is for Absorption
4. Tian et al. (2025) - Measuring Sparse Autoencoder Feature Sensitivity
5. Korznikov et al. (2026) - Sanity Checks for Sparse Autoencoders
6. Subramani et al. (2022) - Steering interventions
7. Zou et al. (2023) - Steering interventions