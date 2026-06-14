# Supervisor Review — Iteration 10

**Score: 6.5 / 10** | **Verdict: continue**

## Dimension Scores

| Dimension | Score | Rationale |
|-----------|:-----:|-----------|
| Novelty | 7 | Phi Modulator Framework is a genuine contribution. BEM and AIS are useful diagnostics. The null-result framing (Phi Invariance Conjecture) is well-articulated and falsifiable. |
| Soundness | 6 | Major figure-table consistency fixes landed (Figures 2, 3, 4). One residual: Figure 5 heatmap shows half_lambda BEM=0.000, contradicting Table 6 (BEM=0.500). CSI's predictive failure undermines its contribution claim. |
| Experiments | 6 | 105 experiments is solid for CIFAR scale. SGD boundary condition is compelling evidence. But: no ImageNet (5+ iterations unfixed), N=3 is underpowered for null-result claims, NoBN data unused. |
| Reproducibility | 7 | Training details are thorough: hyperparameters, seeds, augmentation, optimizer configs all documented. Unified codebase with pluggable modulator interface. Missing: code release mention. |

## Executive Summary

The paper has materially improved since iter_009. The three critical figure-table-text consistency issues that warranted a score decrease last iteration have ALL been fixed:

- **Figure 3**: Now shows correct 7 methods, "Spread: 0.25pp" (matching text and Table 2). No PMP-WD contamination.
- **Figure 4 (BEM scatter)**: half_lambda correctly positioned at BEM ~0.5.
- **Figure 2**: Numbers match Table 2 exactly (verified against all 42 raw summary.json files).

Cross-validation against raw experimental data (iter_003 for AdamW/SGD ResNet-20, iter_005 for VGG-16-BN) confirms all paper tables are accurate.

The score rises from 6.0 to 6.5 because the data integrity crisis is mostly resolved. However, reaching 7.0+ requires addressing the remaining experimental gaps.

## Critical Issues (2)

### 1. Figure 5 Heatmap BEM Bug (Residual Data Integrity)
Figure 5 (fig4_diagnostic_heatmap.png) shows half_lambda BEM = 0.000 while Table 6 reports BEM = 0.500. The scatter plot (Figure 4) correctly shows 0.5, proving the heatmap was not regenerated from the same data. This is the same class of bug that plagued Figures 3 and 4 in prior iterations.

**Fix**: Regenerate the heatmap from Table 6 data. ~10 minutes.

### 2. No ImageNet Experiments
Flagged in 5+ consecutive iterations. Project constraints mandate ImageNet. All comparable papers (CWD, SWD, AlphaDecay) include ImageNet or larger. CIFAR-10/100 are insufficient for claims about "modern deep learning."

**Fix**: Diagnose failure root cause first. Then run ImageNet-100 (100 classes) with 3 key methods x 3 seeds = 9 runs.

## Major Issues (5)

### 3. Statistical Power for Null-Result Claims
N=3 seeds provides ~15-20% power for 0.5% effects. TOST passes for only 6/12 comparisons at delta=1.0%. The paper's headline "all methods equivalent" is stronger than the evidence supports.

### 4. NoBN Data Not Integrated
iter_005 contains NoBN ablation data (constant: 87.74%, CWD: 87.62%, spread 0.12pp). Section 6.2 hypothesizes BN enables Phi Invariance under SGD. This data directly tests the hypothesis at zero compute cost.

### 5. CSI as a Contribution
CSI has arbitrary weights, fails to predict accuracy (rho<0.3, p>0.3). Listed as a contribution in Section 1.3 but provides zero predictive value. Should be demoted to "exploratory diagnostic."

### 6. Section 5.7 Directional Conclusions from Non-Significant Results
Both alignment correlations are non-significant (p=0.121, p=0.858). Drawing directional conclusions ("cumulative may be more informative than worst-case") from two non-significant p-values is statistically inappropriate.

### 7. Abstract Implies Full Factorial Design
"7 methods, 2 optimizers, 2 architectures, 2 datasets" implies 168 runs. Actual: 84 (ResNet-20 full) + 21 (VGG-16-BN, SGD/CIFAR-10 only) = 105. Misleading.

## Minor Issues (2)

### 8. Title Causal Claim
"Why" implies proven causal mechanism. Paper shows observation + hypothesis. "When" is more accurate.

### 9. Proposition 1
"Product of non-negative functions is non-negative" does not merit proposition status.

## What Would Raise the Score

| Action | Cost | Score Impact |
|--------|------|:----------:|
| Fix Figure 5 heatmap | 10 min | 6.5 (clean data integrity) |
| ImageNet-100: 3 methods x 3 seeds | ~8 GPU-hrs | +1.0 (to ~7.5) |
| Integrate NoBN data | 0 compute | +0.25 |
| Add 2 seeds for 4 key comparisons | ~2 GPU-hrs | +0.25 |
| **Total** | **~10 GPU-hrs + 2 hrs work** | **~8.0** |

## Positive Observations

- **Figure consistency is vastly improved.** This was the primary blocker and it is largely resolved.
- **Statistical methodology remains above community norm.** Paired t-tests with Bonferroni, Cohen's d, TOST, power analysis.
- **The Phi taxonomy is the paper's strongest structural contribution.** CWD/random-mask equivalence is genuinely insightful.
- **SGD boundary condition experiment is compelling.** The AdamW-vs-SGD weight norm comparison (1.2% vs 97%) is a strong mechanistic argument.
- **Paper text is internally consistent with tables.** Spread numbers, p-values, and accuracy numbers all match within the text and tables (verified against raw data).
