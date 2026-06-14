# Methodologist Analysis: Experimental Rigor Evaluation

## Overview
Systematic evaluation of experimental design quality and methodological improvements.

## Strengths

### Good Practices
- Clear hypothesis specification (H3: r < -0.3)
- Multiple alpha values for dose-response
- Diverse test prompts
- Proper statistical testing (Spearman correlation)

### Statistical Rigor
- P-value < 0.001 provides strong evidence against null
- Confidence intervals should be added

## Weaknesses

### 1. Pilot vs Full Experiment Inconsistency
- full_h3 uses UAS threshold selection (>1.0, <0.3)
- pilot_h3_null uses UAS ranking (top/bottom 50)
- **These are not directly comparable**
- Should use identical feature selection criteria

### 2. Null Control Design Flaw
- Pass criterion: null_mean < 0.05
- Actual: null_mean = 0.6207
- **12x above threshold**—null controls failed
- Random unit vectors are not a good null model
- Better: shuffled decoder weights or orthogonal directions

### 3. Missing Analysis
- No analysis of individual alpha values
- No analysis by feature type/frequency
- No analysis of prompt-specific effects

### 4. Metric Robustness
- max abs logit difference is sensitive to outliers
- Consider: mean, median, or KL divergence
- Consider: per-token vs per-sequence metrics

### 5. Multiple Comparison Problem
- Testing 5 alpha values × 10 prompts × 100 features
- No correction for multiple comparisons
- Family-wise error rate not controlled

## Recommended Statistical Tests

1. **Effect size with confidence intervals** (e.g., 95% CI)
2. **Permutation test** for null hypothesis
3. **Bootstrap confidence intervals** for Spearman r
4. **Bonferroni or Holm correction** for multiple comparisons

## Improved Experiment Design

### Feature Selection
```
# Consistent across all experiments
top_100_by_uas = features ranked by UAS, take top 100
bottom_100_by_uas = features ranked by UAS, take bottom 100
```

### Null Controls
```
# Shuffled: permute feature indices
shuffled_directions = [W_dec[shuffle(perm)] for _ in range(n)]

# Orthogonal: directions orthogonal to all real features
```

### Metrics
```
# More robust
median_logit_diff = np.median(abs(steered - original))
kl_divergence = compute_kl(original, steered)
```

## Conclusion
The experiments have good structure but need tighter null controls and robustness checks before publication.
