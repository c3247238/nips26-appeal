# Experiment Critique: Component-Isolated Study of SAE Absorption Reduction

## Executive Summary

The experimental design is sound in principle: component-isolated ablation on ground-truth synthetic data is exactly the right approach for causal attribution. However, critical execution flaws undermine the validity of several comparisons: (1) the sparsity-absorption correlation is based on n=4 with confounded data points; (2) TopK vs MultiScale comparison mixes full-experiment and pilot data; (3) MCC ~0.21 across all variants including Random raises fundamental questions about what absorption reduction means when feature recovery is at chance; (4) the orthogonality hyperparameter was not tuned. The honest reporting of limitations is exemplary, but the gap between provisional data and definitive claims is a serious weakness.

## Critical Issues

### 1. The r ~ -0.97 Correlation is Mathematically Fragile (CRITICAL)

The paper's fourth contribution is "Discovery of a strong absorption--L0 sparsity correlation ($r \approx -0.97$ across $n = 4$ variants)." This correlation is based on four data points:

| Variant | L0 | Absorption | Data Quality |
|:---|---:|---:|:---|
| Baseline | 964 | 0.252 | Full (5 replicates) |
| +TopK | 50 | 0.056 | Full (5 replicates) |
| +Orthogonality | 550 | 0.245 | Full (5 replicates) |
| +MultiScale (pilot) | 50 | 0.050 | Pilot (1 replicate, 1024 features) |

**Two critical problems:**

**Problem A: Two points at identical L0 = 50.** TopK and MultiScale both have L0 = 50. With two points stacked vertically, the correlation is driven by only 3 unique L0 values (50, 550, 964). The correlation coefficient with n=4 and 2 degrees of freedom is mathematically fragile. The bootstrap CI [-1.00, -0.72] does not address this: with so few points, any correlation is unstable.

**Problem B: Incomparable data sources.** TopK is full experiment (5 replicates, 16384 features); MultiScale is pilot (1 replicate, 1024 features). Mixing these in a correlation is methodologically unsound. The pilot used different feature counts, different training duration, and no variance estimate.

**What happens if we remove the pilot point?** With n=3 (Baseline, TopK, Orthogonality), the correlation is still strong but based on only 3 points. Adding Gating and Full Matryoshka (which may have very different L0 values) could dramatically change the correlation.

**Fix**: Downgrade the correlation from a primary contribution to an exploratory observation. Recompute r with only full-experiment points (n=3) and report both values. Add a prominent caveat: "This correlation is based on limited data and requires validation with the full 6-variant set."

### 2. TopK vs MultiScale: Apples-to-Oranges Comparison (CRITICAL)

The paper ranks components as: TopK (d=5.51) > MultiScale (d~1.1) > Orthogonality (d=0.14). But TopK and Orthogonality have full 5-replicate data on 16384 features, while MultiScale has only a single pilot replicate on 1024 features.

This comparison violates basic experimental design principles:
- Different feature counts (1024 vs 16384)
- Different training scales (1M vs 2M tokens)
- Different statistical reliability (1 vs 5 replicates)
- Different variance estimates (none vs std across 5 seeds)

The pilot MultiScale result (75.3% reduction) may not generalize to the full 16k setting. The full MultiScale experiment might show substantially different absorption, MCC, or MSE.

**Fix**: Do not rank MultiScale against TopK until full data is available. Present pilot and full-experiment results in separate tables. Add a warning: "The MultiScale result is from a pilot experiment and is not directly comparable to full-experiment results."

### 3. MCC ~0.21 Across All Variants Including Random (MAJOR)

All variants show MCC ~0.21--0.22, including the Random control (MCC = 0.223). The paper acknowledges this: "MCC is not a strong discriminator in this setup, which is why we designated absorption rate as the primary metric."

But this raises a deeper question: **If the SAEs are not recovering ground-truth features (MCC = chance), what does "absorption reduction" actually mean?**

The absorption metric measures whether parent features are suppressed when child features are present. But if the SAE does not recover the ground-truth feature structure, "parent" and "child" are defined by the ground-truth labels, not by what the SAE actually learns. The absorption reduction could reflect:

- **Genuine learning**: The SAE learns hierarchical structure and TopK helps preserve parent features.
- **Sparsity-induced suppression**: With L0=50, the SAE simply suppresses most features (including parents) globally, not just when children are present.
- **Metric artifact**: The absorption formula may be sensitive to sparsity level in ways that do not reflect genuine hierarchical recovery.

The paper does not distinguish these possibilities. The near-perfect reconstruction for Orthogonality (MSE ~3e-5) suggests the SAEs can reconstruct inputs, but MCC ~0.21 suggests they do so without recovering the ground-truth feature basis.

**Fix**: Add a discussion paragraph addressing this concern. Consider adding a control: measure parent suppression on randomly paired (non-hierarchical) features. If TopK suppresses parents equally on random and hierarchical pairs, the effect is sparsity-induced suppression, not genuine absorption reduction.

### 4. Orthogonality Hyperparameter Not Tuned (MAJOR)

The orthogonality penalty uses $\lambda_{\text{ortho}} = 10^{-3}$ with no tuning. The result: near-perfect reconstruction (MSE ~3e-5) but negligible absorption reduction (2.7%).

An orthogonality penalty that drives MSE to ~3e-5 may have over-regularized the decoder, forcing $W_{\text{dec}}^\top W_{\text{dec}} \approx I$ at the expense of meaningful feature learning. A weaker penalty ($\lambda_{\text{ortho}} = 10^{-4}$ or $10^{-5}$) might achieve different absorption-reconstruction trade-offs.

The paper's conclusion that "orthogonality penalties add compute overhead without absorption benefit" is premature without hyperparameter exploration.

**Fix**: Acknowledge that $\lambda_{\text{ortho}}$ was not tuned. Add to Limitations: "The orthogonality coefficient was not tuned; a weaker penalty might yield different absorption-reconstruction trade-offs."

### 5. Incomplete Variant Set Yet Definitive Conclusions (MAJOR)

The paper lists 6 variants in Table 1 but only 3 have full data. The scope note (Section 1.5) is honest about this, but the Abstract, Conclusion, and Contribution 3 present the component ranking as a definitive finding.

This is a structural mismatch between what was promised and what was delivered. A reader who sees "six SAE variants" in the Abstract and Table 1 expects six results.

**Fix**: Add visual indicators to Table 1 showing data status. Soften definitive claims in the Abstract and Conclusion.

## Moderate Issues

### 6. ANOVA Mentioned but Not Reported

Section 3.5 promises a one-way ANOVA across all completed variants. This is not reported in the Results. With only 3 variants and 5 replicates each, the ANOVA would have 2, 12 df and low power. The paper focuses on pairwise effect sizes instead, which is more appropriate.

**Fix**: Either report the ANOVA (F-statistic, p-value) or remove the promise from Section 3.5.

### 7. L0-Matched Comparison Promised but Not Delivered

Section 3.6 promises "absorption per unit L0 to control for sparsity differences." This control is never reported.

**Fix**: Either compute and report L0-normalized absorption or remove the promise.

### 8. No Training Curves or Convergence Diagnostics

With only ~2000 steps, training instability is a real concern. The plan mentions ghost grads and warm-up, but the paper does not report whether training was stable.

**Fix**: Add a brief statement: "All variants converged within 2000 steps with no dead features (monitored via L0 and ghost gradients)."

### 9. No Individual Replicate Values

The paper reports means and standard deviations but not individual replicate values. With n=5, individual points would add transparency.

**Fix**: Add individual replicate values to a supplementary table or show individual points in Figure 2.

## What Works Well

1. **Ground-truth synthetic data eliminates probe artifacts.** The pivot from real-LLM probe-based metrics to SynthSAEBench-16k is well-justified and well-executed.

2. **Random control validates metric discrimination.** The Random SAE achieves absorption = 0.560, far above trained variants, confirming the metric distinguishes structure from randomness.

3. **Honest reporting of negative results.** H3 is explicitly "NOT SUPPORTED." The incomplete variant set is flagged prominently.

4. **Effect sizes reported with confidence.** Cohen's d values with standard deviations provide clear statistical context.

## Summary Table

| Issue | Severity | Section | Fix |
|:---|:---|:---|:---|
| r~-0.97 correlation, n=4 with 2 points at L0=50 | CRITICAL | 4.6, Abstract | Downgrade to exploratory; recompute with n=3 |
| TopK (full) vs MultiScale (pilot) comparison | CRITICAL | 4.2, 4.4, 6.1 | Separate pilot/full tables; don't rank |
| MCC ~0.21 including Random | MAJOR | 4.9, 5.1 | Add discussion of interpretation |
| Orthogonality lambda not tuned | MAJOR | 3.2, 5.2 | Acknowledge in Limitations |
| 3 of 6 variants, definitive claims | MAJOR | Abstract, 6.1 | Add "(provisional)" qualifier |
| ANOVA mentioned not reported | Moderate | 3.5 | Report or remove |
| L0-matched comparison missing | Moderate | 3.6 | Compute or remove promise |
| No training curves | Moderate | 3.2 | Add convergence statement |
| No individual replicates | Minor | 4.2 | Add supplementary table |
