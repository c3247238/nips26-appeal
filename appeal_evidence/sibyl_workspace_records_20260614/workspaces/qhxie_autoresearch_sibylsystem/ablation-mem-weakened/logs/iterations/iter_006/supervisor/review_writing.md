# Supervisor Review: Feature Absorption as Optimal Compression

## Overall Assessment

**Score: 5.5 (Borderline Reject)**

The paper presents an interesting theoretical connection between the Locally Competitive Algorithm (LCA) from neuroscience and SAE feature absorption, showing that the decoder correlation matrix `G = W_dec^T W_dec` is exactly the LCA inhibition matrix for tied-weight SAEs. However, the central predictive tool (the local inhibition graph) fails completely -- precision@20 = 0.0 with p = 1.0 Fisher exact test across 520 predictions. The empirical core of the paper is predominantly null results, honestly reported, but the paper repeatedly highlights the one uncorrected p=0.028 result as evidence throughout the abstract and introduction despite zero significant results after multiple comparison correction (MCP). This is a critical flaw that undermines the paper's credibility.

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 6 | The LCA-SAE connection is theoretically interesting but not empirically validated. The G=W^T W correspondence predates this work by 16 years (Rozell et al. 2008). The paper's contribution is in applying the framework to SAEs, but without successful graph-based prediction, the novelty value is unclear. |
| **Soundness** | 5 | The LCA structural correspondence is mathematically correct. However, the paper's central empirical claim -- that the inhibition graph predicts absorption pairs -- is falsified (precision@20=0.0). The mechanistic framework is undermined by its complete empirical failure. |
| **Experiments** | 5 | 12 statistical tests were performed with proper MCP applied. However, zero results survive correction. The MCC~0.21 on random baseline is a critical validity threat. The H7 trained-vs-random comparison is undermined if the metric itself fails on random baselines. |
| **Reproducibility** | 5 | Random SAE baseline construction is underspecified. Code and data are not publicly available. Prompts for steering experiments are not provided. |

## Critical Issues

### 1. Presenting Non-Significant Results as Evidence

The paper performs 12 statistical tests. After Bonferroni correction (alpha = 0.00417), **zero tests reach significance**. Yet the abstract, introduction, and conclusion all present the uncorrected H1b result (p=0.028 at layer 8) as evidence of a real effect. This is factually incorrect. The correct framing is: "We found no statistically significant evidence that absorption degrades steering effectiveness after multiple comparison correction."

**Evidence from raw data** (`correlation_report_full.md`):
- H1b_L8_Pearson: raw p=0.0278, Bonferroni p=0.3339, BH q=0.167
- H1b_L8_Spearman: raw p=0.0089, Bonferroni p=0.1068, BH q=0.107
- Zero significant results after MCP across all 12 tests

### 2. Metric Validity Threat

The h6_inhibition_graph.json shows MCC~0.21 for the random SAE baseline. This means Hungarian matching yields chance-level recovery regardless of whether the SAE is trained. If the metric fails on random baselines, the paper's central comparison (trained SAE mean=0.034 vs random SAE mean=0.278) may reflect metric sensitivity to dictionary structure rather than genuine absorption differences.

The paper acknowledges this failure in passing but does not adequately address the implication: if the metric is not valid, the contribution evaporates.

### 3. Logical Incoherence: LCA Framing vs. H6 Failure

The paper's LCA framework (Section 3) claims that decoder correlations explain absorption as competitive suppression. The primary empirical test of this framework is H6: does the inhibition graph predict absorption pairs? The answer is definitively **no**: precision@20=0.0, p=1.0, enrichment=0.0x across 520 predictions.

Yet the paper continues to rely on the LCA framework throughout, claiming the mechanism is "supported" by the precision-recall asymmetry (H7). This is logically incoherent -- a framework whose primary predictive test is falsified cannot be considered supported by a separate empirical observation.

The paper's defense is that "the mechanistic framework is supported even when the predictive tool is not." This defense is unconvincing. Either the LCA framework makes testable predictions (in which case H6's failure is fatal) or it does not (in which case the paper is not doing science).

## Major Issues

### 4. Beta-Conditional Effects Not Acknowledged

The absorption-sensitivity correlation is only observable at high steering magnitudes (beta=10, 20). At beta=5 (median), no difference is observed. The paper presents the delta-corrected L8 correlation (r=-0.431, p=0.028) as evidence of a robust finding without acknowledging this conditional nature.

### 5. Post-Hoc Power Analysis

Section 3.6 includes a post-hoc power analysis stating "approximately 20% power to detect a medium effect size." This is methodologically inappropriate -- power should be computed before the experiment, not after observing null results.

### 6. Unmatched OrtSAE Ablation

The OrtSAE comparison (Section 4.5) compares unmatched L0 values: 0.230 at L0~920 vs 0.247 at L0~550. This is the very confound the paper criticizes in other contexts. The ablation conclusion "orthogonality penalty does not appear to reduce absorption" is self-contradictory given the L0 mismatch.

### 7. Underspecified Random Baseline

The random SAE construction ("frozen orthonormal decoder, random encoder") does not specify: encoder initialization distribution, whether the encoder was trained before freezing, or orthonormalization method (QR vs SVD). Different implementations could yield different absorption rates.

## Recommendations

1. **In the abstract**: State explicitly "After multiple comparison correction across 12 tests, no hypothesis reached statistical significance." Remove all language implying the p=0.028 result is evidence of a real effect.

2. **On metric validity**: Add explicit analysis of whether the trained-vs-random difference reflects absorption reduction or metric sensitivity. Compute partial correlation controlling for dictionary geometry metrics.

3. **On LCA framing**: Either drop the LCA connection entirely (refocus as a metric validation paper with honest null results) or provide concrete evidence beyond the failed H6 for why the mechanism exists.

4. **On reproducibility**: Release code, specify random SAE construction in detail, provide steering prompts.

5. **On ablations**: Match L0 values in any architectural comparison, or acknowledge this as a limitation.

## Verdict

**Revise.** The paper has some merit (honest null-result reporting, theoretically interesting LCA connection) but critical flaws (presenting non-significant results as evidence, metric validity threat, logical incoherence between theory and empirical failure) make it not ready for acceptance in its current form. A rigorous revision addressing these issues could produce a stronger paper, but the current version overclaims given the evidence.
