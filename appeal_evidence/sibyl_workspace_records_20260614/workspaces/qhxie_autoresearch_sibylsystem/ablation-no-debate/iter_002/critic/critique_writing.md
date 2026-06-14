# Critique: Writing Quality

## Overview
The paper is generally well-written with clear structure, honest negative results, and appropriate acknowledgment of limitations. However, several writing issues undermine clarity, credibility, and precise claims.

## Strengths

1. **Honest negative results**: The steering (H3) and safety (H_Safe) negative results are reported without spin, with specific p-values and effect sizes. This is the paper's strongest aspect.

2. **Limitation acknowledgment**: The paper explicitly discusses five limitations in Section 5.5, including the synthetic-only scope, post-hoc criterion revision, and single-architecture validation. This level of transparency is commendable.

3. **Clear contribution framing**: The six contributions are clearly enumerated and the methodological contribution (factorial decomposition) is genuinely novel.

4. **Good use of examples**: The capacity-pressure mechanism in Section 5.2 is explained with concrete intuition.

## Weaknesses

### 1. Cross-Experiment Metric Mixing in Table 1
**Severity: Major**

Table 1 presents results from experiments using three different absorption metrics (overlap fraction, Jaccard overlap, cosine-based proportional absorption) without clear labeling. The "Key Metric" column does not distinguish which metric was used for which experiment. A reader comparing "Encoder effect 0.843" (overlap fraction) with "Safety 0.967" (cosine-based) might incorrectly conclude these are on the same scale.

**Fix**: Add a "Metric" sub-column to Table 1 distinguishing the three metrics, or split Table 1 into metric-specific tables.

### 2. Figure 9 Referenced but Never Shown
**Severity: Minor**

The figure list at the end references "Figure 9: figure_9_summary_table.pdf" but this figure is never cited in the text and the inline Table 1 serves as the summary. This creates confusion about whether Figure 9 is supplementary or missing.

**Fix**: Either cite Figure 9 in the text or remove it from the figure list.

### 3. Unfulfilled Methodology Promise
**Severity: Major**

Section 3.5 states: "Section 3.5 promises a one-way ANOVA across all completed variants" but this analysis never appears in the Results. The paper focuses on pairwise comparisons, which are more appropriate given the sample sizes, but the unfulfilled promise undermines trust.

**Fix**: Either add the ANOVA or remove the promise from the methodology.

### 4. Overclaiming Novelty
**Severity: Major**

The paper states the core contribution is "first factorial decomposition" and "encoder dominance." However:
- Oursland (2026) already theoretically derives encoder-decoder asymmetry
- The encoder-decoder distinction is implicit in how SAEs are trained and optimized
- SAEBench already compares architectures and notes sparsity differences

The actual novelty is the **empirical quantification** via controlled factorial design, not the underlying observation.

**Fix**: Reframe as "first empirical quantification of encoder-decoder asymmetry via controlled factorial decomposition" to avoid overclaiming.

### 5. Effect Size Inflation from Zero-Variance Baseline
**Severity: Major**

The random baselines in multiseed_validation.json show std=0 across all seeds, meaning the random SAE absorption rate is effectively deterministic. The paper reports Cohen's d > 10 for the trained vs. random comparison, but this is mathematically inflated when one distribution has near-zero variance. The actual separation is impressive but the effect size is not meaningfully d > 10.

**Fix**: Report the effective effect size using the actual variance in the trained distribution, and acknowledge the unusual zero-variance property of the random baseline.

### 6. Abstract vs. Paper Disconnect on Metric Consistency
**Severity: Minor**

The abstract states the encoder effect is "0.843 +/- 0.082" and decoder effect is "0.011 +/- 0.015" without noting these use the overlap fraction metric. A reader who sees "0.967 absorption" in the safety experiment (cosine-based metric) would reasonably expect these to be on comparable scales. Section 3.3 notes the inconsistency but the abstract does not.

**Fix**: Add a parenthetical in the abstract noting metric: "encoder effect 0.843 (overlap fraction)".

## Summary
The writing is clear and the scientific transparency is strong, but the paper would benefit from tighter integration between its methodological promises and actual results, clearer metric labeling in tables, and more precise novelty claims.
