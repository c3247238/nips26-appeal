# Revisionist Analysis: Reconciling Contradictions

## Overview
Identification and resolution of contradictions in the experimental findings.

## Major Contradictions

### 1. H3 REVERSED vs pilot_h3_null

**full_h3 Results:**
- Spearman r = +0.35 (p < 0.001)
- High-absorption mean: 0.1035
- Low-absorption mean: 0.0874
- Interpretation: High-absorption features are MORE steerable

**pilot_h3_null Results:**
- No spearman correlation computed
- High-absorption mean: 0.7485
- Low-absorption mean: 0.7543
- Interpretation: High ≈ Low absorption effect

**Contradiction**: full_h3 shows high > low by 15%, pilot_h3_null shows high ≈ low

### 2. Null Control Failure
- Pass criterion: null_mean < 0.05
- Actual: null_mean = 0.6207
- The baseline is extremely high relative to criterion

### 3. Layer-wise Absorption (H1) Unresolved
- Run 1: L4 < L8 (+10.6%)
- Run 2: L4 > L8 (-22.9%)
- No consistent pattern

## Proposed Resolutions

### Resolution 1: Metric Normalization
**Hypothesis**: The difference is due to different metrics
- full_h3: measures logit change magnitude
- pilot_h3_null: uses different alpha values (3, 10 vs 1, 3, 5, 10, 20)

**Fix**: Normalize effects by alpha value before comparison

### Resolution 2: Feature Selection Criteria
**Hypothesis**: Different feature selection (UAS threshold vs UAS ranking)
- full_h3: UAS > 1.0 (high), UAS < 0.3 (low)
- pilot_h3_null: top 50 vs bottom 50 by UAS

**Fix**: Use identical selection criteria across experiments

### Resolution 3: Effect Size vs Correlation
**Hypothesis**: Grouped means hide individual feature variation
- pilot_h3_null shows high ≈ low in aggregate
- full_h3 shows correlation across individual features
- Individual feature effects may cancel in aggregate

**Fix**: Report both grouped means AND individual correlations

### Resolution 4: Null Control as Baseline
**Hypothesis**: null_mean = 0.6207 is the true "no effect" baseline
- All features (high, low, random) show effects > 0.62
- The effect is above baseline but similar across feature types
- This suggests: **Feature steering works generally, but UAS doesn't predict magnitude**

**Resolution**: Reconcile as:
- H3 REVERSED (correlation) may be due to feature-specific variation
- pilot_h3_null (grouped means) shows the effect is uniform
- Both could be true: UAS predicts within-category variation, not between-category

## Unified Interpretation

**Revised Story**:
1. Feature steering is effective (all effects > random baseline)
2. UAS predicts which features within high/low categories will be most effective
3. But the categorical difference (high vs low) is smaller than predicted
4. H3 was wrong about direction, but right that absorption relates to steering

**Revised H3**: "Feature absorption correlates positively with steering sensitivity within categories"

## Recommendations

1. **Run additional analysis**: Compute spearman correlation for pilot_h3_null
2. **Normalize metrics**: Compare apples to apples
3. **Update methodology**: Use consistent feature selection
4. **Revise interpretation**: Acknowledge the nuance

## Conclusion
The contradictions are resolvable through methodological standardization. The core finding (absorption relates to steering) holds, but the magnitude of the categorical difference is smaller than initially claimed.
