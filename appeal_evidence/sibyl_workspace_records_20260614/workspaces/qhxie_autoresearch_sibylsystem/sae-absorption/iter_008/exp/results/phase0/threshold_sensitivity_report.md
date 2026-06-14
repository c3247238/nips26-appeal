# Threshold Sensitivity Report

**Generated:** 2026-04-15 23:43
**Verdict:** STRUCTURAL (confidence: HIGH)

## Summary

The control failure (false negatives after SAE encoding) is structural, not threshold-dependent. Thresholds only affect classification of existing failures, not the failures themselves.

## Data Sources

| Source | Description | Grid Size |
|--------|-------------|-----------|
| iter_001 | Subtype taxonomy sensitivity | 5 thresholds x 2 SAE configs |
| iter_006 | Absorption measurement sensitivity | 5x4 (cosine x magnitude gap) |

## Absorption Rate Heatmap (iter_006)

Gemma Scope L12-16k, Gemma 2 2B, n=577 words, 25 letter probes.

| gap \ cos | 0.01 | 0.02 | 0.025 | 0.03 | 0.05 |
|----------|------|------|-------|------|------|
| 0.5 | 15.1% | 15.1% | 15.1% | 15.1% | 15.1% |
| 1.0 | 15.1% | 14.6% | 14.6% | 14.6% | 14.1% |
| 1.5 | 15.1% | 14.2% | 13.5% | 13.5% | 13.2% |
| 2.0 | 15.1% | 13.2% | 12.2% | 12.2% | 11.8% |

**Range:** 11.8% to 15.1% (CV = 0.077)
**False negatives:** Constant at n=87/576 across all 20 cells
**Monotonicity:** Cosine 100%, Gap 100%

## Subtype Taxonomy Stability (iter_001)

Classification of absorbed latents into early/late/partial subtypes.

### L12-16k (n=16 absorbed latents)

| Threshold | Early% | Late% | Partial% |
|-----------|--------|-------|----------|
| 0.2 | 75.0 | 12.5 | 12.5 |
| 0.25 | 75.0 | 12.5 | 12.5 |
| 0.3 | 75.0 | 12.5 | 12.5 |
| 0.35 | 75.0 | 12.5 | 12.5 |
| 0.4 | 93.8 | 6.2 | 0.0 |

### L12-65k (n=65 absorbed latents)

| Threshold | Early% | Late% | Partial% |
|-----------|--------|-------|----------|
| 0.2 | 32.3 | 33.8 | 33.8 |
| 0.25 | 55.4 | 23.1 | 21.5 |
| 0.3 | 72.3 | 13.8 | 13.8 |
| 0.35 | 81.5 | 9.2 | 9.2 |
| 0.4 | 95.4 | 3.1 | 1.5 |


## Conclusion

**STRUCTURAL**: The control failure (false negatives after SAE encoding) is structural, not threshold-dependent. Thresholds only affect classification of existing failures, not the failures themselves.

### Key Evidence

1. False negatives are CONSTANT (n=87) across all 20 grid cells. Varying cosine threshold (0.01-0.05) and magnitude gap (0.5-2.0) does not change how many tokens the probe misclassifies after SAE encoding.
2. Absorption rate CV = 0.077 (< 0.10), classified as STABLE. Rate varies only from 11.8% to 15.1% across 5x4 grid.
3. Maximum absorption reduction from loosest to strictest thresholds: 0.0329 (21.8% relative). Even at the strictest settings (cos=0.05, gap=2.0), absorption remains at 11.8% -- far from zero.
4. Probe quality is the primary driver of false negatives (rho=-0.756, p=0.0000). Letters with low probe F1 have high FN rates regardless of threshold.
5. L12-16k subtype taxonomy is stable: early=[75.0, 75.0, 75.0, 75.0, 93.8] across thresholds (constant 75% at thresholds 0.2-0.35, jumps only at 0.40). The taxonomy is robust to threshold choice for all but the most aggressive threshold.
6. Despite L12-65k subtype shifts, Kruskal-Wallis significance holds in 4/5 thresholds and late>early ordering holds in 5/5 thresholds.

### Implications for Paper

- Absorption is an inherent property of the SAE's learned representation, not an artifact of detection thresholds.
- Improving thresholds cannot eliminate false negatives -- only architectural changes or training modifications can.
- For JumpReLU SAEs specifically: the low absorption rate (~15%) at layer 12 is genuine, not a threshold artifact.
- Probe quality (F1) is a stronger predictor of false negative rate than any absorption detection threshold.
- The 5x4 grid analysis confirms the metric is robust (CV=0.079) and supports cross-threshold comparisons.