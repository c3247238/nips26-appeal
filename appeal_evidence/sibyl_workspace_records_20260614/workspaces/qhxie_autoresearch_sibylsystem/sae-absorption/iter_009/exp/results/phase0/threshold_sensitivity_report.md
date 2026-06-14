# Phase 0.4: Threshold Sensitivity Report

**Task:** phase0_threshold_reporting
**Mode:** FULL (CPU-only analysis)
**Timestamp:** 2026-04-16T03:12:24.087667
**Elapsed:** 0.3s

## Verdict: STRUCTURAL

The control failure (false negatives after SAE encoding) is structural, not threshold-dependent. Thresholds only affect classification of existing failures, not the failures themselves.

## 1. Absorption Rate Grid (5x4)

**Model:** gemma-2-2b | **SAE:** layer_12/width_16k/average_l0_82
**Words:** 577 | **Letters with probes:** 25

| Mag. Gap \ Cos. Thresh. | 0.01 | 0.02 | 0.025 | 0.03 | 0.05 |
|---|---|---|---|---|---|
| 0.5 | 0.1510 | 0.1510 | 0.1510 | 0.1510 | 0.1510 |
| 1.0 | 0.1510 | 0.1458 | 0.1458 | 0.1458 | 0.1406 |
| 1.5 | 0.1510 | 0.1424 | 0.1354 | 0.1354 | 0.1319 |
| 2.0 | 0.1510 | 0.1319 | 0.1215 | 0.1215 | 0.1181 |

**Statistics:** mean=0.1412, std=0.0109, CV=0.0773, range=[0.1181, 0.1510]
**Range as % of max:** 21.8%

### False Negative Analysis
- FN count: **87/576** (15.1%)
- Constant across grid: **True**
- Interpretation: False negatives are FIXED (87/576) regardless of absorption detection thresholds. Thresholds only affect whether an FN is classified as 'absorbed' vs 'unabsorbed'. This means the control failure is STRUCTURAL, not threshold-dependent.

### Probe Quality Correlation
- Spearman rho: **-0.7556** (p=1.3e-05)
- Strong negative correlation between probe F1 and false negative rate. Probe quality is the primary driver of false negatives.

## 2. Subtype Taxonomy Stability

### L12-16k (n=16 latents)
- Early CV: 0.0955 (range: [75.0, 93.8])
- Late range: [6.2, 12.5]
- Partial range: [0.0, 12.5]
- Data-driven threshold (p95): 0.044038746878504745

### L12-65k (n=65 latents)
- Early CV: 0.3240 (range: [32.3, 95.4])
- Late range: [3.1, 33.8]
- Partial range: [1.5, 33.8]
- Data-driven threshold (p95): 0.04883190952241412

### Statistical Stability
- KW significant: 4/5 thresholds
- Late > Early (all configs): 5/5 thresholds
- Full ordering criterion met: True
- Overall pass: True

## 3. Cross-Iteration Consistency

- iter_006 CV: 0.0793
- iter_008 CV: 0.0773
- CV difference: 0.002
- CV consistent: True
- Heatmap consistent: True

## 4. Structural Evidence

1. False negatives are CONSTANT (87/576) across all 20 grid cells. Varying cosine threshold (0.01-0.05) and magnitude gap (0.5-2.0) does not change how many tokens the probe misclassifies after SAE encoding.
2. Absorption rate CV = 0.077 (< 0.10), classified as STABLE. Rate varies only from 11.8% to 15.1% across the 5x4 grid.
3. Maximum absorption reduction from loosest to strictest thresholds: 0.0329 (21.8% relative). Even at the strictest settings, absorption remains at 11.8% -- far from zero.
4. Probe quality is the primary driver of false negatives (rho=-0.756, p=1.3e-05). Letters with low probe F1 have high FN rates regardless of threshold.
5. L12-16k subtype taxonomy is stable: early=[75.0, 75.0, 75.0, 75.0, 93.8] across thresholds (constant 75% at thresholds 0.2-0.35, jumps only at 0.40).
6. Despite L12-65k subtype shifts, Kruskal-Wallis significance holds in 4/5 thresholds and late>early ordering holds in 5/5 thresholds.

## 5. Implications for Paper

- Absorption is an inherent property of the SAE's learned representation, not an artifact of detection thresholds.
- Improving thresholds cannot eliminate false negatives -- only architectural changes or training modifications can.
- For JumpReLU SAEs specifically: the absorption rate (~14%) at layer 12 is genuine, not a threshold artifact.
- Probe quality (F1) is a stronger predictor of false negative rate than any absorption detection threshold.
- The 5x4 grid analysis confirms the metric is robust (CV=0.077) and supports cross-threshold comparisons.

## 6. Appendix Content

- Table: 5x4 heatmap of absorption rates (cosine threshold x magnitude gap)
- Finding: CV=0.077, all 20 cells in [11.8%, 15.1%] range
- Finding: False negatives constant (n=87/576) across all threshold settings
- Finding: Perfect monotonicity in both dimensions
- Conclusion: Absorption measurement is threshold-robust; control failure is structural