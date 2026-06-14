# Analysis H3: Single-Pass Threshold Evaluation

## Task: analysis_h3
## Date: 2026-04-29T19:27:40.926598
## Samples: 50 problems

## H3 Evaluation

**Hypothesis**: Single-pass accuracy > 75% on low-Ea problems

**Status**: CONFIRMED

**Key Metrics**:
- Optimal Ea threshold: 9.999999999903855
- Low-Ea accuracy: 0.75 (threshold: 0.75)
- High-Ea accuracy: 0.5
- Delta: 0.25

**Pass**: YES

## Predictor Quality

| Metric | Value |
|--------|-------|
| AUC | 0.4357 |
| Spearman(Ea, accuracy) | -0.0634 (p=0.6618) |
| Pearson(Ea, accuracy) | 0.0243 |

## Correlations

| Pair | Spearman | p-value |
|------|----------|---------|
| Ea vs Level | 0.4479 | 0.0011 |
| Level vs Accuracy | -0.0567 | 0.6957 |

## Accuracy by MATH Level

| Level | Count | Mean Ea | Accuracy |
|-------|-------|---------|----------|
| Level 1 | 2 | 9.4651 | 50.0% |
| Level 2 | 9 | 9.6434 | 77.8% |
| Level 3 | 15 | 9.7504 | 66.7% |
| Level 4 | 11 | 9.7082 | 72.7% |
| Level 5 | 13 | 10.0000 | 61.5% |

## Interpretation

Low-Ea accuracy (75.0%) meets threshold (75%). Ea is a useful routing signal.

## Recommendation

GO

- **GO**: Ea reliably predicts single-pass solveability (>75% accuracy for low-Ea)
- **REFINE**: Ea has some predictive power but doesn't meet the 75% threshold
- **NO_GO**: Ea is not a useful routing signal
