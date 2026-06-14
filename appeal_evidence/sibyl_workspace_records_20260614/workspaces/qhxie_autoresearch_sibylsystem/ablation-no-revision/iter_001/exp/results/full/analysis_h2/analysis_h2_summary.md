# Analysis H2: Ea-Difficulty Correlation

## Task: analysis_h2
## Date: 2026-04-29T19:22:53.661278
## Samples: 50 problems

## H2 Evaluation

**Hypothesis**: Activation energy correlates with MATH difficulty level

**Status**: CONFIRMED

**Key Metrics**:
- Spearman(Ea, level): 0.4479 (p=0.0011)
- Pearson(Ea, level): 0.4875
- Threshold: 0.4

**Pass**: YES

## H5 Evaluation

**Hypothesis**: Consistency-based Ea matches saturation-derived k0

**Status**: FALSIFIED

**Key Metrics**:
- Spearman(Ea, k0): -0.2191 (p=0.5431)
- Valid k0 pairs: 10/50
- Threshold: 0.5

**Pass**: NO

## Additional Correlations

| Pair | Spearman | p-value |
|------|----------|---------|
| Ea vs Accuracy | -0.0634 | 0.6618 |
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

Ea correlates with MATH difficulty level (Spearman=0.4479, p=0.0011)

Consistency-based Ea does NOT correlate with saturation-derived k0 (Spearman=-0.2191)

## Recommendation

GO

- **GO**: Ea reliably correlates with difficulty (Spearman > 0.4)
- **NO_GO**: Ea does not correlate with difficulty
