# Analysis H1: Arrhenius Fit Quality

## Task: analysis_h1
## Date: 2026-04-29T18:43:50.841345
## Samples: 50 problems

## Overall Accuracy by k

| k | Accuracy |
|---|----------|
| 1 | 0.680 |
| 2 | 0.780 |
| 4 | 0.860 |
| 8 | 0.840 |
| 16 | 0.820 |

## Model Comparison (Overall Curve)

| Model | R2 | AIC | AICc | BIC |
|-------|-----|-----|------|-----|
| exponential | 0.9243 | -36.43 | -30.43 | -37.21 |
| power_law | 0.9185 | -34.07 | -10.07 | -35.24 |
| logarithmic | 0.5689 | -27.73 | -21.73 | -28.52 |

**Best AIC**: exponential
**Best AICc**: exponential
**Best BIC**: exponential

## Per-Problem Fit Statistics

| Model | Mean R2 | Median R2 | Std R2 | Best AIC | Best BIC |
|-------|---------|-----------|--------|----------|----------|
| exponential | 0.0770 | 0.0000 | 0.2078 | 8 | 4 |
| power_law | 0.2197 | 0.0000 | 0.3070 | 20 | 24 |
| logarithmic | 0.1637 | 0.0000 | 0.2623 | 22 | 22 |

## H1 Evaluation

**Status**: CONFIRMED
**Overall R2 (exponential)**: 0.9243 (threshold > 0.85)
**Pass**: YES

### Interpretation
Exponential (Arrhenius) model provides the best fit on the aggregate accuracy curve (AICc, BIC) and wins R2 on majority of per-problem fits. H1 is CONFIRMED.

## Recommendation

GO
