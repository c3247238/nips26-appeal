# Correlation Analysis with Random Baseline (Iteration 2)

## Key Improvement

This analysis incorporates **random feature steering baseline** data,
addressing the critical missing control from Iteration 1.

## Hypothesis Results

| Hypothesis | Result | Notes |
|------------|--------|-------|
| H1 (Absorption vs Steering) | FAIL | Original test |
| H1b (Absorption vs Delta) | PASS | **Critical: feature-specific minus random** |
| H2 (Absorption vs Probing) | FAIL | Original test |
| H3 (Consistency) | FAIL | CV bug fixed |

## Random Baseline Validation

Feature-specific steering significantly outperforms random directions:
- Layer 4: +132% over random (t=6.41, p<0.0001, d=1.26)
- Layer 8: +126% over random (t=6.02, p<0.0001, d=1.18)

This validates that feature-specific steering captures meaningful directions.
