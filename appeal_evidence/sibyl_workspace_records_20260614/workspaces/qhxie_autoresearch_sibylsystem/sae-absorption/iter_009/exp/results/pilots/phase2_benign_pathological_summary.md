# Phase 2.2: Benign vs. Pathological Absorption Diagnostic (H8)

**H8 Verdict**: FALSIFIED
**Pilot**: PASS
**Time**: 20.2 minutes

## Design

- **Hierarchy**: city-continent at L24
- **SAE**: L24_16k
- **Thresholds**: [0.05, 0.1, 0.2]
- **Contexts per entity**: 30
- **Control directions**: 5

## H8 Hypothesis (Benign Absorption)

Only 0.0% of absorption instances are benign (|logit_change| <= 0.1), below 10% falsification criterion.

- **Direction specificity**: YES
- **Parent ablation mean |change|**: 3.9795
- **Random direction mean |change|**: 0.0040

## Classification at Multiple Thresholds

| Threshold | Benign | Pathological | Benign % |
|-----------|--------|--------------|----------|
| 0.05 | 0 | 1471 | 0.0% |
| 0.1 | 0 | 1471 | 0.0% |
| 0.2 | 0 | 1471 | 0.0% |

## Logit Change Distribution

- **Mean**: -3.9795 +/- 0.4177
- **Median**: -3.9664
- **|Mean|**: 3.9795
- **Range**: [-5.6779, -2.3438]
- **N**: 1471
- **Percentiles**: 5th=-4.654, 25th=-4.238, 50th=-3.966, 75th=-3.710, 95th=-3.335

## Statistical Tests

- **t-test vs zero**: t=-365.2691, p=0.000000, significant: True
- **Mann-Whitney (parent vs control)**: U=2163841.0, p=0.000000, significant: True

## Per-Class Results

| Class | Entities | Processed | Mean |delta| | Benign% @0.1 |
|-------|----------|-----------|------------|--------------|
| Europe | 50 | 1471 | 3.9795 | 0.0% |

## Summary

- Entities completed: 50/50
- Total FN instances: 1471
- Total processed: 1471