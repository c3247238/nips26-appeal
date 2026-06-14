# Phase 2.1: Decoder Direction Magnitude on First-Letter (H8 Cross-Domain)

**Cross-Hierarchy Verdict**: CONSISTENT
**Pilot**: PASS
**Time**: 2.7 minutes

## Design

- **Hierarchy**: first-letter at L24 (token_pos=-6)
- **SAE**: L24_16k
- **Thresholds**: [0.05, 0.1, 0.2]
- **Prompts per word**: 15
- **Control directions**: 5

## Cross-Hierarchy Comparison

First-letter mean |logit_change|=6.156 nats vs city-continent 3.979 nats (ratio=1.55). Both show strong direction specificity. Pathological severity is consistent across hierarchy types.

| Metric | First-Letter | City-Continent |
|--------|-------------|----------------|
| Mean |logit change| | 6.156 | 3.979 |
| Control |logit change| | 0.012 | 0.120 |
| Benign % (0.1) | 0.0% | 0.0% |
| N instances | 158 | 1471 |
| Direction specificity | YES | YES |

## Classification at Multiple Thresholds

| Threshold | Benign | Pathological | Benign % |
|-----------|--------|--------------|----------|
| 0.05 | 0 | 158 | 0.0% |
| 0.1 | 0 | 158 | 0.0% |
| 0.2 | 0 | 158 | 0.0% |

## Logit Change Distribution

- **Mean**: -6.1561 +/- 2.1724
- **Median**: -5.6717
- **|Mean|**: 6.1561
- **Range**: [-15.9777, -2.9947]
- **N**: 158
- **Percentiles**: 5th=-10.307, 25th=-6.695, 50th=-5.672, 75th=-4.901, 95th=-3.470

## Statistical Tests

- **t-test vs zero**: t=-35.5080, p=0.000000, significant: True
- **Mann-Whitney (parent vs control)**: U=24964.0, p=0.000000, significant: True

## Per-Letter Results (top by n_processed)

| Letter | Words | FN Processed | Mean |delta| | Control |delta| | Benign% @0.1 |
|--------|-------|-------------|------------|---------------|--------------|
| l | 9 | 65 | 5.9613 | 0.0097 | 0.0% |
| b | 6 | 17 | 4.7030 | 0.0145 | 0.0% |
| v | 2 | 17 | 3.6888 | 0.0104 | 0.0% |
| g | 3 | 14 | 6.1715 | 0.0079 | 0.0% |
| e | 4 | 13 | 8.9283 | 0.0319 | 0.0% |
| w | 2 | 12 | 8.8248 | 0.0090 | 0.0% |
| m | 4 | 6 | 4.9500 | 0.0095 | 0.0% |
| k | 3 | 4 | 10.3835 | 0.0222 | 0.0% |
| a | 2 | 3 | 6.6627 | 0.0155 | 0.0% |
| c | 1 | 3 | 3.9946 | 0.0108 | 0.0% |
| f | 3 | 3 | 5.7758 | 0.0043 | 0.0% |
| h | 1 | 1 | 13.6270 | 0.0093 | 0.0% |

## Summary

- Words tested: 460
- Words with absorption: 40
- Total FN instances: 158
- Total processed: 158