# C2D Taxonomy Pilot Summary

**Model**: GPT-2 Small (open-model anchor)
**SAE**: gpt2-small-res-jb / blocks.8.hook_resid_pre
**Letters tested**: A, B, C, D, E
**Timestamp**: 2026-04-14T18:44:40.356430

## Taxonomy Results

| Letter | Type | Absorb Rate | Type II frac | DAS(k=3) |
|--------|------|-------------|--------------|----------|
| A | None | 0.000 | 0.000 | 0.0117 |
| B | Type_I | 0.113 | 0.113 | 0.0707 |
| C | Type_I | 0.188 | 0.156 | 0.0123 |
| D | Type_I | 0.150 | 0.139 | 0.0455 |
| E | Type_I | 0.762 | 0.208 | 0.0205 |

## Summary Statistics

- Type I count: 4/5 (80.0%)
- Type II count: 0/5 (0.0%)
- Type III count: 0/5 (0.0%)
- None: 1/5
- **Comprehensive absorption rate**: 80.0%

## Pass Criteria

- pass_all_rules_executed: `True`
- pass_at_least_2_types_observed: `True`
- pass_comprehensive_gt_type_i: `True`
- n_distinct_outcomes: `2`
- n_absorption_types_observed: `1`
- comprehensive_rate: `0.8`
- type_i_rate: `0.8`
- c2b_type_i_proxy: `0.044000000000000004`

## Go/No-Go: **GO**

**Runtime**: 9.6s