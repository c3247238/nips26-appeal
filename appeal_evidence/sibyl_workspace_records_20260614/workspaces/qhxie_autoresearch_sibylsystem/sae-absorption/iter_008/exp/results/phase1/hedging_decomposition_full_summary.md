# Phase 0.2 Full: Tightened Hedging Across Domains and SAEs

**Date:** 2026-04-15T23:19:21.828403
**Mode:** PILOT
**Elapsed:** 0.7 minutes

## First-Letter L0 Sensitivity

| Base L0 | Target L0 | FNs | Strict% | Compensatory% | Persistent% | Loose% |
|---------|-----------|-----|---------|---------------|-------------|--------|
| 22 | 176 | 304 | 7.9 | 86.2 | 5.9 | 94.1 |
| 82 | 176 | 74 | 0.0 | 78.4 | 21.6 | 78.4 |

## Cross-Domain Hedging Decomposition

| Hierarchy | SAE Config | FNs | Strict% | Compensatory% | Persistent% | Probe F1 |
|-----------|-----------|-----|---------|---------------|-------------|----------|
| city-language | L24_16k | 3 | 66.7 | 33.3 | 0.0 | 0.779 |

## Paper Summary Table

| Hierarchy | Strict Hedging% | Compensatory% | Persistent% | N FN |
|-----------|----------------|---------------|-------------|------|
| city-language | 66.7 | 33.3 | 0.0 | 3 |
| first-letter | 7.9 | 86.2 | 5.9 | 304 |

## Key Findings

- {"finding": "Strict vs loose hedging gap for city-language", "strict_pct": 66.7, "loose_pct": 100.0, "gap_pp": 33.3}
- {"finding": "Strict vs loose hedging gap for first-letter", "strict_pct": 7.9, "loose_pct": 94.1, "gap_pp": 86.2}
- {"finding": "Strict hedging comparison: city-language vs first-letter", "semantic_strict": 66.7, "firstletter_strict": 7.9, "diff_pp": 58.8}

## Pass Criteria: PASS
- decomposition_for_2_hierarchies_2_configs: True
- three_categories_all_nonempty: True
- absorption_hedging_ratio_reported: True
- l0_sensitivity_tested: True
- overall_pass: True