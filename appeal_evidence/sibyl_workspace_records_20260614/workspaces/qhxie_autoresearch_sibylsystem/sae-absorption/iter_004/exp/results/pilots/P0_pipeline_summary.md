# P0 Pipeline Validation Summary

**Model**: GPT-2 Small (gpt2-small-res-jb SAEs, layer 8)
**Decision**: GO
**Timestamp**: 2026-04-14T17:23:14.148459

## Absorption Rates (Layer 8, ~24k SAE)

| Letter | Absorption Rate | n_absorbed | n_total | In Range [5-50%] |
|--------|----------------|------------|---------|-----------------|
| A | 0.100 | 4 | 40 | YES |
| B | 0.121 | 4 | 33 | YES |
| C | 0.185 | 5 | 27 | YES |
| D | 0.148 | 4 | 27 | YES |
| E | 0.080 | 2 | 25 | YES |

## L0 Statistics

| SAE | Width | Mean L0 | Std L0 |
|-----|-------|---------|--------|
| Narrow | ~24k | 137.4 | 107.1 |
| Wide | ~49k | 124.1 | 137.8 |

L0 difference: 9.7% -> Confound: **NO**

## Gate Decision

- Absorption gate (≥3 letters in [5-50%]): PASS
- Alpha gate (≥1 letter-feature pair in top-10): FAIL
- Pipeline gate: PASS
- **Final (lenient): GO**
