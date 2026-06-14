# P0 Pipeline Validation — FULL Mode (26 Letters)

**Model**: GPT-2 Small (gpt2-small-res-jb SAEs, layer 8)
**Mode**: FULL (all 26 letters)
**Decision**: GO
**Timestamp**: 2026-04-14T19:20:17.236976

## Absorption Rates by Letter (Layer 8, ~24k SAE)

| Letter | Absorption Rate | n_absorbed | n_total | In Range [5-50%] |
|--------|----------------|------------|---------|-----------------|
| A | 0.079 | 3 | 38 | YES |
| B | 0.038 | 1 | 26 | NO |
| C | 0.051 | 2 | 39 | YES |
| D | 0.027 | 1 | 37 | NO |
| E | 0.025 | 1 | 40 | NO |
| F | 0.083 | 1 | 12 | YES |
| G | 0.065 | 2 | 31 | YES |
| H | 0.125 | 1 | 8 | YES |
| I | 0.139 | 5 | 36 | YES |
| J | 0.222 | 2 | 9 | YES |
| K | 0.091 | 2 | 22 | YES |
| L | 0.077 | 1 | 13 | YES |
| M | 0.034 | 1 | 29 | NO |
| N | 0.250 | 3 | 12 | YES |
| O | 0.050 | 2 | 40 | YES |
| P | 0.059 | 2 | 34 | YES |
| Q | 0.130 | 3 | 23 | YES |
| R | 0.062 | 2 | 32 | YES |
| S | 0.106 | 5 | 47 | YES |
| T | 0.079 | 3 | 38 | YES |
| U | 0.000 | 0 | 21 | NO |
| V | 0.125 | 2 | 16 | YES |
| W | 0.080 | 2 | 25 | YES |
| X | 0.000 | 0 | 0 | NO |
| Y | 0.000 | 0 | 10 | NO |
| Z | 0.000 | 0 | 3 | NO |

## Summary Statistics

- **Mean absorption rate**: 0.077
- **Median absorption rate**: 0.071
- **Std**: 0.061
- **Letters in range [5-50%]**: 18/26

## L0 Statistics

| SAE | Width | Mean L0 | Std L0 |
|-----|-------|---------|--------|
| Narrow | ~24k | 137.4 | 107.1 |
| Wide | ~49k | 124.1 | 137.8 |

L0 difference: 9.7% -> Confound: **NO**

## Gate Decision

- Absorption gate (≥13/26 letters in [5-50%]): PASS (18/26)
- Alpha gate (≥1 letter-feature pair in top-10): FAIL
- Pipeline gate (≥13 letters processed): PASS (26/26)
- **Final (lenient): GO**
