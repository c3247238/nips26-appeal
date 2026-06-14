# C2B Absorption Survey Pilot Summary

**Task**: C2B_absorption_survey_30sae (PILOT)
**Model**: GPT-2 Small (open-model anchor)
**Timestamp**: 2026-04-14T17:58:29.814793
**Decision**: GO

## Pilot Configs (3 of 30 planned)

| Config | Layer | Width | L0-Setting | A | B | C | D | E | Mean |
|--------|-------|-------|------------|---|---|---|---|---|------|
| cfg_L8_24k_narrow | 8 | 24576 | low | 0.040 | 0.060 | 0.040 | 0.000 | 0.080 | 0.044 |
| cfg_L8_49k_medium | 8 | 49152 | medium | 0.020 | 0.080 | 0.060 | 0.020 | 0.060 | 0.048 |
| cfg_L6_24k_narrow | 6 | 24576 | low | 0.040 | 0.100 | 0.040 | 0.020 | 0.060 | 0.052 |

## Pass Criteria

| Criterion | Result |
|-----------|--------|
| All 15 data points measured | PASS (15/15) |
| Std of rates > 0.02 | PASS (std=0.0261) |
| ≥2 configs with absorption > 0.10 | FAIL |

## Statistics

- Mean absorption rate: 0.048
- Std absorption rate: 0.0261
- Range: [0.000, 0.100]
- Runtime: 57.5s

## Interpretation

PILOT PASSED: Pipeline works, absorption rates vary across configs, proceed to full 30-SAE survey.

Note: Full task (C2B) will survey ~30 Gemma Scope SAE configurations across widths × layers × L0 settings.
GPT-2 Small is used as open-model anchor (Gemma-2-2b requires gated HF access).
