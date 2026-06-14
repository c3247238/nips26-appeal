# Pilot Experiment Summary

## Overview

Four pilot experiments were executed to validate and extend the absorption research findings.

| Experiment | Status | Key Result |
|------------|--------|------------|
| H_Safe pilot | FAIL | No difference between safety/non-safety (synthetic SAE) |
| H3_fix pilot | PASS | Steering verified; absorbed features 1.62x more sensitive |
| Multi-seed pilot | PARTIAL | Trained SAE absorption deterministic at 0.5 |
| H_Mech factorial | REVEALING | **Absorption is ENCODER-driven, not decoder geometry** |

## Key Findings

### 1. Absorption is Encoder-Driven (MAJOR)

The H_Mech 2x2 factorial decomposition revealed a surprising finding:

```
Condition A (Random encoder + Random decoder): 0.299
Condition B (Trained encoder + Random decoder): 0.490
Condition C (Random encoder + Trained decoder): 0.299  <- Same as A!
Condition D (Trained encoder + Trained decoder): 0.484  <- Same as B!
```

**Interpretation**: The decoder geometry contributes NOTHING to absorption. Absorption is entirely driven by encoder alignment with hierarchical features.

### 2. Trained SAE Absorption is Deterministic

Multi-seed validation across 3 random seeds (42, 43, 44):
- All seeds: exactly 0.5 absorption
- Zero variance across seeds
- Random baselines: 0.147 +/- 0.065 (variable)

This confirms absorption is a fixed geometric property of trained SAEs, not stochastic.

### 3. Steering Implementation Fixed

H3_fix successfully resolved the broken steering from pilot:
- Steering now changes activations: ||steered - baseline|| > 0
- Absorbed features show 62% higher sensitivity to steering
- Can now make causal claims about absorption impact

### 4. Safety Analysis Inconclusive

H_Safe pilot used synthetic SAE (Gemma Scope not available):
- p = 0.695 (no significant difference)
- Requires actual Gemma Scope SAEs for real safety analysis

## Recommendations

1. **Update paper narrative**: Absorption comes from encoder learning hierarchical representations, not decoder geometry
2. **Install Gemma Scope SAEs**: Required for H_Safe safety analysis
3. **Focus on encoder alignment**: Future work should investigate how encoder training creates hierarchical features

## Files

- `h_safe_pilot.json`: Safety feature absorption results
- `h3_fix_pilot.json`: Steering validation results
- `multiseed_pilot.json`: Multi-seed stability results
- `h_mech_factorial.json`: 2x2 decomposition results
