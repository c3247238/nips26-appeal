# C3B Matched RAVEL Comparison — Pilot Summary

**Status:** GO  
**Mode:** PILOT  
**Model:** GPT-2 Small (gpt2-small-res-jb SAEs)  
**Timestamp:** 2026-04-14T18:20:42  
**Runtime:** ~14 seconds  

---

## Pilot Scope

- 4 SAEs selected (2 lowest + 2 highest absorption from C2B + extra layer-4 config)
- 50 RAVEL-analog prompts per SAE (knowledge probing: country capitals, colors, facts)
- Simplified RAVEL metric: logistic regression accuracy on SAE features + feature concentration

---

## SAEs Evaluated

| Config | Layer | Width | Absorption (mean) | Group | RAVEL Score |
|--------|-------|-------|-------------------|-------|-------------|
| cfg_L8_24k_narrow | 8 | 24k | 0.044 | low | 0.2194 |
| cfg_L8_49k_medium | 8 | 49k | 0.048 | low | 0.1743 |
| cfg_L6_24k_narrow | 6 | 24k | 0.052 | high | 0.4646 |
| cfg_L4_24k_deep   | 4 | 24k | 0.167 | high | 0.5999 |

---

## RAVEL Score Components

| Config | CV Accuracy | Feature Concentration | RAVEL Score |
|--------|------------|----------------------|-------------|
| cfg_L8_24k_narrow | 0.081 ± 0.031 | 0.036 | 0.219 |
| cfg_L8_49k_medium | 0.100 ± 0.030 | 0.025 | 0.174 |
| cfg_L6_24k_narrow | 0.120 ± 0.048 | 0.081 | 0.465 |
| cfg_L4_24k_deep   | 0.200 ± 0.026 | 0.131 | 0.600 |

---

## Group Comparison

- **Low-absorption mean RAVEL:** 0.197
- **High-absorption mean RAVEL:** 0.532
- **Paired t-test:** t = -3.72, p = 0.916 (NOT significant; low-absorption does NOT outperform)
- **Cohen's d:** -3.72 (large effect in unexpected direction)

---

## Pass Criteria

| Criterion | Result |
|-----------|--------|
| RAVEL runs for all 4 SAEs | PASS (4/4) |
| Numeric scores produced | PASS |
| Non-degenerate (spread > 0.01) | PASS (spread = 0.426) |
| **Overall** | **PASS (GO)** |

---

## Critical Confounds (Flag for Full Experiment)

1. **Layer depth confound**: The "high absorption" group includes layer 4 SAE which naturally shows higher feature concentration due to different representation at early vs late layers. The matching criteria for full C3B must strictly control for layer.

2. **RAVEL metric validity**: The feature concentration proxy (top-1 feature share) captures absolute concentration, not relative-to-absorption. In early layers, features are less polysemantic regardless of absorption.

3. **Unexpected direction**: High-absorption SAEs scored HIGHER on RAVEL proxy. This contradicts H3 hypothesis. Two interpretations:
   - Our RAVEL proxy is not capturing the intended downstream quality (likely)
   - H3 may not hold for GPT-2 Small (possible)

4. **For full experiment**: Must use proper SAEBench RAVEL implementation on matched same-layer SAEs. The matched comparison must strictly fix layer and model.

---

## Recommendation

**GO for full C3B**, but with critical design changes:
- Fix layer when selecting low/high absorption matched pairs (do NOT cross layers)
- Use proper SAEBench RAVEL evaluation code rather than proxy metric
- Access Gemma Scope SAEs when available (gated HF access needed)
