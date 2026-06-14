# R4-B RAVEL Probes Proper — Pilot Summary

**Task:** r4b_ravel_probes_proper (PILOT)
**Date:** 2026-04-13
**GO/NO-GO:** NO_GO (probe quality gate failure)

## Key Finding

The R4-B pilot confirms that **RAVEL probe training on the target model cannot proceed**
without access to Gemma 2B or Llama-3.1-8B (both HF-gated).

### Probe Quality Results (Best: GPT-2 Medium, layer 20, C=0.01)

| Hierarchy | Probe Acc | Majority | Margin | Gate | Status |
|-----------|-----------|----------|--------|------|--------|
| city-continent | **66.2%** | 33.8% | +32.5pp | 85% (7 classes) | FAIL |
| city-country | **55.3%** | 34.1% | +21.3pp | 80% (50 classes) | FAIL |
| city-language | **60.0%** | 43.4% | +16.6pp | 80% (50 classes) | FAIL |

### Comparison with R3 Qwen2.5-0.5B Probes

| Hierarchy | R3 (Qwen2.5-0.5B) | R4 Best (GPT-2 Med L20) | Delta |
|-----------|-------------------|-------------------------|-------|
| city-continent | 71.4% | 66.2% | -5.2pp |
| city-country | 37.8% | 55.3% | +17.5pp |
| city-language | 36.8% | 60.0% | +23.2pp |

GPT-2 Medium is BETTER for country/language but WORSE for continent.

## Root Cause

1. **Model capability gap**: No available bridge model encodes geographic knowledge at 85%+ accuracy
2. **Random projection destroys signal**: Projecting d=1024 to d=2304 via random matrix
   loses directional information; shuffled control = real absorption rate (gap=0.000 all configs)

## Absorption Metric Results

9 configs completed (3 hierarchies x 3 SAE configs: L5-16k, L12-16k, L12-65k):
- Shuffled control indistinguishable from real (gap=0.000 for all configs)
- Confirms: random projection loses directional semantic signal
- Strong EDA-absorption Spearman rho (0.37-0.69) validates EDA metric robustness

## Paper Implications

- H3 (cross-domain absorption): **UNRESOLVED** pending target model access
- R3 RAVEL results must be presented with prominent methodological caveat
- R3 rho=0.924 also suspect (Qwen projection artifact)
- Recommended paper framing per task_plan fallback_decisions:
  "Preliminary cross-domain evidence pending proper probe training with target model"

## Next Steps

Per task_plan.json fallback_decisions:
- Retain intra-RAVEL coherence (rho=0.924) as main cross-domain evidence WITH CAVEAT
- Acknowledge absolute absorption rates unreliable without proper probes
- Pivot H3 to: existence evidence only, rates pending proper probe training
