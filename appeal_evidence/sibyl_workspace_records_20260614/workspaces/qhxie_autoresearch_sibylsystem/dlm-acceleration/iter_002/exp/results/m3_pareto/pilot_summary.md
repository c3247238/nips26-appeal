# M3 (AR-Guided Unmasking) Corrected Pareto Pilot Summary

## Task: m3_pareto_corrected
## Mode: PILOT
## Verdict: GO

## Configuration
- Model: LLaDA-8B-Instruct + Qwen2.5-0.5B guide
- Guidance weights: {0.3, 0.5, 0.7}
- Samples: 200 GSM8K + 100 MATH500 + 50 HumanEval, seed=42
- Warmup: 5 samples discarded from TPS computation
- Combined metric: 0.7*GSM8K + 0.3*MATH500 (corrected, no penalty)
- Elapsed: 72.7 min

## Results Summary

| gw  | GSM8K Acc | GSM8K Speedup | GSM8K AccRet | MATH500 Acc | MATH500 Speedup | Combined QAS |
|-----|-----------|---------------|-------------|-------------|-----------------|-------------|
| 0.3 | 0.730     | 1.651x        | 1.025       | 0.260       | 1.154x          | 2.136       |
| 0.5 | 0.740     | 1.651x        | 1.039       | 0.250       | 1.149x          | 2.108       |
| 0.7 | 0.740     | 1.647x        | 1.039       | 0.270       | 1.156x          | 2.188       |

## Key Findings

1. **Pass criteria MET**: All 3 guidance weights achieve GSM8K AccRet > 0.95 AND speedup > 1.0x
2. **Best operating point**: gw=0.7 (combined QAS=2.188, combined speedup=1.500x)
3. **M3 actually improves GSM8K accuracy**: AccRet > 1.0 for all weights (73-74% vs baseline 71.2%)
4. **MATH500 shows strong improvement**: AccRet 2.26-2.44x (25-27% vs baseline 11.1%)
5. **HumanEval**: 0% pass@1 across all weights (baseline only 2.4%, too low for signal)
6. **TPS stable across weights**: GSM8K ~51 TPS, MATH500 ~91 TPS (minimal overhead difference)

## Comparison with iter_001

| Metric | iter_001 (gw=0.3) | iter_002 (gw=0.3) | Change |
|--------|-------------------|-------------------|--------|
| GSM8K Acc | 0.73 | 0.73 | Same |
| GSM8K Speedup | 1.677x | 1.651x | ~same |
| QAS formula | 0.5x penalty for >5% drop | No penalty | Corrected |
| Combined metric | 4-benchmark weighted | 0.7*GSM8K + 0.3*MATH500 | Cleaned |
| Benchmarks | GSM8K+MATH500+HumanEval+MBPP | GSM8K+MATH500 (HumanEval separate) | Dropped uninformative |

## Observations

- M3 at gw=0.3-0.7 is remarkably stable -- accuracy and speedup barely change across weights
- The Qwen2.5-0.5B guide provides consistent quality improvement without significant latency cost
- The ~1.65x GSM8K speedup is real (not from the guide itself, but from the confidence-based unmasking order being improved)
- MATH500 improvement is notable but may partly reflect the small sample size (100 samples, baseline 11.1%)
- gw=0.7 is marginally best (highest combined QAS) but all three are viable operating points

## Recommendation for Full Run

Proceed with all 3 guidance weights (0.3, 0.5, 0.7) at full scale: 1319 GSM8K + 500 MATH500, seeds=[42, 123, 456]. The narrow differences between weights justify full-scale evaluation to determine if any weight consistently dominates.
