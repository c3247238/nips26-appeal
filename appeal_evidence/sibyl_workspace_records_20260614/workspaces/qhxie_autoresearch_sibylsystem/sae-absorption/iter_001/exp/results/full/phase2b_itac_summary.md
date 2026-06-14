# Phase 2b: ITAC Evaluation Results

**Mode:** FULL
**Timestamp:** 2026-04-12T15:29:05.293148
**SAE Configs:** L12-16k, L12-65k

## ITAC Efficacy Table (Paper Table 2)

| Config | N Late | N ITAC Targets | Parent FN Before | Parent FN After | FN Reduction % | FVU Change | Mono Rho | Pass |
|--------|--------|----------------|-----------------|-----------------|----------------|------------|----------|------|
| L12-16k | 2 | 1 | 0.000 | 0.000 | 0.00% | 0.2211 | N/A | WEAK |
| L12-65k | 13 | 10 | 0.076 | 0.062 | 3.14% | -0.0423 | 0.655 | PASS |

## Full Mode Pass Criteria

- Any positive FN reduction: PASS
- FVU constraint (< +5%): PASS
- >= 5% relative FN reduction: FAIL (potential H5 falsification)
- **Go/No-Go Decision:** GO

## Notes

- FULL mode: All absorbed latents (no 100-sample cap)
- 500 synthetic inputs per latent (vs 200 in pilot)
- Parent match threshold: 0.25 (vs 0.20 in pilot)
- Null test runs on ALL early/partial latents (not just 20)
- FN measurement uses synthetic activations (decoder column-based parent-positive inputs)
- Global mean FN reduction: 2.69% (over 7 measurements)
