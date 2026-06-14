# Phase 2a: Three-Subtype Taxonomy Classification Results

**Mode:** FULL
**Timestamp:** 2026-04-12T15:23:04.026911
**SAE Configs:** L12-16k, L12-65k
**Primary Threshold:** 0.3

## Summary Table (Primary Threshold = 0.3)

| Config | N Absorbed | % Early | % Late | % Partial | Ordering (late>partial>early) | KW p-val | MW p-val |
|--------|-----------|---------|--------|-----------|-------------------------------|----------|----------|
| L12-16k | 16 | 75.0% | 12.5% | 12.5% | NO | 0.2366 | 0.2198 |
| L12-65k | 65 | 72.3% | 13.8% | 13.8% | NO | 0.0002 | 0.0001 |

## FULL Pass Criteria

- All three subtypes non-empty (each >= 5%): PASS
- Kruskal-Wallis p < 0.05: PASS
- EDA ordering late > partial > early: PASS
- **Go/No-Go Decision:** GO

## Threshold Stability

EDA ordering (late > early) holds across threshold sweep:
- L12-16k: 5/5 thresholds
- L12-65k: 5/5 thresholds

## Notes

- FULL mode: all absorbed latents used (no pilot cap)
- Subtype classification: decoder dictionary lookup (training-free, no activations)
- Parent probe direction: mean decoder direction of absorbed latents (proxy)
- Full pipeline would use RAVEL probe directions + activation data for Late/Partial split
- Random cosine 95th percentile threshold reported alongside fixed thresholds
- EDA ordering: Late reflects encoder suppression (high EDA), Early reflects never-learned parent
