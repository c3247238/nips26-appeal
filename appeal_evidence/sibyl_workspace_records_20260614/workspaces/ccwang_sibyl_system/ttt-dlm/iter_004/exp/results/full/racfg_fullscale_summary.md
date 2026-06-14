# RACFG/A-CFG Full-Scale Evaluation — Countdown-16 (PILOT)

**Verdict: GO**

## Pivot Note

Original RACFG (JSD stability) failed on Dream-7B (0% accuracy everywhere). This evaluation uses A-CFG (confidence-based re-masking) as the effective 'enhanced guidance' method.

## Results

| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |
|--------|----------|-------|-------|------------|----------|-------|
| Vanilla (128) | 0.0% (0/16) | 0.140 | 0.079 | 0.875 | 3.7s | 1.0x |
| Vanilla (256, fair) | 6.2% (1/16) | 0.162 | 0.105 | 0.839 | 7.3s | 2.0x |
| A-CFG w=1.0 | 6.2% (1/16) | 0.115 | 0.057 | 0.900 | 7.4s | ~2.0x |
| A-CFG w=1.5 (BEST) | 12.5% (2/16) | 0.106 | 0.054 | 0.889 | 7.4s | ~2.0x |
| A-CFG w=2.0 | 12.5% (2/16) | 0.115 | 0.052 | 0.885 | 7.4s | ~2.0x |

## Best Method

- **A-CFG w=1.5**: 12.5%
- Beats vanilla + 3pp: True
- Beats vanilla 2x (compute-fair): True

## Flip Analysis

### A-CFG w=1.5 vs Vanilla
- A-CFG wins: 2
- Vanilla wins: 0

### A-CFG w=1.5 vs Vanilla 2x (compute-fair)
- A-CFG wins: 2
- Vanilla-2x wins: 1

## Runtime

- Total: 9.0 minutes
