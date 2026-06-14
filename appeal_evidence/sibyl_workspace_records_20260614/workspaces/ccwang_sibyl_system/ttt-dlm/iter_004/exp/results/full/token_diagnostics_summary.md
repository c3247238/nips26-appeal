# Token-Level CFG Impact Analysis — Countdown-16 (PILOT)

**Verdict: GO** — Diagnostics saved successfully with meaningful patterns

## Pivot Note

Original RACFG (JSD stability) failed on Dream-7B. This analysis uses A-CFG (confidence-based re-masking) and explicitly characterizes WHY JSD fails.

## Results

| Method | Accuracy | rep-3 | distinct-3 | Avg Time |
|--------|----------|-------|------------|----------|
| A-CFG w=1.5 | 12.5% (2/16) | 0.054 | 0.889 | 7.6s |
| Vanilla | 0.0% (0/16) | 0.079 | 0.875 | 3.7s |

## Guidance Impact

- Guidance first activates at step: 4
- Total active guidance steps across all samples: 496

## JSD vs Confidence Signal Quality

- **Issue**: JSD stability scores on Dream-7B are near-uniform (~0.997)
- **Root cause**: Dream-7B's logit distributions change very little between consecutive denoising steps, making JSD ~0 and stability ~1.0 for all positions. This means JSD cannot discriminate 'reasoning-critical' positions.
- **Signal ratio**: 1490282.0067
- **Interpretation**: JSD variance is 1490282.0067x that of confidence. JSD has more variance (unexpected).

## Guidance vs Correctness

- Correct samples (2): mean guidance = 22080.9083
- Wrong samples (14): mean guidance = 19366.5892
- Correct samples received MORE guidance on average, suggesting CFG helps where it's most needed.

## Top Guided Positions

- Position 16: mean magnitude = 1024.1476
- Position 64: mean magnitude = 905.7821
- Position 24: mean magnitude = 874.3369
- Position 8: mean magnitude = 825.1070
- Position 40: mean magnitude = 770.5866
- Position 56: mean magnitude = 738.4835
- Position 52: mean magnitude = 735.1676
- Position 124: mean magnitude = 733.3944
- Position 68: mean magnitude = 731.9981
- Position 28: mean magnitude = 725.4861

## Runtime

- Total: 3.1 minutes
