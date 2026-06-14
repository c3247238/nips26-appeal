# DMI Baseline Reproduction — Countdown-100

## Results
| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time |
|--------|----------|-------|-------|------------|----------|
| Vanilla | 7.0% (7/100) | 0.150 | 0.086 | 0.750 | 3.7s |
| DMI (alpha=0.3) | 8.0% (8/100) | 0.125 | 0.071 | 0.731 | 3.9s |

## Pass Criteria
- Target: 9.3% +/- 2pp → PASS (actual: 8.0%)
- DMI >= Vanilla: PASS
- Overall: **GO**

## Flips
- Vanilla->DMI correct: 5
- DMI->Vanilla correct (regressions): 4

## Runtime
- Total: 12.7 minutes
- Time overhead: 1.06x
