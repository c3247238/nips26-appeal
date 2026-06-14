# BSD Full-Scale Evaluation — Countdown-16 (PILOT)

**Verdict: GO**

## Results

| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time |
|--------|----------|-------|-------|------------|----------|
| Vanilla | 0.0% (0/16) | 0.140 | 0.079 | 0.875 | 3.7s |
| DMI (alpha=0.3) | 12.5% (2/16) | 0.162 | 0.095 | 0.857 | 3.9s |
| **BSD (k=0.75, linear)** | **6.2%** (1/16) | 0.086 | 0.048 | 0.913 | 3.8s |

## Entropy Analysis

- Start entropy: 3.50
- End entropy: 0.00
- Decreasing: 16/16

## Flip Analysis

- BSD correct, DMI wrong: 0
- DMI correct, BSD wrong: 1
- BSD correct, Vanilla wrong: 1
- Vanilla correct, BSD wrong: 0

## Runtime

- Total: 3.1 minutes
