# BSD + A-CFG Combination — Countdown-16 (PILOT)

**Verdict: NO-GO**

## Configuration

- BSD: k_frac=0.75, alpha=linear(0.1->0.8)
- A-CFG: w=1.5, remask_pct=0.1

## Results

| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |
|--------|----------|-------|-------|------------|----------|-------|
| Vanilla (128) | 6.2% (1/16) | 0.160 | 0.095 | 0.865 | 3.7s | 1.0x |
| Vanilla (256, fair) | 12.5% (2/16) | 0.102 | 0.041 | 0.914 | 7.3s | 2.0x |
| DMI (alpha=0.3) | 0.0% (0/16) | 0.000 | 0.000 | 0.405 | 3.8s | ~1.05x |
| BSD (k=0.75) | 12.5% (2/16) | 0.081 | 0.039 | 0.873 | 3.8s | ~1.1x |
| A-CFG (w=1.5) | 12.5% (2/16) | 0.066 | 0.030 | 0.930 | 7.4s | ~2.0x |
| BSD+ACFG (COMBO) | 6.2% (1/16) | 0.079 | 0.046 | 0.915 | 6.6s | ~2.1x |

## Hypothesis H7 Test

- Combination accuracy: 6.2%
- max(BSD, ACFG): 12.5%
- Combo > max(BSD, ACFG): False
- Combo > Vanilla + 3pp: False

## Flip Analysis

### Combo vs BSD
- Combo wins: 0
- BSD wins: 1

### Combo vs A-CFG
- Combo wins: 1
- A-CFG wins: 2

## Runtime

- Total: 8.7 minutes
