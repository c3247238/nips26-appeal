# GSM8K Generalization Test — PILOT (16 samples)

**Verdict: GO**

## Results

| Method | Accuracy | 95% CI | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |
|--------|----------|--------|-------|-------|------------|----------|-------|
| Vanilla | 25.0% (4/16) | [6.2%, 50.0%] | 0.121 | 0.060 | 0.898 | 4.6s | 1.0x |
| DMI (alpha=0.3) | 25.0% (4/16) | [6.2%, 50.0%] | 0.158 | 0.074 | 0.844 | 4.9s | ~1.05x |
| BSD (k=0.75) | 18.8% (3/16) | [0.0%, 37.5%] | 0.056 | 0.027 | 0.909 | 4.6s | ~1.1x |
| **A-CFG (w=1.5)** | **37.5% (6/16)** | [12.5%, 62.5%] | 0.135 | 0.071 | 0.886 | 9.4s | ~2.0x |

## Best Method

- **A-CFG**: 37.5%
- Beats vanilla: True

## Flip Analysis

- BSD_over_vanilla: 2
- vanilla_over_BSD: 3
- A-CFG_over_vanilla: 3
- vanilla_over_A-CFG: 1
- DMI_over_vanilla: 2
- vanilla_over_DMI: 2

## Runtime

- Total: 6.4 minutes
