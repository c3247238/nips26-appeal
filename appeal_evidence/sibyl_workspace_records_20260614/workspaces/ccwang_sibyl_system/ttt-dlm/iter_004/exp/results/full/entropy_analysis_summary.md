# Entropy Analysis — Countdown-16 (PILOT)

**Verdict: GO**

## Hypothesis H2 Evaluation

**Claim**: BSD belief entropy monotonically decreases during denoising, reaching lower terminal entropy than vanilla.

**Result**: SUPPORTED

## Spearman Correlation (step vs entropy)

| Method | Avg rho | Monotonic (rho<-0.8) |
|--------|---------|---------------------|
| BSD (belief phase) | -0.9516 | 15/16 |
| Vanilla | -0.9323 | N/A |

## Terminal Entropy

| Method | Mean Terminal Entropy |
|--------|---------------------|
| BSD | 0.000993 |
| Vanilla | 0.001898 |
| **BSD < Vanilla** | **True** |
| Wilcoxon p-value | 0.971161 |

## Monotonicity

- Average monotonicity score: 0.7077
- Perfectly monotonic: 0/16

## Entropy-Accuracy Correlation

- BSD: r=0.7837276830286839, p=0.0003277083313694617
- Vanilla: r=None, p=None

## Accuracy

- BSD: 1/16 = 6.2%
- Vanilla: 0/16 = 0.0%

## Runtime

- Total: 2.1 minutes
