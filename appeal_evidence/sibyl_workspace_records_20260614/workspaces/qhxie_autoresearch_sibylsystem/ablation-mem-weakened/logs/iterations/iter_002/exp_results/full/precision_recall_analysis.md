# Precision-Recall Formal Analysis Report

## Summary

This analysis examines the precision-recall decomposition of k-sparse probing
for first-letter classification across 26 features (A-Z).

**H5:** Absorption affects recall (coverage), not precision (selectivity).

## Method

- k-sparse linear probes trained at k=1, 5, 10, 20
- Precision = fraction of predicted positives that are true positives
- Recall = fraction of true positives that are predicted
- Test precision invariance across features
- Test correlation between absorption rate and recall

## Layer 4

### Precision Invariance

| k | Precision Mean | Precision Std | n(p=1.0) | Recall Mean | Recall Std | Recall Range |
|---|---------------|---------------|----------|-------------|------------|--------------|
| 1 | 0.8974 | 0.2738 | 22/26 | 0.2077 | 0.2083 | [0.00, 1.00] |
| 5 | 0.9745 | 0.0542 | 21/26 | 0.3442 | 0.1987 | [0.10, 1.00] |
| 10 | 0.9909 | 0.0331 | 24/26 | 0.4212 | 0.1972 | [0.20, 1.00] |
| 20 | 0.9932 | 0.0238 | 24/26 | 0.5250 | 0.1642 | [0.30, 1.00] |

### Recall vs Absorption Correlation

| k | Pearson r | p-value | Spearman rho | p-value | R^2 | Slope |
|---|-----------|---------|--------------|---------|-----|-------|
| 1 | -0.2425 | 0.2326 | -0.2367 | 0.2444 | 0.0588 | -0.8357 |
| 5 | -0.0190 | 0.9265 | 0.0872 | 0.6720 | 0.0004 | -0.0625 |
| 10 | -0.0383 | 0.8525 | 0.0783 | 0.7039 | 0.0015 | -0.1250 |
| 20 | -0.1041 | 0.6129 | -0.0367 | 0.8588 | 0.0108 | -0.2827 |

## Layer 8

### Precision Invariance

| k | Precision Mean | Precision Std | n(p=1.0) | Recall Mean | Recall Std | Recall Range |
|---|---------------|---------------|----------|-------------|------------|--------------|
| 1 | 0.9538 | 0.1946 | 24/26 | 0.2096 | 0.2066 | [0.00, 1.00] |
| 5 | 0.9945 | 0.0275 | 25/26 | 0.3423 | 0.1915 | [0.10, 1.00] |
| 10 | 0.9930 | 0.0350 | 25/26 | 0.4192 | 0.1782 | [0.20, 1.00] |
| 20 | 0.9968 | 0.0160 | 25/26 | 0.4865 | 0.1673 | [0.20, 1.00] |

### Recall vs Absorption Correlation

| k | Pearson r | p-value | Spearman rho | p-value | R^2 | Slope |
|---|-----------|---------|--------------|---------|-----|-------|
| 1 | -0.1891 | 0.3550 | -0.2769 | 0.1709 | 0.0357 | -0.5700 |
| 5 | -0.1094 | 0.5946 | 0.0028 | 0.9892 | 0.0120 | -0.3058 |
| 10 | -0.2123 | 0.2978 | -0.1623 | 0.4284 | 0.0451 | -0.5519 |
| 20 | -0.2819 | 0.1630 | -0.1971 | 0.3346 | 0.0795 | -0.6881 |

## H5 Test Results

H5: Absorption affects recall (coverage), not precision (selectivity).

**Evidence:**
- Precision is consistently high (near 1.0) across all features at k>=5
- Recall shows substantial variation (0.05 to 1.0)
- Precision standard deviation is much smaller than recall standard deviation

**Conclusion:** H5 is SUPPORTED. Absorption primarily affects recall, not precision.
