# C2C PMI Regression Pilot Summary

**Mode:** PILOT
**Task ID:** C2C_pmi_regression
**Timestamp:** 2026-04-14T18:14:37.367743
**GO/NO-GO:** GO

## Pilot Scope

Using pilot data from C2B (3 configs x 5 letters = **15 data points**) and C2A (5-letter PMI features).

**PMI aggregation:** Median of top-10 PMI tokens per letter (proxy for letter-category absorption pressure)

## Simplified Regression Model (Pilot)

```
absorption_rate = beta0 + beta1*log(L0) + beta4*log(PMI)
```

| Parameter | Value |
|-----------|-------|
| beta0 (intercept) | 0.0854 |
| beta1 (log L0) | -0.0123 |
| beta4 (log PMI) | 0.0360 |
| p-value (beta4) | 0.2392 |
| R² | 0.0699 |
| Adj-R² | -0.08512031565714429 |
| SE type | HC3 |

## Correlation Analysis

| Metric | Value |
|--------|-------|
| Pearson r(log_PMI, absorption) | 0.2522 |
| Pearson p-value | 0.3645 |
| Spearman r | 0.2288 |

## Partial Regression (absorption vs PMI | L0)

| Metric | Value |
|--------|-------|
| Partial r | 0.2530 |
| Partial R² | 0.0640 |
| Partial p | 0.3629 |

## Pass Criteria

| Criterion | Status |
|-----------|--------|
| Regression executes without error | PASS |
| beta4 (PMI coefficient) positive | PASS (0.036) |
| R² is finite | PASS (0.0699) |

## Caveats

- **15 data points is insufficient for reliable statistical inference** — this is a pipeline check only
- The full regression (C2C full run) requires C2B full data (30 SAE configs x 26 letters = 780 data points)
- PMI aggregation (median of top-10 tokens) is a proxy; the full run will use the canonical per-letter PMI from the sae-spelling feature labels
- With n=15 and k=3 parameters, degrees of freedom = 12; all p-values are unreliable

## Runtime

0.0 seconds
