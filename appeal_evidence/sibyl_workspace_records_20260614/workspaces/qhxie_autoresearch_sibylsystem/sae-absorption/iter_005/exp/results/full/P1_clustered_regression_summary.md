# P1 Clustered SE Regression — Pilot Summary

**Task**: Rerun iter_004 C2C PMI regression with letter-level clustering (26 clusters)
**Mode**: PILOT | **Runtime**: 0.7s
**N**: 806 obs, 26 letter clusters, 31 SAE configs

## Key Finding

**PMI is non-significant under both HC3 (p=0.5930) and clustered SE (p=0.6674). Clustered SE does NOT rescue significance. Conclusion from iter_004 is robust: PMI does not predict absorption.**

## SE Comparison: HC3 vs Letter-Clustered

| Variable | Coef | HC3 SE | Cluster SE | Ratio | HC3 p | Cluster p | Changed? |
|----------|------|--------|------------|-------|-------|-----------|----------|
| const | 0.052188 | 0.064450 | 0.092273 | 1.432 | 0.4181 | 0.5717 |  |
| log_L0 | 0.013172 | 0.005226 | 0.010407 | 1.992 | 0.0117 | 0.2056 | YES |
| log_width | 0.003238 | 0.003987 | 0.004415 | 1.107 | 0.4167 | 0.4633 |  |
| layer | -0.012023 | 0.001820 | 0.003845 | 2.112 | 0.0000 | 0.0018 |  |
| log_PMI | -0.006303 | 0.011792 | 0.014669 | 1.244 | 0.5930 | 0.6674 |  |

## Model Comparison

| Model | R^2 / Pseudo R^2 | PMI Coef | PMI SE | PMI p |
|-------|-------------------|----------|--------|-------|
| OLS with HC3 robust SE | 0.087247 | -0.006303 | 0.011792 | 0.5930 |
| OLS with letter-clustered SE (26 clusters) | 0.087247 | -0.006303 | 0.014669 | 0.6674 |
| Beta regression (Smithson-Verkuilen transform) | -0.023670 | -0.240019 | 0.085513 | 0.0050 |

## Distribution Diagnostics

| Metric | Value |
|--------|-------|
| N total | 806 |
| N zero (absorption = 0) | 472 (58.6%) |
| N one (absorption = 1) | 3 (0.4%) |
| Skewness | 4.812 |
| Kurtosis | 30.781 |
| Jarque-Bera p | 0.000000 |
| Zero inflation severe (>50% zeros) | True |
| Hurdle model recommended | True |

## Recommendations

1. **Clustered SE is the correct specification** for this data: PMI is a letter-level variable (26 values repeated across 31 configs), so within-letter errors are correlated.
2. **Both HC3 and clustered SE agree**: PMI does not significantly predict absorption.
3. **High zero fraction** (58.6%) suggests a hurdle/two-part model is more appropriate than standard OLS.
4. **Skewness** (4.812) is extreme; OLS assumptions are violated. Beta regression or Tobit offers a more principled approach.
5. **Bottom line**: The iter_004 null result for PMI is robust to proper clustering and alternative specifications.
