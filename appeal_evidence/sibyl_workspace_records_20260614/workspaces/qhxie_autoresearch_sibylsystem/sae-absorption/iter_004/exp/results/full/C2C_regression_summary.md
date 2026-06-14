# C2C PMI Regression Analysis (Full) — Summary

**Mode**: FULL  |  **Model**: GPT-2 Small (open-model anchor)
**Timestamp**: 2026-04-14T22:46:14.935572
**N observations**: 806 (31 SAE configs x 26 letters)
**Runtime**: 1.4 seconds

## H2 Assessment

**NOT SUPPORTED: PMI coefficient is negative (-0.0063)**

## Full Regression Model

```
absorption_rate = b0 + b1*log(L0) + b2*log(width) + b3*layer + b4*log(PMI)
Standard errors: HC3 (heteroskedasticity-robust)
```

| Coefficient | Value | SE | t | p-value | 95% CI |
|------------|-------|-----|---|---------|--------|
| const | 0.052188 | 0.064450 | 0.810 | 0.4181 | [-0.0741, 0.1785] |
| log(L0) | 0.013172 | 0.005226 | 2.521 | 0.0117 | [0.0029, 0.0234] |
| log(width) | 0.003238 | 0.003987 | 0.812 | 0.4167 | [-0.0046, 0.0111] |
| layer | -0.012023 | 0.001820 | -6.606 | 0.0000 | [-0.0156, -0.0085] |
| **log(PMI)** | **-0.006303** | 0.011792 | -0.535 | **0.5930** | [-0.0294, 0.0168] |

| Metric | Value |
|--------|-------|
| R^2 | 0.087247 |
| Adj R^2 | 0.082688 |
| F-statistic | 42.9189 |
| AIC | -1406.32 |

## Partial R^2 for PMI

| Metric | Value |
|--------|-------|
| R^2 (full model) | 0.087247 |
| R^2 (without PMI) | 0.086657 |
| **Partial R^2 (PMI)** | **0.000646** |
| Partial r (PMI \| config) | -0.025414 |
| Partial p-value | 0.471217 |

## Correlation Analysis

| Measure | r | p-value |
|---------|---|---------|
| Raw Pearson (log_PMI vs absorption, all cells) | -0.0243 | 0.4911 |
| Raw Spearman (log_PMI vs absorption, all cells) | -0.1254 | 0.0004 |
| Per-letter Pearson (mean absorption vs log_PMI) | -0.0796 | 0.6991 |
| Per-letter Spearman | 0.0516 | 0.8022 |

## Ablation A3: PMI-Only Model

| Metric | Value |
|--------|-------|
| R^2 | 0.000590 |
| beta_PMI | -0.006303 |
| p-value | 0.5987 |

## Ablation A4: Per-Layer PMI Coefficient Stability

| Layer | beta_PMI | SE | p-value | Sign |
|-------|----------|-----|---------|------|
| 3 | -0.022349 | 0.034576 | 0.5180 | negative |
| 4 | -0.087416 | 0.030619 | 0.0043 | negative |
| 5 | 0.027068 | 0.101305 | 0.7893 | positive |
| 6 | -0.061700 | 0.025403 | 0.0151 | negative |
| 7 | -0.008745 | 0.025924 | 0.7359 | negative |
| 8 | 0.034225 | 0.025382 | 0.1775 | positive |
| 9 | -0.007986 | 0.006050 | 0.1869 | negative |
| 10 | 0.000591 | 0.001319 | 0.6541 | positive |
| 11 | N/A | N/A | N/A | zero variance in absorption |

**Sign consistency across layers**: False

## Visualizations

- Partial regression plot: `C2C_plots/C2C_partial_regression_plot.png`
- Per-layer coefficient plot: `C2C_plots/C2C_per_layer_pmi_coef.png`
- Width effect at layer 8: `C2C_plots/C2C_width_effect_layer8.png`

## Key Findings

1. **PMI coefficient sign**: -0.006303 (negative — contrary to H2 prediction)
2. **Statistical significance**: p=0.5930 (not significant at alpha=0.05)
3. **Partial R^2 for PMI**: 0.0006 (below criterion of 0.10)
4. **Layer coefficient**: -0.012023 (negative — absorption decreases in later layers)
5. **Width coefficient**: 0.003238 (positive — wider SAEs show more absorption)

## Caveats

- Model is GPT-2 Small (not Gemma 2 2B as originally planned due to gated access)
- PMI aggregation uses median of top-10 tokens per letter as proxy for letter-category co-occurrence pressure
- The 31 SAE configs span 3 release families (res_jb, feature_splitting, resid_post, resid_mid) with different training procedures; this is a potential confound
- HC3 SEs do not account for clustering at the letter level (26 repeated measurements per letter); clustered SEs would be more conservative
