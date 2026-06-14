# Rate-Distortion Predictor Validation (FULL MODE) -- H9

## Result: NOT_SUPPORTED (HIGH confidence)

## Key Findings

### Primary Analysis (L24_16k, 131 pairs)
- **Three-factor model**: rho=0.286 (p=9.41e-04)
- **LOO CV**: rho=0.192 (p=2.83e-02)
- **5-fold CV**: rho=0.236
- **R-squared**: 0.1041 (10.4% variance explained)
- **Best individual predictor**: r_parent (rho=-0.239)

### Pooled Analysis (L24_16k + L24_65k, 262 pairs)
- **Three-factor model**: rho=0.250
- **LOO CV**: rho=0.205

### Individual Predictor Correlations (L24_16k)
| Predictor | Spearman rho | p-value | Bootstrap 95% CI |
|-----------|-------------|---------|------------------|
| cos_sim | -0.090 | 0.3068 | [-0.273, 0.090] |
| cos_sim_squared | -0.090 | 0.3068 | [-0.273, 0.090] |
| co_occur | -0.189 | 0.0308 | [-0.358, -0.012] |
| r_parent | -0.239 | 0.0060 | [-0.395, -0.071] |
| neg_r_parent | 0.239 | 0.0060 | [0.071, 0.395] |
| competition_coeff | -0.159 | 0.0705 | [-0.327, 0.014] |

### Cross-Domain Predictor Distributions
- **city-continent** (n=6): absorption=0.324, cos_sim=0.282, co_occur=0.413, r_parent=-0.637
- **city-country** (n=77): absorption=0.703, cos_sim=0.331, co_occur=0.417, r_parent=-1.865
- **city-language** (n=23): absorption=0.277, cos_sim=0.355, co_occur=0.558, r_parent=-0.600
- **first-letter** (n=25): absorption=0.420, cos_sim=0.132, co_occur=1.000, r_parent=-0.221

### Interpretation
The three-factor rate-distortion model does NOT meaningfully predict per-pair absorption rates. Model Spearman rho=0.286 (p=9.41e-04), LOO CV rho=0.192 (p=2.83e-02), 5-fold CV rho=0.236, 10-fold CV rho=0.233. R^2=0.104 (only 10.4% variance explained). Best individual predictor: r_parent (rho=-0.239, p=6.015e-03). The cross-domain characterization stands independently: interventional methods (activation patching) remain necessary for absorption detection. This is the THIRD negative predictive result alongside GAS (rho=0.116) and CMI (rho=0.044).

## Comparison with Prior Results
- **iter_009**: model rho=0.24981249497343308, n=262
- **iter_010 (16k)**: model rho=0.286, n=131
- **iter_010 (pooled)**: model rho=0.250, n=262

## Elapsed: 1.0 minutes
