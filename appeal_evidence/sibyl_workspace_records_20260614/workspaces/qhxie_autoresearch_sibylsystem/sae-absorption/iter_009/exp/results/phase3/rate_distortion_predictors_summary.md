# Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9) -- FULL

**Mode**: FULL
**Time**: 0.6 minutes
**SAEs**: L24_16k, L24_65k
**Total pairs**: 262
**Valid pairs**: 262

## Pairs Breakdown

### By Hierarchy
- **city-continent**: 12 pairs
- **city-country**: 154 pairs
- **city-language**: 46 pairs
- **first-letter**: 50 pairs

### By SAE
- **L24_16k**: 131 pairs
- **L24_65k**: 131 pairs

## Individual Predictor Correlations (All Pairs)

| Predictor | Spearman rho | p-value | Pearson r | n |
|-----------|-------------|---------|-----------|---|
| r_parent | -0.2032 | 0.0009 | -0.2468 | 262 |
| neg_r_parent | 0.2032 | 0.0009 | 0.2468 | 262 |
| co_occur | -0.1730 | 0.0050 | -0.1646 | 262 |
| competition_coeff | -0.1691 | 0.0061 | -0.1631 | 262 |
| cos_sim | -0.1081 | 0.0806 | -0.1119 | 262 |
| cos_sim_squared | -0.1081 | 0.0806 | -0.1395 | 262 |
| cos_sim_abs | -0.1081 | 0.0806 | -0.1119 | 262 |

## Per-Hierarchy Predictor Correlations

### city-language
| Predictor | Spearman rho | p-value | n |
|-----------|-------------|---------|---|
| competition_coeff | -0.2216 | 0.1389 | 46 |
| cos_sim | -0.1674 | 0.2663 | 46 |
| cos_sim_squared | -0.1674 | 0.2663 | 46 |
| cos_sim_abs | -0.1674 | 0.2663 | 46 |
| co_occur | -0.1587 | 0.2920 | 46 |
| r_parent | -0.1298 | 0.3901 | 46 |
| neg_r_parent | 0.1298 | 0.3901 | 46 |

### city-country
| Predictor | Spearman rho | p-value | n |
|-----------|-------------|---------|---|
| cos_sim | -0.3489 | 0.0000 | 154 |
| cos_sim_squared | -0.3489 | 0.0000 | 154 |
| cos_sim_abs | -0.3489 | 0.0000 | 154 |
| r_parent | -0.2631 | 0.0010 | 154 |
| neg_r_parent | 0.2631 | 0.0010 | 154 |
| competition_coeff | -0.1908 | 0.0178 | 154 |
| co_occur | -0.1375 | 0.0890 | 154 |

### first-letter
| Predictor | Spearman rho | p-value | n |
|-----------|-------------|---------|---|
| cos_sim | -0.2235 | 0.1187 | 50 |
| cos_sim_squared | -0.2235 | 0.1187 | 50 |
| cos_sim_abs | -0.2235 | 0.1187 | 50 |
| competition_coeff | -0.2235 | 0.1187 | 50 |
| r_parent | 0.0723 | 0.6177 | 50 |
| neg_r_parent | -0.0723 | 0.6177 | 50 |

### city-continent
| Predictor | Spearman rho | p-value | n |
|-----------|-------------|---------|---|
| co_occur | 0.8213 | 0.0011 | 12 |
| competition_coeff | 0.5990 | 0.0396 | 12 |
| r_parent | 0.5947 | 0.0414 | 12 |
| neg_r_parent | -0.5947 | 0.0414 | 12 |
| cos_sim | -0.2207 | 0.4907 | 12 |
| cos_sim_squared | -0.2207 | 0.4907 | 12 |
| cos_sim_abs | -0.2207 | 0.4907 | 12 |

## Bootstrap 95% CI on Spearman Correlations

| Predictor | Mean rho | CI lower | CI upper | std |
|-----------|----------|----------|----------|-----|
| cos_sim | -0.1072 | -0.2327 | 0.0198 | 0.0648 |
| cos_sim_squared | -0.1072 | -0.2327 | 0.0198 | 0.0648 |
| co_occur | -0.1735 | -0.2958 | -0.0505 | 0.0627 |
| r_parent | -0.2030 | -0.3179 | -0.0811 | 0.0603 |

## Three-Factor Linear Model

**Formula**: absorption ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*r_parent

- **R^2** = 0.0877
- **Spearman rho** = 0.2498 (p = 0.0000)
- **LOO-CV Spearman rho** = 0.2055 (p = 0.0008)
- **LOO-CV MSE** = 0.125747

### Coefficients

- **cos_sim_squared**: -0.483241
- **co_occur**: -0.093413
- **r_parent**: -0.023918
- **intercept**: 0.561017

## Cross-Domain Analysis

| Hierarchy | n | Mean Absorption | Std | Mean cos_sim | Mean co_occur | Mean R_parent |
|-----------|---|-----------------|-----|-------------|---------------|---------------|
| city-language | 46 | 0.2348 | 0.2800 | 0.3632 | 0.5577 | -0.600384 |
| city-country | 154 | 0.6064 | 0.3391 | 0.3475 | 0.4262 | -1.864674 |
| first-letter | 50 | 0.3779 | 0.3624 | 0.1331 | 1.0000 | -0.221146 |
| city-continent | 12 | 0.3243 | 0.3115 | 0.2849 | 0.4129 | -0.636608 |

**ANOVA**: F=17.8637, p=0.0000

## H9 Verdict

**Verdict**: NOT_SUPPORTED
**Confidence**: 0.79
**Evaluation source**: multivariate_model_loo_cv
**rho used**: 0.2055 (p = 0.0008)
**Best individual predictor**: r_parent (rho=-0.2032)
**Target**: rho > 0.5
**Falsification**: rho < 0.3 or p > 0.05
