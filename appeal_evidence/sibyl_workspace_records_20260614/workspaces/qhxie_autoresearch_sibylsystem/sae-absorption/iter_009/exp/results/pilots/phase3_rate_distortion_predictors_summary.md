# Phase 3.1: Rate-Distortion Three-Factor Predictor Model (H9)

**Mode**: PILOT
**Status**: PASS
**Time**: 0.2 minutes
**SAE**: L24_16k
**Total pairs**: 20
**Valid pairs**: 20

## Pairs per Hierarchy

- **first-letter**: 14 pairs
- **city-continent**: 6 pairs

## Individual Predictor Correlations

| Predictor | Spearman rho | p-value | Pearson r | n |
|-----------|-------------|---------|-----------|---|
| cos_sim | -0.0472 | 0.8435 | -0.2214 | 20 |
| cos_sim_squared | -0.2492 | 0.2894 | -0.2483 | 20 |
| co_occur | 0.0412 | 0.8631 | 0.1345 | 20 |
| r_parent | 0.1666 | 0.4826 | 0.2890 | 20 |
| neg_r_parent | -0.1666 | 0.4826 | -0.2890 | 20 |
| competition_coeff | -0.0487 | 0.8383 | -0.1966 | 20 |

## Three-Factor Model

**Formula**: absorption ~ beta_1*cos_sim^2 + beta_2*co_occur + beta_3*r_parent

- R^2 = 0.1582
- Spearman rho = 0.2610 (p = 0.2664)
- Pearson r = 0.3977

### Coefficients

- **cos_sim_squared**: -1.821635
- **co_occur**: 0.118133
- **r_parent**: 0.069537
- **intercept**: 0.458440

### Predicted vs Observed

| Hierarchy | Class | Predicted | Observed | cos_sim | co_occur | R_parent |
|-----------|-------|-----------|----------|---------|----------|----------|
| first-letter | b | 0.4771 | 1.0000 | -0.0250 | 0.2697 | -0.173660 |
| first-letter | e | 0.4504 | 0.0000 | -0.0421 | 0.2740 | -0.534828 |
| first-letter | f | 0.6037 | 1.0000 | -0.0117 | 1.0000 | 0.394302 |
| first-letter | h | 0.6102 | 0.0000 | -0.0486 | 1.0000 | 0.545064 |
| first-letter | j | 0.6747 | 0.3333 | -0.0624 | 1.0000 | 1.513526 |
| first-letter | k | 0.4909 | 0.0000 | 0.0320 | 0.3998 | -0.185543 |
| first-letter | n | 0.4554 | 1.0000 | 0.0332 | 1.0000 | -1.714007 |
| first-letter | q | 0.1024 | 0.0000 | -0.0553 | 1.0000 | -6.739676 |
| first-letter | r | 0.4789 | 0.0000 | 0.0375 | 0.2006 | -0.009428 |
| first-letter | t | 0.4708 | 1.0000 | -0.0415 | 0.1945 | -0.107137 |
| first-letter | u | 0.5265 | 1.0000 | -0.0098 | 1.0000 | -0.716975 |
| first-letter | v | 0.5909 | 1.0000 | -0.0520 | 1.0000 | 0.276694 |
| first-letter | w | 0.4350 | 0.5000 | -0.0259 | 1.0000 | -2.018082 |
| first-letter | y | 0.5668 | 0.0000 | -0.0205 | 1.0000 | -0.130205 |
| city-continent | Africa | 0.0629 | 0.0000 | 0.4763 | 0.4649 | -0.533590 |
| city-continent | Asia | 0.4382 | 0.2222 | 0.2579 | 0.4000 | 0.771340 |
| city-continent | Europe | 0.5281 | 1.0000 | 0.2509 | 0.0000 | 2.651942 |
| city-continent | North America | 0.3554 | 0.1667 | 0.1766 | 0.0526 | -0.754366 |
| city-continent | Oceania | 0.2729 | 0.3333 | 0.2813 | 0.4545 | -1.367485 |
| city-continent | South America | 0.0643 | 0.1000 | 0.2512 | 0.0598 | -4.115994 |

## Cross-Domain Predictor Analysis

| Hierarchy | Mean Absorption | Mean cos_sim | Mean co_occur | Mean R_parent | Mean Competition |
|-----------|-----------------|-------------|---------------|---------------|------------------|
| first-letter | 0.4881 | -0.0209 | 0.7385 | -0.685711 | -0.0185 |
| city-continent | 0.3037 | 0.2824 | 0.2386 | -0.558025 | 0.0795 |

## H9 Verdict

**Verdict**: NOT_SUPPORTED
**Confidence**: 0.74
**Model Spearman rho**: 0.2610
**Best individual predictor**: cos_sim_squared (rho=-0.2492)
**Target**: rho > 0.5
**Falsification**: rho < 0.3 or p > 0.05

## Predictor Statistics

- **cos_sim**: mean=0.0701, std=0.1504
- **co_occur**: mean=0.5885, std=0.3904
- **r_parent**: mean=-0.6474, std=1.9372
