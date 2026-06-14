# Pilot Steering Results (v4 - Reconstruction Fidelity)

## Configuration
- Model: gpt2
- Layer: 8
- Method: SAE reconstruction fidelity analysis

## Summary
- Letters tested: 26
- HIGH absorption: 7
- LOW absorption: 19

## Reconstruction MSE
| Absorption Class | Mean MSE | Std MSE |
|------------------|----------|---------|
| HIGH | 2939454976.0000 | 0.0000 |
| LOW | 2939454970.6105 | 15.7129 |

## Cosine Similarity (Original vs Reconstruction)
| Absorption Class | Mean Cos | Std Cos |
|------------------|----------|---------|
| HIGH | -0.5890 | 0.0000 |
| LOW | -0.5890 | 0.0000 |

## Main Feature Activation
| Absorption Class | Mean Act | Std Act |
|------------------|----------|---------|
| HIGH | 1163.6 | 346.2 |
| LOW | 1376.5 | 307.0 |

## Main Feature in Top-32 Rate
| Absorption Class | Rate |
|------------------|------|
| HIGH | 1.000 |
| LOW | 1.000 |

## Statistical Tests (HIGH vs LOW)
- MSE: t=0.872, p=0.392
- Cosine: t=0.872, p=0.392
- Activation: t=-1.454, p=0.159

## Correlations with Absorption Rate
- MSE: r=0.175
- Cosine: r=0.175
- Activation: r=-0.285

## GO/NO-GO Criteria
- Detectable MSE diff: PASS (5.3895)
- Detectable cosine diff: FAIL (0.0000)
- Detectable activation diff: PASS (212.9)

## Overall: GO

## Interpretation
This experiment tests whether absorbed features are less well-represented in the
SAE reconstruction. If absorption causes the main feature to be "replaced" by child
features in the reconstruction, we should see:
1. Higher MSE for HIGH absorption features (reconstruction misses the main feature)
2. Lower cosine similarity for HIGH absorption features
3. Lower main feature activation for HIGH absorption features

Note: In GPT-2 Small, first-letter features are weak and many letters share features,
which limits the discriminative power of this analysis.
