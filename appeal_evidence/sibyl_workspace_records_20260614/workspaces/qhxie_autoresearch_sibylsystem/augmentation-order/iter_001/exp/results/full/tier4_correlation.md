# Tier 4: NC Correlation and DPI Validation (Pilot)

Generated: 2026-04-02T18:36:16.326589

## Hypothesis Verdict Table

| Hypothesis | Name | Metric | Observed | Verdict |
|-----------|------|--------|----------|---------|
| H1 | Ordering Matters | spread > 0.5% in ≥50% blocks | 3/4 blocks above threshold | **confirmed** |
| H2 | Reversibility-Sorted Wins | CJ→Flip→Crop > Crop→Flip→CJ in ≥50% blocks | 2/4 blocks confirmed | **confirmed** |
| H3 | NC_2 Predicts Accuracy | Spearman rho(NC_2, accuracy) > 0.3 | rho=-0.2, p_perm=0.6946 | **falsified** |
| H4 | MI Predicts Accuracy | Spearman rho(MI, accuracy) > 0.3 | combined_rho=-0.0571 | **inconclusive** |
| H5 | Magnitude Amplifies Spread | spread[M14] > spread[M5] | {'M5': 0.0035, 'M9': 0.0088, 'M14': 0.0} | **falsified** |

## H3: NC_2 vs Accuracy Correlation

- **Spearman rho**: -0.2
- **p-value (approx)**: 0.6831
- **p-value (permutation)**: 0.6946
- **n_orderings**: 6
- **Verdict**: falsified

## H4: MI vs Accuracy Correlation

- **Combined Spearman rho**: -0.0571
  - cifar10: rho=0.5429, p_perm=0.2814
  - cifar100: rho=-0.6571, p_perm=0.1876
- **Verdict**: inconclusive

## H5: Magnitude Interaction

- **Spread by magnitude**: {'M5': 0.0035, 'M9': 0.0088, 'M14': 0.0}
- **Spread increases M5→M14**: False
- **Verdict**: falsified

## Pass Criteria

- H3 complete (rho + p-value computed): True
- H4 complete (combined rho computed): True
- **Overall pass**: True