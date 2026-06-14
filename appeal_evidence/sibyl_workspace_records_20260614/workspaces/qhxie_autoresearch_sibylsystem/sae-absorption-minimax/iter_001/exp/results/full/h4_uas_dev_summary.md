# H4 UAS Development Results

## Task: h4_uas_dev
- Date: 2026-04-26T03:16:15.169337
- Layer: 8 (GPT-2 Small)
- Features analyzed: 100

## Pilot Pass Criteria
- **Target**: Combined UAS achieves r > 0.5 with Chanin absorption
- **Achieved**: r = 0.0460 (p = 6.49e-01)
- **Status**: FAIL

## Best UAS Configuration
- **Formula**: UAS(f) = 0.05 * cos_sim_variance(f) + 5.0 * freq_skewness(f)
- **Spearman r**: 0.0460
- **p-value**: 6.49e-01

## Top 10 Variants
| Rank | Formula | Spearman r | p-value |
|------|---------|------------|---------|
| 1 | alpha * cos_sim_var (alpha=0.1) | -0.4914 | 2.09e-07 |
| 2 | alpha * cos_sim_var (alpha=0.2) | -0.4914 | 2.09e-07 |
| 3 | alpha * cos_sim_var (alpha=0.5) | -0.4914 | 2.09e-07 |
| 4 | alpha * cos_sim_var (alpha=1.0) | -0.4914 | 2.09e-07 |
| 5 | alpha * cos_sim_var (alpha=2.0) | -0.4914 | 2.09e-07 |
| 6 | alpha * cos_sim_var (alpha=5.0) | -0.4914 | 2.09e-07 |
| 7 | alpha * cos_sim_var (alpha=10.0) | -0.4914 | 2.09e-07 |
| 8 | 5.0*cos_sim_var + 0.1*freq_skew | -0.4882 | 2.57e-07 |
| 9 | 2.0*cos_sim_var + 0.1*freq_skew | -0.4817 | 3.89e-07 |
| 10 | 1.0*cos_sim_var + 0.1*freq_skew | -0.4734 | 6.56e-07 |

## Analysis

### Variant A (cos_sim_variance only)
Best: r = -0.4914

### Variant B (freq_skewness only)
Best: r = 0.1588

### Variant C (Combined)
Best: alpha = 0.05, beta = 5.0, r = 0.0460

## Recommendation
Use the combined UAS formula with alpha = 0.05 and beta = 5.0 for all subsequent UAS reporting.
