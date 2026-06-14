# Compute-Fair Comparison — Countdown-16 (PILOT)

**Verdict: NO-GO**

## Compute-Fair Head-to-Head

| Method | FLOPs | Accuracy | Fair Vanilla | Delta | Wins |
|--------|-------|----------|-------------|-------|------|
| BSD (~1.1x) | 1.1x | 6.2% | 12.5% | -6.2pp | 0-1 |
| ReMDM-conf (~1.5x) | 1.5x | 0.0% | 6.2% | -6.2pp | 0-1 |
| A-CFG (~2.0x) | 2.0x | 0.0% | 12.5% | -12.5pp | 0-2 |
| BSD+ACFG (~2.1x) | 2.1x | 6.2% | 6.2% | +0.0pp | 1-1 |
| DMI (~1.05x) | 1.05x | 0.0% | 12.5% | -12.5pp | 0-2 |

## Pareto Frontier (Accuracy vs Compute)

| Method | FLOPs | Accuracy |
|--------|-------|----------|
| vanilla_1x* | 1.0x | 6.2% |
| dmi | 1.1x | 0.0% |
| vanilla_1.1x* | 1.1x | 12.5% |
| bsd | 1.1x | 6.2% |
| vanilla_1.5x | 1.5x | 6.2% |
| remdm_conf | 1.5x | 0.0% |
| vanilla_2.0x | 2.0x | 12.5% |
| acfg | 2.0x | 0.0% |
| vanilla_2.1x | 2.1x | 6.2% |
| bsd_acfg_combo | 2.1x | 6.2% |

Pareto-optimal: vanilla_1x, vanilla_1.1x

## All Methods Summary

| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |
|--------|----------|-------|-------|------------|----------|-------|
| vanilla_1x | 6.2% (1/16) | 0.160 | 0.095 | 0.865 | 3.7s | 1.0x |
| dmi | 0.0% (0/16) | 0.016 | 0.000 | 0.442 | 3.8s | 1.1x |
| vanilla_1.1x | 12.5% (2/16) | 0.116 | 0.066 | 0.901 | 4.0s | 1.1x |
| bsd | 6.2% (1/16) | 0.077 | 0.035 | 0.903 | 3.8s | 1.1x |
| vanilla_1.5x | 6.2% (1/16) | 0.132 | 0.051 | 0.896 | 5.5s | 1.5x |
| remdm_conf | 0.0% (0/16) | 0.059 | 0.026 | 0.919 | 3.8s | 1.5x |
| vanilla_2.0x | 12.5% (2/16) | 0.072 | 0.032 | 0.894 | 7.4s | 2.0x |
| acfg | 0.0% (0/16) | 0.120 | 0.061 | 0.896 | 7.4s | 2.0x |
| vanilla_2.1x | 6.2% (1/16) | 0.141 | 0.077 | 0.865 | 7.7s | 2.1x |
| bsd_acfg_combo | 6.2% (1/16) | 0.080 | 0.042 | 0.883 | 6.5s | 2.1x |

## Runtime

- Total: 14.3 minutes
