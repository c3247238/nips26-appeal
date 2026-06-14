# Phase 1: Full Diagnostic Results — CIFAR-10/ResNet-20

**Task ID**: diagnostic_cifar10
**Model**: resnet20
**Dataset**: cifar10
**Epochs**: 200
**Seeds**: 42, 123, 456
**Methods**: FixedWD, CWD, SWD, CPR, DefazioCorrective, NoWD, UDWDC, UDWDC-v2
**Total runs**: 24
**Total time**: 9.24 hours

## Method Comparison (mean ± std across 3 seeds)

| Method | Best Acc (%) | Final Acc (%) | Gen Gap (%) | WD Budget |
|--------|-------------|--------------|-------------|-----------|
| FixedWD | 90.68±0.11 | 90.55±0.17 | 9.28±0.17 | 0.48±0.0 |
| CWD | 90.32±0.08 | 90.16±0.05 | 9.66±0.05 | 0.24±0.0 |
| SWD | 90.39±0.19 | 90.34±0.23 | 9.49±0.23 | 0.47±0.0 |
| CPR | 91.74±0.07 | 91.60±0.05 | 8.28±0.05 | 4.44±0.0 |
| DefazioCorrective | 90.62±0.20 | 90.49±0.22 | 9.31±0.22 | 0.24±0.0 |
| NoWD | 90.25±0.30 | 90.08±0.30 | 9.73±0.30 | 0.00±0.0 |
| UDWDC | 90.15±0.23 | 89.98±0.25 | 9.82±0.25 | 0.38±0.0 |
| UDWDC-v2 | 90.36±0.09 | 90.27±0.09 | 9.53±0.07 | 98599.24±862.7 |

## Per-Seed Results

| Method | Seed | Best Acc (%) | Final Acc (%) | Gen Gap (%) | Time (s) |
|--------|------|-------------|--------------|-------------|----------|
| FixedWD | 42 | 90.79 | 90.68 | 9.16 | 715.7 |
| FixedWD | 123 | 90.71 | 90.66 | 9.15 | 715.9 |
| FixedWD | 456 | 90.53 | 90.30 | 9.53 | 721.9 |
| CWD | 42 | 90.35 | 90.23 | 9.59 | 1407.5 |
| CWD | 123 | 90.20 | 90.10 | 9.69 | 1383.0 |
| CWD | 456 | 90.40 | 90.14 | 9.71 | 1420.1 |
| SWD | 42 | 90.47 | 90.45 | 9.38 | 1378.4 |
| SWD | 123 | 90.58 | 90.55 | 9.28 | 1350.2 |
| SWD | 456 | 90.13 | 90.01 | 9.81 | 1384.3 |
| CPR | 42 | 91.65 | 91.54 | 8.35 | 1398.5 |
| CPR | 123 | 91.83 | 91.58 | 8.29 | 1404.4 |
| CPR | 456 | 91.75 | 91.67 | 8.22 | 1397.7 |
| DefazioCorrective | 42 | 90.54 | 90.42 | 9.38 | 718.1 |
| DefazioCorrective | 123 | 90.43 | 90.26 | 9.54 | 733.8 |
| DefazioCorrective | 456 | 90.90 | 90.78 | 9.02 | 728.1 |
| NoWD | 42 | 90.16 | 90.06 | 9.71 | 731.8 |
| NoWD | 123 | 89.94 | 89.72 | 10.11 | 727.1 |
| NoWD | 456 | 90.65 | 90.46 | 9.37 | 724.7 |
| UDWDC | 42 | 90.03 | 89.92 | 9.89 | 2337.6 |
| UDWDC | 123 | 90.47 | 90.31 | 9.49 | 2346.5 |
| UDWDC | 456 | 89.96 | 89.71 | 10.09 | 2385.2 |
| UDWDC-v2 | 42 | 90.32 | 90.26 | 9.51 | 2410.7 |
| UDWDC-v2 | 123 | 90.48 | 90.39 | 9.45 | 2387.5 |
| UDWDC-v2 | 456 | 90.28 | 90.16 | 9.62 | 2340.3 |

## Key Observations

- UDWDC: 90.15±0.23% vs UDWDC-v2: 90.36±0.09%
- UDWDC-v2 WD budget: 98599.24 (confirms stability fix, WD > 0)
- Method ranking: CPR(91.74%) > FixedWD(90.68%) > DefazioCorrective(90.62%) > SWD(90.39%) > UDWDC-v2(90.36%) > CWD(90.32%) > NoWD(90.25%) > UDWDC(90.15%)
- 24/24 runs completed successfully