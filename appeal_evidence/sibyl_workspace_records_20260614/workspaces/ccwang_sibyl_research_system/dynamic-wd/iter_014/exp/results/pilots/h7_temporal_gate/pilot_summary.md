# H7 Temporal Predictability Gate — Pilot Summary (Updated)

## Task
Fit degree-4 polynomials to alpha_t (gradient-weight alignment) as a function of epoch
for each layer and each run. Compute R-squared. H7 is falsified if R^2 > 0.85 in >= 70%
of layer-runs (alignment is just a time proxy).

## Data Sources
- **diagnostic_cifar10**: 8 methods x 65 layers = 520 combinations (ResNet-20/CIFAR-10)
- **ablation_cifar100**: 14 variants x 45 layers = 630 combinations (VGG-16-BN/CIFAR-100)
- Total: 1150 layer-method combinations across 21 distinct methods
- Includes both original UDWDC and UDWDC-v2 variants

## Gate Decision: **PASS**

Only **11.8%** of layer-method combinations have R^2 > 0.85, well below the 70% threshold.
The alignment signal carries non-trivial information beyond a simple time proxy.

## Key Statistics
| Metric | Value |
|--------|-------|
| Mean R^2 | 0.5285 |
| Median R^2 | 0.5255 |
| Std R^2 | 0.2419 |
| Min R^2 | 0.0287 |
| Max R^2 | 0.9961 |
| % R^2 > 0.85 | 11.8% |

## Per-Method Findings (Sorted by Mean R^2)
| Method | Mean R^2 | %>0.85 | N |
|--------|----------|--------|---|
| UDWDC_v2_PI_control | 0.4325 | 8.9% | 45 |
| UDWDC_v2_Ki_only | 0.4495 | 0.0% | 45 |
| CPR | 0.4791 | 3.1% | 65 |
| UDWDC_v2_PD_control | 0.4832 | 8.9% | 45 |
| UDWDC_Full_PID | 0.4881 | 2.2% | 45 |
| NoWD | 0.4915 | 1.5% | 65 |
| SWD | 0.4993 | 6.2% | 65 |
| UDWDC_Kd_only | 0.5201 | 15.6% | 45 |
| UDWDC_Ki_only | 0.5245 | 15.6% | 45 |
| UDWDC_v2_Full_PID | 0.5298 | 20.0% | 45 |
| UDWDC_v2_UDWDC_v2 | 0.5298 | 20.0% | 45 |
| UDWDC_Kp_only | 0.5313 | 20.0% | 45 |
| UDWDC_PD_control | 0.5313 | 20.0% | 45 |
| CWD | 0.5341 | 3.1% | 65 |
| UDWDC_v2_Kd_only | 0.5441 | 6.7% | 45 |
| UDWDC | 0.5449 | 7.7% | 65 |
| UDWDC-v2 | 0.5449 | 10.8% | 65 |
| FixedWD | 0.5696 | 19.1% | 110 |
| DefazioCorrective | 0.5818 | 20.0% | 65 |
| UDWDC_v2_Kp_only | 0.6062 | 15.6% | 45 |
| UDWDC_PI_control | 0.6344 | 26.7% | 45 |

## Layer Type Insights
- **conv/weight**: Highest mean R^2 (0.6914), 26.3% > 0.85 — most time-predictable
- **bn/bias**: mean R^2 = 0.4791, only 2.0% > 0.85
- **bn/weight**: mean R^2 = 0.4419, only 0.7% > 0.85 — alignment carries most independent info
- **fc/weight**: Lowest mean R^2 (0.3747), only 1.0% > 0.85

## Key Observations
1. **Gate clearly PASSES**: 11.8% is far below the 70% threshold
2. **v2 variants generally show lower temporal predictability** than original UDWDC variants
   (e.g., UDWDC_v2_Ki_only: 0% > 0.85 vs UDWDC_Ki_only: 15.6%)
3. **Conv layers are most time-predictable** — their alignment smoothly evolves with epoch
4. **FC and BN layers carry the most non-temporal alignment information**
5. **Methods with integral control (PI, Ki) show divergent behavior**: original PI is most
   time-predictable (0.6344), while v2_PI is among the least (0.4325)

## Caveats (Pilot-Specific)
1. Only 10 epochs (pilot mode) — degree-4 polynomial fit with 5 parameters on 10 data points risks overfitting. Full run with 200 epochs will be more informative.
2. Single seed (42) only — no variance estimate across seeds.
3. The high R^2 layers tend to have very small alpha_std (< 0.001), suggesting near-constant alignment that is trivially polynomial-fittable.
4. Updated with UDWDC-v2 and new v2 ablation variants.

## Recommendation
**Proceed with direct alignment comparisons** (no need to use residual alignment). The alignment signal is not merely a time proxy for most layers and methods.

## Pass Criteria Check
- [x] Polynomial fitting completes for all layer-method combinations
- [x] R^2 distribution is computable (1150 valid values)
- [x] R^2 < 0.85 for > 30% of layer-method combinations (88.2% have R^2 < 0.85)
