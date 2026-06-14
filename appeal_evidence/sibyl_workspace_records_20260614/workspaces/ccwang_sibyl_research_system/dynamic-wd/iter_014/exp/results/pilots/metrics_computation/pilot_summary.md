# Metrics Computation Pilot Summary

**Recommendation**: GO
**Timestamp**: 2026-03-25T15:25:58.743269

## Pass Criteria
- [PASS] BEM_computable
- [PASS] CSI_computable
- [PASS] UDWDC_v2_CSI_positive
- [PASS] AIS_computable
- [PASS] all_pass

## Key Findings

- BEM computed for 14/16 method-benchmark pairs
- CSI computed for 8/8 methods on CIFAR-10
- AIS = 0.123427, Spearman rho = 0.195046
- CIFAR-10: Max rank shift between accuracy and BEM = 2
- ImageNet: Max rank shift between accuracy and BEM = 1
- UDWDC-v2 CSI is POSITIVE on CIFAR-10 (stability fix verified)
- UDWDC-v2 WD budget > 0 on ImageNet (no controller collapse)
- Budget confound: 0/6 methods show accuracy within 2% of budget-matched FixedWD

## CIFAR-10 Rankings

| Method | Acc Rank | BEM Rank | Shift |
|--------|----------|----------|-------|
| FixedWD | 1 | 2 | -1 |
| UDWDC | 2 | 1 | 1 |
| CWD | 3 | 3 | 0 |
| UDWDC-v2 | 4 | 4 | 0 |
| NoWD | 5 | - | - |
| DefazioCorrective | 6 | 6 | 0 |
| CPR | 7 | 5 | 2 |
| SWD | 8 | 7 | 1 |

## UDWDC-v2 Stability Fix

### cifar10
- UDWDC_acc: 81.78
- UDWDC-v2_acc: 81.26
- UDWDC_CSI_rho: 0.738626
- UDWDC-v2_CSI_rho: 0.746003
- v2_csi_positive: True
### imagenet
- UDWDC_acc: 0.06
- UDWDC-v2_acc: 0.16
- UDWDC_wd_budget: 0.016
- UDWDC-v2_wd_budget: 0.02926283
- UDWDC-v2_cumulative_v2: 1685.6444236417922
- v2_wd_positive: True