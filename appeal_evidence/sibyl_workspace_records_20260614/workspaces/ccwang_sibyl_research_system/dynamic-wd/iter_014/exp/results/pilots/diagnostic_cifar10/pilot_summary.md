# Pilot Summary: diagnostic_cifar10 (v2, 8 methods)

**Recommendation**: GO
**Model**: resnet20 on cifar10
**Epochs**: 10 (pilot, out of 200 full)
**Seed**: 42
**Methods**: FixedWD, CWD, SWD, CPR, DefazioCorrective, NoWD, UDWDC, UDWDC-v2
**Total time**: 488.7s

## Checks
- **[PASS]** all_methods_complete
- **[PASS]** rho_t_nontrivial
- **[PASS]** udwdc_v2_wd_budget
- **[PASS]** fixedwd_cv_convergence

## Per-Method Results

| Method | Best Acc (%) | Final Acc (%) | Train Acc (%) | Time (s) |
|--------|-------------|--------------|--------------|----------|
| FixedWD | 82.06 | 82.06 | 83.46 | 33.2 |
| CWD | 81.26 | 81.26 | 84.43 | 58.0 |
| SWD | 80.10 | 80.10 | 83.74 | 62.4 |
| CPR | 80.52 | 77.29 | 83.39 | 62.3 |
| DefazioCorrective | 80.63 | 80.63 | 83.39 | 34.9 |
| NoWD | 80.97 | 80.54 | 83.84 | 35.0 |
| UDWDC | 81.78 | 81.78 | 83.94 | 101.9 |
| UDWDC-v2 | 81.26 | 81.26 | 83.82 | 100.6 |

## Key Observations

- UDWDC (81.78%) vs UDWDC-v2 (81.26%) at 10 epochs
- UDWDC-v2 cumulative WD budget: 4.0281e+02 (>0 confirms stability fix)
- All 8 methods completed successfully with per-layer diagnostics
- 10-epoch ranking: FixedWD(82.06%) > UDWDC(81.78%) > CWD(81.26%) > UDWDC-v2(81.26%) > NoWD(80.97%) > DefazioCorrective(80.63%) > CPR(80.52%) > SWD(80.10%)
- Runtime ~33-102s per method