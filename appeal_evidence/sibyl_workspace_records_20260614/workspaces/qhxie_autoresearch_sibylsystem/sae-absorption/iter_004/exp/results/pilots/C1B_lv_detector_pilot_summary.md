# C1B LV Detector Validation — PILOT Summary

**GO/NO-GO: NO_GO**

## Configuration
- Model: GPT-2 Small (open-model anchor)
- SAE: gpt2-small-res-jb / blocks.8.hook_resid_pre (d_sae=24576)
- Calibration letters: A, B, C, D, E
- Test letters: F, G, H
- Tau values tested: [0.5, 0.75, 1.0, 1.25, 1.5]

## Absorption Rates
| Letter | Group | Absorption Rate | N Absorbed | N Total |
|--------|-------|-----------------|------------|---------|
| A | Calib | 0.040 | 2 | 50 |
| B | Calib | 0.160 | 8 | 50 |
| C | Calib | 0.040 | 2 | 50 |
| D | Calib | 0.060 | 3 | 50 |
| E | Calib | 0.140 | 7 | 50 |
| F | Test | 0.100 | 5 | 50 |
| G | Test | 0.100 | 5 | 50 |
| H | Test | 0.120 | 6 | 50 |

## Calibration Results
- Best tau: **0.5** (calib F1=0.158)

| Tau | Precision | Recall | F1 |
|-----|-----------|--------|----|
| 0.50 | 0.090 | 0.636 | 0.158 |
| 0.75 | 0.076 | 0.455 | 0.131 |
| 1.00 | 0.073 | 0.136 | 0.095 |
| 1.25 | 0.062 | 0.045 | 0.053 |
| 1.50 | 0.333 | 0.045 | 0.080 |

## Test Set Results (letters F-H)
- Precision: 0.109
- Recall:    0.688
- F1:        0.188
- ROC-AUC:   0.493

## LV Sharpness Diagnostic
- Sigmoid converged: True
- AIC (sigmoid): -67.396
- AIC (linear): -68.176
- Winner: linear

## Pass Criteria
- F1 > 0.35 on test set: FAIL (F1=0.188)
- Sigmoid fit converged: PASS
- AIC values finite: PASS