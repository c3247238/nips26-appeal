# Pilot (v2): Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN

**Recommendation: GO**

- Epochs: 10 (pilot), Seed: 42
- Model: vgg16_bn, Dataset: cifar100
- **KEY FIX**: All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)
- Total time: 2313.9s (38.6min)

## Pass Criteria

- PASS: all_8_variants_complete
- PASS: no_nan_inf
- PASS: all_above_20pct
- PASS: all_wd_budget_positive

## Results

| Variant | K_p | K_i | K_d | Test Acc (%) | Train Acc (%) | Gap (%) | WD Budget | v2 Budget | Time (s) |
|---------|-----|-----|-----|-------------|--------------|---------|-----------|-----------|----------|
| FixedWD | 0.0 | 0.0 | 0.0 | 28.08 | 24.90 | -3.18 | 0.0290 | None | 44.4 |
| Kp_only | 0.5 | 0.0 | 0.0 | 25.27 | 20.75 | -4.52 | 0.0029 | 541.9806 | 137.4 |
| Ki_only | 0.0 | 0.1 | 0.0 | 26.85 | 21.87 | -4.98 | 0.0089 | 703.2679 | 379.5 |
| Kd_only | 0.0 | 0.0 | 0.3 | 29.50 | 25.16 | -4.34 | 0.0415 | 6559.9357 | 374.7 |
| PI_control | 0.5 | 0.1 | 0.0 | 25.48 | 22.33 | -3.15 | 0.0039 | 554.4030 | 400.0 |
| PD_control | 0.5 | 0.0 | 0.3 | 25.82 | 22.35 | -3.47 | 0.0029 | 542.0546 | 383.2 |
| Full_PID | 0.5 | 0.1 | 0.3 | 26.62 | 23.34 | -3.28 | 0.0039 | 555.8175 | 393.7 |
| UDWDC_v2 | 0.5 | 0.1 | 0.3 | 26.62 | 23.34 | -3.28 | 0.0039 | 555.8175 | 199.7 |

## v1 vs v2 WD Budget Comparison

| Variant | v1 Budget | v2 Budget | Status |
|---------|-----------|-----------|--------|
| FixedWD | 0.029 | 0.029000 | OK |
| Kp_only | 0.0 | 0.002900 | FIXED |
| Ki_only | 0.005 | 0.008909 | OK |
| Kd_only | 0.041 | 0.041539 | OK |
| PI_control | 0.001 | 0.003890 | OK |
| PD_control | 0.0 | 0.002900 | FIXED |
| Full_PID | 0.001 | 0.003890 | OK |
