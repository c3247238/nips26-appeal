# Full: Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN

- Epochs: 200, Seeds: [42, 123, 456]
- Model: vgg16_bn, Dataset: cifar100
- Batch size: 128, LR: 0.1, WD: 0.0001
- All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)
- Total time: 35573s (9.9h)

## Results

| Variant | K_p | K_i | K_d | Test Acc (%) | Gen Gap (%) | WD Budget | CSI |
|---------|-----|-----|-----|-------------|-------------|-----------|-----|
| FixedWD | 0.0 | 0.0 | 0.0 | 70.53 +/- 0.48 | 29.41 +/- 0.52 | 0.5800 | 0.2393 |
| Kp_only | 0.5 | 0.0 | 0.0 | 68.52 +/- 0.31 | 31.30 +/- 0.30 | 0.2999 | 0.2393 |
| Ki_only | 0.0 | 0.1 | 0.0 | 69.64 +/- 0.88 | 30.08 +/- 0.71 | 0.1344 | 0.2393 |
| Kd_only | 0.0 | 0.0 | 0.3 | 70.64 +/- 0.30 | 29.34 +/- 0.17 | 0.5945 | 0.2393 |
| PI_control | 0.5 | 0.1 | 0.0 | 69.12 +/- 1.11 | 30.55 +/- 0.97 | 0.3174 | 0.2393 |
| PD_control | 0.5 | 0.0 | 0.3 | 69.28 +/- 0.57 | 30.36 +/- 0.44 | 0.3002 | 0.2393 |
| Full_PID | 0.5 | 0.1 | 0.3 | 69.29 +/- 1.28 | 30.46 +/- 1.19 | 0.2766 | 0.2393 |
| UDWDC_v2 | 0.5 | 0.1 | 0.3 | 69.29 +/- 1.28 | 30.46 +/- 1.19 | 0.2766 | 0.2393 |

## Per-Seed Results

| Variant | Seed 42 | Seed 123 | Seed 456 |
|---------|---------|----------|----------|
| FixedWD | 70.05% | 70.35% | 71.19% |
| Kp_only | 68.67% | 68.09% | 68.79% |
| Ki_only | 68.76% | 69.33% | 70.84% |
| Kd_only | 70.24% | 70.95% | 70.74% |
| PI_control | 68.16% | 68.51% | 70.68% |
| PD_control | 68.66% | 69.13% | 70.04% |
| Full_PID | 68.81% | 68.02% | 71.05% |
| UDWDC_v2 | 68.81% | 68.02% | 71.05% |

## Key Observations

- **Best variant**: Kd_only (70.64% +/- 0.30%)

### Ranking by Test Accuracy

1. **Kd_only**: 70.64% +/- 0.30% (WD Budget: 0.5945)
2. **FixedWD**: 70.53% +/- 0.48% (WD Budget: 0.5800)
3. **Ki_only**: 69.64% +/- 0.88% (WD Budget: 0.1344)
4. **Full_PID**: 69.29% +/- 1.28% (WD Budget: 0.2766)
5. **UDWDC_v2**: 69.29% +/- 1.28% (WD Budget: 0.2766)
6. **PD_control**: 69.28% +/- 0.57% (WD Budget: 0.3002)
7. **PI_control**: 69.12% +/- 1.11% (WD Budget: 0.3174)
8. **Kp_only**: 68.52% +/- 0.31% (WD Budget: 0.2999)

### Analysis

- **FixedWD baseline**: 70.53% +/- 0.48%
- **Best UDWDC variant**: Kd_only (70.64%)
- **Delta over FixedWD**: +0.11%
- **Kd_only (alignment-derivative) best among UDWDC variants**: 70.64% with highest WD budget (0.5945) close to FixedWD (0.5800)
- **Kp_only (proportional) worst**: 68.52% with reduced WD budget (0.2999)
- **Full PID and UDWDC_v2 identical**: Both 69.29% (same gains K_p=0.5, K_i=0.1, K_d=0.3)
