# CIFAR Analysis Summary

Generated automatically.

## Main Results: Mean +/- Std Test Accuracy (%)

| Method | CIFAR-100 / ResNet-20 | CIFAR-100 / VGG16-BN |
|--------|----------------------|---------------------|
| NoWD | 63.74 +/- 0.40 | 61.07 +/- 0.81 |
| FixedWD | 65.19 +/- 0.21 | 62.14 +/- 0.42 |
| SWD | 64.84 +/- 0.10 | 62.20 +/- 1.21 |
| CWD | 64.55 +/- 0.10 | 62.86 +/- 1.36 |
| CPR | 65.19 +/- 0.07 | 62.22 +/- 1.01 |
| CAWD | 64.52 +/- 0.50 | 61.61 +/- 2.02 |
| EqWD | **65.05 +/- 0.30** | **62.81 +/- 1.31** |

**Best on ResNet-20**: FixedWD (65.19%)

**Best on VGG16-BN**: CWD (62.86%)

## Ablation: Beta Sensitivity

| Beta | Best Test Acc (%) |
|------|-------------------|
| 0.1 | 65.21 |
| 0.5 | 65.07 |
| 1.0 | 65.39 |
| 2.0 | 65.35 |
| 5.0 | 66.07 |

## Ablation: EMA Alpha Sensitivity

| Alpha | Best Test Acc (%) |
|-------|-------------------|
| 0.8 | 65.47 |
| 0.9 | 65.39 |
| 0.95 | 64.81 |
| 0.99 | 64.68 |

## Ablation: Uniform vs Layer-Aware

| Type | Mean +/- Std (%) |
|------|-----------------|
| uniform | 62.81 +/- 1.31 |
| layeraware | 62.32 +/- 1.19 |

## Ablation: VGG16 Without Batch Normalization

| Method | Mean +/- Std (%) |
|--------|-----------------|
| NoWD | 1.00 +/- 0.00 |
| FixedWD | 1.00 +/- 0.00 |
| SWD | 1.00 +/- 0.00 |
| CWD | 1.00 +/- 0.00 |
| CPR | 1.00 +/- 0.00 |
| CAWD | 1.00 +/- 0.00 |
| EqWD | 1.00 +/- 0.00 |

## Figures

- `cifar_methods_comparison.png`: OK
- `ablation_beta.png`: OK
- `ablation_ema.png`: OK
- `ablation_layertype.png`: OK
- `ratio_trajectories.png`: OK
- `wd_heatmap.png`: OK

## Alignment Diagnostic (H3: delta_hat informativeness)

### CIFAR-100/ResNet-20

| Layer | Residual Var Ratio | MI(delta, g|w) |
|-------|-------------------|----------------|
| conv1 | 0.9908 | 0.0000 |
| layer1.0.conv1 | 0.9987 | 0.0000 |
| layer1.0.conv2 | 0.9877 | 0.0000 |
| layer1.1.conv1 | 0.9834 | 0.0000 |
| layer1.1.conv2 | 0.9967 | 0.0000 |
| layer1.2.conv1 | 0.9980 | 0.0000 |
| layer1.2.conv2 | 0.9930 | 0.0000 |
| layer2.0.conv1 | 0.9953 | 0.0000 |
| layer2.0.conv2 | 0.9957 | 0.0000 |
| layer2.0.shortcut.0 | 0.9963 | 0.0000 |

### CIFAR-100/VGG16BN

| Layer | Residual Var Ratio | MI(delta, g|w) |
|-------|-------------------|----------------|
| features.0 | 0.9996 | 0.0000 |
| features.3 | 0.9881 | 0.0000 |
| features.7 | 0.9938 | 0.0000 |
| features.10 | 0.9755 | 0.0000 |
| features.14 | 0.9964 | 0.0000 |
| features.17 | 0.9542 | 0.0000 |
| features.20 | 0.9632 | 0.0000 |
| features.24 | 0.9909 | 0.0000 |
| features.27 | 0.9872 | 0.0000 |
| features.30 | 0.9928 | 0.0000 |
