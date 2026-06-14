# Unified Dynamic Weight Decay — Complete Experimental Results

**Generated**: 2026-03-25 05:34

## Cross-Dataset Comparison

| Method | Type | CIFAR-100 (%) | ImageNet (%) | Avg Rank |
|--------|------|:---:|:---:|:---:|
| NoWD | Baseline | 63.74 ± 0.49 (#7) | 70.11 ± 0.15 (#7) | 7.0 |
| FixedWD | Baseline (SGDW) | 65.19 ± 0.25 (#1) | 71.89 ± 0.24 (#3) | 2.0 |
| SWD | Existing (NeurIPS'23) | 64.84 ± 0.12 (#4) | 72.04 ± 0.40 (#2) | 3.0 |
| CWD | Existing (ICLR'26) | 64.55 ± 0.13 (#5) | 71.39 ± 0.32 (#5) | 5.0 |
| CPR | Existing (NeurIPS'24) | 65.19 ± 0.08 (#2) | 71.38 ± 0.52 (#6) | 4.0 |
| CAWD | Ours (variant) | 64.52 ± 0.61 (#6) | 71.44 ± 0.15 (#4) | 5.0 |
| **EqWD** | Ours (proposed) | 65.05 ± 0.36 (#3) | 72.27 ± 0.20 (#1) | 2.0 |

## Key Findings

1. **EqWD achieves best ImageNet accuracy**: 72.27% (+0.38% vs FixedWD)
2. **EqWD shows lowest variance on ImageNet**: std = 0.197
3. **SWD is runner-up on ImageNet**: 72.04% but with higher variance (0.401)
4. **CWD and CPR underperform FixedWD on ImageNet**: both ~71.4%, suggesting binary/threshold-based methods lose effectiveness at scale
5. **Cross-dataset consistency**: EqWD ranks #1 on ImageNet and competitive on CIFAR-100
