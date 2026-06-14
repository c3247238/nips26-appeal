# Iteration 5 Cross-Architecture Analysis Summary

**Generated**: 2026-03-18 21:22

## Phi Spread Results

| Architecture | Dataset | Optimizer | rho | Phi Spread | Methods | Runs |
|---|---|---|---|---|---|---|
| ResNet-20 | CIFAR-10 | AdamW | rho~0.5 | 0.2500% | 7 | 21 |
| ResNet-20 | CIFAR-100 | AdamW | rho~0.5 | 0.7533% | 7 | 21 |
| ResNet-20 | CIFAR-10 | SGD(original) | rho~0.005 | 0.9133% | 7 | 21 |
| ResNet-20 | CIFAR-100 | SGD(original) | rho~0.005 | 1.7067% | 7 | 21 |
| VGG-16-BN | CIFAR-10 | AdamW | rho~0.5 | 0.2317% | 4 | 11 |
| ResNet-20-NoBN | CIFAR-10 | AdamW | rho~0.5 | N/A | 1 | 3 |
| ResNet-20 | CIFAR-10 | SGD(matched-rho) | rho~0.5 | N/A | 1 | 2 |
| ResNet-20 | CIFAR-10 | AdamW(rho_low) | rho~0.05 | N/A | 1 | 2 |

## NoBN vs BN

- **constant**: BN=90.13% vs NoBN=87.74% (diff=2.4%, Cohen's d=9.1416, large)

## Key Findings

1. **AdamW phi spread remains < 0.5% across ResNet-20 and VGG-16-BN on CIFAR-10** (confidence: medium)
   - Evidence: ResNet-20 phi=0.2500%, VGG-16-BN phi=0.2317%

1. **SGD shows significantly larger phi spread than AdamW on CIFAR-10** (confidence: high)
   - Evidence: SGD phi=0.9133% vs AdamW phi=0.2500% → 3.7x ratio

1. **NoBN reduces accuracy by ~2.2% vs BN on CIFAR-10 (constant method)** (confidence: high)
   - Evidence: BN=90.13% vs NoBN=87.74%

1. **ImageNet and rho_high experiments failed — cannot validate large-scale or high-rho hypotheses** (confidence: N/A (missing data))
   - Evidence: All ImageNet seeds and rho=5.0 experiments in FAILED state

1. **Matched-rho SGD analysis incomplete — cannot determine if rho is the confounding variable** (confidence: N/A (insufficient data))
   - Evidence: Only constant method available for matched-rho SGD; need cwd_hard and no_wd for phi spread computation


## Data Gaps

- ResNet-20-NoBN/CIFAR-10/AdamW: only 1 methods available
- ResNet-20-NoBN/CIFAR-10/AdamW: only 3 total runs
- ResNet-20/CIFAR-10/SGD(matched-rho): only 1 methods available
- ResNet-20/CIFAR-10/SGD(matched-rho): only 2 total runs
- ResNet-20/CIFAR-10/AdamW(rho_low): only 1 methods available
- ResNet-20/CIFAR-10/AdamW(rho_low): only 2 total runs
- Rho-high (rho=5.0): EXPERIMENT FAILED, no data available
- Matched-rho SGD: only constant method completed 200 epochs; cwd_hard, no_wd missing → cannot compute phi spread or ratio
- ImageNet/ResNet-50: ALL experiments FAILED — no large-scale data available
- VGG-16-BN: missing swd, no_wd, random_mask methods; cosine_schedule missing seed_456 → only 4/7 methods available
- NoBN: only constant method x 3 seeds available; cwd_hard and no_wd MISSING → cannot compute phi spread for NoBN
- Rho-high (rho=5.0): EXPERIMENT FAILED
- Matched-rho SGD CIFAR-100: NO DATA
- Matched-rho SGD CIFAR-10: seed_42 only ran 5 epochs (constant method); cwd_hard/no_wd missing entirely
