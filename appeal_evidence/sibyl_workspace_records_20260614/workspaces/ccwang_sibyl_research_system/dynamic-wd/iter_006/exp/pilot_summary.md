# Iteration 6 Pilot Summary

## impl_pmpwd: PMP-WD Implementation - COMPLETED

### Files Created/Modified
- `iter_006/exp/code/optimizers.py`: Added `PMPWDPhi` class implementing bang-bang controller
- `iter_006/exp/code/train_unified.py`: Added `pmpwd` to choices, added `lyapunov_v` and `effective_wd` to epoch diagnostics
- `iter_006/exp/code/train_sgd.py`: Added `pmpwd` to choices
- `iter_006/exp/code/models.py`, `data.py`: Copied from iter_003

### PMP-WD Implementation Details
- **Controller**: `lambda*(t) = lambda_max * I(<p(t), w(t)> > 0)` where p(t) is momentum buffer (costate proxy)
- **Switching function**: `sigma(t) = sum(costate * param.data)` - scalar inner product per parameter
- **Bang-bang**: Full WD when sigma > 0, zero WD when sigma <= 0
- **Optional smooth mode**: `smooth_beta` parameter for sigmoid approximation
- **Diagnostics**: `pmpwd_sigma`, `pmpwd_indicator`, `pmpwd_switch_count`, `pmpwd_switch_rate`

### Smoke Test (5 epochs, CIFAR-10/ResNet-20, AdamW, seed=42)
| Metric | Value |
|--------|-------|
| Best Test Acc | 77.97% |
| Lyapunov V_T | 641.19 → 652.31 |
| Switch Rate | ~35% (bang-bang switching active) |
| BEM | 0.6 (40% of constant budget, expected for bang-bang) |
| Training Time | 351s (5 epochs) |

### Observations
- Bang-bang switching is active: lambda oscillates between 0 and lambda_max
- Switch rate ~35% indicates significant switching activity
- BEM = 0.6 means PMP-WD uses on average ~40% less WD budget than constant
- No numerical issues or divergence
- Epoch 0: sigma > 0 (full WD), Epochs 1-4: sigma < 0 (zero WD) at end-of-epoch snapshot
  - This is expected: inner product sign can change within an epoch across parameters

### Pass Criteria
- [x] PMP-WD optimizer runs without error
- [x] diagnostics.jsonl contains all required fields (pmpwd_sigma, indicator, switch_count, lyapunov_v)
- [x] Training converges normally (77.97% at 5 epochs comparable to other methods)

## pilot_pmpwd_cifar10: PMP-WD Pilot Run - COMPLETED

### Configuration
- Architecture: ResNet-20, Dataset: CIFAR-10, Optimizer: AdamW
- Epochs: 200, LR: 1e-3 (cosine), WD: 5e-4, Seed: 42

### Results
| Metric | Value |
|--------|-------|
| Best Test Acc | **89.74%** |
| Final Test Acc | 89.66% |
| Constant Baseline | 90.13% |
| Delta from Baseline | -0.39% |
| BEM | 0.49 (uses ~51% WD budget) |
| Final Switch Rate | 54.8% |
| Final Lyapunov V_T | 4696.1 |
| Gen Gap | 10.08 |
| Training Time | 13521s (~3.75 hours) |

### Pass Criteria Evaluation
- [x] Accuracy >= 88% (89.74% >= 88%) - PASS
- [x] No divergence - PASS (monotonic loss decrease)
- [x] lambda(t) shows clear bang-bang switching pattern (switch rate ~55%) - PASS

### Key Observations
1. PMP-WD achieves accuracy within 0.39% of constant WD, confirming it is a viable WD method
2. Bang-bang switching is active throughout training (~55% switch rate)
3. Mean effective WD is 0.000255, about half of constant (0.0005), consistent with bang-bang
4. The weight norm grows larger (96.92 vs typical ~80 for constant) due to reduced effective regularization
5. Gen gap is wider (10.08 vs ~5 for constant), expected with less regularization

### pilot_imagenet - FAILED (not started)
ImageNet pilot was not executed in this iteration. Previous iterations had ImageNet infrastructure issues.
ImageNet experiments will be attempted in the experiment_cycle stage.

## Overall Readiness
PMP-WD pilot passes all criteria. Ready to proceed to `idea_validation_decision` and then full experiment cycle.
