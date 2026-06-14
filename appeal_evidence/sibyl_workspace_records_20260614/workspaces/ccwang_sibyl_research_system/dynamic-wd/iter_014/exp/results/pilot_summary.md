# Pilot Summary: setup_framework

## Overall: GO

All pilot checks passed. The framework is ready for full experiments.

## Unit Tests (7/7 PASS)

| Test | Status | Details |
|------|--------|---------|
| UDWDC(K=0) == FixedWD | PASS | Max relative error: 0.00e+00 (threshold 1e-6) |
| Norm computation | PASS | 65 layers checked, all correct |
| Clipping | PASS | Effective WD stays in [0, 10*lambda_base] |
| All optimizers construct | PASS | All 7 methods construct and step without error |
| Diagnostic logger | PASS | Saves/loads correctly, trajectories tracked |
| All models | PASS | ResNet-20 (272K), VGG-16-BN (15.3M), ResNet-50 (25.6M), ViT-S/16 (22M) |
| ResNet-101 | PASS | 44.5M params, correct output shape |

## End-to-End Pilot (7/7 methods)

All 7 methods completed 1-epoch CIFAR-10 training (500 samples) successfully:

| Method | Train Acc | Test Acc | Layers | Time (s) | Effective WD (conv1) |
|--------|-----------|----------|--------|----------|---------------------|
| FixedWD | 14.1% | 9.8% | 65 | 0.77 | 0.000100 |
| CWD | 14.1% | 9.7% | 65 | 0.49 | 0.000050 |
| SWD | 14.1% | 9.8% | 65 | 0.51 | 0.000090 |
| CPR | 14.1% | 9.6% | 65 | 0.51 | 0.001000 |
| DefazioCorrective | 14.1% | 9.8% | 65 | 0.48 | 0.000100 |
| NoWD | 14.1% | 9.6% | 65 | 0.54 | 0.000000 |
| UDWDC | 13.8% | 9.8% | 65 | 0.45 | 0.001000 |

## ImageNet Data Loading

- Train split: 294 shards, loads correctly
- Validation split: 14 shards, labels present and correct
- ResNet-50 forward pass on ImageNet (100 samples): OK

## Diagnostics Verified

- Per-layer rho_t (gradient-to-weight ratio): non-trivial, std > 0.01
- Per-layer alpha_t (gradient-weight alignment cosine): tracked correctly
- Per-layer effective WD: varies across methods as expected
- BN/norm layers correctly excluded from WD (effective_wd = 0.0)

## Key Observations

1. **CWD effective WD**: Lower than FixedWD (0.00005 vs 0.0001) because the binary mask filters ~50% of parameters -- expected behavior
2. **CPR effective WD**: Higher at 0.001 because the augmented Lagrangian penalty kicks in when weight norms exceed the target
3. **UDWDC effective WD**: Hits upper clip bound (0.001 = 10 * 1e-4) in early training with large K_p -- the controller correctly clips
4. **DefazioCorrective**: Same as FixedWD at epoch 0 (lr_ratio = 1.0) -- correct

## Files Produced

```
exp/code/optimizers/{base,fixed_wd,no_wd,udwdc,cwd,swd,cpr,defazio}.py
exp/code/models/{resnet,vgg,vit}.py
exp/code/data/{cifar,imagenet}.py
exp/code/diagnostics/{logger,metrics,alignment}.py
exp/code/train_cifar.py
exp/code/train_imagenet.py
exp/code/test_framework.py
```
