# Planning Critique — Iteration 11

## Overall Assessment: GOOD STRUCTURE, INSUFFICIENT SCOPE

The experimental plan is well-structured within its scope: unified codebase, fair hyperparameters, proper statistical testing. But the scope is too narrow for the claims, and several obvious experiments are missing.

## Major Issues

### 1. Missing ImageNet Experiments
The project spec lists ImageNet as a required dataset with ResNet-50. At 105 experiments (all CIFAR), this is a significant planning gap. ImageNet results would:
- Test scale sensitivity of Phi Invariance
- Provide a more realistic benchmark for practitioner claims
- Enable comparison with published CWD/SWD results (both report ImageNet numbers)

**Estimated cost**: ResNet-50 on ImageNet, 90 epochs, 3 methods x 2 seeds = ~48 GPU-hours on RTX PRO 6000. Feasible within the available hardware.

### 2. Architecture Coverage Too Narrow
Both ResNet-20 and VGG-16-BN use BatchNorm. The paper's most interesting prediction — that Phi Invariance breaks without BN — is never tested. The plan should include:
- ResNet-20 without BN (both AdamW and SGD)
- ViT-Tiny or ViT-Small (LayerNorm instead of BN)

### 3. No BN Ablation Plan
The BN result (Section 6.2) is the most surprising finding but is supported by only 2 methods. A proper ablation requires:
- All 7 methods, ResNet-20 without BN, AdamW, CIFAR-10, 3 seeds = 21 runs
- All 7 methods, ResNet-20 without BN, SGD, CIFAR-10, 3 seeds = 21 runs
Total: 42 additional runs (~4 GPU-hours), which would definitively answer whether BN or AdamW is the primary mechanism.

### 4. AdamWN and AlphaDecay Implementation Gap
These methods are described in the framework but not implemented. The plan should have included at least AdamWN (simple to implement — just a target-norm phi function) as a validation that the framework works for non-trivial modulators.

## Minor Issues

### 5. No Hyperparameter Sensitivity Analysis
The paper uses fixed hyperparameters by design (fairness), but doesn't test whether the invariance result holds across hyperparameter settings. A 2D sweep over (lr, lambda) for constant and CWD would show whether invariance is robust to the operating point.

### 6. Missing Compute Budget Tracking
No wall-clock times or GPU-hour accounting. This makes it impossible to assess whether the 105-experiment claim represents a serious computational investment or a weekend of GPU time.

### 7. No Plan for Overfitting Regime
Section 6.4 lists "overfitting regime" as a limitation. A simple experiment: train ResNet-20 on a 5K-sample subset of CIFAR-10 for 1000 epochs, measure whether WD method matters in the overfitting regime.

## Experiment Count Analysis
Current: 105 experiments (all CIFAR, all BN architectures)
- 84 runs: ResNet-20, 2 optimizers, 2 datasets, 7 methods, 3 seeds
- 21 runs: VGG-16-BN, SGD, CIFAR-10, 7 methods, 3 seeds

Recommended additions (priority order):
1. BN ablation: 42 runs (~4 GPU-hours)
2. ImageNet subset: 18 runs (3 methods x 2 seeds x 3 configs, ~48 GPU-hours)
3. ViT-Tiny: 42 runs (~8 GPU-hours)
4. AdamWN/AlphaDecay on CIFAR: 24 runs (~2 GPU-hours)

Total recommended: ~62 GPU-hours additional. Feasible with 8x RTX PRO 6000 in 1-2 days.

## What's Done Well
- Unified codebase with pluggable interface
- Fair comparison protocol (identical hyperparameters)
- Multi-optimizer design (AdamW vs SGD as boundary condition)
- Proper statistical testing (paired t-tests, TOST, Bonferroni)
- Multi-seed design (3 seeds, though more would be better)
- Cross-architecture validation (ResNet-20 and VGG-16-BN)
