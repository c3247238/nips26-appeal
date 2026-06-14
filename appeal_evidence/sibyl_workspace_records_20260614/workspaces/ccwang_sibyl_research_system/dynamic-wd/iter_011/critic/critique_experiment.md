# Experiment Critique — Iteration 11

## Overall Assessment: MAJOR REVISION REQUIRED

The experimental design is internally consistent and fair (shared hyperparameters, unified codebase, proper seeding), but the scope is far too narrow for the claims made. The paper proposes a general conjecture about weight decay invariance but tests it only on CIFAR with small CNNs.

## Critical Gaps

### 1. No ImageNet Experiments (Major)
The project specification requires ImageNet with ResNet-50. The paper only tests CIFAR-10/100. At NeurIPS/ICML, a "systematic benchmark" paper without ImageNet is incomplete. CIFAR-10/100 are saturated benchmarks where 0.25pp differences are noise regardless of the optimizer.

**Impact**: Reviewers will immediately flag this as insufficient for a paper claiming general principles about weight decay.

### 2. Untested Methods from Table 1 (Major)
Table 1 lists 10 methods including AdamWN and AlphaDecay, but only 7 are tested. The untested methods (AdamWN, AlphaDecay, CWD-soft) are arguably the most sophisticated and the ones practitioners would most want to know about. The tested set is dominated by trivial controls (half-lambda, random mask, no-WD).

**Impact**: The framework claims to "recover" these methods but never validates whether they too exhibit Phi Invariance. The most interesting predictions are untested.

### 3. Statistical Power (Major)
- N=3 seeds gives 80% power for effects >= 0.7% only
- TOST equivalence confirmed for only 6/12 comparisons at delta=1.0%
- At delta=0.5%, even fewer comparisons would pass
- Cohen's d < 0.3 is reported but with N=3, the CI on d is enormous

The "equivalence" conclusion is largely an artifact of low power. The paper is honest about this (Section 6.4), but it doesn't change the fact that the central claim is underpowered.

### 4. Below-SOTA Baselines (Major)
ResNet-20 on CIFAR-10: 90.13% (AdamW), 91.22% (SGD). Published baselines for ResNet-20 are typically 92-93%. This 2-3% gap suggests the training configuration is suboptimal.

**Possible explanations**: (a) AdamW lr=1e-3 may be too low for ResNet-20; (b) No warmup schedule; (c) Limited augmentation (no cutout, mixup).

**Risk**: The invariance finding might reflect a regime where the model is undertrained and nothing can help, rather than a genuine theoretical phenomenon.

### 5. No Vision Transformer (Major)
ViTs with LayerNorm (not BatchNorm) are the dominant architecture in 2026. The paper's most interesting prediction---that Phi Invariance might break without BN---is never tested. ViT-Tiny on CIFAR-100 would take ~30 minutes per seed.

### 6. Architecture Confound: BN vs Adaptive Optimizer
The VGG-16-BN + SGD result (0.16pp spread) is NARROWER than ResNet-20 + AdamW (0.25pp spread). This means:
- BN alone produces stronger invariance than AdamW alone
- The "AdamW's adaptive scaling" explanation in the conjecture may be wrong
- BN may be the real mechanism, with AdamW being coincidentally correlated

This is not addressed in the experimental analysis. Need: (a) ResNet-20 without BN + AdamW; (b) ResNet-20 without BN + SGD; (c) VGG without BN + SGD.

## Minor Issues

### 7. No Compute Cost Reporting
Dynamic methods (CWD sign-alignment, SWD gradient-norm computation) add overhead. If all methods are equivalent in accuracy, the cheapest one wins. No wall-clock times are reported.

### 8. Incomplete BN Ablation
Section 6.2 tests only 2/7 methods without BN. This is insufficient for conclusions.

### 9. No Learning Rate Sensitivity Analysis
The conjecture predicts WD scheduling is irrelevant under AdamW. But what about the interaction between WD and LR? A sweep over lr={5e-4, 1e-3, 3e-3} for 2-3 methods would strengthen the claim.

## What's Done Well
- Unified codebase with pluggable Phi interface
- Fair hyperparameter protocol (no per-method tuning)
- Proper paired t-tests with Bonferroni correction
- TOST equivalence testing (correct statistical approach)
- Clear reporting of effect sizes and p-values
- Both AdamW and SGD tested (good for the boundary condition)
