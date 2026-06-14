# Planning Critique

## Overall Assessment

The experimental design is methodologically sound in its core principles: identical hyperparameters across methods, multi-seed evaluation, proper statistical testing. However, the planning has systematic blind spots that weaken the paper's contribution.

## Critical Issues

### 1. No ImageNet Despite Project Spec Requirement

The project constraints explicitly list ImageNet as a required dataset with ResNet-50. The lessons_learned.md mentions "two iterations of blind failure" on ImageNet. The planning failed to diagnose root causes (OOM? path errors? data issues?) before abandoning the scale-up. This is the single most impactful missing experiment for the paper's claims.

### 2. No Architecture Without Batch Normalization

The planning included NoBN experiments (mentioned in lessons_learned.md with "narrow spread 0.12pp"), but these are absent from the paper. The BN hypothesis (Section 6.2) is central to the paper's narrative but is untested within the paper itself. The planning should have ensured NoBN data was integrated before writing.

## Major Issues

### 3. Incomplete Factorial Design Presented as Complete

The experimental matrix is:
- ResNet-20: 7 methods x 2 optimizers x 2 datasets x 3 seeds = 84 runs (complete)
- VGG-16-BN: 7 methods x 1 optimizer x 1 dataset x 3 seeds = 21 runs (partial)

The missing cells are: VGG-16-BN + AdamW, VGG-16-BN + CIFAR-100. Without these, the cross-architecture claim is limited to a single optimizer-dataset combination.

### 4. No Hyperparameter Sensitivity Analysis

The paper uses lambda=5e-4 for all methods. But the Phi Invariance Conjecture might be lambda-dependent. At lambda=5e-2 (100x larger), dynamic WD strategies might differentiate because the base decay is strong enough to dominate AdamW's implicit norm control. At lambda=5e-6 (100x smaller), all methods trivially converge to no_wd behavior. Testing 2-3 lambda values would strengthen or bound the conjecture.

### 5. Missing Overfitting Regime Test

Section 6.4 acknowledges "all experiments operate in the well-generalized regime." But WD's classical role is as a regularizer against overfitting. Testing on a deliberately overfitting setup (e.g., CIFAR-10 with ResNet-50 and no augmentation) would test whether Phi Invariance holds when WD matters most as a regularizer.

## Minor Issues

- The planning did not account for time to write appendix proofs, which is now the 4th-iteration blocker.
- PMP-WD was planned and partially executed but not integrated into the paper's method set -- planning should have committed to either including or excluding it.
- The Theorem 2 validation task estimated 60 minutes but the correlation turned out non-significant -- there was no contingency plan for what to do with a negative validation result.
