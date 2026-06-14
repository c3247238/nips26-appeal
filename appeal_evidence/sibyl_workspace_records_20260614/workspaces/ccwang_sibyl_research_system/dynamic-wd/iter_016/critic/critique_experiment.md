# Experiment Critique -- Iteration 16

## Overall Assessment

The experimental design is substantially improved from iter_015. ImageNet experiments with 7 methods and 3 seeds are now complete (a major milestone after 11+ iterations of incomplete ImageNet coverage). The CAWD ablation is a valuable addition that isolates the modulation signal choice. Statistical methodology (bootstrap CIs, Cohen's d) is rigorous. However, several critical experimental gaps remain.

## Strengths

1. **Complete ImageNet multi-seed comparison**: All 7 methods x 3 seeds on ImageNet is a genuine achievement. This was the longest-standing gap in the project.

2. **CAWD control ablation**: Testing continuous cosine alignment with the same EMA framework as EqWD is an excellent experimental design that cleanly isolates the ratio vs. alignment signal question.

3. **Statistical sophistication**: Bootstrap CIs, Cohen's d, explicit n=3 caveats -- this is above community norms.

4. **Ablation coverage**: Beta sweep (5 values), alpha sweep (4 values), and layer-type ablation provide reasonable hyperparameter sensitivity analysis.

## Critical Issues

### 1. Effective WD Budget Confound (UNCONTROLLED)

EqWD's phi >= 1 means it always applies MORE total weight decay than FixedWD with the same lambda_base. This is the most important missing control in the entire paper.

**What to do**: Measure EqWD's mean effective lambda over training (e.g., lambda_base * mean(phi_l(t))). Then run FixedWD at that lambda value, 3 seeds, on ImageNet. Cost: ~9 GPU-hours.

**Why it matters**: If budget-matched FixedWD matches EqWD (72.27%), the entire adaptive modulation story collapses to "EqWD implicitly finds a better WD strength." If budget-matched FixedWD is worse, adaptive modulation provides genuine benefit beyond strength.

### 2. Non-Standard ImageNet Training Regime (45 EPOCHS)

The figures clearly show 45-epoch training. Standard ImageNet ResNet-50 training uses 90 epochs (76%+) or 100 epochs. At 45 epochs, the training is dominated by transitional dynamics (warmup + cosine decay knee), which is exactly where EqWD claims advantage. This is not necessarily unfair, but it:
- Must be explicitly disclosed
- Raises the question of whether EqWD's advantage persists at 90 epochs
- Produces absolute numbers (72.27%) that are not directly comparable to published results (76%+)

**What to do**: Run a 90-epoch comparison for at least EqWD, FixedWD, and SWD (3 seeds each). Report both 45-epoch and 90-epoch results.

### 3. CIFAR-10 Table Missing EqWD

Table 6 presents CIFAR-10 results for 6 methods but excludes EqWD. In a paper proposing EqWD, this is a jarring omission. The table also includes "DefazioCorrective" which is not in the main baselines list.

**What to do**: Run EqWD on CIFAR-10 (3 seeds, 200 epochs, ~30 min GPU time) and add to the table. Remove or explain DefazioCorrective.

### 4. AIS Validated on Wrong Benchmark

The ratio sufficiency claim (Contribution 3) is validated via AIS on CIFAR-100/ResNet-20 and VGG-16-BN. But EqWD's advantage is on ImageNet/ResNet-50. Validating a sufficiency claim where the method does NOT win does not support the claim where it DOES win.

**What to do**: Run AIS on ImageNet/ResNet-50 for at least 3-4 representative layers. This requires logging gradient norms, weight norms, and cosine alignments during one training run -- minimal GPU overhead on top of a regular training run.

## Major Issues

### 5. Bayesian Optimization Protocol Unclear

"50 Bayesian optimization trials" is mentioned but:
- What hyperparameters were searched for each baseline? (e.g., SWD has different hyperparameters than CPR)
- What ranges were used?
- What were the final selected hyperparameters?
- Were all 50 trials run at 3 seeds or only the best config?

Without this information, "50 BO trials" could mean anything from rigorous grid search to underfitting the baselines.

### 6. VGG-16-BN Results Abnormally Low

EqWD on VGG-16-BN/CIFAR-100 achieves 62.81%, which is ~10% below typical VGG-16-BN performance on CIFAR-100 (73-74%). No baseline comparison is provided for VGG-16-BN, so the reader cannot tell if this is a training configuration issue or an EqWD-specific problem.

### 7. No Wall-Clock Overhead Measurement Details

The paper claims "approximately 2% wall-clock overhead" but does not report how this was measured (which GPU, which batch size, which architecture, how many runs). A single sentence would suffice.

## Minor Issues

8. Population vs sample std convention still undocumented. At n=3, the difference matters (22%).
9. No learning rate warmup details for ImageNet (how many epochs? linear or constant?).
10. Data augmentation details in appendix are minimal -- "standard normalization" should specify the exact mean/std values used.
11. The paper does not report top-5 accuracy for ImageNet, which is standard.

## Summary of Recommended Experiments (Priority Order)

| Priority | Experiment | GPU-Hours | Blocks |
|----------|-----------|-----------|--------|
| P0 | Budget-matched FixedWD on ImageNet | ~9 | Core claim |
| P0 | State epoch count (0 GPU) | 0 | Reproducibility |
| P1 | 90-epoch ImageNet (EqWD, FixedWD, SWD) | ~18 | Generalizability |
| P1 | EqWD on CIFAR-10 (3 seeds) | ~0.5 | Table completeness |
| P1 | AIS on ImageNet | ~3 | Contribution 3 |
| P2 | Beta=5.0 multi-seed on CIFAR-100 | ~1 | Narrative coherence |
| P2 | VGG-16-BN baselines | ~3 | Context for ablation |
| P3 | BO hyperparameter details | 0 | Reproducibility |
