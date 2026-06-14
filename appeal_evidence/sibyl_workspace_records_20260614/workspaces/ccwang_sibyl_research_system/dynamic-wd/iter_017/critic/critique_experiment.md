# Experiment Critique -- Iteration 017

## Overall Assessment

The experimental setup is competent: 7 methods, 3 seeds, two benchmarks (ImageNet + CIFAR-100), bootstrap CIs and Cohen's d. However, several critical missing experiments undermine the paper's core claims, and a potential data integrity issue (WD heatmap lambda discrepancy) requires immediate resolution.

## Strengths

1. **Multi-seed ImageNet comparison is a genuine asset.** 7 methods x 3 seeds on ImageNet is substantial. This is the paper's strongest experimental contribution.

2. **CAWD ablation is well-designed.** Using the same EMA/beta framework with cosine alignment instead of ratio deviation cleanly isolates the signal choice. The negative result (71.44% < 71.89% FixedWD) is informative.

3. **Bootstrap CIs and Cohen's d are above community norm.** Most papers at NeurIPS simply report mean +/- std without effect size analysis. The paper's statistical methodology is exemplary for n=3.

4. **Ablation coverage is adequate.** Beta sweep (0.1-5.0) and EMA decay sweep (0.8-0.99) provide useful sensitivity information.

## Critical Issues

### 1. Budget-Matched FixedWD Control (BLOCKING)

This is the single most important missing experiment. EqWD's phi_l(t) >= 1 design means the effective average WD over training is higher than lambda_base. The paper's +0.38% improvement over FixedWD could be entirely explained by applying more total regularization.

**What is needed:** Measure EqWD's mean effective lambda during training (average lambda_t^l across all layers and steps for the ImageNet runs). Run FixedWD at that value, 3 seeds. ~9 GPU-hours.

**Why it matters:** If budget-matched FixedWD matches EqWD (72.27%), the contribution is "EqWD automatically discovers a better WD strength" -- still publishable but fundamentally different from "adaptive modulation per se improves generalization." If budget-matched FixedWD underperforms, adaptive modulation is validated.

This was labeled BLOCKING P0 in iter_016 lessons_learned and was not executed in iter_017.

### 2. WD Heatmap Lambda Discrepancy (POTENTIAL DATA INTEGRITY)

Figure 5 (WD heatmap) colorbar shows lambda_t values from ~0.00052 to ~0.00062. The paper states lambda_base = 1e-4 = 0.0001. Even with phi_l(t) = 6.2 (which would require dev_t^l = 5.2 at beta=1.0, far exceeding the observed modest modulation), the maximum should be 0.00062. But the MINIMUM is 0.00052, which corresponds to phi_l(t) = 5.2 -- implying the base WD without modulation is already 5.2x higher than stated.

This is numerically inconsistent. Possible explanations:
- The heatmap was generated with lambda_base = 5e-4 (not 1e-4)
- The heatmap plots a different quantity than lambda_t^l
- The heatmap data comes from a different experiment

This must be resolved immediately. If lambda_base was actually 5e-4 in the experiments, all quantitative results are potentially affected.

### 3. 45-Epoch vs 90-Epoch ImageNet

The paper's ImageNet results use only 45 epochs. Standard ResNet-50 training uses 90 epochs and achieves ~76%+. EqWD's advantage may be inflated by the 45-epoch regime, which is dominated by transitional phases. At minimum, one 90-epoch comparison (EqWD vs FixedWD vs SWD) should validate that the advantage transfers.

## Major Issues

### 4. CIFAR-10 Table Incomplete

Table 6 excludes the paper's own method and includes an undefined baseline. This is an obvious oversight that will be immediately noticed by reviewers.

### 5. AIS on Wrong Benchmark

AIS is measured on CIFAR-100/ResNet-20 and VGG-16-BN, where EqWD doesn't decisively win. Not measured on ImageNet where EqWD shows its primary advantage. Running AIS on ImageNet (even for a subset of layers) would substantially strengthen Contribution 3.

### 6. Beta=5.0 Unvalidated

Single-seed 66.07% is 1.02% above multi-seed default 65.05%. This gap exceeds the std of the multi-seed result (0.36%). Multi-seed validation is essential before highlighting this in the paper.

### 7. VGG-16-BN Results Without Context

62.81% on CIFAR-100/VGG-16-BN appears alarmingly low but the cifar_methods_comparison figure (present in workspace but NOT in the paper) shows all methods at 61-63%. Including this figure would contextualize the result.

### 8. Bayesian Optimization Fairness

50 BO trials over multi-dimensional spaces may under-tune baselines. The best found hyperparameters should be reported so reviewers can assess tuning quality.

## Minor Issues

- No AdamW experiment (significant scope limitation for a 2026 paper)
- AdaDecay (Nakamura & Hong, 2019) not compared -- a directly relevant per-parameter WD method
- Population vs. sample std convention undocumented
- No Transformer architecture experiments
