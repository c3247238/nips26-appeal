# Testable Hypotheses

## H1: Cumulative Alignment Contraction (Theory Validation)

**Statement**: The cumulative alignment-weighted contraction index C_T = sum_t lambda_t * (1 - alpha_t) under alignment-aware WD is strictly larger than under fixed WD with the same total budget (sum_t lambda_t), and C_T correlates negatively with the generalization gap (test_loss - train_loss).

**Expected outcome**: Pearson correlation between C_T and generalization gap across methods: r < -0.7. Alignment-aware WD achieves C_T at least 15% higher than fixed WD.

**Falsification criterion**: If C_T under alignment-aware WD is within 5% of fixed WD, or if the correlation |r| < 0.3, the cumulative contraction theory does not provide actionable advantage.

**Experiment**: Exp 1 — CIFAR-100 / ResNet-20, 3 seeds, track C_T and generalization gap for Fixed SGDW vs. alignment-aware SGDW vs. CWD vs. EqWD.

---

## H2: Gradient-to-Weight Ratio Deviation as WD Signal (Algorithm Validation)

**Statement**: EqWD (lambda_t = lambda_base * (1 + beta * |r_t - r*| / r*)) achieves strictly better generalization than both fixed SGDW and SWD (gradient-norm-only scheduling) on CIFAR-100 and ImageNet, because the ratio r_t = ||g_t||/||w_t|| encodes both gradient magnitude and weight norm simultaneously — a richer signal than ||g_t|| alone.

**Expected outcome**: EqWD (best beta in {0.5, 1.0, 2.0}) achieves >= 0.2% absolute accuracy improvement over SWD on CIFAR-100 (ResNet-20 and VGG-16-BN, mean over 3 seeds), and >= 0.3% on ImageNet (ResNet-50).

**Falsification criterion**: If EqWD does not improve over SWD on >= 2/3 benchmark settings (2 CIFAR + 1 ImageNet), the ratio deviation signal is not incrementally informative over gradient norm alone.

**Experiment**: Core CIFAR experiments (30-60 min/run) + ImageNet experiments (4-6 hr/run).

---

## H3: Alignment Signal Information Content (Prerequisite Diagnostic)

**Statement**: The gradient-weight cosine similarity delta_hat_t contains statistically significant additional information for predicting final test accuracy beyond what is captured by (||g_t||, ||w_t||) alone.

**Expected outcome**: Mutual information I(delta_hat_t; test_acc | ||g_t||, ||w_t||) > 0, with 95% bootstrap CI lower bound > 0, across all 4 CIFAR architecture-dataset pairs.

**Falsification criterion**: If partial MI CI includes 0 for >= 3 of 4 settings, the alignment signal provides no incremental predictive value and alignment-aware WD is theoretically unjustified as a distinct strategy.

**Experiment**: Alignment diagnostic — single runs per setting with 6 WD values, log delta_hat_t, ||g_t||, ||w_t|| per layer per step.

---

## H4: Layer-Type-Aware Alignment (Contrarian Test)

**Statement**: In architectures with batch normalization, the alignment signal delta_hat_t has significantly lower signal-to-noise ratio (SNR) in BN-preceded layers than in non-normalized layers. Consequently, applying alignment-aware WD only to non-normalized layers (with fixed or ratio-only WD for normalized layers) matches or exceeds uniform alignment-aware WD.

**Expected outcome**: SNR(delta_hat_t) in BN layers is < 50% of SNR in non-BN layers. Layer-type-aware EqWD matches or exceeds uniform EqWD by >= 0.1% on VGG-16-BN/CIFAR-100.

**Falsification criterion**: If SNR is similar across layer types (ratio > 0.8), or if uniform alignment-aware WD consistently outperforms the layer-type-aware variant, the layer heterogeneity critique is empirically unsupported.

**Experiment**: VGG-16-BN and VGG-16 (no BN) on CIFAR-100; layer-type ablation of CWD and EqWD.

---

## H5: Ratio Equilibrium Characterization (Theory Validation)

**Statement**: The per-layer gradient-to-weight ratio r_t converges to an equilibrium r* that is accurately predicted by Proposition 3:
- Fixed WD: r* = lambda / gamma (Defazio 2025)
- CWD: r* = lambda / gamma * P(alpha >= 0)
- Scheduled WD: r*(t) tracks lambda_t / gamma_t with measurable lag
- EqWD: r* tracks the target ratio with reduced deviation variance (CSI lower than fixed WD)

**Expected outcome**: For fixed WD, relative error between measured r(T) and predicted r* < 20% across layers. CWD's r* is measurably lower than fixed WD's r*. EqWD's CSI is the lowest among all methods.

**Falsification criterion**: If relative error > 50% for fixed WD (the simplest case), the equilibrium theory is not predictive and the entire ratio framework is questionable.

**Experiment**: Exp 2 — CIFAR-100 / VGG-16-BN, track per-layer r_t for all WD methods.

---

## H6: Budget Equivalence (Null Hypothesis Test)

**Statement**: Under matched compute budgets (equal FLOPs) and matched hyperparameter search budgets (50 Bayesian optimization trials each), at least one dynamic WD method achieves statistically significant improvement >= 0.3% absolute test accuracy over the best fixed WD baseline on ResNet-50/ImageNet-1K.

**Expected outcome**: EqWD or CWD achieves the significance threshold. If no method does, this is an important negative finding.

**Falsification criterion**: If no dynamic WD method achieves CI lower bound > 0 over tuned fixed WD on any primary benchmark, dynamic WD's literature gains are potentially artifacts of unequal tuning budgets.

**Experiment**: Budget Equivalence Test — all methods with identical Optuna budgets on CIFAR and ImageNet.

---

## H7: Continuous vs. Binary Alignment (CAWD Test)

**Statement**: Replacing CWD's binary sign-alignment mask with continuous cosine similarity modulation (CAWD: wd_scale = (1 + cos_sim) / 2) achieves measurably better generalization than CWD, because the continuous signal exploits alignment magnitude, not just sign.

**Expected outcome**: CAWD achieves >= 0.1% improvement over CWD on CIFAR-100 (VGG-16-BN, 3 seeds) and >= 0.2% on ImageNet (ResNet-50).

**Falsification criterion**: If CAWD does not improve over CWD on any benchmark at p < 0.05, the magnitude information in the alignment signal is not incrementally useful and binary masking is sufficient.

**Experiment**: Core CIFAR + ImageNet experiments; include CAWD as one of the tested methods.

---

## Priority Order for Pilot Validation

1. **H3** (alignment informativeness) — prerequisite; determines whether alignment-aware WD is justified
2. **H2** (EqWD vs baselines) — core algorithm validation
3. **H5** (ratio equilibrium) — core theory validation
4. **H4** (layer-type awareness) — addresses the contrarian's key critique
5. **H6** (budget equivalence) — most resource-intensive; run after CIFAR results confirm a promising method
6. **H1** (cumulative contraction) — theoretical; validated alongside H2 experiments
7. **H7** (CAWD) — secondary algorithm comparison
