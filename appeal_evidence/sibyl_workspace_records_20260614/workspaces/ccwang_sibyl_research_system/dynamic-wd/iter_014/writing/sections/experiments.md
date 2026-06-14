# 6. Experiments

We evaluate the unified framework and UDWDC across three dataset-architecture combinations with 7 methods, 3--5 seeds per configuration, and full diagnostic tracking of $\rho_t$, $\alpha_t$, effective WD, and weight norms per layer per epoch.

## 6.1 Experimental Setup

**Datasets.**  CIFAR-10 (50K/10K, 32$\times$32), CIFAR-100 (50K/10K, 32$\times$32), and ImageNet-1K (1.28M/50K, 224$\times$224).

**Architectures.**  ResNet-20 (272K parameters, BN) for CIFAR-10 diagnostic experiments; VGG-16-BN (15.3M parameters, BN) for CIFAR-100 ablation; ResNet-50 (25.6M parameters, BN) for ImageNet main comparison.

**Methods.**  Seven baselines: FixedWD ($\lambda = 10^{-4}$), CWD, SWD, CPR, DefazioCorrective, NoWD ($\lambda = 0$), and UDWDC.  UDWDC-v2 (EMA smoothing + floor clipping) is included as a stability-fixed variant.

**Training protocol.**  SGD with momentum 0.9, cosine annealing from $\eta_0 = 0.1$ to 0.  CIFAR: 200 epochs, batch size 128, seeds $\{42, 123, 456\}$.  ImageNet: 90 epochs, batch size 432 (DDP across 2 GPUs), seeds $\{42, 123, 456, 789, 2024\}$ for FixedWD (5 seeds) and $\{42, 123, 456\}$ for dynamic methods (3 seeds).  Augmentation: RandomCrop with padding 4 + RandomHorizontalFlip (CIFAR); RandomResizedCrop(224) + RandomHorizontalFlip (ImageNet).

**Diagnostics.**  Per-layer $\rho_t^l$, $\alpha_t^l$, $\|w_t^l\|$, $\|g_t^l\|$, and effective $\lambda_t^l$ are logged at every epoch for all runs.

## 6.2 CIFAR-10 Diagnostic Results

Table 3 reports the 200-epoch CIFAR-10/ResNet-20 results.

| Method | Best Acc (%) | Gen Gap (%) | WD Budget | BEM |
|--------|-------------:|------------:|----------:|----:|
| **CPR** | **91.74 $\pm$ 0.07** | **8.28 $\pm$ 0.05** | 4.44 | 0.39 |
| FixedWD | 90.68 $\pm$ 0.11 | 9.28 $\pm$ 0.17 | 0.48 | 0.89 |
| DefazioCorrective | 90.62 $\pm$ 0.20 | 9.31 $\pm$ 0.22 | 0.24 | 1.50 |
| SWD | 90.39 $\pm$ 0.19 | 9.49 $\pm$ 0.23 | 0.47 | 0.30 |
| UDWDC-v2 | 90.36 $\pm$ 0.09 | 9.53 $\pm$ 0.07 | 98599 | $\approx$0 |
| CWD | 90.32 $\pm$ 0.08 | 9.66 $\pm$ 0.05 | 0.24 | 0.12 |
| NoWD | 90.25 $\pm$ 0.30 | 9.73 $\pm$ 0.30 | 0.00 | --- |
| UDWDC | 90.15 $\pm$ 0.23 | 9.82 $\pm$ 0.25 | 0.38 | $-$0.26 |

*Table 3: CIFAR-10/ResNet-20, 200 epochs, 3 seeds.  CPR achieves the highest accuracy by a wide margin (+1.06 pp over FixedWD), driven by its aggressive integral control that accumulates $10\times$ the WD budget.  FixedWD is the strongest simple baseline.  UDWDC v1 underperforms NoWD by 0.10 pp, reflecting its instability (Section 3.3).  UDWDC-v2's enormous WD budget (98599) results from the floor clipping applying nonzero $\lambda$ to all 65 layers including BN layers at every step.  BEM is computed as (acc $-$ acc$_{\text{NoWD}}$) / WD budget.*

**Key observations.**

1. CPR's integral control is highly effective on CIFAR-10: accumulating constraint violations drives $\lambda_{\text{eff}}$ to $10^{-3}$, producing tighter weight norms and the smallest generalization gap (8.28%).

2. CWD's effective WD is approximately 50% of FixedWD (0.24 vs. 0.48).  This 50% reduction is a potential confound: the alignment mask may improve accuracy partly through WD magnitude reduction rather than alignment information alone.

3. UDWDC v1 achieves accuracy below NoWD, indicating that the proportional controller's instability (CSI = $-$2.41) actively harms performance.  The v2 floor-clipping fix recovers to 90.36%, restoring competitive performance at the cost of an inflated WD budget.

4. The spread between the best (CPR, 91.74%) and worst WD method (UDWDC, 90.15%) is 1.59 pp, demonstrating that the choice of WD strategy has meaningful impact on CIFAR-10.

## 6.3 UDWDC Gain Ablation (CIFAR-100/VGG-16-BN)

To isolate the contribution of each PID component, we run 7 gain configurations on CIFAR-100/VGG-16-BN (200 epochs, 3 seeds).  Table 4 reports the results.

| Variant | $K_p$ | $K_i$ | $K_d$ | Acc (%) | WD Budget | Gen Gap (%) |
|---------|------:|------:|------:|--------:|----------:|------------:|
| Kd_only | 0 | 0 | 0.3 | **70.64 $\pm$ 0.30** | 0.594 | **29.34** |
| FixedWD | 0 | 0 | 0 | 70.53 $\pm$ 0.48 | 0.580 | 29.41 |
| Ki_only | 0 | 0.1 | 0 | 69.64 $\pm$ 0.88 | 0.134 | 30.08 |
| Full PID | 0.5 | 0.1 | 0.3 | 69.29 $\pm$ 1.28 | 0.277 | 30.46 |
| PD_control | 0.5 | 0 | 0.3 | 69.28 $\pm$ 0.57 | 0.300 | 30.36 |
| PI_control | 0.5 | 0.1 | 0 | 69.12 $\pm$ 1.11 | 0.317 | 30.55 |
| Kp_only | 0.5 | 0 | 0 | 68.52 $\pm$ 0.31 | 0.300 | 31.30 |

*Table 4: UDWDC gain ablation on CIFAR-100/VGG-16-BN, 200 epochs, 3 seeds.  The alignment-only variant (Kd_only) slightly outperforms FixedWD, while all proportional-containing variants underperform.  High variance in Full PID and PI_control reflects sensitivity to the interaction of gains.*

**Key observations.**

1. The alignment correction alone ($K_d = 0.3$, $K_p = K_i = 0$) matches or slightly exceeds FixedWD (70.64% vs. 70.53%), confirming that alignment information provides marginal value when applied in isolation on this benchmark.

2. Adding proportional control ($K_p > 0$) consistently degrades accuracy.  Kp_only achieves 68.52%, 2 pp below FixedWD.  This reflects the instability documented in Section 3.3: proportional feedback on per-layer $\rho_t$ introduces noise that harms the optimization trajectory.

3. The integral term ($K_i$) partially mitigates the proportional instability: PI_control (69.12%) outperforms Kp_only (68.52%) by 0.60 pp, consistent with the integral term providing error accumulation that smooths noisy proportional corrections.

4. Full PID does not outperform any single-gain variant, suggesting that naively combining all three gains introduces conflicting correction signals.

## 6.4 Alignment Signal Quality: Batch-Size Sweep

We sweep batch sizes $\{64, 128, 256, 512, 1024\}$ on CIFAR-100/VGG-16-BN (10 epochs, seed 42) for FixedWD, CWD, and UDWDC to test Hypothesis 3 (batch-size-dependent alignment signal quality).

The alignment signal-to-noise ratio $\text{SNR} = |\mathbb{E}[\alpha_t]| / \text{Std}[\alpha_t]$ increases monotonically with batch size for both FixedWD and CWD, confirming that larger batches yield cleaner alignment estimates.  UDWDC shows a non-monotonic SNR pattern: SNR increases from batch size 64 to 256, then plateaus or slightly decreases at 512--1024.  This non-monotonicity arises because the proportional controller's per-layer feedback introduces additional variance in the alignment signal that does not average out at larger batch sizes.

The monotonic SNR scaling for CWD supports the recommendation that binary masking (CWD-style) is preferable at $b \leq 256$, where the raw alignment signal is noisy.  Continuous alignment modulation may provide marginal improvement at $b \geq 1024$, but the gains are modest and UDWDC's proportional instability limits its effectiveness.

![Alignment SNR vs batch size](figures/alignment_snr.pdf)

*Figure 4: Alignment signal-to-noise ratio vs. batch size for FixedWD, CWD, and UDWDC on CIFAR-100/VGG-16-BN.  SNR increases monotonically for FixedWD and CWD.  UDWDC's non-monotonic pattern reflects proportional controller noise.*

## 6.5 ImageNet Main Results (ResNet-50, 90 epochs)

Table 5 reports the ImageNet/ResNet-50 results, the primary large-scale evidence for the framework.

| Method | Best Val Acc (%) | Final Val Acc (%) | WD Budget | Seeds |
|--------|----:|----:|----:|----:|
| **CPR** | **74.74 $\pm$ 0.05** | **74.74 $\pm$ 0.04** | 1.20 | 3 |
| FixedWD | 71.72 $\pm$ 0.36 | 71.70 $\pm$ 0.37 | 0.52 | 5 |
| CWD | 70.66 | 70.58 | 0.26 | 1 |
| UDWDC | 69.93 $\pm$ 0.19 | 69.91 $\pm$ 0.19 | 0.70 | 3 |

*Table 5: ImageNet/ResNet-50, 90 epochs.  CPR leads by 3.02 pp over FixedWD, a substantial margin driven by its integral control mechanism.  UDWDC underperforms FixedWD by 1.79 pp, consistent with the CIFAR results indicating proportional-only control is insufficient.  CWD has only 1 completed seed due to compute constraints.*

**Key observations.**

1. CPR's dominance is even more pronounced on ImageNet (+3.02 pp over FixedWD) than on CIFAR-10 (+1.06 pp).  The augmented Lagrangian's integral control accumulates a WD budget $2.3\times$ that of FixedWD, producing tighter weight norm constraints that substantially improve generalization at ImageNet scale.

2. UDWDC underperforms FixedWD by 1.79 pp on ImageNet (69.93% vs. 71.72%), a larger gap than on CIFAR-10 (0.53 pp).  The proportional controller's instability is amplified at scale, where per-layer $\rho_t$ variance is higher due to the larger number of layers (65 weight-bearing layers in ResNet-50 vs. 24 in ResNet-20).

3. CWD's single-seed result (70.66%) falls between FixedWD and UDWDC, consistent with CWD's WD magnitude reduction (50% effective $\lambda$) reducing regularization strength.

4. FixedWD achieves 71.72 $\pm$ 0.36% with $\lambda = 10^{-4}$.  Standard ResNet-50 training recipes typically use $\lambda = 10^{-4}$ with SGD+momentum and achieve 76--77% at 90 epochs.  The lower accuracy here reflects our minimal augmentation protocol (no mixup, cutmix, or RandAugment for CNNs) designed to isolate WD effects from augmentation interactions.

## 6.6 Temporal Predictability of Alignment (H7)

The temporal predictability gate (H7) tests whether the alignment signal $\alpha_t$ carries information beyond what a polynomial function of time can predict.  We fit degree-4 polynomials to $\alpha_t$ trajectories across all layer-method combinations (2640 total: 8 methods $\times$ 330 layer-seed traces from CIFAR-10 and CIFAR-100 experiments).

**Result**: 18.1% of layer-method combinations achieve $R^2 > 0.85$ for polynomial fit, and 28.0% achieve $R^2 > 0.70$.  The mean $R^2$ is 0.279 (median 0.036).  Convolution weight layers show $R^2 = 0.595$ (median 0.825), while BN and bias layers show $R^2 < 0.04$.  H7 passes: the alignment signal is *not* well-approximated by a time-only polynomial for most layer types, confirming it carries temporally complex information.

The bimodal $R^2$ distribution (69.2% below 0.30 but 18.1% above 0.85) reveals a structural split: alignment is highly predictable from time in some layers (primarily large convolutional kernels) but carries non-temporal information in others (BN parameters, biases, shortcut connections).  This suggests that alignment-aware WD methods like CWD derive their value specifically from the layers where time-polynomial fitting fails.

## 6.7 CSI Stability Analysis

The refined CSI (Section 5.2) reveals stability differences invisible to accuracy-only evaluation.

| Method | CSI$_\text{temporal}$ | CSI$_\text{spatial}$ | CSI$_\text{combined}$ | Interpretation |
|--------|----------------------:|---------------------:|----------------------:|----------------|
| FixedWD | 1.000 | 1.000 | 1.000 | Trivially stable |
| DefazioCorrective | 1.000 | 1.000 | 1.000 | Highly stable |
| CWD | 0.997 | 0.997 | 0.997 | Highly stable |
| CPR | 0.957 | 1.000 | 0.978 | Highly stable |
| SWD | 0.902 | 0.907 | 0.904 | Stable |
| UDWDC | $-$5.75 | 0.935 | $-$2.41 | Highly unstable |

*Table 6: CSI stability comparison on CIFAR-10/ResNet-20 (pilot, 10 epochs).  FixedWD is trivially stable (constant $\lambda$).  CWD maintains near-perfect stability despite the binary mask.  UDWDC's negative temporal CSI indicates that the proportional controller's WD coefficient varies more than the underlying $\rho_t$ trajectory, amplifying rather than damping perturbations.*

UDWDC's negative temporal CSI ($-$5.75) is a quantitative signature of proportional controller instability.  The spatial CSI (0.935) remains positive because per-layer $\rho_t$ variation at each step is consistent across layers --- the instability is temporal (oscillation over training steps), not spatial (inconsistency across layers).

![CSI comparison between methods](figures/csi_comparison.pdf)

*Figure 7: CSI comparison across methods.  UDWDC exhibits negative temporal CSI, quantifying the instability documented in Section 3.3.  All other methods maintain CSI $>$ 0.9.*

<!-- FIGURES
- Figure 4: gen_alignment_snr.py, alignment_snr.pdf --- Alignment SNR vs batch size
- Figure 7: gen_csi_comparison.py, csi_comparison.pdf --- CSI stability comparison across methods
- Table 3: inline --- CIFAR-10 main results
- Table 4: inline --- UDWDC gain ablation
- Table 5: inline --- ImageNet main results
- Table 6: inline --- CSI stability comparison
-->
