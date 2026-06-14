# 7. Discussion

## 7.1 Unification Works at the Family Level

The PID parameterization provides a useful taxonomy even where exact fitting fails.  CWD (4.71% fitting error) and CPR (9.57%) are well-captured by the control law, confirming that alignment-based and constraint-based methods operate as derivative and integral controllers, respectively.  SWD (45.81%) and DefazioCorrective (37.56%) resist fitting because they use global gradient statistics or feedforward scheduling rather than per-layer measured $\rho_t$.

The family-level interpretation remains valid: **scheduling-based methods** (SWD, DefazioCorrective) are open-loop or feedforward controllers that prescribe $\lambda_t$ from the training schedule without sensing $\rho_t$; **alignment-based methods** (CWD) add a derivative correction based on geometric signal quality; **constraint-based methods** (CPR) add integral control via penalty accumulation.  This three-way taxonomy clarifies why CPR and CWD achieve different results despite both being "dynamic WD": they control different aspects of the training dynamics.

## 7.2 BEM Reveals Hidden Cost-Effectiveness Differences

Raw accuracy rankings mask important efficiency differences.  On CIFAR-10 (10-epoch pilot), UDWDC achieves rank-1 BEM (55.87) despite rank-2 accuracy (81.78%), while FixedWD achieves rank-1 accuracy (82.06%) but rank-2 BEM (45.42%).  CPR achieves the highest accuracy on both CIFAR-10 (91.74%) and ImageNet (74.74%) but does so by accumulating $10\times$ the WD budget of FixedWD, making its BEM relatively low (0.39 on the 200-epoch data).

This ranking divergence validates the need for multi-dimensional evaluation.  A practitioner choosing a WD method based solely on accuracy would select CPR; one operating under a regularization budget constraint (e.g., in federated learning where WD budget affects privacy guarantees) would select FixedWD or UDWDC.  BEM makes this tradeoff explicit.

## 7.3 The CWD Magnitude Confound

CWD's effective WD is approximately 50% of FixedWD (0.24 vs. 0.48 on CIFAR-10, 0.26 vs. 0.52 on ImageNet).  CWD applies the binary mask $\mathbb{1}[\text{sign}(g_t) = \text{sign}(w_t)]$ element-wise, disabling WD for approximately half the parameters at each step.  This halving raises a confound: does CWD improve performance through alignment information or through WD magnitude reduction?

On CIFAR-10, CWD (90.32 $\pm$ 0.08%) underperforms FixedWD (90.68 $\pm$ 0.11%), suggesting that the magnitude reduction actually hurts on this benchmark.  On ImageNet, CWD's single-seed result (70.66%) falls between FixedWD (71.72%) and UDWDC (69.93%), again below the fixed baseline.  These results suggest that the 50% WD reduction is the dominant effect, and the alignment masking does not compensate for the reduced regularization strength at $\lambda_{\text{base}} = 10^{-4}$.

CWD's published gains (+0.61% on ImageNet ViT-S/16) use $\lambda_{\text{base}} = 0.05$, a $500\times$ larger base coefficient.  At higher $\lambda_{\text{base}}$, halving the effective WD may be beneficial because it reduces over-regularization.  A definitive resolution requires a CWD vs. halved-$\lambda$ ablation: running FixedWD at $\lambda = 0.5 \times \lambda_{\text{base}}$ and comparing directly.  We did not complete this ablation due to compute constraints, and flag it as essential future work.

## 7.4 UDWDC Instability Is a Genuine Finding

UDWDC v1's negative CSI ($-$2.41) and accuracy below NoWD (90.15% vs. 90.25% on CIFAR-10) are not implementation bugs but a fundamental limitation of proportional-only control applied to per-layer $\rho_t$.

The failure mechanism is clear: when $\rho_t^l < \rho^*(t)$ (which occurs systematically for BN layers where gradient norms are small relative to weight norms), the ratio $\rho_t^l / \rho^*(t) < 1$, and the clamp drives $\lambda_t^l$ toward $0.1 \times \lambda_{\text{base}}$.  Across 65 layers, this reduces the effective WD to near-zero, producing a WD budget of 0.38 (vs. FixedWD's 0.48) without the corresponding accuracy benefit.

The v2 floor fix recovers competitive accuracy (90.36%) but at the cost of an enormous WD budget (98599), because the floor clipping applies $\lambda_{\min} = 0.1 \times \lambda_{\text{base}}$ to all 65 layers including BN layers at every step.  This is clearly suboptimal: future work should restrict UDWDC's control loop to weight layers only, excluding BN parameters.

The instability finding has broader implications: **per-layer proportional control of WD is inherently noisy** because $\rho_t^l$ fluctuates substantially within each epoch due to mini-batch stochasticity.  The EMA smoothing in v2 ($\beta = 0.99$) helps but does not eliminate the issue.  CPR's success demonstrates that integral control (penalty accumulation) is a more robust feedback mechanism for WD: it inherently smooths noise through integration and drives the system toward the target monotonically.

## 7.5 Alignment Signal Quality Is Batch-Size Dependent

The alignment SNR increases monotonically with batch size for FixedWD and CWD, confirming that larger batches produce cleaner alignment estimates.  UDWDC's non-monotonic SNR reflects the proportional controller adding noise to the alignment measurement.

This batch-size dependence has practical implications: alignment-aware WD methods (CWD, any future continuous modulation) should be preferred at large batch sizes ($b \geq 512$) where the alignment signal is reliable.  At small batch sizes ($b \leq 256$), the alignment noise may outweigh the information gain, and simple scheduling (SWD-style) or constraint-based (CPR-style) methods are more robust.

## 7.6 Honest Negative Results

1. **UDWDC does not beat tuned FixedWD on accuracy.** On all three benchmarks, FixedWD outperforms UDWDC: by 0.53 pp on CIFAR-10, by 1.26 pp on CIFAR-100 ablation (FixedWD 70.53% vs. Kp_only 68.52%), and by 1.79 pp on ImageNet.  UDWDC's value lies in the theoretical framework and BEM efficiency, not raw performance.

2. **H1 fitting fails for 2/5 methods.** SWD (45.81%) and DefazioCorrective (37.56%) exceed the 20% error threshold.  The unification is valid at the family level (alignment vs. scheduling vs. constraint) but not as a precise quantitative model for all methods.

3. **CPR's integral control produces 10x higher effective WD.** The augmented Lagrangian is aggressive: CPR's WD budget (4.44 on CIFAR-10) is $9.3\times$ FixedWD's (0.48).  While this produces the highest accuracy, it raises questions about whether CPR's gains come from better WD allocation or simply from stronger regularization.

4. **ImageNet results have limited seed coverage.** CWD has only 1 completed seed on ImageNet; UDWDC has 3.  Statistical comparisons (paired t-tests, TOST) require matched seed counts, limiting the conclusions we can draw about CWD vs. UDWDC on ImageNet.

## 7.7 Limitations

- Proposition 3 (layer-differentiated steady states) is restricted to BN architectures; LN behavior may differ.
- UDWDC's proportional-only design inherits steady-state offset that an integral term could eliminate.
- Budget-matched controls for ImageNet are limited to 2-epoch pilots; full 90-epoch budget matching is needed.
- AIS is computed on CIFAR batch sizes; ImageNet AIS requires additional experiments.
- The PID gain mapping is descriptive: it demonstrates that existing methods can be expressed within the framework, but does not prove that the PID parameterization is optimal.

## 7.8 Future Work

1. **Adaptive gain scheduling.** Let $K_p$, $K_i$, $K_d$ vary with training phase: higher $K_p$ during warmup for rapid adjustment, increasing $K_i$ during stable training for offset elimination, and $K_d$ conditioned on batch-size-dependent alignment quality.

2. **Geometry-corrected alignment (Proposition 2) for Adam-family optimizers.** Testing $\delta_t^P$ vs. $\alpha_t$ as the CWD mask criterion with AdamW on transformer architectures.

3. **Extension to language models.** Does the PID framework predict which WD schedule works for GPT-scale pretraining?  CPR's integral control has already shown gains on GPT-2 (Franke et al., 2024); testing whether the PID taxonomy explains this is a natural next step.

4. **Joint $(\eta_t, \lambda_t)$ control.** Multi-input-multi-output (MIMO) control of learning rate and WD simultaneously, using $\rho_t$ and $\alpha_t$ as feedback signals, represents the natural extension of the single-input framework presented here.

<!-- FIGURES
- None
-->
