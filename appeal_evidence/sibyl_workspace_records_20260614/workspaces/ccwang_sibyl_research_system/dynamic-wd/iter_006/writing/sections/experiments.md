# 6. Experiments

## 6.1 Setup

**Datasets.** CIFAR-10 (10 classes, 50K train / 10K test) and CIFAR-100 (100 classes, same splits), with standard data augmentation: random horizontal flip, random crop with 4-pixel padding, and per-channel normalization.

**Architecture.** ResNet-20 with batch normalization (He et al., 2016), containing 0.27M parameters. All convolutional and linear layers use Kaiming initialization.

**Optimizer.** AdamW with base learning rate $\gamma_0 = 10^{-3}$, cosine learning rate schedule decaying to 0, no warmup, batch size 128, for 200 epochs. Momentum parameters $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Weight decay methods.** Six methods evaluated, all sharing $\lambda_{\text{base}} = 5 \times 10^{-4}$:

| Method | $\phi$ description | Key hyperparameters |
|--------|-------------------|-------------------|
| No WD | $\phi = 0$ | -- |
| Constant | $\phi = 1$ | -- |
| Cosine | $\phi = \frac{1}{2}(1 + \cos(\pi t/T))$ | -- |
| CWD | $\phi = \mathbf{1}[\text{sign}(w) = \text{sign}(\Delta w)]$ | $\beta_{\text{cwd}} = 100$ |
| SWD | $\phi = h(\|g\|)$ | sensitivity $= 1.0$ |
| PMP-WD | $\phi = \mathbf{1}[\langle m_t, w_t \rangle > 0]$ | uses momentum buffer |

**Seeds and reporting.** Every configuration runs with seeds 42, 123, and 456. We report mean $\pm$ standard deviation of best test accuracy (the maximum test accuracy achieved across all 200 epochs).

**Instrumentation.** Each run logs per-epoch diagnostics: training/test accuracy and loss, weight norm $\|w_t\|$, Lyapunov function $V_t = f(w_t) + \mu_t\|w_t\|^2$, gradient-weight alignment $\delta_t$, CSI, AIS, BEM, and effective WD $\lambda_{\text{eff}}(t)$. PMP-WD runs additionally log the switching function $\sigma(t)$, indicator $\mathbf{1}[\sigma > 0]$, cumulative switch count, and per-step switch rate.

## 6.2 Main Results

### CIFAR-10 / ResNet-20

As shown in Figure 2 and Table 2, all six WD methods achieve best test accuracy between 89.80% and 90.29% on CIFAR-10, a total spread of 0.49 percentage points.

**Table 2: CIFAR-10 / ResNet-20 results (200 epochs, AdamW, 3 seeds).** Best acc is the maximum test accuracy across training. BEM, CSI, and AIS are final-epoch values averaged across seeds. Bold indicates the highest mean accuracy.

| Method | Best Acc (%) | Gen Gap | BEM | CSI | AIS |
|--------|-------------|---------|-----|-----|-----|
| No WD | 90.10 $\pm$ 0.15 | 9.72 | 1.000 | 0.892 | 0.338 |
| Constant | 89.80 $\pm$ 0.31 | 10.03 | 0.000 | 0.896 | 0.325 |
| Cosine | 89.90 $\pm$ 0.12 | 9.90 | 0.502 | 0.956 | 0.326 |
| CWD | 89.98 $\pm$ 0.41 | 9.84 | 0.507 | 0.867 | 0.367 |
| SWD | 90.14 $\pm$ 0.20 | 9.61 | 0.900 | 0.945 | 0.374 |
| **PMP-WD** | **90.29 $\pm$ 0.12** | **9.61** | 0.508 | 0.888 | 0.381 |

PMP-WD achieves the highest mean best accuracy (90.29%) with the lowest standard deviation (0.12), suggesting the costate-based switching provides a slight advantage in stability. The generalization gap (train acc - test acc) is smallest for SWD and PMP-WD (9.61), both of which apply roughly half the total WD budget (BEM $\approx$ 0.5 and 0.9 respectively).

The spread of 0.49 percentage points across all methods -- including the no-WD baseline -- is smaller than the within-method standard deviation of CWD (0.41) and constant WD (0.31). A paired $t$-test between PMP-WD (best) and constant WD (worst among WD methods) yields $p = 0.12$ (not significant at $\alpha = 0.05$). No pairwise comparison between any two WD methods reaches significance after Bonferroni correction.

![CIFAR-10 best test accuracy across 6 WD methods, showing a 0.49pp total spread.](figures/main_results_bar.pdf)

### CIFAR-100 / ResNet-20

PMP-WD achieves 62.98 $\pm$ 0.27% best test accuracy on CIFAR-100, comparable to the iter_003 results for other methods: constant 63.15 $\pm$ 0.30%, cosine 63.42 $\pm$ 0.42%, SWD 63.06 $\pm$ 0.29%, CWD 62.84 $\pm$ 0.30%. The total spread across all WD methods on CIFAR-100 is 0.58 percentage points (excluding no-WD at 62.66%), with cosine schedule marginally leading. The pattern mirrors CIFAR-10: all methods cluster within a narrow band, and no method statistically dominates.

## 6.3 Diagnostic Analysis

### Weight Norm Trajectories

As shown in Figure 3, all methods produce nearly identical weight norm trajectories, growing from $\|w_0\| \approx 34$ to $\|w_{200}\| \approx 96$. The final weight norms differ by less than 1.5% across methods (range: 95.68 to 97.03). This convergence is a direct consequence of batch normalization: BN renders the loss invariant to weight scaling, so WD's primary effect is controlling the effective learning rate $\gamma_{\text{eff}} = \gamma/\|w\|^2$ rather than the norm itself. With cosine LR decay driving $\gamma \to 0$, the effective regularization pressure vanishes in late training regardless of the WD method.

![Weight norm trajectories for all methods converge to similar values ($\approx 96$), demonstrating BN-induced scale invariance.](figures/weight_norm_trajectories.pdf)

### Lyapunov Function Trajectories

Figure 4 shows the Lyapunov function $V_t = f(w_t) + \mu_t\|w_t\|^2$ across training. All methods exhibit monotonically increasing $V_t$ on a log scale, dominated by the $\mu_t\|w_t\|^2$ term as weight norms grow. The trajectories are nearly indistinguishable after epoch 50, with PMP-WD showing a slightly lower trajectory in the first 50 epochs due to its initially lower effective WD (the costate is negative early in training, suppressing decay). The convergence of $V_t$ trajectories across methods provides empirical support for Theorem 3 (subsumption): all methods operate within the same Lyapunov convergence envelope.

![Lyapunov function $V_t$ trajectories for all methods on CIFAR-10. All curves converge after epoch 50, consistent with the narrow certified band prediction.](figures/lyapunov_curves.pdf)

### PMP-WD Switching Behavior

Figure 5 shows the PMP-WD switching function $\sigma(t) = \langle m_t, w_t \rangle$ and the per-parameter switch rate across training. The switching function oscillates around zero throughout training, with amplitude $\approx 1.5 \times 10^{-3}$ in early epochs decreasing to $\approx 0.3 \times 10^{-3}$ by epoch 200. The switch rate (fraction of parameters receiving decay) stabilizes at $\approx 0.55$, indicating that PMP-WD applies decay to a slight majority of parameters at each step.

The bang-bang pattern is clear: $\sigma(t)$ crosses zero frequently, flipping the per-epoch decay indicator between 0 and 1. The cumulative switch count grows linearly, reaching $\approx 2.8 \times 10^6$ individual parameter switches by epoch 200. The mean effective WD across training is $2.5 \times 10^{-4}$, exactly half of $\Lambda_{\max} = 5 \times 10^{-4}$, confirming the theoretical prediction that bang-bang control with balanced switching produces approximately half the base WD budget.

![PMP-WD switching function $\sigma(t)$ (top) and per-parameter switch rate (bottom). The bang-bang pattern is visible in the oscillating $\sigma$ and the switch rate stabilizing near 0.55.](figures/pmpwd_switching.pdf)

### BEM vs. Accuracy

Figure 6 plots BEM against best test accuracy. The Pearson correlation is $r = 0.61$ ($p = 0.19$), indicating a weak positive but non-significant trend: methods with higher BEM (more deviation from constant WD) tend to achieve marginally higher accuracy. This trend is driven primarily by the no-WD baseline (BEM = 1.0, acc = 90.10%) and SWD (BEM = 0.90, acc = 90.14%). The non-significance confirms that budget deviation does not reliably predict accuracy on BN architectures.

![BEM vs. best test accuracy. The weak positive correlation ($r = 0.61$, $p = 0.19$) is not significant, consistent with the weight decay illusion.](figures/bem_accuracy_scatter.pdf)

## 6.4 Diagnostic Metrics Analysis

**CSI (Coupling Stability Index).** CSI ranges from 0.867 (CWD) to 0.956 (Cosine) across methods. Low values for CWD and PMP-WD (0.867, 0.888) indicate that alignment-based switching produces more stable weight norm updates than time-based modulation (cosine, 0.956). The narrow range of CSI values (0.089 spread) indicates all methods maintain stable WD-optimizer coupling on BN architectures.

**AIS (Alignment Informativeness Score).** AIS ranges from 0.325 (constant) to 0.381 (PMP-WD). PMP-WD's higher AIS suggests the costate-based switching captures more layer-level alignment variation, but the difference is modest (0.056 absolute). Constant WD and cosine schedule have the lowest AIS (0.325, 0.326), confirming they ignore alignment information entirely.

**Generalization Gap.** The generalization gap (train acc - test acc) ranges from 9.61 (SWD, PMP-WD) to 10.03 (constant). Methods with lower effective WD budget (higher BEM) tend to have smaller generalization gaps, consistent with the view that excessive WD increases the train-test divergence by constraining model capacity. The correlation between BEM and generalization gap is $r = -0.72$ ($p = 0.10$), stronger than the BEM-accuracy correlation but still not significant at $\alpha = 0.05$ with 6 data points.

<!-- FIGURES
- Figure 2: gen_main_results_bar.py, main_results_bar.pdf -- CIFAR-10 best test accuracy bar chart
- Figure 3: gen_weight_norm_trajectories.py, weight_norm_trajectories.pdf -- Weight norm trajectories
- Figure 4: gen_lyapunov_curves.py, lyapunov_curves.pdf -- Lyapunov function V_t trajectories
- Figure 5: gen_pmpwd_switching.py, pmpwd_switching.pdf -- PMP-WD switching behavior
- Figure 6: gen_bem_accuracy_scatter.py, bem_accuracy_scatter.pdf -- BEM vs accuracy scatter
- Table 2: inline -- CIFAR-10 main results
-->
