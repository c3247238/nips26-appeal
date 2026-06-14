# 4. Experimental Setup

## 4.1 Implementation

All experiments use a unified `UnifiedAdamW` (and `UnifiedSGD`) optimizer class with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring that all methods share identical optimizer internals and differ only in the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024).

We evaluate seven primary weight decay methods spanning the four modulation axes: **constant** (baseline, $\varphi \equiv 1$), **cwd\_hard** (Cautious Weight Decay with binary sign-alignment mask), **swd** (Scheduled Weight Decay with gradient-norm sensitivity), **cosine\_schedule** (cosine-annealed weight decay), **random\_mask** (Bernoulli mask control with $p = 0.5$), **half\_lambda** (constant weight decay at $\lambda/2$, a budget-matched control), and **no\_wd** (weight decay disabled, $\varphi \equiv 0$).

## 4.2 Training Configuration

**Datasets.** CIFAR-10 and CIFAR-100 (Krizhevsky, 2009), loaded via torchvision with standard train/test splits (50,000/10,000 images). Standard data augmentation: random cropping with 4-pixel padding, random horizontal flipping, and per-channel normalization.

**Architectures.** (a) ResNet-20 in the standard CIFAR configuration ($\sim$270K parameters) with batch normalization (He et al., 2016). (b) VGG-16-BN ($\sim$15M parameters) with batch normalization, providing a second architecture without skip connections.

**Optimizers.** (a) AdamW with decoupled weight decay: $\eta = 10^{-3}$, cosine annealing to zero, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. (b) SGD with momentum 0.9: $\eta = 0.1$, cosine annealing to zero.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for all methods and optimizers. Each dynamic method modulates this value through its Phi function.

**Training duration.** 200 epochs with batch size 128.

**Seeds.** Three independent runs per configuration using seeds $\{42, 123, 456\}$, controlling Python, NumPy, PyTorch, and CUDA random number generators.

**Total experiments.** 126 runs: 7 methods $\times$ 3 seeds $\times$ 2 datasets $\times$ 2 optimizers (AdamW, SGD) for ResNet-20, plus 7 methods $\times$ 3 seeds $\times$ 1 dataset (CIFAR-10) for VGG-16-BN under SGD.

## 4.3 Hyperparameter Fairness Protocol

**All methods use identical base hyperparameters** ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$ for AdamW, $\eta = 0.1$ for SGD) with no per-method grid search. This ensures that observed differences measure the effect of Phi modulation, not hyperparameter luck. Each method may operate at a non-optimal hyperparameter configuration, but this is the price of a fair comparison. We discuss this limitation in Section 6.3.

## 4.4 Diagnostic Logging

We log per-epoch: test accuracy, training loss, per-layer weight norms, CSI, AIS, and BEM. Per-100-step snapshots record gradient-weight cosine similarity per layer, effective learning rate per layer, and Phi modulation values.

---

# 5. Results and Analysis

## 5.1 AdamW: Main Accuracy Comparison

Table 2 presents the AdamW results on ResNet-20. On CIFAR-10, the seven methods achieve mean test accuracies spanning 0.25 percentage points: from 89.88% (SWD) to 90.13% (constant). On CIFAR-100, the range is 0.76 percentage points: from 62.66% (no\_wd) to 63.42% (cosine\_schedule). The constant baseline is the best or near-best method on CIFAR-10; cosine\_schedule leads on CIFAR-100 by 0.27%---neither advantage is statistically significant.

**Table 2: AdamW + ResNet-20 accuracy results.** Mean $\pm$ standard deviation over 3 seeds. Best result per dataset in **bold**.

| Method | CIFAR-10 Acc. (%) | CIFAR-100 Acc. (%) |
|--------|:-----------------:|:------------------:|
| constant | **90.13** $\pm$ 0.31 | 63.15 $\pm$ 0.30 |
| cosine\_schedule | 90.12 $\pm$ **0.07** | **63.42** $\pm$ 0.42 |
| random\_mask | 90.12 $\pm$ 0.30 | 62.87 $\pm$ 0.38 |
| half\_lambda | 90.09 $\pm$ 0.29 | 62.91 $\pm$ 0.47 |
| no\_wd | 90.08 $\pm$ 0.32 | 62.66 $\pm$ 0.38 |
| cwd\_hard | 90.06 $\pm$ 0.24 | 62.84 $\pm$ 0.30 |
| swd | 89.88 $\pm$ 0.25 | 63.06 $\pm$ 0.29 |

Cosine\_schedule achieves the lowest variance on CIFAR-10 ($\sigma = 0.07\%$, compared to $\sigma \approx 0.25$--$0.32\%$ for all other methods), suggesting a potential stability benefit without an accuracy advantage.

Table 3 reports paired $t$-tests of each method against the constant baseline. All $p$-values exceed 0.05 (range: $p = 0.090$ to $p = 0.950$). After Bonferroni correction for six comparisons (significance threshold $p < 0.0083$), no method survives. All Cohen's $d$ effect sizes are below 0.3.

**Table 3: Statistical tests vs. constant baseline (AdamW + ResNet-20).** All comparisons are not statistically significant.

| Method | CIFAR-10 $\Delta$ | $p$-value | CIFAR-100 $\Delta$ | $p$-value |
|--------|:-----------------:|:---------:|:------------------:|:---------:|
| cwd\_hard | $-0.07\%$ | 0.832 | $-0.31\%$ | 0.326 |
| swd | $-0.25\%$ | 0.513 | $-0.10\%$ | 0.801 |
| cosine\_schedule | $-0.01\%$ | 0.935 | $+0.26\%$ | 0.117 |
| random\_mask | $-0.01\%$ | 0.950 | $-0.29\%$ | 0.090 |
| half\_lambda | $-0.05\%$ | 0.901 | $-0.25\%$ | 0.608 |
| no\_wd | $-0.05\%$ | 0.825 | $-0.49\%$ | 0.312 |

The no\_wd ablation is striking: removing weight decay entirely ($\lambda = 0$) yields CIFAR-10 accuracy of 90.08% versus the constant baseline's 90.13% ($\Delta = 0.05\%$, $p = 0.825$). Under AdamW, weight decay magnitude is largely irrelevant at this scale.

**Equivalence testing (TOST).** We apply Two One-Sided Tests with practical equivalence margins $\delta = \pm 0.5\%$ and $\delta = \pm 1.0\%$. At $\delta = \pm 1.0\%$, 6 of 12 method--dataset comparisons confirm equivalence ($p_{\mathrm{TOST}} < 0.05$), including cwd\_hard, cosine\_schedule, random\_mask, and no\_wd on CIFAR-10. With $N = 3$ seeds, our study has 80% power to detect only effects $\geq 0.7\%$; smaller effects remain unresolvable.

## 5.2 SGD: The Boundary Condition

Table 4 presents SGD results on ResNet-20. The pattern differs from AdamW: methods now show a wider accuracy spread with meaningful weight norm variation.

**Table 4: SGD + ResNet-20 accuracy results.** Mean $\pm$ std over 3 seeds.

| Method | CIFAR-10 Acc. (%) | CIFAR-100 Acc. (%) |
|--------|:-----------------:|:------------------:|
| **constant** | **91.22** $\pm$ 0.06 | **65.37** $\pm$ 0.13 |
| cosine\_schedule | 91.20 $\pm$ 0.10 | 65.11 $\pm$ 0.25 |
| cwd\_hard | 90.87 $\pm$ 0.35 | 64.37 $\pm$ 0.47 |
| half\_lambda | 90.84 $\pm$ 0.15 | 64.86 $\pm$ 0.38 |
| random\_mask | 90.77 $\pm$ 0.37 | 64.91 $\pm$ 0.40 |
| swd | 90.71 $\pm$ 0.16 | 64.30 $\pm$ 0.41 |
| no\_wd | 90.30 $\pm$ 0.08 | 63.66 $\pm$ 0.17 |

Under SGD, the no\_wd ablation drops by 0.92% on CIFAR-10 and 1.71% on CIFAR-100 relative to constant---both practically significant. The constant baseline achieves the highest accuracy on both datasets. The CIFAR-10 spread is 0.92% (vs. 0.25% under AdamW); the CIFAR-100 spread is 1.71% (vs. 0.76% under AdamW).

**Weight norm divergence under SGD.** The mechanistic explanation is clear from weight norms: under SGD, no\_wd produces mean final weight norms of 127.06 (CIFAR-10) vs. 64.57 for constant---a 97% increase. Under AdamW, the same comparison shows 97.04 vs. 95.89---only a 1.2% increase. SGD lacks per-parameter adaptive scaling, so weight decay's explicit norm control is the dominant force governing weight norm dynamics. Without it, norms grow substantially, degrading generalization.

This result validates the Phi Invariance Conjecture's predicted boundary condition: the conjecture holds for AdamW (which has adaptive per-parameter scaling) but fails for SGD (which does not).

## 5.3 Cross-Architecture: VGG-16-BN

Table 5 reports SGD + VGG-16-BN results on CIFAR-10, testing whether the Phi Invariance result transfers across architectures.

**Table 5: SGD + VGG-16-BN accuracy results on CIFAR-10.** Mean $\pm$ std over 3 seeds.

| Method | Accuracy (%) |
|--------|:------------:|
| half\_lambda | **92.15** $\pm$ 0.11 |
| swd | 92.11 $\pm$ 0.23 |
| cwd\_hard | 92.06 $\pm$ 0.21 |
| constant | 92.05 $\pm$ 0.05 |
| random\_mask | 92.05 $\pm$ 0.22 |
| no\_wd | 92.03 $\pm$ 0.04 |
| cosine\_schedule | 91.99 $\pm$ 0.26 |

The total spread is 0.16 percentage points---even narrower than ResNet-20 under AdamW (0.25pp). The no\_wd ablation (92.03%) is essentially identical to constant (92.05%). VGG-16-BN with batch normalization exhibits the same Phi Invariance behavior seen in ResNet-20 under AdamW, but now also under SGD.

This result suggests that **batch normalization, not the optimizer's adaptive scaling, is the primary mechanism enabling Phi Invariance in VGG-16-BN.** Batch normalization provides its own implicit weight norm control: it normalizes activations regardless of weight scale, decoupling the effective update from the weight magnitude. The combination of BN + SGD achieves the same invariance that AdamW achieves through adaptive per-parameter scaling.

![Multi-architecture comparison showing accuracy spread across ResNet-20 and VGG-16-BN](figures/multi_arch_comparison.png)

## 5.4 Budget Equivalence Analysis

As shown in Figure 3, BEM values span the full range from 0.0 (constant, half\_lambda) through 0.5 (cosine\_schedule, cwd\_hard, random\_mask) to 0.9 (SWD) and 1.0 (no\_wd). Despite this ten-fold variation in effective weight decay budget, accuracy remains essentially flat under AdamW.

On CIFAR-10 (AdamW), a 10$\times$ variation in effective weight decay budget (BEM range $0.0$--$1.0$) produces less than 0.5% accuracy variation. On CIFAR-100, the spread is 0.76%. Under SGD on CIFAR-100, the same BEM range produces a 1.71% spread---three times larger.

![BEM vs. accuracy scatter plot showing flat relationship under AdamW and steeper relationship under SGD](figures/fig3_bem_vs_accuracy.png)

## 5.5 CSI and AIS Diagnostic Analysis

**CSI analysis.** Table 6 reports Coupling Stability Index values for AdamW + ResNet-20 on CIFAR-10. CSI ranges from 0.838 (SWD, most stable) to 0.964 (no\_wd, least stable). The no\_wd method has the highest CSI because weight norms grow freely without decay. Critically, **CSI does not predict accuracy**: the Spearman rank correlation between CSI and accuracy rank is $\rho < 0.3$ ($p > 0.3$) on both datasets. CSI captures differences in training dynamics, but these dynamics differences do not translate to performance differences under AdamW.

**AIS analysis.** All methods show AIS values in the range 0.280--0.410. The key finding: AIS is **consistent across all weight decay methods**, including random\_mask and no\_wd. The gradient-weight alignment signal has moderate predictive power for loss changes, but this predictive power is an intrinsic property of the network and loss landscape---it is not generated or exploited by weight decay modulation.

This directly challenges CWD's motivation: if AIS is the same for CWD (which conditions on alignment) and random\_mask (which ignores alignment entirely), the alignment signal provides no additional useful information for weight decay decisions.

**Table 6: Diagnostic metrics (AdamW + ResNet-20, CIFAR-10).**

| Method | CSI | AIS | BEM |
|--------|:---:|:---:|:---:|
| constant | 0.841 | 0.336 | 0.000 |
| cosine\_schedule | 0.936 | 0.352 | 0.503 |
| random\_mask | 0.892 | 0.359 | 0.500 |
| half\_lambda | 0.853 | 0.410 | 0.000 |
| no\_wd | 0.964 | 0.343 | 1.000 |
| cwd\_hard | 0.851 | 0.368 | 0.503 |
| swd | 0.838 | 0.360 | 0.900 |

![Diagnostic metrics heatmap showing CSI, AIS, and BEM across methods](figures/fig4_diagnostic_heatmap.png)

## 5.6 Weight Norm Analysis

Under AdamW, all seven methods converge to similar weight norm levels: final mean weight norms range from 95.89 (constant) to 97.04 (no\_wd), a difference of 1.2%. AdamW's adaptive per-parameter step size $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ scales updates inversely with gradient magnitude. When weight decay drives a parameter toward zero, AdamW compensates by increasing the effective step size. Conversely, without weight decay, parameters grow slightly larger, but AdamW's second-moment normalization prevents effective updates from becoming disproportionate.

Under SGD, the picture changes dramatically. The constant baseline produces final weight norms of 64.57 on CIFAR-10, while no\_wd reaches 127.06---a 97% increase. Methods with reduced effective WD (half\_lambda, cosine\_schedule, cwd\_hard, random\_mask) all produce elevated norms of 83--84, reflecting the direct proportionality between WD budget and weight norm control. SWD, which applies near-zero WD for most of training, reaches 104.47.

This contrast between AdamW (1.2% norm variation) and SGD (97% norm variation) provides the mechanistic foundation for the Phi Invariance Conjecture: AdamW's adaptive scaling is an **implicit weight norm control mechanism** that makes explicit Phi modulation a second-order effect.

![Weight norm trajectories comparing AdamW vs. SGD behavior across methods](figures/fig5_weight_norm_trajectories.png)

<!-- FIGURES
- Figure 2: generate_figures.py, fig2_accuracy_comparison.png -- Main accuracy bar chart
- Figure 3: generate_figures.py, fig3_bem_vs_accuracy.png -- BEM vs. accuracy scatter plot
- Figure 4: generate_figures.py, fig4_diagnostic_heatmap.png -- Diagnostic metrics heatmap
- Figure 5: iter_003 figures, fig5_weight_norm_trajectories.png -- Weight norm trajectories
- Figure (multi-arch): multi_arch_comparison.png -- Multi-architecture accuracy comparison
- Table 2: inline -- AdamW + ResNet-20 main results
- Table 3: inline -- Statistical tests vs. constant baseline
- Table 4: inline -- SGD + ResNet-20 results
- Table 5: inline -- SGD + VGG-16-BN results
- Table 6: inline -- Diagnostic metrics
-->
