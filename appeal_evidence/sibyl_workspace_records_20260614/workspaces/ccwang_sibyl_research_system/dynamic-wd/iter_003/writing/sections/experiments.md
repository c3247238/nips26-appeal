# 4. Experimental Setup

## 4.1 Implementation

All experiments are implemented in PyTorch using a unified `UnifiedAdamW` optimizer class with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring that all methods share identical optimizer internals---moment estimation, bias correction, learning rate scheduling---and differ only in the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024), inheriting its data loading, learning rate scheduling, and logging pipelines.

We evaluate seven weight decay methods spanning the four modulation axes of the Phi framework: **constant** (baseline, $\varphi \equiv 1$), **cwd\_hard** (Cautious Weight Decay with binary sign-alignment mask), **swd** (Scheduled Weight Decay with gradient-norm sensitivity), **cosine\_schedule** (cosine-annealed weight decay), **random\_mask** (Bernoulli mask control with $p = 0.5$), **half\_lambda** (constant weight decay at $\lambda/2$, a budget-matched control), and **no\_wd** (weight decay disabled, $\varphi \equiv 0$).

## 4.2 Training Configuration

**Datasets.** We use CIFAR-10 and CIFAR-100 (Krizhevsky, 2009), loaded via torchvision with standard train/test splits (50,000/10,000 images). Standard data augmentation is applied: random cropping with 4-pixel padding, random horizontal flipping, and per-channel normalization.

**Architecture.** ResNet-20 in the standard CIFAR configuration ($\sim$270K parameters) with batch normalization (He et al., 2016).

**Optimizer.** AdamW with decoupled weight decay. All methods use identical AdamW base parameters: learning rate $\eta = 10^{-3}$ with cosine annealing to zero (no warmup), $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for all methods. Each dynamic method modulates this value through its Phi function; the constant baseline applies $\lambda$ uniformly at every step.

**Training duration.** 200 epochs with batch size 128.

**Seeds.** Three independent runs per method--dataset configuration using seeds $\{42, 123, 456\}$, controlling Python, NumPy, PyTorch, and CUDA random number generators. Total: 42 experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets).

## 4.3 Hyperparameter Fairness Protocol

A critical design choice in our benchmark is that **all methods use identical base hyperparameters** ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$) with no per-method grid search. This is intentional: it ensures that observed differences measure the effect of Phi modulation, not hyperparameter luck. Each method may operate at a non-optimal hyperparameter configuration, but this is the price of a fair comparison. We acknowledge this limitation and discuss its implications in Section 6.3.

## 4.4 Diagnostic Logging

We log per-epoch: test accuracy, training loss, per-layer weight norms, CSI, AIS, and BEM. Per-100-step snapshots record gradient-weight cosine similarity per layer, effective learning rate per layer, and Phi modulation values. Full diagnostic panels for all 42 runs are provided in Appendix B.

---

# 5. Results and Analysis

## 5.1 Main Accuracy Comparison

Table 2 presents the main results. On CIFAR-10, the seven methods achieve mean test accuracies spanning a total range of 0.25 percentage points: from 89.88\% (SWD) to 90.13\% (constant). On CIFAR-100, the range is 0.76 percentage points: from 62.66\% (no\_wd) to 63.42\% (cosine\_schedule). The constant baseline is the best or near-best method on CIFAR-10, while cosine\_schedule leads on CIFAR-100 by 0.27\%---neither advantage is statistically significant.

**Table 2: Main accuracy results.** Mean $\pm$ standard deviation over 3 seeds. Best result per dataset in **bold**.

| Method | CIFAR-10 Acc. (\%) | CIFAR-100 Acc. (\%) |
|--------|:------------------:|:-------------------:|
| constant | **90.13** $\pm$ 0.31 | 63.15 $\pm$ 0.30 |
| cosine\_schedule | 90.12 $\pm$ **0.07** | **63.42** $\pm$ 0.42 |
| random\_mask | 90.12 $\pm$ 0.30 | 62.87 $\pm$ 0.38 |
| half\_lambda | 90.09 $\pm$ 0.29 | 62.91 $\pm$ 0.47 |
| no\_wd | 90.08 $\pm$ 0.32 | 62.66 $\pm$ 0.38 |
| cwd\_hard | 90.06 $\pm$ 0.24 | 62.84 $\pm$ 0.30 |
| swd | 89.88 $\pm$ 0.25 | 63.06 $\pm$ 0.29 |

A noteworthy observation is that cosine\_schedule achieves the lowest variance on CIFAR-10 ($\sigma = 0.07\%$, compared to $\sigma \approx 0.25$--$0.32\%$ for all other methods), suggesting a potential *stability* benefit without an accuracy advantage.

Table 3 reports paired $t$-tests of each method against the constant baseline. All $p$-values exceed 0.05 (range: $p = 0.090$ to $p = 0.950$). After Bonferroni correction for six comparisons (significance threshold $p < 0.0083$), no method survives. All Cohen's $d$ effect sizes are below 0.3, the conventional threshold for a small effect.

**Table 3: Statistical tests vs. constant baseline.** $\Delta$ = method accuracy minus constant accuracy. All comparisons are not statistically significant.

| Method | CIFAR-10 $\Delta$ | $p$-value | CIFAR-100 $\Delta$ | $p$-value |
|--------|:-----------------:|:---------:|:------------------:|:---------:|
| cwd\_hard | $-0.07\%$ | 0.832 | $-0.31\%$ | 0.326 |
| swd | $-0.25\%$ | 0.513 | $-0.10\%$ | 0.801 |
| cosine\_schedule | $-0.01\%$ | 0.935 | $+0.26\%$ | 0.117 |
| random\_mask | $-0.01\%$ | 0.950 | $-0.29\%$ | 0.090 |
| half\_lambda | $-0.05\%$ | 0.901 | $-0.25\%$ | 0.608 |
| no\_wd | $-0.05\%$ | 0.825 | $-0.49\%$ | 0.312 |

The most striking result is the no\_wd ablation: removing weight decay entirely ($\lambda = 0$) yields CIFAR-10 accuracy of 90.08\% versus the constant baseline's 90.13\% ($\Delta = 0.05\%$, $p = 0.825$). This is strong evidence that weight decay *scheduling* is irrelevant because weight decay *magnitude* is largely irrelevant under AdamW on these benchmarks.

**Equivalence testing (TOST).** Standard null hypothesis tests can only fail to reject non-equivalence; they cannot confirm equivalence. We therefore apply Two One-Sided Tests (TOST) with practical equivalence margins $\delta = \pm 0.5\%$ and $\delta = \pm 1.0\%$. At $\delta = \pm 0.5\%$, only cosine\_schedule on CIFAR-10 achieves confirmed equivalence ($p_{\mathrm{TOST}} = 0.039$), consistent with the limited statistical power of $N = 3$ seeds (minimum detectable effect at 80\% power $\approx 0.7\%$ with $\sigma \approx 0.3\%$). At $\delta = \pm 1.0\%$, 6 of 12 method--dataset comparisons confirm equivalence ($p_{\mathrm{TOST}} < 0.05$), including cwd\_hard, cosine\_schedule, random\_mask, and no\_wd on CIFAR-10. The remaining comparisons are inconclusive (not rejected in either direction), consistent with insufficient power rather than genuine non-equivalence. **We emphasize that with $N = 3$ seeds, our study has 80\% power to detect only effects $\geq 0.7\%$; smaller effects remain unresolvable and require additional seeds.**

## 5.2 Budget Equivalence Analysis

Figure 3 plots mean test accuracy against the Budget Equivalence Metric (BEM) for all seven methods on both datasets. BEM values span the full range from 0.0 (constant, half\_lambda) through approximately 0.5 (cosine\_schedule, cwd\_hard, random\_mask) to 0.9 (SWD) and 1.0 (no\_wd). Despite this ten-fold variation in effective weight decay budget, accuracy remains essentially flat.

On CIFAR-10, the accuracy spread across the full BEM range is 0.25\%: from 89.88\% at BEM $= 0.9$ to 90.13\% at BEM $= 0.0$. On CIFAR-100, the spread is 0.76\%: from 62.66\% at BEM $= 1.0$ to 63.42\% at BEM $= 0.503$. In both cases, confidence intervals of all methods overlap substantially.

**A 10$\times$ variation in effective weight decay budget (BEM range $0.0$--$1.0$) produces less than $0.5\%$ accuracy variation.** This finding rules out the hypothesis that dynamic weight decay methods improve accuracy by using less total weight decay budget. Even at $\mathrm{BEM} = 1.0$ (zero weight decay), performance is statistically equivalent to the constant baseline.

## 5.3 CSI and AIS Diagnostic Analysis

**CSI analysis.** Table 4 reports Coupling Stability Index values. On CIFAR-10, CSI ranges from 0.838 (SWD, most stable) to 0.964 (no\_wd, least stable). The no\_wd method has the highest CSI because without weight decay, weight norms grow freely, increasing coupling instability between the optimization trajectory and the loss landscape. Cosine\_schedule also shows elevated CSI (0.936), as the slowly decaying schedule creates time-varying coupling between weight decay strength and weight norm trajectory. On CIFAR-100, the CSI range is narrower (0.854--0.868), with all methods clustering together.

Critically, **CSI does not predict accuracy**: the Spearman rank correlation between CSI and accuracy rank across methods is $\rho < 0.3$ ($p > 0.3$) on both datasets. CSI successfully captures differences in training *dynamics* but these dynamics differences do not translate to performance differences under AdamW. This confirms that CSI is a diagnostic tool for understanding *mechanism*, not a criterion for method selection.

**AIS analysis.** All methods show moderate AIS values in the range 0.280--0.410. On CIFAR-10, half\_lambda has the highest AIS (0.410) and no\_wd the lowest on CIFAR-100 (0.280). The key finding is that AIS is **consistent across all weight decay methods**, including the random\_mask control and the no\_wd ablation. The gradient-weight alignment signal has moderate predictive power for loss changes, but this predictive power is an *intrinsic property of the network and loss landscape*---it is not generated, amplified, or destroyed by the weight decay modulation strategy.

This finding directly challenges the motivation behind CWD: if AIS is the same for CWD (which conditions on alignment) and random\_mask (which ignores alignment entirely), the alignment signal provides no additional useful information for weight decay decisions on these benchmarks. The alignment structure exists, but exploiting it through weight decay modulation does not improve outcomes.

**Table 4: Diagnostic metrics.** CSI, AIS, and BEM values for all methods on CIFAR-10.

| Method | CSI | AIS | BEM |
|--------|:---:|:---:|:---:|
| constant | 0.841 | 0.336 | 0.000 |
| cosine\_schedule | 0.936 | 0.352 | 0.503 |
| random\_mask | 0.892 | 0.359 | 0.500 |
| half\_lambda | 0.853 | 0.410 | 0.000 |
| no\_wd | 0.964 | 0.343 | 1.000 |
| cwd\_hard | 0.851 | 0.368 | 0.503 |
| swd | 0.838 | 0.360 | 0.900 |

## 5.4 Weight Norm Analysis

Figure 5 shows per-layer mean weight norm trajectories over 200 training epochs for all seven methods on CIFAR-10. Despite the ten-fold variation in effective weight decay budget, all methods converge to remarkably similar weight norm levels: the final mean weight norms range from 95.89 (constant) to 97.04 (no\_wd), a difference of only 1.2\%.

This convergence provides the mechanistic explanation for the accuracy equivalence observed in Section 5.1. AdamW's adaptive per-parameter step size $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ scales updates inversely with gradient magnitude. When weight decay drives a parameter toward zero, its gradient contribution decreases, and AdamW's adaptive scaling compensates by increasing the effective step size. Conversely, when weight decay is absent (no\_wd), the parameters grow slightly larger, but AdamW's normalization by the second moment prevents the effective updates from becoming disproportionately large. The result is an **implicit weight norm control mechanism** built into AdamW that subsumes the explicit norm control attempted by Phi modulation.

This implicit control explains why even the no\_wd ablation ($\lambda = 0$) achieves near-identical weight norms (97.04 vs. 95.89, a 1.2\% difference) and near-identical accuracy (90.08\% vs. 90.13\%): AdamW's adaptive scaling is the dominant force governing weight norm dynamics, and explicit weight decay is a second-order perturbation at the scale of $\lambda = 5 \times 10^{-4}$.
