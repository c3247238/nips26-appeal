# 4. Experimental Setup

## 4.1 Architectures and Datasets

We evaluate on two architectures spanning a 55$\times$ parameter scale:

- **ResNet-20** ($\sim$270K parameters) on CIFAR-10 and CIFAR-100.
- **VGG-16-BN** ($\sim$15M parameters) on CIFAR-10.

Both architectures use batch normalization (BN). We additionally test ResNet-20-NoBN (BN replaced with identity) on CIFAR-10 to isolate BN's role.

## 4.2 WD Methods

Seven methods spanning four modulation axes:

| Method | $\phi(t, \theta, g)$ | Axis | BEM |
|--------|---------------------|------|-----|
| Constant | $1$ | --- | 0.0 |
| Cosine schedule | $\frac{1}{2}(1 + \cos(\pi t/T))$ | Temporal | $\sim$0.50 |
| CWD (hard) | $\mathbf{1}[\text{sign}(\theta) = \text{sign}(u_t)]$ | Directional | $\sim$0.50 |
| SWD | $\|g\| / \|g\|_{\text{mean}}$ | Temporal | $\sim$0.90 |
| Half-$\lambda$ | $0.5$ | Budget | 0.0 |
| Random mask | $\text{Bernoulli}(0.5)$ | Stochastic | $\sim$0.50 |
| No WD | $0$ | Removal | 1.0 |

## 4.3 Training Configuration

**AdamW:** lr $= 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\lambda = 5 \times 10^{-4}$, cosine LR annealing, 200 epochs, batch size 128. The effective gradient-to-weight ratio $\rho \approx 0.5$.

**SGD (original):** lr $= 0.1$, momentum $= 0.9$, $\lambda = 5 \times 10^{-4}$, cosine annealing, 200 epochs. The effective $\rho \approx 0.005$---100$\times$ lower than AdamW.

**SGD (matched-$\rho$):** lr $= 0.01$, $\lambda = 5 \times 10^{-3}$, momentum $= 0.9$, targeting $\rho \approx 0.5$ to match AdamW and isolate the optimizer-vs-ratio confound.

All configurations use seeds 42, 123, 456 and report mean $\pm$ std over 3 runs.

**Ratio sweep (AdamW):** $\rho_{\text{low}}$ ($\lambda = 5 \times 10^{-5}$, $\rho \approx 0.05$) and $\rho_{\text{high}}$ ($\lambda = 5 \times 10^{-3}$, $\rho \approx 5.0$). The $\rho_{\text{high}}$ regime produced only 5-epoch pilot data (77.69% at epoch 5) before training instability terminated the full run; this data gap is noted explicitly.

## 4.4 Hyperparameter Fairness

All 7 methods share identical base WD ($\lambda$) and learning rate. No per-method tuning is performed. This design choice ensures that accuracy differences reflect intrinsic properties of $\phi$, not hyperparameter optimization. We acknowledge this may disadvantage methods with strong hyperparameter sensitivity (e.g., CWD's $\beta$ parameter).

## 4.5 Diagnostics

Per-epoch tracking: test accuracy, train accuracy, weight norms per layer, gradient norms per layer, $\rho$ per layer, gradient-weight cosine similarity $\hat{\delta}_{l,t}$, CSI, AIS, BEM, and $\phi$ modulation values. This produces $\sim$1,400 scalar measurements per epoch per run.

---

# 5. Results and Analysis

## 5.1 Main Accuracy Comparison (ResNet-20)

Table 1 reports best test accuracy for 7 WD methods across 4 optimizer-dataset configurations on ResNet-20, each averaged over 3 seeds.

**Table 1: ResNet-20 accuracy (mean $\pm$ std, 3 seeds). Bold = best per column.**

| Method | CIFAR-10 AdamW | CIFAR-100 AdamW | CIFAR-10 SGD | CIFAR-100 SGD |
|--------|---------------|-----------------|-------------|---------------|
| Constant | **90.13 $\pm$ 0.31** | 63.15 $\pm$ 0.25 | **91.22 $\pm$ 0.06** | **65.37 $\pm$ 0.13** |
| Cosine | 90.12 $\pm$ 0.07 | **63.42 $\pm$ 0.34** | 91.20 $\pm$ 0.10 | 65.11 $\pm$ 0.25 |
| CWD | 90.06 $\pm$ 0.24 | 62.84 $\pm$ 0.24 | 90.87 $\pm$ 0.35 | 64.37 $\pm$ 0.47 |
| SWD | 89.88 $\pm$ 0.25 | 63.06 $\pm$ 0.24 | 90.71 $\pm$ 0.16 | 64.30 $\pm$ 0.41 |
| Half-$\lambda$ | 90.09 $\pm$ 0.28 | 62.91 $\pm$ 0.38 | 90.84 $\pm$ 0.15 | 64.86 $\pm$ 0.38 |
| Random mask | 90.12 $\pm$ 0.30 | 62.87 $\pm$ 0.31 | 90.77 $\pm$ 0.37 | 64.91 $\pm$ 0.40 |
| No WD | 90.08 $\pm$ 0.31 | 62.66 $\pm$ 0.31 | 90.30 $\pm$ 0.08 | 63.66 $\pm$ 0.17 |
| **$\Phi_{\text{spread}}$** | **0.25** | **0.76** | **0.91** | **1.71** |

Three observations emerge from Table 1:

1. **AdamW Phi invariance.** On CIFAR-10, the 7 methods span only 0.25 percentage points (90.13% to 89.88%). Even the most extreme ablation---complete WD removal ($\phi = 0$)---costs only 0.05% relative to constant WD. On CIFAR-100, the spread widens to 0.76%, with cosine schedule marginally leading (63.42%) and no-WD trailing (62.66%).

2. **SGD shows 3.7$\times$ larger spread.** On CIFAR-10, SGD's Phi spread is 0.91% vs. AdamW's 0.25%. On CIFAR-100, the ratio is 2.3$\times$ (1.71% vs. 0.76%). This difference is partially explained by SGD's 100$\times$ lower $\rho$: SGD operates in a regime where WD constitutes a larger fraction of the total parameter update, making $\phi$ modulation more consequential.

3. **No WD performs worst under SGD.** Without AdamW's implicit $\ell_\infty$-norm constraint (Xie & Li, 2024), removing WD entirely under SGD produces a 0.91% accuracy drop on CIFAR-10 and 1.71% on CIFAR-100, consistent with WD's role as a dynamics modifier (D'Angelo et al., 2024).

**Statistical tests.** Table 2 reports paired t-tests (Bonferroni-corrected) for each method vs. constant WD.

**Table 2: Statistical significance vs. constant WD baseline (ResNet-20, AdamW, CIFAR-10).**

| Method | $\Delta$acc | Raw $p$ | Bonferroni $p$ | Cohen's $d$ |
|--------|-----------|---------|----------------|-------------|
| Cosine | $-$0.01 | 0.94 | 1.00 | 0.03 |
| CWD | $-$0.07 | 0.72 | 1.00 | 0.24 |
| SWD | $-$0.25 | 0.35 | 1.00 | 0.88 |
| Half-$\lambda$ | $-$0.04 | 0.85 | 1.00 | 0.13 |
| Random mask | $-$0.01 | 0.97 | 1.00 | 0.02 |
| No WD | $-$0.05 | 0.84 | 1.00 | 0.16 |

No method achieves $p < 0.05$ after Bonferroni correction. All Cohen's $d$ values are below 1.0, with most below 0.25 (small effect). SWD shows the largest effect size ($d = 0.88$), driven entirely by its 0.25% accuracy deficit, which falls within inter-seed variance.

## 5.2 Multi-Architecture Validation (VGG-16-BN)

VGG-16-BN (15M parameters, 55$\times$ ResNet-20's scale) provides cross-architecture validation. Table 3 reports results on CIFAR-10.

**Table 3: VGG-16-BN CIFAR-10 accuracy (mean $\pm$ std, 3 seeds). Bold = best.**

| Method | Accuracy |
|--------|----------|
| Constant | 92.05 $\pm$ 0.06 |
| Cosine | 91.99 $\pm$ 0.32 |
| CWD | 92.06 $\pm$ 0.26 |
| SWD | 92.11 $\pm$ 0.28 |
| **Half-$\lambda$** | **92.15 $\pm$ 0.13** |
| Random mask | 92.05 $\pm$ 0.27 |
| No WD | 92.03 $\pm$ 0.04 |
| **$\Phi_{\text{spread}}$** | **0.16** |

The VGG-16-BN Phi spread is 0.16%---smaller than ResNet-20's 0.25%. Constant WD (92.05%) is within 0.10% of the best method (half-$\lambda$, 92.15%). The cross-architecture null result confirms that Phi invariance under AdamW at standard $\rho$ is not specific to a single architecture or parameter count.

As shown in Figure 4, the accuracy distributions of all 7 methods overlap substantially on both ResNet-20 and VGG-16-BN, with Phi spread below 0.25% in both cases.

![Multi-architecture accuracy comparison showing 7 WD methods on ResNet-20 and VGG-16-BN, with error bars indicating standard deviation across 3 seeds](figures/multi_arch_comparison.pdf)

## 5.3 SGD vs. AdamW Effect Size

Figure 5 summarizes the Phi spread across all 4 optimizer-dataset configurations, annotated with the corresponding $\rho$ values.

![Phi spread comparison across AdamW and SGD on CIFAR-10 and CIFAR-100, showing SGD's 3.7x larger spread partially explained by its 100x lower rho](figures/sgd_vs_adamw_spread.pdf)

SGD's larger Phi spread (0.91% vs. 0.25% on CIFAR-10; 1.71% vs. 0.76% on CIFAR-100) admits two non-exclusive explanations: (i) SGD lacks AdamW's per-parameter adaptive scaling, which subsumes $\phi$ modulation effects; (ii) SGD's 100$\times$ lower $\rho$ places it in a regime where WD constitutes a proportionally larger fraction of the update, amplifying $\phi$ sensitivity. The matched-$\rho$ SGD experiment (Section 5.5) attempts to disentangle these factors.

## 5.4 NoBN vs. BN Ablation

Removing batch normalization from ResNet-20 produces a 2.4 percentage-point accuracy drop (Table 4), confirming BN's importance for training quality. AIS increases from $\sim$0.35 (BN) to $\sim$0.50 (NoBN), consistent with the theory: without BN's scale-invariance, gradient-weight alignment becomes more informative.

**Table 4: ResNet-20-NoBN vs. ResNet-20 (BN) on CIFAR-10 (AdamW).**

| Architecture | Constant acc | CWD acc | $\Delta$(Constant$-$CWD) | AIS |
|-------------|-------------|---------|------------------------|-----|
| ResNet-20 (BN) | 90.13 $\pm$ 0.31 | 90.06 $\pm$ 0.24 | +0.07 | 0.35 |
| ResNet-20-NoBN | 87.74 $\pm$ 0.20 | 87.64 $\pm$ 0.17 | +0.10 | 0.50 |
| Cohen's $d$ (BN vs NoBN, constant) | | | 9.14 (large) | |

The NoBN Phi spread cannot yet be computed conclusively (only 2 of 7 methods completed with 3 seeds). The available data shows constant (87.74%) slightly outperforming CWD (87.64%), consistent with the BN configuration. NoBN's higher AIS ($\sim$0.50 vs. $\sim$0.35) is a necessary but not sufficient condition for adaptive WD to help; the full threshold from Theorem 1 also depends on $\Delta$CSI and $\sigma^2/n$.

## 5.5 Matched-Ratio SGD (Preliminary)

Matched-$\rho$ SGD (lr $= 0.01$, $\lambda = 5 \times 10^{-3}$, targeting $\rho \approx 0.5$) provides partial data. Table 5 reports available results.

**Table 5: Matched-$\rho$ SGD on CIFAR-10, ResNet-20 (preliminary).**

| Method | Seed 42 | Seed 123 | Seed 456 | Mean |
|--------|---------|----------|----------|------|
| Constant | 76.12$^\dagger$ | 90.94 | 90.89 | 90.92$^*$ |
| CWD | 90.81 | --- | --- | 90.81 |

$^\dagger$Seed 42 constant ran only 5 epochs (training instability at high $\lambda = 5 \times 10^{-3}$); excluded from mean. $^*$Mean over seeds 123, 456 only.

The available data (constant: 90.92% from 2 seeds; CWD: 90.81% from 1 seed) is insufficient for statistical conclusions. The seed 42 anomaly (76.12% from 5 epochs) indicates that high-$\lambda$ SGD is sensitive to initialization, a confound that must be addressed before this ablation can be considered definitive. We report this as preliminary evidence and note the data gap transparently.

## 5.6 Diagnostic Analysis

**BEM vs. accuracy.** BEM ranges from 0.0 (constant, half-$\lambda$) to $\sim$0.90 (SWD), yet accuracy varies by $< 0.25\%$ on CIFAR-10 under AdamW. A 10$\times$ variation in effective WD budget produces negligible accuracy change, demonstrating that AdamW's dynamics are robust to the absolute WD magnitude at standard $\rho$.

**CSI vs. accuracy.** No correlation (Spearman $\rho \approx 0$) between CSI and accuracy across methods. CSI characterizes coupling instability, not performance; methods with high CSI (SWD, random mask) achieve comparable accuracy to low-CSI methods (constant).

**AIS.** AIS values cluster in [0.18, 0.59] across all configurations. AIS is an intrinsic network-dataset property: it does not vary with WD method. BN networks show AIS $\sim$ 0.18--0.40; NoBN networks show AIS $\sim$ 0.31--0.59. The higher NoBN AIS aligns with the Theorem 1 prediction that removing BN increases the alignment benefit, potentially shifting the threshold toward the adaptive-WD-optimal regime.

![Diagnostic metric panel: (a) CSI vs accuracy, (b) AIS vs accuracy, (c) BEM vs accuracy, (d) weight norm trajectories, showing no predictive relationship between diagnostic metrics and test accuracy](figures/diagnostic_panel.pdf)

<!-- FIGURES
- Figure 4: gen_multi_arch_comparison.py, multi_arch_comparison.pdf — Multi-architecture accuracy comparison (ResNet-20 and VGG-16-BN)
- Figure 5: gen_sgd_vs_adamw_spread.py, sgd_vs_adamw_spread.pdf — Phi spread comparison across optimizers
- Figure 7: gen_diagnostic_panel.py, diagnostic_panel.pdf — Diagnostic metric panel (CSI, AIS, BEM vs accuracy)
- Table 1: inline — Main accuracy results (7 methods x 4 configs)
- Table 2: inline — Statistical significance tests
- Table 3: inline — VGG-16-BN results
- Table 4: inline — NoBN vs BN ablation
- Table 5: inline — Matched-rho SGD preliminary results
-->
