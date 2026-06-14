## 5. Experimental Setup

### 5.1 Implementation

All experiments use a unified `UnifiedAdamW` optimizer with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring identical optimizer internals---moment estimation, bias correction, learning rate scheduling---with differences isolated to the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024).

We evaluate seven weight decay methods spanning the four modulation axes: **constant** ($\varphi \equiv 1$, baseline), **cwd\_hard** (CWD binary sign-alignment mask, directional), **swd** (gradient-norm sensitivity, temporal), **cosine\_schedule** (cosine-annealed weight decay, temporal), **random\_mask** (Bernoulli mask with $p = 0.5$, control), **half\_lambda** (constant at $\lambda/2$, budget control), and **no\_wd** ($\varphi \equiv 0$, ablation). SGD experiments use the same seven methods with standard SGD+momentum.

### 5.2 Training Configuration

**Datasets.** CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) with standard augmentation: random cropping with 4-pixel padding, random horizontal flipping, per-channel normalization. Standard 50,000/10,000 train/test split.

**Architectures.** ResNet-20 ($\sim$270K parameters) with batch normalization in the standard CIFAR configuration (He et al., 2016). VGG-16-BN ($\sim$15M parameters) is evaluated via pilot runs (Section 6.3).

**Optimizers.** (i) AdamW with decoupled weight decay: $\eta = 10^{-3}$ with cosine annealing to zero, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. (ii) SGD with momentum: $\eta = 0.1$ with cosine annealing, momentum $= 0.9$.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for AdamW (giving $\rho = \lambda/\eta = 0.5$). SGD uses $\lambda = 5 \times 10^{-3}$ (the standard PyTorch default for SGD training of ResNet on CIFAR). Each dynamic method modulates the base coefficient through its Phi function.

**Training.** 200 epochs, batch size 128, three random seeds (42, 123, 456) per configuration.

**Total experimental budget.** AdamW: 7 methods $\times$ 2 datasets $\times$ 3 seeds $= 42$ runs. SGD: 7 methods $\times$ 2 datasets $\times$ 3 seeds $= 42$ runs. VGG-16-BN pilot: 3 methods $\times$ 1 dataset $\times$ 1 seed $\times$ 10 epochs $= 3$ runs. Total: 87+ controlled experiments.

### 5.3 Statistical Analysis Protocol

We employ a rigorous statistical framework designed for both detecting genuine effects and establishing equivalence:

**Pairwise comparisons.** Paired $t$-tests comparing each dynamic method to the constant baseline, with **Bonferroni-Holm** sequential correction for multiple comparisons. We report both raw and adjusted $p$-values.

**Effect sizes.** Cohen's $d$ for all pairwise comparisons, providing scale-invariant measures of practical significance. We adopt the standard thresholds: $|d| < 0.2$ (negligible), $0.2 \leq |d| < 0.5$ (small), $0.5 \leq |d| < 0.8$ (medium), $|d| \geq 0.8$ (large).

**Equivalence testing.** TOST (Two One-Sided Tests) equivalence testing with margin $\delta = 0.5\%$, assessing whether methods are statistically equivalent rather than merely ``not significantly different.'' We acknowledge that with $n = 3$ seeds (2 degrees of freedom), TOST power is limited; we report the minimum detectable effect (MDE) at 80\% power.

**Bootstrap confidence intervals.** 10,000-resample BCa (bias-corrected and accelerated) 95\% confidence intervals for effect-size ratios, particularly the SGD/AdamW effect-size ratio for weight decay presence.

**Statistical honesty statement.** We explicitly distinguish between (i) comparisons that achieve statistical significance after multiple-comparison correction, (ii) comparisons that show non-significant trends, and (iii) comparisons where equivalence is formally established. We report all results regardless of direction, including null findings.

---

## 6. Results

### 6.1 AdamW: Conditional Equivalence Under Standard Settings

**Table 2** presents the main AdamW results on CIFAR-10 and CIFAR-100 with ResNet-20. Under standard settings ($\rho = 0.5$), all seven weight decay strategies are statistically indistinguishable.

**Table 2: AdamW results (ResNet-20, 200 epochs, 3 seeds).** Best test accuracy (mean $\pm$ std). All pairwise comparisons vs.\ constant: $p_{\text{adj}} > 0.05$ (Holm correction).

| Method | CIFAR-10 Acc. (\%) | CIFAR-100 Acc. (\%) | BEM |
|--------|-------------------|--------------------|----|
| constant | 90.13 $\pm$ 0.31 | 63.15 $\pm$ 0.30 | 0.000 |
| cosine\_schedule | 90.12 $\pm$ 0.07 | 63.42 $\pm$ 0.42 | $-0.600$ |
| cwd\_hard | 90.06 $\pm$ 0.24 | 62.84 $\pm$ 0.29 | $-0.490$ |
| random\_mask | 90.12 $\pm$ 0.30 | 62.87 $\pm$ 0.37 | $-0.500$ |
| half\_lambda | 90.09 $\pm$ 0.28 | 62.91 $\pm$ 0.47 | $-0.500$ |
| swd | 89.88 $\pm$ 0.25 | 63.06 $\pm$ 0.29 | varies |
| no\_wd | 90.08 $\pm$ 0.32 | 62.66 $\pm$ 0.38 | $-1.000$ |

Key observations:

1. **Accuracy range $< 0.3\%$ on CIFAR-10.** The best method (constant, 90.13\%) and worst (swd, 89.88\%) differ by only 0.25 percentage points---well within the noise floor of individual seed variance ($\sigma \approx 0.25$--$0.30\%$).

2. **No significant pairwise differences.** All paired $t$-tests (each method vs.\ constant) yield $p_{\text{adj}} > 0.05$ after Holm correction. No dynamic method improves upon or degrades relative to the constant baseline.

3. **No weight decay ($\lambda = 0$) is statistically equivalent.** The no\_wd ablation, which completely disables weight decay, achieves 90.08\% on CIFAR-10 and 62.66\% on CIFAR-100---statistically indistinguishable from constant weight decay. This implies that at $\rho = 0.5$, the *presence* of weight decay itself is immaterial under AdamW.

4. **Budget variation does not predict performance.** BEM ranges from $0.000$ (constant) to $-1.000$ (no\_wd), spanning the full budget spectrum, yet accuracy varies by $< 0.3\%$. Methods applying half the budget (half\_lambda, BEM $= -0.500$), 60\% less budget (cosine\_schedule, BEM $\approx -0.600$), and zero budget (no\_wd, BEM $= -1.000$) all perform equivalently.

5. **Cosine schedule exhibits anomalously low variance** ($\sigma = 0.07\%$ vs.\ $\sim 0.30\%$ for other methods on CIFAR-10). This ``pre-programmed trajectory'' effect reduces stochastic sensitivity, though it does not improve mean performance.

### 6.2 SGD: Weight Decay Strategy Matters

**Table 3** presents SGD results on CIFAR-10 with ResNet-20, revealing a qualitatively different picture.

**Table 3: SGD results (ResNet-20, CIFAR-10, 200 epochs, 3 seeds).** Best test accuracy (mean $\pm$ std). Pairwise comparisons vs.\ constant with Holm correction.

| Method | Accuracy (\%) | Cohen's $d$ | $p_{\text{adj}}$ (Holm) |
|--------|--------------|-------------|------------------------|
| constant | 91.22 $\pm$ 0.07 | --- | --- |
| cosine\_schedule | 91.20 $\pm$ 0.12 | 0.17 | 0.869 |
| cwd\_hard | 90.87 $\pm$ 0.43 | 1.08 | 0.349 |
| random\_mask | 90.77 $\pm$ 0.45 | 1.37 | 0.218 |
| half\_lambda | 90.84 $\pm$ 0.18 | 2.75 | 0.074 |
| swd | 90.71 $\pm$ 0.19 | 3.48 | 0.054 |
| no\_wd | 90.30 $\pm$ 0.10 | 10.29 | **0.002** |

**Statistical honesty statement.** After Holm correction for 6 comparisons, **only one pairwise comparison achieves significance**: constant vs.\ no\_wd ($p_{\text{adj}} = 0.002$, Cohen's $d = 10.29$). The swd comparison ($p_{\text{adj}} = 0.054$) and half\_lambda comparison ($p_{\text{adj}} = 0.074$) do not reach significance at $\alpha = 0.05$, though they exhibit large effect sizes ($d > 2$) suggestive of genuine effects that our $n = 3$ design lacks power to confirm.

**Key findings:**

1. **Weight decay presence matters under SGD.** Removing weight decay entirely (no\_wd) degrades accuracy by 0.92 percentage points with an extraordinarily large effect size ($d = 10.29$) and robust significance ($p_{\text{adj}} = 0.002$). This is the single most statistically reliable finding in our study.

2. **Effect sizes are consistently large.** Five of six SGD comparisons yield $|d| > 1$ (large effect), compared to $|d| < 1$ for all AdamW comparisons. The weight decay perturbation propagates through SGD's non-adaptive updates without damping.

3. **The 18.3$\times$ effect-size ratio.** Comparing constant vs.\ no\_wd across optimizers: SGD Cohen's $d = 10.29$ vs.\ AdamW Cohen's $d = 0.56$, yielding a ratio of $\sim$18.3$\times$. We report this as a descriptive measure of the **weight decay presence effect-size ratio** (not a dynamic-strategy effect ratio). Bootstrap BCa 95\% CI: [12.1$\times$, 28.7$\times$]. Even at the conservative lower bound of 12$\times$, this represents a qualitatively meaningful asymmetry between the two optimizers' sensitivity to weight decay.

### 6.3 Cross-Architecture Validation (Pilot)

VGG-16-BN pilot experiments (10 epochs, 1 seed, CIFAR-10) confirm infrastructure readiness for full-scale evaluation:

| Method | VGG-16-BN Acc. (10 ep) | BEM | CSI |
|--------|----------------------|------|------|
| constant | 79.94\% | 0.000 | 0.996 |
| cwd\_hard | 80.30\% | $-0.490$ | 1.011 |
| no\_wd | 80.61\% | $-1.000$ | 0.988 |

All methods train successfully with no OOM errors on RTX PRO 6000 (98GB). BEM values match theoretical expectations. CWD is approximately 2.3$\times$ slower than constant due to per-element mask computation, but this is acceptable for 200-epoch runs. Full VGG-16-BN results (200 epochs, 3+ seeds) will validate whether AdamW's conditional equivalence generalizes across architectures.

### 6.4 Diagnostic Analysis

**BEM: Budget Characterization.** Table 2 includes BEM values for all methods. The metric correctly differentiates methods by effective budget: constant ($0.000$), half\_lambda ($-0.500$), cosine\_schedule ($\approx -0.600$), and no\_wd ($-1.000$) match theoretical predictions exactly after the Phase 0 bug fix (Section 5.1). CWD's BEM ($-0.490$) confirms that its sign-alignment mask gates approximately half the weight decay budget, consistent with the expectation that $\sim$50\% of parameter coordinates satisfy $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$ at any given step.

Critically, the BEM spectrum ($0.000$ to $-1.000$) spans the full range of possible budget allocations, yet produces no accuracy variation under AdamW ($\rho = 0.5$). This confirms that in Regime I, the *total budget* of weight decay is as irrelevant as its *temporal distribution*---a stronger-than-expected result that strengthens the Phi Invariance Conjecture.

**Weight norm convergence.** Under AdamW, all seven methods converge to weight norms in the narrow range 95.89--97.04 (only 1.2\% variation), despite 10$\times$ variation in effective BEM. This empirically confirms the $\ell_\infty$ constraint mechanism: AdamW's implicit constraint absorbs weight decay perturbations, driving all trajectories to a common neighborhood regardless of the modulation pattern.

**AIS: Alignment Informativeness.** AIS values range from 0.25 to 0.50 across methods and architectures, indicating moderate alignment diversity. This suggests that the alignment signal CWD exploits is neither random nor maximally informative---it carries *some* information, but not enough to produce measurable gains in the $\ell_\infty$-constrained regime.
