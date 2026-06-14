# 1. Introduction

Weight decay (WD) is among the most ubiquitous hyperparameters in deep learning, yet a paradox has emerged in recent practice: multiple dynamic WD methods---CWD (Chen et al., ICLR 2026), SWD (Xie et al., NeurIPS 2023), AlphaDecay (He et al., NeurIPS 2025)---report consistent improvements over constant WD, while practitioners routinely achieve competitive or identical results with a fixed decay coefficient under AdamW (Loshchilov & Hutter, ICLR 2019). On ResNet-20/CIFAR-10 under AdamW, seven WD strategies spanning binary masking (CWD), gradient-norm scaling (SWD), cosine scheduling, and complete removal of WD all produce accuracies within a 0.25 percentage-point band (90.13% $\pm$ 0.31 for constant vs. 89.88% $\pm$ 0.25 for SWD; all $p > 0.05$ after Bonferroni correction). This raises a concrete question: **when does adaptive weight decay actually help, and when is it unnecessary?**

We answer this question through the lens of stability-optimal control theory. The core insight is that any WD modulation function $\phi(t, \theta, g)$ produces two competing effects: an *alignment benefit* from exploiting informative gradient-weight geometry, and a *stability cost* from perturbing the optimizer's coupling dynamics. The net benefit is governed by the gradient-to-weight ratio $\rho = \|g\| / \|\theta\|$, which determines the operating regime of the optimization trajectory.

## 1.1 Research Gap

Despite the growing landscape of dynamic WD methods---spanning alignment-aware (CWD, AdamO), temporally scheduled (SWD, ADANA), and norm-targeted (AdamWN, AlphaDecay) approaches---four gaps remain open:

1. **No theory explaining constant WD's competitiveness.** D'Angelo et al. (NeurIPS 2024) establish WD as a dynamics modifier rather than a classical regularizer, but do not analyze when modulation strategies outperform the constant baseline.

2. **No unified mathematical treatment.** CWD's binary sign mask, SWD's gradient-norm scaling, cosine schedules, and norm-matched WD lack a common formulation revealing their mathematical connections.

3. **No controlled ratio-regime experiments.** Prior evaluations use inconsistent benchmarks and do not systematically vary the gradient-to-weight ratio---the quantity our theory identifies as the regime boundary.

4. **No optimal WD law from first principles.** Existing dynamic WD methods are heuristic; none derives the WD schedule from an optimality condition on the training dynamics.

## 1.2 Contributions

This paper makes five contributions:

1. **Theorem 1 (Binary Masking Suboptimality).** CWD outperforms constant WD if and only if the Alignment Informativeness Score (AIS) exceeds a noise-dependent threshold: $\text{AIS} > (C\sigma^2/n) \cdot \Delta\text{CSI} / \bar{\lambda}$. At standard $\rho$ in batch-normalized networks, this condition is not met, predicting constant WD's dominance. All 7 out of 7 empirical predictions across optimizer-architecture-dataset combinations are confirmed.

2. **Theorem 2 (Layer-wise CSI Bound).** The generalization gap penalty from per-parameter WD variation is bounded by $2L\sigma^2/n \cdot \text{CSI}_\text{param} \cdot T$. Methods with $\lambda_{\min} = 0$ (CWD, random mask) incur unbounded per-parameter Coupling Stability Index (CSI) during off-steps, explaining why stochastic masking hurts despite moderate aggregate CSI.

3. **Theorem 3 (PMP-Optimal WD) with dual derivation.** We derive the optimal state-feedback WD law $\lambda^*(t) = \text{clip}(\kappa \cdot (\rho^* - \hat{\rho}_t)^+, 0, \lambda_{\max})$ from the stochastic Pontryagin Maximum Principle (PMP), and independently recover the same functional form from renormalization group (RG) beta function analysis. PMP-WD is the first closed-loop WD controller; existing methods are open-loop (cosine schedule) or binary (CWD).

4. **Proposition 1 (Alignment Noise Constraint).** For batch size $b \leq 256$, the coefficient of variation $\text{CV}(\hat{\delta}_t) \gg 1$ for the gradient-weight cosine similarity. Any alignment-aware WD method must therefore use EMA-smoothed signals with aggregation horizon $k \geq 10$ steps. PMP-WD satisfies this by construction through its EMA of $\hat{\rho}_t$.

5. **Systematic experiments.** 150+ runs across 2 architectures (ResNet-20, VGG-16-BN), 2 datasets (CIFAR-10, CIFAR-100), 2 optimizers (AdamW, SGD), and gradient-to-weight ratio regimes from $\rho \approx 0.005$ (SGD) to $\rho \approx 0.5$ (AdamW) confirm the theory: constant WD is optimal at standard ratios; method sensitivity scales with $\rho$, with SGD's 3.7$\times$ larger Phi spread (0.91% vs. 0.25% on CIFAR-10) partially explained by its 100$\times$ lower $\rho$.

## 1.3 Paper Organization

Section 2 reviews related work. Section 3 develops the theoretical framework: the Phi modulator abstraction, diagnostic metrics (BEM, CSI, AIS), and Theorems 1--3 with Proposition 1. Section 4 describes experimental setup. Section 5 presents results across architectures, datasets, optimizers, and ratio regimes. Section 6 discusses implications and limitations. Section 7 concludes.

As shown in Figure 1, method spread increases monotonically with the gradient-to-weight ratio $\rho$, with a phase transition from the "inhibition" regime ($\rho < 0.1$, spread $< 0.1\%$) through "transition" ($0.1 < \rho < 2.0$) to "differentiation" ($\rho > 2.0$, spread $> 0.5\%$).

![Ratio regime diagram showing method spread vs. gradient-to-weight ratio, with three regime zones and data from AdamW and SGD experiments](figures/ratio_regime.pdf)

<!-- FIGURES
- Figure 1: gen_ratio_regime.py, ratio_regime.pdf — Ratio regime diagram: method spread vs log(rho) with three shaded zones
-->
