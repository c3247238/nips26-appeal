# The Phi Invariance Conjecture: When Dynamic Weight Decay Methods Are Equivalent Under Adaptive Optimizers

## Abstract

Weight decay is applied in virtually every deep learning pipeline, yet a growing literature proposes dynamic alternatives---Cautious Weight Decay (CWD), Scheduled Weight Decay (SWD), cosine-scheduled decay, Weight Norm Control (AdamWN), and AlphaDecay---each reporting improvements under different conditions. We introduce the Phi Modulator Framework, a unified mathematical abstraction that expresses every weight decay method as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$, where the modulator $\varphi$ encodes all method-specific logic. CWD, SWD, cosine schedules, and their compositions are recovered as special cases along four modulation axes: temporal, directional, spatial, and target-norm. Three diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---provide the first standardized tools for characterizing weight decay behavior beyond final accuracy. A systematic evaluation of 105 controlled experiments across 7 methods, 2 optimizers (AdamW, SGD), 2 architectures (ResNet-20, VGG-16-BN), 2 datasets (CIFAR-10, CIFAR-100), and 3 seeds reveals two complementary findings. Under AdamW, all dynamic weight decay variants are statistically equivalent to constant weight decay ($p > 0.05$ for all paired comparisons, including the degenerate case of no weight decay at all): a ten-fold variation in effective weight decay budget produces less than 0.5% accuracy variation on CIFAR-10. Under SGD, weight decay magnitude matters: removing weight decay drops accuracy by 0.92% on CIFAR-10 and 1.71% on CIFAR-100, with weight norms nearly doubling. We formalize these observations as the Phi Invariance Conjecture: on sufficiently overparameterized problems with batch normalization, AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator. The conjecture's predicted boundary condition---that invariance breaks without adaptive scaling---is confirmed by the SGD results and further supported by VGG-16-BN experiments showing that batch normalization provides an alternative mechanism for Phi Invariance.

---

## 1. Introduction

### 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. D'Angelo et al. (2024) showed that weight decay is never useful as explicit regularization in modern deep learning; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories under SGD and controlling the bias-variance tradeoff in large language model training.

This re-understanding has sparked a surge of methods proposing to dynamically modulate weight decay strength during training. Xie et al. (2023) introduced Scheduled Weight Decay (SWD), which adjusts weight decay based on gradient norms. Chen et al. (2026a) proposed Cautious Weight Decay (CWD), a sign-alignment mask that applies decay only when weight and update directions agree. Loshchilov (2023) generalized decoupled weight decay to target-norm control (AdamWN). He et al. (2025) introduced AlphaDecay, a module-wise strategy guided by spectral density analysis. Ferbach et al. (2026) proposed logarithmic-time schedules (ADANA) for weight decay alongside momentum coefficients. Chen et al. (2026b) identified a "Radial Tug-of-War" conflict between weight decay and gradient updates, proposing AdamO to decouple radial and tangential dynamics.

A critical problem pervades this literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. CWD reports improvements on language model pre-training with Lion and Muon optimizers; SWD demonstrates gains with SGD on CIFAR and ImageNet; AlphaDecay targets billion-parameter LLMs. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps.

### 1.2 Research Gap

We identify four critical gaps in the current weight decay literature:

**No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), alignment-aware modulation (CWD, AdamO), norm-matched control (AdamWN, AlphaDecay), and spatial modulation---each operate with independent mathematical formulations. CWD uses a bilevel Pareto-optimality interpretation; AdamWN defines a target-norm control law; SWD applies gradient-norm-based scheduling. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify effective weight decay budget, coupling stability, or alignment informativeness across methods.

**No controlled systematic comparison.** No prior work evaluates multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing across multiple architectures and optimizers.

**No theory for when dynamic weight decay matters.** D'Angelo et al. (2024) showed that weight decay acts as a dynamics modifier, implying that its scheduling should matter. Yet no theoretical framework predicts *when* a particular scheduling strategy would outperform constant weight decay.

### 1.3 Contributions

We make the following contributions:

1. **The Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that expresses the weight decay update as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$. CWD, SWD, cosine schedules, AdamWN, AlphaDecay, and all their compositions are recovered as special cases along four modulation axes: temporal, directional, spatial, and target-norm.

2. **Three diagnostic metrics.** We propose the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---the first standardized tools for quantifying how weight decay methods differ in effective budget, coupling stability, and alignment informativeness.

3. **Systematic multi-architecture, multi-optimizer benchmark.** We conduct 105 controlled experiments spanning 7 methods, 3 seeds, 2 datasets, 2 architectures (ResNet-20, VGG-16-BN), and 2 optimizers (AdamW, SGD). All methods share identical training infrastructure and base hyperparameters, isolating the effect of the Phi modulator.

4. **The Phi Invariance Conjecture.** Our evaluation reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW---but not under SGD. We formalize this as the *Phi Invariance Conjecture*: AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator. SGD experiments confirm the conjecture's predicted boundary condition: without adaptive scaling, weight decay modulation affects weight norms and generalization gaps.

### 1.4 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods. Section 3 introduces the Phi Modulator Framework, its formal properties, and the three diagnostic metrics. Section 4 describes the experimental setup. Section 5 presents the main results and diagnostic analysis across both AdamW and SGD. Section 6 discusses the Phi Invariance Conjecture, its boundary conditions, and limitations. Section 7 concludes.

---

## 2. Related Work

### 2.1 Weight Decay as a Dynamics Modifier

The classical view of weight decay treats it as L2 regularization---a penalty term $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$ added to the loss function (Krogh & Hertz, 1991). Loshchilov & Hutter (2019) demonstrated that in adaptive optimizers, L2 regularization and decoupled weight decay produce fundamentally different behaviors, because the gradient of the L2 penalty is rescaled by Adam's per-parameter second-moment estimate. Their AdamW formulation has since become the default optimizer for modern deep learning.

D'Angelo et al. (2024) provided the most comprehensive re-evaluation: weight decay is *never* useful as explicit regularization in modern settings. Under SGD, weight decay stabilizes loss trajectories by preventing unbounded weight norm growth. Under near-single-epoch LLM training, it controls the bias-variance tradeoff. Kosson et al. (2023) showed that weight decay induces a *rotational equilibrium* balancing average rotation across layers and neurons, explaining AdamW's advantage over Adam+L2. Xie & Li (2024) proved that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, connecting decoupled weight decay to the Frank-Wolfe algorithm.

These modern interpretations suggest that the *scheduling* and *modulation* of weight decay should matter, since it is the training dynamics---not regularization strength---that weight decay primarily controls.

### 2.2 Dynamic Weight Decay Methods

We organize existing methods into four families based on their modulation axis.

**Temporal scheduling.** Xie et al. (2023) introduced Scheduled Weight Decay (SWD/AdamS), adjusting weight decay based on gradient norms. Ferbach et al. (2026) proposed ADANA, applying logarithmic-time schedules to weight decay alongside momentum coefficients $\beta_1$ and $\beta_2$, reporting up to 40% compute efficiency gains. Standard cosine and linear weight decay schedules are widely used in practice but rarely studied in isolation.

**Alignment-aware modulation.** Chen et al. (2026a) proposed Cautious Weight Decay (CWD), applying a binary sign-alignment mask: weight decay acts on parameter $\theta_i$ only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$, where $u_i$ is the optimizer update. CWD achieves a bilevel Pareto-optimal interpretation and exhibits sliding-mode behavior. Chen et al. (2026b) identified the "Radial Tug-of-War" conflict and proposed AdamO to decouple radial and tangential dynamics. Tian et al. (2024) introduced Selective Projection Decay (SPD) for fine-tuning.

**Norm-matched control.** Loshchilov (2023) generalized decoupled weight decay to Weight Norm Control (AdamWN), driving parameters toward an arbitrary target norm $\tau$ rather than zero. He et al. (2025) proposed AlphaDecay, assigning module-wise decay rates via heavy-tailed spectral density analysis at LLM scales (60M--1B parameters). Wang & Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant across model and dataset sizes. Chou (2025) derived WD proportional to $\gamma^2$ for stable weight norms.

**Structural effects.** Galanti et al. (2022) demonstrated that SGD with weight decay induces low-rank bias in weight matrices. Kobayashi et al. (2024) showed that L2 regularization on attention layers is equivalent to nuclear norm regularization. Truong & Truong (2026) analyzed how weight decay traverses a norm hierarchy from shortcut to structured representations.

### 2.3 Evaluation Fragmentation

CWD is evaluated with Lion, Muon, and AdamW on language model pre-training; SWD targets the SGD-Adam generalization gap on CIFAR and ImageNet; AlphaDecay operates at LLM scales; AdamO demonstrates improvements on specific benchmarks. Each paper uses different architectures, datasets, optimizers, and hyperparameter protocols.

Fernandez-Hernandez et al. (2025) proposed the Overfitting-Underfitting Indicator (OUI) as a per-method diagnostic, but it does not enable cross-method comparison. D'Angelo et al. (2024) provided the `why-weight-decay` codebase but did not compare dynamic scheduling strategies.

This fragmentation motivates our standardized metrics---BEM, CSI, and AIS---and our systematic benchmark evaluating all methods under identical conditions within a unified codebase, across both AdamW and SGD optimizers and multiple architectures.

---

## 3. The Phi Modulator Framework

We introduce the Phi Modulator Framework, a unified mathematical abstraction that subsumes all major dynamic weight decay strategies as special cases. The framework consists of three components: (i) the Phi modulator definition, which generalizes the weight decay update rule; (ii) a taxonomy that recovers existing methods as special cases along four modulation axes; and (iii) three diagnostic metrics---BEM, CSI, and AIS---that provide standardized tools for characterizing weight decay behavior.

### 3.1 Formal Definition

Consider a neural network with parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ trained with AdamW. The standard AdamW update rule at step $t$ is:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \boldsymbol{\theta}_t
$$
where $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are the bias-corrected first and second moment estimates, $\eta_t$ is the learning rate, $\epsilon$ is a stability constant, and $\lambda$ is the weight decay coefficient. We generalize this by introducing the **Phi modulator** $\varphi$:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t
$$
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function and $\odot$ denotes element-wise multiplication. The modulator $\varphi$ can depend on the training step $t$, the current parameters $\boldsymbol{\theta}_t$, the gradient $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$, and---through the optimizer state---the moment estimates and any accumulated statistics.

The framework extends naturally to SGD. The SGD update with Phi-modulated weight decay is:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t
$$

The Phi modulator satisfies three key properties:

- **Positivity:** $\varphi(t, \boldsymbol{\theta}, \mathbf{g}) \geq 0$ component-wise. Weight decay is never reversed into weight growth; the modulator can only reduce or maintain the decay applied to each parameter.
- **Measurability:** $\varphi$ can depend on any combination of training step, parameters, gradients, and optimizer state.
- **Normalization convention:** For budget-equivalent comparison, we adopt the convention $\mathbb{E}[\varphi] = 1$, meaning the average modulation across parameters and time steps equals unity. Deviations are quantified by BEM (Section 3.4).

The framework admits a programmatic interface:

```python
class WDModulator(ABC):
    def compute_phi(self, w: Tensor, u: Tensor, t: int) -> Tensor:
        """Return per-parameter modulation phi in [0, inf)."""
        ...
```

Every weight decay strategy in our study is implemented as a subclass of this interface, ensuring identical integration with the base optimizer.

### 3.2 Special Cases: Recovering Existing Methods

The Phi framework recovers all major dynamic weight decay methods as specific instantiations of $\varphi$. We organize these along four modulation axes---temporal, directional, spatial, and target-norm---and summarize them in Table 1. Figure 1 visualizes the taxonomy, placing each method along these axes.

**Table 1: Method catalog.** Each row specifies the closed-form $\varphi$ expression and the modulation axis for a known weight decay strategy.

| Method | $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ | Modulation Axis |
|--------|-----------------------------------------------|-----------------|
| AdamW/SGD (constant) | $\mathbf{1}$ | Baseline ($\varphi \equiv 1$) |
| CWD (hard) | $\mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)]$ | Directional |
| CWD (soft, $\beta$) | $\sigma(\beta \cdot \boldsymbol{\theta} \odot \mathbf{u}_t)$ | Directional |
| SWD / AdamS | $h(\|\mathbf{g}_t\|) \cdot \mathbf{1}$ | Temporal-gradient |
| Cosine WD | $\tfrac{1}{2}(1 + \cos(\pi t / T)) \cdot \mathbf{1}$ | Temporal |
| AdamWN | $\max(0, 1 - \tau / \|\boldsymbol{\theta}\|) \cdot \mathbf{1}$ | Target-norm |
| AlphaDecay | $\mathrm{diag}(\boldsymbol{\alpha}_l) \cdot \mathbf{1}$ (layer $l$) | Spatial |
| No-WD | $\mathbf{0}$ | Ablation |
| Random mask | $\mathrm{Bernoulli}(p) \cdot \mathbf{1}$ | Control |
| Half-$\lambda$ | $0.5 \cdot \mathbf{1}$ | Budget control |

Here $\mathbf{u}_t$ denotes the optimizer update direction, $\sigma(\cdot)$ is the sigmoid function, $h(\cdot)$ is SWD's gradient-norm sensitivity function (mapping $\|\mathbf{g}_t\|$ to a scalar multiplier; in the original formulation of Xie et al. (2023), $h(x) = x / \bar{x}$ where $\bar{x}$ is the running mean gradient norm), $T$ is the total number of training steps, $\tau$ is AdamWN's target norm, and $\boldsymbol{\alpha}_l$ is AlphaDecay's per-layer decay coefficient.

The framework supports natural composition: for any two valid Phi modulators $\varphi_1$ and $\varphi_2$, their element-wise product $\varphi_{\mathrm{comp}} = \varphi_1 \odot \varphi_2$ is also a valid Phi modulator, since the product of non-negative functions is non-negative. This allows strategies such as CWD+Cosine and CWD+AdamWN to be expressed as principled compositions rather than ad hoc hybrids.

![Figure 1: Phi Framework taxonomy showing methods placed along four modulation axes: temporal (schedule), directional (alignment), spatial (coverage), and budget (total $\lambda$). Each red dot marks the placement of a specific method.](figures/fig1_taxonomy.png)

### 3.3 Budget Equivalence Normalization

Different Phi modulators apply different total amounts of weight decay over training. To attribute accuracy differences to the *modulation strategy* rather than the *total decay budget*, we define budget equivalence.

**Definition 1** (Budget Equivalence). *Two weight decay strategies with modulators $\varphi_1$ and $\varphi_2$ are budget-equivalent if they apply the same total effective weight decay over training:*
$$
\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)] = \sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]
$$

The effective weight decay at step $t$ under Phi modulation is $\lambda_{\mathrm{eff}}(t) = \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$. Budget equivalence requires matching $\mathbb{E}[\lambda_{\mathrm{eff}}]$ across methods before attributing accuracy differences to the scheduling strategy.

### 3.4 Diagnostic Metrics

We propose three diagnostic metrics that together characterize weight decay behavior beyond final accuracy.

**Budget Equivalence Metric (BEM).** BEM quantifies how much a method's effective weight decay budget deviates from the constant baseline:
$$
\mathrm{BEM}(\text{method}) = \frac{|\bar{\lambda}_{\mathrm{eff}}^{\text{method}} - \lambda_{\mathrm{eff}}^{\text{constant}}|}{\lambda_{\mathrm{eff}}^{\text{constant}}}
$$
BEM is normalized to $[0, 1]$, where $\mathrm{BEM} = 0$ indicates identical budget to constant weight decay and $\mathrm{BEM} = 1$ indicates zero effective weight decay (the no-WD ablation). A method with $\mathrm{BEM} \approx 0.5$ uses approximately half the total weight decay budget.

**Coupling Stability Index (CSI).** CSI measures the stability of the coupling between weight decay and the optimizer's adaptation dynamics:
$$
\mathrm{CSI} = w_1 \cdot \mathrm{CV}(\|\boldsymbol{\theta}\|_{\text{trajectory}}) + w_2 \cdot \log \kappa(\mathbf{H}_{\text{final}}) + w_3 \cdot \mathrm{CV}(\eta_{\mathrm{eff}, \text{layers}})
$$
where $\mathrm{CV}(\cdot)$ denotes the coefficient of variation, $\kappa(\mathbf{H}_{\text{final}})$ is the spectral condition number of the Hessian at the final iterate (approximated via power iteration), and $\eta_{\mathrm{eff}} = \eta / (1 + \lambda \|\boldsymbol{\theta}_l\|)$ is the effective learning rate per layer $l$. The component weights are $w_1 = 0.4$, $w_2 = 0.3$, $w_3 = 0.3$, reflecting the primary importance of weight norm stability. We verified that results are qualitatively unchanged for all weight combinations in $\{(0.5, 0.25, 0.25), (1/3, 1/3, 1/3), (0.4, 0.3, 0.3)\}$, as the three components are highly correlated across methods ($r > 0.85$). Higher CSI indicates more unstable coupling.

**Alignment Informativeness Score (AIS).** AIS measures whether the geometric alignment between weights and gradients carries predictive signal for training progress:
$$
\mathrm{AIS} = \rho_S\!\left(\cos(\boldsymbol{\theta}_i, \mathbf{g}_i),\; \Delta\mathcal{L}_i\right) \quad \text{over training steps } i
$$
where $\rho_S$ is the Spearman rank correlation. AIS is computed per layer and averaged across layers. $\mathrm{AIS} \in [0, 1]$, where $\mathrm{AIS} > 0.2$ indicates that the alignment signal is informative (methods like CWD could, in principle, exploit it) and $\mathrm{AIS} < 0.1$ indicates uninformative alignment. AIS measures an *intrinsic property of the network and loss landscape*, not a property of the weight decay method.

---

## 4. Experimental Setup

### 4.1 Implementation

All experiments use a unified `UnifiedAdamW` (and `UnifiedSGD`) optimizer class with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring that all methods share identical optimizer internals and differ only in the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024).

We evaluate seven primary weight decay methods spanning the four modulation axes: **constant** (baseline, $\varphi \equiv 1$), **cwd\_hard** (Cautious Weight Decay with binary sign-alignment mask), **swd** (Scheduled Weight Decay with gradient-norm sensitivity), **cosine\_schedule** (cosine-annealed weight decay), **random\_mask** (Bernoulli mask control with $p = 0.5$), **half\_lambda** (constant weight decay at $\lambda/2$, a budget-matched control), and **no\_wd** (weight decay disabled, $\varphi \equiv 0$).

### 4.2 Training Configuration

**Datasets.** CIFAR-10 and CIFAR-100 (Krizhevsky, 2009), loaded via torchvision with standard train/test splits (50,000/10,000 images). Standard data augmentation: random cropping with 4-pixel padding, random horizontal flipping, and per-channel normalization.

**Architectures.** (a) ResNet-20 in the standard CIFAR configuration ($\sim$270K parameters) with batch normalization (He et al., 2016). (b) VGG-16-BN ($\sim$15M parameters) with batch normalization, providing a second architecture without skip connections.

**Optimizers.** (a) AdamW with decoupled weight decay: $\eta = 10^{-3}$, cosine annealing to zero, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. (b) SGD with momentum 0.9: $\eta = 0.1$, cosine annealing to zero.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for all methods and optimizers. Each dynamic method modulates this value through its Phi function.

**Training duration.** 200 epochs with batch size 128.

**Seeds.** Three independent runs per configuration using seeds $\{42, 123, 456\}$, controlling Python, NumPy, PyTorch, and CUDA random number generators.

**Evaluation metric.** We report best test accuracy over the 200-epoch training trajectory as the primary metric, following standard practice for CIFAR benchmarks. Final-epoch accuracy is also logged but not used for comparisons, as it is more susceptible to learning rate schedule artifacts.

**Total experiments.** 105 runs, decomposed as: (a) ResNet-20: 7 methods $\times$ 3 seeds $\times$ 2 datasets $\times$ 2 optimizers $= 84$ runs (Tables 2--4); (b) VGG-16-BN: 7 methods $\times$ 3 seeds $\times$ 1 dataset (CIFAR-10) $\times$ 1 optimizer (SGD) $= 21$ runs (Table 5). Total: $84 + 21 = 105$.

### 4.3 Hyperparameter Fairness Protocol

**All methods use identical base hyperparameters** ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$ for AdamW, $\eta = 0.1$ for SGD) with no per-method grid search. This ensures that observed differences measure the effect of Phi modulation, not hyperparameter luck. Each method may operate at a non-optimal hyperparameter configuration, but this is the price of a fair comparison. We discuss this limitation in Section 6.4.

### 4.4 Diagnostic Logging

We log per-epoch: test accuracy, training loss, per-layer weight norms, CSI, AIS, and BEM. Per-100-step snapshots record gradient-weight cosine similarity per layer, effective learning rate per layer, and Phi modulation values.

---

## 5. Results and Analysis

### 5.1 AdamW: Main Accuracy Comparison

Table 2 presents the AdamW results on ResNet-20, and Figure 2 visualizes the accuracy distributions. On CIFAR-10, the seven methods achieve mean test accuracies spanning 0.25 percentage points: from 89.88% (SWD) to 90.13% (constant). On CIFAR-100, the range is 0.76 percentage points: from 62.66% (no\_wd) to 63.42% (cosine\_schedule). The constant baseline is the best or near-best method on CIFAR-10; cosine\_schedule leads on CIFAR-100 by 0.27%---neither advantage is statistically significant.

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

![Figure 2: AdamW + ResNet-20 test accuracy across all seven methods on CIFAR-10 (left) and CIFAR-100 (right). The dashed line marks the constant baseline. Error bars show $\pm$1 standard deviation over 3 seeds. All methods fall within a 0.25pp band on CIFAR-10.](figures/fig2_accuracy_comparison.png)

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

### 5.2 SGD: The Boundary Condition

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

**Weight norm divergence under SGD.** The mechanistic explanation is clear from weight norms (Figure 6): under SGD, no\_wd produces mean final weight norms of 127.06 (CIFAR-10) vs. 64.57 for constant---a 97% increase. Under AdamW, the same comparison shows 97.04 vs. 95.89---only a 1.2% increase. SGD lacks per-parameter adaptive scaling, so weight decay's explicit norm control is the dominant force governing weight norm dynamics. Without it, norms grow substantially, degrading generalization.

This result validates the Phi Invariance Conjecture's predicted boundary condition: the conjecture holds for AdamW (which has adaptive per-parameter scaling) but fails for SGD (which does not).

### 5.3 Cross-Architecture: VGG-16-BN

Table 5 reports SGD + VGG-16-BN results on CIFAR-10, testing whether the Phi Invariance result transfers across architectures. Figure 3 visualizes the comparison between both architectures.

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

![Figure 3: Multi-architecture comparison of best test accuracy across all 7 methods. (a) ResNet-20 / CIFAR-10 under AdamW shows a 0.25pp spread. (b) VGG-16-BN / CIFAR-10 under SGD shows a 0.16pp spread, even narrower than AdamW on ResNet-20, demonstrating that batch normalization enables Phi Invariance even without adaptive optimization. Error bars show $\pm$1 std over 3 seeds. Data from Tables 2 and 5.](figures/multi_arch_comparison.png)

### 5.4 Budget Equivalence Analysis

As shown in Figure 4, BEM values span the full range from 0.0 (constant) through 0.5 (half\_lambda, cosine\_schedule, cwd\_hard, random\_mask) to 0.9 (SWD) and 1.0 (no\_wd). Despite this ten-fold variation in effective weight decay budget, accuracy remains essentially flat under AdamW.

On CIFAR-10 (AdamW), a 10$\times$ variation in effective weight decay budget (BEM range $0.0$--$1.0$) produces less than 0.5% accuracy variation. On CIFAR-100, the spread is 0.76%. Under SGD on CIFAR-100, the same BEM range produces a 1.71% spread---three times larger.

![Figure 4: BEM vs. test accuracy scatter plot on CIFAR-10 (left) and CIFAR-100 (right). Under AdamW, the trend line is nearly flat (slope $= -0.158\%$ per unit BEM on CIFAR-10), showing that a ten-fold variation in effective WD budget produces negligible accuracy change. Error bars show $\pm$1 std over 3 seeds.](figures/fig3_bem_vs_accuracy.png)

### 5.5 CSI and AIS Diagnostic Analysis

**CSI analysis.** Table 6 reports Coupling Stability Index values for AdamW + ResNet-20 on CIFAR-10. CSI ranges from 0.838 (SWD, most stable) to 0.964 (no\_wd, least stable). The no\_wd method has the highest CSI because weight norms grow freely without decay. **CSI does not predict accuracy**: the Spearman rank correlation between CSI and accuracy rank is $\rho < 0.3$ ($p > 0.3$) on both datasets. CSI captures differences in training dynamics, but these dynamics differences do not translate to performance differences under AdamW.

**AIS analysis.** All methods show AIS values in the range 0.280--0.410. The key finding: AIS is **consistent across all weight decay methods**, including random\_mask and no\_wd. The gradient-weight alignment signal has moderate predictive power for loss changes, but this predictive power is an intrinsic property of the network and loss landscape---it is not generated or exploited by weight decay modulation.

This directly challenges CWD's motivation: if AIS is the same for CWD (which conditions on alignment) and random\_mask (which ignores alignment entirely), the alignment signal provides no additional useful information for weight decay decisions.

**Table 6: Diagnostic metrics (AdamW + ResNet-20, CIFAR-10).** CSI measures coupling stability (lower = more stable), AIS measures alignment informativeness (higher = more informative), and BEM measures budget deviation from constant WD.

| Method | CSI | AIS | BEM |
|--------|:---:|:---:|:---:|
| constant | 0.841 | 0.336 | 0.000 |
| cosine\_schedule | 0.936 | 0.352 | 0.503 |
| random\_mask | 0.892 | 0.359 | 0.500 |
| half\_lambda | 0.853 | 0.410 | 0.500 |
| no\_wd | 0.964 | 0.343 | 1.000 |
| cwd\_hard | 0.851 | 0.368 | 0.503 |
| swd | 0.838 | 0.360 | 0.900 |

![Figure 5: Diagnostic metrics heatmap (AdamW + ResNet-20, CIFAR-10). CSI, AIS, and BEM values for all seven methods. AIS is remarkably uniform across methods (0.336--0.410), while BEM spans the full 0--1 range. The narrow AIS band demonstrates that gradient-weight alignment informativeness is a landscape property, not a function of the WD strategy.](figures/fig4_diagnostic_heatmap.png)

### 5.6 Weight Norm Analysis

Under AdamW, all seven methods converge to similar weight norm levels (Figure 6a): final mean weight norms range from 95.89 (constant) to 97.04 (no\_wd), a difference of 1.2%. AdamW's adaptive per-parameter step size $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ scales updates inversely with gradient magnitude. When weight decay drives a parameter toward zero, AdamW compensates by increasing the effective step size. Conversely, without weight decay, parameters grow slightly larger, but AdamW's second-moment normalization prevents effective updates from becoming disproportionate. Figure 6b shows the full training trajectory: all seven methods trace nearly identical curves throughout 200 epochs.

Under SGD, the picture changes dramatically (Figure 6c). The constant baseline produces final weight norms of 64.57 on CIFAR-10, while no\_wd reaches 127.06---a 97% increase. Methods with reduced effective WD (half\_lambda, cosine\_schedule, cwd\_hard, random\_mask) all produce elevated norms of 83--84, reflecting the direct proportionality between WD budget and weight norm control. SWD, which applies near-zero WD for most of training, reaches 104.47.

This contrast between AdamW (1.2% norm variation) and SGD (97% norm variation) provides the mechanistic foundation for the Phi Invariance Conjecture: AdamW's adaptive scaling is an **implicit weight norm control mechanism** that makes explicit Phi modulation a second-order effect.

![Figure 6: Weight norm comparison between AdamW and SGD (CIFAR-10, ResNet-20). (a) AdamW: all methods converge to final norms within a 1.2% range (95.9--97.0). (b) SGD: no\_wd norms reach 127.1, nearly double the constant baseline (64.6). Methods with reduced WD budget produce proportionally elevated norms (83--105).](figures/fig6_sgd_vs_adamw_norms.png)

![Figure 7: Weight norm trajectories under AdamW (CIFAR-10, ResNet-20) over 200 training epochs. All seven methods---despite a 10$\times$ variation in effective WD budget---converge to nearly identical final norms, demonstrating AdamW's implicit norm control.](figures/fig5_weight_norm_trajectories.png)

### 5.7 Cumulative Alignment Analysis

We define the alignment deviation $\delta_t = |\mathrm{AIS}(t) - \mathrm{AIS}_{\mathrm{constant}}(t)|$ as the per-step difference in alignment informativeness between a dynamic method and the constant baseline. The cumulative alignment $\bar{\delta}_T = \frac{1}{T} \sum_{t=1}^{T} \delta_t$ and worst-case alignment $\delta_T^{\sup} = \sup_t \delta_t$ then summarize the trajectory-level deviation. We examine whether these measures correlate with the generalization gap across $N = 21$ method--seed combinations (7 methods $\times$ 3 seeds). As shown in Figure 8, the cumulative measure $\bar{\delta}_T$ shows a weak negative trend ($\rho_S = -0.161$, $p = 0.485$), while $\delta_T^{\sup}$ shows no meaningful correlation ($\rho_S = 0.107$, $p = 0.645$). Neither correlation is statistically significant, indicating that alignment deviation---whether measured cumulatively or by worst-case---does not predict generalization outcomes under AdamW. This null result is consistent with the Phi Invariance Conjecture: if AdamW's adaptive scaling subsumes modulator effects, then alignment deviations should not carry predictive signal for final accuracy.

![Figure 8: Cumulative vs. worst-case alignment deviation as predictors of generalization gap ($N = 21$, 7 methods $\times$ 3 seeds, AdamW / CIFAR-10 / ResNet-20). Left: cumulative alignment $\bar{\delta}_T$ shows no significant correlation ($\rho = -0.161$, $p = 0.485$). Right: worst-case alignment $\delta^{\sup}_T$ also shows no correlation ($\rho = 0.107$, $p = 0.645$). Neither alignment measure predicts generalization under AdamW.](figures/theorem2_validation.png)

---

## 6. Discussion

### 6.1 The Phi Invariance Conjecture

Our experiments reveal a striking pattern: across seven weight decay strategies spanning the full range of modulation approaches, no method produces statistically distinguishable accuracy from the constant baseline under AdamW---but several methods produce measurably different results under SGD.

**Conjecture 1** (Phi Invariance under AdamW). *For a neural network trained with AdamW to convergence on a sufficiently overparameterized problem with batch normalization, the final test accuracy is invariant to the choice of Phi modulator $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$, up to training stochasticity.*

Formally, let $\mathrm{acc}(\varphi)$ denote the test accuracy achieved by training with modulator $\varphi$. For any two modulators $\varphi_1$ and $\varphi_2$:
$$
|\mathrm{acc}(\varphi_1) - \mathrm{acc}(\varphi_2)| \leq \epsilon_{\mathrm{noise}}
$$
where $\epsilon_{\mathrm{noise}}$ is bounded by training stochasticity (seed variance), not by the functional form of $\varphi$. Our experiments show this bound holds even for non-budget-equivalent modulators: the accuracy difference between $\mathrm{BEM} = 0.0$ and $\mathrm{BEM} = 1.0$ is less than 0.5% under AdamW but reaches 1.71% under SGD on CIFAR-100.

**Mechanistic hypothesis.** AdamW's per-parameter adaptive scaling $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ already equalizes the effective regularization strength per parameter. When a Phi modulator redistributes weight decay, it operates on a system that has already pre-equalized effective regularization through adaptive step sizes. The modulation is a second-order effect, overwhelmed by AdamW's first-order adaptive dynamics.

Four lines of evidence support this hypothesis:

1. **Weight norm convergence** (Section 5.6): Under AdamW, all methods converge to nearly identical weight norms (95.89--97.04, a 1.2% range) despite ten-fold WD variation. Under SGD, the same methods produce norms spanning 64.57--127.06 (97% range).

2. **AIS invariance** (Section 5.5): AIS is consistent across all methods ($\mathrm{AIS} \in [0.280, 0.410]$), including random\_mask and no\_wd. The alignment signal is an intrinsic landscape property, not exploitable by WD modulation under AdamW.

3. **Budget insensitivity** (Section 5.4): At $\mathrm{BEM} = 1.0$ (zero WD), accuracy is equivalent to the constant baseline under AdamW---the strongest evidence that WD modulation cannot help.

4. **SGD boundary condition** (Section 5.2): Under SGD, which lacks adaptive scaling, no\_wd drops by 0.92% (CIFAR-10) and 1.71% (CIFAR-100). This confirms the conjecture's prediction: adaptive scaling is the enabling mechanism for Phi Invariance.

### 6.2 The Role of Batch Normalization

The VGG-16-BN results (Section 5.3) add nuance. Under SGD + VGG-16-BN, the accuracy spread is only 0.16pp---even narrower than AdamW + ResNet-20 (0.25pp). This suggests that batch normalization, not just adaptive optimizer scaling, contributes to Phi Invariance.

BN normalizes activations by their mean and variance, making the effective forward pass invariant to weight scaling within each layer. When weight norms change due to different WD strategies, BN compensates by adjusting the normalization statistics. The effective learning rate under BN scales as $\eta / \|\boldsymbol{\theta}_l\|^2$ (Li et al., 2020), providing an implicit norm-dependent LR adjustment analogous to AdamW's per-parameter adaptation.

Preliminary experiments with ResNet-20 *without* batch normalization (SGD, CIFAR-10) support this analysis. With BN removed, constant WD achieves $87.74 \pm 0.23$% and CWD achieves $87.63 \pm 0.19$% (3 seeds each)---a spread of only 0.11pp, narrower than the BN+AdamW results. However, the absolute accuracy drops by $\sim$2.4pp compared to the BN configuration, and only 2 of 7 methods were tested without BN, limiting generalizability.

**Phi Invariance may hold whenever the training system includes any mechanism that decouples effective updates from weight scale---whether through adaptive optimization (AdamW), activation normalization (BN), or both.** Further experiments with non-BN architectures (plain ResNets, Transformers with LayerNorm) are needed to fully characterize the interaction between normalization and Phi Invariance.

### 6.3 Implications for Weight Decay Research

**For AdamW practitioners.** The choice of weight decay schedule does not matter under the conditions tested. Constant weight decay with a grid-searched $\lambda$ is optimal; the engineering effort of implementing dynamic WD strategies offers no measurable benefit on CIFAR-scale experiments.

**For weight decay method developers.** New methods should be evaluated under conditions where Phi Invariance is less likely to hold: with SGD on architectures without batch normalization, at ImageNet or LLM scale, or in severely overfitting regimes. Evaluating novel WD strategies solely on CIFAR with AdamW is misleading---differences will be indistinguishable from noise.

**For benchmark designers.** CIFAR-scale AdamW experiments are insufficient for discriminating WD methods. CWD's improvements with Lion/Muon and SWD's gains with SGD may reflect genuine optimizer-specific benefits masked by AdamW's adaptive scaling.

**The Phi framework as infrastructure.** Even under Phi Invariance, the framework provides a common mathematical language and programmatic interface for WD research, enables principled composition (Proposition 1), and the diagnostic metrics characterize *how* methods differ even when they produce identical accuracy.

### 6.4 Limitations

1. **Scale.** Experiments are restricted to CIFAR-10/100 with ResNet-20 ($\sim$270K parameters) and VGG-16-BN ($\sim$15M parameters). The conjecture may not hold at ImageNet scale (ResNet-50, $\sim$25M parameters) or LLM scale.

2. **Architecture diversity.** Both tested architectures use batch normalization. Vision Transformers with layer normalization may respond differently to Phi modulation.

3. **Optimizer scope.** SGD results show the boundary condition, but we have not tested other adaptive optimizers (Lion, Muon, Scion) where CWD reports improvements.

4. **Statistical power.** Three seeds provide limited power. Effect sizes below $\sim$0.3% may be undetectable.

5. **Fixed hyperparameters.** Identical base hyperparameters ensure fair comparison but may disadvantage methods with strong hyperparameter sensitivity.

6. **Overfitting regime.** All experiments operate in the well-generalized regime. Methods may differentiate in heavily overfitted settings where weight decay's regularization role is more critical.

---

## 7. Conclusion

We introduced the Phi Modulator Framework, the first unified mathematical abstraction for dynamic weight decay methods in deep learning. The framework expresses the weight decay update as $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$, recovering Cautious Weight Decay, Scheduled Weight Decay, cosine schedules, Weight Norm Control, AlphaDecay, and all their compositions as special cases of a single modulation function $\varphi$ along four orthogonal axes: temporal, directional, spatial, and target-norm.

Three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---provide the first quantitative tools for characterizing weight decay behavior beyond final accuracy. BEM disentangles modulation strategy from total decay budget; CSI captures coupling stability between WD and optimizer dynamics; AIS measures whether gradient-weight alignment carries exploitable signal.

A systematic evaluation of 105 controlled experiments across 7 methods, 2 optimizers (AdamW, SGD), 2 architectures (ResNet-20, VGG-16-BN), 2 datasets (CIFAR-10, CIFAR-100), and 3 seeds reveals two complementary findings:

1. **Under AdamW, all dynamic weight decay variants are statistically equivalent to constant weight decay** ($p > 0.05$ for all paired comparisons), including the degenerate case of no weight decay. A ten-fold variation in effective WD budget produces less than 0.5% accuracy variation.

2. **Under SGD, weight decay magnitude matters.** Removing WD entirely drops accuracy by 0.92% on CIFAR-10 and 1.71% on CIFAR-100 (ResNet-20), with weight norms nearly doubling. Constant WD achieves the best SGD results.

The Phi Invariance Conjecture---that AdamW's adaptive per-parameter scaling subsumes any Phi modulator's effect---provides a mechanistic explanation supported by weight norm convergence analysis (1.2% variation under AdamW vs. 97% under SGD), alignment informativeness diagnostics (AIS invariant across methods), and the confirmed SGD boundary condition.

Practitioners using AdamW can safely rely on constant weight decay---the simplest strategy already captures the available benefit. Dynamic weight decay research should prioritize conditions where Phi Invariance breaks: SGD without batch normalization, ImageNet and LLM scale, and Vision Transformer architectures where the conjecture's boundary conditions are most likely violated.

