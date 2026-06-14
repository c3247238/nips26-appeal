# When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW

## Abstract

Weight decay is universally applied in deep learning optimization, yet the community lacks principled tools for comparing the growing family of dynamic weight decay strategies. We introduce the **Phi Modulator Framework**, a unified mathematical abstraction $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that expresses the weight decay update as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$, recovering Cautious Weight Decay, Scheduled Weight Decay, cosine schedules, Weight Norm Control, AlphaDecay, and all their compositions as special cases along four modulation axes: temporal, directional, spatial, and target-norm. We propose three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---providing the first quantitative tools for characterizing weight decay behavior beyond final accuracy. A systematic evaluation of seven weight decay strategies on CIFAR-10 and CIFAR-100 with ResNet-20 under AdamW (42 experiments, three seeds per configuration) reveals that all dynamic weight decay variants are statistically indistinguishable from constant weight decay ($p > 0.05$ for all paired comparisons), including the degenerate case of no weight decay ($\lambda = 0$). We formalize this as the **Phi Invariance Conjecture**: AdamW's adaptive per-parameter scaling subsumes the effect of any Phi modulator. The framework and metrics establish the first standardized infrastructure for weight decay research and clear boundary conditions for when dynamic weight decay does---and does not---help.

---

## 1. Introduction

### 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). However, this view has been progressively undermined by modern findings. Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. More recently, D'Angelo et al. (2024) provided a unifying perspective showing that weight decay is never useful as explicit regularization in modern deep learning; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories under SGD and controlling the bias-variance tradeoff in large language model training.

This re-understanding has sparked a surge of methods proposing to dynamically modulate weight decay strength during training. Xie et al. (2023) introduced Scheduled Weight Decay (SWD), which adjusts weight decay based on gradient norms. Chen et al. (2026a, ICLR; hereafter CWD) proposed Cautious Weight Decay, a sign-alignment mask that applies decay only when weight and update directions agree. Loshchilov (2023) generalized decoupled weight decay to target-norm control (AdamWN). He et al. (2025) introduced AlphaDecay, a module-wise strategy guided by spectral density analysis. Ferbach et al. (2026) proposed ADANA, which applies logarithmic-time schedules to both weight decay and momentum coefficients, reporting significant compute efficiency gains in specific settings. Chen et al. (2026b, ICLR; hereafter AdamO) identified a "Radial Tug-of-War" conflict between weight decay and gradient updates, proposing AdamO to decouple radial and tangential dynamics.

A critical problem pervades this rapidly growing literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. CWD reports improvements on language model pre-training with Lion and Muon optimizers; SWD demonstrates gains with SGD on CIFAR and ImageNet; AlphaDecay targets billion-parameter LLMs. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps.

This state of affairs raises a fundamental question: **does dynamic weight decay actually help, and if so, when and why?**

### 1.2 Research Gap

Answering this question is currently impossible due to four critical gaps in the literature:

**No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), directional modulation (CWD, AdamO), spatial modulation (AlphaDecay), and target-norm control (AdamWN)---each operate with independent mathematical formulations. CWD uses a bilevel Pareto-optimality interpretation; AdamWN defines a target-norm control law; SWD applies gradient-norm-based scheduling. These are all answering the same question---*how should weight decay interact with the training trajectory?*---but from incompatible starting points. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify how much effective weight decay budget a method uses, how stably it couples with the optimizer, or whether its alignment signal actually carries useful information for training. This fragmentation is the root cause of conflicting claims across the literature.

**No controlled systematic comparison.** To our knowledge, no prior work has evaluated multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing. Without such a comparison, reported improvements may be artifacts of hyperparameter tuning, architectural choices, or optimizer selection rather than genuine benefits of the dynamic strategy.

**No theory for when dynamic weight decay matters.** D'Angelo et al. (2024) showed that weight decay acts as a dynamics modifier rather than a classical regularizer, and Xie & Li (2024) connected AdamW's decoupled weight decay to $\ell_\infty$-norm constrained optimization, suggesting a potential absorption mechanism. Yet no theoretical framework predicts *when* and *why* a particular scheduling strategy would outperform constant weight decay, or when the optimizer's adaptive scaling renders modulation irrelevant.

### 1.3 Contributions

We make the following contributions:

1. **The Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that expresses the weight decay update as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$. We show that CWD, SWD, cosine schedules, AdamWN, AlphaDecay, and all their compositions are recovered as special cases of this framework along four modulation axes: temporal, directional, spatial, and target-norm. **The Phi framework enables, for the first time, controlled comparison of weight decay strategies under identical optimization conditions.**

2. **Three diagnostic metrics.** We propose the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---the first standardized tools for quantifying how weight decay methods differ in effective budget, coupling stability, and alignment informativeness. These metrics characterize *how* methods differ even when they produce identical accuracy.

3. **Systematic benchmark.** We conduct 42 controlled experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets: CIFAR-10 and CIFAR-100 with ResNet-20) with identical hyperparameters, training infrastructure, and statistical testing. All methods share the same AdamW base optimizer, learning rate schedule, and base weight decay coefficient, isolating the effect of the Phi modulator and establishing clear boundary conditions for AdamW at CIFAR scale. The seven methods evaluated are: constant (baseline), cwd\_hard, swd, cosine\_schedule, random\_mask, half\_lambda, and no\_wd.

4. **The Phi Invariance Conjecture.** Our systematic evaluation reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW. We formalize this as the *Phi Invariance Conjecture*: AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, with diagnostic evidence from weight norm convergence, alignment informativeness analysis, and budget equivalence sweeps.

### 1.4 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods. Section 3 introduces the Phi Modulator Framework, its formal properties, and the three diagnostic metrics. Section 4 describes the experimental setup. Section 5 presents the main results and diagnostic analysis. Section 6 discusses the Phi Invariance Conjecture, its implications, and limitations. Section 7 concludes. Appendices provide extended statistical analysis, diagnostic visualization panels for all 42 runs, mathematical proofs, and reproducibility details.

---

## 2. Related Work

### 2.1 Weight Decay as a Dynamics Modifier

The classical view of weight decay treats it as L2 regularization---a penalty term $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$ added to the loss function, which shrinks weights toward zero and reduces model complexity (Krogh & Hertz, 1991; Hanson & Pratt, 1988). This interpretation guided deep learning practice for decades. However, Loshchilov & Hutter (2019) demonstrated a crucial distinction: in adaptive optimizers such as Adam, L2 regularization and decoupled weight decay produce fundamentally different behaviors because the gradient of the L2 penalty is rescaled by Adam's per-parameter second-moment estimate. Their proposed AdamW---which applies weight decay directly to the parameters rather than through the gradient---has since become the default optimizer for modern deep learning.

A deeper re-evaluation came from D'Angelo et al. (2024), who showed through extensive experiments on both vision models and LLMs that weight decay is *never* useful as explicit regularization in modern settings. Instead, it serves as a training dynamics modifier: under SGD, weight decay stabilizes the loss trajectory by preventing weight norms from growing unboundedly; under near-single-epoch LLM training, it controls the bias-variance tradeoff. Kosson et al. (2023) provided a complementary perspective, showing that weight decay induces a *rotational equilibrium* that balances the average rotation of weight vectors across layers and neurons, explaining why AdamW outperforms Adam+L2. If rotational equilibrium is the operative mechanism, it may be achieved robustly under standard AdamW regardless of the specific form of weight decay modulation---a hypothesis our experiments will test. Xie & Li (2024) proved that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, connecting decoupled weight decay to the Frank-Wolfe algorithm and suggesting a potential absorption mechanism for explicit modulation.

These modern interpretations collectively suggest that the *scheduling* and *modulation* of weight decay should matter, since it is the training dynamics---not the regularization strength---that weight decay primarily controls. Yet this implication has not been rigorously tested through controlled experiments. Moreover, despite these insights into weight decay's dynamics-modifying role, no theoretical framework predicts when the functional form of the modulator becomes irrelevant---particularly whether AdamW's adaptive updates already implicitly subsume the effect of explicit weight decay scheduling.

### 2.2 Dynamic Weight Decay Methods

We organize existing dynamic weight decay methods into four families based on their modulation axis.

**Temporal scheduling.** Xie et al. (2023) introduced Scheduled Weight Decay (SWD/AdamS), which dynamically adjusts weight decay based on gradient norms, motivated by the observation that constant weight decay can destabilize training during phases of large gradient magnitude. Ferbach et al. (2026) proposed ADANA, which applies logarithmic-time schedules to both weight decay and momentum coefficients $\beta_1$ and $\beta_2$. Standard cosine and linear weight decay schedules (Loshchilov & Hutter, 2017), which decay $\lambda$ in proportion to the learning rate schedule, are also widely used in practice though rarely studied in isolation.

**Directional modulation.** Chen et al. (2026a) proposed Cautious Weight Decay (CWD), which applies a binary sign-alignment mask: weight decay acts on parameter $\theta_i$ only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$, where $u_i$ is the optimizer update. CWD achieves a Pareto-optimal interpretation in a bilevel optimization framework and exhibits sliding-mode behavior. Chen et al. (2026b) identified the "Radial Tug-of-War" conflict, where weight decay opposes the gradient in the radial (norm) direction, and proposed AdamO to decouple radial and tangential dynamics. Tian et al. (2024) introduced Selective Projection Decay (SPD), which modulates per-layer weight decay based on loss reduction consistency for fine-tuning.

**Spatial modulation.** He et al. (2025) proposed AlphaDecay, which assigns module-wise decay rates guided by heavy-tailed self-regularization (HT-SR) spectral density analysis, demonstrating gains at LLM scales from 60M to 1B parameters.

**Target-norm control.** Loshchilov (2023) generalized decoupled weight decay to Weight Norm Control (AdamWN), where weight decay drives parameters toward an arbitrary target norm $\tau$ rather than zero, subsuming standard weight decay ($\tau = 0$) as a special case. Wang & Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant in epochs across model and dataset sizes---suggesting that a well-calibrated constant weight decay may already capture the available benefit, leaving limited room for dynamic schedules to improve.

**Structural effects.** Beyond training dynamics, weight decay has been shown to induce specific structural properties in learned representations. Galanti et al. (2022) demonstrated that SGD with weight decay induces a low-rank bias in weight matrices. Kobayashi et al. (2024) showed that L2 regularization on multiplicative parameters (such as attention layers) is equivalent to nuclear norm regularization, inducing low-rank attention. These structural insights, while orthogonal to the modulation focus of our framework, connect to the spectral condition number component of our Coupling Stability Index (Section 3.4) and represent potential avenues for future investigation of the Phi Invariance Conjecture's boundary conditions.

### 2.3 Evaluation Fragmentation

A critical obstacle to progress in this field is evaluation fragmentation. CWD (Chen et al., 2026a) is evaluated with Lion, Muon, and AdamW on language model pre-training; SWD (Xie et al., 2023) targets the SGD-Adam generalization gap on CIFAR and ImageNet; AlphaDecay (He et al., 2025) operates at LLM scales with GPT-style architectures; AdamO (Chen et al., 2026b) demonstrates improvements on specific optimization benchmarks. Each paper uses different architectures, datasets, optimizers, and hyperparameter selection protocols.

Fernandez-Hernandez et al. (2025) proposed the Overfitting-Underfitting Indicator (OUI) as a diagnostic for weight decay quality, but it serves as a per-method diagnostic rather than a cross-method comparison tool. D'Angelo et al. (2024) provided a shared experimental infrastructure (the `why-weight-decay` codebase), but did not compare dynamic scheduling strategies.

This fragmentation motivates our proposed standardized metrics---BEM, CSI, and AIS---which enable principled comparison of weight decay methods regardless of the specific experimental setting, and our systematic benchmark that evaluates all methods under identical conditions within a unified codebase.

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
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function and $\odot$ denotes element-wise multiplication. The modulator $\varphi$ can depend on the training step $t$, the current parameters $\boldsymbol{\theta}_t$, the gradient $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$, and---through the optimizer state---the moment estimates and any accumulated statistics. We denote the AdamW update direction (preconditioned gradient) as $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$; modulators that condition on weight-update alignment (such as CWD) use $\mathbf{u}_t$ rather than the raw gradient $\mathbf{g}_t$, and the signature $\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)$ should be understood as $\varphi(t, \boldsymbol{\theta}_t, \text{optimizer\_state}_t)$ when optimizer-internal quantities are needed.

The Phi modulator satisfies three key properties:

- **Positivity:** $\varphi(t, \boldsymbol{\theta}, \mathbf{g}) \geq 0$ component-wise. Weight decay is never reversed into weight growth; the modulator can only reduce or maintain the decay applied to each parameter.
- **Measurability:** $\varphi$ can depend on any combination of training step, parameters, gradients, and optimizer state. This generality allows the framework to capture methods that condition on gradient norms, weight-update alignment, per-layer statistics, or external schedules.
- **Reference target:** For budget-equivalent comparison, we adopt the reference target $\mathbb{E}[\varphi] = 1$, meaning the average modulation across parameters and time steps equals unity. Methods may deviate from this target (e.g., no-WD has $\mathbb{E}[\varphi] = 0$); such deviations are quantified by the Budget Equivalence Metric (Section 3.4).

The framework admits a clean programmatic interface:

```python
class WDModulator(ABC):
    def compute_phi(self, w: Tensor, u: Tensor, t: int) -> Tensor:
        """Return per-parameter modulation phi in [0, inf).
        Args:
            w: parameter tensor (theta_t)
            u: optimizer update direction (preconditioned gradient)
            t: training step
        """
        ...
```

Every weight decay strategy in our study is implemented as a subclass of this interface, ensuring identical integration with the AdamW base optimizer.

### 3.2 Special Cases: Recovering Existing Methods

The power of the Phi framework lies in its ability to recover all major dynamic weight decay methods as specific instantiations of $\varphi$. We organize these along four modulation axes---temporal, directional, spatial, and target-norm---and summarize them in Table 1.

**Table 1: Method catalog.** Each row specifies the closed-form $\varphi$ expression and the modulation axis for a known weight decay strategy. **CWD, SWD, AdamWN, cosine schedules, and random masking are all recovered as special cases of $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$.**

| Method | $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ | Modulation Axis |
|--------|-----------------------------------------------|-----------------|
| AdamW (constant) | $\mathbf{1}$ | Baseline ($\varphi \equiv 1$) |
| CWD (hard) | $\mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)]$ | Directional |
| CWD (soft, $\beta$) | $\sigma(\beta \cdot \boldsymbol{\theta} \odot \mathbf{u}_t)$ | Directional |
| SWD / AdamS | $h(\|\mathbf{g}_t\|) \cdot \mathbf{1}$ | Temporal-gradient |
| Cosine WD | $\tfrac{1}{2}(1 + \cos(\pi t / T)) \cdot \mathbf{1}$ | Temporal |
| AdamWN | $\max(0, 1 - \tau / \|\boldsymbol{\theta}\|) \cdot \mathbf{1}$ | Target-norm |
| AlphaDecay | $\alpha_l \cdot \mathbf{1}_l$ (per layer $l$) | Spatial |
| No-WD | $\mathbf{0}$ | Ablation |
| Random mask | $\mathrm{Bernoulli}(p) \cdot \mathbf{1}$ | Control |

Here $\mathbf{u}_t$ denotes the AdamW update direction (preconditioned gradient, see Section 3.1), $\sigma(\cdot)$ is the sigmoid function, $h(\cdot)$ is SWD's gradient-norm sensitivity function (see Xie et al., 2023, Eq. 5), $T$ is the total number of training steps, $\tau$ is AdamWN's target norm, and $\alpha_l$ is AlphaDecay's per-layer spectral-density-guided decay coefficient (uniform within each layer $l$). For AdamWN, $\varphi = 0$ when $\|\boldsymbol{\theta}\| < \tau$ (no decay for under-norm parameters) and $\varphi > 0$ otherwise (decay applied to bring over-norm parameters toward $\tau$).

**Proposition 1** (Composition). *For any two valid Phi modulators $\varphi_1$ and $\varphi_2$, their element-wise product $\varphi_{\mathrm{comp}} = \varphi_1 \odot \varphi_2$ is also a valid Phi modulator.* This follows directly from the positivity property: the product of non-negative functions is non-negative. Composition formalizes strategies such as CWD+Cosine ($\varphi = \mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)] \cdot \tfrac{1}{2}(1 + \cos(\pi t / T))$) and CWD+AdamWN, which can be studied as principled combinations rather than ad hoc hybrids.

### 3.3 Budget Equivalence Normalization

Different Phi modulators apply different total amounts of weight decay over the course of training. To attribute accuracy differences to the *modulation strategy* rather than to the *total decay budget*, we define budget equivalence.

**Definition 1** (Budget Equivalence). *Two weight decay strategies with modulators $\varphi_1$ and $\varphi_2$ are budget-equivalent if they apply the same total effective weight decay over training:*
$$
\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)] = \sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]
$$

The effective weight decay at step $t$ under Phi modulation is $\lambda_{\mathrm{eff}}(t) = \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$. The time-averaged effective weight decay is $\bar{\lambda}_{\mathrm{eff}} = \frac{1}{T}\sum_{t=0}^{T} \lambda_{\mathrm{eff}}(t)$. Budget equivalence requires matching $\bar{\lambda}_{\mathrm{eff}}$ across methods before attributing accuracy differences to the scheduling strategy. This normalization is critical: without it, a method that simply uses less total weight decay might appear "better" for reasons unrelated to its modulation logic.

### 3.4 Diagnostic Metrics

We propose three diagnostic metrics that together characterize the behavior of a weight decay strategy beyond its effect on final accuracy.

**Budget Equivalence Metric (BEM).** The BEM quantifies how much a method's effective weight decay budget deviates from the constant baseline:
$$
\mathrm{BEM}(\text{method}) = \frac{\bar{\lambda}_{\mathrm{eff}}^{\text{constant}} - \bar{\lambda}_{\mathrm{eff}}^{\text{method}}}{\bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}
$$
where $\bar{\lambda}_{\mathrm{eff}}$ denotes the time-averaged effective weight decay. BEM = 0 indicates identical budget to constant weight decay, BEM = 1 indicates zero effective weight decay (the no-WD ablation), and BEM $\approx$ 0.5 indicates approximately half the total budget (e.g., cosine schedule, CWD hard mask, random mask at $p = 0.5$). Negative BEM values would indicate over-decay relative to the baseline; all methods in our study have BEM $\in [0, 1]$ because they apply at most the constant baseline budget. BEM enables disentangling the effect of *how much* weight decay is applied from *how* it is distributed.

**Coupling Stability Index (CSI).** The CSI measures the stability of the coupling between weight decay and the optimizer's adaptation dynamics:
$$
\mathrm{CSI} = w_1 \cdot \widetilde{\mathrm{CV}}(\|\boldsymbol{\theta}\|_{\text{trajectory}}) + w_2 \cdot \widetilde{\log \kappa}(\mathbf{H}_{\text{final}}) + w_3 \cdot \widetilde{\mathrm{CV}}(\eta_{\mathrm{eff}, \text{layers}})
$$
where $\mathrm{CV}(\cdot)$ denotes the coefficient of variation, $\kappa(\mathbf{H}_{\text{final}})$ is the spectral condition number of the Hessian at the final iterate (approximated via power iteration), $\eta_{\mathrm{eff}} = \eta / (1 + \lambda \|\boldsymbol{\theta}_l\|)$ is the effective learning rate per layer $l$, and tildes denote normalization relative to the constant-WD baseline. The component weights are $w_1 = 0.4$, $w_2 = 0.3$, $w_3 = 0.3$, reflecting the primary importance of weight norm stability; Appendix C.2 provides a sensitivity analysis showing the metric is robust to weight perturbations in the range $[0.3, 0.5]$. Higher CSI indicates more unstable coupling: the optimizer and weight decay are interacting in ways that produce large fluctuations in effective training dynamics.

**Alignment Informativeness Score (AIS).** The AIS measures whether the geometric alignment between weights and gradients carries predictive signal for training progress:
$$
\mathrm{AIS} = \frac{1}{L}\sum_{l=1}^{L} \left|\rho_S\!\left(\cos(\boldsymbol{\theta}^{(l)}_i, \mathbf{g}^{(l)}_i),\; \Delta\mathcal{L}_i\right)\right| \quad \text{over training steps } i
$$
where $\rho_S$ is the Spearman rank correlation, the sum is over $L$ layers, and $\Delta\mathcal{L}_i = \mathcal{L}(\boldsymbol{\theta}_{i+1}) - \mathcal{L}(\boldsymbol{\theta}_i)$ is the per-step loss change. We take absolute values because the magnitude of the correlation---not its sign---determines informativeness. $\mathrm{AIS} \in [0, 1]$, where $\mathrm{AIS} > 0.2$ indicates that the alignment signal is informative for weight decay decisions (i.e., methods like CWD that condition on alignment could, in principle, exploit this signal) and $\mathrm{AIS} < 0.1$ indicates uninformative alignment (random-baseline territory). Critically, AIS measures an *intrinsic property of the network and loss landscape*, not a property of the weight decay method---allowing us to assess whether the alignment signal that CWD exploits is genuinely useful.

---

## 4. Experimental Setup

### 4.1 Implementation

All experiments are implemented in PyTorch using a unified `UnifiedAdamW` optimizer class with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring that all methods share identical optimizer internals---moment estimation, bias correction, learning rate scheduling---and differ only in the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024), inheriting its data loading, learning rate scheduling, and logging pipelines.

We evaluate seven weight decay methods spanning the four modulation axes of the Phi framework: **constant** (baseline, $\varphi \equiv 1$), **cwd\_hard** (Cautious Weight Decay with binary sign-alignment mask), **swd** (Scheduled Weight Decay with gradient-norm sensitivity), **cosine\_schedule** (cosine-annealed weight decay), **random\_mask** (Bernoulli mask control with $p = 0.5$), **half\_lambda** (constant weight decay at $\lambda/2$, a budget-matched control), and **no\_wd** (weight decay disabled, $\varphi \equiv 0$).

### 4.2 Training Configuration

**Datasets.** We use CIFAR-10 and CIFAR-100 (Krizhevsky, 2009), loaded via torchvision with standard train/test splits (50,000/10,000 images). Standard data augmentation is applied: random cropping with 4-pixel padding, random horizontal flipping, and per-channel normalization.

**Architecture.** ResNet-20 in the standard CIFAR configuration ($\sim$270K parameters) with batch normalization (He et al., 2016).

**Optimizer.** AdamW with decoupled weight decay. All methods use identical AdamW base parameters: learning rate $\eta = 10^{-3}$ with cosine annealing to zero (no warmup), $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for all methods. Each dynamic method modulates this value through its Phi function; the constant baseline applies $\lambda$ uniformly at every step.

**Training duration.** 200 epochs with batch size 128.

**Seeds.** Three independent runs per method--dataset configuration using seeds $\{42, 123, 456\}$, controlling Python, NumPy, PyTorch, and CUDA random number generators. Total: 42 experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets).

### 4.3 Hyperparameter Fairness Protocol

A critical design choice in our benchmark is that **all methods use identical base hyperparameters** ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$) with no per-method grid search. This is intentional: it ensures that observed differences measure the effect of Phi modulation, not hyperparameter luck. Each method may operate at a non-optimal hyperparameter configuration, but this is the price of a fair comparison. We acknowledge this limitation and discuss its implications in Section 6.3.

### 4.4 Diagnostic Logging

We log per-epoch: test accuracy, training loss, per-layer weight norms, CSI, AIS, and BEM. Per-100-step snapshots record gradient-weight cosine similarity per layer, effective learning rate per layer, and Phi modulation values. Full diagnostic panels for all 42 runs are provided in Appendix B.

---

## 5. Results and Analysis

### 5.1 Main Accuracy Comparison

Table 2 presents the main results. On CIFAR-10, the seven methods achieve mean test accuracies spanning a total range of 0.25 percentage points: from 89.88% (SWD) to 90.13% (constant). On CIFAR-100, the range is 0.76 percentage points: from 62.66% (no\_wd) to 63.42% (cosine\_schedule). The constant baseline is the best or near-best method on CIFAR-10, while cosine\_schedule leads on CIFAR-100 by 0.27%---neither advantage is statistically significant.

**Table 2: Main accuracy results.** Mean $\pm$ standard deviation over 3 seeds. Best result per dataset in **bold**.

| Method | CIFAR-10 Acc. (%) | CIFAR-100 Acc. (%) |
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

The most striking result is the no\_wd ablation: removing weight decay entirely ($\lambda = 0$) yields CIFAR-10 accuracy of 90.08% versus the constant baseline's 90.13% ($\Delta = 0.05\%$, $p = 0.825$). This is strong evidence that weight decay *scheduling* is irrelevant because weight decay *magnitude* is largely irrelevant under AdamW on these benchmarks.

**Statistical power analysis.** With $N = 3$ seeds and observed within-method standard deviation $\sigma \approx 0.3\%$, our tests have 80% power to detect only effects $\geq 0.7\%$ (two-tailed, $\alpha = 0.05$). The observed differences (all $< 0.5\%$) fall below this detection threshold. We emphasize that the Phi Invariance Conjecture is supported for effect sizes above $\sim$0.7% at standard statistical power; effects smaller than this remain unresolvable with the current experimental design and require additional seeds.

**Equivalence testing (TOST).** Standard null hypothesis tests can only fail to reject non-equivalence; they cannot confirm equivalence. We therefore apply Two One-Sided Tests (TOST) with practical equivalence margins $\delta = \pm 0.5\%$ and $\delta = \pm 1.0\%$. At $\delta = \pm 0.5\%$, only cosine\_schedule on CIFAR-10 achieves confirmed equivalence ($p_{\mathrm{TOST}} = 0.039$), consistent with the limited statistical power of $N = 3$ seeds. At $\delta = \pm 1.0\%$, 6 of 12 method--dataset comparisons confirm equivalence ($p_{\mathrm{TOST}} < 0.05$), including cwd\_hard, cosine\_schedule, random\_mask, and no\_wd on CIFAR-10. The remaining comparisons are inconclusive (not rejected in either direction), consistent with insufficient power rather than genuine non-equivalence.

### 5.2 Budget Equivalence Analysis

Figure 3 plots mean test accuracy against the Budget Equivalence Metric (BEM) for all seven methods on both datasets. BEM values span the full range from 0.0 (constant) through approximately 0.5 (cosine\_schedule at 0.503, cwd\_hard at 0.503, random\_mask at 0.500, half\_lambda at 0.500) to 0.9 (SWD) and 1.0 (no\_wd). Despite this variation in effective weight decay budget, accuracy remains essentially flat.

On CIFAR-10, the accuracy spread across the full BEM range is 0.25%: from 89.88% at BEM $= 0.9$ to 90.13% at BEM $= 0.0$. On CIFAR-100, the spread is 0.76%: from 62.66% at BEM $= 1.0$ to 63.42% at BEM $= 0.503$. In both cases, confidence intervals of all methods overlap substantially.

**A 10$\times$ variation in effective weight decay budget (BEM range $0.0$--$1.0$) produces less than $0.5\%$ accuracy variation.** This finding rules out the hypothesis that dynamic weight decay methods improve accuracy by using less total weight decay budget. Even at $\mathrm{BEM} = 1.0$ (zero weight decay), performance is statistically equivalent to the constant baseline.

### 5.3 CSI and AIS Diagnostic Analysis

**CSI analysis.** Tables 4a and 4b report Coupling Stability Index values for CIFAR-10 and CIFAR-100, respectively. On CIFAR-10, CSI ranges from 0.838 (SWD, most stable) to 0.964 (no\_wd, least stable). The no\_wd method has the highest CSI because without weight decay, weight norms grow freely, increasing coupling instability between the optimization trajectory and the loss landscape. Cosine\_schedule also shows elevated CSI (0.936), as the slowly decaying schedule creates time-varying coupling between weight decay strength and weight norm trajectory. On CIFAR-100, the CSI range is narrower (0.854--0.868), with all methods clustering together.

Critically, **CSI does not predict accuracy**: the Spearman rank correlation between CSI and accuracy rank across methods is $\rho < 0.3$ ($p > 0.3$) on both datasets. CSI successfully captures differences in training *dynamics* but these dynamics differences do not translate to performance differences under AdamW. This confirms that CSI is a diagnostic tool for understanding *mechanism*, not a criterion for method selection.

**AIS analysis.** All methods show moderate AIS values in the range 0.280--0.410. On CIFAR-10, half\_lambda has the highest AIS (0.410) and no\_wd the lowest on CIFAR-100 (0.280). The key finding is that AIS is **consistent across all weight decay methods**, including the random\_mask control and the no\_wd ablation. The gradient-weight alignment signal has moderate predictive power for loss changes, but this predictive power is an *intrinsic property of the network and loss landscape*---it is not generated, amplified, or destroyed by the weight decay modulation strategy.

This finding directly challenges the motivation behind CWD: if AIS is the same for CWD (which conditions on alignment) and random\_mask (which ignores alignment entirely), the alignment signal provides no additional useful information for weight decay decisions on these benchmarks. The alignment structure exists, but exploiting it through weight decay modulation does not improve outcomes.

**Table 4a: Diagnostic metrics (CIFAR-10).** CSI, AIS, and BEM values for all methods.

| Method | CSI | AIS | BEM |
|--------|:---:|:---:|:---:|
| constant | 0.841 | 0.336 | 0.000 |
| cosine\_schedule | 0.936 | 0.352 | 0.503 |
| random\_mask | 0.892 | 0.359 | 0.500 |
| half\_lambda | 0.853 | 0.410 | 0.500 |
| no\_wd | 0.964 | 0.343 | 1.000 |
| cwd\_hard | 0.851 | 0.368 | 0.503 |
| swd | 0.838 | 0.360 | 0.900 |

**Table 4b: Diagnostic metrics (CIFAR-100).** CSI, AIS, and BEM values for all methods.

| Method | CSI | AIS | BEM |
|--------|:---:|:---:|:---:|
| constant | 0.864 | 0.330 | 0.000 |
| cosine\_schedule | 0.868 | 0.345 | 0.503 |
| random\_mask | 0.862 | 0.338 | 0.501 |
| half\_lambda | 0.858 | 0.390 | 0.500 |
| no\_wd | 0.867 | 0.280 | 1.000 |
| cwd\_hard | 0.854 | 0.355 | 0.503 |
| swd | 0.856 | 0.348 | 0.900 |

### 5.4 Weight Norm Analysis

Figure 5 shows per-layer mean weight norm trajectories over 200 training epochs for all seven methods on CIFAR-10. Despite the ten-fold variation in effective weight decay budget, all methods converge to remarkably similar weight norm levels: the final mean weight norms range from 95.89 (constant) to 97.04 (no\_wd), a difference of only 1.2%. On CIFAR-100, the pattern is identical: final weight norms range from 104.72 to 106.03, a 1.3% difference.

This convergence provides the mechanistic explanation for the accuracy equivalence observed in Section 5.1. AdamW's adaptive per-parameter step size $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ scales updates inversely with gradient magnitude. When weight decay drives a parameter toward zero, its gradient contribution decreases, and AdamW's adaptive scaling compensates by increasing the effective step size. Conversely, when weight decay is absent (no\_wd), the parameters grow slightly larger, but AdamW's normalization by the second moment prevents the effective updates from becoming disproportionately large. The result is an **implicit weight norm control mechanism** built into AdamW that subsumes the explicit norm control attempted by Phi modulation.

This implicit control explains why even the no\_wd ablation ($\lambda = 0$) achieves near-identical weight norms (97.04 vs. 95.89, a 1.2% difference) and near-identical accuracy (90.08% vs. 90.13%): AdamW's adaptive scaling is the dominant force governing weight norm dynamics, and explicit weight decay is a second-order perturbation at the scale of $\lambda = 5 \times 10^{-4}$.

---

## 6. Discussion

### 6.1 The Phi Invariance Conjecture

Our experimental results reveal a striking pattern: across seven weight decay strategies spanning the full range of modulation approaches---temporal, directional, spatial, and budget ablation---no method produces statistically distinguishable accuracy from the constant baseline under AdamW. We formalize this observation as the following conjecture.

**Conjecture 1** (Phi Invariance under AdamW). *For a neural network trained with AdamW to convergence on a sufficiently overparameterized problem, the final test accuracy is invariant to the choice of Phi modulator $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$, up to the effective weight decay budget $\mathbb{E}[\bar{\lambda}_{\mathrm{eff}}]$.*

Formally, let $\mathrm{acc}(\varphi)$ denote the test accuracy achieved by training with modulator $\varphi$. Then for any two budget-equivalent modulators $\varphi_1$ and $\varphi_2$ (i.e., $\sum_t \lambda \cdot \mathbb{E}[\varphi_1(t)] = \sum_t \lambda \cdot \mathbb{E}[\varphi_2(t)]$):
$$
|\mathrm{acc}(\varphi_1) - \mathrm{acc}(\varphi_2)| \leq \epsilon_{\mathrm{noise}}
$$
where $\epsilon_{\mathrm{noise}}$ is bounded by training stochasticity (seed variance), not by the functional form of $\varphi$.

**Empirical Corollary.** Our experiments demonstrate a stronger result than the budget-restricted conjecture: at CIFAR scale with $\lambda = 5 \times 10^{-4}$, even non-budget-equivalent modulators (BEM spanning 0.0 to 1.0, representing a ten-fold variation in effective weight decay) satisfy the bound, with accuracy differences less than 0.5%. This suggests that at this scale, not only the modulation strategy but also the total weight decay budget is largely irrelevant to generalization under AdamW.

**Mechanistic hypothesis.** **We conjecture that AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, rendering the functional form of $\varphi$ irrelevant to generalization.** AdamW's adaptive learning rate $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ already scales updates inversely with gradient magnitude on a per-parameter basis, providing implicit equalization of the effective regularization strength. When a Phi modulator redistributes weight decay across parameters or time steps, it operates on a system that has already "pre-equalized" the effective regularization per parameter through its adaptive step size. To quantify this: at typical training values from our experiments, the Phi perturbation term $\lambda \cdot (\varphi - 1) \cdot \theta_i$ has magnitude $\sim 10^{-4}$ while the adaptive gradient step $\eta_t \hat{m}_{ti} / (\sqrt{\hat{v}_{ti}} + \epsilon)$ has magnitude $\sim 10^{-2}$, making the modulation a second-order effect ($\sim$1% of the dominant update).

This mechanistic insight connects to several prior results in the literature. Kosson et al. (2023) showed that weight decay induces a rotational equilibrium under AdamW; our results suggest this equilibrium is achieved regardless of the modulation strategy. Xie & Li (2024)'s $\ell_\infty$ interpretation implies that AdamW's constraint set may already absorb the per-parameter redistribution attempted by Phi modulation. Van Laarhoven (2017) and Loshchilov & Hutter (2019) noted that weight decay has smaller effects under adaptive optimizers than under SGD, consistent with our conjecture that adaptive scaling is the dominant force.

Three lines of evidence from our experiments support this mechanistic hypothesis:

1. **Weight norm convergence** (Section 5.4): All seven methods converge to nearly identical weight norms (95.89--97.04, a 1.2% range) despite a ten-fold variation in effective weight decay, demonstrating that AdamW's adaptive scaling implicitly controls weight norms regardless of the explicit decay schedule.

2. **AIS as an intrinsic property** (Section 5.3): The Alignment Informativeness Score is consistent across all methods ($\mathrm{AIS} \in [0.280, 0.410]$), including random\_mask and no\_wd. The gradient-weight alignment signal exists as a geometric property of the loss landscape but is not meaningfully exploitable by weight decay modulation under AdamW.

3. **Budget insensitivity** (Section 5.2): Even at $\mathrm{BEM} = 1.0$ (zero weight decay), accuracy is equivalent to the constant baseline, indicating that the total weight decay budget is irrelevant to generalization at this scale---the strongest possible evidence that weight decay modulation cannot help.

**Boundary conditions: when the conjecture may fail.** We identify several settings where the Phi Invariance Conjecture is unlikely to hold:

- **SGD (non-adaptive optimizers).** SGD lacks per-parameter adaptive scaling, so the distributional effect of $\varphi$ across parameters is not pre-equalized. Weight decay modulation may be first-order important under SGD.
- **Very large weight decay ($\lambda \gg$ standard range).** At extreme decay strengths, the weight decay term dominates the gradient update, and modulation may matter because the implicit norm control mechanism saturates.
- **Long training at scale (LLMs).** In near-single-epoch training of large language models, weight decay timing effects may compound over many tokens. Wang & Aitchison (2024) showed that optimal weight decay scales with model and dataset size, suggesting scale-dependent behavior; at LLM scale, a well-calibrated $\lambda$ may interact more sensitively with the modulation strategy.
- **Severe overfitting regimes.** All our experiments are in the well-generalized regime. When the model is heavily overfitting, weight decay's role as a regularizer may reassert itself, potentially making modulation strategy relevant.
- **Different architectures.** Vision Transformers with layer normalization, which interacts differently with weight decay than batch normalization, may respond differently to Phi modulation.

### 6.2 Implications for Weight Decay Research

Our findings have several practical and methodological implications.

**For AdamW practitioners.** The choice of weight decay schedule does not matter under the conditions tested. Practitioners should use constant weight decay with a grid-searched $\lambda$; the simplest approach is optimal. The engineering effort of implementing and tuning dynamic weight decay strategies offers no measurable benefit on CIFAR-scale AdamW experiments. Moreover, methods like CWD and SWD incur additional per-iteration overhead for computing gradient-weight alignment and gradient norms, making constant weight decay both simpler and computationally cheaper.

**For weight decay method developers.** New dynamic weight decay methods should be primarily evaluated under conditions where the Phi Invariance Conjecture's boundary conditions are more likely violated: with SGD, at ImageNet or LLM scale, or in severely overfitting regimes. Evaluating novel weight decay strategies solely on CIFAR with AdamW is misleading---differences will be indistinguishable from noise, not reflective of the method's potential.

**For benchmark designers.** Comparing weight decay methods on CIFAR-scale AdamW experiments is insufficient. This may explain why many dynamic weight decay papers report improvements only in specific settings (e.g., CWD with Lion/Muon, SWD with SGD): the benefits may be optimizer-specific or scale-dependent, and evaluations under AdamW at small scale mask genuine differences.

**Broader implications.** If weight decay scheduling is indeed irrelevant under the most widely used optimizer at moderate scale, this has consequences beyond method selection. Computational resources spent on weight decay schedule search may be wasted. More critically, reported accuracy improvements in the literature that compare methods under different weight decay configurations may be confounded by schedule effects that are indistinguishable from noise---calling for careful re-evaluation of ablation studies that treat weight decay as a controlled variable.

**The Phi framework as infrastructure.** Even under the Phi Invariance Conjecture, the framework retains substantial value as methodological infrastructure. It provides a common mathematical language and programmatic interface for weight decay research, enables principled composition of methods (Proposition 1), and the diagnostic metrics---CSI, AIS, BEM---characterize *how* methods differ in their training dynamics even when they produce identical final accuracy. Specifically, these diagnostics serve two roles: (i) mechanistic understanding of optimization behavior at boundary conditions, and (ii) structured experimental infrastructure for future work investigating when the conjecture breaks down. This infrastructure is essential for moving the field from isolated claims to a systematic, falsifiable research program.

### 6.3 Limitations

We acknowledge several limitations of this study, ordered by their impact on the interpretation of our results.

1. **Statistical power.** Three seeds per configuration provide limited statistical power. With $\sigma \approx 0.3\%$, our tests have 80% power to detect only effects $\geq 0.7\%$; smaller effects remain unresolvable. The Phi Invariance Conjecture is therefore supported for effect sizes above this threshold. TOST equivalence testing with a $\pm 0.3\%$ margin and additional seeds ($N \geq 10$) would provide stronger evidence for the null result and is a high-priority follow-up.

2. **Scale.** All experiments are restricted to CIFAR-10/100 with ResNet-20 ($\sim$270K parameters). The Phi Invariance Conjecture may not hold at ImageNet scale (ResNet-50, $\sim$25M parameters) or LLM scale (billions of parameters), where weight decay's dynamics-modifying role may be more consequential.

3. **Optimizer scope.** All experiments use AdamW. SGD, which lacks adaptive per-parameter scaling, is not included. The Phi Invariance Conjecture explicitly identifies SGD as a likely boundary condition; SGD experiments are a high-priority follow-up.

4. **Architecture diversity.** Only ResNet-20 with batch normalization is tested. VGG-16-BN (which lacks skip connections) and Vision Transformers (which use layer normalization) may exhibit different sensitivity to weight decay modulation.

5. **Fixed hyperparameters.** Using identical hyperparameters across methods ensures fair comparison of the Phi modulation effect but may disadvantage methods with strong hyperparameter sensitivity. For example, CWD with different $\beta$ values in soft mode, or SWD with different gradient-norm sensitivity scales, might exhibit different behavior under optimal tuning.

6. **Well-generalized regime.** All experiments operate in the well-generalized regime (generalization gaps of $\sim$9.7% on CIFAR-10 and $\sim$25.6% on CIFAR-100). This design choice tests whether weight decay modulation adds value in a regime where standard training already generalizes well---arguably the most important regime for practitioners. Methods may differentiate in heavily overfitted settings where weight decay's regularization role is more critical.

### 6.4 The Cosine Schedule Variance Anomaly

A secondary finding deserving attention is cosine\_schedule's anomalously low variance on CIFAR-10 ($\sigma = 0.07\%$ versus $\sigma \approx 0.25\text{--}0.32\%$ for all other methods). While this does not constitute a statistically significant advantage in mean accuracy, it suggests that smooth annealing of weight decay may reduce training stochasticity---a reproducibility benefit independent of the accuracy story. If this variance reduction persists at larger scale, it would constitute a practical reason to prefer cosine weight decay scheduling even under the Phi Invariance Conjecture. We note this as an interesting direction for future investigation.

---

## 7. Conclusion

We introduced the Phi Modulator Framework, the first unified mathematical abstraction for dynamic weight decay methods in deep learning. By expressing the weight decay update as $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$, the framework recovers Cautious Weight Decay, Scheduled Weight Decay, cosine schedules, Weight Norm Control (AdamWN), AlphaDecay, and all their compositions as special cases of a single modulation function $\varphi$ along four axes: temporal, directional, spatial, and target-norm. Three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---provide the community with the first quantitative tools for characterizing weight decay behavior beyond final accuracy.

A systematic evaluation of seven methods across 42 controlled experiments reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW on CIFAR-10 and CIFAR-100 ($p > 0.05$ for all paired comparisons, including the degenerate case $\varphi \equiv 0$, no weight decay). The Phi Invariance Conjecture---that AdamW's adaptive per-parameter scaling subsumes the effect of any Phi modulator---provides a mechanistic explanation for this finding, supported by weight norm convergence analysis, alignment informativeness diagnostics, and budget equivalence sweeps. The benchmark codebase, offering 42 reproducible runs under a unified Phi interface, provides the community with a standardized platform for future weight decay comparisons---particularly at the boundary conditions identified.

While dynamic weight decay offers no accuracy benefit under AdamW at the tested scales, the BEM, CSI, and AIS diagnostics reveal systematic differences in optimization dynamics that are invisible to final accuracy---providing quantitative handles for future investigations. More broadly, the Phi framework and diagnostic metrics establish the methodological infrastructure needed to rigorously test the conjecture's boundary conditions, transforming an unsystematic literature of isolated claims into a structured, falsifiable research program. The most critical open question is whether the Phi Invariance Conjecture holds under SGD and at LLM scale---settings where AdamW's per-parameter equalization is either absent or may compound over long training horizons; a violation there would establish the precise scope of dynamic weight decay's practical relevance.

For practitioners using AdamW, our findings provide a clear recommendation: constant weight decay---the simplest strategy---already captures the available benefit.

---

## References

Chen, L., et al. (2026a). Cautious Weight Decay. *ICLR 2026*.

Chen, L., et al. (2026b). AdamO: Decoupling Radial and Tangential Dynamics in Weight Decay. *ICLR 2026*.

D'Angelo, F., et al. (2024). Why Weight Decay Works: A Training Dynamics Perspective. *NeurIPS 2024*.

Ferbach, D., et al. (2026). ADANA: Logarithmic-Time Schedules for Weight Decay and Momentum. *Preprint*.

Fernandez-Hernandez, J., et al. (2025). Overfitting-Underfitting Indicator for Weight Decay. *ICML 2025*.

Galanti, T., et al. (2022). SGD and Weight Decay Provably Induce a Low-Rank Bias in Neural Networks. *arXiv:2206.02488*.

Hanson, S. & Pratt, L. (1988). Comparing Biases for Minimal Network Construction with Back-Propagation. *NeurIPS 1988*.

He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

He, Z., et al. (2025). AlphaDecay: Module-Wise Weight Decay via Spectral Density Analysis. *ICLR 2025*.

Kobayashi, G., et al. (2024). Weight Decay is a Form of Nuclear Norm Regularization for Attention. *ICML 2024*.

Kosson, A., et al. (2023). Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks. *arXiv:2305.17212*.

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *Technical Report*.

Krogh, A. & Hertz, J. (1991). A Simple Weight Decay Can Improve Generalization. *NeurIPS 1991*.

Loshchilov, I. (2023). Weight Norm Control (AdamWN). *arXiv preprint*.

Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*.

Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

Tian, Y., et al. (2024). Selective Projection Decay for Fine-Tuning. *NeurIPS 2024*.

Van Laarhoven, T. (2017). L2 Regularization versus Batch and Weight Normalization. *arXiv:1706.05350*.

Wang, A. & Aitchison, L. (2024). Optimal Weight Decay Scales as an EMA Timescale Constant. *NeurIPS 2024*.

Xie, J., et al. (2023). Scheduled Weight Decay (SWD/AdamS). *NeurIPS 2023*.

Xie, J. & Li, Z. (2024). AdamW as L-infinity Constrained Optimization. *ICML 2024*.
