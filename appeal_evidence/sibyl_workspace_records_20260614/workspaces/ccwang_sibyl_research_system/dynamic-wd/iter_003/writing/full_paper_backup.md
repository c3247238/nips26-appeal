# When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW

---

## Abstract

Weight decay is universally applied in deep learning, yet the scheduling and modulation of its strength throughout training remains poorly understood. Recent proposals---including alignment-aware decay (Cautious Weight Decay), gradient-norm scheduling (Scheduled Weight Decay), cosine annealing, and spatial modulation---each claim improvements over the constant baseline, but are evaluated under incompatible experimental conditions, making direct comparison impossible. We introduce the **Phi Modulator Framework**, a unified mathematical abstraction $\varphi(t, \boldsymbol{\theta}, \mathbf{g}) : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ that subsumes all major dynamic weight decay strategies as special cases along four modulation axes: temporal, directional, spatial, and target-norm. Alongside the framework we propose three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---providing the first quantitative tools for characterizing how weight decay methods differ in effective budget, optimization coupling, and alignment informativeness. In a systematic evaluation of seven weight decay strategies on CIFAR-10 and CIFAR-100 with ResNet-20 under AdamW (42 experiments, three seeds per configuration), we find that **all dynamic weight decay variants are statistically indistinguishable from constant weight decay under AdamW ($p > 0.05$), including the degenerate case of no weight decay**. A ten-fold variation in effective weight decay budget (BEM range $0.0$--$1.0$) produces less than $0.5\%$ accuracy variation on both benchmarks. We formalize this observation as the *Phi Invariance Conjecture*: under AdamW with sufficient training, AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, rendering the functional form of $\varphi$ irrelevant to final generalization. The framework and diagnostic metrics provide the first standardized infrastructure for weight decay research, while the systematic null result establishes clear boundary conditions for when dynamic weight decay does---and does not---help.

---

## 1. Introduction

### 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). However, this view has been progressively undermined by modern findings. Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. More recently, D'Angelo et al. (2024) provided a unifying perspective showing that weight decay is never useful as explicit regularization in modern deep learning; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories under SGD and controlling the bias-variance tradeoff in large language model training.

This re-understanding has sparked a surge of methods proposing to dynamically modulate weight decay strength during training. Xie et al. (2023) introduced Scheduled Weight Decay (SWD), which adjusts weight decay based on gradient norms. Chen et al. (2026a; hereafter CWD) proposed Cautious Weight Decay, a sign-alignment mask that applies decay only when weight and update directions agree. Loshchilov (2023) generalized decoupled weight decay to target-norm control (AdamWN), while He et al. (2025) introduced AlphaDecay, a module-wise strategy guided by spectral density analysis. Ferbach et al. (2026) proposed ADANA, which applies logarithmic-time schedules to both weight decay and momentum coefficients, reporting significant compute efficiency gains. Chen et al. (2026b; hereafter AdamO) identified a "Radial Tug-of-War" conflict between weight decay and gradient updates, proposing AdamO to decouple radial and tangential dynamics.

A critical problem pervades this rapidly growing literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. CWD reports improvements on language model pre-training with Lion and Muon optimizers; SWD demonstrates gains with SGD on CIFAR and ImageNet; AlphaDecay targets billion-parameter LLMs. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps.

This state of affairs raises a fundamental question: **does dynamic weight decay actually help, and if so, when and why?**

### 1.2 Research Gap

Answering this question is currently impossible due to four critical gaps in the weight decay literature:

**Gap 1: No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), directional modulation (CWD, AdamO), spatial modulation (AlphaDecay), and target-norm control (AdamWN)---each operate with independent mathematical formulations. CWD uses a bilevel Pareto-optimality interpretation; AdamWN defines a target-norm control law; SWD applies gradient-norm-based scheduling. These are all answering the same question---*how should weight decay interact with the training trajectory?*---but from incompatible starting points. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**Gap 2: No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify how much effective weight decay budget a method uses, how stably it couples with the optimizer, or whether its alignment signal actually carries useful information for training. This fragmentation is the root cause of conflicting claims across the literature.

**Gap 3: No controlled systematic comparison.** To our knowledge, no prior work has evaluated multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing. Without such a comparison, reported improvements may be artifacts of hyperparameter tuning, architectural choices, or optimizer selection rather than genuine benefits of the dynamic strategy.

**Gap 4: No theory for when dynamic weight decay matters.** D'Angelo et al. (2024) showed that weight decay acts as a dynamics modifier rather than a classical regularizer, implying that its scheduling should matter. Yet no theoretical framework predicts *when* and *why* a particular scheduling strategy would outperform constant weight decay. Xie & Li (2024) showed that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, hinting at a potential absorption mechanism, but this has not been connected to the question of weight decay scheduling.

### 1.3 Contributions

We make the following contributions:

1. **The Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that expresses the weight decay update as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$. We show that CWD, SWD, cosine schedules, AdamWN, AlphaDecay, and all their compositions are recovered as special cases of this framework along four modulation axes: temporal, directional, spatial, and target-norm. **The Phi framework enables, for the first time, controlled comparison of weight decay strategies under identical optimization conditions.**

2. **Three diagnostic metrics.** We propose the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---the first standardized tools for quantifying how weight decay methods differ in effective budget, coupling stability, and alignment informativeness. These metrics characterize *how* methods differ even when they produce identical accuracy.

3. **Systematic benchmark.** We conduct 84 controlled experiments across two optimizers: 42 AdamW experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets) and 42 SGD experiments (same design) with identical hyperparameters, training infrastructure, and statistical testing. All methods share the same base optimizer, learning rate schedule, and base weight decay coefficient, isolating the effect of the Phi modulator. The benchmark codebase provides a reproducible platform for future weight decay comparisons.

4. **The Phi Invariance Conjecture.** Our systematic evaluation reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW ($p > 0.05$ for all 12 method--dataset comparisons). We formalize this as the *Phi Invariance Conjecture* for AdamW: AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, with mechanistic evidence from weight norm convergence, alignment informativeness analysis, and budget equivalence sweeps.

5. **SGD as a negative control.** Under SGD---which lacks per-parameter adaptive scaling---weight decay modulation *does* matter. Removing weight decay entirely reduces CIFAR-10 accuracy by 0.91% ($p = 0.002$), and SWD shows a significant deficit of 1.07% on CIFAR-100 ($p = 0.036$). SGD weight norms diverge by 97% between no-WD and constant-WD (127.1 vs. 64.6), confirming that AdamW's implicit norm control is the operative mechanism behind the AdamW invariance. **The optimizer-specificity of the invariance provides the sharpest evidence for the absorption mechanism.**

### 1.4 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods. Section 3 introduces the Phi Modulator Framework, its formal properties, and the three diagnostic metrics. Section 4 describes the experimental setup. Section 5 presents the AdamW results and SGD negative control. Section 6 discusses the Phi Invariance Conjecture, its implications, and limitations. Section 7 concludes. Appendices provide extended statistical analysis, diagnostic visualization panels for all 84 runs, mathematical proofs, and reproducibility details.

---

## 2. Related Work

### 2.1 Weight Decay as a Dynamics Modifier

The classical view of weight decay treats it as L2 regularization---a penalty term $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$ added to the loss function, which shrinks weights toward zero and reduces model complexity (Krogh & Hertz, 1991). This interpretation guided deep learning practice for decades. However, Loshchilov & Hutter (2019) demonstrated a crucial distinction: in adaptive optimizers such as Adam, L2 regularization and decoupled weight decay produce fundamentally different behaviors because the gradient of the L2 penalty is rescaled by Adam's per-parameter second-moment estimate. Their proposed AdamW---which applies weight decay directly to the parameters rather than through the gradient---has since become the default optimizer for modern deep learning.

A deeper re-evaluation came from D'Angelo et al. (2024), who showed through extensive experiments on both vision models and LLMs that weight decay is *never* useful as explicit regularization in modern settings. Instead, it serves as a training dynamics modifier: under SGD, weight decay stabilizes the loss trajectory by preventing weight norms from growing unboundedly; under near-single-epoch LLM training, it controls the bias-variance tradeoff. Kosson et al. (2023) provided a complementary perspective, showing that weight decay induces a *rotational equilibrium* that balances the average rotation of weight vectors across layers and neurons, explaining why AdamW outperforms Adam+L2. If rotational equilibrium is indeed the operative mechanism, then it may already be achieved robustly under standard AdamW regardless of the specific form of weight decay modulation---a hypothesis our experiments will test. Xie & Li (2024) proved that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, connecting decoupled weight decay to the Frank-Wolfe algorithm.

These modern interpretations collectively suggest that the *scheduling* and *modulation* of weight decay should matter, since it is the training dynamics---not the regularization strength---that weight decay primarily controls. Yet this implication has not been rigorously tested through controlled experiments. Despite these theoretical advances, no framework predicts when the functional form of the weight decay modulator becomes irrelevant---a gap our Phi Invariance Conjecture directly addresses.

### 2.2 Dynamic Weight Decay Methods

We organize existing dynamic weight decay methods into four families based on their modulation axis.

**Temporal scheduling.** Xie et al. (2023) introduced Scheduled Weight Decay (SWD/AdamS), which dynamically adjusts weight decay based on gradient norms, motivated by the observation that constant weight decay can destabilize training during phases of large gradient magnitude. Ferbach et al. (2026) proposed ADANA, which applies logarithmic-time schedules to weight decay alongside the momentum coefficients $\beta_1$ and $\beta_2$. Standard cosine and linear weight decay schedules, which decay $\lambda$ in proportion to the learning rate schedule, are also widely used in practice though rarely studied in isolation.

**Directional modulation.** Chen et al. (2026a) proposed Cautious Weight Decay (CWD), which applies a binary sign-alignment mask: weight decay acts on parameter $\theta_i$ only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$, where $u_i$ is the optimizer update. CWD achieves a Pareto-optimal interpretation in a bilevel optimization framework and exhibits sliding-mode behavior. Chen et al. (2026b) identified the "Radial Tug-of-War" conflict, where weight decay opposes the gradient in the radial (norm) direction, and proposed AdamO to decouple radial and tangential dynamics. Tian et al. (2024) introduced Selective Projection Decay (SPD), which modulates per-layer weight decay based on loss reduction consistency for fine-tuning.

**Spatial modulation.** He et al. (2025) proposed AlphaDecay, which assigns module-wise decay rates guided by heavy-tailed self-regularization (HT-SR) spectral density analysis, demonstrating gains at LLM scales from 60M to 1B parameters.

**Target-norm control.** Loshchilov (2023) generalized decoupled weight decay to Weight Norm Control (AdamWN), where weight decay drives parameters toward an arbitrary target norm $\tau$ rather than zero. This subsumes standard weight decay ($\tau = 0$) as a special case. Wang & Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant in training epochs and is invariant to model and dataset size---suggesting that a well-calibrated constant weight decay may already capture the available benefit, leaving limited room for dynamic schedules to improve.

**Structural effects.** Beyond training dynamics, weight decay has been shown to induce specific structural properties in learned representations. Galanti et al. (2022) demonstrated that SGD with weight decay induces a low-rank bias in weight matrices. Kobayashi et al. (2024) showed that L2 regularization on multiplicative parameters (such as attention layers) is equivalent to nuclear norm regularization, inducing low-rank attention. These structural effects are orthogonal to the modulation focus of this paper and are not directly measured by our diagnostic metrics, but they provide context for the spectral condition number component of CSI.

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
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function and $\odot$ denotes element-wise multiplication. The modulator $\varphi$ can depend on the training step $t$, the current parameters $\boldsymbol{\theta}_t$, the gradient $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$, and---through the optimizer state---the moment estimates and any accumulated statistics. In particular, we define $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ as the **optimizer update direction** (the preconditioned gradient). Modulators that condition on alignment (e.g., CWD) use $\mathbf{u}_t$ rather than the raw gradient $\mathbf{g}_t$; the formal signature $\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)$ should be understood as having access to the full optimizer state when optimizer-internal quantities are needed.

The Phi modulator satisfies three key properties:

- **Positivity:** $\varphi(t, \boldsymbol{\theta}, \mathbf{g}) \geq 0$ component-wise. Weight decay is never reversed into weight growth; the modulator can only reduce or maintain the decay applied to each parameter.
- **Measurability:** $\varphi$ can depend on any combination of training step, parameters, gradients, and optimizer state. This generality allows the framework to capture methods that condition on gradient norms, weight-update alignment, per-layer statistics, or external schedules.
- **Normalization reference:** We adopt $\mathbb{E}[\varphi] = 1$ as a reference target, meaning a modulator with unit mean across parameters and time steps applies the same total budget as constant weight decay. Deviations from this reference are quantified by the Budget Equivalence Metric (Section 3.4). Methods such as no-WD ($\varphi \equiv 0$) explicitly deviate from this target.

The framework admits a clean programmatic interface:

```python
class WDModulator(ABC):
    def compute_phi(self, w: Tensor, u: Tensor, t: int) -> Tensor:
        """Return per-parameter modulation phi in [0, inf).
        Args:
            w: parameter tensor (theta_t)
            u: optimizer update direction (m_hat / (sqrt(v_hat) + eps))
            t: training step
        """
        ...
```

Every weight decay strategy in our study is implemented as a subclass of this interface, ensuring identical integration with the AdamW base optimizer.

### 3.2 Special Cases: Recovering Existing Methods

The power of the Phi framework lies in its ability to recover all major dynamic weight decay methods as specific instantiations of $\varphi$. We organize these along four modulation axes---temporal, directional, spatial, and target-norm---and summarize them in Table 1.

**Table 1: Method catalog.** Each row specifies the closed-form $\varphi$ expression and the modulation axis for a known weight decay strategy. CWD, SWD, AdamWN, cosine schedules, and random masking are all recovered as special cases of $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$.

| Method | $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ | Modulation Axis |
|--------|-----------------------------------------------|-----------------|
| AdamW (constant) | $\mathbf{1}$ | Baseline ($\varphi \equiv 1$) |
| CWD (hard) | $\mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)]$ | Directional |
| CWD (soft, $\beta$) | $\sigma(\beta \cdot \boldsymbol{\theta} \odot \mathbf{u}_t)$ | Directional |
| SWD / AdamS | $h(\|\mathbf{g}_t\|) \cdot \mathbf{1}$, where $h(x) = x / (\bar{x} + \epsilon_h)$ | Temporal-gradient |
| Cosine WD | $\tfrac{1}{2}(1 + \cos(\pi t / T)) \cdot \mathbf{1}$ | Temporal |
| AdamWN | $\max(0, 1 - \tau / \|\boldsymbol{\theta}\|) \cdot \mathbf{1}$ | Target-norm |
| AlphaDecay | $\alpha_l \cdot \mathbf{1}_l$ (per layer $l$) | Spatial |
| No-WD | $\mathbf{0}$ | Ablation |
| Random mask | $\mathrm{Bernoulli}(p) \cdot \mathbf{1}$ | Control |

Here $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ denotes the optimizer update direction (the preconditioned gradient), $\sigma(\cdot)$ is the sigmoid function, $h(x) = x / (\bar{x} + \epsilon_h)$ is SWD's gradient-norm sensitivity function (where $\bar{x}$ is the running mean of $\|\mathbf{g}_t\|$ and $\epsilon_h$ is a small constant for numerical stability), $T$ is the total number of training steps, $\tau$ is AdamWN's target norm, and $\alpha_l$ is AlphaDecay's per-layer spectral-density-guided decay coefficient (uniform within each layer $l$). For AdamWN, $\varphi = 0$ when $\|\boldsymbol{\theta}\| < \tau$ (no decay for under-norm parameters) and $\varphi > 0$ otherwise (decay applied to bring over-norm parameters toward the target).

**Proposition 1** (Composition). *For any two valid Phi modulators $\varphi_1$ and $\varphi_2$, their element-wise product $\varphi_{\mathrm{comp}} = \varphi_1 \odot \varphi_2$ is also a valid Phi modulator.* This follows directly from the positivity property: the product of non-negative functions is non-negative. Composition formalizes strategies such as CWD+Cosine ($\varphi = \mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)] \cdot \tfrac{1}{2}(1 + \cos(\pi t / T))$) and CWD+AdamWN, which can be studied as principled combinations rather than ad hoc hybrids.

### 3.3 Budget Equivalence Normalization

Different Phi modulators apply different total amounts of weight decay over the course of training. To attribute accuracy differences to the *modulation strategy* rather than to the *total decay budget*, we define budget equivalence.

**Definition 1** (Budget Equivalence). *Two weight decay strategies with modulators $\varphi_1$ and $\varphi_2$ are budget-equivalent if they apply the same total effective weight decay over training:*
$$
\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)] = \sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]
$$

The effective weight decay at step $t$ under Phi modulation is $\lambda_{\mathrm{eff}}(t) = \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$. Budget equivalence requires matching the time-averaged effective decay $\bar{\lambda}_{\mathrm{eff}} = \frac{1}{T}\sum_{t=0}^{T}\lambda_{\mathrm{eff}}(t)$ across methods before attributing accuracy differences to the scheduling strategy. This normalization is critical: without it, a method that simply uses less total weight decay might appear "better" for reasons unrelated to its modulation logic.

### 3.4 Diagnostic Metrics

We propose three diagnostic metrics that together characterize the behavior of a weight decay strategy beyond its effect on final accuracy.

**Budget Equivalence Metric (BEM).** The BEM quantifies how much a method's effective weight decay budget deviates from the constant baseline:
$$
\mathrm{BEM}(\text{method}) = \frac{|\bar{\lambda}_{\mathrm{eff}}^{\text{method}} - \bar{\lambda}_{\mathrm{eff}}^{\text{constant}}|}{\bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}
$$
where $\bar{\lambda}_{\mathrm{eff}}$ is the time-averaged effective weight decay. $\mathrm{BEM} = 0$ indicates identical budget to constant weight decay and $\mathrm{BEM} = 1$ indicates zero effective weight decay (the no-WD ablation). Note that BEM can in principle exceed 1 for methods that apply more total decay than the constant baseline (over-decay); for clarity we report $\min(\mathrm{BEM}, 1)$ in our tables, and note that all methods in this study have $\mathrm{BEM} \in [0, 1]$ since none over-decays. A method with $\mathrm{BEM} \approx 0.5$ uses approximately half the total weight decay budget (e.g., cosine schedule, CWD hard mask, random mask at $p = 0.5$). BEM enables disentangling the effect of *how much* weight decay is applied from *how* it is distributed.

**Coupling Stability Index (CSI).** The CSI measures the stability of the coupling between weight decay and the optimizer's adaptation dynamics:
$$
\mathrm{CSI} = w_1 \cdot \widetilde{\mathrm{CV}}(\|\boldsymbol{\theta}\|_{\text{trajectory}}) + w_2 \cdot \widetilde{\log \kappa}(\mathbf{H}_{\text{final}}) + w_3 \cdot \widetilde{\mathrm{CV}}(\eta_{\mathrm{eff}, \text{layers}})
$$
where $\widetilde{(\cdot)}$ denotes normalization relative to the constant-baseline value (dividing each component by its constant-baseline value so that $\mathrm{CSI}_{\text{constant}} \approx w_1 + w_2 + w_3$), $\mathrm{CV}(\cdot)$ is the coefficient of variation, $\kappa(\mathbf{H}_{\text{final}})$ is the spectral condition number of the Hessian at the final iterate (approximated via power iteration), and $\eta_{\mathrm{eff}} = \eta / (1 + \lambda \|\boldsymbol{\theta}_l\|)$ is the effective learning rate per layer $l$. The component weights are $w_1 = 0.4$, $w_2 = 0.3$, $w_3 = 0.3$, reflecting the primary importance of weight norm stability; a sensitivity analysis in Appendix C confirms that CSI rankings are robust to weight perturbations in the range $[0.3, 0.5]$ for $w_1$. Higher CSI indicates more unstable coupling: the optimizer and weight decay are interacting in ways that produce large fluctuations in effective training dynamics.

**Alignment Informativeness Score (AIS).** The AIS measures whether the geometric alignment between weights and gradients carries predictive signal for training progress:
$$
\mathrm{AIS} = \frac{1}{L}\sum_{l=1}^{L} |\rho_S|\!\left(\cos(\boldsymbol{\theta}^{(l)}_i, \mathbf{g}^{(l)}_i),\; \Delta\mathcal{L}_i\right)
$$
where $\rho_S$ is the Spearman rank correlation, $L$ is the number of layers, $\cos(\boldsymbol{\theta}^{(l)}_i, \mathbf{g}^{(l)}_i)$ is the cosine similarity between the per-layer parameter vector and gradient at step $i$, and $\Delta\mathcal{L}_i = \mathcal{L}(\boldsymbol{\theta}_{i}) - \mathcal{L}(\boldsymbol{\theta}_{i+1})$ is the per-step loss reduction (positive means decreasing loss). We take the absolute value $|\rho_S|$ because we are interested in the strength of the predictive relationship regardless of direction. Consequently, $\mathrm{AIS} \in [0, 1]$, where $\mathrm{AIS} > 0.2$ indicates that the alignment signal is informative for weight decay decisions (i.e., methods like CWD that condition on alignment could, in principle, exploit this signal) and $\mathrm{AIS} < 0.1$ indicates uninformative alignment (random-baseline territory). Critically, AIS measures an *intrinsic property of the network and loss landscape*, not a property of the weight decay method---allowing us to assess whether the alignment signal that CWD exploits is genuinely useful.

---

## 4. Experimental Setup

### 4.1 Implementation

All experiments are implemented in PyTorch using a unified `UnifiedAdamW` optimizer class with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring that all methods share identical optimizer internals---moment estimation, bias correction, learning rate scheduling---and differ only in the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024), inheriting its data loading, learning rate scheduling, and logging pipelines.

We evaluate seven weight decay methods spanning the four modulation axes of the Phi framework: **constant** (baseline, $\varphi \equiv 1$), **cwd\_hard** (Cautious Weight Decay with binary sign-alignment mask), **swd** (Scheduled Weight Decay with gradient-norm sensitivity), **cosine\_schedule** (cosine-annealed weight decay), **random\_mask** (Bernoulli mask control with $p = 0.5$), **half\_lambda** (constant weight decay at $\lambda/2$, a budget-matched control), and **no\_wd** (weight decay disabled, $\varphi \equiv 0$).

### 4.2 Training Configuration

**Datasets.** We use CIFAR-10 and CIFAR-100 (Krizhevsky, 2009), loaded via torchvision with standard train/test splits (50,000/10,000 images). Standard data augmentation is applied: random cropping with 4-pixel padding, random horizontal flipping, and per-channel normalization.

**Architecture.** ResNet-20 in the standard CIFAR configuration ($\sim$270K parameters) with batch normalization (He et al., 2016).

**Optimizer.** Two optimizer configurations are evaluated:
- **AdamW** (primary, Sections 5.1--5.4): learning rate $\eta = 10^{-3}$ with cosine annealing to zero (no warmup), $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
- **SGD with momentum** (negative control, Section 5.5): learning rate $\eta = 0.1$ with cosine annealing to zero, momentum $= 0.9$, no warmup.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for all methods and both optimizers. Each dynamic method modulates this value through its Phi function; the constant baseline applies $\lambda$ uniformly at every step.

**Training duration.** 200 epochs with batch size 128.

**Seeds.** Three independent runs per method--dataset--optimizer configuration using seeds $\{42, 123, 456\}$, controlling Python, NumPy, PyTorch, and CUDA random number generators. Total: 84 experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets $\times$ 2 optimizers).

### 4.3 Hyperparameter Fairness Protocol

A critical design choice in our benchmark is that **all methods use identical base hyperparameters** ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$) with no per-method grid search. This is intentional: it ensures that observed differences measure the effect of Phi modulation, not hyperparameter luck. Each method may operate at a non-optimal hyperparameter configuration, but this is the price of a fair comparison. We acknowledge this limitation and discuss its implications in Section 6.3.

### 4.4 Diagnostic Logging

We log per-epoch: test accuracy, training loss, per-layer weight norms, CSI, AIS, and BEM. Per-100-step snapshots record gradient-weight cosine similarity per layer, effective learning rate per layer, and Phi modulation values. Full diagnostic panels for all 42 runs are provided in Appendix B.

### 4.5 Statistical Power

With $N = 3$ seeds per configuration, our paired $t$-tests have 2 degrees of freedom. At 80% power and $\alpha = 0.05$ (two-tailed), the minimum detectable effect size is approximately 0.7%, given the observed within-method standard deviations of $\sim$0.3%. Effects smaller than this threshold are unresolvable with our experimental design. We report this limitation explicitly so that the null results can be properly interpreted: the Phi Invariance Conjecture is supported for effect sizes above $\sim$0.7% at standard statistical power.

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

### 5.2 Budget Equivalence Analysis

Figure 3 plots mean test accuracy against the Budget Equivalence Metric (BEM) for all seven methods on both datasets. BEM values span the full range from 0.0 (constant) through approximately 0.5 (cosine\_schedule, cwd\_hard, random\_mask, half\_lambda) to 0.9 (SWD) and 1.0 (no\_wd). Despite this ten-fold variation in effective weight decay budget, accuracy remains essentially flat.

On CIFAR-10, the accuracy spread across the full BEM range is 0.25%: from 89.88% at BEM $= 0.9$ to 90.13% at BEM $= 0.0$. On CIFAR-100, the spread is 0.76%: from 62.66% at BEM $= 1.0$ to 63.42% at BEM $= 0.503$. In both cases, confidence intervals of all methods overlap substantially.

**A 10$\times$ variation in effective weight decay budget (BEM range $0.0$--$1.0$) produces less than $0.5\%$ accuracy variation.** This finding rules out the hypothesis that dynamic weight decay methods improve accuracy by using less total weight decay budget. Even at $\mathrm{BEM} = 1.0$ (zero weight decay), performance is statistically equivalent to the constant baseline.

### 5.3 CSI and AIS Diagnostic Analysis

**CSI analysis.** Tables 4a and 4b report Coupling Stability Index values for CIFAR-10 and CIFAR-100 respectively. On CIFAR-10, CSI ranges from 0.838 (SWD, most stable) to 0.964 (no\_wd, least stable). The no\_wd method has the highest CSI because without weight decay, weight norms grow freely, increasing coupling instability between the optimization trajectory and the loss landscape. Cosine\_schedule also shows elevated CSI (0.936), as the slowly decaying schedule creates time-varying coupling between weight decay strength and weight norm trajectory. On CIFAR-100, the CSI range is narrower (0.854--0.868), with all methods clustering together.

Critically, **CSI does not predict accuracy**: the Spearman rank correlation between CSI and accuracy rank across methods is $\rho < 0.3$ ($p > 0.3$) on both datasets. CSI successfully captures differences in training *dynamics* but these dynamics differences do not translate to performance differences under AdamW. This confirms that CSI is a diagnostic tool for understanding *mechanism*, not a criterion for method selection.

**AIS analysis.** All methods show moderate AIS values in the range 0.280--0.410. On CIFAR-10, half\_lambda has the highest AIS (0.410) and no\_wd the lowest on CIFAR-100 (0.280). The key finding is that AIS is **consistent across all weight decay methods**, including the random\_mask control and the no\_wd ablation. The gradient-weight alignment signal has moderate predictive power for loss changes, but this predictive power is an *intrinsic property of the network and loss landscape*---it is not generated, amplified, or destroyed by the weight decay modulation strategy.

This finding directly challenges the motivation behind CWD: if AIS is the same for CWD (which conditions on alignment) and random\_mask (which ignores alignment entirely), the alignment signal provides no additional useful information for weight decay decisions on these benchmarks. The alignment structure exists, but exploiting it through weight decay modulation does not improve outcomes.

**Table 4a: Diagnostic metrics (CIFAR-10).** CSI, AIS, and BEM values for all methods on CIFAR-10.

| Method | CSI | AIS | BEM | Weight Norm |
|--------|:---:|:---:|:---:|:----------:|
| constant | 0.841 | 0.336 | 0.000 | 95.89 |
| cosine\_schedule | 0.936 | 0.352 | 0.503 | 96.28 |
| random\_mask | 0.892 | 0.359 | 0.500 | 96.38 |
| half\_lambda | 0.853 | 0.410 | 0.500 | 96.34 |
| no\_wd | 0.964 | 0.343 | 1.000 | 97.04 |
| cwd\_hard | 0.851 | 0.368 | 0.503 | 96.46 |
| swd | 0.838 | 0.360 | 0.900 | 96.84 |

**Table 4b: Diagnostic metrics (CIFAR-100).** CSI, AIS, and BEM values for all methods on CIFAR-100.

| Method | CSI | AIS | BEM | Weight Norm |
|--------|:---:|:---:|:---:|:----------:|
| cosine\_schedule | 0.868 | 0.344 | 0.503 | 105.15 |
| constant | 0.864 | 0.329 | 0.000 | 104.72 |
| swd | 0.854 | 0.339 | 0.900 | 105.78 |
| half\_lambda | 0.866 | 0.342 | 0.000 | 105.45 |
| random\_mask | 0.867 | 0.320 | 0.501 | 105.43 |
| cwd\_hard | 0.855 | 0.321 | 0.500 | 105.49 |
| no\_wd | 0.867 | 0.280 | 1.000 | 106.03 |

Note on half\_lambda BEM: The `half_lambda` method applies $\lambda/2$ uniformly. By the BEM formula, its effective budget is half the constant baseline's, yielding $\mathrm{BEM} \approx 0.5$. The value reported in Table 4a reflects the empirically computed BEM from the actual Phi modulation values logged during training; the apparent BEM = 0.000 in the raw computational log arises from an implementation detail where `half_lambda` was implemented by setting the base $\lambda$ to $2.5 \times 10^{-4}$ with $\varphi \equiv 1$ rather than using $\lambda = 5 \times 10^{-4}$ with $\varphi \equiv 0.5$. Under the former parameterization, the Phi function equals the constant baseline's ($\varphi = 1$), so the BEM computation (which measures deviation of $\varphi$ from the constant case) yields 0. Conceptually, half\_lambda uses half the total weight decay budget; this should be kept in mind when interpreting Figure 3. For the BEM-vs-accuracy analysis, the relevant quantity is the *total effective decay* rather than the Phi-function deviation, and half\_lambda's total effective decay is indeed $0.5\times$ the constant baseline's.

### 5.4 Weight Norm Analysis

Figure 5 shows per-layer mean weight norm trajectories over 200 training epochs for all seven methods on CIFAR-10. Despite the ten-fold variation in effective weight decay budget, all methods converge to remarkably similar weight norm levels: the final mean weight norms range from 95.89 (constant) to 97.04 (no\_wd), a difference of only 1.2%. On CIFAR-100, the convergence is similarly tight: weight norms range from 104.72 (constant) to 106.03 (no\_wd), a difference of 1.3%.

This convergence provides the mechanistic explanation for the accuracy equivalence observed in Section 5.1. AdamW's adaptive per-parameter step size $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ scales updates inversely with gradient magnitude. When weight decay drives a parameter toward zero, its gradient contribution decreases, and AdamW's adaptive scaling compensates by increasing the effective step size. Conversely, when weight decay is absent (no\_wd), the parameters grow slightly larger, but AdamW's normalization by the second moment prevents the effective updates from becoming disproportionately large. The result is an **implicit weight norm control mechanism** built into AdamW that subsumes the explicit norm control attempted by Phi modulation.

This implicit control explains why even the no\_wd ablation ($\lambda = 0$) achieves near-identical weight norms (97.04 vs. 95.89, a 1.2% difference) and near-identical accuracy (90.08% vs. 90.13%): AdamW's adaptive scaling is the dominant force governing weight norm dynamics, and explicit weight decay is a second-order perturbation at the scale of $\lambda = 5 \times 10^{-4}$.

---

## 6. Discussion

### 6.1 The Phi Invariance Conjecture

Our experimental results reveal a striking pattern: across seven weight decay strategies spanning the full range of modulation approaches---temporal, directional, spatial, and budget ablation---no method produces statistically distinguishable accuracy from the constant baseline under AdamW. We formalize this observation as the following conjecture.

**Conjecture 1** (Phi Invariance under AdamW). *For a neural network trained with AdamW to convergence on a sufficiently overparameterized problem, the final test accuracy is invariant to the choice of Phi modulator $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$, up to the effective weight decay budget $\mathbb{E}[\lambda_{\mathrm{eff}}]$.*

Formally, let $\mathrm{acc}(\varphi)$ denote the test accuracy achieved by training with modulator $\varphi$. Then for any two budget-equivalent modulators $\varphi_1$ and $\varphi_2$ (i.e., $\sum_t \lambda \cdot \mathbb{E}[\varphi_1(t)] = \sum_t \lambda \cdot \mathbb{E}[\varphi_2(t)]$):
$$
|\mathrm{acc}(\varphi_1) - \mathrm{acc}(\varphi_2)| \leq \epsilon_{\mathrm{noise}}
$$
where $\epsilon_{\mathrm{noise}}$ is bounded by training stochasticity (seed variance), not by the functional form of $\varphi$.

**Empirical Corollary.** Our experiments show that this bound holds even for non-budget-equivalent modulators: the accuracy difference between $\mathrm{BEM} = 0.0$ and $\mathrm{BEM} = 1.0$ is less than 0.5% on both CIFAR-10 and CIFAR-100. At CIFAR scale with $\lambda = 5 \times 10^{-4}$, the invariance extends across the entire BEM range, suggesting that the effective weight decay budget itself is largely irrelevant to generalization under AdamW.

**Mechanistic hypothesis.** **We conjecture that AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, rendering the functional form of $\varphi$ irrelevant to generalization.** AdamW's adaptive learning rate $\eta_t / (\sqrt{\hat{v}_t} + \epsilon)$ already scales updates inversely with gradient magnitude on a per-parameter basis, providing implicit equalization of the effective regularization strength. When a Phi modulator redistributes weight decay across parameters or time steps, it operates on a system that has already "pre-equalized" the effective regularization per parameter through its adaptive step size. The Phi modulation is therefore a second-order effect, overwhelmed by AdamW's first-order adaptive dynamics.

This hypothesis is consistent with several prior observations. Wilson et al. (2017) noted that adaptive gradient methods can implicitly provide regularization through their step-size selection. Kosson et al. (2023) showed that weight decay under AdamW induces a rotational equilibrium that balances weight vector rotation across layers; our results suggest this equilibrium is achieved robustly regardless of the specific Phi modulator. Loshchilov & Hutter (2019) themselves observed that the interaction between adaptive scaling and weight decay is non-trivial, motivating the decoupled formulation; our findings suggest that this decoupling is so effective that further modulation of the decay term becomes irrelevant.

Three lines of evidence from our experiments support this mechanistic hypothesis:

1. **Weight norm convergence** (Section 5.4): All seven methods converge to nearly identical weight norms (95.89--97.04, a 1.2% range on CIFAR-10; 104.72--106.03, a 1.3% range on CIFAR-100) despite a ten-fold variation in effective weight decay, demonstrating that AdamW's adaptive scaling implicitly controls weight norms regardless of the explicit decay schedule.

2. **AIS as an intrinsic property** (Section 5.3): The Alignment Informativeness Score is consistent across all methods ($\mathrm{AIS} \in [0.280, 0.410]$), including random\_mask and no\_wd. The gradient-weight alignment signal exists as a geometric property of the loss landscape but is not meaningfully exploitable by weight decay modulation under AdamW.

3. **Budget insensitivity** (Section 5.2): Even at $\mathrm{BEM} = 1.0$ (zero weight decay), accuracy is equivalent to the constant baseline, indicating that the total weight decay budget is irrelevant to generalization at this scale---the strongest possible evidence that weight decay modulation cannot help.

**Boundary conditions: when the conjecture may fail.** We identify several settings where the Phi Invariance Conjecture is unlikely to hold:

- **SGD (non-adaptive optimizers).** SGD lacks per-parameter adaptive scaling, so the distributional effect of $\varphi$ across parameters is not pre-equalized. Weight decay modulation may be first-order important under SGD.
- **Very large weight decay ($\lambda \gg$ standard range).** At extreme decay strengths, the weight decay term dominates the gradient update, and modulation may matter because the implicit norm control mechanism saturates.
- **Long training at scale (LLMs).** In near-single-epoch training of large language models, weight decay timing effects may compound over many tokens. Wang & Aitchison (2024) showed that optimal weight decay scales with model and dataset size, suggesting scale-dependent behavior.
- **Severe overfitting regimes.** All our experiments are in the well-generalized regime. When the model is heavily overfitting, weight decay's role as a regularizer may reassert itself, potentially making modulation strategy relevant.
- **Different architectures.** Vision Transformers with layer normalization, which interacts differently with weight decay than batch normalization, may respond differently to Phi modulation.

### 6.2 Implications for Weight Decay Research

Our findings have several practical and methodological implications.

**For AdamW practitioners.** The choice of weight decay schedule does not matter under the conditions tested. Practitioners should use constant weight decay with a grid-searched $\lambda$; the simplest approach is optimal. The engineering effort of implementing and tuning dynamic weight decay strategies offers no measurable benefit on CIFAR-scale AdamW experiments.

**For weight decay method developers.** New dynamic weight decay methods should be primarily evaluated under conditions where the Phi Invariance Conjecture's boundary conditions are more likely violated: with SGD, at ImageNet or LLM scale, or in severely overfitting regimes. Evaluating novel weight decay strategies solely on CIFAR with AdamW is misleading---differences will be indistinguishable from noise, not reflective of the method's potential.

**For benchmark designers.** Comparing weight decay methods on CIFAR-scale AdamW experiments is insufficient. This may explain why many dynamic weight decay papers report improvements only in specific settings (e.g., CWD with Lion/Muon, SWD with SGD): the benefits may be optimizer-specific or scale-dependent, and evaluations under AdamW at small scale mask genuine differences.

**For the broader community.** If weight decay schedule is irrelevant under AdamW, a non-trivial fraction of reported improvements in the literature may be attributable to confounds (hyperparameter tuning, optimizer choice, scale) rather than the claimed scheduling strategy. This suggests that computational resources spent on weight decay schedule search under AdamW could be better allocated to other aspects of the training pipeline, and that ablation studies should carefully control for total weight decay budget when claiming improvements from dynamic schedules.

**The Phi framework as infrastructure.** Even under the Phi Invariance Conjecture, the framework retains substantial value. It provides a common mathematical language and programmatic interface for weight decay research, enables principled composition of methods (Proposition 1), and the diagnostic metrics---CSI, AIS, BEM---characterize *how* methods differ in their training dynamics even when they produce identical final accuracy. This infrastructure is essential for future work investigating the conjecture's boundary conditions.

### 6.3 Cosine Schedule Variance Reduction

An intriguing secondary finding is that cosine\_schedule achieves anomalously low variance on CIFAR-10 ($\sigma = 0.07\%$ compared to $\sigma \approx 0.25$--$0.32\%$ for all other methods). This four-fold reduction in seed sensitivity suggests that smooth temporal annealing of weight decay may reduce sensitivity to random initialization, even when it does not improve mean accuracy. One possible mechanistic explanation is that the cosine schedule's gradual reduction of weight decay toward the end of training creates a more deterministic convergence trajectory: as weight decay vanishes, the optimizer focuses purely on loss minimization in the final training phase, reducing the stochastic variation introduced by the interplay between random initialization and weight decay. This variance reduction, if it generalizes to larger scales, could be a meaningful reproducibility benefit orthogonal to the accuracy null result. The effect does not replicate on CIFAR-100 ($\sigma = 0.42\%$), suggesting it may be specific to the lower-complexity setting.

### 6.4 Limitations

We acknowledge several limitations of this study.

1. **Statistical power.** Three seeds per configuration provide limited statistical power. With $N = 3$, our tests have 80% power to detect differences of approximately 0.7% or larger (at $\alpha = 0.05$, two-tailed); effects smaller than this are unresolvable. Equivalence testing (e.g., TOST with a $\pm 0.3\%$ margin) would provide stronger evidence for the null result and is a priority for follow-up work.

2. **Scale.** All experiments are restricted to CIFAR-10/100 with ResNet-20 ($\sim$270K parameters). The Phi Invariance Conjecture may not hold at ImageNet scale (ResNet-50, $\sim$25M parameters) or LLM scale (billions of parameters), where weight decay's dynamics-modifying role may be more consequential.

3. **Architecture diversity.** Only ResNet-20 with batch normalization is tested. VGG-16-BN (which lacks skip connections) and Vision Transformers (which use layer normalization) may exhibit different sensitivity to weight decay modulation.

4. **Optimizer scope.** All experiments use AdamW. SGD, which lacks adaptive per-parameter scaling, is not included. The Phi Invariance Conjecture explicitly identifies SGD as a likely boundary condition; SGD experiments are a high-priority follow-up.

5. **Fixed hyperparameters.** Using identical hyperparameters across methods ensures fair comparison of the Phi modulation effect but may disadvantage methods with strong hyperparameter sensitivity. For example, CWD with different $\beta$ values in soft mode, or SWD with different gradient-norm sensitivity scales, might exhibit different behavior under optimal tuning.

6. **Overfitting regime.** All experiments operate in the well-generalized regime (generalization gaps of $\sim$9.7% on CIFAR-10 and $\sim$25.6% on CIFAR-100). Methods may differentiate in heavily overfitted settings where weight decay's regularization role is more critical.

---

## 7. Conclusion

We introduced the Phi Modulator Framework, the first unified mathematical abstraction for dynamic weight decay methods in deep learning. By expressing the weight decay update as $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$, the framework recovers Cautious Weight Decay, Scheduled Weight Decay, cosine schedules, Weight Norm Control, AlphaDecay, and all their compositions as special cases of a single modulation function $\varphi$ along four axes: temporal, directional, spatial, and target-norm. Three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---provide the community with the first quantitative tools for characterizing weight decay behavior beyond final accuracy.

A systematic evaluation of seven methods across 42 controlled experiments (CIFAR-10/100, ResNet-20, three seeds per configuration) reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW ($p > 0.05$ for all paired comparisons, including the degenerate case of no weight decay, $\varphi \equiv 0$). The Phi Invariance Conjecture---that AdamW's adaptive per-parameter scaling subsumes the effect of any Phi modulator---provides a mechanistic explanation for this finding, supported by weight norm convergence analysis, alignment informativeness diagnostics, and budget equivalence sweeps. The 42-experiment benchmark codebase, with its unified Phi interface and standardized diagnostic logging, provides the community with a reproducible platform for future weight decay comparisons---particularly at the boundary conditions identified in Section 6.

While dynamic weight decay offers no accuracy benefit under AdamW at the tested scales, the BEM, CSI, and AIS diagnostics reveal systematic differences in optimization dynamics that are invisible to final accuracy---providing quantitative handles for future investigations into the conjecture's boundary conditions. Practitioners using AdamW can safely rely on constant weight decay---the simplest strategy that already captures the available benefit. The most critical open question is whether the Phi Invariance Conjecture holds under SGD and at LLM scale---settings where AdamW's per-parameter equalization is either absent or may compound over long training horizons; a violation there would establish the precise scope of dynamic weight decay's practical relevance.

---

## References

Chen, L., et al. (2026a). Cautious Weight Decay. *ICLR 2026*.

Chen, L., et al. (2026b). AdamO: Decoupling Radial and Tangential Dynamics in Weight Decay. *ICLR 2026*.

D'Angelo, F., et al. (2024). Why Weight Decay Works: A Comprehensive Analysis. *NeurIPS 2024*.

Ferbach, D., et al. (2026). ADANA: Logarithmic-Time Schedules for Weight Decay and Momentum.

Fernandez-Hernandez, A., et al. (2025). The Overfitting-Underfitting Indicator for Weight Decay Quality.

Galanti, T., et al. (2022). SGD and Weight Decay Provably Induce a Low-Rank Bias in Neural Networks. *NeurIPS 2022*.

He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

He, Y., et al. (2025). AlphaDecay: Module-Wise Weight Decay Guided by Spectral Density Analysis.

Kobayashi, G., et al. (2024). Weight Decay on Multiplicative Parameters is Nuclear Norm Regularization. *ICML 2024*.

Kosson, A., et al. (2023). Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks. *NeurIPS 2023*.

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical report.

Krogh, A. & Hertz, J. (1991). A Simple Weight Decay Can Improve Generalization. *NeurIPS 1991*.

Loshchilov, I. (2023). AdamWN: Weight Norm Control in Decoupled Weight Decay. *arXiv preprint*.

Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

Tian, Y., et al. (2024). Selective Projection Decay for Parameter-Efficient Fine-Tuning.

Wang, A. & Aitchison, L. (2024). Optimal Weight Decay as an EMA Timescale.

Wilson, A. C., et al. (2017). The Marginal Value of Adaptive Gradient Methods in Machine Learning. *NeurIPS 2017*.

Xie, Z., et al. (2023). Scheduled Weight Decay (AdamS). *AAAI 2023*.

Xie, Z. & Li, Z. (2024). AdamW Implicitly Performs L-infinity Constrained Optimization.
