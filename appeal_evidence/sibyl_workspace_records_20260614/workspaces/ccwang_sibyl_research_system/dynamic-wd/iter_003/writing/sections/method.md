# 3. The Phi Modulator Framework

We introduce the Phi Modulator Framework, a unified mathematical abstraction that subsumes all major dynamic weight decay strategies as special cases. The framework consists of three components: (i) the Phi modulator definition, which generalizes the weight decay update rule; (ii) a taxonomy that recovers existing methods as special cases along four modulation axes; and (iii) three diagnostic metrics---BEM, CSI, and AIS---that provide standardized tools for characterizing weight decay behavior.

## 3.1 Formal Definition

Consider a neural network with parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ trained with AdamW. The standard AdamW update rule at step $t$ is:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \boldsymbol{\theta}_t
$$
where $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are the bias-corrected first and second moment estimates, $\eta_t$ is the learning rate, $\epsilon$ is a stability constant, and $\lambda$ is the weight decay coefficient. We generalize this by introducing the **Phi modulator** $\varphi$:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t
$$
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function and $\odot$ denotes element-wise multiplication. The modulator $\varphi$ can depend on the training step $t$, the current parameters $\boldsymbol{\theta}_t$, the gradient $\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$, and---through the optimizer state---the moment estimates and any accumulated statistics.

The Phi modulator satisfies three key properties:

- **Positivity:** $\varphi(t, \boldsymbol{\theta}, \mathbf{g}) \geq 0$ component-wise. Weight decay is never reversed into weight growth; the modulator can only reduce or maintain the decay applied to each parameter.
- **Measurability:** $\varphi$ can depend on any combination of training step, parameters, gradients, and optimizer state. This generality allows the framework to capture methods that condition on gradient norms, weight-update alignment, per-layer statistics, or external schedules.
- **Normalization convention:** For budget-equivalent comparison, we adopt the convention $\mathbb{E}[\varphi] = 1$, meaning the average modulation across parameters and time steps equals unity. Deviations from this convention are quantified by the Budget Equivalence Metric (Section 3.4).

The framework admits a clean programmatic interface:

```python
class WDModulator(ABC):
    def compute_phi(self, w: Tensor, u: Tensor, t: int) -> Tensor:
        """Return per-parameter modulation phi in [0, inf)."""
        ...
```

where `w` is the parameter tensor, `u` is the optimizer update direction, and `t` is the training step. Every weight decay strategy in our study is implemented as a subclass of this interface, ensuring identical integration with the AdamW base optimizer.

## 3.2 Special Cases: Recovering Existing Methods

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
| AlphaDecay | $\mathrm{diag}(\boldsymbol{\alpha}_l) \cdot \mathbf{1}$ (layer $l$) | Spatial |
| No-WD | $\mathbf{0}$ | Ablation |
| Random mask | $\mathrm{Bernoulli}(p) \cdot \mathbf{1}$ | Control |

Here $\mathbf{u}_t$ denotes the optimizer update direction, $\sigma(\cdot)$ is the sigmoid function, $h(\cdot)$ is SWD's gradient-norm sensitivity function, $T$ is the total number of training steps, $\tau$ is AdamWN's target norm, and $\boldsymbol{\alpha}_l$ is AlphaDecay's per-layer spectral-density-guided decay coefficient.

**Proposition 1** (Composition). *For any two valid Phi modulators $\varphi_1$ and $\varphi_2$, their element-wise product $\varphi_{\mathrm{comp}} = \varphi_1 \odot \varphi_2$ is also a valid Phi modulator.* This follows directly from the positivity property: the product of non-negative functions is non-negative. Composition formalizes strategies such as CWD+Cosine ($\varphi = \mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)] \cdot \tfrac{1}{2}(1 + \cos(\pi t / T))$) and CWD+AdamWN, which can be studied as principled combinations rather than ad hoc hybrids.

## 3.3 Budget Equivalence Normalization

Different Phi modulators apply different total amounts of weight decay over the course of training. To attribute accuracy differences to the *modulation strategy* rather than to the *total decay budget*, we define budget equivalence.

**Definition 1** (Budget Equivalence). *Two weight decay strategies with modulators $\varphi_1$ and $\varphi_2$ are budget-equivalent if they apply the same total effective weight decay over training:*
$$
\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)] = \sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]
$$

The effective weight decay at step $t$ under Phi modulation is $\lambda_{\mathrm{eff}}(t) = \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$. Budget equivalence requires matching $\mathbb{E}[\lambda_{\mathrm{eff}}]$ across methods before attributing accuracy differences to the scheduling strategy. This normalization is critical: without it, a method that simply uses less total weight decay might appear "better" for reasons unrelated to its modulation logic.

## 3.4 Diagnostic Metrics

We propose three diagnostic metrics that together characterize the behavior of a weight decay strategy beyond its effect on final accuracy.

**Budget Equivalence Metric (BEM).** The BEM quantifies how much a method's effective weight decay budget deviates from the constant baseline:
$$
\mathrm{BEM}(\text{method}) = \frac{|\lambda_{\mathrm{eff}}^{\text{method}} - \lambda_{\mathrm{eff}}^{\text{constant}}|}{\lambda_{\mathrm{eff}}^{\text{constant}}}
$$
BEM is normalized to $[0, 1]$, where $\mathrm{BEM} = 0$ indicates identical budget to constant weight decay and $\mathrm{BEM} = 1$ indicates zero effective weight decay (the no-WD ablation). A method with $\mathrm{BEM} \approx 0.5$ uses approximately half the total weight decay budget (e.g., cosine schedule, CWD hard mask, random mask at $p = 0.5$). BEM enables disentangling the effect of *how much* weight decay is applied from *how* it is distributed.

**Coupling Stability Index (CSI).** The CSI measures the stability of the coupling between weight decay and the optimizer's adaptation dynamics:
$$
\mathrm{CSI} = w_1 \cdot \mathrm{CV}(\|\boldsymbol{\theta}\|_{\text{trajectory}}) + w_2 \cdot \log \kappa(\mathbf{H}_{\text{final}}) + w_3 \cdot \mathrm{CV}(\eta_{\mathrm{eff}, \text{layers}})
$$
where $\mathrm{CV}(\cdot)$ denotes the coefficient of variation, $\kappa(\mathbf{H}_{\text{final}})$ is the spectral condition number of the Hessian at the final iterate (approximated via power iteration), and $\eta_{\mathrm{eff}} = \eta / (1 + \lambda \|\boldsymbol{\theta}_l\|)$ is the effective learning rate per layer $l$. The component weights are $w_1 = 0.4$, $w_2 = 0.3$, $w_3 = 0.3$, reflecting the primary importance of weight norm stability. Higher CSI indicates more unstable coupling: the optimizer and weight decay are interacting in ways that produce large fluctuations in effective training dynamics.

**Alignment Informativeness Score (AIS).** The AIS measures whether the geometric alignment between weights and gradients carries predictive signal for training progress:
$$
\mathrm{AIS} = \rho_S\!\left(\cos(\boldsymbol{\theta}_i, \mathbf{g}_i),\; \Delta\mathcal{L}_i\right) \quad \text{over training steps } i
$$
where $\rho_S$ is the Spearman rank correlation. AIS is computed per layer and averaged across layers. $\mathrm{AIS} \in [0, 1]$, where $\mathrm{AIS} > 0.2$ indicates that the alignment signal is informative for weight decay decisions (i.e., methods like CWD that condition on alignment could, in principle, exploit this signal) and $\mathrm{AIS} < 0.1$ indicates uninformative alignment (random-baseline territory). Critically, AIS measures an *intrinsic property of the network and loss landscape*, not a property of the weight decay method---allowing us to assess whether the alignment signal that CWD exploits is genuinely useful.
