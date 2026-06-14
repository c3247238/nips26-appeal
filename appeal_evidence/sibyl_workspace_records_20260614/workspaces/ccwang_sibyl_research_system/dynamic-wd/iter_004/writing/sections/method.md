## 3. The Phi Modulator Framework

We introduce the Phi Modulator Framework, a unified mathematical abstraction that subsumes all major dynamic weight decay strategies as special cases.

### 3.1 Formal Definition

Consider a neural network with parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ trained with AdamW. The standard update rule at step $t$ is:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \boldsymbol{\theta}_t
$$
where $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are bias-corrected first and second moment estimates, $\eta_t$ is the learning rate, $\epsilon$ is a stability constant, and $\lambda$ is the weight decay coefficient. We generalize this by introducing the **Phi modulator** $\varphi$:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t
$$
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function and $\odot$ denotes element-wise multiplication. We denote the AdamW preconditioned update as $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$; modulators conditioning on weight-update alignment (e.g., CWD) use $\mathbf{u}_t$ rather than raw gradient $\mathbf{g}_t$.

The Phi modulator satisfies three key properties:

- **Positivity:** $\varphi(t, \boldsymbol{\theta}, \mathbf{g}) \geq 0$ component-wise. Weight decay is never reversed into weight growth.
- **Measurability:** $\varphi$ may depend on any combination of training step, parameters, gradients, and optimizer state, enabling conditioning on gradient norms, alignment signals, per-layer statistics, or external schedules.
- **Reference target:** $\mathbb{E}[\varphi] = 1$, meaning the expected modulation across parameters and time equals unity. Deviations from this target are quantified by the Budget Equivalence Metric (Section 3.4).

### 3.2 Method Catalog: Recovering Existing Methods

The power of the Phi framework lies in its ability to recover all major dynamic weight decay methods as specific instantiations of $\varphi$. We organize these along four modulation axes in **Table 1**.

**Table 1: Method catalog.** Each row specifies the closed-form $\varphi$ expression and the modulation axis.

| Method | $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ | Modulation Axis |
|--------|-----------------------------------------------|-----------------|
| Constant (baseline) | $\mathbf{1}$ | --- |
| CWD (hard) | $\mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)]$ | Directional |
| SWD / AdamS | $h(\|\mathbf{g}_t\|) \cdot \mathbf{1}$ | Temporal-gradient |
| Cosine schedule | $\tfrac{1}{2}(1 + \cos(\pi t / T)) \cdot \mathbf{1}$ | Temporal |
| AdamWN | $\max(0, 1 - \tau / \|\boldsymbol{\theta}\|) \cdot \mathbf{1}$ | Target-norm |
| AlphaDecay | $\alpha_l \cdot \mathbf{1}_l$ (per layer $l$) | Spatial |
| No-WD | $\mathbf{0}$ | Ablation |
| Random mask | $\mathrm{Bernoulli}(p) \cdot \mathbf{1}$ | Control |
| Half-lambda | $0.5 \cdot \mathbf{1}$ | Budget control |

Here $\mathbf{u}_t$ is the preconditioned update, $h(\cdot)$ is SWD's gradient-norm sensitivity function (Xie et al., 2023), $T$ is the total training steps, $\tau$ is AdamWN's target norm, and $\alpha_l$ is AlphaDecay's per-layer spectral-density-guided coefficient.

**Proposition 1** (Composition). *For any two valid Phi modulators $\varphi_1$ and $\varphi_2$, their element-wise product $\varphi_{\mathrm{comp}} = \varphi_1 \odot \varphi_2$ is also a valid Phi modulator.* This follows directly from the positivity property and formalizes combined strategies (e.g., CWD+Cosine) as principled compositions rather than ad hoc hybrids.

### 3.3 Budget Equivalence Normalization

Different modulators apply different total amounts of weight decay. To attribute accuracy differences to the *modulation strategy* rather than the *total decay budget*, we define budget equivalence.

**Definition 1** (Budget Equivalence). *Two weight decay strategies with modulators $\varphi_1$ and $\varphi_2$ are budget-equivalent if they apply the same total effective weight decay:*
$$
\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)] = \sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]
$$

The effective weight decay at step $t$ is $\lambda_{\mathrm{eff}}(t) = \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$. Budget equivalence normalization is critical: without it, a method that simply applies less total weight decay might appear ``better'' for reasons unrelated to its modulation logic.

### 3.4 Diagnostic Metrics

We propose three metrics that characterize weight decay behavior beyond final accuracy. These are **descriptive characterization tools**---they quantify *how* methods differ operationally, not *which* method performs better.

**Budget Equivalence Metric (BEM).** BEM quantifies how much a method's effective weight decay budget deviates from the constant baseline:
$$
\mathrm{BEM}(\text{method}) = \frac{\bar{\lambda}_{\mathrm{eff}}^{\text{method}} - \bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}{\bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}
$$
where $\bar{\lambda}_{\mathrm{eff}}$ is the time-averaged effective weight decay. BEM $= 0$ indicates identical budget to constant weight decay; BEM $= -0.5$ indicates half the budget (e.g., half\_lambda); BEM $= -1$ indicates zero effective weight decay (no\_wd). Positive BEM indicates over-decay relative to baseline. Verified values: half\_lambda BEM $= -0.500$, cosine\_schedule BEM $\approx -0.600$, no\_wd BEM $= -1.000$, constant BEM $= 0.000$.

**Coupling Stability Index (CSI).** CSI measures the stability of the coupling between weight decay and the optimizer's adaptation dynamics:
$$
\mathrm{CSI}_{\mathrm{rel}} = \frac{\mathrm{CSI}_{\mathrm{raw}}}{\mathrm{CSI}_{\mathrm{constant}}}
$$
where $\mathrm{CSI}_{\mathrm{raw}}$ combines three normalized components: the coefficient of variation of weight norm trajectory, the log spectral condition number, and the coefficient of variation of per-layer effective learning rates. By construction, $\mathrm{CSI}_{\mathrm{rel}}(\text{constant}) = 1.0$. Values greater than 1 indicate less stable coupling than the constant baseline.

**Alignment Informativeness Score (AIS).** AIS measures whether the geometric alignment between weights and gradients carries informative signal. For each layer $l$, we compute $a_l = |\cos(\boldsymbol{\theta}^{(l)}, \mathbf{g}^{(l)})|$, bin the distribution into 10 equal-width bins over $[0, 1]$, and compute the normalized entropy:
$$
\mathrm{AIS} = \frac{1}{L}\sum_{l=1}^{L} \frac{H(\text{bin distribution of } a_l)}{H_{\max}}
$$
where $H_{\max} = \log(10)$. AIS $\in [0, 1]$: values near 1 indicate high alignment diversity (the alignment signal carries information), while values near 0 indicate concentration (alignment is either always high or always low). AIS is an intrinsic property of the network and loss landscape, not of the weight decay method itself. Our experiments show AIS ranges from 0.25 to 0.50 across configurations, indicating moderate alignment diversity.

---

## 4. Theoretical Analysis: The $\rho$ Regime Boundary

### 4.1 The $\rho = \lambda/\eta$ Order Parameter

We define the normalized weight decay strength:
$$
\rho = \frac{\bar{\lambda}}{\eta}
$$
where $\bar{\lambda} = \lambda \cdot \mathbb{E}[\varphi]$ is the effective weight decay rate and $\eta$ is the learning rate. This ratio has a natural physical interpretation: $\rho$ measures the magnitude of the weight decay step relative to the gradient step. At standard settings ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$), $\rho = 0.5$, meaning the weight decay step is half the magnitude of the gradient step.

The $\rho$ parameter connects to multiple independent recent results:

**Theorem 1** (Dual Characterization). *The constraint radius $\tau^* = \eta/\lambda$ from Xie \& Li (2024) and the steady-state gradient-to-weight ratio $R_* \approx \lambda/\eta$ from Defazio (2025) are dual characterizations of $\rho$:*
- $\tau^* = 1/\rho$: larger $\rho$ implies a smaller $\ell_\infty$ constraint ball, i.e., tighter implicit norm constraint.
- $R_* = \rho$: larger $\rho$ implies a higher gradient-to-weight magnitude ratio at steady state.

*This duality is new---neither Xie \& Li nor Defazio make this connection explicit---and provides two complementary routes for experimental validation.*

### 4.2 The Phi Invariance Conjecture

**Conjecture 1** (Phi Invariance Trichotomy). *Define $\rho = \bar{\lambda}/\eta$. There exist constants $0 < \rho_1 < \rho_2$ such that:*

- **Regime I** ($\rho \leq \rho_1$): *For any two budget-equivalent WD schedules, the final loss difference is $O(\rho^2 \cdot V \cdot T \cdot \eta^2)$, where $V$ measures schedule variation. At standard settings ($\rho \approx 0.5$), this bound implies accuracy differences $< 0.1\%$.*

- **Regime II** ($\rho_1 < \rho < \rho_2$): *The loss difference scales as $O(\rho \cdot \sum_t |\lambda_t^{(1)} - \lambda_t^{(2)}| \cdot (1 - \delta_t))$, where $\delta_t$ is the gradient-weight alignment. Alignment-conditioned strategies (e.g., CWD) can provide $O(\rho)$ improvement over uniform schedules.*

- **Regime III** ($\rho \geq \rho_2$): *The WD step competes with the gradient step in magnitude; all modulation strategies produce $O(\rho)$ differences in final loss.*

The conjecture makes three falsifiable predictions:

1. At $\rho = 0.5$ (standard settings), the accuracy spread across budget-equivalent methods should be $< 0.5\%$. **Confirmed**: our experiments show $0.25\%$ on CIFAR-10 under AdamW.
2. At $\rho = 5$ ($\lambda = 5 \times 10^{-3}$), the accuracy spread should be $1$--$3\%$. **Testable** via $\lambda$ sweep experiments.
3. SGD, lacking AdamW's $\ell_\infty$ implicit constraint, should exhibit no Regime I---explaining the observed 18.3$\times$ effect-size ratio.

**Proof elements.** The conjecture rests on three lemmas whose detailed proofs are deferred to Appendix D:
- *Lemma 1* (AdamW $\ell_\infty$ norm stability): Under Adam saturation ($\epsilon / (\sqrt{\hat{v}_i} \cdot |w_i|) < 0.1$), AdamW iterates satisfy $\|w_t\|_\infty \leq \tau^* + O(\eta)$, following from Xie \& Li (2024, Theorem 3.1).
- *Lemma 2* (WD perturbation bound): For two schedules $\lambda_t^{(1)}, \lambda_t^{(2)}$ with the same time-average, the parameter difference $\|\boldsymbol{\theta}_t^{(1)} - \boldsymbol{\theta}_t^{(2)}\|$ is bounded by $O(\sum_{s<t} |\lambda_s^{(1)} - \lambda_s^{(2)}| \cdot \prod_{s'=s+1}^{t} (1 - \lambda_{s'}) \cdot \|w_s\|)$.
- *Lemma 3* (Regime I negligibility): In Regime I, the damped sum in Lemma 2 is $O(\rho^2 \cdot V \cdot T \cdot \eta^2)$, negligible relative to optimization noise.

**Critical assumption**: Lemma 1 requires the Adam saturation condition. We empirically verify this holds for $> 80\%$ of parameters at epoch 100 in our ResNet-20 CIFAR-10 experiments (Appendix D.3).

### 4.3 Why SGD Differs: The Absence of Implicit Constraint

The asymmetry between AdamW and SGD is central to our findings. Under AdamW, the preconditioned update $\mathbf{u}_t = \mathrm{sign}(\hat{\mathbf{m}}_t) \cdot (|\hat{\mathbf{m}}_t| / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon))$ approaches $\pm 1$ per coordinate in the saturated regime, producing an implicit $\ell_\infty$ constraint. This constraint absorbs weight decay perturbations: regardless of the specific modulation pattern $\varphi$, the iterates converge to a neighborhood of the $\ell_\infty$ ball of radius $\tau^* = \eta/\lambda$.

SGD lacks this mechanism entirely. Without per-parameter adaptive scaling, the weight decay step $-\lambda \boldsymbol{\theta}_t$ directly competes with the gradient step $-\eta \nabla \mathcal{L}$, and perturbations to $\lambda$ propagate without damping. Our experiments quantify this: for the constant vs.\ no\_wd comparison, SGD yields Cohen's $d = 10.29$ ($p_{\text{adj}} = 0.002$), while AdamW yields $d < 1$ ($p > 0.5$). This $> 10\times$ difference in effect size is precisely what the absence of Regime I predicts.

**Proposition 2** (SGD Alignment-Optimal Schedule). *Under the alignment-dependent stability bound of Sun et al.\ (CVPR 2025), the optimal budget-allocated WD schedule for SGD is $\lambda_t^* \propto \delta_t / (1 - \delta_t)$, where $\delta_t$ is the gradient-weight alignment at step $t$.* This water-filling solution provides theoretical grounding for CWD's binary mask as a coarse approximation to the optimal schedule. We note, however, that extending Sun et al.'s fixed-$\lambda$ analysis to time-varying $\lambda_t$ involves a formal gap that we flag explicitly.
