# 3. The Phi Modulator Framework

## 3.1 Unified Formulation

Every weight decay method in the literature can be expressed as a modulation of a base decay rate. We formalize this observation with the **phi modulator framework**.

**Definition 1 (Phi Modulator).** Given base weight decay coefficient $\lambda_{\text{base}} > 0$, a WD method is defined by a modulator function $\phi: \mathbb{N} \times \mathbb{R}^d \times \mathbb{R}^d \to [0, C]$ for some constant $C \geq 1$, producing effective weight decay:

$$\lambda_{\text{eff}}(t, w, g) = \phi(t, w, g) \cdot \lambda_{\text{base}}$$

The SGD update with phi-modulated WD reads:

$$w_{t+1} = \bigl(1 - \gamma_t \, \phi(t, w_t, g_t) \, \lambda_{\text{base}}\bigr) w_t - \gamma_t \, g_t$$

Table 1 maps published WD methods to their phi modulators.

**Table 1: Taxonomy of WD methods under the phi modulator framework.** Each method is characterized by the form of $\phi(t, w, g)$, its input dependencies, and output range. $\mathbf{1}[\cdot]$ denotes the indicator function.

| Method | $\phi(t, w, g)$ | Inputs | Range |
|--------|-----------------|--------|-------|
| No WD | $0$ | -- | $\{0\}$ |
| Constant | $1$ | -- | $\{1\}$ |
| Cosine schedule | $\frac{1}{2}(1 + \cos(\pi t / T))$ | $t$ | $[0, 1]$ |
| Half-$\lambda$ | $0.5$ | -- | $\{0.5\}$ |
| CWD (Chen et al., 2026) | $\mathbf{1}[\text{sign}(w_i) = \text{sign}(\Delta w_i)]$ | $w, g$ | $\{0, 1\}$ |
| SWD (Xie et al., 2023) | $h(\|g\|) / h(\|g\|_{\text{ref}})$ | $g$ | $\mathbb{R}_+$ |
| Random mask | $\text{Bernoulli}(p)$ | -- | $\{0, 1\}$ |
| PMP-WD (Ours) | $\mathbf{1}[\langle p(t), w(t) \rangle > 0]$ | $w, p$ | $\{0, 1\}$ |

The framework provides three immediate benefits. First, it makes the assumptions of each method explicit: constant WD ignores all training state, CWD uses parameter-level sign information, SWD uses global gradient norms, and PMP-WD uses the costate-weight inner product. Second, it enables fair comparison by separating the modulation strategy ($\phi$) from the base rate ($\lambda_{\text{base}}$). Third, it reveals structural similarities: CWD, random mask, and PMP-WD all produce binary $\phi \in \{0, 1\}$ values and thus implement bang-bang control, differing only in the switching criterion.

## 3.2 Evaluation Metrics

Comparing WD methods requires metrics that are invariant to trivially different base rates and that capture distinct aspects of the WD-optimization interaction. We propose three diagnostic metrics.

**Budget Equivalence Metric (BEM).** The BEM measures how much the mean effective WD deviates from a constant baseline:

$$\text{BEM} = \frac{|\bar{\lambda}_{\text{eff}} - \lambda_{\text{const}}|}{\lambda_{\text{const}}}, \quad \bar{\lambda}_{\text{eff}} = \frac{1}{T} \sum_{t=1}^{T} \lambda_{\text{eff}}(t)$$

A BEM of 0 indicates the method applies the same total regularization budget as constant WD; a BEM of 1 indicates zero mean effective WD (equivalent to no WD on average). In our CIFAR-10 experiments, constant WD achieves BEM = 0 by definition, CWD and PMP-WD achieve BEM $\approx 0.51$ (applying decay roughly half the time), and SWD achieves BEM = 0.90.

**Coupling Stability Index (CSI).** The CSI quantifies the stability of the WD-optimization coupling via the coefficient of variation of weight norm increments:

$$\text{CSI} = \text{CV}\bigl(\{\|\Delta w_t\| \cdot \text{sign}(\|w_{t+1}\| - \|w_t\|)\}_{t=1}^{T}\bigr)$$

Low CSI indicates smooth, predictable interaction between WD and the optimizer; high CSI indicates oscillatory or unstable coupling. Across our experiments, CSI ranges from 0.82 (PMP-WD) to 0.96 (cosine schedule), indicating all tested methods maintain stable coupling on BN architectures.

**Alignment Informativeness Score (AIS).** The AIS measures the entropy of the per-layer alignment distribution, capturing how much the alignment signal varies across layers:

$$\text{AIS} = -\sum_{l=1}^{L} \hat{p}_l \log \hat{p}_l, \quad \hat{p}_l = \frac{|\delta_l|}{\sum_{l'} |\delta_{l'}|}$$

where $\delta_l = \cos(g_l, w_l)$ is the per-layer gradient-weight alignment. High AIS indicates uniform alignment across layers (uninformative for per-layer modulation); low AIS indicates concentrated alignment in specific layers (informative for targeted modulation). In our experiments, AIS ranges from 0.29 to 0.39 across methods, indicating moderate layer-level variation.

## 3.3 Phi Modulator Framework Diagram

As shown in Figure 1, the phi modulator framework operates as a feedback control loop: the training state $(w_t, g_t, t)$ feeds into the modulator $\phi$, which outputs the effective WD coefficient $\lambda_{\text{eff}}(t)$, governing the weight update. The certified band (Section 4) constrains the admissible range of $\lambda_{\text{eff}}$, and the diagnostic metrics (BEM, CSI, AIS) monitor the quality of the modulation.

![Phi modulator framework: a control loop where training state feeds into the modulator phi, constrained by the certified band.](figures/framework_diagram_desc.md)

<!-- FIGURES
- Figure 1: framework_diagram_desc.md -- Phi modulator framework control loop diagram
- Table 1: inline -- Taxonomy of WD methods under phi modulator framework
-->
