# 3. Theoretical Framework

## 3.1 The Phi Modulator Framework

We unify all dynamic WD methods under a single abstraction. The parameter update with a WD modulator $\phi$ is:

$$\theta_{t+1} = \theta_t - \eta_t \cdot u_t - \lambda \cdot \phi(t, \theta_t, g_t) \cdot \theta_t$$

where $u_t = \hat{m}_t / (\hat{v}_t^{1/2} + \varepsilon)$ is the AdamW update direction, $\lambda$ is the base WD coefficient, and $\phi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is the non-negative Phi modulator function.

Every existing dynamic WD method is a special case of $\phi$:

| Method | $\phi(t, \theta, g)$ | Modulation Axis | BEM |
|--------|---------------------|-----------------|-----|
| Constant | $1$ | None | 0.0 |
| Cosine schedule | $\frac{1}{2}(1 + \cos(\pi t/T))$ | Temporal | $\sim$0.50 |
| CWD (hard) | $\mathbf{1}[\text{sign}(\theta) = \text{sign}(u_t)]$ | Directional | $\sim$0.50 |
| SWD | $\|g\| / \|g\|_{\text{mean}}$ | Temporal | $\sim$0.90 |
| Half-$\lambda$ | $0.5$ | None (rescaled) | 0.0 |
| Random mask | $\text{Bernoulli}(0.5)$ | Stochastic | $\sim$0.50 |
| No WD | $0$ | Complete removal | 1.0 |

This taxonomy reveals that the four "families" of dynamic WD (Section 2.2) differ only in which information $\phi$ conditions on: time index (temporal), gradient-weight geometry (directional), or nothing (structural ablations).

## 3.2 Diagnostic Metrics

We define three quantities to characterize any Phi modulator's effects on training dynamics.

**Budget Equivalence Metric (BEM).** BEM measures how much $\phi$ changes the total WD budget relative to constant WD:

$$\text{BEM} = \frac{|\lambda_{\text{eff}}^{\text{method}} - \lambda_{\text{eff}}^{\text{constant}}|}{\lambda_{\text{eff}}^{\text{constant}}}$$

where $\lambda_{\text{eff}} = \lambda \cdot \mathbb{E}_\theta[\phi]$ averaged over training. BEM = 0 means the method applies the same total decay as constant WD; BEM = 1 means complete removal. SWD has the highest BEM ($\sim$0.90), indicating it applies only $\sim$10% of the constant WD budget, yet achieves comparable accuracy---evidence that absolute WD magnitude matters less than modulation pattern under AdamW.

**Coupling Stability Index (CSI).** CSI measures the stability of the WD-optimizer coupling:

$$\text{CSI} = w_1 \cdot \text{CV}(\|\theta\|_{\text{traj}}) + w_2 \cdot \log\kappa(H) + w_3 \cdot \text{CV}(\eta_{\text{eff, layers}})$$

with weights $(w_1, w_2, w_3) = (0.4, 0.3, 0.3)$ combining weight norm trajectory variation, log spectral condition number of the Hessian approximation, and effective learning rate variation across layers. Higher CSI indicates more unstable coupling between WD and the optimizer.

**Alignment Informativeness Score (AIS).** AIS quantifies whether gradient-weight alignment carries actionable information:

$$\text{AIS} = \text{Spearman}_\rho(\cos(\theta_i, g_i), \Delta\text{loss}_i) \quad \text{(per layer, averaged)}$$

AIS $> 0.2$ indicates that alignment predicts loss improvement. AIS is an intrinsic property of the network-dataset pair, not of the WD method.

## 3.3 Theorem 1: Binary Masking Suboptimality

**Theorem 1.** *Let $\phi_{\text{CWD}}$ be the CWD binary masking modulator and $\phi_{\text{const}} = 1$ the constant modulator. Under stochastic gradient descent with noise variance $\sigma^2$, training set size $n$, and $L$ layers, CWD achieves lower expected test loss than constant WD if and only if:*

$$\text{AIS} > \frac{C\sigma^2}{n} \cdot \frac{\Delta\text{CSI}}{\bar{\lambda}}$$

*where $C$ depends on the loss Lipschitz constant and architecture, $\Delta\text{CSI} = \text{CSI}(\phi_{\text{CWD}}) - \text{CSI}(\phi_{\text{const}})$ is the stability cost of binary masking, and $\bar{\lambda}$ is the time-averaged effective WD.*

**Proof sketch.** Decompose the test loss difference into two terms: (i) the alignment benefit, proportional to AIS times $\bar{\lambda}$, capturing how much CWD's selective decay exploits informative geometry; and (ii) the stability cost, proportional to $\Delta\text{CSI}$ scaled by $C\sigma^2/n$, capturing the generalization penalty from the discontinuous modulation. The condition follows from requiring the alignment benefit to exceed the stability cost. Full proof in Appendix B.1.

**Corollary.** At standard $\rho$ ($\rho \approx 0.5$ under AdamW with $\lambda = 5 \times 10^{-4}$) in batch-normalized (BN) networks, BN's scale-invariance drives weights toward an equilibrium norm, making $\Delta\text{CSI}$ non-negligible while AIS remains moderate ($\sim$0.2--0.4). The stability cost exceeds the alignment benefit, predicting constant WD wins.

**Empirical validation.** Across all 7 tested configurations---$\{\text{AdamW, SGD}\} \times \{\text{CIFAR-10, CIFAR-100}\} \times \{\text{ResNet-20}\}$ plus VGG-16-BN/CIFAR-10---constant WD matches or outperforms CWD (all $p > 0.05$ after Bonferroni correction), confirming all 7 predictions from Theorem 1.

## 3.4 Theorem 2: Layer-wise CSI Bound

**Theorem 2.** *For a per-parameter WD schedule $\{\lambda_{i,t}\}_{i=1}^d$ with per-parameter CSI defined as $\text{CSI}_\text{param} = \max_i \text{CV}(\lambda_{i,\cdot})$, the excess generalization gap is bounded by:*

$$\text{GenGap}(\{\lambda_{i,t}\}) - \text{GenGap}(\bar{\lambda}) \leq \frac{2L\sigma^2}{n} \cdot \text{CSI}_\text{param} \cdot T$$

**Random mask paradox.** This bound explains an empirical puzzle: random masking ($\phi = \text{Bernoulli}(0.5)$) produces moderate aggregate CSI because the expected $\phi$ is constant at 0.5, but its per-parameter $\text{CSI}_\text{param}$ is large---each parameter independently alternates between $\lambda$ and 0. CWD exhibits the same pattern: aggregate CSI is moderate (aligned parameters receive decay, others do not), but individual parameters can have $\lambda_{\min} = 0$ during off-steps, driving $\text{CSI}_\text{param}$ high.

## 3.5 Theorem 3: PMP-Optimal WD and Dual Derivation

We derive the optimal WD schedule from first principles using two independent mathematical routes.

**Stochastic PMP derivation.** Consider the per-layer weight norm dynamics under decoupled WD: $\|\theta_{l,t+1}\|^2 \approx (1 - 2\lambda\phi_{l,t})^2 \|\theta_{l,t}\|^2 + \text{noise}$, with gradient-to-weight ratio $\rho_{l,t} = \|g_{l,t}\| / \|\theta_{l,t}\|$. We formulate the optimal control problem: minimize the integrated deviation of $\rho_{l,t}$ from the steady-state target $\rho^*$ (Defazio 2025: $\rho^* \approx \sqrt{2\lambda/\gamma}$ for normalized layers under AdamW) subject to $\lambda \phi_{l,t} \in [0, \lambda_{\max}]$.

Applying the stochastic Pontryagin Maximum Principle and solving the resulting Riccati equation for the costate variable yields:

**Theorem 3.** *The optimal state-feedback WD law is:*
$$\lambda^*(t) = \text{clip}\left(\kappa \cdot (\rho^* - \hat{\rho}_t)^+, \; 0, \; \lambda_{\max}\right)$$

*where $\hat{\rho}_t$ is the per-layer EMA of $\|g_{l,t}\| / \|\theta_{l,t}\|$ (momentum 0.9), $\rho^*$ is the target steady-state ratio, and $\kappa$ is the feedback gain from the Riccati equation solution (default $\kappa = 1$).*

PMP-WD increases decay when $\hat{\rho}_t < \rho^*$ (weights are too large relative to gradients) and decreases decay when $\hat{\rho}_t > \rho^*$ (gradients dominate). The clipping ensures the control remains in $[0, \lambda_{\max}]$.

**Remark 3.1 (RG beta function convergence).** An independent derivation from renormalization group theory treats the WD coefficient as a running coupling constant with beta function $\beta(\lambda) \propto \lambda \cdot (1 - \hat{\delta}_t^2)$, where $\hat{\delta}_t$ is the gradient-weight cosine similarity. The fixed point analysis yields $\lambda^* = \beta_0 \cdot \hat{\delta}_t^2$, which agrees with the PMP-WD formula in the moderate-alignment regime ($\hat{\delta}_t \in [0.3, 0.7]$) where $\kappa(\rho^* - \hat{\rho}_t)^+ \approx \beta_0 \hat{\delta}_t^2$. The convergence of two independent mathematical frameworks strengthens the theoretical foundation.

**Distinction from AdamC.** Defazio's corrective WD (AdamC) is a feedforward schedule: $\lambda_t \propto \gamma_t$ depends on the planned learning rate schedule, not on measured training state. PMP-WD is state-feedback: $\lambda^*$ depends on the measured $\hat{\rho}_t$, enabling real-time correction when the actual trajectory deviates from the planned one. Feedforward control cannot compensate for unexpected perturbations (batch noise, distribution shift during training); state-feedback does.

## 3.6 Proposition 1: Alignment Noise Design Constraint

**Proposition 1.** *For mini-batch size $b \leq 256$ and full-network cosine similarity $\hat{\delta}_t = \cos(g_t, \theta_t)$:*

$$\text{CV}(\hat{\delta}_t) = \frac{\text{std}(\hat{\delta}_t)}{\text{mean}(\hat{\delta}_t)} \gg 1$$

*for most training steps. Consequently, any alignment-aware WD method that uses the raw single-step $\hat{\delta}_t$ as a feedback signal operates on a noisy compass.*

This result follows from the high dimensionality of $g_t$ and $\theta_t$: the cosine similarity between two random vectors in $\mathbb{R}^d$ concentrates around zero, and mini-batch gradient noise pushes $\hat{\delta}_t$ across the full $[-1, 1]$ range between consecutive steps.

**Corollary (Design Constraint).** Any alignment-aware WD method must use temporally aggregated signals: EMA smoothing with aggregation horizon $k \geq 10$ steps. PMP-WD satisfies this by construction, as $\hat{\rho}_t$ uses per-layer scalar EMA (momentum 0.9, corresponding to $k \approx 10$ effective steps). CWD's binary sign mask is partially robust (sign is a low-dimensional projection of alignment), but the underlying cosine similarity is still noisy.

<!-- FIGURES
- Figure 2: theorem1_regime_desc.md — Theorem 1 regime illustration: alignment benefit vs stability cost
- Figure 3: pmpwd_control_desc.md — PMP-WD control diagram: state-feedback vs feedforward
-->
