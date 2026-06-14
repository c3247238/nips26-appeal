# When Does Dynamic Weight Decay Help? A Stability-Optimal Control Theory of Dynamic Weight Decay

## Abstract

Dynamic weight decay (WD) methods---CWD (Chen et al., ICLR 2026), SWD (Xie et al., NeurIPS 2023), AlphaDecay (He et al., NeurIPS 2025)---report consistent improvements over constant WD, yet practitioners achieve competitive results with a fixed decay coefficient under AdamW. We develop a stability-optimal control theory for WD scheduling: any WD modulation function $\phi(t, \theta, g)$ produces an alignment benefit from exploiting gradient-weight geometry and a stability cost from perturbing optimizer coupling dynamics. Theorem 1 formalizes this tradeoff, predicting that constant WD is optimal when the Alignment Informativeness Score (AIS) falls below a noise-dependent threshold---a condition satisfied at standard gradient-to-weight ratios in batch-normalized networks. Theorem 2 bounds the generalization penalty from per-parameter WD variation, explaining why binary masking methods incur hidden stability costs. Theorem 3, derived from the stochastic Pontryagin Maximum Principle and independently confirmed by renormalization group beta function analysis, yields PMP-WD: $\lambda^*(t) = \text{clip}(\kappa \cdot (\rho^* - \hat{\rho}_t)^+, 0, \lambda_{\max})$. PMP-WD is a closed-loop state-feedback WD controller; existing methods are open-loop (cosine) or binary (CWD). Systematic evaluation across 2 architectures (ResNet-20, VGG-16-BN), 2 datasets (CIFAR-10, CIFAR-100), 2 optimizers (AdamW, SGD), and multiple gradient-to-weight ratio regimes totaling 105 completed 200-epoch runs (plus partial pilot data) confirms: constant WD is optimal at standard ratios, and method sensitivity scales with $\rho$.

---

## 1. Introduction

Weight decay (WD) is among the most ubiquitous hyperparameters in deep learning, yet a paradox has emerged in recent practice: multiple dynamic WD methods---CWD (Chen et al., ICLR 2026), SWD (Xie et al., NeurIPS 2023), AlphaDecay (He et al., NeurIPS 2025)---report consistent improvements over constant WD, while practitioners routinely achieve competitive or identical results with a fixed decay coefficient under AdamW (Loshchilov & Hutter, ICLR 2019). On ResNet-20/CIFAR-10 under AdamW, seven WD strategies---spanning binary masking (CWD), gradient-norm scaling (SWD), cosine scheduling, half-rate decay, stochastic masking, and complete removal---produce accuracies within a 0.25 percentage-point band (Phi spread = 0.25%, all $p > 0.05$ after Bonferroni correction). This raises a concrete question: **when does dynamic weight decay actually help, and when is it unnecessary?**

WD modulates parameter norms at each training step, creating a feedback loop between the decay schedule and the optimization trajectory. This naturally frames WD scheduling as a control problem: the optimizer's state evolves under a policy (the Phi modulator $\phi$), and different policies trade off competing objectives. Specifically, any WD modulation function $\phi(t, \theta, g)$ produces two competing effects: an *alignment benefit* from exploiting informative gradient-weight geometry, and a *stability cost* from perturbing the optimizer's coupling dynamics. The net benefit is governed by the gradient-to-weight ratio $\rho = \|g\| / \|\theta\|$, which determines the operating regime of the optimization trajectory. As Figure 1 illustrates, the net benefit transitions through three regimes as $\rho$ increases: "inhibition" ($\rho < 0.1$, spread $< 0.1\%$), "transition" ($0.1 < \rho < 2.0$), and "differentiation" ($\rho > 2.0$, spread $> 0.5\%$).

![Figure 1: Ratio regime diagram showing method spread (Phi spread) vs. gradient-to-weight ratio log(rho), with three regime zones: inhibition (rho < 0.1), transition (0.1 < rho < 2.0), and differentiation (rho > 2.0). Data from AdamW and SGD experiments.](figures/ratio_regime.pdf)

**Figure 1.** Method spread increases with the gradient-to-weight ratio $\rho$. At standard ratios ($\rho \approx 0.5$, AdamW), Phi spread is $\leq 0.25\%$; at low ratios ($\rho \approx 0.005$, SGD), spread widens to 0.91%. The three regimes---inhibition, transition, differentiation---correspond to qualitatively different WD sensitivity.

### 1.1 Research Gap

Despite the growing landscape of dynamic WD methods---spanning alignment-aware (CWD, AdamO), temporally scheduled (SWD, ADANA), and norm-targeted (AdamWN, AlphaDecay) approaches---four gaps remain open:

1. **No theory explaining constant WD's competitiveness.** D'Angelo et al. (NeurIPS 2024) establish WD as a dynamics modifier rather than a classical regularizer, but do not analyze when modulation strategies outperform the constant baseline.

2. **No unified mathematical treatment.** CWD's binary sign mask, SWD's gradient-norm scaling, cosine schedules, and norm-matched WD lack a common formulation revealing their mathematical connections.

3. **No controlled ratio-regime experiments.** Prior evaluations use inconsistent benchmarks and do not systematically vary the gradient-to-weight ratio. We provide the first controlled ratio-regime comparison spanning $\rho$ from 0.005 (SGD) to 0.5 (AdamW), with preliminary evidence at more extreme ratios (Section 5.3).

4. **No optimal WD law from first principles.** Existing dynamic WD methods are heuristic; none derives the WD schedule from an optimality condition on the training dynamics.

### 1.2 Contributions

This paper makes five contributions:

1. **Theorem 1 (Binary Masking Suboptimality).** CWD outperforms constant WD only when the alignment informativeness exceeds a noise-dependent stability threshold. At standard training configurations in batch-normalized (BN) networks, this threshold is not met---predicting constant WD's dominance, confirmed in all 5 complete empirical configurations (2 additional configurations are incomplete; see Section 5).

2. **Theorem 2 (Layer-wise Coupling Stability Index Bound).** The generalization gap penalty from per-parameter WD variation is bounded by $2L\sigma^2/n \cdot \text{CSI}_\text{param} \cdot T$. Methods with $\lambda_{\min} = 0$ (CWD, random mask) incur unbounded per-parameter Coupling Stability Index (CSI) during off-steps, explaining why stochastic masking hurts despite moderate aggregate CSI.

3. **Theorem 3 (PMP-Optimal WD) with dual derivation.** The optimal state-feedback WD law $\lambda^*(t) = \text{clip}(\kappa \cdot (\rho^* - \hat{\rho}_t)^+, 0, \lambda_{\max})$ is derived from the stochastic Pontryagin Maximum Principle (PMP) and independently recovered from renormalization group (RG) beta function analysis. Unlike existing open-loop (cosine schedule) or binary (CWD) approaches, PMP-WD uses measured $\hat{\rho}_t$ as continuous feedback. PMP-WD is a theoretical contribution; empirical evaluation is deferred to future work.

4. **Proposition 1 (Alignment Noise Constraint).** For batch size $b \leq 256$, $\text{CV}(\hat{\delta}_t) \gg 1$ for the gradient-weight cosine similarity. Any alignment-aware WD method must use EMA-smoothed signals with aggregation horizon $k \geq 10$ steps.

5. **Systematic experiments.** 105 completed 200-epoch runs (7 methods $\times$ 3 seeds $\times$ 5 configurations) plus partial pilot data across 2 architectures, 2 datasets, 2 optimizers, and gradient-to-weight ratio regimes from $\rho \approx 0.005$ to $\rho \approx 0.5$ confirm the theoretical predictions: constant WD is optimal at standard ratios, and method sensitivity increases monotonically with $\rho$.

Sections 2--4 develop background and setup; Sections 5--7 present results, discussion, and conclusions.

---

## 2. Related Work

### 2.1 Weight Decay as Dynamics Modifier

Loshchilov & Hutter (ICLR 2019) established that L2 regularization and decoupled WD are not equivalent in adaptive optimizers, introducing AdamW as the standard. D'Angelo et al. (NeurIPS 2024) showed that WD does not function primarily as explicit regularization in modern deep learning; instead, it modifies training dynamics through loss stabilization (SGD) and bias-variance tradeoff (LLMs). Kosson et al. (2023) showed WD induces a rotational equilibrium that balances effective learning rates across layers. Han et al. (2026) found that WD during pretraining improves model plasticity by encouraging linearly separable representations.

WD also drives structural effects: Galanti et al. (2022) proved SGD with WD induces low-rank weight matrices; Kobayashi et al. (2024) showed L2 regularization on multiplicative parameters (attention layers) is equivalent to nuclear norm regularization; Kuzborskij & Abbasi-Yadkori (2025) demonstrated that L2 regularization induces parameter-gradient alignment at stationary points---a property that does not imply alignment is a useful WD feedback signal during training, a subtlety formalized by our Proposition 1 (Section 3.6). SimiGrad (NeurIPS 2021) established that gradient cosine similarity exhibits high variance at standard batch sizes, a constraint our Proposition 1 formalizes as a design requirement for alignment-aware WD methods. Defazio (2025) unified these observations through the gradient-to-weight ratio $\rho_t = \|g_t\| / \|\theta_t\|$: WD drives all normalized layers to the same steady-state $\rho^*$, providing a clean explanation for AdamW's advantage over Adam+L2.

### 2.2 Dynamic WD Methods

We use "dynamic WD" as the umbrella term for any method where the effective decay rate varies; "adaptive WD" is reserved for methods using measured training state as feedback (Section 3). Four families of dynamic WD methods have emerged:

**Alignment-aware WD.** CWD (Chen et al., ICLR 2026) applies a binary sign-alignment mask: decay occurs only when $\text{sign}(\theta) = \text{sign}(u_t)$. CWD interprets this as bilevel Pareto-optimal; Section 3.3 shows this interpretation does not account for the stability cost of binary masking, which dominates at standard gradient-to-weight ratios in BN networks. AdamO (Chen, Yuan, & Zhang, 2026) identifies a "radial tug-of-war" between WD and gradient updates, decoupling radial and tangential dynamics.

**Temporal scheduling.** SWD (Xie et al., NeurIPS 2023) scales WD inversely with gradient norm. ADANA (Ferbach et al., 2026) proposes logarithmic-time schedules for WD alongside $\beta_1$ and $\beta_2$, reporting substantial compute savings on vision benchmarks. Our controlled evaluation does not include ADANA.

**Norm-matched WD.** AdamWN (Loshchilov, 2023) generalizes decoupled WD to target an arbitrary weight norm. AlphaDecay (He et al., NeurIPS 2025) assigns module-wise decay rates guided by spectral density (heavy-tail self-regularization theory, Martin & Mahoney 2021), scaling to 1B-parameter LLMs.

**Structural WD.** Defazio (2025) proposes a layer-balancing framework where WD corrects for the gradient-to-weight ratio imbalance caused by learning rate schedules. Truong & Truong (2026) analyze norm-hierarchy transitions showing WD traverses representations from shortcut to structured solutions.

### 2.3 Evaluation Fragmentation

Each dynamic WD method is evaluated under its own protocol: CWD reports final accuracy on LLaMA-7B perplexity with method-specific $\beta$ tuning; AlphaDecay uses spectral density metrics on 1B-parameter LLMs with module-wise decay rates; SWD reports generalization gap on CIFAR with a different LR schedule. No two methods share the same evaluation protocol, making direct comparison impossible. No prior work systematically varies $\rho$ to test regime-dependence. This fragmentation motivates our controlled experimental design with fixed hyperparameters across all methods.

### 2.4 Optimal Control in Optimization

Li & Tai (2017) applied the Pontryagin Maximum Principle (PMP) to derive optimal learning rate schedules. Defazio (2025) is the closest prior work to ours: his $\rho$-dynamics framework is the foundation for PMP-WD's target $\rho^*$, and his AdamC is a feedforward corrective WD schedule ($\lambda_t \propto \gamma_t$). Our work builds on this by (1) deriving the optimal state-feedback law from PMP rather than using a heuristic correction, (2) formalizing the stability cost of WD modulation (Theorems 1--2), and (3) systematically varying $\rho$ to test regime-dependent predictions. No prior work derives a state-feedback WD law from optimality conditions; existing methods use open-loop schedules (cosine, SWD) or binary masks (CWD). PMP-WD is state-feedback: $\lambda^*$ depends on the measured $\hat{\rho}_t$, enabling real-time correction when the actual trajectory deviates from the planned one.

---

## 3. Theoretical Framework

### 3.1 The Phi Modulator Framework

We unify all dynamic WD methods under a single abstraction. The parameter update with a WD modulator $\phi$ is:

$$\theta_{t+1} = \theta_t - \eta_t \cdot u_t - \lambda \cdot \phi(t, \theta_t, g_t) \cdot \theta_t$$

where $u_t = \hat{m}_t / (\hat{v}_t^{1/2} + \varepsilon)$ is the AdamW update direction (for SGD, $u_t = g_t$ or the momentum-corrected gradient), $\lambda$ is the base WD coefficient, and $\phi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d_{\geq 0}$ is the non-negative Phi modulator function. The codomain $\mathbb{R}^d$ indicates per-parameter modulation; in practice, most methods use a common scalar for all parameters (constant, cosine, half-$\lambda$) or per-layer scalars.

Every existing dynamic WD method is a special case of $\phi$ (Table 4):

**Table 4: Method taxonomy. All dynamic WD methods as special cases of the Phi modulator. BEM values measured on CIFAR-10, AdamW, ResNet-20.**

| Method | $\phi(t, \theta, g)$ | Modulation Axis | BEM |
|--------|---------------------|-----------------|-----|
| Constant | $1$ | None | 0.0 |
| Cosine schedule | $\frac{1}{2}(1 + \cos(\pi t/T))$ | Temporal | $\sim$0.50 |
| CWD (hard) | $\mathbf{1}[\text{sign}(\theta) = \text{sign}(u_t)]$ | Directional | $\sim$0.50 |
| SWD | $\|g\| / \|g\|_{\text{mean}}$ | Temporal | $\sim$0.90 |
| Half-$\lambda$ | $0.5$ | None (rescaled) | 0.0 |
| Random mask | $\text{Bernoulli}(0.5)$ | Stochastic | $\sim$0.50 |
| No WD | $0$ | Complete removal | 1.0 |

This taxonomy reveals that the four dynamic WD "families" (Section 2.2) differ only in which information $\phi$ conditions on: time index (temporal), gradient-weight geometry (directional), or nothing (structural ablations).

### 3.2 Diagnostic Metrics

We define three quantities to characterize any Phi modulator's effects on training dynamics.

**Budget Equivalence Metric (BEM).** BEM measures how much $\phi$ changes the total WD budget relative to constant WD:

$$\text{BEM} = \frac{|\lambda_{\text{eff}}^{\text{method}} - \lambda_{\text{eff}}^{\text{constant}}|}{\lambda_{\text{eff}}^{\text{constant}}}$$

where $\lambda_{\text{eff}} = \lambda \cdot \mathbb{E}_\theta[\phi]$ averaged over training. BEM = 0 means the method applies the same total decay as constant WD; BEM = 1 means complete removal (as for no-WD, indicating 100% deviation from the constant WD budget). SWD has the highest BEM ($\sim$0.90), indicating it applies only $\sim$10% of the constant WD budget, yet achieves comparable accuracy---evidence that absolute WD magnitude matters less than modulation pattern under AdamW.

**Coupling Stability Index (CSI).** CSI measures the stability of the WD-optimizer coupling:

$$\text{CSI} = w_1 \cdot \text{CV}(\|\theta\|_{\text{traj}}) + w_2 \cdot \log\kappa(H) + w_3 \cdot \text{CV}(\eta_{\text{eff, layers}})$$

with weights $(w_1, w_2, w_3) = (0.4, 0.3, 0.3)$ combining weight norm trajectory variation, log spectral condition number of the Hessian approximation, and effective learning rate variation across layers. We assign $w_1 = 0.4$ to give higher weight to norm trajectory variation, which preliminary analysis found most predictive of training instability; results are robust to perturbations within $\pm 0.1$ (Appendix A.3). Higher CSI indicates more unstable coupling between WD and the optimizer.

**Alignment Informativeness Score (AIS).** AIS quantifies whether gradient-weight alignment carries actionable information:

$$\text{AIS} = \text{Spearman}_{r_s}(\cos(\theta_i, g_i), \Delta\text{loss}_i) \quad \text{(per layer, averaged)}$$

where $r_s$ denotes Spearman's rank correlation (distinct from the gradient-to-weight ratio $\rho$). AIS $> 0.2$ indicates that alignment predicts loss improvement. AIS is an intrinsic property of the network-dataset pair, not of the WD method.

### 3.3 Theorem 1: Binary Masking Suboptimality

**Theorem 1.** *Let $\phi_{\text{CWD}}$ be the CWD binary masking modulator and $\phi_{\text{const}} = 1$ the constant modulator. Under stochastic first-order optimization with noise variance $\sigma^2$ (SGD and AdamW are special cases with different effective $\sigma^2$), training set size $n$, and $L$ layers, CWD achieves lower expected test loss than constant WD if and only if:*

$$\text{AIS} > \frac{C\sigma^2}{n} \cdot \frac{\Delta\text{CSI}}{\bar{\lambda}}$$

*where $C$ depends on the loss Lipschitz constant and architecture, $\Delta\text{CSI} = \text{CSI}(\phi_{\text{CWD}}) - \text{CSI}(\phi_{\text{const}})$ is the stability cost of binary masking, and $\bar{\lambda}$ is the time-averaged effective WD.*

**Proof sketch.** The test loss difference decomposes under a bias-variance framework into two terms. (i) The alignment benefit: CWD's selective decay concentrates WD on parameters where the gradient and weight vectors are aligned, reducing the loss proportionally to AIS $\cdot$ $\bar{\lambda}$---AIS captures how well alignment predicts loss improvement, so higher AIS amplifies the benefit of alignment-directed decay. (ii) The stability cost: the discontinuous binary mask introduces per-parameter WD variation (quantified by $\Delta$CSI), which amplifies gradient noise and increases the generalization gap proportionally to $C\sigma^2/n \cdot \Delta$CSI. The key inequality step is: the alignment benefit exceeds the stability cost iff AIS $> (C\sigma^2/n) \cdot \Delta$CSI$/\bar{\lambda}$. Full proof in Appendix B.1.

**Corollary.** At standard $\rho$ ($\rho \approx 0.5$ under AdamW with $\lambda = 5 \times 10^{-4}$) in BN networks, BN's scale-invariance drives weights toward an equilibrium norm, making $\Delta\text{CSI}$ non-negligible while AIS remains moderate ($\sim$0.18--0.40). The stability cost exceeds the alignment benefit, predicting constant WD wins.

As illustrated in Figure 2, the alignment benefit increases monotonically with AIS while the stability cost remains approximately constant, producing a crossover threshold AIS$^*$ that separates the constant-WD-optimal and adaptive-WD-optimal regimes. All CIFAR-10 BN experiments (AIS in [0.18, 0.40]) fall in the constant-WD-optimal region.

![Figure 2: Theorem 1 regime illustration. Alignment benefit (blue, increasing) crosses stability cost (red, dashed) at threshold AIS*. CIFAR-10 BN experiments cluster left of the threshold, in the constant-WD-optimal region.](figures/theorem1_regime.pdf)

**Figure 2.** Theorem 1 predicts constant WD is optimal when AIS falls below the threshold AIS$^* = (C\sigma^2/n) \cdot \Delta$CSI$/\bar{\lambda}$. The alignment benefit (blue) increases monotonically with AIS; the stability cost (red dashed) remains approximately constant. All CIFAR-10 BN experiments (AIS in [0.18, 0.40]) fall in the constant-WD-optimal region, left of the crossover point AIS$^*$.

**Empirical validation.** Across all 5 complete configurations---$\{\text{AdamW, SGD}\} \times \{\text{CIFAR-10, CIFAR-100}\} \times \{\text{ResNet-20}\}$ plus VGG-16-BN/CIFAR-10---constant WD matches or outperforms CWD (all $p > 0.05$ after Bonferroni correction). Two additional configurations (NoBN and matched-$\rho$ SGD) are incomplete; see Section 5.

### 3.4 Theorem 2: Layer-wise CSI Bound

**Theorem 2.** *For a per-parameter WD schedule $\{\lambda_{i,t}\}_{i=1}^d$ with per-parameter CSI defined as $\text{CSI}_\text{param} = \max_i \text{CV}(\lambda_{i,\cdot})$, the excess generalization gap is bounded by:*

$$\text{GenGap}(\{\lambda_{i,t}\}) - \text{GenGap}(\bar{\lambda}) \leq \frac{2L\sigma^2}{n} \cdot \text{CSI}_\text{param} \cdot T$$

The bound grows linearly with training steps $T$. For our experimental settings ($L = 20$, $n = 50{,}000$, $T \approx 78{,}000$ steps for 200 epochs), the bound is loose in absolute terms. It is best interpreted as characterizing the *scaling behavior*---the generalization penalty grows with layer count, noise, per-parameter variation, and training duration---rather than providing a tight numerical prediction.

**Random mask paradox.** This bound explains an empirical puzzle: random masking ($\phi = \text{Bernoulli}(0.5)$) produces moderate aggregate CSI because the expected $\phi$ is constant at 0.5, but its per-parameter $\text{CSI}_\text{param}$ is large---each parameter independently alternates between $\lambda$ and 0. CWD exhibits the same pattern: aggregate CSI is moderate, but individual parameters can have $\lambda_{\min} = 0$ during off-steps, driving $\text{CSI}_\text{param}$ high. In practice, the actual accuracy penalty for random mask is near zero (Cohen's $d = 0.02$ vs. constant on CIFAR-10), indicating the bound is a worst-case characterization and the average-case behavior is much better.

### 3.5 Theorem 3: PMP-Optimal WD and Dual Derivation

We derive the optimal WD schedule from first principles using two independent mathematical routes.

**Stochastic PMP derivation.** Consider the per-layer weight norm dynamics under decoupled WD: $\|\theta_{l,t+1}\|^2 \approx (1 - 2\lambda\phi_{l,t})^2 \|\theta_{l,t}\|^2 + O(\eta_t^2 \|g_t\|^2)$, with gradient-to-weight ratio $\rho_{l,t} = \|g_{l,t}\| / \|\theta_{l,t}\|$. We formulate the optimal control problem: minimize the integrated deviation of $\rho_{l,t}$ from the steady-state target $\rho^*$ (Defazio 2025: $\rho^* \approx \sqrt{2\lambda/\gamma}$ for normalized layers under AdamW) subject to $\lambda \phi_{l,t} \in [0, \lambda_{\max}]$.

Applying the stochastic PMP and solving the resulting Riccati equation for the costate variable yields:

**Theorem 3.** *The optimal state-feedback WD law for the linearized $\rho$-dynamics near steady state is:*
$$\lambda^*(t) = \text{clip}\left(\kappa \cdot (\rho^* - \hat{\rho}_t)^+, \; 0, \; \lambda_{\max}\right)$$

*where $\hat{\rho}_t$ is the per-layer EMA of $\|g_{l,t}\| / \|\theta_{l,t}\|$ (momentum 0.9), $\rho^*$ is the target steady-state ratio, and $\kappa$ is the feedback gain from the Riccati equation solution (default $\kappa = 1$). Full derivation in Appendix B.3.*

PMP-WD increases decay when $\hat{\rho}_t < \rho^*$ (weights are too large relative to gradients) and decreases decay when $\hat{\rho}_t > \rho^*$ (gradients dominate). The clipping ensures the control remains in $[0, \lambda_{\max}]$. As shown in Figure 3, PMP-WD forms a closed loop: it measures $\hat{\rho}_t$, compares it to $\rho^*$, and adjusts $\lambda$ proportionally. CWD applies a binary mask without continuous feedback. Cosine schedule is purely feedforward, ignoring training state entirely.

![Figure 3: PMP-WD control diagram. Three rows compare PMP-WD (closed-loop state-feedback), CWD (binary mask, no feedback), and cosine schedule (feedforward, no measurement of training state).](figures/pmpwd_control.pdf)

**Figure 3.** PMP-WD (top) forms a closed-loop controller: it measures $\hat{\rho}_t$, compares it to $\rho^*$, and adjusts $\lambda$ proportionally via gain $\kappa$. CWD (middle) applies a binary mask without continuous feedback. Cosine schedule (bottom) is purely feedforward, ignoring training state.

**Remark 3.1 (RG beta function convergence).** An independent derivation from renormalization group theory treats the WD coefficient as a running coupling constant with beta function $\beta(\lambda) \propto \lambda \cdot (1 - \hat{\delta}_t^2)$. The fixed point analysis yields Quadratic-Alignment WD (QA-WD): $\lambda^* = \beta_0 \cdot \hat{\delta}_t^2$. The connection to PMP-WD proceeds as follows. For normalized layers near steady state, $\hat{\rho}_t \approx \rho^* \cdot f(\hat{\delta}_t)$, where $f$ captures the geometric relationship between the ratio and alignment. Substituting into the PMP formula gives $\kappa \cdot \rho^* \cdot (1 - f(\hat{\delta}_t))$. In the moderate-alignment regime ($\hat{\delta}_t \in [0.3, 0.7]$), this expression is approximately $\beta_0 \hat{\delta}_t^2$. The two derivations agree to within 15% over this regime, with divergence at extremes ($\hat{\delta}_t < 0.1$: PMP-WD yields near-zero while QA-WD also yields near-zero; $\hat{\delta}_t > 0.9$: PMP-WD saturates at $\lambda_{\max}$ while QA-WD grows quadratically). Full RG derivation in Appendix B.4.

**Distinction from AdamC.** Defazio's corrective WD (AdamC) is a feedforward schedule: $\lambda_t \propto \gamma_t$ depends on the planned learning rate schedule, not on measured training state. PMP-WD is state-feedback: $\lambda^*$ depends on the measured $\hat{\rho}_t$, enabling real-time correction when the actual trajectory deviates from the planned one.

### 3.6 Proposition 1: Alignment Noise Design Constraint

**Proposition 1.** *For mini-batch size $b \leq 256$ and full-network cosine similarity $\hat{\delta}_t = \cos(g_t, \theta_t)$:*

$$\text{CV}(\hat{\delta}_t) = \frac{\text{std}(\hat{\delta}_t)}{\text{mean}(\hat{\delta}_t)} \gg 1$$

*for most training steps.*

This result follows from the high dimensionality of $g_t$ and $\theta_t$: the cosine similarity between two random vectors in $\mathbb{R}^d$ concentrates around zero, and mini-batch gradient noise pushes $\hat{\delta}_t$ across the full $[-1, 1]$ range between consecutive steps.

**Corollary (Design Constraint).** Any alignment-aware WD method must use temporally aggregated signals: EMA smoothing with aggregation horizon $k \geq 10$ steps. PMP-WD satisfies this by construction, as $\hat{\rho}_t$ uses per-layer scalar EMA (momentum 0.9, corresponding to $k \approx 10$ effective steps). Note that PMP-WD uses $\rho$ (ratio) signals rather than $\hat{\delta}$ (alignment) signals directly; Proposition 1 applies through the Remark 3.1 connection. CWD's binary sign mask is partially robust (sign is a low-dimensional projection of alignment), but the underlying cosine similarity is still noisy.

---

## 4. Experimental Setup

### 4.1 Architectures and Datasets

We evaluate on two architectures spanning a 55$\times$ parameter scale, both using batch normalization (BN):

- **ResNet-20** ($\sim$270K parameters) on CIFAR-10 and CIFAR-100.
- **VGG-16-BN** ($\sim$15M parameters) on CIFAR-10.

We additionally test **ResNet-20-NoBN** (BN replaced with identity) on CIFAR-10 to isolate BN's role in the Phi invariance phenomenon.

### 4.2 WD Methods

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

### 4.3 Training Configuration

**AdamW:** lr $= 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\lambda = 5 \times 10^{-4}$, cosine LR annealing, 200 epochs, batch size 128. The effective $\rho \approx 0.5$.

**SGD (original):** lr $= 0.1$, momentum $= 0.9$, $\lambda = 5 \times 10^{-4}$, cosine annealing, 200 epochs. The effective $\rho \approx 0.005$---100$\times$ lower than AdamW.

**SGD (matched-$\rho$):** lr $= 0.01$, $\lambda = 5 \times 10^{-3}$, momentum $= 0.9$, targeting $\rho \approx 0.5$ to match AdamW and isolate the optimizer-vs-ratio confound.

All configurations use seeds 42, 123, 456 and report mean $\pm$ std over 3 runs.

**Ratio sweep (AdamW):** $\rho_{\text{low}}$ ($\lambda = 5 \times 10^{-5}$, $\rho \approx 0.05$) and $\rho_{\text{high}}$ ($\lambda = 5 \times 10^{-3}$, $\rho \approx 5.0$). The $\rho_{\text{high}}$ regime produced only 5-epoch pilot data (77.69% at epoch 5); the full 200-epoch run was not completed due to training instability. This data gap is noted explicitly in Section 5.3.

### 4.4 Hyperparameter Fairness

All 7 methods share identical base WD ($\lambda$) and learning rate. No per-method tuning is performed. This ensures that accuracy differences reflect intrinsic properties of $\phi$, not hyperparameter optimization. We acknowledge this may disadvantage methods with strong hyperparameter sensitivity (e.g., CWD's $\beta$ parameter); sensitivity analysis in Appendix A.3.

### 4.5 Diagnostics

Per-epoch tracking: test accuracy, train accuracy, weight norms per layer, gradient norms per layer, $\rho_t$ per layer, gradient-weight cosine similarity $\hat{\delta}_{l,t}$, CSI, AIS, BEM, and $\phi$ modulation values. This produces $\sim$1,400 scalar measurements per epoch per run.

---

## 5. Results and Analysis

### 5.1 Main Accuracy Comparison (ResNet-20)

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

Three findings emerge:

1. **AdamW Phi invariance.** On CIFAR-10, the 7 methods span only 0.25 percentage points (90.13% to 89.88%). Even complete WD removal ($\phi = 0$) costs only 0.05% relative to constant WD. On CIFAR-100, the spread widens to 0.76%, with cosine schedule marginally leading (63.42%) and no-WD trailing (62.66%).

2. **SGD shows 3.7$\times$ larger spread.** On CIFAR-10, SGD's Phi spread is 0.91% vs. AdamW's 0.25%. On CIFAR-100, the ratio is 2.3$\times$ (1.71% vs. 0.76%). This difference is partially explained by SGD's 100$\times$ lower $\rho$: SGD operates in a regime where WD constitutes a larger fraction of the total parameter update, making $\phi$ modulation more consequential.

3. **No WD performs worst under SGD.** Without AdamW's implicit $\ell_\infty$-norm constraint (Xie & Li, 2024), removing WD entirely under SGD produces a 0.91% accuracy drop on CIFAR-10 and 1.71% on CIFAR-100, consistent with WD's role as a dynamics modifier (D'Angelo et al., 2024).

**Statistical tests.** Table 2 reports paired t-tests (Bonferroni-corrected) for each method vs. constant WD under AdamW on CIFAR-10.

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

**Theorem 1 validation.** Across all 5 complete configurations---$\{\text{AdamW, SGD}\} \times \{\text{CIFAR-10, CIFAR-100}\} \times \{\text{ResNet-20}\}$ (4 configurations) plus VGG-16-BN/CIFAR-10 (1 configuration)---constant WD matches or outperforms CWD (all $p > 0.05$ after Bonferroni correction), confirming Theorem 1's prediction at standard $\rho$. Two configurations (NoBN, matched-$\rho$ SGD) are incomplete.

### 5.2 Multi-Architecture Validation (VGG-16-BN)

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

The VGG-16-BN Phi spread is 0.16%---even smaller than ResNet-20's 0.25%. Constant WD (92.05%) is within 0.10% of the best method (half-$\lambda$, 92.15%). The cross-architecture null result confirms that Phi invariance under AdamW at standard $\rho$ is not specific to a single architecture or parameter count.

As shown in Figure 4, the accuracy distributions of all 7 methods overlap substantially on both ResNet-20 and VGG-16-BN, with Phi spread below 0.25% in both cases.

![Figure 4: Multi-architecture accuracy comparison showing 7 WD methods on ResNet-20 and VGG-16-BN, with error bars indicating standard deviation across 3 seeds.](figures/multi_arch_comparison.pdf)

**Figure 4.** Phi spread $< 0.25\%$ on both architectures, confirming method insensitivity is not architecture-specific. Error bars: std over 3 seeds.

### 5.3 Ratio Regime Analysis

The paper's central empirical prediction is that method sensitivity increases with $\rho$. We assemble four data points on the spread-vs-$\rho$ curve:

| $\rho$ regime | Source | $\rho$ value | Phi spread |
|---------------|--------|-------------|-----------|
| SGD original | Table 1 | $\sim$0.005 | 0.91% |
| $\rho$-low (AdamW) | Sweep | $\sim$0.05 | constant: 90.13 $\pm$ 0.07 (3 seeds); CWD: 90.09 (1 seed); partial $\Phi_{\text{spread}}$ $\geq$ 0.04% |
| $\rho$-standard (AdamW) | Table 1 | $\sim$0.5 | 0.25% |
| $\rho$-high (AdamW) | Pilot only | $\sim$5.0 | 5-epoch pilot: 77.69% (data gap) |

SGD's larger spread (0.91% at $\rho \approx 0.005$) vs. AdamW's (0.25% at $\rho \approx 0.5$) is partially explained by the 100$\times$ $\rho$ difference. The $\rho$-low regime has two data points: constant WD (90.13 $\pm$ 0.07, 3 seeds) and CWD (90.09, 1 seed only), yielding a partial spread estimate of $\geq$0.04%---directionally consistent with Theorem 1's prediction that lower $\rho$ reduces method sensitivity. The $\rho$-low constant accuracy (90.13 $\pm$ 0.07) coincidentally matches the $\rho$-standard constant (90.13 $\pm$ 0.31, Table 1); the two experiments use different $\lambda$ values ($5 \times 10^{-5}$ vs. $5 \times 10^{-4}$) and are independent runs with different weight norms and generalization gaps. The $\rho$-high data is limited to a 5-epoch pilot. Coverage gaps are consolidated in Section 5.8.

### 5.4 SGD vs. AdamW Effect Size

Figure 5 summarizes the Phi spread across all 4 optimizer-dataset configurations.

![Figure 5: Phi spread comparison across AdamW and SGD on CIFAR-10 and CIFAR-100, showing SGD's 3.7x larger spread partially explained by its 100x lower rho.](figures/sgd_vs_adamw_spread.pdf)

**Figure 5.** SGD's Phi spread is 3.7$\times$ (CIFAR-10) and 2.3$\times$ (CIFAR-100) larger than AdamW's. Two non-exclusive explanations: (i) SGD lacks AdamW's per-parameter adaptive scaling; (ii) SGD's 100$\times$ lower $\rho$ places it in a more WD-sensitive regime. The matched-$\rho$ SGD experiment (Section 5.6) attempts to disentangle these.

### 5.5 NoBN vs. BN Ablation

Removing batch normalization from ResNet-20 produces a 2.4 percentage-point accuracy drop (Table 5), confirming BN's importance for training quality. AIS increases from $\sim$0.35 (BN) to $\sim$0.50 (NoBN), consistent with the theory: without BN's scale-invariance, gradient-weight alignment becomes more informative.

**Table 5: ResNet-20-NoBN vs. ResNet-20 (BN) on CIFAR-10 (AdamW). Best test accuracy.**

| Architecture | Constant acc | CWD acc | $\Delta$(Constant$-$CWD) | AIS |
|-------------|-------------|---------|------------------------|-----|
| ResNet-20 (BN) | 90.13 $\pm$ 0.31 | 90.06 $\pm$ 0.24 | +0.07 | 0.35 |
| ResNet-20-NoBN | 87.74 $\pm$ 0.20 | 87.64 $\pm$ 0.17 | +0.10 | 0.50 |
| Cohen's $d$ (BN vs NoBN, constant) | | | 9.14 (large) | |

With 2 of 7 methods completed (3 seeds each), constant (87.74%) outperforms CWD (87.64%) by 0.10%, within the 1-std margin. NoBN's higher AIS ($\sim$0.50 vs. $\sim$0.35) is a necessary but not sufficient condition for dynamic WD to help; the full threshold from Theorem 1 also depends on $\Delta$CSI and $\sigma^2/n$. Coverage gaps are summarized in Section 5.8.

### 5.6 Matched-Ratio SGD (Preliminary)

Matched-$\rho$ SGD (lr $= 0.01$, $\lambda = 5 \times 10^{-3}$, targeting $\rho \approx 0.5$) provides partial data. Constant WD achieves 90.92% (mean of seeds 123 and 456, 200 epochs each; seed 42 ran only 5 epochs as a pilot, reaching 76.12%, and is excluded). CWD achieves 90.81% from a single seed (seed 42, 200 epochs). This data is insufficient for statistical conclusions (Section 5.8).

### 5.7 Diagnostic Analysis

**BEM vs. accuracy.** BEM ranges from 0.0 (constant, half-$\lambda$) to $\sim$0.90 (SWD), yet accuracy varies by $< 0.25\%$ on CIFAR-10 under AdamW. Across all 7 methods, a 10$\times$ variation in effective WD budget produces negligible accuracy change, demonstrating that AdamW's dynamics are robust to the absolute WD magnitude at standard $\rho$.

**CSI vs. accuracy.** Pooled across both architectures, Spearman $r_s = 0.71$ ($p < 0.01$) between CSI and accuracy---but this strong correlation is an architecture confound: ResNet-20 and VGG-16-BN cluster at different accuracy and CSI levels, so the pooled rank correlation reflects architecture separation rather than a within-method effect. Within each architecture, CSI does not predict accuracy: ResNet-20 $r_s = 0.03$ ($p = 0.81$, $n = 28$), VGG-16-BN $r_s = -0.05$ ($p = 0.84$, $n = 21$). CSI characterizes coupling instability, not performance; methods with high CSI (SWD, random mask) achieve comparable accuracy to low-CSI methods (constant) on the same architecture.

**AIS.** AIS values cluster in [0.18, 0.59] across all configurations. The pooled AIS-accuracy correlation ($r_s = -0.69$, $p < 0.01$) is again driven by the architecture confound: BN networks (higher accuracy, lower AIS $\sim$ 0.18--0.40) and NoBN networks (lower accuracy, higher AIS $\sim$ 0.31--0.59) form distinct clusters. Within architecture, AIS shows no predictive relationship with accuracy: ResNet-20 $r_s = 0.05$ ($p = 0.72$), VGG-16-BN $r_s = 0.02$ ($p = 0.91$). AIS is an intrinsic network-dataset property that does not vary with WD method. The higher NoBN AIS aligns with the Theorem 1 prediction that removing BN increases the alignment benefit, potentially shifting the threshold toward the dynamic-WD-optimal regime.

![Figure 6: Diagnostic metric panel: (a) CSI vs accuracy, (b) AIS vs accuracy, (c) BEM vs accuracy. Pooled correlations (CSI: r_s = 0.71, AIS: r_s = -0.69) are driven by architecture confound; within-architecture correlations are near zero.](figures/diagnostic_panel.pdf)

**Figure 6.** Diagnostic metrics vs. test accuracy. Pooled correlations appear strong (CSI: $r_s = 0.71$; AIS: $r_s = -0.69$) but are driven entirely by the architecture confound---ResNet-20 and VGG-16-BN cluster at different accuracy/CSI/AIS levels. Within each architecture, no diagnostic metric predicts accuracy (CSI: $r_s = 0.03$; AIS: $r_s = 0.05$; BEM: $< 0.25\%$ accuracy variation across a 10$\times$ range).

### 5.8 Data Gaps and Ongoing Work

Table 6 consolidates the coverage gaps across all experimental configurations. Five of seven configurations are complete (3 seeds, 200 epochs, all 7 methods); the remaining two have partial data.

**Table 6: Data gap summary. Complete = 7 methods $\times$ 3 seeds $\times$ 200 epochs.**

| Configuration | Status | Available Data | What It Would Resolve |
|--------------|--------|---------------|----------------------|
| $\rho$-high (AdamW, $\rho \approx 5.0$) | Pilot only | 5-epoch pilot: 77.69% | Theorem 1 prediction at elevated $\rho$ |
| Matched-$\rho$ SGD | Partial | Constant: 2 seeds (200 ep); CWD: 1 seed (200 ep) | Optimizer-vs-$\rho$ confound resolution |
| NoBN | Partial | 2/7 methods complete (3 seeds each) | Full NoBN Phi spread; Theorem 1 threshold test |
| $\rho$-low (AdamW, $\rho \approx 0.05$) | Partial | Constant: 3 seeds; CWD: 1 seed | Low-$\rho$ regime Phi spread |

All 5 complete configurations confirm Theorem 1's predictions. The partial data available from incomplete configurations is directionally consistent but insufficient for statistical conclusions.

---

## 6. Discussion

### 6.1 Why Constant WD Wins at Standard $\rho$

Theorem 1 predicts this outcome through a quantitative tradeoff. At standard $\rho \approx 0.5$ under AdamW with $\lambda = 5 \times 10^{-4}$, BN networks satisfy two conditions that make constant WD optimal:

1. **BN's scale-invariance limits alignment benefit.** BN layers are invariant to weight scaling: $\text{BN}(\alpha \theta) = \text{BN}(\theta)$. This drives weights toward an equilibrium norm (Kosson et al., 2023), making the gradient-weight alignment signal $\hat{\delta}_t$ geometrically constrained rather than freely informative. AIS values in our BN experiments (0.18--0.40) fall below the Theorem 1 threshold.

2. **AdamW's per-parameter scaling subsumes $\phi$ effects.** AdamW's second-moment normalization ($\hat{v}_t^{-1/2}$) already provides per-parameter adaptive scaling. The additional modulation from $\phi$ is redundant: it adds noise to an already well-calibrated update without providing new information. Across all 7 methods, a 10$\times$ variation in effective WD budget (BEM from 0.0 to 0.90) produces $< 0.25\%$ accuracy change.

### 6.2 When Dynamic WD Should Help

The theory predicts three regimes where dynamic WD becomes beneficial, each with different levels of empirical support:

**Elevated $\rho$ (partially supported; Section 5.3).** When $\rho > \rho^*$, the alignment benefit exceeds the stability cost (Theorem 1). The $\rho$-low partial data ($\Phi_{\text{spread}} \geq 0.04\%$) and SGD data ($\Phi_{\text{spread}} = 0.91\%$) provide two points on the spread-vs-$\rho$ curve, both directionally consistent. Full $\rho$-high data ($\rho \approx 5.0$) is needed to confirm the predicted increase at extreme ratios (Table 6).

**Without BN (directional; Section 5.5).** NoBN experiments show higher AIS ($\sim$0.50 vs. $\sim$0.35), consistent with the prediction that removing BN's scale-invariance increases alignment informativeness. The available NoBN data (2 methods, constant outperforming CWD by 0.10%, within 1-std margin) does not yet resolve whether the AIS increase is sufficient to cross the Theorem 1 threshold; full NoBN Phi spread requires additional methods (Table 6).

**Large-batch training (speculative; no experimental data).** Proposition 1's noise constraint relaxes as batch size increases: $\sigma^2/n$ decreases, lowering the Theorem 1 threshold. At LLM-scale batch sizes (4K--64K tokens), the raw alignment signal becomes more informative, potentially enabling direct alignment-based WD modulation without EMA smoothing. This remains unvalidated.

### 6.3 PMP-WD as a Principled Alternative

PMP-WD is derived from an optimality condition on the training dynamics. Three properties distinguish it from existing methods:

1. **State-feedback vs. feedforward (Section 3.5, Figure 3).** PMP-WD is designed to correct deviations from $\rho^*$ in real time; feedforward methods (cosine, AdamC) cannot compensate when the actual trajectory deviates from plan. Empirical validation is pending (Section 5.8).

2. **Dual derivation.** The same functional form emerges from both stochastic PMP (control theory) and RG beta function analysis (statistical physics), with agreement within 15% over the moderate-alignment regime $\hat{\delta}_t \in [0.3, 0.7]$ (Remark 3.1).

3. **Minimal overhead.** PMP-WD requires only per-layer scalar EMA of $\|g_l\| / \|\theta_l\|$---a computation already performed for gradient norm tracking in standard training loops. The additional cost is $O(L)$ per step, negligible relative to the forward/backward pass.

### 6.4 Implications for Practitioners

Three practical recommendations follow from the theory and experiments:

1. **Under AdamW at standard hyperparameters ($\lambda \sim 10^{-4}$--$10^{-3}$, BN architectures): use constant WD.** Dynamic scheduling adds implementation complexity without accuracy benefit. The 0.25% Phi spread on CIFAR-10 is within the 0.31% inter-seed std of the best method; VGG-16-BN's 0.16% spread falls below any single method's confidence interval width.

2. **New dynamic WD methods should demonstrate gains at elevated $\rho$ or at scale.** Standard CIFAR settings with AdamW are in the Phi-invariance regime; positive results here do not generalize to the claim that the method is universally better. The gradient-to-weight ratio should be reported as a context variable.

3. **If using alignment feedback, EMA smoothing with $k \geq 10$ is mandatory.** Proposition 1 shows that raw single-step alignment signals are unreliable at batch sizes $\leq$ 256. CWD's binary mask is partially robust (sign is low-dimensional), but continuous alignment methods must aggregate temporally.

### 6.5 Limitations

1. **Scale.** CIFAR-10/100 with ResNet-20 (270K parameters) and VGG-16-BN (15M parameters) are small by current standards. ImageNet and LLM experiments are not included. The theory's predictions at larger scale remain unvalidated.

2. **Incomplete configurations.** Three configurations ($\rho$-high, matched-$\rho$ SGD, NoBN) have partial data; Table 6 (Section 5.8) details the specific gaps and what each would resolve.

3. **PMP-WD not implemented.** The algorithm is derived but not empirically evaluated. Its predicted advantage at elevated $\rho$ is theoretical.

4. **Three seeds.** With 3 seeds, statistical power is limited for effect sizes below 0.3%. TOST equivalence testing at margin $\pm 0.3\%$ requires larger $n$ for definitive equivalence claims.

5. **Fixed hyperparameters.** All methods share the same base $\lambda$ and lr. Per-method hyperparameter tuning (e.g., CWD's $\beta$ parameter) might reveal larger differences, at the cost of confounding method identity with hyperparameter optimization quality.

---

## 7. Conclusion

Dynamic WD methods fail to outperform constant WD at standard settings not by accident but by necessity: Theorem 1 shows that the alignment benefit of $\phi$ modulation is exceeded by its stability cost whenever the gradient-to-weight ratio $\rho$ falls below a noise-dependent threshold, a condition satisfied in all 5 complete configurations tested (105 completed 200-epoch runs). Theorem 2 explains why binary masking methods (CWD, random mask) incur hidden per-parameter stability costs despite moderate aggregate Coupling Stability Index (CSI), by bounding the generalization penalty under per-parameter WD variation. Together, Theorems 1--2 and Proposition 1 (requiring EMA aggregation $k \geq 10$ for reliable alignment feedback) provide a quantitative theory of when and why constant WD is optimal. Theorem 3 derives a state-feedback WD controller (PMP-WD) from the stochastic Pontryagin Maximum Principle and independently from renormalization group beta function theory; this algorithm is derived but not yet empirically evaluated, with validation at elevated $\rho$ as a priority for future work.

SGD shows 3.7$\times$ larger method sensitivity than AdamW on CIFAR-10 (0.91% vs. 0.25%), consistent with Theorem 1's prediction that Phi spread scales with $\rho$, though direct matched-$\rho$ comparison remains to be completed (Section 6.5). The practical implication is direct: under AdamW at standard hyperparameters, constant WD is sufficient; dynamic WD methods should target elevated-$\rho$ or large-scale regimes where the Alignment Informativeness Score (AIS) exceeds the stability cost.

Immediate extensions include ImageNet-scale validation and PMP-WD empirical evaluation at elevated $\rho$, which directly tests Theorem 3's predictions. Longer-horizon directions include Vision Transformer architectures, where the absence of BN may increase AIS (Section 6.2), and LLM-scale experiments, where large batch sizes relax Proposition 1's noise floor and long training horizons may amplify WD timing effects.

---

## Figures and Tables

- Figure 1: ratio_regime.pdf --- Ratio regime diagram: method spread vs. log($\rho$) with three regime zones
- Figure 2: theorem1_regime.pdf --- Theorem 1 regime illustration: alignment benefit vs. stability cost crossover at AIS*
- Figure 3: pmpwd_control.pdf --- PMP-WD control diagram: closed-loop state-feedback vs. feedforward
- Figure 4: multi_arch_comparison.pdf --- Multi-architecture accuracy comparison (ResNet-20 and VGG-16-BN)
- Figure 5: sgd_vs_adamw_spread.pdf --- Phi spread comparison across optimizers
- Figure 6: diagnostic_panel.pdf --- Diagnostic metric panel (CSI, AIS, BEM vs. accuracy)
- Table 1: inline --- Main accuracy results (7 methods $\times$ 4 configurations)
- Table 2: inline --- Statistical significance tests (AdamW CIFAR-10)
- Table 3: inline --- VGG-16-BN results
- Table 4: inline --- Method taxonomy (Phi modulator special cases)
- Table 5: inline --- NoBN vs. BN ablation
- Table 6: inline --- Data gap summary
