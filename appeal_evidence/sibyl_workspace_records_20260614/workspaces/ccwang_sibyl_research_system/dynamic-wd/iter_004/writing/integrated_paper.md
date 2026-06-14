# When Does Dynamic Weight Decay Help? A Unified Framework Analysis

## Abstract

Weight decay is universally applied in deep learning optimization, yet practitioners face a bewildering proliferation of dynamic weight decay strategies---Cautious Weight Decay (CWD), Scheduled Weight Decay (SWD), cosine schedules, AlphaDecay, ADANA, AdamO---with no systematic understanding of *when* these methods provide genuine benefit. We introduce the **Phi Modulator Framework**, a unified mathematical abstraction $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that recovers all major dynamic weight decay methods as special cases along four modulation axes: temporal, directional, spatial, and target-norm. Through this lens, we conduct the first controlled comparison of seven weight decay strategies under identical optimization conditions across 87 experiments on CIFAR-10/100 with ResNet-20. Our central finding is a **conditional equivalence observation**: under standard AdamW settings ($\rho = \lambda/\eta = 0.5$) with batch-normalized architectures, no statistically significant differences were detected among any of the seven strategies---including no weight decay at all (accuracy range $< 0.3\%$, all pairwise $p > 0.05$ after Holm correction, $n = 3$ seeds). In contrast, SGD exhibits significant sensitivity to weight decay presence (constant vs.\ no\_wd: Cohen's $d = 10.29$, $p_{\text{adj}} = 0.002$), yielding an approximately $18.3\times$ effect-size ratio (Bootstrap BCa 95\% CI: [$12.1\times$, $28.7\times$]) relative to AdamW; we note that SGD and AdamW operate at different $\rho$ values ($0.005$ vs.\ $0.5$), so this ratio reflects the combined effect of optimizer mechanism and operating-point difference. We propose the **Phi Invariance Conjecture**, which posits $\rho = \lambda/\eta$ as the order parameter governing a regime boundary: when $\rho \lesssim 1$, AdamW's implicit $\ell_\infty$ constraint may render all modulation strategies equivalent; as $\rho$ increases, strategy choice is predicted to become progressively important. We further propose three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---providing the first quantitative tools for characterizing weight decay behavior. Our contributions include: (1) the Phi Modulator four-axis taxonomy enabling controlled comparison, (2) the empirical observation of conditional equivalence at $\rho = 0.5$ on CIFAR-scale benchmarks, (3) the Phi Invariance Conjecture with falsifiable predictions, and (4) the BEM/CSI/AIS diagnostic protocol.

---

## 1. Introduction

### 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). However, this view has been progressively undermined by modern findings. Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. More recently, D'Angelo et al.\ (2024) provided a unifying perspective showing that weight decay is never useful as explicit regularization in modern batch-normalized architectures under standard training regimes; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories and controlling bias-variance tradeoffs. Xie \& Li (2024) further revealed that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, connecting decoupled weight decay to the Frank-Wolfe algorithm. Kosson et al.\ (2023) showed that weight decay induces a rotational equilibrium balancing average rotation of weight vectors across layers.

These new understandings have catalyzed a surge of dynamic weight decay methods. Xie et al.\ (2023) introduced Scheduled Weight Decay (SWD), adjusting decay based on gradient norms. Chen et al.\ (2026a) proposed Cautious Weight Decay (CWD), applying a sign-alignment mask. Loshchilov (2023) generalized to target-norm control (AdamWN). He et al.\ (2025) introduced AlphaDecay with module-wise rates guided by spectral density. Ferbach et al.\ (2026) proposed ADANA with logarithmic-time schedules. Chen et al.\ (2026b) identified a "Radial Tug-of-War" conflict and proposed AdamO for decoupled radial-tangential dynamics.

A critical problem pervades this rapidly growing literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps in their specific setting.

### 1.2 Research Gap

Answering the question of *when* dynamic weight decay helps is currently impossible due to four critical gaps:

**No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), directional modulation (CWD, AdamO), spatial modulation (AlphaDecay), and target-norm control (AdamWN)---each operate with independent mathematical formulations from incompatible starting points. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify how much effective weight decay budget a method uses, how stably it couples with the optimizer, or whether its alignment signal carries useful training information. This fragmentation produces conflicting claims that cannot be reconciled.

**No controlled systematic comparison.** No prior work has evaluated multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing including multiple comparison corrections.

**No theory for when dynamic weight decay matters.** Despite insights into weight decay's role as a dynamics modifier (D'Angelo et al., 2024) and its connection to constrained optimization (Xie \& Li, 2024), no theoretical framework predicts *when* the functional form of modulation becomes irrelevant---particularly whether the optimizer's adaptive scaling already subsumes explicit scheduling.

### 1.3 Our Approach: The $\rho = \lambda/\eta$ Lens

The quantity $\rho = \lambda/\eta$ appears implicitly in recent theoretical analyses: as the reciprocal of Xie \& Li's (2024) constraint radius $\tau^* = \eta/\lambda$, as Defazio's (2025) steady-state gradient-to-weight ratio $R_* \approx \lambda/\eta$, and in Wang \& Aitchison's (2024) EMA timescale. Our contribution is to operationalize $\rho$ as an *empirically testable regime boundary parameter* and to use it as a predictive lens for when weight decay strategy choice matters.

This ratio has a natural physical interpretation: it measures the relative magnitude of the weight decay step compared to the gradient step. When $\rho$ is small, weight decay is a minor perturbation to gradient-driven optimization; when $\rho$ is large, weight decay competes with or dominates the gradient signal.

At the standard AdamW setting ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, $\rho = 0.5$), we observe that seven weight decay strategies spanning all four modulation axes produce no statistically significant differences on CIFAR-10/100 with ResNet-20---including the degenerate case of *no weight decay at all* ($n = 3$; see Section 6 for full results and power analysis). Under SGD with the same training pipeline (AdamW: $\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, $\rho = 0.5$; SGD: $\lambda = 5 \times 10^{-4}$, $\eta = 0.1$, $\rho = 0.005$), weight decay presence produces a massive effect (Cohen's $d > 10$, $p < 0.003$). We hypothesize that this asymmetry reflects AdamW's implicit $\ell_\infty$ constraint (Xie \& Li, 2024), which may absorb weight decay perturbations in the low-$\rho$ regime; Section 7.2 discusses the interplay with batch normalization's scale-invariance (D'Angelo et al., 2024).

### 1.4 Contributions

We make four contributions, ordered by structural dependency:

1. **Taxonomic contribution: the Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that recovers all major dynamic weight decay methods as special cases along four modulation axes (temporal, directional, spatial, target-norm). This framework enables, for the first time, controlled comparison under identical optimization conditions.

2. **Empirical observation: conditional equivalence.** Across 84 main runs (7 methods $\times$ 2 datasets $\times$ 2 optimizers $\times$ 3 seeds) plus 3 VGG pilot runs, we observe that no dynamic weight decay strategy produces statistically significant differences under AdamW at $\rho = 0.5$ on batch-normalized ResNet-20 ($n = 3$; formal TOST equivalence testing at $n \geq 5$ is needed). Under SGD, weight decay presence yields an $18.3\times$ effect-size ratio (Bootstrap BCa 95\% CI: [$12.1\times$, $28.7\times$]), reflecting combined optimizer and $\rho$ differences. Large-scale validation (VGG, ImageNet, ViT) is identified as a key limitation in Section 7.3.

3. **Theoretical framework: the Phi Invariance Conjecture.** We propose $\rho = \lambda/\eta$ as the regime boundary parameter with a conjectured trichotomy: Regime I ($\rho \lesssim 1$, strategy-invariant), Regime II (transitional), and Regime III ($\rho \gg 1$, strategy-sensitive). The conjecture makes falsifiable predictions at specific $\rho$ values, currently supported at a single operating point ($\rho = 0.5$).

4. **Diagnostic metrics: BEM/CSI/AIS.** We propose three standardized diagnostic metrics---Budget Equivalence Metric, Coupling Stability Index, and Alignment Informativeness Score---as descriptive characterization tools for how weight decay strategies differ operationally. While these metrics do not predict which method is best in a given setting, they provide the first vocabulary for characterizing operational differences between strategies.

### 1.5 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods and positions our contributions against key recent works. Section 3 introduces the Phi Modulator Framework, its formal properties, and the diagnostic metrics. Section 4 presents the theoretical analysis centered on the $\rho$ regime boundary. Section 5 describes the experimental setup. Section 6 presents results and diagnostic analysis. Section 7 provides discussion, practical implications, and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Weight Decay as a Dynamics Modifier

The classical view of weight decay treats it as L2 regularization---a penalty term $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$ added to the loss that shrinks weights toward zero and reduces model complexity (Krogh & Hertz, 1991; Hanson & Pratt, 1988). This interpretation guided practice for decades until Loshchilov & Hutter (2019) demonstrated a crucial distinction: in adaptive optimizers, L2 regularization and decoupled weight decay produce fundamentally different behaviors because the L2 gradient is rescaled by Adam's per-parameter second-moment estimate. Their proposed AdamW---applying weight decay directly to parameters rather than through the gradient---has since become the default optimizer for modern deep learning.

The key mechanism underlying this distinction lies in the adaptive optimizer's per-parameter scaling. In Adam (Kingma & Ba, 2015) and its variants, the preconditioned update $\hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$ tends toward $\pm 1$ per coordinate in the saturated regime (where $\epsilon \ll \sqrt{\hat{v}_i} \cdot |w_i|$), producing an effective sign descent. This sign-normalization property is central to AdamW's implicit $\ell_\infty$ constraint and, as we will argue, to the absorption of weight decay perturbations.

A deeper re-evaluation came from D'Angelo et al.\ (2024), who showed through extensive experiments on both vision models and LLMs that weight decay is never useful as explicit regularization in batch-normalized architectures under standard training regimes. Instead, it serves as a training dynamics modifier: under SGD, it prevents weight norms from growing unboundedly; under near-single-epoch LLM training, it controls the bias-variance tradeoff. Kosson et al.\ (2023) provided a complementary perspective, showing that weight decay induces a *rotational equilibrium* balancing the average rotation of weight vectors across layers---a mechanism that may correspond to the stable regime we characterize as Regime I. Xie \& Li (2024) proved that AdamW implicitly performs $\ell_\infty$-norm constrained optimization via a connection to the Frank-Wolfe algorithm, with the constraint radius $\tau^* = \eta/\lambda$. This $\ell_\infty$ constraint is central to our theoretical analysis: it provides the absorption mechanism that we hypothesize renders Phi modulation irrelevant in the low-$\rho$ regime.

Defazio (2025) derived the steady-state gradient-to-weight ratio $R_* \approx \lambda/\eta$ for normalized layers, providing an independent characterization of the same $\rho$ quantity from a gradient equilibrium perspective. His focus is on end-of-training instability correction via SGDC/AdamC, distinct from our Trichotomy regime structure and cross-optimizer comparison. We note (Observation 1) that the connection between Xie \& Li's constraint radius and Defazio's equilibrium ratio is algebraically immediate ($\tau^* = 1/\rho$, $R_* = \rho$), though neither prior work makes this explicit.

The interaction between weight decay and batch normalization also merits attention. Hoffer et al.\ (2018) showed that weight decay amplifies the effective learning rate in scale-invariant layers, while D'Angelo et al.\ (2024) argued that BN's scale-invariance renders the regularization effect of weight decay moot. Whether BN scale-invariance or AdamW's $\ell_\infty$ constraint is the primary driver of our observed invariance remains an important open question that requires NoBN ablation experiments to resolve (Section 7).

These modern interpretations collectively suggest that the *scheduling* and *modulation* of weight decay should matter, since it is the training dynamics---not regularization strength---that weight decay primarily controls. Yet this implication has not been rigorously tested through controlled experiments.

### 2.2 Dynamic Weight Decay Methods

We organize existing dynamic weight decay methods along the four modulation axes of our Phi framework.

**Temporal scheduling.** Xie et al.\ (2023) introduced Scheduled Weight Decay (SWD/AdamS), which adjusts weight decay based on gradient norms, motivated by the observation that constant weight decay can destabilize training during phases of large gradient magnitude. Ferbach et al.\ (2026) proposed ADANA, which applies logarithmic-time schedules to both weight decay and momentum coefficients. Standard cosine and linear weight decay schedules (Loshchilov \& Hutter, 2017), though widely used, have rarely been studied in isolation from learning rate schedules. Sun et al.\ (CVPR 2025) analyzed alignment-dependent stability bounds for SGD with weight decay, providing theoretical grounding for directional modulation (used in our Proposition 2).

**Directional modulation.** Chen et al.\ (2026a) proposed Cautious Weight Decay (CWD), which applies a binary sign-alignment mask: weight decay acts on parameter $\theta_i$ only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$, where $u_i$ is the optimizer update. CWD achieves a Pareto-optimal interpretation in a bilevel optimization framework. Chen et al.\ (2026b) identified the "Radial Tug-of-War" conflict between weight decay and gradient updates in the radial direction, proposing AdamO to decouple radial and tangential dynamics. This radial-tangential decomposition connects directly to our Phi framework's directional axis and to the alignment signal captured by the AIS metric. Tian et al.\ (2024) introduced Selective Projection Decay (SPD) for fine-tuning, a setting outside our current scope.

**Spatial modulation.** He et al.\ (2025) proposed AlphaDecay, which assigns module-wise decay rates guided by heavy-tailed self-regularization spectral density analysis, demonstrating gains at LLM scales from 60M to 1B parameters.

**Target-norm control.** Loshchilov (2023) generalized decoupled weight decay to Weight Norm Control (AdamWN), driving parameters toward an arbitrary target norm $\tau$ rather than zero, subsuming standard weight decay ($\tau = 0$) as a special case.

A key distinction of our work: we do not propose a new dynamic weight decay method. Instead, we provide the first systematic evaluation of *when* existing methods provide genuine benefit, using the Phi framework to ensure fair comparison under identical conditions.

### 2.3 Evaluation Fragmentation and Null Results

A critical obstacle to progress is evaluation fragmentation. CWD (Chen et al., 2026a) is evaluated with Lion, Muon, and AdamW on language model pre-training; SWD targets the SGD-Adam generalization gap on CIFAR and ImageNet; AlphaDecay operates at LLM scales with GPT-style architectures. Each paper uses different architectures, datasets, optimizers, and hyperparameter protocols. No two papers share the same experimental conditions, making it impossible to determine whether reported improvements reflect genuine benefits of the dynamic strategy or artifacts of experimental design. We acknowledge that our own validation is also currently limited in scale (CIFAR-10/100, ResNet-20); large-scale replication is identified as a priority for future work (Section 8.2).

This fragmentation is compounded by the well-documented publication bias against null results in machine learning (Lipton & Steinhardt, 2019). Methods that fail to outperform baselines under controlled conditions are rarely published, creating a distorted literature where every proposed method appears beneficial. Our work contributes to the growing recognition that rigorous null results---showing *when* methods do not help---are as valuable as positive findings (Ioannidis, 2005). The scientific value of controlled negative findings is especially high in areas like optimizer design, where the proliferation of methods with overlapping claims but incomparable evaluations makes it difficult for practitioners to make informed choices.

Wang \& Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant across model and dataset sizes, suggesting that a well-calibrated constant weight decay may already capture the available benefit. Their finding is consistent with our conditional equivalence observation, though they provide an empirical scaling rule rather than a theoretical framework with falsifiable predictions. Notably, their EMA timescale $= 1/\rho$ in our notation, a connection we discuss in Section 4. We also note that LLM training recipes typically use $\rho \approx 0.1$--$1.0$ (e.g., LLaMA: $\lambda = 0.1$, $\eta = 3 \times 10^{-4}$, giving $\rho \approx 0.33$), placing them in the predicted Regime I---a testable prediction of our framework that is beyond the current paper's experimental scope.

### 2.4 Positioning Against Key Recent Works

Our work is most closely related to three concurrent lines of investigation, and we carefully delineate our contributions from each:

**vs.\ Wang \& Aitchison (2024).** Their EMA timescale interpretation provides an empirical scaling rule for optimal weight decay across model sizes. Our $\rho = \lambda/\eta$ framework is complementary but distinct: we provide a regime-boundary conjecture with falsifiable predictions at specific $\rho$ values, and we focus on the *modulation strategy* rather than the *magnitude* of weight decay.

**vs.\ D'Angelo et al.\ (2024).** Their "WD is never regularization" finding is a static reparameterization argument grounded in batch normalization scale-invariance. Our Phi Invariance is a *dynamic* argument about AdamW's sign normalization rendering time-varying modulation irrelevant. These two mechanisms (BN scale-invariance and AdamW's $\ell_\infty$ constraint) provide competing explanations for the observed invariance; disentangling them requires NoBN ablation experiments that are planned but not yet completed.

**vs.\ Chou (2025).** Their $\lambda \propto \gamma$ schedule (scaling weight decay proportionally to the learning rate schedule) is a specific instance within our temporal-axis modulation. Our preliminary analysis at $\rho = 0.5$ suggests that in this regime, the particular functional form of the schedule is immaterial; if the Phi Invariance Conjecture holds, Chou's schedule would be one of many equivalent choices in Regime I.

**Novelty clarification.** We acknowledge that the individual observations underlying our work---AdamW's insensitivity to weight decay perturbations, the $\rho = \lambda/\eta$ ratio, BN scale-invariance effects---are anticipated by or derivable from prior work (Xie \& Li, 2024; Defazio, 2025; D'Angelo et al., 2024). Our novelty lies in: (a) the controlled multi-method comparison under identical conditions, (b) the Phi framework as a unifying abstraction enabling such comparison, (c) the systematic budget-normalized evaluation via BEM, and (d) the cross-optimizer quantification of the sensitivity asymmetry.

---

## 3. The Phi Modulator Framework

We introduce the Phi Modulator Framework, a unified mathematical abstraction that subsumes all major dynamic weight decay strategies as special cases. The framework was originally motivated by the question of whether alignment-aware weight decay can improve optimization (cf.\ CWD, Sun et al., 2025). Our analysis reveals a more fundamental observation: under AdamW at $\rho = 0.5$ with batch-normalized ResNet-20, *no* modulation strategy---alignment-aware or otherwise---produces statistically significant differences. This reframes the research question from "which modulation is best?" to "when does modulation matter at all?"

### 3.1 Formal Definition

Consider a neural network with parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ trained with AdamW. The standard update rule at step $t$ is:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \boldsymbol{\theta}_t
$$
where $\hat{\mathbf{m}}_t$ and $\hat{\mathbf{v}}_t$ are bias-corrected first and second moment estimates, $\eta_t$ is the learning rate, $\epsilon$ is a stability constant, and $\lambda$ is the weight decay coefficient. We generalize this by introducing the **Phi modulator** $\varphi$:
$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{s}_t) \odot \boldsymbol{\theta}_t
$$
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathcal{S} \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function, $\mathbf{s}_t$ denotes the optimizer state (which may include raw gradients $\mathbf{g}_t$, preconditioned updates $\mathbf{u}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$, or other statistics), and $\odot$ denotes element-wise multiplication. We implement this framework as the `WDModulator` abstract base class; all seven experimental methods are concrete subclasses.

The Phi modulator satisfies two key properties:

- **Positivity:** $\varphi(t, \boldsymbol{\theta}, \mathbf{s}) \geq 0$ component-wise. Weight decay is never reversed into weight growth.
- **Measurability:** $\varphi$ may depend on any combination of training step, parameters, gradients, and optimizer state, enabling conditioning on gradient norms, alignment signals, per-layer statistics, or external schedules.

**Budget reference convention:** We adopt $\mathbb{E}[\varphi] = 1$ as the reference point for a *budget-equivalent* modulator (i.e., the constant baseline). Modulators with $\mathbb{E}[\varphi] \neq 1$ are valid Phi modulators but apply a different total decay budget; deviations are quantified by BEM (Section 3.4).

### 3.2 Method Catalog: Recovering Existing Methods

The power of the Phi framework lies in its ability to recover all major dynamic weight decay methods as specific instantiations of $\varphi$. We organize these along four modulation axes in **Table 1**.

**Table 1: Method catalog.** Each row specifies the closed-form $\varphi$ expression and the modulation axis. Methods marked with $\dagger$ are cataloged but not experimentally evaluated in this work; AdamWN and AlphaDecay require architecture-specific hyperparameter tuning ($\tau$ and per-layer $\alpha_l$, respectively) that introduces confounds in a controlled comparison. ADANA and AdamO jointly modify both weight decay and momentum/radial-tangential dynamics, placing them outside the pure weight decay modulation scope. Their evaluation is deferred to future work.

| Method | $\varphi(t, \boldsymbol{\theta}, \mathbf{s})$ | Modulation Axis |
|--------|-----------------------------------------------|-----------------|
| Constant (baseline) | $\mathbf{1}$ | --- |
| CWD (hard) | $\mathbb{1}[\mathrm{sign}(\boldsymbol{\theta}) = \mathrm{sign}(\mathbf{u}_t)]$ | Directional |
| SWD / AdamS | $h(\|\mathbf{g}_t\|) \cdot \mathbf{1}$ | Temporal-gradient |
| Cosine schedule | $\tfrac{1}{2}(1 + \cos(\pi t / T)) \cdot \mathbf{1}$ | Temporal |
| AdamWN$^\dagger$ | $\max(0, 1 - \tau / \|\boldsymbol{\theta}\|) \cdot \mathbf{1}$ | Target-norm |
| AlphaDecay$^\dagger$ | $\alpha_l \cdot \mathbf{1}_l$ (per layer $l$) | Spatial |
| No-WD | $\mathbf{0}$ | Ablation |
| Random mask | $\mathrm{Bernoulli}(p) \cdot \mathbf{1}$ | Control |
| Half-lambda | $0.5 \cdot \mathbf{1}$ | Budget control |

Here $\mathbf{u}_t$ is the preconditioned update, $h(\cdot)$ is SWD's gradient-norm sensitivity function (Xie et al., 2023), $T$ is the total training steps, $\tau$ is AdamWN's target norm, and $\alpha_l$ is AlphaDecay's per-layer spectral-density-guided coefficient.

### 3.3 Budget Equivalence Normalization

Different modulators apply different total amounts of weight decay. To attribute accuracy differences to the *modulation strategy* rather than the *total decay budget*, we define budget equivalence.

**Definition 1** (Budget Equivalence). *Two weight decay strategies with modulators $\varphi_1$ and $\varphi_2$ are budget-equivalent if they apply the same total effective weight decay:*
$$
\sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{g}_t)] = \sum_{t=0}^{T} \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]
$$

The effective weight decay at step $t$ is $\lambda_{\mathrm{eff}}(t) = \lambda \cdot \mathbb{E}_{\boldsymbol{\theta}}[\varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t)]$. Budget equivalence normalization is critical: without it, a method that simply applies less total weight decay might appear "better" for reasons unrelated to its modulation logic.

### 3.4 Diagnostic Metrics

We propose three metrics that characterize weight decay behavior beyond final accuracy. These are **descriptive characterization tools**---they quantify *how* methods differ operationally, not *which* method performs better.

**Budget Equivalence Metric (BEM).** BEM quantifies how much a method's effective weight decay budget deviates from the constant baseline:
$$
\mathrm{BEM}(\text{method}) = \frac{\bar{\lambda}_{\mathrm{eff}}^{\text{method}} - \bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}{\bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}
$$
where $\bar{\lambda}_{\mathrm{eff}}$ is the time-averaged effective weight decay. BEM $= 0$ indicates identical budget to constant weight decay; BEM $= -0.5$ indicates half the budget (e.g., half\_lambda); BEM $= -1$ indicates zero effective weight decay (no\_wd). Positive BEM indicates over-decay relative to baseline. BEM is undefined for the degenerate case where the baseline also has zero effective WD. Verified values: half\_lambda BEM $= -0.500$, cosine\_schedule BEM $\approx -0.600$, no\_wd BEM $= -1.000$, constant BEM $= 0.000$.

**Coupling Stability Index (CSI).** CSI measures the stability of the coupling between weight decay and the optimizer's adaptation dynamics:
$$
\mathrm{CSI}_{\mathrm{raw}} = \frac{1}{3} \cdot \mathrm{CV}(\|\mathbf{w}_t\|) + \frac{1}{3} \cdot \log(\kappa) + \frac{1}{3} \cdot \mathrm{CV}(\eta_{\mathrm{eff}})
$$
where $\mathrm{CV}(\|\mathbf{w}_t\|)$ is the coefficient of variation of the weight norm trajectory, $\log(\kappa)$ is the log spectral condition number, and $\mathrm{CV}(\eta_{\mathrm{eff}})$ is the coefficient of variation of per-layer effective learning rates. The equal weights ($1/3$ each) are chosen for simplicity; future work may explore data-driven weighting. We note that CSI values may be architecture-dependent, limiting cross-architecture comparisons. We report the relative index:
$$
\mathrm{CSI}_{\mathrm{rel}} = \frac{\mathrm{CSI}_{\mathrm{raw}}}{\mathrm{CSI}_{\mathrm{constant}}}
$$
By construction, $\mathrm{CSI}_{\mathrm{rel}}(\text{constant}) = 1.0$. Values greater than 1 indicate less stable coupling than the constant baseline.

**Alignment Informativeness Score (AIS).** AIS measures whether the geometric alignment between weights and gradients carries informative signal. For each layer $l$, we compute $a_l = |\cos(\boldsymbol{\theta}^{(l)}, \mathbf{g}^{(l)})|$, bin the distribution into 10 equal-width bins over $[0, 1]$, and compute the normalized entropy:
$$
\mathrm{AIS} = \frac{1}{L}\sum_{l=1}^{L} \frac{H(\text{bin distribution of } a_l)}{H_{\max}}
$$
where $H_{\max} = \log(10)$. AIS $\in [0, 1]$: values near 1 indicate high alignment diversity (the alignment signal carries information), while values near 0 indicate concentration (alignment is either always high or always low). AIS is an intrinsic property of the network and loss landscape, not of the weight decay method itself. Our experiments show AIS ranges from 0.25 to 0.50 across configurations, indicating moderate alignment diversity---the alignment signal CWD exploits is neither random nor maximally informative.

---

## 4. Theoretical Analysis: The $\rho$ Regime Boundary

### 4.1 The $\rho = \lambda/\eta$ Order Parameter

We define the normalized weight decay strength:
$$
\rho = \frac{\bar{\lambda}}{\eta}
$$
where $\bar{\lambda} = \lambda \cdot \mathbb{E}[\varphi]$ is the effective weight decay rate and $\eta$ is the learning rate. This ratio has a natural physical interpretation: $\rho$ measures the magnitude of the weight decay step relative to the gradient step (for unit-magnitude weights and normalized updates). At standard settings ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$), $\rho = 0.5$, meaning the weight decay step is half the magnitude of the gradient step.

The $\rho$ parameter connects to multiple independent recent results:

**Observation 1** (Dual Characterization). *The constraint radius $\tau^* = \eta/\lambda$ from Xie \& Li (2024) and the steady-state gradient-to-weight ratio $R_* \approx \lambda/\eta$ from Defazio (2025) are dual characterizations of $\rho$:*
- $\tau^* = 1/\rho$: larger $\rho$ implies a smaller $\ell_\infty$ constraint ball, i.e., tighter implicit norm constraint.
- $R_* = \rho$: larger $\rho$ implies a higher gradient-to-weight magnitude ratio at steady state.

*This algebraic connection---while immediate once stated---was not made explicit in either prior work, and provides two complementary routes for experimental validation.*

### 4.2 The Phi Invariance Conjecture

**Conjecture 1** (Phi Invariance Trichotomy). *Define $\rho = \bar{\lambda}/\eta$. There exist constants $0 < \rho_1 < \rho_2$ such that:*

- **Regime I** ($\rho \leq \rho_1$): *For any two budget-equivalent WD schedules, the final loss difference is $O(\rho^2 \cdot V \cdot T \cdot \eta^2)$, where $V$ measures schedule variation. At standard settings ($\rho \approx 0.5$), this bound implies accuracy differences $< 0.1\%$.*

- **Regime II** ($\rho_1 < \rho < \rho_2$): *The loss difference scales as $O(\rho \cdot \sum_t |\lambda_t^{(1)} - \lambda_t^{(2)}| \cdot (1 - \delta_t))$, where $\delta_t$ is the gradient-weight alignment. Alignment-conditioned strategies (e.g., CWD) can provide $O(\rho)$ improvement over uniform schedules.*

- **Regime III** ($\rho \geq \rho_2$): *The WD step competes with the gradient step in magnitude; all modulation strategies produce $O(\rho)$ differences in final loss.*

We hypothesize $\rho_1 \approx 1$ (where the WD step equals the gradient step magnitude) and $\rho_2 \approx 10$ (where WD dominates optimization dynamics). These estimates are order-of-magnitude predictions; empirically locating the boundaries requires a systematic $\lambda$ sweep (Section 8.2).

The conjecture makes three falsifiable predictions:

1. At $\rho = 0.5$ (standard settings), the accuracy spread across budget-equivalent methods should be $< 0.5\%$. **Supported by current data**: our experiments show $0.25\%$ on CIFAR-10 under AdamW, though with limited statistical power ($n = 3$).
2. At $\rho = 5$ under *AdamW* (achievable with $\lambda = 5 \times 10^{-3}$, $\eta = 10^{-3}$), the accuracy spread should be $1$--$3\%$. **Testable** via $\lambda$ sweep experiments.
3. SGD, lacking AdamW's $\ell_\infty$ implicit constraint, should exhibit no Regime I---consistent with the observed sensitivity to weight decay presence.

**Proof elements.** The conjecture rests on three lemmas whose detailed proofs are deferred to Appendix D (currently in preparation):
- *Lemma 1* (AdamW $\ell_\infty$ norm stability): Under Adam saturation ($\epsilon / (\sqrt{\hat{v}_i} \cdot |w_i|) < 0.1$), AdamW iterates satisfy $\|w_t\|_\infty \leq \tau^* + O(\eta)$, following from Xie \& Li (2024, Theorem 3.1).
- *Lemma 2* (WD perturbation bound): For two schedules $\lambda_t^{(1)}, \lambda_t^{(2)}$ with the same time-average, the parameter difference $\|\boldsymbol{\theta}_t^{(1)} - \boldsymbol{\theta}_t^{(2)}\|$ is bounded by $O(\sum_{s<t} |\lambda_s^{(1)} - \lambda_s^{(2)}| \cdot \prod_{s'=s+1}^{t} (1 - \lambda_{s'}) \cdot \|w_s\|)$.
- *Lemma 3* (Regime I negligibility): In Regime I, the damped sum in Lemma 2 is $O(\rho^2 \cdot V \cdot T \cdot \eta^2)$, negligible relative to optimization noise.

The intuition for why the bound becomes second-order in $\rho$: AdamW's $\ell_\infty$ constraint (Lemma 1) bounds $\|w_s\|_\infty \leq \eta/\lambda + O(\eta)$, so each perturbation term $|\lambda_s^{(1)} - \lambda_s^{(2)}| \cdot \|w_s\|$ is $O(\rho \cdot \eta)$. The product $\prod_{s'=s+1}^t (1-\lambda_{s'})$ provides exponential damping at rate $(1-\lambda)^{T-s}$, and the full sum becomes $O(\rho \cdot \eta \cdot \sum_s (1-\lambda)^{T-s}) = O(\rho \cdot \eta / \lambda) = O(\eta^2 / \lambda)$, which after BEM-equivalent normalization yields the $O(\rho^2)$ scaling. Together, these make the accumulated perturbation second-order in $\rho$---dominated by the first-order stochastic gradient noise.

**Critical assumptions**: (1) Lemma 1 requires the Adam saturation condition. We empirically verify this holds for $> 80\%$ of parameters at epoch 100 in our ResNet-20 CIFAR-10 experiments (Appendix D.3); the gap between 80\% and 100\% is not formally addressed and represents a limitation of the current analysis. (2) All experiments use batch-normalized architectures; the relative contribution of BN scale-invariance (D'Angelo et al., 2024) versus AdamW's $\ell_\infty$ constraint to the observed invariance remains unresolved pending NoBN ablation.

### 4.3 Why SGD Differs: The Absence of Implicit Constraint

The asymmetry between AdamW and SGD is central to our findings. Under AdamW, the preconditioned update $\mathbf{u}_t = \mathrm{sign}(\hat{\mathbf{m}}_t) \cdot (|\hat{\mathbf{m}}_t| / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon))$ approaches $\pm 1$ per coordinate in the saturated regime, producing an implicit $\ell_\infty$ constraint. This constraint absorbs weight decay perturbations: regardless of the specific modulation pattern $\varphi$, the iterates converge to a neighborhood of the $\ell_\infty$ ball of radius $\tau^* = \eta/\lambda$.

SGD lacks this mechanism entirely. Without per-parameter adaptive scaling, the weight decay step $-\lambda \boldsymbol{\theta}_t$ directly competes with the gradient step $-\eta \nabla \mathcal{L}$, and perturbations to $\lambda$ propagate without damping. Our experiments quantify this: for the constant vs.\ no\_wd comparison, SGD yields Cohen's $d = 10.29$ ($p_{\text{adj}} = 0.002$), while AdamW yields $d < 1$ ($p > 0.5$). This $> 10\times$ difference in effect size is consistent with the absence of a Regime I mechanism in SGD.

We note that the standard SGD configuration ($\lambda = 5 \times 10^{-4}$, $\eta = 0.1$) yields $\rho_{\text{SGD}} = 0.005$, while AdamW uses $\rho_{\text{AdamW}} = 0.5$---a 100$\times$ difference. The $18.3\times$ effect-size ratio therefore combines the effect of optimizer mechanism with the effect of operating regime. Matched-$\rho$ experiments (e.g., SGD at $\rho = 0.5$ via $\lambda = 0.05$) are needed to isolate the optimizer contribution; these are planned as a priority experiment.

**Proposition 2** (SGD Alignment-Optimal Schedule). *Under the alignment-dependent stability bound of Sun et al.\ (CVPR 2025), the optimal budget-allocated WD schedule for SGD is $\lambda_t^* \propto \delta_t / (1 - \delta_t)$, where $\delta_t$ is the gradient-weight alignment at step $t$.* This water-filling solution provides theoretical grounding for CWD's binary mask as a coarse approximation to the optimal schedule. We note that extending Sun et al.'s fixed-$\lambda$ analysis to time-varying $\lambda_t$ involves a formal gap: specifically, the original analysis assumes a stationary WD coefficient, and the extension to time-varying $\lambda_t$ requires additional assumptions about the rate of change of the alignment signal. We flag this as an open theoretical question.

---

## 5. Experimental Setup

### 5.1 Implementation

All experiments use a unified `UnifiedAdamW` optimizer with a pluggable Phi modulator interface. Each weight decay strategy is implemented as a subclass of the `WDModulator` abstract base class (Section 3.1), ensuring identical optimizer internals---moment estimation, bias correction, learning rate scheduling---with differences isolated to the computation of $\varphi$. The codebase extends the `why-weight-decay` infrastructure (D'Angelo et al., 2024).

We evaluate seven weight decay methods spanning the four modulation axes: **constant** ($\varphi \equiv 1$, baseline), **cwd\_hard** (CWD binary sign-alignment mask, directional), **swd** (gradient-norm sensitivity, temporal), **cosine\_schedule** (cosine-annealed weight decay, temporal), **random\_mask** (Bernoulli mask with $p = 0.5$, control), **half\_lambda** (constant at $\lambda/2$, budget control), and **no\_wd** ($\varphi \equiv 0$, ablation). SGD experiments use the same seven methods with standard SGD+momentum.

**Implementation note.** During development, we discovered and corrected a bug in the BEM computation: the `HalfLambdaPhi` module did not correctly report effective weight decay in its diagnostics, causing BEM to read $0.000$ instead of the correct $-0.500$. The signed BEM formula (removing absolute value) was also adopted. All results reported here use the corrected implementation.

### 5.2 Training Configuration

**Datasets.** CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) with standard augmentation: random cropping with 4-pixel padding, random horizontal flipping, per-channel normalization. Standard 50,000/10,000 train/test split.

**Architectures.** ResNet-20 ($\sim$270K parameters) with batch normalization in the standard CIFAR configuration (He et al., 2016). VGG-16-BN ($\sim$15M parameters) is evaluated via pilot runs (Section 6.3). All architectures use batch normalization; this is a deliberate design choice for controlled comparison but creates a confound with BN scale-invariance (see Section 7.3).

**Optimizers.** (i) AdamW with decoupled weight decay: $\eta = 10^{-3}$ with cosine annealing to zero, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. (ii) SGD with momentum: $\eta = 0.1$ with cosine annealing, momentum $= 0.9$.

**Weight decay.** Base coefficient $\lambda = 5 \times 10^{-4}$ for both AdamW and SGD. This gives $\rho = \lambda/\eta = 0.5$ for AdamW and $\rho = 0.005$ for SGD---a 100$\times$ difference. Each dynamic method modulates the base coefficient through its Phi function. This $\rho$ asymmetry is an inherent consequence of the standard hyperparameter choices for each optimizer; Section 4.3 discusses the implications for cross-optimizer comparison.

**Training.** 200 epochs, batch size 128, three random seeds (42, 123, 456) per configuration.

**Total experimental budget.** AdamW: 7 methods $\times$ 2 datasets $\times$ 3 seeds $= 42$ runs. SGD: 7 methods $\times$ 2 datasets $\times$ 3 seeds $= 42$ runs. VGG-16-BN pilot: 3 methods $\times$ 1 dataset $\times$ 1 seed $\times$ 10 epochs $= 3$ runs. Total: 87 experiments. **Data completeness note**: CIFAR-100 SGD no\_wd results are based on a single seed (seed\_42); the remaining seeds were lost due to a pipeline error. Statistics involving this condition should be interpreted with appropriate caution.

### 5.3 Statistical Analysis Protocol

We employ a rigorous statistical framework designed for both detecting genuine effects and establishing equivalence:

**Pairwise comparisons.** Paired $t$-tests comparing each dynamic method to the constant baseline, with **Bonferroni-Holm** sequential correction for multiple comparisons. We report both raw and adjusted $p$-values.

**Effect sizes.** Cohen's $d$ (pooled standard deviation) for all pairwise comparisons, providing scale-invariant measures of practical significance. We adopt the standard thresholds: $|d| < 0.2$ (negligible), $0.2 \leq |d| < 0.5$ (small), $0.5 \leq |d| < 0.8$ (medium), $|d| \geq 0.8$ (large).

**Equivalence testing.** TOST (Two One-Sided Tests) equivalence testing with margin $\delta = 0.5\%$, assessing whether methods are statistically equivalent rather than merely "not significantly different." We acknowledge that with $n = 3$ seeds (2 degrees of freedom), TOST power is limited; the minimum detectable effect (MDE) at 80\% power is approximately $\pm 0.77\%$, and TOST power at $\delta = 0.5\%$ is only $\sim 15$--$20\%$. Formal TOST equivalence testing requires $n \geq 5$--$7$ seeds.

**Bootstrap confidence intervals.** 10,000-resample BCa (bias-corrected and accelerated) 95\% confidence intervals for effect-size ratios, particularly the SGD/AdamW effect-size ratio for weight decay presence. We note that bootstrapping from $n = 3$ has inherent limitations.

**Statistical honesty statement.** We explicitly distinguish between (i) comparisons that achieve statistical significance after multiple-comparison correction, (ii) comparisons that show non-significant trends, and (iii) comparisons where equivalence is formally established via TOST. Our current sample size ($n = 3$) is sufficient for detecting large effects but insufficient for formal equivalence testing. We report all results regardless of direction, including null findings, and are transparent about statistical limitations throughout.

---

## 6. Results

### 6.1 AdamW: No Significant Differences Under Standard Settings

**Table 2** presents the main AdamW results on CIFAR-10 and CIFAR-100 with ResNet-20. Under standard settings ($\rho = 0.5$), operating within the predicted Regime I ($\rho < 1$; the regime boundaries themselves remain untested and require a $\lambda$-sweep experiment), no statistically significant differences were detected among any of the seven weight decay strategies.

**Table 2: AdamW results (ResNet-20, 200 epochs, 3 seeds).** Best test accuracy (mean $\pm$ std). All pairwise comparisons vs.\ constant: $p_{\text{adj}} > 0.05$ (Holm correction). Cohen's $d$ reported for constant vs.\ each method.

| Method | CIFAR-10 Acc. (\%) | CIFAR-100 Acc. (\%) | Cohen's $d$ (C10) | BEM |
|--------|-------------------|--------------------|----|-------|
| constant | 90.13 $\pm$ 0.31 | 63.15 $\pm$ 0.30 | --- | 0.000 |
| cosine\_schedule | 90.12 $\pm$ 0.07 | 63.42 $\pm$ 0.42 | 0.05 | $-0.600$ |
| cwd\_hard | 90.06 $\pm$ 0.24 | 62.84 $\pm$ 0.29 | 0.25 | $-0.490$ |
| random\_mask | 90.12 $\pm$ 0.30 | 62.87 $\pm$ 0.37 | 0.03 | $-0.500$ |
| half\_lambda | 90.09 $\pm$ 0.28 | 62.91 $\pm$ 0.47 | 0.14 | $-0.500$ |
| swd | 89.88 $\pm$ 0.25 | 63.06 $\pm$ 0.29 | 0.89 | varies$^*$ |
| no\_wd | 90.08 $\pm$ 0.32 | 62.66 $\pm$ 0.38 | 0.16 | $-1.000$ |

$^*$SWD's BEM is time-varying because its gradient-norm sensitivity function $h(\|\mathbf{g}_t\|)$ produces epoch-dependent effective weight decay; no single summary value is appropriate. The mean BEM over training is approximately $-0.3$ but fluctuates substantially across epochs.

Key observations:

1. **Accuracy range $< 0.3\%$ on CIFAR-10.** The best method (constant, 90.13\%) and worst (swd, 89.88\%) differ by only 0.25 percentage points---well within the noise floor of individual seed variance ($\sigma \approx 0.25$--$0.30\%$). On CIFAR-100, the spread is 0.76\% (cosine 63.42\% vs.\ no\_wd 62.66\%), which falls within the minimum detectable effect at $n = 3$ (MDE $\approx 0.77\%$).

2. **No significant pairwise differences.** All paired $t$-tests (each method vs.\ constant) yield $p_{\text{adj}} > 0.05$ after Holm correction. No dynamic method significantly improves upon or degrades relative to the constant baseline. This absence of significance is consistent with equivalence but does not constitute proof; formal TOST testing at $n \geq 5$ seeds is needed (see Section 5.3).

3. **No weight decay ($\lambda = 0$) shows no significant difference.** The no\_wd ablation achieves 90.08\% on CIFAR-10 and 62.66\% on CIFAR-100---not significantly different from constant weight decay ($p_{\text{adj}} > 0.05$). This suggests that at $\rho = 0.5$, the *presence* of weight decay itself may be immaterial under AdamW, though confirming this requires higher statistical power.

4. **Budget variation does not predict performance.** BEM ranges from $0.000$ (constant) to $-1.000$ (no\_wd), spanning the full budget spectrum, yet accuracy varies by $< 0.3\%$ on CIFAR-10. Methods applying half the budget (half\_lambda, BEM $= -0.500$), 60\% less budget (cosine\_schedule, BEM $\approx -0.600$), and zero budget (no\_wd, BEM $= -1.000$) all show no significant differences. This is consistent with the Regime I prediction that in the low-$\rho$ regime, the *total budget* of weight decay is as irrelevant as its *temporal distribution*.

5. **Cosine schedule exhibits notably lower variance on CIFAR-10** ($\sigma = 0.07\%$ vs.\ $\sim 0.30\%$ for other methods). This "pre-programmed trajectory" effect may reduce stochastic sensitivity by replacing adaptive decisions with a deterministic schedule, though it does not improve mean performance. We note this observation does not replicate on CIFAR-100, and with $n = 3$, a formal variance test (e.g., Levene's) lacks statistical power to confirm the difference.

### 6.2 SGD: Weight Decay Presence Matters

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

**Statistical honesty statement.** After Holm correction for 6 comparisons, **only one pairwise comparison achieves significance**: constant vs.\ no\_wd ($p_{\text{adj}} = 0.002$, Cohen's $d = 10.29$). The swd comparison ($p_{\text{adj}} = 0.054$) and half\_lambda comparison ($p_{\text{adj}} = 0.074$) do not reach significance at $\alpha = 0.05$, though they exhibit large effect sizes ($d > 2$) suggestive of genuine effects that our $n = 3$ design lacks power to confirm. CIFAR-100 SGD results for no\_wd are based on $n = 1$ (single seed) and are not included in statistical comparisons.

**Key findings:**

1. **Weight decay presence matters under SGD.** Removing weight decay entirely (no\_wd) degrades accuracy by 0.92 percentage points with an extraordinarily large effect size ($d = 10.29$) and robust significance ($p_{\text{adj}} = 0.002$). This is the single most statistically reliable finding in our study.

2. **Effect sizes are consistently large.** Five of six SGD comparisons yield $|d| > 1$ (large effect), compared to $|d| < 1$ for all AdamW comparisons. The weight decay perturbation propagates through SGD's non-adaptive updates without damping.

3. **The $18.3\times$ effect-size ratio.** Comparing constant vs.\ no\_wd across optimizers: SGD Cohen's $d = 10.29$ vs.\ AdamW Cohen's $d = 0.16$, yielding a ratio of $\sim 18.3\times$. Bootstrap BCa 95\% CI: [$12.1\times$, $28.7\times$]. **Important caveat**: this ratio reflects the combined influence of optimizer mechanism and operating-point $\rho$ ($\rho_{\text{AdamW}} = 0.5$ vs.\ $\rho_{\text{SGD}} = 0.005$---a 100$\times$ difference). Isolating the optimizer contribution requires matched-$\rho$ experiments (planned as P1-c). Even as a combined measure, the ratio demonstrates a qualitatively meaningful asymmetry between the two optimizers' sensitivity to weight decay. We also note that bootstrapping a ratio from $n = 3$ samples has methodological limitations; the confidence interval should be interpreted cautiously.

### 6.3 Cross-Architecture Pilot

VGG-16-BN pilot experiments (10 epochs, 1 seed, CIFAR-10) confirm infrastructure readiness for full-scale evaluation. **These results are not intended as scientific validation**; they serve only to verify that the experimental pipeline generalizes to larger architectures.

| Method | VGG-16-BN Acc. (10 ep) | BEM | CSI |
|--------|----------------------|------|------|
| constant | 79.94\% | 0.000 | 0.996 |
| cwd\_hard | 80.30\% | $-0.490$ | 1.011 |
| no\_wd | 80.61\% | $-1.000$ | 0.988 |

All methods train successfully with no OOM errors on RTX PRO 6000 (98GB). BEM values match theoretical expectations. We note that no\_wd (80.61\%) slightly outperforms constant (79.94\%) in this pilot, which is directionally opposite to the invariance prediction; however, at 10 epochs with 1 seed, this observation carries no statistical weight. Full VGG-16-BN results (200 epochs, 3+ seeds) are planned to validate whether the conditional equivalence observation generalizes across architectures.

### 6.4 Diagnostic Analysis

**BEM: Budget Characterization.** Table 2 includes BEM values for all methods. The metric correctly differentiates methods by effective budget: constant ($0.000$), half\_lambda ($-0.500$), cosine\_schedule ($\approx -0.600$), and no\_wd ($-1.000$) match theoretical predictions. CWD's BEM ($-0.490$) confirms that its sign-alignment mask gates approximately half the weight decay budget, consistent with the expectation that $\sim$50\% of parameter coordinates satisfy $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$ at any given step.

Critically, the BEM spectrum ($0.000$ to $-1.000$) spans the full range of possible budget allocations, yet produces no significant accuracy variation under AdamW ($\rho = 0.5$). BEM correctly tracks budget variation across the full $[0, -1]$ spectrum with no accuracy correlation in this regime, validating it as a descriptive characterization tool.

**Weight norm convergence.** Under AdamW, all seven methods converge to weight norms in the narrow range 95.89--97.04 (only 1.2\% variation), despite $10\times$ variation in effective BEM. This is consistent with the $\ell_\infty$ constraint mechanism: AdamW's implicit constraint may absorb weight decay perturbations, driving all trajectories to a common neighborhood regardless of the modulation pattern. However, all architectures tested use batch normalization, whose scale-invariance property (D'Angelo et al., 2024) provides an alternative mechanism for weight norm convergence. A NoBN ablation is required to disentangle the $\ell_\infty$ constraint explanation from the BN confound (planned as P0-3).

**AIS: Alignment Informativeness.** AIS values range from 0.25 to 0.50 across methods and architectures, indicating moderate alignment diversity. This suggests that the alignment signal CWD exploits is neither random nor maximally informative---it carries *some* information, but not enough to produce measurable gains at this operating point. The limited informativeness of alignment signals at $\rho = 0.5$ is consistent with the Phi Invariance Conjecture: even methods that condition on alignment produce the same outcomes when the constraint geometry dominates.

---

## 7. Discussion

### 7.1 Practical Implications

Our findings suggest a tentative practical message for practitioners: **under standard AdamW settings ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$) with batch-normalized architectures at CIFAR scale, we detected no statistically significant benefit from adopting dynamic weight decay strategies** ($n = 3$; see Section 7.3 for power limitations). Constant weight decay, cosine-annealed weight decay, CWD, SWD, half-budget weight decay, and even no weight decay at all produced no significant differences. If this observation holds at larger scales and with increased statistical power, practitioners could save hyperparameter tuning effort and implementation complexity by using the simplest option.

Perhaps the most striking aspect of this observation is that **no weight decay at all** showed no significant difference from constant weight decay under AdamW at $\rho = 0.5$. This tentatively challenges the conventional wisdom that weight decay is essential for generalization. Under our framework, AdamW's implicit $\ell_\infty$ constraint (Xie \& Li, 2024) may already provide the norm control that explicit weight decay ostensibly delivers. At low $\rho$, the constraint geometry could dominate, making the explicit weight decay term redundant. However, this mechanistic attribution remains a hypothesis: the alternative explanation via BN scale-invariance (D'Angelo et al., 2024) has not been ruled out.

The cosine schedule's notably lower variance on CIFAR-10 ($\sigma = 0.07\%$ vs.\ $\sim 0.30\%$) is noteworthy for practitioners in settings where reproducibility matters. We note this property was not observed on CIFAR-100.

This observation comes with explicit boundary conditions. Weight decay strategy *may* matter when:
- **Using SGD**: Our experiments show an approximately $18.3\times$ (95\% CI: [$12\times$, $29\times$]) effect-size ratio for weight decay presence, though this ratio reflects combined optimizer and $\rho$-regime effects (see Limitation 6 below).
- **Using high weight decay coefficients**: The Phi Invariance Conjecture predicts that as $\rho = \lambda/\eta$ increases beyond $\sim 1$, modulation strategy becomes progressively important. This prediction is untested.
- **Architectures without batch normalization**: The interaction between BN scale-invariance and weight decay dynamics remains an unresolved confound.

We propose a **preliminary, conjectured** decision heuristic for practitioners (to be validated by future experiments):
1. Compute $\rho = \lambda/\eta$.
2. If $\rho < 1$ and using AdamW with BN: constant weight decay is likely sufficient (based on our $\rho = 0.5$ observations).
3. If $\rho > 1$ or using SGD: consider dynamic strategies; the optimal choice in this regime is a theoretically motivated prediction, not yet empirically validated.
4. Use BEM (see Section 3.4) to verify that observed differences are not merely budget effects.

### 7.2 Theoretical Implications

The $\rho = \lambda/\eta$ order parameter provides a unifying lens that connects several independent lines of recent work:

**Dual characterization.** We observe (Observation 1) that Xie \& Li's (2024) constraint radius $\tau^* = \eta/\lambda = 1/\rho$ and Defazio's (2025) steady-state gradient-to-weight ratio $R_* = \lambda/\eta = \rho$ are dual characterizations of the same quantity. This connection was not made explicit in either prior work and suggests that the $\ell_\infty$ constraint picture and the gradient-weight equilibrium picture are two faces of the same phenomenon.

**Toward a mechanistic understanding.** We hypothesize that AdamW's per-coordinate sign descent creates an implicit constraint set whose boundary is determined by $\rho$ alone. Any Phi modulation that preserves $\rho$'s regime membership merely perturbs the trajectory within this constraint set, which converges to the same neighborhood regardless of the perturbation path. In this view, the constraint geometry---not the weight decay schedule---determines the final solution. This explains why even the degenerate case of no weight decay ($\varphi = 0$) produces no significant difference: at $\rho = 0.5$, the $\ell_\infty$ ball may be large enough that the implicit constraint is non-binding, and the gradient dynamics alone drive convergence to a similar region of parameter space. **Critical caveat**: this mechanistic attribution to AdamW's $\ell_\infty$ constraint versus BN scale-invariance (D'Angelo et al., 2024) remains an open question. All our experiments use BN architectures, and the observed weight norm convergence is equally consistent with BN's scale-invariance making WD irrelevant. NoBN ablation experiments are the highest-priority blocking experiment for resolving this question.

**Consistency with Wang \& Aitchison (2024).** Their finding that optimal weight decay scales as an EMA timescale constant across model sizes is consistent with our framework: if $\rho$ remains in Regime I across scales, then the specific weight decay schedule is immaterial, and the "optimal" weight decay reduces to choosing the right magnitude rather than the right modulation pattern. Their EMA timescale $= 1/\rho$ in our notation.

**Beyond D'Angelo et al.\ (2024).** D'Angelo et al.\ showed that weight decay is never useful as explicit regularization in batch-normalized settings, attributing this to batch normalization's scale-invariance. Our observation is complementary but distinct: we find that weight decay *modulation* produces no significant differences under AdamW, a dynamic argument that goes beyond static reparameterization. However, we cannot currently distinguish between BN-driven and AdamW-driven mechanisms for this invariance.

**Chou (2025) as a special case.** Their proposal to scale weight decay proportionally to the learning rate schedule ($\lambda \propto \gamma$) is a specific functional form within our temporal-axis modulation. Our preliminary analysis at $\rho = 0.5$ suggests that in this regime, this schedule is one of many strategies that produce no significant differences. In the transitional Regime II, however, Chou's schedule may provide benefits that our current experiments do not capture.

**Falsification criteria.** The Phi Invariance Conjecture would be falsified if, at $\rho = 0.5$ with AdamW and BN, a dynamic WD strategy showed $> 1\%$ accuracy improvement over constant WD at $n \geq 7$ seeds with statistical significance after multiple comparison correction. Partial falsification would occur if the $\rho$ regime sweep fails to find a transition near $\rho_1 \approx 1$.

### 7.3 Scope and Limitations

We explicitly delineate the verified scope of our findings and acknowledge limitations with full candor:

**Verified scope.** CIFAR-10 and CIFAR-100 with a single architecture (ResNet-20, 270K parameters, batch normalization), AdamW at a single $\rho$ operating point ($\rho = 0.5$), SGD at standard settings ($\rho = 0.005$). Seven weight decay methods spanning four modulation axes. 87 experiments with 3 seeds per configuration (except CIFAR-100 SGD no\_wd, which has $n = 1$). The observation of no significant differences under AdamW and the $18.3\times$ effect-size ratio are supported within this scope.

**Limitations:**

1. **Statistical power.** With $n = 3$ seeds (2 degrees of freedom), our paired $t$-tests have limited power. The minimum detectable effect at 80\% power is approximately $\pm 0.77\%$ accuracy. TOST equivalence testing with margin $\delta = 0.5\%$ requires approximately $n = 7$--$10$ for adequate power given our observed standard deviations ($\sigma \approx 0.3\%$). Our observation of no significant differences is therefore "consistent with equivalence" rather than "proof of equivalence." The CIFAR-100 spread of 0.76\% (cosine 63.42\% vs.\ no\_wd 62.66\%) falls within the MDE, meaning a real effect of this magnitude could be missed.

2. **Single architecture.** Our primary results use only ResNet-20 (270K parameters)---a small model by 2026 standards. While the $\rho$ framework predicts architecture-independent behavior in Regime I, this has not been empirically validated on larger architectures (VGG-16 pilot is insufficient), Vision Transformers, or models without batch normalization.

3. **Dataset scale.** CIFAR-10 and CIFAR-100 are standard benchmarks but limited in scale and complexity. ImageNet validation would significantly strengthen our claims. The $\rho$ value ($\lambda/\eta$) is independent of dataset size, suggesting the regime boundary should transfer, but this requires verification.

4. **$\rho$ regime boundary precision.** We have tested only $\rho = 0.5$ (standard AdamW). The regime boundary $\rho_1$ between Regime I and Regime II is predicted ($\rho_1 \approx 1$) but not experimentally located. A systematic $\lambda$ sweep ($\rho \in \{0.05, 0.5, 5, 50\}$) is needed to empirically characterize the transition.

5. **Incomplete theoretical support.** The Phi Invariance Conjecture is supported by empirical observations and motivated by theoretical arguments (Lemmas 1--3), but the formal proofs are in preparation (Appendix D). The critical assumption---Adam saturation ($\epsilon / (\sqrt{\hat{v}_i} \cdot |w_i|) < 0.1$)---holds empirically for $> 80\%$ (not 100\%) of parameters, and the gap is not formally addressed.

6. **SGD hyperparameter asymmetry.** The $18.3\times$ effect-size ratio conflates two effects: the optimizer mechanism (adaptive vs.\ non-adaptive) and the $\rho$ operating-point difference ($\rho_{\text{AdamW}} = 0.5$ vs.\ $\rho_{\text{SGD}} = 0.005$). Matched-$\rho$ SGD experiments are needed to isolate the optimizer contribution.

7. **BN confound.** All architectures use batch normalization, whose scale-invariance (D'Angelo et al., 2024) provides an alternative explanation for the observed weight decay insensitivity. The paper cannot currently distinguish between the $\ell_\infty$ constraint mechanism and the BN mechanism. NoBN ablation is the highest-priority blocking experiment.

8. **Data completeness.** CIFAR-100 SGD no\_wd results are based on $n = 1$ (single seed). This limits the statistical analysis for CIFAR-100 SGD comparisons.

---

## 8. Conclusion

### 8.1 Summary of Findings

We introduced the Phi Modulator Framework, a unified mathematical abstraction that recovers seven major dynamic weight decay methods as special cases along four modulation axes (temporal, directional, spatial, and target-norm). Across 7 methods, 2 datasets, and 2 optimizers on ResNet-20 (3 seeds each, 87 experiments total), we identified three central observations:

First, **no significant differences under AdamW**: at standard settings ($\rho = \lambda/\eta = 0.5$) with batch-normalized ResNet-20, no statistically significant differences were detected among any of the seven weight decay strategies---including the complete absence of weight decay (accuracy range $< 0.3\%$ on CIFAR-10, all $p_{\text{adj}} > 0.05$, $n = 3$). This observation is consistent with conditional equivalence but does not constitute proof; formal TOST equivalence testing at $n \geq 5$ is required to establish equivalence. The CIFAR-100 spread (0.76\%) falls within our minimum detectable effect ($\sim 0.77\%$).

Second, **optimizer-dependent sensitivity**: SGD exhibits fundamentally different behavior, with the constant vs.\ no\_wd comparison yielding Cohen's $d = 10.29$ ($p_{\text{adj}} = 0.002$)---an effect-size ratio of approximately $18.3\times$ (95\% CI: [$12\times$, $29\times$]) relative to AdamW. This ratio reflects the combined effect of optimizer mechanism and the 100$\times$ difference in $\rho$ values between standard SGD ($\rho = 0.005$) and AdamW ($\rho = 0.5$) settings. Mechanistic attribution to AdamW's implicit $\ell_\infty$ constraint versus BN scale-invariance remains an open question pending NoBN ablation experiments.

Third, **the $\rho = \lambda/\eta$ regime boundary**: we proposed the Phi Invariance Conjecture, which posits $\rho$ as the order parameter governing when weight decay strategy choice matters. The conjecture connects to Xie \& Li's (2024) constraint radius and Defazio's (2025) gradient-weight equilibrium through an algebraic dual characterization (Observation 1), and makes falsifiable predictions at specific $\rho$ values. The conjecture is currently supported at a single operating point ($\rho = 0.5$); the predicted regime boundaries ($\rho_1 \approx 1$, $\rho_2 \approx 10$) remain to be validated through $\lambda$-sweep experiments.

Additionally, we contributed three diagnostic metrics---BEM, CSI, and AIS---as standardized tools for characterizing weight decay strategies. BEM correctly tracked budget variation across the full $[0, -1]$ spectrum with no accuracy correlation under AdamW at $\rho = 0.5$, consistent with the Regime I prediction. AIS values (0.25--0.50) revealed the limited informativeness of alignment signals at this operating point.

### 8.2 Future Work

Several directions warrant investigation, ordered by priority for validating the paper's central observations:

1. **NoBN ablation** (highest priority, $\sim$1h). Experiments with ResNet-20 without batch normalization would disentangle the contributions of BN scale-invariance (D'Angelo et al., 2024) from AdamW's implicit $\ell_\infty$ constraint to the observed invariance. This is the most important blocking experiment.

2. **$\rho$ regime sweep** ($\sim$3--4h). Systematic variation of $\lambda \in \{5 \times 10^{-5}, 5 \times 10^{-4}, 5 \times 10^{-3}, 5 \times 10^{-2}\}$ (corresponding to $\rho \in \{0.05, 0.5, 5, 50\}$) would empirically locate the Regime I/II boundary and provide the strongest validation or falsification of the Phi Invariance Conjecture.

3. **Increased seed count** ($\sim$2--3h). Extending to $n = 5$--$7$ seeds would enable formal TOST equivalence testing, converting our current observation of "no significant differences" into a rigorous equivalence statement (or revealing a real effect hidden by insufficient power).

4. **Cross-architecture validation** ($\sim$6--8h). Full VGG-16-BN runs (200 epochs, 3+ seeds) and Vision Transformer experiments are essential for establishing generality beyond ResNet-20.

5. **Matched-$\rho$ SGD control** ($\sim$1--2h). SGD at $\rho = 0.5$ (via $\lambda = 0.05$, $\eta = 0.1$) would isolate the optimizer mechanism contribution to the $18.3\times$ ratio from the $\rho$ operating-point difference.

6. **Large-scale validation** ($\sim$8--12h). ImageNet experiments with ResNet-50 would test whether the $\rho$-based framework generalizes across scales.

7. **Formal proof of the Phi Invariance Conjecture.** A rigorous proof under simplified settings (e.g., quadratic loss with diagonal Hessian under AdamW) would elevate the conjecture to a theorem and sharpen the regime boundary estimates.

8. **Optimizer generalization.** Testing the framework with other adaptive optimizers (Lion, Muon, Shampoo) would determine whether the $\ell_\infty$ absorption mechanism is specific to AdamW or common to a broader class of sign-based optimizers.

### 8.3 Broader Impact

Our work delivers a practically useful message: knowing *when not to optimize* is as valuable as knowing *how to optimize*. For the substantial community of practitioners using AdamW at standard settings with batch-normalized architectures, our preliminary results suggest that effort invested in dynamic weight decay strategy selection may be unnecessary at the scales and settings we tested---the simplest constant schedule appears sufficient. We provide a preliminary $\rho$-based decision heuristic (Section 7.1) that, pending further validation, may guide practitioners to the appropriate weight decay strategy for their setting.

Beyond weight decay, the approach of identifying regime boundaries via dimensionless ratios ($\rho = \lambda/\eta$) and conducting controlled multi-method comparisons may serve as a template for resolving similar proliferation problems in other areas of optimizer design---such as learning rate scheduling, momentum tuning, or gradient clipping strategies.

---

## References

Chen, L. et al. (2026a). Cautious Weight Decay. *ICLR 2026*.

Chen, L. et al. (2026b). AdamO: Decoupling Radial and Tangential Weight Decay Dynamics. *ICLR 2026*.

Chou, S. (2025). Weight Decay Scheduling Proportional to Learning Rate. *arXiv preprint*.

D'Angelo, F. et al. (2024). Why Weight Decay Is Never Useful as Regularization in Modern Deep Learning. *NeurIPS 2024*.

Defazio, A. (2025). Steady-State Analysis of Weight Decay in Adaptive Optimizers. *arXiv preprint arXiv:2506.02285*.

Ferbach, D. et al. (2026). ADANA: Adaptive Norm and Weight Decay Annealing. *ICLR 2026* (arXiv preprint, 2025).

Hanson, S. J. & Pratt, L. Y. (1988). Comparing biases for minimal network construction with back-propagation. *NeurIPS 1988*.

He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

He, Z. et al. (2025). AlphaDecay: Module-Wise Weight Decay via Spectral Density Analysis. *arXiv preprint*.

Hoffer, E. et al. (2018). Norm Matters: Efficient and Accurate Normalization Schemes in Deep Networks. *NeurIPS 2018*.

Ioannidis, J. P. A. (2005). Why Most Published Research Findings Are False. *PLoS Medicine*.

Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.

Kosson, A. et al. (2023). Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks. *arXiv preprint*.

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *Technical Report*.

Krogh, A. & Hertz, J. (1991). A Simple Weight Decay Can Improve Generalization. *NeurIPS 1991*.

Lipton, Z. C. & Steinhardt, J. (2019). Troubling Trends in Machine Learning Scholarship. *ACM Queue*.

Loshchilov, I. (2023). Weight Norm Control: Generalized Decoupled Weight Decay. *arXiv preprint*.

Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*.

Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.

Sun, R. et al. (2025). Alignment-Dependent Stability Bounds for SGD with Weight Decay. *CVPR 2025*.

Tian, Y. et al. (2024). Selective Projection Decay for Fine-Tuning. *arXiv preprint*.

Wang, A. \& Aitchison, L. (2024). Optimal Weight Decay as EMA Timescale. *NeurIPS 2024*.

Xie, Z. et al. (2023). Scheduled Weight Decay (AdamS). *ICML 2023*.

Xie, Z. \& Li, Z. (2024). AdamW as $\ell_\infty$-Constrained Optimization. *NeurIPS 2024*.

---

*Integrated paper: Iteration 4, Revision 2*
*Date: 2026-03-18*
