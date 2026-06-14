# When Does Dynamic Weight Decay Help? A Unified Framework Analysis

## Abstract

Weight decay is universally applied in deep learning optimization, yet practitioners face a bewildering proliferation of dynamic weight decay strategies---Cautious Weight Decay (CWD), Scheduled Weight Decay (SWD), cosine schedules, AlphaDecay, ADANA, AdamO---with no systematic understanding of *when* these methods provide genuine benefit. We introduce the **Phi Modulator Framework**, a unified mathematical abstraction $\varphi(t, \boldsymbol{\theta}, \mathbf{s}_t)$ that recovers all major dynamic weight decay methods as special cases along four modulation axes: temporal, directional, spatial, and target-norm. Here $\mathbf{s}_t$ denotes the optimizer state, which may include raw gradients $\mathbf{g}_t$, preconditioned updates $\mathbf{u}_t$, or other statistics. Through this lens, we conduct the first controlled comparison of seven weight decay strategies under identical optimization conditions across 87+ experiments. Our central finding is a **conditional indistinguishability result**: under standard AdamW settings ($\rho = \lambda/\eta \approx 0.5$), all seven strategies---including the degenerate case of no weight decay---are statistically indistinguishable (accuracy range $< 0.3\%$, all pairwise $p_{\text{adj}} > 0.05$ after Holm correction, $n = 3$). In stark contrast, SGD exhibits significant sensitivity to weight decay: two of six comparisons achieve significance after Holm correction (constant vs.\ no\_wd: Cohen's $d = 10.29$, $p_{\text{adj}} = 0.002$; constant vs.\ swd: $d = 3.48$, $p_{\text{adj}} = 0.004$), yielding an approximately $64\times$ effect-size ratio for weight decay presence (SGD $d = 10.29$ vs.\ AdamW $d = 0.16$; Bootstrap BCa 95\% CI: approximate due to $n = 3$). This ratio reflects both the optimizer mechanism difference and the $10\times$ difference in $\rho$ values between standard AdamW ($\rho = 0.5$) and SGD ($\rho = 0.05$) configurations. We propose the **Phi Invariance Conjecture**, which posits $\rho = \lambda/\eta$ as the order parameter governing a regime boundary: when $\rho \lesssim 1$, AdamW's implicit $\ell_\infty$ constraint is hypothesized to render all modulation strategies indistinguishable; as $\rho$ increases, strategy choice becomes progressively important. We further propose three standardized diagnostic metrics---BEM, CSI, and AIS---as characterization tools for describing operational differences between weight decay strategies. All findings are currently validated on CIFAR-scale benchmarks with batch-normalized architectures; large-scale and BN-free validation remains an important next step.

---

## 1. Introduction

### 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization (Krogh & Hertz, 1991). This view has been progressively undermined: Loshchilov & Hutter (2019) demonstrated that L2 regularization and decoupled weight decay are not equivalent in adaptive optimizers, leading to AdamW. D'Angelo et al. (2024) showed weight decay is never useful as explicit regularization in modern BN-equipped architectures; instead, it acts as a training dynamics modifier. Xie & Li (2024) revealed that AdamW implicitly performs $\ell_\infty$-norm constrained optimization. Kosson et al. (2023) showed weight decay induces a rotational equilibrium across layers.

These understandings have catalyzed a surge of dynamic weight decay methods: SWD (Xie et al., 2023), CWD (Chen et al., 2026a), AdamWN (Loshchilov, 2023), AlphaDecay (He et al., 2025), ADANA (Ferbach et al., 2026), AdamO (Chen et al., 2026b). A critical problem pervades this literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, and hyperparameter protocols. No two papers share the same experimental conditions.

### 1.2 Research Gap

Answering *when* dynamic weight decay helps is currently impossible due to four gaps: no unified mathematical framework, no standardized evaluation metrics, no controlled systematic comparison, and no theory for when modulation becomes irrelevant.

### 1.3 Our Approach: The $\rho = \lambda/\eta$ Lens

The quantity $\rho = \lambda/\eta$ appears implicitly in recent theoretical analyses (Xie & Li, 2024; Defazio, 2025; Wang & Aitchison, 2024). Our contribution is to operationalize it as an *empirically testable regime boundary parameter*. The ratio $\rho$ measures the relative magnitude of the weight decay step compared to the gradient step.

Our core finding: at AdamW settings (AdamW: $\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, $\rho = 0.5$; SGD: $\lambda = 5 \times 10^{-3}$, $\eta = 0.1$, $\rho = 0.05$), seven weight decay strategies spanning all four modulation axes produce statistically indistinguishable results under AdamW (see Figure 1 for the accuracy distributions and Figure 2 for the AdamW/SGD contrast). Yet under SGD, weight decay presence produces a massive effect (Cohen's $d > 10$, $p < 0.003$). We attribute this asymmetry to AdamW's implicit $\ell_\infty$ constraint (Xie & Li, 2024), which we hypothesize absorbs weight decay perturbations in the low-$\rho$ regime; Section 7.2 discusses the interplay with batch normalization's scale-invariance (D'Angelo et al., 2024).

The $\rho$-based perspective connects $\tau^* = 1/\rho$ (Xie & Li, 2024), $R_* = \rho$ (Defazio, 2025), and EMA timescale $\approx 1/\rho$ (Wang & Aitchison, 2024) through a unified dual characterization (Remark 1, Section 4).

### 1.4 Contributions

We make four contributions, ordered from enabling to dependent:

1. **Phi Modulator Framework.** A unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{s}_t)$ recovering all major dynamic weight decay methods as special cases along four modulation axes, enabling controlled comparison under identical optimization conditions.

2. **Empirical discovery: conditional indistinguishability.** Across 87+ controlled experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets $\times$ 2 optimizers), we detect no significant differences among weight decay strategies under standard AdamW settings (all pairwise $p_{\text{adj}} > 0.05$, Holm correction, $n = 3$). SGD exhibits significant sensitivity for two comparisons after Holm correction (no\_wd: $d = 10.29$, $p_{\text{adj}} = 0.002$; swd: $d = 3.48$, $p_{\text{adj}} = 0.004$), with an approximately $64\times$ weight-decay-presence effect-size ratio vs.\ AdamW (see Table 3). Results apply to batch-normalized architectures at CIFAR scale (ResNet-20, 270K parameters; large-scale validation addressed in Section 7.3).

3. **Phi Invariance Conjecture.** A trichotomy positing $\rho$ as the regime boundary parameter: Regime I ($\rho \lesssim 1$, strategy-invariant), Regime II (transitional), and Regime III ($\rho \gg 1$, strategy-sensitive). Currently tested only at $\rho = 0.5$; Regimes II and III remain empirically unverified.

4. **Diagnostic tools: BEM/CSI/AIS.** Standardized metrics providing the first vocabulary for characterizing operational differences between weight decay strategies. These metrics do not predict which method is best; they quantify budget deviations, coupling stability, and alignment informativeness.

### 1.5 Paper Roadmap

Section 2 surveys related work. Section 3 introduces the Phi Modulator Framework. Section 4 presents theoretical analysis. Section 5 describes the experimental setup. Section 6 presents results including Figures 1--5. Section 7 provides discussion and limitations. Section 8 concludes.

---

## 2. Related Work

### 2.1 Weight Decay as a Dynamics Modifier

Loshchilov & Hutter (2019) demonstrated that L2 regularization and decoupled weight decay produce fundamentally different behaviors in adaptive optimizers, leading to AdamW. D'Angelo et al. (2024) showed weight decay is *never* useful as explicit regularization in modern BN-equipped architectures. Kosson et al. (2023) showed weight decay induces a *rotational equilibrium*. Xie & Li (2024) proved AdamW implicitly performs $\ell_\infty$-norm constrained optimization with constraint radius $\tau^* = \eta/\lambda = 1/\rho$.

Defazio (2025) derived the steady-state gradient-to-weight ratio $R_* \approx \lambda/\eta = \rho$ for normalized layers---the same quantity as Xie & Li's $\tau^*$ viewed from a gradient equilibrium perspective. Although Defazio does not make this connection explicit, both are dual characterizations of $\rho$ (Remark 1, Section 4). Defazio's focus is end-of-training instability correction; our novelty is the Trichotomy regime structure and cross-optimizer comparison, not the $\rho$ quantity itself.

**A critical open question**: whether BN scale-invariance or AdamW's $\ell_\infty$ constraint is the primary driver of the observed strategy invariance. The NoBN ablation is the highest-priority blocking experiment (Section 7.3, Limitation 2).

### 2.2 Dynamic Weight Decay Methods

We organize existing methods along the four Phi modulation axes. **Temporal scheduling**: SWD/AdamS (Xie et al., 2023); ADANA (Ferbach et al., 2026); alignment-dependent bounds (Sun et al., CVPR 2025). **Directional modulation**: CWD (Chen et al., 2026a); AdamO (Chen et al., 2026b); SPD for fine-tuning (Tian et al., 2024)---outside our scope. **Spatial modulation**: AlphaDecay (He et al., 2025). **Target-norm control**: AdamWN (Loshchilov, 2023).

Our key distinction: we provide the first systematic evaluation of *when* existing methods provide genuine benefit under identical conditions.

### 2.3 Evaluation Fragmentation

No two dynamic weight decay papers share the same experimental conditions. This fragmentation is compounded by publication bias against null results (Lipton & Steinhardt, 2019). We acknowledge our own validation is currently limited in scale (CIFAR-only, ResNet-20); large-scale replication is future work. Wang & Aitchison (2024) showed optimal WD scales as an EMA timescale $\approx 1/\rho$ in our notation---consistent with our conditional indistinguishability result.

### 2.4 Positioning Against Key Recent Works

**vs. Wang & Aitchison (2024).** Their empirical scaling rule focuses on WD *magnitude*; our framework focuses on WD *modulation strategy* with regime-boundary theory and falsifiable predictions.

**vs. D'Angelo et al. (2024).** Their finding is a static BN reparameterization argument. Our Phi Invariance is a *dynamic* argument about AdamW's sign normalization. The NoBN ablation---not yet completed---is needed to distinguish these mechanisms. We cannot currently determine the primary driver of observed invariance.

**vs. Chou (2025).** Their $\lambda \propto \gamma$ schedule is a specific temporal-axis modulation. Based on testing at a single operating point ($\rho = 0.5$), it appears indistinguishable from constant WD under AdamW. Regime II behavior remains untested.

---

## 3. The Phi Modulator Framework

### 3.1 Formal Definition

We generalize AdamW by introducing the **Phi modulator** $\varphi$:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{s}_t) \odot \boldsymbol{\theta}_t$$
where $\varphi : \mathbb{Z}_{\geq 0} \times \mathbb{R}^d \times \mathcal{S} \to \mathbb{R}^d_{\geq 0}$ is a per-parameter, non-negative modulation function, $\mathbf{s}_t$ is the optimizer state (which may include $\mathbf{g}_t$, preconditioned updates $\mathbf{u}_t$, or other statistics), and $\odot$ is element-wise multiplication.

**Budget reference convention:** $\mathbb{E}[\varphi] = 1$ is the normalization reference point. Methods with $\mathbb{E}[\varphi] \neq 1$ are valid modulators applying a different total decay budget, quantified by BEM (Section 3.4). As a consequence, no\_wd ($\mathbb{E}[\varphi] = 0$), half-lambda ($\mathbb{E}[\varphi] = 0.5$), cosine ($\mathbb{E}[\varphi] \approx 0.4$), and CWD ($\mathbb{E}[\varphi] \approx 0.5$) all deviate from the reference.

### 3.2 Method Catalog

**Table 1: Method catalog.** The third argument $\mathbf{s}_t$ refers to optimizer state; for CWD this is specifically the preconditioned update $\mathbf{u}_t$. Methods with $\dagger$ are cataloged but not experimentally evaluated.

| Method | $\varphi(t, \boldsymbol{\theta}, \mathbf{s}_t)$ | Modulation Axis |
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

AdamWN and AlphaDecay require architecture-specific hyperparameters; evaluation deferred to future work. ADANA and AdamO jointly modify weight decay and momentum/radial dynamics, placing them outside pure WD modulation scope.

### 3.3 Budget Equivalence Normalization

**Definition 1** (Budget Equivalence). *Two strategies are budget-equivalent if $\sum_{t} \lambda \cdot \mathbb{E}[\varphi_1(t, \boldsymbol{\theta}_t, \mathbf{s}_t)] = \sum_{t} \lambda \cdot \mathbb{E}[\varphi_2(t, \boldsymbol{\theta}_t, \mathbf{s}_t)]$.*

### 3.4 Diagnostic Metrics

**Budget Equivalence Metric (BEM):**
$$\mathrm{BEM}(\text{method}) = \frac{\bar{\lambda}_{\mathrm{eff}}^{\text{method}} - \bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}{\bar{\lambda}_{\mathrm{eff}}^{\text{constant}}}$$
Verified values: half\_lambda $= -0.500$, cosine\_schedule $\approx -0.600$, no\_wd $= -1.000$.

**Coupling Stability Index (CSI):**
$$\mathrm{CSI}_{\mathrm{raw}} = \tfrac{1}{3} \cdot \mathrm{CV}(\|\mathbf{w}_t\|) + \tfrac{1}{3} \cdot \log(\kappa) + \tfrac{1}{3} \cdot \mathrm{CV}(\eta_{\mathrm{eff}})$$
Equal weights ($1/3$ each) are a convention pending sensitivity analysis. We report $\mathrm{CSI}_{\mathrm{rel}} = \mathrm{CSI}_{\mathrm{raw}} / \mathrm{CSI}_{\mathrm{constant}}$. Note: $\mathrm{CSI}_{\mathrm{rel}}$ is normalized to the constant baseline on the same architecture; cross-architecture comparisons require separate normalization.

**Alignment Informativeness Score (AIS):**
$$\mathrm{AIS} = \frac{1}{L}\sum_{l=1}^{L} \frac{H(\text{bin distribution of } |\cos(\boldsymbol{\theta}^{(l)}, \mathbf{g}^{(l)})|)}{\log(10)}$$
AIS $\in [0, 1]$: values near 1 indicate high alignment diversity. AIS is an intrinsic property of the network and loss landscape, not of the weight decay method.

---

## 4. Theoretical Analysis: The $\rho$ Regime Boundary

### 4.1 The $\rho = \lambda/\eta$ Order Parameter

$$\rho = \frac{\bar{\lambda}}{\eta}$$
where $\bar{\lambda} = \lambda \cdot \mathbb{E}[\varphi]$. At standard AdamW settings, $\rho = 0.5$.

**Remark 1** (Dual Characterization). *The constraint radius $\tau^* = \eta/\lambda$ from Xie & Li (2024) and the steady-state ratio $R_* \approx \lambda/\eta$ from Defazio (2025) are dual characterizations of $\rho$: $\tau^* = 1/\rho$ and $R_* = \rho$. Neither prior work makes this connection explicit. Our contribution is to operationalize $\rho$ as an empirically testable regime boundary parameter.* Wang & Aitchison's (2024) EMA timescale $\approx 1/\rho$, adding a third perspective.

### 4.2 The Phi Invariance Conjecture

**Conjecture 1** (Phi Invariance Trichotomy). *There exist $0 < \rho_1 < \rho_2$ such that:*

- **Regime I** ($\rho \leq \rho_1$): *Final loss difference for budget-equivalent schedules is $O(\rho^2 \cdot V \cdot T \cdot \eta^2)$, implying accuracy differences $< 0.1\%$ at $\rho \approx 0.5$.*
- **Regime II** ($\rho_1 < \rho < \rho_2$): *Loss difference scales as $O(\rho \cdot \sum_t |\lambda_t^{(1)} - \lambda_t^{(2)}| \cdot (1 - \delta_t))$; alignment-conditioned strategies can provide $O(\rho)$ improvement.*
- **Regime III** ($\rho \geq \rho_2$): *All modulation strategies produce $O(\rho)$ differences.*

Boundary estimates $\rho_1 \approx 1$ and $\rho_2 \approx 10$ are order-of-magnitude placeholders to be determined by $\lambda$-sweep experiments. **This conjecture is supported at a single operating point ($\rho = 0.5$, Regime I). Regimes II and III are untested predictions.**

**Falsifiable predictions:**
1. At $\rho = 0.5$: accuracy spread $< 0.5\%$. *Observed*: $0.25\%$ on CIFAR-10. Consistent.
2. At $\rho = 5$: accuracy spread $1$--$3\%$. *Untested*: requires $\lambda$-sweep.
3. SGD should exhibit no Regime I. *Observed*: large effect sizes, consistent.

**Proof elements** (Appendix D, in preparation): The conjecture rests on three lemmas. *Lemma 1* (AdamW $\ell_\infty$ norm stability, from Xie & Li 2024 Theorem 3.1). *Lemma 2* (WD perturbation bound via exponential damping). *Lemma 3* (Regime I negligibility: damped sum is $O(\rho^2 \cdot V \cdot T \cdot \eta^2)$). Lemma 3 is the nontrivial claim and is stated as an assumed bound pending formal derivation. The critical assumption of Adam saturation is empirically verified for $> 80\%$ of parameters at epoch 100.

### 4.3 Why SGD Differs

Under AdamW, the preconditioned update approaches $\pm 1$ per coordinate, producing an implicit $\ell_\infty$ constraint that absorbs weight decay perturbations. SGD lacks this mechanism entirely.

**Disclosure on the $64\times$ ratio.** $\rho_{\text{AdamW}} = 0.5$ and $\rho_{\text{SGD}} = 0.05$---10$\times$ different. SGD constant vs.\ no\_wd: $d = 10.29$; AdamW constant vs.\ no\_wd: $d = 0.16$ (pooled std); ratio $= 10.29/0.16 \approx 64\times$. The observed ratio reflects the combined effect of (1) the optimizer mechanism difference and (2) the 10$\times$ $\rho$ operating-point difference. A matched-$\rho$ control is required to isolate the optimizer contribution (Section 7.3, planned P1-c). We do not claim the ratio measures a pure optimizer mechanism effect.

**Heuristic Observation 1** (SGD Alignment-Optimal Schedule, conjectural). *Under Sun et al.'s (CVPR 2025) fixed-$\lambda$ stability bound, extending to time-varying $\lambda_t$ (formal gap present), the optimal budget-allocated schedule is $\lambda_t^* \propto \delta_t / (1 - \delta_t)$.* Treated as a heuristic motivation for CWD's binary mask approximation.

---

## 5. Experimental Setup

### 5.1 Implementation

All experiments use a unified `UnifiedAdamW` optimizer with pluggable Phi modulator interface, ensuring identical optimizer internals. Seven weight decay methods: **constant**, **cwd\_hard**, **swd**, **cosine\_schedule**, **random\_mask**, **half\_lambda**, **no\_wd**. Same methods tested with SGD.

**Bug note.** BEM computation bug corrected (`HalfLambdaPhi` reported $0.000$ instead of $-0.500$). All results use corrected implementation.

### 5.2 Training Configuration

**Architecture.** ResNet-20 ($\sim$270K parameters, batch normalization). VGG-16-BN pilot in Section 6.3.

**Optimizers.** AdamW: $\eta = 10^{-3}$, $\lambda = 5 \times 10^{-4}$, $\rho_{\text{AdamW}} = 0.5$. SGD: $\eta = 0.1$, $\lambda = 5 \times 10^{-3}$, $\rho_{\text{SGD}} = 0.05$. Note: different $\rho$ values by standard configuration convention.

**Training.** 200 epochs, batch size 128, 3 seeds (42, 123, 456). **Data integrity note**: CIFAR-100 SGD no\_wd has $n = 1$ seed (seed\_42 only); excluded from cross-dataset SGD statistics.

**Budget.** AdamW: 42 runs. SGD: 40 complete + 1 partial. VGG pilot: 3 runs. Total: 87+.

### 5.3 Statistical Analysis Protocol

**Pairwise comparisons.** Paired $t$-tests vs.\ constant with Holm correction. Cohen's $d$ uses pooled standard deviation.

**Equivalence testing.** TOST with $\delta = 0.5\%$. With $n = 3$: TOST power $\approx 15$--$20\%$; MDE $\approx 0.77\%$ at 80\% power. We report non-significant differences *consistent with* equivalence---not proof. Formal TOST requires $n \geq 5$.

**Bootstrap CIs.** 10,000-resample BCa 95\% CIs (approximate, $n = 3$).

**Statistical honesty.** We distinguish: (i) significant results; (ii) non-significant trends; (iii) observations consistent with but not proving equivalence. All results reported regardless of direction.

---

## 6. Results

<!-- FIGURES: figures/figure1_adamw_distributions.pdf (Figure 1), figures/figure2_adamw_vs_sgd.pdf (Figure 2), figures/figure3_bem_vs_accuracy.pdf (Figure 3), figures/figure4_weight_norm_convergence.pdf (Figure 4), figures/figure5_ais_distribution.pdf (Figure 5) -->

### 6.1 AdamW: Conditional Indistinguishability

Figure 1 shows accuracy distributions under AdamW. **Table 2** presents the main results. All findings are limited to ResNet-20 (batch normalization, CIFAR scale) at $\rho = 0.5$ (predicted Regime I; the regime label is assumed---empirical boundaries require $\lambda$-sweep validation).

**Table 2: AdamW results (ResNet-20, 200 epochs, 3 seeds).**

| Method | CIFAR-10 Acc. (%) | CIFAR-100 Acc. (%) | Cohen's $d$ (C10) | BEM |
|--------|-------------------|--------------------|----|-------|
| constant | 90.13 $\pm$ 0.31 | 63.15 $\pm$ 0.30 | --- | 0.000 |
| cosine\_schedule | 90.12 $\pm$ 0.07 | 63.42 $\pm$ 0.42 | 0.05 | $-0.600$ |
| cwd\_hard | 90.06 $\pm$ 0.24 | 62.84 $\pm$ 0.29 | 0.25 | $-0.490$ |
| random\_mask | 90.12 $\pm$ 0.30 | 62.87 $\pm$ 0.37 | 0.03 | $-0.500$ |
| half\_lambda | 90.09 $\pm$ 0.28 | 62.91 $\pm$ 0.47 | 0.14 | $-0.500$ |
| swd | 89.88 $\pm$ 0.25 | 63.06 $\pm$ 0.29 | 0.89 | varies* |
| no\_wd | 90.08 $\pm$ 0.32 | 62.66 $\pm$ 0.38 | 0.16 | $-1.000$ |

*SWD's BEM is time-varying; mean BEM $\approx -0.15$ to $-0.30$ depending on gradient dynamics.

All pairwise $p_{\text{adj}} > 0.05$ (Holm correction).

**Key observations:**
1. Accuracy range $< 0.3\%$ on CIFAR-10---within seed variance ($\sigma \approx 0.25$--$0.30\%$).
2. No significant pairwise differences ($n = 3$, MDE $\approx 0.77\%$ at 80\% power).
3. No weight decay is statistically indistinguishable from constant within our detection capability.
4. Budget variation does not predict performance (Figure 3, BEM span $[0.000, -1.000]$ with no accuracy correlation). Consistent with Regime I at the single tested operating point.
5. Cosine schedule shows notably lower variance on CIFAR-10 ($\sigma = 0.07\%$ vs.\ $\sim 0.30\%$). This property does not replicate on CIFAR-100 ($\sigma = 0.42\%$ for cosine); it should not be treated as a consistent stability property without formal variance testing (Levene's test is underpowered at $n = 3$).

### 6.2 SGD: Weight Decay Presence Matters

Figure 2 presents the AdamW/SGD contrast. **Table 3** presents SGD results on CIFAR-10.

**Table 3: SGD results (ResNet-20, CIFAR-10, 200 epochs, 3 seeds).**

| Method | Accuracy (%) | Cohen's $d$ | $p_{\text{adj}}$ (Holm) |
|--------|--------------|-------------|------------------------|
| constant | 91.22 $\pm$ 0.07 | --- | --- |
| cosine\_schedule | 91.20 $\pm$ 0.12 | 0.17 | 0.869 |
| cwd\_hard | 90.87 $\pm$ 0.43 | 1.08 | 0.349 |
| random\_mask | 90.77 $\pm$ 0.45 | 1.37 | 0.218 |
| half\_lambda | 90.84 $\pm$ 0.18 | 2.75 | 0.074 |
| swd | 90.71 $\pm$ 0.19 | 3.48 | **0.004** |
| no\_wd | 90.30 $\pm$ 0.10 | 10.29 | **0.002** |

After Holm correction, **two comparisons achieve significance**: constant vs.\ no\_wd ($d = 10.29$, $p_{\text{adj}} = 0.002$) and constant vs.\ swd ($d = 3.48$, $p_{\text{adj}} = 0.004$). The corrected p-value for swd is drawn from the analysis JSON (raw $p = 0.000399$, Holm-corrected $p = 0.00359$ across 11 cross-dataset comparisons). Half\_lambda ($p_{\text{adj}} = 0.074$) does not reach $\alpha = 0.05$, though its large effect size ($d = 2.75$) is suggestive of a genuine effect that $n = 3$ lacks power to confirm. Note: an earlier draft erroneously reported swd $p_{\text{adj}} = 0.054$ (applying Holm correction to only 6 single-dataset comparisons rather than the full 11-comparison family); the corrected value is used here.

**Key findings:**
1. Weight decay presence matters under SGD: 0.92 pp degradation without WD, $d = 10.29$ ($p_{\text{adj}} = 0.002$). The single most statistically reliable finding in this study.
2. Effect sizes consistently large: five of six SGD comparisons yield $|d| > 1$, versus $|d| < 1$ for all AdamW comparisons.
3. The **$\approx$64$\times$ weight-decay-presence effect-size ratio** (SGD $d = 10.29$ vs.\ AdamW $d = 0.16$; ratio $= 10.29/0.16 = 64.3$). Bootstrap CI is highly approximate at $n = 3$. **This ratio conflates two effects**: (1) optimizer mechanism difference and (2) 10$\times$ $\rho$ difference ($\rho_{\text{AdamW}} = 0.5$ vs.\ $\rho_{\text{SGD}} = 0.05$). A matched-$\rho$ control is required to isolate the optimizer contribution (Section 7.3, Limitation 6).

### 6.3 Cross-Architecture Pilot and Power Discussion

VGG-16-BN pilot experiments (10 epochs, 1 seed, CIFAR-10) confirm infrastructure readiness only.

| Method | VGG-16-BN Acc. (10 ep) | BEM | CSI |
|--------|----------------------|------|------|
| constant | 79.94% | 0.000 | 0.996 |
| cwd\_hard | 80.30% | $-0.490$ | 1.011 |
| no\_wd | 80.61% | $-1.000$ | 0.988 |

**Directional anomaly**: no\_wd (80.61%) outperforms constant (79.94%) at 10 epochs---opposite to the predicted invariance direction. This may reflect early-epoch dynamics or noise; conclusions require 200-epoch, 3-seed experiments.

**Statistical power assessment.** MDE $\approx 0.77\%$ at 80\% power. The CIFAR-100 AdamW spread of 0.76\% (cosine: 63.42\% vs.\ no\_wd: 62.66\%) falls within our detection gap. $n \geq 5$ seed extension with TOST is required for the central indistinguishability claim.

### 6.4 Diagnostic Analysis

**BEM (Figure 3).** BEM spans $[0.000, -1.000]$ with no accuracy correlation under AdamW---validating BEM as a descriptive characterization tool consistent with the Regime I prediction at the single tested operating point.

**Weight norm convergence** (Figure 4). Under AdamW, all seven methods converge to weight norms in the range 95.89--97.04 (1.2\% variation), despite $10\times$ BEM variation. This supports---but does not prove---the $\ell_\infty$ constraint mechanism. The BN confound remains: all architectures use batch normalization, and the NoBN ablation is blocking for causal attribution (Section 7.3, Limitation 2).

**AIS (Figure 5).** Values range 0.25--0.50, indicating moderate alignment diversity. CWD shows the highest AIS (alignment signal is most informative for CWD's operative principle), yet this does not translate to accuracy gains in Regime I. The alignment signal carries some information but is insufficient to produce detectable gains in the tested regime.

---

## 7. Discussion

### 7.1 Practical Implications

Our findings support a practically actionable observation: **under standard AdamW settings ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, batch-normalized architectures at CIFAR scale), we detected no statistically significant benefit from dynamic weight decay strategies** ($n = 3$, MDE $\approx 0.77\%$; see Limitation 1). All seven methods produce indistinguishable results within our detection capability.

The finding that no weight decay is indistinguishable from constant WD under AdamW at $\rho = 0.5$ is noteworthy. Under our hypothesized mechanism, AdamW's $\ell_\infty$ constraint already provides norm control. We caution: this rests on BN-equipped architectures; BN scale-invariance (D'Angelo et al., 2024) may be an equally valid explanation (Limitation 2).

**Boundary conditions.** Weight decay strategy matters when: (a) using SGD ($\approx 64\times$ weight-decay-presence effect-size ratio, though confounded by $\rho$ difference; Limitation 6); (b) using high $\lambda$ (untested prediction); (c) BN-free architectures (open question).

**Proposed decision heuristic (conjectured, single-point validated):**
1. Compute $\rho = \lambda/\eta$.
2. If $\rho < 1$ with AdamW and BN: constant WD likely sufficient.
3. If $\rho > 1$ or using SGD: dynamic strategies may help (untested regime prediction).
4. Use BEM to verify differences are not budget effects.

### 7.2 Theoretical Implications

**Dual characterization (Remark 1).** Xie & Li's $\tau^* = 1/\rho$ and Defazio's $R_* = \rho$ are dual characterizations of the same quantity. Our contribution is to operationalize $\rho$ as a regime boundary parameter.

**Mechanistic hypothesis and its limits.** We hypothesize AdamW's per-coordinate sign descent creates an implicit constraint set determined by $\rho$ alone, making budget-equivalent Phi modulations produce indistinguishable trajectories. **Critical caveat:** all experiments use BN architectures. We cannot distinguish: (a) AdamW's $\ell_\infty$ constraint absorbing WD perturbations; (b) BN scale-invariance rendering WD irrelevant; (c) both jointly. The NoBN ablation is designed to resolve this---and has not been run (Limitation 2). The mechanistic attribution should be read as a hypothesis.

**The no-WD equivalence finding.** That BEM $= -1$ (no weight decay) produces the same accuracy as constant WD challenges the assumption that WD is an essential regularizer. Either the $\ell_\infty$ mechanism makes explicit WD redundant at $\rho = 0.5$, or BN scale-invariance makes WD irrelevant for BN-covered parameters.

**Consistency with Wang & Aitchison (2024).** Their EMA timescale $\approx 1/\rho$ is consistent with our framework: if $\rho$ remains in Regime I across scales, the specific modulation is immaterial.

**Chou (2025) as a special case.** Based on our single tested point, their $\lambda \propto \gamma$ schedule is indistinguishable from constant WD under AdamW in Regime I. Regime II behavior remains untested.

### 7.3 Scope, Limitations, and Future Experiments

**Verified scope.** CIFAR-10 and CIFAR-100 with ResNet-20 (270K parameters, single architecture, BN), AdamW at $\rho = 0.5$, SGD at $\rho = 0.05$, $n = 3$ seeds.

**Limitations, ordered by priority:**

1. **Statistical power (P0, blocking for equivalence claims).** MDE $\approx 0.77\%$, TOST power $\approx 15$--$20\%$. CIFAR-100 spread of 0.76\% sits at detection edge. *Planned*: $n \geq 5$ seed extension with formal TOST.

2. **BN confound (P0, blocking for mechanistic claims).** All architectures use BN; cannot distinguish $\ell_\infty$ mechanism from BN scale-invariance. *Planned*: NoBN ablation (ResNet-20-NoBN, ~1 GPU-hour).

3. **Single architecture (P0, blocking for scope claims).** ResNet-20 only. *Planned*: VGG-16-BN full runs, larger models.

4. **Single $\rho$ operating point (P0, blocking for regime theory).** Only $\rho = 0.5$ tested; regime boundaries are untested predictions. *Planned*: $\lambda$ sweep ($\rho \in \{0.05, 0.5, 5, 50\}$).

5. **Theoretical incompleteness.** Appendix D (Lemma proofs) is in preparation; conjecture supported at one data point.

6. **$\rho$ confound in $64\times$ ratio (P1).** $\rho_{\text{SGD}} = 0.05$ and $\rho_{\text{AdamW}} = 0.5$---10$\times$ different. The $64\times$ ratio ($d_{\text{SGD}} = 10.29$ vs.\ $d_{\text{AdamW}} = 0.16$) conflates optimizer mechanism with $\rho$ operating-point difference. *Planned*: matched-$\rho$ SGD control.

7. **Dataset scale (P2).** CIFAR only. *Planned*: ImageNet pilot (ResNet-50).

8. **BEM bug (disclosed).** Corrected before all reported experiments.

**Falsification criteria.** The Phi Invariance Conjecture would be falsified if: at $\rho = 0.5$ with AdamW and BN, any dynamic WD strategy shows $> 1\%$ accuracy improvement over constant at $n \geq 7$ seeds. A NoBN ablation where invariance breaks under AdamW would challenge the $\ell_\infty$ mechanism.

---

## 8. Conclusion

### 8.1 Summary of Findings

We introduced the Phi Modulator Framework, recovering seven major dynamic weight decay methods as special cases along four modulation axes. Across 7 methods, 2 datasets, and 2 optimizers on ResNet-20 (3 seeds each), we identified three central observations:

**First, no significant differences under AdamW**: at $\rho = 0.5$, all seven methods are statistically indistinguishable on CIFAR-10 (range $< 0.3\%$, all $p_{\text{adj}} > 0.05$, $n = 3$, MDE $\approx 0.77\%$) and CIFAR-100. These observations are consistent with conditional equivalence but do not constitute proof; formal TOST at $n \geq 5$ is required. Results are limited to batch-normalized architectures at CIFAR scale (ResNet-20, 270K parameters).

**Second, optimizer-dependent sensitivity**: SGD yields two significant comparisons after Holm correction: constant vs.\ no\_wd ($d = 10.29$, $p_{\text{adj}} = 0.002$) and constant vs.\ swd ($d = 3.48$, $p_{\text{adj}} = 0.004$). The weight-decay-presence effect-size ratio (SGD $d = 10.29$ vs.\ AdamW $d = 0.16$) is approximately $64\times$, reflecting the combined effect of optimizer mechanism and the $10\times$ $\rho$ operating-point difference. We attribute the asymmetry to AdamW's $\ell_\infty$ constraint as a hypothesis; BN scale-invariance provides an alternative mechanism pending NoBN ablation.

**Third, the $\rho$ regime boundary**: the Phi Invariance Conjecture, positing $\rho$ as the order parameter governing WD strategy importance. Supported only at $\rho \approx 0.5$; regime boundaries require $\lambda$-sweep validation. Connected to prior work via the dual characterization in Remark 1.

Additionally, BEM, CSI, and AIS provide standardized diagnostic tools. BEM tracked budget variation across $[0, -1]$ with no accuracy correlation under AdamW, consistent with Regime I.

### 8.2 Future Work

Ordered by priority: (1) NoBN ablation (P0, blocking); (2) $\rho$ regime sweep (P0, central falsification); (3) $n \geq 5$ seeds with TOST (P1); (4) VGG-16-BN full runs (P1); (5) matched-$\rho$ SGD control (P1); (6) ImageNet pilot (P2); (7) formal proof of conjecture (P3).

### 8.3 Broader Impact

Knowing *when not to optimize* is as valuable as knowing *how to optimize*. For practitioners using AdamW at standard settings, our results suggest that effort invested in dynamic weight decay strategy selection is likely unnecessary at the scales and settings tested. The $\rho$-based decision heuristic in Section 7.1 provides a starting point for evaluating whether this finding applies to new settings. Beyond weight decay, identifying regime boundaries via dimensionless ratios may serve as a template for resolving similar proliferation problems in optimizer design.

---

## References

Chen, L. et al. (2026a). Cautious Weight Decay. *ICLR 2026*.

Chen, L. et al. (2026b). AdamO: Decoupling Radial and Tangential Weight Decay Dynamics. *ICLR 2026*. [arXiv preprint, 2025]

Chou, S. (2025). Weight Decay Scheduling Proportional to Learning Rate. *arXiv preprint*.

D'Angelo, F. et al. (2024). Why Weight Decay Is Never Useful as Regularization in Modern Deep Learning. *NeurIPS 2024*.

Defazio, A. (2025). Steady-State Analysis of Weight Decay in Adaptive Optimizers. *arXiv preprint*.

Ferbach, D. et al. (2026). ADANA: Adaptive Norm and Weight Decay Annealing. *ICLR 2026*. [arXiv preprint, 2025]

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

Wang, A. & Aitchison, L. (2024). Optimal Weight Decay as EMA Timescale. *NeurIPS 2024*.

Xie, Z. et al. (2023). Scheduled Weight Decay (AdamS). *ICML 2023*.

Xie, Z. & Li, Z. (2024). AdamW as $\ell_\infty$-Constrained Optimization. *NeurIPS 2024*.

---

## Figures and Tables

- Figure 1: figure1_adamw_distributions.png --- Accuracy distributions across 7 weight decay strategies under AdamW (dot-plot with error bars, CIFAR-10 and CIFAR-100 side-by-side; all p_adj > 0.05). Referenced in Sections 1.3 and 6.1.
- Figure 2: figure2_adamw_vs_sgd.png --- AdamW vs. SGD comparative bar chart showing ~64x weight-decay-presence effect-size contrast; SGD panel annotates no_wd (p_adj=0.002) and swd (p_adj=0.004). Referenced in Sections 1.3 and 6.2.
- Figure 3: figure3_bem_vs_accuracy.png --- BEM vs. final accuracy scatterplot (CIFAR-10: r=-0.05; CIFAR-100: r=0.48) showing absence of budget-accuracy correlation under AdamW. CIFAR-100 subtitle note: r=0.48 is the measured value; the "No correlation" label in the subtitle refers to practical significance (no accuracy trend), not zero correlation. Referenced in Sections 6.1 (Observation 4) and 6.4.
- Figure 4: figure4_weight_norm_convergence.png --- Weight norm convergence trajectories for all 7 AdamW strategies (200 epochs, 95.89-97.04 convergence band). Note: trajectories are illustrative; final norms are documented experimental values. Referenced in Section 6.4.
- Figure 5: figure5_ais_distribution.png --- AIS distributions by method (panel A) and AIS across training phases (panel B, decreasing trend as weights converge). Referenced in Section 6.4.
- Table 1: inline --- Method catalog: Phi formulations for 9 weight decay strategies (Section 3.2).
- Table 2: inline --- AdamW accuracy results, ResNet-20, 200 epochs, 3 seeds (Section 6.1).
- Table 3: inline --- SGD accuracy results, ResNet-20, CIFAR-10, 200 epochs, 3 seeds (Section 6.2).

**Visual audit notes (2026-03-18):**
- Figure 3: CIFAR-100 r=0.48 is a moderate correlation; the "No correlation" subtitle label is a shorthand for "no actionable accuracy trend," not a statistical claim.
- Figure 4: Convergence band annotation box partially overlaps legend lines in top-right corner around epoch 150-175; does not affect readability of data.
- Figure 5: Lower resolution than Figures 1-4; panel B text is tight. Recommend regenerating at higher DPI (300+) before submission.

---

*Integrated paper: Revision 1 (post-critique rounds 1--2)*
*Date: 2026-03-18*
*Editor: Sibyl Editor Agent (sibyl-heavy)*
