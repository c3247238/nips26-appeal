# When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW

## Abstract

Weight decay is universally applied in deep learning optimization, yet practitioners face a bewildering proliferation of dynamic weight decay strategies---Cautious Weight Decay (CWD), Scheduled Weight Decay (SWD), cosine schedules, AlphaDecay, ADANA, AdamO---with no systematic understanding of *when* these methods provide genuine benefit. We introduce the **Phi Modulator Framework**, a unified mathematical abstraction $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that recovers all major dynamic weight decay methods as special cases along four modulation axes: temporal, directional, spatial, and target-norm. Through this lens, we conduct the first controlled comparison of seven weight decay strategies under identical optimization conditions. Our central finding is a **conditional equivalence result**: under standard AdamW settings ($\rho = \lambda/\eta \approx 0.5$), all seven strategies---including the degenerate case of no weight decay---are statistically indistinguishable (accuracy range $< 0.3\%$, all pairwise $p > 0.05$ after Holm correction). In stark contrast, SGD exhibits significant sensitivity to weight decay presence (constant vs.\ no\_wd: Cohen's $d = 10.29$, $p_{\text{adj}} = 0.002$). We formalize these observations through the **Phi Invariance Conjecture**, which posits $\rho = \lambda/\eta$ as the order parameter governing a regime boundary: when $\rho \lesssim 1$, AdamW's implicit $\ell_\infty$ constraint renders all modulation strategies equivalent; as $\rho$ increases, strategy choice becomes progressively important. We further propose three standardized diagnostic metrics---the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---providing the first quantitative tools for characterizing weight decay behavior. Our contributions include: (1) the Phi Modulator four-axis taxonomy, (2) the Phi Invariance Conjecture with falsifiable predictions, (3) the BEM/CSI/AIS evaluation protocol, and (4) systematic cross-optimizer empirical analysis revealing an 18.3$\times$ effect-size ratio between SGD and AdamW for weight decay presence.

---

## 1. Introduction

### 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). However, this view has been progressively undermined by modern findings. Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. More recently, D'Angelo et al.\ (2024) provided a unifying perspective showing that weight decay is never useful as explicit regularization in modern deep learning; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories and controlling bias-variance tradeoffs. Xie \& Li (2024) further revealed that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, connecting decoupled weight decay to the Frank-Wolfe algorithm. Kosson et al.\ (2023) showed that weight decay induces a rotational equilibrium balancing average rotation of weight vectors across layers.

These new understandings have catalyzed a surge of dynamic weight decay methods. Xie et al.\ (2023) introduced Scheduled Weight Decay (SWD), adjusting decay based on gradient norms. Chen et al.\ (2026a) proposed Cautious Weight Decay (CWD), applying a sign-alignment mask. Loshchilov (2023) generalized to target-norm control (AdamWN). He et al.\ (2025) introduced AlphaDecay with module-wise rates guided by spectral density. Ferbach et al.\ (2026) proposed ADANA with logarithmic-time schedules. Chen et al.\ (2026b) identified a ``Radial Tug-of-War'' conflict and proposed AdamO for decoupled radial-tangential dynamics.

A critical problem pervades this rapidly growing literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps in their specific setting.

### 1.2 Research Gap

Answering the question of *when* dynamic weight decay helps is currently impossible due to four critical gaps:

**No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), directional modulation (CWD, AdamO), spatial modulation (AlphaDecay), and target-norm control (AdamWN)---each operate with independent mathematical formulations from incompatible starting points. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify how much effective weight decay budget a method uses, how stably it couples with the optimizer, or whether its alignment signal carries useful training information. This fragmentation produces conflicting claims that cannot be reconciled.

**No controlled systematic comparison.** To our knowledge, no prior work has evaluated multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing including multiple comparison corrections.

**No theory for when dynamic weight decay matters.** Despite insights into weight decay's role as a dynamics modifier (D'Angelo et al., 2024) and its connection to constrained optimization (Xie \& Li, 2024), no theoretical framework predicts *when* the functional form of modulation becomes irrelevant---particularly whether the optimizer's adaptive scaling already subsumes explicit scheduling.

### 1.3 Our Approach: The $\rho = \lambda/\eta$ Lens

We address these gaps by proposing $\rho = \lambda/\eta$---the ratio of weight decay coefficient to learning rate---as the unifying order parameter that governs when dynamic weight decay matters. This ratio has a natural physical interpretation: it measures the relative magnitude of the weight decay step compared to the gradient step. When $\rho$ is small, weight decay is a minor perturbation to gradient-driven optimization; when $\rho$ is large, weight decay competes with or dominates the gradient signal.

Our core finding is striking: at the standard AdamW setting ($\lambda = 5 \times 10^{-4}$, $\eta = 10^{-3}$, giving $\rho = 0.5$), seven weight decay strategies spanning all four modulation axes produce statistically indistinguishable results---including the degenerate case of *no weight decay at all*. Yet under SGD with the same training pipeline, weight decay presence produces a massive effect (Cohen's $d > 10$, $p < 0.003$). We attribute this asymmetry to AdamW's implicit $\ell_\infty$ constraint (Xie \& Li, 2024), which absorbs weight decay perturbations in the low-$\rho$ regime, a mechanism absent in SGD.

This $\rho$-based perspective simultaneously connects to multiple independent recent results: the constraint radius $\tau^* = \eta/\lambda = 1/\rho$ from Xie \& Li (2024), the steady-state gradient-to-weight ratio $R_* \approx \lambda/\eta = \rho$ from Defazio (2025), and the EMA timescale interpretation from Wang \& Aitchison (2024).

### 1.4 Contributions

We make four contributions, ordered by significance:

1. **Empirical discovery: conditional equivalence.** Through 90+ controlled experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets $\times$ 2 optimizers), we establish that all dynamic weight decay strategies are statistically equivalent under standard AdamW settings ($\rho \approx 0.5$), while SGD exhibits significant sensitivity (18.3$\times$ effect-size ratio for weight decay presence). This is a practically actionable finding: practitioners using AdamW at standard settings need not invest effort in weight decay strategy selection.

2. **Theoretical framework: the Phi Invariance Conjecture.** We formalize our findings through the Phi Invariance Conjecture, which posits $\rho = \lambda/\eta$ as the regime boundary parameter with a trichotomy: Regime I ($\rho \lesssim 1$, strategy-invariant), Regime II (transitional), and Regime III ($\rho \gg 1$, strategy-sensitive). The conjecture makes falsifiable predictions at specific $\rho$ values.

3. **Taxonomic contribution: the Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that recovers all major dynamic weight decay methods as special cases along four modulation axes. This framework enables, for the first time, controlled comparison under identical optimization conditions.

4. **Evaluation infrastructure: BEM/CSI/AIS metrics.** We propose three standardized diagnostic metrics---Budget Equivalence Metric, Coupling Stability Index, and Alignment Informativeness Score---as characterization tools for describing how weight decay strategies differ in their operational properties.

### 1.5 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods and positions our contributions against key recent works. Section 3 introduces the Phi Modulator Framework, its formal properties, and the diagnostic metrics. Section 4 presents the theoretical analysis centered on the $\rho$ regime boundary. Section 5 describes the experimental setup. Section 6 presents results. Section 7 provides discussion, practical implications, and limitations. Section 8 concludes.
