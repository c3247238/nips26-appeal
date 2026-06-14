# 1. Introduction

## 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). However, this view has been progressively undermined by modern findings. Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. More recently, D'Angelo et al. (2024) provided a unifying perspective showing that weight decay is never useful as explicit regularization in modern deep learning; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories under SGD and controlling the bias-variance tradeoff in large language model training.

This re-understanding has sparked a surge of methods proposing to dynamically modulate weight decay strength during training. Xie et al. (2023) introduced Scheduled Weight Decay (SWD), which adjusts weight decay based on gradient norms. Chen et al. (2026a) proposed Cautious Weight Decay (CWD), a sign-alignment mask that applies decay only when weight and update directions agree. Loshchilov (2023) generalized decoupled weight decay to target-norm control (AdamWN), while He et al. (2025) introduced AlphaDecay, a module-wise strategy guided by spectral density analysis. Ferbach et al. (2026) proposed logarithmic-time schedules (ADANA) for weight decay alongside momentum coefficients. Chen et al. (2026b) identified a "Radial Tug-of-War" conflict between weight decay and gradient updates, proposing AdamO to decouple radial and tangential dynamics.

A critical problem pervades this rapidly growing literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. CWD reports improvements on language model pre-training with Lion and Muon optimizers; SWD demonstrates gains with SGD on CIFAR and ImageNet; AlphaDecay targets billion-parameter LLMs. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps.

This state of affairs raises a fundamental question: **does dynamic weight decay actually help, and if so, when and why?**

## 1.2 Research Gap

We identify four critical gaps in the current weight decay literature:

**No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), alignment-aware modulation (CWD, AdamO), norm-matched control (AdamWN, AlphaDecay), and spatial modulation---each operate with independent mathematical formulations. CWD uses a bilevel Pareto-optimality interpretation; AdamWN defines a target-norm control law; SWD applies gradient-norm-based scheduling. These are all answering the same question---*how should weight decay interact with the training trajectory?*---but from incompatible starting points. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify how much effective weight decay budget a method uses, how stably it couples with the optimizer, or whether its alignment signal actually carries useful information for training. This fragmentation is the root cause of conflicting claims across the literature.

**No controlled systematic comparison.** To our knowledge, no prior work has evaluated multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing. Without such a comparison, reported improvements may be artifacts of hyperparameter tuning, architectural choices, or optimizer selection rather than genuine benefits of the dynamic strategy.

**No theory for when dynamic weight decay matters.** D'Angelo et al. (2024) showed that weight decay acts as a dynamics modifier rather than a classical regularizer, implying that its scheduling should matter. Yet no theoretical framework predicts *when* and *why* a particular scheduling strategy would outperform constant weight decay.

## 1.3 Contributions

We make the following contributions:

1. **The Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that expresses the weight decay update as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\hat{\mathbf{v}}_t^{1/2} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$. We show that CWD, SWD, cosine schedules, AdamWN, AlphaDecay, and all their compositions are recovered as special cases of this framework along four modulation axes: temporal, directional, spatial, and target-norm. **The Phi framework enables, for the first time, controlled comparison of weight decay strategies under identical optimization conditions.**

2. **Three diagnostic metrics.** We propose the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---the first standardized tools for quantifying how weight decay methods differ in effective budget, coupling stability, and alignment informativeness. These metrics characterize *how* methods differ even when they produce identical accuracy.

3. **Systematic benchmark.** We conduct 42 controlled experiments (7 methods $\times$ 3 seeds $\times$ 2 datasets) with identical hyperparameters, training infrastructure, and statistical testing. All methods share the same AdamW base optimizer, learning rate schedule, and base weight decay coefficient, isolating the effect of the Phi modulator.

4. **The Phi Invariance Conjecture.** Our systematic evaluation reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW. We formalize this as the *Phi Invariance Conjecture*: AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator, with diagnostic evidence from weight norm convergence, alignment informativeness analysis, and budget equivalence sweeps.

## 1.4 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods. Section 3 introduces the Phi Modulator Framework, its formal properties, and the three diagnostic metrics. Section 4 describes the experimental setup. Section 5 presents the main results and diagnostic analysis. Section 6 discusses the Phi Invariance Conjecture, its implications, and limitations. Section 7 concludes. Appendices provide extended statistical analysis, diagnostic visualization panels for all 42 runs, mathematical proofs, and reproducibility details.
