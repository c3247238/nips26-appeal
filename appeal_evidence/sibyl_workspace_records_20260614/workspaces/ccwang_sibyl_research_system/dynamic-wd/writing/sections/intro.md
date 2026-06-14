# 1. Introduction

## 1.1 Motivation

Weight decay is among the most universally applied techniques in deep learning optimization. Virtually every modern training recipe---from small-scale CIFAR classifiers to billion-parameter language models---includes a weight decay coefficient as a core hyperparameter. Yet despite its ubiquity, the community lacks a principled framework for choosing *how* to apply weight decay over the course of training.

The classical understanding treats weight decay as explicit L2 regularization that shrinks weights toward zero, discouraging model complexity (Krogh & Hertz, 1991). Loshchilov & Hutter (2019) demonstrated that L2 regularization and weight decay are not equivalent in adaptive optimizers, leading to the now-standard AdamW formulation that *decouples* weight decay from gradient scaling. D'Angelo et al. (2024) showed that weight decay is never useful as explicit regularization in modern deep learning; instead, it acts as a *training dynamics modifier*---stabilizing loss trajectories under SGD and controlling the bias-variance tradeoff in large language model training.

This re-understanding has sparked a surge of methods proposing to dynamically modulate weight decay strength during training. Xie et al. (2023) introduced Scheduled Weight Decay (SWD), which adjusts weight decay based on gradient norms. Chen et al. (2026a) proposed Cautious Weight Decay (CWD), a sign-alignment mask that applies decay only when weight and update directions agree. Loshchilov (2023) generalized decoupled weight decay to target-norm control (AdamWN). He et al. (2025) introduced AlphaDecay, a module-wise strategy guided by spectral density analysis. Ferbach et al. (2026) proposed logarithmic-time schedules (ADANA) for weight decay alongside momentum coefficients. Chen et al. (2026b) identified a "Radial Tug-of-War" conflict between weight decay and gradient updates, proposing AdamO to decouple radial and tangential dynamics.

A critical problem pervades this rapidly growing literature: **each method is evaluated in isolation**, using different architectures, datasets, optimizers, hyperparameter selection protocols, and evaluation metrics. CWD reports improvements on language model pre-training with Lion and Muon optimizers; SWD demonstrates gains with SGD on CIFAR and ImageNet; AlphaDecay targets billion-parameter LLMs. No two papers share the same experimental conditions, making direct comparison impossible and leaving practitioners unable to determine which---if any---of these dynamic strategies actually helps.

## 1.2 Research Gap

We identify four critical gaps in the current weight decay literature:

**No unified mathematical framework.** The four major families of dynamic weight decay---temporal scheduling (SWD, ADANA), alignment-aware modulation (CWD, AdamO), norm-matched control (AdamWN, AlphaDecay), and spatial modulation---each operate with independent mathematical formulations. CWD uses a bilevel Pareto-optimality interpretation; AdamWN defines a target-norm control law; SWD applies gradient-norm-based scheduling. No existing work reveals whether they are fundamentally different or special cases of a single principle.

**No standardized evaluation metrics.** Each paper reports different metrics under different conditions. There is no standard way to quantify effective weight decay budget, coupling stability, or alignment informativeness across methods.

**No controlled systematic comparison.** No prior work evaluates multiple dynamic weight decay methods within a single codebase, under identical hyperparameters and training conditions, with proper statistical testing across multiple architectures and optimizers.

**No theory for when dynamic weight decay matters.** D'Angelo et al. (2024) showed that weight decay acts as a dynamics modifier, implying that its scheduling should matter. Yet no theoretical framework predicts *when* a particular scheduling strategy would outperform constant weight decay.

## 1.3 Contributions

We make the following contributions:

1. **The Phi Modulator Framework.** We introduce a unified mathematical interface $\varphi(t, \boldsymbol{\theta}, \mathbf{g})$ that expresses the weight decay update as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon) - \lambda \cdot \varphi(t, \boldsymbol{\theta}_t, \mathbf{g}_t) \odot \boldsymbol{\theta}_t$. CWD, SWD, cosine schedules, AdamWN, AlphaDecay, and all their compositions are recovered as special cases along four modulation axes: temporal, directional, spatial, and target-norm.

2. **Three diagnostic metrics.** We propose the Budget Equivalence Metric (BEM), Coupling Stability Index (CSI), and Alignment Informativeness Score (AIS)---the first standardized tools for quantifying how weight decay methods differ in effective budget, coupling stability, and alignment informativeness.

3. **Systematic multi-architecture, multi-optimizer benchmark.** We conduct 126 controlled experiments spanning 7--8 methods, 3 seeds, 2 datasets, 2 architectures (ResNet-20, VGG-16-BN), and 2 optimizers (AdamW, SGD). All methods share identical training infrastructure and base hyperparameters, isolating the effect of the Phi modulator.

4. **The Phi Invariance Conjecture.** Our evaluation reveals that all dynamic weight decay variants are statistically equivalent to constant weight decay under AdamW---but not under SGD. We formalize this as the *Phi Invariance Conjecture*: AdamW's per-parameter adaptive scaling subsumes the effect of any Phi modulator. SGD experiments confirm the conjecture's predicted boundary condition: without adaptive scaling, weight decay modulation affects weight norms and generalization gaps.

## 1.4 Paper Roadmap

Section 2 surveys related work across the four families of dynamic weight decay methods. Section 3 introduces the Phi Modulator Framework, its formal properties, and the three diagnostic metrics. Section 4 describes the experimental setup. Section 5 presents the main results and diagnostic analysis across both AdamW and SGD. Section 6 discusses the Phi Invariance Conjecture, its boundary conditions, and limitations. Section 7 concludes.

<!-- FIGURES
- None
-->
