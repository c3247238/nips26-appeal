# 2. Related Work

## 2.1 Weight Decay as a Dynamics Modifier

The classical view of weight decay treats it as L2 regularization---a penalty term $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$ added to the loss function (Krogh & Hertz, 1991). Loshchilov & Hutter (2019) demonstrated that in adaptive optimizers, L2 regularization and decoupled weight decay produce fundamentally different behaviors, because the gradient of the L2 penalty is rescaled by Adam's per-parameter second-moment estimate. Their AdamW formulation has since become the default optimizer for modern deep learning.

D'Angelo et al. (2024) provided the most comprehensive re-evaluation: weight decay is *never* useful as explicit regularization in modern settings. Under SGD, weight decay stabilizes loss trajectories by preventing unbounded weight norm growth. Under near-single-epoch LLM training, it controls the bias-variance tradeoff. Kosson et al. (2023) showed that weight decay induces a *rotational equilibrium* balancing average rotation across layers and neurons, explaining AdamW's advantage over Adam+L2. Xie & Li (2024) proved that AdamW implicitly performs $\ell_\infty$-norm constrained optimization, connecting decoupled weight decay to the Frank-Wolfe algorithm.

These modern interpretations suggest that the *scheduling* and *modulation* of weight decay should matter, since it is the training dynamics---not regularization strength---that weight decay primarily controls.

## 2.2 Dynamic Weight Decay Methods

We organize existing methods into four families based on their modulation axis.

**Temporal scheduling.** Xie et al. (2023) introduced Scheduled Weight Decay (SWD/AdamS), adjusting weight decay based on gradient norms. Ferbach et al. (2026) proposed ADANA, applying logarithmic-time schedules to weight decay alongside momentum coefficients $\beta_1$ and $\beta_2$, reporting up to 40% compute efficiency gains. Standard cosine and linear weight decay schedules are widely used in practice but rarely studied in isolation.

**Alignment-aware modulation.** Chen et al. (2026a) proposed Cautious Weight Decay (CWD), applying a binary sign-alignment mask: weight decay acts on parameter $\theta_i$ only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$, where $u_i$ is the optimizer update. CWD achieves a bilevel Pareto-optimal interpretation and exhibits sliding-mode behavior. Chen et al. (2026b) identified the "Radial Tug-of-War" conflict and proposed AdamO to decouple radial and tangential dynamics. Tian et al. (2024) introduced Selective Projection Decay (SPD) for fine-tuning.

**Norm-matched control.** Loshchilov (2023) generalized decoupled weight decay to Weight Norm Control (AdamWN), driving parameters toward an arbitrary target norm $\tau$ rather than zero. He et al. (2025) proposed AlphaDecay, assigning module-wise decay rates via heavy-tailed spectral density analysis at LLM scales (60M--1B parameters). Wang & Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant across model and dataset sizes. Chou (2025) derived WD proportional to $\gamma^2$ for stable weight norms.

**Structural effects.** Galanti et al. (2022) demonstrated that SGD with weight decay induces low-rank bias in weight matrices. Kobayashi et al. (2024) showed that L2 regularization on attention layers is equivalent to nuclear norm regularization. Truong & Truong (2026) analyzed how weight decay traverses a norm hierarchy from shortcut to structured representations.

## 2.3 Evaluation Fragmentation

CWD is evaluated with Lion, Muon, and AdamW on language model pre-training; SWD targets the SGD-Adam generalization gap on CIFAR and ImageNet; AlphaDecay operates at LLM scales; AdamO demonstrates improvements on specific benchmarks. Each paper uses different architectures, datasets, optimizers, and hyperparameter protocols.

Fernandez-Hernandez et al. (2025) proposed the Overfitting-Underfitting Indicator (OUI) as a per-method diagnostic, but it does not enable cross-method comparison. D'Angelo et al. (2024) provided the `why-weight-decay` codebase but did not compare dynamic scheduling strategies.

This fragmentation motivates our standardized metrics---BEM, CSI, and AIS---and our systematic benchmark evaluating all methods under identical conditions within a unified codebase, across both AdamW and SGD optimizers and multiple architectures.

<!-- FIGURES
- None
-->
