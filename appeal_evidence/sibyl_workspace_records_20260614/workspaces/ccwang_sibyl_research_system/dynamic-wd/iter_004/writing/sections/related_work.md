## 2. Related Work

### 2.1 Weight Decay as a Dynamics Modifier

The classical view of weight decay treats it as L2 regularization---a penalty term $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$ added to the loss that shrinks weights toward zero and reduces model complexity (Krogh \& Hertz, 1991; Hanson \& Pratt, 1988). This interpretation guided practice for decades until Loshchilov \& Hutter (2019) demonstrated a crucial distinction: in adaptive optimizers, L2 regularization and decoupled weight decay produce fundamentally different behaviors because the L2 gradient is rescaled by Adam's per-parameter second-moment estimate. Their proposed AdamW---applying weight decay directly to parameters rather than through the gradient---has since become the default optimizer for modern deep learning.

A deeper re-evaluation came from D'Angelo et al.\ (2024), who showed through extensive experiments on both vision models and LLMs that weight decay is *never* useful as explicit regularization in modern settings. Instead, it serves as a training dynamics modifier: under SGD, it prevents weight norms from growing unboundedly; under near-single-epoch LLM training, it controls the bias-variance tradeoff. Kosson et al.\ (2023) provided a complementary perspective, showing that weight decay induces a *rotational equilibrium* balancing the average rotation of weight vectors across layers---a mechanism that may operate robustly regardless of the specific form of weight decay modulation. Xie \& Li (2024) proved that AdamW implicitly performs $\ell_\infty$-norm constrained optimization via a connection to the Frank-Wolfe algorithm, with the constraint radius $\tau^* = \eta/\lambda$. This $\ell_\infty$ constraint is central to our theoretical analysis: it provides the absorption mechanism that we hypothesize renders Phi modulation irrelevant in the low-$\rho$ regime.

These modern interpretations collectively suggest that the *scheduling* and *modulation* of weight decay should matter, since it is the training dynamics---not regularization strength---that weight decay primarily controls. Yet this implication has not been rigorously tested through controlled experiments.

### 2.2 Dynamic Weight Decay Methods

We organize existing dynamic weight decay methods along the four modulation axes of our Phi framework.

**Temporal scheduling.** Xie et al.\ (2023) introduced Scheduled Weight Decay (SWD/AdamS), which adjusts weight decay based on gradient norms, motivated by the observation that constant weight decay can destabilize training during phases of large gradient magnitude. Ferbach et al.\ (2026) proposed ADANA, which applies logarithmic-time schedules to both weight decay and momentum coefficients. Standard cosine and linear weight decay schedules (Loshchilov \& Hutter, 2017), though widely used, have rarely been studied in isolation from learning rate schedules.

**Directional modulation.** Chen et al.\ (2026a) proposed Cautious Weight Decay (CWD), which applies a binary sign-alignment mask: weight decay acts on parameter $\theta_i$ only when $\mathrm{sign}(\theta_i) = \mathrm{sign}(u_i)$, where $u_i$ is the optimizer update. CWD achieves a Pareto-optimal interpretation in a bilevel optimization framework. Chen et al.\ (2026b) identified the ``Radial Tug-of-War'' conflict between weight decay and gradient updates in the radial direction, proposing AdamO to decouple radial and tangential dynamics. Tian et al.\ (2024) introduced Selective Projection Decay (SPD) for fine-tuning.

**Spatial modulation.** He et al.\ (2025) proposed AlphaDecay, which assigns module-wise decay rates guided by heavy-tailed self-regularization spectral density analysis, demonstrating gains at LLM scales from 60M to 1B parameters.

**Target-norm control.** Loshchilov (2023) generalized decoupled weight decay to Weight Norm Control (AdamWN), driving parameters toward an arbitrary target norm $\tau$ rather than zero, subsuming standard weight decay ($\tau = 0$) as a special case.

A key distinction of our work: we do not propose a new dynamic weight decay method. Instead, we provide the first systematic evaluation of *when* existing methods provide genuine benefit, using the Phi framework to ensure fair comparison under identical conditions.

### 2.3 Evaluation Fragmentation and the Case for Null Results

A critical obstacle to progress is evaluation fragmentation. CWD (Chen et al., 2026a) is evaluated with Lion, Muon, and AdamW on language model pre-training; SWD targets the SGD-Adam generalization gap on CIFAR and ImageNet; AlphaDecay operates at LLM scales with GPT-style architectures. Each paper uses different architectures, datasets, optimizers, and hyperparameter protocols. No two papers share the same experimental conditions, making it impossible to determine whether reported improvements reflect genuine benefits of the dynamic strategy or artifacts of experimental design.

Wang \& Aitchison (2024) showed that optimal weight decay scales as an EMA timescale constant across model and dataset sizes, suggesting that a well-calibrated constant weight decay may already capture the available benefit. Their finding is consistent with our conditional equivalence result, though they provide an empirical scaling rule rather than a theoretical framework with falsifiable predictions.

Fernandez-Hernandez et al.\ (2025) proposed the Overfitting-Underfitting Indicator (OUI) as a diagnostic for weight decay quality, but it serves as a per-method tool rather than a cross-method comparison metric. D'Angelo et al.\ (2024) provided a shared experimental infrastructure (the `why-weight-decay` codebase) but did not compare dynamic scheduling strategies.

### 2.4 Positioning Against Key Recent Works

Our work is most closely related to three concurrent lines of investigation, and we carefully delineate our contributions from each:

**vs.\ Wang \& Aitchison (2024).** Their EMA timescale interpretation provides an empirical scaling rule for optimal weight decay across model sizes. Our $\rho = \lambda/\eta$ framework is complementary but distinct: we provide a regime-boundary theory with falsifiable predictions at specific $\rho$ values, and we focus on the *modulation strategy* rather than the *magnitude* of weight decay. Notably, their EMA timescale $\approx 1/\rho$ in our notation, suggesting the two perspectives may unify.

**vs.\ D'Angelo et al.\ (2024).** Their ``WD is never regularization'' finding is a static reparameterization argument grounded in batch normalization scale-invariance. Our Phi Invariance is a *dynamic* argument about AdamW's sign normalization rendering time-varying modulation irrelevant. The NoBN ablation experiment (Section 6) directly distinguishes these mechanisms.

**vs.\ Chou (2025).** Their $\lambda \propto \gamma$ schedule (scaling weight decay proportionally to the learning rate schedule) is a specific instance within our Regime I analysis: when $\rho$ remains in the low regime throughout training, the particular functional form of the schedule is immaterial, and their proposed schedule is one of many equivalent choices.
