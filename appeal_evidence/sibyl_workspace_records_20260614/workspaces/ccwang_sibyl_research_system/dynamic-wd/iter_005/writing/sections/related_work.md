# 2. Related Work

## 2.1 Weight Decay as Dynamics Modifier

Loshchilov & Hutter (ICLR 2019) established that L2 regularization and decoupled WD are not equivalent in adaptive optimizers, introducing AdamW as the standard. D'Angelo et al. (NeurIPS 2024) demonstrated that WD never functions as explicit regularization in modern deep learning; instead, it modifies training dynamics through loss stabilization (SGD) and bias-variance tradeoff (LLMs). Kosson et al. (2023) showed WD induces a rotational equilibrium that balances effective learning rates across layers. Han et al. (2026) found that WD during pretraining improves model plasticity by encouraging linearly separable representations.

WD also drives structural effects: Galanti et al. (2022) proved SGD with WD induces low-rank weight matrices; Kobayashi et al. (2024) showed L2 regularization on multiplicative parameters (attention layers) is equivalent to nuclear norm regularization; Kuzborskij & Abbasi-Yadkori (2025) demonstrated that L2 regularization induces parameter-gradient alignment at stationary points. Defazio (2025) unified these observations through the gradient-to-weight ratio $\rho = \|g\| / \|\theta\|$: WD drives all normalized layers to the same steady-state $\rho^*$, providing a clean explanation for AdamW's advantage over Adam+L2.

## 2.2 Dynamic WD Methods

Four families of dynamic WD methods have emerged:

**Alignment-aware WD.** CWD (Chen et al., ICLR 2026) applies a binary sign-alignment mask: decay occurs only when $\text{sign}(\theta) = \text{sign}(u_t)$, where $u_t$ is the optimizer update. CWD interprets this as bilevel Pareto-optimal and reports improvements on LLM and vision tasks. AdamO (Chen, Yuan, & Zhang, 2026) identifies a "radial tug-of-war" between WD and gradient updates, decoupling radial (norm) and tangential (direction) dynamics.

**Temporal scheduling.** SWD (Xie et al., NeurIPS 2023) scales WD inversely with gradient norm, reducing decay when gradients are large. ADANA (Ferbach et al., 2026) proposes logarithmic-time schedules for WD alongside $\beta_1$ and $\beta_2$, reporting 40% compute efficiency gains.

**Norm-matched WD.** AdamWN (Loshchilov, 2023) generalizes decoupled WD to target an arbitrary weight norm rather than zero. AlphaDecay (He et al., NeurIPS 2025) assigns module-wise decay rates guided by spectral density (heavy-tail self-regularization theory), scaling to 1B-parameter LLMs.

**Structural WD.** Defazio (2025) proposes a layer-balancing framework where WD corrects for the gradient-to-weight ratio imbalance caused by learning rate schedules. Truong & Truong (2026) analyze norm-hierarchy transitions showing WD traverses representations from shortcut to structured solutions.

## 2.3 Evaluation Fragmentation

Each method is evaluated on different benchmarks, metrics, and hyperparameter protocols. CWD reports final accuracy improvements; AlphaDecay uses perplexity and spectral density; SWD focuses on gradient norms and generalization gaps. No prior work systematically varies $\rho$ to test regime-dependence of dynamic WD. This fragmentation motivates our controlled experimental design with fixed hyperparameters across all methods and explicit ratio-regime sweeps.

## 2.4 Optimal Control in Optimization

Li & Tai (2017) applied the Pontryagin Maximum Principle (PMP) to derive optimal learning rate schedules. Defazio's corrective WD term (2025) is a feedforward schedule where $\lambda$ depends on the planned $\gamma_t$; it does not incorporate measured state feedback. Our PMP-WD extends optimal control theory from learning rates to WD schedules and, critically, uses state feedback ($\hat{\rho}_t$ measurement) rather than feedforward scheduling, enabling real-time correction of deviations from the optimal trajectory.

<!-- FIGURES
- None
-->
