# Alternative / Backup Ideas

## Backup 1: Feedback Control Unification with RAWD Algorithm

**Title**: A Feedback Control Theory of Weight Decay: Unifying Dynamic WD Methods via Gradient-to-Weight Ratio Control

**Core idea**: Frame all WD methods as different feedback controllers (P, PI, PD, sliding-mode, setpoint) targeting the gradient-to-weight ratio R_t = ||g_t||/||w_t|| as the common plant output. Design an explicit Ratio-Adaptive Weight Decay (RAWD) optimizer: lambda_t = k_P * max(R_hat_t - R*, 0) + lambda_min, with stability-derived gain bound k_P < 2/(L_R * eta).

**Why backup**: The controller framing is conceptually powerful but less theoretically deep than the PMP/Lyapunov approach. It provides excellent intuition and prescriptive value ("which controller architecture matches your training dynamics?") but the stability proofs require characterizing the closed-loop plant, which is essentially as hard as the original convergence problem.

**When to promote**: If the PMP derivation proves too difficult to make rigorous (costate identification p ~ -g_t is crude), the controller framing can serve as the primary conceptual contribution without requiring formal control-theoretic proofs. It also works better as a framing if the empirical results show RAWD providing clear improvements.

**Differentiating factor from front-runner**: Focuses on the algorithm (RAWD) rather than the theoretical framework. Better for an empirical ML venue (ICML poster) vs. the front-runner which targets a theory-heavy venue (NeurIPS oral / JMLR).

---

## Backup 2: When Does Alignment-Aware WD Fail? Architecture-Conditioned Null Result

**Title**: When Does Alignment-Aware Weight Decay Fail? A Systematic Null-Hypothesis Evaluation with AIS Threshold Predictions

**Core idea**: Reframe the paper as primarily empirical/diagnostic. The main contribution is the systematic null-hypothesis test: no dynamic WD method beats constant WD on CIFAR-10/100/ImageNet under compute-controlled conditions with proper BEM normalization. The AIS threshold (< 0.5) provides the theoretical explanation. Include BN vs non-BN architecture comparison to test whether scale invariance is the mechanism.

**Why backup**: This framing de-emphasizes the theoretical framework (Lyapunov, PMP) and focuses on the empirical contribution. It is a safer bet: the null result is already partially established from iter_003 data, and extending to VGG-16-BN and ImageNet is straightforward.

**When to promote**: If the PMP/Lyapunov proofs have serious gaps that cannot be patched, or if the theoretical framework makes predictions that are empirically falsified (e.g., continuous alignment-aware WD also fails despite theory predicting improvement), pivot to this empirical framing.

**Differentiating factor from front-runner**: Lower theoretical ambition but higher empirical rigor. Pre-registered falsification criteria, TOST equivalence tests, adequate statistical power (n=9 seeds). Targets an empirical ML venue or a "negative results" workshop.

---

## Backup 3: Pontryagin-Derived WD with Hill Function Robustification

**Title**: Optimal Control Theory Derives Alignment-Aware Weight Decay: The Pontryagin Costate as the Missing Theoretical Bridge

**Core idea**: Focus exclusively on the PMP derivation as the theoretical anchor. Show that treating lambda(t) as the optimal control variable in the gradient-flow control problem yields lambda_t* proportional to gamma_t * (1 - delta_t) from the Hamiltonian stationarity condition. Extend with Hill-function form lambda_t proportional to (1-delta_t)^n / (delta*^n + (1-delta_t)^n) for noise robustness. Demonstrate that n>1 outperforms n=1 in high-noise (small batch) settings.

**Why backup**: This is the narrowest and deepest version of the front-runner. It focuses on one cross-disciplinary insight (PMP from control theory) rather than attempting the full unification. Less ambitious but more likely to produce a clean, self-contained contribution.

**When to promote**: If the full unification (Phi-Modulator Taxonomy + Lyapunov Variance + PMP + Budget Equivalence + Controller Taxonomy) is too sprawling for a single paper, extract the PMP derivation + Hill function as a focused short paper or workshop contribution.

**Differentiating factor**: Narrower scope, deeper on one theoretical thread. Works as a TMLR submission or a theory workshop paper.
