# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Chen, Liu, Liang & Liu, 2023. "Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts." arXiv:2310.05898** -- Proves Lion with decoupled weight decay solves min f(x) + K*(x), where K*(x) is the convex conjugate of the sign reshaper. Develops a novel Lyapunov function V(x,m) = f(x) + K*(x) + (1/(2*epsilon))‖m - nabla f(x)‖^2 that certifies convergence of the Lion-K family. The key insight: decoupled weight decay is equivalent to adding a convex conjugate penalty, so the "type" of weight decay determines the implicit constraint geometry. This directly motivates our operator-theoretic unification: different WD methods correspond to different choices of the proximal operator / convex regularizer in a composite optimization framework.

2. **Truong et al., 2026. "Why Grokking Takes So Long: A First-Principles Theory of Representational Phase Transitions." arXiv:2603.13331** -- Establishes the Norm-Separation Delay Law: T_grok - T_mem = Theta((1/gamma_eff) * log(‖theta_mem‖^2 / ‖theta_post‖^2)), where gamma_eff = eta*lambda for SGD and gamma_eff >= eta*lambda for AdamW. Uses a discrete Lyapunov contraction argument for the upper bound and dynamical lower bounds. The effective contraction rate gamma_eff = eta*lambda is precisely the "intrinsic learning rate" that governs norm dynamics under weight decay -- this is the same quantity that controls rotational equilibrium (Kosson et al. 2023) and the EMA timescale (Wang & Aitchison 2024).

3. **Wan et al., 2020. "Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD." arXiv:2006.08419** -- Proves that under batch normalization + weight decay + SGD, the training dynamics live on a sphere (weight norm converges at linear rate to an equilibrium). Introduces "angular update" as the true measure of learning progress. The equilibrium condition provides the mathematical foundation for our norm-stability characterization.

4. **Li, Lyu & Arora, 2020. "Reconciling Modern Deep Learning with Traditional Optimization: The Intrinsic Learning Rate." arXiv:2010.02916** -- Defines the intrinsic learning rate eta_intrinsic = eta * lambda, and models SGD dynamics on normalized networks via an SDE where the effective speed of learning is governed entirely by this product. The intrinsic LR is the control variable for all dynamical effects of WD. Proposes the "Fast Equilibrium Conjecture": equilibrium time scales as 1/(eta*lambda), not exponentially.

5. **Kosson, Messmer & Jaggi, 2023. "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks." arXiv:2305.17212** -- Shows WD induces rotational equilibrium: balanced average angular rotation across layers/neurons for Adam, Lion, and SGD with momentum. Angular update converges to steady state proportional to 1/(eta*lambda). This provides the dynamical foundation for the Coupling Stability Index.

6. **Xie & Li, 2024. "Implicit Bias of AdamW: l_inf Norm Constrained Optimization." arXiv:2404.04454** -- Proves that AdamW's decoupled weight decay implicitly solves l_inf constrained optimization (min f(x) s.t. ‖x‖_inf <= 1/lambda). Connected to Frank-Wolfe algorithm. Complements the Lion result (which solves l_1-penalized problems) -- together, these show that the "shape" of the implicit constraint set depends on the optimizer's update geometry.

7. **Galanti, Siegel, Gupte & Poggio, 2022. "SGD and Weight Decay Secretly Minimize the Rank of Your Neural Network." arXiv:2206.05794** -- Proves that SGD + WD induces low-rank bias in weight matrices. The bias is more pronounced with smaller batch, higher LR, or stronger WD. Critical structural effect of WD that is not captured by simple norm-control theories.

8. **D'Angelo, Andriushchenko, Varre & Flammarion, 2023/2024. "Why Do We Need Weight Decay in Modern Deep Learning?" arXiv:2310.04415 (NeurIPS 2024)** -- Unifying empirical perspective: WD for SGD on vision acts through "loss stabilization" (preventing training loss divergence), while WD for LLMs acts through bias-variance tradeoff in near-one-epoch training. Proves WD prevents bfloat16 divergence. Establishes that WD is never useful as explicit regularization in modern deep learning.

9. **Golatkar, Achille & Soatto, 2019. "Time Matters in Regularizing Deep Networks." arXiv:1905.13277** -- Proves the "critical period" phenomenon: WD applied only in the initial transient determines the final solution. WD applied only after the transient has no effect. This has deep implications for WD scheduling: the temporal profile of WD matters more than the total WD budget.

10. **Jacot, Sukenk, Wang & Mondelli, 2024. "Wide Neural Networks Trained with Weight Decay Provably Exhibit Neural Collapse." arXiv:2410.04887** -- First rigorous proof of neural collapse emergence from end-to-end GD training with WD. Requires low training error + balancedness (from WD) + bounded conditioning. WD is essential for the symmetry-inducing dynamics that lead to neural collapse geometry.

11. **Wang, Meng, Chen & Liu, 2020. "The Implicit Bias for Adaptive Optimization Algorithms on Homogeneous Neural Networks." arXiv:2012.06244** -- Proves that Adam/RMSProp (exponential moving average in conditioner) can maximize margin on homogeneous networks, while AdaGrad cannot. Provides a unified framework for analyzing the convergent direction of adaptive optimizers via constructed adaptive gradient flow and surrogate margin. The implicit bias of the optimizer interacts with WD to determine the geometry of the converged solution.

12. **Hwang, 2024. "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information." arXiv:2405.12807** -- Establishes that Adam approximates natural gradient descent on the Fisher-Rao manifold. Derives corrected WD term based on information geometry. The connection Adam <-> Fisher <-> WD is the mathematical bridge between the operator-theoretic and information-geometric views.

### Theoretical Landscape Summary

The theoretical landscape of weight decay can be organized along three mathematical axes:

**Axis 1: Implicit constraint / regularization characterization.** A growing body of work (Chen et al. 2023 on Lion, Xie & Li 2024 on AdamW, Ding et al. 2023 on Adam-family) reveals that each optimizer + WD combination implicitly solves a different constrained / regularized optimization problem. Lion+WD solves l_1-penalized problems (or equivalently l_inf-constrained), AdamW solves l_inf-constrained problems (or equivalently l_1-penalized in a dual sense), and SGD+WD solves l_2-penalized problems. The constraint geometry is determined by the convex conjugate of the optimizer's update mapping. This is known but not yet unified into a single framework that encompasses dynamic (time-varying) WD, alignment-conditioned WD, or target-norm WD.

**Axis 2: Dynamical systems / Lyapunov analysis.** The training trajectory under WD is governed by the intrinsic learning rate eta*lambda (Li et al. 2020), which controls (a) the contraction rate toward rotational equilibrium (Kosson et al. 2023), (b) the grokking delay (Truong et al. 2026), (c) the angular update speed (Wan et al. 2020), and (d) the effective regularization strength (D'Angelo et al. 2024). All of these are manifestations of the same underlying dynamical system. Lyapunov functions have been constructed for specific cases (Lion-K by Chen et al., grokking by Truong et al.) but no general Lyapunov framework exists that accommodates dynamic WD coefficients.

**Axis 3: Structural effects on learned representations.** WD induces low-rank weight matrices (Galanti et al. 2022), neural collapse geometry (Jacot et al. 2024), rotational equilibrium (Kosson et al. 2023), and spherical dynamics (Wan et al. 2020). These structural effects are consequences of the norm-controlling dynamics but are not yet connected to the implicit constraint characterization of Axis 1 or the Lyapunov analysis of Axis 2.

**The critical gap**: No existing framework unifies all three axes. The operator-theoretic view (Axis 1) characterizes what WD does at convergence but says nothing about the trajectory. The Lyapunov view (Axis 2) characterizes the trajectory but assumes a fixed WD coefficient. The structural view (Axis 3) describes emergent properties but does not predict them from the WD strategy. A unified theory must connect (a) the choice of dynamic WD strategy to (b) the convergence trajectory via Lyapunov analysis to (c) the structural properties of the converged solution.

Furthermore, none of the existing theories address the **four sub-approaches** (scheduling, alignment-aware, decoupled, norm-matched) within a single mathematical framework. Each has its own theoretical justification (gradient-norm bounds for SWD, bilevel Pareto for CWD, convex conjugate for decoupled WD, target-norm control for AdamWN) but these justifications are mathematically disconnected.


## Phase 2: Initial Candidates

### Candidate A: Operator-Theoretic Unification via Generalized Proximal Splitting

**Formal claim (Theorem sketch):** Every dynamic weight decay strategy can be characterized as a time-varying proximal operator applied to the parameter updates. Specifically, the general WD update rule

  w_{t+1} = w_t - eta * u_t - eta * Phi_t(w_t)

where u_t is the optimizer's gradient-based update and Phi_t is the WD operator, admits a proximal interpretation:

  Phi_t(w) = (1/eta) * (w - prox_{eta * R_t}(w))

where R_t: R^d -> R is a time-varying, potentially parameter-dependent regularizer. The four WD sub-approaches correspond to specific structural choices for R_t:

- **Decoupled WD** (AdamW): R_t(w) = (lambda/2) * ‖w‖_2^2, constant in t. The proximal operator is prox_{eta*lambda/2 * ‖.‖^2}(w) = w/(1 + eta*lambda), recovering the standard multiplicative decay (1 - eta*lambda)*w.
- **Scheduling WD** (SWD, ADANA): R_t(w) = (lambda(t)/2) * ‖w‖_2^2, where lambda(t) varies according to a schedule. The proximal operator changes in strength over time but retains the isotropic ‖.‖^2 structure.
- **Alignment-aware WD** (CWD): R_t(w) = (lambda/2) * ‖w odot m_t‖_2^2, where m_t = 1_{sign(w) = sign(u_t)} is the alignment mask. The proximal operator applies non-uniformly across coordinates based on the instantaneous geometric relationship between w and the update.
- **Norm-matched WD** (AdamWN): R_t(w) = (lambda/2) * ‖w - tau * w/‖w‖‖_2^2, which penalizes deviation from a target norm tau rather than deviation from zero. The proximal operator contracts toward a sphere rather than toward the origin.

**Key proposition:** Given the proximal characterization, one can analyze convergence of any dynamic WD strategy by constructing a time-varying Lyapunov function:

  V_t(w) = f(w) + R_t(w) + C_t

where C_t is a correction term that accounts for the temporal variation in R_t. For the Lyapunov function to decrease, the rate of change of R_t must be bounded:

  dR_t/dt <= alpha * (V_t - V*)

for some alpha > 0. This yields a **WD stability condition**: the WD schedule can change at most as fast as the optimization is converging. Schedules that violate this condition (e.g., abrupt WD changes) can cause the Lyapunov function to increase, leading to training instability.

**Proof sketch:**
1. (Lemma 1) Establish the proximal characterization for each WD variant using the Moreau decomposition.
2. (Lemma 2) Construct the time-varying Lyapunov function and derive descent conditions under the stochastic gradient noise model of D'Angelo et al.
3. (Theorem 1) Prove convergence rate O(1/sqrt(T)) for the general dynamic WD framework under the WD stability condition.
4. (Corollary) Recover known convergence results for AdamW (Ding et al. 2023), Lion (Chen et al. 2023), and SGD+WD as special cases.

**Empirical prediction:** The WD stability condition predicts that abrupt WD schedule changes (e.g., step-wise WD decay) should cause transient spikes in training loss, while smooth WD schedules (cosine, logarithmic) should not. This is testable by monitoring training loss variance across WD schedule types.

**Connection to existing theory:** Extends Chen et al. (2023)'s Lion-K Lyapunov analysis from constant WD to time-varying WD. Generalizes Xie & Li (2024)'s implicit constraint characterization from AdamW to the full family. Subsumes PathProx (Yang et al. 2022) which treats WD as a proximal operator but only for ReLU networks with fixed WD.

**Novelty estimate:** 8/10 -- The proximal characterization of individual WD variants is partially known (PathProx for fixed WD, Lion-K for sign-based WD), but the unified time-varying framework with Lyapunov analysis and the WD stability condition is new. The key novel element is showing that all four sub-approaches live in the same operator-theoretic space and deriving the stability constraint on temporal variation.


### Candidate B: The Contraction-Alignment Duality and Information-Theoretic Optimality of WD Strategies

**Formal claim (Theorem sketch):** There exists a fundamental duality between the contraction rate of WD (how fast it shrinks weight norms) and the alignment informativeness (how much the gradient-weight alignment signal improves WD decisions). Specifically, define:

  gamma(w,t) = rate of norm contraction under WD strategy at state (w,t)
  I_align(w,t) = mutual information between the alignment signal sign(w) vs sign(nabla f(w)) and the optimal WD direction

Then for any WD strategy, there exists a tradeoff:

  gamma(w,t) * I_align(w,t) <= C * ‖nabla f(w)‖ * ‖w‖

where C is a universal constant depending only on the loss landscape curvature. This is an uncertainty-principle-like bound: you cannot simultaneously achieve maximal contraction rate AND maximal alignment informativeness.

**Consequence 1:** Standard decoupled WD (constant lambda, no alignment) maximizes gamma but sets I_align = 0 (alignment information is ignored). This is optimal when the gradient-weight alignment is uninformative (as found in iterations 0-2 for the nonconvex case).

**Consequence 2:** CWD (alignment-conditioned WD) reduces gamma (by zeroing WD on misaligned coordinates) but maximizes I_align. This is optimal when alignment is informative (e.g., near convergence, in well-conditioned regions).

**Consequence 3:** The optimal dynamic WD strategy adaptively trades off gamma and I_align over the training trajectory, starting alignment-agnostic (high gamma, low I_align) and transitioning to alignment-aware (lower gamma, higher I_align) as the loss landscape geometry shifts.

**Proof sketch:**
1. (Lemma 1) Bound the mutual information I_align using the Fisher information of the alignment signal with respect to the optimal WD direction.
2. (Lemma 2) Relate the contraction rate gamma to the spectral properties of the Hessian at the current point.
3. (Theorem) Derive the tradeoff bound using the data processing inequality and the Cramer-Rao lower bound.
4. (Corollary) Show that the optimal switching time between alignment-agnostic and alignment-aware WD scales as O(1/(eta*lambda) * log(condition_number)).

**Empirical prediction:** CWD should outperform standard WD more strongly in the later phases of training (when alignment becomes informative) than in early phases. The crossover point should correlate with the point where gradient-weight alignment transitions from uniform (uninformative) to structured. This is directly testable by comparing CWD vs AdamW with alignment tracking.

**Connection to existing theory:** Formalizes the empirical finding from iterations 0-2 that alignment is "uninformative at nonconvex scale" into a precise information-theoretic bound. Extends CWD's bilevel Pareto interpretation with a quantitative optimality criterion.

**Novelty estimate:** 7/10 -- The contraction-alignment tradeoff is a genuinely new theoretical construct. The information-theoretic formalization of "alignment informativeness" is novel. However, the proof relies on strong assumptions (curvature bounds, gradient noise model) that may not hold in practice, making the bound potentially vacuous in realistic settings.


### Candidate C: Phase-Transition Theory of Weight Decay Dynamics -- Connecting Norm Control to Representation Quality

**Formal claim (Theorem sketch):** The training trajectory under weight decay undergoes a sequence of phase transitions, each characterized by a change in the effective rank of the weight matrices. Define the effective rank rho_eff(W) = exp(H(sigma(W)/‖sigma(W)‖_1)) where sigma(W) is the singular value vector and H is the Shannon entropy. Then under WD with coefficient lambda, the dynamics satisfy:

  d(rho_eff)/dt = -alpha * lambda * (rho_eff - rho_target(t)) + noise

where rho_target(t) is determined by the training loss landscape geometry at time t. The system exhibits three phases:

1. **Exploration phase** (early training): High effective rank, WD is dominated by gradient signal. The rank remains high because gradient updates inject new directions faster than WD can suppress them.

2. **Consolidation phase** (mid training): Effective rank begins to decrease as WD suppresses low-signal singular values. The transition time T_consolidation = Theta(1/(eta*lambda) * log(d/rho_target)) follows the same logarithmic norm-separation law as grokking.

3. **Compression phase** (late training): Effective rank saturates at rho_target. Further WD primarily rotates the weight vectors within the low-rank subspace (rotational equilibrium).

**Key proposition (connecting to all four sub-approaches):**
- **WD scheduling** controls the transition timings: decreasing lambda delays consolidation, allowing more exploration.
- **Alignment-aware WD** makes the consolidation phase more efficient by preserving high-signal directions.
- **Decoupled WD** (vs L2) ensures the rank reduction is driven by genuine redundancy, not preconditioner distortion.
- **Norm-matched WD** (target norm tau) directly controls the steady-state effective rank in the compression phase.

**Proof sketch:**
1. (Lemma 1) Derive the effective rank dynamics using the matrix SDE for W_t under SGD + WD, following the approach of Galanti et al. (2022) but extending to time-varying WD.
2. (Lemma 2) Characterize the phase transitions using the eigenvalue gap of the Hessian, showing that the consolidation transition occurs when the WD contraction rate exceeds the gradient injection rate for a critical fraction of singular values.
3. (Theorem) Prove the three-phase structure and derive the transition times.
4. (Prediction) The effective rank trajectory under optimal WD should follow a "Goldilocks" profile: high early, controlled descent during consolidation, stable low in compression. Deviations from this profile (too aggressive WD = premature compression; too weak WD = no compression) predict poor generalization.

**Empirical prediction:** (1) The effective rank trajectory should exhibit clear phase transitions detectable by a change-point algorithm. (2) The consolidation onset time should scale as 1/(eta*lambda). (3) Methods that optimize the rank trajectory (alignment-aware, well-scheduled) should achieve better generalization than methods that compress too aggressively or too slowly.

**Connection to existing theory:** Extends Galanti et al. (2022)'s static low-rank bias result to a dynamic, three-phase theory. Connects to Truong et al. (2026)'s norm-separation delay law via the consolidation transition time. Connects to Kosson et al. (2023)'s rotational equilibrium as the steady state of the compression phase.

**Novelty estimate:** 8/10 -- The three-phase structure is partially observed empirically but has not been formally proved. The connection between effective rank dynamics and all four WD sub-approaches is entirely new. The main risk is that the phase transition is smooth (not sharp) in practice, making the three-phase characterization an idealization.


## Phase 3: Self-Critique

### Against Candidate A (Operator-Theoretic Unification)

**Proof soundness attack:** The proximal characterization of CWD is approximate, not exact. CWD applies a binary mask based on instantaneous alignment, which is not the gradient of any smooth regularizer R_t. Specifically, the CWD operator Phi_CWD(w) = lambda * w * 1_{sign(w)=sign(u)} is not the proximal operator of any convex function because the mask is discontinuous and depends on the update direction u_t, which itself depends on the optimization history. The claimed proximal interpretation requires either (a) relaxing to a smooth approximation of the alignment mask, which changes the algorithm, or (b) working with non-convex or non-smooth regularizers, which invalidates the standard proximal convergence theory.

*Search verification*: Confirmed via arXiv search -- CWD's bilevel Pareto interpretation (Chen et al. 2025) does NOT use proximal operators but rather a set-valued variational inclusion framework. The proximal interpretation would be a new theoretical contribution but requires careful handling of the non-smoothness.

**Tightness attack:** The convergence rate O(1/sqrt(T)) for the general framework is no better than standard SGD convergence rates. The WD stability condition, while intuitively reasonable, may be trivially satisfied by all practically used schedules (cosine, linear decay, exponential), making it a vacuous constraint. The framework could be mathematically correct but practically uninformative if it does not differentiate between good and bad WD strategies.

**Relevance attack:** Practitioners do not think in terms of proximal operators. The framework needs to yield a concrete recommendation ("use this WD schedule, not that one") to be practically useful. If the main output is "all existing methods are special cases of our framework," this is a taxonomic contribution rather than a prescriptive one.

**Novelty attack:** PathProx (Yang et al. 2022) already characterizes WD as a proximal operator for ReLU networks. The generalization to time-varying WD is incremental. The Lyapunov analysis for Lion-K (Chen et al. 2023) already provides a sophisticated framework for a specific optimizer + WD combination. The main novelty is the unification across WD variants, but this may be seen as "wrapping existing results in a common notation" rather than proving genuinely new mathematical results.

**Verdict: MODERATE** -- The framework is sound in principle but the CWD proximal characterization is technically problematic and the convergence results may not be tighter than existing work. The unification is valuable taxonomically but needs at least one genuinely new and tight result to be a strong theory paper. The WD stability condition is the most promising novel contribution but needs to be shown to be tight (violated by at least one practical scenario).


### Against Candidate B (Contraction-Alignment Duality)

**Proof soundness attack:** The mutual information I_align is defined between the alignment signal and the "optimal WD direction" -- but the optimal WD direction is not known a priori. If we define it as the direction that minimizes the regularized loss, this creates a circular definition. The data processing inequality argument requires the alignment signal to be a deterministic function of the weights and gradients, which it is, but the Fisher information bound requires regularity conditions on the gradient distribution that may not hold in the nonconvex setting (where gradients can be highly non-Gaussian and heavy-tailed).

**Tightness attack:** The bound gamma * I_align <= C * ‖nabla f‖ * ‖w‖ may be extremely loose. The constant C depends on curvature, which can vary by orders of magnitude across the training trajectory and across layers. In the overparameterized regime where ‖nabla f‖ -> 0 at interpolation, the RHS goes to zero, which trivially implies that both gamma and I_align must also go to zero -- this is vacuous because it just says "after convergence, nothing changes."

**Relevance attack:** The information-theoretic framing is elegant but the practical consequence ("start alignment-agnostic, transition to alignment-aware") is already the default behavior in practice: CWD's mask is inactive when alignment is random (early training) and becomes active as alignment develops structure. The theory may be explaining a phenomenon that practitioners already intuit without providing actionable new guidance.

**Novelty attack:** The idea that alignment informativeness changes over training is already discussed in CWD's paper and in the literature on alignment as implicit regularization (Kuzborskij & Abbasi-Yadkori 2025). The information-theoretic formalization is new but the qualitative insight is not. The "uncertainty principle" analogy may be stretched -- there is no fundamental physical constraint preventing simultaneous high contraction and high alignment usage; the tradeoff may be an artifact of the specific bound technique rather than a deep mathematical truth.

**Verdict: WEAK-TO-MODERATE** -- The central theoretical construct (contraction-alignment duality) is appealing but the formal realization has serious tightness issues. The bound is likely vacuous in realistic settings. The qualitative insight is not sufficiently novel. Would need a much tighter bound or a concrete impossibility result to be a strong theory contribution.


### Against Candidate C (Phase-Transition Theory)

**Proof soundness attack:** The effective rank dynamics d(rho_eff)/dt = -alpha*lambda*(rho_eff - rho_target) + noise is an oversimplification. The effective rank is a nonlinear, non-smooth function of the singular values, and its dynamics under SGD+WD cannot be reduced to a simple ODE without substantial approximation. The singular value dynamics under matrix SGD are governed by a system of coupled SDEs (Benaych-Georges & Nadakuditi 2011) that do not simplify to a single ODE for effective rank. The three-phase structure may be an artifact of the simplification rather than a genuine property of the dynamics.

*Search verification*: Searched for "effective rank dynamics neural network training" -- found Galanti et al.'s static analysis but no dynamic analysis of effective rank under WD. The gap between the static result (WD induces low rank) and the dynamic claim (three-phase transition with predictable timing) is significant and would require substantial new mathematical machinery.

**Tightness attack:** The transition time T_consolidation = Theta(1/(eta*lambda) * log(d/rho_target)) borrows the form from Truong et al.'s grokking delay law. But the grokking delay law is proved for specific algorithmic tasks (modular arithmetic) with specific representation structures (Fourier circuits). Extending it to the effective rank setting for general deep networks requires showing that the norm-separation structure exists in the singular value space, which is a non-trivial generalization. The "Theta" notation hides constants that could make the bound useless in practice.

**Relevance attack:** The "Goldilocks" profile of effective rank is an intuitive description but may not be quantitatively useful. Practitioners already know that too much WD hurts (premature compression) and too little WD hurts (overfitting). The three-phase theory formalizes this common knowledge but does not obviously lead to a better WD algorithm. To be useful, the theory must answer: "given a model and dataset, what is the optimal WD schedule?" -- and the answer requires knowing rho_target(t), which depends on the unknown loss landscape geometry.

**Novelty attack:** The connection between WD and low rank is known (Galanti et al. 2022, Kobayashi et al. 2024). The phase transition structure in training is well-studied in the grokking literature. The novel claim is the three-phase decomposition specifically for effective rank under dynamic WD, which is a reasonable incremental contribution but not a breakthrough insight. The predictions (phase transition timing, rank trajectory shape) need to be sufficiently quantitative and falsifiable.

**Verdict: MODERATE** -- The three-phase theory is the most physically grounded and empirically testable of the three candidates. The main weakness is the gap between the simplified ODE model and the actual matrix dynamics. The predictions are concrete and falsifiable, which is a strength. The connection to all four WD sub-approaches provides a unifying narrative. Would benefit from a pilot experiment measuring effective rank trajectories to validate the phase structure before committing to the full theoretical derivation.


## Phase 4: Refinement

### Dropped: Candidate B (Contraction-Alignment Duality)

Candidate B has the most serious theoretical issues. The bound is likely vacuous, the practical insight is already known, and the proof requires assumptions (regularity of gradient distribution, well-defined optimal WD direction) that are hard to verify. The information-theoretic framing is elegant but the substance is thin.

### Strengthened: Candidate A (Operator-Theoretic Unification)

**Refinements based on self-critique:**

1. **Fix the CWD proximal problem.** Instead of claiming CWD has an exact proximal interpretation, characterize it as an *approximate* proximal operator with a bounded approximation error. Specifically, define the "soft CWD" regularizer R_CWD(w; u) = (lambda/2) * sum_i w_i^2 * sigma(beta * w_i * u_i) where sigma is the sigmoid function and beta -> infinity recovers the hard CWD mask. The soft version IS a smooth, convex-in-w regularizer for fixed u, and its proximal operator is well-defined. The approximation error between soft CWD and hard CWD is O(1/beta), quantifiable.

2. **Strengthen the WD stability condition to be non-trivial.** Show that the stability condition is violated by specific practical scenarios: (a) WD warmup from 0 to lambda in fewer than 1/(eta*lambda) steps, and (b) step-wise WD changes at LR schedule boundaries. Provide a concrete example where violation leads to measurable training instability. This makes the condition non-vacuous and actionable.

3. **Derive a concrete optimality result.** Within the proximal framework, prove that for quadratic losses, the optimal time-varying WD schedule lambda(t) is the one that minimizes the integrated Lyapunov function, and show this optimal schedule has a specific closed-form (involving the Hessian eigenvalues). While quadratic, this yields qualitative insights for the nonconvex case: lambda(t) should decrease when the effective Hessian curvature increases (agreeing with SWD's gradient-norm heuristic).

### Strengthened: Candidate C (Phase-Transition Theory)

**Refinements based on self-critique:**

1. **Replace the effective rank ODE with a more rigorous formulation.** Instead of the scalar ODE, work with the singular value distribution's evolution under SGD + WD. Define the "WD spectral flow": a measure on the singular value spectrum parameterized by training time. Show that WD creates a drift toward zero that competes with gradient updates that push singular values up. The phase transitions correspond to bifurcation points in this spectral flow.

2. **Weaken the quantitative timing claims.** Instead of claiming Theta(1/(eta*lambda) * log(d/rho_target)), state the result as an upper bound with explicit constants. Show that the bound is tight for linear networks (where the singular value dynamics can be solved exactly) and conjecture the extension to nonlinear networks.

3. **Add the critical experimental prediction that distinguishes this theory from alternatives.** The three-phase theory predicts that the *timing* of the consolidation phase (when effective rank begins to drop) should be predictable from (eta, lambda, initial rank). Measure this prediction on CIFAR-10 with ResNet-20 across a grid of (eta, lambda) values. If the prediction holds with R^2 > 0.8, the theory is validated; if not, it must be revised.

### Selected Front-Runner: Candidate A (Operator-Theoretic Unification) + elements of Candidate C

The strongest theoretical contribution combines the operator-theoretic framework of Candidate A (which provides the mathematical unification) with the dynamical insights of Candidate C (which provides the physical intuition and falsifiable predictions). The combined idea:

**A unified proximal-dynamical framework for weight decay that (a) characterizes all four WD sub-approaches as instances of a generalized time-varying proximal operator, (b) derives convergence via a time-varying Lyapunov function with a novel WD stability condition, and (c) connects the proximal structure to the spectral dynamics of weight matrices across three training phases.**


## Phase 5: Final Proposal

### Title
**Weight Decay as Time-Varying Proximal Regularization: A Unified Operator-Theoretic Framework**

### Formal Claims

**Claim 1 (Unification Theorem).** Let Opt be any first-order optimizer (SGD, Adam, Lion, Muon) with a decoupled weight decay step. The combined update can be written as:

  w_{t+1} = prox_{eta_t * R_t}(w_t - eta_t * Opt_update(w_t, g_t, state_t))

where R_t is a time-varying regularizer that depends on the WD strategy:

| WD Strategy | R_t(w) | Proximal Operator |
|---|---|---|
| Standard decoupled (AdamW) | (lambda/2)‖w‖^2 | w/(1+eta*lambda) |
| Scheduled (SWD, ADANA) | (lambda(t)/2)‖w‖^2 | w/(1+eta*lambda(t)) |
| Alignment-aware (soft CWD) | (lambda/2) sum w_i^2 sigma(beta*w_i*u_i) | Coordinate-wise, alignment-dependent |
| Norm-matched (AdamWN) | (lambda/2)‖w - tau*w/‖w‖‖^2 | Contraction toward sphere of radius tau |
| Lp-norm decoupled | (lambda/p)‖w‖_p^p | Element-wise soft-thresholding |

This taxonomy is complete for first-order methods with additive WD.

**Claim 2 (WD Stability Theorem).** Define the time-varying Lyapunov function:

  V_t = f(w_t) + R_t(w_t)

Under standard smoothness assumptions on f and bounded gradient noise, V_t decreases in expectation if and only if:

  |R_{t+1}(w_t) - R_t(w_t)| <= (1 - rho) * eta_t * ‖nabla f(w_t)‖^2 + sigma^2

for some 0 < rho < 1. This is the **WD Stability Condition**. It states that the change in the regularizer between consecutive steps must be bounded by the optimization progress. Equivalently, the WD schedule can change at most as fast as the loss is decreasing.

**Corollary (Stability violation examples):**
- WD warmup from 0 to lambda_max in K < 1/(eta*lambda_max) steps violates the condition when ‖nabla f‖ is small (flat initialization region).
- Step-wise WD increase at epoch boundaries violates the condition if the step size exceeds sigma^2/‖nabla f‖^2.

**Claim 3 (Optimal WD Schedule for Quadratic Losses).** For f(w) = (1/2) w^T H w - b^T w with H having eigenvalues {mu_i}, the WD schedule that minimizes E[V_T] over T steps is:

  lambda_opt(t) = sum_i (mu_i * v_i(t)^2) / (sum_i v_i(t)^2)

where v_i(t) is the component of w_t in the i-th eigendirection of H. This is the curvature-weighted average along the current iterate, and it is monotonically increasing during training (as the iterate moves from low-curvature to high-curvature regions). This theoretically justifies WD warmup strategies.

### Proof Sketch

1. **Claim 1 proof:** For each WD variant, express the WD step as w' = w - eta*Phi(w) and show Phi(w) = (1/eta)(w - prox_{eta*R}(w)) using the Moreau identity. For the standard L2 case this is textbook. For CWD, use the smooth approximation and bound the error. For AdamWN, use the proximal operator of the distance-to-sphere penalty (derived from the Euclidean projection onto the annulus). For Lp norms, use the known proximal operators of Lp penalties.

2. **Claim 2 proof:** Expand V_{t+1} using the descent lemma for f and the proximal inequality for R_t. The key step is bounding the cross-term E[R_{t+1}(w_{t+1}) - R_t(w_{t+1})] which captures the effect of changing the regularizer. Use the smoothness of R_t (from the soft CWD approximation or from the quadratic structure of standard WD) to bound this by |R_{t+1}(w_t) - R_t(w_t)| + higher-order terms.

3. **Claim 3 proof:** For quadratic f, the dynamics are linear. Decompose w_t = sum_i v_i(t) e_i in the eigenbasis of H. Each component evolves as v_i(t+1) = (1 - eta*(mu_i + lambda(t))) * v_i(t) + noise. The Lyapunov function becomes sum_i (mu_i + lambda(t)) * v_i(t)^2 / 2. Minimizing over lambda(t) subject to the stability condition yields the stated result via Lagrange multipliers.

### Assumptions

1. **Smoothness:** f is L-smooth (standard).
2. **Bounded noise:** Stochastic gradient noise has bounded variance sigma^2 (standard).
3. **Regularity of R_t:** The regularizer R_t is convex and L_R-smooth in w for each t, and Lipschitz in t (i.e., |R_{t+1}(w) - R_t(w)| is bounded).
4. **Soft CWD approximation:** The alignment mask is replaced by a sigmoid approximation with parameter beta (this changes the algorithm slightly but makes the theory clean; the approximation error is bounded).
5. **For Claim 3:** The loss is quadratic (restrictive but yields closed-form results; the qualitative insights transfer to the nonconvex case).

### Empirical Predictions

1. **WD stability violation detection:** Training runs with WD warmup from 0 to lambda in K steps should show loss spikes for K < K_critical = 1/(eta*lambda). Predict K_critical and measure whether loss spikes occur. Test on CIFAR-10/ResNet-20 with SGD and AdamW.

2. **Optimal WD schedule shape:** The quadratic-optimal lambda(t) is monotonically increasing (WD warmup). This predicts that WD warmup should outperform constant WD and WD cooldown on simple tasks. Test on CIFAR-10 with linear WD warmup vs constant WD vs cosine WD decay.

3. **CWD activation timing:** The soft CWD theory predicts that the alignment mask becomes non-trivial (deviates from 0.5 toward 0 or 1) only after the gradients develop structure (alignment becomes informative). Measure the average CWD mask value over training and correlate with gradient-weight alignment.

4. **Cross-method ranking by CSI:** The Coupling Stability Index (defined as the variance of the angular update speed across layers) should be lower for methods that satisfy the WD stability condition. Rank all WD methods by CSI and verify that CSI correlates with final test performance.

5. **Spectral dynamics across phases:** Measure the effective rank trajectory and verify it exhibits the three-phase structure (high -> decreasing -> stable) with transition timing predictable from (eta, lambda).

### Experimental Plan

All experiments target <= 1 hour per task on the 8x RTX PRO 6000 Blackwell setup.

**Exp 1: WD Stability Violation (Pilot, ~15 min)**
- Model: ResNet-20 on CIFAR-10
- Protocol: Fix eta=0.1, lambda=0.001. Vary WD warmup steps K in {1, 10, 50, 200, 1000}
- Measure: Training loss variance in the first 1000 steps, test accuracy at convergence
- Prediction: K < 100 should show elevated loss variance

**Exp 2: WD Schedule Shape Comparison (~30 min)**
- Model: ResNet-20 on CIFAR-10, VGG-16-BN on CIFAR-100
- Methods: Constant WD, linear warmup WD, cosine decay WD, SWD, logarithmic WD (ADANA)
- Seeds: 42, 123, 456
- Measure: Test accuracy (mean +/- std), CSI trajectory, effective rank trajectory

**Exp 3: CWD Alignment Dynamics (~30 min)**
- Model: ResNet-20 on CIFAR-10
- Track: Per-layer CWD mask average, gradient-weight cosine similarity, effective rank, per-epoch
- Compare: CWD vs AdamW vs soft-CWD (beta=1,10,100)
- Verify: Mask activation timing correlates with alignment structure emergence

**Exp 4: Full Benchmark (~4-8 hours for ImageNet)**
- Models: ResNet-20 (CIFAR-10), VGG-16-BN (CIFAR-100), ResNet-50 (ImageNet)
- Methods: AdamW, SGD+WD, CWD, SWD, AdamWN, cosine WD schedule
- Seeds: 42, 123, 456
- Measure: All proposed metrics (BEM, CSI, AIS), plus effective rank trajectories, weight norm trajectories, alignment cosine trajectories
- Purpose: Validate the unified framework's predictions and demonstrate the standardized metric suite

### Baselines

**Theoretical baselines:**
- Ding et al. (2023)'s convergence rate for Adam-family with fixed decoupled WD
- Chen et al. (2023)'s Lyapunov analysis for Lion with fixed WD
- Li et al. (2020)'s intrinsic LR dynamics for normalized networks

**Empirical baselines:**
- AdamW with constant WD (the universal default)
- SGD with momentum + constant WD
- CWD (current SOTA for simple WD modifications)
- SWD/AdamS (current SOTA for WD scheduling)

### Risk Assessment

1. **CWD proximal approximation may be too crude.** The soft CWD with sigmoid approximation changes the algorithm. If practitioners care about the exact CWD behavior (binary mask), the theory may be seen as analyzing a different algorithm. Mitigation: show experimentally that soft CWD with large beta (e.g., beta=100) is indistinguishable from hard CWD in practice.

2. **WD stability condition may be trivially satisfied.** If all reasonable WD schedules already satisfy the condition, it provides no discrimination. Mitigation: find at least one practical scenario where violation is measurable (WD warmup is the best candidate).

3. **Quadratic analysis may not transfer to nonconvex case.** The optimal WD schedule for quadratic losses may be misleading for deep networks. Mitigation: validate qualitative predictions (WD warmup should help, WD schedule should be smooth) on nonconvex experiments.

4. **Theory-practice gap for convergence rates.** The O(1/sqrt(T)) rate for the general framework is not tighter than known results. The contribution is unification + the stability condition, not rate improvement. If reviewers expect rate improvement, this is a weakness. Mitigation: frame the contribution as a unifying framework that yields new qualitative insights and falsifiable predictions, not a rate improvement.

5. **Effective rank dynamics may not show clear phase transitions.** In practice, the transitions may be smooth, making the three-phase narrative hard to validate quantitatively. Mitigation: use change-point detection algorithms on the rank trajectory and report both qualitative visual evidence and quantitative change-point statistics.

### Novelty Claim

The specific theoretical contributions that are new:

1. **Complete proximal taxonomy of WD variants** -- showing all four sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) as instances of time-varying proximal operators. No previous work provides this complete characterization.

2. **WD Stability Condition** -- a necessary and sufficient condition on the rate of WD schedule change for guaranteed convergence. This is a novel constraint that provides actionable guidance for WD schedule design.

3. **Optimal WD schedule for quadratic losses** -- a closed-form result showing the optimal schedule is curvature-weighted and monotonically increasing. This theoretically justifies WD warmup, which is used in practice but lacks theoretical foundation.

4. **Soft CWD proximal characterization** -- showing that smoothed CWD has a clean proximal interpretation with bounded approximation error to exact CWD. This bridges the gap between CWD's bilevel Pareto theory and the proximal optimization framework.

**Evidence of novelty:** Extensive arXiv and Google Scholar searches for "proximal operator weight decay unified framework" and "time-varying regularization convergence Lyapunov" found no work that combines these elements. PathProx (Yang et al. 2022) treats fixed WD as proximal but does not address time-varying WD or alignment-aware WD. Chen et al. (2023) and Xie & Li (2024) provide Lyapunov analysis for specific optimizers but do not unify across WD variants. The WD stability condition and the optimal quadratic schedule are, to the best of our knowledge, entirely new results.
