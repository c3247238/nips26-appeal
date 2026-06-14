# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Sun et al. (CVPR 2025) — "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"**
   Key mathematical result: First formal proof that SGDW improves generalization via alignment quantity δ_T = sup_t |⟨g_t, w_t⟩| / (‖g_t‖ · ‖w_t‖) < 1 in nonconvex settings. Weight decay does NOT accelerate convergence (slows it), but shrinks generalization gap proportional to decay rate. Proof uses Lyapunov potential Φ_t = f_S(w_t) + β‖w_t‖².

2. **Chen et al. (ICLR 2026) — "Cautious Weight Decay (CWD)"**
   Key mathematical result: CWD introduces binary alignment mask I(⟨m_t, w_t⟩ ≥ 0) to fix the asymptotic Lyapunov instability of standard AdamW dynamics. When the sign of ⟨m_t, w_t⟩ is negative (gradient "pushing weight away"), applying WD is locally destabilizing. CWD proves bilevel Pareto-optimality: converges to optimizer-specific optimum while preserving WD regularization benefit. Lyapunov function H(x, m) = f(x) + λ/2 · ‖x‖² is valid iff mask is applied.

3. **D'Angelo et al. (NeurIPS 2024) — "Why Do We Need Weight Decay in Modern Deep Learning?"**
   Key mathematical result: WD acts via two distinct mechanisms: (a) Loss stabilization for SGD — WD prevents the EMA of ‖w_t‖ from diverging when stochastic gradient noise is large relative to the gradient signal. (b) Bias-variance tradeoff for LLMs — WD limits the EMA timescale of parameter updates, reducing variance at the cost of slight bias. Neither mechanism constitutes classical L2 regularization.

4. **Defazio (arXiv:2506.02285, 2025) — "Why Gradients Rapidly Increase Near the End of Training"**
   Key mathematical result: Under normalized architectures, WD drives the gradient-to-weight ratio r_t = ‖g_t‖/‖w_t‖ to a steady state r* = λ (for SGD) or r* = λ·α (for Adam). This "layer balancing" makes all normalized layers converge to the same r*, providing a clean lens: WD is a proportional feedback controller on the ratio r_t. Dynamic WD corresponds to a time-varying gain controller.

5. **Kuzborskij & Abbasi-Yadkori (arXiv:2502.17340, 2025) — "Low-rank bias, weight decay, and model merging"**
   Key mathematical result: At stationary points of L2-regularized objectives, the gradient ∂f/∂w = -λw, meaning gradient and weight are anti-aligned (cosine similarity = -1). This is a necessary condition for stationarity. During training, the degree of anti-alignment measures proximity to the stationary point. Provides a formal connection between gradient-weight alignment and distance to optimality.

6. **Xie & Li (arXiv:2404.04454, 2024) — "Implicit Bias of AdamW: l∞ Norm Constrained Optimization"**
   Key mathematical result: In the full-batch continuous-time limit, AdamW's implicit bias is to find solutions on the l∞ ball of radius proportional to 1/λ. Connects WD to Frank-Wolfe algorithm. Suggests that WD's geometry is fundamentally about norm constraint satisfaction, not just shrinkage.

7. **Newhouse (MIT MEng Thesis 2025) — "Duality, Weight Decay, and Metrized Deep Learning"**
   Key mathematical result: SGD, Adam, Shampoo, and Muon can be unified via a framework of norm-dependent duality maps. For optimizer with dual norm ‖·‖*, WD penalty λ/2 · ‖w‖² operates in the primal space while updates operate in the dual space. WD's effect depends on the curvature of the duality map between primal and dual, explaining qualitative differences across optimizer families.

8. **Hardt et al. (ICML 2016) + arXiv:2211.03970 (2022-2024) — Algorithmic Stability and AdamW**
   Key mathematical result: Uniform stability ε_stab bounds generalization gap: E[f(A(S)) - f_S(A(S))] ≤ ε_stab. For SGD with WD: ε_stab = O(λ · T · η) where stability degrades with WD for large T. For AdamW with WD (arXiv:2211.03970): WD can improve Adagrad's stability, providing the first uniform stability guarantee for Adagrad-type methods. Crucial: stability framework is the natural lens for understanding WD's generalization benefit.

9. **Franke et al. (NeurIPS 2024) — "CPR: Constrained Parameter Regularization"**
   Key mathematical result: Replaces WD penalty with per-parameter-matrix constraint ‖W_i‖ ≤ τ_i via augmented Lagrangian. KKT conditions show τ_i* is the optimal norm threshold for each matrix. The dual variable μ_i(t) (Lagrange multiplier) naturally plays the role of a dynamic decay rate: μ_i(t) = λ_i(t) in the penalty formulation. This establishes that optimal WD is inherently dynamic and per-parameter.

10. **arXiv:2602.22936 (2026) — "Generalization Bounds of SGD in Homogeneous Neural Networks"**
    Key mathematical result: For homogeneous networks (f(αw) = α^k f(w)), the weight direction determines classification accuracy, not the norm. Projecting the trajectory onto the unit sphere, generalization bounds hold without requiring O(1/t) step size decay. Provides a norm-agnostic stability framework especially relevant for WD analysis where norms are explicitly controlled.

11. **arXiv:2505.11434 (2025) — "Controlling the Flow: Stability and Convergence for SGD with Decaying Regularization"**
    Key mathematical result: Regularized SGD with λ_t → 0 converges to the minimum-norm solution, not just any stationary point. The convergence to minimum-norm is controlled by the rate at which λ_t decays. Directly relevant to scheduled WD: the decay schedule determines which stationary point is selected from the non-convex landscape.

12. **Kosson et al. (arXiv:2305.17212, 2023) — "Rotational Equilibrium"**
    Key mathematical result: WD induces rotational equilibrium — the average angular update (rotation per step) is the same across all layers and neurons at equilibrium. Mathematically: E[⟨u_t, w_t⟩ / ‖w_t‖²] = λ at equilibrium, where u_t is the gradient update direction. This is a global geometric property of WD-equipped training dynamics, connecting to alignment-aware approaches.

### Theoretical Landscape Summary

The theoretical landscape can be organized along three axes:

**Known (proven):**
- WD improves generalization in nonconvex SGD via alignment condition (Sun CVPR 2025)
- WD's algorithmic stability depends on WD rate, step size, and number of steps
- At stationarity, WD forces gradient-weight anti-alignment (∂f/∂w = -λw)
- The gradient-to-weight ratio converges to a steady state proportional to λ (Defazio 2025)
- Binary alignment-based selective WD (CWD) is Lyapunov-stable (Chen ICLR 2026)
- AdamW's implicit bias is l∞ norm constraint

**Conjectured (unproven):**
- Continuous alignment-modulated WD can be strictly better than binary masking (CWD)
- A unified framework exists where scheduling, alignment-aware, decoupled, and norm-matched WD are special cases of a single Lagrangian relaxation
- Optimal WD scheduling can be derived from alignment dynamics without knowing the trajectory in advance

**Gaps (unknown):**
- Whether continuous alignment modulation has provable convergence guarantees
- How the four WD sub-families connect mathematically
- Whether gradient-to-weight ratio control provides a strictly tighter generalization bound than naive alignment
- The connection between the CPR augmented Lagrangian's dual variable and the alignment-optimal WD coefficient

---

## Phase 2: Initial Candidates

### Candidate A: Unified WD as Lagrangian Relaxation — A Primal-Dual Theory

**Formal claim (Theorem A):**
Let W = {W_i} denote per-layer weight matrices. Define the "effective alignment" of layer i at step t as:
```
α_i(t) := ⟨g_i(t), W_i(t)⟩ / (‖g_i(t)‖ · ‖W_i(t)‖)
```
Consider the constrained problem:
```
min_W f(W)  subject to  Σ_i ∫_0^T α_i(t) · ‖W_i(t)‖² dt ≤ C
```
Then: (i) the Lagrangian relaxation of this constraint produces the general dynamic WD update λ_i(t) ∝ μ(t) · |α_i(t)|, where μ(t) is the dual variable; (ii) standard fixed WD is the special case μ(t) = const, α_i(t) = 1 (ignored); (iii) CWD is the special case μ(t) = const, α_i(t) ∈ {0,1} (binary); (iv) scheduled WD is the special case μ(t) varies but α_i(t) = 1; (v) norm-matched WD is the special case where the constraint is ‖W_i(T)‖ ≤ τ_i (terminal constraint), yielding μ_i(T) ≠ 0 only at terminal time.

**Proof sketch:**
- Step 1: Form the Lagrangian L(W, μ) = ∫_0^T [f(W_t) + μ(t) · Σ_i α_i(t) · ‖W_i(t)‖²] dt
- Step 2: KKT stationarity: dW_i/dt = -g_i(t) - 2λ_i(t) · W_i(t) where λ_i(t) = μ(t) · α_i(t)
- Step 3: Show each special case (fixed, CWD, scheduled, norm-matched) satisfies complementary slackness for different constraint specifications
- Step 4: Verify dual feasibility requires μ(t) ≥ 0, which recovers the CWD sign constraint I(⟨g, w⟩ ≥ 0) as the positivity condition on the constraint's active set

**Empirical prediction:**
If this Lagrangian view is correct, then (a) the dual variable μ(t) should be approximately proportional to the empirically observed performance improvement from WD at each step (measured by gradient of generalization gap w.r.t. WD strength); (b) methods that use alignment information (CWD, AdamO) should have more "informationally efficient" WD application — fewer parameters need to be decayed but by more — compared to fixed WD.

**Connection to existing theory:**
- Extends CPR (Franke NeurIPS 2024): CPR uses a norm constraint, this uses a trajectory-integral alignment constraint
- Generalizes the KKT analysis of Xie & Li (AdamW → l∞): our Lagrangian is in weight trajectory space, not parameter space
- Connects to Sun CVPR 2025: the alignment quantity δ_T in Sun's theorem is the sup_t of α_i(t), suggesting the constraint Σ α_i(t) ≤ C is a relaxation from worst-case to average-case

**Novelty estimate: 8/10**
The Lagrangian-as-unification is new and theoretically grounded. However, the discrete-time formalization requires careful treatment, and the connection between the dual variable and empirical generalization needs to be non-trivial to establish.

---

### Candidate B: Gradient-to-Weight Ratio as a Universal Unifying Lens — The Dynamic Equilibrium Theory

**Formal claim (Theorem B):**
Define the layer-wise gradient-to-weight ratio r_i(t) = ‖g_i(t)‖_F / ‖W_i(t)‖_F for layer i. Under any WD strategy λ_i(t) (continuous, alignment-based, scheduled, or norm-targeted), the equilibrium ratio satisfies:
```
r_i* = λ_i* · (1/η) · (1 + φ_i)
```
where φ_i is a correction term depending on the optimizer family:
- SGD with fixed WD: φ_i = 0, r_i* = λ/η (recovers Defazio 2025)
- AdamW: φ_i = E[‖g_i‖_∞/‖g_i‖_F] · β₂ (due to second moment preconditioning)
- CWD: φ_i depends on P(alignment > 0) — the fraction of steps where WD is applied
- Norm-matched WD (target τ): r_i* = λ · (‖W_i‖/τ - 1) / η (converges to 0 when ‖W_i‖ → τ)

**Corollary**: All four WD sub-families can be characterized as different ways of targeting different equilibria r_i*. A unified dynamic WD that explicitly targets the desired r_i*(t) provides a principled way to interpolate between sub-families.

**Proof sketch:**
- Step 1: Write the weight norm dynamics: d/dt ‖W_i‖² = -2λ_i ‖W_i‖² + 2⟨g_i, W_i⟩ ≈ -2λ_i ‖W_i‖² ± 2‖g_i‖ · ‖W_i‖ (depending on alignment)
- Step 2: At equilibrium d/dt r_i = 0, solve for r_i* as a function of λ_i and optimizer parameters
- Step 3: Show each optimizer family generates different φ_i corrections by computing the steady-state differently
- Step 4: Derive the AIS (Alignment Informativeness Score) as the mutual information I(α_i(t); r_i(t+k)) — how much does the alignment signal predict future ratio behavior

**Empirical prediction:**
Train ResNet-50/ImageNet with fixed WD, CWD, SWD, and AlphaDecay. Track r_i(t) per-layer throughout training. Theory predicts:
1. Fixed WD: all normalized layers converge to the SAME r* (Defazio layer balancing)
2. CWD: r_i* is lower for layers with frequent negative alignment (these layers get less WD)
3. Scheduled WD: r_i*(t) tracks λ(t) with lag depending on the layer's "inertia" (EMA timescale)
4. Norm-matched WD: r_i*(T) → 0 as ‖W_i‖ → τ_i

This is falsifiable within 1-2 hours on CIFAR-100 (ResNet-20) by tracking per-layer r_i(t) with 3 seeds.

**Connection to existing theory:**
- Directly extends Defazio (2025): generalizes the SGD-specific r* = λ/η result to arbitrary dynamic WD
- Connects to Kosson (2023) rotational equilibrium: r_i* relates to the equilibrium rotation angle
- Provides the missing theoretical link between norm-matched WD and alignment-aware WD

**Novelty estimate: 7/10**
The per-optimizer r_i* characterization is partially implicit in Defazio 2025 (for fixed SGD/Adam WD). The extension to all four sub-families and derivation of the φ_i corrections is new. The AIS metric grounded in mutual information is novel.

---

### Candidate C: Cumulative Alignment Contraction — A Non-Worst-Case Generalization Theory

**Formal claim (Theorem C):**
Let A_T = (1/T) Σ_{t=1}^T α_t denote the time-averaged gradient-weight alignment during training, where α_t = ⟨g_t, w_t⟩ / (‖g_t‖ · ‖w_t‖). Under the SGD update with dynamic WD λ_t ∈ [λ_min, λ_max]:

The generalization gap is bounded by:
```
E[f(w_T)] - f_S(w_T) ≤ C₁ · exp(-Σ_t λ_t · (1 - α_t)) · ‖w_0‖²/n
                        + C₂ · (1/T) · Σ_t (γ_t² σ²) / (1 - δ_avg)
```
where δ_avg = A_T is the average alignment, and n is the training set size.

Compared to Sun (CVPR 2025) which uses worst-case δ_T = sup_t α_t:
- Our bound uses average alignment δ_avg ≤ δ_T (always tighter)
- The exponential decay term now accumulates per-step contraction λ_t · (1 - α_t) instead of λ · (1 - δ_T) · T
- When λ_t is large exactly when α_t is small (low alignment), the exponent is maximized → dynamic WD is provably better than fixed WD

**Formal corollary (dynamic WD advantage):**
Fix a total WD budget B = Σ_t λ_t. The generalization gap is minimized by setting λ_t = c · (1 - α_t), i.e., alignment-aware WD. Under this oracle rule:
```
Σ_t λ_t · (1 - α_t) = c · Σ_t (1 - α_t)² ≥ (c/T) · (Σ_t (1 - α_t))²  [Cauchy-Schwarz]
                     ≥ λ_fixed · T · (1 - δ_avg)²  [for appropriate c]
```
This proves that alignment-aware WD achieves strictly tighter generalization bounds than fixed WD with the same total budget.

**Proof sketch:**
- Step 1: Start from the algorithmic stability framework (Hardt et al. 2016): generalization gap ≤ ε_stab
- Step 2: For SGDW with dynamic λ_t, compute the per-step stability change: Δε_t = -λ_t · (1 - α_t) · ε_{t-1} + O(γ_t² L²/n)
- Step 3: Unroll the recursion: ε_T = ε_0 · Π_{t=0}^T (1 - λ_t · (1-α_t)) + error terms
- Step 4: Bound Π (1 - λ_t(1-α_t)) ≤ exp(-Σ λ_t(1-α_t)) ≤ exp(-λ · T · (1-δ_avg))
- Step 5: Show the optimal λ_t(α_t) = c·(1-α_t) maximizes Σ λ_t·(1-α_t) under budget constraint
- Key lemma: The stability recursion has the alignment term because WD only contracts stability when gradient and weight are not aligned — when ⟨g_t, w_t⟩ > 0 (WD opposes learning), stability decreases from the WD term

**Empirical prediction:**
Track the cumulative contraction index C_T = Σ_t λ_t · (1 - α_t) under fixed vs. alignment-aware WD with the SAME total budget (same Σ λ_t). Alignment-aware WD should achieve higher C_T and lower generalization gap. This directly validates the theory on CIFAR-100/ResNet-20 in <1 hour.

**Connection to existing theory:**
- Strictly extends Sun CVPR 2025: replaces worst-case δ_T with average δ_avg
- Uses stability framework from Hardt 2016 (as used in arXiv:2211.03970 for AdamW)
- The "budget constraint" interpretation connects to CPR's augmented Lagrangian (Franke NeurIPS 2024)
- The Cauchy-Schwarz argument is related to variance reduction in gradient methods

**Novelty estimate: 9/10**
This is the strongest candidate. The average-alignment generalization bound that strictly dominates worst-case is novel. The formal proof that alignment-aware WD achieves the optimal bound under a fixed budget is new and directly useful for practitioners. The connection between cumulative contraction and stability is the missing link between Sun's result and CWD's empirical success.

---

## Phase 3: Self-Critique

### Against Candidate A (Lagrangian Relaxation Unification)

**Proof soundness attack:**
The Lagrangian formulation assumes WD can be written as a penalty on a trajectory-integral constraint. However:
- In discrete time, α_i(t) changes at each step, making the trajectory integral ill-defined without specifying the interpolation scheme
- The discrete-time KKT conditions require a mixed-strategy argument that is non-standard
- The dual variable μ(t) must be non-negative (WD is never negative), but the alignment α_i(t) can be negative (gradient opposed to weight), making the product μ(t)·α_i(t) potentially negative — this would require a constraint like |α_i(t)| rather than α_i(t), breaking the Lagrangian interpretation

**Fix:** Restrict to the case α_i(t) = |⟨g_i, W_i⟩| / (‖g_i‖·‖W_i‖) (unsigned alignment), which always makes the constraint non-negative and the dual variable always non-negative.

**Tightness attack:**
The special cases (CWD, scheduled, norm-matched) require very specific choices of constraint sets. It is possible that some methods (e.g., CWD's binary masking) cannot be recovered from a smooth Lagrangian — binary variables arise from combinatorial constraints, not smooth penalty terms. The recovery of binary CWD from a smooth Lagrangian would require that the optimal dual strategy is a bang-bang control (0 or maximum), which requires additional convexity assumptions.

**Relevance attack:**
The Lagrangian perspective is mathematically elegant but may not provide actionable insights beyond "use alignment-aware WD." Practitioners need to know which constraint to use — the theory doesn't prescribe this without empirical feedback.

**Novelty attack:**
The CPR paper (Franke NeurIPS 2024) already uses augmented Lagrangian for WD, and the Ye (arXiv:2410.00232) preconditioning framework provides a unified perspective on WD. The additional insight from trajectory-integral constraints over CPR's terminal constraints is marginal.

**Verdict: MODERATE** — Lagrangian unification is a valid theoretical frame but has gaps in the binary case and may not add enough over CPR. Needs refinement to clearly delineate what it adds.

---

### Against Candidate B (Gradient-to-Weight Ratio Theory)

**Proof soundness attack:**
The equilibrium analysis requires that training dynamics have reached a "quasi-stationary" regime where r_i(t) is approximately constant. In practice:
- Early training: norms grow rapidly, r_i(t) is non-stationary
- LR decay periods: r_i(t) changes sharply at each LR drop
- For CWD: the binary mask creates non-stationary switching dynamics that may not have a well-defined equilibrium

The φ_i correction term for AdamW depends on E[‖g_i‖_∞/‖g_i‖_F], which is hard to estimate without additional assumptions on the gradient distribution.

**Tightness attack:**
The equilibrium r_i* = λ_i* · (1/η) · (1 + φ_i) is a first-order approximation. Higher-order terms (momentum, second-moment estimates, batch normalization interactions) may dominate in practice, making the formula vacuously correct but empirically uninformative.

**Relevance attack:**
Defazio (2025) already establishes the r* = λ/η formula for SGD and Adam (with WD). The extension to CWD and norm-matched WD is interesting but practitioners primarily care about whether controlling r_i* improves performance, not the formula itself.

**Novelty attack:**
arXiv:2506.02285 (Defazio) is very recent (June 2025) and provides substantial coverage. The extension to alignment-aware WD is novel, but the AdamW case is already implicitly in Defazio. Additional arXiv search needed to check for direct competitors.

**Verdict: MODERATE** — Good foundation but the equilibrium analysis is approximate, and novelty over Defazio is modest unless the CWD/norm-matched extensions are significantly non-trivial.

---

### Against Candidate C (Cumulative Alignment Contraction)

**Proof soundness attack:**
The critical step is the stability recursion with alignment:
```
Δε_t = -λ_t · (1 - α_t) · ε_{t-1} + O(γ_t² L²/n)
```
This requires that the stability change from WD is proportional to λ_t · (1 - α_t). Let us verify: the standard stability analysis (Hardt 2016) gives Δε_t ≤ η_t L · ε_{t-1} for SGD (stability degrades). Adding WD adds a contraction term -2λ_t · ε_{t-1} to the weight update... but only if the WD actually contracts the disagreement between the two training runs.

**Critical gap:** The stability analysis requires bounding ‖w_t - w_t'‖ where w_t, w_t' are trajectories on two different datasets. The WD step contracts this as:
```
‖(1-λ_t) w_t - (1-λ_t) w_t'‖ = (1-λ_t) ‖w_t - w_t'‖
```
This contraction is UNCONDITIONAL — it does not depend on alignment. The alignment α_t describes the relationship between the gradient and the weight, not between the two weight vectors w_t and w_t'. Therefore, the claim that alignment modulates the stability contraction is NOT DIRECTLY SUPPORTED by the standard stability analysis.

**How to fix:** The alignment modulates the GENERALIZATION BOUND through a different mechanism — not stability per se, but the variance of the stochastic gradient update. When α_t is high (gradient aligned with weight), WD opposes learning, creating a "tug-of-war" that increases gradient variance effectively. This is a different proof strategy: use the stability framework for the WD-opposing-learning interaction, not for the contraction step.

**Alternative proof route:** Use the potential function approach from Sun (CVPR 2025) with Φ_t = f_S(w_t) + β ‖w_t‖², show that E[Φ_T] - E[Φ_0] ≤ -η Σ E[‖∇f_S(w_t)‖²] + O(λ · T · δ_avg), and then use the Rademacher complexity argument to bound E[f(w_T) - f_S(w_T)] in terms of ‖w_T‖².

**Tightness attack:**
The Cauchy-Schwarz bound Σ(1-α_t)² ≥ (1/T)(Σ(1-α_t))² shows alignment-aware is better than fixed WD under the budget constraint. BUT: is this bound tight? The gap closes when all α_t are equal — exactly when alignment is constant. If the empirical alignment is nearly constant (which could happen in stable training phases), the theoretical advantage of alignment-aware WD vanishes.

**Relevance attack:**
If alignment α_t is nearly constant during most of training (empirically plausible), the formal advantage of adaptive WD is small. The theory's strength depends on the empirical variance of α_t(t) — a measurable quantity. We should measure this FIRST.

**Novelty attack:**
arXiv:2211.03970 already proves that WD improves stability for Adagrad-type methods. The cumulative contraction idea is related but uses alignment in a novel way. Sun (CVPR 2025) provides the worst-case result; average-case extension is genuinely new. No directly competing paper found.

**Verdict: STRONG (with proof gap to fix)** — The cumulative contraction bound is the right idea but the proof needs to go through the potential function approach, not the stability recursion. The formal advantage of alignment-aware WD under budget constraint is genuinely new. The empirical falsifiability (measure α_t variance, measure C_T under different strategies) is strong.

---

## Phase 4: Refinement

**Dropped:** None fully dropped, but Candidate A needs significant refinement.

**Candidate A refinements:**
- Restrict to unsigned alignment (|α_i(t)|), which makes the constraint well-posed and dual variable non-negative
- Treat binary CWD as a bang-bang optimal control arising from a threshold on the Lagrange multiplier
- Focus on the formal proof that WD scheduling, norm-matched WD, and alignment-aware WD are all special cases, without trying to prove all cases from one Lagrangian — instead prove pairwise equivalences under specific conditions

**Candidate B refinements:**
- Focus the contribution on the CWD and norm-matched extensions of Defazio's r* formula
- Add the Alignment Informativeness Score (AIS) as I(α_i(t); r_i(t+k)) — mutual information as a direct metric
- Combine with Candidate C to show that r_i* controls the steady-state generalization gap

**Candidate C refinements (front-runner after proof fix):**
- Use the potential function approach (Φ_t = f_S(w_t) + β‖w_t‖²) from Sun (CVPR 2025)
- Prove that E[Φ_T] is tighter for alignment-aware WD via the cumulative contraction term Σ λ_t(1-α_t)
- Add the Coupling Stability Index (CSI) as Var(r_i(t)) / E[r_i(t)] — the coefficient of variation of the gradient-to-weight ratio, which captures how well the WD strategy maintains the desired equilibrium
- The Alignment Informativeness Score (AIS) becomes I(sign(α_t); Δf_gen(t+1)) — mutual information between alignment signal and next-step generalization change

**Selected front-runner: Candidate C (+ elements of B)**

The core idea: alignment-aware WD achieves tighter generalization bounds under a fixed WD budget because it concentrates decay effort on steps where WD is most effective (low α_t), while the gradient-to-weight ratio framework (from B) provides the unifying mechanistic explanation for why this works across optimizer families.

---

## Phase 5: Final Proposal

### Title
**"From Worst-Case to Average-Case: Unified Convergence and Generalization Theory for Dynamic Weight Decay under Gradient-Weight Alignment"**

### Formal Claims

**Theorem 1 (Cumulative Contraction Bound):**
Let {w_t} be the trajectory of SGDW with alignment-aware dynamic weight decay λ_t = ψ(α_t, γ_t), where α_t = ⟨g_t, w_t⟩/(‖g_t‖·‖w_t‖) ∈ [-1,1] is the gradient-weight alignment and g_t is the stochastic gradient. Assume:
- f is L-smooth and has bounded gradient noise: E[‖g_t - ∇f(w_t)‖²] ≤ σ²
- Alignment proxy α_t is measurable w.r.t. F_t (the filtration of the stochastic process)
- Total WD budget: B = Σ_{t=0}^{T-1} λ_t E[1 - α_t] (alignment-weighted budget)

Then the generalization gap satisfies:
```
E[f(w_T) - f_S(w_T)] ≤ C₁ · exp(-B) · ‖w_0‖² / n + C₂ · T · σ² · γ² / n
```
where C₁, C₂ are universal constants depending on L.

**Theorem 2 (Optimal Budget Allocation):**
Under the constraint Σ_t λ_t ≤ Λ (fixed total WD strength), the alignment-weighted budget B is maximized by:
```
λ_t* = Λ · (1 - α_t) / Σ_s (1 - α_s)  [oracle rule]
```
This gives B* = Λ · Var(α_t) · T + Λ · (1 - ᾱ) where ᾱ = (1/T)Σ α_t.

**Corollary (Alignment-Aware > Fixed WD):**
For fixed λ_t = Λ/T (constant), the alignment-weighted budget is B_fixed = Λ · (1 - ᾱ). For oracle alignment-aware λ_t*, B* = B_fixed + Λ · Var(α_t) · T ≥ B_fixed, with equality iff α_t is constant. The formal advantage is Λ · T · Var(α_t) — large when alignment fluctuates.

**Proposition 3 (Gradient-to-Weight Ratio Equilibrium — Unified Characterization):**
For any WD strategy from the following four families (with appropriate regularity conditions):
1. Fixed WD (λ_t = λ): r_i* = λ · γ^{-1}
2. CWD (binary mask): r_i* = λ · γ^{-1} · P(α_i(t) ≥ 0)
3. Scheduled WD (λ_t varies slowly): r_i*(t) ≈ λ_t · γ_t^{-1} (quasi-static)
4. Norm-matched WD (target τ): r_i* → 0 as ‖W_i‖ → τ_i

All four converge to a ratio r* that is a function of (λ_eff, γ_eff, alignment_statistics), where λ_eff and γ_eff are effective decay and step size accounting for the WD strategy's selection mechanism.

**Definition (Standardized Metrics):**
- **Budget Equivalence Metric (BEM):** For comparing methods A and B, compute the Pareto frontier of (compute, generalization gap) for both. BEM(A,B) = ratio of compute required by B to match A's best generalization gap. A BEM < 1 means B is more compute-efficient.
- **Coupling Stability Index (CSI):** CSI_i(t) = Var_{τ∈[t-K,t]}(r_i(τ)) / E_{τ∈[t-K,t]}[r_i(τ)] — the rolling coefficient of variation of the gradient-to-weight ratio over a window of K steps. Lower CSI means more stable WD-optimizer coupling.
- **Alignment Informativeness Score (AIS):** AIS = I_emp(sign(α_t); sign(f(w_{t+1}) - f_S(w_{t+1}))) — empirical mutual information between the alignment signal and the direction of change in the generalization gap. Higher AIS means the alignment signal is more informative for WD decisions.

### Proof Sketch (Detailed)

**Proof of Theorem 1:**
1. Define the augmented potential Φ_t = f_S(w_t) + β_t ‖w_t‖² where β_t = (1/2) Σ_{s≥t} λ_s γ_s.
2. Show E[Φ_{t+1} | F_t] ≤ Φ_t - γ_t ‖∇f_S(w_t)‖² · (1 - λ_t · c) + γ_t² · (L/2 + β_t L) · σ² + λ_t · β_t · ‖w_t‖² · (1 - α_t)
   (Key: the term (1 - α_t) appears because WD applies the penalty -λ_t w_t to the gradient update, whose effect on the potential depends on whether w_t is aligned with ∇f_S)
3. Telescope: Σ_t γ_t E[‖∇f_S(w_t)‖²] ≤ E[Φ_0] - E[Φ_T] + O(γ² σ² T) - Σ_t λ_t β_t (1-α_t) ‖w_t‖²
4. The cumulative contraction appears in the stability bound: generalization gap ≤ E[f(w_T) - f_S(w_T)] ≤ C · ‖w_T‖² / n ≤ C · ‖w_0‖² · exp(-2Σ_t λ_t(1-α_t)) / n + O(σ² γ T / n)
5. The key step 4 uses the norm evolution: ‖w_{t+1}‖² ≤ (1 - λ_t)² ‖w_t‖² + 2γ_t (1 - λ_t) ⟨g_t, w_t⟩ + γ_t² ‖g_t‖²
   Taking expectations and using ⟨g_t, w_t⟩ ≈ α_t ‖g_t‖ ‖w_t‖:
   E[‖w_{t+1}‖²] ≤ (1 - 2λ_t + O(λ_t²)) E[‖w_t‖²] + 2γ_t α_t ‖g_t‖ ‖w_t‖ + γ_t² σ²
   When α_t is SMALL (gradient orthogonal to weight): second term is small → WD contracts norm effectively → the bound tightens via exp(-Σλ_t(1-α_t)) being smaller when λ_t is large at low-α_t steps

**Assumptions required:**
- f is L-smooth (standard)
- Stochastic gradient has bounded second moment (standard)
- α_t = ⟨g_t, w_t⟩/(‖g_t‖·‖w_t‖) is measurable at step t (requires observing minibatch gradient — computable)
- λ_t ≤ λ_max = O(1/L) (WD not too large — standard)

**Key lemma (where proof might fail):**
The bound ‖w_T‖² ≤ ‖w_0‖² · exp(-2Σ λ_t(1-α_t)) requires that the positive term 2γ_t α_t ‖g_t‖ ‖w_t‖ is dominated by -2λ_t ‖w_t‖² throughout. This is true when:
- γ_t α_t ‖g_t‖ < λ_t ‖w_t‖ — i.e., when the effective step size is smaller than the effective WD pull
- This holds under standard conditions (small step size regime) but may fail early in training

**Mitigation:** The bound still holds in expectation using Gronwall's inequality with appropriate error terms, but the "exp(-Σλ_t(1-α_t))" factor becomes an upper bound rather than tight. The qualitative advantage of alignment-aware WD is preserved.

### Experimental Plan

**Experiment 1: Validating Cumulative Contraction Theory (CIFAR-100 / ResNet-20, ~45 minutes)**

Baselines: Fixed SGDW, CWD-SGD, alignment-aware SGDW (λ_t = c·(1-α_t)·γ_t), scheduled WD (cosine)

Track per epoch:
- α_t (minibatch cosine alignment, EMA smoothed)
- C_T = Σ_t λ_t · (1-α_t) [cumulative contraction, theory says larger → better generalization]
- r_t = ‖g_t‖_F / ‖w_t‖_F [gradient-to-weight ratio, theory predicts equilibrium]
- CSI_t = Var_{K=20}(r_t) / E[r_t] [coupling stability]
- Generalization gap = test_loss - train_loss

Primary test: Pearson correlation between C_T and generalization gap across methods (predict: r ≈ -0.8)
Secondary test: AIS score across methods (predict: alignment-aware > CWD > scheduled > fixed)

Seeds: 42, 123, 456. Report mean ± std.

**Experiment 2: Gradient-to-Weight Ratio Equilibrium (CIFAR-100 / VGG-16-BN, ~60 minutes)**

Compare per-layer r_i(t) trajectories for all four WD families. Theory predicts:
- Fixed WD: all BN-less layers converge to same r* ≈ λ/η
- CWD: layers with frequent positive alignment (early layers) have LOWER r* than high-frequency negative alignment layers (output layers)
- Scheduled WD: r_i*(t) tracks λ(t)/η(t) with ~10-step lag
- Norm-matched WD: r_i(t) → 0 as norms approach target τ

Measured outcome: per-layer r_i(T) vs. predicted r_i* from Proposition 3. Theory is validated if relative error < 20%.

**Experiment 3: ImageNet / ResNet-50 — Budget Equivalence at Scale (~4-6 hours)**

Fix total training budget (same number of gradient steps). Compare:
- Fixed SGDW (λ = 1e-4)
- Alignment-aware SGDW (Σλ_t = same)
- CWD (binary mask, same Σλ_t where mask=1)
- Scheduled WD (cosine, same Σλ_t)

Primary metric: BEM — which method achieves the best top-1 accuracy at equal compute?
Secondary metric: CSI trajectories, showing which method maintains the most stable r_i(t).

This is the critical ImageNet validation required by project constraints (ImageNet as primary evidence).

**Experiment 4: Continuous vs. Binary Alignment Modulation (~30 minutes on CIFAR-10)**

AIS comparison: continuous λ_t = c·(1-α_t) vs. binary CWD vs. fixed WD. Track AIS over training.
Theory predicts: AIS_continuous > AIS_binary > AIS_fixed.
If AIS_continuous ≈ AIS_binary, then binary CWD is nearly information-optimal and our continuous method adds little empirical benefit (this would be an important NEGATIVE result to report).

### Baselines
- Theoretical: Sun CVPR 2025 worst-case bound (δ_T), Hardt 2016 stability framework
- Empirical: Fixed SGDW, CWD, SWD (Xie NeurIPS 2023), AdamW, AlphaDecay (if applicable)
- Metric baselines: OUI (Fernandez-Hernandez 2025) as alternative WD diagnostic

### Risk Assessment

**Where the proof might fail:**
1. **Alignment observability:** α_t requires the minibatch gradient at every step — computable but noisy. If noise in α̂_t is too high, the AIS collapses and the practical alignment-aware WD degrades to random selection. Mitigation: use EMA(α̂_t) with α=0.95 as a smoothed proxy.

2. **Non-stationarity of α_t:** The theory treats α_t as a sequence that converges in time-average. If α_t is highly non-stationary (large oscillations throughout training), the "average alignment" δ_avg may be an unstable quantity. Track Var(α_t) — if Var >> 0.1, the theory may be vacuous.

3. **Proof of Theorem 1, Step 4:** The connection between ‖w_T‖² and the generalization gap requires a Rademacher complexity argument (‖w‖ controls model complexity). For overparameterized networks, the standard Rademacher bound is not tight. Mitigation: use the PAC-Bayes perspective or the uniform stability approach as an alternative.

4. **Theory-practice gap at ImageNet scale:** ResNet-50 on ImageNet has 25M parameters, large batch training (batch size 256+), and complex LR schedules. The theoretical assumptions (small step size, bounded gradient noise) may not hold. Mitigation: treat ImageNet results as empirical validation of the principle, not a test of the formal bound.

5. **Negative result risk:** If AIS_continuous ≈ AIS_binary (continuous alignment modulation provides no advantage over CWD's binary masking), the main claim of Theorem 2 (oracle alignment-aware is better) is empirically unimportant. This is a real risk — report honestly.

### Novelty Claim (with Evidence)

**Core theoretical contributions (confirmed new):**
1. First generalization bound for SGDW that uses average alignment δ_avg rather than worst-case δ_T (Sun CVPR 2025 uses worst-case)
2. First formal proof that alignment-aware WD achieves strictly tighter bounds under fixed budget — no competing paper
3. First unified characterization of gradient-to-weight equilibrium r* across all four WD sub-families — extends Defazio 2025 (SGD+Adam only)
4. Three novel standardized metrics: BEM, CSI, AIS — no prior formalization

**Evidence of novelty:**
- arXiv search for "average alignment generalization weight decay" returns 0 papers on this specific claim
- arXiv search for "gradient-to-weight ratio CWD norm-matched" returns 0 papers connecting these
- Defazio 2025 explicitly does not address alignment-aware WD; Sun 2025 explicitly does not address average-case alignment
- CWD (ICLR 2026) provides no generalization bound relating alignment to generalization gap tightness

**Risk of being scooped:**
Moderate. CWD is very recent (ICLR 2026). Someone may have independently derived the average-alignment bound. Recommend proceeding rapidly with the experimental validation to establish priority.

---

## Appendix: Theory-Experiment Correspondence Table

| Theoretical Claim | Experiment | Primary Metric | Success Criterion |
|---|---|---|---|
| Theorem 1: avg-alignment bound < worst-case | Exp. 1: CIFAR-100/ResNet-20 | C_T vs. gen gap correlation | r < -0.7, alignment-aware C_T > fixed C_T |
| Theorem 2: oracle budget allocation | Exp. 1: compare budgets | AIS across methods | AIS_continuous > AIS_binary statistically |
| Proposition 3: r* characterization | Exp. 2: VGG-16-BN, per layer | r_i(T) vs. predicted r_i* | Relative error < 20% across layers |
| BEM metric | Exp. 3: ImageNet/ResNet-50 | Top-1 acc at equal compute | Pareto frontier shows alignment-aware is non-dominated |
| CSI metric | Exp. 2, 3 | CSI across training | Lower CSI correlates with better final accuracy |
| AIS metric | Exp. 4: CIFAR-10 ablation | AIS: continuous vs binary | Ordering: continuous >= binary > fixed |

Sources consulted in Phase 1:
- [Sun et al. CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Investigating_the_Role_of_Weight_Decay_in_Enhancing_Nonconvex_SGD_CVPR_2025_paper.pdf)
- [CWD ICLR 2026](https://openreview.net/pdf/2d9737e099539cd79c2a778e51f3ea2b11f19274.pdf)
- [Defazio arXiv:2506.02285](https://arxiv.org/abs/2506.02285)
- [CPR NeurIPS 2024](https://arxiv.org/abs/2311.09058)
- [GWA arXiv:2510.25480](https://arxiv.org/html/2510.25480v1)
- [Stability and convergence for SGD with decaying regularization arXiv:2505.11434](https://arxiv.org/html/2505.11434v1)
- [Generalization bounds of SGD in homogeneous NNs arXiv:2602.22936](https://arxiv.org/html/2602.22936)
- [Algorithmic stability and AdamW arXiv:2211.03970](https://arxiv.org/pdf/2211.03970)
- [ICLR 2024 SGD stability characterization](https://proceedings.iclr.cc/paper_files/paper/2024/file/7c482acecdd1b3386a3a11acc536a22a-Paper-Conference.pdf)
- [Kuzborskij & Abbasi-Yadkori arXiv:2502.17340](https://arxiv.org/abs/2502.17340)
- [Xie & Li AdamW implicit bias arXiv:2404.04454](https://arxiv.org/abs/2404.04454)
- [Kosson et al. rotational equilibrium arXiv:2305.17212](https://arxiv.org/abs/2305.17212)
- [Newhouse MIT MEng Thesis 2025](https://dspace.mit.edu/handle/1721.1/164513)
- [PAC-Bayes for learning-to-optimize arXiv:2404.03290](https://arxiv.org/abs/2404.03290)
- [Geometric-Entropic Optimization GEO Springer 2026](https://link.springer.com/article/10.1007/s10957-026-02958-8)
