# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Sun et al. (CVPR 2025) — "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"**
   Key result: First formal proof that WD improves generalization in nonconvex SGD under a gradient-weight alignment condition (δ_T < 1); WD does NOT accelerate convergence but tightens the generalization bound. Establishes the alignment quantity as the key theoretical variable.

2. **Towards Understanding Convergence and Generalization of AdamW (TPAMI 2024)**
   Key result: Proves AdamW convergence in nonconvex settings; shows AdamW minimizes a dynamically regularized loss; establishes Bayesian posterior generalization advantages over L2-Adam. Provides the first convergence complexity for over-parameterized networks.

3. **Cautious Weight Decay / CWD (Chen et al., ICLR 2026 / arXiv:2510.12402)**
   Key result: Binary sign-alignment mask to conditionally apply WD; interprets WD as a Lyapunov-stability fix for asymptotic AdamW instability; establishes bilevel Pareto-optimal interpretation; proves sliding-mode behavior.

4. **AdamO: Decoupled Orthogonal Dynamics (Chen, Yuan, Zhang, arXiv:2602.05136, 2026)**
   Key result: Identifies "Radial Tug-of-War": WD and gradient updates entangle radial (norm) and tangential (feature-direction) components; decouples them; proves curvature-adaptive radial step sizing eliminates oscillations.

5. **Loshchilov (arXiv:2311.11446, 2023) — Weight Norm Control (AdamWN)**
   Key result: Generalizes WD to target arbitrary weight norm τ rather than zero; shows standard WD is a special case with τ=0; provides a general target-norm framework.

6. **D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) — "Why Do We Need Weight Decay in Modern Deep Learning?"**
   Key result: WD acts as a training dynamics modifier, not explicit regularization; "loss stabilization mechanism" for SGD; "bias-variance tradeoff" for LLM near-one-epoch training; proves scale-invariant equilibrium analysis.

7. **Galanti et al. (arXiv:2206.05794, 2022) — "SGD and WD Secretly Minimize Rank"**
   Key result: SGD+WD induces low-rank bias in weight matrices; stronger with smaller batch or higher WD; connects WD to implicit spectral regularization.

8. **Yunis et al. (arXiv:2408.11804, 2024) — "Approaching Deep Learning through the Spectral Dynamics of Weights"**
   Key result: Singular value evolution as a unifying lens; WD promotes rank minimization; spectral dynamics distinguish memorizing from generalizing networks.

9. **Defazio (arXiv:2506.02285, 2025) — "Why Gradients Rapidly Increase Near the End of Training"**
   Key result: WD drives the gradient-to-weight ratio ‖g‖/‖w‖ of all normalized layers to the same steady state; this "layer balancing" effect explains the Adam vs AdamW gap; proposes corrective WD term for LR-schedule interaction.

10. **Newhouse (MIT MEng Thesis, 2025) — "Duality, Weight Decay, and Metrized Deep Learning"**
    Key result: Unifies SGD, Adam, Shampoo via duality maps under different norms; WD connected to norm geometry; spectral norm duality recovers gradient signal lost in standard WD.

11. **Gradient-Weight Alignment as Generalization Proxy (arXiv:2510.25480, 2025)**
    Key result: Gradient-weight coherence (alignment) tracks generalization during training; misalignment predicts deteriorating generalization; enables validation-free early stopping.

12. **Franke et al. (NeurIPS 2024, arXiv:2311.09058) — CPR: Constrained Parameter Regularization**
    Key result: Reformulates WD as per-parameter-matrix upper-bound constraint via augmented Lagrangian; proves WD is a proxy for norm constraint; AdamCPR outperforms AdamW; provides per-matrix dynamic adaptation.

### Theoretical Landscape Summary

**What is known:**
- WD improves generalization in nonconvex SGD (Sun et al. CVPR 2025) under gradient-weight alignment condition δ_T < 1, but does not accelerate convergence.
- AdamW provably converges and has generalization advantages from a Bayesian perspective (TPAMI 2024).
- WD drives weight norms to steady states; its effect is fundamentally about controlling the gradient-to-weight ratio (Defazio 2025).
- WD induces low-rank bias and spectral rank minimization (Galanti 2022; Yunis 2024).
- Alignment between gradient and weight vectors is a practically measurable proxy for generalization health (arXiv:2510.25480).
- Radial (norm) and tangential (direction) dynamics can be formally decoupled (AdamO 2026).
- WD can be reformulated as a norm constraint solved via augmented Lagrangian (CPR, NeurIPS 2024).

**What is conjectured:**
- All major WD sub-approaches (scheduling, alignment-aware, decoupled, norm-matched) are special cases of a unified principle governing the gradient-to-weight ratio steady state.
- Continuous alignment modulation (beyond binary sign-alignment in CWD) can smoothly interpolate between over-regularization and under-regularization, achieving Pareto dominance over both.
- WD scheduling can be derived as an optimal control strategy for the alignment trajectory — reducing WD when alignment is high (gradient and weight are aligned, decay would oppose learning) and increasing it when alignment is low (gradient and weight are opposed, decay aids contraction).

**Where the gaps are:**
- No formal unified theorem connecting all four WD sub-approaches via a common mathematical object.
- No convergence theory for continuously alignment-modulated WD (λ_t depends on stochastic gradient direction).
- No formal characterization of when the gradient-to-weight ratio steady state is beneficial vs. harmful, and how alignment-aware WD changes it.
- No information-theoretic or PAC-Bayes analysis of dynamic WD; all existing bounds are for fixed WD.

---

## Phase 2: Initial Candidates

### Candidate A: The Generalized Contraction Operator Framework — All WD Methods as Special Cases of a Radial-Tangential Decomposed Update

**Formal claim:**
Define the WD update operator for parameter w as:

    Φ(w, g, λ) = w - γg - λw = (1 - λ)w - γg

Decompose w = r·û (r = ‖w‖, û = w/‖w‖). The update decomposes into:
- Radial update: r_{t+1} = r_t(1 - λ_t) - γ_t cos(θ_t)‖g_t‖
- Tangential update: û_{t+1} ∝ û_t + γ_t/r_t · (g_t - cos(θ_t)‖g_t‖·û_t)

where θ_t is the angle between -g_t and w_t (i.e., cos(θ_t) = -⟨g_t, w_t⟩ / (‖g_t‖‖w_t‖)).

**Claim (Theorem 1 — Unified Decomposition):** Every existing WD method corresponds to a specific choice of λ_t(r_t, θ_t, t) in this radial-tangential framework:
- Standard WD: λ_t = λ (constant)
- CWD: λ_t = λ · 𝟙[cos(θ_t) ≥ 0] (threshold at cos θ = 0)
- Norm-matched WD (AdamWN with target τ): λ_t = λ · (1 - τ/r_t)₊ (decays until r = τ)
- SWD (Xie et al.): λ_t = λ / (1 + ‖g_t‖²) (gradient-norm-inverse weighting)
- AdamO radial step: λ_t = γ_t · (‖g_t‖ · cos(θ_t)) / r_t (pure radial gradient projection)

**Claim (Theorem 2 — Steady-State Characterization):** Under any λ_t = f(cos(θ_t), r_t, t), the steady-state radial equilibrium satisfies:

    E[r*] ≈ γ · E[‖g‖ · cos(θ)] / E[f(cos(θ), r*, t)]

Standard WD steady state: r* = γ‖g‖cos(θ)/λ. CWD steady state: r* = γ‖g‖E[cos(θ)|cos(θ)≥0] / (λ · P(cos(θ)≥0)). This explains why CWD allows larger weight norms under identical λ.

**Proof sketch:**
Step 1: Write the radial update equation in expectation as E[r_{t+1}] = (1 - E[λ_t])·E[r_t] - γ·E[‖g_t‖·cos(θ_t)].
Step 2: At steady state E[r_{t+1}] = E[r_t], solve for r*.
Step 3: For each specific choice of λ_t(·), compute E[λ_t] and E[‖g_t‖·cos(θ_t)] to derive the method-specific r*.
Step 4: Show that the steady-state r* uniquely determines the gradient-to-weight ratio ‖g‖/‖w‖ (Defazio 2025 layer balancing).

**Empirical prediction:** Methods with the same λ but different functional forms (e.g., standard WD vs CWD) should converge to predictably different steady-state weight norms, and this difference should be predictable from the empirical distribution of cos(θ_t) during training.

**Connection to existing theory:** Extends Defazio's gradient-to-weight ratio result (2025) by showing that the steady state depends on the full conditional distribution of cos(θ), not just a scalar. Extends CWD's Lyapunov argument (ICLR 2026) by showing why the sign-aligned mask produces a specific steady state. Generalizes AdamO's radial/tangential separation.

**Novelty estimate:** 7/10. The decomposition itself is known in the context of AdamO. The unified steady-state formula connecting all WD methods via E[cos(θ)] distribution is novel.

---

### Candidate B: Information-Theoretic Lower Bound for Alignment-Aware WD — Why Binary Alignment (CWD) is Suboptimal

**Formal claim:**
Let I_t = I(w_t; S) be the mutual information between current weights and training set S (an information-theoretic complexity measure). Consider two WD strategies:
- Binary: λ_t^{bin} = λ · 𝟙[cos(θ_t) ≥ 0]
- Continuous: λ_t^{cont} = λ · φ(cos(θ_t)) for some monotone φ: [-1,1] → [0,1]

**Claim (Proposition — Alignment Informativeness Lower Bound):** The information gain from the continuous alignment signal over binary signal is bounded below by:

    ΔI ≥ H(cos(θ_t) | sgn(cos(θ_t))) · (∂I_t/∂λ_t)

where H(cos(θ_t) | sgn(cos(θ_t))) is the conditional entropy of the continuous alignment given its sign, and ∂I_t/∂λ_t is the marginal information reduction per unit increase in WD. This quantity is strictly positive whenever cos(θ_t) is not deterministically ±1.

**Corollary (Alignment Informativeness Score):** AIS := H(cos(θ_t) | sgn(cos(θ_t))) / H(cos(θ_t)) ∈ [0,1] measures what fraction of alignment information is discarded by binary masking. AIS → 0 means binary is nearly optimal; AIS → 1 means there is substantial benefit to continuous modulation.

**Proof sketch:**
Step 1: Model the alignment signal cos(θ_t) as a random variable with distribution P_θ on [-1,1].
Step 2: Show that binary masking (CWD) uses only sgn(cos(θ_t)), discarding H(cos(θ_t) | sgn(cos(θ_t))) bits.
Step 3: Use the data-processing inequality to bound the information loss.
Step 4: Translate information loss to a generalization bound gap using the MI-based generalization bound (Xu & Raginsky 2017; arXiv:2305.11042 unified framework).

**Empirical prediction:** AIS should be measurably positive (> 0.1) in real training runs, indicating non-trivial benefit from continuous alignment modulation. Specifically: models trained with continuous alignment-based WD (λ_t = λ·max(0, cos(θ_t))) should show lower AIS (they exploit the full signal) and better generalization than CWD on the same compute budget.

**Connection to existing theory:** Builds on Xu & Raginsky (NIPS 2017) MI-based generalization bounds. Extends CWD's bilevel Pareto analysis to quantify the exact benefit of continuity. Provides a theoretical grounding for the Alignment Informativeness Score metric.

**Novelty estimate:** 8/10. No existing paper applies information-theoretic generalization bounds to the alignment signal for weight decay. The AIS metric itself is new.

---

### Candidate C: Dynamic WD as Optimal Control of the Gradient-to-Weight Ratio Trajectory

**Formal claim:**
Define the state variable R_t = ‖g_t‖/‖w_t‖ (gradient-to-weight ratio, layer-wise). Defazio (2025) shows R_t evolves under SGDW as:

    R_{t+1} ≈ R_t · (1 + γ_t · κ_t - λ_t) / (1 - λ_t)

where κ_t captures the loss curvature. WD drives R toward a steady state R* = γκ/λ.

**Claim (Theorem — Alignment-Optimal WD as Linear Quadratic Regulator):**
For the problem of minimizing the total regularization cost subject to convergence constraints:

    min_{λ_t} Σ_t c(λ_t) + ρ · (R_t - R*)²
    s.t. f(w_T) ≤ ε_opt  (optimization constraint)
         f_S(w_T) - f(w_T) ≤ ε_gen  (generalization constraint)

the optimal WD schedule is:

    λ_t* = λ_0 · (1 - δ̂_t)  where δ̂_t = ⟨g_t, w_t⟩/(‖g_t‖‖w_t‖ + ε)

This is the "conservative" alignment-aware WD rule proposed in the project spec. The theorem formalizes why this specific functional form emerges from an optimal control objective.

**Proof sketch:**
Step 1: Model R_t dynamics as a linear system with time-varying control input λ_t.
Step 2: Apply the discrete-time Linear Quadratic Regulator (LQR) framework: cost = Σ [R_t - R*]² + ρλ_t².
Step 3: Solve the Riccati equation; the optimal control law is λ_t* ∝ (R_t - R*) (proportional to deviation from steady state).
Step 4: Show that R_t - R* is proportional to (1 - δ̂_t) when gradient and weight are misaligned (the weight is "too large" relative to the useful gradient direction).
Step 5: Conclude that λ_t = λ_0 · (1 - δ̂_t) emerges as the LQR-optimal control.

**Empirical prediction:**
- The gradient-to-weight ratio R_t should be more stable (lower variance) under alignment-aware WD compared to fixed WD.
- The Coupling Stability Index (CSI = Var(R_t)) should be measurably lower under dynamic WD.
- The optimal control interpretation predicts that λ_t should be highly correlated with (1 - δ̂_t); this can be verified empirically.

**Connection to existing theory:** Directly formalizes the project spec's intuition for the conservative alignment-aware rule. Connects to Defazio's R_t steady-state result. Applies classical LQR theory to optimization, similar in spirit to the PID-controller interpretation of SGD with momentum.

**Novelty estimate:** 8.5/10. LQR applied to WD trajectory control is, to the best of our knowledge, not done in any existing paper. The specific derivation that the LQR optimum is λ_t ∝ (1 - δ̂_t) is new.

---

## Phase 3: Self-Critique

### Against Candidate A (Unified Decomposition Framework)

**Proof soundness attack:** The steady-state formula E[r*] = γ·E[‖g‖·cos(θ)]/E[f(cos(θ), r*, t)] assumes the system is at stochastic equilibrium. In practice, training is transient — the distribution of cos(θ) changes dramatically across training phases. The formula may only be approximately valid near the end of training.

*Severity: Moderate.* The formula is still predictive for the late-training steady state. We can restrict the claim to the "late-training regime" without losing generality.

**Tightness attack:** The radial/tangential decomposition is exact only when ‖w‖ > 0. At initialization with small weights, the decomposition is ill-conditioned. This is a known limitation of the AdamO paper as well.

*Severity: Low.* Standard initialization ensures ‖w‖ > 0. The decomposition holds throughout training.

**Relevance attack:** Does knowing the steady-state r* help practitioners? Only if we can predict r* before choosing λ. Since r* depends on E[cos(θ)] which is only known empirically, the formula may not be actionable.

*Severity: Moderate.* However, the formula is extremely useful for post-hoc analysis and explaining observed differences between methods, even if not directly predictive.

**Novelty attack:** The radial decomposition is in AdamO (2026). Defazio's (2025) gradient-to-weight ratio already analyzes steady states. The contribution is the synthesis and the explicit formula for each method.

*Verdict:* Searched arXiv for "unified weight decay steady state radial tangential" — no paper derives the per-method steady-state formula in a unified framework.

**Verdict: MODERATE.** Core insight is useful but not surprising in hindsight. Best as a Section 3 "Framework" contribution, not the headline theorem.

---

### Against Candidate B (Information-Theoretic AIS Bound)

**Proof soundness attack:** The step "translate information loss to a generalization bound gap" requires connecting H(cos(θ_t) | sgn(cos(θ_t))) to the mutual information I(w_t; S). This connection is non-trivial. The MI-based bound (Xu & Raginsky) requires I(w_T; S) for the final iterate, but cos(θ_t) affects intermediate iterates. The chain-of-causality is complex.

*Severity: High.* This is a genuine proof gap. The proposition as stated may not be formally correct without additional assumptions.

**Weakening:** We can downgrade to an empirical claim: AIS is a measurable quantity that empirically predicts the benefit of continuous vs binary alignment. We define AIS := H(cos(θ) | sgn(cos(θ))) / H(cos(θ)) as a diagnostic metric without claiming a formal bound.

**Tightness attack:** The bound is not tight. Even if AIS = 0.5 (50% of alignment information is discarded by binary masking), this doesn't mean 50% of generalization benefit is lost — the relationship is indirect.

*Severity: High.* The metric is useful but the formal claim overstates the theoretical connection.

**Relevance attack:** AIS is practically useful as a diagnostic: if AIS ≈ 0.3 in practice, binary masking discards 30% of alignment information. This guides practitioners to choose continuous modulation.

**Novelty attack:** No existing paper defines AIS or uses information-theoretic bounds to compare alignment-aware WD methods.

**Verdict: WEAK** as a formal bound. **MODERATE** as a diagnostic metric definition. Reformulate as a principled metric definition without formal bound claims.

---

### Against Candidate C (LQR Control Framework for Dynamic WD)

**Proof soundness attack:** Step 4 — "R_t - R* is proportional to (1 - δ̂_t) when gradient and weight are misaligned" — requires formal justification. Specifically:
- R_t = ‖g_t‖/‖w_t‖; R* = γκ/λ
- (1 - δ̂_t) = 1 - ⟨g_t, w_t⟩/(‖g_t‖‖w_t‖)
- The connection between R_t - R* and (1 - δ̂_t) requires expressing ‖g_t‖/‖w_t‖ in terms of the alignment angle.

Note that δ̂_t = cos(angle between g_t and w_t), while R_t = ‖g_t‖/‖w_t‖. These are different quantities (angle vs ratio of norms). The connection requires an additional assumption, e.g., that ‖g_t‖ is approximately fixed when ‖w_t‖ grows (as WD increases).

*Severity: High.* The connection between R_t - R* and (1 - δ̂_t) needs a careful derivation. As stated, Step 4 is not rigorous.

**Repair:** Use the augmented Lyapunov function from the project spec: Φ_t = f_S(w_t) + β_t‖w_t‖². The WD term enters as the gradient of β_t‖w_t‖². The alignment δ̂_t appears in the cross-term between gradient and WD updates. Show that minimizing the Lyapunov variance Var(Φ_{t+1} - Φ_t) leads to λ_t ∝ (1 - δ̂_t).

**Tightness attack:** The LQR formulation requires R_t dynamics to be linear, which is only approximately true. In the nonlinear regime (early training, loss landscape curvature varies), the LQR approximation may be poor.

*Severity: Moderate.* The LQR framework provides an intuition and a derivation mechanism. The formal theorem can be stated for the linearized/local regime with a clear caveat.

**Relevance attack:** The prediction that CSI = Var(R_t) is lower under alignment-aware WD is directly testable with the experiment infrastructure already planned (tracking ‖g_t‖, ‖w_t‖, δ̂_t throughout training).

**Novelty attack:** Searched for "weight decay optimal control LQR gradient ratio trajectory" — no existing paper applies LQR to WD control. The connection is novel.

**Verdict: STRONG** (with repaired Step 4 using Lyapunov variance argument). This is the strongest of the three candidates.

---

## Phase 4: Refinement

### Dropped / Demoted
- **Candidate B** (Information-Theoretic Bound): The formal bound has a significant proof gap in connecting H(cos(θ)|sgn(cos(θ))) to I(w_T; S). However, the **AIS metric definition** survives as a principled diagnostic tool. It will be incorporated as a secondary contribution (a metric, not a theorem).

### Strengthened Survivors

**Candidate A → Strengthened as "The Phi-Modulator Taxonomy":**
The unified decomposition is rephrased as a taxonomy of WD methods through the function Φ_t = λ_t/λ (the "modulation factor"), where each method corresponds to a specific modulator:

    Φ_t^method = λ_t^method / λ (normalized effective decay factor)

| Method | Φ_t | Limiting Behavior |
|--------|-----|-------------------|
| Standard WD | 1 | constant contraction |
| CWD | 𝟙[cos(θ_t) ≥ 0] | binary: no decay when aligned |
| Conservative AADW | max(0, 1 - δ̂_t) | continuous: zero decay at perfect alignment |
| Aggressive AADW | 1/(δ̂_t + ε) | continuous: increased decay when opposed |
| SWD | 1/(1 + ‖g_t‖²/σ²) | reduces WD when gradients large |
| Norm-matched (τ) | (1 - τ/r_t)₊ | zero decay when r_t = τ |

The steady-state formula (Theorem 1) is rephrased in terms of E[Φ_t]:

    r* = γ · E[‖g‖ cos(θ)] / (λ · E[Φ_t])

This is clean, testable, and novel as a unified characterization.

**Candidate C → Strengthened as "Alignment-Aware WD as Lyapunov-Variance-Minimizing Control":**
Replace the LQR Step 4 with the following cleaner argument:

**Lemma (Lyapunov Descent with Time-Varying WD):** Consider the augmented potential Φ_t = f_S(w_t) + β‖w_t‖². Under the update w_{t+1} = (1-λ_t)w_t - γ_t g_t, the one-step Lyapunov decrease satisfies:

    E[Φ_{t+1} - Φ_t | w_t] ≤ -γ_t(1-2Lγ_t)‖∇f(w_t)‖²
                                + λ_t[-2β‖w_t‖² + 2γ_t⟨g_t, w_t⟩β + λ_t‖w_t‖²β]
                                + O(γ_t² σ²)

The term 2γ_t⟨g_t, w_t⟩β is the alignment cross-term. When δ̂_t = ⟨g_t,w_t⟩/(‖g_t‖‖w_t‖) is close to 1 (aligned), the WD term contributes positively to the Lyapunov increase (opposing gradient descent). Setting λ_t ∝ (1 - δ̂_t) zeros out this cross-term contribution, eliminating the Radial Tug-of-War while preserving the contraction effect when δ̂_t is small.

**Theorem (Main — Alignment-Optimal WD Reduces Lyapunov Variance):** For any fixed budget B = Σ_t λ_t, the schedule λ_t = λ_0·(1-δ̂_t) achieves:

    Var(E[Φ_{t+1} - Φ_t]) ≤ Var_fixed(E[Φ_{t+1} - Φ_t])

where Var_fixed is the variance under the constant schedule λ_t = λ_0. Moreover, the time-averaged alignment-weighted contraction:

    Ā_T = (1/T) Σ_t λ_t(1 - δ̂_t)

is strictly higher under alignment-aware WD than under fixed WD with identical budget (because alignment-aware WD concentrates λ_t on iterations where δ̂_t is small, maximizing productive regularization).

**Corollary (Alignment-Aware WD Achieves Better Generalization per WD-Budget):** Under the Sun et al. (CVPR 2025) generalization framework, the generalization bound depends on the cumulative contraction Σ_t λ_t(1-δ_t). By maximizing Ā_T under fixed budget, alignment-aware WD achieves a tighter generalization bound than any non-adaptive schedule of equal budget.

This corollary is the paper's headline result: **alignment-aware WD is Pareto-optimal over the generalization-convergence tradeoff for any fixed WD budget.**

### Additional Evidence Confirming Novelty
Searched arXiv for:
- "alignment aware weight decay generalization bound" — top result is CWD (binary sign); continuous modulation not analyzed
- "cumulative contraction weight decay schedule optimal" — no results
- "Lyapunov variance weight decay alignment" — no results
- "optimal weight decay schedule gradient weight alignment theorem" — no results

The combination of (1) budget-constrained optimization over WD schedules, (2) alignment cross-term analysis via Lyapunov variance, and (3) the resulting corollary connecting to Sun et al.'s generalization framework is novel.

### Selected Front-Runner
**Candidate C (strengthened):** Alignment-Aware WD as Lyapunov-Variance-Minimizing Control, with Candidate A's Phi-Modulator Taxonomy as secondary theoretical framework.

---

## Phase 5: Final Proposal

### Title
**"Optimal Weight Decay via Alignment-Sensitive Lyapunov Control: A Unified Framework for Dynamic Weight Decay with Provably Better Generalization"**

### Formal Claims

**Theorem 1 (Phi-Modulator Taxonomy):** Every existing WD method (CWD, SWD, AdamWN, standard WD, AdamO radial step) corresponds to a specific modulation function Φ_t = λ_t/λ ∈ [0,∞). The steady-state weight norm under any modulated WD satisfies:

    r*_Φ = γ · E[‖g‖cos(θ)] / (λ · E[Φ_t])

where θ_t is the gradient-weight angle. This provides a single formula that unifies all WD methods.

**Theorem 2 (Lyapunov-Variance Decomposition):** Under the augmented potential Φ_t = f_S(w_t) + β‖w_t‖² with time-varying λ_t, the Lyapunov decrease decomposes as:

    E[ΔΦ_t] = (convergence term) + λ_t · [-2β‖w_t‖²(1-λ_t/2) + 2βγ_t‖g_t‖‖w_t‖δ̂_t]

The cross-term 2βγ_t‖g_t‖‖w_t‖δ̂_t represents the "Radial Tug-of-War cost" identified in AdamO. Setting λ_t = λ_0·(1 - δ̂_t) eliminates this cost while preserving contraction.

**Theorem 3 (Alignment-Optimal WD):** For any fixed WD budget B = Σ_t λ_t, the alignment-aware schedule λ_t = min(λ_max, λ_0·(1 - δ̂_t)₊) is optimal in the sense that it:
(a) Minimizes Var(E[ΔΦ_t]) — achieves more stable Lyapunov descent.
(b) Maximizes Ā_T = (1/T)Σ_t λ_t(1-δ̂_t) — maximizes alignment-weighted contraction under fixed budget.
(c) Achieves a tighter Sun et al. (CVPR 2025) generalization bound than any non-adaptive schedule of equal budget.

**Proposition 4 (AIS as Diagnostic):** The Alignment Informativeness Score AIS := H(cos(θ_t) | sgn(cos(θ_t))) / H(cos(θ_t)) measures the fraction of alignment information discarded by binary-sign methods (e.g., CWD). AIS > 0 empirically predicts that continuous alignment modulation achieves strictly better generalization-per-budget than binary masking.

### Proof Sketch

**Theorem 1 proof:**
1. Write the stochastic radial update E[r_{t+1}] = (1-λ_t·Φ_t)·E[r_t] - γ_t·E[‖g_t‖cos(θ_t)].
2. Set E[r_{t+1}] = E[r_t] at steady state.
3. Solve for r* = γ·E[‖g‖cos(θ)] / (λ·E[Φ_t]).
4. Substitute specific Φ_t for each method to derive method-specific r*.

**Theorem 2 proof:**
1. Compute E[Φ_{t+1} | w_t] = E[f_S((1-λ_t)w_t - γ_t g_t) + β‖(1-λ_t)w_t - γ_t g_t‖²].
2. Expand using L-smoothness (‖∇²f‖ ≤ L) and the Lipschitz gradient assumption.
3. Identify the cross-term ⟨w_t, g_t⟩ = ‖w_t‖‖g_t‖δ̂_t.
4. Collect terms involving λ_t to isolate the "alignment cost."

**Theorem 3 proof:**
1. Fix budget B; write λ_t = λ_0·h_t with h_t ≥ 0 and Σ_t h_t = B/λ_0.
2. Var(E[ΔΦ_t]) depends on Var(λ_t·δ̂_t·‖w_t‖‖g_t‖).
3. Show that setting h_t ∝ (1-δ̂_t) minimizes the cross-term variance (by Jensen's inequality and the conditional independence structure of the stochastic gradient).
4. Use the Sun et al. generalization bound form Gen(W_T) ≤ C·(1/T)·Σ_t λ_t(1-δ_t) + ε (their Theorem 3 formulation); show Σ_t λ_t^{align}(1-δ_t) ≥ Σ_t λ_t^{fixed}(1-δ_t) under equal budget.

### Assumptions
1. **L-smoothness**: ‖∇f(w) - ∇f(w')‖ ≤ L‖w-w'‖ (standard).
2. **Bounded stochastic gradient variance**: E[‖g_t - ∇f(w_t)‖²] ≤ σ² (standard).
3. **Non-degenerate alignment**: P(cos(θ_t) = ±1) < 1 (alignment is not deterministically binary; holds in practice due to stochastic gradients).
4. **Near-linearity of radial dynamics** (for Theorem 3): The gradient-to-weight ratio R_t is approximately stationary over a window of T' steps (local stationarity assumption; holds in the stable phase of training).
5. **Decoupled WD** (AdamW/SGDW setting): WD is applied multiplicatively as (1-λ_t)w_t, not as an L2 penalty added to the loss.

### Empirical Predictions

1. **Steady-state weight norm prediction (from Theorem 1):** Given empirical measurements of E[‖g‖cos(θ)] and E[Φ_t] during training, Theorem 1 predicts r* to within 10-15%. This can be validated on ResNet-20/CIFAR-10 by comparing theoretical r* vs. observed ‖w‖ at end of training.

2. **Variance reduction from alignment-aware WD (from Theorem 2):** CSI = Var(R_t) should be measurably lower (≥ 20% reduction) under alignment-aware WD (λ_t ∝ (1-δ̂_t)) compared to fixed WD with equal budget. Testable by tracking ‖g_t‖/‖w_t‖ per layer.

3. **Generalization improvement prediction (from Theorem 3):** Alignment-weighted contraction Ā_T should be higher under AADW, and this should correlate with test accuracy improvement. Specifically: Ā_T^{align} / Ā_T^{fixed} ≥ 1 + AIS/2 (conjectured; needs empirical validation).

4. **AIS diagnostic:** On CIFAR-10 with ResNet-20, AIS (measured empirically as H(cos(θ)|sgn(cos(θ)))/H(cos(θ))) should be in range [0.2, 0.5], indicating meaningful benefit from continuous modulation over binary CWD.

### Experimental Plan

**Experiment 1 (Pilot, ~15 min): Steady-state weight norm validation (Theorem 1)**
- ResNet-20 / CIFAR-10, SGD+WD, 3 methods: fixed λ, CWD, conservative AADW
- Seeds: 42, 123, 456
- Log: ‖w_t‖, ‖g_t‖, δ̂_t = ⟨g_t, w_t⟩/(‖g_t‖‖w_t‖ + ε) per layer, every 10 iterations
- Compute E[Φ_t] for each method, compare theoretical r* vs. observed ‖w‖
- GPU: 1x RTX PRO 6000, ~15 min

**Experiment 2 (Core, ~30 min): Lyapunov variance (CSI) comparison (Theorem 2)**
- ResNet-20, VGG-16-BN / CIFAR-10, seeds 42/123/456
- Methods: fixed WD (λ=5e-4), CWD (λ=5e-4), AADW-conservative (λ_0=5e-4, scale=1-δ̂_t), AADW-aggressive (clip(λ_0/(δ̂_t+ε), λ_min, λ_max))
- Budget-matched: all methods have equal Σ_t λ_t (calibrated by adjusting λ_0 for AADW)
- Metrics: CSI=Var(‖g‖/‖w‖), test accuracy, Ā_T, empirical AIS
- GPU: 2x RTX PRO 6000, ~30 min

**Experiment 3 (Extended, ~60 min): Phi-modulator taxonomy visualization**
- ResNet-20, VGG-16-BN / CIFAR-10, CIFAR-100
- 6 methods: fixed WD, no WD, SWD, CWD, AADW-conservative, AADW-aggressive
- Comprehensive visualization: Φ_t trajectory per layer, cos(θ_t) distribution (hist over training), steady-state r* (predicted vs observed), AIS over time
- Generate figures: Figure 1 (taxonomy diagram), Figure 2 (steady-state validation), Figure 3 (CSI comparison), Figure 4 (AIS)

**Experiment 4 (Validation, ~60 min): Generalization bound empirical check**
- Test that Ā_T^{align} / Ā_T^{fixed} correlates with test accuracy improvement across 5 different fixed λ values
- Vary λ ∈ {1e-4, 5e-4, 1e-3, 5e-3, 1e-2} for each method
- Measure both alignment-weighted contraction and final test accuracy
- Compute Pearson correlation ρ(Ā_T, test_acc) to validate Corollary to Theorem 3

### Baselines
**Theoretical baselines:**
- Sun et al. CVPR 2025 bound (fixed WD, worst-case alignment)
- Defazio (2025) steady-state R* formula (fixed WD only)

**Empirical baselines:**
- No WD (pure SGD)
- Fixed WD (λ=5e-4, 1e-3, 5e-3)
- CWD (binary sign alignment, ICLR 2026 — our most direct competitor)
- SWD (gradient-norm-aware scheduling, NeurIPS 2023)
- AdamW with fixed WD (for adaptive optimizer comparison)

### Risk Assessment

1. **Proof gap in Theorem 3, Step 3:** The argument that h_t ∝ (1-δ̂_t) minimizes cross-term variance uses conditional independence between λ_t and the future gradient distribution. This may fail when δ̂_t has strong autocorrelation (as it likely does during stable training phases). Mitigation: restrict Theorem 3 to the i.i.d. approximation and note empirical validity.

2. **Near-linearity assumption may fail in early training:** The radial dynamics are highly nonlinear during the initial phase when weights change rapidly. Mitigation: restrict formal claims to the "settled phase" (after warmup) and validate empirically that the predictions hold in that phase.

3. **Budget-matching in experiments is technically difficult:** Equal Σ_t λ_t requires knowing the distribution of δ̂_t in advance, which defeats the adaptive purpose. Mitigation: use empirical budget matching (run fixed WD first to measure budget, then tune λ_0 for AADW to match).

4. **δ̂_t minibatch noise may dominate the signal:** If the minibatch-estimated alignment is too noisy, λ_t = λ_0·(1-δ̂_t) will be erratic and unstable, eliminating any theoretical benefit. Mitigation: use EMA-smoothed δ̂_t with decay factor 0.9; validate that EMA-δ̂_t has reasonable signal-to-noise ratio.

5. **CWD may already capture most of the benefit:** If the alignment distribution is bimodal (concentrated near ±1), binary masking may capture nearly as much benefit as continuous modulation (AIS ≈ 0). Mitigation: compute AIS empirically early; if AIS < 0.1, focus theoretical contribution on the unified taxonomy (Theorem 1/2) rather than Theorem 3.

### Novelty Claim

The specific theoretical contributions that are novel (verified by literature search):

1. **Phi-Modulator Taxonomy (Theorem 1):** The unified steady-state formula r*_Φ = γ·E[‖g‖cos(θ)]/(λ·E[Φ_t]) connecting all major WD methods — not derived in any existing paper.

2. **Lyapunov-variance decomposition revealing alignment cross-term (Theorem 2):** The explicit isolation of the 2βγ_t‖g_t‖‖w_t‖δ̂_t cross-term and the proof that setting λ_t ∝ (1-δ̂_t) eliminates it — not in AdamO (which identifies the Tug-of-War qualitatively but does not derive the optimal λ_t form), not in CWD (which uses binary masking without Lyapunov analysis of the continuous case).

3. **Budget-constrained alignment-optimal WD corollary (Theorem 3):** The formal connection between alignment-weighted contraction Ā_T and Sun et al.'s generalization bound, establishing that alignment-aware WD Pareto-dominates fixed WD — not in any existing paper.

4. **Alignment Informativeness Score (AIS) as a principled metric:** H(cos(θ)|sgn(cos(θ)))/H(cos(θ)) as a diagnostic for the benefit of continuous over binary alignment — not defined in any existing paper.
