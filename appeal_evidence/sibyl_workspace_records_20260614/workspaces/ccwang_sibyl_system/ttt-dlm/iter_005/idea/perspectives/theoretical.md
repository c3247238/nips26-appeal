# Theoretical Perspective: DLM Inference-Time Compute Scaling via Self-Supervised Weight Adaptation

**Agent**: sibyl-theoretical
**Date**: 2026-03-11
**Topic**: Masked Diffusion Language Model Inference-Time Compute Scaling (ReMask-Retry / TTT / TCR)
**Iteration**: 5 (post-pilot evidence round)

---

## Preamble: Theoretical Landscape and Gaps

Recent theoretical advances have established sharp convergence guarantees for masked diffusion language models (MDLMs). Liang et al. (arXiv 2602.22505) prove tight TV-based convergence rates for both Euler and First-Hitting samplers. Dmitriev et al. (arXiv 2602.15008) show that masking diffusion convergence is governed by *effective total correlation* — an intrinsic information-theoretic quantity that adapts to data structure. A convergence theory by Cai et al. (arXiv 2505.21400) demonstrates O(1/T) KL decay with a coefficient proportional to inter-token mutual information. Jeon et al. (arXiv 2510.24088) establish the I-MDCE relation linking cross-entropy losses to mutual information in masked processes.

However, **none of these theories address what happens when model weights are updated during the denoising process** — the core DaL proposal. The existing convergence guarantees assume a fixed denoiser. Introducing TTT (gradient-based weight updates at test time) fundamentally changes the sampling dynamics: the transition kernel itself evolves across denoising steps. This creates a non-stationary reverse process that falls outside all existing theoretical frameworks.

Meanwhile, on the TTT side, Kuwataka & Suzuki (arXiv 2509.25741) provide the first convergence bounds for TTT combined with in-context learning, showing that TTT enables adaptation to both feature vectors and link functions in single-index models. Von Oswald et al. (arXiv 2506.05233, MesaNet) prove that optimal test-time training via conjugate gradient solves an in-context regression objective, unifying DeltaNet, Mamba, and xLSTM as approximate solutions. Jafari & Anbarjafari (arXiv 2511.21882, Equilibrium Transformers) prove linear convergence for iterative latent refinement under energy-based objectives, explicitly unifying deep equilibrium models, diffusion LMs, and TTT as special cases.

**The critical theoretical gap**: No work provides convergence guarantees for the composition of TTT weight updates with discrete diffusion denoising. The DaL proposal operates in this uncharted intersection.

---

## Proposal 1: TTT-in-Denoising as Online Convex Optimization with Improving Side Information

### Angle: New Theoretical Framework (Online Learning Theory -> DLM Denoising)

### Core Theoretical Insight

The DaL framework — updating fast weights W_f via self-supervised gradients at each denoising step — can be formalized as an **online convex optimization (OCO) problem with monotonically improving side information**. This framing yields provable regret bounds and convergence guarantees that the current proposal lacks.

At each denoising step t (from T down to 0), the learner (TTT layer) faces a loss function:

```
l_t(W) = L_ssl(W; x_t, R_t) = sum_{i in R_t} CE(x_i, f_W(h_i))
```

where R_t is the revealed token set at step t, satisfying R_0 ⊂ R_1 ⊂ ... ⊂ R_T (monotone expansion). The learner updates W via gradient descent:

```
W_{t+1} = W_t - eta * grad l_t(W_t)
```

**Key structural property**: Unlike standard OCO where the adversary can choose arbitrary loss functions, in DLM denoising the loss functions {l_t} have a **monotone refinement structure** — each subsequent loss is defined over a *superset* of the data used for the previous loss. This is strictly more favorable than the adversarial OCO setting.

### Theorem Sketch 1: Regret Bound for TTT-in-Denoising

**Setup**: Let W* = argmin_W sum_t l_t(W) be the offline optimal fast weight. Assume l_t is L-Lipschitz and beta-smooth. The TTT layer uses gradient descent with step size eta = 1/(beta * sqrt(T)).

**Claim**: The regret of TTT-in-Denoising satisfies:

```
R(T) = sum_t [l_t(W_t) - l_t(W*)] <= O(sqrt(T) * ||W_0 - W*||^2)
```

But crucially, due to the monotone refinement structure of {R_t}, the **effective regret** (measured against the per-step best response) is tighter:

```
R_eff(T) <= O(sqrt(T)) - Omega(sum_t I(X_{R_t \ R_{t-1}}; X_{M_t} | X_{R_{t-1}}))
```

where the second term captures the **information gain from newly revealed tokens** — this is positive and grows as R_t expands, meaning the TTT learner benefits from the progressively enriched supervision.

**Significance**: This formalizes the intuition that "denoising is learning" — the progressive revelation of tokens provides a curriculum that improves the quality of gradient updates. Standard OCO regret bounds do not capture this structure.

### Theorem Sketch 2: Convergence of the Non-Stationary Reverse Process

The standard DLM convergence theory (Liang et al., 2026) assumes a fixed score function s_theta. With TTT, the score function changes at each step: s_{theta, W_t}. The sampling error decomposes as:

```
KL(p_data || p_sample) <= KL_fixed(T) + KL_drift(T)
```

where KL_fixed is the standard sampling error (O(1/T) from existing theory) and KL_drift captures the effect of weight changes:

```
KL_drift(T) <= sum_t E[||s_{theta,W_{t+1}} - s_{theta,W_t}||^2] * Delta_t
```

If TTT updates are small (||W_{t+1} - W_t|| = O(eta)), and the score function is L_W-Lipschitz in W, then:

```
KL_drift(T) <= T * eta^2 * L_W^2 * max_t ||grad l_t(W_t)||^2
```

**Trade-off**: Larger eta => faster learning (lower l_t) but larger KL_drift. Optimal eta balances these:

```
eta* = O(1 / (L_W * T^{1/2} * G))   where G = max gradient norm
```

This provides the first theoretical justification for why **small learning rates** are necessary for TTT in denoising — too-fast weight updates destabilize the reverse process.

### Connection to P3 Failure

The pilot P3 used lr=0.001 with meta-training. The gate stuck at 0.007 means the effective perturbation to the score function was O(0.007), making KL_drift negligible but also making the TTT contribution negligible. The theoretical prediction: **there exists a non-trivial optimal gate value** that balances learning speed against sampling stability. This is precisely what H_gate aims to find empirically.

### Failure Modes

- The beta-smoothness assumption may not hold for Transformer hidden states (attention creates sharp loss landscapes)
- The monotone refinement structure assumes no remasking; with ReMDM-style remasking, R_t is no longer monotone, breaking the information-gain bonus
- The Lipschitz-in-W assumption for the score function may be violated near phase transitions

### Computational Cost

Pure theory — no GPU cost for the framework itself. Empirical validation of the regret bound (measuring cumulative SSL loss vs optimal): ~2 GPU-hours as a diagnostic add-on to existing experiments.

### Key References

- Hazan (2016). "Introduction to Online Convex Optimization." Foundations and Trends in Optimization.
- Liang et al. (2026). "Sharp Convergence Rates for Masked Diffusion Models." arXiv 2602.22505.
- Dmitriev et al. (2026). "Efficient Sampling with Discrete Diffusion Models: Sharp and Adaptive Guarantees." arXiv 2602.15008.
- Cai et al. (2025). "A Convergence Theory for Diffusion Language Models." arXiv 2505.21400.
- Kuwataka & Suzuki (2025). "Test time training enhances in-context learning of nonlinear functions." arXiv 2509.25741.

---

## Proposal 2: Natural Gradient Interpretation and Fisher-Efficient TTT for DLMs

### Angle: Improve Existing (Information Geometry -> Better TTT Updates)

### Core Theoretical Insight

The current DaL proposal includes "precision-weighted updates" as an enhancement (Section 2.1 of proposal.md), describing it as "a diagonal approximation to natural gradient descent." This theoretical proposal formalizes this claim and derives the **exact conditions** under which precision weighting is optimal, as well as when it fails.

### Fisher Information Matrix for DLM Denoising

At denoising step t, the TTT layer parameterizes a conditional distribution p_W(x_i | h_i) for each revealed position i in R_t. The Fisher Information Matrix (FIM) of the self-supervised loss is:

```
F(W) = E_{x~p_data} [grad_W log p_W(X|h) * grad_W log p_W(X|h)^T]
```

The natural gradient update (Amari, 1998) is:

```
W_{t+1} = W_t - eta * F(W_t)^{-1} * grad l_t(W_t)
```

**Key observation**: For the TTT-MLP architecture (d_model -> d_ttt -> d_model with d_ttt = d_model/8), the FIM has a specific block structure determined by the MLP topology. The full FIM inversion has cost O(d_ttt^3), which is O((d_model/8)^3) — potentially feasible for small TTT dimensions but expensive for every denoising step.

### Precision Weighting as Diagonal Fisher Approximation

The proposal's precision weighting (pi_i = 1/Var[p(x_i|x_t)]) corresponds to a **per-position diagonal approximation** of the FIM. Formally:

```
F_diag = diag(pi_1, pi_2, ..., pi_{|R_t|}) ⊗ I_{d_ttt}
```

This approximation is exact when:
1. Token predictions at different positions are independent (no cross-position correlations in the Fisher)
2. The gradient structure factorizes across positions

**When this fails**: In Transformers with self-attention, token predictions are NOT independent — the attention mechanism creates strong cross-position correlations. The off-diagonal blocks of the Fisher (capturing correlations between gradients at positions i and j) can be large, especially for tokens in close syntactic relationship.

### Theorem Sketch 3: Convergence Rate Improvement from Natural Gradient

Under standard assumptions (L-smooth, mu-strongly convex loss):

- Vanilla gradient descent converges as: ||W_t - W*|| <= (1 - mu/L)^t * ||W_0 - W*||
- Natural gradient converges as: ||W_t - W*||_F <= (1 - 1/kappa_F)^t * ||W_0 - W*||_F

where kappa_F = lambda_max(F) / lambda_min(F) is the condition number of the FIM.

**Critical insight**: For DLM denoising, the FIM condition number **varies with mask ratio**:
- At high mask ratio (r > 0.7): Few revealed tokens, FIM is ill-conditioned (kappa_F >> 1), gradients are noisy => vanilla and natural gradient both struggle
- At low mask ratio (r < 0.3): Many revealed tokens, FIM is well-conditioned (kappa_F ~ 1), vanilla gradient is already near-optimal => natural gradient overhead is wasted
- At critical zone (r in [0.3, 0.7]): Moderate tokens, FIM has meaningful structure, natural gradient provides genuine speedup

**This independently derives the phase-transition scheduling** observed empirically in P2 (SNR peaks at r=0.6). The FIM condition number provides a principled explanation: gradient updates are most informative when the Fisher is well-conditioned but not trivially diagonal.

### Practical Algorithm: K-FAC TTT

Instead of the full FIM inverse or the crude diagonal approximation, use **Kronecker-factored approximate curvature (K-FAC)** for the TTT layer:

```
F ≈ A ⊗ G
```

where A = E[a * a^T] (input activation covariance) and G = E[g * g^T] (gradient covariance). For the small TTT-MLP (d_ttt = d_model/8), the K-FAC inversion costs O(d_ttt^2) per step — modest overhead.

**Hypothesis H_KFAC**: K-FAC TTT updates achieve >= 95% of full natural gradient performance on GSM8K while adding only ~5% computational overhead vs vanilla TTT.

### Connection to SSL-Task Alignment (H_align)

The principal-agent misalignment between L_ssl and task accuracy can be analyzed through the lens of the FIM. If the Fisher of the SSL objective (MLM on revealed tokens) is nearly orthogonal to the Fisher of the task loss (downstream accuracy), then gradient descent on L_ssl moves W in directions that are uninformative for the task. Formally:

```
cos(angle(F_ssl * grad L_ssl, F_task * grad L_task))
```

If this cosine is near zero, precision weighting (or any variant of natural gradient on L_ssl) cannot help — the problem is not the gradient *magnitude* but the gradient *direction*. This provides a theoretical prediction for D0c: if the alignment cosine is low, DaL requires a fundamentally different self-supervised objective, not just better optimization.

### Failure Modes

- K-FAC approximation may be poor for attention-based TTT architectures (the Kronecker structure assumes independent layers)
- The FIM changes at every denoising step, requiring re-estimation (amortization strategies needed)
- The strong-convexity assumption is unlikely to hold globally for the TTT-MLP loss

### Computational Cost

- K-FAC TTT implementation: ~1 day engineering
- K-FAC vs diagonal vs vanilla ablation: ~4-6 GPU-hours
- FIM condition number profiling across mask ratios: ~2 GPU-hours (diagnostic)
- **Success probability: 35%** for K-FAC providing meaningful improvement over diagonal precision weighting

### Key References

- Amari (1998). "Natural Gradient Works Efficiently in Learning." Neural Computation.
- Martens & Grosse (2015). "Optimizing Neural Networks with Kronecker-Factored Approximate Curvature." ICML.
- Jeon et al. (2025). "Information-Theoretic Discrete Diffusion." arXiv 2510.24088.
- Shwartz-Ziv & LeCun (2024). "To Compress or Not to Compress — Self-Supervised Learning and Information Theory: A Review." Entropy.

---

## Proposal 3: Information-Theoretic Bounds on DaL's Maximum Achievable Improvement

### Angle: New Theoretical Framework (Information Theory -> Fundamental Limits)

### Core Theoretical Insight

Before investing further GPU hours in DaL engineering, we should ask: **what is the theoretical maximum improvement that cross-step memory can provide?** This proposal derives information-theoretic upper bounds on the benefit of any cross-step information transfer mechanism (including TTT, GRU, Hopfield, or any future method).

### The Information Island Quantification

At each denoising step t, the standard DLM conditions on the masked sequence x_t. The mutual information between the model's prediction and the target is:

```
I_standard(t) = I(Y_t; X | x_t)
```

where Y_t is the model's prediction and X is the true sequence. With perfect cross-step memory (oracle access to all previous hidden states), the upper bound becomes:

```
I_memory(t) = I(Y_t; X | x_t, h_{T:t+1})
```

where h_{T:t+1} are the hidden states from all previous denoising steps.

**The Information Island Gap**:

```
Delta_I(t) = I_memory(t) - I_standard(t) = I(Y_t; h_{T:t+1} | x_t, X_{R_t})
```

This quantifies how much additional information the hidden states carry beyond what is already available in the current masked sequence. If Delta_I(t) is small for most t, then **no cross-step memory mechanism can significantly help** — the information island problem is mild, and the standard DLM already captures most available signal.

### Theorem Sketch 4: Data Processing Inequality Bound

By the data processing inequality:

```
Delta_I(t) <= I(h_{T:t+1}; X_{M_t} | X_{R_t})
```

The right-hand side is the mutual information between previous hidden states and the *currently masked* tokens, given the *revealed* tokens. This is bounded by:

```
I(h_{T:t+1}; X_{M_t} | X_{R_t}) <= H(X_{M_t} | X_{R_t}) - H(X_{M_t} | X_{R_t}, h_{T:t+1})
```

The first term is the conditional entropy of masked tokens given revealed ones (computable from the data distribution). The second term depends on how much the hidden states reduce uncertainty about masked tokens.

**Key prediction**: For a well-trained DLM backbone (Dream-7B), the backbone hidden states h_t already capture most of the relevant information about X_{M_t} from x_t in a *single forward pass*. The additional information from previous steps' hidden states h_{T:t+1} may be small because:

1. The backbone is a powerful Transformer that computes rich representations from x_t alone
2. Previous steps' hidden states were computed from *noisier* versions of the sequence (more masks)
3. The stochastic unmasking process adds noise that degrades the information in h_{T:t+1}

This provides a formal basis for the contrarian's concern that the "information island" problem may be mild for powerful DLM backbones.

### Empirical Estimation Protocol

The bounds above can be estimated empirically using the variational information bottleneck:

1. Train a small probe network to predict X_{M_t} from h_{T:t+1} (measuring how much info is in cross-step states)
2. Compare with a probe that predicts X_{M_t} from h_t alone (measuring how much info is in current-step state)
3. The gap = empirical estimate of Delta_I(t)

**Diagnostic D0d (NEW)**: Measure Delta_I(t) at mask ratios r in {0.1, 0.3, 0.5, 0.7, 0.9}. If Delta_I(t) < 0.1 bits per token at all mask ratios, the theoretical ceiling for any cross-step memory method is extremely low, and DaL (or any variant) cannot achieve more than marginal improvement.

### Connection to Pilot Evidence

- **P3 failure context**: Even if the gate had opened, the maximum possible improvement is bounded by Delta_I. If Delta_I is small, gate repair cannot save DaL.
- **MetaState underperformance**: MetaState GRU at 43.75% vs vanilla 50.0% (n=16) could be explained by Delta_I being small + the GRU module adding noise that exceeds its information contribution.
- **Existing convergence theory**: Dmitriev et al. (2602.15008) show convergence depends on "effective total correlation" — a related quantity that captures data structure. Low total correlation implies fast convergence of vanilla DLM, leaving less room for TTT improvement.

### Why This Matters for the Overall Research Direction

If Delta_I is proven to be small, the entire DaL/MetaState paradigm is fundamentally limited, and the project should pivot to:
- Training-free methods that improve the *denoising schedule* (Alternative A) rather than adding memory
- Methods that improve the *backbone* itself (fine-tuning, RL alignment) rather than adding lightweight modules
- The diagnostic study paper (Alternative D) explaining *why* cross-step memory provides limited value

If Delta_I is non-trivial (> 0.5 bits per token in the critical zone), then there is headroom for DaL, and the engineering challenges (gate repair, SSL-task alignment) are worth solving.

### Failure Modes

- The variational bound may be loose (the probe network may not extract all available information)
- Delta_I may be distribution-dependent: high for reasoning tasks (where intermediate steps carry chain-of-thought information) but low for text generation (where local context suffices)
- The bound applies to *information content*, not to *algorithmic efficiency* — there may be scenarios where the information exists but no practical algorithm can extract it efficiently

### Computational Cost

- Probe training for Delta_I estimation: ~4-6 GPU-hours
- Analysis across mask ratios and task types: ~2-4 GPU-hours
- Total: ~8 GPU-hours
- **Success probability**: 80% for producing a clear upper bound (the analysis itself is low-risk; the risk is that the bound is too loose to be informative)

### Key References

- Cover & Thomas (2006). "Elements of Information Theory." Wiley.
- Dmitriev et al. (2026). "Efficient Sampling with Discrete Diffusion Models." arXiv 2602.15008.
- Huang et al. (2025). "On the epsilon-Free Inference Complexity of Absorbing Discrete Diffusion." arXiv 2509.21835.
- Conforti et al. (2025). "Non-Asymptotic Convergence of Discrete Diffusion Models." arXiv 2512.00580.
- Liang et al. (2025). "Absorb and Converge: Provable Convergence Guarantee for Absorbing Discrete Diffusion Models." arXiv 2506.02318.

---

## Proposal 4: Expressiveness Separation — TTT-MLP vs GRU as Cross-Step Update Rules

### Angle: Improve Existing (Computational Learning Theory -> Update Rule Comparison)

### Core Theoretical Insight

The core empirical question of DaL — "does TTT-MLP outperform GRU as a cross-step update rule?" (H1) — has a theoretical counterpart: **are there function classes that TTT-MLP can express but GRU cannot, and are these classes relevant to DLM denoising?**

### Expressiveness Hierarchy

The update rules form a strict expressiveness hierarchy:

```
Fixed averaging ⊂ Linear attention ≡ TTT-Linear ⊂ GRU ⊂ TTT-MLP ⊂ Full attention
```

**TTT-Linear ≡ Linear attention**: This equivalence was established by Liu et al. (arXiv 2602.21204). TTT with a linear self-supervised model is mathematically identical to linear attention with learned update rules.

**GRU ⊂ TTT-MLP**: GRU applies a fixed affine transformation gated by sigmoid. TTT-MLP applies gradient descent on an arbitrary loss surface — strictly more expressive because:

1. GRU gates are bounded in [0,1] (sigmoid); TTT-MLP updates have unbounded magnitude (dependent on loss landscape curvature)
2. GRU performs a single update per step; TTT-MLP can perform K inner gradient steps, approximating arbitrary optimization trajectories
3. GRU cannot represent non-monotone update dynamics; TTT-MLP can (via non-convex loss surfaces)

**Formal statement (Theorem Sketch 5)**: There exist sequence-to-sequence tasks computable by a TTT-MLP layer (with d_ttt = d) in T denoising steps that require a GRU with state dimension Omega(d^2) to approximate within epsilon error.

**Proof sketch**: Construct a task where the optimal cross-step update depends on the *second-order statistics* of revealed tokens (covariance structure). TTT-MLP naturally captures this via the gradient of the SSL loss (which depends on the Hessian of the loss w.r.t. the data). GRU, operating with first-order gate-update operations, requires quadratically more state dimensions to represent second-order statistics.

### Relevance to DLM Denoising

The separation theorem is meaningful for DLM denoising if the optimal denoising strategy requires second-order adaptation. Evidence for this:

1. **P2 (phase transition)**: The signal quality varies nonlinearly with mask ratio — optimal updates should depend on the curvature of the loss landscape, not just the gradient direction
2. **Reasoning tasks (GSM8K)**: Mathematical reasoning involves tracking variable bindings and arithmetic constraints — second-order dependencies between token predictions
3. **Predictive coding interpretation**: Precision-weighted prediction errors are second-order quantities (involving variance estimation)

### Hypothesis H_separation

**Statement**: On reasoning benchmarks (GSM8K, ARC-C), the TTT-MLP vs GRU performance gap is statistically significant (p < 0.05) and the gap increases with problem complexity (number of reasoning steps). On text generation (PPL), the gap is not significant.

**Rationale**: The expressiveness separation matters most for tasks requiring second-order adaptation, which correlates with reasoning complexity.

### Experimental Test

Measure TTT-MLP vs GRU performance stratified by:
- Number of reasoning steps in GSM8K solutions (1-2 steps vs 5+ steps)
- Problem difficulty (ARC-C easy vs hard subsets)
- Sequence length

If the separation prediction holds, the gap should widen for harder/longer problems.

### Failure Modes

- The theoretical separation may exist but the practical gap may be tiny (the relevant function class may be well-approximated by GRU)
- The meta-training procedure may not find the TTT-MLP configurations that exploit the expressiveness gap
- GRU's inductive bias (bounded updates via sigmoid) may actually be beneficial for stability, compensating for the expressiveness deficit

### Computational Cost

Theory only — piggybacks on the existing H1 ablation. Additional analysis (stratification by difficulty) costs ~1-2 GPU-hours.

### Key References

- Liu et al. (2026). "TTT-Linear Attention Equivalence." arXiv 2602.21204.
- Sun et al. (2024). "Learning to (Learn at Test Time)." arXiv 2407.04620.
- Von Oswald et al. (2025). "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training." arXiv 2506.05233.
- Merrill & Sabharwal (2023). "The Expressive Power of Transformers with Chain of Thought." ICLR 2024.

---

## Proposal 5: Lyapunov Stability Analysis for TTT-in-Denoising Dynamics

### Angle: Cross-Domain Transfer (Control Theory -> DLM Denoising Stability)

### Core Theoretical Insight

The DaL system is a **coupled dynamical system**: the denoising process evolves the token sequence x_t, while the TTT updates evolve the fast weights W_t. Stability of this coupled system is not guaranteed — fast weight updates could destabilize the denoising trajectory, causing oscillation or divergence. The interdisciplinary perspective (from the proposal) already noted the risk of "integrator windup" from momentum accumulation across sequences. A formal Lyapunov analysis can characterize when the coupled system is stable.

### Lyapunov Function for the Coupled System

Define the Lyapunov function:

```
V(x_t, W_t) = alpha * KL(p_data || p_{theta,W_t}(x_t)) + (1-alpha) * ||W_t - W*_t||^2_F
```

where p_{theta,W_t} is the model's distribution with current fast weights and W*_t is the per-step optimal fast weight.

**Stability condition**: The coupled system is stable (V decreases monotonically) if:

```
dV/dt = alpha * d/dt KL(...) + (1-alpha) * d/dt ||W_t - W*_t||^2 < 0
```

The first term decreases if the denoising process converges (guaranteed by DLM theory for fixed weights). The second term decreases if TTT converges (guaranteed by gradient descent on a smooth loss). The coupled term — the *cross-interaction* between denoising and TTT — is the critical stability concern:

```
Cross-interaction = alpha * <grad_W KL, dW/dt> = -alpha * eta * <grad_W KL, grad_W L_ssl>
```

**Stability sufficient condition**: The cross-interaction is negative (stabilizing) if:

```
<grad_W KL, grad_W L_ssl> > 0
```

i.e., the SSL gradient direction is positively correlated with the direction that reduces KL divergence to the data distribution. This is precisely the **SSL-task alignment** condition (H_align), now given a formal stability interpretation.

### Phase-Dependent Stability

The stability condition may hold in some denoising phases but not others:

- **Early phase (r > 0.7)**: Few revealed tokens, L_ssl gradient is noisy, <grad_W KL, grad_W L_ssl> may be negative (destabilizing) => skip TTT updates
- **Critical phase (r in [0.3, 0.7])**: Good signal quality, positive correlation expected => apply TTT updates
- **Late phase (r < 0.3)**: Most tokens revealed, both gradients well-aligned, but diminishing returns => optional TTT updates

This provides a **control-theoretic derivation of the phase-transition scheduling**, complementing the empirical evidence from P2 and the information-geometric analysis from Proposal 2.

### Gate as Stability Controller

The residual gate beta in DaL (output = backbone + beta * TTT_output) acts as a **gain parameter** in the control system. The stability radius is:

```
beta_max = min(1, 1 / (L_W * eta * G_max))
```

where L_W is the Lipschitz constant of the score w.r.t. W and G_max is the maximum gradient norm. If beta exceeds this threshold, the coupled system may diverge.

**P3 diagnosis**: Gate stuck at 0.007 << beta_max means the system was trivially stable (the TTT perturbation was negligible). The gate repair (sigmoid(-2) = 0.12) must be checked against beta_max to ensure we do not overshoot into instability. The optimal gate balances expressiveness and stability:

```
beta_optimal ≈ beta_max / sqrt(2)    (analogous to critical damping)
```

### Practical Implication: Adaptive Gate Scheduling

Rather than a fixed gate, use an **adaptive gate** that tracks the stability condition:

```
beta_t = min(beta_max_t, beta_target) where beta_max_t = 1 / (L_W * eta * ||grad l_t||)
```

This ensures stability at every denoising step while maximizing the TTT contribution. The implementation requires only computing the gradient norm (already available) and an estimate of L_W (can be tracked with running average).

### Failure Modes

- Lyapunov analysis assumes continuous dynamics; the actual system is discrete (denoising steps are finite)
- The cross-interaction term depends on the unknown data distribution p_data
- Estimating L_W in practice may be inaccurate, leading to conservative or aggressive gates
- The Lyapunov function choice is not unique; different V may yield different stability conditions

### Computational Cost

Pure theory + lightweight diagnostic. Tracking beta_max_t during training adds negligible overhead (~0.01x).

### Key References

- Khalil (2002). "Nonlinear Systems." Prentice Hall.
- Jafari & Anbarjafari (2025). "Closed-Loop Transformers: Autoregressive Modeling as Iterative Latent Equilibrium." arXiv 2511.21882.
- He et al. (2025). "What Exactly Does Guidance Do in Masked Discrete Diffusion Models." arXiv 2506.10971.

---

## Comparative Analysis and Recommendation

| Proposal | Theoretical Depth | Practical Value | Risk | Validates DaL? | Novelty |
|----------|------------------|----------------|------|----------------|---------|
| **1. OCO Regret Bounds** | Very High | Medium (explains phase scheduling) | Low (theory) | Yes (provides guarantees) | High |
| **2. Natural Gradient / K-FAC** | High | High (better optimizer) | Medium | Yes (improves updates) | Medium |
| **3. Information-Theoretic Bounds** | Very High | Critical (determines feasibility ceiling) | Low (diagnostic) | Can falsify DaL | Very High |
| **4. Expressiveness Separation** | High | Medium (justifies TTT-MLP over GRU) | Low (piggybacks on H1) | Yes (core hypothesis) | Medium |
| **5. Lyapunov Stability** | High | High (adaptive gate) | Medium | Yes (stability guarantee) | High |

### Theorist's Recommendation: Priority-Ordered Theory Agenda

**HIGHEST PRIORITY — Proposal 3 (Information-Theoretic Bounds)**

Before any further engineering effort, we must answer: **is there theoretical headroom for cross-step memory in DLMs?** If Delta_I is small, all DaL variants are doomed regardless of gate repair, optimizer choice, or expressiveness advantages. The D0d diagnostic (Delta_I estimation) should run *before or in parallel with* D0c (SSL-task alignment).

**Decision tree**:
- Delta_I > 0.5 bits/token in critical zone => Cross-step memory has significant value. Proceed with DaL + Proposals 1, 2, 4, 5 as the theoretical framework.
- Delta_I in [0.1, 0.5] => Modest headroom. DaL may achieve 1-3% improvement at best. Consider whether this justifies the engineering complexity vs Alternative A.
- Delta_I < 0.1 => The information island problem is theoretically mild. Pivot to Alternative D (diagnostic study) and frame the upper bound as a core contribution.

**SECOND PRIORITY — Proposal 1 (OCO Framework) + Proposal 5 (Lyapunov Stability)**

These two proposals together provide the theoretical backbone for the DaL paper:
- OCO framework justifies phase-transition scheduling and provides regret guarantees
- Lyapunov analysis justifies the gate mechanism and provides stability guarantees
- Together, they transform DaL from "we tried TTT in DLMs and it worked" to "we provide a principled framework for when and how to adapt weights during denoising, with provable guarantees"

**THIRD PRIORITY — Proposal 2 (K-FAC TTT) + Proposal 4 (Expressiveness Separation)**

These provide refinements:
- K-FAC is a practical improvement with theoretical justification (ablation contribution)
- Expressiveness separation provides theoretical grounding for why TTT-MLP > GRU (supports H1)

### Integration into the Paper

If DaL succeeds empirically, the theory section (Section 4 of proposal.md) should include:

1. **Information-theoretic ceiling** (Proposal 3): Upper bound on improvement, showing DaL approaches this ceiling
2. **OCO regret bound** (Proposal 1): Formal guarantee that TTT-in-denoising converges, with rate depending on information gain
3. **Stability analysis** (Proposal 5): Conditions for stable coupling of TTT and denoising, justifying phase scheduling and gate design
4. **Expressiveness separation** (Proposal 4): Formal argument for TTT-MLP > GRU, supported by stratified empirical evidence

If DaL fails empirically, the theory still contributes to Alternative D (diagnostic study):

1. **Information-theoretic ceiling** shows why cross-step memory has limited value
2. **Stability analysis** explains why naively coupling TTT with denoising is unstable
3. **The theoretical framework itself** is a contribution — it provides tools for analyzing any future attempt at adaptive weight updates during diffusion sampling

---

## Lessons Applied from Evolution History

1. **"Contrarian was right about SSL-task misalignment"**: Proposal 3 provides the theoretical framework to determine if this misalignment is structural (information-theoretic) or incidental (engineering). The Delta_I bound is a necessary condition check before committing to DaL.

2. **"First claims need verification"**: All theorem statements are labeled "Theorem Sketch" — these are conjectures with proof outlines, not proven theorems. The paper would need to either prove these formally or present them as conjectured bounds with empirical validation.

3. **"Overestimating DaL's potential"**: Proposal 3 explicitly addresses this by providing an *upper* bound on improvement. The theoretical perspective's most important contribution may be showing that the ceiling is low — an honest negative result grounded in information theory.
