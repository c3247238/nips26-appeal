# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **Li & Cai (2025), "Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models"** (arXiv:2505.21400, NeurIPS 2025) -- Establishes the first convergence theory for DLM sampling. The KL-divergence sampling error decays as O(MI/T) where MI is the mutual information between tokens and T is the number of denoising iterations. Critically, they prove matching upper and lower bounds (up to constants), showing this rate is tight. The result covers the regime T < L (fewer iterations than sequence length), rigorously justifying parallel acceleration.

2. **Zhao & Cai (2026), "Adaptation to Intrinsic Dependence in Diffusion Language Models"** (arXiv:2602.20126) -- Introduces a distribution-agnostic randomized unmasking schedule that adapts to the unknown dependence structure. Convergence guarantees scale as O-tilde(TC/K) and O-tilde(DTC/K), where TC and DTC are the total correlation and dual total correlation. These are intrinsic information-theoretic measures, making the bounds adaptive without hyperparameter tuning.

3. **Jeon & Shin (2025), "Information-Theoretic Discrete Diffusion"** (NeurIPS 2025) -- Establishes the I-MDSE (Information-Minimum Denoising Score Entropy) identity and the I-MDCE relation for masked diffusion, showing that score-based losses are exact mutual information decompositions, not merely variational bounds. This makes likelihood estimation tight and efficient (single forward pass per sample).

4. **Diffusion Information Geometry (DIG)** (October 2025) -- Interprets DLM denoising as gradient flow on the statistical manifold of token distributions under the Fisher-Rao metric. Each diffusion step corresponds to natural gradient descent on an intrinsic information potential (KL divergence). Derives convergence theorems with exponential energy dissipation at rate e^{-2(lambda + |kappa|/2)t}, where kappa is the sectional curvature of the token manifold.

5. **Absorb and Converge** (arXiv:2506.02318, 2025) -- Provides provable convergence guarantees specifically for absorbing (masked) discrete diffusion models using a KL decomposition into forward conditional entropy and cross-entropy components, applicable to non-symmetric rate matrices.

6. **Limits of KV Cache Compression** (arXiv:2503.11108, March 2025) -- Proves information-theoretic lower bounds on KV cache compression for tensor attention: even with (1 +/- eta) multiplicative approximation, memory must be Omega(nd) -- asymptotically the same as exact computation. Uses communication complexity reductions. This result fundamentally constrains what approximate KV caching can achieve.

7. **Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits** (ICLR 2025 Spotlight) -- Shows optimal acceptance probability in multi-draft speculative decoding can be decomposed into importance sampling + single-draft acceptance. Establishes necessary and sufficient conditions for acceptance probability to equal 1. Connects speculative decoding theory to optimal transport and b-matching problems.

8. **Masked Diffusion Models are Secretly Time-Agnostic Masked Models** (ICLR 2025) -- Reveals that MDM training and sampling are theoretically free from the time variable, equivalent to time-agnostic masked models. The first-hitting sampler (FHS) achieves 20x speedup while remaining theoretically equivalent to the original process.

9. **D3PM: Structured Denoising Diffusion in Discrete State-Spaces** (Austin et al., NeurIPS 2021) -- Foundational work establishing mutual-information-based noise scheduling for discrete diffusion. Demonstrated 20x speedup over comparable AR transformers using a mutual information schedule.

10. **Bottlenecked Transformers** (arXiv:2505.16950, May 2025) -- Connects Information Bottleneck (IB) theory to KV cache compression. Argues that periodic KV consolidation creates an optimal compression-retention balance. RNNs impose a hard bottleneck via fixed-size state; transformers' ever-growing cache removes this constraint entirely.

11. **Conditional [MASK] Discrete Diffusion Language Model** (EMNLP 2025) -- Proposes Entropy-based Noise Scheduling (ENS), deciding which tokens to denoise at each timestep based on position entropy H(x_i). Connects information-theoretic quantities to practical scheduling decisions.

12. **On Error Propagation of Diffusion Models** (arXiv:2308.05021) -- Develops a theoretical framework for error propagation in sequential diffusion architectures, decomposing into modular error, cumulative error, and propagation equation. Derives upper bounds on cumulative error and designs bootstrap algorithms for efficient estimation.

### Theoretical Landscape Summary

The theoretical understanding of DLM inference sits at the intersection of several well-developed mathematical frameworks:

**What is known:**
- The sampling error of DLMs decays as O(MI/T) where MI captures inter-token dependence (mutual information, total correlation, dual total correlation). This is tight -- matching lower bounds exist.
- Discrete score-based losses are exact mutual information decompositions (not variational bounds).
- The denoising process can be understood as natural gradient flow on a statistical manifold, with convergence governed by Fisher-Rao curvature.
- KV cache compression faces fundamental Omega(nd) memory lower bounds even with multiplicative approximation tolerance.
- Speculative decoding acceptance probability is characterized by optimal transport / b-matching theory.

**What is conjectured but unproven:**
- Whether the O(MI/T) bound remains tight when KV caching introduces systematic approximation errors at each step (error propagation through composition).
- Whether different acceleration families (caching, early stopping, speculative decoding) contribute independent error terms or interact nonlinearly.
- The information-theoretic optimal adaptive step schedule for a given distribution.
- Whether the "time-agnostic" nature of MDMs (ICLR 2025) implies that step-count-dependent acceleration methods are fundamentally limited.

**Where the gaps are:**
- **Composability theory**: No theoretical framework exists for bounding the error when multiple acceleration methods are stacked. Each method has isolated analysis, but composition effects are purely empirical.
- **Acceleration-quality Pareto optimality**: No formal characterization of the Pareto frontier for DLM speed-quality tradeoffs under combined acceleration.
- **Adaptive scheduling optimality**: Despite several heuristic schedules (JoT, Prophet, SchED), no information-theoretically optimal adaptive schedule has been derived.
- **Cache approximation error propagation**: KV cache approximation introduces per-step errors, but how these compound across T denoising steps is not formally analyzed for DLMs.

---

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Composability Bound for Stacked DLM Acceleration

**Formal claim:** Let M_1, M_2, ..., M_k be k training-free acceleration methods applied to a DLM with T denoising steps. Let epsilon_i(t) denote the per-step error introduced by method M_i at step t (measured in KL divergence or total variation). Then the total sampling error of the composed system satisfies:

KL(p* || p_{composed}) <= KL(p* || p_{vanilla}) + sum_{i=1}^{k} sum_{t=1}^{T} epsilon_i(t) + sum_{i<j} Gamma_{ij}

where Gamma_{ij} is an interaction term that is zero when methods M_i and M_j operate on "orthogonal" computational dimensions (defined precisely via conditional independence of the induced error distributions), and is bounded by min(epsilon_i, epsilon_j) * C_{ij} otherwise, with C_{ij} depending on the overlap of their modification domains.

**Proof sketch:**
1. Model each acceleration method as a perturbation of the exact denoising transition kernel. Method M_i replaces the exact kernel K_t with an approximate kernel K_t^(i), where d_TV(K_t, K_t^(i)) <= epsilon_i(t).
2. Use the data processing inequality to propagate per-step errors: since the reverse process is a Markov chain, the total error after T steps is bounded by the sum of per-step errors (via the triangle inequality for f-divergences applied to Markov chains).
3. For composed methods, define the joint approximate kernel K_t^{composed} and decompose its distance from K_t into individual and interaction terms using a tensorization argument: if M_i modifies the keys/values and M_j modifies the step schedule, they operate on independent components of the computation graph.
4. The interaction term Gamma_{ij} arises when both methods modify overlapping components (e.g., both affect the attention computation). Bound Gamma_{ij} using the joint convexity of KL divergence and the structural decomposition of the attention operation.

**Empirical prediction:** If methods are truly "orthogonal" (Gamma_{ij} approximately 0), then the quality degradation under composition should be approximately additive. If Gamma_{ij} is large, we should observe super-additive degradation. The ComposeAccel experiments can directly test this by measuring quality loss for A alone, B alone, and A+B composed, checking whether loss(A+B) approximately= loss(A) + loss(B).

**Connection to existing theory:** Extends the error propagation framework of diffusion models (arXiv:2308.05021) to the discrete setting, and leverages the convergence theory of Li & Cai (2025) for the baseline DLM error term. The orthogonality condition is inspired by the f-divergence tensorization results in information theory.

**Novelty estimate:** 8/10. The concept of bounding composed approximation errors via tensorization exists in information theory, but its application to DLM acceleration composition is entirely novel. No existing work provides formal guarantees for stacking multiple acceleration methods.

---

### Candidate B: Mutual-Information-Optimal Adaptive Step Schedule for DLMs

**Formal claim:** For a masked DLM with sequence length L and target distribution p* with total correlation TC(p*), the minimum number of denoising steps T* to achieve sampling error KL(p_T || p*) <= epsilon is:

T* = Theta(TC(p*) / epsilon)

Furthermore, the optimal step schedule (determining how many tokens to unmask at each step) can be characterized as the solution to a convex optimization problem that minimizes total compute subject to the error constraint. The optimal schedule satisfies an "equal marginal information gain" principle: at each step t, the tokens unmasked should be those whose conditional mutual information with the remaining masked tokens, given the currently unmasked tokens, is minimized.

**Proof sketch:**
1. From Li & Cai (2025), we know KL(p_T || p*) = O(MI/T). The lower bound shows T* = Omega(MI/epsilon) is necessary. This gives the Theta characterization.
2. Formulate the step schedule as a resource allocation problem: given total budget T, how to distribute the "unmasking budget" across steps. Each step's contribution to error reduction depends on the conditional mutual information structure.
3. Use the chain rule of mutual information: MI(X_1, ..., X_L) = sum_i MI(X_i; X_{<i} | X_{>i}). The optimal schedule should unmask tokens in order of increasing conditional mutual information (easy tokens first, hard tokens last), analogous to the "low-hanging fruit" principle.
4. Prove that this schedule is optimal by showing it minimizes the worst-case error at each step via a greedy exchange argument (similar to matroid optimization).
5. Derive a practical approximation using the model's own confidence scores as a proxy for conditional mutual information.

**Empirical prediction:** (1) The optimal schedule should unmask more tokens in early steps (where most tokens are easy/low-MI) and fewer in late steps (where remaining tokens have high MI with context). This matches the empirical observation in Saber and Prophet. (2) On tasks with heterogeneous token difficulty (e.g., code with boilerplate + logic), the schedule should be more aggressive early and more conservative late, compared to uniform-difficulty tasks (e.g., continuation).

**Connection to existing theory:** Directly extends the O(MI/T) bound of Li & Cai (2025) and the adaptive schedule of Zhao & Cai (2026). The "equal marginal information gain" principle is analogous to water-filling in information theory. The confidence-based proxy connects to Prophet (early answer convergence) and JoT (token-level stopping).

**Novelty estimate:** 7/10. The characterization of T* is essentially known from Li & Cai (2025). The main novelty is the optimal schedule characterization and the connection to practical adaptive methods. The "equal marginal information gain" principle, while natural, has not been formally stated or proven for DLMs.

---

### Candidate C: Composability via Computational Orthogonality -- A Formal Framework

**Formal claim:** Define two acceleration methods M_1 and M_2 as "computationally orthogonal" with respect to a DLM if the computational graph modifications induced by M_1 and M_2 act on disjoint subsets of the tensor operations in the forward pass. Formally, let G = (V, E) be the computation graph of one denoising step, and let S_1, S_2 subset V be the nodes modified by M_1 and M_2 respectively. Then:

**Theorem (Orthogonal Composition):** If S_1 intersect S_2 = empty set and neither S_1 nor S_2 modifies the input-output interface of the other (no "backwash" through shared activations), then:

quality_loss(M_1 compose M_2) = quality_loss(M_1) + quality_loss(M_2) + O(epsilon_1 * epsilon_2)

where epsilon_i = max_t epsilon_i(t) is the worst-case per-step error, and the O(epsilon_1 * epsilon_2) second-order term arises from the interaction through shared intermediate representations.

**Corollary (Taxonomy):** This yields a natural three-class taxonomy of DLM acceleration methods:
- **Class Alpha (Step-count reducers):** JoT, Prophet, SchED -- modify the outer loop (number of steps T). S_alpha = {step scheduler}.
- **Class Beta (Per-step cost reducers via attention):** KV caching methods (EntropyCache, Fast-dLLM, Elastic-Cache, etc.) -- modify the attention computation within each step. S_beta = {K, V, attention weights}.
- **Class Gamma (Per-step cost reducers via model):** ES-dLLM, model scheduling -- modify which layers/parameters are evaluated. S_gamma = {layer selection, model routing}.

Since Alpha methods modify the outer loop while Beta/Gamma methods modify the inner computation, Alpha is approximately orthogonal to both Beta and Gamma. Beta and Gamma methods both modify the forward pass but target different components (attention vs. layers), so they are partially orthogonal with a bounded interaction term.

**Proof sketch:**
1. Define the exact denoising step as a function f_t: X_t -> X_{t-1}. Method M_i replaces f_t with f_t^(i) = f_t + delta_i(t), where delta_i is the perturbation.
2. Composition: f_t^{1,2} = f_t + delta_1(t) + delta_2(t) + cross(delta_1, delta_2). The cross term captures the interaction.
3. When S_1 and S_2 are disjoint in the computation graph, the Jacobian of the cross term decomposes: d(cross)/d(input) = (d delta_1/d z) * (d delta_2/d z), where z is the shared intermediate. Under Lipschitz conditions on the model, this is bounded by epsilon_1 * epsilon_2 * L^2 where L is the Lipschitz constant of the model.
4. Iterate across T steps using the error propagation framework. For Alpha + Beta composition, the Alpha method simply removes some steps, so the Beta error is applied to fewer steps, giving quality_loss(Alpha + Beta) <= quality_loss(Alpha) + (1 - r) * quality_loss(Beta), where r is the fraction of steps removed. This is actually sub-additive -- a synergy.
5. For Beta + Gamma composition, both modify the forward pass, so the interaction term scales as epsilon_beta * epsilon_gamma * T * L^2.

**Empirical prediction:**
1. Composing a step-count reducer (JoT/Prophet) with a KV caching method (EntropyCache) should yield sub-additive quality loss -- the composition performs BETTER than the sum of individual losses predicts, because caching errors apply to fewer steps.
2. Composing two per-step cost reducers (e.g., EntropyCache + ES-dLLM) should yield mildly super-additive quality loss, with the interaction growing with T.
3. The quality loss under triple composition (Alpha + Beta + Gamma) should be dominated by the pairwise Beta-Gamma interaction.

**Connection to existing theory:** The computational graph orthogonality concept draws from modular error analysis in software engineering and compositional verification in formal methods. The Lipschitz-based bound connects to the robustness literature for neural networks. The taxonomy aligns with the empirical observation in ComposeAccel that KV caching + IGSD shows synergy.

**Novelty estimate:** 9/10. This would be the first formal framework for reasoning about composability of DLM acceleration methods. The three-class taxonomy with provable interaction bounds is entirely novel. The closest existing work is the informal "orthogonality" claim in JoT, which is stated without proof or formal definition.

---

## Phase 3: Self-Critique

### Against Candidate A: Information-Theoretic Composability Bound

**Proof soundness attack:**
- The application of the data processing inequality to bound error propagation across steps is sound for exact Markov chains, but DLMs with approximate kernels may violate the Markov property. Specifically, KV caching makes step t's computation depend on cached values from step t-2, t-3, etc., breaking the simple Markov chain structure.
- The tensorization argument for orthogonal methods requires strong independence assumptions on the error distributions that may not hold in practice. The attention mechanism couples all components.
- RISK: The cross-term bound may be vacuous (too loose) in practice. If Gamma_{ij} ~ O(epsilon_i * epsilon_j) but epsilon values are not small (e.g., 0.1-0.3 for aggressive caching), the bound gives Gamma ~ 0.01-0.09, which may not be tight enough to be informative.

**Tightness attack:**
- The additive error bound (sum of per-step errors across T steps) grows linearly with T, which could be vacuous for large T. In contrast, the actual quality loss may be much smaller due to the self-correcting nature of denoising. This suggests the bound is loose by a factor of sqrt(T) or more.
- There may be a trivial construction achieving the bound (e.g., adversarial error alignment), making the bound technically correct but practically useless.

**Relevance attack:**
- Practitioners care about speed-quality Pareto curves, not worst-case bounds. If the bound is loose by 10x, it provides no practical guidance on whether to compose methods. The value would be in the taxonomy and orthogonality conditions, not the numerical bound.

**Novelty attack:**
- The general framework of bounding composed approximation errors via data processing inequality is well-known in information theory. The application to DLM acceleration is novel, but the proof techniques are standard. A reviewer might argue "straightforward application of known tools."

**Verdict:** MODERATE. The framework is valuable but the bounds may be too loose to be actionable. The main contribution would be the formal definition of orthogonality and the taxonomy, not the numerical bounds.

---

### Against Candidate B: MI-Optimal Adaptive Step Schedule

**Proof soundness attack:**
- The greedy exchange argument for optimality of the "low MI first" schedule assumes the MI structure is known and fixed. In practice, the conditional MI changes as tokens are revealed, so the greedy schedule may not be globally optimal.
- The proof requires oracle access to the true conditional mutual information, which is not available. The confidence-score proxy may have high variance, especially in early steps when most tokens are masked.
- The convex optimization formulation assumes the error as a function of the schedule is convex, which has not been verified for discrete diffusion processes.

**Tightness attack:**
- The T* = Theta(TC/epsilon) characterization is essentially already known from Li & Cai (2025) and Zhao & Cai (2026). The main new content is the schedule characterization, which depends on strong assumptions about the MI structure.
- The "water-filling" analogy is elegant but may oversimplify the discrete, combinatorial nature of the token selection problem.

**Relevance attack:**
- Multiple practical adaptive schedules already exist (JoT, Prophet, SchED, Saber) that work well empirically. A theoretical optimal schedule that requires oracle MI information is not directly useful. The gap between theory and practice is large.
- The practical approximation (using model confidence as MI proxy) is exactly what JoT and Prophet already do, so the experimental contribution may be limited.

**Novelty attack:**
- Searched for related work: the "equal marginal information gain" principle is essentially the information-theoretic water-filling solution, which is well-known. The application to DLM step scheduling is new but the mathematical content is incremental.
- Zhao & Cai (2026) already provide an adaptive schedule based on TC/DTC, making this a refinement rather than a new direction.

**Verdict:** WEAK. The formal result is mostly a repackaging of known bounds with an incremental schedule characterization. The practical gap (oracle MI vs. confidence proxy) limits impact. Too much overlap with existing adaptive schedule work.

---

### Against Candidate C: Composability via Computational Orthogonality

**Proof soundness attack:**
- The Lipschitz-based bound on the cross-term requires knowing the Lipschitz constant L of the transformer forward pass. For practical transformers, L can be exponentially large in depth, making the bound O(epsilon_1 * epsilon_2 * L^2) vacuous.
- MITIGATION: Could use local Lipschitz constants or empirically estimated smoothness. Alternatively, the bound could be stated in terms of the Jacobian spectral norm, which is typically much smaller than the global Lipschitz constant.
- The "disjoint computation graph" condition is too strict for real acceleration methods. KV caching modifies the attention computation, which feeds into all subsequent layers. ES-dLLM skips layers, which modifies the input to later attention layers. They are not truly disjoint -- they interact through the residual stream.
- MITIGATION: Replace the strict disjointness with a "weak coupling" condition quantified by the maximum singular value of the cross-Jacobian, which can be empirically measured.

**Tightness attack:**
- The synergy prediction for Alpha + Beta (step reduction + caching) seems tight: if you remove r fraction of steps, you apply caching error to (1-r)*T steps, giving sub-additive behavior. This is a genuine insight.
- The super-additive prediction for Beta + Gamma needs more careful analysis. The interaction might be dampened by the residual stream's averaging effect. The bound could be tighter with architecture-specific analysis.

**Relevance attack:**
- The three-class taxonomy (step-count, per-step attention, per-step model) is highly relevant to practitioners. It provides actionable guidance: "compose methods from different classes for free; compose within-class with caution."
- The formal framework, even if bounds are loose, provides the first principled language for discussing DLM acceleration composability.

**Novelty attack:**
- Extensive search confirms no prior work provides formal composability analysis for DLM acceleration methods. The taxonomy, the orthogonality definition, and the synergy prediction for Alpha+Beta are all novel.
- The closest related concept is the informal "orthogonality" claim in JoT (2026), which is stated without proof. Our framework subsumes and formalizes this claim.

**Verdict:** STRONG. Despite the Lipschitz constant concern (addressable with local smoothness), this idea is the most novel, most relevant, and has the clearest connection to the ComposeAccel experimental program. The taxonomy and synergy predictions are testable and actionable.

---

## Phase 4: Refinement

### Dropped Ideas

**Candidate B (MI-Optimal Adaptive Step Schedule):** Dropped due to (1) high overlap with existing adaptive schedule work, (2) the oracle MI requirement makes the theory impractical, (3) the T* characterization is essentially already known. The "water-filling" insight, while elegant, is incremental.

### Strengthened Survivors

**Candidate C (Composability via Computational Orthogonality)** is selected as the front-runner. Strengthened as follows:

1. **Weakened assumptions:** Replace the strict "disjoint computation graph" condition with a quantitative "weak coupling" condition based on empirically measurable cross-Jacobian spectral norms. This makes the framework applicable to real methods where some interaction exists.

2. **Tightened the synergy result:** For Alpha + Beta composition (step reduction + caching), the sub-additive behavior can be proven more precisely: quality_loss(Alpha + Beta) = (1 - r) * quality_loss(Beta) + quality_loss(Alpha), where r is the step reduction ratio. This is not just sub-additive but has a precise multiplicative factor. This predicts that step-count reducers and per-step cost reducers are "natural allies" -- a testable and surprising claim.

3. **Incorporated Candidate A's error propagation analysis** as a supporting lemma within Candidate C's framework. The per-step error propagation bound becomes the base case, and the composability framework extends it to multiple methods.

4. **Added connection to DIG framework:** The Diffusion Information Geometry perspective (gradient flow on Fisher-Rao manifold) provides geometric intuition for why certain compositions work: if method M_1's perturbation is in the tangent direction of the manifold flow and M_2's perturbation is orthogonal to it, the composition preserves the flow direction. This geometric condition can be related to the computational graph condition.

5. **Concrete experimental predictions formulated** to test each theorem with ComposeAccel's existing experimental setup.

### Front-Runner Selection

**Candidate C: Composability via Computational Orthogonality** is selected because:
- It directly addresses Gap 11 (composition of acceleration methods), the most novel gap in the literature
- It provides the theoretical backbone for ComposeAccel's experimental program
- The three-class taxonomy and synergy/interference predictions are immediately testable
- Novelty is confirmed at 9/10 with no competing formal framework

---

## Phase 5: Final Proposal

### Title
**Computational Orthogonality: An Information-Theoretic Framework for Composable DLM Acceleration**

### Formal Claim

**Main Theorem (Composability Bound):** Let M_alpha be a step-count reducer that removes fraction r of denoising steps, and M_beta be a per-step cost reducer (e.g., KV cache approximation) with per-step error epsilon_beta(t). Under the assumption that M_beta's per-step error is independent of which steps are selected by M_alpha (i.e., epsilon_beta(t) does not depend on the step scheduling), the composed system satisfies:

KL(p* || p_{alpha+beta}) <= KL(p* || p_{alpha}) + (1-r) * sum_{t in S} epsilon_beta(t)

where S is the set of steps retained by M_alpha. Furthermore, this bound is tight up to a factor depending on the spectral norm of the cross-Jacobian J_cross = d(delta_alpha)/d(delta_beta), which is empirically measurable.

**Corollary (Sub-Additivity):** When r > 0, quality_loss(M_alpha + M_beta) < quality_loss(M_alpha) + quality_loss(M_beta). Step-count reducers and per-step cost reducers exhibit fundamental synergy.

**Theorem (Interference Bound):** Let M_beta and M_gamma be two per-step cost reducers modifying overlapping components of the forward pass (e.g., attention approximation + layer skipping). Then:

quality_loss(M_beta + M_gamma) <= quality_loss(M_beta) + quality_loss(M_gamma) + T * epsilon_beta * epsilon_gamma * sigma_max(J_cross)^2

where sigma_max(J_cross) is the maximum singular value of the cross-Jacobian between the two perturbations, and T is the number of retained denoising steps.

**Proposition (Three-Class Taxonomy):** DLM acceleration methods naturally partition into three classes based on which component of the denoising computation they modify:
- Alpha (step count): {JoT, Prophet, SchED, Saber}
- Beta (attention/KV): {EntropyCache, Fast-dLLM, Elastic-Cache, SPA-Cache, Sparse-dLLM, DyLLM, Window-Diffusion}
- Gamma (model architecture): {ES-dLLM, model scheduling}

Inter-class compositions (Alpha+Beta, Alpha+Gamma) exhibit guaranteed sub-additivity. Intra-class compositions (Beta+Beta, Gamma+Gamma) and cross-compute compositions (Beta+Gamma) exhibit bounded interference.

### Proof Sketch

**Key Steps:**

**Lemma 1 (Per-step error propagation):** Model the DLM reverse process as a Markov chain with approximate transition kernels. By the data processing inequality for KL divergence applied to Markov chains:

KL(p*_0 || hat{p}_0) <= sum_{t=1}^{T} E_{x_t ~ p*_t}[KL(K_t(. | x_t) || hat{K}_t(. | x_t))]

This bounds the total error by the sum of expected per-step divergences. (Standard, extends Chen et al. 2023 to discrete setting.)

**Lemma 2 (Step elimination):** When M_alpha removes step t_0 (replacing it with identity), the error at t_0 is the KL divergence between the exact transition and the identity. For remaining steps, the transition kernels are unchanged. The total error redistributes: MI contributions that would be resolved at t_0 are pushed to adjacent steps, but the total MI budget is conserved.

**Lemma 3 (Cross-perturbation bound):** For composed perturbation f_t + delta_beta + delta_gamma + cross(delta_beta, delta_gamma), use the second-order Taylor expansion of KL divergence:

KL(f_t || f_t + delta_beta + delta_gamma) approx= KL(f_t || f_t + delta_beta) + KL(f_t || f_t + delta_gamma) + delta_beta^T * Fisher * delta_gamma

The cross-term delta_beta^T * Fisher * delta_gamma is bounded by epsilon_beta * epsilon_gamma * lambda_max(Fisher), where the Fisher information matrix of the denoising model is the key quantity. When delta_beta and delta_gamma are "orthogonal" in the Fisher metric (i.e., live in orthogonal subspaces of the tangent space), the cross-term vanishes.

**Lemma 4 (Synergy for Alpha + Beta):** The composed error for step reduction (fraction r removed) + per-step approximation (error epsilon per step) is:

E_composed = E_alpha + (1-r) * T * epsilon

Since E_individual_beta = T * epsilon, we have E_composed < E_alpha + E_beta whenever r > 0. The savings are proportional to r.

### Assumptions

1. **Approximate Markov property:** The DLM reverse process with approximate kernels remains approximately Markov. Violated when KV caching introduces long-range dependencies across steps. Can be quantified by the mixing time of the approximate chain.

2. **Lipschitz smoothness:** The denoising function f_t is locally Lipschitz with constant L that is polynomially bounded in model dimension. This is empirically reasonable for transformer architectures but not formally proven.

3. **Error independence:** The per-step error epsilon_i(t) of method M_i does not depend on the errors introduced by other methods at previous steps. This is approximately true when methods operate on different computational components, but may be violated for tightly coupled methods.

4. **Bounded Fisher information:** The Fisher information matrix of the denoising model has bounded spectral norm. This is needed for the cross-perturbation bound. Empirically checkable.

### Empirical Prediction

1. **Sub-additivity test (Alpha + Beta):**
   - Compose JoT (step reducer, removes ~70% of computation) with EntropyCache (KV approximation, ~15-26x speedup).
   - Predict: quality_loss(JoT + EntropyCache) < quality_loss(JoT) + quality_loss(EntropyCache).
   - Specifically, if JoT removes r = 0.5 of steps and EntropyCache has epsilon_beta per step, then quality_loss(composed) should be approximately quality_loss(JoT) + 0.5 * quality_loss(EntropyCache).
   - Benchmark: MMLU, GSM8K, HumanEval on LLaDA-8B-Instruct.

2. **Interference test (Beta + Gamma):**
   - Compose EntropyCache (KV approximation) with ES-dLLM (layer skipping).
   - Predict: quality_loss(composed) > quality_loss(EntropyCache) + quality_loss(ES-dLLM), with the excess proportional to the cross-Jacobian spectral norm.
   - Measure the cross-Jacobian spectral norm empirically (via random projection estimates) and verify it predicts the excess error.

3. **Triple composition (Alpha + Beta + Gamma):**
   - Compose JoT + EntropyCache + ES-dLLM.
   - Predict: the dominant interaction term is the Beta+Gamma interference; the Alpha+Beta synergy partially compensates.
   - Predict: quality_loss(triple) < quality_loss(EntropyCache + ES-dLLM) + quality_loss(JoT).

4. **Cross-Jacobian measurement:**
   - For each pair of methods, estimate sigma_max(J_cross) using the power iteration method on the Jacobian-vector product.
   - Predict: Alpha-Beta pairs have sigma_max approximately 0 (orthogonal); Beta-Gamma pairs have sigma_max in (0.01, 0.3); Beta-Beta pairs have sigma_max in (0.1, 0.8).

### Experimental Plan

All experiments target LLaDA-8B-Instruct on a single GPU. Each experiment under 1 hour.

**Experiment 1: Baseline quality measurements (30 min)**
- Vanilla DLM (T=64 steps): MMLU (5-shot), GSM8K (8-shot), HumanEval (pass@1)
- EntropyCache alone: same benchmarks, measure quality_loss_beta
- JoT alone: same benchmarks, measure quality_loss_alpha
- ES-dLLM alone: same benchmarks, measure quality_loss_gamma

**Experiment 2: Pairwise composition quality (45 min)**
- JoT + EntropyCache: measure quality_loss(alpha + beta)
- EntropyCache + ES-dLLM: measure quality_loss(beta + gamma)
- JoT + ES-dLLM: measure quality_loss(alpha + gamma)
- Verify: loss(alpha + beta) < loss(alpha) + loss(beta) (sub-additivity)
- Verify: loss(beta + gamma) > loss(beta) + loss(gamma) (super-additivity)

**Experiment 3: Cross-Jacobian estimation (30 min)**
- For 100 random input samples, compute Jacobian-vector products for each method pair
- Estimate sigma_max(J_cross) via power iteration (10 iterations)
- Correlate sigma_max with the observed interaction term (excess or deficit in quality loss)

**Experiment 4: Triple composition and Pareto frontier (45 min)**
- JoT + EntropyCache + ES-dLLM at varying aggressiveness levels
- Map the speed-quality Pareto frontier
- Verify theoretical predictions about which combinations dominate

### Baselines

**Theoretical baselines:**
- Naive additive bound: quality_loss(composed) <= sum of individual losses (no interaction)
- Li & Cai (2025) bound: KL = O(MI/T) for vanilla DLM (no acceleration)
- Zhao & Cai (2026) adaptive bound: KL = O-tilde(TC/K) with randomized schedule

**Empirical baselines:**
- Vanilla LLaDA-8B-Instruct (T = 64 steps)
- Each individual method alone (EntropyCache, JoT, ES-dLLM)
- SlowFast + dLLM-Cache (only published composition baseline)

### Risk Assessment

1. **Lipschitz constant may be too large:** The transformer Lipschitz constant can be exponentially large in depth, making the interference bound vacuous. MITIGATION: Use empirically estimated local Lipschitz constants; restrict the bound to apply "in distribution" rather than worst-case.

2. **Error independence assumption may fail:** KV cache errors at step t may bias the attention patterns at step t+1, causing ES-dLLM's layer skipping decisions to be affected. MITIGATION: Measure the correlation between per-step errors empirically; if high, modify the bound to account for correlated errors.

3. **Fisher information estimation cost:** Computing the Fisher information matrix for an 8B model is expensive. MITIGATION: Use the empirical Fisher (average of outer products of gradients) with random projections to estimate the spectral norm, not the full matrix.

4. **Gap between theory and practice:** The bounds may be correct in order but off by large constants, reducing practical value. MITIGATION: Frame the contribution as the taxonomy and qualitative predictions (sub/super-additivity), not the quantitative bounds.

5. **Time-agnostic MDM result (ICLR 2025) may undermine step-count analysis:** If MDMs are truly time-agnostic, the Alpha class distinction may be less meaningful. MITIGATION: Our framework applies to the practical discrete-step implementation, not the continuous-time limit. The time-agnostic result applies to the training objective, not the finite-step sampling procedure.

### Novelty Claim

This work would provide:
1. **The first formal definition of "composability" for DLM acceleration methods**, replacing informal claims with a precise mathematical framework.
2. **The first provable synergy result** (Alpha + Beta sub-additivity), explaining why step-count reducers and per-step cost reducers are natural allies.
3. **The first interference bound** (Beta + Gamma interaction), predicting when compositions fail and by how much.
4. **A principled three-class taxonomy** grounded in computational graph analysis, providing actionable guidance for practitioners.
5. **Empirically testable predictions** that can be validated with ComposeAccel's existing experimental infrastructure.

**Evidence of novelty:** Extensive literature search confirms no prior work provides formal composability analysis for DLM acceleration. The closest related concepts are:
- JoT (2026): claims orthogonality to KV caching without proof or formal definition
- SlowFast + dLLM-Cache: empirical composition of two methods from the same family, no theoretical analysis
- Error propagation framework (arXiv:2308.05021): analyzes single-method error propagation, not composition
- Li & Cai (2025) / Zhao & Cai (2026): analyze vanilla DLM convergence, not accelerated variants

None of these address the composition question theoretically. This gap is confirmed by the literature survey (Gap 11) and the positioning analysis in Section 8 of the literature report.
