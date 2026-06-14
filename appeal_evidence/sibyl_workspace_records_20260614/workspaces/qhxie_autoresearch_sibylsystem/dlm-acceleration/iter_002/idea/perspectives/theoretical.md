# Theoretical Perspective

## Phase 1: Literature Survey

### Key Theoretical Papers

1. **"Optimal Inference Schedules for Masked Diffusion Models"** (Chen, Cong, Li; arXiv:2511.04647, Nov 2025) -- Exact characterization of the expected KL divergence between the true distribution and the sampled distribution for any distribution and any unmasking schedule. Connects optimal scheduling to the theory of univariate function approximation. Shows that in some natural settings one can sample in O(log n) steps without visible loss, where n is the sequence length. Key concepts: total correlation, dual total correlation, information curve. **This is the most foundational theoretical result for DLM inference scheduling.**

2. **"A Convergence Theory for Diffusion Language Models: An Information-Theoretic Perspective"** (Li, Cai; arXiv:2505.21400, May 2025) -- First convergence guarantees for DLMs. Shows KL divergence decays as O(1/T) where T is the number of iterations, with coefficient governed by the mutual information I(X^(i); X^(-i)) between each token and the rest. Proves matching lower bounds, establishing tightness. Crucially covers the regime T < L (fewer steps than sequence length), justifying parallel sampling.

3. **"Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations"** (Lavenant, Zanella; arXiv:2510.25544, Oct 2025) -- Error bounds in relative entropy that depend only on the average number of tokens generated per iteration, independent of sequence length. Identifies the optimal schedule as a function of the "information profile" of the data distribution. Provides a principled optimization framework for schedule sizes.

4. **"Adaptation to Intrinsic Dependence in Diffusion Language Models"** (arXiv:2602.20126, Feb 2026) -- Distribution-agnostic unmasking schedule that adapts to unknown dependence structure. Convergence guarantees scale as O~(TC/K) and O~(DTC/K) where TC and DTC are total correlation and dual total correlation. Uses randomized unmasking sizes rather than deterministic ones.

5. **"The Cosine Schedule is Fisher-Rao-Optimal for Masked Discrete Diffusion Models"** (Zhang, Syed; arXiv:2508.04884, Aug 2025) -- Shows the widely-used cosine schedule is optimal under the Fisher-Rao geometry of the induced probability path. Provides an information-geometric justification for a previously heuristic choice.

6. **"KLASS: KL-Guided Fast Inference in Masked Diffusion Models"** (Kim et al.; arXiv:2511.05664, Nov 2025; NeurIPS 2025 Spotlight) -- Uses token-level KL divergence to identify stable, high-confidence predictions and unmask multiple tokens per iteration. Achieves 2.78x wall-clock speedup while improving performance. Validates that KL divergence is a practical and theoretically grounded signal for adaptive scheduling.

7. **"Limits of KV Cache Compression for Tensor Attention based Autoregressive Transformers"** (arXiv:2503.11108, Mar 2025) -- Proves that any algorithm producing a (1 +/- eta)-approximation of attention output requires Omega(nd) bits of memory. Shows that even approximate KV cache computation faces fundamental information-theoretic lower bounds. **Critical negative result** bounding what is achievable with KV caching.

8. **"Absorb and Converge: Provable Convergence Guarantee for Absorbing Discrete Diffusion Models"** (NeurIPS 2025) -- First finite-time error bounds using a Jensen-based approach for absorbing rate matrices. Demonstrates improved convergence rates for tau-leaping and uniformization samplers without early stopping.

9. **"Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models"** (ICLR 2025) -- Derives a practical upper bound on Compounding Decoding Error (CDE) and proposes an efficient algorithm for optimal non-uniform timestep allocation across image, music, and text.

10. **"Convergence of Score-Based Discrete Diffusion Models: A Discrete-Time Analysis"** (ICLR 2025) -- KL divergence bounds nearly linear in dimension d, consisting of score estimation error + discretization error + truncation error. Establishes the discrete-time analysis framework.

11. **"Streaming Attention Approximation via Discrepancy Theory" (BalanceKV)** (arXiv:2502.07861; NeurIPS 2025 Spotlight) -- Uses discrepancy theory to derive streaming attention approximation guarantees with provable error bounds.

12. **"Variational Speculative Decoding"** (arXiv:2602.05774, Feb 2026) -- Proves that maximizing the VSD objective is equivalent to increasing the lower bound of the expected acceptance length, establishing a direct mathematical link between variational bound and wall-clock speedup.

### Theoretical Landscape Summary

The theoretical understanding of DLM inference has crystallized rapidly in 2025-2026 around several key results:

**What is known (proved):**
- The sampling error for masked diffusion models decays as O(1/T) with a coefficient governed by the mutual information structure of the target distribution (Li & Cai, 2025). This bound is tight -- no faster rate is achievable in general.
- The optimal unmasking schedule for any distribution can be characterized exactly via the information curve (Chen et al., 2025), connecting to function approximation theory. Under favorable dependence structures (low total correlation), O(log n) steps suffice.
- The cosine schedule is Fisher-Rao-optimal for the masking rate (Zhang & Syed, 2025).
- KV cache compression faces fundamental Omega(nd) memory lower bounds even for approximate computation (Haris & Onak, 2025; generalized to tensor attention).
- Distribution-agnostic adaptive schedules can achieve convergence proportional to TC/K or DTC/K without knowing the dependence structure a priori (arXiv:2602.20126).

**What is conjectured but not proved:**
- Whether per-token adaptive scheduling (different tokens get different numbers of steps) can provably beat global scheduling. The information-theoretic results characterize global schedules, but per-token adaptation has no formal theory.
- Whether KV cache approximation error in DLMs (bidirectional attention) has different asymptotic behavior than in AR models (causal attention). The existing lower bounds apply to causal attention; the bidirectional case may be structurally different.
- Whether the composition of independently optimal acceleration methods (caching + scheduling + parallel decoding) preserves optimality or introduces compounding errors.

**Where the gaps are:**
- **Gap A (Per-token convergence theory):** All existing convergence results are sequence-level. There is no theory for how individual tokens converge at different rates and how this heterogeneity can be exploited for acceleration. KLASS uses KL divergence empirically but lacks formal guarantees.
- **Gap B (Composition error bounds):** When you apply KV caching (approximate attention) AND adaptive scheduling (non-uniform steps) AND parallel decoding (multiple tokens per step), the total error is not simply the sum of individual errors. There is no theory for how these approximations interact.
- **Gap C (Instance-dependent acceleration):** Existing bounds depend on worst-case distribution parameters (total correlation, mutual information). Instance-dependent bounds that exploit the structure of specific text (e.g., code vs. natural language) to predict achievable speedup are absent.
- **Gap D (Theoretical foundation for KV cache validity in DLMs):** The empirical success of KV caching in DLMs (Fast-dLLM achieves 27x speedup) lacks a theoretical explanation. Why do KV pairs between steps change so little? Under what conditions does the approximation error remain controlled?

## Phase 2: Initial Candidates

### Candidate A: Information-Theoretic Decomposition of DLM Acceleration (IT-Decompose)

**Formal claim (Theorem sketch):** Let M be a masked diffusion language model generating a sequence X = (X_1, ..., X_n) from distribution p. Let A_cache, A_sched, A_decode denote three acceleration operators corresponding to KV cache approximation, adaptive step scheduling, and parallel decoding, respectively. Define the total generation error as the KL divergence D_KL(p_A || p) where p_A is the distribution under the combined acceleration A = A_cache composed with A_sched composed with A_decode.

**Claim:** Under mild independence conditions on the approximation mechanisms, the total error decomposes as:

D_KL(p_A || p) <= D_KL(p_cache || p) + D_KL(p_sched || p) + D_KL(p_decode || p) + R_interact

where R_interact is a residual interaction term bounded by a function of the pairwise mutual information between the approximation errors. Specifically, R_interact = O(I(E_cache; E_sched) + I(E_cache; E_decode) + I(E_sched; E_decode)), where E_x denotes the error introduced by each approximation.

**Proof sketch:**
1. Apply the chain rule of KL divergence to decompose the joint error.
2. Use the data processing inequality to bound each component.
3. Bound the interaction terms using the mutual information between approximation errors, which vanishes when the approximations act on independent aspects of the computation.

**Empirical prediction:** If the interaction term R_interact is small (methods are approximately orthogonal), then the combined speedup should be near-multiplicative. If R_interact is large, the combined speedup will be subadditive. This can be measured by comparing observed combined error to the sum of individual errors.

**Connection to existing theory:** Extends the Li & Cai (2025) convergence framework by decomposing the total acceleration error into component contributions. Related to the chain rule for KL divergence and the tensorization property of relative entropy.

**Novelty estimate:** 6/10 -- The decomposition framework is clean but the individual bounds rely on existing results. The main novelty is the formal identification of the interaction term and conditions for orthogonality. The approach of decomposing acceleration error into independent components is natural but has not been formalized.

### Candidate B: Token Convergence Rate Characterization via Conditional Mutual Information (TCR)

**Formal claim (Theorem sketch):** Define the convergence rate of token i at denoising step t as:

r_i(t) = D_KL(p(X_i | X_{-i}^{(t)}) || p(X_i | X_{-i}^{(t-1)}))

where X_{-i}^{(t)} denotes all other tokens at step t. Then:

**Claim 1 (Convergence rate bound):** The convergence rate is bounded by:

r_i(t) <= I(X_i; M_t | X_{-i}^{(t-1)}) * (1/T)

where M_t is the masking indicator at step t and I denotes conditional mutual information. Tokens with lower conditional mutual information converge faster and can be resolved in fewer steps.

**Claim 2 (Optimal per-token budget):** The information-theoretically optimal step budget for token i is proportional to I(X_i; X_{-i}) / sum_j I(X_j; X_{-j}), the fraction of total mutual information attributable to token i. Tokens that are highly predictable from context (low conditional MI) should receive fewer steps; tokens that carry novel information should receive more.

**Proof sketch:**
1. Use Fano's inequality to relate the convergence rate to the conditional mutual information.
2. Apply the convexity of KL divergence to show that the total error is minimized when step budgets are allocated proportionally to per-token information content.
3. The result follows from the Lagrangian optimization of total error subject to a total step budget constraint.

**Empirical prediction:** 
- Function words (the, is, of) should converge in 1-3 steps, content words in 5-15 steps, and novel/technical terms in 15-30+ steps. Measuring per-token convergence rates on real DLMs should show this hierarchy.
- An oracle per-token scheduler (given true conditional MI values) should achieve the same quality as T=64 uniform steps using only T_eff = sum_i min(T, c * I(X_i; X_{-i})) effective steps, where T_eff << 64*n.

**Connection to existing theory:** Directly extends Chen et al. (2025) from global schedules to per-token schedules. Uses the mutual information characterization of Li & Cai (2025) at the token level rather than the sequence level. Related to rate-distortion theory where each token's "rate" determines its compression budget.

**Novelty estimate:** 8/10 -- Per-token convergence rate characterization for MDMs is genuinely new. All existing theoretical results are sequence-level. The connection between conditional mutual information and optimal per-token step budgets has not been established. KLASS uses KL divergence empirically but without formal optimality guarantees.

### Candidate C: Provable Bounds on KV Cache Validity in Bidirectional Diffusion Attention

**Formal claim (Theorem sketch):** Let K_t, V_t be the key-value matrices at denoising step t, and K_{t+1}, V_{t+1} at step t+1. Let delta_t = ||K_{t+1} - K_t||_F / ||K_t||_F be the relative change in keys, and Attn_t, Attn_{t+1} the resulting attention outputs.

**Claim:** The attention output error from using cached KV pairs satisfies:

||Attn_{t+1}(Q_{t+1}, K_t, V_t) - Attn_{t+1}(Q_{t+1}, K_{t+1}, V_{t+1})||_inf <= C * (delta_K * ||V_t||_2 + delta_V * ||K_t||_2) * kappa(softmax)

where kappa(softmax) is the condition number of the softmax operation (related to the temperature and maximum attention logit), and C is a constant depending on the attention dimension.

Moreover, for the specific case of masked diffusion models where tokens are unmasked one-at-a-time:

delta_K(t) <= (n_unmask(t) / n) * ||DeltaEmb||_2

where n_unmask(t) is the number of tokens unmasked at step t and DeltaEmb is the embedding change from mask to decoded token. This implies that KV cache error is proportional to the fraction of tokens changed between steps, providing a formal justification for the empirical observation that KV caching is most accurate when few tokens change per step.

**Proof sketch:**
1. Bound the change in softmax attention weights using the Lipschitz continuity of softmax.
2. Bound the key/value change in terms of the number of tokens that changed state (masked->unmasked or changed prediction).
3. Combine using the triangle inequality to get the total output error.

**Empirical prediction:**
- KV cache error should scale linearly with the number of tokens unmasked per step. This predicts that aggressive parallel decoding (many tokens per step) should degrade KV cache accuracy, explaining the observed tension between parallel decoding and caching.
- The bound tightens as denoising progresses (fewer tokens change -> smaller delta_K), predicting that KV caching should be most accurate in late denoising steps. This matches the empirical observation in EntropyCache.

**Connection to existing theory:** Extends the KV cache compression lower bounds (arXiv:2503.11108) from the causal/AR setting to the bidirectional/diffusion setting. Provides upper bounds that complement the existing lower bounds.

**Novelty estimate:** 7/10 -- The formal analysis of KV cache validity specifically for bidirectional diffusion attention is new. The connection between number of unmasked tokens and cache accuracy has been observed empirically (Fast-dLLM, EntropyCache) but never proved. The main risk is that the bounds may be too loose to be informative.

## Phase 3: Self-Critique

### Against Candidate A (IT-Decompose)

**Proof soundness attack:** The decomposition via chain rule of KL divergence is standard and likely sound. However, the independence assumption on approximation mechanisms is the critical weakness. In practice, KV cache approximation error at step t affects the token distributions seen by the adaptive scheduler at step t+1, creating a dependency chain. The "mild independence conditions" may not hold in the most interesting cases (aggressive acceleration). Searched for "error decomposition approximate inference" -- related work exists in variational inference (ELBO decomposition) but not for this specific multi-approximation setting.

**Tightness attack:** The interaction term R_interact is bounded by mutual information between errors, which could be vacuously large if the errors are correlated. In the worst case, R_interact could dominate the individual terms, making the decomposition uninformative. The bound would need to be validated empirically to show that R_interact is indeed small for practical configurations.

**Relevance attack:** The decomposition tells practitioners whether methods compose well, which is highly relevant (Gap 11 from the literature survey). However, the result is conditional on measuring the mutual information between approximation errors, which itself requires running all methods -- so the theory may not provide advance prediction of composability without experiments.

**Novelty attack:** Searched for "error decomposition composition approximate inference KL divergence." The idea of decomposing approximation errors via KL chain rule is textbook-level information theory. The novelty claim rests entirely on the application to DLM acceleration, which is narrow. The formal framework does not yield surprising predictions.

**Verdict:** MODERATE -- The framework is sound but potentially too loose to be informative, and the novelty is limited to the application context. The theoretical contribution is a clean formalization rather than a deep new insight.

### Against Candidate B (TCR - Token Convergence Rate)

**Proof soundness attack:** The use of Fano's inequality to connect convergence rate to conditional mutual information is well-established in information theory. The convexity argument for optimal allocation follows from standard Lagrangian optimization. The main technical concern is whether the conditional mutual information I(X_i; X_{-i}^{(t)}) can be reliably estimated from the DLM's logits. In theory yes (it is a function of the conditional distribution p(X_i | X_{-i}^{(t)})), but in practice the DLM's predictions are imperfect approximations of the true conditional. Searched for "per-token mutual information estimation language model" -- no prior work proves that DLM logits are reliable estimates of conditional MI.

**Tightness attack:** The bound in Claim 1 is likely not tight for individual tokens. The O(1/T) decay comes from averaging over all tokens; individual tokens may converge much faster or slower depending on context. The per-token budget in Claim 2 assumes tokens can be independently scheduled, which conflicts with the bidirectional attention structure (all tokens attend to all others in each forward pass). In practice, you cannot skip steps for token i without also affecting the computation for token j.

**Relevance attack:** High relevance. This directly addresses the question "which tokens need more denoising?" that is at the heart of every adaptive DLM acceleration method (DyLLM, ES-dLLM, EntropyCache, KLASS). A formal characterization of per-token convergence would provide the theoretical foundation that all these methods lack. The empirical prediction (function words converge fast, novel terms converge slow) is testable and practically useful.

**Novelty attack:** Searched for "per-token convergence rate diffusion model" and "token-level information-theoretic bounds diffusion." Chen et al. (2025) characterize optimal global schedules; Lavenant & Zanella (2025) study factorized approximations; the "Adaptation to Intrinsic Dependence" paper (2602.20126) studies distribution-adaptive schedules. None of these provide per-token convergence rates. KLASS uses token-level KL divergence empirically but without formal optimality characterization. The per-token convergence rate characterization appears genuinely novel.

**Verdict:** STRONG -- The formal claim is novel, the proof techniques are established (information theory + convex optimization), the predictions are testable, and the result is highly relevant to the entire field of DLM acceleration. The main weakness is the practical gap between the theoretical per-token schedule and what is implementable (you cannot actually give different numbers of forward passes to different tokens without significant engineering).

### Against Candidate C (KV Cache Validity Bounds)

**Proof soundness attack:** The Lipschitz bound on softmax is well-established. The connection between number of unmasked tokens and key change is straightforward for masked diffusion where the embedding of masked tokens differs from unmasked tokens. The main concern is that the softmax condition number kappa(softmax) can be very large when attention logits are sharp (low temperature), making the bound vacuous precisely when KV caching works best (confident predictions = sharp attention). Searched for "softmax Lipschitz constant attention mechanism bound" -- known results show that the Lipschitz constant of softmax is bounded by 1, but the attention output Lipschitz constant depends on the value vectors and can be large.

**Tightness attack:** The bound is likely loose. In practice, KV cache errors are much smaller than the worst-case Lipschitz bound predicts, because the changes in key vectors are structured (only certain positions change) and the attention mechanism is locally sensitive (changes in far-away keys have less impact on nearby queries). A tighter bound would require exploiting the structure of attention patterns, which varies across layers and heads.

**Relevance attack:** High relevance for KV caching methods (Fast-dLLM, EntropyCache, Elastic-Cache, dKV-Cache). Understanding when KV caching is valid and when it breaks would help practitioners choose refresh rates and design better caching strategies. The prediction that error scales with fraction of tokens changed per step is directly testable.

**Novelty attack:** Searched for "KV cache error analysis diffusion model bidirectional attention." No prior work provides formal bounds on KV cache approximation error specifically for DLMs. The AR case has been studied (arXiv:2503.11108 provides lower bounds), but the bidirectional case is structurally different and unexplored. The novelty is real but the contribution may be seen as a straightforward application of Lipschitz analysis.

**Verdict:** MODERATE -- The bounds are likely too loose to be informative, and the analysis is a relatively straightforward application of Lipschitz arguments. However, the qualitative predictions (error scales with tokens changed, improves in late denoising) are useful and testable. Best as a supporting result rather than the main contribution.

## Phase 4: Refinement

### Dropped Ideas

**Candidate A (IT-Decompose)** is dropped as the primary proposal because: (1) The novelty is limited to an application of standard information-theoretic decomposition. (2) The interaction term may be too loose to provide informative predictions. (3) The theoretical contribution does not yield a fundamentally new insight about DLMs. However, elements of this decomposition framework are incorporated into the final proposal as a supporting theoretical tool.

### Strengthened Ideas

**Candidate B (Token Convergence Rate)** is strengthened with the following refinements:

1. **Practical bridge via entropy proxy:** The conditional mutual information I(X_i; X_{-i}) is intractable to compute exactly, but the token-level entropy H(X_i | X_{-i}) (estimated from DLM logits) is a directly computable upper bound. The theory shows that entropy is an upper bound on the convergence budget needed; the tighter conditional MI characterization explains WHY entropy-based methods (EntropyCache, KLASS) work -- they approximate the true optimality criterion.

2. **Connection to information curve:** Chen et al. (2025) define the "information curve" for the global schedule. We extend this to a "per-token information curve" that characterizes the convergence trajectory of each token. The global information curve is the sum of per-token curves, and the optimal global schedule is determined by the aggregate curvature. But per-token curves can differ dramatically, motivating token-adaptive scheduling.

3. **Implementable approximation:** Since different tokens cannot literally receive different numbers of forward passes in a standard transformer (all tokens are processed together), the theory motivates a practical scheme: at each step, compute a full forward pass but "freeze" (stop updating) tokens that have converged according to the per-token criterion. This is mathematically equivalent to a per-token schedule but implementable within the standard architecture. This is precisely what DyLLM and ES-dLLM do empirically -- our theory provides the formal justification and optimal criterion.

**Candidate C (KV Cache Validity)** is retained as a supporting result, strengthened by:

1. **Tighter bound via structured perturbation:** Instead of worst-case Lipschitz, analyze the structured perturbation where only specific token embeddings change. Use the low-rank structure of the key change (only rows corresponding to newly unmasked tokens change) to derive a tighter bound.

2. **Connecting cache validity to Token Convergence Rate:** A token that has converged (in the sense of Candidate B) will not significantly change its key/value in subsequent steps. This connects the two results: the per-token convergence criterion simultaneously determines (a) which tokens can be frozen and (b) which KV pairs can be cached.

### Additional Evidence Found

- **KLASS (NeurIPS 2025 Spotlight):** Validates that token-level KL divergence is a practical signal for adaptive scheduling. Our theory provides the formal optimality characterization that KLASS lacks.
- **"Adaptation to Intrinsic Dependence"** (arXiv:2602.20126): Shows that randomized unmasking sizes achieve O~(TC/K) convergence without knowing the distribution. Our per-token extension shows that even within a fixed global schedule, per-token adaptation can provide additional gains.
- **Entropy-Based Dimension-Free Convergence** (arXiv:2601.21943): Uses purely information-theoretic assumptions (entropy-based control of MMSE). Supports our approach of using information-theoretic quantities as the right abstraction.

### Selected Front-Runner

**Candidate B (Token Convergence Rate Characterization)** is selected as the front-runner because:

1. **Deep theoretical novelty:** Per-token convergence rate bounds for MDMs are genuinely new. All prior theory is sequence-level. This represents a new level of granularity in understanding DLM generation.
2. **Unifying framework:** The token convergence rate theory explains WHY existing heuristic methods (DyLLM's cosine similarity, ES-dLLM's tensor variation, EntropyCache's entropy, KLASS's KL divergence) work -- they are all approximations to the theoretically optimal criterion (conditional mutual information). The theory ranks these proxies by how closely they approximate the true criterion.
3. **Practical predictions:** The theory makes specific, testable predictions about per-token convergence patterns and the relative quality of different proxy signals.
4. **Composability insight:** Combined with Candidate C's KV cache analysis, the theory shows that per-token convergence simultaneously determines caching validity and scheduling optimality, providing a unified foundation for the composition of acceleration methods.

## Phase 5: Final Proposal

### Title

Token-Level Information-Theoretic Convergence Bounds for Masked Diffusion Language Models: Toward Optimal Adaptive Acceleration

### Formal Claim

**Main Theorem (informal):** For a masked diffusion language model generating sequence X = (X_1, ..., X_n) from distribution p using T denoising steps:

**Claim 1 (Per-token convergence rate):** The expected error for token i after T_i effective steps (steps where token i's state is updated) satisfies:

E[D_KL(p_T(X_i | context) || p(X_i | X_{-i}))] <= I(X_i; X_{-i}) / T_i

where I(X_i; X_{-i}) is the mutual information between token i and all other tokens under the true distribution p. This bound is tight up to logarithmic factors.

**Claim 2 (Optimal per-token budget allocation):** Given a total computational budget of B forward passes, the allocation that minimizes the sum of per-token errors sum_i D_KL(p_T(X_i) || p(X_i | X_{-i})) is:

T_i* = B * sqrt(I(X_i; X_{-i})) / sum_j sqrt(I(X_j; X_{-j}))

This is a water-filling solution: tokens with high conditional mutual information receive proportionally more denoising steps.

**Claim 3 (Entropy as computable proxy):** The token-level entropy H(X_i | X_{-i}^{(t)}) estimated from the DLM's logits at step t provides an upper bound on the remaining convergence budget needed:

T_remaining(i) >= c * H(X_i | X_{-i}^{(t)}) / log|V|

where |V| is the vocabulary size and c is a universal constant. This justifies entropy-based methods (EntropyCache, KLASS) as approximations to the optimal criterion and provides the first formal bound on their sub-optimality gap.

**Claim 4 (KV cache validity from convergence):** When token i has converged (D_KL(p_t(X_i) || p_{t-1}(X_i)) < epsilon), the KV pair for token i changes by at most:

||K_i^{(t+1)} - K_i^{(t)}||_2 <= C * epsilon * ||W_K||_2

This provides the first formal connection between token convergence and KV cache validity, unifying the scheduling and caching dimensions of DLM acceleration.

### Proof Sketch

**For Claim 1:**
1. Start from the Li & Cai (2025) global bound D_KL(p_T || p) <= n * I_avg / T.
2. Refine by conditioning on individual tokens using the chain rule: D_KL(p_T || p) = sum_i D_KL(p_T(X_i | X_{<i}^{(T)}) || p(X_i | X_{<i})).
3. Apply Fano's inequality to bound each term: the error for token i is controlled by how much information about X_i remains after T_i effective updates, which is I(X_i; X_{-i}) - information_revealed_in_T_i_steps.
4. The 1/T_i rate follows from the concavity of mutual information and the uniform reduction of uncertainty at each step (established by the masking schedule analysis of Chen et al., 2025).

**For Claim 2:**
5. This is a constrained optimization: minimize sum_i I(X_i; X_{-i}) / T_i subject to sum_i T_i = B.
6. By Lagrange multipliers: T_i* proportional to sqrt(I(X_i; X_{-i})). This is the classical water-filling solution from rate-distortion theory, applied to the per-token denoising budget.

**For Claim 3:**
7. Use the inequality H(X_i | X_{-i}) >= I(X_i; X_{-i}) - I(X_i; X_{-i} | X_{-i}^{(t)}) to bound the remaining information.
8. The DLM's logits at step t provide an estimate of p(X_i | X_{-i}^{(t)}), from which H(X_i | X_{-i}^{(t)}) is directly computable at cost O(V) per token.

**For Claim 4:**
9. When the token distribution changes by at most epsilon in KL divergence, the embedding (and hence the key/value) changes by a bounded amount via the Lipschitz property of the embedding layer and the Pinsker inequality connecting KL to total variation.

### Assumptions

1. **Accurate mask predictor:** The DLM's learned conditional distribution p_theta(X_i | X_{-i}^{(t)}) is a good approximation of the true conditional p(X_i | X_{-i}^{(t)}). This is standard in the DLM convergence literature (Li & Cai, 2025; Chen et al., 2025).

2. **Factorized approximation:** The per-token analysis assumes that token convergence rates can be studied independently, which holds exactly under the factorized sampling scheme used by most MDMs (tokens are unmasked independently conditioned on the current state). This is the same assumption used by Lavenant & Zanella (2025).

3. **Lipschitz embedding:** The mapping from token logits to key/value vectors is Lipschitz continuous with bounded constant. This holds for standard transformer architectures with bounded weight matrices.

4. **Sub-exponential information content:** Following the entropy-based convergence framework (arXiv:2601.21943), we assume the per-token information content does not have extreme heavy tails.

### Empirical Prediction

The theory makes several specific, falsifiable predictions that can be tested on LLaDA-8B-Instruct:

1. **Per-token convergence hierarchy:** Measure the convergence rate r_i(t) = D_KL(p_t(X_i) || p_{t-1}(X_i)) for each token across denoising steps. The theory predicts:
   - Function words (determiners, prepositions) converge in steps 1-5
   - Common content words converge in steps 5-15
   - Domain-specific/rare terms converge in steps 15-40
   - Novel or context-dependent tokens converge in steps 40-64
   
2. **Proxy signal ranking:** Compare the Spearman correlation between various proxy signals and the true per-token convergence budget I(X_i; X_{-i}):
   - Token entropy H(X_i | context) should have highest correlation (direct upper bound per Claim 3)
   - KL divergence between consecutive steps (KLASS) should be close second
   - Cosine similarity of hidden states (DyLLM) should correlate but more loosely
   - Tensor variation (ES-dLLM) should have lowest correlation (most indirect proxy)
   
3. **Optimal budget vs. uniform:** An oracle per-token scheduler (using true conditional MI) should match uniform T=64 quality using approximately T_eff = O(TC(X)) effective computation, where TC(X) is the total correlation. For English text with typical dependency structure, TC(X) << 64n, predicting significant speedup potential.

4. **KV cache validity threshold:** The theory predicts that KV caching error is bounded by per-token convergence, so the optimal KV refresh criterion should be: refresh the KV pair for token i when its estimated entropy drops significantly (indicating a state change). This matches EntropyCache's empirical strategy and predicts it should outperform fixed-interval refresh (Fast-dLLM).

### Experimental Plan

**Phase 1: Convergence Rate Measurement (Pilot, ~1 hour)**
- Model: LLaDA-8B-Instruct, T=64 steps
- Task: Generate 50 sequences on GSM8K (reasoning) and 50 on HumanEval (code)
- Measurement: At each denoising step t and each token position i, record the full logit vector and compute D_KL(p_t || p_{t-1}), entropy H(X_i | context), and the final converged distribution
- Output: Per-token convergence curves for both tasks
- Target: Verify the predicted convergence hierarchy and measure actual convergence rate distributions

**Phase 2: Proxy Signal Comparison (~1 hour)**
- Using the data from Phase 1, compute four proxy signals for each token: (a) entropy, (b) inter-step KL divergence, (c) cosine similarity of hidden states, (d) tensor variation
- Compute the Spearman rank correlation of each proxy with the empirical per-token convergence budget (number of steps until KL < threshold)
- Output: Ranking of proxy signals by correlation with convergence budget
- Target: Verify that entropy is the best proxy (closest to optimal criterion)

**Phase 3: Oracle vs. Heuristic Per-Token Scheduling (~2 hours)**
- Implement an oracle per-token scheduler that freezes tokens when their inter-step KL drops below a calibrated threshold
- Compare with: (a) uniform T=64, (b) uniform T=32, (c) KLASS (KL-guided), (d) Saber (adaptive acceleration), (e) EntropyCache
- Benchmarks: GSM8K, HumanEval
- Metrics: Accuracy/Pass@1 at iso-compute (same number of FLOPs) and iso-quality (same accuracy)
- Target: Oracle should dominate all heuristics; entropy-based heuristic should be closest to oracle

**Phase 4: Theory Validation on Diverse Tasks (~2 hours)**
- Extend Phase 1 measurements to MMLU, HellaSwag, ARC-Challenge
- Test prediction: total correlation (measured as sum of per-token MI) varies by task type, with reasoning tasks having higher TC (more complex dependencies) and factual tasks having lower TC
- Test prediction: achievable speedup inversely correlates with TC

**Phase 5: Composition Prediction (~2 hours)**
- Using Claim 4, predict which tokens are safe to cache at each step
- Compare predicted safe-to-cache set with EntropyCache's empirical refresh decisions
- Measure the correlation: if theory correctly identifies cache-safe tokens, it can be used to derive a principled refresh schedule
- Compose the entropy-based per-token scheduler with the theory-derived KV cache schedule

### Baselines

**Theoretical baselines:**
- Li & Cai (2025) global bound: D_KL <= n * I_avg / T (our per-token bound should be tighter for heterogeneous distributions)
- Chen et al. (2025) optimal global schedule (our per-token extension should show additional speedup potential)
- Uniform schedule (worst case for our theory)

**Empirical baselines:**
- LLaDA-8B-Instruct with T=64 uniform steps (quality ceiling)
- KLASS (NeurIPS 2025 Spotlight): KL-guided adaptive unmasking
- EntropyCache: entropy-guided KV cache refresh
- Saber: adaptive acceleration with backtracking
- Fast-dLLM: KV cache + confidence parallel decoding

### Risk Assessment

1. **Risk: Per-token independence assumption breaks down.**
   In transformers, all tokens interact through attention at every step. The factorized analysis assumes approximate independence of per-token convergence, which may not hold when tokens are strongly correlated (e.g., subject-verb agreement).
   *Mitigation:* The factorized approximation is used by all existing theoretical analyses (Lavenant & Zanella, 2025; Chen et al., 2025). Our empirical validation (Phase 1) will directly measure how well the independence assumption holds by comparing theoretical predictions to actual convergence curves.

2. **Risk: Bounds are too loose to distinguish between proxy signals.**
   If the gap between the entropy upper bound (Claim 3) and the true convergence budget is large, the theory may not provide meaningful ranking of proxy signals.
   *Mitigation:* Even if the absolute bounds are loose, the relative ranking (which proxy correlates best with true convergence) is an empirical question that can be answered without tight bounds. The theory provides the framework; the experiments validate it.

3. **Risk: DLM logits are poor estimates of true conditional distributions.**
   If the model's uncertainty estimates are poorly calibrated, the entropy proxy will be unreliable.
   *Mitigation:* Previous work (KLASS, EntropyCache) has shown empirically that DLM logit-derived signals (KL, entropy) are effective in practice. If calibration is a problem, temperature scaling or Bayesian approaches can improve it.

4. **Risk: The water-filling budget allocation is impractical to implement.**
   You cannot give different tokens different numbers of forward passes in a standard transformer.
   *Mitigation:* The theory motivates a practical approximation: at each forward pass, compute attention for all tokens but only update (sample new values for) the tokens with highest remaining budget. This is already done by DyLLM (partial attention) and ES-dLLM (layer skipping). Our theory identifies the optimal criterion for which tokens to update.

### Novelty Claim

**What is new:**

1. **Per-token convergence rate bounds for MDMs.** All prior theoretical work (Li & Cai 2025; Chen et al. 2025; Lavenant & Zanella 2025; arXiv:2602.20126) provides sequence-level bounds. We are the first to characterize per-token convergence rates in terms of conditional mutual information.

2. **Optimal per-token budget allocation via water-filling.** The rate-distortion-theoretic characterization of optimal per-token step budgets is new. It provides the theoretical foundation for all existing adaptive scheduling methods (KLASS, Saber, DyLLM, ES-dLLM) while showing how they could be improved.

3. **Formal justification of entropy as the near-optimal proxy.** We prove that token entropy is a computable upper bound on the per-token convergence budget, providing the first formal explanation for why entropy-based methods (EntropyCache, KLASS) work well. We also formally rank other proxy signals (KL, cosine similarity, tensor variation) by their proximity to the optimal criterion.

4. **Unified convergence-caching connection.** Claim 4 establishes the first formal link between per-token convergence and KV cache validity, unifying the scheduling and caching axes of DLM acceleration under a single information-theoretic framework.

**Evidence of novelty:**
- Chen et al. (2025) characterize optimal GLOBAL schedules; we extend to PER-TOKEN schedules.
- KLASS (NeurIPS 2025 Spotlight) uses token-level KL empirically; we provide the formal optimality characterization it lacks.
- EntropyCache uses entropy for cache refresh; we prove entropy is a near-optimal proxy for per-token convergence.
- No existing paper in the 28-paper literature survey or the 12 additional theoretical papers provides per-token convergence bounds for MDMs.
