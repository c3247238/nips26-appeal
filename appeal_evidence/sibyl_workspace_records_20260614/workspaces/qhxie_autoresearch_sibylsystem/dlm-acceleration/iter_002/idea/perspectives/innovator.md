# Innovator Perspective

## Phase 1: Literature Survey

### Key Papers Found

1. **DAWN: Dependency-Aware Fast Inference for Diffusion LLMs** (arXiv:2602.06953, Feb 2026) -- First work to explicitly model inter-token dependencies during parallel decoding in dLLMs; extracts coupling structure to guide which tokens can be safely unmasked in parallel. Directly relevant because it shows that dependency modeling is a viable lever for acceleration, but does not exploit the temporal structure of how dependencies evolve across denoising steps.

2. **Swordsman: Entropy-Driven Adaptive Block Partition** (arXiv:2602.04399, Feb 2026) -- Uses entropy shifts between adjacent tokens to adaptively align block boundaries with semantic/syntactic constituents. Demonstrates that entropy is a reliable proxy for constituent structure, but operates only at the block-partition level and does not exploit within-step token convergence dynamics.

3. **ReMix: Rejection Mixing for Fast Semantic Propagation** (arXiv:2602.22868, Feb 2026) -- Introduces a "continuous mixing state" as an intermediate between masked and decoded tokens, allowing continuous-space refinement before discrete collapse. Key insight: the combinatorial contradiction problem in parallel decoding can be mitigated by operating in a continuous intermediate space. Achieves 2-8x speedup training-free.

4. **CDLM: Consistency Diffusion Language Models** (arXiv:2511.19269, Nov 2025) -- Adapts consistency models from image diffusion to discrete DLMs, distilling multi-step trajectories into few-step jumps. Achieves 20.9x speedup on MBPP-Instruct. Critical reference: shows that consistency distillation works for discrete diffusion, but requires training.

5. **Jump Your Steps (JYS): Optimizing Sampling Schedule of Discrete Diffusion Models** (ICLR 2025) -- Derives a practical upper bound on Compounding Decoding Error (CDE) and proposes an efficient algorithm for optimal non-uniform timestep allocation. Works across image, music, and text. Key theoretical tool for our proposal.

6. **DiCo: Divide and Conquer for Adaptive Parallel Decoding** (arXiv:2602.23792, Feb 2026) -- Discovers that dLLM decoding trajectories naturally exhibit clustering behavior with strong intra-cluster and weak inter-cluster token dependencies. Exploits this via divide-and-conquer parallel decoding. Key insight: local dependency structure is exploitable without global coordination.

7. **DualDiffusion: Speculative Decoding for Masked Diffusion Models** (arXiv:2604.05483, Apr 2026) -- First proper draft-verify framework for pure MDMs (not block-diffusion). Uses a lightweight drafter with efficient approximations + full verifier. Shows the Pareto frontier between steps and accuracy can be pushed. Very recent; limited empirical scope.

8. **Self Speculative Decoding (SSD) for dLLMs** (arXiv:2510.04147, withdrawn from ICLR 2026) -- Uses the dLLM itself as both drafter and verifier via hierarchical verification trees. Achieves 3.46x lossless speedup on LLaDA and Dream. Withdrawn, suggesting potential issues, but the core self-speculation idea for pure MDMs is sound.

9. **Not All Denoising Steps Are Equal: Model Scheduling** (arXiv:2604.02340, Apr 2026) -- Shows that middle denoising steps are most sensitive to model capacity while early/late steps tolerate a smaller model. Achieves only 17% FLOP reduction with modest quality drop. Suggests that a more aggressive temporal decomposition of the denoising process is needed.

10. **Loop-Residual Neural Networks for Iterative Refinement** (arXiv:2409.14199) -- Demonstrates that iteratively looping over a subset of layers with residual connections improves performance proportional to computation time. Key cross-domain insight: adaptive depth via looping is more parameter-efficient than simply stacking more layers.

11. **LayerSkip: Enabling Early Exit and Self-Speculative Decoding** (Meta, 2024-2026) -- For AR models, early layers draft and remaining layers verify. Bridges early exit with self-speculative decoding. 2.16x speedup on summarization. The layer-level decomposition principle transfers to DLMs but requires rethinking for bidirectional attention.

12. **Predictive Coding in Neural Networks** (PLOS Complex Systems 2025, bioRxiv 2025) -- Brain's hierarchical predictive coding: higher areas generate predictions for lower areas; only prediction errors propagate upward. This top-down iterative refinement with error-driven updates is structurally analogous to DLM denoising, where each step refines predictions and only "surprising" (low-confidence) tokens need reprocessing.

### Landscape Summary

The DLM inference acceleration field has exploded in early 2026, with over 20 training-free methods published in January-March alone. The landscape can be organized along three orthogonal axes: (1) **KV cache approximation** (Fast-dLLM, dKV-Cache, Elastic-Cache, EntropyCache, Window-Diffusion) -- deciding when to recompute vs. reuse cached attention; (2) **parallel decoding strategies** (confidence-based, dependency-aware, entropy-driven, adaptive block partition) -- deciding which tokens to unmask simultaneously; and (3) **step reduction** (model scheduling, consistency distillation, optimal timestep allocation) -- deciding how many forward passes to run and how to allocate model capacity across them.

A critical gap emerges at the intersection of these axes: no existing work provides a unified framework that jointly optimizes WHAT to cache, WHICH tokens to process in parallel, and HOW MANY steps to use -- adapting all three decisions dynamically based on the evolving state of the denoising process. Each existing paper optimizes one axis while holding the others fixed or using simple heuristics. This is analogous to the early days of AR model serving, where KV caching, batching, and continuous decoding were developed independently before being unified in systems like vLLM.

Furthermore, a deeper theoretical gap exists: the existing methods lack a principled framework for understanding when and why certain tokens converge faster than others during denoising. The predictive coding framework from neuroscience offers an elegant analogy: in the brain, top-down predictions suppress redundant processing of predictable inputs, and only prediction errors (surprises) drive further computation. An analogous principle for DLMs would allocate denoising computation proportionally to token-level "surprise" -- tokens that are predictable from context should converge in fewer steps with cached attention, while genuinely novel or context-dependent tokens require full attention recomputation across more steps.

## Phase 2: Initial Candidates

### Candidate A: Predictive-Coding-Inspired Hierarchical Denoising (PC-Denoise)

- **Hypothesis**: A training-free DLM inference framework that decomposes the denoising process into a hierarchy of "predictive" and "corrective" passes -- where a cheap predictive pass estimates token states and only tokens with high prediction error receive full corrective computation -- will achieve superior speed-quality tradeoffs compared to methods that treat all tokens uniformly at each step.

- **Cross-domain insight**: From neuroscience's predictive coding theory. The brain does not process all sensory inputs equally at every hierarchical level. Instead, top-down predictions suppress processing of expected inputs, and only prediction errors propagate upward for further processing. The denoising process in DLMs has an analogous structure: at each step, most tokens' states are well-predicted by their state at the previous step (justifying KV caching), while a minority of tokens undergo significant state changes (requiring full recomputation). The key transplant is: rather than using a fixed proxy signal (entropy, cosine similarity, tensor variation) to identify which tokens need recomputation, use the actual prediction error between a cached (predicted) forward pass and a selective full forward pass as the signal, with the error itself determining the scope of the next corrective pass.

- **Evidence for**: (1) DyLLM, ES-dLLM, and EntropyCache all independently demonstrate that token-level importance varies dramatically across steps, with only 10-30% of tokens requiring recomputation at typical steps. (2) Loop-Residual Networks show that iterative refinement with residual connections improves quality proportional to compute. (3) The predictive coding literature shows that prediction-error-driven processing is more efficient than uniform processing in hierarchical systems.

- **Novelty estimate**: 5/10 -- The individual components (cached prediction, selective recomputation, token-level importance) exist in various forms. The novelty is in the principled integration via the predictive coding framework and the self-referential error signal (using prediction error to determine recomputation scope). Several works (DyLLM, ES-dLLM) are close neighbors.

### Candidate B: Temporal Consistency Verification for Training-Free Step Compression (TCV)

- **Hypothesis**: By treating consecutive denoising steps as a "draft" sequence and using the DLM's own forward pass at a later timestep as a verifier, we can skip multiple intermediate steps when the draft sequence is temporally consistent (token distributions change minimally), achieving self-speculative step compression without a separate draft model and without any training.

- **Cross-domain insight**: From speculative decoding in AR models, but with a crucial twist. In AR speculative decoding, a small model drafts tokens and a large model verifies them in parallel. In DLMs, we speculate across TIME rather than across TOKENS -- we hypothesize that the token distributions at step t+k are approximately equal to those at step t (temporal consistency), and verify this by running a single forward pass at step t+k. If verification succeeds, steps t+1 through t+k-1 are skipped. If it fails, we fall back to fine-grained stepping only for the failed tokens. The key insight is that DLMs have a unique property AR models lack: at any point, we can evaluate the full model on the current state and compare it to a predicted future state, because the model operates on the entire sequence simultaneously.

- **Evidence for**: (1) Model scheduling (arXiv:2604.02340) shows that early and late steps are robust to capacity reduction, implying high temporal consistency at those phases. (2) CDLM demonstrates that consistency distillation can compress multiple steps into one, but requires training. (3) JYS (ICLR 2025) provides theoretical tools for optimal non-uniform timestep allocation based on compounding decoding error bounds. (4) The observation that KV caches work well for DLMs (Fast-dLLM, EntropyCache) implies high inter-step similarity, which is exactly the temporal consistency we propose to exploit at the token distribution level rather than the KV level.

- **Novelty estimate**: 8/10 -- Self-speculative decoding exists for block-diffusion models (S2D2) but not for pure MDMs like LLaDA/Dream. The concept of speculating across time steps rather than across model capacity is genuinely new. DualDiffusion uses a separate draft model; SSD was withdrawn. The key differentiator is that our approach requires no modification to the model, no separate draft model, and directly exploits temporal consistency -- a property unique to iterative denoising processes.

### Candidate C: Information-Geometric Step Scheduling via Token Convergence Manifolds (IGSM)

- **Hypothesis**: The optimal denoising schedule for a DLM can be determined by tracking the information-geometric curvature of each token's probability trajectory through the denoising process. Tokens whose probability distributions traverse high-curvature regions of the probability simplex require more denoising steps (fine temporal resolution), while tokens on low-curvature trajectories can be resolved with fewer steps. A training-free algorithm that estimates this curvature online and adapts the per-token step schedule accordingly will outperform both uniform schedules and existing adaptive methods.

- **Cross-domain insight**: From information geometry and optimal control theory. In information geometry, the Fisher information metric on probability distributions defines a natural Riemannian manifold. The "difficulty" of a denoising trajectory can be quantified by the arc length along this manifold -- high curvature means rapid distributional changes that require fine-grained steps to track accurately. This is analogous to adaptive step-size control in ODE solvers, where the step size shrinks when the solution changes rapidly and expands when it is smooth. The transplant: apply adaptive step-size principles from numerical ODE integration to the discrete denoising schedule, using the token-level Fisher information (approximated cheaply from logits) as the curvature signal.

- **Evidence for**: (1) JYS (ICLR 2025) shows that non-uniform timestep allocation reduces compounding decoding error. (2) The model scheduling paper shows that step sensitivity is non-uniform and peaks at intermediate noise levels. (3) In image diffusion, adaptive step-size solvers (DPM-Solver, DDIM with non-uniform schedules) are well-established and provide large speedups. (4) EntropyCache demonstrates that token-level entropy (closely related to Fisher information for categorical distributions) is a cheap and effective proxy signal.

- **Novelty estimate**: 7/10 -- Adaptive step scheduling exists (Saber, PRR), but all use heuristic schedules. The information-geometric formulation with per-token curvature estimation is new for discrete diffusion. The connection to ODE adaptive step-size control provides theoretical grounding that existing methods lack. However, the practical challenge is that Fisher information estimation on discrete distributions may be noisy.

## Phase 3: Self-Critique

### Against Candidate A (PC-Denoise)

- **Prior work attack**: Searching for "predictive coding diffusion language model" and "prediction error guided token selection diffusion" reveals no direct prior work combining predictive coding with DLM inference. However, DyLLM (cosine similarity between adjacent steps), ES-dLLM (tensor variation), and EntropyCache (decoded token entropy) all implement variants of "prediction error" for selecting which tokens to recompute. The conceptual framing as predictive coding is novel, but the mechanism is close to an incremental improvement over DyLLM/ES-dLLM with a different proxy signal. Swordsman's entropy-driven partitioning is also thematically similar.

- **Methodological attack**: The two-pass architecture (cheap prediction + selective correction) adds overhead for the prediction pass. If most tokens require recomputation anyway (e.g., early denoising steps with high noise), the cheap pass is wasted computation. The method's advantage depends on a large fraction of tokens being "predictable" -- this holds for late denoising steps but may not for early ones.

- **Theoretical attack**: The predictive coding analogy is appealing but potentially superficial. In the brain, predictive coding operates across a spatial hierarchy (V1 -> V2 -> IT). In DLMs, the "hierarchy" is temporal (step t -> step t+1). The structural correspondence is loose -- DLM layers are not organized as a sensory processing hierarchy. The "prediction error" in DLMs (change in token logits between steps) may not have the same information-theoretic properties as cortical prediction errors.

- **Scalability attack**: The method adds a selection/routing step at each denoising iteration, which may become a bottleneck for long sequences. The per-token prediction error computation is O(N*V) where V is the vocabulary size, which is non-trivial.

- **Verdict**: MODERATE -- The approach is a reasonable synthesis of existing ideas with a principled framework, but the novelty over DyLLM/ES-dLLM is incremental. The predictive coding framing adds conceptual clarity but may not yield a fundamentally different algorithm.

### Against Candidate B (TCV - Temporal Consistency Verification)

- **Prior work attack**: Searching for "temporal consistency" + "diffusion language model" + "step skipping" yields no direct match. S2D2 (arXiv:2603.25702) is the closest, but it requires block-diffusion architecture (not applicable to pure MDMs). SSD (arXiv:2510.04147) proposes self-speculation for pure MDMs but was withdrawn from ICLR 2026 -- the withdrawal reason is unknown but may indicate fundamental issues. DualDiffusion (arXiv:2604.05483) uses a separate draft model. The specific formulation of speculating across timesteps (predicting that step t+k is approximately equal to step t, then verifying) appears genuinely novel for pure MDMs.

- **Methodological attack**: The verification step requires running a full forward pass at step t+k with the current token states. If verification fails frequently (early denoising, high noise), the method degrades to standard stepping plus overhead. The critical question is: what is the acceptance rate across denoising phases? If it's only high in late phases (where tokens are already mostly converged), the speedup is marginal. Furthermore, the "fallback" for failed tokens needs careful design -- do we recompute all intermediate steps for failed tokens, or just the final one?

- **Theoretical attack**: The temporal consistency assumption is well-motivated for late denoising steps but may break down in the critical middle phase where tokens undergo the most distributional change. The model scheduling paper (arXiv:2604.02340) shows that disagreement between light and heavy models peaks at intermediate noise levels, suggesting low temporal consistency precisely where acceleration would be most valuable.

- **Scalability attack**: The verification pass is a full forward pass, which is the same cost as a regular denoising step. The speedup comes only from skipping intermediate steps. If the average skip length is k, the speedup is at most k/(k+1) -- e.g., skipping 3 steps gives at most 75% fewer forward passes. This is competitive but not transformative compared to KV caching methods that give 10-20x speedups.

- **Verdict**: STRONG -- Despite the methodological concerns, the core idea is genuinely novel for pure MDMs, the theoretical grounding (temporal consistency as a speculative hypothesis, verified by the model itself) is sound, and the approach is composable with KV caching (orthogonal axes of acceleration). The acceptance rate concern can be addressed empirically with adaptive skip lengths. The main weakness is that the speedup ceiling may be lower than KV caching approaches, but the lossless guarantee (via verification) is a significant advantage.

### Against Candidate C (IGSM - Information-Geometric Step Scheduling)

- **Prior work attack**: Searching for "information geometry" + "diffusion language model" + "step schedule" reveals no direct prior work. JYS (ICLR 2025) optimizes timestep allocation but uses compounding decoding error bounds, not information-geometric curvature. Saber uses adaptive acceleration with backtracking. PRR uses a trained convergence controller. The information-geometric formulation is novel, but the practical algorithm may reduce to something similar to JYS or Saber with different proxy signals.

- **Methodological attack**: Computing Fisher information for categorical distributions with vocabulary size V requires the full logit vector -- this is already computed during the forward pass, so the marginal cost is O(N*V) per step (same as entropy). However, the per-token curvature estimation requires comparing Fisher information between consecutive steps, which means we need to store and compare full logit vectors across steps. For N=512 tokens and V=32000, this is 16M floats per step comparison -- manageable but not negligible.

- **Theoretical attack**: The information-geometric framework is elegant but may be over-engineered for the problem. In continuous diffusion (images), the connection to SDEs and the Fisher information metric is well-established. In discrete masked diffusion, tokens live on a discrete simplex and the "trajectory" through this space is not smooth -- the token either stays masked or gets unmasked, with discrete jumps. The continuous geometry formulation may not capture the discrete nature of the process well. The curvature of a trajectory on a discrete simplex is less well-defined than for continuous distributions.

- **Scalability attack**: The per-token adaptive schedule means that different tokens are at different "logical timesteps" during the denoising process. This creates a scheduling complexity: tokens that need more steps must be synchronized with tokens that need fewer. In a batch setting, this heterogeneity reduces parallelism. Existing hardware is optimized for uniform-length computations.

- **Verdict**: MODERATE -- The information-geometric framing is intellectually interesting but the practical algorithm may not differ substantially from entropy-based approaches (EntropyCache, Swordsman). The discrete-vs-continuous tension weakens the theoretical elegance. The heterogeneous per-token scheduling creates implementation complexity.

## Phase 4: Refinement

### Dropped Ideas

- **Candidate A (PC-Denoise)** dropped because: while the predictive coding framing is conceptually interesting, the resulting algorithm is an incremental improvement over DyLLM/ES-dLLM with a different theoretical wrapper. The novelty is primarily in the framing, not the mechanism. In a field with 20+ training-free methods already published in Q1 2026, incremental improvements need much stronger differentiation.

### Strengthened Ideas

- **Candidate B (TCV)** strengthened with the following modifications:
  1. **Adaptive skip length**: Rather than a fixed skip parameter k, use a lightweight heuristic to estimate the expected temporal consistency based on the current denoising phase and token-level entropy distribution. Early phases (high noise) use short skips or no skips; late phases use aggressive skips.
  2. **Token-level selective verification**: Instead of verifying the entire sequence at step t+k, only verify tokens whose predicted state differs significantly from their state at step t (using logit entropy change as a cheap proxy). This reduces verification cost and improves acceptance rates.
  3. **Composition with KV caching**: TCV operates on the step-scheduling axis while KV caching operates on the per-step computation axis. These are orthogonal -- TCV reduces the number of forward passes while KV caching reduces the cost of each forward pass. We can compose TCV with EntropyCache or Fast-dLLM for multiplicative speedups.

- **Candidate C (IGSM)** strengthened by simplifying the theoretical framework:
  1. Replace the full information-geometric curvature with a simpler **logit divergence rate** metric: the KL divergence between token distributions at consecutive steps, normalized by step interval. This is cheaper to compute and avoids the continuous-vs-discrete tension.
  2. Frame the algorithm as an instance of the broader "adaptive step-size ODE solver" paradigm, making it accessible to the broader community.

### Additional Evidence Found

- **SchED: Progress-Aware Confidence Schedules** (arXiv:2512.02892): Proposes smooth confidence schedules that relax thresholds over denoising progress, achieving training-free early exit for individual tokens. This validates the idea that temporal dynamics of confidence are exploitable for step reduction.
- **I-DLM: Introspective Diffusion Language Models** (arXiv:2604.11035): Proposes "introspective strided decoding" that simultaneously generates new tokens and revises prior ones. Shows that DLMs can look backward and forward in the same pass, validating our temporal verification concept.
- **DEER: Draft with Diffusion, Verify with AR** (arXiv:2512.15176): First speculative decoding using a dLLM as drafter without AR models. Shows that pure DLM-based draft-verify is feasible.

### Selected Front-Runner

**Candidate B (Temporal Consistency Verification -- TCV)** is selected as the front-runner because:

1. **Genuine novelty**: Self-speculative step compression for pure MDMs (LLaDA, Dream) without a separate draft model or block-diffusion architecture is an open problem. S2D2 only works for block-diffusion; SSD was withdrawn; DualDiffusion needs a separate drafter. TCV fills this specific gap.
2. **Principled mechanism**: The temporal consistency hypothesis is well-motivated by empirical observations (KV cache effectiveness implies inter-step similarity) and can be verified at inference time without additional training.
3. **Composability**: TCV is orthogonal to KV caching, parallel decoding, and other existing acceleration methods, enabling multiplicative speedup stacking.
4. **Training-free**: Meets the project constraint of no model modification.
5. **Falsifiable**: The hypothesis (temporal consistency enables step skipping with verifiable quality preservation) can be directly tested by measuring acceptance rates and quality degradation.

## Phase 5: Final Proposal

### Title

Temporal Consistency Verification: Training-Free Self-Speculative Step Compression for Masked Diffusion Language Models

### Hypothesis

In masked diffusion language models (MDMs), the token probability distributions at denoising step t and step t+k (for moderate k) are sufficiently similar during early and late denoising phases that intermediate steps can be skipped without quality loss, provided a verification pass at step t+k confirms temporal consistency. A training-free algorithm that adaptively determines skip lengths based on estimated temporal consistency, verifies skipped steps via a single forward pass, and falls back to fine-grained stepping only for tokens that fail verification, will achieve 2-4x step reduction (translating to 1.5-3x wall-clock speedup) with negligible quality degradation on LLaDA-8B-Instruct, and this speedup will compound multiplicatively with existing KV caching methods.

### Motivation

The DLM acceleration field has made enormous progress on reducing the cost of individual forward passes (via KV caching, partial attention, token pruning), but relatively less progress on reducing the NUMBER of forward passes needed. Model scheduling (arXiv:2604.02340) shows that not all steps are equally important, but achieves only 17% FLOP reduction. Consistency distillation (CDLM) achieves dramatic step reduction but requires training. Self-speculative decoding (S2D2) works for block-diffusion but not for the most widely-used pure MDMs (LLaDA, Dream).

The key literature gap is: **there is no training-free method for reducing the number of denoising steps in pure MDMs that provides quality guarantees**. TCV addresses this gap by borrowing the draft-verify paradigm from AR speculative decoding, but applying it across the temporal dimension of the denoising process rather than across model capacity.

### Method

**Core Algorithm: Temporal Consistency Verification (TCV)**

Given a masked diffusion LM $M$, total denoising steps $T$, and a sequence of partially denoised tokens $x_t$ at step $t$:

1. **Predict**: Estimate the token distributions at step $t+k$ by assuming temporal consistency: $\hat{p}_{t+k}(x_i) \approx p_t(x_i)$ for all positions $i$. The "draft" is simply the current state $x_t$ projected forward by $k$ steps without intermediate computation.

2. **Verify**: Run a single forward pass of $M$ at the noise level corresponding to step $t+k$ on the current token states $x_t$ to obtain the actual distributions $p_{t+k}(x_i)$.

3. **Accept/Reject**: For each token $i$, compute the divergence $D_i = \text{KL}(p_{t+k}(\cdot | x_t) \| p_t(\cdot | x_t))$ or a cheaper proxy (max-probability change, entropy change). If $D_i < \tau$ for a threshold $\tau$, accept the skip for token $i$. If $D_i \geq \tau$, mark token $i$ as requiring intermediate computation.

4. **Selective Fallback**: For tokens that fail verification, perform an intermediate step at $t + k/2$ (bisection strategy) and re-verify. This creates an adaptive resolution schedule where "easy" tokens skip many steps and "hard" tokens are refined at fine granularity.

5. **Adaptive Skip Length**: The initial skip length $k$ is determined by:
   - **Phase detection**: Estimate the denoising phase (early/middle/late) from the fraction of unmasked tokens.
   - **Entropy profile**: Compute the mean token entropy. Low entropy (late phase) -> large $k$; high entropy (early phase) -> small $k$ or $k=1$.
   - **Historical acceptance rate**: If the last verification had high acceptance, increase $k$; if low, decrease $k$.

6. **Composition with KV Cache**: TCV operates at the step level; KV caching operates within each step. For each forward pass that TCV decides to execute, EntropyCache or Fast-dLLM can be applied to reduce the per-pass cost. The speedups multiply: if TCV reduces steps by 2.5x and KV caching reduces per-step cost by 10x, total speedup is 25x.

**Theoretical Justification**: The temporal consistency hypothesis is grounded in the observation that the denoising process follows a smooth trajectory in distribution space (this is exactly what makes KV caching work -- if distributions changed dramatically between steps, cached KV pairs would be useless). TCV exploits this smoothness at a coarser granularity: instead of reusing individual KV pairs, it reuses entire distributional states across multiple steps.

### Cross-domain Insight

The transplanted principle is **speculative execution with verification**, borrowed from two domains:

1. **AR speculative decoding**: A fast model speculates, a slow model verifies. In TCV, the "fast model" is the identity function (predicting that the distribution doesn't change), and the "slow model" is the DLM's own forward pass. This is the cheapest possible "draft" -- zero cost.

2. **Adaptive step-size ODE solvers**: When solving ODEs, the step size is increased when the solution is smooth and decreased when it changes rapidly. The local truncation error (analogous to our KL divergence) determines whether a large step is acceptable. TCV applies this principle to the discrete denoising process.

The structural correspondence holds because: (a) the denoising process is an iterative refinement process with smooth trajectories (empirically verified by the success of KV caching), and (b) the DLM provides a built-in verification oracle (a forward pass at any timestep gives the "correct" distribution for that timestep given the current state).

### Experimental Plan

**Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (secondary)

**Baselines**:
- Vanilla denoising (T=64 steps, no acceleration)
- Fast-dLLM (ICLR 2026, KV cache + parallel decoding)
- EntropyCache (entropy-guided KV refresh)
- DualDiffusion (separate draft-verify)
- Saber (adaptive acceleration + backtracking)
- SlowFast sampling

**Benchmarks**:
- Reasoning: MMLU (5-shot), GSM8K (8-shot), ARC-Challenge
- Code: HumanEval, MBPP
- General: HellaSwag, WinoGrande

**Key Metrics**:
- Wall-clock speedup (tokens/second on A100/H100)
- Quality preservation (accuracy delta vs. vanilla baseline)
- Step reduction ratio (accepted skips / total potential skips)
- Composition speedup (TCV + Fast-dLLM vs. Fast-dLLM alone)

**Ablation Studies**:
1. Skip length strategy: fixed k vs. adaptive k vs. bisection fallback
2. Verification criterion: KL divergence vs. entropy change vs. max-prob change
3. Phase-dependent behavior: measure acceptance rates at early/middle/late denoising phases
4. Threshold sensitivity: vary $\tau$ to trace the Pareto frontier of speed vs. quality
5. Composition study: TCV alone, EntropyCache alone, TCV + EntropyCache, TCV + Fast-dLLM

**Falsification Criteria**: The hypothesis is falsified if:
- Acceptance rates are < 30% even with conservative thresholds (meaning temporal consistency is too weak to exploit)
- Quality degradation exceeds 2% absolute on reasoning benchmarks at 2x step reduction
- The composition with KV caching shows subadditive (rather than multiplicative) speedups

### Resource Estimate

- **Model loading**: LLaDA-8B-Instruct requires ~16GB VRAM (FP16), fits on a single A100/H100
- **Per-experiment**: Each benchmark evaluation (e.g., GSM8K 1319 samples) takes ~30-60 minutes at 64 steps; with acceleration, ~10-20 minutes
- **Total pilot experiments**: ~4 hours (baseline + 3 TCV configurations + 2 composition experiments)
- **Full ablation**: ~12-16 hours across all benchmarks, skip strategies, and composition variants
- **Hardware**: 1-2 GPUs (A100 80GB or equivalent)

### Risk Assessment

1. **Risk: Temporal consistency is too weak in middle denoising phases** -- Mitigation: Use adaptive skip lengths that default to k=1 in the middle phase. Even if TCV only accelerates early and late phases, each phase constitutes ~30-40% of total steps, yielding 1.5-2x speedup.

2. **Risk: Verification overhead negates skip savings** -- Mitigation: The verification pass is exactly one forward pass (same cost as a regular step). Even if only 50% of verifications succeed (skip accepted), the net effect is still a speedup as long as the average skip length exceeds 2. The bisection fallback ensures that failed verifications don't waste more than log2(k) additional passes.

3. **Risk: Token-level selective verification is difficult to implement efficiently** -- Mitigation: Start with sequence-level verification (verify all tokens together) as a simpler baseline. Only move to token-level selection if sequence-level acceptance rates are too low. GPU-friendly masking operations can efficiently select subsets of tokens.

### Novelty Claim

**What is new**: (1) The concept of temporal speculation in DLM denoising -- treating consecutive denoising steps as a "draft" to be verified rather than computed sequentially. (2) A zero-cost draft mechanism (the identity function on token states) that is uniquely available in iterative denoising processes but has no analog in AR generation. (3) A principled composition framework where step-level acceleration (TCV) and per-step acceleration (KV caching) provide multiplicative speedups.

**Evidence of novelty**:
- S2D2 (arXiv:2603.25702) requires block-diffusion architecture; TCV works on pure MDMs
- SSD (arXiv:2510.04147) was withdrawn and uses hierarchical verification trees with non-trivial overhead; TCV uses a simpler verify-or-bisect strategy
- DualDiffusion (arXiv:2604.05483) requires a separate lightweight drafter; TCV requires no additional model
- Consistency distillation (CDLM, arXiv:2511.19269) requires training; TCV is training-free
- Model scheduling (arXiv:2604.02340) replaces the model at certain steps; TCV skips steps entirely
- No existing work in the literature survey (28 papers + 12 additional from web search) proposes temporal consistency verification as a mechanism for step compression in pure MDMs
