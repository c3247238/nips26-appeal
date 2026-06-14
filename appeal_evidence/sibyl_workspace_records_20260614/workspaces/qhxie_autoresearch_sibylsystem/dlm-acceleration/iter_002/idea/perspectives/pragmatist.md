# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **Fast-dLLM (NVlabs/Fast-dLLM)** -- ICLR 2026. Training-free KV cache + confidence-aware parallel decoding. Up to 27.6x speedup on LLaDA/Dream. 564 stars, actively maintained. **Code: fully available, well-documented eval scripts for GSM8K/HumanEval.** This is the strongest open-source baseline and our primary build-on target.

2. **EntropyCache (mscheong01/EntropyCache, arXiv 2603.18489)** -- Training-free KV caching using decoded token entropy as O(V) refresh signal. 15.2x-26.4x speedup on LLaDA-8B and Dream-7B. **Code: available on GitHub.** Clean implementation, straightforward to extend. The only method that achieves lossless accuracy on HumanEval for both models.

3. **SlowFast Sampling (LiangrunFlora/Slow-Fast-Sampling)** -- Training-free dynamic two-stage sampling guided by three principles (certainty, convergence, positional). Up to 34.22x when combined with caching. **Code: available, drop-in for LLaDA/Dream.** Important because it explicitly combines with caching -- but the composition is ad-hoc, not systematically studied.

4. **dLLM framework (ZHZisZZ/dllm)** -- Unified training/inference/evaluation pipeline for DLMs. Reproduces official LLaDA/Dream scores. **Code: available.** Best starting point for standardized evaluation.

5. **LLaDA-8B-Instruct (GSAI-ML on HuggingFace)** -- 8B parameter MDM, NeurIPS 2025 oral. The de facto standard test model for DLM acceleration papers. **Model: freely available.** Fits on a single A100 in FP16.

6. **Dream-7B-Instruct (hkunlp)** -- Second standard test model. **Model: freely available.** Essential for cross-model generalization claims.

7. **"How Efficient Are Diffusion LMs?" (arXiv 2510.18480)** -- Critical evaluation paper. Key finding: acceleration methods show large gains at batch=1 but diminish at larger batch sizes. Proposes Tokens-Per-Forward-Step (TPF) as unified metric. **No code, but methodology is essential for fair evaluation.**

8. **ES-dLLM (arXiv 2603.10088)** -- Training-free early-layer skipping via tensor variation + confidence. 5.6x-16.8x speedup. Operates on a different axis (layer computation reduction) than KV caching or step reduction. **Code status: check repo.** Potentially orthogonal to cache-based methods.

9. **DyLLM (arXiv 2603.08026)** -- Training-free saliency-based token selection + partial attention. Up to 9.6x speedup. Uses cosine similarity between adjacent step hidden states. **Code status: check repo.** Another candidate for composition study.

10. **WeDLM (Tencent/WeDLM)** -- Causal attention DLM with native KV cache compatibility. 3-6x speedup over vLLM-optimized AR baselines. **Code: available on GitHub.** Important reference for "what a well-engineered DLM looks like" but requires fine-tuning from AR models, so not training-free.

11. **DARE (arXiv 2604.04215)** -- Unified post-training + eval framework for dLLMs built on verl + OpenCompass. Covers LLaDA, Dream, SDAR. **Code: available.** Best infrastructure for standardized, reproducible benchmarking across model families.

12. **Window-Diffusion (vhicrgit/Window-Diffusion)** -- Windowed token pruning + sliding cache. Claims up to 99x speedup. **Code: available.** Extreme speedup claim warrants skepticism -- likely sacrifices significant quality at that level.

### Landscape Summary

**What actually works in practice (engineering reality check):**

The field has >20 training-free acceleration papers published in Q1 2026 alone. However, the practical picture is much simpler than it appears:

- **KV caching is the single biggest lever.** Fast-dLLM, EntropyCache, Elastic-Cache, and dKV-Cache all achieve 10-25x speedups by avoiding redundant attention computation. The core insight is the same across all of them: inter-step KV pairs are highly correlated, so you can reuse most of them and only refresh a subset.

- **Parallel decoding is the second lever.** Decoding multiple tokens per step (Fast-dLLM confidence-based, SlowFast certainty-based) gives another 2-5x on top of caching. But there is a hard accuracy-parallelism tradeoff -- pushing more tokens per step almost always costs accuracy.

- **No one has done a proper composition study.** Every paper evaluates its method alone or with at most one other method. SlowFast mentions "combined with caching" but does not systematically vary which cache method. The question "which combinations are additive and which are subadditive?" is completely unanswered.

- **Evaluation is inconsistent.** Papers use different step counts (32 vs 64 vs 128), different benchmarks, different hardware, and critically different batch sizes. The "How Efficient Are DLMs?" paper shows that batch=1 results can be very misleading.

- **The practical gap**: Despite all these papers, there is no single recipe that says "use method X + Y + Z with these hyperparameters on LLaDA-8B to get the best speed-quality tradeoff for reasoning/code tasks." This is the recipe that practitioners actually need.


## Phase 2: Initial Candidates

### Candidate A: "The DLM Acceleration Cookbook" -- A Systematic Composition and Pareto Study of Training-Free Acceleration Methods

- **Hypothesis**: Existing training-free DLM acceleration methods operate on orthogonal axes (KV caching, parallel decoding, step scheduling, token selection) and their speedups compose near-multiplicatively when properly combined. A systematic evaluation of all pairwise and three-way combinations will reveal (a) which pairs are truly orthogonal, (b) the Pareto-optimal combination for each task category, and (c) simple heuristic rules for selecting the best combination given a target speed-quality budget.

- **Implementation sketch**: Start from Fast-dLLM's codebase (best maintained, supports LLaDA/Dream). Integrate EntropyCache's refresh signal as a drop-in replacement for Fast-dLLM's cache refresh. Add SlowFast's two-stage sampling as the step scheduler. Evaluate all {cache method} x {parallel decoding strategy} x {step scheduling} combinations on a standardized benchmark suite.

- **Simplest version**: Just compare Fast-dLLM alone vs. EntropyCache alone vs. SlowFast alone vs. Fast-dLLM+SlowFast vs. EntropyCache+SlowFast on GSM8K and HumanEval with LLaDA-8B-Instruct. 5 configurations, 2 benchmarks, 1 model = 10 runs. Each run ~30min. Total: ~5 hours.

- **Time estimate**: Pilot 2 hours; full study 12-16 hours (adding Dream-7B, more benchmarks, more combinations, ablations).

- **Reusable components**: Fast-dLLM repo (eval scripts, cache implementation), EntropyCache repo (entropy signal), SlowFast repo (sampling strategy), LLaDA-8B-Instruct and Dream-7B-Instruct checkpoints.

### Candidate B: Information-Geometric Step Distillation (IGSD) -- Principled Adaptive Step Scheduling via Logit Divergence

- **Hypothesis**: The optimal number of denoising steps varies dramatically per-token and per-phase, and can be determined training-free by measuring the KL divergence between consecutive-step logit distributions. A simple algorithm that skips steps when KL divergence is below a threshold and refines when it is above will match or exceed Saber/PRR's speed-quality tradeoff while being simpler to implement and more theoretically grounded.

- **Implementation sketch**: Modify the denoising loop in LLaDA's inference code. After each forward pass, compute per-token KL divergence between current and previous step logits. If max KL < threshold, skip the next step. If mean KL > threshold, halve the step interval. This is essentially an adaptive ODE solver applied to the discrete denoising process.

- **Simplest version**: Implement the basic KL-threshold step skipper (no per-token adaptation, just sequence-level). Compare against uniform 64-step, uniform 32-step, and Saber on GSM8K. ~2 hours.

- **Time estimate**: Pilot 2 hours; full ablation (varying thresholds, per-token vs. sequence-level, different benchmarks) 8-10 hours.

- **Reusable components**: LLaDA inference code, standard eval harness. The KL computation adds ~5 lines of code on top of the existing forward pass.

### Candidate C: "Strong Baseline Done Right" -- Properly Tuned Fast-dLLM with Hardware-Aware Profiling

- **Hypothesis**: Fast-dLLM's reported 27.6x speedup uses aggressive settings that sacrifice quality. A carefully tuned Fast-dLLM with conservative hyperparameters, combined with basic engineering optimizations (Flash Attention, torch.compile, optimal batch size selection), can achieve >10x practical speedup with <1% quality degradation on standard benchmarks -- and this "boring" properly-tuned baseline beats most of the fancier methods published in 2026.

- **Implementation sketch**: Take Fast-dLLM's codebase. Profile it with torch.profiler to identify actual bottlenecks (is it memory-bound or compute-bound?). Apply Flash Attention (already standard), torch.compile, and CUDA graph where applicable. Do a systematic hyperparameter sweep of cache refresh rate and confidence threshold. Report the Pareto frontier of speedup vs. quality degradation at each hyperparameter setting.

- **Simplest version**: Run Fast-dLLM with 5 different confidence thresholds on GSM8K, measure both wall-clock speedup and accuracy. Profile one run to identify the bottleneck. ~3 hours.

- **Time estimate**: Pilot 3 hours; full study with Dream-7B, multiple benchmarks, profiling, and optimization 10-12 hours.

- **Reusable components**: Fast-dLLM repo, Flash Attention, torch profiler, standard benchmarks.


## Phase 3: Self-Critique

### Against Candidate A (Composition Study)

- **Implementation reality check**: Integrating multiple methods from different codebases is engineering-heavy. Fast-dLLM, EntropyCache, and SlowFast each modify the denoising loop differently. Making them composable requires understanding and potentially refactoring their internal abstractions. Estimate: 2-3 days of integration work before experiments can start. This is doable but not trivial.

- **Reproducibility attack**: Strong. Each component has open-source code. The evaluation protocol (benchmark suite, metrics, hardware) is well-defined. Other researchers can reproduce by installing the same repos and running the same scripts. The main risk is version/dependency conflicts between repos.

- **Baseline sanity check**: The strongest simple baseline is Fast-dLLM with both cache and parallel decoding enabled (up to 27.6x). Our composition study needs to show that adding EntropyCache's signal or SlowFast's scheduling ON TOP of Fast-dLLM gives meaningful additional speedup without quality loss. If the gain is <10% additive, the paper is thin.

- **Scope attack**: The study covers LLaDA-8B and Dream-7B (standard), reasoning and code benchmarks (standard). This is sufficient scope for a composition study. However, the paper's contribution is empirical/systematic rather than algorithmic, which may be seen as "just an ablation study" by reviewers at top venues.

- **Verdict**: STRONG -- This fills the single biggest practical gap (no composition study exists), uses all-open-source code, and produces immediately actionable results. The risk is that it is perceived as "engineering" rather than "research." Mitigated by: (a) deriving simple rules for when methods compose well, (b) providing theoretical analysis of why certain combinations are orthogonal, (c) building a clean, reusable benchmarking framework.

### Against Candidate B (IGSD)

- **Implementation reality check**: The core algorithm is genuinely simple -- add KL divergence computation after each forward pass and use it to decide step skipping. This is maybe 50 lines of code on top of LLaDA's inference loop. The entropy computation is essentially free since we already have the logits. Implementation risk is very low.

- **Reproducibility attack**: Strong. The algorithm is fully specified by a single threshold parameter. No hidden hyperparameters. Any researcher can implement it from the paper description in an afternoon.

- **Baseline sanity check**: Saber (arXiv 2510.18165) already does adaptive step scheduling with backtracking, achieving 251.4% speedup on code generation. Our approach needs to show clear advantages over Saber -- either simpler (fewer hyperparameters), faster (less overhead), or better quality preservation. The KL-threshold approach is simpler than Saber's two-component (acceleration + backtracking) design, which is a real advantage.

- **Scope attack**: Step scheduling is useful but the speedup ceiling is lower than KV caching. KV caching gives 10-25x; step scheduling gives 2-4x. If the paper only shows 2-3x speedup, it may seem incremental. However, step scheduling is ORTHOGONAL to KV caching, so the real value is in composition -- showing that IGSD + Fast-dLLM gives multiplicative speedup.

- **Verdict**: MODERATE -- The idea is clean and implementable but risks being incremental over Saber/PRR. The information-geometric framing is nice but may not translate to a fundamentally better algorithm. The strongest pitch is as a "simple, principled replacement for heuristic step scheduling" -- but this may not be enough for a top venue as a standalone paper. Better as a component of the composition study (Candidate A).

### Against Candidate C (Strong Baseline)

- **Implementation reality check**: Fast-dLLM is well-maintained and has clear eval scripts. Adding torch.profiler and sweeping hyperparameters is standard ML engineering. This is the lowest-risk candidate.

- **Reproducibility attack**: Perfect. Everything is already open-source and documented.

- **Baseline sanity check**: This IS the baseline. The question is whether "properly tuning an existing method" constitutes a publishable contribution. At top venues, probably not as a standalone paper. But as Section 3 of a composition study, it is essential -- you need to establish what the properly-tuned baseline actually achieves before showing that composition beats it.

- **Scope attack**: A hardware profiling study of DLM inference is genuinely useful for the community and fills Gap 8 (hardware-aware DLM optimization) from the literature survey. But on its own, it reads more like a tech report than a research paper.

- **Verdict**: WEAK as standalone paper, but ESSENTIAL as a component of Candidate A. The profiling insights and properly-tuned baseline make the composition study rigorous rather than superficial.


## Phase 4: Refinement

### Dropped Ideas

- **Candidate C** dropped as standalone proposal. Its content (hardware profiling, hyperparameter sweep) is folded into Candidate A as the "baseline establishment" phase.

### Strengthened Ideas

**Candidate A (Composition Study)** is strengthened by absorbing elements from B and C:

1. **Folded in Candidate C's profiling**: Phase 1 of the composition study is a hardware-aware profiling of each individual method. This establishes the per-method Pareto frontier and identifies the compute/memory bottleneck of each, which informs which combinations are likely to be orthogonal (methods that target different bottlenecks are more likely to compose well).

2. **Folded in Candidate B's IGSD as one of the methods**: The step scheduling axis is represented by IGSD (our simple KL-threshold skipper) and SlowFast. This makes the composition study cover three axes cleanly: (a) KV caching (Fast-dLLM cache vs. EntropyCache), (b) parallel decoding (confidence-based vs. certainty-based), (c) step scheduling (IGSD vs. SlowFast vs. uniform).

3. **Added decomposition analysis**: For each combination, we decompose the total speedup into contributions from each axis. This reveals whether speedups are multiplicative (orthogonal methods), additive (partially overlapping), or subadditive (conflicting optimizations). This decomposition is the key theoretical contribution that elevates the paper above "just an ablation study."

4. **Added practical recommendation tables**: For each task category (reasoning, code, general knowledge), provide a recommended method combination with specific hyperparameters. This is the "cookbook" element that practitioners will actually use.

**Candidate B (IGSD)** is retained as a secondary proposal in case the composition study is too engineering-heavy:

1. **Simplified**: Dropped the "information-geometric" framing (too grandiose for a simple threshold). Renamed to "Logit Divergence Step Scheduling" (LDSS). The algorithm is: compute KL(p_t || p_{t-1}) after each step, skip next step if KL < tau. That is it.

2. **Added composition hook**: LDSS is designed to be composable with any KV caching method. We show LDSS + EntropyCache and LDSS + Fast-dLLM as standard composition experiments.

### Additional Searches Performed

Searched for "composing multiple DLM acceleration methods orthogonal 2026" -- confirmed that no systematic composition study exists. The closest is SlowFast's mention of "combined with caching" and Fast-dLLM's "KV Cache + parallel decoding" combination, but neither systematically explores the full combination space.

Searched for WeDLM (Tencent) -- interesting reference for what a well-engineered DLM looks like (3-6x over vLLM-optimized baselines), but requires fine-tuning from AR models so not applicable to our training-free constraint.

### Selected Front-Runner

**Candidate A: "The DLM Acceleration Cookbook"** is selected because:

1. **Highest practical impact**: Every DLM practitioner needs to know which methods to combine. This paper gives the answer.
2. **Lowest novelty risk**: We are not claiming a new algorithm; we are claiming a new understanding of how existing algorithms interact. This is harder to scoop because it requires significant engineering effort that most individual method authors will not undertake.
3. **Builds on all-open-source code**: Fast-dLLM (564 stars, NVIDIA), EntropyCache, SlowFast, LLaDA, Dream all have available code and models.
4. **Includes Candidate B as a component**: IGSD/LDSS serves as our novel step scheduling method within the study, giving us an algorithmic contribution on top of the empirical one.
5. **Includes Candidate C as a component**: Hardware profiling provides the analytical backbone that explains WHY certain combinations work.
6. **Directly addresses Gap 11** from the literature survey (composition of acceleration methods) and partially Gap 3 (KV cache Pareto frontier) and Gap 8 (hardware-aware profiling).


## Phase 5: Final Proposal

### Title

Composing Training-Free Acceleration Methods for Diffusion Language Models: A Systematic Orthogonality and Pareto Study

### Hypothesis

Training-free DLM acceleration methods that target different computational axes -- KV caching (reducing per-step attention cost), parallel decoding (reducing per-step token iterations), and adaptive step scheduling (reducing total step count) -- are approximately orthogonal and compose near-multiplicatively, yielding combined speedups of 30-50x on LLaDA-8B-Instruct with <2% quality degradation. Furthermore, the optimal combination varies by task type (reasoning vs. code vs. general), and simple heuristic rules can identify the best combination for each task category.

### Motivation

The DLM inference acceleration literature has exploded in 2025-2026, with >20 training-free methods published in Q1 2026 alone. Each paper evaluates its method in isolation or with at most one other method. Practitioners face an overwhelming landscape with no guidance on which methods to use, let alone how to combine them. Meanwhile, the critical evaluation paper "How Efficient Are Diffusion LMs?" (arXiv 2510.18480) shows that reported speedups can be misleading due to inconsistent evaluation setups.

The practical gap is simple: **no one knows which combinations of existing methods actually work together, and no one has measured the Pareto frontier of the combined design space under a unified evaluation protocol.** This paper fills that gap.

### Method

**Phase 1: Individual Method Profiling (establishing the baseline)**

For each method below, we (a) reproduce published results, (b) profile with torch.profiler to identify bottlenecks, and (c) map the speed-quality Pareto frontier by sweeping key hyperparameters:

| Axis | Methods | Key Hyperparameter |
|------|---------|-------------------|
| KV Caching | Fast-dLLM cache, EntropyCache, Elastic-Cache | Cache refresh rate / entropy threshold |
| Parallel Decoding | Fast-dLLM confidence-based, SlowFast certainty-based | Confidence/certainty threshold |
| Step Scheduling | Uniform (T=32,48,64), IGSD (our KL-threshold skipper), Saber | Step count / KL threshold / backtracking strength |

Total: 3 cache methods x 2 parallel decoding strategies x 3 step schedules = 18 individual configurations + baselines.

**Phase 2: Pairwise Composition Analysis**

For each pair of methods from different axes, measure:
- Combined speedup vs. sum-of-individual-speedups (tests additivity)
- Combined speedup vs. product-of-individual-speedups (tests multiplicativity)
- Quality degradation relative to vanilla baseline
- Profile the combined pipeline to identify new bottlenecks

Key pairwise combinations (~10 pairs): {cache x decode}, {cache x schedule}, {decode x schedule}.

**Phase 3: Three-Way Composition and Pareto Frontier**

Combine the best-performing method from each axis. Sweep the combined hyperparameter space (3 axes x 3 settings = 27 configurations per benchmark). Identify the Pareto-optimal configurations for each task category.

**Phase 4: IGSD -- Our Novel Step Scheduling Component**

IGSD (Information-Geometric Step Distillation) is a simple training-free step scheduler:
1. After each forward pass at step t, compute the per-token KL divergence between current logits and previous-step logits.
2. Compute the max and mean KL across all masked tokens.
3. If max KL < tau_skip: skip the next step entirely (temporal consistency holds).
4. If mean KL > tau_refine: halve the step interval for the next segment (distributional shift detected).
5. Otherwise: proceed normally.

IGSD requires storing one extra set of logits (shape [N, V]) and adds O(N*V) computation per step (negligible vs. the forward pass). The threshold tau is set by a simple calibration: run 5 samples and set tau_skip to the 75th percentile of observed inter-step KL values.

**Phase 5: Decomposition Analysis**

For each three-way combination, decompose the total speedup into axis contributions:
- S_total = S_cache * S_decode * S_schedule * (1 - interference_term)
- The interference term quantifies how much the methods conflict. Orthogonal methods have interference ~0; conflicting methods have interference > 0.

This decomposition is derived from profiling data: each axis reduces a specific component of wall-clock time (attention computation, token iterations per step, total steps), and the total speedup is the product of per-component speedups minus any shared overhead.

### Simplest Version

The absolute minimum experiment that tests the core claim (composition of methods yields near-multiplicative speedup):

1. Run Fast-dLLM alone on GSM8K with LLaDA-8B-Instruct -> measure speedup S1 and accuracy A1
2. Run EntropyCache alone on GSM8K -> measure S2, A2
3. Run SlowFast alone on GSM8K -> measure S3, A3
4. Run Fast-dLLM + SlowFast on GSM8K -> measure S_combined, A_combined
5. Compare S_combined vs. S1 * S3 (multiplicativity test)

This is 5 runs, each ~30 minutes. Total: ~2.5 hours. If S_combined >= 0.8 * S1 * S3 with A_combined >= min(A1, A3) - 2%, the core hypothesis is confirmed.

### Baselines

1. **Vanilla LLaDA-8B-Instruct** (T=64 steps, no acceleration): The quality ceiling. Expected: ~52% on MMLU, ~55% on GSM8K, ~30% on HumanEval (from published numbers).

2. **Fast-dLLM (best published config)**: The speed champion. Expected: 10-27x speedup depending on benchmark and quality tolerance.

3. **EntropyCache (published config)**: The quality champion among cache methods. Expected: 15-26x speedup with near-lossless quality.

4. **SlowFast (published config)**: The sampling champion. Expected: up to 34x when combined with basic caching.

5. **Saber (published config)**: The step scheduling champion for code tasks. Expected: 251% speedup on code generation.

### Experimental Plan

**Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (generalization)

**Benchmarks**:
- Reasoning: GSM8K (8-shot), MMLU (5-shot)
- Code: HumanEval (0-shot), MBPP (0-shot)
- General: HellaSwag (10-shot), ARC-Challenge (25-shot)

**Hardware**: Single A100 80GB (standard for DLM inference papers). All experiments at batch=1 (standard) AND batch=8 (following "How Efficient Are DLMs?" recommendation).

**Metrics**:
- Wall-clock tokens-per-second (TPS)
- Accuracy / Pass@1 (task-specific)
- Tokens-Per-Forward-Step (TPF, from "How Efficient Are DLMs?")
- GPU memory peak usage
- Speedup decomposition (cache contribution, decode contribution, schedule contribution)

**Ablation Schedule**:
1. Individual method Pareto frontiers (18 configs x 6 benchmarks x 2 models = ~216 runs, batch parallelizable)
2. Pairwise composition (10 pairs x 6 benchmarks x 2 models = ~120 runs)
3. Three-way composition with IGSD (9 configs x 6 benchmarks x 2 models = ~108 runs)
4. Batch size sensitivity (best 3 configs x 6 benchmarks x 4 batch sizes x 2 models = ~144 runs)

Total runs: ~588. At ~15-30 min per run (accelerated): ~150-300 GPU-hours. This is achievable in 5-10 days on a single A100 with automation, or 2-3 days with 4 GPUs.

**Pilot plan (first 3 hours)**:
1. Set up Fast-dLLM repo, verify it runs on local GPU (30 min)
2. Run Fast-dLLM alone on GSM8K subset (100 samples) -> baseline numbers (20 min)
3. Integrate EntropyCache's refresh signal into Fast-dLLM's cache (60 min)
4. Run combined Fast-dLLM + EntropyCache on GSM8K subset -> first composition data point (20 min)
5. Implement IGSD step skipper in LLaDA's denoising loop (30 min)
6. Run IGSD alone on GSM8K subset -> first novel method data point (20 min)

### Resource Estimate

- **GPU**: 1x A100 80GB minimum, 4x A100 preferred for parallelism
- **Wall-clock (pilot)**: 3 hours for initial validation
- **Wall-clock (full study)**: 5-10 days single GPU, 2-3 days 4 GPUs
- **GPU-hours (full study)**: 150-300 (comparable to training a small model; very reasonable for a systems paper)
- **Model sizes**: LLaDA-8B (~16GB FP16), Dream-7B (~14GB FP16). Both fit comfortably on A100 with room for KV cache
- **Software dependencies**: PyTorch, transformers, flash-attn, Fast-dLLM, EntropyCache, SlowFast repos. All pip-installable or git-clonable

### Risk Assessment

1. **Risk: Method integration is harder than expected** (different repos use different abstractions for the denoising loop)
   - *Likelihood*: Medium
   - *Mitigation*: Use the dLLM framework (ZHZisZZ/dllm) as a common substrate -- it already integrates Fast-dLLM and provides a clean abstraction layer. Re-implement EntropyCache's signal and SlowFast's schedule within this framework rather than trying to merge three separate codebases.

2. **Risk: Compositions are subadditive (methods conflict rather than compose)**
   - *Likelihood*: Low-Medium (KV cache + step scheduling should be orthogonal since they target different aspects of computation)
   - *Mitigation*: This is actually a publishable result either way. If compositions are subadditive, we analyze WHY and identify the interference mechanism. "Here is why you cannot combine X and Y" is as useful as "here is how to combine X and Y."

3. **Risk: Reproducing published baselines fails** (environment differences, undocumented hyperparameters)
   - *Likelihood*: Low for Fast-dLLM (well-maintained, NVIDIA), Medium for EntropyCache and SlowFast (newer repos)
   - *Mitigation*: Start with Fast-dLLM (most reliable). If a method's code does not reproduce, contact authors or re-implement from the paper. Budget 1 day for reproduction debugging.

4. **Risk: Paper is perceived as "just an ablation study" by reviewers**
   - *Likelihood*: Medium
   - *Mitigation*: (a) IGSD provides a novel algorithmic contribution. (b) The decomposition analysis provides theoretical insight into WHY methods compose well or poorly. (c) The practical recommendation tables provide immediate value. (d) Frame as a "benchmarking and analysis" paper (precedent: "How Efficient Are DLMs?", "Generative Frontiers" -- both are meta-analysis papers accepted at top venues in this area).

5. **Risk: The field moves too fast and someone publishes a similar composition study before us**
   - *Likelihood*: Medium (this is an obvious gap that others will notice)
   - *Mitigation*: Execute fast. The pilot phase (3 hours) gives us early results to validate the approach. The full study (5-10 days) is fast enough to stay ahead. Use automation to parallelize benchmark runs.

### Novelty Claim

**What is new:**

1. **First systematic composition study** of training-free DLM acceleration methods across three orthogonal axes (KV caching, parallel decoding, step scheduling). No existing paper evaluates more than two axes simultaneously.

2. **IGSD (Information-Geometric Step Distillation)**: A novel, extremely simple training-free step scheduler based on inter-step logit KL divergence. ~50 lines of code, one hyperparameter, grounded in the observation that smooth denoising trajectories allow step skipping.

3. **Speedup decomposition framework**: A methodology for attributing combined speedup to per-axis contributions, revealing orthogonality vs. interference between methods.

4. **Practical recommendation tables**: The first task-specific guidance for which method combinations to use under different speed-quality budgets, validated across 2 models and 6 benchmarks.

**What is NOT new (and we do not claim):**

- Individual acceleration methods (Fast-dLLM, EntropyCache, SlowFast, Saber) -- these are prior work.
- The observation that KV caching works for DLMs -- well-established since Fast-dLLM.
- The idea of adaptive step scheduling -- prior art in Saber, PRR, JYS.

**Why this matters despite limited algorithmic novelty:**

The AR model serving ecosystem became practical not because of any single technique but because someone (the vLLM team) systematically combined PagedAttention + continuous batching + prefix caching and showed they compose well. DLM inference is at the same inflection point: individual techniques exist but no one has assembled them into a practical, composable system. This paper is the first step toward a "vLLM for DLMs."
