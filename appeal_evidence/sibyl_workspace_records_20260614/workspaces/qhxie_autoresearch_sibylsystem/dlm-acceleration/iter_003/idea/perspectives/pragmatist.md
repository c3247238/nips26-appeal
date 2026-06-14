# Pragmatist Perspective

## Phase 1: Literature Survey

### Key Resources Found

1. **Fast-dLLM (NVlabs/Fast-dLLM)** -- ICLR 2026. Training-free KV cache + confidence-aware parallel decoding. Up to 27.6x speedup on LLaDA/Dream. **Code: fully available, well-maintained.** The reference implementation with kernel-level cache integration that our iter_001/002 M1 lacked. This is the engineering standard we failed to match.

2. **EntropyCache (mscheong01/EntropyCache)** -- Training-free KV caching, O(V) entropy-guided refresh. 15-26x speedup. **Code: available.** We used this as M1 in iter_001/002 but our integration achieved only 1.16x because we lacked kernel-level sparse attention. The published speedups require proper CUDA kernel integration -- this is an engineering bottleneck we must honestly account for.

3. **SlowFast Sampling (LiangrunFlora/Slow-Fast-Sampling)** -- Training-free dynamic 2-stage sampling. Up to 34.22x when combined with dLLM-Cache. **Code: available, drop-in for LLaDA/Dream.** The only published method that explicitly composes with caching and reports combined numbers. Critical reference for understanding what composition looks like when done properly.

4. **ES-dLLM (arXiv 2603.10088)** -- Training-free early-layer skipping via tensor variation + confidence. 5.6-16.8x; 1.85x OVER SOTA cache methods. **Code: check repo.** Important because it operates on a genuinely different axis (depth/layer computation) rather than the same axes we already tried. Potentially the orthogonal axis we were looking for.

5. **DyLLM (arXiv 2603.08026)** -- Training-free saliency-based token selection + partial attention. Up to 9.6x. **Code: check repo.** Another "different axis" method -- selective token recomputation rather than cache/step/parallel. Uses cosine similarity proxy for token importance.

6. **WeDLM (Tencent/WeDLM)** -- Causal attention DLM with native KV cache. 3-6x over vLLM-optimized baselines. **Code: available on GitHub.** Not training-free (requires fine-tuning from AR), but demonstrates that the right architectural choice (causal attention + topological reordering) solves the KV cache problem cleanly. Relevant for framing "the right way to do KV caching in DLMs."

7. **"How Efficient Are DLMs?" (arXiv 2510.18480)** -- Critical evaluation paper. Key finding: acceleration gains diminish at larger batch sizes. **Essential methodology reference.** Our iter_002 batch sensitivity experiments confirmed this finding -- DLM acceleration is primarily a batch=1 phenomenon.

8. **DARE (arXiv 2604.04215)** -- Unified post-training + eval framework for dLLMs. Covers LLaDA, Dream, SDAR. **Code: available.** The most comprehensive dLLM evaluation substrate -- would be ideal for standardized benchmarking.

9. **dInfer (inclusionAI/dInfer)** -- Modular DLM inference framework. 1100+ TPS on HumanEval at batch=1 on 8xH800. **Code: available.** Important because it demonstrates what production-grade DLM inference looks like with proper engineering.

10. **JoT (arXiv 2602.11133, Anonym-cybersudo/JoT)** -- Per-token early stopping. 7.2x at 98.3% quality. Claims orthogonality to KV caching but does not validate. **Code: available.** A composition test candidate we did NOT try in iter_001/002.

11. **Prophet (arXiv 2508.19982, pixeli99/Prophet)** -- Early answer convergence + commit decoding. Up to 3.4x step reduction. **Code: available.** Task-specific (math/code answer regions) but effective where applicable. Not tried in our previous iterations.

12. **I-DLM (arXiv 2604.11035)** -- First DLM to match AR quality. 5900 tok/s at concurrency=32. Requires training from Qwen3-8B. **Code: available.** Sets the new quality ceiling. Relevant because it shows the field is moving toward hybrid AR-diffusion architectures where the composition question is entirely different.

### Landscape Summary

**What two failed iterations taught us about engineering reality:**

After iter_001 and iter_002 of ComposeAccel, we have hard empirical evidence for several uncomfortable truths:

1. **KV caching without kernel-level integration is a no-op.** Our M1 (EntropyCache) achieved 1.16x speedup -- essentially nothing. The published 15-26x speedups require CUDA-level sparse attention kernels that we did not implement. Fast-dLLM achieves its numbers because NVIDIA engineers wrote proper kernels. This is not a research problem; it is an engineering problem. Any proposal that depends on KV caching speedup without kernel integration will repeat this failure.

2. **IGSD (our novel step scheduler) is functionally equivalent to naive step truncation.** The KL-divergence gate adds no measurable accuracy benefit over simply using fewer steps (tau=0.0 = tau=0.9). This kills the algorithmic novelty of IGSD as a standalone contribution.

3. **Methods predominantly interfere rather than synergize.** M1+M3 ortho=0.41-0.52 (destructive), M3+IGSD ortho=0.61-0.84 (partial interference). The "three orthogonal axes multiply" hypothesis was falsified. Only M1+IGSD was "near-orthogonal" but M1 was a no-op, so this is meaningless.

4. **Open-source DLMs are far behind AR models.** Best composed DLM: ~52% GSM8K at ~96 TPS. Qwen2.5-7B with HF Transformers: 96% GSM8K at 71-471 TPS. This gap is not closable by training-free acceleration alone.

5. **Code generation benchmarks are broken.** HumanEval 2.4%, MBPP 0% baselines on LLaDA-8B make code benchmarks unusable.

**The practical gap that remains:**

Given these failures, the composition study as originally conceived (combine 3 axes, show multiplicative speedup) is dead. However, the broader question is alive and even more interesting: **why do DLM acceleration methods interfere, and what does this tell us about the fundamental structure of the denoising process?**

The field needs a paper that honestly characterizes WHAT works, WHAT doesn't, and WHY -- rather than claiming synergies that don't exist.


## Phase 2: Initial Candidates

### Candidate A: "Why DLM Acceleration Methods Interfere: A Diagnostic Study of Denoising Process Constraints"

- **Hypothesis**: Training-free DLM acceleration methods interfere because they implicitly share a common bottleneck -- the information carried by masked-token predictions at each step is coupled across tokens and across steps. KV caching assumes inter-step KV stability; parallel decoding assumes inter-token independence; step scheduling assumes step-wise monotonic convergence. These assumptions cannot all hold simultaneously because denoising is a coupled process.

- **Implementation sketch**: Re-analyze our iter_001/002 data (15 experiment groups already run). Add targeted diagnostic experiments: (a) measure per-step mutual information between token predictions, (b) measure KV drift under different methods, (c) correlate interference severity with task complexity. Build on Fast-dLLM codebase for proper kernel-level KV cache implementation.

- **Simplest version**: Take our existing M1+M3 pilot data (ortho=0.41-0.52). Add one diagnostic measurement: compute the cosine similarity of KV states between steps 30-31 under vanilla vs. M3-guided unmasking. If M3 guidance changes KV drift patterns (which it must, since it changes unmasking order), this explains M1+M3 interference directly. This is 2 hours of implementation + analysis.

- **Time estimate**: Pilot 3 hours (diagnostic measurements on existing data). Full study 15-20 hours (add proper Fast-dLLM kernel integration + more diagnostics + Dream-7B replication).

- **Reusable components**: All iter_001/002 experiment data, Fast-dLLM codebase, LLaDA-8B-Instruct checkpoint, existing evaluation scripts.


### Candidate B: "Unified Proxy Signal Comparison for Token-Adaptive DLM Inference"

- **Hypothesis**: The proliferating proxy signals for token importance in DLM inference (EntropyCache's decoded token entropy, DyLLM's attention context cosine similarity, ES-dLLM's tensor variation + confidence, Focus-dLLM's past-confidence indicator, Sparse-dLLM's attention-based saliency) are all noisy estimates of the same underlying quantity -- "how much will this token's representation change in the next step?" A simple, unified benchmark comparing all proxy signals on their prediction accuracy vs. compute cost will reveal which signals are most informative per FLOP and whether any combination of cheap signals matches an expensive oracle.

- **Implementation sketch**: For each proxy signal, implement the estimator and measure (a) its correlation with actual token representation change (ground truth from full forward passes), (b) its compute cost in FLOPs and wall-clock time, and (c) the downstream acceleration quality when used to make cache/skip/attend decisions. Use LLaDA-8B-Instruct on GSM8K and HumanEval. Start from EntropyCache and DyLLM codebases.

- **Simplest version**: Implement just two signals -- EntropyCache's entropy and DyLLM's cosine similarity -- on 50 samples. Measure their correlation with actual KV drift (cosine distance of KV states between consecutive steps). Compare the Pareto curve (accuracy of prediction vs. compute cost). This is ~4 hours.

- **Time estimate**: Pilot 4 hours. Full study (5 proxy signals, 2 models, 3 benchmarks) 20-25 hours.

- **Reusable components**: EntropyCache repo, DyLLM repo (or re-implementation from paper), ES-dLLM methodology, LLaDA-8B, Dream-7B checkpoints.


### Candidate C: "Strong Baseline Done Right: Fast-dLLM with Proper Engineering vs. the Zoo of Methods"

- **Hypothesis**: A properly configured Fast-dLLM (with kernel-level KV cache, tuned confidence threshold, and Flash Attention) achieves >15x speedup with <3% accuracy degradation on LLaDA-8B-Instruct, and this single well-engineered baseline beats the majority of the 20+ methods published in Q1 2026 that do not have proper kernel integration. The practical message: the single biggest acceleration lever for DLMs is not algorithmic innovation but engineering quality of KV cache integration.

- **Implementation sketch**: Clone Fast-dLLM repo. Run on LLaDA-8B-Instruct with published configs on GSM8K, MMLU, HumanEval, MBPP. Profile with torch.profiler. Sweep confidence threshold (0.5-0.95 in 0.05 increments). Report the Pareto frontier. Then add JoT (token-level early stopping) on top -- this is the one composition that has a plausible orthogonality claim (KV cache reduces per-step cost, JoT reduces step count -- genuinely different axes). Compare against our iter_001/002 M1 results to quantify the engineering vs. algorithm gap.

- **Simplest version**: Run Fast-dLLM out-of-the-box on GSM8K (LLaDA-8B-Instruct), measure wall-clock TPS and accuracy. Compare against our iter_002 M1 (EntropyCache without kernel integration). If the gap is >5x, that is the paper's key finding: engineering dominates algorithms. This is ~2 hours.

- **Time estimate**: Pilot 2 hours. Full study with sweeps, profiling, JoT composition, Dream-7B: 15-20 hours.

- **Reusable components**: Fast-dLLM repo (complete with eval scripts), JoT repo, LLaDA-8B-Instruct, Dream-7B-Instruct.


## Phase 3: Self-Critique

### Against Candidate A (Interference Diagnostic)

- **Implementation reality check**: The diagnostic measurements (mutual information, KV drift correlation) are straightforward to implement -- they are analysis of existing forward pass outputs. The key engineering risk is integrating Fast-dLLM's kernel-level KV cache to make M1 actually work. Without this, we are just analyzing the same broken M1 from iter_001/002. WebSearch confirmed Fast-dLLM is well-maintained with eval scripts, so this is lower risk than building from scratch.

- **Reproducibility attack**: Moderate. The diagnostic methodology is well-specified but the "interference explanation" requires a coherent causal narrative. There is a risk that we measure correlations but cannot establish causation. The paper could end up being "here are some observations about interference" without a compelling mechanistic story.

- **Baseline sanity check**: No existing paper provides a systematic interference analysis. The closest is "How Efficient Are DLMs?" which critiques evaluation practices but does not diagnose interference between methods. This is genuinely novel but risky -- it is hard to write a compelling paper about negative results.

- **Scope attack**: Interference analysis is niche. The audience is specifically DLM acceleration researchers, which is a growing but still small community. Broader ML audience may not care about "why acceleration methods for a niche model family don't compose."

- **Verdict**: MODERATE -- Intellectually interesting, genuinely novel diagnostic contribution, but high risk of producing a paper that is "correct but boring." The negative result framing from iter_002's verdict ("composition reveals why DLM denoising resists modular acceleration") is actually compelling if executed well, but requires a strong mechanistic explanation.


### Against Candidate B (Proxy Signal Comparison)

- **Implementation reality check**: Implementing 5 different proxy signals from 5 different papers is engineering-heavy. Each has its own codebase, abstractions, and assumptions. DyLLM and ES-dLLM are both from March 2026 -- their code may be immature. The core measurement (correlation with actual KV drift) is clean and well-defined, but collecting the ground truth (full forward passes for every step) is computationally expensive.

- **Reproducibility attack**: Strong. The methodology (measure proxy accuracy vs. ground truth) is simple and reproducible. The proxy signals themselves are well-specified in the papers.

- **Baseline sanity check**: Gap 10 and Gap 13 from the literature survey explicitly call out the need for this comparison. No paper has done it. However, it risks being perceived as "just a benchmark" without algorithmic contribution. Adding a simple "ensemble proxy" (majority vote of cheap signals) provides a minimal algorithmic contribution.

- **Scope attack**: Proxy signals are used across all DLM acceleration methods. A unified comparison has broad relevance to anyone building DLM acceleration. The paper directly addresses Gap 10 and Gap 13 from the literature survey.

- **Verdict**: MODERATE -- Clean, well-scoped, fills an identified gap. But the engineering effort of implementing 5 signals correctly is substantial, and the paper may be thin if the conclusion is just "signal X is best." Needs the ensemble contribution to be publishable.


### Against Candidate C (Strong Baseline)

- **Implementation reality check**: The lowest-risk candidate. Fast-dLLM is NVIDIA-maintained, well-documented, and has complete eval scripts. Running it out-of-the-box is ~30 minutes of setup. The profiling and sweep add ~6 hours. Adding JoT on top is the only integration risk, but JoT operates at a higher level (deciding when to stop) and should not conflict with Fast-dLLM's cache.

- **Reproducibility attack**: Perfect. Fast-dLLM's results are published and reproducible. Our only claim is "here is the Pareto frontier and here is what happens when you add JoT."

- **Baseline sanity check**: The paper's thesis is that this baseline IS the strongest approach. If Fast-dLLM + JoT achieves, say, 20x speedup at 97% accuracy retention, and most of the 20+ methods from Q1 2026 achieve less than this in our hands (because they lack kernel integration), that is a significant practical finding. However, this is a "meta" finding, not an algorithmic contribution. Top venues may see this as a systems paper or a benchmark paper rather than a research contribution.

- **Scope attack**: Very broad impact -- any DLM practitioner benefits from knowing "just use Fast-dLLM + JoT." But the novelty is thin unless we discover something surprising (e.g., JoT + Fast-dLLM is actually subadditive, or one specific threshold setting universally dominates).

- **Verdict**: STRONG for practical impact, WEAK for novelty. As a standalone paper, likely insufficient for NeurIPS/ICML unless paired with substantial analysis of WHY the engineering baseline dominates. But as a foundation for Candidate A (interference analysis) or Candidate B (proxy comparison), it provides the credible baseline that iter_001/002 lacked.


## Phase 4: Refinement

### Dropped Ideas

- **Candidate B** (Proxy Signal Comparison) dropped as primary proposal. Engineering effort (5 signal implementations from immature codebases) is too high for the expected novelty contribution. However, the core idea (comparing proxy signals) is folded into the final proposal as one diagnostic dimension.

### Strengthened Ideas

**Merged A+C: "When Engineering Beats Algorithms: A Diagnostic Study of DLM Acceleration Composition"**

The key insight from Phase 3 is that Candidate C (strong baseline) solves Candidate A's biggest weakness (broken M1), and Candidate A (interference analysis) solves Candidate C's biggest weakness (thin novelty). Together:

1. **Phase 1 (from Candidate C)**: Establish the proper engineering baseline. Run Fast-dLLM with kernel-level KV cache on LLaDA-8B-Instruct. This is the M1 that iter_001/002 should have had. Measure the actual speedup achievable with proper engineering. This gives us a credible reference point.

2. **Phase 2 (composition test with JoT)**: Fast-dLLM + JoT is the most plausible composition because they operate on genuinely different axes (per-step compute reduction vs. step count reduction). If this succeeds (near-multiplicative), it contradicts our iter_002 finding that methods interfere -- and the reason would be that iter_002's interference was caused by broken M1, not by fundamental denoising constraints. If it ALSO interferes, that strengthens the interference narrative.

3. **Phase 3 (from Candidate A)**: Diagnostic analysis of WHY Fast-dLLM + JoT composes well or poorly. Measure KV drift under JoT's early stopping vs. full denoising. Measure whether JoT's confidence signal is calibrated correctly when KV cache introduces approximation error. This provides the mechanistic explanation.

4. **Phase 4 (generalization)**: Repeat on Dream-7B-Instruct. Compare with our iter_002 data to quantify the "engineering gap" (proper kernel vs. naive Python implementation).

**Why this is the right proposal for iteration 3:**

- It directly addresses the root cause of iter_001/002 failure: M1 was broken because of engineering, not because of fundamental algorithmic limitations.
- It uses the strongest available open-source implementation (Fast-dLLM from NVIDIA) instead of building from scratch.
- It tests a fresh composition pair (Fast-dLLM + JoT) that we have not tried before, with a clear a priori reason to expect orthogonality.
- It provides diagnostic analysis (the "why" behind composition success or failure) that no existing paper offers.
- It can be executed in ~15-20 GPU hours, well within our resource budget.
- It produces a publishable result regardless of outcome: composition success validates the "engineering matters" thesis; composition failure with proper engineering validates the "fundamental interference" thesis.

### Selected Front-Runner

**"When Engineering Beats Algorithms: A Diagnostic Study of DLM Acceleration Composition"**


## Phase 5: Final Proposal

### Title

When Engineering Beats Algorithms: A Diagnostic Study of DLM Acceleration Composition with Kernel-Level KV Caching

### Hypothesis

The failure of training-free DLM acceleration methods to compose (observed in prior work) is primarily an engineering artifact of inadequate KV cache implementation, not a fundamental property of the denoising process. With kernel-level KV cache integration (Fast-dLLM), combining KV caching with token-level early stopping (JoT) will achieve near-multiplicative speedup (composition ratio > 0.7) because these methods target genuinely orthogonal computation axes: per-step attention cost vs. total step count.

### Motivation

The DLM acceleration literature has exploded in 2025-2026, with >20 training-free methods published. Two critical questions remain unanswered:

1. **Do methods actually compose?** Our own prior work (ComposeAccel, iter_001/002) found predominantly interference (ortho=0.41-0.84 for most pairs). But our implementation had a critical flaw: the KV cache component (M1/EntropyCache) achieved only 1.16x speedup due to lack of kernel-level integration -- effectively a no-op. This means ALL composition experiments involving M1 were confounded.

2. **Is the engineering quality of the implementation the dominant variable?** Fast-dLLM achieves 27.6x with proper CUDA kernels. Our Python-level EntropyCache achieves 1.16x. The 24x gap suggests that implementation quality dominates algorithmic design. If this is true, the practical advice for the community is: "invest in engineering, not in inventing new algorithms."

This paper resolves these questions by running the first composition study with production-quality KV cache implementation.

### Method

**Phase 0: Environment Setup and Baseline Calibration (2 GPU hours)**

1. Clone and set up Fast-dLLM repo on our GPU server.
2. Download LLaDA-8B-Instruct and Dream-7B-Instruct.
3. Run vanilla LLaDA-8B-Instruct (T=64) on GSM8K (full 1319 samples, 3 seeds) -- establish the canonical baseline TPS and accuracy.
4. Standardize measurement protocol: generation-only timing (exclude prompt encoding), post-warmup (5 samples), batch=1.

**Phase 1: Fast-dLLM Pareto Frontier (4 GPU hours)**

Run Fast-dLLM on LLaDA-8B-Instruct with systematic hyperparameter sweep:
- Cache refresh rate: {every step, every 2 steps, every 4 steps, every 8 steps}
- Confidence threshold for parallel decoding: {0.5, 0.7, 0.8, 0.9, 0.95}
- Benchmarks: GSM8K (reasoning), HumanEval (code -- even if LLaDA baseline is low, relative comparison is valid), MMLU (knowledge)
- Measure: TPS, accuracy, peak GPU memory

This gives us 20 configurations x 3 benchmarks = 60 runs. At ~5 min each = 5 GPU hours.

Expected output: Pareto frontier of speedup vs. accuracy for Fast-dLLM alone.

**Phase 2: JoT Pareto Frontier (3 GPU hours)**

Run JoT (token-level early stopping) on LLaDA-8B-Instruct:
- Confidence threshold: {0.8, 0.9, 0.95, 0.99}
- Spatial modulation: {on, off}
- Same 3 benchmarks

8 configurations x 3 benchmarks = 24 runs. At ~10 min each (full step computation + JoT overhead) = 4 GPU hours.

Expected output: Pareto frontier of JoT alone.

**Phase 3: Composition -- Fast-dLLM + JoT (5 GPU hours)**

Select the Pareto-optimal configurations from Phases 1 and 2 (top 3 from each). Run all 9 combinations on 3 benchmarks with 3 seeds.

9 combinations x 3 benchmarks x 3 seeds = 81 runs. At ~5 min each = ~7 GPU hours.

Key measurements:
- Combined speedup vs. product of individual speedups (composition ratio)
- Accuracy degradation relative to vanilla baseline
- KV drift analysis: for each combination, measure average cosine distance of KV states between consecutive steps. Does JoT's early stopping affect KV cache validity?
- Confidence calibration: does Fast-dLLM's cache approximation error degrade JoT's confidence signal?

**Phase 4: Engineering Gap Quantification (2 GPU hours)**

Run our iter_002 M1 (EntropyCache, Python-level) on the SAME benchmark configurations used for Fast-dLLM in Phase 1. Direct comparison:
- Same model, same benchmarks, same hyperparameter settings (where applicable)
- Measure the speedup gap between kernel-level (Fast-dLLM) and Python-level (M1) caching
- Profile both implementations with torch.profiler to identify WHERE the time is spent

**Phase 5: Cross-Model Generalization (3 GPU hours)**

Repeat Phase 3's top 3 compositions on Dream-7B-Instruct.

**Phase 6: Diagnostic Analysis (1 GPU hour + analysis time)**

For the best-composing and worst-composing configurations:
- Visualize per-step token confidence evolution under combined acceleration
- Measure mutual information between adjacent-step predictions
- Compare confidence calibration curves (predicted vs. actual accuracy) with and without KV cache

### Simplest Version

The absolute minimum experiment (2 hours):
1. Run Fast-dLLM on LLaDA-8B-Instruct GSM8K (default config) -> S1, A1
2. Run JoT on LLaDA-8B-Instruct GSM8K (threshold=0.95) -> S2, A2
3. Run Fast-dLLM + JoT on LLaDA-8B-Instruct GSM8K -> S_combined, A_combined
4. Run our iter_002 M1 on LLaDA-8B-Instruct GSM8K -> S_M1, A_M1
5. Compute: composition_ratio = S_combined / (S1 * S2). Engineering_gap = S1 / S_M1.

If composition_ratio > 0.7, the core hypothesis (engineering was the problem, not fundamental interference) is confirmed. If engineering_gap > 5x, the "engineering dominates algorithms" thesis is confirmed.

### Baselines

1. **Vanilla LLaDA-8B-Instruct** (T=64 steps): Quality ceiling and speed floor. Expected: ~71% GSM8K, ~31 TPS (from our iter_002 data).

2. **Fast-dLLM (published config)**: The engineering reference. Expected: 10-27x speedup depending on quality tolerance (from published numbers; we need to reproduce these).

3. **Our iter_002 M1 (EntropyCache, Python-level)**: The broken baseline that demonstrates the engineering gap. Known: 1.16x speedup.

4. **Our iter_002 M1+IGSD (best composition from prior work)**: Known: 8.88x speedup at 52% accuracy retention. The question is whether Fast-dLLM + JoT can beat this Pareto point.

5. **JoT (published config)**: 7.2x at 98.3% quality retention (from paper). We reproduce this.

### Experimental Plan

**Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (generalization)

**Benchmarks**:
- Reasoning: GSM8K (full 1319, 3 seeds for key configs)
- Knowledge: MMLU (5-shot)
- Code: HumanEval (0-shot) -- even with low baseline, relative speedup comparisons are valid

**Hardware**: Available local GPUs (from config.yaml: up to 4 GPUs, IDs [0,1,4,5]). All timing at batch=1 (standard for DLM papers) with supplementary batch=4 for key configs.

**Metrics**:
- Wall-clock TPS (generation-only, post-warmup)
- Accuracy / Pass@1
- Composition ratio: S_combined / (S_individual_1 * S_individual_2)
- Engineering gap: S_kernel_level / S_python_level for same method
- KV drift: average cosine distance of KV states between consecutive steps
- Confidence calibration error: |P(correct | confidence=c) - c| averaged over c

**Ablation Schedule**:
1. Baseline calibration (1 run, 3 seeds) -> 3 runs
2. Fast-dLLM Pareto sweep (20 configs x 3 benchmarks) -> 60 runs
3. JoT Pareto sweep (8 configs x 3 benchmarks) -> 24 runs
4. Composition (9 configs x 3 benchmarks x 3 seeds) -> 81 runs
5. Engineering gap (5 configs x 3 benchmarks) -> 15 runs
6. Dream-7B top configs (5 configs x 3 benchmarks x 3 seeds) -> 45 runs
7. Diagnostic analysis (2 configs x 3 benchmarks) -> 6 runs

Total: ~234 runs. At ~5-10 min each: ~20-40 GPU-hours. With 4 GPUs parallel: 5-10 wall-clock hours. Well within 1-day budget.

**Pilot plan (first 3 hours)**:
1. Set up Fast-dLLM repo on local GPU (30 min)
2. Run Fast-dLLM default config on GSM8K subset (200 samples) -> first speedup number (15 min)
3. Run our iter_002 M1 on same subset -> engineering gap measurement (15 min)
4. Set up JoT repo (30 min)
5. Run JoT default on GSM8K subset (15 min)
6. Run Fast-dLLM + JoT on GSM8K subset -> first composition data point (15 min)
7. Analyze: Is composition ratio > 0.7? Is engineering gap > 5x? (30 min)

### Resource Estimate

- **GPU**: 1x A100/H100 minimum (Fast-dLLM designed for these), 4 GPUs preferred for parallelism
- **Model sizes**: LLaDA-8B (~16GB FP16), Dream-7B (~14GB FP16). Both fit on single GPU with KV cache room
- **Wall-clock (pilot)**: 3 hours for initial validation
- **Wall-clock (full study)**: 2-3 days single GPU, 1 day with 4 GPUs
- **GPU-hours**: 20-40 total. Very moderate for a systems/analysis paper
- **Software dependencies**: Fast-dLLM repo, JoT repo, LLaDA/Dream HuggingFace checkpoints, torch.profiler, standard eval harness
- **Per-task estimate**: Each individual experiment run should complete in 5-15 minutes, well within the 1-hour-per-task guideline

### Risk Assessment

1. **Risk: Fast-dLLM does not reproduce published speedups on our hardware**
   - *Likelihood*: Low (NVIDIA maintains the repo, ICLR 2026 accepted)
   - *Mitigation*: Profile to identify bottleneck. If hardware-specific, report our numbers honestly and note the discrepancy. The engineering gap comparison with M1 is valid regardless.

2. **Risk: Fast-dLLM + JoT also shows interference (composition ratio < 0.5)**
   - *Likelihood*: Medium (JoT changes when tokens "freeze," which could affect cache validity)
   - *Mitigation*: This is a publishable negative result that strengthens the "fundamental interference" thesis. The paper then becomes "interference persists even with proper engineering, confirming it is a property of the denoising process." Include diagnostic analysis explaining the mechanism.

3. **Risk: JoT repo does not support LLaDA-8B-Instruct directly**
   - *Likelihood*: Medium (JoT paper evaluates on both LLaDA-8B and Dream-7B, so support should exist)
   - *Mitigation*: JoT's algorithm is simple (per-token confidence check). If repo integration fails, re-implement from paper description (~100 lines of Python).

4. **Risk: Paper perceived as "just running existing methods"**
   - *Likelihood*: Medium-High
   - *Mitigation*: (a) The diagnostic analysis (KV drift under composition, confidence calibration interaction) provides mechanistic insight no existing paper offers. (b) The engineering gap quantification is a novel and practically important finding. (c) The "interference or orthogonality under proper engineering" question is the RIGHT question for the field right now, and we are the first to ask it correctly. (d) We have two prior iterations of data showing what goes wrong with improper engineering, providing unique "lessons learned" content.

5. **Risk: The "engineering beats algorithms" thesis is already obvious**
   - *Likelihood*: Low -- it seems obvious in hindsight but NO ONE has quantified it. The 24x gap between kernel-level and Python-level caching is a shocking number that the community needs to see. Many papers report Python-level speeds and claim victories.
   - *Mitigation*: Frame as a "call to arms" for proper engineering standards in DLM acceleration research. This has precedent: "How Efficient Are DLMs?" did exactly this for evaluation practices and was well-received.

### Novelty Claim

**What is new:**

1. **First composition study with production-quality KV cache implementation.** All prior composition work (including our own ComposeAccel iter_001/002) used Python-level KV cache integration that achieved minimal speedup. We are the first to use kernel-level implementation (Fast-dLLM) and test composition with a genuinely different method (JoT).

2. **Engineering gap quantification.** First paper to measure the gap between Python-level and kernel-level DLM KV cache implementation on the SAME benchmarks with the SAME model. This quantifies how much of reported "algorithmic" acceleration is actually "engineering" acceleration.

3. **Diagnostic analysis of composition mechanisms.** First paper to measure KV drift, confidence calibration interaction, and mutual information dynamics under combined acceleration methods. This provides the "why" behind composition success or failure.

4. **Controlled comparison across two prior failed iterations.** Unique position: we have 15 experiment groups from two prior iterations where composition failed due to engineering limitations. Direct comparison with proper engineering isolates the engineering variable.

**What is NOT new (and we do not claim):**

- Fast-dLLM's KV caching method (NVIDIA's prior work)
- JoT's token-level early stopping (prior work)
- The observation that DLM inference is slow (well-established)
- The idea that engineering quality matters for systems papers (common sense, but unquantified in DLM context)

**Why this matters:**

The DLM acceleration field is producing diminishing-return algorithmic innovations (>20 methods, all reporting 5-30x speedups, none systematically tested together with proper engineering). This paper provides the field with (a) the first credible composition data point, (b) a quantification of how much engineering matters, and (c) diagnostic tools for understanding WHY methods compose or interfere. This is exactly the kind of "adult supervision" paper that a maturing field needs.
