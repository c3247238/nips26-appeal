# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: DLM parallel decoding provides a "free" speedup over autoregressive generation**
   - Evidence challenging it:
     - [ParallelBench (ICLR 2026)](https://arxiv.org/abs/2510.04767): The conditional independence assumption causes parallel decoding to ignore token dependencies, leading to "dramatic quality degradation in real-world scenarios." dLLMs under parallel decoding fail on tasks trivial for humans and AR LLMs (Waiting Line, Text Writing, Puzzles).
     - [AUP: When Accuracy Meets Parallelism (Hao AI Lab, UCSD)](https://hao-ai-lab.github.io/blogs/text-diffusion/): "The speedup offered by dLLMs is not a free lunch." All dLLM methods land on the same accuracy-parallelism curve; AR + speculative decoding still achieves top AUP scores.
     - [Theoretical Benefit and Limitation of DLMs (NeurIPS 2025)](https://arxiv.org/abs/2502.09622): For tasks requiring low sequence error rate (reasoning, code), sampling steps must scale *linearly* with sequence length, eliminating the parallel speedup entirely.

2. **Assumption: KV caching can be straightforwardly adapted from AR models to DLMs**
   - Evidence challenging it:
     - [dKV-Cache (NeurIPS 2025)](https://arxiv.org/abs/2505.15781): Naive KV caching "collapses to near-zero accuracy" in DLMs. Token representations evolve fundamentally across denoising steps, making cached keys/values stale in ways that do not occur in AR decoding.
     - [Elastic-Cache](https://arxiv.org/abs/2510.14973): Layer-dependent staleness -- deeper layers exhibit larger KV drift, requiring adaptive refresh policies that AR models do not need.
     - [Limits of KV Cache Compression (arXiv 2503.11108)](https://arxiv.org/abs/2503.11108): Information-theoretic lower bounds show any algorithm producing (1+/-eta)-approximation of attention output requires Omega(nd) bits, placing fundamental limits on cache compression.

3. **Assumption: Reported speedup numbers (27x, 45x, 99x) reflect real-world usable acceleration**
   - Evidence challenging it:
     - [How Efficient Are Diffusion LMs? (arXiv 2510.18480)](https://arxiv.org/abs/2510.18480): Speedup gains "diminish as batch size grows, eventually falling behind AR." LLaMA3-8B achieves 13.7x higher throughput than LLaDA-8B under fair conditions. Acceleration strategies that shine at batch=1 lose their advantage at batch>=8.
     - JetBrains AI Blog: "The best quality comes when unmasking one token per step, which slows things down and makes these models not differ much from AR models in practice."
     - [Generative Frontiers (arXiv 2604.02718)](https://arxiv.org/abs/2604.02718): Methodological critique showing evaluation inconsistencies -- different papers use different step counts, benchmarks, hardware, and batch sizes, making cross-paper speedup comparisons unreliable.

4. **Assumption: DLMs' bidirectional attention provides inherent advantages for reasoning**
   - Evidence challenging it:
     - [The Flexibility Trap (arXiv 2601.15165)](https://arxiv.org/abs/2601.15165): Arbitrary-order generation *narrows* rather than expands reasoning capability. dLLMs exploit order flexibility to bypass high-entropy "reasoning spark" tokens (logical connectives like "Therefore", "Thus", "Since"), leading to premature collapse of the solution space.
     - [LogicDiff (arXiv 2603.26771)](https://arxiv.org/abs/2603.26771): Replacing confidence-based unmasking with logic-role-guided unmasking improves LLaDA-8B from 22.0% to 60.7% on GSM8K *without modifying any model parameter*. This proves the reasoning deficit is in the decoding strategy, not the model representations.
     - [Lost in Diffusion (arXiv 2604.10556)](https://arxiv.org/abs/2604.10556): dLLMs exhibit higher propensity for hallucination than AR counterparts. Failure modes include premature termination, incomplete denoising, and context intrusion. Dream-7B fails to recognize non-existent entities in 98.50% of cases.

5. **Assumption: Composing multiple acceleration methods yields multiplicative speedups**
   - Evidence challenging it:
     - [How Efficient Are Diffusion LMs?](https://arxiv.org/abs/2510.18480): When the system is already compute-bound (which happens quickly with DLMs due to O(L^2) bidirectional attention), stacking acceleration techniques yields progressively smaller returns.
     - No paper has demonstrated that combining KV caching + parallel decoding + step scheduling yields anywhere close to multiplicative speedup while maintaining quality. SlowFast mentions "combined with caching" but the composition is ad-hoc, not systematically studied.
     - The theoretical perspective (workspace) identifies the composition error interaction term R_interact as uncharacterized -- independent optimality of each method does NOT imply combined optimality.

6. **Assumption: Training-free acceleration is sufficient for competitive DLM inference**
   - Evidence challenging it:
     - [Bitter Lesson for DLMs in Agentic Workflows (arXiv 2601.12979)](https://arxiv.org/abs/2601.12979): "Current dLLMs fail to serve as reliable agentic backbones, particularly in multi-turn interaction scenarios." Systematic failures in embodied and tool-calling settings.
     - [d3LLM (Hao AI Lab)](https://arxiv.org/abs/2601.07568): Even with pseudo-trajectory distillation (which IS training-based), the best open-source dLLM achieves only 5x over AR. Mercury and Gemini Diffusion (10x+ over AR) are closed-source with undisclosed architectural innovations -- suggesting the real breakthroughs require fundamental training changes, not inference tricks.
     - Fast-dLLM v2 (2.5x over AR) requires ~1B fine-tuning tokens; D2F requires distillation training; Block Diffusion requires retraining from scratch. The training-free methods that claim the highest speedups (Window-Diffusion 99x, Elastic-Cache 45x) also have the most quality degradation.

### Landscape of Doubt

The DLM acceleration field in early 2026 presents a textbook case of **hype-reality divergence**. The narrative being sold is: "DLMs offer parallel decoding -> training-free methods unlock massive speedups -> composition of methods will close the gap with AR." But the evidence tells a very different story:

1. **The Parallel Decoding Mirage**: The theoretical promise of O(1) parallel decoding crashes into the reality of token dependencies. For any task where correctness matters (reasoning, code, structured output), you need near-sequential unmasking -- which means DLMs lose their one claimed advantage. Feng et al. (2025) prove this is not a fixable engineering problem but a *fundamental theoretical limitation* of the MDM framework: achieving low sequence error rate requires O(L) steps.

2. **The Speedup Inflation Problem**: Papers report batch=1 speedups on cherry-picked benchmarks with carefully tuned hyperparameters. Under realistic serving conditions (batch>=8, mixed workloads), AR models with vLLM + speculative decoding remain more efficient. The 27x-99x speedup numbers that populate the literature are misleading without jointly reporting quality degradation and batch-size sensitivity.

3. **The Composition Fantasy**: Every paper optimizes one lever in isolation. Nobody has shown that combining KV caching, parallel decoding, step scheduling, and token pruning yields a system that is actually faster than a well-tuned AR baseline at equal quality. This "composition gap" is the most glaring blind spot in the current research direction.

4. **The Decoding Order Problem**: The Flexibility Trap and LogicDiff papers reveal that the default confidence-based unmasking strategy -- which is assumed by nearly every acceleration paper -- is fundamentally flawed for reasoning. Acceleration methods that assume this default strategy and simply try to make it faster are optimizing a broken baseline.

---

## Phase 2: Initial Candidates

### Candidate A: "The Emperor Has No Speedup" -- Exposing the Composition Ceiling of Training-Free DLM Acceleration

- **Challenged assumption**: Multiple training-free acceleration methods (KV caching, parallel decoding, adaptive scheduling, token pruning) compose multiplicatively to give speedups beyond what any single method achieves, while maintaining quality.
- **Evidence against**: No paper demonstrates this. The theoretical framework from our workspace's theoretical perspective predicts an interaction term R_interact that could make composition subadditive. The "How Efficient Are Diffusion LMs?" paper shows batch-size scaling already erodes single-method speedups. Compute-bound regimes arrive faster for DLMs due to O(L^2) attention.
- **Contrarian hypothesis**: The composition of training-free DLM acceleration methods is fundamentally subadditive -- combining the top 3-4 methods yields at most 1.5x improvement over the single best method, not the 3-4x improvement that linear composition would predict. Furthermore, at realistic batch sizes (>=8), the composed system is SLOWER than AR + speculative decoding at equal quality.
- **Exploitation plan**: Systematically implement and compose the top training-free methods (Fast-dLLM KV cache, SlowFast sampling, EntropyCache refresh, ES-dLLM layer skipping) on LLaDA-8B. Measure the actual composition ceiling. Compare against a properly tuned LLaMA3-8B + EAGLE speculative decoding baseline at batch sizes 1, 4, 8, 16, 32. Report the AUP metric (Hao AI Lab's accuracy-under-parallelism) for fair comparison.
- **Novelty estimate**: 7/10

### Candidate B: "Rethinking the Unmasking Order" -- DLM Acceleration is Optimizing the Wrong Objective

- **Challenged assumption**: The confidence-based unmasking order used by virtually all DLM acceleration papers is a reasonable default that acceleration methods should try to speed up.
- **Evidence against**: The Flexibility Trap shows confidence-based unmasking systematically avoids high-entropy reasoning tokens. LogicDiff improves GSM8K by +38.7pp just by changing unmasking order. ParallelBench shows confidence-based parallel decoding fails catastrophically on dependency-heavy tasks.
- **Contrarian hypothesis**: The dominant DLM acceleration paradigm -- "make the existing decoding process faster" -- is solving the wrong problem. The real bottleneck is not computational speed but decoding ORDER. An acceleration method that first fixes the unmasking order (via dependency-awareness or logic-role guidance) and THEN applies speed optimizations will achieve better quality-speed Pareto frontiers than methods that aggressively speed up a broken unmasking strategy.
- **Exploitation plan**: Implement a two-stage pipeline: (1) LogicDiff-style dependency-aware unmasking order (lightweight classifier, <6% overhead), then (2) apply KV caching and adaptive scheduling ON TOP of the corrected order. Compare against: (a) vanilla Fast-dLLM (aggressive speed, broken order), (b) LogicDiff alone (fixed order, no speed), (c) our combined approach (fixed order + speed). Evaluate on reasoning (GSM8K, MATH500) and code (HumanEval, MBPP).
- **Novelty estimate**: 8/10

### Candidate C: "The Honest Benchmark" -- When Are DLMs Actually Faster Than AR Models?

- **Challenged assumption**: There exists a practical operating regime where open-source DLMs with training-free acceleration are faster than well-optimized AR models at equal or better quality.
- **Evidence against**: "How Efficient Are Diffusion LMs?" shows LLaMA3-8B is 13.7x faster than LLaDA-8B. Even with the best training-free acceleration, the gap barely closes. Mercury/Gemini Diffusion demonstrate feasibility at the closed-source level, but their techniques are unknown. The Bitter Lesson paper shows DLMs fail as agentic backbones.
- **Contrarian hypothesis**: For open-source DLMs with training-free methods, the speed-quality Pareto frontier NEVER dominates a well-tuned AR baseline (LLaMA3-8B + vLLM + speculative decoding) on reasoning and coding tasks. The only regime where DLMs win is low-dependency generation (e.g., infilling, simple completion) at batch=1.
- **Exploitation plan**: Construct the first comprehensive, fair comparison: LLaDA-8B with every training-free acceleration method vs. LLaMA3-8B with vLLM + EAGLE-2. Vary: batch size (1-32), sequence length (64-1024), task type (reasoning, code, infilling, planning). Plot Pareto frontiers. Identify the exact boundary conditions where DLMs win/lose.
- **Novelty estimate**: 6/10

---

## Phase 3: Self-Critique

### Against Candidate A (Composition Ceiling)

- **Steelman for conventional view**: SlowFast Sampling already claims up to 34.22x when combined with caching, suggesting composition CAN work. Fast-dLLM combines KV cache + parallel decoding and achieves 27.6x -- this IS a composition of two methods. The d3LLM blog acknowledges the accuracy-parallelism tradeoff but shows their framework (distillation + entropy decoding + KV cache) achieves 10x with minimal accuracy loss, demonstrating successful three-way composition.
- **Cherry-picking check**: I may be selectively emphasizing the "How Efficient Are Diffusion LMs?" batch-size results while ignoring that most real-world inference (interactive chat) IS batch=1. For single-user interactive use, batch=1 speedups are the relevant metric.
- **Confounding check**: The compute-bound argument depends on hardware. On next-gen hardware (H200, B200), the memory-compute balance shifts. DLMs might benefit MORE from hardware improvements than AR models because their workload has higher arithmetic intensity.
- **Actionability check**: Even if I am right that composition is subadditive, this is a negative result that narrows expectations. It is publishable as a systematic study, but does it lead to a better method? Only if we can identify WHY composition fails and propose a fix.
- **Verdict**: MODERATE -- The core insight (composition may be subadditive) is valuable and testable, but the batch=1 counterargument weakens the practical impact. The study needs to include both batch=1 and batched settings to be fair.

### Against Candidate B (Wrong Unmasking Order)

- **Steelman for conventional view**: LogicDiff's +38.7pp on GSM8K is remarkable, but it adds a lightweight classifier that must be trained. This means the "fix" is not truly training-free -- it requires supervision on logical role labels. Also, the Flexibility Trap paper's analysis focuses on reasoning tasks. For generation tasks that do NOT require sequential logic (poetry, creative writing, infilling), confidence-based unmasking may be perfectly fine.
- **Cherry-picking check**: I am heavily weighting the GSM8K result (22.0% -> 60.7%) which is dramatic. But LLaDA-8B-Instruct already scores much higher on GSM8K with proper prompting and more steps (the 22% is with aggressive parallel decoding). The improvement from fixing unmasking order is real but may be smaller when starting from a properly tuned baseline.
- **Confounding check**: LogicDiff's improvement could be confounded by the fact that it forces a more sequential (premises-first) order, which is effectively closer to AR decoding. The "fix" might just be "make DLMs decode more like AR models," which undermines the entire DLM proposition.
- **Actionability check**: YES -- this directly leads to a better system: order-aware acceleration. If you first fix the order, then speed up the fixed order, you get a strictly better Pareto frontier. This is constructive, not just a "gotcha."
- **Verdict**: STRONG -- The insight is well-grounded, the LogicDiff evidence is compelling, and the proposed method (order-first, speed-second) is constructive. The main weakness is that LogicDiff's classifier requires some training, making the full pipeline not purely training-free. However, the classifier is tiny (4.2M params, 0.05% of base model) and could potentially be replaced with a heuristic or transferred across models.

### Against Candidate C (Honest Benchmark)

- **Steelman for conventional view**: Mercury achieves 5-10x over AR models at competitive quality. While closed-source, this proves the concept is viable. d3LLM achieves 5x over AR with only pseudo-trajectory distillation. The gap is closing rapidly. DLMs also offer unique capabilities AR models lack: infilling, flexible-length generation, bidirectional context.
- **Cherry-picking check**: I am comparing the WORST of DLMs (LLaDA with training-free methods) against the BEST of AR (LLaMA3 + vLLM + EAGLE). A fairer comparison would be vanilla LLaMA3 (no speculative decoding) vs. LLaDA + Fast-dLLM.
- **Confounding check**: The capability differences (infilling, planning, bidirectional context) mean the comparison is not apples-to-apples. DLMs may be slower on shared tasks but offer UNIQUE tasks where AR cannot compete at all.
- **Actionability check**: A systematic benchmark is useful as infrastructure, but this is more of a "report card" than a "new method." It identifies the problem space but does not solve anything.
- **Verdict**: WEAK -- The honest benchmark is valuable, but it has been partially done by "How Efficient Are Diffusion LMs?" and ParallelBench. The unique contribution would need to be the composition analysis and the identification of DLM-advantaged task categories, which overlaps with Candidate A.

---

## Phase 4: Refinement

### Dropped
- **Candidate C** (Honest Benchmark): Too close to existing work (ParallelBench, "How Efficient Are Diffusion LMs?", AUP blog). The composition analysis from Candidate A already provides the most impactful negative result. Candidate C's "identify where DLMs win" component can be folded into the final proposal.

### Strengthened Candidates

**Candidate B (front-runner)** is strengthened and merged with elements of Candidate A:

The refined proposal combines:
1. **The order-first insight from Candidate B**: The dominant acceleration paradigm optimizes the wrong objective. Fixing the unmasking order BEFORE accelerating is strictly better.
2. **The composition analysis from Candidate A**: Systematically test whether acceleration methods compose better when applied to a corrected decoding order vs. the default broken order.
3. **The task-dependence insight from Candidate C**: The unmasking order problem is task-dependent -- reasoning tasks need order correction, infilling tasks may not.

**Additional corroboration found**:
- The Innovator perspective (workspace) proposes Temporal Consistency Verification (TCV) for step compression -- but this assumes the default unmasking order is correct. Our contrarian insight is that TCV and similar methods are building on a flawed foundation for reasoning tasks.
- The Pragmatist perspective (workspace) proposes a "DLM Acceleration Cookbook" composition study -- but does not consider whether the unmasking order affects composability. Our proposal adds the critical dimension: order-aware vs. order-agnostic composition.
- The JustGRPO paper (arXiv 2601.15165) shows AR-order training + parallel inference is viable, achieving 89.1% on GSM8K. This is a training-based fix. Our proposal tests whether a training-free order correction (lighter than JustGRPO) can provide similar benefits when composed with acceleration.

### Selected Front-Runner

**"Order-First Acceleration: Rethinking DLM Inference by Fixing What to Accelerate Before How to Accelerate It"**

The core thesis: The DLM acceleration community has spent a year optimizing the SPEED of a fundamentally flawed decoding strategy. A simple reordering of priorities -- first fix the unmasking order, then apply speed optimizations -- yields a better quality-speed Pareto frontier than any amount of speed optimization on the default strategy. Furthermore, the composition of acceleration methods is more additive (less subadditive) when applied to a corrected order, because the corrected order reduces the inter-method error interaction term.

---

## Phase 5: Final Proposal

### Title
**"Order-First Acceleration: Why DLM Speedup Methods Are Optimizing a Broken Baseline, and How Fixing the Unmasking Order Unlocks Better Composition"**

### Challenged Assumption
The entire DLM acceleration literature (Fast-dLLM, EntropyCache, SlowFast, ES-dLLM, DyLLM, Window-Diffusion, etc.) implicitly assumes that the default confidence-based unmasking strategy is a reasonable baseline to accelerate. None of these papers question WHETHER the tokens being decoded at each step are the RIGHT tokens to decode. They focus exclusively on making each step computationally cheaper or skipping steps entirely, while accepting the default token-selection order.

### Evidence

**For the challenged assumption (conventional view):**
- Fast-dLLM (ICLR 2026) achieves 27.6x speedup with negligible quality loss on standard benchmarks using the default order.
- EntropyCache achieves lossless accuracy on HumanEval for both LLaDA and Dream.
- Mercury (closed-source) achieves 5-10x over AR at competitive quality -- suggesting the paradigm can work.
- For low-dependency tasks (infilling, simple completion), the unmasking order matters less.

**Against the challenged assumption (our evidence):**
- [Flexibility Trap (arXiv 2601.15165)](https://arxiv.org/abs/2601.15165): Confidence-based unmasking systematically bypasses high-entropy logical connectives ("Therefore", "Thus", "Since") that are critical for reasoning. This NARROWS the solution space.
- [LogicDiff (arXiv 2603.26771)](https://arxiv.org/abs/2603.26771): Replacing confidence-based with logic-role-guided unmasking improves GSM8K from 22.0% to 60.7% (+38.7pp) with <6% speed overhead. The reasoning deficit is in the ORDER, not the model.
- [ParallelBench (ICLR 2026)](https://arxiv.org/abs/2510.04767): Parallel decoding fails catastrophically when token dependencies are strong, because the confidence-based selection ignores dependency structure.
- [Theoretical Benefit and Limitation (NeurIPS 2025)](https://arxiv.org/abs/2502.09622): Achieving low sequence error rate requires O(L) steps, fundamentally because parallel sampling introduces irreducible errors in dependent token sequences.
- [Lost in Diffusion (arXiv 2604.10556)](https://arxiv.org/abs/2604.10556): DLMs exhibit unique hallucination modes (premature termination, incomplete denoising) linked to the decoding process, not model capacity.

### Hypothesis
**A two-stage "order-first" acceleration pipeline** -- (Stage 1) correct the unmasking order using dependency-aware scheduling, then (Stage 2) apply standard acceleration methods (KV caching, adaptive steps, parallel decoding) on top of the corrected order -- achieves a strictly better quality-speed Pareto frontier than applying the same acceleration methods to the default confidence-based order. Furthermore, acceleration methods compose more additively under the corrected order because dependency-aware scheduling reduces the interaction term between KV cache errors and parallel decoding errors.

### Method

**Stage 1: Lightweight Dependency-Aware Unmasking (training-free variant)**

Rather than LogicDiff's trained classifier, we propose a training-free proxy for dependency-aware unmasking:
- **Entropy-based priority inversion**: At each step, instead of unmasking the highest-confidence tokens first (which avoids hard tokens), unmask tokens with MODERATE confidence first (entropy in the middle tercile). Extreme-low-entropy tokens are likely trivially predictable and can wait; extreme-high-entropy tokens may benefit from more context; moderate-entropy tokens are at the decision boundary and resolving them provides maximum information gain for subsequent tokens.
- **Positional dependency heuristic**: For reasoning tasks, tokens at logical connective positions (identified by simple pattern matching on partial sequence or by embedding similarity to connective prototypes) receive priority boosting.
- **Formally**: priority(token_i, step_t) = alpha * entropy_mid(token_i, t) + beta * dep_score(token_i) + (1-alpha-beta) * confidence(token_i, t), where entropy_mid is a bell-shaped function of entropy peaked at the median, and dep_score is the positional dependency heuristic.

**Stage 2: Composed Acceleration**
- Apply Fast-dLLM's approximate KV cache (or EntropyCache's entropy-guided refresh) to the order-corrected decoding.
- Apply adaptive step scheduling (SlowFast-style slow/fast phase transition) to the order-corrected decoding.
- Measure: (a) quality at each speedup level, (b) composition additivity (is KV cache + step scheduling better together than each alone?), (c) comparison with AR + speculative decoding baseline.

**Stage 3: Composition Analysis**
- Implement 4 methods independently: (M1) order correction, (M2) KV caching, (M3) adaptive steps, (M4) parallel decoding.
- Test all 15 non-trivial combinations (single, pairwise, triple, quadruple).
- For each combination, measure the "composition ratio" = actual_speedup / product_of_individual_speedups.
- Hypothesis: composition ratio is closer to 1.0 (multiplicative) when M1 is included, and closer to 0.3-0.5 (subadditive) when M1 is absent.

### Experimental Plan

**Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (secondary).

**Baselines (properly tuned, no strawman)**:
- LLaDA-8B vanilla (T steps = sequence length, one token per step) -- quality ceiling
- LLaDA-8B + Fast-dLLM (default order, KV cache + parallel decoding) -- speed-optimized default
- LLaDA-8B + EntropyCache -- alternative cache strategy
- LLaMA3-8B + vLLM (no speculative decoding) -- AR baseline
- LLaMA3-8B + vLLM + speculative decoding (EAGLE-2) -- best AR baseline

**Benchmarks**:
- **Reasoning**: GSM8K (8-shot), MATH500 (4-shot)
- **Code**: HumanEval (0-shot), MBPP (3-shot)
- **Low-dependency**: CNN/DailyMail summarization, text infilling
- **Planning**: Countdown (from ParallelBench / Dream eval)

**Metrics**:
- Accuracy / Pass@1 (quality)
- Tokens per second (speed)
- AUP: Accuracy Under Parallelism (Hao AI Lab metric)
- Composition ratio (our metric)

**Hardware**: Single A100-80GB or H100, FP16.

**Time budget**:
- Pilot (order correction alone on GSM8K): ~30 minutes
- Single-method baselines (5 configs x 2 benchmarks): ~5 hours
- Composition study (15 combinations x 4 benchmarks): ~20 hours
- AR comparison: ~3 hours
- Total: ~30 hours, parallelizable to ~10 hours with 3 GPUs

### Risk Assessment

**What if the mainstream view turns out to be correct?**
- If composition IS near-multiplicative even with default order: We report this as a positive finding. The composition study itself is valuable regardless of whether our hypothesis about order-dependence is confirmed.
- If order correction provides negligible improvement on top of acceleration: This would mean the Flexibility Trap is already mitigated by aggressive parallel decoding (because high-entropy tokens get resolved eventually through multiple passes). We report the conditions under which order matters vs. does not.
- If the entropy-based priority inversion fails to match LogicDiff: We fall back to using LogicDiff's tiny classifier (4.2M params) as the order correction stage, accepting the minimal training requirement.

**Mitigation strategies**:
- The composition study alone is publishable even if the order-correction hypothesis fails.
- The systematic comparison against AR + speculative decoding is independently valuable.
- The composition ratio metric is a novel analytical tool regardless of specific results.

### Novelty Claim

The specific insight: **DLM acceleration methods compose subadditively BECAUSE the default unmasking order introduces correlated errors that multiply when methods are stacked.** Fixing the order decorrelates the errors, making composition more additive. This is the first paper to (a) identify unmasking order as the root cause of poor DLM acceleration composition, (b) propose a training-free order correction that enables better method composition, and (c) provide the first systematic composition analysis of >3 training-free DLM acceleration methods under both default and corrected decoding orders.

No existing paper addresses the interaction between decoding order and acceleration method composability. The Flexibility Trap and LogicDiff identify the order problem for quality but do not connect it to acceleration. Fast-dLLM, EntropyCache, and SlowFast optimize speed but do not question the order. Our contribution bridges these two lines of work.
