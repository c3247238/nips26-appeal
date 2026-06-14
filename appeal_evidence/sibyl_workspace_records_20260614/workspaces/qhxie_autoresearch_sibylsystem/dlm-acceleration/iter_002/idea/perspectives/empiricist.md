# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **"How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices"** (arXiv:2510.18480, 2025) -- The single most important methodological paper for this project. Demonstrates that speedup gains at batch=1 diminish or vanish at batch>=8. Proposes Tokens-Per-Forward-Step (TPF) as a unified efficiency metric. Key finding: LLaMA3-8B achieves 13.7x higher throughput than LLaDA-8B under fair batched conditions. Critical for establishing honest evaluation baselines.

2. **"Generative Frontiers: Why Evaluation Matters for Diffusion Language Models"** (arXiv:2604.02718, 2026) -- Shows that minor temperature changes are sufficient to completely reverse relative performance rankings between DLM methods. A single operating point comparison (even with both speed and accuracy) is unreliable. Proposes frontier analysis (Pareto curves over temperature/steps). Essential for avoiding false positives in our experiments.

3. **KLASS: KL-Guided Fast Inference in Masked Diffusion Models** (arXiv:2511.05664, NeurIPS 2025 Spotlight) -- Uses token-level KL divergence to identify stable high-confidence predictions, achieving 2.78x wall-clock speedup while *improving* accuracy. Code available (github.com/shkim0116/KLASS). Important because it validates KL divergence as a practical and theoretically grounded proxy signal, and because it explicitly compares against confidence-only methods showing that confidence alone is *insufficient* -- a crucial methodological lesson.

4. **"Theoretical Benefit and Limitation of Diffusion Language Model"** (arXiv:2502.09622, NeurIPS 2025) -- Proves that achieving low SER (sequence error rate) requires O(L) denoising steps, eliminating parallel speedup for reasoning/code tasks. Distinguishes TER (token-level) vs. SER (sequence-level) metrics. Essential for understanding which evaluation metric is appropriate for which task type.

5. **ParallelBench** (arXiv:2510.04767, ICLR 2026) -- Purpose-built benchmark for evaluating parallel decoding quality in DLMs. Shows confidence-based parallel decoding fails catastrophically on dependency-heavy tasks. Necessary control benchmark for any composition study.

6. **AUP: When Accuracy Meets Parallelism** (Hao AI Lab, UCSD, 2025) -- Proposes the Accuracy-Under-Parallelism metric for fair comparison. Key finding: AR + speculative decoding still achieves top AUP scores. All dLLM methods land on the same accuracy-parallelism curve. Provides the correct baseline comparison framework.

7. **COVER: Stop the Flip-Flop** (arXiv:2602.06161, 2026) -- Introduces context-preserving verification that avoids flip-flop oscillations in revocable decoding. Achieves up to 11.64x speedup on Dream-7B. Methodologically important for demonstrating a verification mechanism that prevents the degenerate cycling that plagues naive confidence-based approaches.

8. **ODB-dLLM: Orchestrating Dual-Boundaries** (arXiv:2511.21759, PKU, 2025) -- Hardware-aware acceleration combining adaptive length prediction with jump-share speculative decoding. Claims 46-162x speedup over baseline dLLM. Code available (github.com/PKU-SEC-Lab/ODB-dLLM). Important as a strong baseline that addresses the prefill/decode arithmetic intensity gap -- but the extreme speedup claims demand independent verification.

9. **"Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow"** (arXiv:2601.15593, 2026) -- Evaluates 8 MDLMs (up to 100B parameters) across 58 benchmarks. Key negative result: MDLMs still lag behind AR models, and parallel probabilistic modeling weakens inter-token dependencies. Provides the broadest empirical evaluation of MDLMs to date.

10. **"The Flexibility Trap"** (arXiv:2601.15165, 2026) + **LogicDiff** (arXiv:2603.26771, 2026) -- Together demonstrate that confidence-based unmasking systematically avoids high-entropy reasoning tokens. LogicDiff improves LLaDA-8B from 22.0% to 60.7% on GSM8K by changing only unmasking order. Critical confound: any acceleration method evaluated on reasoning benchmarks must control for the unmasking order variable.

11. **Lost in Diffusion** (arXiv:2604.10556, 2026) -- Documents hallucination patterns in dLLMs: premature termination, incomplete denoising, context intrusion. Dream-7B fails to recognize non-existent entities 98.5% of the time. Identifies failure modes that any acceleration method must not exacerbate.

12. **DARE: Diffusion LLM Alignment and Reinforcement Executor** (arXiv:2604.04215, 2026) -- Unified post-training + evaluation framework across LLaDA/Dream/SDAR/LLaDA2.x. Built on verl + OpenCompass. Most comprehensive standardized evaluation substrate available. The correct infrastructure choice for reproducible benchmarking.

### Experimental Landscape

**What has been properly tested:**
- KV caching for DLMs: Multiple independent implementations (Fast-dLLM, dKV-Cache, Elastic-Cache, EntropyCache, FreeCache) all confirm that inter-step KV similarity is exploitable for 10-25x single-sequence speedup. This is a robust finding.
- Confidence-based parallel decoding: Multiple papers confirm that aggressive parallelism degrades quality. This accuracy-parallelism tradeoff is well-established.
- Batch-size scaling: "How Efficient Are Diffusion LMs?" provides controlled evidence that DLM advantages erode at larger batch sizes.

**What is accepted without proper evidence:**
- **Composition claim**: No controlled experiment has systematically measured whether KV caching + step scheduling + parallel decoding compose multiplicatively, additively, or subadditively. Every paper measures its own technique in isolation or with ad-hoc single-method pairings.
- **Speedup generalization across tasks**: Most acceleration papers report on GSM8K + HumanEval. Whether the same method settings that work for math also work for code, general reasoning, planning, and long-context tasks is untested.
- **Per-token convergence rate predictions**: While KLASS, DyLLM, ES-dLLM, and EntropyCache all use different proxy signals for token importance (KL divergence, cosine similarity, tensor variation, entropy), no controlled comparison of these proxy signals under identical conditions exists.
- **Cross-model transferability**: Methods tuned on LLaDA-8B are assumed to work comparably on Dream-7B and vice versa, but the hyperparameter sensitivity across models is unmeasured.

**Where methodological gaps are most severe:**
- No standard protocol for measuring wall-clock speedup (hardware, batch size, sequence length, number of seeds all vary across papers).
- No standard for reporting quality degradation (some use accuracy, some perplexity, some BLEU/ROUGE, some pass@k; temperature settings vary and can reverse rankings).
- No falsification criteria: papers never specify what quality drop would constitute a failure of their method.

---

## Phase 2: Initial Candidates

### Candidate A: Controlled Composition Study with Proxy Signal Comparison (CompoBench)

- **Core hypothesis (falsifiable)**: The composition of the top three training-free DLM acceleration techniques (KV caching, adaptive step scheduling, token-level importance-based selective computation) yields at most 1.3x additional speedup over the single best technique alone, while maintaining <2% absolute accuracy degradation on GSM8K and HumanEval, when evaluated at both batch=1 and batch=8 on LLaDA-8B-Instruct with 3 random seeds.

- **Falsification criterion**: If any three-way composition achieves >2x improvement over the single best method at <2% quality loss on both benchmarks and both batch sizes, the subadditivity hypothesis is disproved.

- **Evaluation protocol**:
  - Models: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (generalization)
  - Benchmarks: GSM8K (8-shot, reasoning), HumanEval (0-shot, code), MMLU (5-shot, knowledge), ARC-Challenge (0-shot, general)
  - Metrics: Accuracy + wall-clock TPS + TPF (Tokens-Per-Forward-Step)
  - Statistical tests: Bootstrap 95% CI with 3 seeds (42, 123, 456); paired t-test for pairwise method comparisons
  - Batch sizes: 1, 4, 8
  - Temperature: Report Pareto frontier over T in {0.0, 0.3, 0.5, 0.7, 1.0} to avoid single-point cherry-picking per "Generative Frontiers"

- **Ablation plan**:
  - Layer 1 (KV cache): {No cache, Fast-dLLM cache, EntropyCache} -- isolate cache mechanism
  - Layer 2 (Step scheduling): {Uniform 64, Uniform 32, SlowFast, KLASS-style KL-guided} -- isolate scheduling strategy
  - Layer 3 (Token selection): {All tokens, DyLLM cosine similarity, ES-dLLM tensor variation, entropy-based} -- isolate which proxy signal is used for selective computation
  - All 3x3x4 = 36 combinations measured, plus the vanilla no-acceleration baseline = 37 total configurations per model per benchmark

- **Confounders identified**:
  - Temperature: controls whether acceleration or quality dominates; must sweep
  - Unmasking order: LogicDiff shows order matters for reasoning; must hold constant (use default confidence-based) to isolate acceleration effects
  - Hardware: results are hardware-specific; must report roofline characteristics
  - Random seed: must use 3+ seeds; report variance
  - Sequence length: fixed canvas size in DLMs means different tasks have different effective compute; must report separately

- **Pilot design**: Run Fast-dLLM vs. EntropyCache vs. SlowFast vs. Fast-dLLM+SlowFast vs. EntropyCache+SlowFast on GSM8K (100 examples, seed 42) with LLaDA-8B. 5 configs x ~10 min each = ~50 min. This tells us whether any two-method composition gives a meaningful boost over the single best.

### Candidate B: Information-Geometric Step Distillation (IGSD) with Falsification-First Design

- **Core hypothesis (falsifiable)**: Per-token KL divergence between consecutive denoising steps is a sufficient statistic for deciding whether to skip a step, and a simple threshold-based step skipper using this signal achieves equivalent quality to 64-step uniform decoding using at most 40 effective forward passes (37.5% step reduction) on GSM8K and HumanEval with LLaDA-8B-Instruct.

- **Falsification criterion**: If the best threshold setting requires >50 effective steps to match 64-step quality (within 1% absolute accuracy), then per-token KL divergence is an insufficient signal for step scheduling, and the hypothesis is disproved.

- **Evaluation protocol**:
  - Model: LLaDA-8B-Instruct
  - Benchmarks: GSM8K (reasoning), HumanEval (code), MMLU (knowledge)
  - Metrics: Accuracy at varying effective step budgets (10, 20, 30, 40, 50, 60, 64); wall-clock TPS
  - Statistical tests: Bootstrap 95% CI, 3 seeds
  - Comparison baselines: Uniform 64-step, uniform 32-step, KLASS, Saber (if reproducible)
  - Falsification analysis: Plot accuracy vs. effective steps; identify the minimum steps needed to reach 64-step quality per benchmark

- **Ablation plan**:
  - Signal comparison: {Per-token KL divergence, sequence-level mean KL, max-token KL, entropy change, cosine similarity of logits} -- which signal best predicts step skippability?
  - Threshold sweep: tau in {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} -- identify sensitivity
  - Per-token vs. global: Does per-token adaptive scheduling outperform global (same skip decision for all tokens)?
  - Phase analysis: Measure KL divergence across the full denoising trajectory to characterize which phases are skippable (early/middle/late)

- **Confounders identified**:
  - The KL computation itself has cost O(N*V); must account for this overhead in wall-clock measurements
  - Per-token KL may be noisy with small vocabulary probability differences; may need smoothing
  - The "critical middle phase" identified by model scheduling literature may have inherently high KL, making skipping impossible precisely when it would be most valuable
  - Quality metric choice (TER vs. SER) may give different conclusions per the NeurIPS 2025 theory paper

- **Pilot design**: Implement KL-threshold step skipper (sequence-level, simplest version: 5 lines of code added to LLaDA inference loop). Run on GSM8K (100 samples, seed 42) with thresholds {0.01, 0.05, 0.1}. Compare effective steps used vs. 64-step baseline accuracy. ~15 min.

### Candidate C: Proxy Signal Bake-Off with Controlled Confound Elimination

- **Core hypothesis (falsifiable)**: Among the four proxy signals used in the literature for token importance estimation (KL divergence per KLASS, cosine similarity per DyLLM, tensor variation per ES-dLLM, entropy per EntropyCache), KL divergence achieves the best rank correlation (Spearman rho > 0.85) with an oracle importance signal (measured by ablating each token and recording quality change), while having at most 2x the computational overhead of the cheapest signal (entropy).

- **Falsification criterion**: If no proxy signal achieves Spearman rho > 0.70 with the oracle signal, then all existing proxy signals are poor approximations of true token importance, and a fundamentally different approach is needed. If entropy achieves rho > 0.80 while being >5x cheaper than KL, then complexity is unjustified.

- **Evaluation protocol**:
  - Model: LLaDA-8B-Instruct
  - Dataset: 200 examples from GSM8K + 164 from HumanEval
  - Oracle measurement: For each token at each denoising step, compute accuracy when that token's computation is skipped vs. not skipped (expensive but gold-standard)
  - Proxy signals: {KL divergence, cosine similarity, L1 tensor variation, L2 tensor variation, entropy, confidence (max probability)}
  - Metrics: Spearman rank correlation with oracle per step phase (early/middle/late); computational cost in FLOPs per evaluation
  - Statistical tests: Bootstrap 95% CI for correlation estimates

- **Ablation plan**:
  - Phase analysis: Compute correlation separately for early (steps 1-20), middle (21-45), and late (46-64) phases
  - Task dependence: Is the best proxy different for reasoning vs. code tasks?
  - Layer dependence: ES-dLLM skips early layers; does signal quality depend on which layer we measure at?
  - Computational cost: Profile each signal in isolation; measure FLOPs and wall-clock overhead

- **Confounders identified**:
  - Oracle itself may be noisy (removing one token from context may have stochastic effects; need averaging over seeds)
  - Phase definition is arbitrary; sensitivity analysis over phase boundaries needed
  - Proxy signals may correlate with each other but not with the oracle (multicollinearity trap)
  - Different tokens may need different proxy signals (no single best)

- **Pilot design**: Compute all 6 proxy signals + oracle for 20 GSM8K examples at 3 representative denoising steps (step 5/early, step 32/middle, step 60/late). Compute rank correlation matrix. ~15 min per step x 3 = ~45 min. Immediately reveals whether proxy signals even correlate with the oracle.

---

## Phase 3: Self-Critique

### Against Candidate A (CompoBench -- Controlled Composition Study)

- **Confound attack**: The 36-configuration grid assumes that each layer (cache, scheduling, selection) can be independently toggled. In practice, some combinations may require non-trivial engineering integration (e.g., DyLLM's selective attention is entangled with how KV cache works). If integration requires ad-hoc modifications, the "composition" is no longer clean -- it becomes a new hybrid method. The confound is that implementation coupling may produce results that are artifacts of the specific integration rather than properties of the abstract composition.

- **Statistical attack**: With 37 configurations, 3 seeds each, and 4 benchmarks at 5 temperatures and 3 batch sizes, the full grid is 37 x 3 x 4 x 5 x 3 = 6,660 runs. Even at 10 min per run, that is 1,110 hours. The full grid is computationally infeasible. Must prioritize: start with a 2D grid (cache x scheduling, 3x4=12 configs) at batch=1 and T=0.5, then expand selectively. But selective expansion risks missing interactions that only emerge at specific settings.

- **Benchmark attack**: GSM8K, HumanEval, MMLU, and ARC-C are standard but may not capture the tasks where composition matters most. ParallelBench and planning tasks (Countdown, Sudoku) would be more diagnostic of parallel decoding quality. However, adding more benchmarks compounds the combinatorial explosion.

- **Ablation completeness attack**: The grid measures pairwise and three-way interactions but does not explain *why* certain combinations are subadditive. Without a causal mechanism, the study is purely descriptive. Needs to include diagnostic measurements (e.g., KV drift under different scheduling, token importance distribution under different caching) to provide explanatory power.

- **Verdict**: STRONG -- This fills the single biggest empirical gap and its design is methodologically rigorous. The combinatorial explosion is real but manageable with a phased approach. The descriptive nature is a weakness but is mitigated by including diagnostic measurements. No other paper has attempted this, and the results would be immediately actionable.

### Against Candidate B (IGSD -- KL-Threshold Step Skipper)

- **Confound attack**: The hypothesis that KL divergence between consecutive steps is sufficient for skip decisions assumes that important state changes are always reflected in logit distributions. However, hidden state changes may accumulate across layers before manifesting in the output logits -- the KL at the output layer could be low even when internal representations are changing significantly. This is the "delayed expression" confound: skipping a step might seem safe by the KL criterion but could propagate errors that only manifest later.

- **Statistical attack**: The falsification criterion (>50 steps to match 64-step quality within 1%) is relatively generous. If the method achieves 50 effective steps for 1% quality loss, that is only a 22% step reduction -- much less impressive than competing methods. The threshold should be tighter: can we get to 30 effective steps? Also, 1% absolute accuracy on GSM8K (where LLaDA-8B scores ~77%) is ~0.77 percentage points -- this may be within noise for 3 seeds.

- **Benchmark attack**: GSM8K and HumanEval are appropriate for reasoning/code. But MMLU (multiple choice) may not stress the denoising process as much because the output is short. Need to include a generation-heavy benchmark (e.g., code generation with long outputs) where step count matters more.

- **Ablation completeness attack**: The signal comparison ablation is strong. But it misses a key control: does the KL signal correlate with actual downstream quality degradation? The signal might be a poor predictor of quality loss even if it captures distributional change. Need to explicitly measure: "when KL is low and we skip, what is the actual quality impact?"

- **Verdict**: MODERATE -- The core idea is sound and the implementation is trivially simple (5 lines of code). But the novelty is limited: KLASS already uses KL divergence for adaptive unmasking (not step skipping, but related). The contribution would be the controlled comparison across multiple proxy signals and the explicit falsification framework. The risk is that the results confirm what KLASS already suggests without adding much.

### Against Candidate C (Proxy Signal Bake-Off)

- **Confound attack**: The oracle signal (accuracy when a single token's computation is skipped) is methodologically problematic. Skipping one token from attention changes the context for all other tokens. The oracle measures the effect of removing a token from computation, not the importance of recomputing that token. These are subtly different: a token might be unimportant to skip (low oracle impact) but important to recompute (its new value matters for future steps). The oracle conflates immediate quality impact with multi-step propagation.

- **Statistical attack**: With 200 + 164 = 364 examples, ~256 tokens per example, and ~64 steps, the oracle requires 364 x 256 x 64 = ~6M forward passes for full coverage. This is completely infeasible. Must sample: 20 examples x 50 tokens x 3 steps = 3,000 forward passes. But 20 examples may not be representative, and sampling 50 tokens risks missing the tail of the importance distribution (rare high-importance tokens).

- **Benchmark attack**: The oracle is defined as accuracy change when a token is skipped. But this accuracy is measured at the *end* of the full denoising process. A token that matters at step 10 might not matter if its effect is corrected by steps 30-64. The oracle conflates step-local importance with global importance. A proper oracle would need to measure importance per-step, which further compounds the computational cost.

- **Ablation completeness attack**: The phase analysis (early/middle/late) is a strength, but the phase boundaries are arbitrary. A continuous analysis (sliding window over step number) would be more informative. Also, the study does not address whether the *best* proxy signal depends on the caching method being used -- DyLLM's cosine similarity might be best when no cache is used, while EntropyCache's entropy might be best when KV caching is active.

- **Verdict**: MODERATE -- The question is important but the oracle definition is problematic and the computational cost is prohibitive for a thorough study. A simplified version (proxy correlation with *each other* rather than with an oracle, plus end-to-end accuracy at different proxy thresholds) would be more feasible and still useful. The proxy inter-correlation study is actually novel and useful in its own right.

---

## Phase 4: Refinement

### Dropped Ideas

**Candidate C (Proxy Signal Bake-Off)** is dropped as a standalone proposal due to the intractable oracle computation and the problematic oracle definition. However, the core insight -- comparing proxy signals under controlled conditions -- is incorporated into Candidate A as a sub-ablation and into Candidate B as a signal comparison experiment.

### Strengthened Survivors

**Candidate A (CompoBench)** strengthened with:
1. **Phased design**: Phase 1: 2D grid (3 cache x 4 scheduling = 12 configs) at batch=1 on GSM8K + HumanEval, 3 seeds. Phase 2: Add token selection layer for promising cache-scheduling pairs (top 3 x 4 selection = 12 configs). Phase 3: Add Dream-7B, batch=4/8, more benchmarks for best combinations. Total: ~36 + 12 + 20 = ~68 runs x 30 min = ~34 hours = feasible.
2. **Diagnostic measurements**: At each configuration, record (a) per-step KV drift magnitude, (b) per-token confidence distribution, (c) effective parallelism ratio, (d) cache hit rate. These explain *why* certain compositions work or fail.
3. **Pareto frontier analysis per "Generative Frontiers"**: Sweep temperature and report the full Pareto curve, not single operating points.
4. **AR baseline**: Include LLaMA3-8B with vLLM + EAGLE-2 speculative decoding as the gold standard comparison at all batch sizes.

**Candidate B (IGSD)** strengthened with:
1. **Tighter falsification criterion**: Must reach <40 effective steps for <1% quality drop. If this fails, the signal is insufficient.
2. **Explicit KL overhead accounting**: The KL computation cost is included in the wall-clock measurements, not ignored.
3. **Multi-signal comparison embedded**: The step skipper is tested with all 4 proxy signals (KL, cosine similarity, tensor variation, entropy), not just KL. This absorbs the useful part of Candidate C.
4. **Phase characterization**: Map the KL landscape across the full 64-step trajectory to identify inherent skippability windows. This is a pure measurement contribution independent of whether the step skipper works well.
5. **Composability test**: Combine the step skipper with the best KV cache method from Candidate A results. Report whether step reduction + KV caching compose.

### Selected Front-Runner

**Candidate A (CompoBench)** is selected as the front-runner for the following reasons:

1. **Fills the single largest empirical gap**: No controlled composition study exists. Every perspective document in this workspace identifies "Gap 11 -- Composition of acceleration methods" as a top priority. The pragmatist, contrarian, theoretical, and innovator perspectives all independently highlight this gap.

2. **Maximum experimental rigor**: The phased design with full Pareto frontier analysis, multiple batch sizes, AR baseline, diagnostic measurements, and 3+ seeds sets a new methodological standard for the field.

3. **Falsifiable and decisive**: The subadditivity hypothesis either holds or it does not. The study produces clear, actionable results regardless of outcome.

4. **Absorbs contributions from other candidates**: The proxy signal comparison from Candidate C and the KL-based step scheduling from Candidate B are embedded as sub-experiments within the composition study.

5. **Practical impact**: Produces the first "acceleration recipe" -- a recommended combination of methods with specific hyperparameters for each task type. This is what practitioners actually need.

---

## Phase 5: Final Proposal

### Title

**CompoAccel: A Controlled Composition Study of Training-Free Acceleration Methods for Diffusion Language Models**

### Hypothesis

The composition of training-free DLM acceleration methods operating on orthogonal computational axes (KV cache approximation, denoising step scheduling, token-level selective computation) is **subadditive**: the combined speedup is significantly less than the product of individual speedups (specifically, the composition ratio = combined_speedup / product_of_individual_speedups < 0.5) while quality degradation is superadditive (combined quality loss > sum of individual quality losses). This subadditivity is task-dependent: reasoning tasks (GSM8K) exhibit stronger subadditivity than code tasks (HumanEval) due to higher inter-token dependency in reasoning chains.

### Falsification Criterion

The hypothesis is **disproved** if:
- The composition ratio exceeds 0.7 for ANY three-way combination at <2% accuracy loss on BOTH GSM8K and HumanEval at batch=1, OR
- The combined quality loss is less than 1.2x the sum of individual quality losses for the majority (>60%) of combinations.

If the composition ratio is between 0.5 and 0.7, the result is **inconclusive** and requires expanded evaluation.

### Method

**Phase 1 -- Individual Method Baselines (2 hours)**
Implement and benchmark 10 individual configurations on LLaDA-8B-Instruct:
- KV cache: {None, Fast-dLLM, EntropyCache}
- Step scheduling: {Uniform-64, Uniform-32, SlowFast, KL-threshold (new)}
- Token selection: {All tokens, DyLLM-style cosine selection}

For each: measure accuracy (GSM8K 100 samples, HumanEval 164 problems), wall-clock TPS, and effective step count. Three seeds. Batch=1.

**Phase 2 -- Pairwise Composition (4 hours)**
Test all feasible pairwise combinations from Phase 1 (estimated 12-15 pairs). Measure the same metrics. Compute the composition ratio for each pair.

**Phase 3 -- Three-Way Composition (3 hours)**
Test the top 5 three-way combinations identified from Phase 2 Pareto analysis. Measure at batch=1 and batch=8.

**Phase 4 -- Cross-Model Generalization (3 hours)**
Replicate the top 3 configurations on Dream-7B-Instruct to test hyperparameter transferability.

**Phase 5 -- AR Baseline Comparison (2 hours)**
Benchmark LLaMA3-8B with vLLM + speculative decoding at batch=1, 4, 8 on the same benchmarks. Establish the "honest" speed-quality comparison.

**Phase 6 -- Diagnostic Analysis (2 hours)**
For all compositions: record per-step KV drift, effective parallelism, cache hit rate, token confidence histograms. Explain composition behavior.

### Evaluation Protocol

- **Primary benchmarks**: GSM8K (8-shot, 100-1319 samples), HumanEval (0-shot, 164 problems)
- **Secondary benchmarks**: MMLU (5-shot, 1000 samples), ARC-Challenge (0-shot)
- **Metrics**: Accuracy with bootstrap 95% CI; wall-clock tokens-per-second; Tokens-Per-Forward-Step (TPF)
- **Statistical tests**: Paired bootstrap test for accuracy differences; Wilcoxon signed-rank test for speedup ratios
- **Temperature sweep**: T in {0.0, 0.5, 1.0} for Pareto frontier (per "Generative Frontiers" recommendations)
- **Random seeds**: 42, 123, 456 (minimum 3)
- **Batch sizes**: 1 (interactive), 4 (small batch), 8 (realistic serving)

### Ablation Schedule

| Ablation | What it tests | Expected outcome |
|----------|--------------|-----------------|
| Remove KV cache from best composition | Contribution of caching | Largest individual speedup contributor (10-20x alone) |
| Remove step scheduling from best composition | Contribution of scheduling | Moderate contributor (1.5-3x alone) |
| Remove token selection from best composition | Contribution of selective computation | Moderate contributor (2-5x alone) |
| Swap EntropyCache for Fast-dLLM cache in best composition | Cache mechanism sensitivity | Similar speedup, possibly different quality tradeoff |
| Swap KL-threshold for SlowFast scheduling | Scheduling mechanism sensitivity | Comparable if both exploit the same temporal structure |
| Vary temperature in best composition | Sensitivity to generation temperature | Rankings may shift per "Generative Frontiers" |
| Vary batch size (1->8) in best composition | Scaling behavior | Expect speedup advantage to erode at batch=8 |
| Apply best LLaDA-8B config to Dream-7B | Cross-model transfer | Expect <30% hyperparameter transfer degradation if methods are model-agnostic |

### Control Experiments

1. **Vanilla baseline**: LLaDA-8B-Instruct with 64-step uniform decoding, no acceleration. Establishes quality ceiling.
2. **AR baseline**: LLaMA3-8B with vLLM + EAGLE-2 speculative decoding. Establishes speed ceiling.
3. **Oracle upper bound**: Run each acceleration method with a per-method oracle that knows the optimal hyperparameters for each example. Measures the gap between practical and optimal performance.
4. **Random combination control**: Randomly combine cache/scheduling/selection decisions per step. If random composition performs comparably to systematic composition, the systematic study adds no value.
5. **Unmasking order control**: Run the best composition with both default (confidence-based) and reversed (anti-confidence) unmasking order. If results flip, acceleration is confounded with decoding order.

### Pilot Design

**Duration**: <15 minutes.
**Protocol**: Run 5 configurations {vanilla-64, Fast-dLLM-only, EntropyCache-only, SlowFast-only, Fast-dLLM+SlowFast} on GSM8K (100 samples, seed 42) with LLaDA-8B-Instruct at batch=1, T=0.5.
**Decision criterion**: If Fast-dLLM+SlowFast achieves <10% additional speedup over the faster of {Fast-dLLM-only, SlowFast-only} alone, the subadditivity hypothesis is supported and the full study proceeds. If it achieves >50% additional speedup, the hypothesis is challenged and the study must explore why.

### Resource Estimate

- **GPU**: 1x A100 80GB or equivalent
- **Model**: LLaDA-8B-Instruct (~16GB FP16), Dream-7B-Instruct (~14GB FP16), LLaMA3-8B (~16GB FP16)
- **Pilot**: ~15 minutes
- **Phase 1-2**: ~6 hours
- **Phase 3-6**: ~10 hours
- **Total full experiment**: ~16-20 hours across 2-3 days
- **Codebase**: Build on Fast-dLLM (NVlabs) + EntropyCache (mscheong01) + SlowFast (LiangrunFlora). Minimal new code (~200-300 lines for composition framework + KL-threshold scheduler).

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Integration complexity across codebases | HIGH | Start with Fast-dLLM as the base; add EntropyCache's signal as a drop-in; add SlowFast as a wrapper. Test each pair before attempting three-way. |
| Combinatorial explosion of configurations | HIGH | Phased approach: 2D grid first, expand selectively. Use sequential elimination (drop methods that show no benefit in Phase 1). |
| Results are hardware-specific | MEDIUM | Report roofline characteristics; measure on both A100 and H100 if available; report FLOPs in addition to wall-clock time. |
| Temperature sensitivity invalidates single-point results | MEDIUM | Sweep temperature and report Pareto frontier for every configuration. |
| Methods interact in unexpected ways (non-monotonic composition) | MEDIUM | Full factorial on the critical 2D grid; diagnostic measurements to explain anomalies. |
| Negative result (all compositions are subadditive) is hard to publish | LOW | A rigorous negative result with diagnostic explanations IS publishable (cf. "How Efficient Are Diffusion LMs?"). Pair with the proxy signal comparison as a constructive contribution. |

### Novelty Claim

This paper makes the following specific empirical contributions that have not been made before:

1. **First controlled composition study**: The first systematic, factorial measurement of how training-free DLM acceleration methods compose, with the composition ratio quantified across task types, batch sizes, and generation temperatures.

2. **First cross-signal comparison under identical conditions**: The first study comparing KL divergence, cosine similarity, tensor variation, and entropy as proxy signals for token importance, using the same model, benchmarks, and evaluation protocol.

3. **First honest DLM-vs-AR comparison with composition**: The first comparison between the best *composed* DLM acceleration (not single-method) against a properly optimized AR baseline (vLLM + speculative decoding) at realistic batch sizes.

4. **Diagnostic characterization of composition behavior**: The first measurements of how KV drift, effective parallelism, and token confidence distributions change under composed acceleration, explaining *why* certain combinations succeed or fail.

5. **Reproducible methodology**: A standardized composition evaluation framework with open-source code, following "Generative Frontiers" best practices, that other researchers can use to benchmark future acceleration methods.
