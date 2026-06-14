# Empiricist Perspective

## Phase 1: Literature Survey

### Methodology Resources

1. **"How Efficient Are Diffusion Language Models? A Critical Examination of Efficiency Evaluation Practices"** (arXiv 2510.18480) -- Proposes Tokens-Per-Forward-Step (TPF) as a unified efficiency metric; identifies fundamental reproducibility issues in DLM benchmarking; demonstrates that open-source DLMs are paradoxically slower than AR models under fair comparison. This is the single most important methodological reference for our study.

2. **"Generative Frontiers: Why Evaluation Matters for DLMs"** (arXiv 2604.02718) -- Shows that generative perplexity and unigram entropy correspond to the two KL-divergence components; demonstrates that single-metric comparisons are unreliable across different ELBO formulations. Critical for understanding why we need multi-metric evaluation.

3. **EntropyCache unified benchmark** (arXiv 2603.18489) -- Re-implemented Fast-dLLM, Elastic-Cache, d2Cache, and baseline parallel decoding within a single codebase for fair head-to-head comparison. This is the closest existing work to a unified KV-cache evaluation and provides the methodology template for our composability study.

4. **DARE: Diffusion LLM Alignment and Reinforcement Executor** (arXiv 2604.04215) -- Unified post-training + evaluation framework covering LLaDA, Dream, SDAR, LLaDA2.x; standardizes evaluation across MDLMs and BDLMs. Provides the infrastructure substrate for reproducible DLM evaluation.

5. **JoT: Just on Time** (arXiv 2602.11133) -- Explicitly claims orthogonality to KV-caching methods but does NOT empirically validate the combination. This unvalidated claim is a direct experimental opportunity: testing JoT + EntropyCache composition would be a novel data point.

6. **SlowFast Sampling** (arXiv 2506.10848) -- The ONLY published two-method composition in DLM acceleration (SlowFast + dLLM-Cache, achieving 34.22x). Provides the sole composability baseline but covers only one pair within the same method family (sampling + caching). No cross-family composition was tested.

7. **COVER: Context-Preserving Verification** (arXiv 2602.06161) -- Introduces KV-cache-override verification for revocable diffusion decoding; the dual-view KV mechanism (masked seeds vs. cached context) is a verification primitive that could interact with KV caching methods in non-obvious ways.

8. **SSD: Self Speculative Decoding for Diffusion LLMs** (arXiv 2510.04147) -- Introduces the concept of using a DLM as both drafter and verifier. The verification strategy differs fundamentally from AR speculative decoding because DLMs produce per-position marginals, not sequential conditionals. Critical prior art for any IGSD variant.

9. **"Theoretical Benefit and Limitation of Diffusion Language Model"** (arXiv 2502.09622) -- Shows that the efficiency conclusion is highly sensitive to metric choice: MDMs achieve near-optimal Token Error Rate (TER) with constant steps, but Sequence Error Rate (SER) requires linear steps. This theoretical result must inform our experimental design.

10. **AbGen: Evaluating LLMs in Ablation Study Design** (ACL 2025) -- Meta-methodology paper on ablation study design quality; highlights gap between automated evaluation and human assessment. Relevant for validating our composability ablation design.

11. **Fast-dLLM** (ICLR 2026, arXiv 2505.22618) -- Establishes the block-wise KV cache + confidence-aware parallel decoding paradigm. The ICLR acceptance and NVIDIA backing make this the de facto standard baseline for training-free DLM acceleration.

12. **Elastic-Cache** (ICLR 2026, arXiv 2510.14973) -- Adaptive layer-aware cache refresh; achieves 45.1x on long sequences. The depth-aware selective recomputation is mechanistically different from EntropyCache's token-entropy trigger, creating a clear axis for composability analysis.

### Experimental Landscape

**What has been properly tested (single-method):**
- KV caching methods have been individually benchmarked extensively (Fast-dLLM, EntropyCache, Elastic-Cache, dKV-Cache, SPA-Cache, Sparse-dLLM). EntropyCache provides the most rigorous unified comparison within this family.
- Early stopping methods (JoT, Prophet, SchED) have been tested individually on LLaDA-8B and Dream-7B across GSM8K, MMLU, HellaSwag, HumanEval.
- Speculative decoding variants (SSD, COVER, S2D2, DualDiffusion) have been tested individually but on different models and benchmarks.

**What is accepted without proper evidence:**
- JoT's claim of orthogonality to KV caching: STATED but NEVER EMPIRICALLY VALIDATED in combination.
- The assumption that acceleration methods targeting different "dimensions" (per-step cost vs. step count vs. per-token computation) compose linearly. No paper tests this.
- The implicit assumption in many papers that speedup numbers from different papers on the same model are comparable. Hardware, batch size, sequence length, number of denoising steps, confidence thresholds, and decoding strategies all vary.

**Where methodological gaps exist:**
- No paper has tested more than one cross-family combination. SlowFast + dLLM-Cache is the only published composition and covers a single pair within the same family.
- No paper reports interaction effects (super-additive or sub-additive speedup when combining methods).
- No paper systematically identifies failure modes of acceleration composition -- where combining two individually-good methods produces degraded quality or paradoxically slower inference.
- Evaluation metrics vary wildly across papers: some report tokens/second, others report speedup ratios relative to different baselines, others report step reduction. No paper uses a unified multi-metric protocol for composability analysis.

---

## Phase 2: Initial Candidates

### Candidate A: ComposeAccel -- Systematic Composability Atlas for Training-Free DLM Acceleration

- **Hypothesis**: Training-free DLM acceleration methods from different families (KV caching, early stopping, speculative decoding) can be composed, but the interaction is non-trivial: some pairs are super-additive (synergistic), some are sub-additive (interfering), and some produce quality-degrading failure modes invisible when testing methods individually. Specifically, we predict that KV caching (EntropyCache) + token-level early stopping (JoT) will achieve >80% of the product of individual speedups while retaining >95% of baseline quality on MMLU/GSM8K.

- **Falsification criterion**: If NO pair of methods from different families achieves >60% of the product of their individual speedups while retaining >90% baseline quality, the composability thesis is falsified. If ALL pairs achieve >90% of the product, the failure-mode atlas contribution is nullified (methods compose trivially).

- **Evaluation protocol**:
  - Models: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (validation)
  - Benchmarks: MMLU (general reasoning), GSM8K (math), HumanEval (code), HellaSwag (commonsense) -- covering the four standard DLM evaluation domains
  - Metrics: (1) Accuracy on each benchmark, (2) Wall-clock tokens/second on identical hardware, (3) FLOPs/token estimated via profiling, (4) Composability ratio = actual combined speedup / (speedup_A x speedup_B), (5) Quality retention = combined accuracy / max(individual accuracies)
  - Statistical tests: Bootstrap 95% CI over 3 seeds; paired t-test for each pairwise comparison; Bonferroni correction for multiple comparisons across method pairs
  - Hardware: Single A100 80GB (or whatever is available), fixed across all experiments

- **Ablation plan**:
  - A1: Each method alone (EntropyCache, JoT, IGSD) vs. vanilla baseline -- establishes individual effect sizes
  - A2: Each pairwise combination -- measures interaction effects
  - A3: Triple combination -- tests whether three-way composition is feasible
  - A4: Sensitivity sweep: vary confidence threshold (for parallel decoding), entropy threshold (for cache refresh), early-stopping threshold (for JoT) -- identifies which hyperparameters dominate the interaction

- **Confounders identified**:
  - C1: Shared hyperparameter coupling -- both EntropyCache and JoT use confidence-like thresholds; they may compete for the same "easy" tokens, creating an illusion of orthogonality when actually double-counting
  - C2: Sequence length dependence -- composition effects may differ dramatically between short (128-token) and long (1024-token) generation
  - C3: Task-dependent interactions -- methods may compose well on MMLU (short answers) but fail on GSM8K (long chain-of-thought) or HumanEval (structured code)
  - C4: Denoising step count sensitivity -- most papers use different numbers of total steps; the composition landscape may shift at different step budgets

- **Pilot design**: Run EntropyCache alone, JoT alone, and EntropyCache+JoT on LLaDA-8B-Instruct with 100 samples from GSM8K. Measure: (1) do they produce valid outputs when combined? (2) what is the composability ratio? (3) are there obvious failure modes (output collapse, infinite loops, NaN)? Target: <15 minutes on a single GPU.

### Candidate B: KV Cache Pareto Frontier -- Unified Comparison of Cache Refresh Strategies

- **Hypothesis**: The Pareto frontier of KV cache refresh strategies for DLMs is dominated by a small number of design choices (refresh trigger type x refresh granularity x layer-awareness), and the optimal strategy varies systematically with task type. Specifically, entropy-triggered refresh (EntropyCache) dominates on short-generation tasks, while attention-aware refresh (Elastic-Cache) dominates on long-generation tasks.

- **Falsification criterion**: If a single method dominates the Pareto frontier across ALL task types and generation lengths, the task-dependence hypothesis is falsified. If the frontier is noisy with no clear pattern, the design-choice taxonomy is not explanatory.

- **Evaluation protocol**:
  - Methods: EntropyCache, Elastic-Cache, Fast-dLLM (block-wise), dKV-Cache, SPA-Cache, Sparse-dLLM, d2Cache, vanilla (no cache)
  - Model: LLaDA-8B-Instruct
  - Benchmarks: GSM8K (medium generation), HumanEval (structured), MMLU (short), HellaSwag (short), CNN/DailyMail (long generation)
  - Metric: Accuracy vs. tokens/second Pareto curve; area-under-Pareto-curve (AUP-like metric)
  - Unified codebase: Extend EntropyCache's existing unified implementation to include remaining methods

- **Ablation plan**:
  - B1: Vary refresh frequency for each method on a fixed task -- trace the quality-speed tradeoff curve
  - B2: Fix speedup target (e.g., 10x) and compare quality across methods and tasks
  - B3: Profile per-step FLOPs breakdown -- identify where each method saves computation

- **Confounders identified**:
  - Different papers use different confidence thresholds for parallel decoding alongside caching -- must fix this to isolate cache effect
  - Implementation quality differences -- must re-implement all in a unified codebase (EntropyCache already provides a partial foundation)
  - Warm-up effects: first few denoising steps may have different cache dynamics than steady-state

- **Pilot design**: Re-run EntropyCache's unified comparison on 50 GSM8K samples, extending to include Elastic-Cache if not already present. Verify reproduction of published numbers. <15 minutes.

### Candidate C: Self-Speculative Verification for Pure MDMs -- Controlled Experiment on Draft Quality vs. Verification Cost

- **Hypothesis**: In pure MDMs (LLaDA, Dream), using fewer denoising steps as a "draft" and more steps as "verification" (the IGSD concept) achieves a better speed-quality Pareto than simply reducing step count uniformly, because the verification step can catch and correct errors introduced by the coarse draft. Specifically, a 4-step draft + 2-step verify will match the quality of 8-step uniform decoding at 60-70% of the wall-clock time.

- **Falsification criterion**: If N-step draft + M-step verify (for any N+M < full steps) produces WORSE quality than (N+M)-step uniform decoding at the SAME total compute, the speculative approach adds overhead without benefit and the hypothesis is falsified.

- **Evaluation protocol**:
  - Model: LLaDA-8B-Instruct
  - Benchmarks: GSM8K, MMLU, HumanEval
  - Comparison: (1) K-step uniform, (2) N-step draft + M-step verify where N+M=K, (3) SSD baseline
  - Key metric: Quality at iso-compute (same total FLOPs)
  - Statistical: 3 seeds, bootstrap CI

- **Ablation plan**:
  - C1: Sweep draft-to-verify ratio: 2+6, 3+5, 4+4, 5+3, 6+2 for 8 total steps
  - C2: Vary the acceptance criterion: strict (reject if any token changed), medium (reject if >20% changed), loose (accept always)
  - C3: Token-freezing strategy: freeze high-confidence tokens from draft vs. recompute all in verify

- **Confounders identified**:
  - The "draft" and "verify" phases in MDMs are not cleanly separable like in AR speculative decoding -- both are denoising passes on the same masked sequence
  - Acceptance criterion for MDMs is fundamentally different from AR (per-position marginals vs. sequential conditionals) -- SSD addresses this but the criterion may not generalize
  - Compute is not perfectly proportional to step count due to varying numbers of masked tokens per step

- **Pilot design**: Implement the simplest version: run LLaDA-8B with 4 steps (draft), freeze top-50%-confidence tokens, run 4 more steps on remaining tokens (verify). Compare to 8 uniform steps on 50 GSM8K samples. <15 minutes.

---

## Phase 3: Self-Critique

### Against Candidate A: ComposeAccel Composability Atlas

- **Confound attack**: The biggest confound is hyperparameter coupling. EntropyCache uses a decoded-token-entropy threshold to decide when to refresh the cache. JoT uses a prediction-confidence threshold to decide when to stop refining a token. Both thresholds interact with the parallel-decoding confidence threshold. When combining methods, the three thresholds form a 3D hyperparameter space that has NEVER been jointly explored. The reported "orthogonality" of JoT may only hold at JoT's default threshold when caching is absent. The interaction surface could be highly non-convex, with good operating points being narrow. **Mitigation**: Grid search over a 3x3x3 threshold grid for the pilot, then refine. This is expensive but necessary.

- **Statistical attack**: With 3 seeds and 4 benchmarks x ~6 method configurations (3 singles, 3 pairs, 1 triple), we have 24 comparisons. With Bonferroni correction, alpha = 0.05/24 = 0.002 per test. Given typical variance in DLM evaluation (2-5% accuracy std across seeds on GSM8K), detecting a 5% quality degradation from composition requires at least 3 seeds, which is borderline. For detecting subtle interaction effects (e.g., 2% quality drop), we would need 5+ seeds. **Mitigation**: Use 5 seeds for the primary experiments; use bootstrap CI rather than parametric tests.

- **Benchmark attack**: MMLU and HellaSwag are primarily evaluated in a zero-shot / few-shot classification format, where DLMs need to generate only a short answer token or letter. Composition effects may be trivially positive on such short-generation tasks because most acceleration methods have negligible impact when generation length is tiny. The real stress test is GSM8K (chain-of-thought, ~100-200 tokens) and HumanEval (code, ~50-150 tokens). We should add at least one long-generation benchmark (e.g., CNN/DailyMail summarization or MBPP). **Risk**: Without a long-generation benchmark, the atlas may be misleadingly positive.

- **Ablation completeness attack**: The proposed ablation (individual, pairwise, triple) is the right structure but misses a critical dimension: the ORDER of application. If EntropyCache modifies the KV state and then JoT makes early-stopping decisions based on predictions conditioned on cached (approximate) KVs, the composition result may depend on the implementation order. Are tokens stopped early by JoT included in EntropyCache's entropy calculation? This implementation-level detail could dominate the interaction effect. **Mitigation**: Test both orderings where ordering is meaningful (e.g., cache-then-stop vs. stop-then-cache).

- **Verdict**: **STRONG** -- The experimental question (do methods compose?) is clean, falsifiable, and no one has answered it. The confounds are real but manageable with careful experimental design. The main risk is that the hyperparameter interaction space is large, but a structured grid search with pilot-informed narrowing is feasible within the time budget.

### Against Candidate B: KV Cache Pareto Frontier

- **Confound attack**: EntropyCache already provides a partial unified comparison (Fast-dLLM, Elastic-Cache, d2Cache). Extending this to 8 methods requires re-implementing SPA-Cache, Sparse-dLLM, and dKV-Cache in the same codebase -- a significant engineering effort with high risk of implementation bugs that could confound the comparison. Each method has subtle implementation details (e.g., Elastic-Cache's depth-aware schedule requires layer-by-layer profiling) that may not transfer cleanly. **Risk**: Incorrect re-implementations would invalidate the entire study.

- **Statistical attack**: Pareto frontier comparison is inherently ordinal (dominance) rather than metric (statistical test). With 8 methods x 5 benchmarks, we're comparing 40 Pareto points. Standard statistical tests for Pareto dominance are weak and often inconclusive. **Risk**: We may produce nice-looking Pareto plots without being able to make statistically rigorous claims.

- **Benchmark attack**: This is fundamentally a systems benchmarking paper, and the value depends entirely on how many methods are included and how fairly they're compared. EntropyCache already covers the most important methods. Our marginal contribution over EntropyCache's Table 1 is unclear -- we'd be adding 2-3 more methods and 1-2 more benchmarks. **Risk**: This may be seen as an incremental extension of EntropyCache's already-published comparison rather than a novel contribution.

- **Ablation completeness attack**: Profiling per-step FLOPs (B3) is informative but not an ablation in the traditional sense. The real ablation would be to decompose each method into its component design choices (refresh trigger, granularity, layer-awareness) and test each component independently. This is essentially redesigning 8 methods, which is beyond scope.

- **Verdict**: **MODERATE** -- Scientifically valuable but faces three problems: (1) high engineering effort with implementation-correctness risk, (2) marginal novelty over EntropyCache's existing comparison, (3) difficult to make statistically rigorous claims about Pareto dominance. Better as a supporting experiment within a larger study than as a standalone contribution.

### Against Candidate C: Self-Speculative for Pure MDMs

- **Confound attack**: The fundamental problem is that "draft" and "verify" in MDMs are not cleanly separable. In AR speculative decoding, the draft model generates a complete continuation and the target model verifies it in a single parallel pass. In MDMs, both phases are denoising passes over the same sequence -- the "verifier" sees the draft's partially-demasked sequence and continues denoising. This means the "verification" is not independent: it's conditioned on the draft's specific token choices, creating a confound between draft quality and verification power. SSD addresses this with a self-drafting mechanism, but our IGSD variant's token-freezing approach introduces a different confound: freezing high-confidence tokens may cause the verifier to compensate for frozen errors rather than detecting them.

- **Statistical attack**: The expected effect size is small. If 8-step uniform decoding achieves X% accuracy, we're hoping that 4+4 draft-verify achieves X% at 60-70% of the wall-clock time. But the time savings from the verify phase (which only denoises unfrozen tokens) depends on how many tokens are frozen, which varies per sample. With high variance in frozen-token count, the speedup distribution will be wide, requiring many samples for reliable measurement. **Risk**: Borderline statistical significance with 3 seeds.

- **Benchmark attack**: SSD (arXiv 2510.04147) already demonstrates self-speculative decoding for DLMs. DualDiffusion (arXiv 2604.05250) also proposes a draft+verify framework for MDMs. Our IGSD variant's novelty over SSD is the specific token-freezing mechanism and its interaction with KV caching. As a standalone method, the novelty is limited (4/10 per the literature survey's assessment). The value is as a composability analysis vehicle, not as a standalone contribution.

- **Ablation completeness attack**: The draft-to-verify ratio sweep (C1) is well-designed. However, the acceptance criterion ablation (C2) is under-specified: what does "reject" mean in the MDM context? In AR speculative decoding, rejection means reverting to the target model's distribution at the first rejected token. In MDMs, rejection could mean re-masking tokens, re-running from scratch, or accepting with degraded confidence. Each choice has different compute and quality implications. **Risk**: Without a principled acceptance criterion grounded in MDM probability theory, the ablation is ad hoc.

- **Verdict**: **MODERATE** -- The concept is sound but faces strong prior art (SSD, DualDiffusion) and fundamental conceptual challenges in defining clean draft/verify semantics for MDMs. Best pursued as part of the composability study (Candidate A) where its value is as an additional method to compose, not as the primary contribution.

---

## Phase 4: Refinement

### Dropped
- **Candidate B (KV Cache Pareto Frontier)** is dropped as the primary contribution because: (1) EntropyCache already provides a substantial unified comparison, reducing our marginal novelty; (2) the engineering effort of re-implementing 8 methods in a unified codebase is disproportionate to the expected contribution; (3) Pareto frontier claims are statistically weak. However, elements of Candidate B (specifically the EntropyCache baseline comparison on our benchmarks) will be incorporated into Candidate A as a supporting experiment.

### Strengthened: Candidate A (ComposeAccel Composability Atlas) -- Selected as Front-Runner

**Additional controls added:**

1. **Composition-order control**: For each pair (X+Y), test both X-first-Y-second and Y-first-X-second orderings where the implementation allows. Report whether ordering matters.

2. **Iso-compute comparison**: For every combined method that achieves speedup S, also test the single best method tuned to achieve the SAME speedup S by adjusting its hyperparameters. This controls for the possibility that "composition" is just a proxy for "more aggressive single-method hyperparameters." If EntropyCache at a more aggressive threshold achieves the same speed-quality tradeoff as EntropyCache+JoT, the composition adds no value.

3. **Failure mode taxonomy**: Pre-specify the failure modes to look for: (a) quality collapse (accuracy drops >10% from best single method), (b) speed regression (combination is slower than best single method), (c) mode collapse (output diversity drops, measured by unique n-grams), (d) degenerate generation (repetitive text, empty outputs, NaN), (e) task-selective failure (quality preserved on some benchmarks but collapses on others).

4. **Long-generation benchmark added**: Include MBPP-Instruct (code, medium-length generation) alongside GSM8K, MMLU, HumanEval. This covers four distinct generation profiles: short classification (MMLU), medium reasoning (GSM8K), medium code (HumanEval), longer code (MBPP).

5. **Baseline strengthening**: Add SlowFast+dLLM-Cache as a composition baseline (the only published combination). If our cross-family compositions cannot beat this intra-family composition, the cross-family composability claim weakens.

**Tightened falsification criterion:**

The experiment has TWO falsifiable claims:
1. **Composability claim**: At least one cross-family pair achieves composability ratio >0.6 (i.e., >60% of the product of individual speedups) while retaining >95% of single-best-method quality. Falsified if all pairs fail this threshold.
2. **Failure mode claim**: At least one pair exhibits a non-trivial failure mode (quality drop >5% from best individual, or speed regression) that is NOT predicted by individual method performance alone. Falsified if all compositions either work perfectly or fail in obvious ways (e.g., implementation incompatibility).

### Strengthened: Candidate C (IGSD) -- Retained as Composability Vehicle

**Modifications:**
- Not pursued as standalone contribution. IGSD is included as the third method family (speculative) in Candidate A's composability atlas.
- The draft/verify semantics are simplified: use SSD's established acceptance criterion rather than inventing a new one. Our contribution is testing SSD-style self-speculation in composition with KV caching and early stopping, not the speculation mechanism itself.
- If SSD code is not available for pure MDMs, implement the simplest possible version: K/2 step draft with temperature reduction, K/2 step verify on remaining masked tokens. The method need not be novel -- it only needs to represent the "speculative" acceleration family in the composition study.

### Selected Front-Runner: Candidate A (ComposeAccel Composability Atlas)

**Rationale**: The composability gap (Gap 11 in the literature survey) is the strongest novel contribution with 9/10 novelty rating. The experimental design is clean and falsifiable. The failure mode atlas (identifying WHERE composition breaks) is potentially more valuable than the positive composability results, because it would guide future work in this crowded field. The main risk (hyperparameter interaction space) is manageable with structured grid search + pilot experiments.

---

## Phase 5: Final Proposal

- **Title**: ComposeAccel: A Systematic Composability Atlas for Training-Free Diffusion Language Model Acceleration

- **Hypothesis**: Training-free DLM acceleration methods from different acceleration families (KV caching, token-level early stopping, self-speculative decoding) can be composed for compounding speedups, but the interaction is non-trivial -- some pairs are synergistic (composability ratio >0.8), some are interfering (ratio <0.6), and some produce quality-degrading failure modes invisible when testing methods individually. At least one cross-family pair will achieve composability ratio >0.6 with >95% quality retention.

- **Falsification criterion**: The composability thesis is falsified if ALL cross-family pairs achieve composability ratio <0.6 at any quality retention level >90%. The failure-mode atlas is falsified if no pair exhibits a non-trivial failure mode (>5% quality drop or speed regression not predicted by individual performance).

- **Method**: Systematic pairwise and triple composition of three representative training-free methods:
  - **M1 (KV Caching)**: EntropyCache -- reduces per-step compute via approximate KV reuse
  - **M2 (Early Stopping)**: JoT -- reduces total step count via per-token early exit
  - **M3 (Self-Speculative)**: SSD-style self-speculation or simplified IGSD -- reduces effective steps via draft-verify
  - These three represent orthogonal acceleration dimensions: per-step cost (M1), step count (M2), and effective parallelism (M3)

- **Evaluation protocol**:
  - **Primary benchmarks**: MMLU (0-shot), GSM8K (8-shot CoT), HumanEval (0-shot), MBPP-Instruct (0-shot) -- all established public benchmarks used consistently in the DLM literature
  - **Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (generalization validation)
  - **Metrics**:
    - Accuracy on each benchmark (exact match for GSM8K, pass@1 for code, accuracy for classification)
    - Wall-clock throughput (tokens/second) on identical hardware (single GPU, fixed batch size=1)
    - FLOPs/token via PyTorch profiling
    - Composability ratio = actual combined speedup / product of individual speedups
    - Quality retention ratio = combined accuracy / best single-method accuracy
    - AUP (Accuracy Under Parallelism, from d3LLM) where applicable
  - **Statistical test plan**:
    - 5 random seeds per configuration
    - Bootstrap 95% CI for all metrics
    - Paired t-test for pairwise comparisons with Bonferroni correction (alpha = 0.05 / number of comparisons)
    - Effect size (Cohen's d) reported for all comparisons
  - **Hardware**: Single A100-80GB or equivalent; all experiments on identical hardware; report GPU model, CUDA version, PyTorch version

- **Ablation schedule**:

  | ID | Ablation | What it tests | Expected outcome |
  |----|----------|---------------|------------------|
  | A1 | M1 alone vs. baseline | EntropyCache individual speedup | 15-26x speedup, <2% quality loss (reproducing published) |
  | A2 | M2 alone vs. baseline | JoT individual speedup | 5-7x speedup, <2% quality loss (reproducing published) |
  | A3 | M3 alone vs. baseline | Self-spec individual speedup | 2-4x speedup, <3% quality loss |
  | A4 | M1+M2 vs. best(M1, M2) | KV cache + early stopping interaction | Composability ratio 0.5-0.9; potential threshold coupling |
  | A5 | M1+M3 vs. best(M1, M3) | KV cache + speculation interaction | Potentially synergistic (IGSD's frozen tokens improve cache hit rate) |
  | A6 | M2+M3 vs. best(M2, M3) | Early stopping + speculation interaction | Potentially interfering (both reduce effective steps, diminishing returns) |
  | A7 | M1+M2+M3 vs. best pair | Triple composition ceiling | Diminishing returns expected; ceiling test |
  | A8 | Iso-compute control | Is composition better than aggressive single method? | Critical control: if aggressive M1 alone matches M1+M2, composition is unnecessary |
  | A9 | Composition-order test | Does application order matter? | Implementation-specific; documents a confound |
  | A10 | SlowFast+dLLM-Cache baseline | Published intra-family composition | Calibration: our cross-family compositions should aim to match or exceed |
  | A11 | Threshold sensitivity sweep | Which hyperparameters dominate the interaction? | Entropy threshold x confidence threshold x stopping threshold 3D surface |
  | A12 | Generation length sweep | How does composition interact with output length? | Short (MMLU) vs. medium (GSM8K) vs. long (MBPP) |

- **Control experiments**:
  - **Null composition control**: Run M1 with randomized cache refresh decisions (random instead of entropy-guided) + M2 with random stopping. If this random composition achieves similar composability ratio as the real composition, the specific method designs are not contributing -- the speedup is just from aggressive approximation.
  - **Degradation detection**: For each combination, also measure output perplexity under an independent AR model (e.g., Qwen2.5-7B). Quality metrics like accuracy may miss subtle degradation modes (e.g., correct answer but garbled reasoning chain).
  - **Iso-compute single-method control**: For each composed configuration that achieves combined speedup S, tune the best individual method to match speedup S and compare quality. This is the most important control for the entire study.

- **Pilot design**: On LLaDA-8B-Instruct, 100 samples from GSM8K (8-shot CoT):
  1. Run vanilla baseline (full denoising, no acceleration) -- establish quality ceiling
  2. Run EntropyCache alone -- verify reproduction of published numbers
  3. Run JoT alone -- verify reproduction
  4. Run EntropyCache + JoT -- test basic composability
  5. Measure: accuracy, tokens/second, composability ratio, presence of failure modes (output collapse, degenerate text)
  - Expected pilot duration: <15 minutes per configuration, ~1 hour total for 5 configs
  - Go/no-go criterion: If EntropyCache+JoT crashes, produces NaN, or shows >20% quality degradation, the composition implementation needs debugging before proceeding. If composability ratio is >0.3, proceed to full experiments.

- **Resource estimate**:
  - Model: LLaDA-8B-Instruct (~16GB in fp16) -- fits on single A100-80GB
  - Pilot experiments: ~1 hour (5 configurations x 100 samples x ~1 min each)
  - Full single-method experiments (A1-A3): ~3 hours (3 methods x 4 benchmarks x full eval sets)
  - Full composition experiments (A4-A7): ~4 hours (4 combinations x 4 benchmarks)
  - Control experiments (A8-A10): ~3 hours
  - Sensitivity sweeps (A11-A12): ~4 hours
  - Total: ~15 GPU-hours for the primary experiments. Well within the 1-hour-per-task constraint if tasks are parallelized across GPUs or run sequentially with pilot-informed prioritization.
  - Dream-7B validation: +50% compute budget (~8 GPU-hours)

- **Risk assessment**:

  | Risk | Severity | Likelihood | Mitigation |
  |------|----------|------------|------------|
  | Implementation incompatibility between methods | High | Medium | Use EntropyCache's unified codebase as foundation; implement M2 and M3 as modules within the same framework |
  | Hyperparameter interaction space too large | Medium | High | Structured grid search with pilot-informed narrowing; report sensitivity analysis honestly |
  | All compositions trivially succeed (no failure modes) | Medium | Low | If so, the positive composability result is still novel; the failure mode atlas becomes "all compositions work" which is also valuable |
  | All compositions trivially fail | High | Low | Would falsify composability thesis; still publishable as a negative result paper if methodology is rigorous |
  | Reproduction failure on individual methods | High | Medium | Run reproduction checks first (pilot phase); use official codebases where available |
  | Speedup numbers not matching published claims | Medium | High | Report our numbers honestly; attribute differences to hardware, implementation, evaluation protocol |

- **Novelty claim**: The empirical contribution is answering an open experimental question that no prior work has addressed: **do training-free DLM acceleration methods from different families compose, and if so, how?** The composability atlas (systematic pairwise interaction matrix with composability ratios) and the failure mode atlas (systematic documentation of where and why composition breaks) are both first-of-their-kind experimental artifacts. This is not a new method paper -- it is a rigorous experimental investigation that the field needs to make informed decisions about how to combine the 15+ acceleration methods that have been published in the last year.
