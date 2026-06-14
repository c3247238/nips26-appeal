# Contrarian Perspective

## Phase 1: Literature Survey

### Assumptions Challenged

1. **Assumption: Training-free acceleration methods compose orthogonally with minimal quality loss**
   - Evidence challenging it:
     - ParallelBench (ICLR 2026, arXiv 2510.04767) demonstrates that parallel decoding under the conditional independence assumption causes *dramatic* quality degradation on tasks with strong token dependencies -- degradation that standard benchmarks (MMLU, GSM8K) fail to detect.
     - "How Efficient Are Diffusion Language Models?" (arXiv 2510.18480) shows that acceleration strategies like dual cache and parallel decoding mainly offer gains at small batch sizes, with benefits diminishing upon scaling. The paper explicitly notes that "the wide divergence in experimental configurations across studies often renders their efficiency claims incomparable."
     - Elastic-Cache (arXiv 2510.14973) documents that KV drift accumulates *multiplicatively* across layers, meaning stale caches introduce compounding errors with depth.
     - ICLR 2026 KV Cache Transform Coding notes that "existing KV cache compression techniques tend to be brittle, and accuracy degradation prohibits combining methods for compounded benefits."

2. **Assumption: Speedup numbers reported by individual papers are reliable and comparable**
   - Evidence challenging it:
     - "How Efficient Are Diffusion Language Models?" identifies three major evaluation issues: (1) evaluations limited to single-instance inference or fixed generation lengths, (2) hybrid or proprietary implementations "blurring the lines between algorithmic innovation and engineering-level optimizations," (3) lack of standardized evaluation protocols making efficiency claims incomparable.
     - "Generative Frontiers" (arXiv 2604.02718) shows that generative perplexity and unigram entropy correspond to KL divergence components, meaning "minor entropy differences can produce large perplexity shifts and cause single-point comparisons to reflect inference settings rather than model capability."
     - D2F (arXiv 2508.09192) explicitly notes that "none of the existing open-source dLLMs have achieved superior inference speed over AR LLMs of similar size" -- suggesting earlier speedup claims were relative to vanilla diffusion baselines, not AR models.

3. **Assumption: KV caching is the key bottleneck and solving it unlocks DLM potential**
   - Evidence challenging it:
     - "Theoretical Benefit and Limitation of Diffusion Language Model" (NeurIPS 2025, arXiv 2502.09622) proves that MDMs *inherently* require sampling steps scaling with sequence length when evaluated by sequence error rate (accuracy), not just perplexity. KV caching reduces per-step cost but does not address the fundamental step-count requirement.
     - "Lost in Diffusion" (arXiv 2604.10556) documents that dLLMs exhibit higher hallucination propensity than AR counterparts even without acceleration, with Dream-7B failing to recognize non-existent entities in 98.50% of cases. Acceleration on a fundamentally hallucination-prone model amplifies, not mitigates, these issues.
     - The practical impact is bounded: even the aggressive Window-Diffusion (99x claimed speedup) admits it "may sacrifice global context."

4. **Assumption: DLM acceleration methods designed for pure MDMs (LLaDA, Dream) will remain relevant**
   - Evidence challenging it:
     - I-DLM (arXiv 2604.11035, April 2026) is the first DLM to match AR quality by using a hybrid architecture with introspective strided decoding that has *native KV cache support*. This eliminates the primary motivation for most training-free KV approximation methods.
     - ReFusion (ICLR 2026) achieves full KV cache reuse via slot-level design, making approximate caching schemes redundant.
     - LLaDA 2.0/2.1 and SDAR all moved to block-diffusion or hybrid AR-diffusion architectures, fundamentally changing the computational profile.
     - The field is converging on hybrid architectures that make pure-MDM acceleration a dead end.

### Landscape of Doubt

The DLM acceleration field is in a paradoxical state. There are 40+ papers proposing training-free acceleration methods, most claiming 5-100x speedups, yet:

(a) **No open-source DLM actually outperforms an equivalently-sized AR model in end-to-end throughput on standard hardware.** The papers that claim "faster than AR" either use training-based methods (d3LLM, D2F, I-DLM) or are closed-source (Mercury, Gemini Diffusion, Seed Diffusion).

(b) **The speedup numbers are not directly comparable.** Different papers use different baselines (vanilla LLaDA vs. Fast-dLLM vs. AR models), different hardware (A100 vs. H100 vs. H200), different batch sizes (1 vs. 32), and different generation lengths. A "27.6x speedup" over vanilla LLaDA is not the same as a 27.6x speedup over vLLM-served Qwen2.5-7B.

(c) **Quality preservation claims are tested on benchmarks that systematically underestimate degradation.** ParallelBench (ICLR 2026) demonstrates that tasks trivial for AR models and humans become catastrophically difficult for dLLMs under parallel decoding -- and these tasks are *not* covered by standard evaluations (MMLU, GSM8K, MBPP).

(d) **The architecture is moving out from under the methods.** The most exciting DLMs (I-DLM, ReFusion, LLaDA 2.0) are no longer pure MDMs. They are hybrid AR-diffusion models with native KV cache support. Acceleration methods designed for pure MDMs may be solving yesterday's problem.

(e) **Composition is assumed to be additive but is more likely subadditive or even destructive.** Each acceleration method introduces approximation errors (stale KV caches, premature token commitments, skipped layers). When composed, these errors interact in non-obvious ways. The one published composition (SlowFast + dLLM-Cache) is within the same method family -- not a cross-family test.

---

## Phase 2: Initial Candidates

### Candidate A: "The Composition Mirage: Why Training-Free DLM Acceleration Methods Destructively Interfere"

- **Challenged assumption**: That training-free acceleration methods (KV caching, step reduction, parallel decoding, speculative decoding) compose orthogonally, yielding compounded speedups with preserved quality.
- **Evidence against**:
  - KV drift accumulates multiplicatively across layers (Elastic-Cache). Adding step reduction means fewer opportunities to refresh stale caches, compounding drift errors.
  - ParallelBench shows that parallel decoding already degrades quality on dependency-heavy tasks. KV caching further degrades the representation quality available for confidence-based decisions.
  - The "Theoretical Benefit and Limitation" paper proves MDMs need O(L) steps for sequence-level accuracy. Combining step reduction (fewer steps) with KV approximation (worse per-step quality) pushes in exactly the wrong direction -- fewer, worse steps.
  - No paper has published a rigorous pairwise composability matrix. The field assumes composability without testing it.
- **Contrarian hypothesis**: Cross-family composition of training-free acceleration methods on pure MDMs yields *subadditive* or *destructive* speedup-quality tradeoffs. Specifically: (1) KV caching + step reduction has a destructive interaction because stale caches need more steps to self-correct, not fewer; (2) parallel decoding + any approximation method has a destructive interaction because confidence signals used for adaptive parallelism are themselves degraded by the approximation.
- **Exploitation plan**: Build the first rigorous composability atlas with controlled experiments. The atlas itself (showing where composition fails) is the primary contribution. The failure modes and interference mechanisms are the novelty, not any new acceleration method.
- **Novelty estimate**: 8/10 -- No one has done this. The closest is JoT's *unvalidated claim* of orthogonality to KV caching.

### Candidate B: "Rethinking DLM Acceleration: The Architecture Is the Bottleneck, Not the Algorithm"

- **Challenged assumption**: That the right approach to DLM acceleration is developing clever inference-time algorithms (caching, scheduling, speculation) on top of existing pure MDM architectures.
- **Evidence against**:
  - I-DLM matches AR quality and achieves 5,900 tok/s with *only architectural changes* (introspective strided decoding) -- far exceeding any training-free method applied to vanilla LLaDA/Dream.
  - ReFusion achieves 18x speedup over MDMs and 2.33x over ARMs through architectural redesign (slot-level plan-and-infill), not inference tricks.
  - d3LLM and D2F achieve 5-10x speedups over AR through distillation/training, dwarfing training-free methods.
  - The theoretical limitation paper proves that MDMs *inherently* require O(L) steps for accuracy -- meaning no training-free inference trick can overcome this fundamental bottleneck.
- **Contrarian hypothesis**: Training-free acceleration methods for pure MDMs are collectively a dead end. The speed-quality Pareto frontier is dominated by architectural innovations (hybrid AR-diffusion, block diffusion, introspective consistency) that provide fundamentally better tradeoffs. Resources spent on composing training-free methods on pure MDMs would be better spent on acceleration for the next-generation hybrid architectures.
- **Exploitation plan**: Demonstrate that the best training-free composition on LLaDA-8B still cannot match the speed-quality Pareto of I-DLM or ReFusion. Then show that transferring training-free techniques to hybrid architectures is non-trivial (the methods exploit properties specific to pure MDMs that hybrid architectures don't have).
- **Novelty estimate**: 6/10 -- The architecture-vs-algorithm argument is implicit in the literature but nobody has made it explicit with head-to-head experiments.

### Candidate C: "The Evaluation Illusion: DLM Acceleration Claims Cannot Be Trusted"

- **Challenged assumption**: That reported speedup numbers for DLM acceleration methods are meaningful and that "negligible quality loss" claims are accurate.
- **Evidence against**:
  - "How Efficient Are Diffusion Language Models?" shows evaluation is done at batch=1, single-instance, fixed-length -- not production-relevant conditions.
  - ParallelBench reveals that standard benchmarks (MMLU, GSM8K) systematically fail to detect quality degradation from parallel decoding. Tasks trivial for humans become catastrophic for accelerated dLLMs.
  - "Lost in Diffusion" shows dLLMs hallucinate more than AR models *before* acceleration. Acceleration papers evaluate on benchmarks that don't test hallucination faithfulness.
  - "Generative Frontiers" demonstrates that single-point perplexity comparisons "reflect inference settings rather than model capability."
  - Speedup baselines vary wildly: "27.6x" over vanilla LLaDA is ~2x over Fast-dLLM, which is still slower than vLLM-served AR models.
- **Contrarian hypothesis**: If DLM acceleration methods were evaluated under a fair, production-relevant protocol (batched inference, dependency-heavy tasks, hallucination benchmarks, comparison against optimized AR baselines), most claimed speedups would disappear or be accompanied by unacceptable quality degradation.
- **Exploitation plan**: Design and execute a "DLM Acceleration Reality Check" -- a rigorous evaluation protocol that tests top methods under (1) batched inference at various batch sizes, (2) ParallelBench + hallucination benchmarks, (3) head-to-head with vLLM-served AR baselines on the same hardware. Publish the (likely sobering) results.
- **Novelty estimate**: 7/10 -- "How Efficient Are Diffusion Language Models?" starts this but doesn't finish it. A full reality check with 2026 methods would be new.

---

## Phase 3: Self-Critique

### Against Candidate A (Composition Mirage)

- **Steelman**: The mainstream view that acceleration methods can compose is plausible because they operate at different levels of the compute stack:
  - KV caching reduces per-step FLOPS (computation level)
  - Step reduction reduces total steps (schedule level)
  - Parallel decoding reduces per-step latency (parallelism level)
  - Speculative decoding reduces verification cost (algorithmic level)
  
  Methods at different levels *should* be composable because they don't directly compete for the same resource. The SlowFast + dLLM-Cache combination *does* work, suggesting at least within-family composition is viable. JoT's design *was* intentionally built to be orthogonal to KV caching. The fact that nobody has tested cross-family composition doesn't mean it fails -- it just means nobody has tried.

- **Cherry-picking check**: Am I selectively citing the theoretical O(L) step requirement while ignoring that Prophet shows 97% of GSM8K instances need only half the steps? Yes -- the theoretical worst case may not be the practical common case. The Elastic-Cache "multiplicative drift" finding is for *naive* caching; adaptive methods like EntropyCache specifically address this. I need to be careful not to use worst-case theory to condemn best-practice engineering.

- **Confounding check**: The lack of published composition studies could be explained by: (1) it's hard engineering work and nobody has bothered yet, not because it fails, (2) the field is young and papers are incentivized to show single-method results first. The absence of evidence is not evidence of absence.

- **Actionability check**: STRONG. Even if composition turns out to work well, the atlas of *when* and *why* it works/fails is a valuable contribution. If it fails, that's a strong negative result. If it succeeds, the atlas is a practical guide. Either way, the experimental methodology is novel and publishable.

- **Verdict**: **STRONG** -- The contrarian position survives. Even the steelman admits nobody has tested this. The worst case (composition works fine) still produces a valuable atlas. The best case (composition fails in interesting ways) is a high-impact negative result.

### Against Candidate B (Architecture Is the Bottleneck)

- **Steelman**: Training-free methods have a massive practical advantage: they work on *existing* deployed models without retraining. I-DLM requires 4.5B training tokens on 8xH100. ReFusion requires 1.22B tokens from Qwen3-8B. Most practitioners don't have these resources. Training-free methods serve a fundamentally different use case (post-hoc acceleration of already-deployed checkpoints) that architectural changes cannot address. Furthermore, training-free insights (which tokens are stable, which steps are redundant) often *inform* the next architectural design. The methods are complementary, not competing.

- **Cherry-picking check**: I'm comparing the *best* training-based results (I-DLM) against *average* training-free results. But Fast-dLLM achieves 27.6x speedup training-free, which is competitive. Window-Diffusion claims 99x. If these hold up under fair evaluation, the architecture argument weakens considerably.

- **Confounding check**: The comparison is unfair because I-DLM and ReFusion start from a *better base model* (Qwen3-8B, which is already superior to LLaDA-8B in raw quality). The speedup comparison conflates architecture improvements with base model improvements.

- **Actionability check**: WEAK-MODERATE. Telling the community "training-free is a dead end" without providing a better alternative for existing deployments is not constructive. And if the conclusion is "use I-DLM instead," that's just a literature pointer, not a research contribution.

- **Verdict**: **WEAK** -- The contrarian position is partially correct (architecture matters more than algorithms) but the practical argument for training-free methods is strong, and the comparison is confounded by base model quality differences. This angle works better as a framing device within a larger study than as a standalone research contribution.

### Against Candidate C (Evaluation Illusion)

- **Steelman**: The field is already self-correcting. ParallelBench, "How Efficient Are Diffusion Language Models?", "Generative Frontiers", and "Lost in Diffusion" are all recent papers that address evaluation problems. The community is aware of these issues and actively fixing them. Furthermore, engineering-level speedups *do* matter even if they're not comparable across papers -- each paper's within-paper ablations are typically fair, even if cross-paper comparisons are not.

- **Cherry-picking check**: Am I emphasizing the worst evaluation practices while ignoring papers that do rigorous evaluation? Fast-dLLM (ICLR 2026) and CDLM (MLSys 2026) were accepted at top venues with presumably thorough review. EntropyCache compares against multiple baselines under controlled conditions.

- **Confounding check**: The evaluation problems are real but not unique to DLM acceleration -- they're endemic to ML systems papers. This is not a DLM-specific insight.

- **Actionability check**: MODERATE. A comprehensive "reality check" paper would be useful but risks being a purely negative contribution. The community already knows evaluation is hard. The value would come from the specific failure modes discovered, not the general critique.

- **Verdict**: **MODERATE** -- Valid concerns but the field is already self-correcting, and the contribution risks being incremental over existing critique papers. Better used as methodology foundation for Candidate A rather than standalone.

---

## Phase 4: Refinement

### Dropped

**Candidate B** (Architecture Is the Bottleneck): Dropped because the steelman argument is too strong. Training-free methods serve a real practical need, and the comparison with I-DLM/ReFusion is confounded by base model quality. The architecture insight is correct but not actionable as a standalone research contribution.

### Strengthened: Candidate A + elements of Candidate C

The surviving front-runner merges Candidate A's core thesis (composability is untested and likely destructive) with Candidate C's methodological rigor (evaluation must include dependency-heavy tasks, batched settings, and AR baselines).

**Refined thesis**: The DLM acceleration field's implicit assumption that training-free methods compose orthogonally is not just untested -- it is likely wrong due to specific, identifiable interference mechanisms between error sources. The composability atlas should be evaluated under a rigorous protocol that includes ParallelBench-style dependency tasks, not just standard benchmarks, to avoid the evaluation illusion that masks composition-induced degradation.

**Specific interference mechanisms to test**:

1. **KV-staleness x step-reduction interference**: KV caches become stale between refresh points. Reducing the total number of denoising steps means each step represents a larger state transition, making staleness worse per remaining step. The self-correction mechanism (subsequent fresh steps fixing stale-cache errors) has fewer opportunities to operate. Prediction: KV caching + step reduction yields *worse* quality than either alone at the same speedup point.

2. **Confidence-signal degradation under approximation**: Methods like Fast-dLLM, JoT, and Prophet rely on confidence signals (logit entropy, prediction stability) to make adaptive decisions. KV caching degrades the representations from which these signals are computed. Prediction: confidence-based adaptive methods become *less* adaptive (more conservative or more error-prone) when combined with KV approximation.

3. **Error amplification in parallel decoding under sparse attention**: Parallel decoding already violates token dependencies. Adding sparse attention or token pruning (DyLLM, ES-dLLM, Window-Diffusion) further removes the contextual information needed to maintain coherence across parallel-decoded tokens. Prediction: the quality cliff in ParallelBench-style tasks becomes steeper when sparse attention is combined with aggressive parallel decoding.

**Additional corroboration found**:
- The ICLR 2026 KV Cache Transform Coding paper explicitly states: "existing KV cache compression techniques tend to be brittle, and accuracy degradation prohibits combining methods for compounded benefits." This is stated for AR models -- the problem is likely *worse* for DLMs where caches are already approximate.
- Prophet's finding that "extended deliberation can be counterproductive, dropping accuracy from 86.2% to 71.9%" demonstrates that *more* computation (more justification tokens) can actually *hurt* DLMs -- contradicting the naive assumption that errors self-correct given enough steps.

### Selected Front-Runner: Candidate A (refined)

**"The Composition Mirage: When Training-Free DLM Acceleration Methods Destructively Interfere"**

---

## Phase 5: Final Proposal

### Title
Rethinking Composability in DLM Acceleration: A Systematic Study of Cross-Method Interference

### Challenged Assumption
The DLM acceleration community implicitly assumes that training-free acceleration methods -- KV caching, step reduction/early stopping, and parallel decoding -- compose orthogonally, yielding compounded speedups with preserved quality. This assumption drives the research program of "ComposeAccel"-style work and motivates the development of each method in isolation. Yet no published work has rigorously tested this assumption. We challenge it with specific, testable predictions about *how* and *why* composition fails.

### Evidence

**For the mainstream view (composability should work)**:
- Methods operate at different computational levels (per-step cost vs. step count vs. parallelism), suggesting orthogonal acceleration dimensions.
- SlowFast + dLLM-Cache demonstrates successful within-family composition.
- JoT was explicitly designed to be orthogonal to KV caching.
- Individually, each method preserves >95% quality on standard benchmarks.

**Against the mainstream view (composability likely fails)**:
- KV drift accumulates multiplicatively across layers; adaptive caching addresses this but introduces new approximation errors (Elastic-Cache, arXiv 2510.14973).
- Parallel decoding quality degrades *catastrophically* on dependency-heavy tasks (ParallelBench, ICLR 2026).
- MDMs require O(L) steps for sequence-level accuracy (NeurIPS 2025 theory paper); combining step reduction with per-step approximation pushes the double-wrong direction.
- "How Efficient Are Diffusion Language Models?" (arXiv 2510.18480) shows acceleration benefits diminish at larger batch sizes, suggesting method gains are not additive.
- KV cache compression methods are "brittle, and accuracy degradation prohibits combining methods for compounded benefits" (ICLR 2026 KVTC).
- dLLMs already hallucinate more than AR models before acceleration (arXiv 2604.10556); composing multiple approximation sources amplifies this.
- Extended computation can *reduce* accuracy in dLLMs (Prophet, arXiv 2508.19982: accuracy drops from 86.2% to 71.9% with more deliberation tokens).

### Hypothesis
Cross-family composition of training-free DLM acceleration methods exhibits three specific interference mechanisms:
1. **KV-staleness amplification under step reduction**: Fewer total steps means larger per-step state transitions, making cached KVs staler per remaining step. The self-correction budget (fresh-compute steps that can fix stale errors) shrinks proportionally.
2. **Confidence signal corruption under approximation**: Adaptive methods (confidence-based parallel decoding, entropy-guided caching, token-level early stopping) rely on signals computed from model representations. When these representations are themselves approximate (due to KV caching or sparse attention), the signals degrade -- leading to either over-conservative decisions (negating speedup) or over-aggressive decisions (amplifying quality loss).
3. **Dependency violation compounding under multiple approximations**: Each approximation method independently violates different aspects of the full-attention, full-step computation. When composed, the violations compound because each method assumes the other's outputs are "correct enough" to base decisions on.

### Method
1. **Baseline establishment**: Implement 4 representative training-free methods on LLaDA-8B-Instruct:
   - M1: KV caching (EntropyCache -- entropy-guided refresh)
   - M2: Step reduction (JoT -- per-token early stopping)
   - M3: Parallel decoding (Fast-dLLM confidence-aware parallel decode)
   - M4: Speculative self-decoding (IGSD -- coarse-step draft + fine-step verify)
   
2. **Solo evaluation**: Each method alone across standard + extended benchmarks.

3. **Pairwise composition**: All 6 pairs (M1+M2, M1+M3, M1+M4, M2+M3, M2+M4, M3+M4), measuring speedup and quality.

4. **Interference diagnosis**: For each pair showing subadditive or destructive behavior:
   - Measure KV staleness (cosine distance between cached and fresh KVs per layer)
   - Measure confidence signal accuracy (correlation between confidence scores under approximation vs. ground truth)
   - Measure token dependency violation rate on ParallelBench tasks

5. **Extended evaluation protocol** (addressing Candidate C's concerns):
   - Standard benchmarks: MMLU, GSM8K, HumanEval, MBPP
   - Dependency-heavy: ParallelBench (17 tasks)
   - Hallucination: PreciseWikiQA, NonExistentRefusal (from "Lost in Diffusion")
   - Batched inference: batch sizes 1, 4, 8, 16
   - Hardware: single A100-80GB or H100 (controlled)

### Experimental Plan
- **Models**: LLaDA-8B-Instruct (primary), Dream-7B-Instruct (validation)
- **Codebase**: Build on Fast-dLLM (NVIDIA) + EntropyCache + JoT
- **Hardware**: Single GPU (A100/H100) to eliminate multi-GPU confounders
- **Per-experiment budget**: ~30-45 minutes per configuration (4 solo + 6 pairs + baselines = ~16 configurations, total ~12 hours)
- **Key metrics**: 
  - Wall-clock speedup (tok/s) relative to both vanilla DLM and vLLM-served Qwen2.5-7B
  - Quality on each benchmark (accuracy, pass@1, hallucination rate)
  - Composition efficiency ratio = (Quality_AB / Quality_A * Quality_B) -- values <1 indicate destructive interference
  - AUP (Accuracy Under Parallelism) from d3LLM

### Baselines
- Vanilla LLaDA-8B-Instruct (no acceleration, full steps, full attention)
- vLLM-served Qwen2.5-7B-Instruct (AR reference for fair speed-quality comparison)
- Each method solo (to compute composition efficiency ratios)
- SlowFast + dLLM-Cache (only published composition baseline)

No strawman baselines. Every baseline is either the original paper's implementation or a well-maintained community implementation.

### Risk Assessment
**What if the mainstream view turns out to be correct?**

If all pairwise compositions show additive or super-additive speedup with preserved quality, this is still a valuable publication:
- The composability atlas is a practical resource for practitioners.
- The extended evaluation protocol (ParallelBench, hallucination, batched) provides data points nobody else has.
- The interference diagnosis methodology (KV staleness measurement, confidence signal accuracy) becomes a diagnostic tool for future methods.

The paper pivots from "Rethinking Composability" to "A Practitioner's Guide to Composing DLM Acceleration Methods" -- publishable either way.

**What if all compositions are destructive?**

This would be a strong negative result demonstrating that the implicit composability assumption is wrong. This is arguably more impactful and more publishable, as it redirects research effort toward architectural solutions (I-DLM, ReFusion) that avoid the composition problem entirely.

### Novelty Claim
The specific insight is that training-free DLM acceleration methods introduce *correlated* approximation errors (because they all ultimately depend on the same model representations), and composition forces the model to operate under multiple simultaneous approximations whose interactions are non-trivially harmful. This is distinct from the AR setting where methods like prefix caching, quantization, and speculative decoding compose more cleanly because the AR causal structure provides a natural isolation boundary between past (exact) and future (approximate) computation. DLMs lack this boundary -- everything is bidirectional and everything is approximate -- making composition fundamentally harder.
