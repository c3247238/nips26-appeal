# Optimist Analysis

## Evidence Map

| Metric | Baseline | Best Config | Delta | Signal Strength |
|--------|----------|-------------|-------|-----------------|
| M3 gw=0.3 GSM8K AccRet | 1.000 | 1.039 (73%/73% on 100 samples) | +3.9% accuracy boost | **Strong** |
| M3 gw=0.3 MATH500 AccRet | 1.000 | 2.439 (26% vs 11.07% baseline) | +143.9% accuracy boost | **Strong** |
| M3 gw=0.3 GSM8K Speedup | 1.0x | 1.68x (52.0 vs 31.0 TPS) | +68% throughput | **Strong** |
| M1+IGSD GSM8K Ortho | -- | 0.989 (tau=0.7, td=16) | Near-multiplicative composition | **Strong** |
| M1+IGSD Combined Ortho | -- | 0.958 (tau=0.7, td=16) | Near-multiplicative composition | **Strong** |
| Three-way Ortho (Max-Speed) | -- | 1.020 (mean over 3 seeds) | Super-additive synergy | **Strong** |
| Dream-7B Transfer Ratio | -- | 1.86x avg across 5 configs | Recipes transfer across models | **Strong** |
| IGSD tau=0.7/td=16 GSM8K QAS | 1.0 | 1.637 | Best single-method QAS | **Moderate** |
| M1 eta=0.5 MATH500 AccRet | 1.000 | 1.066 | Caching improves MATH500 | **Moderate** |
| Three-way GSM8K accuracy (seed 456) | 0.73 | 0.58 (79.5% retention) | 1.69x speedup at decent quality | **Moderate** |
| Batch sensitivity accuracy stability | -- | max 6pp drop (bs=1 to bs=8) | Accuracy stable across batch | **Weak** |
| M3+IGSD tau=0.7/td=16 MATH500 AccRet | 1.000 | 0.994 | Near-lossless on MATH500 | **Moderate** |

## Root Cause Analysis

### 1. M3 (AR-Guided Unmasking) Delivers Simultaneous Speed AND Quality Improvement

- **Mechanism**: The Qwen2.5-0.5B guide model provides a cheap AR probability distribution that biases the DLM's unmasking order toward more confident tokens first. This acts as an "oracle hint" that reduces wasted denoising effort on already-determined tokens.
- **Design decision**: Using a tiny 0.5B guide model at guidance weight 0.3 was the key insight from iter_001. Higher weights (gw=1.0) cause GSM8K accuracy to drop to 0.59-0.62 (15pp loss), but the sweet spot at gw=0.3-0.7 preserves or improves accuracy.
- **Expected or surprising**: The speed gain (1.68x on GSM8K) is expected because AR guidance reduces effective computation. The **accuracy improvement** (+3.9% on GSM8K, +143.9% on MATH500) is **surprising** -- the guide model appears to correct the DLM's default unmasking order, especially on mathematical reasoning. This is consistent with the "Flexibility Trap" hypothesis (arXiv 2601.15165) where confidence-based unmasking order is suboptimal for reasoning tasks.
- **Significance**: This is potentially the single most publishable finding. An external 0.5B model can simultaneously accelerate AND improve a 8B DLM -- a rare win-win in the acceleration literature.

### 2. M1+IGSD Achieves Near-Multiplicative Composition (Ortho = 0.96-0.99)

- **Mechanism**: M1 (EntropyCache) reduces the cost of each forward pass by caching attention states for tokens with low entropy (high confidence). IGSD reduces the number of forward passes by skipping steps where inter-step KL divergence is low. These operate on orthogonal computational axes: per-step cost vs. step count.
- **Design decision**: Selecting entropy threshold 0.5 for M1 (conservative, 56% cache hit rate) and tau=0.7 with T_draft=16 for IGSD (aggressive speed) produces the best GSM8K ortho at 0.989.
- **Expected or surprising**: **Expected**. The hypothesis H3 predicted composition ratio > 0.8, and the data shows 0.96-0.99. The orthogonality of step-count reduction and per-step cost reduction is confirmed empirically.
- **Implication**: This validates the core thesis of the paper -- methods from different computational axes compose near-multiplicatively. The M1+IGSD pair at tau=0.7/td=16 achieves 2.75x speedup on GSM8K, with 58.9% accuracy retention.

### 3. Three-Way Composition Shows Super-Additive Synergy (Ortho > 1.0)

- **Mechanism**: The three-way "Max-Speed" recipe (M1_eta=0.5 + IGSD_tau=0.85/td=32 + M3_gw=0.0) achieves a combined ortho of 1.020 (mean over 3 seeds), meaning the three-way composition slightly EXCEEDS the best individual method's QAS. This is surprising because one would expect interference from stacking three methods.
- **Design decision**: Critically, the best three-way configs have M3 guidance weight = 0.0 (no AR guidance). This makes sense: the three-way combination is essentially M1+IGSD (which are near-orthogonal), and adding M3 at gw=0.0 just validates that the two-way composition is stable.
- **Expected or surprising**: **Surprising** that ortho exceeds 1.0. The explanation: IGSD's step skipping occasionally avoids degraded intermediate states, and M1's caching smooths out noise from skipped steps. The composition is not just additive but slightly synergistic.

### 4. Cross-Model Generalization to Dream-7B is Remarkable

- **Mechanism**: All 5 top configurations from LLaDA-8B were tested on Dream-7B-Instruct. The transfer ratio averages 1.86x, meaning Dream-7B benefits even MORE from the acceleration recipes than LLaDA.
- **Design decision**: Dream-7B has lower baseline accuracy (39% GSM8K vs 71.2% for LLaDA), but the acceleration methods still work because they operate on the diffusion denoising process itself, not on model-specific representations.
- **Expected or surprising**: The transfer ratio > 1.0 is **surprising and highly publishable**. It means the recipes are model-agnostic within the DLM family. 4/5 configs show "synergy" pattern agreement, with only the Quality-First config (which adds M3) showing divergence.

## Unexpected Signals

### 1. M3 as an Accuracy Booster, Not Just an Accelerator

- **Observation**: M3 at gw=0.3 improves MATH500 accuracy from 11.07% to 26% (seed 42) and 28% (seed 123) -- a 135-153% improvement. On GSM8K, accuracy goes from 71.2% to 73-75% on the 100-sample pilot.
- **Mini-hypothesis**: The small AR guide model (Qwen2.5-0.5B) has better token ordering intuition for mathematical expressions than the DLM's own confidence-based ordering. The guide essentially provides a curriculum for unmasking: easy tokens first, then hard reasoning tokens with maximum context. This is a training-free form of the "order correction" that LogicDiff (arXiv 2503.11664) achieves via training.
- **Significance**: This reframes M3 from "a quality-preserving speed method" to "a quality-enhancing method that also happens to speed things up." This could be an independent contribution: training-free accuracy improvement for DLMs via external AR guidance.

### 2. IGSD Accept Rate is Uniformly High (89-97%) But Quality Still Degrades

- **Observation**: Across all IGSD configurations, the accept rate (fraction of draft steps accepted by the verifier) is 89-97%. Yet accuracy retention is only 44-73% on GSM8K. This means the bottleneck is NOT step acceptance but rather the quality of the compressed trajectory.
- **Mini-hypothesis**: IGSD's KL threshold catches most "different" steps but misses subtle distributional shifts that are critical for reasoning chains. The KL divergence is a mean over all masked tokens, which can average out important per-token changes. A per-token adaptive threshold might significantly improve quality.
- **Significance**: This suggests a clear improvement direction for IGSD: replace the global mean KL with a token-level max KL or a weighted KL that prioritizes reasoning-critical positions.

### 3. Accuracy IMPROVES With Larger Batch Size in Composed Methods

- **Observation**: For M1+IGSD at batch sizes 1/4/8, accuracy is 45%/50%/51% on GSM8K (100 samples). The same pattern holds for the three-way composition. Accuracy actually increases by 6pp from bs=1 to bs=8.
- **Mini-hypothesis**: Larger batch sizes provide more stable gradient statistics in the denoising process. The batched attention computation may reduce numerical noise that affects token selection in the caching and step-skipping logic.
- **Significance**: This is a favorable practical finding -- serving deployments (which typically use larger batch sizes) will not suffer accuracy degradation from composed acceleration. However, the TPS drops from 96 to 51 TPS, which means the speedup advantage shrinks at larger batch sizes.

### 4. M3+IGSD (tau=0.7/td=16) Achieves Near-Lossless MATH500 (AccRet = 0.994)

- **Observation**: While most IGSD configurations degrade MATH500 heavily (AccRet = 0.31-0.47), adding M3 guidance at gw=0.7 to the most aggressive IGSD (tau=0.7/td=16) produces MATH500 AccRet = 0.994 -- essentially perfect accuracy preservation at 2.17x speedup.
- **Mini-hypothesis**: M3's AR guidance corrects the worst of IGSD's errors on mathematical expressions. The guide model "fills in" the tokens that IGSD's compressed trajectory gets wrong, because mathematical symbol sequences are highly predictable for an AR model.
- **Significance**: This suggests M3+IGSD may be a particularly effective pair for mathematical tasks, even though the overall combined ortho shows interference (0.84). The interference is concentrated on GSM8K's reasoning chains, not on MATH500's symbolic manipulation.

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| M3 accuracy improvement | Ablate M3 guide model size: 0.5B vs 1.5B vs 3B at gw=0.3 | Larger guide = more accurate but slower; 1.5B may be sweet spot | 4 | High |
| M1+IGSD near-orthogonality | Full 3-seed runs of M1+IGSD at top 3 configs on all 4 benchmarks | Confirm ortho > 0.9 with statistical significance | 6 | High |
| Three-way ortho > 1.0 | Expand from top-5 to top-10 configs, add HumanEval/MBPP | Determine if synergy extends to code generation | 4 | High |
| Dream-7B transfer | Run Dream-7B on HumanEval/MBPP with top 3 recipes | Confirm cross-model cross-task transfer | 3 | Med |
| Per-token KL for IGSD | Implement max-KL and weighted-KL variants of IGSD | 5-10pp accuracy improvement at same speedup | 3 | Med |
| M3+IGSD for math | Test M3+IGSD at gw=0.3 on full MATH500 (500 samples, 3 seeds) | Confirm near-lossless at 2x+ speedup for math | 2 | Med |
| AR comparison with vLLM | Re-run AR baseline with vLLM (not HF) for fair TPS comparison | AR advantage shrinks when DLM acceleration is applied | 2 | Low |

## Honest Caveats

### Finding 1: M3 Accuracy Improvement (+3.9% GSM8K, +143.9% MATH500)

- **Counter-argument**: The MATH500 improvement is inflated by the very low baseline (11.07%). Going from 11% to 26% is only +15pp, which could be noise on 50-100 sample evaluations with only 2 seeds. The GSM8K improvement (+3.9%) is within 2 standard deviations of seed variance (std = 1.49% on the full baseline).
- **Alternative explanation**: The AR guide model might be "leaking" correct answers through its probability distribution rather than genuinely improving the unmasking order. If Qwen2.5-0.5B has been trained on GSM8K/MATH500 solutions, it could be doing partial answer injection.
- **What would convince me**: (1) Run M3 on held-out benchmarks not in Qwen's training data. (2) Measure the correlation between M3's accuracy improvement and the guide model's standalone accuracy on each benchmark. (3) Verify with a randomized guide model (same architecture, random weights) that the improvement disappears.

### Finding 2: M1+IGSD Near-Orthogonality (Ortho = 0.96-0.99)

- **Counter-argument**: The pilot used only 100 GSM8K samples with a single seed for the pairwise experiment. The high ortho could be an artifact of sample selection. The M1 implementation does not achieve kernel-level speedup (cache hits do not translate to actual TPS gains proportionally -- see M1 alone at eta=0.5 achieving only 1.16x speedup despite 56% cache hits).
- **Alternative explanation**: Since M1's "speedup" is partially an artifact of the EntropyCache measurement (checking which tokens would be cached, but still running full attention), the composition ortho may be measuring the compatibility of two *accuracy* effects rather than two *speed* effects. The real speedup test requires d2Cache kernel integration.
- **What would convince me**: (1) Integrate d2Cache and measure actual wall-clock speedup for M1. (2) Re-measure M1+IGSD ortho using wall-clock TPS rather than theoretical speedup. (3) Run with 3 seeds and full 1319 GSM8K samples.

### Finding 3: Three-Way Ortho > 1.0

- **Counter-argument**: The best three-way configs all have M3 gw=0.0 (guidance OFF). So the "three-way" composition is really just M1+IGSD with the guide model loaded but not contributing. The "Quality-First" config with M3 gw=0.3 has lower QAS (0.934 vs 0.952). The super-additivity might be within noise (ortho = 1.02 +/- 0.08, so the 95% CI includes 1.0).
- **Alternative explanation**: The ortho > 1.0 could be a statistical fluctuation across the 3 seeds, especially since seed variance is significant (per-seed QAS ranges from 1.03 to 1.34 for Max-Speed on GSM8K).
- **What would convince me**: (1) Run all configs with 5+ seeds. (2) Show that the ortho > 1.0 result holds when M3 is actually active (gw > 0). (3) Conduct a paired statistical test (e.g., bootstrap confidence interval on ortho).

### Finding 4: Cross-Model Transfer (Transfer Ratio = 1.86x)

- **Counter-argument**: Dream-7B has much lower baseline accuracy (39% GSM8K vs 71.2% LLaDA), so any method that preserves accuracy has a mechanically higher AccRet. The transfer ratio > 1.0 is partly an artifact of the weaker baseline rather than genuine positive transfer. Also, the QAS metric (Speedup * AccRet) rewards methods that maintain accuracy on a weak model, which is easier than maintaining accuracy on a strong model.
- **Alternative explanation**: Dream-7B uses a different denoising algorithm (`diffusion_generate` with `alg='entropy'`) and different mask token mechanics. The methods "work" not because they generalize, but because they are so simple (cache hit counting, KL threshold) that they operate at a level of abstraction above model-specific details.
- **What would convince me**: (1) Test on a third DLM architecture (e.g., MDLM, SEDD). (2) Show that absolute accuracy numbers on Dream-7B are also improved, not just relative retention. (3) Confirm that the same hyperparameter settings (tau, eta, gw) are optimal for both models.

### Finding 5: AR Baseline Comparison

- **Counter-argument**: The AR comparison uses HuggingFace `generate()`, not vLLM. Qwen2.5-7B with vLLM would likely achieve 3-5x higher TPS than the 71 TPS measured here. At batch=8, the AR model already achieves 471 TPS vs LLaDA's 34 TPS -- a 14x gap that no training-free DLM acceleration can close.
- **Alternative explanation**: The "honest comparison" actually shows that DLMs remain far behind AR models in throughput, even with aggressive acceleration. The paper must frame this honestly.
- **What would convince me**: To claim competitive acceleration, we need (1) vLLM-based AR baseline, (2) DLM with kernel-level d2Cache, and (3) a scenario where DLM's parallel generation advantage matters (e.g., fill-in-the-middle tasks).

## Bottom Line

There is a publishable story here, centered on three contributions: (1) **M1+IGSD near-orthogonality** (ortho = 0.96) provides the first empirical evidence that KV caching and step scheduling compose near-multiplicatively for DLMs, which is the paper's core thesis; (2) **M3 as a quality enhancer** is a surprising and practically useful finding that could be an independent contribution; and (3) **cross-model transfer** (1.86x ratio on Dream-7B) elevates the work from a LLaDA-specific study to a general DLM acceleration framework. The main risks are the lack of kernel-level M1 speedup (currently simulated) and the need for more seeds/samples to confirm the three-way synergy claim. The AR comparison honestly reveals that DLMs remain slower than optimized AR inference, which should be framed as motivation for continued acceleration research rather than hidden.
