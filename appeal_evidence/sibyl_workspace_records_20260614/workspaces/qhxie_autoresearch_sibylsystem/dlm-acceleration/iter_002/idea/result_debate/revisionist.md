# Revisionist Analysis: ComposeAccel iter_002 Results

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| **H1: Composition Subadditivity** — Three-way composition ratio < 0.5 on GSM8K, > 0.7 on simpler tasks | **Refuted** | Three-way ortho on GSM8K *exceeds* 1.0 (mean 1.02 for Max-Speed, 1.03 for Balanced-A), not < 0.5. MATH500 ortho is 0.67-0.77, not > 0.7. The direction of task dependence is inverted from prediction. | High |
| **H2: Quality-First Composition Dominates** — M1+M3 yields strictly better Pareto frontier than single-axis aggressive acceleration | **Refuted** | M1+M3 shows *interference*, not synergy. Combined ortho = 0.41-0.43 (severe). GSM8K speedup is 0.86x (a slowdown vs. baseline). M3 alone (QAS 1.68-2.19) dominates M1+M3 (QAS 0.88-0.90) on every metric. The "quality insurance" hypothesis failed: M3's guidance overhead drowns out M1's marginal speedup. | High |
| **H3: IGSD as Composable Module** — M1+IGSD composition ratio > 0.8 | **Partially Confirmed** | GSM8K ortho for M1+IGSD ranges from 0.885-0.989 depending on config, confirming near-multiplicativity on reasoning tasks. But MATH500 ortho is only 0.64-0.67, failing the > 0.8 threshold. M1 and IGSD compose well in one dimension but poorly in another. | Medium |
| **H4: Task-Dependent Optimal Recipes** — Different tasks favor different method combinations | **Confirmed (but inverted from prediction)** | GSM8K: M1+IGSD achieves near-orthogonal composition (ortho ~0.96). Code benchmarks (HumanEval/MBPP) are unusable — baseline pass@1 is 2.4% / 0%. MATH500 uniformly shows worse composition ratios than GSM8K across all pairs. The prediction that "code tolerates aggressive acceleration" cannot be tested because the model cannot generate code. | Medium |
| **H5: KL as Sufficient Signal** — 37.5% step reduction at iso-accuracy | **Refuted** | At T_draft=32 (50% step reduction), GSM8K accuracy drops to 49.5% (32% degradation from 73% baseline). At T_draft=48, accuracy improves but speedup approaches 1.0x. The KL signal identifies temporal consistency but cannot prevent the cumulative error from missed correction steps. No threshold achieves iso-accuracy with meaningful step reduction. | High |
| **H6: IGSD Phase Characterization** — Inverted-U KL profile | **Inconclusive** | Raw KL profiles show high variance across samples. Sample 0 shows elevated KL at steps 36-55 (late phase, not mid-phase as predicted). The "inverted-U" shape is not clearly present; instead, KL appears noisy with sporadic spikes throughout. The kl_non_monotonic test in pass_criteria returned false. Additional evidence: systematic averaging across all 100 profiles is needed to resolve whether a statistical trend exists beneath the noise. | Low |

## 2. Surprise Analysis

### Surprise 1: M1+M3 composition is destructive, not synergistic (deviation: QAS dropped from predicted ~2.5 to actual 0.88)

**Wrong assumption:** We assumed M1 (KV cache) and M3 (AR guidance) operate on orthogonal axes — M1 reduces per-step attention cost while M3 improves token quality through an external signal. These should compose cleanly because they target different bottlenecks.

**What actually happened:** M1 with entropy threshold 0.5 achieved only 1.16x speedup (not the projected 2-3x from cache hit rates of 56-93%). The M1 implementation runs full forward passes regardless of cache hits — it only marks which attention heads *would* be skippable, but does not actually skip computation. Combined with M3's overhead (which adds a second model forward pass per step), M1+M3 runs *slower* than vanilla baseline (0.86x on GSM8K). The composition "works" in the sense that accuracy is preserved (AccRet ~1.02), but the speedup is negative. The d2Cache kernel-level integration pilot confirmed this: d2Cache's model was 15x slower due to requiring eager attention instead of SDPA/FlashAttention.

**Root cause:** The M1 "speedup" in our measurement is an artifact of TPS variance, not real computation savings. Without kernel-level sparse attention, entropy-based caching is purely a measurement tool (cache hit rate) rather than an acceleration technique. We were measuring composition of a method that does not actually accelerate.

### Surprise 2: Three-way composition outperforms pairwise on GSM8K (ortho > 1.0)

**Wrong assumption:** Adding a third method should increase interference, pushing the composition ratio further below 1.0.

**What actually happened:** The best three-way configs (M1+IGSD+M3_gw=0.0, i.e., M3 disabled) achieve ortho 1.02-1.03. But crucially, all top-5 three-way configs have `guidance_weight=0.0` (M3 turned off) or `guidance_weight=0.3` (minimal M3). The "three-way" result is effectively a two-way M1+IGSD result. The Quality-First recipe (M1+IGSD+M3_gw=0.3) drops to ortho 0.49, confirming that adding M3 creates interference. The Pareto search correctly identified that the best strategy is to NOT use M3.

### Surprise 3: Qwen2.5-7B-Instruct vastly outperforms LLaDA-8B on GSM8K (96% vs. 71.2%)

**Wrong assumption:** We planned an "honest DLM-vs-AR comparison" assuming competitive baselines. The proposal framed DLM acceleration as a way to close the gap with AR models.

**What actually happened:** The AR baseline (Qwen2.5-7B, batch=1, greedy) achieves 96% GSM8K accuracy at 71 TPS. LLaDA-8B at 64 steps achieves 71.2% at 34 TPS. Even the best accelerated LLaDA configuration (M1+IGSD, ~1.7x speedup) reaches only ~52% accuracy at ~58 TPS. The AR model is simultaneously faster AND more accurate by a massive margin (34% accuracy gap, 2.1x TPS gap). This makes the entire "composition recipe" framing less impactful: even a hypothetical 10x DLM speedup would not close the quality gap.

### Surprise 4: Dream-7B shows *higher* composition ortho than LLaDA-8B (QAS 1.97 vs. 1.07)

**Wrong assumption:** We expected LLaDA results to transfer to Dream with some degradation, since Dream is a different architecture.

**What actually happened:** Dream-7B's baseline accuracy is much lower (36% GSM8K vs. 73% for LLaDA). When accelerated, Dream achieves AccRet of 1.25 (accuracy actually improves with M1+IGSD) — likely because the stochastic masking pattern at 32 draft steps happens to produce a beneficial regularization effect for Dream's weaker baseline. This inflates QAS artificially. The "improvement" is a statistical artifact of low baseline performance combined with sampling noise, not a genuine finding about composition quality.

### Surprise 5: Batch size kills composition benefits (QAS drops from 1.01 at bs=1 to 0.60 at bs=8)

**Wrong assumption:** We expected composition benefits to be batch-size invariant because individual methods scale with batch size.

**What actually happened:** At batch_size=4, M1+IGSD speedup drops from 1.64x to 0.96x (below 1.0x — a slowdown). At batch_size=8, speedup is 0.87x. IGSD's speculative draft-verify loop introduces serialization that conflicts with batched inference. Each sample in the batch may accept/reject at different steps, requiring synchronization points that eliminate parallelism gains. This is a fundamental architectural limitation of speculative-style methods in batched settings.

## 3. Mental Model Revision

**Previous mental model:** DLM acceleration methods decompose into three orthogonal axes (KV cache, step scheduling, AR guidance). The research question was "how do they compose?" implying that meaningful composition is achievable and the interesting question is the composition ratio.

**Revised mental model:** The three axes are NOT orthogonal — they interact through the shared denoising trajectory in ways that make composition analysis a study of *interference patterns* rather than *synergy potential*. More fundamentally, the M1 axis (KV cache) does not actually accelerate inference without kernel-level integration (which fails on current DLM architectures due to FlashAttention/SDPA dependency). The M3 axis (AR guidance) provides quality improvement at the cost of latency, making it a quality method, not an acceleration method. Only IGSD (step scheduling) provides genuine acceleration, but at severe quality cost (30-40% accuracy degradation). The real finding is that *there is no free lunch in DLM acceleration*: every method trades quality for speed on a fixed frontier, and composition shifts along this frontier rather than expanding it.

## 4. Reframing Test

**Original research question:** "Which combinations of training-free DLM acceleration methods are truly orthogonal, and what is the Pareto frontier of the combined design space?"

**Revised research question (if starting today):** "Why do training-free DLM acceleration methods interfere rather than compose, and what does this interference reveal about the computational structure of diffusion language model denoising?"

The original question presupposes that composition is valuable and the interesting finding is the degree of orthogonality. The data shows that composition is largely destructive (M1+M3, M3+IGSD) or provides only marginal benefit over the dominant individual method (M1+IGSD ortho ~1.0, meaning composition is exactly as good as the best single method, not better). A better framing would investigate *why* DLM denoising resists modular acceleration — this connects to deeper questions about the information-theoretic structure of the denoising process and has more lasting intellectual value than a Pareto frontier plot.

Additionally, the massive AR-vs-DLM gap (96% vs. 71% on GSM8K, 71 vs. 34 TPS) suggests a more honest framing: "Can training-free acceleration close the practical gap between DLMs and AR models?" Answer: No, not remotely. This is a publishable negative result if framed carefully.

## 5. New Hypothesis Generation

### NH1: DLM denoising steps are informationally non-decomposable

**Statement:** The inter-step dependencies in DLM denoising are fundamentally different from the KV dependencies in AR models: each denoising step reads and writes *all* token positions simultaneously, creating a fully-connected dependency graph that cannot be locally approximated. This predicts that any method that approximates or skips a denoising step will degrade quality proportionally to the fraction of the step's information that is lost, regardless of how the approximation is implemented.

**Falsification test:** Measure the mutual information between step t and step t+1 logits across all positions. If steps in the late phase (t > 50) contribute < 1 bit per token, then skipping them should be safe. If they contribute > 5 bits per token even at convergence, the non-decomposability hypothesis holds.

**Experiment:** Run the 100-sample KL profile analysis with proper averaging and also compute per-token mutual information estimates. Compare against the IGSD accuracy degradation curve: if MI correlates with degradation, the hypothesis is supported.

### NH2: M1+M3 interference is caused by attention pattern corruption from cached states

**Statement:** M1's entropy-based caching corrupts the attention pattern that M3's AR guidance relies on for quality improvement. Specifically, when M1 skips attention computation for "converged" heads, the resulting output provides a systematically biased context for M3's Qwen guidance signal, causing the guidance to amplify errors rather than correct them.

**Falsification test:** Run M1+M3 with M1 applied only to the first 32 steps and M3 applied only to the last 32 steps (non-overlapping). If interference disappears (ortho > 0.8), the corrupted-attention-pattern hypothesis is confirmed. If interference persists, the methods interfere through a different mechanism (e.g., both compete for control over the unmasking order).

**Experiment:** Two-phase hybrid: steps 1-32 use M1 only, steps 33-64 use M3 only. Compare to full overlap M1+M3. Estimated runtime: 15 minutes on 100 GSM8K samples.

### NH3: IGSD's draft quality is limited by the verifier, not the drafter

**Statement:** IGSD's accept rate is high (~96%) because the verifier (LLaDA at 64 steps) is not discriminative enough to reject low-quality drafts. The 30-40% accuracy degradation occurs not because bad drafts are accepted (the accept rate suggests they pass verification), but because the verifier's own correction capacity is insufficient to fix the errors introduced by fewer draft steps.

**Falsification test:** Run IGSD with a stronger verifier (e.g., 128 steps instead of 64) at T_draft=32. If accuracy retention improves to > 80% with minimal speedup loss, the verifier-bottleneck hypothesis is confirmed. If accuracy retention stays at ~60%, the problem is in the draft phase.

**Experiment:** Modify IGSD to use T_full=128 instead of 64. Run on 100 GSM8K samples. Compare accuracy retention vs. T_full=64. Estimated runtime: 20 minutes (one extra forward pass per verification).
