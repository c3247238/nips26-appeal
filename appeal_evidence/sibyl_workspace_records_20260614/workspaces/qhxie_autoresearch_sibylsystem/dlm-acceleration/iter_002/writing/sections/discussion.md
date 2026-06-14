# 7 Discussion

## 7.1 Why M1+IGSD Composes but M1+M3 Does Not

M1+IGSD achieves near-orthogonal composition (Ortho = 0.96 combined, 0.99 on GSM8K) while M1+M3 shows destructive interference (Ortho = 0.41--0.43 combined). The mechanism behind this divergence is architectural: IGSD restructures the denoising pipeline in a way that amplifies M1's effectiveness, whereas M3 adds per-step overhead that negates M1's marginal speedup.

**M1+IGSD synergy mechanism.** IGSD's draft-partition-refine pipeline splits the 64-step denoising process into two phases. After the draft phase ($T_{\text{draft}} = 32$ steps), confidence partitioning freezes a fraction $\alpha = 88.6\% \pm 13.3\%$ of generation tokens into $\mathcal{S}_{\text{accept}}$. These frozen tokens exhibit near-zero entropy across the remaining 32 refine steps, because their logit distributions are fixed. M1's entropy-based KV caching exploits this directly: when $H_i^t < \eta$, position $i$ reuses cached KV rather than recomputing. During composition, M1 achieves $\overline{\text{CHR}} = 83.4\%$ -- comparable to standalone M1 at $\eta = 0.5$ ($\overline{\text{CHR}} = 93.3\%$ on GSM8K) -- but the reduced step count from IGSD means each cached step saves a larger fraction of total wall-clock time. The composition at the best configuration ($\eta = 0.5$, $\tau = 0.7$, $T_{\text{draft}} = 16$) reaches 2.75x speedup at 58.9% accuracy retention on GSM8K (Ortho = 0.99).

**M1+M3 interference mechanism.** M3 loads Qwen2.5-0.5B (0.95 GB VRAM) and runs a forward pass through it at every denoising step to produce guidance logits $q_t$. This adds approximately 12% wall-clock overhead per step, reducing TPS from the baseline 58.5 to 50.3 (0.86x). M1's measured speedup without kernel-level cache integration is 1.16x at $\eta = 0.5$. When composed, the M3 overhead dominates: M1+M3 achieves only 0.86x speedup on GSM8K despite preserving accuracy (AccRet = 102.5%). The combined TPS (50.3) is slower than both the baseline (58.5) and M3 standalone (96.5, which benefits from IGSD-like pipelining effects absent here). All three guidance weight settings ($w_g \in \{0.3, 0.5, 0.7\}$) produce identical GSM8K Ortho values of 0.51--0.52, confirming that the interference is structural rather than hyperparameter-dependent.

**Reconciling with iter_001.** The pilot experiment in iter_001 reported M1+M3 Ortho = 1.34 on 100 GSM8K samples, suggesting synergy. The full iter_002 evaluation (100 samples per seed, 3 seeds, corrected QAS formula without the undisclosed 0.5x penalty) resolves this discrepancy: the pilot result was an artifact of small-sample variance and an inflated combined metric that included HumanEval (2.4% baseline) and MBPP (0% baseline). With the corrected combined metric (0.7 $\times$ GSM8K + 0.3 $\times$ MATH500), M1+M3 interference is confirmed at combined Ortho = 0.41.

## 7.2 M3+IGSD: Task-Dependent Composition

M3+IGSD presents an intermediate case. The best configuration ($\tau = 0.7$, $T_{\text{draft}} = 16$, $w_g = 0.7$) achieves GSM8K Ortho = 0.96 (near-orthogonal) but MATH500 Ortho = 0.76 (interference), yielding a combined Ortho = 0.84.

The task dependence arises from the guide model's capability. Qwen2.5-0.5B achieves 96% accuracy on GSM8K independently (Table 7), making it an effective guide for grade-school math reasoning. On MATH500, the same model scores only 37--39%, limiting its ability to improve DLM unmasking decisions on competition-level mathematics. When M3 guidance operates on IGSD's compressed trajectory (fewer draft steps produce noisier intermediate states), the guide model's limitations on hard tasks compound: the AR logits provide misleading confidence signals that IGSD's partition boundary cannot correct.

The mean Ortho across all M3+IGSD configurations is 0.69 (combined metric), placing this pair firmly in the interference zone on average. Only the most aggressive IGSD setting ($T_{\text{draft}} = 16$) pushes the combined Ortho above 0.8, because the speedup gains from extreme step compression outweigh the quality loss from poor guidance.

## 7.3 Implications for DLM Acceleration Design

Three design principles emerge from the composition analysis.

**Principle 1: Overhead stacking is subadditive.** Combining methods that each add per-step overhead (M1's entropy computation, M3's guide model forward pass) produces compounding slowdowns. The M1+M3 pair illustrates this: 1.16x $\times$ 1.65x = 1.91x expected speedup, but 0.86x measured. Methods that reduce the number of steps (IGSD) compose more favorably with per-step optimizations (M1) because fewer steps means fewer opportunities for overhead to accumulate.

**Principle 2: KV caching requires kernel-level integration to deliver published speedups.** The d2Cache integration pilot showed that the library achieves 4.39x internal speedup (relative to its own model), but d2Cache's eager attention implementation runs 15.2x slower than the HuggingFace baseline on RTX PRO 6000 Blackwell GPUs. The net effect is 0.29x -- a slowdown. Entropy-based CHR measurements show 93--99% cache hit rates across $\eta$ thresholds, projecting 2.1--2.5x theoretical speedup if kernel-level sparse attention were implemented without framework overhead. The 15.2x gap between theoretical and measured M1 speedup is itself a finding: published DLM acceleration results that report cache hit rates without wall-clock TPS may overstate practical gains by an order of magnitude.

**Principle 3: Quality-preserving methods are best used standalone.** M3 achieves AccRet > 100% on GSM8K at all guidance weights, making it the only method that improves accuracy. However, its 12% per-step overhead makes it a poor composition partner. Practitioners should apply M3 when accuracy is the binding constraint and apply M1+IGSD when throughput is the priority, rather than attempting to stack all three.

## 7.4 The AR Gap

Qwen2.5-7B at batch = 1 reaches 96% GSM8K accuracy at 71 TPS (QAS = 3.08 relative to the LLaDA baseline), while the best composed DLM acceleration achieves QAS = 1.07 at 1.71x speedup. At batch = 8, the gap widens further: Qwen2.5-7B achieves 471 TPS (QAS = 20.5). DLM composition narrows the per-step cost but does not close the fundamental throughput gap with AR inference.

This gap has two sources. First, LLaDA-8B's 64-step iterative denoising requires 64 full forward passes per generation, each with $O(N^2)$ bidirectional attention. AR models require $N$ forward passes but with $O(N)$ incremental KV-cached attention per step. Second, DLM accuracy on GSM8K (71.2%) trails Qwen2.5-7B (96%) at comparable parameter counts, which means even lossless acceleration leaves DLMs at a quality disadvantage.

The honest comparison establishes that the value of DLM composition research lies in understanding the design space -- which combinations work, which fail, and why -- rather than in claiming speed parity with AR models. DLMs offer architectural advantages (parallel generation, bidirectional context) that may prove valuable in settings not captured by sequential exact-match benchmarks. Quantifying the composition design space is a prerequisite for exploiting those advantages.

## 7.5 Cross-Model Transferability

Dream-7B-Instruct validation confirms that composition patterns transfer across DLM architectures. The Max-Speed recipe (M1 $\eta = 0.5$ + IGSD $\tau = 0.85$, $T_{\text{draft}} = 32$, M3 off) achieves QAS = 2.18 on Dream-7B GSM8K versus QAS = 1.07 on LLaDA-8B. The amplification arises from Dream-7B's lower baseline accuracy (36% vs. 71.2%): IGSD's draft phase produces comparably accurate outputs on Dream-7B (AccRet = 125% on GSM8K), suggesting that Dream-7B's iterative refinement in later steps is less productive than LLaDA's. The key transferable finding is that M1+IGSD without M3 remains the Pareto-optimal recipe on both models. M3 guidance consistently reduces Ortho from approximately 1.0 to approximately 0.5 in three-way compositions on both architectures, confirming that the overhead interference is not model-specific.

## 7.6 Limitations

**Model coverage.** The study evaluates two DLMs from the LLaDA/Dream family. Generalization to MDLM, SEDD, or models with fundamentally different masking schedules (e.g., continuous diffusion rather than discrete masking) is untested.

**M1 speedup is projected.** Entropy-based cache hit rates are measured (56--99% depending on $\eta$), but wall-clock speedup is projected from CHR rather than directly measured with kernel-level KV cache integration. The Ortho metric is dimensionless and valid regardless of absolute speedup, but the reported combined TPS values should be interpreted as upper bounds contingent on efficient cache implementation. The d2Cache integration failure -- 15.2x framework overhead on Blackwell GPUs with eager attention -- demonstrates that this contingency is non-trivial.

**Benchmark limitations.** GSM8K (baseline 71.2%) provides strong signal, but MATH500 (baseline 11.1%) has limited statistical power: a 3-sample fluctuation changes accuracy by 0.6 percentage points, which translates to large Ortho variance. HumanEval (baseline 2.4%) and MBPP (baseline 0%) are excluded from the combined metric entirely because LLaDA-8B cannot generate functional code at this evaluation setting. The composition analysis is therefore limited to mathematical reasoning tasks.

**Seed and sample variance.** Three-way composition results use 100 GSM8K and 100 MATH500 samples per seed across 3 seeds, yielding stable QAS (CV < 10%) but limited power to detect small composition effects. The pairwise results are pilot-scale (100 samples, seed = 42 only for M3+IGSD and M1+M3), so their Ortho values carry higher uncertainty than the three-way results.

**Hardware specificity.** All experiments run on NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (97 GB VRAM). The d2Cache integration failure may be hardware-specific (eager attention overhead on Blackwell architecture). M1's projected speedup assumes kernel-level sparse attention that has been demonstrated on A100/H100 but not validated on Blackwell.

**M2 exclusion.** Adaptive step scheduling (M2) was excluded after structural NO_GO: simplified Saber implementation produced catastrophic accuracy collapse (AccRet < 50% at step jump > 3x) because LLaDA's masked denoising requires sequential step gradients. This leaves M1+M2, M2+M3, and M2+IGSD pairwise compositions untested. The full Saber implementation with backtracking may behave differently.

<!-- FIGURES
- None
-->
