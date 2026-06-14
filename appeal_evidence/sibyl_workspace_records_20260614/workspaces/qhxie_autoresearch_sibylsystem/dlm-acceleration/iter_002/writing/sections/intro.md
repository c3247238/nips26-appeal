# Introduction

LLaDA-8B-Instruct achieves 71.2% accuracy on GSM8K but generates text at 34 tokens per second (TPS) --- 2.1x slower than Qwen2.5-7B at batch=1 (71 TPS) and 13.9x slower at batch=8 (471 TPS). This speed deficit is structural: each of the $T = 64$ denoising steps in a diffusion language model (DLM) requires a full $O(N^2)$ forward pass, and the global mask-state update between steps prevents standard KV cache reuse. In Q1 2026 alone, over 20 training-free acceleration methods have targeted this bottleneck across three computational axes: KV cache approximation [Fast-dLLM, EntropyCache, dKV-Cache, Elastic-Cache], adaptive step scheduling [Saber, PRR], and guided or speculative decoding [FlashDLM, SSD, SSMD, S2D2, DualDiffusion].

Every one of these methods is evaluated in isolation. Practitioners who must deploy DLMs face a concrete unanswered question: which combination of methods maximizes throughput at acceptable quality? The answer is not obvious. Methods targeting different computational bottlenecks should compose multiplicatively, but DLM's global mask-state coupling creates hidden interaction risks --- a speed gain on one axis can silently degrade the quality signal another axis depends on.

## The Composition Gap

Three specific problems block progress toward practical DLM deployment:

**Unknown composability.** No published work measures pairwise or higher-order orthogonality between DLM acceleration methods. TORS (arXiv:2603.00763) identifies the same gap for text-to-image diffusion, noting that training-free methods are "developed independently, leaving compatibility unexplored." Kolbeinsson et al. (2024) measure composable interventions for LLMs (compression, editing, pruning), but these are model-modification techniques, not inference acceleration. For DLM inference acceleration, zero composition data exists.

**No failure characterization.** Published methods report average-case results under method-specific protocols. A practitioner cannot predict from these numbers whether combining entropy-based KV caching ($\eta = 0.5$, 1.16x speedup, 94.5% accuracy retention) with AR-guided unmasking ($w_g = 0.3$, 1.65x speedup, 102.5% accuracy retention) will yield 1.91x speedup or 0.86x slowdown. Our experiments show the answer is 0.86x --- destructive interference from overhead stacking. Without composition data, every deployment is a gamble.

**No honest AR comparison.** DLM acceleration papers benchmark against the unaccelerated DLM baseline. None compare against properly optimized autoregressive (AR) inference at the same model scale. Without this comparison, the practical value of DLM acceleration remains unclear. Qwen2.5-7B achieves 96% GSM8K accuracy at 71 TPS (batch=1), establishing a Quality-Adjusted Speedup (QAS) of 3.08 relative to LLaDA's baseline --- a gap that composed DLM acceleration narrows but does not close.

Figure 1 shows the speed-quality landscape that motivates this work. Individual methods cluster in distinct regions (M3 preserves quality, IGSD trades accuracy for speed, M1 offers marginal measured speedup), and compositions span the space unpredictably. The M1+M3 pair falls *below* both individual methods --- a result invisible without systematic composition measurement.

![Speed-quality landscape showing individual methods and compositions on LLaDA-8B-Instruct (GSM8K). Points below the Pareto frontier represent suboptimal or interfering compositions. M1+M3 achieves 0.86x speedup (a net slowdown) despite 102.5% accuracy retention, falling below both M1 alone and M3 alone in QAS due to overhead stacking from dual model loading.](figures/fig_teaser.pdf)

## Contributions

We present **ComposeAccel**, the first controlled factorial study of training-free DLM acceleration composition. Our contributions are:

1. **First systematic composition study** of three training-free DLM acceleration families across three axes, with a formal orthogonality metric (Ortho) and corrected Quality-Adjusted Speedup (QAS). We evaluate all three pairwise combinations and five three-way configurations on LLaDA-8B-Instruct across GSM8K (1319 samples) and MATH500 (500 samples) with three-seed validation.

2. **Composition taxonomy.** M1+IGSD achieves near-orthogonal composition (Ortho = 0.96 on the combined metric, 2.75x speedup on GSM8K). M3+IGSD is task-dependent (near-orthogonal on GSM8K with Ortho = 0.96, interference on MATH500). M1+M3 shows destructive interference (Ortho = 0.41--0.52) driven by the 12% per-step overhead of the guide model negating M1's marginal 1.16x measured speedup. M3 guidance ($w_g > 0$) consistently reduces Ortho from approximately 1.0 to approximately 0.5 in three-way compositions.

3. **IGSD (Information-Geometric Step Distillation).** A 50-line training-free step scheduler using inter-step logit KL divergence to partition tokens into draft and refine phases. At $T_{\text{draft}} = 32$, $\tau = 0.9$, IGSD achieves 1.71x speedup at 67.8% accuracy retention (QAS = 1.16 on GSM8K). $T_{\text{draft}} = 32$ is Pareto-optimal; the per-step KL divergence profile is monotonically decreasing (not inverted-U as hypothesized), explaining IGSD's low sensitivity to the threshold $\tau$.

4. **Cross-model validation.** Top 5 recipes validated on Dream-7B-Instruct show amplified composition effects (Dream M1+IGSD+M3$_{\text{off}}$ QAS = 2.18 on GSM8K vs. LLaDA QAS = 1.07), confirming that composition patterns transfer across DLM architectures.

5. **Honest AR comparison.** Qwen2.5-7B at batch=1 achieves QAS = 3.08 relative to the LLaDA baseline (96% GSM8K accuracy, 71 TPS). The best composed DLM acceleration (M1+IGSD, QAS = 1.07) falls 2.9x short. DLM acceleration does not close the speed gap with AR models; the value of this study lies in characterizing the composition design space, not in claiming speed parity.

6. **Practical recipes.** Task-specific recommended combinations with validated hyperparameters: M1 ($\eta = 0.5$) + IGSD ($\tau = 0.85$, $T_{\text{draft}} = 32$) without M3 guidance for maximum speed; M3 ($w_g = 0.3$) standalone for quality-preserving acceleration.

## Scope Statement

ComposeAccel is an analysis paper, not a methods paper. IGSD is a composability study vehicle --- simple by design so that composition effects are attributable to method interactions rather than implementation complexity. The primary model is LLaDA-8B-Instruct; Dream-7B-Instruct provides cross-model validation. Primary benchmarks are GSM8K (1319 samples, exact match) and MATH500 (500 samples, exact match); HumanEval is reported in the appendix (2.4% baseline, uninformative). MBPP is excluded (0% baseline). M1 reports measured cache hit rate (56--99%) and projected speedup (2.27--2.47x) because d2Cache kernel integration produced 15.2x framework overhead due to eager attention incompatibility on RTX PRO 6000 Blackwell GPUs; the M1 measured speedup (1.16x) reflects our simplified Python implementation without kernel-level cache reuse. M2 (adaptive step scheduling) is excluded as a structural NO_GO and reported as a negative result in Section 7.

Section 3 surveys the DLM acceleration landscape and the composability gap in the literature. Section 4 defines the methods, metrics, and experimental setup. Section 5 reports single-method, pairwise, and three-way composition results with IGSD ablations. Section 6 presents cross-model validation on Dream-7B and the AR baseline comparison. Section 7 discusses composition mechanisms, design implications, and limitations.

<!-- FIGURES
- Figure 1: gen_fig_teaser.py, fig_teaser.pdf — Speed-quality landscape teaser scatter plot showing individual methods and compositions with Pareto frontier
-->
