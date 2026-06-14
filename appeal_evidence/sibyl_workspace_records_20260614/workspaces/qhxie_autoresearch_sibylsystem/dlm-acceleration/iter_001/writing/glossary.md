# Glossary

Unified terminology for all paper sections. Writers, critics, and editors must use these exact terms and spellings.

---

## Core Concepts

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| Masked Diffusion Language Model (MDM) | A language model that generates text by iteratively unmasking tokens from a fully-masked canvas, using bidirectional attention at each denoising step. | Use "MDM" after first expansion. Do NOT use "discrete diffusion model" or "DLM" interchangeably in the paper body (use MDM consistently). "DLM" acceptable only when quoting external literature. |
| Denoising step | A single forward pass of the MDM that predicts token identities for currently masked positions and selectively unmasks a subset. | Not "diffusion step" or "sampling step." Always "denoising step." |
| Mask state | The binary assignment of each position to masked or unmasked at a given step $t$. The mask state changes at every denoising step. | "Mask state" (two words), not "mask-state" except in compound modifiers (e.g., "mask-state coupling"). |
| Mask-state coupling | The phenomenon where each token's unmasking probability depends on the mask state of all other tokens, creating global interdependence between denoising decisions. | Hyphenated when used as a modifier: "mask-state coupling." |
| Composability | The property that two acceleration methods can be applied simultaneously without destructive interference. Quantified by the orthogonality metric. | Not "combinability" or "composition." Always "composability." |
| Orthogonality (Ortho) | Metric measuring whether two methods compose multiplicatively. Ortho = combined speedup / product of individual speedups. | Capitalize "Ortho" when referring to the specific metric value. Lowercase "orthogonal" as adjective. |
| Super-multiplicative synergy | Ortho > 1.0: the combined speedup exceeds the product of individual speedups. | Not "super-linear" (which implies a different mathematical relationship). |
| Destructive interference | Ortho < 0.8: the combined speedup is substantially less than the product of individual speedups. | Not "negative interference" or "conflict." |
| Quality-Adjusted Speedup (QAS) | Composite metric: Speedup x Accuracy Retention. Penalizes methods that sacrifice accuracy for speed. | Always expand on first use, then "QAS." |
| Accuracy retention | Ratio of accelerated method accuracy to baseline accuracy. | Not "accuracy preservation" or "quality retention." |

## Methods

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| KV-cache approximation | Reusing Key-Value attention matrices from previous denoising steps for positions with stable representations, instead of recomputing them. | "KV-cache" (hyphenated), not "KV cache" or "KVcache." |
| EntropyCache | A specific KV-cache method that uses decoded token entropy to decide whether to refresh or reuse cached KV entries. | One word, camel case: "EntropyCache." |
| Adaptive step scheduling | Reducing total denoising steps by unmasking more tokens per step based on per-token confidence scores. | Not "step reduction" or "schedule acceleration." |
| AR-guided unmasking | Using a lightweight autoregressive model to bias the unmasking order toward tokens the AR model predicts with high confidence. | "AR-guided" (hyphenated). "AR" for autoregressive, always expanded on first use. |
| CD-SSD | Coarse-Draft Self-Speculative Denoising. A training-free method that uses a reduced-step draft pass (T_draft=16 vs. T_full=64) of the same MDM to produce candidate tokens, then selectively refines uncertain positions. Renamed from IGSD to avoid name collision with Info-Gain Sampler (Yang et al., arXiv:2602.18176). Concurrent with SSD (Gao et al., arXiv:2510.04147) and SSMD (Campbell et al., arXiv:2510.03929). | Always "CD-SSD" in this paper. Expand "Coarse-Draft Self-Speculative Denoising" on first use. Do NOT use "IGSD" in the paper body. |
| Draft phase | IGSD's first stage: run the MDM with $T_{\text{draft}}$ steps to produce a coarse output and per-token confidence scores. | "Draft phase" (not "drafting stage" or "coarse pass"). |
| Refine phase | IGSD's second stage: run full $T_{\text{full}}$ denoising steps on low-confidence positions while keeping high-confidence tokens frozen. | "Refine phase" (not "verification phase" or "refinement stage"). |
| Frozen tokens | Tokens in $S_{\text{accept}}$ that are fixed during IGSD's refine phase and not subject to further denoising. | "Frozen tokens" (not "accepted tokens" or "cached tokens"). |
| Self-speculative decoding | A speculative decoding approach where the same model serves as both draft generator and verifier, differing only in computational budget. | Hyphenate as modifier: "self-speculative approach." |

## Method IDs

| ID | Full Name | Status |
|----|-----------|--------|
| M1 | KV-Cache Approximation (EntropyCache) | Active (operating point: $\eta = 2.0$) |
| M2 | Adaptive Step Scheduling (Simplified Saber) | NO_GO (negative result, reported in failure atlas; simplified implementation without backtracking) |
| M3 | AR-Guided Unmasking (Qwen2.5-0.5B) | Active (reasoning only; operating point: $w = 0.3$, combined speedup 1.33x) |
| CD-SSD | Coarse-Draft Self-Speculative Denoising (formerly IGSD) | Active (operating point: $\tau = 0.9$, $T_{\text{draft}} = 16$); tau=0.0 ablation confirms step reduction is the primary speedup driver, not confidence partitioning |

## Models

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| LLaDA-8B-Instruct | The primary evaluation model: GSAI-ML/LLaDA-8B-Instruct from HuggingFace. An 8-billion parameter MDM fine-tuned for instruction following. | Always include "-Instruct" suffix. Full name on first use, then "LLaDA-8B" is acceptable. |
| Dream-7B-Instruct | Secondary evaluation model for cross-model validation: hkunlp/dream-7b-instruct. | "Dream-7B" after first use. |
| Qwen2.5-0.5B | Lightweight AR model used as the supervisor in M3 (AR-guided unmasking). | Always "Qwen2.5-0.5B" (include version and size). |
| LLaDA | The model family/architecture. Uses block-based semi-autoregressive generation with bidirectional masked denoising. | "LLaDA" for the architecture/family; "LLaDA-8B-Instruct" for the specific model. |

## Benchmarks

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| GSM8K | Grade school math reasoning benchmark (1319 test samples). Metric: exact match. | "GSM8K" (all caps). Primary reasoning benchmark. |
| MATH500 | 500-problem subset of the MATH benchmark. Metric: exact match. | "MATH500" (all caps, no space). |
| HumanEval | Code generation benchmark (164 problems). Metric: pass@1. | "HumanEval" (camel case). Degenerate baseline (2.4% pass@1 for LLaDA-8B). |
| MBPP | Mostly Basic Python Programs benchmark (257 samples). Metric: pass@1. | "MBPP" (all caps). Degenerate baseline (0.0% pass@1 for LLaDA-8B). |
| Degenerate baseline | A benchmark where the baseline model's accuracy is too low (< 5%) for acceleration metrics to be meaningful. | Use "degenerate baseline" explicitly when discussing HumanEval/MBPP limitations. |

## Infrastructure

| Term | Definition | Usage Notes |
|------|-----------|-------------|
| bf16 | Brain floating-point 16-bit precision. All experiments use bf16 inference. | Lowercase "bf16." Not "bfloat16" or "BF16" in running text. |
| TPS | Tokens per second. Wall-clock throughput metric. | "TPS" after first expansion. |
| Wall-clock time | Elapsed real time (not CPU time or GPU time). All throughput measurements use wall-clock time. | "Wall-clock" (hyphenated as modifier). |
| Warm-up samples | Initial samples discarded from TPS measurement to exclude JIT compilation and cache warm-up overhead. | "Warm-up" (hyphenated). |

## External Methods (Referenced but Not Implemented)

| Term | Definition |
|------|-----------|
| SSD | Self-Speculative Decoding for discrete language models. Achieves 3.46x lossless speedup. |
| Fast-dLLM | Training-free acceleration via approximate KV-cache and parallel decoding. Up to 27.6x speedup. |
| DualDiffusion | Speculative decoding for MDMs using an external draft model. |
| DFlash | Block diffusion as speculative draft for AR target model verification. |
| Saber | Adaptive acceleration with backtracking-enhanced remasking. Training-free. |
| Block Diffusion | Hybrid AR-diffusion: autoregressive across blocks, parallel within blocks. Training-based. |

## Key Resolved Findings (for writer reference)

| Finding | Status | Evidence |
|---------|--------|----------|
| tau=0.0 paradox | RESOLVED: CD-SSD(tau=0.0) ≈ naive-T16 (-5.8% QAS diff, within noise) | full_tau0_comparison.json |
| Wilcoxon p<0.05 for H4 | FABRICATED — remove; use ranking observation only | No test in task_dependence_full.json |
| QAS formula inconsistency | FIXED in outline: use standard Speedup×AccRet for all methods; no hidden 0.5× penalty | critique_experiment.md |
| M3 speedup 1.68x vs. 1.33x | USE 1.33x (combined); 1.68x is GSM8K-specific only | m3_pareto_full.json |
| IGSD → CD-SSD rename | COMPLETE in outline/glossary/notation | novelty_report |

## Banned Terms

Do NOT use these terms in the paper. Use the preferred alternative.

| Banned | Preferred |
|--------|-----------|
| DLM (in paper body) | MDM |
| diffusion step | denoising step |
| sampling step | denoising step |
| KV cache (no hyphen) | KV-cache |
| finetuning | fine-tuning |
| fewshot | few-shot |
| zeroshot | zero-shot |
| novel (without quantification) | [describe what is new with specific evidence] |
| significantly (without p-value or effect size) | [use exact numbers] |
| state-of-the-art (as standalone claim) | [compare with specific methods and numbers] |
| in recent years | [cite specific timeframe or papers] |
| IGSD (in paper body) | CD-SSD |
| "first training-free, no-auxiliary-model speculative MDM" | "training-free, reduced-step speculative denoising concurrent with SSD and SSMD" |
| "binary composability" (as universal law) | "binary pattern observed across the three method pairs evaluated" |
| "p < 0.05" for H4 task dependence | "QAS rankings differ between reasoning and coding; formal statistical testing requires more methods" |
