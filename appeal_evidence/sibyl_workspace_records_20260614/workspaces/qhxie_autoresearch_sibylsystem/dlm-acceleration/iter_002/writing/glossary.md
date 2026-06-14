# Glossary -- ComposeAccel (Iteration 2)

All section writers, critics, and the editor reference this file to ensure consistent terminology throughout the paper. Preferred phrasing is indicated; alternatives are listed to avoid.

---

## Core Concepts

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| Diffusion Language Model (DLM) | A language model that generates text via iterative denoising of masked tokens. | "diffusion language model" (lowercase unless starting a sentence); abbreviate as "DLM" after first use |
| Masked Diffusion Language Model (MDM) | A DLM that operates via discrete masking and unmasking (e.g., LLaDA, Dream). Synonym for DLM in this paper. | Use "DLM" consistently; "MDM" acceptable in related work when citing papers that use it |
| Autoregressive model (AR) | A model that generates tokens left-to-right, one at a time. | "autoregressive model" or "AR model"; NOT "auto-regressive" |
| Training-free acceleration | Inference speedup techniques that require no additional model training or fine-tuning. | "training-free"; NOT "training free" (no hyphen) or "inference-time optimization" |
| Denoising step | One forward pass of the DLM during generation, transitioning from step $t$ to $t+1$. | "denoising step" or "step"; NOT "iteration" (ambiguous with experiment iterations) |

## Methods

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| M1 (Entropy-Based KV Caching) | Reuses key-value matrices across denoising steps for positions with low entropy. | "M1" or "entropy-based KV caching"; NOT "EntropyCache" (which refers to the specific prior work) |
| M2 (Adaptive Step Scheduling) | Reduces total denoising steps by unmasking more tokens per step. Excluded (NO_GO). | "M2" or "adaptive step scheduling"; always note "excluded" or "NO_GO" when mentioned |
| M3 (AR-Guided Unmasking) | Uses a lightweight AR model to guide which masked tokens to unmask at each step. | "M3" or "AR-guided unmasking"; NOT "FlashDLM" (which refers to the prior work) |
| IGSD (Information-Geometric Step Distillation) | Our training-free speculative step scheduler using inter-step KL divergence to partition tokens into draft/refine phases. | "IGSD"; spell out on first use; NOT "CD-SSD" (iter_001 name, deprecated) |
| Draft phase | The first $T_{\text{draft}}$ steps of IGSD where all tokens are denoised. | "draft phase"; NOT "draft stage" |
| Refine phase | Steps $T_{\text{draft}}$ to $T_{\text{full}}$ where only non-frozen tokens continue denoising. | "refine phase"; NOT "verification phase" (we do not verify) |
| Confidence partitioning | The mechanism by which IGSD splits tokens into frozen ($\mathcal{S}_{\text{accept}}$) and active ($\mathcal{S}_{\text{reject}}$) sets. | "confidence partitioning" or "confidence gate" |
| Frozen tokens | Tokens in $\mathcal{S}_{\text{accept}}$ whose values are fixed after the draft phase. | "frozen tokens"; NOT "accepted tokens" or "locked tokens" |

## Metrics

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| Quality-Adjusted Speedup (QAS) | $\text{Speedup} \times \text{Accuracy Retention}$. A single scalar combining speed and quality. | "QAS"; spell out on first use |
| Accuracy retention (AccRet) | $\text{Acc(method)} / \text{Acc(baseline)}$. Fraction of baseline accuracy preserved. | "accuracy retention"; NOT "accuracy ratio" or "quality retention" |
| Orthogonality (Ortho) | $\text{QAS(A+B)} / \max(\text{QAS(A)}, \text{QAS(B)})$. Measures composition efficiency. | "orthogonality" or "Ortho score"; NOT "composability score" |
| Cache hit rate (CHR) | Fraction of token positions that reuse cached KV rather than recomputing. | "cache hit rate" or "CHR"; NOT "cache ratio" |
| Accept rate | Fraction of IGSD draft steps where the draft output is accepted. | "accept rate"; NOT "acceptance rate" |
| Tokens per second (TPS) | Wall-clock throughput measured as total generated tokens divided by generation time. | "TPS"; NOT "throughput" (too vague) or "tok/s" |

## Benchmarks

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| GSM8K | Grade school math word problems. 1319 samples. Exact match evaluation. | "GSM8K"; NOT "GSM-8K" or "gsm8k" |
| MATH500 | 500 competition mathematics problems. Exact match evaluation. | "MATH500"; NOT "MATH-500" or "math500" |
| HumanEval | 164 Python programming problems. pass@1 evaluation. Reported in appendix only. | "HumanEval"; NOT "Human-Eval" |
| MBPP | Mostly Basic Python Programs. Dropped (0% baseline). | "MBPP"; do not reference in main text |

## Models

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| LLaDA-8B-Instruct | 8-billion parameter masked diffusion language model. Primary evaluation model. | "LLaDA-8B-Instruct" or "LLaDA-8B" after first use |
| Dream-7B-Instruct | 7-billion parameter masked diffusion language model. Cross-model validation. | "Dream-7B-Instruct" or "Dream-7B" after first use |
| Qwen2.5-0.5B | 0.5B parameter AR model used as guide model in M3 and as draft model in AR speculative decoding. | "Qwen2.5-0.5B"; NOT "Qwen-0.5B" |
| Qwen2.5-7B-Instruct | 7B parameter AR model used as AR baseline comparison. | "Qwen2.5-7B-Instruct" or "Qwen2.5-7B" |

## Hardware and Implementation

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| RTX PRO 6000 Blackwell | NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. 97 GB VRAM. | "RTX PRO 6000 Blackwell" on first use; "RTX PRO 6000" subsequently |
| d2Cache | Third-party kernel-level KV cache library for DLMs. Integration attempted but failed due to framework overhead. | "d2Cache"; NOT "D2Cache" or "d2cache" |
| Projected speedup | Speedup estimated from measured cache hit rate, not directly measured via wall-clock TPS improvement. | "projected speedup"; always distinguish from "measured speedup" |

## Composition Terminology

| Term | Definition | Preferred Phrasing |
|------|-----------|-------------------|
| Pairwise composition | Combining exactly two acceleration methods. | "pairwise composition"; NOT "binary combination" |
| Three-way composition | Combining all three methods (M1 + IGSD + M3). | "three-way composition"; NOT "triple combination" |
| Synergy | Ortho > 1.0; composition is strictly better than either component alone. | "synergy" |
| Near-orthogonal | 0.8 <= Ortho <= 1.0; composition preserves most individual benefit. | "near-orthogonal"; hyphenated |
| Interference | Ortho < 0.8; composition degrades performance below the best individual method. | "interference" or "destructive interference" |
| Composition ratio | $\text{Combined speedup} / \prod(\text{individual speedups})$. Measures multiplicativity. | "composition ratio"; NOT "composition efficiency" |
| Overhead stacking | The phenomenon where per-step overhead from multiple methods accumulates subadditively. | "overhead stacking" |
| NO_GO | Verdict for a method that fails structurally and cannot be included in compositions. | "NO_GO"; uppercase with underscore |

## Abbreviations Reference

| Abbreviation | Expansion |
|-------------|-----------|
| DLM | Diffusion Language Model |
| AR | Autoregressive |
| QAS | Quality-Adjusted Speedup |
| CHR | Cache Hit Rate |
| TPS | Tokens Per Second |
| KL | Kullback-Leibler (divergence) |
| IGSD | Information-Geometric Step Distillation |
| AccRet | Accuracy Retention |
| Ortho | Orthogonality metric |
