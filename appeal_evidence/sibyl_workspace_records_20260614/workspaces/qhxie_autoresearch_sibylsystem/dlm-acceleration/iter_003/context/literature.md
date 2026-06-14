# Literature Survey Report

**Research Topic**: DLM (Diffusion Language Model) Inference Acceleration
**Survey Date**: 2026-04-15 (updated, iteration 3)
**arXiv Search Keywords**:
- `"diffusion language model" inference acceleration`
- `"masked diffusion" OR "discrete diffusion" language model speculative decoding KV cache`
- `fast-dLLM "block diffusion" OR "step reduction" OR "early exit" diffusion language model decoding`
- `"diffusion language model" survey review challenges limitations`
- `DyLLM saliency token selection partial attention diffusion LLM`
- `DARE diffusion language model alignment reinforcement executor`
- `S2D2 self-speculation block diffusion training-free`
- `d1 scaling reasoning diffusion large language model reinforcement learning`
- `ES-dLLM early skipping diffusion language model`
- `Focus-dLLM long-context confidence guided attention sparsification`
- `Prophet early answer convergence diffusion language model`
- `JoT token-level early stopping diffusion language model`
- `LoSA locality-aware sparse attention block diffusion`
- `I-DLM introspective diffusion language model`
- `CDLM consistency diffusion language model faster sampling`
- `ReFusion diffusion parallel autoregressive decoding`
- `d3LLM pseudo-trajectory distillation ultra-fast diffusion`
- `DAWN dependency-aware fast inference diffusion LLM`
- `dLLM-Serve production diffusion LLM serving`
- `SPA-Cache singular proxies adaptive caching diffusion`
- `Sparse-dLLM dynamic cache eviction sparse attention`

**Web Search Keywords**:
- `diffusion language model inference acceleration state of the art 2025`
- `LLaDA Dream diffusion language model benchmark evaluation 2025`
- `diffusion language model github open source code inference acceleration 2025`
- `diffusion language model inference faster than autoregressive token throughput benchmark 2025 2026`
- `DARE DyLLM S2D2 model scheduling denoising steps masked diffusion 2026`
- `speculative diffusion decoding self-speculation block diffusion 2026`
- `ES-dLLM early skipping diffusion language model arxiv 2603 2026`
- `Focus-dLLM long-context diffusion LLM confidence guided arxiv 2602 2026`
- `Gemini Diffusion Seed Diffusion Mercury commercial diffusion LLM throughput`
- `diffusion language model acceleration composability composition methods 2026`
- `I-DLM introspective diffusion language model trending 2026`
- `LLaDA 2.0 2.1 SDAR second generation diffusion language model 2026`
- `dLLM-Serve production diffusion LLM serving system 2026`

---

## 1. Field Overview

Diffusion Language Models (DLMs), also called discrete diffusion models or masked diffusion models (MDMs), have emerged as a compelling alternative to autoregressive (AR) language models. Unlike AR models that generate tokens sequentially left-to-right, DLMs operate by iteratively denoising a masked or noised sequence, enabling bidirectional context modeling and parallel token sampling. Key open-source representatives include **LLaDA 8B** (NeurIPS 2025 oral, ML-GSAI), **Dream 7B** (HKU NLP), **TESS 2** (UW/Allen AI), **LLaDA-MoE** (7B with only 1.4B active parameters), and the new **LLaDA 2.0** family (up to 100B MoE, Ant Group). Closed-source DLMs such as **Gemini Diffusion** (Google DeepMind) and **Mercury** (Inception) have demonstrated throughputs exceeding 1,400 tokens/second, 5-10x faster than comparably-sized AR models.

Despite architectural advantages, the open-source DLM community faces a fundamental efficiency paradox: in practice, most open-source DLMs (LLaDA, Dream) are **slower** than equivalently-sized AR LLMs. The root causes are: (1) bidirectional attention incompatibility with standard KV caching -- each denoising step requires a full $O(N^2)$ forward pass over the entire sequence; (2) quality degradation when multiple tokens are decoded in parallel per step, forcing near-sequential unmasking; (3) fixed-length generation canvas that inflates unnecessary computation; and (4) absence of direct equivalents to AR acceleration techniques like speculative decoding, prefix caching, and chunked prefill.

As of mid-April 2026, the field has entered a **maturation phase**: multiple acceleration families are well-established (KV caching, parallel decoding, speculative methods, step reduction, distillation, sparse attention, early stopping), and the frontier has shifted from "can individual methods work?" to three open questions: (1) **can methods compose?** (the ComposeAccel question), (2) **can DLMs match AR quality?** (answered affirmatively by I-DLM, April 2026), and (3) **can DLMs be served in production?** (addressed by dLLM-Serve, ReFusion). The **second generation** of DLMs (LLaDA 2.0/2.1, SDAR, I-DLM, ReFusion) increasingly uses hybrid AR-diffusion architectures with native KV cache support, which changes the composability landscape for acceleration methods.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | **FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion** | arXiv 2505.21467 | 2025 | FreeCache (training-free KV approximation) + Guided Diffusion (lightweight AR supervisor); 12.14x speedup | AR guidance model adds memory; quality drop at very few steps |
| 2 | **Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding** | NVlabs / ICLR 2026 | 2025 | Approximate KV cache for block-wise decoding + confidence-aware parallel decoding; up to 27.6x speedup | Requires block-wise generation regime; not applicable to all pretrained DLMs |
| 3 | **dKV-Cache: The Cache for Diffusion Language Models** | arXiv 2505.15781 | 2025 | Delayed and conditioned KV caching; two variants -- near-lossless (dKV-Cache-Decode) and aggressive (dKV-Cache-Greedy); 2-10x speedup | Greedy variant degrades quality; training-free but needs careful calibration |
| 4 | **Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking** | arXiv 2510.18165 | 2025 | Training-free: adaptive acceleration (more tokens per step as context builds) + backtracking mechanism; 251.4% speedup on code generation | Evaluated primarily on code; backtracking adds complexity |
| 5 | **Window-Diffusion: Accelerating DLM Inference with Windowed Token Pruning and Caching** | arXiv 2601.20332 | 2026 | Token-level analysis revealing structural locality; sliding window with active tokens, buffer tokens (cached KV), and pruned far-field tokens; up to 99x speedup | May sacrifice global context; evaluated mainly on LLaDA and Dream |
| 6 | **Elastic-Cache: Attention-aware Adaptive KV Refresh for Diffusion LLMs** | arXiv 2510.14973 | 2025 | Adaptive per-layer refresh schedule; drift-aware cache update via most-attended token signal; 8.7x on short, 45.1x on long sequences | Architecture-specific tuning may be needed |
| 7 | **EntropyCache: Decoded Token Entropy-Guided KV Caching for Diffusion LLMs** | arXiv 2603.18489 | 2026 | O(V) constant-cost refresh decision using decoded token entropy; avoids scaling overhead with context length; 15-26x speedup | Entropy signal may not generalize across all task types |
| 8 | **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models** | arXiv 2503.09573 | 2025 | Hybrid AR-diffusion: autoregressive across blocks, parallel within blocks; enables KV caching and flexible-length generation; SOTA on LM benchmarks | Requires retraining from scratch or fine-tuning |
| 9 | **Fast-dLLM v2: Efficient Block-Diffusion LLM** | arXiv 2509.26328 | 2025 | Block diffusion adapted from pretrained AR models with ~1B fine-tuning tokens (500x less than Dream); hierarchical (block-level + sub-block) cache; 2.5x over AR | Still requires fine-tuning; speedup moderate vs. pure DLM baselines |
| 10 | **DFlash: Block Diffusion for Flash Speculative Decoding** | arXiv 2602.06036 | 2026 | Block diffusion draft model for AR target model verification; >6x lossless speedup, 2.5x better than EAGLE-3 | Cross-architecture setup; draft model must be maintained separately |
| 11 | **Progressive Refinement Regulation (PRR)** | arXiv 2603.04514 | 2026 | Token-wise convergence controller using full decoding rollout trajectories; temperature-based distribution shaping; accelerates DLM decoding | Requires training a lightweight controller |
| 12 | **DualDiffusion: Speculative Decoding for Masked Diffusion Models** | arXiv 2604.05250 | 2026 | Draft (fast approximation) + verify (accurate verifier) framework for MDMs; better Pareto frontier on MMLU/GSM8K | Very recent; limited empirical scope |
| 13 | **MEDAL: Monte Carlo Tree Search for DLM Inference** | arXiv 2512.12168 | 2025 | MCTS at initialization stage to find optimal unmasking trajectories; 22% improvement over baselines; inference-time scaling | Not a latency reduction method; focuses on quality over speed |
| 14 | **Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing (D2F)** | arXiv 2508.09192 | 2025 | First open-source DLM to surpass AR throughput; block-wise sequential + KV cache; AR-diffusion hybrid | Requires distillation training |
| 15 | **Encoder-Decoder Diffusion Language Models (E2D2)** | arXiv 2510.22852 | 2025 | Encoder for clean tokens + lightweight decoder for denoising; reduces computation per step; better quality-throughput trade-off | Architecture change requires retraining |
| 16 | **LLaDA-MoE: A Sparse MoE Diffusion Language Model** | arXiv 2509.24389 | 2025 | First MoE-based DLM; 7B parameters, 1.4B active; trained on 20T tokens; SOTA among diffusion LMs | MoE routing overhead; no targeted inference acceleration beyond sparse activation |
| 17 | **TESS 2: A Large-Scale Generalist Diffusion Language Model** | arXiv 2502.13917 | 2025 | Instruction-following DLM matching AR models; reward guidance at inference time; improved with more inference compute | No specific latency optimization |
| 18 | **Advancing Block Diffusion LMs for Test-Time Scaling (BACD + TCCF)** | arXiv 2602.09555 | 2026 | Bounded Adaptive Confidence Decoding (BACD) + Think Coarse Critic Fine (TCCF) for CoT reasoning; 2.26x speedup + 11.2 points on AIME24 over baselines | Complex two-phase pipeline; specialized for reasoning tasks |
| 19 | **DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention** | arXiv 2603.08026 | 2026 | Training-free; identifies salient tokens via cosine similarity of attention contexts between adjacent steps; recomputes only salient tokens; up to 9.6x speedup on LLaDA and Dream | Saliency threshold requires tuning; less explored in long-context settings |
| 20 | **DARE: Diffusion LLM Alignment and Reinforcement Executor** | arXiv 2604.04215 | 2026 | Unified post-training + evaluation framework for dLLMs (built on verl + OpenCompass); supports SFT, PEFT, preference optimization, RL for MDLMs and BDLMs; covers LLaDA, Dream, SDAR, LLaDA2.x | Not an inference acceleration work; primarily a research infrastructure contribution |
| 21 | **Not All Denoising Steps Are Equal: Model Scheduling for Faster MDLMs** | arXiv 2604.02340 | 2026 | Architecture-agnostic model scheduling; replaces middle steps with a smaller model (sandwich schedule); up to 17% FLOP reduction with modest quality degradation; no retraining required | Only 17% FLOP reduction; does not directly reduce wall-clock latency as much as caching methods |
| 22 | **S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation** | arXiv 2603.25702 | 2026 | Self-speculative decoding for block-diffusion LMs; same model acts as drafter (block diffusion mode) and verifier (AR mode, block size=1); up to 4.7x speedup; evaluated on SDAR, Fast-dLLM v2, LLaDA2.1 | Requires block-diffusion architecture; not applicable to pure masked-diffusion models (LLaDA, Dream) |
| 23 | **d1: Scaling Reasoning in Diffusion LLMs via RL** | arXiv 2504.12216 (NeurIPS 2025) | 2025 | Masked SFT + diffu-GRPO (first policy gradient for masked dLLMs); nearly doubled planning task performance; emergent self-correction; best GSM8K among dLLMs | Not an inference efficiency work; focuses on reasoning quality; relevant for understanding dLLM capabilities |
| 24 | **Generative Frontiers: Why Evaluation Matters for DLMs** | arXiv 2604.02718 | 2026 | Methodological critique of DLM evaluation; proposes principled augmentations for reliable comparison; discusses limitations of single-metric comparisons | Meta-paper; no algorithmic contribution |
| 25 | **ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping** | arXiv 2603.10088 | 2026 | Training-free; skip early-layer computation for unimportant tokens using tensor variation + confidence scores across iterations; up to 5.6x-16.8x speedup; 226-309 TPS on H200 | Relies on iteration-to-iteration similarity; must be re-evaluated at different step counts |
| 26 | **Focus-dLLM: Accelerating Long-Context DLM Inference via Confidence-Guided Context Focusing** | arXiv 2602.02159 | 2026 | Training-free attention sparsification for long contexts; past-confidence-guided indicator predicts unmasked token positions for next step + sink-aware pruning; evaluated on LongBench at 16K context | Specialized for long-context; tested on UltraLLaDA; limited benchmark coverage |
| 27 | **Mercury: Ultra-Fast Language Models Based on Diffusion** | arXiv 2506.17298 | 2025 | First commercial-scale diffusion LLM; 5-10x faster than AR baselines; HumanEval 90.0% (Small), 25ms average latency in Copilot Arena; demonstrates viability of commercial DLM serving | Closed-source; no public model weights |
| 28 | **Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference** | arXiv 2508.02193 | 2025 | ByteDance DLM; achieves 2,146 tokens/s on H20 GPUs; new SOTA speed-quality Pareto for code; competitive with Mercury and Gemini Diffusion on code benchmarks | Closed-source; focused on code generation |
| 29 | **Prophet: Diffusion Language Models Know the Answer Before Decoding** | arXiv 2508.19982 | 2025 | Training-free early commit decoding; exploits early answer convergence -- correct answer identified by half steps before final decoding; up to 3.4x step reduction on LLaDA-8B and Dream-7B | Designed for tasks with identifiable answer regions (math, code, planning); not general-purpose text generation |
| 30 | **JoT (Just on Time): Token-Level Early Stopping for Diffusion Language Models** | arXiv 2602.11133 | 2026 | Training-free per-token early stopping; monitors prediction confidence at each position independently; adaptive threshold with spatial modulation; 7.2x speedup retaining 98.3% quality on Dream-7B and LLaDA-8B | Orthogonal to per-step latency methods (KV caching); threshold tuning required |
| 31 | **SchED: Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules** | arXiv 2512.02892 | 2025 | Training-free, model-agnostic early-exit; aggregates full-span logit margins with smooth progress-dependent threshold; 3.8-4.0x speedup retaining 99.8-100% quality on instruction-tuned models | Conservative speedup compared to caching methods; primarily reduces step count, not per-step cost |
| 32 | **SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models** | arXiv 2602.02544 | 2026 | Low-dimensional singular proxy for update-critical token identification; adaptive layer-wise budget allocation; up to 8x throughput over vanilla, 2-4x over existing cache baselines | Relatively modest improvement over best existing cache methods |
| 33 | **Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction** | arXiv 2508.02558 / AAAI 2026 | 2025 | First training-free framework integrating dynamic cache eviction with sparse attention; exploits persistent cross-layer sparsity; up to 10x throughput improvement on LLaDA and Dream | Attention patterns may differ across model families |
| 34 | **I-DLM: Introspective Diffusion Language Models** | arXiv 2604.11035 | 2026 | First DLM to match AR quality; introspective strided decoding (ISD) verifies tokens while advancing new ones; I-DLM-8B matches Qwen3-8B; 5,900 tok/s at concurrency=32 on H100 (3.8x over SDAR); only 4.5B training tokens | Requires training (converts Qwen3-8B); not a pure inference acceleration technique |
| 35 | **Soft-Masked Diffusion Language Models** | arXiv 2510.17206 / ICLR 2026 | 2025 | Dynamically blends mask token embedding with top-k predicted token embeddings; only 3 extra parameters; improves performance in high-throughput settings; compatible with Fast-dLLM caching | Requires fine-tuning; quality gains modest on non-coding tasks |
| 36 | **ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding** | arXiv 2512.13586 / ICLR 2026 | 2025 | First MDM with full KV cache reuse; slot-level plan-and-infill decoding; 34% performance gain over prior MDMs; 18x speedup over MDMs, 2.33x over ARMs | Requires training from AR checkpoint (Qwen3-8B); slot design constrains generation flexibility |
| 37 | **CDLM: Consistency Diffusion Language Models for Faster Sampling** | arXiv 2511.19269 / MLSys 2026 | 2025 | Consistency modeling + block-wise causal attention + KV cache compatibility; 3.6-14.5x lower latency; up to 20.9x throughput on MBPP-Instruct | Training-based (LoRA fine-tuning); bounded by teacher quality |
| 38 | **d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation** | arXiv 2601.07568 | 2026 | Pseudo-trajectory distillation teaches confident early-step decoding; entropy-based multi-block decoding + KV-cache refresh; 10x over vanilla LLaDA/Dream, 5x over AR; SGLang integration | Training-based; accuracy-parallelism trade-off inherent |
| 39 | **DAWN: Dependency-Aware Fast Inference for Diffusion LLMs** | arXiv 2602.06953 | 2026 | Training-free dependency-aware parallel decoding; extracts inter-token dependencies to guide more effective parallel strategies | Very recent; limited benchmark comparisons |
| 40 | **LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models** | arXiv 2604.12056 | 2026 | Exploits locality of representation changes; sparse prefix attention only for active tokens; up to 4.14x attention speedup on RTX A6000; +9 points accuracy at aggressive sparsity | Specific to block-wise DLMs; may not apply to pure MDMs |
| 41 | **dLLM-Serve: Taming the Memory Footprint Crisis for Production Diffusion LLM Serving** | arXiv 2512.17077 | 2026 | First production serving system for DLMs; logit-aware activation budgeting, phase-multiplexed scheduler, head-centric sparse attention; 1.6-1.8x throughput, 4x tail latency reduction | Evaluated only on LLaDA-8B; early prototype stage |
| 42 | **LLaDA 2.0: Scaling Up Diffusion Language Models to 100B** | arXiv 2512.15745 | 2025 | First 100B MoE DLM; AR-to-MDLM continual pre-training; block diffusion; LLaDA2.0-flash-CAP at 535 tok/s | Closed weights for 100B variant |
| 43 | **LLaDA 2.1: Speeding Up Text Diffusion via Token Editing** | arXiv 2602.08676 | 2026 | Token editing with confidence-based decoding; improved inference speed over LLaDA 2.0 | Block-diffusion architecture; different acceleration profile than pure MDMs |

---

## 3. SOTA Methods and Benchmarks

### Current Best Open-Source DLMs (Capability)

| Model | Scale | Training Data | Notable Benchmarks |
|-------|-------|--------------|-------------------|
| I-DLM-8B (April 2026) | 8B | 4.5B tokens (from Qwen3-8B) | Matches Qwen3-8B on ARC-C, IFEval, MMLU; 69.6 AIME-24; 45.7 LiveCodeBench-v6 |
| LLaDA 8B (NeurIPS 2025 oral) | 8B | 2.3T tokens | Competitive with LLaMA3 8B on 15 zero/few-shot tasks; SOTA on reversal poem task over GPT-4o |
| Dream 7B | 7B | -- | Substantial advantage over Qwen2.5 on planning tasks (Countdown, Sudoku, Trip planning) |
| LLaDA-MoE 7B-A1B | 7B (1.4B active) | 20T tokens | SOTA among DLMs; comparable to Qwen2.5-3B-Instruct with fewer active params |
| LLaDA 2.0-flash | 100B MoE | -- | First 100B DLM; 535 tok/s with parallel decoding |
| ReFusion-8B (ICLR 2026) | 8B | 1.22B tokens (from Qwen3-8B) | 34% gain over prior MDMs; 2.33x speedup over ARMs |
| TESS 2 | ~7B | -- | Matches/exceeds strong AR baselines; improves with inference-time compute |

### Acceleration Benchmarks (Inference Speed)

| Method | Base Model | Speedup | Quality | Training-free? | Key Technique |
|--------|-----------|---------|---------|---------------|---------------|
| Fast-dLLM (ICLR 2026) | LLaDA/Dream | Up to 27.6x | Negligible loss | Yes | Block-wise KV cache + confidence parallel decoding |
| FreeCache + Guided Diffusion (FlashDLM) | LLaDA-8B / Dream-7B | 12.14x-13.29x | Negligible loss | Yes | KV approximation + AR-guided unmasking |
| SlowFast + dLLM-Cache | LLaDA | Up to 34.22x | Marginal degradation | Yes | Dynamic 2-stage sampling + caching |
| Learn2PD + Dual Cache | -- | 57.51x throughput | -- | No | Learned parallel decoding |
| Elastic-Cache | LLaDA variants | 45.1x (long seq) | Preserved | Yes | Adaptive depth-aware cache refresh |
| Window-Diffusion | LLaDA / Dream | Up to 99x | Partially preserved | Yes | Windowed token pruning + sliding cache |
| EntropyCache | LLaDA-8B / Dream-7B | 15.2x-26.4x | Competitive | Yes | Entropy-guided KV refresh |
| ES-dLLM | LLaDA-8B / Dream-7B | 5.6x-16.8x; 1.85x over SOTA cache | Preserved | Yes | Early-layer skipping via tensor variation + confidence |
| DyLLM | LLaDA-8B / Dream-7B | Up to 9.6x | Largely preserved | Yes | Saliency-based token selection + partial attention |
| Sparse-dLLM (AAAI 2026) | LLaDA / Dream | Up to 10x | Comparable | Yes | Dynamic cache eviction + sparse attention |
| SPA-Cache | DLMs | Up to 8x; 2-4x over cache baselines | Preserved | Yes | Singular proxy adaptive caching |
| JoT | Dream-7B / LLaDA-8B | 7.2x (98.3% quality) | Near-lossless | Yes | Per-token early stopping |
| Prophet | LLaDA-8B / Dream-7B | Up to 3.4x step reduction | Preserved | Yes | Early answer convergence + commit decoding |
| SchED | Dream / LLaDA | 3.8-4.0x (99.8-100% quality) | Near-lossless | Yes | Progress-aware confidence early-exit |
| Saber (code) | DLM | 251.4% speedup | +1.9% Pass@1 | Yes | Adaptive acceleration + backtracking |
| S2D2 | SDAR / Fast-dLLM v2 / LLaDA2.1 | Up to 4.7x over AR | Improved accuracy | Yes | Self-speculative (block-diffusion + AR mode) |
| ReFusion (ICLR 2026) | Qwen3-8B adapted | 18x over MDMs; 2.33x over ARMs | 34% gain over MDMs | No (fine-tune) | Slot-level plan-and-infill + full KV cache |
| CDLM (MLSys 2026) | Dream-7B / LLaDA-8B | 3.6x-14.5x; up to 20.9x on MBPP | Competitive | No (LoRA fine-tune) | Consistency modeling + block-wise causal mask |
| d3LLM | LLaDA / Dream | 10x over vanilla DLM; 5x over AR | Competitive | No (distillation) | Pseudo-trajectory distillation + entropy multi-block decoding |
| I-DLM | Qwen3-8B adapted | 5,900 tok/s (3.8x over SDAR) | Matches Qwen3-8B | No (training) | Introspective strided decoding |
| D2F | -- | Faster than AR | Competitive | No (distillation) | Block-wise + AR KV cache hybrid |
| Fast-dLLM v2 | AR-adapted (7B) | 2.5x over AR | Matches AR baseline | No (fine-tune) | Hierarchical block/sub-block cache |
| DFlash | AR target model | >6x lossless | Lossless | No (block diffusion draft) | Block diffusion as speculative draft |
| Focus-dLLM | UltraLLaDA (16K) | Significant long-context speedup | Competitive on LongBench | Yes | Confidence-guided attention sparsification + sink-aware pruning |
| Seed Diffusion (ByteDance) | -- | 2,146 tokens/sec on H20 | Competitive (code) | Closed source | N/A |
| Mercury (Inception, closed) | -- | 5-10x over AR; 25ms p50 latency | HumanEval 90.0% | Closed source | N/A |
| Gemini Diffusion (Google, closed) | -- | ~1,479 tokens/sec | Matches Gemini 2.0 Flash-Lite on coding | Closed source | N/A |

### Mainstream Evaluation Benchmarks

- **General reasoning**: MMLU, HellaSwag, ARC-C/E, WinoGrande, PIQA
- **Math**: GSM8K, MATH500, AIME2024/2025, GPQA
- **Code**: HumanEval, MBPP, HumanEvalInfilling, McEval, LiveCodeBench
- **Planning**: Countdown, Sudoku, Trip Planning
- **Long-context / Summarization**: CNN/DailyMail (for FlashDLM evaluation), LongBench
- **Throughput metrics**: Tokens-per-second (TPS) on A100/H100/H20; latency-per-token; flops-per-token; AUP (Accuracy Under Parallelism, from d3LLM)

---

## 4. Identified Research Gaps

- **Gap 1 -- No unified DLM inference engine**: AR models have vLLM, TGI, TensorRT-LLM. DLMs lack a production-quality inference engine that combines KV caching, batching, and continuous decoding efficiently. dLLM-Serve and dInfer are early efforts. SGLang has begun adding d3LLM support, signaling growing ecosystem maturity.

- **Gap 2 -- Batched inference scaling gap**: AR models benefit strongly from batched inference (increased arithmetic intensity). Current DLM acceleration benchmarks focus almost entirely on single-sequence latency. Fair throughput comparisons under batched workloads show AR models still dominate for most open-source DLMs. dLLM-Serve's phase-multiplexed scheduler is a first step.

- **Gap 3 -- KV cache approximation quality-speed Pareto frontier**: Existing methods (Fast-dLLM, dKV-Cache, Elastic-Cache, EntropyCache, SPA-Cache, Sparse-dLLM) all take different approaches to deciding *when* and *where* to refresh the KV cache. There is no systematic comparative study of the quality-speed Pareto frontier of these methods under a unified evaluation protocol.

- **Gap 4 -- Speculative decoding for full-attention DLMs**: Partially addressed by SSD (arXiv:2510.04147) and SSMD (arXiv:2510.03929). DualDiffusion, DFlash, and S2D2 apply speculative ideas but target block-diffusion models. Self-speculative approaches for pure MDMs (LLaDA, Dream) remain less explored. Our IGSD/CD-SSD variant targets this remaining sub-gap.

- **Gap 5 -- Adaptive denoising schedules**: PRR, Saber, Prophet, JoT, and SchED each address different aspects of non-uniform step/token scheduling. However, a principled information-theoretic framework for optimal step scheduling that minimizes total compute while preserving quality is still nascent.

- **Gap 6 -- Variable-length generation efficiency**: DLMs require a fixed-length canvas, causing overallocation waste. LR-DLLM addresses quality in this setting, but there is no solution that jointly optimizes variable-length generation quality *and* inference speed.

- **Gap 7 -- Cross-model transferability of acceleration methods**: Most acceleration works are evaluated on LLaDA and Dream specifically. It is unclear how well techniques generalize across different DLM architectures (encoder-decoder, MoE, adapted-from-AR, trained-from-scratch, block-diffusion vs. pure MDM).

- **Gap 8 -- Hardware-aware DLM optimization**: No work systematically analyzes roofline models for DLM inference on modern accelerators (H100/H200/B200). dLLM-Serve identifies the Refresh/Reuse phase oscillation but does not provide a full roofline characterization.

- **Gap 9 -- Model scheduling under strict latency budgets**: "Not All Denoising Steps Are Equal" (arXiv 2604.02340) shows only ~17% FLOP reduction by substituting a smaller model at less-critical steps. More aggressive cascade scheduling is unexplored.

- **Gap 10 -- Saliency estimation overhead**: DyLLM, ES-dLLM, and Focus-dLLM each use different proxy signals (cosine similarity, tensor variation, confidence). A rigorous comparison of proxy signal accuracy vs. compute cost is needed.

- **Gap 11 -- Composition of acceleration methods**: Most papers test one technique in isolation. **This is the gap ComposeAccel directly addresses.** Since the previous survey, no new paper has appeared that performs systematic cross-family composability analysis. JoT explicitly claims orthogonality to KV-caching methods but does not empirically validate combinations. SlowFast + dLLM-Cache is the only published composition but covers a single pair within the same family.

- **Gap 12 -- Long-context DLM inference efficiency**: Focus-dLLM and LoSA are the first papers to specifically address long-context DLM inference. Systematic study of how acceleration techniques interact with long-context settings is largely missing.

- **Gap 13 -- Early-layer token importance estimation cost**: ES-dLLM and DyLLM both estimate token importance across iterations but with different proxy signals. A rigorous comparison is needed.

- **Gap 14 -- Variable-length generation with acceleration**: Most acceleration papers assume fixed-length generation. Combining acceleration techniques with variable-length generation remains underexplored.

- **Gap 15 (NEW) -- Transition to hybrid AR-diffusion architectures**: I-DLM, ReFusion, LLaDA 2.0/2.1, and SDAR all use hybrid architectures with native KV cache support. Acceleration methods designed for pure MDMs (EntropyCache, DyLLM, ES-dLLM) may not transfer directly. The composability landscape for hybrid architectures is entirely unstudied.

- **Gap 16 (NEW) -- Introspective consistency and acceleration interaction**: I-DLM shows that enforcing introspective consistency during training dramatically improves quality. Whether acceleration methods (KV caching, early stopping) preserve or damage introspective consistency is unknown.

---

## 5. Available Resources

### Open-Source Code

| Repository | Description | License |
|-----------|-------------|---------|
| [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | Official NVIDIA implementation of Fast-dLLM; supports LLaDA and Dream; block-wise KV cache + parallel decoding | Check repo |
| [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | Official LLaDA 8B + LLaDA-MoE pretrained models and inference code | Apache 2.0 |
| [ML-GSAI/ReFusion](https://github.com/ML-GSAI/ReFusion) | ReFusion: parallel AR decoding MDM with full KV cache reuse (ICLR 2026) | Check repo |
| [Introspective-Diffusion/I-DLM](https://github.com/Introspective-Diffusion/I-DLM) | I-DLM: first DLM to match AR quality; introspective strided decoding | Check repo |
| [SqueezeAILab/CDLM](https://github.com/SqueezeAILab/CDLM) | CDLM consistency distillation for DLMs (MLSys 2026) | Check repo |
| [hao-ai-lab/d3LLM](https://github.com/hao-ai-lab/d3LLM) | d3LLM ultra-fast DLM via pseudo-trajectory distillation; SGLang integration | Check repo |
| [IBM/soft-masked-diffusion-language-models](https://github.com/IBM/soft-masked-diffusion-language-models) | Soft-Masked DLMs (ICLR 2026); built on Dream-7B | Check repo |
| [pixeli99/Prophet](https://github.com/pixeli99/Prophet) | Prophet early answer convergence + commit decoding for DLMs | Check repo |
| [Anonym-cybersudo/JoT](https://github.com/Anonym-cybersudo/JoT) | JoT per-token early stopping for DLMs | Check repo |
| [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) | DiffuGPT + DiffuLLaMA -- AR-to-diffusion adaptation (ICLR 2025) | Check repo |
| [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) | Simple DLM framework; integrates Fast-dLLM cache/parallel decoding; diffu-GRPO training | Check repo |
| [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | Modular DLM inference framework: model, diffusion manager, decoder, KV-cache manager | Check repo |
| [inclusionAI/LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X) | LLaDA 2.0/2.1 series from Ant Group | Check repo |
| [pengzhangzhi/Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) | Full pipeline: data -> training -> evaluation -> inference for code-focused DLMs | Check repo |
| [VILA-Lab/Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) | Survey + curated list of DLM papers, code, and models | N/A |
| [mscheong01/EntropyCache](https://github.com/mscheong01/EntropyCache) | EntropyCache training-free KV caching for LLaDA-8B-Instruct and Dream-7B-Instruct | Check repo |
| [m-arriola.com/e2d2](https://m-arriola.com/e2d2) | E2D2 encoder-decoder diffusion; code + model weights | Check repo |
| [m-arriola.com/bd3lms](https://m-arriola.com/bd3lms) | Block Diffusion (BD3-LMs); code + model weights + blog | Check repo |
| [vhicrgit/Window-Diffusion](https://github.com/vhicrgit/Window-Diffusion) | Window-Diffusion windowed token pruning + caching | Check repo |
| [hamishivi/tess-2](https://github.com/hamishivi/tess-2) | TESS 2 generalist instruction-following DLM | Check repo |
| [phymhan/S2D2](https://github.com/phymhan/S2D2) | S2D2 self-speculative decoding for block-diffusion LMs | Check repo |
| [LiangrunFlora/Slow-Fast-Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling) | SlowFast Sampling official PyTorch implementation; drop-in for LLaDA-8B and Dream-7B | Check repo |
| [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | d1 masked SFT + diffu-GRPO RL for LLaDA; reasoning benchmarks | Check repo |
| [bansky-cl/diffusion-nlp-paper-arxiv](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv) | Auto-tracked arxiv papers on diffusion NLP; useful for staying current | N/A |

### Pretrained Models (HuggingFace)

| Model | HF Hub ID | Size |
|-------|-----------|------|
| LLaDA 8B Base/Instruct | `GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-8B-Instruct` | 8B |
| LLaDA 1.5 | `GSAI-ML/LLaDA-1.5` | -- |
| LLaDA-MoE 7B-A1B | Huggingface (see paper) | 7B / 1.4B active |
| Dream 7B Instruct | `hkunlp/dream-7b-instruct` | 7B |
| TESS 2 | `hamishivi/tess-2` | ~7B |
| d3LLM-LLaDA / d3LLM-Dream / d3LLM-Dream-Coder | hao-ai-lab (HF) | 8B / 7B |
| I-DLM-8B | Introspective-Diffusion (HF) | 8B |

### Evaluation Frameworks

- **DARE** (arXiv 2604.04215): Unified post-training + eval framework for dLLMs built on verl + OpenCompass; covers MDLMs and BDLMs; supports LLaDA, Dream, SDAR, LLaDA2.x; MMLU, GSM8K, MATH, GPQA, AIME, HumanEval, MBPP, planning tasks
- **dLLM framework** (ZHZisZZ): Reproduces official scores for LLaDA/Dream; standardized training + inference + evaluation pipeline; now supports diffu-GRPO
- **lm-evaluation-harness**: Compatible with DLMs via HuggingFace integration
- **"How Efficient Are Diffusion LMs?"** (arXiv 2510.18480): Critical evaluation practices paper; proposes Tokens-Per-Forward-Step (TPF) as unified efficiency metric; empirical benchmarking under batched settings
- **AUP metric** (from d3LLM): Accuracy Under Parallelism; jointly measures accuracy and parallelism for DLM evaluation

---

## 6. Implications for Idea Generation

**Directions worth exploring:**

1. **Unified quality-speed evaluation framework for DLM KV caching**: Multiple competing KV caching approaches (dKV-Cache, Fast-dLLM, Elastic-Cache, EntropyCache, FlashDLM-FreeCache, Window-Diffusion, SPA-Cache, Sparse-dLLM) exist with incompatible evaluation setups. A rigorous systematic comparison + a principled design space analysis (cache granularity x refresh policy x task type x model architecture) would be high-impact and publishable.

2. **Training-free speculative decoding within MDMs**: SSD and SSMD have partially filled this gap. Our IGSD/CD-SSD variant offers a different tradeoff (approximate coarse-step draft vs. lossless hierarchical verification). The key differentiator is composability: IGSD's frozen-token REFINE phase creates uniquely favorable conditions for KV-caching synergy not tested with SSD.

3. **Confidence-calibrated adaptive step scheduling**: Prophet, JoT, and SchED each address early stopping but from different angles (answer-level convergence, token-level exit, progress-aware schedule). An information-theoretic framework unifying these approaches could yield a principled algorithm.

4. **Batch-level and hardware-aware profiling of DLM inference**: dLLM-Serve identifies the Refresh/Reuse phase oscillation. A full roofline analysis on H100/A100 would identify where algorithmic work can have the most impact.

5. **Saliency-driven partial attention with lightweight proxy signals**: DyLLM, ES-dLLM, Focus-dLLM, and LoSA all use different proxy signals for token importance. An entropy-based or gradient-free proxy signal (O(V) like EntropyCache) to decide which tokens need full attention vs. cached attention could yield a unified framework.

6. **Composition study of multiple training-free techniques (ComposeAccel)**: No existing work systematically evaluates cross-family composability. JoT explicitly claims orthogonality to KV caching but does not validate. SlowFast + dLLM-Cache is the only published two-method composition. Our pairwise composability atlas (M1+IGSD synergy, failure modes) remains fully novel.

7. **Acceleration for hybrid AR-diffusion architectures**: I-DLM, ReFusion, LLaDA 2.0/2.1 represent the next generation. Acceleration methods designed for pure MDMs may not transfer. Composability analysis for this new architecture class is entirely open.

8. **Self-speculative decoding for pure MDLMs (non-block-diffusion)**: S2D2 works elegantly for block-diffusion models. An analogous mechanism for pure MDMs (LLaDA, Dream) that doesn't require a separate draft model -- e.g., drafting with fewer denoising steps and verifying with more -- is our IGSD approach.

**Saturated / crowded directions:**

- Simple confidence-threshold sampling -- many variants already exist (JoT, Prophet, SchED, Fast-dLLM); marginal gains likely.
- Training-based improvements to DLM quality without addressing inference speed -- already well-covered by LLaDA, Dream, TESS 2, LLaDA-MoE, I-DLM.
- Basic KV cache reuse strategies (uniform refresh) -- multiple approaches already published (Fast-dLLM, dKV-Cache, Elastic-Cache, dLLM-Cache, EntropyCache, SPA-Cache, Sparse-dLLM); extremely crowded.
- Single-method step reduction via early stopping -- Prophet, JoT, SchED all cover this space with diminishing marginal novelty.

**Cross-domain analogies with potential:**

- *Speculative decoding* (AR world) -> drafting within the denoising process using a fast approximate model and accepting with a probability correction (MDM-specific acceptance criterion needed).
- *Flash Attention* memory efficiency tricks -> adapted to the bidirectional full-sequence attention of DLMs; SageAttention 8-bit attention is broadly applicable.
- *Mixture of Experts sparse activation* -> LLaDA-MoE demonstrates this is viable; combining MoE sparsity with token-adaptive denoising could compound efficiency gains.
- *Consistency/Distillation models* (image diffusion) -> CDLM demonstrates this works for DLMs; extending to larger scales and pure MDMs is the next step.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| Fast-dLLM (NVlabs/Fast-dLLM) | High | Check | **Adopt/Extend** | Best-in-class training-free KV cache + parallel decoding; supports LLaDA/Dream; ICLR 2026; actively maintained by NVIDIA |
| LLaDA 8B / Dream 7B (HF models) | High | Apache 2.0 / Check | **Adopt** | Standard benchmarking baselines; inference infrastructure well-established |
| EntropyCache (mscheong01) | High | Check | **Extend** | Our M1 implementation is based on this; clean entropy-guided refresh; straightforward to swap in alternative refresh signals |
| dInfer (inclusionAI/dInfer) | Medium | Check | **Extend** | Modular framework with KV-cache manager abstraction; good base for implementing new caching policies |
| ZHZisZZ/dllm | Medium | Check | **Adopt** | Unified inference + eval pipeline; fastest way to reproduce baselines and add new acceleration modules |
| Block Diffusion / E2D2 (m-arriola) | Medium | Check | **Extend** | Reusable block diffusion infrastructure; useful if exploring encoder-decoder or AR-hybrid acceleration |
| lm-evaluation-harness | High | MIT | **Adopt** | Industry-standard evaluation; ensures fair comparison with prior work |
| DARE framework (arXiv 2604.04215) | High | Check | **Adopt** | Unified eval + post-training across LLaDA/Dream/SDAR/LLaDA2.x; most comprehensive dLLM research substrate |
| S2D2 (phymhan/S2D2) | Medium | Check | **Extend** | Clean self-speculative baseline for block-diffusion; useful reference for our IGSD MDM analog |
| SlowFast Sampling (LiangrunFlora) | High | Check | **Extend** | Best training-free sampling baseline with open code; drop-in for LLaDA/Dream |
| Prophet (pixeli99/Prophet) | Medium | Check | **Extend** | Training-free early stopping baseline; potential composition candidate for ComposeAccel |
| JoT (Anonym-cybersudo/JoT) | Medium | Check | **Extend** | Per-token early stopping; explicitly claims orthogonality to KV caching -- composability test candidate |
| d3LLM (hao-ai-lab/d3LLM) | Low | Check | **Reference** | Training-based; different paradigm but SGLang integration useful for serving context |
| dLLM-Serve | Medium | Check | **Reference** | Production serving system design; useful for framing practical impact of composability results |

**Reusable evaluation frameworks**: Fast-dLLM's benchmarking scripts, DARE, lm-evaluation-harness, AUP metric from d3LLM.

**Reusable data pipelines**: LLaDA and Dream both use standard open instruction-tuning datasets; the dLLM framework standardizes loading.

**Reusable pretrained checkpoints**: `GSAI-ML/LLaDA-8B-Instruct` and `hkunlp/dream-7b-instruct` are the standard test models; always use both for reproducibility.

---

## 8. Relevance to ComposeAccel (Iteration 3 Specific)

### New papers since iteration 2 survey (April 14) that are directly relevant:

1. **I-DLM (arXiv 2604.11035, April 13)**: The first DLM to match AR quality. Uses introspective strided decoding. Relevant because: (a) demonstrates hybrid AR-diffusion architecture is the future direction, (b) raises the question of whether ComposeAccel's composability findings for pure MDMs transfer to hybrid architectures, (c) sets a new quality ceiling for DLMs that reframes the speed-quality tradeoff discussion.

2. **LoSA (arXiv 2604.12056, April 13)**: Locality-aware sparse attention for block-wise DLMs. Relevant because: (a) represents a new acceleration family (sparse attention) not covered in ComposeAccel's original 3-family design, (b) exploits the same "stable vs. active token" dichotomy that drives IGSD's frozen-token mechanism.

3. **CDLM (arXiv 2511.19269, accepted MLSys 2026)**: Consistency distillation for DLMs with KV cache compatibility. Relevant because: (a) this is exactly our backup candidate cand_b, but CDLM has been published and accepted, (b) confirms that consistency distillation for DLMs works, (c) CDLM is training-based (LoRA), which differentiates it from our training-free focus.

4. **ReFusion (arXiv 2512.13586, ICLR 2026)**: First MDM with full KV cache reuse. Relevant because: (a) slot-level design fundamentally changes the KV caching landscape for DLMs, (b) composability of acceleration methods on ReFusion-type architectures is an entirely new question.

### Confirmed positioning of ComposeAccel:

The survey confirms that **Gap 11 (composition of acceleration methods) remains fully open** as of April 15, 2026. No paper published since our previous survey performs systematic cross-family composability analysis. The closest is JoT's claim of orthogonality to KV caching, which is stated but not empirically validated in composition. SlowFast + dLLM-Cache is the only published two-method composition, and it covers a single pair within the same method family.

The ComposeAccel contributions remain novel:
- **C1 (composability atlas)**: 9/10 novelty -- no competitor
- **C2 (IGSD/CD-SSD)**: 4/10 standalone; value is as composability analysis vehicle (SSD, SSMD are prior art for the concept)
- **C3 (failure mode atlas)**: 9/10 novelty -- no competitor

### New potential composition candidates identified:

- **JoT + M1 (EntropyCache)**: JoT claims orthogonality to KV caching. If validated, this would be a new composability data point for the atlas.
- **Prophet + M1**: Prophet's early commit decoding could reduce the number of steps where KV caching needs to be applied.
- **LoSA + M1**: LoSA targets sparse attention within blocks; EntropyCache targets KV refresh across steps. These operate on orthogonal dimensions.

---

*Survey completed by sibyl-literature agent. All speedup numbers are as reported in the respective papers; hardware configurations vary -- see individual papers for experimental details.*
