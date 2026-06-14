

## Project Spec
# 项目: dlm-acceleration

## 研究主题
基于 DLM（Diffusion Language Model）的推理加速

## 背景与动机
Diffusion Language Models（如 MDLM、SEDD 等）通过多步去噪过程生成文本，具有天然的并行生成能力，但推理速度相较自回归模型仍有差距。如何在不重新训练模型的前提下，利用 DLM 的并行性和独特的扩散过程特性加速推理，是一个重要的开放问题。

## 初始想法

**核心方向**: Systematic comparison of training-free acceleration methods for diffusion language models — evaluate orthogonality and composability of KV caching, adaptive step scheduling, AR-guided unmasking, and speculative decoding on LLaDA-8B-Instruct across reasoning and coding benchmarks.

**四种方法**:
1. **KV Caching**: 复用相邻扩散步之间相似的 KV 对，减少重复注意力计算
2. **Adaptive Step Scheduling**: 动态调整扩散步数，高置信 token 早确定，低置信 token 多步精化
3. **AR-guided Unmasking**: 用自回归模型指导 mask token 的解码顺序或候选生成
4. **Speculative Decoding**: 用小模型草稿 + DLM 大模型并行验证，利用 DLM 天然并行性加速

**核心研究问题**:
- 各方法单独能带来多少加速（speed-accuracy tradeoff）？
- 哪些方法可以正交组合（互不干扰地叠加加速）？
- 组合后的上限加速比是多少？
- 在推理（reasoning）和代码（coding）任务上表现是否一致？

**目标模型**: LLaDA-8B-Instruct

## 关键参考文献
- LLaDA (Large Language Diffusion with mAsking): 目标测试模型
- MDLM (arxiv: 2406.07524): Masked Diffusion Language Model
- SEDD: Score Entropy Discrete Diffusion
- Speculative Decoding: SpecInfer, Fast and Slow 等
- 系统将自动搜索相关文献

## 可用资源
- GPU: 本地多张 GPU
- 服务器: default (local)
- 远程路径: /home/qhxie/sibyl_system

## 实验约束
- 实验类型: training-free / 轻量实验（无大规模训练）
- 模型规模: 小到中等（优先使用开源 DLM 检查点，如 MDLM-base 等）
- 时间预算: 单实验约 1 小时以内；Pilot 10-15 分钟
- 优先验证 training-free 方法

## 目标产出
- 完整学术论文（NeurIPS/ICML/ICLR 水平）

## 特殊需求
- 所有方法必须 training-free（不修改模型权重）
- 评测基准覆盖推理（reasoning）和代码（coding）两类任务
- 必须系统评估方法之间的正交性和可组合性（不只是单独评估各方法）
- LLaDA-8B-Instruct 作为主要测试模型


## 文献调研报告（请仔细阅读，避免重复已有工作）
# Literature Survey Report

**Research Topic**: DLM (Diffusion Language Model) Inference Acceleration
**Survey Date**: 2026-04-14 (updated)
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

---

## 1. Field Overview

Diffusion Language Models (DLMs), also called discrete diffusion models or masked diffusion models (MDMs), have emerged as a compelling alternative to autoregressive (AR) language models. Unlike AR models that generate tokens sequentially left-to-right, DLMs operate by iteratively denoising a masked or noised sequence, enabling bidirectional context modeling and parallel token sampling. Key open-source representatives include **LLaDA 8B** (NeurIPS 2025 oral, ML-GSAI), **Dream 7B** (HKU NLP), **TESS 2** (UW/Allen AI), and **LLaDA-MoE** (7B with only 1.4B active parameters). Closed-source DLMs such as **Gemini Diffusion** (Google DeepMind) and **Mercury** (Inception) have demonstrated throughputs exceeding 1,400 tokens/second, 5–10× faster than comparably-sized AR models.

Despite architectural advantages, the open-source DLM community faces a fundamental efficiency paradox: in practice, most open-source DLMs (LLaDA, Dream) are **slower** than equivalently-sized AR LLMs. The root causes are: (1) bidirectional attention incompatibility with standard KV caching — each denoising step requires a full $O(N^2)$ forward pass over the entire sequence; (2) quality degradation when multiple tokens are decoded in parallel per step, forcing near-sequential unmasking; (3) fixed-length generation canvas that inflates unnecessary computation; and (4) absence of direct equivalents to AR acceleration techniques like speculative decoding, prefix caching, and chunked prefill.

As of early 2026, the field has produced a rich set of training-free and training-based acceleration techniques — KV approximation caching, sampling step reduction, speculative decoding adaptations, block diffusion, and hybrid AR-diffusion architectures — that are rapidly closing the inference gap. The field has moved from "can DLMs match AR quality?" (answered affirmatively in 2025) to "can DLMs exceed AR inference throughput in open-source settings?" (an active frontier in 2026).

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | **FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion** | arXiv 2505.21467 | 2025 | FreeCache (training-free KV approximation) + Guided Diffusion (lightweight AR supervisor); 12.14× speedup | AR guidance model adds memory; quality drop at very few steps |
| 2 | **Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding** | NVlabs / ICLR 2026 | 2025 | Approximate KV cache for block-wise decoding + confidence-aware parallel decoding; up to 27.6× speedup | Requires block-wise generation regime; not applicable to all pretrained DLMs |
| 3 | **dKV-Cache: The Cache for Diffusion Language Models** | arXiv 2505.15781 | 2025 | Delayed and conditioned KV caching; two variants — near-lossless (dKV-Cache-Decode) and aggressive (dKV-Cache-Greedy); 2–10× speedup | Greedy variant degrades quality; training-free but needs careful calibration |
| 4 | **Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking** | arXiv 2510.18165 | 2025 | Training-free: adaptive acceleration (more tokens per step as context builds) + backtracking mechanism; 251.4% speedup on code generation | Evaluated primarily on code; backtracking adds complexity |
| 5 | **Window-Diffusion: Accelerating DLM Inference with Windowed Token Pruning and Caching** | arXiv 2601.20332 | 2026 | Token-level analysis revealing structural locality; sliding window with active tokens, buffer tokens (cached KV), and pruned far-field tokens; up to 99× speedup | May sacrifice global context; evaluated mainly on LLaDA and Dream |
| 6 | **Elastic-Cache: Attention-aware Adaptive KV Refresh for Diffusion LLMs** | arXiv 2510.14973 | 2025 | Adaptive per-layer refresh schedule; drift-aware cache update via most-attended token signal; 8.7× on short, 45.1× on long sequences | Architecture-specific tuning may be needed |
| 7 | **EntropyCache: Decoded Token Entropy-Guided KV Caching for Diffusion LLMs** | arXiv 2603.18489 | 2026 | O(V) constant-cost refresh decision using decoded token entropy; avoids scaling overhead with context length; 15–26× speedup | Entropy signal may not generalize across all task types |
| 8 | **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models** | arXiv 2503.09573 | 2025 | Hybrid AR-diffusion: autoregressive across blocks, parallel within blocks; enables KV caching and flexible-length generation; SOTA on LM benchmarks | Requires retraining from scratch or fine-tuning |
| 9 | **Fast-dLLM v2: Efficient Block-Diffusion LLM** | arXiv 2509.26328 | 2025 | Block diffusion adapted from pretrained AR models with ~1B fine-tuning tokens (500× less than Dream); hierarchical (block-level + sub-block) cache; 2.5× over AR | Still requires fine-tuning; speedup moderate vs. pure DLM baselines |
| 10 | **DFlash: Block Diffusion for Flash Speculative Decoding** | arXiv 2602.06036 | 2026 | Block diffusion draft model for AR target model verification; >6× lossless speedup, 2.5× better than EAGLE-3 | Cross-architecture setup; draft model must be maintained separately |
| 11 | **Progressive Refinement Regulation (PRR)** | arXiv 2603.04514 | 2026 | Token-wise convergence controller using full decoding rollout trajectories; temperature-based distribution shaping; accelerates DLM decoding | Requires training a lightweight controller |
| 12 | **DualDiffusion: Speculative Decoding for Masked Diffusion Models** | arXiv 2604.05250 | 2026 | Draft (fast approximation) + verify (accurate verifier) framework for MDMs; better Pareto frontier on MMLU/GSM8K | Very recent; limited empirical scope |
| 13 | **MEDAL: Monte Carlo Tree Search for DLM Inference** | arXiv 2512.12168 | 2025 | MCTS at initialization stage to find optimal unmasking trajectories; 22% improvement over baselines; inference-time scaling | Not a latency reduction method; focuses on quality over speed |
| 14 | **Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing (D2F)** | arXiv 2508.09192 | 2025 | First open-source DLM to surpass AR throughput; block-wise sequential + KV cache; AR-diffusion hybrid | Requires distillation training |
| 15 | **Encoder-Decoder Diffusion Language Models (E2D2)** | arXiv 2510.22852 | 2025 | Encoder for clean tokens + lightweight decoder for denoising; reduces computation per step; better quality-throughput trade-off | Architecture change requires retraining |
| 16 | **LLaDA-MoE: A Sparse MoE Diffusion Language Model** | arXiv 2509.24389 | 2025 | First MoE-based DLM; 7B parameters, 1.4B active; trained on 20T tokens; SOTA among diffusion LMs | MoE routing overhead; no targeted inference acceleration beyond sparse activation |
| 17 | **TESS 2: A Large-Scale Generalist Diffusion Language Model** | arXiv 2502.13917 | 2025 | Instruction-following DLM matching AR models; reward guidance at inference time; improved with more inference compute | No specific latency optimization |
| 18 | **Advancing Block Diffusion LMs for Test-Time Scaling (BACD + TCCF)** | arXiv 2602.09555 | 2026 | Bounded Adaptive Confidence Decoding (BACD) + Think Coarse Critic Fine (TCCF) for CoT reasoning; 2.26× speedup + 11.2 points on AIME24 over baselines | Complex two-phase pipeline; specialized for reasoning tasks |
| 19 | **DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention** | arXiv 2603.08026 | 2026 | Training-free; identifies salient tokens via cosine similarity of attention contexts between adjacent steps; recomputes only salient tokens; up to 9.6× speedup on LLaDA and Dream | Saliency threshold requires tuning; less explored in long-context settings |
| 20 | **DARE: Diffusion LLM Alignment and Reinforcement Executor** | arXiv 2604.04215 | 2026 | Unified post-training + evaluation framework for dLLMs (built on verl + OpenCompass); supports SFT, PEFT, preference optimization, RL for MDLMs and BDLMs; covers LLaDA, Dream, SDAR, LLaDA2.x | Not an inference acceleration work; primarily a research infrastructure contribution |
| 21 | **Not All Denoising Steps Are Equal: Model Scheduling for Faster MDLMs** | arXiv 2604.02340 | 2026 | Architecture-agnostic model scheduling; replaces middle steps with a smaller model (sandwich schedule); up to 17% FLOP reduction with modest quality degradation; no retraining required | Only 17% FLOP reduction; does not directly reduce wall-clock latency as much as caching methods |
| 22 | **S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation** | arXiv 2603.25702 | 2026 | Self-speculative decoding for block-diffusion LMs; same model acts as drafter (block diffusion mode) and verifier (AR mode, block size=1); up to 4.7× speedup; evaluated on SDAR, Fast-dLLM v2, LLaDA2.1 | Requires block-diffusion architecture; not applicable to pure masked-diffusion models (LLaDA, Dream) |
| 23 | **d1: Scaling Reasoning in Diffusion LLMs via RL** | arXiv 2504.12216 (NeurIPS 2025) | 2025 | Masked SFT + diffu-GRPO (first policy gradient for masked dLLMs); nearly doubled planning task performance; emergent self-correction; best GSM8K among dLLMs | Not an inference efficiency work; focuses on reasoning quality; relevant for understanding dLLM capabilities |
| 24 | **Generative Frontiers: Why Evaluation Matters for DLMs** | arXiv 2604.02718 | 2026 | Methodological critique of DLM evaluation; proposes principled augmentations for reliable comparison; discusses limitations of single-metric comparisons | Meta-paper; no algorithmic contribution |
| 25 | **ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping** | arXiv 2603.10088 | 2026 | Training-free; skip early-layer computation for unimportant tokens using tensor variation + confidence scores across iterations; up to 5.6×–16.8× speedup; 226–309 TPS on H200 | Relies on iteration-to-iteration similarity; must be re-evaluated at different step counts |
| 26 | **Focus-dLLM: Accelerating Long-Context DLM Inference via Confidence-Guided Context Focusing** | arXiv 2602.02159 | 2026 | Training-free attention sparsification for long contexts; past-confidence-guided indicator predicts unmasked token positions for next step + sink-aware pruning; evaluated on LongBench at 16K context | Specialized for long-context; tested on UltraLLaDA; limited benchmark coverage |
| 27 | **Mercury: Ultra-Fast Language Models Based on Diffusion** | arXiv 2506.17298 | 2025 | First commercial-scale diffusion LLM; 5–10× faster than AR baselines; HumanEval 90.0% (Small), 25ms average latency in Copilot Arena; demonstrates viability of commercial DLM serving | Closed-source; no public model weights |
| 28 | **Seed Diffusion: A Large-Scale Diffusion Language Model with High-Speed Inference** | arXiv 2508.02193 | 2025 | ByteDance DLM; achieves 2,146 tokens/s on H20 GPUs; new SOTA speed-quality Pareto for code; competitive with Mercury and Gemini Diffusion on code benchmarks | Closed-source; focused on code generation |

---

## 3. SOTA Methods and Benchmarks

### Current Best Open-Source DLMs (Capability)

| Model | Scale | Training Data | Notable Benchmarks |
|-------|-------|--------------|-------------------|
| LLaDA 8B (NeurIPS 2025 oral) | 8B | 2.3T tokens | Competitive with LLaMA3 8B on 15 zero/few-shot tasks; SOTA on reversal poem task over GPT-4o |
| Dream 7B | 7B | — | Substantial advantage over Qwen2.5 on planning tasks (Countdown, Sudoku, Trip planning) |
| LLaDA-MoE 7B-A1B | 7B (1.4B active) | 20T tokens | SOTA among DLMs; comparable to Qwen2.5-3B-Instruct with fewer active params |
| TESS 2 | ~7B | — | Matches/exceeds strong AR baselines; improves with inference-time compute |

### Acceleration Benchmarks (Inference Speed)

| Method | Base Model | Speedup | Quality | Training-free? | Key Technique |
|--------|-----------|---------|---------|---------------|---------------|
| Fast-dLLM (ICLR 2026) | LLaDA/Dream | Up to 27.6× | Negligible loss | Yes | Block-wise KV cache + confidence parallel decoding |
| FreeCache + Guided Diffusion (FlashDLM) | LLaDA-8B / Dream-7B | 12.14×–13.29× | Negligible loss | Yes | KV approximation + AR-guided unmasking |
| SlowFast + dLLM-Cache | LLaDA | Up to 34.22× | Marginal degradation | Yes | Dynamic 2-stage sampling + caching |
| Learn2PD + Dual Cache | — | 57.51× throughput | — | No | Learned parallel decoding |
| Elastic-Cache | LLaDA variants | 45.1× (long seq) | Preserved | Yes | Adaptive depth-aware cache refresh |
| Window-Diffusion | LLaDA / Dream | Up to 99× | Partially preserved | Yes | Windowed token pruning + sliding cache |
| D2F | — | Faster than AR | Competitive | No (distillation) | Block-wise + AR KV cache hybrid |
| Fast-dLLM v2 | AR-adapted (7B) | 2.5× over AR | Matches AR baseline | No (fine-tune) | Hierarchical block/sub-block cache |
| DFlash | AR target model | >6× lossless | Lossless | No (block diffusion draft) | Block diffusion as speculative draft |
| EntropyCache | LLaDA-8B / Dream-7B | 15.2×–26.4× | Competitive | Yes | Entropy-guided KV refresh |
| Saber (code) | DLM | 251.4% speedup | +1.9% Pass@1 | Yes | Adaptive acceleration + backtracking |
| DyLLM | LLaDA-8B / Dream-7B | Up to 9.6× | Largely preserved | Yes | Saliency-based token selection + partial attention |
| ES-dLLM | LLaDA-8B / Dream-7B | 5.6×–16.8×; 1.85× over SOTA cache | Preserved | Yes | Early-layer skipping via tensor variation + confidence |
| Focus-dLLM | UltraLLaDA (16K) | Significant long-context speedup | Competitive on LongBench | Yes | Confidence-guided attention sparsification + sink-aware pruning |
| S2D2 | SDAR / Fast-dLLM v2 / LLaDA2.1 | Up to 4.7× over AR | Improved accuracy | Yes | Self-speculative (block-diffusion + AR mode) |
| Seed Diffusion (ByteDance) | — | 2,146 tokens/sec on H20 | Competitive (code) | Closed source | N/A |
| Mercury (Inception, closed) | — | 5–10× over AR; 25ms p50 latency | HumanEval 90.0% | Closed source | N/A |
| Gemini Diffusion (Google, closed) | — | ~1,479 tokens/sec | Matches Gemini 2.0 Flash-Lite on coding | Closed source | N/A |

### Mainstream Evaluation Benchmarks

- **General reasoning**: MMLU, HellaSwag, ARC-C/E, WinoGrande, PIQA
- **Math**: GSM8K, MATH500, AIME2024/2025, GPQA
- **Code**: HumanEval, MBPP, HumanEvalInfilling, McEval
- **Planning**: Countdown, Sudoku, Trip Planning
- **Long-context / Summarization**: CNN/DailyMail (for FlashDLM evaluation)
- **Throughput metrics**: Tokens-per-second (TPS) on A100/H100/H20; latency-per-token; flops-per-token

---

## 4. Identified Research Gaps

- **Gap 1 — No unified DLM inference engine**: AR models have vLLM, TGI, TensorRT-LLM. DLMs lack a production-quality inference engine that combines KV caching, batching, and continuous decoding efficiently. The `dInfer` framework (inclusionAI) is an early effort but far from mature.

- **Gap 2 — Batched inference scaling gap**: AR models benefit strongly from batched inference (increased arithmetic intensity). Current DLM acceleration benchmarks focus almost entirely on single-sequence latency. Fair throughput comparisons under batched workloads show AR models still dominate for most open-source DLMs.

- **Gap 3 — KV cache approximation quality-speed Pareto frontier**: Existing methods (Fast-dLLM, dKV-Cache, Elastic-Cache, EntropyCache) all take different approaches to deciding *when* and *where* to refresh the KV cache. There is no systematic comparative study of the quality-speed Pareto frontier of these methods under a unified evaluation protocol.

- **Gap 4 — Speculative decoding for full-attention DLMs**: DualDiffusion and DFlash apply speculative decoding ideas to DLMs, but the former requires a draft-verifier pair within the DLM regime, and the latter uses a DLM draft for an AR target. An efficient self-speculative or tree-based decoding scheme purely within the MDM framework is underexplored.

- **Gap 5 — Adaptive denoising schedules**: Tokens converge at different rates but most DLMs use uniform denoising schedules. PRR and Saber begin to address this, but principled data-driven or learned non-uniform schedules that minimize total compute while preserving quality are still nascent.

- **Gap 6 — Variable-length generation efficiency**: DLMs require a fixed-length canvas, causing overallocation waste. LR-DLLM addresses quality in this setting, but there is no solution that jointly optimizes variable-length generation quality *and* inference speed.

- **Gap 7 — Cross-model transferability of acceleration methods**: Most acceleration works are evaluated on LLaDA and Dream specifically. It is unclear how well techniques generalize across different DLM architectures (encoder-decoder, MoE, adapted-from-AR, trained-from-scratch).

- **Gap 8 — Hardware-aware DLM optimization**: No work systematically analyzes roofline models for DLM inference on modern accelerators (H100/H200/B200). Given the bidirectional attention, DLMs may have fundamentally different memory-bandwidth vs. compute bottlenecks than AR models.

- **Gap 9 — Model scheduling under strict latency budgets**: "Not All Denoising Steps Are Equal" (arXiv 2604.02340) shows only ~17% FLOP reduction by substituting a smaller model at less-critical steps. More aggressive cascade scheduling (small model early/late, full model only in the middle "sensitive window") with a jointly-optimized schedule could compound savings and is unexplored.

- **Gap 10 — Saliency estimation overhead**: DyLLM measures token saliency via cosine similarity between adjacent step representations. For long sequences and large models, this inter-step comparison itself may become costly. Lightweight proxy saliency signals (e.g., masked-token probability change, entropy-based like EntropyCache) that avoid full-sequence comparisons are underexplored as a substitute.

- **Gap 11 — Composition of acceleration methods**: Most papers test one technique in isolation. A principled composition study (e.g., DyLLM + EntropyCache + SlowFast + model scheduling) to understand complementarity vs. diminishing returns across task types would have immediate practical value.

- **Gap 12 — Long-context DLM inference efficiency**: Focus-dLLM is the first paper to specifically address long-context (16K+) DLM inference via attention sparsification. However, it covers only a single model (UltraLLaDA) on LongBench. A systematic study of how various acceleration techniques (KV cache, saliency, token pruning) interact with long-context settings is largely missing.

- **Gap 13 — Early-layer token importance estimation cost**: ES-dLLM and DyLLM both estimate token importance across iterations but with different proxy signals (tensor variation vs. cosine similarity). A rigorous comparison of these proxy signals' accuracy vs. compute cost is needed, along with theoretical grounding for when each proxy is most appropriate.

- **Gap 14 — Variable-length generation with acceleration**: Most acceleration papers assume fixed-length generation (e.g., 256 or 512 tokens). Combining acceleration techniques with variable-length generation (where the output canvas itself is adaptive) remains underexplored and is particularly important for practical deployment.

---

## 5. Available Resources

### Open-Source Code

| Repository | Description | License |
|-----------|-------------|---------|
| [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | Official NVIDIA implementation of Fast-dLLM; supports LLaDA and Dream; block-wise KV cache + parallel decoding | Check repo |
| [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | Official LLaDA 8B + LLaDA-MoE pretrained models and inference code | Apache 2.0 |
| [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) | DiffuGPT + DiffuLLaMA — AR-to-diffusion adaptation (ICLR 2025) | Check repo |
| [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) | Simple DLM framework; integrates Fast-dLLM cache/parallel decoding; Tiny-A2D small adapted models | Check repo |
| [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | Modular DLM inference framework: model, diffusion manager, decoder, KV-cache manager | Check repo |
| [pengzhangzhi/Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) | Full pipeline: data → training → evaluation → inference for code-focused DLMs | Check repo |
| [VILA-Lab/Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) | Survey + curated list of DLM papers, code, and models | N/A |
| [mscheong01/EntropyCache](https://github.com/mscheong01/EntropyCache) | EntropyCache training-free KV caching for LLaDA-8B-Instruct and Dream-7B-Instruct | Check repo |
| [m-arriola.com/e2d2](https://m-arriola.com/e2d2) | E2D2 encoder-decoder diffusion; code + model weights | Check repo |
| [m-arriola.com/bd3lms](https://m-arriola.com/bd3lms) | Block Diffusion (BD3-LMs); code + model weights + blog | Check repo |
| [vhicrgit/Window-Diffusion](https://github.com/vhicrgit/Window-Diffusion) | Window-Diffusion windowed token pruning + caching | Check repo |
| [hamishivi/tess-2](https://github.com/hamishivi/tess-2) | TESS 2 generalist instruction-following DLM | Check repo |
| [phymhan/S2D2](https://github.com/phymhan/S2D2) | S2D2 self-speculative decoding for block-diffusion LMs; covers SDAR, Fast-dLLM v2, LLaDA2.1 | Check repo |
| [LiangrunFlora/Slow-Fast-Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling) | SlowFast Sampling official PyTorch implementation; drop-in for LLaDA-8B and Dream-7B | Check repo |
| [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) | d1 masked SFT + diffu-GRPO RL for LLaDA; reasoning benchmarks | Check repo |
| [bansky-cl/diffusion-nlp-paper-arxiv](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv) | Auto-tracked arxiv papers on diffusion NLP; useful for staying current | N/A |

### Pretrained Models (HuggingFace)

| Model | HF Hub ID | Size |
|-------|-----------|------|
| LLaDA 8B Base/Instruct | `GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-8B-Instruct` | 8B |
| LLaDA 1.5 | `GSAI-ML/LLaDA-1.5` | — |
| LLaDA-MoE 7B-A1B | Huggingface (see paper) | 7B / 1.4B active |
| Dream 7B Instruct | `hkunlp/dream-7b-instruct` | 7B |
| TESS 2 | `hamishivi/tess-2` | ~7B |

### Evaluation Frameworks

- **DARE** (arXiv 2604.04215): Unified post-training + eval framework for dLLMs built on verl + OpenCompass; covers MDLMs and BDLMs; supports LLaDA, Dream, SDAR, LLaDA2.x; MMLU, GSM8K, MATH, GPQA, AIME, HumanEval, MBPP, planning tasks
- **dLLM framework** (ZHZisZZ): Reproduces official scores for LLaDA/Dream; standardized training + inference + evaluation pipeline
- **lm-evaluation-harness**: Compatible with DLMs via HuggingFace integration
- **"How Efficient Are Diffusion LMs?"** (arXiv 2510.18480): Critical evaluation practices paper; proposes Tokens-Per-Forward-Step (TPF) as unified efficiency metric; empirical benchmarking under batched settings

---

## 6. Implications for Idea Generation

**Directions worth exploring:**

1. **Unified quality-speed evaluation framework for DLM KV caching**: Multiple competing KV caching approaches (dKV-Cache, Fast-dLLM, Elastic-Cache, EntropyCache, FlashDLM-FreeCache, Window-Diffusion) exist with incompatible evaluation setups. A rigorous systematic comparison + a principled design space analysis (cache granularity × refresh policy × task type × model architecture) would be high-impact and publishable.

2. **Training-free speculative decoding within MDMs**: Leverage a smaller/sparser version of the same model (e.g., early-layer exit or quantized version) as a draft network, then verify with the full model — directly analogous to AR speculative decoding but adapted for the non-autoregressive, bidirectional setting. DualDiffusion makes an early attempt but uses external approximations; there is room for a principled token-acceptance criterion for MDMs.

3. **Confidence-calibrated adaptive step scheduling**: Saber and PRR show that non-uniform per-token step allocation improves both speed and quality. An information-theoretic framework for optimal step scheduling (minimizing expected remaining entropy per token) could yield a principled and broadly applicable algorithm.

4. **Batch-level and hardware-aware profiling of DLM inference**: A roofline analysis and systematic profiling of DLM bottlenecks under batched inference (memory-bound vs. compute-bound regimes) on H100/A100 would identify where algorithmic work can have the most impact and would expose the real gap vs. AR models.

5. **Distillation-free AR-to-DLM adaptation for inference speed**: Fast-dLLM v2 requires ~1B tokens of fine-tuning. Can parameter-efficient methods (LoRA, lightweight adapters) further reduce this to zero-shot or few-shot conversion while retaining the KV caching benefits of block diffusion?

6. **Saliency-driven partial attention with lightweight proxy signals**: DyLLM identifies salient tokens via inter-step cosine similarity, which has quadratic overhead. ES-dLLM uses tensor variation (L1/L2 norm of hidden state change) as a cheaper signal. Focus-dLLM uses previous-step token confidence. An entropy-based or gradient-free proxy signal (O(V) like EntropyCache) to decide which tokens need full attention vs. cached attention could yield a training-free, highly scalable partial-attention framework. A unified comparison of these proxy signals (cosine similarity vs. tensor variation vs. confidence vs. entropy) under the same evaluation protocol would clarify which to use in which context.

7. **Composition study of multiple training-free techniques**: No existing work systematically evaluates whether DyLLM + EntropyCache, or SlowFast + dLLM-Cache + model scheduling, are additive or subadditive. A principled ablation and composition study (along with a theory of when methods are complementary) would clarify the optimization landscape.

8. **Self-speculative decoding for pure MDLMs (non-block-diffusion)**: S2D2 works elegantly for block-diffusion models (where block-size=1 gives an AR mode). An analogous "self" speculative mechanism for pure MDMs (LLaDA, Dream) that doesn't require a separate draft model — e.g., drafting with fewer denoising steps and verifying with more — is an open problem.

**Saturated / crowded directions:**

- Simple confidence-threshold sampling (Masked tokens with low confidence are skipped) — many variants already exist; marginal gains likely.
- Training-based improvements to DLM quality (perplexity, benchmark scores) without addressing inference speed — already well-covered by LLaDA, Dream, TESS 2, ADLM, LLaDA-MoE.
- Basic KV cache reuse strategies (applying uniform cache refresh to all tokens) — multiple approaches already published (Fast-dLLM, dKV-Cache, Elastic-Cache, dLLM-Cache, EntropyCache); the space is crowded and marginal improvements unlikely without principled differentiation.

**Cross-domain analogies with potential:**

- *Speculative decoding* (AR world) → drafting within the denoising process using a fast approximate model and accepting with a probability correction (MDM-specific acceptance criterion needed).
- *Flash Attention* memory efficiency tricks → adapted to the bidirectional full-sequence attention of DLMs; SageAttention 8-bit attention is broadly applicable.
- *Mixture of Experts sparse activation* → LLaDA-MoE demonstrates this is viable; combining MoE sparsity with token-adaptive denoising could compound efficiency gains.
- *Consistency/Distillation models* (image diffusion) → Consistency distillation for DLMs to reduce steps from 64→4 while preserving quality is largely unexplored in the language domain.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| Fast-dLLM (NVlabs/Fast-dLLM) | High | Check | **Adopt/Extend** | Best-in-class training-free KV cache + parallel decoding; supports LLaDA/Dream; actively maintained by NVIDIA |
| LLaDA 8B / Dream 7B (HF models) | High | Apache 2.0 / Check | **Adopt** | Standard benchmarking baselines; inference infrastructure well-established |
| dInfer (inclusionAI/dInfer) | Medium | Check | **Extend** | Modular framework with KV-cache manager abstraction; good base for implementing new caching policies |
| ZHZisZZ/dllm | Medium | Check | **Adopt** | Unified inference + eval pipeline; fastest way to reproduce baselines and add new acceleration modules |
| EntropyCache (mscheong01) | Medium | Check | **Extend** | Clean implementation of entropy-guided refresh; straightforward to swap in alternative refresh signals |
| Block Diffusion / E2D2 (m-arriola) | Medium | Check | **Extend** | Reusable block diffusion infrastructure; useful if exploring encoder-decoder or AR-hybrid acceleration |
| lm-evaluation-harness | High | MIT | **Adopt** | Industry-standard evaluation; ensures fair comparison with prior work |
| DARE framework (arXiv 2604.04215) | High | Check | **Adopt** | Unified eval + post-training across LLaDA/Dream/SDAR/LLaDA2.x; built on verl+OpenCompass; most comprehensive dLLM research substrate available |
| S2D2 (phymhan/S2D2) | Medium | Check | **Extend** | Clean self-speculative baseline for block-diffusion; covers SDAR/LLaDA2.x/Fast-dLLM v2; useful reference for MDM analog |
| SlowFast Sampling (LiangrunFlora) | High | Check | **Extend** | Best training-free sampling baseline with open code; drop-in for LLaDA/Dream; combining with new saliency signals is natural extension |

**Reusable evaluation frameworks**: Fast-dLLM's benchmarking scripts, DARE, lm-evaluation-harness.

**Reusable data pipelines**: LLaDA and Dream both use standard open instruction-tuning datasets; the dLLM framework standardizes loading.

**Reusable pretrained checkpoints**: `GSAI-ML/LLaDA-8B-Instruct` and `hkunlp/dream-7b-instruct` are the standard test models; always use both for reproducibility.

---

*Survey completed by sibyl-literature agent. All speedup numbers are as reported in the respective papers; hardware configurations vary — see individual papers for experimental details.*
