# 文献调研报告

**研究主题**: 在 training-free 条件下改进 Diffusion Language Models (DLMs) 的生成性能，或在维持性能的同时提升 DLMs 的推理速度
**调研时间**: 2026-03-11
**arXiv 搜索关键词**: `"diffusion language model" OR "discrete diffusion" text generation`, `ti:"diffusion language model" inference acceleration`, `"training-free" OR "sampling strategy" discrete diffusion text`, `ti:"Fast-dLLM" OR ti:"block diffusion" OR ti:"MDLM"`, `"masked diffusion" "noise schedule" OR "confidence" OR "entropy" token selection denoising`
**Web 搜索关键词**: `diffusion language models state of the art 2025 MDLM SEDD`, `discrete diffusion language model inference speed acceleration training-free 2025`, `diffusion language model benchmark leaderboard text generation perplexity 2025`, `LLaDA Dream-7B sampling strategy improvement training-free`, `CDLM consistency diffusion language model faster sampling`

## 1. 领域现状摘要

Diffusion Language Models (DLMs) 近两年经历了快速发展，已从实验性概念走向可与自回归（AR）模型竞争的成熟范式。核心范式是**离散扩散**（discrete diffusion），尤其是**masked diffusion**——通过 absorbing state（mask token）进行前向加噪、反向去噪来生成文本。代表性基础工作包括 SEDD（2023, ICML 2024 最佳论文）、MDLM（NeurIPS 2024）、以及 Reparameterized Discrete Diffusion（RDLM, 2023）。

在规模化方面，LLaDA-8B（2025年2月）首次证明了 8B 参数量的 masked diffusion LLM 可与同规模 AR 模型竞争；Dream 7B（2025年8月）在仅用 LLaDA 四分之一训练数据的条件下全面超越前者；LLaDA 2.0（2025年12月）将规模推至 100B 参数，与 Qwen3-30B 持平；Google Gemini Diffusion（2025年）首次实现商业级 AR 模型性能平齐，推理速度达 1479 tokens/s。ByteDance 的 Seed Diffusion 在 H20 GPU 上达到 2146 tokens/s。

然而，DLMs 在复杂推理（GPQA Diamond 40.4% vs 56.5%）和通用知识（Global MMLU 69.1% vs 79.0%）等任务上仍落后于 AR 模型。推理效率是另一大瓶颈——标准 masked diffusion 不支持 KV cache，每步需全量重计算，实际推理速度受限。因此，**training-free 的采样策略改进**和**推理加速**成为当前最活跃的研究方向。

## 2. 核心参考文献

### 2.1 基础模型与架构

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | SEDD: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution | arXiv 2310.16834 | 2023 | 提出 score entropy loss 将 score matching 推广到离散空间；PPL 优于 GPT-2；32x 少的网络评估即可达到相似质量 | 训练成本高；规模化能力有限 |
| 2 | MDLM: Simple and Effective Masked Diffusion Language Models | arXiv 2406.07524 | 2024 | 工程化 masked diffusion 训练；比 SEDD PPL 改进 17%；简洁有效的框架 | 仍与 AR 模型有差距 |
| 3 | LLaDA: Large Language Diffusion with mAsking | GitHub ML-GSAI | 2025 | 首个 8B 参数 masked diffusion LLM；从零训练；证明规模化可行 | 推理速度慢；复杂推理弱 |
| 4 | Dream 7B: Diffusion Large Language Models | arXiv 2508.15487 | 2025 | AR-LLM 初始化 + 上下文自适应 token 级噪声重调度；仅用 0.6T tokens 训练全面超越 LLaDA | 推理效率问题未完全解决 |
| 5 | LLaDA 2.0: Scaling Up Diffusion Language Models to 100B | arXiv 2512.15745 | 2025 | 最大文本扩散模型（100B）；性能与 Qwen3-30B 持平 | 计算资源需求极高 |
| 6 | ADLM: Anchored Diffusion Language Model | ICLR 2025 | 2025 | PPL 20.14（OWT），比 MDLM 改进 12.3%；7/7 基准上 SOTA zero-shot PPL | 需要额外训练 |
| 7 | BD3-LMs: Block Diffusion Language Models | ICLR 2025 Oral | 2025 | 插值 AR 和扩散；支持变长生成和 KV cache；SOTA PPL | 需要训练 block 结构 |
| 8 | RDLM: A Reparameterized Discrete Diffusion Model | arXiv 2302.05737 | 2023 | 等价重参数化采样公式；更灵活的训练和解码 | 早期工作，规模小 |

### 2.2 Training-Free 推理加速（核心方向）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 9 | **Fast-dLLM**: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding | arXiv 2505.22618 | 2025 | Block-wise 近似 KV Cache + confidence-aware 并行解码；13.3x 加速（长文本 27.6x） | KV cache 为近似方案；依赖 confidence 阈值 |
| 10 | **DCD**: Deferred Commitment Decoding for Diffusion Language Models | arXiv 2601.02076 | 2026 | Certainty-aware 滑动窗口；低不确定性 token 提前 commit，高不确定性延迟；平均 +1.73%，最高 +16.5% | 加速幅度有限 |
| 11 | **SureLock**: Stopping Computation for Converged Tokens | arXiv 2602.06412 | 2026 | 锁定已收敛 token 位置跳过 query/FFN 计算；FLOPs 减少 30-50% | 需要判断收敛条件 |
| 12 | **EB-Sampler**: Accelerated Sampling via Entropy Bounded Unmasking | arXiv 2505.24857 | 2025 | 动态多 token 解码（entropy bound）；2-3x 加速无性能损失 | 通用性待验证 |
| 13 | **DUS**: Dilated Unmasking Scheduler | arXiv 2506.19037 | 2025 | 将位置分为非相邻 dilated groups 并行解码；最小化 joint entropy upper bound | 理论分析强但实际加速受限 |
| 14 | **TABES**: Trajectory-Aware Backward-on-Entropy Steering | arXiv 2602.00250 | 2026 | 梯度引导推理框架；Token Influence Score + ActiveQueryAttention；优越的推理时 Pareto 前沿 | 需要反向传播，计算成本非零 |
| 15 | **LR-DLLM**: Length-Regularized Inference for DLLMs | arXiv 2602.07546 | 2026 | 解决变长生成中的 length-induced bias；不修改模型或训练 | 专注长度问题，不通用 |

### 2.3 Training-Free 采样策略（生成质量改进）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 16 | **Demystifying MaskGIT Sampler**: Adaptive Order Selection in Masked Diffusion | arXiv 2510.04525 | 2025 | 揭示 MaskGIT 隐含温度采样机制；提出 moment sampler + partial caching + exploration-exploitation hybrid | 主要在图像域验证 |
| 17 | **VMD**: Variational Masked Diffusion Models | arXiv 2510.23606 | 2025 | 引入隐变量建模 token 间依赖；改善全局一致性 | 需要额外训练 |
| 18 | Simple Guidance Mechanisms for Discrete Diffusion Models | arXiv 2412.10193 | 2024 | 推导 classifier-free/classifier-based guidance for discrete diffusion；uniform noise diffusion 更可引导 | 主要面向可控生成 |
| 19 | Diffusion-EAGS: Conditional MASK Discrete Diffusion | arXiv 2411.06438 | 2024 | 熵自适应 Gibbs 采样 + 熵基噪声调度；最佳质量-多样性权衡 | 规模较小 |
| 20 | DDOT: Flexible-length Text Infilling with Optimal Transport | arXiv 2506.13579 | 2025 | OT position coupling 实现灵活长度位置 infilling | 专注 infilling 任务 |

### 2.4 需要训练的加速方法（对比参考）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 21 | **CDLM**: Consistency Diffusion Language Models | arXiv 2511.19269 | 2025 | 一致性建模 + block-wise causal attention 兼容 KV cache；3.6x-14.5x 低延迟 | 需要额外 fine-tuning |
| 22 | **D2F**: Discrete Diffusion Forcing | arXiv 2508.09192 | 2025 | 学习解码 monotonically 递增 mask ratio blocks；首个超 AR 推理速度的开源 dLLM（119.9 tok/s） | 需要训练 |
| 23 | **ARMD**: Auto-Regressive Masked Diffusion | arXiv 2601.16971 | 2026 | Block-wise causal 重构；strided 并行生成；SOTA 并行文本生成基准 | 需要训练 |
| 24 | **CCDD**: Coevolutionary Continuous Discrete Diffusion | arXiv 2510.03206 | 2025 | 连续+离散联合扩散；理论上更强表达力 | 架构复杂 |
| 25 | **MetaState**: Persistent Working Memory for DLLMs | arXiv 2603.01331 | 2026 | Cross-step 持久记忆（cross-attention + GRU）；冻结 backbone；改善跨步一致性 | 需要训练额外模块 |

## 3. SOTA 方法与基准

### 3.1 当前 SOTA 模型

| 模型 | 参数量 | 类型 | 关键指标 | 速度 |
|------|--------|------|---------|------|
| LLaDA 2.0 | 100B | Masked Diffusion | 与 Qwen3-30B 持平 | - |
| Dream 7B | 7B | Masked Diffusion | 超越 LLaDA-8B（全面） | - |
| ADLM | - | Anchored Diffusion | OWT PPL 20.14 | - |
| BD3-LM | - | Block Diffusion | SOTA PPL among DLMs | 支持 KV cache |
| Gemini Diffusion | - | 商业 | 首个商业级 AR 平齐 | 1479 tok/s |
| Seed Diffusion | - | 商业 | 代码专注 | 2146 tok/s (H20) |
| D2F-Dream-7B | 7B | Diffusion Forcing | 超越 AR 推理速度 | 119.9 tok/s (GSM8K) |

### 3.2 主流评测基准

- **语言建模**: OpenWebText PPL, text8 bpc, Lambada, PubMed, ArXiv
- **推理**: GSM8K, MATH500, GPQA Diamond
- **代码**: HumanEval, MBPP, HumanEvalInfilling, McEval
- **通用**: MMLU, MMLU-Pro, BBH, ARC, HellaSwag
- **扩散特有**: 生成 PPL、infilling 质量、quality-speed trade-off curve

### 3.3 主流评测指标

- Perplexity (PPL) / bits-per-character (bpc)
- Pass@1 / Pass@10（代码）
- Accuracy（推理/分类）
- 推理速度 (tokens/sec)
- NFE (Number of Function Evaluations)
- FLOPs reduction ratio

## 4. 已识别的研究空白

- **空白 1: 统一的 training-free 采样框架缺失**。当前 training-free 方法分散于不同维度（KV cache、并行解码、token 选择顺序、entropy-based unmasking），缺少一个统一框架将这些正交技术组合为最优采样策略。

- **空白 2: Token unmasking 顺序的理论最优解未知**。现有方法使用 confidence-based（top-1 logit）、entropy-based、margin-based（top1-top2）等启发式规则选择解码顺序，但缺乏理论证明哪种策略在什么条件下最优。TABES 提出了梯度引导方法，但计算成本高。

- **空白 3: 采样步数的自适应调度不足**。大多数方法使用固定步数或简单线性调度，缺少根据生成内容复杂度动态调整步数的 training-free 方案（类似 AR 模型的 early stopping）。EB-Sampler 做了初步探索。

- **空白 4: 跨步信息传递在 training-free 设定下未被探索**。MetaState 通过训练额外模块实现跨步记忆，但是否存在不需要训练的跨步信息复用方案（如利用中间激活的相似性）尚未被研究。SureLock 的锁定机制是一个方向但仅针对已收敛 token。

- **空白 5: DLM 在推理任务上的 training-free 改进**。DLM 在 GSM8K、MATH 等推理任务上与 AR 模型差距显著，是否可以通过 training-free 的采样策略（如类似 best-of-N、self-consistency 的方案）专门改善推理性能。

- **空白 6: 注意力模式优化**。DLM 使用 bidirectional full attention，但不同去噪步骤中注意力模式可能有冗余。是否可以 training-free 地裁剪或稀疏化注意力（SureLock 做了部分探索），在不损失质量的前提下加速。

## 5. 可用资源

### 开源代码
- **LLaDA**: https://github.com/ML-GSAI/LLaDA （8B masked diffusion LLM）
- **Dream 7B**: https://github.com/DreamLM/Dream （7B，含 Base 和 Instruct）
- **MDLM**: https://github.com/kuleshov-group/mdlm （NeurIPS 2024）
- **SEDD**: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- **BD3-LM**: https://github.com/kuleshov-group/bd3lms （ICLR 2025 Oral）
- **CDLM**: https://github.com/SqueezeAILab/CDLM
- **Fast-dLLM**: NVlabs 开源框架
- **SureLock**: https://daioba.github.io/surelock
- **USD3**: https://github.com/LingxiaoShawn/USD3

### 数据集
- **OpenWebText**: 标准语言建模基准
- **text8**: 字符级语言建模
- **GSM8K / MATH500**: 数学推理
- **HumanEval / MBPP**: 代码生成
- **MMLU / BBH / ARC**: 通用能力

### 预训练模型
- LLaDA-8B (HuggingFace: GSAI/LLaDA-8B-Base, LLaDA-8B-Instruct)
- Dream-7B (HuggingFace: Dream-org/Dream-v0-Base-7B, Dream-v0-Instruct-7B)
- MDLM checkpoints (via GitHub)

### 综述资源
- Awesome-DLMs: https://github.com/VILA-Lab/Awesome-DLMs
- A Survey on Diffusion Language Models: arXiv 2508.10875
- Discrete Diffusion in LLMs: A Survey: arXiv 2506.13759

## 6. 对 Idea 生成的启示

### 值得探索的方向

1. **组合正交加速技术的统一采样框架**：Fast-dLLM 的 KV cache + EB-Sampler 的动态 unmasking + SureLock 的计算跳过 + DUS 的 dilated scheduling，这些方法理论上正交可组合，但尚无系统性研究。一个统一的、可配置的 training-free 采样器可能获得 super-linear 的加速效果。

2. **利用去噪轨迹信息的 training-free 方案**：当前每步重新计算所有 token 的概率分布，但相邻步骤的分布变化通常很小。可以设计基于分布变化量（KL divergence / JS divergence）的自适应计算策略——变化大的 token 重计算，变化小的复用上步结果。这是 SureLock 思路的推广，但无需"锁定"，而是"软插值"。

3. **Token 解码顺序的信息论最优化**：DUS 最小化 entropy upper bound 但使用固定 dilated pattern。可以用 training-free 的方式（如基于当前步 logit 分布的互信息估计）动态规划最优解码顺序，平衡 token 间的依赖关系和并行度。

4. **DLM 专用的 inference-time scaling**：AR 模型已有 best-of-N、majority voting、self-consistency 等推理时扩展方案。DLM 的并行生成特性天然支持更高效的 inference-time scaling——例如，在同一去噪轨迹的不同步骤进行分支和剪枝（类似 beam search 但在 diffusion 空间中操作）。

### 已被充分探索的方向（避免重复）

- 基于 confidence 的简单 token 选择策略（top-1, entropy, margin）已被广泛研究
- KV cache 的 block-wise 近似已由 Fast-dLLM 和 CDLM 覆盖
- 一致性蒸馏（consistency distillation）属于 training-based，非本课题范围

### 跨域借鉴潜力

- **图像扩散模型的 training-free 加速**：FSampler 的 epsilon extrapolation、DDIM 的 skip-step 策略、DPM-Solver 的高阶 ODE solver，这些在连续扩散中成功的 training-free 方法能否迁移到离散扩散场景
- **AR 模型的 speculative decoding**：DCD 已借鉴了类似思路，但更深入的 draft-verify 范式（如 Medusa、EAGLE）能否适配 DLM 的并行去噪框架
- **Mamba / SSM 的选择性计算**：DLM 中不同 token 的去噪难度不同，类似 Mamba 的选择性机制能否 training-free 地应用于注意力计算的动态分配
