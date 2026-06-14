

## Project Spec
# 项目: dlm-improve

## 研究主题
在 training-free 条件下改进 Diffusion Language Models (DLMs) 的生成性能，或在维持性能的同时提升 DLMs 的推理速度。

## 背景与动机
Diffusion Language Models（如 MDLM, SEDD, Dream, LLaDA, MMaDA）作为自回归模型的替代范式，具有并行生成、灵活编辑等优势，但在采样质量和推理速度上仍存在明显短板。现有加速方法（如 Block Diffusion, FAST-DLLM V2）虽有进展，但 sampling strategy 仍有较大改进空间。本项目聚焦 training-free 方法，从扩散模型的底层数学原理出发，寻找改进采样过程的理论依据和工程实现。

## 初始想法
- 从扩散过程的数学公式推导出发，分析现有 sampling strategy 的理论缺陷
- 从底层机制（noise schedule、transition kernel、denoising dynamics）寻找改进点
- 探索自适应步数/token 级别的采样策略
- 结合离散扩散过程的特殊性质设计更高效的 sampler
- 研究 confidence-based 或 entropy-based 的动态去噪策略

## 关键参考文献
- MDLM (Masked Diffusion Language Models)
- SEDD (Score Entropy Discrete Diffusion)
- Dream (Diffusion Reasoning with Enhanced Abilities for Machines)
- LLaDA (Large Language Diffusion with Attention)
- MMaDA (Multi-Modal Diffusion with Attention)
- Block Diffusion
- FAST-DLLM V2: Efficient Block-Diffusion LLM
- Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling
- 其他由文献调研阶段补充

## 可用资源
- GPU: 4x NVIDIA RTX PRO 6000
- 服务器: cs8000d (SSH MCP connection: cs8000d, fallback: default)
- 远程路径: /home/ccwang/sibyl_system
- 单个实验如有需要或者资源空闲可尝试多卡并行提升效率
- 实验要考虑大 batch size，以提升训练/推理速度，充分利用显存

## 实验约束
- 实验类型: **优先 training-free**；允许 1 小时内的 LoRA 轻量训练
- 训练任务时间预算: ~1 小时
- 评测任务: **不受时间限制**，按 benchmark 完整跑完
- 统计显著性: **不需要**。不换 seed 多跑，benchmark 合理且无过度下采样时以单次结果为准
- Pilot 采样: **最少 100 条**，不接受 n<100 的 pilot。benchmark 条数本身小于 100 条的除外
- 模型规模: 使用各 DLM 论文的开源预训练模型（通常为中小规模），建议先从小尺寸（0.6B,4B）开始

## 评测策略
- 使用社区通用 benchmark（如各 DLM 论文常用的 text8, One-Billion-Word, lambada, GSM8K 等）
- Pilot 实验可选用小型 benchmark 或 benchmark 子集（>=100 条）
- 正式实验使用完整 benchmark，不做下采样
- 对标模型: 各 DLM 的原始采样策略 + 已有加速方法（Block Diffusion, FAST-DLLM V2 等）

## 目标产出
- **顶会论文**（NeurIPS / ICML / ACL 级别）
- 质量期望: **weak accept 及以上**（borderline+）
- 论文模板: **NeurIPS LaTeX 模板**
- 论文语言: **全英文**
- 图像可视化：可以使用可视化图像来解释和表达含义思想想法的，要尝试使用可视化图像进行表示，不要拘泥于表格和折现柱状曲线图，还要考虑热力图、注意力图等解释模型内部机制的可视化方法

## 方法论要求
- **从原理公式推导**：不仅仅是工程 trick，要有数学/理论支撑
- **底层机制分析**：深入分析扩散过程的 transition dynamics、noise schedule 等
- **理论→实验**：先建立理论框架，再用实验验证理论预测
- 结合实际工程实现，确保方法可复现

## 特殊配置
- 飞书同步: 启用
- Codex 审查: 启用
- 写作模式: parallel（加速写作）


## User's Initial Ideas
- 从扩散过程的数学公式推导出发，分析现有 sampling strategy 的理论缺陷
- 从底层机制（noise schedule、transition kernel、denoising dynamics）寻找改进点
- 探索自适应步数/token 级别的采样策略
- 结合离散扩散过程的特殊性质设计更高效的 sampler
- 研究 confidence-based 或 entropy-based 的动态去噪策略

## Seed References (from user)
- MDLM (Masked Diffusion Language Models)
- SEDD (Score Entropy Discrete Diffusion)
- Dream (Diffusion Reasoning with Enhanced Abilities for Machines)
- LLaDA (Large Language Diffusion with Attention)
- MMaDA (Multi-Modal Diffusion with Attention)
- Block Diffusion
- FAST-DLLM V2: Efficient Block-Diffusion LLM
- Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling
- 其他由文献调研阶段补充

## 文献调研报告（请仔细阅读，避免重复已有工作）
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


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Iteration 1 Proposal (Refinement Round 2)

## 标题

**REVISION ATLAS: A Compute-Normalized Diagnostic Benchmark for When Revision Helps Diffusion Language Models**

## 一句话摘要

本轮不再尝试把 `TIGER` 包装成主方法胜利，而是转向一个更诚实也更有可发表性的主线：

**系统研究 training-free revision 在 DLM 上何时有效、何时失效，以及为什么 calibration / entropy / instability 更像诊断信号，而不是可靠控制律。**

## 为什么要从 TIGER 转向 REVISION ATLAS

最新 pilot 已经给出足够清晰的裁决：

1. `TIGER-64+3` 在 GSM8K shortlist 上只与 `Entropy-Revise-64+3` 打平，没能形成新的方法胜利。
2. 虽然 TIGER 超过了 `Prophet` 和 `DNB`，但明显落后于 `CORE-proxy`，因此方法新颖性与实证强度都不够。
3. `task gating` 只部分修复了 code syntax failure，而且 reasoning 还掉了 1 个点，因此它更像 boundary fix，而不是统一方法。

这些结果并不意味着本轮失败。相反，它们把真正稳固的贡献指向了另一个方向：

- honest compute accounting
- task-dependent revision failure analysis
- code boundary fragility 的机制解释
- calibration / entropy / instability 的角色重估

## 核心研究问题

1. 在 DLM 中，`matched actual compute` 是否会改变 training-free 方法的真实排序？
2. revision 的收益究竟来自哪里：局部 uncertainty，还是任务结构脆弱性？
3. 为什么某些 signal 在诊断上有效，却无法稳定转化为方法收益？
4. code 上的 cheap guard 能保护到什么程度，为什么依然不够？

## 候选池与本轮选择

### 主候选：`cand_diag`

将论文组织成一篇 compute-normalized diagnostic benchmark paper。

目标不是宣称“我们发明了最强 sampler”，而是回答：

**在 DLM 上，revision 的收益边界在哪里？**

### 次候选：`cand_signal`

如果 scope 需要继续收缩，则退化为更窄的 signal audit：

- entropy
- instability
- calibration

比较它们作为 revision predictors 的有效性与局限。

### 边界候选：`cand_guardrail`

把 syntax-aware guard 保留为 code 边界附录，而不是主论文中心。

## 核心主张

### Claim 1: Honest compute accounting is not bookkeeping; it changes conclusions

用名义步数比较 DLM revision 方法会误导读者。

必须联合报告：

- actual NFE
- wall-clock
- tokens/sec
- batch size / backend / compile 状态

否则“公平对比”只是表面公平。

### Claim 2: Revision benefit is task-dependent, not universally helpful

reasoning 与 code 对局部 revision 的容忍度不同。

- reasoning 允许一定局部编辑
- code 受全局语法与执行结构约束，更容易被局部 revision 破坏

### Claim 3: Diagnostic usefulness does not imply control usefulness

即使 calibration / entropy / instability 能预测错误，它们也不一定能转化为更强的 revision policy。

这是本轮最重要的概念澄清。

### Claim 4: Cheap gating is an appendix-worthy safeguard, not the paper headline

gating 可以减少 syntax failure，但它并没有把 revision 变成跨任务统一可用的主方法。

因此它应被写成：

- boundary ablation
- appendix evidence
- scope control tool

而不是 headline contribution。

## 论文结构草案

### Section 1: Problem Reframing

说明为什么从 method win 转向 diagnostic benchmark：

- null result 不是噪声，而是结论的一部分
- revision 的问题不只是“能否提升”，而是“何时应该介入”

### Section 2: Honest Protocol

统一协议：

- exact extraction audit
- held-out calibration split
- matched actual compute
- batch-maximized throughput setup

### Section 3: Benchmark Matrix

比较以下方法：

- `Standard-64`
- `DNB-84`
- `Prophet-64`
- `CORE-proxy-64`
- `Entropy-Revise-64+3`
- `TIGER-64+3`

并沿三个维度解读：

- quality
- compute
- task fragility

### Section 4: Boundary Analysis

用 HumanEval / MBPP 边界结果展示：

- syntax failure 可以被部分抑制
- runtime / semantic correctness 仍然难以修复

### Section 5: Why Signal Quality Does Not Equal Method Quality

解释为什么 calibration / entropy / instability 的 diagnostic value 不自动转化为 control value。

## 下一阶段 planning 的最小任务

1. 固化 benchmark framing 与主表结构
2. 明确哪些结果已经足够写主文，哪些只需最小增量补齐
3. 设计 factorized ablation：
   - task type
   - signal type
   - compute accounting
   - guard on/off
4. 设计图像化表达，而不只是一张 accuracy 表

## 明确不再做的事

1. 不再把 `math500_transfer` 当作 TIGER 方法扩展验证
2. 不再宣称 calibration correction 是主要增益来源
3. 不再把 code revision 当成已经解决的问题

## 成功标准

如果 planning 后的论文主张能满足以下三条中的两条，就继续沿 `cand_diag` 推进：

1. 贡献问题清晰且与现有 DLM literature 有明显区分
2. 现有结果已经足以支撑主表与主图的 70% 以上
3. 只需少量增量实验就能形成完整故事，而非重开大规模方法搜索

## 当前决定

本轮 `idea_debate` 的决定是：

**保持 `cand_diag` 为主候选，保留 `cand_signal` 与 `cand_guardrail` 作为备选或附录方向；`cand_tiger` 不再作为 method-forward 主线。**


## 当前可检验假设
# Testable Hypotheses for REVISION ATLAS

## Scope

这一轮已经不是继续为 `TIGER` 寻找方法胜利，而是把主问题改写为：

**在 training-free DLM revision 中，什么时候 revision 有用，什么时候它会破坏结构，以及哪些 signal 只适合诊断而不适合作为控制律。**

- Primary benchmark: `GSM8K`
- Secondary reasoning benchmark: `MATH500`（仅在 planning 后确认为必要增量）
- Boundary benchmark: `HumanEval` / `MBPP` 仅作为 code fragility 边界证据
- Primary model: `LLaDA-8B-Instruct`
- Primary contribution type: `compute-normalized diagnostic benchmark`, not `new method win`

## H1: Honest Compute Accounting Changes the Ranking

**Statement**: 在 matched actual compute（actual NFE、wall-clock、tokens/sec）下，training-free revision methods 的排序会显著不同于按名义步数比较时的排序。

**Measurement**:
- Compare `Standard-64`, `DNB-84`, `Prophet-64`, `CORE-proxy-64`, `Entropy-Revise-64+3`, `TIGER-64+3`
- Report exact-match accuracy, actual NFE, throughput, latency, batch size
- Explicitly distinguish nominal step count from actual compute

**Expected outcome**:
- 至少一个“按名义 NFE 看起来公平”的比较，在真实 compute 维度上并不公平
- Honest accounting 会改变读者对方法优先级的判断

**Falsification**:
- 排序在名义 NFE 与实际 compute 两种口径下基本不变
- 那么 compute-normalized benchmark 的中心论点会变弱

## H2: Revision Benefit Is Task-Dependent and Linked to Structural Fragility

**Statement**: revision 的收益更接近“任务结构脆弱性”问题，而不是单纯的 token-level uncertainty 问题。

**Measurement**:
- Reasoning tasks: compare draft vs revision benefit
- Code tasks: compare syntax/runtime failure changes after revision
- Analyze whether local uncertainty metrics can predict global task success

**Expected outcome**:
- reasoning 对局部 revision 更宽容
- code 对局部 revision 更敏感，因为全局语法/执行约束更强

**Falsification**:
- reasoning 与 code 对 revision 的响应没有显著差异
- 或者所有差异都可由简单 entropy 阈值解释

## H3: Calibration and Instability Are Better Diagnostics Than Controllers

**Statement**: calibration、entropy、instability 都提供有价值的错误诊断，但不足以稳定地充当 revision control law。

**Measurement**:
- Compare ECE, entropy-error correlation, instability-benefit correlation
- Compare diagnostic quality against realized accuracy gains
- Analyze cases where a strong signal still fails to yield a method win

**Expected outcome**:
- 这些信号能解释失败模式
- 但“diagnostic usefulness” 不等于 “method usefulness”

**Falsification**:
- 某一个 signal 在多个任务上都稳定给出方法胜利
- 那么论文应重新回到 method story，而不是 benchmark story

## H4: Cheap Gating Is a Boundary Fix, Not a General Solution

**Statement**: task gating 或 syntax guard 能部分减少 code 上的崩坏，但无法把 revision 变成跨任务稳健的统一策略。

**Measurement**:
- Compare `ungated revision` vs `gated revision`
- Report reasoning delta, parse failure delta, runtime failure delta

**Expected outcome**:
- gating 降低 syntax failure
- 但不能恢复到 Standard baseline，也不能支撑主论文 headline

**Falsification**:
- gating 在 reasoning 与 code 上都显著改善，并稳定优于全部强基线
- 那意味着 method line 应被重新打开

## Decision Rule

进入下一阶段 planning 时，默认围绕 `cand_diag` 组织实验与论文。

只有在后续最小增量实验里同时满足以下条件，才重新考虑恢复 method-forward 叙事：

1. 某个 signal-guided revision 在 reasoning 上显著超过 `Entropy-Revise`
2. 该优势在第二 reasoning benchmark 上复现
3. code boundary 上不再明显弱于 `Standard`

否则坚持当前主张：

**REVISION ATLAS: revision 何时有用、何时有害，以及为什么 calibration correction 本身并不构成足够的控制方法。**


## 小型实验真实反馈（必须基于这些证据修正 idea，不能忽略负结果）
# DCD Optional Pilot Summary

- Task: `baseline_dcd_optional`
- Verdict: `NO_GO`
- Reason: the low-cost `DCD-lite-64` pilot completed cleanly, but accuracy was only `0.26` on the 100-sample GSM8K pilot, well below the already reproduced `Standard-64` / `DNB-84` pilot accuracy of `0.36`.
- Compute profile: `actual_nfe=65.0`, `latency_sec=157.41`, `tokens_per_sec=162.63`, `batch_size=57`, `attention_backend=eager`
- Decision: keep the result as appendix/concurrent-work evidence, but do not spend more engineering time on a faithful DCD reimplementation unless the paper narrative later requires a stricter head-to-head baseline.


## 小型实验结构化信号（供你提炼 go/no-go / confidence / hypothesis status）
{
  "overall_recommendation": "NO_GO",
  "selected_candidate_id": "shared",
  "candidates": [
    {
      "candidate_id": "shared",
      "task_id": "baseline_dcd_optional",
      "go_no_go": "NO_GO",
      "confidence": 0.78,
      "supported_hypotheses": [],
      "failed_assumptions": [
        "Optional DCD baseline is cheap enough to add and likely competitive on the current GSM8K pilot stack."
      ],
      "key_metrics": {
        "accuracy": 0.26,
        "actual_nfe": 65.0,
        "latency_sec": 157.41,
        "tokens_per_sec": 162.63,
        "batch_size": 57
      },
      "notes": "Low-cost approximate DCD-lite pilot ran successfully, but it underperformed Standard-64/DNB-84 on the same 100-sample GSM8K pilot stack. Do not invest in a faithful DCD reproduction unless concurrent-work comparison becomes mandatory."
    }
  ]
}


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_diag",
      "title": "REVISION ATLAS: A Compute-Normalized Diagnostic Benchmark for Diffusion LM Revision",
      "status": "selected",
      "summary": "把这轮主贡献从方法宣称转为诊断与基准研究：系统回答 revision 何时有用、何时有害、哪些信号只是诊断有效但不足以做控制。",
      "hypotheses": [
        "H1: 在 matched actual compute 下，不同 training-free revision 方法的排序与名义 NFE 排序显著不同。",
        "H2: revision 收益主要由任务结构脆弱性而非单纯校准误差决定。",
        "H3: reasoning 与 code 对局部 revision 的容忍度显著不同，cheap gating 只能部分缓解 code 退化。",
        "H4: calibration / entropy / instability 更适合作为 diagnostic signals，而不是稳定的方法控制律。"
      ],
      "pilot_focus": "复用已有 GSM8K shortlist、gating boundary、calibration held-out 结果，下一步在 planning 中补齐 factorized benchmark 设计、图表与最小增量实验。",
      "success_probability": 0.9,
      "key_perspectives": [
        "empiricist（诚实 compute accounting）",
        "contrarian（防止把 null result 包装成方法胜利）",
        "theoretical（结构脆弱性与不可逆 commit）",
        "codex-reviewer（scope 收缩与论文可讲述性）"
      ],
      "compute_per_pilot": "以现有结果复用为主，新增实验限定为最小必要补充"
    },
    {
      "candidate_id": "cand_signal",
      "title": "Signal Audit: Revisiting Entropy, Instability, and Calibration as Revision Predictors",
      "status": "alive",
      "summary": "更窄的候选：不追求完整 benchmark 论文，而是把重点放在 revision target selection signal 的诊断比较上。",
      "hypotheses": [
        "不同 signal 在不同任务上的排序能力存在系统性差异",
        "signal ranking quality 与最终 accuracy gain 之间并非单调关系"
      ],
      "pilot_focus": "围绕 signal quality、error correlation、revision benefit correlation 的小型消融",
      "success_probability": 0.54,
      "key_perspectives": [
        "theoretical",
        "empiricist"
      ],
      "compute_per_pilot": "低"
    },
    {
      "candidate_id": "cand_guardrail",
      "title": "Boundary Guard: Syntax-Aware Safety Rails for Diffusion LM Revision on Code",
      "status": "alive",
      "summary": "更弱的边界候选：把 code fragility 当作单独边界问题，研究轻量 syntax guard 的保护作用，但不再作为主论文主线。",
      "hypotheses": [
        "syntax guard 能显著降低 parse failure",
        "syntax guard 仍无法完全恢复 reasoning-style revision 在 code 上的退化"
      ],
      "pilot_focus": "仅保留 HumanEval/MBPP 边界验证与附录材料",
      "success_probability": 0.33,
      "key_perspectives": [
        "pragmatist",
        "contrarian"
      ],
      "compute_per_pilot": "低"
    },
    {
      "candidate_id": "cand_tiger",
      "title": "TIGER: Task-Gated Instability-Guided Editing for Diffusion Language Models",
      "status": "pivoted_out",
      "summary": "作为 method-forward 主叙事已被本轮 pilot 否决：TIGER 未超过 entropy revision，且显著落后于 CORE-proxy。",
      "hypotheses": [
        "instability-guided revision beats raw entropy",
        "task gating blocks code damage without hurting reasoning"
      ],
      "pilot_focus": "已完成并触发 pivot",
      "success_probability": 0.08,
      "key_perspectives": [
        "仅保留为反例与附录材料"
      ],
      "compute_per_pilot": "N/A"
    }
  ],
  "iteration": 1,
  "selected_candidate_id": "cand_diag",
  "selection_rationale": [
    "TIGER 作为方法主线在 GSM8K shortlist 上只与 entropy revision 打平，未达到继续放大的门槛。",
    "已有证据最强的正向贡献来自 honest compute accounting、task-dependent failure analysis 与 boundary fragility。",
    "cand_diag 能最大化复用现有结果，同时让论文主张更诚实、更可防守。"
  ],
  "pivot_triggers": {
    "tiger_to_diag_a": "Instability-guided revision did not beat raw-entropy revision on GSM8K shortlist.",
    "tiger_to_diag_b": "TIGER beat Prophet and DNB but remained well below CORE-proxy under matched actual compute.",
    "tiger_to_diag_c": "Task gating only partially repaired code behavior and slightly hurt reasoning."
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

- Decision: `PIVOT`
- Selected candidate: `cand_diag`
- Confidence: `0.93`

## Why

- GSM8K shortlist failed the core TIGER method gate: `TIGER = 0.39`, `Entropy = 0.39`.
- TIGER beat `Prophet` and `DNB`, but remained materially below `CORE-proxy = 0.46`.
- Gating helped code syntax failures, but did not restore a method-forward win and slightly reduced reasoning accuracy.

## Next Actions

- Reframe the contribution as a compute-normalized diagnostic benchmark study.
- Mark `math500_transfer` as skipped by pivot, not as a missing run.
- Keep code gating as boundary/appendix evidence only.
- Let the next iteration, if approved, start from `cand_diag`.


## 上一轮 validation 结构化决策
{
  "decision": "PIVOT",
  "selected_candidate_id": "cand_diag",
  "confidence": 0.93,
  "reasons": [
    "GSM8K shortlist failed the main method criterion: TIGER tied raw-entropy revision at 0.39 instead of beating it.",
    "Although TIGER outperformed Prophet and DNB, it remained well below CORE-proxy, so the method story is not strong enough.",
    "Task gating only partially repaired code behavior: gated TIGER improved HumanEval syntax failures but still trailed the Standard code baseline and slightly hurt reasoning accuracy."
  ],
  "next_actions": [
    "Reframe the contribution as cand_diag: a compute-normalized diagnostic benchmark study of when revision helps and when it hurts.",
    "Treat math500_transfer as skipped by pivot rather than a missing experiment.",
    "Use gating results as appendix evidence for code-task fragility, not as the main claim.",
    "Prepare the next iteration around the diagnostic/benchmark framing if the quality gate approves a new cycle."
  ],
  "dropped_candidates": [
    "cand_tiger"
  ]
}