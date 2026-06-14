# 文献调研报告

**研究主题**: 遮蔽扩散语言模型的推理时计算扩展（ReMask-Retry / TTT / TCR）
**调研时间**: 2026-03-10（第六轮更新）
**arXiv 搜索关键词**: `"masked diffusion" AND "language model"`, `"inference-time compute" AND "diffusion"`, `"test-time compute" OR "inference-time scaling" AND "language model"`, `"LLaDA" OR "Dream" AND "diffusion language model"`, `"d1" AND "diffusion" AND "reasoning"`, `"survey" AND "diffusion language model"`, `ti:"remasking" OR ti:"ReMDM"`, `"MDPO" OR "ProSeCo" OR "PUMA"`, `"self-rewarding" AND "sequential monte carlo"`, `"TABES" OR "backward-on-entropy"`, `"reward-guided stitching"`, `"DCoLT" OR "CJ-GRPO"`, `"test-time scaling" AND "diffusion language model"`, `"discrete diffusion" AND "text generation"`, `ti:"reward-free guidance" AND "diffusion"`, `ti:"scaling behavior" AND "discrete diffusion"`, `ti:"MDLM" OR ti:"discrete diffusion" AND "text generation"`, `"d1" OR "diffu-GRPO" AND "diffusion language" AND "reinforcement learning"`, `"block diffusion" OR "semi-autoregressive" AND "diffusion language"`, `ti:"masked diffusion" AND ("language model" OR "text generation")`, `"test-time compute" AND ("diffusion" OR "discrete diffusion") AND "language"`, `ti:"LLaDA" OR ti:"DREAM" OR (ti:"discrete diffusion" AND ti:"reasoning")`, `"remask" OR "remasking" OR "retry" AND "diffusion" AND "language"`, `ti:"diffusion" AND (ti:"test-time" OR ti:"inference") AND ti:"language"`
**Web 搜索关键词**: `masked diffusion language model inference-time compute scaling`, `MDLM discrete diffusion state of the art`, `test-time compute scaling survey 2025`, `LLaDA reasoning benchmark`, `ReMask remasking retry inference scaling GitHub`, `RemeDi self-reflective remasking ICLR 2026`, `TReASURe DTS tree search diffusion`, `ReMDM GitHub implementation`, `diffusion language model reward-guided stitching best-of-N`, `SEDD score entropy discrete diffusion`, `Dream 7B diffusion reasoning`, `ReMDM remasking discrete diffusion inference-time scaling arxiv 2025`, `LLaDA Dream diffusion large language model 7B 8B reasoning benchmark 2025`, `diffu-GRPO reinforcement learning masked diffusion language model reasoning 2025`, `diffusion language model test-time scaling remasking retry best-of-N verification 2025 2026 github`, `Gemini Diffusion Google discrete diffusion language model 2025`, `test-time compute scaling non-autoregressive retry remasking 2025`, `discrete diffusion language model benchmark comparison autoregressive 2025 2026`, `masked diffusion language model test-time compute scaling 2025 2026 state of the art`, `LLaDA discrete diffusion language model reasoning benchmark 2025 2026`, `ReMDM remasking diffusion model test-time scaling inference compute`, `DREAM diffusion language model "test-time" OR "inference scaling" 2025 2026`, `"d1" diffusion reasoning reinforcement learning LLaDA Dream scaling 2025`, `"RFG" OR "LATTS" OR "TReASURe" OR "Prism" diffusion language model test-time scaling 2025 2026 arxiv`, `Dream 7B diffusion language model arxiv 2508.15487 benchmark results`

---

## 1. 领域现状摘要

遮蔽扩散语言模型（Masked Diffusion Language Models, MDLMs）已从 2024 年的学术探索阶段迅速发展为工业级可用的生成范式。以 MDLM（Sahoo et al., NeurIPS 2024）为起点，LLaDA 8B（NeurIPS 2025 Oral）和 Dream 7B 两大开源模型将离散扩散推进到与同规模自回归模型（如 LLaMA3 8B）可比的水平，同时具备并行解码、任意顺序生成和内建自修正的独特优势。字节跳动的 Seed Diffusion 和 NVIDIA 的 Genmol 等工业系统已在生产环境中部署 MDLM 技术。后续发展包括 LLaDA 1.5（VRPO 偏好对齐）、LLaDA-MoE（稀疏 MoE, 7B 参数仅激活 1.4B）、LLaDA 2.0（100B 规模）、LLaDA-o（多模态理解+生成）、Dream-Coder（代码生成特化）、以及多模态扩展（LLaDA-V, Dream-VL/VLA）。

推理时计算扩展（Test-Time Scaling, TTS）是当前 LLM 研究的核心方向之一。对于自回归模型，TTS 已通过 Best-of-N、树搜索、思维链等方法取得显著进展（综述见 Zhang et al., 2025; Ji et al., 2025）。然而，**将 TTS 应用于扩散语言模型**是一个新兴且快速增长的研究方向——扩散模型的迭代去噪过程天然适合推理时的计算分配与质量-速度权衡，但也面临独特挑战：token 一旦揭示即不可逆、似然函数不可解析、以及并行解码与树搜索的兼容性问题。

2025-2026 年间，该交叉领域出现了爆发式增长。**第一阶段（2025 上半年）**：ReMDM 首次证明 remasking 可实现推理时扩展；d1 将 RL 引入 dLLM 推理。**第二阶段（2025 下半年）**：Prism、UnMaskFork、TReASURe 等方法将树搜索/MCTS 适配到遮蔽扩散；RemeDi 通过自反思 remasking 在 ICLR 2026 获录用；RFG 实现奖励无关的推理引导；R3 提出 PRM 引导块 remasking；Saber 实现代码生成自适应加速；MRO 通过多奖励优化 token 相关性；Diffuse Thinking 提出 DLM-AR 协作推理。**第三阶段（2026 初）**：MDPO 用 RL 弥合训练-推理差距并提出即插即用的 RCR；CORE 通过上下文鲁棒性探测实现免训练推理时修正；Self-Rewarding SMC 引入多粒子自奖励采样；TABES/BoE 用梯度引导实现帕累托最优推理时扩展；ProSeCo 训练自校正网络实现 2-3x 加速；Reward-Guided Stitching 跨轨迹拼接最优推理步骤；BACD+TCCF 推进块扩散 TTS；ETS 提出能量引导的训练无关 RL 对齐；COVER 解决 revocable decoding 的振荡问题。理论层面，Svete & Sabharwal 证明 MDMs 与循环 Transformer 等价，Jiang et al. 证明 DLM + remasking 是最优并行采样器。这一领域正处于方法论快速迭代、基准模型不断扩大的关键时期。

当前的核心张力在于：(1) dLLM 的 likelihood 不可解析导致标准 RL/搜索方法难以应用；(2) 重遮蔽策略的设计空间尚未被充分探索；(3) 如何在不依赖外部验证器的情况下实现高效的推理时扩展仍是开放问题；(4) 信息论分析表明 remasking 启发式方法无法保证分布正确性，但验证方法开销呈指数增长。此外，Google Gemini Diffusion 在 2025 年 5 月成为首个商业级扩散语言模型（1,479 tok/s），证明离散扩散可在工业规模上运行，但其闭源性质意味着学术社区仍依赖 LLaDA/Dream 等开源模型进行研究。RL 后训练方面，GDPO 和 AGRPO 等新方法在方差控制和单步优化上取得进展，AGRPO 在 Countdown (+59.4%) 和 Sudoku (+69.7%) 上展示了极强的推理提升潜力。半自回归架构（Block Diffusion 及其变体 Diffusion-in-Diffusion）也在 2025-2026 年迅速崛起，弥合了全并行扩散和自回归之间的效率-质量鸿沟。

**第六轮更新（2026-03-10）新增重要动态**：

- **LLaDA2.1**（2026-02）引入 Token-to-Token (T2T) 编辑机制，将传统 Mask-to-Token 方案与 T2T 精化结合，实现速度模式（892 TPS on HumanEval+）和质量模式的可配置切换。更重要的是，LLaDA2.1 实现了**首个大规模 dLLM RL 框架**，基于梯度估计稳定化技术。

- **LookUM**（2025-11）提出 Lookahead Unmasking，通过不确定性度量进行路径验证，无需外部奖励模型。关键发现：base LLaDA + LookUM 可匹配 RL 后训练的 LLaDA 1.5 性能，表明不确定性验证提供了与 RL 正交的收益。

- **MEDAL**（2025-12）将 MCTS 引入 DLM 推理初始化阶段，探索最优揭示轨迹作为后续精化的起点，多基准最高提升 22%。

- **LATTS**（2025-10）将 CoT 推理从空间序列延伸重构为时间潜在精化过程，在潜在表示空间进行迭代自精化，GSM8K +4.1%, MATH +4.8%, MBPP +3.2%。

- **DLM-AR 混合架构**涌现多种范式：Planner and Executor (DDLM 规划 + ARM 执行)、Reward-Guided Stitching (扩散采样 + PRM 评分 + AR 求解)、LaDiR (VAE 潜在扩散推理)、LLMs to Diffusion Finetuning (AR 模型加装扩散能力实现 TTS)，共同指向**扩散模型与 AR 模型互补**的趋势。

- **生成顺序统一框架**：OeMDM/LoMDM 将 MDM、ARM 和块扩散统一到单一顺序表达框架中，学习上下文依赖的生成顺序。DUS 提出 dilated 分组并行揭示最小化联合熵增长，无需修改去噪器。两者均暗示**最优生成顺序是数据/任务依赖的**，固定顺序（左到右或置信度排序）均非最优。

- **推理时扩展方法论加速分化**：一侧是高成本高质量路线（UnMaskFork MCTS, MEDAL 树搜索, PG-DLM 粒子 Gibbs），另一侧是轻量训练无关路线（RFG 似然比引导, LookUM 不确定性验证, DUS dilated 调度, STaRR 动态阈值化）。**轻量路线**在实用性上更具吸引力。

---

## 2. 核心参考文献

### 2.1 基础模型与架构

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | Simple and Effective Masked Diffusion Language Models (MDLM) | arXiv 2406.07524, NeurIPS 2024 | 2024 | 提出 Rao-Blackwellized 训练目标，证明简单 masked diffusion 性能远超预期，接近 AR 困惑度 | 采样时 token 揭示后不可修改；开放文本生成质量仍有差距 |
| 2 | Discrete Diffusion Modeling by Estimating Ratios (SEDD) | ICML 2024 Best Paper, arXiv 2310.16834 | 2024 | Score entropy 损失将 score matching 推广到离散空间，困惑度比前代方法降低 25-75%；无需温度缩放即可生成高质量文本 | 采样速度仍不及 AR 模型 |
| 3 | Large Language Diffusion Models (LLaDA) | arXiv 2502.09992, NeurIPS 2025 Oral | 2025 | 8B 参数从头训练的 masked diffusion LLM；打破逆向诅咒；ICL 能力与 LLaMA3 8B 可比 | 数学/代码推理弱于同级 AR 模型；缺少自校正能力 |
| 4 | Dream 7B: Diffusion Large Language Models | arXiv 2508.15487 | 2025 | 从 AR checkpoint 初始化 + context-adaptive noise rescheduling；代码/数学/规划能力强（Countdown 16.0 vs AR 6.2）；最强开源扩散 LLM | 依赖 AR 预训练权重初始化 |
| 5 | LLaDA 1.5: Variance-Reduced Preference Optimization | arXiv 2505.19223 | 2025 | VRPO 框架解决 ELBO 高方差问题，实现 MDM 偏好对齐；GSM8K +4.7, HumanEval +3.0 | 提升幅度有限 |
| 6 | LLaDA-MoE: Sparse MoE Diffusion Language Model | arXiv 2509.24389 | 2025 | 7B 参数仅激活 1.4B，20T tokens 训练；超越 LLaDA/LLaDA 1.5/Dream | MoE 路由与扩散过程的交互尚未深入研究 |
| 7 | Dream-Coder 7B | arXiv 2509.01142 | 2025 | 代码特化扩散 LLM；emergent any-order generation；LiveCodeBench 21.4% pass@1 | 代码领域特化 |
| 8 | Scaling Beyond Masked Diffusion Language Models | arXiv 2602.15014 | 2026 | 首个离散扩散缩放定律研究；uniform-state 扩散在 GSM8K 超越 AR 和 masked diffusion；masked diffusion 可用简单 CE 目标提高 12% FLOPs 效率 | 挑战了"masked diffusion 必然最优"的假设 |
| 9 | LLaDA-o: Effective and Length-Adaptive Omni Diffusion Model | arXiv 2603.01068 | 2026 | Mixture of Diffusion (MoD) 框架解耦文本理解（离散扩散）和视觉生成（连续扩散）；数据驱动的长度自适应策略；DPG-Bench 87.04 | 聚焦多模态，非纯文本推理 |
| 9a | **LLaDA2.1: Speeding Up Text Diffusion via Token Editing** | arXiv 2602.08676 | 2026 | T2T 编辑 + M2T 联合解码方案；Speedy Mode（降 M2T 阈值 + T2T 精化）和 Quality Mode 可配置；**首个大规模 dLLM RL 框架**（稳定梯度估计）；LLaDA2.1-Flash (100B) HumanEval+ 892 TPS，33 个基准全面评测 | 两种模式的最优切换策略需人工配置 |
| 9b | **Beyond Autoregression: Discrete Diffusion for Complex Reasoning (MGDM)** | arXiv 2410.14157 | 2024 | Multi-Granularity Diffusion Modeling 按难度优先学习子目标；Countdown 91.5% vs AR 45.8%, Sudoku 100% vs 20.7%；首次在复杂推理上证明扩散模型远超 AR | 任务特定训练；未在自然语言推理基准上验证 |

### 2.2 推理时计算扩展方法（核心方向）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 10 | **Remasking Discrete Diffusion Models with Inference-Time Scaling (ReMDM)** | arXiv 2503.00307, ICML 2025 | 2025 | **本研究直接前驱**。提出 remasking 采样器（ReMDM-cap, rescale, conf, loop），从自定义后向过程推导；首次证明增加采样步数可提升 masked diffusion 文本质量至接近 AR 水平；分子设计中推进可控性 Pareto 前沿 | 仅在小模型（MDLM ~170M）验证；未探索推理密集型任务（数学/代码） |
| 11 | **RemeDi: Self-Reflective Remasking for Diffusion LMs** | arXiv 2509.23653, ICLR 2026 | 2025 | 双流 Transformer 架构：Token Prediction Stream (TPS) + Unmasking Policy Stream (UPS)；联合预测 token 和逐 token 置信度分数；SFT 教模型检测/重遮蔽错误 token + RL 优化完整轨迹；开源 DLM 中 SOTA | 需要修改模型架构（双流）；训练成本增加 |
| 12 | **MDPO: Masked Diffusion Policy Optimization** | arXiv 2508.13148 | 2025 | 将 MDLMs 去噪轨迹建模为序列决策问题，用 RL 弥合训练-推理差距；利用扩散的 Markov 性质显式训练；提出 Running Confidence Remasking (RCR) 作为即插即用推理替代方案；MATH500 +9.6%, Countdown +54.2%（60x 更少梯度更新即匹配先前 SOTA） | RL 训练成本不低；RCR 的理论保证有限 |
| 13 | **CORE: Context-Robust Remasking** | arXiv 2602.04096 | 2026 | 无需训练的推理时修正框架；通过探测 token 对遮蔽上下文扰动的敏感度识别"上下文脆弱"token，而非依赖静态置信度分数；鲁棒优化目标；LLaDA-8B-Base 上 MBPP +9.2pp | 需要额外前向传播来探测敏感度，计算开销较大 |
| 14 | **UnMaskFork: Test-Time Scaling via Deterministic Action Branching** | arXiv 2602.04344 | 2026 | 将揭示轨迹建模为搜索树 + MCTS；确定性部分揭示动作（非随机采样）；代码和数学基准上超越随机采样基线 | MCTS 开销大；需要多个 MDM 并行推理 |
| 15 | **Self-Rewarding Sequential Monte Carlo for MDLMs** | arXiv 2602.01849 | 2026 | 多粒子并行扩散过程 + 轨迹级置信度作为自奖励信号分配粒子重要性权重；迭代加权重采样引导生成朝向高质量样本；无需额外训练或奖励模型 | 粒子数与质量的扩展效率有待进一步分析 |
| 16 | **TABES: Trajectory-Aware Backward-on-Entropy Steering** | arXiv 2602.00250 | 2026 | 梯度引导推理框架；Token Influence Score (TIS) 从轨迹代价泛函一阶展开推导，单次反向传播近似无限步前瞻；ActiveQueryAttention 稀疏伴随原语降低反向传播复杂度；帕累托最优推理时扩展 | 反向传播增加延迟；稀疏近似的理论保证有限 |
| 17 | **Test-Time Scaling with DLMs via Reward-Guided Stitching** | arXiv 2602.22871 | 2026 | 扩散模型并行采样多条推理轨迹 -> PRM 打分每个中间步骤 -> 跨轨迹拼接最优步骤 -> AR 模型重算最终答案；6 个数学/代码任务平均 +23.8%；延迟比传统扩散模型降低 1.8x | 依赖外部 PRM 和 AR solver，非端到端 |
| 18 | **Learn from Your Mistakes: Self-Correcting MDMs (ProSeCo)** | arXiv 2602.11590 | 2026 | 训练模型同时执行 unmasking 和 correction；复用 MDM 去噪网络输出作为 corrector 训练输入；Progressive Self-Correction 实现 2-3x 更快采样和 ~1.3x 基准提升；同时支持推理时计算扩展 | 需要额外 corrector 训练阶段 |
| 19 | **Prism: Efficient TTS via Hierarchical Search and Self-Verification** | arXiv 2602.01842 | 2026 | 层次轨迹搜索（HTS）+ 局部 remasking 分支 + 自验证反馈（SVF，通过自评估 prompt 获取）；在 LLaDA 8B / Dream 7B / LLaDA 2.0-mini 上匹配 Best-of-N 但 NFE 更少 | 自验证质量受限于基础模型能力 |
| 20 | **TReASURe: Tree Reward-Aligned Search for Masked Diffusion** | arXiv 2509.23146 | 2025 | UnmaskBranch（单次模型调用即可生成多分支）+ ResubstituteScore（确定性重代入评分）；低方差树搜索 | 需要外部奖励模型 |
| 21 | **PG-DLM: Particle Gibbs Sampling for Diffusion LMs** | arXiv 2507.08390 | 2025 | 粒子 Gibbs 采样实现轨迹级优化；理论收敛保证；系统分析四维推理时扩展权衡；发现 scaling iterations 最优 | 计算开销显著 |
| 22 | **Generation Order and Parallel Decoding in MDMs** | arXiv 2602.00286 | 2026 | 信息论框架分析 MDMs 生成顺序和并行化偏差；Easy-First 解码随模型误差增大收益更高；并行解码的采样误差可导致任意大 Reverse KL 散度；验证可消除误差但代价呈指数；**remasking 虽高效但无法保证分布正确性** | 理论分析为主，实验限于 Block-HMM 和算术推理 |
| 23 | **ILRR: Inference-Time Steering for MDLMs** | arXiv 2601.21647 | 2026 | 学习无关的推理时控制框架，通过对齐内部激活引导生成；属性准确度提升 10-60pp，仅需一次额外并行前向传播 | 聚焦属性控制（情感等），非推理任务 |
| 24a | **RFG: Reward-Free Guidance for dLLM Reasoning** | arXiv 2509.25604 | 2025 | 将过程奖励参数化为增强/参考 dLLM 的对数似然比；无需显式奖励模型即可引导推理轨迹；4 个数学/代码基准上一致 +9.2%；训练无关通用框架 | 需要"增强模型"（RL 或 SFT 后训练的 dLLM） |
| 24b | **STaRR: Spatial-Temporal Token-Dynamics-Aware Responsive Remasking** | arXiv 2601.04205 | 2025 | 训练无关框架；基于 token 置信度的时间方差和空间偏差进行细粒度动态阈值化；平均 4.1x 加速（最高 8.9x）同时保持精度 | 聚焦加速而非质量提升 |
| 24c | **R3: Review, Remask, Refine** | arXiv 2507.08018 | 2025 | PRM 评审中间生成块 -> 低分块按比例 remask -> 模型精化目标段落；无需额外训练；适用于 LLaDA 和 BD3-LM | 依赖外部 PRM |
| 24d | **Saber: Adaptive Acceleration + Backtracking Remasking** | arXiv 2510.18165 | 2025 | 代码生成专用 DLM 采样算法；自适应加速（代码上下文越完整越快）+ 回溯 remasking 修正错误 token；Pass@1 +1.9%，推理加速 251.4% | 代码领域特化 |
| 24e | **COVER: Cache Override Verification for Efficient Revision** | arXiv 2602.06161 | 2026 | 解决 revocable decoding 中的 flip-flop 振荡问题；leave-one-out 验证 + KV cache override 在单次前向传播内完成；stability-aware 分数平衡不确定性、下游影响和缓存漂移 | 聚焦解码加速而非推理质量 |
| 24f | **Advancing Block Diffusion for TTS (BACD + TCCF)** | arXiv 2602.09555 | 2026 | 块扩散 TTS 统一框架：Bounded Adaptive Confidence Decoding (难度感知采样) + Think Coarse, Critic Fine (大块探索 + 小块精化)；TDAR-8B 上 AIME24 +11.2, 2.26x 加速 | 限于块扩散模型（TDAR） |
| 24g | **ETS: Energy-Guided Test-Time Scaling** | arXiv 2601.21484 | 2026 | 训练无关推理时方法，直接从最优 RL 策略采样；将转移概率分解为参考策略 + 能量项；在线 Monte Carlo 估计能量项，可证收敛；适用于 AR 和扩散语言模型 | 计算开销依赖 MC 采样数 |
| 24h | **Diffuse Thinking: DLM as Thought Proposer** | arXiv 2510.27469 | 2025 | 协作推理框架：DLM 并行采样多条候选推理轨迹 -> LLM 评估质量；利用扩散模型高效多样采样优势减轻 AR 推理计算负担 | 需要 AR 模型协作，非端到端 |
| 24i | **MRO: Multi-Reward Optimization** | arXiv 2510.21473 | 2025 | 定义 intra-sequence 和 inter-sequence token 相关性；通过 test-time scaling + reject sampling + RL 用多奖励直接优化 token 相关性；group step 和重要性采样减少奖励方差 | 多阶段训练复杂 |
| 24j | **LookUM: Lookahead Unmasking** | arXiv 2511.05563 | 2025 | 将采样重构为揭示路径选择问题；路径生成器 + 不确定性验证器 + 重要性采样选择最终路径；仅需 2-3 条路径即达峰值；**base LLaDA + LookUM 匹配 RL 后训练 LLaDA 1.5**，证明不确定性验证与 RL 提供正交收益 | 多路径的显存占用 |
| 24k | **MEDAL: MCTS for Diffusion LM Inference** | arXiv 2512.12168 | 2025 | 在初始化阶段用 MCTS 探索最优揭示轨迹，为后续精化提供鲁棒起点；训练无关；多基准最高 +22% | MCTS 搜索预算需调优 |
| 24l | **LATTS: Latent Space Test Time Scaling** | OpenReview | 2025 | 将 CoT 推理从空间（序列延伸）重构为时间（潜在空间迭代精化）；与传统顺序 TTS 不同，可与 MDM 并行解码兼容；GSM8K +4.1%, MATH +4.8%, MBPP +3.2% | 潜在空间操作的可解释性有限 |
| 24m | **DUS: Dilated Unmasking Scheduler** | arXiv 2506.19037 | 2025 | 推理时即插即用方法：将序列位置分成非相邻 dilated 组并行揭示，最小化联合熵增长上界；无需修改去噪器即可恢复并行揭示损失的大部分性能；GSM8K/MATH500/HumanEval/MBPP/BBH/MMLU-Pro 上超越 confidence-based planners | 分组策略固定，未根据内容自适应 |
| 24n | **LLMs to Diffusion Finetuning** | arXiv 2501.15781 | 2025 | 为 AR 预训练模型加装扩散能力的微调方法，不修改原始权重；增加扩散步数 -> 准确率单调提升（推理时计算扩展）；自适应 ODE solver 自主决定所需计算量；与传统微调方法兼容且正交 | 扩散步数增加的边际收益递减 |
| 24o | **Planner and Executor: DDLM + ARM Hybrid** | arXiv 2510.15244 | 2025 | DDLM 规划 + ARM 执行的混合架构；文本空间和潜在空间两种通信方式；潜在空间通信 DART-5 从 27.0% 提升到 54.0%，AIME24 从 0.0% 到 14.0%；64 token 规划 + 5 token 执行超越 Qwen3.1-7B（节省 44x tokens） | 两阶段推理增加系统复杂度 |
| 24p | **LaDiR: Latent Diffusion Reasoner** | arXiv 2510.04573 | 2025 | VAE 将推理步骤编码为 thought token 块的紧凑潜在表示；潜在扩散模型以块为单位去噪，支持自适应推理时计算；并行生成多样推理轨迹，全局规划和修订；数学推理和规划基准上超越 AR/扩散/潜在推理基线 | VAE 编码-解码增加额外计算 |

### 2.3 推理能力增强（RL + 推理后训练）

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 24 | **d1: Scaling Reasoning in Diffusion LLMs via RL** | arXiv 2504.12216 | 2025 | 首个 dLLM 推理 RL 框架；masked SFT + diffu-GRPO | RL 方差大；one-step 似然近似偏差大 |
| 25 | **wd1: Weighted Policy Optimization** | arXiv 2507.08838 | 2025 | 无比率策略优化；MATH500 44.2%, GSM8K 84.5%；仅 20 步 RL 训练 | 仍依赖 ELBO 近似 |
| 26 | **DCoLT: Diffusion Chain of Lateral Thought** | arXiv 2505.10446 | 2025 | 将扩散中间步骤视为潜在"思考"动作，用 outcome-based RL 优化整条推理轨迹；Unmasking Policy Module (UPM) 基于 Plackett-Luce 模型优化 unmask 顺序；LLaDA GSM8K +9.8%, MBPP +11.4%, HumanEval +19.5% | 需要 16 H800 GPU 训练 |
| 27 | **CJ-GRPO: Consistency Trajectory RL for MDLMs** | arXiv 2509.23924 | 2025 | EOS Early Rejection (EOSER) + Ascending Step-Size (ASS) 解码调度器，释放全扩散式解码潜力；Consistency Trajectory GRPO 强调 rollout 与优化轨迹一致性，减少跳步优化误差 | 限于 LLaDA-8B-Instruct 验证 |
| 28 | **DiFFPO: Training dLLMs to Reason Fast and Furious** | arXiv 2510.02212 | 2025 | 联合训练采样器/控制器 + dLLM 策略；自适应推理阈值 | 训练复杂度高 |
| 29 | **DiSPO: Diffusion-State Policy Optimization** | arXiv 2602.06462 | 2026 | 中间填充决策的直接优化 | 仅在 LLaDA-8B-Instruct 验证 |
| 30 | **SPG: Sandwiched Policy Gradient** | arXiv 2510.09541 | 2025 | 上下界夹逼真实对数似然消除策略梯度偏差 | 计算成本高于 ELBO 方法 |
| 31 | Order-Token Search | arXiv 2601.20339 | 2026 | 联合搜索生成顺序和 token 值；无需 RL 即匹配 d1-LLaDA | 搜索空间指数增长 |
| 31a | **DSFT: Diffusion SFT for Math/Logic Patterns** | arXiv 2509.18164 | 2025 | 调整 masking 策略和损失函数引导 dLLM 理解数学/逻辑模式；可与预训练和 RL 灵活结合；数学 +5-10%, 逻辑 +2% | 小规模数据验证 |
| 31b | **T\*: Progressive Block Scaling via TraceRL** | arXiv 2601.11214 | 2026 | TraceRL 训练课程实现从小块到大块的平滑过渡；AR 初始化的 MDM 逐步扩大块大小提高并行度；数学推理基准性能损失极小 | 限于块扩散场景 |

### 2.4 训练改进与架构

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 32 | **PUMA: Progressive UnMAsking** | arXiv 2602.10314 | 2026 | 对齐训练时和推理时的 masking 模式，将优化聚焦于推理对齐的 mask 上，预训练加速 ~2.5x；与 AR 初始化互补 | 仅在 125M 规模验证 |
| 33 | **Discrete Stochastic Localization (DSL)** | arXiv 2602.16169 | 2026 | SNR-invariant 去噪器跨连续噪声级别训练，桥接中间草稿噪声和 mask 端点；MAUVE 大幅提升，比 MDLM+ReMDM 少 ~4x 评估即达同等质量 | 仅在 OpenWebText 验证，推理任务未测试 |
| 34 | **On Powerful Ways to Generate: AR, Diffusion, and Beyond** | arXiv 2510.06190 | 2025 | 证明 MDM 具有计算通用性（步数匹配 PRAM 最优并行复杂度）但 any-order 生成不扩展 ARM 可解问题；提出 any-process generation（remask+insert+delete）使可解问题类显著扩大 | 理论导向 |
| 35 | Soft-Masked Diffusion Language Models | arXiv 2510.17206 | 2025 | 软遮蔽动态混合 mask embedding 与 top-k 预测 embedding | 增加推理计算量 |
| 36 | RIV: Recursive Introspection Mask Diffusion VLM | arXiv 2509.23625 | 2025 | unmask -> introspection -> remask 递归流程；检测逻辑错误 | 多模态场景 |
| 37 | Tri-Modal Masked Diffusion Model | arXiv 2602.21472 | 2026 | 首个三模态（文本+图像+音频）MDM 从头预训练 3B 参数；多模态缩放律研究；SDE 重参数化解耦物理/逻辑 batch size | 聚焦多模态，非推理扩展 |
| 37a | **Scaling Behavior of Discrete Diffusion Language Models** | arXiv 2512.10858 | 2025 | 首次系统研究不同噪声类型的 DLM 缩放行为；uniform diffusion 需更多参数但更少数据（数据受限场景有利）；DLM 需约 16x 更多计算匹配 ALM 同等 NLL；10B 参数 uniform diffusion 模型 | 缩放定律与推理时扩展的交互未研究 |
| 37b | **Loopholing Discrete Diffusion** | arXiv 2510.19304 | 2025 | 确定性潜在通路保留采样后的分布信息，打破"采样墙"；PPL 降低 61%；自条件训练策略；Countdown/Game of 24 推理改善 | 新架构，需从头训练 |
| 37c | **CADD: Continuously Augmented Discrete Diffusion** | arXiv 2510.01329 | 2025 | 连续潜在空间增强离散状态空间；mask token 用含噪但信息丰富的潜在向量表示而非空信息；模式覆盖与模式寻求的可控权衡 | 增加模型复杂度 |
| 37d | **VADD: Variational Autoencoding Discrete Diffusion** | arXiv 2505.17384 | 2025 | 潜在变量建模增强维度间关联；少步去噪时质量显著提升 | 额外的识别模型开销 |
| 37e | **On the Reasoning Abilities of Masked Diffusion Language Models** | arXiv 2510.13117 | 2025 | **理论重要**：证明 MDMs 与多项式填充的循环 Transformer (PLTs) 等价；MDMs 可解决所有 CoT-augmented Transformer 可解问题；正则语言等问题类上 MDMs 本质上比 CoT Transformer 更高效 | 理论分析，有限精度对数宽度设定 |
| 37f | **Diffusion Language Models are Provably Optimal Parallel Samplers** | arXiv 2512.25014 | 2025 | **理论重要**：证明 DLM + CoT 可用最优步数模拟任意并行采样算法；remasking 或 revision + CoT 进一步实现最优空间复杂度；revision 与 remasking 严格比无 revision 更强；为 DLM 并行采样优势和 remasking 必要性提供理论基础 | 理论导向 |
| 37g | **Constrained Discrete Diffusion (CDD)** | arXiv 2503.09790 | 2025 | 可微约束优化直接嵌入扩散采样过程；毒性控制零违规率；训练无关 | 约束优化开销；非推理任务导向 |
| 37h | **Block Diffusion: Interpolating Between AR and Diffusion** | arXiv 2503.09573, NeurIPS 2025 | 2025 | 半自回归架构：逐块生成+块内扩散去噪；KV caching + 并行 token 采样提高效率；任意长度生成；扩散模型语言建模新 SOTA | 块边界可能引入不连贯性 |
| 37i | **Diffusion in Diffusion: Reclaiming Global Coherence** | arXiv 2601.13599 | 2026 | Draft-then-refine 框架：块扩散快速生成草稿 -> 全局双向扩散精化；snapshot confidence remasking 识别关键 token；PPL 从 25.7 降至 21.9，仅用 26% 微调预算 | 两阶段流程增加延迟 |
| 37j | **Relaxing Positional Alignment in MDLMs** | arXiv 2601.22947 | 2026 | 发现 MDLMs 对位置偏移高度敏感；引入 CTC 目标的 <slack> token 放松严格位置监督；5 个开放文本生成基准一致提升 | 需要修改训练流程 |
| 37k | **Soft-Masked Diffusion Language Models (SM)** | arXiv 2510.17206, NeurIPS 2025 | 2025 | 动态混合 mask embedding 与 top-k 预测 embedding；保留前步计算信息；Dream-7B/Dream-Coder-7B 代码基准一致提升 | 增加推理时计算量 |
| 37l | **GDPO: Group Diffusion Policy Optimization** | arXiv 2510.08554 | 2025 | 半确定性 Monte Carlo 方案减少 ELBO 方差爆炸；可证更低方差的策略梯度估计；数学/推理/代码基准超越 diffu-GRPO | 确定性积分近似的局限性 |
| 37m | **AGRPO: Amortized Group Relative Policy Optimization** | arXiv 2510.04019 | 2025 | 利用 dLLM 的多步 Markov 性质优化单步去噪而非整序列；GSM8K +9.9%, MATH500 +4.6%, Countdown +59.4%, Sudoku +69.7%；4x 更快采样仅极小性能损失 | 单步优化的近似误差 |
| 37n | **TESS 2: Large-Scale Generalist Diffusion LM** | arXiv 2502.13917 | 2025 | 从 AR 模型持续预训练适配为扩散 LM；提出 reward guidance（推理时引导对齐无需重训练）；推理时计算扩展显示持续提升 | 依赖 AR 预训练 |
| 37o | **Latent Thought Models (LTMs)** | arXiv 2502.01567 | 2025 | 显式潜在思维向量 + 变分贝叶斯推理；推理时计算（迭代次数和潜在向量数）提供额外扩展维度；样本和参数效率优于 AR 和离散扩散；涌现 few-shot ICL 推理能力 | 新范式，与现有 MDM 不兼容 |
| 37p | **Improving Text Style Transfer using MDMs with Inference-time Scaling** | arXiv 2508.10995 | 2025 | 首次将 MDMs + 推理时扩展应用于文本风格迁移；verifier-based 推理时扩展 + 预训练 embedding 模型做软值验证器；MDMs 超越 AR 风格迁移基线 | 应用场景特化 |
| 37q | **FlashBlock: Attention Caching for Block Diffusion** | arXiv 2602.05305 | 2026 | 发现块扩散中块外 attention 跨步稳定；缓存块外 attention 输出减少重复计算；1.44x 吞吐量提升，1.6x attention 时间减少 | 工程优化而非方法论创新 |
| 37q2 | **OeMDM/LoMDM: Unifying MDMs with Various Generation Orders** | arXiv 2602.02112 | 2026 | Order-Expressive MDM 统一 MDM、ARM、Block Diffusion 到单一框架；LoMDM 联合学习生成顺序和扩散骨干（单一目标从头训练）；上下文依赖的生成顺序；多个语言建模基准超越各类离散扩散模型 | 从头训练成本高 |
| 37q3 | **CARD: Causal Autoregressive Diffusion** | arXiv 2601.22031 | 2026 | 在严格因果注意力掩码下实现扩散过程；soft-tailed masking + 上下文感知重加权保证优化稳定性；动态并行解码基于置信度自适应生成可变长度序列；结合 KV caching 和 ARM 级数据效率 + 并行解码延迟优势；训练延迟比块扩散降低 3x | 新架构需从头训练 |
| 37q4 | **WeDLM: Reconciling DLMs with Standard Causal Attention** | arXiv 2512.22737 | 2025 | 基于标准因果注意力的扩散解码框架，保留 prefix KV caching；Topological Reordering 保持逻辑位置；流式解码持续提交高置信 token；在 vLLM 对比下实现推理基准 ~3x 加速，低熵场景 ~10x | 需要拓扑重排序预处理 |
| 37q5 | **DFlash: Block Diffusion for Speculative Decoding** | arXiv 2602.06036 | 2026 | 轻量块扩散模型做并行 draft；条件生成基于目标模型上下文特征；lossless 加速超过 6x，比 EAGLE-3 高 2.5x | 需要训练额外 draft 模型 |
| 37q6 | **Stable-DiffCoder: Block Diffusion for Code** | arXiv 2601.15892 | 2026 | 复用 Seed-Coder 架构和数据；块扩散 CPT + block-wise clipped noise schedule；同数据同架构下整体超越 AR 对应模型；代码编辑和推理上扩散建模展现结构优势 | 代码领域特化 |
| 37q7 | **CCDD: Coevolutionary Continuous-Discrete Diffusion** | arXiv 2510.03206 | 2025 | 联合连续表示空间和离散 token 空间的多模态扩散过程；**理论证明连续扩散比离散扩散和循环 Transformer 有更强表达力**；单一模型同时在两个空间去噪；兼具连续空间的丰富语义和离散空间的可训练性 | 双空间扩散增加复杂度 |
| 37r | **Time-Annealed Perturbation Sampling (TAPS)** | arXiv 2601.22629 | 2026 | 发现 Diffusion-LMs 的时序分工：早期步决定全局语义，后期步精化局部词汇；训练无关推理策略：早期鼓励语义分支，后期减少扰动保持流畅；兼容 LLaDA 和 TraDo | 聚焦多样性而非推理质量 |
| 37s | **Gemini Diffusion (Google DeepMind)** | Google I/O 2025 | 2025 | 首个商业级扩散语言模型；1,479 tokens/sec（5x 快于可比模型）；约 16 步 reverse-diffusion 生成完整 token 块；LiveCodeBench 30.9% 超越 Gemini 2.0 Flash-Lite；内建错误修正 | 闭源；复杂推理仍弱于 AR（GPQA 40.4% vs 56.5%）|

### 2.5 综述

| 序号 | 标题 | 来源 | 年份 | 核心贡献 |
|------|------|------|------|---------|
| 38 | A Survey on Diffusion Language Models | arXiv 2508.10875 | 2025 | 最全面的 DLM 综述；GitHub: VILA-Lab/Awesome-DLMs |
| 39 | Discrete Diffusion in Large Language and Multimodal Models: A Survey | arXiv 2506.13759 | 2025 | 离散扩散专项综述：覆盖工业部署 |
| 40 | What, How, Where, and How Well? A Survey on TTS in LLMs | arXiv 2503.24235 | 2025 | 四维 TTS 分析框架 |
| 41 | A Survey of Test-Time Compute | arXiv 2501.02497 | 2025 | System-1 到 System-2 的演进框架 |

---

## 3. SOTA 方法与基准

### 3.1 主流基座模型

| 模型 | 参数量 | 类型 | 代表性能 | 代码 |
|------|--------|------|---------|------|
| SEDD | ~1B | Score Entropy Diffusion | ICML 2024 最佳论文；超越 GPT-2；32x 更少网络评估即可匹配质量 | [GitHub](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) |
| MDLM | ~170M | Masked Diffusion | LM1B 上超越所有先前扩散模型；接近 GPT-2 PPL | [GitHub](https://github.com/kuleshov-group/mdlm) |
| LLaDA 8B | 8B | Masked Diffusion | 与 LLaMA3 8B 竞争；GSM8K ~67% (SFT)；打破逆向诅咒 | [GitHub](https://github.com/ML-GSAI/LLaDA) |
| LLaDA 1.5 | 8B | MDM + VRPO | GSM8K +4.7, HumanEval +3.0, IFEval +4.0 (over SFT) | [项目页](https://ml-gsai.github.io/LLaDA-1.5-Demo/) |
| LLaDA-MoE | 7B (1.4B active) | Sparse MoE MDM | 超越 LLaDA/LLaDA 1.5/Dream；接近 Qwen2.5-3B | HuggingFace |
| LLaDA 2.0 | 100B | Masked Diffusion | 最大规模 MDM | [项目页](https://arxiv.org/pdf/2512.15745) |
| Dream 7B | 7B | Masked Diffusion | Countdown 16.0 vs AR 6.2；规划 SOTA | [GitHub](https://github.com/DreamLM/Dream) |
| Dream-Coder 7B | 7B | MDM (Code) | LiveCodeBench 21.4% pass@1 | 开源 |
| Gemini Diffusion | 未公开 | 离散扩散 (闭源) | 1,479 tok/s；LiveCodeBench 30.9% | 闭源 (Google) |
| TESS 2 | ~7B | AR 适配扩散 | 匹配/超越同级 AR；reward guidance 推理时对齐 | [GitHub](https://github.com/hamishivi/tess-2) |
| LLaDA 2.1 (Mini/Flash) | 16B / 100B | MDM + T2T Edit + RL | HumanEval+ 892 TPS; 33 基准全面评测; 首个大规模 dLLM RL | [arXiv](https://arxiv.org/abs/2602.08676) |

### 3.2 推理时扩展方法对比

| 方法 | 类型 | 需要训练 | 外部验证器 | 主要基准 | 关键结果 |
|------|------|---------|-----------|---------|---------|
| ReMDM | Remasking 采样器 | 否（即插即用） | 否 | OWT 文本生成、分子设计 | 增加步数 -> 接近 AR 质量 |
| RemeDi | 双流自反思 | 是（SFT + RL） | 否（内建 UPS） | GSM8K, MATH, 代码 | 开源 DLM SOTA |
| MDPO + RCR | RL 训练 + 即插即用 remasking | 是 (MDPO) / 否 (RCR) | 否 | MATH500, Countdown | 60x 更少梯度更新匹配 SOTA；RCR 可独立使用 |
| CORE | 上下文鲁棒 remasking | 否 | 否 | MBPP, 推理/代码 | MBPP +9.2pp；探测上下文敏感度而非静态置信度 |
| ProSeCo | 自校正训练 | 是 | 否 | 条件/无条件生成 | 2-3x 更快采样 + ~1.3x 基准提升 |
| Self-Rewarding SMC | 粒子自奖励采样 | 否 | 否 | 多种 MDM + 基准 | 无需训练或外部奖励；并行计算转化为质量 |
| TABES/BoE | 梯度引导轨迹优化 | 否 | 否 | 通用 | 帕累托最优推理时扩展前沿 |
| Reward-Guided Stitching | 跨轨迹步骤拼接 | 否（需 PRM + AR solver） | 是 | 6 个数学/代码任务 | 平均 +23.8%；延迟降低 1.8x |
| Prism | 层次搜索 + SVF | 否 | 否（自验证） | GSM8K, MATH, 代码 | 匹配 BoN, NFE 大幅减少 |
| UnMaskFork | MCTS + 确定性分支 | 否 | 是 | 代码、数学 | 超越随机采样基线 |
| TReASURe | 树搜索 + 重代入评分 | 否 | 是 | 通用 | 低方差；单次调用即可多分支 |
| PG-DLM | 粒子 Gibbs | 否 | 是 | 控制生成 | 理论收敛；scaling iterations 最优 |
| d1 (diffu-GRPO) | RL 后训练 | 是 | 否 | GSM8K, MATH500 | 显著提升推理 |
| wd1 | 加权策略优化 | 是（20步RL） | 否 | MATH500 44.2%, GSM8K 84.5% | 超越 d1, 低成本 |
| DCoLT | Outcome-based RL | 是 | 否 | GSM8K, MATH, 代码 | LLaDA GSM8K +9.8%, HumanEval +19.5% |
| CJ-GRPO | 一致性轨迹 RL | 是 | 否 | 数学、规划 | 更少解码步完成全扩散式生成 |
| DiFFPO | 联合训练采样器 | 是 | 否 | 数学、规划 | Pareto 前沿最优 |
| DiSPO | 中间状态优化 | 是 | 否 | 数学、规划 | 超越 diffu-GRPO |
| Order-Token Search | 联合顺序-token 搜索 | 否 | 否 | GSM8K, MATH | 匹配 d1 无需 RL |
| RFG | 奖励无关引导 | 否 | 否（需增强模型） | 数学/代码 4 个基准 | 一致 +9.2%；通用训练无关框架 |
| STaRR | 动态 remasking 阈值 | 否 | 否 | 通用 | 4.1-8.9x 加速 |
| R3 | PRM 引导块 remasking | 否 | 是 (PRM) | 文本生成 | 聚焦弱区域精化 |
| Saber | 自适应加速 + 回溯 | 否 | 否 | 代码生成 | Pass@1 +1.9%, 推理加速 251.4% |
| COVER | 缓存覆盖验证 | 否 | 否 | 解码加速 | 消除 flip-flop 振荡 |
| BACD+TCCF | 块扩散 TTS | 否 | 否 | AIME24, 数学 | AIME24 +11.2, 2.26x 加速 |
| ETS | 能量引导采样 | 否 | 否 | 推理/代码/科学 | 训练无关 RL 对齐；可证收敛 |
| Diffuse Thinking | DLM + AR 协作 | 否 | 是 (LLM evaluator) | 复杂推理 | DLM 生成候选，LLM 评估 |
| MRO | 多奖励优化 | 是 | 否 | 推理 | token 相关性优化 + 采样加速 |
| GDPO | 半确定性 MC 策略梯度 | 是 | 否 | 数学/推理/代码 | 可证低方差；超越 diffu-GRPO |
| AGRPO | 单步策略优化 | 是 | 否 | GSM8K, MATH500, Countdown, Sudoku | +9.9%~+69.7%；4x 更快采样 |
| Text Style Transfer w/ MDM | Verifier-based TTS | 否 | 是 (embedding verifier) | 风格迁移 | MDMs 超越 AR 基线 |
| TAPS | 时间退火扰动 | 否 | 否 | 创意写作/推理 | 训练无关多样性提升；兼容 LLaDA |
| LookUM | 路径选择 + 不确定性验证 | 否 | 否（自验证） | GSM8K, MATH, 代码 | 2-3 路径即峰值；匹配 RL 后训练效果 |
| MEDAL | MCTS 初始化 | 否 | 否 | 多基准 | 最高 +22%；训练无关；推理时扩展 |
| LATTS | 潜在空间迭代精化 | 否 | 否 | GSM8K, MATH, MBPP | +4.1%/+4.8%/+3.2%；兼容并行解码 |
| DUS | Dilated 分组并行揭示 | 否 | 否 | GSM8K, MATH, 代码, BBH, MMLU-Pro | 超越 confidence planners；无需修改模型 |
| LLM-to-Diffusion | AR 加装扩散微调 | 是（微调） | 否 | 下游任务 | 扩散步数增加 -> 准确率单调提升 |
| PUMA | 训练对齐 | 是 | 否 | 语言建模 | 预训练加速 2.5x |
| DSL | 训练对齐 | 是 | 否 | OWT (MAUVE) | 比 MDLM+ReMDM 少 4x 评估 |

### 3.3 主要评测基准

- **数学推理**: GSM8K, MATH500, AMC, AIME, Countdown
- **代码生成**: HumanEval, MBPP, LiveCodeBench, BigCodeBench, CRUXEval
- **规划**: Sudoku, Blocksworld, Trip Planning
- **文本质量**: 困惑度（PPL）, MAUVE, 生成多样性
- **对齐**: IFEval, Arena-Hard, AlpacaEval
- **控制生成**: 毒性控制, 情感控制, 语言可接受性
- **通用理解**: MMLU, HellaSwag, ARC, WinoGrande
- **分子设计**: QED, SA 等分子属性

---

## 4. 已识别的研究空白

- **空白 1: Remasking 策略的系统性比较**。ReMDM 提出了四种策略（cap, rescale, conf, loop），MDPO 提出了 RCR，RemeDi 提出了学习型双流策略，CORE 提出了上下文鲁棒性探测，Prism 提出了局部 remasking 分支，但缺少在**相同基准、相同模型**上的统一对比实验。不同策略在不同任务类型上的适用性尚不清楚。

- **空白 2: Remasking + 推理任务的深入研究**。ReMDM 主要在无条件文本生成和分子设计上验证，尚未系统探索 remasking 对数学推理、代码生成等推理密集型任务的影响。d1/wd1 等 RL 方法和 Prism/UnMaskFork 等搜索方法均独立发展，**remasking 采样器与 RL 后训练的结合**是明显空白。

- **空白 3: 自适应 remasking 预算分配**。现有方法多使用固定的 remasking 比例或步数。如何根据输入难度动态分配 remasking 计算预算（类似 AR 模型的动态 CoT 长度）尚未被充分研究。DiFFPO 做了初步探索，但限于全局阈值。

- **空白 4: Remasking 的理论理解**。arXiv 2602.00286 的信息论分析指出 remasking 启发式无法保证分布正确性，而验证方法代价呈指数。ReMDM 从自定义后向过程推导了 remasking，PG-DLM 从粒子 Gibbs 采样角度分析了轨迹级优化，但**统一理论框架**（信息论视角、与 MCMC 的精确对应、最优 remasking 比例的推导等）缺少严格分析。

- **空白 5: 轻量级 retry/remasking 机制**。现有树搜索方法计算开销大。CORE 需要额外前向传播，TABES 需要反向传播。如何设计轻量级的 retry 机制——不依赖完整树搜索、不需要外部验证器、不需修改架构、仅通过模型自身预测概率驱动 remasking——是实用性的关键。Self-Rewarding SMC 做了有益探索但仍需多粒子开销。

- **空白 6: 小模型验证**。大部分推理时扩展工作在 7B-8B 模型上进行，缺少在小模型（<1B）上系统验证推理时扩展效果的研究。PUMA 仅在 125M 验证，DSL 仅在 OWT 验证，需要更系统的小模型实验。

- **空白 7: 不同扩散族的 TTS 行为差异**。Sahoo et al. (2026) 的 scaling law 研究显示 uniform-state 扩散在 GSM8K 上优于 masked 扩散（尽管 PPL 更差），暗示 remasking 可能并非唯一的最优策略。不同扩散族在推理时扩展下的系统对比缺失。

- **空白 8: 训练-推理对齐与 remasking 的交互**。PUMA 和 DSL 从训练角度对齐训练-推理分布，MDPO 从 RL 角度弥合差距，但"先对齐训练再加 remasking"的联合效果未被研究。训练时引入 remasking 是否能使推理时 remasking 更有效？

- **空白 9: 最优生成顺序与 TTS 的交互**。OeMDM/LoMDM 证明了学习上下文依赖的生成顺序可显著提升质量，DUS 证明了 dilated 分组优于 confidence-based 顺序。但"最优生成顺序 + remasking/TTS"的联合效果未被研究——生成顺序优化与推理时计算分配是独立还是协同？

- **空白 10: 扩散-AR 混合架构的推理时计算分配**。Planner and Executor、Reward-Guided Stitching、LaDiR 等工作证明了 DLM-AR 混合的有效性，但**如何在扩散模块和 AR 模块之间最优分配推理时计算**（扩散多用几步 vs AR 多采几条？）缺乏系统研究。LookUM 的"不确定性验证与 RL 正交"这一发现暗示可能存在互补的计算分配策略。

- **空白 11: T2T 编辑与 remasking 的统一**。LLaDA2.1 的 Token-to-Token 编辑本质上是 remasking 的推广（从 mask->token 扩展到 token->token），但其与现有 remasking 策略（ReMDM, CORE, RemeDi）的理论关系和联合应用未被研究。

---

## 5. 可用资源

### 开源代码
- **ReMDM**: [kuleshov-group/remdm](https://github.com/kuleshov-group/remdm) — remasking 采样器实现（cap, rescale, conf, loop 策略），支持 MDLM
- **MDLM**: [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm) — NeurIPS 2024，基础 MDM 训练和采样
- **SEDD**: [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) — ICML 2024 最佳论文
- **LLaDA**: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) — 8B masked diffusion LLM
- **Dream 7B**: [DreamLM/Dream](https://github.com/DreamLM/Dream) — 最强开源扩散 LLM
- **MDPO**: [autonomousvision/mdpo](https://github.com/autonomousvision/mdpo) — RL + RCR remasking
- **RemeDi**: [maple-research-lab/RemeDi](https://github.com/maple-research-lab/RemeDi) — ICLR 2026，自反思 remasking
- **Self-Rewarding SMC**: [Algolzw/self-rewarding-smc](https://github.com/Algolzw/self-rewarding-smc) — 粒子采样方法
- **Reward-Guided Stitching**: [roymiles/diffusion-stitching](https://github.com/roymiles/diffusion-stitching) — 跨轨迹拼接
- **PUMA**: [JaeyeonKim01/PUMA](https://github.com/JaeyeonKim01/PUMA) — 渐进式 unmasking 训练
- **CJ-GRPO**: [yjyddq/EOSER-ASS-RL](https://github.com/yjyddq/EOSER-ASS-RL) — 一致性轨迹 RL
- **Prism**: [viiika/Prism](https://github.com/viiika/Prism) — 层次搜索 + 自验证
- **d1**: [dllm-reasoning.github.io](https://dllm-reasoning.github.io/) — diffu-GRPO 训练
- **Scaling Beyond MDLMs**: [s-sahoo.github.io/scaling-dllms](http://s-sahoo.github.io/scaling-dllms) — 缩放律研究
- **Awesome-DLMs**: [VILA-Lab/Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) — 扩散语言模型论文集合
- **RFG**: [项目页](https://arxiv.org/abs/2509.25604) — 奖励无关引导
- **Saber**: [代码](https://github.com/Algolzw/self-rewarding-smc) — 代码生成自适应采样
- **DSFT**: [匿名代码](https://anonymous.4open.science/r/DSFT-0FFB/) — 扩散 SFT 策略
- **ETS**: [项目页](https://arxiv.org/abs/2601.21484) — 能量引导推理时扩展
- **Diffusion Stitching**: [roymiles/diffusion-stitching](https://github.com/roymiles/diffusion-stitching) — 奖励引导跨轨迹拼接
- **Awesome-Inference-Time-Scaling**: [ThreeSR/Awesome-Inference-Time-Scaling](https://github.com/ThreeSR/Awesome-Inference-Time-Scaling) — 推理时扩展论文集合
- **Block Diffusion**: [m-arriola.com/bd3lms](https://m-arriola.com/bd3lms) — 半自回归扩散语言模型
- **TESS 2**: [hamishivi/tess-2](https://github.com/hamishivi/tess-2) — AR 适配扩散 LM + reward guidance
- **Soft-Masked DLMs**: [IBM/soft-masked-diffusion-language-models](https://github.com/IBM/soft-masked-diffusion-language-models) — 软遮蔽扩散
- **LLaDA-o**: [ML-GSAI/LLaDA-o](https://github.com/ML-GSAI/LLaDA-o) — 多模态 omni 扩散模型
- **Gemini Diffusion**: [deepmind.google/models/gemini-diffusion](https://deepmind.google/models/gemini-diffusion/) — Google 闭源扩散 LM
- **LLaDA2.1**: [arXiv](https://arxiv.org/abs/2602.08676) — T2T + RL 大规模 dLLM
- **MEDAL**: [arXiv](https://arxiv.org/abs/2512.12168) — MCTS 初始化 DLM 推理
- **LookUM**: [arXiv](https://arxiv.org/abs/2511.05563) — Lookahead 不确定性验证
- **OeMDM/LoMDM**: [arXiv](https://arxiv.org/abs/2602.02112) — 统一生成顺序框架
- **CARD**: [arXiv](https://arxiv.org/abs/2601.22031) — 因果自回归扩散
- **WeDLM**: [arXiv](https://arxiv.org/abs/2512.22737) — 因果注意力扩散解码
- **DFlash**: [arXiv](https://arxiv.org/abs/2602.06036) — 块扩散投机解码
- **Stable-DiffCoder**: [arXiv](https://arxiv.org/abs/2601.15892) — 块扩散代码模型
- **CCDD**: [arXiv](https://arxiv.org/abs/2510.03206) — 连续-离散联合扩散
- **DUS**: [arXiv](https://arxiv.org/abs/2506.19037) — Dilated Unmasking Scheduler
- **MGDM**: [GitHub](https://github.com/HKUNLP/diffusion-vs-ar) — 多粒度扩散建模
- **dInfer**: [GitHub](https://github.com/inclusionAI/dInfer) — 高效 dLLM 推理框架，LLaDA-MoE 上 >1100 TPS
- **Planner and Executor**: [arXiv](https://arxiv.org/abs/2510.15244) — DDLM+ARM 混合推理
- **LaDiR**: [arXiv](https://arxiv.org/abs/2510.04573) — 潜在扩散推理器
- **Awesome-Inference-Time-Scaling**: [GitHub](https://github.com/Dereck0602/Awesome_Test_Time_LLMs) — 推理时扩展论文集

### 数据集
- **GSM8K / MATH500 / AMC / AIME**: 数学推理标准基准
- **HumanEval / MBPP / LiveCodeBench**: 代码生成
- **Countdown / Sudoku**: 规划与约束满足
- **OpenWebText / LM1B**: 语言建模困惑度评测

### 预训练模型
- **LLaDA-8B / LLaDA-8B-Instruct**: HuggingFace (BAAI/GSAI)
- **LLaDA 1.5**: HuggingFace (GSAI lab)
- **LLaDA-MoE-7B-A1B-Instruct**: HuggingFace
- **Dream-7B / Dream-7B-Instruct**: HuggingFace (DreamLM)
- **Dream-Coder-7B / Dream-Coder-7B-Instruct**: HuggingFace
- **MDLM checkpoints**: GitHub kuleshov-group
- **Qwen2-0.5B / Qwen2.5-7B**: 可用作小规模实验基座或 AR 初始化源

---

## 6. 对 Idea 生成的启示

### 值得探索的方向

1. **ReMask-Retry 统一框架**：将 ReMDM 的 remasking 机制与 retry/self-correction 语义统一，设计一种"生成-评估-选择性回退-重生成"的推理时扩展范式。这结合了 ReMDM 的理论基础和 RemeDi 的自反思思想，但**不修改模型架构**（与 RemeDi 不同），而是利用模型自身的预测概率作为免费的置信度信号。CORE 的上下文扰动探测和 Self-Rewarding SMC 的自奖励思想可进一步增强。

2. **置信度驱动的自适应 remasking**：利用 MDM 本身的预测概率（softmax 分布的熵或 top-1 概率）作为零成本置信度信号，决定哪些 token 需要 remask。这比 RemeDi 更轻量（无需训练 UPS），比 ReMDM 更灵活（动态适应输入难度），比 CORE 更高效（无需额外前向传播探测）。MDPO 的 RCR 走了这个方向但使用的是运行时置信度而非自适应预算。

3. **Remasking + RL 后训练协同**：MDPO 分别提出了 RL 训练和 RCR remasking，DCoLT 提出了 UPM 优化 unmask 顺序，但"先 RL 训练再 remasking 采样"的联合效果未被系统研究。加上 PUMA/DSL 的训练对齐，形成"训练对齐 -> RL 优化 -> remasking 采样"三阶段协同方案。

4. **推理时 Pareto 前沿优化**：系统研究 remasking 步数/比例 vs 生成质量的 Pareto 前沿（参考 PG-DLM 的四维分析和 DiFFPO 的联合采样器训练），寻找不同任务的最优计算-质量权衡点。

5. **小模型实验平台**：在 MDLM 规模（~170M）上系统验证各种 remasking 策略对推理任务的效果，建立可复现的实验基线，再 scale up 到 LLaDA/Dream。DSL 已证明训练改进在小模型上可行。

### 已被充分研究的方向（谨慎进入）

- 单纯的 Best-of-N 采样（已有大量工作以此为 baseline）
- 在 LLaDA 上做标准 SFT/偏好对齐（LLaDA 1.5 已覆盖）
- 完整树搜索框架（Prism, UnMaskFork, TReASURe, DTS 已覆盖多种思路）
- 纯 RL 似然近似改进（diffu-GRPO, wd1, SPG, DiSPO, DCoLT, CJ-GRPO 已覆盖多种方案）
- 静态 confidence 阈值的 remasking（CORE 和信息论分析已指出根本缺陷）

6. **潜在空间推理时扩展**：LATTS 将 CoT 从序列空间重构为潜在表示的时间迭代精化，LaDiR 用 VAE 编码推理步骤为紧凑潜在表示，CCDD 证明连续空间比离散空间有更强表达力。这三者共同指向一个方向：**在连续潜在空间而非离散 token 空间进行推理时扩展**，可能更自然地利用扩散模型的迭代特性。

7. **生成顺序优化作为免费的 TTS**：OeMDM/LoMDM 和 DUS 表明学习/优化生成顺序本身就能显著提升质量，且**不增加总步数**（只是改变步的分配方式）。这是一种"零额外计算"的推理时扩展，值得与 remasking 结合。

8. **不确定性驱动的轻量 TTS**：LookUM 的核心发现（2-3 条路径即达峰值、匹配 RL 效果）暗示**高效的不确定性量化比暴力搜索更重要**。这与 CORE 的上下文扰动探测和 TABES 的轨迹代价泛函方向一致，但 LookUM 更轻量。

### 跨域借鉴潜力

- **MCMC/退火采样** -> remasking 可视为 Gibbs 采样的一种形式（PG-DLM 已初步探索），可借鉴退火温度调度策略
- **连续扩散图像模型的推理时扩展**（CVPR 2025 EvoSearch; Flow Matching inference-time scaling）-> 连续扩散的 verifier/guidance 方法可能可适配到离散情形
- **AR 模型的 process reward model (PRM)** -> 为 dLLM 的中间去噪状态设计过程奖励，指导 remasking 决策（Reward-Guided Stitching 已初步探索但依赖 AR PRM）
- **蛋白质折叠中的迭代精化**（如 AlphaFold2 的 recycling）可为 MDLMs 的 remasking 循环提供结构化先验
- **RIV 的 Introspection Training** -> 虽然是多模态工作，但其"内省检测逻辑错误"的训练范式可迁移到纯文本 dLLM 的 remasking 决策
- **RFG 的对数似然比引导** -> 用增强/参考模型的似然比代替显式奖励模型，可低成本实现推理引导；与 remasking 结合可实现"引导哪些 token 需要 remask"
- **Loopholing 的信息保留** -> 采样后通过确定性通路保留分布信息的思想可启发 remasking 中的"软回退"策略
- **CDD 的约束嵌入** -> 将硬约束直接嵌入扩散采样过程的方法可用于数学推理中的格式/逻辑约束
- **R3 的 PRM 分块精化** -> PRM 评分 + 按比例 remask 的思路简单有效，可与 ReMDM 的 remasking 采样器深度结合

---

*注: 本报告基于 arXiv 论文搜索和 Web 搜索综合整理，涵盖 2023-2026 年 95+ 篇核心文献。第六轮更新（2026-03-10）新增：LLaDA2.1（T2T 编辑 + 首个大规模 dLLM RL, 100B, 892 TPS）、LookUM（Lookahead 不确定性验证, base 匹配 RL 后训练效果）、MEDAL（MCTS 推理初始化, +22%）、LATTS（潜在空间推理时扩展）、DUS（Dilated Unmasking Scheduler）、OeMDM/LoMDM（统一生成顺序框架）、CARD（因果自回归扩散）、WeDLM（因果注意力扩散, vLLM 对比 3-10x 加速）、DFlash（块扩散投机解码, 6x lossless 加速）、Stable-DiffCoder（块扩散代码模型超越 AR）、CCDD（连续-离散联合扩散, 理论更强表达力）、MGDM（多粒度扩散推理, Countdown 91.5% vs AR 45.8%）、Planner and Executor（DDLM+ARM 混合, 44x token 节省）、LaDiR（潜在扩散推理器）、LLMs to Diffusion Finetuning（AR 加装扩散 TTS）、dInfer（高效推理框架, >1100 TPS）等 15+ 篇新文献。新增研究空白 9-11（生成顺序+TTS 交互、DLM-AR 混合计算分配、T2T 编辑与 remasking 统一）。*
