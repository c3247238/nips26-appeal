# 文献调研报告

**研究主题**: 遮蔽扩散语言模型的推理时计算扩展（ReMask-Retry / TTT / TCR）
**调研时间**: 2026-03-10（第五轮迭代，重新聚焦 TTT × DLM 融合）
**arXiv 搜索关键词**: `"masked diffusion language model" AND ("test-time" OR "inference-time" OR "remasking")`, `"discrete diffusion" AND ("scaling" OR "compute" OR "sampling") AND "language"`, `"test-time training" AND ("language model" OR "LLM" OR "memorize")`, `"diffusion language model" AND ("reasoning" OR "benchmark" OR "math" OR "code")`, `"diffusion language" AND ("reward" OR "guidance" OR "alignment" OR "verifier") AND "inference"`, `ti:"Titans" AND "memorize" AND "test time"`
**Web 搜索关键词**: `masked diffusion language model test-time compute scaling 2025 2026 state of the art`, `Dream 7B MDLM LLaDA discrete diffusion language model benchmark evaluation 2025`, `test-time training TTT language model memory Titans 2025 2026`, `MetaState persistent working memory discrete diffusion language model 2026`, `MDPO running confidence remasking RCR masked diffusion policy optimization 2025`, `LLaDA 2 100B diffusion language model scaling 2026`, `TReASURe tree search test-time alignment masked diffusion language model 2025`, `discrete stochastic localization DSL remasking ReMDM compute efficiency 2026`, `CoRe context robust remasking diffusion language model arxiv 2602.04096`

---

## 1. 领域现状摘要

遮蔽扩散语言模型（Masked Diffusion Language Models, MDLMs）已从 2024 年的学术探索迅速发展为具有工业竞争力的生成范式。当前格局以三个核心模型为锚点：**MDLM**（Sahoo et al., NeurIPS 2024）奠定了基础训练目标，**LLaDA 8B**（NeurIPS 2025 Oral）证明了从头训练的可行性，**Dream 7B** 通过 AR checkpoint 初始化在规划任务上展现了独特优势（Countdown 16.0 vs AR 6.2）。2025 下半年至 2026 初，该领域经历了爆发式增长——**LLaDA 2.0** 将参数量推至 100B（MoE 架构），**LLaDA 2.1** 引入 Token-to-Token 编辑机制实现 892 TPS 的极速推理。

**推理时计算扩展（Test-Time Scaling, TTS）**是当前 DLM 研究最活跃的方向。扩散模型的迭代去噪过程天然适合推理时计算分配，但也面临独特挑战：token 一旦揭示即不可逆、似然函数不可解析、并行解码与树搜索的兼容性问题。2025-2026 年间涌现了大量方法：从 ReMDM 的 principled remasking，到 MDPO/RCR 的训练-推理差距弥合，到 TReASURe/UnMaskFork/Prism 的树搜索/MCTS 适配，到 Self-Rewarding SMC 的多粒子采样，到 DSL 的训练改进实现 4x remasking 效率提升。

**本轮迭代的核心新发现**：**MetaState**（2026-03-02，arXiv 2603.01331）首次将持久化工作记忆引入 DLM，通过跨去噪步骤的 GRU 记忆模块解决"信息孤岛"（Information Island）问题——标准 dLLM 每步仅条件化于当前 hard-masked 序列，中间连续表示在采样后丢弃。MetaState 由三个可训练模块组成：Cross-Attention Mixer（读取骨干激活到记忆槽）、GRU-style Updater（跨步整合信息）、Cross-Attention Injector（将更新后的记忆注入骨干激活）。在 LLaDA-8B 和 Dream-7B 上，MetaState 保持骨干冻结，仅引入极少可训练参数即持续提升准确率。**这与我们探索的 TTT × DLM 融合方向高度契合，但 MetaState 使用固定更新规则（GRU），而非 TTT 的核心思想（自监督梯度步更新）。**

同时，**TTT 领域**也在快速发展——TTT-E2E 实现了 128K 上下文的等效全注意力性能（2.7x 加速），Titans 提出了具备 momentum + weight decay 遗忘机制的长期记忆模块（>2M 上下文），TTT Done Right（LaCT）将 chunk 更新扩展到百万 token 级别并首次应用于 14B AR 视频扩散模型，SR-TTT 通过 surprisal-aware 选择性路由解决了 TTT 的精确召回难题。**这两条线的交汇——将 TTT 的自监督权重更新机制引入 DLM 的迭代去噪过程——是一个尚未被探索的重要研究空白。**

---

## 2. 核心参考文献

### 2.1 基础模型与架构

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | MDLM: Simple and Effective Masked Diffusion Language Models | arXiv 2406.07524, NeurIPS 2024 | 2024 | Rao-Blackwellized 训练目标，简单 masked diffusion 接近 AR 困惑度 | token 揭示后不可修改 |
| 2 | LLaDA: Large Language Diffusion Models | arXiv 2502.09992, NeurIPS 2025 Oral | 2025 | 8B masked diffusion LLM，打破逆向诅咒，ICL 能力与 LLaMA3 8B 可比 | 数学/代码推理弱于同级 AR |
| 3 | Dream 7B: Diffusion Large Language Models | arXiv 2508.15487 | 2025 | AR 初始化 + context-adaptive noise rescheduling；规划任务远超 AR（Sudoku 81.0 vs 21.0） | 依赖 AR 预训练权重 |
| 4 | LLaDA 2.0: Scaling Up Diffusion Language Models to 100B | arXiv 2512.15745 | 2025 | 首个 100B dLLM（MoE），3 阶段 block-level WSD 训练，535 TPS 推理速度 | 闭源训练细节 |
| 5 | dLLM: Simple Diffusion Language Modeling | arXiv 2602.22661 | 2026 | 统一框架：训练、推理、评估标准化，支持 LLaDA/Dream 复现和微调 | 框架工具，非新方法 |
| 6 | Scaling Beyond Masked Diffusion Language Models | arXiv 2602.15014 | 2026 | 首个 DLM scaling law 研究；uniform diffusion 在 GSM8K 上超越 masked 和 AR；PPL 跨算法比较有误导性 | 最大仅 1.7B 参数 |
| 7 | Scaling Behavior of Discrete Diffusion Language Models | arXiv 2512.10858 | 2025 | uniform diffusion 在数据受限场景更优；首个 10B uniform diffusion 模型 | 未覆盖推理任务 |

### 2.2 推理时计算扩展：Remasking 与采样策略

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 8 | ReMDM: Remasking Discrete Diffusion Models with Inference-Time Scaling | arXiv 2503.00307 | 2025 | 首次证明 principled remasking 实现推理时扩展；更多步数接近 AR 质量 | 采样速度慢 |
| 9 | MDPO + RCR: Masked Diffusion Policy Optimization | arXiv 2508.13148 | 2025 | RL 弥合训练-推理差距（60x 更少梯度更新）；RCR 即插即用追踪 per-position confidence | 需要额外 RL 训练 |
| 10 | DSL: Discrete Stochastic Localization | arXiv 2602.16169 | 2026 | SNR 不变去噪器统一连续-离散 corruption；ReMDM 采样效率 4x 提升，MAUVE 显著提升 | 需要微调 |
| 11 | CoRe: Context-Robust Remasking | arXiv 2602.04096 | 2026 | 上下文扰动探测 token 脆弱性，免训练推理修正；MBPP +9.2% | 额外前向传播开销 |
| 12 | Self-Rewarding SMC for MDLMs | arXiv 2602.01849 | 2026 | 多粒子并行采样 + 轨迹级自奖励重采样，无需外部 reward | 粒子数增加计算线性增长 |
| 13 | Where-to-Unmask: Ground-Truth-Guided Unmasking Order | arXiv 2602.09501 | 2026 | Gt-Margin oracle 揭示顺序显著提升推理；学习 supervised unmasking planner | 需要 ground-truth 训练 |
| 14 | COVER: Cache Override Verification for Efficient Revision | arXiv 2602.06161 | 2026 | 解决 revocable decoding flip-flop 振荡；KV cache 单次前向验证 | 特化于并行解码 |

### 2.3 推理时计算扩展：树搜索与高级采样

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 15 | UnMaskFork: Test-Time Scaling via MCTS | arXiv 2602.04344 | 2026 | MCTS 树搜索适配 MDLM；确定性 partial unmasking 构建搜索树；代码/数学强 | 计算密集 |
| 16 | TReASURe: Tree Reward-Aligned Search for MDLMs | arXiv 2509.23146 | 2025 | UnmaskBranch 多样化分支 + ResubstituteScore 低方差评分；低 NFE 下 SOTA | 需要外部 reward |
| 17 | Prism: Hierarchical Search + Self-Verification for dLLMs | arXiv 2602.01842 | 2026 | HTS 动态剪枝 + 局部分支 + SVF 自验证（无需外部验证器）；匹配 BoN 性能，更少 NFE | 搜索空间受限 |
| 18 | PG-DLM: Particle Gibbs Sampling | arXiv 2507.08390 | 2025 | 粒子 Gibbs 轨迹级精化 + 理论收敛保证；scaling iterations 最优 reward-PPL trade-off | 计算开销大 |
| 19 | Reward-Guided Stitching | arXiv 2602.22871 | 2026 | 扩散采样 + PRM 步级评分 + AR 求解器拼接；6 任务 +23.8%，1.8x 延迟降低 | 需 PRM + AR 模型 |
| 20 | ETS: Energy-Guided Test-Time Scaling | arXiv 2601.21484 | 2026 | 训练无关 RL 对齐；在线 Monte Carlo 能量估计；跨 MLM/DLM/AR 通用 | 能量估计方差 |
| 21 | Inference-Time Scaling via Importance Weighting | arXiv 2505.22524 | 2025 | SMC + 最优 proposal 设计；跨域通用（文本/蛋白质/图像） | 最优 proposal 需训练 |

### 2.4 DLM 推理效率与控制

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 22 | DyLLM: Saliency-based Token Selection | arXiv 2603.08026 | 2026 | 训练无关；选择性计算 salient tokens，最高 9.6x 吞吐量 | 可能丢失低显著性重要 token |
| 23 | NAP: Non-Autoregressive Parallel DLMs | arXiv 2602.23225 | 2026 | 诊断 DLM 退化为 AR-like 解码 = 训练数据顺序结构导致；数据中心方案 | 概念验证 |
| 24 | Diffusion LMs Are Natively Length-Aware | arXiv 2603.06123 | 2026 | 零样本上下文窗口裁剪，显著 FLOPs 节省 | 长输出效果有限 |
| 25 | Free Lunch for Pass@k: Diverse Sampling | arXiv 2603.04893 | 2026 | 批量特征空间排斥提升 pass@k，训练无关 | 仅 pass@k 场景 |
| 26 | Activation Steering for MDLMs | arXiv 2512.24143 | 2025 | 表示空间干预；安全方向跨语言迁移 | 仅安全性验证 |
| 27 | ILRR: Inference-Time Steering for DLMs | arXiv 2601.21647 | 2026 | 参考序列激活对齐；属性准确率 +10-60% | 需要参考序列 |
| 28 | Style Transfer via MDMs with Inference-Time Scaling | arXiv 2508.10995 | 2025 | 验证器引导 TTS 用于风格迁移；MDM 优于 AR | 任务特化 |
| 29 | Skip to the Good Part: Layer Skipping in DLM vs AR | arXiv 2603.07475 | 2026 | DLM 表示更层次化、早期层冗余多；native dLLM 可跳过 18.75% 层保持 90% 性能 | AR 初始化 DLM 保留 AR 动态 |

### 2.5 TTT（Test-Time Training）与记忆机制 — 核心关注

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| **30** | **MetaState: Persistent Working Memory for dLLMs** | **arXiv 2603.01331** | **2026** | **首个 DLM 跨步记忆**：GRU Updater + Cross-Attn Mixer/Injector；冻结骨干 + 轻量可训练参数；K-step unrolling 训练；LLaDA-8B/Dream-7B 上一致提升 | 固定 GRU 更新（非自适应梯度），仅跨去噪步骤不跨序列 |
| **31** | **Learning to (Learn at Test Time): RNNs with Expressive Hidden States** | **arXiv 2407.04620** | **2024** | **TTT 范式奠基**：隐状态 = ML 模型，更新规则 = 自监督梯度步；TTT-Linear/TTT-MLP；125M-1.3B 参数验证 | 内存 I/O 瓶颈 |
| **32** | **Titans: Learning to Memorize at Test Time** | **arXiv 2501.00663** | **2025** | 神经长期记忆 + 注意力短期记忆；momentum + weight decay 遗忘；>2M 上下文；短期+长期分离 | 训练复杂度高 |
| **33** | **End-to-End TTT for Long Context (TTT-E2E)** | **arXiv 2512.23675** | **2025** | 端到端 TTT：推理时 NTP + 训练时 meta-learning；匹配全注意力，128K 2.7x 加速，2M 35x 加速 | 仅验证于 AR 模型 |
| **34** | **TTT Done Right (LaCT)** | **arXiv 2505.23884** | **2025** | 超大 chunk 更新（2K-1M tokens）；非线性 state 达模型 40% 参数；14B AR 视频扩散模型验证 | 尚未应用于语言 DLM |
| **35** | **SR-TTT: Surprisal-Aware Residual TTT** | **arXiv 2603.06642** | **2026** | 损失门控稀疏记忆解决 TTT 精确召回问题；高 surprisal token 路由到残差精确注意力 cache | 额外记忆开销 |
| 36 | VDS-TTT: Verifier-Driven Sample Selection for TTT | arXiv 2505.19475 | 2025 | 验证器驱动伪标签选择 + LoRA 适配；最高 +32.29% | 需训练验证器 |
| 37 | TTT-Discover: Learning to Discover at Test Time | arXiv 2601.16175 | 2026 | RL + TTT 科学发现；GPU kernel 2x 优化 | 专注于优化/搜索 |
| 38 | Local Mixtures of Experts: Essentially Free TTT via Model Merging | arXiv 2505.14136 | 2025 | TTMM 用 model merging 近似 TTT，100x 更快；性能随 expert 数量提升 | 需预训练多个 expert |
| 39 | TTT for Few-Shot Learning | arXiv 2411.07279 | 2024 | ARC 上 TTT 比 fine-tuned baseline 高 6x；8B 模型 53% 匹配人类平均 | 需要任务 examples |

### 2.6 DLM 推理能力分析

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 40 | Reasoning or Rationalization? in MDLMs | arXiv 2603.01190 | 2026 | MDLMs 先收敛 verdict 再 rationalize；强制 reasoning-first 反降性能（86.2→71.9%） | 仅事实验证任务 |
| 41 | Adaptation to Intrinsic Dependence in DLMs | arXiv 2602.20126 | 2026 | 分布无关随机 unmasking schedule；收敛速率 ∝ 数据总相关性 | 理论导向 |
| 42 | CCDD: Coevolutionary Continuous Discrete Diffusion | arXiv 2510.03206 | 2025 | 证明连续扩散表达力 > 离散扩散和 looped Transformer；联合连续-离散空间 | 训练复杂 |
| 43 | Double Descent in AR vs Discrete Diffusion | arXiv 2509.24974 | 2025 | DLM 需要更大容量和更多 epoch 才达到插值阈值；足够容量后与 AR 相当 | 小规模实验 |

---

## 3. SOTA 方法与基准

### 3.1 当前最强模型

| 模型 | 参数量 | 类型 | 关键优势 | 可用性 |
|------|--------|------|---------|--------|
| LLaDA 2.0-flash | 100B (MoE) | Masked Diffusion | 代码/数学/工具使用竞争力，535 TPS | HuggingFace |
| Dream 7B Instruct | 7B | Masked Diffusion | 规划任务 SOTA（Countdown 16.0, Sudoku 81.0） | GitHub/HuggingFace |
| LLaDA 8B Instruct | 8B | Masked Diffusion | 通用基准可比 LLaMA3 8B | HuggingFace |
| LLaDA 2.0-mini | 16B | Masked Diffusion | 密集架构，全面可比 AR | HuggingFace |

### 3.2 推理时扩展方法对比

| 方法 | 类别 | 需训练 | 需外部 reward | 典型提升 | 计算开销 |
|------|------|--------|--------------|---------|---------|
| ReMDM | Remasking | 否 | 否 | PPL 接近 AR | 步数线性 |
| MDPO+RCR | RL+Remasking | 是/否 | 否 | MATH +9.6%, Countdown +54.2% | RCR 近零 |
| DSL | 训练改进 | 是(微调) | 否 | MAUVE 4x 更少步数 | 训练成本 |
| UnMaskFork | MCTS | 否 | 可选 | 代码/数学强 | 多 GPU |
| TReASURe | 树搜索 | 否 | 是 | 低 NFE SOTA | reward 模型 |
| Prism | 层次搜索 | 否 | 否(自验证) | 匹配 BoN | 更少 NFE |
| Self-Rewarding SMC | 粒子采样 | 否 | 否 | 显著质量提升 | 粒子数线性 |
| **MetaState** | **跨步记忆** | **是(轻量)** | 否 | 一致性提升 | 可忽略参数 |
| CoRe | 上下文鲁棒 | 否 | 否 | MBPP +9.2% | 额外前向 |
| Stitching | DLM+AR 混合 | 否 | 是(PRM) | 6 任务 +23.8% | AR 求解器 |

### 3.3 主流评测基准

- **推理**: GSM8K, MATH500, ARC-E/C, Countdown, Sudoku, Trip Planning
- **代码**: HumanEval(+), MBPP(+), LiveCodeBench
- **通用**: MMLU, HellaSwag, WinoGrande, PIQA
- **生成质量**: PPL (GPT-2 评估), MAUVE, Self-BLEU, Distinct-N
- **对齐/控制**: 情感控制, 毒性控制, CoLA 语言可接受性

---

## 4. 已识别的研究空白

### 空白 1（核心）: TTT 自监督权重更新 × DLM 迭代去噪 = 未探索的交叉点

MetaState 证明跨步记忆对 DLM 有效，但使用固定 GRU 更新规则。TTT 的核心创新——**隐状态是可学习模型，更新规则是自监督梯度步**——完全未被引入 DLM。DLM 的多步去噪天然提供"多次观察同一数据"的机会：每步新揭示的 token 可作为自监督信号更新"快权重"，使模型在去噪过程中逐步积累序列理解。这一方向在已有文献中**零覆盖**。

**具体差距**:
- MetaState: 固定 GRU 更新 → 缺乏自适应性
- TTT-Linear/MLP: 自监督梯度步更新 → 未用于 DLM
- Titans: momentum + weight decay 遗忘 → 未用于非 AR 架构
- TTT-E2E: 端到端 meta-learning → 仅验证于 AR 滑动窗口

### 空白 2: DLM 的"信息孤岛"问题仅有 MetaState 一个解决方案

MetaState（2026-03-02）是目前唯一直接处理此问题的工作。更强大的记忆机制均未被尝试：
- Titans 的 momentum + weight decay 遗忘
- TTT-E2E 的端到端 meta-learning
- SR-TTT 的 surprisal-aware 选择性记忆
- LaCT 的超大 chunk 更新（非线性 state 达模型 40%）

### 空白 3: 跨步记忆 + 推理时扩展的联合优化

MetaState 提供跨步记忆，各种 TTS 方法提供搜索/优化，但两者**尚未联合**。跨步记忆可为 TTS 搜索/评分提供更好的信息基础（减少重复计算），TTS 多轨迹探索可为记忆模块提供更丰富训练信号。

### 空白 4: DLM 推理能力的根本限制

"Reasoning or Rationalization?" 揭示 MDLMs 先收敛 verdict 后 rationalize，强制 reasoning-first 反而降低性能。NAP 发现 DLM 退化为 AR-like 解码源于训练数据顺序结构。这暗示 DLM 的推理提升可能需要**根本不同的记忆/状态机制**，而非简单的推理时计算扩展。TTT 层的自适应权重更新可能是突破口。

### 空白 5: DLM 中的遗忘与选择性记忆

TTT 模型面临的关键问题是早期信息被后续梯度更新冲刷（SR-TTT 发现的 catastrophic forgetting）。DLM 的 remasking 过程本质上就是一种"遗忘再重建"机制。如何设计 DLM-specific 的记忆遗忘策略（例如利用 remasking 概率调控记忆权重衰减）是未解问题。

---

## 5. 可用资源

### 5.1 开源代码

- **ReMDM**: https://github.com/kuleshov-group/remdm
- **MDPO + RCR**: https://github.com/autonomousvision/mdpo
- **Self-Rewarding SMC**: https://github.com/Algolzw/self-rewarding-smc
- **Prism**: https://github.com/viiika/Prism
- **dLLM Framework**: 统一 DLM 训练/推理/评估
- **Dream 7B**: https://github.com/DreamLM/Dream
- **LLaDA 2.0**: https://github.com/inclusionAI/LLaDA2.X
- **Reward-Guided Stitching**: https://github.com/roymiles/diffusion-stitching
- **NAP**: https://github.com/pixeli99/NAP
- **Diverse Sampling**: https://github.com/sean-lamont/odd
- **SR-TTT**: https://github.com/swamynathanvp/Surprisal-Aware-Residual-Test-Time-Training
- **TTT-E2E**: https://test-time-training.github.io/e2e.pdf (code available)

### 5.2 预训练模型

- **LLaDA 8B / 8B-Instruct**: HuggingFace `GSAI-ML/LLaDA-8B-Instruct`
- **Dream 7B / 7B-Instruct**: HuggingFace `DreamLM/Dream-7B-Instruct`
- **LLaDA 2.0-mini (16B)** / **LLaDA 2.0-flash (100B MoE)**: HuggingFace (inclusionAI)
- **MDLM checkpoints**: 通过 dLLM 框架

### 5.3 数据集与评测

- **推理**: GSM8K, MATH500, ARC-E/C, Countdown, Sudoku
- **代码**: HumanEval(+), MBPP(+), LiveCodeBench
- **通用**: MMLU, HellaSwag, WinoGrande
- **生成质量**: PPL, MAUVE, Self-BLEU, Distinct-N
- **语言建模**: OpenWebText, The Pile

---

## 6. 对 Idea 生成的启示

### 方向一（强烈推荐）：TTT 层作为 DLM 的跨步自适应记忆

MetaState 证明跨步记忆有效，但其 GRU 更新是固定的、非自适应的。**将 TTT 核心思想引入 DLM**——去噪过程中每步揭示的新 token 作为自监督信号，通过梯度下降更新 MLP 快权重——可实现更强的跨步记忆：

- **架构**: 在 DLM Transformer 中间层插入 TTT 层（类似 Titans），隐状态在去噪步骤间持久化
- **自监督信号**: 已揭示 token 的 NTP/MLM loss 驱动快权重更新
- **核心优势**: DLM 的多步去噪天然提供 TTT 所需的"多次观察"机会，且不需要额外推理 pass
- **与 MetaState 的本质区别**: MetaState = 固定更新规则（GRU），TTT-DLM = **学习的更新规则**（自监督梯度下降）
- **训练**: 可参考 MetaState 的 K-step unrolling 或 TTT-E2E 的 meta-learning

### 方向二（值得探索）：DLM 去噪过程即隐式 TTT

DLM 迭代去噪本身可视为 test-time computation——模型在生成时反复处理同一序列的不同 masked 版本。可以研究：
- 将去噪过程形式化为 TTT（不仅更新 token 预测，还更新部分模型权重）
- DSL 的 SNR-invariant 训练是否暗示更优的"去噪即 TTT"范式
- 利用 MDPO 的 RL 框架将 TTT 目标融入训练

### 方向三（潜在突破）：Surprisal-Aware 选择性记忆用于 DLM

SR-TTT 的洞察——高 surprisal token 路由到精确 cache——可以迁移到 DLM：
- 低置信度/高 surprisal 的 token 位置分配更多记忆容量
- remasking 概率与记忆权重衰减耦合（remasked → 降低记忆权重）
- 与 CoRe 的上下文鲁棒性探测互补

### 方向四（已充分验证，应避免）：简单 Best-of-N / 置信度 remasking

本项目前 18 轮已充分验证：Best-of-N 完全无效，置信度 remasking 在小模型上导致文本退化。应避免，除非结合 CoRe、Where-to-Unmask Gt-Margin、或 MDPO RCR 等全新评分机制。

### 总体建议

**核心洞察**: DLM 的迭代去噪与 TTT 的自监督学习有深层结构对应——TTT 通过自监督梯度步压缩上下文到权重（记忆），DLM 通过迭代去噪逐步揭示 token（生成）。将两者融合——**在去噪过程中同时更新预测和部分权重**——可能是推动 DLM 推理能力的关键突破。MetaState 迈出第一步（跨步状态传递），但未触及 TTT 核心（自监督权重更新）。这一空白既有理论吸引力（TTT 与扩散过程的数学联系），也有实践可行性（冻结骨干 + 轻量 TTT 层 + DLM 天然多步结构），且 MetaState 的存在证明了该方向的基本可行性。
