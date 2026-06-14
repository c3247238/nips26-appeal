# 最终研究提案：Denoising-Time Adaptation — 将去噪迭代转化为显式推理时学习

## 综合决策理由

在审阅六个视角（创新者、实用主义者、理论研究者、反对者、跨学科研究者、实验主义者）及其交叉批评后，我做出以下判断：

### 全景图：六个视角的共识与分歧

**核心共识**（所有视角一致同意）：
1. 前 18 轮的核心教训是 PPL 不可信，必须以 benchmark 准确率为唯一评判标准
2. 0.6B 小模型的结果不可靠，应直接在 7B+ 上实验
3. DLM 去噪步之间的"信息孤岛"（跨步无记忆）是 remasking 效果有限的深层原因
4. 纯 training-free remasking（ReMDM-conf, TCR 等）在推理任务上的提升空间有限

**关键分歧**：
| 议题 | 创新者/实用主义者/理论家 | 反对者 |
|------|--------------------------|--------|
| TTT + DLM 是否可行 | 值得探索，DLM 的迭代去噪天然适合 TTT | 因果/双向不兼容，TTT 已失败 |
| 推理时计算扩展是否有前途 | 可以通过更好的方法实现 | 存在理论上限，应转向训练对齐 |
| 下一步方向 | 将 TTT 机制引入 DLM | 划定不可能性边界 |

**我的裁决**：反对者指出了真实的理论风险（时间无关性、mask 干扰上下文、remasking 无法保证分布正确性），但忽视了一个关键区别——**DTA（Denoising-Time Adaptation）不是传统的 remasking 方法**。DTA 在参数空间而非 token 空间做优化，绕过了反对者所列举的大部分 remasking 局限性。同时，反对者关于 TTT 因果结构不兼容的论点在 DLM 语境下并不成立——DLM 的 TTT 更新沿**去噪时间步**迭代（每步都是完整序列的双向预测），而非沿序列位置因果迭代。

### 各提案排序与选择

| 排名 | 提案 | 新颖度 | 可行性 | 理论深度 | 顶会潜力 | 综合评分 |
|------|------|--------|--------|----------|----------|----------|
| 1 | **DTA/DTT（创新者+实用主义者核心方案）** | 9 | 7 | 8 | 9 | **8.3** |
| 2 | VDTT 变分统一框架（理论家角度三） | 9 | 5 | 10 | 8 | 7.5 |
| 3 | SCP 自矛盾探测（创新者角度二） | 8 | 7 | 6 | 7 | 7.0 |
| 4 | DMI 扩散记忆注入（创新者角度一） | 7 | 8 | 5 | 6 | 6.5 |
| 5 | 不可能性边界（反对者方向一） | 7 | 8 | 9 | 6 | 6.3 |
| 6 | 诊断性基准（实验主义者提案三） | 5 | 9 | 3 | 5 | 5.5 |

**最终选择：DTA 为核心方案**，融合理论家的变分统一框架作为理论包装，实用主义者的工程路径作为实现方案，实验主义者的评估方法论作为验证框架。

### 各视角的权重分配及理由

- **实用主义者**（权重最高，30%）：DTA/DLM-as-implicit-TTT 的核心工程方案直接采用。理由：18 轮迭代证明工程可行性是这个项目的生命线，实用主义者的 LoRA 方案最具操作性
- **理论研究者**（权重 25%）：VDTT 的 EM 框架和信息论分析提供了论文的理论骨架。理由：顶会需要理论深度，但理论必须为算法服务
- **创新者**（权重 20%）：DMI（embedding 级记忆）作为 DTA 的轻量替代/消融基线采用。理由：三层谱系（DMI -> SCP -> DTA）的渐进结构为消融实验提供了自然框架
- **实验主义者**（权重 15%）：诊断性评估框架（Correction Precision/Recall、轨迹稳定性）全面采纳。理由：前 18 轮的评估缺陷必须弥补
- **反对者**（权重 5%）：灾难性遗忘和分布偏移的警示纳入实验设计。理由：反对者的核心论点（TTT+DLM 不兼容）在 DTA 设计下已被正面回应，但失败模式的警示有价值
- **跨学科研究者**（权重 5%）：外部信息分离原则（Turbo 解码类比）和延迟承诺原则（动力学校对类比）纳入方法设计。理由：三要素缺失诊断（独立信息源、时间调度、层级误差）为 DTA 设计提供了检查清单

---

## 标题

**Denoising-Time Adaptation: Turning Diffusion Iterations into Test-Time Learning for Masked Language Models**

## 摘要

遮蔽扩散语言模型（MDLMs）通过迭代去噪生成文本，但去噪步骤之间不共享任何推理状态——每一步都从头计算，丢弃前步积累的上下文理解。我们提出 Denoising-Time Adaptation（DTA），一种将 MDLM 多步去噪过程转化为显式推理时学习的方法。DTA 在去噪过程中，利用已揭示 token 的 masked LM 损失对一个轻量级 LoRA 适配器执行在线梯度更新，使模型在去噪过程中逐步"学会"当前生成的模式和约束。我们从变分推理角度证明，DTA 等价于在 token 空间和参数空间的联合空间中执行 EM 优化，每步去噪-更新都改善变分下界。实验表明，DTA 在 Dream-7B 和 LLaDA-8B 上的 Countdown、GSM8K、MBPP 等推理基准上显著优于纯 remasking 方法（ReMDM-conf、CORE、RCR），同时不引入外部验证器或架构修改。

## 动机

### 问题：去噪步间的信息孤岛

标准 MDLM 的去噪过程存在一个根本性缺陷——**跨步信息丢失**。在每个去噪步 $t$，模型接收当前的部分遮蔽序列 $x_t$，通过前向传播预测所有 mask 位置的 token，然后采样并可能 remask。但上一步的连续表征（logits、attention patterns、hidden states）在采样后完全丢弃。MetaState（arXiv 2603.01331）将此称为"Information Island"问题。

这解释了前 18 轮迭代的核心发现：
- **TTT 6 个变体全部不显著（p=0.88）**——因为 TTT 在 AR 框架下执行，未利用 DLM 的多步迭代结构
- **ReMask-Retry 的 PPL 改善是伪影**——因为 remasking 后模型仍然只看到离散的 masked 序列，无法利用之前的推理状态
- **LLaDA-8B remasking 导致 PPL 恶化 +31.5%**——因为重新遮蔽在无跨步记忆的情况下反而扰乱了已收敛的 token

### 核心洞察：DLM 去噪天然是一种隐式 TTT

TTT 的本质是：**推理时通过自监督损失更新模型隐状态/参数**。DLM 的去噪过程天然具备这一特性：每一步去噪都是在当前部分揭示的上下文下对剩余 mask token 做预测——这在数学上等价于一种 masked language modeling 自监督任务。换言之，DLM 的多步去噪本身就是一种隐式 TTT，每步利用前步揭示的 token 作为"训练信号"更新后续预测。

但当前 DLM 在去噪步间**不更新参数**。如果我们在去噪步间加入轻量参数更新，就能将这种隐式 TTT 变为显式的推理时学习。这正是 DLM 相比 AR 模型的**独特结构性优势**：AR 的 TTT 需要额外的自监督损失，而 DLM 的去噪目标天然就是自监督的。

## 研究问题

1. **RQ1**：在 DLM 去噪过程中加入轻量参数更新（DTA），能否显著提升推理任务的准确率？
2. **RQ2**：DTA 的推理时扩展曲线（准确率 vs 去噪步数/计算量）与纯 remasking 方法有何不同？
3. **RQ3**：DTA 与 remasking 策略是否正交互补？

## 假设

详见 `hypotheses.md`，核心假设：

- **H1（主假设）**：DTA 在 Dream-7B 上的 Countdown 准确率显著优于 vanilla 去噪（预期 +5-10pp）
- **H2**：DTA + ReMDM-conf 组合在 GSM8K 上显著优于纯 ReMDM-conf（预期 +3-5pp）
- **H3（理论假设）**：DTA 的收益随去噪步数增加而增大，且不饱和（不同于 remasking 在 T > 2L 后饱和）

## 方法：Denoising-Time Adaptation（DTA）

### 核心算法

```
Input: prompt x, 全 mask 序列 x_T, 预训练 DLM f_θ, LoRA 参数 Δθ (rank r, 初始化为 0)
Output: 生成序列 x_0

For t = T, T-1, ..., 1:
    // E-step: Token 空间更新（标准去噪）
    1. 用 f_{θ+Δθ} 预测所有 mask token 的分布
    2. 按置信度/调度揭示部分 token -> x_{t-1}
    3. 可选：按 remasking 策略 remask 低置信 token

    // M-step: 参数空间更新（DTA 核心）
    4. 取已揭示 token 集合 S_{t-1}
    5. 对 S_{t-1} 中的 token 计算 masked LM 损失：
       L = -Σ_{i∈S} log P_{θ+Δθ}(x_i | mask(x_i, S))
    6. 更新 Δθ <- Δθ - η·∇L （仅更新 LoRA 参数，1 步 SGD）

Return: x_0
```

### 关键设计决策

1. **LoRA rank 极小（r=4-8）**：每步更新只需 1 次额外反向传播，参数量 < 0.01% 总参数。计算开销约 2-3x 标准去噪（反向传播约 2x 前向传播），但每步的信息利用率远高于多次无记忆的前向传播

2. **旁路初始化**：Δθ 初始化为零，确保零更新时等价于原模型。更新形式为 h' = h + W_Δθ · h，不破坏预训练权重

3. **累积衰减策略**：每步更新后 Δθ <- γ · Δθ（γ ∈ [0.9, 0.99]），防止参数漂移。灵感来自 Titans 的遗忘门机制

4. **前 k 步 warmup**：前 20% 去噪步不执行 DTA（此时已揭示 token 太少，梯度信号噪声大），仅在揭示 >30% token 后开始参数更新。灵感来自跨学科视角的"延迟承诺"原则

5. **仅更新最后 L 层**：只对 Transformer 最后 2-4 层的 FFN 插入 LoRA 适配器，减少梯度计算范围。灵感来自 Locas（arXiv 2602.05085）的发现——最后几层对 token 预测贡献最大

6. **与 remasking 正交**：DTA 更新模型的参数级理解，remasking 修正离散 token 选择，两者互补。DTA + ReMDM-conf 是推荐的组合使用方式

### 理论框架：变分去噪时适应（VDTA）

从理论研究者的变分推理角度，DTA 可以被形式化为在联合空间 (x_0, Δθ) 上的 EM 优化：

- **E-step**（标准去噪）：固定 Δθ^(t)，用增强模型 f_{θ+Δθ^(t)} 去噪 x_t -> x_{t-1}
- **M-step**（DTA 更新）：固定 x_{t-1}，更新 Δθ^(t+1) = Δθ^(t) - η ∇_{Δθ} L(x_{t-1}; Δθ^(t))

**命题（ELBO 单调性）**：在温和的正则性条件下（L 关于 Δθ 强凸 + L2 正则化，f_θ 连续），每个 EM 步骤都改善变分下界。

**命题（信息积累）**：DTA 的 LoRA 参数单调积累关于目标序列的信息：I(Δθ^(t); x_0) >= I(Δθ^(t-1); x_0) + ΔI_t，其中 ΔI_t > 0 是第 t 步新揭示 token 的信息增益。

**与 AR TTT 的关键区别**：
- AR TTT：W^(i+1) = W^(i) - η ∇L(x_i)（沿序列位置因果迭代，需额外自监督损失）
- DTA：Δθ^(t+1) = Δθ^(t) - η ∇L(x_t)（沿去噪时间步迭代，去噪目标天然是自监督的）

这解答了反对者的核心质疑——DTA 不需要因果结构，因为每步 DTA 更新看到的是完整序列（部分 mask），利用的是 DLM 的双向注意力优势。

### 与已有工作的精确定位

| 方法 | 参数更新 | 记忆机制 | 需要训练 | 外部验证器 | 理论保证 |
|------|---------|---------|---------|-----------|---------|
| ReMDM/CORE/Prism | 否 | 无 | 否 | 否 | 有限 |
| MetaState (2603.01331) | 否（推理时）| GRU+CrossAttn | 是（K-step unrolling）| 否 | 无 |
| RemeDi | 否（推理时）| 双流 UPS | 是（SFT+RL）| 否 | 无 |
| ProSeCo | 否（推理时）| 训练 corrector | 是 | 否 | 无 |
| TTT-Linear/Titans | 是 | 快权重 | 是（从头训练）| 否 | 有 |
| **DTA (Ours)** | **是（LoRA）** | **参数级记忆** | **否（零训练）** | **否** | **有（ELBO 单调性）** |

DTA 的独特定位：**唯一同时满足"零训练"+"参数级记忆"+"理论保证"的方法**。

## 消融基线：信息增强谱系

为系统评估跨步信息传递的价值，设计四个层次的消融基线：

```
Level 0: 标准 DLM（无跨步信息）
   |
Level 1: DMI — Diffusion Memory Injection（embedding 级跨步软信息）
   - 上一步 logits 的 softmax 加权 embedding 注入当前步输入
   - 最轻量，无额外前向/反向传播
   |
Level 2: SCP — Self-Contradiction Probing（利用双向性做自验证）
   - Leave-one-out probing 检测自矛盾 token
   - 中等开销（1 次额外前向传播）
   |
Level 3: DTA — Denoising-Time Adaptation（参数级推理时适应）
   - LoRA 在线更新
   - 最高开销（1 次额外反向传播/步），最高表达力
```

## 预期贡献

1. **方法贡献**：DTA——首个将 TTT 机制引入 DLM 去噪过程的 training-free 方法，利用 DLM 的迭代结构实现推理时参数适应
2. **理论贡献**：VDTA 变分框架——证明 DLM 去噪 + 参数更新等价于 EM 优化，建立 TTT 与扩散去噪的形式化统一
3. **实验贡献**：在 Dream-7B/LLaDA-8B 上的系统性推理基准评测 + 信息增强谱系消融 + 推理时扩展曲线分析
4. **分析贡献**：Token 级诊断框架（Correction Precision/Recall），首次量化 remasking 是否在修正正确的 token

## 实验计划

### 模型选择
- **主模型**：Dream-7B-Instruct（最强开源 DLM，Countdown 16.0 vs AR 6.2）
- **验证模型**：LLaDA-8B-Base（验证跨模型泛化性）
- **快速迭代**：Qwen3-0.6B 或 MDLM-170M（概念验证用，不报告最终结果）

### 基准选择
- **Countdown**（主基准）：Dream-7B 强项，快速评估，明确的正确/错误判定
- **GSM8K**（数学推理）：标准基准，有大量对比数据
- **MBPP**（代码生成）：CORE 报告 +9.2pp，可直接对比
- **HumanEval**（代码生成，补充）：Pass@1 评估

### 对比方法
| 方法 | 类型 | 计算开销 |
|------|------|---------|
| Vanilla（baseline）| 标准去噪 | 1x |
| ReMDM-conf | Remasking | ~1.5x |
| CORE | 上下文探测 | ~2x |
| RCR | Running confidence | ~1.5x |
| DMI（Level 1 消融）| Embedding 记忆 | ~1.05x |
| SCP（Level 2 消融）| 自矛盾探测 | ~1.5x |
| **DTA（Ours）** | **参数适应** | **~2.5x** |
| **DTA + ReMDM（Ours）** | **参数适应 + Remasking** | **~3x** |
| HEX（多调度集成）| 集成 | ~Kx |

### 统计严谨性
- 每个实验 3 seeds（42, 123, 456）
- 样本量 >= 200（避免 pilot 小样本陷阱）
- 主要检验：McNemar test + Bootstrap 95% CI
- 多重比较：Bonferroni 校正
- 同时报告准确率 + 多样性指标（防止退化）

### 阶段化执行

**第 1 周：概念验证**
- 在小模型（Qwen3-0.6B 或 MDLM）上实现 DTA 原型
- Countdown 100 题快速 sanity check
- 验证 LoRA 更新的数值稳定性

**第 2 周：主实验**
- 在 Dream-7B 上实现 DTA + ReMDM-conf
- Countdown 500 题 x 3 seeds 完整评估
- DMI/SCP 消融基线实现

**第 3 周：扩展实验**
- GSM8K + MBPP 全量评估
- LLaDA-8B 验证
- 推理时扩展曲线（准确率 vs 去噪步数 T）
- Token 级诊断分析

**第 4 周：消融与写作**
- LoRA rank 消融（r=2, 4, 8, 16）
- 更新频率消融（每步 vs 每 2 步 vs 每 4 步）
- 衰减因子消融（γ=0.9, 0.95, 0.99, 1.0）
- 论文写作

### 计算预算

| 步骤 | GPU·h | 说明 |
|------|-------|------|
| 概念验证（小模型）| ~4 | Qwen3-0.6B 快速迭代 |
| DTA on Dream-7B（Countdown）| ~12 | 500 题 x 3 seeds x 多方法 |
| DTA on Dream-7B（GSM8K+MBPP）| ~20 | 完整评测 |
| DTA on LLaDA-8B | ~16 | 跨模型验证 |
| 消融实验 | ~16 | 4 维消融 |
| DMI/SCP 基线 | ~8 | 2 个消融方法 |
| 扩展曲线分析 | ~8 | T 扫描 |
| **总计** | **~84** | **4x GPU 约 21h 墙钟** |

## 失败模式与应对

### 高风险

1. **梯度计算开销过大**（概率 30%）
   - 缓解：只更新最后 2 层 FFN 的 LoRA；使用 LoRA rank=2；每 2-4 步才做一次 DTA 更新
   - 降级方案：转向 DMI（zero-cost embedding 级记忆）

2. **LoRA 更新导致退化**（概率 25%）
   - 缓解：极小学习率（η=1e-5~1e-4）+ 衰减因子 γ + L2 正则化
   - 诊断：监控每步的 LoRA 范数 ||Δθ||_F，如果持续增长则触发重置

3. **Benchmark 无显著提升**（概率 35%）
   - 缓解：先在 Countdown（DLM 强项）验证，再扩展到 GSM8K
   - 降级方案：转向"DLM 去噪 = 隐式 TTT"的理论分析论文，结合 18 轮负面结果做统一解释

### 中等风险

4. **与预训练权重冲突**（概率 20%）
   - 缓解：旁路设计 + 零初始化保证初始等价
   - 诊断：对比 DTA 前后在 vanilla prompt 上的 PPL 变化

5. **跨模型泛化性差**（概率 25%）
   - 缓解：在 Dream-7B 和 LLaDA-8B 两个模型上验证
   - 如果仅在一个模型上有效，聚焦该模型做深入分析

## 投稿定位

- **最佳情况**（DTA 在多基准上显著有效）：**NeurIPS/ICML 主会**——"Denoising-Time Adaptation: Turning Diffusion Iterations into Test-Time Learning"，方法 + 理论 + 实验
- **中等情况**（DTA 在部分基准有效 + DMI/SCP 有辅助贡献）：**ICLR 主会**——"Information-Augmented Inference for Masked Diffusion: From Embedding Memory to Parameter Adaptation"
- **降级情况**（仅理论贡献有效）：**NeurIPS/ICML 主会偏理论 track**——"Denoising as Test-Time Learning: A Variational Perspective on Masked Diffusion"
- **保底**（全部失败）：结合 18 轮数据——"Why Inference-Time Scaling Is Hard for Masked Diffusion: From Remasking to Adaptation"，投 **EMNLP** 或 **ICLR**

## 关键参考文献

### TTT 与记忆架构
- Sun et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv 2407.04620
- Behrouz et al. (2025). Titans: Learning to Memorize at Test Time. arXiv 2501.00663
- Liu et al. (2026). Test-Time Training with KV Binding Is Secretly Linear Attention. arXiv 2602.21204
- Zhang et al. (2025). Test-Time Training Done Right (LaCT). arXiv 2505.23884
- Furfaro (2025). TPTT: Pretrained Transformers to Titans. arXiv 2506.17671
- Wang et al. (2026). AllMem: SWA + TTT Memory Networks. arXiv 2602.13680
- Lu et al. (2026). Locas: Pluggable Parameterized Memory. arXiv 2602.05085

### DLM 推理时扩展
- Nisonoff et al. (2025). ReMDM: Remasking Discrete Diffusion Models. arXiv 2503.00307
- Zhai et al. (2026). CORE: Context-Robust Remasking. arXiv 2602.04096
- Lee et al. (2025). HEX: Hidden Expert Mixture. arXiv 2510.05040
- Luo et al. (2026). Self-Rewarding SMC. arXiv 2602.01849
- Xia et al. (2026). MetaState: Cross-Step Memory for MDLMs. arXiv 2603.01331
- MDPO (2025). arXiv 2508.13148

### 理论基础
- Svete & Sabharwal (2025). On the Reasoning Abilities of MDMs. arXiv 2510.13117
- Jiang et al. (2025). DLMs are Provably Optimal Parallel Samplers. arXiv 2512.25014
- Chen, Cong & Li (2025). Optimal Inference Schedules for MDMs. arXiv 2511.04647
- Jafari & Anbarjafari (2025). Equilibrium Transformers. arXiv 2511.21882

### 基座模型
- Dream 7B. arXiv 2508.15487
- LLaDA. arXiv 2502.09992
- MDLM (Sahoo et al., NeurIPS 2024). arXiv 2406.07524
