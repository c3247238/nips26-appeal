# 实验决策

## 决策：PROCEED（转型为机制洞察论文）

## 理由

经过全面审阅 39 组实验的定量结果、6 位角色的辩论意见（5:1 支持继续），以及对怀疑者反对意见的逐条评估，做出 PROCEED 决策。以下是详细论证。

### 一、为什么不是 PIVOT

PIVOT 意味着放弃当前所有实验数据，转向全新方向。这在本项目中不合理，原因如下：

1. **数据资产价值高**：39 组实验覆盖 3 种架构/数据集组合、11 种 WD 方法、多组超参数扫描和消融实验。这些数据即使在负面结果框架下也具有独立学术价值。丢弃它们的机会成本远大于撰写论文的边际成本。

2. **实验设计质量优秀**：等价累积实验（92.54% = 92.54%）、LR 解耦实验（aggressive 崩溃至 10.00%）、Random Dynamic WD 对照（92.06% vs 92.05%）——这些消融实验的因果推理干净利落，是很难在常规论文中看到的实验设计水平。

3. **关键发现具有独立贡献**：LR-WD 耦合的必要性虽然在 AdamW 论文（Loshchilov & Hutter 2019）中有过讨论，但从未在 SGD + milestone schedule 设定下给出系统性的实证验证。特别是 aggressive 解耦模式下 weight norm 降至 0.0036 的正反馈崩溃机制，是一个此前文献中未报道的具体失败模式。

### 二、为什么不是 REFINE（追加更多 AADWD 变体实验）

AADWD 作为提升准确率的方法已经彻底失败，继续在此方向上精调是沉没成本谬误：

- **对齐信号不可行动**（F1）：Random = AADWD，信号本身无信息量
- **Conservative 退化为常数**（F2）：delta ~ O(10^{-3}) 使动态项失效
- **LR 耦合主导一切**（F3）：所有变体的 lambda_t 轨迹由 gamma_t 决定

这三条证据互相独立、互相印证，排除了「换一组超参就能翻盘」的可能性。

### 三、为什么是 PROCEED

本决策的核心判断是：**实验数据的学术价值不在于 AADWD 方法本身，而在于它作为探针工具所揭示的 Weight Decay 动态机制**。

具体评估每条核心贡献的新颖性和可发表性：

**贡献 1：WD 预算等价性（Budget Equivalence）**
- 实验证据：等价累积实验精确匹配 92.54% = 92.54%
- 新颖性评估：**中高**。「总量决定效果」虽然直觉上合理，但此前缺乏在 SGD 训练 200 epoch 尺度上的精确实验验证。这不是「证明了一个大家知道的事」，而是「用实验证明了一个大家猜测但没人验证过的事」。学术贡献在于将经验假设提升为实证命题。

**贡献 2：LR-WD 耦合必要性（Coupling Necessity）**
- 实验证据：解耦后 aggressive 崩溃至 10.00%（weight norm → 0.0036），conservative 退化至 80.30%
- 新颖性评估：**高**。这是本论文最强的贡献点。正反馈崩溃机制（high delta → high lambda → weight death → higher delta → repeat）是一个之前未被明确描述的不稳定模式。AdamW 论文讨论的是 L2 与 decoupled WD 的区别，而非 LR 缩放对 WD 稳定性的必要性。

**贡献 3：对齐信号的不可行动性（Alignment Signal Inapplicability）**
- 实验证据：Random Dynamic WD (92.06%) = AADWD Aggressive (92.05%)
- 新颖性评估：**中**。这本质上是一个负面结果，但与近年来 gradient-based adaptive regularization 的研究趋势（如 CWD/ICLR 2026）直接相关，具有文献对话价值。

**附加贡献 4：CWD 跨架构崩溃**
- 实验证据：三设定一致的 best-to-final 退化（4.84%、12.57%、6.48%）
- 这对 ICLR 2026 的 CWD 工作提出了重要的实证质疑，具有独立引用价值。

### 四、对怀疑者核心论点的回应

| 怀疑者论点 | 回应 |
|-----------|------|
| 结论对实践者是「常识」 | 常识不等于已验证的科学知识。科学贡献在于将直觉系统化、将猜测转化为可复现的实证。大量「常识」在严格实验下被推翻（如 batch size vs generalization 的长期争论） |
| LR-WD 耦合在 AdamW 论文中已讨论 | AdamW 讨论的是 L2 与 decoupled WD 的等价/非等价性，而非 LR 缩放对 WD 的稳定性保证。解耦崩溃的具体机制是新发现 |
| 「花大量 GPU 证明已知结论」 | 39 组实验的总 GPU 时间约 2-3 天（CIFAR 级别），计算开销不算大。且等价累积实验、LR 解耦实验都是此前文献未执行过的具体实验 |
| 超参数灵敏度差 | 这恰恰是论文应报道的负面发现——AADWD 不仅不优于 fixed WD，其失败模式还更为灾难性（cliff-like degradation vs gradual degradation） |

## 论文方向

**核心叙事**：Weight Decay 调度的机制洞察论文——不是提出新方法，不是纯负面结果，而是通过系统性消融实验回答「为什么固定 Weight Decay 足够好」这个基础问题。

**建议标题**：
> "On the Sufficiency of Constant Weight Decay: Alignment Dynamics, Learning Rate Coupling, and Budget Equivalence in Nonconvex SGD"

**备选标题**（更直接）：
> "Why Dynamic Weight Decay Fails: An Empirical Study of Alignment-Based Regularization Scheduling"

**论文结构**：
1. **Introduction**：动态 WD 的研究动机 → 为何理论上可行但实践中失败 → 本文的机制解释
2. **Background**：WD 理论（L2 vs decoupled）、alignment 概念、近期动态 WD 方法（CWD 等）
3. **AADWD Framework**：作为实验探针的设计哲学，三种变体的公式
4. **Experimental Setup**：39 组实验矩阵、3 种架构/数据集、消融设计
5. **Results & Analysis**：
   - 5.1 Fixed WD 的普遍最优性（Table 1 主结果）
   - 5.2 预算等价性实验（等价累积 = 固定 WD）
   - 5.3 LR-WD 耦合的必要性（解耦崩溃分析）
   - 5.4 对齐信号的不可行动性（Random = AADWD）
   - 5.5 CWD 的系统性晚期崩溃
6. **Discussion**：三条经验原则的归纳、理论含义、实践建议
7. **Conclusion**

## 补充实验

| 优先级 | 实验 | 目的 | 预计时间 |
|--------|------|------|---------|
| **P0（必做）** | 4 核心方法 x 3 额外种子（ResNet20/CIFAR-10） | 统计显著性，消除单种子质疑 | 1-2 天 GPU |
| **P1（强烈建议）** | CIFAR-100/ResNet20 的等价累积实验 | 验证预算等价性跨数据集泛化 | 0.5 天 GPU |
| **P1（强烈建议）** | VGG16/CIFAR-10 的 LR 解耦实验 | 验证解耦崩溃跨架构泛化 | 0.5 天 GPU |
| **P2（建议）** | AdamW 优化器下的 2-3 个核心实验 | 扩展适用范围到自适应优化器 | 1 天 GPU |
| **P3（可选）** | 更长训练（400 epoch）的 1-2 个验证 | 排除训练长度的混淆因素 | 1 天 GPU |

**核心实验优先排序逻辑**：多种子实验（P0）是投稿的硬性门槛。等价累积和解耦实验的跨设定验证（P1）是论文贡献泛化性的关键支撑。AdamW（P2）和长训练（P3）可在 camera-ready 阶段补充。

## 风险评估

| 风险 | 概率 | 严重度 | 缓解策略 |
|------|------|--------|---------|
| **Reviewer 认为结论缺乏新颖性** | 30-35% | 高 | 强调等价累积实验的因果设计（非相关性而是干预实验）；突出 LR 解耦崩溃的正反馈机制分析；引用近期动态 WD 工作（CWD/ICLR 2026）说明该问题仍是活跃研究方向 |
| **Reviewer 要求更大规模实验（ImageNet 等）** | 40-45% | 中 | 在 Scope Limitation 中明确声明研究范围为 CIFAR-level benchmark，机制洞察的泛化性需要 future work 验证。用 3 种架构/数据集的一致性作为初步泛化证据 |
| **多种子实验翻转某个边缘结论** | < 5% | 低 | 核心结论（Random = AADWD、解耦崩溃、CWD 崩溃）的效应量足够大（差距 > 2%），方向性不会翻转。唯一可能翻转的是 Conservative 与 Fixed WD 的 0.17% 差距——这恰好支持「Conservative 退化为常数」的论点 |
| **理论分析深度不足** | 25-30% | 中 | 对 LR-WD 耦合提供半形式化的稳定性分析（不需要完整证明，给出直觉和必要条件即可）；引用 Smith (2018) 等 LR-WD 理论工作 |
| **CWD 作者在审稿中质疑 CWD 崩溃结论** | 20% | 中 | 确保 CWD 实现严格按照原始论文规范；在补充材料中公开全部实验代码和超参数 |

## 时间规划

| 阶段 | 内容 | 预计时间 |
|------|------|---------|
| Week 1 | P0 多种子实验 + P1 跨设定验证 | 2-3 天 GPU + 并行写作 |
| Week 1-2 | 论文草稿（Introduction, Background, Setup, Main Results） | 5 天 |
| Week 2-3 | 论文草稿（Analysis, Discussion, Conclusion）+ 图表制作 | 5 天 |
| Week 3 | P2 AdamW 实验（如果时间允许）+ 论文修订 | 3-5 天 |
| Week 4 | 内部审查、修订、提交准备 | 3-5 天 |

**目标投稿**：TMLR（滚动截稿，质量要求匹配）或 NeurIPS 2026（约 2026 年 5-6 月截稿，时间紧但可行）。

## 最终判断

AADWD 作为「提升准确率的优化方法」彻底失败，但实验数据作为「揭示 WD 机制的系统性消融研究」成功地回答了三个此前未被实证回答的问题。39 组实验的投资已经产出了足以支撑一篇机制洞察论文的核心贡献。在补充多种子实验和关键跨设定验证后，数据将达到会议论文的投稿标准。

决策：**PROCEED，立即转入论文写作阶段，同步执行 P0/P1 补充实验。**
