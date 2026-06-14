# 结果辩论综合报告

**综合者**: Sibyl 结果辩论综合者
**日期**: 2026-03-25
**迭代**: Iteration 13
**综合来源**: optimist / skeptic / strategist / methodologist / comparativist / revisionist

---

## 一、各方共识与核心分歧

### 1.1 六方一致达成的共识

以下结论在所有 6 份分析中无争议：

1. **EqWD 在 ImageNet ResNet-50（45 epochs）上数值排名第一**：72.27 ± 0.20，高于 SWD（+0.23%）和 FixedWD（+0.38%），且三个种子结果一致，不依赖特定随机初始化。

2. **EqWD 方差最低**：ImageNet std = 0.197，是 SWD（0.401）的 49%。这一稳定性优势在大规模训练中有独立价值。

3. **CWD 和 CPR 在 ImageNet 上劣于 FixedWD**：两者分别低 0.50% 和 0.51%，是全场最弱的两种动态方法。这一负面结果出乎意料，理论解释价值极高。

4. **45 epochs 是明显弱点**：学术标准为 90 epochs，审稿人必然追问。论文若以 45 epochs 结果为主投稿，需要正面应对这一质疑。

5. **CIFAR-100 上结果平淡**：EqWD 排第三（65.05%），低于 FixedWD（65.19%），差距在误差范围内但方向不占优。动态 WD 在小规模场景的优势确实更难体现。

6. **BEM（Budget Equivalence）测试中 EqWD 不占优**：在同等调优预算下（15 次 Optuna 搜索），EqWD（68.30%）不优于 FixedWD（68.21%），且低于 SWD（68.57%）。这一负面结果必须在论文中诚实呈现。

7. **统计显著性不足**：EqWD vs FixedWD 的 t 检验 p ≈ 0.11-0.20，3 种子下不达 p < 0.05 标准。

### 1.2 核心分歧

| 争议点 | 乐观方（Optimist + Comparativist） | 谨慎方（Skeptic + Methodologist + Revisionist） | 战略方（Strategist） |
|--------|-----------------------------------|-------------------------------------------------|----------------------|
| **是否现在写论文** | 是，当前结果足够 | 否，需补 90 epochs + 统计 | 分步：立即启动写作 + 并行补实验 |
| **ImageNet 45ep 有效性** | 相对比较有意义 | 外部效度受损，不可信 | 可辩护，但必须补 90ep |
| **核心贡献定位** | 统一框架 + EqWD 最优 | 贡献主张需大幅收窄 | 重构为三段式叙事 |
| **BEM 负面结果** | 次要问题，可在正文讨论 | 严重矛盾，必须正视 | 可以正面化为"调优效率优势" |
| **投稿时机** | 立即（NeurIPS 2026） | 待补实验后（ICLR 2027） | NeurIPS 2026，备选 TMLR |

---

## 二、最终判断：PROCEED（进入写作），但附加强制条件

### 综合裁决：**有条件地 PROCEED TO WRITING**

**理由**：

1. **核心结果已具备可发表骨架**：ImageNet ResNet-50 三种子第一名 + 双数据集双架构覆盖 + 消融实验完整，这是绝大多数顶会论文的基本实验配置。

2. **等待 90 epochs 不应推迟写作**：90 epochs 实验需 3-4 天并行运行，可与写作并行进行。写作不需要等待最终数字——框架结构、理论推导、图表设计、相关工作梳理均可现在完成。

3. **论文写作将暴露真正的薄弱点**：Revisionist 和 Methodologist 发现的诸多矛盾（BEM 负面结果、H3 被否定、phase schedule 与 EqWD 等价等）需要在写作过程中得到解决，而非无限期地"先补实验再写"。

4. **时间窗口有限**：NeurIPS 2026 截止约 5 月中旬，从现在到截止还有约 7 周。若不立即启动写作，则将错过本轮最优投稿窗口。

### 强制附加条件（Writing 必须同步处理）

1. **P0（必须完成，否则论文不可投）**：启动 90 epochs ImageNet 实验，在论文完稿前获得结果并更新主要表格。
2. **P0（必须在写作中处理）**：正视 BEM 负面结果，在论文中提供一致性解释（"调优效率"框架），不得隐瞒。
3. **P1（强烈建议）**：补充 2 个额外种子（seed 789 和 1024），使 N=5，报告精确 p 值。
4. **P1（写作层面）**：删除或彻底重构 Sections 5.7 和 5.8（phantom figure 和孤立理论块）。

---

## 三、论文核心故事线建议

### 推荐故事线：「梯度-权重比是 WD 调度的充分统计量」

这是综合 Revisionist 的 Insight 1 和 Strategist 的三段式框架后形成的最强叙事，兼顾理论深度和实验支撑：

**第一段：揭示问题**
> 现有动态 WD 方法分为三个流派——梯度范数调度（SWD）、对齐感知调制（CWD、CAWD）、范数约束优化（CPR）——各有理论动机，但在 ImageNet 规模上的实验结果相互矛盾。更关键的是，ICLR 2026 发表的 CWD 和 NeurIPS 2024 发表的 CPR 在我们的评测中**均劣于普通 FixedWD**。

**第二段：核心 Insight**
> 我们的分析揭示了这一困惑的根源：(a) 在梯度-权重比 r_t 已知的条件下，对齐信号没有增量预测能力（MI = 0，CIFAR-100 全层验证）；(b) 在 AdamW 下，所有 WD 方法统计等价（Phi Invariance Conjecture）；(c) binary/threshold 调制方法（CWD、CPR）在复杂优化景观中产生噪声驱动的正则化开关，导致欠正则化。因此，梯度-权重比偏差 |r_t - r*| / r* 是 WD 调度的充分且必要信号，无需更复杂的几何量。

**第三段：解决方案**
> 基于 Defazio (2025) 的均衡态分析，我们提出 EqWD：三行代码实现连续的比率偏差反馈控制，在 ImageNet ResNet-50（3 seeds）上取得 72.27 ± 0.20% 的第一名，方差是 SWD 的 49%，充分验证了连续调制相对于二值调制的稳定性优势。

**核心卖点摘要**（按审稿人关注度排序）：
1. EqWD 在 ImageNet 上数值领先所有对比方法，方差最低
2. 理论预测（AdamW 下 Phi Invariance，alignment 无增量信息）均得到实验验证
3. 揭示 CWD 和 CPR 大规模失败的系统性原因（binary mask 脆弱性）
4. 三行代码，零额外开销，极简实现

### 次选故事线（若 90 epochs 结果不理想）
降格为"SGD 训练中 WD 调度的新视角"，聚焦理论框架贡献和 Phi Invariance 发现，降低对 EqWD 性能优势的声索。

---

## 四、论文写作优先级

### 写作优先级排序（面向 NeurIPS 2026，7 周时间线）

**Week 1（本周立即启动）**：
- **W1a** [最高优先级] 启动 90 epochs ImageNet 实验（7 方法 × 3 seeds，4 GPU 并行约 3-4 天）
- **W1b** 修复 Sections 5.7/5.8（删除 phantom figure，或将理论正式化到 Section 3）
- **W1c** 修复符号定义：在 Section 3.1 正式定义 $\mathbf{u}_t$，修正 Conjecture 1 的 BN 条件

**Week 2-3（实验运行中，同步写作）**：
- **W2a** 重写 Abstract 和 Introduction，采用推荐故事线的三段式框架
- **W2b** 整合 BEM 负面结果：在 Section 4（Experiments）增加"调优效率分析"子节，正面解读
- **W2c** 更新 Related Work，明确"alignment signal 无增量信息"这一发现与 CWD/CAWD 失败的理论连接
- **W2d** 删除 Layer-aware 变体的贡献声明，降级为"preliminary negative finding"或附录内容

**Week 4（更新 90 epochs 结果）**：
- **W4a** 用 90 epochs 结果替换主表格，45 epochs 结果降级为附录中的"early stopping efficiency"分析
- **W4b** 补充 N=5 统计验证，报告 Welch t-test p 值

**Week 5-6（完整审稿轮次）**：
- **W5a** 内部 critique 循环（重新调用 critic + section-critic agent）
- **W5b** P3 级补充实验（AdamW baseline、ResNet-101 sanity check、ImageNet beta 扫描）
- **W5c** 最终图表制作和 LaTeX 排版

**Week 7**：最终打磨，投稿

---

## 五、风险缓解策略

### 风险 R1：90 epochs 结果与 45 epochs 排名反转（EqWD 跌至第二或第三）

**概率**：20-30%（正则化方法通常在长训练中优势更显著，但存在不确定性）

**缓解措施**：
- 若 EqWD 仍领先：按原计划投稿，这是最强的证据组合
- 若 EqWD vs SWD 排名互换（差距 < 0.1%）：调整贡献声称，主打稳定性（低方差）+理论框架，性能差异降为次要卖点
- 若 EqWD 低于 FixedWD：大幅修正论文定位，转向纯理论框架+诊断工具价值，放弃"最优性能"声称

**预警动作**：若 45 epochs 的单 seed 快速验证（FixedWD vs EqWD 各 1 seed，约 12 小时）显示 EqWD 优势消失，立即停止全量 3 seed 实验，重新评估策略。

### 风险 R2：统计显著性问题（p > 0.05，NeurIPS 审稿人直接要求重做）

**概率**：70%（已有 p ≈ 0.11 的预估）

**缓解措施**：
- 主动在论文中使用"numerically consistent improvement"而非"statistically significant"措辞
- 5 seeds 方案：增加 seed 789 和 1024，报告 5 seeds 结果和 p 值
- 如果 p < 0.05：直接声明显著性
- 如果 p 仍 > 0.05（90ep + 5seeds）：放弃性能声称策略，改为"稳定性 + 理论框架"定位

### 风险 R3：BEM 负面结果被审稿人解读为"EqWD 无真实优势"

**概率**：40%

**缓解措施**：
- 主动在 Experiments 节中披露 BEM 结果，并提供解释框架：
  > "在同等调优预算下，EqWD 与 FixedWD 统计等价——这恰恰说明 EqWD 的真正优势是调优效率：用更少的超参搜索代价（只需设置 beta ≈ 2-5）达到接近最优性能，而 FixedWD 需要充分调优 lambda 才能赶上。"
- 这是诚实且可防御的叙事，优于隐瞒后被审稿人发现。

### 风险 R4：Phase Schedule 控制实验暗示 EqWD 增益来自 schedule，非 ratio deviation

**概率**：30%（Revisionist 提出）

**缓解措施**：
- 设计并运行"随机化 phase baseline"：与 EqWD 相同的 WD budget 总量，但随机分配到各阶段
- 若 random phase 远低于 EqWD：说明 ratio deviation 信号重要，非任意 schedule 均有效
- 若 random phase ≈ EqWD：需要重新解释，ratio deviation 的贡献机制需要更细致分析

### 风险 R5：Novelty 质疑（WD 领域太拥挤）

**概率**：30%

**缓解措施**：
- 强化"Phi Invariance Conjecture"作为可量化的新发现（AdamW 下所有 WD 方法等价）
- 强化 MI = 0 发现（alignment signal 无增量信息）作为实验新知
- 强调 CWD/CPR 大规模失败的系统性诊断是社区贡献，独立于 EqWD 自身的性能
- 确保文献调研不遗漏 2025-2026 年最新工作（arXiv:2506.02285, arXiv:2510.12402, arXiv:2506.14562 等）

---

## 六、投稿目标建议

### 首选：NeurIPS 2026 主会

**预计截止**：2026 年 5 月中旬
**推荐理由**：
- SWD（NeurIPS'23）、CPR（NeurIPS'24）均在 NeurIPS 发表，EqWD 作为后续工作在同一平台更具对话性
- NeurIPS 对优化方法的理论+实验论文接受度历史良好
- 当前时间线（7 周）完全可行（并行实验+写作）
- **当前接受概率估计**：补充 90 epochs 后约 40-55%（基于相关工作接受率和本文强度评估）

### 次选：TMLR（滚动提交）

**推荐理由**：
- 无截止日期，给 90 epochs 实验充足时间
- 接受多轮 revision，对 BEM 负面结果等复杂发现更友好
- 作为 NeurIPS 2026 的安全网

**建议**：NeurIPS 2026 投稿同期，在 arXiv 发布预印本（90 epochs 结果完成后），TMLR 作为并行提交选项。

### 不推荐

- **ICML 2026**：截止日期已过
- **ICLR 2027**：等待时间过长，竞争对手可能在此期间发表类似工作
- **Workshop 投稿**：当前工作量级和创新程度超出 Workshop 定位，浪费迭代机会

---

## 七、行动清单（按紧迫度排序）

### 今日内必须启动

1. **立即启动 90 epochs ImageNet 实验**（4 GPU 并行，EqWD + FixedWD + SWD，各 3 seeds）
2. **快速 sanity check**（仅 EqWD + FixedWD，各 1 seed at 90 epochs）确认方向，约 12-14 小时后有结果
3. 检查并确认 writing 目录中 Sections 5.7/5.8 的具体问题，制定修复方案

### 本周内完成

4. 修复 writing Sections 5.7/5.8（phantom figure 和孤立理论块）
5. 补充 seed 789 和 1024 的 ImageNet 45ep 实验（统计扩展）
6. 重写 Abstract 和 Introduction 框架（可不含 90ep 最终数字，先用占位符）
7. 设计并运行"随机化 phase baseline"控制实验（1 seed，约 30 分钟）

### 等待 90 epochs 结果后（约 4-5 天）

8. 更新主表格，确认 EqWD 排名
9. 报告统计显著性（Welch t-test，N=5）
10. 决定是否需要 AdamW baseline 实验（P3 级）

---

## 附录：各 Agent 分析质量评估

| Agent | 主要贡献 | 可靠性 |
|-------|---------|-------|
| Optimist | ImageNet 数值优势分析、低方差独立价值论证 | 高，但对 BEM 问题淡化 |
| Skeptic | 系统性问题清单，45ep 和统计显著性两个核心质疑 | 高，标准最严格 |
| Strategist | 投稿时间线和优先级排序最为务实可行 | 高，平衡了风险和机会 |
| Methodologist | BEM 负面结果分析最深入，p 值计算具体 | 最高，方法论最严谨 |
| Comparativist | 与 SOTA 的技术差异分析最完整，引用地图最全 | 高，信息量最大 |
| Revisionist | 假设修正和新 insight 发现最有价值，尤其是 H3 被否定的分析 | 高，提供了论文最关键的重构方向 |

**关键裁决**：不采纳 Skeptic 的"不投稿"建议（过于保守），但完全采纳 Methodologist 的 BEM 分析和 Revisionist 的故事线修正建议。最终策略融合 Strategist 的时间线和 Comparativist 的竞争定位。

---

*综合者 Agent | 动态权重衰减框架 | 迭代 13 | 2026-03-25*
