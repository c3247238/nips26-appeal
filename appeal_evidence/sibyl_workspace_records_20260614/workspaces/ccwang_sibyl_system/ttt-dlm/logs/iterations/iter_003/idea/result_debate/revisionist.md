# 修正主义者分析：从数据回溯理论——DTA 实验结果对原始假设体系的挑战

## 总体立场

我的任务是从实验数据出发，反向审视我们的假设和心智模型。实验数据传递了一个清晰但令人不适的信号：**我们对 DLM 去噪过程的理解存在根本性偏差**。DTA 的核心叙事——"去噪步间的信息孤岛可以通过参数级适应弥合"——目前没有得到实验支持。但数据中藏着比"方法无效"更深刻的信息，值得仔细挖掘。

---

## 1. 假设审计：逐条对照实验证据

### H1：DTA 在 Countdown 上显著有效 → **初步证伪（待 Full-Scale 确认）**

- **预期**：DTA 在 Dream-7B Countdown 上 +5-10pp
- **实际**：
  - Pilot (16 样本)：DTA 6.2% vs Vanilla 12.5%（-6.2pp，方向相反）
  - Full-Scale (500 × 3 seeds)：DTA 尚未完成，但所有完成方法中 Vanilla 4.7%、ReMDM 4.4%、RCR 5.7%、DMI **9.3%**
  - Task 5a 全方法对比：DTA 6.2% vs Vanilla 12.5%（一致的负方向）
- **证据强度**：中等。16 样本统计上不足以证伪（1 题差异），但 DTA 在 Countdown、GSM8K、LLaDA 三个维度上**一致性地低于 vanilla**，这种系统性趋势不太可能是纯噪声。
- **关键反思**：预期 +5-10pp 的效应量是基于什么？回顾 proposal，这个数字似乎是"合理猜测"而非任何先验计算。我们对 DTA 的效果量没有任何理论上的预测，这本身就是一个规划缺陷。

### H2：DTA + Remasking 组合优于纯 Remasking → **初步证伪**

- **预期**：DTA + ReMDM-conf 在 GSM8K 上 +3-5pp
- **实际**：
  - GSM8K pilot：DTA+ReMDM 18.8% vs ReMDM-conf 37.5%（-18.7pp，方向强烈相反）
  - Countdown pilot：DTA+ReMDM 6.2% vs ReMDM-conf 6.2%（无差异）
  - MBPP：DTA+ReMDM 12.5% vs DTA 37.5%（组合反而大幅下降）
  - LLaDA GSM8K：DTA+ReMDM 31.2% vs Vanilla 43.8%（-12.6pp）
- **证据强度**：中-高。跨三个 benchmark 和两个模型，DTA+ReMDM **一致地劣于**至少一个单独方法。
- **关键反思**：H2 的"正交互补"假设基于一个错误的前提——DTA 和 remasking 在不同空间操作因此应该互补。但实验表明它们**互相干扰**：DTA 更新参数后改变了模型的置信度分布，使 remasking 的阈值判断失准；remasking 又改变了 revealed token 集合，使 DTA 的自监督信号不稳定。

### H3：DTA 推理时扩展不饱和 → **无法判定（实验数据不足）**

- Task 6a 扫描了 T = 64, 128, 256, 512，但 16 样本下所有数据点都在噪声范围内（0%-12.5%）
- **没有任何方法**展现出清晰的扩展趋势
- 这个假设既未被支持也未被证伪，需要 Full-Scale 数据

### H4：信息增强谱系单调递增 → **部分证伪，且发现了预期外的排序**

- **预期**：DTA (Level 3) > SCP (Level 2) > DMI (Level 1) > Vanilla (Level 0)
- **Full-Scale 实际排序**（Countdown-500）：**DMI (9.3%) > RCR (5.7%) > Vanilla (4.7%) > ReMDM (4.4%)**
- **Pilot 排序**（task_5a, 16 样本）：Vanilla (12.5%) > ReMDM/RCR/DTA/DTA+ReMDM/SCP (6.2%) > DMI (0%)
- **矛盾**：Pilot 和 Full-Scale 的 DMI 排序完全翻转！Pilot 中 DMI 最差（0%），Full-Scale 中 DMI 最好（9.3%）。
- **关键反思**：这个翻转极其重要。它不仅证伪了"单调递增"假设，更揭示了一个深层问题：**信息增强的价值不是由机制复杂度决定的，而可能由机制与任务特性的匹配度决定**。DMI 的"软 embedding 注入"在大样本上表现出稳健的改善，而更复杂的 DTA 和 SCP 没有。

### H5：DMI 的 logit carry-over 改善预测 → **Full-Scale 数据支持（出乎意料）**

- DMI 在 Full-Scale Countdown 上 9.3% vs Vanilla 4.7%，约 2x 改善
- 这是所有完成方法中**唯一**表现出明确正向效应的
- **但与 pilot 矛盾**（pilot 中 DMI 0%），说明效应的方向在小样本上完全不可预测

### H6：SCP Correction Precision > ReMDM → **Pilot 数据支持**

- SCP Precision 76.9% vs ReMDM 31.3%（task_8b 诊断数据）
- 这是少数得到数据支持的假设之一
- 但 SCP 的高精度并未转化为准确率优势，说明**精准识别错误 token 是必要但不充分条件**

### H7：DTA 信息积累单调性 → **代理指标支持，但意义存疑**

- Task 8c 显示预测置信度 0.969 → 0.989 → 0.995 随去噪步递增
- 但这个指标对 vanilla 也成立（去噪过程天然使置信度上升）
- **DTA 的增量贡献不明确**——vanilla 的置信度轨迹未作为对照

### H8：gamma 最优范围 [0.9, 0.99] → **无法判定（pilot 全部 0%）**

- Task 7b 扫描了 6 个 gamma 值，所有 DTA 配置均 0% 准确率
- Norm 行为符合预期（gamma 越大 norm 越大），但这仅说明衰减机制工作正常，不说明任何 gamma 值下 DTA 有效

### H9：ReMDM Correction Precision < 50% → **支持（31.3%）**

- ReMDM-conf 的 Correction Precision 仅 31.3%，近 70% 的 remask 操作作用在正确 token 上
- **这是一个有力的 negative finding**，对所有基于置信度的 remasking 方法构成根本性质疑

### H10：DTA 不引入退化 → **支持**

- DTA 的 distinct-2 和 rep-3 指标与 vanilla 接近甚至略优
- LoRA 范数极小（delta norm ~1e-5），DTA 更新在 bfloat16 精度下接近数值下限
- **但这也引出新问题**：如果 DTA 的参数更新如此微小，它到底做了什么？参数几乎未变但计算开销 4-5x，这不是"安全"而是"无效"的另一种表述。

---

## 2. 意外发现分析：数据教给我们什么？

### 意外 #1：DMI 是唯一稳健有效的方法

这是整个实验中最大的意外。DMI 是设计中**最简单**的方法——仅在 embedding 层注入上步 logit 的软信号——但它是唯一在 Full-Scale 上展现出明确正向效应的方法（9.3% vs 4.7%，约 2x）。

**这告诉我们什么？**
- 跨步信息的价值确实存在，但获取这个价值不需要参数级适应
- Embedding 空间的软信号比参数空间的梯度更新更"温和"，不会扰乱已收敛的去噪轨迹
- **DLM 去噪过程对参数扰动极其敏感，但对输入扰动（embedding 混合）有一定容忍度**

### 意外 #2：MBPP 是 DTA 唯一正向的 benchmark

DTA 在 MBPP 上 37.5% vs Vanilla 25.0%（+12.5pp），而在 Countdown 和 GSM8K 上均负。这个异常值的可能解释：

- 代码生成的 token 间依赖结构与算术推理不同——代码有更多的局部模式（函数名、缩进、语法结构），DTA 的 masked re-prediction 可能更容易学到这些局部模式
- 但 16 样本下 +2 题的差异统计意义极弱
- **如果在 Full-Scale 上确认，这将是"任务依赖性"叙事的有力证据**

### 意外 #3：Pilot 与 Full-Scale 的系统性不一致

| 指标 | Pilot 声称 | Full-Scale 实际 |
|------|-----------|----------------|
| Entropy remasking PPL | -24.9% | -0.5% |
| TCR PPL | -22.9% | -2.8% |
| Anneal PPL | -16.5% | +8.1% |
| DMI Countdown acc | 0% (16样本) | 9.3% (500×3) |

**两个方向上的翻转**同时发生：PPL 指标上的正面结果消失了，准确率指标上的负面结果（DMI）翻转为正面。这不仅是"样本量不足"的问题，更说明：

- **PPL 和 accuracy 在 DLM 评估中可以完全脱钩**
- Pilot 阶段的任何结论都不可信——不仅正面结论不可信，**负面结论也不可信**
- 这对 DTA 本身有一个隐含意义：DTA 在 pilot 上的负面表现也可能在 Full-Scale 上翻转（虽然可能性不高，因为 DTA 在多个 pilot 中一致为负）

### 意外 #4：LoRA 范数的极小值

Task 8c 显示 DTA 的 delta norm 仅 ~1e-5，B norm ~1e-4。这意味着 DTA 的参数更新**在 bfloat16 精度下接近有效数值下限**。可能的解释：

- lr=5e-4 在 AdamW 下配合 gamma=0.95 的衰减，使得有效学习率极低
- 每步的 masked re-prediction loss 本身就很小（0.02-0.19），梯度量级不足
- **DTA 可能根本没有在做有意义的参数更新**——它只是一个昂贵的恒等映射

---

## 3. 心智模型更新：我们哪里理解错了？

### 错误认知 #1："信息孤岛"是 DLM 推理任务表现差的主要原因

原始提案的核心叙事是：DLM 去噪步间不共享推理状态（信息孤岛），这限制了推理能力。但实验数据显示：

- **弥合信息孤岛（DTA、DMI、SCP）不能在大多数 benchmark 上显著提升准确率**
- DMI 的改善可能来自一个不同的机制——不是"弥合信息孤岛"而是"软化采样决策"（soft embedding 使模型不那么早做出 hard commit）
- **信息孤岛可能不是瓶颈**。真正的瓶颈可能是模型本身的推理能力（Dream-7B 的 Countdown 基线只有 4.7%），而非推理时计算的利用方式

### 错误认知 #2："DLM 去噪天然是隐式 TTT"

提案的核心洞察是：DLM 的每步去噪等价于一个 masked LM 自监督任务，因此可以通过参数更新将其变为显式 TTT。但数据揭示了一个关键区别：

- **TTT 在 AR 模型上有效是因为每个 token 的预测直接影响下游 token**——参数更新改善了"此刻的预测"，这个改善立即被"下一个 token 的生成"利用
- **DLM 的去噪步间，参数更新改善了"整个序列的重预测"，但这个改善被下一步的 remask/unmask 操作部分或全部覆盖**
- 换言之，DTA 的参数更新和 DLM 的去噪调度之间存在"信息利用时序不匹配"——DTA 更新的知识在下一步被新的 mask pattern 覆盖

**修正后的理解**：DLM 去噪不是隐式 TTT。TTT 需要参数更新的效果**被后续步骤持续利用**，而 DLM 的去噪过程在每步重新决定 mask pattern，使得前步的参数适应部分失效。这是 AR TTT 和 DLM DTA 之间的**结构性不对称**——反对者在辩论中指出的风险确实兑现了。

### 错误认知 #3："参数空间优化优于 token 空间优化"

提案的层级假设是 DTA (参数级) > SCP (验证级) > DMI (embedding 级) > Vanilla。实验数据**完全翻转了这个层级**：

- **Embedding 级（DMI）效果最好**
- **参数级（DTA）效果最差**（甚至可能有害）
- **验证级（SCP）介于两者之间，但开销过大**

**修正后的理解**：在 DLM 去噪语境下，干预的"深度"与"效果"可能是**反向关系**。越深层的干预（参数修改）越可能扰乱去噪过程的内部动力学；越浅层的干预（embedding 混合）越可能被模型自身的注意力机制吸收和利用。这是因为 DLM 的预训练权重编码了一套精密的去噪动力学，参数级干预破坏了这套动力学，而 embedding 级干预只是"轻推"了输入分布。

### 错误认知 #4：对反对者的过度否定

在 proposal 的决策中，反对者的观点被赋予最低权重（5%）。回顾反对者的核心论点：

> "因果/双向不兼容，TTT 已失败"
> "存在理论上限，应转向训练对齐"

数据显示反对者在以下方面是正确的：
1. TTT 机制（参数在线更新）确实在 DLM 上不如预期——不是因为"因果/双向不兼容"，而是因为"参数更新与去噪调度的时序不匹配"
2. 推理时计算的提升空间确实有限——所有方法在 Countdown 上的绝对准确率都很低（<10%），说明瓶颈可能在模型能力而非推理时策略

**应该给反对者 20-25% 的权重**，而非 5%。反对者的具体机制推理不完全正确，但他们对风险的直觉判断是准确的。

---

## 4. 重新构框提案

### 原始构框
"DLM 去噪步间存在信息孤岛 → 参数级适应（DTA）弥合信息孤岛 → 推理能力提升"

### 数据支持的替代构框

**构框 A：轻量干预假说**
"DLM 去噪过程对深层干预（参数修改）敏感但对浅层干预（embedding 混合）容忍 → 最有效的推理时增强是最小化对去噪动力学的扰动 → DMI 式的软信号注入是最优策略"

**构框 B：去噪动力学不可逆假说**
"DLM 的预训练去噪动力学是一个精密调优的系统，任何在线修改（无论参数级还是 token 级）都有可能破坏这个系统 → 推理时增强应该在去噪过程**之外**操作（如多采样+选择、外部验证器），而非修改去噪过程本身"

**构框 C：任务依赖性假说**
"推理时增强的效果取决于任务的 token 依赖结构 → 代码生成（MBPP）受益于 DTA 是因为代码有更多局部可学习模式 → 算术推理（Countdown、GSM8K）不受益是因为关键推理步骤分布在序列中无法被局部模式捕捉"

我倾向于**构框 A + C 的组合**作为最佳替代理论。构框 B 过于悲观（DMI 的 Full-Scale 成功否定了"任何干预都有害"）。

---

## 5. 新假设

### NH1：浅层干预定理
**在 DLM 去噪过程中，推理时增强的有效性与干预深度成反比。** Embedding 级（DMI）> Token 级（ReMDM）> 参数级（DTA）。这是因为 DLM 的注意力机制可以吸收浅层扰动但无法补偿深层参数变化。

### NH2：DMI 的机制不是"信息传递"而是"决策延迟"
DMI 的 soft embedding 使模型在早期去噪步不那么急于做出 hard token 选择。改善不是因为"跨步信息"，而是因为**延迟了不可逆的采样决策**。这可以通过比较 DMI 的 token 变化率 vs vanilla 的 token 变化率来检验。

### NH3：DTA 的失败是学习率/精度问题，而非方法论问题
LoRA delta norm 仅 ~1e-5，在 bfloat16 精度下接近有效下限。可能 DTA 需要更高学习率（1e-3 到 1e-2）或 float32 梯度累积。当前的"安全"超参数选择使 DTA 退化为昂贵的恒等映射。

### NH4：模型基线能力是推理时增强的下界
当模型基线准确率 <10%（如 Dream-7B Countdown 4.7%），推理时增强的绝对收益空间极小。**推理时增强可能只在模型已有一定能力的 benchmark 上有效**（如 LLaDA-8B GSM8K 43.8%），而在模型能力不足的 benchmark 上无论怎么增强都无法突破瓶颈。

### NH5：Correction Precision 是推理时增强的天花板指标
H9 确认 ReMDM Precision 仅 31.3%。**如果一个方法无法准确识别哪些 token 需要修正，那么再多的修正机会也无法转化为准确率提升。** 任何 remasking 类方法的上限受限于其 Correction Precision。

---

## 6. Pivot vs Iterate 建议

### 推荐：**部分 Pivot — 从 DTA 转向 DMI-Enhanced 去噪**

**理由**：
1. DTA 在当前超参数设置下几乎确定无效（一致的负向结果）
2. DMI 是唯一在 Full-Scale 上有正向信号的方法
3. 论文的核心贡献可以从"DTA 方法提出"转向"系统性消融揭示浅层干预优势 + token 级诊断框架"
4. DMI 的 2x 改善（9.3% vs 4.7%）虽然绝对值不大，但在 training-free、近零开销的前提下已是有意义的结果

### 具体行动建议

**立即做**：
1. 等待 DTA Full-Scale 完成——虽然 pilot 一致为负，但 DMI 的 pilot 也是负的（0%），在 Full-Scale 翻转为正面。不排除 DTA 也有类似翻转，尽管可能性较低
2. 完成 SCP Full-Scale 以确认 Level 2 的完整定位
3. 补充 DMI 的 alpha 消融（0.1, 0.2, 0.5）以找到最优配置

**如果 DTA Full-Scale 仍为负**：
1. 论文重新定位为"Inference-Time Enhancement for Masked Diffusion: Why Deeper Isn't Better"
2. 核心贡献变为：(a) 信息增强谱系的完整消融，(b) 浅层干预优势的发现，(c) Token 级诊断框架（Correction Precision/Recall），(d) DMI 作为实用的 zero-cost 增强方法
3. DTA 的负面结果作为"为什么参数级适应在 DLM 上失败"的分析章节

**如果 DTA Full-Scale 出现翻转**：
1. 保持原始 proposal 框架
2. 重点分析为什么 pilot/Full-Scale 出现系统性不一致
3. 强调大样本验证的方法学贡献

### 不推荐完全放弃
- 当前数据量（尤其是 DTA 只有 pilot 级别）不足以完全否定 DTA
- DMI 的 pilot → Full-Scale 翻转先例告诫我们不要过早下结论
- 但**应将研究资源从"优化 DTA 超参数"转向"理解 DMI 为什么有效"和"完成 Full-Scale 统计检验"**

---

## 附录：证据可信度评级

| 证据来源 | 样本量 | 可信度 | 备注 |
|---------|--------|--------|------|
| Full-Scale Countdown (vanilla/ReMDM/RCR/DMI) | 500 × 3 seeds | **高** | 统计可靠 |
| Full-Scale PPL (旧迭代) | 101 × 3 seeds | **中-高** | PPL 指标本身有争议 |
| Pilot Countdown 全方法 (task_5a) | 16 × 1 seed | **低** | 1 题 = 6.25%，噪声极大 |
| Pilot GSM8K (task_5b) | 16 × 1 seed | **低** | 同上 |
| Pilot MBPP (task_5c) | 16 × 1 seed | **低** | DTA 正向结果待验证 |
| Scaling Curve (task_6a) | 16 × 4T | **极低** | 完全在噪声中 |
| Rank/Gamma/Freq Ablations (task_7*) | 16 × 1 seed | **低** | 无法区分超参数效应 |
| LLaDA Cross-Model (task_8a) | 16 × 1 seed | **低** | 方向信号有参考价值 |
| Token Diagnostics (task_8b) | 16 × 1 seed | **中** | Precision/Recall 作为比率指标对样本量更鲁棒 |
| Degradation Analysis (task_8c) | 16 × 1 seed | **中** | 范数轨迹是连续指标 |
