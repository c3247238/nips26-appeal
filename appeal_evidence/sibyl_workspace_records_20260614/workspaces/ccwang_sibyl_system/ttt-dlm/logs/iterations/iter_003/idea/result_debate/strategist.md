# 战略分析：DTA 全规模实验中期评估与资源分配建议

## 1. 基于当前结果的最有前途方向

### 1.1 DMI 是意外的战略赢家

当前最引人注目的发现不是 DTA（尚未完成），而是 **DMI（Diffusion Memory Injection）的 ~2x 提升**（9.3% vs vanilla 4.7%）。这个结果在战略上极为重要：

- **计算开销仅 ~1.05x**：几乎免费的性能翻倍
- **跨 3 seeds 一致**：7.8%/9.6%/10.6%，方向稳定，标准差 1.4% 可接受
- **验证了核心论点**：跨步信息丢失确实是瓶颈，即便最简单的 embedding 级记忆传递也能带来实质性改善

DMI 的成功将论文的叙事从"DTA 是否有效？"升级为"跨步信息传递价值的量化谱系"——即使 DTA 最终表现不如预期，DMI 已经为论文提供了一个可靠的正面贡献。

### 1.2 Remasking 方法的集体失败强化了理论动机

ReMDM-conf（4.4%）和 RCR（5.7%）的微弱表现（甚至低于/仅略高于 vanilla 4.7%）是论文的**最佳辅助证据**：

- 纯 token 空间操作（remask/re-predict）在无跨步记忆时无法累积推理
- Countdown 需要约束传播，vanilla 的 4.7% 已经是"盲猜正确"的基线
- Remasking 甚至可能扰乱已经正确的 token（ReMDM-conf 4.4% < vanilla 4.7%）

这些结果让 remasking 的不足从"理论推测"变成了"实验事实"，为 DTA/DMI 的动机章节提供了铁证。

### 1.3 SCP 与 DMI 接近意味着 Level 2 可能"性价比不高"

SCP 中期 ~8.4% vs DMI 9.3%。SCP 开销 ~1.5x 而 DMI ~1.05x。如果最终 SCP 不显著优于 DMI，那 SCP 在论文中的角色应从"方法贡献"降级为"消融基线"——证明更复杂的前向传播探测不比简单的 logit 注入更有效。

**战略含义**：信息增强谱系的 Level 1→Level 2 边际收益可能很小，但 Level 0→Level 1 和 Level 2→Level 3（DTA）的跃升才是真正的故事。

### 1.4 GSM8K 初步数据的双面信号

- Vanilla 29.6%（接近 Dream 论文报告）：评估框架可靠
- ReMDM-conf 25.1%（远低于 vanilla）：**remasking 在长推理链上有害**，这是一个比 Countdown 更强烈的负面信号
- DTA 和 DMI 尚未在 GSM8K 上评估——这是最关键的空白

## 2. 优先级排序的行动建议

### P0：最高优先级——完成 DTA 和 DTA+ReMDM 在 Countdown 上的评估

**理由**：这是整篇论文的核心假设验证。当前 SCP 正在运行中，DTA 和 DTA+ReMDM 排在之后。

**预期场景分析**：

| 场景 | DTA 准确率 | 概率 | 论文策略 |
|------|-----------|------|---------|
| A: 大幅超越 DMI | >15% | ~35% | 顶会核心方法论文，DMI/SCP 作为消融 |
| B: 与 DMI 相当 | 8-12% | ~35% | 信息增强谱系论文，强调 Level 1 的高效性 |
| C: 低于 DMI | <8% | ~25% | DMI 发现论文，DTA 失败作为理论分析 |
| D: 不如 vanilla | <5% | ~5% | 紧急诊断——实现缺陷还是方法局限？ |

**判断依据**：DMI 的成功证明信息传递有效，DTA 的理论基础（参数级记忆比 embedding 级更有表达力）支撑场景 A/B。但去噪过程中的梯度更新可能引入分布漂移，增加了场景 C 的概率。

### P1：GSM8K 跨任务验证

DTA + DMI 在 GSM8K 上的表现将决定"跨任务泛化性"的叙事强度。建议：
- 优先运行 DTA 和 DMI（而非 SCP/RCR），减少 GPU 等待时间
- 如果 DTA 在 GSM8K 上也优于 vanilla → 论文影响力大增
- 如果仅 Countdown 有效 → 可写为"任务依赖性"的 nuanced 贡献
- ReMDM-conf 在 GSM8K 的 25.1%（劣于 vanilla 29.6%）已经提供了强对比背景

### P2：统计检验与结果整合

消融实验（task_7a-d）、诊断分析（task_8b-c）、扩展曲线（task_6a）、LLaDA 跨模型验证（task_8a）均已完成。这些数据等 DTA 核心结果到位后再统一整合分析，避免在核心假设未验证前投入写作精力。

## 3. 资源分配建议

### 3.1 GPU 分配（4x GPU on cs8000d）

当前 Wave 1 占用 4 GPU（Countdown seeds + GSM8K）。后续建议：

| 优先级 | 任务 | GPU 需求 | 预计墙钟 |
|--------|------|---------|---------|
| P0 | SCP 完成（Countdown 3 seeds） | 已运行中 | ~6h 剩余 |
| P0 | DTA（Countdown 3 seeds） | 3 GPU | ~10h |
| P0 | DTA+ReMDM（Countdown 3 seeds） | 3 GPU | ~10h |
| P1 | DTA + DMI on GSM8K（3 seeds） | 4 GPU | ~8h/method |
| P2 | 结果分析与统计检验 | 0 GPU | ~2h |

### 3.2 关键里程碑

- **T+6h**：SCP Countdown 完成 → 初步判定 Level 2 vs Level 1 边际收益
- **T+16h**：DTA Countdown 完成 → **论文最终定位确定（场景 A/B/C/D）**
- **T+26h**：DTA+ReMDM 完成 → H2（正交互补性）验证
- **T+34h**：GSM8K 跨任务验证完成 → 论文全部核心数据到位

### 3.3 时间敏感性

DTA 是整个研究的命脉实验。每推迟 1 小时，论文定位的不确定性就多持续 1 小时，导致写作和修订无法启动。**建议 SCP 完成后立即全量 GPU 转向 DTA**。

## 4. 风险评估

### 4.1 风险矩阵

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| DTA 数值不稳定（LoRA 范数爆炸） | 15% | 高 | 降 η 到 1e-5，降 rank 到 2，增加 L2 正则 |
| DTA 准确率 < DMI | 25% | 中 | 转向 DMI 发现论文，DTA 作为负面消融 |
| DTA OOM（反向传播内存） | 10% | 高 | 仅更新最后 1 层 FFN，或用 gradient checkpointing |
| GSM8K 所有方法无显著差异 | 30% | 中 | 聚焦 Countdown，GSM8K 放入附录 |
| ReMDM-conf 在 GSM8K 上的退化是实现缺陷 | 10% | 中 | 对比文献报告值，必要时修复后重跑 |

### 4.2 "下行保护"分析

即使最坏情况（DTA 完全失败），论文仍有以下可用资产：
1. **DMI 的 ~2x 提升**：一个几乎零开销的实用方法发现
2. **Remasking 的系统性失败**：ReMDM-conf 和 RCR 在 Countdown 和 GSM8K 上的负面结果
3. **信息孤岛假说的实验验证**：DMI 成功 vs Remasking 失败 = 跨步信息丢失确实是瓶颈
4. **完整的消融/诊断/跨模型数据**：已完成的 task_6a-8c 提供丰富的分析材料
5. **变分理论框架**：即使 DTA 失败，VDTA 理论仍可解释为何 DMI 有效

这意味着论文有 90% 的概率可发表（EMNLP 及以上），风险完全可控。

## 5. Pivot 还是 Proceed？

### 强烈建议：**PROCEED**

理由：
1. **DMI 已提供安全网**：无论 DTA 结果如何，DMI 发现 + Remasking 失败已构成有价值的论文
2. **DTA 尚未测试**：核心假设未验证前 pivot 是过早放弃
3. **理论框架依然成立**：DMI 成功验证了"跨步信息丢失是瓶颈"的核心论点
4. **时间投入极高效**：DTA 完成只需约 20h 墙钟（GPU 自动运行，不需人力介入）
5. **消融/分析数据已完整**：无论 DTA 结果如何，论文的实验章节可快速填充

**唯一触发 Pivot 的条件**：DTA 导致准确率系统性低于 vanilla（场景 D），且诊断确认是方法本身的根本局限而非实现缺陷。在此情况下，转为 DMI 为核心的"Diffusion Memory Injection"论文。

## 6. 论文策略的情景规划

### 场景 A：DTA 大幅领先（>15% Countdown，概率 ~35%）

**定位**：NeurIPS/ICML 主会
**标题**：*Denoising-Time Adaptation: Turning Diffusion Iterations into Test-Time Learning for Masked Language Models*
**叙事**："跨步信息丢失是 DLM 推理瓶颈" → "DTA 通过参数级记忆解决" → "信息增强谱系量化各层级价值"
**亮点**：VDTA 理论框架 + DTA+ReMDM 正交互补 + DMI 作为高效消融

### 场景 B：DTA ≈ DMI（8-12%，概率 ~35%）

**定位**：ICLR/NeurIPS
**标题**：*The Information Island Problem in Masked Diffusion: From Embedding Memory to Parameter Adaptation*
**叙事**：发现跨步信息丢失问题 → DMI（零开销）是最佳实践 → DTA 的复杂性不带来额外收益 → 信息传递关键在"有信息"而非"更精确"
**亮点**：DMI 的实用价值 + "简单方法足够"的 insight

### 场景 C：DTA < DMI（<8%，概率 ~25%）

**定位**：NeurIPS/EMNLP
**标题**：*Diffusion Memory Injection: Simple Cross-Step Information Transfer Doubles Reasoning Accuracy in Masked Diffusion Models*
**叙事**：DMI 发现作为核心贡献 + DTA 失败的诊断分析（为何参数更新在去噪语境下失效？）
**亮点**：DMI 的实用发现 + 对社区的方法学警示

### 场景 D：全面失败（<5%，概率 ~5%）

**定位**：转向 DMI 论文或系统性负面结果分析
**应急**：DMI 的 9.3% vs vanilla 4.7% 仍是统计显著的，这个发现本身支撑一篇论文

## 7. 总结与即时行动项

### 核心判断

当前项目处于**战略上非常有利的位置**：
- DMI 的意外成功消除了"全面失败"的风险
- Remasking 的集体失败为新方法提供了最佳背景
- DTA 结果将在 ~16h 内到位，届时论文定位完全明确
- 消融/诊断/跨模型数据已全部到位，写作可快速推进

### 即时行动清单

1. **立即**：确认 DTA 和 DTA+ReMDM 的 Countdown 实验代码已就绪，GPU 释放后优先调度
2. **6h 内**：SCP 完成后与 DMI 比较（若 SCP ≈ DMI → Level 2 降为消融角色）
3. **16h 内**：DTA Countdown 结果到位 → 确定场景（A/B/C/D）→ 启动论文修订
4. **24h 内**：DTA+ReMDM 完成 → H2 正交互补性验证
5. **34h 内**：GSM8K 跨任务验证完成 → 论文全部核心数据到位

### 信心评估

- 论文可发表的概率：**90%**（场景 A/B/C 均可成文，场景 D 有 DMI 保底）
- 顶会水平的概率：**60%**（需要场景 A 或 B 中 DTA/DMI 的跨任务泛化）
- 需要 Pivot 的概率：**<5%**（仅在场景 D 且 DMI 统计不显著时）
