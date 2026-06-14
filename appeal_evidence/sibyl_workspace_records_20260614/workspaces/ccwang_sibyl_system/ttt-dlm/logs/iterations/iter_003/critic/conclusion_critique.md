# Conclusion 章节评审意见

## 评分: 6/10

## 优点

1. **核心结论表述清晰有力。** "none of these strategies produce statistically significant quality improvements"一句加粗处理，直接点明负面结果，没有含糊其辞。读者能在第一段就抓住论文的核心发现。

2. **三个 key lessons 提炼精准。** PPL 不可靠、小样本高估、内部信号不足，这三个教训确实是本研究最具社区价值的贡献，概括得当。

3. **"constructive implication"段落有效地将负面结果转化为建设性方向。** 指出成功方法的共同线索是"引入外部信息源"，这一归纳具有理论指导意义。

4. **篇幅控制得当。** 作为结论章节，没有引入新的分析或过度重复前文，基本保持了总结性质。

## 不足

1. **与 Discussion 的重复度过高（约 70%）。** Conclusion 的三个 key lessons 几乎是 Discussion 6.1-6.2 的逐段压缩，未提供任何新的视角或更高层次的抽象。学术论文的 Conclusion 应在 Discussion 基础上做更高维度的归纳，而非简单重述。当前写法让读者感觉在读 Discussion 的缩写版。

2. **缺少对论文贡献的明确编号列举。** Outline 中明确列出了本文的三大贡献（PPL 不可靠性证据、规模依赖退化陷阱、DLM 推理时扩展困难的实证），但 Conclusion 中这些贡献被嵌入在叙述性文字中，没有显式标注。建议使用编号列表明确列出贡献，方便审稿人评估。

3. **局限性讨论过于简略。** 仅一句话带过了两个局限性（开放文本 vs 推理任务、未测试 ReMDM 采样器），而 Discussion 6.5 节列出了五个。Conclusion 中不需要重复所有细节，但至少应提及评估指标局限性（PPL 为主）这一最关键的限制。

4. **未来方向过于模糊。** "exploring minimal training interventions (e.g., lightweight remasking SFT)"是唯一具体的建议，其余（"extending to reasoning benchmarks"）太过泛泛。Discussion 6.4 中提到的外部验证器、RL 后训练、混合架构三个方向在 Conclusion 中仅一笔带过，缺少优先级排序和可操作性。

5. **缺少对方法论贡献的强调。** 本文提出的"100 prompts x 3 seeds"评估基线是一个重要的方法论贡献，但在 Conclusion 中仅作为 second key lesson 的附属内容出现，未被提升到贡献层面。

6. **最后一句过于理想化。** "redirecting effort from training-free heuristics toward methods that fundamentally enrich the information available during inference"是一个很强的声明，但本文的实验覆盖范围（仅开放文本生成、仅三个模型、仅 PPL 评估）是否足以支撑如此宏大的社区建议？应加入适当的 hedge。

7. **与 outline 的偏差。** Outline 结论部分明确包含"未来方向"作为独立段落（推理任务验证、简化版 RemeDi、理论分析），当前 Conclusion 将未来方向压缩成一句话混入局限性段落中，结构不清。

## 具体修改建议

1. **重构为"贡献 + 教训 + 局限 + 展望"四段式结构。**
   - 第一段：核心发现与实验规模总结（保留当前第一段，略微精简）
   - 第二段：明确编号列出本文三大贡献（负面结果证据、方法论建议、对 DLM TTS 机理的洞察）
   - 第三段：主要局限性（2-3 个最关键的，不重复 Discussion）
   - 第四段：未来方向（按优先级排序，给出具体可操作的研究问题）

2. **提升抽象层次，减少与 Discussion 的重复。** Conclusion 不应重述"0.6B 模型退化率 71%"等具体数据，而应聚焦于更宏观的归纳。例如："本文从三个互补维度证明了轻量级策略的根本局限性：信号维度（内部不确定性估计不可靠）、优化维度（去噪收敛到局部最优）、评估维度（PPL 无法捕捉质量差异）。"

3. **明确列出贡献。** 建议格式：
   ```
   本文的主要贡献包括：
   (1) 首个系统性的 DLM 轻量级推理时扩展负面结果...
   (2) 揭示了 PPL 作为 DLM remasking 评估指标的不可靠性...
   (3) 提出了 DLM 推理时扩展实验的最低评估标准...
   ```

4. **未来方向独立成段并排序。** 建议按可行性和影响力排序：
   - 近期：在推理任务（GSM8K、HumanEval）上验证本文发现
   - 中期：探索轻量级训练干预（remasking SFT、quality-score head）
   - 远期：DLM 内部信号不足的理论分析

5. **软化最后一句的语气。** 将"redirecting effort"改为"encouraging the exploration of"或类似表述，并加入 caveat（"at least for open-ended generation tasks"）。

6. **补充方法数量的准确性。** 第一段提到"eight lightweight strategies"，但具体列举了 ReMask-Retry、六个 TTT 变体、Best-of-N、TCR、温度退火、熵引导重掩码、并行投票，这远超八个。应统一计数方式（是按大类还是按具体变体），与 Abstract 和 Methods 保持一致。

## 交叉一致性检查（Discussion vs Conclusion）

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 统计结论一致性 | 一致 | 两章均报告 p > 0.25，无显著改善 |
| 方法数量表述 | **不一致** | Discussion 未明确计数；Conclusion 说"eight"但列举超过八个 |
| PPL 不可靠性论证 | 重复过多 | Discussion 6.1 第三段和 Conclusion 第二段高度重复 |
| 局限性覆盖 | **不一致** | Discussion 列出 5 个局限性，Conclusion 仅提及 2 个，且遗漏了最重要的"评估局限性（PPL 为主）" |
| 未来方向 | **不一致** | Discussion 6.4 给出 3 个具体方向（外部验证器、RL、混合架构），Conclusion 压缩成 1 句话且重点不同（强调 reasoning benchmarks 和 lightweight SFT） |
| 0.6B 退化数据 | 一致 | 两章均引用 71% 退化率和 47.5% PPL 下降 |
| "外部信息源"核心论点 | 一致 | 两章均强调成功方法需要引入外部信息 |
| Pilot vs Full-scale 数据 | 一致 | Discussion 表格数据与 Conclusion 引用的 16-25% 范围吻合 |
| 与 outline 的对齐 | **两章均有偏差** | Discussion 丢失了"规模依赖行为差异"分析；Conclusion 丢失了"未来方向"独立段落和贡献编号列表 |
