# 西比拉全局经验总结 (自动生成)


## EXPERIMENT 类问题

影响 agent: experimenter, server_experimenter, planner

- [MEDIUM] DMI 是论文核心贡献但缺乏系统消融。alpha 仅 0.3、tau 仅 0.5。无负对照（随机 embedding 注入）、无 hard-prediction 基线、无 multi-step lookback。 (出现 2 次, 权重 1.8)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## WRITING 类问题

影响 agent: sequential_writer, section_writer, editor, codex_writer

- [MEDIUM] 所有 6 张图仍为文字描述，无渲染图表。连续 4 轮未解决。实验驱动论文无图表是投稿基本门槛问题。 (出现 2 次, 权重 1.8)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] 论文篇幅仍远超 NeurIPS 6,000 词限制。连续 2 轮未解决。可能直接 desk reject。 (出现 2 次, 权重 1.8)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] 参考文献占位符：多处 arXiv:2505.xxxxx 未替换。 (出现 2 次, 权重 1.8)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] 代码开源计划仍未提及。实验驱动论文无代码开源承诺降低可复现性。 (出现 2 次, 权重 1.8)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。

## 成功模式 (继续保持)

- Pipeline 完整跑通：从文献调研到 LaTeX 编译，全流程自动化成功 (出现 1 次)
- 写作质量出色（8.5/10）：结构清晰、措辞精确、自我批评诚实 (出现 1 次)
- 局限性声明全面诚实：Section 5.3 主动标记了 N=6、confounded comparison 等所有关键限制 (出现 1 次)
- GPU 并行调度成功：两个实验在 2 GPU 上成功并行执行 (出现 1 次)
- Result debate 质量高：六个视角的讨论深入，正确识别了 H1 被否定 (出现 1 次)
- 多轮修订有效：经过两轮 critique-revision 循环，写作一致性显著提升 (出现 1 次)
- 数据结构化存储：实验结果以 per-prompt JSON 存储，具备可复现性 (出现 1 次)
- 诚实报告负面结果：DTA null result (4.8% vs 4.7%)、DTA+ReMDM degradation (3.6%)、pilot 膨胀 (~24pp) 均被明确报告——科学诚实的典范 (出现 1 次)
- 叙事重构的果断性：DTA 数据出来后立即重构标题和框架，没有试图掩饰失败 (出现 1 次)
- DMI 作为实用贡献的定位：~2x 改善、~1.2x 开销、<10 行代码——社区可实际采用 (出现 1 次)
