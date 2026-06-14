# 本轮迭代教训

## 必须改进

- **论文必须重新定位——停止将未验证/已证伪方法放入标题**: A-CFG 在 N=100 决策门中被判定 NO-GO（2% vs vanilla 4%），BSD 在所有评估中不如 DMI。标题和摘要必须围绕 DMI（唯一经验证方法）和系统性设计空间约束重构。这是连续 3 轮的 recurring pattern，必须在迭代 5 终止。

- **DMI 消融是投稿 blocking 任务**: DMI 是唯一核心贡献但连续 2 轮未获消融实验。alpha/tau sweep、随机 embedding 负对照、hard top-1 基线——这些是审稿人必问的问题。迭代 5 第一天就应启动。

- **图表渲染是投稿基本门槛**: 连续 4 轮零渲染图表。数据和描述均已就绪。至少 Figure 1 (teaser)、Figure 4 (ablation)、Figure 6 (diagnostics)。不需要 GPU，可与消融实验并行。

- **篇幅压缩到 6,000 词**: 将 BSD 详细分析、所有 n=16 消融、Section 5.1-5.3 机制推测移至附录。正文只保留 DMI 验证 + 设计空间约束总结 + 方法论贡献。

- **修复 compute-fair 数据一致性**: 当前 Table 4 混合了两次不同 run 的数据（不同 seed/样本集）。所有方法必须在同一次评估中、相同条件下运行。

- **DMI 提升到正文独立 Section**: DMI 不应在 Appendix D。作为唯一经验证的贡献，它应获得最详细的正文处理。BSD 降级为 Discussion 中的 negative result 分析。

## 需要注意

- **N=100 决策门是成功的方法论改进**: 本轮成功用 19 分钟判定 A-CFG NO-GO，避免了 ~18 GPU-hours 浪费。未来所有新方法都应先过 N=100 决策门再做全规模。

- **'泛化优于简单方法'假设在 MDLM 中反复失败**: DTA、BSD、A-CFG 均不如更简单的 DMI 或 vanilla。迭代 5 不应再探索更复杂的方法，而应深入理解 DMI 为什么有效。

- **Entropy-accuracy 相关性（r=0.78）基于 n=16 中 1/16 correct，统计可信度极低**: 降级为描述性观察，不做因果推断。

- **LLaDA-8B 多模型验证**: 如果 DMI 在 LLaDA-8B 上也有效，显著增强贡献可信度。A-CFG 在 LLaDA-8B 上的 73.5 vs Dream-7B 上的 NO-GO 说明模型特异性很强。

- **A-CFG 应重新定位为 "transfer experiment"**: 不是原创贡献（Arriaga et al. 2025 提出），在 Dream-7B 上无修改应用且失败。

- **时间计划校准**: 实际/计划比为 0.18（5.6x 过估）。迭代 5 计划时使用此校准比。

## 做得好的（继续保持）

- **N=100 决策门成功落地**: 上轮提出的"早期证伪检查点"建议在本轮成功实施并证明价值
- **科学诚实性一以贯之**: A-CFG NO-GO、BSD 不如 DMI、pilot 膨胀均如实记录——4 轮迭代的一贯品质
- **实验零失败率**: 18/18 任务全部完成，实验 pipeline 成熟可靠
- **Pre-registered hypotheses 框架**: 11 个假设 × 验证表格，科学透明性典范
- **Information Island Problem 概念**: 被所有审查者一致认可为有价值的学术贡献
- **快速止损能力**: A-CFG NO-GO 后立即停止投入，DTA null result 后立即转向
