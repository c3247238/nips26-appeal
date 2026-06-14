# Lessons from Iteration 15

## Must Improve

- **[INTEGRITY FIXES BEFORE ANYTHING ELSE -- BLOCKING]**: iter_015 评分从 7.0 降至 6.5，唯一原因是 iter_014 标记的完整性问题在 iter_015 未修复。iter_016 的第一个动作必须是以下四项零 GPU 成本文本修复，合计约 4 小时：
  1. AIS=0.566 → 0.123，加 LOO-CV R^2=-0.18 警告（30 min）
  2. Table 1 CWD K_d → 承认 scale=0.5 而非 K_d（1 hr）
  3. CSI 标准化为单一公式，重算 200-epoch 值（2 hr）
  4. Theorem 1 → Conjecture 1，移除"Theoretical analysis"贡献（30 min）
  **不做这些就做其他任何工作都是错误的优先级排序。**

- **[CWD HALVED-LAMBDA -- P0, MUST EXECUTE IN ITER_016]**: FixedWD @ lambda=5e-5，CIFAR-10 (3 seeds, 200 epochs) + ImageNet (3 seeds, 90 epochs)。总计约 7 GPU-hours。已标记 P0 四个迭代从未执行。这是项目中信息密度最高的实验。如果 halved-lambda 匹配 CWD (90.32%)，则 alignment mask 无价值；如果 CWD 仍更好，alignment 确实携带信息。

- **[COMPLETE IMAGENET NoWD -- P0]**: NoWD x 3 seeds, 90 epochs。这是 BEM 分母，没有它 BEM 在 ImageNet 尺度不可计算。约 24 GPU-hours。使用 GPU pair 0-1。

- **[TITLE SCOPE MUST MATCH CONTENT]**: "Unified Feedback Control Framework" 在 2/5 方法拟合失败时不准确。改为 "A PID-Style Taxonomy for Dynamic Weight Decay" 或类似表述。

- **[UDWDC IS A DIAGNOSTIC PROBE, NOT A CONTRIBUTION]**: 从贡献列表移除。重新定义为 "proportional-only control is insufficient" 的诊断工具。

## Watch Out

- **[Do NOT rewrite the paper]**: 大幅重写历史上导致 -2.0 评分崩塌（iter_003, iter_006）。只做定向编辑。

- **[Action plan must be consumed by next writing agent]**: reflection → action_plan 的反馈回路在 plan-to-execution 边界断裂。写作 agent 必须逐项检查 action_plan.json 中的 issues_classified 并标记哪些已修复。

- **[Population vs sample std]**: 全部表格改用 ddof=1。N=3 时差异高达 22%。

- **[CPR budget confound]**: CPR 使用 2.3x WD budget。需 budget-matched FixedWD @ lambda=2e-4/3e-4。

- **[UDWDC-v2 BN bug]**: Floor clipping 对 41 BN 层施加 lambda_min，产生 205,000x WD budget。一行代码修复。

- **[BEM negative result must be reported]**: EqWD 在 Bayesian optimization 下不击败 tuned FixedWD。隐瞒已执行的负面结果比披露更有害。

- **[Figure 1 is TikZ, not data]**: 替换为真实实验图。

## Keep Doing (Success Patterns)

- **交叉验证纪律**: 所有表格数值与原始 JSON 零差异，连续 3+ 迭代保持。扩展至指标值和相关系数。

- **负面结果诚实报告**: 摘要预注册、Section 7.6 合并五个负面发现、Section 3.3 "engineering patches"。这是论文最强差异化优势。

- **严格统计方法**: paired t-tests + Bonferroni、Cohen's d、TOST、power analysis。维持此水平。

- **Pre-generated 图表管道**: 13 张出版质量图表已就绪。嵌入而非重新生成。

- **定向编辑优于重写**: 增量改进安全可靠。

- **证据先行写作**: 先检查原始数据再写声称。AIS=0.566 的教训：违反此模式是完整性问题的根源。
