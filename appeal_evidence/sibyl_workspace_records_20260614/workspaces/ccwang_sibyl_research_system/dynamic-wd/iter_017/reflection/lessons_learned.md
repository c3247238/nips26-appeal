# Lessons from Iteration 16

## Must Improve

- **[BUDGET-MATCHED FIXEDWD -- BLOCKING, P0]**: EqWD 的 phi_l(t) >= 1 设计意味着它总是比 FixedWD 施加更多 WD。+0.38% ImageNet 改进可能完全来自更强正则化。在 iter_017 的第一个动作必须是：测量 EqWD 训练过程的平均有效 lambda，然后运行 FixedWD 在该 lambda 值，3 seeds，ImageNet。~9 GPU-hours。这决定论文叙事是"自适应调制有益"还是"自动发现更好的 WD 强度"。两者都可发表，但必须知道是哪一种。**不做这个实验就做其他工作都是错误的优先级。**

- **[VERIFY DATA DISCREPANCIES -- BLOCKING, P0]**: 两个可能的事实性错误必须立即确认：
  1. Lambda: 原始数据显示 1e-4，论文声称 5e-4（5x 差异）
  2. LR schedule: 原始数据显示 step decay（epoch 30 处 10x 下降），论文声称 cosine annealing
  如果论文描述与实际代码不符，这是可重现性错误。验证代码，修正论文。~2 小时。

- **[STATE 45-EPOCH IMAGENET EXPLICITLY]**: 论文从未明确说明 ImageNet epoch 数。标准训练 90 epochs 达到 ~76%，EqWD 仅 72.27%。必须在 Section 4.1 和 appendix 中明确声明 "45 epochs"。0 GPU。

- **[90-EPOCH IMAGENET COMPARISON]**: 运行 EqWD、FixedWD、SWD 各 3 seeds，90 epochs。验证 EqWD 优势是否在标准训练长度下保持。~18 GPU-hours。与 budget-matched 实验并行运行。

- **[EQWD ON CIFAR-10]**: 论文主方法在三个基准表之一中缺失。运行 EqWD + CAWD on CIFAR-10，3 seeds，200 epochs。~0.5 GPU-hours。同时移除或解释未定义的 DefazioCorrective。

- **[ADAMW EXPERIMENT]**: 2026 年论文仅测试 SGDW 是严重范围限制。至少运行一个 AdamW + EqWD 实验（ResNet-50 CIFAR-100，~30 min）。

## Watch Out

- **[DO NOT REWRITE THE PAPER]**: 大幅重写历史上导致 -1.5 分平均崩塌（iter_003: -2.7, iter_006: -2.0, iter_013: -0.5）。iter_016 的 pivot 是必要的例外，但未来应只做定向编辑。

- **[CONSUME ACTION PLAN]**: iter_015 action_plan 没有被 iter_016 执行。写作 agent 必须在开始写作前逐项检查 action_plan.json 中的 issues_classified，标记哪些已处理。

- **[PROPOSITION 2 IS VACUOUS]**: "if alignment is a function of norms, then ratio captures alignment" 是同义反复。降级为 Remark/Observation。让 AIS 实验承载论证。

- **[SINGLE-SEED CLAIMS REQUIRE VALIDATION]**: beta=5.0 的 66.07% 是单种子，与 3-seed 均值 65.05% 差 1%。30 分钟多种子验证可解决。不要在论文中依赖单种子结果。

- **[FIGURE-DATA CONSISTENCY CHECK ON PIVOT]**: CIFAR-10 表从旧框架遗留，包含 DefazioCorrective（未定义）且缺少 EqWD。每次论文 pivot 时必须检查所有表格和图表是否与新框架一致。

- **[DIFFERENTIATE FROM DEFAZIO]**: Defazio (arXiv:2506.02285) 使用相同的 gradient-to-weight ratio 动力学。必须在 Related Work 中明确区分 EqWD 与 Defazio 的 corrective term。

## Keep Doing (Success Patterns)

- **范围缩减是最有效的完整性修复**：删除无法支撑的声称比修补更安全、更彻底。iter_016 证明了这一点。

- **CAWD 负面结果 = 方法论贡献**：精心设计的消融实验的负面结果比正面结果更有信息价值。维持这种实验设计水平。

- **诚实的统计校准**：bootstrap CIs 包含零时明确声明，"directional trend" 而非 "significant improvement"。这是论文最强的差异化优势，必须维持。

- **数据先行写作**：先检查原始数据再写声称。AIS=0.566 的教训不能重演。

- **定向编辑优于重写**：增量改进安全可靠。每次重写平均 -1.5 分 + 3-5 迭代恢复期。

- **完整 ImageNet 多种子比较**：7 方法 x 3 seeds 是真正的实验里程碑。扩展到 90 epochs 以获得可比较的绝对数值。

- **出版质量图表管道**：9 张图表质量高且可复用。添加 ImageNet 版本的 ratio trajectories 和 WD heatmap 以加强叙事。
