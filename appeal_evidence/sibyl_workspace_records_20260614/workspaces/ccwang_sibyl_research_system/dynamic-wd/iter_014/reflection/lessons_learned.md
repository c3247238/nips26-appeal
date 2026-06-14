# Lessons from Iteration 14

## Must Improve

- **[DATA INTEGRITY -- HIGHEST PRIORITY]**: AIS=0.566 在论文中无任何支持数据。实际测量值为 AIS=0.123（Spearman rho=0.195）。这是伪造数据，必须在任何其他工作之前替换。同样，CWD K_d 映射声称（Table 1: "K_d > 0, Derivative only"）被拟合数据直接推翻（K_d≈0，拟合由 scale=0.5 驱动）。交叉验证必须覆盖**所有**数值声称，不仅限于准确率表格。

- **[COMPLETE IMAGENET -- #1 EXPERIMENT PRIORITY]**: ImageNet 仅有 4/7 方法，已标记 10+ 个迭代。剩余 28/40 runs 必须完成：NoWD（BEM 分母必需）、SWD、DefazioCorrective（各 3 seeds）、CWD（额外 2 seeds）。8 GPUs 并行可在 2-3 天内完成。绝不再延迟。

- **[RUN CWD HALVED-LAMBDA ABLATION IMMEDIATELY]**: FixedWD @ lambda=5e-5 on CIFAR-10 (3 seeds, 200 epochs) + ImageNet (3 seeds, 90 epochs)。总计约 7 GPU-hours。这是项目中最高信息密度/GPU-hour 比的实验。已被标记为 P0 三个迭代但从未执行。必须在下一迭代的前 2 小时内启动。

- **[FIX CSI FORMULA]**: 论文中有三个互相矛盾的 CSI 公式，Table 6 的 CSI=-5.75 在任何已声明公式下不可能。标准化为一个公式：CSI = 1/(1 + Var_t[lambda_eff^l / mean_t[lambda_eff^l]])，从 200-epoch 全量数据重新计算。替换 10-epoch pilot 值。

- **[RESOLVE THEOREM STATUS]**: Theorem 1 和 Propositions 2-3 无证明，已 6+ 个迭代未解决。最快路径：降格为 Conjecture/Hypothesis（30 分钟文字编辑），从贡献列表移除"理论分析"。执行此操作。

## Watch Out

- **[Do NOT rewrite the paper again]**: 评分在大幅重写时崩塌：-2.0（iter_003），-2.0（iter_006），-0.5（iter_013）。当前论文结构合理。仅做定向编辑和添加实验。

- **[Title scope must match content]**: "Unified Feedback Control Framework"在 2/5 方法拟合失败时不准确。改为反映实际范围的标题，如"A PID-Style Taxonomy for Dynamic Weight Decay"。

- **[UDWDC is a diagnostic probe, NOT a contribution]**: UDWDC 在所有基准上劣于 FixedWD 和 NoWD。将其从贡献列表中移除为"proposed algorithm"。明确标注为"proportional-only control is insufficient"的诊断工具。

- **[Population vs sample std]**: 全部表格使用 ddof=1 样本标准差。当前 ImageNet 结果使用总体标准差，在 N=3-5 时差异显著。

- **[CPR budget confound is the most important unresolved question]**: CPR 的 WD budget 是 FixedWD 的 2.3x。没有预算匹配控制（FixedWD @ lambda=2e-4/3e-4，90 epochs），CPR 的 3.02pp 优势无法归因于更好的 WD 分配 vs 更强的正则化。

- **[UDWDC-v2 BN layer bug]**: Floor clipping 对所有 65 层（含 41 BN 层）施加 lambda_min，产生 205,000x WD budget。修复一行代码：仅对 weight 层应用控制。在报告 v2 结果前必须修复。

- **[AIS LOO-CV R^2 is negative]**: 这意味着 AIS 的样本外预测能力为零。论文必须明确承认这一点，而非声称 AIS 是有效的诊断指标。

## Keep Doing (Success Patterns)

- **数据交叉验证**：Supervisor 本次交叉验证了所有 CIFAR 和 ImageNet 表格数值与原始 JSON 文件，零差异。这是论文可信度的基础。扩展到指标值和相关系数。

- **负面结果诚实报告**：摘要预注册负面发现、Section 7.6 合并五个负面结果、Section 3.3 ("engineering patches, not principled solutions") — 这些是论文最强的差异化优势，产生审稿人好感。

- **Pre-generated 图表管道**：13 张出版质量图表已在 exp/results/full/figures/ 准备就绪。嵌入而非重新生成。

- **3 GPU 并行调度**：ImageNet 实验的 gpu0/gpu3/gpu6 并行策略是最佳实践。所有大规模实验都应采用。

- **证据先行写作**：先检查原始数据，再写声称。违反此模式（AIS=0.566）是完整性问题的根源。

- **定向编辑优于重写**：增量改进（+0.25/迭代）比大幅重构（-2.0 风险）安全得多。

- **严格统计方法**：paired t-tests with Bonferroni correction, Cohen's d, TOST equivalence testing, power analysis — 高于社区标准，维持此水平。
