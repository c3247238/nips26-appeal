# Lessons from Iteration 8

## Must Improve

- **数据完整性检查完全缺失，且问题在恶化**: 从 iter_005 的符号错误，到 iter_007 的重复 variant，到 iter_008 的 4/5 replicate 数据复制和 1k-vs-16k 规模虚报。下一次迭代必须在分析和写作之前强制执行三项检查：(1) 跨 variant 重复检测（replicate metrics 的 MD5 哈希），(2) num_features 与计划规格的核对，(3) 所有承诺实验都有输出文件的检查。

- **gpu_progress.json "completed" 状态不可靠**: e2_l0_matched、e3_ksweep、e4_dead_latent、e5_training_dynamics 都被标记为 "completed"，但实际结果文件不存在。完成状态必须要求验证输出文件，区分 "planned" 和 "verified complete"。

- **论文核心声明缺乏关键对照实验**: "absorption is a sparsity phenomenon" 这一中心声明缺少 L0-matched ablation 这一关键对照。该实验被承诺、被描述、被标记完成，但从未执行。没有它，中心声明只能是一个假设。

- **Negative explained_variance 需要立即调查**: 所有训练变体都显示负 explained_variance（-0.14 到 -0.37）。Orthogonality 的 MSE~3e-5 但 explained_variance=-0.37，这在数学上不一致。可能是计算错误，也可能是 SAE 完全没有学到任何东西。

- **数值声明在不同章节之间漂移**: Cohen's d 在 statistical_analysis.json 中是 4.93，在 full_summary.json 中是 5.51。相关系数在摘要中被更新为 n=7，但讨论中仍引用过时的 pilot 数字。论文中的每个数字都必须可追溯到一个单一的真实来源文件。

## Watch Out

- **1-2 分钟的训练时间可能意味着未收敛**: 2M 样本在 1-2 分钟内完成训练是可疑的。必须添加收敛诊断（损失曲线、最终损失值），并验证训练是否真正完成。

- **Dead latent 危机削弱实际建议**: TopK 有 81.6% 的 dead latents。在不解决这一病理的情况下推荐 TopK 是误导性的。必须分析失败模式，而不仅仅是 headline metrics。

- **基于聚合均值的 correlation 会夸大信号**: L0-absorption correlation 使用 7 个 variant means 而不是 35 个 individual replicates。这是聚合偏差。必须在最细粒度的数据上计算 correlation。

- **Feature count 不匹配是致命的**: 声称 16k 但实际运行 1k 不是小差异---如果被审稿人发现，这是研究欺诈。在写一个字之前验证规模。

- **每次 pivot 都引入新的完整性问题**: 系统没有收敛。分数轨迹：7.0 -> 5.0 -> 5.5 -> 4.5。需要停止添加新实验，先修复基础架构。

- **Matryoshka 的 "antagonistic interaction" 声明基于复制数据**: 4/5 的 replicate 与 MultiScale 完全相同。additive expectation 为 -0.142 在物理上不可能（吸收率不能为负）。对于 bounded metrics，应使用 relative risk 框架，而不是 additive model。

## Keep Doing (success patterns)

- **Component-isolated design**: 一次只改变一个架构组件，这种方法论上是正确的，能够实现因果归因。保持这一设计原则。

- **Ground-truth absorption measurement**: 从已知的 parent-child 关系中直接测量，消除了困扰早期迭代的 probe artifacts。这是一个重大的方法论优势。

- **Random control validation**: Random control (A=0.534 vs trained range 0.055-0.261) 验证了 metric discrimination。继续使用强对照。

- **Honest negative result reporting**: Orthogonality null result (d=0.13) 直接反驳了 OrtSAE 的 65% 声明，且没有 spin。这种语气对于顶级会议来说完全正确。

- **Strong visual narrative**: Figures 2-6 形成了一个连贯的视觉故事。继续投资于清晰、自包含的图表。

- **Clear practical recommendations**: Section 5.4 提供了三个具体的、可操作的建议，直接与证据挂钩。保持这种读者友好的格式。

- **Writing quality**: 散文清晰、直接，基本避免了 banned patterns。8/10 的写作质量是所有迭代中最高的。
