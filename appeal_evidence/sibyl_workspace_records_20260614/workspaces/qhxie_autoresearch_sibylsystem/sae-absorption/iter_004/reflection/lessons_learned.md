# Lessons from Iteration 4

**日期**: 2026-04-14 | **分数**: 6.5/10 | **轨迹**: 改善（迭代 1-3 = 5.5 -> 迭代 4 = 6.5, +1.0）

---

## Must Improve（必须改进）

- **[ANALYSIS -- BLOCKING] H3 宽度/L0 混淆必须在下一轮实验前解决**：所有 5 个高吸收 SAE 为 1M 宽度（L0 16-58），所有 5 个低吸收 SAE 为 16k/65k（L0 137-297）。偏相关未包含 L0 作为协变量。SCR 偏相关从 r=-0.431 跳至 partial r=-0.677 是抑制变量效应。必须执行三项零 GPU 分析：(1) L0 作为协变量，(2) 宽度层内分层相关（16k/65k/1M），(3) SCR 抑制变量顺序诊断。如果 H3 在控制后消失，论文需要根本性重构。

- **[WRITING -- BLOCKING] 92.3% 分类率必须标记为"上界"**：论文自身的 suspicious_flags 承认 Type II 率"CRITICAL[ly] likely inflated"。n_comparison_tokens=0 使得 Type II 识别基于全局回退基线而非字母特定比较。摘要必须将 92.3% 标记为 "(upper bound)"，以验证率 Type I=3.8%、Chanin=80.8% 作为主要报告数字。

- **[WRITING -- BLOCKING] "Validated quality indicator" 必须降级为 "quality-correlated metric"**：相关证据不能"验证"指标。"Validated" 暗示干预证据（降低 absorption 后质量改善），这在论文中不存在。

- **[WRITING -- BLOCKING] Figure 1 (LV 框架概念图) 必须生成**：fig_lv_framework.pdf 不存在，但引言和 Section 3 引用"Figure 1"。这阻塞论文编译和外部审查。

- **[PLANNING] 混淆控制分析必须在实验计划阶段预先指定**：H3 宽度混淆、H2 聚类标准误、H4 显著性检验、H1 预过滤/系数区分——全部在事后审查中发现。任何相关分析必须在计划中列出所有需要的分层（宽度层内、协变量控制、聚类结构）。

---

## Watch Out（注意事项）

- **[EXPERIMENT] 安全探针（Table 6）不应出现在主结果中**：n=3 SAE，3+ 混淆因子（层/宽度/hook），非单调模式（最高吸收 = 最小探针差距 0.051）。将其移至附录。SAEBench 相关结果独立成立。

- **[EXPERIMENT] 跨模型缺口必须在摘要/引言中前置**：H1/H2/H4 在 GPT-2 Small，H3 在 Gemma Scope。不要在 Section 7.5 中才提及。读者必须在看到结果前了解两模型设计。

- **[ANALYSIS] PMI 回归需要字母级聚类标准误**：806 观测 = 31 配置 x 26 字母。L0 的 p=0.012 在字母聚类下可能不显著。Durbin-Watson=1.335 和 skewness=5.186 进一步质疑 OLS 适用性。

- **[ANALYSIS] H1 LV 检测器的 ROC-AUC<0.5 比 F1=0.128 更有信息量**：alpha_ij 反预测吸收。第一区间 [0, 0.1] 吸收率 0.848。机制原因：被吸收父特征频率被抑制导致 alpha_ij 反常地低。这是比"检测失败"更深刻的发现，应突出报告。

- **[ANALYSIS] H4 DAS "部分支持" 需要正式显著性检验**：42.3% 正斜率率 < 80% 目标，每字母方差巨大。无 bootstrap CIs、无配对 Wilcoxon 检验。DAS 实际样本量 40/字母，非方法论声称的 10,000 tokens。

- **[ANALYSIS] Type III 计数为零部分是评估顺序伪影**：letter B 的 DAS(k=3)=0.695 > 0.6 阈值，但因顺序评估先被分类为 Type II。应在 Section 5.3 说明。

- **[WRITING] 标题 "When Features Compete" 暗示竞争框架有效**：论文主要发现是竞争检测失败。考虑修改标题反映实际贡献。

- **[PLANNING] 时间估计偏保守约 5-7x**：C2B 计划 480 min vs 实际 78 min, C3B 计划 120 min vs 实际 2 min。RTX PRO 6000 Blackwell 的实际性能远超保守估计。下轮规划应使用更紧凑的时间预算。

---

## Keep Doing（保持的良好实践）

- **研究转型的决断力**：放弃三次迭代的 OMP/EncNorm 积累，转向全新 LV 框架。这是分数从 5.5 跃升至 6.5 的根本原因。当渐进修补无法解决结构性问题时，果断转型。

- **预注册判定准则**：H1-H4 各有明确的量化成功/失败阈值（F1>0.35, partial R^2>0.10, |r|<0.2, 80% monotonicity），在报告结果前公开声明。这使负面结果具有发表价值。必须在所有未来迭代中维持。

- **全规模实验执行**：连续三次迭代的首要阻断问题（pilot-to-full）在本迭代得到根本解决。所有 13 项任务标记 mode='FULL'。这证明完整的研究转型比修补旧流水线更有效。

- **诚实的负面结果报告**：H1（F1=0.128，低于余弦基线）、H2（partial R^2=0.0006）、H4（42.3% vs 80%）均透明报告。这建立了 H3 正面发现的可信度。

- **公共数据的战略利用**：SAEBench 预计算数据使得 54-SAE 相关分析无需自行训练。识别已有公共数据集是效率优化的典范。

- **统计工具箱正确应用**：Bonferroni 校正、偏相关、Cohen's d、匹配对分析——所有统计方法均正确应用，无例外。

- **声明-数据一致性**：论文数值与 source JSON 交叉验证一致。这是数据诚信的范本。

- **GPU 执行零失败**：13 项任务全部成功，双卡并行调度成熟稳定。

- **证据质量自评**：论文自身生成 suspicious_flags 标记已知问题（Type II 膨胀）。虽然摘要淡化了这些警告，但分析过程的诚实性是可信的。
