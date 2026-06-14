# Lessons from Iteration 2

**日期**: 2026-04-13 | **分数**: 5.5/10 | **轨迹**: 停滞

---

## Must Improve（必须改进）

- **[PIPELINE — BLOCKING] 写作 Agent 必须检查 action_plan.json 中所有 blocking=true 的问题**：迭代 1 和迭代 2 均将数值不一致标记为 BLOCKING，但写作 Agent 每次都在未验证的情况下推进，导致相同错误循环出现。下一迭代必须在写作 Agent 启动前运行一个前置检查：读取 action_plan.json，验证所有 blocking=true 的问题（当前为 W001、W002、W003、W005）已在 paper.md 中得到修正。如果未解决，拒绝启动写作。

- **[SOUNDNESS — BLOCKING] Proposition 1 贡献顺序必须重排**：P1 的几何预测被 B1 直接证伪（AUROC=0.318，方向相反）。三次迭代均被 supervisor 标记。下一次写作必须把贡献顺序改为：(1) EDA 实证检测器（主要贡献），(2) RD 理论框架（解释吸收何时是损失最优，而非几何预测器），(3) Proposition 2（机制猜想）。Section 3.2 必须加一句承认 P1 几何预测在后收敛状态下反向。

- **[EXPERIMENT — BLOCKING] 标签集必须在写作前统一**：B1_pairwise_eda.json 用 threshold=0.29 得到 n_pos=63，D1/D2 用不同阈值得到 n_pos=50，精确标签 n=18——三套标签集在 Table 1 中未区分。下一迭代计划阶段必须加入"标签集统一审计"任务（约 15 分钟），生成一个从所有标签集到 Table 1 行的映射表，在写作前完成。

- **[SOUNDNESS — BLOCKING] EDA_norm 必须提升为主要指标**：D2 显示 EDA_norm AUROC=0.738，DeLong p=0.0007 显著优于 EDA_base=0.650。继续用 EDA_base 作为摘要主要数字是对自身工作的系统性低报。摘要、简介、Table 1、结论均应使用 EDA_norm=0.738。

- **[WRITING — BLOCKING] 四处数值不一致必须机器验证后再进入写作**：(a) AUROC=0.206（应为 0.350），(b) 吸收率范围（应统一为 0.876-0.978），(c) Spearman ρ 三值（应统一为 ρ=-0.482 from E1），(d) Table 1 标注 n_pos=50（应为 n_pos=63 for B1 rows）——这些错误应通过脚本从 source JSON 直接验证，而不是依赖人工校对。

- **[EXPERIMENT] 候选集声明必须基于 Precision@k 数据**：D1 显示 Precision@50/100/500 = 0。摘要中的"tractable candidate set"声明直接违背原始数据。替换为："EDA provides statistical enrichment at the distributional level (AUROC=0.738, AUPRC=2.09× base rate) but top-k shortlisting requires k > 500 to retrieve any absorbed feature."

---

## Watch Out（注意事项）

- **[ANALYSIS] 编码器范数 DeLong p=0.153 意味着 EDA vs 编码器范数差异不显著**：论文不能声称 EDA 机制上优于编码器范数，除非提供区分证据（如在多义性控制子集上分离两个指标的实验）。如果 EDA_norm 升格为主要指标，需要重新计算 EDA_norm vs 编码器范数的 DeLong 检验。

- **[EXPERIMENT] 跨域语义层次分析在 n=20 下没有统计功效**：animate_inanimate 和 noun_proper 层次均为 n=20 个词，需要 n≈500 才能以 80% 功效检测 5% 吸收率。声明语义吸收"不存在"是过度声明，应改为"在当前样本量下未被检测到"。

- **[EXPERIMENT] B3 跨架构对比有钩子点混淆**：Standard SAE 用 resid_pre，TopK SAE 用 resid_post——不同网络位置。不加注意就声称 TopK 的弱 EDA 信号是稀疏度配方效果的证据，会被审稿人识破。

- **[REPRODUCIBILITY] sae-spelling commit hash 等可复现性信息仍缺失**：这是连续两轮迭代的遗留项。在 Section 3.4 中添加 sae-spelling repo URL、commit hash、FeatureAbsorptionCalculator 参数、SAELens release identifier 和 18 个精确标签的 appendix 表。

- **[ANALYSIS] Hysteresis 结论必须删除"亚稳态"声明**：E2 数据（0.959 → 0.960）在天花板饱和状态下不能区分亚稳态与其他三种同等合理的解释。结论应直接说明无法在测试的 L0 范围内逃离吸收状态。

---

## Keep Doing（保持的良好实践）

- **负面结果报告的金标准**：H1 方向证伪、H3 ASI 证伪（AUROC=0.421 低于零假设均值）、相变假说证伪（LRT p=0.456）均用具体数字和预注册目标对比报告。这在三次迭代评审中都被明确认可为论文最强部分。每次迭代都要保持这个标准。

- **统计工具箱完整且正确**：Bootstrap CIs（10k 重采样）、DeLong test、置换零假设 z 分数、Cohen's d、Mann-Whitney U、Kruskal-Wallis 均正确应用。继续使用，并确保 AUPRC 总是在极端类不平衡场景（n_pos < 50）中报告。

- **GPT-2 Small 开放模型精确标签锚点**：n_pos=18 精确 Chanin 标签是本研究最可复现的核心证据。下一轮应扩展：计算 GPT-2 L10 的精确标签 EDA 作为 Table 1 的第二行。

- **审计系统（79 项声明、0 不一致）**：自动化声明-证据链验证是科学诚信的范本。继续运行并在每次写作前使用审计输出作为事实核查基线。

- **交叉方向指标 cos(ê_p, d_c)**：AUROC=0.730，z=6.38，Cohen's d=0.552，p=2.8e-9 是论文最强实证结果。这是论文的关键贡献之一，应作为主要贡献而非次要发现来呈现。

- **写作无滥用词汇**：无"In recent years..."、"groundbreaking"、"to the best of our knowledge"等套话。Discussion 中 failure mode 分析（Section 5.3）的写作风格是范本：每个负面结果配有具体数字 + 竞争性机制解释。
