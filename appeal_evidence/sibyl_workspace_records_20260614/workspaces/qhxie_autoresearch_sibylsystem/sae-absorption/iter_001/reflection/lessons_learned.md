# Lessons from Iteration 1

**Date**: 2026-04-13 | **Score**: 5.5/10 | **Trajectory**: Stagnant

---

## Must Improve（必须改进）

- **[WRITING — BLOCKING] EDA 排序事实错误必须在下一次写作开始前强制更正**：Section 6.2 的"EDA 排序 late > partial > early 成立"与 phase2a_taxonomy.json 中的 `eda_ordering_holds=false` 直接矛盾（两个配置均为 false；partial 中位数 EDA 0.652 < early 中位数 EDA 0.668 at L12-65k）。这是迭代 0 的 BLOCKING 问题，迭代 1 仍然存在于论文中。下一迭代开始写作前必须验证：Section 6.2 文本与 phase2a_taxonomy.json 中实际数值一致。同时更正 Table 3 中 early absorption 的 EDA signal 描述（实际数据显示 early-type EDA 高于 partial-type，而非"Low"）。

- **[WRITING — BLOCKING] 摘要 DeLong 对比数字必须更正**：摘要当前写"+0.396 AUROC"紧跟"AUROC = 0.776 on Gemma Scope L12-16k"，实际 +0.396 来自 L5-16k，L12-16k 的差距是 +0.553。这是新增的事实性错误，需在下一轮提交前优先修正。

- **[NOVELTY] 定理必须正式证明或降格为 Proposition**：Theorem 1 的证明草案已连续被迭代 0 和迭代 1 的评审标记为循环论证。在下一次评审中它同样会被发现。解决方案只有两个：(a) 在附录 B 中提供从 SDL 损失平稳性条件出发的逐步代数推导；(b) 降格为"Proposition 1（直觉动机）"并明确写出这只是启发式论点。任何第三条路都是将问题推迟。

- **[NOVELTY] EDA = SAEBench 指标必须在 Introduction 和 Section 3.1 中明确披露**：Introduction 仍声称"We introduce EDA"而未提及 SAEBench 的 encoder_decoder_cosine_sim（Pearson r > 0.999）。这已在迭代 0 标记、迭代 1 反思中再次确认。这是任何熟悉 SAEBench 的审稿人会立即抓住的问题。必须在 Section 3.1 中加入一段明确声明该指标的已有来源，并重新定位 EDA 的贡献为"理论基础 + 系统性实证验证"。

- **[PIPELINE — BLOCKING] 阻断性问题必须作为前置门槛强制执行**：迭代 0 的行动计划将 EDA 排序更正标记为"BLOCKING: 在任何进一步写作之前更正"。写作阶段仍然在没有应用此更正的情况下推进。下一迭代需要在流水线中加入一个可机器验证的前置检查：在启动写作 agent 之前，验证 action_plan.json 中所有 blocking=true 的问题已经解决。如果未解决，流水线不得启动写作。

- **[EXPERIMENT] D-EDA/EDA 等价张力必须在下一轮解决**：R4 直接验证显示 EDA = 1 - dec_cos 是精确恒等式（delta AUROC = 0.0）。但 R3 的 D-EDA 在 GPT2-L10 上 AUROC = 0.762 vs EDA 0.336——需要 D-EDA 是不同的计算。论文没有协调这一矛盾。在下一轮：审查 R3 D-EDA 的代码实现，确认它是否是真正的 LASSO 分解或仅仅是负的 dec_cosine。根据结论更新 Table 1 和 Section 3.3。

---

## Watch Out（注意事项）

- **[EXPERIMENT] 分类体系的"Early 主导"仍然高度依赖阈值且样本量不足**：tau=0.2 时 early 仅为 32%（L12-65k），tau=0.3 时为 72%，tau=0.4 时为 95%。L12-16k（n=16）的 KW p=0.237 不显著。在扩展到 4 个以上 SAE 配置并在多个 tau 值下保持稳健之前，不得使用"重构了领域认知"这样的强声明词汇。替换为"初步证据表明（preliminary evidence suggests）"。

- **[EXPERIMENT] Gemma 2B 访问受限是两轮迭代都未解决的结构性阻断**：R4 访问检查确认无法访问。所有 Gemma Scope AUROC 值都在代理标签上。GPT-2 Small 已经证明是最干净的验证锚点（精确标签、开放访问），下一轮应将 GPT-2 扩展到更多配置（L10 的 taxonomy，更多 layer 的 EDA）。同时继续申请 Gemma 2B 访问权限（研究访问申请通常需要 1-3 天审核）。

- **[SYSTEM] Section 7.1 的"56%/20x"数字未经计算验证**：当前文本中这两个数字无法从源数据推导（19% ≠ 56%）。这类未验证的量化声明会直接导致审稿人对整篇论文的可信度产生质疑。每一个在论文中出现的量化声明都应在提交前追溯到来源数据并验证计算。

- **[REPRODUCIBILITY] Figure 1 缺失、附录 B/C 缺失是提交级阻断**：Figure 1 只有文字描述文件而没有 PDF。两个附录在正文中被引用但不存在于 paper.md 中。这些问题必须在论文提交前解决——没有方法图和引用的附录是审稿人立即驳回的理由。

- **[ANALYSIS] 极小正例子集的 Bootstrap CI 会产生误导性精度**：n_pos=3 的 polysemantic AUROC CI [0.842, 0.979] 是数值伪像，不是真实置信区间。凡是 n_pos < 10 的子群体分析，必须在文中明确注明 CI 不可靠。

---

## Keep Doing（保持的良好实践）

- **负面结果报告的金标准**：Section 7.3 的三个证伪假设（H3、H5、H6）的报告方式是模范——具体数字、预注册目标对比、机制解释。这在这轮评审中被明确认可为论文最强的部分。每一次迭代都要保持这个标准。

- **R4 自适应补充实验轮**：R4 的 GPT-2 精确标签验证和 shuffled control 为 H3 证伪提供了正确的方法基础。这种在写作审查后发现证据缺口并快速执行补充实验的模式应标准化为每次迭代的必要步骤（"pre-submission experiment gap audit"）。

- **GPT-2 作为开放模型验证锚点**：GPT-2 Small L6 的精确标签结果（AUROC=0.629，n_pos=67）是本论文中最可复现、最可信的证据。下一轮应扩展此策略：GPT-2 L10 的 taxonomy 分析（n_pos=33，exact labels），GPT-2 更多层的 EDA scaling 分析。

- **统计工具箱完整且正确**：Bootstrap CIs（10k 重采样）、DeLong test、Cohen's d、Mann-Whitney U、Kruskal-Wallis、Table 1 中的 shuffled null 列——这套工具箱合适。继续使用，并增加 AUPRC 用于极端类不平衡场景。

- **结果辩论机制有效过滤过度乐观偏差**：多角度辩论（乐观/怀疑/策略/综合）在两轮迭代中均有效产生了有针对性的修正建议，并防止了整体放弃当前研究方向的倾向。继续使用。

- **写作规范无滥用词汇**：无"In recent years..."、"to the best of our knowledge..."、"groundbreaking"等套话。Section 4.3 的 pilot-to-full 坍塌诚信报告是科学写作的范本。继续保持。
