# Lessons from Iteration 3

**日期**: 2026-04-14 | **分数**: 5.5/10 | **轨迹**: 停滞（迭代 2 = 迭代 3 = 5.5）

---

## Must Improve（必须改进）

- **[PIPELINE — BLOCKING × 3] Pilot-to-full 升级机制必须硬编码进 task_plan.json**：三次迭代均未能执行全规模实验。迭代 4 task_plan.json 必须包含：(a) 每个 pilot 任务通过后自动生成全规模任务；(b) 写作阶段前置验证步骤，检查所有核心结果 JSON 是否标记 mode='FULL'，否则拒绝进入写作。这不是"建议"，是阻断规则。

- **[EXPERIMENT — BLOCKING] C2 必须在 26 字母 × 1000+ tokens 全规模重跑**：当前 OMP 结果（3 字母 × 30 tokens，97.8% 天花板）在统计上无任何意义。必须报告每字母 95% bootstrap CI on (AR_OMP - AR_FF)，并加入功效分析。在未完成全规模重跑前，论文不能以"证伪"语言描述 OMP 结果。

- **[SOUNDNESS — BLOCKING] TopK-32k AUROC=0.837 必须重新定性**：E1 和 A3 均确认 IG 测量下 TopK-32k 吸收率为 0%。n_pos=77 是解码器对齐代理标签，不是吸收标签。摘要和 Table 1 中的"跨架构复现"表述是事实性错误。必须在论文中明确标注："此 AUROC 衡量 EncNorm 预测解码器余弦对齐方向的能力，不是吸收检测。"

- **[SOUNDNESS — BLOCKING] EncNorm 机理叙事必须降级或删除**：A2 数据显示信号在 5 层中的 3 层（L2、L8、L10）反向。第 3.1 节的梯度竞争推导是后验推测且与多数层的数据矛盾。必须将 EncNorm 描述改为"在 GPT-2 Small L6 层实证发现的启发式指标，机理基础尚不清楚"，而非已建立的理论。

- **[WRITING — BLOCKING × 3] 6 张图必须在写作阶段前生成**：连续三轮迭代进入写作时 figures/ 目录中无当前迭代的图。下一轮计划中必须包含图表生成作为独立任务，与全规模实验并行执行，在写作 Agent 启动前完成。

- **[PLANNING — BLOCKING] 标签集必须分为发现集和验证集**：同一 n_pos=18 标签集跨三轮用于发现和验证是循环评估。迭代 4 必须在 b/c/d/f/g 字母上生成新 held-out 标签。AUROC 最终报告值必须使用 held-out 集，不能再用 iter_001 R4 标签同时作为发现和验证集。

- **[EXPERIMENT] 模型多样性问题必须在规划阶段前解决**：Gemma Scope 已连续两轮遭遇 401。HuggingFace gated license 接受是一次性人工操作，必须在制定 iter_004 计划之前完成，而不是作为实验任务之一规划。备选：Pythia SAEs（EleutherAI 完全公开访问）。

---

## Watch Out（注意事项）

- **[ANALYSIS] EncNorm 独立于 EDA 的贡献尚未建立**：Spearman r=0.712，共享 51% 方差。AUROC 提升可能来自分布形状而非独立信息。迭代 4 必须运行：EDA+EncNorm 联合 LR vs EDA 单独 LR 的 DeLong 测试。如果不显著，EncNorm 的边际贡献无法声明。

- **[ANALYSIS] O_jaccard 覆盖范围必须分层报告**：18 个被吸收特征中有 9 个因不在字母集中而 O_jaccard=0（构造性约束）。必须分别报告 9 个有覆盖特征的 AUROC 和 9 个无覆盖特征的 AUROC。不要将两组混合的 AUROC=0.721 作为 O_jaccard 检测能力的唯一指标。

- **[ANALYSIS] F1 宽度恢复 67% 必须有 hook-confound 基线**：Standard-24k（resid_pre）与 TopK-32k（resid_post）在不同激活空间。必须计算 L6 resid_pre 与 resid_post 之间的余弦相似度基线，才能解读 67% 的实质意义。两个重复匹配索引（16435 被 absorbed_id=2406 和 11270 共享；29309 被 24154 和 7371 共享）需要调查是否为假匹配。

- **[WRITING] 确定性语言与统计功效必须匹配**："Decisively falsifies"、"unambiguous"等表述不能用于零功效实验的零结果。替换为："provides evidence against"、"is consistent with falsifying"。

- **[IDEATION] 弱/强摊销差距假说的区分必须明确**：OMP 实验测试的是"弱版本"（推理时固定解码器）。Costa et al. MP-SAE 测试的是"强版本"（训练时联合优化迭代编码器）。Section 5.1 的实践建议必须区分这两个版本，不能混同。

---

## Keep Doing（保持的良好实践）

- **负面结果报告金标准**：H2 零结果（OMP）、D2 probe 失败、E1 Gemma 401、F1 hook-confound 全部以具体数字和根因分析主动披露。这是论文在三轮评审中始终获得认可的最强部分。每次迭代必须维持这个标准。

- **预注册判定准则**：Section 3.3 在报告结果前明确声明判定标准（"在实验运行前预先确立"）。这是增加 OMP 结果可信度的关键方法论强项，务必保留并在 iter_004 扩展到其他假设。

- **统计工具箱完整且正确**：Bootstrap CI（10k 重采样）、DeLong 测试、Cohen's d、Mann-Whitney U 在整个 iter_003 实验中均正确应用。AUPRC 在极端类不平衡场景中的使用规范。继续保持。

- **数值声明与 source JSON 一致性**：Paper.md 中 12 项抽查数值（AUROC CI、DeLong z、Cohen's d、F1 恢复计数等）全部与 source JSON 一致。这是数据诚信的范本，维持 source-first 的写作规范。

- **双 GPU 并行调度**：10 个任务约 2 小时完成，无失败。调度效率良好，保持并在全规模实验时同样采用。

- **GPT-2 Small 金标签锚点**：n_pos=18 Chanin IG 精确标签是论文最可复现的核心证据。AUROC=0.757 在此锚点上是可信的。未来实验继续以此作为方法有效性的基准线，同时扩展到 held-out 集以建立独立验证。
