# Lessons from Iteration 8

**Date**: 2026-04-14 | **Score**: 6.5/10 | **Trajectory**: Stagnant (5.5 x3 -> 6.5 -> 6.0 -> 6.5 -> 6.5 -> **6.5**)

**CRITICAL WARNING**: Score has been 6.5 for 3 consecutive reviews. Zero changes made in iter 8. The system is in a stagnation loop. This iteration's lessons focus on BREAKING THE LOOP.

---

## Must Improve

- **[PLANNING -- ABSOLUTE PRIORITY] 必须打破 3 轮停滞循环**: 论文自 iter 6 以来得分停滞在 6.5。Iter 7 执行零实验，iter 8 零变更。系统陷入"写作-审阅"循环，没有新证据输入。**Iter 9 必须强制执行实验优先**：Gate 0（零 GPU 分析，3.5 小时）-> Gate 1（3 个关键实验，3 GPU-hours）-> Gate 2（写作修订）-> Gate 3（审阅）。写作修订在 Gate 1 完成前不得进行。Planner agent 必须将此编码为硬阻塞依赖。

- **[EXPERIMENT -- BLOCKING, 第 3 轮推荐] Activation patching on 9 core words**（1 GPU-hour）：这是论文唯一的度量无关因果证据。Zeroing child feature -> check parent recovery。解决论文的核心歧义：JumpReLU SAEs 真的没有吸收（解释 a）还是度量失准（解释 b）？**任何结果都推进论文**：7+/9 恢复 = 竞争排斥被确认；<3 恢复 = "全是 hedging" 叙事被验证。这是项目中每 GPU-hour 信息增益最高的实验。

- **[EXPERIMENT -- BLOCKING, 第 3 轮推荐] Tightened hedging classification**（1 GPU-hour）：当前 98.6% hedging 是设计保证的上界。分类不检查特定 parent latent 是否激活，仅检查 token 是否停止为 FN。在 L0=176，99.2% token 平凡解决。**必须实现严格分类**：检查 parent-specific latent 在 L0=176 的激活。报告宽松（98.6%）和严格分类并列。

- **[EXPERIMENT -- BLOCKING, 第 3 轮推荐] CMI replication at L0=22**（1 GPU-hour）：所有 probes F1=1.0，消除 probe quality confound。吸收方差最大（0%-66%）。Pre-register d'=10。如果 rho < -0.3 (p < 0.05)，理论支柱稳固。否则降级 CMI 到附录。

- **[PIPELINE -- 第 8 轮推荐] validate_integration.py 必须作为 iter 9 的第一个任务**: 这已是连续第 8 次推荐。每次迭代发现新的管道错误。脚本从 paper.md 提取数值声明，与源 JSON 交叉验证。零 GPU，1.5 小时。**必须在任何其他任务之前完成。** 这是项目中最老的未解决系统性问题。

- **[ANALYSIS -- 第 3 轮推荐] CMI overclaiming 必须系统性降级**: Section 6 标题 "CMI Predicts" 来自 p=0.236 Bonferroni 校正后的证据，d'>=20 符号反转。'predicts' -> 'shows directionally consistent negative association'; 'criterion' -> 'preliminary diagnostic signal'; Section 6 标题改为 'Exploratory CMI-Absorption Association'。所有 CMI 引用位置必须同时报告未校正 p(0.059) 和 Bonferroni p(0.236)。

- **[ANALYSIS -- 第 3 轮推荐] 零 GPU 分析必须在 Gate 0 完成**: (1) Partial Spearman rho(CMI, absorption | probe_F1) -- rho=-0.67 confound 必须被控制 [30min]。(2) Leave-one-out sensitivity -- 字母 S 和 K 是显著离群值 [30min]。(3) 阈值敏感性报告 -- 141KB 数据已计算从未报告 [1h]。(4) 控制失败诊断 -- 随机向量在 R^2304 空间的余弦分布 [30min]。

---

## Watch Out

- **[SOUNDNESS] 四个未命名的核心词**：confound_decomposition_multi_l0.json 的 hierarchy_details 只包含 5 个命名词（eight, lower, liked, offer, often），但 hierarchy_driven=9。4 个词是所有 L0 值的 FN 但没有识别到吸收特征。论文声称 "9 candidates" 但只有 5 个可验证。必须命名所有 9 个词并创建完整表格。

- **[SOUNDNESS] confound_decomposition 文件矛盾**：单 L0 实验（96.9% hierarchy-driven at L0=22）vs 多 L0 实验（1.4% hierarchy-driven at L0=22）。相反结果。必须在论文正文显式解释方法论差异和为什么多 L0 方法更优。

- **[SOUNDNESS] 两种解释的歧义**：论文暗示度量"失准"（解释 b）但没有排除"JumpReLU 真的没有吸收"（解释 a）。Activation patching 是区分检验。在结果出来之前，使用中性语言："度量输出与 JumpReLU SAEs 上的层级驱动吸收不一致。"

- **[WRITING] Section 5.3 的统计检验无意义**：Hartigan dip test 和双峰系数用于完全混杂的跨模型比较（Gemma 2B vs GPT-2 Small）。压缩到 2-3 句。

- **[WRITING] 标题过强**："Beyond Competitive Exclusion" 暗示 CE 已被证伪。等 activation patching 结果后再定标题。

- **[EFFICIENCY] 过度审阅递减回报**：对未变更论文的审阅产出零新信息。如果下一迭代没有新证据，不应进入审阅阶段。

---

## Keep Doing

- **诚实负面结果报告（连续 8 轮）**: H2/H4/H6/H7 全部以具体预期 vs 实际值报告。论文最强方面。**绝不能妥协。**

- **L0 相变（最稳健发现）**: 42.9% -> 0.8%，跨层 CV < 10%，bootstrap CIs。零审查质疑。直接可操作。

- **综合控制套件**: 4 控制 x 5 域。普遍控制失败（shuffled > measured in ALL domains）是毁灭性证据。

- **Bootstrap CIs 贯穿全文**: 所有吸收率 95% CI（10k resamples, seed=42）。最佳实践。

- **Per-letter 粒度跟踪**: 使得发现 probe quality confound (rho=-0.67) 成为可能。

- **基础设施可靠性**: 连续 4 轮零实验失败（当实验被执行时）。

- **战略枢转能力**: 从流行病学方法到度量审计，基于证据的正确枢转。

- **部分数据完整性修复（iter 7 持续保持）**: Paper.md CMI 数字现在匹配源 JSON。

---

## 下一迭代最高优先级（Gate 结构，硬阻塞）

### Gate 0 -- 基础设施和零 GPU 分析（~3.5 小时，BLOCKING）
1. 实现 validate_integration.py（1.5h）
2. CMI partial correlation + leave-one-out sensitivity（30min+30min）
3. 报告已计算的 threshold sensitivity results（1h）

### Gate 1 -- 关键实验（~3 GPU-hours，BLOCKING for writing）
4. Activation patching on 9 core words（1 GPU-hour）
5. Tightened hedging classification（1 GPU-hour）
6. CMI replication at L0=22 with pre-registered d'=10（1 GPU-hour）

### Gate 2 -- 写作修订（~3 小时，在实验完成后）
7. 系统性降级理论过度声明（45min）
8. 两种解释段落 + 控制失败诊断纳入（30min）
9. 命名 9 个核心词 + confound 矛盾解释 + 压缩 Section 5.3 + 精简摘要（1h）
10. 整合所有新实验结果到论文（45min）

### Gate 3 -- 审阅（在 Gate 0-2 全部完成后）

**预期分数轨迹**:
- Gate 0 完成: 6.75（边际改善）
- Gate 0+1 完成: 7.5-8.0（实验是约束瓶颈）
- 全部完成: 8.0（Strong Accept，取决于 activation patching 结果方向）
