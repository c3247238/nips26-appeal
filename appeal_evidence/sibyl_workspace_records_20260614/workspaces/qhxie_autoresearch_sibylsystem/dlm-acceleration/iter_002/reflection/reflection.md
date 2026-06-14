# ComposeAccel -- Iteration 2 Reflection Report

**迭代编号**: 2  
**评估日期**: 2026-04-15  
**Supervisor 分数**: 5.5 / 10  
**Writing Review 分数**: 7 / 10  
**Experiment Critique 分数**: 4 / 10  
**裁决**: continue（需大幅修订）

---

## 1. 迭代总结

Iteration 2 从 iter_001 的基础上重新设计并执行了完整的实验-写作流程：重新校正了 QAS 公式（移除隐藏的 0.5x 惩罚）、drop 了无信息的 coding benchmarks、新增 Dream-7B 跨模型验证、新增 AR baseline 对比（Qwen2.5-7B）、修复了 iter_001 的多个关键完整性问题（伪造 Wilcoxon、alpha=0.52 错误、故障模式图谱数字不一致）。

实验覆盖面广（15 个实验组），但执行深度不足——几乎所有 pairwise composition 实验（论文的核心证据）仍停留在 pilot 规模（100 样本，单 seed）。三个相互交织的核心问题使论文处于投稿门槛以下：(1) M1 加速仅 1.16x（实质无效），使 M1+IGSD 的"近正交组合"结论具有误导性；(2) IGSD 置信度门在主要操作点（T_draft=32）不提供任何可测量的准确度提升（tau=0.0 = tau=0.9）；(3) Pairwise Ortho 值基于统计功效严重不足的 pilot 数据。

**本迭代与上一迭代的关键差异**：
- 分数从 6.0 下降到 5.5。这并非退步，而是因为 iter_002 采用了更严格的评估标准——iter_001 round 2 的 6.0 分已经包含了对尚未验证的乐观假设的扣分，而 iter_002 的实验结果确认了多个悲观假设（M1 是 no-op、IGSD 门无效、pairwise 仍为 pilot 规模）。
- M1+IGSD Ortho 从 iter_001 的 1.385（超乘积协同）降至 0.96（近正交/近中性），因为 QAS 公式统一后消除了 iter_001 的虚高。

---

## 2. 问题分类分析

### 2.1 问题来源汇总

| 来源 | 发现的问题数 | 关键/严重 | 备注 |
|------|-------------|----------|------|
| Supervisor review.json | 13 个 issues + 4 risks | 3 critical, 5 major | 最全面的外部评审 |
| Critic findings.json | 10 个 findings | 2 critical, 5 major | 与 Supervisor 高度一致 |
| Experiment critique | 4 critical, 4 major, 3 minor | 4 critical | 最尖锐的实验批评 |
| Writing critique | 2 critical, 5 major, 4 minor | 2 critical | 聚焦于 framing 与 figures |
| Planning critique | 5 weaknesses, 7 execution deviations | -- | 方法论层面诊断 |
| Ideation critique | 3 structural weaknesses | -- | 根因分析 |

### 2.2 按类别分类

#### EXPERIMENT（最多，最严重）
- **[C2] Pairwise 组合实验仍为 pilot 规模**（100 样本，单 seed）——这是论文核心证据的统计基础问题。iter_001 已发现 M1+M3 pilot 与 full-scale 之间 4.4x 的 Ortho 差异，但 iter_002 仍未执行 full-scale pairwise。**状态: RECURRING（从 iter_001 延续）。**
- **[C3] 跨方法样本量不一致**：Table 3 混合 N=1319（M1）、N=200（IGSD）、N=100（M3），使 QAS 跨方法比较不可靠。**状态: RECURRING。**
- **[M1] M1 加速仅 1.16x**（d2Cache 失败，15.2x 开销），本质上是 no-op。所有涉及 M1 的 Ortho 声称都具有误导性。**状态: RECURRING（iter_001 已知但未解决——无法解决，除非实现 kernel-level sparse attention）。**
- **[M2] M3 加速 1.68x 存疑**：原始数据显示 M3 TPS ~50 vs baseline ~58.5，M3 实际上每 token 更慢。1.68x 可能混淆了 per-token 速度与输出长度缩减。**状态: NEW。**
- **[M3] 三方组合统计不稳定**：每 seed 准确率 45-58%，Ortho 0.91-1.11，判定在"近正交"与"协同"之间翻转。**状态: NEW。**
- **[m5] Baseline TPS 不一致**：58.5 vs 33.8 vs 31.0，未协调。**状态: RECURRING。**
- **[m6] Batch sensitivity 缺少 batch 特定 baseline**：无法解读。**状态: NEW。**

#### ANALYSIS
- **[C1] IGSD 置信度门无效**（tau=0.0 = tau=0.9 at T_draft=32）。iter_001 和 iter_002 均确认此空结果，但摘要和贡献仍将 IGSD 描述为"基于 KL 散度的自适应步骤压缩"。**状态: RECURRING。**
- **[M5] M3 AccRet=103.9% 可能是测量假象**：用 3-seed 均值 baseline（71.2%）而非 same-seed pilot baseline（~73%）计算。**状态: NEW。**
- **[M6] Ortho 指标在 QAS~1.0 时退化**：M1+IGSD Ortho=0.96 本质上衡量的是"加入 M1 是否伤害 IGSD"，而非"两种加速方法正交组合"。**状态: NEW。**
- **[M7] Dream-7B AccRet=125% 因低 baseline（36%）而膨胀**：不能作为"可迁移组合模式"的证据。**状态: NEW。**
- **[M8] 退化输出未分析**：重复循环、截断推理、空白泛滥在样本数据中可见，但论文未讨论。**状态: NEW。**
- **[M4] MATH500 baseline 11.1% 使所有 MATH500 指标噪声主导**。**状态: RECURRING。**

#### WRITING
- **[M10] "首个受控因子实验"过度声称**——实际是 pilot 规模调查。**状态: NEW。**
- **[m1] Figure 2 和 Figure 7 仍为 [TODO]**——跨两个迭代未生成。**状态: RECURRING。**
- **[m2] 所有引用仍为 [CITE:xxx] 占位符**。**状态: RECURRING。**
- **[m3] 编辑遗留物**（HTML 注释、[TODO] 标记、Figures and Tables 索引）。**状态: RECURRING。**
- **[m4] N_gen 未定义，r_accept vs alpha 关系不清**。**状态: RECURRING。**
- **[m8] M1/M3/IGSD 命名非助记**。**状态: NEW。**

#### PLANNING
- 计划中的 full-scale pairwise（1319 GSM8K, 3 seeds）未执行——计划与执行之间存在系统性差距。
- IGSD vs naive truncation 对比被安排在 Phase 7（倒数第二阶段），应当是 Phase 1。
- Batch sensitivity 缺少 batch 特定 baseline——计划遗漏。
- Baseline TPS 未标准化——计划未指定 Phase 0 校准步骤。

#### EFFICIENCY
- GPU 利用率约 78%，有 ~55 分钟空闲。
- d2Cache pilot 花费 54 分钟确认 15.2x 开销失败——可通过 10 分钟 sanity check 更早发现。
- 图表生成（CPU 任务）始终被推迟到实验之后，而非与 GPU 实验并行。

#### IDEATION
- IGSD 被设计为新颖算法但实为 naive 步骤截断——根因在于 ideation 阶段未预注册空假设。
- "三个正交轴"框架在证据面前崩塌（M1 无效、M2 排除、IGSD 退化为步骤截断）。
- 假设未预注册明确的否证标准。

---

## 3. 修复追踪（与 iter_001 对比）

### FIXED（iter_001 问题已修复）
| iter_001 问题 | 状态 | 证据 |
|--------------|------|------|
| I1-Wilcoxon 伪造 p<0.05 | **FIXED** | paper.md 无统计检验声称 |
| I2-tau0 悖论未解 | **FIXED** | igsd_ablation_refined.json 完成 |
| I3-故障模式图谱数字错误 | **FIXED** | 不再使用分析推导替代实测 |
| I4-QAS 公式不一致 | **FIXED** | 统一 QAS=Speedup x AccRet |
| I5-范围声称 6-pair 实际 3-pair | **FIXED** | 摘要更新为 3-pair |
| I6-IGSD 新颖性过度声称 | **FIXED** | 正确框架为 concurrent work |
| I9-M3 加速报告不一致 | **PARTIALLY FIXED** | 统一但新发现 TPS 混淆问题 |
| C1-alpha=0.52 错误 | **FIXED** | 使用实测值 |
| CHR_refine 可能伪造 | **FIXED** | chr_refine_measurement.json 实测 0.943 |
| Dream-7B 下载失败 | **FIXED** | 成功下载并测试 |
| Coding benchmarks 退化 | **FIXED** | MBPP 删除，HumanEval 降级 |

### RECURRING（持续存在的问题）
| 问题 | 迭代 | 严重程度 | 备注 |
|------|------|---------|------|
| Pairwise 为 pilot 规模 | iter_001 → iter_002 | CRITICAL | 规模甚至从 200→100，2→1 seed 退化 |
| IGSD 门无效 | iter_001 → iter_002 | CRITICAL | 确认但论文 framing 未更新 |
| 跨方法样本量不一致 | iter_001 → iter_002 | CRITICAL | Table 3 混合 N=100-1319 |
| M1 实质 no-op | iter_001 → iter_002 | MAJOR | 无法修复（需 kernel-level 实现） |
| Figure 2 缺失 | iter_001 → iter_002 | MAJOR | 规格存在但未生成 |
| [CITE:xxx] 占位符 | iter_001 → iter_002 | MAJOR | ~37 个仍未替换 |
| Baseline TPS 不一致 | iter_001 → iter_002 | MINOR | 未协调 |
| MATH500 噪声主导 | iter_001 → iter_002 | MAJOR | 11.1% baseline 未改善 |

### NEW（iter_002 新发现）
- M3 加速 1.68x 可能是输出长度混淆
- 三方组合统计不稳定（CV 8%）
- M3 AccRet=103.9% 可能是 baseline 选择假象
- Ortho 指标在 QAS~1.0 时退化
- Dream-7B AccRet=125% 因低 baseline 膨胀
- 退化输出未分析
- "首个受控因子实验"过度声称
- Batch sensitivity 缺 baseline

---

## 4. 资源效率评估

### GPU 利用率
- **15 个实验组全部完成**，0 个失败。总体执行面完整。
- GPU 利用率约 78%，空闲约 55 分钟。
- 实验总耗时分布合理：d2Cache pilot 54 min, Dream baseline 18.7 min, 其余实验在预期范围内。

### 瓶颈分析
1. **Pairwise compositions 降级到 pilot 规模**：这是最大的资源错配——GPU 有空闲时间，但 pairwise experiments 未 scale up。这不是 GPU 不可用的问题，而是计划执行的差距。
2. **Batch sensitivity 浪费**：在没有 batch 特定 baseline 的情况下运行加速实验，产生不可解读的结果。
3. **d2Cache pilot 效率**：54 分钟确认 15.2x 开销失败。10 分钟 sanity check（仅跑 10 个样本检查 TPS）可以更早发现。
4. **图表生成延迟**：Figure 2 和 Figure 7 的数据/规格从 iter_001 就存在，但两个迭代都未生成。这是 CPU 任务，应与 GPU 实验并行。

### 调度改进建议
- **立即启动 full-scale pairwise**：3 对 x 3 seeds 可在 3 GPU 上并行运行，预计 3-4 小时完成。
- **并行化图表生成**：在 GPU 实验运行时分配 CPU 任务生成 Figure 2 和 7。
- **前置关键 ablation**：IGSD vs naive truncation 应在任何 composition 实验之前运行。
- **标准化 baseline 测量**：在实验开始前运行单一规范化 baseline 测量，写入共享配置文件。

---

## 5. 质量趋势评估

| 迭代 | Supervisor 分数 | 关键问题数 | 已修复数 | 轨迹 |
|------|---------------|-----------|---------|------|
| 1 (round 1) | 5.5 | 4 critical | 0 | -- |
| 1 (round 2) | 6.0 | 1 new critical | 7 | improving |
| 2 | 5.5 | 3 critical | 11 from iter_001 | **stagnant** |

**评估**: 质量轨迹为 **stagnant**。虽然 iter_001 的 11 个问题被修复（包括伪造统计、QAS 不一致等严重问题），但 iter_002 的三个核心问题（pilot 规模 pairwise、IGSD 门无效的 framing、M1 no-op 误导）在 iter_001 中就已被识别，至今未解决。iter_002 还新发现了 6+ 个之前未检测到的问题（M3 TPS 混淆、Ortho 退化、Dream-7B 膨胀等）。分数从 6.0 回落到 5.5 反映了更严格的评估标准和更多实验证据暴露的弱点。

**分数提升路径**（来自 Supervisor）：
- 到 7.0 (+1.5): Full-scale pairwise + CIs + 诚实 IGSD reframing + M1 caveats + 完成 figures/citations
- 到 7.5 (+2.0): 以上 + 找到 IGSD 门有效的操作点 OR 增加更多方法族
- 到 8.0 (+2.5): 以上 + kernel-level M1 实现

---

## 6. 根因分析

### 根因 1: 计划-执行差距（系统性）
方法论文档详细规划了 full-scale experiments（1319 GSM8K, 3 seeds），但没有执行。没有 hard gating 规则强制最低实验规模。这是两个迭代都存在的系统性问题。

**修复方向**: 在 task_plan 中增加 hard gate——"Pairwise Ortho 声称需要 N>=500 且 >=2 seeds；基于 N<500 的声称必须包含 CIs 并标记为 pilot estimate。"

### 根因 2: 关键 Ablation 后置（ideation 缺陷）
IGSD vs naive truncation 是 IGSD 贡献声称的基础对比，但被安排在 Phase 7（最后阶段）。当确认 IGSD 门无效时，所有涉及 IGSD 贡献叙述的 downstream 工作已完成。

**修复方向**: 在 ideation 阶段为每个声称的新算法预注册空假设（"IGSD vs naive truncation"）。将此作为 Phase 1 实验。

### 根因 3: Baseline 标准化缺失（方法论缺陷）
多个 baseline TPS 值共存（31.0, 33.8, 58.5），没有单一规范化参考。M3 的 TPS 混淆直接源于测量协议中没有分离 per-token speed 和 output-length effects。

**修复方向**: Phase 0 = baseline 标准化。单一参考 TPS，固定测量协议（generation-only, post-warmup, batch=1），写入共享配置。定义 speedup 时分别报告 TPS ratio 和 per-sample latency ratio。

---

## 7. 系统自检响应

未发现 `logs/self_check_diagnostics.json`。无需特别响应。

---

## 8. 成功模式提炼

1. **诚实报告负面结果**: 本迭代的知识诚信是最大的亮点。M2 NO_GO、d2Cache 失败、AR gap、IGSD null ablation、M1+M3 干扰——全部透明报告。Supervisor 和 Reviewer 均认为这是罕见且值得赞赏的。这为论文建立了可信度，应当在后续迭代中保持。

2. **实验覆盖面全面**: 15 个实验组覆盖了完整的因子设计，包括个体 Pareto、pairwise、three-way、跨模型、AR 对比、batch sensitivity 和 ablation。这是结构良好的实验设计。

3. **iter_001 完整性问题的系统性修复**: 11 个 iter_001 问题被修复，包括伪造统计、QAS 不一致、alpha 错误等。双轮 Supervisor + Critic 审查证明了多轮独立审查的价值。

4. **CHR_refine 实测验证**: 从分析推导升级为实验测量（0.943），消除了 iter_001 的数据伪造疑虑。

5. **Dream-7B 跨模型成功**: 定性模式一致（M1+IGSD 最优、M3 反效果），尽管量级因低 baseline 膨胀。

---

## 9. 下一步行动优先级

### 第一梯队（阻断投稿，must-do）
1. **Full-scale pairwise replication**（8-10 GPU hrs）——最高影响单一改进
2. **IGSD contribution reframing**（2 hr text）——解决核心诚实性问题
3. **M1 caveats 添加**（2 hr text）——解决误导性问题
4. **Figure 2 + Figure 7 生成**（3 hr CPU）——跨两迭代的阻断项
5. **Citation 替换**（2-3 hr）——投稿阻断项

### 第二梯队（显著提升质量）
6. M3 speedup 测量澄清（1 GPU hr）
7. Statistical Limitations 子节（1 hr text）
8. 退化输出分析（2 hr）
9. Dream-7B 重新 framing（1 hr text）
10. 软化 "factorial study" 声称（30 min text）

### 第三梯队（完善细节）
11. 编辑遗留物清理（30 min）
12. N_gen 定义、r_accept/alpha 澄清（30 min）
13. Baseline TPS 标准化（1 hr）
14. Batch sensitivity baseline 补充（0.5 GPU hr）
15. AR 对比 vLLM caveat（15 min）
