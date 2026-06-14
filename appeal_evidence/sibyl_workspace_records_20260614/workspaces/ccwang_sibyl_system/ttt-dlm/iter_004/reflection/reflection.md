# Iteration 4 反思报告

**日期**: 2026-03-10
**迭代**: 4
**质量分数**: Supervisor 5/10, Critic 4/10（Final Review）, 较上轮持平

---

## 1. 本轮迭代总结

迭代 4 是一轮完全重构方向的迭代——从 DTA（Denoising-Time Adaptation）框架转向 BSD（Belief-State Diffusion）+ A-CFG（Adaptive Classifier-Free Guidance）双方法框架。本轮完成了 18 个实验任务，覆盖 BSD/A-CFG 的 pilot 评估、多维消融、entropy 分析、GSM8K 扩展、compute-fair 比较和统计检验。然而，核心问题在于：**两个标题方法均仅在 n=16 pilot 规模上评估，所有 bootstrap 95% CI 均包含零，无法做出有意义的统计推断。**

**关键发现**：
- A-CFG 在 N=100 决策门实验中被判定为 **NO-GO**（vanilla 4% vs A-CFG 2%），这直接推翻了 pilot n=16 上的 +12.5pp 改善假象
- BSD 在所有评估中不如其特例 DMI（Countdown: 6.2% vs 12.5%, GSM8K: 18.8% vs 25.0%）
- DMI 仍然是唯一经全规模验证的有效方法（9.3% vs 4.7%, p < 0.05）
- compute-fair 分析表明 vanilla 步数扩展在匹配计算量下与所有方法持平

**核心矛盾**：论文标题和摘要以 BSD 和 A-CFG 为双主角，但 A-CFG 在 N=100 验证中失败（NO-GO），BSD 不如 DMI。论文实际只有 DMI 一个经验证的贡献。

---

## 2. 各类问题分析

### 2.1 EXPERIMENT（实验设计）

**[CRITICAL][RECURRING] 核心方法缺少全规模验证 → 已通过 N=100 决策门部分解决，但暴露更严重问题**

上轮要求 BSD/A-CFG 全规模验证。本轮引入了 N=100 决策门，这是一个改进。但决策门结果是 **A-CFG NO-GO**（vanilla 4% vs A-CFG 2%），直接证伪了 pilot n=16 上的 +12.5pp 假象。这验证了迭代 3 记录的 pilot-to-full-scale 系统性膨胀规律（最高 24pp）。

**然而论文仍以 A-CFG 作为标题贡献**。这是证据-声明失配的核心。

**[CRITICAL][NEW] compute-fair 数据一致性问题**

Supervisor 发现 `compute_fair_summary.md` 与主 pilot 结果使用了**不同的随机种子或样本集**：DMI 主 pilot 12.5% vs compute-fair 0%，A-CFG 主 pilot 12.5% vs compute-fair 0%。Table 4 混合了两次不同 run 的数据，属于苹果对橘子的对比。

**[MAJOR][RECURRING] 单模型评估（Dream-7B only）**

连续两轮提出但未解决。A-CFG 原始论文在 LLaDA-8B 上报告 GSM8K 73.5，本文 Dream-7B 上仅 pilot +12.5pp（现已被 N=100 证伪）。不做多模型评估无法判断 findings 的模型特异性。

**[MAJOR][NEW] Entropy-accuracy 相关性统计可信度存疑**

r = 0.784, p < 0.001 来自 n=16 样本，其中仅 1/16 correct。单个数据点的 entropy 值可以完全左右相关系数。需要提供原始数据对和统计方法说明。

**[MAJOR][NEW] A-CFG re-masking percentage 未消融**

m=10% 是关键超参但未做消融（5%, 10%, 20%, 30%）。

### 2.2 WRITING（写作质量）

**[CRITICAL][RECURRING] Abstract 与 Title 过度承诺**

上轮已指出 DTA→DMI 的 framing 重构需求。本轮虽然换了方向（BSD/A-CFG），但同一模式再次出现：标题和摘要将未验证（甚至已被证伪）的方法作为主要贡献。这是**连续 3 轮的 recurring pattern**（迭代 2: DTA overclaim, 迭代 3: DTA 仍在标题, 迭代 4: BSD/A-CFG 在标题）。

**[CRITICAL][RECURRING] 论文篇幅问题**

上轮提出 ~15,000 词 vs NeurIPS 6,000 词限制。本轮未解决。

**[MAJOR][RECURRING] 图表缺失**

连续 4 轮提出。所有 6 张图仍为文字描述。对于实验驱动论文，这是投稿基本要求。

**[MAJOR][NEW] DMI 结构位置倒置**

DMI 是唯一经验证的方法，但被放在 Appendix D。未验证的 BSD 获得了 Section 3 最详细的处理。结构上应该倒过来。

**[MAJOR][NEW] A-CFG 不是原创贡献但被呈现为原创**

A-CFG 由 Arriaga et al. (2025) 提出，在 Dream-7B 上应用无任何修改。应重新定位为"迁移实验"。

**[MINOR][RECURRING] 参考文献不完整、notation 不一致**

r = 0.784 vs r = 0.78; k_frac 表示法不统一; LLaDA 2.0/2.1 缺少 citation; Appendix C/D 未包含。

### 2.3 PLANNING（规划）

**[HIGH][RECURRING] Pilot 声明驱动决策的模式仍未根治**

迭代 3 记录了 pilot 24pp 膨胀教训。迭代 4 引入 N=100 决策门（改进），但 BSD 仍仅在 n=16 上评估，论文仍基于 n=16 数据构建机制解释。

**[HIGH][NEW] 研究方向频繁转换但深度不足**

迭代 1-2: PPL-based methods → 迭代 3: DTA + 全规模 → 迭代 4: BSD + A-CFG。每次转向都带来新的 pilot-only 方法，而 DMI 这个唯一成功的方法从未获得足够的消融支持。

### 2.4 IDEATION（构思）

**[MEDIUM][RECURRING] 泛化优于简单方法的假设反复失败**

BSD（DMI 的泛化）不如 DMI。DTA（更复杂的方法）不如 vanilla。A-CFG 在 N=100 上不如 vanilla。系统性模式：简单方法在 MDLM 推理时改进中更有效。

### 2.5 EFFICIENCY（资源效率）

**[MEDIUM][NEW] 实验资源分配不均**

18 个任务中大部分资源花在 BSD/A-CFG 的消融和扩展上，而 DMI——唯一有效方法——在本轮没有获得消融实验（alpha/tau sweep 在上轮已提出但仍未执行）。

---

## 3. 修复追踪

### 已修复（FIXED）
1. **引入 N=100 决策门**: 上轮要求的"早期证伪检查点"已部分实现。A-CFG 的 pilot_acfg_repro 任务使用 N=100 并正确判定 NO-GO
2. **统计检验执行**: gpu_progress 显示 `statistical_tests` 任务已完成，上轮的"统计检验零执行"问题已解决
3. **DTA 方向放弃**: 不再投入 DTA 实验资源（上轮建议"迭代 4 不应做 DTA 方向新实验"）
4. **研究方向快速转换**: 在 A-CFG N=100 NO-GO 后没有继续盲目投入

### 反复出现（RECURRING）
1. **Abstract/Title 过度承诺**: 连续 3 轮（迭代 2-4），核心模式未改变
2. **图表缺失**: 连续 4 轮
3. **篇幅超限**: 连续 2 轮
4. **核心方法缺少全规模验证**: 连续 3 轮（DTA→BSD/A-CFG 换了方法但问题相同）
5. **DMI 消融缺失**: 连续 2 轮
6. **单模型评估**: 连续 2 轮
7. **参考文献占位符**: 连续 3 轮

### 新发现（NEW）
1. compute-fair 数据一致性问题
2. Entropy-accuracy 统计可信度
3. DMI 结构位置倒置
4. A-CFG 原创性定位问题
5. BSD 泛化不如特例

---

## 4. 质量趋势判断

**趋势: 停滞（stagnant）**

| 迭代 | 分数 | 核心问题 |
|------|------|---------|
| 1 | 5.0 | 温度退火 bug，PPL 评估框架 |
| 2 | 5.0 | 方向转向准确率，统计检验缺失 |
| 3 | 5.0 | DTA null result，pilot 膨胀 |
| 4 | 5.0 | BSD/A-CFG 未验证，A-CFG NO-GO |

连续 4 轮质量分数为 5.0。每轮都有实质性实验进展，但同类结构性问题反复出现：方法未经全规模验证就放入标题、图表缺失、篇幅超限。**系统在"探索新方法 → pilot 看似有效 → 全规模验证失败/未做 → 论文 overclaim"的循环中**。

---

## 5. 资源效率评估

### GPU 利用率
- **总实验时间**: 125 分钟（实际） vs 700 分钟（计划），校准比 0.18
- **实验窗口**: 07:37 - 14:34（约 7 小时），实际 GPU 工作仅 125 分钟
- **利用率估算**: ~30%（考虑 2 GPU 并行的部分任务）
- **空闲原因**: 实验间的 agent 推理、代码调试、结果分析占用大量 wall-clock 时间

### 瓶颈分析
1. **Agent 推理/写作占比过高**: 7 小时窗口中仅 2 小时 GPU 工作，其余 5 小时为 agent 操作
2. **计划过度估计**: 700 分钟计划 vs 125 分钟实际，说明计划阶段对 GPU 任务时间估计偏差 5.6x
3. **DMI 消融未排入任务列表**: 连续 2 轮被列为"高优先级"但未获得 GPU 资源

### 正面进展
- **A-CFG N=100 决策门高效**: 仅用 19 分钟就判定 NO-GO，避免了全规模实验的浪费
- **零任务失败**: 18/18 任务全部完成
- **并行利用**: 部分任务使用 2 GPU 并行

---

## 6. 成功模式

1. **N=100 决策门是重要方法论改进**: pilot_acfg_repro 用 N=100 正确判定 A-CFG NO-GO，避免了全规模实验浪费（~18 GPU-hours），验证了上轮提出的"早期证伪检查点"建议
2. **科学诚实性持续优秀**: A-CFG NO-GO 结果如实记录，没有试图掩饰或换角度解读
3. **实验覆盖全面**: 18 个任务覆盖 4 个方法（BSD/A-CFG/DMI/RACFG）× 多维消融 + 跨基准 + compute-fair
4. **Pre-registered hypotheses**: 11 个假设的验证表格是科学透明性的典范
5. **Token-level 诊断框架**: Correction Precision/Recall 为 DLM 社区提供了新的分析词汇
6. **Information island problem**: 清晰的问题定义，被 Supervisor 和 Critic 一致认可为有价值

---

## 7. 根因分析

### 为什么质量分数 4 轮未提升？

**根因 1: 方法探索优先于论文完善**

每轮迭代都投入大量资源探索新方法（iter1: anneal fix, iter2: accuracy pivot, iter3: DTA full-scale, iter4: BSD/A-CFG），而论文基础设施（图表、篇幅、统计检验、完整 appendix）从未被优先处理。结果是论文始终处于"有新发现但呈现不完整"的状态。

**根因 2: Pilot-to-full-scale 膨胀教训未被制度化**

虽然迭代 3 记录了 24pp 膨胀并在迭代 4 引入了 N=100 决策门，但论文叙事仍基于 n=16 pilot 数据构建。决策门用于实验方向决策（正确），但论文写作仍将 pilot 结果作为贡献呈现（错误）。

**根因 3: DMI 一直被低估**

DMI 从迭代 3 开始就是唯一经验证的有效方法，但连续 2 轮被放在 Appendix 或作为 baseline 处理。系统偏向"更复杂=更有价值"的假设，忽略了简单方法的实用价值。

---

## 8. 系统自检响应

`logs/self_check_diagnostics.json` 不存在。基于手动分析的诊断：

1. **质量停滞警告**: 连续 4 轮 5.0 分，需要根本性策略调整
2. **Recurring issues 积累**: 7 个 recurring 问题中 5 个从未解决（图表、篇幅、单模型、DMI 消融、参考文献）
3. **方向振荡**: 4 轮 3 次方向转换（PPL→accuracy→DTA→BSD/A-CFG），每次都带来新的 pilot-only 方法

---

## 9. 迭代 5 战略建议

基于 4 轮迭代的累积教训，迭代 5 应该从"探索新方法"模式切换到"完善已验证贡献"模式：

1. **立即停止探索新方法**: DMI 是唯一经验证的方法，BSD 和 A-CFG 均已被证伪或无法证实
2. **论文重新定位**: 从"两个新方法论文"重定位为"MDLM 推理时计算扩展的系统性研究 + DMI 实用贡献 + 设计空间约束"
3. **DMI 消融是最高优先级实验**: alpha/tau sweep, 负对照, hard baseline
4. **论文基础设施**: 图表渲染、篇幅压缩到 6000 词、Appendix 完善
5. **多模型验证**: 至少加 LLaDA-8B 的 DMI 评估
