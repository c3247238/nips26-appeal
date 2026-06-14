# Reflection Report -- Iteration 15

## "Unified Feedback Control Framework for Dynamic Weight Decay"

**Date:** 2026-04-02
**Iteration:** 15
**Quality Scores:** Supervisor 6.5/10 (JSON), Supervisor 6.5/10 (Markdown), Critic ~6.5/10
**Trajectory:** 下降 (iter_014: 7.0 -> iter_015: 6.5)
**Verdict:** ITERATE -- 关键完整性问题连续两次迭代未修复，评分首次从 7.0 回落至 6.5

---

## 1. 迭代总结

Iteration 15 是本项目第 15 轮迭代。**评分从 7.0 降至 6.5**，这是自 iter_012 以来首次下降。

**下降根因：** Supervisor 和 Critic 在 iter_014 已明确标记的三个关键完整性问题（CWD K_d 映射错误、AIS=0.566 伪造数据、CSI 公式不一致）在本迭代的论文中完全未修复。Supervisor 明确表示："unfixed critical issues across iterations should not receive the same or better score"。

**本迭代完成了什么：**
- 完成 review 和 critique 阶段的写作审查
- 所有 CIFAR 和 ImageNet 数值交叉验证零差异（表格准确性持续保持）
- H3 falsification 传播正确
- 负面结果报告结构维持

**本迭代未完成什么（全部与 iter_014 相同）：**
- CWD K_d 映射未修正（Table 1 仍声称 "K_d > 0"）
- AIS=0.566 未替换（实际值 0.123）
- CSI 公式未标准化（三个矛盾定义仍存在）
- Theorem 1 / Propositions 2-3 仍无证明
- CWD halved-lambda ablation 仍未执行（已标记 P0 四个迭代）
- ImageNet 仍为 12/40（已标记 11+ 个迭代）
- UDWDC-v2 BN bug 未修复
- Budget-matched controls 未运行

---

## 2. 问题分析（按类别）

### ANALYSIS（分析问题）-- 5 个问题，3 critical + 2 major

这是本迭代评分下降的主要驱动因素。三个 critical 分析问题从 iter_014 原样带入 iter_015，零修复：

1. **CWD K_d 映射错误（RECURRING, 第 2 次）**：Table 1 声称 CWD 对应 "K_d > 0, Derivative only"，但拟合数据显示 K_d=0.0000，拟合完全由 scale=0.5 驱动。这是论文核心映射表中的一个可验证错误。零 GPU 消耗的文本修复。

2. **AIS=0.566 伪造数据（RECURRING, 第 2 次）**：论文声称 AIS=0.566，实际测量值为 0.123。LOO-CV R^2=-0.18（零预测能力）。这不是四舍五入误差，是 4.6x 的偏差。零 GPU 消耗的文本修复。

3. **CSI 公式不一致（RECURRING, 第 3+ 次）**：三个互相矛盾的公式，UDWDC CSI_temporal=-5.75 在任何已声明公式下不可能产生。中央不稳定性发现的计算基础不可验证。

4. **Theorem 无证明（RECURRING, 第 7+ 次）**：Theorem 1 和 Propositions 2-3 连续 7 个迭代无证明。论文仍声称 "Theoretical analysis" 为贡献。

5. **Lyapunov 函数矛盾（RECURRING）**：V_t 在训练中经验上递增，直接与 Theorem 1 保证矛盾。如果降格为 Conjecture，此问题自动降级为开放问题。

### EXPERIMENT（实验问题）-- 7 个问题，4 critical + 3 major

1. **ImageNet 12/40 完成（RECURRING, 第 11+ 次）**：SWD、DefazioCorrective、NoWD 无结果。CWD 仅 1 seed。BEM 在 ImageNet 尺度不可计算。这是本项目历史上最顽固的未解决问题。

2. **CWD halved-lambda ablation 未执行（RECURRING, 第 4+ 次）**：FixedWD @ lambda=5e-5 vs CWD @ lambda=1e-4。仅需 ~1 GPU-hour on CIFAR-10。信息/GPU-hour 比最高的实验。四个迭代标记为 P0 但从未启动。

3. **CPR budget confound（RECURRING, 第 2+ 次）**：CPR 使用 2.3x WD budget，无预算匹配控制。需 FixedWD @ lambda=2e-4/3e-4，各 3 seeds，90 epochs。

4. **UDWDC-v2 BN bug（RECURRING, 第 3+ 次）**：Floor clipping 对 41 BN 层施加 lambda_min，产生 205,000x WD budget。一行代码修复未执行。

5. **Population vs sample std（RECURRING）**：ImageNet 结果使用 ddof=0，N=3-5 时差异显著。

6. **Figure 1 使用 pilot 数据（NEW）**：10-epoch pilot data 用于论文核心声明图。200-epoch 完整数据存在但未使用。

7. **无 budget-matched FixedWD controls for ImageNet（RECURRING）**。

### WRITING（写作问题）-- 4 个问题

1. **Title 过度承诺（RECURRING, 第 2+ 次）**："Unified Feedback Control Framework" 在 2/5 方法拟合失败时不准确。
2. **UDWDC 列为贡献但劣于 NoWD（RECURRING）**：应重新定义为诊断工具。
3. **Figure 1 是 TikZ 示意图而非数据（RECURRING）**：图注声称为实验数据。
4. **Notation 不一致（RECURRING）**：delta_t/alpha_t 双重标注，引用不存在的 notation table。

### EFFICIENCY（效率问题）-- 2 个问题

1. **GPU 利用率持续低下**：整个项目周期 GPU 利用率约 35%。写作迭代（iter_007-012）期间 8 GPU 完全闲置。
2. **时间估算系统性偏差**：CIFAR diagnostic 计划 60min 实际 554min (9.2x)。计划准确度仍无改善。

---

## 3. Fix Tracking（修复追踪）

### FIXED（从 iter_014 已修复）
- **无**。iter_015 没有修复 iter_014 action_plan 中的任何问题。这是评分下降的直接原因。

### RECURRING（反复出现）
| 问题 | 首次标记 | 反复次数 | 严重性 |
|------|---------|---------|--------|
| ImageNet 不完整 | iter_004 | 11+ | critical |
| Theorem 无证明 | iter_005 | 7+ | critical |
| CSI 公式不一致 | iter_007 | 4+ | critical |
| CWD halved-lambda ablation | iter_012 | 4 | critical |
| AIS 伪造数据 | iter_014 | 2 | critical |
| CWD K_d 映射错误 | iter_014 | 2 | critical |
| UDWDC-v2 BN bug | iter_014 | 2 | major |
| Population vs sample std | iter_014 | 2 | medium |
| Title 过度承诺 | iter_013 | 3 | major |
| GPU 低利用率 | iter_010 | 5+ | medium |

### NEW（新发现）
- Figure 1 使用 10-epoch pilot 数据而非 200-epoch 完整数据（minor，但影响论文核心声明）
- Batch size 432 + LR scaling 未说明（reproducibility，minor）

---

## 4. 质量趋势评估

```
iter_002: 8.2  ████████░░  (peak, negative-result paper)
iter_003: 5.5  █████░░░░░  (pivot to unified framework, -2.7)
iter_004: 5.5  █████░░░░░  (stagnant)
iter_005: 7.0  ███████░░░  (VGG-16-BN complete, +1.5)
iter_006: 5.0  █████░░░░░  (rewrite collapse, -2.0)
iter_007: 6.5  ██████░░░░  (+1.5)
iter_008: 6.5  ██████░░░░  (stagnant)
iter_009: 6.0  ██████░░░░  (-0.5)
iter_010: 6.5  ██████░░░░  (+0.5)
iter_011: 6.75 ██████░░░░  (+0.25)
iter_012: 7.0  ███████░░░  (+0.25)
iter_013: 6.5  ██████░░░░  (pivot to EqWD, -0.5)
iter_014: 7.0  ███████░░░  (+0.5)
iter_015: 6.5  ██████░░░░  (-0.5, regression)
```

**轨迹判定：停滞/下降。** 评分在 6.5-7.0 之间振荡已 7 个迭代。文本优化的天花板已在 iter_012 达到。iter_015 的回落证实：不解决完整性问题和不执行关键实验，评分只会下降不会上升。

---

## 5. 根因分析

### 为什么关键问题持续不修复？

**核心矛盾**：iter_014 action_plan 清楚列出了 20 个问题和 7 个优先项，其中前 4 项（AIS、CWD K_d、ImageNet、halved-lambda）都标记为最高优先。但 iter_015 一个都没做。

可能原因分析：
1. **实验/写作循环失调**：系统在写作阶段循环（review → critique → reflection），但未在 reflection 后插入实验执行。需要 orchestrator 在 reflection 后强制进入 experiment 阶段。
2. **零 GPU 成本修复被忽略**：CWD K_d、AIS、CSI、Theorem demotion 都是纯文本编辑（合计 2 小时），但写作阶段未执行这些修复。这表明写作 agent 未读取或未遵循 action_plan。
3. **P0 标记无执行机制**：CWD halved-lambda 连续 4 次标记为 P0 但无人执行。P0 标记在当前系统中是建议而非命令。

### 系统性弱点

1. **反馈回路断裂**：reflection → action_plan → (gap) → next iteration writing。action_plan 中的具体修复建议未被下一轮写作 agent 作为输入读取并逐项执行。
2. **实验执行惰性**：从 iter_007 到 iter_015（9 个迭代），GPU 仅在 iter_013 的 ImageNet 启动和 iter_014 的少量运行中使用。大量写作迭代浪费了 8 GPU 的计算资源。
3. **指标自引用循环**：BEM、CSI、AIS 都在同一实验内定义、计算和验证，缺乏外部锚定。

---

## 6. 资源效率评估

### GPU 利用率
- **整体利用率**: ~35%（估算）
- **iter_015 GPU 利用率**: 0%（纯 review/critique 阶段）
- **闲置 GPU-minutes**: 无法精确计算（gpu_progress.json 不存在），但 8 GPU x 24h/day x 约 65% 闲置 = 约 7,488 GPU-minutes/day 浪费

### 瓶颈分析
1. **实验执行延迟**：ImageNet 12/40 自 iter_013 启动以来无新进展。8 GPU 可在 2-3 天内完成剩余 28 runs。
2. **写作-实验串行化**：写作阶段（review → critique → reflection）不消耗 GPU，但系统未安排并行实验。
3. **零成本修复未执行**：4 项纯文本修复（合计约 2 小时工作量）可将评分从 6.5 提升至 7.0-7.5。

### 调度改进建议
- 在 reflection 后立即启动零 GPU 成本的文本修复（AIS、CWD K_d、CSI、Theorem demotion）
- 同时在后台启动 CWD halved-lambda ablation（~1 GPU-hour）
- ImageNet 剩余 28 runs 使用 3 对 GPU 并行（NoWD/SWD/DefazioCorrective）

---

## 7. 成功模式（值得保持）

1. **交叉验证纪律**: 所有 CIFAR 和 ImageNet 表格数值与原始 JSON 零差异。这是论文可信度的基础。
2. **负面结果诚实报告**: UDWDC 失败、fitting 失败、H3 falsification 全部透明报告。审稿人好感的来源。
3. **严格统计方法**: paired t-tests with Bonferroni correction、Cohen's d、TOST equivalence testing。
4. **Pre-generated 图表管道**: 13 张出版质量图表已就绪。
5. **定向编辑优于重写**: 增量改进（+0.25/迭代）比大幅重构（-2.0 风险）安全。

---

## 8. 致命警告

**如果 iter_016 仍不执行以下 4 项零 GPU 成本文本修复，评分将继续下降至 6.0 或更低：**
1. 替换 AIS=0.566 为 0.123 并报告 LOO-CV R^2=-0.18
2. 修正 Table 1 CWD K_d 映射（承认 scale=0.5 而非 K_d）
3. 标准化 CSI 为单一公式
4. 将 Theorem 1 降格为 Conjecture，移除 "Theoretical analysis" 贡献

这四项修复合计约 2 小时文本编辑工作，零 GPU 消耗，可将评分恢复至 7.0-7.5。

**不做这些修复就去做其他任何工作（包括实验）都是错误的优先级排序。**
