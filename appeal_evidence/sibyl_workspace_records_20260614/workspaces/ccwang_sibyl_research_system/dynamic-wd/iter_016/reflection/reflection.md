# Reflection Report -- Iteration 16

## "Equilibrium-Driven Weight Decay: Adaptive Per-Layer Regularization via Gradient-Weight Ratio Dynamics"

**Date:** 2026-04-02
**Iteration:** 16
**Quality Scores:** Supervisor 7.0/10, Critic ~7.0/10, Writing Review 7.5/10 (prose quality)
**Trajectory:** 恢复 (iter_015: 6.5 -> iter_016: 7.0)，但停滞于 7.0 天花板（iter_005, iter_012, iter_014 均为 7.0）
**Verdict:** ITERATE -- 核心实验空缺仍未填补（WD 通胀控制、90-epoch ImageNet、AdamW）

---

## 1. 迭代总结

Iteration 16 是本项目历史上最重要的战略转折点。经过 15 轮在"PID 统一框架 / Phi 不变猜想"方向上的反复摇摆（评分在 5.0-8.2 之间剧烈波动），本迭代做出了决定性的战略选择：**完全放弃 PID 框架，聚焦于 EqWD 作为独立方法论文**。

### 本迭代的主要成就

1. **范围缩减成功解决了全部 4 个关键完整性问题**：
   - AIS=0.566 伪造数据 → PID 框架移除，AIS 重新定义为经验诊断，"residual variance ratio > 0.95"
   - CWD K_d 映射错误 → CWD 降级为普通基线，不再声称 PID 增益
   - CSI 公式矛盾 → CSI 完全移除
   - Theorem 1 无证明 → 所有 PID 理论删除，新的 Proposition 1-2 范围适当

2. **完成所有 7 方法 x 3 种子 ImageNet 实验**（45 epochs）：这是自 iter_004 以来持续 11+ 轮的最大实验空缺，终于填补。EqWD 72.27 +/- 0.20% vs FixedWD 71.89 +/- 0.24%，Cohen's d=1.72。

3. **CAWD 消融控制**：一个精心设计的对照实验，将 ratio 信号与 alignment 信号隔离，证明 alignment 计算不必要。这是优秀的实验方法论。

4. **论文内部一致性恢复**：supervisor 验证所有数值声称与原始数据零差异（ddof=1 正确使用）。

5. **诚实的局限性报告**：论文主动承认 WD 通胀混淆变量、SGDW-only 范围、CIFAR-100 上 EqWD 不优于 FixedWD。

### 本迭代未完成的关键任务

来自 iter_015 action_plan 的 18 个问题中：
- **4 个通过范围缩减间接解决**（AIS、CWD K_d、CSI、Theorem -- 不是修复，而是删除）
- **3 个不再适用**（UDWDC BN bug、标题范围、UDWDC 贡献 -- 因 UDWDC 移除而失效）
- **11 个仍未解决或转化为新形式**

---

## 2. 问题分类与修复追踪

### 已修复的问题（从 iter_015 action_plan 追踪）

| iter_015 问题 | 状态 | 修复方式 |
|---|---|---|
| AIS=0.566 伪造数据 | **FIXED** | 移除 PID 框架；AIS 重定义为诊断 |
| CWD K_d 映射错误 | **FIXED** | CWD 降级为基线，不再声称 PID 增益 |
| CSI 公式矛盾 | **FIXED** | CSI 完全移除 |
| Theorem 1 无证明 | **FIXED** | 所有 PID 理论移除 |
| 标题范围不匹配 | **FIXED** | 新标题聚焦 EqWD |
| UDWDC 贡献过度声称 | **FIXED** | UDWDC 完全移除 |
| UDWDC-v2 BN bug | **N/A** | UDWDC 移除 |
| Figure 1 TikZ 非数据图 | **FIXED** | 新 Figure 1 是真实实验对比图 |
| ImageNet 不完整（12/40）| **PARTIAL** | 7 方法 x 3 种子完成，但仅 45 epochs |
| Population vs sample std | **FIXED** | 现在使用 ddof=1 |

### 仍未解决的问题（从 iter_015 继承并转化）

| 问题 | 原形式 | 新形式 | 状态 |
|---|---|---|---|
| WD 预算混淆 | CPR 2.3x budget | EqWD phi>=1 通胀 | **RECURRING** |
| Budget-matched FixedWD | lambda=2e-4/3e-4 | lambda 匹配 EqWD 有效平均 | **RECURRING** |
| ImageNet 标准训练 | 12/40 完成 | 45 vs 90 epoch | **RECURRING** |
| BEM 负面结果隐瞒 | BEM 排名矛盾 | 改为 WD 通胀叙事 | **PARTIALLY FIXED** |
| GPU 利用率低 | 35% | 未知（无 gpu_progress.json）| **RECURRING** |

### 新发现的问题

1. **CRITICAL: EqWD WD 通胀混淆变量** -- phi_l(t) >= 1 设计意味着 EqWD 总是比 FixedWD 施加更多 WD。+0.38% 改进可能完全来自更强的正则化，而非自适应调制。论文承认但未实验控制。
2. **CRITICAL: 45-epoch ImageNet 非标准训练** -- 标准 ResNet-50 90-epoch 达到 ~76%，EqWD 仅 72.27%。论文未明确说明 epoch 数。
3. **MAJOR: LR schedule 描述可能错误** -- 论文声称 cosine annealing，但原始数据显示 step decay（epoch 30 处 10x 下降）。
4. **MAJOR: Lambda 值 5x 差异** -- 原始数据显示 1e-4，论文声称 5e-4。必须确认。
5. **MAJOR: Proposition 2 是空真命题** -- "if alignment is a function of norms, then the ratio captures alignment" 是同义反复。
6. **MAJOR: 无 AdamW 实验** -- 2026 年论文仅测试 SGDW 是严重范围限制。
7. **MAJOR: Beta=5.0 单种子声称** -- 66.07% 可能是噪音；与 3 种子默认值 65.05% 差距 1%，但未验证。
8. **MAJOR: CIFAR-10 表缺 EqWD** -- 论文主方法在三个基准表之一中缺失。
9. **MAJOR: AIS 仅在 CIFAR-100 验证** -- EqWD 在 ImageNet 上赢，但 ratio 充分性仅在 CIFAR-100 验证。
10. **MAJOR: DefazioCorrective 在 CIFAR-10 表中未定义** -- 与论文其余部分不一致。

---

## 3. 质量趋势评估

### 评分历史

| Iteration | Score | Key Change |
|---|---|---|
| iter_000 | 5.0 | 初始状态 |
| iter_001 | 7.8 | 负面结果论文，结构良好 |
| iter_002 | 8.2 | **历史最高**，论文接近发表就绪 |
| iter_003 | 5.5 | PID 框架 pivot，引入新需求 |
| iter_004 | 5.5 | 停滞 |
| iter_005 | 7.0 | VGG-16-BN 完成 |
| iter_006 | 5.0 | 大幅重写导致崩塌 |
| iter_007 | 6.5 | 部分恢复 |
| iter_008 | 6.5 | 停滞 |
| iter_009 | 6.0 | 数据完整性问题 |
| iter_010 | 6.5 | 图表一致性修复 |
| iter_011 | 6.75 | 所有数据完整性问题解决 |
| iter_012 | 7.0 | 文本改进天花板 |
| iter_013 | 6.5 | EqWD pivot，新空缺 |
| iter_014 | 7.0 | 恢复，但发现伪造数据 |
| iter_015 | 6.5 | 完整性问题未修复导致下降 |
| **iter_016** | **7.0** | **完全重写为 EqWD 论文，完整性问题通过范围缩减解决** |

**轨迹判定：停滞（stagnant）**。评分在 6.5-7.0 区间已维持 8 个迭代（iter_009-016）。论文已三次触及 7.0 天花板（iter_005, iter_012, iter_014, iter_016）但从未突破。突破需要实验层面的进展。

### 质量震荡根因分析

本项目经历了三次大幅重写导致的评分崩塌：
1. iter_003: 5.5（从 8.2 的负面结果论文 pivot 到 PID 框架）
2. iter_006: 5.0（PMP-WD 大幅重写）
3. iter_013: 6.5（从 PID 框架 pivot 到 EqWD）

**关键教训**：每次大幅重写平均导致 -1.5 分下降，需要 3-5 个迭代恢复。iter_016 是 iter_013 pivot 后第 3 个迭代，刚恢复到 7.0。这验证了"定向编辑优于重写"的经验法则。

---

## 4. 系统性模式

### 反复出现的系统模式

1. **WD 预算混淆是项目基因问题**：从 iter_013 的 CPR 2.3x budget 到 iter_016 的 EqWD phi>=1，不同方法形式变化但核心混淆不变。Budget-matched control 是这个研究方向的必做实验，已拖延 4+ 轮。

2. **Action plan 到执行的断裂**：iter_015 action plan 要求的 4 个零 GPU 文本修复（AIS、CWD K_d、CSI、Theorem）没有被执行。iter_016 选择了完全重写而非定向修复。虽然重写的效果更好（间接解决了所有 4 个问题），但这不是计划执行的结果，而是偶然的副作用。

3. **45-epoch 非标准训练的隐匿**：论文从未明确声明 ImageNet 训练长度。这种关键实验细节的遗漏模式在多个迭代中重复出现（之前是 batch size 432 未说明、LR scaling 未说明）。

4. **单种子声称的风险容忍度过高**：beta=5.0 的 66.07% 是单种子结果，但论文用它支持"substantial headroom"叙事。N=3 时 multi-seed 验证只需 30 分钟，成本极低。

5. **CIFAR-10 表是遗留版本**：包含 DefazioCorrective（未在论文中定义）且缺少 EqWD（论文主方法）。跨迭代 pivot 时遗留数据表未更新是反复出现的问题。

---

## 5. 资源效率评估

### GPU 利用率

无 `gpu_progress.json` 文件，无法精确评估本迭代 GPU 利用率。但根据完成的工作量推断：

- **本迭代完成了 7 方法 x 3 种子 x 45 epoch ImageNet 实验**（~21 次完整训练运行）
- 加上 CIFAR 实验（ablation 等），估计使用了 50-100 GPU-hours
- 8 GPU 机器上，这大约是 6-12 小时 wall clock（如果充分并行化）
- 由于本迭代也完成了大量写作工作，实际 GPU 利用率可能在 40-60%

### 瓶颈分析

1. **45-epoch 限制**：选择 45 epoch 而非 90 epoch 节省了约 50% 的 ImageNet 训练时间，但代价是产出不可与标准文献比较的结果。这是错误的时间/质量权衡。
2. **缺失的低成本实验**：EqWD on CIFAR-10（~0.5 GPU-hr）、beta=5.0 multi-seed（~1 GPU-hr）、AIS on ImageNet（~3 GPU-hr）合计 ~5 GPU-hours，不到 1 小时 wall clock，却能填补 3 个 major gap。
3. **Budget-matched FixedWD 未执行**：~9 GPU-hours，是论文核心声称的最关键控制实验。

### 改进建议

- **并行调度所有独立实验**：budget-matched FixedWD（GPU 0-1）、90-epoch ImageNet（GPU 2-5）、CIFAR-10 EqWD + beta=5.0（GPU 6）、AIS on ImageNet（GPU 7）可以完全并行
- **预期 wall clock**：最长的 90-epoch 实验约 8-10 小时。其余实验在此期间完成。一天内可填补所有实验空缺。
- **文本修复可与实验并行**：LR schedule 验证、epoch 数声明、Proposition 2 降级、numbering fix 等零 GPU 成本工作应与实验同时进行

---

## 6. 成功模式提取

### 本迭代的成功决策

1. **完全 pivot 到聚焦方法论文**：放弃"统一框架"的过度承诺，聚焦于 EqWD 单一方法。这是项目历史上最正确的战略决策。Supervisor 评价："significant improvement in clarity and coherence"。

2. **通过范围缩减解决完整性问题**：与其修复 AIS=0.566（替换数字）、CWD K_d（修改表格）、CSI（统一公式），论文选择移除整个有问题的理论框架。这比定向修复更安全、更彻底。

3. **CAWD 负面结果作为正面贡献**：将 alignment-based WD 的失败转化为"ratio sufficiency"论点的证据。这是将负面结果转化为贡献的范例。

4. **诚实的统计报告**：bootstrap CIs 包含零时明确声明，使用"directional trend"而非"significant improvement"。这种校准过的声称水平是论文最强的差异化优势。

5. **出版质量图表管道**：9 张图表质量高、信息量大、与文本一致。

### 可复用的成功模式

- **范围缩减是最有效的完整性修复**：如果一个声称无法用证据支撑，删除它比修补它更好。
- **负面结果 = 方法论贡献**：精心设计的消融实验（CAWD）的负面结果比正面结果更有信息价值。
- **统计诚实赢得审稿人好感**：主动承认局限性比被审稿人发现更好。

---

## 7. 根因分析

### 为什么评分停滞在 7.0？

1. **核心声称未被验证**：EqWD 声称"自适应调制提供泛化益处"，但 phi>=1 意味着 EqWD 总是施加更多 WD。没有 budget-matched 控制，"自适应"和"更多"是不可区分的。

2. **实验范围过窄**：SGDW + CNNs 在 2026 年是少数派。没有 AdamW 实验意味着论文对主流实践无直接影响。

3. **非标准 ImageNet 训练产生不可比较的数字**：72.27% 在文献中没有对标物。审稿人无法将此与已知方法比较。

4. **理论贡献薄弱**：Proposition 2 是同义反复。AIS 仅在非关键基准上验证。剩余的理论贡献（Proposition 1 equilibrium result）是 Defazio 结果的直接推论。

### 突破路径

**到达 7.5（可投稿水平）**的最短路径：
1. 解决 lambda 和 LR schedule 差异（0 GPU，2 小时验证）
2. Budget-matched FixedWD on ImageNet（9 GPU-hours）
3. EqWD on CIFAR-10 + numbering fix（0.5 GPU-hours + 10 min）
4. Beta=5.0 multi-seed validation（1 GPU-hour）

**到达 8.0（强投稿水平）**需要额外：
5. 至少一个 AdamW 实验（0.5 GPU-hours）
6. 90-epoch ImageNet 对比（18 GPU-hours）
7. AIS on ImageNet validation（3 GPU-hours）
8. Proposition 2 降级为 Remark（0 GPU, 15 min）

---

## 8. 自检诊断响应

未发现 `self_check_diagnostics.json` 文件。无需特别响应。

---

## 9. 总结

Iteration 16 完成了项目最重要的战略转型——从一篇过度承诺的"统一框架"论文转变为一篇诚实聚焦的方法论文。这解决了困扰项目 5+ 迭代的所有完整性问题。但 7.0 的天花板依然存在，原因从完整性问题转变为实验证据不足。

**下一迭代的首要任务**：在进行任何写作之前，先执行 budget-matched FixedWD 控制实验。这是一个 9 GPU-hours 的实验，但它决定了论文的核心叙事是"自适应调制有益"还是"EqWD 自动发现更好的 WD 强度"。两种叙事都可发表，但论文必须知道自己属于哪一种。
