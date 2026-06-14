# Reflection Report — Iteration 8
## "When Does Dynamic Weight Decay Help? A Unified Framework Analysis"

**Date:** 2026-03-19
**Iteration:** 8
**Quality Score:** 6.5/10 (Supervisor: 6.5, Critic: ~6.5, Writing Review: 6.5)
**Trajectory:** 持平 (iter_007: 6.5 -> iter_008: 6.5)
**Verdict:** ITERATE — 四个关键问题连续两次迭代未取得进展

---

## 1. 迭代总结

Iteration 8 在 iter_007 基础上做了以下改进：
- **BEM half_lambda 修正**: Table 6 现在报告 BEM=0.500（数学正确值），解决了先前 BEM=0.000 的 bug
- **Theorem 2 重新表述**: 累积对齐分析重新表述为不显著观察，措辞更诚实
- **Figure 3/8 图注更新**: 部分图注修正

**未解决的关键问题（与 iter_007 完全相同）：**
1. 数据溯源不一致：Table 2 使用 iter_003 数据，PMP-WD/Figure 8 使用 iter_006 数据，跨迭代偏移 0.33pp > 论文声明的 0.25pp 方法间差异
2. 无附录证明：第 5 次连续迭代缺失 4 个定理的证明
3. 无 ImageNet 实验
4. Lyapunov 证书矛盾：V_t 经验上递增，与 Theorem 1 的保证矛盾

**新发现的问题：**
- PMP-WD "幽灵方法"：Figures 3/8/9 中出现 PMP-WD，但论文正文 7 方法目录中未定义
- Figure 3(a) 显示 0.49pp spread，而 Section 5.2 文本声称 0.92pp spread（实为 SGD 数据误归因到 AdamW 讨论中）
- Figure 4 (BEM scatter) 中 half_lambda 仍绘制在 BEM~0.0 位置，与 Table 6 的 BEM=0.500 矛盾
- 论文末尾保留了内部 "Figures and Tables" 构建清单

---

## 2. 按类别的问题分析

### SOUNDNESS (致命级)

**数据溯源不一致 [RECURRING, 3rd iteration]**
Table 2 使用 iter_003 数据（constant=90.13, seeds: 90.48/90.03/89.89），PMP-WD 和 Figure 8 使用 iter_006 数据（constant=89.80, seeds: 89.72/90.15/89.54）。同一配置跨迭代的 0.33pp 偏移超过论文核心发现的 0.25pp 方法间 spread。这意味着论文的"所有方法等价"结论可能被不受控的批间方差混淆。这是最高优先级修复项。

**Lyapunov 证书矛盾 [RECURRING, 4th iteration]**
Section 5.7 声称"band 快速收窄"，但 Figure 8 (certified_band.png) 显示混乱振荡线条，无可见收窄。V_t 经验上递增，直接矛盾 Theorem 1。Critic 建议直接删除 Section 5.7 和 Figure 8——鉴于 5 次迭代未能解决，这是务实选择。

**理论-实验优化器不匹配 [RECURRING, 3rd iteration]**
所有定理假设 SGD 动力学，所有主要实验使用 AdamW。论文从未讨论这一鸿沟。

### EXPERIMENT (关键级)

**无 ImageNet [RECURRING, 4th iteration]**
项目约束明确要求 ImageNet。3+ 次尝试失败但无根因诊断。CIFAR 对 2026 年的顶会来说是玩具规模。

**PMP-WD 不一致集成 [NEW]**
PMP-WD 出现在 Figure 3/8/9 但不在 Table 1/2/3 中。p=0.12 未报告。论文声称 7 种方法但 PMP-WD 是第 8 种。

**NoBN 消融数据未使用 [RECURRING, 2nd iteration]**
iter_005 的 NoBN 数据（constant: 87.74±0.21, CWD: 87.62±0.13）存在但未在论文中报告。NoBN spread (0.12pp) 比 VGG-16-BN (0.16pp) 更窄，可能颠覆 BN 机制解释。

**统计功效不足 [RECURRING]**
N=3 seeds 对 TOST 的功效仅 15-20%。6/12 比较通过等价检验（delta=1.0%），无法支撑"所有方法统计等价"的表述。

### WRITING (严重级)

**Figure-text 不一致 [NEW]**
- Figure 3(a): 显示 6 种方法 + 0.49pp spread; Section 5.2: 声称 0.92pp（实为 SGD 数据）; Table 2: 0.25pp
- Figure 4: half_lambda 绘制在 BEM~0.0; Table 6: BEM=0.500
- Figure 3 两个 panel 使用不同方法集，无法有效做跨架构比较

**无附录证明 [RECURRING, 5th iteration]**
论文引用 Appendix A/A.3 但附录不存在。理论密集型论文无证明将被直接拒稿。

**摘要-正文范围脱节 [RECURRING, 2nd iteration]**
摘要声称"7 methods, 2 optimizers, 2 architectures, 105 experiments"，暗示全因子设计。实际 VGG-16-BN 仅在 SGD/CIFAR-10 下测试。

**内部构建清单残留 [NEW]**
论文末尾 lines 432-448 包含"Figures and Tables"文件索引，是内部文档而非论文内容。

### ANALYSIS (主要级)

**Theorem 2 验证为阴性 [FIXED in presentation, RECURRING in substance]**
rho=-0.379, p=0.121。Section 5.8 现在诚实报告，但 Theorem 2 仍列为正式贡献。应降级为命题。Section 5.8 使用的符号（delta_t, bar_delta_T）在 Section 3 中未定义。

**CSI 无预测价值 [RECURRING]**
CSI 与准确率的 rho<0.3, p>0.3。任意权重（0.4/0.3/0.3）组合的指标无预测能力，质疑其作为"贡献"的合理性。

---

## 3. 资源效率评估

### GPU 利用率
Iteration 8 运行了**零新 GPU 实验**。所有工作为分析和写作修订。鉴于有大量零计算成本的数据集成工作可做（VGG-16-BN、NoBN、SGD 数据），这一决策是合理的——但集成工作也未完成。

### 瓶颈分析
1. **数据集成停滞**: VGG-16-BN（iter_005）、SGD（iter_003）、NoBN（iter_005）数据均存在但连续 2 次迭代未被集成到论文中。这是零计算成本、最高 ROI 的工作。
2. **Appendix 写作连续 5 次迭代未分配时间**: 系统性地低优先化附录写作。
3. **ImageNet 连续 4 次失败无诊断**: 未投入时间诊断根因（OOM？数据路径？）。
4. **Figure 重生成未执行**: PMP-WD 幽灵方法和 BEM 散点图不一致已被识别但未修复。

### 调度改进建议
- 下一迭代应以零计算任务为先：数据集成、figure 重生成、appendix 写作
- ImageNet 需先做 2 小时诊断再投入 GPU 时间
- 将一致性 rerun（21 runs, ~6 GPU-hours）作为实验最高优先级

---

## 4. 质量趋势评估

| Iter | Score | Trajectory |
|------|:-----:|-----------|
| 0 | 5.0 | - |
| 1 | 7.8 | +2.8 |
| 2 | 8.2 | +0.4 (peak) |
| 3 | 5.5 | -2.7 (pivot to unified framework) |
| 4 | 5.5 | 0.0 |
| 5 | 7.0 | +1.5 |
| 6 | 5.0 | -2.0 |
| 7 | 6.5 | +1.5 |
| 8 | 6.5 | 0.0 |

**趋势**: 停滞。分数在 5.0-7.0 区间振荡，从未恢复到 iter_002 的 8.2 峰值。核心原因：
1. Pivot 后引入了大量新理论要求（4 个定理、3 个指标），但证明和验证持续缺失
2. 每次写作重写引入的新不一致性速度超过旧问题修复速度
3. 已存在的实验数据未被集成到论文中

---

## 5. 根因分析

### 为什么分数停滞在 6.5？

**根因 1: 数据溯源问题是结构性的**
论文的核心发现（0.25pp inter-method spread）建立在混合不同迭代数据的基础上。0.33pp 的跨迭代偏移使这一发现在统计上不可信。修复需要一次统一 rerun（21 runs），但连续 2 次迭代未执行。

**根因 2: 理论承诺超越验证能力**
论文承诺了 4 个定理但 0 个证明。Theorem 1 被数据矛盾，Theorem 2 验证为阴性，CSI 无预测价值。理论框架的信誉严重受损。

**根因 3: 增量修复被全面重写替代**
每次迭代倾向于全面重写论文而非增量编辑。这导致每次重写引入新的不一致性（如 PMP-WD 出现在图中但不在文中）。

**根因 4: 零计算任务被持续推迟**
VGG-16-BN、SGD、NoBN 数据集成是零计算成本的高 ROI 任务，但连续多次迭代未执行。附录写作同样被推迟。

---

## 6. Fix Tracking (与 iter_007 action_plan.json 对比)

### FIXED
- **A1 (BEM half_lambda)**: Table 6 现在正确报告 BEM=0.500。但 Figure 4 仍显示旧值。部分修复。
- **E1 (Theorem 2 presentation)**: Section 5.8 现在诚实报告 p=0.121 为不显著。改进但未完全降级。

### RECURRING (未取得进展)
- **E2 (V_t contradiction)**: 第 4 次标记，未解决
- **E3 (Scope narrow / data integration)**: 第 3 次标记，VGG/SGD/NoBN 数据仍未集成
- **E4 (SGD-AdamW mismatch)**: 第 3 次标记，未讨论
- **E5 (Data provenance)**: 第 3 次标记，未修复
- **W1 (Missing figures)**: 部分修复（图已生成但内容不一致）
- **W2 (Missing appendices)**: 第 5 次标记，0 进展
- **W3 (Theorem numbering)**: 未修复
- **W4 (Abstract-body disconnect)**: 未修复
- **P1 (Writing-experiment decoupling)**: 第 8 次迭代，仍然存在
- **P2 (ImageNet failures)**: 第 4 次标记，无诊断

### NEW
- PMP-WD 幽灵方法（Figures 3/8/9 中出现但论文未定义）
- Figure 3(a) spread 0.49pp vs text 0.92pp vs Table 2 0.25pp 三方不一致
- Figure 4 BEM scatter 与 Table 6 不一致
- Build manifest 残留在论文末尾
- Section 5.8 使用未定义符号

---

## 7. 成功模式提取

1. **统计诚信**: Paired t-tests、Bonferroni 校正、Cohen's d、TOST 等价检验、显式功效分析。这是论文相对于社区规范的竞争优势。
2. **Phi modulator 分类法**: Table 1 连续 6 次迭代获得一致好评。CWD/random-mask/PMP-WD 的 bang-bang 控制洞察是最强结构性贡献。
3. **"Weight decay illusion" 框架**: 吸引人、令人印象深刻的学术框架。
4. **负面结果诚实报告**: Theorem 2 验证展示了如何科学地处理阴性结果——计算相关性、报告 p 值、让数据说话。
5. **三阶段审查流水线**: Supervisor、critic、writing review 捕获不重叠的问题。
