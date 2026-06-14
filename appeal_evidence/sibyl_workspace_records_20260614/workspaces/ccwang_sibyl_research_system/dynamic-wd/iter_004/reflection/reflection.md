# Reflection Report — Iteration 4 (Updated Post-Revision)
## "When Does Dynamic Weight Decay Help? A Unified Framework Analysis Under AdamW"

**Date:** 2026-03-18
**Iteration:** 4
**Quality Score:** 6.5 / 10 (Supervisor review_writing.md), 6.5 / 10 (final_review.md), 6 / 10 (writing review)
**Consensus Score:** 6.5 / 10 — ITERATE
**Trajectory:** 从 Iteration 3 的 5.5 小幅提升至 6.5（写作改善，无新实验数据）

---

## 1. 迭代总结

Iteration 4 在 Iteration 3 的基础上进行了以下工作：
- **新增 5 个图表**：Figure 1-5（AdamW 分布、AdamW vs SGD 对比、BEM vs 准确率、权重范数收敛、AIS 分布），从零图到五图
- **VGG-16-BN pilot 实验**：3 个配置（constant, cwd_hard, no_wd）× 1 seed × 10 epochs — 但仅为 pilot，无统计权重
- **统计透明度显著提升**：明确区分"无显著差异"与"已证明等价"，报告 TOST 功效仅 15-20%
- **BEM bug 修复**：half_lambda BEM 从错误的 0.000 修正为 -0.500
- **Phi Invariance Conjecture 可证伪化**：加入显式预测 1-3 和边界条件
- **写作修订**：摘要、结论、限制章节全面改进；统计诚信声明

**未完成的关键任务**（均为 Iteration 3 反思报告中的 P0 级要求）：
- ImageNet 实验：未启动
- VGG-16-BN 完整实验（200 epochs, 3 seeds）：未执行
- NoBN 消融实验：未执行
- rho 扫描实验：未执行
- 种子数从 3 提升至 5+：未执行

---

## 2. 问题分类

### 实验不足 — 核心瓶颈（占拒稿理由的 ~60%）

| 问题编号 | 描述 | 严重性 | 状态 | 来源 |
|---------|------|--------|------|------|
| E1 | 仅 ResNet-20 (270K params) 一种架构，VGG pilot 无统计效力 | CRITICAL | 未解决 | 全部4个评审 |
| E2 | rho 仅测试 0.5 一个点，三元体系（Regime I/II/III）无任何验证 | CRITICAL | 未解决 | Supervisor M3, Critic M1 |
| E3 | BN 混淆因素：所有实验用 BN 架构，无法区分 AdamW 机制 vs BN 尺度不变性 | CRITICAL | 未解决 | Supervisor M4, Critic M2, Codex 2.3 |
| E4 | N=3 种子统计功效不足，MDE=0.77%，CIFAR-100 差距正好在检测阈值 | HIGH | 未解决 | Supervisor M2, Critic M3, Codex 2.2 |
| E5 | SGD vs AdamW 18.3x 比率混淆 rho 100x 差异，缺 matched-rho SGD 对照 | HIGH | 未解决 | Supervisor M5, Critic M4 |
| E6 | Phi 框架四轴仅覆盖两轴（时序+方向），空间和目标范数无数据 | MEDIUM | 未解决 | 延续自 iter_003 |
| E7 | CIFAR-100 SGD no_wd 仅 N=1（单种子） | LOW | 未解决 | Supervisor m2 |

### 理论深度 — 次要但重要

| 问题编号 | 描述 | 严重性 | 状态 |
|---------|------|--------|------|
| T1 | Lemma 1-3 证明 "in preparation"，附录 D 不存在 | HIGH | 未解决 |
| T2 | Proposition 2 存在 "formal gap"，应降级为猜想 | MEDIUM | 未解决 |
| T3 | CSI 指标三成分等权无理论依据，未验证预测能力 | MEDIUM | 未解决 |
| T4 | BEM/CSI/AIS 三个指标均为描述性统计，无外部验证 | MEDIUM | 新发现 (Codex 2.6) |

### 写作质量 — 显著改善但存在硬伤

| 问题编号 | 描述 | 严重性 | 状态 |
|---------|------|--------|------|
| W1 | Figure 2 SGD swd 显著性标注 (p=0.004) 与正文 (p=0.054) 矛盾 | CRITICAL | 新发现 (Writing Review) |
| W2 | Cohen's d 公式标注错误：声称 paired formula，实际使用 unpaired pooled | CRITICAL | 新发现 (Writing Review) |
| W3 | Figure 4 权重范数轨迹为"illustrative"重建，非实测数据 | HIGH | 新发现 (Writing Review) |
| W4 | Figure 3 CIFAR-100 面板标题 "No correlation" 但显示 r=0.48 | HIGH | 新发现 (Writing Review) |
| W5 | 段落 6.4 引用 Figure 5 但应为 Figure 4 | MEDIUM | 新发现 (Writing Review) |
| W6 | 摘要 313+ 词过长，前置过多统计数字 | MEDIUM | 未解决 |
| W7 | 附录 B/C/D 被引用但均不存在 | MEDIUM | 延续 |
| W8 | 图表 section 文件名与实际文件不一致 | LOW | 新发现 (Writing Review) |
| W9 | "Remark 1" vs "Observation 1" 命名不一致 | LOW | 新发现 (Writing Review) |
| W10 | Table 3 cwd_hard Cohen's d 应为 1.13 而非 1.08 | LOW | 新发现 (Writing Review) |

### 呈现与可视化

| 问题编号 | 描述 | 严重性 | 状态 |
|---------|------|--------|------|
| V1 | 缺少框架概览图（Phi 四轴分类图） | MEDIUM | 未解决 |
| V2 | 缺少决策流程图（7.1 节实用建议的可视化） | LOW | 新发现 |
| V3 | Figure 5 (AIS) 右面板过小，图例重叠不可读 | LOW | 新发现 |

---

## 3. 本轮改进了什么（vs Iteration 3 的 5.5 分）

### 实质性改进

1. **从零图到五图**：这是本轮最大的进步。Figure 1 和 Figure 3 被写作评审评为"视觉上令人信服"，直接支撑核心论点
2. **BEM bug 修复**：half_lambda BEM=0.000 → -0.500，消除了一个数据完整性问题
3. **统计诚信提升**：显式声明 N=3 的功效限制、TOST 不可行性、MDE 量化 — 评审一致认可此透明度"罕见且有价值"
4. **可证伪化框架**：Phi Invariance Conjecture 加入三条显式预测和精确边界条件，审稿人友好
5. **SGD 数据完整性修复**：iter_003 中被发现的 SGD p 值虚高问题已在最终稿中使用修正值
6. **CWD 与 Chen et al. 2026a 的矛盾**：Codex 指出了论文未讨论 CWD 在 Lion/Muon 下有效但 AdamW 下无效的解释

### 未能提升分数的原因

**核心矛盾未解决**：所有评审一致指出的根本问题 — 论文的理论雄心（命名猜想、三元体系、三个指标）与实证基础（单架构、单 rho、单尺度、N=3）之间的严重不匹配 — 在本轮中完全没有改善。Iteration 3 反思报告明确标记为 P0 的实验（ImageNet、VGG 完整、NoBN）一个都没执行。

**得分持平原因分析**：
- 写作改进（+0.5~1.0 分潜力）被新发现的 Figure 2 矛盾、Cohen's d 公式错误等硬伤抵消（-0.5~1.0 分）
- 图表从无到有是重要进步，但图表本身存在多处错误（illustrative 轨迹、交叉引用错误、面板标题矛盾）
- VGG pilot（10 epochs, 1 seed）被所有评审一致判定为"无统计权重"，不构成有效贡献

---

## 4. 经验教训

### L1. 实验执行是硬约束，写作优化是弹性投资
连续两个迭代的核心教训：论文质量的天花板由实验规模决定，而非写作质量。Iteration 3 和 4 都把大量资源投入写作修订（多轮 editor pass），但实验覆盖几乎没有扩展。写作从 "good" 到 "excellent" 的边际收益远小于从 "1 architecture" 到 "2+ architectures" 的边际收益。

### L2. Pilot 实验必须在迭代初期执行，不能替代完整实验
VGG pilot（10 epochs, 1 seed）消耗了时间但没有贡献任何可发表的证据。Pilot 的价值在于方向验证，应在迭代最初几小时完成，然后立即启动完整实验。

### L3. 图表生成后必须与正文数据交叉验证
Figure 2 的 p=0.004 vs 正文 p=0.054 矛盾、Figure 3 的 "No correlation" vs r=0.48 矛盾、Figure 4 的 illustrative 数据标注 — 这些都是生成图表后未进行系统性数据一致性检查的后果。

### L4. 反思报告的 P0 建议必须转化为迭代初始的硬约束
Iteration 3 反思报告明确列出 8 个优先事项，其中 P0 级有 4 个。Iteration 4 一个 P0 也未完成。反思报告沦为 "write-only" 文档。需要在编排器层面建立机制：反思报告的 P0 项直接写入 config/experiment_plan 作为 gating 条件。

### L5. 论文野心与证据规模必须同步调整
如果无法执行足够的实验，必须主动缩小论文声明范围。当前论文可以选择：(A) 保持统一框架野心，但必须有 3+ 架构、2+ 尺度的实验支撑；(B) 缩小为"ResNet-20/CIFAR 尺度下 AdamW 权重衰减调度无差异的实证研究"，降低对大规模实验的要求。

### L6. 数据一致性检查应自动化
两个迭代连续出现数据完整性问题（iter_003 的 SGD p 值虚高、iter_004 的 Figure/正文矛盾），说明人工检查不可靠。需要建立自动化的 figure-data consistency check。

---

## 5. Iteration 5 优先改进方向

### P0: 阻断性实验（必须在写作前完成）

| 优先级 | 实验 | 预计时间 | 解决问题 |
|--------|------|---------|---------|
| P0-1 | NoBN 消融：ResNet-20 无 BN，AdamW，constant/cwd_hard/no_wd，3 seeds | ~1-2h | E3 (BN 混淆) |
| P0-2 | rho 扫描：rho={0.05, 5.0}，constant/cwd_hard/no_wd，3 seeds | ~3-4h | E2 (三元体系验证) |
| P0-3 | Matched-rho SGD：lambda=0.05, eta=0.1 (rho=0.5)，3 seeds | ~1-2h | E5 (18.3x 混淆) |
| P0-4 | VGG-16-BN 完整：200 epochs，constant/cwd_hard/no_wd，3 seeds | ~4-6h | E1 (单架构) |

**总计约 10-14 GPU-hours**。8x RTX PRO 6000 可在 2-3 小时内并行完成全部 P0 实验。

### P1: 统计增强

| 优先级 | 任务 | 预计时间 |
|--------|------|---------|
| P1-1 | 核心 AdamW 配置增加至 N=5 seeds（CIFAR-10，7 methods） | ~2-3h |
| P1-2 | CIFAR-100 SGD no_wd 补充至 3 seeds | ~0.5h |

### P2: 写作硬伤修复（必须在下轮修订中完成）

| 优先级 | 任务 |
|--------|------|
| P2-1 | 重新生成 Figure 2：移除 SGD swd 显著性标注 |
| P2-2 | 修正 5.3 节 Cohen's d 公式描述为 unpaired pooled |
| P2-3 | Figure 4 改用实测权重范数数据（从 epoch_metrics.jsonl）或改为柱状图 |
| P2-4 | 修正 Figure 3 CIFAR-100 面板标题（r=0.48 不应标注 "No correlation"） |
| P2-5 | 修正 6.4 节 Figure 交叉引用（Figure 5 → Figure 4） |
| P2-6 | Table 3 cwd_hard Cohen's d 从 1.08 修正为 1.13 |
| P2-7 | 统一 "Remark 1" / "Observation 1" 命名 |
| P2-8 | 更新 Figures section 文件名与内容描述 |

### P3: 理论补充

| 优先级 | 任务 |
|--------|------|
| P3-1 | 完成 Appendix D Lemma 证明（或至少完整证明纲要） |
| P3-2 | Proposition 2 降级为 Conjecture 或补充 formal gap |
| P3-3 | 讨论 CWD vs random_mask 在 Lion/Muon 下的差异（回应 Chen et al. 2026a） |
| P3-4 | AdamW 更新公式中衰减项是否包含 eta 的 convention 显式说明 |

### P4: 呈现优化

| 优先级 | 任务 |
|--------|------|
| P4-1 | 添加 Phi 框架概览图（四轴分类） |
| P4-2 | 压缩摘要至 <250 词 |
| P4-3 | 添加实践决策流程图 |
| P4-4 | Figure 5 (AIS) 增大尺寸，减少图例重叠 |

---

## 6. 质量趋势

| 迭代 | 核心贡献 | 实验规模 | 理论 | 图表 | 得分 |
|------|---------|---------|------|------|------|
| 2 | AADWD 负面结果 | ResNet-20, CIFAR | 无 | 0 | ~6.5-7 |
| 3 | Phi 框架 + 零假设 | ResNet-20, CIFAR, 84 runs | 平凡 (Prop. 1) | 6 | 5.5-6.0 |
| 4 | Phi 框架 + 猜想可证伪化 | ResNet-20, CIFAR, 87 runs + VGG pilot | 改善但未完成 | 5 (含错误) | 5.5 |

**趋势判断：连续两轮停滞。** 写作和理论框架在改进，但实验规模 — 唯一的硬约束 — 几乎没有变化（84 runs → 87 runs，VGG pilot 不计入有效数据）。

**关键洞察**：如果 Iteration 5 完成 P0 级实验（NoBN + rho sweep + matched-rho SGD + VGG full），预计可将分数提升至 7-8 分（Critic 和 Codex 均明确给出此预期）。这些实验仅需约 10-14 GPU-hours，在 8 块 GPU 上可在 2-3 小时内完成。**实验执行不足是唯一的系统性瓶颈。**

---

## 7. 系统性模式（跨迭代）

### 模式 1: "写作优先"陷阱
连续三轮迭代中，系统在实验不充分时即启动写作流水线，导致写作修订消耗了本可用于实验的迭代时间。
**建议**：Iteration 5 严格执行"实验完成 → 写作"的串行流程。编排器应在 P0 实验未完成时阻止进入 writing 阶段。

### 模式 2: P0 实验的执行失败率 100%
Iteration 3 反思定义了 4 个 P0 实验，Iteration 4 完成了 0 个。这不是资源问题（8 块 GPU 足够），是编排优先级问题。
**建议**：将 P0 实验写入 experiment_state.json 的 mandatory 字段，编排器在 P0 未完成时阻止进入 writing 阶段。

### 模式 3: 图表生成与数据一致性断裂
每轮迭代的图表都会出现新的数据不一致问题（iter_003: SGD p 值虚高；iter_004: Figure 2 矛盾、Figure 3 标题错误、Figure 4 非实测数据）。
**建议**：建立自动化的 figure-data consistency check 脚本，在 writing_final_review 前强制执行。

### 模式 4: Pilot 实验价值被高估
VGG pilot (10 epochs, 1 seed) 消耗了流水线时间但无发表价值，所有评审一致判定其无效。
**建议**：pilot 仅用于参数调优和可行性验证（<15 分钟），验证通过后立即启动完整实验，不纳入论文讨论。

### 模式 5: 论文野心膨胀但证据不跟进
Iteration 2 → 3 从负面结果扩展到统一框架，引入新要求（更广范围、更深理论、指标正确性）但未配套扩展实验。Iteration 3 → 4 进一步加入可证伪化框架和三元体系，但 rho sweep 未执行。
**建议**：如果 P0 实验在 Iteration 5 中仍无法全部完成，必须主动将论文降级为"CIFAR 尺度实证研究"，删除三元体系猜想和多数指标。

---

## 8. Iteration 5 成功标准

- **最低标准（及格）**：完成 P0-1 (NoBN) 和 P0-2 (rho sweep)，修复 P2 级写作硬伤。预期分数：6.5-7.0
- **目标标准（优秀）**：完成全部 P0 + P1-1 (seeds=5)，修复全部 P2/P3。预期分数：7.0-8.0
- **理想标准（提交就绪）**：全部 P0-P3 + ImageNet ResNet-50 至少 1 seed。预期分数：8.0+

**时间预算参考**：
- P0 实验：2-3 小时（8 GPU 并行）
- P1 实验：2-3 小时（可与 P0 交错）
- P2 写作修复：1-2 小时
- P3 理论补充：2-3 小时
- 总计：~8-11 小时，在单迭代内可完成

**Iteration 5 的唯一核心信条：实验先行，一切写作和理论工作必须等待 P0 实验全部完成。**
