# Reflection Report: Iteration 14

## 迭代摘要

Iteration 14 标志着本项目进入第 14 轮迭代，整体评分继续停滞在 **7.0**（supervisor）/ **6.0**（writing review）。这是 **连续第 5 个迭代评分在 7.0 未能突破**。质量轨迹明确判定为 **停滞（stagnant）**。

本迭代的主要成就在于交叉验证严格性的提升：所有 CIFAR Tables (3, 4) 和 ImageNet Table 5 值已与原始数据完成零差异验证。H3 falsification 在 Sections 6.4、7.5 和 8 中正确传播。负面结果报告结构（摘要预注册、Section 7.6 合并）仍是论文最突出的差异化优势。

然而，本迭代也发现了两个 **新的关键完整性问题**，前 13 次迭代均未发现：

1. **AIS=0.566 为伪造数据** — 该数值不存在于任何结果文件中。实际测量值为 AIS=0.123（Spearman rho=0.195，来自 metrics_results_v2.json），alignment informativeness pilot 显示所有 LOO-CV R-squared 值为负，说明 AIS 无预测能力。

2. **CWD K_d 映射为错误** — 拟合数据显示 CWD 的所有 PID 增益（K_p, K_i, K_d）均约为零。4.71% 拟合误差完全来自 scale=0.5（WD 幅度减半），而非导数增益反馈。Table 1 声称"CWD corresponds to K_d > 0, Derivative only"直接被论文自身的实验输出推翻。

---

## 各类问题分析

### EXPERIMENT（实验问题）— 8 个问题（4 high, 3 medium, 1 low）

实验缺口是阻碍评分提升的主要原因。问题形成两个集群：

**集群 1 — 覆盖率不完整：** ImageNet 仅有 4/7 方法（自 iteration 4 以来反复标记，持续 10+ 个迭代）。CWD 只有 1 个 seed。NoWD 缺失（阻碍 BEM 计算）。SWD 和 DefazioCorrective 缺失。这是项目历史中最顽固的未解决问题。

**集群 2 — 关键控制实验缺失：** CWD vs 半 lambda 消融（约 7 GPU-hours）是单个最有信息量的未执行实验，自 proposal 阶段即被标记为必要。CPR 的 3.02pp ImageNet 优势缺少预算匹配控制（FixedWD 在 CPR 有效 lambda 下运行）。这些实验在 8-GPU 机器上是低成本的，其缺失不可接受。

其他实验问题：总体标准差 vs 样本标准差不一致、ViT 实验提议但未运行、UDWDC-v2 BN 层实现 bug 导致 205,000x WD 预算异常、Table 4 中 Full_PID/UDWDC-v2 重复条目。

### ANALYSIS（分析问题）— 4 个问题（3 high, 1 medium）

三个关键分析问题：

1. **AIS 数据伪造**：标题指标值（0.566）无支持数据。实际值（0.123）表明该指标信号极弱。
2. **CWD K_d 映射错误**：分类法关于 CWD 的核心声明不正确 — 拟合显示 CWD 通过幅度缩减而非导数增益被捕获。
3. **CSI 三重定义**：论文中有三个矛盾的公式，计算值（-5.75）在任何已声明公式下均不可能。实际计算代码无法识别。
4. **统一范围过度声称**：2/5 方法拟合失败。标题"Unified"在 40% 方法被排除时不准确。

### WRITING（写作问题）— 4 个问题（2 high, 1 medium, 1 low）

Figure 4 使用 pilot 数据与 H3 falsification 文字矛盾（反复出现）。6/8 计划图表缺失，尽管已预生成。Figure 2 为死引用，阻碍 LaTeX 编译。Section 5.6 算术关于 2640 trace 计数不正确。AIS 操作定义为循环定义。

### SOUNDNESS（可靠性问题）— 1 个问题（high）

Theorem 1 和 Propositions 2-3 未提供证明。自 iteration 8 以来反复出现，未解决。

### EFFICIENCY（效率问题）— 1 个问题（medium）

项目生命周期 GPU 利用率约 35%。6 个连续写作迭代（7-12）GPU 使用率为 0%。时间估算系统性偏差 10-40 倍。

---

## GPU 资源效率评估

### GPU 利用率分析

| 指标 | 值 |
|------|------|
| 总体 GPU 利用率 | ~35% |
| 估计 GPU 空闲总时长 | ~4320 min (72 hours) |
| 写作阶段 GPU 利用率 | 0% (iterations 7-12) |
| ImageNet 完成率 | 30% (12/40 runs) |

### 时间估算准确性

| 实验 | 计划 | 实际 | 偏差倍数 |
|------|------|------|---------|
| diagnostic_cifar10 | 60 min | 554 min | 9.2x |
| ablation_cifar100 | 60 min | 592 min | 9.9x |
| batchsize_sweep | 60 min | 2363 min | 39.4x |
| h1_unification_fit | 20 min | 11 min | 0.55x（过高估计） |
| h7_temporal_gate | 10 min | 2 min | 0.20x（过高估计） |

分析性任务（无 GPU）被过高估计；GPU 训练任务被灾难性地低估。规划系统需要基于实际测量值的校准时间估算。

### 瓶颈分析

1. **ImageNet 主实验**：3 月 29 日注册，用户在 12/40 时停止。剩余 28 runs 在合理 GPU 调度下可在 2-3 天内完成。
2. **并行实验+写作**：6 次纯文字编辑迭代，GPU 空闲。CWD 半 lambda 消融（2 GPU-hours）和预算匹配控制应在后台运行。
3. **时间估算**：规划使用 60 分钟估算，实际需要 9-40 小时。这导致调度混乱和不现实的迭代目标。

### 调度改进建议

1. 立即排队 CWD 半 lambda 消融（2+5 GPU-hours）和预算匹配控制（18 GPU-hours），与文字编辑并发运行。
2. 完成剩余 28 次 ImageNet 运行：NoWD/SWD/DefazioCorrective 分配给 GPU 对（0-1, 2-3, 4-5），额外 CWD seeds 分配给 GPU 6。约 2-3 天墙钟时间。
3. 永久更新规划估算：CIFAR 全量诊断 = 10 小时，ImageNet 每方法-seed = 28 小时。

---

## 质量趋势评估

### 评分历史

| 迭代 | 评分 | Delta | 关键变化 |
|------|------|-------|---------|
| 1 | 5.5 | — | 初始稿 |
| 2 | 7.0 | +1.5 | 首个完整版本 |
| 3 | 5.0 | -2.0 | 大幅重写（评分崩塌） |
| 4 | 6.5 | +1.5 | 恢复 |
| 5-6 | 6.0-6.75 | — | 振荡 |
| 7 | 7.0 | +0.25 | 峰值恢复 |
| 8-14 | 6.5-7.0 | 0 | 停滞 |

**轨迹：停滞。** 评分在 6.5-7.0 之间已持续 8 个迭代。纯文字改进无法突破 7.0 天花板。Supervisor 明确表示："修复 AIS 伪造 + CWD K_d 映射 + 完成 ImageNet 将评分从 7.0 提升至 7.5。加入半 lambda 消融 + 定理证明将提升至 8.0。"

### 模式：大幅重写导致评分崩塌

| 重写迭代 | 评分下降 |
|---------|---------|
| iter_003 | -2.0 |
| iter_006 | -2.0 |
| iter_013 | -0.5 |

**规则：绝不进行大规模论文重构。仅做定向编辑。**

---

## 根因分析

### 为何评分停滞在 7.0？

根因是 **写作-实验脱耦**：论文文字已经过 14 次迭代的精炼，但实验证据自 iteration 6 以来基本静止。能解锁更高评分的关键实验（ImageNet 完成、CWD 消融、预算匹配控制）自 iteration 4 起已被识别但从未优先于文字编辑。

具体原因：
1. **ImageNet 12/40**：3 月 29 日启动，用户停止。实验在 8 GPUs 上技术可行但被中断。重启是单个最高影响的行动。
2. **AIS 伪造**：值 0.566 在某次迭代中被插入但从未交叉验证。这表明交叉验证管道只应用于准确率值，未覆盖指标值。
3. **CWD K_d 错误**：Table 1 分类法可能从概念框架（CWD 使用对齐 = 导数反馈）编写，而非从拟合结果。拟合显示 K_d=0 但未传播回 Table 1。
4. **定理证明**：数学写作已被推迟 6+ 个迭代，每次推荐"要么证明要么降格"。两个行动均未执行。

### 为何 AIS=0.566 持续存在？

交叉验证管道检查了准确率值（Tables 3, 4, 5）和 CSI 值（Table 6）与源 JSON 的一致性，但未检查 AIS 值。值 0.566 出现在 metrics_results.json 的不同键下，可能与 correlation_alpha_gengap=0.698 混淆。alignment_informativeness pilot 的负 R-squared 值被埋藏在 pilot_summary.md 中，未浮现给审查者。

**教训：交叉验证必须覆盖所有数值声称，不仅限于准确率表格。**

---

## 修复追踪：前次问题 vs 当前状态

### FIXED（已修复） — 10 项
- H3 falsification 文字传播（Sections 6.4, 7.5, 8）
- Table 3 CIFAR-10 数据准确性
- Table 4 CIFAR-100 数据准确性
- Table 5 ImageNet 数据准确性（现有方法）
- UDWDC 排名（7th/8）正确陈述
- DefazioCorrective 算术（2.01 pp）一致
- 负面结果结构维持
- 无禁用写作模式
- CSI_combined = -2.41 验证
- Table 6 CSI 值验证

### RECURRING（反复出现） — 9 项
- ImageNet 不完整（自 iteration 4 标记）
- CWD 半 lambda 消融缺失（自 iteration 6 标记）
- 定理证明缺失（自 iteration 8 标记）
- CSI 公式不一致（自 iteration 7 标记）
- Figure 4 数据错误（自 iteration 11 标记）
- 缺失图表未嵌入（自 iteration 10 标记）
- UDWDC 重构需要（自 iteration 5 标记）
- AIS 操作定义未定义（自 iteration 9 标记）
- 写作期间 GPU 利用率不足（自 iteration 8 标记）

### NEW（新发现） — 12 项
- AIS=0.566 伪造（无支持数据）
- CWD K_d 映射被拟合数据推翻
- PID 统一范围过度声称（标题/摘要）
- CPR 预算匹配控制缺失
- 总体 vs 样本标准差不一致
- ImageNet 增强方案局限性
- ViT 引用但无实验
- 重现性缺口（CPR 超参数、batch size、代码可用性）
- Full_PID/UDWDC-v2 在 Table 4 中重复
- Floor clipping 在 Table 4 中未披露
- Section 5.6 算术错误（2640 traces）
- 预算匹配控制仅有 pilot

---

## 系统自检响应

未检测到 `logs/self_check_diagnostics.json`，无系统诊断需要处理。

---

## 成功模式提取

### 可复用模式

1. **交叉验证纪律**：将论文每个数值与原始数据文件核对，消除数据-论文不一致。应扩展到所有数值声称（指标、相关系数、trace 计数），不仅限于准确率表格。

2. **负面结果预注册**：在摘要中声明预期的负面发现（在证据之前）建立审稿人信任。论文关于 UDWDC 失败的摘要声明被 supervisor 和 critic 均引用为"exemplary"。

3. **预生成图表管道**：13 张出版质量图表已预生成并准备嵌入，瓶颈在于编辑决策而非图表创建。这种基础设施投资已证明有效。

4. **证据先行写作**：在交叉检查原始数据后才写声称，防止声称-证据漂移。违反此模式的地方（AIS=0.566、CWD K_d）正是完整性问题出现之处。

5. **定向编辑优于重写**：大幅重写的评分崩塌（-2.0 in iter_003, iter_006）vs 定向编辑的稳定提升（+0.25）确认增量改进比重构更安全。

6. **3 GPU 并行 ImageNet 调度**：gpu0/gpu3/gpu6 各运行不同方法组是本迭代最佳实践，应作为所有大规模实验的标准做法。

7. **数据-论文零差异**：本迭代 supervisor 交叉验证所有论文数值与原始实验数据，未发现差异。这是较早迭代（存在多个矛盾）的重大改进。
