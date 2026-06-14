# Lessons from Iteration 9

**Date**: 2026-04-15 | **Score**: 7.0/10 | **Trajectory**: Improving (5.5 x3 -> 6.5 -> 6.0 -> 6.5 -> 6.5 -> 6.5 -> **7.0**)

**3-iteration stagnation loop BROKEN.** Experiment-first strategy worked. Score improved +0.5. Path to 8.0 is now clear.

---

## Must Improve

- **[SOUNDNESS -- IMMEDIATE, ZERO COST] 修复 12.3% 虚假数字**: Section 5.2 声称 L0=82 严格 hedging 为 12.3%。原始数据 (hedging_decomposition_full.json, L0_82_to_176) 显示 strict_hedging=0, 0.0%。这个数字在任何源数据文件中都不存在。是写作生成的幻觉。修正为 0.0%——这反而加强了论文的论点。修复后运行 validate_integration 检查。

- **[EXPERIMENT -- 最高优先级, 1-2 GPU-hours] 降级探针消融实验**: 这是解决论文核心不确定性的唯一实验。向 first-letter 探针注入标签噪声，降级到 F1={0.70, 0.80, 0.85, 0.90}，重新测量吸收率。如果降级后 first-letter 吸收率匹配 RAVEL 的率，跨域变化是探针伪影。如果率保持不同，层级效应是真实的。Supervisor 和 critic 都将此列为第一优先级。

- **[EXPERIMENT -- 高优先级, 1-2 GPU-hours] L24 层激活补丁**: 当前因果证据 (d=1.33) 来自 L12（5.7% 吸收）。论文的核心发现（L24 的 34.5% 吸收）没有因果验证。因果证据与主要发现脱节。必须在 L24 层扩展激活补丁实验。

- **[PIPELINE -- 第 9 轮推荐, 1.5 小时, 零 GPU] validate_integration.py**: 12.3% 的幻觉数字完美证明了这个脚本的必要性。从 paper.md 提取数值声明，与源 JSON 交叉验证。每次写作修订后运行。这是项目中最老的未解决系统性问题，第 9 次推荐。

- **[WRITING -- 零 GPU, 1-2 小时] 生成缺失的图 5 和图 6**: Sections 5.1 和 5.2 包含论文四个贡献中的两个，但没有任何视觉支持。图 5（激活补丁配对点图）和图 6（hedging 分解堆叠柱状图）可以从现有 JSON 数据生成，零 GPU 成本。

- **[WRITING -- 零 GPU, 15 分钟] 修复损坏的交叉引用**: "Table 7 in Section 4.4" 和 "Section 8.6" 都指向不存在的位置。修复前者为内联数据或删除引用，修复后者为 "Section 8.5"。

- **[WRITING -- 零 GPU, 10 分钟] 修改标题**: 当前标题以 "The Absorption Tax" 开头——一个全面失败的理论框架 (rho=-0.20)。改为反映实际贡献的标题，如 "Beyond First-Letter Spelling: Cross-Layer and Cross-Domain Characterization of SAE Feature Absorption"。

- **[SOUNDNESS -- 零 GPU, 30 分钟] 调查字母 G 在 hedging 中的主导地位**: G 占严格 hedging 的 37.5%（9/24 案例）。排除 G 和 A 后，严格 hedging 降至约 2.9%。必须计算排除 G 的统计数据并调查 G 的异常原因。

- **[WRITING -- 零 GPU, 30 分钟] 重构摘要**: 以无混杂的发现（15 倍层变化、d=1.33 因果证据）引领。将跨域变化标记为 "suggestive, confounded by probe quality"。精简到 <220 词。

- **[WRITING -- 零 GPU, 15 分钟] 删除层级 vs 架构优势声称**: 比较 L24 层级 p=0.005 与 L12 架构 p=0.87 来声称 "hierarchy matters more" 是统计谬误——不同数据、不同层、不同混杂。

---

## Watch Out

- **[SOUNDNESS] 写作生成幻觉**: 12.3% 是一种新的失败模式。之前的数据错误是管道不匹配（访问错误的 JSON 字段），不是凭空捏造的数字。写作 agent 需要生成后验证。所有数值声明必须与源 JSON 交叉检查。

- **[SOUNDNESS] 探针 F1 差异**: 整合记录 activation patching 使用 probe_f1=0.883，但论文暗示 F1>=0.97。必须明确每个实验使用了哪个探针以及其 F1。

- **[ANALYSIS] 跨域 hedging 比较无效**: N=3 vs N=304，不同层，不同架构。66.7% vs 7.9% 的比较误导性极强。

- **[ANALYSIS] city-continent 与 first-letter 统计不可区分**: p=0.83 和 p=0.93。叙事不应将其归入 "跨域变化"。

- **[EXPERIMENT] 时序异常**: 激活补丁计划 60 分钟，实际 2 分钟。可能使用了缓存结果。如果是，需要记录。

- **[ANALYSIS] pilot-to-full 方向反转**: L12 pilot (semantic > first-letter) 在 L24 反转。这进一步证明探针质量驱动了表观吸收差异。应在论文中明确讨论。

- **[WRITING] 附录被引用但不存在**: GAS (B), CMI (C), Tax (D), 阈值敏感性。全部承诺但未写。要么写，要么删除引用。

---

## Keep Doing

- **诚实负面结果报告（连续 9 轮）**: GAS/CMI/Tax 全部确定性负面，透明报告。论文最强方面。**绝不能妥协。**

- **实验优先策略（本轮验证）**: 3 轮停滞后，执行实验 -> 分数提升。分数改善与实验执行完美相关。继续保持实验优先。

- **激活补丁实验设计**: 匹配幅度的随机 latent 控制、多统计检验、restricted analysis 作为稳健性检查。d=1.33 是大效应。

- **Hedging 三分解**: 严格 7.9% / 补偿 86.2% / 持久 5.9%，带 bootstrap CIs。干净、可操作、对社区有用。

- **跨层吸收表征**: 15 倍变化，无探针质量混杂 (F1>=0.97)。论文最强单一结果。

- **Bootstrap CIs 贯穿全文**: 10k resamples, seed=42。最佳实践。

- **统计严谨性**: Cohen's d, Wilcoxon, Mann-Whitney, 排列检验, 多重比较意识。方法论始终健全。

- **基础设施可靠性**: 连续 5 轮零实验失败。

---

## 下一迭代最高优先级（Gate 结构）

### Gate 0 -- 立即修复 + 基础设施（零 GPU, ~3 小时）
1. 修复 12.3% 虚假数字 [5 min]
2. 修复损坏的交叉引用 [10 min]
3. 实现 validate_integration.py 并在当前论文上运行 [1.5h]
4. 修复探针 F1 差异 [15 min]
5. 调查字母 G hedging 主导地位 [30 min]
6. 计算实例级吸收率 [30 min]

### Gate 1 -- 关键实验（2-4 GPU-hours, BLOCKING）
7. 降级探针消融实验 F1={0.70, 0.80, 0.85, 0.90} [1-2 GPU-hours]
8. L24 激活补丁 [1-2 GPU-hours]

### Gate 2 -- 图表和写作（零 GPU, ~4 小时）
9. 生成图 5（激活补丁点图）和图 6（hedging 堆叠柱状图）[1h]
10. 渲染图 2（管道示意图）[30 min]
11. 写附录章节 (GAS, CMI, Tax, 阈值敏感性) [2h]
12. 重构摘要 (<220 词) + 修改标题 [30 min]
13. 降级架构比较章节 + 删除层级 vs 架构声称 [15 min]
14. 提升 restricted activation patching analysis [15 min]
15. 限定跨域声称为 L24-only [15 min]

### Gate 3 -- 审阅（Gates 0-2 全部完成后）

**预期分数轨迹**:
- Gate 0 完成: 7.25（消除虚假数字、修复引用）
- Gate 0+1（降级探针确认层级效应）: 7.5-8.0
- Gate 0+1（降级探针显示探针伪影）: 7.5（重构为层依赖 + 因果 + hedging）
- 全部完成: 8.0（Accept）
