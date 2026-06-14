# Iteration 5 Reflection Report

**项目**: SAE Feature Absorption (sae-absorption)
**迭代**: 5
**日期**: 2026-04-15
**综合评分**: 6.0/10 (Supervisor Review)
**维度评分**: Novelty 7, Soundness 5, Experiments 6, Reproducibility 5
**Writing Review 评分**: 7/10
**裁定**: REVISE
**质量轨迹**: 改善中 (5.5 x3 -> 6.5 -> 6.0, 但 6.0 来自显著更严格的审查标准)

---

## 一、迭代摘要

迭代 5 完成了一次完整的四阶段实验流水线，是项目历史上方法论最严谨的一轮：

1. **Phase 1 -- L0 混杂因素解析** (48 SAEs, Gemma 2 2B): 四方法收敛设计（偏相关、分层分析、Baron-Kenny 中介、Rosenbaum 敏感性分析）首次在 SAE 评估中应用流行病学因果推断方法
2. **Phase 2 -- 跨领域吸收测量** (GPT-2 Small, 3552 城市): 首次在知识层级上测量 feature absorption
3. **Phase 3 -- 吸收缩放曲面** (420 SAEs, SAEBench): 迄今 SAE 文献中最大规模的 absorption 2D 表征
4. **Phase 4 -- 分类法校正** (26 字母, WikiText-103): 审计 Chanin et al. 原始 92.3% 综合吸收率

**执行统计**: 14/14 任务完成，零失败。总实际 GPU 时间约 9.5 分钟，总实际 CPU 时间约 8 分钟。基础设施完全可靠。

### 核心发现

| 假设 | 裁定 | 关键数字 | 评估 |
|------|------|----------|------|
| H1: 吸收因果链 | SUPPORTED_WITH_CAVEATS | SP-F1 partial r = -0.746 (p=1.2e-9); SCR Sobel z=3.62 (p=2.9e-4) | 统计证据强，但因果声明过强 |
| H2: 跨领域吸收 | **FALSIFIED** (实际) / PARTIALLY_SUPPORTED (错误标签) | 0% cosine-calibrated; 100% shuffled control | 指标在 GPT-2 Small 上无效 |
| H3: 缩放曲面 | STRONGLY_SUPPORTED | GAM interaction p=3.1e-15; R^2=0.693; N=420 | 最强结果，但缺少架构协变量 |
| H5: 分类法校正 | CORRECTION_LARGE (实际) / CORRECTION_MINIMAL (错误标签) | 92.3% -> 19.2% (delta=-73.1 pp) | 重要审计发现，但 JSON 报告了错误值 |

### 关键进展 (相对 iter_004)

- 从失败的 LV 竞争排斥框架成功转型为流行病学因果推断方法
- **所有实验在 FULL 模式运行** -- 解决了 iter_001-003 的 3 轮系统性 PILOT 模式阻塞
- 混杂因素控制现在是预注册实验计划的组成部分（四方法收敛设计）
- 7 张实验图已生成（对比 iter_001-003 的零张图）
- 论文正文数值与源 JSON 一致（12 项交叉验证全部通过）

### 关键退步

- 名义评分从 6.5 降至 6.0，反映 Supervisor 采用了空前细致的逐数字交叉验证标准
- final_results.json 出现 4+ 个数据完整性错误（新型失败模式）
- 因果过度声明被首次系统性识别和诊断
- Table 3 系数混淆是新引入的写作缺陷

---

## 二、问题按类别分析

### SOUNDNESS（可靠性）-- 5 个 critical/high 问题

**[CRITICAL] SND001: 因果过度声明**
论文使用 "causally mediates"、"causal chain"、"causal status of absorption" 描述 Baron-Kenny 中介分析结果。证据来自 48 个横截面观测值，Baron-Kenny 方法不建立因果关系。致命性证据：论文自身 within-width matching（4 对精确匹配、23 对中位数分割、16 对三分位匹配）全部 Gamma=1.0，因果证据在控制宽度后完全消失。替代解释（SAE 特征质量同时驱动低吸收和高质量）与所有数据一致。

Supervisor 明确判定：此问题足以独立导致 NeurIPS 拒稿，尤其是被分配到因果推断领域的审稿人。修复路径：纯语言降级 + 结构重组，零 GPU。

**[CRITICAL] SND002: Table 3 系数混淆**
路径 a, b 标记 "(std)" 匹配标准化值。但 c' 在 Table 3 中为 -0.029（标准化），摘要中为 -0.003（非标准化）。间接效应 ab=0.025 匹配非标准化值而非标准化（0.2478）。两种类型在同一表格中混合且无标注。任何检查 a*b=ab 算术的审稿人都会立即发现问题。

来源验证：P1_mediation.json unstandardized direct SCR = -0.00290 (rounds to -0.003), standardized = -0.0291 (rounds to -0.029)。Writing review 独立确认了此不一致。

**[HIGH] SND003: SP-F1 抑制效应解释脆弱**
SP-F1 总效应 c 不显著（p=0.45），Baron-Kenny 第 3 步失败。按 Zhao et al. (2010) 分类，这是 "indirect-only mediation"，弱于经典中介。竞争解释（L0 通过更好重建直接改善 SP-F1，部分抵消通过 absorption 的间接负面效应）未得到充分讨论。论文将此标记为 "central finding"，但 SCR 的完全中介（所有 Baron-Kenny 步骤满足）是更可靠的主要发现。

**[HIGH] SND004: 多重比较未校正**
50+ 个检验：4 指标 x (偏相关 + 分层 3 层 + 中介 + Rosenbaum 5 策略)。alpha=0.05 下预期 2-3 个假阳性。TPP 偏相关 (p=0.022) 和 TPP Sobel z (p=0.037) 处于边界，FDR 校正后可能失去显著性。SP-F1 (p=1.2e-9) 和 SCR (p=6.6e-5) 在任何合理校正下均稳健。

**[HIGH] SND005: Rosenbaum Gamma 跨宽度膨胀**
Gamma=2.65（Mahalanobis 匹配，17 对）vs Gamma=1.0（所有 within-width 策略）。Mahalanobis 匹配允许跨宽度配对；由于 absorption 与 width 强混淆，跨宽度配对可能放大表面鲁棒性。论文报告了此结果（Table 4）但将 Gamma=2.65 作为标题鲁棒性证据，未充分强调 within-width 证据为 null。

### EXPERIMENT（实验）-- 4 个 critical/high 问题

**[CRITICAL] EXP001: H2 分类错误**
H2 按预注册证伪标准已被证伪：shuffled baseline = 100%，3x shuffled baseline 不可能满足。cosine-calibrated 检测 0%。dominance-based 的吸收率等于 false-negative rate（Table 5 中 R_abs = FN Rate 逐行一致）。但 abstract、final_results.json、final_results_summary.md 均标记为 "PARTIALLY_SUPPORTED"。论文正文 Section 4.2 正确描述了结果，形成内部矛盾。

**[CRITICAL] EXP002: 数据管道完整性失败**
final_results.json 中至少 4 个字段错误：
1. H5 corrected_rate = 0.923 (实际 0.192), delta = 0.0 (实际 -0.731), verdict = CORRECTION_MINIMAL (应为 CORRECTION_LARGE)
2. 全部 4 个 Sobel z 值来自 alternative_direction_test（反向中介路径 Quality->Absorption->L0），而非主要中介路径：TPP z=2.63 (应为 2.08), SCR z=3.89 (应为 3.62), SP-F1 z=4.33 (应为 4.08)
3. n_full_mediations = 0 (应为 2: SCR + TPP)
4. phase_boundary_detected = false (P3_scaling_surface.json 源数据为 true)

论文正文使用了正确数值。但 final_results.json 是 "canonical machine-readable artifact"，下游自动化（reflection、evolution、Lark sync）均消费错误数据。根因：集成代码从错误 JSON 路径读取。

**[HIGH] EXP003: Phase 2 模型不足**
GPT-2 Small (124M) + 24k SAE，98% 特征在城市提示上为死亡特征。~500 个活跃特征对应 3,552 城市 x 5 属性类型 -- 字典容量灾难性不足。0% cosine-calibrated 是模型选择的必然结果，非关于 absorption 的证据。原始设计为 Gemma 2 2B + Gemma Scope SAEs (16k-1M 特征)。摘要和 Contribution 2 仍暗示实验回答了跨领域吸收问题。

**[HIGH] EXP004: Phase 3 GAM 架构混杂**
420 SAEs = 360 标准 L1/TopK + 54 JumpReLU + 6 未分类。JumpReLU 使用不同稀疏机制（直通估计器训练的 L0），集中在特定 width/L0 组合。GAM 包含 layer 但不包含 architecture。交互项可能部分捕获架构差异而非真正的 width-L0 交互物理。

验证方案：添加 architecture_class 作为 GAM 因子协变量。在 360 个标准 SAEs 上独立运行交互 GAM（n=360 仍有充足统计功效）。如果交互项在单一架构内仍显著，发现稳健；否则 Contribution 3 大幅削弱。30 分钟零 GPU 分析。

### WRITING（写作）-- 4 个 high/medium 问题

**[HIGH] WRI001: 图编号跳跃和缺失视觉支持**
图 3 直接跳到图 5（无图 4）。Section 4.3（梯度分析/相边界检测）和 Section 4.4（分类法校正）没有任何图表。Section 4.3 描述了 "443 points with gradient magnitude exceeding the 70th percentile threshold (0.69)"，全以文字叙述无可视化 -- 对以"缩放曲面"为贡献的结果部分不可接受。7 张图已生成但仅 4 张引用。"Figures and Tables" 部分列出原始文件名 (fig1_partial_correlations.pdf) -- 生产伪影。

**[MEDIUM] WRI002: 摘要过密**
230 词中嵌入 7 个定量值（r, p, c', z, Gamma, R^2, pp）。读者必须处理 5 个独立统计声明后才到达总结句。结构埋葬了主旨。

**[MEDIUM] WRI003: Bradford Hill 表应在补充材料**
Section 5.1 ~800 词，包含 9 行 Bradford Hill 准则表（Table 7）。Bradford Hill 是非正式启发式，非严格因果框架。其在正文中的展开可能激怒熟悉因果推断的审稿人。

**[LOW] WRI004: "Super-absorber" 未定义**
术语出现在 Section 4.2、5.2 和 6 中，无正式定义，不在 glossary 中。Feature 8213 是示例但术语泛化使用。

### ANALYSIS（分析）-- 3 个 medium 问题

**[MEDIUM] ANL001: 分类法双率混淆**
corrected Type II comprehensive rate = 19.2% vs Chanin false-negative rate = 73.1%。54 百分点差距意味着两个操作定义在大多数字母上对 "absorption" 的判定不一致。论文两者并列报告但未充分调和。Conclusion 同时说 "the original 92.3% comprehensive rate is an artifact" 和 "the Chanin false-negative metric independently validates 73.1% absorption" -- 将两个深度矛盾的数字呈现为互补。

**[MEDIUM] ANL002: 反向中介更强但未报告**
Quality->Absorption->L0 路径的 Sobel z 更强（TPP: 2.63 vs 2.08, SCR: 3.89 vs 3.62, SP-F1: 4.33 vs 4.08）。虽然时序排序支持正向路径（L0 是训练超参数），反向路径的更强统计量至少应在 Discussion 或补充材料中报告以维护透明性。注意：final_results.json 意外报告了反向值而非正向值，这本身暴露了集成代码缺陷。

**[MEDIUM] ANL003: Rosenbaum 匹配策略差异未充分问题化**
跨宽度 Gamma=2.65 vs within-width Gamma=1.0 的差异揭示了核心张力：absorption 的质量效应完全来自跨宽度比较。这不是一个可以一句话带过的 caveat，而是一个需要正面对待的核心限制。

### PLANNING（规划）-- 3 个 medium 问题

**[MEDIUM] PLAN001: 缺乏 within-width 功效分析**
n=15-18 per stratum，检测 rho=0.3 的功效仅 24-45%。within-width null 可能是功效不足而非效应不存在。需要最低样本量分析（80% power at alpha=0.05 检测 rho=0.3 需要 n~84）。

**[MEDIUM] PLAN002: 备选模型未预验证**
GPT-2 Small 作为 Gemma 2B 备选，未经 5 分钟预检（检查活跃特征比例和余弦相似度）就直接投入 Phase 2 完整实验。一次 5 分钟 pilot 即可揭示 98% dead feature 问题，避免浪费 Phase 2 全部实验时间。

**[MEDIUM] PLAN003: 缺乏集成交叉检查**
无自动化验证步骤。final_results.json 错误未被发现直到 Supervisor/Critic 手动逐数字核查。一个简单的 diff 脚本比较 final_results.json 与源 JSON 可在零成本下防止此类错误。

### PIPELINE（流水线）-- 1 个 medium 问题

**[MEDIUM] PIPE001: 集成脚本从错误 JSON 路径读取**
alternative_direction_test（反向中介）而非 primary sobel_test（正向中介）。这是 EXP002 的根因。

---

## 三、Fix Tracking（修复追踪）

### 已修复 (来自 iter_004 action_plan)

| iter_004 ID | 描述 | 状态 | 验证证据 |
|---|---|---|---|
| EXP001 | H3 width-L0 confound 未控制 L0 | **FIXED** | iter_005 直接添加 L0 协变量；3/4 指标保持 \|partial_r\| > 0.2 |
| EXP002 | 92.3% taxonomy rate inflated | **FIXED** | Phase 4 校正至 19.2%（论文正文） |
| EXP006 | PMI regression without clustered SE | **FIXED** | 26-cluster SE 使 L0 从 p=0.012 降至 p=0.206 |
| WRI001 | 'Validated quality indicator' overclaim | **FIXED** | 替换为 'quality-correlated'，正文无 'validated' 残留 |
| WRI003 | 图表在写作阶段缺失 | **FIXED** | 7 张图生成（4 张引用，3 张待引用） |
| PLAN001 | 混淆控制未预指定 | **FIXED** | 四方法收敛设计在实验计划中预注册 |
| EXP005 | 跨模型缺口 | **PARTIALLY FIXED** | 设计中明确分离模型/阶段；但 GPT-2 Small 备选方案削弱 Phase 2 |
| PILOT-TO-FULL | 所有实验困于 PILOT 模式 | **FIXED** | iter_005 全部 14 项 FULL 模式，零 PILOT 残留 |
| WRI002 | 标题暗示失败框架有效 | **PARTIALLY FIXED** | 标题不再引用 LV 框架；但仍可进一步优化 |

### 循环问题

| 问题 | 迭代历史 | 状态 |
|---|---|---|
| 图表集成滞后于生成 | iter_001: 0张; iter_004: 4/5引用; iter_005: 4/7引用 | **RECURRING** (改善趋势) |
| 数据管道完整性 | iter_001-003: PILOT模式; iter_005: JSON字段错误 | **RECURRING** (新形式) |
| 过度声明（framing bias） | iter_004: "validated quality indicator"; iter_005: "causal mediation" | **RECURRING** (升级) |

### 新问题（iter_005 首次出现）

共 12 个新问题：因果过度声明 (SND001)、Table 3 系数混淆 (SND002)、SP-F1 抑制效应解释脆弱 (SND003)、多重比较未校正 (SND004)、Rosenbaum Gamma 膨胀 (SND005)、H2 分类错误 (EXP001)、数据管道完整性失败 (EXP002)、Phase 2 模型不足 (EXP003)、GAM 架构混杂 (EXP004)、分类法双率混淆 (ANL001)、备选模型未预验证 (PLAN002)、集成交叉检查缺失 (PLAN003)。

---

## 四、资源效率分析

### GPU/CPU 利用率

| 任务 | 类型 | 计划(min) | 实际(min) | 效率比 |
|------|------|-----------|-----------|--------|
| P1_confound_go_nogo | CPU | 15 | 1 | 15x |
| P1_clustered_regression | CPU | 15 | 1 | 15x |
| P1_mediation | CPU | 25 | 1 | 25x |
| P1_width_stratified | CPU | 20 | 1 | 20x |
| P1_scr_suppression | CPU | 10 | <0.1 | >100x |
| P1_rosenbaum | CPU | 20 | <0.1 | >200x |
| P1_synthesis | CPU | 20 | 4 | 5x |
| P2_probe_training | GPU | 45 | 0.5 | 90x |
| P2_controls | GPU | 45 | 1 | 45x |
| P2_absorption_measurement | GPU | 60 | 1 | 60x |
| P2_crossdomain_comparison | CPU | 30 | <0.1 | >300x |
| P3_scaling_surface | CPU | 45 | <0.1 | >450x |
| P4_taxonomy_correction | GPU | 45 | 7 | 6.4x |
| final_integration | CPU | 30 | <0.1 | >300x |

**总计划时间**: ~425 分钟 (~7.1 小时)
**总实际时间**: ~17.7 分钟
**实际 GPU 使用**: ~9.5 分钟（P2_probe_training 0.5 + P2_controls 1 + P2_absorption 1 + P4_taxonomy 7）
**GPU 空闲时间**: ~15 分钟（任务间调度）
**GPU 利用率估算**: 85%（活跃 GPU 期间）

**关键洞察**: 本迭代展示了 "零 GPU 分析范式" 的成功。Phase 1（48 SAEs 偏相关/中介/Rosenbaum）和 Phase 3（420 SAEs GAM）完全在 CPU 上运行，基于预计算的 SAEBench 数据。最高影响力的发现（SP-F1 抑制效应、420-SAE 交互项）均来自 CPU 分析。GPU 仅用于 Phase 2 探针训练和 Phase 4 SAE 推理。

### 瓶颈分析

真正的瓶颈**不在实验阶段**，而在：
1. **写作修订** -- 系数混淆、因果语言、图编号等产生大量需要修复的问题
2. **数据集成** -- final_results.json 从错误路径读取，传播错误到下游
3. **审查发现-修复循环** -- Supervisor/Critic 在审查阶段发现实验阶段应预防的问题

### 调度优化建议

1. **集成验证脚本**: 在 final_integration 末尾添加自动化交叉检查，逐字段比较 final_results.json 与源 JSON。零成本防止 EXP002。
2. **架构分层 GAM**: 在 P3_scaling_surface 中额外运行 30 秒（添加 architecture_class 协变量），避免后续补做。
3. **FDR 校正**: 在 P1_synthesis 中自动运行 Benjamini-Hochberg，避免审查阶段发现缺失。
4. **图表自动引用**: 写作第一轮自动包含所有已生成图表，而非修订中补充。
5. **系数类型一致性检查**: 中介分析输出时同时生成 standardized 和 unstandardized 的完整表格，标注明确。

---

## 五、质量趋势评估

| 迭代 | 分数 | 关键变化 | 审查标准 |
|------|------|----------|----------|
| 0 (iter_001) | 5.5 | EDA 指标、RAVEL 探针错误、PILOT 模式 | 标准 |
| 1 (iter_002) | 5.5 | 数值不一致循环、Proposition 1 证伪 | 标准 |
| 2 (iter_003) | 5.5 | 停滞：PILOT 未解决、EncNorm 矛盾 | 标准 |
| 3 (iter_004) | 6.5 | **突破**：LV 框架 + FULL 模式 + 下游关联 | 标准+ |
| 4 (iter_005) | 6.0 | 流行病学方法 + 缩放曲面 + 分类法校正 | **严格** (逐数字核查) |

**轨迹判定: 改善中 (improving)**

理由：
1. 名义分数从 6.5 降至 6.0，但 Supervisor 明确表示这是审查标准升级的结果。iter_005 的 Supervisor 进行了逐数字交叉验证（Table 3 系数 vs 源 JSON、a*b=ab 算术检查、within-width null 因果含义诊断、final_results.json vs 源数据比对）。iter_004 的 Supervisor 也很严格但未达到此粒度。
2. 论文的实证基础比任何前一迭代更强：N=420 缩放曲面（文献中最大规模）、四方法收敛混杂因素分析（SAE 评估中首次）、预注册证伪标准、诚实负面结果。
3. 所有阻断性问题均为零 GPU 写作/分析修复。
4. Supervisor 提供了明确的到 7.0 (Weak Accept) 的 7 步路径，每步均具体可验证。
5. iter_001-003 的系统性阻塞（PILOT 模式、图表缺失、混杂因素未控制）全部已解决。

### 到 NeurIPS 投稿的路径

| 目标分数 | 所需行动 | 预计工时 | GPU 需求 |
|----------|----------|----------|----------|
| 7.0 (Weak Accept) | 修复 7 项 zero-GPU 问题 | 6-8 小时 | 零 |
| 7.5 (Borderline Accept) | 额外：功效分析、GAM 规格、竞争解释 | +2-3 小时 | 零 |
| 8.0 (Accept) | 额外：Gemma 2B Phase 2 或 OrtSAE 干预 | +4-8 小时 | 4-6 GPU 小时 |

---

## 六、成功模式提取

### 方法论创新（保持并加强）
流行病学因果推断方法（偏相关 + Baron-Kenny 中介 + Rosenbaum 敏感性分析 + Bradford Hill 准则）首次应用于 SAE 评估。这些方法在社会科学中是教科书内容，但在 SAE/mechanistic interpretability 文献中完全新颖。**这是论文的核心新颖性来源，必须继续坚持**。

### 四方法收敛设计（保持）
同一假设 (H1) 用 4 种独立方法从不同假设前提测试。跨方法收敛（SP-F1 和 SCR 在偏相关、中介、部分匹配策略中均显著）大幅增强结论可信度。下一迭代应继续采用。

### 预注册 + 诚实负面结果（连续 5 轮保持，核心竞争力）
所有假设均有预注册的量化成功/失败阈值。负面结果（within-width null、0% cosine-calibrated、unlearning null、模型退步）以具体数字和根因分析透明报告。这是论文在所有审查中（Supervisor、Critic、Writing Review）**始终获得最高评价的方面**。绝不能妥协。

### 零 GPU 分析范式（高效）
Phase 1 和 Phase 3 完全在 CPU 上运行，基于 SAEBench 预计算数据和 iter_004 元数据。最高影响力/成本比。证明了严谨统计分析不需要额外 GPU 计算。

### 声明-数据一致性（论文正文）
尽管 final_results.json 有错误，paper.md 中 12 项交叉验证全部匹配源数据。source-first 写作规范有效。

### 零实验失败（连续 2 轮）
14/14 任务成功（iter_005），13/13 任务成功（iter_004）。基础设施完全可靠。

---

## 七、根因分析

### 为什么 Supervisor 评分从 6.5 降至 6.0？

**审查标准显著升级**：iter_005 的 Supervisor 进行了系统性交叉验证：
- 逐条对比 Table 3 系数与 P1_mediation.json 源数据
- 检查 a*b = ab 算术一致性
- 诊断 within-width null 的因果含义
- 比对 final_results.json 全部字段与源 JSON
- 独立验证 H2 是否满足预注册证伪标准

这揭示了三类在 iter_004 标准下不会被发现的问题：系数类型混合（需要查看源 JSON 的两个不同字段）、数据管道从错误路径读取（需要比对嵌套 JSON 结构）、因果声明与方法论证据不匹配（需要理解 Baron-Kenny 因果推断的哲学限制）。

### 为什么出现新型错误？

**复杂性升级带来新失败模式**：
1. 中介分析同时生成标准化和非标准化系数 -- 需要显式选择和标注，之前的简单相关分析无此问题
2. 中介分析同时计算正向和反向路径 -- 集成代码必须明确选择正确路径
3. Phase 4 分类法校正产生了与原始率截然不同的结果 -- 集成代码未更新字段

**根因：缺乏自动化质量门控**。人工审查发现了问题，但应该由自动化脚本在集成阶段即时捕获。

### 因果过度声明的系统性根源

iter_004 有 "validated quality indicator" 过度声明（语言层面），iter_005 有 "causal mediation" 过度声明（方法论层面）。共同根因：**强化叙事的激励导致系统性偏向过度声明**。"Absorption causally mediates quality" 比 "Absorption statistically correlates with quality after controlling for L0" 更有发表力度。需要在写作 agent 的 prompt 中显式加入反过度声明检查。

---

## 八、系统自检响应

未发现 `logs/self_check_diagnostics.json` 文件。无系统自检诊断需要响应。

---

## 九、下一迭代建议

### 立即行动（零 GPU，预计 6-8 小时）

1. **[BLOCKING]** 修复 final_results.json 数据完整性（1h）
2. **[BLOCKING]** 降级因果语言 + 提升 within-width null 至 Results 显著位置（2h）
3. **[BLOCKING]** 修复 Table 3 系数统一性 + 算术一致性验证（1h）
4. **[BLOCKING]** H2 -> FALSIFIED，重构为指标有效性发现（1h）
5. **[HIGH]** 架构分层 GAM + 360 标准 SAE 子集分析（30min 分析 + 30min 写作）
6. **[HIGH]** FDR 校正（Benjamini-Hochberg）+ 报告校正前后 p 值（30min 分析 + 30min 写作）
7. **[HIGH]** 修复图编号 + 添加梯度曲面图和分类法校正图（1h）
8. **[HIGH]** 扩展抑制效应讨论 + 提升 SCR 完全中介为主要发现（1h）

### 补充分析（零 GPU，2-3 小时）

9. 功效分析：within-width 检测 rho=0.3 需要 n~84，当前 n=15-18 功效仅 24-45%（30min）
10. 调和分类法双率 19.2% vs 73.1%：magnitude-ratio vs false-negative（30min）
11. 报告反向中介 Sobel z + L0 时序排序论证（30min）
12. GAM 规格细节：spline 基函数、knots、penalty、link function（15min）
13. 定义 "super-absorber" + 添加 glossary 条目（10min）
14. "introduced here" -> "applied here for the first time to SAE evaluation"（5min）
15. 移除 "Figures and Tables" 伪影部分 + 移动 Bradford Hill 表到补充（15min）

### 未来方向（需要 GPU）

- **最高优先级**: Gemma 2 2B 上重复 Phase 2（需模型访问 + 4-6 GPU 小时）
- **可选**: OrtSAE 干预研究（建立因果关系的唯一路径，需重新训练 SAE）
- **可选**: Beta-GAM 敏感性检查（验证 Gaussian GAM 在 [0,1] 有界响应上的稳健性）

### 系统改进（防止循环问题）

- 在 final_integration 任务末尾添加自动化交叉检查脚本
- 在写作 agent prompt 中添加反过度声明检查（"any causal language must cite interventional evidence"）
- 在中介分析输出中强制同时生成双系数表（standardized + unstandardized）并明确标注
