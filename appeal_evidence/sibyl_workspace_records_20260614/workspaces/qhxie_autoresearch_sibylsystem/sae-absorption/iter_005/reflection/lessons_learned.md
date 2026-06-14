# Lessons from Iteration 5

**日期**: 2026-04-15 | **分数**: 6.0/10 | **轨迹**: 改善中（5.5 x3 -> 6.5 -> 6.0, 更严格审查标准）

---

## Must Improve（必须改进）

- **[SOUNDNESS -- BLOCKING] 因果语言必须与证据强度匹配**: Baron-Kenny 中介分析 + 横截面数据只能支持"统计中介"，不能支持"因果中介"。Within-width matching 全部 Gamma=1.0 直接削弱因果叙事。替代解释（SAE 特征质量同时驱动低吸收和高质量）与所有数据一致。所有 'causally mediates'、'causal chain'、'causal status' 替换为 'statistically mediates'、'statistical association'、'the statistical case for'。Within-width null 必须在 Results (Section 4.1) 中显著呈现，而非埋在 Discussion 末尾。在 Limitations 中明确命名 "feature quality" 替代解释。

- **[SOUNDNESS -- BLOCKING] 表格中系数类型必须统一且明确标注**: Table 3 混合了标准化和非标准化系数，摘要与表格使用不同系数类型（SCR c' = -0.003 摘要 vs -0.029 表格）。选择一种类型（推荐非标准化），在表头和脚注中明确标注，验证 ab = a*b 和 c - c' = ab 算术一致。任何审稿人检查这一算术都会发现错误。

- **[DATA INTEGRITY -- BLOCKING] final_results.json 必须有自动化交叉检查**: 当前集成代码从错误 JSON 路径读取 Sobel z（alternative_direction_test 反向路径而非 primary sobel_test 正向路径），分类法校正率使用了旧值（0.923 vs 实际 0.192），n_full_mediations=0（实际 2），phase_boundary_detected=false（实际 true）。下一迭代的 final_integration 任务末尾必须运行验证脚本，逐字段对比 final_results.json 与源 JSON，不一致时报错退出。

- **[EXPERIMENT -- BLOCKING] H2 必须重新分类为 FALSIFIED**: Shuffled control=100% 使预注册证伪标准（3x shuffled baseline）不可能满足。Cosine-calibrated 检测 0%。Dominance-based rate = false-negative rate（Table 5 逐行一致）。'PARTIALLY_SUPPORTED' 标签误导所有下游消费者。Phase 2 是合法的指标有效性发现，必须如实标记为 FALSIFIED on GPT-2 Small。

- **[EXPERIMENT -- HIGH] Phase 3 GAM 必须包含架构协变量**: 360 L1 vs 54 JumpReLU 跨不同 width/L0 范围，不同稀疏机制。30 分钟零 GPU 分析即可验证交互项是否在单一架构内仍然显著。如果不显著，Contribution 3 需要重新定性。

- **[ANALYSIS -- HIGH] 必须应用多重比较校正**: 50+ 个检验中 TPP 结果处于边界（偏相关 p=0.022, Sobel p=0.037）。Benjamini-Hochberg FDR 校正后可能失去显著性。必须报告校正前后 p 值，诚实承认哪些结果在严格校正下不成立。SP-F1 (p=1.2e-9) 和 SCR (p=6.6e-5) 在任何校正下安全。

---

## Watch Out（注意事项）

- **[ANALYSIS] SP-F1 抑制效应有竞争解释**: 总效应 c 不显著（p=0.45），Baron-Kenny 第 3 步失败。这是 Zhao et al. (2010) "indirect-only mediation"，弱于经典中介。竞争解释：L0 通过更好重建直接改善 SP-F1（direct effect c'=-0.270, p=0.001），部分抵消通过 absorption 的间接负面效应。不应将 SP-F1 抑制标记为 "central finding" -- **SCR 的完全中介（所有 Baron-Kenny 步骤满足, Sobel z=3.62, p=2.9e-4）是更可靠的主要发现**。

- **[ANALYSIS] Rosenbaum Gamma 跨宽度膨胀**: Gamma=2.65（Mahalanobis, 允许跨宽度配对）vs Gamma=1.0（所有 within-width 策略）。这不是一个 caveat，这是核心张力：absorption 的质量效应完全来自跨宽度比较。当 width 被设计控制（而非统计控制）时，效应消失。必须在讨论中正面对待，不能一句话带过。

- **[ANALYSIS] 分类法双率（19.2% vs 73.1%）必须调和**: magnitude-ratio（持续幅度抑制, 19.2%）和 false-negative（完全消失, 73.1%）测量不同现象。54 百分点差距揭示 GPT-2 Small 的吸收主要是完全抑制而非部分抑制。必须声明哪个是主要验证指标。不能同时呈现两者为"互补"且不做调和。

- **[ANALYSIS] 反向中介路径更强但未报告**: Quality->Absorption->L0 路径的 Sobel z 比正向路径更强（TPP: 2.63 vs 2.08, SCR: 3.89 vs 3.62）。虽然时序排序支持正向路径（L0 是训练前设定的超参数），但应报告反向统计量，并解释为何正向路径优先。透明性建立信任。

- **[PLANNING] 备选模型必须预验证**: GPT-2 Small 作为 Gemma 2B 备选未经 5 分钟预检就投入 Phase 2。检查活跃特征比例（>5%）和探针方向余弦相似度（>0.05）即可避免在注定失败的实验上浪费资源。

- **[PIPELINE] 集成脚本必须区分正向/反向中介路径**: EXP002 的根因是从 alternative_direction_test 而非 sobel_test 读取。当源 JSON 包含多个嵌套路径时，集成代码必须用路径常量而非模式匹配。

- **[WRITING] 摘要不能塞入 7 个定量值**: 重构为 "问题 -> 定性贡献 -> 关键数字(3-4个) -> 含义" 结构。先给读者 qualitative take-home，再给 quantitative evidence。

- **[WRITING] Bradford Hill 表应在补充材料**: 9 行表格在正文中占据过多空间，且 Bradford Hill 是非正式启发式。在因果过度声明背景下，其存在可能适得其反。

- **[WRITING] 图编号不能有跳跃**: 从图 3 直接跳到图 5 是不可接受的。已生成图必须在写作第一轮全部引用并按顺序编号。

---

## Keep Doing（保持的良好实践）

- **流行病学方法首次应用于 SAE 评估**: 偏相关 + Baron-Kenny 中介 + Rosenbaum 敏感性分析 + Bradford Hill 准则在 SAE/mechanistic interpretability 文献中是全新的。这是论文核心新颖性的源泉，下一迭代必须继续加强（修复因果语言后，方法论贡献本身是稳固的）。

- **四方法收敛设计**: 同一假设用 4 种独立方法从不同假设前提测试。SP-F1 和 SCR 在偏相关、中介和部分匹配策略中均显著 -- 跨方法收敛大幅增强结论可信度。必须继续。

- **预注册 Go/No-Go 标准**: |partial_r| > 0.2 阈值使 pilot 在 1 分钟内产生清晰 GO 决策。所有假设有预注册证伪标准。这是负面结果具有发表价值的基础。绝不能丢弃。

- **诚实负面结果报告（连续 5 轮）**: within-width null、0% cosine-calibrated 率、unlearning null、模型退步、GPT-2 Small 局限性 -- 全部以具体数字和根因分析透明报告。**这是论文在 Supervisor、Critic 和 Writing Review 三方审查中始终获得最高评价的方面**。绝不能妥协。如果下一迭代发现 architecture GAM 削弱交互项或 FDR 校正移除 TPP，必须同样诚实报告。

- **零 GPU 分析范式**: Phase 1（48 SAEs CPU）和 Phase 3（420 SAEs SAEBench 缓存）证明高质量分析不需要额外 GPU。最高影响力/成本比。继续优先利用预计算公共数据。

- **论文正文数值与源 JSON 一致**: 12 项抽查全部匹配源数据。尽管 final_results.json 有错误，paper.md 保持了数据诚信。source-first 写作规范（先查源 JSON，再写入论文）必须维持。

- **完整统计工具箱正确应用**: Bootstrap CI（10k 重采样）、VIF 诊断、聚类 SE、Hurdle 模型、Beta 回归、Mann-Whitney U、Kruskal-Wallis、DeLong test、Cohen's d -- 全部正确应用。下一迭代添加 FDR 校正和 power analysis 进一步完善。

- **零实验失败**: 14/14 任务成功完成（连续 2 轮零失败）。基础设施完全可靠。

---

## 下一迭代最高优先级（零 GPU，6-8 小时）

1. 修复 final_results.json + 添加验证脚本（1h）
2. 降级因果语言 + 提升 within-width null 位置 + 添加替代解释（2h）
3. 修复 Table 3 系数统一性 + 算术验证（1h）
4. H2 -> FALSIFIED + 重构为指标有效性发现（1h）
5. 架构分层 GAM + FDR 校正（1h）
6. 修复图编号 + 添加缺失图（1h）
7. 扩展抑制效应讨论 + 功效分析 + 双率调和（1h）

完成以上 7 项后，预计可达 **7.0 (Weak Accept)**。到 7.5 需额外添加反向中介报告、GAM 规格细节和竞争解释深化。到 8.0 需要 Gemma 2B Phase 2 复制或 OrtSAE 干预研究。
