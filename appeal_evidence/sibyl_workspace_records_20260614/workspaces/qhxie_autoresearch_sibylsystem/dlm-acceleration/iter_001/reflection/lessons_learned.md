# Lessons from Iteration 1 (Round 2 Update)

## Must Improve（下次迭代必须改正）

- **[SOUNDNESS] 每个定量声称必须锚定到具体的原始 JSON 文件和字段名**：alpha=0.52 在 7+ 处论文位置出现，但无法追踪到任何 JSON 文件中的字段（实测值为 0.881，来自 igsd_pareto_full.json avg_accept_rate）。规则：写入论文的每个数字必须能在原始实验 artifact 中找到对应字段。建议在论文写作草稿中保留"[source: filename.json, field: field_name, value: X]"内部注释，在最终版本前删除。

- **[EXPERIMENT] 任何定量声明必须有实测证据，禁止 analytical derivation 替代**：（保留首轮教训）本次迭代故障模式图谱（failure_mode_atlas）、Wilcoxon 检验三处均使用分析推导替代实测，导致多个 critical integrity 问题。虽已修复，但 alpha=0.52 错误属于同类：写入了"预期"数字而非实测数字。规则：实验结果 JSON 的 `elapsed_minutes=0.0` 是危险信号。

- **[EXPERIMENT] QAS 或任何评估公式改变时，必须同步更新论文方法论章节**：（保留首轮教训，已修复）IGSD 使用了 0.5x 惩罚但未写入 Section 2.1，导致跨方法比较无效。已通过统一公式修复。下次任何公式变更必须立即更新对应论文章节。

- **[WRITING] 研究范围变更（NO_GO/PIVOT）必须立即传播到摘要和贡献项**：（保留首轮教训，已修复）M2 NO_GO 后研究缩减到 3-pair，但摘要仍声称 6-pair。已修复。规则：每次范围改变后，abstract + contribution list 必须立即同步审查。

- **[EFFICIENCY] 参数扫描的时间估算必须基于配置网格大小，而非简单的 pilot 外推**：（保留首轮教训）full_igsd_pareto 估计 90 min 但实际 160 min，因为 4 tau × 4 T_draft × 3 seeds = 48 配置。估算公式：`n_configs × n_seeds × (pilot_time_per_config / n_pilot_seeds)`。

## Watch Out（下次迭代需警惕）

- **Supervisor 多轮审查必要性**：第一轮 Supervisor 审查遗漏了 7+ 处 alpha=0.52 错误（直到第二轮才发现）。对于关键机制声称（frozen token fraction、CHR、synergy magnitude），Supervisor 需要逐行扫描论文中的每个数字并与原始 JSON 对比，不能只验证"已知可疑"的声明。

- **内部一致性作为错误检测信号**：alpha=0.52 与 CHR_refine=0.940 内部不一致（若只有 52% token 冻结，平均 CHR 不可能达到 94%；但若 88% token 冻结，94% CHR 完全合理）。下次写论文时，在发布前系统性检查所有配对声称的内部一致性。

- **"Binary composability" 语言强度与证据强度不匹配**：2-seed, 15%-scale 实验支撑 "binary pattern in three evaluated pairs" 而非 "binary composability is a structural property of MDM inference." 使用强度与证据匹配的语言：实验观察 → 观察级语言；理论推导 → 声明为理论。

- **M2 NO_GO 适用范围限定**：Simplified Saber（无 backtracking）的 NO_GO 结论不可直接推广到完整 Saber 算法。每次引用 M2 NO_GO 时必须附加限定词。

- **Batch size=1 限制影响部署建议可信度**：所有实验在 batch_size=1 运行，但生产 MDM 推理使用 batch_size=8-32。部署建议必须包含批量大小局限性声明或实测验证。

- **FastDLLM 等未复现方法不能与复现方法在同一表格中直接比较**：Table 1 混用不同硬件/协议的数字，需在 caption 中明确区分。

## Keep Doing（下次迭代保持）

- **双轮 Supervisor + Critic 审查**：第二轮审查发现了第一轮遗漏的关键错误，证明多轮独立审查是必要的。继续保持。

- **实测数据优先于分析推导**：tau=0.0 悖论通过运行实验（full_tau0_comparison.json）解决，产生了清晰的科学结果（CD-SSD(tau=0.0)=naive-T16，M1+naive-T16 7.40x 确认）。实验中发现的"负面"结果（partition 无价值）反而丰富了论文叙述。

- **JSON artifact 可交叉验证性**：CHR_refine=0.940 在两个 seed 文件中均可直接验证，full_pairwise_ortho.json per-seed Ortho [1.292, 1.478] 与论文一致。维护 JSON 作为机器可读权威数据源。

- **论文局限性部分的透明度**：Section 4.4 诚实披露 M1 实现差距、2-seed 规模、单模型评估，Writing Review 评分 8/10 claim-evidence integrity。这种透明度使 Reviewer 更容易接受论文的边界声明。

- **Pilot 实验体系有效**：6 个 Pilot 正确识别关键超参数和可行性。M2 早期 NO_GO 避免了无效计算。继续在每个新方法上先运行 Pilot。

- **部署配方的具体性**：三条部署规则（M1+CD-SSD 通用、M3 推理任务、never M2）基于数据，足够具体，对从业者有实际价值。继续提供基于数据的具体部署指导。

- **IGSD → CD-SSD 重命名完整执行**：paper.md 中零残留 "IGSD" 实例（Writing Review 验证），展示了命名规范执行彻底性。类似的术语一致性规范需延续到 Iter 2。
