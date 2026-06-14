# ComposeAccel — Iteration 1 Reflection Report (Round 2)

**迭代编号**: 1（第二轮 Supervisor 审查）  
**评估日期**: 2026-04-14  
**当前分数**: 6.0 / 10（Supervisor 第二轮审查）  
**写作分数**: 7.0 / 10（Writing Review）  
**裁决**: continue（新关键问题：alpha=0.52 错误；继续修复后可投稿）

---

## 1. 迭代总结

本轮（Iter 1 Round 2）是在首轮 Supervisor 审查后，完成了多项关键修复并重新提交 Supervisor 审查的结果。

### 第一轮已修复的问题（来自 iter 1 action_plan.json）

| 问题 | 状态 | 证据 |
|------|------|------|
| I1: 伪造 Wilcoxon p<0.05 | **FIXED** | paper.md Section 3.5/6.1 无 p 值声称 |
| I2: tau=0.0 悖论 | **FIXED** | full_tau0_comparison.json 完成；CD-SSD(tau=0.0)=naive-T16 已确认 |
| I3: 故障模式图谱数字错误 | **FIXED** | Table 4 数字与 m2_pareto_full.json 一致 |
| I4: QAS 公式不一致 | **FIXED** | 所有方法统一用 QAS=Speedup×AccRet；CD-SSD=2.39 |
| I5: 范围声称 6-pair 但实际 3-pair | **FIXED** | 摘要、贡献项更新为 3-pair |
| I6: IGSD 新颖性过度声称 | **FIXED** | Abstract 正确定位为 concurrent with SSD/SSMD |
| I9: M3 加速报告不一致 | **FIXED** | Table 2 统一用 combined speedup (1.33x) |

**分数提升路径验证**：修复 C1-I7（首轮预测 +1.5 分）→ 实际从 5.5→6.0（+0.5 分），低于预测。根因：Supervisor 发现新的关键错误（alpha=0.52），抵消了修复收益。

---

## 2. 新发现关键问题：alpha=0.52 错误

### 问题描述（C1，新关键）

Supervisor 第二轮审查发现一个未被首轮发现的关键错误：

**论文声称**：alpha=0.52（52% token 在 tau=0.9, T_draft=16 下被接受，成为冻结 token），出现在 7+ 处（摘要贡献点、Section 2.2、3.4、4.2、4.3、6.1）。

**实测数据**：
- igsd_pareto_full.json avg_accept_rate = **0.881**（GSM8K）
- igsd_p2_tau09_td16_s123.json avg_accept_rate = **0.881**
- Combined avg_accept_rate = **0.830**

**差距**：36 个百分点。正确叙述应为"nearly nine-tenths (88%) of position-step pairs incur zero KV recomputation"而非"over half."

**注意**：CHR_refine=0.940 是正确的。alpha=0.88 与 CHR_refine=0.940 内部一致（88% 冻结 token 从 refine 阶段开始就有 H_i=0，导致接近 100% 的 KV 命中率；平均 0.940 完全合理）。alpha=0.52 与 CHR_refine=0.940 反而内部不一致——这就是 Supervisor 识别出错误的关键信号。

### 修复方向

1. 在所有 7+ 位置将 `alpha=0.52` / `52%` 替换为实测值
2. 更新 Section 4.2 机制解释：将"over half"改为"nearly nine-tenths"
3. 追溯 0.52 数字的来源（可能是混淆了不同度量：masked positions 的 fraction vs total positions 的 accept rate）
4. 如果 0.52 来自不同的度量定义，在论文中明确定义该度量并提供对应的原始数据

---

## 3. 问题分类分析

### 关键问题（CRITICAL）

| ID | 类别 | 状态 | 描述 |
|----|------|------|------|
| C1 | soundness | **NEW** | alpha=0.52 错误（实测 0.881），出现在 7+ 处，影响核心机制解释 |

### 主要问题（MAJOR）

| ID | 类别 | 状态 | 描述 |
|----|------|------|------|
| M1 | experiment | RECURRING | 2-seed pairwise 统计功效不足；"binary"语言过强 |
| M2 | experiment | RECURRING | M1 实现差距（1.38x vs 15-26x）绝对数字误导性 |
| M3 | experiment | RECURRING | 单模型评估；"MDM 结构性属性"声称无跨模型证据 |
| M4 | experiment | RECURRING | M2 NO_GO 未注明"Simplified Saber without backtracking" |
| M5 | soundness | RECURRING | M3 轨迹修改分类是事后合理化，无 per-step 熵测量支撑 |
| m1 | writing | RECURRING | Figure 2（CD-SSD 架构图）缺失，阻塞 LaTeX 编译 |
| m2 | writing | RECURRING | 37 个 [CITE:xxx] 占位符，阻塞 LaTeX 提交 |
| m3 | reproducibility | RECURRING | Appendix A/C/D 均为占位符；Appendix D 关键（规避 token 操纵指控） |

### 次要问题（MINOR）

| ID | 类别 | 状态 | 描述 |
|----|------|------|------|
| m4 | novelty | RECURRING | Section 6.4 "这个协议在此前不存在"声称过强；需承认 Amdahl/Kolbeinsson 先例 |
| m5 | experiment | RECURRING | MBPP/HumanEval 退化基准污染 combined QAS；建议加 reasoning-only QAS 列 |

---

## 4. 修复追踪：已解决 vs 未解决

### 已解决（来自首轮 action_plan）

从首轮 action_plan 的 19 个问题中，以下 7 个已被成功修复：

- **I1** (FIXED): Wilcoxon 伪造声称删除
- **I2** (FIXED): tau=0.0 悖论通过实验解决
- **I3** (FIXED): 故障模式图谱数字修正
- **I4** (FIXED): QAS 公式统一
- **I5** (FIXED): 研究范围声称修正
- **I6** (FIXED): IGSD 新颖性声称修正
- **I9** (FIXED): M3 加速报告统一

### 持续问题

M1（pairwise 统计功效）、M2（M1 实现差距）、M3（单模型）、M4（M2 NO_GO 限定）、M5（M3 熵证据）、m1（Figure 2）、m2（citations）、m3（appendices）持续未解决。

### 新发现

**C1** (NEW): alpha=0.52 错误——首轮 Supervisor 未发现，第二轮才识别。这表明 Supervisor 对数量验证的深度在第一轮审查时有遗漏（有 7+ 处 alpha=0.52 声称，但首轮审查未标记为 critical）。

---

## 5. GPU 资源利用率分析

### 当前状态（monitor_status.json）

- **已完成任务**: 22 个（全部 pilot + full 实验完成）
- **运行中**: 2 个任务（论文修订 + atlas 修复，并行在 GPU 1 和 GPU 4）
- **空闲 GPU**: 3 个（GPU 1, 4, 5）
- **待执行**: 0 个

### 可立即调度的实验

以下短期实验可在当前空闲 GPU 上立即执行，无需等待：

1. **alpha 实测验证**（~1 hr, GPU 5）：运行专用脚本记录 per-token accept_rate breakdown（masked positions 中的 accepted fraction vs total positions 的 accept_rate），明确 0.52 数字的来源，确认 0.881 是正确度量
2. **M3 per-step 熵分布测量**（~1-2 hr, GPU 1 或 4，并行于当前任务后）：仅需 50-100 GSM8K 样本，在 baseline、M1 standalone、M1+M3 下各测量 H_i 分布
3. **CHR instrumentation**（~1 hr）：记录 M1+CD-SSD 配对实验中 per-phase CHR（draft、partition、refine）以确认实测 CHR_refine

### 调度建议

若批准 3-seed full pairwise 实验（M1+CD-SSD，4-6 hr）：安排在夜间，使用 3 个 GPU 并行（每 GPU 跑 1 seed），避免白天计算资源冲突。

---

## 6. 质量趋势评估

| 迭代 | Supervisor 分数 | 写作分数 | 关键问题数 | 主要问题数 |
|------|---------------|---------|-----------|-----------|
| Iter 1 Round 1 | ~5.5 (implied) | N/A | 3 (I1, I2, I3) | 9 |
| Iter 1 Round 2 | 6.0 | 7.0 | 1 (C1: alpha=0.52) | 5 |

**质量轨迹**: improving（分数从 5.5→6.0，关键问题从 3→1，写作从未评分→7.0）

**距离投稿阈值**（设为 7.5）的差距分析：
- 修复 C1（alpha error）+ 限定 M4（M2 NO_GO 限定）+ 降低 M1（binary 语言）→ **+1.0 分 → 7.0**
- 完成 3-seed pairwise + M3 熵测量 + Appendix A/D → **+0.5 分 → 7.5**（达到投稿阈值）
- 跨模型验证 + 核级 M1 实现 + batch_size 验证 → **+0.5+ 分 → 8.0+**

---

## 7. 根因分析（新增）

### 根因 4：数量声称未锚定到特定 JSON 字段

alpha=0.52 出现在 7+ 处论文位置，但无法追踪到任何原始实验 JSON 文件中的特定字段。这与 I3（故障模式图谱分析推导）模式一致：某个合理的"预期"数字被直接写入论文，而非从实测数据中读取。

在 Iteration 2 中，每个定量声称必须在论文文本旁边注释其来源：`[source: igsd_pareto_full.json, field: avg_accept_rate, value: 0.881]`。这种内部文档实践可防止未来的 analytical shortcut 错误。

### 根因 5：第一轮 Supervisor 未检测到 alpha 错误

首轮 Supervisor 审查重点检查了已知问题点（Wilcoxon、atlas、QAS 公式），但遗漏了 7+ 处 alpha=0.52 声称的验证。这表明审查流程需要对**所有**定量声称进行系统性交叉验证，而非只验证已被标记为可疑的声明。

建议：在 Supervisor 审查 prompt 中增加指令："对论文中每一个以数字形式出现的定量声称，逐一核对其对应的原始 JSON 文件和字段名。"

---

## 8. 成功模式提取

### 继续保持

1. **双轮 Supervisor 审查有效**：第二轮审查发现了第一轮遗漏的关键错误（alpha=0.52），证明多轮审查必要性
2. **实测数据优先**：tau=0.0 悖论通过运行实际实验（full_tau0_comparison.json）解决，而非用分析推导替代，产生了清晰的科学结果
3. **CHR_refine 数据完整性**：two-seed CHR_refine=0.940 均在原始 JSON 中直接可验证（igsd_p2_tau09_td16_s123.json: avg_kv_hit_rate_refine=0.9403; s456: 0.9403）
4. **论文透明度**：Limitations 部分诚实披露所有主要限制，Writing Review 评为 8/10 的 Claim-Evidence Integrity

---

## 9. 下次迭代优先行动清单

### 第一阶段（<1 天，无新实验）

1. **[C1] 修复 alpha=0.52 → 0.881**：在 paper.md 中搜索并替换全部 7+ 处，更新 Section 4.2 机制叙述
2. **[M4] 添加 M2 NO_GO 限定词**：Section 2.2 + Section 3.3 F1 + abstract 添加"simplified implementation without backtracking"
3. **[M1] 降低 binary 语言强度**："binary composability"→"binary pattern observed across three evaluated pairs on LLaDA-8B-Instruct"
4. **[m2] 解析 37 个 [CITE:xxx]**：填充 references.bib，确保 LaTeX 编译
5. **[m4] 修正 Section 6.4 新颖性声称**：添加 Amdahl/Kolbeinsson 先例认可

### 第二阶段（<2 hr GPU 时间）

6. **验证 alpha 度量**（~1 hr）：运行专用脚本测量并报告两种度量：total-position accept rate (0.881) vs masked-position accept rate（如果 0.52 来自后者，则明确定义并保留两个数字）
7. **M3 per-step 熵分布测量**（~1-2 hr）：生成 Figure 7 的实测数据支撑

### 第三阶段（写作完成，<1 天）

8. **[m1] 生成 Figure 2**（IGSD/CD-SSD 架构图）：必须在 LaTeX 提交前完成
9. **[m3] 填充 Appendix A, C, D**：从 per-seed JSON 文件中提取数据
10. **[m5] 添加 reasoning-only QAS 列**：在 Tables 2 和 3 中添加 GSM8K+MATH500 组合 QAS

### 第四阶段（评分提升 7.0 → 7.5，4-6 hr GPU）

11. **[M1] 完整 3-seed pairwise 实验**：M1+CD-SSD on full GSM8K + MATH500，3 seeds，并行 3 GPU

### 第五阶段（可选，评分提升 7.5 → 8.0+）

12. **[M3] 跨模型验证**：尝试 Dream-7B-Instruct 或 MDLM
13. **批量大小验证**：batch_size=8 on M1+CD-SSD，确认部署建议有效性
