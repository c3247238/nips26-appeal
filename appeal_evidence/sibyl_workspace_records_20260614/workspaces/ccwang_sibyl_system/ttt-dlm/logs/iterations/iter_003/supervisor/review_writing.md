# Supervisor 写作阶段终审报告（第五轮）

**角色**: 独立第三方高级研究监督
**审查对象**: 完整论文 (paper.md)、全部实验数据 (full_scale_summary.json, interim_countdown_results.json)、终审报告 (review.md)、批评反馈 (critique/)、视觉审计 (visual_audit.md)、实验分析 (experiment_analysis.md)、前四轮监督报告
**日期**: 2026-03-10（第五轮监督审查——论文重构后全面独立审查）

---

## 总体质量评估: 6.0/10

论文在第三轮迭代中进行了重大重构：标题改为 "Beyond Token Space: An Information Augmentation Spectrum for Masked Diffusion Language Models"，核心叙事从"DTA 方法论文"转向"信息增强谱系框架 + DMI 作为主要实证贡献"。所有 7 个方法的 Countdown-500 全规模数据现已完整（full_scale_summary.json）。这是论文质量的重要转折点。

**评分较前一轮（5.5/10）上调 0.5 分的原因**：
1. DTA 全规模数据现已完成（4.8% vs 4.7% vanilla），证实了 null result——论文能够诚实报告这一结果
2. SCP 全规模数据也已完成（9.1%），强化了谱系框架的完整性
3. 重构后的叙事更符合实际证据：DMI 和谱系框架作为主要贡献，DTA 作为理论框架的实证探索
4. DTA+ReMDM 的 degradation (3.6%) 被报告，尽管需要更深入的讨论
5. 论文的 critique → revision 循环已执行两轮（revision_round_1.marker, revision_round_2.marker），修正了多项问题

---

## 1. 实验数据完整性独立验证

### 1.1 全规模 Countdown-500 交叉验证

来源：`full_scale_summary.json` vs `interim_countdown_results.json` vs 论文 Table 1

| 方法 | full_scale_summary.json | interim (旧) | 论文 Table 1 | 一致性 |
|------|------------------------|-------------|-------------|--------|
| Vanilla | 4.7% | 4.7% | 4.7 ± 0.6 | **通过** |
| ReMDM-conf | 4.4% | 4.4% | 4.4 ± 1.0 | **通过** |
| RCR | 5.7% | 5.7% | 5.7 ± 0.6 | **通过** |
| DMI | 9.3% | 9.3% | 9.3 ± 1.4 | **通过** |
| SCP | 9.1% | ~8.4% (interim) | 9.1 ± 0.6 | **更新后通过** |
| DTA | 4.8% | pending | 4.8 ± 0.5 | **新数据，通过** |
| DTA+ReMDM | 3.6% | pending | 3.6 ± 1.2 | **新数据，通过** |

**SCP 全规模值（9.1%）与中期值（~8.4%）一致**：收敛到略高值是正常的，因为 interim 仅 150/500 样本。

**DTA 的 null result（4.8% vs 4.7%）**：+0.1pp 差异在 0.5% std 下不显著。论文在 Table 1、Abstract、Discussion 中均正确报告了这一结果。

**DTA+ReMDM 的 degradation（3.6%）**：低于 DTA（4.8%）和 ReMDM-conf（4.4%），且 seed 间方差最大（1.2%）。这直接矛盾了 Section 3.4 的"正交互补"声明。

### 1.2 GSM8K 部分数据验证

| 方法 | full_scale_summary.json | 论文 | 一致性 |
|------|------------------------|------|--------|
| Vanilla | 29.6% (1300/1319) | 29.6% | **通过** |
| DTA | 29.0% (900/1319) | — | **不在论文中** |
| DTA+ReMDM | 24.5% (400/1319) | — | **不在论文中** |
| ReMDM-conf | 21.3% (1300/1319) | — | **不在论文中** |

**关键发现**：full_scale_summary.json 包含 GSM8K 的部分全规模数据，显示 DTA 在 GSM8K 上也基本等于 vanilla（29.0% vs 29.6%），而 ReMDM-conf 大幅低于 vanilla（21.3% vs 29.6%）。这些数据在论文中未报告——论文 Table 2 仍使用 N=16 pilot 数据。**这是一个遗漏**：已有的部分全规模 GSM8K 数据应至少以脚注形式报告。

### 1.3 假设验证状态

| 假设 | 原始预期 | 全规模结果 | 状态 |
|------|---------|-----------|------|
| H1: DTA Countdown +5-10pp | +5-10pp | +0.1pp | **被证伪** |
| H2: DTA+ReMDM > ReMDM | +3-5pp | -2.8pp (GSM8K partial) | **被证伪** |
| H3: DTA 扩展曲线不饱和 | 对数增长 | N=16 pilot 不可结论 | **未验证** |
| H4: Level 3 > Level 2 > Level 1 > Level 0 | 严格递增 | DMI ≈ SCP > DTA ≈ Vanilla > DTA+ReMDM | **被证伪**（反转） |
| H5: DMI 改善预测质量 | +1pp+ | +4.6pp | **验证通过** |
| H6: SCP > ReMDM 精准度 | SCP > ReMDM | 76.9% vs 31.3% | **验证通过** |
| H9: ReMDM Correction Precision < 50% | < 50% | 31.3% | **验证通过** |
| H10: DTA 不引入退化 | rep-3 不增 | rep-3 不增 | **验证通过** |

**H4 的证伪是论文最大的惊喜发现**：信息增强谱系并非单调递增，而是呈现"浅层干预优于深层"的模式。论文已将此重新框架为核心贡献（"shallower interventions outperform deeper ones"），这是一个诚实且有洞察力的转变。

---

## 2. 致命问题（CRITICAL）

### 2.1 DTA 的标题地位与实证证据的根本矛盾

**状态**：论文标题已从 "Denoising-Time Adaptation" 更新为 "Beyond Token Space: An Information Augmentation Spectrum"——这是正确的方向。但论文内部仍存在残留矛盾：

- Abstract 第一句仍将 DTA 置于核心位置（"We introduce the Information Augmentation Spectrum, a four-level hierarchy..."），但 Level 3 (DTA) 是谱系中唯一失败的方法
- Contribution 1 仍标题为 "Denoising-Time Adaptation (DTA)"，将 DTA 列为第一贡献
- Contribution 2 是 "VDTA theoretical framework"——为一个 null result 方法提供理论保证的框架
- Section 3.4 仍声称 DTA 和 remasking "orthogonal and naturally complementary"，但 DTA+ReMDM (3.6%) 直接证伪了这一声明
- Section 3.4 最后一句推荐 "DTA+ReMDM-conf...for production use"——这对一个在所有全规模数据上表现最差的方法组合，构成严重误导

**修复方案**：
1. 将 Contribution 1 重新框架为 "Information Augmentation Spectrum"（实证框架），DMI 作为核心实证贡献
2. DTA 降级为 Contribution 3 的一部分（"comprehensive empirical analysis including the surprising null result..."）
3. 删除 Section 3.4 最后一句关于"recommended configuration for production use"的声明
4. 在 Section 3.4 添加全规模结果的讨论：DTA+ReMDM 在 Countdown 上 degradation 而非互补

### 2.2 VDTA 理论框架为无效方法提供理论保证

**状态**：Proposition 1 声称 ELBO 单调性，但 DTA 全规模 null result 意味着理论保证与实践脱节。

问题层次：
1. **强凸性假设不成立**：weight decay 加入非凸 neural network loss 不等于强凸。论文在 Section 6.5 承认了这一点，但 Abstract 和 Contribution 2 仍然突出 "ELBO monotonicity" 和 "theoretical guarantee"
2. **即使 ELBO 改善，不等于准确率改善**：ELBO 是生成模型的似然度代理，而非任务准确率。DTA 可能确实改善了 ELBO（无法直接验证），但生成质量不提升
3. **Proof sketch 不完整**：E-step "moves toward higher-likelihood regions" 未处理离散采样的不可微分性

**修复方案**：
1. 将 Proposition 1 降级为 "Conceptual Framework" 或 "Motivating Analysis"
2. 在 Abstract 中删除 "proving ELBO monotonicity"，替换为 "providing a conceptual framework"
3. 在 Section 3.2 开头明确声明：这是动机分析而非严格证明，全规模结果显示理论与实践之间存在 gap
4. 讨论 ELBO 改善 ≠ 准确率改善的原因（ELBO 是 token 级似然，不捕获 task-specific 约束）

### 2.3 缺失关键比较方法

**状态**：review.md 和前轮报告均指出，论文未与最直接的竞争方法比较。

- **MetaState (Xia et al., 2026)**：解决完全相同的 "Information Island" 问题，使用 GRU + CrossAttention。论文在 Section 2.1 和 2.2 中引用了 MetaState 但未提供性能比较
- **CORE (Zhai et al., 2026)**：在 MBPP 上用 LLaDA-8B 实现 +9.2pp。论文在 Section 2.2 引用但未比较

**风险**：任何审稿人都会问 "为什么不与 MetaState 和 CORE 比较"。即使 MetaState 需要训练（不公平比较），至少应讨论性能差距和 training-free 约束下的 tradeoff。

---

## 3. 重大问题（HIGH）

### 3.1 Abstract 仍然 oversell DTA

Abstract 最后一句："DTA achieves only 4.8\% ($\approx$ vanilla)" 是诚实的，但前文 Contribution 1 中包含 "pilot results show DTA's strongest gains on code generation (MBPP: +12.5pp)"——在论文自身已记录 pilot 平均膨胀 24pp 的背景下，将一个 N=16 pilot 结果放在 Abstract 的 Contribution 列表中是不当的。

**更严重的问题**：full_scale_summary.json 中的 GSM8K partial 数据显示 DTA 在 GSM8K 上也是 null result（29.0% vs 29.6%）。如果这些数据被纳入论文，DTA 在两个有全规模数据的 benchmark 上均为 null result，进一步削弱 MBPP pilot 声明的可信度。

**修复**：从 Abstract 的 Contribution 列表中移除 MBPP pilot result，或至少用 "preliminary pilot-scale" 明确标记。

### 3.2 DTA+ReMDM degradation 的根因未分析

DTA+ReMDM 在 Countdown 上 3.6% —— 低于 DTA (4.8%)、ReMDM-conf (4.4%) 和 vanilla (4.7%)。seed 间方差极大（2.4%, 3.6%, 4.8%）。

Section 3.4 声称 DTA 和 remasking "operate in fundamentally different spaces" 因此 "naturally complementary"。这一声明被全规模数据直接证伪。论文在 Discussion 中需要：
1. 承认组合方法失败
2. 提供可能的解释（remasking 的 token churning 破坏 DTA 的 MLM 训练信号？DTA 的参数漂移在 remasking 引入的不稳定性下放大？）
3. 删除或大幅修改 Section 3.4 的"互补性"叙事

### 3.3 统计检验仍未报告

论文在 Section 4.1 承诺 "McNemar's test with Bootstrap 95\% confidence intervals and Bonferroni correction"，但 Table 1 和所有结果段落仅报告 mean ± std。

- Abstract 声称 DMI "$p < 0.05$" 但未在 Results 中展示检验过程
- 7 个方法 × 3 pairwise comparison = 至少 6 个 McNemar tests 需要报告
- Bonferroni 校正后 DMI 是否仍然显著？6 个比较下 α = 0.05/6 ≈ 0.008

**这是承诺与交付的脱节**。论文方法部分承诺了严格的统计检验框架，但结果部分完全未执行。

### 3.4 4 个 TODO 图表 + 1 个 TODO 表格

根据 visual_audit.md：
- Figure 3 (Token-Level Diagnostics): **未生成**
- Figure 4 (LoRA Norm Trajectories): **未生成**
- Figure 5 (Scaling Curves): **未生成**（数据不足）
- Figure 1 (DTA Overview): 描述就绪但**未生成**
- Figure 2 (Spectrum Overview): 描述就绪但**未生成**
- Table 5 (Full Ablation): **未创建**

一篇实验驱动的论文仅有 1 个已生成的 Figure（task_sensitivity.pdf）和 5 个 inline tables。**这在任何顶会审稿中都会被认为是不完整的提交**。

### 3.5 论文篇幅仍严重超限

paper.md 约 63,656 字节（含 LaTeX 标记），估计正文约 12,000-15,000 词。NeurIPS 正文限制约 6,000 词（不含参考文献和附录）。论文需要压缩 50% 以上。

### 3.6 前四轮报告中的 PPL 自矛盾问题

这一问题在当前版本中已不再适用——论文已完全重构为准确率评估框架，PPL 不再是核心指标。**前四轮的 CRITICAL 2.1 已解决**。但 Section 6.4 Lesson 1 中关于 "Perplexity is unreliable" 的讨论仍引用了与当前论文无直接关系的 Phase 1 PPL 实验——这部分内容需要简化或移至附录。

---

## 4. 中等问题（MEDIUM）

### 4.1 Countdown 基线与文献报告值的差距

论文报告 Vanilla Dream-7B Countdown 准确率为 4.7%，而 Dream 论文报告 16.0%。差距 >3x。

可能原因：prompt 格式差异、样本集不同、去噪步数配置差异（论文用 128 步，Dream 可能用更多）。但论文未讨论此差距。这会让审稿人质疑评估 pipeline 的正确性。

**建议**：在 Section 4.1 或脚注中明确讨论差距，确认可能的原因，强调所有方法使用相同 pipeline（相对比较有效）。

### 4.2 SCP 全规模数据的统计意义

SCP (9.1% ± 0.6%) 和 DMI (9.3% ± 1.4%) 在 Countdown 上几乎相同，但 SCP 的 FLOP 开销是 DMI 的 ~6x。这一比较是论文的重要发现之一，但缺乏 pairwise McNemar test 来确认两者之间是否有统计显著差异（直觉是没有）。

### 4.3 "Training-free" 术语使用

DTA 在推理时执行梯度更新——这在技术上属于 "no pre-training required" 而非 "training-free"。论文在 Table (positioning) 中将 DTA 标记为 "Training-free ✓"，这可能误导读者认为 DTA 不涉及任何形式的训练/优化。

**建议**：定义 "training-free" 为 "不需要额外的预训练或微调数据/步骤"，区别于 "optimization-free"。

### 4.4 参考文献占位符

多处 `arXiv:2505.xxxxx` 未替换为真实 arXiv ID。总引用约 30+ 条（改善），但占位符的存在降低了可信度。

### 4.5 GSM8K partial 全规模数据未整合

full_scale_summary.json 包含 GSM8K seed 42 的部分全规模数据（vanilla 29.6%, DTA 29.0%, DTA+ReMDM 24.5%, ReMDM-conf 21.3%）。这些数据在论文中完全缺失。特别是 DTA 在 GSM8K 上也是 null result（29.0% vs 29.6%），而 Table 2 的 pilot 数据（DTA 12.5% vs vanilla 25.0%）给出了相反的信号。这进一步证实了 Section 6.4 Lesson 2（pilot 膨胀），应当报告。

---

## 5. 低优先级问题（LOW）

### 5.1 Contribution 4 中 "DTA shows task-dependent effectiveness" 的声明

基于 N=16 pilot 声称"task-dependent effectiveness"，在两个有全规模数据的 benchmark (Countdown, GSM8K partial) 上 DTA 均为 null result 的情况下，这一声明缺乏支撑。

### 5.2 Table 4 (Scaling) 完全基于 N=16 数据

非单调模式（vanilla: 12.5% at T=128, 0.0% at T=256）纯粹是 N=16 噪声。论文在文中承认了这一点，但仍用整个 Section 4.5 展示这些无信息数据。建议大幅压缩或移至附录。

### 5.3 代码开源计划未提及

实验驱动论文无代码开源承诺会降低可复现性评分。

### 5.4 "origin" 采样策略仍需更正式的定义

Section 3.1 中用括号内注释定义了 "origin"，但这不是一个标准术语。需要更正式的引用或定义。

---

## 6. 正面评价

### 6.1 重大改进（相比前四轮）

1. **叙事重构成功**：论文从"DTA 方法论文"转向"信息增强谱系框架"，更符合实际证据。标题更改是正确的决策。
2. **诚实报告负面结果**：DTA 的 null result (4.8% vs 4.7%)、DTA+ReMDM 的 degradation (3.6%)、pilot 膨胀 (~24pp) 均被明确报告。这是科学诚实的典范。
3. **谱系框架有分析价值**：Vanilla < DMI < SCP < DTA 的四级框架提供了系统性的消融结构，"浅层优于深层"的发现是反直觉且有启发性的。
4. **DMI 作为实用贡献**：~2x 改善、~1.2x 开销、<10 行代码实现——这是一个有实际部署价值的发现。
5. **Token-level 诊断指标**：Correction Precision/Recall 和 trajectory stability 是有原创性的分析工具。
6. **两轮 critique → revision 循环**：revision_round_1.marker 和 revision_round_2.marker 证明论文经历了系统性的内部审查。

### 6.2 核心科学贡献

1. **DMI 的 embedding-level 跨步信息传递**：简单、有效、低开销——最可能被社区采用的贡献
2. **Remasking 失败的定量诊断**：ReMDM-conf 31.3% correction precision、94.8 unstable positions——为理解 remasking 局限性提供了定量基础
3. **DTA 作为 null result 的学术价值**：参数级推理时适应不改善约束推理任务——这个负面结果对社区有警示价值
4. **"浅层优于深层"发现**：挑战了"更丰富的表征 → 更好的推理时扩展"的直觉假设

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 说明 |
|------|------|------|------|
| DTA null result + 理论框架矛盾导致拒稿 | **中-高** | 致命 | 审稿人可能认为核心方法失败 = 论文价值不足 |
| 缺少 MetaState/CORE 比较被要求补充 | **高** | 重大 | 最直接的竞争方法未比较 |
| 4+ TODO 图表被认为不完整 | **高** | 重大 | 5 个 pending figures + 1 pending table |
| 统计检验缺失（承诺但未交付） | **高** | 重大 | 方法部分承诺了 McNemar + Bootstrap + Bonferroni |
| 论文超长（~2x NeurIPS limit）被直接 desk reject | **高** | 致命 | 必须压缩 50%+ |
| DTA+ReMDM degradation 与"正交互补"矛盾 | **中** | 重大 | Section 3.4 需要大幅修改 |
| Countdown 基线差距（4.7% vs Dream 报告 16.0%）质疑评估正确性 | **中** | 重大 | 需要明确解释 |
| MBPP pilot 声明在两个 benchmark 全规模 null result 下失去可信度 | **中** | 中等 | Abstract 中不应突出 |
| GSM8K partial 全规模数据不在论文中 | **中** | 中等 | 信息遗漏 |

**综合投稿接受概率评估**：
- 当前状态提交至 NeurIPS/ICML 主会：**15-25%**（篇幅超限可能直接 desk reject）
- 解决 CRITICAL + HIGH 后：**45-55%**
- 最佳投稿目标：NeurIPS 2026 或 ICLR 2027 poster track
- 保底投稿目标：EMNLP 2026 Findings

---

## 8. 综合评分

| 维度 | 第一轮 | 第二轮 | 第三轮 | 第四轮 | 第五轮 | 变化说明 |
|------|--------|--------|--------|--------|--------|---------|
| 逻辑一致性 | 4 | 4 | 3.5 | 4 | 5 | ↑ 叙事重构更符合证据 |
| 清晰度 | 5 | 5 | 5 | 5 | 6 | ↑ 重构后结构更清晰 |
| 统计严谨性 | 8 | 7.5 | 7 | 7.5 | 5 | ↓ 承诺了检验但未交付 |
| 实验覆盖 | 4 | 4.5 | 4.5 | 4.5 | 6.5 | ↑ 7/7 Countdown + GSM8K partial |
| 影响力 | 7 | 7 | 7 | 7 | 7 | → DMI + 谱系 + 诊断框架 |
| 写作质量 | 5 | 4.5 | 4 | 4 | 5 | ↑ 重构改善，但仍过长 |
| 可复现性 | 5 | 5 | 4.5 | 5 | 6 | ↑ Algorithm 1 完整，config 详尽 |
| **总体** | **6** | **5.5** | **5.0** | **5.5** | **6.0** | ↑ 重构 + DTA 全规模数据 |

---

## 9. 优先修改建议

### Tier 1: 投稿前必须完成（blocking，估计 5-7 天）

1. **[C2.1] 重新定位 DTA 在论文中的角色**
   - Contribution 1 改为 "Information Augmentation Spectrum"，DMI 为核心实证贡献
   - DTA 整合到 "comprehensive empirical analysis" 中作为有启发性的 null result
   - 删除 Section 3.4 "recommended for production use" 声明，添加 degradation 讨论
   - **工作量: 1 天**

2. **[C2.2] 降级 VDTA 理论声明**
   - Proposition 1 → "Conceptual Framework" 或 "Motivating Analysis"
   - Abstract 和 Contribution 2: "proving ELBO monotonicity" → "providing a conceptual framework"
   - 讨论理论与实践 gap（ELBO ≠ 准确率）
   - **工作量: 4 小时**

3. **[H3.3] 执行并报告统计检验**
   - DMI vs vanilla McNemar test + Bootstrap 95% CI
   - SCP vs vanilla 同上
   - DMI vs SCP pairwise test
   - Bonferroni correction for multiple comparisons
   - **工作量: 4 小时（数据已有，纯计算+写作）**

4. **[H3.4] 生成关键图表**
   - Figure 1 (DTA overview): TikZ
   - Figure 2 (Spectrum): TikZ
   - Figure 3 (Token-level diagnostics): matplotlib bar chart
   - **工作量: 2 天**

5. **[H3.5] 大幅压缩论文到 NeurIPS 限制**
   - 当前 ~15,000 词 → 目标 6,000 词正文 + 附录
   - 将 Section 4.4 (Cross-Model), 4.5 (Scaling), Design Decisions 详情移至附录
   - Section 6.4 (18 Iterations) 压缩为 1 段
   - **工作量: 2 天**

### Tier 2: 强烈建议（估计 3-4 天）

6. **[C2.3] 添加 MetaState/CORE 讨论**（即使无实验比较）
   - 在 Discussion 中添加定性比较
   - 说明 training-free 约束下的 tradeoff
   - **工作量: 4 小时**

7. **[M4.1] 报告 Countdown 基线差距原因**
   - 在 Section 4.1 添加脚注或段落
   - **工作量: 2 小时**

8. **[M4.5] 整合 GSM8K partial 全规模数据**
   - 在 Section 4.3 添加全规模基线和 DTA 数据
   - 讨论 pilot vs full-scale 差异（DTA pilot 12.5% → full ~29.0% ≈ vanilla）
   - **工作量: 4 小时**

9. **[H3.1] 从 Abstract Contribution 列表中移除/降调 MBPP pilot**
   - **工作量: 30 分钟**

10. **[H3.2] 分析 DTA+ReMDM degradation 根因**
    - 在 Discussion 添加 1-2 段分析
    - **工作量: 2 小时**

### Tier 3: 理想情况完成

11. 完成 GSM8K 全规模 3 seeds 全部方法
12. 完成 MBPP 全规模（验证/证伪 pilot 声明）
13. 生成 Figure 4 (LoRA norm trajectories)
14. 编译 Table 5 (ablation)
15. 补充代码开源承诺
16. 定义 "training-free" 与 "optimization-free" 的区别

---

## 10. 与前四轮监督报告的累积对比

| 维度 | 第一轮 | 第二轮 | 第三轮 | 第四轮 | 第五轮 | 趋势 |
|------|--------|--------|--------|--------|--------|------|
| 逻辑一致性 | 4 | 4 | 3.5 | 4 | 5 | ↑ 重构后叙事更一致 |
| 统计严谨性 | 8 | 7.5 | 7 | 7.5 | 5 | ↓ 新框架下统计承诺未兑现 |
| 实验覆盖 | 4 | 4.5 | 4.5 | 4.5 | 6.5 | ↑ 7/7 方法全规模完成 |
| 写作质量 | 5 | 4.5 | 4 | 4 | 5 | ↑ 重构改善，仍过长 |
| **总体** | **6** | **5.5** | **5.0** | **5.5** | **6.0** | ↑ 全规模数据 + 叙事重构 |

**第五轮评估的核心结论**：论文在第三轮迭代中经历了重大方向调整——从推销 DTA 转向以信息增强谱系为核心框架，诚实报告 DTA 的 null result。这是正确的科学决策。当前版本的核心瓶颈已从"科学方向性问题"转变为"执行完成度问题"：统计检验需要执行、图表需要生成、篇幅需要压缩。这些都是工程性问题而非科学性问题。

最关键的三项改进（占 80% 价值）：
1. **压缩篇幅到 6,000 词**（否则可能 desk reject）
2. **执行统计检验并报告**（兑现方法部分的承诺）
3. **重新定位 DTA + 降级 VDTA 理论声明**（消除 framing-evidence 矛盾）

如果能在 1 周内完成 Tier 1，论文质量可从 6.0/10 提升到 7.5-8.0/10，达到 NeurIPS poster 的竞争力。DMI + 谱系框架 + 诊断工具 + 诚实负面结果 = 一篇有实质贡献的论文，只需要在执行层面完善。
