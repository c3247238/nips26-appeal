# Supervisor Review: Beyond Remasking — BSD & A-CFG for MDLM Inference-Time Reasoning

**Reviewer:** Sibyl Supervisor (Independent Oversight)
**Date:** 2026-03-10
**Iteration:** 4
**Overall Quality Score:** 5/10

---

## 1. 总体评估

本论文围绕 masked diffusion language models (MDLMs) 的推理时计算扩展问题，提出了两种 training-free 方法（BSD 和 A-CFG），并报告了大量负面结果。论文在**科学诚实性**方面做得出色——pre-registered hypotheses、明确标注 pilot vs. full-scale 结果、11 个假设中 4 个 falsified 均如实报告。然而，论文存在**证据与声明之间的严重失配**：两个标题方法（BSD、A-CFG）仅有 n=16 的 pilot 数据支撑，而唯一经全规模验证的方法（DMI）在概念上极为简单且属于已知方法家族。

---

## 2. 质量评分（分维度）

| 维度 | 分数 | 说明 |
|------|------|------|
| 研究动机与问题定义 | 8/10 | Information island problem 是 well-articulated 的问题定义，有独立验证（MetaState） |
| 方法创新性 | 4/10 | BSD 是 DMI 的泛化但表现更差；A-CFG 是已有方法的直接应用 |
| 实验严谨性 | 3/10 | 核心方法仅 n=16 pilot；存在数据一致性问题（见下） |
| 写作质量 | 7/10 | 结构清晰、方法描述精确、Discussion 有深度 |
| 负面结果贡献 | 7/10 | 系统性失败模式分析有价值，但 n=16 限制了机制解释的可信度 |
| 发表就绪度 | 3/10 | 缺少 BSD/A-CFG 全规模验证、缺少渲染图表、缺少 appendix |

---

## 3. Critical Issues（必须修复）

### 3.1 数据一致性问题 [CRITICAL]

**发现严重的实验数据不一致**。`compute_fair_summary.md` 中的 pilot 结果与论文中引用的数字**完全不同**：

| 方法 | 论文中 Countdown-16 | compute_fair_summary.md |
|------|---------------------|------------------------|
| DMI | 12.5% (2/16) | 0.0% (0/16) |
| A-CFG (w=1.5) | 12.5% (2/16) | 0.0% (0/16) |
| Vanilla (128 steps) | 0.0% (0/16) | 6.2% (1/16) |
| BSD | 6.2% (1/16) | 6.2% (1/16) |

这意味着 compute-fair 评估使用了**不同的随机种子或不同的 16 个样本**，与主表中的 pilot 结果不可直接对比。论文 Table 4 的 compute-fair 比较使用了主表的准确率（12.5% 等）与 compute_fair_summary 中的 vanilla-matched 准确率拼接在一起，这是**苹果对橘子的对比**。

**建议**：
1. 所有 pilot 结果必须使用同一 seed、同一 16 个样本
2. compute-fair 实验必须在同一次运行中生成所有方法的结果
3. 明确说明 pilot 样本选择方法（哪 16 个问题、选择标准）

### 3.2 核心方法缺少全规模验证 [CRITICAL]

BSD 和 A-CFG 是论文的两个标题贡献，但二者**均未在全规模（Countdown-500, 3 seeds）上验证**。本项目自身的历史已经反复证明 pilot 结果不可信：

- Iteration 3: entropy_r20_mean pilot -24.9% → full-scale -0.5% (偏差 24.4pp)
- Iteration 3: tcr_r30_s32 pilot -22.9% → full-scale -2.8% (偏差 20.1pp)
- Iteration 3: anneal pilot -16.5% → full-scale +8.1% (完全反转)

在这样的 track record 下，A-CFG 的 pilot 12.5% 可能在全规模上缩水至与 vanilla 无异。这不是推测——这是本项目的**经验规律**。

### 3.3 Abstract 与 Title 过度承诺 [CRITICAL]

Abstract 以等量篇幅介绍 BSD ($\rho = -0.95$, $r = 0.78$) 和 A-CFG (+12.5pp) 的结果，但**完全未提及这些结果来自 n=16 pilot**。一个只浏览 abstract 的读者会以为这些是经过验证的结论。

Title 中 "Continuous Belief States and Classifier-Free Guidance" 暗示两个并列的新方法贡献，但 A-CFG 是 Arriaga et al. (2025) 的直接应用，BSD 在所有测试中表现不如其特例 DMI。Title 应该诚实反映贡献层次。

---

## 4. Major Issues

### 4.1 BSD 是更差的 DMI

BSD 被定义为 DMI 的泛化（DMI 是 $k = T$、$\alpha$ 固定的特例），但在所有评估中 BSD 均不如 DMI：
- Countdown-16: BSD 6.2% vs DMI 12.5%
- GSM8K-16: BSD 18.8% vs DMI 25.0% (甚至低于 vanilla)

更关键的是 $k$-parameter 分析：只有 $k_{frac} = 0.75$（最短的 belief phase，仅 25% 步数）能工作。这意味着 BSD 的核心卖点——"连续信念演化"——在实际中必须被最小化才能运作。论文声称 "BSD produces qualitatively different internal representations" 但其 distinctive feature 在实际操作中是有害的。

**建议**：诚实讨论为何泛化版本不如特例，并考虑 BSD 是否应被定位为"理论分析工具"而非"实用改进方法"。

### 4.2 Entropy-Accuracy 相关性的统计可信度

r = 0.784, p < 0.001，但这来自 n=16 个样本，其中仅 1 个是 correct（BSD 在 Countdown-16 上准确率 6.2% = 1/16）。一个二元变量（correct/incorrect）与连续变量（entropy）的相关性在 1:15 的分类比中极度不稳定。**单个数据点的 entropy 值可以完全左右这个相关系数**。

p < 0.001 在如此极端的不平衡下需要特别说明计算方法（point-biserial? permutation test?），并提供 raw data 供审稿人验证。

### 4.3 A-CFG 不是原创贡献

论文在 Section 2.4 明确承认 A-CFG 由 Arriaga et al. (2025) 提出并在 LLaDA-8B 上取得 GSM8K 73.5 的结果。在 Dream-7B 上"应用"A-CFG 而无任何修改，这是实验验证/迁移实验，不是方法贡献。然而 title、abstract 均将 A-CFG 作为主要贡献之一呈现。

### 4.4 缺少与强 baseline 的对比

论文讨论了 wd1（GSM8K 84.5%）、Prism、d1 等方法但未与之对比。虽然这些方法需要训练，但它们在相同 benchmark 上的绝对性能水平（84.5% vs 本文最好的 9.3%）提供了重要上下文。论文应在 Discussion 中明确讨论这个差距及其含义。

---

## 5. Minor Issues

1. **r = 0.784 vs r = 0.78**：abstract 和正文中精度不一致，应统一
2. **Table 3 中 BSD+A-CFG 在 GSM8K 列为 "---"**：应说明为何未在 GSM8K 上测试该组合
3. **"immediately deployable in any MDLM inference pipeline" (DMI)**：过度声明——DMI 在 GSM8K 上零改善，应限定为 structured reasoning
4. **Figures 均为文字描述**：终审阶段应有渲染图表，当前状态不可评审
5. **Appendix C (DTA) 和 D (DMI) 被引用但未包含**：关键方法细节缺失
6. **LLaDA 2.0、LLaDA 2.1 缺少正式 citation**：如无 arXiv ID 应注明 "forthcoming" 或删除
7. **Section 3.2 中 $k_{frac}$ 表示法不一致**：部分地方用 $k_{frac} = 0.75$，部分用 $k = 3T/4$，应统一
8. **Conclusion 段落过长**：5 个 bolded 贡献点可压缩为 3 个

---

## 6. 实验结果交叉验证

### 6.1 DMI Full-Scale 验证（唯一可信结果）

DMI 的 Countdown-500 3-seed 结果在 `exp/results/summary.md` 中报告：
- s42=7.8%, s123=9.6%, s456=10.6%, Mean=9.3%, Std=1.4%
- 与论文 Table 2 一致 ✓
- McNemar p < 0.05, Cohen's h = 0.72 ✓

**但注意**：论文中 DMI full-scale Cohen's h = 0.72 出现在 pilot 段落下方，可能造成混淆。应将 full-scale 和 pilot 的 effect size 明确分离。

### 6.2 Pilot 数据来源不一致

如 Section 3.1 所述，`compute_fair_summary.md` 与 `bsd_fullscale_summary.md` / `racfg_fullscale_summary.md` 的 pilot 结果数值不同。这表明不同的实验脚本使用了不同的种子或样本集。论文中的 Table 4（compute-fair）似乎混合了两次不同 pilot run 的数据。

### 6.3 Full-Scale PPL 结果（Iteration 3 遗产）

`full_results.json` 包含 iteration 3 的 PPL-based 全规模结果（entropy, tcr, anneal），确认了论文中关于"pilot PPL 改善在全规模上消失"的声明：
- entropy: pilot -24.9% → full -0.5% (p=0.530) ✓
- tcr: pilot -22.9% → full -2.8% (p=0.254) ✓
- anneal: pilot -16.5% → full +8.1% (p=0.636) ✓

---

## 7. 改进建议（按优先级）

### P0（阻塞发表）
1. **BSD 和 A-CFG 全规模验证**：Countdown-500, 3 seeds，这是唯一能使论文从"interim report"升级为"完成研究"的步骤
2. **修复 compute-fair 数据一致性**：使用同一 pilot seed/样本集重新运行所有方法的 compute-fair 对比
3. **重写 Abstract**：明确标注 BSD/A-CFG 结果来自 n=16 pilot
4. **渲染所有 Figures**

### P1（显著提升质量）
5. **重定位 A-CFG**：从"我们提出的方法"改为"我们将 A-CFG (Arriaga et al., 2025) 应用于 Dream-7B"
6. **将 DMI 提升为核心贡献**：作为唯一经验证的方法，DMI 不应被放在 Appendix
7. **A-CFG re-masking percentage ablation**：m=10% 是关键超参但未做消融
8. **补充 Appendix C (DTA) 和 D (DMI)**

### P2（锦上添花）
9. **多模型评估**（至少 Dream-7B + LLaDA-8B）
10. **BSD 更多 $k$ 值探索**（$k_{frac} = 0.85, 0.90$）
11. **质性错误分析**：BSD/DMI/A-CFG 分别解决哪类问题？
12. **Title 修改**：反映实际贡献层次

---

## 8. 风险评估

| 风险 | 严重度 | 概率 | 影响 |
|------|--------|------|------|
| A-CFG 全规模验证失败 | High | 50%+ | 论文核心叙事需重构 |
| BSD 全规模仍不如 DMI | Medium | 80%+ | BSD 降级为理论分析工具 |
| 审稿人指出 A-CFG 非原创 | High | 90%+ | 直接拒稿理由之一 |
| Compute-fair 数据不一致被发现 | Critical | 如不修复则 100% | 严重损害论文可信度 |
| n=16 pilot 机制解释被质疑 | Medium | 80%+ | Discussion 需要大幅缩减 |

---

## 9. 结论

本论文有三个真正有价值的内核：(1) information island problem 的清晰定义，(2) DMI 作为经验证的 near-zero-cost 改进，(3) 系统性负面结果对设计空间的约束。但当前版本试图将这些包装成一篇"两个新方法 + 深度分析"的论文，而证据不足以支撑这一定位。

**最关键的下一步**：完成 BSD 和 A-CFG 的全规模验证。这将在 ~18 GPU·h 内决定论文的最终定位——是 NeurIPS/ICML 主会论文还是 workshop/EMNLP 的系统性研究报告。在验证完成之前，论文的证据-声明比严重失衡。

**当前发表就绪度**：Below threshold (NeurIPS/ICML)。
**如完成全规模验证且 A-CFG 成立**：Borderline accept。
**如 A-CFG 全规模失败**：需重构为 DMI + 系统性研究论文，适合 EMNLP 或 workshop。
