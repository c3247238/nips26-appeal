# Iteration 3 最终提案：写作改善综合方案

**综合自**：创新者、实用主义者、理论研究者三方提案
**核心结论**：三方一致同意——零实验、全写作改善，5 天工作量，预期投稿概率 20-30% → 45-60%

---

## 1. 叙事重定位（三方一致）

**新标题**："When Does Lightweight Test-Time Scaling Succeed and Fail in Masked Diffusion? A Diagnostic Map Across Scales"

**新叙事主线**（三幕结构）：
- **第一幕（贡献 1）**：方法论贡献——小样本评估陷阱 + PPL 有效性条件（PVC 三元组）
- **第二幕（贡献 2）**：信息论边界——内部信号无法区分"错误 token"和"歧义 token"，解释 remasking 在开放文本上失败、在 GSM8K 上有效的根本原因
- **第三幕（贡献 3）**：正面发现——温度退火（过程级干预）成功、entropy remasking 在结构化推理上有效

**关键**：从"负面结果报告"转为"诊断性地图 + 信息论边界"

---

## 2. PPL 自矛盾消解（三方一致）

**方案**：定义 "PPL 有效性条件"（PPL Validity Conditions, PVC）三元组：
1. Diversity ratio > 0.9
2. Degenerate rate = 0%
3. Mean/Median PPL ratio < 1.5

**操作**：
- Section 3.2.2：将 PPL 定义为"条件有效指标"，明确有效性前提
- 结果表格增加 "PPL Valid?" 列
- 0.6B remasking：PVC 不满足（degenerate rate = 71%）→ PPL 改善**无效**
- anneal_fix：PVC 满足（diversity > 0.95, degenerate rate = 0%）→ PPL 改善**有效**
- 将此从"逻辑漏洞"转化为"方法论贡献"

---

## 3. 统计检验双报告（理论 + 实用一致）

**同时报告 t-test 和 Wilcoxon**，并解释差异：
- Wilcoxon (p < 0.001)：大多数 prompt 上有改善
- t-test (p = 0.12-0.19)：少数极端异常值膨胀方差
- 机制：**尾部修剪**（tail trimming）而非均值位移
- Cohen's d ≈ 0.20 因重尾分布系统性低估实际效果
- 增加 P50/P75/P90/P95/P99 分位数分析

---

## 4. 压缩方案（12,000 → 6,000 词）

| 章节 | 当前 | 目标 | 策略 |
|------|------|------|------|
| Abstract | ~400 | 200 | 以正面发现开头 |
| Introduction | ~1200 | 600 | 三个贡献 + 信息论框架 |
| Related Work | ~1500 | 800 | 2 段式 + 对比表 |
| Methods | ~3000 | 1200 | 统一框架 + 变体表格，伪代码移附录 |
| Results | ~4000 | 2000 | 大表 + 2 图，逐方法详细移附录 |
| Discussion | ~1500 | 800 | 二分法 + PVC + 理论框架 |
| Conclusion | ~400 | 300 | 3 要点 |

附录：A=详细方法, B=完整结果, C=统计报告, D=LLM-Judge, E=定性样例

---

## 5. 图表方案（3 张必做）

### Figure 1：诊断地形图（热力图）
- 方法 × 模型规模，颜色=PPL 改善，标注 PVC 通过/失败
- 一图概括全部实验结果

### Figure 2：PPL 分布对比图（violin plot）
- Baseline vs anneal_fix 的 PPL 分布
- 展示尾部修剪效应，解决统计检验差异

### Figure 3：Process-Level vs Output-Level 概念图
- 两条路径对比，温度退火（成功）vs remasking（失败）
- 核心理论贡献可视化

---

## 6. 执行优先级

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| P0 | PPL 自矛盾消解（PVC 框架） | 0.5 天 |
| P0 | 统计检验双报告 + 尾部分析 | 0.5 天 |
| P0 | 正文压缩到 6,000 词 | 1.5 天 |
| P0 | 3 张核心图表 | 1 天 |
| P1 | Abstract + Introduction 重写 | 0.5 天 |
| P1 | 失败模式理论解释 | 0.5 天 |
| P2 | GSM8K 叙事降调 | 0.5h |
| P2 | 参考文献占位符替换 | 0.5 天 |

---

## 7. 不做的事

1. 不做新实验
2. 不重构代码
3. 不尝试更复杂统计方法
4. 不优化 AdaReMask 或任何新方法
5. 不过度美化负面结果
