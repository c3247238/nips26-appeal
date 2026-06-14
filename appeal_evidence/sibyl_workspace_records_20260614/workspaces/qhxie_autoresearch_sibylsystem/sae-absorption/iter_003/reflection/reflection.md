# Reflection Report — Iteration 3

**Date**: 2026-04-14 | **Score**: 5.5/10 | **Verdict**: CONTINUE | **Quality Trajectory**: 停滞

---

## 执行摘要

迭代 3 未能推动分数超过迭代 2 的 5.5 基线。研究方向依然可行且具有一定新颖性（OMP oracle 设计、EncNorm 实证信号、O_jaccard 独立互补），但核心执行问题与前两轮如出一辙：**所有实验仍在 PILOT 模式运行，未能按方法论规定的规模执行**。Supervisor 和 Critic 均将此列为首要阻断项。本次迭代出现了三项新的 critical 问题（TopK-32k 标签虚假声明、EncNorm 机理叙事被自身数据否定、六张图全部缺失），同时遗留了来自迭代 2 的两项系统性循环问题（实验规模不足、标签集循环使用）。

论文距离 NeurIPS 2026 主会接受所需的 7.5+ 分需要清晰的修复路径：(1) OMP 全规模重跑，(2) TopK-32k 结果重新定性，(3) 生成所有 6 张图，(4) 引入至少一个额外模型的验证。这些修复在技术上均可实现，且没有需要重新设计实验的根本性障碍。

---

## 问题分类分析

### EXPERIMENT（实验）

**[CRITICAL] EXP001 — OMP 实验天花板效应，从未完成全规模**

C2 在 97.8% 基线吸收率下只跑了 3 字母 × 30 tokens，标准误差约 0.037。预注册的判定准则"OMP >= 80% FF 吸收率"在此天花板下无论真实效应如何都会被自动满足。论文声称"毫无疑问地证伪"是不可辩护的——零功效下的零结果不等于证伪。根本原因：pilot 通过后没有触发全规模运行的机制。

**[CRITICAL] EXP002 — 所有核心实验停留在 PILOT 规模**

A1、A2、A3、B1、B2、C2、D2 结果 JSON 均标记 mode='PILOT'。这是第三次迭代仍然存在的系统性问题。B1 令牌数为目标的 5%（10k/200k），C2 令牌数为目标的 0.9%（90/10000）。

**[CRITICAL] EXP003 — 跨架构"复现"使用了非吸收代理标签**

E1 和 A3 均确认 TopK-32k SAE 在 IG 测量下吸收率为 0%。n_pos=77 标签来自解码器余弦对齐（cosine ≥ 0.30），而非吸收检测流程。论文在摘要和 Table 1 中将 AUROC=0.837 作为"跨架构复现"是事实性错误。唯一有效的吸收检测声明是 AUROC=0.757（n_pos=18 金标签）。

**[MAJOR] EXP005 — 跨层次实验完全无效**

D2 实体类型吸收率 = 0%，而负对照 = 5%，方向与 H3 相反。D2 说明承认负对照用了错误的 probe。H3 从未被有效测试。

**[MAJOR] EXP009 — O_jaccard AUROC 存在结构性零覆盖**

18 个被吸收特征中有 9 个因不在字母特征集中而 O_jaccard=0（构造性约束）。AUROC=0.721 将可检测特征（9 个）与构造性零分（9 个）混合计算，未分别报告。

### SOUNDNESS（可靠性）

**[CRITICAL] EXP004 — EncNorm 机理叙事被自身数据否定**

论文第 3.1 节主张梯度竞争推高 ||w_enc,j||_2。A2 层分析：L2 比值=0.877（被吸收特征范数更低），L4=1.055，L6=1.267，L8=0.891，L10=0.933。信号在 5 层中的 3 层反向。这是用后验推测而非推导支撑的机制故事，与多数层的实验数据矛盾。

**[MAJOR] EXP007 — EncNorm 独立于 EDA 的贡献未建立**

Spearman r(EncNorm, EDA) = 0.712，共享约 51% 方差。AUROC 提升（0.757 vs 0.650）可能完全来自 EncNorm 分布形状对极端类不平衡（18/24576）更有利，而非独立的吸收信号。未进行偏分析控制 EDA。

**[MAJOR] EXP006 — F1 宽度恢复 67% 有未量化的钩子点混淆**

Standard-24k 用 resid_pre（注意力前），TopK-32k 用 resid_post（注意力后）。两者在不同激活空间。两个被吸收特征（2406、11270）共享 best_matching_32k_idx=16435；另两个（24154、7371）共享 29309——疑似假匹配。未计算 resid_pre 与 resid_post 之间的余弦相似度基线。

### WRITING（写作）

**[CRITICAL] WRI001 — 6 张必需图全部缺失**

论文引用 fig_teaser、fig_enc_norm_hist、fig_omp_design、fig_roc_curves、fig_ablation、fig_omp_results，但这些文件均不存在于 writing/latex/figures/。现有 PDF（fig1_eda_method.pdf 等）是旧迭代的图。LaTeX 编译会产生断链引用。

**[MAJOR] WRI002 — 确定性语言与统计功效不匹配**

"Decisively falsifies"、"unambiguous"、"decisive"等表述与零功效实验的实际证据不符。

**[MAJOR] WRI003 + WRI004 + WRI005 — 结构冗余、MP-SAE 未讨论、Jaccard 定义不一致**

Section 4.1 逐字复制 Section 3.4。Costa et al. MP-SAE 未讨论导致第 5.1 节实践建议具有误导性。论文正文使用 f_k > f_j 而 notation.md 和 B2 实验使用 f_i > 3*f_j。

### PLANNING（规划）

**[CRITICAL] PLAN001 — Pilot-to-full 升级逻辑缺失（第三次循环）**

task_plan.json 没有"全规模运行触发"机制。Go/no-go 门通过后，实验默认保留在 PILOT 结果上不再扩展。这是三轮迭代中最重要的流程漏洞。

**[MAJOR] PLAN002 — 标签集循环使用，验证集不独立**

同一 n_pos=18 标签集跨三轮迭代用于假设发现和验证。EncNorm 在迭代 2 这 18 个标签上被发现，在迭代 3 同一批标签上被"验证"。这是循环评估，不构成独立验证。

---

## 资源效率分析

**实际 GPU 利用率**: 约 85%，双 GPU 并行调度正常（任务 A1/B1/C1/D1 并行，A2/A3/B2 并行，C2/D2 并行）。

**空闲时段**: 约 15 分钟（任务切换间隙）。

**瓶颈阶段**:
1. **Pilot-to-full 升级缺失**：所有实验在 PILOT 规模完成后没有继续扩展，导致整个实验阶段等效于无效 GPU 时间（对于论文提交而言）。
2. **图表生成缺失**：6 张图在写作阶段之前从未生成，成为提交阻断项。

**效率建议**:
- 在每个 pilot 门通过后立即生成全规模任务（自动触发，不依赖人工判断）
- 将图表生成作为写作前的独立并行任务，与全规模实验同步运行
- 将 held-out 标签生成作为每次迭代第一批任务之一

---

## 质量趋势评估

| 维度 | 迭代 1 | 迭代 2 | 迭代 3 | 趋势 |
|------|--------|--------|--------|------|
| 新颖性 | 6 | 7 | 7 | 持平 |
| 可靠性 | 4 | 4 | 4 | 停滞 |
| 实验严谨性 | 3 | 4 | 4 | 微升后停滞 |
| 可复现性 | 4 | 5 | 5 | 微升后停滞 |
| **总分** | **4.5** | **5.5** | **5.5** | **停滞** |

迭代 1→2 有明显进步（EDA 理论基础、统计工具箱建立）。迭代 2→3 完全停滞。根本原因是相同的实验规模问题在三轮中未被解决。

---

## 根因分析

**技术问题 vs 流程问题的分离**：

本轮迭代的研究方向选择是合理的。OMP oracle 设计是有效的（如果在有统计功效的规模下运行）。EncNorm 的实证信号是真实的（AUROC=0.757，Cohen's d=0.971）。O_jaccard 结构独立性是一个真实发现。**问题不在于研究假设，而在于执行流程**。

**三个核心流程失效**：

1. **Pilot-to-full 升级逻辑缺失**（三轮循环）：Planner 生成了 pilot 任务，pilot 通过了，但没有机制自动生成和执行全规模任务。这不需要算法改进，只需要在 task_plan.json 中加入明确的"full scale after pilot" 阶段。

2. **标签集验证集分离缺失**（三轮循环）：发现集和验证集使用同一批 18 个标签，迭代改进的指标在实质上是对 18 个特征点的过拟合检验。需要一次性完成 held-out 标签生成。

3. **跨模型访问未预解决**（两轮循环）：Gemma Scope 在迭代 2 也遇到障碍，迭代 3 仍然 401。HuggingFace gated license 的接受是一次性人工操作，需要在规划阶段前完成而不是作为实验任务之一。

---

## 迭代改善模式

**已改善（相比迭代 2）**：
- Claim-evidence integrity 提升：paper.md 与 source JSON 数值一致性从"部分不一致"到"全面一致"（12 项核查全部通过）
- 负面结果报告依然是论文最强部分（D2 probe 失败、E1 Gemma 拒绝、F1 hook-confound 均主动披露）
- 写作风格无滥用词汇，数字先行的写作规范维持

**未改善（持续循环）**：
- 实验规模（第三次）
- 图表生成（第三次）
- 模型多样性（第二次）
- 标签集独立验证（第二次）

---

## 迭代 4 明确优先级

以下清单按阻断严重性排序，前 3 项是 NeurIPS 7.0 分门槛的直接前提：

1. **C2-full**: 26 字母 × 1000 tokens，bootstrap CI on (AR_OMP - AR_FF)，power analysis
2. **TopK-32k 重新定性**: 将 AUROC=0.837 改为"decoder-letter-probe alignment prediction"，移出主结果或加注脚
3. **生成 6 张图**: 从 source JSON 生成，写作阶段前完成
4. **Gemma Scope 许可（人工操作）**: 在实验规划之前接受 HuggingFace gated license
5. **Held-out 标签生成**: 在 b, c, d, f, g 字母上生成新标签用于独立验证 EncNorm AUROC
6. **B1-full**: 200k tokens 共现图，确保 O_jaccard 统计可靠
7. **Hook-confound 量化**: 计算 L6 resid_pre vs resid_post 余弦相似度基线（30 分钟实验）
8. **EncNorm 独立性分析**: EDA+EncNorm 联合 LR vs EDA 单独 LR，DeLong 测试
9. **写作修复**: Section 4.1 删除、WRI002 确定性语言弱化、WRI004 MP-SAE 讨论、WRI005 定义统一
