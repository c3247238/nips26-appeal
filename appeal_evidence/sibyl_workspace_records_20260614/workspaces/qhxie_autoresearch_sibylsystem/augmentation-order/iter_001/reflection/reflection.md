# Iteration 0 Reflection Report — Augmentation Order Study

**Generated**: 2026-04-03  
**Iteration**: 0  
**Overall Quality Score**: 4.0 / 10 (supervisor review) | Writing: 7/10 (intro) → 4/10 (experiments)  
**Status**: Pilot-complete, full-scale experiments BLOCKED pending execution  

---

## 迭代摘要

本次迭代 (Iteration 0) 完成了完整的流水线首轮：Idea 生成与筛选 → 新颖性验证 → Pilot 实验规划 → 所有 Tier (0/1/2/3/4) Pilot 实验执行 → 结果辩论 → 论文撰写（全章节）→ 章节批评与修订 → Supervisor 终审。

**核心发现**：研究问题本身（augmentation 操作顺序对精度的影响）是新颖且真实的——两篇综述明确将此标注为未解决问题，截至 2026 年 4 月无并发竞争论文。然而，所有实验结果均来自极度欠拟合的 Pilot 阶段（10 epoch、100 样本子集、单 seed），导致 Supervisor 评分只有 4.0/10，主要障碍是**实验规模不足，无法支撑任何假设结论**。

研究方向正确，下一迭代的核心任务极其明确：**执行 Full Tier 1 实验**（6 orderings × 2 architectures × 2 datasets × 5 seeds × 200 epochs，约 20 GPU-h）。

---

## 问题分类分析

### 1. EXPERIMENT — 高优先级（6 个问题）

**EXP-001（阻塞性）**：所有 Tier 1 实验仅有单 seed、10 epoch、100 样本。ResNet-18/CIFAR-10 精度 10–11%（恰好在 10 分类随机猜测基准），ViT/CIFAR-100 精度 2.6–2.9%（100 类随机基准约 1%，几乎无学习）。在这两个 block 上的任何排序差异都无法区分信号与噪声。H1 "confirmed" 和 H2 "confirmed" 的声明完全不成立——n=1 时无法执行配对 t 检验、ANOVA 或 Bonferroni 校正。

**EXP-002（误导性表格）**：Table 1 将排序实验（10 epoch、100 样本，精度 10–46%）与 baseline（30 epoch、全数据集，精度 57–92%）并排呈现。两组训练条件相差约 500 倍数据量和 3 倍 epoch，任何跨组比较都是无效的，但版式上无法让读者避免做出比较。

**EXP-003（关键数字未验证）**：Tier 2 的 9.01 pp 类别级排序差异——论文最引人注目的发现——来自单 seed、单 architecture-dataset 组合、5k 样本、10 epoch。这个数字是否在 50–200 epoch、全数据集、多 seed 下存活完全未知。在提交 Full Tier 2（18 GPU-h）之前必须先运行一个 Tier 2 确认 Pilot（1.5 GPU-h，~45 分钟）。

**EXP-004（假设编号混乱）**：H2 在 proposal.md 定义为"架构差异敏感性（ViT vs CNN 有不同排序偏好）"，在 final_summary.json 中悄然重定义为"可逆性排序（CJ→Flip→Crop）超过常规"。原始 H2 的预注册测试（双因素 ANOVA，eta^2>0.05）从未执行；悄悄用胜负计数替代。这是假设追踪失控的典型案例。

**EXP-005（Tier 3 标签倒置）**：Tier 3 使用从 CIFAR-100/ResNet-18 block 选出的"最佳"/"最差"排序，但在 M5 量级下，标注的"最差"排序 (CJ→Flip→Crop) 以 0.5123 > 0.5088 反超了"最佳"排序。M14 下两个排序的精度完全相同（0.245，精确到小数点后 3 位），这一数字可疑且需要调查。

### 2. SOUNDNESS — 理论漏洞（2 个高优先级问题）

**SOUND-001（定理证明缺失核心步骤）**：Theorem 1（排序依赖泛化界）的证明草图跳过了关键的 bubble-sort 分解论证。"由 Lipschitz 性质，W_2(mu_sigma, mu_sigma') <= sum NC_2" 这一步需要将 sigma -> sigma' 分解为相邻对换序列，并在每步应用三角不等式。任何具有最优传输背景的审稿人都会立即发现这个漏洞。

**SOUND-002（DPI 逻辑缺口）**：收缩系数 eta_i 被定义为分布无关的最坏情形（对所有 p,q 取上确界），使其成为信道本身的性质，与在流水线中的位置无关。但排序依赖信息损失的论证需要 eta_i 依赖于输入分布 nu（nu 随之前变换的不同而改变）。当前定义从根本上与排序依赖的声明矛盾。

### 3. ANALYSIS — 分析错误（4 个问题）

**ANALYSIS-001（NC_2 代理严重不足）**：NC_2 用 100 样本和 100 个随机投影在 3072 维空间（32×32×3）中计算 SWD——在如此高维空间中，100 样本实际上产生随机数。H3 的"证伪"结论（rho=-0.20）不是 NC_2 框架的反证，而是代理计算不充分的证据。预注册计划规定 10k 样本、1000 次投影。

**ANALYSIS-002（MI 估计使用随机质量编码器）**：InfoNCE MI 估计使用 10 epoch 训练、100 样本的编码器（ResNet-18 在此规模的 CIFAR-10 上精度约 10%，相当于随机）。从近随机编码器提取的 MI 估计不承载任何关于任务相关互信息的信号。CIFAR-10 和 CIFAR-100 上的符号翻转（rho=+0.54 vs -0.66）几乎可以肯定是编码器噪声，而非真实的任务难度调制效应。

**ANALYSIS-003（Cohen's d 数值无效）**：n=1 seed 时，每个排序条件只有 1 个观测值，within-ordering 方差为 0，Cohen's d 在技术上未定义。分析脚本产生了 2.27–2.71 的 d 值，这些数字创造了虚假的大效应量印象，具有误导性。

**ANALYSIS-004（DPI 预测与观测结果矛盾）**：DPI 可逆性原理预测 CJ→Flip→Crop 最佳，但实际上在最多 block 中胜出的是 Flip→Crop→CJ。论文随后通过将 Flip-first 说成"与可逆性推理一致"来事后合理化，但这是两个不同的排序——特定预测（CJ-first 最优）未被证实。

### 4. WRITING — 写作问题（4 个问题）

**WRITING-001（Introduction 过度声明）**：现在时态呈现 Pilot 结果为已建立事实，缺少 Pilot 限定语。Discussion 的实践建议部分提供了"使用 Flip→Crop→CJ 作为默认值"这样具体的建议，未附加 Pilot 警告。

**WRITING-002（5/6 计划图表缺失）**：Figure 2（排序热图）、Figure 3（NC 散点图）、Figure 4（MI 条形图）、Figure 5（量级曲线）、Figure 6（架构小提琴图）全部缺失。Experiments 章节引用了 \ref{fig:magnitude} 但没有实际图表。

**WRITING-003（方法与结果的训练条件不匹配）**：方法章节描述 200 epoch、5 seed 的完整协议，但结果只报告 10 epoch、1 seed、100 样本的 Pilot 数据，让读者预期完整协议结果。

**WRITING-004（可逆性分类不一致）**：RandomHorizontalFlip 被称为"完全可逆的"（bijection，eta=1）但分类为"中等可逆性"，这在信息论层面自相矛盾。

### 5. IDEATION — 叙事框架需重建

**IDEATION-001（理论叙事与证据矛盾）**：NC_2 和 DPI 可逆性原理在 Pilot 阶段均被证伪，但论文仍将其作为正面贡献来宣传。正确框架应该是：从实证发现（排序影响精度；ViT 比 CNN 更敏感；交错排序优于块状排序）出发，将 NC/DPI 框架定位为"我们测试并发现不足的理论候选"，将像素空间分布测量与优化中介学习结果之间的差距定为新颖理论贡献。三个新假设（NH1: 任务难度调制最优排序；NH2: 量级-排序差异倒 U 形；NH3: 特征空间 NC_2 恢复预测力）提供了尚未探索的方向。

---

## 资源效率分析

**GPU 利用率**：`gpu_progress.json` 中 `timings` 字段为空，无法量化实际 GPU 空闲时间。所有已完成任务均使用单 GPU。

**并行调度机会**：
- Tier 1 的四个子任务（tier1_resnet18_cifar10、tier1_resnet18_cifar100、tier1_vit_cifar10、tier1_vit_cifar100）在 pilot_tier0 完成后完全独立，可在 4 个 GPU 上同时运行，预计将 ~40h 单 GPU 时间压缩至 ~10h 墙钟时间。
- Tier 2 确认 Pilot（1.5 GPU-h）可与 Tier 1 同时在第 5 个 GPU 上运行，无需等待 Tier 1 完成。
- Tier 4a NC_2 重计算（CPU-only）和图表生成可在 GPU 运行 Tier 1 期间并行进行。

**瓶颈阶段**：`experiment_cycle`（Full Tier 1 尚未执行）是当前主要瓶颈。写作和分析阶段执行良好，但所有结论需要依赖 Full Tier 1 的数据。

---

## 质量趋势评估

本次迭代是项目第 0 轮，没有历史趋势可比较。当前质量得分：

| 维度 | 分数 |
|------|------|
| Novelty（新颖性） | 7/10 |
| Soundness（理论严谨性） | 4/10 |
| Experiments（实验充分性） | 2/10 |
| Reproducibility（可重复性） | 5/10 |
| Writing（写作质量，intro） | 7/10 |
| Writing（写作质量，experiments） | 4/10 |
| **Overall** | **4.0/10** |

预期轨迹：Full Tier 1 + 理论修复 + 叙事重建后，得分应可达到 6.5–8.0（见 supervisor 分析中的具体升分路径）。

---

## 根因分析

**最核心根因**：实验从 Pilot 阶段直接进入写作阶段，而没有先运行 Full 实验。任务计划中所有 Tier 1 子任务都标注了 `"pilot"` 字段，且 `gpu_progress.json` 显示所有任务均已在 Pilot 模式下"完成"。Orchestrator 在 Pilot pass criteria 满足后即推进到下一阶段，没有强制要求切换到 full-scale 运行后再进行写作。

这是一个流水线设计问题：Pilot 模式本应只用于快速验证 idea 可行性（约 10–15 分钟），然后触发 Full 实验；但实际流程将 Pilot 结果直接传递给写作阶段，跳过了 Full 实验步骤。

**次要根因**：
1. 分析脚本缺少 n>=2 的前置检查，在 n=1 时仍然产生 Cohen's d 和假设 "confirmed/falsified" 判定
2. NC_2 代理在不足规格下运行（100 vs 10k 样本）未触发警告或暂停
3. H2 假设在 pipeline 中途被重定义，没有版本控制或一致性检查

---

## 系统自检响应

无 `logs/self_check_diagnostics.json` 文件，系统自检未执行。建议在下一迭代开始前运行系统自检，特别是验证分析脚本中的 n>=2 前置检查是否存在，以及 pilot_summary 的 pass criteria 是否正确区分了"pilot 通过"和"full-scale 已完成"。

---

## 成功模式提炼

1. **诚实报告负面结果**：H3 和 H5 的证伪以具体数字呈现，这是少见的科学诚实，并且本身构成有价值的发现
2. **预注册假设设计**：每个假设都有明确的指标、阈值和证伪标准，这种严谨性超过大多数 ML 论文
3. **配对 Seed 设计的动机说明**：方法章节清楚地解释了配对设计如何支持更有力的统计检验
4. **四层实验结构**：Tier 1（全因子）→ Tier 2（类别级）→ Tier 3（量级交互）→ Tier 4（理论验证）的层次设计覆盖多个维度且各自有明确边界
5. **"零成本超参数"框架**：在 Introduction 中有效传达了排序对从业者的实际意义
6. **假设判定汇总表（Table 3）**：为审稿人提供了一目了然的预注册阈值与观测值对比，是强有力的组织工具
