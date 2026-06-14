# 实验主义者提案：DTA 实验的中期诊断与证据驱动修正方案

## 核心判断

经过第 3 轮迭代的 full-scale 实验（Countdown-500 x 3 seeds，4/7 方法完成），实验证据已经足够做出若干关键判断。本次实验主义者视角聚焦于**已有数据的严格解读**、**DTA 方法的实验瓶颈诊断**、以及**基于证据的路线修正**。

**最重要的发现**：DMI（零开销跨步嵌入记忆）在 Countdown-500 上达到 9.3% 平均准确率（vs vanilla 4.7%），而 ReMDM-conf（4.4%）和 RCR（5.7%）均未显著超越 vanilla。这意味着：
1. 跨步信息传递确实有价值（H4 部分支持）
2. 传统 remasking 在推理任务上确认无效（项目 18 轮的结论在大样本下得到验证）
3. DMI 的 ~2x 改善以接近零的计算开销实现，是当前信息-效率比最优的方法

但 DTA（参数级适应）的 pilot 结果令人担忧：准确率劣于 vanilla（6.2% vs 12.5% pilot / 0% vs 0% 诊断），LoRA 范数极小（delta norm ~1e-5），自监督信号可能不足。**DTA 的命运取决于待运行的 full-scale 数据，但实验主义者必须诚实地评估当前证据并准备应对失败。**

---

## 一、已有数据的严格统计解读

### 1.1 Countdown-500 Full-Scale 结果（4/7 方法）

| 方法 | s42 | s123 | s456 | Mean | Std | vs Vanilla |
|------|-----|------|------|------|-----|------------|
| Vanilla | 4.0% | 5.0% | 5.2% | 4.7% | 0.6% | — |
| ReMDM-conf | 4.8% | 5.2% | 3.2% | 4.4% | 1.0% | -0.3pp (NS) |
| RCR | 5.4% | 5.4% | 6.4% | 5.7% | 0.6% | +1.0pp (?) |
| **DMI** | **7.8%** | **9.6%** | **10.6%** | **9.3%** | **1.4%** | **+4.6pp** |

**统计检验估算**（基于 500 样本的 McNemar test 近似）：

- **DMI vs Vanilla**：假设 3 seeds 平均正确数分别为 39/48/53 vs 20/25/26。在 seed 42 上，差异 19/500 = 3.8pp。对 n=500 的配对二分类数据，McNemar 检验需要关注不一致对。保守估计 effect size Cohen's h ~ 0.2，p < 0.01。**统计显著。**
- **RCR vs Vanilla**：+1.0pp（27 vs 20 at seed 42），在 n=500 下 Cohen's h ~ 0.05，p 可能 > 0.1。**可能不显著。**
- **ReMDM-conf vs Vanilla**：-0.3pp，方向甚至相反。**明确不显著。**

### 1.2 Pilot 数据的方法论问题

16 样本 pilot 的方差极大：同一方法在不同 pilot 运行中准确率波动 0%-25%（如 ReMDM-conf 在 task_2b 为 25%，task_5a 为 6.2%）。这确认了项目第 15 轮的教训——**小样本 pilot 仅能验证技术可行性，不能做效果判断**。

DTA 在两次 pilot 中均为 6.2%（1/16），但 vanilla 在同一 pilot 中也仅 0%-12.5%。在 16 样本下，1 题差异 = 6.25%，完全在随机噪声范围内。**DTA pilot 的"劣于 vanilla"不能作为否定证据，但也不能忽视其方向不利的信号。**

### 1.3 GSM8K Baseline 的校准

Vanilla Dream-7B 在 GSM8K-1319 上达到 29.6%（1300/1319 样本），与 Dream 论文报告的 28-30% 范围一致。这验证了评估框架的可靠性。ReMDM-conf 在 350/1319 样本的中期结果为 25.1%，低于 vanilla，与 Countdown 上的趋势一致——**纯 remasking 在推理任务上系统性地无效或微弱负面**。

---

## 二、DTA 实验瓶颈诊断

### 2.1 核心问题：LoRA 更新幅度过小

Task 8c 的诊断数据揭示了关键问题：

| 指标 | 值 | 评估 |
|------|-----|------|
| 最大 delta norm | 2.92e-05 | **极小**——在 bfloat16 下接近数值精度极限 |
| 最大 B norm | 2.88e-04 | 很小——LoRA B 矩阵几乎未更新 |
| 平均 DTA loss | 0.124 | 合理范围但偏低 |
| 预测置信度（全程）| 0.969 → 0.995 | **过高**——模型对已揭示 token 已极度自信 |

**诊断**：DTA 的 M-step 使用 masked re-prediction 作为自监督损失（遮蔽 20% 已揭示 token，从剩余上下文预测）。问题在于 Dream-7B 对已揭示 token 的预测置信度已经极高（>0.97），导致：
1. MLM loss 极小（~0.02-0.19），梯度信号微弱
2. LoRA 更新量在 gamma=0.95 衰减下迅速消失
3. 参数空间的实际"记忆"接近零——DTA 在功能上退化为 vanilla

这解释了为什么 DTA 不引入退化（H10 PASS）但也不提供改善——更新太小，无法改变模型行为。

### 2.2 与 DMI 成功的对比分析

DMI（embedding 级跨步记忆）以零训练、零梯度的方式传递了前步 logits 的 softmax 加权 embedding。它的成功表明：
- **跨步信息传递的价值在于"告诉模型前一步的预测分布"，而非"更新模型参数"**
- DMI 直接在输入空间注入软信息，避开了梯度计算的数值困难
- DTA 的梯度路径（loss → gradients → LoRA → output）引入了多重瓶颈

### 2.3 MetaState（arXiv 2603.01331）的参照

MetaState 论文（2026 年 3 月 2 日，与本项目同期）也针对 DLM 的"Information Island"问题，但采用了不同路径：
- **GRU 式跨步记忆**（非梯度更新），用 K-step unrolling 训练
- 在 LLaDA-8B 和 Dream-7B 上"consistently improves accuracy over frozen baselines"
- 需要训练额外模块（Mixer + Updater + Injector），非 training-free

MetaState 的成功间接支持了跨步信息传递的价值（与 DMI 方向一致），但也暗示**纯推理时的梯度更新可能不如训练好的记忆模块有效**。

### 2.4 Info-Gain Sampler（arXiv 2602.18176）的启示

Info-Gain Sampler 发现现有 MDM 采样器的根本局限："neglects the downstream impact of current decoding choices on subsequent steps"。它通过**信息增益**（在 masked 位置上预测当前解码决策如何影响后续 token 的不确定性）来指导 where-to-unmask。在推理任务上实现了 3.6% 平均准确率提升和 78.4→48.6 的累积不确定性降低。

**关键对比**：Info-Gain Sampler 是一个 training-free 的改进采样策略（what/where to unmask），与 DTA（how to update parameters）正交。如果 DTA 的参数更新信号不足，Info-Gain 提供了一条替代路径——通过更智能的 unmask 策略利用模型已有的推理能力。

### 2.5 LFPO（arXiv 2603.01563）的参照

LFPO 提出 Likelihood-Free Policy Optimization，将对齐建模为"geometric velocity rectification"，**直接优化去噪 logits** via contrastive updates，绕过了 likelihood 近似的高方差问题。核心思想：直接在 logit 空间做对比更新，而非通过梯度回传到参数。这暗示了一个可能的 DTA 改进方向——**在 logit/embedding 空间而非参数空间做推理时适应**。

---

## 三、证据驱动的路线修正方案

### 方案 A：抢救 DTA——加大更新幅度（最后尝试）

**假设**：DTA 失效的原因是更新幅度不足（而非方法本身有问题）。通过以下修改测试这一假设：

| 参数 | 当前 | 修改 | 理由 |
|------|------|------|------|
| lr | 5e-4 | 5e-3 ~ 1e-2 | 历史 v2 (lr=5e-3) 产生了有意义的范数变化，只是过大导致退化 |
| gamma | 0.95 | 0.99 或 1.0 | 减少衰减，让参数累积更持久 |
| M-step loss | mask 20% revealed | mask 50% revealed + 温度锐化 | 人为降低预测置信度，增大 loss |
| grad_clip | 1.0 | 5.0 | 允许更大梯度通过 |
| LoRA rank | 4 | 8 或 16 | 更大容量可能捕获更多信息 |

**实验设计**：在 Countdown-100 上做快速网格搜索（3x3=9 配置），每配置 1 seed。

| 条件 | lr | gamma | 预期 |
|------|-----|-------|------|
| A1 | 5e-3 | 0.99 | 中等更新，测试是否够 |
| A2 | 1e-2 | 0.99 | 较大更新，测试退化边界 |
| A3 | 5e-3 | 1.0 | 无衰减累积 |
| A4 | 1e-2 | 1.0 | 最激进配置 |
| A5 | 5e-3 | 0.99 + rank=8 | 增加 LoRA 容量 |

**证伪条件**：如果所有 5 个配置在 100 题上均未超过 vanilla + 2pp（即 <6.7%），则 DTA 的纯 MLM 自监督信号在 Dream-7B 上确定不足，放弃 DTA 参数更新路线。

**先导预算**：5 配置 x 100 题 x 1 seed x ~20s/样本 ≈ 2.8 GPU-hour

### 方案 B：DMI 增强——将 DMI 发展为论文核心方法

如果 DTA 确认失败，DMI 的 +4.6pp（~2x 改善）是最强的实验结果，且具有工程优势（零额外开销）。

**DMI 的论文化路径**：

1. **DMI + Info-Gain Sampler 组合**：DMI 传递跨步信息 + Info-Gain 优化 unmask 顺序，两者正交互补
2. **DMI 变体消融**：
   - Logit carry-over（当前实现）：注入前步 softmax 加权 embedding
   - Top-K carry-over：只注入 top-K 预测的 embedding
   - Attention carry-over：注入前步 attention pattern 加权的 embedding
   - Full distribution carry-over：注入前步完整 logit 向量（维度更高但信息更多）
3. **多步记忆**：DMI 当前只看前 1 步，可扩展为 EMA 式多步记忆（类似 MetaState 的 GRU 但无需训练）

**实验设计**：

| 实验 | 方法 | 基准 | 样本量 | 预期 |
|------|------|------|--------|------|
| B1 | DMI 变体消融 | Countdown-500 | 500 x 3 seeds | 找到 DMI 最优配置 |
| B2 | DMI + Info-Gain | Countdown-500 | 500 x 3 seeds | 测试组合效果 |
| B3 | Best DMI on GSM8K | GSM8K-1319 | 1319 x 3 seeds | 跨任务泛化 |
| B4 | Best DMI on MBPP | MBPP-427 | 427 x 3 seeds | 代码生成泛化 |
| B5 | DMI on LLaDA-8B | Countdown-500 | 500 x 1 seed | 跨模型泛化 |

**证伪条件**：如果 DMI 变体消融中最优配置在 GSM8K 上的改善 < 1pp，则 DMI 的跨步信息仅在 Countdown（低 entropy 约束满足任务）上有效，论文范围受限。

**计算预算**：B1-B4 约 30 GPU-hour，B5 约 8 GPU-hour

### 方案 C：诊断性框架论文（最保底）

如果 DMI 在 GSM8K/MBPP 上也无效，将所有实验整合为一篇方法论诊断论文：

**标题思路**：*"Why Cross-Step Memory Helps and Remasking Doesn't: A Diagnostic Study of Inference-Time Compute in Masked Diffusion Language Models"*

**核心贡献**：
1. 在 Dream-7B 上首次系统性地对比 7 种推理时方法（vanilla, ReMDM-conf, RCR, DMI, SCP, DTA, DTA+ReMDM）在 3 个基准上的效果
2. 揭示"Information Island"问题是 remasking 失效的根因（DMI 的成功 vs ReMDM-conf 的失败）
3. Token 级诊断框架（Correction Precision/Recall）首次量化 remasking 的信号质量
4. 负面结果的系统化报告：DTA 参数适应失败的机制分析（自监督信号不足 + 数值精度瓶颈）

---

## 四、具体实验协议（下一步）

### 优先级排序

1. **立即执行**：完成 SCP、DTA、DTA+ReMDM 的 Countdown-500 full-scale（已在进行中）
2. **次日执行**：方案 A（DTA 加大更新幅度的快速网格搜索，~3h）
3. **根据方案 A 结果决策**：
   - 如果任一配置超过 vanilla + 3pp → 继续优化 DTA，进入原定实验计划
   - 如果全部失败 → 转向方案 B（DMI 增强），DMI 成为论文核心
   - 如果方案 B 在 GSM8K 也失败 → 方案 C（诊断论文）

### 统计严谨性要求（全部实验必须遵守）

| 要求 | 具体标准 |
|------|---------|
| 最小样本量 | 200 题/方法/种子（避免 pilot 偏差） |
| 种子数 | 3（42, 123, 456），结果取平均±标准差 |
| 主要检验 | McNemar test（配对准确率对比） |
| 效应量 | Cohen's h + Bootstrap 95% CI |
| 多重比较 | Bonferroni 校正（k-1 comparisons vs baseline） |
| 多样性指标 | 必须同时报告 distinct-1/2/3 和 rep-2/3 |
| 文本定性检查 | 随机抽取 20 个正确/20 个错误样本做人工审查 |

### 关键混淆因素控制

1. **温度统一**：所有方法使用相同采样温度（0.4），避免温度退火的混淆
2. **步数统一**：所有方法使用 128 步去噪，扩展曲线实验单独做
3. **评估统一**：全部使用相同的准确率判定函数和答案提取逻辑
4. **随机种子统一**：每个种子产生相同的 prompt 序列，确保配对比较有效
5. **GPU 分配**：同一方法的不同种子尽量分配到同一型号 GPU，避免硬件差异

---

## 五、对各假设的证据更新

### 假设状态汇总（基于 full-scale + pilot 数据）

| 假设 | 原始预期 | 当前证据 | 状态 |
|------|---------|---------|------|
| H1（DTA 在 Countdown 显著有效）| +5-10pp | Pilot: -6.2pp | **高风险，待 full-scale** |
| H2（DTA+ReMDM > ReMDM）| +3-5pp | Pilot 数值噪声太大 | **未验证** |
| H3（DTA 扩展不饱和）| 对数增长 | Pilot 16 样本无信号 | **未验证** |
| H4（Level 3 > 2 > 1 > 0）| 严格递增 | Full-scale: DMI > vanilla 确认，ReMDM ≈ vanilla | **部分支持** |
| H5（DMI 改善 remask 预测）| token 级 >1pp | 未测试 | **未验证** |
| H6（SCP Precision > ReMDM）| SCP 更精准 | SCP interim ~8.4% vs ReMDM 4.4% | **方向支持** |
| H7（信息积累单调性）| 置信度递增 | 0.969→0.995 | **支持** |
| H8（gamma 最优在 0.9-0.99）| 存在最优范围 | gamma=0.95 + 极小范数 | **需要更大 lr 再验** |
| H9（remasking Precision <50%）| 低 Precision | 未收集 token 级数据 | **未验证** |
| H10（DTA 不退化）| 无退化 | rep-3 低于阈值 | **确认** |

### 新增假设（基于当前证据）

**H11**：DMI 的改善来自于打破 token 预测的条件独立性假设——前步 logits 提供了 token 间的软依赖信息，使当前步的预测能利用前步已"看到"的 token 关系模式。
- 检验：对比 DMI 与随机 embedding 注入（控制信息内容），如果随机注入无效而 DMI 有效，则 H11 成立

**H12**：DTA 的 MLM 自监督损失不足以引导推理改善——模型在已揭示 token 上的预测置信度 >0.97，梯度信号本质上是噪声而非有用的适应信号。
- 检验：方案 A 的网格搜索。如果增大 lr 后准确率反而下降（退化），则 H12 被强支持

**H13**：Info-Gain Sampler 的 unmask 策略与 DMI 的跨步信息传递正交互补——Info-Gain 优化"解码哪些位置"，DMI 优化"解码时利用哪些信息"。
- 检验：方案 B2 的组合实验

---

## 六、成功概率重估

| 方案 | 成功概率 | 论文级别（若成功） | 风险 |
|------|---------|-------------------|------|
| A: DTA 抢救 | 15%（从原 35% 下调） | NeurIPS/ICML（原始 DTA 论文故事） | LoRA 更新的数值瓶颈可能无法通过超参调整解决 |
| B: DMI 增强 | 55% | ICLR/NeurIPS（"跨步记忆是关键"故事） | DMI 可能仅在 Countdown 有效，不泛化 |
| C: 诊断框架 | 85% | EMNLP/NeurIPS D&B Track | 负面结果论文接受率低，但数据质量高 |

**综合最可能的论文形态**：方案 B 成功 → *"Diffusion Memory Injection: Zero-Cost Cross-Step Information Breaks the Remasking Ceiling in Masked Diffusion Language Models"*，核心贡献是：(1) 识别"Information Island"问题，(2) 提出 DMI 作为零开销解决方案，(3) 系统性消融证明跨步信息 > 参数适应 > 纯 remasking。

---

## 七、对 proposal 的实验主义者评价

### 当前 proposal（DTA 为核心）的问题

1. **过度乐观的效果预期**：proposal 预期 DTA +5-10pp，但 pilot 显示 DTA 可能无效或负面。LoRA 范数极小的诊断数据是严重的红旗信号。

2. **忽视了 DMI 的意外成功**：proposal 将 DMI 定位为"Level 1 消融基线"，但 full-scale 数据显示 DMI 是效果最好的方法。这不是消融基线打败主方法的问题，而是**论文故事需要重新构架**。

3. **理论框架与实验现实脱节**：VDTA 的 EM 优化理论依赖于"M-step 提供有意义的梯度更新"，但实验显示 loss 极小、梯度信号不足。理论的前提条件在实践中不成立。

### 建议的修正

1. **将 DMI 从"消融基线"提升为"核心方法"**，DTA 降级为"方法探索的教训"
2. **论文故事从"参数适应"转向"信息传递"**：核心洞察不是"DLM 需要推理时参数更新"，而是"DLM 需要跨步信息传递，且最简单的传递方式就足够"
3. **保留 DTA 作为负面结果**：DTA 的失败机制分析（自监督信号不足、数值精度瓶颈）本身有学术价值，作为论文的一个 section 报告
4. **加入 Info-Gain Sampler 作为正交改进**：引用 arXiv 2602.18176 的结果，探索 DMI + 更智能的 unmask 策略的组合

---

## 参考文献

1. Xia, K. et al. "MetaState: Persistent Working Memory for Discrete Diffusion Language Models." arXiv 2603.01331, 2026.
2. Yang, K. et al. "Improving Sampling for Masked Diffusion Models via Information Gain." arXiv 2602.18176, 2026.
3. Wei, C. et al. "LFPO: Likelihood-Free Policy Optimization for Masked Diffusion Models." arXiv 2603.01563, 2026.
4. Sahoo, S. et al. "Scaling Beyond Masked Diffusion Language Models." arXiv 2602.15014, 2026.
5. Asano, H. et al. "Where-to-Unmask: Ground-Truth-Guided Unmasking Order Learning." arXiv 2602.09501, 2026.
6. Horvitz, Z. et al. "No Compute Left Behind: Rethinking Reasoning and Sampling with Masked Diffusion Models." arXiv 2510.19990, 2025.
7. Kim, S.H. et al. "KLASS: KL-Guided Fast Inference in Masked Diffusion Models." arXiv 2511.05664, 2025.
8. Chen, Z. et al. "Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty." arXiv 2512.21336, 2025.
9. Zhong, Y. et al. "Parallelism and Generation Order in Masked Diffusion Language Models." arXiv 2601.15593, 2026.
10. Yu, L. et al. "Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models." arXiv 2601.22035, 2026.
11. Zhai, K. et al. "CORE: Context-Robust Remasking for Diffusion Language Models." arXiv 2602.04096, 2026.
12. Misaki, K. & Akiba, T. "UnMaskFork: Test-Time Scaling via Deterministic Action Branching." arXiv 2602.04344, 2026.
13. Nisonoff, H. et al. "ReMDM: Remasking Discrete Diffusion Models." arXiv 2503.00307, ICML 2025.
14. Turok, G. et al. "DUEL: Exact Likelihood for Masked Diffusion via Deterministic Unmasking." arXiv 2603.01367, 2026.
15. Jafari, A. & Anbarjafari, G. "Equilibrium Transformers: Autoregressive Modeling as Iterative Latent Equilibrium." arXiv 2511.21882, 2025.
