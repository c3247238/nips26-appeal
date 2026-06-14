# Experiments 章节评审意见

## 评分: 8/10

## 优点

1. **两阶段实验设计具有教育价值**：Phase 1 发现 PPL-gaming 陷阱 -> Phase 2 针对性设计多指标评估 -> Pilot 筛选 -> Full-scale 验证。这种自我修正的叙事让负面结果更有说服力，也为社区提供了方法论层面的贡献。

2. **统计严谨性高**：Full-scale 评估使用 101 prompts x 3 seeds，配对 t 检验 + Wilcoxon + Bootstrap CI + Bonferroni 四重检验，统计功效分析表明可检测 5% 的 PPL 改善。这远超 DLM 领域的一般评估标准。

3. **Pilot vs Full-scale 对比分析（Section 5.4）是全文最有价值的发现之一**：Table 5 清晰展示了 20-25 个百分点的缩水，三个原因（小样本、非代表性 prompt、单 seed 放大）分析到位。"DLM 推理时扩展实验需要 >= 100 prompts x >= 3 seeds"的建议具有实际指导意义。

4. **Phase 1 的 PPL-gaming 发现具有警示意义**：Table 2 清晰展示了 0.6B 模型上 PPL 下降与 diversity 崩溃的反向关系，"adaptive remasking 71% degeneracy rate"是令人印象深刻的反面案例。

5. **计算开销分析（Section 5.5）提供了实用参考**：即使方法有效，1.5-1.9x 的开销换取 <3% 的改善也不经济——这一论点加强了负面结论。

6. **Summary（Section 5.6）高度凝练**：7 条发现涵盖了所有实验维度，措辞精确，便于读者快速获取关键信息。

## 不足

1. **Per-seed 分析表格（Section 5.3.3）缺少 baseline 数据**：Table 中 Vanilla 行所有 seed 列均为"---"，使读者无法判断 method 的 per-seed PPL 是否优于对应 seed 的 baseline。这是一个关键遗漏——没有 per-seed baseline，CV 的解释力大打折扣（读者无法区分"seed 间方差"和"方法 x seed 交互效应"）。

2. **缺少定性分析 / 生成样例**：全文没有展示任何生成文本样例。对于负面结果论文，展示几组 baseline vs method 的输出对比可以帮助读者直观理解"为什么重掩码没有改善质量"——是产生了不同但等质量的文本，还是引入了特定类型的错误？Section 5.2.1 提到了一个退化样例（"computers to learn to data..."），但这只是 parallel voting 的，缺少 TCR/entropy 的对比。

3. **Phase 1 与 Phase 2 的模型/设置差异过大，削弱了跨 phase 推论**：Phase 1 用 0.6B MDLM 和 8B LLaDA，Phase 2 用 7B Dream。模型架构、训练数据、采样算法均不同。Phase 1 发现的 PPL-gaming 现象是否在 Dream-7B 上复现？文中未验证。如果 Dream-7B 本身不存在 PPL-gaming 问题（Table 4 显示 0% degeneracy），那么 Phase 1 的教训在 Phase 2 语境下的相关性需要更明确地论证。

4. **Best-of-N 仅在 Phase 1（0.6B）测试**：Section 5.2.1 的 Best-of-N 结果使用 0.6B 模型，且 scoring 用的是模型自身置信度（不是 GPL-2 PPL？——与 Methods 4.5.3 说的 GPT-2 PPL scoring 不一致）。Phase 2 未在 Dream-7B 上测试 Best-of-N (with GPT-2 scoring)，这是一个明显的实验缺口。Best-of-N 是最简单的 TTC baseline，缺少它在主模型上的数据使结论不够完整。

5. **统计功效声明需要补充细节**：Section 5.1 声称"can detect a 5% PPL improvement with power > 0.80"，但未给出效应量和方差的假设。DLM 生成的 per-prompt PPL 方差极大（个别 prompt PPL 从 <6 到 >80），配对差异的标准差是多少？在这种高方差下，power 分析的假设是否合理？

6. **温度退火结果缺少与 baseline 温度的公平对比**：Anneal (0.6->0.2) 的平均温度约为 0.4，恰好等于 baseline 温度。Section 5.3.2 报告其 median PPL +8.1%，但未分析这是否仅仅因为 anneal 在早期步骤使用了高温（0.6）。与固定 tau=0.3 或 tau=0.5 的对比数据缺失。

7. **LLaDA-8B 实验（Section 5.2.3）仅用 1 seed**：用 32 prompts x 1 seed 得出"ReMask-Retry worsens PPL by +31.5%"的结论，但未报告 p-value 或 CI。按照本文自己在 Section 5.4 中对小样本评估的批评标准，这一实验的证据强度不足。

## 具体修改建议

1. **补充 per-seed baseline PPL**：在 Table（Section 5.3.3）中添加 Vanilla 的 per-seed median PPL，使读者可以计算每个 seed 下的 delta PPL。格式示例：

   | Method | Seed 42 | Seed 123 | Seed 456 | CV |
   |--------|---------|----------|----------|----|
   | Vanilla | 16.50 | 15.80 | 16.40 | X% |
   | TCR | 16.40 | 14.93 | 16.09 | 5.0% |
   | ... | ... | ... | ... | ... |

2. **增加 1-2 个生成样例对比**：在正文或附录中展示 2-3 个 prompt 的 baseline vs TCR vs entropy 生成文本（选择 PPL 改善最大和最小的 prompt），帮助读者直观理解"横向漂移"现象。

3. **在 Phase 2 补充 Best-of-N (K=2, K=3) 在 Dream-7B 上的结果**：使用 GPT-2 PPL 作为 scoring function（与 Methods 描述一致）。这可以作为一个计算开销的 calibration point：如果 Best-of-N (2x compute) 也无效，则进一步支持"DLM 生成方差不足以支撑 selection-based 改善"的论点。

4. **统计功效分析补充假设**：报告配对 PPL 差异的标准差（从 Full-scale 数据可以直接计算），并据此验证 power > 0.80 的声明。

5. **为 LLaDA-8B 实验补充统计检验**：报告 paired t-test p-value 和 Bootstrap CI，或明确标注此实验为"exploratory / preliminary"并在结论中相应弱化。

6. **温度退火补充消融**：至少报告固定 tau=0.3 和 tau=0.5 的 Full-scale 结果，以区分"退火策略无效"和"平均温度选择不当"。

7. **修正 Best-of-N scoring function 的描述不一致**：Section 5.2.1 表述为"selecting the best of K independent generations by model confidence"，但 Methods 4.5.3 说用 GPT-2 PPL。需要澄清 Phase 1 和 Phase 2 的 scoring function 是否相同，如果不同，需要说明原因。

## 与 Method 章节的一致性问题

- **Degeneracy rate 阈值不一致**：Method Table 1 定义为 diversity < 0.5，Experiments Section 5.1 定义为 diversity < 0.3。必须统一，否则 Phase 1（使用 0.5 阈值）和 Phase 2 的结果不可比。
- **Qwen3-0.6B 模型性质矛盾**：Methods 称其为"autoregressive models"，Experiments 称其为"masked diffusion model fine-tuned from Qwen3"。这是一个事实性错误，必须修正。
- **Best-of-N scoring function 矛盾**：Methods 说 GPT-2 PPL，Experiments Phase 1 暗示用 model confidence。需要统一。
- **评估框架重复**：Methods 4.2 和 Experiments 5.1 的内容重叠度约 70%。建议将评估框架统一放在 Experiments 5.1（更自然的位置），Methods 中仅引用。
- **温度 baseline 描述**：Methods 明确说 tau=0.4（经 pilot 选定），Experiments 说"model default"。如果 0.4 就是 model default 则需要明确说明；如果不是，则有矛盾。
