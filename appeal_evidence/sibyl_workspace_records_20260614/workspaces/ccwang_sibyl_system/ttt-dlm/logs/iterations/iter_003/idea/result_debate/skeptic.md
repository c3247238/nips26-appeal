# 怀疑派分析：DTA 全规模实验中期结果的统计与方法论审查

## 总体判断

当前结果严重不完整（4/7 方法完成 Countdown，GSM8K 仅 1 个 seed），且**核心方法 DTA 尚无任何数据**。在这种情况下，任何关于 DTA 有效性的判断都是纯粹的推测。以下逐项审查已有数据中的统计问题、潜在混淆因素和缺失证据。

---

## 1. 统计严谨性问题

### 1.1 样本量与检验力不足

Countdown 仅 500 题 x 3 seeds = 1500 总样本（每条件 500 样本，但只有 3 个独立随机种子）。关键问题：

- **3 seeds 不是 3 个独立实验**。每个 seed 的 500 个样本共享同一个随机数序列，seed 之间的方差可能被 seed-specific artifacts 驱动，而非方法本身的效果。
- DMI 的标准差为 1.4%（在 3 个点上），但 vanilla 的标准差仅 0.6%。这意味着 DMI 结果的**变异性是 vanilla 的 2.3 倍**——seed 选择对 DMI 结果的影响远大于对 vanilla 的影响。
- 以 DMI 9.3% vs Vanilla 4.7% 的效应量 (4.6pp)，在 500 样本上 McNemar test 应该有足够的检验力。但**我们还没看到实际的 McNemar p 值**——只有 means 和 stds，完全没有 paired 统计检验结果。

**要求**：必须报告每对方法的 McNemar test p 值 + Bootstrap 95% CI，而非仅报告均值和标准差。

### 1.2 多重比较未校正

提案中承诺 Bonferroni 校正，但当前报告完全没有提及。7 种方法的 pairwise 比较产生 21 个假设检验，Bonferroni 校正后 α = 0.05/21 ≈ 0.0024。即便 DMI 的效应量看起来很大，是否能通过这个严格阈值完全未知。

### 1.3 缺乏 Effect Size 报告

只报告准确率绝对值，没有 Cohen's h、odds ratio 或其他标准化效应量度量。4.7% vs 9.3% 在低基线上看起来是"翻倍"，但绝对提升仅 4.6pp——在推理任务的大尺度上这是否有实际意义需要讨论。

---

## 2. DMI 结果的替代解释

DMI (9.3%) 是当前唯一看起来有实质性提升的方法。但以下替代解释必须排除：

### 2.1 Logit 注入可能引入了隐式温度调节

DMI 将上一步 logits 的 softmax 加权 embedding 注入当前步输入。这本质上在输入端加入了一个**软化的先验分布**。这可能等价于一种隐式的温度退火——不是因为"跨步记忆"有效，而是因为 soft embedding 平滑了输入表征，减少了 mask token 的分布尖峰。

**验证实验**：用随机 logits（从同一模型但不同序列生成）做 DMI，看是否有类似提升。如果有，说明 DMI 的收益来自"软化"而非"记忆"。

### 2.2 DMI 可能只是增加了有效输入信息量

DMI 将离散 mask embedding 替换为连续的 soft embedding。这在信息论意义上增加了每个位置的**输入比特数**——模型看到的不再是 [MASK] 这 1 bit 信息，而是整个词表上的概率分布。这等价于在标准去噪的基础上额外提供了一个"提示"，无论该提示是否来自"前步记忆"。

**验证实验**：用 oracle soft embedding（从 ground truth 生成）做 DMI，看上界。如果 oracle DMI 的提升远大于实际 DMI，说明当前 DMI 的信息利用率很低。

### 2.3 Seed 456 的异常值

DMI 在 seed 456 上得到 10.6%，显著高于 seed 42 的 7.8%。2.8pp 的 seed 间差异（relative 36% 变化）表明结果可能高度依赖于特定样本子集的难度分布。Vanilla 的 seed 间变异仅 1.2pp (relative 25%)。

---

## 3. ReMDM-conf 和 RCR 的"失败"需要质疑

### 3.1 ReMDM-conf 均值低于 Vanilla

ReMDM-conf (4.4%) < Vanilla (4.7%)，虽然差异仅 0.3pp，但方向是错的。如果 remasking 方法在比 vanilla 更差的方向上波动，说明：

- Countdown 任务可能不适合 remasking（任务特异性问题）
- 或者 ReMDM-conf 的超参数（confidence threshold、remask ratio）未被调优

**关键问题**：ReMDM-conf 的超参数是否做了 grid search？还是直接使用了原论文的默认值？如果是后者，这不是公平比较——ReMDM-conf 是针对其他任务（如 text infilling）优化的。

### 3.2 RCR 的微小提升 (5.7% vs 4.7%) 统计显著吗？

1pp 的提升在 500 样本上，McNemar test 的检验力极低。这极有可能是 p > 0.1 的不显著差异。如果 RCR 不显著，那所谓的"Level 0 < Level 1 (DMI) < Level 2 (SCP) < Level 3 (DTA)"的谱系结构在底部两层就已经不成立了。

---

## 4. GSM8K 数据几乎不可用

### 4.1 只有 1 个 seed，无法做任何统计推断

Vanilla 29.6% (s42) 和 ReMDM-conf 25.1% (s42, 仅 350/1319 完成) 这两个数字：

- 不能做 paired 比较（只有 1 个 seed）
- ReMDM-conf 甚至还没跑完（27% 进度）
- 29.6% vanilla 与 Dream 论文报告的数据一致，只能说明评估框架没有明显 bug，但不能说明任何关于 DTA 的事情

### 4.2 ReMDM-conf 在 GSM8K 上的退化信号

25.1% < 29.6% (vanilla)，如果最终确认 ReMDM-conf 在 GSM8K 上也低于 vanilla，这将严重质疑 H2（DTA + ReMDM-conf 互补性假设）的前提——如果 ReMDM-conf 本身就有害，叠加 DTA 能否挽救？

---

## 5. 缺失的关键证据

### 5.1 DTA 本身的数据完全缺失

作为核心贡献的 DTA 方法，目前**没有任何实验结果**。所有关于 DTA 有效性的讨论都是基于提案中的预期效应量 (+5-10pp)。这是最根本的问题：我们在讨论一个还没有结果的方法。

### 5.2 计算开销未量化

DMI 的 9.3% 相比 Vanilla 4.7% 看起来不错，但：

- DMI 的实际推理时间是多少？提案说"~1.05x"，但没有实测数据。
- 如果 DMI 实际上因为 soft embedding 计算带来了 1.5x 开销，那性价比就不如声称的那么好。
- 同理，SCP 和 DTA 的实际开销必须实测——提案中的 "~1.5x" 和 "~2.5x" 是估算。

### 5.3 没有逐样本分析

在 500 个 Countdown 样本中，DMI 是在哪些类型的问题上提升的？

- 是否只在简单问题上将"差一点对"变成"对了"？
- 还是在困难问题上也有提升？
- 如果 DMI 的提升集中在简单问题上，这可能是低信息量的改善——简单问题上的准确率本来就容易波动。

### 5.4 没有 Token 级诊断数据

提案中承诺的 Correction Precision/Recall 完全没有报告。没有这些数据，无法判断：

- ReMDM-conf 为什么不如 vanilla——是 remask 了正确 token？还是 remask 数量不够？
- DMI 的提升来自哪些 token 位置的改善

### 5.5 没有退化检查

H10 承诺检查 DTA 不引入文本退化（n-gram 重复率、词汇多样性），但当前连 DMI 的退化指标都没报告。

---

## 6. Proxy Metric Gaming 检查

### 6.1 Countdown 作为主基准的代表性存疑

Countdown 是一个极其特殊的任务——4 个数字通过加减乘除凑目标数。这个任务的特性：

- 解空间极小（有限组合）
- 纯数字推理，无自然语言理解需求
- Dream-7B 的 vanilla 准确率仅 4.7%，远低于 Dream 论文声称的 16.0%

**最后一点尤其令人警觉**：如果我们的 vanilla 基线 (4.7%) 与论文报告 (16.0%) 差距如此之大，说明我们的评估 pipeline 可能存在问题（prompt 格式不同？评估脚本错误？样本集不同？去噪步数不同？）。如果基线就是错的，所有相对改善都不可信。

### 6.2 "~2x 改善"的幻觉

Summary 中称 DMI 提供"~2x improvement"（9.3% / 4.7% ≈ 2x）。这种相对改善的表述在低基线上极具误导性——从 4.7% 到 9.3% 的绝对提升仅 4.6pp，在 Countdown 的 500 题上约等于多答对 23 道题。这个效应量在统计检验中可能远没有"2x"听起来那么令人信服。

### 6.3 SCP 的 interim 结果不可参考

SCP "~8.4% at 150/500 samples" 是 interim 结果。在 Countdown 这种高方差任务上，前 150 题和后 350 题的难度分布可能完全不同（如果样本是按顺序而非随机处理的）。Interim 结果可能会在完整 500 题后大幅变化。

---

## 7. 方法论层面的深层问题

### 7.1 DTA 的理论假设可能不成立

DTA 的 ELBO 单调性命题需要"L 关于 Δθ 强凸 + L2 正则化"。但：

- Transformer 的 loss landscape 几乎不可能是强凸的
- LoRA 参数化后的 loss 仍是非凸的
- 所以 ELBO 单调性**只在局部成立**，全局可能有多个极小值
- 如果 DTA 陷入了不好的局部极小值，性能可能反而下降

### 7.2 "零训练"声称存疑

DTA 声称是"zero-training"，但它在推理时做了 LoRA 参数更新——这**就是训练**。"零预训练/零微调"可能是更准确的说法，但"零训练"在审稿人眼中可能被认为是误导性表述。

### 7.3 计算开销的公平性

DTA 的 ~2.5x 开销意味着用相同计算量，vanilla 方法可以跑更多的去噪步或更多的 samples。**在固定计算预算下**，DTA 是否仍然优于 vanilla x 2.5 步？或优于 Best-of-3 vanilla sampling？这种公平比较完全缺失。

---

## 8. 额外所需实验

### 8.1 最高优先级

1. **Vanilla 基线校准**：确认 4.7% 与 Dream 论文 16.0% 的差距原因。如果是 prompt 格式或评估方法不同，所有比较都需要重做。
2. **DMI 的消融**：随机 logit 注入对照实验——排除 DMI 的收益来自"软化输入"而非"跨步记忆"。
3. **DTA 的实际结果**：在讨论 DTA 之前，先等 DTA 跑完。

### 8.2 高优先级

4. **McNemar + Bootstrap CI**：所有已完成的 pairwise 比较。
5. **逐样本难度分析**：按 Countdown 题目难度分层，看 DMI 在哪些难度区间有效。
6. **计算效率公平比较**：在相同计算预算下，DTA vs vanilla-more-steps vs Best-of-K。

### 8.3 中优先级

7. **ReMDM-conf 超参数搜索**：如果使用了论文默认值，尝试任务特定的超参数调优。
8. **Token 级 Correction Precision/Recall**：完成提案承诺的诊断指标。

---

## 总结：当前证据不支持任何强结论

| 声明 | 证据评级 | 理由 |
|------|---------|------|
| "DMI provides clear improvement" | **弱** | 无 paired 统计检验，高 seed 间变异，替代解释未排除 |
| "Pure remasking is insufficient" | **中等偏弱** | ReMDM-conf 低于 vanilla 可能是超参数问题而非方法缺陷 |
| "DTA will significantly outperform vanilla" | **无证据** | DTA 实验尚未开始 |
| "GSM8K baseline validates our framework" | **中等** | 与论文报告一致，但 Countdown 基线差距（4.7% vs 16.0%）令人担忧 |

**核心建议**：在 DTA 结果出来之前，停止对 DTA 有效性的任何乐观推测。当前最重要的事情是 (1) 确认 vanilla 基线与 Dream 论文的差距原因，(2) 完成所有方法的统计检验，(3) 等待 DTA 实际数据。
