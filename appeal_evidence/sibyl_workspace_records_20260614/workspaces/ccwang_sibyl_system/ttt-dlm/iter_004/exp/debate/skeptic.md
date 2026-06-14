# 怀疑论者审查报告

**角色**: 最大怀疑态度审视实验结果
**审查对象**: Full-scale 验证（101 prompts x 3 seeds）及整体研究方向
**日期**: 2026-03-08

---

## 0. 最关键问题：评测指标的根本缺陷

**实验全程使用 GPT-2 (124M) PPL 作为唯一质量指标。这个选择使得"所有方法均无效"的结论本身就不可靠。**

回顾研究日记 Iteration 015 的发现：PPL 可以被退化文本 gaming（重复 token 降低 PPL），同时也可以因为生成了"不同但同样合理"的文本而升高 PPL。PPL 衡量的是"AR 模型认为文本有多可预测"，而非"文本质量有多高"。

一个 124M 参数的 GPT-2 模型作为 7B 扩散模型输出的裁判，本身就是荒谬的。GPT-2 的语言理解能力远低于 Dream-7B——让一个能力更弱的模型去评判能力更强的模型的输出，这相当于让小学生给大学教授的论文打分。

**因此，full-scale 实验的"NOT SIG"结论可能完全是指标选择的产物，而非方法本身的失败。**

---

## 1. 实验设计的系统性缺陷

### 1.1 评估指标是致命伤

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| GPT-2 作为 judge | 致命 | 124M 评判 7B 输出，能力不对等 |
| 仅用 PPL | 致命 | 没有 GSM8K/MMLU/HumanEval 等 benchmark |
| 无人类评估 | 严重 | 文本质量需要人类判断 |
| 无 LLM-as-judge | 严重 | 至少应该用 GPT-4/Claude 评分 |
| diversity 不是质量指标 | 中等 | diversity=0.967 只说明文本不重复，不说明文本好 |

**关键质疑**：Iteration 015 已经证明 PPL 在 DLM 上是不可靠的——0.6B 模型上 PPL 改善 -47.5% 实际是退化文本。那为什么在 Dream-7B 上仍然信任 PPL 的"不显著"结论？PPL 不可靠是双向的：它既可能虚报改善，也可能漏报改善。

### 1.2 Pilot 设计从根本上就是错误的

Pilot 仅用 16 prompts x 1 seed。对于 DLM 这种高方差随机生成模型：
- 16 个样本的统计功效（statistical power）接近于零
- 单 seed 完全无法控制随机变异
- Pilot prompts 全部是"简单科学问题"——选择偏差极其严重

更深层的问题：**pilot 的目的是筛选方法进入 full-scale**。如果筛选器本身是噪声主导的，那么被筛选出来的方法大概率只是噪声受益者。这是一个典型的"赢家诅咒"（winner's curse）——pilot 中表现最好的方法，恰恰是最可能回归均值的。

### 1.3 温度退火 pilot 数据存在异常

检查 `task2b_temp_annealing.json`：`anneal_lin_06_02` 和 `anneal_cos_08_02` 的 PPL 数组几乎完全相同（15 个值中 14 个一模一样），且生成的文本也高度重叠（photosynthesis、DNA/RNA 等文本完全相同）。这暗示：
- 要么退火策略根本没有改变生成过程（温度变化太小）
- 要么代码有 bug，不同配置实际上跑了相同的参数
- 无论哪种情况，pilot 中 anneal 的 -16.5% 改善本身就值得质疑

---

## 2. PPL 在 Dream-7B 上是否仍然不可靠？

**答案是极其可能不可靠，但方向不同于 0.6B。**

在 0.6B MDLM 上，PPL 不可靠的方向是**虚报改善**（退化文本 gaming）。在 Dream-7B 上，PPL 不可靠的方向可能是**漏报改善**：

1. **Dream-7B 的 baseline PPL 极高**（median 16.23, mean 30.1）。作为对比，0.6B MDLM 的 baseline PPL 仅 3.8。这说明 GPT-2 本来就不理解 Dream-7B 的输出风格。

2. **mean 和 median 差距巨大**：vanilla mean=30.1 vs median=16.23（比值 1.85x），entropy mean=41.8 vs median=16.15（比值 2.59x）。这种极端偏斜说明存在大量 PPL 异常值，统计检验在这种分布上的可靠性大打折扣。

3. **GPT-2 的 tokenizer 与 Dream-7B 不同**。交叉架构 PPL 在 tokenizer 不匹配时可能产生系统性偏差。

### 应该用什么指标？

| 指标 | 优先级 | 理由 |
|------|--------|------|
| GSM8K accuracy | 最高 | 直接衡量推理能力改善 |
| MMLU accuracy | 最高 | 知识覆盖面 |
| HumanEval pass@1 | 高 | 代码能力 |
| AlpacaEval (GPT-4 judge) | 高 | 指令遵循质量 |
| 人类盲评 A/B 对比 | 高 | 黄金标准 |
| MAUVE score | 中 | 分布级别文本质量 |
| Self-BLEU + Distinct-n | 辅助 | 多样性控制 |

**PPL 应该降级为辅助指标，决不能作为方法是否有效的唯一判据。**

---

## 3. Pilot vs Full-scale 差异分析

### 3.1 数据对照

| 方法 | Pilot PPL(med) | Full PPL(med) | Pilot 改善 | Full 改善 | 消失幅度 |
|------|---------------|---------------|-----------|----------|---------|
| entropy_r20_mean | 12.88 | 16.15 | -24.9% | -0.5% | 24.4pp |
| tcr_r30_s32 | 13.22 | 15.77 | -22.9% | -2.8% | 20.1pp |
| anneal_lin_06_02 | 14.32 | 17.55 | -16.5% | +8.1% | 24.6pp |

三个方法的消失幅度惊人地一致（20-25 个百分点）。这不太像是随机波动——随机波动会产生不同幅度的变化。

### 3.2 可能的深层原因

**假说 A：Prompt 分布偏移**
Pilot 用了 16 个"简单科学问题"（machine learning, photosynthesis, DNA vs RNA, black holes 等）。这些问题有确定性答案，模型生成的文本内容高度规范。Full-scale 的 101 prompts 可能包含更多开放性、主观性问题，而重遮蔽方法对开放性问题无效——因为没有"正确答案"可以趋近。

**假说 B：Pilot baseline 异常偏高**
Pilot baseline PPL(med) = 17.15，Full-scale baseline PPL(med) = 16.23。Pilot 的 baseline 更差，给改善留出了更多空间。如果 16 个 pilot prompts 恰好是模型初始生成质量较差的样本，那么重遮蔽改善了最差的那些——但这只是回归均值，不是方法有效。

**假说 C：seed=42 恰好是幸运种子**
Pilot 只用了 seed=42。检查 full-scale 中 seed=42 的 per-seed 结果：
- TCR: s42=16.40（最差）
- Entropy: s42=15.90（不是最好）
- Anneal: s42=18.39（最差）

这很有意思——seed=42 在 full-scale 中反而表现最差。这暗示 pilot 的"改善"可能是 seed=42 在小样本上的偶然波动，而不是 seed 本身的特性。

**假说 D：代码或评估 pipeline 在 pilot 和 full-scale 间有变化**
如果 pilot 和 full-scale 使用了不同版本的代码、不同的 GPT-2 加载方式、不同的 tokenizer 设置，那么结果差异可能是工程问题，而非方法问题。这需要验证。

---

## 4. IGIR 是否会面临同样命运？

**几乎可以确定会。**

### 4.1 IGIR 的核心组件分析

IGIR = IB 引导自适应重遮蔽 + MH 接受准则 + 退火调度

让我逐一质疑：

**IB 引导（信息瓶颈 Score）**：
- IB Score 本质上还是用模型自身的 logits/entropy 来决定重遮蔽哪些 token
- 与 entropy_r20_mean 的区别仅在于理论包装更精致
- Entropy 引导在 full-scale 上已经失败（-0.5%, p=0.530）
- 换一个理论框架来计算相同的信号，不会改变信号本身的信息量

**MH 接受准则**：
- MH 接受/拒绝需要一个可靠的"能量函数"
- 如果能量函数是 PPL——我们已经证明 PPL 不可靠
- 如果能量函数是模型自身的 confidence——Iteration 010 已经证明 model confidence 与 AR PPL 反相关
- 没有可靠的能量函数，MH 框架再优美也是空中楼阁

**退火调度**：
- anneal_lin_06_02 在 full-scale 上不仅无效，还恶化了 PPL (+8.1%)
- IGIR 中加入退火调度不会改变这个事实

### 4.2 更根本的问题

所有这些方法共享一个前提假设：**DLM 去噪过程中存在可以通过后处理修复的错误模式**。

但 Dream-7B 的 trajectory stability 高达 ~0.96（TCR 实验数据），意味着模型在去噪过程中的预测已经非常稳定。如果模型对 96% 的 token 都有稳定预测，那"修复不稳定 token"的策略能改善的空间只有 4%。即使这 4% 全部修复成功，对整体质量的影响也微乎其微。

**IGIR 不会比 TCR/Entropy/Anneal 好到哪里去，因为它们操作的是同一个已经高度稳定的去噪过程。**

---

## 5. 研究方向是否应该放弃？

### 5.1 需要区分两个问题

**问题 A："推理时改进 DLM" 这个方向本身是否有价值？**
有。但前提是我们能正确评测。用 PPL 得出的"无效"结论本身就不可信。在用 benchmark 评测之前，下任何结论都为时过早。

**问题 B："用当前的方法论框架（PPL 评测 + remasking 变体）继续" 是否值得？**
不值得。我们已经在同一个框架下尝试了 6+ 种变体（ReMask-Retry, TCR, Entropy Remasking, Temperature Annealing, Parallel Voting, Block-Reset TTT），全部在 PPL 上不显著。再增加 IGIR 只是在同一条死路上多走一步。

### 5.2 如果继续，必须满足的条件

1. **立即切换到 benchmark 评测**：GSM8K accuracy 是最低要求。没有 benchmark 结果的实验不应该被运行。
2. **验证 PPL 假阴性的可能性**：取 full-scale 中 entropy_r20_mean 的生成文本，用 GPT-4 做 pairwise 质量评判。如果 GPT-4 认为 entropy 输出更好但 PPL 没有反映，那问题在指标不在方法。
3. **只有在 benchmark 上看到 >1% 的改善才值得继续深入**。

### 5.3 如果放弃，价值在哪里

连续两轮 full-scale 实验（ReMask-Retry + 本轮 4 种方法）都是负面结果，这本身就是有价值的贡献：
- 证明简单的 post-hoc remasking 策略在 Dream-7B 上不产生 PPL 改善
- 证明 pilot 实验在 DLM 上的不可靠性（winner's curse）
- 证明 PPL 作为 DLM 质量指标的双向不可靠性

但这些负面结果的前提是**我们有可靠的评测指标来确认方法确实无效**。目前我们没有。

---

## 6. 总结：最严厉的判决

### 我们确定知道的

1. PPL 在 DLM 评测中是不可靠的（Iteration 015 已证明）
2. Pilot 实验（16 samples x 1 seed）对 DLM 方法没有筛选能力
3. Dream-7B 的去噪轨迹已经高度稳定（stability ~0.96），后处理空间极小

### 我们不确定的

1. **方法是否真的无效？** 在没有 benchmark 评测的情况下，"NOT SIG" on PPL 不能等同于"方法无效"
2. **是指标的问题还是方法的问题？** 如果 entropy remasking 改善了推理质量但 GPT-2 PPL 没有反映，我们会错误地放弃一个有效方法
3. **Dream-7B 的高 baseline PPL 是否说明 GPT-2 根本不理解它的输出？** Vanilla PPL=16.23 vs MDLM-0.6B 的 PPL=3.8，差距 4x 以上

### 行动建议（优先级排序）

1. **[紧急] 用 GSM8K 跑一次 vanilla vs entropy_r20_mean 对比**。50 题足矣。如果 accuracy 有变化，说明问题在指标；如果没变化，说明方法确实无效。这一步不做，所有后续判断都没有基础。

2. **[紧急] 用 Claude/GPT-4 做 20 对 pairwise 盲评**。让大模型判断哪个输出质量更高。成本极低但信息量极高。

3. **[条件] 只有在 #1 和 #2 都确认方法无效后，才决定是否放弃方向或继续 IGIR。**

4. **[无论如何] 把"PPL 在 DLM 上不可靠"写成论文的核心贡献。** 这个发现比任何改善方法都更有学术价值。

---

## 附录：温度退火数据异常

`anneal_lin_06_02` 和 `anneal_cos_08_02` 的 PPL 数组：

```
lin_06_02: [13.51, 17.05, 11.96, 20.27, 63.15, 15.04, 12.25, 13.86, 19.82, 8.39, 5.75, 11.19, 84.35, 40.84, 14.32]
cos_08_02: [28.18, 17.05, 11.96, 20.27, 63.15, 15.04, 12.25, 13.86, 19.82, 8.39, 5.75, 11.19, 84.35, 40.84, 14.32]
```

14/15 个值完全相同。这两个配置应该使用不同的温度调度曲线，不可能产生几乎完全相同的输出。**强烈怀疑 pilot 代码中温度退火逻辑未正确生效**，实际运行的可能都是同一个配置。如果确认是 bug，那 anneal 从一开始就不应该进入 full-scale。
