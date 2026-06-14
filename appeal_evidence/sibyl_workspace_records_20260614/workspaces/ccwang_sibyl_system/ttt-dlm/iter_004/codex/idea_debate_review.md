# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-10
**模型**: Codex (GPT-5)
**Thread ID**: 019cd4da-90f0-7091-a1a5-a393266ed7db

## 总体判断

这不是一个"坏想法"，但它现在更像一个高风险探索计划，而不是一份已经收敛到可稳健投稿的研究方案。最有价值的主线是 `BSD`，因为它直接建立在唯一已验证有效的 `DMI` 之上；最脆弱的部分是 `RACFG` 的效果预期、`BSD+RACFG` 的协同叙事，以及从 `Countdown` 向 `GSM8K` 的泛化承诺。

如果我是独立评审，我会给出一个偏谨慎的结论：**可以做，但必须先把"是否真的比更简单 baseline 更强"这件事钉死，否则很容易被审稿人判成 idea stacking + underpowered evaluation。**

## 评审意见

### 1. 方法论严谨性

1. `BSD` 的核心风险不是简单的 norm mismatch，而是 **输入流形错配**。概率加权 embedding 虽然落在 token embedding 的凸包里，但模型训练时见到的是"离散 token embedding + mask embedding"的输入分布，不是"连续语义叠加体"。`L2` 归一化只能对齐长度，不能对齐方向分布、协方差结构、LayerNorm 前统计量，也不能解决"多个候选 token 语义叠加后形成伪语义点"的问题。

2. `EMA` 的"收敛"目前没有理论保证。这里的更新不是线性滤波，而是"模型去噪算子 + belief feedback + EMA"的闭环非线性系统。除非你能证明某种局部压缩性、单调势函数，或者至少给出经验上的稳定域，否则 `H2` 这种"熵单调递减"更像愿望而不是理论结论。

3. `RACFG` 的跨步 `JSD` 信号未必稳定，也未必和"正确性"同向。两个高熵分布之间的 `JSD` 可能很小，但这只是"都不确定"，不是"稳定"；相反，一个错误但高度自信的错误 belief 也可能呈现低 `JSD`。换言之，`JSD` 更像"局部动态平稳性"指标，而不是"应不应该保留"的可靠判据。

4. `BSD` 还有一个更隐蔽的问题：它可能破坏注意力模式。原模型可能把 `mask embedding` 当成一个强结构性锚点；你完全拿掉它，早层 attention 可能从"等候填充"的模式变成"被半成品语义提前污染"的模式。这不是小修小补能补回来的风险。

### 2. 假设合理性

1. **H1: BSD >= 14%** 有一定逻辑，但偏乐观。从 `DMI 9.3%` 到 `BSD 14%` 是 `+4.7pp` 绝对提升，幅度已经接近 `vanilla -> DMI` 的一大块。考虑到 `BSD` 比 `DMI` 激进得多，收益不是不可能，但风险远高于 proposal 里表现出来的语气。

2. **H2: 信念熵单调递减** 不合理。好的推理系统有时必须先"推翻早期错误确定性"，这会导致局部熵回升。真正合理的版本应是"后期总体趋于收缩"或"正确样本的 belief trajectory 更稳定"，而不是全程单调。

3. **H3: k 存在最优值** 大概率为真，但科研价值有限。这更像超参数事实，不是强 hypothesis。可以作为消融结果，不应占据太多理论篇幅。

4. **H4: RACFG >= 15%** 支撑不足。前文已经有多轮 remasking/TTT 类失败史，而 `RACFG` 仍属于 inference-time intervention，只是换了 signal。除非先证明 `JSD` 对错误位置有比置信度更高的辨识力，否则这个目标偏像拍脑袋。

5. **H5/H6: >= 2pp** 在 `Countdown-500` 上统计上很脆弱。500 个样本里，2 个百分点只对应 10 个样本。没有 paired test、多随机种子和更大样本集，这个量级的差异很难讲得硬。

6. **H7: BSD + RACFG >= 18%** 过于乐观。这两个方法并不一定正交，反而可能都在干预同一类"不稳定 belief"问题。默认"可加成"，但更现实的先验应该是"可能互相抵消，或 RACFG 只在无 BSD 时有效"。

7. **H8: GSM8K 泛化** 是最弱的假设。`Countdown` 更接近结构化搜索/数值约束任务；`GSM8K` 牵涉自然语言理解、长链推理、答案格式鲁棒性。很多在 Countdown 上有效的 test-time trick 到 GSM8K 会直接失效。

### 3. 实验设计

1. 总预算 `58 GPU·h` 对"探索"够，对"定论"不够。如果要做等算力对比、多随机种子、A-CFG 复现、DMI baseline、BSD/RACFG 多超参消融，再加 GSM8K，58 GPU·h 很容易不够，尤其 `RACFG` 还可能有 `2x` 前向开销。

2. `Phase 1` 的 `4 GPU·h` 太薄。想同时做 A-CFG 复现、BSD pilot、RACFG pilot、DMI baseline，这更像 smoke test，不足以做 go/no-go 决策。

3. 当前实验最大的问题不是"少"，而是 **不可识别**。如果 `BSD + RACFG` 有提升，你很难说清楚是 belief-state 有效、guidance 有效、schedule 有效、还是纯粹增加计算量有效。审稿人最容易攻击的点就是：**你到底在验证新原理，还是在堆 inference heuristics？**

4. 关键消融还不够完整。至少要覆盖：`belief` 的稀疏度（full vs top-k）、是否做温度缩放、`EMA alpha`、注入比例 `lambda_t`、何时开始 hard reveal、在哪些 layer 注入、`JSD` 的窗口长度、是否做 temporal smoothing、以及最重要的 **等 FLOPs 下 vanilla-more-steps / more-samples / Best-of-N / A-CFG / DMI** 对比。

5. 统计检验需要改。建议主指标用 paired bootstrap 或 McNemar，而不是只报单点准确率；同时给 `Wilson CI`。对 `H5/H6` 这种 2pp 级别小差异，要么增大样本，要么降级成 exploratory claim。

### 4. 竞争定位

1. 相对 `MetaState`，差异化是真实的。"training-free memory" 是一个明确卖点。

2. 相对 `A-CFG`、`CORE`、`HEX`，差异化目前只成立了一半。`RACFG` 本质上还是"换一个置信/稳定性信号 + 换一个 schedule"。如果没有非常干净的 equal-compute gain，外部审稿人很可能把它看成 incremental variant，而不是新范式。

3. 可能忽略的不是某一篇论文，而是一整类更危险的对手。最危险的 baseline 是：`更多 diffusion steps`、`更多随机采样`、`schedule ensemble`、`seed ensemble`、`verifier/reranker`、以及任何"保存/重用前步 hidden state 或 logits"的 inference-time persistence 方法。如果这些简单方法在等算力下接近甚至超过你，novelty 叙事会迅速塌掉。

### 5. 被忽略的风险

1. `L2` 归一化不足以解决 `BSD` 的 OOD。它最多修长度，不修流形、统计量和语义叠加失真。

2. `RACFG` 的 `2x` 前向开销很可能被"vanilla 增步"吃掉。在 diffusion 模型里，额外一步常常比复杂 guidance 更稳。必须给出 **accuracy-compute Pareto curve**，否则优势不可信。

3. `BSD` 的连续 belief 演化确实可能导致注意力崩溃。尤其在早期高不确定阶段，混合 embedding 可能让模型既不像看见 token，也不像看见 mask，导致早层 representation 漂移。

4. 从 `Countdown` 到 `GSM8K` 的泛化目前过于乐观。应降级为 bonus result，除非先证明机制不是"任务特定的数值搜索增益"。

### 6. 改进建议

1. **把主线收缩成一个核心问题：跨步 belief 保留是否真的有效。** 建议把 `BSD` 作为主贡献，`RACFG` 降为次贡献或附加实验。现在两件事同时推，失败后很难知道是谁的问题。

2. **不要一开始就"完全替代 mask embedding"。** 先做保守版：`x_t = (1 - lambda_t) * e_mask + lambda_t * b_t`，并让 `lambda_t` 随时间增长。这会把 `BSD` 从"硬替换"变成"受控插值"，更接近 `DMI` 的成功经验，也更容易调试。

3. **把 belief 做成 top-k sparse belief，不要 full-vocab 混合。** 这既减轻 OOD，又减少"语义糊化"。同时可比较 `top-2 / top-4 / top-8`。

4. **先验证 JSD 有没有判别力，再谈 RACFG。** 单独做一个分析：JSD、熵、margin、max-prob 谁更能预测最终错误位置？如果 JSD 的 AUROC / rank correlation 不显著优于简单置信度，RACFG 应直接止损。

5. **强制采用等算力主评测。** 每个方法都报告：
   - 与 `vanilla-more-steps` 等 FLOPs 对比
   - 与 `more-samples / Best-of-N` 等 FLOPs 对比
   - wall-clock 对比
   - accuracy-compute Pareto curve
   没有这组图，论文说服力会明显不足。

6. **设立明确止损标准。** 例如：
   - `BSD` 在 3 个随机种子下若不能稳定超过 `DMI` 至少 `+1.5pp`，停止扩展
   - `JSD` 若不能优于熵/置信度 signal，停止 `RACFG`
   - `BSD+RACFG` 若在等算力下不优于 `vanilla-more-steps`，不再写协同故事

7. **调整论文策略。** 最稳的投稿路径不是"我有两个都很强的新方法"，而是：
   - 主结论：跨步 belief retention 在 MDLM reasoning 中是有效的
   - 次结论：很多看似合理的 inference-time scaling 技术在 MDLM 上并不稳
   - 附加：RACFG 作为探索性扩展
   这样即使组合效果一般，仍然有一篇扎实的 empirical/mechanistic paper。

## 评分

**6/10**

理由：
1. 有真实信号。`DMI` 已经给了一个非零锚点，说明"跨步信息保留"不是空想。
2. `BSD` 有研究价值。它比前几轮那些 TTT/remasking 迁移更贴近 MDLM 的生成机制。
3. 但当前版本对 `RACFG`、协同效应、跨任务泛化都偏乐观。
4. 最关键的是：**实验设计还不足以支撑你想讲的强故事**。在现在这个形态下，支持它作为内部高风险探索项目，但不会评为"已经具备稳定外投条件"的成熟提案。
