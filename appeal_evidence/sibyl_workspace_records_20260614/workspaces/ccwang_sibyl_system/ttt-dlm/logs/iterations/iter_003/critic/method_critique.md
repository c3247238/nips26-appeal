# Method 章节评审意见

## 评分: 7/10

## 优点

1. **结构清晰，覆盖全面**：7 类方法按统一框架呈现（算法伪代码 + 动机 + 变体 + 与其他方法的关系），读者可以快速理解每种策略的核心思想和相互联系。Section 4.11 的方法分类表尤其有用，从信号源、是否修改参数、计算开销三个维度做了精炼总结。

2. **"轻量级"边界定义明确**：Section 4.1 明确排除了需要训练、需要外部验证器、或超过 2x 计算开销的方法，为负面结果的解释力设定了清晰的 scope。

3. **伪代码规范**：每种方法都配有 Algorithm 伪代码，形式统一，便于复现。TCR 的 stability score 公式（Section 4.6.2）和 entropy 公式（Section 4.7.2）定义严谨。

4. **评估框架设计合理**：Section 4.2 的多指标体系（PPL + diversity + degeneracy rate）直接回应了 Phase 1 发现的 PPL-gaming 问题，体现了方法论上的自我修正。统计检验方案（配对 t 检验 + Wilcoxon + Bootstrap CI + Bonferroni）足够严格。

5. **方法间关系阐述充分**：每个方法末尾的 "Relationship to Other Methods" 小节将本文方法置于更广泛的文献背景中（ReMDM、PG-DLM、RemeDi 等），有助于读者理解贡献定位。

## 不足

1. **评估框架（4.2）与方法描述混合放置**：Section 4.2 详细描述了模型选择、指标体系、统计检验、实验规模——这些内容本质上属于 Experiments 章节的 Setup。将其放在 Methods 中导致两章之间大量重复（对比 experiments.md 的 Section 5.1，几乎相同的表格和描述重复出现）。这违反了学术写作中 Methods 与 Experiments 的标准分工。

2. **TTT 描述过于粗略**：Section 4.10 列出 6 个 TTT 变体，但仅给出名称和一句话描述，没有具体的超参数设置（学习率、TTT 步数、LoRA rank 等）。相比之下，TCR 和 entropy-guided remasking 的消融配置写得很细。如果 TTT 是本文评估的重要方法之一，应给予同等的细节深度；如果因为 TTT 仅在 Phase 1 使用而简化，应明确说明。

3. **Parallel Voting 的失败分析放置不当**：Section 4.9.3 "Failure Analysis" 包含了具体的实验结果（66% disagreement rate、PPL +114% 等），这些是实验发现而非方法描述。Methods 章节应描述方法本身，失败分析应留给 Experiments。

4. **缺少方法组合的讨论**：Section 4.8.4 提到温度退火"could in principle be combined with" remasking 方法，但全文未探索任何方法组合。对于负面结果论文，至少应在 Methods 中说明为何未测试组合策略，或将其列为局限性。

5. **Best-of-N 的 scoring function 选择未充分论证**：Section 4.5.3 使用 GPT-2 PPL 作为 scoring function，但这与评估指标完全相同，构成循环评估（用 PPL 选择 -> 用 PPL 评估）。应讨论这一选择的局限性，或解释为何不使用模型自身的 likelihood（虽然 DLM 的 likelihood 不可解析，但应明确说明）。

6. **符号一致性问题**：Section 4.1 用 $T$ 表示总步数、$t$ 从 $T$ 到 $1$ 递减，但 Section 4.6.2 的 stability score 公式中 $t$ 的求和范围 $\sum_{t=T-w+1}^{T}$ 与 Algorithm 3 中 step 2 的 $\sum_{t=1}^{w}$ 不一致——前者是步数编号，后者是窗口内的索引。需要统一。

7. **"Healthy Range" 定义缺乏依据**：Table 1 中 PPL < 50 和 Bigram Diversity > 0.90 的"健康阈值"没有给出来源或理由。这些阈值看似经验性的，但在后续实验中被用作质量门控标准，需要说明其依据。

## 具体修改建议

1. **将 Section 4.2（评估框架）移至 Experiments 章节**，Methods 中仅保留一段简要说明（"我们采用跨架构 PPL + 多样性指标 + 配对统计检验的评估框架，详见 Section 5.1"），避免与 experiments.md Section 5.1 的大量重复。

2. **补充 TTT 的超参数细节**：至少给出 learning rate 范围、TTT 步数、batch size、LoRA rank（如适用）。如果篇幅限制，可将详细配置移至附录，但 Methods 中应有足够信息让读者理解每个变体的关键差异。

3. **将 Section 4.9.3 的实验数据移至 Experiments**：Methods 中保留方法描述和"可能的失败模式"的理论分析，但具体数字（66% disagreement、PPL +114%）应出现在实验结果中。

4. **统一 stability score 的数学符号**：建议在 Algorithm 3 和 Section 4.6.2 中使用相同的索引体系，例如统一用 $t \in \{T-w+1, \ldots, T\}$ 表示最后 $w$ 步。

5. **增加一段关于方法组合的讨论**（~100 字）：说明为何本文未测试 TCR + 温度退火等组合策略（例如："组合策略的搜索空间呈指数增长，超出本文的计算预算；此外，单一方法均未显示显著效果，组合不太可能产生质量跃升"）。

6. **为 Best-of-N 添加循环评估的讨论**：在 Section 4.5.3 末尾添加一段，承认 scoring function 与评估指标相同的问题，并论证这在本文语境下是合理的（因为 Best-of-N 作为 oracle upper bound，用相同指标选择恰好给出了最有利的结果，如果连这样都无效，则更强地支持负面结论）。

7. **为 "Healthy Range" 提供依据**：引用相关文献或说明是基于本文 baseline 实验的经验观察。

## 与 Experiments 章节的一致性问题

- **重复内容**：Methods 4.2 与 Experiments 5.1 的模型列表、指标表、统计方法描述高度重复，需要整合（见上述建议 1）。
- **Degeneracy rate 阈值不一致**：Methods Table 1 定义 degeneracy 为 diversity < 0.5，但 Experiments Section 5.1 的指标表定义为 diversity < 0.3。需要统一。
- **温度基线不一致**：Methods 4.3.2 说 baseline 温度为 0.4（基于 pilot 实验），但 Experiments 5.1 说"Temperature is fixed at the model default unless otherwise noted"。需要明确 0.4 是否为 model default。
- **Phase 1 模型描述差异**：Methods 4.2.1 将 Qwen3-0.6B 描述为"autoregressive models used as base for TTT experiments"，但 Experiments 5.1 将其称为"masked diffusion model fine-tuned from the Qwen3 architecture"。前者暗示 AR 模型，后者明确为 DLM。需要统一。
