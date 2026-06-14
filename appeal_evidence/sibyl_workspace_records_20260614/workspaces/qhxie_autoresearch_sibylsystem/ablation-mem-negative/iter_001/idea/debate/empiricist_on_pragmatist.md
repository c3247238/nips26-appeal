# Round 1: Empiricist 对 Pragmatist 的批评

## 评分：7/10

## 批评要点

### 1. 跨架构基准测试（CAAB）的实验设计过于乐观
Pragmatist提出的CAAB实验设计存在几个未充分考虑的问题：
- **训练非JumpReLU SAE的时间被低估**：在Gemma-2-2B上训练一个16k宽度的TopK/BatchTopK SAE需要数小时，不是"快速训练"。5种架构×4层=20个SAE，总训练时间可能超过20小时。
- **Gated SAE的训练复杂性**：Gated SAE需要特殊训练流程（交替优化门控和幅度），在SAELens中的支持不如JumpReLU成熟。
- **公平比较困难**：不同架构的最优超参数不同，如何确保比较公平？

### 2. 层间传播分析的方法论缺陷
- **Activation patching的因果解释力有限**：修改第L层后观察第L+1层变化，但无法区分"直接因果影响"和"间接相关变化"。
- **WordNet层次结构到神经网络特征的直接映射过于简化**：WordNet是人工构建的语义层次，神经网络中的"层次"可能是不同的结构。

### 3. 时间预算不合理
- Full实验总计6小时的估计过于乐观
- 仅SAE训练就可能需要20+小时
- 建议：使用预训练SAE（如GemmaScope的JumpReLU + 社区提供的其他架构SAE），避免自行训练

## 建议改进
1. 优先使用预训练SAE，减少训练开销
2. 将CAAB限制在2-3种架构（JumpReLU、BatchTopK、Matryoshka）
3. 层间传播分析使用更简单的"相关性分析"替代activation patching
