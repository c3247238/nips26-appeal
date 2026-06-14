**核心方向**: Systematic comparison of training-free acceleration methods for diffusion language models — evaluate orthogonality and composability of KV caching, adaptive step scheduling, AR-guided unmasking, and speculative decoding on LLaDA-8B-Instruct across reasoning and coding benchmarks.

**四种方法**:
1. **KV Caching**: 复用相邻扩散步之间相似的 KV 对，减少重复注意力计算
2. **Adaptive Step Scheduling**: 动态调整扩散步数，高置信 token 早确定，低置信 token 多步精化
3. **AR-guided Unmasking**: 用自回归模型指导 mask token 的解码顺序或候选生成
4. **Speculative Decoding**: 用小模型草稿 + DLM 大模型并行验证，利用 DLM 天然并行性加速

**核心研究问题**:
- 各方法单独能带来多少加速（speed-accuracy tradeoff）？
- 哪些方法可以正交组合（互不干扰地叠加加速）？
- 组合后的上限加速比是多少？
- 在推理（reasoning）和代码（coding）任务上表现是否一致？

**目标模型**: LLaDA-8B-Instruct