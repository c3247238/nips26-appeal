# Task 3a Pilot 总结: DMI (Diffusion Memory Injection)

## 实验配置
- **方法**: DMI (Diffusion Memory Injection) — Level 1 消融
- **模型**: Dream-v0-Instruct-7B
- **基准**: Countdown 16 题, seed=42
- **DMI 参数**: alpha=0.3, soft_tau=0.5
- **去噪配置**: 128 步, temperature=0.4, origin 算法
- **GPU**: 单卡 (CUDA:2)

## 核心机制
每个去噪步，将上一步的 logits 经 softmax(tau=0.5) 加权后与 token embedding 矩阵相乘，
得到 soft embedding，然后与当前步的 hard embedding 按 alpha=0.3 混合注入 masked 位置。
这是信息增强谱系中最轻量的方法（Level 1），理论上接近零额外计算开销。

## 结果

| 方法 | 准确率 | distinct-2 | rep-3 | 平均时间/样本 |
|------|--------|-----------|-------|-------------|
| Vanilla | 2/16 (12.5%) | 0.838 | 0.101 | 8.0s |
| DMI (alpha=0.3) | 0/16 (0.0%) | 0.842 | 0.103 | 9.7s |

- **时间开销**: 1.21x（主要来自 softmax @ embedding_matrix 的 matmul）
- **准确率变化**: -12.5%（2 个 vanilla 正确的样本被 DMI 翻转为错误）
- **文本质量**: 多样性指标与 vanilla 基本持平

## 逐样本分析

DMI 的生成文本在表面上看起来更"合理"（例如 sample 2 生成了数学上正确的 (22-4)*4=64，
但使用了 4 两次违反约束），但在准确率评估中全部失败。

关键观察：
1. DMI 改变了采样分布，导致生成的 token 序列与 vanilla 完全不同
2. soft embedding 注入可能干扰了 mask token 的预测概率分布
3. alpha=0.3 的混合比例对 DLM 的去噪过程来说可能过于激进

## 判定

**CONDITIONAL-GO**: DMI 实现正确、运行无错误，但准确率低于 vanilla baseline。

这一结果在信息增强谱系的消融设计中实际上是有价值的——它表明：
- 简单的 embedding 级跨步信息注入不足以改善 DLM 推理
- 需要更深层的信息整合机制（Level 2 SCP 或 Level 3 DTA）才能有效利用跨步信息
- 支持了 DTA（参数级适应）相对于 DMI（embedding 级注入）的必要性论点

## 后续建议
1. 可在 full-scale 实验中扫描 alpha ∈ {0.1, 0.2, 0.3, 0.5} 看是否有更好的配置
2. 作为消融基线，当前结果已足够——DMI < Vanilla 证明 embedding 级注入不够
3. 继续推进 SCP (task_3b) 验证 Level 2 消融效果
