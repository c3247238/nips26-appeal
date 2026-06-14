# 项目: dlm-acceleration

## 研究主题
基于 DLM（Diffusion Language Model）的推理加速

## 背景与动机
Diffusion Language Models（如 MDLM、SEDD 等）通过多步去噪过程生成文本，具有天然的并行生成能力，但推理速度相较自回归模型仍有差距。如何在不重新训练模型的前提下，利用 DLM 的并行性和独特的扩散过程特性加速推理，是一个重要的开放问题。

## 初始想法

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

## 关键参考文献
- LLaDA (Large Language Diffusion with mAsking): 目标测试模型
- MDLM (arxiv: 2406.07524): Masked Diffusion Language Model
- SEDD: Score Entropy Discrete Diffusion
- Speculative Decoding: SpecInfer, Fast and Slow 等
- 系统将自动搜索相关文献

## 可用资源
- GPU: 本地多张 GPU
- 服务器: default (local)
- 远程路径: /home/qhxie/sibyl_system

## 实验约束
- 实验类型: training-free / 轻量实验（无大规模训练）
- 模型规模: 小到中等（优先使用开源 DLM 检查点，如 MDLM-base 等）
- 时间预算: 单实验约 1 小时以内；Pilot 10-15 分钟
- 优先验证 training-free 方法

## 目标产出
- 完整学术论文（NeurIPS/ICML/ICLR 水平）

## 特殊需求
- 所有方法必须 training-free（不修改模型权重）
- 评测基准覆盖推理（reasoning）和代码（coding）两类任务
- 必须系统评估方法之间的正交性和可组合性（不只是单独评估各方法）
- LLaDA-8B-Instruct 作为主要测试模型
