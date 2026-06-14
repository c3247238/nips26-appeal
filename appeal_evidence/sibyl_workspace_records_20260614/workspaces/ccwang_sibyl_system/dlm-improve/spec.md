# 项目: dlm-improve

## 研究主题
在 training-free 条件下改进 Diffusion Language Models (DLMs) 的生成性能，或在维持性能的同时提升 DLMs 的推理速度。

## 背景与动机
Diffusion Language Models（如 MDLM, SEDD, Dream, LLaDA, MMaDA）作为自回归模型的替代范式，具有并行生成、灵活编辑等优势，但在采样质量和推理速度上仍存在明显短板。现有加速方法（如 Block Diffusion, FAST-DLLM V2）虽有进展，但 sampling strategy 仍有较大改进空间。本项目聚焦 training-free 方法，从扩散模型的底层数学原理出发，寻找改进采样过程的理论依据和工程实现。

## 初始想法
- 从扩散过程的数学公式推导出发，分析现有 sampling strategy 的理论缺陷
- 从底层机制（noise schedule、transition kernel、denoising dynamics）寻找改进点
- 探索自适应步数/token 级别的采样策略
- 结合离散扩散过程的特殊性质设计更高效的 sampler
- 研究 confidence-based 或 entropy-based 的动态去噪策略

## 关键参考文献
- MDLM (Masked Diffusion Language Models)
- SEDD (Score Entropy Discrete Diffusion)
- Dream (Diffusion Reasoning with Enhanced Abilities for Machines)
- LLaDA (Large Language Diffusion with Attention)
- MMaDA (Multi-Modal Diffusion with Attention)
- Block Diffusion
- FAST-DLLM V2: Efficient Block-Diffusion LLM
- Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling
- 其他由文献调研阶段补充

## 可用资源
- GPU: 4x NVIDIA RTX PRO 6000
- 服务器: cs8000d (SSH MCP connection: cs8000d, fallback: default)
- 远程路径: /home/ccwang/sibyl_system
- 单个实验如有需要或者资源空闲可尝试多卡并行提升效率
- 实验要考虑大 batch size，以提升训练/推理速度，充分利用显存

## 实验约束
- 实验类型: **优先 training-free**；允许 1 小时内的 LoRA 轻量训练
- 训练任务时间预算: ~1 小时
- 评测任务: **不受时间限制**，按 benchmark 完整跑完
- 统计显著性: **不需要**。不换 seed 多跑，benchmark 合理且无过度下采样时以单次结果为准
- Pilot 采样: **最少 100 条**，不接受 n<100 的 pilot。benchmark 条数本身小于 100 条的除外
- 模型规模: 使用各 DLM 论文的开源预训练模型（通常为中小规模），建议先从小尺寸（0.6B,4B）开始

## 评测策略
- 使用社区通用 benchmark（如各 DLM 论文常用的 text8, One-Billion-Word, lambada, GSM8K 等）
- Pilot 实验可选用小型 benchmark 或 benchmark 子集（>=100 条）
- 正式实验使用完整 benchmark，不做下采样
- 对标模型: 各 DLM 的原始采样策略 + 已有加速方法（Block Diffusion, FAST-DLLM V2 等）

## 目标产出
- **顶会论文**（NeurIPS / ICML / ACL 级别）
- 质量期望: **weak accept 及以上**（borderline+）
- 论文模板: **NeurIPS LaTeX 模板**
- 论文语言: **全英文**
- 图像可视化：可以使用可视化图像来解释和表达含义思想想法的，要尝试使用可视化图像进行表示，不要拘泥于表格和折现柱状曲线图，还要考虑热力图、注意力图等解释模型内部机制的可视化方法

## 方法论要求
- **从原理公式推导**：不仅仅是工程 trick，要有数学/理论支撑
- **底层机制分析**：深入分析扩散过程的 transition dynamics、noise schedule 等
- **理论→实验**：先建立理论框架，再用实验验证理论预测
- 结合实际工程实现，确保方法可复现

## 特殊配置
- 飞书同步: 启用
- Codex 审查: 启用
- 写作模式: parallel（加速写作）
