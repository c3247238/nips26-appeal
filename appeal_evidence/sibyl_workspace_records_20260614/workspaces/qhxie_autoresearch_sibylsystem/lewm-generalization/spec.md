# 项目: lewm-generalization

## 研究主题

Cross-domain compositional generalization of LeWorldModel: evaluate whether physical concepts learned in training environments transfer to unseen combinations via zero-shot probing and LoRA adaptation, identify failure modes and generalization boundaries.

## 背景与动机

LeWorldModel 是一个在多个物理领域（如刚体、流体、软体）上学习物理概念的世界模型。然而，它能否将这些领域的知识组合并迁移到训练时未见过的物理场景（跨域组合泛化）尚不明确。

**关键问题**：
1. 训练域中学到的物理概念（如重力、碰撞、摩擦）能否零样本迁移到跨域组合场景？
2. LoRA 快速适应能否弥补零样本的泛化差距？
3. 哪类物理概念组合最难泛化（失败模式）？泛化边界在哪里？

**学术意义**：理解世界模型的组合泛化能力，为构建更通用的物理推理模型提供实证依据。

## 初始想法

**评估框架**：
- **零样本探测（Zero-shot Probing）**：直接在跨域未见组合上测试 LeWorldModel，衡量基础泛化能力
- **LoRA 适应（LoRA Adaptation）**：用少量跨域样本进行 LoRA 微调，测试快速适应后的性能上界
- **对比分析**：零样本 vs LoRA，量化适应前后的性能差距

**实验设计思路**：
1. 构建跨域组合测试集（如重力+流体、碰撞+软体等训练未见的组合）
2. 用 LeWorldModel 原始权重直接预测（zero-shot）
3. 插入 LoRA 层后用少量样本微调，再测试
4. 分析失败案例，识别泛化边界

**预期假设**：
- LeWorldModel 在单域内泛化良好，但跨域组合上存在明显性能下降
- LoRA 能有效弥补部分但非全部泛化差距
- 某些物理概念组合（涉及不同力学规律的场景）比其他组合更难泛化

## 关键参考文献

- 暂无（系统自动检索 LeWorldModel 相关论文及跨域泛化文献）
- 待补充：LeWorldModel 原始论文 arXiv URL
- 待补充：Compositional generalization in world models 相关文献

## 可用资源

- GPU: 本地 GPU（数量待确认）
- 实验规模: 轻量（适配本地计算能力）

## 实验约束

- 实验类型: 轻量实验（单次实验 < 1 小时）
- 优先零样本评估，LoRA 微调作为辅助验证
- 先用 pilot 实验验证评估框架可行性

## 目标产出

- 学术论文（研究 LeWorldModel 跨域组合泛化的系统性评估）

## 特殊需求

- 需要访问 LeWorldModel 官方代码库和预训练权重
- 评估指标需与物理预测任务对齐（如轨迹预测误差、物理属性预测准确率）
