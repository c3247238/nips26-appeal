# Iteration 1 最终提案

## 标题
Towards a Task-Agnostic Quantification of Feature Absorption in Sparse Autoencoders

## 核心贡献
1. 提出一种任务无关的 feature absorption 量化框架，不依赖于 first-letter 等特定任务
2. 在多个预训练 SAE（GemmaScope, GPT-2 SAE）上系统验证 absorption 的普遍程度
3. 量化 absorption 对下游可解释性任务（如概念探测、特征干预）的因果影响

## 方法
- 基于 SAELens 加载预训练 SAE
- 利用 LLM 自动标注特征语义，构建层次化特征对
- 设计通用指标：衡量父特征在子特征激活时的“缺失率”
- 通过特征干预实验评估 absorption 对下游任务的影响

## 预期产出
- NeurIPS/ICLR 级别论文一篇
- 开源评估代码与数据集
