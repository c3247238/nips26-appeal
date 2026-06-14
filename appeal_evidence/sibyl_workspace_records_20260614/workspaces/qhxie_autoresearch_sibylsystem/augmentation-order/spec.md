# 项目: augmentation-order

## 研究主题
Does the order of data augmentation operations affect image classification performance? A systematic study on augmentation ordering effects in modern CNNs and ViTs.

## 背景与动机
Data augmentation is a standard technique in image classification training. Common pipelines (AutoAugment, RandAugment, TrivialAugment) apply multiple transforms sequentially, but the effect of ordering these operations has been largely overlooked. Some operations (e.g., color jitter followed by crop vs. crop then color jitter) may interact differently with the image content, leading to subtle but consistent accuracy differences. Understanding this could lead to better augmentation pipeline design principles.

## 初始想法
1. Train the same model (ResNet-18 on CIFAR-10) with augmentation pipelines in different orderings (geometric first vs. color first vs. mixed)
2. Compare: fixed order vs. random order shuffling at each step
3. Extend to ViT-small to check if architecture changes the sensitivity to ordering
4. Hypothesis: geometric transformations (crop, flip) should precede color transformations for better performance

## 关键参考文献
- AutoAugment: https://arxiv.org/abs/1805.09501
- RandAugment: https://arxiv.org/abs/1909.13719
- TrivialAugment: https://arxiv.org/abs/2103.10158

## 可用资源
- GPU: 2x on default SSH server (8x RTX PRO 6000 Blackwell 97GB available)
- 服务器: default
- 远程路径: /home/qhxie/sibyl_system

## 实验约束
- 实验类型: 轻量训练
- 数据集: CIFAR-10（完整集，50k训练图像）
- 模型规模: ResNet-18, ViT-Small
- 单次实验时间预算: 15-30 分钟
- Pilot: CIFAR-10 子集 5000 张，5 epoch，ResNet-18

## 目标产出
- 论文（NeurIPS 2026 workshop 或主会议 position paper）

## 特殊需求
- 每个实验尽量控制在 30 分钟以内
- 优先跑 ResNet-18，ViT-Small 作为扩展实验
