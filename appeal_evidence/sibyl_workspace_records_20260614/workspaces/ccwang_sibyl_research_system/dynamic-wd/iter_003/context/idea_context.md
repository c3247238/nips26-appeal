# Idea Context for Unified Dynamic Weight Decay Framework

## Research Topic
统一动态权重衰减框架（Unified Dynamic Weight Decay Framework）：
将 WD scheduling、alignment-aware WD、decoupled WD、norm-matched WD 等方法统一到一套理论框架中，
提出标准化评测体系（Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score），
通过系统性数学推导和大规模可视化分析，揭示当前领域的核心问题并指明发展方向。

## Key Requirements
- 数据集: CIFAR-10, CIFAR-100, ImageNet
- 架构: ResNet-20, VGG-16-BN, ResNet-50 (ImageNet), Vision Transformer
- 多种子: 至少 3 个种子（42, 123, 456），报告 mean ± std
- 计算资源: 8x RTX PRO 6000 Blackwell (98GB each)
- 重点: 大量公式论证 + 可视化分析图表

## Previous Iteration Findings (Iterations 0-2)
1. Budget Equivalence: mean(λ_t) = fixed λ → identical performance
2. LR-WD Coupling Necessity: Decoupling causes catastrophic collapse (92% → 10%)
3. Alignment Signal Inapplicability: grad-weight alignment uninformative at nonconvex scale

## Literature Survey (28 references)
See literature.md for full details. Key categories:
- WD Scheduling: SWD, ADANA, interaction with LR schedules
- Alignment-Aware WD: CWD (binary sign-alignment), AdamO (radial/tangential), SPD (layer-wise)
- Decoupled WD: AdamW, Lp-norm generalization, Huber decay
- Norm-Matched WD: AdamWN, AlphaDecay, gamma^2 scaling, EMA timescale

## 7 Research Gaps
1. No unified theoretical framework connecting all four sub-approaches
2. No standardized evaluation metrics
3. Continuous alignment modulation unexplored
4. Mathematical connections between sub-approaches uncharacterized
5. Interaction with non-Euclidean optimizers unknown
6. No systematic visualization/diagnostic framework
7. Scale-dependent behavior poorly understood
