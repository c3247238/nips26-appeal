# 怀疑论者分析：实验结果批判性审查

**Agent**: 怀疑论者（Skeptic）
**日期**: 2026-03-25
**审查对象**: 动态权重衰减框架（EqWD）实验结果

---

## 总体立场

当前结果存在多项系统性缺陷，不足以支撑关于 EqWD 优越性的强力结论。以下逐条列出核心疑虑，按严重程度排序。

---

## 1. 45 Epochs ImageNet 训练严重不足（严重）

**问题**: ImageNet 标准训练为 90 epochs，当前仅训练 45 epochs，相当于标准的一半。

- 45 epochs 时大多数模型尚未完全收敛，不同方法的相对排名可能在后期发生逆转。
- 正则化方法（如 WD scheduling）的效果通常在训练后期才充分体现——网络权重范数开始饱和时，WD 的调节作用才更为关键。EqWD 的核心优势（budget-equivalent 调度）可能被截短训练掩盖或人为放大。
- 学术界的 ImageNet ResNet-50 基准普遍采用 90 epochs（Top-1 ~76-77%），而当前 EqWD 仅达 72.27%，表明训练根本未达收敛。在如此欠训练的条件下，任何方法的排名都缺乏代表性。
- **结论**: 必须补充 90 epochs 完整训练结果，否则 ImageNet 实验不具备投稿说服力。

---

## 2. EqWD vs SWD 差距微弱，统计显著性存疑（严重）

**问题**: EqWD（72.27 ± 0.20）vs SWD（72.04 ± 0.40），绝对差距仅 **+0.23%**。

- 两者置信区间存在大量重叠：EqWD 范围约 [72.07, 72.47]，SWD 范围约 [71.64, 72.44]。重叠区间宽达 0.37%，远大于均值差距 0.23%。
- 3 个种子样本量极小，无法进行可靠的统计检验。标准做法应至少 5 个种子，并报告 p 值（t 检验或 Wilcoxon 检验）及置信区间。
- SWD 的标准差（0.40）是 EqWD（0.20）的 2 倍，说明 SWD 的最优种子可能超过 EqWD 均值。仅报告均值排名掩盖了这一关键信息。
- 0.23% 的差异在工业界视为噪声，在学术界也需要严格的统计论证才能作为贡献声明。
- **结论**: 在统计显著性未经证明之前，不能声称 EqWD 优于 SWD。

---

## 3. 多架构验证不完整，泛化能力不明（中等）

**问题**: CIFAR-100 仅报告了 ResNet-20 结果，缺少 VGG-16-BN 对应结果。

- ResNet-20 和 VGG-16-BN 在架构上差异显著：残差连接的存在使权重范数的演化模式不同；BN 层的归一化行为也与 ResNet shortcut 产生交互。
- EqWD 作为"统一框架"，理论上应在多种架构上均表现稳健。若在 VGG-16-BN 上 EqWD 排名下滑（例如 CWD 或 SWD 更优），则统一框架的声明将被大幅削弱。
- 声称框架具有"universality"而只用一种 CIFAR 架构验证，逻辑上存在根本缺口。
- **结论**: 必须补充 CIFAR-100 VGG-16-BN 的完整对比实验。

---

## 4. 缺少关键控制实验，因果机制未被隔离（中等）

**问题**: 无法区分 EqWD 的性能增益来自哪个具体机制。

- EqWD 同时引入了 budget equivalence 约束和 phase-aware scheduling。缺少以下消融实验：
  - **Phase schedule replay alone**: 仅用 EqWD 的 phase 调度曲线，不加 budget 约束，性能如何？
  - **Gradient-norm-only baseline**: 仅用梯度范数信号调整 WD，不加 budget equivalent 归一化，性能如何？
  - **Budget constraint alone**: 固定 WD 总量不变，但不做 phase-aware 调度，性能如何？
- 没有这些消融，审稿人有理由认为：EqWD 的增益可能完全来自更激进的 WD schedule（即 SWD 的变体），而非所谓的"budget equivalence"新原理。
- 这是理论 vs 实证一致性的核心问题，无法回避。
- **结论**: 至少需要 2 个消融对照组，明确分离各机制贡献。

---

## 5. 超参数敏感性风险（中等）

**问题**: beta=2.0 是否是唯一鲁棒的最优值？

- 若 EqWD 的性能对 beta 高度敏感（例如 beta=1.5 或 beta=2.5 时性能下降超过 0.3%），则说明该方法实用性有限，不是通用框架。
- 没有 beta 敏感性扫描（beta ∈ {1.0, 1.5, 2.0, 2.5, 3.0}），无法证明 beta=2.0 是"合理默认值"而非"精心调出的最优点"。
- 在不同数据集（CIFAR vs ImageNet）上，同一 beta 值是否保持最优？跨 setting 的 beta 鲁棒性未被验证。
- 超参数过拟合是 WD 研究领域的常见陷阱，审稿人必然会追问。
- **结论**: 需要提供至少 4-5 个 beta 值的对比，报告性能变化曲线。

---

## 6. 与现代优化器缺少对比（中等）

**问题**: 所有对比方法均假设 SGD + Momentum 优化器，完全回避了 AdamW 和 Lion。

- AdamW 已内置解耦 WD，是 NLP 和视觉 Transformer 的事实标准；Lion 则提出了全新的正则化范式。这两类优化器已在 ImageNet 上大量超越 SGD + FixedWD。
- 若 EqWD 在 SGD 生态中排名第一，但 AdamW with cosine schedule 的简单基线就能达到 75%+（90 epochs），则整个对比体系的 practical relevance 存疑。
- 论文定位为"统一框架"，应讨论 EqWD 原理是否可迁移到 AdamW（即 decoupled WD scheduling for Adam），而非仅限于 SGD。
- **结论**: 至少需要一个 AdamW baseline，并在 related work 中明确说明框架与 adaptive optimizer 的关系边界。

---

## 7. Novelty 风险：WD 研究领域已严重拥挤（轻微但不可忽视）

**问题**: Weight decay scheduling / adaptive WD 领域近年来发表了大量工作。

- SWD（scheduled WD）、CWD（curriculum WD）、CAWD（cosine adaptive WD）本身已是相对成熟的方向。将这些方法"统一"到一个框架，需要证明统一框架带来了独立于各方法的新洞察，而非仅仅是后验的理论整合。
- "Budget Equivalence Metric"、"Coupling Stability Index"、"Alignment Informativeness Score"这三个评测指标的提出需要充分的理论动机和实证验证——若这些指标与最终性能的相关性不够强，则其价值将受质疑。
- ICLR/NeurIPS 审稿人对"统一框架"类论文要求极高：需要框架能预测新方法、或解释现有方法的失败模式，而非仅描述它们。
- **结论**: 需要明确展示框架的预测能力（即框架能指导设计新的 WD 方法并验证其有效性），而非停留在描述和统一层面。

---

## 综合评估

| 问题 | 严重程度 | 是否可补救 |
|------|---------|-----------|
| 45 epochs 不足 | 严重 | 需补充 90 epochs 实验 |
| 统计显著性缺失 | 严重 | 需增加种子数 + p 值报告 |
| 多架构验证缺失 | 中等 | 需补充 VGG-16-BN 结果 |
| 消融实验缺失 | 中等 | 需设计 2-3 个消融对照 |
| 超参数敏感性未验证 | 中等 | 需 beta 扫描实验 |
| 现代优化器对比缺失 | 中等 | 需至少 AdamW baseline |
| Novelty 定位风险 | 轻微 | 需强化框架预测能力论述 |

**当前实验结果不足以支撑高质量论文投稿。最关键的两项补救措施为：(1) 完成 90 epochs ImageNet 训练；(2) 增加种子数至 5+ 并报告统计显著性。**

---

*怀疑论者 Agent | 动态权重衰减框架批判性审查*
