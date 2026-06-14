# Iteration 5 新颖性检查报告

**日期**: 2026-03-18
**检查范围**: 2025-2026 年发表的 WD 相关论文是否与本项目核心贡献重叠

---

## 1. 核心贡献的新颖性状态

### 贡献 1: 统一 Phi 调制框架
**状态**: 新颖性保持

无论文提出跨越 WD scheduling / alignment-aware / decoupled / norm-matched 四个轴的统一数学框架。最接近的竞争者：
- CWD (ICLR 2026): 仅 alignment-aware 一个轴，binary mask
- AlphaDecay (2025): 仅 norm-matched 一个轴，module-wise
- CPR (NeurIPS 2024): 仅 norm-matched 一个轴，约束优化
- ADANA (2026): 仅 scheduling 一个轴，对数时间

**风险**: 低。每篇论文都聚焦单一轴，无人尝试统一。

### 贡献 2: BEM / CSI / AIS 标准化指标
**状态**: 新颖性保持

无论文提出跨方法对比的标准化评测框架。OUI (arXiv:2504.17160) 是最接近的，但它仅是一个诊断工具（过拟合-欠拟合指示器），不是跨方法比较指标。

**风险**: 低。

### 贡献 3: Phi Invariance Conjecture（AdamW 下动态 WD 无效）
**状态**: 新颖性保持，但理论解释需要更新

多篇新论文从不同角度间接支撑了这一猜想：
- D'Angelo et al. (NeurIPS 2024): WD 从不作为显式正则化有用
- Kosson et al. (ICLR 2026): WD 主要作用是控制 relative updates
- Fan et al. (2025): 稳态 singular value spectrum ∝ sqrt(eta/lambda)
- GradientStabilizer (2025): Adam 对 WD 不敏感

**但无论文**直接提出和验证 "所有动态 WD 变体在 AdamW 下等价" 的系统性结论。我们的 14 种变体 x 2 数据集 x 3 种子的系统性实验是独特的。

**风险**: 中低。理论基础已被他人建立（ELR 吸收机制），但系统性实验验证是我们的独特贡献。

### 贡献 4: 大规模可视化分析
**状态**: 新颖性保持

无论文提供覆盖所有主要 WD 方法的系统性可视化（weight norm trajectories, gradient-weight alignment, spectral density, ELR, coupling stability）。

**风险**: 低。

---

## 2. 新颖性增强建议

### 2.1 纳入 CPR 作为第五个 WD 轴

**建议**: 在 Phi 框架中增加 "constrained WD" 作为第五个子类，证明 CPR 的硬约束等价于分段 Phi 函数：
```
phi(w) = 1,           if ||w|| < tau
phi(w) = 0,           if ||w|| >= tau
```
这将 CPR 自然纳入框架，增强统一性的说服力。

### 2.2 利用 GradientStabilizer 结果

**建议**: 设计 "GradientStabilizer + 动态 WD" 的交互实验。如果 GS 消除了 WD 敏感性但动态 WD 仍在 SGD 下有效，则提供了更精确的 Phi invariance 边界条件。

### 2.3 引入 sqrt(eta/lambda) 稳态分析

**建议**: 利用 Fan et al. 的经验发现，证明在 sqrt(eta/lambda) 稳态下，Phi modulation 对 effective singular value spectrum 的影响是 second-order perturbation，从而解释 Phi invariance 的数学机制。

---

## 3. 总体新颖性评估

| 贡献 | 新颖性 | 风险 | 建议 |
|------|--------|------|------|
| 统一 Phi 框架 | 高 | 低 | 纳入 CPR 作为第五轴 |
| BEM/CSI/AIS 指标 | 高 | 低 | 无变化 |
| Phi Invariance Conjecture | 中高 | 中低 | 加强理论解释（ELR 吸收 + sqrt(eta/lambda)） |
| 系统性可视化 | 高 | 低 | 无变化 |
| ImageNet 大规模验证 | 中 | 低 | 确保涵盖 ResNet-50 和 ViT |

**总体评估**: 本项目的核心新颖性在 Iteration 5 仍然成立。2024-2026 年的新论文主要是**强化了理论基础**（ELR 吸收、scale invariance、rotational equilibrium），而非直接竞争统一框架的定位。最大的战略调整是将论文从 "提出新方法" 重新定位为 "提供理论统一和系统性分析"。
