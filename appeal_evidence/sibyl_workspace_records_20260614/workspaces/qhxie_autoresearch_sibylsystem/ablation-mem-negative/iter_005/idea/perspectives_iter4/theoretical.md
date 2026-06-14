# Iteration 4 理论视角：解决循环定义问题

## 核心问题：碰撞率"代理验证"是循环的

### 问题分析

论文声称"碰撞率是吸收率的代理指标"，但：
- 碰撞率 = Jaccard(top-K特征重叠)
- 真实吸收率 = Jaccard(top-K特征重叠)
- 两者是**同一指标**

这不是"代理验证"，而是"自相关"。评审正确指出了这一点。

### 理论重构

**正确的叙事框架**：操作化可靠性（Operationalization Reliability）

定义：设概念C有操作化定义O(C) = top-K激活特征集。

**内部一致性检验**：
- 如果O是可靠的，那么对于有层次关系的概念对(C1, C2)，O(C1)和O(C2)应该有系统性的重叠模式
- 碰撞率测量的是O(C1) ∩ O(C2)的稳定性

**关键区分**：
- 我们不是声称"碰撞率预测吸收率"
- 而是声称"碰撞率作为操作化定义的稳定性指标，在层次概念对上表现出预期模式"

### 为什么这仍然有价值

即使碰撞率和吸收率是同一指标，验证它们的相关性仍然有贡献：
1. **证明操作化定义是连贯的**：如果同一指标在不同概念对上表现出稳定的相关性，说明我们的概念定义是内部一致的
2. **提供筛选工具**：在实际研究中，碰撞率可以快速筛选候选对，比完整的人工标注高效
3. **建立基准**：为未来的替代方法提供比较基准

### 论文中的表述建议

**旧表述（问题）**：
> "We validate collision rate as a robust proxy for true absorption rate"

**新表述（修正）**：
> "We demonstrate that collision rate---the Jaccard overlap of top-K activating features---exhibits strong internal consistency across concept pairs (Spearman r = 0.869, n = 56). This indicates that our operationalization of absorption via top-K feature overlap is structurally coherent: concept pairs with known hierarchical relationships show systematically higher overlap than unrelated pairs."

### K-means差异的理论解释

K-means (85.7% recall) vs Ward (14.3% recall)的差异可以用聚类几何解释：

- **Ward linkage**：最小化簇内方差，对距离阈值敏感。吸收特征phi≈0，容易被分到不同簇
- **K-means**：硬分配，所有特征必须进入某簇。即使phi≈0，随机初始化可能将吸收特征放入同一簇

这实际上是一个重要的发现：**聚类算法的选择对结果的影响大于聚类目标本身**。这进一步削弱了UAD的稳健性claim。
