# 修正主义分析：AADWD 实验结果

## 核心立场

AADWD 的「失败」不是方法的失败，而是问题设定（problem formulation）的失败。实验数据指向了一个比原始提案更深刻的研究方向：**weight decay 在标准 SGD 训练中的动态无关性（dynamic irrelevance）**。本文提出对原始研究框架的根本性修正。

## 修正一：对齐信号的量级诊断

原始框架假设 delta_t（梯度-参数对齐度）具有足够的动态范围来驱动有意义的 WD 适应。实验揭示 delta 约 10^{-3}，导致 conservative 公式中 (1-delta) 约 0.999。这个数字本身就是一个重要的经验发现。

**修正后的理解**：在标准 CIFAR 训练中（ResNet20, 200 epoch, milestone LR schedule），梯度方向与参数方向之间的平均对齐度极低。这并非训练异常——在高维空间中，两个随机向量的余弦相似度期望接近零（维度 d 越高，期望越接近 0）。ResNet20 的参数空间维度约 270K，在这种尺度下，delta 约 10^{-3} 恰恰是统计预期。

**修正建议**：如果要让对齐信号「可行动」，需要在 delta 本来就较大的场景中使用——例如 fine-tuning（参数接近预训练方向，梯度可能高度对齐）、few-shot learning（数据稀少导致梯度方差大）、或者 layer-wise 而非 global 的对齐度计算。

## 修正二：重新定义「动态」的层次

原始框架在 per-step 级别调整 WD（每个 mini-batch 更新一次 lambda_t）。实验表明这种高频适应被 gamma_t（LR schedule）完全主导——LR 在 milestone 处的 10 倍跳变完全淹没了 delta_t 的 O(10^{-3}) 量级变化。

但 Stagewise WD（92.44%）的表现暗示了一个可行的中间方案：**epoch-level 或 phase-level** 的 WD 适应。在 milestone 前后，训练动态有质的变化（learning rate 下降 10 倍，loss landscape 变得更平坦），WD 的最优值可能确实不同。实验中 stagewise WD 的成功正是因为它在正确的时间尺度上进行了调整。

**修正建议**：将 AADWD 从 per-step 适应重新设计为 phase-level 适应，在每个 LR milestone 之间用累积对齐统计量决定下一阶段的 WD 值。这避免了 gamma_t 的主导效应，同时利用了更稳定的相位平均对齐度。

## 修正三：等价累积定理的理论化

Equivalent Cumulative WD 实验（92.54% = 92.54%）产出了一个可以形式化为定理的经验发现：

**猜想（WD 均值等价性）**：对于 milestone-based LR schedule + step-decayed 训练，如果两种 WD 调度 {lambda_t} 和 {lambda'_t} 满足在每个 LR 阶段内的时间平均相等，即对所有阶段 k，(1/T_k) * sum_{t in phase_k} lambda_t = (1/T_k) * sum_{t in phase_k} lambda'_t，则它们的最终泛化性能差异为 O(1/T_k)（其中 T_k 是该阶段的步数）。

这个猜想如果成立，将彻底改变 WD 调度的理论框架：从「寻找最优时间序列 {lambda_t}」简化为「寻找每个阶段的最优均值 {bar_lambda_k}」。这是一个从无限维搜索空间到 K 维搜索空间的降维（K = LR milestones 数量，通常 2-3）。

**修正建议**：将此猜想作为论文的核心理论贡献，用现有的等价累积实验作为支撑证据，并设计 2-3 个额外的简单实验来验证（例如，在不同阶段使用不同均值但总均值相同的 WD 调度）。

## 修正四：LR-WD 耦合作为设计原则

解耦实验的灾难性结果（aggressive 崩溃至 10.00%，conservative 退化至 80.30%）不应仅仅被理解为一个失败案例，而应该被提升为一个正面的设计原则：

**原则：在 L2-equivalent WD 中，LR-WD 耦合是稳定性的必要条件。**

这个原则的含义远超 AADWD 本身。它解释了为什么：
1. PyTorch 的默认 SGD 实现使用 L2 正则化（天然 LR-WD 耦合）而非 decoupled WD
2. AdamW 的成功部分来自于 Adam 自身提供了类似的隐式缩放（通过自适应学习率）
3. 周期性 LR schedule（cosine annealing）与固定 WD 的组合如此稳健——因为 LR 的变化自动调节了 effective WD

**修正建议**：将此原则作为论文的第二核心贡献，提供 formal 分析证明在什么条件下解耦会导致正反馈不稳定性。

## 修正五：CWD 崩溃的机制解释

CWD 在三个设定上的一致崩溃（CIFAR-10/ResNet20: 91.79%->86.95%, CIFAR-100: 66.84%->54.27%, VGG16: 92.95%->86.47%）值得深入分析。CWD 的 coordinate-wise decay 在训练晚期（LR 已降至 0.001）时，仍然对每个坐标施加 sign-based 的衰减。在低 LR 阶段，梯度更新量很小（~LR * grad = 0.001 * grad），而 WD 的绝对量不变（lambda * w = 5e-4 * w），导致 WD 相对于梯度更新的比例急剧上升。coordinate-wise 机制在这种情况下无法自我抑制。

**修正建议**：这一分析可以扩展为对所有 sign-based WD 方法的通用警告，具有方法论贡献。

## 修正后的论文框架

**标题**：「Weight Decay Dynamics in Nonconvex SGD: Mean Sufficiency, Coupling Necessity, and the Failure of Alignment-Based Adaptation」

**三个核心贡献**：
1. **均值充分性定理**（Mean Sufficiency）：WD 的时间平均量决定泛化性能，时间分配模式在统计精度内不影响最终结果（实验证据：92.54% = 92.54%）
2. **耦合必要性原则**（Coupling Necessity）：LR-WD 耦合是 L2-equivalent WD 稳定性的必要条件，解耦导致正反馈崩溃（实验证据：解耦后 aggressive 10.00%, conservative 80.30%）
3. **对齐信号量级分析**（Alignment Scale Analysis）：标准 SGD 训练中 delta 约 10^{-3}，不足以驱动有意义的 per-step 适应；这是高维参数空间的固有属性（理论分析 + 经验验证）

## 明确判断

**应该继续此研究方向，但需要根本性地重构论文叙事。** 不是「我们提出 AADWD 但它不行」，而是「我们发现了 weight decay 动态的三个基本定律」。现有数据完全足够，可能仅需 2-3 个小型补充实验（均值等价性在不同阶段的验证）。
