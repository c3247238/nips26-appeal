# 实用主义者视角（Pragmatist Perspective）

## 核心判断

**研究方向值得推进，但当前的数据质量不足以支撑任何假说结论。** 现有 Tier 1 实验全部是 pilot 模式（10 epochs, 100 样本子集, 单 seed），模型远未收敛——ResNet-18 在 CIFAR-10 上仅 ~10% 准确率（接近随机猜测），ViT 在 CIFAR-100 上仅 ~2.8%。在模型尚未学到有意义表征的情况下，所有关于 ordering 效果的结论都不可靠。**必须先用完整训练（200 epochs, 全量数据, 多 seed）重跑 Tier 1，才有资格讨论任何假说。**

---

## 一、对实验数据的工程审计

### 1.1 致命问题：模型未收敛

| 实验块 | 最佳准确率 | 预期完整训练准确率 | 差距 |
|---|---|---|---|
| ResNet-18 × CIFAR-10 | 10.97% | ~94-95% | **84%+** |
| ResNet-18 × CIFAR-100 | 46.63% | ~77-78% | ~31% |
| ViT-Small × CIFAR-10 | 19.70% | ~80-85% | ~65% |
| ViT-Small × CIFAR-100 | 2.89% | ~58-60% | ~56% |

**ResNet-18 在 CIFAR-10 上 10% 准确率 = 10 类随机猜测水平。** 这意味着模型几乎没有从数据中学到任何东西。在这个阶段观察到的 ordering "效应"（0.96% spread）完全可能是：
- 初始化随机性（只有 1 个 seed）
- 优化器在 loss landscape 早期的随机游走差异
- 100 样本子集本身的采样偏差

数据增强的效果主要体现在训练后期——当模型开始过拟合时，augmentation 通过增加数据多样性来正则化。**在模型远未过拟合的 10 个 epoch 里，augmentation ordering 的差异基本不可能被 SGD 的学习动态放大到统计显著的水平。**

### 1.2 统计验证完全缺失

- **t_stat 和 p_value 全部为 null**：单 seed 无法计算任何统计检验
- **Cohen's d 被人为膨胀**：报告中的 Cohen's d > 2.0 看起来很大，但这是因为 spread 被单点估计的方差（接近零）除，没有实际意义
- **所有"confirmed"和"falsified"判定都基于不可靠的单点估计**

### 1.3 Baseline 数据与 ordering 数据不可比

Baseline 结果显示 ResNet-18 × CIFAR-10 达到 91.91%（conventional ordering），而对应的 ordering 实验结果是 10.19%。两者显然不在同一实验设置下运行。Baseline 可能用了完整数据 + 更多 epochs，ordering 实验用的是 100 样本 + 10 epochs。**在论文中将这两组数据放在同一张表里是不可接受的——必须确保所有比较在相同设置下完成。**

---

## 二、对假说状态的务实评估

### H1（ordering 显著影响准确率）：**无法判定，需要重新验证**

- 当前"confirmed"判定不成立：在未收敛模型上的 0.96% spread 没有统计意义
- 但我对最终结果**谨慎乐观**：PBA 的 epoch-level schedule ablation 已经显示 ordering 在某种粒度上有效果，per-image ordering 合理推断也可能有效果
- **关键问题**：效果大小在完整训练后可能缩小（augmentation ordering 的影响被 200 epochs 的 SGD 平均化）还是放大（augmentation 在后期训练更重要）？这只有跑了才知道

### H2（DPI 可逆性排序优于传统排序）：**2/4 = 硬币翻面概率，不算确认**

- CJ→Flip→Crop vs. Crop→Flip→CJ 在 2/4 块中胜出 = 50% 胜率 = 零信息量
- 即使在完整训练后，"2/4 blocks confirmed" 的门槛也设得太低——4 个比较中赢 2 个不能排除随机性
- **建议**：将 H2 的确认门槛提高到 3/4 blocks 或 paired t-test p < 0.05

### H3（NC_2 预测准确率排序）：**合理的 falsification，但 SWD proxy 本身可疑**

- Spearman rho = -0.20 确实很弱
- 但 SWD（Sliced Wasserstein Distance）作为 W_2 的 proxy 在低维（32×32 图像）上的近似质量本身值得怀疑
- **实用建议**：如果要在论文中保留 NC_2 理论贡献，需要明确区分"理论 bound 是否成立"（数学问题，与实验无关）和"SWD proxy 能否在实践中预测"（工程问题）。前者可以保留，后者已失败

### H4（InfoNCE MI 与准确率相关）：**CIFAR-10 正相关，CIFAR-100 负相关 = 噪声**

- rho = +0.54 和 rho = -0.66 在两个数据集上方向相反，combined rho = -0.06
- 这不是"不确定"，这是"没有关系"
- **建议**：在论文中简洁报告为 negative result，不要过度解读

### H5（高 magnitude 放大 ordering 效果）：**M14 spread = 0.00% 很有意思，但可能是 bug**

- M5→M9 spread 递增（0.35%→0.88%）符合直觉
- M14 突然归零 = 要么高 magnitude 导致训练不稳定（所有 ordering 都同样崩溃），要么存在数值 bug
- **必须检查**：M14 条件下所有 ordering 的绝对准确率是否都很低且接近——如果是，说明 M14 破坏了训练信号

---

## 三、Tier 2 类别级排序数据的可信度

Tier 2 报告的 9.01% spread 是整个研究中最大的信号，但：
- 同样基于 5k 子集 pilot
- 使用了 6 个 ops（Crop, Flip, Rotation + ColorJitter, Grayscale, GaussianBlur），操作集比 Tier 1 大得多
- Best（interleaved P→G, 29.39%）vs. worst（all-geo-first, 20.38%）的差距确实很大
- **但 29% 的绝对准确率说明模型同样远未收敛**

如果这个 9% spread 在完整训练后仍然存在（即使缩小到 2-3%），这将是论文最强的发现。**建议将 Tier 2 提升为主实验之一，而不是附属实验。**

---

## 四、对论文框架的实用建议

### 4.1 强叙事（如果数据支持）

**"类别级排序效应远大于操作级排序效应"** 是最有实用价值的发现。实践者不会关心 Crop 和 Flip 谁先——他们关心的是"所有空间变换一起做 vs. 所有颜色变换一起做，哪个更好？"

建议的叙事结构：
1. 操作级 ordering 有统计显著但较小的效果（~0.5-1%）→ 回答 RQ1
2. 类别级 ordering 效果更大（~2-5%）→ 主要实用贡献
3. 架构差异存在但不剧烈 → 回答 RQ2
4. NC_2 理论 bound 成立但 SWD proxy 不够 → 诚实的理论贡献
5. 实用推荐：interleaved P-G ordering 是一个合理的默认值

### 4.2 论文定位

**NeurIPS 2026 workshop paper 是现实的目标。** 主会议 position paper 需要更强的实验证据（ImageNet-scale）和更深的理论贡献。当前的实验规模（CIFAR-10/100, ResNet-18, ViT-Small）适合 workshop。

如果要冲主会议，需要：
- 至少在 ImageNet subset（或 Tiny ImageNet 224×224）上验证
- NC_2 bound 要有更好的 empirical proxy，或者放弃 NC_2 换一个更实用的理论框架
- 5 seeds × 完整训练的 Tier 1 + Tier 2 数据

### 4.3 时间和资源现实估计

| 任务 | GPU-hours | 墙钟时间 (2 GPUs) | 阻塞性 |
|---|---|---|---|
| 重跑 Tier 1 完整训练（6 orderings × 2 arch × 2 dataset × 5 seeds = 120 runs） | ~20 | ~10h | **必须做** |
| 重跑 Tier 2 完整训练（5 orderings × 2 arch × 2 dataset × 5 seeds = 100 runs） | ~18 | ~9h | **强烈建议** |
| 重跑 Tier 3 magnitude（2 orderings × 3 magnitudes × 5 seeds = 30 runs） | ~6 | ~3h | 可选 |
| Tier 4 NC_2 + MI（纯计算，无训练） | ~1 | ~30min | 可选 |
| **总计** | **~45** | **~23h** | |

**实际预期**：2 块 RTX PRO 6000 Blackwell（97GB），每个 ResNet-18 CIFAR-10 训练 200 epochs 约 5-8 分钟，ViT-Small 约 15-20 分钟。120 个 Tier 1 runs 可以在 10 小时内完成。

---

## 五、对现有提案的具体修改建议

### 5.1 降低理论期望，提高实验质量

原提案中理论贡献（NC bound + DPI principle）占了很大篇幅，但 H3 已被 falsify，H4 不确定。**建议将论文重心从"theory-grounded study"调整为"systematic empirical study with theoretical analysis"。** 理论贡献作为 Section 3 保留但降低为辅助角色，实验结果作为 Section 4-5 的核心。

### 5.2 操作集需要扩充

当前只用了 3 个操作（Crop, Flip, ColorJitter），3! = 6 个排列。这太少了：
- 6 个数据点上的统计检验 power 很低
- 审稿人会质疑 3 个操作能否代表实际管道

**建议增加到 4-5 个操作**：加入 RandomRotation 和 GaussianBlur（或 RandomErasing）。4 个操作给出 24 个排列，统计 power 显著提升，同时计算成本增加 4 倍（可控）。

### 5.3 明确区分"pilot 信号"和"confirmed result"

validation_decision.md 中将 pilot 结果称为"confirmed"是不准确的。在论文和后续流程中，必须区分：
- **Pilot signal**: 10 epochs, 100 samples, 1 seed → 仅作为"是否值得投入完整实验"的决策依据
- **Confirmed result**: 200 epochs, 完整数据集, 5 seeds, paired t-test p < 0.05 → 可以在论文中声称

### 5.4 "Best ordering" 在不同块之间不一致

- ResNet-18 × CIFAR-10 最佳：CJ→Flip→Crop
- ResNet-18 × CIFAR-100 最佳：Flip→Crop→CJ
- ViT-Small × CIFAR-10 最佳：Flip→CJ→Crop

没有一个 ordering 在所有块中都是最佳的。这要么意味着 ordering 效果确实是 architecture/dataset dependent（有趣的发现），要么意味着这些差异全是噪声（因为是 pilot 数据）。**完整训练后必须重新验证一致性。**

---

## 六、最终建议

### 必须做（阻塞论文写作）

1. **重跑 Tier 1 完整实验**：200 epochs, 完整数据集, 5 seeds。这是论文的数据基础，不可跳过
2. **重跑 Tier 2 类别级排序**：完整训练。如果 9% 的 pilot spread 在完整训练后仍有 2%+ 留存，这将是论文的杀手级发现
3. **修复 baseline/ordering 数据不一致**：确保所有比较在同一设置下完成

### 应该做（显著提升论文质量）

4. 增加操作数到 4 个（加 RandomRotation），4! = 24 排列
5. 调查 M14 spread = 0 的原因（检查训练曲线和绝对准确率）
6. 将论文框架从 "theory-first" 调整为 "empirical-first with theoretical analysis"

### 可以不做（时间不够就跳过）

7. 跑 ImageNet 或 Tiny ImageNet 验证
8. 优化 NC_2 的 empirical proxy（换一种距离度量替代 SWD）
9. 可微排列学习（Gumbel-Sinkhorn）——工程复杂度太高，ROI 不值得

---

## 七、风险与预期结果的务实评估

| 场景 | 概率 | 应对策略 |
|---|---|---|
| 完整训练后 ordering spread > 0.5% 且统计显著 | 45% | 按原提案推进，聚焦实验贡献 |
| 完整训练后 ordering spread 0.2-0.5%，勉强显著 | 30% | 聚焦类别级 ordering（Tier 2）作为主贡献；操作级作为辅助发现 |
| 完整训练后 ordering spread < 0.2%，统计不显著 | 20% | 转为 negative result paper："Augmentation Order Doesn't Matter" — 仍然可发表（见提案中的 N0 假说） |
| 类别级 ordering 效果很强（2%+）但操作级不显著 | 5% | 重新定位论文为"Category-Level Augmentation Pipeline Design" |

**底线**：这个研究方向无论结果如何都能产出一篇论文。关键是**不要在未收敛的 pilot 数据上写论文**——投入 23 小时 GPU 时间跑完整实验是绝对必要的。
