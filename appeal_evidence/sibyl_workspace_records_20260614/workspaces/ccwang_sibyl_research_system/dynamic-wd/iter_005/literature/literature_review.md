# Iteration 5 文献调研报告

**研究主题**: 统一动态权重衰减框架（Unified Dynamic Weight Decay Framework）
**调研日期**: 2026-03-18
**重点方向**: 最新 WD 理论进展、AdamW 隐式正则化、动态/自适应 WD、WD 与优化器交互、ImageNet 基准

---

## 1. 调研背景与 Iteration 4 遗留问题

Iteration 4 文献调研已建立 35 篇核心参考文献的知识库。Iteration 5 调研聚焦以下增量目标：

1. **补充 2025 下半年至 2026 年初新发表的论文**（Iteration 4 调研截止约 2026 年 3 月初）
2. **深化 WD 与优化器交互的理论理解**——特别是 scale invariance、ELR、rotational equilibrium 的最新进展
3. **确认 ICLR 2026 / NeurIPS 2025 / CVPR 2025 正式录用的 WD 相关论文**
4. **为 Iteration 5 实验设计（ImageNet、VGG-16-BN、多种子扩展）提供文献支撑**
5. **识别新的潜在竞争者或威胁论文**

---

## 2. 新发现论文（Iteration 4 未覆盖）

### 2.1 Robust Layerwise Scaling Rules by Proper Weight Decay Tuning

**引用**: Fan, Z., Liu, Y., Zhao, Q., Yuan, A., & Gu, Q. (2025). arXiv: [2510.15262](https://arxiv.org/abs/2510.15262)

**核心贡献**:
- 在现代 scale-invariant 架构中，muP 的学习率迁移在训练进入 optimizer-governed steady state 后会退化
- 提出 AdamW 的 weight decay 层级缩放规则：λ_matrix ∝ sqrt(d)，使 sublayer gain 在不同宽度下保持不变
- 经验发现每个矩阵参数的奇异值谱范数 ∝ sqrt(eta/lambda)，形状近似不变
- 向量参数（bias/normalization）应设 lambda=0，矩阵参数的 LR ∝ d^{-1}

**与本项目的关联**:
- **直接支撑 Phi 框架的 scale-dependent 分析**：Phi modulation 在不同宽度下的效果可能需要匹配这一缩放规则
- 提供了 sqrt(eta/lambda) 稳态关系的经验验证——可用于 BEM 指标的理论基础
- 补充 Kosson et al. (ICLR 2026) 的 WD > muP 结论，提供层级细粒度视角

**BibTeX**:
```bibtex
@article{fan2025robust,
  title={Robust Layerwise Scaling Rules by Proper Weight Decay Tuning},
  author={Zhiyuan Fan and Yifeng Liu and Qingyue Zhao and Angela Yuan and Quanquan Gu},
  journal={arXiv preprint arXiv:2510.15262},
  year={2025}
}
```

---

### 2.2 Constrained Parameter Regularization (CPR)

**引用**: Franke, J.K.H., Hefenbrock, M., Koehler, G., & Hutter, F. (2024). *Improving Deep Learning Optimization through Constrained Parameter Regularization.* NeurIPS 2024. arXiv: [2311.09058](https://arxiv.org/abs/2311.09058)

**核心贡献**:
- 提出 weight decay 的替代方案：对每个参数矩阵的统计量（如 L2 范数）施加上界约束
- 通过增广 Lagrangian 方法动态调整每个参数组的惩罚系数
- AdamCPR 在相同预算下优于 AdamW，且仅需 2/3 的 compute budget 达到相同性能
- 可对抗 grokking 现象

**与本项目的关联**:
- **CPR 是 norm-matched WD 子类的直接竞争者**：它将 "target norm" 概念实现为约束优化
- 为 CSI（Coupling Stability Index）提供了一个自然的对比基准——CPR 的 Lagrangian multiplier 动态本质上是一种自适应 coupling
- 需要在 Related Work 中讨论 CPR 与 AdamWN（Loshchilov）的关系

**BibTeX**:
```bibtex
@inproceedings{franke2024cpr,
  title={Improving Deep Learning Optimization through Constrained Parameter Regularization},
  author={J{\"o}rg K.H. Franke and Michael Hefenbrock and Gregor Koehler and Frank Hutter},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```

---

### 2.3 GradientStabilizer: Fix the Norm, Not the Gradient

**引用**: arXiv: [2502.17055](https://arxiv.org/abs/2502.17055) (2025 年 2 月)

**核心贡献**:
- 提出轻量级 gradient transform，保留瞬时梯度方向但用统计稳定估计替换更新幅度
- 证明稳定化后的幅度在 spike steps 上是均匀有界的
- 在 LLM 预训练（FP16）、量化感知训练（FP4）、ImageNet 分类、RL、时序预测中一致改善稳定性
- **大幅降低 Adam 对 weight decay 强度的敏感性**——传统 gradient clipping 反而加剧了这一敏感性

**与本项目的关联**:
- **与 Phi Invariance Conjecture 的重要交叉**：如果 GradientStabilizer 已经降低了 WD 敏感性，那么动态 WD 在 Adam+GS 下可能更不必要
- 为 AIS（Alignment Informativeness Score）提供了对照——GradientStabilizer 通过 norm stabilization 而非 alignment 来解决 WD 敏感性
- ImageNet 实验结果可作为 Iteration 5 ImageNet 基准的参考

**BibTeX**:
```bibtex
@article{gradientstabilizer2025,
  title={GradientStabilizer: Fix the Norm, Not the Gradient},
  author={Anonymous},
  journal={arXiv preprint arXiv:2502.17055},
  year={2025}
}
```

---

### 2.4 Understanding the Generalization of Stochastic Gradient Adam

**引用**: arXiv: [2510.11354](https://arxiv.org/abs/2510.11354) (2025)

**核心贡献**:
- 严格证明在 large-batch regime 下，Adam 和 AdamW 收敛到测试误差较差的解（即使有适当的 WD）
- 在 mini-batch 训练中，随机 Adam 和 AdamW 在适当 WD 下可在非凸设置中实现近零测试误差
- 揭示了 batch size 与 WD 交互的理论本质

**与本项目的关联**:
- 为 Phi Invariance Conjecture 提供了理论基础：在 large-batch regime 下 WD 的效果被抑制
- 我们的实验使用 batch size 128（mini-batch），理论预测动态 WD 应有效——但实际上无效（在 AdamW 下），说明 mechanism 不仅仅是 batch size
- 可在 Discussion 中引用，讨论 batch size 作为 Phi invariance 的调节变量

**BibTeX**:
```bibtex
@article{adam_generalization2025,
  title={Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks},
  author={Anonymous},
  journal={arXiv preprint arXiv:2510.11354},
  year={2025}
}
```

---

### 2.5 Kosson et al. — Weight Decay May Matter More Than muP（ICLR 2026 正式录用确认）

**更新**: Iteration 4 中 arXiv:2510.19093 已确认正式录用为 **ICLR 2026 Conference Paper**。

**新增细节**:
- WD 使相对更新 ∥Delta W∥/∥W∥ ∝ sqrt(eta * lambda)，与 muP 的 initialization-based scaling 不同
- 在 AdamW 下，WD 使权重范数收敛到稳态，relative updates 成为 WD 的函数
- Scale-invariant 网络中梯度始终正交于权重——WD 的唯一作用是控制权重范数从而调节 ELR

**BibTeX 更新**:
```bibtex
@inproceedings{kosson2026wd_mup,
  title={Weight Decay May Matter More Than {$\mu$P} for Learning Rate Transfer in Practice},
  author={Atli Kosson and Jeremy Welborn and Yang Liu and Martin Jaggi and Xi Chen},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

---

### 2.6 CWD — Cautious Weight Decay（ICLR 2026 正式录用确认）

**更新**: arXiv:2510.12402 已确认正式录用为 **ICLR 2026 Conference Paper**。

**新增理论细节**（从 camera-ready 版本）:
- 通过 LaSalle 不变性原理证明带 CWD 的动量算法收敛到原始目标的驻点集
- CWD 在 smooth nonconvex setting 下的离散时间 Adam 收敛率已证明
- Gemma 338M/986M/2B 上的 LLM 预训练结果：CWD 在 cosine 和 WSD schedule 下均有改善

---

### 2.7 Vision Transformers that Never Stop Learning (ARROW)

**引用**: arXiv: [2603.07787](https://arxiv.org/abs/2603.07787) (2026 年 3 月)

**核心贡献**:
- 研究 ViT 在持续学习中的可塑性丧失
- 注意力模块在浅层保持稳定但在深层变得不稳定
- 提出 ARROW 优化器：自适应学习率和衰减调度，应用于最后的注意力层

**与本项目的关联**:
- 提供了 layer-specific WD 在 ViT 中重要性的额外证据
- 与 Kobayashi et al. (NeurIPS 2024) 的 attention-specific WD 解耦建议一致
- 为 ViT 实验设计提供 layer-wise 分析思路

---

## 3. 已知论文的会议录用状态更新

| 论文 | Iteration 4 状态 | Iteration 5 更新 |
|------|-----------------|-----------------|
| CWD (Chen et al.) | arXiv 2510.12402 | **ICLR 2026 Conference Paper** 已确认 |
| Kosson et al. WD > muP | arXiv 2510.19093 | **ICLR 2026 Conference Paper** 已确认 |
| Kosson et al. Rotational Equilibrium | ICML 2024 | 无变化 |
| D'Angelo et al. | NeurIPS 2024 | 无变化 |
| Sun et al. nonconvex SGD | CVPR 2025 | 无变化 |
| CPR (Franke et al.) | --- | **NeurIPS 2024 Conference Paper**（新发现）|
| AdamHD (Guo & Fan) | arXiv 2511.14721 | 状态未变 |
| ADANA (Ferbach et al.) | arXiv 2602.05298 | 状态未变 |
| AdamO (Chen et al.) | arXiv 2602.05136 | 状态未变 |

---

## 4. 主题综合分析

### 4.1 WD 作为 ELR 调节器的理论共识（2024-2026）

Iteration 5 调研确认了一个跨多篇论文的**理论共识**：在带有 normalization layers 的 scale-invariant 网络中，weight decay 的主要作用机制是**调节有效学习率（ELR）**，而非经典的显式正则化。支撑这一共识的论文链为：

1. van Laarhoven (2017): 首次推导 ELR 假说
2. Kosson et al. (ICML 2024): 旋转平衡理论——WD 平衡层间学习
3. D'Angelo et al. (NeurIPS 2024): WD 从不作为显式正则化有用
4. Defazio (2025): WD 控制梯度-权重比到稳态
5. **Fan et al. (2025, NEW)**: 稳态奇异值谱 ∝ sqrt(eta/lambda)
6. **Kosson et al. (ICLR 2026, NEW 确认)**: WD 使 relative updates ∝ sqrt(eta*lambda)

**对 Phi 框架的启示**: 如果 WD 的核心作用已经被 ELR 调节完全解释，那么 Phi modulation（动态调整 WD）本质上等价于**隐式地调整有效学习率**。在 AdamW 中，自适应梯度缩放已经提供了 per-parameter 的 ELR 控制，因此 Phi modulation 被 AdamW 内部机制"吸收"——这正是 Phi Invariance Conjecture 的理论基础。

### 4.2 WD 缩放规则的最新进展

| 缩放规则 | 来源 | 公式 | 适用场景 |
|---------|------|------|---------|
| WD ∝ gamma^2 | Chou (2025) | 稳定权重范数 | 通用 AdamW |
| WD ∝ sqrt(d) | Fan et al. (2025) | 层级 sublayer gain 不变 | 宽度迁移 |
| WD as EMA timescale | Wang & Aitchison (2024) | optimal WD = f(epochs) | 跨规模迁移 |
| WD > muP | Kosson et al. (ICLR 2026) | sqrt(eta*lambda) 控制 relative updates | LR 迁移 |

**对实验设计的启示**: Iteration 5 的 ImageNet 实验应验证这些缩放规则在 ResNet-50 上的适用性。特别是 lambda ∝ sqrt(d) 规则是否影响 Phi modulation 的效果。

### 4.3 WD 替代方案的竞争格局

Iteration 5 识别了两个新的 WD 替代方案：

1. **CPR (NeurIPS 2024)**: 约束优化替代 WD。AdamCPR 用 2/3 compute 达到 AdamW 性能。核心思路是将 WD 的 norm control 功能实现为硬约束。
2. **GradientStabilizer (2025)**: 通过 gradient norm stabilization 消除 Adam 对 WD 强度的敏感性。不替代 WD 但大幅降低其重要性。

这两个方法从不同角度挑战了 "需要精心调优 WD" 的假设——前者用约束替代 penalty，后者降低 WD 敏感性使精确调优变得不必要。对我们论文的启示：**统一框架不仅需要统一现有 WD 方法，还需要解释为何 CPR 和 GradientStabilizer 能替代/降低 WD 的重要性**。

### 4.4 非凸 SGD 下 WD 的泛化理论

Sun et al. (CVPR 2025) 提供了首个 WD 在非凸 SGD 中泛化收益的理论证明。关键发现：
- WD **不加速** SGD 的收敛
- WD **确实改善** 泛化（首次非凸理论证明）
- 结果可扩展到 sign-based 方法（SignSGD）

结合 arXiv:2510.11354（Adam 泛化理论）的发现：
- Large-batch Adam/AdamW：即使有 WD 也收敛到差解
- Mini-batch Adam/AdamW：适当 WD 可实现好泛化

**综合结论**: WD 的泛化效果取决于 (optimizer, batch_size, scale) 三元组。Phi Invariance 可能只在特定三元组下成立。

### 4.5 ImageNet 上的 WD 实验基准汇总

| 来源 | 架构 | WD 设置 | 关键发现 |
|------|------|---------|---------|
| ResNet strikes back (2021) | ResNet-50 | 自适应 WD + label smoothing | 79.8% top-1 |
| Revisiting ResNets (2021) | ResNet-200 | 更多正则化时降低 WD | 79.0 → 82.2% |
| CPR (NeurIPS 2024) | GPT-2 | AdamCPR vs AdamW | CPR 用 2/3 budget 达到相同分数 |
| GradientStabilizer (2025) | ImageNet | Adam+GS | 降低 WD 敏感性 |
| Fan et al. (2025) | 多规模 | lambda ∝ sqrt(d) | sublayer gain 不变 |
| NOVAK (2026) | CIFAR/ImageNet | 14 优化器对比 | WD 与 alpha_eff 耦合退化 4-8pp |

**Iteration 5 ImageNet 实验设计建议**:
- ResNet-50, 90 epochs, AdamW baseline: lambda=1e-4 (标准) 和 5e-4 (与 CIFAR 一致)
- 对比 constant / CWD / cosine_schedule / no_wd
- 3 seeds (42, 123, 456)
- 额外测试 SGD+momentum 基线以验证 optimizer-dependent Phi invariance

---

## 5. 竞争威胁评估

### 5.1 高威胁

| 论文 | 威胁类型 | 影响评估 |
|------|---------|---------|
| CWD (ICLR 2026) | 直接竞争 | CWD 已是 alignment-aware WD 的标杆。但 CWD 仅使用 binary mask，我们的框架提供连续 modulation 和理论统一——差异化明确 |
| CPR (NeurIPS 2024) | 替代方案 | 证明 WD 可被约束优化替代。需在 Related Work 中讨论并解释为何统一 WD 方法仍有价值 |

### 5.2 中等威胁

| 论文 | 威胁类型 | 影响评估 |
|------|---------|---------|
| Kosson et al. (ICLR 2026) | 理论基础重叠 | WD 作为 LR 调节器的分析与我们的 ELR 框架重叠。但他们关注迁移，我们关注动态调制——互补 |
| Fan et al. (2025) | 缩放规则竞争 | 提供了 layerwise WD 缩放规则，可能使统一框架的 norm-matched 分支显得冗余。需将其作为特例纳入 |
| GradientStabilizer (2025) | 降低 WD 重要性 | 如果 gradient stabilization 使 WD 不再敏感，动态 WD 的价值进一步被质疑。需在 Discussion 中讨论 |

### 5.3 低威胁

| 论文 | 理由 |
|------|------|
| ARROW (2026) | 专注持续学习场景，不与 from-scratch 训练竞争 |
| Adam 泛化理论 (2025) | 理论贡献，不提出新 WD 方法 |
| AdamHD (2025) | 不同惩罚形式（Huber），不涉及动态调制 |

---

## 6. 对 Iteration 5 论文的具体建议

### 6.1 必须新增的引用

| 论文 | 论文中的位置 | 引用理由 |
|------|------------|---------|
| Fan et al. (2025) arXiv:2510.15262 | Related Work / Theory | layerwise WD 缩放规则；sqrt(eta/lambda) 稳态 |
| CPR (NeurIPS 2024) arXiv:2311.09058 | Related Work | WD 的约束优化替代方案 |
| GradientStabilizer (2025) arXiv:2502.17055 | Discussion | 降低 WD 敏感性的替代思路 |

### 6.2 需要更新的引用

| 引用 | 更新内容 |
|------|---------|
| CWD (Chen et al.) | 更新为 ICLR 2026 Conference Paper |
| Kosson et al. WD > muP | 更新为 ICLR 2026 Conference Paper |

### 6.3 新的理论论点

1. **ELR 吸收机制的形式化**: 利用 Fan et al. 的 sqrt(eta/lambda) 稳态和 Kosson et al. 的 sqrt(eta*lambda) relative updates，形式化证明 Phi modulation 在 scale-invariant networks + AdamW 下被 ELR 调节吸收的条件
2. **CPR 作为 norm-matched WD 的特例**: 证明 CPR 的硬约束等价于在 Phi 框架中设定 phi = 0 when ∥w∥ >= tau（分段 Phi 函数）
3. **Batch size 调节效应**: 利用 arXiv:2510.11354 的理论，讨论 Phi invariance 可能在 large-batch 下更强、mini-batch 下更弱

### 6.4 实验设计建议

1. **ImageNet ResNet-50**: 90 epochs, AdamW (lr=1e-3, lambda=1e-4), cosine LR schedule, 3 seeds
2. **layerwise WD 缩放消融**: 测试 Fan et al. 的 lambda ∝ sqrt(d) vs uniform lambda
3. **CPR 基线**: 实现 AdamCPR 作为额外对比方法
4. **GradientStabilizer 交互**: 测试 CWD + GradientStabilizer 组合是否改变 Phi invariance

---

## 7. 完整参考文献列表（Iteration 5 新增/更新）

### 新增论文（6 篇）

1. Fan, Z. et al. (2025). Robust Layerwise Scaling Rules by Proper Weight Decay Tuning. arXiv:2510.15262
2. Franke, J.K.H. et al. (2024). Improving Deep Learning Optimization through Constrained Parameter Regularization. NeurIPS 2024. arXiv:2311.09058
3. GradientStabilizer (2025). Fix the Norm, Not the Gradient. arXiv:2502.17055
4. Adam Generalization (2025). Understanding the Generalization of Stochastic Gradient Adam. arXiv:2510.11354
5. ARROW (2026). Vision Transformers that Never Stop Learning. arXiv:2603.07787
6. SimbaV2 (2025). Hyperspherical Normalization for Scalable Deep RL. arXiv:2502.15280

### 会议状态更新（2 篇）

7. Chen et al. CWD: arXiv:2510.12402 → **ICLR 2026 Conference Paper**
8. Kosson et al. WD > muP: arXiv:2510.19093 → **ICLR 2026 Conference Paper**

### Iteration 4 已有但未变化（确认存在，8 篇）

9. Wilson et al. (2017). NIPS 2017. arXiv:1705.08292
10. van Laarhoven (2017). arXiv:1706.05350
11. Lakens (2017). SPPS 8(4). TOST 等价性检验
12. Springer (2025). ViT gradient-aware WD. DOI:10.1007/s00138-025-01686-9
13. Cerebras (2025). D2Z linear decay. arXiv:2502.15938
14. Universal WSD (2026). arXiv:2601.09000
15. Damian et al. (2021). SGD implicit regularization. arXiv:2101.12176
16. Smith et al. (2024). Neural networks learn support. arXiv:2406.11110

---

## 8. 领域发展趋势总结

### 2024-2026 年 WD 研究的三大趋势

**趋势 1: 从 "如何设定 WD" 到 "理解 WD 的本质"**
学界正从提出新 WD 方法转向深入理解 WD 为何有效。2024-2026 的标志性论文（D'Angelo, Kosson, Defazio, Fan）共同构建了一个一致的图景：WD 在 scale-invariant 网络中本质上是 ELR 调节器。这一理解使得提出新的 WD 变体变得更困难——因为任何 WD 变体最终都只是间接调节 ELR。

**趋势 2: WD 的"去敏感化"**
GradientStabilizer (2025) 和 CPR (NeurIPS 2024) 从不同角度降低了 WD 调优的重要性。前者通过 gradient norm stabilization 使模型对 WD 值不敏感；后者通过约束优化完全替代 WD penalty。这一趋势暗示：**精心设计的动态 WD 可能是一个日渐过时的研究方向**。

**趋势 3: 层级/模块级差异化**
AlphaDecay (2025)、Fan et al. (2025)、CPR (2024) 都指向同一方向：**不同层/模块需要不同的 WD 处理**。这与 Phi 框架的 per-parameter modulation 设计理念一致，但挑战在于层级差异化是否已被 AdamW 的 per-parameter adaptive scaling 内部解决。

### 对统一框架的战略定位建议

我们的统一框架应重新定位为**分析工具**而非**新方法提出**：
1. **解释力**: 为什么现有的多种 WD 方法（CWD、SWD、AlphaDecay、CPR）最终效果相近？
2. **边界条件**: Phi invariance 在何种条件下成立/不成立？（optimizer x scale x architecture）
3. **标准化评测**: BEM/CSI/AIS 提供统一的比较语言
4. **理论贡献**: 形式化 ELR 吸收机制，证明动态 WD 在特定条件下被吸收的充要条件
