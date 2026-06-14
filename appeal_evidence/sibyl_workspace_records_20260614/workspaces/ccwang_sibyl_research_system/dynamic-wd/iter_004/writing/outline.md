# 论文详细大纲

**标题**: When Does Dynamic Weight Decay Help? A Unified Framework Analysis

**核心主题**: 通过 Phi Modulator 统一框架和 ρ = λ/η regime 边界理论，系统性回答"动态权重衰减何时有用"这一实际问题。

**目标会议**: NeurIPS 2026（主会 borderline → competitive，取决于 P0/P1 实验结果）

---

## Abstract（约 250 词）

**结构**: 问题 → 方法 → 发现 → 贡献

- **问题**: 近年涌现大量动态 WD 方法（CWD、SWD、cosine schedule、AlphaDecay、ADANA、AdamO 等），但缺乏系统性理解：何时需要调整 WD 策略？
- **方法**: 提出 Phi Modulator Framework，将所有动态 WD 方法统一为 φ(t, θ, g) 的特例，沿四个调制轴（temporal, directional, spatial, target-norm）分类；提出 BEM/CSI/AIS 三个标准化诊断指标
- **核心发现**:
  - 在标准 AdamW 设定（ρ = λ/η ≈ 0.5）下，7 种 WD 策略统计不可区分（极差 < 0.3%），包括完全不用 WD
  - SGD 下 WD 策略选择显著影响性能（constant vs no_wd: Cohen's d > 10, p < 0.003）
  - 提出 ρ = λ/η 作为 regime 边界参数的 Phi 不变性猜想
- **贡献总结**: (1) Phi Modulator 统一框架 + 四轴分类法，(2) Phi 不变性猜想及条件性等价性的系统性实证，(3) BEM/CSI/AIS 标准化评测体系，(4) 跨优化器/架构/数据集的大规模对比

---

## 1. Introduction（约 1500 词）

### 1.1 Motivation（~400 词）
- WD 是深度学习优化中最普遍应用的技术之一
- 近期文献重新理解 WD：不是经典 L2 正则化，而是训练动力学修改器（D'Angelo 2024）、ℓ∞ 约束优化（Xie & Li 2024）、旋转平衡（Kosson 2023）
- 这些新理解催生了大量动态 WD 方法：SWD（Xie 2023）、CWD（Chen 2026a）、AdamWN（Loshchilov 2023）、AlphaDecay（He 2025）、ADANA（Ferbach 2026）、AdamO（Chen 2026b）
- 关键问题：**每种方法在不同条件下独立评估**，无法直接比较 → 从业者无法判断哪种策略真正有用

### 1.2 Research Gap（~400 词）
四个关键空白：
1. **无统一数学框架**: 四大 WD 家族各有独立数学表述，无法揭示它们是否是同一原理的特例
2. **无标准化评测指标**: 各论文报告不同指标、不同条件下的结果
3. **无控制实验对比**: 没有在统一代码库、相同超参数下的系统性对比
4. **无理论预测何时动态 WD 重要**: 缺乏 regime boundary 的理论框架

### 1.3 Our Approach: The ρ = λ/η Lens（~300 词）
- 提出以 ρ = λ/η（归一化 WD 强度）作为统一视角
- 预览核心发现：ρ < 1（标准设定）→ 策略无关；ρ 增大 → 策略开始重要
- SGD 缺乏 AdamW 的 ℓ∞ 隐式约束 → 没有 Regime I → 解释 18.3× 效应比
- 这一视角同时联系了 5+ 篇独立近期论文（Xie & Li 2024, D'Angelo 2024, Wang & Aitchison 2024, Defazio 2025, Chou 2025）

### 1.4 Contributions（~200 词）
按重要性排序：
1. **实证发现**: AdamW 标准设定下的条件性等价性 + SGD/AdamW 系统性对比（18.3× 效应比）
2. **理论框架**: ρ = λ/η 作为 regime 边界的 Phi 不变性猜想（Conjecture），含可证伪预测
3. **分类学贡献**: Phi Modulator 四轴分类法，首次将 7+ 种方法纳入统一接口
4. **评测工具**: BEM/CSI/AIS 作为 WD 策略标准化特征描述工具

### 1.5 Paper Roadmap（~100 词）
- §2 Related Work → §3 Phi Framework + Metrics → §4 Theoretical Analysis (ρ regime) → §5 Experimental Setup → §6 Results → §7 Discussion → §8 Conclusion

---

## 2. Related Work（约 1200 词）

### 2.1 Weight Decay as a Dynamics Modifier（~300 词）
- 经典 L2 正则化观点（Krogh & Hertz 1991）
- Loshchilov & Hutter (2019): L2 ≠ decoupled WD in adaptive optimizers → AdamW
- D'Angelo et al. (2024): WD 从未作为正则化有效，而是训练动力学修改器
- Kosson et al. (2023): 旋转平衡机制
- Xie & Li (2024): AdamW = ℓ∞ 约束优化（Frank-Wolfe 连接）
- **定位**: 这些现代理解为我们的 ρ regime 分析提供了理论基础

### 2.2 Dynamic Weight Decay Methods（~400 词）
按四轴组织：
- **Temporal**: SWD/AdamS (Xie 2023), ADANA (Ferbach 2026), cosine/linear schedules
- **Directional**: CWD (Chen 2026a), AdamO (Chen 2026b), SPD (Tian 2024)
- **Spatial**: AlphaDecay (He 2025)
- **Target-norm**: AdamWN (Loshchilov 2023)
- **本文差异化**: 我们不提出新方法，而是系统性评测和理论分析这些方法的有效条件

### 2.3 Evaluation Fragmentation and Null Results（~300 词）
- 评测碎片化问题：每篇论文使用不同架构/数据集/优化器/超参数
- Wang & Aitchison (2024): 最优 WD 作为 EMA 时间尺度常数 → 暗示常数 WD 可能已足够
- **与本文关系**: 他们的 EMA 时间尺度 ≈ 我们 ρ 的倒数的物理解释

### 2.4 Positioning Against Key Recent Works（~200 词）
- vs. Wang & Aitchison (2024): 他们是经验 scaling rule；我们是含可证伪预测的形式化 Trichotomy
- vs. D'Angelo (2024): 他们的 BN scale-invariance 是静态重参数化论证；我们的 Phi 不变性是关于 AdamW sign normalization 的动态论证
- vs. Chou (2025): 他们的 λ ∝ γ schedule 是我们 Regime I 分析的特例

---

## 3. The Phi Modulator Framework（约 1500 词）

### 3.1 Formal Definition（~400 词）
- 标准 AdamW 更新规则
- 引入 Phi modulator φ(t, θ, g): θ_{t+1} = θ_t - η_t û_t - λ · φ(t, θ_t, g_t) ⊙ θ_t
- 三个关键性质：Positivity, Measurability, Reference target (E[φ] = 1)
- 程序化接口：`WDModulator` ABC

### 3.2 Method Catalog: Recovering Existing Methods（~400 词）
- **Table 1**: 7+ 种方法的 φ 闭式表达和调制轴分类
  - constant: φ ≡ 1
  - CWD hard: φ = 𝟙[sign(θ) = sign(u)]（directional）
  - SWD: φ = h(‖g‖)（temporal-gradient）
  - cosine_schedule: φ = ½(1 + cos(πt/T))（temporal）
  - AdamWN: φ = max(0, 1 - τ/‖θ‖)（target-norm）
  - AlphaDecay: φ = α_l（spatial）
  - no_wd: φ ≡ 0（ablation）
  - random_mask: φ = Bernoulli(p)（control）
- **Proposition 1 (Composition)**: φ₁ ⊙ φ₂ 也是合法 Phi modulator → 组合策略的形式化

### 3.3 Budget Equivalence Normalization（~200 词）
- Definition 1 (Budget Equivalence): 总有效 WD 预算匹配
- 有效 WD: λ_eff(t) = λ · E_θ[φ(t, θ_t, g_t)]
- 重要性：不做预算归一化，方法差异可能仅反映总衰减量的不同

### 3.4 Diagnostic Metrics（~500 词）
- **BEM (Budget Equivalence Metric)**: 衡量方法的有效 WD 预算偏离
  - 公式: BEM = (λ̄_constant - λ̄_method) / λ̄_constant（已修复：有符号，无 abs）
  - 定位：描述性特征工具（quantify budget deviation），非性能预测指标
  - 验证值：half_lambda BEM = -0.500, cosine BEM ≈ -0.600, no_wd BEM = -1.000

- **CSI (Coupling Stability Index)**: 衡量 WD 与优化器适应动力学的耦合稳定性
  - 三个子成分：weight norm CV, spectral condition, effective LR CV
  - 相对归一化：CSI_rel = CSI_raw / CSI_constant

- **AIS (Alignment Informativeness Score)**: 衡量权重-梯度对齐信号是否具有训练信息量
  - 基于 Spearman 相关的层平均
  - 定位：网络/损失面的内在属性，非 WD 方法属性
  - 实际范围：0.25–0.50（中等对齐多样性）

- **指标体系定位声明**: 这些是 WD 策略的标准化特征描述工具（characterization tools），用于理解"方法之间如何不同"，而非预测"哪个方法更好"

---

## 4. Theoretical Analysis: The ρ Regime Boundary（约 1200 词）

### 4.1 The ρ = λ/η Order Parameter（~300 词）
- 定义 ρ = λ̄/η（归一化 WD 强度）
- 物理解释：ρ 衡量 WD step 相对于 gradient step 的大小
- 与现有理论的联系：
  - Xie & Li (2024) 约束半径 τ* = η/λ = 1/ρ
  - Defazio (2025) 稳态梯度-权重比 R_* ≈ λ/η = ρ
  - **Theorem T2 (Dual Characterization)**: 这两个量是 ρ 的对偶表征（新联系）

### 4.2 Phi Invariance Conjecture（~500 词）
- **Conjecture 1 (Phi Invariance Trichotomy)**:
  定义 ρ = λ̄/η，存在常数 0 < ρ₁ < ρ₂ 使得：
  - **Regime I** (ρ ≤ ρ₁): 任意两个预算等价的 WD schedule 产生 O(ρ² · V · T · η²) 的最终损失差异 → 标准设定（ρ ≈ 0.5）下 < 0.1%
  - **Regime II** (ρ₁ < ρ < ρ₂): 损失差异 scales as O(ρ · Σ|λ₁ - λ₂| · (1 - δ_t))
  - **Regime III** (ρ ≥ ρ₂): WD step 与 gradient step 竞争，所有调制策略产生 O(ρ) 差异

- **定位**: 明确标注为 Conjecture（非 Theorem），附实验验证
- **可证伪预测**:
  - ρ = 0.5 → accuracy spread < 0.5%（已确认：CIFAR-10 上 0.25%）
  - ρ = 5 → accuracy spread 1–3%（待 λ sweep 验证）
  - SGD 缺乏 ℓ∞ 约束 → 无 Regime I → 解释 18.3× 比值

- **证明要素概览**（如果截稿前能形式化）:
  - Lemma 1: AdamW ℓ∞ norm stability (from Xie & Li 2024)
  - Lemma 2: WD perturbation bound via triangle inequality
  - Lemma 3: Regime I negligibility via damped sum
  - 关键假设: Adam saturation condition（需实验验证 ε/(h_i|w_i|) < 0.1）

### 4.3 Why SGD Differs: The Absence of Implicit Constraint（~400 词）
- AdamW: sign(m̂/√v̂) 产生 ℓ∞ 隐式约束 → WD perturbation 被吸收
- SGD: 无适应性缩放 → WD perturbation 直接影响梯度步
- 定量分析：SGD constant vs no_wd 的 Cohen's d = 10.29 vs AdamW 的 d < 1
- **Theorem T3 (SGD Optimal Schedule)**: 在对齐相关稳定性边界下，SGD 的最优预算分配 WD schedule 为 λ_t* ∝ δ_t/(1-δ_t)（water-filling 解）→ 为 CWD 的二值 mask 提供理论基础

---

## 5. Experimental Setup（约 800 词）

### 5.1 Implementation（~200 词）
- 统一 `UnifiedAdamW` + pluggable Phi modulator 接口
- 7 种 WD 方法：constant, cwd_hard, swd, cosine_schedule, random_mask, half_lambda, no_wd
- 代码基于 `why-weight-decay` (D'Angelo 2024) 扩展

### 5.2 Training Configuration（~300 词）
- **数据集**: CIFAR-10, CIFAR-100（标准增强）；ImageNet（如 P1-2 完成）
- **架构**: ResNet-20 (270K params, w/ BN)；VGG-16-BN（如 P0-4 完成）；ResNet-20-NoBN（BN 消融）
- **优化器**: AdamW (η=1e-3, cosine annealing, β₁=0.9, β₂=0.999) + SGD 对照 (lr=0.1, momentum=0.9, cosine annealing)
- **WD**: λ = 5e-4（标准）；λ sweep: {5e-5, 5e-4, 5e-3, 5e-2}（ρ regime 验证）
- **训练**: 200 epochs, batch size 128
- **随机种子**: 至少 3 seeds (42, 123, 456)，关键对比增至 5 seeds

### 5.3 Statistical Analysis Protocol（~300 词）
- 配对 t 检验 + Bonferroni-Holm 多重比较校正
- TOST 等价性检验（δ = 0.5%）
- Cohen's d 效应量
- Bootstrap BCa 95% CI（用于 18.3× 效应比）
- Post-hoc power analysis: 报告 MDE (minimum detectable effect)
- **统计诚实声明**: 明确报告哪些对比显著、哪些不显著

---

## 6. Results（约 2500 词）

### 6.1 AdamW: Conditional Equivalence Under Standard Settings（~600 词）
- **Table 2**: AdamW 主结果表（CIFAR-10/100, ResNet-20, 7 methods × 3+ seeds）
  - 所有方法精度极差 < 0.3%（CIFAR-10）
  - 所有配对 t 检验 p > 0.05（Holm 校正后）
  - TOST 等价性检验结果（附 power analysis）
- **Figure 1**: Box plot / violin plot 展示 7 种方法精度分布（视觉化等价性）
- 讨论：no_wd (λ=0) 也统计等价 → 在 ρ ≈ 0.5 设定下，WD 的存在与否都不重要

### 6.2 SGD: Weight Decay Strategy Matters（~600 词）
- **Table 3**: SGD 主结果表（CIFAR-10/100, ResNet-20, 7 methods × 3 seeds）
  - constant vs no_wd: Δ = 0.91%, Cohen's d = 10.29, p_adj = 0.002（显著）
  - **诚实声明**: Holm 校正后**仅此一个对比显著**，swd (p_adj=0.054) 和 half_lambda (p_adj=0.062) 均未达显著
- **Figure 2**: AdamW vs SGD 精度分布对比图（双 panel）
- **18.3× 效应比**: 定义为 SGD effect size / AdamW effect size (constant vs no_wd)
  - 明确标注为"WD 存在性效应比"（非动态策略效应比）
  - 附 Bootstrap BCa 95% CI
  - 即使保守下界 ~12×，仍是有意义的定量发现

### 6.3 Cross-Architecture Validation（~400 词）
*[如果 VGG-16-BN 全量实验完成]*
- **Table 4**: VGG-16-BN 结果（CIFAR-10/100, AdamW + SGD）
- AdamW 不变性是否跨架构保持？SGD 效应是否跨架构复制？
- 两种叙事路径预案：
  - Path A（不变性保持）: 强通用性声称
  - Path B（不变性打破）: 架构条件性分析

### 6.4 BN Ablation: Mechanism Identification（~400 词）
*[如果 NoBN 实验完成]*
- **Table 5**: ResNet-20-NoBN 结果
- 两条叙事路径：
  - Path A（NoBN 仍不变）: AdamW 的 sign normalization/ℓ∞ 约束是内在原因 → 强贡献
  - Path B（NoBN 打破不变性）: BN 是必要条件 → 重定位为 "BN + AdamW 交互效应"

### 6.5 ρ Regime Sweep（~500 词）
*[如果 λ sweep 实验完成]*
- **Table 6**: λ ∈ {5e-5, 5e-4, 5e-3, 5e-2} 对应 ρ ∈ {0.05, 0.5, 5, 50}
- **Figure 3 (论文最核心图)**: x 轴 = ρ, y 轴 = 方法间精度极差（或标准差）
  - 预测：ρ < 1 时极差 < 0.5%，ρ 增大时极差单调增长
  - 如果观察到 phase transition → 最强证据
  - 如果仅单调增长 → 连续谱叙事
- 与 Trichotomy 预测的定量对比

---

## 7. Diagnostic Analysis（约 1000 词）

### 7.1 BEM: Budget Characterization（~300 词）
- **Table 7**: 所有方法的 BEM 值
  - constant: 0.000, half_lambda: -0.500, cosine: -0.600, no_wd: -1.000
  - CWD/SWD: 实际 BEM 值与理论预期对比
- 讨论：BEM 正确区分了方法的预算特征，但预算差异**并未**导致精度差异（在 AdamW Regime I 下）

### 7.2 CSI: Coupling Stability（~300 词）
- 各方法 CSI 值对比
- 讨论：CSI 差异在 Regime I 下是否与性能相关？

### 7.3 AIS: Alignment Informativeness（~300 词）
- 各方法/架构/数据集的 AIS 值
- 关键问题：AIS 在 BN 网络中是否被压制？（与 D'Angelo 2024 BN scale-invariance 的关系）
- 如果 AIS 在 NoBN 中显著更高 → 支持 Contrarian 预测

### 7.4 Weight Norm Trajectory Analysis（~100 词）
- **Figure 4**: 训练过程中各方法的权重范数轨迹
- AdamW 下所有方法收敛到相近的范数（ℓ∞ 约束效应）

---

## 8. Discussion（约 1200 词）

### 8.1 Practical Implications（~300 词）
- **核心信息给从业者**: 在标准 AdamW 设定（λ=5e-4, η=1e-3, BN 架构）下，**无需费心调优 WD 策略**
- 何时需要关注 WD 策略：使用 SGD 时；使用高 λ 时；非 BN 架构时（待验证）
- 决策流程图：optimizer → ρ → regime → recommendation

### 8.2 Theoretical Implications（~300 词）
- ρ = λ/η 作为统一视角联系了多篇独立工作
- Dual characterization (τ* = 1/ρ from Xie & Li; R_* = ρ from Defazio) 的新颖性
- 与 Wang & Aitchison (2024) EMA timescale 的联系

### 8.3 Scope and Boundary Conditions（~300 词）
- 当前验证范围：CIFAR-10/100, ResNet-20, standard hyperparams
- 未验证但有预测的 regime：大规模（ImageNet）、大模型（LLM）、非标准 λ
- BN vs NoBN 的影响（如果有 NoBN 数据）

### 8.4 Limitations（~300 词）
坦诚列出：
1. **统计功效**: n=3 (或 n=5) 的 TOST power 有限；报告 MDE
2. **架构覆盖**: 仅 ResNet-20 (+ VGG-16-BN)；未覆盖 ViT、大规模 LLM
3. **数据集规模**: CIFAR 级别；ImageNet 仅 pilot（如果有）
4. **ρ regime 边界精度**: 仅测试离散 ρ 值，边界为近似
5. **Conjecture 非 Theorem**: 形式化证明留给 future work（除非截稿前完成）
6. **BEM bug 历史**: 坦诚说明发现和修复过程

---

## 9. Conclusion（约 400 词）

### 9.1 Summary of Findings（~150 词）
- Phi Modulator 框架首次统一了 7+ 种动态 WD 方法
- 在标准 AdamW 设定下，所有动态 WD 策略统计等价 → 从业者可节省调优成本
- ρ = λ/η 作为 regime 边界参数，提供了可证伪的理论预测
- SGD/AdamW 18.3× 效应比揭示了优化器选择对 WD 敏感性的决定性影响

### 9.2 Future Work（~200 词）
1. 形式化 Phi 不变性猜想的数学证明
2. 大规模验证：ImageNet + LLM pre-training
3. ρ regime 边界的精确刻画（连续 λ sweep）
4. Regime II/III 中的最优 WD 策略设计
5. 与其他优化器（Lion, Muon, Shampoo）的交互

### 9.3 Broader Impact（~50 词）
- 负面结果的价值：知道"什么时候不需要做"与"需要做什么"同样重要
- 节省研究者和从业者在 WD 调优上的无效工作量

---

## 附录结构

### Appendix A: Extended Statistical Tables
- 所有配对 t 检验的完整结果（含 p 值、效应量、CI）
- TOST 等价性检验详细结果
- Bootstrap 分析详情

### Appendix B: Diagnostic Visualization Panels
- 每个实验配置的训练曲线（loss + accuracy vs epoch）
- 权重范数轨迹（per-layer + aggregate）
- BEM/CSI/AIS 随训练进程的变化

### Appendix C: Metric Sensitivity Analysis
- CSI 权重选择的 sensitivity analysis
- AIS 采样策略的稳健性
- BEM 在不同 normalization 下的行为

### Appendix D: Mathematical Details
- Phi 不变性猜想的证明要素（Lemma 1-3 详细推导）
- Adam saturation condition 的实验验证
- SGD optimal schedule 推导

### Appendix E: Implementation Details
- 代码架构和接口说明
- 超参数完整列表
- 计算资源使用

---

## 图表清单

### 正文图表

| 编号 | 类型 | 内容 | 地位 |
|------|------|------|------|
| **Figure 1** | Box/Violin plot | AdamW 7 方法精度分布（CIFAR-10/100） | 核心：视觉化等价性 |
| **Figure 2** | 双 panel 对比图 | AdamW vs SGD 精度分布 | 核心：18.3× 效应比视觉化 |
| **Figure 3** | Line/scatter plot | ρ vs 方法间精度极差 | **论文最核心图**：regime boundary 证据 |
| **Figure 4** | 训练曲线 | 权重范数轨迹（AdamW vs SGD） | 支撑：ℓ∞ 约束效应可视化 |
| **Figure 5** | 框架图/taxonomy | Phi Modulator 四轴分类示意图 | 支撑：框架概览 |
| **Table 1** | Method catalog | 7 种方法的 φ 闭式表达 | 核心：框架定义 |
| **Table 2** | 主结果表 | AdamW 全量结果 + 统计检验 | 核心 |
| **Table 3** | 主结果表 | SGD 全量结果 + 统计检验 | 核心 |
| **Table 4** | 结果表 | VGG-16-BN 跨架构结果 | 条件性（视实验完成情况） |
| **Table 5** | 结果表 | NoBN 消融结果 | 条件性（视实验完成情况） |
| **Table 6** | 结果表 | λ sweep / ρ regime 结果 | 条件性（视实验完成情况） |
| **Table 7** | 诊断表 | BEM/CSI/AIS 完整值 | 支撑 |

### 附录图表
- Training curve panels (per-config)
- Per-layer weight norm heatmaps
- BEM/CSI/AIS evolution plots
- Adam saturation condition verification plot

---

## 关键叙事策略

### 三段式核心叙事

1. **问题**: 大量动态 WD 方法被提出，但何时真正需要调整 WD 策略缺乏系统性理解
2. **发现**: 通过 Phi Modulator 框架统一这些方法后，我们发现：在标准 AdamW 设定（ρ ≈ 0.5）下所有方法统计等价；但在 SGD 或高 ρ 下策略选择开始重要
3. **贡献**: (a) 条件性等价性的首次系统验证，(b) ρ = λ/η regime boundary 的可证伪理论，(c) Phi 四维分类 + BEM 评测工具

### 叙事红线
- **不说**: "我们测试了 7 种动态 WD 方法，它们在 AdamW 下都不工作"
- **要说**: "我们发现了一个 regime boundary：ρ < 1 时策略选择不重要，ρ 增大时开始重要"
- **用 Conjecture 而非 Theorem**（除非有形式化证明）
- **度量体系定位为描述性工具**，避免审稿人追问"预测了什么"
- **SGD 数据诚实报告**：Holm 校正后仅 1 个显著效应
- **Limitations 坦诚**：n=3 局限、架构覆盖有限、BN 影响

### 条件性内容决策树

```
NoBN 实验结果
├── 不变性保持 → §6.4 写 "AdamW mechanism is intrinsic" (强贡献)
└── 不变性打破 → §6.4 写 "BN + AdamW interaction" (仍有价值但重定位)

ρ sweep 结果
├── 支持 Trichotomy → §6.5 + Figure 3 成为论文最亮眼内容
├── 单调增长（无 phase transition）→ 连续谱叙事
└── 不支持 → 删除 §4.2 Trichotomy，转为纯经验发现

VGG-16-BN 结果
├── 不变性保持 → §6.3 写跨架构通用性
└── 不变性打破 → §6.3 写架构条件性分析
```

---

## 页数估算

| 章节 | 预估页数 |
|------|---------|
| Abstract | 0.3 |
| Introduction | 1.5 |
| Related Work | 1.2 |
| Phi Framework | 1.5 |
| Theoretical Analysis | 1.2 |
| Experimental Setup | 0.8 |
| Results | 2.5 |
| Diagnostic Analysis | 1.0 |
| Discussion | 1.2 |
| Conclusion | 0.4 |
| References | 1.0 |
| **正文总计** | **~12.6** |
| Appendix | 4-6 |

**备注**: NeurIPS 正文限制 9 页（不含 references）。需要压缩约 2.5 页：
- §7 Diagnostic Analysis 可部分移入附录（-0.8 页）
- §4 Theoretical Analysis 可精简（-0.5 页）
- §6 Results 精简叙述（-0.5 页）
- §8 Discussion 精简（-0.7 页）

---

*大纲生成者: Sibyl Outline Writer*
*基于: iter_004 result debate synthesis + proposal + iter_003 integrated paper + pilot summary*
*日期: 2026-03-18*
