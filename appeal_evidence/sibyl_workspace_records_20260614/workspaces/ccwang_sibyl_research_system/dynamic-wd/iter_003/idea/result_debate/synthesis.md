# 结果辩论综合报告
## Iteration 3 - Unified Dynamic Weight Decay Framework

**综合者**: Sibyl Result Synthesizer
**日期**: 2026-03-18
**辩论参与者**: 乐观者、怀疑论者、方法论者、战略顾问、比较分析者、修正主义者

---

## 一、共识要点

### 1.1 当前实验的核心发现（六方一致认同）
- **42 个实验（7 方法 × 3 种子 × 2 数据集）全部与 constant 基线无统计显著差异**
- CIFAR-10 精度极差仅 0.25%（89.88%-90.13%），CIFAR-100 极差 0.76%
- 所有 p 值 > 0.05，最小 p = 0.090（CIFAR-100 random_mask）
- **even no_wd (λ=0) 表现与 constant 基线无显著差异**——这是最强的证据
- BEM 从 0.0 到 1.0 变化 10 倍，但精度基本不变——WD 预算等价性是红鲱鱼

### 1.2 Phi 框架的方法论价值（六方一致）
- 统一接口 `φ(t, w, g)` 是合理的抽象，消除了跨方法比较的实现混淆因素
- 首次在同一代码库、同一超参数下系统性比较 7 种 WD 方法
- 框架本身可复用于未来 WD 研究

### 1.3 诊断指标的初步价值
- CSI/AIS/BEM 是首次提出的 WD 诊断工具
- BEM 成功量化了"WD 预算与精度无关"这一发现
- CSI 区分了不同方法的训练动态特征（即使精度相同）
- AIS 揭示了对齐信号是网络固有属性而非 WD 方法决定

## 二、分歧与争议

### 2.1 结论的外推范围（主要分歧）

| 立场 | 代表 | 观点 |
|------|------|------|
| **可泛化** | 乐观者 | AdamW 的自适应性使 WD 方法选择在一般情况下无关紧要 |
| **严格限定** | 怀疑论者、方法论者 | 结论仅对 ResNet-20 + CIFAR + AdamW 有效；大模型/SGD 可能完全不同 |
| **有条件泛化** | 修正主义者 | 提出"Phi 不变性定理"——在 AdamW 充分训练条件下成立 |

**综合判断**: 采用"严格限定 + 理论猜想"策略。论文正文严格限定结论范围，同时在 Discussion 中提出 Phi Invariance 猜想作为理论贡献。

### 2.2 统计充分性

| 问题 | 严重程度 | 建议 |
|------|---------|------|
| 3 种子功效不足 | **高** | 增加至 5-10 种子；补充等价性检验 (TOST) |
| 未执行 Bonferroni 校正 | 中 | 6 个比较需要 p < 0.0083 阈值 |
| 缺少 bootstrap CI | 中 | 补充 95% bootstrap 置信区间 |
| 缺少 Cohen's d | 中 | 报告效果量 |

### 2.3 缺失实验的优先级

**方法论者和怀疑论者的核心批评**: 方法论文档中规划的多项实验未执行。

| 优先级 | 实验 | GPU-时 | 理由 |
|--------|------|--------|------|
| **P0 (必须)** | SGD + constant WD 基线 | ~5h | 分离 AdamW 自适应性假说 |
| **P0 (必须)** | VGG-16-BN 跨架构验证 | ~10h | 架构泛化性 |
| **P1 (强烈建议)** | CWD inverted mask (C3) | ~3h | 验证对齐方向假说 H4 |
| **P1 (强烈建议)** | 增加种子至 5 个 | ~14h | 统计功效 |
| **P2 (建议)** | 每方法超参数搜索 | ~20h | 内部效度完整性 |
| **P3 (未来)** | ImageNet + ResNet-50 | ~72h | 外部效度 |
| **P3 (未来)** | ViT-S/16 | ~48h | Transformer 架构 |

## 三、行动决策

### 决策: **PROCEED to writing（附补充实验条件）**

**理由**:
1. 当前 42 个实验已构成论文的核心证据——足够支撑 CIFAR-scale 结论
2. P0/P1 补充实验（SGD, VGG-16, C3, 更多种子）可在写作阶段并行完成
3. 论文 framing 采用战略顾问建议: "Unified Framework + Systematic Benchmark"
4. 修正主义者的"Phi Invariance"叙事作为 Discussion 亮点

### 论文定位
- **标题方向**: "When Does Dynamic Weight Decay Help? A Unified Framework Analysis"
- **核心贡献**:
  1. Phi 调制器框架——统一所有 WD 方法的数学抽象
  2. BEM/CSI/AIS 诊断指标套件——首个标准化 WD 评测工具
  3. 42 实验的系统性负结果——AdamW 下 WD 调度选择无关紧要
  4. Phi Invariance 猜想——为未来研究提供理论方向
- **目标会议**: NeurIPS 2026 Datasets & Benchmarks Track（主要）/ TMLR（备选）

### 补充实验计划（与写作并行）
在当前实验基础上立即启动:
1. SGD baseline (3 seeds × 2 datasets × constant WD) → 6 runs
2. VGG-16-BN (3 seeds × 2 datasets × 7 methods) → 42 runs
3. CWD inverted mask (3 seeds × 2 datasets) → 6 runs
4. 增加种子 789, 999 (2 extra seeds × 2 datasets × 7 methods) → 28 runs

总计 82 runs 额外实验，与论文写作并行。

## 四、风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| SGD 实验显示 WD 方法确实重要 | 30% | 高——需要重写叙事 | 不是风险而是新发现，更强化论文价值 |
| VGG 结果与 ResNet 不一致 | 20% | 中——需要解释 | 增加讨论深度 |
| 审稿人要求 ImageNet | 60% | 中——rebuttal 难度大 | 在 Discussion 中 upfront 承认限制 |
| 指标定义被质疑 | 40% | 低——补充理论推导 | 加强数学基础章节 |

---

*综合时间: 2026-03-18*
*基于 6 位辩论者的独立分析报告*
