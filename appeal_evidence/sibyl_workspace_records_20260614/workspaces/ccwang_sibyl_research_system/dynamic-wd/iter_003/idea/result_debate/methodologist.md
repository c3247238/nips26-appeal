# 方法论评审报告：统一动态权重衰减框架（Iter 003）

**评审角色**: 方法论者 (Methodologist)
**评审日期**: 2026-03-18
**评审对象**: Unified Dynamic Weight Decay Framework — 第三轮全量实验

---

## 1. 总体评价

本实验在方法论设计层面展示出相当严谨的规划意图——Phi 模块化接口、多对照组、假说驱动的实验分层结构均属良好实践。然而，实际执行的实验与方法论文档中描述的完整计划存在显著缺口：Phase 2 仅完成了 7 种方法在两个数据集上的 ResNet-20 运行，VGG-16-BN、消融组（CWD inverted mask、AdamWN、AlphaDecay）、Warmup-K 扫描等关键组均未执行。Phase 3（ImageNet）和 Phase 4（ViT）均未启动。这一缺口直接影响结论的可信度范围。

---

## 2. 内部效度评估

### 2.1 控制变量的充分性

**正面**:
- 所有方法共享同一 `UnifiedAdamW + Phi modulator` 实现，消除了优化器代码差异带来的混淆因素。
- 训练超参数在所有方法间完全固定（LR=1e-3, WD=5e-4, 200 epochs, cosine annealing, batch 128），确保比较公平性。
- 3 个独立随机种子（42、123、456）提供了有限的方差估计。

**问题**:

1. **超参数搜索协议未被遵守**: 方法论文档明确规定了 LR grid（4 值）× WD grid（6 值）的超参数搜索，以 seed 42 选最优后再在三种 seed 上评估。实际执行使用固定 HP（LR=1e-3, WD=5e-4），未经每种方法独立调参。这意味着：对于 CWD、SWD 等方法，当前超参数组合可能并非其最优区间，**性能差距可能部分归因于超参数偏差而非方法本身的劣势**。这是本实验内部效度最严重的潜在缺陷。

2. **SGD 基线缺失**: 方法论中将 SGD + momentum + constant WD 列为"经典基线"，实际结果表中完全缺失。在没有 SGD 基线的情况下，"AdamW 的自适应步长已提供足够隐式正则化"这一解释性结论无法得到对照支撑。

3. **对照组严重不足**: CWD 伪造检验组（C1 effective-lambda matched、C2 random binary mask matched sparsity、C3 inverted mask、C4 soft CWD）中，实际只运行了 random_mask（C2 的近似版本）和 half_lambda（C1 的粗略替代）。inverted mask（C3）和 soft CWD（C4）均未出现在结果中，使得"alignment 方向是否重要"这一核心假说（H4）无法从现有数据中得出确定性结论。

4. **数据增强未验证一致性**: 结果文件未记录实际使用的 data augmentation pipeline 是否与方法论一致（应为 RandomCrop + RandomHorizontalFlip for CIFAR）。若各方法的 augmentation 配置存在差异，将引入额外混淆。

### 2.2 Pilot 验证的杀停准则未被执行

方法论规定 Pilot Phase kill criterion 之一为：**"AdamW baseline < 91% → codebase bug"**。然而实际结果中 constant WD 在 CIFAR-10 上仅达 90.13%，未达 91%阈值，却未见杀停或调查记录。这可能意味着：
- Pilot 阶段的 kill criterion 被降低或绕过，
- 或者 ResNet-20 在该超参数组合下的合理上限本身就低于 91%（方法论中的阈值设置本身不合理）。

无论哪种情况，都需要在最终论文中明确说明，否则读者会怀疑基础实现的正确性。

---

## 3. 外部效度评估

### 3.1 结论的可泛化范围

当前数据支持的结论严格限定于：**ResNet-20 + AdamW + cosine-annealing LR + CIFAR-scale 数据集**。

以下泛化方向均未被当前实验覆盖，存在实质性的外部效度风险：

| 泛化维度 | 当前状态 | 风险等级 |
|---------|---------|---------|
| 更大模型（ResNet-50/101, ViT） | 未执行 Phase 3/4 | 高 |
| 更大数据集（ImageNet-1K） | 未执行 Phase 3 | 高 |
| 不同优化器（SGD, Adam, Lion） | SGD 基线缺失 | 高 |
| Transformer 架构（无 BatchNorm） | 未执行 Phase 4 | 高 |
| 不同任务（目标检测、NLP） | 未规划 | 中 |
| 不同 base WD 取值（5e-5 / 5e-3） | 未见超参扫描结果 | 中 |

**关键风险**: 论文的核心负面结论（"动态 WD 不优于常数 WD"）在仅有 CIFAR/ResNet-20 数据支撑的情况下，极易被审稿人以"不充分的规模"为由拒绝。特别是已有文献（如 CWD 原始论文 D'Angelo et al.）在更大规模设置中声称存在收益，而本实验未在同等规模下复现。

### 3.2 数据集代表性问题

CIFAR-10 和 CIFAR-100 均为 32×32 低分辨率基准，任务难度有限。这两个数据集上观察到的"WD 策略无关性"可能反映的是：
- 小数据集上正则化需求本身较低（过拟合压力不足以让 WD 策略产生差异）；
- ResNet-20 参数量（~270K）远小于 ImageNet 级模型，WD 对参数分布的影响更为均匀。

因此，"budget equivalence is a red herring"这一措辞过强，应限定为"在 CIFAR/ResNet-20 规模下，budget equivalence 与准确率无显著相关"。

---

## 4. 统计方法评估

### 4.1 Paired t-test 的使用

**问题 1：样本量不足导致检验功效极低**

3 个种子的 paired t-test 自由度为 df=2，对应临界值 t(0.025, 2) = 4.303。在当前观测到的效应量（最大差异约 0.49% for no_wd on CIFAR-100）和标准差（~0.3-0.4%）下：

```
效应量 d ≈ 0.49 / 0.38 ≈ 1.29
t 统计量 = d * sqrt(n) = 1.29 * sqrt(3) ≈ 2.23
```

这个 t 值远低于 df=2 下的临界值 4.303，因此 p > 0.05 是必然结果——**不是因为没有效应，而是因为样本量不足以检测到这个量级的效应**。换言之，当前的"统计不显著"结论在统计学上几乎没有意义。

要在 d=0.5（实践显著性阈值）、power=0.8、alpha=0.05 的条件下达到足够功效，需要至少 **n≥26 个独立种子**，而非 3 个。

**问题 2：Bonferroni 校正未被执行**

方法论文档明确要求 Bonferroni 校正（6 组比较），但 analysis.md 中的 p 值未标注是否经过校正。若为未校正值，在 6 组比较的多重检验情境下，显著性阈值应调整为 p < 0.05/6 ≈ 0.0083，这使得正面结果更难出现，但负面结论也不能直接用未校正 p 值来声称"不显著"。

**问题 3：Cohen's d 缺失**

方法论要求报告所有成对比较的效应量（Cohen's d），但结果表中完全没有 d 值。在样本量不足导致功效偏低的背景下，效应量才是评估"实践显著性"的核心指标。其缺失使得"差异在实践中微不足道"的结论缺乏量化支撑。

### 4.2 Bootstrap CI 缺失

方法论规定使用 1000 次 bootstrap 重采样计算 95% CI，但 analysis.md 中仅报告均值和标准差，没有置信区间。在 n=3 的极小样本下，基于正态分布假设的标准误（std/sqrt(3)）的可靠性远低于 bootstrap CI。

### 4.3 Spearman 秩相关缺失

方法论要求通过 Spearman rho 验证 CSI/AIS 对准确率排名的预测性（H5）。结果中虽然讨论了 CSI/AIS 的特征，但未提供任何相关系数。"CSI does not correlate with accuracy"这一结论若无统计检验支撑，属于定性观察而非科学声明。

---

## 5. 可复现性评估

### 5.1 方法论文档中的可复现性清单执行状态

方法论第 11 节给出了明确的可复现性清单（6 项），但无法从现有结果文件中确认其执行状态：

| 清单项 | 状态 | 说明 |
|-------|------|------|
| 所有随机种子显式设置（torch/numpy/CUDA/Python hash） | 未确认 | 结果文件未记录 CUDA/Python hash seed 设置 |
| 精确包版本锁定在 requirements.txt | 未确认 | 结果文件中未引用 |
| 训练配置保存为 JSON | 部分满足 | full_summary.json 记录了 HP，但未见完整 config JSON |
| Best validation 和 final epoch checkpoint 保存 | 未确认 | 无 checkpoint 路径记录 |
| 原始指标记录到 CSV/JSONL | 部分满足 | accs per seed 已记录，但无 per-epoch 曲线 |
| 诊断快照每 100 步一次 | 未确认 | 结果文件中不含 per-step 快照数据 |

**关键缺口**: 结果文件中没有提供 Git commit hash 或代码版本标识，使得第三方无法精确复现实验环境。

### 5.2 超参数记录的完整性

full_summary.json 仅记录了最终准确率和自定义指标（CSI/AIS/BEM/Weight Norm），缺少：
- 实际使用的 LR schedule 参数（warmup steps 数量）
- 数据增强的具体参数（如 RandomCrop padding size）
- CWD/SWD 的方法特有超参数（beta、sensitivity）
- 每个 seed 的独立训练时间

---

## 6. CSI/AIS/BEM 指标定义的数学严谨性

### 6.1 BEM（Budget Equivalence Metric）

方法论中定义为：
```
BEM(method, baseline) = (acc_method - acc_baseline) / acc_baseline
```

**问题**:
1. 这个公式是相对准确率差异，数值范围取决于基准方法的绝对准确率，不具有"等效预算"的直接含义。真正意义上的 budget equivalence 应该是对有效 L2 正则化量的比较，而非准确率比较。
2. 结果表中 BEM 的取值（0.000、0.503、0.900、1.000）显示这并非连续的准确率差异，而是某种离散编码（half_lambda=0.000 与 constant 相同，BEM 应等于 0），与公式定义不一致。这暗示实际使用的 BEM 定义可能与方法论文档描述不同。
3. "BEM ≈ 0 confirms budget equivalence" 的判断准则需要明确阈值（如 |BEM| < 0.1%），否则无法客观判定等效性。

**建议**: 将 BEM 重新定义为有效权重衰减量的积分差异（normalized by constant WD integral），使其直接反映 WD 预算的物理含义：
```
BEM = (integral_0^T lambda_t dt) / (lambda_constant * T) - 1
```

### 6.2 CSI（Coupling Stability Index）

定义为三项加权组合：
```
CSI = 0.4 * CV(||w||_trajectory) + 0.3 * log(kappa(H_final)) + 0.3 * CV(eff_LR_layers)
```

**问题**:
1. **量纲不一致**: CV（变异系数）是无量纲的比率（通常在 0-1 范围），而 `log(kappa)` 对于病态矩阵可以远大于 1（深度网络中 Hessian 条件数可达 10^4，log kappa ~ 9.2），这导致 log kappa 项主导 CSI 计算，权重设计（0.4/0.3/0.3）实际上失去意义。
2. **近似质量未验证**: kappa 通过 power iteration 近似，但对于 ResNet-20 的最终 Hessian，power iteration 的收敛质量取决于迭代次数，结果文件中未说明。
3. **"高 CSI = 不稳定耦合"还是"高 CSI = 稳定耦合"**: 方法论和结果文件对 CSI 的解释方向相互矛盾。no_wd 有最高 CSI（0.96）且"expected as weight growth is completely unconstrained"暗示高 CSI = 不稳定，但指标名称（Coupling Stability Index）通常暗示高值 = 更稳定。需要明确 CSI 的方向性语义。
4. **标准化缺失**: 三项均未归一化到相同量纲，不同实验环境下 CSI 的绝对值无法横向比较。

### 6.3 AIS（Alignment Informativeness Score）

定义为：
```
AIS = corr(cos(w_i, g_i), delta_loss_i)   over training steps i
```

**问题**:
1. **delta_loss 的定义**: `delta_loss_i` 是步内 loss 下降量还是某个窗口内的 loss 变化？这对相关系数的符号和量级有根本影响。方法论中未说明。
2. **相关系数的选择**: 使用 Pearson 相关系数还是 Spearman 秩相关？对于非高斯分布的对齐余弦值和 loss 变化量，Spearman 更鲁棒，但方法论未指定。
3. **层间聚合方式**: "Computed per layer and averaged" 是简单均值还是加权平均（按参数量）？不同聚合方式对总体 AIS 的影响可能超过方法间差异。
4. **时间滞后问题**: 权重-梯度对齐（cos(w_i, g_i)）和 loss 变化之间存在固有时间滞后（当前步的 loss 变化主要反映上一步的梯度方向），直接计算同步相关系数会低估真实关联。
5. **AIS > 0.2 = "informative" 阈值的来源**: 方法论给出了 AIS > 0.2 为 informative 的判断标准，但未说明这个阈值的理论依据或校准来源。

---

## 7. 建议补充的实验

### 7.1 统计功效补救（优先级：极高）

在不增加计算成本的前提下提升统计结论可信度：

1. **增加种子数量至 5-8**: 仅需重跑 2-5 个额外种子，将 t-test 自由度从 2 提升到 4-7，检验功效显著提高。建议最低 5 个种子（df=4, t_crit=2.776）。
2. **执行 Bonferroni 校正并报告调整后 p 值**。
3. **计算并报告 Cohen's d**: 使用现有 3 个种子的数据即可计算，无需新实验。

### 7.2 关键对照组补充（优先级：高）

4. **Inverted mask（C3）实验**: 必须执行以支持 H4"alignment 方向有意义"。如果 inverted mask 与 CWD 性能相当，则 alignment 假设被推翻；如果更差，则支持 alignment 方向确实重要。这是整个 CWD 假说检验的核心控制。
5. **SGD baseline**: 补充 SGD + constant WD 基线，使"AdamW 自适应学习率已提供足够正则化"的解释有对照支撑。
6. **per-method 最优超参数实验**: 为至少 CWD 和 SWD 执行独立超参数搜索（LR ∈ {1e-4, 3e-4, 1e-3} × WD ∈ {1e-4, 5e-4, 1e-3}），排除"hyperparameter mismatch"的替代解释。

### 7.3 规模验证实验（优先级：高，论文发表必需）

7. **CIFAR-100 + ResNet-50 实验**: 比 ImageNet 计算成本低（可在 4 GPU 上 ~2h 完成），但能验证更大模型下的结论一致性。
8. **ImageNet Phase 3（6 方法 × 3 seeds × 90 epochs）**: 方法论已规划，属于当前最重要的未完成工作。在仅有 CIFAR 结果的情况下向 ICML/NeurIPS 投稿负面结论将面临严峻审稿压力。

### 7.4 消融实验（优先级：中）

9. **WD Warmup-K 扫描（H2）**: K ∈ {1, 10, 50, 200, 1000}，WD=5e-3 条件下，验证稳定性条件的实际影响。这是唯一一个能产生清晰理论预测（K < K_critical → loss spike）并有望展示显著差异的假说。
10. **Soft CWD beta 扫描（H1）**: beta ∈ {10, 50, 100, 500, 1000}，验证 hard/soft CWD 的近似质量，同时提供连续版本的额外参照点。
11. **BEM baseline 匹配验证**: 对每种动态 WD 方法，明确计算其有效 WD 积分，并对比一个积分等值的 constant WD，确保 BEM 对照的严格性。

### 7.5 指标验证实验（优先级：中）

12. **CSI/AIS 与已知正则化强度的相关性验证**: 在 WD ∈ {0, 1e-5, 1e-4, 5e-4, 5e-3, 5e-2} 的扫描中计算 CSI/AIS，如果这两个指标有意义，它们应该单调反映正则化强度的变化。如果不单调，指标定义需要重新审视。

---

## 8. 结论

当前实验数据在内部效度上存在三个值得关注的隐患：未执行 per-method 超参数搜索（可能对某些方法不公平）、关键对照组缺失（无法充分检验 H4）、统计功效不足（3 个种子的 t-test 在当前效应量下几乎必然不显著）。这些问题不会推翻现有的负面结论，但会使得该结论的置信度难以令同行信服。

外部效度方面，结论目前严格局限于 CIFAR/ResNet-20/AdamW 条件，缺乏向 ImageNet 规模的迁移验证。在没有 Phase 3 数据的情况下，"动态 WD 无益"的结论对于顶会投稿而言是不充分的。

CSI 和 AIS 指标在数学定义上存在量纲不一致（CSI）、时间滞后未处理（AIS）、方向语义模糊（CSI）等问题，需要在论文中更精确地形式化，或通过针对性实验（建议 12）验证其有效性。

可复现性整体尚可，但仍需补充 Git 版本标识、完整训练配置 JSON、以及明确的随机种子设置范围（特别是 CUDA determinism 配置），才能达到 ML 顶会可复现性审查的标准。
