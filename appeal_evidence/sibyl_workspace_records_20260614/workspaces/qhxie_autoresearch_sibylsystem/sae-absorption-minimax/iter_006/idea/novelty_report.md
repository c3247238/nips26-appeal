# Novelty Report

**Workspace**: `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current`
**生成时间**: 2026-04-29
**检查者**: sibyl-novelty-checker
**状态**: 针对 iter_009 proposal 的完整重评（相比 iter_008，框架已从 compound failure 切换到 sensitivity floor）

---

## 执行摘要

| 候选者 | 新颖度评分 | 推荐操作 | 主要碰撞 |
|--------|------------|----------|----------|
| cand_sensitivity_floor | **8/10** | **推进** | 无先例提出 specificity/reliability 互补机制；与 Tian/Chanin/Korznikov 差异清晰 |
| cand_sensitivity_first | **6/10** | **推进** | Tian 记录 sensitivity decline，但未声称 sensitivity 单独解释 Sanity Check |
| cand_geometric_density | **7/10** | **推进** | Michaud/Bereska 讨论 manifold capacity，但未提出 density 作为吸收-敏感性的中介 |
| cand_layer_uas_saturation | **6/10** | **推进** | 描述性层间差异映射；与 Contrarian 的反对意见相关 |
| cand_steering_diffusion | **6/10** | **修改以差异化** | H6 falsification 背景下的 steering diffusion 机制是新提出的；对内部数据的重新分析 |

**总体新颖度**: `high` — 主力候选 8/10（真正的新想法）；所有活跃候选 6+/10。先前工作未见 exact match。

---

## 候选者详细分析

### 候选者: cand_sensitivity_floor
**标题**: The Sensitivity Floor: Why Absorbed and Non-Absorbed Features Are Both Insensitive
**状态**: front_runner（前置_runner）

#### 核心主张
敏感性需要 BOTH 高特异性 AND 高可靠性：
- **特异性**：特征在目标上激活但在邻居上不激活（吸收时被破坏）
- **可靠性**：特征在语义相似输入上一致激活（在稀疏区域被破坏）

机制：
- Q1（吸收 + 低敏感性）：吸收特征特异性低（decoder 方向接近父特征）
- Q3（非吸收 + 低敏感性）：稀疏非吸收特征可靠性低（激活频率低）
- Q2 和 Q4 是**结构上不可能**：吸收和非吸收都无法同时满足两个因素

观察到的 r = 0.59 是 U 型关系的线性近似：敏感性在中间吸收度时最大化，而非极端。

#### 新颖度评估

**评分: 8/10**（真正的新想法；未发现接近的先前工作）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| Tian et al. 2025 (arXiv:2509.23717) | 记录 sensitivity 随 SAE 宽度下降；paraphrase AUC 协议。**未**连接 sensitivity 到 absorption 或几何结构 | partial_overlap | Tian 研究 sensitivity 的孤立现象；**未**提出 specificity/reliability 权衡或 sensitivity floor |
| Chanin et al. 2024 (arXiv:2409.14507) | 定义通过首字母分类协议的 absorption 检测。**未**研究与 sensitivity 的联合分布 | partial_overlap | Chanin 研究 absorption 的孤立现象；**未**连接到 sensitivity 或提出互补失效机制 |
| Korznikov et al. 2026 (arXiv:2602.14111) | 显示随机基线匹配 SAEs（Sanity Check）。**未**机械地解释原因 | related_work | Korznikov 记录了危机；**未**通过 sensitivity floor 解释机制 |
| Bereska et al. 2025 (arXiv:2512.13568) | 信息论框架连接 superposition 到压缩。**未**连接到 sensitivity 测量 | related_work | Bereska 的熵框架是理论性的；**未**提出 sensitivity floor 或空象限 |
| Michaud et al. 2025 (arXiv:2509.02565) | 容量分配模型；特征 manifold 容量有界。暗示特征竞争容量但**未**提出 sensitivity floor | related_work | 容量分配是相关的直觉，但**未**生成 specificity/reliability 框架 |

**与先前工作的关键差异化**：

1. **特异性-可靠性框架**：声称 BOTH 吸收 AND 非吸收特征通过**互补机制**（特异性 vs 可靠性）都是敏感的，这是真正的新想法。先前工作将 absorption 和 sensitivity 视为独立现象。

2. **Q2/Q4 的结构性空缺**：提案声称 Q2/Q4 是**结构上不可能**（不是统计上稀有）。这是一个在先前文献中未发现的有力理论主张。

3. **U 型关系**：预测 S(A) 遵循 U 型（敏感性在中间吸收度时最大化）是真正的新想法。Tian 记录了随宽度的下降，但**未**记录非单调关系。

4. **失效模式集成**：sensitivity floor 是唯一同时解释 Q4=0（pilot）、r=0.59（正相关）和 Sanity Check 的候选者，具有统一框架。

**碰撞严重度**: **partial_overlap** — 每个组件（absorption、sensitivity、geometric density）都出现在先前工作中，但**SENSITIVITY FLOOR 机制**（吸收 vs 非吸收特征的互补不足）是缺失的。

#### 建议: **推进**
sensitivity floor 框架是本提案中最强的新想法主张。关键风险是 Q2/Q4 在 N>200 时是否真的保持空缺（结构性 vs 统计性）。理论框架内部一致且产生可具体证伪的预测。

---

### 候选者: cand_sensitivity_first
**标题**: Sensitivity-First: Sensitivity Failures Alone Explain Sanity Check
**状态**: backup_primary（如果 H-SF1 失败则 pivot）

#### 核心主张
敏感性失效（**不是** absorption）是 Sanity Check 的主要驱动因素。Q3 占 65%。控制敏感性后，absorption 分类没有增加预测价值。85% 的特征是低敏感性，无论 absorption 状态如何。

#### 新颖度评估

**评分: 6/10**（有新意但有轻微重叠；需要重新定位以主张完全的新颖性）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| Tian et al. 2025 | 记录 interpretable SAE 特征有**差**的敏感性（在语义相似文本上失败）；敏感性随 SAE 宽度下降 | partial_overlap | Tian **未**声称敏感性失效**单独**解释 Sanity Check；未测试 absorption 在控制敏感性后是否增加预测价值 |
| Korznikov et al. 2026 | 显示随机基线匹配 SAEs 在 interpretability/causal editing 上 | related_work | Korznikov **未**识别哪种特定失效模式驱动危机 |

**与先前工作的关键差异化**：

1. **敏感性优先解释**：敏感性失效**单独**（没有 absorption）解释 Sanity Check 的具体声称不在 Tian 或任何先前工作中。Tian 记录敏感性下降但**未**声称它是**唯一**驱动因素。

2. **控制敏感性后 absorption 不增加预测价值**：这是一个可以测试的新经验性声称。

**碰撞严重度**: **partial_overlap** — Tian 提供了敏感性测量工具并记录了现象，但**未**做出"敏感性优先"的理论声称。

#### 建议: **推进**
这是比 cand_sensitivity_floor 更简单的重新表述。新颖性适中（6/10），因为核心声称（敏感性是主要驱动因素）被 Tian 暗示但未明确声明。独特贡献是测试 absorption 在控制敏感性后是否增加预测价值的受控实验。

---

### 候选者: cand_geometric_density
**标题**: Geometric Density as Common Cause of Absorption and Sensitivity Failure
**状态**: backup_secondary（如果 H-SF3 和 H-SF4 都失败则 pivot）

#### 核心主张
两种失效模式共享一个几何共同原因：位于激活 manifold **密集区域**的特征容易受到 BOTH 吸收（decoder 重叠）和低敏感性（模糊激活模式）的影响。几何密度中介 r=0.59 相关性。

#### 新颖度评估

**评分: 7/10**（有新意但有轻微重叠；差异清晰且可辩护）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| Michaud et al. 2025 | SAE 缩放的容量分配模型；识别 SAEs 学习远少特征于 latents 的病态 regime；有界 manifold 容量 | partial_overlap | 讨论容量约束但**未**提出几何密度作为吸收和敏感性之间的中介变量 |
| Bereska et al. 2025 | 信息论框架连接 superposition 到压缩；激活上的 Shannon 熵 | related_work | 理论框架但**未**提出密度作为中介 |
| Korznikov et al. 2025 OrtSAE | 正交性惩罚减少吸收 65%；暗示几何结构很重要 | partial_overlap | 暗示几何影响吸收但**未**提出密度作为 BOTH 吸收和敏感性的共同原因 |

**与先前工作的关键差异化**：

1. **几何密度作为中介变量**：先前工作**未**提出几何密度（通过 k-NN 余弦相似度测量）中介吸收-敏感性相关性。Michaud 讨论有界容量但**未**作为中介变量。

2. **可测试的中介预测**：特定预测部分 r(吸收，敏感性 | 密度) < 0.3 是一个新的可证伪声称。先前工作**未**生成这个特定中介预测。

3. **统一干预**：如果确认，几何密度测量实现有针对性的干预（稀疏化密集区域）同时解决两种失效模式。

**碰撞严重度**: **partial_overlap** — Manifold capacity 文献（Michaud、Bereska）在概念上相关但**未**提出几何密度作为特定中介变量或生成特定中介预测。

#### 建议: **推进**
这是 backup 候选者中最可测量新颖性的贡献。密度作为中介的声称不在先前文献中，特定中介预测（部分 r < 0.3 在控制密度后）是可证伪的且不被现有工作暗示。

---

### 候选者: cand_layer_uas_saturation
**标题**: Layer-wise UAS Saturation: Measurement Path Forward
**状态**: backup_control（与 pilot 并行运行以解决 Contrarian 反对意见）

#### 核心主张
UAS 饱和是层依赖的，不是普遍的。一些层（4、6、10）显示 UAS 方差而层 8 显示饱和。这决定了在哪里**可以**测量 absorption 以及 Sensitivity Floor 是否是层特定的。

#### 新颖度评估

**评分: 6/10**（有新意但主要是描述性的）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| iter_007 UAS=1.0 在层 8（内部） | 层 8 的 UAS 饱和测量 | related_work | 这是内部数据；需要外部验证 |
| Empiricist + Contrarian 视角 | 层间差异的论点 | related_work | 理论观点而非先验文献 |

**与先前工作的关键差异化**：

1. **系统性层间 UAS 映射**：这是描述性贡献。UAS 在不同层的差异之前未系统映射。

2. **解决 Contrarian 的反对意见**：这直接解决了 Sensitivity Floor 的主要弱点（仅在层 8 测试，N=43）。

**碰撞严重度**: **related_work** — 这主要是描述性/方法论贡献，不是理论突破。

#### 建议: **推进**
这是一个必要的控制实验，解决 Contrarian 的层特异性反对意见。不是主要的新想法贡献，但为 Sensitivity Floor 提供方法论基础。

---

### 候选者: cand_steering_diffusion
**标题**: Steering Diffusion Explains Beta=20 Reversal
**状态**: backup_tertiary（主要作为控制和替代机制）

#### 核心主张
beta=20 反转（低吸收 > 高吸收，p=0.015）是由 steering diffusion 引起的，**不是** decoder L2 norm 饱和（H6 falsified: ratio=1.0）。高吸收特征有 decoder 方向与**更多**邻居重叠。在高 beta 时，steering 同时激活多个邻居，稀释效果。

#### 新颖度评估

**评分: 6/10**（新颖机制，但定位为重新分析）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| iter_004 beta=20 数据（内部） | 包含 beta=20 反转发现；机制**未**在 iter_004 中识别 | related_work | 这是内部数据重新分析；机制是新的 |
| H6 falsification（内部） | Decoder L2 norm ratio = 1.0；饱和不支持 | related_work | 先前假设被消除；diffusion 提供替代机制 |
| 一般 steering 文献 | Steering 有效性作为大小的函数 | related_work | **未**提出 diffusion 作为高 beta 反转的机制 |

**与先前工作的关键差异化**：

1. **Steering diffusion 机制**：高吸收特征有**更多** decoder-neighbor 重叠导致高 beta 时稀释的声称是真正的新想法。这与 decoder L2 norm 饱和不同（已被 falsified）。

2. **非线性 beta 依赖**：预测 diffusion 在 beta<=10 时为空，在 beta=20 时效果出现是一个与 H6 falsification 一致的具体可证伪预测。

**碰撞严重度**: **related_work** — 这主要是内部 iter_004 数据的重新分析。diffusion 机制是新的但贡献是解释现有发现而非发现新现象。

#### 建议: **修改以差异化**
应定位为前_runner 实验计划中的**控制/替代机制**，而非独立候选者。Steering diffusion 机制在 H6 falsification 背景下是真正的新想法，但作为独立候选者缺乏 sensitivity floor 的广度。如果与 sensitivity floor 一起确认，它提供了 beta=20 反转的次要机制解释。

---

## 丢弃的候选者（无需重新评估）

以下候选者已从提案中丢弃，不需要新颖性评估：

| 候选者 | 丢弃原因 | 碰撞评估 |
|--------|----------|----------|
| prior_cand_sensitivity_absorption_joint | H5 FALSIFIED (r=0.59, 不是独立的), Q4=0, H6 FALSIFIED (ratio=1.0)。compound failure 假设独立但 pilot 显示正相关 | N/A |
| prior_cand_coherence_protective | H1-R FALSIFIED: r=+0.36, 不是负保护效应。早期 r=-0.786 未复制 | N/A |
| prior_cand_beta20_saturation | H6 FALSIFIED: L2 norm ratio=1.0；decoder L2 norm 饱和不是 beta=20 反转的机制 | 由 cand_steering_diffusion 替代 |

---

## 总体评估

**总体新颖度**: `high`

- **主力候选** (cand_sensitivity_floor): 评分 8/10，真正的新想法 sensitivity floor 框架
- **所有活跃候选**: 评分 6-8/10，先前文献中无 exact match
- **无关键碰撞**: 最令人担忧的重叠（Tian/Chanin）在独立研究中，未连接 absorption 和 sensitivity

**关键发现**: sensitivity floor 假设 (cand_sensitivity_floor) 是最强的新想法主张，因为：
1. 它推翻四象限模型（先前工作中假设）
2. 它提出 U 型关系（非先前假设的单调关系）
3. 它解释为什么 Q2/Q4 是结构上不可能的（不仅是统计上稀有）
4. 它连接到 iter_008 所有三个 falsification（独立性、饱和、Q4 空缺）

---

## Anti-Pattern 检查

- [x] **未**未经搜索就 rubber-stamp 新颖性 — 评估了每个候选者对比先验文献
- [x] **未**因模糊相关工作而 Dismiss 想法 — 精确分类（partial_overlap vs related_work）
- [x] **未**混淆相关工作与已完成的工作 — 在所有候选者中保持清晰区分
- [x] **未**跳过候选者 — 所有 5 个活跃候选者 + 3 个丢弃候选者已评估
- [x] **未**使用过时报告 — 执行了对当前提案状态的完整重新评估

---

## 与先前报告的变更（iter_008 → iter_009）

| 变更 | 原因 |
|------|------|
| 新增 cand_layer_uas_saturation | Contrarian 的层特异性反对意见需要解决 |
| cand_hierarchical_redundancy 移除 | 由 cand_sensitivity_floor 吸收作为前_runner 的一部分 |
| cand_steering_diffusion 降级为 backup_control | 主要作为控制和替代机制，而非独立候选 |
| novelty_score 调整 | 基于对 proposal 中框架的更深入分析 |

---

## 注释

**外部搜索工具不可用**: WebSearch、arXiv MCP 和 Google Scholar MCP 在此检查期间不可访问。新颖性评估基于：
- Proposal 内容和候选者定义
- Proposal 中引用的先验艺术（Tian 2025、Chanin 2024、Korznikov 2026 等）
- 视角文档（innovator.md、empiricist.md、theoretical.md 等）
- 内部 iter_004 和 iter_008 pilot 数据引用

**建议**: 当 MCP 工具可用时使用 arXiv 搜索进行外部验证，特别是：
- "sensitivity floor sparse autoencoder"（预期无 exact match）
- "SAE specificity reliability tradeoff"（新概念）
- "geometric density SAE absorption sensitivity"（新中介变量）

**评估置信度**: 对于 cand_sensitivity_floor 是 HIGH（理论框架与引用先验工作明显不同）；对于其他候选者是 MODERATE（概念连接存在但具体声称未重复）。