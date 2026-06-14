# 创新者提案：遮蔽扩散语言模型的推理时计算扩展（第 4 轮迭代）

**日期**: 2026-03-10
**视角**: 大胆创新、跨领域迁移、反直觉思路

---

## 0. 前情诊断：实验数据的精确画面

经过 18 轮前序迭代 + 第 3 轮完整实验，当前数据构成了清晰的路线图：

**已确认的死路**：
- TTT 6 个变体在 AR 框架下统计不显著 (p=0.88)
- Best-of-N 3x 计算量反而 PPL +6.9%
- Pure remasking (ReMDM-conf 4.4%, RCR 5.7%) 在 Countdown-500 上无改善 vs vanilla 4.7%
- DTA (LoRA 在线更新) pilot 6.2% < vanilla 12.5%——梯度信号太弱（MLM loss 已在 0.005-0.032）
- SCP 计算开销 12x 但无显著提升

**唯一确认的正面信号——DMI**：
- Countdown-500 三 seed 平均 **9.3% vs vanilla 4.7%**——近 2x 改善
- 零额外计算开销（仅缓存上一步 logits + softmax @ embedding 加权平均）
- 这是整个项目 20+ 轮迭代中**唯一**在 full-scale、多 seed 下有实质改善的方法

**新发现的关键文献（本轮搜索新增）**：

1. **LRD (Latent Refinement Decoding, arXiv 2510.11052)**：保留 mask 位置的分布混合态（而非硬 mask），让模型建立全局一致的"信念状态"。GSM8K +2.9, MATH500 +3.8, 同时加速 10.6x。**与 DMI 的核心思想高度相关**——都在探索"保留连续信息替代硬 mask"。

2. **ReMix (Rejection Mixing, arXiv 2602.22868)**：引入 Continuous Mixing State 作为 mask 和 decoded token 之间的中间态，连续空间精化后再坍缩为离散 token。Training-free, 2-8x 加速无质量下降。

3. **A-CFG (Adaptive Classifier-Free Guidance, arXiv 2505.20199)**：基于模型预测置信度动态构造无条件输入（低置信 token 临时 re-mask），GSM8K 73.5（超越 LLaMA3 8B 的 53.1），GPQA +3.9。**直接在推理任务上证明了 CFG 对 DLM 推理的有效性**。

4. **CFG 时间调度改进 (arXiv 2507.08965)**：理论证明高 guidance 在早期（高 mask rate）有害，后期（低 mask rate）有益。提出改进的 CFG 机制，一行代码即可实现。

5. **EvoToken-DLM (arXiv 2601.07351)**：用连续 token 进化替代硬 mask，渐进式更新 token 分布。

6. **PRR (Progressive Refinement Regulation, arXiv 2603.04514)**：学习轻量 token 级控制器调节精化温度，利用跨步信息加速解码。

**我的核心诊断**：第 3 轮提出的 ASD（Adaptive Soft Diffusion）方向是正确的——DMI 的成功 + LRD/ReMix/EvoToken 的独立验证共同指向一个核心洞察：**DLM 的根本瓶颈在于 mask embedding 的信息贫乏——用富含语义的连续表示替代硬 mask 是正确方向**。但第 3 轮的 ASD 设计过于保守（仅在 embedding 层做混合），且忽视了 A-CFG 这一强大的正交工具。本轮提案将沿这一方向做更大胆、更深层的创新。

---

## 1. 角度一：Belief-State Diffusion (BSD) —— 将整个去噪过程从离散跳跃重构为连续信念演化（改进现有方法）

### 1.1 核心洞察

DMI 成功、DTA 失败的对比揭示了一个关键教训：**改善 DLM 推理的最优杠杆不在参数空间（梯度更新），而在表示空间（token 的连续表征）**。

当前 DLM 的去噪过程在表示空间是一个"硬跳跃"过程：mask embedding → discrete token embedding，每步的连续信息在 argmax 采样时被截断。LRD 证明保留分布混合态（distributional mixture）可以在多个推理基准上同时提升质量和速度。ReMix 独立验证了连续中间态的价值。

但 LRD 和 ReMix 都有一个共同局限：它们是**解码加速方法**，聚焦于用更少的步数达到相同质量，而非用相同步数达到更高质量。我们可以翻转这个视角：**如果连续表示在加速时不损失质量，那在质量导向场景下用更多步，应该能带来实质性提升**。

### 1.2 提案：Belief-State Diffusion (BSD)

**核心 idea**：将 mask 位置的表示从硬 mask embedding 替换为"信念状态"（belief state）——一个由模型前一步预测分布加权的连续混合 embedding，每步迭代精化这个信念而非做硬采样。只在最后 k 步才做硬揭示。

**具体方案**：

1. **信念状态定义**：对每个 mask 位置 i，维护一个连续信念向量：
   $$b_i^t = \sum_{v \in V} P_\theta(x_i = v | x_t) \cdot e_v$$
   其中 $e_v$ 是 token v 的 embedding，$P_\theta$ 是模型预测概率。这等价于 DMI 的 logit-weighted embedding，但 BSD 将其作为**主要表示**（而非仅作为混合注入）。

2. **信念精化循环（前 T-k 步）**：
   - 输入：prompt 的 token embedding + 生成位置的信念向量 $b^t$
   - 模型前向传播：$\ell^t = f_\theta(\text{prompt} \oplus b^t)$
   - 信念更新：$b_i^{t+1} = (1-\alpha^t) \cdot b_i^t + \alpha^t \cdot \sum_v \text{softmax}(\ell^t_i)_v \cdot e_v$
   - 关键：**不做 argmax 采样，不做 mask/unmask**，信念向量在连续空间平滑演化

3. **硬揭示阶段（最后 k 步）**：
   - 从信念向量 $b^{T-k}$ 出发，恢复标准的 mask-unmask 流程
   - 按置信度排序揭示 token
   - 此时信念向量已高度集中（低熵），揭示的 token 质量远高于从随机 mask 开始

4. **与 DMI 的关键区别**：
   - DMI：mask embedding + alpha * soft_embedding（固定混合）
   - BSD：**完全用信念向量替代 mask embedding**，连续演化 T-k 步
   - BSD 的信念向量在前向传播时直接输入 Transformer，无需额外的混合操作
   - BSD 利用全部 T 步做精化，DMI 每步独立

5. **与 LRD 的关键区别**：
   - LRD：mask embedding 和预测 embedding 的固定比例混合 + KL 收敛检测（目标是加速）
   - BSD：**质量导向**的连续信念演化 + 自适应更新率 + 仅最后 k 步硬揭示
   - BSD 的更新率 α^t 随去噪进度动态调整（早期小/稳定，后期大/收敛）
   - BSD 不做早停，用全部计算预算最大化质量

### 1.3 理论动机

**信息论视角**：标准 DLM 的 argmax 采样在每步引入量化噪声——logit 分布的尾部信息被丢弃。对于推理任务，这些尾部信息中包含关键的替代路径。BSD 的信念向量保留完整分布信息，等价于用无限精度的"软 token"进行去噪。

**与连续扩散的桥梁**：BSD 可以被视为将连续扩散模型的精化过程（逐步降噪 x_t → x_0）嫁接到离散扩散的框架中。连续扩散的成功部分归因于表示空间的平滑性——BSD 将这种平滑性引入 masked diffusion。

### 1.4 假设

**H1（主假设）**：BSD 在 Dream-7B Countdown-500 上的准确率将显著优于 DMI（预期 14-18% vs DMI 9.3%），因为连续信念演化避免了硬采样的量化损失，且利用全部去噪步做精化。

**H2**：BSD 的信念向量在去噪过程中熵单调递减（信息论预测），且最终熵低于 vanilla 去噪的对应指标——这意味着 BSD 在揭示时做出更确定的决策。

**H3**：BSD 与 A-CFG 正交互补——A-CFG 改善单步预测分布的质量，BSD 改善跨步信息传递。

### 1.5 实验计划

| 实验 | 模型 | 基准 | 指标 | GPU 时间 |
|------|------|------|------|---------|
| BSD 概念验证 | Dream-7B | Countdown-16 pilot | Accuracy | ~0.5h |
| BSD k 消融 (k=T/4, T/2, 3T/4) | Dream-7B | Countdown-100 | Accuracy vs k | ~4h |
| BSD alpha 策略消融 | Dream-7B | Countdown-100 | Accuracy vs alpha schedule | ~4h |
| BSD vs DMI/vanilla | Dream-7B | Countdown-500 x 3 seeds | Accuracy, McNemar | ~8h |
| BSD + A-CFG | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~12h |
| BSD on GSM8K | Dream-7B | GSM8K-1319 x 1 seed | Accuracy | ~6h |

**计算成本**：~35 GPU·h（4x GPU 约 9h），**成功概率 55%**

### 1.6 失败模式

- **分布外输入**（概率 30%）：信念向量 $b^t$ 不在模型训练分布中（训练时只见过 mask embedding 和 real token embedding）。缓解：(1) 归一化 $b^t$ 使其范数与 token embedding 一致；(2) 如果完全替代失败，降级为 DMI 式混合（$\beta \cdot \text{mask\_emb} + (1-\beta) \cdot b^t$），$\beta$ 从 0.9 线性衰减到 0.1
- **信念振荡不收敛**（概率 20%）：logit 分布在步间来回翻转。缓解：EMA 更新（$\alpha^t$ 小于 0.5）+ 温度退火（后期降低采样温度）
- **推理任务特殊性**：数学推理中 token 间有强依赖，连续混合可能破坏算术结构。缓解：先在 Countdown（结构化推理）验证，再扩展到 GSM8K（自由文本推理）

---

## 2. 角度二：Adaptive Classifier-Free Guidance for Reasoning (ACFG-R) —— 将 A-CFG 与跨步信息融合构建推理专用引导系统（跨领域迁移）

### 2.1 跨领域灵感

A-CFG (arXiv 2505.20199) 是一个被当前项目**完全忽视**但极其强大的工具——它在 LLaDA-8B 上将 GSM8K 从标准水平推到 73.5（超越 LLaMA3 8B 的 53.1），GPQA +3.9。这不是边际改善，这是**变革性**的提升。

A-CFG 的核心思想简洁到令人惊叹：在每步去噪时，找到模型最不确信的 token，临时把它们 re-mask 掉，得到一个"无条件"输入。然后用条件预测（正常输入）和无条件预测（re-masked 输入）做 CFG 引导。这利用了 DLM 的天然 mask 机制——不需要像 AR 模型那样训练两个模型或做 dropout。

但 A-CFG 有两个关键局限：
1. **只看当前步的置信度**——无跨步记忆，无法利用前步积累的推理上下文
2. **re-mask 策略是静态的**——总是 mask 低置信 token，但在推理任务中，"关键 token"和"低置信 token"并不总是同一组

### 2.2 提案：ACFG-R — 融合跨步记忆的推理导向 CFG

**核心 idea**：将 A-CFG 的 CFG 引导与 DMI 的跨步记忆融合，构建一个"推理感知"的引导系统——用跨步信念稳定性（而非单步置信度）识别关键 token，用 CFG 放大这些 token 的推理信号。

**具体方案**：

1. **跨步不稳定性信号**（DMI 增强版）：
   - 维护每个位置的 logit EMA：$\bar{\ell}_i^t = \lambda \bar{\ell}_i^{t-1} + (1-\lambda) \ell_i^t$
   - 计算跨步稳定性：$S_i^t = 1 - \text{JSD}(P_i^t \| P_i^{t-1})$（Jensen-Shannon 散度）
   - 低稳定性 = 模型在这个位置"犹豫不决" = 推理的关键决策点

2. **稳定性引导的 CFG 构造**：
   - 选择 top-m% 最不稳定位置（而非 A-CFG 的最低置信位置）临时 re-mask
   - 两次前向传播：$\ell^+ = f_\theta(x_t)$, $\ell^- = f_\theta(\tilde{x}_t)$
   - CFG 引导：$\ell_{\text{guided}}^i = \ell^{+,i} + w(S_i^t) \cdot (\ell^{+,i} - \ell^{-,i})$
   - 引导强度 $w(S_i^t)$ 与不稳定性正相关：不稳定位置获得更强引导

3. **时间调度**（基于 CFG 调度理论 arXiv 2507.08965）：
   - 早期步骤（mask rate > 70%）：低/无 guidance（模型信息不足，高 guidance 有害）
   - 中期步骤（mask rate 30-70%）：逐步增加 guidance（推理核心窗口）
   - 后期步骤（mask rate < 30%）：最高 guidance（精化关键 token）

4. **与 BSD 的组合**：
   - BSD 在前 T-k 步提供连续信念精化
   - ACFG-R 在最后 k 步的硬揭示阶段引导关键 token 选择
   - 两者互补：BSD 提供信息积累，ACFG-R 提供信号放大

### 2.3 与 A-CFG 和 CORE 的精确定位

| 维度 | A-CFG | CORE | ACFG-R (Ours) |
|------|-------|------|---------------|
| re-mask 信号 | 单步低置信 | 单步上下文扰动 | **跨步不稳定性** |
| 引导方式 | CFG logit 外推 | Remask 候选筛选 | **稳定性加权 CFG** |
| 跨步记忆 | 无 | 无 | **有（logit EMA）** |
| 时间调度 | 固定 w | 固定扰动率 | **理论驱动退火** |
| 额外前向传播 | 1x | K次 | **1x** |
| 推理任务验证 | **GSM8K 73.5** | MBPP +9.2 | 预期更高 |

### 2.4 假设

**H4**：ACFG-R 在 Dream-7B Countdown-500 上的准确率将显著优于 vanilla 和 DMI（预期 15-20%），因为 CFG 引导直接放大推理关键信号。

**H5**：稳定性引导的 re-mask（跨步 JSD）优于 A-CFG 的单步置信度 re-mask，因为跨步信号消除了单步采样噪声。

**H6**：CFG 时间调度（早低后高）优于固定 guidance weight，验证理论预测。

### 2.5 实验计划

| 实验 | 模型 | 基准 | 指标 | GPU 时间 |
|------|------|------|------|---------|
| A-CFG 复现 | Dream-7B | Countdown-100 | Accuracy | ~2h |
| ACFG-R 概念验证 | Dream-7B | Countdown-16 pilot | Accuracy | ~0.5h |
| 稳定性 vs 置信度 re-mask | Dream-7B | Countdown-100 | Accuracy | ~4h |
| ACFG-R 时间调度消融 | Dream-7B | Countdown-100 | Accuracy vs schedule | ~4h |
| ACFG-R vs A-CFG vs vanilla/DMI | Dream-7B | Countdown-500 x 3 seeds | Accuracy, McNemar | ~12h |
| ACFG-R + BSD | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~12h |
| ACFG-R on GSM8K | Dream-7B | GSM8K-1319 x 1 seed | Accuracy | ~8h |

**计算成本**：~43 GPU·h（4x GPU 约 11h），**成功概率 60%**

### 2.6 失败模式

- **Dream-7B 不支持 CFG**（概率 15%）：Dream 的条件/无条件训练方式可能与 CFG 不兼容。缓解：先确认 Dream-7B 是否支持标准 CFG（检查有无无条件训练），如不支持则转向 LLaDA-8B（A-CFG 论文已在 LLaDA 上验证）
- **跨步 JSD 计算不稳定**（概率 10%）：概率分布噪声大。缓解：使用 EMA 平滑后的分布计算 JSD
- **guidance 过强导致退化**（概率 25%）：CFG 外推产生 OOD logits。缓解：guidance 上限 w_max = 2.0，配合温度校准

---

## 3. 角度三：Dual-Granularity Denoising (DGD) —— 同时在 token 级和 chunk 级做去噪（全新方法）

### 3.1 核心动机

所有现有 DLM TTS 方法都在**同一粒度**上操作——token 级。但推理任务具有**层次结构**：
- **全局级**：推理步骤的顺序和结构（先做什么再做什么）
- **局部级**：每个推理步骤内的具体 token（数字、操作符、变量名）

当前 DLM 的去噪过程试图同时解决这两个层次——在一个单一的 mask-unmask 过程中既确定全局结构又确定局部细节。但理论和实验都表明这是次优的：
- TAPS (arXiv 2601.22629) 发现 DLM 有**时序分工**——早期步决定全局语义，后期步精化局部词汇
- BACD+TCCF (arXiv 2602.09555) 的 "Think Coarse, Critic Fine" 在块扩散中验证了双粒度策略的有效性（AIME24 +11.2）
- MGDM (arXiv 2410.14157) 通过多粒度扩散在 Countdown 达到 91.5%（vs AR 45.8%），虽然需要训练

### 3.2 提案：Dual-Granularity Denoising (DGD)

**核心 idea**：将去噪过程分为两个交错的粒度——chunk 级（确定推理结构的"骨架"）和 token 级（精化骨架中的具体内容）。这是一种 training-free 的方法，利用现有 DLM 的能力在不同抽象层次上做去噪。

**具体方案**：

1. **Phase 1: 骨架去噪（前 T/2 步）**：
   - 将生成序列分为 C 个 chunk（每个 chunk ~16-32 token）
   - 每个去噪步只揭示每个 chunk 中**最确信的 1-2 个 token**（"锚点 token"）
   - 效果：快速建立推理的全局骨架——"24 = ___ + ___ + ___" 或 "Step 1: ___ Step 2: ___"
   - 锚点选择：跨 chunk 的全局 top-k 置信度排序，但保证每个 chunk 至少有 1 个锚点

2. **Phase 2: 精化去噪（后 T/2 步）**：
   - 基于骨架中的锚点 token 做标准去噪
   - 此时已揭示的锚点为每个 chunk 提供了上下文约束
   - 锚点之间的 token 受到**双向约束**——前后锚点共同约束中间 token 的选择
   - 可选：在这个阶段加入 BSD 的信念精化或 ACFG-R 的 CFG 引导

3. **锚点质量验证**（Phase 1 → 2 过渡时）：
   - 用模型的 leave-one-out 概率验证每个锚点 token
   - 低质量锚点（P(anchor | context \ anchor) < 阈值）remask 重新选择
   - 这是一个低成本的"结构审查"步骤

4. **与 TAPS 的互补性**：
   - TAPS 在早期鼓励语义分支（多样性），DGD 在早期确定结构骨架（一致性）
   - 两者可以组合：DGD 的骨架 + TAPS 在 chunk 内部的多样化精化

### 3.3 与已有方法的区别

| 维度 | BACD+TCCF | MGDM | R3 | DGD (Ours) |
|------|-----------|------|-----|-----------|
| 粒度定义 | 块大小（固定） | 子目标难度（训练学习） | PRM 打分的块 | **chunk 锚点（自适应）** |
| 训练需求 | 否 | **是** | 否（需 PRM） | **否** |
| 模型限制 | 块扩散专用 | 专用训练 | 需 PRM | **通用 MDM** |
| 适用场景 | 块扩散推理 | Countdown/Sudoku | 文本生成 | **通用推理** |
| 双粒度方式 | 大块探索+小块精化 | 难度优先学习 | 块级评审 | **骨架建立+内容精化** |

### 3.4 假设

**H7**：DGD 在 Dream-7B Countdown-500 上的准确率将显著优于 vanilla（预期 12-16%），因为先建立推理骨架再精化内容避免了结构错误的传播。

**H8**：DGD 的锚点 token 中，数字和操作符的比例显著高于非锚点 token——验证 DGD 确实在优先揭示推理关键位置。

**H9**：DGD + ACFG-R 组合将产生协同效果（预期 18-22%），因为 DGD 提供结构约束、ACFG-R 放大推理信号。

### 3.5 实验计划

| 实验 | 模型 | 基准 | 指标 | GPU 时间 |
|------|------|------|------|---------|
| DGD 概念验证 | Dream-7B | Countdown-16 pilot | Accuracy | ~0.5h |
| DGD chunk 大小消融 | Dream-7B | Countdown-100 | Accuracy vs chunk_size | ~4h |
| DGD Phase 比例消融 | Dream-7B | Countdown-100 | Accuracy vs T1/T2 ratio | ~4h |
| DGD vs vanilla/DMI | Dream-7B | Countdown-500 x 3 seeds | Accuracy, McNemar | ~8h |
| DGD + ACFG-R | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~12h |
| DGD on GSM8K | Dream-7B | GSM8K-1319 x 1 seed | Accuracy | ~8h |
| 锚点 token 类型分析 | Dream-7B | Countdown-500 | Token type distribution | ~2h |

**计算成本**：~39 GPU·h（4x GPU 约 10h），**成功概率 50%**

### 3.6 失败模式

- **骨架锚点选择错误**（概率 25%）：错误的锚点 token 导致后续精化偏离正确路径。缓解：Phase 1→2 过渡时的验证步骤；增加每 chunk 的锚点数量
- **chunk 边界切分问题**（概率 15%）：固定大小 chunk 可能切断语义单元。缓解：根据特殊 token（换行符、逗号等）做自适应切分
- **两阶段切换突兀**（概率 20%）：Phase 1 到 Phase 2 的切换可能导致不连贯。缓解：设置 2-3 步的过渡期，逐步增加每步的揭示数量

---

## 4. 综合策略与优先级排序

### 4.1 三个提案的互补性与层次关系

```
信号来源         作用层次        计算开销      与 DMI 的关系
BSD:   连续信念演化     表示空间       ~1.1x      DMI 的深度推广
ACFG-R: CFG + 跨步记忆  预测分布空间    ~2x        DMI 提供记忆，CFG 放大信号
DGD:   双粒度去噪       结构空间       ~1x        与 DMI 正交
```

三者在不同层次上操作，完全正交：
- BSD 改善**表示**（mask token 的连续化）
- ACFG-R 改善**预测**（CFG 引导推理信号）
- DGD 改善**结构**（粗到细的去噪策略）

最优组合可能是：**DGD（结构层）+ BSD（表示层）+ ACFG-R（预测层）**——三层同时优化。

### 4.2 推荐实验顺序

**Phase 1（1.5 天）——独立 pilot（三个方向并行，3x GPU 并行）**：
- GPU 0: ACFG-R pilot（Countdown-16）——因为 A-CFG 已有强验证，成功概率最高
- GPU 1: BSD pilot（Countdown-16）——DMI 的自然推广，验证连续信念可行性
- GPU 2: DGD pilot（Countdown-16）——全新方向，需要快速验证
- GPU 3: A-CFG 复现（Countdown-100）——确认 CFG 在 Dream-7B 上是否有效

**Phase 2（2 天）——最优方向 full-scale + 消融**：
- 选择 pilot 中准确率最高的方向做 Countdown-500 x 3 seeds
- 并行做关键超参数消融
- 如果多个方向都有信号，优先测试两两组合

**Phase 3（1.5 天）——组合 + 扩展**：
- 最优单方法 + 最优组合在 GSM8K 上评测
- 与已有 baseline（vanilla, DMI, ReMDM-conf）统一对比
- 统计检验

**Phase 4（1 天）——分析 + 写作准备**：
- Token 级诊断（哪些 token 被 CFG 改变、哪些成为骨架锚点）
- 信念向量熵轨迹可视化
- 与 MetaState、LRD、ReMix 的定位对比

### 4.3 投稿定位

**最佳情况（BSD + ACFG-R + DGD 至少两个成功，Countdown ≥ 18%，GSM8K 显著提升）**：
**NeurIPS/ICML 主会**——"Beyond Remasking: Multi-Layer Inference-Time Scaling for Masked Diffusion Language Models"

核心卖点：
1. 首个同时在表示层、预测层、结构层优化 DLM 推理时扩展的统一框架
2. Training-free + 多个推理基准上显著超越 remasking baseline
3. 信念状态演化的信息论分析

**中等情况（ACFG-R 单独成功，Countdown ≥ 15%）**：
**NeurIPS/ICLR 主会**——"Reasoning-Aware Guidance for Diffusion Language Models: From Confidence to Stability"

核心卖点：A-CFG 的推理增强版，跨步稳定性引导 + 理论驱动时间调度

**保底情况（仅 BSD 在 Countdown 上优于 DMI）**：
**ICLR**——"Belief-State Diffusion: Continuous Token Representations for Masked Diffusion Language Models"

聚焦 DMI → BSD 的推广，连续表示替代硬 mask 的系统研究

### 4.4 总计算预算

| 方案 | GPU·h | 成功概率 | 潜在影响 | 优先级 |
|------|-------|---------|---------|--------|
| ACFG-R | ~43 | **60%** | **高**（A-CFG 已验证 CFG 对 DLM 推理有效） | **1** |
| BSD | ~35 | 55% | 中-高（DMI 自然推广，最低风险） | 2 |
| DGD | ~39 | 50% | 高（全新方向，如成功新颖度最高） | 3 |
| **总计** | **~117** | -- | 至少一个成功的概率 ~89% | -- |

4x GPU 约需 30h 总计算时间。

---

## 5. 关键文献引用

### 本轮新发现的直接前驱（必须引用）

1. **LRD** (Zhu et al., 2025, arXiv 2510.11052) — 信念状态精化框架，GSM8K +2.9, MATH500 +3.8, 10.6x 加速。BSD 的灵感来源。
2. **ReMix** (Ye et al., 2026, arXiv 2602.22868) — Continuous Mixing State 框架，2-8x 加速无质量损失。验证连续中间态对 DLM 有效。
3. **A-CFG** (NeurIPS 2025, arXiv 2505.20199) — 自适应 CFG for DLM，GSM8K 73.5 超越 LLaMA3 8B。ACFG-R 的直接基础。
4. **CFG 时间调度改进** (Rojas et al., 2025, arXiv 2507.08965) — 证明高 guidance 早期有害后期有益。ACFG-R 时间调度的理论基础。
5. **EvoToken-DLM** (arXiv 2601.07351) — 连续 token 进化替代硬 mask。BSD 的独立验证。
6. **PRR** (Wan et al., 2026, arXiv 2603.04514) — 跨步信息用于加速精化。验证跨步信号的价值。

### 已知的直接对标

7. **MetaState** (Xia et al., 2026, arXiv 2603.01331) — 需训练的跨步记忆。BSD/ACFG-R 的 training-free 对标。
8. **DMI** (本项目第 3 轮，9.3% vs vanilla 4.7%) — BSD 和 ACFG-R 的出发点。
9. **CORE** (Zhai et al., 2026, arXiv 2602.04096) — 上下文扰动 remasking。ACFG-R 的对标。
10. **TAPS** (Wu et al., 2026, arXiv 2601.22629) — 时序分工发现。DGD 的理论动机。
11. **BACD+TCCF** (Lu et al., 2026, arXiv 2602.09555) — Think Coarse, Critic Fine 块扩散双粒度。DGD 的块扩散对标。
12. **MGDM** (arXiv 2410.14157) — 多粒度扩散推理训练。DGD 的训练版对标。

### 跨领域灵感

13. **CCD** (Tang et al., 2026, arXiv 2602.18232) — AR 对比解码。ACFG-R 的 AR 域先驱。
14. **TCCF "Think Coarse, Critic Fine"** — 粗到细推理范式。DGD 的概念来源。

### 基座模型与推理基准

15. Dream 7B (arXiv 2508.15487) — 主实验模型
16. LLaDA 8B (arXiv 2502.09992) — 验证模型
17. ReMDM (arXiv 2503.00307) — Remasking 基线
