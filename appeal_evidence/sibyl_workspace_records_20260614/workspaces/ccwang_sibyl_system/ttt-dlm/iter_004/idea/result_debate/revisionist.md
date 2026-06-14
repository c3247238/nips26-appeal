# 修正主义者分析（第 4 轮迭代）：从 BSD/RACFG 提案到实验现实——信念更新与方向重评

## 总体立场

本轮迭代标志着项目的第三次重大方向调整。第 3 轮的核心结论——DMI 是唯一有效方法（9.3% vs vanilla 4.7%）——催生了 BSD（Belief-State Diffusion）和 RACFG（Reasoning-Aware CFG）两大新提案。但当前实验数据呈现出一幅比预期更复杂的图景：Full-Scale Countdown 部分结果已出（4/7 方法完成），DMI 的"2x 改善"叙事面临严峻挑战，同时新的正面信号（特别是 DMI 本身在全新实验框架下的表现）尚待完整数据确认。

我的任务是从数据出发，重新审视我们的信念系统——哪些假设已被否定，哪些意外发现需要重新理解，哪些心智模型需要修正。

---

## 1. 假设审计：逐条对照实验证据

### H1：BSD 在 Countdown 上显著优于 DMI（主假设）→ **尚未测试**

- **预期**：BSD 14-18% accuracy（vs DMI 9.3%, vanilla 4.7%）
- **当前状态**：BSD 实验尚未运行。但与 H1 高度相关的 DMI 基线在本轮实验框架下出现了**令人困扰的不一致**：
  - 第 3 轮 Full-Scale：DMI Countdown-500 = **9.3%**（3 seeds, 标杆结果）
  - 本轮 task_3a Pilot (16 样本)：DMI = **0.0%**（alpha=0.3, tau=0.5）
  - 本轮 task_5a 全方法 Pilot (16 样本)：DMI = **0.0%**
- **关键警示**：BSD 的理论基础建立在 DMI 的成功之上——"DMI 的固定比例 embedding 混合有效 → 连续信念演化应该更有效"。但如果 DMI 的效果在新实验框架下不稳健，BSD 的整个论证链将从根基动摇。
- **证据评估**：不足以判断。DMI 在 pilot (N=16) 上的 0% 不能否定它在 Full-Scale 上 9.3% 的结论（第 3 轮已证明 pilot 与 Full-Scale 可以完全翻转），但也提示 DMI 的 alpha/tau 超参数设置可能与实验框架交互。

### H2：BSD 信念向量熵单调递减 → **尚未测试**

- 理论预测合理，但本轮没有运行任何 BSD 实验来检验。
- 值得注意的是，task_8c（第 3 轮）已显示 vanilla 去噪过程本身的预测置信度从 0.969 → 0.995 单调递增（等价于熵递减）。**BSD 需要证明的不仅是熵递减，而是熵递减得比 vanilla 更快/更低——否则 EMA 信念积累没有增量贡献。**

### H3：BSD 的 k 参数存在最优值 → **尚未测试**

- 无实验数据。但此假设的合理性依赖于 BSD Phase 1（连续信念精化）是否产生有意义的信息增益。如果 Phase 1 等价于一个缓慢的恒等映射（信念向量收敛但不比 vanilla 更好），k 的选择将无关紧要。

### H4：RACFG 在 Countdown 上显著优于 vanilla → **尚未直接测试**

- **预期**：≥15% accuracy
- **间接证据**：本轮未运行 RACFG 或 A-CFG 实验，但 RACFG 的理论基础——A-CFG 在 LLaDA-8B GSM8K 上 73.5%——来自文献报告，尚未在 Dream-7B 上复现。
- **风险信号**：提案中标注了 15% 的 "Dream-7B CFG 不兼容"风险。A-CFG 在 LLaDA 上的成功不保证能迁移到 Dream，因为两个模型的架构、训练数据、mask 策略都不同。
- **状态**：悬而未决，但计算代价高（2x FLOPs），如果 Dream-7B 本身不支持 CFG，将浪费大量 GPU 时间。

### H5：跨步稳定性引导优于单步置信度引导 → **尚未测试**

- 无数据。但第 3 轮的 TCR（轨迹一致性重遮蔽）经验提供了一个警示：Dream-7B 的平均轨迹稳定性高达 ~0.96，意味着跨步 JSD 信号可能极弱。如果大多数位置的 JSD ≈ 0，RACFG 的"识别推理关键位置"策略将退化为随机选择。

### H6：CFG 时间调度优于固定权重 → **尚未测试**

- 无数据。但第 3 轮温度退火（anneal_lin_06_02）在 Full-Scale 上不仅无效还恶化了 PPL (+8.1%)，虽然退火温度与 CFG 权重调度是不同的概念，但底层逻辑相似（"早期探索、晚期利用"），这个负面先验不可忽视。

### H7：BSD + RACFG 产生协同效应 → **尚未测试**

- 完全依赖于 H1 和 H4 都成立。在两个前提假设都未验证的情况下，H7 的预测（≥18%）是纯理论推演。

### H8：组合方法在 GSM8K 上泛化 → **部分可参照**

- Vanilla Dream-7B GSM8K ≈ 29.6%（本轮 Full-Scale 数据，与 Dream 论文一致）
- ReMDM-conf GSM8K s42 = 25.1%（350/1319 进度，**劣于 vanilla**）
- 第 3 轮 pilot：ReMDM-conf GSM8K = 37.5% > Vanilla 25.0%（但 N=16）
- **再次出现 Pilot vs Full-Scale 翻转**：ReMDM-conf 在 GSM8K pilot 上最好（37.5%），在 Full-Scale 上初步看比 vanilla 差（25.1% vs 29.6%）。
- **核心启示**：我们对"方法是否泛化"的判断完全受困于不可靠的 pilot 数据。

### 跨迭代对比：DMI Full-Scale 数据的稳健性问题

这是本轮最需要认真对待的问题。第 3 轮 DMI Countdown-500（3 seeds）= 9.3%，是整个项目 20+ 轮迭代中唯一的"明星结果"。但：

| 数据源 | DMI 表现 | Vanilla | DMI vs Vanilla |
|--------|---------|---------|---------------|
| 第 3 轮 Full-Scale (500×3) | 9.3% | 4.7% | **+4.6pp ✓** |
| 第 4 轮 task_3a Pilot (16) | 0.0% | 12.5% | **-12.5pp ✗** |
| 第 4 轮 task_5a Pilot (16) | 0.0% | 12.5% | **-12.5pp ✗** |

**Codex 评审（GPT-5）指出的"基线失真"风险是头号威胁**：Vanilla 4.7% 与 Dream 论文报告的 16.0% 存在 3.4x 差距。如果这个差距来自评估协议差异（prompt 格式、温度、答案抽取规则、stop condition），那么 DMI 的"2x 改善"可能建立在一个失真的基线之上。

---

## 2. 意外发现分析：数据教给我们什么？

### 意外 #1：Full-Scale Countdown 部分结果对 DMI 的 "2x 改善" 提出新的验证

本轮 Full-Scale 数据（4/7 方法完成，3 seeds）：

| 方法 | Mean Accuracy | vs Vanilla |
|------|-------------|-----------|
| Vanilla | 4.7% | — |
| ReMDM-conf | 4.4% | -0.3pp |
| RCR | 5.7% | +1.0pp |
| **DMI** | **9.3%** | **+4.6pp** |
| SCP | ~8.4%* | +3.7pp |
| DTA | pending | — |
| DTA+ReMDM | pending | — |

**DMI 仍然是领跑者**，但 SCP 的中间结果（150/500 样本时 ~8.4%）暗示：
- SCP 可能接近 DMI 的效果水平
- 但 SCP 计算开销 12x，DMI 接近零开销——如果效果相近，DMI 在实用性上碾压 SCP
- **这反而强化了 DMI 的故事**：最简单的方法（embedding 级）在效率-效果 tradeoff 上最优

### 意外 #2：ReMDM-conf 在 GSM8K 上从 pilot 最佳跌落为 Full-Scale 最差

| 基准 | ReMDM Pilot (N=16) | ReMDM Full-Scale |
|------|-------------------|-----------------|
| Countdown | 6.2% (pilot), 25% (旧 pilot) | 4.4% (完成) |
| GSM8K | 37.5% (pilot) | 25.1% (进行中) |

ReMDM-conf 在 GSM8K 上的 pilot → Full-Scale 下降（37.5% → 25.1%，劣于 vanilla 29.6%）完美复现了第 3 轮 PPL 指标上的 "winner's curse" 模式。**这是第四个独立的 Pilot-Full-Scale 翻转案例**（前三个：entropy PPL、TCR PPL、anneal PPL），构成了项目最稳健的元发现。

### 意外 #3：DTA 在 MBPP pilot 上的正向信号是唯一的亮点

- DTA MBPP pilot: 37.5% vs Vanilla 25.0% (+12.5pp)
- DTA Countdown pilot: 6.2% vs Vanilla 12.5% (-6.3pp)
- DTA GSM8K pilot: 12.5% vs Vanilla 25.0% (-12.5pp)
- DTA LLaDA Countdown: 0.0% vs Vanilla 12.5% (-12.5pp)
- DTA LLaDA GSM8K: 18.8% vs Vanilla 43.8% (-25.0pp)

**MBPP 是唯一正向的**。5 个实验中 4 个负向，1 个正向——且正向那个是 N=16 pilot。统计上，这更可能是噪声而非信号。但如果验证为真，将指向一个重要的任务依赖性假说：DTA 的 LoRA 在线更新可能对**有强局部结构的任务**（代码）有效，对需要**全局推理链的任务**（数学、算术）无效。

### 意外 #4：Vanilla 基线在不同实验间的不一致

| 实验 | Vanilla Accuracy |
|------|-----------------|
| 第 3 轮 Full-Scale Countdown-500 | 4.7% |
| 第 4 轮 task_5a Pilot (16) | 12.5% |
| 第 4 轮 Full-Scale Countdown (in progress) | 4.7% |
| task_1a Pilot (16) | 18.75% |
| task_2a 对比 (16) | 12.5% |
| GSM8K Full-Scale | 29.6% |
| Dream 论文报告 Countdown | 16.0% |

Vanilla 在 Countdown 16 样本 pilot 上的范围是 **4.7% 到 18.75%**——变异幅度超过 4 倍。这直接证明 16 样本 pilot 对任何 DLM 方法评估都没有筛选能力。但 Vanilla Full-Scale 4.7% 与 Dream 论文 16.0% 的差距需要解释——Codex 评审正确指出这是"头号风险"。

---

## 3. 心智模型更新：我们哪里理解错了？

### 错误认知 #1："DMI 的成功证明了表示空间干预的方向正确"——这个推论过早

第 3 轮的核心叙事是："DMI 的成功 + 文献验证（LRD、ReMix）→ 连续表示是正确方向 → BSD 应该更好"。但这个推理链的每一步都有问题：

1. **DMI 的成功机制不明**。Codex 评审列出了三个强替代解释：(a) 隐式温度/熵调节——soft embedding 等价于降低采样温度；(b) 格式/约束改善而非推理改善——Countdown 的准确率受"表达式合法性"影响；(c) 额外跨步信息传递——但不是"学习"意义上的。**在没有对照实验（随机 embedding 扰动、entropy-matched temperature baseline）拆解机制之前，从 DMI 跳跃到 BSD 缺乏因果依据。**

2. **LRD/ReMix 是加速方法，不是质量提升方法**。提案中反复引用 LRD（GSM8K +2.9）和 ReMix（2-8x 加速无损）作为"连续表示方向验证"，但这两个工作的核心目标是加速去噪而非提升推理能力。它们的成功不能直接推导出"表示空间干预能提升推理准确率"。

3. **BSD 的 OOD 风险被低估**。BSD 提案中 Phase 1 完全用信念向量替代 mask embedding——这意味着 Transformer 的输入空间发生了根本性变化。即使做了 L2 归一化，embedding 的方向分布与预训练分布可能完全不同。提案中估计 OOD 概率为 30%，但鉴于 DTA 的 LoRA delta norm ~1e-5 就已经导致性能下降，我认为 **BSD 的 OOD 概率应上调至 50-60%**。

### 错误认知 #2："A-CFG 在 LLaDA 上 73.5 → RACFG 在 Dream 上也应有效"——架构差异未被充分考虑

A-CFG 的变革性结果来自 LLaDA-8B，一个与 Dream-7B 不同的架构。关键差异：
- LLaDA 使用 Llama-3 backbone + 额外的 mask modeling head
- Dream 使用不同的预训练策略和 mask 处理
- A-CFG 的"低置信度 token 重新 mask → 构建无条件输入 → CFG"逻辑依赖于模型的 mask 处理方式

**我们连 A-CFG 在 Dream-7B 上是否能复现都没有验证，就开始设计 RACFG（A-CFG 的增强版）。** 这违反了实验科学的基本原则——在增强一个方法之前，先确认基础方法可行。

### 错误认知 #3："Pilot 不可靠"但我们仍在依赖 pilot 做决策

项目已积累了至少 4 个 Pilot-Full-Scale 翻转案例，但提案仍然设计了大量 pilot 阶段（Phase 1: 独立 pilot，每个实验仅 16-100 样本）。更根本的问题是：**如果 pilot 不可靠，那么 BSD/RACFG 的 Phase 1 pilot 结论也将不可靠**。

- 建议的改变：取消 pilot 阶段，直接在 Countdown-500 × 3 seeds 上运行，哪怕只测试 1-2 个方法。
- 或者，如果必须做 pilot，将样本量提升到 ≥100，且至少 2 seeds。
- **16 样本 pilot 应该从项目方法论中永久移除。**

### 错误认知 #4：权重分配中对反对者的系统性低估——再次重演

第 3 轮提案给反对者 5% 权重，后来承认应该给 20-25%。第 4 轮提案给反对者 20% 权重——进步了，但反对者的两个核心论点仍未被充分吸收：

1. **"推理时计算扩展存在逆向效应"**——DTA 在多数 benchmark 上**损害**准确率，不仅是"无效"。SCP 以 12x 开销换来与 DMI 接近的效果。这些数据支持"逆向效应"而非"零效应"。
2. **"Benchmark 驱动评估而非 PPL"**——第 4 轮确实切换到了 benchmark，这是正确的。但 vanilla 基线与论文报告的差距（4.7% vs 16.0%）暗示 benchmark 评估本身的 protocol 可能有问题。

---

## 4. 重新构框提案

### 原始构框（第 4 轮提案）
"DMI 成功 + 文献验证 → 连续表示是正确方向 → BSD（表示层连续信念）+ RACFG（预测层 CFG 引导）双层优化 → 大幅提升推理准确率"

### 数据支持的替代构框

**构框 A：DMI 机制拆解优先**

在构建 BSD 之前，必须先理解 DMI 为什么有效。核心实验：
1. 随机 embedding 扰动（等幅度但不来自 logits）→ 如果也有效，DMI 的价值在于"扰动"而非"信息"
2. Temperature-matched baseline（降低温度使 entropy 匹配 DMI）→ 如果也有效，DMI 等价于温度调节
3. 格式分析（只看"表达式合法"的子集中的准确率）→ 排除"修复格式"的混淆因素

**如果 DMI 的改善主要来自隐式温度调节，BSD 的整个连续信念叙事就失去了基础。**

**构框 B：计算公平性为核心叙事**

现有数据的最清晰叙事不是"方法 X 有效"，而是"在等计算预算下，不同干预策略的效率差异"：

| 方法 | Countdown Acc | 相对开销 | 效率 (acc/compute) |
|------|-------------|---------|-------------------|
| Vanilla | 4.7% | 1.0x | 4.7 |
| DMI | 9.3% | ~1.1x | 8.5 |
| RCR | 5.7% | ~1.5x | 3.8 |
| SCP | ~8.4% | ~12x | 0.7 |
| DTA | pending | ~4x | pending |

DMI 的效率 (8.5) 是所有方法中最高的，SCP (0.7) 最低。**论文可以定位为"推理时增强的效率分析"，而非"新方法论文"。**

**构框 C：方法论贡献——DLM 评估的不可靠性**

项目最稳健的发现不是任何具体方法的有效性，而是：
1. **16 样本 pilot 对 DLM 没有筛选能力**（4+ 个翻转案例）
2. **PPL 在 DLM 上不可靠**（已在第 3 轮系统证明）
3. **Vanilla 基线本身可能不稳定**（4.7% vs 论文 16.0%）

这些元层面的发现对 DLM 社区的价值可能大于任何具体方法的增量改善。

---

## 5. 新假设

### NH1：DMI 的改善主要来自隐式 softmax 温度调节
**Statement**: 将 DMI 的 soft embedding 替换为 temperature-matched vanilla（在采样温度上等效）后，accuracy 差异 <2pp。
**Test**: Temperature sweep (τ = 0.2, 0.3, 0.35) on Countdown-500 vs DMI (alpha=0.3, tau=0.5)。
**Rationale**: DMI 的 soft embedding 本质上是一个低熵的 token 分布，与降低采样温度的效果可能重叠。

### NH2：BSD 的 OOD 问题将主导其性能
**Statement**: BSD Phase 1 的全信念向量替代方案（不与 mask_emb 混合）导致 Countdown accuracy < vanilla。
**Test**: BSD pilot (100 samples, 2 seeds)，对比全替代 vs DMI-style 混合。
**Rationale**: Dream-7B 对输入 embedding 空间的偏移极其敏感（DTA delta norm ~1e-5 就导致性能下降）。

### NH3：A-CFG 在 Dream-7B 上不产生与 LLaDA 同量级的改善
**Statement**: A-CFG 在 Dream-7B Countdown-500 上的 accuracy < vanilla + 3pp（远低于 LLaDA 上的变革性结果）。
**Test**: 先复现 A-CFG on Dream-7B，再判断是否值得开发 RACFG。
**Rationale**: Dream 和 LLaDA 的 mask 处理机制不同；Dream 论文未报告 CFG 结果，可能有未公开的兼容性问题。

### NH4：Correction Precision 是所有 remasking 方法效果的天花板
**Statement**: 对于任何 remasking 策略，如果 Correction Precision < 50%，则该策略在 accuracy 上不优于 vanilla。
**Test**: 汇总所有方法的 Correction Precision（已有 ReMDM 31.3%, SCP 76.9%）并与 accuracy 改善做回归。
**Rationale**: 如果修正操作有 >50% 的概率作用在"正确 token"上，净效果为负。

### NH5：Full-Scale 验证需要最小 N=200 × 2 seeds 才能检测 3pp 以上的效应
**Statement**: 对于 Dream-7B Countdown 的 baseline ~5% accuracy，检测 3pp 效应（5% → 8%）在 80% power 下需要 N ≥ 200 per seed。
**Test**: Post-hoc power analysis on existing Full-Scale data。
**Rationale**: 项目反复出现 pilot-full-scale 翻转，核心原因是统计功效不足。量化最小样本量是未来实验设计的基础设施。

---

## 6. Pivot vs Iterate 建议

### 推荐：**暂停 BSD/RACFG 开发，优先完成当前 Full-Scale 数据收集 + DMI 机制拆解**

**理由**：

1. **当前 Full-Scale 实验尚未完成**（DTA、DTA+ReMDM、SCP 在 Countdown 上待完成；GSM8K 大部分方法待完成）。在未收齐数据之前切换到新方法是资源浪费——第 3 轮已经付出了这个代价（PPL 实验做了一半就切换到 accuracy），第 4 轮不应重蹈覆辙。

2. **DMI 的因果机制不明**。BSD 整个方案建立在"DMI 证明了连续表示方向"之上，但 Codex 评审正确指出存在 ≥3 个替代解释。**不做机制拆解就推进 BSD，是在沙滩上建楼。**

3. **RACFG 依赖的 A-CFG 未在 Dream-7B 上验证**。提案中设计了"Decision Gate 1: If A-CFG fails on Dream-7B, switch to LLaDA-8B"——这个门控测试应该**最先做**，而不是在 Phase 1 中。如果 A-CFG 在 Dream-7B 上失败，RACFG 的全部 ~20 GPU·h 预算将需要重新分配。

4. **基线审计未完成**。Vanilla 4.7% vs Dream 论文 16.0% 的差距是整篇论文可信度的基础。如果评审问"为什么你的 baseline 与原始论文差 3.4x"而我们没有答案，论文将被 desk reject。

### 具体优先级

| 优先级 | 行动 | 目的 | 预计时间 |
|--------|------|------|---------|
| **P0** | 完成当前 Full-Scale 实验（DTA/SCP/DTA+ReMDM 在 Countdown + 所有方法 GSM8K） | 收齐第 4 轮数据 | 2-3 天（GPU 时间） |
| **P0** | Baseline audit：对齐 Dream 论文的评估 protocol | 解释 4.7% vs 16.0% 差距 | 1 天 |
| **P1** | DMI 机制拆解（3 个对照实验） | 确认 BSD 的理论基础是否成立 | 2 天 |
| **P1** | A-CFG 在 Dream-7B 上的复现 | 确认 RACFG 的前提是否成立 | 1 天 |
| **P2** | BSD pilot（如果 DMI 机制验证通过） | 测试核心假设 H1 | 2 天 |
| **P2** | RACFG pilot（如果 A-CFG 复现成功） | 测试核心假设 H4 | 2 天 |

### 不推荐的行动

1. **不推荐立即实现 BSD 完整算法**。在 DMI 机制未拆解、A-CFG 未验证的情况下，编写 BSD 代码是过早优化。
2. **不推荐继续使用 16 样本 pilot**。项目已有充分证据证明 N=16 pilot 对 DLM 没有筛选能力。
3. **不推荐在 PPL 指标上做任何新实验**。PPL 的不可靠性已是项目共识。

### 论文方向建议

如果 Full-Scale 数据确认 DMI 在 Countdown 上仍然有效（9.3%），且 mechanism 拆解显示改善来自跨步信息传递（而非隐式温度调节），那么论文的最强方向是：

**"Shallow is All You Need: Why Embedding-Level Intervention Outperforms Parameter-Level Adaptation in Masked Diffusion Language Models"**

核心贡献：
1. 完整的信息增强谱系消融（DMI > SCP >> DTA/ReMDM/RCR）
2. DMI 机制拆解（因果分析，排除替代解释）
3. 方法论贡献：16-sample pilot 不可靠性的系统性证据 + DLM 评估 protocol 建议
4. DMI 作为零开销推理时增强的实用方法

如果 DMI 机制拆解显示改善主要来自温度调节或格式修复，则论文方向应调整为：

**"The Limits of Training-Free Inference-Time Scaling for Masked Diffusion Language Models: A Multi-Scale Diagnostic Study"**

核心贡献：负面结果系统化 + 评估方法论 + 理论分析（为什么 training-free 方法的天花板极低）。

---

## 附录：证据可信度评级（第 4 轮数据）

| 证据来源 | 样本量 | 可信度 | 备注 |
|---------|--------|--------|------|
| Full-Scale Countdown 4/7 方法 (第 4 轮) | 500 × 3 seeds | **高** | DMI/RCR/ReMDM/vanilla 已完成 |
| Full-Scale GSM8K vanilla | 1300/1319 | **高** | 接近完成 |
| Full-Scale GSM8K ReMDM | 350/1319 | **中** | 进行中 |
| SCP interim Countdown | 150/500 | **中** | 临时数据 |
| 所有 N=16 pilot 数据 | 16 × 1 seed | **极低** | 已有 4 个翻转案例证明不可靠 |
| 第 3 轮 Full-Scale PPL 数据 | 101 × 3 seeds | **中** | PPL 指标本身有争议 |
| DMI Countdown 9.3% (第 3 轮) | 500 × 3 seeds | **高（数据层面）/ 中（解释层面）** | 数据可靠但因果机制不明 |
| Codex 评审 4.8/10 | — | **高（参考价值）** | 外部独立评估 |
