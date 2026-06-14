# 战略分析：BSD + RACFG 实验全面评估与下一步方向

## 1. 基于当前结果的最有前途方向

### 1.1 实验事实全景

第 4 轮迭代完成了 BSD、RACFG、A-CFG 的完整 pilot 评估（n=16）和部分消融实验（n=100），并在 Countdown-500 三 seed 全规模上完成了先前方法的对比。全部数据汇总如下：

**Countdown-500 全规模（3 seeds, 已完成）**：

| 方法 | 均值准确率 | FLOPs | 状态 |
|------|-----------|-------|------|
| Vanilla | 4.7% | 1.0x | ✓ |
| **DMI** | **9.3%** | **~1.05x** | ✓ 最佳性价比 |
| SCP | 9.1% | ~1.5x | ✓ |
| DTA | 4.8% | ~2.5x | ✓ 无效 |
| RCR | 5.7% | ~2.5x | ✓ |
| ReMDM-conf | 4.4% | ~1.5x | ✓ |
| DTA+ReMDM | 3.6% | ~3x | ✓ 负面 |

**BSD/RACFG Pilot（n=16，Countdown）**：

| 方法 | Pilot 准确率 | FLOPs | 状态 |
|------|-------------|-------|------|
| BSD (k=0.75) | 6.2% | ~1.1x | 最佳 BSD 配置 |
| A-CFG (w=1.5) | 12.5% | ~2.0x | 最佳引导方法 |
| BSD+A-CFG combo | 6.2% | ~2.1x | 组合无协同 |
| RACFG (JSD) | 0.0% | ~1.8x | 完全失败 |

**GSM8K Pilot（n=16）**：

| 方法 | Pilot 准确率 | FLOPs |
|------|-------------|-------|
| Vanilla | 25.0% | 1.0x |
| DMI | 25.0% | ~1.05x |
| BSD | 18.8% | ~1.1x |
| **A-CFG (w=1.5)** | **37.5%** | ~2.0x |

### 1.2 战略赢家：A-CFG

A-CFG 是本轮的**关键发现**。在 Countdown pilot 上 12.5%（与 DMI 并列第一），在 GSM8K pilot 上 37.5%（绝对领先）。尽管 n=16 的统计功效有限，但 A-CFG 在 GSM8K 上的 +12.5pp 提升方向性强烈，且有 LLaDA-8B 上的文献验证（GSM8K 73.5）做独立背书。

**A-CFG 的战略价值**：
- 文献级背书：A-CFG 是 NeurIPS 2025 接收论文，Dream-7B 上的复现本身有价值
- GSM8K 效果显著：37.5% vs vanilla 25.0%，如果全规模验证成立，这是论文的核心卖点
- 计算开销合理：2x FLOPs（1 次额外前向传播/步），远低于 DTA 的 2.5x + 反向传播

### 1.3 RACFG 的彻底失败与教训

**核心假设 H5 被证伪**：跨步 JSD 稳定性信号在 Dream-7B 上完全无效。根本原因是 Dream-7B 的跨步概率分布极其稳定（stability ~0.997），无法提供有意义的"不稳定"信号。

这是一个**模型架构层面的限制**，不是超参数问题——所有 remask_pct（5%/10%/20%）和 ema_lambda（0.3/0.7）的组合均产出 0% 准确率。

**教训**：提案中"跨步稳定性→推理关键位置"的直觉在理论上合理，但忽视了 Dream-7B 去噪过程的一个关键特性——去噪步间的概率分布变化极小。这意味着 RACFG 的差异化叙事（vs A-CFG）不成立，必须从论文叙事中移除。

### 1.4 BSD 的表现评估

BSD pilot 准确率 6.2%（1/16），低于 DMI 的 12.5%（2/16）。BSD 的信念熵确实单调递减（H2 得到支持，15/16 个样本 rho < -0.8），但这种信息论上的优雅并未转化为准确率优势。

**BSD 的定位调整**：
- 作为独立方法：低于 DMI，不足以成为论文核心
- 信息论贡献：熵分析有价值，但需要全规模数据验证
- 与 A-CFG 的组合失败意味着 BSD 不能作为"协同"叙事的一半

### 1.5 计算公平比较的残酷事实

compute-fair 实验显示，在 n=16 pilot 上，**没有任何方法在 Pareto 前沿上超越 vanilla（等量 FLOPs 下的更多去噪步）**。这是一个严重的战略风险——如果全规模验证仍然如此，所有方法的价值主张将被"直接加步"所瓦解。

但需注意：这是 n=16 的结果，统计功效极低。DMI 在 Countdown-500 上的 9.3% vs vanilla 4.7% 已经在全规模上证明了跨步信息传递的价值。关键问题是 **A-CFG 在全规模上是否也能超越等 FLOPs 的 vanilla**。

## 2. 最有前途的方向（优先排序）

### 方向 1（最高优先）：A-CFG 全规模验证 + Dream-7B 第一报告

**理由**：
- A-CFG 是文献验证的方法，Dream-7B 上尚无公开结果
- GSM8K pilot +12.5pp 是所有方法中最强的信号
- 全规模验证（500×3 seeds Countdown + 1319 GSM8K）是论文是否能升级到顶会水平的分水岭
- 如果 A-CFG Countdown-500 ≥ 12% 或 GSM8K-1319 ≥ 33%，论文定位可升至 NeurIPS/ICLR

**所需资源**：~15 GPU·h（Countdown 3 seeds × 2x FLOPs + GSM8K 3 seeds × 2x FLOPs）

### 方向 2：DMI + A-CFG 正交组合

**理由**：
- DMI 和 A-CFG 在不同层面操作：DMI 改善表示（embedding 注入），A-CFG 改善预测（CFG 引导）
- BSD + A-CFG 组合失败可能是 BSD 本身太弱，而非组合不可行
- DMI 已全规模验证（9.3%），A-CFG 是 pilot 最佳——两者组合是自然选项
- DMI 仅 1.05x FLOPs，与 A-CFG 的 2x FLOPs 叠加后仍仅 ~2.1x

**具体方案**：在 A-CFG 的每步去噪中，同时将 mask embedding 替换为 DMI 的 logit 加权混合。即：
```
input_emb[mask_pos] = α · mask_emb + (1-α) · Σ softmax(ℓ^{t-1}) · e_v
# 然后在此基础上执行 A-CFG 的 CFG 引导
```

**所需资源**：~10 GPU·h（pilot 验证 1h + 全规模 9h）

### 方向 3：DMI 超参数优化 + 变体探索

**理由**：
- DMI 是唯一经全规模验证有效的方法（9.3%），但 α=0.3 是未经系统优化的值
- α 的最优值可能在不同位置类型（数字/运算符/文本）上不同 → 位置自适应 DMI
- 温度调节的 DMI（类似 BSD 的 τ_t 但更轻量）可能提升效果
- 即使 A-CFG 全规模失败，优化后的 DMI 仍是论文核心

**所需资源**：~8 GPU·h

### 方向 4：MBPP 代码生成评估

**理由**：
- 第三个 benchmark（Countdown + GSM8K + MBPP）显著增强论文可信度
- MBPP pilot 25% baseline，且 DTA 在 MBPP 上是唯一有正信号的 benchmark（+12.5pp pilot）
- A-CFG 在代码生成上的效果是开放问题——如果有效，叙事从"数学推理"扩展到"通用推理"

**所需资源**：~12 GPU·h

## 3. 资源分配建议

### 3.1 GPU 时间分配（总计 ~45 GPU·h, 4 GPU × ~12h 墙钟）

| 优先级 | 任务 | GPU·h | 预期产出 |
|--------|------|-------|---------|
| **P0** | A-CFG Countdown-500 ×3 seeds | 8 | 核心数据点 |
| **P0** | A-CFG GSM8K-1319 ×3 seeds | 10 | 跨任务泛化 |
| **P1** | DMI+A-CFG 组合 pilot (n=100) | 2 | 组合可行性 |
| **P1** | DMI α 优化 (n=100) | 3 | 优化核心方法 |
| **P2** | DMI+A-CFG 全规模 (if pilot GO) | 10 | 组合方法 |
| **P2** | MBPP baseline + A-CFG + DMI | 12 | 第三 benchmark |

### 3.2 人力/Token 时间分配

| 优先级 | 任务 | 时间 | 条件 |
|--------|------|------|------|
| **P0** | Countdown-500 + GSM8K 全规模统计分析 | 2h | A-CFG 全规模数据到位后 |
| **P0** | 论文定位确定 + 大纲修订 | 2h | 统计分析完成后 |
| **P1** | 写作启动（方法/实验章节） | 4h | 论文定位确定后 |
| **P2** | 图表准备（Pareto 曲线、熵轨迹、消融热力图） | 3h | 实验数据全部到位后 |

### 3.3 关键里程碑

- **T+6h**: A-CFG Countdown-500 seed 42 完成 → 初步判断 A-CFG 全规模效果
- **T+12h**: A-CFG Countdown-500 三 seed 完成 → **论文核心定位锁定**
- **T+18h**: A-CFG GSM8K-1319 seed 42 完成 → 跨任务信号
- **T+24h**: DMI+A-CFG 组合 pilot → 决定是否开发组合方法

## 4. 风险评估

### 4.1 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| A-CFG Countdown-500 无显著提升（<7%）| 30% | 高 | 转向 DMI 核心论文 + A-CFG 仅在 GSM8K 上报告 |
| A-CFG GSM8K 全规模 < vanilla + 3pp | 25% | 高 | Countdown+DMI 为核心，GSM8K 放附录 |
| compute-fair 比较中 vanilla 加步始终更优 | 20% | 极高 | 强调 DMI 的零开销特性（1.05x FLOPs 内无法加步） |
| DMI+A-CFG 组合无协同（类似 BSD+A-CFG）| 40% | 中 | 分别报告，不强调组合 |
| MetaState 或 A-CFG 后续论文抢发 | 15% | 高 | 加速写作，强调 Dream-7B 独特数据点 |
| 全规模统计检验无一显著 | 15% | 高 | 增大样本量到 1000；使用 Bootstrap CI + 效果量 |

### 4.2 下行保护分析

即使最坏情况（A-CFG 全规模也失败），论文仍有以下资产：

1. **DMI 9.3% vs vanilla 4.7%**（全规模三 seed 验证）
2. **7 种方法的系统性对比**（vanilla, DMI, SCP, DTA, RCR, ReMDM-conf, DTA+ReMDM）
3. **DTA 失败 + RACFG 失败的诊断分析**（参数级更新无效 + JSD 在 Dream-7B 上无信息量）
4. **信息传递谱系的逆向发现**（浅层干预优于深层干预）
5. **BSD 的信念熵理论分析**（即使准确率不优，信息论贡献仍有价值）

论文可发表概率仍 >85%（EMNLP 及以上）。

## 5. Pivot 还是 Proceed？

### 判决：**PROCEED**（附带叙事重构）

**理由**：

1. **A-CFG 全规模验证是必须完成的**。GSM8K pilot +12.5pp 的信号太强，不验证就放弃是战略失误。计算成本仅 ~18 GPU·h（4 GPU 下 ~5h 墙钟），完全可负担。

2. **DMI 的安全网依然坚固**。无论 A-CFG 全规模结果如何，DMI 9.3%（全规模三 seed）仍是可靠的论文核心。

3. **叙事需要从"BSD+RACFG 三层架构"重构为"A-CFG + DMI 实证发现"**。原提案的 BSD+RACFG 组合叙事已被数据否定（RACFG 完全失败，BSD+A-CFG 无协同）。新叙事应围绕：
   - A-CFG 在 Dream-7B 上的首次验证
   - DMI 的零开销 2x 提升发现
   - RACFG 失败的诊断分析（Dream-7B 去噪稳定性的特性）
   - 可选：DMI+A-CFG 组合

### Pivot 触发条件

以下情况触发 Pivot：

1. **A-CFG Countdown-500 seed 42 准确率 < 6%** → 计算下一步前先诊断；如果是实现问题则修复后重跑，如果是方法限制则转向备选 3（实证系统研究论文）
2. **DMI+A-CFG 组合 pilot 导致文本退化（rep-3 > vanilla + 50%）** → 放弃组合方向
3. **全规模统计分析中 DMI vs vanilla 的 McNemar 也不显著（p > 0.1）** → 紧急审视 1500 个样本的数据质量

## 6. 论文策略的情景规划（更新版）

### 场景 A：A-CFG Countdown ≥ 12% + GSM8K ≥ 33%（概率 ~35%）

**定位**：NeurIPS/ICLR Main
**标题**：*"Adaptive Guidance Meets Diffusion Memory: Training-Free Inference-Time Scaling for Masked Diffusion Language Models"*
**叙事**：
- A-CFG 在 Dream-7B 上的首次成功验证（Countdown + GSM8K + 可能的 MBPP）
- DMI 的零开销发现作为互补方法
- 7 种方法的系统性对比 + 计算量公平分析
- RACFG 失败的诊断作为理论贡献（DLM 去噪稳定性的新见解）

### 场景 B：A-CFG 仅在 GSM8K 有效，Countdown 无效（概率 ~25%）

**定位**：ICLR/NeurIPS
**标题**：*"The Task-Dependent Nature of Inference-Time Scaling in Masked Diffusion Language Models"*
**叙事**：
- A-CFG 在 GSM8K 有效但 Countdown 无效 → 推理增强方法的任务依赖性
- DMI 在 Countdown 有效但 GSM8K 无效 → 不同任务需要不同的信息增强策略
- 这是一个有价值的 insight：不存在 universal 的 DLM 推理增强方法

### 场景 C：A-CFG 全规模也失败（概率 ~25%）

**定位**：NeurIPS/EMNLP
**标题**：*"Diffusion Memory Injection: Simple Cross-Step Information Transfer Doubles Reasoning in Masked Diffusion Models"*
**叙事**：
- DMI 9.3% vs vanilla 4.7%（~2x 改善，零开销）为核心贡献
- 10+ 种方法的系统性负面结果分析
- "浅层干预优于深层干预"的反直觉发现
- DTA、RACFG、BSD 失败的诊断框架

### 场景 D：如果 DMI+A-CFG 组合有效（概率 ~30%）

**定位**：NeurIPS/ICML Main
**标题**：*"Belief Injection Meets Classifier-Free Guidance: Orthogonal Axes for Masked Diffusion Inference Scaling"*
**叙事**：最强叙事——两个正交方向的组合产生协同效应
**风险**：此场景取决于 P1 的 pilot 验证

## 7. 总结与即时行动清单

### 核心判断

项目经历了重大叙事转折：提案的 BSD+RACFG 三层架构未得到实验支持（RACFG 完全失败，BSD+A-CFG 无协同），但 **A-CFG 的 GSM8K 效果 + DMI 的 Countdown 效果构成了一个更务实、更可信的双核心论文**。

当前处于**信息不完整但方向明确**的状态：A-CFG 全规模数据将在 ~12h 内到位，届时论文定位可完全锁定。

### 即时行动清单

1. **立即启动**：A-CFG (w=1.5, fixed) Countdown-500 × 3 seeds（GPU 0-2）
2. **立即启动**：A-CFG (w=1.5, fixed) GSM8K-1319 seed 42（GPU 3）
3. **6h 内**：根据 A-CFG Countdown seed 42 结果确认全规模方向
4. **12h 内**：DMI+A-CFG 组合 pilot（n=100，Countdown）
5. **18h 内**：全部 Countdown + GSM8K 全规模统计分析
6. **24h 内**：论文大纲定稿，开始方法/实验章节写作

### 信心评估

- 论文可发表概率：**88%**（DMI 保底 + 系统性实验数据丰富）
- 顶会水平概率：**45%**（需要 A-CFG 全规模成功 + 跨任务泛化）
- 需要 Pivot 概率：**<10%**（仅在 DMI 统计不显著 + A-CFG 全面失败时）
- 预计论文定位锁定时间：**12h**

### 假设验证状态汇总

| 假设 | 状态 | 备注 |
|------|------|------|
| H1 (BSD > DMI) | **待验证**（pilot 方向负面）| BSD 6.2% < DMI 12.5%（pilot） |
| H2 (BSD 熵递减) | **支持** | 15/16 样本 rho < -0.8 |
| H3 (k 最优值) | **不支持** | k=0.75 最佳，非中间值 |
| H4 (RACFG > vanilla) | **转向 A-CFG 验证** | RACFG 0%，A-CFG 12.5%（pilot） |
| H5 (JSD > 置信度) | **证伪** | A-CFG 12.5% >> RACFG 0% |
| H6 (时间调度 > 固定) | **证伪** | 所有调度方案 ≤ fixed |
| H7 (BSD+RACFG 协同) | **证伪** | combo 6.2% < max(BSD, A-CFG) 12.5% |
| H8 (GSM8K 泛化) | **A-CFG 方向积极** | A-CFG 37.5% vs vanilla 25.0%（pilot） |
