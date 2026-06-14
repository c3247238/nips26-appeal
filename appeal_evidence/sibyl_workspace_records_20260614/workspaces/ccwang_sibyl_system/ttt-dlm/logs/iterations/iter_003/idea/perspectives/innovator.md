# 创新者提案：遮蔽扩散语言模型的推理时计算扩展（第 3 轮迭代）

**日期**: 2026-03-09
**视角**: 大胆创新、跨领域迁移、反直觉思路

---

## 0. 前情诊断：实验数据揭示的真正信号

经过 18 轮前序迭代 + 当前迭代的大量实验，我们有了前所未有的清晰画面：

**已确认的死路**：
- TTT 6 个变体在 AR 框架下统计不显著 (p=0.88)
- Best-of-N 3x 计算量反而 PPL +6.9%
- Pure remasking (ReMDM-conf, RCR) 在 Countdown-500 上无改善 (4.4%, 5.7% vs vanilla 4.7%)
- DTA (LoRA 在线更新) pilot 上 6.2% < vanilla 12.5%，方向不确定
- SCP (Self-Contradiction Probing) 计算开销 12x 但无准确率提升

**唯一的正面信号——DMI**：
- **Countdown-500 三 seed 平均 9.3% vs vanilla 4.7%**——近 2x 改善！
- 这是整个项目历史上第一个在 full-scale、多 seed 验证下具有实质改善的方法
- DMI 的开销接近零（仅缓存上一步 logits + softmax @ embedding 加权平均）
- 这意味着**跨步软信息确实对 DLM 有价值**，但必须以正确的方式传递

**我的核心诊断**：DMI 的成功不是偶然——它揭示了一个被忽视的信号通道。标准 DLM 在 remasking 后丢弃所有连续信息，而 DMI 证明了哪怕最简单的 logit carry-over 也能提供有意义的信号。但 DTA 的失败说明，单纯的参数更新（梯度信号）可能不是利用这个信号的最优方式。**真正的机会在于找到比 DMI 更强但比 DTA 更稳定的信息传递机制**。

---

## 1. 角度一：Adaptive Soft Diffusion（ASD）—— 让 DMI 从固定混合进化为动态调制（改进现有方法）

### 1.1 核心洞察

DMI 的成功证明了跨步软信息的价值，但当前 DMI 有两个根本限制：
1. **固定混合比** (alpha=0.3)：所有位置、所有步骤使用相同的 alpha，无法适应不同 token 的不确定性
2. **仅看上一步**：只传递 t-1 步的 logits，丢弃更早步骤的信息

TC-LoRA (arXiv 2510.09561) 在连续扩散模型中提出了一个关键思想：**用超网络根据扩散时间步和条件动态生成适配器权重**。我们可以借鉴这一思路，但不需要训练——直接用模型自身的 entropy 信号来动态调制 DMI 的混合强度。

### 1.2 提案：Adaptive Soft Diffusion (ASD)

**核心 idea**：将 DMI 从固定 alpha 混合升级为 **token-level、step-level 双重自适应** 的软信息注入。

**具体方案**：

1. **Token-Level 自适应 alpha**：
   $$\alpha_i^t = \sigma\left(\frac{H(\ell_{t-1}^i) - \bar{H}^t}{\tau}\right) \cdot \alpha_{\max}$$
   其中 $H(\ell_{t-1}^i)$ 是位置 i 上一步 logits 的熵，$\bar{H}^t$ 是所有 mask 位置的平均熵。
   - 高熵位置（模型不确定）：更多依赖上一步的软信息（alpha 大）
   - 低熵位置（模型确信）：更多依赖当前 mask embedding（alpha 小）

2. **Step-Level 退火**：alpha_max 随去噪进度衰减：
   $$\alpha_{\max}^t = \alpha_0 \cdot \left(1 - \frac{t}{T}\right)^\beta$$
   早期步骤（大量 mask）依赖更多软信息引导；后期步骤（大部分已揭示）alpha 趋近零，模型依赖自身能力

3. **Exponential Moving Average 记忆**：不只看上一步，维护 logits 的 EMA：
   $$\bar{\ell}^t = \lambda \cdot \bar{\ell}^{t-1} + (1-\lambda) \cdot \ell^t$$
   这使得 ASD 携带多步历史信息，而非仅 1 步

4. **与 remasking 策略的整合**：ASD 的 token-level alpha 可以同时作为 remasking 的指导信号——高 alpha 位置（高不确定性）是 remask 的候选目标，实现"信息注入 + 选择性修正"的双重效果

### 1.3 与 MetaState 的关键区别

MetaState (arXiv 2603.01331) 用 cross-attention Mixer + GRU Updater + cross-attention Injector 实现跨步记忆，但需要 K-step unrolling 训练。ASD 是**完全 training-free** 的，所有信号来自模型自身的 logit 分布，零额外参数。

| 维度 | MetaState | ASD |
|------|-----------|-----|
| 记忆载体 | 训练的 memory slots | Logit EMA（零参数）|
| 更新方式 | GRU 门控（需训练）| Exponential moving average（无训练）|
| 注入方式 | Cross-attention Injector（需训练）| Entropy-gated embedding 混合 |
| 自适应性 | 端到端学习 | Token entropy 驱动的动态 alpha |
| 训练需求 | K-step unrolling 微调 | **零训练** |

### 1.4 假设

**H1**：ASD 在 Dream-7B Countdown-500 上的准确率将显著优于固定 alpha DMI（预期 12-15% vs DMI 的 9.3%），因为自适应 alpha 将信息集中注入最需要的位置。

**H2**：ASD + ReMDM-conf 组合在 Countdown-500 上的准确率将优于单独的 ASD 或 ReMDM-conf，因为 ASD 提供信息、ReMDM 提供纠错，两者正交互补。

**H3**：ASD 的 EMA 记忆（多步）优于 DMI 的单步记忆，因为推理任务需要跨多步积累的上下文。

### 1.5 实验计划

| 实验 | 模型 | 基准 | 指标 | GPU 时间 |
|------|------|------|------|---------|
| ASD 概念验证 | Dream-7B | Countdown-16 pilot | Accuracy | ~0.5h |
| ASD alpha 消融 | Dream-7B | Countdown-100 | Accuracy vs alpha_0, beta, lambda | ~4h |
| ASD vs DMI | Dream-7B | Countdown-500 x 3 seeds | Accuracy, McNemar | ~8h |
| ASD + ReMDM | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~12h |
| ASD on GSM8K | Dream-7B | GSM8K-1319 x 1 seed | Accuracy | ~6h |

**计算成本**：~30 GPU·h（4x GPU 约 7.5h），**成功概率 60%**

### 1.6 失败模式

- **自适应 alpha 引入的方差**：动态 alpha 可能在某些样本上过于激进。对策：设置 alpha_max 上限（0.5），并在 step-level 退火中确保后期步骤 alpha→0
- **EMA 记忆可能引入错误记忆**：早期步骤的 logits 噪声大，EMA 可能传播错误。对策：warmup 前 20% 步不启动 EMA；lambda 选择偏小（0.3-0.5）确保最近步骤权重大

---

## 2. 角度二：Contrastive Denoising Guidance（CDG）—— 用 DLM 的双向性构建免训练对比引导（跨领域迁移）

### 2.1 跨领域灵感

AR 模型中，Contrastive Decoding (CD) 通过对比强/弱模型的 logits 差异来提升生成质量——这一简洁且有效的范式在 2025-2026 年催生了大量工作（DoLa、CCD、TeGu 等）。但 CD 尚未被系统引入 DLM 领域。

关键洞察：**DLM 的去噪过程天然提供了"弱模型"和"强模型"的对比对象**——
- "弱模型"：在高 mask rate 下（早期步骤）的预测，信息不足
- "强模型"：在低 mask rate 下（后期步骤）的预测，上下文丰富

在同一去噪步内，我们可以人工构造一个"更弱"的版本：**随机额外遮蔽部分已揭示 token**，得到一个信息退化的输入，让模型在这个退化输入上预测。然后用正常预测与退化预测的差异来增强正常预测的信号——这本质上就是 Contrastive Decoding 在 DLM 中的自然对应。

### 2.2 提案：Contrastive Denoising Guidance (CDG)

**核心 idea**：在每个去噪步，构造一个信息退化的"amateur"输入（随机额外遮蔽 p% 已揭示 token），用正常预测和退化预测的 logits 差异来放大信号、抑制噪声。

**具体流程**：

1. 在去噪步 t，当前序列 $x_t$ 有 $S_t$ 个已揭示 token
2. **构造 amateur 输入**：随机额外遮蔽 $S_t$ 中 p% 的 token，得到 $\tilde{x}_t$
3. **两次前向传播**：
   - 正常预测：$\ell^+ = f_\theta(x_t)$
   - 退化预测：$\ell^- = f_\theta(\tilde{x}_t)$
4. **对比融合**（仅对 mask 位置）：
   $$\ell_{\text{CDG}}^i = (1 + \gamma) \cdot \ell^{+,i} - \gamma \cdot \ell^{-,i}$$
   其中 $\gamma > 0$ 是对比强度
5. 从 $\text{softmax}(\ell_{\text{CDG}} / T)$ 采样

**直觉**：CDG 放大了"有完整上下文时模型更确信"的 token，抑制了"即使上下文缺失模型也给出的"平凡预测（常见词、重复模式）。这对推理任务尤其有价值——数学推理中的关键数字和操作符高度依赖上下文，而"Let me think"等填充文本不依赖上下文。

### 2.3 与 CORE 的对比

CORE (arXiv 2602.04096) 也利用上下文扰动，但其目标是**识别脆弱 token 做 remasking**。CDG 的目标完全不同——CDG 不做 remasking，而是**直接修改预测分布**，引导模型生成更依赖上下文的 token。

| 维度 | CORE | CDG |
|------|------|-----|
| 目标 | 识别脆弱 token → remask | 增强上下文依赖信号 → 更好的采样 |
| 扰动对象 | 随机 mask 上下文 | 随机额外 mask 已揭示 token |
| 使用方式 | Remasking 指导 | Logit 对比融合 |
| 前向传播数 | ~K 次扰动 | 2 次（正常 + amateur）|
| 与 DMI 兼容性 | 独立 | CDG 可与 ASD 叠加 |

### 2.4 与 CCD (Confidence-Driven Contrastive Decoding) 的区别

CCD (arXiv 2602.18232) 在 AR 模型中用置信度驱动的对比解码减少推理错误。CDG 将这一思想迁移到 DLM，但有两个关键区别：
- CCD 在 low-confidence token 位置做干预，CDG 在所有 mask 位置做干预
- CCD 用 high-confidence token 的 placeholder 构造 amateur，CDG 用随机额外 masking 构造 amateur（利用 DLM 的 mask 机制）

### 2.5 假设

**H4**：CDG 在 Dream-7B Countdown-500 上的准确率将显著优于 vanilla（预期 +3-5pp），因为对比引导放大了推理关键 token 的信号。

**H5**：CDG + ASD 叠加效果优于单独使用——ASD 提供跨步记忆，CDG 提供步内信号增强，两者在不同维度上互补。

### 2.6 实验计划

| 实验 | 模型 | 基准 | 指标 | GPU 时间 |
|------|------|------|------|---------|
| CDG 概念验证 | Dream-7B | Countdown-16 pilot | Accuracy | ~0.5h |
| CDG gamma 消融 | Dream-7B | Countdown-100 | Accuracy vs gamma, p | ~4h |
| CDG vs vanilla/DMI | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~12h |
| CDG + ASD | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~12h |
| CDG on GSM8K | Dream-7B | GSM8K x 1 seed | Accuracy | ~8h |

**计算成本**：~36 GPU·h（4x GPU 约 9h），**成功概率 50%**

### 2.7 失败模式

- **对比过度**：gamma 过大可能导致 logits 分布极端化，产出低概率 token。对策：gamma 上限 0.5，配合温度 T 校准
- **amateur 信号太弱**：如果只额外 mask 10% token，正常和退化预测差异可能微乎其微。对策：从 p=30% 开始扫描，确保足够的信息差
- **推理任务中 token 间依赖性过强**：额外 masking 可能破坏关键依赖链。对策：优先 mask 低置信度（即不那么关键的）token，保留高置信度 token

---

## 3. 角度三：Denoising Trajectory Distillation（DTD）—— 让去噪轨迹自身成为学习信号（全新方法）

### 3.1 核心动机

DTA 的失败揭示了一个深刻的问题：**在去噪过程中，MLM 自监督损失（对已揭示 token 做 mask-and-predict）无法提供足够的信号来改善推理**。模型对已揭示 token 的 re-prediction 置信度已经很高（loss 在 0.005-0.032 范围），梯度信号微弱。

但换一个角度思考：去噪过程中最有价值的信号不在"已揭示 token 的 re-prediction"，而在**去噪轨迹本身的动态模式**。一个成功的去噪轨迹和一个失败的去噪轨迹有什么区别？

**灵感来源**：
- **Noise Hypernetwork (arXiv 2508.09968)**：将 test-time compute scaling 的知识蒸馏回模型，用超网络学习 reward-tilted 初始噪声分布
- **Free Lunch for Pass@k (arXiv 2603.04893)**：发现 DLM 天然具有多样性优势，增加 pass@k 的多样性是"免费午餐"
- **Grokking 的几何分析 (arXiv 2602.16746)**：发现 Transformer 训练沿低维子空间演化，且正交方向的曲率增长预示泛化

### 3.2 提案：Denoising Trajectory Distillation (DTD)

**核心 idea**：并行运行 K 条去噪轨迹，在每个中间步骤用轨迹间的一致性/分歧来动态调整剩余步骤的预测，无需外部验证器，也无需梯度更新。

**具体方案**：

1. **并行轨迹启动**：从同一 prompt 出发，用不同随机种子启动 K 条独立的去噪轨迹（K=4-8），每条使用标准去噪过程

2. **轨迹一致性评估**（每 M 步执行一次）：
   - 对每个 mask 位置 i，收集 K 条轨迹的当前预测 $\{p_1^i, ..., p_K^i\}$
   - 计算一致性分数：$C_i = 1 - H(\bar{p}^i) / \log V$（所有轨迹平均 logits 的熵归一化）
   - 高一致性位置：K 条轨迹一致预测同一 token → 大概率是"容易/正确"的位置
   - 低一致性位置：K 条轨迹分歧大 → "困难/不确定"的位置

3. **一致性引导的 logit 混合**：
   - 对高一致性位置 (C_i > 0.8)：用轨迹平均 logits 替代各自的 logits（ensemble 效果）
   - 对低一致性位置 (C_i < 0.3)：**保持各轨迹独立**，维持多样性
   - 中等一致性：线性插值

4. **轨迹级筛选与拼接**：
   - 每 M 步评估每条轨迹的"轨迹质量分数"：$Q_k = \frac{1}{|S_t|} \sum_{i \in S_t} p_k(x_i | x_t^k)$（已揭示 token 的平均自信度）
   - 最终输出：选择质量分数最高的轨迹，或跨轨迹选择高一致性位置的最佳 token

**与 Self-Rewarding SMC 的区别**：
- Self-Rewarding SMC (arXiv 2602.01849) 用粒子权重做重要性采样（概率框架）
- DTD 用一致性做 logit 混合（信息融合框架），计算上更轻量
- DTD 的关键创新在于**利用轨迹分歧作为 difficulty indicator**，对困难位置保持多样性而非强行 ensemble

### 3.3 与并行投票 (Parallel Vote) 的区别

之前的并行投票方法 (task_2c) 完全失败（PPL +114%），因为它在最终步对所有位置做多数投票，导致不连贯文本。DTD 有三个关键改进：
1. **渐进式融合**：不在最终步才融合，而是每 M 步渐进引导
2. **一致性感知**：只在高一致性位置做 ensemble，低一致性位置保持独立
3. **轨迹级而非 token 级**：最终输出整条轨迹而非 token 拼接

### 3.4 假设

**H6**：DTD (K=4) 在 Dream-7B Countdown-500 上的准确率将显著优于 vanilla 和 DMI（预期 12-16%），因为多轨迹一致性提供了正确性的隐式验证信号。

**H7**：DTD 的计算开销与 K 线性相关，但通过 batch 化前向传播，实际 wall-clock 增加 < K x（预期 K=4 时 ~2.5x，因可并行）。

**H8**：DTD + ASD 组合（ASD 提供跨步记忆 + DTD 提供跨轨迹一致性）将是最优配置。

### 3.5 实验计划

| 实验 | 模型 | 基准 | 指标 | GPU 时间 |
|------|------|------|------|---------|
| DTD 概念验证 (K=4) | Dream-7B | Countdown-16 pilot | Accuracy | ~1h |
| DTD K 消融 (K=2,4,8) | Dream-7B | Countdown-100 | Accuracy vs K | ~8h |
| DTD vs DMI/vanilla | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~16h |
| DTD + ASD | Dream-7B | Countdown-500 x 3 seeds | Accuracy | ~16h |
| DTD on GSM8K | Dream-7B | GSM8K x 1 seed | Accuracy | ~12h |

**计算成本**：~53 GPU·h（4x GPU 约 13h），**成功概率 45%**

### 3.6 失败模式

- **一致性 ≠ 正确性**：所有轨迹可能一致地预测错误 token（集体幻觉）。对策：将一致性引导限制在 C_i > 0.8 的极高一致性位置
- **K 条轨迹内存开销**：K=8 需要 8x 的 KV cache。对策：用 batch inference（单次前向传播 batch_size=K），Dream-7B 15GB × 8 = 120GB 超出 98GB 单卡。实际限制 K=4 单卡或 K=8 双卡
- **渐进融合可能破坏轨迹独立性**：早期融合后各轨迹趋同，丧失多样性。对策：融合间隔 M 设较大（如 M=T/4），且只对高一致性位置融合

---

## 4. 综合策略与优先级排序

### 4.1 三个提案的互补性

三个角度基于不同的信号来源和作用机制：

```
信号来源       作用机制        计算开销
ASD:  跨步 logit 历史  →  动态 embedding 注入   ~1.1x
CDG:  步内上下文对比    →  logit 分布增强        ~2x
DTD:  跨轨迹一致性      →  ensemble + 筛选       ~2.5-4x (K=4)
```

三者完全正交，理论上可以叠加：ASD + CDG + DTD。

### 4.2 推荐实验顺序

1. **Phase 1（2 天）**：ASD 概念验证 + CDG 概念验证（独立 pilot，可并行）
   - 快速确认哪些方向有信号
   - 如果 ASD 在 pilot 上显著优于 DMI，立即进入 full-scale

2. **Phase 2（3 天）**：最优方法 full-scale 评估 + DTD pilot
   - Countdown-500 x 3 seeds 完整评估
   - DTD 在 16-sample pilot 上验证可行性

3. **Phase 3（2 天）**：组合实验 + GSM8K/MBPP 扩展
   - 最优单方法 + 组合方法在 GSM8K 上评估
   - 消融实验（alpha schedule、gamma、K）

4. **Phase 4（2 天）**：统计检验 + 论文写作
   - McNemar test + Bootstrap CI
   - 与 MetaState、CORE、Prism 等方法的定位对比

### 4.3 投稿定位

**如果 ASD 和/或 CDG 成功（>10% Countdown，显著优于 vanilla）**：
方法论论文——"Soft Information Persistence in Masked Diffusion: Training-Free Cross-Step Memory and Contrastive Guidance"。投稿 NeurIPS/ICML 主会。核心卖点：首个证明 DLM 跨步信息传递可免训练实现显著改善的方法，对比 MetaState（需训练）有明确优势。

**如果 DTD 成功（>15% Countdown）**：
升级为更完整的论文——"Beyond Remasking: Multi-Signal Inference-Time Scaling for Diffusion Language Models"，整合 ASD+CDG+DTD 三层方法。投稿 NeurIPS/ICLR 主会。

**如果全部失败但 DMI 的 9.3% 可复现**：
聚焦 DMI 的正面结果 + 负面发现的系统分析——"What Works and What Doesn't: A Systematic Study of Inference-Time Scaling for Masked Diffusion Language Models"。投稿 EMNLP 或 ICLR。

### 4.4 总计算预算

| 方案 | GPU·h | 成功概率 | 潜在影响 |
|------|-------|---------|---------|
| ASD | ~30 | 60% | 中-高（DMI 的自然进化，最低风险）|
| CDG | ~36 | 50% | 高（跨领域迁移，新颖度高）|
| DTD | ~53 | 45% | 高（多轨迹框架，如成功影响力大）|
| **总计** | **~119** | -- | 至少一个成功的概率 ~88% |

4x GPU 约需 30h 总计算时间。考虑到当前 full-scale 实验已在运行，可以在等待 DTA/SCP 结果的同时启动 ASD 和 CDG 的 pilot。

---

## 5. 关键文献引用

### 直接前驱
1. **ReMDM** (Nisonoff et al., 2025, arXiv 2503.00307) — remasking 框架基石
2. **MetaState** (Xia et al., 2026, arXiv 2603.01331) — 首次提出 DLM "Information Island" 问题；DMI/ASD 的动机来源
3. **CORE** (Zhai et al., 2026, arXiv 2602.04096) — 上下文扰动用于 remasking，CDG 的对标方法
4. **Soft-Masked Diffusion** (Hersche et al., 2025, arXiv 2510.17206) — DMI/ASD 的 embedding 混合先驱
5. **Self-Rewarding SMC** (Luo et al., 2026, arXiv 2602.01849) — DTD 的多粒子框架启发

### 跨领域灵感
6. **CCD: Confidence-Driven Contrastive Decoding** (Tang et al., 2026, arXiv 2602.18232) — CDG 的 AR 领域前驱
7. **TC-LoRA** (arXiv 2510.09561) — 动态条件 LoRA for diffusion，ASD 自适应 alpha 的灵感
8. **Noise Hypernetwork** (Eyring et al., 2025, arXiv 2508.09968) — 将 test-time scaling 知识蒸馏回模型
9. **Free Lunch for Pass@k** (Lamont et al., 2026, arXiv 2603.04893) — DLM 天然多样性优势
10. **TeGu: Temporal Guidance** (Zheng & Li, 2026, arXiv 2601.21744) — 时间维度对比引导

### 理论基础
11. **MDMs are Time-Agnostic** (Zheng et al., 2024, arXiv 2409.02908) — DLM 去噪的时间无关性
12. **Generation Order in MDMs** (arXiv 2602.00286) — remasking 无法保证分布正确性
13. **DLMs are Optimal Parallel Samplers** (Jiang et al., 2025, arXiv 2512.25014) — remasking 的理论必要性
14. **Scaling Beyond MDLMs** (Sahoo et al., 2026, arXiv 2602.15014) — 离散扩散缩放定律

### 项目内部证据
15. 前 18 轮迭代数据（`logs/iterations/`）——TTT/Best-of-N/ReMask-Retry 系统失败
16. 当前迭代 DTA pilot——参数级适应 6.2% < vanilla 12.5%
17. 当前迭代 DMI full-scale——**9.3% vs vanilla 4.7%（~2x 改善，唯一正面结果）**
