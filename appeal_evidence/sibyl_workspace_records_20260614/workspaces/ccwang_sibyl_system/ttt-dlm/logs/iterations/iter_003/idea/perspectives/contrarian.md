# 反对者提案（Iteration 3）：DTA 的七个致命假设

**日期**: 2026-03-09
**视角**: 系统性反驳、压力测试、反面证据挖掘

---

## 核心立场

当前提案选择了 DTA（Denoising-Time Adaptation）作为核心方案——在去噪过程中用 LoRA 在线更新参数以实现"推理时学习"。这个想法在叙事上很优雅："DLM 的去噪天然是隐式 TTT，只需显式化即可"。但**优雅的叙事不等于正确的科学**。我将系统性地挑战 DTA 的七个隐含假设，每一个都有文献证据支撑。如果这些假设中有任何一个不成立，DTA 的整个论证链条就会断裂。

---

## 假设 1：「去噪步间的参数更新产生的梯度信号质量足够高」

### DTA 的隐含假设

DTA 在每步去噪后，用已揭示 token 的 masked LM 损失计算梯度更新 LoRA 参数。这假设：(a) 已揭示的 token 提供了关于目标序列的高质量梯度信号，(b) 这些梯度的方向在参数空间中是有益的。

### 反面证据

**事实 1：TTT 的梯度信号在代码生成中失败得很彻底。** Barnes (arXiv 2602.07670) 在 120B 参数模型上对比了 TTT 与 Best-of-N，发现 TTT 的"equivalent K"低于 1——**比单次推理更差**。失败模式是**过度锐化（over-sharpening）**：梯度更新将分布坍缩到次优区域。这不是小模型的问题（120B 参数），也不是步数的问题（1-5 步梯度更新），而是**梯度适应本身在 verifiable tasks 上的结构性缺陷**。

**事实 2：DLM 的去噪损失与 AR 的 next-token prediction 损失有本质区别。** TTT 在 AR 模型上成功的核心前提是 next-token prediction 损失与下游任务（推理、代码）之间的强相关性。但 DLM 的 masked LM 损失是**双向的、位置无关的**——模型同时预测所有 mask 位置，梯度信号是所有位置的平均。这种平均化的梯度能否在参数空间中找到正确的更新方向，目前没有任何理论或实验证据。

**事实 3：Hübotter et al. (arXiv 2509.24510) 的理论分析表明 TTT 的有效性来自"泛化后的特化"。** 该工作证明 TTT 在基础模型"全局欠参数化"时最有效——TTT 通过将容量聚焦到与测试任务相关的概念上实现特化。但 DTA 的设定完全不同：LoRA 参数从零初始化，在 T 步去噪中从极低信息状态开始积累。这是**冷启动**，不是特化。模型需要在 20-100 步去噪中从零学到足够的任务特定知识，这对梯度信号的质量和一致性提出了极高要求。

### 反对方向

如果梯度信号质量不够，DTA 的 LoRA 更新要么无效（梯度太小/方向错误），要么有害（过度锐化）。应首先在**可控环境**中验证：在完全已知答案的合成任务上，DTA 的梯度方向是否确实指向正确的参数空间区域。如果连这个验证都无法通过，后续的 benchmark 实验将毫无意义。

---

## 假设 2：「LoRA 的表达力足以捕捉去噪过程中需要的适应性变化」

### DTA 的隐含假设

DTA 使用极小的 LoRA rank（r=4-8），参数量 < 0.01% 总参数。这假设去噪过程中需要的"适应性"变化是低秩的。

### 反面证据

**事实 1：MetaState (arXiv 2603.01331) 解决同一问题时选择了完全不同的架构。** MetaState 用 GRU 风格的 Updater + 交叉注意力来桥接去噪步——这是一个**专门训练的、高表达力的跨步记忆机制**，需要 K-step unrolling 训练。如果低秩 LoRA 更新就足够了，MetaState 为什么要设计如此复杂的架构？反过来说，MetaState 的存在暗示**去噪步间的信息传递需要的表达力远超低秩线性适配器**。

**事实 2：Locas (arXiv 2602.05085) 的关键发现是"正确初始化至关重要"。** Locas 证明了低秩参数记忆的有效性高度依赖于**利用模型参数、激活和/或梯度进行有原则的初始化**。DTA 使用零初始化（"旁路初始化"），这恰恰是 Locas 证明效果最差的方式。Locas 需要预训练数据上的初始化过程来找到好的起点——DTA 没有这个奢侈。

**事实 3：GIFT (arXiv 2509.20863) 发现 DLM 的 LoRA 微调需要 token 重要性加权。** 该工作在 DLM 上应用 LoRA 时，必须根据 token 的熵赋予不同的重要性权重才能获得稳定且有效的微调。DTA 的在线更新使用的是所有已揭示 token 的均匀 masked LM 损失——没有重要性加权机制。在去噪早期，已揭示的 token 多为高置信度的"简单" token，这些 token 的梯度信号可能与模型真正需要学习的"困难" token 无关甚至矛盾。

---

## 假设 3：「去噪过程中的参数更新不会导致灾难性漂移」

### DTA 的隐含假设

DTA 在 T 步去噪中累积更新 LoRA 参数，用衰减因子 γ ∈ [0.9, 0.99] 防止漂移。假设这足以保持数值稳定性。

### 反面证据

**事实 1：Buffer layers (arXiv 2510.21271) 的核心发现是——即使是轻量级的参数更新也会导致灾难性遗忘。** 该工作专门研究了 test-time adaptation 中的遗忘问题，发现"修改模型核心参数"（即使是 normalization 层）会在在线适应过程中导致严重的灾难性遗忘。他们的解决方案是引入完全独立的 Buffer 层，**保持预训练骨架完全不变**。DTA 通过 LoRA 修改的恰恰是模型的核心 FFN 层。

**事实 2：LATTA (arXiv 2510.05530) 证明确定性 TTT 更新在复杂损失面上不稳定。** 该工作的核心论点是：TTT 的确定性梯度更新容易陷入糟糕的局部最小值，导致性能退化。他们的解决方案是引入 SGLD 噪声扰动——但 DTA 的单步 SGD 更新没有这种稳定性机制。

**事实 3：DLM 的 T 步去噪放大了漂移风险。** AR 模型的 TTT 是**一次性**的（在前向传播前适应一次），而 DTA 需要在 T=64-256 步中**连续更新**参数。即使每步的漂移很小，累积效应也可能显著。衰减因子 γ=0.99 在 T=100 步后仍保留 γ^100 ≈ 37% 的初始更新——但前 20 步的更新（此时只有少量 token 被揭示，梯度信号最差）在第 100 步仍然影响着模型。这种"坏的早期梯度长期残留"的问题是 DTA 独有的风险。

---

## 假设 4：「ELBO 单调性命题为 DTA 提供了理论保证」

### DTA 的隐含假设

提案声称 DTA 等价于 EM 优化，每步改善变分下界。这暗示 DTA 有理论收敛保证。

### 反面证据

**事实 1：该命题的前提条件在实践中几乎不可能满足。** 命题要求 L 关于 Δθ **强凸** + L2 正则化 + f_θ 连续。但神经网络的损失面**不是凸的**，更不用说强凸。深度 Transformer 的 masked LM 损失关于 LoRA 参数的 Hessian 是高度非凸的。声称"温和的正则性条件"实际上是在假设一个与现实完全脱节的理想化模型。

**事实 2：EM 算法的收敛保证在神经网络中失效。** 经典 EM 的收敛保证依赖于**完全最大化 M-step**。DTA 的 M-step 是**单步 SGD**，这是一个极度近似的 M-step。在非凸优化中，单步 SGD 不保证改善变分下界——它甚至可能恶化下界。用"EM"来包装 DTA 是一种修辞策略，而非真正的理论保证。

**事实 3：信息积累命题 I(Δθ^(t); x_0) >= I(Δθ^(t-1); x_0) + ΔI_t 要求 ΔI_t > 0。** 但如果第 t 步新揭示的 token 是"容易的"（高置信度、低信息量），ΔI_t 可能接近零甚至为负（如果梯度更新引入噪声）。这个命题在最需要它成立的场景下（困难的推理任务，早期揭示的 token 信息量低）最可能失败。

---

## 假设 5：「DTA 与 remasking 策略正交互补」

### DTA 的隐含假设

提案声称"DTA 更新参数级理解，remasking 修正离散 token 选择，两者互补"。DTA + ReMDM-conf 是推荐的组合方式。

### 反面证据

**事实 1：Remasking 破坏 DTA 的梯度信号基础。** DTA 的更新基于"已揭示 token"的 masked LM 损失。但 remasking 会在每步**重新遮蔽部分已揭示的 token**。这意味着：(a) DTA 在第 t 步基于 token A 计算了梯度并更新了 LoRA；(b) 在第 t+1 步 token A 被 remask 了；(c) 第 t 步的 LoRA 更新现在是基于**已不存在的信息**做出的——这是一种"幽灵梯度"。

**事实 2：Piskorz et al. (arXiv 2511.21338) 证明 mask 主动干扰上下文理解。** 如果 remasking 增加了 mask 比例，模型对上下文的理解变差，那么 DTA 在 remasking 后的梯度更新的质量也会变差。两种方法不是互补，而是**互相干扰**——remasking 恶化 DTA 的梯度质量，DTA 的参数漂移可能使 remasking 的置信度估计失准。

**事实 3：COVER (arXiv 2602.06161) 发现 revocable decoding 存在 flip-flop 振荡问题。** 当 token 被反复 mask 和 unmask 时，模型的预测会在两个状态之间振荡。DTA 的参数更新会使这种振荡更加不稳定——因为 LoRA 参数记住了振荡历史的梯度。

---

## 假设 6：「DLM 的去噪过程"天然是隐式 TTT"」

### DTA 的隐含假设

这是 DTA 最核心的叙事：每步去噪等价于一次自监督学习，只需"显式化"即可。

### 反面证据

**事实 1：隐式学习 ≠ 显式学习可以改善它。** 标准去噪过程中，模型通过前向传播从当前 mask 序列中提取信息——这是"推理"，不是"学习"。推理和学习在神经网络中是根本不同的操作：推理保持参数不变，利用已有知识；学习修改参数，获取新知识。仅仅因为两者都涉及"从部分信息中预测"，不意味着将推理转为学习就一定有益。类比：人在做数独时，每填一个数就获取了新的约束信息（"推理"），但你不需要修改大脑的突触权重（"学习"）来利用这些信息。

**事实 2：Bansal et al. (arXiv 2512.13898) 的关键发现是 TTT 对长上下文有效，但现有推理时策略（如产生更多 thinking tokens）"rapidly diminishing returns and fail at long context"。** 他们将 TTT 的优势归因于克服"score dilution"——一种静态自注意力的固有缺陷。但 DLM 的去噪序列通常很短（128-512 tokens），不存在长上下文的 score dilution 问题。DTA 想要解决的"信息孤岛"问题可能根本不需要参数更新来解决——标准的双向注意力在短序列上已经足够有效。

**事实 3：JitRL (arXiv 2601.18510) 证明无梯度的推理时优化可以超越 TTT。** 该工作提出了一种完全不用梯度更新的推理时策略优化方法，通过非参数记忆 + logit 调制实现，理论上是 KL 约束策略优化的闭式解。在 WebArena 上超越了计算昂贵的微调方法，且成本低 30 倍以上。这直接质疑了"推理时改善必须通过梯度更新"的假设。

---

## 假设 7：「2.5-3x 的计算开销是可接受的」

### DTA 的隐含假设

提案估计 DTA 的计算开销约为标准去噪的 2.5x（因为每步额外的反向传播约为 2x 前向传播成本），DTA + ReMDM 约 3x。

### 反面证据

**事实 1：这个估计严重低估了实际开销。** 反向传播的成本不仅是 2x 前向传播——它还需要存储所有中间激活以计算梯度。对于 7B 模型的 LoRA（即使只更新最后 2-4 层），每步的内存开销大约翻倍。在 4x GPU（24GB 或 80GB）上，这可能导致 batch size 下降或 OOM。实际开销可能是 4-5x 而非 2.5x。

**事实 2：Sahoo et al. (arXiv 2512.10858) 已经证明 DLM 需要 ~16x 计算才能匹配 AR 模型。** 在这个基础上再加 3x DTA 开销，总成本是 AR 模型的 ~48x。以相同的计算预算，AR 模型 + standard TTT/CoT/Best-of-N 可以获得远更大的提升。DTA 的竞争对手不是"无 DTA 的 DLM"，而是"相同计算预算下的最优策略"——很可能是直接用 AR 模型。

**事实 3：LatentSeek (arXiv 2505.13308) 在潜在空间中做推理时适应，通常在"几次迭代内"就收敛。** 如果参数空间的适应需要 T=64-256 步才能积累足够信息，而潜在空间的适应只需要几步，这暗示**参数空间不是推理时适应的正确抽象层**。

---

## 对 DTA 提案其他要素的质疑

### 消融基线谱系（DMI → SCP → DTA）的问题

DMI（embedding 级跨步软信息注入）的设计与 MetaState 高度重叠但粗糙得多。MetaState 用 GRU + 交叉注意力 + K-step unrolling 训练来做跨步信息传递，而 DMI 只是简单地将上一步 logits 注入。如果 DMI 有效，MetaState 的复杂设计就是多余的；如果 DMI 无效，它作为消融基线没有诊断价值（因为失败原因不清——是方法本身不好还是信息传递不重要？）。

### "唯一同时满足'零训练 + 参数级记忆 + 理论保证'"的定位问题

这个定位是通过精心选择对比维度实现的。如果加入"无梯度计算"这一列，DTA 就不再独特了。如果加入"已有实验验证"，DTA 是表中唯一没有任何实验结果的方法。在没有实验验证之前宣称独特定位，是**预售而非科学**。

---

## 基于反面真相的替代建议

### 建议 1：先验证再投入——用最小可行实验否证或确认

在投入 84 GPU·h 之前，花 2 GPU·h 做以下诊断：

1. **梯度方向测试**：在 Countdown（有明确正确答案）上，运行 DTA 的完整去噪过程。在每步记录：(a) LoRA 参数的 L2 范数变化，(b) 最终答案的准确率 vs vanilla，(c) 梯度与"正确参数方向"（用正确答案的监督梯度定义）的余弦相似度。如果余弦相似度接近零或为负，则 DTA 的梯度信号无效，应立即放弃。

2. **MetaState 对比测试**：MetaState 已发布代码（或至少有详细实现描述），在 Dream-7B 上直接测试其 Countdown 准确率。如果 MetaState（需要训练）比 DTA（无需训练）效果差或相当，说明参数更新不是瓶颈；如果 MetaState 远优于 DTA，说明 DTA 的低秩假设有问题。

3. **简单基线验证**：在 DTA 之前，先测试一个极简基线——**温度退火 + 增加去噪步数**。前 18 轮的温度退火已经是唯一有效的方法。如果增加步数 + 温度退火的效果与 DTA 相当，DTA 的额外复杂性就不值得。

### 建议 2：如果坚持参数适应方向，应关注 MetaState 而非 DTA

MetaState 与 DTA 解决的是同一个问题（信息孤岛），但 MetaState 的方案更成熟：
- **有 K-step unrolling 训练**：暴露在多步去噪动态中学习，而非盲目的在线梯度更新
- **保持骨架完全冻结**：不修改预训练权重，避免灾难性遗忘
- **固定大小的工作记忆**：与序列长度无关，计算开销可控
- **已有初步实验结果**：在 LLaDA-8B 和 Dream-7B 上"consistently improves accuracy over frozen baselines"

如果要做跨步记忆方面的工作，更明智的策略是**改进 MetaState**（如 training-free 版本、更高效的记忆机制）而非从头提出一个未经验证的替代方案。

### 建议 3：重新审视"不可能性边界"方向

上一轮提案中的方向 1（推理时计算扩展的信息论上界）仍然是最有学术价值的方向。18 轮的负面结果 + DTA 可能的失败 = 充分的实验证据，支撑一篇"为什么推理时扩展在 DLM 上如此困难"的深度分析论文。这种论文的引用率远高于"又一种 marginal improvement 的方法"。

---

## 诚实的风险评估

| 场景 | 概率 | 后果 |
|------|------|------|
| DTA 在 Countdown 上显著有效（+5pp 以上）| 15% | 证明反对者错误，好消息 |
| DTA 有微弱效果（+1-3pp），统计不显著 | 40% | 浪费 2 周时间，回到起点 |
| DTA 无效果或负面效果 | 35% | 验证反对者预测，但时间已浪费 |
| DTA 导致数值不稳定（LoRA 发散/NaN） | 10% | 需要大量调试，延迟进度 |

**总体判断**：DTA 有 45% 概率浪费时间、10% 概率导致技术问题。在这种风险 profile 下，至少应该将 84 GPU·h 的预算分成两期——**第一期 4 GPU·h 做诊断实验**（上述建议 1），只有在诊断结果积极时才进入第二期主实验。这是**最低限度的风险管理**。

---

## 参考文献

- Barnes (2026). Surprisal-Guided Selection: Compute-Optimal Test-Time Strategies. arXiv 2602.07670
- Hübotter et al. (2025). Specialization after Generalization: TTT in Foundation Models. arXiv 2509.24510
- Xia et al. (2026). MetaState: Persistent Working Memory for Discrete Diffusion LMs. arXiv 2603.01331
- Lu et al. (2026). Locas: Pluggable Parameterized Memory. arXiv 2602.05085
- Xu et al. (2025). GIFT: Guided Importance-Aware Fine-Tuning for Diffusion LMs. arXiv 2509.20863
- Kim et al. (2025). Buffer layers for Test-Time Adaptation. arXiv 2510.21271
- Vejendla (2025). LATTA: Langevin-Anchored Test-Time Adaptation. arXiv 2510.05530
- Bansal et al. (2025). Let's (not) just put things in Context: TTT for Long-Context LLMs. arXiv 2512.13898
- Li et al. (2026). JitRL: Just-In-Time Reinforcement Learning without Gradient Updates. arXiv 2601.18510
- Li et al. (2025). LatentSeek: Reasoning via Test-Time Policy Gradient in Latent Space. arXiv 2505.13308
- Piskorz et al. (2025). Masks Can Be Distracting. arXiv 2511.21338
- Zhang et al. (2026). COVER: Cache Override Verification. arXiv 2602.06161
- Sahoo et al. (2025). Scaling Behavior of Discrete Diffusion Language Models. arXiv 2512.10858
- Zheng et al. (2024). Masked Diffusion Models Are Secretly Time-Agnostic. arXiv 2409.02908
- Jiang et al. (2026). Generation Order and Parallel Decoding in MDMs. arXiv 2602.00286
