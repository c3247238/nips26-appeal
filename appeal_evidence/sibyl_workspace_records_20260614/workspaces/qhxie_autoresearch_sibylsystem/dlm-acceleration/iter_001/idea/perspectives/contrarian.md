# 反对者视角：DLM 推理加速批判

## 核心质疑

本研究提案的核心假设是：KV caching、adaptive step scheduling、AR-guided unmasking 和 speculative decoding 四种方法可以正交组合以实现叠加加速。这个假设存在三个根本性问题：

1. **"正交性"是一个未经验证的乐观假设**。这四种方法在计算图上存在深层耦合——它们都在修改同一个去噪迭代过程的不同方面，但这些修改会相互干扰。声称它们是"正交的"相当于声称对一个非线性系统的四种独立扰动可以线性叠加，这在一般情况下是不成立的。

2. **"systematic comparison" 的新颖性存疑**。文献调查中已经明确列出 Gap 3（"no systematic comparative study"），但这个 gap 的存在可能恰恰说明社区认为这种比较的学术价值有限——如果真有价值，以 NVIDIA（Fast-dLLM）、NeurIPS（LLaDA）等团队的资源，早就做了。更可能的原因是：各方法的实现细节和评估条件差异太大，使得公平比较本身就是一个工程问题而非科学问题。

3. **目标模型选择过于单一**。仅在 LLaDA-8B-Instruct 上验证，无法回答"这些加速技术是否泛化"这一更重要的问题。Dream 7B 的架构差异（如其不同的 noise schedule 和 sampling strategy）可能导致完全不同的正交性结论。

## 各方法的根本局限

### KV Caching

**核心矛盾：近似 KV cache 的理论基础薄弱。**

AR 模型的 KV cache 是精确的——因果掩码保证了已生成 token 的 KV 对在后续步骤中完全不变。DLM 的 KV cache 本质上是一种近似（approximation），因为双向注意力意味着每个 token 的 KV 表示依赖于整个序列的当前状态，包括所有仍在变化的 masked token。

**具体局限：**

- **误差累积无理论保障**。Fast-dLLM、dKV-Cache、Elastic-Cache、EntropyCache 等方法都依赖经验观察（"KV 在相邻步骤间变化不大"）来正当化缓存复用。但这个经验观察在去噪过程的关键转折点（如大量 token 同时从 mask 转为确定状态时）可能严重失效。没有任何工作给出了 KV cache 近似误差对最终生成质量影响的理论上界。

- **局部性假设在推理任务中可能不成立**。Window-Diffusion 声称 99x 加速，但其核心假设（"远处 token 对当前 token 的注意力贡献可忽略"）在需要长距离推理的任务（如多步数学证明、代码中的跨函数引用）中很可能失效。该方法主要在 LLaDA 和 Dream 上评估，且 99x 的数字是在最激进的窗口设置下取得的，实际可用的加速比可能远低于此。

- **不同 KV cache 方法的刷新策略相互冲突**。Elastic-Cache 用注意力权重驱动刷新，EntropyCache 用 token entropy 驱动刷新，Fast-dLLM 用 block-wise 策略。如果要组合 KV caching 和其他加速方法，到底用哪种刷新策略？这些策略与 adaptive step scheduling 的交互效果完全未知。

- **Sahoo et al. (2602.15014) 的发现暗示更深层问题**。"Scaling Beyond Masked Diffusion" 论文显示 masked diffusion 的 perplexity 优势并不总是转化为下游任务优势——uniform-state diffusion 在 GSM8K 上超越了 masked diffusion，尽管 perplexity 更差。这意味着我们对 DLM 内部表示的理解还很不充分，在此基础上做 KV cache 近似的假设（"KV 变化小 = 可以安全缓存"）可能从根基上就有问题。

### Adaptive Step Scheduling

**核心矛盾：置信度不等于正确性。**

- **置信度校准问题**。DLM 的 token 置信度（通常用预测概率衡量）并不一定能反映实际正确性。在 AR 模型中，这个问题已经有大量研究（calibration literature），而在 DLM 中几乎没有被系统研究。一个高置信但错误的 token 如果被提前确定（early commit），其错误将在后续步骤中被当作"已确定的上下文"传播，导致级联错误。

- **Saber 的 backtracking 机制承认了这个问题**。Saber 引入了回溯机制来修正提前确定的错误 token，但这本身就证明了 adaptive scheduling 的内在不可靠性。更关键的是，backtracking 引入了额外复杂性，使得加速效果的理论分析几乎不可能——你无法预测需要多少次回溯。

- **DCD (Deferred Commitment Decoding) 的发现更加令人担忧**。DCD 论文 (2601.02076) 明确指出了"Boundary-Induced Context Truncation"问题——block 边界处的 token 被迫在缺乏足够上下文的情况下做出承诺。这不仅是 block diffusion 的问题，而是所有 adaptive scheduling 方法的共性隐患：任何提前确定 token 的决策都在隐式地截断该 token 可获得的未来上下文。

- **任务依赖性极强**。Saber 主要在代码生成上评估（251.4% 加速），但代码具有高度局部结构性（括号匹配、缩进模式等），使得高置信早停特别有效。在开放式文本生成或需要长距离一致性的推理任务中，这种优势可能大幅缩水。

### AR-guided Unmasking

**核心矛盾：引入了 DLM 本来要消除的瓶颈。**

- **根本悖论**。DLM 的核心价值主张是"摆脱自回归约束，实现并行生成"。AR-guided unmasking 又引入了一个 AR 模型来指导解码顺序，这实际上是在用 DLM 来近似一个 AR 系统。如果最终需要 AR 模型来获得好的解码质量，这严重质疑了 DLM 范式的独立价值。

- **额外内存和计算开销**。FlashDLM 使用 AR 模型作为"supervisor"，这意味着推理时需要在内存中同时加载两个模型。即使 AR guidance 模型较小，在 8B 级别的 DLM 上叠加一个哪怕 1-2B 的 AR 模型也会显著增加内存压力，尤其在 batch inference 场景下。

- **AR 模型本身的局限传递给 DLM**。如果 AR guidance 模型在某些 token 上给出了错误的解码顺序建议，DLM 会继承这些错误。更糟糕的是，AR 模型的错误模式（如在长序列末尾质量退化）与 DLM 的错误模式不同，组合后的系统可能在两种模式的交叉点上表现得比任何单一系统都差。

- **训练分布不匹配**。AR guidance 模型是在 AR 生成的数据分布上训练的，但它需要在 DLM 的去噪过程中提供指导——这两个分布之间存在根本性差异。DLM 的中间状态（部分 masked 序列）不在 AR 模型的训练分布内，其输出的可靠性存疑。

### Speculative Decoding

**核心矛盾：DLM 缺乏 AR speculative decoding 的关键前提条件。**

- **验证机制根本不同**。AR speculative decoding 的优雅之处在于：target model 可以通过一次前向传播同时验证所有 draft token，且有精确的 acceptance-rejection 准则保证输出分布与 target model 完全一致（lossless）。在 DLM 中，不存在这样的一致性保证——因为 DLM 的生成过程本身就是随机的和迭代的，没有一个"ground truth"分布可以作为 acceptance 的参照。

- **DualDiffusion 的局限性**。DualDiffusion (2604.05250) 提出了 draft-verify 框架，但其验证器仍然需要执行完整的去噪过程来判断 draft 质量——这就回到了"用计算换计算"的困境。与 AR speculative decoding 中 target model 的单次前向传播验证不同，DLM 的验证本身就是昂贵的。

- **DiffuSpec 的方向值得注意但走反了**。DiffuSpec (2510.02358) 用 DLM 作为 AR 模型的 drafter——这实际上是把 DLM 降级为 AR 系统的辅助工具，而非加速 DLM 本身。这进一步说明了"在 DLM 内部做 speculative decoding"的困难。

- **Self-speculative 的可行性存疑**。文献提到用 early-layer exit 或量化版本作为 draft network。但 DLM 的 early-layer 表示与 full-model 表示之间的差异模式与 AR 模型不同（因为双向注意力），early exit 是否能产生有意义的 draft 尚无理论或经验支持。

## "看起来正交但实际上不是"的方法对

### KV Caching + Adaptive Step Scheduling

这两种方法表面上作用于不同维度（KV caching 减少单步计算，adaptive scheduling 减少总步数），但存在深层耦合：

- **KV cache 的有效性依赖于相邻步骤间 KV 的相似性**。Adaptive step scheduling 改变了 token 在不同步骤中被确定的模式——高置信 token 提前退出意味着序列的 mask 模式在步骤间变化更剧烈，这**直接破坏了 KV cache 的核心假设**（"相邻步骤的 KV 变化小"）。
- **当多个 token 在同一步被确定时（adaptive scheduling 的典型行为），下一步的 KV 表示会发生突变**，使得之前缓存的 KV 完全失效。
- **实验验证：没有任何现有工作同时测试这两种方法的组合效果**。这不是偶然的——它反映了研究者对这种耦合的隐性认知。

### AR-guided Unmasking + Speculative Decoding

- 两者都试图引入一个"辅助模型"来指导 DLM 的生成过程。如果同时使用，需要三个模型（AR guide + draft model + target DLM），内存和延迟开销可能完全抵消加速收益。
- AR guidance 确定解码顺序后，speculative decoding 的 draft 需要遵循这个顺序——但 draft model 的训练可能并不知道这个顺序约束，导致 draft 质量下降。

### Adaptive Step Scheduling + Speculative Decoding

- Speculative decoding 的 draft 假设了一个固定的去噪过程（固定步数、固定 schedule）。如果 adaptive scheduling 动态改变步数和 token 确定模式，draft model 的预测基础被破坏。
- Draft model 无法预测 target model 的 adaptive decisions，因为这些决策依赖于 target model 的内部置信度——而 draft model 看不到这些信息。

## 真实研究缺口（被忽视的）

### 缺口 1：DLM 在 Batched Inference 下的根本劣势

**这是当前最大的被忽视的缺口。** 几乎所有 DLM 加速论文都只报告 single-sequence latency，但实际部署场景中 throughput（tokens/second across a batch）才是关键指标。

AR 模型在 batch inference 下有独特优势：KV cache 允许每个新 token 的计算量为 O(1) attention + O(d) FFN，独立于序列长度。DLM 的每一步都需要对 **整个序列** 做 full attention（O(N^2)），即使只有少数 token 仍在变化。这意味着：

- **AR 的 batch 计算是 memory-bound**（受限于 KV cache 的内存带宽），可以通过增大 batch size 来提高 GPU 利用率。
- **DLM 的 batch 计算是 compute-bound**（O(N^2) attention 消耗大量 FLOPs），batch size 受限于计算量而非内存。
- 在 batch 场景下，AR 模型可能比 DLM 快一个数量级以上——但没有人做过系统的 roofline analysis 来量化这个差距。

即使所有四种加速方法完美组合，如果无法解决 batch inference 效率问题，DLM 在实际部署中仍然无法与 AR 竞争。这个研究选择了忽视这个根本性问题，转而追求单序列场景下的加速，可能导致研究结果缺乏实际影响力。

### 缺口 2：DLM 加速方法的鲁棒性与 failure mode 分析

现有的所有加速方法都报告了平均加速比和平均质量保持。但没有人研究：

- **哪些输入模式会导致加速方法灾难性失效？** 例如，KV cache 在哪种序列模式下会产生不可接受的质量退化？Adaptive scheduling 在哪种任务上会因过度早停而产生完全错误的输出？
- **加速方法之间的 failure mode 是否相关？** 如果 KV caching 失效的场景恰好也是 adaptive scheduling 失效的场景（都是因为去噪过程中 token 表示剧烈变化），那么组合它们不会带来鲁棒性收益。
- **Worst-case 分析**。在 safety-critical 应用中，average-case 加速无关紧要——worst-case 行为才是部署瓶颈。

### 缺口 3：DLM 内部表示动态的理论理解

所有现有加速方法本质上都是基于经验观察（"KV 变化小"、"置信度与正确性相关"、"局部性成立"）。这些观察缺乏理论基础：

- 为什么 DLM 的 KV 在相邻步骤间变化小？这是否是模型架构（transformer）的固有属性，还是训练数据/目标的偶然结果？
- DLM 去噪过程中是否存在 phase transition（相变）——即某些步骤中表示突然剧变？如果存在，所有基于"连续性"假设的加速方法都会在这些步骤失效。
- FourierSampler (2601.23182) 的频域分析提供了一些初步洞察（低频 = 全局结构，高频 = 局部细节），但这远未构成一个可以指导加速方法设计的理论框架。

## 建议研究方向

基于以上批判分析，如果坚持"DLM 推理加速"这个大方向，我建议以下替代或补充方向：

1. **Failure mode-aware acceleration（故障感知加速）**：不是追求最大平均加速比，而是设计能自动检测"当前加速策略即将失效"并回退到安全模式的方法。这比简单的组合更有学术价值和实用价值。

2. **Batch-level DLM inference optimization（批量推理优化）**：直面 DLM 在 batch 场景下的劣势，研究如何利用 DLM 的独特属性（如不同样本间 mask pattern 的差异性）来设计 DLM-specific 的 batched inference kernel，而非简单移植 AR 的 batch 策略。

3. **理论驱动的 KV cache 设计**：先建立 DLM 去噪过程中 KV 表示动态变化的理论模型（如借鉴 ODE/SDE 连续扩散模型的理论工具），再基于理论推导最优的 cache 刷新策略，而非用启发式方法在实验中调参。

4. **DLM 的 consistency distillation（一致性蒸馏）**：这在图像扩散模型中已经非常成功（Consistency Models 可以将采样步数从 1000 降到 1-4 步），但在离散文本扩散中几乎未被探索。这可能是真正能"改变游戏规则"的方向，远比组合现有方法有潜力。文献调查中将此列为"cross-domain analogy with potential"，但实际上这应该是最优先探索的方向。
