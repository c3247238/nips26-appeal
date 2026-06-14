# Contrarian Perspective: 遮蔽扩散语言模型的推理时计算扩展

## 核心立场

本文件系统性地挑战当前 DLM + TTT/推理时计算扩展研究中的三个主流假设，并基于反面证据提出替代研究方向。目标不是否定一切，而是通过压力测试找出最脆弱的假设和最有价值的缝隙。

---

## 假设一：「推理时计算扩展对 DLM 天然友好，因为 DLM 已经是迭代式的」

### 挑战

这是当前研究最广泛接受的前提——DLM 通过多步去噪生成文本，因此「增加推理步数 = 自然的计算扩展」。但这种直觉存在根本性问题：

1. **并行去噪破坏 token 间依赖关系**。Zhong et al. (2026, arXiv 2601.15593) 对 8 个主流 MDLM（最高 100B 参数）在 58 个 benchmark 上的大规模评估表明：MDLMs **仍然落后于同规模 AR 模型**，核心原因是并行概率建模削弱了 token 间依赖。增加去噪步数并不能修复这个结构性缺陷。

2. **位置对齐脆弱性**。Ye et al. (2026, arXiv 2601.22947) 发现 MDLM 的严格位置预测使解码对 token 错位极其敏感——仅一个位置偏移就能严重破坏语义。这意味着重遮蔽策略（ReMask-Retry、TCR）中的 token 替换可能引入位置不一致，反而恶化输出。

3. **推理时计算存在逆向扩展（Inverse Scaling）现象**。Gema et al. (2025, arXiv 2507.14417) 在大量任务上证明：延长推理推理长度会**恶化**性能。五种失败模式包括：被无关信息干扰、过拟合问题框架、从合理先验滑向虚假相关、复杂推演中丧失焦点、以及扩大有害行为。这些现象在 DLM 的迭代去噪中同样适用——更多步数意味着更多积累错误的机会。

4. **本项目 18 轮迭代的实证否定**。ReMask-Retry 的 PPL 改善被证明由文本退化（重复）驱动；Best-of-N 完全无效（3x 计算量反而 +6.9%）；TTT 6 个变体统计不显著（p=0.88）。这不是方法实现的问题，而是前提假设的问题。

### 反面证据

- **Feng et al. (2025, arXiv 2502.09622)**：「Theoretical Benefit and Limitation of Diffusion Language Model」明确证明 MDM 的效率-精度 trade-off **高度敏感于评估指标选择**，PPL 可能完全无法反映实际生成质量。
- **Zheng et al. (2024, arXiv 2409.02908)**：「Masked Diffusion Models Are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling」揭示 MDM 的采样机制通过降低有效温度来人为压低 PPL，而非真正提升生成质量。
- **Chen et al. (2025, arXiv 2504.02181)**：推理时计算扩展的综述明确指出存在 **diminishing returns**，额外计算轮次提供递减收益，且某些任务呈现 U 形扩展曲线。
- **Balachandran et al. (2025, arXiv 2504.00294)**：跨任务分析表明推理时计算扩展的效果**高度依赖于任务复杂度**，在复杂任务上 diminishing returns 尤为明显。

### 替代研究方向

**方向 A：DLM 的 Generate-then-Edit 范式**

与其试图在去噪过程中扩展计算，不如拥抱 Zhong et al. 提出的 Generate-then-Edit 范式：先快速并行生成（利用 DLM 的速度优势），再用轻量编辑模型精修。这从根本上绕过了「更多去噪步 ≠ 更好质量」的困境。

- 计算成本：GPT-2 级别编辑模型 + Dream-7B 生成器，单 GPU 可行
- 成功概率：60%（范式已有理论支撑但尚未充分验证）

---

## 假设二：「TTT（Test-Time Training）可以为 DLM 注入记忆能力，因为 DLM 的迭代结构天然适合 TTT」

### 挑战

spec 中提出将 TTT 层插入 DLM 以赋予推理时记忆能力。这看似优雅，但忽略了几个关键矛盾：

1. **TTT 的成功建立在序列因果结构之上**。Sun et al. 的 TTT 层、Titans 的记忆模块，以及 TTT-E2E (arXiv 2512.23675) 都是为**自回归**序列建模设计的——它们利用 next-token prediction 作为 self-supervised 信号在推理时更新权重。DLM 的去噪过程**没有因果结构**：所有位置同时被预测，不存在「下一个 token」的概念。直接移植 TTT 的 self-supervised 目标在理论上不成立。

2. **TTT 的序列性瓶颈与 DLM 的并行性矛盾**。TTT 层本质上是序列处理的——每个 chunk 的权重更新依赖于前一个 chunk 的结果。这与 DLM 的核心优势（并行解码）直接冲突。插入 TTT 层意味着放弃 DLM 的速度优势，得不偿失。正如 TTT-Done-Right (arXiv 2505.23884) 所指出：TTT 的 canonical 机制是**内在序列化的**（inherently sequential），小 chunk 更新严重瓶颈现代加速器的并行吞吐。

3. **TTT 在 LLM 上的效果高度不稳定**。Islah et al. (OpenReview) 的研究「Exploring The Effectiveness of Test Time Learning In LLMs for Long Contexts」发现：TTT 的负面结果不应被视为方法失败，而是当模型已经充分学习了目标分布时，test-time 更新**无法有意义地改进**模型。这恰好解释了本项目 TTT 6 个变体全部统计不显著的结果——DLM 的去噪过程中，模型对 mask prediction 的能力已经饱和。

4. **线性注意力的记忆容量存在理论上限**。TTT 层的核心是用线性注意力作为 fast weight 存储。但当序列长度超过维度容量（overcapacity regime）时，检索错误因干扰而不可避免。对于 DLM 的去噪轨迹，每步都需要「记住」之前去噪的结果，这个记忆需求可能很快超过线性注意力的容量上限。

### 反面证据

- **TTT-E2E 需要双 MLP 设计防止灾难性遗忘**（arXiv 2512.23675）：原始 MLP 保存通用能力，新 MLP 学习上下文特定模式。这说明 TTT 的权重更新本身是**破坏性的**——直接插入 DLM 可能破坏去噪能力。
- **Hu et al. (2025, arXiv 2505.20633)**：「Test-Time Learning for Large Language Models」的综述指出 TTL 的效果在不同数据分布下**变化极大**，某些分布下反而降低性能。
- **Sheng et al. (2025, arXiv 2506.24000)**：「The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models」发现许多 TTA 方法的「进步」是虚幻的，实际上低于简单 zero-shot baseline 的表现。

### 替代研究方向

**方向 B：用 DLM 自身的扩散轨迹作为隐式记忆，无需外部 TTT 模块**

关键洞察：DLM 的多步去噪轨迹本身就编码了「决策历史」。与其插入 TTT 层来显式存储记忆，不如利用**轨迹一致性**（trajectory consistency）作为质量信号——在多次去噪中稳定出现的 token 模式可视为「隐式记忆」。

具体方案：对同一输入运行 K 次独立去噪（K=3-5），在每个去噪步骤中，提取跨轨迹的 token 共识（consensus）。不一致的位置即为模型「不确定」的位置，可以有针对性地增加该位置的去噪步数。

这个方向本质上是 TCR（轨迹一致性重遮蔽）的推广，但从「记忆」的角度重新理解：多轨迹共识 = 分布式记忆存储，不一致 = 记忆冲突需要额外计算解决。

- 计算成本：K=3 轨迹 × Dream-7B，约 3x 单次推理成本，单 GPU（A100）约 15 分钟/batch
- 成功概率：45%（TCR 第 18 轮 mean PPL 有改善但 median 未变，说明方法有效但不稳定）

---

## 假设三：「PPL 下降 = 生成质量提升，只要同时检查多样性就够了」

### 挑战

本项目从第 15 轮开始引入多样性指标作为补充，但「PPL + 多样性」的双指标框架仍然存在根本缺陷：

1. **PPL 可以被系统性 gaming 而不触发多样性警报**。Zheng et al. (arXiv 2409.02908) 证明 MDM 的采样机制通过降低有效温度来人为压低 PPL。这种 PPL 降低不需要重复文本（因此不触发多样性指标），而是通过让模型更加保守（更偏向高频 token）来实现。结果是更低的 PPL + 正常的多样性 + 更平淡无味的文本。

2. **PPL 在 DLM 和 AR 模型之间不可比**。Feng et al. (arXiv 2502.09622) 的理论分析表明 MDM 和 AR 的 PPL 有不同的含义。Sahoo et al. (arXiv 2602.15014, 「Scaling Beyond Masked Diffusion Language Models」) 也发现「PPL 在同一扩散族内有参考价值，但跨范式比较时意义有限」。这意味着用 AR 模型（如 GPT-2）评估 DLM 生成的文本的 PPL 本身就是有问题的——你在用一个范式的标准衡量另一个范式的输出。

3. **PPL 无法反映推理能力**。本项目的目标是顶会论文（NeurIPS/ICML/ICLR），而 2025-2026 年的评审标准越来越偏向 benchmark 评估而非 PPL。Zhong et al. (arXiv 2601.15593) 在 58 个 benchmark 上的评估清楚表明：PPL 改善和 benchmark 得分之间**没有一致的正相关**。一个 PPL 更低的 DLM 可能在推理任务上反而更差。

4. **低 PPL 文本可以是退化的**。研究表明退化文本可以表现出异常低的 PPL 值（arXiv 2507.01844），因为模型可以通过输出高频、安全的 token 组合来降低 perplexity，而这些组合可能语义上空洞。多样性指标（如 distinct-n）可以捕捉重复但无法捕捉这种「安全但无意义」的退化。

### 反面证据

- **Spec 自己的教训已经证实了这一点**：0.6B 模型的 PPL "提升"被证明是重复伪影。但更深层的问题是，即使在 Dream-7B 上 PPL 没有退化问题，TCR 的 mean PPL 改善也没有反映在 median 上——说明 PPL 改善来自少数极端样本，大多数样本没有变化。
- **Long-context PPL 的不可靠性**（OpenReview: "What is Wrong with Perplexity for Long-context Language Modeling?"）：PPL 通过对所有 token 平均来忽略关键 token，这个问题在 DLM 生成的文本中可能更严重——因为 DLM 的去噪过程中「关键 token」和「填充 token」的区分比 AR 模型更不明确。

### 替代研究方向

**方向 C：以 Benchmark 驱动的评估为核心，PPL 仅作为早期 sanity check**

与其试图修复 PPL 作为指标的缺陷，不如彻底转向 benchmark 驱动的评估。具体建议：

1. **选用快速 benchmark 做迭代评估**：GSM8K（数学推理，~1300 题，Dream-7B 单 GPU ~2 小时可完成）、HumanEval（代码生成，164 题，~30 分钟）、MMLU-Pro 子集（知识评估，可抽样 500 题，~1 小时）。
2. **PPL 降级为 sanity check**：仅在实验初期用 PPL 确认方法没有严重退化，之后完全转向 benchmark。
3. **引入任务特定的推理质量评估**：比如在 GSM8K 上不仅看最终准确率，还看中间推理步骤的一致性——这直接评估 DLM 的「多步推理」能力。

这个方向的关键优势：与 2025-2026 年顶会的评审标准一致（Zhong et al. 在 58 个 benchmark 上评估的做法已经成为 DLM 评估的标杆），且能直接回答「推理时计算扩展是否提升了 DLM 的推理能力」这个核心问题。

- 计算成本：Dream-7B 在 GSM8K + HumanEval + MMLU-Pro 子集上，4x GPU 约需 4-6 小时
- 成功概率：本方向不是「研究方向」而是「评估范式转变」，应作为所有其他方向的基础设施

---

## 综合提案：从 TTT 幻觉到扩散原生的推理时计算扩展

### 核心论点

**TTT 是为自回归模型设计的解决方案，强行移植到 DLM 是方向性错误**。DLM 需要的不是 AR 世界的工具，而是利用自身独特结构（并行去噪、非因果注意力、多步轨迹）的原生方案。

### 建议的研究框架

**DLM-Native Inference Scaling: 利用扩散轨迹结构实现自适应计算分配**

核心思想：不是均匀增加所有位置的去噪步数，而是根据轨迹信号自适应地分配计算资源。

1. **Trajectory Entropy Profiling**：在标准去噪过程中，计算每个位置在跨步骤之间的 logit 变化熵。高熵位置 = 模型不确定 = 需要更多计算。
2. **Selective Re-diffusion**：仅对高熵位置进行额外去噪步骤（类似 TCR 但有理论依据）。关键区别：重遮蔽的目标由信息论指标（跨步骤 KL 散度）驱动，而非启发式置信度。
3. **Dilated Scheduling 集成**：利用 Luxembourg et al. (arXiv 2506.19037) 的 DUS 方法，将高熵位置组织为非相邻的 dilated groups 并行处理，保留 DLM 的速度优势。

### 与现有工作的关系
- 比 ReMask-Retry 更有原则性（信息论驱动 vs 置信度启发式）
- 比 TCR 更通用（不限于轨迹一致性，可推广到任意轨迹信号）
- 比 TTT 更原生（不引入外部模块，不破坏并行性）
- 与 DUS (arXiv 2506.19037)、TReASURe (arXiv 2509.23146)、DiSPO (arXiv 2602.06462) 等最新工作形成互补

### 风险评估

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| 熵计算额外开销 | 中 | logit 缓存 + batch 计算，overhead < 10% |
| 高熵位置可能是噪声而非信号 | 高 | 需要实验验证熵阈值和去噪效果的相关性 |
| 与 DUS 方法可能存在冲突 | 低 | DUS 解决「哪些位置一起去噪」，我们解决「哪些位置需要更多步骤」，正交 |
| PPL 改善但 benchmark 不变 | 中 | 从一开始就用 benchmark 评估，不依赖 PPL |

### 计算预算估计

- Dream-7B 模型：4x GPU (A100)
- 每次实验（1000 样本 × GSM8K）：约 2-3 小时
- 完整方案（含 ablation + baseline 对比）：约 1 周
- 成功概率：40%（方法有理论基础但实验验证充满不确定性——这是诚实估计）

---

## 引用来源

1. Zhong et al. (2026). "Parallelism and Generation Order in Masked Diffusion Language Models: Limits Today, Potential Tomorrow." arXiv:2601.15593
2. Ye et al. (2026). "Relaxing Positional Alignment in Masked Diffusion Language Models." arXiv:2601.22947
3. Gema et al. (2025). "Inverse Scaling in Test-Time Compute." arXiv:2507.14417
4. Feng et al. (2025). "Theoretical Benefit and Limitation of Diffusion Language Model." arXiv:2502.09622
5. Zheng et al. (2024). "Masked Diffusion Models Are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling." arXiv:2409.02908
6. Sahoo et al. (2026). "Scaling Beyond Masked Diffusion Language Models." arXiv:2602.15014
7. Luxembourg et al. (2025). "Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models." arXiv:2506.19037
8. Yu et al. (2025). "TReASURe: Tree Reward-Aligned Search in Masked Diffusion Language Models." arXiv:2509.23146
9. Oba et al. (2026). "DiSPO: Diffusion-State Policy Optimization for Masked Diffusion Language Models." arXiv:2602.06462
10. Huang et al. (2026). "Tuning the Implicit Regularizer of Masked Diffusion Language Models." arXiv:2601.22450
11. Tandon et al. (2025). "End-to-End Test-Time Training for Long Context." arXiv:2512.23675
12. Zhang et al. (2025). "Test-Time Training Done Right." arXiv:2505.23884
13. Hu et al. (2025). "Test-Time Learning for Large Language Models." arXiv:2505.20633
14. Sheng et al. (2025). "The Illusion of Progress? A Critical Look at Test-Time Adaptation for VLMs." arXiv:2506.24000
15. Chen et al. (2025). "A Survey of Scaling in Large Language Model Reasoning." arXiv:2504.02181
16. Balachandran et al. (2025). "Inference-time Scaling for Complex Tasks: Where We Stand and What Lies Ahead." arXiv:2504.00294
17. Snell et al. (2024). "Scaling LLM Test-Time Compute Optimally." arXiv:2408.03314
18. Islah et al. (2025). "Exploring The Effectiveness of Test Time Learning In LLMs for Long Contexts." OpenReview
19. Nahin et al. (2025). "Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling." arXiv:2510.08592
