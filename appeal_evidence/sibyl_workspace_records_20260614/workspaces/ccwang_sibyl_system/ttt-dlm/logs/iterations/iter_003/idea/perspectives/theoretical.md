# 理论研究者提案：DLM 推理时记忆与计算扩展的数学基础

## 总体判断

当前项目已完成 18 轮迭代，核心发现是：**简单的 remasking/retry 策略在 DLM 上效果有限，而项目 spec 明确要求探索将 TTT（Test-Time Training）的结构与思想引入 DLM**。从理论角度看，这一方向具有深刻的数学动机：DLM 的迭代去噪过程与 TTT 的推理时权重适应本质上都是"在推理时通过迭代优化来改善输出"，但它们作用于不同的空间——DLM 优化 token 空间中的离散状态，TTT 优化模型参数空间中的连续权重。**将两者统一的理论框架不仅能解释前 18 轮的负面结果，更能指出突破方向。**

本提案从三个不同角度（改进现有方法、跨域迁移、提出新方法）提出理论驱动的研究方案。

---

## 背景：TTT 与 DLM 的数学对偶性

### TTT 的核心形式化

TTT（Sun et al., 2024; Behrouz et al., 2025 "Titans"）的本质是在推理时通过自监督目标更新模型的一部分权重（称为"快权重" $W_{\text{fast}}$）：

$$W_{\text{fast}}^{(t+1)} = W_{\text{fast}}^{(t)} - \eta \nabla_{W} \mathcal{L}_{\text{self}}(x_{\leq t}; W_{\text{fast}}^{(t)})$$

其中 $\mathcal{L}_{\text{self}}$ 是自监督损失（如掩码重构）。Liu et al.（2026, "TTT with KV Binding Is Secretly Linear Attention"）证明了一个关键理论结果：**广泛的 TTT 架构族在数学上等价于一种带学习核的线性注意力算子**。具体地，TTT 的快权重更新可以写为：

$$W_{\text{fast}}^{(T)} = \sum_{t=1}^{T} \alpha_t v_t k_t^\top$$

这与线性注意力 $\text{Attn}(q, K, V) = q \sum_t v_t k_t^\top$ 的形式完全对应，其中 $\alpha_t$ 是学习到的权重。

### DLM 去噪的形式化

MDM（MDLM, LLaDA, Dream）的去噪过程是：在时间步 $t$ ，给定部分遮蔽序列 $x_t$，模型预测每个被遮蔽位置的 token 分布 $p_\theta(x_0^i | x_t)$。去噪过程的本质是**以条件重构为目标的迭代优化**：

$$x_{t-1} = \text{Unmask}(x_t, p_\theta(\cdot | x_t), \text{schedule}(t))$$

### 对偶性

| 维度 | TTT | DLM 去噪 |
|------|-----|----------|
| 优化空间 | 参数空间 $W \in \mathbb{R}^{d \times d}$ | Token 空间 $x \in \mathcal{V}^n$ |
| 优化目标 | $\min_W \mathcal{L}_{\text{self}}(x; W)$ | $\min_{x_0} D_{\text{KL}}(q(x_0|x_t) \| p_\theta(x_0|x_t))$ |
| 迭代机制 | 梯度下降更新权重 | 逐步 unmask 更新 token |
| 信息积累 | 压缩上下文到快权重 | 从噪声恢复信号 |
| 记忆形式 | 权重矩阵 $W_{\text{fast}}$（连续） | 已揭示 token 集合（离散） |

**关键洞察**：DLM 的去噪过程缺少 TTT 的核心能力——**跨步骤的连续状态记忆**。每个去噪步骤中，模型对所有 token 执行独立的前向传播，不保留上一步的"推理状态"。这解释了为什么简单 remasking 无效：remask 后重新预测等价于从头推理，丢失了之前步骤积累的上下文理解。

---

## 角度一：改进现有方法——DLM 去噪作为隐式 TTT 的收敛性分析

### 1.1 核心理论问题

DLM 的多步去噪过程可以被视为一种**隐式的 test-time 优化**：模型在推理时通过 $T$ 步迭代逐步改善输出。然而，与 TTT 不同，DLM 的"优化"发生在 token 空间而非参数空间，且不更新任何权重。

**命题 1（DLM 去噪的不动点表征）**：设 $f_\theta: \mathcal{V}^n \to \Delta(\mathcal{V})^n$ 为 MDM 的去噪网络。定义完全去噪映射 $F_\theta(x) = \arg\max f_\theta(x)$（对每个位置取最大概率 token）。若存在 $x^*$ 使得 $F_\theta(x^*) = x^*$，则 $x^*$ 是去噪过程的**不动点**。

**理论预测**：
- 当模型能力充足时（7B-8B），不动点存在且通常非唯一（多模态分布），去噪过程在不同吸引域的不动点间"横向漂移"——这解释了 Dream-7B 上 remasking 无效的实验结果
- 当模型能力不足时（0.6B），不动点退化为低多样性序列（高频 token 重复），这解释了小模型的模式坍缩
- 累积噪声（LLaDA-8B）对应于去噪轨迹偏离不动点的吸引域

Chen, Cong & Li（2025, "Optimal Inference Schedules for Masked Diffusion Models"）已给出了 MDM 采样误差的精确刻画，证明了与单变量函数逼近理论的优美联系，并给出了基于分布**总相关性**（total correlation）和**对偶总相关性**（dual total correlation）的采样步数下界。具体地，他们证明当 $TC(X) = o(n)$ 时，MDM 可以在 $O(\log n)$ 步内完成采样而不产生可见的性能损失。

### 1.2 理论贡献：Remasking 的收敛率界

**定理草案（非正式）**：设 MDM 的去噪网络满足 Lipschitz 条件 $\|f_\theta(x) - f_\theta(x')\|_1 \leq L \|x - x'\|_H$（Hamming 距离），remasking 比例为 $\gamma \in (0,1)$。则 $k$ 步 remasking 后：

$$\mathbb{E}[d_H(x_k, x^*)] \leq (1 - \gamma(1 - L))^k \cdot d_H(x_0, x^*)$$

当 $L < 1$（去噪网络是压缩映射）时收敛；当 $L \geq 1$（非压缩）时发散。

**关键预测**：
- 小模型 $L \approx 1$（softmax 分布尖锐，接近不压缩） → 收敛到退化不动点
- 大模型 $L < 1$ 但接近 1 → 收敛缓慢，需要大量 remasking 步数才能改善
- **remasking 的有效性根本受限于去噪网络的 Lipschitz 常数**，这是一个内在约束，不可通过采样策略设计绕过

这一理论可以与 IterRef（Lee et al., 2025, arXiv 2511.05562）的"去噪-加噪迭代精化"框架统一：IterRef 证明了在离散扩散中，通过交替的加噪-去噪步骤可以实现收敛到真实分布的理论保证。

### 1.3 实验设计

- **验证 Lipschitz 常数**：在 MDLM/Dream/LLaDA 上经验估计去噪网络的局部 Lipschitz 常数，通过随机扰动输入并测量输出变化
- **收敛率测量**：记录 remasking 迭代过程中与参考不动点的 Hamming 距离变化，验证指数收敛/发散预测
- **成功概率 P(success) ≈ 60%**：理论框架成立的前提是去噪网络的 Lipschitz 性可经验验证
- **计算成本**：每个模型约 2-4 GPU 小时（仅需前向传播，无训练）

---

## 角度二：跨域迁移——将 TTT 记忆层嵌入 DLM 去噪过程

### 2.1 数学动机

TTT 的核心价值是提供**跨序列的连续状态记忆**，而 DLM 去噪的每个时间步都是无状态的（给定 $x_t$，独立预测 $x_0$）。这意味着 DLM **在去噪的不同时间步之间不共享任何推理状态**。

**信息论论证**：设 DLM 在时间步 $t$ 的预测为 $\hat{x}_0^{(t)}$，去噪过程产生的中间状态序列为 $\{h_t^{(1)}, h_t^{(2)}, \ldots, h_t^{(T)}\}$（各层隐藏状态）。在标准 DLM 中：

$$I(h_t^{(l)}; h_{t-1}^{(l)}) = 0 \quad \forall l, t$$

即不同去噪步之间的隐藏状态互信息为零——每步完全从头计算。这是一种极大的信息浪费。

**TTT 记忆的信息增益**：若引入跨步骤的快权重 $W_{\text{fast}}^{(t)}$，且该快权重通过自监督目标在去噪过程中在线更新，则：

$$I(W_{\text{fast}}^{(t)}; x_0) \geq I(W_{\text{fast}}^{(t-1)}; x_0) + \Delta I_t$$

其中 $\Delta I_t > 0$ 是第 $t$ 步去噪揭示的新 token 带来的信息增益。快权重**单调积累关于目标序列的信息**，而不像标准 DLM 那样每步丢弃。

### 2.2 方案：Denoising-Time Memory (DTM)

**核心思想**：在 DLM 的 Transformer 层中插入轻量级 TTT 层（基于 Liu et al., 2026 的线性注意力等价形式），使去噪过程获得跨步骤记忆能力。

**形式化定义**：

标准 DLM 的第 $l$ 层在时间步 $t$：
$$h_t^{(l)} = \text{TransformerLayer}^{(l)}(h_t^{(l-1)})$$

引入 DTM 后：
$$\tilde{h}_t^{(l)} = \text{TransformerLayer}^{(l)}(h_t^{(l-1)}) + \alpha \cdot \text{DTM}^{(l)}(h_t^{(l-1)}; W_{\text{fast},t}^{(l)})$$

其中 DTM 层定义为：
$$\text{DTM}(h; W_{\text{fast}}) = h \cdot W_{\text{fast}}$$
$$W_{\text{fast},t}^{(l)} = W_{\text{fast},t-1}^{(l)} - \eta \nabla_W \mathcal{L}_{\text{denoise}}(\hat{x}_0^{(t-1)}, x_t; W)$$

自监督目标 $\mathcal{L}_{\text{denoise}}$ 是当前去噪步的预测一致性：已在步骤 $t-1$ 揭示的 token 应在步骤 $t$ 被正确预测。

**为什么这比 remasking 更有力**：

1. **信息积累而非信息丢弃**：remasking 通过重新遮蔽 token 来"重试"，但丢失了上一步的推理结果。DTM 将推理结果压缩到快权重中持续利用
2. **连续优化而非离散搜索**：remasking 在离散 token 空间搜索，DTM 在连续参数空间优化，后者有更好的收敛保证
3. **DLM 天然适合**：DLM 已有多步迭代结构，TTT 层只需在现有步骤间传递快权重，无需额外迭代

**与 Titans（Behrouz et al., 2025）的关系**：Titans 将神经记忆模块引入自回归 Transformer，记忆通过梯度下降在 token 序列上在线更新。DTM 将类似的记忆机制引入扩散 Transformer，但记忆在**去噪时间步**上更新而非在**序列位置**上更新。这是一个本质区别：

- Titans: $W_{\text{fast}}^{(i+1)} = W_{\text{fast}}^{(i)} - \eta \nabla \mathcal{L}(x_i)$（沿序列位置迭代）
- DTM: $W_{\text{fast}}^{(t+1)} = W_{\text{fast}}^{(t)} - \eta \nabla \mathcal{L}(x_t)$（沿去噪时间步迭代）

**与 ATLAS（Behrouz et al., 2025）的关系**：ATLAS 通过优化"当前+历史 token"来克服在线学习的"只看最后一个"局限。DTM 可以类似地优化"当前去噪状态+历史去噪状态"，提升记忆容量。

**与 LaCT（Zhang et al., 2025, "TTT Done Right"）的关系**：LaCT 证明大 chunk 更新（2K-1M tokens）比小 minibatch 更高效。在 DLM 中，去噪步骤的输入天然是完整序列（所有 token，部分被遮蔽），因此 DTM 的更新天然是"大 chunk"的，硬件利用率优于传统 TTT。

### 2.3 理论保证

**命题 2（DTM 的收敛性）**：设 DTM 的自监督损失 $\mathcal{L}_{\text{denoise}}$ 关于 $W_{\text{fast}}$ 是 $\mu$-强凸的（线性层 + L2 正则化可保证），学习率 $\eta < 2/L_W$（$L_W$ 为损失的 Lipschitz 梯度常数）。则经过 $T$ 步去噪后：

$$\|W_{\text{fast}}^{(T)} - W_{\text{fast}}^*\|_F \leq (1 - \eta\mu)^T \|W_{\text{fast}}^{(0)} - W_{\text{fast}}^*\|_F$$

其中 $W_{\text{fast}}^*$ 是最优快权重。这提供了**指数收敛保证**，而标准 remasking 缺乏此类保证。

**命题 3（信息论下界）**：DTM 在 $T$ 步去噪后积累的信息量满足：

$$I(W_{\text{fast}}^{(T)}; x_0) \geq \sum_{t=1}^{T} I(\hat{x}_0^{(t)} \setminus \hat{x}_0^{(t-1)}; x_0 | W_{\text{fast}}^{(t-1)})$$

即每步新揭示的 token 贡献正信息增益，且该信息被快权重持久保存。

### 2.4 实现路径

**Training-free 方案（优先）**：
- 使用预训练 DLM（Dream-7B / LLaDA-8B）的中间层隐藏状态作为 key-value 对
- DTM 层初始化为零矩阵（残差连接保证不影响初始性能）
- 快权重通过去噪过程中的在线梯度下降更新，无需额外训练
- 自监督目标：已揭示 token 的重构损失

**LoRA 微调方案**：
- 在预训练 DLM 中插入 DTM 层后，用 LoRA 在少量数据上微调
- 微调目标：对齐 DTM 增强后的预测与标准去噪目标
- 预计 1-2 个 epoch 即可收敛（DTM 层参数量仅为主模型的 0.5-2%）

### 2.5 实验设计

- **Benchmark**: Countdown（规划推理，Dream 已有基线 16.0）、GSM8K（数学推理）、MBPP（代码生成）
- **基线对比**: 标准去噪 vs ReMDM-conf vs CORE vs DTM（training-free）vs DTM（LoRA）
- **成功概率 P(success) ≈ 45%**：核心风险在于 training-free DTM 可能因快权重初始化不当导致去噪轨迹偏移。LoRA 微调方案更稳健但需要训练
- **计算成本**: Training-free 方案每模型约 4-8 GPU 小时；LoRA 方案每模型约 16-24 GPU 小时

---

## 角度三：新方法——DLM 去噪过程的变分 TTT 统一框架

### 3.1 数学基础：去噪即 test-time 变分推理

提出将 DLM 去噪和 TTT 统一在变分推理框架下：

**统一目标**：给定观测 $y$（prompt + 部分生成），推理隐变量 $z$（目标序列 + 推理状态）：

$$\max_{q \in \mathcal{Q}} \mathbb{E}_{z \sim q}[\log p(y | z)] - D_{\text{KL}}(q(z) \| p(z))$$

**DLM 的去噪对应于**：$z = x_0$（目标 token 序列），$q$ 由去噪过程隐式定义，$\mathcal{Q}$ 限制在可由去噪网络 $f_\theta$ 表达的分布族内。

**TTT 对应于**：$z = (x_0, W_{\text{fast}})$（目标序列 + 快权重），$q$ 由梯度下降在参数空间定义的轨迹隐式定义。

**统一后的 Variational Denoising-Time Training (VDTT)**：
$$z = (x_0, W_{\text{fast}})$$
$$q(z) = q(x_0 | W_{\text{fast}}) \cdot q(W_{\text{fast}})$$

在每个去噪步 $t$：
1. **E-step（token 更新）**：固定 $W_{\text{fast}}^{(t)}$，用增强后的网络 $f_{\theta, W_{\text{fast}}^{(t)}}$ 去噪 $x_t \to x_{t-1}$
2. **M-step（记忆更新）**：固定 $x_{t-1}$，更新 $W_{\text{fast}}^{(t+1)} = W_{\text{fast}}^{(t)} - \eta \nabla_W \mathcal{L}(x_{t-1}; W_{\text{fast}}^{(t)})$

这是一种**EM 算法**：E-step 在 token 空间推理，M-step 在参数空间学习。交替执行保证 ELBO 单调递增。

### 3.2 理论分析

**定理草案 1（VDTT 的 ELBO 单调性）**：设 $\mathcal{L}_{\text{VDTT}}(x_t, W_{\text{fast}}^{(t)})$ 为 VDTT 在步骤 $t$ 的变分下界。在温和的正则性条件下（$\mathcal{L}_{\text{denoise}}$ 强凸，$f_\theta$ 连续）：

$$\mathcal{L}_{\text{VDTT}}(x_{t-1}, W_{\text{fast}}^{(t+1)}) \geq \mathcal{L}_{\text{VDTT}}(x_t, W_{\text{fast}}^{(t)})$$

即每个去噪-更新步骤都改善变分下界。

**定理草案 2（与 Equilibrium Transformer 的联系）**：VDTT 是 Equilibrium Transformer（Jafari & Anbarjafari, 2025, arXiv 2511.21882）的一种特例。EqT 通过在潜在空间中最小化能量函数来迭代精化表示直到达到自一致平衡——VDTT 在"token 空间 + 参数空间"的联合空间中做同样的事。这为 VDTT 提供了**MAP 推理**的理论解释：

$$\text{VDTT converges to } (x_0^*, W_{\text{fast}}^*) = \arg\max_{x_0, W} p(y | x_0, W) p(x_0) p(W)$$

**定理草案 3（DLM + VDTT 的计算复杂性优势）**：基于 Svete & Sabharwal（2025, arXiv 2510.13117）证明的 MDM 与循环 Transformer 等价性，以及 Jiang et al.（2025, arXiv 2512.25014）证明的 DLM + remasking 是最优并行采样器：

VDTT 将 DLM 的计算通用性与 TTT 的记忆能力结合，可以在 $O(T \cdot n)$ 时间内（$T$ 去噪步，$n$ 序列长度）解决需要 $O(T \cdot n^2)$ 的全注意力 Transformer 才能解决的问题——因为快权重提供了 $O(1)$ 的跨步骤信息检索，而标准 DLM 需要在每步重新计算全部注意力。

### 3.3 与现有理论的关系

| 现有工作 | 与 VDTT 的关系 |
|---------|---------------|
| ReMDM（自定义后向过程） | VDTT 的 E-step 是 ReMDM 的推广（增加了 $W_{\text{fast}}$ 条件） |
| PG-DLM（粒子 Gibbs） | VDTT 可视为单粒子特例，快权重替代了多粒子的功能 |
| ETS（能量引导） | VDTT 的 M-step 隐式定义了能量函数 $E(x) = \mathcal{L}_{\text{denoise}}(x; W_{\text{fast}})$ |
| IterRef（迭代精化） | IterRef 的"加噪-去噪"循环是 VDTT E-step 的特例（无 M-step） |
| RemeDi（双流架构） | VDTT 不修改架构，通过快权重隐式实现类似 UPS 的功能 |
| Miras（Behrouz et al., 2025） | VDTT 的记忆模块设计可以借鉴 Miras 的"注意力偏置 + 遗忘门 + 记忆学习算法"四维设计空间 |

### 3.4 实验设计

**阶段 1：概念验证（2 周）**
- 在 MDLM（~170M）上实现 VDTT，验证 ELBO 单调性
- Benchmark: Countdown（简单结构化推理）
- 预期：即使小模型也能通过记忆积累改善多步推理

**阶段 2：规模化（2 周）**
- 在 Dream-7B 上实现 training-free VDTT
- Benchmark: GSM8K, MATH500, Countdown
- 与 CORE、ReMDM-conf、RCR 对比

**阶段 3：分析与论文（1 周）**
- 消融实验：E-step only（标准去噪）vs M-step only（纯 TTT）vs VDTT
- 信息论分析：测量 $I(W_{\text{fast}}^{(t)}; x_0)$ 随去噪步的增长曲线
- 与 IterRef 的收敛性对比

**成功概率 P(success) ≈ 40%**：这是最有野心的方案，理论框架新颖但实现复杂。核心风险：(1) E-step 和 M-step 的交互可能导致不稳定；(2) 快权重更新的梯度计算增加 30-50% 推理开销

---

## 失败模式分析

### 共性风险

1. **快权重初始化问题**：零初始化的快权重在前几步可能无法提供有用信息，反而引入噪声。缓解：前 $k$ 步使用标准去噪（warm-up），后续步骤启用 DTM/VDTT
2. **梯度消失/爆炸**：在去噪时间步上的反向传播可能不稳定。缓解：使用 Liu et al. 的线性注意力等价形式，避免显式梯度下降
3. **分布偏移**：引入快权重改变了去噪网络的行为，可能导致 OOD 预测。缓解：小学习率 $\eta$ + L2 正则化限制快权重偏离幅度

### 方案特定风险

| 方案 | 主要风险 | 缓解策略 |
|------|---------|---------|
| 角度一（收敛性分析） | Lipschitz 常数不可经验估计 | 使用局部随机采样近似 |
| 角度二（DTM 嵌入） | Training-free 方案效果不足 | 降级到 LoRA 微调 |
| 角度三（VDTT） | EM 交替优化不收敛 | 减小 M-step 学习率，增大 E-step 步数 |

---

## 论文贡献定位

### 最佳情况（3 个角度均成功）
- **理论**：DLM 去噪与 TTT 的变分统一框架（VDTT），含 ELBO 单调性证明
- **方法**：Denoising-Time Memory（DTM），首个将 TTT 记忆引入 DLM 去噪的方法
- **实验**：在 Dream-7B/LLaDA-8B 上的推理 benchmark 验证
- **目标会议**：NeurIPS/ICML 正文

### 最可能情况（角度二成功，角度三部分成功）
- **理论**：DTM 的信息论分析 + 收敛性保证
- **方法**：DTM 的 training-free 和 LoRA 两种实现
- **实验**：Countdown + GSM8K 上的验证
- **目标会议**：NeurIPS/ICML 正文

### 最差情况（仅角度一成功）
- **理论**：DLM 去噪的 Lipschitz 收敛性分析 + 前 18 轮结果的统一解释
- **实验**：经验验证收敛率预测
- **目标会议**：ICLR 或 NeurIPS 研讨会

---

## 推荐执行顺序

1. **角度二（DTM）优先**：最直接回应 spec 的 TTT 集成需求，且有清晰的实现路径
2. **角度一（收敛性分析）并行推进**：理论分析不需要 GPU，可与实验并行
3. **角度三（VDTT）作为升级路径**：若 DTM 的 training-free 方案有效，则 VDTT 提供更优雅的理论包装

**总预估工时**：4-5 周（1 周理论 + 2 周实现 + 1 周实验 + 1 周论文）
**总 GPU 成本**：约 100-200 GPU 小时（4x GPU 约 25-50 小时墙钟时间）

---

## 关键参考文献

### TTT 与记忆架构
- Sun et al. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv 2407.04620
- Behrouz et al. (2025). Titans: Learning to Memorize at Test Time. arXiv 2501.00663
- Behrouz et al. (2025). ATLAS: Learning to Optimally Memorize the Context at Test Time. arXiv 2505.23735
- Liu et al. (2026). Test-Time Training with KV Binding Is Secretly Linear Attention. arXiv 2602.21204
- Zhang et al. (2025). Test-Time Training Done Right (LaCT). arXiv 2505.23884
- Li et al. (2025). TNT: Improving Chunkwise Training for Test-Time Memorization. arXiv 2511.07343
- Behrouz et al. (2025). Miras: It's All Connected (Attentional Bias, Retention, Online Optimization). arXiv 2504.13173

### DLM 理论
- Chen, Cong & Li (2025). Optimal Inference Schedules for Masked Diffusion Models. arXiv 2511.04647
- Jeon et al. (2025). Information-Theoretic Discrete Diffusion. arXiv 2510.24088
- Svete & Sabharwal (2025). On the Reasoning Abilities of MDLMs. arXiv 2510.13117
- Jiang et al. (2025). DLMs are Provably Optimal Parallel Samplers. arXiv 2512.25014
- Li & Cai (2026). Generation Order and Parallel Decoding in MDMs. arXiv 2602.00286

### DLM 推理时扩展
- Nisonoff et al. (2025). ReMDM: Remasking Discrete Diffusion Models. arXiv 2503.00307
- Lee et al. (2025). IterRef: Effective Test-Time Scaling via Iterative Refinement. arXiv 2511.05562
- Liu et al. (2025). RemeDi: Self-Reflective Remasking. arXiv 2509.23653
- CORE (2026). Context-Robust Remasking. arXiv 2602.04096
- Self-Rewarding SMC (2026). arXiv 2602.01849

### 统一框架
- Jafari & Anbarjafari (2025). Closed-Loop / Equilibrium Transformers. arXiv 2511.21882
- Latent Thought Models (2025). arXiv 2502.01567
