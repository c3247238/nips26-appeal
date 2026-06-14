# Theoretical Perspective: 遮蔽扩散语言模型的推理时计算扩展

## 研究定位

从数学与信息论角度审视遮蔽扩散语言模型（MDLMs）中 TTT（Test-Time Training）与 remasking 的深层联系，提出具有理论保证的推理时计算扩展方法。核心观察：**TTT 层的隐状态自监督更新与 MDLMs 的迭代去噪过程在数学结构上存在深刻同构**——两者都是在推理时通过迭代优化来改善预测质量。

---

## 角度一：TTT-Denoiser — 将 TTT 的自监督更新融入 MDLMs 的去噪过程

### 数学动机

TTT 层（Sun et al., 2024, arXiv:2407.04620）的核心思想是将隐状态本身视为一个机器学习模型，更新规则是自监督学习的一步梯度下降。设 TTT 层的隐状态为参数化模型 $W_t$，则在处理序列 token $x_t$ 时：

$$W_{t+1} = W_t - \eta \nabla_{W_t} \mathcal{L}_{\text{SSL}}(W_t; x_t)$$

其中 $\mathcal{L}_{\text{SSL}}$ 是自监督损失（如重建损失）。

MDLMs 的去噪过程可以类似地形式化：在扩散步 $s$ 处，模型 $f_\theta$ 对遮蔽序列 $\mathbf{x}^{(s)}$ 的预测为 $\hat{\mathbf{x}} = f_\theta(\mathbf{x}^{(s)})$，然后根据策略（置信度、随机等）选择性地揭示/重遮蔽 token。

**关键洞察**：如果我们在每个去噪步 $s$ 中，用当前已揭示的 token 作为"监督信号"对模型的一个轻量子网络执行自监督更新，那么 MDLMs 的去噪过程就变成了一个**在推理时动态积累上下文记忆的过程**。这与 Titans（Behrouz & Zhong, 2025, arXiv:2501.00663）的长期记忆模块和 MesaNet（von Oswald et al., 2025, arXiv:2506.05233）的最优测试时训练思想一脉相承。

### 具体方案

在预训练好的 MDLMs（如 LLaDA-8B 或 Dream-7B）的 Transformer 层之间插入轻量 TTT 模块：

1. **TTT-Linear 插入层**：在每个去噪步中，TTT-Linear 层的 fast weight $W^{(s)}$ 通过对当前已揭示 token 的自监督学习进行更新：
   $$W^{(s+1)} = W^{(s)} - \eta \nabla_W \sum_{i \in \text{unmasked}} \ell(W^{(s)} \cdot h_i, h_i')$$
   其中 $h_i$ 是第 $i$ 个 token 的隐状态表示，$h_i'$ 是其目标表示。

2. **跨步记忆传递**：与标准 MDLMs 不同（每步独立前向传播），TTT 模块的 fast weight 跨去噪步持久化，形成"去噪轨迹记忆"——模型逐步积累对序列的理解。

3. **理论联系**：从 MCMC 角度看，每个去噪步相当于 Gibbs 采样的一步（PG-DLM, arXiv:2507.08390），而 TTT 更新相当于在采样过程中动态调整采样器参数。这可以被形式化为一种**自适应 MCMC**（Adaptive MCMC），其理论收敛性已有成熟框架（Roberts & Rosenthal, 2007）。

### 理论保证

- **收敛性**：基于 Zhang et al.（2026, arXiv:2602.00286）的信息论框架，remasking 采样器的分布偏差 $D_{\mathrm{KL}}(p_{\text{remask}} \| p_{\text{data}})$ 受限于模型预测误差和 remasking 步数。TTT 更新通过减小模型预测误差（在当前上下文中），直接降低这个上界。
- **互信息界**：Chen et al.（2025, arXiv:2505.21400）证明了扩散语言模型的收敛速率紧密依赖于迭代步数 $T$ 和目标分布的互信息 $I(X_i; X_{-i})$。TTT 更新可以解释为在每步动态降低有效互信息（通过将已揭示 token 的信息编码进 fast weight），从而加速收敛。
- **误差界**：Lavenant & Zanella（2025, arXiv:2510.25544）给出了 MDLMs 因式化近似的误差界，与序列长度无关，仅依赖于每步平均生成的 token 数。TTT 更新通过改善因式化近似的质量（条件独立假设在上下文适应后更准确），可进一步收紧此界。

### 计算成本估算

- **额外开销**：每个去噪步增加一次 TTT-Linear 前向+反向传播。TTT-Linear 的参数量为 $O(d^2)$（$d$ 为隐状态维度），梯度计算为 $O(n \cdot d^2)$（$n$ 为序列长度）。对于 LLaDA-8B（$d=4096$），单步 TTT 更新约增加 5-10% 计算量。
- **总计算量**：假设 $T$ 步去噪，TTT 增加的总计算量约为 $T \times 0.05 \times C_{\text{forward}}$，即约 $5T\%$ 的额外开销。
- **成功概率**：60-70%。主要风险在于 TTT 更新可能在少量 token 上过拟合，导致分布漂移。需要正则化（如 dropout、权重衰减）来控制。

### 失败模式

1. **过拟合风险**：早期去噪步中已揭示 token 极少，TTT 更新的监督信号不足，可能导致 fast weight 退化。
2. **训练-推理不匹配**：预训练的 MDLMs 未见过 TTT 模块，插入后可能破坏已学习的表示。需要少量适配训练（LoRA-style）。
3. **速度开销不成比例**：若 TTT 更新的质量收益低于其计算开销，方法不可行。需对比 "多跑几步去噪" vs "加 TTT" 的效率。

---

## 角度二：信息论最优 Remasking — 通过互信息估计指导 token 重遮蔽

### 数学动机

现有 remasking 策略主要基于启发式规则：
- **ReMDM**（arXiv:2503.00307）：cap/rescale/conf/loop，从自定义后向过程推导但无最优性保证
- **CORE**（arXiv:2602.04096）：上下文扰动探测，计算开销大
- **MDPO-RCR**（arXiv:2508.13148）：运行时置信度 remasking，理论基础有限
- **LookUM**（arXiv:2511.05563）：不确定性验证，但路径选择缺少信息论最优性

Zhang et al.（2026, arXiv:2602.00286）的信息论分析揭示了关键限制：**任何 remasking 启发式都无法保证分布正确性**，且验证方法代价呈指数。但该分析也指出了一个未被利用的方向：**条件互信息 $I(X_i; X_j \mid \mathbf{X}_{\text{revealed}})$ 可以指导最优 remasking 决策**。

### 具体方案：MI-Remask（Mutual Information-guided Remasking）

1. **条件互信息估计**：利用模型的 softmax 预测分布 $p_\theta(x_i \mid \mathbf{x}^{(s)})$ 近似计算相邻 token 间的条件互信息：
   $$\hat{I}(X_i; X_j \mid \mathbf{X}_{\text{rev}}) \approx H(X_i \mid \mathbf{X}_{\text{rev}}) - H(X_i \mid X_j, \mathbf{X}_{\text{rev}})$$
   其中 $H(X_i \mid \mathbf{X}_{\text{rev}}) = -\sum_v p_\theta(x_i=v \mid \mathbf{x}^{(s)}) \log p_\theta(x_i=v \mid \mathbf{x}^{(s)})$

2. **最优 remasking 准则**：优先重遮蔽那些**与已揭示 token 条件互信息最高但当前预测不确定的 token**。直觉：这些 token 最有可能从更多上下文中获益。形式化为：
   $$\text{remask\_score}(i) = H(X_i \mid \mathbf{X}_{\text{rev}}) \cdot \sum_{j \in \text{unmasked}} \hat{I}(X_i; X_j \mid \mathbf{X}_{\text{rev}})$$
   即高熵且高依赖性的 token 最需要 remasking。

3. **与 DUS（arXiv:2506.19037）的联系**：DUS 通过 dilated 分组来最小化联合熵增长上界。MI-Remask 可以看作 DUS 的信息论推广——不使用固定的 dilated 模式，而是根据实际的条件互信息动态分组。

4. **理论框架**：将 remasking 建模为**信息获取问题（Information Acquisition Problem）**。每步的 remasking 决策是在"探索（重遮蔽以获取更好预测）"和"利用（保留当前预测）"之间的权衡。这与主动学习（Active Learning）和 Bayesian Experimental Design 有精确对应。

### 理论保证

- **贝叶斯最优性**：在完美模型假设下（$p_\theta = p_{\text{data}}$），MI-Remask 的 remasking 决策是贝叶斯最优的——它最大化了每步 remasking 带来的期望信息增益。
- **亚最优界**：在模型不完美的情况下，可以证明 MI-Remask 的分布偏差上界为 $D_{\mathrm{KL}}(p_{\text{MI-remask}} \| p_{\text{data}}) \leq D_{\mathrm{KL}}(p_\theta \| p_{\text{data}}) + O(T^{-1})$，其中 $T$ 是 remasking 步数。这比无引导 remasking 的界更紧。
- **与 Anchored DLM 的联系**：Rout et al.（2025, arXiv:2505.18456）提出的 ADLM 使用信息论策略决定 token 遮蔽顺序（高信息 token 先被遮蔽/后被揭示）。MI-Remask 可以看作 ADLM 的推理时对偶——在推理时根据信息论准则决定 remasking 目标。

### 计算成本估算

- **互信息估计**：利用现有的 softmax 分布，不需要额外前向传播。主要开销在于计算 token 对之间的条件互信息矩阵 $O(n^2 \cdot V)$（$V$ 为词汇量），但可通过局部窗口近似降至 $O(n \cdot w \cdot V)$（$w$ 为窗口大小）。
- **实际开销**：相比标准 remasking（仅需比较置信度分数，$O(n)$），MI-Remask 增加约 $O(n \cdot w)$ 的计算量，可忽略不计。
- **成功概率**：50-60%。主要风险在于互信息的近似估计质量。当模型 $p_\theta$ 与真实分布差距较大时，互信息估计可能严重偏差。

### 失败模式

1. **互信息估计偏差**：模型 $p_\theta$ 的 softmax 分布可能无法准确反映真实的条件互信息，特别是在高度非局部依赖的情况下。
2. **计算开销**：对于长序列，即使局部近似，$O(n \cdot w \cdot V)$ 可能仍然显著。需要更高效的近似方法（如低秩近似、采样估计）。
3. **与因式化假设的冲突**：MI-Remask 试图捕捉 token 间依赖，但 MDLMs 的因式化预测本身假设条件独立。这两者之间的张力可能限制方法的有效性。

---

## 角度三：去噪-记忆统一架构 — MDLMs 的迭代去噪作为自适应 MCMC 与 TTT 的统一框架

### 数学动机

这一角度提出一个统一理论框架，将以下三个看似不同的范式连接起来：

1. **TTT 层的自监督更新**：$W_{t+1} = W_t - \eta \nabla \mathcal{L}_{\text{SSL}}(W_t; x_t)$
2. **MDLMs 的 remasking 去噪**：$\mathbf{x}^{(s+1)} = \text{remask}(\mathbf{x}^{(s)}, f_\theta(\mathbf{x}^{(s)}))$
3. **自适应 MCMC 采样**：$(\mathbf{x}^{(s+1)}, \lambda^{(s+1)}) = \text{kernel}(\mathbf{x}^{(s)}, \lambda^{(s)})$，其中 $\lambda$ 是采样器参数

**核心定理（待证明）**：MDLMs 的 remasking 去噪过程 + TTT 更新可以被形式化为一个**双层优化问题**：

- **外层**（去噪）：选择序列 $\mathbf{x}$ 最小化负对数似然 $-\log p_{\text{data}}(\mathbf{x})$
- **内层**（TTT 更新）：调整模型参数 $W$ 最小化自监督损失 $\mathcal{L}_{\text{SSL}}(W; \mathbf{x}_{\text{revealed}})$

这个双层结构与元学习（Meta-Learning）中的 MAML 框架有精确对应：外层是"任务"（生成高质量序列），内层是"适应"（根据当前上下文调整模型）。

### 与 Equilibrium Transformers 的联系

Jafari & Anbarjafari（2025, arXiv:2511.21882）提出的 Closed-Loop Transformers / Equilibrium Transformers (EqT) 将 TTT 和扩散语言模型统一为能量函数最小化的特例。他们证明 EqT 执行的是潜在能量模型中的近似 MAP 推理，并给出了线性收敛保证。

我们可以将此框架特化到 MDLMs：

$$\text{MDLMs + TTT} \equiv \text{交替最小化} \begin{cases} \mathbf{x}^{(s+1)} = \arg\min_{\mathbf{x}} E_\theta(\mathbf{x} \mid \mathbf{x}^{(s)}_{\text{rev}}) & \text{（去噪步）} \\ W^{(s+1)} = \arg\min_W \mathcal{L}(W; \mathbf{x}^{(s+1)}_{\text{rev}}) & \text{（TTT 步）} \end{cases}$$

其中 $E_\theta$ 是隐式能量函数。这种交替最小化的收敛性可以通过经典的 block coordinate descent 理论保证。

### 具体方案：DiffTTT（Diffusion Test-Time Training）

提出一种**无需修改预训练模型架构**的方法：

1. **侧路 TTT 模块**：受 Locas（Lu et al., 2026, arXiv:2602.05085）启发，将 TTT 模块实现为与主网络平行的低秩侧路（side-way FFN），共享主网络的激活和梯度进行初始化。

2. **去噪步内的 TTT 更新**：
   - 步骤 A：标准 MDLMs 前向传播，获得预测 $\hat{\mathbf{x}} = f_\theta(\mathbf{x}^{(s)})$
   - 步骤 B：对侧路模块执行 TTT 更新：用已揭示 token 的重建误差作为自监督信号
   - 步骤 C：将侧路模块的输出与主网络输出融合，形成增强预测 $\hat{\mathbf{x}}_{\text{aug}}$
   - 步骤 D：基于 $\hat{\mathbf{x}}_{\text{aug}}$ 执行 remasking 决策

3. **记忆持久化**：侧路模块的参数 $W^{(s)}$ 跨去噪步持久化。随着更多 token 被揭示，侧路模块逐步积累对当前序列的"记忆"，类似 Titans 的长期记忆模块。

4. **自适应步数**：基于 Prophet（Li et al., 2025, arXiv:2508.19982）的早期收敛发现，TTT 更新可以提供更准确的收敛判据——当侧路模块的自监督损失不再下降时，序列已经收敛，可以提前停止去噪。

### 理论保证

- **收敛性**：交替最小化在凸条件下全局收敛，非凸条件下收敛到临界点。MDLMs 的能量函数通常非凸，但实证上表现良好（EqT 理论保证了线性收敛率 $O(\rho^t)$，$\rho < 1$）。
- **正则化解释**：Huang & Mirzasoleiman（2026, arXiv:2601.22450）揭示了 MDLMs 中掩码采样分布的隐式正则化效应。TTT 更新可以被视为一种**显式正则化**，通过自监督目标将生成过程锚定在已观测上下文上，防止分布漂移。
- **与 DUEL 的互补**：Turok et al.（2026, arXiv:2603.01367）的 DUEL 框架实现了 MDLMs 的精确似然评估。DiffTTT 的理论分析可以利用 DUEL 的精确似然来量化 TTT 更新对生成分布的影响，而非依赖 ELBO 近似。

### 计算成本估算

- **侧路模块参数**：低秩 $r=64$，每层增加 $2 \times d \times r = 2 \times 4096 \times 64 = 524K$ 参数。总计约 0.02% 的额外参数（与 Locas 一致）。
- **每步开销**：TTT 更新需要一次小规模反向传播，约增加 3-5% 的单步计算量。
- **总开销**：$T$ 步去噪 × (1 + 0.05) ≈ 1.05T 倍标准 MDLMs 开销。
- **成功概率**：55-65%。最大不确定性在于侧路模块能否在如此少的参数下有效捕捉序列级记忆。

### 失败模式

1. **低秩瓶颈**：$r=64$ 的侧路可能信息容量不足以编码复杂的序列级依赖。
2. **梯度噪声**：早期去噪步中已揭示 token 少，TTT 梯度方差大，可能导致不稳定。
3. **适配训练的挑战**：将侧路模块融入预训练 MDLMs 需要少量适配训练。若适配数据不足或训练不当，侧路可能输出噪声而非有用信号。

---

## 方案对比与推荐

| 维度 | 角度一: TTT-Denoiser | 角度二: MI-Remask | 角度三: DiffTTT |
|------|---------------------|-------------------|-----------------|
| 类型 | 改进现有方法 | 新方法（信息论） | 跨域迁移 + 新方法 |
| 需要训练 | 是（轻量适配） | 否（训练无关） | 是（侧路适配） |
| 理论保证 | 自适应 MCMC 收敛 | 贝叶斯最优 remasking | 交替最小化收敛 |
| 计算开销 | +5-10%/步 | +<1%/步 | +3-5%/步 |
| 成功概率 | 60-70% | 50-60% | 55-65% |
| 创新度 | 中高 | 高 | 最高 |
| 实验难度 | 中 | 低 | 高 |
| 论文档次 | ICML/NeurIPS | NeurIPS/ICLR | NeurIPS/ICML |

### 推荐策略

**首选方案**：角度三（DiffTTT）作为主线，角度二（MI-Remask）作为理论工具。

理由：
1. DiffTTT 直接回应了 spec.md 中的核心问题——"能否将 TTT 作为插入层插入到现有 DLM 中，为其直接提供推理时记忆能力"。
2. MI-Remask 的信息论框架可以为 DiffTTT 中的 remasking 决策提供理论指导，两者自然结合。
3. 统一框架的理论贡献（TTT + MDLMs + MCMC 三者的形式化联系）具有独立的学术价值，即使实验结果不完全符合预期，理论贡献仍可支撑一篇高质量论文。

### 实验路线图

1. **Phase 0（1 天）**：在 MDLM-170M 上实现 MI-Remask，验证信息论 remasking 相比 ReMDM 的优势。使用 GSM8K/MATH500 作为快速迭代基准。
2. **Phase 1（2-3 天）**：在 MDLM-170M 上实现 DiffTTT 侧路模块，验证 TTT 更新对去噪质量的影响。对比 "标准去噪 $T$ 步" vs "TTT 增强去噪 $T/2$ 步" 的质量-效率权衡。
3. **Phase 2（3-4 天）**：Scale up 到 Dream-7B 或 LLaDA-8B。在 GSM8K、MATH500、HumanEval 上全面评测。对比 ReMDM、CORE、LookUM 等 baseline。
4. **Phase 3（2 天）**：完成理论分析（收敛证明、误差界推导）和消融实验。

---

## 关键理论文献基础

本提案建立在以下已有理论工作之上：

1. **收敛理论**：Chen et al.（2025, arXiv:2505.21400）— 扩散语言模型的信息论收敛理论，证明收敛速率与迭代步数 $T$ 和互信息的紧密依赖关系（matching lower bound）。
2. **误差界与最优调度**：Lavenant & Zanella（2025, arXiv:2510.25544）— MDLMs 因式化近似的误差界（相对熵），与序列长度无关，仅依赖每步平均生成 token 数。
3. **信息论分析**：Zhang et al.（2026, arXiv:2602.00286）— 生成顺序和并行解码的信息论框架；证明 remasking 无法保证分布正确性但实践有效。
4. **隐式正则化**：Huang & Mirzasoleiman（2026, arXiv:2601.22450）— MDLMs 掩码采样的隐式正则化效应和信息论界。
5. **精确似然**：Turok et al.（2026, arXiv:2603.01367）— DUEL 框架实现 MDLMs 精确似然评估。
6. **信息论离散扩散**：Jeon et al.（2025, arXiv:2510.24088）— 离散扩散的信息论基础。
7. **最优解码路径**：Z. Chen et al.（2025, arXiv:2512.21336）— 通过不确定性量化优化 MDMs 解码路径。
8. **锚定扩散**：Rout et al.（2025, arXiv:2505.18456）— ADLM 的信息论 token 遮蔽策略。
9. **TTT 基础**：Sun et al.（2024, arXiv:2407.04620）— TTT-Linear/TTT-MLP，隐状态即机器学习模型。
10. **Titans**：Behrouz & Zhong（2025, arXiv:2501.00663）— 学习在测试时记忆，三层记忆架构。
11. **MesaNet**：von Oswald et al.（2025, arXiv:2506.05233）— 最优 TTT 的序列建模，共轭梯度求解器。
12. **Equilibrium Transformers**：Jafari & Anbarjafari（2025, arXiv:2511.21882）— 统一 TTT、扩散 LM 和迭代精化为能量最小化。
13. **Locas**：Lu et al.（2026, arXiv:2602.05085）— 局部支持参数化记忆，0.02% 参数即可存储上下文。
14. **AllMem**：Wang et al.（2026, arXiv:2602.13680）— SWA + TTT 混合架构，128K 上下文超越全注意力。
15. **计算通用性**：Svete & Sabharwal（2025, arXiv:2510.06190）— MDMs 等价于循环 Transformer，any-process generation 扩展可解问题类。
16. **最优并行采样**：Jiang et al.（2025, arXiv:2512.25014）— DLM + remasking 是最优并行采样器的形式化证明。
17. **CreditDecoding**：Wang et al.（2025, arXiv:2510.06133）— Trace Credit 量化 token 收敛潜力，历史 logits 融合加速。
18. **Prophet**：Li et al.（2025, arXiv:2508.19982）— 早期答案收敛现象，动态停止准则。
