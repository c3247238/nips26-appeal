# 实用主义者提案（Iteration 4 — TTT × DLM 融合，修订版）

## 核心判断

前 18 轮迭代的核心教训已经非常清晰：(1) training-free remasking 在 PPL 上的改善是退化伪影；(2) 小模型（0.6B）结果不可靠；(3) PPL 不可作为主指标。Spec 提出了全新方向——**将 TTT 的记忆/学习能力引入 DLM**。上一轮提案的 DTA（Denoise-Time Adaptation）方向理论上成立，但工程实现的可行性和风险需要更务实地重新评估。

本轮提案在保持 DTA 核心思想的同时，**重点增加了两个实用性更强的方向**：一个利用现有开源工具零成本实现，另一个将 DLM 的迭代去噪重新定位为产业界更关注的 draft-refine 范式。三个方向按 ROI 排序，确保至少有一个在 4 周内可以产出可发表结果。

---

## 方向 1（最高 ROI，推荐主攻）：Tolerator + DTA 两阶段推理时扩展

### 核心洞察

Tolerator（arXiv 2510.05090）提出了一个关键但被低估的框架："先填满，后精化"（Finish First, Perfect Later）。其核心是两阶段解码：(1) 快速用标准去噪填满所有 token；(2) 迭代 remask 一个子集并重解码，利用其余已揭示 token 作为上下文进行交叉验证。这在 5 个 benchmark 上在相同计算预算下一致超越 baseline。

**关键洞察**：Tolerator 的第二阶段（iterative refinement）本质上就是在做 TTT——每次 remask-decode 循环都是一次自监督学习（用已有 token 预测被遮蔽 token）。如果我们在这个精化阶段加入 **轻量 LoRA 参数更新**（DTA），就能将这种"伪 TTT"转化为真正的参数级推理时学习。

### 与现有工作的精确区分

| 方法 | 阶段 1（填充） | 阶段 2（精化） | 参数更新 | 额外训练 |
|------|---------------|---------------|---------|---------|
| Vanilla DLM | 标准去噪 | 无 | 否 | 否 |
| ReMDM | 标准去噪+remask | remask 循环 | 否 | 否 |
| Tolerator | 标准去噪 | cross-validation remask | 否 | 否 |
| CORE | 标准去噪 | 上下文扰动 remask | 否 | 否 |
| RemeDi | 双流去噪 | UPS 引导 remask | 否 | 是（架构修改） |
| **Tolerator+DTA (ours)** | **标准去噪** | **cross-validation remask + LoRA 更新** | **是** | **否（运行时）** |

### 具体方案

```
阶段 1: Sequence Fill-Up（标准 DLM 去噪，与 Tolerator 相同）
    Input: 全 mask 序列 x_T, 预训练 DLM f_θ
    For t = T, ..., 1: 标准去噪揭示 token → x_0（完整序列）

阶段 2: DTA-Enhanced Refinement（核心创新）
    Input: x_0, LoRA 参数 Δθ (初始化为 0)
    For r = 1, ..., R (精化轮数，默认 R=3-5):
        1. 按 Tolerator 策略选择 k% token 进行 remask → x_r^masked
        2. 用 f_{θ+Δθ} 预测被遮蔽 token → x_r^pred
        3. [DTA 步] 计算 MLM 损失 L = -Σ log P(x_0[i] | x_r^masked)
           其中 x_0[i] 是原始预测的 token（作为伪标签）
           更新 Δθ ← Δθ - α∇L（LoRA rank=4, 1 步 SGD, lr=1e-5）
        4. 用更新后的 f_{θ+Δθ} 重新预测 → 接受置信度提升的替换
    Output: 精化后的 x_0'
```

### 为什么这个方案工程可行

1. **Tolerator 已有代码和验证**：代码公开，已在 5 个 benchmark 验证有效
2. **DTA 只需在 Tolerator 的精化循环中加 1 次反向传播**：极小的代码改动
3. **LoRA rank=4 的内存开销可忽略**：对 7B 模型增加 ~2MB 参数
4. **精化轮数少（3-5 轮）**：总计增加 3-5 次反向传播，在合理计算预算内
5. **失败模式可控**：如果 DTA 无效，退化为标准 Tolerator（仍然有效）

### 计算资源估算

| 步骤 | GPU 需求 | 时间 |
|------|---------|------|
| 实现 Tolerator 基线 | 1× GPU (7B) | 2 天 |
| 加入 DTA 模块 | 同上 | 1 天 |
| Countdown 快速验证（200 样本） | 1× GPU | 4 小时 |
| GSM8K 完整评估（1319 样本） | 2× GPU | 1 天 |
| MBPP（500 样本） | 1× GPU | 8 小时 |
| Ablation（R, rank, lr 扫参） | 2× GPU | 2 天 |
| **总计** | | **~1.5 周** |

### 成功概率

- Tolerator 基线复现：**90%**（代码公开，结果已验证）
- DTA 在 Tolerator 之上有额外提升：**55-65%**
- 论文发表概率：**60-70%**（即使 DTA 边际提升小，"DLM 去噪 = 隐式 TTT"的理论框架 + Tolerator+DTA 的系统验证本身有贡献）

---

## 方向 2（稳健备选）：HEX + DTA 多专家自适应

### 核心洞察

HEX（arXiv 2510.05040）发现 DLM 隐式学习了多个半自回归专家，不同 block schedule 暴露不同专家行为。GSM8K 上 majority vote 从 24.7% → 88.1%（3.56x），这是目前 DLM 推理时扩展最强的 training-free baseline。

**关键洞察**：HEX 的每个 block schedule 生成一个独立去噪轨迹。如果我们对每个轨迹独立运行 DTA（独立 LoRA 初始化），相当于**每个"隐式专家"获得针对自身去噪上下文的自适应记忆**。这比单一 DTA 更能发挥 DLM 的多专家结构性优势。

### 方案

```
Input: prompt, K 种 block schedules {s_1, ..., s_K}
For each schedule s_k in parallel:
    1. 用 schedule s_k 运行标准去噪 → 候选序列 c_k
    2. [DTA] 在 c_k 上运行 R 轮 LoRA 精化（独立 Δθ_k）
    3. 输出精化后的 c_k'
Aggregation: majority vote on {c_1', ..., c_K'}
```

### 与 HEX 的差异

- HEX = 多 schedule + majority vote（纯采样策略）
- **Ours** = 多 schedule + **per-expert DTA** + majority vote（采样 + 学习）
- 每个专家的 DTA 是独立的，不会跨专家干扰

### 计算资源

- 与 HEX 相比额外开销：K × R 次反向传播（K=8 schedules, R=3 精化步 = 24 次额外反向传播）
- 总计约 2× HEX 的计算量（但 HEX 本身需要 K 次完整去噪，所以 DTA 边际开销占比 ~30-50%）

### 成功概率

- HEX 基线复现：**85%**（方法清晰，但需要自行实现 block schedule 变换）
- DTA 在 HEX 之上有额外提升：**40-50%**（HEX 的 88.1% 已经很高，边际空间有限）
- 论文发表概率：**50-60%**

### 风险

- HEX 88.1% 的 GSM8K 结果极强，留给 DTA 的改进空间有限
- 多 schedule 的 DTA 计算开销是线性倍增的
- **降级策略**：如果 DTA 无效，至少我们有 HEX 在 Dream-7B 上的复现数据（HEX 原文用 LLaDA，Dream 上的结果本身是新贡献）

---

## 方向 3（高风险高回报）：DLM 去噪过程的形式化 TTT 理论 + DTA 验证

### 核心洞察

这是上一轮提案方向 1 的精炼版。核心理论命题：

**命题**：令 DLM 的 T 步去噪过程为 {x_T → x_{T-1} → ... → x_0}，每步 t 用参数 θ 的模型 f_θ 在当前 mask 模式下预测所有 masked token。这在数学上等价于一个 T 步在线学习过程，其中：
- "训练数据"= 前步揭示的 token（递增序列）
- "损失函数"= masked language modeling loss
- "模型更新"= 隐式的（通过改变 mask 模式改变输入分布）

AR 模型的 TTT 需要额外设计自监督损失（如 next-token prediction on augmented input），但 DLM 的去噪目标**天然就是自监督的**——这是 DLM 相比 AR 的独特结构性优势。

Equilibrium Transformers (arXiv 2511.21882) 已经证明了一个统一框架：深度平衡模型、扩散语言模型和 TTT 都是"闭环迭代精化"的特例。我们的工作可以在此基础上给出 DLM 专属的更精细分析。

### 理论贡献定位

1. **定理 1**（DLM 去噪 = 隐式在线学习）：证明 DLM 的 T 步去噪等价于对递增观测序列的 T 步在线凸优化，遗憾界与步数 T 和 mask schedule 有关
2. **定理 2**（DTA 加速收敛）：证明在去噪步间加入参数更新（DTA）可降低遗憾界的常数因子，特别是在 mask schedule 不最优时
3. **推论**（DLM vs AR 的 TTT 效率）：AR 模型需要 O(T²) 次前向传播做 T 步 TTT（每步需重新处理完整上下文），DLM 只需 O(T) 次（每步在当前 mask 下单次前向传播）

### 实验验证

用方向 1 的 Tolerator+DTA 或方向 2 的 HEX+DTA 作为实验载体，验证理论预测。

### 计算资源

- 理论分析：0 GPU，1 周
- 实验验证：与方向 1/2 共享
- **总计：额外 1 周理论工作**

### 成功概率

- 理论框架成立：**75%**（EqT 已铺路，但需要处理离散空间的技术细节）
- 实验验证理论预测：**50%**（理论和实践可能有差距）
- 论文发表（顶会）：**55-65%**

---

## 综合执行计划

### 推荐策略：方向 1 为主线，方向 3 理论并行推进

**第 1 周**：方向 1 实现 + 方向 3 理论分析
- 复现 Tolerator baseline（Dream-7B 上）
- 实现 DTA 模块（LoRA rank=4, 精化阶段加 backward pass）
- Countdown 快速 sanity check
- 并行：写理论框架初稿

**第 2 周**：主实验 + 对比
- GSM8K + MBPP 完整评估
- 对比 baseline：vanilla Dream-7B, ReMDM-conf, CORE, RCR, Tolerator (无 DTA), HEX (majority vote)
- Ablation：精化轮数 R ∈ {1, 3, 5, 8}, LoRA rank ∈ {2, 4, 8}, 学习率扫参

**第 3 周**：深化 + 备选
- 如果方向 1 有效：加入 NFE 公平对比（固定总前向评估次数，比较各方法性能）
- 如果方向 1 无效：切换到方向 2（HEX+DTA），复用 DTA 模块
- 补充：DTA 在不同精化阶段的贡献分析（early remask vs late remask 的 DTA 效果）

**第 4-5 周**：论文写作
- 理论（DLM = 隐式 TTT）+ 方法（Tolerator+DTA / HEX+DTA）+ 实验 + 分析

### 关键文献引用

**TTT 基础**：
- TTT-Linear/MLP (Sun et al., arXiv 2407.04620): 原始 TTT 层
- Titans (Behrouz et al., arXiv 2501.00663): 神经记忆模块
- Test-Time Learning for LLMs (arXiv 2505.20633): LoRA-based TTT for LLMs，与我们的 DTA 最直接相关
- Equilibrium Transformers (Jafari, arXiv 2511.21882): 统一框架（EqT = DLM = TTT）
- TPTT (Furfaro, arXiv 2506.17671): 预训练→Titans 转化，开源 (github.com/fabienfrfr/tptt)

**DLM 推理时扩展（实践基线）**：
- Tolerator (Tian et al., arXiv 2510.05090): 两阶段 fill-up + cross-validation refinement，代码公开
- HEX (Lee et al., arXiv 2510.05040): 多 schedule 集成，GSM8K 24.7%→88.1%
- ReMDM (arXiv 2503.00307): remasking 采样器
- CORE (arXiv 2602.04096): 上下文鲁棒 remasking
- MDPO/RCR (arXiv 2508.13148): RL + 即插即用 remasking
- Prism (arXiv 2602.01842): 层次搜索 + 自验证
- LookUM (arXiv 2511.05563): 不确定性验证，base 匹配 RL 后训练效果
- APS (Rout et al., arXiv 2510.02291): anchored remasking for posterior sampling

**DLM 基座模型**：
- Dream 7B (arXiv 2508.15487): 主实验模型
- LLaDA 8B (arXiv 2502.09992): 补充验证

**开源代码资源**：
- Tolerator: 代码公开（arXiv 2510.05090 附代码）
- Dream: https://github.com/DreamLM/Dream
- dLLM 统一框架: https://github.com/ZHZisZZ/dllm（支持 LoRA，方便实现 DTA）
- MDPO/RCR: https://github.com/autonomousvision/mdpo
- Prism: https://github.com/viiika/Prism
- Self-Rewarding SMC: https://github.com/Algolzw/self-rewarding-smc

---

## 不建议做的事

1. **不要从头训练 TTT-DLM 混合模型**——计算资源远超可用范围
2. **不要在 0.6B 模型上实验**——前 18 轮证明小模型 remasking 结果不可靠
3. **不要用 PPL 作为主指标**——必须用 benchmark accuracy（Countdown, GSM8K, MBPP）
4. **不要同时跑三个方向**——集中在方向 1，方向 2 作为 backup，方向 3 纯理论并行
5. **不要忽略 NFE 公平对比**——DTA 增加了反向传播成本，必须在同等总计算预算下比较
6. **不要重复上一轮的方向 2（TTT 层插入）**——TPTT 已有类似工作，且需要微调可能破坏 DLM 去噪能力，ROI 不如 Tolerator+DTA

---

## 一句话总结

**DLM 的去噪精化阶段（如 Tolerator 的 cross-validation refinement）天然具备 TTT 的自监督结构——在精化循环中加入极小的 LoRA 参数更新（DTA），以近零额外工程成本将隐式 TTT 转化为显式推理时学习，这是目前 ROI 最高的 TTT × DLM 融合路径。**
