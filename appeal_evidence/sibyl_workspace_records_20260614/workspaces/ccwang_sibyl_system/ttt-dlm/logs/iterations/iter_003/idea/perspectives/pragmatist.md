# 实用主义者提案（Iteration 4 — TTT × DLM 融合）

## 核心判断

前 18 轮迭代证明了一个关键事实：**training-free remasking 的推理时扩展效果极其有限**——0.6B 模型上 PPL 改善是退化伪影，8B 模型上 PPL 反而恶化。但 spec 提出了一个全新且极具价值的方向：**将 TTT（Test-Time Training）的记忆机制引入 DLM**。这不是简单的"给 DLM 加 remasking"，而是从架构层面赋予 DLM 推理时学习能力。

从工程可行性角度，我评估了三条实现路径，按 ROI 排序如下。

---

## 方向 1（推荐）：DLM 去噪迭代作为隐式 TTT —— 理论框架 + 轻量验证

### 核心洞察

TTT 的本质是：**推理时通过自监督损失更新模型隐状态/参数**。DLM 的去噪过程天然具备这一特性：每一步去噪都是在当前部分揭示的上下文下，对剩余 mask token 做预测——这在数学上等价于一种 **masked language modeling 自监督任务**。换言之，**DLM 的多步去噪本身就是一种隐式 TTT**，每步利用前步揭示的 token 作为"训练信号"更新后续预测。

但当前 DLM 在去噪步之间**不更新参数**（frozen weights），仅通过不同 mask 模式下的前向传播获取新信息。如果我们在去噪步间加入**轻量参数更新**（如低秩适配），就能将这种隐式 TTT 变为显式的。

### 与已有工作的区别

| 方法 | 参数更新 | 记忆机制 | 需要训练 |
|------|---------|---------|---------|
| ReMDM/CORE/Prism | 否 | 无（纯采样策略） | 否 |
| RemeDi | 否（推理时） | 双流架构 UPS | 是（修改架构） |
| ProSeCo | 否（推理时） | 训练 corrector | 是（额外训练） |
| TTT-Linear/MLP | 是（隐状态=模型） | 自监督更新隐状态 | 是（从头训练） |
| **本方案** | **是（LoRA 参数）** | **去噪步自监督** | **轻量 LoRA 适配** |

### 具体方案

**阶段 1：理论分析（1 周）**
- 建立 DLM 去噪过程与 TTT 的数学对应关系
- 证明：DLM 的 T 步去噪等价于 TTT 的 T 步在线学习，目标函数为 masked token 预测损失
- 分析 DLM 相比 AR 模型做 TTT 的结构性优势：AR 的 TTT 需要额外的自监督损失（如 reconstruction），DLM 的去噪目标天然就是自监督的
- 参考 Equilibrium Transformers (arXiv 2511.21882) 的框架——该工作证明迭代潜在精化可视为近似 MAP 推理

**阶段 2：Denoise-Time Adaptation (DTA)（2 周）**

在 DLM 去噪过程中插入轻量参数更新：

```
Input: 全 mask 序列 x_T, 预训练 DLM f_θ, LoRA 参数 Δθ (初始化为 0)
For t = T, T-1, ..., 1:
    1. 用 f_{θ+Δθ} 预测所有 mask token
    2. 按置信度/调度揭示部分 token → x_{t-1}
    3. [DTA 步] 用已揭示 token 作为监督信号，
       计算 masked LM 损失 L = -log P(revealed | x_{t-1})
       更新 Δθ ← Δθ - α∇L（仅更新 LoRA，1 步 SGD）
    4. 可选：按策略 remask 低置信度 token
Output: x_0
```

关键设计：
- **LoRA rank 极小（r=4-8）**：每步更新只需 1 次额外反向传播，参数量 < 0.01% 总参数
- **累积 vs 重置**：可选择跨步累积 LoRA 或每步重置（实验决定）
- **仅在前 k 步做 DTA**：后期步骤信息增益递减，可节省计算
- **与 remasking 互补**：DTA 更新模型理解，remasking 修正低置信 token，两者正交

**阶段 3：基准验证（2 周）**
- 模型：Dream-7B（最强开源 DLM，有 AR 初始化基础）
- 基准：Countdown（DLM 强项，快速评估）→ GSM8K（推理）→ MBPP（代码）
- 对比：vanilla decoding, ReMDM-conf, CORE, RCR, Best-of-N, **DTA (ours)**
- 计算预算控制：固定 NFE（前向评估次数），比较同等计算下的性能

### 计算资源估算

| 步骤 | GPU 需求 | 时间 |
|------|---------|------|
| 理论分析 | 0 GPU | 1 周 |
| DTA 实现（Dream-7B） | 1× A100 80GB | 3 天 |
| Countdown 实验（~200 样本） | 1× A100 | 2 天 |
| GSM8K 实验（1319 样本） | 2× A100 | 3 天 |
| MBPP 实验（500 样本） | 1× A100 | 2 天 |
| 对比实验 + ablation | 2× A100 | 3 天 |
| **总计** | | **~5 周** |

### 成功概率评估
- 理论框架成立概率：**85%**（DLM 去噪与 TTT 的对应关系在数学上是直接的）
- DTA 显著提升 benchmark 概率：**50-60%**（主要风险：LoRA 更新可能在低步数下不稳定）
- 论文发表（顶会）概率：**55-65%**（如果理论+实验都 work）

### 失败模式与应对
1. **LoRA 更新不稳定**：降低学习率；改用 momentum；只在信息增益大的步做更新
2. **计算开销过大**：只对前 3-5 步做 DTA；用 GaLore 等低内存方法；固定 attention 层只更新 FFN
3. **与 baseline 无显著差异**：聚焦理论贡献（"DLM = 隐式 TTT"的形式化证明本身有价值）

---

## 方向 2（备选）：TTT 层作为可插拔记忆模块

### 核心思路

不修改 DLM 的去噪过程，而是**在预训练 DLM 中插入 TTT 层**（类似 TPTT 框架），让模型在去噪的每一步都能访问长程记忆。

### 已有基础

- **TPTT** (arXiv 2506.17671)：已实现将预训练 Transformer 转化为 Titans 架构，支持 LoRA 微调，代码开源 (https://github.com/fabienfrfr/tptt)
- **AllMem** (arXiv 2602.13680)：用 SWA + TTT 记忆网络替换 attention，4K 窗口在 37K LongBench 上仅掉 0.83 分
- **Locas** (arXiv 2602.05085)：可插拔参数化记忆，0.02% 额外参数即可存储长上下文信息
- **lucidrains/titans-pytorch**：非官方 PyTorch 实现，可直接使用

### 方案

```
1. 取 Dream-7B 预训练权重
2. 在 [4, 8, 12, 16] 层后插入 TTT-Linear 模块（参考 TPTT）
3. LoRA (r=16) 微调：
   - 数据：GSM8K train set + Countdown 合成数据
   - 目标：标准 masked LM loss
   - 训练量：~2K 步（参考 wd1 的 20 步 RL 极低成本方案）
4. 推理时 TTT 层自动积累去噪上下文作为记忆
```

### 与方向 1 的对比

| | 方向 1 (DTA) | 方向 2 (TTT 层插入) |
|---|---|---|
| 修改范围 | 仅推理流程 | 模型架构 |
| 训练需求 | 无（运行时适配） | 轻量 LoRA 微调 |
| 理论贡献 | 强（DLM = 隐式 TTT） | 中（工程集成） |
| 实验风险 | 中（更新稳定性） | 中（微调可能破坏 DLM 去噪能力） |
| 新颖度 | 高 | 中（TPTT 已证明可行性） |

### 计算资源
- LoRA 微调 Dream-7B：2× A100，~1 天
- 评估：同方向 1
- **总计：~4 周**

### 成功概率
- 微调成功概率：**70%**（TPTT 已验证可行性，但 DLM 的 mask 机制可能与 TTT 层有冲突）
- 显著提升概率：**40-50%**（可能 work 但提升幅度未必超过 RL 方法）
- 论文概率：**40-50%**（工程贡献为主，新颖度不如方向 1）

---

## 方向 3（探索性）：HEX 启发的多策略集成 + DTA

### 核心洞察

HEX (arXiv 2510.05040) 的关键发现：**DLM 隐式学习了多个半自回归专家**，不同 mask 调度暴露不同的专家行为。单一固定调度会坍缩性能。HEX 通过多调度 majority vote 在 GSM8K 上从 24.7% 提升到 88.1%，无需训练。

这启发了一个想法：**DTA + 多调度集成**。DTA 在每种调度下适配不同的 LoRA 参数，相当于让每个"隐式专家"都获得针对性的推理时记忆。

### 方案
1. 生成 K 条不同 block schedule 的去噪轨迹
2. 每条轨迹独立运行 DTA（独立 LoRA 初始化）
3. 最终 majority vote 或 PRM 引导选择

### 风险
- 计算开销 = K × DTA，可能过大
- 与 HEX 的纯集成方案相比，DTA 的边际收益不确定

### 成功概率：**35-45%**（但如果 work，效果可能最强）

---

## 综合推荐与执行计划

### 推荐策略：方向 1 为主线，方向 2 为保底

**第 1-2 周**：方向 1 理论分析
- 建立 DLM 去噪 ↔ TTT 在线学习的数学对应
- 写出理论框架初稿
- 如果理论框架说不通，立即转方向 2

**第 3-4 周**：DTA 实现与 Countdown 快速验证
- 在 Dream-7B 上实现 DTA
- Countdown 快速 sanity check（~200 样本，几小时可完成）
- 如果 Countdown 上 DTA 无改善，尝试方向 2 的 TTT 层插入

**第 5 周**：主实验
- 选择更优方案（DTA 或 TTT 层），在 GSM8K + MBPP 上做完整评估
- 与 ReMDM-conf, CORE, RCR, HEX, Best-of-N 做公平对比

**第 6-7 周**：论文写作
- 理论 + 实验 + 分析

### 关键文献引用

**TTT 系列**：
- TTT-Linear/MLP (Sun et al., arXiv 2407.04620): 原始 TTT 层提案
- Titans (Behrouz et al., arXiv 2501.00663): 神经记忆模块，长程记忆
- LaCT / TTT Done Right (Zhang et al., arXiv 2505.23884): 大 chunk TTT，14B 规模验证
- MesaNet (von Oswald et al., arXiv 2506.05233): 最优在线学习的 TTT 变体
- TNT (Li et al., arXiv 2511.07343): 解耦训练效率与推理性能的 TTT 训练范式
- TPTT (Furfaro, arXiv 2506.17671): 预训练 Transformer → Titans 转化框架，支持 LoRA
- AllMem (Wang et al., arXiv 2602.13680): SWA + TTT 记忆网络混合架构
- Locas (Lu et al., arXiv 2602.05085): 可插拔参数化记忆模块
- PonderTTT (Sim, arXiv 2601.00894): 自适应 TTT 更新门控，重构损失驱动
- REFINE (Hwang et al., arXiv 2602.16704): RL 增强 TTT 快速权重，NSP 目标
- Equilibrium Transformers (Jafari, arXiv 2511.21882): 闭环迭代精化 = 近似 MAP 推理
- DRDT3 (Huang et al., arXiv 2501.06718): TTT 层 + 扩散精化（RL 领域）

**DLM 推理时扩展**：
- HEX (Lee et al., arXiv 2510.05040): 多调度集成，GSM8K 24.7%→88.1%，无需训练
- ReMDM (arXiv 2503.00307): remasking 采样器
- CORE (arXiv 2602.04096): 上下文鲁棒 remasking
- MDPO/RCR (arXiv 2508.13148): RL + 即插即用 remasking
- Prism (arXiv 2602.01842): 层次搜索 + 自验证

**开源代码资源**：
- TPTT: https://github.com/fabienfrfr/tptt （预训练→Titans 转化）
- lucidrains/titans-pytorch: https://github.com/lucidrains/titans-pytorch （Titans 非官方实现）
- Dream: https://github.com/DreamLM/Dream （基座模型）
- dLLM: https://github.com/ZHZisZZ/dllm （统一 DLM 框架，支持 LoRA）
- MDPO: https://github.com/autonomousvision/mdpo （RCR 实现）

---

## 不建议做的事

1. **不要尝试从头训练 TTT-DLM 混合架构**。训练 7B 模型从头需要的资源远超我们的能力。必须基于预训练权重做轻量适配。
2. **不要同时做三个方向**。资源有限，集中在方向 1（理论最强）+ 方向 2 作为保底。
3. **不要忽略 HEX 的结果**。HEX 在 GSM8K 上 3.56x 提升是非常强的 baseline，任何新方法必须在此基础上证明附加价值。
4. **不要用 PPL 作为主要指标**。前 18 轮的教训刻骨铭心——必须用 benchmark accuracy（Countdown, GSM8K, MBPP）作为主评估指标。
5. **不要在 0.6B 模型上浪费时间**。前 18 轮证明小模型的 remasking 结果不可靠，直接在 7B 上做实验。

---

## 一句话总结

**DLM 的多步去噪天然就是一种 masked LM 自监督学习过程——如果我们在去噪步间加入参数更新，就能将这种隐式 TTT 转化为显式的推理时学习，这是 DLM 相比 AR 模型的独特结构性优势。**
