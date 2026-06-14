# Backup Ideas for Potential Pivot（第 4 轮迭代）

## 备选方案 1：Tolerator + BSD 两阶段信念精化

### 核心思路

如果 BSD 的全流程信念精化（Phase 1 连续演化 + Phase 2 硬揭示）失败——例如信念向量的 OOD 问题无法通过归一化解决——则切换到 Tolerator 的"先填满后精化"框架，仅在精化阶段使用 BSD 信念向量。

### 具体方案

1. **阶段 1（填充）**：标准 DLM 去噪填满所有 token（与 Tolerator 完全一致）
2. **阶段 2（BSD 精化）**：
   - 按 Tolerator 策略选择 k% token remask
   - 对 remask 位置用 BSD 信念向量（而非 mask embedding）重新去噪
   - 信念向量初始化为阶段 1 的 logit 分布加权 embedding
   - 精化 R 轮（R=3-5），每轮使用 BSD 的 EMA 更新

### 为什么这是合理的 pivot

- Tolerator 的代码和结果已公开验证，阶段 1 复现风险极低
- BSD 精化仅在已有完整序列的基础上运行，信念向量不再完全 OOD（有已揭示 token 的上下文支撑）
- 计算开销与标准 Tolerator 相当（精化阶段替换 mask embedding 为信念向量，无额外前向传播）

### 成功概率：50%

### 预计计算量：~25 GPU·h

---

## 备选方案 2：HEX + RACFG 多专家推理引导

### 核心思路

如果 BSD 失败但 RACFG（CFG 引导）在 Dream-7B 上有效，则将 RACFG 与 HEX 的多专家框架结合。HEX 通过不同 block schedule 暴露 DLM 的多个隐式专家，majority vote 从 24.7% → 88.1%。在每个专家轨迹中加入 RACFG 引导，可以提升每个独立轨迹的质量，从而提升 majority vote 的上限。

### 具体方案

```
For each block schedule s_k (K=8):
    1. 用 schedule s_k + RACFG 引导运行去噪
    2. RACFG 的跨步稳定性信号在每个 schedule 内独立计算
    3. 输出引导精化后的候选 c_k
Aggregation: majority vote on {c_1, ..., c_K}
```

### 优势

- HEX 是目前 DLM TTS 最强的 training-free 基线（GSM8K 88.1%）
- RACFG 在每个轨迹内独立运行，不引入跨轨迹耦合
- HEX 原文用 LLaDA，Dream-7B 上的复现本身是新贡献

### 风险

- HEX 88.1% 已极高，RACFG 的边际空间有限
- K 个轨迹 × 2x RACFG = 2K × 前向传播，计算开销线性倍增

### 成功概率：45%

### 预计计算量：~40 GPU·h

---

## 备选方案 3：DLM 推理时扩展的实证系统研究（负面结果论文升级版）

### 核心思路

如果 BSD 和 RACFG 都失败，则利用项目 20+ 轮迭代积累的海量实验数据，撰写一篇系统性的实证研究论文：从信息传递谱系（vanilla → DMI → BSD/RACFG）的角度，量化不同层次的跨步信息对推理质量的边际贡献。

### 论文框架

**Title**: "The Information Hierarchy of Inference-Time Scaling in Masked Diffusion Language Models: What Works, What Doesn't, and Why"

**核心贡献**：
1. **跨步信息传递的四层谱系**：无信息（vanilla）→ 表示级（DMI/BSD）→ 预测级（RACFG/A-CFG）→ 参数级（DTA），系统量化每层的边际收益
2. **20+ 方法的统一对比**：在同一评估框架下对比 ReMDM、CORE、RCR、DMI、BSD、RACFG、DTA、HEX 等
3. **计算量公平的 Pareto 曲线**：FLOPs vs 准确率的完整图景
4. **失败模式分类学**：文本退化、梯度信号不足、OOD 输入、分布正确性等
5. **DMI 作为 sweet spot 的发现**：零额外计算 + 2x 改善的独特价值

### 成功概率：70%（数据已有，即使新方法失败也可发表）

### 目标会议：EMNLP/NeurIPS（系统研究 track）

---

## Pivot 决策树

```
Phase 1 Pilot 结果
├── A-CFG 在 Dream-7B 有效 + BSD pilot 通过
│   → 继续主方案（BSD + RACFG）
│
├── A-CFG 有效 + BSD pilot 失败（OOD）
│   ├── DMI-style fallback 有效 → 降级 BSD 为 DMI+α
│   └── 完全失败 → 转向备选 2（HEX + RACFG）
│
├── A-CFG 在 Dream-7B 无效
│   ├── BSD pilot 通过 → BSD standalone 论文
│   └── BSD 也失败 → 转向备选 1（Tolerator + BSD）
│
└── 全部失败
    → 备选 3（实证系统研究论文）
```
