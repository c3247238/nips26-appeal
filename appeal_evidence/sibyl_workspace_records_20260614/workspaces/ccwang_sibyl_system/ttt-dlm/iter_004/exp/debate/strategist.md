# 战略顾问分析报告

**日期**: 2026-03-08
**项目**: TTT-DLM（遮蔽扩散语言模型推理时计算扩展）
**分析人**: 战略顾问 (Strategist Agent)

---

## 一、形势总结

经过 20+ 轮迭代，项目已系统性探索了以下推理时改进策略：

| 方法类别 | 具体方法 | 结果 |
|---------|---------|------|
| 参数适应 (TTT) | 6 个变体（Block-Reset, Prompt-TTT 等） | 全部不显著 (p=0.88) |
| 过程适应 (ReMask-Retry) | 固定/自适应比例 | PPL 改善由文本退化驱动 |
| 计算分配 (Best-of-N) | N=2,3 | 完全无效甚至恶化 |
| 轨迹一致性 (TCR) | r30/r50 on Dream-7B | mean PPL 改善但 median 无变化，不显著 |
| 温度退火 | 线性/余弦退火 | full-scale 不显著 (p=0.636) |
| 熵引导重遮蔽 | entropy_r20_mean | full-scale 不显著 (p=0.530) |
| 并行投票 | — | 完全失败 (+16.4%) |

**核心事实**：在 101 prompts x 3 seeds 的 full-scale 验证中，**没有任何 training-free 推理时方法在 Dream-7B 上产生统计显著的质量改善**。Pilot 阶段的大幅改善（-16% 到 -25%）全部是小样本噪声假象。

---

## 二、PIVOT vs PROCEED 决策

### 推荐：**有条件 PROCEED（限时测试 IGIR on benchmark），同时准备 PIVOT 备案**

#### 支持 PROCEED 的理由

1. **IGIR 提案质量高，且未经测试**。IGIR 融合了 IB 理论框架 + MH 接受准则 + 退火调度，是目前唯一同时具备理论保证和抗退化机制的方案。此前失败的方法都是"单一信号 + 贪心接受"，IGIR 在机制层面有本质区别。

2. **此前所有实验都用 PPL 作为评估指标**。项目自身已证明 PPL 不可靠（iteration 015 的核心发现）。IGIR 提案明确要求以 **benchmark 准确率**（GSM8K/MATH500）为唯一判据，这是此前未做过的评估方式。在 benchmark 上的结论可能与 PPL 上截然不同。

3. **IGIR-Lite 实现成本极低**。核心算法约 300 行 Python，Phase 0 验证仅需 3-5 天，Phase 1 概念验证 5-7 天。总计 **8-12 天即可得出 Go/No-Go 结论**。

4. **Dream-7B 在 Countdown 上已显著优于 AR**（16.0 vs 6.2），说明 DLM 在推理任务上有独特优势，推理时扩展可能在这类"DLM 本身就强"的任务上更容易看到增益。

#### 支持 PIVOT 的理由

1. **20+ 轮全部失败的基本面**。所有 training-free 方法在 full-scale 上均失败，这强烈暗示 DLM 推理时改进的天花板可能接近于零。IGIR 虽然机制更精致，但如果根本性障碍在于"DLM 的去噪过程已经接近模型能力上限"，那么更聪明的重遮蔽选择也无法突破。

2. **竞争格局不利**。RemeDi（ICLR 2026）、d1/wd1、DiSPO、Prism 等方法已占据推理时扩展的主要位点。即使 IGIR 有效，其贡献的差异化窗口在缩小。

3. **时间成本的机会成本**。IGIR 完整方案（Phase 0-5）需要 30-44 天。如果中途失败，将消耗 2-6 周时间而无可发表产出。

#### 决策框架

```
IGIR Phase 0（3-5 天）
  ├── IB-Score 近似质量 r > 0.5 且 Q 分数区分正确/错误答案 → Phase 1
  ├── 任一假设不成立 → 立即 PIVOT
  └── Phase 1（5-7 天）
       ├── GSM8K 100 题子集准确率 >= vanilla +1.5% → PROCEED to Phase 2
       └── 无改善 → PIVOT（已获得理论+负面实验的论文素材）
```

**硬止损线：Phase 1 结束（约 10-12 天）后若无 benchmark 改善，立即 PIVOT。**

---

## 三、如果 PROCEED：最高 ROI 的下一步

### 推荐路径：IGIR-Lite on GSM8K（Dream-7B）

**为什么 GSM8K 而不是 PPL：**
- 项目已用 15 轮证明 PPL 不可靠，继续用 PPL 无论结果正负都不可信
- GSM8K 有明确对错判定（准确率），不可被 gaming
- Dream-7B 已有 GSM8K baseline 数字（文献可查），对比方便
- 100 题子集 x 3 seeds 的计算量可控（预计 2-4 GPU 小时）

**执行优先级：**

| 优先级 | 任务 | 预计时间 | 产出 |
|--------|------|---------|------|
| P0 | Dream-7B vanilla GSM8K 100 题 baseline | 1 天 | 对比基准 |
| P0 | 在 MDLM-170M 上验证 IB-Score Level 1 vs Level 3 相关性 | 1-2 天 | Go/No-Go 决策 |
| P1 | IGIR-Lite on GSM8K 100 题 | 3-5 天 | 核心结果 |
| P2 | 若 P1 有效，扩展到 GSM8K 全集 + MATH500 | 5-7 天 | 主实验 |

**关键约束：**
- 只用 benchmark 准确率判断，PPL 仅作参考
- 每个实验必须 3 seeds
- 生成文本必须定性检查（防退化）

---

## 四、如果 PIVOT：推荐新方向

按可行性和发表潜力排序：

### 方向 A：负面结果 + 理论分析论文（最安全，最快产出）

**定位**：EMNLP 2026 main/Findings 或 NeurIPS 2026 workshop

**核心叙事**：
> "Why Training-Free Inference-Time Scaling Fails for Masked Diffusion Language Models: A Systematic Study"

**内容**：
1. 系统性负面结果：7 类方法（TTT 6变体、ReMask-Retry、Best-of-N、TCR、温度退火、熵引导、并行投票）全部失败
2. PPL 不可靠性发现：重复文本 gaming PPL 的机制分析
3. 规模效应：0.6B（退化驱动假象）vs 7B/8B（真正无效）
4. IGIR 的 IB 理论框架（4 个定理）作为理论贡献
5. 行动建议：DLM 推理时扩展需要训练（RemeDi、d1 验证了这一点）

**优势**：
- 现有数据已足够支撑（18+ 轮实验记录）
- 负面结果论文在 EMNLP 有传统（Findings 接受度高）
- 理论框架（IB + 次模 + MCMC 对应）提升学术价值
- 撰写时间约 7-10 天

**风险**：
- NeurIPS/ICML 正文困难（负面结果难进顶会正文）
- 需要与 ReMDM 的正面结果和解（ReMDM 在小模型上有效）

### 方向 B：Training-based 轻量方法（需额外实验，但目标更高）

**定位**：NeurIPS/ICML 2026 正文

**核心 idea**：
- 基于 PRISM 的思路，训练轻量 adapter（LoRA）学习 token-level 质量分数
- 但不修改架构（与 RemeDi 的双流不同），而是在现有 Dream-7B 上 attach LoRA
- 用 RL（类似 d1 的 diffu-GRPO）训练 remasking 决策
- 差异化：RemeDi 需要从头训练双流架构，本方法在任意预训练 DLM 上即插即用

**优势**：
- 解决了"training-free 天花板"问题
- 与 RemeDi 有清晰差异化（即插即用 vs 架构修改）
- 目标会议更高

**风险**：
- 需要 RL 训练基础设施（d1 代码可复用）
- 训练时间不确定（可能需要 1-2 周 GPU 时间）
- 与 d1/wd1/DiSPO 等方法竞争

### 方向 C：DLM 评估方法论文（侧面切入）

**定位**：ACL/EMNLP 2026 main

**核心叙事**：
> "Beyond Perplexity: Rethinking Evaluation of Masked Diffusion Language Models"

**内容**：
- 系统研究 PPL 在 DLM 评估中的失效模式
- 提出 PPL-diversity 联合评估框架
- 跨模型规模（0.6B/7B/8B）、跨评估器（GPT-2/Qwen）的实证分析
- 设计抗 gaming 的 DLM 评估指标

**优势**：工具性贡献，对整个 DLM 社区有价值
**风险**：可能被认为 scope 太窄

---

## 五、时间预算分析

### 当前投入

| 维度 | 已投入 |
|------|--------|
| 迭代轮数 | 20+ 轮 |
| 主要方法 | 7 类（TTT/ReMask/BoN/TCR/退火/熵引导/投票） |
| 模型规模 | 3 个（0.6B/7B/8B） |
| 关键发现 | PPL 不可靠、所有 training-free 方法失败 |

### 边际收益递减分析

```
前 10 轮（iter 1-10）：发现 ReMask-Retry 有效 → 高价值
iter 11-15：发现 PPL gaming → 关键负面发现 → 高价值
iter 16-17：LLaDA-8B 验证 + 评估论文框架 → 中等价值
iter 18-20+：TCR + 熵引导 + 退火 full-scale 失败 → 递减
```

**结论**：项目已进入边际收益递减区。继续同类型实验（不同的 PPL-based 重遮蔽策略）几乎不可能产出新发现。

### 推荐时间预算

| 路径 | 预算 | 截止条件 |
|------|------|---------|
| IGIR on benchmark | **最多 12 天** | Phase 0 失败 → 3 天止损；Phase 1 无改善 → 10 天止损 |
| 负面结果论文撰写 | **7-10 天** | 可与 IGIR 并行准备 outline |
| Training-based 方法 | **15-20 天** | 仅当 IGIR 和负面论文均不满意时启动 |

**总体建议**：从现在起，最多再投入 **3 周（~20 天）**。无论结果如何，20 天后必须进入论文撰写阶段。

---

## 六、发表策略

### 场景分析

#### 场景 1：IGIR on benchmark 有效（准确率提升 >= 1.5%）

- **目标**：NeurIPS 2026 / ICML 2026 正文
- **论文结构**：方法论文（IGIR）+ 理论贡献（4 个定理）+ 系统实验
- **独特卖点**：唯一同时满足 training-free + 无验证器 + 理论保证 + 抗退化的方法
- **补充材料**：18 轮迭代的负面结果作为 motivation（"为什么需要 IGIR"）
- **概率评估**：25%（基于 20+ 轮失败的先验）

#### 场景 2：IGIR 也失败（最可能场景）

- **目标**：EMNLP 2026 main/Findings（deadline 通常在 6 月）
- **论文标题方向**：
  - "The Ceiling of Training-Free Inference Scaling for Masked Diffusion LMs"
  - "When Remasking Doesn't Help: A Systematic Study of Test-Time Compute for dLLMs"
- **核心贡献打包**：
  1. **系统性负面结果**（7+ 方法，3 个模型规模）— 对社区有警示价值
  2. **PPL 不可靠性发现** — 实用贡献，影响所有用 PPL 评估 DLM 的工作
  3. **IB 理论框架**（即使方法无效，理论本身有独立价值）— 学术贡献
  4. **最优重遮蔽比例的闭式界**（定理 2）— 为未来工作提供定量指导
- **叙事策略**："负面结果 + 理论分析 + 方法论贡献"三位一体
- **概率评估**：60%

#### 场景 3：转向 training-based 方法

- **目标**：NeurIPS 2026 正文
- **风险**：时间紧张，与 d1/wd1/RemeDi 竞争
- **概率评估**：15%（仅当前两个场景都不满意时）

### 负面结果包装策略

1. **不要把论文写成"我们试了很多方法都失败了"**。这读起来像实验报告，不像论文。

2. **正确的叙事框架**：
   - 提出一个清晰的研究问题："Training-free 推理时扩展能否提升 DLM 生成质量？"
   - 系统性实验回答这个问题："在 X 个方法、Y 个模型、Z 个评估维度下，答案是否定的"
   - 分析根因："为什么失败？"（DLM 去噪过程的信息论极限、模型校准度不足等）
   - 理论框架："IB 分析揭示了什么条件下重遮蔽才有望成功"
   - 行动指南："社区应转向 training-based 方法（RemeDi、d1 已验证）"

3. **对标论文**：
   - Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration" — 发现 greedy/beam search 导致退化，推动了 nucleus sampling 的采用
   - 本工作的类似定位：发现 training-free remasking 的系统性失败，推动社区转向 training-based 方法

4. **会议选择**：
   - **EMNLP Findings**（最现实）：接受负面结果和分析性工作
   - **NeurIPS Workshop**（快速发表）：有多个相关 workshop（Efficient ML, Foundation Models）
   - **COLM 2026**（如存在）：语言模型专项会议
   - **ACL 2026 main**（如理论足够强）：需要 IB 框架的完整证明

---

## 七、总结与行动建议

### 即刻行动（今天）

1. 确认 Dream-7B 在 GSM8K 上的 vanilla baseline 准确率（查文献或快速实验）
2. 开始 IGIR Phase 0 的 IB-Score 近似质量验证（MDLM-170M 上）
3. 同步准备负面结果论文的 outline（不浪费已有数据）

### 短期（1-2 周）

4. 完成 IGIR Phase 0-1 的 Go/No-Go 决策
5. 若 Go → Phase 2 全面评测
6. 若 No-Go → 全力转入论文撰写

### 中期（3 周内）

7. 无论哪条路径，第 3 周必须有论文初稿
8. 寻找合适的 deadline（EMNLP 2026 通常 6 月截稿）

### 核心原则

> **不要继续在 PPL 上做实验。要么上 benchmark，要么写论文。**
> 项目最大的风险不是"方法不 work"，而是"无限迭代不产出"。
> 20+ 轮迭代已经产生了足够的数据和洞见来支撑一篇有价值的论文。
> 现在的问题不是"能不能做更好的实验"，而是"如何将已有发现最大化包装"。

---

*战略顾问分析完毕。建议主会话根据此分析制定具体执行计划。*
