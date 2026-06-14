# 方法论审查：DTA 实验的内部效度、外部效度与可复现性

## 总体评估

本研究在实验方法论上展现了较强的基础设施能力（多 seed 验证、pilot-then-full-scale 流程、统一评估框架），但存在若干系统性方法学缺陷，可能导致核心结论——无论是"方法无效"还是"方法有效"——都不够可靠。以下从六个维度逐一审查。

---

## 1. Baseline 公平性审查

### 1.1 Baseline 调优不对等（严重）

DTA 经历了 4 轮超参数迭代（v1→v4），最终配置为 lr=5e-4, AdamW, gamma=0.95, mask 20%。然而，对比方法的调优程度高度不对等：

- **ReMDM-conf**: 经过至少 1 轮调参（第一版 remask_ratio=0.2 导致 0% 准确率，修正为 0.1 + stop_frac=0.8）
- **RCR**: 经过 2 轮调参（第一版极度激进 188 tokens/样本，修正为 threshold_scale=0.7, cap=0.15），但 pilot 仍为 0%
- **DMI**: 混合系数 alpha=0.3 似乎为固定值，未见消融
- **CORE**: task_plan.json 中列为对比方法，但实际结果中未出现

**核心问题**: DTA 的调参在 16 样本 pilot 上进行，这个样本量不足以可靠评估任何超参数配置。v4 的"合理范围"判断基于 LoRA 范数和 loss 值而非下游准确率——这意味着 DTA 的超参数可能根本没有被有效优化。

**建议**: 对所有方法执行同等粒度的超参数搜索，或者至少明确报告每个方法的搜索预算（配置数 x 样本数）。如果 DTA 只测了 4 个配置 x 16 样本，而 TCR 测了 6 个配置 x 16 样本，这种不对等应在论文中说明。

### 1.2 CORE 缺失（中等）

提案中列出 CORE（Context-Robust Remasking）为关键对比方法，但在所有实验结果中未见 CORE 的数据。这是一个与 DTA 开销相当（~2x）的 remasking 方法。缺少 CORE 使 DTA 的定位——"唯一同时满足零训练+参数级记忆+理论保证"——无法完整验证。

### 1.3 HEX 缺失（低风险但需说明）

HEX（多调度集成）也在对比表中但无实验数据。作为"纯 token 空间"方法的上界，缺少 HEX 使信息增强谱系的对比不完整。

---

## 2. 指标适当性审查

### 2.1 PPL 作为主指标的根本矛盾（致命）

前序迭代（6 种策略 x 303 样本）的核心发现之一是"PPL 不可信"，但该结论本身完全依赖 PPL 得出。这是一个逻辑自矛盾：

- Vanilla PPL(mean)=30.1 vs PPL(median)=16.23 → 分布极端右偏
- Entropy PPL(mean)=41.8 vs PPL(median)=16.15 → 存在极端异常值
- 使用 GPT-2 (124M) 评估 7B 模型输出 → 评估模型容量严重不足

**方法论判断**: GPT-2 PPL 不适合作为 DLM 推理时改进的主评估指标。交叉架构 PPL 在模型能力差距巨大时（124M vs 7B），更多反映的是 GPT-2 的建模能力而非生成文本的质量。

**建议**:
1. 用同规模模型（如 GPT-2 XL 1.5B 或 Llama-3-8B）计算 PPL
2. 将 PPL 降级为辅助指标，以任务准确率（Countdown, GSM8K, MBPP）作为主指标
3. 如果保留 PPL，必须报告截断 PPL（去除最高 5% 异常值）以减少右偏影响

### 2.2 准确率作为主指标是正确选择（正面）

新迭代（DTA 方向）改用 Countdown/GSM8K/MBPP 的准确率作为主指标，这是方法论上的重要改进。准确率有明确的正确/错误判定标准，不受评估模型容量限制。

### 2.3 多样性指标过于粗糙（轻微）

Distinct-1/2/3 和 Rep-2/3 作为退化检测指标是合理的，但不足以捕捉更细微的文本质量变化。例如：
- DTA 可能使模型生成"更安全但更无趣"的文本，distinct-n 不会下降但语义多样性降低
- Countdown 任务中，多样性指标几乎无意义——我们关心的是答案正确性，而非文本风格

**建议**: 对 Countdown 等结构化任务，完全用准确率替代多样性指标。

### 2.4 LLM Judge 的解析偏差（中等）

怀疑论者已指出 idx=64 的 "TIE" 被误判为 vanilla win。30 对比较中 0 个 tie 在统计上极不合理。此外：
- LLM judge 仅在前序迭代（开放文本生成）中使用，新迭代（DTA）未使用
- 如果在 DTA 迭代中引入 LLM judge，需要修复解析逻辑

---

## 3. 评估协议审查

### 3.1 数据泄漏检查（通过）

- Countdown 为自生成的数学约束问题，不存在训练集泄漏风险
- GSM8K test set 与 Dream-7B 训练数据的重叠未检查，但 GSM8K 是标准基准，这是该领域的通行做法
- MBPP sanitized test set 同上

**判断**: 无明显数据泄漏风险。

### 3.2 Pilot-to-Full-Scale 不一致是方法论警钟（严重）

| 方法 | Pilot PPL 改善 | Full PPL 改善 | 差距 |
|------|---------------|---------------|------|
| entropy_r20_mean | -24.9% | -0.5% | 24.4pp |
| tcr_r30_s32 | -22.9% | -2.8% | 20.1pp |
| anneal_lin_06_02 | -16.5% | +8.1% | 24.6pp |

这不仅仅是"小样本噪声"——pilot 和 full-scale 的差距系统性地在 20pp 以上。这暗示 16 样本 pilot 的样本选择可能存在系统性偏差（"均为简单科学问题"），不是简单的随机波动。

**方法论影响**: 新迭代中 DTA 的所有 pilot 也是 16 样本 x 1 seed。如果 DTA 在 pilot 中表现为 6.2%（vs vanilla 12.5%），这个结论与 full-scale 结果的关系同样不可靠。DMI 在 pilot 中 0%（task_5a pilot）但在 full-scale 中 9.3%——这完全逆转了方向。

**建议**:
1. pilot 至少应使用 50 样本 + 2 seeds
2. pilot 样本应从全集中随机抽取，而非手动挑选
3. 在论文中明确报告 pilot 与 full-scale 的一致性分析

### 3.3 超参数选择偏差（中等）

DTA 的默认超参数（lr=5e-4, rank=4, gamma=0.95）在 16 样本 pilot 上基于"LoRA 范数在合理范围"而非"下游任务准确率"选定。这意味着：
- 最优超参数可能未被发现
- 如果 DTA 在 full-scale 中表现不佳，可能是超参数问题而非方法问题

消融实验（task_7a/7b）在 pilot 规模（16 样本）下全部无法区分配置优劣（所有 rank 都是 6.2%）。**这些消融在统计上没有信息量**，不应用于任何结论。

### 3.4 跨阶段 Baseline 不一致（轻微但需注意）

- Pilot baseline vanilla_t04: PPL(med)=17.15, Countdown 准确率 18.8%
- Full-scale baseline vanilla: PPL(med)=16.23, Countdown 准确率 4.7%

Full-scale 的 Countdown 准确率（4.7%）远低于 pilot（18.8%）。这可能因为：
1. Pilot 使用自选的 16 个问题 vs full-scale 使用 500 个标准化问题
2. Pilot 的问题可能更简单
3. Countdown 问题集本身的难度分布可能高度不均

**这进一步印证了 pilot 样本选择偏差的存在。**

---

## 4. 消融完整性审查

### 4.1 信息增强谱系（部分完成）

提案设计了四层消融谱系：
| Level | 方法 | Full-Scale 状态 |
|-------|------|----------------|
| 0 | Vanilla | 完成（4.7%）|
| 1 | DMI | 完成（9.3%）|
| 2 | SCP | 进行中（~8.4% interim）|
| 3 | DTA | 待完成 |
| 3+ | DTA+ReMDM | 待完成 |

**发现**: DMI 的 2x 改善（9.3% vs 4.7%）是最出人意料的正面结果。然而 DMI 在 pilot 中的 0% 准确率与 full-scale 的 9.3% 完全矛盾——这再次证明 pilot 结果不可信。

### 4.2 DTA 核心消融（均不可靠）

已完成的 pilot 消融（16 样本）：
- **LoRA rank** (task_7a): r={2,4,8,16} 全部 6.2%，无信息量
- **衰减因子** (task_7b): 未在已读文件中看到 full-scale 结果
- **更新频率** (task_7c): 未在已读文件中看到 full-scale 结果
- **插入层数** (task_7d): 未在已读文件中看到 full-scale 结果

**判断**: 所有消融在 pilot 规模下都无法得出可靠结论。Full-scale 消融（200 样本）是最低要求。

### 4.3 缺失的关键消融

1. **Warmup 比例消融**: 提案中提到扫描 0%/10%/20%/30%，但未见任何结果。Warmup 策略直接影响 DTA 何时开始更新，是最可能影响效果的超参数之一。
2. **学习率消融**: DTA v1→v4 调参涉及 lr 从 1e-4 到 5e-3 再到 5e-4，但这不是系统性消融——每次只测一个 lr。需要在 full-scale 上系统扫描。
3. **M-step 设计消融**: v3 的 self-consistency loss 与 v4 的 mask-and-predict loss 是完全不同的自监督信号。论文需要对此做严肃消融，解释为什么 mask-and-predict 优于 self-consistency。

---

## 5. 可复现性评估

### 5.1 正面因素

- **随机种子**: 3 seeds (42, 123, 456)，覆盖基本的可重复性
- **硬件规格明确**: NVIDIA RTX PRO 6000 Blackwell (98GB), cs8000d 服务器
- **软件版本记录**: Python 3.12, PyTorch 2.8.0, Transformers 4.51.3, PEFT 0.18.1
- **超参数完整记录**: 所有方法的关键超参数在 pilot summary 中有详细记录
- **生成文本保存**: GSM8K 结果中保存了完整的生成文本和预测值

### 5.2 可复现性风险

1. **实验代码散布**: 实验代码分散在 20+ 个 Python 文件中（task_2a_dta_pilot.py, task_5a_countdown_all_methods.py 等），没有统一的运行入口或 README。独立研究者难以重现实验流程。

2. **DTA 实现细节不透明**:
   - "mask ~20% revealed tokens, predict from remaining context" 的具体实现——是随机 mask 还是按某种策略选择？
   - AdamW 的 beta1/beta2/epsilon 参数未记录
   - grad_clip=1.0 的应用方式（全局范数 vs 逐参数）未说明

3. **Countdown 数据集版本**: "16 个自生成 Countdown 问题" vs "500 个 Countdown 问题"——这两个数据集的来源和构造方式不同。Pilot 使用"自生成"问题，full-scale 来源未明确。

4. **Dream-7B 去噪实现**: 是使用 Dream 官方仓库的 `origin` 采样算法还是自行实现？如果基于官方实现，是否有修改？

5. **DMI/SCP 的具体实现**: 这两个方法是本研究新提出的消融基线，没有现有论文可参考。如果不开源代码，无法复现。

### 5.3 建议

- 整合所有实验代码为统一的 CLI 工具，提供 `run_experiment.py --method dta --benchmark countdown --samples 500 --seed 42` 式的接口
- 在论文附录中提供 DTA M-step 的完整伪代码（包括 mask 选择策略、梯度裁剪方式等）
- 开源 DMI 和 SCP 的实现，作为社区可复用的 DLM 消融基线

---

## 6. 方法论强化建议

### 6.1 高优先级（阻塞论文可信度）

1. **完成 DTA 和 DTA+ReMDM 的 full-scale Countdown 评估**: 当前 interim 结果只有 Vanilla/ReMDM/RCR/DMI 四个方法。没有 DTA 的 full-scale 数据，论文的核心主张无法验证。

2. **报告效应量和统计功效**:
   - Cohen's d（或非参数等价物）量化效应大小
   - Post-hoc 功效分析：当前 500×3 样本能检测到多小的效应？
   - 如果功效分析显示只能检测 >5% 的绝对改善，那 DMI 的 +4.6pp（从 4.7% 到 9.3%）可能刚好在检测边界

3. **将 PPL 降级为辅助指标**: 在所有分析中以任务准确率为主指标，PPL 仅作为文本退化的辅助监控。

### 6.2 中优先级（提升论文质量）

4. **统一 pilot 设计**: 未来所有 pilot 使用 50 样本 × 2 seeds，从全集随机抽样。在论文中明确报告 pilot 设计并讨论其局限性。

5. **GSM8K 扩展验证**: 当前仅有 50 题 × 3 seeds 的小规模数据。Entropy 的 +6.7pp 绝对改善（34% → 40.7%）是关键信号，需要在 200+ 题上确认。注意这是前序迭代的 remasking 方法在任务准确率上的信号，与新迭代 DTA 方向相关但需区分。

6. **跨模型验证（LLaDA）的样本量扩大**: 当前 LLaDA pilot 仅 16 样本，且 DTA 在 Countdown 和 GSM8K 上都不如 vanilla。需要在 ≥100 样本上确认——16 样本下 DMI 也是 0%，但 full-scale 变成了 9.3%。

### 6.3 低优先级（锦上添花）

7. **anneal_fix 的完整统计检验**: anneal_fix_lin_08_03 (PPL_median=14.87) vs vanilla (16.23) 的配对检验。如果显著，说明 temperature schedule 是一个被低估的改进维度。

8. **文本长度控制**: 报告各方法的平均生成长度，做长度标准化的分析。

---

## 总结判断

### 方法论成熟度：中等偏上（65/100）

**优势**:
- 多 seed 验证框架
- 信息增强谱系的系统性消融设计
- 从 PPL 转向任务准确率的方向校正
- 统计检验的意识（McNemar + Bootstrap CI + Bonferroni）

**核心缺陷**:
- Pilot 设计存在系统性偏差（16 样本、可能偏向简单问题），导致 pilot→full-scale 结果方向逆转
- DTA 的超参数优化在统计上不可靠的样本量上进行
- 核心消融（rank/gamma/频率/层数）全部停留在无信息量的 pilot 阶段
- DTA 本身的 full-scale 结果尚未完成，论文核心主张无法验证
- PPL 指标的逻辑自矛盾未解决

**关键风险**:

DTA 可能因为超参数选择不当（在 16 样本上"调参"）而表现不佳，被误判为"方法无效"。DMI 的案例（pilot 0% → full-scale 9.3%）证明 pilot 结果可以完全误导。在 DTA full-scale 结果出来之前，任何关于 DTA 有效性或无效性的结论都不应被采纳。

**最终建议**: 暂缓对 DTA 的效果做定性判断。优先完成 full-scale 实验（DTA/DTA+ReMDM on Countdown 500×3seeds），然后在 full-scale 数据上重新评估所有假设。如果 DTA full-scale 仍不如 vanilla，需要严肃考虑：(1) 超参数搜索是否充分，(2) M-step 的自监督信号设计是否存在根本问题——mask-and-predict 可能让模型学到 token 共现统计而非推理能力。
