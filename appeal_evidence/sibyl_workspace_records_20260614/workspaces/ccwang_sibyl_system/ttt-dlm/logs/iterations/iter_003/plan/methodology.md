# 实验方法论：Denoising-Time Adaptation (DTA)

## 1. 研究目标

基于 18 轮迭代的核心教训（PPL 不可信、remasking 效果有限、跨步信息丢失是根本原因），验证 DTA 方法的有效性：

**核心问题**：在 DLM 去噪过程中加入轻量级 LoRA 参数更新（DTA），能否显著提升推理任务准确率？

具体子问题：
1. DTA 是否在 Countdown/GSM8K 等推理基准上显著优于 vanilla 去噪和纯 remasking？（H1, H2）
2. DTA 的推理时扩展曲线是否不同于 remasking 的饱和行为？（H3）
3. 信息增强谱系（DMI < SCP < DTA）中各层级的边际收益如何？（H4）
4. DTA 与 remasking 策略是否正交互补？（H2）

## 2. 模型与基础设施

### 2.1 目标模型
- **Dream-v0-Instruct-7B**（主模型）：最强开源 DLM
  - 路径: `/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B`
  - Mask Token ID: 151666
  - 采样算法: `origin`
- **LLaDA-8B-Base**（验证模型）：验证跨模型泛化性
  - HuggingFace: `GSAI/LLaDA-8B-Base`

### 2.2 计算资源
- GPU: 4x（GPU 0, 2, 4, 5，cs8000d 服务器）
- Conda 环境: `sibyl_ttt-dlm`
- 远程路径: `/home/ccwang/sibyl_system`

## 3. 方法设计

### 3.1 DTA 核心算法

在 DLM 去噪循环中，每个去噪步后执行：
1. **E-step**（标准去噪）：用 f_{θ+Δθ} 预测 mask token，按置信度/调度揭示部分 token
2. **M-step**（DTA 更新）：对已揭示 token 计算 masked LM 损失，对 LoRA 参数执行 1 步 SGD

关键超参数：
- LoRA rank: r ∈ {2, 4, 8, 16}（默认 r=4）
- LoRA 插入位置: 最后 L 层 FFN（默认 L=2）
- 学习率: η ∈ {1e-5, 5e-5, 1e-4, 5e-4}（默认 1e-4）
- 衰减因子: γ ∈ {0.9, 0.95, 0.99, 1.0}（默认 0.95）
- Warmup: 前 20% 去噪步不执行 DTA
- 去噪总步数: T=128（默认），扫描 {64, 128, 256, 512}

### 3.2 消融基线：信息增强谱系

- **Level 0 (Vanilla)**：标准去噪，无跨步信息
- **Level 1 (DMI)**：Diffusion Memory Injection，上一步 logits 的 softmax 加权 embedding 注入当前步输入，~0 额外开销
- **Level 2 (SCP)**：Self-Contradiction Probing，leave-one-out 检测自矛盾 token，1 次额外前向传播
- **Level 3 (DTA)**：LoRA 在线更新，1 次额外反向传播

### 3.3 对比方法
| 方法 | 类型 | 计算开销 |
|------|------|---------|
| Vanilla | 标准去噪 | 1x |
| ReMDM-conf | Remasking | ~1.5x |
| CORE | 上下文探测 | ~2x |
| RCR | Running confidence | ~1.5x |
| DMI (消融) | Embedding 记忆 | ~1.05x |
| SCP (消融) | 自矛盾探测 | ~1.5x |
| **DTA** | **参数适应** | **~2.5x** |
| **DTA + ReMDM** | **参数适应 + Remasking** | **~3x** |

## 4. 评估体系

### 4.1 主要基准
| 基准 | 任务 | 指标 | 样本数 | 预计时间/样本 |
|------|------|------|--------|-------------|
| **Countdown** | 规划/约束 | 准确率 | 500 | ~10s |
| **GSM8K** | 数学推理 | EM 准确率 | 1319 (全集) | ~15s |
| **MBPP** | 代码生成 | Pass@1 | 500 | ~15s |
| **HumanEval** | 代码生成 | Pass@1 | 164 | ~20s |

### 4.2 辅助指标
- distinct-1/2/3（词汇多样性）
- rep-2/3（n-gram 重复率）
- 计算开销（wall-clock 相对倍数）
- LoRA 参数范数 ||Δθ||_F 轨迹

### 4.3 诊断指标（H9 相关）
- Correction Precision：被 remask 的 token 中确实是错误的比例
- Correction Recall：真正错误的 token 被 remask 的比例
- 轨迹稳定性：相邻去噪步 token 变化率

### 4.4 统计检验
- McNemar test + Bootstrap 95% CI
- 500 题 × 3 seeds (42, 123, 456)
- Bonferroni 校正（多重比较）
- 同时报告准确率 + 多样性指标

## 5. 实验阶段

### Phase 0: 环境与框架（前置）
- 验证 Dream-7B 加载与生成
- 搭建统一评估框架（Countdown, GSM8K, MBPP, HumanEval）

### Phase 1: Vanilla Baselines
- Dream-7B vanilla 在所有基准上的 baseline

### Phase 2: DTA 核心实现与 Pilot
- DTA 原型实现（LoRA 在线更新逻辑）
- Countdown 16 题 pilot 验证数值稳定性和基本有效性
- ReMDM-conf baseline 复现

### Phase 3: 消融基线实现
- DMI 实现（Level 1）
- SCP 实现（Level 2）
- RCR baseline 复现

### Phase 4: DTA + ReMDM 组合
- 实现 DTA + ReMDM-conf 组合
- Countdown pilot 验证正交互补性

### Phase 5: Full-Scale 验证
- Countdown 500 题 × 3 seeds 全方法对比
- GSM8K 全集评估
- MBPP/HumanEval 评估

### Phase 6: 推理时扩展曲线
- 步数扫描 T ∈ {64, 128, 256, 512}
- 对比 DTA vs ReMDM-conf 的扩展行为
- 验证 H3（DTA 不饱和）

### Phase 7: 消融实验
- LoRA rank 消融（r=2, 4, 8, 16）
- 更新频率消融（每步 / 每 2 步 / 每 4 步）
- 衰减因子消融（γ=0.9, 0.95, 0.99, 1.0）
- 插入层数消融（最后 1/2/4 层）
- Warmup 比例消融（0%, 10%, 20%, 30%）

### Phase 8: 跨模型验证与诊断
- LLaDA-8B 上 DTA 验证
- Token 级诊断分析（Correction Precision/Recall）
- 退化检测（H10）

## 6. 可复现性

- 所有实验固定随机种子
- 保存完整生成文本
- 记录 Python/PyTorch/Transformers/PEFT 版本
- 结果保存为 JSON，包含中间数据和超参数
- LoRA 参数轨迹（||Δθ||_F per step）保存为日志

## 7. 论文定位

**目标**: NeurIPS/ICML 主会（8+ 页）

**核心贡献**:
1. **方法**: DTA——首个将 TTT 机制引入 DLM 去噪过程的 training-free 方法
2. **理论**: VDTA 变分框架——证明去噪 + 参数更新等价于 EM 优化
3. **实验**: Dream-7B/LLaDA-8B 上系统性推理基准评测 + 信息增强谱系消融
4. **分析**: Token 级诊断框架，量化 remasking 的信号质量问题

**论文结构**:
- Section 1: Introduction（DLM 跨步信息丢失 → DTA 动机）
- Section 2: Background & Related Work（DLM, TTT, Remasking 方法）
- Section 3: Method（DTA 算法 + VDTA 理论框架）
- Section 4: Experiments（主基准结果 + 扩展曲线 + 组合实验）
- Section 5: Analysis（消融 + 诊断 + 跨模型 + 信息增强谱系）
- Section 6: Discussion（局限性 + 与 18 轮历史的对比）
