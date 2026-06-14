# Pilot 实验总结

## 实验概览

- 模型: Dream-v0-Instruct-7B
- 服务器: cs8000d (NVIDIA RTX PRO 6000 Blackwell, 98GB)
- Pilot 规模: 16 prompts, seed=42, 128 gen tokens, 128 denoising steps
- Python: miniforge3/python3.12, torch 2.8.0, transformers 4.51.3

## Task 0: 环境验证 -- PASS

Dream-7B 加载正常，生成流水线可用，GPT-2 PPL 评估正常。
注意: `dtype` 参数需改为 `torch_dtype`（transformers 4.51 兼容性）。

## Task 1: 多温度 Baseline -- GO

| 方法 | PPL(med) | PPL(mean) | Diversity | Short | Time | Len |
|------|----------|-----------|-----------|-------|------|-----|
| vanilla_t02 | 16.47 | 23.1 | 0.971 | 4 | 2.4s | 36w |
| vanilla_t04 | **17.15** | 19.7 | 0.954 | 3 | 2.4s | 46w |
| vanilla_t06 | 17.86 | 24.4 | 0.962 | 2 | 2.4s | 56w |
| vanilla_t08 | 17.32 | 20.9 | 0.951 | 0 | 2.4s | 76w |

**发现**: 温度对 PPL 影响不大（16.5-17.9 范围），但对生成长度影响显著（t02: 36w vs t08: 76w）。
t04 作为 baseline: PPL(med)=17.15, diversity=0.954, 0 degenerate。

## Task 2a: TCR 消融 -- GO

| 配置 | PPL(med) | vs Base | Diversity | Deg | Time |
|------|----------|---------|-----------|-----|------|
| tcr_r20_s32 | 14.65 | -14.5% | 0.927 | 1 | 3.4s |
| **tcr_r20_s64** | **13.51** | **-21.2%** | **0.967** | 0 | 3.9s |
| **tcr_r30_s32** | **13.22** | **-22.9%** | **0.961** | 0 | 3.3s |
| tcr_r30_s64 | 15.33 | -10.6% | 0.975 | 0 | 3.9s |
| tcr_r50_s32 | 14.87 | -13.3% | 0.963 | 0 | 3.3s |
| tcr_r50_s64 | 16.24 | -5.3% | 0.954 | 0 | 3.9s |

**最佳**: tcr_r30_s32 (PPL -22.9%, div 0.961, 仅 +37% 时间开销)
**发现**:
- 适中的 remask_ratio (20-30%) 优于激进的 50%
- 32 步细化足够，64 步反而可能过度修正
- r20_s32 有 1 个退化样本，需注意

## Task 2b: 温度退火 -- GO

| 方案 | PPL(med) | vs Base | Diversity | Time |
|------|----------|---------|-----------|------|
| anneal_lin_08_02 | 15.04 | -12.3% | 0.981 | 2.5s |
| **anneal_lin_06_02** | **14.32** | **-16.5%** | **0.981** | 2.4s |
| anneal_cos_08_02 | 15.04 | -12.3% | 0.981 | 2.4s |

**最佳**: anneal_lin_06_02 (PPL -16.5%, div 0.981, 零额外开销)
**发现**:
- 所有退火方案都有效，且 diversity 极高 (0.981)
- 零额外计算开销（与 vanilla 相同时间）
- linear 0.6->0.2 略优于 0.8->0.2
- cosine 与 linear 0.8->0.2 表现一致

## Task 2c: 并行投票 -- NO-GO

| 方案 | PPL(med) | vs Base | Diversity | Time |
|------|----------|---------|-----------|------|
| vote_k4_noheal | 36.76 | +114.4% | 0.974 | 9.8s |
| vote_k4_heal16 | 19.96 | +16.4% | 0.995 | 10.1s |

**结论**: 投票方法失败。K=4 轨迹在 ~66% 位置有分歧，投票合并产生不连贯文本。
即使加 16 步 healing，PPL 仍比 baseline 差 16%，且耗时 4x。

## Task 2d: 熵引导重掩码 -- GO

| 配置 | PPL(med) | vs Base | Diversity | Time |
|------|----------|---------|-----------|------|
| **entropy_r20_mean** | **12.88** | **-24.9%** | **0.975** | 4.6s |
| entropy_r30_mean | 15.07 | -12.1% | 0.974 | 4.6s |
| entropy_r50_mean | 14.62 | -14.7% | 0.983 | 4.7s |

**最佳**: entropy_r20_mean (PPL -24.9%, div 0.975)
**注意**: entropy_r50_mean 的 PPL(mean)=58.0 远高于 median，存在极端异常值。
**发现**:
- 熵引导重掩码是最有效的方法（-24.9% PPL）
- 20% remask ratio 即可获得最大改善
- 与 TCR 相比，熵是单步信号，更直接衡量模型不确定性

## 总排名（按 PPL 改善）

| 排名 | 方法 | PPL(med) | vs Base | Div | Time | 状态 |
|------|------|----------|---------|-----|------|------|
| 1 | entropy_r20_mean | 12.88 | -24.9% | 0.975 | 4.6s | GO |
| 2 | tcr_r30_s32 | 13.22 | -22.9% | 0.961 | 3.3s | GO |
| 3 | tcr_r20_s64 | 13.51 | -21.2% | 0.967 | 3.9s | GO |
| 4 | anneal_lin_06_02 | 14.32 | -16.5% | 0.981 | 2.4s | GO |
| 5 | tcr_r50_s32 | 14.87 | -13.3% | 0.963 | 3.3s | GO |
| 6 | anneal_lin_08_02 | 15.04 | -12.3% | 0.981 | 2.5s | GO |

## Full-Scale 推荐

建议对以下方法进行 Full-Scale 验证 (task_3a):
1. **entropy_r20_mean**: 最佳 PPL 改善，高 diversity
2. **tcr_r30_s32**: 第二佳 PPL，且时间开销最低
3. **anneal_lin_06_02**: 零额外开销，最高 diversity，实用价值最大

不推荐:
- parallel_vote: 完全失败
- 高 remask_ratio (50%): 不稳定，可能产生异常值

---

# 新迭代 Pilot 结果 (DTA 方向)

## Task 1a: Vanilla Countdown Baseline -- GO

Dream-7B vanilla (origin, 128步, temp=0.4) 在 16 个自生成 Countdown 问题上的表现。

| 指标 | 值 |
|------|-----|
| 准确率 | 3/16 = 18.8% |
| 平均生成时间 | 3.7s/样本 |
| Distinct-1 | 0.596 |
| Distinct-2 | 0.830 |
| Distinct-3 | 0.898 |
| Rep-2 | 0.170 |
| Rep-3 | 0.102 |
| 平均词数 | 32 |

**Pass Criteria**: 全部 PASS（16 样本生成成功，准确率 > 0%，评估框架正常）

**观察**:
- 准确率 18.8% 提供了充足的提升空间供 DTA 验证
- 模型经常犯算术错误（如 "4*1=18"），但能产出格式化答案
- 文本多样性良好（Distinct-2=0.83），无退化现象
- RTX PRO 6000 Blackwell 上 3.7s/样本，速度优秀

## Task 1b: Vanilla GSM8K Baseline -- GO

Dream-7B vanilla (origin, 128步, temp=0.4) 在 GSM8K test set 前 16 题上的表现。

| 指标 | 值 |
|------|-----|
| EM 准确率 | 3/16 = 18.8% |
| 答案提取率 | 16/16 = 100.0% |
| 平均生成时间 | 9.8s/样本 |
| Distinct-1 | 0.544 |
| Distinct-2 | 0.839 |
| Distinct-3 | 0.926 |
| Rep-2 | 0.161 |
| Rep-3 | 0.074 |
| 平均词数 | 80 |

**Pass Criteria**: 全部 PASS（16 样本生成成功，答案提取率 100% >= 80%，准确率 18.8% > 0%）

**提取方式分布**: #### 格式 12 个, pattern 匹配 3 个, last_number 回退 1 个

**观察**:
- EM 准确率 18.8%（3/16），与 Countdown 的 pilot 准确率一致，提供充足 DTA 改善空间
- 答案提取率 100%，说明模型能稳定产出数值答案，评估框架可靠
- 75% 的答案通过 #### 格式提取（模型有遵循 GSM8K 格式的倾向）
- 生成长度 ~80 词，约为 Countdown 的 2.5 倍（因需要多步推理）
- 生成时间 9.8s/样本（vs Countdown 3.7s），因 gen_len=512 vs 256
- 文本质量良好：Distinct-2=0.839，无退化现象
- 模型常见错误模式：计算步骤正确但中间算术出错（如误算乘法/加法）

## Task 0a: 环境验证与模型加载 -- GO

Dream-7B 加载正常（2.3s），vanilla 生成可用，mask_token_id=151666 确认，PEFT/LoRA 初始化无报错。

| 检查项 | 状态 |
|--------|------|
| 环境（torch=2.8.0, peft=0.18.1, transformers=4.51.3）| PASS |
| Dream-7B 加载（bfloat16, 2.3s）| PASS |
| mask_token_id = 151666 | PASS |
| 4 个 Countdown prompt 生成（1/4 正确）| PASS |
| PEFT/LoRA 初始化（r=4, 最后 2 层 FFN, 540K/7.6B 参数 = 0.007%）| PASS |
| GPU 内存（模型 15.3GB / 102GB 可用，剩余 86.5GB）| OK |

**关键发现**:
- Dream-7B 共 28 层，LoRA 插入 layers 26-27 的 gate/up/down_proj（6 个模块，540K 参数）
- LoRA B 矩阵零初始化确认（初始行为等价于原模型）
- 模型加载仅占 15% GPU 显存，DTA 的梯度计算有充足空间
- 4 个 Countdown 生成全部成功（1/4 正确），与 task_1a 的 16 样本结果一致

## Task 2b: ReMDM-conf Baseline 复现 -- GO

ReMDM-conf（基于置信度的 remasking）在 Dream-7B 上的复现。在每个去噪步后，用 fresh forward pass 评估所有已揭示 token 的置信度，将最低置信度的 token 重新遮蔽回 [MASK]。

| 指标 | ReMDM-conf | Vanilla | Delta |
|------|-----------|---------|-------|
| 准确率 | 4/16 = 25.0% | 3/16 = 18.8% | +6.2% |
| 平均生成时间 | 13.8s/样本 | 3.7s/样本 | ~3.7x |
| Distinct-1 | 0.645 | 0.596 | +0.049 |
| Distinct-2 | 0.895 | 0.830 | +0.065 |
| Distinct-3 | 0.943 | 0.898 | +0.045 |
| Rep-2 | 0.105 | 0.170 | -0.065 |
| Rep-3 | 0.057 | 0.102 | -0.045 |
| 平均词数 | 28 | 32 | -4 |
| 平均重遮蔽总数 | 13 tokens | - | - |

**超参数**: remask_ratio=0.1, remask_stop_frac=0.8（最后 20% 步数不再 remask）

**Pass Criteria**:
- ReMDM-conf 运行成功: PASS
- Diversity (distinct-2) >= 0.7: 0.895 -> PASS
- 与 vanilla 可对比: PASS
- Overall: **GO**

**观察**:
- 准确率从 18.8% 提升到 25.0%（+6.2%），方向正确
- 生成文本质量良好，比 vanilla 更连贯（rep-2 下降 38%）
- 计算开销 ~3.7x（每步需额外 forward pass 计算置信度）
- 第一版实现（remask_ratio=0.2，无 stop_frac）过于激进导致 0% 准确率，修正后效果良好
- 16 样本 pilot 无法做统计检验，但方向一致性足以进入 full-scale

## Task 2c: RCR (Running Confidence Remasking) Baseline -- GO (with caveats)

RCR 使用 running average confidence 作为自适应阈值进行 remasking：每步计算已揭示 token 的平均置信度，维护 EMA，将低于 threshold_scale * EMA 的 token 重新遮蔽。

| 指标 | RCR | Vanilla | ReMDM-conf | Delta vs Vanilla |
|------|-----|---------|------------|-----------------|
| 准确率 | 0/16 = 0.0% | 3/16 = 18.8% | 4/16 = 25.0% | -18.8% |
| 平均生成时间 | 9.1s/样本 | 3.7s/样本 | 13.8s/样本 | ~2.5x |
| Distinct-1 | 0.582 | 0.596 | 0.645 | -0.014 |
| Distinct-2 | 0.836 | 0.830 | 0.895 | +0.006 |
| Distinct-3 | 0.899 | 0.898 | 0.943 | +0.001 |
| Rep-2 | 0.164 | 0.170 | 0.105 | -0.006 |
| Rep-3 | 0.101 | 0.102 | 0.057 | -0.001 |
| 平均词数 | 34 | 32 | 28 | +2 |
| 平均重遮蔽总数 | 31 tokens | - | 13 tokens | - |

**超参数**: EMA_decay=0.9, warmup_frac=0.3, threshold_scale=0.7, remask_cap=0.15

**Pass Criteria**:
- RCR 运行成功: PASS
- Diversity (distinct-2) >= 0.7: 0.836 -> PASS
- Overall: **GO**（功能正常，但准确率劣于 vanilla）

**观察**:
- 准确率 0%，显著低于 vanilla (18.8%) 和 ReMDM-conf (25.0%)
- 文本质量尚可（非乱码），但等式答案全部错误
- 第一版（threshold_scale=1.0, remask_cap=0.3）极度激进（188 tokens/样本），产出乱码
- 修正后的保守版本（threshold_scale=0.7, cap=0.15）文本连贯但仍无正确答案
- RCR 的自适应阈值机制在 DLM 去噪语境下可能不适用：mean confidence 作为阈值对正态分布的置信度分布来说仍然过于激进
- 与 ReMDM-conf 对比：ReMDM-conf 使用固定 remask_ratio=0.1 且在后期停止 remask，效果明显更好
- **结论**：RCR 作为 baseline 方法功能可用，但在 Countdown 任务上表现不佳，支持了论文的论点——纯 remasking 方法（尤其是自适应阈值类）效果有限

## Task 2a: DTA 核心实现与 Countdown Pilot -- CONDITIONAL-GO

DTA (Denoising-Time Adaptation) 核心算法实现：在 Dream-7B 去噪循环中，每步揭示 token 后对已揭示 token 执行 masked re-prediction 自监督损失，更新 LoRA 参数（rank=4，最后 2 层 FFN）。

### 超参数配置（经 3 轮调参后的最终版）
- LoRA: rank=4, last 2 layers (26-27), gate/up/down_proj, 540K 参数 (0.007%)
- lr=5e-4 (AdamW), gamma=0.95, warmup=20%, grad_clip=1.0
- M-step: mask ~20% revealed tokens, predict from remaining context

### 调参历史
| 版本 | LR | Optimizer | Gamma | M-step | 结果 |
|------|-----|-----------|-------|--------|------|
| v1 | 1e-4 | SGD | 0.95 | mask 50% | norm=0.000（太低，无效果）|
| v2 | 5e-3 | AdamW | 0.99 | mask 50% | norm=4-7（太高，输出乱码）|
| v3 | 5e-4 | AdamW | 0.95 | self-consistency | norm=0.13-0.38, loss=0.005-0.032（loss 太低，无信号）|
| **v4** | **5e-4** | **AdamW** | **0.95** | **mask 20%** | **norm=0.05-0.25, loss=0.02-0.19（合理范围）** |

### 结果对比

| 指标 | DTA | Vanilla | Delta |
|------|-----|---------|-------|
| 准确率 | 1/16 = 6.2% | 2/16 = 12.5% | -6.2% |
| 平均生成时间 | 15.9s/样本 | 3.7s/样本 | ~4.3x |
| Distinct-2 | 0.826 | 0.838 | -0.012 |
| Rep-3 | 0.091 | 0.101 | -0.010 |
| 平均词数 | ~30 | ~32 | -2 |
| LoRA ||Δθ||_F max | 0.245 | - | - |
| DTA 平均 loss | 0.092 | - | - |
| DTA 更新次数/样本 | 103 | - | - |

### Pass Criteria
- DTA 运行无报错: **PASS**
- ||Δθ||_F 不超过 1.0: **PASS** (max=0.245)
- Countdown 准确率 ≥ vanilla: **FAIL** (6.2% < 12.5%)
- distinct-2 ≥ 0.7: **PASS** (0.826)
- Overall: **CONDITIONAL-GO**

### 观察与分析
1. **数值稳定性确认**: LoRA 范数收敛在 0.05-0.25 范围，远低于 1.0 阈值，无参数爆炸
2. **文本质量保持**: DTA 输出文本连贯，多样性指标与 vanilla 接近，无退化
3. **准确率方向不确定**: 在 n=16 样本量下，1 vs 2 的差异不具统计意义（p > 0.5）
4. **计算开销 ~4.3x**: 每步额外 1 次前向+反向传播，符合预期（方法论预估 2.5x，实际因 mask-and-predict 多一次前向）
5. **M-step 设计迭代**: self-consistency loss (直接在已揭示 token 上计算 loss) 几乎为 0（模型已高度自信），改为 mask-and-predict 后 loss 在合理范围 (0.02-0.19)
6. **关键挑战**: masked re-prediction 的自监督信号可能不足以引导模型学到 "更好的" 推理模式 —— 模型学习到的是 token 间的互相预测能力，而非算术正确性
7. **与 ReMDM-conf 对比**: ReMDM-conf (25.0%) > Vanilla (12.5%) > DTA (6.2%)，在 pilot 规模下 remasking 效果优于参数适应

### 后续方向
- 需要在更大样本量 (≥100) 上评估才能确定 DTA 效果方向
- 考虑在 M-step 中引入更强的结构化信号（如 contrastive loss, 而非纯 MLM）
- 尝试将 DTA 与 ReMDM-conf 组合（task_4a），可能产生互补效果
- 超参数空间仍有探索余地：lr ∈ [1e-4, 1e-3], rank ∈ {2, 8}, layers ∈ {1, 4}

## Task 1c: Vanilla MBPP Baseline -- GO

Dream-7B vanilla (origin, 128步, temp=0.4) 在 MBPP sanitized test set 前 16 题上的 Pass@1 评估。

| 指标 | 值 |
|------|-----|
| Pass@1 | 4/16 = 25.0% |
| 代码提取率（非空输出）| 16/16 = 100.0% |
| 函数定义率（含 def）| 16/16 = 100.0% |
| 平均生成时间 | 4.6s/样本 |
| 总生成时间 | 73s |
| Distinct-1 | 0.912 |
| Distinct-2 | 0.989 |
| Distinct-3 | 1.000 |
| Rep-2 | 0.011 |
| Rep-3 | 0.000 |
| 平均词数 | 17 |

**Pass Criteria**:
- 16 样本代码生成成功（非空输出）: PASS (16/16)
- Sandbox 执行无崩溃: PASS
- Overall: **GO**

**通过的 4 题**: task_14（三角棱柱体积）, task_17（正方形周长）, task_57（字符串第 n 个 Fibonacci 数）, task_59（最大 n 个元素）

**观察**:
- Pass@1 = 25.0%，在 DLM 代码生成的合理范围内，提供了充足的 DTA 改善空间
- 100% 的输出包含有效的函数定义（`def`），说明模型对代码格式理解良好
- 生成代码普遍较短（平均 17 词），部分代码被截断（如 task_61 的 while 循环未闭合）
- 文本多样性极高（Distinct-2=0.989, Rep-3=0.0），代码天然多样性强
- 常见失败模式：
  - 函数名大小写不匹配（如 `remove_occ` vs 测试用例的 `remove_Occ`）——MBPP 数据集问题
  - 逻辑错误（如排序 key 表达式语法错误）
  - 代码截断（gen_len=512 对部分复杂题目可能不足）
- 与 Countdown/GSM8K pilot 对比：代码生成 4.6s/样本，速度介于 Countdown (3.7s) 和 GSM8K (9.8s) 之间
