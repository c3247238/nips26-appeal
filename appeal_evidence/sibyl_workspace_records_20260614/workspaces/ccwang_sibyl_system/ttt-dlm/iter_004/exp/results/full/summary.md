# Full-Scale 验证结果 (Task 3a)

## 实验配置

- **模型**: Dream-v0-Instruct-7B (掩码扩散语言模型)
- **服务器**: cs8000d, NVIDIA RTX PRO 6000 Blackwell (98GB)
- **规模**: 101 prompts x 3 seeds (42, 123, 456) = 303 样本/方法
- **生成参数**: 128 tokens, 128 去噪步, origin 采样算法
- **评估**: GPT-2 (124M) 交叉架构 PPL
- **GPU 分配**: vanilla GPU 0, entropy GPU 0 (串行), tcr GPU 1, anneal GPU 7

## 结果总览

| 方法 | PPL(med) | PPL(mean) | vs Baseline | p-value | Div | Deg | 时间 | 状态 |
|------|----------|-----------|-------------|---------|-----|-----|------|------|
| **vanilla** (baseline) | 16.23 | 30.1 | -- | -- | 0.967 | 0 | 2.4s | -- |
| **tcr_r30_s32** | 15.77 | 23.8 | -2.8% | 0.254 | 0.967 | 0 | 3.5s | NOT SIG |
| **entropy_r20_mean** | 16.15 | 41.8 | -0.5% | 0.530 | 0.967 | 0 | 4.6s | NOT SIG |
| **anneal_lin_06_02** | 17.55 | 27.0 | +8.1% | 0.636 | 0.971 | 0 | 2.4s | NOT SIG |

## 统计检验详情

### TCR (r30_s32) -- 最接近改善但不显著
- Paired t-test: t=-1.146, **p=0.254**
- Wilcoxon: p=0.999
- Bootstrap 95% CI: [-17.82, +2.29] (包含 0)
- 改善 prompt: 52/101 (51.5%)
- Per-seed PPL(med): s42=16.40, s123=14.93, s456=16.09

### Entropy (r20_mean) -- 几乎无效果
- Paired t-test: t=0.631, **p=0.530**
- Wilcoxon: p=0.497
- Bootstrap 95% CI: [-14.20, +52.15] (巨大方差，包含 0)
- 改善 prompt: 56/101 (55.4%)
- Per-seed PPL(med): s42=15.90, s123=14.64, s456=17.34
- 注意: PPL(mean)=41.8 远高于 median，存在极端异常值

### Anneal (lin_06_02) -- PPL 反而恶化
- Paired t-test: t=-0.475, **p=0.636**
- Wilcoxon: p=0.561
- Bootstrap 95% CI: [-16.11, +7.04]
- 改善 prompt: 48/101 (47.5%)
- Per-seed PPL(med): s42=18.39, s123=18.25, s456=16.32

## 关键发现

### 1. Pilot 结果未能在 Full-Scale 重现
| 方法 | Pilot PPL 改善 | Full PPL 改善 | 差距 |
|------|---------------|---------------|------|
| entropy_r20_mean | **-24.9%** | -0.5% | 24.4pp |
| tcr_r30_s32 | **-22.9%** | -2.8% | 20.1pp |
| anneal_lin_06_02 | **-16.5%** | +8.1% | 24.6pp |

**原因分析**:
- Pilot 仅 16 prompts x 1 seed，方差极大
- 16 个 pilot prompts 均为简单科学问题，可能偏向容易改善的类型
- 单 seed 的随机波动在小样本上被放大

### 2. 所有方法的核心问题: 无统计显著性
- 三个方法的 p-value 均 > 0.25
- Bootstrap CI 均包含 0
- 改善/恶化 prompt 比例接近 50/50
- **结论: 推理时改进策略在 Dream-7B 上不产生统计显著的质量改善**

### 3. 与前序 ReMask-Retry 负面结果一致
- 本研究在 ReMask-Retry 失败后探索了 4 种新策略
- TCR、温度退火、熵引导重掩码、并行投票 -- 全部无效
- 强化了原始论文的核心发现: DLM 的推理时计算扩展可能存在根本性困难

### 4. 多样性保持良好
- 所有方法 diversity >= 0.967，0 退化
- 不存在"以退化换 PPL"的问题
- 但也意味着方法没有实质性改变生成行为

## 计算开销
| 方法 | 每 prompt 时间 | 相对开销 | 总运行时间 |
|------|--------------|---------|-----------|
| vanilla | 2.4s | 1.0x | ~12 min/seed |
| anneal | 2.4s | 1.0x | ~12 min/seed |
| tcr | 3.5s | 1.5x | ~18 min/seed |
| entropy | 4.6s | 1.9x | ~23 min/seed |

## 结论

**所有三个推荐方法在 Full-Scale 验证中均未产生统计显著的 PPL 改善。**

这是一个重要的负面结果，进一步证实了 DLM 推理时改进策略的困难性。Pilot 阶段的大幅改善（-16% 到 -25%）完全是小样本噪声造成的假象。

**建议下一步**:
1. 将此负面结果与 ReMask-Retry 合并，强化论文的负面发现叙事
2. 不再继续 task_3b（计算-质量曲线）和 task_4a/4b（token 分析），因为没有有效方法可分析
3. 考虑转向: (a) 理论分析为什么推理时改进在 DLM 上困难，(b) 探索需要训练的方法
