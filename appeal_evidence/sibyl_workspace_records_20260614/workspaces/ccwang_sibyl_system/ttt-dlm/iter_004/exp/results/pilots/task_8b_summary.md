# Task 8b Pilot 总结：Token 级诊断分析（16 样本）

## 实验概要

在 16 个 Countdown 问题上运行 4 种方法（Vanilla, ReMDM-conf, SCP, DTA），记录每步去噪的 token 变化轨迹，计算 Correction Precision/Recall。

## 准确率

| 方法 | 准确率 | 正确数 | 平均时间/样本 |
|------|--------|--------|-------------|
| Vanilla | 12.5% | 2 | 3.7s |
| ReMDM-conf | 6.2% | 1 | 6.5s |
| SCP | 12.5% | 2 | 26.0s |
| DTA | 12.5% | 2 | 15.3s |

## 核心指标：Correction Precision / Recall

| 方法 | Correction Precision | Correction Recall | 总 Remask 数 |
|------|---------------------|-------------------|-------------|
| ReMDM-conf | **0.313 +/- 0.084** | 0.468 +/- 0.223 | ~295/样本 |
| SCP | **0.769 +/- 0.135** | 0.608 +/- 0.181 | ~14/样本 |

**Correction Precision 定义**：被 remask 的 token 中，确实是错误的（与最终输出不同）比例。
**Correction Recall 定义**：当步中所有错误 token 中，被 remask 的比例。

## 假设检验

### H9: ReMDM-conf Correction Precision < 50% → **SUPPORTED**
- ReMDM-conf 平均 Precision = 31.3%
- 意味着 ReMDM-conf remask 的 token 中，**近 70% 其实是正确的**
- 这解释了为什么 ReMDM-conf 在推理任务上效果有限——它在"修正"本来就对的 token

### H6: SCP Correction Precision > ReMDM-conf → **SUPPORTED**
- SCP Precision (76.9%) 远高于 ReMDM Precision (31.3%)，差距 +45.5pp
- SCP 的 leave-one-out 探测能更精准地识别真正有问题的 token
- 但 SCP 计算开销巨大（~7x vanilla），且 remask 数量远少于 ReMDM-conf

## 轨迹稳定性分析

| 方法 | 平均变化率 | 不稳定位置数 | 高度不稳定(>5次变化) |
|------|-----------|-------------|---------------------|
| Vanilla | 0.0364 | 0.0 | 0 |
| ReMDM-conf | **0.0971** | **94.8** | 大量 |
| SCP | 0.0378 | 11.9 | 少量 |
| DTA | 0.0338 | 0.0 | 0 |

- ReMDM-conf 的不稳定位置数（94.8）远高于其他方法，说明 remasking 导致大量 token 反复变化
- DTA 和 Vanilla 一样稳定（无 remasking），但 DTA 通过参数更新隐式提升预测质量
- SCP 介于两者之间，remask 数量少但更精准

## DTA 信息积累（H7 代理指标）

- early/late 准确率都是 1.000——这是因为 oracle 定义为最终输出本身，DTA 没有 remasking 所以所有 revealed token 天然匹配最终输出
- 这个指标需要改进：应该用 Vanilla 的最终输出作为 oracle，而非各方法自己的最终输出
- 替代方案：在 full-scale 实验中用跨方法的 oracle（如 DTA+ReMDM 的最终输出）

## 通过标准

- **Token 轨迹记录成功**: PASS
- **Correction Precision/Recall 计算无报错**: PASS
- **总体**: PASS

## 关键发现与意义

1. **ReMDM-conf 的根本问题**：Precision 仅 31.3%，意味着它 remask 了大量正确 token。这是一个重要的 negative finding，解释了为什么纯 remasking 在推理任务上效果有限。

2. **SCP 精度高但效率低**：Precision 76.9% 说明 leave-one-out 探测能准确识别问题 token，但每样本 26s（7x vanilla）的开销使其不实用。

3. **DTA 的稳定性优势**：DTA 不引入额外的 token 不稳定性（unstable_positions=0），通过参数空间而非 token 空间优化，避免了 remasking 的信号质量问题。

4. **轨迹稳定性视角**：ReMDM-conf 平均 94.8 个不稳定位置（约 37% 的生成区域），说明大量 token 在去噪过程中反复被修改，这种"token 搅动"可能反而降低生成质量。

## 墙钟时间

总计约 14 分钟（Vanilla 60s + ReMDM 104s + SCP 416s + DTA 245s + 开销）。
