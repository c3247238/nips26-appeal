# Task 5c Pilot 总结：MBPP DTA 评估

## 实验配置
- 模型：Dream-v0-Instruct-7B
- 基准：MBPP Sanitized (16 样本)
- 方法：Vanilla, DTA, DTA+ReMDM
- 去噪步数：128, 温度：0.4, 生成长度：512
- DTA: LoRA rank=4, 最后 2 层 FFN, lr=5e-4, gamma=0.95, warmup=20%
- ReMDM: remask_ratio=0.1, stop_frac=0.8

## 结果汇总

| 方法 | Pass@1 | 通过数 | 平均时间 | Dist-2 | Rep-3 |
|------|--------|--------|---------|--------|-------|
| Vanilla | 25.0% | 4/16 | 4.5s | 0.978 | 0.002 |
| **DTA** | **37.5%** | **6/16** | 19.0s | 0.978 | 0.007 |
| DTA+ReMDM | 12.5% | 2/16 | 22.6s | 0.966 | 0.023 |

## 关键发现

1. **DTA 显著优于 Vanilla**：Pass@1 从 25.0% 提升到 37.5%（+12.5pp），方向正确。
   - DTA 额外通过了 task_58（判断符号相反）和 task_62（找最小数）两个 Vanilla 未通过的任务。
   - DTA 保留了 Vanilla 通过的 task_17（正方形周长）、task_19（查重）、task_56（数字检查）。
   - DTA 丢失了 task_14（三角棱柱体积），但额外新增 task_64（元组排序）通过。

2. **DTA+ReMDM 表现反常**：Pass@1 降至 12.5%，远低于 DTA alone。
   - 这与 Countdown 和 GSM8K 的 pilot 结果一致：DTA+ReMDM 组合在 pilot 规模上不如单独方法。
   - 可能原因：remasking 干扰了代码生成的结构一致性（代码对 token 顺序更敏感）。

3. **无文本退化**：所有方法的 distinct-2 ≥ 0.96，rep-3 ≤ 0.023，代码质量正常。

4. **LoRA 稳定性良好**：最大范数 0.088（DTA alone），0.145（DTA+ReMDM），均远低于 1.0 阈值。

5. **计算开销**：DTA 约 4.2x vanilla，DTA+ReMDM 约 5.0x vanilla。

## 通过标准

- [x] 所有 3 方法成功运行
- [x] DTA Pass@1 >= Vanilla（37.5% >= 25.0%）
- **判定：GO**

## 总时长
- 墙钟时间：746 秒（约 12 分钟）
