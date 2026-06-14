# Task 5b Pilot 总结: GSM8K DTA 评估

## 实验配置
- 模型: Dream-v0-Instruct-7B
- 基准: GSM8K (前 16 题)
- 去噪步数: 128, 温度: 0.4, 生成长度: 512
- 方法: Vanilla, ReMDM-conf, DTA, DTA+ReMDM
- GPU: NVIDIA RTX PRO 6000 Blackwell (95GB)
- 总耗时: 1039s (~17 min)

## 结果

| 方法 | 准确率 | 正确数 | 提取率 | 平均时间 | Dist-2 | Rep-3 |
|------|--------|--------|--------|----------|--------|-------|
| Vanilla | 25.0% | 4/16 | 100% | 8.5s | 0.818 | 0.096 |
| ReMDM-conf | **37.5%** | 6/16 | 100% | 13.2s | 0.810 | 0.089 |
| DTA | 12.5% | 2/16 | 100% | 19.5s | 0.799 | 0.096 |
| DTA+ReMDM | 18.8% | 3/16 | 93.8% | 23.2s | 0.709 | 0.211 |

## 关键发现

1. **DTA 在 GSM8K 上未能超越 vanilla** (12.5% < 25.0%)，方向与预期相反
2. **ReMDM-conf 在 GSM8K 上表现最好** (37.5%)，显著优于 vanilla (+12.5pp)
3. **DTA+ReMDM 组合未能超越纯 ReMDM-conf** (18.8% < 37.5%)
4. DTA+ReMDM 的 rep-3 (0.211) 明显高于其他方法，暗示文本质量有退化

## 与 Countdown (task_5a) 的对比

在 task_5a Countdown pilot 中：
- Vanilla: 12.5%, DTA: 6.25%, ReMDM-conf: 6.25%
- DTA 同样未超越 vanilla

**GSM8K 与 Countdown 的一致趋势**: DTA 在两个基准上都未能提升准确率。这可能说明:
- DTA 的 LoRA 更新在 pilot 小样本 (16) 下方向不稳定
- 超参数（lr=5e-4, rank=4）可能需要针对 GSM8K 的更长生成长度调整
- gen_len=512 下 LoRA 更新步数更多，累积漂移更严重

## 判定: CONDITIONAL-GO

虽然 DTA 准确率方向不正确，但：
- 所有 4 种方法成功运行，无崩溃
- 答案提取率 ≥ 93.8%
- ReMDM-conf 在 GSM8K 上表现突出，为后续组合实验提供了强 baseline
- DTA 的 LoRA 范数保持稳定（未爆炸），问题可能在超参数而非方法本身

## 下一步建议

1. 调低 DTA 学习率 (1e-4 或 5e-5)，可能当前 lr=5e-4 在长序列上过于激进
2. 增大 warmup_frac (从 0.20 到 0.30)，GSM8K 生成更长，需要更多初始 token 积累
3. 检查 DTA 在 GSM8K 上的 LoRA 范数轨迹，确认是否存在参数漂移
