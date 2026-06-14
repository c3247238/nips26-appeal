# Project Memory

记录这个项目特有的长期约束、偏好和已确认决策。

- 这里只写项目层信息，不要重复系统层规则
- 长期有效的约束优先，例如数据集、资源预算、论文目标、禁用方案
- 当系统策略与项目特殊要求冲突时，在这里明确写出覆盖规则

## 全局实验约束（长期有效）

- **最大 batch size**: 所有实验必须尽可能使用最大 batch size，充分利用 GPU 显存
  - RTX PRO 6000: 97GB VRAM，LLaDA-8B 加载后占 ~15GB，剩余 ~82GB 可用于 batch
  - 建议先做 batch size 探测（从大到小二分搜索），找到不 OOM 的最大值
  - 多卡实验同理，每卡都要最大化 batch size
- **Pilot vs Full 实验策略**:
  - 前期允许做小型 pilot，用于快速筛掉无效方法、验证信号是否存在
  - 一旦主方法确定，后续必须切换到全量实验，在完整 benchmark 上验证方法有效性
  - 不允许仅凭小样本 pilot 就把方法写成最终有效结论

## 证据源约定

- 最新 pilot / full 结论以 `current/exp/results/pilot_summary.json`、`current/exp/results/pilot_summary.md`、`current/exp/results/summary.md` 为权威
- 不要把阶段性实验指标长期硬编码在 memory 里；实验执行细节请写到对应 agent overlay
