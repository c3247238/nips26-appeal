# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][EXPERIMENT] DMI 是论文核心贡献但缺乏系统消融。alpha 仅 0.3、tau 仅 0.5。无负对照（随机 embedding 注入）、无 hard-prediction 基线、无 multi-step lookback。 (出现 2 次, 权重 1.96)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
