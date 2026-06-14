# 理论研究者视角

## 总判断

这一轮最有理论价值的不是设计新启发式，而是澄清一个被现有 DLM inference 文献普遍混淆的问题：`diagnostic signal quality`、`intervention policy quality` 与 `compute-realized gain` 并不等价。

## 我支持的候选方向

### 候选 A：Observer-Controller Decomposition as Protocol

- 理论命题：
  - `d(s)` 衡量 observer 对错误风险的诊断能力
  - `g(s)` 衡量在给定 policy 与 compute budget 下，diagnostic signal 转化为控制增益的 realized outcome
- 贡献：证明两者必须分开报告，否则“好信号”会被错误解释成“好方法”。

### 候选 B：Bucket Taxonomy as Mechanism Evidence

- 理论命题：
  - aggregate gain 可以被分解为不同 bucket 的净和
  - 同一总体增益可能由完全不同的 bucket 结构产生
- 贡献：为 failure taxonomy 提供可操作定义，而不是叙事性分类。

### 候选 C：Runtime Fairness as Evaluation Object

- 理论命题：
  - nominal steps 或 even actual NFE 都不足以完全定义可比 compute
  - realized latency 受 batchability、cache locality、backend compileability 影响
- 贡献：把“公平比较条件”本身 formalize 成 protocol requirement。

## 排名

1. 候选 A
2. 候选 C
3. 候选 B

## 关键写法建议

- 正文中不要把 observer/controller mismatch 只写成经验现象，要写成 evaluation protocol 的必要区分。
- appendix 应明确：
  - `d(s)` 的统计定义与来源 artifact
  - `g(s)` 的统计定义与来源 artifact
  - 两者的解释范围与不可互推边界
