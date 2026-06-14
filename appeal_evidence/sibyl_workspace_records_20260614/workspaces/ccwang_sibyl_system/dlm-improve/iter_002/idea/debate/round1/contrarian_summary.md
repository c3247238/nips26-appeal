# Contrarian Round 1 Summary

## 总判断

这一轮 6 份稿子整体方向是收敛的，说明大家已经吸收了“不要回到 hero controller / 不要重启 calibration-aware / 不要写 generic scheduler”这几个硬边界。真正的分歧不在方向大类，而在于：

- 是否还要留一个 `Minimal Controller` 候选；
- `Bucket` 应被写成主线还是支撑证据；
- `Protocol / fairness` 是 hygiene 补件还是论文核心贡献。

从 contrarian 视角看，最需要压制的是任何会让 reviewer 说出下面三句话的候选：

1. “这只是 narrative selection，把已有负面结果包装成一个新故事。”
2. “这里的方法差异和实现差异没有分开，存在 implementation confound。”
3. “这个理论/机制说法太满了，但证据只到 aggregate 或少量 boundary slices。”

## 我建议直接砍掉或显著降级的候选

### 1. 砍掉作为独立候选的 `Minimal Controller for Decoupling`

原因不是它完全没价值，而是它在本轮最容易制造额外风险：

- reviewer 会自然把它当 method-adjacent contribution；
- 需要新的 compute、实现和 fairness 控制；
- 一旦实验不漂亮，会同时伤主线和资源分配；
- 即使结果漂亮，也会被质疑是不是又把 diagnostic paper 偷偷写回方法论文。

如果真要保留，只能作为 **最后阶段的小型 sanity probe**，而且不能进入标题级或摘要级贡献列表。

### 2. 砍掉任何把 bucket 直接写成完整 mechanism taxonomy 的版本

当前 bucket 值得做，但不能写得过度：

- 现阶段最稳的是 `fixed / harmed / no-effect` 与任务边界的对照；
- 不稳的是把它升级成“我们已经解释了 revision 的机制结构”。

bucket 是证据增强器，不是一个可以脱离主线独立站住的 hero claim。

### 3. 砍掉任何把跨任务边界写成 regime law 的倾向

尤其是 code / reasoning / structure-sensitive 这些 slicing，只能写成 boundary evidence 或 stress test：

- 当前 cross-task evidence 仍薄；
- reviewer 很容易抓“你们是不是用小 slice 过度外推”。

## 我建议只保留的 2-3 个最稳方向

### 保留 1：Protocol-First Diagnostic Paper

这是最不容易被打成 narrative selection 的主线，前提是别把它缩成“补 appendix”。

正确定位应是：

- `honest compute fairness`
- `runtime-lineage discipline`
- `claim-to-asset mapping`

也就是把“怎么比较才不自欺”做成论文贡献，而不是只把它当 hygiene。

### 保留 2：Bucket-Oriented Failure Analysis

这不是独立 hero idea，而是最关键的证据支柱。

正确定位应是：

- 用 `fixed / harmed / no-effect` 把 aggregate result 打散；
- 再和 reasoning / code / structure-sensitive boundary 轻量对齐；
- 明确哪些样本受益，哪些样本被伤害，而不是只讲平均收益。

这条线之所以稳，是因为它最直接回应 reviewer：“你们到底知道 revision 改善了谁、伤害了谁吗？”

### 保留 3：Observer/Controller Decoupling

这条线可以保留，但必须 **降调成 falsification-style claim**：

- 不是“我们提出了一个更好的分解框架”；
- 而是“我们验证了一个常见隐含假设并发现它并不自动成立”。

只有在这条线上，理论表达才有价值；否则就会被打成漂亮 formalism。

## 建议的最终收束方式

如果只允许留下 2 个 headline 方向，我的建议是：

1. `Protocol-First Diagnostic Paper`
2. `Bucket-Oriented Failure Analysis`

如果允许留下 3 个，我才会把下面这一条作为第三位：

3. `Observer/Controller Decoupling`

但第三条必须建立在前两条资产补齐之后，且不能再引出一个新的 controller 候选。

## 一句话决策

本轮应该砍掉 `Minimal Controller` 作为正式候选，只保留 **protocol/fairness 主线 + bucket failure evidence**，并把 `observer/controller decoupling` 收束成一个低承诺、可证伪的支撑论点；这是最不容易被 reviewer 打成 narrative selection 或 implementation confound 的组合。
