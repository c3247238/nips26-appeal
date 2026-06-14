# Result Debate Synthesis

## 最终共识

iter_002 的多条证据已经收敛到一个非常明确的方向：

1. 论文主线应固定为 **compute-normalized diagnostic / protocol paper**；
2. 主证据层是 `cand_bucket`，因为它把 aggregate gain 转成了样本级结构证据；
3. 支撑协议层是 `cand_protocol`，负责 runtime fairness、claim-to-asset lineage、observer/controller reporting discipline；
4. `Observer-Controller Split` 只能是 framing，不再是单独 execution candidate；
5. `cand_min_controller_probe` 已通过 skip artifact 显式关闭，不应再被重开。

## 这轮结果真正说明了什么

- `+3pp` headline gain 只有在 bucket 分解之后才有解释力；
- runtime fairness 只有在 batch size、backend、compile、actual path 显式落盘后才可信；
- minimal seed closure 只支持方向一致，不支持强稳健性；
- 论文现在最强的价值是“把 revision gain 的解释条件说清楚”，而不是“提出新 controller”。

## 执行动作

1. 将 `cand_bucket` 定为唯一前进候选；
2. 将 `cand_protocol` 明确写入 support lane；
3. 不再新增本轮 full experiment batch；
4. 直接进入 supervisor decision 与 writing 阶段。
