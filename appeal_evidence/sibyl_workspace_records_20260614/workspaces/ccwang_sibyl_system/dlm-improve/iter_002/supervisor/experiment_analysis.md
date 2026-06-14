# 实验结果分析

## 实验结果概要

iter_002 没有再把项目推进成新的 full experiment 扩张，而是完成了上一轮 review / reflection 明确要求的证据收口：

1. `runtime_probe_iter2.json` 确认当前主机可行路径是 `eager|compile=True`，并给出 `safe_batch_size=57`；
2. `benefit_bucket_audit_pilot.json` 把 headline pair (`Standard-64` vs `Entropy-Revise-64+3`) 分解为 `fixed=7 / harmed=4 / no_effect=89`，coverage 达到 `100%`；
3. `canonical_asset_manifest.json`、`runtime_fairness_matrix.json`、`observer_controller_protocol.json` 把 claim-to-asset lineage 与 runtime fairness 写成 reviewer 可审计协议；
4. `seed_sensitivity_spotcheck.json` 给出三 seed 最小稳健性检查，delta 分别为 `[+0.03, +0.01, +0.01]`，方向均为 `entropy_better`；
5. `min_controller_decoupling_probe.json` 明确记录本轮没有再开 controller 新线，而是有意识地关闭该 optional probe，避免 story drift。

## 各方观点总结

- 乐观者：bucket、protocol、seed 三条线已经把 diagnostic story 变成可 defend 的证据链。
- 怀疑者：必须严格下调 claim 强度，尤其不能把 sign consistency 写成 full robustness，也不能把 protocol 写成新 hero contribution。
- 战略家：当前最优路线是停止新增实验扩张，直接进入结果综合和写作收口。

## 分析

### 1. 方法可行性

如果问题是“是否值得继续当前论文方向”，答案是肯定的；如果问题是“是否还需要再开一轮 full experiment 才能继续”，答案是否定的。当前方向已经从 method-forward 故事稳定转成 diagnostic/protocol 论文，所需主证据已经齐备。

### 2. 性能表现

本轮最重要的表现不是更高的 aggregate 分数，而是把 aggregate gain 的来源讲清楚。`+3pp` 与 `7 fixed - 4 harmed` 一致，说明 headline gain 现在有了样本级解释结构，而不是只剩下平均值。

### 3. 改进空间

当前改进空间主要在写作而不是实验：

1. 把 bucket-level evidence 写成 Results 主线；
2. 把 runtime-lineage protocol 放入 setup / appendix；
3. 把 minimal seed closure 写成 sign consistency，而不是 robustness；
4. 显式说明 optional controller probe 为 `NO_GO`。

### 4. 时间成本

继续开新实验的边际收益已经低于写作收口的边际收益。再跑 controller/full batch 不仅会耗费时间，还会重新引入 narrative drift；直接进入写作则能最大化当前证据价值。

### 5. 怀疑论者的批评是否致命

不是致命否定，而是边界约束。只要我们接受“更窄但更诚实”的写法，这些批评会提高论文可信度，而不会逼迫再次 PIVOT。

## 决策理由

我选择 `PROCEED`，因为：

1. iter_002 需要的 evidence gaps 已经闭环；
2. 主线论文定位已经稳定，不再需要通过新实验寻找身份；
3. 当前最值钱的下一步是 result synthesis 与 writing，而不是扩大实验面；
4. optional controller probe 已被显式关闭，继续扩实验只会增加混乱。

## DECISION: PROCEED

DECISION: PROCEED
