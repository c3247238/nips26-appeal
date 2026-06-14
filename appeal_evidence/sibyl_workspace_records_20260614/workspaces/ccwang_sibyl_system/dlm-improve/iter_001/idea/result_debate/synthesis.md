# Result Debate Synthesis

**Synthesizer**: main control plane
**Date**: 2026-03-12
**Inputs**: `optimist`, `skeptic`, `strategist`, `methodologist`, `comparativist`, `revisionist` 三轮辩论

## 1. 最终共识

六个视角在第三轮后已经收敛到一条清晰主线：

1. 当前最值得继续推进的不是 `TIGER` 方法故事，而是 `cand_diag` 的 diagnostic / honest-compute 论文路线。
2. 主文最安全的三条核心主张是：
   - honest compute 会改变 DLM revision 方法的排序与 Pareto 叙事；
   - observer 和 controller 在 DLM 中存在稳定错位；
   - revision gain 是 task-dependent 的，不能从单一 benchmark 或单一 signal 直接外推。
3. code 结果不应再承担“泛化成功”角色，只能作为 boundary / appendix 证据：
   - syntax guard 能修浅层合法性；
   - 但不能恢复 runtime / semantic correctness。
4. `TIGER` 应该从前台主方法彻底降级为 failed-but-informative controller case。
5. 下一步最值钱的动作不是再发明新控制律，而是把 `benefit buckets / failure taxonomy` 补成正式结果资产。

## 2. 仍存分歧

虽然总体方向一致，但仍有三点残留分歧：

1. **论文定位的措辞强度**
   - `optimist` 倾向于把它写成 “纠偏论文 / diagnostic benchmark paper”
   - `skeptic` 认为当前证据更像 “compute-normalized diagnostic study”
   - 综合判断：暂时不要自称 community benchmark standard-setter，更安全的表述是 `diagnostic study + evaluation protocol + failure taxonomy`

2. **honest compute 结论的强度**
   - `optimist` 认为 ranking reorder 已足够构成主文 claim
   - `skeptic` 与 `methodologist` 认为现阶段应写成 “can change key comparisons / can change Pareto position”，而不是暗示普遍颠覆全部排序

3. **observer-controller gap 的外推程度**
   - 多数角色同意这条线很强
   - 但 `skeptic` 和 `methodologist` 坚持：当前更安全的说法是 “under the tested policies, good observers do not reliably become good controllers”，不要提前上升到一般性定律

## 3. 最终论文 Framing 裁决

最终裁决如下：

- **保留主线**：`cand_diag`
- **放弃主线**：`cand_tiger` 的 method-forward 叙事
- **最终定位**：
  - 不是 “another stronger decoder”
  - 不是 “benchmark SOTA paper”
  - 而是 “针对 training-free DLM revision/search 的 honest-compute diagnostic study”

主文应围绕三件事组织：

1. `honest compute accounting`
2. `observer-controller mismatch`
3. `task-dependent revision response`

appendix 承接：

1. code boundary / shallow-vs-deep failure
2. runtime fairness details（`batch_size / backend / compile / peak_vram / actual_nfe / latency / throughput`）
3. 细碎的 `TIGER / DNB / Prophet / MATH500` 展开表

必须主动删除的旧叙事：

1. “我们快找到更强 revision 方法了”
2. “reasoning 是统一 revision gain 场景”
3. “更强 observer 自然会导出更强 controller”
4. “code guard 是泛化成功前兆”

## 4. 唯一 P0

唯一 P0 是：

**补出 `benefit_bucket_audit` / `failure buckets` 的统一结果资产。**

原因很直接：

- 这一步能把当前 diagnostic story 从 aggregate-level 表格升级成机制证据；
- 它是 `methodologist`、`strategist`、`revisionist`、`optimist` 四方共同支持的最短增益路径；
- 如果不补这一步，论文会更像“结果整理”，而不是“机制论文”。

## 5. 后续动作列表

按优先级排序，后续动作应是：

1. 产出 `benefit_bucket_audit.json`
   - 至少区分：
     - `draft-correct-then-harmed`
     - `draft-wrong-then-fixed`
     - `revision-no-effect`
2. 产出 runtime fairness appendix 表
   - 统一汇总：
     - `batch_size`
     - `compile`
     - `backend`
     - `peak_vram`
     - `actual_nfe`
     - `latency`
     - `throughput`
3. 做一个最小 `seed_sensitivity_spotcheck`
   - 只覆盖最关键比较，不要求大规模重跑
4. 产出 `asset_lineage_summary`
   - 特别说明 `diag_signal_gap_audit` 的资产来源与协议边界
5. 写作时强制 result-first 收缩
   - 主文只讲三条主张
   - appendix 收编 code boundary 与 runtime details

## 6. 一句话结论

这轮结果最强的价值，不在于找到一个更强 revision policy，而在于把 DLM revision 文献里混在一起的三件事拆开了：

**honest compute 改写排序，observer/controller 并不等价，而 revision 是否有用取决于 recoverability。**
