# 迭代 2 反思报告

## 本轮迭代总结

本轮质量轨迹是**小幅上升但仍未过门**。`logs/quality_trend.md` 记录的分数从 6.0 上升到 6.4，说明方向是对的，但 supervisor 仍给出 `revise`，核心原因已经从“有没有新证据”转成“证据包是否自包含、论述是否严格对齐证据边界”。

这一轮最有价值的进展是三类最小闭环证据都真正落地了：

- `exp/results/benefit_bucket_audit_pilot.json` 给出了 `7 fixed / 4 harmed / 89 no_effect` 的 bucket 分解；
- `exp/results/seed_sensitivity_spotcheck.json` 给出了三 seed 同向但幅度较小的 sign consistency；
- `exp/results/runtime_probe_iter2.json` 给出了当前 host 上的 runtime contract：`eager|compile=True`、safe batch size 57。

同时，这轮也做出了正确的**不做什么**的决定：`exp/results/min_controller_decoupling_probe.json` 明确把额外 controller probe 记为 `NO_GO`，避免为了“更像方法论文”而重新引入 story drift。

需要明确说明的是：`exp/results/summary.md` 在当前 workspace 中仍然缺失，因此本次反思无法按角色协议读取 canonical summary，只能改用代表性结果资产进行交叉核对。实际使用的代表性证据包括：

- `exp/results/runtime_probe_iter2.json`
- `exp/results/benefit_bucket_audit_pilot.json`
- `exp/results/seed_sensitivity_spotcheck.json`
- `exp/results/runtime_fairness_matrix.json`
- `exp/results/observer_controller_protocol.json`
- `exp/results/canonical_asset_manifest.json`
- `exp/results/min_controller_decoupling_probe.json`

## 修复追踪

### 已修复或明显缓解

- 上轮最关键的缺口之一是 **benefit buckets 仍停留在 future work**。这一轮已用 `benefit_bucket_audit_pilot.json` 和 `benefit_bucket_examples_pilot.json` 补上样本级证据，属于明确修复。
- 上轮“**不能继续只靠单 seed**”的硬问题已得到部分缓解。`seed_sensitivity_spotcheck.json` 给出三 seed 同向，虽然还不是 robustness study，但已经不再是完全无不确定性信息的状态。
- 上轮要求写透的 **signal audit 定义** 已经产出 `observer_controller_protocol.json` 与 `current/plan/signal_audit_protocol.md`，定义层面较上一轮更清楚。
- 上轮提到的 **iteration 目录卫生** 也有改善：`current` 现在指向 `iter_002`，不再停留在旧 iteration。

### 反复出现的问题

- **证据边界与 headline 口径不完全匹配** 仍然反复出现。虽然这次有了 bucket 与 seed 证据，但论文仍把 `n=100` 的 audited slice 说得过于接近 full-benchmark story。
- **runtime fairness / honest-compute 的 reviewer-friendly 封装不足** 继续出现。artifact 存在了，但 current audited contract 与 historical rows 仍被混放在一个 bundle 里。
- **scholarly packaging 不足** 仍是稳定阻塞项，尤其是 Related Work 的引用与 bibliography 支撑。
- **效率问题从“没做优化”转成“做了但审计不到位”**。这一轮已经用了 compile 和 batch probe，但系统仍拿不出任务级时序来证明并行度与 idle time 管理。

### 本轮新暴露的问题

- `exp/results/summary.md` 缺失，导致 canonical evidence bundle 仍不是 reviewer 拿到就能复核的自包含包。
- `canonical_asset_manifest.json`、`observer_controller_protocol.json`、`runtime_fairness_matrix.json` 里仍保留 `/home/ccwang/...` 绝对路径，破坏了 portability。
- 论文把 observer-controller split 写得比当前证据更强，而当前资产更多是在定义 protocol，而不是验证一个新的经验规律。

## 各类问题分析

### ANALYSIS

- 当前最危险的分析问题是**observer-controller 叙事仍强于证据**。`observer_controller_protocol.json` 主要是在定义 `d(s)` 与 `g(s)`，而不是通过 matched empirical test 去证明“observer 强不等于 controller 强”这一更一般的科学命题。当前最稳妥的写法，应当是“bucket decomposition + runtime audit 改变了 headline 的解读”，而不是把 split 本身写成已被验证的规律。
- 第二个分析问题是**runtime fairness bundle 仍混杂异质 execution regime**。当前 runtime probe 推荐的是 `eager|compile=True`、batch 57，但 fairness matrix 仍引用 compile=False、batch 32/1 和 batch 115 的历史比较行。这会让 reviewer 把 honest-compute 结果理解成 implementation artifact，而不是 protocol insight。

### EXPERIMENT

- 当前 headline 的实验边界仍然过窄。bucket audit 只覆盖 100 个样本，而 runtime probe 明确说明 GSM8K 在磁盘上有 1319 个样本；这意味着 audited slice 只有约 7.6% 的 benchmark 覆盖。
- seed 证据虽已从 0 扩展到 3，但 `delta = [0.03, 0.01, 0.01]` 的中心质量仍然是**小幅正向且脆弱**，不能继续用 best-seed headline 取代 multi-seed center of mass。

### WRITING

- 写作最大的问题已不是结构散乱，而是**学术包装没有完全跟上证据收缩**。Related Work 缺 citation anchors，导致定位看起来更像内部说明而不是可提交论文。
- 另一处写作问题是**重复的自我定位语句过多**。多份评审都提到“不是 hero-controller paper”这层信息已经足够清楚，继续重复会挤压本该用来放 hard caveat 与 prior-art contrast 的空间。

### PLANNING / PIPELINE

- proposal 锁定的 deliverable 名称与当前产物仍不完全一致。现在存在的是 `_pilot` 资产，但 proposal 和 manuscript 的语气更像在描述 canonical full audit。
- `summary.md` 缺失与 absolute-path lineage 说明**交付收口没有被当作 done criteria 的一部分**。这不是新实验不足，而是 pipeline 最后的 artifact-packaging 没有被强约束。

### EFFICIENCY

- 这轮已经不是“完全没做优化”。`runtime_probe_iter2.json` 明确记录了 `compile=True` 可用，且 safe batch size 被探到 57；bucket 与 seed 资产也沿用了这一 runtime contract。这比项目记忆中批评的 batch=1 / 无 compile 状态前进了一步。
- 但效率层面依然没有形成可审计闭环。`gpu_progress.json` 只有 `completed` 列表，没有 timings；`monitor_status.json` 只给出完成时的快照，无法回答 GPU 空闲了多久、任务之间是否被不必要串行化。
- 从显存角度看，bucket audit 中 entropy 路径峰值约 39.2GB，而 GPU 总显存约 97.3GB；这说明当前 contract 更像 safety envelope，而不是针对真实 prompt 长度分布优化到极限的 throughput envelope。

## 资源效率评估

### 已落地的优化

- `runtime_probe_iter2.json` 明确探测了当前主机的 safe batch size = 57；
- `compile=True` 已经进入当前 audited runtime contract；
- `shared_runtime_probe_gpu_profile.json` 给出了单 probe 的显式 utilization 指标 28.3%，并记录了 eager backend、max batch 57；
- `min_controller_decoupling_probe.json` 通过 `NO_GO` 决策避免了额外耗费 GPU 去追一个会重新扩张 story 的实验。

### 仍然存在的效率问题

- `flash-attn` 在当前 host 上仍不可用，因此 attention backend 仍是 `eager`；
- 没有任务级 timing，所以**无法可靠计算 total GPU idle minutes**；
- `monitor_status.json` 虽显示当前没有 running task，且可见 GPU 处于空闲，但没有任何证据说明 bucket、seed、protocol 这些无强依赖任务是否曾被并行调度；
- 当前 batch 57 是基于 768-token probe 的安全值，但 bucket audit 的 `prompt_max_len` 只有 147，说明仍存在继续上探 task-specific batch 的空间。

### 结论

本轮效率状态可以概括为：**优化开始落地，但效率治理还没有变成一等证据对象**。系统已经能做 compile 与 batch probing，也能在 scope 控制上避免无意义新实验；但只要 telemetry 仍然缺时序和占用区间，下一轮就仍然无法用数据证明“GPU 没有被浪费”。

## 质量趋势判断

`logs/quality_trend.md` 给出的轨迹是 6.0 → 6.4，属于**改善但尚未脱离 revise 区间**。结合本轮三路评审，趋势判断如下：

- **上升**：证据比上一轮更闭环，method-forward 叙事继续被压缩；
- **未过门**：剩余阻塞已集中在 portability、uncertainty、runtime reconciliation 和 scholarly packaging；
- **有平台期风险**：如果下一轮继续主要增加 prose，而不先解决自包含 evidence bundle 与 lineage 清理，分数很可能停在 6.x 到 7.0 附近。

## 根因分析

1. **done criteria 仍偏向“产生资产”，而不是“资产可被 reviewer 直接消费”**。这解释了为什么 JSON 已经很多，但 `summary.md`、relative lineage 和 portable bundle 仍未闭环。
2. **旧资产被复用得太快，新的口径说明跟得不够快**。这解释了为什么 runtime fairness 一边强调 current contract，一边又继续带着历史 batch/compile 行。
3. **系统已经学会克制 story drift，却还没把这种克制变成模板化收口动作**。所以 `NO_GO` 能做对，但 proposal/manuscript 的 canonical deliverable 名称仍然滞后。
4. **效率 telemetry 不是本轮实验 done criteria 的一部分**。因此 compile 和 batch probe 做了，空闲时间和并行度却仍不可审计。

## 系统自检响应

`logs/self_check_diagnostics.json` 在本 workspace 中不存在，因此本轮没有可直接回应的系统自检诊断结果。这里需要额外提醒两点：

- 缺少该文件并不代表系统没有问题，只代表这轮没有留下独立的 self-check 证据；
- 下一轮若继续把效率与可复核性当作关键议题，最好把 self-check diagnostics 也纳入 standard artifact bundle，而不是完全依赖 reviewer 文本反馈。

## 下一轮最应该聚焦的事情

1. 先把 `current/` 做成真正可移植、自包含的 canonical evidence bundle；
2. 再把论文 headline 严格收缩到 `n=100 audited slice + 3-seed sign consistency only`；
3. 然后补 runtime reconciliation、uncertainty framing 与 Related Work 的引用骨架；
4. 最后再谈是否还需要任何额外实验，而不是反过来。
