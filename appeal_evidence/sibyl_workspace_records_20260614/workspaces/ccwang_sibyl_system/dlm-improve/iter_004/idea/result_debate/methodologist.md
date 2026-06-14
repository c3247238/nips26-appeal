# Methodologist

## 1. Baseline Fairness

- 这轮 full-scale 比较在“同数据集、同 draft 步数、同 revision 上限、同 eager backend、同 compile 关闭”这一层面上基本公平。四个主要臂都在 `GSM8K` 全量 `1319` 样本上评估，`draft_steps=64`，revision 上限一致。
- 但这份公平只成立于“实际执行合同”内部，不完全符合原始方法学承诺。`methodology.md` 与 runtime-contract 工件都强调了 compile/backend parity 与 max-safe batch probing，而最终运行全部落在 `compile=false`，candidate / sham / shared controls 的 effective batch 分别是 `54 / 52 / 57`。
- shared controls 合理，但仍缺一个 plain draft / no-revision 基线，因此我们无法判断 `cand_espd=0.4041` 到底是在 revision 家族内部略优，还是相对不修订基线也有实质增益。
- fixed-frontier sham 是本轮亮点，因为它对齐了 `frontier_ratio=0.1211` 并复用 stopping family；但 candidate 与 sham 的 VRAM、batch、auxiliary overhead 差异过大，所以它仍不是完全 matched mechanism control。

## 2. Metric Appropriateness

- `GSM8K accuracy` 是合理主指标，`fixed/harmed/net_repaired`、`avg_nfe`、`active_frontier_ratio` 也确实贴近方法声称的 routing/stopping 机制。
- 但 headline 的解释边界必须收紧。candidate 的 `equal-quality speed=124.42 tok/s` 仍低于 `CARD-84 126.08` 和 `RAND-84 128.00`，所以它不是“最快”，而是“相对 fixed-frontier sham 更快”。
- `quality_tolerance=0.01` 让 equal-quality band 变得偏宽。当前 candidate 相对 shared controls 与 sham 的质量差都在 `0.5~0.8pp` 之间，这更像弱正信号，而不是强结论。
- runtime ledger 暴露出当前 speed 指标的真正含义仍不够清楚：candidate 的端到端时间几乎完全由 auxiliary path 主导，因此仅报告 `tok/s` 容易掩盖“快慢究竟来自 model forward，还是 frontier bookkeeping”。

## 3. Protocol Audit

- 最积极的一点是，本轮确实遵守了 full-stage 的核心禁止项：没有在缺少 shared controls 与 sham 的情况下直接宣布 success。
- 但存在 runtime-contract drift：原始 best path 期待 compile-on，而 full-scale 实际全部在 compile-off 上完成。这不会直接破坏 arm 间公平，却削弱了 protocol discipline 的可信度。
- fixed-frontier 依赖 candidate 的 frontier ratio 和 stopping rule 来构造 sham，在工程上合理，但在方法学上必须被写成 preregistered matched-control construction，否则容易被解读为后验贴合。
- proposal 把 `cand_bsr` 写成 front-runner，而 full-scale 实际只验证 `cand_espd`。因此这组 full-scale 结果只能支持 speed-line 的方法学审查，不能外推成整个 proposal 已被验证。

## 4. Ablation Completeness

- 对 `cand_espd` 来说，当前 ablation 只能算“完成了最关键的一刀”，还远不完整。
- 已完成部分：fixed-frontier sham 不能复现 candidate 的综合表现，说明 routing signal 有证据价值。
- 未完成部分：
  - 没有 retention ratio sweep
  - 没有 stopping-threshold ablation
  - 没有 `entropy frontier + fixed 3 steps`
  - 没有 `random frontier + entropy stopping`
  - 没有 plain draft / no-revision 基线
- 对更大的 iteration 4 proposal 而言，object-level line 的 mechanism controls (`RandSpan-84`, `EntropySpan-NoBoundary`, `BoundaryLock-RandomSpan`) 仍未 full-scale 实例化，因此全局 ablation completeness 仍不足。

## 5. Reproducibility

- 当前工件比前几轮明显更强：bundle JSON、candidate JSON、fixed-frontier JSON、shared-controls JSON 都是 machine-readable，足够支持内部 result debate。
- 但离外部可完全复现还有距离：
  - 关键合同与 manifest 仍引用远端绝对路径
  - 没有 commit hash、seed、完整启动命令
  - compile-on 预期与 compile-off 实际之间缺少解释性桥接
- 当前状态更像“可审计”，还不算“可无歧义复跑”。

## 6. Specific Methodology Fixes

- 补一个 plain draft / no-revision baseline，并放入 unified table。
- 把 runtime contract drift 变成显式结论；若保留 compile-off，就要明说本轮结论仅适用于 eager/no-compile contract。
- 给 `quality_at_equal_compute` 与 `speed_at_equal_quality_band` 加 uncertainty layer，至少做 paired bootstrap 或 exact interval。
- 再补两刀机制拆分：`entropy frontier + fixed 3 steps` 和 `random frontier + entropy stopping`。
- 把 `sample_manifest.json`、`runtime_contract.json`、`batch_probe.json` 镜像回本地 workspace，并在结果 JSON 中引用本地路径。
- 严格收窄 claim scope：当前最稳妥的结论是 **`cand_espd` 在当前 GSM8K full contract 下提供了可信但幅度不大的 routing signal，支持继续推进 speed-line，但还不支持“系统性优于 shared controls”或“整个 iteration 4 proposal 已被 full-scale 验证”。**
