# Iteration 3 Hypotheses

## H1：Aggregate Gain 不足以解释真实效应结构

- 命题：
  - 在当前 tested policies 下，仅报告 aggregate revision gain 会掩盖不同样本 bucket 的净效应来源。
- 预期证据：
  - `benefit_bucket_audit.json` 中存在清晰的 `fixed / harmed / no-effect` 结构差异。

## H2：Observer Quality 不自动转化为 Controller Gain

- 命题：
  - 在当前 evidence scope 下，更强的 observer signal 不保证带来更强的 realized controller gain。
- 预期证据：
  - `diag_signal_gap_audit.json` 与 `observer_controller_protocol.json` 能支持将 `d(s)` 与 `g(s)` 分开解释。

## H3：Realized Compute Fairness 会改变结论解释

- 命题：
  - nominal steps 或 even actual NFE 都不足以单独定义可信比较；batchability、latency、backend、compileability 会影响 headline ordering 的解释。
- 预期证据：
  - `runtime_fairness_matrix.json` 与 `canonical_asset_manifest.json` 能显示 claim-to-asset 的公平比较条件。

## H4：Boundary Evidence 只能支持边界判断，不能支持跨任务规律

- 命题：
  - 当前 MATH500 / HumanEval 证据只足以支持 boundary-sensitive interpretation，不足以支持稳定 regime claim。
- 预期证据：
  - proposal、paper framing 与 artifact 显式把 cross-task 结果限制在 boundary evidence。

## H5：最小稳健性检查是质量门而不是可选件

- 命题：
  - headline pairwise comparison 至少需要最小 seed spot-check 才能作为可信 diagnostic evidence 使用。
- 预期证据：
  - `seed_sensitivity_spotcheck.json` 报告 delta 方向一致性，而非单次 seed 结果。
