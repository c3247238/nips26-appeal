# Proposal Seed After Round 2

## 论文定位

本轮论文不再是 generic DLM improvement paper，也不是 TIGER / Calibration-Aware hero paper。

目标定位为：

- `compute-normalized diagnostic / protocol paper`
- 核心关注：
  - `observer quality` 与 `controller gain` 的非等价性
  - revision gain 的 `fixed / harmed / no-effect` 样本级来源
  - realized compute fairness / runtime-lineage 对结论可信度的影响

## 当前建议的层级结构

### 主命题（低承诺 framing）

在当前 tested policies 与 current evidence 下，`observer quality != controller gain`；因此 DLM revision 不应只按 aggregate gain 报告，而应同时报告 observer/controller split、bucket-level outcomes 与 realized compute fairness。

### 主证据层

`Benefit-Bucket / Recoverability Analysis`

- 关键 artifact：
  - `benefit_bucket_audit.json`
  - `benefit_bucket_examples.json`
- 关键问题：
  - revision 修了谁
  - revision 伤了谁
  - revision 对谁无效
- 允许的扩展：
  - shallow vs deep failure
  - reasoning vs code boundary
- 不允许的过度 claim：
  - 完整 failure taxonomy
  - 跨任务 regime law

### 协议护城河

`Runtime-Lineage / Honest-Compute Protocol`

- 关键 artifact：
  - `canonical_asset_manifest.json`
  - `runtime_fairness_matrix.json`
  - `observer_controller_protocol.json`
- 关键作用：
  - reviewer-facing auditable mapping
  - 避免 method effect 与 implementation confound 混淆
  - 给主证据层提供可信比较条件

## serious execution candidates

round 2 后建议 serious execution candidates 只保留 2 个：

1. `Benefit-Bucket / Recoverability Analysis`
2. `Runtime-Lineage / Honest-Compute Protocol`

`Observer-Controller Split` 保留为主问题 framing / falsification-style claim，不再作为独立执行候选。

## planning 优先级

1. 先做 `benefit_bucket_audit.json`
2. 并行做最小 `seed_sensitivity_spotcheck.json`
3. 立刻补 `canonical_asset_manifest.json` 与 `runtime_fairness_matrix.json`
4. 最后再把 `observer/controller split` 收口到 proposal / intro / discussion 语言

## 需要 round 3 最终确认的问题

1. `Observer-Controller Split` 是否只保留为 framing，还是应进入 contribution bullet？
2. `Runtime-Lineage Protocol` 是独立 contribution，还是仅作为 main evidence 的 support lane？
3. abstract 里应该保留几个 headline claim，2 个还是 3 个？
4. `seed_sensitivity_spotcheck` 应被写入主 proposal deliverables，还是只写进 planning appendix？
