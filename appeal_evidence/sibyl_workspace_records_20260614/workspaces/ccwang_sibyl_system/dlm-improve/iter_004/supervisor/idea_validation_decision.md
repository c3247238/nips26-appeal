# Idea Validation Decision

## Decision

`ADVANCE`

## Why

这轮 phase-1 screening 已经把两个 serious candidates 明确分开了：

- `cand_espd`
  - `screening_verdict = ADVANCE`
  - `quality_at_equal_compute = 0.32`
  - `speed_at_equal_quality_band = 140.45 tok/s`
  - `vs RAND-84`: `net_repaired = +1`
  - `vs ESPD-FixedFrontier`: `quality_gap = 0.0`，但 `speed_gain = +36.11 tok/s`
  - 结论：它给出了这轮最干净的正信号。固定 frontier sham 已经说明收益不只是“ frontier 缩小了”，而是 entropy-routed compute allocation 本身有额外信息量。

- `cand_bsr`
  - `screening_verdict = REFINE`
  - `quality_at_equal_compute = 0.34`
  - `vs RAND-84`: `net_repaired = +2`
  - 但最强 sham `EntropySpan-NoBoundary = 0.36`，高于 `cand_bsr_v1 = 0.34`
  - 结论：它不是失败，而是当前 `boundary-lock` 合同还没站稳。对象线有信号，但现版本还不该直接进入 full experiments。

因此这轮不该 `PIVOT`，因为我们已经有一个可执行的 survivor；也不该整体 `REFINE`，因为 `cand_espd` 已经满足“继续投更多实验预算”的最低条件。最合理的控制平面决策是：

- 只 `ADVANCE` 一个候选：`cand_espd`
- 把 `cand_bsr` 保留为 refine line，而不是作为本轮 selected candidate

## Risks To Carry Forward

1. `cand_espd` 当前最强证据是“优于 fixed-frontier sham”，不是“绝对吞吐已经超过所有 shared controls”；下一阶段必须把这一点诚实写成 full-benchmark risk。
2. `cand_bsr` 的 object-level hypothesis 仍然活着，但 `boundary_story_broken = true`，说明 v1 机制叙事不能直接照搬到 full stage。
3. `cand_mgcd` 与 `cand_dsg` 仍应保持 archived negative controls，不允许重新包装成当前 survivor。

## Required Next Moves

1. 进入 `experiment_cycle` 时只让 `cand_espd` 成为 `selected_candidate_id`，并围绕 unified runtime contract 做更完整的 speed/quality validation。
2. 在 full-stage 计划里显式加入 reviewer-facing 风险检查：`cand_espd` 必须继续报告相对 `CARD-84` / `RAND-84` 的绝对速度与质量，而不只报告相对 `FixedFrontier` 的机制优势。
3. 把 `cand_bsr` 改写成 `cand_bsr_v2` 的 refine track：重新审视 boundary lock，而不是继续宣传 `v1` 已经是稳定赢家。
4. 维持 `cand_mgcd`、`cand_dsg` 为 dropped / archived negatives，避免旧 front-runner 叙事回流。

SELECTED_CANDIDATE: cand_espd
CONFIDENCE: 0.78
DECISION: ADVANCE
