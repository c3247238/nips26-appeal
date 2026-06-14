# Candidate Pool After Round 1

## 背景

round 1 的 6 份 perspective 与 summary 已出现明显收敛。当前不再保留“再做一个 generic new controller / scheduler”作为 serious candidate。

## 当前候选池

### Candidate A: Observer-Controller Split

- 核心命题：
  - `observer quality != controller gain`
  - 需要把 `d(s)` 与 `g(s)` 分开定义、分开报告
- 当前定位：
  - 更像主问题或 falsification-style claim
- 主要价值：
  - 为整篇 diagnostic paper 定义研究对象
- 主要风险：
  - 如果写得太满，容易变成漂亮但证据不足的理论包装

### Candidate B: Benefit-Bucket / Recoverability Analysis

- 核心命题：
  - 用 `fixed / harmed / no-effect` 拆开 aggregate revision gain
  - 必要时加入 shallow vs deep failure / reasoning vs code boundary
- 当前定位：
  - 最关键的机制证据与 quality-gate 补件
- 主要价值：
  - 直接回答“revision 到底修了谁、伤了谁”
- 主要风险：
  - 如果直接写成完整 failure taxonomy 或 regime law，会 outrun evidence

### Candidate C: Runtime-Lineage / Honest-Compute Protocol

- 核心命题：
  - DLM inference 比较必须报告 realized compute fairness，而不是只看 nominal steps
  - 需要 canonical asset manifest / runtime fairness matrix / observer-controller protocol
- 当前定位：
  - protocol backbone / credibility shield / reviewer-facing artifact
- 主要价值：
  - 避免 reviewer 用 implementation confound 否定主结论
- 主要风险：
  - 如果只写成 hygiene appendix，会显得正确但不锋利

## 当前共识

- 明确 DROP：
  - `Minimal Controller for Decoupling` 作为独立 serious candidate
  - 任何 `TIGER hero`、`Calibration-Aware`、generic new controller/scheduler
- 当前共识最强的组合是：
  - `B` 作为最急需的证据闭环
  - `C` 作为 protocol / fairness 护城河
  - `A` 作为主问题定义或低承诺 falsification claim

## Round 2 需要回答的问题

1. 三个 candidate 中，谁是“主线”，谁是“支撑层”，谁只能当 appendix-grade artifact？
2. `Observer-Controller Split` 应该是标题级 claim，还是 discussion / framing 级 claim？
3. `Benefit-Bucket` 应该被写成 mechanism evidence，还是仅写成 error analysis / failure audit？
4. `Runtime-Lineage Protocol` 应该是独立贡献，还是只作为 supporting protocol backbone？
5. round 2 结束后是否应该把 serious candidate 从 3 个压缩到 2 个？
