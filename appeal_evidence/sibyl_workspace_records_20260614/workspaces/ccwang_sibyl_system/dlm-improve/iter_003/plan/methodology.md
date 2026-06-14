# Iteration-3 Negative-Case Planning Methodology

## 1. Planning Envelope

- Workspace: `current` (`iter_003`)
- Planning detail: `samples=100`, `seeds=[42]`, `timeout=900s`
- Selected candidate: `cand_negative_audit_pivot`
- Planning mode: **post-pilot consolidation**, not new controller exploration

本轮 planning 不再设计新的 pilot arm，也不再为 `cand_audit_mainline` 追加补救性实验。原因不是偷懒，而是当前 `idea_validation_decision` 已明确要求：继续推进负案例主线，且不要在本轮重新打开 controller family。`pilot_evidence_closure_v1` 已经完成，本轮计划的核心是把它转成 reviewer-safe、paper-ready 的 evidence package。

## 2. Goal And Claim Ceiling

目标不是再证明 `CARD-84` 有效，而是把当前结果稳定写成一个 **audited negative case study**。

允许的最强结论上限：

1. `CARD-84` 相对 `DNB-84` 在 GSM8K 上存在局部信号；
2. 该信号未能与 `RAND-84` 清晰分离，因此不能上升为正向 controller claim；
3. entropy 最多只保留为 `risk marker`，不能写成已验证的 intervention rule；
4. 当前 evidence scope 只覆盖 audited slice 与 current-only artifacts，不支持 broader benchmark generality。

## 3. Why No New Pilot Is Planned

本轮 planner detail 虽然仍给出 `samples=100, seed=42`，但这组预算已经被 `pilot_evidence_closure_v1` 完整消费并产出决定性结论。继续重跑同一 pilot 不会改变当前决策，反而会让 proposal 滑回“也许再试一次就能赢”的错误方向。

因此，本轮 setup / pilot_experiments 应服务于以下事情：

1. 验证现有 evidence bundle 仍然自包含；
2. 把负案例的 claim ceiling 冻结成可执行文本；
3. 整理 paper-ready tables、figure specs 与 harm profile；
4. 把 reviewer 可能继续追问的 follow-up 单独记成 future work，而不是偷偷变成新主线。

## 4. Evidence Bundle To Reuse

### 4.1 Required existing artifacts

- `current/exp/pilot_evidence_closure_v1/setup/sample_manifest.json`
- `current/exp/pilot_evidence_closure_v1/setup/runtime_contract.json`
- `current/exp/pilot_evidence_closure_v1/setup/batch_probe.json`
- `current/exp/pilot_evidence_closure_v1/arms/dnb64/seed42/metrics.json`
- `current/exp/pilot_evidence_closure_v1/arms/dnb84/seed42/metrics.json`
- `current/exp/pilot_evidence_closure_v1/arms/card84/seed42/metrics.json`
- `current/exp/pilot_evidence_closure_v1/arms/rand84/seed42/metrics.json`
- `current/exp/pilot_evidence_closure_v1/analysis/per_sample_audit.csv`
- `current/exp/pilot_evidence_closure_v1/analysis/transition_matrix.csv`
- `current/exp/pilot_evidence_closure_v1/analysis/claim_to_asset_map.json`
- `current/exp/pilot_evidence_closure_v1/analysis/code_failure_modes.md`
- `current/exp/pilot_evidence_closure_v1/analysis/summary.md`
- `current/exp/pilot_evidence_closure_v1/analysis/decision.json`

### 4.2 Interpretation policy

- `DNB-84` 承担 `compute-matched active control`
- `RAND-84` 承担 `sham control`
- `per_sample_audit.csv` 与 `code_failure_modes.md` 承担 `harm profile`
- `runtime_contract.json` 只承担“当前运行条件被显式记录”，不承担“runtime fairness 已被完全解决”

## 5. Planning Tasks

### 5.1 Validation task

先验证现有 pilot artifacts 是否仍然 current-only、joinable、路径自包含。若 validation 失败，允许修补路径、索引和 packaging 文件；不允许借机重开实验。

### 5.2 Claim freezing task

把 proposal 中的 paper object 转成一个可执行 claim scope map，明确：

- allowed wording
- forbidden wording
- `risk marker` vs `intervention rule`
- 当前必须主动承认的局限

### 5.3 Paper-ready packaging task

从现有 analysis 产物中整理：

- 主结果表
- repair/harm 对照表
- MBPP harm profile 表
- figure specs

这些输出服务于写作，而不是服务于新实验。

### 5.4 Follow-up gap list task

单独写一个 reviewer-driven follow-up 列表，记录未来若被追问时最小可接受的补强方向，如更强 sham control 或 external-prior reinterpretation。该列表必须明确标记为 **future work only**。

## 6. Metrics And What They Mean

### 6.1 Primary numbers

- `accuracy`
- `fixed`
- `harmed`
- `unchanged_correct`
- `unchanged_wrong`
- `net_repaired`

### 6.2 Current interpretive rule

- `CARD-84 > DNB-84` 只说明局部信号存在
- `CARD-84 ≈ RAND-84` 说明正向 controller claim 不成立
- `MBPP` 未分离 + harm taxonomy 说明 task dependence 只能写成边界，而不能写成机制解释

## 7. Pass / No-Go Rule For This Planning Stage

本轮 planning 只有一个 pass 条件：

> 负案例主线必须被转成一个 current-only、sample-level 可复核、语言边界清楚的 paper-ready package。

若仍出现以下任一情况，则 planning 视为失败：

1. 计划重新打开 `cand_audit_mainline`
2. 计划默认重跑已完成的 pilot
3. proposal wording 仍允许 almost-win salvage
4. runtime contract 被写成强贡献而不是显式记录条件

## 8. Expected Visualizations

- Table 1: 四个实验臂在 GSM8K / MBPP 上的主结果表
- Figure 2: `fixed / harmed / unchanged` stacked bar chart
- Figure 3: `DNB-84` 与 `RAND-84` 如何改写对 `CARD-84` 的解释示意图
- Figure 4: MBPP harm profile bar chart
- Appendix Table: artifact integrity checklist
- Appendix Table: allowed vs forbidden claim wording table

## 9. Risks

1. proposal 仍可能偷偷滑回 `almost works` 叙事；
2. reviewer 可能认为这只是“认真做对照”，而不是可发表对象；
3. runtime assets 若继续被写大，会削弱整篇可信度；
4. future work 列表若不隔离干净，会重新污染当前主线。
