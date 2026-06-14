# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-12  
**模型**: 本地 Codex 审查回退模式（当前 runtime 未提供 `mcp__codex__codex`，按 skill 兜底要求执行独立第三方本地审查）

## 评审意见

### 主要发现

#### 1. 最终 `proposal.md` 已基本守住方向，没有明显滑回 `TIGER hero` / `Calibration-Aware` / generic controller

这是这轮 debate 最大的正面结果。  
最终 proposal 已明确把旧方向排除，并把论文定位收束为 diagnostic / protocol paper，而不是方法论文，[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L5) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L16)、[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L123) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L129) 都很清楚。

我没有在最终 proposal 里看到实质性“回头”。  
round1 的 perspectives 和早期 debate 里确实还保留过 `Minimal Controller` 讨论，但在 round2/round3 和最终 proposal 中已被降为 appendix-only optional probe，这个收口是成功的。

判断：这一项通过。

#### 2. proposal 的层级基本清楚，但“主命题”与“headline claims”之间还有一处内在张力，建议再统一一次

最终 proposal 同时在做两件事：

- 一方面把 `observer/controller split` 定义为上位主命题，[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L20) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L28)、[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L32) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L43)
- 另一方面又在 abstract 建议里把 headline claims 压成 “bucket + runtime fairness”，并把 observer/controller split 降为 framing sentence，[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L188) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L195)

这两者并不完全矛盾，但在 reviewer 眼里会形成一个问题：

- 到底 paper 最想 claim 的是什么？
- 是一个上位 diagnostic finding？
- 还是两个更实的 artifact-driven findings？

我的建议不是改方向，而是只保留一个更一致的说法：

- 主命题仍然保留 `aggregate gain 不足以解释真实效应结构`
- `observer/controller split` 作为这句主命题里的关键解释对象
- abstract contribution bullets 只保留 2 条：
  - `bucket-level recoverability evidence`
  - `realized compute / runtime-lineage credibility`

这样层级才会前后一致。

判断：结构可用，但还需要一次措辞去歧义。

#### 3. 当前最危险的不是旧 hero drift，而是把“局部观察”写成“规范性 reporting requirement”

proposal 的主命题后半句目前写成：

> “因此结论必须同时结合 bucket-level outcomes 与 realized compute fairness 来解释。”  
见 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L22)

以及：

> “DLM revision 不应只按 aggregate gain 报告……”  
见 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L195)

这类写法是本轮最容易 evidence outrun 的地方。  
原因不是方向错，而是当前资产更像：

- 在当前 tested policies / current evidence scope 下，
- 只报 aggregate gain 会隐藏关键信息，
- 因此我们提供 bucket + runtime-lineage 这套更完整的诊断视角

而不是：

- 我们已经足以提出一个更广的 reporting norm

这点在 `theoretical_on_seed.md` 和 `contrarian_on_seed.md` 里其实已经被很好地提醒过，[theoretical_on_seed.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round3/theoretical_on_seed.md#L44) 到 [theoretical_on_seed.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round3/theoretical_on_seed.md#L63)、[contrarian_on_seed.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round3/contrarian_on_seed.md#L31) 到 [contrarian_on_seed.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round3/contrarian_on_seed.md#L51)。

建议把所有“必须”“统一采用”“标准评价框架”类表述继续降成：

- `in our tested setup`
- `under the current evidence scope`
- `provides a more complete diagnostic view`

判断：这是当前 proposal 最大的真实风险。

#### 4. deliverables 基本够支撑 planning，但 `cand_protocol` 的完成标准还不够具体验收化

正面评价：

当前 deliverables 已经明显优于前几轮，至少把最重要的 6 个交付物都写死了，[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L131) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L148)：

- `benefit_bucket_audit.json`
- `benefit_bucket_examples.json`
- `canonical_asset_manifest.json`
- `runtime_fairness_matrix.json`
- `observer_controller_protocol.json`
- `seed_sensitivity_spotcheck.json`

这已经足够进入 planning，不再像空 proposal。

但我仍建议补一层更 reviewer-facing 的完成标准，否则 `cand_protocol` 还会偏抽象：

- `canonical_asset_manifest.json` 必须明确 claim -> table/figure -> JSON source 的一一映射
- `runtime_fairness_matrix.json` 必须显式列出 nominal NFE、actual NFE、latency、throughput、batch、backend、compile
- `observer_controller_protocol.json` 必须写清 `d(s)` / `g(s)` 的统计定义、数据来源、解释边界
- `seed_sensitivity_spotcheck.json` 必须提前写明只检查 sign consistency，不暗示 full stability proof

这些内容在 debate 里已经反复出现，尤其是 `empiricist_summary.md` 和 `pragmatist_on_seed.md`，[empiricist_summary.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round1/empiricist_summary.md#L89) 到 [empiricist_summary.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round1/empiricist_summary.md#L106)、[pragmatist_on_seed.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round3/pragmatist_on_seed.md#L63) 到 [pragmatist_on_seed.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/debate/round3/pragmatist_on_seed.md#L103)。

判断：planning 已可启动，但最好把 protocol lane 的验收口径再写死一次。

### 次要发现

#### 5. `benefit buckets` 已被正确抬成主证据层，但仍需防止终稿写成“完整 taxonomy”

这一点在所有视角稿里几乎都有共识，尤其是 `contrarian.md`、`empiricist.md`、`theoretical.md`。  
最终 proposal 也已经写了“禁止完整 failure taxonomy”和“禁止跨任务稳定 regime law”，这很好，[proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L55) 到 [proposal.md](/Users/cwan0785/sibyl-system/workspaces/dlm-improve/current/idea/proposal.md#L62)。

这里我的建议只是继续守边界，不需要改方向。

#### 6. round artifacts 中仍有早期 `Minimal Controller` 讨论，后续写作和 planning 不应再反向引用它们

这些文件作为 debate 过程没问题，但后续如果自动汇总脚本会扫全目录，需要确保不会把早期 `候选 C` 重新吸回主提案。  
最稳的做法是在后续综合阶段只引用：

- `proposal.md`
- `proposal_seed_round2.md`
- `round2/*_on_pool.md`
- `round3/*_on_seed.md`

而不要把 round1 的全部候选描述当现行立场。

## 综合判断

### 对 4 个重点问题的回答

1. 是否仍有回到 `TIGER hero` / `Calibration-Aware` / generic controller 的漂移？

- **最终 proposal 没有明显漂移。**
- 早期 debate 文件里有历史残影，但已在最终收口中被成功降级。

2. proposal 的层级是否清楚？

- **大体清楚。**
- 但“主命题”与“headline claims”的关系仍需再做一次一页纸级别的统一表述。

3. claim 是否仍 outrun evidence？

- **有一处中等风险。**
- 主要是从“当前设置下的局部观察”滑向“更广 reporting requirement”的措辞。

4. deliverables 是否足够支撑 planning？

- **基本足够。**
- 但 `cand_protocol` 还需要更明确的验收字段，避免执行时再次变成抽象口号。

## 建议动作

1. 在进入 planning 前，只再改一轮 `proposal.md` 的主命题措辞：
   - 把所有“必须”型规范表述降成 `in our tested setup` / `under current evidence scope`
2. 保持 abstract 只留 2 条 headline claims：
   - bucket-level recoverability evidence
   - realized compute / runtime-lineage credibility
3. 在 proposal 或 planning 文档里补一小段 `cand_protocol` 验收标准：
   - manifest 必含 claim-to-asset map
   - fairness matrix 必含 batch/backend/compile/latency/throughput/NFE
   - observer protocol 必含 `d(s)` / `g(s)` 定义与边界
4. 后续自动综合时，以 `proposal.md + round2 + round3` 为准，不再把 round1 的早期候选重新吸回主线。

## 评分

7.8/10

这已经是一份能进入 planning 的 proposal，但还没有到“可以零风险直接进写作”的程度。  
最主要的问题不是方向错误，而是最后 10% 的措辞收口和 protocol 验收标准还不够硬。
