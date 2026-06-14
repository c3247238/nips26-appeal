# Iteration 4 Result Summary

## Current Stage

当前 workspace 已完成 `experiment_cycle`，正在进入 `result_debate`。本文件用于给结果辩论提供最新、统一的证据入口。

## Full-Scale Bundle Verdict

- bundle: `espd_fullscale_bundle_v1`
- decision: `ADVANCE`
- dataset: `GSM8K`
- sample_count: `1319`
- selected_candidate: `cand_espd`

## Main Quantitative Results

### Candidate: `cand_espd`

- accuracy: `0.4041` (`533/1319`)
- equal-quality speed: `124.42 tok/s`
- avg_nfe: `67.93`
- active_frontier_ratio: `0.1211`
- effective_batch_size: `54`
- comparisons:
  - vs `CARD-84`: `fixed=52`, `harmed=42`, `net_repaired=+10`
  - vs `RAND-84`: `fixed=73`, `harmed=65`, `net_repaired=+8`

### Sham / Mechanism Control: `ESPD-FixedFrontier`

- accuracy: `0.3988` (`526/1319`)
- equal-quality speed: `105.73 tok/s`
- avg_nfe: `68.0`
- active_frontier_ratio: `0.1211`
- effective_batch_size: `52`
- comparison vs `cand_espd`: `fixed=57`, `harmed=62`, `net_repaired=-5`

### Shared Controls

- `CARD-84`: accuracy `0.3965`, speed `126.08 tok/s`, `net_repaired_vs_rand=-2`
- `RAND-84`: accuracy `0.3980`, speed `128.00 tok/s`

## What The Evidence Currently Supports

1. `cand_espd` 在统一 runtime contract 下没有质量崩塌，且相对 `ESPD-FixedFrontier` 保留了 `+0.53pp` accuracy 与 `+18.69 tok/s` 速度优势。
2. `entropy` 更像 **routing / stopping** 信号，而不是直接 semantic controller；固定 frontier sham 没能复现候选的综合表现。
3. `cand_espd` 对 shared controls 的优势是真实但不大：相对 `RAND-84` 只高 `+0.61pp`，相对 `CARD-84` 只高 `+0.76pp`。
4. 这轮 full-scale bundle 支持把 `cand_espd` 作为 serious speed-line contribution 继续推进，但还不足以写成压倒性 headline claim。

## Risks And Open Questions

1. shared controls 速度仍更快，说明 `cand_espd` 的优势不是“绝对吞吐最大”，而是“在近似质量带下保留更好 speed/quality trade-off”。
2. `proposal.md` 里原本把 `cand_bsr` 设为 quality front-runner；本轮 full-scale 主要验证的是 `cand_espd`，因此对象线主张仍未被这组 full-scale 结果直接证成。
3. `compile` 与 `flash_attention` 本轮都未启用，runtime story 仍应表述为当前 contract 下的结果，而不是全局最优工程实现。
4. publication-facing narrative 必须诚实区分：
   - `cand_espd` 的证据是 full-scale `GSM8K`
   - `cand_bsr` / object-level framing 仍主要来自 proposal 与 earlier pilot rationale

## Recommended Debate Focus

1. `cand_espd` 的 gain 到底是足够强的研究信号，还是只够作为工程上温和但可信的 speed-line?
2. `ESPD-FixedFrontier` 的负结果是否足以支持“entropy-routed compute”这一机制解释?
3. 在 `cand_espd` full-scale 已经给出正信号的情况下，iteration 4 的主线应该 `PROCEED` 还是继续 `PIVOT` 回 object-level line?

## Primary Artifacts

- `exp/results/espd_fullscale_bundle_v1.json`
- `exp/results/espd_gsm8k_full_v1.json`
- `exp/results/espd_fixed_frontier_gsm8k_full_v1.json`
- `exp/results/gsm8k_controls_full_v1.json`
