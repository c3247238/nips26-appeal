# Idea Validation Decision

- Decision: `ADVANCE`
- Selected candidate: `cand_negative_audit_pivot`
- Confidence: `0.96`

## 为什么

- pilot 已经把原问题判清楚了：`CARD-84` 虽然击败了 compute-matched `DNB-84`，但没有通过更严格的 sham-control gate，`card84_vs_rand84` 在 GSM8K 上只有 `net_repaired = 1`，低于门槛 `>= 2`，因此不再支持正向 controller 主张。
- 这次 pivot 不是“待执行”，而是已经在 workspace 内落地：`proposal`、`idea debate synthesis`、`planning` 和 packaging 产物都已经统一切到 `cand_negative_audit_pivot`。
- 四个 packaging 任务现已全部完成，且 current-only artifacts 可 join；因此当前正确动作是沿着已选定的 negative-case lane 继续推进，而不是再回到新一轮 debate。

## 下一步

- 保持 `cand_negative_audit_pivot` 作为 iteration 3 唯一激活主线。
- 直接带着 `runtime_contract.json`、`per_sample_audit.csv`、`transition_matrix.csv`、`claim_to_asset_map.json`、`code_failure_modes.md` 和新生成的 packaging 结果进入后续综合/写作阶段。
- 严格保持 wording ceiling：entropy 只能写成 `risk marker`，不能写成 validated intervention rule；`CARD-84` 不能写成 winning method。
- 本轮不再重开 controller family、trajectory addon 或新的 pilot 分支，默认沿 negative-case framing 向前推进。
