# Strategist

## 1. 最有希望的方向

当前最有希望的方向是把 `cand_espd` 明确收敛成一条 **可信但克制的 speed-line contribution**。理由很清楚：

- full-scale `GSM8K` 上，`cand_espd=0.4041`，高于 `RAND-84=0.3980` 与 `CARD-84=0.3965`
- 对 matched sham `ESPD-FixedFrontier` 同时保住了 `+0.53pp` 的质量差与 `+18.69 tok/s` 的速度优势
- 但 shared controls 的 absolute throughput 仍更快，因此这条线最适合写成“更好的 speed/quality trade-off”，而不是“全面更快”

因此 iteration 4 的主线不应再维持 proposal 中“`cand_bsr` 是默认 front-runner”的排序，而应改成：**当前有 full-scale 证据的是 `cand_espd`，object-level line 退回 challenger 身份。**

## 2. 优先级明确的下一步

### Priority 1

先把 `cand_espd` 的 claim scope 锁死，然后做一个额外关键验证：

- 第二个 benchmark，或第二个 DLM
- shared controls 与 fixed-frontier sham 必须同时保留
- 目标不是大规模，而是验证 `candidate > sham` 是否外推

### Priority 2

补 runtime-lineage / honest-compute 叙事闭环：

- 把 `equal-quality speed`、raw wall-clock、extra forwards、auxiliary overhead、batch、VRAM 放进统一 artifact
- 明说当前结论只适用于 `eager + no-compile + no-flash` contract

### Priority 3

对 `cand_bsr` 只保留低成本 continuation option：

- 做小规模 confirmation pilot
- 不开新的 full-scale
- 等 `cand_espd` 的外推验证回来后再决定是否升主线

### Priority 4

工程优化单独做，不和方法增益混写：

- max-safe batch probing
- `flash_attention_2`
- `torch.compile`
- method split multi-GPU

## 3. 资源分配建议

- `60%` 资源投给 `cand_espd` 的第二个关键验证
- `25%` 资源投给 runtime artifact / reviewer packaging
- `15%` 资源保留给 `cand_bsr` 的低成本继续探索

核心逻辑：当前最稀缺的不是“更多候选”，而是“把已经出现的 full-scale 正信号变成 submission-grade 证据”。

## 4. 风险评估

- `cand_espd` 最大风险不是崩塌，而是贡献过窄；如果第二次外推失败，它会迅速退化成单案例 speed artifact。
- `cand_bsr` 最大风险是抢走主线，导致 iteration 4 再次回到“proposal 强、evidence 弱”的状态。
- 工程优化路线的风险是 narrative contamination；一旦把 compile/flash/batch 提升和方法效果混在一起，当前 full-scale 证据的归因边界会再次失守。

## 5. PIVOT 还是 PROCEED

**结论：PROCEED，但必须是 narrowed PROCEED。**

- 不是继续沿着原 proposal 的双主线平均发力
- 也不是再次大 pivot 回 object-first narrative
- 而是以 `cand_espd` 作为 iteration 4 当前主线，完成它的外推验证与 reviewer 包装
- 同时把 `cand_bsr` 降为 challenger，而不是默认赢家

如果下一次关键验证仍保持 `candidate > fixed-frontier` 且不对 shared controls 崩塌，就继续推进；若外推失败，再正式考虑回到 object-level 主线。
