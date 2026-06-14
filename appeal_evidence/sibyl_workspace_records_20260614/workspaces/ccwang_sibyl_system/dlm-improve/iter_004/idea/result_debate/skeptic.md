# Skeptic

## 1. 统计与显著性顾虑

- headline 非常脆弱。`cand_espd` accuracy `0.4041`，相对 `RAND-84` 只高 `+0.61pp`，相对 `CARD-84` 只高 `+0.76pp`，相对 `ESPD-FixedFrontier` 只高 `+0.53pp`。这不是明显胜出，只是边缘优势。
- 粗略看 Wilson 区间，这几组 accuracy 的 95% 区间高度重叠，没有形成干净分离。
- paired 计数也不够硬：
  - `cand_espd vs RAND-84`: `fixed=73`, `harmed=65`, `net=+8`
  - `cand_espd vs CARD-84`: `fixed=52`, `harmed=42`, `net=+10`
  - `ESPD-FixedFrontier vs cand_espd`: `fixed=57`, `harmed=62`, `net=-5`
  这些更像方向性信号，不像稳固显著优势。
- bundle 的 `ADVANCE` 是工程 gate，不是统计 gate。当前 `quality_tolerance=0.01`、`speed_gain_floor_tok_per_sec=1.0`，更适合做筛选，不适合直接升级成论文结论。

## 2. 混杂与替代解释

- 最大混杂来自实现差异。虽然 candidate 和 fixed-frontier nominal 上共享 `frontier_ratio=0.1211`，但 runtime 明显不同：
  - `peak_vram_mb`: `15289` vs `41973`
  - `effective_batch_size`: `54` vs `52`
  - `auxiliary_overhead_sec`: `2569.81` vs `3049.62`
  这说明 sham 不是纯机制控制，而是“机制 + 实现路径”混合控制。
- `CARD-84` 在 shared controls 内反而略输 `RAND-84`，说明 entropy 信号并没有稳定转化为 semantic revision gain。于是 `cand_espd` 的小优势完全可能只是避开了 `CARD` 的副作用，而不是真正证明了更强机制。
- stopped-step histogram `702 / 4 / 613` 太极端，像硬阈值 artifact，不像平滑的自适应 stopping 机制。
- `avg_tokens_changed` 也不匹配：`cand_espd=4.16`，`ESPD-FixedFrontier=0.29`。如果 intervention intensity 都不接近，就很难说 candidate 是“更聪明”，而不是“做得更多”。

## 3. Proxy-Metric Gaming 检查

- 最危险的 proxy 是 `equal-quality speed`。当前 `cand_espd 124.42 tok/s` 实际仍低于 `CARD-84 126.08` 与 `RAND-84 128.00`。所以 candidate 不是绝对更快，只是相对 fixed-frontier sham 更快。
- 端到端 latency 上，candidate 也是慢的：`2713.81s`，高于 `RAND-84 2638.06s`、`CARD-84 2678.23s`。如果后续只拿 equal-quality band 讲故事，容易被 reviewer 视为换指标后显得更优。
- runtime ledger 显示 `cand_espd` 有 `3868` 次 extra forward passes，绝大多数时间消耗落在 auxiliary bookkeeping 上。若把它表述成“更高效 decode”，会有 accounting artifact 的嫌疑。

## 4. 缺失证据

- 没有 multi-seed stability。
- 没有 primary endpoint 预注册。
- 没有 frontier entropy 与实际 uplift 的相关性分析。
- 没有 reviewer-friendly runtime lineage artifact，把 backend / batch / extra forwards / latency / peak VRAM 放进一张统一表。
- 没有跨任务验证。proposal 里明确点名了 `MBPP`，但当前 full-scale 只有 `GSM8K`。
- 没有把这条 speed-line 和 proposal 主命题重新接起来。当前结果最多说明 `cand_espd` 值得继续，不说明 object-level proposal 已被验证。

## 5. 必须补的实验

- 至少做 `3 seeds` 的 `cand_espd / RAND-84 / CARD-84 / ESPD-FixedFrontier` 重复，并报告 paired bootstrap 或 McNemar interval。
- 做真正 matched-intervention 的 sham：
  - 匹配 `tokens_changed`
  - 匹配 `extra_forward_passes`
  - 匹配 `effective_batch_size`
  - 匹配 `auxiliary_overhead_sec`
- 做 stopping ablation：当前阈值、no early stop、fixed 1/2/3 extra steps、threshold grid。
- 做 touched-token / frontier-entropy calibration：验证高 frontier entropy 是否真的对应更高修复收益。
- 做跨任务复现，优先 `MBPP`，并显式报告 `syntax-valid`、`exec-valid`、accuracy 三者 trade-off。

## 结论

当前最保守、最诚实的结论是：**`cand_espd` 只是一个“值得继续验证的弱阳性 speed-line”，还不是已被显著证成的机制性突破。**
