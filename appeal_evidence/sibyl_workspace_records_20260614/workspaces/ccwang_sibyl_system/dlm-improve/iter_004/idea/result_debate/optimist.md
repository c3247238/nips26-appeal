# Optimist

## 1. 什么奏效了

- `cand_espd` 在 full-scale `GSM8K` `n=1319` 上拿到了当前 workspace 里最硬的一组新证据：accuracy `0.4041`，高于 `CARD-84` 的 `0.3965`、`RAND-84` 的 `0.3980`，bundle verdict 为 `ADVANCE`。
- 这不是靠更高 compute 粗暴堆出来的。`cand_espd avg_nfe=67.93`，和 `CARD-84` 的 `68.0` 基本同级；active frontier ratio 只有 `0.1211`，说明额外 compute 被压缩在小范围高风险区域。
- 最关键的积极信号来自 sham control：`ESPD-FixedFrontier` 同样保留 `0.1211` frontier ratio，但 accuracy 只有 `0.3988`，equal-quality speed 只有 `105.73 tok/s`；`cand_espd` 则达到 `124.42 tok/s`，速度优势 `+18.69 tok/s`，质量优势 `+0.53pp`。
- pairwise repair 也不是虚高：相对 `RAND-84`，`cand_espd` 有 `73` 个 fixed、`65` 个 harmed，净修复 `+8`；相对 `CARD-84`，净修复 `+10`。在 shared controls 已经极其接近的情况下还能挤出净优势，这比弱 baseline 上的大胜更有价值。

## 2. 意外积极发现

- proposal 中 `cand_espd` 原本只是速度线 backup，但现在它实际上成了 iteration 4 里**唯一被 full-scale 结果直接支持的 serious line**。
- 当前运行环境并不理想：`eager` backend、`compile=false`、`flash_attention=false`。在这种保守实现下，`cand_espd` 仍能保住正信号，说明后续工程优化还有上升空间。
- `cand_espd peak_vram_mb=15289`，远低于 fixed-frontier 的 `41973` 和 shared controls 的 `40910`。即使这里还混有实现差异，这仍然暗示 entropy-routed frontier 可能天然更有显存友好性。
- stopped-step histogram 呈现明显两极化：`702` 个样本在第 `1` 步停，`613` 个样本走满第 `3` 步。这种“很快够了”与“确实值得继续修”并存的结构，很像一个可被继续研究的真实 compute-allocation 现象。

## 3. 最有前景的扩展

- 把 `cand_espd` 从“速度线备选”升级成一个清晰的 entropy-routed compute 研究子线。当前最有价值的不是继续加复杂模块，而是把已有 story 讲干净：在 honest-compute 合同下，candidate 比 matched sham 快 `18.69 tok/s`，同时保住了质量。
- 把 `cand_espd` 和 `cand_bsr` 写成互补关系，而不是竞争关系。最新结果支持的是：entropy 更像 observer / router，而不一定是 semantic controller；这恰好为后续 object-level line 提供了更清晰的分工。
- 深挖 frontier 行为本身。当前已有 `avg_tokens_changed=4.16`、`frontier_ratio=0.1211`、`stopped_step_histogram`，完全可以继续分析 1-step stop 和 3-step max 两组样本分别贡献了多少 repair、多少 harm。
- 优先做工程放大而不改算法内核：max-safe batch probing、`flash_attention_2`、`torch.compile`、task-level multi-GPU split。当前信号已经存在，最便宜的放大器就是把工程约束兑现到位。

## 4. 如何往前推进

- 结果辩论阶段应承认：`cand_espd` 已经从 backup 升级为 current mainline candidate。它不是压倒性突破，但它是当前最可信、最完整的一条 full-scale 证据链。
- 下一步最值得做的不是“再想一个更花哨的方法”，而是补一个关键外推验证：第二个 benchmark 或第二个 DLM，只要能再次保住 `candidate > fixed-frontier sham`，这条线就会从 weak positive 变成真正 submission-grade 的 speed contribution。
- paper-facing 叙事要主动拥抱“小而可信”的胜利：shared controls 自己几乎打平，但 `cand_espd` 仍然在 `GSM8K full` 上挤出 `+0.61pp` 到 `+0.76pp` 的质量优势，并保住 `1.1768x` 的 routing-vs-sham 速度差。
- 最乐观也最稳妥的总结是：**entropy-routed compute 已经被 full-scale 结果证明“值得认真推进”，而不是继续停留在 pilot 假说层。**
