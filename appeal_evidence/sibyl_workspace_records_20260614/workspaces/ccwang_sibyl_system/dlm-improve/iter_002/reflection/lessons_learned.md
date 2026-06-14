# 本轮迭代教训

## 必须改进

- [headline 必须严格降口径]: 下一轮所有摘要、引言、结果表述都要统一成 `n=100 audited slice` 与 `3-seed sign consistency only`，不能再让 reviewer 读出 full-benchmark win 的印象。
- [evidence bundle 必须自包含]: 补出 `exp/results/summary.md`，并把所有 `/home/ccwang/...` lineage 改成 workspace-relative 或 iteration-relative；交付前必须验证 `current/` 单独拷走也能复核。
- [observer-controller 只能写成 framing]: 当前证据只足以支持 protocol framing，不足以支持更强的科学定律式表述；没有 matched empirical test 就不要再拔高。
- [runtime fairness 要拆清 current vs historical]: 同一个 fairness bundle 里不能继续混放 current audited runtime contract 与历史 batch/compile 行，否则 honest-compute 仍会被看成 implementation confound。
- [效率 telemetry 要补任务时序]: 下一轮若还要讨论 GPU 利用率，必须先让 `gpu_progress.json` 记录任务开始/结束、占用 GPU 与并行快照。

## 需要注意

- [当前的 compile + batch probe 是进步，但不是终点]: safe batch 57 是 768-token safety envelope，不等于真实 prompt 分布下的最大吞吐配置。
- [proposal 语言不能继续强于实际交付]: 如果只交付 `_pilot` 资产，就要在 proposal/manuscript 里明确承认 pilot-slice scope。
- [缺少 self-check diagnostics 就等于少一层保险]: 如果下一轮还以可复核性和效率为重点，最好把 self-check 也作为标准 artifact 留档。

## 做得好的（继续保持）

- [证据驱动收缩叙事]: method-forward 转向 diagnostic/protocol 不是包装，而是被 JSON 资产逼出来的正确收缩。
- [bucket + seed + runtime 三件套已成形]: 这一轮终于有了最小闭环证据包，为 focused revision 打下了基础。
- [NO_GO 决策是正确的]: 没有为了追新方法再开 controller probe，避免了 story drift 和额外实现混淆。
- [多路审查已形成稳定共识]: critic、supervisor 和 final critic 的问题定位高度一致，下一轮可以更聚焦地修正，而不是重新找方向。
