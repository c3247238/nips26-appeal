# Pilot Summary For Iteration 4

## Status

iteration 4 的 screening pilot 已执行完成，当前建议是 `PIVOT`，**不要**进入 full benchmark。

## Headline Result

- `cand_mgcd / MGCD-lite` 在 `GSM8K audited slice` 上 `accuracy=0.22`，相对 `CARD-84` 的 `net_repaired=-5`，相对 `RAND-84` 的 `net_repaired=-4`
- `cand_mgcd / MGCD-lite` 在 `MBPP audited slice` 上 `accuracy=0.0`，相对 `CARD-84` 与 `RAND-84` 都是 `net_repaired=-2`
- `cand_dsg / DSG` 在 `GSM8K audited slice` 上 `accuracy=0.30`，相对 `CARD-84` 的 `net_repaired=-1`，相对 `RAND-84` 的 `net_repaired=0`
- `cand_dsg / DSG` 在 `MBPP audited slice` 上 `accuracy=0.02`，相对 `CARD-84` 与 `RAND-84` 都是 `net_repaired=-1`

## Decision-Relevant Reading

这轮 pilot 给出的不是“新方法已经站住”，而是一个比较干净的负结果：

- `MGCD-lite` 在两个 audited slice 上都没有越过 `RAND-84`
- `DSG` 比 `MGCD-lite` 更轻、更干净，但仍然没有越过 `RAND-84`
- 因此 `H1 / Sham-Control Separation` 被否证，当前候选程序里没有 survivor 可以直接进入 full benchmark

## What This Means

按 `pilot_plan.json` 里写死的 gate：

> 如果 `MGCD-lite` 和 `DSG` 都未能超过 `RAND-84`，就返回 `idea_debate`，而不是启动 full benchmark。

所以 iteration 4 现在不该做的是：

- 不推进 full benchmark
- 不把 `MGCD` 继续保留为 front-runner
- 不把 `DSG` 的“更轻但仍不分离”写成 positive signal

更合理的下一步是：

1. 把这轮结果收束成负例证据包
2. 回到 `idea_debate`
3. 重新审视 `repair object` 本身，必要时改为 `BSR` 或新的候选，而不是继续加预算给 `MGCD / DSG`

## Runtime Notes

- 四个 pilot 都在 `batch_size=50` 下完成；safe batch probe 为 `182`
- `MGCD-lite` 两个 pilot 都跑在 `eager + compile=false`
- `DSG` 两个 pilot 都跑在 `eager + compile=true`
- 执行过程中本地 monitor 有 ghost-running / stale state 漂移，最终以远端 `DONE` marker 作为权威完成信号
