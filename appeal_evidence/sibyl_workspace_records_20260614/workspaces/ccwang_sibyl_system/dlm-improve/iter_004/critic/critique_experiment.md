# 实验批评

## 总评

当前实验最大的优点是：有 full-scale、不是只停留在 pilot；有 sham control、不是只和弱 baseline 比；有 paired repair/harm、不是只报 aggregate accuracy。最大的缺点是：这些优点还没有拼成一个完全无歧义的 mechanism test。

## 主要问题

### 1. fixed-frontier sham 仍是部分不匹配控制

这是当前最关键的实验问题。

full-scale bundle 中：

- `cand_espd` effective batch = `54`
- `ESPD-FixedFrontier` effective batch = `52`
- shared controls effective batch = `57`

同时：

- `cand_espd` peak VRAM = `15289 MB`
- `ESPD-FixedFrontier` peak VRAM = `41973 MB`
- shared controls peak VRAM = `40910 MB`

再加上：

- `cand_espd avg_tokens_changed = 4.16`
- `ESPD-FixedFrontier avg_tokens_changed = 0.29`

这意味着 candidate 与 sham 虽然 frontier ratio matched，但 execution behavior 和 revision behavior 并没有 matched。paper 可以说“stronger sham still fails to reproduce candidate trade-off”，但还不能说“mechanism 已被干净隔离”。

### 2. runtime contract 口径不完全闭合

`iteration4_runtime_contract_v2.json` 的最佳路径是：

- eager
- compile=true
- safe_batch=57

而 full-scale bundle 实际执行是：

- eager
- compile=false
- batch 52-57 之间

这会产生两个 reviewer 问题：

1. 为什么没有沿 frozen best path 跑 full bundle？
2. retained contract 与 best path 不一致时，headline result 到底代表什么？

如果这个问题不在 paper 中主动回答，runtime discipline 会从优点变成弱点。

### 3. 单 benchmark 限制仍然过重

methodology 早就把 `MATH500` 设为第二 reasoning benchmark，把 `MBPP` 设为 boundary validation。现在 full paper 只有 `GSM8K`。这会直接限制你所有更广的实验结论：

- 不能声称是 general dLLM intervention principle
- 不能声称 routing interpretation 在 reasoning tasks 上稳定成立
- 更不能外推到 code or structured outputs

### 4. pilot-to-full 路径缺少实验流程透明化

当前 artifact 里存在三层重要状态：

- `pilot_summary.json`：MGCD/DSG 失败，应回到 idea_debate
- `screening_decision_bundle_v1.json`：BSR refine，ESPD advance
- `summary.md` / `espd_fullscale_bundle_v1.json`：只剩 ESPD full-scale

这条路径实验上是合理的，但 paper 里几乎没写。结果就是外部读者看不到 candidate selection flow，只看到“最后跑的是这个方法”。这很容易被误读为后验筛选。

### 5. 统计层面没有错，但必须更收敛

你已经主动写了 Wilson 区间和 McNemar-style descriptive checks，这很好。但当前 paired counts 还是很小：

- vs RAND-84: `+8`
- vs CARD-84: `+10`
- sham vs candidate: `-5`

因此 paper 最稳的实验语气不是“candidate beats baseline”，而是：

- candidate preserves a bounded positive direction under stronger control

任何再向前半步的措辞都容易显得过。

## 代理指标与 metric gaming 风险

当前并没有明显的“用 proxy gaming 换来 headline gain”证据，但仍有两处需要警惕：

1. `equal-quality speed` 容易掩盖 absolute throughput 未赢 shared controls 这一事实。
2. `active_frontier_ratio` matched 并不等于 touched-token dynamics matched，fixed-frontier 的 `avg_tokens_changed` 过低本身就提示它与 candidate 实际行为不同。

## 实验层面最应优先补的内容

1. 一张 candidate/sham/shared-controls 的 confound matrix。
2. 一次 tighter sham rerun，尽量缩小 batch、tokens_changed、execution behavior 差异。
3. 一个第二 benchmark，哪怕只补一个 reasoning benchmark，也比继续强化单点 GSM8K 叙事更值。
