# Result Debate Verdict

## Score

**6.7 / 10**

## Key Conclusion

iteration 4 当前最可信的结论不是“我们已经找到了正确 repair object”，而是：

> 在统一 runtime contract 下，`cand_espd` 作为 entropy-routed compute 方案，相对 `ESPD-FixedFrontier` sham 展现出可信但幅度不大的 full-scale 正信号，值得继续推进。

## What We Actually Learned

1. `cand_espd` 是当前唯一被 full-scale 结果直接支持的 serious line。
2. entropy 更像 router / stopping signal，不像 semantic controller。
3. fixed-frontier sham 没能复现 candidate 的质量/速度组合，因此 routing 位置选择本身有证据价值。
4. 当前结果幅度小、统计不稳、外推未做，不能写成 breakthrough。
5. proposal 排序必须更新：`cand_espd` 升主线，`cand_bsr` 降为 challenger。

## Decision

**PROCEED**

但这是 **narrowed PROCEED**：

- 继续推进 `cand_espd`
- 收窄 claim scope
- 先补最小方法学缺口，再扩大 narrative

## Immediate Action Plan

1. 生成 runtime-lineage artifact，统一展示 wall-clock、equal-quality speed、batch、VRAM、extra forwards、auxiliary overhead。
2. 做一个关键外推验证：第二个 benchmark 或第二个 DLM，并保留 shared controls + fixed-frontier sham。
3. 做一个 routing/stopping 拆分 ablation。
4. `cand_bsr` 仅保留低成本 continuation，不开新的 full-scale。
