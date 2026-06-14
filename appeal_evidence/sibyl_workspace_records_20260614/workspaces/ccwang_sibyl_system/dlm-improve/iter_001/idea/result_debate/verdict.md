# Result Debate Verdict

## Score

**7.1 / 10**

## One-Line Conclusion

继续推进 `cand_diag`，但把论文明确写成 **compute-normalized diagnostic study**，不要再写成新 revision 方法或 benchmark standard-setter。

## What We Know

1. honest compute 会改变关键 DLM 方法比较结论。
2. observer 的诊断价值不自动转化为 controller 的控制收益。
3. revision response 具有 task-/recoverability-dependent 特征。
4. code guard 只能证明 shallow safeguard，不证明 deep recovery。

## Biggest Risk

如果不补 `benefit buckets + uncertainty`，这篇稿子会被看成“有意思但不够硬的 pilot diagnostic story”。

## P0

**补 `benefit_bucket_audit`。**

这是把论文从 aggregate result summary 升级成 mechanism paper 的最短一步。

## Immediate Action Plan

1. 先做 `benefit_bucket_audit.json`。
2. 再补 runtime fairness table 与 asset lineage summary。
3. 做最小 seed sensitivity / uncertainty spot-check。
4. 立刻重写 Results 骨架与 framing，主文只保留三条：honest compute、observer/controller mismatch、task-dependent response。
