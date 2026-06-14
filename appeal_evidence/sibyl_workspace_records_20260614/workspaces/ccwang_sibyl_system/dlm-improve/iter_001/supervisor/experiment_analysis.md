# 实验结果分析

## 实验结果概要

本轮 full-cycle 结果已经不再支持 `TIGER` 作为前台方法主张，但清楚支持一条更稳的 `cand_diag` 诊断论文路线。

关键发现：

1. `diag_compute_curve_gsm8k.json` 证实 honest compute 会改变关键比较与 Pareto 叙事：
   - `CORE-proxy-64` 出现 `7.81%` 的 nominal-vs-actual compute 偏差
   - 至少有一组 pairwise comparison 在 actual compute 下发生重排
2. `diag_signal_gap_audit.json` 证实 observer-controller mismatch：
   - calibration `diagnostic_score = 0.6225` 但 `control_effectiveness = 0.0`
   - entropy 也表现出“诊断有用、控制不稳”的结构
3. `diag_math500_shortlist.json` 显示 GSM8K 排序不能直接外推到第二 reasoning benchmark，支持 task-dependent revision response。
4. `diag_humaneval_guard_boundary.json` 显示 code 侧的 cheap guard 只修浅层 syntax failure，不能恢复 runtime / pass@1，适合 appendix-only boundary positioning。

同时，Codex 独立审查明确指出：

- 当前 framing 绝不能回到 method-forward / calibration-aware 叙事
- calibration protocol、compute accounting 与 benchmark 定位仍有明显风险
- 当前最稳定位是 `compute-normalized diagnostic study`，而不是 benchmark standard-setter

## 各方观点总结

- 乐观者：四个诊断结果已经形成可发表 backbone，最强价值是把 honest compute、observer/controller mismatch 与 recoverability 放到同一张图里。
- 怀疑论者：当前最大风险是把 `n=100/50` 的 diagnostic slice 写成完整 benchmark；最危险缺口是 lack of uncertainty / full-scale support。
- 战略家：应继续沿 `cand_diag` 前进，但主文只讲 honest compute、observer/controller mismatch、task-dependent response，其他全部收缩到 appendix。

## 分析

### 1. 方法可行性

如果问题是“`TIGER` 这个方法故事是否成立”，答案是否定的；但如果问题是“`cand_diag` 这个诊断主张是否成立”，答案是肯定的。当前方向已经不再依赖某个新 controller 必胜，而是依赖一组相互支撑的诊断证据。

### 2. 性能表现

这轮结果不支持“更强方法”主张，但支持“更诚实比较口径 + 更清晰 failure taxonomy”主张。也就是说，headline 不应是多赢几个点，而应是：

- nominal compute 不够诚实；
- 强 observer 不等于强 controller；
- revision 的收益边界受 recoverability 支配。

### 3. 改进空间

当前方向仍有非常明确、且工作量可控的下一步：

1. 补 `benefit_bucket_audit`
2. 补 runtime fairness appendix 表
3. 做最小 `seed_sensitivity_spotcheck`
4. 明确 `asset_lineage_summary`

这些动作都不是重开新方法，而是把当前 diagnostic story 从“说得通”推进到“更可 defend”。

### 4. 时间成本

如果现在再 PIVOT，意味着重新建立另一套候选方案的 pilot 闭环，时间成本明显更高；而继续沿 `cand_diag` 推进，只需要针对现有证据补最关键的机制资产，边际成本更低、路径更短。

### 5. 怀疑论者的批评是否致命

`skeptic` 与 Codex review 提出的批评很严厉，但并不构成“必须重开方向”的致命否定。它们真正否定的是：

- 过大的 benchmark 命名
- 过强的一般性机制定律
- 任何 method-forward / calibration-aware 夸大写法

只要我们接受这些收缩条件，这些批评会把项目推向更诚实的论文，而不是逼迫再次 PIVOT。

## 决策理由

我选择 `PROCEED`，因为：

1. 当前方向的核心假设并未被否定，只是需要严格收缩主张强度；
2. 结果已经形成 coherent diagnostic story，且比重新开新方向更接近可发表状态；
3. 下一步补强项清楚、工作量可控，并且直接服务于当前主线；
4. 再次 PIVOT 的收益预期低于沿 `cand_diag` 做 targeted gap-filling。

但这个 `PROCEED` 有三个硬条件：

1. 不再恢复 `TIGER` 的主方法叙事；
2. 不再把当前结果叫成完整 benchmark standard；
3. 后续所有工作优先服务于 `benefit buckets / failure taxonomy` 和 honest framing。

## DECISION: PROCEED

DECISION: PROCEED
