# Iteration 1 Proposal

## 标题

**RIDE-Bench: Revision in Diffusion LMs under Honest Compute**

## 一句话摘要

把本轮主贡献从“提出更复杂的 revision 方法”切换为“建立一个 compute-normalized 的诊断框架”，系统回答：**revision 何时帮助 DLM，何时伤害 DLM，以及为什么更强的观测信号并不自动转化为更强的控制策略。**

## Evidence-Driven Revisions

这一版 proposal 是在真实 pilot 证据上重写的，不再延续 TIGER 的方法主叙事。

### 被强化的发现

1. `diagnostic_calibration_heldout.json` 表明 entropy / calibration 有稳定诊断价值：
   - 最大 `entropy_error_corr = -0.6225`
2. `tiger_gating_boundary.json` 表明 code failure 确实具有结构脆弱性：
   - gated TIGER 把 syntax failure 从 `0.48` 降到 `0.28`

### 被削弱或证伪的命题

1. `gsm8k_main_shortlist.json` 表明 TIGER 没有超过 entropy revision：
   - `TIGER = 0.39`
   - `Entropy = 0.39`
2. 同一个 shortlist 中，`CORE-proxy = 0.46` 明显更强，说明 TIGER 不具备 method-forward 说服力。
3. code gating 不是通用修复：
   - HumanEval `gated pass@1 = 0.04`
   - 仍低于 `Standard = 0.06`
   - runtime failure 还从 `0.50` 升到 `0.68`

### 因此必须修改的地方

- 不再把 `TIGER` 当成前台主方法
- 不再把 code 结果当成“任务泛化成功”
- 不再把 calibration correction 当作性能因果来源
- 重开候选池，但以 `cand_diag` 为 front-runner

## Motivation

当前 training-free DLM 研究有三个共同问题：

1. 很多方法仍按 nominal steps 比较，而不是按 actual compute 比较
2. revision 常被当成普适增强器，但这轮数据表明它的收益高度依赖任务结构
3. calibration / entropy / instability 等信号往往被直接用作控制律，却缺少对“观测器”和“控制器”角色分离的讨论

如果这些问题不澄清，社区会持续在“更复杂的 revision policy”上投入，但很可能只是重复制造局部而脆弱的收益。

## Front-Runner

### `cand_diag`

**主张**：

构建一个围绕 `Standard / DNB / Prophet / CORE-proxy / Entropy revision / TIGER` 的 compute-normalized diagnostic benchmark，重点不在推出新 sampler，而在回答以下机制问题：

1. revision 的收益是否依赖任务的可恢复性与结构刚性？
2. actual compute 匹配后，training-free 方法排序会如何重排？
3. entropy / instability / calibration 是不是更适合作为诊断观测器，而不是直接控制器？
4. code 边界上的浅层合法性修复，为什么不能自动转化为执行成功？

## Research Questions

1. 在 matched actual compute 下，revision-family 方法与 trajectory-control / baseline-family 方法的相对排名如何变化？
2. reasoning 与 code 是否位于不同的 revision response regime？
3. calibration / entropy / instability 对 revision benefit 的预测力是否高于它们对最终 accuracy 的直接预测力？
4. syntax guard 这样的轻量防护，是否只修复浅层 failure，而不能修复深层结构错误？

## Hypotheses

### H1

一旦用 actual compute 而不是 nominal NFE 对齐，training-free DLM 方法的排名会发生可复现重排，且复杂 revision 方法的优势会缩小。

### H2

revision 的收益是 task-dependent 的：

- reasoning 可能出现小幅提升
- code 更容易出现结构性副作用

### H3

entropy / instability / calibration 对“revision 是否值得触发”的预测力，会高于它们对最终任务正确率的直接解释力；这支持它们作为**诊断工具**而非**直接控制律**。

### H4

syntax guard 等轻量 gating 能降低 parse failure，但无法单独恢复 runtime / semantic correctness。

## Expected Contributions

1. **Compute-normalized ranking**：
   明确展示 actual compute accounting 如何改变方法排序。
2. **Revision phase diagram**：
   用任务结构与 compute 联合刻画 revision 的 gain / harm boundary。
3. **Failure taxonomy**：
   将 syntax failure、runtime failure、semantic failure 分开讨论，而不是只报总分。
4. **Observer vs Controller framing**：
   说明 calibration / entropy / instability 更像观测器，不应自动被神化为控制器。

## Experiment Plan

### Phase A: 诊断切片

- `GSM8K 100`
- `HumanEval 50`
- 方法：`Standard / DNB / Prophet / CORE-proxy / Entropy / TIGER`
- 指标：
  - reasoning: accuracy / actual NFE / latency / TPS
  - code: pass@1 / syntax failure / runtime failure

### Phase B: Benefit Buckets

把样本分成三类：

1. draft 已正确但 revision 伤害
2. draft 错误但 revision 修复
3. revision 几乎无作用

这一步是把总体均值拆成机制证据。

### Phase C: Observer Diagnostics

比较 entropy / instability / calibration 三类信号，对 revision benefit 的预测力，而不是只比较它们对最终 accuracy 的相关性。

## Backup Ideas

### `cand_minimal`

验证极简干预是否已经足够强，命题是：

**simple-enough beats clever**

如果成立，它会成为更锋利的对照论文。

### `cand_factorization`

如果 diagnostics 强烈暗示问题根源在 token 独立更新带来的 factorization error，就转向 dependency-aware grouping / audit。

## What Changed from the Previous Round

1. 从 `TIGER` 方法主线，转成 `cand_diag` 诊断主线
2. 从“继续扩大 benchmark 看 TIGER 会不会赢”，转成“重新定义什么才算真正的 gain”
3. 从“code 作为泛化证据”，转成“code 作为结构边界与 failure analysis”
4. 从“寻找更强控制信号”，转成“区分观测器与控制器”

## Risks

1. benchmark 叙事如果没有机制层概括，会显得像结果整理
2. `cand_diag` 必须给出 phase diagram 与 benefit buckets，否则洞见不够锋利
3. `cand_minimal` 如果太强，可能逼迫我们再次收缩 front-runner

## Current Decision

当前最合理的选择不是继续推进 TIGER，而是：

**以 `cand_diag` 作为新的 front-runner，围绕 honest compute、task recoverability 和 revision failure taxonomy 组织下一轮 planning。**
