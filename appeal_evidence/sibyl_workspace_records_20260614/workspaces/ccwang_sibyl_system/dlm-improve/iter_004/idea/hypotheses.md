# Iteration 4 Hypotheses

## 总原则

本轮假设不再服务于 `MGCD comeback`。它们服务于一个新的、围绕 **局部 repair object** 与 **entropy-routed compute** 的 candidate program。

## H1: Local Repair Object Hypothesis

> 若 repair object 才是当前 training-free DLM revision 的关键错位，那么 `cand_bsr` 应在 matched compute 下首次超过 `RAND-84`，并主要通过减少 stable-region harm 获得收益。

### 通过信号

- `cand_bsr > RAND-84`
- `harmed stable tokens` 下降
- touched span 更局部、边界更稳定

### 否证条件

- `cand_bsr <= RAND-84`
- 或收益主要来自更大范围重写

## H2: Entropy-As-Routing Hypothesis

> 若 `raw entropy` 更适合作为 routing / stopping 信号，而不是直接作为 semantic controller，那么 `cand_espd` 应能在不显著恶化质量的前提下实现明显速度收益。

### 通过信号

- `cand_espd` 获得显著 wall-clock / throughput 提升
- `active frontier ratio` 明显收缩
- 与 `RAND-84` / `CARD-84` 相比没有出现明显质量崩塌

### 否证条件

- 速度收益不显著
- 或速度提升伴随明显质量下降
- 或收益只在 runtime mismatch 下出现

## H3: Benefit-Gating Hypothesis

> 若 `observer -> action` 之间还缺一个 benefit estimator，那么 `cand_ugr` 应在 plain `cand_bsr` 之上提高 repair/harm ratio，并减少无效 revision。

### 通过信号

- `cand_ugr` 的 `repair/harm` 比高于 `cand_bsr`
- touched islands 更少但净收益不下降

### 否证条件

- 与 `cand_bsr` 相比没有更好的 repair/harm ratio
- 或 uplift 层引入 hidden branching / hidden compute

## H4: Runtime-Parity Robustness Hypothesis

> 任一 candidate 的 gain 只有在统一 runtime contract 下仍成立，才可解释为方法增益。

### 通过信号

- backend 一致
- compile 策略一致
- max-safe batch size 经实际 probe 确认
- matched compute / matched wall-clock 约束被满足

### 否证条件

- 某 candidate 依赖不同 backend / compile 才显得更强
- batch 或 hidden compute 不可比

## H5: Structured-Code Robustness Hypothesis

> 在 `MBPP` 这类结构化任务上，局部 repair + syntax guard 应比 plain revision 更能保持 `syntax-valid` 与 `exec-valid`。

### 通过信号

- `cand_bsr + syntax_guard` 提高或至少不恶化 `syntax-valid`
- `exec-valid` 不因局部修复而显著下降

### 否证条件

- 结构化约束没有带来更好的 valid-rate
- 或 code task accuracy 提升只是用大量 syntax breakage 换来的

## Screening Gate

### 进入 MBPP 的条件

- `cand_bsr` 在 `GSM8K audited slice` 上超过 `RAND-84`

### 进入 conditional UGR pilot 的条件

- `cand_bsr` 或 `cand_espd` 至少有一条出现真实正信号

### 进入 full benchmark 的条件

- H1 未被否证
- H4 未被否证
- candidate metadata、proposal 与 pilot artifacts 保持一致

## Archived Negative Statements

- `cand_mgcd` 不再是 serious hypothesis，只是 archived negative control
- `cand_dsg` 不再是 serious hypothesis，只是 lighter negative control
