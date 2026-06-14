# 一个经审计的负案例：更强 sham control 如何改写 Training-Free DLM Revision 的 small-gain 解释

## 一句话 paper object

本文不是方法论文，也不是领域级 protocol 论文，而是一篇 **audited negative case study**：展示一个看似成立的 entropy-guided training-free DLM revision 小增益，在 matched-compute baseline、sham control、sample-level audit 与 current-only artifact closure 下，失去了正向方法解释资格。

## 已锁定的证据结论

### pilot readout

- `DNB-64`: overall `0.17`, GSM8K `0.30`, MBPP `0.04`
- `DNB-84`: overall `0.10`, GSM8K `0.18`, MBPP `0.02`
- `CARD-84`: overall `0.18`, GSM8K `0.32`, MBPP `0.04`
- `RAND-84`: overall `0.17`, GSM8K `0.30`, MBPP `0.04`

### 决定性 gate

- `CARD-84` 相对 compute-matched `DNB-84` 的 GSM8K `net repaired samples = +7`
- `CARD-84` 相对 sham control `RAND-84` 的 GSM8K `net repaired samples = +1`
- 预设 gate `gsm8k_card_beats_rand84_by_2_net_repaired = false`
- `artifacts_joinable = true`
- `MBPP` 没有把 `CARD-84` 与 `RAND-84` 分开

### 当前最安全的解释

在 small-gain regime 下，`compute-matched baseline` 只能排除额外算力解释，不能排除 generic targeting 或 perturbation 带来的伪改进；因此 `sham control` 会实质改变结论。

## 允许写的主张

### 最强允许版本

> 在 current-only、matched-compute、sham-controlled 的审计条件下，一个看似 promising 的 entropy-guided minimal revision 配置虽能相对 `DNB-84` 产生局部 reasoning 修复信号，但未能与 budget-matched random targeting 清晰分离。本文报告的是一个经审计的负案例，以及支撑这一判别的最小审计证据结构。

### 明确不允许的版本

- entropy-guided revision reliably improves DLM reasoning
- observer-guided controller is validated
- CARD-84 is the winning inference method
- 我们提出了一套新的 protocol 范式
- 这是一个差点成功的方法
- reasoning / code divergence 的机制已被解释

## 本文真正的价值

### 价值 1：一个被更强负对照拦下来的 near-miss case

`CARD-84 > DNB-84` 但 `CARD-84 ≈ RAND-84` 不是普通的“方法没赢”，而是一个边界清晰的 near-miss case。它说明在 DLM iterative denoising setting 里，仅做 compute-matched 比较仍不足以支撑正向方法语言。

### 价值 2：最小可复核审计模板

本文不是宣称发明新范式，而是给出一个**最小可复核审计模板**，由四部分组成：

1. `compute-matched active control`：`DNB-84`
2. `sham control`：`RAND-84`
3. `sample-level audit`：`per_sample_audit.csv`、`transition_matrix.csv`
4. `artifact closure`：`runtime_contract.json`、`claim_to_asset_map.json`、`code_failure_modes.md`

这个模板的价值是支持当前负案例的可信解释，而不是把自己写成独立的大贡献。

### 价值 3：把 entropy 降格为 risk marker

当前证据最多支持把 entropy 视为 `risk marker`，而不是已验证的 intervention rule。也就是说，它可能标记了脆弱样本，但还不能支撑“entropy-guided targeting 优于 budget-matched random targeting”的更强说法。

## 论文不写什么

- 不写正向 controller 论文
- 不写 general DLM gain
- 不写 protocol-first manifesto
- 不写 theorem-like identifiability
- 不写 trajectory mechanism 作为主贡献
- 不写 runtime fairness 已被充分解决

## 叙事骨架

### opening

training-free DLM inference 很容易产生“小而诱人的 gain”。危险不在于这些 gain 一定为假，而在于它们太容易在缺少 sham control 的条件下被过度解释。

### middle

我们选择一个最小 reference case，并在最不利于正向叙事的条件下重新解释它：`DNB-84` 排除额外 compute，`RAND-84` 排除 generic targeting，sample-level audit 与 current-only artifacts 保证结论可复核。结果表明，局部信号仍存在，但不再配得上正向方法主张。

### landing

因此，本文最可信的结论不是“发现了更好的 revision controller”，而是“在 DLM small-gain regime 下，更强 sham control 会改写对局部增益的解释”。

## 对评审必须主动承认的事

- 当前 evidence scope 是 audited slice，不是 full benchmark generalization
- `CARD-84` 没有击穿更严格的 `RAND-84` gate
- trajectory addon 本轮被明确跳过，不是遗漏
- current runtime contract 是稳定 eager 路径；compile / flash-attn 不构成当前主文卖点
- artifact closure 让负案例可信，但不自动等于 protocol invention

## 下一阶段写作指令

### planning

- 围绕 `audited negative case study` 组织标题、摘要、图表与资产映射
- 不新增 controller family 或额外 pilot 支线
- 所有图表服务于 `DNB-84` / `RAND-84` 如何改写解释，而不是服务于 `CARD-84` 造势

### writing

- 摘要与引言必须先写 negative case，再写 minimal audit template
- 所有正向措辞都要被 `CARD-84 ≈ RAND-84` 约束
- 允许写 `localized signal`，不允许写 `validated efficacy`
- 允许写 `risk marker`，不允许写 `validated targeting rule`

## 一句话提案

这篇 iteration-3 论文要写成：**一个看似成立的 training-free DLM revision 小增益，经过 matched-compute、sham control 与 sample-level audit 后，不再配得上正向方法解释；我们公开这个被更强负对照拦下来的负案例，以及支撑这一判别的最小审计模板。**
