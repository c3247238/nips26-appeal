# Revisionist View

## 修正结论

这一轮最重要的知识更新，不是“entropy 可能有点用”，而是我们必须系统性修正对 entropy、targeting 和对照设计的心智模型。旧模型默认：

1. 只要比 compute-matched baseline 好，就足以支持正向方法解释
2. entropy-guided targeting 的局部修复可以自然外推为更强 controller
3. random targeting 只是弱 baselines，不会真正改变解释

现在这三条都需要被修正。

## 1. 我们对 entropy 的认知如何更新

`claim_scope_map.json` 已经把 entropy 的最终地位写得很明确：

- allowed: `risk marker`
- forbidden: `validated targeting rule`

这意味着我们需要从“entropy is a control principle”退到“entropy may flag where things are brittle”。这个变化不是措辞游戏，而是理论位置的实质下调：

1. entropy 可能识别脆弱区域
2. 但识别脆弱，不等于知道怎样介入更好
3. 更不等于能够稳定超出 budget-matched random targeting

## 2. 我们对 targeting 的认知如何更新

`decision.json` 中最关键的更新不是 `CARD-84 > DNB-84`，而是：

- `CARD-84` 对 `RAND-84` 的优势没有过 gate

这迫使我们放弃旧的 mental shortcut：

- “只要 active targeting 比 compute-matched baseline 好，就说明 targeting rule 有效”

新的模型应该是：

1. compute-matched baseline 只能排除 extra-compute 解释
2. sham control 才开始逼近 targeting value 的识别
3. 如果对 sham control 不能充分分离，那么最安全的解释就只能回到 audited negative case

## 3. 我们对 sham control 的认知如何更新

以前 sham control 更像“锦上添花的 rigor”。现在它必须上升为：

- 改写解释的核心工具

因为当前就是一个典型例子：

1. 没有 `RAND-84` 时，`CARD-84 > DNB-84` 很容易被写成 promising revision signal
2. 加上 `RAND-84` 后，叙事中心从“方法小胜”变成“解释被更强负对照改写”

这说明 sham control 在 small-gain DLM setting 中不是附属组件，而是决定结论性质的关键环节。

## 4. 这轮负结果真正教会了我们什么

### 4.1 教会我们小增益最怕被写大

当前结果最有信息量的地方，在于它处在一个非常危险的边界：

- 局部上看，确实有正信号
- 但一旦对照更强，信号就不再足以支撑正向语言

这类 near-miss 情况正是最容易在文献中被过度解释的。

### 4.2 教会我们 paper object 也需要被主动 pivot

本轮真正成功的，不只是实验层面的判断，而是 paper object 的成功 pivot：

- 从“正向 controller 论文”
- 转成 “audited negative case study”

这说明当数据已经否定旧主线时，最好的修正不是继续搜补强实验，而是尽快把对象本身改写成与证据匹配的论文。

### 4.3 教会我们 future work 也必须受审计约束

`followup_gaps.md` 明确禁止：

- 重开 `cand_audit_mainline`
- 默认重跑 `pilot_evidence_closure_v1`
- 用 future work 暗示 `CARD-84` 只是差一点就成立

这很重要。因为如果 future work 继续偷带旧主线，那么当前 negative-case 的诚实边界就会被写作自己破坏。

## 5. 新的心智模型

我建议把当前 iteration 3 的 mental model 更新为：

**在 DLM small-gain regime 下，局部修复信号首先要经过 stronger sham control 的审计，才能决定它是“可升级的方法证据”，还是“只能被保留为 risk marker 的负案例信号”。**

这条模型比旧版本更难听，但更接近当前数据，也更值得继续带入写作阶段。
