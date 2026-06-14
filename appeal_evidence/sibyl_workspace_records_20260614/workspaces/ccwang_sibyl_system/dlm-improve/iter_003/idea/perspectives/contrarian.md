# Contrarian 视角：这不是“被审计后幸存的论文对象”，更像“被审计后勉强止损的失败记录”

我不认同当前 proposal 已经找到了一个足够稳的可发表对象。它确实比继续硬写 `CARD-84` 正向方法更诚实，但这不等于它已经变成了一篇强的 skeptical audit 论文。现在最危险的自欺是：把“我们成功阻止自己过度宣称”包装成“这本身就是主要 scientific contribution”。

## 最可能被 reviewer 打穿的点

- “负结果”不自动等于“协议贡献”。
- 证据范围太窄，支撑不了“DLM claim hygiene / protocol”这种大标题。
- 核心结果仍然过于“无事发生”，容易让 reviewer 觉得 paper 没有正信息密度。
- proposal 仍在偷偷借用“差点成功”的势能。
- `runtime fairness` 现在更像风险暴露点，而不是护城河。

## 如果必须保留这条线，最低限度怎么写

最保守、最诚实的写法不是“protocol-first skeptical audit paper”，而是“一个经过严格负对照后被否定的 entropy-guided revision case study，以及一套 supporting audit bundle”。主语必须从“我们贡献了一套 reviewer-friendly protocol”收缩成“我们报告一个在 stricter sham control 下失败的 selective revision 案例，并公开其复核资产”。

## 对 proposal 的三条改写建议

1. 把“protocol paper”改成“negative case study + reproducibility bundle”。
2. 停止反复暗示 `CARD-84` 还保留某种半成立的正向地位。
3. 把“为什么这值得发”从自我卫生转成外部判别价值：强调在小 gain regime 下，`compute-matched baseline` 不足以排除伪改进，`sham control` 会实质改变结论。

## 最终态度（支持/保留/反对 + 1-10分）

反对，3/10。不是因为 pivot 错了，而是因为当前资产最多支持一篇非常克制的负案例报告，不支持一篇标题偏大的 DLM protocol paper。只有把 claim ceiling 再砍一刀，这条线才值得保留。
