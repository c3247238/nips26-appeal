# Pragmatist 立场：把这轮结果收缩成“可证伪的负对照审计案例”

从务实角度看，`cand_negative_audit_pivot` 仍然是当前最可能写成边界清楚、审稿人能接受的对象，但前提是继续收缩，而不是拔高。现有证据足以支持“一个看似成立的 training-free DLM revision gain，在 matched compute、sham control 和 sample-level audit 下没有站稳”，却不足以支持“我们提出了一个新的 protocol 范式”或“我们对 DLM revision 有一般性发现”。最稳的方向，是一篇小而硬的 falsification-oriented audit paper。

## 最值得保留的论文对象

最值得保留的不是 `CARD-84` 的局部有效性，而是“在 DLM 这种容易被 test-time 小增益诱导出正向故事的 setting 里，负对照和 compute-matched 审计会改变结论”。论文对象应当是：围绕 training-free DLM revision claim 的审计型负结果论文，提出一个最小但必要的审计闭环，并用 `CARD-84 vs DNB-84 vs RAND-84` 这个案例证明，没有 sham control 的局部 gain 很容易被误判为方法胜利。

## 最大的执行风险

- 稿件会被审稿人看成“我们自己的方法没赢，所以转写成经验教训”。
- proposal 继续把自己写成 `protocol-first contribution`、`claim hygiene framework` 这类大词，容易显得贡献虚大于实。
- DLM-specific 新意如果不钉死在“iterative denoising 下的小增益特别容易被 compute 与 random targeting 混淆”，审稿人会觉得这只是任何 ML paper 都该做的常规 control。

## 对 proposal 的三条改写建议

1. 把论文对象从“skeptical audit protocol”改成“falsification case study with a minimal audit template”。
2. 把 `CARD-84` 明确降格为“被审计案例”，不要再保留任何“也许它还有局部方法价值”的主文暧昧空间。
3. 把 DLM-specific 新意钉死在“iterative denoising 下的小增益特别容易被 compute 与 random targeting 混淆”这一点上。

## 最终态度（支持/保留/反对 + 1-10分）

保留，7/10。我同意继续走 `cand_negative_audit_pivot`，但必须再降一档野心：不要写成“新的论文范式”，而要写成“我们用一个 DLM revision 负案例证明，严格负对照会改变结论，并给出一个最小可复用的审计模板”。
