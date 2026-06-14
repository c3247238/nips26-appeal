# 创新者视角：把“差点成立的方法”写成“高价值的识别协议”

我不主张盲从 `cand_negative_audit_pivot` 的现有写法，但我认可 pivot 的方向。创新者视角下，这篇东西最强的版本不是“我们诚实地报告一个负结果”，而是“我们把 DLM 中最容易被误写成方法增益的小信号，升级为一个可复用的 falsification object”。论文对象应从单次 `CARD-84` 失利，抬升为一套专门针对 training-free revision / sampling 小增益的审计协议与 claim ceiling 框架，而 `CARD-84 vs DNB-84 vs RAND-84` 只是第一个被这个协议成功压测的案例。

## 最值得保留的新意

- 最值得保留的不是 `CARD-84` 本身，而是“matched compute + sham control + per-sample join + runtime contract”这一整套专门为 DLM 小增益设计的 falsification protocol。
- `CARD-84 > DNB-84` 但 `CARD-84 ≈ RAND-84` 这个结构很有价值，因为它不是平庸的全败，而是“近阳性、但被更强负对照打回去”的边界案例。
- reasoning 与 code 没有对称分离这一点也值得保留，但不能写成机制胜利；更适合写成“task-dependent claim ceiling map”的开端。

## 最大的过度叙事风险

- 最大风险是把这篇文章偷偷写回“entropy-targeted revision almost works”。
- 第二个风险是把单一案例外推成“DLM inference paper 都必须遵循的通用标准”。
- 第三个风险是把 `selective compute / compute eligibility` 偷渡成主对象。

## 对 proposal 的三条改写建议

1. 把 paper object 从 “skeptical audit of CARD-84” 改写成 “a falsification protocol for small claimed gains in training-free DLM inference”，让 `CARD-84` 明确降格为 probe，而不是主角。
2. 把贡献表述从 “协议成功阻止过度叙事” 再往前推半步，改成 “一种 near-miss claim audit：专门处理那类看起来赢过 compute baseline、却未能穿透 sham control 的结果。”
3. 把 reasoning/code 张力改写成边界发现而非机制解释：强调 stronger controls 下局部修复信号如何失效，而不是解释机制。

## 最终态度（支持/保留/反对 + 1-10分）

保留，7/10。当前主线可以成立，但前提是把它从“写得很诚实的负结果”升级成“识别近阳性伪胜利的协议论文”；如果只是按现有 proposal 直写 skeptical audit，新意还差半级。
