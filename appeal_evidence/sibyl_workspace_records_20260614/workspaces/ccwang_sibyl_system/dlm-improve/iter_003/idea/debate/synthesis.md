# Idea Debate Synthesis

## 共识

- 当前主线必须锁定为 `cand_negative_audit_pivot`。
- `CARD-84` 相对 `DNB-84` 在 GSM8K 上有局部信号，但相对 `RAND-84` 只剩 `+1 net repaired`，未通过更严格 sham-control gate。
- `artifacts_joinable = true` 使 skeptical read 成为干净、可复核的结论，而不是实验烂尾后的被动降级。
- 所有视角都反对把 observer signal 直接翻译成 controller efficacy。
- 论文价值如果存在，必须来自“更强负对照改变了解释”，而不是“我们诚实承认自己没赢”。

## 主要分歧

- 分歧不在于是否 pivot，而在于 paper object 应该写多大。
- 最保守的一派只接受 `negative case study + reproducibility bundle`。
- 中间派接受 `minimal audit template`，但坚决反对 protocol invention。
- 稍积极的一派希望保留 `near-miss claim audit` 的识别性新意，但也同意不能写成 almost-win。

## 锁定对象

一篇关于 training-free DLM small-gain claims 的 **audited negative case study**：证明在 matched-compute、sham control 与 sample-level artifact closure 下，一个看似成立的 revision gain 会失去正向方法解释资格。

## 必须删除的表述

- entropy-guided revision reliably improves DLM reasoning
- observer-guided controller is validated
- CARD-84 is the winning inference method
- protocol-first contribution
- new paradigm / new interface
- almost works / near-win

## proposal 必须新增的句子

- 在 small-gain regime 下，`compute-matched baseline` 只能排除额外算力解释，不能排除 generic targeting 或 perturbation 带来的伪改进，因此 `sham control` 会实质改变结论。
- 本文将 entropy 仅视为 `risk marker`，而不把它视为已验证的 intervention rule。
- 我们报告的核心对象不是一个成功的方法，而是一个经过更严格负对照后被拦下来的 near-miss case，以及支撑这一判别的最小审计证据结构。

## 综合结论

需要**重写** proposal。不是改方向，而是把当前相互拉扯的 `skeptical audit`、`protocol paper`、`near-miss object` 收束成单一对象：低 claim ceiling、强证据结构、弱方法野心的 falsification case study。
