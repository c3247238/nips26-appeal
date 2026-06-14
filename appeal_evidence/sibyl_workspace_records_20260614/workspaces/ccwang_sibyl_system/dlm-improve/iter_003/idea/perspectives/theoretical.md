# Theoretical 视角：它可以成立，但只能作为“可证伪边界的协议论文”

我的独立判断是：`cand_negative_audit_pivot` 恰好是当前证据下唯一概念上不自相矛盾的对象。理论上仍然站得住的，不是 `CARD-84` 作为 controller 的有效性，而是一个更弱但更干净的命题：在 training-free DLM revision 里，observer、controller、runtime 这三层如果不被显式拆开并放进 sham-controlled、matched-compute 的证伪协议里，小幅 gain 没有资格被解释为方法进步。

## 仍然成立的核心命题

- `CARD-84 > DNB-84` 说明“定向 revision 可能含有局部可用信号”这一弱命题还活着。
- `CARD-84 ≈ RAND-84` 同时说明这种局部信号还不足以上升为“observer-guided controller 已成立”。
- artifact closure、sample-level join、runtime drift repair、claim-to-asset map 落盘，定义了一个可复核的 falsification interface。
- observer / controller / runtime 的显式分层本身有方法论价值。

## 概念上最危险的越界

- 把 `CARD-84` 写成“被验证的 reference controller”。
- 把 observer 质量直接翻译成 controller 有效性。
- 把 skeptical audit 说成 identifiability、near-oracle observer 或 theorem-like 框架。
- 把 reasoning slice 上的局部信号外推成 DLM revision 的一般规律。

## 对 proposal 的三条改写建议

1. 把主命题从“skeptical audit of a failed gain”改写成“minimum falsification protocol for training-free DLM revision claims”。
2. 把 `CARD-84` 的角色进一步降格，统一表述为“audited reference controller”。
3. 在 proposal 里显式加入一段 `claim ladder`：observer carries localized risk information；controller beats compute-matched baseline；controller beats sham control。当前证据不能越过第三层门槛。

## 最终态度（支持/保留/反对 + 1-10分）

保留，7/10。我支持继续沿 `cand_negative_audit_pivot` 写下去，但只支持它作为一篇“协议与主张边界”的论文，不支持任何伪装成弱理论论文或弱方法论文的版本。
