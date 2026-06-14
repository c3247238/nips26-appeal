# 创新者视角：把“方法失败”转化为更强的问题设定

## 我的判断

这轮最有创新性的方向，不是继续给 `TIGER` 叠更复杂的启发式，而是把论文问题改写成：

**在 DLM 中，revision 到底什么时候有用，什么时候只是昂贵幻觉？**

这个问题比“再提一个 selective revision sampler”更真实，也更容易形成领域级贡献。

## 我支持的主候选

### 候选 1：`cand_diag` 作为主线

把当前工作组织成一篇 **compute-normalized diagnostic / benchmark paper**：

- 比较 `Standard-64 / DNB-84 / Prophet-64 / CORE-proxy-64 / Entropy-Revise-64+3 / TIGER-64+3`
- 严格报告 `accuracy / actual NFE / wall-clock / tokens-per-second / batch regime`
- 把 `HumanEval` 放到 boundary appendix，专门回答 “为什么 code 对局部 revision 更脆弱”
- 把 calibration 从“改进方法”降级为“解释机制”

这个 framing 的创新点在于：

1. 它把 DLM 领域里常见的“名义步数公平”替换成“真实 compute 公平”
2. 它把 revision 的收益与失败都纳入同一个解释框架
3. 它把 code failure 不再视为脏点，而视为结构约束失配的关键证据

## 我保留的副候选

### 候选 2：`syntax-guard appendix`

不把 gating 当主方法，而是把它作为“最小干预边界修复”附录：

- 证明 cheap syntax guard 能显著减少 parse failure
- 同时诚实承认它不能恢复 reasoning story，也不能恢复 Standard code baseline

## 我反对的方向

- 不应再把 `TIGER` 包装成方法胜利
- 不应把 `math500_transfer` 当成必须补跑的“缺口”
- 不应为了保住旧叙事再引入更复杂的 instability / calibration / gating 组合

如果核心 pilot 已经表明 `TIGER = entropy`，那继续加复杂度只会把论文推向不可证伪。

## 下一轮最值得被证明的新论点

我建议把下一轮提案的 headline 改成：

**When Revision Helps Diffusion LMs, When It Hurts, and Why Honest Compute Accounting Changes the Story**

并围绕以下三个新论点组织：

1. `revision benefit` 是强任务依赖的，不存在一个普适的“局部修补万能药”
2. 一旦用 `actual compute` 对齐，方法排序会明显变化
3. code 上的局部 revision failure 暗示问题不是“信号不准”，而是“结构约束不兼容”

## 结论

创新性并没有消失，只是从“再发明一个 revision 方法”转到了“提出一个更真实、更难回避的问题并给出第一套诚实答案”。这正是我支持 `cand_diag` 的原因。
