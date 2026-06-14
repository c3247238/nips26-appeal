# Codex 独立审查：idea_debate

## 总评

这次重开 `idea_debate` 之后的 proposal 明显比上一版更稳，也更像是从证据里长出来的，而不是从旧方法里硬拧出来的。

我支持 `cand_diag` 成为新的 front-runner。

## 我认可的点

1. 不再试图把 `TIGER = 0.39` 与 `Entropy = 0.39` 解释成“其实还是方法成功”
2. 终于承认 `CORE-proxy = 0.46` 这一事实足以压垮 TIGER 的 method-forward 叙事
3. 把 code 结果重新定义为结构边界，而不是强行包装成泛化成功
4. 候选池重开得比较健康：
   - `cand_diag` 前台
   - `cand_minimal`、`cand_factorization` 作为真正有区分度的备胎

## 主要风险

1. **诊断论文仍可能写成“结果汇编”**
   如果没有 benefit buckets、phase diagram、observer vs controller 这种机制层组织，读者会觉得只是换了个说法。

2. **备胎候选可能重新分散注意力**
   `cand_minimal` 很有吸引力，但不能让 planning 又一次变成三线并行。

3. **若没有第二个清晰切片，front-runner 仍可能显得过窄**
   当前 GSM8K + HumanEval 已经能支撑故事骨架，但最好再补一个便宜的 reasoning slice，而不是直接全量扩张。

## 建议的 guardrails

1. planning 只允许一个主线实验包：`cand_diag`
2. `cand_minimal` 只保留一个 100-sample cheap slot
3. `cand_factorization` 只做 audit，不做 sampler 重实现
4. 所有图和表都围绕一个核心问题组织：
   - revision 的 gain / harm boundary 到底长什么样

## 审查结论

我支持把当前 proposal 送入下一阶段 planning。

原因不是“它终于找到了完美方法”，而是：

**它终于把失败条件本身当成研究对象，而这正是这轮数据最值得发表的部分。**
