# 实验主义者视角：把 proposal 建在已经成立的证据链上

## 我的判断

这轮最可靠的路线只有一条：

**围绕已经完成的 pilot，做一篇协议严格、compute 诚实、结论可复现的 diagnostic / benchmark 论文。**

## 为什么我不支持继续方法扩张

从实验设计角度看，`TIGER` 主线已经遇到三个明确否决信号：

1. 与 `entropy revision` 打平
2. 明显弱于 `CORE-proxy`
3. code 边界只得到部分修复

在这种情况下继续扩方法，只会让后续结果更难解释。

## 我支持的主假设

- `H1`：真实 compute 匹配会改变方法排序
- `H2`：revision 收益具有任务依赖性
- `H3`：calibration 对诊断有用，但因果控制作用有限
- `H4`：cheap syntax guard 只能部分缓解结构性失败

## 下一阶段我要求的实验纪律

1. 所有对比都报告 `actual NFE`
2. 所有对比都附带 `wall-clock / TPS / batch regime`
3. 不再把 `nominal step count` 当成公平比较的唯一依据
4. 不再把未完成 transfer benchmark 当成“必须补洞”，除非它改变主结论

## 结论

如果我们想让下一阶段 planning 不再漂移，就必须让 idea stage 的文本只依赖已经成立的证据，而不是继续替未来可能出现的结果预支结论。`cand_diag` 正是符合这一原则的候选。
