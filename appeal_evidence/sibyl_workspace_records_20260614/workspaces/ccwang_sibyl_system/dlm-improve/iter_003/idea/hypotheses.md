# Iteration-3 假设结算与后续写作约束

## 已被 pilot 支持的判断

### R1：matched-compute 审计是必要的

`CARD-84` 在 `GSM8K` 上相对 `DNB-84` 的 `net repaired samples = +7`，说明单纯把额外 NFE 对齐后，确实能改变我们对 revision gain 的读法。

### R2：artifact closure 本身已经成为贡献的一部分

当前四个实验臂已经可以 sample-level join，且 runtime drift、重复记录与 task-id 对齐问题已修复，因此“current-only 可复核 evidence bundle”是成立的。

### R3：本轮最强素材是审计协议成功阻止 narrative overreach

主结果不再是“方法更强”，而是“更严格的协议成功阻止了一个容易被误写成正向故事的局部增益”。

## 已被 pilot 否掉的判断

### F1：entropy-targeted revision 已被证明稳健优于随机 revision

不成立。`CARD-84` 相对 `RAND-84` 在 `GSM8K` 上只有 `+1` 的 `net repaired samples`，未通过更严格 sham-control gate。

### F2：当前 iteration 可以继续写正向 controller claim

不成立。即使 reasoning slice 上仍有信号，也不能越过负对照结果把它写成主论文中心结论。

### F3：trajectory addon 是当前主结论的必要补丁

不成立。本轮 trajectory 只会增加解释层花样，不能弥补 core sham-control 未分离的问题。

## 后续写作必须遵守的约束

### C1：允许保留的最强 wording

- 可以写：`CARD-84` 在 compute-matched baseline 上显示出局部 reasoning 信号
- 必须同时写：该信号没有在更严格 sham control 下清晰分离，因此不能作为稳健的正向 controller claim

### C2：不允许的 wording

- 不允许写“entropy-guided revision reliably improves DLM reasoning”
- 不允许写“observer-guided controller is validated”
- 不允许把 `CARD-84` 描述成当前论文的胜利方法

### C3：当前真正的 paper object

围绕 claim hygiene、runtime fairness、matched compute、sham control、per-sample audit、failure taxonomy 组织 skeptical audit 论文，而不是继续扩展 controller family。
