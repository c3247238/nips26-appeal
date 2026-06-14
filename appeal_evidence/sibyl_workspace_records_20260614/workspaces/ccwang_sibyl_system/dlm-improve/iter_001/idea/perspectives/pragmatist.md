# 实用主义者视角：收缩 scope，优先把论文做成

## 我的判断

现在最重要的不是“还能不能再救一下 TIGER”，而是让这轮工作变成一条能稳定产出的论文路线。

从工程和时间预算看，`cand_diag` 明显优于继续方法赌博。

## 为什么主线应该换成 `cand_diag`

当前证据已经足够说明三件事：

1. `TIGER` 没有超过 `entropy revision`
2. `CORE-proxy` 仍然是更强的对手
3. `gating` 只能部分修 code，且 reasoning 还掉点

这意味着如果继续押方法线，我们接下来会同时承担更多实现复杂度、更多 benchmark 扩展成本、以及更弱的论文可讲述性。

而 `cand_diag` 的优势恰恰相反：

- 现有实验已经覆盖了主结论的大部分骨架
- 只需要少量补充实验就能闭环
- 论文叙事更诚实，也更不怕 reviewer 追问

## 我建议的最小可交付版本

### 主表只保留真正必要的方法

- `Standard-64`
- `DNB-84`
- `Prophet-64`
- `CORE-proxy-64`
- `Entropy-Revise-64+3`
- `TIGER-64+3`

### benchmark 只保留三类

- `GSM8K`：主 reasoning benchmark
- `MATH500`：只在主叙事需要时扩展
- `HumanEval`：边界 / appendix，不进 headline

### 指标只保留 reviewer 真会看的一组

- accuracy / pass@1
- actual NFE
- wall-clock
- tokens/sec
- batch size / compile / attention backend

## 我对下一阶段 planning 的明确建议

1. 不再增加新方法名词
2. 不再引入新的 calibration variant
3. 不再把 code 当成“也许还能救回来”的第二主线
4. 所有实验都围绕“验证论文主句是否成立”来排优先级

## 结论

如果目标是“持续运行不要停下来”，那最强动作不是继续赌 `TIGER`，而是把当前证据迅速收束成能进入 planning 的 `cand_diag` 主线。
