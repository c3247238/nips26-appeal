# 规划批评

## 总评

planning 本身并不差，问题在于执行后的 paper 没有把 planning 的纪律性完整继承下来。换句话说，计划层是“先筛再放大”，写作层却更像“已经收敛成单线研究故事”。这会让 planning 看起来比最后成稿更严谨。

## 主要问题

### 1. methodology 与最终 paper 的对齐度不足

methodology 明确规定：

- `cand_bsr` 是唯一 quality front-runner
- `cand_espd` 是并行 speed line
- `cand_ugr` 不进入 phase-1 调度
- full benchmark 只有在 screening signal 出现后才实例化
- 第二 benchmark 和 code boundary validation 都是重要后续要求

最终 paper 只兑现了其中一部分：

- screening signal 的确存在
- `cand_espd` 的确进入了 full-scale

但没兑现的部分没有被当成 planning debt 写出来：

- `cand_bsr` 主线没有 full follow-up
- 第二 benchmark 没做
- code boundary validation 没做

这会让 planning 看起来像只在内部成立，而不是对最终研究产物真正有约束力。

### 2. candidate selection 规则应写成显式流程

现在真正决定 full-scale 的不是 proposal，而是 screening bundle。这个逻辑在 artifact 中是存在的，在 paper 中却不是显式流程。

planning 角度最欠缺的是一张简单的 gate table：

- Gate A: MGCD/DSG fail -> return to idea_debate
- Gate B: BSR refine / ESPD advance
- Gate C: only advanced candidate enters full-scale

如果把这个 gate table 放进 appendix，整个 iteration 4 会显得非常干净。

### 3. No-Go rule 在 paper 中消失了

methodology 的一个优点是有明确的 no-go rule：

- 若 `cand_bsr_v1` 与 `cand_espd` 都无法提供对 `RAND-84` 的可信 signal，则不进入 full benchmark

这说明 planning 不是“总要找点东西写成正面结果”，而是真有停机条件。现在 paper 没把这种 planning discipline 用出来，损失了很重要的可信度来源。

### 4. runtime planning 的长处没转成论文长处

planning 对 runtime contract 其实写得很细：

- batch probe
- flash / fallback
- compile parity
- auxiliary ledger
- harness alignment

但 paper 里这些更像背景说明，而不是“为什么这个结果比一般 small-gain paper 更可信”的核心资产。

## planning 层面的核心批评

当前最大的问题不是计划不好，而是：

- 计划的“约束性”没有被文章保留下来。

于是 paper 读起来像一个自然收敛的方法故事，实际上它更像一个经过多次 gate 和筛选后留下的 bounded result。后者其实更可信，但你现在没有把这点写出来。

## 建议

1. 增加一页 planning-to-execution trace，显示 methodology 的 gate 如何映射到实际执行。
2. 把未兑现的 planning items 变成显式 debt list，而不是让 reviewer 自己发现它们没出现。
3. 把 no-go rule 与 candidate ladder 写进 appendix，强化这轮研究不是 opportunistic search，而是受约束的 program execution。
