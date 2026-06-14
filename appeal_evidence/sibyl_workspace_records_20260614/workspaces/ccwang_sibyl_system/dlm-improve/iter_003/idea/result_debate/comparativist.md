# Comparativist View

## 对外定位结论

如果把这轮结果放到更广的 training-free DLM revision / calibration / control 文献里看，它最合理的位置不是“another stronger decoding method”，而是一个 **negative-case audit contribution**。它对外部工作的价值，不在于刷新 SOTA，而在于提供一个更强对照条件下的解释纠偏案例。

## 1. 真实贡献边界

`proposal.md` 已经把 contribution ceiling 收得很清楚：当前 paper object 是 **audited negative case study**，不是方法论文，也不是 protocol manifesto。这个定位和当前证据是匹配的。

与常见的正向 small-gain 论文相比，我们这轮资产多了一层 reviewer 很在意、但同类工作常常没有完全补齐的东西：

1. matched-compute baseline
2. budget-matched sham control
3. sample-level audit
4. current-only artifact closure

因此，这篇稿子的真实贡献更像是：

- 不是证明某个 revision signal 赢了
- 而是证明 **在更强负对照下，一个原本可被写成局部成功的信号应被降格解释**

## 2. 为什么它更像 negative-case / audit contribution，而不是 method contribution

从现有表格看：

- `CARD-84` vs `DNB-84` on GSM8K: `net_repaired = +7`
- `CARD-84` vs `RAND-84` on GSM8K: `net_repaired = +1`
- `CARD-84` vs `RAND-84` on MBPP: `0`

这组数字足以说明：

1. 有局部信号
2. 但 signal 不能脱离 sham control 独立成立

这恰好是 negative-case audit 最有价值的情况：不是“什么都没有”，而是“如果没有更强对照，你几乎就会把它写成正向增益”。这比彻底失败的 null result 更有论文张力。

## 3. reviewer 最可能如何定位它

如果稿子写得克制，reviewer 更可能把它看成以下三类之一：

1. 一篇关于 training-free DLM revision 的 skeptical case study
2. 一篇 small-gain interpretation correction / evaluation hygiene paper
3. 一篇以 audit template 为 supporting asset 的 negative result paper

如果稿子写得不克制，reviewer 会立即把它打回成以下负面定位：

1. 作者的方法没有赢，所以把故事改写成 protocol
2. `CARD-84` 没和 `RAND-84` 拉开，还在暗示接近方法成功
3. 审计模板被夸大成新范式，但证据其实只有 audited slice

## 4. 对外叙事最该避免的误区

### 4.1 避免和 method SOTA 做正面竞争口径

当前结果根本不需要也不适合去争“谁是更强 inference method”。这条路只会把稿子拖回不利战场。对外最稳的姿态是：

- we are not claiming a better controller
- we are showing why a stronger sham control changes the interpretation

### 4.2 避免把 minimal audit template 写成 community standard

现在更合适的语言是：

- minimal auditable template
- current evidence-supported template

而不是：

- new protocol paradigm
- benchmark standard-setter

## 5. 我给这篇稿子的比较分析定位

一句话说，这篇稿子最好的位置是：

**它不是靠更高分数进入文献，而是靠更强负对照下的解释纠偏进入文献。**

如果沿这个方向写，贡献边界会更真实，也更容易在 related work 中自洽地说明“为什么这不是 another method paper”。这正是当前 iteration 3 最值得保住的地方。
