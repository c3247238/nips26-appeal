# Contrarian On Pool Round 2

## 总判断

round1 之后的 candidate pool 已经比最初健康很多，至少明确砍掉了 `Minimal Controller`、`TIGER hero`、`Calibration-Aware` 和 generic new scheduler。这一步非常关键，因为 reviewer 最容易一票否决的，正是那类“明明证据基础还是 diagnostic，结果又硬拐回方法论文”的候选。

但即使只剩下 A/B/C 三个候选，它们的风险结构也并不对称。  
从 contrarian 视角看，当前真正需要回答的不是“哪一个听起来最高级”，而是：

- 哪个最容易因为 claim 过满被 reviewer 打成 evidence-thin；
- 哪个最能帮助整篇 paper 避开 narrative selection / implementation confound；
- 哪个应该是主线，哪个应该降成支撑层。

## 1) 哪个候选最容易被 reviewer 打穿

我认为最容易被打穿的是 **Candidate A: Observer-Controller Split**。

不是因为它没价值，而是因为它最容易被写成一种“概念上很聪明、但证据仍不够厚”的对象。reviewer 最可能的攻击路径会是：

- 你们是不是只是把常识性直觉包装成新术语？
- `observer quality != controller gain` 这个说法到底是一般规律，还是只在你们当前某几个 pairwise comparison 上成立？
- `d(s)` 和 `g(s)` 的定义有没有 appendix-grade 清晰度？
- 如果没有 bucket audit、runtime fairness matrix、claim-to-asset 映射，那这是不是一个漂亮但无法严肃审计的 framing？

换句话说，A 最大的问题不是方向错误，而是**最容易 outrun evidence**。  
一旦写成下面这种口吻，就很危险：

- “我们提出了一个新的 observer-controller decomposition framework”
- “我们从理论上证明 diagnostic signal 与 controller gain 的关系必须如何如何”
- “我们建立了一个统一解释 DLM inference 的框架”

这些话 reviewer 会非常敏感，因为当前资产更像支持一个 **falsification-style finding**，而不是一个 fully-formed theory framework。

## 2) 哪个候选最值得保留

我认为最值得保留的是 **Candidate C: Runtime-Lineage / Honest-Compute Protocol**。

原因很简单：它最不依赖叙事美感，最直接面向 reviewer 的真实攻击面。

当前整篇 paper 最大的外部风险，不是“idea 不新”，而是 reviewer 会怀疑：

- 你们是不是把方法差异和实现差异混在一起了？
- 你们是不是 nominally compute-matched，但 realized runtime 并不公平？
- 你们是不是靠不同 batch path、backend、compileability、proxy path 支撑了结论？

Candidate C 正好是唯一一个正面回答这些问题的候选。它的价值不只是 hygiene，而是：

- 它让你们的其它 claim 不那么容易被一记 implementation confound 直接推翻；
- 它和 literature 里的 `FlashDLM`、`LSP`、`Scaling Beyond Masked Diffusion Language Models` 是方法论同频的；
- 它不需要你们发明新 controller，也不需要过度承诺机制解释。

我会更明确地说：

- 如果没有 C，A 和 B 都可能被 reviewer 说成“建立在不透明比较条件上的结论”；
- 有了 C，A 和 B 才有机会被当成严肃 diagnostic evidence，而不是 selective storytelling。

## 3) 是否必须从 3 个压到 2 个

**我认为必须压到 2 个 headline candidate。**

原因不是 3 个完全放不下，而是 3 个并列时，最容易产生一种危险观感：

- 你们既想有理论对象；
- 又想有 bucket 机制分析；
- 还想把 protocol/fairness 单独算一个贡献；
- 结果每个都讲一点，但没有一个讲透。

从 reviewer 视角看，这会像“概念很多，实证抓手分散”。  
当前更稳的做法应该是：

### headline 1：Candidate C `Runtime-Lineage / Honest-Compute Protocol`

- 负责定义比较纪律与 reviewer-facing credibility
- 这是整篇 paper 的 protocol backbone

### headline 2：Candidate B `Benefit-Bucket / Recoverability Analysis`

- 负责提供最关键的样本级机制证据
- 这是整篇 paper 的 main evidence layer

### Candidate A 的位置

我不建议把 A 继续保留为第三个并列 headline candidate。  
更稳的处理方式是：

- 把 A 降成 **framing / problem statement / discussion-level falsification claim**
- 即：它解释为什么要做 B 和 C
- 但它不单独占一个 headline contribution slot

也就是说，不是彻底 DROP A，而是**取消它作为并列 serious candidate 的地位**。

## 4) 哪些词句 / claim 必须降调

下面这些词句如果继续出现在摘要级、标题级或 contribution bullet 里，我认为风险很高，必须降调。

### 对 Candidate A 必须降调的表述

必须避免：

- “提出新的 observer-controller split framework”
- “统一解释 DLM inference 的理论框架”
- “证明了 observer quality 与 controller gain 的系统性关系”
- “建立了 diagnostic signal 到 intervention utility 的一般理论”

建议改成：

- “检验一个常见隐含假设：较强 observer 不自动带来较强 controller gain”
- “在当前 DLM revision setting 下观察到 observer/controller decoupling”
- “将 observer quality 与 realized control gain 分开报告”

### 对 Candidate B 必须降调的表述

必须避免：

- “完整 failure taxonomy”
- “recoverability mechanism 已被解释清楚”
- “跨任务稳定的 revision mechanism”
- “reasoning vs code 的普适规律”

建议改成：

- “bucket-oriented failure analysis”
- “fixed / harmed / no-effect 的样本级拆解”
- “boundary-sensitive evidence”
- “在当前任务切片中的 recoverability audit”

### 对 Candidate C 必须降调但保留锋利度的表述

必须避免把它写得太像纯 hygiene：

- “我们更诚实地报告 runtime”
- “我们补充了更完整的 appendix”
- “我们提供了一些实现细节”

这会把最值得保留的候选写弱。

建议改成：

- “realized compute fairness protocol”
- “runtime-lineage discipline”
- “claim-to-asset auditable mapping”
- “避免将方法效应与实现效应混淆的比较协议”

也就是说，C 要避免两个极端：

- 不能写成抽象大理论；
- 也不能写成普通 hygiene 补件。

## 我建议的最终收束

如果 round2 之后必须给出一个更硬的排序，我的建议是：

1. **保留 Candidate C 作为主线 backbone**
2. **保留 Candidate B 作为主证据层**
3. **将 Candidate A 降为 framing / falsification-style support，不再作为并列 headline**

这个组合的好处是：

- 最不容易被打成 narrative selection；
- 最不容易被 implementation confound 一击致命；
- 同时又不会完全失去论文的辨识度，因为 A 仍然可以作为 framing 出现在引言与讨论中。

## 一句话结论

最容易被 reviewer 打穿的是 **A: Observer-Controller Split**，因为它最容易概念跑在证据前面；最值得保留的是 **C: Runtime-Lineage / Honest-Compute Protocol**，因为它最直接挡 reviewer 的 confound 攻击；我建议 **必须从 3 个压到 2 个 headline candidate**，保留 **C+B**，把 **A 降成低承诺的 framing / falsification claim**。
