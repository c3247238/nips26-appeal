# 选题与构思批评

## 总评

iteration 4 的 ideation 其实做对了一件重要的事：在 `MGCD/DSG` 负结果后没有继续救旧方法，而是重建了 candidate program。这一点科学上是加分项。问题在于，新的 candidate ladder 到 manuscript 之间没有完整映射，导致构思层的严谨性没有完全传递到最终 paper。

## 主要问题

### 1. proposal 的真正主问题与最终 paper 的主问题不再一致

proposal 把本轮 serious questions 收紧为两个：

- repair object 是否选错了？
- entropy 是否更适合做 routing/stopping？

其中更核心、更带 pivot 含义的其实是前者，也就是 `cand_bsr` 对 object-level 假设的检验。`cand_espd` 更像并行 speed line。

但最终 paper 只写成了第二个问题的 full-scale study。这样会出现一个结构性后果：

- 研究 program 很有野心，像在重立 training-free DLM 的 intervention object；
- 最终 paper 却落在较窄的一条 routing 解释线上。

这本身没有错，但如果不解释，就会让读者以为一开始就只打算做 ESPD。

### 2. candidate ladder 的“为什么是这三个”没有进入论文语境

proposal 中 `cand_bsr / cand_espd / cand_ugr` 的分工很清楚：

- `cand_bsr`：quality line
- `cand_espd`：speed line
- `cand_ugr`：conditional backup

这个 ladder 其实很适合成为 paper 的 research program figure 或 appendix note，因为它体现了研究不是随机试点子，而是围绕失败机制重排候选空间。

现在 paper 几乎没有用这个资产，导致 ideation 的优势被浪费了。

### 3. 当前 novelty 真正站得住的部分比 proposal 更窄

proposal 的潜在新意有两层：

- object-level 重立题
- entropy routing reinterpretation

当前真正被 full artifact 支撑的只有第二层。也就是说，paper 如果继续借 proposal 的 broader ambition 来抬 novelty，会不稳。

更诚实的 novel idea 表述应该是：

- 我们不是解决了 training-free dLLM revision 的主问题。
- 我们是在一个 small-gain regime 里证明：entropy 更像 routing/stopping signal，且这个解释通过 stronger sham control 仍站得住。

### 4. alternatives 的价值还没被用起来

`idea/alternatives.md` 里其实已经为 reviewer 预埋了几条退路：

- DCD-lite / Late-Commit Revision
- Cache-First Speed Lane
- AR-Guided Diffusion Lite

这些替代项的存在可以帮助 paper 回答一个很重要的问题：

- “为什么你们没有继续在别的方向上扩展？”

因为 alternatives 文档已经给了研究管理上的理由：不是想不到，而是为了保持归因清晰主动不升格。这一点若写进 appendix，会增强整套 ideation 的可信度。

## ideation 层面的核心批评

当前最危险的地方不是 idea 不够好，而是：

- proposal 的大构思 > screening 的中间状态 > paper 的最终落点

这三层之间缺一张显式映射表。

如果没有这张映射表，reviewer 会把正常的 research narrowing 误读成 opportunistic narrowing。

## 建议

1. 在 paper appendix 增加一个 candidate-program timeline，把 `MGCD/DSG -> BSR/ESPD/UGR -> screening -> ESPD full-scale` 画出来。
2. 明确写出：当前 paper 只 claim candidate ladder 中的 `cand_espd` 这一支，而不是整个 iteration 4 research program 已完成。
3. 若要保住 broader ideation 的价值，把 `cand_bsr` 留作 next-step discussion，而不是完全淡出。
