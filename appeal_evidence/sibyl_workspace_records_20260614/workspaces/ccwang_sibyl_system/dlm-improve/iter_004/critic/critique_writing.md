# 写作批评

## 总评

这版 `paper.md` 的最大优点是诚实，最大问题是诚实之上仍残留了 narrative compression。它已经不怎么 overclaim 结果本身，但仍在压缩一段实际上相当复杂的研究转向史：从 object-level pivot，到 speed-line screening，再到 full-scale 只剩 `cand_espd`。这段历史没有被充分写出来，于是 paper 会显得比真实证据更“自然”、更线性。

## 主要问题

### 1. 主线切换解释不充分

- 当前 paper 把 `cand_espd` 写成完整论文主角。
- 但 proposal 与 methodology 仍把 `cand_bsr` 定义为 quality front-runner，且 object-level repair 才是 iteration 4 的原始 pivot 理由。
- 这会让熟悉 artifact 的读者产生一个很直接的疑问：为什么最后 mainline 不是你一开始说最重要的对象线？

需要补的不是更多修辞，而是一段明白的 evidence-first transition：

- 先说明 MGCD/DSG 的负证据如何迫使 proposal pivot。
- 再说明 screening bundle 为什么给出 `cand_bsr=REFINE`、`cand_espd=ADVANCE`。
- 最后说明为什么 full-scale 当前只对 `cand_espd` 成立，因此 manuscript 采取 speed-line-first 的 evidence ordering。

### 2. 贡献句太多，中心 punchline 被稀释

当前 paper 同时想承担四件事：

- entropy reinterpretation
- sham-control discipline
- runtime-lineage discipline
- iteration 4 研究主线更新

这些都重要，但以当前效果量，不适合在摘要和引言中并列成同等级贡献。否则 reviewer 会抓住其中最弱的一条把整篇 paper 拖下去。

更稳的结构应该是：

- 主贡献：`cand_espd` 在 full-scale GSM8K 上给出一个 survives stronger sham 的 bounded routing result。
- 支撑贡献 1：paired repair/harm 让 small-gain outcome 更可审。
- 支撑贡献 2：runtime lineage 与 claim boundary 把结果从“看起来有 gain”变成“可被 reviewer audit 的 gain”。

### 3. observer/controller 语言仍略大

paper 其实已经多次声明“我们不 claim semantic controller validated”，但 introduction 和 discussion 仍让 observer/controller split 看起来像一个更广的理论结论。就当前证据而言，这还不够稳。

更安全的写法不是：

- “we shift entropy from controller to router” 作为已完成理论重构

而是：

- “current evidence for cand_espd is more consistent with a routing/stopping role than with a semantic-controller role”

前者像定论，后者像基于当前 bundle 的解释。

### 4. speed line 的命名容易被反打

paper 现在已经承认 `cand_espd` 不是绝对最快，但这个 caveat 还不够靠前。因为 reviewer 第一反应很可能是：

- 你叫 speed line，为什么 `RAND-84` 和 `CARD-84` 比你更快？

这里需要更直接：

- `cand_espd` is not an absolute throughput winner.
- The positive result is candidate-versus-sham trade-off preservation under the retained contract.

这句话应该进入 abstract 尾句或 Table 1 附近，而不是只留在 discussion。

## 写作层面最该立刻修的三件事

1. 在 introduction 明确写出 proposal priority 与 evidence priority 的分离。
2. 收缩 contributions，保留一个主 punchline 和两个 support points。
3. 把 speed-line 的负 caveat 提前，避免 reviewer 先入为主地把 paper 理解成“新加速器”。 

## 可以保留的优点

- 局限性表述整体是诚实的。
- related work scaffold 已经比一般内部文稿成熟得多。
- conclusion 没有硬抬结果，这是正确方向。
