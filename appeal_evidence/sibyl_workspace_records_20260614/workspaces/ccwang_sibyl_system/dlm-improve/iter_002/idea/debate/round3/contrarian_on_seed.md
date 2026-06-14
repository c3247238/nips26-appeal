# Contrarian On Seed Round 3

## 总判断

这版 `proposal_seed_round2.md` 已经比最初健康得多，至少没有再滑回 generic method paper，也明确把 `Observer-Controller Split` 从独立执行候选里降了下来。这是正确方向。

但从 contrarian 视角看，它仍然有一个核心风险：

- 结构上已经压缩到 2 个 serious execution candidates；
- 语言上却还保留着 3 条都想“像 contribution” 的倾向。

如果不继续收口，reviewer 很容易读成：

- 你们既想讲一个协议贡献；
- 又想讲一个 bucket 机制贡献；
- 还想讲一个 observer/controller 的高层命题；
- 结果三个都讲一点，但每个都还不够硬。

所以 round3 的目标不是再加内容，而是**决定谁是真 contribution，谁只是 framing**。

## 1) proposal 里 reviewer 最可能攻击哪里

我认为 reviewer 最可能攻击的是这句主命题：

> 在当前 tested policies 与 current evidence 下，`observer quality != controller gain`；因此 DLM revision 不应只按 aggregate gain 报告，而应同时报告 observer/controller split、bucket-level outcomes 与 realized compute fairness。

问题不在前半句，而在后半句的“因此”。

reviewer 很可能会质疑：

- 你们观察到某些 tested policies 下 observer/controller 不等价，为什么就能推出一套更广的 reporting requirement？
- 这个 reporting requirement 是一般协议原则，还是你们这个项目里的 scoped lesson？
- 如果 `benefit_bucket_audit` 和 `runtime_fairness_matrix` 还未完全落地，是否又在用一个半成型的高层命题去统领全篇？

更直接地说，这里最危险的地方是：

- **从局部 falsification 过快上升到一般 protocol statement**。

这很容易被 reviewer 说成：

- “你们发现了一个合理现象，但把它写成了过大的 meta-claim。”

其次，reviewer 还会攻击主证据层里对 `reasoning vs code boundary` 的提法。  
哪怕 seed 已经写了“不允许跨任务 regime law”，只要正文或摘要里仍有“recoverability 在不同任务上如何如何”的语气，reviewer 就会追问：

- 这些 boundary 是真规律，还是只是有限切片？
- HumanEval / MATH500 这类 slice 的证据够不够支撑你们的 framing？

所以 seed 当前最脆弱的点有两个：

1. `observer/controller split` 从局部现象升格过快
2. `bucket + boundary` 很容易不小心被写成泛化规律

## 2) 哪个 contribution 应降级

我认为必须继续降级的 contribution 是 **`Observer-Controller Split`**。

不是 DROP，而是进一步明确：

- 它不是 contribution bullet；
- 它不是 abstract headline claim；
- 它不是一个与 B/C 并列的 deliverable。

它最稳的位置应该是：

- introduction 里的研究问题定义；
- discussion 里的 falsification-style takeaway；
- protocol note 里的解释性桥梁。

如果把它继续放进 contribution bullet，reviewer 非常容易说：

- 这到底是一个被验证的协议贡献，还是一个还没完全被定义清楚的 framing？
- 你们是不是在用概念包装来补证据厚度？

反过来说，`Benefit-Bucket / Recoverability Analysis` 和 `Runtime-Lineage / Honest-Compute Protocol` 至少都有清晰 artifact 对象，可以被 reviewer 检查、复算、审计。  
`Observer-Controller Split` 暂时更像这些 artifact 共同支持的 **interpretive lens**，而不是独立 deliverable。

## 3) 是否保留 2 条还是 3 条 abstract claims

**我建议 abstract 只保留 2 条 headline claims，不要 3 条。**

原因很简单：  
这篇 paper 最危险的不是“贡献太少”，而是“贡献点太多导致每个都显得薄”。

如果 abstract 写 3 条，很容易变成：

1. 我们发现 observer/controller 不等价
2. 我们做了 bucket recoverability analysis
3. 我们提出 honest-compute / runtime-lineage protocol

这种三连写法会让 reviewer 产生两个反应：

- 这是不是三篇小 paper 拼在一起？
- 其中哪一条才是最主要、最实证扎实的？

更稳的写法应该是两条：

### Claim 1

`Runtime-Lineage / Honest-Compute Protocol`

定位：

- 这定义了如何避免 implementation confound；
- 它是整篇 paper 的 protocol backbone。

### Claim 2

`Benefit-Bucket / Recoverability Analysis`

定位：

- 这提供样本级证据；
- 它解释 aggregate revision gain 的来源与代价。

至于 `Observer-Controller Split`，我建议不要以 claim 形式出现，而是放在 abstract 的 framing sentence 里，例如：

- motivated by the observation that stronger observers do not automatically yield stronger controller gains...

这样它能保留辨识度，但不占用一个独立 claim slot。

## 4) 最终推荐结构

我建议最终 proposal / abstract / intro 的结构按下面收束。

### 第一层：Paper identity

先定义 paper 是什么：

- 不是 generic DLM improvement
- 不是新 controller paper
- 而是 `compute-normalized diagnostic / protocol study`

### 第二层：Framing sentence

用一句低承诺的话引出问题：

- 在当前 tested policies 下，我们观察到 stronger observer quality 并不自动转化为 stronger controller gain。

注意：

- 这里只说 observed mismatch
- 不说 framework
- 不说 general law
- 不说理论统一

### 第三层：Contribution 1

`Runtime-Lineage / Honest-Compute Protocol`

应写成：

- claim-to-asset auditable mapping
- runtime fairness matrix
- observer/controller reporting discipline

作用：

- 这是 protocol backbone
- 也是 reviewer-facing credibility shield

### 第四层：Contribution 2

`Benefit-Bucket / Recoverability Analysis`

应写成：

- fixed / harmed / no-effect sample-level decomposition
- representative examples
- boundary-sensitive evidence

注意：

- 只写 evidence / audit
- 不写 full failure taxonomy
- 不写 cross-task law

### 第五层：Planning / appendix-level support

`seed_sensitivity_spotcheck`

我建议：

- 写进 proposal deliverables，但不要升级成 abstract claim；
- 它是质量门封口件，不是 headline contribution。

## 必须进一步降调的词句

下面这些说法我建议在 seed 的后续版本中尽量删掉或替换：

### 对 Observer-Controller Split

不要写：

- “主命题”
- “核心贡献”
- “统一解释”
- “framework”

建议改成：

- “motivation”
- “framing observation”
- “falsification-style finding”
- “scoped diagnostic observation”

### 对 Bucket

不要写：

- “机制解释”
- “recoverability theory”
- “failure taxonomy”

建议改成：

- “样本级拆解”
- “failure audit”
- “recoverability evidence”

### 对 Protocol

不要写得太弱：

- “补充 appendix”
- “更诚实地报告”

建议保留锋利表达：

- “runtime-lineage discipline”
- “realized compute fairness protocol”
- “auditable mapping”

## 一句话终裁

这版 seed 最容易被 reviewer 攻击的，是把 `observer/controller split` 从局部观察写成过大的 protocol 主命题；我建议继续把它降级为 framing，只保留 **2 条 abstract claims**：`Runtime-Lineage / Honest-Compute Protocol` 与 `Benefit-Bucket / Recoverability Analysis`，最终结构应是 **A 负责引题，C 做协议护城河，B 做主证据层**。
