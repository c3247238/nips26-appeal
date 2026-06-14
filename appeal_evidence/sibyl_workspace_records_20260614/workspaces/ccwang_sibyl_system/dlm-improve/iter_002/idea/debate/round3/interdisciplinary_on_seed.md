# interdisciplinary on seed

## 总判断

从 interdisciplinary 视角看，`proposal_seed_round2.md` 已经基本走到正确终点了。  
它第一次把三件原本容易混在一起的东西分了层：

1. **主问题 framing**
2. **主证据层**
3. **协议护城河**

这正是把论文从“结果整理”推进到“诊断科学”的必要条件。  
如果继续保持这种层级纪律，而不是重新让三个候选并列争主线，那么这篇 paper 已经**足以被写成诊断科学，而不只是结果整理**。

但“足以”是有条件的：

- 前提是 `benefit_bucket_audit.json` 真正落盘；
- 前提是 runtime-lineage artifacts 真正 reviewer-friendly；
- 前提是 observer/controller split 不被写成超出证据的理论大命题。

也就是说，当前 seed 的结构已经对了，剩下主要是**不要在最后一公里重新失控**。

---

## 1) 当前结构是否足以把论文写成诊断科学而不是结果整理

我的回答是：**是，结构上已经足够；但成立与否取决于执行时是否守住层级边界。**

为什么现在它终于不像单纯结果整理：

### 因为它已经不再只是“报三个现象”

如果只是：

- GSM8K 有一个 compute reorder
- signal audit 说明 calibration/entropy 有信息
- HumanEval 说明 code boundary 很脆

那仍然只是精致的结果整理。

### 现在它开始像诊断科学

因为 seed 已经提出了一个更像成熟诊断研究的结构：

- **先界定 diagnostic question**  
  `observer quality != controller gain`
- **再给 case-level evidence**  
  `fixed / harmed / no-effect`
- **再给 measurement/protocol closure**  
  `runtime-lineage / honest-compute`

这三层一旦都落地，论文就不再只是说“我们看到了这些现象”，而是在说：

> 我们建立了一种更像诊断学的证据组织方式，用来区分 signal、intervention、outcome 和 realized protocol。

这就是“诊断科学感”的来源。

### 但仍然有一个最大风险

如果最后写作时把：

- observer/controller split
- bucket analysis
- runtime fairness

重新并列写成三个同等 headline claim，论文会再次变散。  
那样它又会退回“整理出三条 interesting findings”的状态，而不是一个层级明确的诊断框架。

所以我的判断是：

- **结构已经足够**
- **危险在于最后写作时重新去层级化**

---

## 2) 哪个层级最关键

最关键的是：

## **主证据层：Benefit-Bucket / Recoverability Analysis**

原因非常明确。

### framing 本身不能把 paper 变成诊断科学

`observer quality != controller gain` 这个句子很重要，但它本身更像一个研究问题或 falsification framing。  
它告诉读者“该问什么”，但不能单独告诉读者“你们已经证明了什么”。

### protocol 本身也不能把 paper 变成诊断科学

runtime-lineage / honest-compute protocol 很重要，但它更像：

- measurement discipline
- auditability
- credibility shield

它决定 paper 是否可信，却不自动决定 paper 是否有机制洞见。

### 真正把 paper 从 aggregate story 拉起来的，是 bucket 层

因为只有 bucket 层能回答：

- 为什么 revision 平均上看似有 gain，但实际上可能只修了少量 case？
- 为什么 code 上 syntax 改善没有转成 execution success？
- 为什么同样的 observer signal，在不同 case 上 intervention value 不一样？

这些问题都是“诊断科学问题”，不是简单的 protocol hygiene 问题。

所以如果要排关键层级：

1. **主证据层（B）**
2. **协议护城河（C）**
3. **主问题 framing（A）**

注意这个排序不是说 A 不重要，而是说：

- A 定义问题
- B 提供诊断证据
- C 保证证据可信

在这三者里，**B 最不可替代**。

---

## 3) 哪些 framing 只能辅助不能进主张

这里必须收得很死，否则最容易在终稿里 overclaim。

### 只能辅助写作、不能进入主张的内容

#### 1. `clinical diagnostic test vs intervention policy`

这是我最喜欢的 framing，但它只能用来帮助读者理解：

- signal 像 diagnostic test
- revision/gate 像 intervention policy
- outcome 像 treatment utility

不能直接写进主张，例如：

- “我们建立了 DLM revision 的 clinical diagnostic framework”
- “我们发现了 DLM revision 的治疗阈值结构”

这些都太重。  
医学类比适合当 framing language，不适合当 contribution wording。

#### 2. `FRACAS / FMEA / reliability dossier`

这类可靠性工程类比非常适合指导 artifact 设计：

- benefit bucket
- runtime fairness matrix
- canonical manifest

但不能写成：

- “本文提出 DLM inference 的 FRACAS framework”

因为我们目前还没有完整做到工程意义上的 failure reporting -> corrective action -> closure revalidation 全链。

#### 3. `observer-based control / separation principle`

这是最容易写过头的一类。

可以帮助解释：

- observer 与 controller 不应混报

但不能写成：

- “我们证明了 DLM revision 中存在控制论意义上的 separation principle”

我们并没有那种 formal result。  
这个类比只能帮助 structure，不能升格为理论主张。

#### 4. `measurement science / experimental physics`

可以帮助 honest compute 写得更稳：

- nominal budget 不等于 realized protocol

但不能写成：

- “我们建立了 DLM inference 的 measurement theory”

这种写法会把一篇很好的 protocol paper 不必要地吹成 grand theory。

### 哪些可以进入主张

真正能进入主张的，必须是和当前资产直接对接、且不借助重类比也成立的内容：

1. 在 tested policies 下，observer quality 不自动转化为 controller gain
2. revision 的 aggregate gain 必须拆成 bucket-level outcome 才可解释
3. realized compute fairness 会影响 headline comparison 的可信解释

这些是“无需类比也成立”的主张。  
类比只是帮助把它们写得更易懂。

---

## 4) 最终推荐结构

我建议的最终结构非常明确：

## A. 主问题 framing（低承诺）

一句话版本：

> 在当前 tested policies 与 current evidence 下，observer quality does not automatically become controller gain.

这句话应该出现在：

- title 副标题或 intro 核心问题
- abstract 的 framing 句
- discussion 的回收句

但**不要**让它承担最重的 contribution 负担。

---

## B. 主证据层（正文核心）

### `Benefit-Bucket / Recoverability Analysis`

这是正文 Results 最该占带宽的部分。

它应承担的角色是：

- 把 aggregate revision gain 拆开
- 把 reasoning/code 的 boundary 差异具体化
- 把 signal/intervention mismatch 变成 case-level evidence

这部分是 paper 最像“诊断科学”的核心。

如果只能保一个最核心 deliverable，就是这个。

---

## C. 协议护城河（方法/附录/补充材料）

### `Runtime-Lineage / Honest-Compute Protocol`

它的定位不该是“顺手补一个 appendix”，而应是：

- protocol contribution
- reviewer-facing auditable layer
- 主证据层的可信度底座

但它仍然是**support lane**，不是最主要的洞见来源。

---

## D. seed spot-check 的位置

`seed_sensitivity_spotcheck` 不应该被写成 headline contribution。  
它最合适的位置是：

- main proposal deliverable 里点名
- 但在论文结构里更像 uncertainty closure / quality gate

也就是说：

- planning 上它是 P0
- paper 上它不是 headline claim

---

## E. abstract / contribution bullets 的推荐数量

我建议 abstract 最终只保留 **2 个 headline claims + 1 个 protocol clause**：

### headline claim 1

- revision gain 必须拆成 bucket-level outcome 才能解释

### headline claim 2

- good observers do not automatically become good controllers under the tested policies

### protocol clause

- honest-compute / runtime-lineage reporting is necessary for credible comparison

不要把它写成三个完全并列的大 claim。  
最好写成：

- 一个主问题
- 一个主证据
- 一个 protocol requirement

---

## 最终一句话判断

当前 seed 已经足以把论文写成“诊断科学”而不是“结果整理”，但前提是最后必须坚持这套层级：**A 只负责 framing，B 负责真正的诊断证据，C 负责 protocol closure；所有医学、控制论、可靠性工程类比都只能帮助写作，不能直接升级成论文主张。**
