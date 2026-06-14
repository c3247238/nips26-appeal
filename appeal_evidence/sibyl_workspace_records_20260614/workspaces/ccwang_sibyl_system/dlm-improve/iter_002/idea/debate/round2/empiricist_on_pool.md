# Empiricist on Candidate Pool (Round 2)

## 总判断

从 empiricist 视角看，round 1 之后的 candidate pool 已经足够收敛，不应再把它理解为三个并列“研究创意”。它们更像三层结构：

1. `Benefit-Bucket / Recoverability Analysis` 是最关键的机制补证据层  
2. `Runtime-Lineage / Honest-Compute Protocol` 是可信度与可审计性层  
3. `Observer-Controller Split` 是问题定义与低承诺 claim 层  

如果目标是尽快通过质量门，那么排序不能按“概念新颖度”排，而必须按“最小充分证据闭环”排。

---

## 1) 三个候选各自需要的最小 artifact

### Candidate A：Observer-Controller Split

### 最小 artifact

- `exp/results/observer_controller_protocol.json`
- `exp/results/diag_signal_gap_audit.json`（已有，但需要被 protocol 化引用）
- 可选：`exp/results/observer_controller_examples.json`

### artifact 最少要回答的问题

- `d(s)` 是什么，如何定义，来自哪个 source asset
- `g(s)` 是什么，如何定义，来自哪个 source asset
- calibration / entropy / instability 三者各自的 measurement object 是什么
- screen metric 与 shortlist metric 的关系是什么
- 哪些 claim 只是 falsification-style，哪些不能推出

### 没有这些 artifact 不能成立的说法

- “observer quality != controller gain” 是一个被清楚验证过的主张
- “calibration 是 strongest observer” 这类表述

### empiricist 评价

这条线重要，但它的证据不应只靠 prose。若没有 protocol artifact，它会非常容易被 reviewer 视为漂亮包装。

---

### Candidate B：Benefit-Bucket / Recoverability Analysis

### 最小 artifact

- `exp/results/benefit_bucket_audit.json`
- `exp/results/benefit_bucket_examples.json`

### artifact 最少要包含

- headline pairwise comparison 定义
- `fixed / harmed / no-effect` 三类计数与比例
- 每类样本的平均 `tokens_changed`
- 每类样本的平均 signal value
- draft / revised correctness 对齐
- 每类 3-5 个代表样本
- 可选的轻量扩展：
  - reasoning vs code boundary
  - shallow vs deep failure

### 没有这些 artifact 不能成立的说法

- “我们理解 revision 到底修了谁、伤了谁”
- “failure taxonomy 已经从 aggregate 走向机制证据”
- “task-dependent boundary 有了样本级支撑”

### empiricist 评价

这是当前唯一一个能直接把论文从 aggregate story 推进到 mechanism evidence 的候选。若只允许优先做一件事，应该先做它。

---

### Candidate C：Runtime-Lineage / Honest-Compute Protocol

### 最小 artifact

- `exp/results/canonical_asset_manifest.json`
- `exp/results/runtime_fairness_matrix.json`

### artifact 最少要包含

- claim 到 source JSON 的一一映射
- figure/table 到 source JSON 的一一映射
- nominal NFE / actual NFE / latency / throughput / batch / backend / compile 的统一 fairness 表
- 哪些差异被视为 algorithmic cost
- 哪些差异被视为 implementation confound

### 没有这些 artifact 不能成立的说法

- “honest compute 是论文贡献，而不是 hygiene”
- “当前 headline ordering 在 reviewer-friendly fairness 条件下可审计”

### empiricist 评价

这条线对质量门极重要，但它更像 backbone，不像独立主故事。它会显著降低 reviewer 的 confound 攻击面，但本身不能替代机制证据。

---

## 2) 谁应该进主线，谁只能当 support

### 应进主线

#### 主线 1：Benefit-Bucket / Recoverability Analysis

原因：

- 这是最直接的 blocker closure。
- 没有它，diagnostic paper 很难证明自己不只是“重新讲已有结果”。
- 它能把 reasoning 主结果与 code boundary 结果连到同一套 failure evidence 上。

#### 主线 2：Observer-Controller Split

但只能以 **低承诺、falsification-style claim** 进入主线。

原因：

- 它定义了论文真正的问题对象。
- 但它必须依附于 protocol artifact 和 bucket evidence，不能单独站立。

### 只能当 support

#### Support：Runtime-Lineage / Honest-Compute Protocol

原因：

- 它是 protocol backbone、credibility shield、reviewer-facing asset。
- 它极其重要，但更适合作为 supporting contribution，而不是唯一 headline。
- 如果把它单独推成主线，论文容易显得“正确但不锋利”。

---

## 3) 是否压缩到 2 个 serious candidates

### 结论

是，应该压缩到 2 个 serious candidates。

### 我建议保留的 2 个 serious candidates

1. `Benefit-Bucket / Recoverability Analysis`
2. `Observer-Controller Split`

### 为什么不把 Runtime-Lineage / Honest-Compute Protocol 也算作第 3 个 serious candidate

不是因为它不重要，而是因为：

- 它对质量门是必需的，但更像“所有 serious candidate 的共同基础设施”
- 它是 support lane，不是 competing headline lane
- 把它单独列成第 3 个 serious candidate，会让 planning 看起来像三条并行主线，实际上会分散焦点

### 更准确的结构

- serious candidate 1：机制证据
- serious candidate 2：主问题定义
- support lane：protocol / runtime fairness / asset lineage

---

## 4) 哪个排序最利于通过质量门

### 最优排序

1. `Benefit-Bucket / Recoverability Analysis`
2. `Runtime-Lineage / Honest-Compute Protocol`
3. `Observer-Controller Split`

### 为什么这个排序最利于过质量门

#### 第一位必须是 Bucket

因为当前最缺的是“样本级机制证据”。没有 bucket，论文仍然只是 aggregate diagnostic story。

#### 第二位必须是 Runtime-Lineage Protocol

因为 bucket 做出来以后，reviewer 下一步就会问：  
“这些 bucket 是不是只是在 implementation confound 下看到的？”

所以 manifest / fairness matrix 必须紧跟其后，形成 reviewer-friendly 闭环。

#### 第三位才是 Observer-Controller Split

因为这条线本质上是 claim organization layer。  
它只有在前两者补齐后，才会显得是“被证据支撑的问题定义”，而不是先验包装。

---

## 我建议的 round2 后收口方式

### 正式 serious candidates

1. `bucket_mechanism_closure`
2. `observer_controller_split`

### supporting lane

1. `runtime_lineage_honest_compute_protocol`

### planning 顺序

1. 先做 `benefit_bucket_audit.json`
2. 再做 `canonical_asset_manifest.json` + `runtime_fairness_matrix.json`
3. 最后把 `observer_controller_protocol.json` 收口为 appendix-grade 定义

---

## 一句话结论

为了最有利于通过质量门，round2 应把 serious candidates 压缩到 2 个：`Bucket-Mechanism` 和 `Observer-Controller Split`；其中真正先做的是 bucket，其次是 runtime-lineage support，最后才轮到把 observer/controller 写成标题级但低承诺的主张。

