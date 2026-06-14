# Empiricist on Proposal Seed (Round 3)

## 总判断

从 empiricist 视角看，`proposal_seed_round2.md` 已经基本走到了正确方向：不再追 generic DLM improvement，也不再给 `Observer-Controller Split` 分配独立执行线，而是把重心放回最能补质量门的 artifact 与 evidence closure。

最后一轮最需要做的，不是再发明新结构，而是把 proposal 里哪些 deliverables 必须被显式点名、哪些 claim 必须主动收缩、哪些内容只能降为 support lane 彻底钉死。

---

## 1) proposal 里必须显式点名的 deliverables

我建议 proposal 正文必须明确点名下面 5 个 deliverables，不能只留在 planning 或 appendix 里。

### 必须显式点名 1：`benefit_bucket_audit.json`

原因：

- 这是当前最核心的机制补证据件。
- 没有它，论文仍然只是 aggregate diagnostic story。
- 它必须被写成主 deliverable，而不是“后续可选分析”。

### 必须显式点名 2：`benefit_bucket_examples.json`

原因：

- bucket 不能只有 aggregate count，必须带 representative cases。
- reviewer 真正相信机制分析，往往靠样本级证据而不是只看比例表。

### 必须显式点名 3：`seed_sensitivity_spotcheck.json`

原因：

- 当前 headline 结果仍然依赖单 seed、小切片。
- 这不是可以藏在 appendix 里的次要补件，而是最低充分稳健性检查。
- 如果 proposal 不显式写这一项，后续非常容易再次被挤掉。

### 必须显式点名 4：`canonical_asset_manifest.json`

原因：

- 这是 reviewer-facing audit map。
- honest compute / runtime fairness 如果没有 manifest，就会继续显得散、弱、难核对。

### 必须显式点名 5：`runtime_fairness_matrix.json`

原因：

- 这是把公平比较条件从 prose 变成 artifact 的关键件。
- 如果 proposal 里不明确写这项，`Runtime-Lineage Protocol` 很容易退化成口头 hygiene。

### 可以点名但可从属于 protocol lane 的 deliverable

- `observer_controller_protocol.json`

这项也重要，但它相对更像“定义收口件”。在 empiricist 排序里，它不如上面 5 项紧迫。若 proposal 需要控制篇幅，可以把它放在 protocol lane 里统一说明，而不必单列成 headline deliverable。

---

## 2) abstract claims 最多保留几条

### 结论

最多保留 **2 条 headline claims**。

### 为什么不能保留 3 条

当前证据结构还不够厚。若 abstract 同时高调承诺：

1. observer/controller split
2. bucket-level mechanism
3. runtime-lineage / honest-compute protocol

那么很容易在审稿视角里变成“三个都对，但每个都还差一口气”。  
对质量门最有利的策略不是 claim 数量多，而是 claim 密度高、artifact 对应关系清楚。

### 我建议 abstract 只保留的 2 条

#### Claim 1

在当前 tested policies 与 current evidence 下，`observer quality != controller gain`，因此 revision 结果不能只按 aggregate gain 报告。

#### Claim 2

revision gain 必须通过 `bucket-level outcomes + realized compute fairness` 一起解释，否则结论容易被 implementation confound 或 aggregate masking 误导。

### 不建议单独保留为 abstract 独立 claim 的内容

- runtime-lineage protocol 本身
- canonical asset manifest 本身
- seed spot-check 本身

这些都应该为上面两条主张服务，而不是自己升格成第三条 headline claim。

---

## 3) 哪些内容只能作为 support lane

### Support lane 1：`Runtime-Lineage / Honest-Compute Protocol`

原因：

- 它极其重要，但更像“可信比较条件”而不是主问题本身。
- 它负责让主证据层与主 claim 不被 reviewer 用 confound 轻易击穿。
- 如果把它写成独立主命题，paper 容易显得像 protocol hygiene paper，而不是有解释力的 diagnostic paper。

### Support lane 2：`seed_sensitivity_spotcheck`

原因：

- 这是 quality-gate 必需件，但角色是 robustness closure，不是主贡献。
- 应在 proposal 里被显式点名为 deliverable，但在论文结构里应属于 support evidence。

### Support lane 3：`observer_controller_protocol.json`

原因：

- 它是定义澄清与 appendix-grade formalization。
- 它应该支持 intro / discussion / appendix 的 claim clarity，而不应成为独立 narrative lane。

### Support lane 4：任何 minimal controller / auxiliary probe

原因：

- 一旦进主线，就会重新滑回 method-adjacent paper。
- 如果最终还有余量，只能作为 appendix-only sanity probe。

---

## 4) 最终推荐结构

### 我建议的 proposal 最终结构

#### A. 论文定位

`compute-normalized diagnostic paper`

#### B. 主命题

只保留一个低承诺主命题：

- 在当前 tested policies 与 current evidence 下，observer quality 与 controller gain 不自动等价，因此 DLM revision 需要同时报告 bucket-level outcomes 与 realized compute fairness。

#### C. 主证据层

`Benefit-Bucket / Recoverability Analysis`

必须包含：

- `benefit_bucket_audit.json`
- `benefit_bucket_examples.json`

这是 proposal 里唯一应该被写成“main evidence engine”的部分。

#### D. 支撑协议层

`Runtime-Lineage / Honest-Compute Protocol`

必须包含：

- `canonical_asset_manifest.json`
- `runtime_fairness_matrix.json`
- `observer_controller_protocol.json`

这层的职责是：

- 让主证据可审计
- 让 observer/controller claim 有 protocol 定义
- 避免 implementation confound

#### E. 稳健性收口层

`Minimal Robustness Closure`

必须包含：

- `seed_sensitivity_spotcheck.json`

这层不应被写成 contribution bullet，更像 explicit quality-gate closure。

---

## 对 proposal_seed_round2 的具体回答

### 问题 1：`Observer-Controller Split` 是否只保留为 framing，还是应进入 contribution bullet？

我的回答：

- 可以进入 contribution bullet，但只能以 **低承诺 framing claim** 的形式进入
- 不能被写成一个独立执行 candidate
- 更不能被写成“我们提出了一个新 decomposition framework”

### 问题 2：`Runtime-Lineage Protocol` 是独立 contribution，还是仅作为 main evidence 的 support lane？

我的回答：

- 应写成 **support lane + protocol backbone**
- 可以在 contribution 里出现，但不能占据 headline 位

### 问题 3：abstract 里应该保留几个 headline claim，2 个还是 3 个？

我的回答：

- **2 个**

### 问题 4：`seed_sensitivity_spotcheck` 应被写入主 proposal deliverables，还是只写进 planning appendix？

我的回答：

- 必须写入主 proposal deliverables
- 但在论文 narrative 中仍属于 support evidence，而不是 headline contribution

---

## 一句话结论

empiricist 的最终建议是：proposal 必须显式点名 `benefit_bucket_audit.json`、`benefit_bucket_examples.json`、`seed_sensitivity_spotcheck.json`、`canonical_asset_manifest.json`、`runtime_fairness_matrix.json`；abstract 最多保留 2 条主张；真正的主线只应有“bucket 主证据”，其余都应作为 support lane 为质量门服务。

