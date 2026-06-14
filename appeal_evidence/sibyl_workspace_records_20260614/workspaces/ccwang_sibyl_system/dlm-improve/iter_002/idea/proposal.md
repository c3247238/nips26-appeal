# Iteration 3 Final Proposal

## 最终裁决

本轮最终 proposal 明确收束为一篇 **compute-normalized diagnostic / protocol paper**，不再回到 `TIGER hero`、`Calibration-Aware` 或任何 generic new controller / scheduler 叙事。

## 论文定位

这篇 paper 的身份应明确为：

- 一篇围绕 training-free DLM revision 的 **诊断型论文**
- 不是“提出更强 controller”的方法论文
- 不是“我们更诚实地报 runtime”的 hygiene note
- 而是回答一个更具体的问题：
  - 为什么 aggregate revision gain 不足以解释真实效应结构；
  - 为什么 observer quality、controller gain 与 realized compute fairness 必须分开处理。

## 主命题

本轮只保留一个低承诺、可证伪、与现有资产直接对接的主命题：

> 在当前 tested policies 与 current evidence scope 下，`aggregate gain alone` 不足以解释 training-free DLM revision 的真实效应结构；更强的 observer quality 不会自动转化为更强的 controller gain，因此结论必须同时结合 bucket-level outcomes 与 realized compute fairness 来解释。

这里要特别注意：

- `observer/controller split` 是主问题定义，不是独立 hero contribution
- 它只能写成 scoped diagnostic finding，不能写成一般理论
- 任何超出当前 evidence scope 的 regime law、统一框架、普适评价标准都不成立

## A / B / C 层级（固定，不再并列）

### A. 主问题定义：`Observer-Controller Split`

- 角色：
  - 标题、摘要、引言、讨论中的上位问题定义
- 允许的表述强度：
  - stronger observer does not automatically become stronger controller gain
  - aggregate gain can hide this mismatch
- 禁止的表述强度：
  - 新 framework
  - 一般理论
  - 普适标准
  - 独立 serious execution candidate

### B. 主证据层：`Benefit-Bucket / Recoverability Analysis`

- 角色：
  - 全文最核心的 evidence engine
  - 把 aggregate story 变成样本级机制证据
- 必须回答的问题：
  - revision 修了谁
  - revision 伤了谁
  - revision 对谁无效
  - reasoning / code boundary 上哪些只是浅层修复，哪些涉及深层失败
- 允许的表述强度：
  - bucket-level recoverability evidence
  - fixed / harmed / no-effect decomposition
  - boundary-sensitive failure audit
- 禁止的表述强度：
  - 完整 failure taxonomy
  - 跨任务稳定 regime law
  - 已经解释清楚全部机制

### C. 支撑协议层：`Runtime-Lineage / Honest-Compute Protocol`

- 角色：
  - reviewer-facing credibility shield
  - claim-to-asset auditable mapping
  - method effect 与 implementation confound 的隔离条件
- 它非常重要，但只能作为：
  - support lane
  - protocol backbone
  - appendix-grade artifact bundle
- 它不能被写成：
  - 独立 hero claim
  - “我们提出新的 benchmark standard”
  - 纯 hygiene appendix

## 主证据层与 support lane 的明确分工

### 主证据层

唯一的主证据层是：

- `Benefit-Bucket / Recoverability Analysis`

原因：

- 只有它能把 GSM8K 主结果、signal audit 和 HumanEval boundary 真正连起来
- 只有它能把 “revision 何时帮助、何时伤害” 从 aggregate 均值推进到样本级证据
- 没有它，diagnostic thesis 仍然只是结果整理

### Support lane

support lane 包含两类内容：

1. `Runtime-Lineage / Honest-Compute Protocol`
2. `Minimal Robustness Closure`

其中：

- protocol lane 负责让结论可审计
- robustness lane 负责让 headline delta 不再只靠单 seed 站立

它们都必须做，但都不应抢走主证据层的位置。

## Serious Execution Candidates

最终 serious execution candidates 固定为 2 个：

### 1. `cand_bucket`

- 标题：`Benefit-Bucket / Recoverability Analysis`
- 地位：主证据层
- 目标：产出 reviewer 能直接检查的样本级机制证据

### 2. `cand_protocol`

- 标题：`Runtime-Lineage / Honest-Compute Protocol`
- 地位：支撑协议层
- 目标：把 honest compute、asset lineage、observer/controller reporting discipline 做成可审计 artifact

明确不再作为 serious execution candidate 的内容：

- `Observer-Controller Split`
  - 它保留，但只作为主命题 / framing shell
- `Minimal Controller for Decoupling`
  - drop 为 appendix-only optional probe
- 任何 `TIGER hero`、`Calibration-Aware`、generic controller/scheduler

## 必做 Deliverables

本轮 proposal 必须显式锁定以下 deliverables，不能再放到 future work：

### 主证据 deliverables

1. `exp/results/benefit_bucket_audit.json`
2. `exp/results/benefit_bucket_examples.json`

### 支撑协议 deliverables

3. `exp/results/canonical_asset_manifest.json`
4. `exp/results/runtime_fairness_matrix.json`
5. `exp/results/observer_controller_protocol.json`

### 质量门 deliverable

6. `exp/results/seed_sensitivity_spotcheck.json`

其中优先级必须理解为：

- `benefit_bucket_audit` 是主任务
- `seed_sensitivity_spotcheck` 是质量门，不是可选项
- protocol bundle 是 credibility closure，不是写作装饰

## Planning 优先级

### Priority 1：先做 bucket audit

- 先覆盖 headline GSM8K 主 pairwise comparison
- 必须给出：
  - `fixed / harmed / no-effect` 计数与比例
  - 每类平均 `tokens_changed`
  - 每类平均 signal value
  - draft / revised correctness 对齐
  - 代表样本

### Priority 2：并行做最小 seed spot-check

- 只做 headline pairwise comparison
- 范围控制在最小必要规模
- 目标不是 full replication，而是检查 delta 方向一致性

### Priority 3：收 protocol bundle

- `canonical_asset_manifest.json`
- `runtime_fairness_matrix.json`
- `observer_controller_protocol.json`

这三者要一起做，不能再拆散，否则最容易再次出现 claim、table、JSON 彼此脱节。

### Priority 4：最后再收口语言

- 等 artifact 先落盘
- 再把 `observer/controller split` 写进 intro / abstract / discussion
- 保持 paper.md 与 main.tex 同步

## Abstract / Contribution 层级建议

abstract 最多保留 2 条 headline claims：

1. `aggregate gain` 不足以解释 revision 的真实效应结构，必须看 bucket-level outcomes
2. 这些结论只有在 realized compute fairness 与 runtime-lineage 可审计的前提下才可信

`observer/controller split` 应作为 framing sentence 出现，而不是第三条并列 headline claim。

## 上轮经验教训的吸收

本轮 proposal 明确吸收以下教训：

1. `benefit buckets` 不能再停留在 future work
   - 否则 diagnostic thesis 仍然只是 aggregate story
2. `seed spot-check` 仍是硬缺口
   - 不再接受只靠单 seed 的 headline ordering
3. `signal audit` 的定义必须 appendix-grade 写透
   - `d(s)` / `g(s)` 的来源、含义、可比性都要有 protocol artifact
4. `runtime fairness` 必须 reviewer-friendly
   - honest compute 不能散落在 JSON 和 prose 中
5. cross-task 证据继续只按 boundary slices 处理
   - 不把 MATH500 / HumanEval 写成稳定规律
6. proposal、paper.md、main.tex 必须保持叙事一致
   - 不再允许 ideation 已 pivot，但写作仍残留旧 hero 叙事

## Support Lane 的明确边界

下面这些内容只能作为 support lane 或 appendix-level add-on：

- `seed_sensitivity_spotcheck`
- `observer_controller_protocol`
- 任何 minimal controller sanity probe
- 任何跨学科类比（clinical / control / FRACAS / measurement science）

这些内容可以帮助组织论文，但不能升级成新的独立主张。

## 备选 pivot（只保留 2 个）

### Backup 1：`cand_minimal_scope`

- 如果 bucket audit 证据不够锋利，就进一步收缩成：
  - `honest compute + bucket audit` 的最小诊断论文
- 明确放弃更强 observer/controller 表达

### Backup 2：`cand_boundary_audit`

- 如果 GSM8K headline pair 不够稳，但 boundary 证据更有解释力，就转向：
  - `reasoning benefit vs code harm` 的结构边界审计稿
- 仍然不回到新 controller 叙事

## 当前最终决定

当前 iteration 的最终决定如下：

- **论文定位**：compute-normalized diagnostic / protocol paper
- **主命题**：aggregate gain 不足以解释 revision 效应结构；observer quality 不自动变成 controller gain
- **主证据层**：Benefit-Bucket / Recoverability Analysis
- **support lane**：Runtime-Lineage / Honest-Compute Protocol + Minimal Robustness Closure
- **serious execution candidates**：仅保留 `cand_bucket` 与 `cand_protocol`
- **planning 第一优先**：bucket audit
- **明确不再回头**：TIGER hero / Calibration-Aware / generic controller
