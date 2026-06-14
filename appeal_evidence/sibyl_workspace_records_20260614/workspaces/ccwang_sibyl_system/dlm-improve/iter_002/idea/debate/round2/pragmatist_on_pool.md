# pragmatist 对 round2 candidate pool 的判断

## 总判断

从 pragmatist 视角，round2 不该再把 A/B/C 当成三个并列研发方向。它们其实分别对应三种不同层级：

- `A` 更像主问题定义 / claim framing
- `B` 是最关键的机制证据补件
- `C` 是 reviewer-facing 的可信度护城河

所以 round2 的任务不是“选哪一个最酷”，而是把它们压缩成能直接进入 planning 的执行结构。

## 1) A / B / C 的执行优先级

### 第一优先：B `Benefit-Bucket / Recoverability Analysis`

- 这是当前最直接的质量门 blocker。
- 没有它，paper 仍然只是 aggregate story。
- 它也是最省 compute、最容易快速落地的一项。

### 第二优先：C `Runtime-Lineage / Honest-Compute Protocol`

- 这是 honest compute 叙事能不能站稳的最关键 reviewer artifact。
- 它不能替代 B，但能显著降低 implementation confound 攻击面。
- 同时它还能为后续 planning / pilot 提供统一 runtime contract。

### 第三优先：A `Observer-Controller Split`

- 它重要，但更适合作为主问题定义与 framing，而不是先做成独立执行线。
- 当前如果把 A 提到最前面，容易先去抽象 `d(s)` / `g(s)` 定义，却没有先补 reviewer 真会追问的证据。
- pragmatist 判断是：A 必须保留，但执行上要依附 B/C，而不是抢前置资源。

## 2) 哪些应进 planning 主线，哪些只能做 supporting artifact

### 应进入 planning 主线

#### 主线 1：B `Benefit-Bucket / Recoverability Analysis`

- 这是唯一必须进 planning 主线的实证候选。
- 具体应拆成：
  - `benefit_bucket_audit.json`
  - 代表样本与 bucket summary
  - 仅限 headline pairwise comparison 起步

#### 主线 2：最小化的稳健性封口

- 严格说这不在 A/B/C 名字里，但从 pragmatist 角度，它应作为 B 的直接配套子任务并入 planning 主线。
- 也就是：
  - `minimal seed spot-check`
  - 只做 headline pairwise
  - 不单独升格成新 candidate，但必须和 B 一起执行

### 只能做 supporting artifact

#### C `Runtime-Lineage / Honest-Compute Protocol`

- 它非常重要，但更适合作为 supporting backbone，而不是 headline experiment lane。
- 原因：
  - 它主要是整理与声明 discipline
  - 对 paper credibility 很关键
  - 但单独拿出来不会像 B 那样直接补 mechanism hole
- 最合适的定位：
  - planning 中作为并行 supporting artifact
  - writing 中作为 appendix-grade / reviewer-facing artifact

#### A `Observer-Controller Split`

- 它应作为 framing-level 主问题与低承诺 claim 保留。
- 不应作为独立 experiment lane。
- 最合适的定位：
  - proposal / intro / discussion / protocol note 的上位组织框架
  - 支撑 B 和 C 的解释，而不是先独立做一套新验证

## 3) 是否压缩到 2 个 serious candidates

### 我的结论：要，压缩到 2 个 serious candidates

我建议 serious candidates 只保留：

1. `B: Benefit-Bucket / Recoverability Analysis`
2. `C: Runtime-Lineage / Honest-Compute Protocol`

### 为什么 A 不作为第三个 serious candidate

- A 当然不能丢，但它更像“整篇 paper 的问题定义方式”，不是一个独立执行候选。
- 如果把 A 继续保留成第三个 serious candidate，planning 很容易重新变成概念竞争：
  - 先定义 `d(s)` / `g(s)`
  - 先讨论 observer/controller 的 formal distinction
  - 先打磨 framing
- 这会拖慢最短补证据路径。

### A 应该怎么保留

- 作为低承诺 framing：
  - `observer quality != controller gain`
  - 但只在“under the tested policies / current shortlist / current runtime stack”范围内陈述
- 换句话说：
  - `A` 保留为 paper-level claim shell
  - `B/C` 才是 round2 后真正要推进的 serious execution candidates

## 4) 最短补证据路径如何安排

### 我建议的最短路径

#### Step 1：先做 B 的最小可交付版本

- 范围只限 GSM8K headline main pair
- 目标产物：
  - `benefit_bucket_audit.json`
  - bucket examples / summary
- 要求：
  - fixed / harmed / no-effect
  - 代表样本
  - 先不扩成 full failure taxonomy

#### Step 2：并行做最小 seed spot-check

- 这是 B 的质量门封口，不单独升格为新 candidate
- 范围：
  - headline pairwise
  - `>=100` 样本
  - `3 seeds`
- 目标：
  - 只判断 delta 方向是否稳定
  - 不做 full replication

#### Step 3：同步补 C 的 runtime fairness manifest

- 只覆盖 shortlist 核心方法
- 目标产物：
  - canonical asset manifest
  - runtime fairness matrix
  - honest compute note
- 目的：
  - 给 B 的解释提供 reviewer-facing 护城河

#### Step 4：最后再用 A 收口语言

- 不是先做 A，再做 B/C
- 而是等 B/C 有了资产后，再把 A 写成一个低承诺、可证伪、范围清楚的 framing：
  - `observer quality != controller gain`
  - 但不写成强理论定律

## 最终建议给 planning 的结构

如果下一轮要进入 planning，我建议直接写成下面的结构，而不是再围绕 A/B/C 继续抽象讨论：

1. `mainline_task_1`: bucket audit
2. `mainline_task_2`: minimal seed spot-check
3. `supporting_task_1`: runtime fairness / asset-lineage manifest
4. `writing_shell`: observer-controller split 作为低承诺 framing

## 一句话结论

round2 应把 serious candidates 压缩到 2 个：`B` 进 planning 主线、`C` 做并行 supporting artifact，而 `A` 只保留为 framing shell；最短补证据路径是 `bucket audit -> minimal seed spot-check -> runtime fairness manifest -> 再用 observer/controller split 收口语言`。
