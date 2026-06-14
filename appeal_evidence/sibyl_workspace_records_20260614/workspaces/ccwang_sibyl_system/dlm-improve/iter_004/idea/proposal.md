# Iteration 4 Proposal

## Title

Evidence-Driven Pivot to Local Repair Objects and Entropy-Routed Compute for Training-Free DLMs

## Abstract

iteration 4 的真实 screening pilot 已经给出一个必须正视的负结论：`cand_mgcd` 与 `cand_dsg` 都没有在 audited slice 上穿透 `RAND-84`，因此旧的 `MGCD front-runner` 叙事必须终止。新的 proposal 不再把研究主语设为“如何把 controller 做得更复杂”，而是改写为两个更可证伪的问题：第一，DLM revision 的正确干预对象是否其实是 **span / boundary / uncertainty island**，而不是 token-wise 或全局 stateful controller；第二，`raw entropy` 是否更适合作为 **compute routing / stopping** 信号，而不是直接作为 semantic controller。基于 6 个视角的一致意见，我们将 serious candidate pool 收束为 3 个对象：`cand_bsr` 作为新的质量主线，`cand_espd` 作为速度主线备选，`cand_ugr` 作为只在前两者出现信号后才继续推进的 stretch candidate。当前 front-runner 选为 `cand_bsr`，因为它得到 innovator / contrarian / interdisciplinary / empiricist 的最强交集支持，也最直接回应了本轮 `PIVOT` 的核心诊断：**observer signal 可能没错，错的是 repair object**。

## Evidence-Driven Revisions

### 本轮明确改动

1. `cand_mgcd` 与 `cand_dsg` 从 serious pool 中移除，只保留为 archived negative controls。
2. 旧 `BSR` 不再作为 `MGCD` 的 complexity control 附庸，而是重写为新的 object-level mainline：`cand_bsr`。
3. 新增独立速度线 `cand_espd`，把 entropy 从“修哪里”改写成“哪里值得继续花 compute”。
4. 保留 `cand_ugr` 作为第三候选，但只有在 `cand_bsr` 或 `cand_espd` 已显示早期正信号时才晋级。
5. 下一轮 pilot 必须回到项目长期约束：`n>=100`、max-safe batch probing、compile/backend parity、代码任务单独报告 `syntax-valid` 与 `exec-valid`。

### 哪些旧判断被削弱或否证

- “更复杂的 training-free controller 会自然超过 sham control” 被削弱，至少在 `MGCD-lite` / `DSG` 这轮实现上不成立。
- “`raw entropy` 应直接驱动语义 revision” 被削弱；当前更稳健的解释是它更像 observer / routing signal。
- “只要再补一些工程细节，`MGCD` 就可能自然转正” 被否证；这轮证据不足以支持继续投入 narrative budget。

### 哪些信号仍值得保留

- `raw entropy` 对错误有信息量。
- 更长 denoising trajectory 不是单调更好，存在 budget reallocation 空间。
- revision 本身并非完全无效，但当前失败更像是 **错修了对象** 或 **把观察直接升级成干预**。

## Synthesis Rationale

### 六个视角的共识

- `MGCD/DSG` 不能再作为 serious candidate pool 的存活对象。
- 下一轮必须围绕 **局部对象** 或 **compute routing** 重建候选池，而不是继续追逐更重 controller。
- pilot 设计必须优先 falsification，尤其要保留 `RAND-84` sham control，并把 runtime parity 写成硬约束。

### 主要分歧

- `pragmatist` 与 `theoretical` 更偏向先做速度线（`cand_espd` / `cand_srts`）。
- `innovator`、`interdisciplinary`、`contrarian`、`empiricist` 更偏向先做对象线（`SIR / SGR / BSR++ / CBR` 一族）。
- 第三个 serious candidate 是应该押 `UGR` 这样的 benefit-estimation，还是押 `late-commit / cache-first` 这样的更保守 speed lane，存在分歧。

### 我的综合判断

我给 `contrarian` 与 `empiricist` 更高权重，因为这一轮最需要的是服从负证据，而不是继续包装旧主线；我也给 `interdisciplinary` 与 `innovator` 较高权重，因为它们与前两者在“局部对象重立题”上出现了强一致。`pragmatist` 与 `theoretical` 的价值主要体现在把第二候选压成更清晰的速度线，并把 `n>=100`、`<=1 GPU-hour`、风险预算这些可执行约束写实。综合后，最稳的 front-runner 不是单一视角里最炫的新名字，而是一个合并后的对象：

> `cand_bsr` = BSR / SIR / SGR / CBR family 的统一版本，即“先诊断 uncertainty island，再做 boundary-stable、局部、可审计的 repair”。

## Motivation

当前最重要的不是证明 DLM training-free revision 一定能赢，而是把失败机制重新定位清楚。iteration 4 的真实负证据提示：

- `observer != controller`
- `signal exists != intervention validated`
- `token-wise repair object may be wrong`

因此新的 proposal 追求的不是“大方法一次成功”，而是一个更窄、更硬、更可证伪的 candidate program：

1. 先回答局部 repair object 是否正确。
2. 再回答 entropy 是否更适合做 routing / stopping。
3. 最后才测试更昂贵的 benefit-estimation 是否值得。

## Novelty Boundary

这轮 proposal 明确不声称以下事情：

- 不是新的 global controller family
- 不是 trained / search-heavy test-time scaling
- 不是把 sampler / backend / compile gain 写成方法 gain
- 不是“再做一个 heuristic bundle 看能不能涨点”

这轮真正要检验的是两个更窄的机制命题：

1. repair object 是否该从 token 改成 uncertainty island / span / boundary
2. entropy 是否该从 controller 信号改成 routing / stopping 信号

## Research Questions

### RQ1

把 revision 对象从 token 改成 uncertainty island / span / boundary，是否会第一次带来对 `RAND-84` 的真实 sham-control separation？

### RQ2

`raw entropy` 的最好用途究竟是 **semantic controller**，还是 **compute routing / early stopping / active frontier shrinking**？

### RQ3

如果局部对象线与速度线都出现正信号，是否可以再引入极便宜的 uplift estimator，进一步减少无效修复与稳定区域副损伤？

### RQ4

在统一 runtime contract 下，任何 observed gain 能否在 compile/backend parity、max batch probing、matched compute 约束下依然成立？

### RQ5

对于 `MBPP` 这类结构化输出任务，局部 repair + syntax guard 是否能减少 syntax breakage，而不是通过更多重写制造新的 execution harm？

## Candidate Program

### cand_bsr

`Boundary-Stable Repair`

- 角色：当前 `front_runner`
- 来源综合：
  - `innovator` 的 `SIR`
  - `interdisciplinary` 的 `SGR`
  - `empiricist` 的 `BSR`
  - `theoretical` 的 `CBR`
- 核心思想：
  - 用 `raw entropy`、局部稳定度、邻域张力等信号先识别 uncertainty island
  - 对 island 做 span aggregation，而不是 token-wise 零碎改写
  - 用 boundary locking / stable boundary protection 限制修复范围
  - 只在局部区域执行最小必要 revision，避免把高风险观察升级成全局重写
- 当前冻结版本：
  - 在 planning 中必须先冻结为 `cand_bsr_v1`
  - `cand_bsr_v1` 只允许包含 5 个模块：
    - island score
    - span merge rule
    - boundary lock width
    - max revision steps
    - accept / reject rule
  - 除这 5 项外，不允许再偷偷加入额外 draft、额外 verifier 或 side search；若需要新增模块，必须升版为 `cand_bsr_v2`
- 为什么它现在排第一：
  - 这是对本轮负证据最直接的响应
  - 相比 `MGCD` 更容易归因，相比 `DSG` 更像真正换了 repair object
  - 跨四个视角出现了同一方向上的独立收敛
- 下一轮 success signal：
  - `GSM8K audited slice` 上超越 `RAND-84`
  - `harmed stable tokens` 下降
  - `MBPP` 上 `syntax-valid` 与 `exec-valid` 不恶化
- kill rule：
  - 若 `cand_bsr` 不能超过 `RAND-84`，直接 kill
  - 若收益只来自更大范围改写，而不是 harm containment，直接 kill

### cand_espd

`Entropy-Stable Parallel Decoding`

- 角色：速度主线 backup
- 来源综合：
  - `pragmatist` 的 `ESPD`
  - `theoretical` 的 `SRTS`
  - `interdisciplinary` 的 `HBR`
  - `empiricist` 的 `EBR`
- 核心思想：
  - 不再让 entropy 直接决定“修哪里”，而是决定“哪里还值得继续花 compute”
  - 用 active frontier shrinking、risk-bounded stopping、budget reallocation 把计算集中到高风险区域
  - 目标是 wall-clock / throughput 改善，不追求解释成 semantic controller
- 为什么保留：
  - 速度本来就是用户明确目标
  - 这条线与当前正信号最一致：entropy 有信息量，长轨迹非单调
  - 它和 `cand_bsr` 互补，不会重复烧同类 GPU 预算
- 下一轮 success signal：
  - 在 matched wall-clock 或 matched budget 下实现显著提速
  - 不出现对 `RAND-84` 的明显崩塌
- kill rule：
  - 若既没有 `>=1.5x` 速度收益，也没有质量收益，kill
  - 若收益只来自 runtime mismatch 或 hidden compute，kill

### cand_ugr

`Uplift-Gated Revision`

- 角色：stretch backup
- 来源综合：
  - `innovator` 的 `UGR`
  - `contrarian` 的“不要把 observation 直接升级成 intervention”
  - `theoretical` 的 risk-allocation framing
- 核心思想：
  - 在 `cand_bsr` 识别出的 island 上，再加一个极便宜的 uplift / benefit 估计器
  - 只有当估计净收益为正时才执行 revision
  - 它直接针对这轮最核心的断裂：`observer signal informative` 不等于 `controller action beneficial`
- 为什么只排第三：
  - 新颖，但更容易引入 hidden branching
  - 若 `cand_bsr` 自身都没有 signal，引入 uplift 层只会增加复杂度
- 晋级条件：
  - 只有 `cand_bsr` 或 `cand_espd` 已显示早期正信号时，才允许进入正式 screening
- kill rule：
  - 若 uplift estimate 与 plain `cand_bsr` 相比没有更高的 repair/harm ratio，kill

## Mechanism Controls

为了避免下一轮又把 bundle gain 误写成机制 gain，以下 sham controls 必须进入 planning：

- `RandSpan-84`
  - span 长度与 touched token 数匹配 `cand_bsr`
  - 但 span 随机
- `EntropySpan-NoBoundary`
  - 使用同样的 entropy island
  - 但去掉 boundary protection
- `BoundaryLock-RandomSpan`
  - 有 boundary lock
  - 但 span 不由 uncertainty-driven diagnosis 产生
- `ESPD-FixedFrontier`
  - active frontier ratio 与 `cand_espd` 匹配
  - 但 routing 不看 entropy
- `SyntaxGuard-only`
  - 只加结构约束
  - 不加新的 local repair object

## Archived Negative Controls

### cand_mgcd

- 状态：`dropped`
- 保留用途：
  - 作为 failure precedent
  - 作为 hidden-compute / richer-controller 反例
- 不再允许的写法：
  - `MGCD front-runner`
  - `MGCD 只差一点成功`

### cand_dsg

- 状态：`dropped`
- 保留用途：
  - 作为“单轨 signal-only gate 仍可能不够”的反例
- 不再允许的写法：
  - 把 `DSG` 包装成正向 speed fallback
  - 把 `DSG` 的 `vs_rand84_net=0` 写成支持性证据

## Hypotheses

### H1: Local Repair Object Hypothesis

若真正错误的是 repair object，而不是 observer signal，那么 `cand_bsr` 应在 matched compute 下首次超过 `RAND-84`，且主要收益来自 harm containment 而不是大范围重写。

### H2: Entropy-As-Routing Hypothesis

若 entropy 更适合 routing 而不是 controller，那么 `cand_espd` 应能在不显著恶化质量的前提下，把 compute 从稳定区域移开并获得显著速度收益。

### H3: Benefit-Gating Hypothesis

若 “observer -> action” 之间还缺一个 benefit estimator，那么 `cand_ugr` 应在 plain `cand_bsr` 之上提高 repair/harm ratio，并减少不必要的 touched islands。

### H4: Runtime-Parity Robustness Hypothesis

任何 gain 只有在 compile/backend parity、max-safe batch probing、matched compute 下仍成立，才可被解释为方法增益。

### H5: Structured-Code Robustness Hypothesis

对 `MBPP` 之类任务，局部 repair + syntax guard 应带来更好的 `syntax-valid` / `exec-valid` 平衡，而不是提升部分 accuracy 同时制造大量结构性失效。

## Screening Pilot Plan

### P0. Shared Runtime Contract

- `sample_count >= 100`
- 若总样本只有 `100`，默认拆成 `50 design + 50 confirm`
- 更理想配置是 `100 design + 100 confirm`
- 先 probe safe max batch size，再正式跑
- 首选 `flash_attention_2 + torch.compile`，不可用时必须记录回落原因
- 所有 candidate / baseline 使用同一 backend 与 compile 策略
- 所有 auxiliary logic 必须单独记账：
  - 额外前向次数
  - 额外 wall-clock 开销
  - side model / syntax guard 开销
- 代码任务强制记录：
  - `syntax-valid`
  - `exec-valid`
  - `repair_count`
  - `harm_count`

### P1. cand_bsr on GSM8K Audited Slice

- 比较对象：
  - `RAND-84`
  - `CARD-84`
  - `cand_bsr`
  - `RandSpan-84`
  - `EntropySpan-NoBoundary`
  - `BoundaryLock-RandomSpan`
- 判定：
  - 若 `cand_bsr <= RAND-84`，quality-first 线直接失败
  - 若超越 `RAND-84` 且 harm containment 更好，则进入 `MBPP`

### P2. cand_espd on GSM8K Audited Slice

- 比较对象：
  - `RAND-84`
  - `CARD-84`
  - `cand_espd`
  - `ESPD-FixedFrontier`
- 重点日志：
  - wall-clock
  - tokens/sec
  - active frontier ratio
  - equal-compute quality
  - equal-quality speed
  - stopped-step distribution
- 判定：
  - 若无显著速度收益且质量无净增益，直接失败

### P3. cand_bsr on MBPP + Syntax-Guard Ablation

- 只在 P1 有正信号时执行
- 必做对照：
  - `cand_bsr`
  - `cand_bsr + syntax_guard`
  - `SyntaxGuard-only`
- 重点看：
  - `syntax-valid`
  - `exec-valid`
  - `repair/harm` 分解

### P4. cand_ugr Conditional Pilot

- 只在 `cand_bsr` 或 `cand_espd` 已显示正信号时执行
- 目的不是立刻追求 headline gain，而是检验 uplift gate 是否真能减少无效干预

## Promotion Gates

### Hard Gates

- 任一 serious candidate 若未超过 `RAND-84`，不得进入 full benchmark
- 若 runtime contract 仍不统一，不得把结果写成方法增益
- 若 proposal / hypothesis / candidate metadata 与 pilot artifact 再次漂移，整轮视为未准备好晋级

### Upgrade Gates

- 只有 `cand_bsr` 真正超过 `RAND-84`，才允许把 “repair object 重立题” 写成正向结论
- 只有 `cand_espd` 在统一 runtime 下保住质量并带来速度收益，才允许把 entropy-routing 写成速度贡献
- 只有 `cand_ugr` 提高 repair/harm ratio，才允许增加 benefit-estimation 层

## Unified Reporting Contract

下一轮每个 serious candidate 都必须进入同一套 2D 报表，而不是各报最有利指标：

### Table A: quality @ equal compute

- accuracy / pass rate
- repair_count
- harm_count
- harmed_stable_tokens

### Table B: speed @ equal quality band

- wall-clock
- tokens/sec
- active frontier ratio
- extra forward count

代码任务额外统一报告：

- `syntax-valid`
- `exec-valid`
- `pass@1`

## Expected Contributions

### 如果下一轮 pilot 成功

1. 证明 training-free DLM 的主问题可能是 repair object，而不只是 controller 强弱
2. 给出一条更可信的速度线：entropy 作为 routing / stopping 信号
3. 建立一个 reviewer 更容易接受的 claim ladder：
   - observer signal
   - local repair object
   - compute routing
   - optional benefit gating

### 如果下一轮 pilot 再次失败

1. 更强的否证：说明即使改成局部对象与 routing，training-free DLM 仍未越过 sham control
2. 更干净的归因：失败发生在 object、routing，还是 uplift gating
3. 更少的 story drift：因为这轮已经明确放弃 `MGCD comeback`

## What Changed From The Previous Round

- 从 “`MGCD / BSR / DSG` 里选 winner” 改成 “`BSR / ESPD / UGR` 的新 candidate program”
- 从 “stateful controller 是否成立” 改成 “repair object 与 routing signal 是否对题”
- 从 “继续救旧 front-runner” 改成 “把旧 front-runner 归档成负例控制”
- 从 “先想大方法” 改成 “先做能在 `<=1 GPU-hour` 内被 falsify 的 pilot”
