# pragmatist 对 round2 proposal seed 的最终意见

## 总判断

从 pragmatist 视角，`proposal_seed_round2.md` 已经基本收对了，方向上不需要再扩，也不该再把任何 method-adjacent 候选放回主池。最后一轮最重要的任务不是“再找更优雅的说法”，而是把执行层和交付层写死，避免进入 planning 时又重新漂移。

## 1) serious execution candidates 是否最终定为 2 个

### 我的结论：是，最终就定为 2 个

应最终固定为：

1. `Benefit-Bucket / Recoverability Analysis`
2. `Runtime-Lineage / Honest-Compute Protocol`

### 为什么只保留这 2 个

- 这 2 个都直接对应当前 paper 的硬缺口。
- 两者都能在不引入新 fairness confound 的情况下落地。
- 两者都能转化为 reviewer 一眼能检查的 artifact，而不是新一轮方法叙事。

### 为什么不把 `Observer-Controller Split` 算作第三个 serious execution candidate

- 它是上位 framing，不是独立执行线。
- 如果继续把它当第三个 serious candidate，planning 很容易又先去打磨定义、概念和 contribution bullet，而不是先补证据。
- pragmatist 判断是：`Observer-Controller Split` 必须保留，但只能作为低承诺主命题 / falsification-style shell，不能再单列成 execution candidate。

## 2) planning 顺序是否合理

### 我的结论：基本合理，但要再明确一点

当前 seed 的顺序是：

1. `benefit_bucket_audit.json`
2. 并行最小 `seed_sensitivity_spotcheck.json`
3. `canonical_asset_manifest.json` 与 `runtime_fairness_matrix.json`
4. 最后再收口 `observer/controller split`

这个顺序整体是对的，我支持。

### 我建议的更明确版本

#### Step 1：先做 `benefit_bucket_audit`

- 这是唯一最直接补 mechanism hole 的主任务。
- 不应再被任何定义澄清、写作重组或 protocol note 抢前置资源。

#### Step 2：并行做最小 `seed_sensitivity_spotcheck`

- 这项虽然不在 2 个 serious candidates 名字里，但它是 planning 主线里的必备 quality-gate 子任务。
- 它的功能不是新主张，而是给主证据层封口。

#### Step 3：同步收 `canonical_asset_manifest + runtime_fairness_matrix + observer_controller_protocol`

- 这里我建议把三个 supporting artifact 作为一个 protocol bundle 处理，而不是散着做。
- 否则最容易出现 manifest 先写了、fairness table 还没补齐、protocol note 又和 JSON 脱节的问题。

#### Step 4：最后才把 `observer/controller split` 写进 proposal / intro / discussion

- 顺序必须是“先有 artifact，再有 framing”
- 而不是“先把 A 写得很漂亮，再回头补证据”

## 3) 哪些 deliverables 必须明确写进 proposal

### 必须明确写进 proposal 主体的 deliverables

我建议 proposal 里不要只写抽象方向，必须把下面这些文件级 deliverables 明确列进去：

#### 主证据 deliverables

- `benefit_bucket_audit.json`
- `benefit_bucket_examples.json`

原因：

- 这是 serious candidate 1 的实体化交付
- 不写进去，后面最容易再次被降级成“之后有时间再做”

#### 协议护城河 deliverables

- `canonical_asset_manifest.json`
- `runtime_fairness_matrix.json`
- `observer_controller_protocol.json`

原因：

- 这是 serious candidate 2 的最小可审计包
- 少任何一个，runtime-lineage story 都会变松

#### 必须写进 proposal 的质量门 deliverable

- `seed_sensitivity_spotcheck.json`

### 关于 seed spot-check，我的明确意见

`seed_sensitivity_spotcheck` 不能只写进 planning appendix，必须写进 proposal deliverables。

原因很简单：

- 它虽然不是独立 serious candidate，但它是主证据层能否成立的质量门。
- 如果不在 proposal 里写明，执行时最容易被当作“时间不够可以先跳过”的可选件。
- 当前 workspace 的 lessons learned 已经明确说这是硬缺口，不应再模糊处理。

## 4) 最终推荐结构

### 我推荐的最终 proposal 结构

#### A. 论文定位

- `compute-normalized diagnostic / protocol paper`
- 明确排除：
  - TIGER hero
  - Calibration-Aware hero
  - generic new controller / scheduler

#### B. 主命题（低承诺 framing）

- 在当前 tested policies 与 current evidence 下，`observer quality != controller gain`
- 因此 revision 论文不能只报 aggregate gain，必须同时报告：
  - bucket-level outcomes
  - realized compute fairness
  - observer/controller split

这里的关键是：

- `observer/controller split` 放在主命题里
- 但不要把它写成单独 execution candidate 或强理论定律

#### C. Serious Execution Candidate 1

- `Benefit-Bucket / Recoverability Analysis`
- 这是主证据层
- 也是 planning 主线第一优先

#### D. Serious Execution Candidate 2

- `Runtime-Lineage / Honest-Compute Protocol`
- 这是协议护城河
- 是并行 supporting lane，但仍属于 serious execution candidate

#### E. Quality-Gate Deliverable

- `seed_sensitivity_spotcheck.json`
- 定位：
  - 不是单独 candidate
  - 但必须作为 proposal 明写的 deliverable
  - 并在 planning 中与 Candidate 1 同步推进

#### F. 写作收口策略

- 先产 artifact
- 再写 claim
- 最后再决定 abstract 里保留 2 个还是 3 个 headline claim

### 关于 abstract headline claim 的 pragmatist 建议

我建议 abstract 最终只保留 `2` 个 headline claim，不要扩到 `3` 个并列 headline：

1. bucket-level revision evidence
2. honest compute / runtime-lineage changes what comparisons are credible

`observer-controller split` 应作为统领这两个 claim 的 framing 句出现，而不是再单列一个平行 headline。这样最稳，也最不容易显得 claim 过满。

## 最终一句话结论

最终应固定为 `2` 个 serious execution candidates：`Benefit-Bucket / Recoverability Analysis` 与 `Runtime-Lineage / Honest-Compute Protocol`；planning 顺序基本合理，但必须把 `seed_sensitivity_spotcheck.json` 明确写进 proposal deliverables，并坚持 `先 artifact、后 framing、再收 abstract` 的最终结构。
