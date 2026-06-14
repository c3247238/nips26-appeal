# Empiricist Round 1 Summary

## 总判断

从实验主义者视角，当前 candidate pool 应该继续围绕 `diagnostic / protocol / evidence closure` 收缩，而不是重新扩成 generic method paper。六份稿件里分歧其实不大，真正的差别主要在优先级排序，而不是方向本身。

我建议 round1 之后只保留下面 3 个 candidate pool；如果必须进一步压缩到 2 个，就先保留前两个。

---

## Pool 1：Bucket-Mechanism Closure

### 是否保留

KEEP

### 为什么必须保留

- 这是当前最直接的质量门 blocker。
- 没有 bucket，diagnostic thesis 仍然只是 aggregate story。
- 这也是把 GSM8K reasoning 结果与 HumanEval boundary 结果连成统一机制叙事的最短路径。

### 最小 artifact 条件

- `exp/results/benefit_bucket_audit.json`
- `exp/results/benefit_bucket_examples.json`

### artifact 至少要包含

- pairwise comparison 定义
- fixed / harmed / no-effect 三类计数与比例
- 每类的平均 `tokens_changed`
- 每类的平均 signal 值
- 每类 3-5 个代表样本
- draft / revised correctness 对齐

### 若没有这些 artifact，就不能成立的 claim

- “我们理解 revision 何时帮助、何时伤害”
- “failure taxonomy 已经形成”
- “observer/controller split 具有机制解释”

---

## Pool 2：Minimal Robustness Closure

### 是否保留

KEEP

### 为什么必须保留

- 当前 headline 结果仍主要由单 seed、小切片支撑。
- 不需要做完整多 seed，只需要最小 spot-check 来挡住 reviewer 的第一轮质疑。
- 这是成本最低、但对可信度提升最直接的补件。

### 最小 artifact 条件

- `exp/results/seed_sensitivity_spotcheck.json`
- 可选：`exp/results/seed_sensitivity_examples.json`

### artifact 至少要包含

- 2-3 个 seed 的 headline pairwise comparison
- 每个方法每个 seed 的 accuracy
- pairwise delta 的 sign consistency
- delta 的 min / max / mean
- 哪些样本在不同 seed 下最不稳定

### 若没有这些 artifact，就不能成立的 claim

- “当前 headline ordering 足够稳定，可以作为 scoped diagnostic evidence”
- “Entropy/TIGER 的关系不是 seed=42 偶然现象”

---

## Pool 3：Protocol / Artifact Closure

### 是否保留

KEEP

### 为什么值得保留

- 如果论文要站稳在 `compute-normalized diagnostic study`，就必须把 artifact lineage 做成 reviewer-friendly package。
- 这一池不会增加 headline 分数，但会显著提升可复现性和审稿可检查性。
- 它与 Pool 1/2 完全兼容，且是最终收口必需件。

### 最小 artifact 条件

- `exp/results/canonical_asset_manifest.json`
- `exp/results/runtime_fairness_matrix.json`
- `exp/results/observer_controller_protocol.json`

### artifact 至少要包含

- claim 到 source asset 的一一映射
- figure/table 到 JSON 的一一映射
- nominal NFE / actual NFE / latency / throughput / batch / backend / compile 的统一 fairness 表
- `d(s)` / `g(s)` 的定义、来源、解释范围

### 若没有这些 artifact，就不能成立的 claim

- “honest compute 是 protocol contribution，而不只是 reporting nicety”
- “observer quality 与 controller gain 已被清楚地区分”
- “paper package 足够可审计”

---

## 我不建议进入主池的方向

### 1. Minimal Controller for Decoupling

暂不进入主池，只能作为 add-on。

原因：

- 它最容易重新滑回 method paper。
- 在 bucket / seed / manifest 没补齐前，它的边际价值低于上述三池。
- 如果做，也必须严格限定为 decoupling probe，而不是 headline method。

### 2. 任何新的 generic controller / scheduler 方向

DROP

原因：

- 与当前 evidence base 不匹配。
- 会稀释已经收紧好的 diagnostic framing。
- 会引入新的 fairness confound 与叙事负担。

---

## round1 后的建议 candidate pool

### 如果保留 3 个

1. `bucket_mechanism_closure`
2. `minimal_robustness_closure`
3. `protocol_artifact_closure`

### 如果只保留 2 个

1. `bucket_mechanism_closure`
2. `minimal_robustness_closure`

`protocol_artifact_closure` 仍要做，但可以视作 supporting lane，而不是单独 headline lane。

---

## 一句话结论

round1 之后最合理的 pool 不是“新方法 + 新 benchmark”，而是 `bucket + seed + protocol artifact` 三件套；没有这三类最小 artifact，当前 diagnostic paper 仍然难过质量门。

