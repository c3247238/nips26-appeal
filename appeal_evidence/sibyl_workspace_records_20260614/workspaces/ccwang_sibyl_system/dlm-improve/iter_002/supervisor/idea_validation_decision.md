# Idea Validation Decision

- Decision: `ADVANCE`
- Selected candidate: `cand_bucket`
- Confidence: `0.93`

## 为什么

- `cand_bucket` 已经完成本轮最关键的主证据闭环：`coverage=100%`，`fixed=7`，`harmed=4`，`no_effect=89`，并且 aggregate `+3pp` 与 bucket 净效应一致。
- `cand_protocol` 已经完成，但它的角色是 support lane，不应与主证据层并列争夺唯一前进候选。
- `cand_observer_split` 现在只能保留为 framing；如果把它重新升级成 execution candidate，会把论文拉回 hero-method 叙事。
- `cand_minimal_scope` 与 `cand_boundary_audit` 作为备选还成立，但当前没有必要替换主线。

## 下一步

- 保持 `cand_bucket` 为主证据层，`cand_protocol` 作为支撑协议层。
- 本轮不再新增 full experiment batch；直接进入结果综合与写作收口。
- 写作时只保留三条安全主张：bucket-level outcomes、runtime-lineage auditability、minimal seed sign consistency。
