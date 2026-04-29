# Figure 4: Claim-Evidence Graph

## 目的

展示一个 paper claim 应如何连接到具体 artifacts 和 validation checks。该图支撑 Boundary Harness、Memory Harness 和 human audit burden 的评价指标。

## 正文位置

Section 4 Theory 或 Section 7 Evaluation Agenda。也可在 `sae-absorption-kimi` 案例后使用，说明为什么 narrative generation 必须发生在 validation 之后。

## 核心论证点

- Claim 不是孤立文本，而是 evidence graph 的上层节点。
- Negative evidence 和 contradiction links 是 first-class objects。
- Data validation 是 narrative generation 的前置条件。

## ASCII 草图

```text
Paper Claim
   | scope + maturity + last-updated
   v
Table / Figure  ---- contradiction links ---- Negative Evidence
   |
   v
Aggregated Result JSON
   |
   v
Analysis Script + Config
   |
   v
Raw Logs / Seeds / Checkpoints
   |
   v
Validation Tests
   | duplicate check / scale check / missing output check
   v
Claim Gate: allow / downgrade / block
```

## 与 harness components 的关系

- Boundary Harness：validation tests 和 claim gate 阻止坏证据进入叙事。
- Memory Harness：negative evidence 进入长期 memory，影响未来 planning 和 critique。
- Perspective Harness：critic/supervisor 可以沿 graph 审查 claim。
- Efficiency Harness：快速 validation 可能比新跑长实验更有 evidence value。

## 正式插图注意事项

- 使用带标签的 nodes：claim、figure/table、aggregate、script、raw logs、validation、negative evidence。
- 将 negative evidence 画成旁路节点，不要画成 footnote。
- 加入 metadata tags：maturity、scope、last updated、source path。
- `Claim Gate` 用三路输出表示：allow、downgrade、block。

## 图像生成提示词

```text
Create a clean systems-paper diagram titled "Claim-Evidence Graph". Top node: Paper Claim with metadata tags scope, maturity, last updated. Downward chain: Table/Figure, Aggregated Result JSON, Analysis Script + Config, Raw Logs / Seeds / Checkpoints, Validation Tests. Add a side node "Negative Evidence" connected to Table/Figure and Paper Claim with contradiction links. At the bottom add "Claim Gate: allow / downgrade / block". White background, vector-like, high contrast, no decorative elements.
```
