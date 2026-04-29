# Figure 6: Human Researcher Intuition vs Agent Trial-and-Error Harness

## 目的

直接支撑文章第一层逻辑：人类 researcher 的领域直觉来自多年试错；agent 缺少这种历史，因此必须通过 research harness 把大量受控试错转化为可复用经验。新版重点是加入中间桥梁：将感性的 research intuition 操作化为 `trial-to-behavior conversion`。

## 正文位置

Section 1 Introduction。建议放在开篇 1-1.5 页内，用作本文 framing 的直观图。

## 核心论证点

- 左侧：人类研究员通过长期深耕形成 tacit research intuition。
- 中间：本文的操作化转换：intuition = past trials changing future research behavior。
- 右侧：agent 通过 controlled trials、memory、reflection、guardrails、multi-agent critique 和 scheduler 形成外部化、可审计的 research intuition substrate。
- 这不是说 agent 已经拥有人类直觉，而是说 harness 是积累类似能力的必要条件。

## ASCII 草图

```text
Human Researcher                      Operational Bridge                  Agent + Harness
----------------                      ------------------                  ----------------
Years in one field                    research intuition                  many controlled trials
failed experiments        ----->      = past trials change      ----->    pilot/full gates
review feedback                       future behavior                     validation + trace
lab memory                            plan / experiment / claim           reflection + memory
metric intuition                      scheduler / writing                 multi-agent critique
compute habits                                                               scheduler + sanity checks

        |                                      |                                      |
        v                                      v                                      v
 internal tacit intuition          trial-to-behavior conversion        externalized auditable intuition
```

## 版式建议

```text
+-------------------------+     +-------------------------------+     +-----------------------------+
| Human Researcher        |     | Operationalization            |     | Agent + Research Harness    |
|                         |     |                               |     |                             |
| long-term projects      | --> | research intuition means:     | --> | controlled trial loop       |
| repeated failures       |     | past trial history changes    |     | reflection + memory         |
| advisor/reviewer input  |     | future research behavior      |     | evidence boundary           |
| lab notebooks           |     |                               |     | multi-agent critique        |
| tacit metric judgment   |     | plan / experiment / claim     |     | artifact trace              |
| compute experience      |     | validation / scheduler / text |     | scheduler + sanity checks   |
|                         |     |                               |     |                             |
| -> internal intuition   |     | -> measurable conversion      |     | -> externalized intuition   |
+-------------------------+     +-------------------------------+     +-----------------------------+
```

## 正式插图注意事项

- 避免把人类画成卡通角色；可以用简洁 silhouette 或只用信息框。
- 右侧不要画成 robot brain；重点是 harness components。
- 中间桥梁必须突出，因为这是用户希望保留但理性化的核心概念。
- 图注应明确：本文不是声称 agent 已获得人类直觉，而是主张必须通过 harness 化试错把直觉转化为可审计行为更新。
- 可用三栏结构，适合跨栏。

## 图像生成提示词

```text
Create a clean academic comparison diagram titled "Research Intuition as Trial-to-Behavior Conversion". Use three panels. Left panel: Human Researcher, with stacked labels years in field, failed experiments, review feedback, lab memory, tacit metric judgment, internal intuition. Middle panel: Operationalization, with the formula "research intuition = past trials changing future behavior" and small labels plan, experiment, claim, validation, scheduler, writing. Right panel: Agent + Research Harness, with controlled trials, reflection, institutional memory, evidence boundary, multi-agent critique, artifact trace, scheduler, externalized auditable intuition. Balanced three-column layout, vector-like, white background, high readability, no cartoon robots, no decorative gradients.
```
