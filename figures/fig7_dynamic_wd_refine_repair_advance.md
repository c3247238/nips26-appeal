# Figure 7: Dynamic-WD Case Flow: REFINE -> Repair -> ADVANCE

## 目的

用一个积极主案例解释 Sibyl 如何把 trial signal 转化为未来科研行为更新。`dynamic-wd` 中，系统没有把不稳定 pilot 或过强 claim 写成完整论文，而是先 REFINE，插入 algorithm repair、tests、claim narrowing 和 control priorities；修复验证后再 ADVANCE。

## 正文位置

Section 6.1：`dynamic-wd` 深案例。该图可以作为正文最重要的 case flow 图。

## 核心论证点

- Sibyl 的优势不是“没有失败”，而是能够把 evidence challenge 转化为具体行动。
- REFINE 是积极科研行为：阻止 premature writing，并把问题落实为 repair tasks。
- Repair 后 ADVANCE 不是盲目前进，而是基于 tests、pilot evidence、remaining risks 和 scoped claims 的 calibrated advancement。
- 这张图直接支撑 trial-to-behavior conversion：trial signal 改变了 plan、algorithm、validation、claim 和 full-run readiness。

## ASCII 草图

```text
Trial signals / evidence challenges
   stability concern
   zero-budget collapse
   budget confound
   raw/paper mismatch
   weakened hypotheses
          |
          v
+------------------+
| Supervisor Gate  |
|     REFINE       |
+------------------+
          |
          v
Research behavior updates
   -> insert UDWDC-v2 stability fix
   -> floor clipping + EMA smoothing
   -> budget assertion + unit tests
   -> narrow universal claim to taxonomy
   -> prioritize controls / raw-log checks
          |
          v
Verification artifacts
   -> tests pass
   -> pilot metrics reviewed
   -> risks explicitly retained
          |
          v
+------------------+
| Supervisor Gate  |
|     ADVANCE      |
+------------------+
          |
          v
Full experiments with calibrated claims
```

## 图形版草图

```text
+------------------------------+     +-------------------------+     +------------------------------+
| Evidence Challenge           | --> | Sibyl Harness Response  | --> | Behavior Update              |
|                              |     |                         |     |                              |
| stability concern            |     | REFINE gate             |     | UDWDC-v2 repair task          |
| zero-budget collapse         |     | boundary + trace        |     | unit tests + budget assertion |
| budget confound              |     | resource planning       |     | controls prioritized          |
| raw/paper mismatch           |     | artifact audit          |     | raw-log cross-check           |
| overbroad framework claim    |     | supervisor calibration  |     | narrowed taxonomy claim       |
+------------------------------+     +-------------------------+     +------------------------------+
                                                                  |
                                                                  v
                                             +--------------------------------------+
                                             | Verified Repair + Explicit Risks     |
                                             | -> calibrated ADVANCE to full runs   |
                                             +--------------------------------------+
```

## 正式插图注意事项

- 左侧不要叫 failure；用 `Evidence Challenge` 或 `Trial Signal`。
- 中间突出 `REFINE` gate 和 `ADVANCE` gate。
- 右侧突出 behavior update，而不是只写 “lesson learned”。
- 不写未审计精确数字；可写 `tests pass`，具体 `9/9` 如需使用必须在正文前再次审计。
- 图注说明：该图展示 process evidence，不声称 UDWDC 的最终 domain result 已成立。

## 图像生成提示词

```text
Create a clean NeurIPS-style case-flow diagram titled "Dynamic-WD: Trial Signals to Research Behavior Updates". Left column labeled Trial Signals / Evidence Challenges with compact labels: stability concern, zero-budget collapse, budget confound, raw/paper mismatch, weakened hypotheses. Middle column labeled Sibyl Harness Response with a prominent REFINE gate, boundary + trace, resource planning, supervisor calibration. Right column labeled Behavior Updates with UDWDC-v2 repair task, floor clipping + EMA, budget assertion + unit tests, controls prioritized, raw-log cross-check, narrowed taxonomy claim. Then show a lower step labeled Verified Repair + Explicit Risks leading to an ADVANCE gate and "full experiments with calibrated claims". Vector-like, white background, high contrast, minimal color, no decorative gradients, no cartoon characters.
```
