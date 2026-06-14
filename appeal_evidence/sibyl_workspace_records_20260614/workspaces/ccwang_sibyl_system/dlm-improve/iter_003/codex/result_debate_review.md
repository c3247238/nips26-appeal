# Codex 独立评审 - result_debate

**评审时间**: 2026-03-12  
**评审角色**: `sibyl-codex-reviewer`  
**审查对象**: `current/idea/result_debate/synthesis.md`、`verdict.md`，并对照六份角色分析、`proposal.md` 与 packaging 资产

## 总体判断

当前这版 `result_debate` 已经明显站在正确轨道上了：它没有再把 `CARD-84` 往正向 controller 论文上拽，而是把核心价值收束到 **audited negative case + minimal audit template**。这与 `proposal.md`、`idea_validation_decision.json` 和 `claim_scope_map.json` 是一致的。

我给出的总体结论是：

- **方向正确，可以继续推进写作**
- **主结论基本与证据匹配**
- **仍需持续防止措辞回弹到 method-forward 叙事**

## Findings

### 1. `synthesis.md` 的结论边界基本安全

优点是它已经明确写出：

1. `CARD-84` 相对 `DNB-84` 只有 localized signal
2. 该 signal 没有和 `RAND-84` cleanly separate
3. entropy 只能写成 `risk marker`
4. stronger sham control 改写解释

这四条与 `claim_scope_map.json` 的 allowed / forbidden claims 是对齐的，没有出现明显越界。

### 2. `verdict.md` 的 P0 设置是合理的

把 P0 设成“进入写作与 reviewer-defense 阶段，并把 `claim_scope_map.json` 作为硬约束执行”，是符合当前控制面状态的。因为本轮 full cycle 已在 `task_plan.json` 中被显式标记为 `skipped_intentionally`，继续等待 GPU 或默认重开实验都会重新引入无谓漂移。

### 3. 仍需警惕两类回弹

第一类回弹是：

- 在摘要、引言或图题中把 `CARD-84 > DNB-84` 写成 headline，而把 `CARD-84 ≈ RAND-84` 写成后文限定条件

第二类回弹是：

- 把 future work 写成“更强 sham control 也许能证明 CARD-84 其实是对的”

这两类写法都会破坏当前 negative-case 的可信度。

## 建议

1. 主文所有涉及 `CARD-84` 的句子都应做一次 against-`claim_scope_map.json` 的逐句检查。
2. 图表顺序必须让 `RAND-84` 尽早出现，不能只把它放在 appendix。
3. `validation_report.json` 中的 audited-slice / current-only bundle 信息应主动进入 limitations 或 setup disclosure。
4. `followup_gaps.md` 的 forbidden moves 应直接转写进写作阶段的 internal checklist。

## 最终结论

这版 `result_debate` 资产已经足够支撑控制平面继续前进到 `experiment_decision` / writing stages。当前最重要的不是再补一个会动摇主线的新实验，而是守住已经对齐好的 negative-case framing，不让写作阶段重新滑回“方法差一点成功”的旧叙事。
