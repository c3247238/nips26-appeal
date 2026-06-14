# Result Debate Verdict

## Score

**7.8 / 10**

## One-Line Conclusion

继续推进 `cand_negative_audit_pivot`，把 iteration 3 明确写成 **audited negative case study**；不再重开正向 controller 主线，也不再默认追加 full-cycle 实验。

## What We Know

1. `CARD-84` 相对 compute-matched `DNB-84` 在 GSM8K 上有 localized signal（`net_repaired = +7`）。
2. 该信号没有和 sham control `RAND-84` cleanly separate（`net_repaired = +1`）。
3. 当前 current-only artifact bundle 已 joinable、可复核、可直接支撑写作。
4. entropy 最多只能写成 `risk marker`，不能写成 validated targeting rule。

## Biggest Risk

写作阶段重新滑回“method almost works / CARD-84 is a winner / entropy-guided targeting works”这类超出 claim ceiling 的话术。

## P0

**进入写作与 reviewer-defense 阶段，并把 `claim_scope_map.json` 作为硬约束执行。**

## Immediate Action Plan

1. 以 `negative case -> minimal audit template -> implications` 的顺序进入 outline / section writing。
2. 用 `figure_specs.md` 固定图表顺序，让 `CARD-84 > DNB-84 but CARD-84 ≈ RAND-84` 成为读者最先看到的信息。
3. 用 `validation_report.json` 和 `claim_scope_map.json` 主动披露 audited-slice scope 与 forbidden claims。
4. 只把 `followup_gaps.md` 中允许的项目保留为 future work，明确禁止把 future work 写成“差一点成功”。
