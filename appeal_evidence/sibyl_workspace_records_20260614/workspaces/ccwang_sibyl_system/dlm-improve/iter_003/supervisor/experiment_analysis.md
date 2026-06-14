# 实验结果分析

## 实验结果概要

iteration 3 没有继续扩张 full experiment，而是把已经完成的 pilot evidence bundle 收口成一个 reviewer-safe 的负案例包。当前可用的核心资产包括：

1. `summary.md` 与 `decision.json` 已明确给出四个实验臂的主读数；
2. `validation_report.json` 确认当前 evidence bundle 是 current-only、自包含、可 join 的；
3. `claim_scope_map.json` 已经把 allowed / forbidden wording 冻结；
4. `main_results_table.csv`、`repair_harm_table.csv`、`harm_profile_table.csv` 与 `figure_specs.md` 已经完成 paper-ready packaging；
5. `result_debate` 六个视角与综合裁决已经收敛到同一条 negative-case 主线。

## 各方观点总结

- 乐观视角认为：局部 GSM8K 信号、artifact closure 和最小审计模板仍然构成正面价值。
- 怀疑视角认为：`CARD-84 ≈ RAND-84` 是决定性事实，必须彻底取消正向 controller claim。
- 方法学视角认为：本轮最大成功是 baseline / sham control / sample-level audit / claim ceiling 终于形成闭环。
- 战略视角认为：本轮最优动作是进入写作，而不是重新打开 full experiments。
- 比较分析视角认为：本文不在正向 DLM method frontier 上取胜，但可以填补 small-gain claim 的审计与解释空白。
- 修正主义视角认为：entropy 的地位必须从 controller rule 降为 observer-side risk marker。

## 分析

### 1. 方法可行性

如果问题是“当前 evidence 是否支持继续推进本轮论文对象”，答案是肯定的。  
如果问题是“是否还需要再开新实验才能继续”，答案是否定的。

### 2. 当前最可信的科学结论

当前最可信的结论不是 `CARD-84` 赢了，而是：

- `CARD-84 > DNB-84` 只说明局部信号存在；
- `CARD-84 ≈ RAND-84` 说明正向 controller claim 不成立；
- stronger sham control 会实质改写 small-gain interpretation；
- entropy 在当前证据下最多保留为 `risk marker`。

### 3. 是否需要继续实验

本轮不需要。原因有三点：

1. `task_plan.json` 已显式把 full cycle 标记为 `skipped_intentionally`；
2. `followup_gaps.md` 已经把允许的补强项隔离为 future work；
3. 新实验的边际收益已经低于写作与 framing 收益。

### 4. 最大风险

最大风险不是证据不够，而是写作重新滑回 forbidden wording，例如：

- `CARD-84 is the winning inference method`
- `entropy-guided revision reliably improves DLM reasoning`
- `the method almost works`

## 决策理由

我选择 `PROCEED`，因为：

1. 当前 negative-case paper object 已经在 proposal、planning、packaging、result debate 四层完全对齐；
2. 本轮的主要信息增益已经完成，继续实验只会破坏叙事纪律；
3. 最值钱的下一步是 outline、sections 和 integration，而不是再开新的 controller exploration。

## DECISION: PROCEED

DECISION: PROCEED
