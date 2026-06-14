# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-13 19:52:38 AEDT
**模型**: Codex (GPT-5)

## 评审意见

### 总评

这轮 `idea_debate` 的主方向是对的，而且明显比上一轮“继续救 `MGCD/DSG`”更诚实、更受证据约束。`idea/proposal.md`、`idea/hypotheses.md` 与 `idea/candidates.json` 已经把叙事重心从 “richer controller” 改写为 “repair object 是否选错” 与 “entropy 是否更适合做 routing/stopping”，这一步是必要且正确的。六个 perspective 也基本在两点上收敛：`cand_mgcd` / `cand_dsg` 应退出 serious pool；下一轮必须先做更硬的 falsification，而不是再做 front-runner 包装。

但我不会给这份 synthesis 很高分，因为它现在更像是一个“候选家族组合计划”，还不是一个已经冻结边界、能清楚被否证的实验对象。最突出的问题不是方向错，而是 **`cand_bsr` 仍然太像一个 family name，而不是一个 canonical method**；如果这一点不先收紧，下一轮很容易再次发生 front-runner drift，最后即使跑出正结果，也很难说清楚到底是哪一个机制起作用。

另外，我没有读到单独 materialize 的 `idea/debate/` critique 文件；`idea/candidates.json` 也明确写了 “idea/debate critique files were not separately materialized”。这不会阻止我继续评审，但它意味着目前的 synthesis traceability 不完整，很多 objection/resolution 仍然埋在六份 perspective memo 里，而不是显式沉淀为 debate ledger。

我的独立评分是：**7.4/10**。  
理由不是“这个方向不值得做”，而是“这个方向值得做，但还没到 reviewer-safe 的方法定义状态”。

### 这轮真正做对了什么

- **接受了负证据。** `idea/proposal.md` 明确把 `cand_mgcd` 与 `cand_dsg` 降级为 archived negative controls，这和六个 perspective 的共识一致，也符合本轮 `PIVOT` 的精神。
- **把 runtime parity 上升为硬约束。** `idea/hypotheses.md` 和 `idea/proposal.md` 都把 compile/backend parity、max-safe batch probing、matched compute 写成硬门槛，这对避免“工程条件伪装成方法收益”非常关键。
- **把质量线与速度线拆开。** `cand_bsr` 作为 quality-first 对象、`cand_espd` 作为 speed-first 对象，比把所有目标都塞进一个 controller 里更干净，也更容易归因。
- **相关工作脚手架明显改善。** `context/literature.md` 已经能支撑 reviewer-facing framing：DLM test-time scaling、calibration/failure prediction caution、sampler/runtime attribution，这让这轮 proposal 至少不再像“只是在内部循环想点子”。

### 被忽略或低估的风险

1. **`cand_bsr` 仍然定义过宽，存在再次漂移的高风险。**  
   `idea/proposal.md` 直接把 `SIR / SGR / BSR / CBR` 合并为一个统一对象，这在 synthesis 阶段可以理解，但在 experiment 阶段会变成危险信号：island detection、span aggregation、boundary locking、accept/reject rule、revision budget 这几个模块都可能被来回调整。若不先冻结 `cand_bsr_v1`，下一轮任何正结果都可能是 “family search” 而不是 “repair object hypothesis” 的证据。

2. **`cand_espd` 很容易退化成系统优化故事，而不是研究贡献。**  
   `pragmatist.md` 明确把 runtime floor、缓存、parallel decoding、`dInfer` 一类系统路径看得很重；`context/literature.md` 也说明 speed-up 生态已经很拥挤。如果下一轮 `cand_espd` 的收益主要来自 compile、cache、frontier shrink 的工程实现，而不是 entropy-routed compute 的机制，那最后得到的会是一个不错的系统优化结果，但不是当前 proposal 宣称的科学问题答案。

3. **当前晋级路径仍然偏 GSM8K 单 slice 驱动，存在任务和切片过拟合风险。**  
   现在的 gate 是：先看 `GSM8K audited slice`，通过后才进 MBPP / UGR。这个设计适合快速 kill，但也会让 narrative 继续围绕一个 audited slice 旋转。若新候选刚好对这 100 个样本有效，却不具备跨任务或跨 slice 稳定性，proposal 很可能再次过早宣布“object/routing 成立”。

4. **显式 debate ledger 缺失，会削弱后续写作的可追溯性。**  
   我能从 perspectives 里看出 objection 的存在，但看不到独立 debate artifact。对内部协作这可能还能忍，对后续 paper/response 阶段则会成为问题，因为你们很难快速回答“为什么放弃 A、为什么保留 B、哪条 objection 被哪条证据驳回”。

### 方法论问题

1. **当前 baseline 还不足以支撑 “repair object 改对了” 这个核心 claim。**  
   `RAND-84` / `CARD-84` 是必要对照，但它们不足以区分以下三个来源：
   - 是 entropy island detection 有效；
   - 还是 span aggregation 有效；
   - 还是 boundary locking 在减少 stable-region harm。  
   如果只拿 `cand_bsr` 对 `RAND-84` / `CARD-84`，即使赢了，也只能证明“这个 bundle 有效”，还不能证明 “repair object hypothesis” 本身成立。

2. **不同 candidate 的主评估轴不完全统一，仍有 cherry-picking 空间。**  
   `cand_bsr` 主要强调 matched compute 下的质量提升；`cand_espd` 主要强调 matched wall-clock 或 matched budget 下的速度收益。这个区分在研究设计上有道理，但如果没有统一的 2D 报表，最终很容易出现：质量线选对自己有利的轴，速度线也选对自己有利的轴，两个候选都“看起来赢了”，但没有一个结果能被公平放在同一比较框架里。

3. **`cand_ugr` 与 `syntax_guard` 的 hidden compute 约束还不够硬。**  
   proposal 里写了 “cheap uplift estimator” 和 “syntax guard”，但没有真正冻结额外 compute ledger。只要 acceptance estimator 多一次外部打分、syntax guard 多若干 rejection/retry，最后就有可能把方法收益和辅助计算混在一起。

4. **`n>=100` 比 `n=50` 好，但对 small-gain 问题仍然偏薄。**  
   这轮已经知道 DLM 是 small-gain regime，且本轮关键改写就是避免把微弱结果讲成大突破。在这种背景下，`n>=100` 更适合 screening，不足以稳妥支持“机制成立”的判断。若没有 design/test split、种子方差和置信区间，下一轮仍可能因为 sample noise 误判。

5. **MBPP 上的 `syntax_guard` 现在被写成 ablation，但它很可能本身就是主要效应来源。**  
   如果 `cand_bsr + syntax_guard` 好于 plain `cand_bsr`，你们需要回答：提升到底来自 local repair，还是来自 generic structural constraint？如果没有 `syntax_guard-only` baseline，MBPP 线很容易把科学问题混成工程约束问题。

### 我认为下一轮必须加上的具体修订

1. **冻结一个 `cand_bsr_v1`，不要再用 family 叙事直接进 pilot。**  
   至少把下面几项写死：
   - island score 的定义；
   - span merge rule；
   - boundary lock width；
   - 最大 revision step 数；
   - 接受/拒绝 revision 的最小规则。  
   任何新模块都不要悄悄塞进 `cand_bsr`，而应升级为 `cand_bsr_v2` 或单独 ablation。

2. **补一组真正能拆机制的 sham controls。**  
   我建议最少加以下对照：
   - `RandSpan-84`: 触碰 token 数与 span 长度分布匹配 `cand_bsr`，但 span 随机；
   - `EntropySpan-NoBoundary`: 有 entropy island，但去掉 boundary protection；
   - `BoundaryLock-RandomSpan`: 有 boundary lock，但 span 不是 uncertainty-driven；
   - `ESPD-RandomFrontier` 或 `FixedFrontier`: active frontier ratio 匹配，但 routing 不看 entropy；
   - `SyntaxGuard-only`: 在 baseline 上单独加 guard。  
   这组控制能把 “检测有效” 和 “干预对象有效” 拆开。

3. **把下一轮结果统一报成两张主表，而不是每条线各报一套最有利指标。**  
   我建议每个 serious candidate 都必须报告：
   - `quality @ equal compute`
   - `speed @ equal quality band`
   - `repair_count / harm_count / harmed_stable_tokens`
   - 代码任务上的 `syntax-valid / exec-valid / pass@1`  
   这样 `cand_bsr` 和 `cand_espd` 才能放进同一个公平框架里。

4. **把 screening slice 明确拆成设计集和确认集。**  
   如果这轮只能拿到 `n=100`，那我建议：
   - 用前 `50` 做 threshold/rule freeze；
   - 用后 `50` 做一次性 confirm；  
   更理想的是 `100 + hold-out 100`。不然下一轮仍然容易在 audited slice 上做隐性调参。

5. **给所有 auxiliary logic 设置硬预算上限。**  
   对 `cand_ugr`、`syntax_guard`、任何 side model 或 verifier，至少写死：
   - 模型上限；
   - 额外前向次数上限；
   - wall-clock overhead 上限；
   - 必须单独记账的日志字段。  
   否则 “cheap” 只是叙事标签，不是方法约束。

6. **为 `cand_ugr` 增加 uplift calibration 检验，不要只看最终 accuracy。**  
   如果它声称自己是 benefit estimator，那么下一轮必须检查：
   - predicted uplift 与 observed net benefit 是否单调对应；
   - 是否真的提升了 `repair/harm ratio`；
   - 是否只是更保守、少做事而看起来“更稳”。  
   若做不到这三点，`cand_ugr` 应立即停止消耗主线预算。

7. **单独 materialize 一份 debate ledger。**  
   建议新增类似 `idea/debate_summary.md` 的文件，至少包含：
   - 被保留的 objections；
   - 被驳回的 objections；
   - 每条 objection 对应的证据来源；
   - 最终为什么选 `cand_bsr / cand_espd / cand_ugr`。  
   这会大幅提升下一轮 synthesis、writing 和 rebuttal 的可追溯性。

8. **把 novelty boundary 再收紧一层。**  
   结合 `context/literature.md`，建议在 proposal 里加 3-4 句非常明确的话：
   - 你们不是在提出新的 global controller；
   - 不是在重复 Prism/MetaState 一类 trained or search-heavy test-time scaling；
   - 不是把 sampler/system gain 误写成方法 gain；
   - 你们真正要检验的是 `repair object` 与 `routing signal` 两个更窄的机制假设。  
   这会显著减少 reviewer 把工作误读成“另一种 heuristic bundle”的风险。

### 我的建议性决策

我支持继续推进，但只支持以下版本：

- `cand_bsr`：下一轮 **唯一 quality front-runner**，前提是先冻结成 `v1`；
- `cand_espd`：并行的 **speed line**，不要把它写成 `cand_bsr` 失败后的救火备胎；
- `cand_ugr`：严格条件触发，且在 `cand_bsr` 或 `cand_espd` 出现真实信号前，不要给它 narrative budget。

我不建议下一轮过早宣称以下结论：

- “repair object hypothesis 已成立”
- “entropy-routing 已成为正式贡献”
- “code robustness 已被证明改善”

这些都还是下一轮应该被验证的对象，不是当前 synthesis 已经可以宣布的结果。

## 评分

**7.4/10**
