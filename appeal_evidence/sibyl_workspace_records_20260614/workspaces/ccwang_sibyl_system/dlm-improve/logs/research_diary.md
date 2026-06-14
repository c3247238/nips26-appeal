

# Iteration 0

**Score**: 6.0/10
**Issues**: 17
**Trajectory**: stagnant

## Reflection
# 迭代 0 反思报告

## 本轮迭代总结

本轮是首次完整迭代，从文献调研到论文初稿全流程跑通。核心贡献——DLM 校准诊断（SC-ECE vs Oracle-ECE）和基于熵的修正解码（CARD）——已在全规模 GSM8K (n=1319) 上验证。6 个假设中 4 个得到验证或产生有价值的负面结果，pipeline 自动化成功。

但论文存在多个严重问题，当前评审分数 6.0/10 (REVISE)，距离 top venue 接受有实质性差距。

## 各类问题分析

### EXPERIMENT（实验）— 最严重

1. **单 seed 评估（CRITICAL）**：所有结果基于 seed=42，无方差估计。Abstract 中的 p<0.002 仅反映 binomial 噪声，不反映 DLM 随机解码的 seed-to-seed 变异性。+4.0% 的改善可能不稳健。
2. **单基准阳性结果（CRITICAL）**：6 个 model-benchmark 对中仅 1 个（LLaDA-8B/GSM8K）展现清晰 CARD 优势。HumanEval -10pp，MMLU 0 信号，MBPP 地板，Dream-7B 噪声。
3. **无竞品对比（CRITICAL）**：综述了 10+ 修正方法但实验中未对比任何一个。CORE 是最近竞争者。
4. **Pilot-only 消融（MAJOR）**：Table 3 在 n=100 上做出设计决策，但 entropy vs random revision 仅差 1pp，可能在 full-scale 逆转。
5. **数据一致性错误（CRITICAL）**：论文使用 pilot 校准数字（SC-ECE=0.220）而 full-scale 数据（SC-ECE=0.203）已可用。

### WRITING（写作）

1. **标题误导**：'Calibration-Aware' 暗示方法包含校准组件，但实际校准修正被证明无用，方法仅用 raw entropy。
2. **选择性报告**：HumanEval 负面结果（-10pp）从 Table 4 中被省略。
3. **占位符图表和空白附录**：图表未渲染，Appendix B/C 未填充。
4. **Entropy-Revise-64 命名矛盾**：名称暗示熵定

## Review Summary
revise The paper presents a genuinely novel calibration diagnostic for DLMs (SC-ECE vs Oracle-ECE) and a clean narrative from diagnosis to method design. However, serious data consistency issues (paper uses pilot n=100 calibration numbers when full-scale n=1319 data exists), single-seed evaluation, single-benchmark positive result, no competitor comparison, and unpopulated appendices collectively prevent acceptance at a top venue. The diagnostic contribution is near-publishable; the method evalu

## Critique Summary
CARD delivers a genuine but narrow contribution: a well-executed calibration diagnostic study and a simple entropy-revision method that wins +4.0 pp on GSM8K (LLaDA-8B) over compute-matched baselines. However, the empirical evidence is dangerously thin — single seed, single model with significant gains (GSM8K only), 63% pilot-to-full shrinkage, and outright failure on code generation. The paper's title says 'Calibration-Aware' but calibration awareness contributes zero measured benefit; the meth


# Iteration 1

**Score**: 6.4/10
**Issues**: 9
**Fixed**: 4
**Trajectory**: improving

## Reflection
# 迭代 1 反思报告

## 本轮迭代总结

这一轮最重要的进展不是又多做了一个方法，而是把论文真正纠正成了它应该成为的样子：从旧的 method-forward / CARD hero 叙事，转成了 compute-normalized diagnostic study。这个方向变化不是包装，而是证据驱动的。`final_pareto_synthesis.json`、`diag_compute_curve_gsm8k.json`、`diag_signal_gap_audit.json`、`diag_math500_shortlist.json` 和 `diag_humaneval_guard_boundary.json` 现在共同支持的是一个更诚实也更稳定的主张：honest compute 会改变关键比较，observer quality 不等于 controller gain，而 revision 对任务结构高度敏感。

同时，这一轮把写作链真正收口到了可运行状态。LaTeX 模板、图表复制、protocol figure、bibliography 和 PDF 编译已经完成，`writing_latex` 不再是半成品阶段。`review` 阶段也成功形成了 critic / supervisor / codex 三路独立审查，并给出了相对一致的判断：论文方向是对的，但证据封口还不够。

## 各类问题分析

### ANALYSIS

1. **benefit-bucket audit 仍缺失（HIGH）**  
   这是当前最实质的证据缺口。论文已经在平均结果层面说明“revision 有时帮助、有时伤害”，但还没有把 fixed / harmed / no-effect 样本分离出来，因此机制性主张仍停留在 aggregate-only 层面。

2. **pilot-to-current shrinkage narrative 仍未显式化（MEDIUM）**  
   外部读者仍看不到“为什么 earlier pilot story 更像 method win，而 current story 更像 diagnostic study”的完整解释。虽然内部已经做了正确战略转向，但正文还没把这件事说透。

3. **honest-compute fair

## Review Summary
revise The paper has a credible diagnostic reframing and the headline tables are numerically supported by the current result artifacts, but the quality gate is still blocked by aggregate-only evidence, underspecified signal-audit definitions, and incomplete uncertainty accounting. I used exp/results/final_pareto_synthesis.json as the representative summary because exp/results/summary.md is absent, and cross-checked the manuscript against gsm8k_main_shortlist.json, diag_signal_gap_audit.json, dia

## Critique Summary
The paper has found the right pivot, but the current draft still outruns its evidence on three fronts: the honest-compute result is narrower and more implementation-confounded than the prose suggests, the promised mechanism-level taxonomy is still missing, and the task-dependence story rests on single-seed small-slice evidence.


# Iteration 2

**Score**: 6.4/10
**Issues**: 8
**Fixed**: 4
**Trajectory**: improving

## Reflection
# 迭代 2 反思报告

## 本轮迭代总结

本轮质量轨迹是**小幅上升但仍未过门**。`logs/quality_trend.md` 记录的分数从 6.0 上升到 6.4，说明方向是对的，但 supervisor 仍给出 `revise`，核心原因已经从“有没有新证据”转成“证据包是否自包含、论述是否严格对齐证据边界”。

这一轮最有价值的进展是三类最小闭环证据都真正落地了：

- `exp/results/benefit_bucket_audit_pilot.json` 给出了 `7 fixed / 4 harmed / 89 no_effect` 的 bucket 分解；
- `exp/results/seed_sensitivity_spotcheck.json` 给出了三 seed 同向但幅度较小的 sign consistency；
- `exp/results/runtime_probe_iter2.json` 给出了当前 host 上的 runtime contract：`eager|compile=True`、safe batch size 57。

同时，这轮也做出了正确的**不做什么**的决定：`exp/results/min_controller_decoupling_probe.json` 明确把额外 controller probe 记为 `NO_GO`，避免为了“更像方法论文”而重新引入 story drift。

需要明确说明的是：`exp/results/summary.md` 在当前 workspace 中仍然缺失，因此本次反思无法按角色协议读取 canonical summary，只能改用代表性结果资产进行交叉核对。实际使用的代表性证据包括：

- `exp/results/runtime_probe_iter2.json`
- `exp/results/benefit_bucket_audit_pilot.json`
- `exp/results/seed_sensitivity_spotcheck.json`
- `exp/results/runtime_fairness_matrix.json`
- `exp/results/observer_controller_protocol.json`
- `exp/result

## Review Summary
revise 论文已经完成从 method-forward 叙事到 diagnostic/protocol 叙事的可信收缩，`paper.md` 中关于 GSM8K headline pair 的 `+3pp`、`7 fixed / 4 harmed / 89 no_effect` 和三种子 sign consistency 基本能被当前 workspace 资产支持；但 `exp/results/summary.md` 缺失，canonical lineage 资产在 `current` workspace 内不自洽，honest-compute 论证仍混杂异质 runtime 条件，因此还不适合通过质量门。

## Critique Summary
The workspace has a disciplined diagnostic pivot, but the current paper still overclaims relative to the evidence. The strongest supported contribution is a narrow GSM8K bucket audit plus honest-compute reporting; the weaker parts are the observer-controller thesis, the mixed runtime lineage bundle, and the portability/reproducibility story.


# Iteration 3

**Score**: 7.2/10
**Issues**: 5
**Fixed**: 4
**Trajectory**: improving

## Reflection
# Iteration 3 Reflection

## 总判断

这一轮最重要的进展不是“又找到一个更强方法”，而是把论文稳定收缩成一个可信的 negative-case submission package。`CARD-84` 没有再被包装成 winning controller；核心结论已经稳定成：

- `CARD-84 > DNB-84` 只能支持 active-control gain
- `CARD-84 ≈ RAND-84` 说明 attribution 不能落到 entropy-guided targeting
- 因而论文贡献应定位为 audited negative case + audit template，而不是新控制器

同时，LaTeX 链路、图、参考文献和 PDF 编译都已经打通，三路 review 也给出了高度一致的判断：论文现在是诚实、连贯、可推进的，但还没有达到强 submission-ready 状态。

## 本轮真正修复了什么

1. 写作链闭环了：`paper.md`、`review.md`、LaTeX 源和 `main.pdf` 已经对齐，不再停留在“内容大致完成但产物不完整”的状态。
2. 负面结果 framing 稳定了：没有再尝试把 `CARD-84` 救回正面方法故事，sham-control 结论被完整保留。
3. 审查机制可消费了：critic、supervisor、codex 三路审查已经沉淀为稳定产物，能够直接供 reflection 和 quality gate 使用。
4. 参考文献和模板不再是硬阻塞：NeurIPS 模板、BibTeX、图表复制与 PDF 编译已经跑通。

## 为什么还没过线

当前 supervisor score 是 `7.1/10`，说明问题已经从“方向不可信”降成“submission package 仍有几个 reviewer-facing 缺口”。这些缺口集中在四件事上：

1. **claim scope 仍需更硬地收口**：headline 仍容易被读成 broader DLM revision statement，而当前证据只足以支撑 `n=100 audited slice` 的 bounded case study。
2. **runtime lineage

## Review Summary
continue This is now a coherent and honest negative-case paper with a credible evidence bundle, but it still reads as a narrowly scoped audited slice study whose audit-template significance must be framed carefully. The manuscript is strong enough to continue through the pipeline, yet not broad or polished enough to be treated as complete without reflection-stage tightening.

## Critique Summary
The paper is now a coherent negative-case study, but it still depends on a narrow audited slice, thin reviewer-facing positioning, and incomplete reproducibility packaging. The core scientific risk is no longer overclaiming a controller win; it is that reviewers may still read the paper as a single-case narrative whose broader audit-template value is asserted more strongly than demonstrated.


# Iteration 4

**Score**: 7.6/10
**Issues**: 5
**Fixed**: 9
**Trajectory**: improving

## Reflection
# 本轮反思报告

## 本轮总结

这一轮最重要的变化不是再做出一个更大的 headline，而是把 iteration 4 的论文对象彻底收紧成一个 reviewer-safe 的 bounded contribution。`paper.md`、`review.md`、`writing/latex/main.tex` 和 `writing/latex/main.pdf` 现在已经在同一条叙事线上：`cand_espd` 的 full-scale GSM8K 结果支持一个关于 entropy-routed compute 的有限机制性结论，但不支持 broader benchmark-dominance claim。

质量上，本轮已经明显优于上一轮。终审从 6.8 提升到 7.3，supervisor review 给出 7.6，说明 submission-facing packaging gap 基本被补上。推动这一提升的关键动作不是新实验，而是三类 reviewer-facing 修补：`primary endpoint`、`claim boundary`、`runtime lineage / artifact-release`。

## 问题分类

### SYSTEM

- Feishu/Lark 同步仍被本地 OAuth refresh `invalid_grant` 阻塞，虽然 registry 已安全，但 pending queue 继续积累。

### EXPERIMENT

- 当前 full-scale 证据仍只覆盖 GSM8K；缺少外部 benchmark。
- `cand_espd` vs `ESPD-FixedFrontier` 的 sham 仍是 partially matched，机制解释仍未完全封口。

### WRITING

- reviewer-facing packaging gap 已大幅修复，但 proposal object-line front-runner 与 final paper speed-line mainline 的转向还可以再写得更显性。

### ANALYSIS

- strongest claim 依然是 bounded attribution signal，而不是 clean mechanis

## Review Summary
continue 当前 iteration 4 产物已经形成一个可信的 bounded contribution package：paper、review 和 LaTeX/PDF 基本对齐，claim boundary 明确，最强证据是 cand_espd 对 fixed-frontier sham 的 full-scale GSM8K separation。

## Critique Summary
当前 iteration 4 的 strongest evidence 确实来自 cand_espd 的 full-scale GSM8K bundle，但 paper、proposal、methodology 与 pilot history 之间仍存在明显 narrative drift。最严重的问题不是结果为负，而是对象线与速度线的主张边界、runtime contract 的一致性、以及 mechanism claim 的隔离程度还没有被整理成 reviewer 一眼可审的闭环。
