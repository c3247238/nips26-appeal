# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-11
**模型**: Codex (GPT-5.4 high)
**轮次**: Round 3 (post-debate structural reorientation)

## 评审意见

### Overall Verdict

This is an intellectually ambitious proposal with some real originality, but in its current form it overstates both the strength of the evidence and the probability of a top-tier outcome. My independent read is that the proposal is strongest as a sharply scoped study of inference-time compute allocation in MDLMs, and weakest when it tries to simultaneously be an algorithm paper, a theory paper, a test-time-learning paper, and a benchmarking paper. Right now, the center of gravity is promising-but-fragile, not NeurIPS/ICML-ready.

### Most Important Concerns

1. The proposal is being built on very weak positive empirical signals. `+5pp` with `p=0.182`, `+12.5pp` on `n=16`, and repeated negative/null results on cross-step memory do not yet support a confident multi-track expansion.
2. The proposal has strong anti-overfitting language, but the process described still has a "many shots on goal" problem. After `22+ iterations`, thresholds like `p<0.10`, `gate > 0.10`, and `>=2% absolute` need a clearer statistical foundation or they risk looking post hoc.
3. The key conceptual premise, that MDLMs contain exploitable "information islands" that dense denoising misses, is still a hypothesis rather than an established phenomenon.
4. The theory agenda is too broad relative to the empirical uncertainty. Several proposed results sound elegant, but not obviously necessary, tight, or publishable if the empirical signal remains modest.

### 1. Scientific Rigor: 6/10

The proposal is stronger than average on explicit falsification: the hard kill criteria are good, and the insistence on diagnostics before fully committing to DaL is scientifically healthy. That said, the statistical foundation is still shaky.

Specific issues:
- The current pilot evidence is mostly suggestive, not persuasive. `Info-Gain Soft +5pp on GSM8K-100 (p=0.182)` is a useful hint, not evidence of effect.
- Small-sample wins like `A-CFG +12.5pp on GSM8K-16` are too noisy to support design decisions.
- `22+ iterations` with no significant positive result on cross-step memory should be treated as substantial negative evidence, not merely "inconclusive."
- Using `p<0.10` as a go/no-go threshold can be acceptable for exploratory work, but only if paired with preregistration, confidence intervals, effect-size targets, and correction for multiple comparisons.

The proposal would be much more rigorous if it defined:
- One primary hypothesis per track.
- One primary metric per track.
- One fixed evaluation budget.
- One correction plan for multiple testing.

### 2. Novelty and Contribution: 7/10

There is real novelty here, but it is uneven across tracks.

What looks genuinely interesting:
- The "extrinsic vs endogenous information" framing could be valuable if it is operationalized, not just narrated.
- IGGD is potentially novel if it truly uses conditional mutual information to guide unmasking in a way that beats simpler uncertainty-based or heuristic schedules under equal compute.

What is less clearly novel:
- DaL feels more like a synthesis of existing test-time adaptation / TTT ideas applied to DLMs than a fundamentally new concept.
- COBA is useful, but as stated it sounds more like an evaluation framework than a main technical contribution.

The biggest novelty risk is that IGGD may end up reading as a specialized version of existing themes:
- uncertainty-guided decoding,
- adaptive compute allocation,
- information-gain acquisition,
- iterative refinement scheduling.

So the contribution is not yet "obviously new"; it is "potentially new if sharply distinguished from adjacent work."

### 3. Experimental Design: 5/10

The phased structure is sensible. The execution plan is not yet strong enough.

What works:
- Diagnostics-first thinking is correct.
- Kill criteria are better than the usual vague "we will pivot if needed."
- COBA as a compute-normalized evaluation layer is a smart hedge.

What worries me:
- The proposal seems overly anchored to GSM8K-style reasoning. That is too narrow for claims about inference scaling in MDLMs.
- The thresholds look only partly calibrated. `gate > 0.10` and `r(SSL, accuracy) > 0.3` feel heuristic rather than justified by prior distributions or power analysis.
- There is not enough emphasis on strict compute fairness. FLOPs, wall-clock latency, number of denoising steps, and memory overhead should all be reported.
- The plan needs a truly untouched holdout set of tasks and models. Otherwise the large D x W grid becomes a tuning machine.

Blind spots:
- Generalization across model sizes.
- Generalization across task types, not just math reasoning.
- Sensitivity to estimator noise in conditional MI.
- Whether gains come from better ordering or simply "more selective extra compute."

### 4. Risk Management: 5/10

The dual-track plus COBA strategy is sensible, but the confidence estimates are too optimistic.

What is credible:
- IGGD as a training-free line is a good low-cost bet.
- DaL being conditional on diagnostics is exactly the right stance.
- COBA does provide a fallback artifact if the methods are mixed or negative.

What is not credible enough:
- `65% publishable probability` for COBA-only seems high unless COBA becomes a broadly reusable benchmark/measurement framework with unusually strong empirical coverage.
- `Aggregate P(publishable) >= 80%` is, in my view, materially inflated.

The hidden problem is correlated failure:
- If MDLMs do not actually exhibit large exploitable information asymmetries, IGGD weakens.
- If SSL improvement is structurally misaligned with downstream accuracy, DaL weakens.
- If dense denoising is already near-Pareto-optimal, both weaken.
- COBA then risks becoming a well-executed negative-results paper, which is harder to place at NeurIPS/ICML than the proposal implies.

### 5. Theoretical Depth: 5/10

The theory ideas are smart, but currently too diffuse.

Best candidates:
- A formal quantity like `Delta_I`, if it is estimable and predictive.
- Approximation guarantees for greedy/adaptive unmasking under approximate submodularity or bounded interaction assumptions.
- A separation result showing when lightweight TTT modules cannot express useful cross-step memory.

Weaker candidates:
- Generic OCO regret bounds that do not tightly match the actual decoding/control problem.
- Lyapunov-style stability analysis unless instability is empirically central.
- Broad Fisher-information analysis without a clean bridge to the algorithmic decision rule.

My concern is not lack of sophistication. It is lack of focus. A top-venue paper will benefit more from one tight theorem connected to one operational quantity than from five partially relevant formal threads.

### 6. Weaknesses and Blind Spots: 4/10

This is the area where the proposal is least convincing.

The biggest unaddressed questions are:

1. **Is the core phenomenon real enough to matter?**
The proposal still has not directly established that MDLM denoising trajectories contain large, exploitable heterogeneity in information value across mask positions and steps.

2. **Is the SSL-task mismatch merely empirical, or structural?**
The DaL pilots already suggest the latter. `SSL loss -52.7%` with `accuracy -1.0pp` and a gate stuck at `0.007` is a serious warning, not a minor tuning issue.

3. **Is the proposal too multi-paper to succeed as one paper?**
IGGD, DaL, COBA, and the theory stack could each be their own work. Combined, they risk diluting the main claim.

4. **Has negative evidence been given enough weight?**
`ReMDM -9pp`, `MetaState-GRU 43.75% vs 50.0%`, and repeated cross-step nulls should induce stronger pruning of the search space.

5. **Is venue fit being assessed realistically?**
A sophisticated diagnostic/negative-result paper may be valuable, but it is not automatically NeurIPS/ICML main-track material.

### 7. Actionable Recommendations

The good news is that the proposal can be improved a lot with a few concrete changes.

1. Make **one** claim central:
"Can inference-time compute in MDLMs be reallocated more effectively than dense denoising under equal budget?"

2. Treat **D0c/D0d as absolute gates**, not soft guidance.
No more DaL expansion until the proposal shows that SSL or diagnostic signals predict downstream gains on held-out instances.

3. Pre-register a minimal decisive evaluation:
- 2 MDLM families
- 3 task families
- fixed compute budgets
- 1 primary metric
- 1 primary comparison
- 1 untouched holdout benchmark

4. Strengthen baselines for IGGD:
- random order
- confidence/entropy/margin-based schedules
- stepwise heuristic schedules
- dense denoising under matched budget
- any prior info-gain / uncertainty-guided iterative refinement baseline that is even remotely adjacent

5. Narrow the theory:
- choose one formal object (`Delta_I` or an adaptive-submodularity surrogate),
- one theorem family,
- one falsifiable empirical prediction derived from it.

6. Reframe COBA realistically:
- as a robust measurement artifact and ablation framework,
- not as the "safety net" that guarantees a top-tier paper.

7. Add a hard paper-splitting option now:
- Paper A: IGGD + theory
- Paper B: COBA + diagnostic study of why TTT fails/succeeds in MDLMs

That split may increase actual publication probability more than forcing everything into one story.

## 评分

**6/10**

This is a serious and thoughtful proposal with above-average creativity, better-than-average self-critique, and a credible high-upside track in IGGD. But the current draft is still too optimistic relative to the evidence, too broad relative to the signal, and too willing to treat exploratory pilot results as support for a top-venue program. If narrowed aggressively, with preregistered diagnostics and stronger baseline discipline, it could become a strong paper. If kept in its current expansive form, it is more likely to produce an interesting partial result or a solid workshop/findings paper than a NeurIPS/ICML main-track hit.

---

## 历史记录

### Round 2 评审 (2026-03-11, GPT-5.4 high) — 4/10

Round 2 主要批评：
- Gate repair 作为 root cause fix 因果论证不够有说服力
- P3 失败 + MetaState-GRU < vanilla 增加负面先验
- 需要 causal falsification suite（forced gate, sham updates）
- D0c 应使用 partial/nonlinear analyses + prospective validation
- 建议 time-box DaL，将 Alternative A+D 移到 portfolio 中心

Round 3 评分上升 2 分（4→6），原因：
- 结构性重新定位：从 DaL-first 转向 dual-track，IGGD 作为 co-equal track
- 采纳了 D0c/D0d 作为硬性前置条件
- "extrinsic vs endogenous" 理论框架提供了统一解释
- 杀死条件更加严格（>=2% absolute at p<0.10）
- 但仍然过于乐观（P(publishable) >= 80% 被高估）、scope 过广、pilot 证据过弱

### Round 1 评审 (2026-03-10, GPT-5.4 high) — 5/10

Round 1 主要批评已被后续修订部分采纳：
- Compute fairness 改用 FLOPs（已采纳）
- Revealed tokens 是 pseudo-labels 的风险（已确认，但未完全解决）
- 缺少 baselines（部分补充）
- Hypotheses 过多（已精简）
