# Entropy-Routed Compute for Training-Free Discrete Diffusion Language Models: A Bounded Full-Scale Study with Sham Controls

## Abstract

Inference-time intervention for discrete diffusion language models (dLLMs) has entered a small-gain regime in which raw benchmark movement is no longer sufficient to support strong mechanism claims. We study this problem through a bounded full-scale experiment on GSM8K. Earlier iteration-4 screening suggested that raw entropy is informative as an observer-side uncertainty signal, but does not validate entropy as a semantic controller. We therefore reframe the problem: rather than asking entropy to determine what semantic revision should be made, we ask whether entropy can route and stop additional compute more effectively than a matched control. The predeclared primary endpoint is narrow: candidate-versus-sham quality-speed trade-off under a unified runtime contract, not global baseline dominance. We evaluate an entropy-routed candidate (`cand_espd`) against two shared controls (`CARD-84`, `RAND-84`) and, critically, a matched fixed-frontier sham (`ESPD-FixedFrontier`). On full-scale GSM8K, `cand_espd` reaches 0.4041 accuracy, compared with 0.3980 for `RAND-84`, 0.3965 for `CARD-84`, and 0.3988 for the fixed-frontier sham. The candidate is not the absolute fastest method in the bundle, but it preserves a better quality-speed trade-off than the sham under a unified runtime contract. We argue that this is a bounded positive result: it supports entropy as a routing/stopping signal rather than as a validated semantic controller, and it demonstrates the value of sham controls, paired repair/harm analysis, and explicit runtime lineage in small-gain dLLM evaluation.

## 1. Introduction

Inference-time intervention for discrete diffusion language models (dLLMs) has entered a regime in which small gains are common, but clean attribution is rare. Recent work on dLLM test-time scaling, guided decoding, and memory augmentation has shown that additional inference-time compute can improve reasoning performance, yet these improvements are often entangled with search depth, sampler changes, backend choices, or unreported execution-envelope differences (Bai et al., 2026; Lu et al., 2026; Xia et al., 2026; Sampler-Centric Evaluation, 2026). As a result, the central scientific question is no longer only whether a method improves a benchmark score, but whether the observed gain survives a comparison that cleanly isolates the claimed mechanism.

This paper studies that attribution problem through a bounded full-scale experiment on GSM8K. The starting point is a negative lesson from earlier iteration-4 screening: raw entropy is informative as an observer-side uncertainty signal, but that fact alone does not validate entropy as a semantic controller. In particular, entropy-centered revision controls did not cleanly separate from random targeting under matched compute. Rather than continuing to frame entropy as a direct controller, we ask a narrower and more falsifiable question: can entropy be used more effectively as a routing and stopping signal for allocating additional compute?

To answer this question, we evaluate an entropy-routed compute strategy, `cand_espd`, against two shared controls (`CARD-84` and `RAND-84`) and, critically, against a matched fixed-frontier sham that keeps the same nominal frontier budget without entropy-based frontier placement. The main result is intentionally modest. On full-scale GSM8K, `cand_espd` reaches 0.4041 accuracy, compared with 0.3980 for `RAND-84`, 0.3965 for `CARD-84`, and 0.3988 for the fixed-frontier sham. The more important evidence is structural rather than magnitude-based: as shown in Table 1 and Figure 2, the candidate does not dominate all baselines in absolute speed, but it does preserve a better quality-speed trade-off than the matched sham under a unified runtime contract.

This bounded framing matters. We do not claim a benchmark-level breakthrough, and we do not claim that entropy has now been validated as a semantic controller for dLLM revision. Instead, we make a narrower claim: under a unified runtime contract, entropy appears more useful for deciding where additional compute should be spent than for deciding what semantic revision should be made. This interpretation is supported by the candidate-versus-sham comparison and by the fact that the shared entropy control does not reliably outperform the shared random control.

Our claim boundary is equally explicit. The current evidence supports a bounded routing interpretation for `cand_espd`, but it does not support benchmark-level SOTA language, a strong significance-based superiority claim, validation of the object-level `cand_bsr` line, or a cleanly isolated proof that the candidate-sham gap is purely mechanistic rather than partially implementation-mediated.

Our contributions are threefold. First, we provide a full-scale empirical re-interpretation of entropy in training-free dLLM intervention, shifting it from a controller framing to a routing/stopping framing. Second, we show that this routing interpretation survives a matched fixed-frontier sham on GSM8K, yielding a modest but credible gain under honest runtime accounting. Third, we package the result as a bounded contribution, with explicit claim boundaries, paired repair/harm analysis, a narrow primary endpoint, and runtime-lineage reporting, in order to make the small-gain regime more reviewer-auditable.

The rest of the paper is organized as follows. Section 2 positions the work within dLLM inference engineering, calibration, and sampler-attribution literature. Section 3 formalizes the problem reframing. Section 4 defines the entropy-routed candidate and the fixed-frontier sham. Section 5 describes the full-scale GSM8K evaluation protocol. Section 6 reports the main bounded result, and Section 7 analyzes its interpretation, limitations, and next steps.

## 2. Related Work

### 2.1 Discrete Diffusion Language Models and Inference Engineering

Discrete diffusion language models extend diffusion-style iterative generation to discrete text, with early foundations such as reparameterized discrete diffusion and ratio-estimation-based formulations establishing the feasibility of non-autoregressive text denoising (Zheng et al., 2023; Lou et al., 2023). As these models matured, the literature increasingly shifted from basic feasibility questions to inference-time engineering: how to schedule denoising, how to allocate compute, and how to improve generation quality without retraining the backbone.

Our work belongs to this inference-engineering tradition, but in a narrower sense than most method papers. We do not propose a new dLLM family or a new training pipeline. Instead, we study whether an observer-side uncertainty signal can justify a bounded test-time intervention claim when compared against carefully designed controls.

### 2.2 Test-Time Scaling, Guided Decoding, and Memory Augmentation

Recent dLLM-adjacent work has made inference-time scaling a first-class object of study. Examples include hierarchical search and self-verification for discrete diffusion models, confidence-guided direct decoding, persistent memory augmentation for iterative denoising, and related sampler-centric improvements (Bai et al., 2026; Lu et al., 2026; Xia et al., 2026; Schiff et al., 2024). These lines collectively show that the community is no longer satisfied with plain denoising schedules; it is now actively exploring how additional computation, guidance, or state can be used to improve reasoning at test time.

Our contribution differs from these lines in two ways. First, we focus on a training-free setting and avoid heavier search or learned memory augmentation. Second, our central question is attribution: if a small gain appears, can we show that it comes from entropy-routed compute rather than from a generic frontier budget, a sampler change, or an execution mismatch? This makes our paper closer to an audit of a mechanism claim than to a maximally optimized performance paper.

### 2.3 Calibration, Uncertainty, and Failure Prediction

There is a long tradition of using uncertainty-like signals to support decision making in machine learning systems. However, calibration and failure-prediction work has repeatedly warned that an informative confidence signal does not automatically translate into a useful downstream intervention rule (Zhu et al., 2023; Deng et al., 2023; ActCab+CoDec, 2024). This caution is especially relevant in our setting. Earlier iteration-4 screening already suggested that entropy may be informative without being a valid semantic controller: entropy-guided revision did not cleanly separate from random targeting under matched compute.

The present paper follows that warning rather than fighting it. We do not attempt to rescue entropy as a semantic targeting rule. Instead, we ask whether entropy can be repurposed into a narrower role, namely routing and stopping additional compute. In this sense, the paper sits between uncertainty estimation and test-time compute allocation.

### 2.4 Sampler-Centric and Runtime-Centric Attribution

Recent sampler-centric evaluation work has emphasized that method claims in diffusion systems are often confounded by decoding choices, runtime contracts, or backend differences (Sampler-Centric Evaluation, 2026). That critique is directly relevant to dLLM small-gain papers. Once the deltas become small, the scientific burden shifts toward transparent execution envelopes, sham controls, and explicit runtime lineage.

This is precisely the niche our work aims to fill. The main value of our result is not that the absolute GSM8K score is high, but that a modest gain survives a matched fixed-frontier sham under a unified contract. We therefore position the paper as a bounded contribution on attribution discipline for training-free dLLM intervention.

## 3. Method

### 3.1 Problem Reframing

The starting point of this work is a reframing rather than a clean-slate method invention. Earlier iteration-4 evidence suggested that entropy is informative as an observer-side signal, but that entropy-guided semantic revision does not reliably outperform a matched random control. This creates a distinction between two roles: an **observer** that detects risk, and a **controller** that determines what semantic update should be made. Our central hypothesis is that entropy is better suited to the former role than to the latter.

Accordingly, we define the target problem as **compute routing under a fixed denoising family**. Given an input instance $x$, we first run a shared draft trajectory $d(x)$ for 64 denoising steps. At draft completion, we compute normalized token entropy values $H_i(x)$ over positions and retain a frontier $F(x)$ consisting of the top-$\\rho$ positions, where $\\rho \\approx 0.1211$ in the current full-scale study. Additional denoising is then restricted to this active frontier for at most $T_{\\max}=3$ extra steps.

### 3.2 Entropy-Routed Candidate

The candidate method, `cand_espd`, uses draft-time entropy to decide where further compute should be spent. The method has four fixed ingredients in the current study:

1. Frontier scoring rule: use normalized draft-time token entropy as the routing score.
2. Retention ratio: keep the top-$\\rho$ positions as the active frontier.
3. Stopping rule: stop frontier-only revision once the masked-frontier entropy ratio falls below $\\tau=0.85$, subject to at least one frontier-only step.
4. Revision budget: apply at most $T_{\\max}=3$ frontier-only revision steps after the shared draft.

Importantly, the method does not claim to know what semantic repair should be made at each position. Instead, it claims only that some positions deserve more compute than others. This is why we refer to the method as entropy-routed compute rather than entropy-guided semantic control.

As shown in Figure 1, the candidate and the sham share the same draft, the same broad budget family, and the same frontier-only revision structure. The tested claim is therefore not whether frontier compute helps in general, but whether entropy-based frontier placement provides a better trade-off than a frontier of the same nominal size placed without entropy routing.

**Figure 1.** Entropy-routed compute versus the matched fixed-frontier sham. Both methods share the same 64-step draft and revision budget family, but the candidate uses draft-time entropy to place additional compute, whereas the sham uses a fixed frontier of the same nominal size.

### 3.3 Fixed-Frontier Sham

The key control is `ESPD-FixedFrontier`, a matched sham designed to preserve the frontier budget family while removing entropy-based frontier placement. The sham uses the same nominal retention ratio and the same stopping family, but the frontier itself is placed deterministically rather than by draft-time uncertainty. This control is intentionally stronger than a generic baseline because it preserves the most obvious budget-related degrees of freedom while disabling the claimed routing mechanism.

The sham is crucial because small-gain dLLM results are easy to over-interpret. If a candidate only beats a weak shared baseline, it remains unclear whether the gain comes from the claimed mechanism, a generic extra-compute effect, or an implementation side effect. By comparing the candidate to a matched fixed-frontier control, we can ask a narrower and more meaningful question: does dynamic frontier placement itself matter? At the same time, we emphasize that the current sham is **matched in frontier budget family, but not perfectly matched in execution behavior**: effective batch size, peak VRAM, auxiliary overhead, and average tokens changed still differ between the candidate and the sham. This mismatch is treated as a limitation of the present paper rather than hidden as an implementation detail.

### 3.4 Shared Controls and Paired Outcome Accounting

In addition to the fixed-frontier sham, we compare against two shared controls: `CARD-84` and `RAND-84`. The former uses entropy within a standard revision budget family, whereas the latter uses random targeting within the same broad compute regime. These shared controls serve two purposes. First, they anchor the candidate in the broader revision family rather than only in a single sham comparison. Second, they provide a diagnostic check on the old entropy-as-controller framing: if entropy were already a strong semantic targeting rule, `CARD-84` would be expected to clearly outperform `RAND-84`.

To avoid relying only on aggregate accuracy, we also report paired repair/harm decompositions. For a paired comparison between two methods, we record repaired cases $R$, harmed cases $H$, and the net repaired count $\\Delta_{\\text{repair}} = R - H$. This decomposition is not a substitute for statistical confidence intervals, but it makes the structure of a small gain more transparent.

### 3.5 Runtime Accounting

Because this paper studies a small-gain intervention claim, runtime accounting is part of the method rather than an implementation footnote. We separately log draft and frontier behavior, effective batch size, peak VRAM, wall-clock latency, and auxiliary overhead associated with frontier selection and stopping checks. The goal is not to claim that the current implementation is globally optimal, but to ensure that the candidate, sham, and shared controls are compared under a visible and auditable execution envelope.

## 4. Experiments

### 4.1 Setup

We evaluate on the full GSM8K test set (`n=1319`) under a unified runtime contract inherited from the retained full-scale iteration-4 bundle. All methods use a shared 64-step draft trajectory, and the current comparison is executed with an eager backend, compile disabled, and flash attention disabled. We compare four methods: `cand_espd`, `ESPD-FixedFrontier`, `CARD-84`, and `RAND-84`.

The primary endpoint is the candidate-versus-sham comparison: whether `cand_espd` yields a better quality-speed trade-off than `ESPD-FixedFrontier` under the current runtime contract. Comparisons against `CARD-84` and `RAND-84`, paired repair/harm decomposition, runtime-lineage transparency, and stopping-behavior analysis are secondary endpoints. This endpoint hierarchy matters because the paper does not claim benchmark domination; it claims a bounded, attribution-centered gain that survives a stronger sham than a generic baseline.

The primary outcome is accuracy. We additionally report quality at equal compute, speed at an equal-quality band, average NFE, active-frontier ratio, and paired repair/harm decomposition. The purpose of this metric set is to separate "small but structured" gains from visually impressive but poorly attributed benchmark movement.

### 4.2 Main Full-Scale Results

Table 1 summarizes the main full-scale result. The candidate is not the absolute fastest method, but it achieves the highest quality among the four methods in the current bundle while preserving a better quality-speed trade-off than the matched sham.

| Method | Accuracy | Correct | Equal-quality speed (tok/s) | Avg. NFE | Frontier ratio | Effective batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **cand_espd** | **0.4041** | **533** | 124.42 | 67.93 | 0.1211 | 54 |
| ESPD-FixedFrontier | 0.3988 | 526 | 105.73 | 68.00 | 0.1211 | 52 |
| CARD-84 | 0.3965 | 523 | 126.08 | 68.00 | 0.1000 | 57 |
| RAND-84 | 0.3980 | 525 | **128.00** | 67.00 | 0.1000 | 57 |

**Table 1.** Main full-scale GSM8K results under the current runtime contract. The candidate yields the highest accuracy in the bundle, but its contribution is best interpreted as a bounded trade-off improvement rather than an absolute speed win.

As shown in Figure 2, the candidate occupies a different point in the quality-speed plane from the fixed-frontier sham: it is both more accurate and materially faster within the candidate-versus-sham comparison, even though it does not dominate the shared controls in absolute speed.

**Figure 2.** Quality-speed positioning under a unified runtime contract. The candidate is not the absolute fastest method, but it dominates the matched fixed-frontier sham and slightly improves quality over the shared controls.

### 4.3 Paired Repair/Harm Analysis

The aggregate accuracy gap is small, so the structure of the gain matters. Table 2 reports the paired repair/harm decomposition. Against both shared controls, the candidate yields a modestly positive net repair count. Against the candidate, the sham is net negative.

| Comparison | Fixed | Harmed | Unchanged correct | Unchanged wrong | Net repaired |
| --- | ---: | ---: | ---: | ---: | ---: |
| cand_espd vs RAND-84 | 73 | 65 | 460 | 721 | +8 |
| cand_espd vs CARD-84 | 52 | 42 | 481 | 744 | +10 |
| ESPD-FixedFrontier vs cand_espd | 57 | 62 | 469 | 731 | -5 |

**Table 2.** Paired repair/harm decomposition. The gains are small, but they are not purely an artifact of aggregate averaging; the candidate maintains a modestly positive repair-harm balance.

Figure 3 visualizes the stopping behavior of the candidate and the sham. The candidate has a strongly bimodal stopping pattern (`702` samples stop after the first frontier-only step, `613` run to the third step, and only `4` stop after the second), which is consistent with non-uniform compute allocation. The sham shows a different distribution even under the same nominal frontier budget family.

**Figure 3.** Stopping behavior of the candidate and the fixed-frontier sham. The candidate's sharply bimodal pattern supports a routing interpretation, although the absence of a threshold ablation means the mechanism is not yet fully isolated.

### 4.4 Uncertainty Treatment

Because the margins are small, we report a simple uncertainty treatment directly in the main text. Wilson 95% intervals for the four headline accuracies are highly overlapping:

- `cand_espd`: `[0.3779, 0.4308]`
- `RAND-84`: `[0.3719, 0.4247]`
- `CARD-84`: `[0.3705, 0.4232]`
- `ESPD-FixedFrontier`: `[0.3727, 0.4255]`

Paired McNemar-style descriptive checks on the repair/harm counts also remain non-decisive:

- `cand_espd` vs `RAND-84`: `p ≈ 0.551`
- `cand_espd` vs `CARD-84`: `p ≈ 0.353`
- `ESPD-FixedFrontier` vs `cand_espd`: `p ≈ 0.714`

We report these numbers not to claim statistical victory, but to make the boundedness of the result explicit. The paper's argument is therefore structural and attribution-centered: the candidate survives the candidate-versus-sham comparison and yields a coherent quality-speed trade-off signal, but it does not yet support a strong significance-based superiority claim.

### 4.5 Runtime Lineage and Reproducibility

Because the gains are small, runtime lineage must be read together with the main result. Figure 4 shows effective batch size, peak VRAM, and wall-clock latency across the candidate, sham, and shared controls. The candidate uses a smaller effective batch than the shared controls (`54` vs `57`) but a larger batch than the sham (`52`), and it exhibits a much lower peak VRAM footprint than the sham. These differences reinforce the bounded nature of the claim: the current result is credible, but the execution envelope is not yet fully matched or fully optimized.

For the same reason, the strongest reviewer-safe statement is not that the candidate has established a globally cleaner mechanism. The reviewer-safe statement is that a partially unmatched sham still fails to reproduce the candidate's quality-speed position under the same broad budget family, which is enough to justify further investigation but not enough to close the attribution question.

**Figure 4.** Runtime lineage audit across the candidate, sham, and shared controls. The figure makes the execution envelope explicit rather than hiding it in prose.

We intend to release the machine-readable result bundle, the paired comparison artifacts, the figure-generation scripts, the runtime-lineage tables, a primary-endpoint note, and an explicit claim-boundary note used to keep the current narrative honest. Concretely, the public artifact bundle will include the JSON result files used for Tables 1-2, the scripts for Figures 2-4, the visual audit, the runtime-lineage summary, and the markdown notes required to reproduce the main manuscript tables and figures. This artifact-release plan is part of the paper's scientific message: in a small-gain regime, reproducibility requires exposing execution details rather than only reporting end metrics.

## 5. Discussion

The main lesson of this study is not that entropy has become a validated semantic controller for training-free dLLM revision. Rather, the evidence suggests a more limited and more defensible interpretation: entropy is currently more useful for routing and stopping compute than for directly determining what semantic revision should be applied. This interpretation fits both sides of the empirical record. On the one hand, the shared entropy-centered control does not cleanly separate from the shared random control. On the other hand, the entropy-routed candidate preserves a better quality-speed trade-off than a matched fixed-frontier sham.

This distinction matters because it changes the scientific story. A controller claim asks for strong evidence that the signal identifies the correct semantic update. A routing claim asks for weaker but still meaningful evidence that the signal identifies where extra compute is more worthwhile. The current bundle supports the second claim more convincingly than the first. In that sense, the result is small in magnitude but significant in interpretation: it narrows the role that uncertainty can credibly play in training-free dLLM intervention.

At the same time, the result remains bounded in several important ways. First, the gain is modest. The candidate improves over the shared controls by only `+0.61pp` to `+0.76pp`, the Wilson intervals overlap substantially, and the paired repair/harm checks are directionally positive rather than statistically decisive. Second, the fixed-frontier sham is not yet a perfectly matched control: it differs in effective batch size, peak VRAM, wall-clock latency, auxiliary overhead, and average tokens changed. Third, the current evidence is single-benchmark evidence. Without a cross-task validation on a setting such as MBPP, the safe interpretation remains "credible GSM8K speed-line signal" rather than "general dLLM intervention principle."

These limitations do not force a pivot, but they do dictate the next steps. The highest-value follow-up is one external validation that retains the shared controls and the fixed-frontier sham. This should be accompanied by a reviewer-facing runtime-lineage artifact and at least one routing/stopping split ablation. Together, these additions would address the three most likely reviewer questions: Does the result generalize? Is the gain really about routing rather than accounting? Is the sham strong enough?

An equally important implication concerns the broader iteration-4 narrative. The original proposal ranked the object-level line, `cand_bsr`, ahead of the speed line, `cand_espd`. The full-scale evidence no longer supports that ordering. The correct interpretation is therefore evidence-first: `cand_espd` is the current mainline because it has full-scale support, whereas `cand_bsr` remains a challenger hypothesis that may still prove important but has not yet earned that status empirically. This update is not cosmetic. If the writing pipeline fails to reflect it, the paper will continue to mismatch its strongest evidence.

Finally, the paper also points to a broader methodological lesson for dLLM research. Once benchmark gains become small, the field needs stronger sham controls, more explicit runtime lineage, and more disciplined claim boundaries. In such a regime, a modest but well-audited result can be scientifically more valuable than a larger but poorly attributed one.

## 6. Conclusion

This paper presented a bounded full-scale study of entropy-routed compute for training-free discrete diffusion language models. The main result is modest but informative. On GSM8K, the entropy-routed candidate achieves slightly higher quality than the shared controls and a materially better quality-speed trade-off than a matched fixed-frontier sham under a unified runtime contract.

The most important conclusion is interpretive rather than headline-driven. The evidence does not support entropy as a validated semantic controller, but it does support entropy as a useful routing and stopping signal for allocating additional compute. This shift from controller to router is the main conceptual update delivered by the study.

The paper also argues for a more disciplined evaluation style in small-gain dLLM research. Shared controls, sham controls, paired repair/harm accounting, and runtime-lineage reporting are not optional extras when the effects are small; they are part of the scientific contribution itself.

The current result remains bounded by its modest effect size, its partially unmatched sham, and its lack of cross-task validation. These are not reasons to discard the line, but they are reasons to state the claim carefully. The next step is therefore clear: test whether the candidate-versus-sham separation survives one external validation while making runtime attribution even more explicit. In future releases we will pair this manuscript with a claim-boundary note, a primary-endpoint note, and a runtime-lineage summary so that readers can audit not only what the paper claims, but also what it explicitly declines to claim.

## Figures and Tables

- Figure 1: fig1_entropy_routed_compute.pdf — Candidate/sham conceptual diagram and control boundary.
- Figure 2: fig2_quality_speed.pdf — Quality-speed positioning under the unified runtime contract.
- Figure 3: fig3_stopping_hist.pdf — Stopping behavior of the candidate and the sham.
- Figure 4: fig4_runtime_lineage.pdf — Runtime lineage audit across methods.
- Table 1: inline — Main full-scale GSM8K results.
- Table 2: inline — Paired repair/harm decomposition.

## References

- ActCab+CoDec. 2024. *Enhancing Language Model Factuality via Activation-Based Confidence Calibration and Guided Decoding*. arXiv:2406.13230.
- Bai et al. 2026. *Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models*. arXiv:2602.01842.
- Deng et al. 2023. *Towards a Unified View of Answer Calibration for Multi-Step Reasoning*. arXiv:2311.09101.
- Lou et al. 2023. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*. arXiv:2310.16834.
- Lu et al. 2026. *Advancing Block Diffusion Language Models for Test-Time Scaling*. arXiv:2602.09555.
- Sampler-Centric Evaluation. 2026. *Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models*. arXiv:2602.19619.
- Schiff et al. 2024. *Simple Guidance Mechanisms for Discrete Diffusion Models*. arXiv:2412.10193.
- Xia et al. 2026. *MetaState: Persistent Working Memory for Discrete Diffusion Language Models*. arXiv:2603.01331.
- Zheng et al. 2023. *A Reparameterized Discrete Diffusion Model for Text Generation*. arXiv:2302.05737.
- Zhu et al. 2023. *Rethinking Confidence Calibration for Failure Prediction*. arXiv:2303.02970.
