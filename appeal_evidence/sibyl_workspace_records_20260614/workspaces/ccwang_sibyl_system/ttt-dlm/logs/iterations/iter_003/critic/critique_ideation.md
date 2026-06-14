# Ideation Critique (Iteration 3 — Post-DTA Full-Scale Results)

**Reviewer**: Critic Agent
**Date**: 2026-03-10
**Scope**: `idea/proposal.md`, `idea/final_proposal.md`, `idea/alternatives.md`, `idea/hypotheses.md`

---

## Overall Assessment: 5.5/10

The ideation process was ambitious and intellectually rich. The "DLM denoising as implicit TTT" insight is genuinely novel and the information augmentation spectrum is a well-designed conceptual framework. However, the core hypothesis---that making this implicit TTT explicit via LoRA updates would improve reasoning accuracy---has been **falsified** by full-scale experiments. The ideation stage overestimated the alignment between self-supervised MLM loss and task-level correctness, a failure mode that was foreseeable from the proposal's own risk analysis but insufficiently weighted.

---

## CRITICAL Issues

### I1. Core Hypothesis Falsified — Ideation Failed to Anticipate the Fundamental Misalignment

The proposal's H1 predicted "+5-10pp" improvement from DTA on Countdown. Full-scale result: +0.1pp (4.8% vs 4.7%). H2 predicted DTA+ReMDM > ReMDM alone. Full-scale result: DTA+ReMDM (3.6%) < ReMDM (4.4%). H3 predicted non-saturating scaling curves. Pilot data shows no meaningful signal at any scale.

The proposal identified this as a risk (Section "Failure Modes", item 3: "Benchmark improvement <2pp, probability 35%"). But 35% was an underestimate---the deeper issue is that **the proposal's fundamental premise is flawed**. The claim that "DLM denoising is structurally analogous to TTT" is true at a formal level, but the analogy breaks down at the most critical point: AR TTT updates occur on a meaningful self-supervised signal (next-token prediction reflects the sequence's actual distribution), while DTA's M-step uses MLM on tokens the model itself just predicted. The model is essentially learning to agree with its own predictions---a near-tautological training signal.

The proposal's risk section identified "梯度计算开销过大" and "LoRA 更新导致退化" as high-risk items but **did not identify the self-supervision signal quality problem as the primary risk**. This is the most important blind spot in the ideation.

### I2. DMI Was Undervalued as "Level 1 Ablation" — Should Have Been the Primary Proposal

The proposal ranks DMI as a "Level 1 ablation baseline" below SCP and DTA. Full-scale results: DMI (9.3%) is the best-performing method by a wide margin. The proposal's ranking was based on theoretical expressivity (parameters > tokens > embeddings), but the empirical reality is that DMI's simple soft-embedding injection outperforms everything else at near-zero cost.

The ideation process privileged theoretical elegance over empirical pragmatism. DMI should have been investigated as a first-class method with its own systematic evaluation plan, not relegated to an ablation baseline.

---

## HIGH Issues

### I3. The "Information Island" Framing Survives, But Proposed Solutions Don't

The proposal's diagnosis---that DLM denoising discards cross-step information and this is a fundamental bottleneck---is validated by the DMI results. DMI's success proves that even minimal cross-step information (1-step embedding lookback) has substantial value. But the proposed solutions (SCP, DTA) that aim to provide *more* cross-step information show diminishing or zero returns:

| Method | Cross-Step Info | Accuracy | Overhead |
|--------|----------------|----------|----------|
| DMI | 1-step embedding | 9.3% | 1.2x |
| SCP | Token-level contradiction | 9.1% | 7x |
| DTA | Full-trajectory parameters | 4.8% | 4x |

This reveals a crucial insight the proposal missed: **the bottleneck is not the amount of cross-step information, but its type and granularity**. DMI provides exactly the right type of information (soft embedding context from the previous step) at exactly the right granularity (embedding level). More expensive, higher-expressivity methods don't help because the problem isn't a lack of information---it's that the model's predictions are already locally optimal given the available context, and parameter updates don't change the model's fundamental reasoning capability.

### I4. Hypothesis Structure Had Wrong Priority Ordering

The hypotheses (H1-H10) were ordered with DTA hypotheses first (H1-H3) and DMI hypotheses later (H4-H5). In retrospect, H5 ("DMI's logit carry-over improves remasking prediction quality") was closer to the actual discovery than H1. The hypothesis structure should have been organized by risk level and speed of falsification, not by theoretical ambition.

### I5. Alternatives Section Is Now the Primary Narrative

The pivot decision tree in `alternatives.md` contemplated three backup plans:
1. DMI + SCP combination (probability 70%)
2. Theory-only paper (probability 60%)
3. TTT layers with lightweight training (probability 50%)

Alternative 1 has materialized: DMI works, SCP is comparable but expensive. The paper should pivot to this narrative. Alternative 2 (theory paper) is also viable if the VDTA framework is reframed as explaining *why* parameter-space adaptation is insufficient for reasoning tasks.

---

## MEDIUM Issues

### I6. The "Contrarian" Perspective Was Right

The proposal gave the contrarian perspective only 5% weight ("DTA 与 DLM 不兼容"). In retrospect, the contrarian's core argument---that the self-supervised signal in DLM denoising is too weak/misaligned for meaningful parameter adaptation---was correct. The proposal's rebuttal ("DTA operates in parameter space, bypassing remasking limitations") was technically accurate but empirically irrelevant: operating in parameter space doesn't help if the gradient signal is uninformative for the task.

The ideation process systematically underweighted dissenting views. The contrarian's concerns should have triggered an early falsification experiment: run DTA on 50 Countdown problems before committing to full-scale evaluation.

### I7. Cross-Disciplinary Insights Were Decorative

The proposal cited Turbo decoding analogy and dynamical correction analogy from the cross-disciplinary perspective. These analogies influenced design decisions (warmup period, "delayed commitment principle") but none of these design choices affected the fundamental outcome. The proposal used interdisciplinary analogies as decoration rather than as sources of falsifiable predictions.

### I8. The Proposal's Confidence Was Uncalibrated

The proposal assigned DTA a "综合评分 8.3/10" and ranked it as the top proposal. DMI was not even listed in the proposal rankings because it was considered a trivial ablation baseline. This massive miscalibration between predicted and actual value suggests the evaluation rubric over-weighted novelty and theoretical depth relative to empirical likelihood of success.

---

## Positive Notes

1. **Information augmentation spectrum**: This is a genuinely useful conceptual framework that survives the DTA failure. The four-level hierarchy (vanilla < DMI < SCP < DTA) provides a principled way to think about cross-step information transfer.

2. **Literature integration**: The proposal synthesizes a large body of work (TTT, DLM inference scaling, remasking methods) into a coherent narrative. The positioning table, while needing an empirical column, is well-designed.

3. **Fallback planning**: The alternatives section anticipated failure modes and provided viable pivot paths. The project was not paralyzed by DTA's failure.

4. **VDTA theoretical framework**: Despite not translating into empirical gains, the EM interpretation of DLM denoising + parameter updates is a novel theoretical contribution with potential independent value.
