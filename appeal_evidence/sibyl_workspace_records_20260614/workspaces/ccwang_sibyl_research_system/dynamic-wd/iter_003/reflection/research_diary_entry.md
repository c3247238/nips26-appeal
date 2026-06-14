# Research Diary — Iteration 3

**Date:** 2026-03-18
**Project:** Unified Dynamic Weight Decay Framework
**Stage Completed:** Full cycle (experiments → analysis → writing → review)

---

## What We Did

Iteration 3 completed a full research cycle: 42 controlled experiments (7 WD methods × 3 seeds × 2 datasets on ResNet-20 with AdamW), result analysis with 6-agent debate, paper writing (sequential mode), and multi-reviewer assessment (final critic, supervisor, Codex).

**Key deliverables:**
- Complete experimental results on CIFAR-10/100 with ResNet-20/AdamW
- Phi Modulator Framework with four-axis taxonomy
- Three diagnostic metrics (BEM, CSI, AIS)
- Phi Invariance Conjecture (formalized null result)
- Full paper draft (integrated_paper.md + LaTeX)

## What We Found

**The core finding is robust:** All dynamic WD methods are statistically indistinguishable from constant WD under AdamW on CIFAR-10/100 with ResNet-20. Even no_wd (λ=0) performs comparably. Weight norms converge to a narrow range (95.89-97.04) despite 10× BEM variation.

**The unreported SGD data is gold:** SGD baseline experiments were run but not analyzed for the paper. They show constant WD achieves 91.22% vs no_wd 90.30% on CIFAR-10 under SGD (Δ=0.92%, ~8× the AdamW gap). This directly validates the optimizer-specificity claim and is the single most impactful addition for Iteration 4.

## Review Scores

| Reviewer | Score |
|----------|-------|
| Final Critic | 5.5 |
| Supervisor | 6.0 |
| Codex | 6.0 |
| Critic Findings | 5.5 |
| **Average** | **5.75** |

All reviewers: ITERATE. Unanimous agreement on strong conceptual foundation but insufficient evidence.

## Top Issues Identified

1. **Critically narrow experimental scope** — only ResNet-20 on CIFAR-10/100 with one optimizer and one hyperparameter setting
2. **Mathematical errors in all three metrics** — BEM boundedness wrong, half_lambda BEM buggy, AIS range wrong, CSI unnormalized
3. **Insufficient statistical power** — 3 seeds gives only 2 df, minimum detectable effect ~0.7%
4. **SGD data unreported** — most impactful single omission
5. **Shallow theory** — only trivial Proposition 1, no formal proof of invariance

## Key Insight from Codex

Codex raised a unique concern: at λ=5×10⁻⁴, weight decay may be a second-order perturbation (~1% of gradient step), making all modulators trivially equivalent regardless of AdamW's adaptive properties. Testing at higher λ is essential to distinguish the two explanations.

## Plan for Iteration 4

**P0 (must-fix):** Report SGD data, add ImageNet experiments, fix all metric definitions, add VGG-16-BN, increase to 5 seeds.

**P1 (should-fix):** Prove quadratic-loss invariance theorem, add perturbation analysis, TOST testing, training curves, implicit regularization literature engagement, higher λ experiments.

**Estimated GPU budget:** 22-30 hours (parallelizable across 8 GPUs).

## Lessons Learned

- Never leave collected data unanalyzed — the SGD results were already there
- Metric papers need extra-careful boundary-case checking
- Null results demand stronger evidence than positive results
- 3 seeds is never enough for equivalence claims
- Address confounds proactively instead of letting reviewers find them
- Codex provides genuinely orthogonal perspective — worth the cost

---

*Next iteration target: 7.0-7.5*
