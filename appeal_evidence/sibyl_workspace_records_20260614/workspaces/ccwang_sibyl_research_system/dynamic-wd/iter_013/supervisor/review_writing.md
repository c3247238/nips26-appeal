# Supervisor Review: Iteration 13

**Score: 6.5 / 10** | **Verdict: CONTINUE**

**Reviewer calibration**: This score reflects a paper that has a promising algorithmic idea and real ImageNet experiments, but with several critical gaps that would lead to rejection at a top venue in its current form. The score can reach 7.5-8.0 with targeted experiments (estimated 1-2 weeks of GPU time).

---

## Executive Summary

EqWD (Equilibrium-Driven Weight Decay) is a clean, well-motivated algorithm that modulates weight decay per layer based on the deviation of the gradient-to-weight ratio from its EMA-tracked equilibrium. The connection to Defazio's ratio equilibrium theory is genuine and the algorithm is simple (3 lines of code). On ImageNet ResNet-50 (45 epochs, 3 seeds), EqWD achieves 72.27 +/- 0.20%, numerically the best among all methods with the lowest variance.

However, the paper has two critical confounds, one factual error, and several major gaps that collectively prevent it from reaching acceptance threshold.

---

## Dimension Scores

### 1. Novelty & Significance: 7/10

**Strengths**: The idea of using ratio deviation (not raw ratio, not gradient norm, not alignment) as a WD modulation signal is genuinely new. No prior work uses |r_t - r*| / r* for this purpose. The connection between Defazio's diagnostic result and a practical algorithm is a natural but non-obvious translation. The negative finding about alignment-based methods (CAWD, CWD) on ImageNet is independently valuable.

**Weaknesses**: The novelty is incremental -- it is a specific instantiation within a well-explored space (adaptive weight decay). The contribution is narrow: one modulation formula, one optimizer (SGDW), CNNs only. A reviewer could argue this is a workshop paper that presents one variant in a large design space without fundamental new insight.

### 2. Technical Soundness: 6/10

**Strengths**: The algorithm is clearly specified and easy to implement. Proposition 1 (equilibrium recovery) is correct by construction. The design rationale for each component (EMA, normalization, additive form) is thoughtful.

**Weaknesses**:
- **Proposition 2 is tautological.** It states "if alignment is a function of norms, then the ratio is sufficient" -- this is true by definition of sufficient statistic. The real question is whether the antecedent holds, which is an empirical claim, not a proposition.
- **AIS diagnostic only on CIFAR-100.** The ratio sufficiency claim is central to the paper but verified only on a dataset where EqWD does not even win. It must be verified on ImageNet.
- **LR schedule factual error.** The paper claims cosine annealing (Section 4.1) but the raw data shows step decay (lr=0.1 for epochs 1-29, lr=0.01 for epochs 30-45). The entire mechanistic narrative about "cosine decay transitions" (Section 4.4, item 2) references a schedule that was not used. This is a serious accuracy issue.

### 3. Experimental Rigor: 6/10

**Strengths**:
- 7 methods compared fairly on ImageNet with identical conditions
- 3 seeds with honest statistical reporting (bootstrap CI, Cohen's d)
- Good ablation coverage (beta, alpha, layer-type)
- Self-aware limitations section (Section 5.6)

**Weaknesses**:
- **Critical confound: effective WD inflation.** EqWD always increases WD (phi >= 1), so the effective average WD is higher than FixedWD. The missing control (FixedWD with higher lambda) is the single most important experiment not conducted. Without it, the contribution could be "use more WD" -- a trivially known insight.
- **45-epoch regime is non-standard.** While internally valid, method rankings can change at 90 epochs. The paper acknowledges this but does not address it experimentally.
- **BEM test omitted.** A budget equivalence test was conducted (EqWD 68.30% vs FixedWD 68.21% under equal tuning) but is not reported. Omitting negative results is worse than reporting them honestly.
- **LR schedule mismatch.** Step decay was used but cosine is claimed. This must be resolved.
- **CIFAR-100 is weak.** EqWD ranks 3rd with default beta. The beta=5.0 result (66.07%) is single-seed.
- **n=3 is underpowered.** The key EqWD vs SWD comparison has a CI that includes zero.

### 4. Reproducibility: 7/10

**Strengths**: Algorithm is simple and precisely specified. Hyperparameters are documented. Training details (epochs, batch size, lr, etc.) are provided. The claim of "3 lines of code" makes reimplementation trivial.

**Weaknesses**: Some hyperparameter discrepancies (lambda=1e-4 in data vs 5e-4 in text; step decay vs cosine in text) could confuse reproducers. No code repository mentioned.

---

## Critical Issues (Would Cause Rejection)

1. **Effective WD inflation confound is uncontrolled.** This is the single most important gap. Run FixedWD with lambda in {5e-4, 6e-4, 7e-4, 8e-4} on ImageNet to isolate timing vs strength.

2. **45-epoch ImageNet is non-standard.** Run 90-epoch experiments for at least EqWD, FixedWD, SWD (top-3) with 3 seeds.

## Major Issues (Significantly Weaken the Paper)

3. **LR schedule factual error.** Paper says cosine, data shows step decay. Must be corrected; mechanistic narrative needs updating.

4. **Proposition 2 is tautological.** Downgrade to empirical finding. Run AIS on ImageNet.

5. **BEM negative result omitted.** Report honestly, frame as tuning efficiency advantage.

6. **Statistical power insufficient.** n=3 is underpowered. Add 2 seeds (789, 1024) for n=5.

7. **CIFAR-100 default-beta results are weak.** Confirm beta=5.0 with 3 seeds.

## Minor Issues

8. VGG-16-BN high variance makes those results uninformative.
9. No EMA r* tracking diagnostic plot.
10. Lambda discrepancy (1e-4 vs 5e-4) between data and text.
11. No vestigial unified-framework over-claims remaining (verify).

---

## What Would Raise the Score

| Action | Estimated Score Impact | Compute Cost |
|--------|----------------------|-------------|
| FixedWD higher-lambda control on ImageNet | +0.5 (to 7.0) | ~3 GPU-days |
| 90-epoch ImageNet (top-3 methods, 3 seeds) | +0.5 (to 7.0-7.5) | ~6 GPU-days |
| Fix LR schedule error + n=5 seeds | +0.5 (to 7.5) | ~2 GPU-days |
| AIS diagnostic on ImageNet | +0.25 | ~0.5 GPU-days |
| Beta=5.0 multi-seed on CIFAR-100 | +0.25 | ~0.1 GPU-days |

All five actions together: estimated final score 7.5-8.0, sufficient for NeurIPS borderline accept.

---

## Cross-Validation Results

**ImageNet EqWD raw data verification:**
- Seed 42: best_top1 = 72.456
- Seed 123: best_top1 = 72.064
- Seed 456: best_top1 = 72.294
- Computed mean = 72.271, std = 0.196
- Paper reports: 72.27 +/- 0.20 -- **MATCHES**

**FixedWD seed 42**: best_top1 = 71.834, paper mean = 71.89 -- **Consistent**

**SWD seed 42**: best_top1 = 72.324, paper mean = 72.04 -- **Consistent**

**LR schedule verification**: Raw data shows lr=0.1 for epochs 1-29, lr=0.01 for epochs 30-45. This is step decay with decay factor 0.1 at epoch 30. Paper claims cosine annealing. **DISCREPANCY.**

**WD lambda verification**: Raw data shows wd_lambda=0.0001 (1e-4). Paper claims 5e-4. **POSSIBLE DISCREPANCY** (need to check if Bayesian optimization tuned this).

---

## Verdict: CONTINUE

The paper is not ready for submission. The effective WD inflation confound and the 45-epoch limitation are both addressable with targeted experiments. The LR schedule error must be corrected. With ~2 weeks of focused effort (experiments + writing fixes), the paper can reach 7.5-8.0.

Priority order for next iteration:
1. Fix the LR schedule factual error (immediate)
2. Run FixedWD higher-lambda control on ImageNet
3. Run 90-epoch ImageNet for top-3 methods
4. Add seeds for n=5
5. Report BEM results honestly
6. Run AIS on ImageNet
7. Confirm beta=5.0 on CIFAR-100 multi-seed

---

*Supervisor Review | Iteration 13 | 2026-03-25*
