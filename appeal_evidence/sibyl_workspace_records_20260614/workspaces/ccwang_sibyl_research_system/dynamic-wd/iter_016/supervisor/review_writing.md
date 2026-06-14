# Supervisor Review — Iteration 16 (current)

**Score: 7.0 / 10.0** | **Verdict: CONTINUE** | **Date: 2026-04-02**

## Dimension Scores

| Dimension | Score |
|---|---|
| Novelty | 7 |
| Soundness | 7 |
| Experiments | 7 |
| Reproducibility | 7 |

## Executive Summary

The paper has undergone a transformative rewrite from the previous "PID taxonomy" framing to a focused single-method contribution: **EqWD (Equilibrium-Driven Weight Decay)**. This is a significant improvement in clarity and coherence. The method is conceptually clean -- modulate per-layer weight decay based on deviation of the gradient-to-weight ratio from its EMA-tracked equilibrium. The experimental results on ImageNet (72.27 +/- 0.20% vs FixedWD 71.89 +/- 0.24%) are verified against raw data and correctly reported with sample std (ddof=1). The paper is honest about limitations (SGDW only, CNN only, margin over SWD not statistically significant).

The previous iteration's critical integrity issues (fabricated AIS=0.566, contradicted CWD K_d mapping, inconsistent CSI formulas, incomplete ImageNet results) have been **resolved through a complete scope reduction**: the problematic PID taxonomy, CSI, and UDWDC have been removed, and AIS is now described correctly as an empirical diagnostic with appropriate caveats.

However, the paper remains a **borderline contribution** for the following reasons:

1. The novelty is limited to applying EMA-tracked deviation as a modulation signal -- a natural idea once Defazio's equilibrium result exists
2. Experimental scope is narrow (SGDW + CNNs only, no AdamW/Transformer validation)
3. The 45-epoch ImageNet training regime produces low absolute accuracies that limit comparability with the literature
4. The CAWD negative result is interesting but lacks mechanism analysis
5. The effective WD inflation confound (EqWD systematically applies more total WD) is acknowledged but not experimentally controlled for

## Score Evolution

| Iteration | Score | Key Change |
|---|---|---|
| iter_015 | 6.5 | PID taxonomy with unfixed integrity issues |
| **current** | **7.0** | Complete rewrite to focused EqWD paper; all prior integrity issues resolved |

The +0.5 improvement reflects the successful resolution of all four critical integrity issues from iter_015 (CWD K_d mapping, AIS=0.566, CSI formulas, incomplete experiments) through scope reduction. The paper is now internally consistent and honest, but needs stronger experimental evidence to advance further.

---

## Critical Issues (1)

### 1. EFFECTIVE WD INFLATION CONFOUND

**Category:** Experiment | **Severity:** CRITICAL

EqWD's modulation factor phi_l(t) >= 1, meaning it **ONLY increases** weight decay relative to FixedWD. The paper acknowledges this (Section 3.2, "effective average weight decay over training is systematically higher for EqWD") and mentions it in limitations, but does not run the definitive control experiment: FixedWD with lambda tuned to match EqWD's average effective WD.

Without this control, EqWD's +0.38% over FixedWD could be entirely explained by EqWD simply applying more weight decay overall, not by adaptive modulation. This is the most important missing experiment because it directly challenges the paper's central claim.

**Suggestion:** Compute the average effective lambda over training for EqWD (trivial from logged ratio trajectories), then run FixedWD at that lambda value with 3 seeds on ImageNet. Cost: ~3 GPU-hours. This resolves the paper's most fundamental confound. Alternatively, report the average effective lambda so reviewers can assess the confound magnitude.

---

## Major Issues (7)

### 2. 45-EPOCH IMAGENET TRAINING PRODUCES LOW ABSOLUTE ACCURACIES

**Category:** Experiment

EqWD achieves 72.27% on ImageNet ResNet-50 vs the standard ~76% at 90 epochs. The 45-epoch regime produces results in a low-accuracy regime that limits comparability with the literature. CWD's published results (ViT-S 79.45% with AdamW, ICLR 2026) operate in a completely different accuracy range.

**Suggestion:** At minimum, clearly disclose the actual learning rate schedule. Ideally, run one 90-epoch comparison to validate that the relative advantage transfers.

### 3. LEARNING RATE SCHEDULE DESCRIPTION MAY BE INCORRECT

**Category:** Soundness

The paper states "initial learning rate 0.1 with cosine annealing" throughout. However, raw experimental data shows lr=0.1 for epochs 1-29, then lr=0.01 for epochs 30-45 -- a step decay schedule with a 10x drop at epoch 30, NOT cosine annealing. If confirmed, this is a factual error that affects reproducibility and the narrative about "cosine decay transitions."

**Suggestion:** Verify the training code's actual LR schedule. If it is step decay, correct all references to "cosine annealing" and update the transitional phase discussion.

### 4. PROPOSITION 2 (RATIO SUFFICIENCY) IS CIRCULAR

**Category:** Soundness

Proposition 2 states "if alignment deviation is also a function of gradient and weight norms, then the ratio is sufficient for alignment-relevant information." The "if" condition is the entire content of the claim -- the proposition is a tautology. The empirical AIS verification is only performed on CIFAR-100/ResNet-20 and VGG-16-BN, not on ImageNet where the main results are.

**Suggestion:** Either run AIS on ImageNet (even a subset) or explicitly acknowledge AIS was not measured at ImageNet scale. Consider downgrading Proposition 2 to an empirical observation.

### 5. NO ADAM/ADAMW EXPERIMENTS

**Category:** Experiment

The paper exclusively uses SGDW, which is a minority optimizer in modern deep learning. AdamW is the dominant optimizer for virtually all transformer and many CNN pipelines. The ratio dynamics under Adam's second-moment scaling may differ fundamentally.

**Suggestion:** Run one ResNet-50 CIFAR-100 experiment with AdamW + EqWD (~30 min). This single experiment would dramatically strengthen practical relevance.

### 6. ABLATION BETA=5.0 SINGLE-SEED CONCERN

**Category:** Experiment

The ablation claims beta=5.0 "substantially exceeds all baselines" at 66.07%, but this is a single seed. High-beta may increase variance, and the single-seed ablation for beta=1.0 (65.39%) already exceeds the 3-seed mean (65.05%).

**Suggestion:** Run beta=5.0 with 3 seeds on CIFAR-100 (~30 min). If the mean holds, promote the finding; if it drops, the single-seed result was noise.

### 7. WEIGHT DECAY LAMBDA VALUE DISCREPANCY

**Category:** Soundness

The raw experimental data shows wd_lambda=0.0001 (1e-4), but the paper states lambda=5e-4 throughout. This is a 5x discrepancy that directly affects reproducibility.

**Suggestion:** Verify the actual lambda used in experiments. Correct whichever source (paper or data JSON) is incorrect.

### 8. LIMITED NOVELTY BEYOND DEFAZIO'S INSIGHT

**Category:** Novelty

The core idea -- track equilibrium of r_t via EMA, modulate WD based on deviation -- is a natural next step once Defazio's equilibrium result exists. The additive form 1 + beta * |dev| is the simplest possible modulation.

**Suggestion:** Strengthen the novelty argument by showing alternative formulations are inferior through ablation, and providing deeper analysis of why EMA tracking is superior to the theoretical lambda/gamma target.

---

## Minor Issues (3)

### 9. CIFAR-10 TABLE MISSING EQWD

Table 7 (CIFAR-10 diagnostic) does not include EqWD or CAWD. The proposed method is missing from a diagnostic benchmark.

### 10. VGG-16-BN RESULTS VERY WEAK

EqWD on CIFAR-100/VGG-16-BN achieves 62.81 +/- 1.31% -- well below FixedWD's 65.19% with ResNet-20. The high variance and low accuracy suggest poor architecture interaction. The paper attributes this to "VGG's sensitivity" without investigation.

### 11. CONCLUSION NUMBERING ERROR

The Conclusion section lists future work as "First... Second... Third... Fifth..." -- skipping "Fourth." Minor formatting issue.

---

## Risks

1. Lambda value discrepancy (1e-4 in data vs 5e-4 in paper text) is a potential data integrity issue
2. Learning rate schedule discrepancy (step decay vs cosine annealing) undermines the transitional phase narrative
3. Effective WD inflation confound means the central claim is unvalidated
4. 45-epoch results in low-accuracy regime may not generalize to standard training
5. No AdamW validation limits practical impact
6. Single-seed beta=5.0 result may not replicate

## Evidence Gaps

1. FixedWD at EqWD's average effective lambda (WD inflation control)
2. Verification of actual lambda_base (5e-4 or 1e-4)
3. Verification of actual LR schedule (cosine or step decay)
4. AIS diagnostic on ImageNet
5. Multi-seed validation of beta=5.0
6. At least one AdamW experiment
7. EqWD results on CIFAR-10 diagnostic
8. 90-epoch ImageNet run for standard-length validation

## Path to Higher Scores

**To reach 7.5:** (1) Resolve lambda and LR schedule discrepancies, (2) run the WD inflation control experiment, (3) add EqWD to CIFAR-10 table, (4) multi-seed validate beta=5.0.

**To reach 8.0:** Additionally (5) run one AdamW experiment, (6) run a 90-epoch ImageNet comparison, (7) provide AIS validation on ImageNet.

## Strengths Worth Preserving

- Honest reporting of limitations, negative results (CAWD), and proper statistical methodology (bootstrap CIs, Cohen's d, ddof=1)
- Clean scope reduction from overambitious PID taxonomy to focused single-method contribution
- Internal consistency of experimental claims with raw data (verified by cross-validation)
- Transparent acknowledgment of the WD inflation confound in the limitations section
