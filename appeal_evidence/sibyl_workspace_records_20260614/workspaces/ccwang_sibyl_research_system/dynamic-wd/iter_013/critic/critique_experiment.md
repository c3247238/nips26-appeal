# Experiment Critique

## Executive Summary

The experimental execution is competent---multi-seed ImageNet runs, comprehensive baselines, ablation studies---but several critical gaps undermine the core claims.

## CRITICAL: Budget Equivalence Contradicts Central Thesis

The most important experimental result in this paper is buried and not reported: **the budget equivalence test shows no dynamic WD method beats tuned FixedWD**. Under Bayesian optimization with 15 trials:

| Method | Tuned Mean (%) | Seeds |
|--------|:-:|:-:|
| SWD | 68.57 +/- 0.09 | [68.70, 68.51, 68.51] |
| CWD | 68.39 +/- 0.24 | [68.34, 68.70, 68.13] |
| EqWD | 68.30 +/- 0.15 | [68.49, 68.13, 68.28] |
| FixedWD | 68.21 +/- 0.38 | [68.73, 68.10, 67.81] |

EqWD at 68.30% is barely above FixedWD at 68.21%, well within noise. SWD actually beats EqWD. This means the main paper's EqWD advantage is likely explained by: (a) using default hyperparameters that happen to be near-optimal, or (b) effective WD inflation from the phi >= 1 design.

**The paper MUST report this result.** Omitting it is intellectually dishonest.

## CRITICAL: Effective WD Inflation Confound

EqWD's modulation factor phi(t) = 1 + beta * dev is always >= 1. Over an ImageNet training run, the average phi is likely ~1.05-1.15, meaning EqWD at lambda_base = 5e-4 applies effective average WD of ~5.5e-4 to 5.75e-4.

**The missing control**: FixedWD at lambda = 6e-4 or 7e-4. Without this, we cannot distinguish "better-timed WD" from "more WD is better at this operating point." This is the single most important missing experiment.

## CRITICAL: Control Experiments Appear Corrupted

The gradient-norm-only control reproduces the main results exactly:

| Method | Main Exp | GradNorm-Only Control |
|--------|:-:|:-:|
| EqWD | 65.05 +/- 0.36 | 65.05 +/- 0.36 |
| FixedWD | 65.19 +/- 0.25 | 65.19 +/- 0.25 |
| SWD | 64.84 +/- 0.12 | 64.84 +/- 0.12 |

The phase-schedule control also reproduces EqWD exactly (65.05). These controls should produce DIFFERENT numbers if properly implemented. Either the implementations are buggy or the results directories were copied. This must be investigated.

## MAJOR: 45-Epoch ImageNet Regime

45 epochs is half the standard 90-epoch protocol. At 45 epochs:
- The network is undertrained (~72% vs ~76% at 90 epochs)
- Transitional phases (warmup, LR transitions) occupy proportionally more training
- This systematically advantages EqWD, whose mechanism targets transitional phases
- No single-seed 90-epoch validation was performed

## MAJOR: Budget Equivalence Protocol Deviation

The methodology planned 50 Optuna trials at 200 epochs. The actual experiment used 15 trials at 100 epochs. This is a 70% reduction in search budget and 50% reduction in training epochs. The deviation is not documented.

## MAJOR: No AdamW Validation

SGDW-only experiments in 2026 are insufficient. AdamW is the default optimizer for virtually all modern training. The paper acknowledges this limitation but does not attempt even a single CIFAR-level AdamW experiment.

## POSITIVE: What Works Well

1. **Multi-seed consistency**: All core experiments use 3 seeds with properly reported statistics
2. **Comprehensive baselines**: 6 comparison methods spanning all 4 WD streams
3. **Honest reporting**: The paper acknowledges most limitations explicitly (Section 5.6)
4. **AIS diagnostic**: The alignment informativeness analysis is well-designed and provides genuine insight
5. **Statistical methodology**: Bootstrap CIs, Cohen's d, appropriate caveats about n=3 power
