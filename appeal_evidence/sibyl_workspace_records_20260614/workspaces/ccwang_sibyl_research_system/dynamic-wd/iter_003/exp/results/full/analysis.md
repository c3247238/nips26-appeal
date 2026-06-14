# Full Experiment Results Analysis
## Unified Dynamic Weight Decay Framework - Iteration 3

### Experiment Configuration
- **Datasets**: CIFAR-10, CIFAR-100
- **Architecture**: ResNet-20
- **WD Methods**: constant, cwd_hard, swd, cosine_schedule, random_mask, half_lambda, no_wd
- **Seeds**: 42, 123, 456 (3 independent runs per configuration)
- **Epochs**: 200, **LR**: 1e-3 (cosine annealing), **Base WD**: 5e-4
- **Optimizer**: UnifiedAdamW with Phi modulator framework
- **Total experiments**: 42 (7 methods × 3 seeds × 2 datasets)

### Key Results

#### CIFAR-10 / ResNet-20

| Method | Mean Acc (%) | Std (%) | CSI | AIS | BEM | Weight Norm |
|--------|:-----------:|:-------:|:---:|:---:|:---:|:----------:|
| constant | **90.13** | 0.31 | 0.841 | 0.336 | 0.000 | 95.89 |
| cosine_schedule | **90.12** | **0.07** | 0.936 | 0.352 | 0.503 | 96.28 |
| random_mask | **90.12** | 0.30 | 0.892 | 0.359 | 0.500 | 96.38 |
| half_lambda | 90.09 | 0.29 | 0.853 | 0.410 | 0.000 | 96.34 |
| no_wd | 90.08 | 0.32 | 0.964 | 0.343 | 1.000 | 97.04 |
| cwd_hard | 90.06 | 0.24 | 0.851 | 0.368 | 0.503 | 96.46 |
| swd | 89.88 | 0.25 | 0.838 | 0.360 | 0.900 | 96.84 |

**Spread**: 0.25% (89.88% - 90.13%) — within noise margin.

#### CIFAR-100 / ResNet-20

| Method | Mean Acc (%) | Std (%) | CSI | AIS | BEM | Weight Norm |
|--------|:-----------:|:-------:|:---:|:---:|:---:|:----------:|
| cosine_schedule | **63.42** | 0.42 | 0.868 | 0.344 | 0.503 | 105.15 |
| constant | 63.15 | 0.30 | 0.864 | 0.329 | 0.000 | 104.72 |
| swd | 63.06 | 0.29 | 0.854 | 0.339 | 0.900 | 105.78 |
| half_lambda | 62.91 | 0.47 | 0.866 | 0.342 | 0.000 | 105.45 |
| random_mask | 62.87 | 0.38 | 0.867 | 0.320 | 0.501 | 105.43 |
| cwd_hard | 62.84 | 0.30 | 0.855 | 0.321 | 0.500 | 105.49 |
| no_wd | 62.66 | 0.38 | 0.867 | 0.280 | 1.000 | 106.03 |

**Spread**: 0.76% (62.66% - 63.42%) — larger but still within overlap of confidence intervals.

### Statistical Tests (Paired t-test vs Constant Baseline)

All comparisons are **not statistically significant** (p > 0.05):

| Method | CIFAR-10 Δ | p-value | CIFAR-100 Δ | p-value |
|--------|:----------:|:-------:|:-----------:|:-------:|
| cwd_hard | -0.07% | 0.832 | -0.31% | 0.326 |
| swd | -0.25% | 0.513 | -0.10% | 0.801 |
| cosine_schedule | -0.01% | 0.935 | +0.26% | 0.117 |
| random_mask | -0.01% | 0.950 | -0.29% | 0.090 |
| half_lambda | -0.05% | 0.901 | -0.25% | 0.608 |
| no_wd | -0.05% | 0.825 | -0.49% | 0.312 |

### Diagnostic Metric Analysis

#### CSI (Coupling Stability Index)
- All methods converge to high CSI (~0.85-0.96) at training end
- **no_wd** has highest CSI (0.96) — expected, as weight growth is completely unconstrained
- **cosine_schedule** also high (0.94) — schedule-weight norm coupling is less stable
- CSI does **not correlate with accuracy** — it measures dynamics, not performance

#### AIS (Alignment Informativeness Score)
- All methods show moderate AIS (~0.28-0.41)
- **half_lambda** has slightly higher AIS on CIFAR-10 (0.41) — alignment signal is more diverse
- AIS is consistent across methods — alignment diversity is not method-dependent
- This suggests alignment information is an **intrinsic network property**, not modulated by WD method

#### BEM (Budget Equivalence Metric)
- Perfect budget equivalence: constant (0.00), no_wd (1.00), swd (0.90), cwd/cosine/random (~0.50)
- Despite 10x variation in WD budgets (BEM 0.0 to 1.0), **accuracy differences are < 0.5%**
- **Critical finding**: WD budget equivalence is irrelevant — the total WD budget does not determine accuracy

### Key Findings

1. **All dynamic WD methods fail to improve over constant baseline** on both CIFAR-10 and CIFAR-100
2. **Even no WD (λ=0) achieves comparable accuracy** — weight decay is not critical for AdamW on these benchmarks
3. **Budget equivalence is a red herring** — methods with vastly different effective WD budgets perform identically
4. **Alignment-based methods (CWD, SWD) don't help** — gradient-weight alignment does not provide useful signal for WD modulation
5. **Cosine schedule has lowest variance** on CIFAR-10 (σ=0.07%) — potentially more stable but not more accurate
6. **Generalization gap is uniform** (~9.7% CIFAR-10, ~25.6% CIFAR-100) across all methods
7. **Weight norms converge similarly** regardless of WD method — AdamW's built-in adaptive step size dominates

### Implications for the Phi Modulator Framework

The Phi modulator framework successfully unifies all WD methods under one interface, enabling systematic comparison. The framework itself is a methodological contribution. However, the empirical results show that **the choice of Phi function does not matter** for final accuracy — at least on CIFAR-scale problems with ResNet-20 and AdamW.

This supports the hypothesis that **AdamW's per-parameter adaptive learning rate already provides sufficient implicit regularization**, making the WD schedule/modulation superfluous. The coupling between learning rate and weight decay (the "decoupled" aspect of AdamW) may already be optimal in the constant case.
