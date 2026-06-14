# Experiment Critique

## Overall Assessment

The experimental design shows strong methodology (multi-seed, controlled protocols, diagnostic tracking) but suffers from critical incompleteness (ImageNet 4/7 methods), a known implementation bug (UDWDC-v2 BN layers), and missing ablations that would resolve the paper's most important open questions.

## Strengths

1. **Controlled protocol**: Same augmentation, same optimizer (SGD with momentum 0.9), same LR schedule (cosine annealing), same seed set across all methods. This is a model of fair comparison.

2. **Multi-seed evaluation**: 3 seeds (42, 123, 456) for CIFAR, 3-5 for ImageNet. Mean +/- std reported throughout.

3. **Diagnostic tracking**: Per-layer rho_t^l, alpha_t^l, ||w_t^l||, ||g_t^l||, and effective lambda_t^l logged every epoch. This is unusually thorough.

4. **Three-benchmark coverage**: CIFAR-10 (small), CIFAR-100 (medium), ImageNet (large) provides scale diversity.

## Critical Experimental Issues

### Issue 1: Incomplete ImageNet (CRITICAL -- 10+ iterations unresolved)

Only 4/7 methods completed on ImageNet: CPR, FixedWD, CWD (1 seed), UDWDC. Missing:
- **NoWD**: Required as BEM denominator. Without it, BEM is not computable on ImageNet. BEM is Contribution 3.
- **SWD**: Required for complete scheduling-family comparison.
- **DefazioCorrective**: Required for complete scheduling-family comparison.
- **CWD**: 2 additional seeds needed for valid statistical comparison.

Total missing: 11 runs (3+3+3+2). With 8 GPUs and ~5 hours/run, this is approximately 7 GPU-days. There is no computational excuse for not completing this in 14+ iterations.

**Impact**: Without full ImageNet coverage, the paper cannot claim "comprehensive experiments." The ImageNet BEM comparison -- the most impactful possible result -- is blocked.

### Issue 2: UDWDC-v2 BN Layer Bug (MAJOR)

UDWDC-v2 applies floor clipping (lambda_min = 0.1 * lambda_base) to ALL 65 layers, including 41 BN layers. BN layers are scale-invariant; WD on them functions as an effective LR modifier, not regularization. This produces WD budget = 98599, which is 205,000x FixedWD's 0.48.

Consequences:
- Table 3 UDWDC-v2 WD budget is meaningless.
- BEM for UDWDC-v2 is approximately 0 (accuracy improvement / enormous budget).
- The v2 "fix" is worse than the original problem: it does not address instability; it just floods the system with WD on irrelevant layers.

**Fix**: One-line code change to skip BN layers in the UDWDC control loop. Re-run 2 experiments (CIFAR-10, CIFAR-100, 3 seeds each).

### Issue 3: Missing Critical Ablations

Two ablations would resolve the paper's two most important open questions:

**A. CWD halved-lambda ablation (P0, ~7 GPU-hours)**
- FixedWD @ lambda=5e-5 on CIFAR-10 (3 seeds) + ImageNet (3 seeds)
- Resolves: Does CWD's alignment mask contribute beyond magnitude reduction?
- If halved-lambda FixedWD matches CWD: CWD is just "use less WD"
- If halved-lambda FixedWD differs: alignment mask has independent effect
- This has been flagged as P0 for 3+ iterations. Not running it is inexcusable.

**B. CPR budget-matched ablation (P1, ~8 GPU-hours)**
- FixedWD @ lambda=2e-4 and lambda=3e-4 on ImageNet (3 seeds each)
- Resolves: Does CPR's 3.02pp advantage come from feedback control or more regularization?
- CPR uses 2.3x FixedWD's WD budget on ImageNet
- Without this, CPR's dominance is confounded by regularization strength

### Issue 4: Data-Paper Inconsistency History

Previous iterations flagged conflicting accuracy numbers between summary.json and epoch_metrics.jsonl for VGG-16-BN (Table 3 in CIFAR-100 experiments). While the supervisor verified current numbers match summary.json, the existence of two conflicting data sources (summary.json vs epoch_metrics.jsonl) is a reproducibility hazard.

The experiment results directories (exp/results/) are currently empty, preventing independent verification of any reported number from workspace artifacts.

### Issue 5: Population vs Sample Standard Deviation

At N=3 seeds, the difference between population std (ddof=0) and sample std (ddof=1) is a factor of sqrt(3/2) = 1.22. For CPR on ImageNet (reported as +/- 0.05%), sample std would be +/- 0.06%. This affects Welch's t-test results and reported confidence intervals. The paper does not specify which convention is used.

## Batch Size Sweep Analysis (Section 5.4)

The batch size sweep (Table 7) is well-designed and produces an important negative result: H3 (CWD SNR increases monotonically with batch size) is definitively falsified. CWD collapses at large batch sizes (std=7.16% at bs=1024), which is a genuine finding.

However, the interpretation could be stronger: the catastrophic CWD failure at bs >= 512 suggests CWD should carry a safety warning for large-batch training regimes, which the paper underemphasizes.

## Gain Ablation (Section 5.3)

The 7-variant gain ablation on CIFAR-100/VGG-16-BN is methodologically sound. Key findings are clear:
- K_d alone slightly outperforms FixedWD (0.11pp, within noise)
- K_p consistently degrades performance
- Full PID does not outperform any single-gain variant

However, the ablation uses UDWDC-v2 floor clipping, which applies WD to BN layers. The results may change after the BN layer bug fix.

## Recommendations Priority

| Priority | Action | GPU-hours | Impact |
|----------|--------|-----------|--------|
| P0 | Complete ImageNet (11 runs) | ~55h | Enables BEM, full comparison |
| P0 | CWD halved-lambda ablation | ~7h | Resolves CWD attribution |
| P1 | Fix UDWDC-v2 BN bug + re-run | ~8h | Corrects WD budget, BEM |
| P1 | CPR budget-matched ablation | ~8h | Resolves CPR attribution |
| P2 | Population vs sample std audit | ~1h | Fixes statistical precision |
