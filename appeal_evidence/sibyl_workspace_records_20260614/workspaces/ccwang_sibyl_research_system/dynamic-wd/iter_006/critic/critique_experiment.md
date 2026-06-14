# Experiment Critique

## Overall Assessment: 4/10 (Major gaps relative to plan)

The executed experiments are methodologically sound (proper multi-seed evaluation, statistical testing, diagnostic instrumentation) but woefully incomplete relative to both the proposal's experimental plan and the project's explicit constraints (ImageNet, VGG-16-BN required).

## Critical Experimental Gaps

### 1. ImageNet: Zero Data (Mandatory per project constraints)
The proposal planned Phase 3C: ResNet-50, 90 epochs, 4 methods x 3 seeds = 12 runs. Status: ALL FAILED per supervisor analysis. No ImageNet results appear anywhere in the paper. Every top-tier WD paper includes ImageNet or LLM experiments. Without scale validation, the paper's claims are confined to 0.27M-parameter models on 32x32 images.

**Impact**: Desk-rejection risk at ICML/NeurIPS/ICLR. Acceptable at AISTATS/workshop only.

### 2. VGG-16-BN: Data Exists but Not in Paper
Iter_005 ran VGG-16-BN on CIFAR-10 with 4/7 methods (constant, half_lambda, CWD, cosine_schedule). The phi spread was 0.16%. This data supports the narrow-band thesis but is absent from the paper entirely. The methodology planned 8 methods on VGG-16-BN; only 4 were run.

**Impact**: Missed opportunity. Including VGG data would strengthen the multi-architecture claim at zero additional compute cost.

### 3. NoBN Ablation: Incomplete and Possibly Contradictory
NoBN data exists for only 2 WD methods (constant: 87.74 +/- 0.21%, CWD: 87.62 +/- 0.13%) plus 1 seed of no_wd (87.79%). The spread between constant and CWD is 0.12pp -- STILL narrow. This contradicts the paper's prediction that "without BN, the band widens."

Missing NoBN methods: cosine_schedule, SWD, half_lambda, random_mask, PMP-WD. Without these, the BN-narrowing claim is untested. If the NoBN spread is also narrow, the paper's theoretical explanation (BN causes narrowing via scale invariance) collapses, and the narrow spread is simply a property of CIFAR-10 difficulty.

**Impact**: The BN-narrowing narrative -- a central explanatory mechanism -- may be wrong.

### 4. Cumulative Alignment Grid: Not Executed
Phase 4 of the methodology (4 WD strengths x 4 schedules x 2 architectures = 96 configs x 3 seeds) was designed to validate Theorem 2. Zero runs were executed. Without this, Theorem 2 is a theoretical claim with no empirical evidence.

### 5. SGD vs AdamW Comparison: Incomplete
The methodology planned AdamW comparison (Phase 5). Iter_003 has SGD baseline data but iter_006 switched to AdamW. The SGD data shows wider phi spread (0.91% on CIFAR-10 vs 0.49% with AdamW). This is relevant -- it suggests the optimizer (not just BN) affects the certified band width -- but is not discussed in the paper.

## Methodological Concerns

### A. Theory-Experiment Optimizer Mismatch
All theorems assume SGD dynamics. All experiments use AdamW. The PMP-WD derivation uses SGD momentum as costate proxy but applies it to AdamW's first moment, which is qualitatively different (bias correction, second moment scaling).

### B. Best-Epoch vs Final-Epoch Accuracy
The paper reports "best test accuracy" (max across all epochs). This is a common but problematic metric -- it rewards methods that overfit briefly. PMP-WD's advantage shrinks when using final-epoch accuracy (90.12 +/- 0.06% vs constant 89.80 +/- 0.31%).

### C. 3 Seeds Is Minimum
While 3 seeds is standard for CIFAR, the narrow margins (0.49pp total spread) mean the statistical power is very low. The proposal planned 5 seeds for budget-matched comparison. Only 3 were used throughout.

### D. Cross-Iteration Inconsistency
Method ordering reverses between iter_003 and iter_006:
- Iter_003: constant BEST (90.13%), SWD WORST (89.88%)
- Iter_006: PMP-WD BEST (90.29%), constant WORST (89.80%)

Constant dropped 0.33% between iterations; SWD improved 0.26%. This confirms the ordering is noise, but the paper presents iter_006 ordering as meaningful ("PMP-WD achieves the highest mean accuracy... consistent with optimality").

### E. No Hyperparameter Sensitivity Analysis
PMP-WD uses Lambda_max = 5e-4 (same as base WD). What happens at Lambda_max = 1e-3 or 1e-4? The proposal planned rho_high experiments (rho=5.0) which ALL FAILED. Without sensitivity analysis, it is unclear whether PMP-WD's performance is robust or tuned to a single operating point.

## What Was Done Well

1. **Proper statistical testing**: Paired t-tests with Bonferroni correction, effect sizes reported
2. **Diagnostic instrumentation**: Per-epoch logging of V_t, alignment, norms, switching function
3. **Control experiments**: Random mask, half_lambda, no_wd provide useful baselines
4. **Multi-seed evaluation**: All results use 3 seeds with mean +/- std reporting
5. **PMP-WD implementation**: Successfully implemented and produces valid results

## Experiment Priority for Completion

| Priority | Experiment | GPU-hours | Risk | Impact |
|----------|-----------|-----------|------|--------|
| P0 | Add VGG-16-BN CIFAR-10 to paper (data exists) | 0 | None | Medium |
| P0 | Add half_lambda + random_mask to results table | 0 | None | Medium |
| P1 | Complete NoBN ablation (5 more methods x 3 seeds) | 5h | Medium | Critical |
| P1 | Validate Theorem 2 (cumulative alignment test) | 2h analysis | Low | Critical |
| P1 | Add subsumption verification table | 1h analysis | Low | High |
| P2 | ImageNet/ResNet-50 (constant + CWD + PMP-WD x 3 seeds) | 48h | High | Critical |
| P2 | Certified band visualization | 2h analysis | Low | High |
| P3 | SGD replication of iter_006 experiment | 8h | Low | Medium |
