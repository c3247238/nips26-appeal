# Supervisor Review: Iteration 11

**Paper:** The Phi Invariance Conjecture: When Dynamic Weight Decay Methods Are Equivalent Under Adaptive Optimizers
**Score:** 6.75 / 10 (previous: 6.5)
**Verdict:** CONTINUE (not yet conference-publishable; 7.0+ threshold not met)

---

## Executive Summary

Iteration 11 achieves a milestone: **all data integrity issues are resolved**. The three chronic figure-table-text inconsistencies that plagued iterations 7-10 are fixed and visually verified. The paper is now internally consistent. However, the fundamental experimental limitations remain unchanged, preventing the score from reaching the 7.0 publishable threshold.

## What Improved (iter_010 -> iter_011)

### 1. Figure 8 (theorem2_validation.png) - FIXED (+0.25)
- **Before:** Contained PMP-WD (a method not in the study) -- desk-reject-level error
- **After:** Shows exactly 7 methods in legend (Constant, Cosine, CWD, SWD, Half-lambda, Random, No WD)
- **Verified:** N=21 data points (7 methods x 3 seeds), correlations rho=-0.161/p=0.485 and rho=0.107/p=0.645 match paper text exactly

### 2. Figure 5 (diagnostic heatmap) - FIXED (last critical data bug)
- **Before (iter_010):** half_lambda BEM = 0.000 while Table 6 said 0.500
- **After:** half_lambda BEM = 0.500, matching Table 6 perfectly
- **Impact:** Eliminates the last figure-table contradiction

### 3. Title - FIXED (overclaim removed)
- **Before:** "Why Dynamic Weight Decay Methods Are Equivalent..."
- **After:** "When Dynamic Weight Decay Methods Are Equivalent..."
- **Impact:** Removes causal claim the paper cannot support; "When" correctly frames it as empirical characterization

### 4. Section 5.7 Alignment Analysis Text - CORRECTED
- N=21 (7x3), rho_cumul=-0.161 p=0.485, rho_worst=0.107 p=0.645
- Correctly framed as null results (improved from iter_010's directional claims)

## Visual Verification of All Figures

| Figure | File | Content | Table Match | Issues |
|--------|------|---------|-------------|--------|
| Fig 1 | fig1_taxonomy.png | 4 axes, methods placed correctly | N/A | Clean |
| Fig 2 | fig2_accuracy_comparison.png | 7 methods, bar chart with error bars | Table 2 values match | Clean |
| Fig 3 | multi_arch_comparison.png | 7 methods, spreads 0.25pp/0.16pp | Tables 2,5 match | Clean |
| Fig 4 | fig3_bem_vs_accuracy.png | 7 methods, slopes labeled | Table 2 consistent | Clean |
| Fig 5 | fig4_diagnostic_heatmap.png | 7 methods, BEM values correct | Table 6 match (half_lambda=0.500) | **FIXED** |
| Fig 6 | fig6_sgd_vs_adamw_norms.png | 7 methods each, values labeled | Text Section 5.6 match | Clean |
| Fig 7 | fig5_weight_norm_trajectories.png | 7 methods converging | Text Section 5.6 match | Clean |
| Fig 8 | theorem2_validation.png | 7 methods, N=21, correlations labeled | Text Section 5.7 match | **FIXED** |

**Orphan:** certified_band.png exists in figures/ with PMP-WD in legend but is NOT referenced in paper.md. Low risk but should be cleaned up.

## What Still Prevents 7.0+

### Critical: No ImageNet Experiments (flagged 6+ iterations)
The paper claims insights about "sufficiently overparameterized problems" and "modern deep learning" but all evidence is from CIFAR-10/100 with models of 270K-15M parameters. By 2026 venue standards, this is the most predictable reviewer objection. The project spec mandates ImageNet. This is the single highest-ROI investment: 9 runs on ImageNet-100 would raise the score by ~0.75 points.

### Major: Insufficient Statistical Power for Null-Result Claims
N=3 seeds gives ~15-20% power to detect 0.5% effects. TOST equivalence is confirmed for only 6/12 comparisons at delta=1.0%. The paper claims "all methods are statistically equivalent" but half the comparisons are underpowered, not confirmed. A null-result paper faces a higher evidentiary bar than a positive-result paper.

### Major: CSI as a Claimed Contribution
CSI does not predict accuracy (rho < 0.3, p > 0.3 by the paper's own admission). Its component weights (0.4, 0.3, 0.3) are arbitrary with inadequate sensitivity analysis. Listing it as contribution #2 in Section 1.3 alongside BEM and AIS overstates its utility. Demoting CSI to "exploratory diagnostic" is a text-only fix.

### Major: NoBN Hypothesis Undertested
Section 6.2 builds a narrative around BN enabling Phi Invariance under SGD, but only 2 of 7 methods were tested without BN. The existing data (constant: 87.74%, CWD: 87.63%, spread 0.11pp) actually suggests invariance may hold WITHOUT BN -- which would undermine the BN-mechanism narrative but strengthen the overall Phi Invariance story. Running all 7 methods without BN (21 runs, ~3 GPU-hours) would definitively resolve this.

### Major: Abstract Implies Full Factorial Design
"105 experiments across 7 methods, 2 optimizers, 2 architectures, 2 datasets" reads as 7x2x2x2x3=168 runs. The actual non-factorial design (84 ResNet-20 + 21 VGG-16-BN) should be stated explicitly.

## Dimension Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | 7.0 | Phi framework is a genuine unifying contribution; BEM and AIS are useful diagnostics; the conjecture is well-formulated |
| Soundness | 6.5 | All data integrity issues fixed; CSI non-predictive; TOST underpowered for half the comparisons; NoBN hypothesis undertested |
| Experiments | 6.0 | 105 runs across 2 optimizers, 2 architectures, 2 datasets is decent coverage BUT no ImageNet, N=3 seeds, non-factorial VGG-16-BN |
| Reproducibility | 7.0 | Clear method descriptions, unified codebase, explicit hyperparameters, seeds reported |

## Priority Action List (Ranked by ROI)

1. **ImageNet-100** (9 runs, ~6-8 GPU-hours) -> score to ~7.5
2. **Full NoBN ablation** (21 runs, ~3 GPU-hours) -> validates/revises BN mechanism
3. **2 more seeds** for 4 key comparisons (8 runs, ~2 GPU-hours) -> TOST power
4. **Demote CSI** from contribution to diagnostic (text-only, 30 min) -> soundness +0.25
5. **Fix abstract** non-factorial disclosure (text-only, 10 min)
6. **Delete or regenerate certified_band.png** (10 min)

Total compute: ~13 GPU-hours. Total text work: ~2 hours.

## Score Trajectory

| Iteration | Score | Key Changes |
|-----------|-------|-------------|
| iter_005 | 5.5 | Initial NoBN data collected |
| iter_007 | 5.0 | Figure-table inconsistencies introduced |
| iter_008 | 6.5 | Partial figure fixes |
| iter_009 | 6.5 | More figure fixes |
| iter_010 | 6.5 | Heatmap BEM bug remains; title overclaim |
| **iter_011** | **6.75** | **All data integrity fixed; title fixed** |

## Bottom Line

The paper is now **internally consistent** for the first time since iter_007. This is meaningful progress. But internal consistency is a necessary condition, not a sufficient one. The path to 7.0+ runs through ImageNet experiments and stronger statistical evidence, not through more text revision. The next iteration should be experiment-focused.
