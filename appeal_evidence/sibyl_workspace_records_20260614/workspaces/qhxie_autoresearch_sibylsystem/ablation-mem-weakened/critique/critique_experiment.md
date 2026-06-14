# Critique of Experiments: Competitive Suppression in SAEs

## Overview

The experimental design is thorough in breadth but suffers from critical statistical and methodological flaws that undermine the core claims.

## Critical Issues

### 1. Multiple Comparisons Problem (Critical)

**Setup**: 12 statistical tests across 2 layers, 3 hypotheses (H1, H1b, H2), 2 correlation types (Pearson, Spearman).

**Results**:
- Bonferroni alpha = 0.05/12 = 0.00417
- BH-FDR q < 0.05
- Zero tests survive either correction
- Lowest corrected p = 0.107 (H1b L8 Spearman, BH-FDR)

**Problem**: The paper presents the uncorrected p=0.028 (H1b L8 Pearson) as evidence of absorption effects. This is cherry-picked post-hoc from H1 (the pre-registered primary). After proper correction, there is NO statistical evidence for any hypothesis.

**Verdict**: The statistical analysis is fundamentally underpowered for the claims made.

### 2. Post-Hoc Power Analysis (Critical)

**Claim**: Section 6.2 states "approximately 20% power to detect a medium effect size (r=0.5)."

**Problem**: Post-hoc power analysis is methodologically questionable. Power should be computed before the experiment to determine sample size. Computing power after observing null results to explain them is circular.

**Verdict**: Remove post-hoc power analysis. If power is insufficient, acknowledge it as a design limitation.

### 3. H6 Falsification is Decisive (Critical)

**Setup**: Predict absorption pairs from top-20 decoder correlation neighbors.

**Result**: Precision@20 = 0.0. Fisher exact test p = 1.0.

**Problem**: This is not "near zero" — it is exactly zero. None of the 520 predictions (26 features × 20 neighbors) corresponded to an absorption pair. This is a complete falsification of the primary hypothesis.

**Verdict**: The paper cannot claim the graph is a contribution when it predicts nothing.

### 4. OrtSAE Ablation Comparison is Confounded (Major)

**Claim**: "OrtSAE without penalty (L0~920): 0.230±0.052 vs OrtSAE with penalty (L0~550): 0.247±0.048"

**Problem**: L0 values differ (920 vs 550) — the very confound the paper criticizes others for. L0 is a key determinant of absorption behavior.

**Verdict**: This comparison is invalid. The ablation conclusion is self-contradictory.

### 5. Probe Quality Confound (Major)

**Observation**: Absorption rate correlates with probe F1 at rho=-0.67 (p<0.001).

**Problem**: Low-CMI letters may be inherently harder to probe, causing both low estimated CMI and artificially high absorption rates. The CMI-absorption analysis never controls for probe quality.

**Verdict**: CMI-absorption correlations may be confounded by probe difficulty.

### 6. Dimension Instability in CMI Analysis (Major)

**Observation**: CMI-absorption correlation at d'=10 (rho=-0.383, p=0.059 uncorrected) reverses sign at d'>=20.

**Problem**: Sign reversal across dimensions is a qualitative failure. d'=10 was not pre-registered as the primary dimension.

**Verdict**: Cherry-picking d'=10 is misleading.

## What Works

1. **H5 precision-recall asymmetry**: This is the one replicable finding. The design is sound and the result is consistent.

2. **Cross-layer validation**: Testing at L4 and L8 to check consistency was good practice.

3. **Random SAE baseline**: Comparing trained vs random SAEs is a good idea (though the comparison is confounded).

4. **Delta-corrected metrics**: Baseline subtraction is methodologically sound.

## Recommendations

1. **Pre-register hypotheses** before data collection.
2. **Match L0** in ablation comparisons or acknowledge the confound.
3. **Control for probe quality** in CMI-absorption analysis.
4. **Drop H6 as contribution** — it was falsified decisively.
5. **Compute power before the experiment** for future studies.
