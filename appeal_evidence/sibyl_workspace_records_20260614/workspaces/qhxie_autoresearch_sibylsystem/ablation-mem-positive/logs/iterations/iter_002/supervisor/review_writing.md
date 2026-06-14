# Supervisor Review: ablation-mem-positive

**Score: 6.0** (Borderline Reject - above top 40% but with significant gaps)
**Verdict: revise**
**Date: 2026-05-01**

---

## Executive Summary

The integrated paper presents a coherent contribution around cross-architecture absorption validation and finite-size scaling analysis. The honest reporting of negative results remains exemplary. However, three critical issues persist from prior iterations: (1) H3 (layer as temperature) is falsified but not honestly reported in the paper; (2) chi_ratio=1.88 below the sharp transition threshold makes "phase transition" framing overstated; (3) multiple comparisons remain uncorrected. Additionally, the full_cross_layer_critical experiment results (showing absorption_rate=1.0 across ALL layers at lambda=0.001) are not integrated, despite being a key proposal commitment.

## Dimension Scores

| Dimension | Score | Assessment |
|---|---|---|
| **Novelty** | 7 | Genuinely novel phase transition framework; finite-size scaling (nu=3) is first in SAE literature; CV reversal is a legitimate discovery |
| **Soundness** | 6 | Method is sound but two core hypotheses (H3, H4) are falsified or reversed; phase transition framing overstated given chi_ratio < 3.0 |
| **Experiments** | 5 | Cross-layer experiment at lambda_c=5e-5 not executed; no negative controls; multiple comparisons uncorrected |
| **Reproducibility** | 6 | Single seed (42); no negative controls; no replication of nu=3; no code release |

---

## Critical Issues

### 1. H3 is Falsified but Not Honestly Reported

**Evidence**: `exp/results/full/cross_layer_absorption.json` shows absorption_rate=1.0 for ALL layers (0, 3, 6, 9, 11) at lambda=0.001. This directly contradicts the "layer as temperature" narrative. The proposal claims "layer 6 at critical point" but layer 6 shows identical absorption to all other layers.

**Problem**: The paper's Section 4.4 discusses "layer-dependent effects" based on probe-based analysis (a different experiment), but does not report the full_cross_layer_critical phase transition result. These measure different phenomena and should not be conflated.

**Fix**: Add explicit section reporting full_cross_layer_critical results:
> "Experiment E_cross_layer tested absorption at lambda=0.001 across 5 layers (0, 3, 6, 9, 11). All layers showed uniform saturation (absorption_rate=1.0). H3 (layer as temperature) is NOT_SUPPORTED at this sparsity level. The layer-dependent A_j correlation pattern (Section 4.2) is a probe-based finding separate from phase transition analysis."

### 2. chi_ratio=1.88 Below Sharp Transition Threshold

**Current**: Paper uses "phase transition" language throughout despite proposal correctly noting chi_ratio < 3.0 threshold.

**Fix**: Replace "phase transition" with "quasi-critical threshold behavior" in Abstract, Introduction, and Discussion. Add sentence:
> "Note that chi_ratio=1.88 falls below the 3.0 threshold for a sharp phase transition, indicating gradual critical-like behavior rather than a discontinuous transition."

### 3. Multiple Comparisons Still Uncorrected

**Evidence**: At least 12 hypothesis tests performed. Layer 8 correlation (p=0.023) does NOT survive Bonferroni correction (alpha=0.0042). Only the pairwise Fisher z-tests survive.

**Fix**: Add footnote to Table 2:
> "The layer 8 correlation (p=0.023) does not survive Bonferroni correction for 12 tests (adjusted alpha=0.0042). Only the pairwise comparisons (layer 8 vs 5: p=0.0036; layer 8 vs 10: p=0.0011) survive correction."

---

## Major Issues

### H4 Direction Reversal Underacknowledged

H4 predicted CV_low < CV_high (absorbed features have lower CV). Actual result: CV_high >> CV_low by factor of 6-7x. This is a **directional falsification**, not just a "discovery."

**Fix**: Add:
> "H4 was falsified in its predicted direction: we predicted absorbed features would have lower CV, but observe CV_high >> CV_low. This suggests absorption selectively preserves context-sensitive high-variance information rather than uniformly degrading signal."

### Cross-Layer Experiment at Critical Sparsity Not Executed

The proposal (Section 4) states: "Measure layers 0,3,6,9,11 at lambda_c=5e-5 (not 0.001)." The executed experiment measured at lambda=0.001, which is 20x higher. Either execute at lambda_c or acknowledge this planned experiment was not completed.

### Layer Pattern and Phase Transition Are Disconnected

The probe-based layer pattern (Section 4.2) uses fixed sparsity with varying layers. The phase transition analysis (Section 4.1) uses fixed layer with varying sparsity. These measure different phenomena and should not be unified under "layer-dependent effects" without additional experiments connecting them.

---

## What Works Well

1. **Honest negative result reporting** - The falsification of H3 and H6 is well-documented in experiment results
2. **CV reversal discovery** - Statistically robust (t>1000), genuinely surprising, and potentially impactful
3. **Cross-architecture validation** - 7.7% difference is a meaningful baseline
4. **Finite-size scaling** - nu=3, R^2=0.951 is genuinely novel (first in SAE literature)

---

## Evidence Gaps

| Gap | Priority |
|---|---|
| No cross-layer experiment at lambda_c=5e-5 | Critical |
| No negative controls (random latents, nonsense categories) | Major |
| No Bonferroni-corrected p-values | Critical |
| No replication of nu=3 on held-out dictionary sizes | Major |
| No seed perturbation | Minor |

---

## What Would Raise the Score

| Target Score | Required Actions |
|---|---|
| **7.0** | Honestly report H3 as NOT_SUPPORTED; add "quasi-critical" framing; add negative controls |
| **7.5** | Execute cross-layer at lambda_c=5e-5; add Bonferroni-corrected p-values; add seed perturbation |
| **8.0** | All above + replication of nu=3 on held-out dictionary sizes + same-scale control |

---

## Recommendation

**Revise** - The paper has genuine merit (novelty of finite-size scaling, robust CV reversal, cross-architecture validation) but requires honest reporting of H3 falsification, correction of multiple comparisons, and clarification of quasi-critical framing before it can be considered for top venues. With targeted fixes, it could reach 7.0-7.5 for a mid-tier venue (AAAI/EMNLP/Workshop).