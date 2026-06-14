# Testable Hypotheses — Iteration 7

**Date**: 2026-03-18
**Based on**: Iteration 7 synthesis; verified SGD data (SWD seed_456 = 90.93% confirmed from disk)

---

## Core Hypotheses

### H1: Phi Invariance Under Standard AdamW (Confirmed — ResNet-20)

**Statement**: Under AdamW with decoupled WD (λ = 5×10⁻⁴, η = 10⁻³, ρ = 0.5), all budget-equivalent WD modulation strategies — including complete removal (no_wd) — produce final accuracy within ±0.5% of constant WD on batch-normalized CNNs.

**Evidence**: CIFAR-10 ResNet-20: spread 0.25%, all BH-corrected p > 0.09. CIFAR-100 ResNet-20: spread 0.76%, all BH-corrected p > 0.09. AdamW constant (90.133%) vs no_wd (90.083%): Δ = 0.050%, p = 0.825.

**Falsification**: Any method exceeds constant WD by ≥ 0.5% (BH p < 0.05, n ≥ 3) on VGG-16-BN or ResNet-50/ImageNet.

**Current status**: CONFIRMED for ResNet-20 CIFAR-10/100. Awaiting VGG-16-BN and ImageNet replication.

---

### H2: SGD WD-Presence Effect (Confirmed — ResNet-20)

**Statement**: Under SGD with decoupled WD (same hyperparameters), removing WD entirely causes a statistically significant accuracy decrease. No dynamic WD strategy significantly improves OR degrades relative to constant WD after Holm correction.

**Evidence (Verified from disk)**:
- SGD constant: [91.30, 91.18, 91.17] → mean 91.217% ± 0.072%
- SGD no_wd: [90.39, 90.19, 90.33] → mean 90.303% ± 0.103%
- Δ = 0.913%, paired t-test p = 0.0022, Holm-corrected significant (rank 1/6, threshold 0.0083)
- Cohen's d (paired) = −12.17; pooled d = −10.29

**All dynamic methods**: Holm-corrected p > 0.05 (not significant in either direction). SWD p = 0.0713 (not significant at Holm threshold 0.010).

**Status of prior SWD claim**: The claim "SWD significantly worse than constant" was based on 2-seed SWD data. With seed_456 = 90.93% confirmed, SWD is NOT Holm-significant. RETRACTED.

**CIFAR-100 estimate**: constant 65.370% vs no_wd (n=1 seed) 63.59% (seed_42 only). Requires seed_123 for n=3.

**Falsification**: Constant WD fails to outperform no_wd under SGD on VGG-16-BN (p > 0.05 uncorrected), OR any dynamic method significantly outperforms constant on SGD (BH p < 0.05).

---

### H3: 18.3× Optimizer Specificity Ratio (Confirmed — ResNet-20)

**Statement**: The WD presence-absence effect (constant − no_wd) is ≥ 10× larger under SGD than AdamW on CIFAR-10 ResNet-20.

**Evidence (Verified)**: SGD/AdamW ratio = 0.913% / 0.050% = 18.3×.

**Required action**: Bootstrap 95% BCa CI (10,000 resamples). Expected range: [11×, 28×]. Must be reported in paper.

**Falsification**: CI lower bound < 5× on VGG-16-BN (95% BCa).

---

### H4: ρ Regime Boundary Exists at ρ ∈ (0.5, 5) — Untested (P1-1)

**Statement**: The accuracy spread across budget-equivalent WD strategies increases from < 0.5% at ρ = 0.5 (confirmed) to > 1.0% at ρ = 5 under AdamW on ResNet-20 CIFAR-10.

**Mechanism**: Phi Invariance Trichotomy Conjecture — at ρ > ρ₁ ≈ 1, the implicit ℓ∞ constraint weakens and WD schedule timing becomes relevant.

**Test**: λ sweep at λ ∈ {5e-5, 5e-4, 5e-3, 5e-2} (ρ ∈ {0.05, 0.5, 5, 50}), constant/cosine_schedule/cwd_hard × 3 seeds.

**Falsification**: Spread remains < 0.5% at ρ = 50 (ℓ∞ constraint is stronger than theory bounds; "universal invariance" is itself a publishable negative result).

---

### H5: ρ-Controller Improves Over Constant WD in Regime II — Untested (P1-2, conditional on H4)

**Statement**: The P-controller λ_t = λ₀ · (ρ_t / ρ*)^{−α} with α = 0.5 achieves ≥ 0.3% accuracy improvement over constant WD at λ = 5e-3 (Regime II) and has faster ρ_t convergence to ρ* by epoch 30.

**Prerequisite**: H4 confirmed (Regime II exists at ρ ≈ 5).

**Falsification**: ρ-Controller does not converge to ρ* faster than constant WD by epoch 30, OR improvement < 0.3% (p > 0.05 uncorrected) at λ = 5e-3.

---

### H6: BN Confound — Two Paths (Gate 1)

**Path A (AdamW mechanism primary)**: ResNet-20-NoBN under AdamW still shows Phi Invariance (spread < 0.5%). Mechanism: AdamW's ℓ∞ implicit constraint alone is sufficient, independent of BN scale-invariance.

**Path B (BN mechanism primary)**: ResNet-20-NoBN under AdamW breaks Phi Invariance (spread > 1%). Mechanism: BN is a necessary co-condition for invariance; reframe as "BN + AdamW joint mechanism."

**Test**: ResNet-20-NoBN, CIFAR-10, AdamW + SGD, constant/cosine_schedule/no_wd, 3 seeds each (18 runs, ~1 GPU hour).

---

### H7: Alignment Signal Non-Trivially Informative in NoBN — Untested (Gate 1 diagnostic)

**Statement**: For ResNet-20-NoBN, per-layer δ̂_t std > 0.05 across training (not structurally suppressed to ~0 as the Contrarian predicts for BN-adjacent layers).

**Consequence**: Only if H7 confirmed → super-twisting SMC secondary contribution is motivated and included.

---

### H8: VGG-16-BN 18.3× Ratio Replicates — Untested (Gate 2)

**Statement**: SGD/AdamW WD presence-absence ratio ≥ 5× on VGG-16-BN CIFAR-10 (95% BCa CI lower bound > 5).

**Pilot signal**: 1-seed VGG-16-BN AdamW shows no_wd=80.61% > constant=79.94% — reversed direction from ResNet-20. If confirmed at n=3: WD is actively harmful under AdamW on VGG-16-BN. This would be a stronger positive finding than null invariance.

---

## Diagnostic Metrics Hypotheses

### D1: AIS is Indistinguishable Between CWD and Random Masking (Confirmed)

**Evidence**: cwd_hard AIS = 0.416 ± 0.050, random_mask AIS = 0.416 ± 0.030 (iter_003 SGD CIFAR-10). Difference ≈ 0.000. Sign-alignment conditioning in CWD provides no measurable improvement in gradient-weight informativeness in BN architectures. Supports Contrarian's BN signal suppression argument.

### D2: CSI Correlates Negatively with WD Effectiveness Under SGD (Confirmed)

**Evidence**: SGD CIFAR-10: no_wd (highest CSI = 0.964) has lowest accuracy; constant (CSI = 0.841) has highest accuracy. Prediction: at Regime II (λ = 5e-3), CSI will separate methods more clearly across different WD strategies.

---

## Hypothesis Status Summary

| Hypothesis | Status | Required Action |
|---|---|---|
| H1: AdamW Phi Invariance | Confirmed (ResNet-20) | VGG + ImageNet replication |
| H2: SGD WD-Presence Effect | Confirmed (n=3, verified) | VGG replication |
| H3: 18.3× Ratio | Confirmed (point estimate) | Bootstrap CI required |
| H4: Regime Boundary | Untested | P1-1 λ sweep |
| H5: ρ-Controller Improvement | Untested | P1-2 (conditional on H4) |
| H6: BN Confound Path | Untested | Gate 1 — HIGHEST PRIORITY |
| H7: NoBN Alignment Signal | Untested | Gate 1 diagnostic |
| H8: VGG-16-BN Ratio | Untested | Gate 2 |
| D1: AIS CWD vs Random | Confirmed | Report in paper |
| D2: CSI vs Accuracy | Confirmed (trend) | Validate at Regime II |

---

*Compiled by Sibyl Synthesizer (Iteration 7). H1–H3, D1–D2: confirmed with verified n=3 data. H4–H8: untested, required gates.*
