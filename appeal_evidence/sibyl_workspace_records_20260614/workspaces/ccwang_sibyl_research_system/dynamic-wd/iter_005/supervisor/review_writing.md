# Supervisor Review: "When Does Dynamic Weight Decay Help? A Stability-Optimal Control Theory of Dynamic Weight Decay"

**Reviewer**: Independent Supervisor (NeurIPS-calibrated)
**Date**: 2026-03-19
**Iteration**: 5 (post VGG-16-BN completion, rho_low/NoBN expansion)
**Score**: 7.0 / 10 (Weak Accept -- interesting direction, notable gaps)
**Verdict**: continue

---

## Executive Summary

The paper develops a stability-optimal control theory for WD scheduling, making a genuine theoretical contribution through three theorems: (1) binary masking suboptimality via alignment-stability tradeoff, (2) layer-wise CSI bound for per-parameter WD variation, and (3) PMP-WD optimal state-feedback law derived from dual routes (stochastic PMP and RG beta function). The central empirical finding -- constant WD is optimal at standard rho in BN networks -- is robust and well-executed across 122 completed 200-epoch runs spanning 2 architectures, 2 datasets, 2 optimizers, and multiple rho regimes. The paper's statistical honesty (explicit TOST power limitations, comprehensive data gap table, falsifiable predictions) is exemplary and above community norms.

However, three structural weaknesses prevent a higher score:

1. **PMP-WD is unvalidated**: The paper's primary algorithmic contribution (Theorem 3) is derived but explicitly deferred to "future work." This is the paper's most serious gap.

2. **rho-high regime is untested**: The regime diagram (Figure 1) shows three zones, but only the "inhibition" zone has comprehensive data. The "differentiation" zone (rho > 2.0), where the theory makes its most interesting prediction, has only a 5-epoch pilot.

3. **Statistical power insufficient for null-result claims**: N=3 seeds provides only 15-20% TOST power at the delta=0.5% margin practitioners care about.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 7/10 | The stability-optimal control framing is genuinely new. No prior work derives a state-feedback WD law from optimality conditions. The dual derivation (PMP + RG) strengthens Theorem 3. Proposition 1 (alignment noise constraint) is a useful design principle. However, the Phi modulator taxonomy is more systematization than theory (Proposition 1 in the framework is trivially true), and the core null result partially overlaps with D'Angelo et al. (2024) and Kosson et al. (2023). |
| **Soundness** | 7/10 | Theorems 1-3 are well-stated with appropriate caveats (linearized dynamics, near-steady-state approximation). The proof sketches are convincing. The dual derivation with 15% agreement in the moderate-alignment regime is a strong validation signal. Weaknesses: CSI weights are unjustified with a missing appendix, the u_t vs g_t ambiguity persists, and BEM has a known implementation gap for half_lambda. |
| **Experiments** | 6/10 | The experimental base is substantial (122 completed runs) and well-designed (fixed hyperparameters, multi-seed, multi-optimizer). VGG-16-BN completion (all 7 methods, Phi spread = 0.16%) is a significant addition. However: PMP-WD is unvalidated, rho-high is untested, matched-rho SGD is incomplete, and N=3 provides insufficient TOST power. The paper also underreports its own data (rho-low CWD has 3 seeds, not 1; NoBN has 3 methods, not 2). |
| **Reproducibility** | 7/10 | Training configs, hyperparameters, and seed values are documented. Per-epoch diagnostics (1,400 scalars per epoch) enable replication. Weaknesses: Appendices B.1-B.4 (theorem proofs) and A.3 (CSI sensitivity) are referenced but may not exist; BEM pipeline computes wrong value for half_lambda; Cohen's d formula is mislabeled. |

---

## Cross-Validation Findings

All VGG-16-BN numbers in Table 3 match raw summary.json data exactly:
- constant: 92.05 +/- 0.06 (seeds: 92.00, 92.03, 92.12) -- MATCH
- half_lambda: 92.15 +/- 0.13 (seeds: 92.18, 92.00, 92.26) -- MATCH
- Phi spread: 0.16% -- MATCH

**Data staleness discovered**: rho-low CWD has 3 completed seeds (89.81, 90.09, 89.96), mean 89.95 +/- 0.14. The paper states "CWD: 90.09 (1 seed only)." The actual rho-low Phi spread is 0.18% (constant 90.13 vs CWD 89.95), not the ">= 0.04%" in the paper. This is a factual error that understates the paper's evidence.

NoBN has 3 methods partially complete: constant (87.74 +/- 0.21, 3 seeds), CWD (87.62 +/- 0.12, 3 seeds), no_wd (87.79, 1 seed). The paper claims "2 of 7 methods completed."

Total completed runs: 42 (AdamW ResNet) + 42 (SGD ResNet) + 21 (VGG) + 6 (rho_low) + 7 (NoBN) + 4 (matched-rho) = 122, not the "105" in the abstract.

---

## Critical Issues (2)

### 1. PMP-WD Unvalidated (experiment, critical)

The paper's Contribution #3 is "Theorem 3 (PMP-Optimal WD) with dual derivation." The algorithm formula is lambda*(t) = clip(kappa * (rho* - rho_hat_t)+, 0, lambda_max). Implementation is estimated at ~30 LOC. Yet the paper explicitly defers evaluation: "PMP-WD is a theoretical contribution; empirical evaluation is deferred to future work."

A reviewer will ask: if PMP-WD is principled enough to deserve a theorem, why not run it? At rho=0.5, the theory predicts PMP-WD matches constant WD (a confirmation). At elevated rho, the theory predicts PMP-WD outperforms (the interesting case). Either result strengthens the paper. Estimated cost: ~2 GPU-hours.

**Impact on score**: This alone caps the paper at 7.5. Validating PMP-WD at even one rho setting would raise the score by 0.5 points.

### 2. rho-High Regime Untested (experiment, critical)

Figure 1 shows three regime zones (inhibition, transition, differentiation), but the "differentiation" zone (rho > 2.0) has only a 5-epoch pilot at rho=5.0 showing 77.69%. The paper attributes this to "training instability" but does not investigate further. The regime diagram's most interesting zone -- where the theory predicts dynamic WD methods should start outperforming constant WD -- has no validated data.

If rho-high genuinely cannot train stably, this is itself an important boundary finding. If it is a hyperparameter issue (fixable with warmup or gradient clipping), the data gap is unnecessary.

---

## Major Issues (4)

### 3. Statistical Power Insufficient for Null-Result Claims

N=3 seeds provides MDE ~0.7% at 80% power. TOST at delta=0.5% has 15-20% power. The paper acknowledges this but still claims methods are "statistically indistinguishable." Adding 2 seeds (N=5) for 4 key methods would raise TOST power to ~55%.

### 4. Paper Underreports Available Data

Multiple sections report stale data counts that understate the evidence. This should be a simple fix that makes the paper factually accurate and slightly stronger.

### 5. Run Count Discrepancy

Abstract says 105, actual count is 122. Update to the accurate number.

### 6. SGD/AdamW Confound Unresolved

The 3.7x spread ratio headline conflates optimizer mechanism with 100x rho difference. Matched-rho SGD (2 seeds constant, 1 seed CWD) is insufficient to disentangle. 5 additional runs would resolve this.

---

## Minor Issues (3)

7. **u_t vs g_t ambiguity**: CWD's sign alignment uses u_t (preconditioned) but the framework takes g_t. Define u_t explicitly.

8. **CSI weights unjustified**: Appendix A.3 referenced but does not exist. CSI has no within-architecture predictive value (r_s = 0.03 for ResNet-20). Consider demoting.

9. **CIFAR-100 SGD table missing**: The 1.71% Phi spread is cited but no dedicated analysis table with p-values is provided.

---

## What Would Raise the Score

| Action | Estimated Cost | Score Impact |
|--------|---------------|-------------|
| Implement + run PMP-WD at rho=0.5 and rho=2.0 | ~2 GPU-hours | +0.5 to +1.0 |
| Complete matched-rho SGD (5 additional runs) | ~2 GPU-hours | +0.3 |
| Add 2 seeds for 4 key methods (16 runs) | ~4 GPU-hours | +0.3 |
| Update stale data, fix run count | 0 compute | +0.1 |
| **Total** | **~8 GPU-hours** | **+1.0 to +1.5** |

With all four actions, the paper would reach 8.0-8.5 (Accept territory).

---

## What the Paper Does Well

1. **Statistical honesty is exemplary.** Explicit TOST power limitations, MDE quantification, data gap table with prioritized filling order. This is above community norms for both positive and null-result papers.

2. **Dual derivation of Theorem 3.** Having PMP and RG beta function independently converge to the same functional form (within 15% in the moderate-alignment regime) is strong theoretical evidence.

3. **Comprehensive data gap table (Table 6).** Explicitly listing what is missing, why it matters, and what it would resolve is rare and reviewer-friendly.

4. **VGG-16-BN completion.** The full 7-method x 3-seed VGG result (Phi spread = 0.16%) is the strongest evidence for cross-architecture invariance.

5. **Falsifiable predictions.** Section 3.3's Theorem 1 Corollary makes explicit, testable predictions about which regimes should break invariance.

---

*Review generated by sibyl-supervisor on 2026-03-19. Score: 7.0. Verdict: continue. The paper has strong theoretical foundations and honest reporting but needs PMP-WD validation and rho-high data to reach acceptance threshold.*
