# Pilot Summary: task_F1_theory_revised

**Status:** GO  
**Date:** 2026-04-13  
**Mode:** PILOT

## Pilot Criterion
"The derivation reaches a conclusion without circular arguments. Label explicitly if any step is a heuristic. Also generate Figure 1 as a PDF."

## Outcome: PASS

## What Was Produced

### Theory Document: `exp/results/full/F1_theory_revised.md`

The revised theory document contains:

1. **Part I: Proposition 1 (maintained from iter_001)** — λ > sin²(θ_{p,c}) absorption threshold.
   Non-circular: compares two explicit candidate solutions. Correct proof structure preserved.

2. **Part II: Revised geometric prediction** — EDA as the intra-feature fingerprint of absorption.
   - Explains why the original prediction (large decoder-decoder cosine) was wrong in the post-convergence setting
   - Derives why the absorbed child encoder is pulled toward the parent decoder direction (Proposition 2)
   - Proposition 2 explicitly labeled as **MECHANISTIC CONJECTURE** (conditions C2 and C3 require empirical verification)

3. **Part III: Revised Impossibility Theorem** — explicitly acknowledges the unresolved tension
   between EDA magnitude predictions and empirical observations. No false proof claims.

4. **Part IV: Architectural mitigations** through EDA lens.
   - Matryoshka: hierarchical slots prevent dissociation
   - OrtSAE: larger decoder angles reduce absorption pressure
   - ATM SAE: lower parent feature cost reduces encoder pull

5. **Part V: Pilot assessment** — what theory correctly predicts vs. open tensions.

### Figure 1: `exp/results/full/fig1_eda_method.pdf` and `.png`

Three-panel diagram:
- **Left:** Non-absorbed feature (encoder ≈ decoder, EDA ≈ 0)
- **Middle:** Formula and proposition insets
- **Right:** Absorbed feature (encoder pulled toward parent decoder, large angle vs. child decoder)

## Key Theoretical Contributions

| Claim | Status |
|-------|--------|
| Proposition 1 (energy comparison) | **PROVEN** — non-circular |
| Co-occurrence frequency cancels | **PROVEN** |
| Proposition 2 (encoder pull) | **MECHANISTIC CONJECTURE** — explicitly labeled |
| EDA magnitude prediction | **UNRESOLVED TENSION** — explicitly documented |
| L10 failure hypothesis | **HYPOTHESIS** — testable in experiments |

## GO/NO-GO Assessment

**GO** — Pilot criterion satisfied. Theory document is internally consistent, non-circular,
has explicit labels for all heuristic steps, and Figure 1 is generated as PDF.
The unresolved empirical tension is a feature, not a bug: it identifies what experiments B1_pairwise_eda
and B1_eda_decomposition must resolve.
