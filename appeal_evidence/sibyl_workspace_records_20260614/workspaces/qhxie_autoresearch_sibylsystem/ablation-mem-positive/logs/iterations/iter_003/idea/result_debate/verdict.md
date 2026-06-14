# Verdict: CV-Based Actionability Decomposition

## Overall Assessment

**Result Quality Score: 7.5 / 10**

The primary finding is **statistically robust and genuine**: CV > 1.0 threshold predicts which absorbed SAE features are steerable, with 47% larger steering effects for high-CV vs low-CV absorbed features across all three steering strengths (p < 0.01, BH-corrected). This partially resolves the Basu et al. actionability paradox by showing absorbed features are not uniformly non-steerable.

---

## Key Conclusion

**High-CV absorbed features (CV > 1.0) are functionally equivalent to non-absorbed features in steering potential** (0.0975 vs 0.102 mean effect — only 4.5% difference), while low-CV absorbed features show substantially reduced steering effects (0.31 at +5 vs 0.52 for high-CV). This CV-based decomposition provides practitioners with a cheap predictor (no steering experiments needed) for which absorbed features are worth targeting.

The phase transition at λ_c = 5e-5 is confirmed with finite-size scaling (R² = 0.951), but the transition is "soft" (chi_ratio = 1.88 < 3.0 threshold), not "sharp" as initially framed.

---

## Critical Concerns

| Concern | Severity | Status |
|---------|----------|--------|
| Cross-architecture validation missing | CRITICAL | Gemma-2 replication PENDING |
| Post-hoc CV threshold | SERIOUS | Data-driven; not prospectively validated |
| Mechanism unknown | MODERATE | Orthogonality ruled out; Fano factor pending |
| Low-CV sampling bias | MODERATE | 106 total, 30 tested (28% of population) |

---

## Action Plan: PROCEED

**Recommendation**: Continue with validation experiments before paper submission.

| Priority | Experiment | GPU Hours | Purpose |
|----------|-----------|----------|---------|
| 1 | Gemma-2-2B replication | 2.0 | Validate generalization beyond GPT-2 |
| 2 | Held-out feature validation | 1.0 | Address post-hoc threshold concern |
| 3 | Fano factor control | 1.0 | Investigate mechanism / rule out magnitude confound |
| 4 | Test all 106 low-CV features | 0.5 | Address sampling bias |

**Total investment: ~4.5 GPU hours**

**Pivot trigger**: If Gemma-2 fails AND held-out validation fails → pivot to descriptive claim only.

**Target venue**: AAAI/EMNLP with current findings; ICLR workshop if Gemma-2 succeeds.

---

## What This Work IS and IS NOT

**The work IS**:
- First evidence that absorbed features decompose into steerable/non-steerable subpopulations
- A cheap predictor (CV) for steering feasibility without requiring steering experiments
- Statistically robust on GPT-2 Small with proper controls and BH correction

**The work is NOT**:
- A confirmed universal finding (cross-architecture validation pending)
- A resolved paradox (only partial evidence for domain-specificity)
- A mechanistic explanation (orthogonality ruled out, mechanism unknown)

---

*Verdict by Result Debate Synthesizer*
*Based on: optimist.md, skeptic.md, strategist.md, methodologist.md, comparativist.md, revisionist.md*