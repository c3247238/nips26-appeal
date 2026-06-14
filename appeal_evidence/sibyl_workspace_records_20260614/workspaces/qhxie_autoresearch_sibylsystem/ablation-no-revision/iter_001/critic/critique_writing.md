# Writing Critique: The Limits of Consistency-Based Activation Energy for Problem-Level Routing

## Overall Assessment

The paper is well-structured with a clear logical arc and commendable honesty about negative results. Writing quality is generally good, with specific numbers and avoided banned patterns. However, critical issues---a formula-data inconsistency, missing bibliography, and contradictory source data---must be resolved before external review.

**Score: 6/10** (would be 7/10 without the critical issues)

---

## Critical Issues

### 1. Ea Formula-Data Inconsistency (Showstopper)

Section 3.2 defines Ea = -ln(c0). The data in analysis_h3.json shows Ea = 9.465 with c0 = 0.10565. But -ln(0.10565) ≈ 2.247, not 9.465. This is mathematically impossible.

The notation.md Usage Note 3 incorrectly states "Ea ≈ 9.47 corresponds to c0 ≈ 0.106"---this is false under the stated formula.

**Impact**: Readers cannot reproduce the analysis. The entire Ea interpretation may be based on a different computation than what the paper describes.

**Fix**: Verify the actual formula in analysis_h3.py and analysis_h2.py. Correct either the formula or the data throughout.

### 2. Missing Bibliography

The paper cites [1] through [11] but has no References section. This is unacceptable for any paper intended for submission.

**Fix**: Add complete bibliographic entries for all citations. Verify that "Li 2026" is a real, citable work.

### 3. H3 Source Data Contradicts Paper

analysis_h3.json labels H3 as "CONFIRMED" with note "Ea is a useful routing signal." The paper correctly interprets the same data as falsified due to AUC=0.436. The 75% threshold pass is a post-hoc artifact, but the source summary ignores this.

**Fix**: Update source data files to reflect the correct interpretation. Add explicit note explaining why threshold test is overridden by AUC/Spearman.

---

## Major Issues

### 4. Table 8 Lacks Quantitative Data

Section 6.2 claims "Execution errors dominate low-Ea failures" but Table 8 only has qualitative descriptions. No counts, proportions, or statistics support this claim. The outline planned "count/proportion" columns but they are absent.

**Fix**: Either perform quantitative error classification or soften the claim.

### 5. Post-Hoc Threshold = Data Leakage

The "optimal threshold" of 9.9999999999 is optimized on the evaluation data. The 75% figure and the 25pp "error floor" derived from it are optimistic upper bounds, not measured quantities.

**Fix**: Clarify that 25pp is an upper bound. Consider reporting a cross-validated threshold for a realistic estimate.

### 6. "Prove" Is Too Strong

Section 5.3: "These values prove Ea cannot predict single-pass solveability." Statistical evidence demonstrates; it does not prove.

**Fix**: Replace "prove" with "demonstrate" or "show."

### 7. Inconsistent Terminology

- "Arrhenius kinetics" vs. "Arrhenius-like kinetics" (glossary prefers the latter)
- Table 7 status capitalization inconsistent ("Confirmed" vs. "**Falsified**")

**Fix**: Standardize throughout.

### 8. Section 3.2 Product Interpretation Is Misleading

"The product P_∞ · k0 has units of samples and can be interpreted as a problem-difficulty parameter: larger values indicate slower convergence toward the ceiling."

This conflates ceiling and rate. For fixed P_∞, larger k0 means slower convergence, but the product itself is not a clean difficulty parameter.

**Fix**: Remove or rephrase carefully.

---

## Minor Issues

### 9. Missing Appendix Content

The outline plans appendices (per-problem fits, Ea distribution, full problem data, extraction audit) but none are present.

**Fix**: Add at minimum the full problem data table.

### 10. Level 5 Ea Near-Zero Variance

σ ≈ 1.9e-6 for Level 5 suggests algorithmic saturation. The paper mentions this but does not explain why.

**Fix**: Investigate and explain the artifact, or report raw c0 values.

### 11. Figure 6 Filename Mismatch

The text references "Figure 6" but the file is `figures/hypothesis_summary.pdf`. The caption content is correct but the filename does not match the figure number.

**Fix**: Rename file to `fig6_routing_signals.pdf` or similar.

---

## What Works Well

1. **Honest negative results**: The paper does not spin the AUC=0.436 finding. It clearly states that Ea-based routing fails.
2. **Strong caveats**: The "critical caveat" in Section 5.1 and the limitations section show intellectual honesty.
3. **Clear structure**: The logical arc (motivation → theory → experiment → falsification → diagnosis) is sound.
4. **Specific numbers**: Claims are backed with specific metrics (R²=0.924, Spearman=0.448, AUC=0.436).

---

## Recommendations for Editor

1. **Block submission** until the Ea formula-data inconsistency is resolved.
2. **Require bibliography** before any external distribution.
3. **Require quantitative Table 8** or softening of the error classification claim.
4. **Standardize terminology** and fix "prove" language.
5. **Add confidence intervals** for key statistics to address the small sample size concern.
