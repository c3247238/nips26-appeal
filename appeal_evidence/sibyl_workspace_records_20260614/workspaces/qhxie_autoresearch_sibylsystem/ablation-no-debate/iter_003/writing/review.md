# Writing Quality Review

## Summary

This paper investigates feature absorption in Sparse Autoencoders (SAEs)—a phenomenon where child features substitute for parent features in sparse representations—and presents a 2x2 factorial decomposition identifying encoder alignment as the primary driver, with decoder training playing a configuration-dependent regulatory role. The paper establishes that the sensitivity-absorption Pareto frontier degenerates in full experiments, and conducts preliminary safety-critical feature analysis on Gemma Scope SAEs. The core mechanistic finding (encoder-driven absorption) is solid and well-evidenced by the factorial design. Remaining issues include a critical abstract/body inconsistency on pilot vs. full experiment numbers, figure reference errors (Figure 2 referenced in Section 2.1 for conceptual illustration but Figure 2 is the factorial bar chart), a circular sentence in Section 5.2, and a pilot-only R² still appearing in the abstract.

---

## Detailed Assessment

### Structural Coherence: 7/10

The paper follows a standard structure: Abstract → Introduction → Background → Method → Results → Discussion → Conclusion, and the logical flow is generally sound. The revision correctly reports full experiment data throughout the Results section.

Issues:

1. **Figure 2 referenced in wrong place (Section 2.2, line 70)**: "Figure 2 illustrates the absorption phenomenon conceptually" appears in Section 2.2 (Background), but Figure 2 is the H_Mech factorial bar chart showing Conditions A/B/C/D. The conceptual illustration of absorption is Figure 1. This is a reference error that will confuse readers.

2. **Figure 1 not properly introduced in Introduction**: The outline specifies "Figure reference: Figure 1 (teaser - key result showing encoder vs decoder contribution)" in the Introduction contributions roadmap, but the paper only mentions Figure 1 in passing at line 46 ("when and why absorption occurs (Figure 1)"). This is too late—the figure should be previewed in the contributions roadmap.

3. **Abstract-body tension on B > D**: The abstract states "Condition B produces absorption rates of 0.076 versus 0.017 for full training (Condition D)" but these are seed-42 pilot numbers. The full 5-seed experiment found B=0.055 vs D=0.017 (delta=0.037). The abstract should either report full experiment numbers or clearly label these as pilot results. Reviewers will notice the discrepancy.

4. **Section 5.2 contains a circular sentence**: "Our empirical results reveal that sensitivity and absorption trade off approximately as sensitivity decreases as absorption increases" (around line 362) is tautological. The same paragraph contains "approximately as" which is unclear phrasing.

### Notation & Terminology Consistency: 8/10

The paper is largely consistent with notation.md and glossary.md. Most canonical terms are used correctly:

- "Multi-child proportional absorption" defined and used correctly
- "Pareto frontier" American English spelling, consistent
- "Feature" usage generally correct (not "neuron" or "unit")
- $A_{multi}(p)$ notation used in Section 2.4 and Section 3.6
- $R^2$ for coefficient of determination
- H_Mech, H_Comp, H_Pareto, H_Safe hypothesis names consistent

Issues:

1. **"Cross-model validation" vs. "held-out validation"**: Section 4.5 heading says "Cross-Model Validation" while Table 6 caption says "Held-Out Validation"—mild inconsistency for the same result.

2. **"Uncertainty relation" undefined**: Section 5.2 uses "sensitivity-absorption uncertainty relation" but this term does not appear in glossary.md or notation.md. If referencing an "uncertainty relation," it should be explicitly introduced in the body where the frontier is discussed.

No critical violations that would fundamentally confuse a reader.

### Claim-Evidence Integrity: 7/10

The revision correctly replaced pilot data with full experiment data in Tables 2-4 and throughout the Results section. Most claims are now supported by appropriate evidence.

Issues:

1. **Abstract reports pilot numbers, body reports full experiment numbers**: The abstract (line 7-8) says "Condition B produces absorption rates of 0.076 versus 0.017 for full training (Condition D)"—these are seed-42 pilot results. The full 5-seed results are B=0.055, D=0.017 (delta=0.037). The abstract should either report the full experiment numbers (0.055 vs 0.017) or explicitly label these as "pilot experiment (seed 42)." This inconsistency will catch a reviewer's eye.

2. **Section 5.1 mechanistic speculation**: "During training, the encoder learns to align child feature directions with parent features when they co-activate frequently" (line 352)—the factorial design shows correlation (encoder training → absorption) but does not directly measure co-activation frequency. This mechanistic speculation is reasonable but should be labeled as inference rather than established fact.

3. **GPT-2 Small validation (Section 4.5)**: Single held-out split with no error bars or statistical tests. The text says "The same absorption pattern replicates on real model SAEs" but without statistical validation this is an observation, not a confirmed result.

4. **Pilot R² = 0.963 still referenced**: This R² appears in the abstract and Section 5.2 but the actual full experiment found degenerate results (absorption = 0 across all L0 levels). The pilot R² = 0.963 is no longer relevant to the paper's main findings and should be removed or demoted.

### Visual Communication: 8/10

All 5 figures have been generated (PNG and PDF formats). The visual_audit.md confirms all figures exist. The remaining issue is purely referential:

1. **Figure 2 referenced in wrong context**: Section 2.2 says "Figure 2 illustrates the absorption phenomenon" but Figure 2 is the H_Mech 2x2 factorial bar chart, not a conceptual illustration. Figure 1 should be referenced there instead.

2. **Figure 1 not previewed in Introduction**: The outline promises a teaser figure reference in the Introduction's contributions roadmap, but Figure 1 is never mentioned there. Add "This key result is illustrated in Figure 1" to the contribution preview.

3. **All 5 figures are referenced in text before they appear** (after fixing the Figure 1 reference issue).

### Writing Quality: 7/10

The writing is generally clear and precise. Tables are well-formatted. Limitations are honestly acknowledged. Methodological details are reproducible.

**Banned patterns found**:

1. **"fundamentally"** in Section 5.2: "is likely fundamental to the SAE training objective"—vague without quantification. Remove or replace with specifics.

2. **"overturns"** in Section 5.1: "This finding overturns the prevailing decoder geometry hypothesis"—strong language. Given that decoder geometry is not uniformly wrong (C ≈ A fails but B ≈ D holds), "refines" or "complicates" may be more accurate.

**Unclear or ambiguous sentences**:

1. Section 5.2: "Our empirical results reveal that sensitivity and absorption trade off approximately as sensitivity decreases as absorption increases" — tautological and unclear. Delete or rewrite to state what the trade-off actually is (or is not, given the degenerate result).

2. Section 1: "This encoder-driven mechanism has critical implications" — "critical" is vague. Specify the actual implications.

**Positive writing elements**:

- Introduction problem statement is clear and well-motivated
- Method section is thorough and reproducible
- Tables 2-7 are well-formatted and informative
- Limitations are honestly acknowledged (Section 5.5)
- Negative results properly labeled as FAILED or INCONCLUSIVE
- Discussion distinguishes pilot from full experiment findings

---

## Issues for the Editor

1. **[Critical] Abstract Uses Pilot Numbers, Body Uses Full Experiment Numbers**: The abstract reports "Condition B: 0.076, Condition D: 0.017" which are seed-42 pilot results. The full 5-seed experiment found B=0.055, D=0.017 (delta=0.037). **Fix**: Update the abstract to report full experiment results (0.055 vs 0.017) with delta=0.037, or clearly label the abstract numbers as "pilot experiment (seed 42)." The paper is about full experiments, so the abstract should reflect that.

2. **[Critical] Figure 2 Referenced in Wrong Place**: Section 2.2 says "Figure 2 illustrates the absorption phenomenon conceptually" but Figure 2 is the H_Mech factorial bar chart (encoder vs decoder conditions). **Fix**: Change "Figure 2" to "Figure 1" in Section 2.2. The conceptual illustration of absorption phenomenon is Figure 1. Move the Figure 2 reference to Section 4.1 where the factorial results are discussed.

3. **[Major] Figure 1 Not Previewed in Introduction**: The outline promises "Figure reference: Figure 1 (teaser - key result showing encoder vs decoder contribution)" in the Introduction contributions roadmap, but the paper never previews Figure 1 there. **Fix**: Add "This key result is illustrated in Figure 1" to the Contributions roadmap paragraph in the Introduction (around line 34 after listing contributions).

4. **[Major] Section 5.2 Circular Sentence**: "Our empirical results reveal that sensitivity and absorption trade off approximately as sensitivity decreases as absorption increases" is tautological. **Fix**: Delete this sentence. The paragraph that follows explains the actual findings (degenerate result, absorption = 0 across all L0 levels) clearly without the tautology.

5. **[Major] Pilot R² = 0.963 Still in Abstract**: The abstract mentions "pilot experiments suggested a trade-off frontier ($R^2 = 0.963$)" but this is no longer a main finding—full experiments found degenerate results. **Fix**: Remove the pilot R² from the abstract. The abstract should state "full experiments found degenerate results (absorption = 0 across all L0 levels), indicating no trade-off frontier exists in synthetic SAEs" without referencing the pilot R².

6. **[Minor] Section 4.5 Needs More Motivation**: One sentence introducing the GPT-2 Small validation is minimal. **Fix**: Expand to 2-3 sentences explaining (1) why cross-model validation matters, (2) what it adds beyond MLP experiments, and (3) what the result confirms.

7. **[Minor] "Fundamentally" and "Overturns"**: Section 5.2 "likely fundamental" should be "likely inherent" or specify what makes it fundamental. Section 5.1 "overturns" should be "refines" or "complicates"—the decoder geometry hypothesis is not entirely wrong, just incomplete. **Fix**: Change "overturns" → "refines"; change "fundamental" → specific language about why absorption is inherent to the SAE objective.

---

## What Works Well

1. **Strong mechanistic insight**: The 2x2 factorial design is a clean, elegant experiment that demonstrates encoder-driven absorption. The pass criterion (|B-D| < 0.1) is met with delta = 0.037 and clearly documented with 5-seed statistics in Table 2.

2. **Honest negative result reporting**: H_Comp (R² = 0.04) and H_Pareto (degenerate) are properly labeled as FAILED and INCONCLUSIVE. Section 4.4 correctly identifies H_Safe pilot as "INCONCLUSIVE" rather than overclaiming a negative result.

3. **Methodological transparency**: Table 1 (experimental configuration), hierarchy generation details, and training protocols are thorough and reproducible.

4. **Visual elements now complete**: All 5 figures have been generated and are referenced in the text. This resolves the critical visual communication gap from prior review.

5. **Clear problem motivation**: The introduction establishes why absorption matters for safety-critical interpretability, grounding the work in real applications.

---

## Score Calculation

- Structural Coherence: 7/10
- Notation & Terminology: 8/10
- Claim-Evidence Integrity: 7/10
- Visual Communication: 8/10
- Writing Quality: 7/10

**Average**: 7.4 → **7**

---

**SCORE: 7**

**Recommendation**: The paper is well-written and ready for external review with minor revisions. The critical issues (abstract/body number inconsistency, figure reference errors) are fixable in one revision pass. The core scientific contribution—encoder-driven absorption with decoder regulatory role—is clearly presented and well-evidenced. The visual_audit.md confirms all figures are now present. A score of 7 means "well-written, no internal contradictions, figures support the narrative"—the remaining issues are polish-level, not structural.