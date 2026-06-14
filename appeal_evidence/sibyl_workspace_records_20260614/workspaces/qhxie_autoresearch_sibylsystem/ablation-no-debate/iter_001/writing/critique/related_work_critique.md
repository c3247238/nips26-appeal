# Critique: Related Work

## Summary Assessment
The Related Work section provides solid coverage of the four relevant research strands (SAEs, absorption, sanity checks, steering) and introduces the competitive exclusion analogy as a theoretical framing. The writing is clear and well-structured. However, three issues significantly weaken the section: the Safety-Critical Features subsection makes claims that contradict the experiments (which did not test real Gemma Scope features), the Synthesis overstates contributions that were not validated, and the steering/absorption causal claim is contradicted by the Experiments section results.

## Score: 6/10
**Justification**: The section is well-organized and covers the right literature, but it makes two critical cross-section consistency errors: (1) Safety-Critical Features discusses real Gemma Scope safety features as if they were analyzed, when the experiments never ran Part C, and (2) the Synthesis claims steering "tests" causal/epistemic distinction when Experiments show steering found no improvement for ANY features (a deeper failure). These inconsistencies will confuse or alarm reviewers.

---

## Critical Issues

### Issue 1: Safety-Critical Feature Claims Are Unsupported (Cross-Section Contradiction)
- **Location**: Lines 31-39 ("Steering and Feature Sensitivity" subsection)
- **Quote**: "Our work extends this by testing whether resistance to steering correlates with absorption status."
- **Problem**: The paper's own experiments section does NOT test real Gemma Scope safety features (H_Safe was not implemented, per outline.md Section 6.4: "Safety-critical features were not tested in this study"). Yet this subsection frames the paper as having tested safety-critical features via steering. This is a direct contradiction between Related Work and Experiments. The intro (lines 14-15) also promises "Do safety-critical features exhibit elevated absorption?" as a research question -- yet no answer appears anywhere in the paper.
- **Fix**: Remove the framing that implies the paper tested safety-critical features. Instead, frame steering as testing general feature sensitivity (which was run on synthetic features). For safety features, change to future work framing: "A key open question is whether these patterns extend to safety-critical features in real language models -- we leave this to future work."

### Issue 2: Synthesis Overstates What Was Demonstrated
- **Location**: Lines 49-52 ("Synthesis" subsection)
- **Quote**: "Our steering experiments (Section 6.3) test this distinction directly."
- **Problem**: The steering experiments (H3) found zero improvement for BOTH absorbed and non-absorbed features. The intro frames this as testing "epistemic vs causal" -- if steering helped absorbed features but not non-absorbed, that would support causal. But zero improvement for both is a deeper failure. The Related Work cannot honestly say the experiments "test this distinction directly" when the results are a flat null for all features.
- **Fix**: Rewrite the Synthesis to acknowledge the null finding accurately: "Our steering experiments found no sensitivity improvement for either absorbed or non-absorbed features, suggesting that the steering approach itself may be insufficient to test this distinction, or that both phenomena are epistemic."

---

## Major Issues

### Issue 3: MultiScale SAEs Citation is Incomplete
- **Location**: Line 19
- **Quote**: "Gao et al. (2025) proposed MultiScale SAEs..."
- **Problem**: The citation is author-year only; there is no arXiv ID or full reference. Compare with all other citations (e.g., "Chanin et al. (2024)", "Korznikov et al. (2026)"). The proposal.md lists this work but also lacks an arXiv ID. The outline.md lists "Muchane et al. (2025). Hierarchical SAE. arXiv:2506.01197" as a reference -- which is NOT the same as MultiScale SAEs by Gao. Related Work must clarify which work is being discussed.
- **Fix**: Either find and include the correct arXiv citation for MultiScale SAEs, or replace with "Muchane et al. (2025)" if that is the intended reference. Clarify which work the paper is comparing against.

### Issue 4: L1 Regularization Citation is Wrong
- **Location**: Line 9
- **Quote**: "Wright et al. (2010)" listed as L1 regularization reference
- **Problem**: Wright et al. (2010) is a reference for L1 regularization? The standard reference for L1 regularization / sparse coding is Tibshirani (1996) "Regression Shrinkage and Selection via the LASSO" or Tropp (2004). Wright et al. (2010) is likely "Feature selection: Legacy of wrapper approaches" or similar. This needs verification.
- **Fix**: Replace with the canonical L1 regularization reference or verify the correct citation.

### Issue 5: Steering Section Framing Contradicts Experimental Results
- **Location**: Lines 31-39
- **Quote**: "Steering experiments provide a methodology for testing causal relationships... Our work extends this by testing whether resistance to steering correlates with absorption status."
- **Problem**: The Experiments section shows that steering toward parent direction produces NO improvement for ANY features (absorbed or non-absorbed). The section frames steering as a causal test, but the actual result is that the steering intervention fails generally. The implication that "non-absorbed features would respond" is unsupported by any data in the paper.
- **Fix**: Be more cautious: "Steering experiments provide a methodology for testing causal relationships... Our work tests whether steering toward parent directions can restore sensitivity to absorbed features, and whether non-absorbed features show differential steering response."

---

## Minor Issues

- **Line 9**: "Gao et al. (2025)" appears twice -- once for jumps (Sharkey & Sharkey, 2024) and once for MultiScale SAEs. The first Gao reference (for learned top-k) appears without a clear connection to the surrounding text about sparsity mechanisms. Consider merging into one sentence.
- **Line 19**: MultiScale SAEs described as "complementary" -- this is good framing but could be more specific about what complementary means.
- **Line 25**: "multiple baseline conditions: random decoder, shuffled features, and permuted encoder" -- matches the paper's baselines exactly, which is good. No change needed.
- **Line 43**: "two species competing for the same resources cannot coexist at constant population values" -- this is a good definition but the Lotka-Volterra model on line 44-45 is referenced without explaining how it maps to features.
- **Lines 43-47**: The competitive exclusion subsection is the weakest in terms of claim-evidence. It states an analogy but provides no evidence that the analogy holds. This is appropriate for Related Work (it's prior work discussing the analogy), but the Synthesis should be honest about whether the analogy was validated or falsified.

---

## Visual Element Assessment
- [ ] No figures or tables are planned for Related Work -- this is appropriate.
- [ ] No redundant visuals.

---

## What Works Well

1. **Lines 5-9** (SAE subsection): Excellent opening with specific examples of recovered features (syntactic structures, semantic concepts, emotional states, reasoning processes). The technical description of TopK and encoder/decoder equations is accurate and appropriately concise.

2. **Lines 11-17** (Feature Absorption subsection): The explanation of the saturation problem is one of the clearest descriptions in the section. "Ablating one child allows remaining children to reconstruct the parent" is exactly the right intuition and directly motivates the paper's methodology. This is the best-written part of the Related Work.

3. **Lines 21-25** (Sanity Checks subsection): Korznikov et al. (2026) framing is well-integrated and directly motivates the paper's baseline methodology. The connection between random baselines and the paper's three conditions (random decoder, shuffled features, permuted encoder) is made explicit.

---

## Cross-Section Consistency Checklist

- [ ] Terminology: Consistent use of "SAE", "feature absorption", "child/parent features" throughout.
- [x] Notation: $x' = W_{dec} f + b_{dec}$ matches notation.md (where reconstruction is separate from encoding).
- [x] Claims vs. proposal: Absorption definition matches proposal.md.
- [ ] Claims vs. experiments: **VIOLATED** -- Steering framing implies it can differentiate absorbed vs. non-absorbed features; experiments show no differentiation.
- [ ] Claims vs. experiments: **VIOLATED** -- Safety-critical feature analysis is described as a contribution; experiments never ran Part C (H_Safe not implemented).
- [x] Citations: All major prior work (Bricken, Chanin, Korznikov, O'Brien, Tian) is correctly cited with years.
- [ ] Citation completeness: Gao et al. (2025) and Wright et al. (2010) lack verification.
