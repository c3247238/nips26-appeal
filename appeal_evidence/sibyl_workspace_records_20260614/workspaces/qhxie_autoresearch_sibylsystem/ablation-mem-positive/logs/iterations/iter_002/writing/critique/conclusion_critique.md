# Critique: Conclusion

## Summary Assessment
The conclusion is well-structured and faithful to the experimental results. It summarizes contributions clearly, acknowledges limitations honestly, and proposes sensible future work. However, two critical inconsistencies exist between claims in the conclusion and empirical results in the experiments section: (1) the "steering validation not executed" limitation contradicts the steering effectiveness results reported in Section 4.8 (Table 7), and (2) the closing remarks overstate the mechanistic interpretation of the actionability paradox when the experiments only demonstrate a correlational relationship.

## Score: 7/10
**Justification**: The conclusion effectively synthesizes the paper's findings and maintains cross-section consistency on most quantitative claims. It loses points primarily because L4 claims steering validation was "not executed" when Table 7 in the experiments section directly measures "Steering Effectiveness by CV." Additionally, C6 and the closing remarks attribute a mechanistic explanation (specialized child channels) to the actionability paradox when only correlational evidence (CV predicts steering effect size) was obtained.

## Critical Issues

### Issue 1: Steering Validation Claim Contradicts Experiments Section
- **Location**: Section 6.2, Limitation L4; Section 6.1, Contribution C6
- **Quote**: "L4. Steering validation not executed. The CV-steering hypothesis (that high-CV absorbed features route through specialized child channels) remains untested."
- **Problem**: The experiments section (Section 4.8, Table 7) explicitly reports "Steering Effectiveness by CV" with results showing High-CV features (mean steering effect = 0.153) vs Low-CV features (mean steering effect = 0.075). The claim that steering validation "remains untested" directly contradicts the empirical steering results in Table 7. Either the experiments section results are not about the same thing (in which case the figure needs clarification), or the limitation is factually incorrect.
- **Fix**: If Table 7 in the experiments section IS the steering validation, then L4 should be revised to acknowledge what was tested and what remains untested. For example: "L4. Child channel mechanism not validated. While our experiments confirm that high-CV absorbed features show larger steering effects (Table 7, Section 4.8), the specialized child channel routing hypothesis—proposed as the mechanism connecting CV to actionability—remains untested. Activation patching experiments are needed to confirm whether high-CV absorbed features route through child channels that resist steering."
- **Also fix C6**: C6 currently states the paper "connected" findings to the actionability paradox, which is appropriate, but the parenthetical "(Basu et al., 2026): absorbed features route through specialized child channels that resist steering intervention" should be qualified as a proposed mechanism, not an established finding. Consider: "We proposed a CV-based mechanism connecting absorption to the actionability paradox (Basu et al., 2026): high-CV absorbed features may route through specialized child channels that resist steering intervention. Our experiments confirm that CV predicts steering effectiveness (Section 4.8), but the child channel routing hypothesis remains to be validated."

### Issue 2: Closing Remarks Overclaims Mechanistic Explanation
- **Location**: Section 6.4 Closing Remarks, paragraph 2
- **Quote**: "Our connection to the actionability paradox provides a principled mechanism for why near-perfect absorption detection (98.2% AUROC) fails to predict steering utility: high-CV absorbed features route through specialized child channels that resist direct intervention."
- **Problem**: This sentence presents the child-channel routing mechanism as an established explanation for the actionability paradox, but the paper only provides correlational evidence. Section 4.8 shows CV predicts steering effect size (high-CV = more steerable), not that child channels resist steering. The mechanism (child channel routing) was never directly tested. The experiments section states: "This supports the hypothesis that CV positively predicts steering utility" — hypothesis, not established mechanism. The conclusion's claim goes beyond what the data supports.
- **Fix**: Revise to accurately reflect the strength of evidence. For example: "Our CV-based analysis provides a potential mechanism for the actionability paradox: high-CV absorbed features may route through specialized child channels that resist direct intervention. Section 4.8 confirms that CV predicts steering effectiveness, consistent with this hypothesis, though direct validation of the child channel routing mechanism remains future work."

## Major Issues

### Issue 3: Section 4.8 Results Absent from Contributions Summary
- **Location**: Section 6.1 Summary of Contributions
- **Problem**: Section 4.8 ("Steering Effectiveness by CV") reports a novel empirical finding: high-CV absorbed features exhibit approximately 2x larger steering effects than low-CV absorbed features (0.153 vs 0.075). This is a noteworthy finding that directly connects the paper's CV analysis to steering utility. However, it does not appear anywhere in the six contributions (C1-C6) listed in Section 6.1. The reader is left to infer its importance from the Limitations and Future Work sections.
- **Fix**: Either (a) add a C7 for the steering effectiveness finding, or (b) fold it into C6 if C6 is reframed as empirical contribution rather than mechanistic proposal. The contribution format should make clear whether each item is an empirical finding or a theoretical proposal.

### Issue 4: Variance Paradox Framing Conflates Discovery and Mechanism
- **Location**: Section 6.1, Contribution C3; Section 6.3, Future Work F4
- **Quote**: "Genuine discovery requiring new theoretical explanation: absorption may selectively preserve context-sensitive specialized information rather than uniformly degrading signal."
- **Problem**: C3 frames the variance paradox as a "discovery requiring new theoretical explanation," which is correct. However, the explanation offered (absorption preserves context-sensitive specialized information) is presented in the same sentence as if it were established. The actual evidence only shows that absorbed features have higher CV; it does not directly establish that they are "context-sensitive specialized information." F4 appropriately frames this as requiring theoretical formalization using "causal mediation analysis" or "information-theoretic bounds," which is more careful language.
- **Fix**: The phrasing in C3 could be tightened: "suggesting that absorption selectively preserves context-sensitive high-variance information" — the word "suggesting" properly hedges the inferential leap. Ensure this hedging is consistent throughout C3 and does not drift into claiming the mechanism is established.

### Issue 5: F3 Description of CV-Steering Validation Plan Is Vague
- **Location**: Section 6.3, Future Work F3
- **Quote**: "Select 30 high-CV (absorbed) and 30 low-CV (absorbed) features; test steering effectiveness at tau in {3, 5, 7}. If high-CV features show lower steering sensitivity than low-CV features, this would confirm..."
- **Problem**: The conditional ("if high-CV features show lower steering sensitivity") contradicts the empirical result in Section 4.8, which found that HIGH-CV features show HIGHER steering sensitivity (0.153 vs 0.075). The future work plan tests the wrong direction. Either Section 4.8 should be cited as already testing this, or F3 should be corrected to test the refined hypothesis consistent with the observed direction.
- **Fix**: Revise F3 to align with the empirical finding. For example: "If high-CV absorbed features show HIGHER steering sensitivity than low-CV absorbed features (as suggested by Table 7, Section 4.8), this would confirm the CV-steering correlation. To validate the child channel mechanism, activation patching experiments comparing whether the steering effect is mediated by child feature activation are needed."

## Minor Issues

- **Section 6.2, L3**: "Cross-layer measurement at the wrong sparsity" is an excellent, precise framing. Well done.
- **Section 6.3, F2**: Correctly identifies that cross-layer measurement at lambda_c (not 0.001) is needed. Good.
- **Figure references in closing**: The conclusion references no figures, yet the outline plan includes Figure 7 (Hypothesis Test Summary). Consider adding a figure reference to the closing remarks for visual anchoring.
- **C5 closing phrase**: "at lambda_c ≈ 5e-5 rather than lambda = 0.001" — consistent with experiments and glossary.
- **Overall structure**: The 6.1/6.2/6.3/6.4 structure is clear and appropriate. No issues here.

## Visual Element Assessment
- [ ] Figures/tables match outline plan — The conclusion itself contains no figures or tables, which is appropriate for a conclusion section. However, the outline plan (Figure 7: Hypothesis Test Summary) is referenced in the outline but not clearly tied to the conclusion's narrative. The closing remarks would benefit from a figure reference.
- [x] All visuals referenced before appearance — N/A (no visuals in conclusion)
- [x] Captions are self-explanatory — N/A
- [x] No text-heavy sections that need visual support — The section is appropriately concise

## What Works Well

1. **L3 framing is exemplary**: "Cross-layer measurement at wrong sparsity" is a precise, honest characterization that reviewers will appreciate. It correctly identifies the methodological error and points to the correct measurement approach.

2. **F4 (Theoretical formalization)** is well-scoped: The reference to Pearl (2009) for causal mediation analysis and Cui et al. (2026) for information-theoretic bounds is appropriately specific and actionable.

3. **Opening paragraph synthesis is accurate**: The three-sentence summary in the first paragraph correctly synthesizes the quasi-critical threshold, finite-size scaling, and variance paradox without overclaiming.

4. **C2 correctly identifies "first quantitative measurement"**: This is an appropriate "first" claim because the finite-size scaling measurement with nu = 3 is genuinely novel in the SAE literature.
