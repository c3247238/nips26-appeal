# Critique: Research Questions and Hypotheses (Section 3)

## Summary Assessment

The hypotheses section is entirely disconnected from the rest of the paper. While the Introduction, Method, Discussion, and Conclusion have all been restructured around the H6-H10 "Local Inhibition Graph" framework, this section still presents only H1-H4 from the original absorption-degradation study. The section does not mention H6-H10 at all, creating a severe structural inconsistency that would confuse any reader and undermine the paper's coherence.

## Score: 2/10
**Justification**: The section is internally consistent and well-written for the H1-H4 framework, but it is completely wrong for the current paper. A hypotheses section that does not contain the paper's actual hypotheses is a critical failure. To reach the next level, this section must be entirely rewritten to present H6-H10 with their corresponding research questions, falsification criteria, and connections to the competitive suppression theory.

## Critical Issues

### Issue 1: Section Contains H1-H4 Instead of H6-H10
- **Location**: Entire section
- **Quote**: "We test four directional hypotheses derived from RQ1 and RQ2: H1 (Raw steering), H1b (Delta steering), H2 (Probing), H3 (Consistency)"
- **Problem**: The paper's actual contribution, as stated in the Introduction (1.3-1.5), Method (4.1-4.8), and Conclusion (8.1-8.2), revolves around H6-H10: graph prediction of absorption pairs, inhibition explaining precision-recall asymmetry, at-risk feature prediction, layer-dependent structure, and homeostatic rebalancing. This section contains none of these. A reader who skips to Section 3 after reading the Introduction will be completely lost.
- **Cross-section impact**: The Method section (4.1) presents a six-phase pipeline with hypotheses H6-H10, but there is no preceding section that defines these hypotheses. The reader encounters H6-H10 in the Method without ever having seen them defined. The Discussion (6.1-6.5) and Conclusion (8.1-8.3) repeatedly reference H6-H10 as if they were formally introduced, but they never were.
- **Fix**: Completely rewrite this section to present H6-H10. Each hypothesis should include: (a) the formal statement, (b) the directional prediction with specific thresholds, (c) the connection to the competitive suppression theory, and (d) falsification criteria.

### Issue 2: Research Questions Do Not Match the Paper's Stated RQs
- **Location**: Section 3.1
- **Quote**: "RQ1: Does feature absorption cause measurable degradation in downstream interpretability tasks? RQ2: Is the absorption-degradation relationship consistent across model layers? RQ3: Can we derive actionable guidance for SAE practitioners regarding absorbed features?"
- **Problem**: These RQs are from the original absorption-degradation study. The Introduction (1.3) presents five different RQs: RQ1 (graph predicts absorption pairs), RQ2 (inhibition explains precision-recall asymmetry), RQ3 (graph predicts at-risk features), RQ4 (layer-dependent structure), RQ5 (homeostatic rebalancing). The RQs in Section 3 do not match.
- **Fix**: Replace RQ1-RQ3 with RQ1-RQ5 as defined in the Introduction. Ensure exact textual consistency.

### Issue 3: Missing Falsification Criteria for H6-H10
- **Location**: Section 3.3
- **Quote**: "H1, H1b, and H2 are not supported if the Pearson correlation is non-negative (r >= 0) or fails to reach significance (p >= 0.05). H3 is not supported if slopes have opposite signs or differ substantially in magnitude (CV >= 0.5)."
- **Problem**: These falsification criteria apply to H1-H3 only. H6-H10 need their own falsification criteria, which are currently scattered throughout the Method section (4.4-4.8). For example, H6's falsification criterion (precision@20 <= 0.05) is in Section 4.4, not in the hypotheses section where it belongs.
- **Fix**: Add falsification criteria for H6-H10 in Section 3.3. Consolidate all criteria in one place for readability.

### Issue 4: H1b Labeling Is Confusing (No H1a Defined)
- **Location**: Section 3.2
- **Quote**: "H1 (Raw steering)" and "H1b (Delta steering)"
- **Problem**: The section uses "H1b" without ever defining "H1a". This ad-hoc labeling is confusing. The Method section (4.4-4.5) does not use H1a either. The labeling appears to have emerged during revision without systematic planning.
- **Fix**: Since this section will be rewritten for H6-H10, this issue becomes moot. However, ensure H6-H10 labeling is consistent and sequential with no gaps.

## Major Issues

### Issue 5: No Connection to Competitive Suppression Theory
- **Location**: Entire section
- **Problem**: The hypotheses section makes no mention of competitive suppression, the LCA framework, or decoder correlations. These are the theoretical foundations of the paper, yet the hypotheses are framed purely in terms of absorption-degradation correlations without theoretical grounding. The Introduction (1.2) develops the competitive suppression theory at length; the hypotheses should derive directly from it.
- **Fix**: For each H6-H10 hypothesis, explicitly state: "This hypothesis follows from the competitive suppression theory because..." and explain the mechanistic prediction.

### Issue 6: H3 Criterion Changed Without Documentation
- **Location**: Section 3.2, H3
- **Quote**: "CV = sigma / |mu| < 0.5" (with absolute value)
- **Problem**: The notation.md defines CV = sigma / mu (no absolute value). The Method section (4.6) also uses CV = sigma / |mu|. This three-way inconsistency (notation.md without abs, hypotheses.md and method.md with abs) creates confusion. The absolute value was likely added to handle cases where slopes have opposite signs, but this change was not documented.
- **Cross-section impact**: The notation table shows a different formula than the one used in the paper. Readers consulting the notation table will compute different values.
- **Fix**: Update notation.md to match: "CV = sigma / |mu|". Add a footnote explaining why the absolute value is necessary (slopes can have opposite signs, making raw CV negative and uninterpretable).

## Minor Issues

- **Section 3.2, H1b**: "The H1b hypothesis is critical because random feature steering achieves non-negligible success (34--38% in our data)" --- This is a methodological justification, not a theoretical one. For H6-H10, each hypothesis should be justified by the competitive suppression theory.
- **Section 3.2, H2**: "k = 5" is described as "informed by pilot observations" but no pilot data is referenced. For H6-H10, justify k values (10, 20, 50) with similar reasoning.
- **Section 3.3**: The falsification criteria are binary (supported/not supported) but the paper's actual results for H1-H3 are nuanced (uncorrected vs. corrected significance). Consider adding a "partially supported" category or clarifying the interpretation rules.

## Visual Element Assessment
- [ ] Figures/tables match outline plan: N/A (no visuals in this section)
- [ ] All visuals referenced before appearance: N/A
- [ ] Captions are self-explanatory: N/A
- [ ] No text-heavy sections that need visual support: The section is text-only, which is appropriate for hypotheses. A table summarizing H6-H10 with predictions and criteria would improve readability.

## What Works Well

1. **The falsification criteria structure is sound**: The binary supported/not-supported framework with explicit thresholds is methodologically rigorous. This structure should be retained for H6-H10.

2. **The directional prediction format is clear**: Each hypothesis states the expected direction of effect (negative correlation, consistency criterion). This format should be maintained for H6-H10.

3. **The connection between RQs and hypotheses is logical**: RQ1 maps to H1/H1b/H2, RQ2 maps to H3. This RQ-to-hypothesis mapping should be replicated for RQ1-RQ5 and H6-H10.
