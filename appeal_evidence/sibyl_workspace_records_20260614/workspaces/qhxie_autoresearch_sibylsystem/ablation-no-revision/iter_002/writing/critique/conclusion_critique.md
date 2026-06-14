# Critique: Conclusion

## Summary Assessment
The conclusion effectively summarizes negative results and provides concrete practical recommendations grounded in the data. Its primary strength is honesty about what failed and what remains untested. However, it contains a significant framing contradiction in the opening paragraph ("uniform" central finding vs. layer 4's 49.3% absorption), mischaracterizes H4's status as "falsified" when it is properly an untested/uninformative result, and includes a "worst layer" reference that contradicts the paper's own layer-dependence story. The open questions are well-articulated but the H2 protocol specificity is insufficient for replication.

## Score: 6/10
**Justification**: The conclusion is competent but falls short of the 8+ range that would indicate a polished, rigorous closing. Three critical issues (the "uniform" contradiction, H4 misframing, and "worst layer" error) each represent substantive errors that a reviewer at a top venue would flag. These are not typos but framing problems that affect how readers interpret the paper's contributions.

## Critical Issues

### Issue 1: "Uniform" central finding contradicts the paper's own data
- **Location**: Section 11.1, first paragraph, lines 3-4
- **Quote**: "the central finding is uniform: absorption as defined by the $A_f$ metric is too rare to constitute a primary failure mode"
- **Problem**: The finding is NOT uniform. Layer 4 has 49.3% of latents with $A_f > 0.5$ — this is a 260x difference from layer 8's 0.19%. The conclusion's own data in Table 2 shows absorption prevalence ranging from 19.5% (layer 0) to 49.3% (layer 4) to 17.3% (layer 10). "Uniform" is factually wrong. The glossary's preferred terminology table explicitly says: "layer-dependent" is preferred over "layer-wise," and the outline.md critical correction #7 says: "Remove 'uniform' — layer 4 has 49.3% absorption, which is NOT uniform."
- **Fix**: Change to: "at depth, absorption is rare and layer-dependent... with the notable exception of layer 4." Or: "Across six audited layers, absorption is near-zero at depth (0.19% at layer 8, 17.3% at layer 10) but peaks at layer 4 (49.3%), revealing a layer-dependent inverted-U pattern rather than uniform rarity."

### Issue 2: H4 mischaracterized as "falsified" — it is uninformative/untested
- **Location**: Section 11.1, paragraph on H4
- **Quote**: "Both subsets yielded 0.0 faithfulness, making the comparison uninformative."
- **Problem**: The section labels H4 as falsified in the summary table but then correctly calls it "uninformative" in the body. The opening summary calls it "falsified; H4 moved in the hypothesized direction but with limited practical significance." H4 is neither falsified nor confirmed — it was not successfully tested. The glossary (preferred terminology entry 12) explicitly states: "inconclusive — 'falsified' when experiment design was flawed." The outline.md critical correction #3 says: "Reframe H4 as 'untested' or 'uninformative' explicitly, not falsified."
- **Fix**: In the summary table (line 7), change "Four were falsified" to "Three were falsified; H4 was uninformative; H2 was not tested." The H4 paragraph should be more explicit: "H4 remains unconducted — the pilot results do not support any conclusion about whether absorption level predicts circuit-level causal importance."

### Issue 3: "Worst layer" misidentifies the layer with highest absorption
- **Location**: Section 11.2, second paragraph, line 9
- **Quote**: "below 0.5% even at the worst layer"
- **Problem**: Layer 4 has the highest absorption (49.3%), making it the worst layer by absorption prevalence, not layer 8. The paper's own Section 5.1 H1 results state "49.3% at layer 4 (exploratory)" and the inverted-U figure clearly shows layer 4 is the peak. "Worst layer" implies layer 8 here, which is factually incorrect. The outline.md critical correction #7 already flags this: "Layer 4 is the HIGHEST absorption layer (49.3%), not the lowest."
- **Fix**: Change "worst layer" to "the deepest audited layers (8 and 10)" or "at depth." Alternatively: "below 0.5% at layers 8 and 10."

## Major Issues

### Issue 4: H2 pre-registered protocol is insufficient for replication
- **Location**: Section 11.1, H2 paragraph (lines 17-18)
- **Quote**: "The pre-registered analysis — Spearman correlation between median token frequency and absorption score — remains a legitimate open question, but running it without redesigning the surrounding framework would produce isolated results of unclear value."
- **Problem**: This describes WHAT was measured but not HOW to replicate it. A reader who wants to reproduce or extend H2 gets no protocol. The proposal.md and outline.md contain a detailed H2 protocol (compute median token frequency for each latent's top-20 activating tokens, bin into quartiles Q1-Q4, compute Spearman rank correlation, falsify if r_s >= 0 or ratio < 2x). The conclusion references H2 as pending but does not include the protocol in any form.
- **Fix**: Add a one-sentence summary of the pre-registered protocol, referencing Section 4 for the full specification: "The pre-registered protocol specifies computing Spearman correlation between median token frequency (over Pile validation) and $A_f$; full details are in Section 4."

### Issue 5: Practical recommendations do not adequately qualify H4's conclusion
- **Location**: Section 11.3, first recommendation (lines 33-34)
- **Quote**: "The faithful circuit tracing depends on the complete dictionary, not on a curated subset. This holds regardless of absorption level."
- **Problem**: The data shows both absorption subsets (low and high) yield 0.0, while full SAE yields 0.289. But this comparison conflates two confounders: (1) dictionary completeness (10% vs 100% of latents) and (2) absorption level. The conclusion attributes the faithfulness difference to dictionary completeness, but the experiment cannot isolate absorption level as the causal factor because both subsets failed entirely. The outline.md critical correction #12 flags this: "H4 conclusion conflates absorption selection with dictionary coverage. Add sentence acknowledging the confound."
- **Fix**: Add: "The H4 pilot cannot isolate absorption level as the causal factor, because both absorption subsets (10% of latents) fail entirely regardless of absorption level. The comparison is confounded by dictionary coverage. A properly designed experiment would compare full SAE dictionaries at layers with different absorption profiles (e.g., layer 4 vs. layer 8)."

### Issue 6: H5 practical significance framing overstates the finding
- **Location**: Section 11.3, second recommendation
- **Quote**: "H5 demonstrates a monotonic reduction in absorption rate with dictionary size: 2.25% at 2K latents, 0.56% at 8K, 0.19% at 24K. The effect is consistent across all metrics and holds even when random controls are subtracted."
- **Problem**: The conclusion omits the critical subsample bias caveat. The experiments used subselections of a single 24K SAE, prioritized by absorbable latents — this is an upper-bound estimate, not a clean comparison. The outline.md critical correction #10 says: "Add one sentence bounding the bias in H5 section." The experiments section acknowledges this in Section 5.4: "this is an upper-bound estimate. Whether the same relationship holds for independently trained SAEs of different sizes is an open question."
- **Fix**: Add one sentence: "This analysis uses subselections of a single 24K SAE, which may overestimate absorption at smaller dictionary sizes; the relationship for independently trained SAEs of different sizes remains an open question."

## Minor Issues

- **Line 3**: "uniform" — change to "layer-dependent" (see Critical Issue 1)
- **Line 7**: "Four were falsified; H5 moved in the hypothesized direction" — H5 is not "moved in direction," it is confirmed in direction, not falsified. Use consistent terminology from the glossary: "not falsified" for H5.
- **Line 22**: "moderate confidence" — vague. What level of confidence? A reviewer might ask for specifics.
- **Line 23**: "relatively independent directions" — "relatively" hedges unnecessarily. The data supports "SAE latents at layers 8 and 10 represent largely independent directions."
- **Line 34**: "This holds regardless of absorption level" — add acknowledgment of H4 design flaw confounding (see Major Issue 5).
- **Line 41**: "Any researcher can apply this framework" — the conclusion does not specify the license or code availability. If code is available, reference it explicitly.

## Visual Element Assessment

- [x] No figures planned for conclusion (correct — none needed)
- [x] No tables in conclusion (correct — summary table is in experiments)
- [ ] Conclusion does not reference any figures from earlier sections — this is fine for a closing section, but a brief forward reference to Figure 1 (the inverted-U pattern) when discussing the layer-dependence finding would help readers connect the conclusion to the data.

## What Works Well

1. **H2 honest acknowledgment** (lines 17-18): "H2 was not tested. ... running it without redesigning the surrounding framework would produce isolated results of unclear value." This is exactly the right framing — clearly distinguishing between "not tested" and "falsified."

2. **Open questions section** (lines 41-43): The four open questions are specific, actionable, and directly derived from the data. "Whether absorption manifests differently in GemmaScope SAEs" and "whether explicit anti-absorption regularization during SAE training reduces absorption further" are both genuine extensions that the paper enables.

3. **Random dictionary control validation** (line 19): "Random dictionary controls score 0.00% by construction, confirming the metric detects real structure. The metric is strict, not broken." This is a strong, specific statement that validates the methodology.

4. **Practical recommendation structure** (Section 11.3): Three clear, actionable recommendations derived directly from H4, H5, and H3 respectively. Practitioners can act on these without reading the full paper.