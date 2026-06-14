# Critique: Conclusion

## Summary Assessment
The Conclusion provides a well-structured summary that echoes the intro's questions and accurately reflects the experimental outcomes. Its major strength is the honest treatment of negative results (H2, H3) and the acknowledgment of the untested H_Safe hypothesis. The closing sentence strikes an appropriate epistemic tone. The primary weakness is an overclaim in the opening sentence about being "the first rigorous characterization" that is not sufficiently supported, and a minor mismatch with the proposal's alpha steering range.

## Score: 8/10
**Justification**: Solid conclusion that correctly summarizes results and appropriately hedges on safety implications. Would score 9/10 with the overclaim fixed and minor inconsistencies addressed. The 2-point gap from a top score reflects the overclaim issue and minor cross-reference mismatches.

## Critical Issues

### Issue 1: Overclaim in Opening Sentence
- **Location**: Paragraph 1, line 1
- **Quote**: "This paper provides the first rigorous characterization of feature absorption in sparse autoencoders using multi-child proportional ablation on synthetic hierarchies with ground truth structure."
- **Problem**: "The first rigorous characterization" is an overclaim. Chanin et al. (2024) already characterized absorption rigorously; what this paper does is fix the *measurement methodology*. The intro (line 3) correctly frames the contribution as resolving a measurement crisis, not characterizing absorption for the first time. The conclusion should be consistent with this framing.
- **Fix**: Rewrite as: "This paper resolves the measurement crisis in feature absorption by introducing multi-child proportional ablation, enabling differentiation between trained SAEs and random baselines on synthetic hierarchies with ground truth structure." This preserves the genuine contribution (new methodology) without overclaiming scope.

## Major Issues

### Issue 2: Steering Alpha Range Inconsistency with Proposal
- **Location**: Paragraph 3, implicit reference to H3 steering
- **Quote**: "steering absorbed features toward parent directions produces no sensitivity improvement"
- **Problem**: The proposal specifies steering alpha values of {0.05, 0.1, 0.15, 0.2, 0.25} (Part B, step 3), but the experiments section reports {0.0, 0.1, 0.2}. The conclusion does not specify the alpha range, but the discrepancy between proposal and experiment should be flagged. Additionally, the conclusion's phrase "produces no sensitivity improvement (0.0 improvement for both absorbed and non-absorbed features)" directly quotes Table 4 but could be read as claiming a precise zero.
- **Fix**: Either add a brief parenthetical "(alpha in {0.1, 0.2})" or remove the "(0.0 improvement)" precision and say "produces no measurable sensitivity improvement."

### Issue 3: H_Safe Claim May Overreach
- **Location**: Paragraph 4, lines 2-4
- **Quote**: "the positive frequency-absorption correlation suggests that commonly activating features (which include many safety-relevant behaviors) may be more susceptible to absorption. SAE-based safety analysis may face limitations beyond disproportionate absorption of critical features."
- **Problem**: This infers safety implications from a correlation (rho = +0.17) on synthetic hierarchies. The intro's RQ3 explicitly acknowledges this was not tested: "Do safety-critical features exhibit elevated absorption? ... We address this question in Section 6." The conclusion should not use synthetic-hierarchy evidence to make safety claims, even hedged ones. The correlation on synthetic features does not establish that "commonly activating features include many safety-relevant behaviors" -- that relationship was never measured.
- **Fix**: Remove the safety implication inference from the frequency-absorption correlation. The section already says H_Safe was not tested "due to resource constraints" -- the logical conclusion is that safety implications remain unknown, not that the correlation suggests elevated risk. Replace with: "The positive frequency-absorption correlation on synthetic hierarchies remains to be validated on real language model features, including safety-critical ones."

## Minor Issues

- **Glossary mismatch, "epistemic" vs "epistemic"**: The glossary defines "epistemic failure" as a limitation in knowledge, and the text uses "epistemic rather than causal" correctly. No change needed.
- **Glossary mismatch, "steering"**: The glossary says "steering" is preferred over "intervention." The text correctly uses "steering" throughout. Consistent.
- **Notation: absorption rate as decimal**: The glossary specifies decimal (0.50) not percentage (50%). The conclusion correctly uses 0.50 and 0.059. Consistent.
- **"Absorption may be epistemic rather than causal"**: The experiments section discusses this interpretation and the conclusion appropriately echoes it. Consistent.
- **Future work ordering**: The conclusion lists four future directions (real features, patching, mitigation, downstream performance). The outline's conclusion says three (real features, alternative interventions, safety features). The fourth direction (downstream performance) was added without corresponding outline update. This is a minor completeness gap.
- **"H_Safe -- whether safety-critical features are disproportionately absorbed -- was not tested"**: Uses "H_Safe" notation. The experiments section uses the full description without the H_Safe shorthand. Minor terminology inconsistency.

## Visual Element Assessment
- [x] No figures/tables planned for this section (correct decision -- conclusions should not introduce new visuals)
- [x] No text-heavy sections that need visual support
- [x] Appropriate restraint on visual elements

## What Works Well

1. **Paragraph 2 -- Central result framing**: "Our central result is that absorption is real" followed by specific statistics (0.50 vs 0.059, d=8.94, p < 10^-133) directly echoes the intro's promise and provides concrete evidence. This is exactly how a conclusion should handle the main positive finding.

2. **Paragraph 3 -- Honest negative result treatment**: "two negative results constrain the practical implications" is a clear, non-defensive acknowledgment. The ecological prediction framing ("contrary to ecological predictions") correctly contextualizes H2's failure. The epistemic/causal distinction is well-drawn and supported by H3 evidence.

3. **Paragraph 6 -- Methodological contribution framing**: "resolves a measurement crisis" correctly positions the contribution without overclaiming. The explanation of why single-feature ablation saturates is clear and accurate.

4. **Closing sentence**: "Absorption can be measured, but what it measures -- and whether it matters for downstream applications -- remains an open question for the field." This is an excellent epistemic hedge that neither overclaims nor underplays the contribution.
