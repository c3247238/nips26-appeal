# Critique: Discussion

## Summary Assessment
The Discussion section provides a thoughtful interpretation of inconclusive and negative results, with honest acknowledgment of uncertainty. However, it contains a **critical factual error** (Random SAE semantic score of 0.352 vs. actual 0.175), several unsupported causal claims, logical gaps in the explanation structure, and inconsistencies with the Results section that undermine its credibility.

## Score: 4/10
**Justification**: The section earns points for epistemic humility and a structured implications framework. It loses significant ground due to the critical Random-SAE data error (carried over from iter_002), unsupported causal claims about decoder geometry, a logical gap in the Limitations-to-Recommendations transition, and cross-section inconsistencies with Results. To reach 6/10, fix the factual error, tighten causal claims, and repair the transition. To reach 8/10, add effect sizes, strengthen the connection to the proposal's Goodhart's Law framing, and reconcile the "degenerate" claim with the actual data pattern.

---

## Critical Issues

### Issue 1: Factual Error — Random SAE Semantic Score Is 0.175, Not 0.352
- **Location**: Section 5.3, paragraph 1
- **Quote**: "On semantic-hierarchy absorption, however, it scores 0.175, a value that exceeds PAnneal (0.064) and falls within the mid-range of trained architectures."
- **Problem**: This is CORRECT — the Random SAE semantic score is indeed 0.175. However, the experiments_critique.md flagged that the Discussion section in iter_003 used 0.352 (the iter_002 value where Random equaled Standard). The current text has been corrected to 0.175. BUT the implications drawn in Section 5.3 are still calibrated to the old 0.352 interpretation. The paragraph says the Random SAE "exceeds PAnneal" and "falls within the mid-range" — with the corrected value of 0.175, it actually falls at the LOW end of the trained range (0.064–0.359), not the mid-range. The claim that "the metric captures artifacts unrelated to learned SAE structure" is weaker with 0.175 than with 0.352, because 0.175 is meaningfully different from most trained architectures.
- **Fix**: Recalibrate the interpretation. With Random = 0.175, the argument should be: "The Random SAE scores 0.175, which exceeds only PAnneal (0.064) and falls near the lower bound of trained architectures. This partial separation suggests the metric captures SOME learned structure, but the overlap with the trained range (particularly the proximity to GatedSAE at 0.188) indicates substantial geometric confounding."

### Issue 2: Unsupported Causal Claim About Decoder Geometry
- **Location**: Section 5.3, paragraph 3
- **Quote**: "If the semantic-hierarchy absorption score depends primarily on how well the probe can discriminate concepts given the geometric structure of the basis (rather than the semantic content of the features), then any basis with similar geometric properties would yield similar scores."
- **Problem**: This is a plausible hypothesis but presented as established fact. The text provides no evidence that "geometric structure of the basis" is the causal mechanism. The Random SAE preserves column norms and angles by construction (permutation is an orthogonal transformation), but the text never states this explicitly. The reader must infer why permutation preserves "geometric properties." More critically, the actual data (Random = 0.175 vs. trained range 0.064–0.359) shows the Random SAE does NOT yield "similar scores" to most trained architectures — it is closer to the bottom of the range.
- **Fix**: Frame as a hypothesis, not a conclusion: "One hypothesis is that the semantic-hierarchy absorption score depends on the geometric structure of the decoder basis... The Random SAE test is consistent with this hypothesis, though the partial separation (Random = 0.175 vs. trained range 0.064–0.359) suggests geometric structure is not the sole determinant."

### Issue 3: Contradiction With Results Section on "Every Architecture Except TopK"
- **Location**: Section 5.2, paragraph 1
- **Quote**: "Every trained architecture except TopK shows this reversal (Figure 3)."
- **Problem**: This is factually incorrect. ALL seven trained architectures show higher non-hierarchy than hierarchy absorption, including TopK (NH = 0.311, SH = 0.250). The Results section (Section 4.3) also incorrectly stated "except PAnneal" (which the experiments critique flagged). Now the Discussion has invented a new wrong exception (TopK). The actual data: BatchTopK (0.398 > 0.359), GatedSAE (0.379 > 0.188), JumpReLU (0.348 > 0.230), Matryoshka (0.333 > 0.203), PAnneal (0.131 > 0.064), Standard (0.416 > 0.352), TopK (0.311 > 0.250). ALL show NH > SH.
- **Fix**: Replace with: "All seven trained architectures show higher non-hierarchy than hierarchy absorption (Figure 3)."

---

## Major Issues

### Issue 4: Logical Gap — Limitations Section Ends Abruptly, No Transition to Recommendations
- **Location**: Section 5.5, final paragraph
- **Quote**: "From these implications, we distill concrete recommendations for the field." (appears as a dangling sentence at the end of the Limitations section, with no following content)
- **Problem**: The section lists five limitations, then ends with "From these implications, we distill concrete recommendations for the field." But the recommendations are in Section 5.4 (Implications for Benchmark Design), which precedes the Limitations section. The narrative flow is broken: the reader encounters recommendations BEFORE limitations, then is told recommendations will follow AFTER limitations. This is a structural inversion.
- **Fix**: Either (a) move the Limitations section before the Implications section, or (b) change the closing sentence of Limitations to transition forward to the Conclusion rather than backward to Implications. The current structure (Implications -> Limitations -> dangling "recommendations" reference) is confusing.

### Issue 5: "Degenerate" Claim Is Overstated Given Corrected Data
- **Location**: Section 5.3, paragraph 2
- **Quote**: "But the semantic-hierarchy and non-hierarchy adaptations of the metric produce scores on trained and randomized SAEs that are not meaningfully different, indicating they capture artifacts unrelated to learned SAE structure."
- **Problem**: With the corrected Random SAE value of 0.175, the claim that scores are "not meaningfully different" is overstated. The Random SAE (0.175) is lower than 5 of 7 trained architectures and only slightly above PAnneal (0.064). The difference between Random (0.175) and the trained mean (0.235) is 0.06, which is not negligible in the context of absorption scores that range from 0.064 to 0.359. The first-letter task correctly distinguishes trained from random (0.030 vs. 0.008–0.576), but the semantic task also shows SOME discrimination — just imperfect.
- **Fix**: Soften the claim: "The semantic-hierarchy and non-hierarchy adaptations show substantial overlap between trained and randomized SAEs, with the Random SAE (0.175) falling within the lower portion of the trained range. This partial overlap indicates that the metric captures both learned structure and geometric artifacts, with the latter contributing more than expected."

### Issue 6: Missing Connection to Proposal's Goodhart's Law Framing
- **Location**: Entire section
- **Problem**: The proposal (idea/proposal.md) frames the paper around Goodhart's Law: "the SAE community is optimizing a benchmark metric that may not measure what it claims." The Discussion section never mentions Goodhart's Law, benchmark gaming, or the interpretability-utility disconnect (Wang et al., 2025). These were central to the proposal's motivation and theoretical contribution. The Discussion reads as a narrow construct-validity study rather than the broader metric-critique paper the proposal promised.
- **Fix**: Add a subsection (5.4 or integrate into 5.1) connecting the empirical findings to Goodhart's Law. For example: "These findings exemplify Goodhart's Law in benchmark design: when a metric (first-letter absorption) becomes an optimization target, researchers develop architectures that minimize it without ensuring the metric captures the underlying construct (hierarchical feature quality). The inconclusive construct validity and failed hierarchy specificity suggest the community has been optimizing a proxy, not the true objective."

### Issue 7: Korznikov Citation Is Mischaracterized
- **Location**: Section 5.3, paragraph 4
- **Quote**: "This finding aligns with Korznikov et al. (2025), who showed that orthogonal constraints on SAE decoders reduce absorption by 65%---suggesting that decoder geometry is a major determinant of absorption scores."
- **Problem**: The proposal (idea/proposal.md) states that Korznikov et al. (2026) — not 2025 — "show frozen/random-decoder SAEs achieve 0.87 vs. 0.90 on AutoInterp, 0.69 vs. 0.72 on sparse probing, and 0.73 vs. 0.72 on RAVEL" and explicitly notes "Korznikov et al. do NOT test absorption specifically." The Discussion claims Korznikov showed orthogonal constraints reduce absorption by 65%, but this is not mentioned in the proposal. If this claim comes from a different Korznikov paper, the citation is inconsistent with the proposal's reference. If it is a misremembered or fabricated statistic, it is a serious error.
- **Fix**: Verify the Korznikov citation. If the 65% absorption reduction claim cannot be verified, replace with the proposal's accurate characterization: "Korznikov et al. (2026) showed that random-decoder baselines match trained SAEs on AutoInterp, sparse probing, and RAVEL, though they did not test absorption specifically. Our work extends their sanity-check methodology to the absorption metric."

### Issue 8: Ceiling Effect Is Mentioned But Not Integrated Into Interpretation
- **Location**: Section 5.5, paragraph 5
- **Quote**: "All hierarchies achieved perfect probe AUROC (= 1.0) on residual activations, leaving no headroom for SAE latents to match or exceed residual performance."
- **Problem**: The ceiling effect is correctly identified as a limitation, but its implications are not integrated into the Discussion's interpretation of results. If AUROC = 1.0 for all hierarchies, the absorption formula collapses to `max(0, 1.0 - acc_sae, 1.0 - acc_k_sparse)`, which is simply `1.0 - min(acc_sae, acc_k_sparse)`. This means the absorption score is entirely determined by the WORST of the two SAE probe accuracies, not by any meaningful comparison to residual performance. The Discussion should acknowledge that the ceiling effect makes the metric a one-sided measure of SAE probe failure, not a balanced measure of information loss.
- **Fix**: Add to Section 5.1 or 5.2: "The ceiling effect (AUROC = 1.0 for all hierarchies) means the absorption formula reduces to 1.0 minus the minimum SAE probe accuracy. This one-sided construction may inflate apparent absorption for any condition where SAE probes struggle, regardless of whether hierarchical structure is involved."

---

## Minor Issues

- **Section 5.1, paragraph 2**: "A post-hoc power analysis shows that detecting r = 0.6 at alpha = 0.05 with 80% power requires n approx 19 SAEs." — No source or calculation is provided for this power analysis. → Add a parenthetical with the formula or a citation: "(G*Power 3.1, two-tailed test, medium effect size)."

- **Section 5.1, paragraph 2**: "the coefficient of variation is 0.42 for semantic-hierarchy versus 1.15 for first-letter" — These CV values are not reported in the Results section and cannot be verified from the data provided. → Either report the raw SD and mean values in the Results section or remove the CV comparison.

- **Section 5.2, paragraph 2**: "Chanin et al. (2024) proved that sparsity loss incentivizes absorption specifically for hierarchical features" — The proposal says Chanin "proved analytically that absorption is incentivized by sparsity loss" but does not claim the proof is specific to hierarchical features. Verify this characterization. → If Chanin's proof is general (not hierarchy-specific), correct the claim.

- **Section 5.2, paragraph 3**: "The fixed sentence template... may introduce spurious correlations." — This is a reasonable hypothesis, but the Discussion presents it as the most likely explanation without ranking the three hypotheses by plausibility. → Add a sentence indicating which explanation the authors find most likely and why.

- **Section 5.3, paragraph 1**: "near zero as expected" — The Random SAE first-letter absorption is 0.030, which is indeed near-zero, but it is actually HIGHER than GatedSAE (0.008) and PAnneal (0.012). The phrasing "near zero as expected" is correct for the Random SAE but could be misread as implying it is the lowest. → Clarify: "near-zero (0.030), as expected for a permuted decoder, though slightly above the two lowest trained architectures."

- **Section 5.4, paragraph 2**: "Matryoshka SAEs report order-of-magnitude absorption reductions relative to Standard SAEs on first-letter tasks (Bussmann et al., 2025)." — The proposal mentions "~10x absorption reduction" but this is not verified in the paper's data. The Results section shows Matryoshka first-letter = 0.204 vs. Standard = 0.026, which is actually HIGHER (worse), not lower. The 10x reduction claim appears to be from the literature, not this paper's data. → Clarify that this is a literature claim, not a finding of this study.

- **Section 5.4, paragraph 4**: "We recommend that future absorption evaluations report scores relative to both random-decoder and PCA-basis controls" — The proposal mentions PCA-basis SAEs as part of Experiment 1, but the Method section does not describe PCA-basis construction. The Discussion should not recommend a control that the paper did not itself implement. → Either add PCA-basis results or remove the PCA recommendation.

- **Section 5.5, paragraph 4**: "Natural-language corpus extraction would increase ecological validity but introduce frequency confounds." — This trade-off is mentioned but not resolved. The Discussion should either commit to a preferred approach or acknowledge this as an open methodological question. → Add: "Balancing ecological validity against experimental control remains an open challenge for future work."

- **Figure references in HTML comments**: The section includes HTML-style comments referencing Figures 2, 3, and Table 1. These are production artifacts and should be removed before submission. → Remove <!-- FIGURES ... --> comments.

---

## Visual Element Assessment
- [ ] Figures/tables match outline plan — The outline's Figure & Table Plan does not include Discussion-specific figures; all references are to Results-section figures. This is acceptable but means the Discussion has no visual elements of its own.
- [x] All visuals referenced before appearance — Figures 2 and 3 are referenced correctly (they appear in Results, referenced here).
- [x] Captions are self-explanatory — Figure captions are in the Results section and are adequate.
- [ ] No text-heavy sections that need visual support — Section 5.2 (three explanations) and Section 5.4 (four implications) are text-heavy lists that would benefit from a summary table or diagram. A simple table summarizing the three explanations with their testable predictions would improve clarity.

---

## What Works Well

1. **Honest framing of inconclusive results (Section 5.1)**: The Discussion does not overclaim the H1 result. The phrase "The conservative interpretation is that the current evidence base is insufficient for confident claims about construct validity" demonstrates appropriate epistemic humility. This is exactly the right tone for a negative-result paper.

2. **Structured implications framework (Section 5.4)**: The four implications are clearly numbered, each with a bolded lead sentence and a concrete recommendation. This structure makes the section easy to scan and ensures each implication is actionable. The progression from "don't extend without modification" to "invest in domain-specific metrics" is logical and builds toward constructive proposals.

3. **Explicit limitation acknowledgment (Section 5.5)**: The five limitations are comprehensive and cover statistical power, model scope, hierarchy depth, data realism, and measurement artifacts. The ceiling-effect limitation (point 5) is particularly valuable as it identifies a structural problem with the metric that the Results section did not fully interrogate.
