# Critique: Discussion

## Summary Assessment
The Discussion section consolidates the paper's main findings into four coherent subsections, covering EDA's regime-specificity, the cross-paradigm non-correlation, negative results, and limitations. The writing is generally honest and avoids overclaiming. However, a factual inconsistency in ITAC numbers between Discussion and Experiments undermines credibility, several mechanistic explanations are presented as established facts despite lacking supporting evidence, the central practical recommendation ("top 5% captures majority") has no backing data, and the 72--75% early-absorption figure is used as a precise constant without uncertainty quantification despite small sample sizes.

## Score: 6/10
**Justification**: The structure and intellectual honesty are strong --- negative results are reported transparently and integrated into a coherent narrative. Reaching 7+ requires: (1) fixing the ITAC number discrepancy, (2) grounding or flagging as hypothetical every speculative mechanistic explanation, (3) backing or removing the "top 5%" practical claim, (4) adding uncertainty quantification for the 72--75% early-absorption figure, and (5) tightening 7.3 which partly recapitulates experiment results instead of synthesizing.

## Critical Issues

### Issue 1: ITAC best-case FN reduction inconsistent between Discussion and Experiments
- **Location**: Section 7.3, paragraph 2
- **Quote**: "with the best individual case reaching 22.7% reduction ($j_{\text{idx}} = 61217$)"
- **Problem**: Table 3 in Section 6.4 reports the best individual case for $j = 61217$ as achieving **18.9%** FN reduction. The Discussion says 22.7%. The Introduction (Section 1) also claims 22.7%. A reviewer cross-checking Table 3 against the Discussion will immediately flag this inconsistency. Since the Table is the primary evidence artifact, this constitutes a factual error in a falsification claim --- the most scrutinized part of any paper.
- **Fix**: Reconcile across all sections. If 18.9% (per Table 3) is correct, update Section 7.3 and the Introduction to 18.9%. If 22.7% is the correct number from source data, update Table 3. The number must match exactly.

### Issue 2: Unsupported "top 5% captures majority of absorbed latents" practical recommendation
- **Location**: Section 7.1, final paragraph
- **Quote**: "At L12-16k, focusing on the top 5% of latents by EDA captures the majority of absorbed latents while reducing the supervised evaluation budget by 20$\times$."
- **Problem**: No precision-at-top-5% or recall-at-top-5% number appears anywhere in Section 4, Table 1, or any figure. "Captures the majority" is vague (51%? 70%? 90%?). The "20x reduction" figure appears here for the first time with no derivation. This is the single most actionable claim for practitioners, and it has zero supporting evidence in the paper.
- **Fix**: Either (a) compute recall@top-5% for L12-16k (fraction of 16 absorbed latents in the top 819 of 16,384 by EDA) and report the exact number in Section 4, referencing it here, or (b) remove the specific "top 5%" and "20x" numbers and replace with a qualitative recommendation grounded in the reported AUROC = 0.776 and Cohen's d = 0.843.

## Major Issues

### Issue 3: Speculative mid-layer mechanistic explanation presented as fact
- **Location**: Section 7.1, paragraph 2
- **Quote**: "First, mid-layer features (L5, L12) exhibit more hierarchical organization than early or late layers: parent-child feature pairs that produce absorption concentrate in the representation strata where the model transitions from syntactic to semantic processing."
- **Problem**: No experiment in this paper measures hierarchical organization per layer. No citation is provided for a "syntactic to semantic processing" transition at layers 5--12 in Gemma 2 2B. Three layers (5, 12, 19) cannot establish a processing-stage boundary. This is speculative interpretation stated as established fact.
- **Fix**: Either cite a specific external reference establishing layer-specific hierarchy density in Gemma 2 2B or similar models, or rewrite as: "One possible explanation is that mid-layer features exhibit more hierarchical organization, but this paper does not directly measure hierarchy density per layer."

### Issue 4: D-EDA recommended for "wider SAEs" despite failing on all wide configurations
- **Location**: Section 7.1, practical recommendation paragraph
- **Quote**: "For wider SAEs or deep layers where EDA alone underperforms, D-EDA provides complementary signal: at GPT2-L10, where EDA yields AUROC = 0.336, D-EDA achieves AUROC = 0.762 [0.686, 0.830]."
- **Problem**: Section 7.3 reports D-EDA achieves AUROC = 0.499--0.602 on all six Gemma Scope configurations, including the wide ones. The single D-EDA success (GPT2-L10) has $d_{\text{SAE}} = 24576$ --- a narrower SAE than Gemma Scope 65k. Recommending D-EDA for "wider SAEs" in the same section that reports D-EDA failure on wider SAEs is self-contradictory.
- **Fix**: Narrow the recommendation to "deeper layers in narrow-dictionary SAEs" --- the evidence supports the layer claim (L10 deeper than L6) but not the width claim. Explicitly state D-EDA does not generalize to wide SAEs.

### Issue 5: Section 7.2 introduces new statistical analysis not grounded in Section 5
- **Location**: Section 7.2, "Hierarchy granularity" paragraph
- **Quote**: "The Spearman correlation between class count and absorption rate ($\rho$ = 0.40, $p$ = 0.60) is suggestive but underpowered with only four hierarchies."
- **Problem**: This correlation is not reported in Section 5 or any table/figure. New quantitative results should not appear for the first time in the Discussion. Calling a non-significant result with $n = 4$ "suggestive" is misleading; $p = 0.60$ is uninformative.
- **Fix**: Either move this analysis to Section 5.4 and reference it here, or remove the specific correlation and state qualitatively that "class-count modulation is plausible but with only four hierarchies cannot be tested statistically."

### Issue 6: The 72--75% early-absorption figure lacks uncertainty quantification
- **Location**: Sections 7.1, 7.3, 7.4 (repeated across Discussion)
- **Problem**: The 72--75% figure derives from L12-16k ($n = 16$, 75.0%) and L12-65k ($n = 65$, 72.3%). With $n = 16$, a 95% binomial CI on 75% is approximately [47.6%, 92.7%] --- extremely wide. The Discussion treats this as a precise constant and builds the entire remediation argument on it. A reviewer will immediately note the small sample sizes.
- **Fix**: Report binomial CIs. At L12-65k ($n = 65$, 72.3%), the 95% CI is approximately [60%, 83%] --- still supports "majority." At L12-16k ($n = 16$, 75%), the CI is wide. Acknowledge: "Early absorption constitutes 72--75% of absorbed latents (95% CI: [60%, 83%] at L12-65k), confirming that the majority are dictionary-coverage failures even at the lower confidence bound."

### Issue 7: Section 7.3 is overlong and recapitulates experiments instead of synthesizing
- **Location**: Entire Section 7.3
- **Problem**: At approximately 420 words, 7.3 is the longest subsection. It restates detail from Sections 4, 5, and 6 (ITAC FN reduction 3.14%, FVU change -4.23%). The regression equation "$\log(\text{absorption}) = -31.0 + 0.59 \cdot \log(\text{width}) + 4.78 \cdot \log(L_0) - 0.048 \cdot \text{layer}$" and $R^2 = 0.18$ appear here for the first time --- new quantitative results embedded in the Discussion rather than in Experiments or an appendix. Discussion should synthesize, not present.
- **Fix**: Move the regression equation to a table in Section 4 or Appendix D. Reduce 7.3 to approximately 250 words by removing already-reported numbers and focusing on the convergent conclusion: "Three negative results collectively confirm that the dominant absorption problem is a training-time dictionary coverage failure." Reference relevant tables rather than restating values.

### Issue 8: Taxonomy explanation in 7.1 introduces new untested mechanistic claims
- **Location**: Section 7.1, paragraph 4
- **Quote**: "any elevated EDA in early-type latents reflects incidental misalignment from the biconvex loss landscape, not the child-parent suppression mechanism"
- **Problem**: This claim --- that residual EDA in early-type latents is "incidental misalignment" --- is a new mechanistic explanation not tested or discussed in Section 6. Section 6 reports that early-type latents have lower EDA than late-type but does not characterize the source of their residual EDA.
- **Fix**: Qualify: "We hypothesize that elevated EDA in early-type latents reflects incidental misalignment from the biconvex loss landscape rather than child-parent suppression, but this remains untested."

### Issue 9: Limitations section omits the single-task basis for the taxonomy
- **Location**: Section 7.4
- **Problem**: The taxonomy (early/late/partial) is derived entirely from first-letter spelling absorption on two SAE configurations. Section 7.4 lists "Limited SAE configurations" but does not flag that the taxonomy itself is task-specific. A reviewer will ask: "You measured absorption cross-domain in Section 5, but only taxonomized first-letter absorption. Does 72--75% early-absorption dominance hold for RAVEL hierarchies?"
- **Fix**: Add a limitation: "The three-subtype taxonomy is validated only on first-letter spelling absorption. Whether early-absorption dominance generalizes to entity-attribute hierarchies remains an open question, pending high-quality probes for RAVEL parent directions."

## Minor Issues
- **7.1, paragraph 1**: "cross-model confirmation at GPT2-L6" --- the glossary specifies "GPT-2 Small" not "GPT2-L6." Use "GPT-2 Small layer 6" on first use in this section for consistency.
- **7.1, paragraph 3**: "EDA achieves AUROC = 0.922 [0.842, 0.979] on polysemantic latents" --- Section 4.5 warns this relies on only 3 positive examples with potentially unreliable bootstrap CIs. The Discussion should repeat this caveat.
- **7.2, paragraph 1**: "This non-correlation does not imply that absorption is domain-specific or that one measurement paradigm is invalid." --- Defensive phrasing that adds no information. Delete and proceed directly to the three structural explanations.
- **7.2, paragraph 3**: "directions projected via random orthonormal QR decomposition" --- unnecessary methodological detail for a Discussion; a reference to Section 5.1 suffices.
- **7.3, H6 paragraph**: The log-linear model coefficients lack standard errors. Without SEs, a reviewer cannot assess whether the positive width coefficient (0.59) differs meaningfully from zero.
- **7.3, H6 paragraph**: "The sparsity pressure that causes absorption increases with dictionary size because more latents compete for activation, and the biconvex loss landscape has more partial minima" --- mechanistic claim without citation. Either cite Tang et al. (2025) or flag as hypothesis.
- **7.3**: "Three pre-registered hypotheses were falsified" --- if not formally pre-registered (OSF, AsPredicted), use "pre-specified" instead.
- **7.4, paragraph 1**: "GPT2-L6 with exact labels (AUROC = 0.629) provides the cleanest validation point" appears for the fourth time across the paper. Shorten to a back-reference: "GPT-2 Small layer 6 (Section 4.2)."
- **7.4, paragraph 4**: "confidence intervals are wide and statistical power is limited" --- this is generic. State the specific consequence: "the scaling sign-reversal test (H6) should be interpreted as 'no evidence for a reversal' rather than 'strong evidence against it.'" (The section does this at the end; move it forward to contextualize earlier.)
- **7.2, last paragraph**: "the most important direction for extending this work" competes with the Conclusion's prioritized future work list. Pick one canonical ordering.
- **Notation**: Section 7.3 uses "sign reversal" without defining it. Add a brief clarification on first use: "sign reversal (i.e., wider SAEs absorbing less after controlling for $L_0$)."

## Visual Element Assessment
- [x] Figures/tables match outline plan (outline specifies no figures for Discussion)
- [x] All visuals referenced before appearance (N/A --- no visuals)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support --- Section 7.3 would benefit from a compact negative-results summary table: columns for hypothesis, predicted outcome, actual outcome, and implication. This would replace approximately 150 words of prose.

## What Works Well
1. **Section 7.3's convergent synthesis** ("These three negatives converge on a single conclusion that reinforces the taxonomy's central finding") is the section's strongest passage. Integrating three independent failures --- D-EDA, ITAC, H6 --- into a single prescriptive message about training-time dictionary coverage is genuine intellectual synthesis, not mere summary. The passage names specific alternative approaches (masked regularization, MP-SAE, hierarchically-aware training objectives) and connects each to the taxonomy.
2. **Section 7.2 handles the cross-paradigm non-correlation with appropriate scholarly nuance.** The distinction between "not correlated" and "measuring different things" is correctly drawn. The closing question ("whether absorption severity is hierarchy-structure-dependent or whether a universal absorption propensity exists") is the right stance --- hypothesis-generating rather than overclaiming.
3. **Section 7.4 is systematically transparent.** Specific probe accuracies (71.4%, 37.8%, 36.8%), the synthetic-activations-only ITAC caveat, the small-$n$ acknowledgment for Spearman correlations, and the distinction between "no evidence for" versus "strong evidence against" demonstrate the kind of methodological honesty that builds reviewer trust.
