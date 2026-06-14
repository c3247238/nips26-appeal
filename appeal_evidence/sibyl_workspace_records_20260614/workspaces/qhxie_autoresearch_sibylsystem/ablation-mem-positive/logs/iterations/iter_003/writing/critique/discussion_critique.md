# Critique: Discussion

## Summary Assessment
The Discussion section is well-structured and presents a coherent interpretation of the experimental results. It appropriately reports negative results (H6 falsified), connects findings to the broader literature on the actionability paradox, and provides practical implications. The main weaknesses are: (1) some claims lack direct statistical backing in context, (2) the variance paradox 733x ratio is asserted without experimental citation, (3) a banned pattern appears in the first paragraph, and (4) the mechanism interpretation oversells certainty without adequate hedging.

## Score: 7.5/10
**Justification**: This is a solid discussion section that would score around 7 at a mid-tier venue. It correctly interprets results, reports negative findings honestly, and connects to practitioner concerns. To reach 8+, the section would need: (1) more precise statistical references when making significance claims, (2) proper citation of where key numbers (733x) come from in the actual experiments, (3) removal of the banned pattern, and (4) consistent hedging on the mechanistic interpretation.

---

## Critical Issues

### Issue 1: "significantly larger" without immediate statistical backing
- **Location**: Section 5.1, paragraph 2 (line 9: "significantly larger than low-CV absorbed features")
- **Quote**: "Figure 1 shows that high-CV absorbed features (CV > 1.0) produce steering effects of 0.308, 0.522, and 0.745 logit change at strengths +3, +5, and +7 respectively—significantly larger than low-CV absorbed features"
- **Problem**: The word "significantly" claims statistical significance but the supporting p-value is in Table 1, not referenced in this sentence. A reader seeing "significantly larger" expects to see p < 0.01 or t-statistic immediately nearby. The table follows, but the claim should cite it: "significantly larger (t = 9.96, p < 0.01)."
- **Fix**: Change to "significantly larger (t = 9.96, p < 0.01 at strength +3)" or add explicit table reference: "as shown in Table 1."

### Issue 2: 733x ratio asserted without experimental citation
- **Location**: Section 5.3, paragraph 2 (line 35: "absorbed features exhibit CV approximately 7.33 compared to 0.01 for non-absorbed features (733x ratio)")
- **Quote**: "This mechanistic interpretation aligns with the variance paradox observation: absorbed features exhibit CV approximately 7.33 compared to 0.01 for non-absorbed features (733x ratio)"
- **Problem**: The variance paradox is described in the proposal as coming from initial analysis, but the Discussion presents this 733x ratio as established experimental fact without citing a specific experimental result file or table. The ratio 7.33 / 0.01 = 733x is arithmetic, but the underlying CV values should be cited from an experiment.
- **Fix**: Add citation: "as measured in exp/results/full/cv_full_analysis.json" or "as reported in our pilot experiments (pilot_summary.md)."

### Issue 3: Mechanism presented as established fact, not hypothesis
- **Location**: Section 5.3, paragraph 1-3
- **Quote**: "When a parent feature is absorbed, steering the parent activates the child feature, which then affects model outputs. The routing behavior of the child feature determines whether steering propagates to the output."
- **Problem**: This framing presents the causal mediation mechanism as confirmed, but the discussion itself acknowledges the mechanism is speculative ("We propose a causal mediation mechanism"). The bypass routing and context-sensitive routing explanations are hypotheses derived from correlation, not proven causal pathways. The phrase "The routing behavior of the child feature determines" is definitive without hedging.
- **Fix**: Add hedging language: "We hypothesize that..." or "A plausible mechanism is...". Change "The routing behavior of the child feature determines..." to "The routing behavior of the child feature may determine..." Also in paragraph 2, "High-CV features route through context-sensitive specialized channels" should become "High-CV features may route through context-sensitive specialized channels."

---

## Major Issues

### Issue 4: "confirms" overstates what the effect ratio proves
- **Location**: Section 5.1, paragraph 2 (line 11: "The aggregate effect ratio of 1.47x confirms that absorbed features are not uniformly non-steerable")
- **Quote**: "The aggregate effect ratio of 1.47x confirms that absorbed features are not uniformly non-steerable"
- **Problem**: The effect ratio of 1.47x shows a difference between groups, but "confirms" is too strong. The statistical tests (t = 9.96, p < 0.01) confirm H1; the ratio itself is a descriptive statistic describing the magnitude of the difference. The word "confirms" should be reserved for statistical inference.
- **Fix**: Change to "The aggregate effect ratio of 1.47x indicates that absorbed features decompose into steerable and non-steerable subpopulations."

### Issue 5: H6 falsification not integrated with mechanism narrative
- **Location**: Section 5.4 (Why Orthogonality Does Not Explain the Effect)
- **Problem**: Section 5.3 proposes that CV predicts steering via routing behavior. Section 5.4 then shows orthogonality does not predict steering, which constrains the mechanism. But the Discussion does not explain how this constraint should affect our interpretation of the mechanism. If orthogonality (a decoder property) doesn't predict steering, but CV (an activation property) does, this suggests the mechanism operates at the activation level, not the decoder level. This nuance is missing.
- **Fix**: Add a sentence at the end of 5.4: "This constraint suggests the routing mechanism operates at the level of activation patterns (captured by CV) rather than decoder geometry. The CV-steering correlation is not mediated by decoder orthogonality, consistent with context-sensitive vs bypass routing occurring in activation space."

### Issue 6: Non-absorbed baseline limitation is outdated
- **Location**: Section 5.6, Limitation 4 (lines 67-68: "we have not yet analyzed whether robust absorbed features achieve comparable steering effectiveness to non-absorbed features")
- **Quote**: "Non-absorbed baseline comparison incomplete: While we measured non-absorbed feature steering for context, we have not yet analyzed whether robust absorbed features achieve comparable steering effectiveness to non-absorbed features, or whether absorption degrades steering even for high-CV features."
- **Problem**: This limitation appears to be outdated. The experiments section (4.5) reports that non-absorbed features showed mean steering effect of 0.102 (SD: 0.078), compared to 0.097 for absorbed high-CV features — "the difference is 0.0045, which is not practically significant." If this analysis was completed (full_non_absorbed_baseline_DONE exists), the limitation should reflect what was found rather than claiming the analysis is incomplete.
- **Fix**: Update to reflect completed analysis: "Absorbed high-CV features show comparable steering effectiveness to non-absorbed features (0.097 vs 0.102, difference = 0.0045), suggesting absorption per se does not destroy steering potential for high-CV features. However, whether this holds across different SAE architectures remains to be tested."

### Issue 7: Non-absorbed baseline not discussed despite importance
- **Location**: Section 5 (Discussion), outline item 5.4
- **Problem**: The experiments section (4.5) reports that absorbed high-CV features (0.097) and non-absorbed features (0.102) show essentially equivalent steering effects. This is a key result: it means "robust absorbed" features are not degraded relative to non-absorbed. The Discussion does not mention this at all. This is arguably the most practically important finding for interpretability practitioners — absorbed features with high CV behave equivalently to non-absorbed features for steering purposes.
- **Fix**: Add a paragraph in Section 5.2 (CV as Practical Predictor) or a new 5.3.x subsection discussing this result: "Furthermore, absorbed high-CV features achieve steering effects comparable to non-absorbed features (0.097 vs 0.102), suggesting that the CV-based decomposition identifies absorbed features that are functionally equivalent to non-absorbed features for steering purposes."

---

## Minor Issues

- **Banned pattern**: Section 5.1, paragraph 1: "In recent years..." does not appear, but the phrase "This finding cast doubt on the practical utility" uses "cast doubt" which is somewhat vague. More importantly, Section 5.3 paragraph 1 uses "creates mediated steering where..." and "creating zero net effect..." which are slightly hollow constructions. Rewrite as "This allows steering effects to propagate" and "which counteracts the parent steering, resulting in zero net effect at the output."
- **Location**: Section 5.1, paragraph 2: "non-clinical LLM domains" — should this be "non-clinical LLM domain" (singular)? The paper is about one domain (GPT-2), so "domain" singular may be more precise.
- **Table in 5.2**: Numbers in table (0.308, 0.210, 1.47x) are rounded from experiments section values (0.3079, 0.2103, 1.46x). Verify rounding is consistent and doesn't introduce misleading precision differences.
- **Location**: Section 5.5, bullet 4: "High-CV features may tolerate stronger interventions" — this is plausible but not directly evidenced. Consider adding hedge: "High-CV features may tolerate stronger interventions, though this remains to be tested."
- **Location**: Section 5.6, bullet 1: "Detailed analysis of those results is pending" — this should specify when/if results will be available or whether they will be included in the paper.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan — **PARTIAL**: Outline shows Discussion references Figure 1 and Figure 4 (experiments figures), but Figure 5 (mechanism diagram) is listed as `fig5_mechanism_desc.md` in the discussion.md file — this is a markdown description file, not a rendered diagram. The mechanism architecture diagram described in the outline does not exist as a proper figure file.
- [x] All visuals referenced before appearance — Figure 1 referenced at line 9, Figure 4 at line 41, both appear in experiments section before discussion
- [ ] Captions are self-explanatory — N/A for Discussion (no figures in this section)
- [ ] No text-heavy sections that need visual support — **FAIL**: Section 5.3 (mechanistic interpretation) is text-heavy with no visual support. The bypass vs context-sensitive routing contrast would benefit from a proper architecture diagram. If Figure 5 cannot be rendered as a proper figure, the mechanism description should be self-contained without implying there is a figure.

---

## What Works Well

1. **Section 5.4 is a model of how to report negative results**: The statistics are specific (Welch's t = 1.77, p = 0.079; Pearson r = -0.136; Spearman rho = -0.204), the finding is clearly labeled "NOT_SUPPORTED," and the interpretation correctly frames this as constraining the mechanism rather than invalidating the main finding. This is exactly how to handle falsified hypotheses.

2. **Paragraph 2 of Section 5.1** correctly connects domain specificity to CV distribution as a plausible explanation for Basu et al.'s universal failure. The logical chain (clinical features predominantly low-CV → universal failure) is sound and appropriately hedged as a proposal rather than a proven claim.

3. **Section 5.5** (Implications for Interpretability Practice) provides actionable, specific guidance with numbered bullets. This is exactly what practitioners need and avoids vague speculation. Points 1-4 are concrete and technically grounded.

4. **Section 5.6** (Limitations) is honest and thorough. The bullet format keeps it scannable and each limitation is specific (model scope, measurement limitation, threshold generalizability). Once Issue 6 is fixed to reflect the completed non-absorbed baseline, this section will be complete.

---

## Recommendations

1. **Add mechanism hedge**: Change "We propose a causal mediation mechanism explaining why CV predicts steering effectiveness" to "We propose a hypothetical causal mediation mechanism" throughout Section 5.3. Change "The routing behavior of the child feature determines" to "The routing behavior of the child feature may determine."

2. **Integrate H6 into mechanism discussion**: After reporting H6 falsification, explicitly state what this implies about the mechanism — the CV-steering correlation operates at the activation level, not decoder geometry.

3. **Add non-absorbed baseline paragraph**: Reference the experiments section 4.5 finding that robust absorbed ≈ non-absorbed steering effectiveness (0.097 vs 0.102). This is a positive result that should be highlighted in Discussion — absorbed features with high CV are functionally equivalent to non-absorbed for steering.

4. **Fix "confirms" overstatement**: Change to "indicates" or "provides evidence that."

5. **Add statistical reference to "significantly larger"**: Change to "significantly larger (t = 9.96, p < 0.01 at strength +3)" citing Table 1.

6. **Cite the 733x ratio properly**: Add reference to exp/results/full/cv_full_analysis.json or pilot_summary.md.

7. **Clarify Figure 5 status**: If fig5_mechanism_desc.md is not a proper rendered figure, remove references to it as a visual element and ensure the mechanism description is self-contained without implying there is an architecture diagram.