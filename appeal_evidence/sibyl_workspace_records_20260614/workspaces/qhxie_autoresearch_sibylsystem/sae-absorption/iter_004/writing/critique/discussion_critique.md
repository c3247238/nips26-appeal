# Critique: Discussion

## Summary Assessment
The Discussion section (Section 7) is well-structured, covering five substantive subsections and a limitations block. It engages honestly with negative results and provides specific numbers throughout. The section's core strength is its willingness to separate conceptual contributions from empirical failures (e.g., Section 7.1's nuanced treatment of the LV framework). However, several claims overshoot the evidence, the safety probe discussion is overly defensive, and the limitations section omits important threats to validity that a careful reviewer would notice.

## Score: 6/10
**Justification**: The section is competent and honest about failures, which earns credit. To reach 7-8, it needs: (1) tighter claim-evidence alignment for the "strongest empirical result" framing, (2) a more critical treatment of the cross-model gap between detection experiments (GPT-2) and downstream correlation (Gemma), (3) removal of hedging language that weakens rather than clarifies, and (4) a limitations section that addresses the correlation-vs-causation gap head-on rather than burying it in future work.

## Critical Issues

### Issue 1: "First direct test" overclaim without causal warrant
- **Location**: Section 7.3, paragraph 2
- **Quote**: "These correlations provide the first direct test of the assumed causal chain from absorption to downstream interpretability."
- **Problem**: Correlations are not a "direct test" of a "causal chain." The word "direct" and "causal chain" together imply causal evidence, but what was measured is a cross-sectional correlation across SAE configurations. The intro (paragraph on "Metric versus impact") uses identical language: "the first direct evidence that absorption reduction translates to downstream SAE improvement." A reviewer at NeurIPS/ICML will flag this as a causal overclaim from correlational data. The Discussion is where this distinction must be made explicit, not reinforced.
- **Fix**: Replace "the first direct test of the assumed causal chain" with "the first systematic correlational evidence linking absorption scores to downstream performance." Add one sentence acknowledging that causal claims would require intervention experiments (ablating absorbed features and measuring downstream change), which is currently only mentioned in passing in the Conclusion's future work.

### Issue 2: Cross-model gap between H1/H2 and H3 insufficiently addressed
- **Location**: Section 7.5, paragraph 2
- **Quote**: "The key question is whether our negative findings (H1, H2) would hold on Gemma-2-2B."
- **Problem**: The section frames the cross-model issue as "would H1/H2 replicate on Gemma?" but ignores the reverse direction: the strongest result (H3) is measured on Gemma Scope SAEs, while all mechanistic analysis is on GPT-2 Small. A reviewer will ask: does absorption measured by the Chanin metric on GPT-2 even correlate with the SAEBench absorption score computed on Gemma? The paper never establishes this bridge. Section 7.5 asserts that the H3 Gemma result "provides a partial bridge" but does not quantify how partial this bridge is, nor acknowledge that the absorption taxonomy (92.3%) and the downstream correlation ($r = -0.595$) were measured on different models and cannot be directly linked.
- **Fix**: Add a paragraph explicitly acknowledging that the detection/taxonomy results (GPT-2) and the downstream results (Gemma) cannot be composed into a single causal narrative without same-model replication. State that the 92.3% comprehensive rate on GPT-2 and the $r = -0.595$ on Gemma are independent findings whose connection is assumed, not demonstrated.

### Issue 3: Partial correlation strengthening presented as unambiguous positive, but it could indicate confound misspecification
- **Location**: Section 7.3, paragraph 1
- **Quote**: "Partial correlations strengthen after controlling for confounds, indicating that absorption captures genuine quality variation beyond what width and layer predict alone."
- **Problem**: Partial correlations that *increase* after adding controls can indicate genuine signal, but they can also indicate confound misspecification or suppressor effects. With only 54 data points and 3 control variables (width, layer, architecture class), the partial correlation estimates have wide confidence intervals that the section never reports. The jump from $r = -0.431$ to partial $r = -0.677$ for SCR (a 57% increase in magnitude) is large enough to raise suspicion of a suppressor variable. The Discussion should note this possibility rather than treating the strengthening as straightforwardly confirmatory.
- **Fix**: Add a sentence noting that partial correlations can inflate under suppressor effects and that the SCR jump ($r = -0.431 \to -0.677$) warrants caution. Report confidence intervals for partial $r$ or note that they are unavailable.

## Major Issues

### Issue 4: Section 7.1 salvage claim is vague
- **Location**: Section 7.1, paragraph 3
- **Quote**: "The conceptual contribution survives: co-activation geometry and frequency imbalance are the right variables for characterizing absorption risk."
- **Problem**: What does "right variables" mean operationally? The LV detector using these exact variables achieved F1 = 0.128. The section claims the variables are correct but the threshold is wrong, yet provides no evidence that a non-threshold-based model using $\sigma_{ij}$ and $f_j/f_i$ as continuous predictors would do better. Without such evidence, the claim that the "conceptual contribution survives" is wishful rather than supported.
- **Fix**: Either (a) report a continuous-predictor result (e.g., Pearson $r$ between $\alpha_{ij}$ and absorption rate, or an AUC treating $\alpha_{ij}$ as a continuous score), or (b) soften the claim to "the variables may still have predictive value as continuous features, but this remains to be demonstrated."

### Issue 5: Unlearning exception is under-analyzed
- **Location**: Section 7.3, paragraph 5
- **Quote**: "The non-significant correlation ($r = -0.175$, $p = 0.280$) with unlearning performance, combined with the smaller sample size ($n = 40$ vs. $n = 54$ for other tasks), suggests that unlearning depends on factors other than feature fidelity"
- **Problem**: The section attributes the null result to "factors other than feature fidelity" but does not consider the simpler explanation: with $n = 40$ and $r = -0.175$, the study is underpowered to detect a moderate effect (power $\approx 0.22$ for $r = -0.2$ at $\alpha = 0.0125$). Before concluding that unlearning truly differs from the other tasks, the Discussion should report a power analysis or at minimum note that the null result is consistent with both "no effect" and "effect too small to detect at this sample size."
- **Fix**: Add a power analysis note: "At $n = 40$ and Bonferroni-corrected $\alpha = 0.0125$, this comparison has approximately 22% power to detect $r = -0.2$, leaving the question open rather than resolved."

### Issue 6: Safety probe paragraph is excessively defensive
- **Location**: Section 7.3, final paragraph ("Safety probe nuance")
- **Quote**: "This non-monotonicity likely reflects confounds in the GPT-2 experiment: the three SAEs differ in layer (5, 8, 10), width (768 to 32k), and architecture --- not just absorption level."
- **Problem**: The paragraph reads as damage control. It spends 7 lines explaining away a result that contradicts the section's main thesis (absorption predicts downstream quality) and then pivots to the Gemma result as "cleaner evidence." While the confound analysis is correct, the Discussion should also consider the possibility that the safety probe result is accurate: absorption may genuinely not predict safety task performance at the single-model level. The unlearning null result on Gemma ($r = -0.175$) provides some support for this interpretation. A more balanced treatment would strengthen the section's credibility.
- **Fix**: Restructure the paragraph to present both interpretations: (1) confounds explain the non-monotonicity, and (2) safety-specific tasks may genuinely be less sensitive to absorption, consistent with the unlearning null result. Let the reader weigh the evidence.

### Issue 7: Limitations section misses clustering concern
- **Location**: Section 7.6, paragraph on "Standard errors"
- **Quote**: "The PMI regression uses HC3 robust standard errors but does not cluster at the letter level (26 repeated measurements per letter across SAE configurations)."
- **Problem**: This is mentioned but not quantified. With 26 letters repeated across 31 configurations, the effective degrees of freedom are much smaller than the 806 observations suggest. Clustered standard errors could inflate the $p$-value for $L_0$ (currently $p = 0.012$) to non-significance, potentially undermining the "layer and $L_0$ are the strongest predictors" claim that the Discussion builds on in Section 7.2. The limitation is acknowledged but its implications are not traced through to the claims that depend on it.
- **Fix**: Either run the clustered analysis and report results, or add a sentence noting that the $L_0$ coefficient significance ($p = 0.012$) is marginal and could plausibly become non-significant under letter-level clustering, leaving layer as the only robust predictor.

## Minor Issues

- **Section 7.1, paragraph 1**: "the Chanin metric's any-absorption rate of 80.8% (21/26 letters)" -- this number appears here and in the experiments section but is not in the outline. Add a brief parenthetical noting which threshold defines "any-absorption" for the Chanin metric.
- **Section 7.2, paragraph 2**: "layer index ($\beta_3 = -0.012$, $p < 0.0001$)" -- the experiments section (Table 3) reports $p < 0.001$, not $p < 0.0001$. Use consistent precision; prefer $p < 0.001$ to match the table.
- **Section 7.2, paragraph 3**: The arXiv reference "arXiv:2604.06495" is cited inline without author names. All other references in the paper use author-year format. Standardize.
- **Section 7.3, paragraph 1**: The partial $r = -0.677$ for SCR is described as "the largest effect" but then explained as "approximately 46% of residual variance." The 46% comes from $r^2 = 0.458$, which is correct. Consider stating the $r^2$ explicitly to avoid forcing the reader to compute it.
- **Section 7.4, paragraph 1**: "24/26 letter features (92.3%)" -- the math is correct. No issue.
- **Section 7.4, paragraph 3**: "The comparison set contains zero tokens for most letters ($n_{\text{comparison}} = 0$)" -- This is a significant methodological issue buried in a caveat paragraph. Promote this to the limitations section and consider flagging it in the experiments section as well.
- **Section 7.5, paragraph 1**: "Gemma-2-2B requires gated HuggingFace access that was not available during experimentation" -- This reads as an excuse. Rephrase to focus on what was done: "We selected GPT-2 Small as the primary model because it is fully open, deterministic, and has the best-studied absorption ground truth."
- **Section 7.6**: No figure or table references in the limitations section. Consider adding a cross-reference to the relevant tables (e.g., Table 3 for the PMI regression, Table 4 for the taxonomy) to help readers verify claims.
- **Terminology**: Section 7.1 uses "the LV competitive exclusion model" and "the LV framework" interchangeably. The glossary specifies no preferred form for the framework name. Pick one and use it consistently.
- **Section 7.3**: "Prior to this work, the research community treated absorption reduction as an intrinsic goal" -- this is a strong sociological claim. Add a citation or soften to "Prior to this work, no published study directly tested whether..."

## Visual Element Assessment
- [x] Figures/tables match outline plan (Discussion has no planned figures; none are included)
- [x] All visuals referenced before appearance (N/A -- no figures in this section)
- [x] Captions are self-explanatory (N/A)
- [ ] No text-heavy sections that need visual support -- Section 7.3 would benefit from a compact summary table of the four downstream correlations (raw $r$, partial $r$, significance) rather than presenting them as a bulleted list. This table already exists in the experiments section (Table 5), so a forward reference would suffice.

## What Works Well
1. **Section 7.1's structural analysis of why LV fails**: The two-point decomposition (zero-sum assumption violation, frequency-rarity confound) is specific, mechanistic, and goes beyond "it didn't work." This is the kind of failure analysis that reviewers respect.
2. **Section 7.2's practical implication**: The pivot from "PMI doesn't predict absorption" to "therefore corpus curation won't help; target the training algorithm instead" is a clean, useful inference that adds value beyond the null result. The specific mention of masked regularization and OrtSAE as the correct intervention targets is actionable.
3. **Section 7.4's honest Type II caveat**: The paper could have buried the inflation concern; instead it dedicates a full paragraph to explaining why the 88.5% Type II rate is likely an overestimate, including the specific methodological problems ($n_{\text{comparison}} = 0$, global mean fallback). This transparency will earn reviewer trust.
