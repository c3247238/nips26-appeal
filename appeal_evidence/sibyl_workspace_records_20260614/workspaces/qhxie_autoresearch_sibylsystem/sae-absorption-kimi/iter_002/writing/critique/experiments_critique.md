# Critique: Results (Section 3)

## Summary Assessment
The Results section presents a well-structured empirical narrative with clear hypothesis testing and honest reporting of inconclusive findings. The numerical evidence is specific, claims are carefully hedged, and the Random-SAE control finding is highlighted with appropriate weight. However, there are several issues with data consistency, statistical framing, and cross-section alignment that need attention before this section is submission-ready.

## Score: 7/10
**Justification**: The section earns a solid 7 for honest reporting, specific numbers, and logical organization. It loses points for: (1) a data inconsistency in the tau_fs robustness table that undermines credibility, (2) missing per-hierarchy breakdown evidence promised in the outline, (3) a paragraph that prematurely signals a transition to Discussion, and (4) some statistical claims that are slightly overstated. Fixing these would bring it to an 8 or 9.

---

## Critical Issues

### Issue 1: Data Inconsistency in Table 2 (tau_fs Robustness)
- **Location**: Section 3.4, Table 2, rows for tau_fs = 0.01 and tau_fs = 0.05
- **Quote**: "| 0.01 | 0.468 | $-0.394$ | 0.981 | $-4.748$ | 0.003 |" and "| 0.05 | 0.471 | $-0.379$ | 0.979 | $-4.748$ | 0.003 |"
- **Problem**: The t-statistic and p-value for the paired t-test are identical across all three tau_fs thresholds (-4.748, 0.003). This is mathematically impossible if the t-test compares semantic-hierarchy vs. non-hierarchy absorption, because those scores *do* depend on tau_fs (the k-sparse probe threshold changes which latents are retained, which changes acc_k-sparse, which changes A_full). The source data in `statistical_analysis_summary.json` confirms this: the tau_fs robustness entries all show the same t and p values, which appears to be a copy-paste error in the JSON generation pipeline.
- **Fix**: Recompute the paired t-test at each tau_fs threshold independently. The semantic-hierarchy and non-hierarchy absorption scores should be recomputed for tau_fs = 0.01 and tau_fs = 0.05 (not just tau_fs = 0.03), and the t-test should use those recomputed scores. If the scores truly do not change with tau_fs (which would itself be noteworthy), state this explicitly and explain why. Otherwise, correct the table with the actual values.

### Issue 2: Premature Transition Sentence
- **Location**: Section 3.6, final paragraph, line 81
- **Quote**: "These results demand careful interpretation of what they mean for the field."
- **Problem**: This sentence is the transition bridge from Results to Discussion (see outline.md: "Results → Discussion: 'These results demand careful interpretation...'"). It belongs at the end of the Results section, but its placement after the GPT-2 replication subsection makes it feel like an orphaned fragment. More importantly, it is redundant because the Discussion section already begins with its own framing. In the outline, this bridge is listed under "Transition Logic Summary" as the paragraph that *leads into* Discussion---it should not appear verbatim in the Results text itself.
- **Fix**: Remove this sentence. The Results section should end with the GPT-2 replication findings. The Discussion section's opening paragraph ("Interpreting the Inconclusive Construct Validity") already provides the natural transition.

---

## Major Issues

### Issue 3: Missing Per-Hierarchy Breakdown
- **Location**: Section 3.1 (Main Results)
- **Problem**: The outline promises "Table 4: Per-Hierarchy Absorption Scores (Appendix)" with "10 hierarchies x 8 SAEs, cell = absorption score." The Results section mentions "Table 1 reports per-architecture absorption scores across all three conditions" but never references the per-hierarchy breakdown. The source data in `statistical_analysis_summary.json` contains `per_hierarchy_absorption_mean` (10 hierarchies with mean scores), but there is no per-SAE, per-hierarchy matrix shown anywhere. For a construct-validity study, readers need to see whether the aggregate pattern holds across individual hierarchies or is driven by outliers.
- **Fix**: Add a reference to Appendix Table 4 in Section 3.1: "Per-hierarchy breakdowns are reported in Appendix Table 4." Ensure Appendix Table 4 is actually generated with the full 10 x 8 matrix. Alternatively, add a brief sentence in 3.1 noting the range of per-hierarchy variation (e.g., "Per-hierarchy scores ranged from 0.097 to 0.390, with 'fruit' showing the lowest and 'container' the highest mean absorption across architectures").

### Issue 4: "Most Consequential Finding" Claim Is Unsubstantiated
- **Location**: Section 3.5, paragraph 2
- **Quote**: "This is the most consequential finding in the study."
- **Problem**: The authorial voice here makes a strong evaluative claim without justification. Why is the Random-SAE finding "most consequential" compared to the hierarchy-specificity failure (H2 rejection) or the inconclusive construct validity (H1)? The Discussion section (4.3) calls the Random-SAE anomaly "the most striking finding," which is more defensible because it comes after the full results have been presented. In the Results section, this superlative feels premature and opinionated.
- **Fix**: Replace "most consequential" with a more neutral framing: "This finding has direct implications for how the metric is interpreted." Or, if the superlative is retained, add a brief justification: "...because it demonstrates that the metric on semantic tasks is insensitive to learned structure, a property that the theoretical motivation for absorption assumes."

### Issue 5: Statistical Claim Overstated --- "Identical (to three decimal places)"
- **Location**: Section 3.5, paragraph 1
- **Quote**: "both identical (to three decimal places) to the Standard SAE"
- **Problem**: The Standard SAE scores 0.3516666666666667 on semantic-hierarchy and 0.41605263157894734 on non-hierarchy. The Random SAE scores exactly the same values (per `statistical_analysis_summary.json`). This is not "identical to three decimal places"---it is *exactly* identical to machine precision. The phrasing "to three decimal places" understates the precision and raises a subtle question: why are they identical to all digits? This suggests the Random SAE's semantic-hierarchy absorption is computed from the *same* SAE latents as the Standard SAE (i.e., the permutation only affects the decoder, but the absorption formula uses encoder outputs for the probe, so the scores should indeed match). The text should explain this mechanism rather than hand-waving with "to three decimal places."
- **Fix**: Replace with: "exactly identical to the Standard SAE---because the absorption formula depends on the SAE encoder output, and the Random SAE only permutes the decoder directions, leaving encoder activations unchanged." This turns a potentially confusing observation into an explanatory strength.

### Issue 6: H1 Framing Inconsistency with Method Section
- **Location**: Section 3.2
- **Quote**: "H1 is neither supported nor rejected---the evidence is too weak to draw a conclusion about construct validity."
- **Problem**: The Method section (2.6) states: "H1 is supported if r > 0.6 and the CI excludes 0; rejected if the CI includes values < 0.3." The observed r = 0.463 does not meet the "supported" threshold (r > 0.6), and the CI [-0.389, 0.981] *does* include values < 0.3 (it includes -0.389). By the pre-registered criteria, H1 should be classified as "rejected," not "inconclusive." The Results section reclassifies it as "inconclusive," which contradicts the Method section's explicit decision rule.
- **Fix**: Either (a) change the Results section to classify H1 as "rejected" per the pre-registered criteria, with a caveat that the point estimate is moderate, or (b) revise the Method section's criteria to match the "inconclusive" framing (e.g., "rejected only if the CI excludes 0.3 entirely"). Option (a) is preferred because it preserves pre-registration integrity.

---

## Minor Issues

- **Section 3.1, line 5**: "confirming that the selected architectures cover the full absorption-rate spectrum" --- "full spectrum" is overstated; 8 architectures from one model family do not cover the "full" spectrum of possible SAE designs. Fix: "covering a broad range of absorption rates."

- **Section 3.1, line 20**: "These inversions suggest that the two tasks measure different phenomena." --- This is an interpretive claim that belongs in Discussion, not Results. Fix: move to Discussion 4.1 or soften to "These inversions are consistent with the hypothesis that the two tasks measure different phenomena."

- **Section 3.2, line 26**: "Because the interval includes values below 0.3 and above 0.6, H1 is neither supported nor rejected" --- The "above 0.6" part is irrelevant to the rejection criterion (which is about values < 0.3). Fix: "Because the interval includes values below 0.3, H1 is rejected by the pre-registered criterion, though the moderate point estimate leaves room for uncertainty."

- **Section 3.3, line 34**: "Non-hierarchy scores are significantly *higher* than hierarchy scores---the opposite of the predicted direction." --- The italics on "higher" are effective for emphasis, but the phrase "opposite of the predicted direction" assumes the reader remembers H2's prediction from the Method section. A brief reminder would help: "...the opposite of H2's prediction that hierarchy scores would exceed non-hierarchy scores."

- **Section 3.4, line 52**: "The hierarchy specificity rejection ... is identical across all thresholds because the paired t-test compares the same architectures on two conditions independent of tau_fs." --- This explanation is incorrect. The paired t-test *does* depend on tau_fs because the absorption scores (which feed into the t-test) depend on acc_k-sparse, which depends on tau_fs. The fact that the t-values are identical in the data is a bug (see Critical Issue 1), not a mathematical truth. Fix: remove this incorrect explanation and instead report the actual recomputed values.

- **Section 3.6, line 77**: "Absolute scores on GPT-2 are near-zero compared to Pythia-160M" --- "near-zero" is vague. The Standard SAE scores 0.000 (exactly zero) on GPT-2 semantic-hierarchy. Fix: "Absolute scores on GPT-2 are at or near zero (Standard: 0.000, TopK: 0.003)..."

- **Table 1 caption**: The caption says "Best (lowest) per-column values are bolded" but the bolding is inconsistent. GatedSAE first-letter (0.008) and TopK first-letter (0.576) are both bolded---but 0.008 is the *lowest* (best) and 0.576 is the *highest* (worst). The caption says "best (lowest)" but the bolding appears to mark both best and worst. Fix: clarify the bolding convention or remove bolding from the highest values.

- **Figure 2 caption**: "The wide interval reflects small sample size ($n = 7$) and high variance" --- The caption mentions $n = 7$ but the figure may include the Random SAE point. Clarify in the caption whether the scatter shows 7 or 8 points.

- **Missing reference to Section 2**: The Results section never explicitly references the Method section (e.g., "As described in Section 2.4..."). Adding 1-2 cross-references would strengthen the paper's internal coherence.

---

## Visual Element Assessment

- [x] Figures/tables match outline plan (5 figures, 3 tables all present)
- [x] All visuals referenced before appearance (Table 1, Figure 1, etc. all referenced in text before display)
- [ ] **Captions are self-explanatory** --- Figure 4 caption mentions "H1 threshold of r = 0.6" but this threshold is only defined in the Method section; a reader skimming figures might not know what H1 is. Add a brief parenthetical: "(the pre-registered threshold for supporting construct validity)".
- [ ] **No text-heavy sections that need visual support** --- Section 3.2 (H1) is text-heavy with only a scatter plot. A small inset table showing the raw correlation values alongside the CI would improve scannability.

---

## What Works Well

1. **Honest reporting of inconclusive results.** Section 3.2 does not overclaim: "H1 is neither supported nor rejected---the evidence is too weak to draw a conclusion." This is exactly how a top venue expects negative/inconclusive results to be reported. The bootstrap CI is reported prominently, and the small-n limitation is acknowledged.

2. **The Random-SAE control is presented with maximal impact.** Section 3.5 leads with the striking numerical comparison (0.352 vs. 0.352) and immediately draws the correct implication. The contrast with first-letter absorption (0.030) is highlighted, showing the authors understand which comparisons matter.

3. **Specific numbers throughout.** Every claim is anchored to a specific value: "r = 0.463," "t = -4.748," "0.096 points, a 41% relative increase." There are no vague qualifiers like "significantly higher" without numbers. This is the standard for a 7+ score section.

4. **Logical subsection ordering.** The section follows the hypothesis structure (H1 → H2 → H3 → Control → Replication), which mirrors the Method section and makes the paper easy to follow. Each subsection has a clear claim, evidence, and interpretation.
