# Critique: Experiments (Sections 4, 5, and 6)

## Summary Assessment
The Experiments section is the backbone of this paper and is generally well-executed: hypotheses are pre-registered with quantitative success criteria, negative and null results are reported transparently, and the statistical methodology (Bonferroni correction, partial correlations, HC3 robust SEs) is appropriate. The section's main weakness is a persistent mismatch between what was proposed (Gemma-2-2B) and what was delivered (GPT-2 Small), which undermines the scope claims made in the introduction and proposal. Several tables conflate experiment sections with result sections in ways that could be tightened, and the taxonomy result (Section 5.3) leans heavily on a Type II metric whose inflation is acknowledged but insufficiently quarantined from the headline claim.

## Score: 7/10
**Justification**: Solid empirical work with honest reporting of failures. The H3 result is genuinely valuable and well-supported. Drops from 8 because: (1) the Type II taxonomy claim is presented as a "Key Positive Result" despite the acknowledged inflation caveat, (2) the safety probe experiment is underpowered and confounded in a way that weakens rather than supports H3, (3) the outline's Figure 3 (partial regression plot) is missing from the text, and (4) several cross-section inconsistencies exist with the method section. Reaching 8/10 requires tightening the taxonomy framing, removing or downgrading the safety probe, adding Figure 3, and resolving notation/numbering inconsistencies.

## Critical Issues

### Issue 1: Missing Figure 3 (Partial Regression Plot)
- **Location**: Section 5.2 (H2 --- Corpus PMI)
- **Quote**: The outline specifies "Figure 3: Partial Regression Plot --- Absorption vs. log(PMI), residualized" for Section 5.2.
- **Problem**: The section describes the PMI regression results in detail but never includes or references Figure 3, which the outline explicitly plans. The figure comment block at the end of the file lists Figures 2, 4, 5, 6 but skips Figure 3 entirely. This leaves a gap in the visual evidence for the paper's null result.
- **Fix**: Add `![Partial Regression Plot](figures/fig_partial_regression.pdf)` and a reference sentence (e.g., "Figure 3 plots the partial regression of absorption rate on log(PMI) after residualizing SAE configuration variables, showing a flat relationship consistent with partial $R^2 = 0.0006$.") in Section 5.2.

### Issue 2: "Key Positive Result" label on a result with acknowledged inflation
- **Location**: Section 5.3, heading
- **Quote**: "5.3 Absorption Taxonomy --- Key Positive Result"
- **Problem**: The section itself states "The Type II rate is likely inflated" and that parent features were identified by a heuristic rather than ground truth. The discussion (Section 7.4) further notes "A conservative interpretation... would only count Type I (3.8%) and Type III (0%) as validated absorption." Labeling this as the "Key Positive Result" when the conservative validated rate is 3.8% (not 92.3%) sets up a credibility problem with reviewers. The headline number is doing most of the rhetorical work, but the caveats undermine it.
- **Fix**: Change the heading to "5.3 Absorption Taxonomy --- Exploratory Result" or "5.3 Absorption Taxonomy --- Extended Measurement." Move the caveat paragraph to immediately after the 92.3% claim (before the figure reference), not after it. Consider adding a parenthetical to the 92.3% figure: "92.3% (upper bound; see caveat below)."

## Major Issues

### Issue 3: Safety probe experiment is confounded and detracts from H3
- **Location**: Section 5.4, paragraphs on safety probe
- **Quote**: "The probe gap is 0.118 (lowest absorption), 0.148 (median), and 0.051 (highest) --- not monotonically increasing with absorption."
- **Problem**: The three SAEs differ simultaneously in layer (5, 8, 10), width (768, 24k, 32k), and architecture --- making the comparison uninterpretable. The section acknowledges this ("making it impossible to isolate absorption's causal effect") but still presents the data as if it contributes evidence. With only 3 data points, 100 prompts, and completely confounded variables, this sub-experiment adds noise rather than signal. A reviewer will seize on this as evidence that the authors cannot control for confounds, potentially undermining the stronger SAEBench correlation result that follows the same H3 hypothesis.
- **Fix**: Either (a) move the safety probe results to an appendix and briefly note in Section 5.4 that a pilot safety experiment was inconclusive due to confounding (with appendix reference), or (b) retain it but explicitly label it as "Pilot: Underpowered Confounded Comparison" in the table heading and reduce its word count by 50%.

### Issue 4: Table/Figure numbering inconsistency with outline
- **Location**: Sections 5.3 and 5.4
- **Quote**: "Table 4: Absorption Taxonomy Summary" and "Table 5: Absorption vs. Downstream Correlation Matrix"
- **Problem**: The outline maps Table 4 to the downstream correlation matrix and Table 5 to the safety probe results. The section text swaps these: Table 4 is the taxonomy summary, Table 5 is the downstream correlations, and Table 6 is the safety probe. Similarly, Figure 5 in the outline is the downstream scatter but the text references it as an unnamed figure. This numbering inconsistency will cause confusion during LaTeX conversion and cross-section referencing.
- **Fix**: Either update the outline to match the section numbering (preferred, since the section ordering is logical) or renumber the section tables to match the outline. Ensure all figure references use consistent numbering.

### Issue 5: DAS($k$=3) estimation method inconsistency between Method and Experiments
- **Location**: Section 4.3 (H4 evaluation protocol)
- **Quote**: "$\text{DAS}(k{=}3)$ is estimated via logistic regression of parent activation on its top-3 children by $\alpha_{ij}$."
- **Problem**: The method section (3.2) defines DAS estimation as a four-step procedure involving McFadden pseudo-$R^2$ of logistic regression. The experiments section reduces this to a one-line description ("logistic regression of parent activation on its top-3 children") that omits the pseudo-$R^2$ step entirely. A reader who sees only Section 4.3 would not know how DAS is actually computed. More problematically, the DAS values in Table 7 (mean DAS($k$=3) $\approx$ 0.2--0.3) are never validated against any ground truth --- there is no way to know if these DAS values are meaningful at their reported magnitudes.
- **Fix**: Add a brief reference ("estimated via McFadden pseudo-$R^2$ as described in Section 3.2") in Section 4.3. In Section 5.5, add a sentence noting how DAS values should be interpreted (e.g., "DAS($k$=3) = 0.320 means the top-3 children collectively explain approximately 32% of the parent's activation variance").

### Issue 6: Decoder cosine pre-filter coverage is buried in ablations
- **Location**: Section 6.2
- **Quote**: "Coverage at the default $\cos(\mathbf{d}_i, \mathbf{d}_j) > 0.15$ threshold is 34.0% of absorbed pairs --- far below the 80% target."
- **Problem**: This is not an ablation finding --- it is a fundamental limitation of the H1 methodology. If the pre-filter only captures 34% of absorbed pairs, the LV detector can never achieve high recall regardless of the $\alpha_{ij}$ threshold. This should be prominently discussed in Section 5.1 (H1 results) as a primary failure mode, not relegated to an ablation. Reviewers will ask why the detector fails; this is arguably the main reason, and it is hidden.
- **Fix**: Add a paragraph to Section 5.1 after the main results: "A contributing factor to the low recall is the decoder cosine pre-filter ($\cos > 0.15$), which captures only 34.0% of absorbed pairs (Ablation A2). The majority of absorption occurs between latent pairs with decoder cosine similarity below 0.15, meaning the candidate set excludes most true positives before $\alpha_{ij}$ is even computed." Then cross-reference the ablation.

## Minor Issues
- **Section 4.1, paragraph 1**: "hook_resid_pre" and "hook_resid_post" are implementation details that interrupt the flow. Consider a footnote or a compact format like "`resid_pre` (pre-attention)" to clarify without cluttering.
- **Section 4.2**: "Letters A--M (13 letters) serve as the calibration set; letters N--Z (13 letters) serve as the held-out test set." This split is alphabetically ordered, which could introduce systematic bias (e.g., letter frequency correlates with alphabet position). A brief justification for why this split is acceptable (or acknowledgment that a random split would be preferable) would be helpful.
- **Section 4.3, H1**: "Success criterion: F1 $> 0.35$." The proposal states "F1 > 0.65" as the success criterion for H1. The section has already lowered this from the proposal --- this should be noted, or the proposal should be updated.
- **Section 4.4**: Baseline 2 is described as "the Chanin probe-directed metric" which "defines the target variable, not a competing predictor." If it is not a competing predictor, it is not a baseline in the usual sense. Relabel as "Ground Truth" rather than listing it as a baseline.
- **Section 5.1**: "The first $\alpha_{ij}$ bin ($[0, 0.1]$, $n = 369$) shows an anomalously high absorption rate of 0.848." This is an important diagnostic but the explanation (both features have similar low frequencies) is not immediately obvious. A one-sentence example would help: e.g., "For instance, two low-frequency sibling features for 'Q'-starting words both appear absorbed by the same parent."
- **Section 5.2**: "Per-layer stability (ablation A4)" --- this ablation is labeled A4 in the text but A4 does not appear in the ablation section (Section 6 has A1, A2, A3 only). Either add a Section 6.4 for A4 or relabel this as an inline analysis rather than an ablation reference.
- **Section 5.4, Table 5**: The header says "$n$" but uses different sample sizes across rows ($n = 54$, $n = 49$, $n = 54$, $n = 40$). A footnote explaining why SCR and unlearning have fewer observations would help.
- **Section 5.5**: "57.7% of letters have non-positive slope with width --- meeting the 60% target marginally." 57.7% does not meet 60%. This should say "approaching but not meeting the 60% target" or "falling just short of the 60% target."
- **Section 5.6**: The heading says "Cross-Architecture Generalization" but the subsection is numbered 5.6, not listed in the outline's Section 5. The outline lists it as "5.6 Cross-Architecture Generalization (H1 supplement)" which matches, but the experiments section text lacks a transition from 5.5 to 5.6 --- a bridging sentence would help.
- **Section 6.1**: "F1 peaks at $\tau = 0.4$ (test F1 = 0.136)" contradicts Section 5.1 which states the best threshold is $\tau = 0.5$ (F1 = 0.128). If 0.4 gives 0.136 which is higher than 0.128 at 0.5, then the main result should report the actual best threshold. Clarify whether 0.5 was the pre-registered threshold and 0.4 is the post-hoc best, or if this is an error.
- **Section 6, missing A4**: The outline mentions "Per-layer regression stability" as an ablation. Section 5.2 references it as "ablation A4." No A4 appears in Section 6.

## Visual Element Assessment
- [x] Figures/tables match outline plan (except Figure 3 is missing)
- [x] All visuals referenced before appearance (Figures 2, 4, 5, 6 are referenced in context)
- [x] Captions are self-explanatory (inline table headers serve this role)
- [ ] No text-heavy sections that need visual support --- Section 5.4's matched RAVEL comparison could benefit from a small bar chart showing the 5-vs-5 comparison
- [ ] Figure 3 (partial regression plot from outline) is entirely absent from the section

## What Works Well
1. **Transparent negative result reporting (Sections 5.1, 5.2).** The H1 and H2 results are presented without hedging or excuse-making. The sharpness test with AIC comparison is a clean diagnostic that directly tests the LV-specific prediction rather than just reporting low F1. The interpretation paragraph in 5.1 ("Absorption is better modeled as a graded phenomenon than a binary exclusion event") is crisp and honest.

2. **H3 statistical rigor (Section 5.4).** The downstream impact analysis is the paper's strongest section. Bonferroni correction, partial correlations, matched comparison with paired $t$-test, and Cohen's $d$ are all appropriate and well-reported. The partial correlation strengthening ($r = -0.595 \to r_{\text{partial}} = -0.661$) is a compelling finding that survives confound control.

3. **Pre-registered success criteria.** Every hypothesis has a quantitative pass/fail threshold stated before the results. This is rare in the interpretability literature and substantially increases the paper's credibility. The framework of "pre-registered prediction vs. observed outcome" makes the negative results (H1, H2) as informative as the positive ones (H3).
