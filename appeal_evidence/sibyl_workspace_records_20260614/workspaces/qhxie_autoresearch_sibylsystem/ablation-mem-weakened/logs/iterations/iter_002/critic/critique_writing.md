# Writing Critique: Local Inhibition Graph Framework

## Summary

The paper has undergone a major pivot from a null-result correlation study to the Local Inhibition Graph (LIG) framework. The writing is generally clear and well-structured, with good use of mathematical formalism and logical flow. However, critical writing issues persist that undermine the paper's credibility: (1) the significance tease from prior iterations survives in the new framing; (2) H6-H10 predictions are presented as established findings rather than proposed experiments; (3) the abstract overclaims by listing unvalidated contributions; (4) precision invariance is overstated by suppressing k=1 results; (5) the decision tree is biased toward "PROCEED" outcomes.

## Critical Issues

### 1. H6-H10 Predictions Presented as Established Findings (CRITICAL)

The paper repeatedly treats untested predictions as if they were validated results. Examples:

- Abstract: "The graph predicts known absorption pairs with enrichment over chance" --- this is a prediction, not a result.
- Section 5.3: "The competitive suppression framework explains all key findings" --- the framework has not been empirically tested.
- Section 6.3: "Practitioners can identify at-risk features without running absorption metrics" --- H8 has not been run.
- Section 8.2: "The graph predicts known absorption pairs with enrichment over chance and identifies at-risk features before running absorption metrics" --- listed as Contribution 2, but neither has been validated.

The paper does have a "Proposed Validation Experiments" subsection (5.2), but the surrounding text treats the predictions as conclusions. This creates a fundamental confusion about what has been done vs. what is planned.

**Fix**: Use future tense for all H6-H10 claims throughout the paper. Change "the graph predicts" to "we predict the graph will predict." Remove H10 from the contribution list until executed. Add a prominent disclaimer in the abstract: "The framework proposes five validation experiments (H6-H10) that remain to be executed."

### 2. Significance Tease Persists (CRITICAL -- Flagged in Prior Iterations)

The abstract and Section 5.1 foreground the uncorrected H1b trend (r=-0.431, p=0.028 at layer 8) with extensive discussion, despite explicitly stating it does not survive Bonferroni correction (p=0.334) or BH-FDR (q=0.167). The uncorrected p-value comes first; the correction appears second. For a paper that pivoted to a new framework partly because H1-H5 were null, this tease is especially counterproductive.

**Fix**: Lead with corrected results. State "No hypothesis survived multiple comparison correction" as the primary finding from H1-H5. Move uncorrected trends to a footnote. The LIG framework should be motivated by the precision-recall asymmetry (H5, genuinely supported) and the theoretical gap, not by a teased uncorrected trend.

## Major Issues

### 3. Abstract Overclaims by Listing Unvalidated Contributions

The abstract lists five contributions, four of which depend on H6-H10 validation:
1. "First connection between LCA lateral inhibition and SAE absorption" --- valid (theoretical)
2. "First local inhibition graph for SAE diagnostics" --- depends on H6
3. "Mechanistic explanation for precision-recall asymmetry" --- depends on H7
4. "First training-free post-hoc repair for absorption" --- depends on H10
5. "Integration of prior empirical findings into a unified framework" --- valid (retrospective)

Contributions 2, 3, and 4 are speculative. Listing them as established contributions is misleading.

**Fix**: Restructure contributions into "Theoretical Contributions" (1, 5) and "Proposed Empirical Contributions" (2, 3, 4) with explicit caveat that the latter depend on H6-H10 validation.

### 4. Precision Invariance Claim Is Overstated

Section 5.1 states "precision is nearly invariant" at k=5, with 21-25/26 features at 1.0. However, the k=1 results (precision_mean=0.897 at layer 4, with one feature at 0.0) are suppressed. The k-dependency is an informative finding that is hidden because it weakens the precision invariance claim central to the competitive suppression explanation.

**Fix**: Report precision at all k values (1, 5, 10, 20). Qualify the claim: "At k>=5, precision is near-invariant; at k=1, precision shows more variance." Discuss the k-dependency as an informative finding about probe richness.

### 5. "Exact" Structural Correspondence Claim Is Overstated for Untied SAE

The paper repeatedly calls the correspondence "exact" (abstract, Section 3.1, Section 6.1). However, gpt2-small-res-jb uses untied weights. The paper acknowledges this briefly ("Even with untied weights... the structural correspondence holds approximately") but the "exact" claim dominates.

**Fix**: Lead with "approximate" for the actual SAE used. State: "For tied-weight SAEs, the correspondence is exact. For the untied SAE used in this study (gpt2-small-res-jb), the correlation between W_dec^T W_dec and W_enc^T W_enc is r=X.XX, indicating the correspondence holds approximately."

### 6. Random Baseline Miscalculation

The proposal.md states expected precision@20 ~ 0.004 (20/24000), but the correct value for GPT-2 Small res-jb is 20/24576 = 0.000814. This 5x error inflates the enrichment claim and appears in multiple documents.

**Fix**: Correct all instances to 0.000814. Recalculate enrichment: precision@20 = 0.10 is 123x enrichment, not 25x.

## Minor Issues

7. **"Confound" language overstates causal claim**: The abstract says "identifying the random baseline confound as a critical issue." The paper showed metrics differ, not that raw metrics are causally confounded. Soften to "demonstrating that raw steering metrics mask feature-specific degradation."

8. **MCP caveat repetition**: The multiple comparison correction is mentioned in the abstract, Section 1.5, Section 5.1, and Section 6.2. Use brief cross-references after the first mention.

9. **Table 2 (integration table) presents predictions as explanations**: The table maps prior findings to "inhibition explanations" as if the mechanism is proven. Add a column labeled "Predicted Explanation (Pending H6-H10 Validation)" to make the speculative status clear.

## What Works Well

1. **Clear mathematical formalism**: The LCA dynamics, SAE forward pass, and structural correspondence are presented with proper notation and a proof sketch.
2. **Logical flow**: The paper moves from motivation to theory to methodology to results to discussion in a natural progression.
3. **Honest null-result reporting**: H1-H5 are reported accurately with exact statistics and proper corrections.
4. **Good use of tables**: The hypothesis summary table, integration table, and layer-level absorption table are clear and informative.
5. **Risk assessment**: The risk assessment table in the proposal is thorough and includes genuine failure modes.
