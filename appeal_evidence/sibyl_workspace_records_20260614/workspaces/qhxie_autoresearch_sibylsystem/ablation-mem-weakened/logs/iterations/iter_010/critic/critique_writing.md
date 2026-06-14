# Critique: Writing (Iteration 9/10)

## Overview

The paper has a fundamental structural problem: the title and framing ("Feature Absorption as Optimal Compression") imply success for the LCA-SAE correspondence framework, but the primary predictive hypothesis (H6: graph edges predict absorption pairs) is **falsified** with precision@20=0.0. The abstract and introduction overstate what was achieved.

## Critical Issues

### 1. Title-Content Mismatch (CRITICAL)

**Problem**: The title "Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts" claims an affirmative finding about absorption being optimal compression. But the LCA-SAE framework's primary predictive tool (the local inhibition graph) failed completely.

**What happened**:
- H6 (graph predicts absorption pairs): precision@20=0.0, p=1.0 - FALSIFIED
- H8 (graph predicts at-risk features): r=+0.12, p=0.55 - NOT SUPPORTED

The paper's intellectual hook (LCA connection providing mechanistic explanation) is legitimate, but the title implies the optimal compression framing is the empirical contribution, when really the empirical contribution is the null result + metric validation.

**Recommendation**: Choose one framing:
1. **Option A (Safe)**: "Feature Absorption Does Not Degrade SAE Interpretability: Null Results and Metric Validation" - emphasizes honest reporting
2. **Option B (Bold)**: "Competitive Suppression as the Mechanism Behind SAE Feature Absorption" - leads with the mechanistic explanation, explicitly states H6 falsified as informative negative result

### 2. Abstract Overclaims (CRITICAL)

**Problem**: The abstract says "precision@20 = 0.0" but frames it as a "falsified primary hypothesis" rather than the lead finding. It says "the mechanistic framework is strongly supported" but does not clarify that the predictive tool failed.

**Original abstract (lines 1-9)**:
> Feature absorption in sparse autoencoders (SAEs)---where general parent features fail to fire when more specific child features are present---has been characterized as a failure mode requiring architectural intervention. We present the first connection between the Locally Competitive Algorithm (LCA) from neuroscience and SAE feature absorption, showing that the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs.

**Issue**: This says "showing that G=W_dec^T W_dec is exactly the LCA inhibition matrix" - which is mathematically trivial for tied-weight SAEs (it's just the definition of G). The novelty claim is weak.

**Recommendation**: Rewrite abstract:
1. Lead with the null result: "We tested whether feature absorption degrades steering effectiveness and sparse probing accuracy in SAEs. Across 12 statistical tests with multiple comparison correction, zero hypotheses survived. Absorption does not significantly degrade downstream tasks."
2. Add metric validation: "Critically, random SAEs show 8x higher measured absorption than trained SAEs (0.278 vs 0.034, p<0.001), suggesting current metrics conflate dictionary structure with learned pathology."
3. Then introduce the mechanistic framework: "We show that decoder correlations encode competitive suppression relationships, explaining why absorption affects recall but not precision."

### 3. H1b Misrepresentation (CRITICAL)

**Problem**: Section 4.5 says "layer 8 exhibits the strongest absorption-steering correlation (r=-0.431, p=0.028 for delta-corrected steering, H1b)". This raw p-value is presented as a finding, but it does NOT survive Bonferroni correction (corrected p=0.334).

**Evidence from correlation_report_full.json**:
```json
{
  "name": "H1b_L8_Pearson",
  "p_value": 0.027825150948467228,
  "bonferroni_p": 0.33390181138160674,
  "bonferroni_rejected": false,
  "bh_qvalue": 0.16695090569080337,
  "bh_rejected": false
}
```

The paper should never cite this as evidence of an effect.

**Recommendation**: Remove H1b as supporting evidence. Add explicit statement: "No hypotheses survived multiple comparison correction (Bonferroni alpha=0.00417 for 12 tests). The uncorrected p=0.028 at layer 8 for delta-corrected steering is non-significant."

## Major Issues

### 4. Figure/File Reference Inconsistency (MAJOR)

**Problem**: Section 4.4 (line 251) says: "![Precision-recall asymmetry in k-sparse probing. Precision remains near 1.0 across all features while recall varies widely, consistent with competitive suppression reducing coverage without affecting selectivity.](figures/fig7_precision_recall.pdf)"

But Section 6 (line 380) says: "Figure 3: `fig7_precision_recall.pdf` --- Precision-recall asymmetry scatter plot"

This is internally consistent but confusing. The text says "Figure 3" but the file is named "fig7_precision_recall.pdf". More critically, I cannot verify these figures exist in the workspace.

**Recommendation**: Check that `writing/latex/figures/` or `exp/results/figures/` contains these files. If not, generate them or remove references.

### 5. Table 3 Values Unverified (MAJOR)

**Problem**: Table 3 in Section 4.2 reports clustering coefficient (0.002--0.005) and std edge weight (0.089--0.102). The review feedback flags these as "not from verifiable source".

**Recommendation**: Add footnote to Table 3: "Computed from decoder correlation graph using networkx. Source: exp/results/full/h6_inhibition_graph.json, rows with layer=l8, columns [clustering_coefficient, edge_weight_std]."

## Summary

The writing has a core framing problem: it presents the LCA-SAE connection as the contribution while burying the fact that the primary predictive tool failed. The null results and metric validation are actually the strongest parts of the paper, but they're treated as secondary findings. The paper should lead with what was actually found, not what was hypothesized.

**Action items**:
1. Rewrite abstract to lead with null result + metric validation
2. Choose title that matches actual findings
3. Remove H1b p=0.028 as evidence (not significant after MCP)
4. Verify and fix figure references
5. Add data provenance to Table 3