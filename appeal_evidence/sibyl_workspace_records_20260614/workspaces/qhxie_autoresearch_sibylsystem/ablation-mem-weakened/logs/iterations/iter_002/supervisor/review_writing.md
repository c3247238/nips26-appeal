# Supervisor Review: The Local Inhibition Graph Framework

**Reviewer**: Independent senior research supervisor (NeurIPS/ICML-calibrated)
**Date**: 2026-04-29
**Iteration**: 2 (LIG Framework Pivot)
**Overall Score**: 5.5 / 10 (Reject)
**Verdict**: CONTINUE (execute gatekeeper experiment H6)

---

## Executive Summary

This paper proposes the Local Inhibition Graph (LIG) framework, connecting Rozell et al.'s Locally Competitive Algorithm (LCA) to feature absorption in Sparse Autoencoders. The core theoretical insight---that W_dec^T W_dec is exactly the LCA inhibition matrix for tied-weight SAEs---is genuinely novel and has not been articulated in the SAE literature. The framework provides a mechanistic explanation for absorption as competitive suppression and yields testable predictions about decoder correlations predicting absorption pairs.

However, the paper suffers from a fundamental problem: **the gatekeeper experiment (H6) has not been executed, yet the paper presents its predictions as established findings throughout.** The abstract claims "the graph predicts known absorption pairs with enrichment over chance," Section 5.3 states "the competitive suppression framework explains all key findings," and the contribution list includes "first local inhibition graph for SAE diagnostics" and "first training-free post-hoc repair"---all depending on experiments that have not been run.

This is not a minor issue. It is a structural problem that undermines the entire empirical contribution. A paper that presents predictions as results is misleading regardless of how elegant the theory is.

The theoretical contribution alone (the LCA-SAE correspondence) is worth approximately a 5.5. With H6 validated (precision@20 >= 0.10, representing 123x enrichment over chance), the score would rise to 7.5-8.0. Without H6, this paper cannot be submitted.

---

## Dimension Scores

| Dimension | Score | Weight | Rationale |
|-----------|-------|--------|-----------|
| Novelty & Significance | 7 | High | The LCA-SAE connection is genuinely new. The competitive suppression mechanism is elegant. However, the novelty claim depends on H6 validation---without it, the correspondence risks being dismissed as "rebranding." The precision-recall asymmetry explanation is compelling but also untested. |
| Technical Soundness | 6 | High | The mathematical correspondence is correct for tied-weight SAEs. The proof sketch in Section 3.1 is sound. However: (1) the "exact" claim is overstated for untied SAEs (the actual case), (2) H10 has sign ambiguity, (3) the significance tease persists, (4) H8 is circular. |
| Experimental Rigor | 3 | High | H1-H5 were executed and reported honestly. But H6-H10---the experiments that validate the new framework---have NOT been executed. The paper presents predictions as findings. This is a critical failure of experimental rigor. The random baseline miscalculation (0.004 vs 0.000814) further undermines credibility. |
| Reproducibility | 5 | Medium | H1-H5 methodology is described in detail. But H6-H10 are described as "proposed" while their predictions are presented as results, creating confusion. No code repository link. No SAELens version pinned. The untied-weight approximation quality is not quantified. |

**Overall**: 5.5 (weighted average, calibrated to NeurIPS standards)

---

## Cross-Validation of Paper Claims Against Raw Data

I independently verified the paper's reported numbers against the source JSON files. Here are the findings:

### Verification 1: H1-H5 Statistics (CORRECT)

**Paper claims:** H1b at layer 8: r=-0.431, uncorrected p=0.028; Bonferroni p=0.334; BH-FDR q=0.167

**Source data (correlation_report_full.json):**
- pearson_r: -0.4312704523611132
- pearson_p: 0.027825150948467228
- bonferroni_p: 0.33390181138160674
- bh_qvalue: 0.16695090569080337

**Verdict**: Numbers match exactly. Paper correctly reports that H1b does not survive correction.

### Verification 2: Precision-Recall Decomposition (PARTIALLY OVERSTATED)

**Paper claims (Section 5.1):** "Precision is nearly invariant across features: 21/26 features at layer 4 and 25/26 at layer 8 achieve perfect precision (1.0)."

**Source data (precision_recall_analysis.json):**
- Layer 4, k=5: precision_mean=0.9745, n_precision_one=21/26, precision_min=0.818
- Layer 8, k=5: precision_mean=0.9945, n_precision_one=25/26, precision_min=0.857
- Layer 4, k=1: precision_mean=0.897, n_precision_one=22/26, precision_min=0.0
- Layer 8, k=1: precision_mean=0.954, n_precision_one=24/26, precision_min=0.0

**Verdict**: The k=5 claim is accurate. However, the k=1 results (precision_mean=0.897 at layer 4, with one feature at 0.0) are suppressed. The precision invariance claim only holds at k>=5, which is central to the competitive suppression explanation but is not disclosed as k-dependent.

### Verification 3: Random Baseline Miscalculation (CONFIRMED)

**Paper/proposal claims:** Expected precision@20 ~ 0.004 (20/24000)

**Correct value:** GPT-2 Small res-jb has d_dict=24576, so expected precision@20 = 20/24576 = 0.000814

**Verdict**: The paper understates the chance baseline by 5x, inflating the enrichment claim. Precision@20 = 0.10 would be 123x enrichment over chance, not 25x.

### Verification 4: EC50 Feature U Contradiction (CONFIRMED, NOT HIGHLIGHTED)

**Source data (ec50_analysis.json):**
- Feature U at layer 8: absorption=0.242 (highest), EC50=9.17 (lowest in layer)

**Verdict**: This striking null result directly contradicts H4 (absorption reduces steering efficiency) but is not highlighted in the LIG framing paper. It was prominently discussed in prior iterations.

### Verification 5: H6-H10 Execution Status (NOT EXECUTED)

**Paper claims:** "The graph predicts known absorption pairs with enrichment over chance" (abstract)

**Actual status:** No H6-H10 results exist in exp/results/. The experiments are described in Section 5.2 as "proposed validation experiments" but their predictions are presented as established findings throughout the paper.

**Verdict**: This is the most critical verification failure. The paper's central empirical claims are entirely speculative.

---

## Critical Issues (Would Trigger Rejection)

### 1. H6 Not Executed, Yet Predictions Presented as Findings (Critical)

The gatekeeper experiment---testing whether decoder correlations predict absorption pairs---has not been run. Yet the paper states:

- Abstract: "The graph predicts known absorption pairs with enrichment over chance"
- Section 5.3: "The competitive suppression framework explains all key findings"
- Section 6.3: "Practitioners can identify at-risk features without running absorption metrics"
- Section 8.2 (Contributions): "The graph predicts known absorption pairs with enrichment over chance and identifies at-risk features before running absorption metrics"

Section 5.2 is correctly titled "Proposed Validation Experiments," but the surrounding text treats H6-H10 predictions as conclusions. This creates a fundamental confusion about what has been done vs. what is planned.

**Impact**: Reviewers will correctly identify this as presenting predictions as results. This is a fatal flaw for any empirical paper.

**Fix**: Execute H6 immediately. Until then, use future tense for ALL H6-H10 claims. Restructure contributions into "Theoretical Contributions" (validated) and "Proposed Empirical Contributions" (pending H6-H10). Add a prominent disclaimer in the abstract.

### 2. "Exact" Correspondence Claim Overstated for Untied SAE (Critical)

The paper repeatedly calls the correspondence "exact" (abstract, Section 3.1, Section 6.1). However, gpt2-small-res-jb uses untied weights. The paper acknowledges this briefly ("Even with untied weights... the structural correspondence holds approximately") but the "exact" claim dominates.

**Impact**: Reviewers familiar with SAE architectures will immediately note that standard trained SAEs use untied weights. The "exact" claim is misleading for the actual experimental system.

**Fix**: Quantify the approximation. Report the correlation between W_dec^T W_dec and W_enc^T W_enc for gpt2-small-res-jb. Lead with "approximate" for the actual SAE.

### 3. Significance Tease Persists (Critical)

The paper foregrounds the uncorrected H1b trend (r=-0.431, p=0.028 at layer 8) with extensive discussion in Section 5.1 and the abstract, despite explicitly stating it does not survive Bonferroni correction (p=0.334) or BH-FDR (q=0.167). The uncorrected p-value comes first; the correction appears second. This was flagged in iterations 0-2 but persists in the new LIG framing.

**Impact**: For a paper that pivoted to a new framework partly because H1-H5 were null, this tease is especially counterproductive.

**Fix**: Lead with corrected results. State "No hypothesis survived multiple comparison correction" as the primary finding from H1-H5. Move uncorrected trends to a footnote.

### 4. H8 Circularity (Critical)

H8 (graph predicts at-risk features) uses the same data for both graph construction and validation. Correlating total_inhibition (derived from decoder correlations) with absorption_rate (also derived from decoder correlations and activations) on the same 26 features is methodologically invalid.

**Impact**: This is a textbook case of circular analysis. Reviewers will reject any claims based on H8.

**Fix**: Use LOOCV or cross-layer prediction. If n=26 is too small, treat H8 as purely exploratory.

---

## Major Issues (Significantly Weaken the Paper)

### 5. H10 Sign Ambiguity

The rebalancing formula z'_i = z_i + alpha * inh_i increases parent activation when G_ij > 0 and z_j > 0. But the mechanism is unclear---this is "arbitrary activation boosting" rather than homeostatic rebalancing.

**Fix**: Test both additive and subtractive rules empirically. If neither works, drop H10 claims.

### 6. Random Baseline Miscalculation

Expected precision@20 is stated as ~0.004 (20/24000) but the correct value is 20/24576 = 0.000814. This 5x error inflates the enrichment claim.

**Fix**: Correct all instances. Recalculate: precision@20 = 0.10 is 123x enrichment, not 25x.

### 7. Underpowered Sample Size

With n=26 features and 4-6 high-absorption features per layer, power to detect |r| >= 0.3 is approximately 25%. The paper acknowledges this for H1-H5 but makes correlation-based claims for H7-H8 without power analysis.

**Fix**: Add power analysis for H7-H8. Treat as exploratory if power < 50%.

### 8. Precision Invariance Overstated

The paper reports precision at k=5 but suppresses k=1 results (precision_mean=0.897 at layer 4, with one feature at 0.0). The precision invariance claim is central to competitive suppression but only holds at k>=5.

**Fix**: Report precision at all k values with discussion of k-dependency.

### 9. Novelty Risk: "Rebranding" Critique

The LCA-SAE correspondence is exact by definition for tied-weight SAEs. Its scientific value depends entirely on whether it yields validated predictions (H6). Without H6, reviewers may dismiss it as "rebranding."

**Fix**: Acknowledge this critique explicitly. Frame the correspondence as a "productive theoretical bridge" rather than a "discovery."

### 10. Graph Specificity Untested

The inhibition graph may predict ANY correlated decoder pair, not specifically absorption pairs. Without a non-absorbed correlated pair control, enrichment could reflect generic correlation structure.

**Fix**: Add non-absorbed pair control to H6.

### 11. Decision Tree Bias

Even "H6 not validated" leads to "retain as theoretical speculation" rather than a clean pivot. The partial-validation branch is ambiguous.

**Fix**: Sharpen thresholds: precision@20 < 0.05 -> PIVOT. 0.05-0.10 -> diagnostic-only. >= 0.10 -> full framework.

---

## Minor Issues

- **Pythia cross-validation buried**: r=-0.041, p=0.841 shows no signal and supports the null-result interpretation but is not discussed.
- **Feature U EC50 contradiction not highlighted**: Lowest EC50 (9.17) despite highest absorption (24.2%) directly contradicts H4.
- **"Confound" language overstates causal claim**: The abstract uses "random baseline confound" but the paper only shows metrics differ, not causal confounding.
- **No SAELens version pinned**: Reproducibility requires exact versions.
- **H10 presented as contribution**: Listed in abstract and conclusion despite being entirely speculative.

---

## What Works Well

1. **Genuinely novel theoretical insight**: The LCA-SAE structural correspondence has not been articulated before.
2. **Elegant mechanistic explanation**: Competitive suppression naturally explains precision-recall asymmetry.
3. **Honest null-result reporting**: H1-H5 are reported accurately with proper corrections.
4. **Training-free methodology**: The graph is computationally feasible and practical.
5. **Clear mathematical formalism**: The proof sketch and notation are sound.
6. **Good use of tables**: The integration table (Table 2) is conceptually useful.

---

## Risks for Submission

1. **H6 failure**: If precision@20 < 0.05, the entire empirical contribution collapses.
2. **"Rebranding" critique**: Without H6, the correspondence is a definitional identity, not a discovery.
3. **Concurrent work**: Korznikov et al. (2026) challenges SAE credibility, potentially rendering absorption less relevant.
4. **Tang et al. (2025)**: Alternative theoretical explanation for absorption (spurious local minima) competes with the inhibition framework.
5. **Small feature set**: n=26 is underpowered for correlation analyses even if H6 validates.

---

## Path to Improvement

**To raise score to 7.5 (Borderline Accept)**:
- Execute H6 and report precision@k, recall@k, Fisher exact test
- Quantify tied-vs-untied approximation quality
- Add non-absorbed pair control to H6
- Fix precision invariance claim (report all k values)
- Remove/reframe all H6-H10 predictions that have not been executed
- Sharpen decision tree with commit-to-pivot thresholds
- Eliminate significance tease language

**To reach 8.0+ (Accept)**:
- All of the above, PLUS:
- H6 validates with precision@20 >= 0.10 (123x enrichment)
- Execute H7-H8 with proper controls (LOOCV for H8)
- Add cross-model validation (Gemma-2-2B or Pythia with same metric)
- Expand feature set beyond first-letter (WordNet hierarchies)

**Alternative path**: If H6 fails (precision@20 < 0.05), PIVOT to Alternative C (Trade-off Analysis) as recommended by the result debate. This reframes absorption as a predictable trade-off consequence and uses the same data.

---

## Final Verdict

**Score: 5.5 / 10 (Reject)**

The Local Inhibition Graph framework is a genuinely novel theoretical contribution. The LCA-SAE structural correspondence is elegant, the competitive suppression mechanism is compelling, and the precision-recall asymmetry explanation is natural. However, **the paper presents predictions as established findings**, which is a fatal flaw for any empirical submission.

The gatekeeper experiment (H6) is a ~15-minute computation on pretrained weights. It has not been executed. Until it is, the paper's empirical claims are entirely speculative.

**Recommendation**: Execute H6 immediately. If precision@20 >= 0.10, proceed with full framework validation. If precision@20 < 0.05, pivot to Alternative C (Trade-off Analysis) or retain only the theoretical correspondence as a speculative discussion point.

The theoretical contribution alone is not sufficient for a top-tier venue. The field values theory that generates validated predictions. Execute the predictions.
