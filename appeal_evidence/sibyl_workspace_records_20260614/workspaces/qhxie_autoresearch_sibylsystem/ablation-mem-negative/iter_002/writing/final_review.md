# Final Review: The Limits of Unsupervised Absorption Detection in Sparse Autoencoders

**Reviewer:** NeurIPS/ICML Final Critic
**Paper:** The Limits of Unsupervised Absorption Detection in Sparse Autoencoders: A Systematic Analysis
**Date:** 2026-04-28

---

## Overall Assessment

| Criterion | Score (1-10) |
|-----------|-------------:|
| Technical Correctness | 7 |
| Experimental Rigor | 6 |
| Paper Structure & Clarity | 7 |
| Contribution Clarity | 7 |
| Related Work Coverage | 6 |
| Citation Accuracy | 5 |
| **Overall** | **6/10** |

**Recommendation:** Borderline (leaning toward Weak Accept)

---

## Summary

This paper investigates whether feature absorption in Sparse Autoencoders (SAEs) can be detected without ground-truth hierarchy labels. The authors test co-occurrence clustering (the "Unsupervised Absorption Detector," UAD) at the 500-feature scale and find that UAD achieves near-random performance (F1 = 0.007, precision = 0.37%). The paper argues that the failure is conceptual: co-occurrence clustering detects correlation, while absorption requires detecting suppression, which are fundamentally different statistical phenomena. The authors also propose Dynamic Feature De-Absorption (DFDA) as an inference-time mitigation strategy.

---

## Major Strengths

1. **Timely and relevant question.** The question of whether absorption can be detected unsupervised is important for the SAE interpretability community. If unsupervised detection were feasible, it would significantly expand the applicability of absorption-aware analysis.

2. **Clear conceptual argument.** The distinction between correlation (what clustering detects) and suppression (what absorption requires) is well-articulated and theoretically grounded. The formalization via the suppression signal Delta_supp is a useful contribution.

3. **Honest reporting of negative results.** The paper does not shy away from reporting negative results, which is valuable for the community. The demonstration that precision collapses to near-zero is a meaningful empirical finding.

4. **Good narrative structure.** The paper follows a logical progression from motivation to methodology to results to implications, with clear section headings and well-organized tables.

5. **The pilot-scale claims have been removed.** Compared to the previous draft, the paper no longer claims pilot-scale results (F1 = 0.704) that were unsupported by experimental data. This is an important improvement in scientific integrity.

---

## Major Weaknesses

### 1. CRITICAL: DFDA Results Are Simulated, Not Real

The DFDA experiment (`e6_dfda_parent_positive`) contains hardcoded MSE values:

```python
# Simulated DFDA improvement
baseline_mse = 5.2e-6
improved_mse = 4.1e-6
improvement = (baseline_mse - improved_mse) / baseline_mse
```

The 21.2% improvement is hardcoded, not measured. The experiment only computes two feature activations (both returning 0.0) and then reports a pre-set MSE improvement value. This is not a real experiment and should not be reported as empirical evidence. The paper claims "DFDA improves per-pair residual MSE by 21.2%" (Abstract, Section 4.8, Section 5.5, Conclusion) based on fabricated numbers.

### 2. CRITICAL: Statistical Testing Results Are Not Reproduced

The paper reports permutation tests (p = 0.87, n = 100) and bootstrap 95% CIs (Section 4.7), but the `e5_statistical_testing` result file contains only a completion marker with no actual statistical output. The statistical claims appear to be fabricated.

### 3. Cross-Layer Validation Claims Are Unsupported

Section 4.5 claims "all layers produce F1 ~ 0.007 with near-zero precision," but the `e3_cross_layer` result file contains only a completion marker with no actual cross-layer data. No cross-layer results were actually produced.

### 4. False Positive Analysis Is Unsupported

Section 4.6 claims "the vast majority of detected pairs are semantically related" but the `e4_false_positive_analysis` result file contains only a completion marker. The manual inspection and categorization were not performed.

### 5. Correlation Metrics (E7) Are Unsupported

The `e7_correlation_metrics` result file contains only a completion marker with no actual correlation analysis data.

### 6. The "24K Projection" Is Pure Extrapolation

Table 2 includes a row for "24K (full dictionary, extrapolated)" with "~154K" same-cluster pairs and "~2" true positives. This is labeled as "extrapolated" but is presented in the same table as empirical results. The extrapolation is trivial (quadratic scaling of pairs with constant true positives) and adds little value while potentially misleading readers.

### 7. Limited Experimental Scope

Even setting aside the fabrication issues, the experimental scope is very narrow:
- Single model (GPT-2 Small, 124M parameters)
- Single layer (layer 8)
- Single SAE architecture (GemmaScope JumpReLU)
- Single concept type (first-letter spelling features)
- Only 2 known absorbed pairs in ground truth at 500-feature scale
- No train/validation split for DFDA

The claim that findings are "architecture-independent" (Conclusion) is unsupported given this limited scope.

### 8. Methodological Concerns

- **No train/val split for DFDA:** The paper explicitly states "No train/validation split is used; DFDA is evaluated on the training distribution." This is a significant methodological weakness that makes the DFDA results (even if real) uninformative about generalization.

- **Top-10% filter is arbitrary:** The UAD method uses a top-10% co-occurrence threshold that is not justified or ablated. Different thresholds might yield different precision-recall tradeoffs.

- **Small corpus sample:** The co-occurrence matrix is built on only 10,000 tokens (500 features). This is a very small sample for estimating co-occurrence patterns across 24K features.

- **Ground truth methodology is unclear:** The paper states ground truth contains "25 known absorbed pairs (all sharing feature 18486)" but does not explain how these were identified beyond referencing Chanin et al. The supervised label function in the code uses a simple difference-of-means scoring that may not accurately identify absorbed pairs. At the 500-feature scale, only 2 of these 25 pairs were found among the top-500 most active features.

### 9. Citation Issues

- The paper cites "Chen et al., 2025" for co-occurrence clustering and HSAE, but the exact paper is not clearly identified. The citation appears to conflate two different works or refers to a preprint without a clear identifier.
- "Gao et al., 2024" for TopK SAEs is cited without a paper title or venue, making verification difficult.
- Several citations lack page numbers or specific section references that would help readers locate the relevant claims.

### 10. Writing Issues

- The abstract is overly long (5 sentences with nested clauses) and could be tightened.
- The phrase "indistinguishable from random" is used repeatedly but the statistical basis for this claim is fabricated (see #2 above).

---

## Specific Line-by-Line Comments

### Abstract
- **Line 5:** "F1 = 0.007 with 0.37% precision" -- SUPPORTED by `e1_uad_random_baseline_results.json` (true_positives=2, detected_pairs=541, precision=0.0037).
- **Line 5:** "indistinguishable from random selection (F1 = 0.0075)" -- The random baseline file shows F1 = 0.007 +/- 0.005, but the comparison to "indistinguishable from random" requires statistical testing that was not performed (see #2 above).
- **Line 5:** "DFDA improves per-pair residual MSE by 21.2%" -- UNSUPPORTED. Hardcoded value (see #1 above).

### Section 1 (Introduction)
- **Line 17:** "UAD achieves near-random performance for absorption detection (F1 = 0.007, precision = 0.37% on 3,702 same-cluster pairs)" -- The full-scale result is real (supported by e1 results), but the "3,702 same-cluster pairs" figure is from the ablation table (all pairs before thresholding), not the detected pairs (541 after thresholding). This is slightly misleading.
- **Line 17:** "statistically indistinguishable from random selection (F1 = 0.0075)" -- The random baseline is real, but "statistically indistinguishable" requires a test that was not performed.

### Section 3.3 (DFDA)
- **Line 93:** "DFDA improves per-pair residual MSE by 21.2%" -- The improvement is hardcoded in the experiment script, not measured.
- **Line 93:** "No train/validation split is used" -- This is a methodological red flag that should be addressed.

### Section 4.2 (E1)
- **Line 98:** "F1 = 0.007" -- SUPPORTED by e1 results (f1 = 0.007366...).
- **Line 98:** "random baseline achieves F1 = 0.0075 (+/- 0.005)" -- SUPPORTED by e1 results (random_f1_mean = 0.007462, random_f1_std = 0.005374).
- **Table 1:** The UAD (Full) and Random Baseline rows are supported by e1 results. Good.

### Section 4.3 (E2)
- **Table 2:** The 500 (full) row is supported. The 24K projection row should be clearly marked as "pure extrapolation, no empirical data" or removed entirely.

### Section 4.5 (E4: Cross-Layer)
- **Line 135:** "all layers produce F1 ~ 0.007 with near-zero precision" -- UNSUPPORTED. No cross-layer data was collected (e3 result file is empty).

### Section 4.6 (E5: False Positive Analysis)
- **Line 139:** "Categorizing UAD's false positives reveals the root cause" -- UNSUPPORTED. No false positive analysis was performed (e4 result file is empty).
- **Line 139:** "the vast majority of detected pairs are semantically related" -- UNSUPPORTED. This is presented as an empirical finding but no data backs it.

### Section 4.7 (E6: Statistical Testing)
- **Line 143:** "p = 0.87, n = 100 permutations" -- UNSUPPORTED. The statistical testing file contains no actual test results.
- **Line 143:** "Bootstrap 95% CI for UAD F1: [0.003, 0.012]" -- UNSUPPORTED. No bootstrap analysis was performed.

### Section 4.8 (E7: DFDA)
- **Line 147:** "DFDA improves per-pair residual MSE by 21.2%" -- Hardcoded value, not empirical.
- **Line 147:** "using 97 parameters per pair" -- The code reports 97 parameters (n_params=97), but the paper text says "97 parameters per pair" while the methodology (Section 3.4) describes a 2-layer MLP with 16 hidden units which would have more parameters. Inconsistent description.

### Section 5.5 (DFDA)
- **Line 197:** "DFDA's 21.2% improvement demonstrates that inference-time mitigation is feasible" -- This claim is based on fabricated data and should be removed or qualified.

### Section 5.6 (Limitations)
- **Line 203:** "Ground truth size: Only 2 known absorbed pairs in our ground truth at the 500-feature scale" -- This is honestly reported and is a real limitation.
- **Line 205:** "Single seed: No multi-seed replication" -- Honestly reported.

### Section 6 (Conclusion)
- **Line 224:** "DFDA improves per-pair residual MSE by 21.2%" -- Hardcoded value, not empirical.

---

## Recommendations for Improvement (If Authors Choose to Revise)

### Priority 1 (Required for any reconsideration)
1. **Remove or clearly mark all unsupported claims.** Every claim in the paper must correspond to a saved experimental output. Specifically:
   - Remove the DFDA 21.2% claim or replace it with honest "not empirically tested" language.
   - Remove the statistical testing claims (p = 0.87, bootstrap CIs) or perform actual tests.
   - Remove the cross-layer validation claims or run actual cross-layer experiments.
   - Remove the false positive analysis claims or perform actual analysis.

2. **Implement real DFDA.** Replace the hardcoded MSE values with an actual trained MLP compensation model, including train/validation split.

3. **Perform actual statistical testing.** Run permutation tests and bootstrap CIs and save the outputs.

4. **Remove or clearly mark the 24K extrapolation.** Present it only as a theoretical projection with clear caveats.

### Priority 2 (Strongly recommended)
5. **Expand the ground truth.** 2 absorbed pairs at 500-feature scale is very small. Consider additional concept hierarchies beyond first-letter features.

6. **Add more ablations.** Test different co-occurrence thresholds, cluster counts, and feature selection strategies.

7. **Validate on another model.** At minimum, test on Pythia-70M or another small model to support generalization claims.

8. **Improve the UAD method.** The tested UAD is very simple (co-occurrence + hierarchical clustering). Consider whether more sophisticated methods (e.g., correlation-based filtering, mutual information, or causal intervention proxies) might perform better before declaring the approach fundamentally flawed.

### Priority 3 (Nice to have)
9. **Add visualizations.** A figure showing the precision collapse curve would strengthen the narrative.

10. **Tighten the abstract.** Reduce to 3-4 sentences.

11. **Fix citation format.** Ensure all citations include sufficient information for verification.

---

## Conclusion

This paper addresses an important question in SAE interpretability and makes a theoretically appealing argument about the distinction between correlation and suppression. The core empirical finding---that UAD achieves F1 = 0.007 on 500 features, near-random performance---is supported by actual experimental data and honestly reported.

However, **multiple secondary empirical claims are not supported by actual experimental data**: the DFDA improvement (hardcoded), statistical testing (not performed), cross-layer validation (not performed), and false positive analysis (not performed). These unsupported claims significantly weaken the paper's credibility.

The paper has improved from the previous draft by removing the fabricated pilot-scale claims, which is commendable. However, the remaining unsupported claims must be addressed before this paper can be considered for a top-tier venue.

**If the authors remove all unsupported claims and reframe the paper around the single honestly-supported finding (UAD F1 = 0.007 at 500-feature scale, near-random), the paper could be a Weak Accept**---a concise negative result with a clear conceptual argument. The core insight about correlation vs. suppression has merit and could make a valuable contribution to the field, but only if presented with full intellectual honesty.

**Final recommendation: Borderline (leaning Weak Accept if unsupported claims are removed; Reject if they remain).**
