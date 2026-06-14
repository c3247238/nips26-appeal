# Methodologist Audit: Local Inhibition Graph Proposal

## Scope

This audit examines the experimental methodology of the proposed "Local Inhibition Graph" study (iter_009+), which pivots from a prior correlation-study direction (iter_001-008). The audit covers both the **existing empirical data** (iter_001-008) that the new proposal builds upon and the **proposed methodology** for the inhibition graph experiments.

---

## 1. Baseline Fairness Audit

### 1.1 Existing Data (iter_001-008): Absorption vs. Downstream Tasks

| Baseline / Comparison | Fairness Assessment | Issue Severity |
|---|---|---|
| **Random baseline for steering** | The delta-corrected metric (H1b) subtracts a random-word steering baseline from feature-specific steering. This is methodologically sound---it isolates the specific effect from non-specific activation boosts. | None |
| **Same data splits?** | All experiments use the same 20-word vocabulary per letter, same 100 samples, same seed (42). Consistent across layers and tasks. | None |
| **Published numbers comparison** | No external published numbers are used as baselines. All numbers are internally computed. | N/A |
| **Steering strength sweep** | Six strengths tested ([1.0, 2.0, 5.0, 10.0, 20.0, 50.0]), but success is binary at strength=50. No analysis of whether the *shape* of the dose-response curve differs by absorption level. | Minor |

**Finding:** The baselines in the existing data are generally fair. The delta-correction (random baseline subtraction) is a strength, not a weakness. However, the EC50 analysis (H4) uses a Hill fit with fixed `n=5` for all features, which may overfit some curves and underfit others---the reported R^2 values are suspiciously high (0.99+), suggesting the model may be fitting noise.

### 1.2 Proposed Inhibition Graph Study (iter_009+)

| Baseline | Fairness Assessment | Issue Severity |
|---|---|---|
| **Random graph** | Shuffle latent indices; expected precision@20 ~ 0.004. This is a valid and necessary baseline. | None |
| **Non-absorbed pair control** | Test graph edges for correlated but non-absorbed pairs. Critical for ruling out "any correlation predicts absorption." | None |
| **Identity graph** | Only self-loops. Tests whether correlations beyond self-similarity matter. Less critical but harmless. | None |
| **Same SAE, same layer?** | The proposal uses the same SAE (gpt2-small-res-jb) as the ground truth absorption data. This is correct---the graph is constructed from the same weights used to generate the activations. | None |

**Concern:** The random baseline precision@20 = 20/24000 = 0.00083, not 0.004 as stated in the proposal. With 24K latents and 20 neighbors, the expected precision under random guessing is 20/24000 = 0.00083. The proposal states "~0.004 chance" which appears to be a 5x overestimate. This inflates the apparent magnitude of the effect.

---

## 2. Metric-Claim Alignment

### 2.1 Existing Data

| Claim | Metric | Alignment | Gap |
|---|---|---|---|
| "Absorption degrades steering" | Steering success rate at strength=50 | Misaligned: steering success is binary and ceiling-biased (most features achieve 100% at strength=50). A continuous metric (e.g., EC50, probability lift) would better capture degradation. | Steering success is too coarse |
| "Absorption degrades probing" | k-sparse probe F1 at k=5 | Partially aligned: F1 captures both precision and recall, but the claim is about degradation, and F1 variance is driven more by feature selectivity than absorption. | F1 conflates multiple factors |
| "Precision is invariant" | Precision at k=5,10,20 | Well-aligned: precision measures selectivity, and the data show near-perfect precision across all features. | None |
| "Recall varies" | Recall at k=5,10,20 | Well-aligned: recall measures coverage, and the data show wide variance. | None |
| "Delta-corrected correlation" | Pearson r on (steering_success - baseline_success) vs. absorption_rate | Well-aligned: isolates specific effect. | None |

**Critical Gap:** The steering metric (success rate at strength=50) has a ceiling effect. Feature U (24.2% absorption) still achieves 100% steering success. This does not mean absorption has no effect---it means the metric is insensitive at high steering strength. The EC50 analysis was intended to address this but produced null results, possibly due to the fixed Hill parameter.

### 2.2 Proposed Inhibition Graph Study

| Claim | Metric | Alignment | Gap |
|---|---|---|---|
| "Graph edges predict absorption pairs" | Precision@k, Recall@k, AUPR | Well-aligned: directly tests whether graph neighbors are absorption pairs. | None |
| "Inhibition explains precision-recall asymmetry" | Correlation(total_inhibition, recall) and correlation(total_inhibition, precision) | Well-aligned: tests the mechanistic prediction that inhibition reduces recall but not precision. | None |
| "Graph predicts at-risk features" | Correlation(graph_stats, absorption_rate) | Partially aligned: absorption_rate is the outcome, but "at-risk" implies prediction *before* running the Chanin metric. The proposal uses the same data for both graph construction and absorption detection---this is circular if not carefully handled. | Risk of data leakage |
| "Homeostatic rebalancing restores parent firing" | Parent firing rate before/after, reconstruction error | Well-aligned: directly tests the repair mechanism. | None |

**Critical Gap for H8:** The proposal states "Can the inhibition graph predict which features are at risk of absorption *before* running the Chanin metric?" But the methodology computes graph statistics on the same SAE and tests against absorption rates computed on the same SAE. This is not true prediction---it is post-hoc correlation. To claim "prediction," the graph would need to be constructed on a *different* SAE (e.g., different layer, different model) and tested against absorption on the target SAE. Alternatively, a train/test split of features could be used.

---

## 3. Validity Threats Checklist

### 3.1 Data Leakage

- [x] **Test data in training set?** N/A---no training is performed. All experiments are inference-only on pretrained SAEs.
- [x] **Ground truth contamination?** The absorption detection uses the same prompts for both finding parent latents and detecting absorption. The `find_feature_latents` function uses target prompts to identify selective latents, and `detect_absorption` uses the same vocabulary words. This means the "parent latent" is selected to be maximally selective on the same data used to measure absorption. This is a form of selection bias---the parent is pre-selected to show the phenomenon.

**Severity: MEDIUM.** The parent latent selection procedure (max selectivity on target vs. other words) guarantees that the parent will show strong differential activation. This may inflate absorption rates for features that are genuinely selective but not necessarily "absorbed" in the Chanin sense.

### 3.2 Contamination

- [x] **Benchmark answers in pretraining?** The first-letter task (A-Z words) is not a standard benchmark. The vocabulary words ("apple", "book", etc.) are common English words almost certainly in GPT-2's pretraining data. However, the task is not about "answering" but about measuring activation patterns. Contamination is not a validity threat here.

### 3.3 Selection Bias

- [x] **Hyperparameters tuned on test set?** The absorption detection threshold (r > 0.3, p < 0.05 for child-parent correlation) was not tuned on the test set---it was fixed a priori. However, the *parent latent selection* (top-k selectivity) is data-dependent and may introduce bias.
- [x] **Feature selection?** Only 26 first-letter features are tested. These were chosen because they have clear hierarchical structure (A-words vs. specific A-words). This is a convenience sample, not a random sample of SAE features. Generalization to semantic features is untested.

**Severity: MEDIUM.** The feature set is small (n=26) and non-random. Statistical power is limited, and the sample may not represent the broader population of SAE features.

### 3.4 Overfitting to Evaluation

- [x] **Single benchmark?** The first-letter task is the only task used. The proposal acknowledges this limitation and suggests WordNet hierarchies as future work.
- [x] **Single model?** GPT-2 Small (124M) is the primary model. Pythia-70m was tested as cross-validation but is even smaller. The proposal includes Gemma-2-2B as optional cross-validation, but this was not executed due to access issues.

**Severity: HIGH.** Single model, single SAE family, single task type. External validity is severely limited.

### 3.5 Multiple Comparisons

- [x] **Correction applied?** The existing data applied Bonferroni and BH-FDR corrections across 12 tests. Zero hypotheses survive correction. The proposal does not explicitly address multiple comparisons for the new hypotheses (H6-H10), which will involve multiple k values (20, 50, 100), multiple layers (0, 4, 8, 10), and multiple graph statistics.

**Severity: MEDIUM.** If uncorrected, the new study risks false positives. With 4 layers x 3 k values x 2 metrics (precision, recall) = 24 tests for H6 alone, the family-wise error rate is substantial.

---

## 4. Ablation Gap Analysis

### 4.1 Existing Study (iter_001-008)

| Component | Ablation Tested? | How? | Gap |
|---|---|---|---|
| Absorption metric (differential correlation) | Partially | Random baseline subtraction (H1b) | No ablation of the r>0.3 threshold |
| Steering strength | No | Only strength=50 used for H1/H2 | EC50 tested but not as primary metric |
| Layer selection | Partially | Layers 4 and 8 tested | No systematic layer sweep |
| Feature type | No | Only first-letter features | No semantic features |

### 4.2 Proposed Inhibition Graph Study

| Component | Ablation Tested? | How? | Gap |
|---|---|---|---|
| Top-k neighbor selection (k=20,50,100) | Yes | Multiple k values tested | None |
| Signed vs. absolute correlation | No | Proposal uses signed G_ij | Does sign matter? Should absolute value be used? |
| Self-loops excluded | Yes | Identity graph baseline | None |
| Different graph construction methods | No | Only decoder correlation | No comparison to encoder correlation, activation correlation, or causal intervention |
| Homeostatic alpha | Yes | Sweep [0.1, 0.5, 1.0, 2.0, 5.0] | None |

**Critical Gap:** The proposal constructs the graph from decoder correlations (W_dec^T W_dec) but does not test whether encoder correlations (W_enc^T W_enc) or activation correlations would produce similar or better predictions. The LCA framework specifically uses the inhibition matrix G = W_dec^T W_dec, so this is theoretically justified, but an empirical comparison would strengthen the claim.

**Another Gap:** The proposal does not test whether the graph predicts *non-absorbed* correlated pairs. If decoder correlations simply predict "any correlated pair" (absorbed or not), then the graph is not specific to absorption.

---

## 5. Reproducibility Score

| Criterion | Score (1-5) | Evidence |
|---|---|---|
| Random seeds fixed | 5 | Seed=42 used throughout; deterministic where possible |
| All hyperparameters specified | 4 | Most hyperparameters documented (thresholds, k values, strengths). Hill fit uses fixed n=5 without justification. |
| Code available | 4 | Experiment scripts exist in `exp/` directory. Not packaged as a reproducible repository (no requirements.txt pinned versions, no Docker). |
| Hardware documented | 3 | "Single 24GB GPU" specified, but exact model not recorded in results. |
| Could reproduce within 10%? | 3 | Likely for graph construction (deterministic). Less likely for absorption detection (depends on SAE loading, which may vary by SAELens version). |

**Overall Reproducibility Score: 3.5/5**

**Issues:**
1. SAELens version not pinned. SAE loading behavior may vary across versions.
2. No Docker or environment snapshot.
3. The `find_feature_latents` function uses `np.random.shuffle` with fixed seed, but the shuffle is on `other_words` which depends on dictionary iteration order (Python 3.7+ preserves insertion order, but this is an implicit assumption).
4. Hardware specs not recorded in result files.

---

## 6. Top-3 Methodology Improvements

### Recommendation 1: Address the Multiple Comparisons Problem (Effort: Low, Impact: High)

**Problem:** The new study will run many tests (layers x k values x metrics) without explicit multiple comparison correction. H1b in the existing data was significant at p=0.028 uncorrected but did not survive Bonferroni or BH-FDR.

**Action:** Pre-register a primary analysis (e.g., H6 at layer 8, k=20) and treat all other tests as exploratory. Report both uncorrected and corrected p-values. Use a hierarchical testing structure: test H6 first; only if significant, proceed to H7-H10.

**Effort:** Low (documentation and analysis protocol change)
**Credibility Impact:** High (prevents false positive claims)

### Recommendation 2: Add a True Prediction Test for H8 (Effort: Medium, Impact: High)

**Problem:** H8 claims the graph can "predict" at-risk features "before running the Chanin metric," but the proposed methodology tests correlation on the same data.

**Action:** Use a leave-one-feature-out cross-validation: construct the graph using 25 features, predict absorption for the held-out feature, repeat 26 times. Alternatively, construct the graph on layer 4 and predict absorption on layer 8 (or vice versa). This would provide genuine predictive validity.

**Effort:** Medium (requires restructuring the analysis pipeline)
**Credibility Impact:** High (transforms a circular claim into a predictive one)

### Recommendation 3: Test Graph Specificity with Non-Absorbed Control (Effort: Low, Impact: Medium)

**Problem:** The graph may predict "any correlated pair," not specifically "absorbed pairs."

**Action:** Explicitly test whether graph edges are enriched for (a) absorbed pairs vs. (b) correlated but non-absorbed pairs. Define "correlated but non-absorbed" as child latents with r > 0.3 to parent but p > 0.05 (or using an alternative absorption metric). If the graph is equally predictive of both, the specificity claim fails.

**Effort:** Low (can be computed from existing data)
**Credibility Impact:** Medium (strengthens the mechanistic claim)

---

## 7. Additional Concerns

### 7.1 The "Precision@20 >= 0.10" Threshold

The proposal sets precision@20 >= 0.10 as the success threshold for H6, compared to ~0.004 chance. This is a 25x enrichment, which sounds impressive. However:
- With 24K latents and only ~5-10 absorption pairs per feature (estimated from existing data), the maximum possible recall@20 is very low.
- Precision@20 = 0.10 means 2 out of 20 neighbors are absorption pairs. This is a weak signal that may not justify the "graph predicts absorption" claim.
- The threshold should be justified by effect size, not just "above chance."

### 7.2 The LCA Structural Correspondence Claim

The proposal states "W_dec^T W_dec = G_LCA" is "exact, not metaphorical." This is mathematically correct for tied-weight SAEs. However:
- The gpt2-small-res-jb SAE uses **untied weights** (W_enc != W_dec^T). The structural correspondence is approximate, not exact.
- The proposal acknowledges this ("Even with untied weights, the structural correspondence holds") but does not quantify the approximation error.
- For the claim to be rigorous, the correlation between W_dec^T W_dec and W_enc^T W_enc should be reported.

### 7.3 Homeostatic Rebalancing: Causal Interpretation

The rebalancing formula z'_i = z_i + alpha * sum_j G_ij * z_j is described as "restoring parent firing." However:
- This is a post-hoc adjustment, not a causal intervention. It modifies activations after they are computed.
- The reconstruction error constraint (||a - W_dec * z'|| <= (1+epsilon) * ||a - W_dec * z||) ensures the adjustment is small, but it does not prove that the parent firing increase is due to "undoing inhibition" rather than simply boosting all activations.
- A causal test would use activation patching or ablation to show that removing the child latent specifically restores parent firing.

---

## 8. Summary Table

| Issue | Severity | Effort to Fix | Recommended Action |
|---|---|---|---|
| Random baseline precision overestimated (0.004 vs 0.00083) | Medium | Low | Correct the baseline calculation |
| H8 circularity (same data for graph and absorption) | High | Medium | Use LOOCV or cross-layer prediction |
| No multiple comparison correction planned | Medium | Low | Pre-register primary analysis; report corrected p-values |
| Graph specificity untested | Medium | Low | Add non-absorbed correlated pair control |
| Single model, single task, single SAE family | High | High | Execute Gemma cross-validation; add semantic features |
| Tied vs. untied weights not quantified | Medium | Low | Report correlation between W_dec^T W_dec and W_enc^T W_enc |
| Hill fit with fixed n=5 | Low | Low | Test sensitivity to n; report fit diagnostics |
| Parent latent selection on same data | Medium | Medium | Use held-out prompts for parent selection |
| Small sample (n=26 features) | Medium | High | Cannot fix without new data; acknowledge limitation |
| Rebalancing causal interpretation weak | Medium | Medium | Add activation patching control |

---

## 9. Overall Assessment

The Local Inhibition Graph proposal is **methodologically sound in principle** but has several **implementation gaps** that threaten internal validity:

1. **The strongest concern is circularity in H8** (graph "predicts" absorption using the same data). This must be addressed with cross-validation or cross-layer testing.
2. **The multiple comparisons problem** was already fatal to H1b in the existing data. The new study must pre-register a primary analysis.
3. **External validity is severely limited** by single-model, single-task design. The optional Gemma cross-validation should be treated as mandatory, not optional.
4. **The baseline precision calculation is incorrect** (0.004 vs 0.00083), which inflates the apparent effect size.

**Verdict:** The proposal is worth pursuing, but with mandatory methodological fixes before any claims are made. The theoretical contribution (LCA-SAE connection) does not depend on empirical validation and can be stated independently. The empirical claims (graph predicts absorption, explains precision-recall asymmetry) require stricter controls.
