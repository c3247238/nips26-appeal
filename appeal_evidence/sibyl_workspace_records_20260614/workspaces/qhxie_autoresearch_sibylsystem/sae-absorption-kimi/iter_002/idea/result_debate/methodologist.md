# Methodologist Audit: Construct Validity of the SAEBench Feature Absorption Metric

## Executive Summary

This audit scrutinizes the experimental methodology of Iteration 2, which tests whether the SAEBench first-letter absorption metric generalizes to semantic hierarchies. The study design is conceptually sound, but several methodological issues threaten the validity of the conclusions. Most critically: (1) the Random-SAE control implementation is inconsistent between the official first-letter eval and the custom probe pipelines, (2) the probe AUROC ceiling effect (all AUROCs = 1.0) suggests the classification task is trivially easy, (3) the small-n correlation (n=7) with wide bootstrap CIs renders H1 inconclusive by design, and (4) H2 is rejected in the wrong direction (semantic < non-hierarchy), which contradicts the core claim of hierarchy specificity.

---

## 1. Baseline Fairness Audit

### 1.1 Official First-Letter Eval vs. Custom Probe Pipelines

| Aspect | First-Letter (Official) | Semantic-Hierarchy (Custom) | Non-Hierarchy (Custom) | Issue |
|--------|------------------------|----------------------------|------------------------|-------|
| **Probe type** | Logistic regression | Logistic regression | Logistic regression | Matched |
| **Probe training** | 200 Adam epochs, lr=0.05 | `max_iter=500`, `lbfgs` solver | `max_iter=500`, `lbfgs` solver | **MISMATCH** |
| **Input pooling** | Mean-pooled over non-padding | Mean-pooled over non-padding | Mean-pooled over non-padding | Matched |
| **k-sparse** | k=10 (default) | k=10 | k=10 | Matched |
| **Absorption formula** | `max(0, (resid-sae)/resid, (resid-ksparse)/resid)` | Same | Same | Matched |
| **Dataset** | Natural text corpus | Synthetic templates (100 samples/concept) | Synthetic templates (100 samples/concept) | **MISMATCH** |
| **Task type** | Multi-class (26 letters) | Multi-class (2-3 children) | Binary (2 words) | **ASYMMETRY** |

**Finding:** The probe training protocol differs between official and custom evaluations. The official SAEBench eval uses 200 Adam epochs with lr=0.05 (per `AbsorptionEvalConfig` defaults), while the custom pipelines use scikit-learn's `LogisticRegression(max_iter=500, solver="lbfgs")` which uses a different optimization algorithm (quasi-Newton vs. Adam) and convergence criteria. This is a moderate asymmetry -- both are logistic regression, but the optimization landscapes differ.

**More importantly:** The official eval uses natural text data (OpenWebText-derived), while the custom pipelines use synthetic template sentences. This is a significant confound: the first-letter task operates on natural language distributions, while the semantic tasks operate on highly structured template sentences. Any difference in absorption could be attributed to dataset characteristics rather than task semantics.

### 1.2 Random-SAE Control Inconsistency (CRITICAL)

The Random-SAE control is implemented differently across the three conditions:

- **First-letter eval** (`firstletter_pythia.py`, line 118-128): Permutes decoder directions only (`W_dec[perm]`, `W_enc[:, perm]`, `b_enc[perm]`). This preserves the latent structure but shuffles feature identities.
- **Semantic-hierarchy eval** (`semantic_hierarchy_pythia.py`, line 132-182): Completely randomizes `W_enc`, `b_enc`, AND `W_dec` with Gaussian initialization matched to original statistics. This destroys all learned structure.
- **Non-hierarchy eval** (`nonhierarchy_control_pythia.py`, line 132-182): Same complete randomization as semantic-hierarchy.

**Impact:** The first-letter Random-SAE still uses the original encoder/decoder geometry (just permuted), which may retain some absorption-like structure. The custom pipelines use fully random weights, which should produce near-zero absorption. The observed Random-SAE scores confirm this:
- First-letter: 0.030 (near-zero, as expected)
- Semantic-hierarchy: 0.175 (non-zero -- concerning)
- Non-hierarchy: 0.233 (non-zero -- concerning)

The non-zero Random-SAE scores on custom tasks (0.175 and 0.233) suggest the absorption metric captures artifacts even on random representations. This is flagged in Section 3.

### 1.3 SAE Selection Fairness

The 7 SAE families are all from the same source repository (`adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108`), same layer (8), same trainer (`trainer_0`). This is fair -- all architectures were trained with the same hyperparameter budget on the same data. However, only one trainer per family is evaluated. If `trainer_0` happens to be an outlier (e.g., converged to a local minimum), the results may not generalize to other training runs.

---

## 2. Metric-Claim Alignment

### 2.1 Claim-to-Metric Mapping

| Claim | Metric | Alignment | Gap |
|-------|--------|-----------|-----|
| "First-letter absorption predicts semantic-hierarchy absorption" | Pearson r (first-letter vs. semantic) | Moderate | Correlation does not imply construct validity; need causal or convergent validity evidence |
| "Metric is specific to hierarchical features" | Paired t-test (semantic vs. non-hierarchy) | Poor | **REJECTED in wrong direction** -- non-hierarchy scores are HIGHER |
| "Metric is robust to threshold changes" | Pearson r at tau_fs = 0.01, 0.03, 0.05 | Moderate | Only tests first-letter robustness, not semantic-hierarchy robustness |

### 2.2 The Hierarchy Specificity Problem (CRITICAL)

H2 claims semantic-hierarchy absorption should be higher than non-hierarchy control absorption. The results show the **opposite**:

| Architecture | Semantic-Hierarchy | Non-Hierarchy | Difference |
|-------------|-------------------:|--------------:|-----------:|
| BatchTopK | 0.359 | 0.398 | -0.039 |
| GatedSAE | 0.188 | 0.379 | -0.191 |
| JumpRelu | 0.230 | 0.348 | -0.118 |
| Matryoshka | 0.203 | 0.333 | -0.130 |
| PAnneal | 0.064 | 0.131 | -0.067 |
| Standard | 0.352 | 0.416 | -0.064 |
| TopK | 0.250 | 0.311 | -0.061 |
| **Mean** | **0.235** | **0.331** | **-0.096** |

Paired t-test: t = -4.748, p = 0.003. The metric is **significantly MORE sensitive to non-hierarchical correlated features than to semantic hierarchies**.

**Interpretation crisis:** If the absorption metric is supposed to detect hierarchical feature absorption, it should show higher scores on hierarchies than on arbitrary correlations. The fact that it shows the opposite undermines the central claim. Either:
1. The semantic hierarchies are not truly hierarchical in the model's representation, OR
2. The non-hierarchy pairs are more "absorbable" due to stronger correlation in the training data, OR
3. The absorption metric itself is not hierarchy-specific.

The methodology does not distinguish between these explanations.

### 2.3 Measurement Gap: What Does Absorption Actually Measure?

The absorption score is defined as:
```
absorption = max(0, (resid_acc - sae_acc)/resid_acc, (resid_acc - k_sparse_acc)/resid_acc)
```

This measures the **relative accuracy drop** when moving from residual-stream activations to SAE latents. However:
- It conflates "feature is missing" with "feature is represented differently"
- It does not distinguish between linear separability (probe accuracy) and actual causal necessity (ablation)
- The k-sparse component assumes top-k features should suffice, which may not hold for complex semantic concepts

The proposal acknowledges this limitation ("conservative underestimate"), but the methodology does not address it.

---

## 3. Validity Threats Checklist

### 3.1 Data Leakage
- [x] **NO EVIDENCE OF LEAKAGE.** The semantic hierarchies use synthetic template sentences not present in the pretraining corpus. The non-hierarchy pairs also use synthetic templates. However, the words themselves (e.g., "building", "house") are common and appear in pretraining data. This is expected and not a leakage issue per se.

### 3.2 Contamination
- [ ] **POTENTIAL CONTAMINATION.** The first-letter eval uses natural text data (likely from the model's pretraining distribution). The words used in semantic hierarchies (e.g., "animal", "dog") are extremely common in pretraining. If the model has strong prior associations, the ground-truth probe may be learning these priors rather than the specific hierarchical relationship.

### 3.3 Selection Bias / Test-Set Contamination
- [x] **NO DIRECT TEST-SET TUNING.** Hyperparameters were fixed a priori (k=10, tau_fs=0.03, 200 probe epochs). However:
  - The 10 hierarchies were selected from WordNet based on single-token constraints AND pilot results (pilot used 3 hierarchies, full used 10). The pilot may have influenced hierarchy selection.
  - The probe AUROC threshold of 0.7 was set a priori, but all hierarchies achieved AUROC = 1.0, suggesting the threshold was never actually tested.

### 3.4 Overfitting to Evaluation
- [ ] **CONCERN: TRIVIAL PROBE TASK.** Every single hierarchy and pair achieves probe AUROC = 1.0 on residual-stream activations. This means the ground-truth probe achieves perfect classification on the training data. With only 100 samples per concept and logistic regression, perfect AUROC suggests:
  1. The concepts are linearly perfectly separable in residual space (unlikely for semantically similar concepts), OR
  2. The synthetic template sentences create spurious correlations that make classification trivial (e.g., template structure leaks label information).

**Evidence:** All 10 hierarchies show `resid_auroc = 1.0` for all 8 SAEs. All 10 pairs show `resid_auroc = 1.0` (or 0.9975). This is a ceiling effect that renders the absorption formula's numerator (`resid_acc - sae_acc`) maximally sensitive to any drop in SAE accuracy, potentially inflating absorption scores artificially.

### 3.5 Floor/Ceiling Effects
- **Ceiling effect on resid_acc:** All resid_acc values are 0.95-1.0. With such high baseline accuracy, even small drops in SAE accuracy produce large absorption scores.
- **Example:** If resid_acc = 1.0 and sae_acc = 0.95, absorption = 0.05. But if resid_acc = 0.7 and sae_acc = 0.65, absorption = 0.071. The metric is not normalized for baseline difficulty.
- **The proposal proposed standardizing by probe AUROC** ("Standardize absorption scores by probe AUROC before correlating"), but this was NOT implemented.

---

## 4. Ablation Gap Analysis

| Proposed Component | Ablation Present? | Notes |
|-------------------|-------------------|-------|
| **tau_fs threshold** | Yes | Tested at 0.01, 0.03, 0.05. Correlation stable (~0.46-0.47). |
| **k-sparse component** | No | The absorption formula uses k-sparse latents, but no ablation tests absorption without the k-sparse term. |
| **Synthetic vs. natural data** | No | All custom tasks use synthetic templates. No ablation using natural corpus data. |
| **Multi-class vs. binary probe** | Partial | Hierarchies use multi-class (2-3 children), pairs use binary. But this is confounded with hierarchy vs. non-hierarchy. |
| **Mean-pooling vs. token-specific** | No | All probes use mean-pooled activations. No ablation using token-position-specific activations. |
| **Template diversity** | No | Only 19 templates for hierarchies, 20 for pairs. No ablation with more/less diverse templates. |
| **Random-SAE type** | No | Different randomization between first-letter (permutation) and custom (full randomization). |

**Missing ablation that would be most informative:**
- **Standardized absorption scores:** Normalize by probe AUROC to control for baseline difficulty. The proposal explicitly mentions this but it was not implemented.
- **Natural vs. synthetic data:** Run the semantic-hierarchy probe on natural corpus sentences containing the target words, not synthetic templates.
- **k-sparse only vs. full SAE:** Test whether the k-sparse component drives the correlation.

---

## 5. Reproducibility Score: 3/5

| Criterion | Status | Notes |
|-----------|--------|-------|
| Random seeds fixed | Partial | `SEED = 42` for numpy/torch. But sklearn's `LogisticRegression` has no explicit random state, and `lbfgs` is deterministic. SAEBench eval seed is fixed. |
| All hyperparameters specified | Yes | k=10, tau_fs values, samples_per_concept=100, layer=8, etc. all documented. |
| Code available | Yes | All scripts saved in `iter_002/exp/scripts/`. |
| Data available | Partial | WordNet hierarchies are listed, but the synthetic template generation is ad-hoc. No persistent dataset artifact. |
| Hardware documented | Minimal | "local" GPU model not specified. Pythia-160M fits on most GPUs. |
| Could reproduce within 10% | Uncertain | The template sentences are hard-coded in the scripts, so exact replication is possible. But the Random-SAE implementation differs across scripts, which could affect reproducibility of control scores. |

**Issues:**
1. No `requirements.txt` or environment snapshot for the exact package versions used.
2. GPU model not recorded (only "local" in gpu_progress).
3. The SAEBench version/commit is not pinned.

---

## 6. Top-3 Recommendations

### Recommendation 1: Fix the Random-SAE Control and Re-evaluate (High Impact, Low Effort)

**Problem:** The Random-SAE is implemented inconsistently across conditions.

**Action:** Use the SAME randomization strategy for all three conditions. The full randomization (Gaussian weights matched to statistics) used in the custom pipelines is the correct approach -- the permutation-only approach in first-letter eval does not actually destroy learned structure. Re-run the first-letter eval with fully random weights.

**Expected outcome:** If the first-letter Random-SAE score increases significantly, the current low score (0.030) may be an artifact of the permutation strategy. This would change the correlation baseline.

### Recommendation 2: Address the Trivial Probe Task (High Impact, Medium Effort)

**Problem:** All probe AUROCs = 1.0 indicates the classification task is trivial, likely due to template structure leaking label information.

**Action:**
1. Increase template diversity (100+ unique templates per hierarchy).
2. Add template-mixing controls where the same templates are used with shuffled word assignments.
3. Report absorption on a subset of hierarchies with lower resid_auroc (<0.95) to test whether the correlation holds when the baseline is not at ceiling.

**Alternative:** Use natural corpus sentences instead of synthetic templates. Extract sentences from a corpus that contain the target words, then filter for hierarchical context.

**Expected outcome:** If absorption scores decrease significantly with more natural/diverse data, the current scores may be inflated by template artifacts.

### Recommendation 3: Implement Standardized Absorption Scores (Medium Impact, Low Effort)

**Problem:** The proposal explicitly recommended standardizing absorption scores by probe AUROC before correlating, but this was not implemented. With all resid_aurocs = 1.0, this is less urgent, but if Recommendation 2 produces variable AUROCs, standardization becomes critical.

**Action:** Compute `standardized_absorption = absorption / (1 - baseline_chance)` or `absorption * (1 - resid_auroc)` to penalize hierarchies with trivial classification. Re-compute correlations using standardized scores.

**Expected outcome:** If the correlation strengthens after standardization, the construct validity claim gains support. If it weakens, the metric may be confounded by task difficulty.

---

## Appendix: Data Quality Red Flags

1. **All AUROCs = 1.0:** Every hierarchy, every pair, every SAE achieves perfect or near-perfect AUROC on residual-stream probes. This is statistically implausible for 80 distinct classification tasks (10 hierarchies x 8 SAEs + 10 pairs x 8 SAEs) unless the task is trivial.

2. **GPT-2 replication shows near-zero absorption:** On GPT-2 small, both Standard and TopK SAEs show mean hierarchy absorption of 0.0 and 0.003 respectively. This is dramatically lower than Pythia-160M (0.064-0.352). The replication does not replicate -- it suggests the effect is model-specific or the GPT-2 SAEs are from a different training regime.

3. **Semantic < Non-hierarchy for ALL architectures:** The consistent direction of the H2 violation (semantic always lower than non-hierarchy) across all 7 real SAEs is not random noise. It is a systematic pattern that requires explanation.

4. **Bootstrap CIs include both strong positive and negative correlations:** The H1 bootstrap CI [-0.39, 0.98] is so wide that it is essentially uninformative. With n=7, this is expected, but it means the study is underpowered for its primary hypothesis.
