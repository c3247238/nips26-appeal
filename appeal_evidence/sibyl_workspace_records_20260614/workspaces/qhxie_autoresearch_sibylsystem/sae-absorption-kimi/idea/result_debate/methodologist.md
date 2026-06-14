# Methodologist Analysis: Component-Isolated SAE Absorption Study

## Executive Summary

The full experiment (7 variants x 5 replicates each, seeds 42, 123, 456, 789, 1011) was executed on synthetic hierarchical data (1024 features, 256 hidden dim, 32 root trees, depth 3, branching factor 4) with 2M training samples per replicate. The methodology is **generally sound** for a pilot-scale component-isolation study, but several **credibility-threatening issues** must be addressed before the paper can support its central claim that "absorption is a sparsity phenomenon."

---

## 1. Baseline Fairness Audit

### Component-Isolated Design: Correctly Implemented

The one-component-at-a-time design is the study's methodological strength. Each variant modifies exactly one architectural component from the Baseline ReLU + L1 SAE:

| Variant | Modification | Fairness Assessment |
|---------|-------------|---------------------|
| Baseline | Standard ReLU + L1 (lambda=5e-3) | Fair -- uses SAELens standard runner |
| TopK | Replaces L1 with TopK sparsity (k=50) | Fair -- same runner, different sparsity mechanism |
| MultiScale | Adds nested dictionaries (3 widths) | Fair -- MatryoshkaBatchTopK runner |
| Orthogonality | Adds decoder orthogonality penalty (lambda=1e-3) | **UNFAIR -- custom training loop** |
| Gating | Replaces ReLU with gated detection/magnitude | Fair -- GatedTrainingSAE runner |
| Full Matryoshka | TopK + MultiScale combined | Fair -- same as MultiScale runner |
| Random Control | Untrained random decoder | Fair -- no training, serves as sanity check |

### Critical Asymmetry: Orthogonality Uses a Custom Training Loop

The Orthogonality variant is trained with a **hand-written PyTorch loop** (`OrthogonalitySAE` class, lines 542-746 in `run_full.py`), while all other trained variants use SAELens's `SyntheticSAERunner` with its built-in training logic, learning rate scheduling, and evaluation pipeline.

**Specific differences that could confound results:**

1. **No L1 warm-up**: The custom loop applies a fixed `5e-3 * sparsity_loss` from step 0, while Baseline/TopK/Gating use `l1_warm_up_steps=2000` (gradual L1 ramp).
2. **No learning rate scheduling**: The custom loop uses a fixed Adam LR of 1e-3; SAELens runners may have implicit scheduling.
3. **Different reconstruction-sparsity trade-off**: The custom loop uses `recon_loss + 5e-3 * sparsity_loss + 1e-3 * ortho_loss`, while the Baseline uses SAELens's internal loss balancing. The orthogonality penalty may dominate the optimization landscape, pushing the model toward a different basin.
4. **Decoder renormalization after every step**: The custom loop renormalizes decoder rows after each gradient step (`sae.W_dec.data = sae.W_dec.data / (sae.W_dec.data.norm(dim=1, keepdim=True) + 1e-6)`). No other variant has this constraint, which could artificially improve reconstruction MSE (observed: 3.17e-05 vs. Baseline 0.0104 -- a 300x difference) by preventing decoder magnitude drift.

**Impact on the orthogonality null result (d = 0.14):** The custom training loop's decoder renormalization and different loss landscape could produce an L0 of ~550 (intermediate between Baseline ~964 and TopK 50) through a different optimization trajectory. If the L0-absorption relationship is causal, this intermediate L0 *should* produce intermediate absorption -- but the claim is that orthogonality "has negligible effect." The null result may reflect the L0 level rather than the orthogonality penalty itself, but the custom loop makes this interpretation ambiguous.

**Recommendation**: Re-run Orthogonality using SAELens's standard runner with a custom loss hook, or at minimum document the training loop differences explicitly and acknowledge this as a limitation.

### Missing Baseline: L0-Matched L1 SAE (Critical Control Not Executed)

The methodology document (`plan/methodology.md`, line 28-30) explicitly states:

> "To disentangle sparsity from architecture, we train L1 SAEs with tuned lambda to achieve L0 = 50 (matching TopK) and L0 = 550 (matching Orthogonality)."

**This experiment was NOT executed.** The `task_plan.json` lists task `e2_l0_matched` as a dependency of the analysis, but no `l0_matched_results.json` exists in `exp/results/full/`. The statistical analysis (`statistical_analysis.json`) computes L0-absorption correlation across variant *means* (r = 0.865) but does NOT test the causal claim: "at matched L0, different architectures show comparable absorption."

**This is the single most important missing experiment.** Without it, the central claim that "absorption is a sparsity phenomenon" rests on correlational evidence (L0 correlates with absorption across variants) rather than causal evidence (matching L0 eliminates architectural differences).

---

## 2. Metric-Claim Alignment

### Primary Metric: Ground-Truth Absorption Rate

**Alignment: STRONG.** The absorption rate is computed using known parent-child relationships from the synthetic hierarchy. This is a genuine methodological advance over probe-based absorption metrics on real LLMs, which the proposal correctly identifies as confounded.

**Computation details (from `run_full.py`, lines 100-178):**
- Latent-to-feature assignment: cosine similarity between decoder rows and ground-truth feature vectors
- Absorption event: for each parent firing, check if ANY child-assigned latent activates above threshold 0.05
- Rate: absorption_count / parent_fire_count

**Potential issue**: The threshold of 0.05 is arbitrary and not justified. Sensitivity analysis at thresholds {0.01, 0.05, 0.1, 0.2} would strengthen credibility.

### Secondary Metrics: Mixed Alignment

| Metric | Claim It Supports | Alignment | Issue |
|--------|------------------|-----------|-------|
| Reconstruction MSE | "No reconstruction-absorption trade-off" | MODERATE | Orthogonality MSE is 300x lower due to custom loop; not comparable |
| L0 sparsity | "Sparsity drives absorption" | STRONG | Directly measures the hypothesized mediator |
| Dead latent rate | "TopK has dead latent crisis" | STRONG | TopK: 1672/2048 dead; MultiScale: 1155/2048; Baseline: 0 dead |
| Feature recovery MCC | "Feature recovery quality" | WEAK | MCC ~0.21-0.22 across ALL variants including Random; metric is degenerate (correctly noted in pilot) |
| Hedging score | "Absorption-hedging trade-off" | WEAK | Hedging ~0.23-0.24 constant across all variants; no discrimination |
| Classification accuracy | "Feature classification quality" | MODERATE | TopK/MultiScale/Matryoshka ~97.4% vs. Baseline/Gating ~52%; large gap not explained |

### Classification Accuracy Anomaly

A striking unexplained result: TopK, MultiScale, and Full Matryoshka achieve ~97.4% classification accuracy, while Baseline and Gating achieve ~52-53%. This 45-point gap is not discussed in any summary. The classification task (assigning latents to ground-truth features via Hungarian matching) appears to be trivial for low-absorption variants and near-random for high-absorption variants. This may be a ceiling/floor artifact rather than meaningful signal.

---

## 3. Validity Threats Checklist

### [ ] Data Leakage
**Status: NOT APPLICABLE.** Synthetic data with known hierarchy; no train/test split in the conventional sense. The evaluation uses fresh samples from the same generative process, which is appropriate for measuring population-level properties.

### [ ] Contamination
**Status: NOT APPLICABLE.** No pretraining data; features are generated synthetically.

### [ ] Selection Bias / Hyperparameter Tuning on Test Set
**Status: MODERATE THREAT.** The L1 coefficient (5e-3) and TopK k (50) were "pilot-validated" on the same synthetic data structure (though with different seeds). There is no held-out validation set for hyperparameter selection. However, the pilot and full experiments use different seeds, which provides some independence.

**More concerning**: The orthogonality lambda (1e-3) is described as "standard" but no justification is given. If this value was chosen to produce a specific L0 (~550), it constitutes indirect tuning.

### [ ] Overfitting to Evaluation
**Status: LOW THREAT for absorption, MODERATE for MCC.** Absorption is measured on fresh samples with a deterministic protocol. MCC uses Hungarian matching on the full decoder, which can overfit to the specific random seed of feature generation (though results are consistent across seeds).

### [ ] Synthetic-to-Real Gap
**Status: HIGH THREAT.** This is explicitly acknowledged in the proposal (Risk Assessment: "Synthetic data doesn't match LLM feature structure -- Severity: High"). The paper's central claim is about SAE architectures in general, but all evidence comes from a synthetic benchmark with:
- Exactly 1024 features (vs. LLMs with ~50k-1M neurons)
- Perfect hierarchical structure (vs. unknown LLM feature structure)
- No superposition (features are orthogonal by construction in the synthetic model)

The proposal promises "Phase 2 real-LLM validation" but this has not been executed.

---

## 4. Ablation Gap Analysis

### Proposed Components and Their Ablation Status

| Component | Ablation Experiment? | Status |
|-----------|---------------------|--------|
| TopK sparsity (k=50) | TopK vs. Baseline | COMPLETE (d = 4.93) |
| MultiScale dictionaries | MultiScale vs. Baseline | COMPLETE (d = 4.81) |
| Decoder orthogonality | Orthogonality vs. Baseline | COMPLETE but confounded by custom loop (d = 0.13) |
| Gating mechanism | Gating vs. Baseline | COMPLETE (d = -0.17, no effect) |
| Full Matryoshka (combined) | Matryoshka vs. Baseline | COMPLETE (d = 4.31) |
| **L0-matched L1 baseline** | **L1 tuned to L0=50 and L0=550** | **MISSING -- CRITICAL** |
| **k-sweep (dose-response)** | **TopK with k in {10, 25, 50, 100, 200, 500}** | **MISSING** |
| **Dead latent correction** | **Absorption computed on active latents only** | **MISSING** |

### Component Interaction Analysis

The `component_interaction.json` claims "antagonistic interaction" for Full Matryoshka, but the computation is mathematically flawed:

```
additive_expected = baseline - topk_reduction - multiscale_reduction
                = 0.252 - 0.197 - 0.197 = -0.142
```

This subtracts both reductions from baseline, implying absorption could go negative. The correct additive model should be:

```
additive_expected = baseline * (1 - reduction_topk) * (1 - reduction_multiscale)
                = 0.252 * (1 - 0.78) * (1 - 0.78) = 0.012
```

Or, if effects are on log-odds:
```
logit(baseline) = log(0.252/0.748) = -1.08
expected logit = -1.08 - 1.47 - 1.50 = -4.05
expected probability = 0.017
```

The observed Full Matryoshka absorption (0.066) is actually **higher** than either TopK (0.056) or MultiScale (0.055) alone, suggesting **redundancy** (the components target the same mechanism -- sparsity) rather than antagonism. Both TopK and MultiScale enforce L0=50; combining them cannot reduce sparsity further.

---

## 5. Reproducibility Score: 3/5

| Criterion | Status | Notes |
|-----------|--------|-------|
| Random seeds fixed | YES | 5 seeds: 42, 123, 456, 789, 1011 |
| All hyperparameters specified | PARTIAL | L1 coeff, TopK k, ortho lambda specified; but no LR schedule details, no warm-up specifics for custom loop |
| Code available | YES | `exp/run_full.py` is well-structured and readable |
| Data generation specified | YES | HierarchyConfig with 32 roots, bf=4, depth=3 |
| Hardware documented | PARTIAL | "RTX PRO 6000 Blackwell" mentioned in summary; no GPU memory requirements |
| SAELens version pinned | CLAIMED but NOT VERIFIED | `methodology.md` says "SAELens version pinned" but no version number found in results |
| PyTorch deterministic mode | NOT CONFIRMED | No `torch.use_deterministic_algorithms()` call visible |
| Could reproduce within 10% | PROBABLY | For Baseline/TopK/MultiScale/Gating (standard runners); UNLIKELY for Orthogonality (custom loop) |

**Score justification**: The standard variants are highly reproducible. The orthogonality variant's custom training loop introduces unquantified variance. The missing L0-matched and k-sweep experiments reduce the reproducibility of the central claim.

---

## 6. Top-3 Recommendations

### Recommendation 1: Execute the L0-Matched Control (Effort: High, Credibility Impact: CRITICAL)

**What**: Train Baseline L1 SAEs with tuned lambda to achieve L0 = 50 (matching TopK) and L0 = 550 (matching Orthogonality). Use 3 replicates each.

**Why**: This is the only experiment that can causally test whether absorption is a sparsity phenomenon or an architectural one. If L0-matched Baseline achieves absorption comparable to TopK at L0=50, the central claim is proven. If not, TopK has an additional architectural benefit beyond sparsity.

**Expected outcome**: Given the near-perfect L0-absorption correlation (r = 0.865 across variant means), the L0-matched baseline should match TopK's absorption. This would transform the paper from "correlational observation" to "causal demonstration."

**Time estimate**: ~30 minutes (3 replicates x 2 L0 targets x ~5 min each).

### Recommendation 2: Fix the Orthogonality Training Loop or Add a Fair Comparison (Effort: Medium, Credibility Impact: HIGH)

**What**: Either (a) re-implement orthogonality as a SAELens runner plugin with identical training dynamics, or (b) add an explicit comparison showing that the custom loop's decoder renormalization does not artificially suppress absorption.

**Why**: The current orthogonality result (d = 0.14, 2.7% reduction) is the paper's most provocative claim -- it directly contradicts OrtSAE's reported 65% reduction. But reviewers will immediately question whether the null result is due to the orthogonality penalty being ineffective, or due to the custom training loop diverging from standard SAE optimization. The 300x lower MSE (3.17e-05 vs. 0.0104) is a red flag that the orthogonality SAE is solving a different optimization problem.

**Alternative**: If re-running is infeasible, explicitly acknowledge the custom loop limitation and frame the orthogonality result as "preliminary" rather than "definitive."

### Recommendation 3: Run the TopK Dose-Response Sweep (Effort: Low-Medium, Credibility Impact: HIGH)

**What**: Train TopK SAEs with k in {10, 25, 50, 100, 200, 500} (one replicate each, ~30 min total).

**Why**: The paper claims absorption is "primarily a sparsity-level phenomenon" with "L0-absorption correlation r ~ -0.99." But the current correlation is computed across only 7 variant means (r = 0.865, p = 0.012). A dose-response curve within a single architecture (TopK) would:
1. Provide much stronger evidence for the sparsity-absorption causal link
2. Characterize the functional form (linear? logistic? threshold?)
3. Enable predictions for real-LLM SAEs with known L0 values

**Expected outcome**: Absorption should increase monotonically with k. If the relationship is smooth and predictable, it strongly supports the sparsity-mediation hypothesis. If there are discontinuities or non-monotonic regions, it suggests additional architectural effects.

---

## 7. Additional Concerns

### 7.1 Effect Size Inflation in Proposal

The proposal claims Cohen's d = 5.51 for TopK vs. Baseline, but the statistical analysis reports d = 4.93. The discrepancy appears to come from using different standard deviation estimates (pooled vs. baseline-only). Both are "extremely large" by conventional standards, but consistency in reporting is important.

### 7.2 Missing Holm-Bonferroni Correction

The methodology pre-registers Holm-Bonferroni correction for multiple comparisons, but the `statistical_analysis.json` reports raw p-values without correction. With 6 pairwise comparisons against baseline, the family-wise error rate at alpha=0.05 is 1 - (0.95)^6 = 26.5%. The corrected threshold for the largest p-value would be 0.05/6 = 0.0083. All significant comparisons (TopK, MultiScale, Matryoshka, Random) remain significant at this threshold, but the correction should be applied and reported.

### 7.3 Sample Size Justification

The proposal mentions "5 replicates for variance estimation" but does not justify why 5 is sufficient. With Cohen's d ~ 5, even n=2 per group would provide >99% power. For the orthogonality null effect (d = 0.14), n=5 provides only ~6% power to detect a small effect -- essentially, the experiment is underpowered to detect a true small effect of orthogonality. This is acceptable for ruling out large effects (the paper's claim), but the power limitation should be acknowledged.

### 7.4 Explained Variance Metric is Broken

The explained variance values are consistently negative for all trained variants (e.g., TopK: -0.38 +/- 0.05; Baseline: -0.88 +/- 0.16), while the Random control shows -116. The pilot summary correctly identifies this as "numerical instability" and recommends using MSE instead. The full results should drop explained variance entirely or fix the computation.

---

## 8. Summary Table: Methodology Strengths and Weaknesses

| Aspect | Rating | Evidence |
|--------|--------|----------|
| Component-isolated design | STRONG | One component varied at a time |
| Ground-truth absorption metric | STRONG | No probe ambiguity |
| Multiple replicates | ADEQUATE | 5 seeds, sufficient for large effects |
| Random control | STRONG | Validates metric discrimination (0.534 vs. 0.252 baseline) |
| Baseline fairness | WEAK | Orthogonality uses custom loop |
| L0-matched control | MISSING | Critical for causal claim |
| Dose-response curve | MISSING | Would strengthen sparsity claim |
| Statistical corrections | PARTIAL | No Holm-Bonferroni applied |
| Synthetic-to-real bridge | MISSING | Phase 2 not executed |
| Reproducibility | ADEQUATE | Code clear, but SAELens version not specified |

---

## 9. Verdict

The experiment provides **strong correlational evidence** that L0 sparsity and absorption are related, and **strong evidence** that TopK and MultiScale reduce absorption dramatically. However, the central causal claim -- that "absorption is a sparsity phenomenon, not an architectural one" -- **cannot be accepted without the L0-matched control experiment.**

The orthogonality null result is **methodologically confounded** by the custom training loop and should either be re-run with fair comparison or downgraded from a "direct contradiction of OrtSAE" to a "preliminary finding requiring validation."

**Recommendation to authors**: Execute the L0-matched control and the k-sweep before submitting. These two experiments (combined ~1 hour of GPU time) would transform the paper from an interesting observational study into a definitive causal analysis.
