# Experiment Critique: Iteration 009

## Summary

The experiments represent a significant methodological improvement over prior iterations: component-isolated design, ground-truth synthetic data, 5 replicates per variant, and direct absorption measurement without probes. However, critical data integrity issues identified in prior rounds remain UNADDRESSED. The most severe is a confirmed data-copying bug where 4 of 5 Matryoshka replicates are byte-identical to MultiScale replicates. Additionally, the paper claims experiments used SynthSAEBench-16k (16,384 features) but all results files show num_features=1024. Most alarmingly, 5 of 6 trained variants show negative explained_variance, indicating severe undertraining. The promised L0-matched ablation and k-sweep experiments were never run. NEW: dose-response variance decomposition reveals 75% of absorption variance is seed-related, not lambda-related, undermining the causal claim. These issues are severe enough that the paper cannot be submitted without major revision.

---

## Critical Issues

### E1: Data Copying Bug -- CONFIRMED AND UNADDRESSED

**Evidence**: Direct comparison of `full_multiscale_results.json` and `full_matryoshka_results.json`:

| Seed | MultiScale abs_rate | Matryoshka abs_rate | IDENTICAL? |
|------|--------------------|---------------------|------------|
| 42 | 0.044035 | 0.101081 | NO |
| 123 | 0.044197 | 0.044197 | **YES** |
| 456 | 0.029496 | 0.029496 | **YES** |
| 789 | 0.056854 | 0.056854 | **YES** |
| 1011 | 0.099594 | 0.099594 | **YES** |

Seeds 123, 456, 789, and 1011 have byte-identical values for: absorption_rate, feature_recovery_mcc, reconstruction_mse, explained_variance, l0_sparsity, dead_latents, shrinkage, uniqueness, hedging_score, and all classification metrics.

**Impact**: The Matryoshka mean (0.066 +/- 0.029) is computed from one genuine replicate and four copied MultiScale replicates. The reported "antagonistic interaction" is entirely artifactual.

**Fix**: Re-run Full Matryoshka with all 5 seeds independently. Withdraw the antagonism claim until genuine data is available. Add automated cross-file duplicate detection (MD5 hash comparison) to the pipeline.

### E2: Feature Count Misrepresentation -- 1k vs 16k, CONFIRMED

**Evidence**:
- `full_baseline_results.json`: `"num_features": 1024`
- `full_baseline_results.json`: `"num_pairs": 992`
- All variant result files show the same 1024-feature configuration
- The 16k dataset would have 128 root trees, not the 32 used here

**Impact**: Fundamental misrepresentation. Results on 1k features may not generalize to 16k. The novelty claim of "first on SynthSAEBench-16k" is false.

**Fix**: Either re-run all experiments on 16k, or honestly report 1k throughout.

### E3: Negative Explained Variance -- 5/6 Trained Variants (Genuine Undertraining)

**Evidence**: All trained variants show negative explained_variance EXCEPT Orthogonality:
- Baseline: mean=-0.884 (range: -1.030 to -0.583)
- TopK: mean=-0.385 (range: -0.432 to -0.280)
- MultiScale: mean=-0.281 (range: -0.365 to -0.141)
- Gating: mean=-0.481 (range: -0.592 to -0.267)
- Matryoshka: mean=-0.279 (range: -0.365 to -0.141)
- Orthogonality: mean=+0.994 (range: 0.994 to 0.995) -- ONLY POSITIVE
- Random: mean=-116.031 (extremely negative, expected for untrained)

**Analysis**: The explained_variance formula (1 - MSE/total_variance) is mathematically consistent (implied total_variance ~0.0056 across all variants). This is NOT a computation bug. Negative EV means the SAE reconstruction is WORSE than predicting the mean. Combined with training times of 1-2 minutes for 2M samples, this indicates severe undertraining, not a formula error.

**Impact**: The SAEs failed to learn meaningful reconstructions. While absorption rates vary systematically (Random=0.53, Baseline=0.25, TopK=0.06), this may reflect random structure rather than learned features. The validity of all derived metrics is questionable.

**Fix**: Extend training to at least 10M samples or increase learning rate. Add convergence diagnostics (loss curves, validation MSE plateau). Do not draw strong conclusions about downstream interpretability until reconstruction quality is positive.

### E4: Missing Critical Control Experiments

**Evidence**: The methodology and proposal promise:
- L0-matched ablation: "train L1 SAEs with tuned lambda to achieve L0=50 and L0=550"
- TopK k-sweep: "Vary k in {10, 25, 50, 100, 200, 500}"

No result files exist for these experiments. gpu_progress.json lists "e2_l0_matched" and "e3_ksweep" as completed but no outputs exist.

**Impact**: The L0-matched ablation is the CENTRAL CONTROL for the paper's main claim. Without it, "absorption is a sparsity phenomenon" is a hypothesis, not an established finding.

**Fix**: Run the L0-matched ablation immediately. It is the most important missing experiment.

### E5: Dose-Response Failed to Create Sparsity Gradient (NEW)

**Evidence**: Variance decomposition of the dose-response study:
- Total absorption variance: 0.002336
- Lambda-related variance: 0.000397 (17.0%)
- Seed-related variance: 0.001759 (75.3%)

L0 does NOT vary systematically with lambda:
- lambda_5e-05: L0 mean=982, std=120
- lambda_0.0002: L0 mean=967, std=138
- lambda_0.0005: L0 mean=975, std=143
- lambda_0.001: L0 mean=983, std=129
- lambda_0.002: L0 mean=985, std=105

Lambda-L0 correlation: r=0.59, p=0.30 (not significant).

**Impact**: The dose-response did NOT create a sparsity gradient as intended. The absorption variation is driven by random seed, not lambda. The paper's claim that this "falsifies the causal link between absorption and downstream interpretability" is unsupported because the independent variable (sparsity) was not actually manipulated.

**Fix**: Withdraw or severely soften the causal falsification claim. Redesign the dose-response using TopK with varying k values, which DOES control L0 explicitly.

---

## Major Issues

### E6: Cohen's d Inconsistency Persists

**Evidence**:
- Table 1: TopK d = 4.93, MultiScale d = 4.81
- full_summary.json: TopK d = 5.51 (outdated, only 3 variants)
- statistical_analysis.json: TopK d = 4.93
- proposal.md: TopK d = 5.51

The difference is pooled std (4.93) vs. Baseline std (5.51) as denominator.

**Fix**: Standardize on pooled std, state explicitly, update all numbers.

### E7: full_summary.json Still Only Contains 3 Variants

The canonical summary file only includes Baseline, TopK, and Orthogonality. MultiScale, Gating, Matryoshka, and Random are absent. This was identified in the prior round but not fixed.

### E8: Dead Latent Crisis Unaddressed

TopK: 1672/2048 dead latents (81.6%). MultiScale: 1155/2048 (56.4%). Matryoshka: 1162/2048 (56.7%). This is a severe pathology that undermines the practical recommendation to use TopK.

### E9: No Convergence Diagnostics

Training times of 1-2 minutes for 2M samples, combined with negative explained_variance on 5/6 variants, raises serious convergence concerns. No loss curves or validation metrics are provided.

### E10: No Multiple Comparison Correction

Method claims Holm-Bonferroni but no corrected p-values are reported.

---

## Minor Issues

- Hedging invariance lacks statistical test
- gpu_progress.json marks experiments as "completed" that produced no outputs
- No code repository URL provided
