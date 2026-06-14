# Planning Critique: Iteration 009

## Executive Summary

The plan (plan/methodology.md) is methodologically sound in its design: component-isolated ablation, ground-truth synthetic data, 5 replicates per variant, and appropriate statistical methods. However, the plan was not executed as specified, and critical data integrity issues identified in prior rounds remain unaddressed. The most critical failures are: (1) the L0-matched ablation -- the central control for the paper's main claim -- was never run; (2) the plan claimed 16k features but the implementation used 1k; (3) the k-sweep and real-LLM validation were never executed; (4) no data validation pipeline caught the Matryoshka copying bug or the negative explained_variance anomaly; (5) the dose-response study failed to create a sparsity gradient (L1 regularization did not vary L0 systematically). The gap between plan and execution is the primary planning failure.

## Plan vs. Execution Comparison

| Experiment | Plan Promise | Actual Execution | Gap |
|------------|-------------|------------------|-----|
| Full experiment (6 variants x 5 replicates) | Run on 16k features | Ran on 1k features | CRITICAL |
| L0-matched ablation | Train L1 SAEs with tuned lambda for L0=50, 550 | NOT RUN | CRITICAL |
| TopK k-sweep | Vary k in {10, 25, 50, 100, 200, 500} | NOT RUN | MAJOR |
| Real-LLM validation | Test on Pythia-160M or Gemma-2-2B | NOT RUN | MAJOR |
| Statistical analysis | ANOVA, Tukey HSD, Cohen's d, Holm-Bonferroni | Partial (no corrected p-values) | MAJOR |
| Convergence diagnostics | Loss curves, validation MSE | NOT PROVIDED | MAJOR |
| Data integrity check | Cross-file duplicate detection | NOT IMPLEMENTED | CRITICAL |
| Explained variance sanity check | All variants should have EV >= 0 | 5/6 variants have EV < 0 | CRITICAL |
| Dose-response sparsity gradient | L1 lambda sweep creates L0 variation | L0 ~980 across all lambdas | CRITICAL |

## Methodology Assessment

### Strengths

1. **Component-isolated design**: Varying one component at a time enables causal attribution. This is the correct approach.
2. **Ground-truth measurement**: Direct absorption from known parent-child relationships eliminates probe artifacts.
3. **Multiple controls**: Random control, L0-matched comparison, and reconstruction-absorption Pareto frontier provide robust validation (in theory).
4. **Appropriate statistical methods**: One-way ANOVA, Tukey HSD, Cohen's d, and correlation analysis are all standard and appropriate.
5. **5 replicates with fixed seeds**: Variance estimation is reliable for the variants that were genuinely run.

### Weaknesses

1. **Scale mismatch**: The plan specifies SynthSAEBench-16k but the implementation used 1k features. This was not caught by any verification step.
2. **L0-matched ablation not prioritized**: This is the central control for the sparsity-mediation claim. It should have been run before any writing began.
3. **No data validation pipeline**: The Matryoshka data-copying bug (4/5 replicates identical to MultiScale) would have been caught by a simple cross-file duplicate check. The negative explained_variance across 5/6 variants would have been caught by a sanity check.
4. **Missing real-LLM validation**: The plan includes Phase 2 validation but it was never executed.
5. **Training time estimates were wrong**: The plan estimates ~60 min per variant but actual training took 1-2 minutes. While this is not a problem per se, combined with negative explained_variance it suggests training did not converge.
6. **No convergence criteria**: The plan does not specify how to verify that training converged (loss plateau, validation MSE, etc.).
7. **No explained_variance sanity check**: The plan did not include a requirement that explained_variance >= 0 for trained models.
8. **Dose-response design flaw**: The plan assumed L1 lambda sweep would create a sparsity gradient. In practice, L0 remained ~980 across all lambda levels. TopK with varying k would have been a better design.

## Task Plan Assessment

The gpu_progress.json shows all tasks as "completed" including "e2_l0_matched" and "e3_ksweep". However, no result files exist for these tasks. This indicates that tasks were marked complete without producing actual output. The planning process should have:
- Verified that result files exist before marking tasks complete
- Included a data integrity check in the completion criteria
- Had a clear escalation path when experiments could not be run as planned

## Risk Assessment Accuracy

| Threat | Likelihood (Plan) | Actual Outcome |
|--------|-------------------|----------------|
| L0-matched shows sparsity is sole driver | Medium | NOT TESTED |
| MultiScale overturns TopK dominance | Low | Did not happen; co-dominant |
| Synthetic data doesn't match LLM structure | High | Acknowledged; Phase 2 not run |
| Dead latent crisis undermines TopK | Medium | Confirmed (81.6% dead latents); not adequately addressed |
| Negative explained_variance indicates undertraining | NOT IDENTIFIED | Confirmed (5/6 variants negative); completely unaddressed |
| Data copying between variants | NOT IDENTIFIED | Confirmed (4/5 Matryoshka = MultiScale); completely unaddressed |
| Dose-response fails to create sparsity gradient | NOT IDENTIFIED | Confirmed (L0 ~980 across all lambdas); undermines causal claim |

The plan failed to identify the negative explained_variance, data-copying, and dose-response gradient risks, which are now the most alarming findings.

## The Explained Variance Anomaly

The most significant unanticipated finding is that 5/6 trained variants have negative explained_variance:

| Variant | explained_variance | Interpretation |
|---------|-------------------|----------------|
| Baseline | -0.884 | WORSE than mean predictor |
| TopK | -0.385 | WORSE than mean predictor |
| MultiScale | -0.281 | WORSE than mean predictor |
| Orthogonality | +0.994 | Better than mean predictor |
| Gating | -0.481 | WORSE than mean predictor |
| Matryoshka | -0.279 | WORSE than mean predictor |
| Random | -116.031 | Expected for untrained |

The formula (1 - MSE/total_variance) is mathematically consistent (implied total_variance ~0.0056 across variants). This is genuine undertraining, not a computation bug. The orthogonality variant achieves MSE ~3e-5 (near-perfect reconstruction) with positive EV, but all other trained variants have MSE ~0.007-0.01 with NEGATIVE EV. Combined with 1-2 minute training times for 2M samples, this strongly suggests insufficient training.

## The Dose-Response Gradient Failure (NEW)

The dose-response study was intended to create a sparsity gradient by varying L1 lambda. However:

| Lambda | Mean L0 | Std L0 | Mean Absorption |
|--------|---------|--------|-----------------|
| 5e-05 | 982 | 120 | 0.199 |
| 0.0002 | 967 | 138 | 0.207 |
| 0.0005 | 975 | 143 | 0.217 |
| 0.001 | 983 | 129 | 0.236 |
| 0.002 | 985 | 105 | 0.255 |

Lambda-L0 correlation: r=0.59, p=0.30 (not significant).
Variance decomposition: 75.3% seed-related, 17.0% lambda-related.

The L1 regularization failed to produce a systematic sparsity gradient. The dose-response did NOT test what it claimed to test. A TopK k-sweep (varying k explicitly) would have been the correct design.

## Recommendations for Planning

1. **Add a data integrity gate**: Before any analysis or writing, verify that:
   - All replicate files are unique (no cross-variant copying) -- use MD5 hashes
   - The feature count matches the plan specification
   - All promised experiments have actual result files
   - Key metrics pass sanity checks (explained_variance >= 0 for trained models, MSE >= 0, dead_latents < 50% for "viable" architectures)
2. **Prioritize critical controls**: The L0-matched ablation is not optional -- it is essential for the central claim.
3. **Verify scale before writing**: Confirm num_features in result files matches the paper's claims.
4. **Add convergence diagnostics**: Require loss curves or final loss values as part of the output.
5. **Distinguish "planned" from "verified complete"**: Marking tasks as complete in gpu_progress.json should require verification of output files AND sanity checks.
6. **Add metric sanity checks**: explained_variance should be >= 0 for any model that learns. Flag violations automatically and halt the pipeline.
7. **Add cross-variant duplicate detection**: Compare MD5 hashes of replicate data across variants to detect copying bugs.
8. **Redesign dose-response**: Use TopK with varying k (explicit L0 control) instead of L1 lambda sweep (which failed to produce gradient).
