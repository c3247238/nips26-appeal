# Planning Critique: Local Inhibition Graph Framework

## Summary

The methodology is well-designed on paper: training-free approach, six-phase pipeline, random baseline controls, clear falsification thresholds, and power analysis. However, the planning has a fundamental flaw: it describes experiments that have not been executed. Beyond this, several planning issues weaken the proposed methodology: (1) H8 circularity was not anticipated; (2) H10 sign ambiguity was not addressed in the planning; (3) graph specificity control is missing; (4) n=26 is underpowered for correlation analyses; (5) the k=5 post-hoc choice persists; (6) Pythia cross-validation used a different metric than planned.

## Critical Issues

### 1. Planning Describes Experiments That Have Not Been Executed (CRITICAL)

The methodology.md presents a complete six-phase experimental plan with estimated times, metrics, and expected outcomes. However, Phases 2-6 (H6-H10) have not been executed. The planning document reads as if the experiments are complete, with sentences like "We compute precision@k, recall@k, and AUPR" and "H6 is supported if..." This creates a dangerous illusion of completed work.

**Fix**: Restructure methodology.md to clearly separate "Completed Experiments" (H1-H5) from "Proposed Experiments" (H6-H10). Use future tense for all proposed experiments. Add a disclaimer: "The following phases describe proposed validation experiments that have not yet been executed."

### 2. H8 Circularity Was Not Anticipated in Planning (CRITICAL)

The planning does not address the circularity problem in H8: using the same data (decoder correlations + Chanin absorption on first-letter features) for both graph construction and validation. This is a fundamental methodological flaw that should have been caught in the planning stage. The result debate synthesis flagged it as a "must-fix" gap.

**Fix**: Revise H8 planning to use LOOCV or cross-layer prediction. Add explicit discussion of circularity and how the proposed controls address it.

## Major Issues

### 3. H10 Sign Ambiguity Not Addressed in Planning

The planning specifies a single rebalancing formula (z'_i = z_i + alpha * inh_i) without acknowledging the sign ambiguity. The Skeptic identified this as a genuine problem during the result debate. Good planning should anticipate such ambiguities and specify how they will be resolved.

**Fix**: Revise H10 planning to test BOTH additive and subtractive rules. Specify the decision criterion: "The sign that produces the largest parent firing increase with reconstruction error < 5% will be selected."

### 4. Graph Specificity Control Is Missing

The planning includes three baselines (random graph, non-absorbed control, identity graph) but the "non-absorbed control" is underspecified. The planning states: "Test graph edges for correlated but non-absorbed pairs" without specifying how these pairs will be selected or what comparison will be made.

**Fix**: Specify the non-absorbed control in detail: "For each parent latent, identify the top-k most correlated neighbors that are NOT flagged as absorption pairs by the Chanin metric. Compare the mean edge weight of true absorption pairs vs. non-absorbed correlated pairs using a Welch t-test. If both groups have similar edge weights, the graph is not absorption-specific."

### 5. Sample Size Is Underpowered

With n=26 features, the planning acknowledges limited power for H1-H5 but does not address power for H7-H8. H7 predicts r(recall, inhibition) < -0.3; H8 predicts r > 0.3. Power to detect |r| = 0.3 with n=26 is approximately 25%.

**Fix**: Add power analysis for H7 and H8. If power < 50%, plan to expand the feature set (WordNet hierarchies) or treat these as exploratory.

### 6. k=5 Post-Hoc Choice Persists

The planning specifies k=5 as the "primary analysis point" for precision-recall without acknowledging it was selected post-hoc. The precision invariance claim (central to the competitive suppression explanation) depends on this choice.

**Fix**: Acknowledge the post-hoc nature: "The k=5 sparsity level was selected based on preliminary observation of the data; it was not pre-registered. We report results across all k values for completeness."

### 7. Pythia Cross-Validation Used Different Metric Than Planned

The planning describes Pythia cross-validation as a "direct replication" but the actual execution used "mean_embedding_similarity" rather than the binary success rate used for GPT-2. This metric mismatch limits comparability and was not anticipated in the planning.

**Fix**: Acknowledge the metric difference in the planning. Specify that cross-model validation should use identical metrics, or explicitly justify any metric differences.

## Minor Issues

8. **Gemma-2-2B access blocked**: Planned Gemma experiments were blocked by gated HuggingFace access. Pythia-70M was substituted but is much smaller (70M vs. 2B parameters).

9. **No SAELens version pinned**: Reproducibility requires exact version specification.

10. **Exact Chanin threshold not specified**: The differential correlation threshold for absorption detection is not stated.

## What Works Well

1. **Training-free methodology** lowers barrier to replication
2. **Six-phase pipeline** is logical and well-structured
3. **Clear falsification thresholds** for all hypotheses
4. **Random baseline control** is specified
5. **Multiple comparison correction** is planned
6. **Power analysis** is included for H1-H5
7. **Risk assessment table** is thorough
8. **Resource estimates** are realistic (~2 GPU-hours)
