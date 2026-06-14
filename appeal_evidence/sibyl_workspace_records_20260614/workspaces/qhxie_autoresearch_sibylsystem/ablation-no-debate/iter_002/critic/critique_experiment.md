# Critique: Experiment Design and Execution

## Overview
The experimental design is solid with appropriate use of synthetic ground truth, multi-seed validation, and both synthetic and real SAE testing. However, several methodological choices limit the reliability and generalizability of the conclusions.

## Strengths

1. **Ground-truth hierarchy**: Using synthetic data with known parent-child relationships is the correct approach for measuring absorption without relying on human-interpretable labels. This is the right methodology.

2. **Multi-seed validation**: The 5-seed replication across stochastic hierarchy generation addresses the zero-variance concern from the pilot and strengthens confidence in the robustness claim.

3. **Factorial structure**: The 2x2 factorial design is the appropriate method for decomposing encoder vs. decoder contributions. The design is sound in principle.

4. **Real SAE validation**: Testing on GPT-2 Small SAEs (instead of only synthetic) for H_Safe is the right approach, even though it produced a null result.

## Weaknesses

### 1. Post-Hoc Criterion Revision for H_Mech
**Severity: Critical**

The original H_Mech criteria (B approx D, C approx A) failed on 14 of 15 runs because the decoder compensation effect (D < B) was not anticipated. The paper revised the criteria post-hoc to encoder effect > 0.5 and decoder effect < 0.1. This revision is not pre-registered and undermines the confirmatory status of the result.

From multiseed_validation.json evidence, the pilot was misleading: pilot B=0.490 vs D=0.484 (looked like B≈D), but full experiment showed B=0.861 vs D=0.436 (B >> D). The pilot gave false confidence in the original criteria.

**Fix**: Pre-register revised criteria or acknowledge the result as exploratory evidence for the encoder hypothesis.

### 2. Zero Variance in Random Baselines
**Severity: Major**

The random SAE baselines in multiseed_validation.json show std ≈ 0 (1e-17 to 1e-18) across all seeds. This means the random absorption rate is effectively deterministic regardless of seed. This is highly unusual and raises questions about:
- Whether the absorption measurement is sensitive to the random SAE's actual learned state
- Whether the random baseline is a property of the TopK activation function itself (producing the same sparsity pattern regardless of weight initialization)
- Whether the effect size (Cohen's d > 10) is mathematically inflated

The raw absorption values for random SAEs vary slightly (0.024 to 0.049) but the std within each seed is machine epsilon.

**Fix**: Investigate the zero-variance phenomenon, report it explicitly, and recalculate effect sizes with appropriate variance estimates.

### 3. d_model = 128 Is Far Below Real LLM Scale
**Severity: Major**

All synthetic experiments use d_model=128, while real LLM residual streams operate at 768 to 12,288 dimensions. The encoder-decoder dynamics at d=128 with d_sae=4096 (32x expansion) may be qualitatively different from d=4096 with d_sae=16,000+ (4x expansion). The overcompletion ratio matters for absorption dynamics.

The paper acknowledges this as a limitation but does not provide even a single d=512 validation point to establish scalability.

**Fix**: Add a d_model=512 condition to the factorial design to establish whether encoder dominance holds at intermediate scale.

### 4. Held-Out Generalization n=1 Is Too Small
**Severity: Major**

The held-out generalization test holds out n=1 hierarchy per seed (20% of 5 hierarchies), yielding only 5 data points for the train-test correlation. The reported r=0.998 has 3 degrees of freedom and an unstated but likely wide confidence interval.

A correlation based on 5 points is mathematically fragile. The paper acknowledges this but does not address it.

**Fix**: Hold out n=3 hierarchies per seed (60% test split) to get 15 data points for the correlation.

### 5. Metric Inconsistency Across Experiments
**Severity: Major**

Three different absorption metrics are used:
- Factorial decomposition: "overlap fraction" (intersection / parent set)
- Multi-seed: "Jaccard overlap" (the paper claims Jaccard but the formula appears identical to overlap fraction)
- Safety analysis: "cosine-based proportional absorption"

The paper acknowledges this in Section 3.3 but does not resolve it. Table 1 presents results from all three metrics without clear labeling.

**Fix**: Pick one canonical metric, recompute all experiments, or clearly label metric-specific sections.

### 6. Steering Sensitivity Ratio Uses Wrong Proxy
**Severity: Minor**

The steering sensitivity metric measures "change in reconstruction quality when the feature is ablated." This measures reconstruction importance, not steering responsiveness. A feature could be reconstruction-important but not steering-responsive (if steering does not change its activation magnitude).

The steering literature typically measures steering responsiveness as change in activation magnitude or downstream logit affected.

**Fix**: Use activation magnitude change as the primary steering responsiveness metric, not ablation sensitivity.

### 7. One-Way ANOVA Promised But Never Run
**Severity: Major**

Section 3.5 methodology promises "a one-way ANOVA across all completed variants" but this is never reported in the Results. The paper uses pairwise comparisons instead, which is more appropriate given the sample sizes, but the unfulfilled promise undermines credibility.

**Fix**: Either add the ANOVA or remove the promise from the methodology.

## Summary
The experimental design is methodologically sound in its use of ground-truth hierarchies and factorial decomposition, but post-hoc criterion changes, metric inconsistency, and small-scale synthetic experiments limit the reliability of the core claims. The core H_Mech finding is real but should be framed as exploratory given the criterion revision.
