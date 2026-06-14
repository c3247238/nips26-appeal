# Experiment Critique: Iteration 5

## Overall Assessment

Iteration 5 produces one strong result (Phase 3: 420-SAE scaling surface with significant interaction), one solid-with-caveats result (Phase 1: confound resolution where 3/4 metrics survive L0 control), one conclusive negative misframed as partial success (Phase 2: cross-domain on GPT-2 Small), and one artifact correction that introduces confusion rather than clarity (Phase 4: taxonomy). The most damaging experimental finding is a data pipeline integrity failure: `final_results.json` contains wrong Sobel z values (from the reverse mediation path) and wrong taxonomy rates (92.3% instead of the correct 19.2%).

---

## Phase 1: Confound Resolution

**Verdict**: SOLID with critical caveats on causal claims.

### Strengths

1. The go/no-go criterion is cleanly met: 3/4 quality metrics retain |partial_r| > 0.2 after L0 control. The sparse probing suppression effect (r strengthens from -0.664 to -0.746) is statistically robust.

2. SCR full mediation is the cleanest result: all four Baron-Kenny steps satisfied, direct effect c' = -0.003 (n.s.), bootstrap CI for indirect effect [0.007, 0.048] excludes zero. Sobel z = 3.62, p = 2.9e-4.

3. Rosenbaum Gamma = 2.65 for TPP under Mahalanobis matching is a strong robustness bound.

4. VIF diagnostics (all < 1.1) confirm no multicollinearity. L0-width correlation r = -0.279 is much weaker than feared.

### Critical weaknesses

1. **Within-width matching null**: Three within-width matching strategies all show Gamma = 1.0 for ALL metrics. When width is held constant, there is zero detectable effect of absorption on quality. The causal evidence comes entirely from cross-width comparisons where absorption covaries with width. This is consistent with width (or a latent variable correlated with width) as a common cause of both absorption and quality, rather than absorption mediating L0's effect.

2. **SP-F1 total effect is null (p = 0.45)**: For the metric with the "strongest" suppression effect, there is no overall L0-to-quality relationship. By Zhao et al. (2010), this is "indirect-only" mediation driven by opposing pathways, weaker than classical mediation.

3. **n = 48**: With 4 covariates, the effective degrees of freedom for mediation are limited. Proportion mediated CIs are extremely wide (SP-F1: [-4.15, 4.83]), indicating high instability.

4. **Single architecture, single model**: All 48 SAEs are JumpReLU from Gemma 2 2B. Zero generalization evidence.

### Numerical verification

- Table 1 numbers are consistent with source P1_confound_go_nogo.json.
- Table 3 TPP Sobel z = 2.08 (p = 0.037) is CORRECT per P1_mediation.json.
- final_results.json TPP Sobel z = 2.63 (p = 0.0085) is WRONG -- pulled from `alternative_direction_test` (reverse mediation path).
- Table 3 SCR c' = -0.029 is the standardized coefficient. Abstract uses c' = -0.003, the unstandardized value. Coefficient type is inconsistently labeled.

---

## Phase 2: Cross-Domain Absorption

**Verdict**: CONCLUSIVE NEGATIVE, misframed as "PARTIALLY_SUPPORTED."

### The definitive negative result

- Dominance-based metric: 11.3-96.2% absorption across knowledge domains
- Shuffled-hierarchy control: 100% absorption (randomized labels, same metric)
- Cosine-calibrated metric: 0% across all thresholds

H2 is falsified by the paper's own pre-registered criterion (absorption must exceed 3x shuffled baseline; shuffled baseline = 100%). The "PARTIALLY_SUPPORTED" verdict in final_results.json is indefensible.

### Why dominance-based rates are meaningless

Table 5 reveals the mechanism: R_abs equals FN Rate in every single row. At dominance threshold 1.0, any non-split feature with nonzero activation satisfies the dominance criterion. The metric reduces to "what fraction of tokens lack dedicated split features." This is SAE dictionary sparsity, not absorption.

### The model is fundamentally underpowered

GPT-2 Small (124M) with a 24k SAE where 98% of features are dead on city prompts cannot represent knowledge features as dedicated latents. With ~500 live features for 3,552 cities across dozens of geographic categories, the experiment tests whether a catastrophically undersized dictionary has knowledge features -- not whether absorption occurs on knowledge hierarchies. The 0% cosine-calibrated result is entirely expected from dictionary sparsity alone.

### Salvageable contribution

The metric failure diagnosis is a genuine methodological contribution. The dominance-cosine discrepancy reveals that the Chanin metric does not generalize without modification. This should be framed as "metric validation failure" not "partial absorption evidence."

---

## Phase 3: Scaling Surface

**Verdict**: STRONG -- the best result in the paper.

### Strengths

1. N = 420 SAEs gives genuine statistical power.
2. Interaction p = 3.1e-15 is highly significant, robust to multiple testing.
3. Model comparison is clean: Linear (R2=0.488) < Additive (R2=0.620) < Interaction (R2=0.693).
4. Practical implications are immediately actionable: L0 > 14 threshold, concurrent width-L0 scaling.

### Weaknesses

1. **Architecture confound**: 360 L1 + 54 JumpReLU + 6 unclassified. Architecture is NOT a covariate in the GAM. JumpReLU SAEs occupy specific width-L0 combinations. The "interaction" could partly capture architecture differences.

2. **"Phase boundary" language is overstated**: The pilot summary reports `phase_boundary_detected: false`. The gradient ridge (443 points above 70th percentile) marks a sensitivity zone, not a phase transition. The physics terminology implies sharper transitions than the data show.

3. **30.7% unexplained variance**: Residual structure (by architecture, layer, etc.) is not analyzed.

---

## Phase 4: Taxonomy Correction

**Verdict**: Important artifact correction, but confusing dual-rate reporting.

### Key finding

The corrected comprehensive rate drops from 92.3% to 19.2% -- a 73-point reduction. This confirms the original rate was an artifact of missing comparison baselines. The Chanin false-negative metric independently detects absorption in 73.1% of letters.

### Critical problem

The paper reports two rates (19.2% vs 73.1%) without adequately reconciling them. A reader will not know whether absorption prevalence is 19.2% or 73.1%. The two metrics measure different things (magnitude suppression vs. complete failure-to-fire) but the paper does not clearly explain this distinction or declare which is the validated primary metric.

### Data pipeline failure

final_results.json reports corrected_rate = 0.923, delta = 0.0, verdict = "CORRECTION_MINIMAL." The actual corrected rate is 0.192 (from P4_taxonomy_correction.json). This is the second independent failure in the same summary file.

---

## Data Pipeline Integrity

Two independent failures in final_results.json make the summary file unreliable:

1. **Sobel z values**: All four mediation Sobel z values come from `alternative_direction_test` (reverse path Quality -> Absorption -> L0) instead of the primary mediation path.

2. **Taxonomy rates**: corrected_rate=0.923 (wrong) instead of 0.192 (correct). H5 verdict="CORRECTION_MINIMAL" instead of "CORRECTION_LARGE."

Any downstream consumer of this file receives wrong numbers. The integration code must be audited and fixed before the next iteration.
