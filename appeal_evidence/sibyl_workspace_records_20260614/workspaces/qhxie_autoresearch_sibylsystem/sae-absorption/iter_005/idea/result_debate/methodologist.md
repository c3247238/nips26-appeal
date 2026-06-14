# Methodology Audit: Iteration 5

**Auditor**: Methodologist Agent
**Date**: 2026-04-15
**Scope**: Experimental methodology of all four phases (confound resolution, cross-domain absorption, scaling surface, taxonomy correction)

---

## 1. Baseline Fairness Audit

### Phase 1 (Confound Resolution)

**Baselines used**: Null model (absorption uncorrelated after confound control) and alternative model (L0 alone explains quality variation).

**Fairness assessment**: ADEQUATE WITH CAVEATS.
- The partial correlation analysis (P1 go/no-go) properly compares the same dataset (48 SAEs with L0 data) under two covariate sets (with vs. without L0). This is a fair within-sample comparison.
- The mediation analysis uses the same sample for all paths (L0->absorption, absorption->quality, L0->quality). No asymmetric tuning.
- **CONCERN**: 6 SAEs were excluded because they lacked L0 values (canonical SAEs). This could introduce selection bias if canonical SAEs systematically differ in absorption behavior. The analysis goes from 54 to 48 SAEs, and no sensitivity check was reported for whether the excluded SAEs would change conclusions.

### Phase 2 (Cross-Domain Absorption)

**Baselines used**: Shuffled hierarchy control, random probe direction control, first-letter baseline from same SAEs.

**CRITICAL ASYMMETRY**: The cross-domain absorption measurement (Phase 2) used GPT-2 Small (124M params) as a fallback because Gemma 2 2B was gated. However, the original proposal designed the experiment for Gemma 2 2B with Gemma Scope SAEs. The actual SAE used (`gpt2-small-res-jb`, 24k features) has 98% dead features, which fundamentally changes the experimental conditions. The first-letter baseline used the same SAE (fair within Phase 2), but the results cannot be directly compared to the Phase 3 scaling surface (which uses Gemma 2 2B SAEBench data from 420 Gemma Scope SAEs). **This model mismatch is the single largest methodological weakness of the iteration.**

### Phase 3 (Scaling Surface)

**Baselines used**: Linear model, additive GAM, interaction GAM.

**Fairness assessment**: GOOD. The three-model comparison (linear -> additive -> interaction) provides a proper nested model comparison. The 420-SAE sample from SAEBench uses precomputed metrics that were generated under a standardized protocol. No tuning asymmetry detected.

### Phase 4 (Taxonomy Correction)

**Baselines used**: Original taxonomy rates from iter_004, frequency-matched comparison tokens, WikiText-103 corpus activations.

**CRITICAL DISCREPANCY**: The final_results_summary.md reports "Original rate: 92.3%, Corrected rate: 92.3% (delta = 0.0%)" but the actual P4_taxonomy_correction.json shows a **dramatically different** result: corrected comprehensive rate = **19.2%** (down from 92.3%), with corrected Type II rate = 15.4% (down from 88.5%). The summary file appears to have reported the wrong numbers. The actual correction found that 19/26 letters changed classification, primarily because parent features are too letter-specific for meaningful non-letter comparison baselines. The final_results.json top-level `H5_taxonomy_correction` entry reports `delta: 0.0` which contradicts the per-letter evidence. **This is a data reporting error that must be resolved before publication.**

---

## 2. Metric-Claim Alignment

| Claimed Contribution | Evaluation Metric | Alignment |
|-----|-----|-----|
| C1: Absorption causally mediates L0->quality | Partial correlation with L0 control; mediation indirect effect; Rosenbaum Gamma | GOOD -- three independent methods testing the same causal claim. However, "causal" is a strong word for observational data with n=48. |
| C2: Cross-domain absorption exists on knowledge hierarchies | Absorption rate per attribute; comparison vs. shuffled control | POOR -- the dominance-based metric was invalidated by 100% shuffled control rate, and the cosine-calibrated metric found 0%. The metric does not actually measure what was claimed. |
| C3: Nonlinear scaling surface | GAM interaction p-value; R-squared comparison | GOOD -- the interaction term directly tests the hypothesis, and model comparison provides clear evidence hierarchy. |
| C2 revised: Absorption metric does not transfer | Comparison between dominance-based and cosine-calibrated rates; shuffled vs. real rates | ADEQUATE as a negative/diagnostic result, but does not answer the original question (whether absorption occurs in knowledge hierarchies). |
| C4: Taxonomy correction | Type II corrected rate; Chanin absorption rate | MIXED -- the Chanin metric (73.1% any-absorption) provides genuine validation, but the headline taxonomy correction is confounded by the reporting discrepancy noted above. |

**Measurement gaps**:
- The proposal claimed C2 would test "whether the 15-35% absorption rate generalizes beyond first-letter spelling." The experiment cannot answer this because the metric itself failed, not because absorption does not exist. The cosine-calibrated metric finding 0% with a threshold of 0.05 while first-letter selectivity-based detection finds 85-93% on the same model suggests the two metrics are measuring different things. No intermediate metric was tried.
- For C1, "causal" mediation from cross-sectional data is inherently limited. The Bradford Hill assessment honestly acknowledges this (temporality: "plausible but not established"), but the paper title uses "causal chain" language.

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: NO CONCERN. All analyses use precomputed SAEBench metrics or iter_004 data. No training involved in Phase 1 or Phase 3. Phase 2 probes use 80/20 splits. Phase 4 uses external WikiText-103 corpus.

- [x] **Contamination**: LOW CONCERN. SAEBench benchmarks are standardized. The absorption metric is computed from SAE feature activations, not model predictions on benchmarks.

- [ ] **Selection bias**: MODERATE CONCERN.
  - 6 canonical SAEs excluded from Phase 1 (L0=null). No sensitivity check reported.
  - Phase 2 fallback from Gemma 2 2B to GPT-2 Small introduces severe selection bias: GPT-2 Small has far less factual knowledge, different SAE quality characteristics, and 98% dead features. The proposal explicitly designed Phase 2 for Gemma 2 2B because "country-language pair is inherently entangled per SAEBench finding." Running on GPT-2 Small does not test the same hypothesis.
  - Continent probe accuracy: 45-52% (vs. 85% quality gate specified in methodology). The methodology stated "Probe accuracy >= 85% required to proceed; reject layers/attributes below threshold." This gate was retroactively relaxed to 40% for Tier 2 probes. While pragmatic, this is a post-hoc decision that was not pre-registered.

- [ ] **Hyperparameter tuning on test set**: MODERATE CONCERN.
  - The absorption threshold sweep in Phase 2 (selectivity thresholds 2-10, dominance thresholds 0.5-3.0) produces a wide range of absorption rates (from 2.6% to 100%) depending on parameters. The "headline" rate used for reporting (selectivity=3, dominance=0.5) appears to have been selected post-hoc to produce rates in the 50-85% range, comparable to first-letter. The 100% rates at high selectivity thresholds and the 100% shuffled controls were discovered after the threshold was chosen.
  - Phase 1 mediation: The "proportion mediated" for sparse probing is 4.785 (>100%), which is statistically unstable because the total effect is near zero. This was flagged but not resolved -- the proportion mediated is meaningless when the total effect confidence interval includes zero.

- [ ] **Overfitting to evaluation**: MODERATE CONCERN.
  - All Phase 1 results come from 48 Gemma 2 2B SAEs. No cross-model validation. The proposal proposed GPT-2 Small as an "open-model anchor" for Phase 4 but not for Phase 1. If the absorption-quality correlation is model-specific, the finding does not generalize.
  - Phase 3's 420 SAEs are all Gemma 2 2B (6 width values, 3 layers, multiple L0 settings). The interaction may be architecture-specific.

---

## 4. Ablation Gap Analysis

| Proposed Component | Corresponding Ablation | Status |
|-----|-----|-----|
| L0 as covariate | Compare partial correlations with/without L0 | PRESENT (P1 go/no-go) |
| Width-stratified analysis | Within-stratum vs. pooled correlations | PRESENT (P1 width-stratified) |
| Mediation (L0->Absorption->Quality) | Compare total vs. direct effect | PRESENT (P1 mediation) |
| Rosenbaum sensitivity | Multiple matching strategies compared | PRESENT (5 strategies) |
| Cross-domain absorption metric | Shuffled control, random probe control | PRESENT but controls invalidated the metric |
| Cosine-calibrated absorption | Sweep over cosine thresholds (0.05-0.3) | PRESENT |
| GAM interaction term | Additive vs. interaction model comparison | PRESENT |
| Taxonomy Type II correction | Strategy A/B/C comparison | PRESENT |

**MISSING ablations**:
1. **Excluded SAE sensitivity**: No ablation checking whether re-including the 6 canonical SAEs (with imputed L0) changes Phase 1 conclusions.
2. **Layer-specific analysis for Phase 1**: Mediation and partial correlations are pooled across layers 5/12/19. The within-layer analysis is limited to the width-stratified analysis. Given that layer is identified as the primary suppressor for SCR, layer-specific mediation would strengthen the causal argument.
3. **Dead feature ablation for Phase 2**: The SAE has 98% dead features, but no ablation checks whether results change when restricting to alive features only. The controls show 100% absorption on shuffled data -- this could be entirely driven by the concentration of activations in a handful of alive features (especially "super-absorber 8213" mentioned in the interpretation).
4. **Model-scale ablation for Phase 2**: No comparison between GPT-2 Small and any larger model, despite the proposal identifying model scale as a key variable.
5. **Architecture ablation for Phase 3**: The 420 SAEs include both standard and JumpReLU architectures (54 JumpReLU, 360 standard, 6 unknown). No ablation checks whether the interaction effect holds within each architecture class separately.

---

## 5. Reproducibility Score

| Criterion | Score | Justification |
|-----|-----|-----|
| Random seeds fixed | 5/5 | Seeds 42/123/456 used consistently. Bootstrap seed sequences documented. |
| Hyperparameters specified | 4/5 | Most specified (probe C=1.0, max_iter=1000, bootstrap=10000). Phase 2 selectivity/dominance thresholds swept but "default" choice not pre-registered. |
| Code/data availability | 3/5 | All experiment code exists locally. SAEBench data from HuggingFace is public. iter_004 data is project-internal but would need to be released. GPT-2 Small SAE is public via SAELens. |
| Hardware requirements | 3/5 | GPU requirement documented for Phase 2/4. Specific GPU model not documented. Phase 1/3 are zero-GPU. |
| Reproducibility within 10% | 3/5 | Phase 1 and Phase 3 are deterministic (fixed seeds, bootstrap). Phase 2 results would vary significantly on a different model (Gemma 2B vs GPT-2 Small). Phase 4 depends on WikiText-103 sentence sampling. |

**Overall Reproducibility Score: 3.6 / 5**

Key reproducibility risks: (a) Phase 2 model fallback means the exact experiment cannot be replicated on Gemma 2 2B without HuggingFace access; (b) the Phase 4 taxonomy correction result contradicts the summary file, making it unclear which number is correct.

---

## 6. Top-3 Methodology Improvements

### Recommendation 1: Resolve the Phase 4 reporting discrepancy (HIGHEST PRIORITY, LOW EFFORT)

**Issue**: The final_results_summary.md and final_results.json report "corrected rate = 92.3%, delta = 0.0" but the detailed P4_taxonomy_correction.json shows corrected comprehensive rate = 19.2% (from 92.3%) and 19/26 letters changed classification. This is a factual contradiction in the results that will immediately undermine reviewer trust.

**Action**: Determine which result is correct by re-examining the code that produced both files. If the detailed per-letter analysis is correct (19.2%), update the summary and integration files. If both are correct under different definitions (e.g., "corrected" = non-letter-context baseline vs. "validated" = Chanin-based), define both clearly and report the Chanin 73.1% any-absorption rate as the primary metric.

**Impact**: Eliminates the most damaging credibility threat with near-zero experimental effort.

### Recommendation 2: Repeat Phase 2 on Gemma 2 2B or acknowledge the model mismatch as a limitation (HIGH PRIORITY, MODERATE EFFORT)

**Issue**: Phase 2 was designed for Gemma 2 2B but executed on GPT-2 Small. The 98% dead feature rate in the GPT-2 Small SAE makes any absorption measurement fundamentally unreliable -- the dominance metric collapses to measuring which of the ~500 alive features has highest activation, not whether the dominant feature is aligned with the probe direction. The shuffled control rate of 100% confirms this.

**Action (preferred)**: Obtain HuggingFace token for Gemma 2 2B and re-run Phase 2 on the target model with Gemma Scope SAEs (which have far lower dead feature rates).

**Action (fallback)**: If model access remains blocked, reframe C2 as a diagnostic contribution: "The Chanin absorption metric, designed for first-letter spelling on models with low dead-feature rates, does not transfer to knowledge hierarchies on small models with high dead-feature rates." Report the cosine-calibrated 0% and dominance-based 100% shuffled rate as evidence that the metric itself needs adaptation, not as evidence about absorption.

**Impact**: Either produces the cross-domain evidence the paper needs (preferred) or honestly characterizes the limitation (fallback). Currently, the "PARTIALLY_SUPPORTED" verdict is misleading because the metric was invalidated.

### Recommendation 3: Add within-architecture ablation for Phase 3 interaction (MEDIUM PRIORITY, LOW EFFORT)

**Issue**: The 420-SAE dataset contains 54 JumpReLU, 360 standard, and 6 unknown architecture SAEs. JumpReLU SAEs may have systematically different absorption behavior (different activation function structure). The interaction effect (p=3.1e-15) could be partially driven by architecture confounds if JumpReLU SAEs cluster at specific width/L0 combinations.

**Action**: Fit the interaction GAM separately for standard-only (n=360) and JumpReLU-only (n=54) SAEs. If the interaction remains significant for standard SAEs alone, the finding is robust. If it weakens or disappears, architecture is a confound that must be reported.

**Impact**: This is a zero-GPU, 15-minute analysis that substantially strengthens the Phase 3 claim. Given that the interaction p-value is extremely small (3.1e-15), it will likely survive, but the check is necessary for methodological rigor.

---

## Summary of Verdict

**Overall methodology quality**: MODERATE-TO-GOOD for Phases 1, 3, and 4 (taken individually). POOR for Phase 2 due to the model fallback invalidating the cross-domain metric.

**Credibility-threatening issues** (must fix before publication):
1. Phase 4 reporting discrepancy (92.3% vs. 19.2%)
2. Phase 2 model mismatch rendering H2 verdict unreliable
3. Phase 2 "PARTIALLY_SUPPORTED" is misleading -- the metric was invalidated, not the hypothesis partially confirmed

**Strengths**:
- Phase 1's multi-method convergence (partial correlations, mediation, Rosenbaum, stratification) is genuinely strong for observational causal inference. The suppression effect discovery (sparse probing strengthening after L0 control) is a noteworthy finding.
- Phase 3's 420-SAE sample with formal interaction testing is methodologically clean.
- Phase 4's three-strategy correction approach is thorough, and the Chanin-based validation at 73.1% is a credible number.
- Controls and ablations are generally well-designed for the analyses that used the correct model.
- Pre-registered falsification criteria are clearly stated and honestly evaluated.
