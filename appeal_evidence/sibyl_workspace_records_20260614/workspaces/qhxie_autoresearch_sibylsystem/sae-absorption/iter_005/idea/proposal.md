# Final Research Proposal: Iteration 5

## Title

**Beyond the Spelling Task: Resolving Confounds, Extending Domains, and Mapping the Scaling Surface of Feature Absorption in Sparse Autoencoders**

## Abstract

Feature absorption -- where SAE latents silently fail to fire on inputs they should represent because a more specific co-occurring feature has subsumed their role -- is the most important identified failure mode of sparse autoencoders for mechanistic interpretability. Yet the field's empirical understanding rests on a dangerously narrow foundation: a single evaluation task (first-letter spelling), uncontrolled confounds between absorption and SAE width/L0, and an inflated taxonomy rate (92.3%) built on a measurement artifact. This proposal addresses all three weaknesses in a unified, training-free study. **Contribution 1 (Confound Resolution)**: We establish whether the absorption-quality correlation (r=-0.595 across 54 Gemma Scope SAEs) reflects a genuine causal mechanism or an ecological fallacy, using width-stratified partial correlations with L0 as covariate, formal mediation analysis, and Rosenbaum sensitivity bounds -- epidemiological methods applied for the first time to SAE evaluation. **Contribution 2 (Cross-Domain Generalization)**: We provide the first measurement of feature absorption on knowledge hierarchies (city-country, city-continent, city-language) using RAVEL entity-attribute data, directly testing whether the 15-35% absorption rate generalizes beyond first-letter spelling. **Contribution 3 (Absorption Scaling Surface)**: We construct the first empirical absorption phase surface in (log(width), log(L0)) space using 200+ SAEBench-scored SAEs, testing for nonlinear interaction structure that would indicate phase-transition-like behavior. Together, these three contributions transform absorption from a narrowly-validated curiosity into a rigorously characterized, cross-domain phenomenon with actionable implications for SAE selection and architecture design.

## Motivation

### The State of the Problem

Four iterations of this project have established a clear picture: feature absorption is real, present in every tested SAE, and correlates with downstream quality degradation. The iter_4 H3 finding -- a correlation of r=-0.595 (partial r=-0.661 controlling for width and layer) between absorption rate and three independent downstream quality metrics across 54 Gemma Scope SAEs -- is potentially the strongest result in the absorption literature. If the causal chain holds, absorption is not merely a theoretical curiosity but a practical quality indicator that directly informs SAE selection.

However, the iter_4 reflection identified three **blocking** issues that prevent this finding from reaching publication quality:

1. **Width/L0 confound (BLOCKING)**: All 5 high-absorption SAEs are 1M width (L0 16-58); all 5 low-absorption SAEs are 16k/65k (L0 137-297). The partial correlations control for log(width) and layer but **not L0**. If absorption is merely a proxy for L0 (which is itself correlated with width), then the entire absorption-quality narrative collapses into "use SAEs with higher L0." The iter_4 reflection explicitly states: reaching 7.0+ requires zero-GPU confound control, not new experiments.

2. **Single-task evaluation (BLOCKING)**: Every absorption measurement in the literature uses the first-letter spelling task -- a synthetic task with an unusually clean, deterministic hierarchy. At least 5 papers explicitly call out this limitation (Chanin et al., SAEBench, SynthSAEBench, OrtSAE, KronSAE). The contrarian perspective correctly identifies this as a potential source of systematic overestimation: the first-letter task's sharp hierarchy maximizes the sparsity incentive for absorption, while fuzzier real-world hierarchies may exhibit much lower rates.

3. **Taxonomy inflation (BLOCKING)**: The 92.3% combined absorption rate is reported prominently in the paper but rests on a measurement artifact: nearly all letters have n_comparison_tokens=0, forcing a fallback to global mean baseline. The 88.5% Type II rate is an artifact of this fallback, not evidence of partial absorption. Only Type I (3.8%) and Chanin replication (80.8%) have been validated.

### Why This Combination of Contributions

The three contributions are not independent -- they form a logically coherent progression:

- **Confound resolution** (Contribution 1) determines whether the absorption-quality link is real. If it is, we have the field's strongest empirical finding; if not, we know absorption is an epiphenomenon of width/L0 and the entire mitigation research program is misdirected.

- **Cross-domain generalization** (Contribution 2) tests whether absorption exists at all outside the first-letter task. If it does, the phenomenon is fundamental to how SAEs handle hierarchical features; if not, it is an artifact of one particular hierarchy structure.

- **Scaling surface** (Contribution 3) maps the parameter space where absorption occurs, directly enabling practitioners to select SAE hyperparameters that minimize absorption. Combined with Contribution 1, it provides a complete picture: absorption is real, generalizes, and follows predictable scaling.

### Methodological Novelty

The methodological contribution is equally important: transplanting epidemiological causal inference methods (mediation analysis, propensity matching, Rosenbaum sensitivity analysis, Bradford Hill criteria) to SAE evaluation. The SAE community currently relies entirely on correlation-based comparisons between architectures. These methods, standard in medical and social science research, provide rigorous tools for establishing causal claims from observational data -- exactly the challenge faced when comparing SAEs that differ in multiple correlated hyperparameters.

## Research Questions

**RQ1 (Confound Resolution)**: Does absorption causally mediate the effect of L0 on downstream SAE quality, or is the observed correlation an artifact of the width/L0 confound?

**RQ2 (Cross-Domain Generalization)**: Does feature absorption occur in knowledge-domain hierarchies (city-country, city-continent, city-language) at rates exceeding a shuffled-hierarchy baseline, and how do these rates compare to first-letter absorption?

**RQ3 (Scaling Surface)**: How does absorption rate vary jointly with SAE width and L0 across 200+ SAEs, and does the absorption surface exhibit nonlinear interaction structure (phase-boundary-like behavior)?

## Hypotheses

**H1 (Absorption-Quality Causal Chain)**: After controlling for log(L0) as an additional covariate, the partial correlation between absorption and sparse probing accuracy remains significant at |r| > 0.3, p < 0.05. Within at least 2 of 3 width strata (16k, 65k, 1M with sufficient n), the Spearman correlation between absorption and quality metrics has bootstrap 95% CI excluding 0. Absorption mediates > 30% of L0's total effect on downstream quality (indirect effect proportion > 0.30, bootstrap CI excluding 0).

**H2 (Cross-Domain Absorption Existence)**: On Gemma 2 2B with Gemma Scope 16k SAEs at layers 8-17, the absorption rate for city-country features (measured via adapted Chanin metric with probes trained on the correct model) exceeds 10% and exceeds 3x the shuffled-hierarchy baseline.

**H3 (Absorption Scaling Surface)**: In a GAM fit of absorption rate on (log(width), log(L0)), the interaction term ti(log(width), log(L0)) is significant (p < 0.05), indicating that absorption depends on the joint structure of width and L0, not either independently.

**H4 (Cross-Domain Absorption Type)**: Early-type absorption (decoder-absent, no decoder direction within cosine > 0.3 of parent probe) dominates at > 50% of absorbed instances for knowledge-domain hierarchies, consistent with the iter_001 finding on first-letter features.

## Expected Contributions

1. **First rigorous confound control for the absorption-quality relationship**: Disentangling absorption's independent contribution via stratification, mediation analysis, and sensitivity analysis -- establishing or refuting absorption as a causal quality indicator.

2. **First measurement of absorption on knowledge hierarchies**: Using RAVEL entity-attribute data with probes correctly trained on the target model and shuffled-hierarchy controls -- establishing or refuting cross-domain generalizability.

3. **First absorption scaling surface**: A 2D empirical map of absorption rate in (log(width), log(L0)) space across 200+ SAEs, with formal test for nonlinear interaction structure.

4. **First application of epidemiological causal methods to SAE evaluation**: Establishing a methodological precedent (mediation analysis, Rosenbaum bounds) for rigorous causal claims in the SAE evaluation literature.

5. **Corrected taxonomy with honest uncertainty bounds**: Recomputed absorption subtypes using proper comparison tokens, replacing the inflated 92.3% with a validated estimate and transparent upper/lower bounds.

## Method

### Phase 1: Confound Resolution (Zero GPU, ~3 hours)

**Step 1.1: L0 as Covariate** (Critical go/no-go)
- Load the existing iter_4 54-SAE dataset with absorption scores, quality metrics, width, layer, architecture class
- Add log(L0) as covariate: compute partial_corr(absorption, sparse_probing | log_width, layer, arch_class, log_L0)
- Compute for all four quality metrics: sparse probing, RAVEL, SCR, unlearning
- **Go/no-go**: If all four partial correlations drop below |0.2|, the causal chain hypothesis is falsified. Report as a strong negative result and pivot narrative to "width/L0, not absorption, drives quality."

**Step 1.2: Width-Stratified Analysis**
- Partition 54 SAEs into width groups (16k: ~15 SAEs, 65k: ~15 SAEs, 1M: ~18 SAEs)
- Within each group: Spearman correlation between absorption and each quality metric
- BCa bootstrap 95% CIs (10,000 resamples) per stratum
- Report per-stratum sample sizes, effect sizes, and CIs transparently

**Step 1.3: Mediation Analysis**
- Test causal path: L0 -> Absorption -> Quality
- Total effect: quality ~ log(L0) + log(width) + layer
- Direct effect: quality ~ log(L0) + absorption + log(width) + layer
- Indirect effect = total - direct; proportion mediated = indirect / total
- Sobel test + bootstrap CI (10,000 resamples) on indirect effect
- Report for all four quality metrics independently

**Step 1.4: Rosenbaum Sensitivity Analysis**
- Match high-absorption and low-absorption SAEs on width (exact) and L0 (nearest neighbor, caliper 0.2 SD)
- For each matched pair, compute quality difference
- Compute Rosenbaum Gamma at which Wilcoxon signed-rank test becomes non-significant
- Gamma > 1.5 = moderate robustness; Gamma > 2.0 = strong robustness

**Step 1.5: SCR Suppression Variable Diagnosis**
- Sequentially add controls: width-only, layer-only, arch-only, L0-only
- Identify which covariate produces the SCR partial r jump from -0.431 to -0.677
- Report suppression variable decomposition

**Step 1.6: Clustered Standard Error Regression**
- Rerun C2C PMI regression with letter-level clustering (26 clusters)
- Report HC3 and clustered SE results side by side
- Consider beta regression or zero-inflated model given skewness=5.186

### Phase 2: Cross-Domain Absorption (4-6 GPU-hours)

**Step 2.1: Probe Training on Correct Model**
- Load Gemma 2 2B via TransformerLens
- Load RAVEL city dataset (3000+ cities with country, continent, language attributes)
- Construct prompt templates: "{City} is located in the country:", "{City} is on the continent:", "{City} speaks the language:"
- Train multi-class logistic regression probes at layers 8, 12, 17 (3 seeds each)
- Validate probe accuracy (reject if < 85%)
- Note: country-language pair is inherently entangled per SAEBench finding -- report separately

**Step 2.2: Absorption Measurement**
- Load Gemma Scope SAEs (16k, 65k) at layers 8, 12, 17
- Port sae-spelling absorption metric to knowledge hierarchies:
  - k-sparse probing for feature splits (generic, direct transfer)
  - False-negative identification (generic)
  - Integrated-gradients ablation with KnowledgeGrader (~200 lines adaptation)
- Measure absorption rate per attribute per layer per SAE width
- Threshold sweep: cosine similarity threshold in {0.01, 0.025, 0.05, 0.1}, dominance threshold in {0.5, 1.0, 2.0}

**Step 2.3: Controls**
- Shuffled hierarchy control: randomize city-attribute mappings, re-measure absorption (expect < 5%)
- Random probe direction control: random unit vectors as "probe" directions (expect < 5%)
- First-letter baseline: replicate Chanin et al. first-letter measurement on same SAEs for direct comparison
- Dead feature exclusion: report separately for dead vs. alive features

**Step 2.4: Cross-Domain Comparison**
- Side-by-side absorption rates: first-letter vs. country vs. continent vs. language
- Compute hierarchy sharpness: mutual information between entity and attribute value
- Correlate absorption severity with hierarchy sharpness
- Early/late absorption taxonomy on knowledge features (tau sweep 0.2-0.4)

### Phase 3: Absorption Scaling Surface (Zero GPU if using SAEBench data, ~1 hour)

**Step 3.1: Data Collection**
- Download SAEBench precomputed absorption scores for 200+ Gemma Scope SAEs
- Extract width, L0, layer, architecture class for each SAE
- Merge with downstream quality metrics (sparse probing, RAVEL, SCR)

**Step 3.2: Phase Surface Construction**
- Plot absorption rate as 2D heatmap in (log(width), log(L0)) space
- Fit GAM: absorption ~ s(log(width)) + s(log(L0)) + ti(log(width), log(L0))
- Test interaction term significance
- If significant: the absorption surface has structure that cannot be explained by width or L0 independently
- Visualize as contour plot with labeled phase regions

**Step 3.3: Phase Boundary Detection**
- Compute gradient of the GAM surface
- Identify regions of maximal gradient magnitude (candidate phase boundaries)
- Report whether a sharp transition zone exists between low-absorption and high-absorption regions
- If no sharp boundary: report the smooth scaling law with confidence bands

### Phase 4: Taxonomy Correction (1-2 GPU-hours)

**Step 4.1: Proper Comparison Tokens**
- For each letter with n_comparison_tokens=0, use tokens from the same log-frequency band as the target letter's tokens (frequency-matched sampling)
- Alternatively: use sae-spelling ground-truth parent feature IDs from Chanin et al. IG labels to compute proper comparison baselines

**Step 4.2: Recomputed Classification**
- Rerun Type II classification with corrected baselines
- Report three numbers: Type I validated rate, Type II corrected rate, Chanin replication rate
- Mark original 92.3% as "upper bound (uncorrected)" wherever it appears
- Compute corrected combined rate with bootstrap CI

## Experimental Plan

| Priority | Task | Type | GPU-hours | Time | Dependency |
|----------|------|------|-----------|------|------------|
| P0 | Step 1.1: L0 covariate (go/no-go) | Zero GPU | 0 | 1h | None |
| P0 | Step 1.2-1.6: Full confound analysis | Zero GPU | 0 | 2h | Step 1.1 |
| P1 | Step 2.1: RAVEL probe training | GPU | 1.5h | 1.5h | None |
| P1 | Step 3.1-3.3: Scaling surface | Zero GPU | 0 | 1h | None |
| P1 | Step 4.1-4.2: Taxonomy correction | GPU | 1h | 1h | None |
| P2 | Step 2.2-2.3: Absorption measurement | GPU | 3h | 3h | Step 2.1 |
| P2 | Step 2.4: Cross-domain comparison | GPU | 1h | 1h | Step 2.2 |

**Total**: ~5h zero-GPU analysis + 6.5 GPU-hours. All individual tasks within 1-hour constraint except Phase 2 full experiment (can be split into per-layer sub-tasks).

**Pilot Design** (15 minutes):
- Step 1.1 with L0 covariate is the pilot. If partial correlation drops below |0.2| for all metrics, the H1 causal chain is falsified and Phase 1 redirects to characterizing width/L0 as the true driver.
- Step 2.1 pilot: Train city-country probe at layer 12 only, 3 high-frequency + 3 low-frequency countries. If probe accuracy < 75%, abort cross-domain.

## Falsification Criteria (Pre-registered)

**H1 falsified if**: L0-controlled partial correlation between absorption and all four quality metrics drops below |0.2|, AND width-stratified correlations are non-significant in all strata, AND mediation indirect effect bootstrap CI includes 0, AND Rosenbaum Gamma < 1.2.

**H2 falsified if**: No RAVEL hierarchy shows absorption exceeding 3x shuffled baseline after probe quality gate (accuracy >= 85%). Alternatively: absorption rate < 5% across all knowledge attributes.

**H3 falsified if**: GAM interaction term is non-significant (p > 0.10) and the absorption surface is well-described by additive effects of log(width) and log(L0) independently.

**Paper killed if**: H1 AND H2 are simultaneously falsified (absorption is neither a causal quality driver nor a cross-domain phenomenon). In this case, the paper pivots to an "empirical audit" framing: "Feature absorption is less severe, less general, and less causally important than previously believed."

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| L0 confound kills H1 | High | 35% | This is the most informative outcome. Establishes width/L0 as the true driver -- a strong negative result. |
| RAVEL probes low quality on Gemma 2B | Medium | 20% | Prior work reports >90% accuracy; fallback to city-continent (simplest hierarchy) |
| Gemma 2B model access blocked | Medium | 15% | Fall back to GPT-2 Small (guaranteed access) for cross-domain; use Gemma Scope SAEs only for scaling surface |
| Width strata have n < 10 | High | 40% | Use nonparametric methods; report exact p-values and bootstrap CIs; supplement with 200+ SAE SAEBench data |
| Absorption near zero on knowledge tasks | Low | 20% | Publishable negative result that distinguishes knowledge from syntactic features |
| SAEBench data unavailable for 200+ SAEs | Low | 10% | Fall back to the 54-SAE dataset for scaling analysis |

## Novelty Assessment

### Candidate Front-Runner: Absorption-Quality Causal Chain + Cross-Domain Generalization

**Prior art search results**:
- No paper has performed mediation analysis on the absorption-quality relationship (searched: "SAE mediation analysis", "SAE propensity matching", "SAE causal inference quality" -- zero relevant results)
- No paper has measured absorption on RAVEL knowledge hierarchies. SAE-RAVEL (Chaudhary & Geiger, 2024) measured disentanglement, not absorption. The gap is explicitly noted in Chanin et al. and at least 4 other papers.
- No paper has produced a formal 2D absorption scaling surface with interaction term analysis. The closest is Chanin et al. Figures 9b/9c showing absorption vs. L0 at different widths, but without formal regression or interaction testing.
- No paper has applied Rosenbaum bounds or mediation analysis to any SAE evaluation question.

**Differentiation from prior iterations**:
- Iter_1-3 proposed novel detection methods (EDA, EncNorm, ITAC) -- all were either not novel (EDA = SAEBench metric) or near-null (ITAC 3% FN reduction). This iteration does NOT propose new detection methods.
- Iter_4 proposed Lotka-Volterra framework and achieved the H3 correlation finding but did NOT control for confounds. This iteration directly resolves that blocking issue.
- The cross-domain contribution fills Gaps 2 and 6 from the literature survey. The confound resolution fills Gap 1 (quantitative causal theory). The scaling surface fills Gap 3 partially (joint scaling).

### Candidate Backup A: Phase Diagram with Theoretical Scaffolding

If the confound analysis shows absorption is not a causal quality mediator, the backup contribution becomes the scaling surface alone, enriched with the theoretical phase diagram framework from the theoretical perspective:
- Three regimes: hedging (low width), absorption (intermediate width/low L0), recovery (high width/high L0)
- Predictions from the L0 budget analysis: absorption rate ~ max(0, 1 - L0/(d_eff + 1))
- Validated on the empirical surface data

### Candidate Backup B: Cross-Domain Characterization as Primary

If the scaling surface shows no interesting structure, the backup pivots to making cross-domain absorption the headline contribution:
- First evidence that absorption generalizes (or does not generalize) beyond first-letter spelling
- Correlation between absorption severity and hierarchy sharpness (MI between parent and child)
- Practical implications for safety-relevant features

## What Changed from Previous Iterations

1. **No new detection methods**: Iters 1-3 spent effort on novel detectors (EDA, EncNorm, ITAC) that either were not novel or did not work. This iteration redirects entirely to **empirical characterization and confound control** -- the highest-ROI direction identified by the iter_4 reflection.

2. **L0 as explicit covariate**: The single most important analysis identified as blocking by iter_4 but never executed. This is now P0 priority.

3. **Cross-domain measurement on correct model**: Iter_1 attempted RAVEL cross-domain but trained probes on the wrong model (Qwen2.5-0.5B projected to Gemma 2B). This iteration requires probes trained on the target model with shuffled-hierarchy controls -- the lesson from iter_1's falsified H3.

4. **Taxonomy correction**: The 92.3% rate is honestly acknowledged as an upper bound inflated by the n_comparison_tokens=0 fallback. It is corrected rather than defended.

5. **Epidemiological causal methods**: The mediation analysis, propensity matching, and Rosenbaum bounds are a genuine methodological contribution that no prior SAE paper has attempted.

## Perspectives Weighting

This synthesis weighted the six perspectives as follows:

- **Innovator** (weight: HIGH): The phase diagram concept and the compressed-sensing analogy for the scaling surface are adopted. The causal abstraction score is deferred to future work as theoretically elegant but practically complex.
- **Pragmatist** (weight: HIGHEST): The execution-first philosophy drives the entire proposal. A1 (L0 covariate) as critical go/no-go is directly from the pragmatist. The RAVEL cross-domain approach uses existing infrastructure (sae-spelling + SAELens + RAVEL + Gemma Scope).
- **Theoretical** (weight: MODERATE): The three-regime phase diagram framework provides theoretical scaffolding for the scaling surface, but the emphasis is empirical rather than proving theorems. The L0 budget argument is used qualitatively.
- **Contrarian** (weight: HIGH): The L0 confound concern and single-task limitation are taken seriously as BLOCKING issues, not minor caveats. The contrarian's hypothesis that >30% of reported absorption is L0-induced hedging motivates the confound analysis. However, the extreme claim ("studying a dead paradigm") is rejected given SAEs remain the community standard.
- **Interdisciplinary** (weight: LOW for this iteration): The immunodominance framework is intellectually rich but adds complexity without addressing the blocking issues. Elements are retained for the discussion section (active suppression signature, epitope focusing as explanation for masked regularization). Full development deferred to a future iteration.
- **Empiricist** (weight: HIGHEST): The falsification criteria, statistical test plan, sample size requirements, ablation schedule, and control experiments are all drawn from the empiricist. The honest framing of encoder-decoder cosine as an existing SAEBench metric (not a novel contribution) is maintained.
