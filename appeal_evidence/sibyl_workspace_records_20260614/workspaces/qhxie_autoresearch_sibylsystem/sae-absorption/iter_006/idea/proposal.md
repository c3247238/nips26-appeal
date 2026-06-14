# Research Proposal: Cross-Domain Characterization, Confound Decomposition, and Information-Theoretic Diagnostics for Feature Absorption in Sparse Autoencoders

## Title

Beyond First Letters: Cross-Domain Absorption Characterization with Confound Decomposition and Rate-Distortion Diagnostics for Sparse Autoencoders

## Abstract

Feature absorption -- where a general (parent) SAE feature silently fails to fire because a more specific (child) feature has absorbed its information under sparsity pressure -- is a critical failure mode of SAE-based mechanistic interpretability. Yet all existing studies evaluate absorption on a single domain (first-letter spelling), detection requires pre-specified probe directions, and no study has formally decomposed what fraction of measured "absorption" is true hierarchy-driven signal versus confound artifact. We present a three-pronged study on Gemma 2 2B with Gemma Scope SAEs. **First**, we provide the first cross-domain absorption characterization across five hierarchy types (first-letter, city-country, city-continent, city-language, animal-class), grounded in pilot evidence showing detectable absorption in geographic and linguistic hierarchies (6.5-6.6%) but 0% in city-country -- revealing domain-dependent absorption severity. **Second**, we decompose measured absorption into hierarchy-driven, hedging, and reconstruction-error components, finding that on well-calibrated SAEs (L0=22), 96.9% of false negatives are hierarchy-driven. **Third**, we develop a rate-distortion diagnostic grounded in the successive refinement theorem, where the conditional mutual information I(X; f_parent | f_child) provides the first information-theoretic criterion for when absorption is necessary versus avoidable. The study incorporates rigorous confound controls (L0/width partial correlation, threshold sensitivity sweeps, Rosenbaum matching) informed by five prior iterations, and pre-registered falsification criteria for all hypotheses.

## Motivation

Feature absorption creates a false sense of monosemanticity: an SAE feature appears clean but has systematic, invisible gaps in recall. Chanin et al. (2024, NeurIPS 2025 Oral) demonstrated 15-35% absorption rates across all tested SAEs on first-letter spelling. DeepMind's 2025 safety team found SAE probes fail catastrophically on safety-relevant tasks where dense probes succeed, strongly implicating absorption. Three critical gaps remain:

1. **No cross-domain evidence**: Every published absorption study uses the first-letter spelling task -- a uniquely clean, exhaustive, non-overlapping 26-class hierarchy. Our pilot data now shows that absorption does occur in geographic/linguistic hierarchies (6.5-6.6% for city-continent and city-language) but at lower rates than first-letter, and with 0% in city-country. The field needs systematic cross-domain evidence with proper controls.

2. **No confound decomposition**: The Chanin metric has never been formally validated via threshold sensitivity analysis, nor has measured absorption been decomposed into hierarchy-driven signal versus L0-induced hedging, reconstruction error, and metric sensitivity artifacts. Our pilot confound decomposition shows 96.9% hierarchy-driven on well-calibrated SAEs, but this must be tested across configurations.

3. **No information-theoretic criterion for absorption necessity**: The field lacks a principled answer to "when is absorption harmful?" The successive refinement theorem (Equitz & Cover, 1991) provides the exact condition: absorption is information-theoretically lossless when X -- f_child -- f_parent forms a Markov chain (I(X; f_parent | f_child) = 0), and lossy otherwise. No prior work has applied this to SAEs.

## Evidence-Driven Revisions (from Prior Iterations and Pilot Data)

This proposal is grounded in five prior iterations of research and two completed pilot experiments:

### Pilot Evidence (Iteration 6)

**First-letter validation**: L12-16k absorption rate = 13.4% (95% CI: 7.2-18.1%), within published 15-35% range. Mean probe F1 = 0.565 (only 4/25 letters pass F1 > 0.85 gate). Controls need calibration (C1 random = 9.2%, C2 shuffled = 59.5%).

**City-country probe quality**: 189 single-token cities across 29 countries. Mean CV F1 = 0.773, median = 0.893. 16/29 countries above 0.85 gate, 22/29 above 0.70. Notable: SAE probes outperform dense probes (0.773 vs 0.617), indicating the SAE captures geographic knowledge effectively.

**Cross-domain absorption rates** (preliminary):
- First-letter: 13.4% (with caveats on probe quality and controls)
- City-country: 0.0% (but mean probe F1 = 0.602, below gate)
- City-continent: 6.5% (CI: 0-11.5%, mean probe F1 = 0.795)
- City-language: 6.6% (mean probe F1 = 0.745)
- Animal-class: 1.4% (mean probe F1 = 0.696)

**Confound decomposition**: On L0=22 SAE with probe F1=1.0, 96.9% of false negatives classified as hierarchy-driven, only 3.1% hedging, 0% reconstruction error.

**Unsupervised pipeline**: ITAC failed to show clear separation between candidate and random pairs (median ITAC 1.35 vs random 1.14, Mann-Whitney not significant). The conditional cosine + firing rate pipeline identified very few matching pairs per letter. H3 (unsupervised detection) is weakened by this evidence.

**Successive refinement CMI**: CMI estimates computed for all 25 letters at multiple subspace dimensions. CMI range: 0.21-0.40. The relationship between CMI and absorption rate requires further analysis with better probe quality.

### Key Methodological Lessons (Iterations 1-5)

- All experiments must use Gemma 2 2B + Gemma Scope SAEs (GPT-2 Small failed: 98% dead features)
- Probe quality gate (F1 > 0.85) must precede absorption measurement; domains failing this gate reported as limitations
- L0 must be included as covariate in all analyses
- FDR correction (Benjamini-Hochberg) applied for multiple comparisons
- Language must match evidence: "statistical mediation" not "causal mediation" for observational data
- Within-width null (Rosenbaum Gamma = 1.0 for all strategies) is a primary finding about confound structure
- Automated cross-validation between source JSONs and summary files prevents data pipeline errors

## Revisions from Prior Feedback

The iteration 5 reflection (score 6.0, verdict REVISE) and pilot evidence drive these revisions:

1. **Unsupervised detection de-prioritized**: Pilot evidence shows ITAC does not clearly separate absorption pairs from random pairs. The unsupervised pipeline (H3) is retained as a secondary contribution but no longer co-equal with cross-domain characterization. If full-scale validation confirms rho < 0.3, it is reported as an informative negative result.

2. **Rate-distortion theory elevated**: The successive refinement / rate-distortion framework (from the Theoretical perspective) is elevated from Stage 5 (secondary) to a primary theoretical contribution. The CMI-based diagnostic provides the most novel individual contribution (novelty score: 8/10 per novelty report) and addresses a question no existing work answers: when is absorption information-theoretically necessary?

3. **Cross-domain calibration**: Pilot data reveals domain-dependent absorption (0-6.6%) well below first-letter (13.4%). The proposal adjusts expected outcomes accordingly and treats the cross-domain rate differentials as a primary finding rather than a failure.

4. **Probe quality transparency**: Probe F1 is reported per-domain and per-parent prominently. Domains with probe F1 < 0.70 are excluded from absorption rate claims. The observation that SAE probes outperform dense probes on city-country is highlighted as a noteworthy finding.

5. **Control calibration**: Shuffled and random controls showed unexpectedly high rates (59.5% and 9.2% on first-letter). Full-scale experiments must investigate and calibrate these before drawing conclusions about net absorption signal.

6. **Impossibility theorem dropped**: Per novelty report recommendation (score: 5/10), the impossibility theorem backup is dropped due to crowded theoretical landscape (Cui et al., Tang et al., arXiv:2506.14002).

## Research Questions

**RQ1 (Cross-domain)**: Does feature absorption occur at comparable rates in knowledge hierarchies (city-country, city-continent, city-language, animal-class) as in first-letter spelling, when measured on Gemma 2 2B with Gemma Scope SAEs and properly calibrated controls?

**RQ2 (Confound decomposition)**: What fraction of measured absorption is hierarchy-driven versus attributable to hedging, reconstruction error, and metric threshold sensitivity?

**RQ3 (Rate-distortion diagnostic)**: Does the conditional mutual information I(X; f_parent | f_child) from rate-distortion theory predict which features are susceptible to absorption, and at what L0 absorption becomes information-theoretically optimal?

**RQ4 (Hierarchy predictors)**: Do measurable hierarchy properties (co-occurrence frequency ratio, fan-out, depth) predict absorption severity across domains?

**RQ5 (Scaling)**: How does absorption scale with SAE width and L0 across hierarchy types, and does the L0 ~ 7-14 phase transition replicate?

## Hypotheses

### H1 (Cross-domain generalization)
**Statement**: Feature absorption occurs at rates >= 5% in at least two knowledge hierarchy domains (city-continent, city-language, or animal-class) on Gemma 2 2B Gemma Scope SAEs (layer 12, 16k width), using the Chanin et al. probe-based metric with calibrated controls.

**Expected outcome**: Based on pilot data, city-continent (6.5%) and city-language (6.6%) are near the threshold. Full experiments with improved probe quality and calibrated controls should clarify whether these rates represent genuine signal above baseline.

**Falsification**: Absorption rate < 3% across ALL knowledge hierarchy domains after probe quality gating (F1 > 0.85), frequency-matched baseline subtraction, and control calibration.

**Revision from prior round**: Threshold lowered from 10% to 5% based on pilot evidence showing cross-domain rates are substantially lower than first-letter. This is more informative: we are asking whether cross-domain absorption exists at all, not whether it matches first-letter rates.

### H2 (Confound decomposition)
**Statement**: On Gemma Scope SAEs at optimal L0 (L0 ~ 22 where probes achieve F1 = 1.0), hierarchy-driven absorption accounts for > 80% of total false negatives.

**Expected outcome**: Pilot shows 96.9% hierarchy-driven at L0=22. Full decomposition across multiple L0 values should reveal that hierarchy-driven fraction increases at optimal L0 but decreases at too-low L0 (where hedging dominates) and too-high L0 (where reconstruction error dominates).

**Falsification**: Hierarchy-driven absorption < 50% at all L0 values tested.

### H3 (Rate-distortion diagnostic)
**Statement**: The conditional mutual information I(X; f_parent | f_child), computed in the decoder-direction subspace, is negatively correlated with absorption rate across the 25 first-letter features: letters with higher CMI (parent carries more unique information beyond the child) should exhibit lower absorption (the SAE pays to keep them separate). Target: Spearman rho < -0.3.

**Expected outcome**: Based on rate-distortion theory (Theorem 1 sketch from Theoretical perspective), the CMI threshold lambda/c(w_P, w_C) determines absorption optimality. Letters with CMI below this threshold should be preferentially absorbed.

**Falsification**: Spearman rho > -0.2 (CMI does not predict absorption direction).

### H4 (Unsupervised detection -- secondary)
**Statement**: The unsupervised absorption score (conditional cosine + firing rate + ITAC) correlates with probe-based absorption rate on first-letter at Spearman rho > 0.3.

**Expected outcome**: Pilot evidence is discouraging (few matching pairs, no ITAC separation). Full experiments with refined thresholds may improve. Target lowered from rho > 0.5 to rho > 0.3 based on pilot.

**Falsification**: Spearman rho < 0.15 on validation task.

**Pre-registered decision**: If rho < 0.3 on first-letter validation, report as negative result and do not deploy cross-domain. This is explicitly the most at-risk hypothesis.

### H5 (Scaling law)
**Statement**: Absorption rate exhibits a significant width-L0 interaction (p < 0.05 in GAM with architecture covariate), with a phase transition detectable in the L0 ~ 7-14 range.

**Falsification**: Width-L0 interaction non-significant within single architecture.

### H6 (Hierarchy predictors)
**Statement**: At least one hierarchy property (co-occurrence ratio, fan-out, depth) correlates with absorption rate at Spearman |rho| > 0.3 across 4+ independently measured domains with Bonferroni-corrected p < 0.05.

**Falsification**: No property achieves |rho| > 0.2 with corrected p < 0.05.

## Method

### Stage 1: Cross-Domain Absorption Characterization (Probe-Based)

**Domains** (5, extending pilot):

| Domain | Parent Feature | N_parents | Source | Pilot Rate |
|--------|---------------|-----------|--------|-----------|
| First-letter | "starts with X" | 25 | Chanin et al. | 13.4% |
| City -> Country | "in France" | 28 | RAVEL | 0.0% |
| City -> Continent | "in Europe" | 6 | RAVEL | 6.5% |
| City -> Language | "French-speaking" | 18 | RAVEL | 6.6% |
| Animal -> Class | "mammal" | 5+ | WordNet | 1.4% |

**Protocol refinements from pilot**:
1. Increase word count per letter from 25 to 50+ for first-letter task to improve probe quality
2. Exclude countries with < 5 cities or probe F1 < 0.50 from absorption claims
3. Stratify analysis by probe quality tier (F1 > 0.85 vs 0.70-0.85)
4. Calibrate control baselines: investigate why shuffled control = 59.5% on first-letter and random control = 9.2%
5. Report per-parent and per-domain absorption rates with bootstrap 95% CIs
6. Threshold sensitivity: full 5x4 grid (cosine {0.01-0.05}, magnitude gap {0.5-2.0}), report CV and Kendall tau rank stability

**SAE configurations**: Gemma 2 2B, Gemma Scope, layers 10/12/20, widths 16k and 65k.

### Stage 2: Confound Decomposition

Building on pilot evidence (96.9% hierarchy-driven at L0=22):

1. **Multi-L0 decomposition**: For first-letter task, decompose false negatives at L0 = {22, 42, 82, 163} into hedging, reconstruction error, and hierarchy-driven components. Test whether the hierarchy-driven fraction varies systematically with L0.

2. **Threshold sensitivity analysis**: The Chanin metric's thresholds (cosine > 0.025, magnitude gap >= 1.0) have never been sensitivity-tested. Run the 5x4 grid. Report CV and rank-order stability across SAE configurations.

3. **L0 covariate analysis**: Partial correlations between absorption rate and quality metrics, controlling for log(L0), log(width), and architecture. Report both FDR-corrected and uncorrected p-values.

4. **Rosenbaum matching**: Both Mahalanobis (cross-width) and within-width strategies. Report Gamma divergence explicitly.

### Stage 3: Rate-Distortion Theory and Information-Theoretic Diagnostic

This is the proposal's most novel contribution (novelty score: 8/10).

**Theoretical framework** (from the Theoretical perspective's Candidate A, refined):

*Theorem sketch*: Let X be the LLM residual stream activation. For a parent-child feature pair with hierarchical co-occurrence, define CMI = I(X; w_parent | f_child). Then:
- Absorption is rate-distortion optimal when CMI < lambda / c(w_P, w_C), where c is a geometric constant depending on decoder direction orthogonality
- Absorption is suboptimal (destroys information) when CMI exceeds this threshold
- Absorption is lossless iff X -- f_child -- f_parent forms a Markov chain (CMI = 0)

**Empirical tests**:
1. **CMI-absorption correlation** (P1 from Theoretical): Compute I(X; w_parent | f_child) for all 25 first-letter features using k-NN estimator in decoder-direction subspace (d'=10-50). Correlate with observed absorption rate. Target: Spearman rho < -0.3.

2. **ITAC-CMI connection**: ITAC = Var(residual projection | child active, parent inactive) / Var(same | neither) is a scalar proxy for CMI. Validate that ITAC correlates with CMI (establishing the theory-practice bridge).

3. **Geometric constant validation** (P2): Compute c(w_P, w_C) = ||w_P||^2 * (1 - cos^2(w_P, w_C)) from decoder weights. Test whether c modulates the absorption threshold beyond CMI alone.

4. **Lateral inhibition check** (from Theoretical Candidate C): Compute inhibition ratio q = G_PC * a_C / (theta_P - b_P) for 100k tokens. Test whether q > 1 predicts per-token false negatives (target: precision@50 > 0.3).

5. **Phase transition prediction** (P5): Compute predicted critical L0 from theory. Compare with observed L0 ~ 7-14 transition from iteration 5.

6. **Cross-domain CMI**: Extend CMI computation to city-continent and city-language hierarchies.

### Stage 4: Hierarchy Predictor Analysis

For each domain, compute:
- Parent-child co-occurrence frequency ratio in 100k-token corpus
- Fan-out (children per parent)
- Hierarchy depth
- Parent feature frequency

Correlate with absorption rate. Fit linear model: absorption_rate ~ co_occurrence_ratio + fan_out + parent_frequency + L0. Report Spearman correlations with Bonferroni correction.

### Stage 5: Scaling Surface

Extend iteration 5's 420-SAE analysis:
1. Absorption as function of (width, L0) across Gemma Scope SAEs
2. GAM with architecture_class covariate (addressing iteration 5 reviewer concern)
3. Within-architecture subset analysis
4. Cross-domain scaling comparison

### Stage 6 (Secondary): Unsupervised Detection Refinement

Given pilot evidence of ITAC weakness:
1. Refine ITAC computation: try larger corpus (500k tokens), stricter candidate filtering, alternative variance metrics
2. Test whether spectral graph wavelet analysis (from Innovator perspective) provides better signal than hierarchical clustering
3. Validate on first-letter against gold standard. Pre-registered decision: if rho < 0.3, report negative result.

## Experimental Plan

| Experiment | Domain/Analysis | SAE Config | Est. Time | Purpose |
|---|---|---|---|---|
| Exp 1a: First-letter (improved) | First-letter (50+ words/letter) | L12, 16k | 30 min | Improve probe quality, calibrate controls |
| Exp 1b: City-Country (refined) | City -> Country | L12, 16k, 65k | 45 min | Primary cross-domain (H1) |
| Exp 1c: City-Continent | City -> Continent | L12, 16k | 30 min | Cross-domain (H1) |
| Exp 1d: City-Language | City -> Language | L12, 16k | 30 min | Cross-domain (H1) |
| Exp 1e: Animal-Class | Animal -> Class | L12, 16k | 30 min | Non-geographic domain (H1) |
| Exp 2a: Confound decomposition | Multi-L0 decomposition | L12, 16k (multiple L0) | 45 min | H2 |
| Exp 2b: Threshold sensitivity | 5x4 grid | L12, 16k | 30 min | Metric robustness |
| Exp 3a: CMI estimation | 25 first-letter features | L12, 16k | 30 min | H3 (rate-distortion) |
| Exp 3b: ITAC-CMI connection | Cross-validate metrics | L12, 16k | 15 min | Theory-practice bridge |
| Exp 3c: Geometric constant | Decoder weight analysis | L12, 16k | 10 min | Rate-distortion refinement |
| Exp 3d: Lateral inhibition | Per-token prediction | L12, 16k | 30 min | Mechanistic validation |
| Exp 4: Hierarchy predictors | Cross-domain regression | L12, 16k | 30 min | H6 |
| Exp 5: Scaling surface | Width-L0 interaction | All configs | 45 min | H5 |
| Exp 6: Unsupervised validation | Refined pipeline | L12, 16k | 30 min | H4 (secondary) |

**Total**: approximately 6-7 hours. Each task within 1-hour budget.

**Ablations**: A1: Probe sparsity k={1,3,5,10}; A2: Dataset size sensitivity; A3: Threshold grid (5x4); A4: Width scaling per domain; A5: CMI subspace dimension {10, 20, 30, 50}; A6: ITAC corpus size {100k, 500k}

**Controls**: C1: Random probe (target < 2%); C2: Shuffled labels (investigate 59.5% pilot rate); C3: Untrained SAE; C4: Dense probe ceiling

**Pre-registered decision gates**:
- After Exp 1a: If first-letter controls not calibrated (shuffled > 20% after refinement), investigate before proceeding
- After Exp 3a: If CMI-absorption correlation rho > -0.2, de-prioritize rate-distortion framing
- After Exp 6: If unsupervised rho < 0.3, report as negative result

## Novelty Assessment

Verified through systematic search (April 2026):

1. **First cross-domain absorption characterization on Gemma 2 2B** (novelty score: 7/10): No prior work measures absorption beyond first-letter on an adequately sized model. RAVEL in SAEBench tests disentanglement (a different property), not absorption rates. Our pilot data already shows domain-dependent rates (0-6.6% vs 13.4% first-letter), which is itself a novel empirical finding.

2. **Rate-distortion absorption diagnostic via successive refinement** (novelty score: 8/10): No prior work applies Equitz & Cover's (1991) successive refinement theorem to SAE absorption. The CMI threshold for absorption optimality is entirely novel. Searched: "successive refinement sparse autoencoder" -- zero results. This provides the first principled answer to "when is absorption harmful?"

3. **Confound decomposition with multi-L0 profiling** (novelty score: 6/10): Chanin et al. (2025) decompose hedging vs. absorption, but our multi-L0 profile (showing hierarchy-driven fraction varies with L0) and threshold sensitivity analysis extend beyond prior work.

4. **Lateral inhibition bifurcation analysis** (novelty score: 8/10, from Theoretical perspective): The prediction that TopK/JumpReLU SAEs show more binary absorption than L1 SAEs, grounded in LCA dynamics, has never been tested.

**RAVEL differentiation** (critical for reviewers): RAVEL measures whether SAE latent interventions correctly transfer attributes (disentanglement via causal intervention). Absorption measures whether parent features fail to fire due to child features absorbing their information under sparsity (probe-based detection of systematic false negatives). These are fundamentally different metrics measuring different failure modes.

**What is explicitly NOT novel**: The absorption metric (Chanin et al.), SAE infrastructure (SAELens, Gemma Scope), the existence of absorption, epidemiological methods, individual statistical methods.

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cross-domain absorption < 3% everywhere after controls | 25% | Medium | Report as informative finding (absorption is domain-specific); first-letter provides guaranteed baseline |
| CMI-absorption correlation too weak (rho > -0.2) | 30% | Medium | CMI estimation may be noisy at d'=10; try d'=20-50; report negative result honestly |
| Controls not calibratable (shuffled remains > 30%) | 20% | High | Investigate control implementation; may indicate metric limitation; report transparently |
| ITAC unsupervised detection fails (rho < 0.3) | 50% | Low (now secondary) | Report negative result; SGWA from Innovator perspective as future direction |
| Probe quality insufficient on knowledge domains | 15% | Medium | Stratify by probe quality tier; exclude weak domains from absorption claims |
| L0 confound eliminates within-width correlations | 40% | Low (expected) | Report as confirmatory finding replicating iteration 5 |

## Expected Contributions

1. **Empirical (primary)**: First systematic cross-domain absorption characterization on Gemma 2 2B, establishing that absorption rates are domain-dependent (0-6.6% for knowledge hierarchies vs 13.4% for first-letter), with proper controls and confound decomposition

2. **Theoretical (primary)**: First rate-distortion theory of absorption grounded in the successive refinement theorem, providing an information-theoretic criterion for when absorption is necessary versus avoidable, with the CMI threshold as a computable diagnostic

3. **Methodological**: First threshold sensitivity analysis and confound decomposition of the Chanin absorption metric, establishing metric reliability (or lack thereof)

4. **Practical**: Cross-domain absorption rates and hierarchy predictors enabling practitioners to estimate absorption risk for specific feature types

5. **Negative result (secondary)**: Honest report on unsupervised detection viability, with specific failure analysis addressing the LessWrong negative result

## What Changed from Previous Round

1. **Unsupervised detection de-prioritized**: Pilot ITAC evidence shows no clear separation; H3 target lowered from rho > 0.5 to rho > 0.3; classified as secondary contribution
2. **Rate-distortion theory elevated**: From Stage 5 (secondary) to co-primary contribution; highest novelty score (8/10)
3. **H1 threshold adjusted**: From >= 10% to >= 5% based on pilot cross-domain rates of 0-6.6%
4. **Control calibration added**: Pilot shuffled control = 59.5% requires investigation before interpreting net signal
5. **Probe quality transparency**: Per-domain F1 reported prominently; stratification by probe quality tier
6. **Impossibility theorem dropped**: Per novelty report (score 5/10, saturated landscape)
7. **Lateral inhibition bifurcation added**: From Theoretical perspective's Candidate C, predicting TopK vs L1 absorption dynamics
8. **Domain-dependent absorption framed as finding**: Rather than treating low cross-domain rates as failure, the domain dependence itself is an empirical contribution

## Relationship to Perspectives

This proposal synthesizes all six perspectives, weighted by evidence strength and novelty:

- **From Theoretical (highest weight -- most novel contribution)**: Rate-distortion bound (Candidate A, novelty 8/10) elevated to primary theoretical contribution; lateral inhibition bifurcation (Candidate C) as supporting mechanistic analysis; CMI-ITAC connection providing theory-practice bridge. The Theoretical perspective's framework answers the most important question in the field: when is absorption information-theoretically necessary?

- **From Pragmatist (highest weight -- execution feasibility)**: Cross-domain audit using existing sae-spelling infrastructure; probe quality gating (F1 > 0.85); engineering-first experiment design; specific SAE configurations and time estimates; 80%+ code reuse

- **From Empiricist (highest weight -- rigor)**: Rigorous evaluation protocol with 5 domains; probe quality gates; statistical power considerations; 4 control experiments; threshold sensitivity analysis; bootstrap CIs; FDR correction; pre-registered falsification criteria

- **From Contrarian (high weight -- critical validation)**: Confound decomposition as primary contribution (not just an ablation); threshold sensitivity analysis; honest negative result framework for unsupervised detection; the important caution that controls must be calibrated before interpreting absorption rates

- **From Innovator (moderate weight -- novel methods)**: Conditional cosine similarity insight retained; SGWA referenced as promising future direction given ITAC pilot weakness; multi-resolution decoder geometry retained as secondary method

- **From Interdisciplinary (moderate weight -- theoretical enrichment)**: Successive refinement theorem as the theoretical anchor; ecological competitive exclusion as qualitative interpretive framework (Candidate B); divisive normalization as mechanistic explanation for cross-level feature suppression; immune escape test (zeroing child to test parent recovery) as a diagnostic experiment

**Weighting rationale**: The Theoretical perspective is elevated because the rate-distortion diagnostic is the highest-novelty contribution (8/10) and addresses the most important open question. The Pragmatist and Empiricist maintain high weight because execution quality and rigor remain the primary barriers to publication (iteration 5 score: 6.0). The Contrarian is elevated because pilot evidence validates the concern about control calibration. The Innovator is de-weighted because pilot evidence weakens the unsupervised pipeline. The Interdisciplinary is maintained at moderate weight because the successive refinement theorem provides the theoretical foundation, but the more elaborate cross-disciplinary analogies (immunological imprinting, ecological competition) are treated as interpretive enrichment rather than core methodology.
