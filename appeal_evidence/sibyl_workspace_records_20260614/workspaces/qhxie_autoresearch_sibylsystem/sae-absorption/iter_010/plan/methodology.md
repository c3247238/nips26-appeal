# Methodology: Iteration 10 -- Probe Degradation Ablation, Paper Integrity Overhaul, and Final Experiments

## Overview

Iteration 10 addresses the score regression (7.0 -> 6.5) from the iter-9 review by executing three categories of work:

1. **Phase 0 (BLOCKING, zero GPU):** Critical paper integrity fixes -- propagate corrected FULL-mode activation patching data, reframe overclaims, build data validation infrastructure, generate missing figures
2. **Phase 1 (HIGHEST PRIORITY, GPU):** Probe degradation ablation (H10) -- the single most important remaining experiment, resolving whether cross-domain variation is a genuine hierarchy effect or probe artifact
3. **Phase 2 (GPU):** Complete remaining experiments -- decoder direction magnitude on first-letter (H8 cross-domain consistency), rate-distortion predictor validation (H9)

The project is mature (9 prior iterations, 23/23 tasks completed in iter_006, 10/10 in iter_009). This iteration focuses on paper-quality improvements and the one critical missing experiment (H10), not on new discovery.

## Key Findings from Prior Iterations (Authoritative Evidence Base)

### Corrected FULL-mode Data (iter_009)
- **Activation patching (city-continent):** d=1.50, p<1e-20, recovery 62% (PRIMARY) / 80% (ALL absorbers) vs 5.2% control. CORRECTED from buggy pilot d=-0.91.
- **Activation patching (city-language):** d=0.75, p<1e-18. CORRECTED from buggy pilot.
- **Activation patching (first-letter):** d=1.33, p=0.000218, recovery 32.5% vs 1.5% control. Consistent across iterations.
- **Universal competitive exclusion:** Confirmed across ALL hierarchy types. This is the paper's strongest causal finding.

### Established Results
- **Cross-domain absorption (L24 16k):** first-letter 27.1%, city-country 45.1%, city-continent 42.9%, city-language 11.6%. Within-RAVEL Kruskal-Wallis p=7.4e-66.
- **Hedging tightened:** strict 0-22.6%, compensatory 85.3%, loose 92.6% (near-tautological). Reusable methodological contribution.
- **H8 (decoder direction magnitude):** 3.98 nats vs 0.12 control on city-continent L24. Circularity identified.
- **H9 (rate-distortion):** model rho=0.261, p=0.266. NOT SUPPORTED. No individual predictor significant.
- **Architecture:** No significant effect (p=0.53-0.87). Hierarchy >> architecture.

### Critical Issues from iter-9 Review (Score Regression Causes)
1. Stale buggy pilot data in Section 5.2 (d=-0.91 instead of d=1.50)
2. Benign/pathological circularity not acknowledged
3. Headline overclaiming ("100% pathological", "4x range", "no architecture effect")
4. Missing figures 4-6
5. Data provenance errors (F1 discrepancies, benign count discrepancies)
6. Probe degradation ablation never executed

## Experimental Setup

### Model and SAEs
- **Model:** Gemma 2 2B (google/gemma-2-2b) via TransformerLens
- **Primary SAEs:** Gemma Scope JumpReLU 16k at layer 24 (the recommended layer with best probes and highest absorption)
- **All inference-only:** No SAE training required

### Datasets and Hierarchies
- **First-letter spelling:** sae_spelling pipeline (26 letters, ~222 test words). Positive control with F1=1.0 probes.
- **City-continent:** RAVEL (hij/ravel), 6 continent classes, ~173 entities. F1=0.84 at L24.
- **City-country:** RAVEL, ~80 country classes. F1=0.79 at L24.
- **City-language:** RAVEL, ~20 language classes. F1=0.82 at L24.

### Software Dependencies
- SAELens v6.39+ (pre-trained SAE loading, activation encoding)
- TransformerLens (model loading, hook-based intervention)
- sae_spelling (first-letter absorption pipeline)
- RAVEL dataset (hij/ravel on HuggingFace)
- scipy, sklearn, numpy, matplotlib

## Phase 0: Critical Paper Fixes (BLOCKING, Zero GPU)

### Step 0.1: Data Integrity Overhaul (~1.5 hours CPU)
- Create `exp/results/data_manifest.json` mapping every numerical claim in the paper to its source data file and JSON field path
- Resolve known discrepancies: F1=0.97 vs 1.0 (first-letter probes), benign count 1471 vs 1464, aggregation method documentation
- Implement `exp/code/validate_integration.py` -- automated cross-check script that:
  - Reads the paper LaTeX and extracts all numerical claims
  - Matches each claim to the source JSON in data_manifest
  - Reports mismatches with file:line references
  - Exit 0 if all match, exit 1 with report if any mismatch

### Step 0.2: Propagate Corrected FULL-mode Data (~1 hour CPU)
- Update Section 5.2 with corrected cross-domain patching: city-continent d=1.50 (not d=-0.91), city-language d=0.75
- Update abstract: "mechanism confirmed cross-domain" with corrected effect sizes
- Remove "concentrated vs. distributed absorption" dichotomy (based on buggy pilot data)
- Verification: grep for "d=-0.91", "0.05% recovery" should return zero hits

### Step 0.3: Reframe Benign/Pathological (~30 min CPU)
- Acknowledge circularity in Section 5.3: the diagnostic measures decoder geometry, not computational redundancy
- Reframe: "child decoders carry large-magnitude parent-direction information (3.98 nats vs 0.12 control)"
- Describe what a genuine computational-redundancy test would require (activation-level ablation z_parent=0, path patching)
- Scope to city-continent L24 only

### Step 0.4: Reframe Headlines (~30 min CPU)
- "4x range" qualified as descriptive, not all pairwise-significant
- "100% pathological" scoped to "across 1,471 instances from city-continent at L24"
- "No architecture effect" reframed as "effect not detected with limited power (12 observations)"
- Separate descriptive from inferential claims throughout

### Step 0.5: Generate Missing Figures (~1 hour CPU)
- fig4_patching_comparison.pdf: paired dot plot of recovery rates by hierarchy (first-letter, city-continent, city-language)
- fig5_pathological_histogram.pdf: histogram of |logit change| with random-direction control overlay
- fig6_architecture_comparison.pdf: grouped bar chart of absorption rates by architecture x hierarchy

## Phase 1: Probe Degradation Ablation (HIGHEST-PRIORITY NEW EXPERIMENT)

### Rationale
The core uncertainty: does cross-domain variation in absorption rates reflect genuine hierarchy differences, or merely probe quality differences? First-letter probes achieve F1=1.0 while RAVEL probes are F1=0.79-0.84. The correlation between probe F1 and measured absorption (rho=-0.756) is suspicious.

### Method
1. **Load trained first-letter probes at L24** (F1=1.0). These exist in `iter_009/exp/results/phase1/sae_spelling_probes/layer_24/`.
2. **Inject label noise to degrade probe quality** to F1={0.70, 0.80, 0.85, 0.90}:
   - For each target F1 level: randomly flip labels in the test set until probe F1 drops to the target
   - Use seed=42 for reproducibility
   - Verify degradation: re-evaluate probe F1 after noise injection
3. **Re-run the full absorption measurement pipeline** at L24 with 16k SAE at each degraded level:
   - Same pipeline as iter_009 phase1_absorption_firstletter
   - Use the sae_spelling framework with modified probes
   - Measure: absorption_rate, n_fn, n_absorbed, bootstrap 95% CI
4. **Also run at the ORIGINAL F1=1.0** as internal control (should match iter_009 results)
5. **Plot absorption rate vs probe F1** with RAVEL absorption rates overlaid at their respective F1 levels
6. **Key comparison:** Does first-letter absorption at F1=0.80 approach city-country (45.1%) or city-continent (42.9%) absorption?

### Expected Outcomes
- **If genuine hierarchy effect:** First-letter absorption increases modestly (27% -> maybe 30-35% at F1=0.80) but remains well below city-country (45.1%). Gap persists. Cross-domain finding is strengthened.
- **If probe artifact:** First-letter absorption increases dramatically (27% -> ~40-45% at F1=0.80), closely matching RAVEL rates. Cross-domain variation collapses. This is an important methodological caution.
- **Both outcomes are publishable.** This experiment resolves the single biggest remaining ambiguity.

### Falsification Criterion
Degraded first-letter rates at F1=0.80 fall within 5 pp of city-language/city-country rates at equivalent probe quality.

### Controls
- Internal control: F1=1.0 should reproduce iter_009 results within CI
- Noise injection control: verify that label noise actually reduces test-set F1 to target (not training-set F1)

## Phase 2: Complete Missing Experiments

### Step 2.1: Decoder Direction Magnitude on First-Letter (H8 Cross-Domain Consistency) (~0.5 GPU-hours)
The iter-9 result (3.98 nats, 0% benign) was on city-continent only. To test cross-domain consistency:
1. Replicate the decoder direction ablation on first-letter at L24 16k
2. For each first-letter absorption instance: ablate parent direction from child decoder, measure logit change
3. Compare |logit change| distribution with city-continent result
4. Control: random direction of same norm
5. Report: is the magnitude consistent? Does hierarchy type affect pathological severity?

### Step 2.2: Rate-Distortion Predictor Validation with Corrected Data (~0.5 GPU-hours)
Iter-9 tested with 20 pairs and found rho=0.261 (p=0.266). With corrected activation patching data and more pairs:
1. Pool all parent-child pairs across all hierarchies from iter-9 results
2. Compute per-pair: cos_sim(d_parent, d_child), P(child|parent), R(parent)
3. Fit three-factor model: absorption_prob ~ beta_1*cos_sim^2 + beta_2*co_occur - beta_3*R(parent)
4. Evaluate with Spearman rho
5. Also test cross-domain: do predictor distributions explain hierarchy-level differences?
6. Target: rho > 0.5. Falsification: rho < 0.3 or p > 0.05.

## Baselines and Controls
- **Random direction baseline:** replace parent probe with random direction, measure "absorption"
- **Shuffled hierarchy control:** randomly reassign parent labels, re-measure absorption
- **Noise injection control (Phase 1):** verify degraded probe F1 matches target
- **Internal replication (Phase 1):** F1=1.0 condition should match iter_009 baseline
- **Random decoder direction control (Phase 2.1):** same norm, random direction

## Statistical Analysis
- Bootstrap 95% CI (10,000 resamples) for all absorption rates
- Regression analysis: absorption rate vs probe F1 (Phase 1)
- Comparison: degraded first-letter rates vs RAVEL rates at matched F1
- Wilcoxon signed-rank for decoder direction magnitude (Phase 2.1)
- Spearman rank correlation for rate-distortion predictors (Phase 2.2)

## Expected Visualizations

### Main Text (from Phase 1)
- **Figure 7 (NEW):** Absorption rate vs probe F1 curve for first-letter, with RAVEL points overlaid at their F1 levels
- **Table 5 (NEW):** Probe degradation ablation results (F1, absorption_rate, CI, delta_from_RAVEL)

### Main Text (from Phase 0 -- generated from existing data)
- **Figure 4:** Cross-domain activation patching recovery (paired dot plot)
- **Figure 5:** Decoder direction magnitude histogram with control overlay
- **Figure 6:** Architecture comparison grouped bar chart

### Appendix (from Phase 2)
- **Table A5:** Decoder direction magnitude on first-letter vs city-continent
- **Table A6:** Rate-distortion predictor correlations (pooled cross-domain)

## Resource Estimate

- **GPU:** Single GPU >= 24GB VRAM
- **GPU compute:** ~3.5 GPU-hours total
  - Phase 1 (probe degradation): ~2 GPU-hours (4 degradation levels x ~30 min each)
  - Phase 2.1 (decoder magnitude): ~0.5 GPU-hours
  - Phase 2.2 (rate-distortion): ~0.5 GPU-hours
  - Phase 0.5 (figure generation): ~0.5 GPU-hours (may need model loading for data extraction)
- **CPU compute:** ~6 hours (Phase 0 paper fixes + writing)
- **Wall-clock total:** ~10 hours with 1 GPU
- **Storage:** Reuses existing ~10GB cached activations

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Probe degradation shows variation IS probe artifact | High | 25% | BOTH outcomes publishable. If probe-driven: major methodological caution for the field |
| Noise injection doesn't degrade probe cleanly | Low | 10% | Use multiple noise injection strategies (uniform, per-class weighted) |
| Rate-distortion still fails with more data | Medium | 60% | Report alongside GAS/CMI as 3rd negative result |
| Decoder magnitude different on first-letter | Low | 20% | Both outcomes informative about hierarchy-dependent mechanisms |
| Data integrity check reveals more errors | Medium | 30% | validate_integration.py catches them automatically |
