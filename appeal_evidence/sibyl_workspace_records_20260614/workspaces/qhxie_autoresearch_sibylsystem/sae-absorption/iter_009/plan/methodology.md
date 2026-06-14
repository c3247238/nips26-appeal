# Methodology: The Absorption Tax -- Cross-Domain SAE Feature Absorption Characterization

## Overview

This experiment plan designs the full-mode experiments for iteration 9, building on 8 prior iterations of pilot evidence and the iter_008 consolidation summary. The paper has two primary contributions:

1. **First cross-domain absorption characterization** on Gemma 2 2B using RAVEL entity-attribute hierarchies
2. **Causal mechanism confirmation** via activation patching + benign/pathological diagnostic

Plus tertiary contributions (rate-distortion predictors) and honestly reported negative results (GAS, CMI, Absorption Tax quantitative).

## Key Findings from Prior Iterations (iter_008 Consolidation)

- **Activation patching CONFIRMED**: 32.5% recovery vs 1.5% control, Wilcoxon p=0.000218, Cohen's d=1.33 (n=25 words)
- **Tightened hedging CONFIRMED**: strict 7.9% vs loose 94.1%, 86.2% compensatory
- **CMI NOT SUPPORTED**: rho=0.044, p=0.83
- **GAS DEFINITIVE NEGATIVE**: rho=0.116, AUROC=0.571 at 640k tokens
- **Cross-domain variation CONFIRMED**: ANOVA p=0.005; 4/6 pairwise comparisons significant
- **H2' REFUTED at L24**: first-letter shows HIGHEST absorption (34.5%), not semantic hierarchies
- **Layer dependence NOVEL**: absorption concentrates at L24 (34.5%) vs L6/L12/L18 (2-9%)
- **Probe quality**: first-letter F1=0.97 at L24; RAVEL best F1=0.84 (city-continent at L24)
- **Architecture**: no significant architecture effect (ANOVA p=0.87); hierarchy >> architecture
- **Threshold sensitivity**: structural, not threshold-dependent (CV=0.077, FN constant)

## What Remains: Critical Gaps for Full-Mode Paper

### Gap 1: Benign vs. Pathological Absorption (H8 -- UNTESTED)
The Contrarian's key insight: not all absorption is harmful. If the model's computation doesn't independently use the parent feature when the child is active, absorption is faithful representation. This requires:
- Ablating parent direction from child latent decoder for each confirmed absorption instance
- Measuring downstream logit change
- Classifying benign (<=0.1 logit change) vs pathological (>0.1)
- Per-hierarchy ratios

### Gap 2: Rate-Distortion Three-Factor Predictors (H9 -- UNTESTED)
Testing whether absorption is predictable from:
- cos_sim(decoder_parent, decoder_child)
- P(child active | parent active)
- R(parent) = reconstruction importance
Target: Spearman rho > 0.5 across pooled parent-child pairs.

### Gap 3: Cross-Hierarchy Activation Patching
Iter_008 activation patching was first-letter only (n=25 words). Must extend to RAVEL hierarchies (city-continent, city-language) to confirm causal mechanism is cross-domain.

### Gap 4: Multi-Layer Architecture Comparison
Iter_008 architecture comparison was layer 12 only. Layer 24 data exists for JumpReLU but not for BatchTopK/Matryoshka. Need L24 comparison if SAEBench SAEs are available.

### Gap 5: Absorption-Hedging Decomposition Across Hierarchies
Iter_008 hedging decomposition was primarily first-letter. Need cross-domain hedging to test H3 (ratio varies by hierarchy type).

## Experimental Setup

### Model and SAEs
- **Model**: Gemma 2 2B (google/gemma-2-2b) via TransformerLens
- **Primary SAEs**: Gemma Scope JumpReLU 16k and 65k at layers 6, 12, 18, 24
- **Architecture comparison SAEs**: SAEBench BatchTopK 16k, Matryoshka 32k at layer 12 (and layer 24 if available)
- **All inference-only**: no SAE training required

### Datasets and Hierarchies
- **First-letter spelling**: sae_spelling pipeline (26 letters, ~222 test words). Positive control.
- **City-continent**: RAVEL (hij/ravel), 6 continent classes, ~173 entities
- **City-country**: RAVEL, ~80 country classes, highly imbalanced (France >> Liechtenstein)
- **City-language**: RAVEL, ~20 language classes, intermediate balance

### Probe Training Protocol
- Logistic regression (sklearn) at layers 6, 12, 18, 24
- Frequency-balanced sampling for imbalanced hierarchies (city-country)
- 80/20 train/test split, seed=42
- Quality gate: F1 > 0.90 strict, >= 0.80 relaxed with documented caveat
- Best confirmed layer: L24 for all RAVEL hierarchies (from iter_008)
- First-letter uses sae_spelling binary probe (F1=1.0 at all layers) + sklearn backup

### Absorption Measurement Protocol
- Adapted Chanin et al. pipeline from sae_spelling
- Integrated-gradients attribution for parent-child pair identification
- False negative classification: absorbed vs hedged vs residual
- Bootstrap 95% CI (10k resamples) on all rates
- Bonferroni correction for multiple hierarchy comparisons

### Activation Patching Protocol (from pyvene / TransformerLens)
- For each absorption pair: cache SAE activations, zero child feature, pass modified activations through remaining layers, check parent probe recovery
- Control: zero random non-child feature (matched activation magnitude)
- Statistics: Wilcoxon signed-rank test, bootstrap CI on recovery difference
- Extend to cross-domain pairs (city-continent, city-language)

### Benign vs. Pathological Diagnostic (NEW)
- For each confirmed absorption instance from Phase 1:
  1. Ablate parent direction component from child latent's decoder vector
  2. Run model with modified SAE reconstruction
  3. Measure logit change for parent-relevant tokens
- Benign: logit change <= 0.1 (parent computationally redundant in context)
- Pathological: logit change > 0.1 (parent has independent causal effects)
- Report distribution + multiple thresholds (0.05, 0.1, 0.2) for robustness

### Rate-Distortion Predictor Computation (NEW)
- From Phase 1, collect all identified parent-child pairs across hierarchies
- Compute per-pair: cos_sim(d_parent, d_child), P(child|parent), R(parent)
- Fit: absorption_probability ~ beta_1*cos_sim^2 + beta_2*co_occur - beta_3*R(parent)
- Evaluate: Spearman rho between predicted and observed

## Baselines and Controls
- **Random direction baseline**: replace parent probe with random direction, measure "absorption"
- **Shuffled hierarchy control**: randomly reassign parent labels, re-measure absorption
- **Probe-only baseline**: absorption rate attributable to probe imperfection alone
- **Random latent zeroing control**: for activation patching, zero random non-child feature

## Statistical Analysis
- Bootstrap 95% CI (10,000 resamples) for all absorption rates
- Paired permutation test for cross-domain comparisons
- Bonferroni correction for 6 pairwise comparisons (4 hierarchies choose 2)
- Wilcoxon signed-rank test for activation patching recovery rates
- ANOVA for architecture comparison (architecture x hierarchy)
- Spearman rank correlation for rate-distortion predictors

## Expected Visualizations

### Main Text
- **Table 1**: Cross-domain absorption rates (hierarchy x SAE config) at L24 with 95% CI
- **Table 2**: Activation patching results per word/entity: recovery rate (child vs control)
- **Figure 1**: Layer-dependent absorption profile (L6, L12, L18, L24 x hierarchy type)
- **Figure 2**: Absorption-hedging decomposition stacked bar chart per hierarchy
- **Figure 3**: Benign vs pathological absorption ratio per hierarchy (pie/stacked bar)
- **Figure 4**: Violin plot of recovery rate distribution (child-zeroed vs control-zeroed)
- **Table 3**: Architecture comparison (JumpReLU vs BatchTopK vs Matryoshka x hierarchy)

### Appendix
- **Table A1**: Probe quality (F1) for all hierarchy x layer combinations
- **Table A2**: Threshold sensitivity heatmap (5x4 grid)
- **Figure A1**: GAS vs absorption scatter (negative result)
- **Table A3**: CMI correlation at L0=22 (negative result)
- **Table A4**: Absorption Tax T(G) values per hierarchy (not supported)
- **Figure A2**: Rate-distortion three-factor model scatter plot

## Resource Estimate

- **GPU**: Single GPU >= 24GB VRAM (RTX PRO 6000 Blackwell 95GB available)
- **Total compute**: ~8.5 GPU-hours, ~10 hours wall-clock
- **Storage**: ~10GB for cached activations
- **Software**: SAELens v6.39+, TransformerLens, sae_spelling, RAVEL (hij/ravel), scipy, sklearn

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| RAVEL probes below 0.90 | Relaxed gate 0.80 with caveat; L24 confirmed best (F1=0.84) |
| Activation patching effect weaker at scale | Iter_008 showed d=1.33 at n=25; robust |
| Benign/pathological diagnostic noisy | Multiple thresholds; report distribution |
| Rate-distortion rho < 0.3 | Report as negative result; cross-domain characterization stands independently |
| SAEBench SAEs not available at L24 | Restrict arch comparison to L12; document as limitation |
| Cross-domain activation patching underpowered | Start with pairs from Phase 1; report with exact n and effect size |
